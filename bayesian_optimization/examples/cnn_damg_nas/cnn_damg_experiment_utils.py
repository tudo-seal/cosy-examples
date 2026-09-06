"""What the two cnn_damg_nas experiments share: the search, the persistence, the timing.

The USPS and the CIFAR-10 script differ in their dataset, their search-space constants and their
command line.  Everything between them, the acquisition optimizer, the ask/tell loop, the logging
and the provenance record, was the same code in both files, character for character, and that is
what lives here now.

The reason is not tidiness.  While it stood twice, every correction had to be made twice, and
statements had drifted from what the code did: both metadata blocks named a survivor selection the
run did not use, both called the mutation rate mandatory for a reason that had been repaired two
packages earlier, and both quoted a kernel prior variance that no longer existed.  A description
written by hand next to the thing it describes will do that.  So the description here is read off
the constructed objects, see :func:`describe_search`, rather than written beside them.

Seven artifacts per run, and they answer different questions:

* ``<run>.csv``, one row per objective-function evaluation: the phase, the index within that
  phase, the pretty-printed structure, the measured metrics and the two wall clocks, training and
  acquisition.  Flushed after each row, so an interrupted run keeps its results.  Whether the
  search finds better networks is read here.
* ``<run>_terms.pickle``, the same evaluations with the term rather than its rendering, one record
  each, written by the same call so that it cannot be omitted.  Whether this run can be analyzed
  afterwards is decided here and only here: a rendered structure cannot be fed back into a kernel,
  so without this file every offline question, which kernel orders the run or what a different
  round count would have predicted, costs a full retraining.  See :mod:`cnn_damg_term_pool`.
* ``<run>_config.json``, the run's provenance: dataset, full search-space configuration, loop
  parameters, device, timings, library versions.  Without it the CSVs of two runs are
  indistinguishable except by filename.
* ``<run>_trace.csv``, one row per pass of the outer loop: the acquisition value at the pick, the
  posterior mean and deviation there, the incumbent, the best so far, the fallback flag.  Whether
  the loop converges is read here.
* ``<run>_ea.csv``, one row per generation per pass of the inner evolutionary search.  Whether the
  acquisition optimizer optimizes is read here and nowhere else, because the loop keeps only the
  final population of the last pass.
* ``<run>_surrogate.csv``, one row per pass: the leave-one-out calibration, the held-out fit, the
  log marginal likelihood and the fitted kernel hyperparameters.  Whether the model gets better is
  read here.  The acceptance checks answer only whether it ended up usable.
* ``<run>_diagnostics.json``, the acceptance checks over the finished run.

A random-search baseline is the same script with ``--n-pre-samples 30 --n-iterations 0``.  The
initial dataset already comes from the size-uniform sampler, so a run with no passes evaluates
thirty drawn terms and nothing else, through the same code and into the same artifacts.  The
provenance names the run kind, read off ``n_iterations`` rather than off a flag beside it, so the
two cannot disagree.

That baseline is an independent sample at the same budget, not a paired one, and the difference
matters for how the comparison is read.  Two processes at the same seed do not draw the same terms,
because the synthesized program comes out in a different rule order per process and the stream
differs with it.  The distribution is unaffected, which is what makes the comparison sound.  What
is lost is the variance reduction a common initial design would have bought.  Pairing the two means
running both from one process, which is what the paired baseline of :func:`run_ask_tell_search`
does.

Training is not deterministic, since the weights are drawn and the mini-batches are shuffled, and
no seed is fixed unless the caller passes one.  The values recorded here are the reference to
compare against when a structure is retrained later, and small deviations on re-evaluation are
expected.
"""

import _thread
import contextlib
import csv
import json
import os
import random
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from cosy.core import Synthesizer
from cosy.evolutionary_algorithms import (
    EvolutionarySearch,
    ExpScalarization,
    Generations,
    GenerousConservativeReplacement,
    RankBasedSelection,
    ResolutionMutation,
    SampledInitialization,
    ScalarFitnessComparator,
    SubtreeSwap,
)
from cosy.search import DepthBoundedRandomSampler, SizeUniformSampler, generator_query
from cosy.search.determinize import determinize

from bayesian_optimization.diagnostics import (
    read_calibration,
    read_fit,
    read_frontier,
    read_gram,
    read_trace,
    trace_columns,
    trace_rows,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_network_algebras import (
    P50,
    P50_CORRECTED_CIFAR,
    P50_CORRECTED_DIGITS,
    TrainingProtocol,
    apply_he_initialisation,
    pytorch_components_algebra,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_network_algebras import (
    learner as raw_learner,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_pool import (
    TermPoolWriter,
    read_term_pool,
)
from bayesian_optimization.examples.cnn_damg_nas.recognizable_cnn_damg_repo import (
    check_alphabet,
)
from bayesian_optimization.initial_sampling import distinct_prefix

# --- Time budgets ---------------------------------------------------------------------------
# Not limits: thresholds past which a step reports that it is still running.  A run of this kind is
# long by design, so "still going" and "stuck" look identical from outside, and the difference
# otherwise only shows up hours later in a log that stops mid-sentence.
#
# The numbers are orders of magnitude, not expectations.  They are thresholds for reporting, so a
# threshold that has gone out of date makes a run noisier or quieter, never wrong.
SPACE_CONSTRUCTION_WARN_SECONDS = 300
# The two steps that turn the synthesized program into one that can be counted: the determinization
# and the counting construction of the size-uniform sampler.  Both are paid once per run, and both
# grow with the size of the program rather than with the number of evaluations.
DETERMINIZATION_WARN_SECONDS = 900
PER_EVALUATION_WARN_SECONDS = 900
ACQUISITION_WARN_SECONDS = 600
# The one hard limit, and it is deliberate: an acquisition optimization that has run for an hour is
# not slow, it is stuck.  The draw cost of the depth-bounded sampler has a heavy tail on spaces
# whose parameters are literals, and with a positive mutation rate that tail sits in the loop rather
# than switched off.  Nothing is lost by stopping, because the evaluation log flushes every row as
# it is written.  Set to None to only report.
#: When to give up on one acquisition maximization.  A limit and not a warning, because a hanging
#: inner search is a known failure mode here rather than a slow day.
#:
#: It bounds a legitimate duration, so it has to be read against the space being searched.  One
#: acquisition step converts every offspring of every generation into a graph, so its cost is the
#: population times the generations times the conversion cost of one term, and that conversion cost
#: grows with the term.  A large architecture at a large population can exceed an hour with nothing
#: wrong, which is why a run may raise the limit rather than edit it here.
ACQUISITION_HARD_LIMIT_SECONDS = 3600


# --- The program a run searches ---------------------------------------------------------------
# The bound D of the loop's own sampler, on the size of a term rather than on its depth.  Size-
# uniform sampling draws a realized size uniformly and then an inhabitant of that size uniformly,
# and the size of a term is the number of function symbols it writes.
#
# The bound has to admit something, and it has to admit it at a cost the counting can pay.  Below
# the size of the smallest term of a space it admits nothing at all, and the initializer says so
# rather than returning a short population.  Raising it widens the design at the cost of drawing
# larger architectures, and every one of those is a network this experiment trains.
DEFAULT_SIZE_BOUND = 200
# The bound of a depth-bounded sampler, which the evolutionary operators take and which a run takes
# for its own draws where the counting is unaffordable.  A depth-bounded sampler runs a depth-first
# search whose clause order is drawn uniformly and independently at each node, and it discards a
# child whose partial inhabitant exceeds the bound.
#
# It stands beside the size bound rather than anywhere else, because the one mistake worth
# preventing is reading the two as the same number.  A depth of 1000 and a size of 200 bound
# different things, and the convergence argument for an evolutionary search asks that every
# individual a run can hold lies within the bound of its sampler, which is a statement about one
# measure at a time.
DEFAULT_DEPTH_BOUND = 1000
# Well above the product state count any configuration of this example reaches, and still a bound.
# A recognizable constraint asks for a finite carrier, and an abstraction without one makes the
# fixed point run forever instead of reporting itself.
DETERMINIZATION_STATE_LIMIT = 1_000_000


class DeterminizedSizeUniformSampler:
    """Draws size-uniformly from the determinized program, for a loop that searches the coupled one.

    Why the two programs cannot be the same one.  Counting needs the determinized form, because the
    coupled program's swap laws read a hole, so no table indexed by the non-terminal is right and
    the counting would have to build the retained search tree instead.  Searching needs the coupled
    form, because the determinization is a product construction and the product is the larger
    program, every rule of which the mutation's residual query and the recombination's membership
    test walk.  One pass of the inner search runs on the order of the population times the mutation
    rate times the generations mutations, and as many recombinations at the crossover rate, so the
    difference between the two programs is paid once per operator application.

    Why this is sound.  ``determinize`` derives exactly the terms the original derives, along
    exactly one branch each, so the two programs are one language.
    ``tests/test_recognizable_cnn_repo.py`` pins that by counting both and comparing the rows.  A
    term drawn here is therefore an inhabitant of the space the loop searches, which is the whole of
    what the loop asks of a sampler.

    What it refuses.  A sampler is handed the query it is meant to answer, and this one answers a
    different query than the one it is given, which is safe exactly as long as the given query is
    the one it was built to stand in for.  So it checks rather than assuming: a partial-term query,
    another program or another requested type all raise.  The mutation poses partial-term queries
    and must never reach this object, and it has its own sampler for that reason.

    Attributes:
        size_bound (int): The bound D on the term size, forwarded to the inner sampler.
        counting (str): ``"table"``, named so that :func:`describe_sampler` can read it off.
    """

    def __init__(self, determinization, coupled_space, coupled_request, size_bound, rng):
        """Build the sampler.

        Args:
            determinization: The ``Determinization`` of the program the loop searches.
            coupled_space: That program, as synthesized, which is what the loop hands in its query.
            coupled_request: The requested type, likewise.
            size_bound (int): The bound D on the term size.
            rng (random.Random): The source of randomness.
        """
        self._determinization = determinization
        self._coupled_space = coupled_space
        self._coupled_request = coupled_request
        self._inner = SizeUniformSampler(size_bound, rng, counting="table")
        self._query = generator_query(determinization.space, determinization.start)
        self.size_bound = size_bound
        self.counting = "table"

    def _checked(self, query):
        """Verify the query is the one this sampler stands in for, and return its own.

        Args:
            query: The query the caller handed in.

        Returns:
            The generator query over the determinized program.

        Raises:
            ValueError: If the query is a partial-term query, or names another program or another
                requested type.  Answering it from the determinized program would then be a
                different question than the one asked.
        """
        if query.tree is not None:
            msg = (
                "this sampler answers the generator query of its program and nothing else; it "
                "was handed a partial-term query, which asks for the completions of one term and "
                "is not a question the determinized program can be asked in its place"
            )
            raise ValueError(msg)
        if query.solution_space is not self._coupled_space:
            msg = (
                "this sampler stands in for one program and was handed a query against another; "
                "the terms it draws would be inhabitants of a space the caller is not searching"
            )
            raise ValueError(msg)
        if query.start != self._coupled_request:
            msg = (
                f"this sampler stands in for the request {self._coupled_request} and was handed "
                f"a query for {query.start}"
            )
            raise ValueError(msg)
        return self._query

    def sample(self, query):
        """Stream the completions in size-uniform order, without replacement.

        Args:
            query: The loop's generator query over the coupled program.

        Yields:
            Tree: Inhabitants of that program, drawn by counting the determinized one.
        """
        yield from self._inner.sample(self._checked(query))

    def at_least(self, query, count):
        """Decide whether the bound admits at least ``count`` inhabitants.

        Args:
            query: The loop's generator query over the coupled program.
            count (int): The number asked for.

        Returns:
            bool: Whether the bound admits that many.
        """
        return self._inner.at_least(self._checked(query), count)

    def forget(self):
        """Drop the cached counting construction."""
        self._inner.forget()


@dataclass(frozen=True)
class SearchProgram:
    """What a run searches, the symbol to query it at, and the sampler it draws its own terms from.

    The program is always the one the repository synthesizes.  The evolutionary operators walk it on
    every mutation and every recombination, and they walk the larger program on the determinized
    form, see :class:`DeterminizedSizeUniformSampler`.  Where the determinization happens at all it
    happens inside the sampler, which is the only component that counts.

    The three travel together because choosing them apart is how a run breaks.  The counting sampler
    applies to a determinized program and to no other, while the depth-bounded one is the answer
    where the counting is unaffordable, and pairing the wrong two leaves the loop with an
    initializer that works and a duplicate fallback that cannot run.  That is the trap the
    ``sampler`` parameter of ``BayesianOptimization`` was introduced to close, and here it is closed
    one level up: there is one decision, :func:`build_search` makes it, and the invalid combinations
    are not expressible.

    Attributes:
        space: The program to search, as synthesized.
        request: The symbol to query it at, which is the requested type.
        sampler: The loop's own source of terms, for the initial dataset and the duplicate
            fallback.
        provenance (dict): What the construction cost and how large it came out, for the run
            record.  Read off the objects, not written beside them.
    """

    space: Any
    request: Any
    sampler: Any
    provenance: dict


def build_search(
    repository,
    target,
    *,
    sampling="size-uniform",
    size_bound=DEFAULT_SIZE_BOUND,
    depth_bound=DEFAULT_DEPTH_BOUND,
    seed=0,
    state_limit=DETERMINIZATION_STATE_LIMIT,
):
    """Build the program a run searches together with the sampler that fits it.

    Why the two are one decision.  ``CNNrepository`` states its four swap laws as term predicates
    over two sibling holes, and a predicate that reads a hole turns the residual there into a
    relation.  For several holes the residual need not be a product of the single-hole residuals: a
    hole occurring at two positions couples them syntactically, and an external predicate may relate
    distinct holes, so a subterm admissible at one hole is admissible only for certain fillings of
    another.  No table indexed by the non-terminal can be right about that, so the counting would
    have to build the retained search tree instead, and on this space that tree is what makes the
    sampler not slow but unfinishable.

    ``RecognizableCNNrepository`` states the same four laws as an abstraction with a relation, which
    is the form the determinization asks for, and
    :func:`cosy.search.determinize.determinize` pushes them into the non-terminals.  The result
    derives exactly the same terms along exactly one branch each and carries no predicate over a
    hole, so the size table applies to it and only to it.

    Which of the two modes a configuration can afford is a property of the repository rather than of
    the method.  A configuration that caps its linear layers keeps the product small enough to
    count, and one that does not can reach a product whose size table does not finish.  That is why
    the mode is a parameter here and not a decision this module makes.

    Args:
        repository: The repository.  ``sampling="size-uniform"`` needs one whose constraints are
            recognizable, which ``RecognizableCNNrepository`` is.  The plain ``CNNrepository`` is
            refused by the determinization, which names the clauses it cannot compile.
        target: The requested type.
        sampling (str): ``"size-uniform"`` determinizes and counts from the program.
            ``"depth-bounded"`` searches the program as synthesized and never counts.
            (Default value = "size-uniform")
        size_bound (int): The bound D on the term size, for the size-uniform sampler.
            (Default value = :data:`DEFAULT_SIZE_BOUND`)
        depth_bound (int): The bound on the term depth, for the depth-bounded sampler.
            (Default value = :data:`DEFAULT_DEPTH_BOUND`)
        seed (int): The sampler's seed.
        state_limit (int): How many product states the determinization may reach.
            (Default value = :data:`DETERMINIZATION_STATE_LIMIT`)

    Returns:
        SearchProgram: The program, the symbol to query, the sampler, and the provenance.

    Raises:
        ValueError: If ``sampling`` is neither of the two names.  Which sampler a space admits is
            not something to guess a default for.
    """
    if sampling not in ("size-uniform", "depth-bounded"):
        msg = (
            f"sampling selects how the loop draws its own terms and is 'size-uniform' or "
            f"'depth-bounded', not {sampling!r}"
        )
        raise ValueError(msg)

    started = time.time()
    with step_budget("search-space construction", SPACE_CONSTRUCTION_WARN_SECONDS):
        coupled = Synthesizer(repository.specification(), {}).construct_solution_space(
            target
        ).prune()
    construction_seconds = time.time() - started
    coupled_nonterminals = len(tuple(coupled.nonterminals()))
    coupled_rules = sum(len(coupled.get(nt) or ()) for nt in coupled.nonterminals())
    print(
        f"Search space construction took {construction_seconds:.2f}s "
        f"({coupled_nonterminals} non-terminals, {coupled_rules} rules)",
        flush=True,
    )

    provenance = {
        "repository": type(repository).__name__,
        "sampling": sampling,
        "coupled_nonterminals": coupled_nonterminals,
        "coupled_rules": coupled_rules,
        "search_space_construction_seconds": construction_seconds,
    }

    if sampling == "depth-bounded":
        # The program as synthesized, predicates and all.  Nothing counts it, so nothing has to.
        return SearchProgram(
            space=coupled,
            request=target,
            sampler=DepthBoundedRandomSampler(depth_bound, random.Random(seed)),
            provenance={**provenance, "depth_bound": depth_bound},
        )

    # check_alphabet first: the abstraction is stated against a fixed alphabet, and a terminal it
    # has never seen would otherwise be folded into the "everything else" state, deciding the laws
    # on a state that cannot represent them.  It refuses the space instead.
    alphabet = check_alphabet(coupled)
    started = time.time()
    with step_budget("determinization", DETERMINIZATION_WARN_SECONDS):
        determinization = determinize(coupled, target, state_limit=state_limit)
    determinization_seconds = time.time() - started
    rules = sum(
        len(determinization.space.get(nt) or ()) for nt in determinization.space.nonterminals()
    )
    print(
        f"Determinization took {determinization_seconds:.2f}s "
        f"({determinization.state_count} product states, {rules} rules)",
        flush=True,
    )

    # The coupled program is what the loop searches, here as in the other mode.  Only the sampler
    # sees the determinized one, because counting is the only thing it is better at.
    return SearchProgram(
        space=coupled,
        request=target,
        sampler=DeterminizedSizeUniformSampler(
            determinization, coupled, target, size_bound, random.Random(seed)
        ),
        provenance={
            **provenance,
            "size_bound": size_bound,
            "determinization_seconds": determinization_seconds,
            "determinization_state_count": determinization.state_count,
            "determinization_rules": rules,
            "determinization_state_limit": state_limit,
            "abstraction_count": len(determinization.abstractions),
            "terminals": alphabet["terminals"],
        },
    )


# --- The acquisition optimizer --------------------------------------------------------------
# An evolutionary search converges to a best individual under five conditions, and four of them are
# decided by the components chosen below.  The fifth asks that the individuals a run can hold form a
# finite set, and that is a property of the space rather than of the components: the requested type
# carries structure literals over finite collections, so the set is finite.
#
# One of the four asks for an exhaustive sampler, which means that on every resolution query and for
# every completion within the sampler's bound, the first element of the stream is that completion
# with positive probability.  It is "every resolution query" that carries the weight here, because
# the mutation poses a residual query rather than a generator query, and a sampler that reaches only
# the generator queries would leave the mutation rate buying nothing.
#
# The position distribution is the other half, and it is not uniform over all positions: the
# operator draws among the non-leaves and the root.  A leaf holding a literal is a position where
# the residual query answers with the term already there, since the clause matcher pins a constant
# argument even at the opened position, so drawing one spends the mutation on the identity, and this
# repository is built from literal parameters.  Excluding the leaves keeps reachability, because the
# root is always a mutation point and the residual there is the generator query.
DEFAULT_SELECTION_PRESSURE = 1.7
# The convergence conditions ask for a crossover rate below 1 and a mutation rate above 0 and fix
# nothing else, so the two numbers are a choice, and this is the one place it is made.
#
# The mutation rate is per offspring and the operator is a macro-mutation.  ``ResolutionMutation``
# draws a position among the non-leaves and the root and replaces the whole subtree there with a
# fresh draw, so nothing here is analogous to flipping a bit.  How much of a term one mutation
# replaces therefore depends on how large the terms of the space are: on a large term the root is a
# rarer draw and an ordinary mutation replaces a smaller share, so the same rate shakes a large
# space less than a small one.  0.03 is chosen for the architectures this example searches, whose
# terms are large.  The two smaller example spaces of this repository sit lower still, at 0.02 in
# simple_nas and at 0.02 in the DAMG example.
#
# Convergence is untouched by the value.  The root is always a mutation point, and a mutation with
# an exhaustive sampler maps any individual to any other with positive probability through the root,
# so reachability holds for every positive rate.  The rate decides how hard the search is shaken,
# not what it can reach.  What the value is not is optimized: whether the inner search finds the
# argmax of the acquisition function is not measured here.
DEFAULT_CROSSOVER_RATE = 0.9
DEFAULT_MUTATION_RATE = 0.03
# Why the evolutionary operators stay depth-bounded while the loop's own sampler is size-uniform, on
# the very same program.  The two ask different questions.  The loop poses one generator query for
# the whole run, so a size-uniform sampler builds its counting construction once and every later
# draw reads the table it built.  The mutation poses a residual query at a fresh position of a fresh
# individual every time, and ``SizeUniformSampler`` keeps only the last construction it built, so a
# size-uniform mutation would pay that construction again per draw, and one pass of the loop makes
# as many draws as it makes mutations.  A depth-bounded draw never counts.
#
# Nothing in the convergence argument is given up by that.  The depth-bounded sampler is exhaustive
# on every resolution query, which is the condition the operators have to meet, and the finiteness
# condition comes from the space rather than from the bound.


def build_acquisition_optimizer(
    *,
    population_size: int,
    generations: int,
    depth_bound: int = DEFAULT_DEPTH_BOUND,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    selection_pressure: float = DEFAULT_SELECTION_PRESSURE,
    seed: int = 0,
) -> EvolutionarySearch[Any, Any, Any]:
    """Build the evolutionary search that maximizes the acquisition function.

    Each component gets its own generator, derived from ``seed``, so that changing one operator's
    consumption does not shift every later draw of the run.

    Rank-based rather than fitness-proportional parent selection.  Fitness-proportional weights
    would be the raw acquisition values, and expected improvement spans orders of magnitude on a
    confident surrogate, so a single individual takes nearly all of the selection probability.  Rank
    selection depends on the ordering alone.  It gives every member of the population a positive
    probability, which is one of the convergence conditions, as long as the pressure stays below 2:
    at 1.7 the worst individual keeps a share of 0.3 divided by the population size.

    Args:
        population_size (int): The population size.
        generations (int): The termination bound.
        depth_bound (int): The bound of the samplers, on the depth of a term.
        crossover_rate (float): The crossover rate, which the convergence conditions need below 1.
        mutation_rate (float): The mutation rate, which they need above 0.
        selection_pressure (float): The rank-selection pressure, below 2 so that every member keeps
            a positive probability.
        seed (int): The base seed.  Every component derives its own generator from it.

    Returns:
        EvolutionarySearch[Any, Any, Any]: The configured search.
    """
    def rng(offset: int) -> random.Random:
        """Derive one component's generator.

        Args:
            offset (int): The component's index.

        Returns:
            random.Random: Its generator.
        """
        return random.Random(seed * 100 + offset)

    return EvolutionarySearch(
        initializer=SampledInitialization(DepthBoundedRandomSampler(depth_bound, rng(0))),
        mutation=ResolutionMutation(DepthBoundedRandomSampler(depth_bound, rng(1)), rng(2)),
        # No max_size: the finiteness the convergence argument asks for comes from the space itself,
        # and a size cap on the acceptance test would be a second measure beside the sampler's depth
        # bound, which is the one confusion this module is written to avoid.
        recombination=SubtreeSwap(rng(3)),
        parent_selection=RankBasedSelection(selection_pressure, rng=rng(4)),
        # Generous and conservative, which is the condition on survivor selection and the component
        # that carries the convergence guarantee.  Truncation keeps a best member and is therefore
        # conservative, but it gives a worse member no chance of surviving, so it is not generous.
        survivor_selection=GenerousConservativeReplacement(ExpScalarization(), rng(5)),
        termination=Generations(generations),
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        rng=rng(6),
        comparator=ScalarFitnessComparator(True),  # the acquisition is always maximized
    )


def describe_sampler(sampler) -> dict:
    """Read a sampler's kind and bounds off the object, for the run record.

    ``repr`` of a sampler is its class and its address, which says nothing a second run can be
    compared against, and the two samplers this repository uses are bounded in different measures,
    so which bound it was is exactly the question a config file has to answer.

    Args:
        sampler: The sampler the loop draws its own terms from.

    Returns:
        dict: Its class name and whichever bounds it carries, with ``None`` for the ones it does
            not.
    """
    return {
        "type": type(sampler).__name__,
        # A term size and a term depth are not the same number, and a record that reports one of
        # them under the other's name describes a run that never happened.
        "size_bound": getattr(sampler, "size_bound", None),
        "depth_bound": getattr(sampler, "depth_bound", None),
        "counting": getattr(sampler, "counting", None),
    }


def _as_list(values: Any) -> list | None:
    """Render an optional sequence for a record.

    Args:
        values (Any): The sequence, or None.

    Returns:
        list | None: The values as a list, or None.
    """
    return None if values is None else list(values)


def describe_repository(repo: Any) -> dict:
    """Read the search space of a run off the repository that will build it.

    The block this returns used to be written from the module constants of the experiment while
    the repository was built from a different set of them, so a run of the VGG cell recorded the
    tutorial cell's dimensions beside a correct count of its own non-terminals. One field of that
    block was read off the object, and it is the one that disagreed with the rest.

    Args:
        repo (Any): The repository the run searches over.

    Returns:
        dict: The parameters that decide which networks the space holds, as the repository has
            them. ``num_feature_dimensions`` is a count and not a parameter: it is what the widths
            close to under parallel sums, and it is what the space is large or small because of.
    """
    return {
        "linear_feature_dimensions": list(repo.linear_feature_dimensions),
        "channel_dimensions": list(repo.channel_dimensions),
        "height_width_dimensions": [list(pair) for pair in repo.height_width_dimensions],
        "kernel_dimensions": [list(pair) for pair in repo.kernel_dimensions],
        "pooling_kernel_dimensions": [list(pair) for pair in repo.pooling_kernel_dimensions],
        "stride_values": list(repo.stride_values),
        "padding_values": list(repo.padding_values),
        "max_parallel_width": repo.max_parallel_width,
        "max_lin_layer_dim": repo.max_lin_layer_dim,
        # Normalized rather than as given: the repository appends 0 and 1 if they are missing,
        # because they are the neutral values of the sum and the product component, and the space
        # was built from the normalized list.
        "constant_values": list(repo.constant_values),
        "learning_rate_values": list(repo.learning_rate_values),
        "n_epoch_values": list(repo.n_epoch_values),
        # None where the repository was given none, since the optimizer combinator then substitutes
        # its own single value and a list here would claim the caller chose it.
        "weight_decay_values": _as_list(repo.weight_decay_values),
        "momentum_values": _as_list(repo.momentum_values),
        "num_feature_dimensions": len(repo.feature_dimensions),
    }


def describe_search(evo_alg: EvolutionarySearch[Any, Any, Any]) -> dict:
    """Read a run's evolutionary configuration off the object that will run it.

    The provenance record used to be written by hand beside the construction, and it drifted: it
    named ``AgeBasedReplacement`` for runs that used truncation, and called ``mutation_rate = 0``
    mandatory long after the defect behind that rule was fixed.  A metadata block that cannot be
    read off the object is a comment, and comments go stale silently.

    Args:
        evo_alg (EvolutionarySearch[Any, Any, Any]): The configured search.

    Returns:
        dict: The component names and numeric parameters, as they actually are.
    """
    def name(component: object) -> str:
        """Render one component for the record.

        Args:
            component (object): The component.

        Returns:
            str: Its class name.
        """
        return type(component).__name__

    return {
        "initializer": name(evo_alg.initializer),
        "mutation": name(evo_alg.mutation),
        "recombination": name(evo_alg.recombination),
        "parent_selection": name(evo_alg.parent_selection),
        "survivor_selection": name(evo_alg.survivor_selection),
        "termination": name(evo_alg.termination),
        "comparator": name(evo_alg.comparator),
        "population_size": evo_alg.population_size,
        "crossover_rate": evo_alg.crossover_rate,
        "mutation_rate": evo_alg.mutation_rate,
    }


@contextlib.contextmanager
def step_budget(label, expected_seconds, hard_limit_seconds=None):
    """Report a step that outruns its expected duration, while it is still running.

    A run of these experiments is long by design, since a single pass trains a network, so "it is
    still going" and "it is stuck" look alike from outside, and the difference only shows up hours
    later in a log that ends mid-sentence.  This makes the difference observable: a watchdog thread
    prints once the step passes the duration it was expected to take, and again at every doubling
    of that, so a step that runs 60x its budget says so 6 times instead of 60.

    It reports and it does not kill.  Aborting a step that trains a network for 15 minutes because
    it took 20 would destroy more than it saves.  ``hard_limit_seconds`` exists for the steps where
    hanging is a known failure mode rather than a slow day, such as sampling on a space whose draw
    cost has a heavy tail.  It raises ``KeyboardInterrupt`` in the main thread, so the surrounding
    ``with`` blocks still close their files.

    Args:
        label (str): What is being timed, as it should read in the log.
        expected_seconds (float): The duration beyond which the step is worth reporting.  Not a
            limit: exceeding it is normal on a slower machine, which is why the message says what
            was expected instead of claiming a failure.
        hard_limit_seconds (float | None): Abort the run past this duration, or None to only
            report. (Default value = None)

    Yields:
        None: The context in which the step runs.
    """
    if expected_seconds <= 0:
        # A threshold of zero makes the wait below return at once, every time, so the watchdog
        # prints as fast as the interpreter allows and the step it was watching gets no processor.
        # A caller computing this as a per-item budget times a count reaches zero the moment the
        # count is zero, so it is worth catching rather than documenting.
        msg = f"the expected duration of {label!r} must be positive, got {expected_seconds}"
        raise ValueError(msg)

    started = time.time()
    finished = threading.Event()

    def watch():
        """Report at the expected duration and at every doubling of it."""
        threshold = float(expected_seconds)
        while True:
            # The hard limit is its own deadline, not something checked when a report happens to
            # be due.  Waking only at the reporting thresholds made a limit of 3600 s fire at 4800,
            # the first doubling past it, and a limit below the first threshold never fire at all.
            deadlines = [threshold]
            if hard_limit_seconds is not None:
                deadlines.append(float(hard_limit_seconds))
            wake_at = min(d for d in deadlines if d > time.time() - started) \
                if any(d > time.time() - started for d in deadlines) else 0.0
            if finished.wait(timeout=max(wake_at - (time.time() - started), 0.0)):
                return
            elapsed = time.time() - started
            if hard_limit_seconds is not None and elapsed >= hard_limit_seconds:
                print(
                    f"[watchdog] {label}: past the hard limit of {hard_limit_seconds:.0f}s, "
                    "interrupting the run",
                    flush=True,
                )
                _thread.interrupt_main()
                return
            if elapsed < threshold:
                continue
            print(
                f"[watchdog] {label}: running for {elapsed:.0f}s, expected about "
                f"{expected_seconds:.0f}s",
                flush=True,
            )
            threshold *= 2

    watcher = threading.Thread(target=watch, name=f"watchdog:{label}", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        finished.set()
        elapsed = time.time() - started
        if elapsed > expected_seconds:
            print(
                f"[watchdog] {label}: finished after {elapsed:.0f}s, "
                f"{elapsed / max(expected_seconds, 1e-9):.1f}x the expected duration",
                flush=True,
            )


CSV_COLUMNS = [
    "phase",            # "pre_sample" | "bo_step" | "random_sample"
    "index",            # position within the initial sample, or 0-indexed BO iteration
    "structure",        # pretty-printed term (architecture + loss + optimizer + epochs)
    "objective_value",  # raw validation loss, exactly what the objective function returned
    # The validation accuracy of the same trained model, which is what the search maximizes.  Loss
    # and accuracy do not order the candidates the same way, and a lower loss can come with a lower
    # accuracy, so both are recorded.
    "accuracy",
    # The held-out accuracy.  Never a search signal: ranking candidates by it selects on the split
    # that is meant to stay untouched, and the column sits in the same row as the validation number,
    # which makes it easy to read by accident.  Empty for runs without a test split.
    "test_accuracy",
    # The number of function symbols the term writes, which is the axis the loop's own sampler
    # stratifies along.  Without it a size-uniform initial design cannot be shown to be one.
    "term_size",
    "n_params",         # trainable parameter count, to see whether the loop drifts to large nets
    "train_seconds",    # wall-clock training time
    # Whether the numerics gave out, and how far the training got before they did.  Without these a
    # network that stopped in its first epoch is indistinguishable from one that trained through and
    # is merely bad: the abort still yields a finite accuracy, and on logits that are all not a
    # number ``argmax`` returns class 0, so the accuracy becomes that class's frequency.  Divergence
    # depends on the initial weights rather than on the architecture, so it belongs in the noise
    # column and not in the variance between architectures, and it is the one property of an
    # evaluation that cannot be recovered once the run is over.
    "diverged",
    "epochs_completed",
    # Beside the training time, because the two are the run's whole wall clock and which of them
    # dominates decides where a run is spending itself: only the training uses the accelerator, so a
    # run that spends most of its time here is bound by the processor however fast the card is.
    # Empty for pre_sample rows, which have no acquisition step.
    "acquisition_seconds",
    # The three columns below decide how the row may be read at all.  When ``suggest()`` cannot
    # find a novel candidate it replaces the optimizer's result with a random fallback sample, and
    # ``acquisition_value`` then describes that replacement.  A run whose loop-pass rows all report
    # a fallback was random search, and without these columns that is indistinguishable in the
    # finished file from a run that was not.  Empty for pre_sample rows, which have no acquisition
    # step.
    "acquisition_value",
    "fallback_used",
    "fallback_attempts",
    "timestamp",
    # The four columns of a repeated measurement.  ``accuracy`` above is their mean, which is what
    # the search optimizes, and these say what it is a mean of: how many trainings, their individual
    # values, their spread, and whether any single one of them diverged.  A run that kept only the
    # mean could not afterwards tell a candidate that measured 0.72 three times from one that
    # measured 0.50, 0.72 and 0.94, and the whole reason for repeating is that those two are not the
    # same finding.  Empty for a run with one training per candidate, where no repetition happened
    # to describe.
    "n_repeats",
    "accuracy_runs",
    "accuracy_std",
    "diverged_runs",
]


def split_train_validation(x, y, val_fraction=0.1, seed=20260803):
    """Split a training set into a training and a validation part, deterministically.

    Deterministic on purpose, and with its own seed.  Two runs have to see the same split, or their
    numbers are not comparable, and the split must not move when the search seed moves, or a sweep
    over seeds would silently be a sweep over splits as well.

    The default keeps a tenth of the training set for validation.  There is no single standard for
    the share, and the trade is plain: a larger validation part measures an accuracy more precisely,
    while a larger training part reaches a higher one.  What decides the trade is which of the two
    error terms is larger, the sampling error of the validation set or the spread of the training
    itself, and on these datasets the training spread is the larger of the two, so the split favors
    training data.

    Args:
        x (torch.Tensor): Training features.
        y (torch.Tensor): Training labels.
        val_fraction (float): Share that becomes validation. (Default value = 0.1)
        seed (int): Seed of the permutation. (Default value = 20260803)

    Returns:
        tuple: ``(x_train, y_train, x_val, y_val)``.
    """
    n = x.shape[0]
    n_val = int(round(n * val_fraction))
    if not 0 < n_val < n:
        msg = f"val_fraction {val_fraction} yields {n_val} of {n} samples"
        raise ValueError(msg)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(n, generator=generator).to(x.device)
    val_idx, train_idx = permutation[:n_val], permutation[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def _train_candidate_once(tree, x, y, x_val, y_val, batch_size, protocol,
                          x_test, y_test, seed):
    """Interpret, initialize and train one candidate exactly once.

    Split out of :func:`evaluate_candidate` so that repeating a measurement repeats all of it.  A
    repetition that reused the interpreted model would train an already-trained network and measure
    something else entirely.  ``Tree.interpret`` builds a new one on every call, so the
    interpretation belongs inside the repeated part rather than before it.

    Args:
        seed (int | None): Seeds the weight initialization and the mini-batch shuffling of this one
            training.  ``None`` leaves the global generator alone, which is unseeded training.  The
            caller is responsible for restoring the generator state, see
            :func:`evaluate_candidate`.

    Returns:
        dict: The metrics of this single training.
    """
    if seed is not None:
        # Before ``interpret``, because the weights are drawn while the modules are constructed.
        # Seeding afterwards would leave the initialization as the one unseeded step.
        torch.manual_seed(seed)

    model, loss_fn, optimizer_factory, epochs, scheduler_factory = tree.interpret(
        pytorch_components_algebra())
    n_params = sum(p.numel() for p in model.parameters())
    input_features = x.shape[-1]

    # Initialization happens here rather than inside ``learner``, because this is the one point
    # where a freshly interpreted model is in hand, and ``Tree.interpret`` builds a new one on every
    # call, so there is no risk of initializing a model that already trained.
    if protocol is not None and protocol.init == "he":
        apply_he_initialisation(model)

    # ``learner`` fills this in, and its own comment says why the flag has to leave the function.
    # The measurement is reported exactly as taken either way: this names it, it does not replace
    # it.
    training = {}
    started = time.time()
    objective_value = raw_learner(input_features, model, loss_fn, optimizer_factory, epochs,
                                  x, y, x_val, y_val, batch_size=batch_size, report=training,
                                  protocol=protocol, scheduler=scheduler_factory)
    train_seconds = time.time() - started

    # `learner` already moved the model onto the data's device and trained it in place.
    with torch.inference_mode():
        predictions = model(x_val).argmax(dim=-1)
        accuracy = (predictions == y_val).float().mean().item()
        # Recorded, never optimized against, see the docstring.
        test_accuracy = None
        if x_test is not None and y_test is not None:
            test_predictions = model(x_test).argmax(dim=-1)
            test_accuracy = (test_predictions == y_test).float().mean().item()

    return {
        "objective_value": objective_value,
        "accuracy": accuracy,
        "test_accuracy": test_accuracy,
        "n_params": n_params,
        "train_seconds": train_seconds,
        "diverged": training["diverged"],
        "epochs_completed": training["epochs_completed"],
    }


def evaluate_candidate(tree, x, y, x_val, y_val, batch_size, protocol=None,
                       x_test=None, y_test=None, repeats=1, training_seeds=None):
    """Train one candidate and return all recorded metrics.

    Uses ``pytorch_components_algebra`` rather than ``pytorch_function_algebra`` so that the trained
    model stays accessible for the accuracy and parameter measurements.  The training itself goes
    through the very same ``learner`` routine the function algebra would have used, so the objective
    value is directly comparable to runs that used it.

    What the search sees is the validation split.  ``accuracy`` and ``objective_value`` are measured
    on ``x_val``, and that is the number the loop maximizes.  If a test split is passed as well, its
    accuracy is recorded alongside as ``test_accuracy``, for the record only.

    ``test_accuracy`` is not a search signal and must not become one.  It sits in the same row as
    the validation number, which makes it easy to read by accident, and a table that ranks
    candidates by it has selected on the split that was meant to stay untouched.  Ranking, model
    choice and every stopping rule read ``accuracy``.

    Repetitions average the training noise away and never hide it.  The same architecture trained
    twice does not reach the same accuracy, and the spread between two trainings can be wider than
    the difference a search is trying to read, so a single training cannot decide between two
    candidates.  ``repeats`` trains the candidate that many times and hands the mean to the search,
    which divides that noise by the square root of the count.  The individual measurements survive
    in ``accuracy_runs`` and their spread in ``accuracy_std``: the mean is what the loop optimizes,
    the spread is what says whether a difference is readable at all, and a run that reported only
    the mean would have thrown away the very number that justified the averaging.

    Repetitions are neither free nor a substitute for a repeated run.  They average one
    architecture's training noise, while a sweep over search seeds asks a different question, namely
    whether the search finds the same thing twice.

    Args:
        x_val (torch.Tensor): The features the search is scored on.
        y_val (torch.Tensor): Their labels.
        protocol (TrainingProtocol | None): What the training does beyond reading the term, such as
            clipping, weight initialization and augmentation.  None means the default protocol, so
            an unset protocol changes nothing about an existing run.
        x_test (torch.Tensor | None): Held-out features, recorded but never optimized against.
        y_test (torch.Tensor | None): Their labels.
        repeats (int): How many independent trainings the reported metrics average over.
            (Default value = 1)
        training_seeds (Sequence[int] | None): One seed per repetition, or ``None`` for unseeded
            training.  Seeded repetitions make a run reproducible and make two runs comparable
            candidate by candidate, which an unseeded average does not.  The global generator is
            restored afterwards, so seeding the training cannot move the search's own draws, which
            come from the sampler and the acquisition optimizer and have to stay where the run's own
            seed put them. (Default value = None)

    Returns:
        dict: The averaged metrics, plus the per-repetition measurements they average over.

    Raises:
        ValueError: If ``repeats`` is not positive, if ``training_seeds`` has a different length, or
            if the repetitions disagree on the parameter count.  The last one would mean the same
            term interpreted to two different networks, which no averaging can repair.
    """
    if repeats < 1:
        msg = f"repeats must be at least 1, got {repeats}"
        raise ValueError(msg)
    if training_seeds is not None and len(training_seeds) != repeats:
        msg = (
            f"training_seeds has {len(training_seeds)} entries for {repeats} repetitions; "
            f"one seed per repetition or None for unseeded training"
        )
        raise ValueError(msg)

    seeds = list(training_seeds) if training_seeds is not None else [None] * repeats

    # The training's generator state is borrowed, not taken.  The loop's own sampler and the
    # acquisition-optimizing search draw from the same global generator, and a run whose training
    # advanced it would search differently than the same run with one training per candidate.
    rng_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        runs = [
            _train_candidate_once(tree, x, y, x_val, y_val, batch_size, protocol,
                                  x_test, y_test, seed)
            for seed in seeds
        ]
    finally:
        torch.set_rng_state(rng_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    parameter_counts = {run["n_params"] for run in runs}
    if len(parameter_counts) != 1:
        msg = (
            f"the {repeats} repetitions of one term produced different parameter counts "
            f"{sorted(parameter_counts)}; the same term must interpret to the same network"
        )
        raise ValueError(msg)

    accuracies = [run["accuracy"] for run in runs]
    objective_values = [run["objective_value"] for run in runs]
    # A test split is either configured for the whole run or for none of it, so the repetitions
    # agree.  None stays None rather than becoming a number no measurement produced.
    test_accuracies = [run["test_accuracy"] for run in runs]
    measured_test = [value for value in test_accuracies if value is not None]

    return {
        "objective_value": statistics.fmean(objective_values),
        "accuracy": statistics.fmean(accuracies),
        "test_accuracy": statistics.fmean(measured_test) if measured_test else None,
        "n_params": parameter_counts.pop(),
        # The sum, not the mean: this is what the run actually spent, and the column is read as
        # wall clock.
        "train_seconds": sum(run["train_seconds"] for run in runs),
        # Any repetition that diverged makes the candidate one whose numerics gave out.  Averaging
        # a diverged training with two healthy ones would report the mean of two different things,
        # and the flag is what says so.
        "diverged": any(run["diverged"] for run in runs),
        # The earliest stop, for the same reason: it is the one that says how far the worst of the
        # repetitions got.
        "epochs_completed": min(run["epochs_completed"] for run in runs),
        "n_repeats": repeats,
        "training_seeds": None if training_seeds is None else list(training_seeds),
        "accuracy_runs": accuracies,
        # The sample standard deviation of the repetitions.  It is undefined for a single one, and
        # left undefined rather than reported as 0.0, which would claim that a spread was measured
        # to be zero when none was measured at all.
        "accuracy_std": statistics.stdev(accuracies) if repeats > 1 else None,
        "objective_value_runs": objective_values,
        "test_accuracy_runs": test_accuracies,
        "diverged_runs": [run["diverged"] for run in runs],
    }


def _joined(values):
    """Render a per-repetition list into one CSV cell, or an empty cell if there is none.

    Args:
        values (Sequence | None): The per-repetition measurements.

    Returns:
        str: The values separated by single spaces, or "" when nothing was measured.
    """
    if not values:
        return ""
    return " ".join(str(value) for value in values)


class EvaluationLogger:
    """Writes one row and one term record per evaluation, flushed as they go.

    The files stay open for the whole run on purpose.  The point of this logger is that results
    survive a crash after hours of training, so every row is written and flushed when it happens
    rather than collected and dumped at the end.  That is what the class's own ``__enter__`` and
    ``__exit__`` are for: it is the context manager, so ``open`` here is not a leak.

    Both artifacts come from one call, because the term record is the one a caller would forget.
    The CSV keeps a rendering of the structure, which no kernel can read back, while the
    ``_terms.pickle`` beside it keeps the term itself, and every question asked of a finished run
    offline needs the second file.  Threading a separate writer through the three call sites would
    have let a fourth one omit it silently, and a run that omits it cannot be repaired afterwards.
    It has to be retrained.  See :mod:`cnn_damg_term_pool`.

    Args:
        path (str): The run's CSV.  The term records go beside it as ``<run>_terms.pickle``.
        pretty_algebra (Callable): The algebra that renders a term into the CSV's structure column.
        provenance (dict): Written into the term file's header, so the pickle identifies its own
            origin on a machine that received nothing else. (Default value = None)
    """

    def __init__(self, path, pretty_algebra, provenance=None):
        self._pretty_algebra = pretty_algebra
        self._file = open(path, "w", newline="")  # noqa: SIM115, see the class docstring
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_COLUMNS)
        self._file.flush()
        try:
            self._terms = TermPoolWriter(
                _sibling_path(path, "_terms.pickle"), provenance=provenance
            )
        except BaseException:
            # The CSV is already open at this point, and a caller that never got an instance back
            # cannot close it, since there is no ``__exit__`` for an object whose ``__init__``
            # raised.
            self._file.close()
            raise

    def log(self, phase, index, tree, metrics, suggestion=None, acquisition_seconds=None):
        """Append one evaluation, as one CSV row and one term record.

        ``metrics`` is an :func:`evaluate_candidate` result.  A missing key is written as an empty
        cell rather than raising, so a partially instrumented run still records its structures and
        objective values.

        ``suggestion`` is the ``Suggestion`` that produced this candidate, and it carries the
        acquisition value and the fallback state.  Pre-sample rows have none, and their cells stay
        empty: an empty cell says that there was no acquisition step here, whereas writing False
        would claim that a fallback was ruled out which was never evaluated.
        ``acquisition_seconds`` is empty for the same rows and for the same reason.
        """
        structure = tree.interpret(self._pretty_algebra())
        diagnostics = (suggestion.diagnostics or {}) if suggestion is not None else {}
        self._writer.writerow([
            phase,
            index,
            structure,
            metrics.get("objective_value", ""),
            metrics.get("accuracy", ""),
            "" if metrics.get("test_accuracy") is None else metrics["test_accuracy"],
            tree.size,
            metrics.get("n_params", ""),
            metrics.get("train_seconds", ""),
            metrics.get("diverged", ""),
            metrics.get("epochs_completed", ""),
            "" if acquisition_seconds is None else acquisition_seconds,
            "" if suggestion is None else suggestion.acquisition_value,
            diagnostics.get("fallback_used", ""),
            diagnostics.get("fallback_attempts", ""),
            time.time(),
            metrics.get("n_repeats", ""),
            # Space-separated rather than a nested list, so the cell stays one CSV field and
            # ``str.split()`` reads it back without a parser.
            _joined(metrics.get("accuracy_runs")),
            "" if metrics.get("accuracy_std") is None else metrics["accuracy_std"],
            _joined(metrics.get("diverged_runs")),
        ])
        self._file.flush()  # persist at once, so a crash mid-run loses no completed row
        self._terms.write(phase, index, tree, metrics)

    def close(self):
        try:
            self._terms.close()
        finally:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


EA_CSV_COLUMNS = [
    "bo_iteration",      # which pass of the outer loop this inner run belongs to
    "generation",        # 0 is the initial population
    "best",              # b, the fittest individual of the whole inner run so far
    "population_best",   # the fittest member of THIS generation, which b may already beat
    "population_mean",
    "population_worst",
    "distinct_members",  # a collapsed population repeats itself
    # The generation the best individual was last replaced in.  The gap to ``generation`` is how
    # long the inner search has been stalled, which is what a termination bound is set against.
    "last_improvement",
    # Zero says that every offspring of this generation was discarded by the acceptance test, which
    # the fitness columns alone do not show.
    "offspring",
]

SURROGATE_CSV_COLUMNS = [
    "bo_iteration",
    "n_train",                        # how many pairs the pass conditioned on
    "log_marginal_likelihood",
    # The leave-one-out calibration, per pass rather than once at the end.  Read the two spreads
    # together: the spread about zero and the spread about their own mean differ by the bias, which
    # is the signature of a surrogate whose deviations stay at the prior.
    "calibration_root_mean_square",
    "calibration_standard_deviation",
    "calibration_maximum_absolute",
    "calibration_outside_two",
    # The held-out fit, per pass.  The rank correlation alone does not separate a good surrogate
    # from a collapsed one, since a surrogate that predicts one value for everything can still order
    # a few ties by chance.  The prediction spread against the objective spread is what does, so all
    # three are here.
    "fit_size",
    "fit_rank_correlation",
    "fit_prediction_spread",
    "fit_objective_spread",
    "fit_residual_root_mean_square",
    # What the model selection actually did.  A kernel whose amplitudes never move is one whose fit
    # had nothing to adjust, and nothing else in these artifacts would say so, because sklearn runs
    # its optimizer over an empty parameter vector without complaining.
    "kernel_hyperparameters",
]


def kernel_hyperparameters(surrogate):
    """Return the fitted kernel's hyperparameters as ``{name: value}``.

    ``theta`` holds the logarithm of the value of every hyperparameter that lives on a log scale,
    which is all of the ones here, so it is exponentiated back into the amplitudes and noise levels
    a reader recognizes.  Walked with an explicit index rather than zipped, because a hyperparameter
    may hold several elements and a zip would then silently pair names with the wrong numbers.

    Args:
        surrogate: A fitted ``GaussianProcessRegressor``.

    Returns:
        dict: One entry per hyperparameter of the fitted kernel.
    """
    values = np.exp(surrogate.kernel_.theta)
    result = {}
    index = 0
    for hyperparameter in surrogate.kernel_.hyperparameters:
        count = hyperparameter.n_elements
        result[hyperparameter.name] = (
            float(values[index])
            if count == 1
            else [float(value) for value in values[index:index + count]]
        )
        index += count
    return result


class EAGenerationLogger:
    """Writes the trajectory of every acquisition maximization, one row per generation.

    The frontier read says where the answer sits in the population it came from.  This says whether
    the inner run got there by improving.  Convergence is a statement about the best individual over
    the generations, and without these rows a run holds no evidence for or against it, because the
    loop keeps only the last generation of the last pass.

    Flushed per pass for the same reason the evaluation log is: the inner run is finished before the
    network trains, and a run interrupted during the training should keep it.
    """

    def __init__(self, path):
        self._file = open(path, "w", newline="")  # noqa: SIM115, closed by __exit__
        self._writer = csv.writer(self._file)
        self._writer.writerow(EA_CSV_COLUMNS)
        self._file.flush()

    def log(self, bo_iteration, acquisition_run):
        """Append one inner run's generations.

        Args:
            bo_iteration (int): The pass of the outer loop.
            acquisition_run: The ``AcquisitionRun`` the pass recorded, or None.

        Raises:
            ValueError: If the run recorded no generations at all.  Every inner run yields at
                least its initial population, so an empty sequence means the recording was not
                asked for, and a silently empty file would read as if the search had done nothing.
        """
        if acquisition_run is None:
            return
        if not acquisition_run.generations:
            msg = (
                f"pass {bo_iteration} recorded an acquisition run with no generations; every run "
                f"yields at least the zeroth, so this is a maximization that was not asked to "
                f"record its trajectory rather than one that had none"
            )
            raise ValueError(msg)
        for record in acquisition_run.generations:
            self._writer.writerow([
                bo_iteration,
                record.generation,
                record.best,
                record.population_best,
                record.population_mean,
                record.population_worst,
                record.distinct_members,
                record.last_improvement,
                record.offspring,
            ])
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class SurrogateLogger:
    """Writes what the surrogate of each pass believed, and how well.

    The acceptance checks read the surrogate once, at the end.  That answers whether the model was
    usable and not whether it became usable, and the second question is the one a plot of a run is
    about: a loop whose calibration improves as evaluations arrive is working, and one whose
    calibration stays where it started is over-confident from the first pass to the last.

    The held-out fit costs one extra Gaussian-process fit per pass, which is seconds at these
    dataset sizes against a pass that trains a neural network.
    """

    def __init__(self, path):
        self._file = open(path, "w", newline="")  # noqa: SIM115, closed by __exit__
        self._writer = csv.writer(self._file)
        self._writer.writerow(SURROGATE_CSV_COLUMNS)
        self._file.flush()

    def log(self, bo_iteration, optimizer):
        """Append one pass's surrogate reads.

        Args:
            bo_iteration (int): The pass of the outer loop.
            optimizer (BayesianOptimization): The loop, just after ``suggest()``.
        """
        surrogate = optimizer.surrogate
        if surrogate is None:
            return
        calibration = read_calibration(surrogate)

        # The same split by parity that the diagnostics use, over the data this pass saw.  The
        # cells stay empty where the split is not one a fit read is about:
        #
        # * fewer than two held-out terms, because a scatter of one point has no spread and no
        #   rank;
        # * a term on both sides, or twice on the conditioning side, because the surrogate would
        #   then have seen what it is asked to predict, and a noise-free Gaussian process
        #   reproduces its training values exactly, so the scatter would sit on the diagonal
        #   whatever the kernel does.  That is the one thing this read exists to rule out.
        #
        # The loop's rejection path makes the second case unreachable through ``suggest()``.  It is
        # guarded because a caller may seed the dataset with ``x0`` directly, and a run should not
        # die in its logger.
        snapshot = optimizer.get_state_snapshot()
        terms, values = snapshot["x_list"], snapshot["y_list"]
        conditioned, held_out = terms[0::2], terms[1::2]
        fit = None
        readable = (
            len(held_out) >= 2
            and conditioned
            and len(set(conditioned)) == len(conditioned)
            and not set(conditioned) & set(held_out)
        )
        if readable:
            fit = read_fit(
                optimizer.surrogate_over(conditioned, values[0::2]), held_out, values[1::2]
            )

        self._writer.writerow([
            bo_iteration,
            len(surrogate.X_train_),
            surrogate.log_marginal_likelihood_value_,
            calibration.root_mean_square,
            calibration.standard_deviation,
            calibration.maximum_absolute,
            calibration.outside_two,
            "" if fit is None else fit.size,
            "" if fit is None or fit.rank_correlation is None else fit.rank_correlation,
            "" if fit is None else fit.prediction_spread,
            "" if fit is None else fit.objective_spread,
            "" if fit is None else fit.residual_root_mean_square,
            json.dumps(kernel_hyperparameters(surrogate), sort_keys=True),
        ])
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def dataset_to_tensors(dataset, device, num_workers=0):
    """Materialize a whole split as one pair of tensors on ``device``.

    Flat-feature-vector convention: images become ``(N, C*H*W)``, matching the R^n -> R^m convention
    the repository and the algebras use everywhere else.  Labels stay class indices, long, for
    ``cross_entropy_loss``.  The split is loaded in a single batch, which keeps every candidate
    evaluation free of dataloader overhead, at the price of holding the whole split in the device's
    memory at once.

    Args:
        dataset: A dataset the caller has already constructed.  Nothing is downloaded here.
        device (torch.device): Where the tensors go.
        num_workers (int): Workers for the one-shot load.  0 avoids the multiprocessing overhead
            that buys nothing for a single batch. (Default value = 0)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The features and the labels.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=num_workers
    )
    images, labels = next(iter(loader))
    return images.reshape(images.shape[0], -1).to(device), labels.to(device).long()


def objective_direction(objective):
    """Turn the ``--objective`` choice into the two things the run needs from it.

    The loop maximizes, always.  A loss therefore enters it negated and comes back out negated, and
    every number read off the loop goes through the returned converter.  That the two scripts
    disagreed on one line of this, one reporting the loop's value and the other the metric's, is why
    it is one function now.

    Args:
        objective (str): ``"accuracy"`` (maximized) or ``"loss"`` (minimized).

    Returns:
        tuple[bool, Callable[[float], float]]: Whether greater is better, and the converter from
            what the loop reports back into the metric's own scale and sign.

    Raises:
        ValueError: If the objective is neither of the two.  The direction of a search is not
            something to guess a default for.
    """
    if objective not in ("accuracy", "loss"):
        msg = f"objective must be 'accuracy' or 'loss', got {objective!r}"
        raise ValueError(msg)
    greater_is_better = objective == "accuracy"

    def as_reported(value):
        """Turn a value the loop reports back into the metric's own scale and sign.

        Args:
            value (float): What the loop reported.

        Returns:
            float: The metric.
        """
        return float(value) if greater_is_better else -float(value)

    return greater_is_better, as_reported


#: The structure lengths the two length-named targets ask for.  Written out rather than imported
#: from the tooling that also states them: an example must not depend on the tooling, and a
#: dictionary of two entries is the cheaper duplicate.
TARGET_LENGTHS = {"L3": 3, "L5": 5}

#: The targets whose length is fixed by a position list rather than chosen.  A structure length is a
#: contradiction for every one of them, so they are refused together.  Each is built on a reference
#: architecture and carries its length in the positions it names.
POSITION_TARGETS = ("HEAD", "TUT1", "TUT2", "TUT3", "VGGM", "VGGN")


def resolve_target_choice(target, structure_length, default_length):
    """Turn ``--target`` and ``--structure-length`` into the one pair a run can be built from.

    The two arguments say the same thing in different vocabularies, since ``--target L3`` and
    ``--structure-length 3`` name one cell, and the position targets say something neither can:
    their length is fixed by their position list rather than chosen.  Resolving them in one place is
    what keeps the two experiment scripts from drifting apart on it.

    A contradiction raises rather than resolves.  Silently letting one argument win would produce
    a run whose CSV says one cell and whose target is another, and every number in it is plausible
    for the cell it claims to be.

    Args:
        target (str | None): ``"L3"``, ``"L5"``, ``"HEAD"``, or None to go by the length alone.
        structure_length (int | None): The explicit length, or None if it was not given.
        default_length (int): The script's default length, used when neither argument is given.

    Returns:
        tuple[str | None, int | None]: The cell label to record, and the structure length, which is
            None for a target whose length is its position list's.

    Raises:
        ValueError: If the two arguments contradict each other, or if a length is given alongside
            the head target, which has no length to set.
    """
    if target in POSITION_TARGETS:
        if structure_length is not None:
            msg = (
                f"--target {target} takes no --structure-length (got {structure_length}): this "
                f"target's length is fixed by its position list, which is what makes it more than "
                f"a sub-space of the length targets"
            )
            raise ValueError(msg)
        return target, None

    if target is not None:
        wanted = TARGET_LENGTHS[target]
        if structure_length is not None and structure_length != wanted:
            msg = (
                f"--target {target} is length {wanted}, but --structure-length says "
                f"{structure_length}; drop one of the two"
            )
            raise ValueError(msg)
        return target, wanted

    length = default_length if structure_length is None else structure_length
    # A free length that happens to be one of the two named ones is that cell and says so in the
    # record.  Any other length carries no cell label.
    label = next((name for name, size in TARGET_LENGTHS.items() if size == length), None)
    return label, length


def make_objective(x, y, x_val, y_val, batch_size, greater_is_better,
                   protocol=None, x_test=None, y_test=None, repeats=1, training_seeds=None):
    """Build the objective function and the store its measurements land in.

    Every candidate's full metric set is kept, keyed by its term: the loop's ``initialize()``
    evaluates the pre-samples internally and hands back objective values alone, so metrics measured
    during those evaluations have nowhere else to go.

    Args:
        x (torch.Tensor): Training features.
        y (torch.Tensor): Training labels.
        x_val (torch.Tensor): Validation features, which are what the loop is scored on.
        y_val (torch.Tensor): Validation labels.
        batch_size (int): Mini-batch size for training.
        protocol (TrainingProtocol | None): The training protocol, or None for the default.
        x_test (torch.Tensor | None): Held-out features, recorded but never optimized against.
        y_test (torch.Tensor | None): Their labels.
        greater_is_better (bool): Whether the loop maximizes the metric directly.
        repeats (int): How many trainings each candidate's value averages over, see
            :func:`evaluate_candidate`. (Default value = 1)
        training_seeds (Sequence[int] | None): One seed per repetition, or None for unseeded
            training. (Default value = None)

    Returns:
        tuple[Callable, dict]: The objective, and the measurement store it writes into.
    """
    metrics_by_tree = {}

    def f_obj(tree):
        """Train one candidate and return what the loop maximizes.

        Args:
            tree: The candidate term.

        Returns:
            float: The accuracy, or the negated loss, averaged over ``repeats`` trainings.
        """
        metrics = evaluate_candidate(tree, x, y, x_val, y_val, batch_size,
                                     protocol=protocol, x_test=x_test, y_test=y_test,
                                     repeats=repeats, training_seeds=training_seeds)
        metrics_by_tree[tree] = metrics
        # Both metrics always reach the CSV.  Only which one the loop maximizes changes.
        return metrics["accuracy"] if greater_is_better else -metrics["objective_value"]

    return f_obj, metrics_by_tree


def _draw_prefix(optimizer, count):
    """Take ``count`` pairwise distinct terms from one stream of the loop's own sampler.

    One stream and not ``count`` draws.  Calling ``sample()`` twice would start two streams from an
    advanced generator, and the second could repeat the first, which is exactly the property a
    paired baseline is built on.

    The distinctness is produced here, not inherited.  Every prefix of a size-uniform stream is a
    sample without replacement, so on that sampler the terms are distinct by construction.  The
    depth-bounded sampler says the opposite of itself: its stream is a sequence of independent
    draws, and independent draws may repeat a term.  A configuration whose determinization is
    unaffordable runs exactly that sampler, so the guarantee cannot be assumed for every run.

    The repeats are therefore rejected and counted.  Zero is the size-uniform case, where the
    guarantee holds.  A positive number is the repair where it does not, and it goes into the run's
    provenance rather than into a comment.

    Asked through ``optimizer.query`` rather than through a query of our own, so that the sampler's
    counting construction is the one the loop will use.  A second query object would build it again.

    Args:
        optimizer (BayesianOptimization): The configured loop, before ``initialize()``.
        count (int): How many terms to take.

    Returns:
        tuple[list, int]: The terms in stream order, and how many repeats were rejected.

    Raises:
        RuntimeError: If the loop has no search space to draw from, or if the stream cannot supply
            ``count`` distinct terms.  A short design is not a design, and filling it up with
            repeats would make the baseline evaluate the same network twice.
    """
    query = optimizer.query
    if query is None or optimizer.sampler is None:
        msg = (
            "a paired baseline draws its terms from the loop's own sampler, and this loop has "
            "no search space or no sampler to draw them from"
        )
        raise RuntimeError(msg)
    return distinct_prefix(optimizer.sampler, query, count)


def _check_resumed_design(resume_design, drawn_prefix):
    """Refuse a resumed design whose terms are not the ones this run would have drawn.

    The resumed values describe those terms.  If the sampler now produces different ones, after a
    changed seed or a changed cell or with a sampler whose order is not reproducible, then pairing
    value ``i`` with term ``i`` pairs a measurement with a network it was not taken from, and every
    number after that is about nothing.  With a fixed seed the two streams agree, so this normally
    passes.  It exists for when it does not.

    Args:
        resume_design (list[tuple]): ``(term, metrics)`` as loaded.
        drawn_prefix (list): The terms this run drew.

    Raises:
        ValueError: On a different length, or on the first term that differs.
    """
    if len(resume_design) != len(drawn_prefix):
        msg = (
            f"the resumed design holds {len(resume_design)} terms but this run draws "
            f"{len(drawn_prefix)}; the size of the initial design must match the run being "
            f"continued"
        )
        raise ValueError(msg)
    for index, ((resumed, _metrics), drawn) in enumerate(zip(resume_design, drawn_prefix, strict=True)):
        if resumed != drawn:
            msg = (
                f"resumed term {index} is not the term this run drew at that position; the "
                f"sampler is producing a different stream, so the loaded values describe other "
                f"networks than the ones being paired with them"
            )
            raise ValueError(msg)


def load_initial_design(path, expected, phase="pre_sample"):
    """Take a finished initial design out of an interrupted run's term pool.

    A run that dies after its initial design has spent hours of accelerator time on trainings whose
    results are complete and on disk, and restarting it repeats every one of them to arrive at the
    same numbers.  The terms and their metrics are in ``<run>_terms.pickle``, and this reads them
    back so that the next run can start where the last one got to.

    The provenance is checked, not assumed.  A design measured under different epochs, a different
    cell or a different number of repetitions is not this run's design, and silently conditioning a
    surrogate on it would produce a run whose dataset nobody can describe.  The fields compared are
    the ones that change what a number means, and everything else may differ.

    Args:
        path (str): The ``<run>_terms.pickle`` of the interrupted run.
        expected (dict): The fields the current run requires to match.
        phase (str): Which phase's records to take. (Default value = "pre_sample")

    Returns:
        list[tuple]: ``(term, metrics)`` in the order they were measured.

    Raises:
        ValueError: If the pool carries no matching provenance, or none of the requested phase.
    """
    header, records = read_term_pool(path)
    provenance = header.get("provenance") or {}
    mismatched = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatched:
        detail = ", ".join(f"{k}: pool has {p!r}, run wants {w!r}" for k, (p, w) in mismatched.items())
        msg = (
            f"{path} was measured under a different configuration and its values do not describe "
            f"this run's candidates ({detail})"
        )
        raise ValueError(msg)

    design = [(record.term, record.metrics) for record in records if record.phase == phase]
    if not design:
        phases = sorted({record.phase for record in records})
        msg = f"{path} holds no {phase!r} records; it has {phases}"
        raise ValueError(msg)
    return design


def run_ask_tell_search(
    *,
    optimizer,
    f_obj,
    metrics_by_tree,
    as_reported,
    objective,
    greater_is_better,
    n_pre_samples,
    n_iterations,
    csv_path,
    pretty_algebra,
    provenance=None,
    baseline=False,
    verbose=True,
    acquisition_hard_limit=ACQUISITION_HARD_LIMIT_SECONDS,
    resume_design=None,
):
    """Run the loop through the ask/tell interface, logging every evaluation as it happens.

    Ask and tell rather than the closed ``optimize()``.  Every evaluated structure, pre-samples and
    loop passes alike, is written and flushed as it is produced, so results survive an interruption
    instead of existing only after a successful ``finalize()``.  That is the whole reason this is
    not four lines.

    Args:
        optimizer (BayesianOptimization): The configured loop.
        f_obj (Callable): The objective, as :func:`make_objective` returns it.
        metrics_by_tree (dict): Its measurement store.
        as_reported (Callable[[float], float]): The converter from :func:`objective_direction`.
        objective (str): The metric's name, for the printed summary.
        greater_is_better (bool): Whether the loop maximizes it directly.
        n_pre_samples (int): How many candidates the initializer draws and evaluates.
        n_iterations (int): The budget of loop passes.
        csv_path (str): Where the per-evaluation rows go.
        pretty_algebra (Callable): The algebra that renders a term for the CSV.
        provenance (dict): The run's provenance record, written into the header of
            ``<run>_terms.pickle``, so that the term file identifies its own origin when it travels
            without the JSON beside it, which is what happens when a pool is trained on one machine
            and analyzed on another. (Default value = None)
        baseline (bool): Also run a random search of the same budget from the same initial design.
            Whether the loop beats drawing at random is a question about the passes only if both
            start from the same place, so the two share their initial evaluations rather than each
            drawing their own, which is also why this costs one extra training per pass and not one
            per evaluation. (Default value = False)
        verbose (bool): Passed to ``suggest``. (Default value = True)
        acquisition_hard_limit (float): Seconds after which one acquisition maximization is given up
            on.  It bounds a legitimate duration, so it belongs to the cell being searched rather
            than to the code, and a large architecture at a large population can outlast the
            default. (Default value = ACQUISITION_HARD_LIMIT_SECONDS)

    Returns:
        tuple[dict, float, dict]: The loop's result, its wall-clock duration, and the summary to
            merge into the provenance record.
    """
    print(f"Starting Bayesian Optimization: n_pre_samples={n_pre_samples}, "
          f"n_iterations={n_iterations}")
    print(f"Logging every evaluated structure to {csv_path}")

    started = time.time()
    # The paired baseline needs its terms before the loop starts, because the ones the loop
    # initializes on have to be the same objects.  One stream of as many draws as the two runs
    # together evaluate: on the size-uniform sampler every prefix of that stream is a sample without
    # replacement, so its first entries are a valid initial design on their own, and the two runs
    # share a starting point instead of each getting their own.
    drawn = None
    repeats_rejected = 0
    if baseline:
        with step_budget(
            f"drawing {n_pre_samples + n_iterations} terms for the paired baseline",
            ACQUISITION_WARN_SECONDS,
        ):
            drawn, repeats_rejected = _draw_prefix(
                optimizer, n_pre_samples + n_iterations
            )
            if repeats_rejected:
                print(
                    f"the sampler repeated {repeats_rejected} term(s) while drawing "
                    f"{n_pre_samples + n_iterations} for the paired design; they were rejected.  "
                    f"On the size-uniform sampler this number is 0, since every prefix of its "
                    f"stream is a sample without replacement",
                    flush=True,
                )

    with (
        EvaluationLogger(csv_path, pretty_algebra, provenance=provenance) as logger,
        EAGenerationLogger(_sibling_path(csv_path, "_ea.csv")) as ea_logger,
        SurrogateLogger(_sibling_path(csv_path, "_surrogate.csv")) as surrogate_logger,
    ):
        # The initial design is written as it is measured, not after it is complete.
        #
        # The whole reason this function is not four lines is that results survive an interruption,
        # and that used to hold for the loop passes and not for the initial design: the pre-samples
        # were evaluated into a list and logged afterwards, so a run interrupted during
        # initialization left an empty CSV however many networks it had trained.  With repeated
        # measurements that window is hours rather than minutes, and hours in which nothing on disk
        # says the run is producing anything at all is also the difference between "still running"
        # and "stuck" seen from outside.
        #
        # Only the paired path can do this, because there the terms are known before the loop
        # starts and can be measured one at a time in a known order.  Without a baseline the loop
        # draws and evaluates them internally and there is nothing to interleave with, so that path
        # keeps the older behavior and says so below rather than pretending otherwise.
        with step_budget(
            f"initialization ({n_pre_samples} networks)",
            n_pre_samples * PER_EVALUATION_WARN_SECONDS,
        ):
            if drawn is None:
                optimizer.initialize(objective=f_obj, initial_size=n_pre_samples)
                logged_prefix = None
            else:
                logged_prefix = drawn[:n_pre_samples]
                if resume_design is not None:
                    _check_resumed_design(resume_design, logged_prefix)
                values = []
                for idx, tree in enumerate(logged_prefix):
                    if resume_design is not None:
                        # Measured already, in the run this one continues.  The value is taken
                        # rather than measured again, which is the whole point, and it goes into
                        # ``metrics_by_tree`` so that nothing downstream can tell the difference.
                        metrics = resume_design[idx][1]
                        metrics_by_tree[tree] = metrics
                        values.append(
                            metrics["accuracy"] if greater_is_better
                            else -metrics["objective_value"]
                        )
                        source = " (resumed)"
                    else:
                        values.append(f_obj(tree))
                        metrics = metrics_by_tree[tree]
                        source = ""
                    logger.log("pre_sample", idx, tree, metrics)
                    print(f"  pre_sample[{idx}]: objective={as_reported(values[-1]):.5f} "
                          f"accuracy={metrics['accuracy']:.4f} "
                          f"params={metrics['n_params']} "
                          f"train={metrics['train_seconds']:.1f}s{source}", flush=True)
                optimizer.initialize(x0=logged_prefix, y0=values)

        snapshot = optimizer.get_state_snapshot()
        for idx, (tree, value) in enumerate(
            zip(snapshot["x_list"], snapshot["y_list"], strict=True)
        ):
            # No substitute for a missing measurement: every pre-sample went through the
            # objective, so a term without metrics means the loop handed back one it did not
            # evaluate, and that is worth stopping for rather than filling in.  Checked on both
            # paths, because the check is about what the loop reports and not about who wrote the
            # row.
            if tree not in metrics_by_tree:
                msg = (
                    f"pre-sample {idx} was never evaluated by the objective, yet the loop reports "
                    f"a value for it: {tree}"
                )
                raise KeyError(msg)
            if logged_prefix is not None:
                continue  # already written above, while it was measured
            metrics = metrics_by_tree[tree]
            logger.log("pre_sample", idx, tree, metrics)
            print(f"  pre_sample[{idx}]: objective={as_reported(value):.5f} "
                  f"accuracy={metrics['accuracy']:.4f} "
                  f"params={metrics['n_params']} "
                  f"train={metrics['train_seconds']:.1f}s", flush=True)

        for step in range(n_iterations):
            acquisition_started = time.time()
            with step_budget(
                f"BO step {step}: acquisition optimization",
                ACQUISITION_WARN_SECONDS,
                hard_limit_seconds=acquisition_hard_limit,
            ):
                # ``record_population``: the frontier read of the acceptance checks is a statement
                # about one maximization, and it cannot be rebuilt afterwards.  A surrogate refitted
                # on the whole dataset scores the same population differently, so members that lost
                # to the pick at the time can beat it later.  It has to be the population as it was
                # scored, and it is also what carries the per-generation records of the inner
                # search.
                suggestion = optimizer.suggest(verbose=verbose, record_population=True)
            acquisition_seconds = time.time() - acquisition_started
            # Written before the network trains: the inner search is finished at this point, and a
            # run interrupted during the training then still holds the generation it produced.
            ea_logger.log(step, optimizer.last_acquisition_run)
            surrogate_logger.log(step, optimizer)
            with step_budget(
                f"BO step {step}: training the suggested network",
                PER_EVALUATION_WARN_SECONDS,
            ):
                # Not named ``y``: ``f_obj`` closes over the label tensor of that name, and
                # rebinding it here would clobber it for every later call.
                objective_value = f_obj(suggestion.candidate)
            optimizer.observe(suggestion.candidate, objective_value)
            iteration = suggestion.diagnostics["iteration"] if suggestion.diagnostics else None
            metrics = metrics_by_tree[suggestion.candidate]
            logger.log("bo_step", iteration, suggestion.candidate, metrics, suggestion=suggestion,
                       acquisition_seconds=acquisition_seconds)
            # The fallback flag decides how this line reads: after a fallback the candidate is a
            # random sample rather than the acquisition optimizer's choice.  Seeing it live is the
            # difference between noticing a degraded run and reading it out of the CSV a day later.
            fallback = (suggestion.diagnostics or {}).get("fallback_used", "?")
            print(f"  bo_step[{iteration}]: objective={as_reported(objective_value):.5f} "
                  f"accuracy={metrics['accuracy']:.4f} params={metrics['n_params']} "
                  f"train={metrics['train_seconds']:.1f}s fallback={fallback}", flush=True)

        result = optimizer.finalize()
        bo_time = time.time() - started

        # The baseline's own evaluations, after the loop's, so that an interruption here leaves
        # the Bayesian run complete.  These terms come from the same stream as the shared prefix,
        # so together with it they are the draws a random search of this budget makes.
        baseline_rows = []
        if drawn is not None:
            for offset, tree in enumerate(drawn[n_pre_samples:]):
                with step_budget(
                    f"baseline sample {offset}: training the drawn network",
                    PER_EVALUATION_WARN_SECONDS,
                ):
                    value = f_obj(tree)
                metrics = metrics_by_tree[tree]
                logger.log("random_sample", offset, tree, metrics)
                baseline_rows.append(value)
                print(f"  random_sample[{offset}]: objective={as_reported(value):.5f} "
                      f"accuracy={metrics['accuracy']:.4f} params={metrics['n_params']} "
                      f"train={metrics['train_seconds']:.1f}s", flush=True)

    n_evaluations = n_pre_samples + n_iterations
    # The same rule as at the pre-samples: a term the loop reports must be one it was given a
    # value for, and a missing measurement is a fact rather than an empty dict.
    if result["best_tree"] not in metrics_by_tree:
        msg = f"the loop returned a best term that was never evaluated: {result['best_tree']}"
        raise KeyError(msg)
    best_metrics = metrics_by_tree[result["best_tree"]]
    print(f"Bayesian Optimization took {bo_time:.2f}s ({n_evaluations} network evaluations, "
          f"{bo_time / max(n_evaluations, 1):.2f}s per evaluation on average)")
    print(f"Best {objective} ({'max' if greater_is_better else 'min'}): "
          f"{as_reported(result['best_y']):.5f}")
    if "accuracy" in best_metrics:
        print(f"Best tree's validation accuracy: {best_metrics['accuracy'] * 100:.2f}%")
    if "objective_value" in best_metrics:
        print(f"Best tree's validation loss: {best_metrics['objective_value']:.5f}")
    print(f"Best tree:\n{result['best_tree'].interpret(pretty_algebra())}")

    summary = {
        "bayesian_optimization_seconds": bo_time,
        "n_evaluations": n_evaluations,
        "mean_seconds_per_evaluation": bo_time / max(n_evaluations, 1),
        "total_training_seconds": sum(m["train_seconds"] for m in metrics_by_tree.values()),
        # In the metric's own scale and sign, like every other number read out of the loop.  The
        # value the loop itself maximized stands beside it, named for what it is, because on a loss
        # objective the two differ and a record that shows only one is ambiguous.
        "best_objective_value": as_reported(result["best_y"]),
        "best_maximized_value": float(result["best_y"]),
        "best_accuracy": best_metrics.get("accuracy"),
        # Read off the budget rather than taken as a flag.  A loop with no passes evaluates its
        # initial dataset and stops, and that dataset is a size-uniform sample, so such a run is
        # random search over the same space, at the same budget, through the same code.  A separate
        # switch could disagree with the numbers, and this cannot.
        "run_kind": (
            "random_search_baseline" if n_iterations == 0
            else "bayesian_optimization_with_paired_baseline" if baseline
            else "bayesian_optimization"
        ),
        # How many repeats the sampler produced while the design was drawn, and how many the loop
        # itself had to redraw.  Zero on the size-uniform sampler, where every prefix is a sample
        # without replacement.  On the depth-bounded sampler the draws are independent and may
        # repeat, so the number says how much work the rejection had to do.  It is in the record
        # because a run whose initial design needed repair is a run whose sampler does not supply
        # what its dataset needs.
        "initial_repeats_rejected": (
            repeats_rejected if drawn is not None else optimizer.initial_repeats_rejected
        ),
        "completed": True,
    }
    if drawn is not None:
        # The two curves this run produces, at the same budget and from the same initial points.
        # Both in the metric's own scale and sign, like every other number here.
        shared = [optimizer.get_state_snapshot()["y_list"][index] for index in range(n_pre_samples)]
        summary["baseline_best_objective_value"] = as_reported(max([*shared, *baseline_rows]))
        summary["shared_initial_best_objective_value"] = as_reported(max(shared))
        summary["baseline_evaluations"] = n_pre_samples + n_iterations
        summary["shared_initial_evaluations"] = n_pre_samples
    return result, bo_time, summary


def metadata_path_for(csv_path):
    """Return the provenance path beside a run's CSV: ``run.csv`` becomes ``run_config.json``."""
    base, _ext = os.path.splitext(csv_path)
    return f"{base}_config.json"


def _sibling_path(csv_path, suffix):
    """Return a path beside a run's CSV: ``run.csv`` becomes ``run<suffix>``."""
    base, _ext = os.path.splitext(csv_path)
    return f"{base}{suffix}"


def write_run_diagnostics(csv_path, optimizer, result):
    """Run the acceptance checks over the finished run and write them out.

    A run that produced numbers has not thereby produced readable numbers, and these reads are what
    separates the two.  They are ordered by what they need, the kernel first, then the mean, then
    the uncertainty, then the inner evolutionary run, and a failure found early makes the later ones
    uninformative rather than wrong: an uninformative kernel gives a flat fit scatter, flat
    residuals and a flat acquisition landscape, and only the first of them says why.

    Every field here is a measurement and none is a verdict.  Each read names a failure mode without
    the threshold that separates it, because a threshold belongs to a search space and not to the
    method, and eleven terms of one chain and four hundred convolutional architectures do not share
    one.  So this writes the numbers and leaves the reading to whoever compares two runs.

    Two artifacts: ``<run>_diagnostics.json`` with the five reads, and ``<run>_trace.csv`` with the
    per-pass table behind the fifth.

    Called after the CSV and the metadata are on disk, so that a read which raises, a surrogate that
    will not fit for instance, costs the diagnostics and not the run.

    Each read states what it needs, and a read whose input is not there is written as ``null``.
    That is not a substitute value and not a swallowed failure: a smoke test of three evaluations
    has one held-out term, and the fit read is a statement about a scatter that needs two, so "there
    was not enough of a run to read this" is the true answer and it is what the file says.  A read
    that fails on data it was given still raises.

    Args:
        csv_path (str): The run's CSV.  The artifacts are written beside it.
        optimizer (BayesianOptimization): The loop, after ``finalize()``.
        result (dict): What ``finalize()`` returned.

    Returns:
        dict: The diagnostics, as they were written.
    """
    terms = list(result["x"])
    values = [float(value) for value in result["y"]]

    # --- 1. The kernel, before any conditioning ----------------------------------------------
    # ``optimizer.kernel`` and not the fitted one.  A kernel matrix betrays a broken kernel before
    # any Gaussian process is conditioned, and that reading is about the kernel as it was
    # constructed, before a fit chose its scales.
    gram = read_gram(optimizer.kernel(terms), objective=values) if len(terms) >= 2 else None

    # --- 2. The mean, on terms the surrogate has not seen -------------------------------------
    # Condition on the terms of even index and predict the odd ones.  Splitting by parity keeps the
    # conditioning half spread over the whole run instead of over one region of it, which would
    # measure extrapolation, a different and harder question.
    conditioned_terms, conditioned_values = terms[0::2], values[0::2]
    held_out_terms, held_out_values = terms[1::2], values[1::2]
    # The same three conditions ``SurrogateLogger`` states: two held-out terms to make a scatter, a
    # conditioning half without repeats, and no term on both sides.  A surrogate that has seen what
    # it predicts reproduces it, and the read would report a diagonal it did not earn.
    readable = (
        len(held_out_terms) >= 2
        and conditioned_terms
        and len(set(conditioned_terms)) == len(conditioned_terms)
        and not set(conditioned_terms) & set(held_out_terms)
    )
    fit = (
        read_fit(
            optimizer.surrogate_over(conditioned_terms, conditioned_values),
            held_out_terms,
            held_out_values,
        )
        if readable
        else None
    )

    # --- 3. The uncertainty, over the whole dataset -------------------------------------------
    # Not ``result["gp_model"]``, which is the surrogate of the last ``suggest()`` and never saw the
    # evaluation the run ended on.  A calibration is a statement about the data that was collected.
    calibration = read_calibration(optimizer.surrogate_over_dataset()) if terms else None

    # --- 4. The inner evolutionary run --------------------------------------------------------
    run = optimizer.last_acquisition_run
    frontier = None if run is None else read_frontier(run)

    # --- 5. The loop itself -------------------------------------------------------------------
    trace = read_trace(result["trace"]) if result["trace"] else None

    diagnostics = {
        # null where the run was too short for the read, never a filled-in number.
        "gram": None if gram is None else {
            "size": gram.size,
            "symmetry_error": gram.symmetry_error,
            "minimum_eigenvalue": gram.minimum_eigenvalue,
            "condition_number": gram.condition_number,
            "diagonal_spread": gram.diagonal_spread,
            "off_diagonal_mean": gram.off_diagonal_mean,
            "off_diagonal_spread": gram.off_diagonal_spread,
            "seriation_neighbour_similarity": gram.seriation_neighbour_similarity,
            "coordinate_spread": gram.coordinate_spread,
            # The kernel's own coordinate against the objective.  A low value here is a statement
            # about this kernel on these terms and not a defect by itself, since the leading
            # principal component of a kernel need not be the direction the objective varies along.
            "objective_alignment": gram.objective_alignment,
        },
        "fit": None if fit is None else {
            "size": fit.size,
            "rank_correlation": fit.rank_correlation,
            "predictions_constant": fit.predictions_constant,
            "prediction_spread": fit.prediction_spread,
            "objective_spread": fit.objective_spread,
            "regression_slope": fit.regression_slope,
            "residual_root_mean_square": fit.residual_root_mean_square,
            "true_values": list(fit.true_values),
            "predicted_values": list(fit.predicted_values),
        },
        "calibration": None if calibration is None else {
            "size": calibration.size,
            "mean_absolute": calibration.mean_absolute,
            "maximum_absolute": calibration.maximum_absolute,
            # The spread about zero and the spread about their own mean, side by side.  The
            # difference between the two is bias, and residuals that carry a single sign with a
            # small spread are the signature of a surrogate whose deviations stayed at the prior.
            # Reading only one of the two hides it.
            "root_mean_square": calibration.root_mean_square,
            "standard_deviation": calibration.standard_deviation,
            "outside_two": calibration.outside_two,
            "positive_fraction": calibration.positive_fraction,
            "log_marginal_likelihood": calibration.log_marginal_likelihood,
            "targets_normalized": calibration.targets_normalized,
            "standardized_residuals": list(calibration.standardized_residuals),
        },
        # None rather than an empty record.  That no acquisition maximization was recorded is a
        # different statement from one that was recorded and held nothing.
        "frontier": None if frontier is None else {
            "size": frontier.size,
            "distinct_members": frontier.distinct_members,
            "pick_in_population": frontier.pick_in_population,
            "known_members": frontier.known_members,
            "mean_at_pick": frontier.mean_at_pick,
            "deviation_at_pick": frontier.deviation_at_pick,
            "score_at_pick": frontier.score_at_pick,
            "fallback_used": frontier.fallback_used,
            "dominating_members": frontier.dominating_members,
            "on_frontier": frontier.on_frontier,
            "higher_scored_members": frontier.higher_scored_members,
            "mean_spread": frontier.mean_spread,
            "deviation_spread": frontier.deviation_spread,
        },
        "trace": None if trace is None else {
            "passes": trace.passes,
            "acquisition_trend": trace.acquisition_trend,
            "acquisition_first": trace.acquisition_first,
            "acquisition_last": trace.acquisition_last,
            "acquisition_at_zero": trace.acquisition_at_zero,
            "deviation_spread": trace.deviation_spread,
            "deviation_minimum": trace.deviation_minimum,
            "deviation_maximum": trace.deviation_maximum,
            "fallbacks": trace.fallbacks,
            "distinct_fraction": trace.distinct_fraction,
            "improvements": trace.improvements,
            "stalled_passes": trace.stalled_passes,
            "best_trace": list(trace.best_trace),
        },
        # The reads carry no thresholds, and neither does this file.  What the fields mean is in
        # ``bayesian_optimization/diagnostics``, one remark per read.
        "read_this_with": "bayesian_optimization.diagnostics",
    }

    path = _sibling_path(csv_path, "_diagnostics.json")
    with open(path, "w") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True, default=float)
    print(f"Acceptance checks written to {path}", flush=True)

    if result["trace"]:
        trace_path = _sibling_path(csv_path, "_trace.csv")
        with open(trace_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(trace_columns())
            writer.writerows(trace_rows(result["trace"]))
        print(f"Run trace written to {trace_path}", flush=True)

    return diagnostics


def write_run_metadata(csv_path, metadata):
    """Write the run's provenance next to its CSV and return the path used."""
    path = metadata_path_for(csv_path)
    enriched = {
        **metadata,
        "csv_path": csv_path,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "written_at": time.time(),
    }
    with open(path, "w") as f:
        json.dump(enriched, f, indent=2, sort_keys=True)
    return path
