"""The program a run searches, the sampler that fits it, and the record of both.

Nothing here depends on one search space.  The check that a repository's abstraction covers the
alphabet of the program it synthesizes is the caller's, see :func:`build_search`.
"""

import copy
import random
import time
from dataclasses import dataclass
from typing import Any

from cosy.core import Synthesizer
from cosy.evolutionary_algorithms import EvolutionarySearch
from cosy.search import DepthBoundedRandomSampler, SizeUniformSampler, generator_query
from cosy.search.determinize import determinize

from bayesian_optimization.runs.budgets import step_budget

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

    def twin(self, rng):
        """A sampler of the same program that draws its stream from ``rng`` and shares the table.

        The counting table depends on the program and the bound alone, and it is the expensive
        part: minutes on a large program.  A second sampler of the same program, a paired
        comparison's other arm, takes it over instead of counting again.  It is built here first
        where it has not been, since a table copied unbuilt would be built by each sampler.  The
        inner sampler is always queried with this sampler's own query, so the two share the
        cached table whatever query their loops hand them.

        Args:
            rng (random.Random): The twin's source of randomness.

        Returns:
            DeterminizedSizeUniformSampler: The twin.
        """
        self._inner.at_least(self._query, 1)
        twin = copy.copy(self)
        twin._inner = copy.copy(self._inner)
        twin._inner.rng = rng
        return twin


def twin_sampler(sampler, query, rng):
    """A sampler that draws the stream ``sampler`` draws, from ``rng``, sharing its table.

    What a paired comparison's second arm needs: the stream the first arm's design was drawn from,
    past the design, without counting the program again.  Given a random source seeded as the
    first sampler's was, the twin draws the same terms in the same order.

    A sampler that counts, cosy's ``SizeUniformSampler``, caches its table per query object: the
    table is built here for ``query`` before it is shared, and a twin drawing for another query
    object counts again.  :class:`DeterminizedSizeUniformSampler` queries its own program and
    shares its table whatever query it is handed.  A sampler that does not count has nothing to
    share and is copied with the new random source.

    Args:
        sampler: The sampler to twin.
        query: The query it draws for.
        rng (random.Random): The twin's source of randomness.

    Returns:
        A sampler of the same kind.

    Raises:
        TypeError: If the sampler names no random source a twin could draw from.
    """
    if isinstance(sampler, DeterminizedSizeUniformSampler):
        return sampler.twin(rng)
    if not hasattr(sampler, "rng"):
        msg = f"{type(sampler).__name__} names no rng, so there is no stream to draw a twin of"
        raise TypeError(msg)
    if isinstance(sampler, SizeUniformSampler):
        sampler.at_least(query, 1)
    twin = copy.copy(sampler)
    twin.rng = rng
    return twin


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
    check_alphabet,
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
        check_alphabet (Callable[[Any], dict]): The check that the repository's abstraction
            covers the alphabet of the synthesized program, as ``make_alphabet_check`` of
            :mod:`bayesian_optimization.examples.recognizable_swap_laws` builds it.  It is
            called on the program before the determinization, and the ``terminals`` it reports
            go into the provenance, sorted.  Required and without a default, because a search that
            skipped the check would determinize a program its abstraction may not cover, and
            nothing would say so.  ``sampling="depth-bounded"`` does not call it.
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
            # Sorted here rather than trusted sorted: the record is compared across runs.
            "terminals": sorted(alphabet["terminals"]),
        },
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
