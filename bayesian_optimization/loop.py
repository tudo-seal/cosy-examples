"""The ask/tell loop every search strategy of this package runs: its states, its dataset, its design.

A strategy decides which term to evaluate next; everything around that decision is the same for
every strategy and lives here once.  :class:`AskTellLoop` owns the state machine (``BOState``), the
dataset of evaluated terms and their values, the refusal of a value that is not finite, the design
phase in which the initial design is handed out term by term, and the result a run finalizes into.
What a strategy adds is one method, the proposal of a pass, and optionally a check of a pass before
it enters the dataset and a record of it after.  :class:`~bayesian_optimization.bo.BayesianOptimization`
is the strategy that proposes by maximizing an acquisition over a Gaussian process surrogate.

It was factored out of ``BayesianOptimization`` so that a second strategy runs the *same* loop
rather than a copy of it: two copies are two chances to drift apart.
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar

import numpy as np
from cosy.core.solution_space import SolutionSpace
from cosy.evolutionary_algorithms import Initializer, SampledInitialization
from cosy.search import Sampler, SizeUniformSampler, generator_query

from .diagnostics import enable_verbose_logging, get_logger, log_suggestion
from .initial_sampling import _sample_fallback_tree
from .state import BOState, Diagnostics, Suggestion

NT = TypeVar("NT", bound=Hashable)
T = TypeVar("T", bound=Hashable)
G = TypeVar("G", bound=Hashable)

# The bound of the sampler built when the caller names none.  A placeholder that fits toy
# spaces.  See the ``sampler`` parameter of ``BayesianOptimization`` for why a real space needs its own.
DEFAULT_SIZE_BOUND = 100


def _finite_or_raise(value: Any, candidate: Any) -> float:
    """Return ``value`` as a float, refusing anything that is not finite.

    The objective is modeled as a total function on the terms, so the theory has no notion of a
    failed evaluation, and this framework does not invent one.  A network that fails to train, an
    interpretation that fails, a measurement that returns nothing: none of them get a substitute
    number.  The failure propagates out of the loop, visibly, at the pass that caused it.

    ``nan`` is the one worth naming.  It survives the fit, comes back out of the posterior at every
    term, ties every acquisition value, and leaves a run that has stopped optimizing while still
    producing suggestions.

    Args:
        value (Any): The observed value.
        candidate (Any): The term it was observed at, for the message.

    Returns:
        float: The value.

    Raises:
        ValueError: If the value is not a finite real number.
    """
    observed = float(value)
    if not math.isfinite(observed):
        msg = (
            f"the objective returned {observed} at {candidate}, and a value that is not finite "
            f"cannot be observed: it makes the posterior undefined everywhere rather than at one "
            f"term.  A failed evaluation is a failure, not a number, so let it propagate."
        )
        raise ValueError(msg)
    return observed


def _require_hashable(candidates: Sequence[Any], source: str) -> None:
    """Refuse an initial design that holds a term which cannot be hashed.

    The dataset is a set of terms, and every membership test this loop runs on it goes through a
    hash: the fallback that replaces a repeat, the set the suggestion path tests a candidate
    against, and the dictionary the pairs are checked for contradictions in.  A design that
    cannot be hashed cannot become that dataset, and it is refused here rather than at the first
    of those tests, which reaches the caller as a bare message about a type.

    Asked of both designs this class accepts, and before either is paid for.  A drawn design is
    deduplicated against a set on the way in, and a supplied one may cost an evaluation of the
    quality measure per term, so a design refused afterwards is a design refused too late.

    Args:
        candidates (Sequence[Any]): The design to check.
        source (str): Where the design came from, for the message.

    Raises:
        TypeError: If a candidate cannot be hashed.
    """
    for item in candidates:
        try:
            hash(item)
        except TypeError as unhashable:
            msg = (
                f"All candidates in {source} must be hashable.  Got an unhashable item of type "
                f"{type(item).__name__!r}."
            )
            raise TypeError(msg) from unhashable


def _distinct_dataset(
    drawn: Sequence[Any], sampler: Sampler, query: Any, count: int
) -> tuple[list[Any], int]:
    """Make an initial design pairwise distinct, keeping what the initializer chose.

    Deduplicates in place rather than redrawing the whole design, because the initializer's terms
    are the initializer's answer.  The kernel-diverse initializer draws each member biased away
    from the members already drawn, so throwing the set away to draw a fresh one would silently
    replace the informed design with the model-agnostic one.  Only the places a repeat occupied
    are filled again.

    Structural comparison here (``Tree.__eq__``), while the fallback rejects a repeat with a set.
    The two answer alike as long as every label hashes consistently with its equality, compares
    symmetrically, and does not change after its term is built, so the scan is a check on the
    fallback rather than a second answer about identity.  See
    :func:`~bayesian_optimization.initial_sampling.distinct_prefix`.

    The set the fallback tests against is one set, carried through the draws.  A term takes its
    hash in its constructor and keeps it, so a set built again over the same terms holds the same
    terms under the same hashes and answers the same question, and one built per replacement asks
    that question of the whole design again for each of them.  Holding one means the design has to
    be hashable whether or not a repeat occurs, which is why
    :meth:`AskTellLoop.initialize` establishes that first, in its own words rather than in the
    interpreter's.

    Args:
        drawn (Sequence[Any]): What the initializer returned.
        sampler (Sampler): The loop's sampler, for the replacements.
        query (Any): The generator query.
        count (int): The design size that was asked for.

    Returns:
        tuple[list[Any], int]: The distinct design, and how many repeats were replaced.

    Raises:
        RuntimeError: If a replacement cannot be drawn.  ``_sample_fallback_tree`` raises on an
            exhausted space, and a design topped up with repeats is what this prevents.  Also if a
            drawn replacement turns out to equal a term the fallback's set had passed it against,
            which takes a label that hashes against its own equality, compares asymmetrically, or
            changed after its term was built.
        TypeError: If a term cannot be hashed.  The design is held as a set here, so a caller
            that has not checked its own design meets the interpreter's message rather than one
            of its own.
    """
    kept: list[Any] = []
    seen: set[Any] = set()
    repeats = 0
    for candidate in drawn:
        if any(candidate == other for other in kept):
            repeats += 1
        else:
            kept.append(candidate)
            seen.add(candidate)
    while len(kept) < count:
        replacement = _sample_fallback_tree(sampler, query, seen)
        if any(replacement == other for other in kept):
            msg = (
                "the fallback returned a term already in the initial design.  It rejects a "
                "repeat with a set, which asks kept == candidate, while this scan asks "
                "candidate == kept, and a term hashes once when it is built.  So a label in one "
                "of these terms hashes against its own equality, compares asymmetrically, or "
                "changed after its term was built"
            )
            raise RuntimeError(msg)
        kept.append(replacement)
        seen.add(replacement)
    return kept, repeats


def _check_request_against_space(search_space: Any, request: Any) -> None:
    """Refuse a request the search space has no rules for, before a query is built from the pair.

    A space queried at a non-terminal it has no rules for answers with the empty stream, and each
    consumer downstream then reports what it can see, which is the sampler.  The initial design
    and the evolutionary run both say that fewer inhabitants than were asked for lie inside the
    sampler's bound and offer to widen it, the duplicate fallback says the bounded space is
    exhausted, and the ``query`` property says nothing at all and hands back a query whose every
    draw is empty.  Widening the bound is the one repair those messages name, and it is the one
    that cannot help.

    Membership, not a test against ``None``.  A non-terminal of some other space, a mistyped
    target for instance, fails through those same messages, and a non-terminal is anything
    hashable, so a space whose non-terminal is ``None`` is queried at ``None`` and draws terms.
    The test is a lookup in the space's own rule table, ``SolutionSpace.__contains__``.

    Args:
        search_space (Any): The space, or None where there is none to query.
        request (Any): The non-terminal the space would be queried at.

    Raises:
        ValueError: If the space has no rules for the request.
    """
    if search_space is None or request in search_space:
        return
    msg = (
        f"the search space has no rules for the request {request}, so nothing can be drawn from "
        "it.  The request is the non-terminal the space is queried at, and every term the loop "
        "draws is an inhabitant of it: pass the non-terminal the space was synthesized for.  The "
        "default None belongs to the ask/tell engine, which runs with search_space=None and poses "
        "no query at all."
    )
    raise ValueError(msg)


@dataclass
class _Outstanding:
    """A suggestion no completed :meth:`AskTellLoop.observe` has answered yet.

    Attributes:
        suggestion (Suggestion): What was handed out.
        design (bool): Whether it is a term of the design phase rather than a pass.
        row (int | None): The row of the dataset :meth:`AskTellLoop.observe` put its term in, once
            it has; the suggestion has its value if and only if a value stands in that row.
    """

    suggestion: Suggestion
    design: bool
    row: int | None = None


class AskTellLoop(Generic[NT, T, G]):
    """The ask/tell loop over a CoSy solution space, without a strategy of its own.

    ``initialize`` builds the initial dataset, or starts the design phase that builds it;
    ``suggest`` returns the next term to evaluate, a design term first and then a pass of the
    strategy; ``observe`` takes the value back; ``finalize`` closes the run into its result.  The
    states run ``UNINITIALIZED``, ``DESIGN`` while a design phase awaits values, ``INITIALIZED``,
    then ``SUGGESTED`` and ``OBSERVED`` in alternation, and ``FINALIZED``; ``reset`` returns to the
    first.  A pass is whatever :meth:`_propose` returns, and a subclass supplies it.

    Parameters
    ----------
    search_space:
        CoSy solution space.  May be ``None`` when the design is always handed over.
    request:
        The non-terminal the search space is queried at; see ``BayesianOptimization``.
    initializer:
        Where a drawn initial design comes from; ``None`` selects sampled initialization with the
        sampler below.
    seed:
        RNG seed of the default sampler.
    sampler:
        The loop's own source of terms; ``None`` builds ``SizeUniformSampler(DEFAULT_SIZE_BOUND,
        Random(seed))`` counting from a materialized search tree, a placeholder for toy spaces,
        and says so in a warning once per run.
    """

    #: The phase a pass of this strategy is recorded under in a run's rows and term records.
    PASS_PHASE: ClassVar[str] = "pass"

    def __init__(
        self,
        search_space: SolutionSpace[NT, T, G] | None,
        request: NT | None = None,
        *,
        initializer: Initializer[NT, T, G] | None = None,
        seed: int | None = None,
        sampler: Sampler | None = None,
    ) -> None:
        # The pair is refused where the caller writes it down, so a mistyped target costs no
        # evaluation of the objective.  This reads the arguments, so it is not the guarantee on
        # its own: both attributes are public, and the pair is checked again in ``_query``, where
        # it becomes a query.
        _check_request_against_space(search_space, request)

        self.search_space = search_space
        self.request = request
        self.initializer = initializer
        self.seed = seed
        self.sampler = sampler

        # --- Ask/Tell state ---------------------------------------------------
        self._bo_state: BOState = BOState.UNINITIALIZED
        self._x_list: list[Any] = []
        self._y_list: list[float] = []
        self._x_set: set[Any] = set()
        self._initial_repeats_rejected: int = 0
        self._last_suggestion: Suggestion | None = None
        self._sampler: Sampler | None = None
        self._initializer: Initializer[NT, T, G] | None = None
        self._generator_query: Any = None
        self._iteration: int = 0
        # The design phase: the initial design in the order it is handed out, and the position of
        # the next term to hand out.
        self._design: tuple[Any, ...] = ()
        self._design_next: int = 0
        # The suggestions no completed observe() has answered, by term: whether each is a design
        # term or a pass, and the row its term went into once observe() wrote it.  Kept apart from
        # the diagnostics, which the caller holds and could alter.
        self._outstanding: dict[Any, _Outstanding] = {}
        # One record per closed pass, where the strategy keeps one; see :meth:`_record_pass`.
        self._trace: list[Any] = []
        self._logger: logging.Logger = get_logger("loop")

    def _query(self) -> Any:
        """Return the generator query naming the search space and the requested type.

        The one access path to a synthesized search space: every evolutionary component poses
        resolution queries, so this is what the initializer and the evolutionary search receive
        instead of the space itself.

        Without a search space there is no query to pose.  That configuration exists, because the
        ask/tell engine also drives optimizers that are not cosy's and the tests exercise it with
        stubs, and those callers do not read the query.  Every path that genuinely needs a search
        space checks for one before reaching this, and says so.

        **One object, and it is not a micro-optimization.**  The query is determined by the
        search space and the request, and the first call freezes that pair, so a second object would
        denote the same SLAD-tree, but ``SizeUniformSampler`` keys its counting construction by
        query *identity*, precisely because comparing a partial-term query structurally costs more
        than the lookup saves.  Minting a fresh query per call therefore made every caller pay the
        counting again: measured on the determinized CNN search space of the CIFAR-10 experiment
        at ``D = 200``, that is 93 s for the initial dataset and another 93 s for each duplicate
        fallback, against 0.04 s per draw once it is built.  Sharing the object is what makes the
        sampler's own cache reachable at all.

        Returns:
            Any: The generator query, or None if this optimizer has no search space.

        Raises:
            ValueError: If the search space has no rules for the request.
        """
        if self.search_space is None:
            return None
        if self._generator_query is None:
            # The pair the constructor read is not necessarily the pair the query is built from,
            # because both attributes are public and writable.  Checking again at the one place
            # the two are turned into a query is what makes the refusal hold for every query this
            # class hands out, and it is what makes the silent path loud.
            _check_request_against_space(self.search_space, self.request)
            self._generator_query = generator_query(self.search_space, self.request)
        return self._generator_query

    # -------------------------------------------------------------------------
    # Ask/Tell API
    # -------------------------------------------------------------------------

    def initialize(
        self,
        *,
        objective: Callable[[Any], float] | None = None,
        x0: Sequence[Any] | None = None,
        y0: Sequence[float] | None = None,
        initial_size: int = 10,
        design: Sequence[Any] | None = None,
    ) -> None:
        """Build the initial dataset, or start the design phase that builds it.

        This is the loop's first step, the dataset ``D <- ((t, sigma(q(t))) | t in init(mu_0))``
        of the terms the initializer draws paired with their objective values, with the
        engineering layer's addition that the caller may hand the pairs over ready-made.  With an
        objective, or with ``x0`` and ``y0``, the dataset is complete when this returns and the
        state is INITIALIZED.

        Without either, the design is the loop's first phase.  ``initialize(initial_size=mu_0)``
        draws the design exactly as the closed path draws it, evaluates nothing, and enters DESIGN;
        ``initialize(design=terms)`` does the same with terms the caller hands over.
        :meth:`suggest` then hands the design out in order and :meth:`observe` takes each value
        back, and after the last one the state is INITIALIZED and the passes begin.  The design
        is readable as :attr:`design` from the moment it exists.

        Parameters
        ----------
        objective:
            The objective ``sigma . q``, which is maximized.  With ``x0`` it is required when
            ``y0`` is ``None``; without ``x0`` and without ``y0`` its absence starts the design
            phase.
        x0:
            Initial candidate trees.  When ``None`` the initializer draws them
            (requires a non-``None`` ``search_space``).  Either way every term has to be
            hashable, and a design that is not is refused before a value is read for it.
        y0:
            Initial objective values.  When ``None`` ``objective`` is called on each element
            of ``x0``.
        initial_size:
            The initial size ``mu_0``: how many terms the initializer draws when ``x0`` is
            ``None``.
        design:
            Terms whose values come back one at a time through :meth:`observe`, handed out by
            :meth:`suggest` in this order.  A fixed design, given rather than drawn, so it needs
            no search space.  It takes no ``x0``, ``y0`` or ``objective`` beside it, and it must
            not repeat a term, since a repeated design term is an evaluation spent on a value the
            dataset already holds.  An empty design is a complete one.
        """
        if self._bo_state != BOState.UNINITIALIZED:
            raise RuntimeError(
                f"Cannot re-initialize: current state is {self._bo_state.value}. "
                "Call reset() first."
            )
        if design is not None:
            # Checked before anything is built, so a refused call leaves nothing behind.
            if x0 is not None or y0 is not None or objective is not None:
                raise ValueError(
                    "a design is handed over as terms whose values come back through observe(), "
                    "so it takes no x0, y0 or objective beside it"
                )
            design_terms = list(design)
            _require_hashable(design_terms, "design")
            if len(set(design_terms)) != len(design_terms):
                raise ValueError(
                    "the design repeats a term: a repeated design term is an evaluation spent on "
                    "a value the dataset already holds"
                )
            # A design handed over is not drawn, but the passes after it still replace a
            # duplicate with a fresh draw, so the sampler is built here as on every other path.
            self._resolve_sampling()
            self._start_design(design_terms, 0)
            return

        self._resolve_sampling()

        if x0 is None:
            if self.search_space is None:
                raise NotImplementedError(
                    "initialize() without x0 requires a real search_space."
                )
            x0_list, repeats = self._draw_design(initial_size)
        else:
            x0_list = list(x0)
            _require_hashable(x0_list, "x0")
            # A design the caller hands over was not drawn here, so this loop redrew nothing in
            # it.  The count is about the repair, not about the design.
            repeats = 0

        if x0 is None and y0 is None and objective is None:
            # The design phase: drawn above exactly as the closed path draws it, and evaluated by
            # the caller one term at a time, the values coming back through observe().
            self._start_design(x0_list, repeats)
            return

        # Resolve y0.  The lengths are compared before a single value is read, so that a mismatch
        # is reported as such rather than truncating the longer of the two.
        if y0 is None:
            if objective is None:
                raise ValueError(
                    "objective must be provided when y0 is None.  Terms whose values are to "
                    "come back through observe() are handed over as design=."
                )
            y0_list = [_finite_or_raise(objective(t), t) for t in x0_list]
        else:
            if len(x0_list) != len(y0):
                raise ValueError(
                    f"len(x0)={len(x0_list)} != len(y0)={len(y0)}."
                )
            y0_list = [_finite_or_raise(v, t) for t, v in zip(x0_list, y0, strict=True)]

        self._x_list = x0_list
        self._y_list = y0_list
        # Checked here rather than at the fit: a dataset that contradicts itself does so the moment
        # it is handed over, and the first suggest() is far from the caller who assembled it.
        self._distinct_pairs()
        self._x_set = set(self._x_list)
        # The count is stored with the dataset it describes and not where it is computed.
        # Everything between the draw and the last line of this method can raise, the objective
        # above all, and a failure there leaves the state UNINITIALIZED, which permits a second
        # initialize().  Storing the count at the draw would carry the count of the abandoned
        # design into whatever design the caller supplies next.
        self._initial_repeats_rejected = repeats
        self._last_suggestion = None
        self._iteration = 0
        self._bo_state = BOState.INITIALIZED

    def _draw_design(self, size: int) -> tuple[list[Any], int]:
        """Draw an initial design of ``size`` pairwise distinct terms, and count the repeats redrawn.

        The loop's own draw: the configured initializer, then the repair that makes the design a
        set.  A strategy that draws its terms another way, from one stream for design and passes
        alike, overrides this.

        Args:
            size (int): The size of the design.

        Returns:
            tuple[list[Any], int]: The design, and how many repeated terms were redrawn.
        """
        assert self._initializer is not None
        assert self._sampler is not None
        # The previous code drew a pool a hundred times the size and thinned it with a
        # greedy determinantal point process, which is related work rather than either of the
        # initializers this loop admits, and it warned and came back short where the stream
        # simply raises.  Both of those initializers raise rather than return a short
        # population.
        x0_list = list(self._initializer.initialize(self._query(), size))
        # An initializer of the caller's own may answer with anything.  The two this
        # package ships return terms, which are hashable by construction, so an unhashable
        # candidate here is a broken initializer rather than an unusual term.
        _require_hashable(x0_list, "the design the initializer returned")
        # And then the dataset is made a *set*, which is not something the initializer can be
        # asked for.  Sampled initialization builds a population, and a population is a finite
        # multiset, so repeats are admissible there by definition.  The loop's dataset is not:
        # a repeated term is a training point carrying no observation the previous one did
        # not, one evaluation of the budget spent on nothing.
        #
        # The comment that used to stand here inherited the guarantee from the default
        # sampler, whose stream lists each inhabitant within the bound exactly once, so that
        # every prefix of it is a sample without replacement.  But ``sampler`` is a parameter
        # of this class, and the depth-bounded random sampler draws independently, so its
        # draws may repeat a term.  A guarantee that holds only until someone uses the
        # parameter is the kind of unwritten invariant this API must not have.
        x0_list, repeats = _distinct_dataset(
            x0_list, self._sampler, self._query(), size
        )
        if repeats:
            self._logger.info(
                "the initializer returned %d repeated term(s) in an initial design of %d.  "
                "They were redrawn, as suggest() redraws a duplicate candidate.  The "
                "size-uniform sampler lists each inhabitant within its bound exactly once, so "
                "under it this number is 0",
                repeats, size,
            )
        return x0_list, repeats

    def _resolve_sampling(self) -> None:
        """Build the sampler and the initializer the loop draws from, where there is a space.

        The sampler is built whenever there is a space to draw from: the duplicate fallback of
        :meth:`suggest` draws from it whichever initializer the caller chose, and whether the
        design was drawn, handed over as terms, or handed over with its values.
        """
        if self._sampler is None and self.search_space is not None:
            if self.sampler is not None:
                self._sampler = self.sampler
            else:
                self._sampler = SizeUniformSampler(DEFAULT_SIZE_BOUND, random.Random(self.seed))
                # Said where it is built, which is once per run: reset() drops the sampler.
                self._logger.warning(
                    "no sampler was given, so every term the loop draws itself -- a design it "
                    "draws, a replacement for a duplicate, a random search's passes -- comes from "
                    "SizeUniformSampler(%d, Random(%r)): terms of up to %d symbols, counted over "
                    "the derivation tree cosy's default construction builds. That suits small "
                    "spaces only; pass sampler= (the run layer's build_search pairs a program with "
                    "the sampler that fits it).",
                    DEFAULT_SIZE_BOUND, self.seed, DEFAULT_SIZE_BOUND,
                )
        if self._initializer is None and self._sampler is not None:
            self._initializer = (
                SampledInitialization(self._sampler)
                if self.initializer is None
                else self.initializer
            )

    def _start_design(
        self,
        terms: list[Any],
        repeats: int,
    ) -> None:
        """Enter the design phase with ``terms`` as the design, and an empty dataset.

        Args:
            terms (list[Any]): The design, pairwise distinct, in the order it is handed out.
            repeats (int): How many repeats the draw had to redraw, 0 for a design handed over.
        """
        self._design = tuple(terms)
        self._design_next = 0
        self._outstanding = {}
        self._x_list = []
        self._y_list = []
        self._x_set = set()
        self._initial_repeats_rejected = repeats
        self._last_suggestion = None
        self._iteration = 0
        # An empty design is complete the moment it exists, as ``initialize(x0=[], y0=[])`` is.
        self._bo_state = BOState.DESIGN if self._design else BOState.INITIALIZED

    @property
    def design(self) -> tuple[Any, ...]:
        """The initial design of the design phase, in the order :meth:`suggest` hands it out.

        Empty where the design was not handed out through the phase: before :meth:`initialize`,
        after :meth:`reset`, and where :meth:`initialize` received the values with the terms or an
        objective to compute them.  It stays readable once the phase is over, as the record of
        what the design was.

        Returns:
            tuple[Any, ...]: The design terms.
        """
        return self._design

    def _suggest_design_term(self, verbose: bool) -> Suggestion:
        """Hand out the next term of the design, which maximizes nothing and fits nothing.

        Args:
            verbose (bool): Log the suggestion.

        Returns:
            Suggestion: The design term, without an acquisition value, with the design phase and
                the term's position in its diagnostics.
        """
        index = self._design_next
        diagnostics: Diagnostics = {
            "timestamp": time.time(),
            "iteration": self._iteration,
            "phase": "design",
            "design_index": index,
        }
        suggestion = Suggestion(
            candidate=self._design[index],
            acquisition_value=None,
            diagnostics=diagnostics,
        )
        self._design_next = index + 1
        self._outstanding[suggestion.candidate] = _Outstanding(suggestion, design=True)
        self._last_suggestion = suggestion
        self._bo_state = BOState.SUGGESTED
        if verbose:
            enable_verbose_logging()
        log_suggestion(self._logger, suggestion)
        return suggestion

    def _distinct_pairs(self) -> tuple[list[Any], list[float]]:
        """Return the distinct pairs of the dataset, in order of first appearance.

        The loop conditions the surrogate on these rather than on the whole dataset, because an
        evaluation may repeat and an exact observation repeats identically.  The order is the one
        the pairs first arrived in, so that two runs from the same seed condition on the same
        matrix rather than on a permutation of it.

        Returns:
            tuple[list[Any], list[float]]: The distinct terms and their values.

        Raises:
            ValueError: If one term carries two different values.  The loop observes a *function*,
                so that is not a repetition.  Keeping one of the two would pick a measurement on
                the caller's behalf, and keeping both is the singular Gram matrix this clause
                exists to prevent.  A quality measure that genuinely varies between calls is the
                stochastic case, and that one is modeled with a noise term on the diagonal rather
                than with two rows.
        """
        values: dict[Any, float] = {}
        terms: list[Any] = []
        observations: list[float] = []
        for candidate, value in zip(self._x_list, self._y_list, strict=True):
            if candidate in values:
                if values[candidate] != value:
                    raise ValueError(
                        f"the dataset gives {candidate} two different values, "
                        f"{values[candidate]} and {value}: the loop observes a function, so a "
                        f"repeated term repeats its value.  A quality measure that varies between "
                        f"calls is the stochastic case, which belongs on the diagonal of the "
                        f"kernel (WhiteKernel), not in the dataset."
                    )
                continue
            values[candidate] = value
            terms.append(candidate)
            observations.append(value)
        return terms, observations

    @property
    def query(self) -> Any:
        """The generator query this loop poses, or ``None`` where there is no search space.

        The one object, so that a caller who wants to draw from :attr:`sampler` themselves, as a
        paired random-search baseline does, draws through the same query the loop uses, and the
        sampler's counting construction is built once for both.  Asking the sampler with a query
        of one's own would be correct and would pay the counting twice.

        Returns:
            Any: The query, or None.

        Raises:
            ValueError: If the search space has no rules for the request.  This is the path that
                used to report nothing and hand back a query every draw from which is empty.
        """
        return self._query()

    def observe(self, candidate: Any, y: float) -> None:
        """Record the objective value for the last suggested candidate.

        A term of the design phase goes into the dataset without counting as a pass and without a
        record of the strategy's; after the design's last value the state is INITIALIZED.  A pass
        goes into the dataset and counts, and the strategy records it where it keeps a trace
        (``BayesianOptimization``: a row of its run trace).

        Parameters
        ----------
        candidate:
            Must match the candidate from the last ``suggest()`` call.
        y:
            The objective value, exactly as measured.

        Raises
        ------
        RuntimeError
            If called outside the SUGGESTED state.
        ValueError
            If ``candidate`` does not match the last suggestion, if ``y`` is not finite, or if the
            strategy refuses the pass before its value enters the dataset
            (``BayesianOptimization``: no diagnostics a trace row can be read from).
        """
        if self._bo_state != BOState.SUGGESTED:
            raise RuntimeError(
                f"observe() is not allowed in state {self._bo_state.value}.  "
                "Call suggest() first."
            )
        if self._last_suggestion is None:
            raise RuntimeError("Internal error: _last_suggestion is None in SUGGESTED state.")
        if candidate != self._last_suggestion.candidate:
            raise ValueError(
                "Observed candidate does not match the last suggested candidate."
            )

        # Record the tree suggest() produced, not the caller's argument.  The check above is
        # structural on purpose, since a caller may legitimately hand back a reconstructed tree,
        # but the observation set has to hold exactly what was suggested.
        recorded = self._last_suggestion.candidate
        value = _finite_or_raise(y, recorded)
        entry = self._outstanding[recorded]
        if entry.design:
            # A design term goes into the dataset and nowhere else: it has no acquisition value
            # and no pick, so it is no row of the trace and no pass of the count.  The phase ends
            # with the value of its last term, and the passes begin.
            entry.row = len(self._x_list)
            self._x_list.append(recorded)
            self._x_set.add(recorded)
            self._y_list.append(value)
            del self._outstanding[recorded]
            self._bo_state = (
                BOState.DESIGN if self._design_next < len(self._design) else BOState.INITIALIZED
            )
            return
        # Checked before the dataset grows, because the row is written after it has.  A suggestion
        # no row can be read from would otherwise leave a term and a value behind that no trace
        # row and no iteration count mention, and the state stays at SUGGESTED, so the very same
        # call is accepted again and appends them a second time.
        self._check_pass_suggestion(self._last_suggestion)
        entry.row = len(self._x_list)
        self._x_list.append(recorded)
        self._x_set.add(recorded)
        self._y_list.append(value)
        self._record_pass(self._last_suggestion, value)
        self._iteration += 1
        self._bo_state = BOState.OBSERVED
        # Answered only now: an observe() that raised after the dataset grew leaves the entry, as
        # it leaves the state at SUGGESTED, and the row says the value is there.
        del self._outstanding[recorded]

    @property
    def initial_repeats_rejected(self) -> int:
        """How many repeated terms the initializer produced and this loop redrew.

        Zero under the size-uniform sampler, whose stream lists each inhabitant within the bound
        exactly once, so that every prefix of it is a sample without replacement.  A nonzero value
        there says that guarantee did not hold, and under the depth-bounded random sampler, whose
        draws are independent and may repeat, it says how much the repair had to do.  Reported
        rather than absorbed: a run whose initial design needed redrawing is a run whose sampler
        does not supply what its dataset needs, and the record has to be able to say so.

        A design the caller hands to ``initialize()`` reports 0 whatever it contains, because the
        number counts what this loop redrew and it redrew nothing there.

        Returns:
            int: The count, 0 before ``initialize()`` and 0 again after ``reset()``.
        """
        return self._initial_repeats_rejected

    def reset(self) -> None:
        """Reset to UNINITIALIZED, clearing all observations.

        Constructor parameters are preserved.  The **default** initializer is rebuilt with it, so
        a second run repeats the first: its sampler is seeded from ``self.seed``, and dropping it
        here is what makes the stream start over rather than continue.

        An initializer handed to the constructor is a different matter.  It carries its own
        random state, which this cannot reach and does not reset.  A second run under one of those
        continues where the first left off, and two runs of an otherwise identical configuration
        draw different populations.  A caller who wants them to agree constructs a fresh
        initializer between runs.
        """
        self._bo_state = BOState.UNINITIALIZED
        self._x_list = []
        self._y_list = []
        self._x_set = set()
        self._initial_repeats_rejected = 0
        self._last_suggestion = None
        self._iteration = 0
        self._sampler = None
        self._initializer = None
        self._trace = []
        self._design = ()
        self._design_next = 0
        self._outstanding = {}

    def best(self) -> tuple[Any, float]:
        """Return the best observation as ``(candidate, y)``.

        Best is the **largest** observed value: the loop maximises.

        Raises
        ------
        RuntimeError
            If called before any observations have been made.
        """
        if not self._y_list:
            raise RuntimeError("No observations available yet.")
        idx = int(np.argmax(np.array(self._y_list, dtype=float)))
        return self._x_list[idx], float(self._y_list[idx])

    def finalize(self) -> dict[str, Any]:
        """Transition to FINALIZED and return the optimization result.

        Returns
        -------
        dict with keys:
            ``best_tree``, ``best_y``, ``x``, ``y``, ``gp_model``, ``iterations``, ``trace``,
            ``dropped_suggestion``, ``design_remaining``.

            ``gp_model`` is the model the strategy reports, ``None`` for one without a model such as
            ``RandomSearch``.  For ``BayesianOptimization`` it is the surrogate the **last**
            :meth:`suggest` fitted, and nothing is fitted after it, so the model has seen nothing
            the run appended since.  A run that ends on an observation reports a model that never
            saw the pair it ended on.  A run that ends on an outstanding suggestion reports the
            model of that pass, which saw every pair the dataset held, because that pass appended
            none of its own.  Either way the fit is over the *distinct* pairs of what it was handed,
            which is fewer than the dataset holds whenever a term repeats in it.  A diagnostic that
            wants a surrogate over the whole dataset, as the fit scatter and the leave-one-out
            calibration of the acceptance checks do, asks :meth:`surrogate_over_dataset` for one
            rather than reading this.

            ``BayesianOptimization``'s :attr:`last_acquisition_run` is left standing too, but it
            describes that same pass only where that :meth:`suggest` was asked to record a
            population.  Where it was not, the attribute still holds the most recent pass that was
            asked, which is an earlier one, or ``None`` if no pass was ever asked.

            ``iterations`` counts the passes :meth:`observe` closed.  A suggestion that never got
            a value is not one of them, and it is not a row of ``trace`` either.

            ``dropped_suggestion`` is the candidate of a suggestion that no value ever reached.
            It is ``None`` where no suggestion was open, and also where an open one already had
            its value in the dataset.  Finalizing with a suggestion open is allowed, and it is how
            an aborted run closes: a failing evaluation raises between
            :meth:`suggest` and :meth:`observe`, and this call is what puts such a run into
            ``FINALIZED`` and names in one answer what it collected and which candidate it gave
            up on.  What such a candidate must not do is vanish.  An evaluation of the ask/tell
            layer may live outside this process and may already have been paid for, and
            :meth:`observe` is the only way to get it into the dataset, so the term is named
            here and logged instead.  The suggestion itself is
            cleared on every path, so a snapshot taken after this reports none outstanding.

            ``trace`` is the per-pass table of the trace readings.  See :attr:`trace`.

            ``design_remaining`` lists, in the design's order, the terms of the design phase that
            no value reached: empty once the design is complete, and empty where the design was
            not handed out through the phase.  A run finalized during its design keeps every value
            it observed and names here what it never measured.

        Raises
        ------
        RuntimeError
            If called in an invalid state, or on an empty dataset.  The loop answers with a term
            of maximal observed value, and a run that observed nothing has none: reporting
            ``(None, nan)`` instead reads downstream as a completed run with a worthless optimum.
        """
        if self._bo_state not in (
            BOState.DESIGN, BOState.INITIALIZED, BOState.OBSERVED, BOState.SUGGESTED
        ):
            raise RuntimeError(
                f"finalize() is not allowed in state {self._bo_state.value}."
            )

        best_tree, best_y = self.best()

        # A suggestion was dropped when no value reached it: when observe() never wrote its term,
        # or wrote the term and not the value.  The state does not answer that question: observe()
        # writes the term and its value before it moves the state, so a run interrupted in between
        # sits in SUGGESTED with the value already recorded, and calling that term dropped would
        # state the reverse of the truth.  The row the term went into answers it exactly, because
        # the two lists are appended in step and read by position everywhere else, so the
        # suggestion has its value if and only if a y entry stands in that row.  Membership in the
        # duplicate index is not the same test: the index is written between the two appends, so an
        # interrupt there leaves it claiming a value the dataset does not hold.  Nor is the term's
        # having some value: that is the same answer only while a suggested term is always new.
        valued_terms = self._x_list[: len(self._y_list)]
        dropped = None
        for entry in self._outstanding.values():
            if entry.row is not None and entry.row < len(self._y_list):
                continue
            dropped = entry.suggestion.candidate
            self._logger.warning(
                "the run is finalized with the suggestion %s still outstanding.  No value ever "
                "reached it, so the dataset holds none for the term, and it is not among the %d "
                "passes this result counts.  An evaluation that fails leaves the "
                "closed loop in exactly this state, and a caller who did measure the term hands "
                "it to observe() before finalizing.  The result names it under "
                "dropped_suggestion.",
                dropped,
                self._iteration,
            )
        self._last_suggestion = None
        self._outstanding = {}
        # The design terms without a value: the outstanding one, if it is a design term, and every
        # one the phase never handed out.  Empty once the design is complete.
        valued = set(valued_terms)
        design_remaining = [term for term in self._design if term not in valued]

        self._bo_state = BOState.FINALIZED
        return {
            "best_tree": best_tree,
            "best_y": best_y,
            "x": np.asarray(self._x_list, dtype=object),
            "y": np.asarray(self._y_list, dtype=float),
            "gp_model": self._result_model(),
            "iterations": self._iteration,
            "trace": self.trace,
            "dropped_suggestion": dropped,
            "design_remaining": design_remaining,
        }

    def get_state_snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of the current Ask/Tell state."""
        valued = set(self._x_list[: len(self._y_list)])
        return {
            "state": self._bo_state.value,
            "x_list": list(self._x_list),
            "y_list": list(self._y_list),
            "iteration": self._iteration,
            "last_suggestion": (
                None if self._last_suggestion is None
                else self._last_suggestion.candidate
            ),
            # The design phase: the design, and how many of its terms still await a value.
            "design": list(self._design),
            "design_remaining": sum(1 for term in self._design if term not in valued),
        }

    # -------------------------------------------------------------------------
    # The closed loop
    # -------------------------------------------------------------------------

    def suggest(self, *, verbose: bool = False) -> Suggestion:
        """Return the next term to evaluate: a design term in DESIGN, a pass of the strategy after.

        Args:
            verbose (bool): Log the suggestion. (Default value = False)

        Returns:
            Suggestion: The term, with the diagnostics of its phase.

        Raises:
            RuntimeError: Outside the states a suggestion belongs in.
        """
        return self._suggest(verbose, lambda: self._propose(verbose))

    def _suggest(self, verbose: bool, propose: Callable[[], Suggestion]) -> Suggestion:
        """The state machine around a suggestion, the same for every strategy.

        A design term in DESIGN; otherwise the pass ``propose`` returns, recorded as outstanding
        and logged.  A strategy whose :meth:`suggest` takes arguments of its own calls this with
        a ``propose`` that closes over them.

        Args:
            verbose (bool): Log the suggestion.
            propose (Callable[[], Suggestion]): The strategy's proposal of a pass.

        Returns:
            Suggestion: The term.

        Raises:
            RuntimeError: Outside the states a suggestion belongs in.
        """
        if self._bo_state == BOState.DESIGN:
            return self._suggest_design_term(verbose)
        if self._bo_state not in (BOState.INITIALIZED, BOState.OBSERVED):
            raise RuntimeError(
                f"suggest() is not allowed in state {self._bo_state.value}."
            )
        suggestion = propose()
        self._last_suggestion = suggestion
        self._outstanding[suggestion.candidate] = _Outstanding(suggestion, design=False)
        self._bo_state = BOState.SUGGESTED
        log_suggestion(self._logger, suggestion)
        return suggestion

    def _propose(self, verbose: bool) -> Suggestion:
        """Propose the next pass: the one method a strategy has to supply.

        Args:
            verbose (bool): Whether the caller asked for verbose logging, which the strategy
                switches on where it sees fit.

        Returns:
            Suggestion: A term the dataset does not hold, with the diagnostics of a pass.

        Raises:
            NotImplementedError: Always, on the loop without a strategy.
        """
        raise NotImplementedError(
            f"{type(self).__name__} proposes no pass: a strategy supplies _propose()"
        )

    def _check_pass_suggestion(self, suggestion: Suggestion) -> None:
        """Refuse a pass before its value enters the dataset; nothing to refuse by default.

        Args:
            suggestion (Suggestion): The pass whose value is about to be recorded.
        """

    def _record_pass(self, suggestion: Suggestion, observed: float) -> None:
        """Record a closed pass after its value entered the dataset; nothing by default.

        Args:
            suggestion (Suggestion): The pass.
            observed (float): Its value, already checked for finiteness.
        """

    def _result_model(self) -> Any:
        """The model a result reports under ``gp_model``; a strategy without one reports ``None``.

        Returns:
            Any: The model, or None.
        """
        return None

    @property
    def trace(self) -> list[Any]:
        """One record per closed pass, where the strategy keeps one; empty where it keeps none.

        Returns:
            list[Any]: The records.
        """
        return list(self._trace)

    def check_configuration(self) -> None:
        """Refuse, before anything is evaluated, a configuration no pass could use.

        The loop without a strategy has nothing to refuse; a strategy with configuration overrides
        this, and a caller asks it before :meth:`initialize`.
        """
