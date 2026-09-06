"""The two fitness objectives the acquisition optimizer builds.

These had no tests of their own: they were exercised only through whole BO runs, where a known
point that outranks a novel one shows up as a fallback warning several layers away, if at all.
The properties below are the ones the layer promises in its docstrings.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from cosy.core.tree import Tree

from bayesian_optimization.acquisition_function import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from bayesian_optimization.acquisition_optimizer import (
    AcquisitionOptimizer,
    _make_acquisition_objective_batch,
    _make_acquisition_objective_single,
    resolve_fitness_mode,
)
from tests.unit.test_acquisition import FixedPosterior

# Three candidates whose UCB scores are all far below zero, so a floor that does not look at them
# lands above.  This is the configuration that made the fixed 0.0 sentinel a maximizer.
_KNOWN = Tree("evaluated_already")
_NOVEL_A = Tree("novel_a")
_NOVEL_B = Tree("novel_b")
_NEGATIVE = FixedPosterior(
    {_KNOWN: (-2.0, 0.1), _NOVEL_A: (-5.0, 0.1), _NOVEL_B: (-9.0, 0.1)}
)


def _ucb() -> UpperConfidenceBound:
    return UpperConfidenceBound(gp=_NEGATIVE, beta=2.0, known_points={_KNOWN})


# ---------------------------------------------------------------------------
# The batch objective
# ---------------------------------------------------------------------------

def test_the_floor_holds_across_generations_not_only_within_one():
    """The floor is recomputed from every genuine score seen so far, not from the current batch.

    A generation that happens to hold only mild candidates would otherwise fix a floor that the
    next generation's candidates fall below, since a better candidate scores lower under a
    negative UCB, and the known point would win in that later generation.
    """
    objective = _make_acquisition_objective_batch(_ucb())

    first = objective([_KNOWN, _NOVEL_A])
    assert first[_KNOWN] < first[_NOVEL_A]

    second = objective([_KNOWN, _NOVEL_B])
    assert second[_KNOWN] < second[_NOVEL_B]
    assert second[_KNOWN] < first[_NOVEL_A], "the floor must not rise back over an earlier score"


def test_the_batch_objective_scores_every_sample_it_is_given():
    objective = _make_acquisition_objective_batch(_ucb())
    scores = objective([_KNOWN, _NOVEL_A, _NOVEL_B])
    assert set(scores) == {_KNOWN, _NOVEL_A, _NOVEL_B}


def test_the_batch_objective_rejects_a_non_term():
    objective = _make_acquisition_objective_batch(_ucb())
    with pytest.raises(TypeError):
        objective([_NOVEL_A, None])


def test_a_known_point_is_never_asked_of_the_surrogate():
    """It has no score of its own, and asking would also cost a predict call per generation."""
    asked: list[Any] = []

    class Recording(FixedPosterior):
        def predict(self, X: Any, return_std: bool = False) -> Any:
            asked.extend(X)
            return super().predict(X, return_std)

    gp = Recording(dict(_NEGATIVE.posterior))
    objective = _make_acquisition_objective_batch(
        UpperConfidenceBound(gp=gp, beta=2.0, known_points={_KNOWN})
    )
    objective([_KNOWN, _NOVEL_A])
    assert _KNOWN not in asked


# ---------------------------------------------------------------------------
# The single objective
# ---------------------------------------------------------------------------

def test_the_single_objective_floors_a_known_point_whenever_it_is_asked():
    """Asked first or asked last, the known point must lose either way.

    It used to derive its floor from the scores cached so far, which is empty on the first call:
    a known point asked before anything else scored ``-1.0`` and beat every genuine candidate
    below that.  A bounded acquisition has a fixed floor and does not depend on the order.
    """
    pi = ProbabilityOfImprovement(gp=_NEGATIVE, incumbent=0.0, known_points={_KNOWN})

    first_asked = _make_acquisition_objective_single(pi)
    known_first = first_asked(_KNOWN)
    novel_after = first_asked(_NOVEL_A)

    last_asked = _make_acquisition_objective_single(pi)
    novel_before = last_asked(_NOVEL_A)
    known_last = last_asked(_KNOWN)

    assert known_first == known_last, "the floor must not depend on when it is asked"
    assert known_first < novel_after
    assert known_last < novel_before


def test_the_single_objective_refuses_an_acquisition_without_a_lower_bound():
    """UCB's floor needs the other scores of the generation, and alone there are none."""
    with pytest.raises(ValueError, match="batch"):
        _make_acquisition_objective_single(_ucb())


def test_an_unbounded_acquisition_is_fine_alone_when_nothing_is_known_yet():
    """Without known points there is no floor to place, so the mode is usable."""
    objective = _make_acquisition_objective_single(
        UpperConfidenceBound(gp=_NEGATIVE, beta=2.0)
    )
    assert objective(_NOVEL_A) == pytest.approx(-5.0 + 2.0 * 0.1)


def test_the_single_objective_rejects_a_non_term():
    objective = _make_acquisition_objective_single(
        ExpectedImprovement(gp=_NEGATIVE, incumbent=0.0, known_points={_KNOWN})
    )
    with pytest.raises(TypeError):
        objective("not a term")


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------

class RecordingSearch:
    """An evolutionary search that records how it was called and returns a fixed candidate."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, str]] = []
        self.objectives: list[Any] = []

    def evolutionary_best(self, query: Any, objective: Any, mode: str) -> Any:
        self.calls.append((query, mode))
        self.objectives.append(objective)
        return _NOVEL_A

    def evolutionary_stream(self, query: Any, objective: Any, mode: str) -> Any:
        """Yield one generation, with the fields the record of a generation reads."""
        self.calls.append((query, mode))
        self.objectives.append(objective)
        population = [_NOVEL_A, _NOVEL_B]
        scores = (
            objective(population)
            if mode == "batch"
            else {candidate: objective(candidate) for candidate in population}
        )
        yield SimpleNamespace(
            generation=0,
            population=population,
            offspring=[],
            fitness=scores,
            best=_NOVEL_A,
            best_fitness=scores[_NOVEL_A],
            last_improvement=0,
        )


@pytest.mark.parametrize("mode", ["batch", "single"])
def test_the_requested_mode_reaches_the_search(mode):
    """The mode decides which objective is built, so the search must receive the one requested."""
    search = RecordingSearch()
    af = ProbabilityOfImprovement(gp=_NEGATIVE, incumbent=0.0, known_points={_KNOWN})

    result = AcquisitionOptimizer(search).maximize(af, query="a query", mode=mode)

    assert result is _NOVEL_A
    assert search.calls == [("a query", mode)]


@pytest.mark.parametrize("mode", ["auto", "batch"])
def test_the_mode_the_search_defaults_to_is_the_batch_mode(mode):
    """Both ``"auto"`` and ``"batch"`` reach the search as ``"batch"``, with the batch objective.

    ``"auto"`` is the mode the search itself defaults to, and it means: score a generation in one
    call where the fitness function takes a list.  The objective handed over is written here, and
    the batch one is what that reading selects, so the answer is known before the search is
    called.  It used to fall through to the single-sample objective instead, which returns the
    same scores in the same order and rebuilds the training side of the kernel matrix once per
    candidate to do it.

    The search is told the resolved mode rather than the word asked for, so what it makes of
    ``"auto"`` cannot disagree with the objective it was handed.
    """
    search = RecordingSearch()
    af = ProbabilityOfImprovement(gp=_NEGATIVE, incumbent=0.0, known_points={_KNOWN})

    result = AcquisitionOptimizer(search).maximize(af, query="a query", mode=mode)

    assert result is _NOVEL_A
    assert search.calls == [("a query", "batch")]
    scores = search.objectives[0]([_NOVEL_A, _NOVEL_B])
    assert set(scores) == {_NOVEL_A, _NOVEL_B}, "the batch objective takes a whole generation"


@pytest.mark.parametrize("mode", ["auto", "batch"])
def test_the_population_entry_resolves_the_mode_as_well(mode):
    """Both entries read the mode, and a caller cannot be asked to know which one resolves it."""
    search = RecordingSearch()
    af = ProbabilityOfImprovement(gp=_NEGATIVE, incumbent=0.0, known_points={_KNOWN})

    best, population, generations = AcquisitionOptimizer(search).maximize_with_population(
        af, query="a query", mode=mode
    )

    assert best is _NOVEL_A
    assert population == [_NOVEL_A, _NOVEL_B]
    assert len(generations) == 1
    assert search.calls == [("a query", "batch")]


@pytest.mark.parametrize("mode", ["btach", "BATCH", "single ", "", None, 1, ["batch"]])
def test_a_mode_outside_the_three_is_refused_before_the_search_runs(mode):
    """A mode this adapter cannot serve is an error, not a reason to take the other path.

    Every value but ``"batch"`` used to build the single-sample objective and reach the search
    unchanged.  The two objectives answer alike, so nothing in the run distinguished a mistyped
    ``"batch"`` from a deliberate ``"single"``, and the misreading only ever cost time.

    A value that cannot be hashed is refused for not naming a mode rather than for not being
    hashable, which is what a membership test against a set would have made of it.
    """
    search = RecordingSearch()
    af = ProbabilityOfImprovement(gp=_NEGATIVE, incumbent=0.0, known_points={_KNOWN})

    with pytest.raises(ValueError, match="fitness mode"):
        AcquisitionOptimizer(search).maximize(af, query="a query", mode=mode)

    assert search.calls == [], "the mode is read before anything is evaluated"


# ---------------------------------------------------------------------------
# The cache of the single objective, and what its key is
# ---------------------------------------------------------------------------

class CountingPosterior(FixedPosterior):
    """A posterior that records how often the surrogate was actually asked."""

    def __init__(self, posterior: dict[Any, tuple[float, float]]) -> None:
        super().__init__(posterior)
        self.calls = 0

    def predict(self, X: Any, return_std: bool = False) -> Any:
        self.calls += 1
        return super().predict(X, return_std)


class AsksInTurn:
    """A search that scores the candidates it was built with, one generation each."""

    def __init__(self, *candidates: Any) -> None:
        self.candidates = candidates
        self.scores: list[float] = []

    def evolutionary_best(self, query: Any, objective: Any, mode: str) -> Any:
        self.scores = [objective(candidate) for candidate in self.candidates]
        return self.candidates[0]


def _bounded(posterior: Any) -> ProbabilityOfImprovement:
    return ProbabilityOfImprovement(gp=posterior, incumbent=0.0, known_points=set())


def test_the_single_objective_asks_the_surrogate_once_per_candidate():
    """A candidate that survives into the next generation is scored again and asked once.

    The single-sample path pays one full surrogate call per scoring, and asking the surrogate
    rebuilds the training side of the kernel matrix once per distinct candidate.  A search that
    keeps no fitness cache of its own scores the same candidate once per generation it survives.
    ``EvolutionarySearch`` does keep one for the whole run, so this cache is what holds the cost
    down for any other search the optimizer is handed.
    """
    posterior = CountingPosterior({_NOVEL_A: (-5.0, 0.1)})
    search = AsksInTurn(_NOVEL_A, _NOVEL_A)

    AcquisitionOptimizer(search).maximize(_bounded(posterior), query="a query", mode="single")

    assert search.scores[0] == search.scores[1]
    assert posterior.calls == 1, "the second scoring of the same candidate must come from the cache"


def test_the_cached_score_is_keyed_by_the_term_and_not_by_the_object():
    """Two terms that are equal are one candidate, and the search hands over whichever it built.

    A search produces its candidates by recombination, so the object that carries a term in a
    later generation is rarely the object an earlier generation was scored on.  A cache keyed by
    identity would miss every one of them and still look like a cache.
    """
    rebuilt = Tree("novel_a")
    assert rebuilt == _NOVEL_A and rebuilt is not _NOVEL_A

    posterior = CountingPosterior({_NOVEL_A: (-5.0, 0.1)})
    search = AsksInTurn(_NOVEL_A, rebuilt)

    AcquisitionOptimizer(search).maximize(_bounded(posterior), query="a query", mode="single")

    assert search.scores[0] == search.scores[1]
    assert posterior.calls == 1


def test_the_cache_belongs_to_one_maximization_and_not_to_the_optimizer():
    """Each maximization builds its own objective, and the scores of a pass belong to that pass.

    A pass conditions the surrogate on everything observed so far, so the score of a term changes
    from one pass to the next.  A cache that outlived its maximization would answer the second
    pass with the numbers of the first, and the run would optimize a surrogate it no longer has.
    """
    posterior = CountingPosterior({_NOVEL_A: (-5.0, 0.1)})
    optimizer = AcquisitionOptimizer(AsksInTurn(_NOVEL_A, _NOVEL_A))

    optimizer.maximize(_bounded(posterior), query="a query", mode="single")
    optimizer.maximize(_bounded(posterior), query="a query", mode="single")

    assert posterior.calls == 2


# ---------------------------------------------------------------------------
# A stream that yields nothing
# ---------------------------------------------------------------------------

def test_a_stream_that_yields_no_generation_at_all_is_refused():
    """No generation and no candidate are different answers, and only one of them is a result.

    ``maximize`` returns whatever the search found, and ``None`` there means the search ran and
    found nothing.  A stream that ends before its zeroth generation means it never ran, and
    reading the population and the frontier off it would report an empty run as a finished one.
    """
    class Silent:
        def evolutionary_stream(self, query: Any, objective: Any, mode: str) -> Any:
            return iter(())

    af = ProbabilityOfImprovement(gp=_NEGATIVE, incumbent=0.0, known_points={_KNOWN})

    with pytest.raises(RuntimeError, match="yielded no generation at all"):
        AcquisitionOptimizer(Silent()).maximize_with_population(af, query="a query")


# ---------------------------------------------------------------------------
# The three modes, read against the search that receives them
# ---------------------------------------------------------------------------

def test_resolving_auto_here_gives_what_the_search_would_have_read_itself():
    """The three modes are the search's own, and ``auto`` is resolved before it can read it.

    The search decides ``auto`` by the annotation of the fitness function it was handed: a single
    positional parameter annotated as a collection means a whole generation per call.  Both
    objectives are written in this package, so the answer is fixed here, and fixing it is only
    honest while the two readings agree.  This holds them to that, on the search's own reading
    rather than on a restatement of it.
    """
    from cosy.evolutionary_algorithms.evolutionary import EvolutionarySearch

    af = ProbabilityOfImprovement(gp=_NEGATIVE, incumbent=0.0, known_points={_KNOWN})
    reads_as_batch = EvolutionarySearch._looks_like_batch_fitness_function

    assert resolve_fitness_mode("auto") == "batch"
    assert reads_as_batch(_make_acquisition_objective_batch(af)) is True
    assert reads_as_batch(_make_acquisition_objective_single(af)) is False
