"""The two fitness objectives the acquisition optimizer builds.

These had no tests of their own: they were exercised only through whole BO runs, where a known
point that outranks a novel one shows up as a fallback warning several layers away, if at all.
The properties below are the ones the layer promises in its docstrings.
"""
from __future__ import annotations

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

    def evolutionary_best(self, query: Any, objective: Any, mode: str) -> Any:
        self.calls.append((query, mode))
        return _NOVEL_A


@pytest.mark.parametrize("mode", ["batch", "single"])
def test_the_requested_mode_reaches_the_search(mode):
    """The mode decides which objective is built, so the search must receive the one requested."""
    search = RecordingSearch()
    af = ProbabilityOfImprovement(gp=_NEGATIVE, incumbent=0.0, known_points={_KNOWN})

    result = AcquisitionOptimizer(search).maximize(af, query="a query", mode=mode)

    assert result is _NOVEL_A
    assert search.calls == [("a query", mode)]
