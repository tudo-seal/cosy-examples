"""Expected hypervolume improvement with the cost taken exactly: one modelled value, one exact cost.

The value is maximized and modelled by the surrogate; the cost is minimized and known for every
term, so the value is the only random variable (Loka et al., "Bi-objective Bayesian optimization of
engineering problems with cheap and expensive cost functions", Engineering with Computers 39, 2023,
Eqs. 12 to 15 of the authors' manuscript).

The score is checked against an oracle that shares nothing with the implementation's form: the
hypervolume of the front with the candidate added, computed as the area of a union of boxes over
the grid of all their coordinates, integrated against the candidate's normal density by quadrature.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pytest
from cosy.core.tree import Tree
from scipy import integrate
from scipy.stats import norm

from bayesian_optimization.acquisition_function import ExpectedHypervolumeImprovement

# ---------------------------------------------------------------------------
# A posterior placed by hand, and the oracle
# ---------------------------------------------------------------------------


class PlacedPosterior:
    """A stand-in for the surrogate that reports a chosen ``(m, s)`` per term."""

    def __init__(self, posterior: dict[Any, tuple[float, float]]) -> None:
        self.posterior = posterior

    def predict(self, X: Any, return_std: bool = False) -> Any:
        pairs = [self.posterior[t] for t in X]
        mean = np.array([m for m, _ in pairs], dtype=float)
        if not return_std:
            return mean
        return mean, np.array([s for _, s in pairs], dtype=float)


def union_area(boxes: list[tuple[float, float, float, float]]) -> float:
    """The area of a union of axis-parallel boxes ``(x0, x1, y0, y1)``.

    Every cell between two consecutive coordinates of the boxes lies wholly inside some box or
    wholly outside all of them, so the area is the sum of the cells whose centre some box covers.
    """
    xs = sorted({x for x0, x1, _, _ in boxes for x in (x0, x1)})
    ys = sorted({y for _, _, y0, y1 in boxes for y in (y0, y1)})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        for low, high in zip(ys, ys[1:]):
            x, y = (left + right) / 2, (low + high) / 2
            if any(x0 <= x <= x1 and y0 <= y <= y1 for x0, x1, y0, y1 in boxes):
                area += (right - left) * (high - low)
    return area


def hypervolume(points: list[tuple[float, float]], reference: tuple[float, float]) -> float:
    """The area the pairs ``(value, cost)`` dominate within the reference.

    A pair dominates the values up to its own and the costs from its own up to the reference.
    """
    reference_value, reference_cost = reference
    return union_area(
        [
            (reference_value, v, c, reference_cost)
            for v, c in points
            if v > reference_value and c < reference_cost
        ]
    )


def expected_improvement_by_quadrature(
    front: list[tuple[float, float]],
    reference: tuple[float, float],
    cost: float,
    mean: float,
    deviation: float,
) -> float:
    """``E[HV(front + (V, cost)) - HV(front)]`` for ``V ~ N(mean, deviation^2)``, by quadrature."""
    base = hypervolume(front, reference)

    def gain(value: float) -> float:
        return hypervolume([*front, (value, cost)], reference) - base

    if deviation == 0.0:
        return gain(mean)
    density = norm(mean, deviation).pdf
    low, high = mean - 12 * deviation, mean + 12 * deviation
    # The gain is piecewise linear in the value, with kinks at the front's values and the
    # reference value.
    kinks = {reference[0], *(value for value, _ in front)}
    edges = sorted({low, high, *(kink for kink in kinks if low < kink < high)})
    return sum(
        float(
            integrate.quad(
                lambda v: gain(v) * density(v), a, b, epsabs=1e-13, epsrel=1e-11, limit=200
            )[0]
        )
        for a, b in zip(edges, edges[1:])
    )


def scored(
    front: list[tuple[float, float]],
    reference: tuple[float, float],
    candidates: list[tuple[float, float, float]],
    known: set[Any] | None = None,
) -> tuple[list[Tree[Any]], dict[Tree[Any], float]]:
    """Score candidates ``(mean, deviation, cost)`` through the public batch path.

    Returns:
        tuple[list[Tree], dict[Tree, float]]: The terms, in the order of ``candidates``, and
            their scores.
    """
    terms = [Tree(f"candidate{index}") for index in range(len(candidates))]
    costs = {term: cost for term, (_, _, cost) in zip(terms, candidates)}
    posterior = PlacedPosterior({term: (m, s) for term, (m, s, _) in zip(terms, candidates)})
    acquisition = ExpectedHypervolumeImprovement(
        posterior, cost=costs.__getitem__, front=front, reference=reference, known_points=known
    )
    return terms, acquisition.evaluate_batch(terms)


# ---------------------------------------------------------------------------
# The score is the expected hypervolume improvement
# ---------------------------------------------------------------------------


def test_the_score_is_the_expected_hypervolume_improvement_on_random_fronts_with_ties():
    """Random observed pairs on a coarse grid, so that values and costs tie, against the oracle.

    The candidates' costs fall between, on and past the front's costs and the reference cost, and
    a third of the candidates are certain.
    """
    rng = random.Random(7)
    compared = nonzero = 0
    for _trial in range(60):
        front = [
            (rng.choice([0.1, 0.3, 0.5, 0.7, 0.9, rng.random()]), float(rng.randint(1, 9)))
            for _ in range(6)
        ]
        reference = (rng.choice([0.0, 0.2]), rng.choice([8.0, 10.0]))
        candidates = []
        for _ in range(8):
            cost = rng.choice(
                [float(rng.randint(0, 11)), rng.uniform(0, 11), *(c for _, c in front)]
            )
            deviation = 0.0 if rng.random() < 1 / 3 else rng.uniform(0.01, 0.4)
            candidates.append((rng.uniform(-0.2, 1.2), deviation, cost))
        terms, scores = scored(front, reference, candidates)
        for term, (mean, deviation, cost) in zip(terms, candidates):
            expected = expected_improvement_by_quadrature(front, reference, cost, mean, deviation)
            assert scores[term] == pytest.approx(expected, rel=1e-7, abs=1e-10), (
                front,
                reference,
                mean,
                deviation,
                cost,
            )
            compared += 1
            nonzero += expected > 1e-6
    assert compared == 480
    # The control, counted by the oracle alone (262 of the 480 for this seed): the comparison is
    # not one of zeros.
    assert nonzero >= 200


def test_with_no_front_the_score_is_the_width_times_the_improvement_over_the_reference():
    """Nothing observed inside the reference: the candidate's box is the improvement.

    The score is ``(r_c - c) EI(r_g)``, here ``6 phi(0)``.
    """
    terms, scores = scored([], (0.0, 10.0), [(0.0, 1.0, 4.0)])
    assert scores[terms[0]] == pytest.approx(6.0 * norm.pdf(0.0))


def test_one_observed_point_splits_the_width_into_two_levels():
    """A point ``(v1, c1)`` above the candidate's cost splits the width at ``c1``.

    The score is ``(c1 - c) EI(r_g) + (r_c - c1) EI(v1)``, computed here by hand.
    """
    mean, deviation = 0.6, 0.2
    terms, scores = scored([(0.5, 3.0)], (0.0, 10.0), [(mean, deviation, 1.0)])

    def ei(level: float) -> float:
        z = (mean - level) / deviation
        return float((mean - level) * norm.cdf(z) + deviation * norm.pdf(z))

    assert scores[terms[0]] == pytest.approx((3.0 - 1.0) * ei(0.0) + (10.0 - 3.0) * ei(0.5))


def test_a_certain_candidate_scores_its_hypervolume_improvement():
    """At zero deviation the expectation is the improvement itself.

    Here a box of 0.8 by 6 less its overlap of 0.5 by 6 with the front's.
    """
    terms, scores = scored([(0.5, 2.0)], (0.0, 10.0), [(0.8, 0.0, 4.0)])
    assert scores[terms[0]] == pytest.approx(0.8 * 6.0 - 0.5 * 6.0)


@pytest.mark.parametrize("cost", [10.0, 10.5, 1e9])
def test_a_candidate_at_or_past_the_reference_cost_adds_nothing(cost):
    """No cost above the reference counts, however good the value.

    Args:
        cost (float): The candidate's cost, at or past the reference cost of 10.
    """
    terms, scores = scored([(0.5, 2.0)], (0.0, 10.0), [(5.0, 1.0, cost)])
    assert scores[terms[0]] == 0.0


def test_a_candidate_the_front_dominates_for_certain_adds_nothing():
    """A certain value no higher than the front's at a cost no lower adds no area."""
    terms, scores = scored(
        [(0.5, 2.0), (0.7, 5.0)], (0.0, 10.0), [(0.7, 0.0, 5.0), (0.5, 0.0, 3.0)]
    )
    assert scores[terms[0]] == 0.0
    assert scores[terms[1]] == 0.0


def test_dominated_and_outside_pairs_change_nothing():
    """Only the non-dominated pairs inside the reference shape the levels.

    The others, dominated or outside the reference point, leave every score where it was.
    """
    front = [(0.4, 2.0), (0.6, 4.0), (0.8, 7.0)]
    noise = [(0.3, 3.0), (0.5, 6.0), (0.4, 2.0), (-1.0, 1.0), (2.0, 12.0), (0.2, 9.5)]
    candidates = [(0.5, 0.1, 1.0), (0.7, 0.3, 3.0), (0.9, 0.05, 6.5), (0.6, 0.0, 0.5)]
    terms, plain = scored(front, (0.0, 10.0), candidates)
    _, noisy = scored(front + noise, (0.0, 10.0), candidates)
    assert [noisy[term] for term in terms] == [plain[term] for term in terms]
    assert all(plain[term] > 0.0 for term in terms), "the control: every candidate improves"


def test_the_score_rises_with_the_mean_and_the_deviation_and_falls_with_the_cost():
    """Monotone in each coordinate: a higher or less certain value, or a lower cost, never loses."""
    rng = random.Random(11)
    front = [(0.3, 2.0), (0.5, 4.0), (0.6, 6.0), (0.85, 8.0)]
    for _trial in range(200):
        mean, deviation, cost = rng.uniform(0, 1), rng.uniform(0, 0.3), rng.uniform(0, 10)
        step = rng.uniform(0.001, 0.5)
        terms, scores = scored(
            front,
            (0.0, 10.0),
            [
                (mean, deviation, cost),
                (mean + step, deviation, cost),
                (mean, deviation + step, cost),
                (mean, deviation, cost + step),
            ],
        )
        base, higher, wider, dearer = (scores[term] for term in terms)
        assert higher >= base
        assert wider >= base
        assert dearer <= base


# ---------------------------------------------------------------------------
# The loop's contract: the floor, and refusals
# ---------------------------------------------------------------------------


def test_a_known_point_scores_below_every_genuine_score():
    """The score is never negative, so the floor sits at -1 and a known point cannot win."""
    candidates = [(0.9, 0.1, 1.0), (0.0, 0.0, 12.0)]
    terms = [Tree(f"candidate{index}") for index in range(2)]
    terms, scores = scored([(0.5, 2.0)], (0.0, 10.0), candidates, known={terms[0]})
    assert scores[terms[0]] == -1.0
    assert scores[terms[1]] == 0.0
    assert ExpectedHypervolumeImprovement.lower_bound == 0.0


@pytest.mark.parametrize("cost", [math.nan, math.inf, -math.inf])
def test_a_cost_that_is_not_finite_is_refused_naming_the_term(cost):
    """A cost is a measurement, and a failed one is refused rather than scored.

    Args:
        cost (float): The cost the cost function answers.
    """
    with pytest.raises(ValueError, match=r"the cost of .*candidate0.* is not finite"):
        scored([(0.5, 2.0)], (0.0, 10.0), [(0.5, 0.1, cost)])


@pytest.mark.parametrize("reference", [(math.nan, 10.0), (0.0, math.inf), (0.0,), "far"])
def test_a_reference_point_that_is_not_two_finite_numbers_is_refused(reference):
    """The reference bounds the measured area, so one without a finite bound is no reference.

    Args:
        reference: What is handed in as the reference point.
    """
    with pytest.raises(ValueError, match="the reference point"):
        ExpectedHypervolumeImprovement(
            PlacedPosterior({}), cost=float, front=[], reference=reference
        )


@pytest.mark.parametrize("cost", [None, "size", 3.0])
def test_a_cost_that_is_not_a_function_is_refused(cost):
    """The cost is read off every term, so it has to be something a term can be handed to.

    Args:
        cost: What is handed in as the cost.
    """
    with pytest.raises(TypeError, match="the cost of a term"):
        ExpectedHypervolumeImprovement(
            PlacedPosterior({}), cost=cost, front=[], reference=(0.0, 1.0)
        )


def test_an_observed_pair_that_is_not_finite_is_refused():
    """The loop observes finite values only, so a pair that is not finite came from elsewhere."""
    with pytest.raises(ValueError, match="an observed pair"):
        ExpectedHypervolumeImprovement(
            PlacedPosterior({}), cost=float, front=[(math.nan, 1.0)], reference=(0.0, 1.0)
        )


def test_the_score_cannot_be_read_off_the_posterior_alone():
    """Without the candidates there are no costs: the posterior-only score names the one to call."""
    acquisition = ExpectedHypervolumeImprovement(
        PlacedPosterior({}), cost=float, front=[], reference=(0.0, 1.0)
    )
    with pytest.raises(TypeError, match="score_terms"):
        acquisition.score(np.array([0.5]), np.array([0.1]))
