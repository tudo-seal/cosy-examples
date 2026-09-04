"""The mechanics of the acceptance checks, and their edges.

``tests/validation/test_bo_diagnostics.py`` holds the reads to the reference numbers of the
diagnostic figures.  This module holds them to their contracts: what they refuse, what they
answer where a statistic does not exist, and how the loop feeds them.  The two are split
because they fail for different reasons.  A wrong formula breaks the first, a wrong signature
or a swallowed edge case breaks this one.
"""
from __future__ import annotations

import csv
import io
import logging

import numpy as np
import pytest
from cosy.core.tree import Tree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from bayesian_optimization import (
    AcquisitionRun,
    BayesianOptimization,
    ExpectedImprovement,
    ProbabilityOfImprovement,
    TraceRecord,
    UpperConfidenceBound,
    kernel_objective_alignment,
    read_calibration,
    read_fit,
    read_frontier,
    read_gram,
    read_trace,
    trace_columns,
    trace_rows,
)
from bayesian_optimization.diagnostics import (
    rank_correlation,
    spread,
    warn_if_exploitation_stalls,
)

# ---------------------------------------------------------------------------
# Fixtures: a plane the tests can place candidates on by hand
# ---------------------------------------------------------------------------


class _PlacedPosterior:
    """A surrogate that answers a chosen ``(m, s)`` at each term.

    The reads are about the plane, and a fitted model reaches only the corners of it that its data
    happens to produce.  This puts a candidate exactly where a test needs it, which is the same
    reason :class:`MarginalPosterior` is a protocol rather than the regressor.
    """

    def __init__(self, placement: dict) -> None:
        self._placement = placement

    def predict(self, X, return_std: bool = False):
        means = np.array([self._placement[term][0] for term in X], dtype=float)
        if not return_std:
            return means
        deviations = np.array([self._placement[term][1] for term in X], dtype=float)
        return means, deviations


def _term(name: str) -> Tree:
    return Tree(name, ())


@pytest.fixture
def fitted_gp():
    x = np.arange(6.0).reshape(-1, 1)
    y = np.array([0.1, 0.4, 0.9, 0.7, 0.3, 0.2])
    return GaussianProcessRegressor(
        kernel=RBF(1.5), alpha=1e-6, optimizer=None, normalize_y=False
    ).fit(x, y)


# ---------------------------------------------------------------------------
# The shared statistics
# ---------------------------------------------------------------------------


def test_a_constant_sequence_has_no_ranks():
    """``None``, not ``nan``: the value has to fail every comparison a caller writes.

    ``nan > 0.5`` and ``nan < 0.5`` are both false, so a threshold check on a ``nan`` reports
    whichever answer the caller happened to write the check for.
    """
    assert rank_correlation([1.0, 2.0, 3.0], [1.0, 1.0, 1.0]) is None
    assert rank_correlation([1.0, 1.0], [1.0, 2.0]) is None
    assert rank_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("first", "second"),
    [([1.0, 2.0], [1.0]), ([1.0], [2.0]), ([1.0, np.inf], [1.0, 2.0])],
)
def test_a_rank_correlation_refuses_what_it_cannot_rank(first, second):
    with pytest.raises(ValueError):
        rank_correlation(first, second)


def test_the_range_of_an_empty_sequence_is_refused():
    with pytest.raises(ValueError, match="empty"):
        spread([])


# ---------------------------------------------------------------------------
# read_gram
# ---------------------------------------------------------------------------


def test_the_gram_read_refuses_a_matrix_that_is_not_a_kernel_matrix():
    with pytest.raises(ValueError, match="square"):
        read_gram(np.ones((2, 3)))
    with pytest.raises(ValueError, match="off-diagonal"):
        read_gram(np.ones((1, 1)))
    with pytest.raises(ValueError, match="not finite"):
        read_gram(np.array([[1.0, np.nan], [np.nan, 1.0]]))


def test_a_non_positive_self_similarity_is_a_broken_kernel_not_a_small_number():
    """It is also where the normalization would divide by zero, and neither gets a substitute."""
    with pytest.raises(ValueError, match="squared norm"):
        read_gram(np.array([[0.0, 0.0], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="squared norm"):
        read_gram(np.array([[-1.0, 0.0], [0.0, 1.0]]))


def test_the_gram_read_reports_asymmetry_rather_than_symmetrising_it_away():
    """A kernel is symmetric by definition, so an asymmetric matrix is a defect to be seen.

    The eigenvalues are still reported, off the symmetrized matrix, because there is no other
    sensible reading of them, but the asymmetry is on the record beside them.

    The second matrix is the one that says *where* it is measured.  With a unit diagonal the
    normalization is the identity and the two readings coincide.  Scaling one term apart, the raw
    asymmetry is 0.5 and the normalized one 0.05, and only the second is a number a caller can
    compare against a fixed tolerance.
    """
    assert read_gram(np.array([[1.0, 0.4], [0.9, 1.0]])).symmetry_error == pytest.approx(0.5)
    assert read_gram(
        np.array([[1.0, 0.4], [0.9, 100.0]])
    ).symmetry_error == pytest.approx(0.05)


def test_the_minimum_eigenvalue_is_the_minimum():
    """A one-sided bound cannot tell the smallest eigenvalue from the largest.

    Everywhere else this field is asserted as ``> -1e-8``, which the maximum satisfies too, so
    the claim it carries, that a value well below zero is a defect, needs a matrix with a known
    spectrum.
    """
    read = read_gram(np.array([[1.0, 0.9], [0.9, 1.0]]))
    assert read.minimum_eigenvalue == pytest.approx(0.1)


def test_a_kernel_that_says_everything_is_alike_has_no_axis():
    """The degenerate case the seriation cannot report on its own.

    With every coordinate equal, ``argsort`` returns the order the terms were passed in, and a
    picture drawn along it shows the caller's own indexing dressed as kernel geometry.  The
    coordinate spread is what says the axis is not there.
    """
    read = read_gram(np.ones((6, 6)))
    assert read.coordinate_spread == 0.0
    assert read.off_diagonal_mean == pytest.approx(1.0)
    # And a healthy matrix has one.
    assert read_gram(np.array([[1.0, 0.5, 0.1], [0.5, 1.0, 0.5], [0.1, 0.5, 1.0]]))\
        .coordinate_spread > 0.0


def test_the_objective_has_to_match_the_matrix():
    with pytest.raises(ValueError, match="pairs them by position"):
        read_gram(np.eye(3) + 0.1, objective=[1.0, 2.0])


def test_the_diagonal_spread_survives_the_normalization_that_removes_it():
    """The one measurement taken before normalizing, which is why it is taken at all."""
    matrix = np.array([[1.0, 1.0, 1.0], [1.0, 4.0, 2.0], [1.0, 2.0, 100.0]])
    read = read_gram(matrix)
    assert read.diagonal_spread == pytest.approx(100.0)
    assert read.diagonal == (1.0, 4.0, 100.0)
    assert np.allclose(np.diag(read.normalized_matrix), 1.0)


# ---------------------------------------------------------------------------
# read_fit and the alignment
# ---------------------------------------------------------------------------


def test_the_fit_read_refuses_a_mismatched_or_degenerate_held_out_set(fitted_gp):
    with pytest.raises(ValueError, match="by position"):
        read_fit(fitted_gp, [[0.0], [1.0]], [0.1])
    with pytest.raises(ValueError, match="at least two"):
        read_fit(fitted_gp, [[0.0]], [0.1])
    with pytest.raises(ValueError, match="not finite"):
        read_fit(fitted_gp, [[0.0], [1.0]], [0.1, np.nan])


def test_a_constant_objective_leaves_the_slope_undefined(fitted_gp):
    """No line through a constant x has a slope, and ``None`` says so where a fit would not."""
    read = read_fit(fitted_gp, [[0.5], [1.5], [2.5]], [0.3, 0.3, 0.3])
    assert read.regression_slope is None
    assert read.rank_correlation is None
    assert read.objective_spread == 0.0


def test_the_slope_is_the_slope_of_predicted_on_true():
    """Which axis is which, on a fit far enough from the diagonal to tell.

    On a near-perfect fit both orientations land near one, so a tolerance wide enough to accept
    the right answer accepts the wrong one as well.  Here they are 0.91 and 1.045.
    """

    class _Fixed:
        def predict(self, X, return_std: bool = False):
            return np.array([0.1, 1.3, 1.7, 3.4, 3.6])

    read = read_fit(_Fixed(), [[float(i)] for i in range(5)], [0.0, 1.0, 2.0, 3.0, 4.0])
    assert read.regression_slope == pytest.approx(0.91, abs=0.005)


def test_a_collapsed_prediction_is_not_a_constant_prediction():
    """``predictions_constant`` is exact, and the difference is the point of it.

    A surrogate whose kernel transfers nothing predicts the prior mean everywhere, which comes
    out as values around 1e-87: collapsed, and not equal.  A tolerance here would report them as
    one value and hide that the read turns on the *spread* instead, which is where a collapsed
    surrogate is caught.  It can still rank its candidates within a vanishing span, so the rank
    correlation means nothing unless the spread is read beside it.
    """

    class _Collapsed:
        def predict(self, X, return_std: bool = False):
            return np.array([1.2e-88, 9.9e-88, 2.3e-87, 2.2e-87])

    read = read_fit(_Collapsed(), [[float(i)] for i in range(4)], [0.1, 0.4, 0.9, 0.7])
    assert read.predictions_constant is False
    assert 0.0 < read.prediction_spread < 1e-80

    class _Constant:
        def predict(self, X, return_std: bool = False):
            return np.full(4, 0.5)

    constant = read_fit(_Constant(), [[float(i)] for i in range(4)], [0.1, 0.4, 0.9, 0.7])
    assert constant.predictions_constant is True
    assert constant.prediction_spread == 0.0


def test_a_surrogate_that_answers_badly_is_reported_rather_than_read(fitted_gp):
    class _Broken:
        def predict(self, X, return_std: bool = False):
            return np.array([np.nan] * len(X))

    with pytest.raises(ValueError, match="not finite"):
        read_fit(_Broken(), [[0.0], [1.0]], [0.1, 0.2])

    class _WrongLength:
        def predict(self, X, return_std: bool = False):
            return np.array([0.0])

    with pytest.raises(ValueError, match="predictions"):
        read_fit(_WrongLength(), [[0.0], [1.0]], [0.1, 0.2])


def test_the_alignment_needs_more_than_one_pair():
    with pytest.raises(ValueError, match="at most one pair"):
        kernel_objective_alignment(np.eye(2) + 0.1, [1.0, 2.0])


def test_the_alignment_is_one_when_the_kernel_is_the_objective_distance():
    """The definition read back: build the kernel *from* ``-|q(t) - q(t')|`` and it aligns fully."""
    values = [0.0, 1.0, 2.5, 4.0]
    matrix = np.array(
        [[-abs(a - b) for b in values] for a in values], dtype=float
    ) + 10.0  # a positive diagonal: the statistic only ranks the off-diagonal entries
    assert kernel_objective_alignment(matrix, values) == pytest.approx(1.0)


def test_the_alignment_ranks_the_distinct_pairs_and_not_the_diagonal():
    """The alignment ranks the pairs of distinct members, and a diagonal would change the answer.

    The obvious fixture cannot see this: a kernel built from the objective distance has a constant
    diagonal that ranks alongside the closest pairs, so including it leaves the correlation at one.
    Here the self-similarities are set below every off-diagonal entry, so a statistic that ranked
    them too would come out somewhere else entirely.
    """
    values = [0.0, 1.0, 2.5, 4.0]
    matrix = np.array([[-abs(a - b) for b in values] for a in values], dtype=float) + 10.0
    np.fill_diagonal(matrix, -100.0)

    assert kernel_objective_alignment(matrix, values) == pytest.approx(1.0)

    rows, columns = np.triu_indices(4, k=0)
    with_diagonal = rank_correlation(
        matrix[rows, columns],
        [-abs(values[r] - values[c]) for r, c in zip(rows, columns, strict=True)],
    )
    assert with_diagonal is not None
    assert with_diagonal < 0.7, "the fixture has to be one the diagonal would spoil"


# ---------------------------------------------------------------------------
# read_calibration
# ---------------------------------------------------------------------------


def test_the_calibration_read_needs_a_fitted_surrogate():
    with pytest.raises(AttributeError, match="fitted"):
        read_calibration(GaussianProcessRegressor(kernel=RBF(1.0)))


def test_leaving_one_of_one_out_leaves_nothing():
    surrogate = GaussianProcessRegressor(
        kernel=RBF(1.0), alpha=1e-6, optimizer=None
    ).fit(np.array([[0.0]]), np.array([1.0]))
    with pytest.raises(ValueError, match="no other one"):
        read_calibration(surrogate)


_CALIBRATION_X = np.arange(7.0).reshape(-1, 1)
_CALIBRATION_Y = np.array([0.10, 0.42, 0.91, 0.73, 0.35, 0.22, 0.58])


def _calibration(targets, kernel, *, normalize_y=False, alpha=0.0, optimizer=None):
    return read_calibration(
        GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, optimizer=optimizer, normalize_y=normalize_y
        ).fit(_CALIBRATION_X, targets)
    )


def test_the_three_residual_fields_are_one_decomposition():
    """``residual = z * s``, the identity that says which field is which.

    ``predicted_deviations`` is the denominator of the standardization and is otherwise never
    read, so nothing else would notice it being inverted, which leaves the numbers plausible and
    useless as standard deviations.
    """
    read = _calibration(_CALIBRATION_Y, RBF(1.5))
    assert np.allclose(
        np.asarray(read.residuals),
        np.asarray(read.standardized_residuals) * np.asarray(read.predicted_deviations),
    )
    assert all(value > 0.0 for value in read.predicted_deviations)


def test_the_spread_about_zero_and_the_spread_about_the_mean_are_kept_apart():
    """A bias of fifty standard deviations, which one of the two fields cannot see.

    The calibration rule compares the standardized residuals against the standard normal, which
    is centered at zero.  A surrogate whose residuals all sit near +50 is overconfident by that
    rule, while their spread *about their own mean* is a third of one, the underconfident
    reading.  Both numbers are reported, and the rule is attached to the first.
    """
    shifted = _calibration(_CALIBRATION_Y + 50.0, RBF(0.05))
    assert min(shifted.standardized_residuals) > 40.0
    assert shifted.root_mean_square > 40.0
    assert shifted.standard_deviation < 1.0
    assert shifted.positive_fraction == 1.0


@pytest.mark.parametrize(
    "factor",
    [
        np.array([[1.0, 0.0], [1.0, 1e-160]]),  # a pivot that underflows: inverse runs to inf
        np.array([[1.0, 0.0], [0.0, 0.0]]),  # a zero pivot: inverse holds nan
    ],
)
def test_a_numerically_singular_kernel_matrix_is_refused(factor):
    """The documented failure of the leave-one-out variance, exercised in both of its forms.

    It is not the failure the guard was first written for.  ``1 / [K^-1]_ii`` is a variance, so
    the guard asked whether the diagonal of the inverse was positive.  It always is: the
    inverse of a Cholesky product is positive definite.  What a collapsing factor actually
    produces is ``inf`` or ``nan``, and those pass ``> 0``, so the guard as written could not
    fire, and the mutation probe found it by the only symptom a dead branch has, that removing
    it changes nothing.

    Both factors are set by hand.  The route to them is real, near-duplicate rows under too small
    a diagonal term, but which machine reaches it depends on its rounding, and a test that hunts
    for that is a test of the machine.
    """
    fitted = GaussianProcessRegressor(
        kernel=RBF(1.5), alpha=0.0, optimizer=None, normalize_y=False
    ).fit(_CALIBRATION_X, _CALIBRATION_Y)
    fitted.L_ = factor
    fitted.alpha_ = np.array([1.0, 1.0])
    fitted.y_train_ = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="numerically singular"):
        read_calibration(fitted)


def test_a_calibration_measures_the_objectives_scale_against_the_kernels_amplitude():
    """The finding behind :attr:`CalibrationRead.targets_normalized`.

    A Gaussian process's posterior variance does not depend on the values it observed, so the
    denominator of a standardized residual is a function of the kernel alone while the numerator
    is in the units of the objective.  Scale the objective by ten under a kernel whose amplitude
    is fixed and every residual grows by ten: the read reports overconfidence that is nothing but
    a change of units.

    So a spread far above one is a statement about the model only once the scale is pinned, and
    the test states both ways of pinning it, the normalization the framework offers and the
    model selection that fits the amplitude.
    """
    scaled = [
        _calibration(scale * _CALIBRATION_Y, RBF(1.5)).maximum_absolute
        for scale in (1.0, 10.0, 100.0)
    ]
    assert scaled == pytest.approx([1.640, 16.399, 163.991], rel=1e-3)


def test_normalizing_the_targets_pins_the_scale():
    """``normalize_y`` sets the target variance to one, and the reading stops moving with units."""
    normalized = [
        _calibration(
            scale * _CALIBRATION_Y + 7.0, RBF(1.5), normalize_y=True
        ).maximum_absolute
        for scale in (1.0, 10.0, 100.0)
    ]
    assert normalized == pytest.approx([normalized[0]] * 3, rel=1e-9)
    assert _calibration(_CALIBRATION_Y, RBF(1.5), normalize_y=True).targets_normalized
    assert not _calibration(_CALIBRATION_Y, RBF(1.5)).targets_normalized


def test_a_fitted_amplitude_pins_the_scale_and_pins_it_better():
    """Model selection is the repair for a scale-dependent reading, and here is what it repairs.

    An amplitude the marginal likelihood can move absorbs the scale of the objective outright, and
    it lands closer to a calibrated spread than the fixed-amplitude normalization does.  Which is
    why ``BayesianOptimization`` warns when a kernel offers the optimizer nothing to fit: its
    default kernel declares no hyperparameters, and then ``normalize_y`` is the only thing between
    the residuals and the units of the objective.
    """
    fitted = [
        _calibration(
            scale * _CALIBRATION_Y,
            ConstantKernel(1.0, (1e-5, 1e5)) * RBF(1.5, "fixed"),
            alpha=1e-10,
            optimizer="fmin_l_bfgs_b",
        ).maximum_absolute
        for scale in (1.0, 10.0, 100.0)
    ]
    # Not exact, unlike the normalization above: the amplitude is found by an optimizer, and it
    # converges to its own tolerance rather than to a closed form.
    assert fitted == pytest.approx([fitted[0]] * 3, rel=1e-4)
    assert fitted[0] == pytest.approx(2.11, abs=0.01)

    normalized = _calibration(
        _CALIBRATION_Y, RBF(1.5), normalize_y=True
    ).maximum_absolute
    assert abs(fitted[0] - 2.0) < abs(normalized - 2.0)


# ---------------------------------------------------------------------------
# read_frontier
# ---------------------------------------------------------------------------


def test_the_frontier_read_needs_a_population():
    posterior = _PlacedPosterior({_term("a"): (1.0, 0.5)})
    acquisition = ExpectedImprovement(posterior, incumbent=0.0)
    with pytest.raises(ValueError, match="population is empty"):
        read_frontier(AcquisitionRun(acquisition, _term("a"), ()))


def test_a_dominated_pick_is_reported_as_dominated():
    """One member up and to the right of the pick, and the read says so under both readings."""
    pick, better, worse = _term("pick"), _term("better"), _term("worse")
    posterior = _PlacedPosterior(
        {pick: (1.0, 0.5), better: (2.0, 0.8), worse: (0.1, 0.1)}
    )
    acquisition = ExpectedImprovement(posterior, incumbent=0.0)

    read = read_frontier(AcquisitionRun(acquisition, pick, (pick, better, worse)))
    assert read.dominating_members == 1
    assert not read.on_frontier
    assert read.higher_scored_members == 1
    assert read.pick_in_population


def test_probability_of_improvement_parts_the_two_frontier_readings():
    """The reservation the read documents, constructed.

    Above the threshold ``Phi((m - theta) / s)`` falls in the deviation, so a member with a larger
    mean *and* a larger deviation can score below the pick.  Dominance then reports a failure that
    the scoring does not, and a run under probability of improvement is read by the scoring.
    """
    pick, dominator = _term("pick"), _term("dominator")
    posterior = _PlacedPosterior({pick: (2.0, 0.1), dominator: (2.1, 5.0)})
    pi = ProbabilityOfImprovement(posterior, incumbent=1.0)

    assert pi(pick) > pi(dominator)
    read = read_frontier(AcquisitionRun(pi, pick, (pick, dominator)))
    assert read.dominating_members == 1
    assert not read.on_frontier
    assert read.higher_scored_members == 0

    # Expected improvement rises in both coordinates, so the two readings cannot disagree there.
    ei = ExpectedImprovement(posterior, incumbent=1.0)
    ei_read = read_frontier(AcquisitionRun(ei, pick, (pick, dominator)))
    assert ei_read.dominating_members == 1
    assert ei_read.higher_scored_members == 1


def test_a_tie_on_one_axis_still_dominates():
    """The boundary of dominance: at least as large in both coordinates, strictly larger in one.

    Every other fixture separates its members on both coordinates at once, where ``>=`` and ``>``
    agree.  A member that matches the pick's deviation exactly and beats its mean dominates, and a
    read that required both to be strict would call the pick undominated.
    """
    pick, equal_deviation = _term("pick"), _term("taller")
    posterior = _PlacedPosterior({pick: (1.0, 0.5), equal_deviation: (2.0, 0.5)})
    read = read_frontier(
        AcquisitionRun(
            ExpectedImprovement(posterior, incumbent=0.0), pick, (pick, equal_deviation)
        )
    )
    assert read.dominating_members == 1
    assert not read.on_frontier

    # A member equal on *both* axes ties rather than dominates: it is the pick's own copy.
    twin = _term("twin")
    same = _PlacedPosterior({pick: (1.0, 0.5), twin: (1.0, 0.5)})
    tied = read_frontier(
        AcquisitionRun(ExpectedImprovement(same, incumbent=0.0), pick, (pick, twin))
    )
    assert tied.dominating_members == 0
    assert tied.higher_scored_members == 0


def test_the_two_spreads_are_the_ranges_of_the_two_coordinates():
    """Which range is which, on a population where they differ.

    The collapsed fixture puts both at zero, where a swap is invisible, and the two coordinates
    mean opposite things: a population spread in the mean and pinned in the deviation is the
    reverse diagnosis of one spread in the deviation and pinned in the mean.
    """
    pick, wide, narrow = _term("pick"), _term("wide"), _term("narrow")
    posterior = _PlacedPosterior(
        {pick: (0.0, 0.1), wide: (2.0, 0.3), narrow: (1.0, 0.9)}
    )
    read = read_frontier(
        AcquisitionRun(
            ExpectedImprovement(posterior, incumbent=0.0), pick, (pick, wide, narrow)
        )
    )
    assert read.mean_spread == pytest.approx(2.0)
    assert read.deviation_spread == pytest.approx(0.8)
    assert read.mean_spread == pytest.approx(
        max(m for m, _, _ in read.points) - min(m for m, _, _ in read.points)
    )
    assert read.deviation_spread == pytest.approx(
        max(s for _, s, _ in read.points) - min(s for _, s, _ in read.points)
    )


def test_the_frontier_read_refuses_what_is_not_a_term():
    """A kernel handed a string may raise, or may hash it and answer a number.

    The second is the case the acquisition layer already refuses everywhere else: a number that
    reached a diagnostic that way is indistinguishable from a measurement.
    """
    pick = _term("pick")
    posterior = _PlacedPosterior({pick: (1.0, 0.5)})
    acquisition = ExpectedImprovement(posterior, incumbent=0.0)

    with pytest.raises(TypeError, match="scores terms"):
        read_frontier(AcquisitionRun(acquisition, pick, (pick, "not a term")))
    with pytest.raises(TypeError, match="scores terms"):
        read_frontier(AcquisitionRun(acquisition, "not a term", (pick,)))


def test_a_pick_the_final_population_no_longer_holds_is_still_placed():
    """cosy's driver answers with the best of the whole run, which a later generation may drop."""
    pick = _term("dropped")
    members = [_term("a"), _term("b")]
    posterior = _PlacedPosterior(
        {pick: (3.0, 1.0), members[0]: (1.0, 0.2), members[1]: (0.5, 0.1)}
    )
    read = read_frontier(
        AcquisitionRun(ExpectedImprovement(posterior, incumbent=0.0), pick, tuple(members))
    )

    assert not read.pick_in_population
    assert read.on_frontier
    assert read.size == 2


def test_known_points_are_counted_because_they_were_not_scored_at_their_position():
    """They ran under the floor, so their place in the plane does not explain their fate."""
    pick, known = _term("pick"), _term("known")
    posterior = _PlacedPosterior({pick: (1.0, 0.5), known: (2.0, 0.9)})
    acquisition = UpperConfidenceBound(
        posterior, beta=2.0, known_points={known}
    )
    read = read_frontier(AcquisitionRun(acquisition, pick, (pick, known)))
    assert read.known_members == 1
    # The score reported is the genuine one, not the floor: otherwise a pick and a member would be
    # compared by two different maps.
    assert read.points[1][2] == pytest.approx(2.0 + 2.0 * 0.9)


# ---------------------------------------------------------------------------
# read_trace
# ---------------------------------------------------------------------------


def _record(
    iteration, acquisition, deviation, best, *, incumbent=None, fallback=False, distinct=None
):
    """One row.  ``incumbent`` defaults to ``best``, which is a pass that improved on nothing."""
    return TraceRecord(
        iteration=iteration,
        acquisition=acquisition,
        mean=0.5,
        deviation=deviation,
        incumbent=best if incumbent is None else incumbent,
        observed=best,
        best=best,
        fallback_used=fallback,
        evaluations=3 + iteration + 1,
        distinct_observations=distinct if distinct is not None else 3 + iteration + 1,
    )


def _run(bests, acquisitions, deviations, *, start=0.0):
    """A trace of a run whose best observed value follows ``bests``.

    Each row's incumbent is what stood before that pass, so the improvements come out as the run
    made them.  That is the field the read counts, and the one a row-against-row comparison
    would get wrong at the first pass.
    """
    previous = start
    records = []
    for index, best in enumerate(bests):
        records.append(
            _record(
                index, acquisitions[index], deviations[index], best, incumbent=previous
            )
        )
        previous = best
    return records


def test_an_empty_trace_has_no_reading():
    with pytest.raises(ValueError, match="holds none"):
        read_trace([])


def test_a_healthy_trace_decays_and_improves():
    read = read_trace(
        _run([0.1, 0.3, 0.5, 0.5], [0.40, 0.25, 0.10, 0.02], [0.50, 0.40, 0.30, 0.20])
    )

    assert read.passes == 4
    assert read.acquisition_trend == pytest.approx(-1.0)
    assert read.acquisition_first == pytest.approx(0.40)
    assert read.acquisition_last == pytest.approx(0.02)
    assert read.improvements == 3
    assert read.stalled_passes == 1
    assert read.fallbacks == 0
    assert read.distinct_fraction == pytest.approx(1.0)
    assert read.best_trace == (0.1, 0.3, 0.5, 0.5)


def test_the_first_pass_counts_as_an_improvement_when_it_is_one():
    """A run whose first pass beats the initial design, and nothing after it does.

    The row-against-row reading would call this a run without a single improvement, because the
    first row has no predecessor to beat.  The incumbent the row carries is the predecessor.
    """
    read = read_trace(_run([0.9, 0.9, 0.9], [0.3, 0.2, 0.1], [0.4, 0.4, 0.4], start=0.2))
    assert read.improvements == 1
    assert read.stalled_passes == 2


def test_a_flat_trace_has_no_trend_and_says_so():
    """A never decaying acquisition is the degenerate landscape, and the flat case has no trend."""
    read = read_trace([_record(index, 0.3, 0.5, 0.1) for index in range(4)])
    assert read.acquisition_trend is None
    assert read.improvements == 0
    # No pass ever improved, so the whole run is the stall.
    assert read.stalled_passes == 4


def test_one_pass_is_a_point_and_a_point_has_no_trend():
    read = read_trace([_record(0, 0.3, 0.5, 0.1)])
    assert read.acquisition_trend is None
    assert read.passes == 1
    assert read.stalled_passes == 1
    assert read_trace(_run([0.5], [0.3], [0.5])).stalled_passes == 0


def test_a_pinned_deviation_is_visible_as_a_pinned_deviation():
    """Zero and the prior maximum are the two pins, and the spread catches both."""
    exploiting = read_trace([_record(index, 0.1, 0.0, 0.1) for index in range(3)])
    assert exploiting.deviation_spread == 0.0
    assert exploiting.deviation_maximum == 0.0

    exploring = read_trace([_record(index, 0.1, 1.0, 0.1) for index in range(3)])
    assert exploring.deviation_spread == 0.0
    assert exploring.deviation_minimum == 1.0


def test_a_varying_deviation_is_summarised_by_its_range_and_its_ends():
    """The same three fields on a trace that is *not* pinned.

    The pinned cases above cannot tell a range from any other spread statistic, nor a minimum from
    a maximum, because constant data makes all of them equal.  This one separates them: the range
    is the range, and the two ends are the two ends, which is what "pinned at zero or at the
    prior maximum" needs in order to say which pin it was.
    """
    read = read_trace(
        [
            _record(0, 0.3, 0.2, 0.1),
            _record(1, 0.2, 0.9, 0.1),
            _record(2, 0.1, 0.5, 0.1),
        ]
    )
    assert read.deviation_minimum == pytest.approx(0.2)
    assert read.deviation_maximum == pytest.approx(0.9)
    assert read.deviation_spread == pytest.approx(0.7)
    assert read.deviation_spread == pytest.approx(
        read.deviation_maximum - read.deviation_minimum
    )


def test_a_run_of_fallbacks_is_a_run_of_random_search():
    read = read_trace([_record(index, 0.1, 0.4, 0.1, fallback=True) for index in range(3)])
    assert read.fallbacks == 3


def test_an_acquisition_at_its_floor_is_counted():
    read = read_trace(
        [_record(0, 0.3, 0.4, 0.1), _record(1, 0.0, 0.4, 0.1), _record(2, 0.0, 0.4, 0.1)]
    )
    assert read.acquisition_at_zero == 2


def test_a_negative_acquisition_is_not_an_acquisition_at_zero():
    """At zero means at zero, not at or below.

    The upper confidence bound is unbounded below, so a healthy UCB run carries negative values
    at its picks.  Counting those as floor hits would report the failure mode of a *bounded*
    acquisition on a run that cannot have it.
    """
    read = read_trace(
        [_record(0, -2.0, 0.4, 0.1), _record(1, -1.0, 0.4, 0.1), _record(2, 0.5, 0.4, 0.1)]
    )
    assert read.acquisition_at_zero == 0


def test_a_repeated_evaluation_shows_in_the_distinct_fraction():
    """Under the rejection path the fraction is one, and below one says a duplicate got through.

    The counts come off the dataset, so they can disagree, which is the whole content of the
    field.  Derived from the pass number instead, as an earlier version did, the fraction was one
    on every trace a loop could produce, including this one.
    """
    records = [
        TraceRecord(
            iteration=index,
            acquisition=0.3,
            mean=0.5,
            deviation=0.4,
            incumbent=0.1,
            observed=0.1,
            best=0.1,
            fallback_used=False,
            evaluations=5 + index,
            distinct_observations=4,
        )
        for index in range(2)
    ]
    assert read_trace(records).distinct_fraction == pytest.approx(4 / 6)


def test_a_trace_that_contradicts_itself_is_refused():
    """Every real column is checked, not only the three the summary averages.

    A non-finite incumbent is the one that hides: ``best > nan`` is false, so the improvement
    count would quietly drop the passes it touches instead of reporting a broken trace.
    """
    with pytest.raises(ValueError, match="incumbent that is not finite"):
        read_trace([_record(0, 0.3, 0.4, 0.1, incumbent=float("nan"))])
    with pytest.raises(ValueError, match="mean that is not finite"):
        read_trace(
            [
                TraceRecord(0, 0.3, float("nan"), 0.4, 0.1, 0.1, 0.1, False, 4, 4),
            ]
        )
    with pytest.raises(ValueError, match="more distinct terms than evaluations"):
        read_trace([TraceRecord(0, 0.3, 0.5, 0.4, 0.1, 0.1, 0.1, False, 2, 3)])
    with pytest.raises(ValueError, match="fewer than one evaluation"):
        read_trace([TraceRecord(0, 0.3, 0.5, 0.4, 0.1, 0.1, 0.1, False, 0, 0)])


def test_the_trace_writes_as_a_table():
    records = [_record(0, 0.3, 0.4, 0.1), _record(1, 0.2, 0.4, 0.2)]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(trace_columns())
    writer.writerows(trace_rows(records))

    lines = buffer.getvalue().strip().splitlines()
    assert lines[0].split(",")[:3] == ["iteration", "acquisition", "mean"]
    assert len(lines) == 3
    assert len(trace_columns()) == len(trace_rows(records)[0])


# ---------------------------------------------------------------------------
# The exploitation warning
# ---------------------------------------------------------------------------


def test_the_exploitation_warning_fires_at_the_floor_and_not_above_it(caplog):
    logger = logging.getLogger("test.exploitation")
    with caplog.at_level(logging.WARNING, logger="test.exploitation"):
        assert warn_if_exploitation_stalls(
            logger,
            iteration=3,
            acquisition_value=0.0,
            lower_bound=0.0,
            acquisition_name="ExpectedImprovement",
        )
        assert not warn_if_exploitation_stalls(
            logger,
            iteration=4,
            acquisition_value=1e-12,
            lower_bound=0.0,
            acquisition_name="ExpectedImprovement",
        )
        # No lower bound, no such moment: UCB is unbounded below.
        assert not warn_if_exploitation_stalls(
            logger,
            iteration=5,
            acquisition_value=-4.0,
            lower_bound=None,
            acquisition_name="UpperConfidenceBound",
        )
    assert len(caplog.records) == 1
    assert "read_calibration" in caplog.records[0].getMessage()


# ---------------------------------------------------------------------------
# What the loop hands the reads
# ---------------------------------------------------------------------------


class _ChainOptimizer:
    """An acquisition maximizer over a fixed candidate set, with a final population.

    It carries both interfaces on purpose: ``evolutionary_best`` is what a maximization needs,
    and ``evolutionary_stream`` is the extra the frontier read asks for.
    """

    def __init__(self, candidates) -> None:
        self.candidates = list(candidates)
        self.population_size = 3

    def _best(self, objective, mode):
        scores = objective(self.candidates) if mode == "batch" else {
            candidate: objective(candidate) for candidate in self.candidates
        }
        return max(scores, key=lambda candidate: scores[candidate])

    def evolutionary_best(self, query, objective, mode="auto"):
        return self._best(objective, mode)

    def evolutionary_stream(self, query, objective, mode="auto"):
        """Yield one generation, shaped exactly like cosy's ``EAState``.

        Every field the real driver carries is here, because the recording of the inner run's
        trajectory reads them: a stub that carried only ``population`` and ``best`` would let a
        caller of ``maximize_with_population`` pass the tests and fail on the real driver.
        """
        best = self._best(objective, mode)
        population = self.candidates[:3]
        scores = objective(population) if mode == "batch" else {
            candidate: objective(candidate) for candidate in population
        }

        class _State:
            generation = 0
            last_improvement = 0

        state = _State()
        state.offspring = []
        state.population = population
        state.fitness = dict(scores)
        state.best = best
        state.best_fitness = max(scores.values())
        yield state


@pytest.fixture
def toy_loop():
    terms = [Tree(f"t{index}", ()) for index in range(6)]
    values = {term: 0.1 * index for index, term in enumerate(terms)}

    class _CountingKernel(RBF):
        pass

    bo = BayesianOptimization(
        None,
        optimizer=_ChainOptimizer(terms),
        kernel=_IndexKernel(terms),
        kernel_optimizer=None,
        gp_normalize_y=False,
        seed=0,
    )
    return bo, terms, values


class _IndexKernel(RBF):
    """A squared exponential over the position of a term in a fixed list.

    The loop's terms are not vectors, and this is the smallest kernel that makes them behave like
    one so that the loop can be exercised end to end without a search space.
    """

    def __init__(self, terms=None, length_scale=1.5) -> None:
        super().__init__(length_scale=length_scale)
        self.terms = terms

    @property
    def requires_vector_input(self) -> bool:
        return False

    def _index(self, X):
        order = {term: position for position, term in enumerate(self.terms)}
        return np.array([[float(order[term])] for term in X])

    def __call__(self, X, Y=None, eval_gradient=False):
        return super().__call__(
            self._index(X), None if Y is None else self._index(Y), eval_gradient
        )

    def diag(self, X):
        return np.ones(len(X))



def test_the_loop_records_the_pick_in_the_plane(toy_loop):
    """``m`` and ``s`` at the pick reach the diagnostics the trace is written from."""
    bo, terms, values = toy_loop
    bo.initialize(x0=terms[:3], y0=[values[term] for term in terms[:3]])
    suggestion = bo.suggest()

    diagnostics = suggestion.diagnostics
    assert diagnostics is not None
    assert "mean_at_pick" in diagnostics
    assert "deviation_at_pick" in diagnostics
    assert diagnostics["deviation_at_pick"] > 0.0


def test_the_trace_columns_carry_what_the_diagnostics_measured(toy_loop):
    """Each column tied back to the key it is written from, by value.

    Two numbers of the same shape sit next to each other here, the posterior mean and the
    posterior deviation at the pick, and a table that swaps them stays perfectly readable while
    every deviation reading afterwards describes the mean.  Nothing but this identity catches it.
    """
    bo, terms, values = toy_loop
    bo.initialize(x0=terms[:3], y0=[values[term] for term in terms[:3]])
    suggestion = bo.suggest()
    diagnostics = suggestion.diagnostics
    assert diagnostics is not None
    bo.observe(suggestion.candidate, values[suggestion.candidate])

    row = bo.trace[0]
    assert row.iteration == diagnostics["iteration"]
    assert row.acquisition == suggestion.acquisition_value
    assert row.mean == diagnostics["mean_at_pick"]
    assert row.deviation == diagnostics["deviation_at_pick"]
    assert row.incumbent == diagnostics["incumbent"]
    assert row.fallback_used == diagnostics["fallback_used"]
    assert row.observed == pytest.approx(values[suggestion.candidate])
    assert row.evaluations == 4
    assert row.distinct_observations == 4


def test_the_loop_builds_a_trace_that_reads(toy_loop):
    bo, terms, values = toy_loop
    result = bo.optimize(lambda term: values[term], budget=3, x0=terms[:3],
                         y0=[values[term] for term in terms[:3]])

    trace = result["trace"]
    assert len(trace) == 3
    assert [record.iteration for record in trace] == [0, 1, 2]
    assert trace[-1].evaluations == trace[-1].distinct_observations == 6

    read = read_trace(trace)
    assert read.passes == 3
    assert read.distinct_fraction == pytest.approx(1.0)
    assert read.best_trace[-1] == pytest.approx(max(values.values()))
    assert read.deviation_minimum <= read.deviation_maximum


def test_the_trace_covers_the_passes_and_not_the_initial_design(toy_loop):
    """A row for the design would be a row of blanks: it has no pick and no acquisition value."""
    bo, terms, values = toy_loop
    bo.initialize(x0=terms[:3], y0=[values[term] for term in terms[:3]])
    assert bo.trace == []


def test_the_trace_a_caller_gets_is_not_the_one_the_run_keeps(toy_loop):
    """A caller who edits what they were handed must not edit the run's own record."""
    bo, terms, values = toy_loop
    bo.optimize(lambda term: values[term], budget=2, x0=terms[:3],
                y0=[values[term] for term in terms[:3]])

    handed_out = bo.trace
    handed_out.clear()
    assert len(bo.trace) == 2


def test_reset_clears_what_a_second_run_must_not_inherit(toy_loop):
    bo, terms, values = toy_loop
    bo.optimize(lambda term: values[term], budget=1, x0=terms[:3],
                y0=[values[term] for term in terms[:3]], record_population=True)
    assert bo.trace
    assert bo.last_acquisition_run is not None

    bo.reset()
    assert bo.trace == []
    assert bo.last_acquisition_run is None
    # The model-selection warning belongs to a fit, and a second run fits again.
    assert bo._warned_about_model_selection is False
    assert bo._warned_about_exploitation is False


def test_the_maximisation_is_recorded_only_when_it_is_asked_for(toy_loop):
    """``None`` says it was not asked for, which is not the same as an empty record."""
    bo, terms, values = toy_loop
    bo.initialize(x0=terms[:3], y0=[values[term] for term in terms[:3]])

    suggestion = bo.suggest()
    assert bo.last_acquisition_run is None

    bo.observe(suggestion.candidate, values[suggestion.candidate])
    bo.suggest(record_population=True)
    run = bo.last_acquisition_run
    assert run is not None
    assert len(run.population) == 3


def test_the_recorded_acquisition_does_not_change_after_its_pass(toy_loop):
    """The whole point of recording it, and the defect that made recording it useless.

    The acquisition used to be handed the loop's live set of observed terms, so the next
    ``observe()`` reached into the object that had already been filed away: asked afterwards, it
    answered the known-point floor at the very term it had itself chosen.  Everything the frontier
    read computes from it would then describe a function no maximization ran.
    """
    bo, terms, values = toy_loop
    bo.initialize(x0=terms[:3], y0=[values[term] for term in terms[:3]])
    suggestion = bo.suggest(record_population=True)

    run = bo.last_acquisition_run
    assert run is not None
    assert run.pick == suggestion.candidate
    assert run.acquisition.gp is bo._model
    assert run.acquisition.incumbent == pytest.approx(max(values[t] for t in terms[:3]))
    assert run.acquisition.known_points == set(terms[:3])

    during = run.acquisition(run.pick)
    bo.observe(suggestion.candidate, values[suggestion.candidate])

    assert run.acquisition(run.pick) == pytest.approx(during)
    assert run.acquisition.known_points == set(terms[:3])


@pytest.mark.parametrize(
    ("name", "acquisition", "attribute", "value"),
    [
        ("ExpectedImprovement", ExpectedImprovement, "incumbent", 0.2),
        ("ProbabilityOfImprovement", ProbabilityOfImprovement, "margin", 0.75),
        ("UpperConfidenceBound", UpperConfidenceBound, "beta", 3.5),
    ],
)
def test_a_pass_builds_the_acquisition_its_name_asks_for(
    toy_loop, name, acquisition, attribute, value
):
    """Each of the three names builds its own class, and that class carries the parameter it reads.

    Only the parameter of the acquisition a run builds is that run's parameter, so the exploration
    parameter reaches an upper confidence bound and the margin a probability of improvement.  A
    name wired to the wrong class, or a parameter left at its default on the way in, would score a
    run differently rather than fail it, and the recorded acquisition is where that is readable.
    """
    bo, terms, values = toy_loop
    bo.acquisition_function = name
    bo.pi_margin = value if attribute == "margin" else 0.0
    bo.ucb_beta = value if attribute == "beta" else 2.0
    bo.initialize(x0=terms[:3], y0=[values[term] for term in terms[:3]])

    bo.suggest(record_population=True)

    run = bo.last_acquisition_run
    assert run is not None
    assert type(run.acquisition) is acquisition
    assert getattr(run.acquisition, attribute) == pytest.approx(value)


def test_the_recorded_maximisation_reads_as_a_frontier(toy_loop):
    """The fourth check on a run rather than on a fixture, which is what recording it is for."""
    bo, terms, values = toy_loop
    bo.optimize(lambda term: values[term], budget=2, x0=terms[:3],
                y0=[values[term] for term in terms[:3]], record_population=True)

    run = bo.last_acquisition_run
    assert run is not None
    read = read_frontier(run)
    assert read.size == len(run.population)
    assert read.on_frontier
    assert read.higher_scored_members == 0
    assert read.fallback_used == run.fallback_used


def test_the_loop_warns_when_the_acquisition_sits_at_its_floor(toy_loop, caplog):
    """The wiring of the overconfidence chain, not only the function that words it.

    ``warn_if_exploitation_stalls`` has its own tests, but that ``suggest()`` calls it has none,
    and the two are the sort of pair that drifts apart silently.  A threshold nothing can beat
    puts probability of improvement at zero everywhere, which is the condition.
    """
    bo, terms, values = toy_loop
    bo.acquisition_function = "ProbabilityOfImprovement"
    bo.pi_margin = 100.0
    bo.initialize(x0=terms[:3], y0=[values[term] for term in terms[:3]])

    with caplog.at_level(logging.WARNING, logger="bayesian_optimization.bo"):
        suggestion = bo.suggest()

    assert suggestion.acquisition_value == pytest.approx(0.0)
    assert "its smallest possible value" in caplog.text
    assert "read_calibration" in caplog.text
    assert bo._warned_about_exploitation


def test_the_kernel_a_pass_fitted_is_readable_off_the_surrogate():
    """The kernel a pass fitted is public, and the one the constructor got keeps its values.

    ``suggest()`` fits a Gaussian process and maximizes its acquisition against that model, so
    ``kernel_`` of the model carries the hyperparameters the choice was made under.  The property
    ``surrogate`` is the way to that model while the run is still going, and ``finalize()``
    reports the same object at the end under ``gp_model``.  This loop sets ``kernel_optimizer``,
    so the fit moves ``length_scale`` off the value the kernel was constructed with, and a fit
    that wrote through to ``bo.kernel`` would move it there too.  Each pass replaces the model
    the property reports, and ``reset()`` drops it.
    """
    terms = [Tree(f"t{index}", ()) for index in range(6)]
    values = {term: 0.1 * index for index, term in enumerate(terms)}
    bo = BayesianOptimization(
        None,
        optimizer=_ChainOptimizer(terms),
        kernel=_IndexKernel(terms),
        kernel_optimizer="fmin_l_bfgs_b",
        gp_normalize_y=False,
        seed=0,
    )
    constructed = bo.kernel.theta.copy()
    bo.initialize(x0=terms[:3], y0=[values[term] for term in terms[:3]])
    assert bo.surrogate is None, "no pass has fitted a model yet"

    first = bo.suggest()
    fitted = bo.surrogate
    assert fitted is not None
    assert not np.array_equal(fitted.kernel_.theta, constructed), "the fit moved length_scale"
    assert np.array_equal(bo.kernel.theta, constructed), "bo.kernel keeps its constructed values"
    bo.observe(first.candidate, values[first.candidate])

    second = bo.suggest()
    assert bo.surrogate is not fitted, "each pass reports its own model, not the first one"
    bo.observe(second.candidate, values[second.candidate])

    assert bo.finalize()["gp_model"] is bo.surrogate, "the two reads of the fitted model disagree"

    bo.reset()
    assert bo.surrogate is None, "a reset takes the last fitted model with it"


def test_the_surrogate_over_the_dataset_has_seen_the_final_evaluation(toy_loop):
    """The gap ``finalize()`` documents, closed: its model is one observation short of the run."""
    bo, terms, values = toy_loop
    result = bo.optimize(lambda term: values[term], budget=2, x0=terms[:3],
                         y0=[values[term] for term in terms[:3]])

    assert len(result["x"]) == 5
    assert len(result["gp_model"].X_train_) == 4
    assert len(bo.surrogate_over_dataset().X_train_) == 5

    read = read_calibration(bo.surrogate_over_dataset())
    assert read.size == 5


def test_the_surrogate_over_the_dataset_conditions_on_the_distinct_pairs(toy_loop):
    """A dataset handed in from outside may repeat a term, and the conditioning may not.

    Two identical rows are linearly dependent, which makes the Gram matrix singular.  That is why
    the loop conditions the Gaussian process on the distinct pairs of its dataset rather than on
    every row.  The loop's own proposals are novel by the rejection path, so only an ``x0`` like
    this one reaches the case.
    """
    bo, terms, values = toy_loop
    bo.initialize(
        x0=[terms[0], terms[1], terms[0]],
        y0=[values[terms[0]], values[terms[1]], values[terms[0]]],
    )
    assert len(bo.surrogate_over_dataset().X_train_) == 2


def test_a_surrogate_over_an_empty_dataset_is_refused():
    bo = BayesianOptimization(None)
    with pytest.raises(RuntimeError, match="dataset is empty"):
        bo.surrogate_over_dataset()


# ---------------------------------------------------------------------------
# The trajectory of the inner run
# ---------------------------------------------------------------------------


def test_the_maximisation_records_one_entry_per_generation(toy_loop):
    """Convergence of an evolutionary run is a claim about its best member over the generations.

    These records are where that claim becomes visible.  The frontier read places the answer
    among the population it came from, which is a statement about the *last* generation: it
    cannot tell a search that climbed from one that never moved.  Without these records a real
    run has no evidence either way, because the loop keeps only the final population of the
    final pass.
    """
    bo, terms, values = toy_loop
    bo.initialize(x0=terms[:3], y0=[values[t] for t in terms[:3]])
    bo.suggest(record_population=True)

    run = bo.last_acquisition_run
    assert run is not None
    assert len(run.generations) == 1, "the stub driver yields exactly one generation"
    record = run.generations[0]
    assert record.generation == 0
    assert record.population_worst <= record.population_mean <= record.population_best
    assert record.best >= record.population_best or record.best == record.population_best
    assert record.distinct_members == 3
    assert record.offspring == 0


def test_a_maximisation_that_was_not_asked_to_record_carries_no_generations(toy_loop):
    """Empty is "not asked", and it has to stay distinguishable from "no generations".

    Every inner run yields at least its zeroth generation, so an empty sequence can only mean the
    recording was off.  That is why the CSV writer refuses an acquisition run whose generations
    are empty rather than writing a file that reads as a search that did nothing.
    """
    bo, terms, values = toy_loop
    bo.initialize(x0=terms[:3], y0=[values[t] for t in terms[:3]])
    bo.suggest(record_population=False)

    assert bo.last_acquisition_run is None


def test_a_generation_record_refuses_a_vector_valued_fitness():
    """The summary is arithmetic, and arithmetic on a Pareto tuple means nothing.

    An acquisition function returns a scalar, so this cannot happen through the loop, but the
    optimizer is a component a caller may drive directly, and averaging ``(0.1, 0.9)`` with
    ``(0.9, 0.1)`` into one number is the kind of answer that reads as a measurement.
    """
    from bayesian_optimization.acquisition_optimizer import _generation_record

    class _VectorState:
        generation = 0
        best = "a"
        best_fitness = (0.1, 0.9)
        last_improvement = 0

    _VectorState.population = ["a"]
    _VectorState.fitness = {"a": (0.1, 0.9)}
    _VectorState.offspring = []

    with pytest.raises(TypeError, match="scalar"):
        _generation_record(_VectorState())
