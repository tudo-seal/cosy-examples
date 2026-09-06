"""The four acceptance checks, on the figures they are the checks of.

The thesis draws its diagnostic figures on an idealized setting: the chain language, the lists of
lengths zero to ten, and a squared-exponential covariance on the term size.  It states numbers for
them in its text and beside each figure, and those numbers come from a figure script written for
the thesis without looking at this implementation.  The reads of
:mod:`bayesian_optimization.diagnostics` therefore have an oracle that no part of this repository
produced, and this module holds them to it.

Two settings, and both are needed.  The idealized one pins the *formulas* against the thesis: the
leave-one-out closed form, the fit scatter, the four kernel-matrix readings, the mean-deviation
plane.  The chain **space**, cosy's terms and this repository's subtree kernel, pins the reads
against the objects the framework actually carries, where a term is a term rather than its length.

What is deliberately absent: a threshold.  The thesis names failure modes and does not quantify
them, so each test here compares a healthy configuration against a broken one and asks the read to
separate them, which is the claim the acceptance checks make.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from cosy.search import generator_query
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from bayesian_optimization import (
    AcquisitionRun,
    ExpectedImprovement,
    OrderedRootedSubtreeKernel,
    ProbabilityOfImprovement,
    kernel_objective_alignment,
    read_calibration,
    read_fit,
    read_frontier,
    read_gram,
)
from tests.spaces import CHAIN, chain_space, chain_terms

# The idealized setting of the figures: the lengths, the objective and the lengthscales, as the
# thesis fixes them beside each figure.  ``JITTER`` is the figure script's value, while the thesis
# quotes its log marginal likelihoods at 1e-6 instead.  Both are stated where they are used.
_LENGTHS = list(range(11))
_IDEAL, _CONSTANT, _DIAGONAL, _OVERCONFIDENT = 1.8, 50.0, 0.05, 2.05
_JITTER = 1e-8


def _objective(length: float) -> float:
    """The idealized objective of the figures, ``exp(-(l-6)^2/8) + 0.06 sin(2.2 l)``."""
    return math.exp(-((length - 6) ** 2) / 8.0) + 0.06 * math.sin(2.2 * length)


def _surrogate(
    lengths: list[int], lengthscale: float, *, jitter: float = _JITTER
) -> GaussianProcessRegressor:
    """Condition the idealized surrogate on the given lengths.

    Model selection is off: the figures fix the lengthscale, and three of the four are the
    degenerate ones that a marginal-likelihood fit exists to reject.
    """
    x = np.asarray(lengths, dtype=float).reshape(-1, 1)
    y = np.asarray([_objective(length) for length in lengths], dtype=float)
    return GaussianProcessRegressor(
        kernel=RBF(lengthscale), alpha=jitter, optimizer=None, normalize_y=False
    ).fit(x, y)


def _squared_exponential(lengthscale: float) -> np.ndarray:
    """The kernel matrix of the figures' covariance over all eleven lengths."""
    grid = np.asarray(_LENGTHS, dtype=float)
    matrix: np.ndarray = np.exp(-((grid[:, None] - grid[None, :]) ** 2) / (2.0 * lengthscale**2))
    return matrix


# ---------------------------------------------------------------------------
# The calibration figure and its reading: the leave-one-out closed form
# ---------------------------------------------------------------------------

# The standardized residuals the thesis prints, one panel per lengthscale.
_CHAPTER_RESIDUALS = {
    _IDEAL: [0.303, -0.747, 1.274, -1.784, 2.158, -2.293, 2.170, -1.790, 1.288, -0.760, 0.315],
    _OVERCONFIDENT: [
        4.460, -5.872, 7.196, -8.299, 9.038, -9.303, 9.057, -8.325, 7.228, -5.908, 4.495,
    ],
    _DIAGONAL: [0.011, 0.092, 0.078, 0.343, 0.642, 0.822, 1.036, 0.901, 0.550, 0.373, 0.135],
}


@pytest.mark.parametrize("lengthscale", list(_CHAPTER_RESIDUALS))
def test_the_closed_form_reproduces_the_calibration_figure(lengthscale):
    """The closed form reproduces every standardized residual the thesis prints, to its digits.

    The thesis computes them by conditioning on the other ten terms eleven times over, while this
    read takes them off one Cholesky factor (Rasmussen and Williams, Section 5.4.2).  Agreeing on
    all thirty-three values is what says the closed form is the same quantity.

    A relative tolerance, and the loosest of the module: the overconfident panel agrees to four
    significant digits rather than to the printed three, and the reason is that panel's own
    subject.  Its kernel matrix has a smallest eigenvalue of 4.8e-6, so the diagonal term of 1e-8
    moves it by two parts in a thousand, and the two computations place that term differently.
    The closed form adds it at every point including the left-out one, which makes the leave-one-
    out variance the predictive variance of the *observation*.  Refitting adds it only at the ten
    conditioned points, leaving the variance of the latent value.  Where the matrix is not on the
    edge of invertibility the difference is invisible: the calibrated and the underconfident
    panels match to the last printed digit, and
    :func:`test_the_closed_form_agrees_with_refitting_without_a_diagonal_term` pins the identity
    itself.
    """
    read = read_calibration(_surrogate(_LENGTHS, lengthscale))
    assert read.standardized_residuals == pytest.approx(
        _CHAPTER_RESIDUALS[lengthscale], rel=1e-3, abs=5e-4
    )


def test_the_closed_form_agrees_with_refitting_without_a_diagonal_term():
    """The closed form *is* leave-one-out, to machine precision.

    Without a diagonal term the two computations describe the same distribution, and then nothing
    but rounding separates them, on the most awkward lengthscale of the thesis, which is where a
    formula that was merely close would show it.
    """
    surrogate = _surrogate(_LENGTHS, _OVERCONFIDENT, jitter=0.0)
    closed = read_calibration(surrogate).standardized_residuals

    refitted = []
    for left_out in _LENGTHS:
        rest = [length for length in _LENGTHS if length != left_out]
        mean, deviation = _surrogate(rest, _OVERCONFIDENT, jitter=0.0).predict(
            [[float(left_out)]], return_std=True
        )
        refitted.append((_objective(left_out) - mean[0]) / deviation[0])

    assert closed == pytest.approx(refitted, abs=1e-8)


def test_the_calibration_read_separates_the_three_surrogates():
    """The residuals expose a surrogate that predicts well and misreports its confidence.

    That is the reading the thesis attaches to the three panels, and the read carries it as summary
    statistics.  Calibrated: "they scatter with both signs on the scale of the rules, and none
    falls far outside", mean 1.35 and maximum 2.29.  Overconfident: "the residuals leave the rules
    by factors", maximum 9.30, while "the raw residuals of that surrogate stay moderate", at most
    0.39, which is why the fit scatter alone would understate the failure.  Underconfident: "they
    stay small against the rules, and they carry a single sign".
    """
    calibrated = read_calibration(_surrogate(_LENGTHS, _IDEAL))
    overconfident = read_calibration(_surrogate(_LENGTHS, _OVERCONFIDENT))
    underconfident = read_calibration(_surrogate(_LENGTHS, _DIAGONAL))

    assert calibrated.mean_absolute == pytest.approx(1.35, abs=0.01)
    assert calibrated.maximum_absolute == pytest.approx(2.29, abs=0.01)
    assert calibrated.outside_two == 3
    assert 0.0 < calibrated.positive_fraction < 1.0

    assert overconfident.maximum_absolute == pytest.approx(9.30, abs=0.01)
    assert max(abs(value) for value in overconfident.residuals) == pytest.approx(
        0.39, abs=0.01
    )
    # "By factors": measured, the spread grows by 4.8, on residuals that stay a quarter of the
    # size.  The calibrated spread is itself 1.53 rather than 1, because eleven points of a smooth
    # objective are not a sample from the prior, and the thesis asks for "on the scale of the
    # rules", not for a fitted variance.
    assert overconfident.standard_deviation > 4.0 * calibrated.standard_deviation
    assert max(abs(value) for value in overconfident.residuals) < max(
        abs(value) for value in underconfident.residuals
    )

    assert underconfident.maximum_absolute < 1.1
    assert underconfident.positive_fraction == 1.0
    assert underconfident.standard_deviation < calibrated.standard_deviation


def test_the_marginal_likelihood_prefers_the_calibrated_surrogate():
    """Model selection is the repair: the marginal likelihood separates the three surrogates.

    It prefers the calibrated one, which is the claim the thesis makes for it.  The numbers stated
    there, -0.66, -11.89 and -36.92, are quoted at a diagonal term of 1e-6 rather than at the 1e-8
    the residual panels use, as the configuration beside the calibration figure says.
    """
    likelihoods = {
        lengthscale: read_calibration(
            _surrogate(_LENGTHS, lengthscale, jitter=1e-6)
        ).log_marginal_likelihood
        for lengthscale in (_IDEAL, _DIAGONAL, _OVERCONFIDENT)
    }
    assert likelihoods[_IDEAL] == pytest.approx(-0.66, abs=0.01)
    assert likelihoods[_DIAGONAL] == pytest.approx(-11.89, abs=0.01)
    assert likelihoods[_OVERCONFIDENT] == pytest.approx(-36.92, abs=0.01)
    assert likelihoods[_IDEAL] > likelihoods[_DIAGONAL] > likelihoods[_OVERCONFIDENT]


# ---------------------------------------------------------------------------
# The held-out scatter of predicted against true values, and its reading
# ---------------------------------------------------------------------------

# The (true, predicted) pairs the thesis plots, conditioned on the even lengths and predicted at
# the odd ones.
_CHAPTER_SCATTER = {
    _IDEAL: [
        (0.0924, 0.0016), (0.3433, 0.3029), (0.8225, 0.9453), (0.9007, 0.8595), (0.3735, 0.2849),
    ],
    _DIAGONAL: [
        (0.0924, 0.0), (0.3433, 0.0), (0.8225, 0.0), (0.9007, 0.0), (0.3735, 0.0),
    ],
}


@pytest.mark.parametrize("lengthscale", list(_CHAPTER_SCATTER))
def test_the_fit_read_reproduces_the_scatter_figure(lengthscale):
    """Both panels of the predicted-against-true scatter of the thesis, point by point."""
    held_out = [1, 3, 5, 7, 9]
    read = read_fit(
        _surrogate([0, 2, 4, 6, 8, 10], lengthscale),
        [[float(length)] for length in held_out],
        [_objective(length) for length in held_out],
    )
    expected = _CHAPTER_SCATTER[lengthscale]
    assert read.true_values == pytest.approx([true for true, _ in expected], abs=5e-5)
    assert read.predicted_values == pytest.approx(
        [predicted for _, predicted in expected], abs=5e-5
    )


def test_the_fit_read_separates_the_diagonal_from_the_horizontal_cloud():
    """"Points near the diagonal ... a horizontal cloud means that the kernel transfers nothing."

    And the finding that came out of measuring it: **the rank correlation, which the thesis names
    as the robust summary of this picture, is the same in both panels.**  The degenerate surrogate
    does not answer one value.  It answers values of the order of 1e-87, whose *order* still tracks
    the objective, and a rank statistic cannot tell a spread of 0.9 from a spread of 2e-87.  Both
    panels come out at 0.8.

    So the two panels are separated by the spread of the predictions against the spread of the
    objective, and the read reports both.  The reasoning of the thesis is not wrong, since the
    acquisition does consume comparisons, but a surrogate whose comparisons live below the
    resolution of every downstream computation makes no usable ones, and the rank correlation
    cannot see that.
    """
    held_out = [1, 3, 5, 7, 9]
    values = [_objective(length) for length in held_out]
    terms = [[float(length)] for length in held_out]

    healthy = read_fit(_surrogate([0, 2, 4, 6, 8, 10], _IDEAL), terms, values)
    broken = read_fit(_surrogate([0, 2, 4, 6, 8, 10], _DIAGONAL), terms, values)

    # Not one: the fitting panel of the thesis swaps one pair of neighboring ranks, so its rank
    # correlation over the five held-out terms is exactly 0.8.  "Positive and stable" is the
    # claim, and a test that demanded a perfect ranking would be testing a stronger one than the
    # figure makes.
    assert healthy.rank_correlation == pytest.approx(0.8)
    assert broken.rank_correlation == pytest.approx(healthy.rank_correlation)

    assert healthy.regression_slope == pytest.approx(1.0, abs=0.2)
    assert healthy.residual_root_mean_square < 0.1
    assert healthy.prediction_spread > 0.5 * healthy.objective_spread

    assert broken.prediction_spread < 1e-80
    assert broken.objective_spread > 0.8
    assert broken.residual_root_mean_square > 0.5


# ---------------------------------------------------------------------------
# The four kernel-matrix panels and their reading
# ---------------------------------------------------------------------------


def test_the_gram_read_separates_the_four_panels_of_the_figure():
    """"A nearly constant matrix ... discriminates nothing ... a nearly diagonal matrix is the
    other extreme ... a healthy matrix shows structure between the extremes."

    One statistic cannot separate the three: the near-constant and the near-diagonal matrix both
    have almost no spread off the diagonal, and they differ in where that flat value sits.  So the
    read reports the level and the spread, and the two together place all three panels.
    """
    healthy = read_gram(_squared_exponential(_IDEAL))
    constant = read_gram(_squared_exponential(_CONSTANT))
    diagonal = read_gram(_squared_exponential(_DIAGONAL))

    assert constant.off_diagonal_mean > 0.98
    assert constant.off_diagonal_spread < 0.03

    assert diagonal.off_diagonal_mean < 1e-6
    assert diagonal.off_diagonal_spread < 1e-6

    assert 0.1 < healthy.off_diagonal_mean < 0.9
    assert healthy.off_diagonal_spread > 0.5

    # Near-duplicate rows leave a kernel matrix ill-conditioned, and the constant matrix here is
    # eleven copies of one row.
    assert constant.condition_number > 1e6 * healthy.condition_number

    for read in (healthy, constant, diagonal):
        assert read.symmetry_error < 1e-12
        assert read.minimum_eigenvalue > -1e-8
        assert read.diagonal_spread == pytest.approx(1.0)


def _neighbour_similarity(matrix: np.ndarray, order: tuple[int, ...]) -> float:
    """The mean similarity between terms an ordering puts next to each other."""
    return float(np.mean([matrix[a][b] for a, b in itertools.pairwise(order)]))


def test_the_subtree_kernel_seriates_the_chain_into_its_size_order():
    """"A seriation of their kernel matrix ... places similar terms side by side."

    On this repository's own kernel it does exactly that: the first kernel principal component of
    the normalized subtree kernel over the chain recovers the length order outright, and the terms
    it puts next to each other are more similar than two terms drawn at random.  The direction is
    the reversed one here, which carries nothing, because the sign of an eigenvector is not
    determined and the read only fixes it so that repeated runs agree.
    """
    matrix = OrderedRootedSubtreeKernel(normalize=True)(chain_terms(21))
    read = read_gram(matrix)

    assert read.seriation in (tuple(_LENGTHS), tuple(reversed(_LENGTHS)))
    assert read.seriation_neighbour_similarity == pytest.approx(
        _neighbour_similarity(matrix, tuple(_LENGTHS))
    )
    assert read.seriation_neighbour_similarity > read.off_diagonal_mean


def test_the_squared_exponential_seriation_folds_the_ends_of_the_chain():
    """The recommended term axis folds the ends of the chain, on the covariance of the thesis.

    A landscape picture needs an axis, and the thesis offers the first kernel principal component
    of the kernel matrix as one, on the ground that such a seriation places similar terms side by
    side.  It calls the chain the case where the kernel geometry and the term size agree.  They do
    not agree at the ends: under the squared exponential of the figures the component runs
    monotonically through the middle and turns over the outermost three positions at each end, so
    the seriation opens ``2, 1, 3, 0, 4`` and seats the lists of lengths zero and four together,
    whose similarity is 0.085 against the 0.857 of true neighbors.

    The mechanism is structural, since the leading eigenvector of a centered stationary kernel on
    a grid is its lowest oscillating mode and that mode turns at the boundary, but **whether it
    bites depends on the lengthscale**, and an earlier version of this test claimed otherwise.
    Measured over the eleven lists, the fold is there up to about ``l = 3.65`` and gone above it,
    where the covariance is flat enough across the whole grid that the leading mode is monotone.
    The boundary is asserted from both sides, so a change to the component that moved it would
    show.

    The direction is not asserted anywhere here.  A seriation is determined up to reversal, and on
    a landscape symmetric about its middle nothing in the kernel prefers an end, so the read fixes
    the direction only so that repeated reads agree.  See
    :func:`test_the_seriation_does_not_turn_on_a_rounding_difference`.

    The kernels this framework actually uses are counting kernels and not stationary, which is why
    the test above passes: the finding is about the recommendation, not about the reads.
    """
    folded = read_gram(_squared_exponential(_IDEAL)).seriation
    expected = (2, 1, 3, 0, 4, 5, 6, 10, 7, 9, 8)
    assert folded in (expected, tuple(reversed(expected)))

    for lengthscale in (0.6, _IDEAL, 3.0, 3.6):
        matrix = _squared_exponential(lengthscale)
        read = read_gram(matrix)

        assert read.seriation not in (tuple(_LENGTHS), tuple(reversed(_LENGTHS)))
        # The axis still carries something, in that neighbors beat random pairs, and it is not
        # the order the terms have.  Both halves matter: the first is why the picture looks
        # plausible, the second is why it misleads.
        assert read.seriation_neighbour_similarity > read.off_diagonal_mean
        assert read.seriation_neighbour_similarity < _neighbour_similarity(
            matrix, tuple(_LENGTHS)
        )

    # Above the boundary the recommendation is simply right, and the read says so with the
    # neighbor similarity of the true order.
    for lengthscale in (3.7, 5.0, 20.0):
        read = read_gram(_squared_exponential(lengthscale))
        assert read.seriation in (tuple(_LENGTHS), tuple(reversed(_LENGTHS)))


def test_the_seriation_does_not_turn_on_a_rounding_difference():
    """The seriation holds its direction on the matrix that breaks the obvious sign convention.

    On the chain the leading component is exactly antisymmetric, so its two largest magnitudes are
    equal, and measured they differ by 2.2e-16.  A convention that orients by the largest magnitude
    is then decided by rounding, and three spellings of the same squared exponential that agree to
    3.3e-16 come out in two different directions.  They must not.
    """
    grid = np.asarray(_LENGTHS, dtype=float)
    difference = grid[:, None] - grid[None, :]
    spellings = [
        np.exp(-(difference**2) / (2.0 * _IDEAL**2)),
        np.exp(-0.5 * (difference / _IDEAL) ** 2),
        RBF(_IDEAL)(grid.reshape(-1, 1)),
    ]
    assert max(np.max(np.abs(spellings[0] - other)) for other in spellings[1:]) < 1e-15

    seriations = {read_gram(matrix).seriation for matrix in spellings}
    assert len(seriations) == 1, "three spellings of one kernel, three readings"

    # And it survives a perturbation far larger than the tie it used to turn on.
    nudged = spellings[0].copy()
    nudged[0, 3] -= 1e-14
    nudged[3, 0] -= 1e-14
    assert read_gram(nudged).seriation == next(iter(seriations))

    # The convention itself, stated: the first coordinate clearly away from zero is positive.
    # Agreement above says the three matrices are read the same way, not *which* way, and LAPACK
    # is deterministic enough to give the same answer to all three unaided.  Without this line the
    # convention could be deleted and nothing here would notice: measured, the unoriented
    # component starts at -0.513.
    for matrix in [*spellings, _squared_exponential(0.6), _squared_exponential(5.0)]:
        coordinate = read_gram(matrix).coordinate
        largest = max(abs(value) for value in coordinate)
        first = next(
            value for value in coordinate if abs(value) > 1e-8 * largest
        )
        assert first > 0.0


def test_a_kernel_without_geometry_seriates_into_nothing():
    """The negative control for the seriation: the near-diagonal matrix.

    Every term resembles only itself, so there is no ordering under which neighbors are similar,
    and the axis the read produces is an artifact of a degenerate eigendecomposition.  The neighbor
    similarity says so, and the objective alignment does not, which is why it is not asked: with
    the eigenvalues of the centered matrix all equal, which eigenvector comes back first is a
    property of the LAPACK build rather than of the kernel.
    """
    read = read_gram(
        _squared_exponential(_DIAGONAL),
        objective=[_objective(length) for length in _LENGTHS],
    )
    assert read.seriation_neighbour_similarity < 1e-6
    assert read.off_diagonal_mean < 1e-6


def test_the_seriation_of_a_healthy_matrix_tracks_the_objective():
    """The blocks of a healthy matrix track the objective, read as an axis-objective alignment.

    The subtree kernel's axis is the length, and the idealized objective is smooth in the length,
    so the two correlate.  The negative control is an objective the kernel cannot see: values
    assigned by parity put the extremes of the objective at neighboring lengths.
    """
    matrix = OrderedRootedSubtreeKernel(normalize=True)(chain_terms(21))

    smooth = read_gram(matrix, objective=[_objective(length) for length in _LENGTHS])
    parity = read_gram(matrix, objective=[float(length % 2) for length in _LENGTHS])

    assert smooth.objective_alignment is not None
    assert parity.objective_alignment is not None
    assert smooth.objective_alignment > parity.objective_alignment


def test_the_counting_kernel_shows_the_scale_spread_the_chapter_names():
    """The subtree kernel's values on the chain grow from one to one hundred eleven.

    Measured on this repository's subtree kernel over cosy's chain terms, not on a formula: the
    closed form ``k(a,b) = a*b + min(a,b) + 1`` is what ``chain_space`` documents, and the
    diagonal spread is the hazard the thesis attaches to counting kernels, where the
    self-similarity grows with the term size and large terms dominate the scale.  Dividing each
    entry by the square roots of the two self-similarities removes it, which is the remedy the
    thesis names and also why the read takes this measurement before normalizing.
    """
    terms = chain_terms(21)
    assert len(terms) == 11

    raw = OrderedRootedSubtreeKernel(normalize=False)(terms)
    assert raw[0][0] == pytest.approx(1.0)
    assert raw[10][10] == pytest.approx(111.0)
    for a in range(11):
        for b in range(11):
            assert raw[a][b] == pytest.approx(a * b + min(a, b) + 1)

    unnormalized = read_gram(raw)
    assert unnormalized.diagonal_spread == pytest.approx(111.0)

    normalized = read_gram(OrderedRootedSubtreeKernel(normalize=True)(terms))
    assert normalized.diagonal_spread == pytest.approx(1.0)
    assert normalized.minimum_eigenvalue > -1e-8
    # The read normalizes whatever it is given, so both calls above see the same matrix, which is
    # why the two seriations agreeing says nothing and is not asserted.  What the pair does say
    # is that the kernel's own ``normalize`` flag and the read's normalization are the same map.
    assert np.allclose(
        unnormalized.normalized_matrix, OrderedRootedSubtreeKernel(normalize=True)(terms)
    )


def test_the_alignment_of_the_subtree_kernel_on_the_chain():
    """The alignment statistic orders three objectives on one kernel the way the kernel sees them.

    Kernel-objective alignment is the Spearman rank correlation between the kernel values of pairs
    of distinct terms and the negated absolute differences of their scalarized objective values, so
    a high value means that the kernel rates pairs as similar where the objective values lie close
    together.  The subtree kernel on the chain compares lengths: the length itself aligns almost
    perfectly, the smooth objective of the figures aligns weakly but positively, since its peak at
    six lets two lengths far apart share a value, and parity, which puts the objective's extremes
    at neighboring lengths, aligns negatively.

    The last one is the negative control: a statistic that reported a high value there would be
    reporting the kernel and not its relation to the objective.  Both kernels of this module are
    put through the same three, since a spread this wide should not depend on which kernel drew
    the axis.
    """
    for matrix in (
        OrderedRootedSubtreeKernel(normalize=True)(chain_terms(21)),
        _squared_exponential(_IDEAL),
    ):
        by_size = kernel_objective_alignment(matrix, [float(length) for length in _LENGTHS])
        smooth = kernel_objective_alignment(
            matrix, [_objective(length) for length in _LENGTHS]
        )
        parity = kernel_objective_alignment(matrix, [float(length % 2) for length in _LENGTHS])

        assert by_size is not None and smooth is not None and parity is not None
        assert by_size > smooth > 0.0 > parity


# ---------------------------------------------------------------------------
# The mean-deviation plane and its reading
# ---------------------------------------------------------------------------


class _LengthPosterior:
    """The idealized surrogate, addressed by chain terms instead of by lengths.

    The frontier read scores terms, and the figure's surrogate lives on the size.  On the chain
    they are the same information, one inhabitant per length, so this maps the one onto the other
    and lets the read run on the objects it will meet in a real run.
    """

    def __init__(self, surrogate: GaussianProcessRegressor, terms: list) -> None:
        self._surrogate = surrogate
        self._length_of = {term: index for index, term in enumerate(terms)}

    def predict(self, X, return_std: bool = False):
        lengths = np.asarray([[float(self._length_of[term])] for term in X])
        return self._surrogate.predict(lengths, return_std=return_std)


def test_the_returned_term_of_the_acquisition_plane_is_its_argmax():
    """With five terms evaluated, the acquisition maximizer is the term of length three.

    The thesis reads its mean-deviation plane at that state: "the maximizer at length three wins
    by uncertainty, although the candidates at lengths four and eight promise higher values".  So
    the pick is *not* the candidate of largest mean, and a read that only looked at the mean would
    call it wrong.  The frontier is the two coordinates together.
    """
    terms = chain_terms(21)
    observed = [1, 5, 9, 7, 6]
    posterior = _LengthPosterior(_surrogate(observed, _IDEAL), terms)
    incumbent = max(_objective(length) for length in observed)
    candidates = [terms[length] for length in _LENGTHS if length not in observed]

    acquisition = ExpectedImprovement(posterior, incumbent=incumbent)
    pick = max(candidates, key=acquisition)
    assert pick == terms[3]

    read = read_frontier(AcquisitionRun(acquisition, pick, tuple(candidates)))
    assert read.on_frontier
    assert read.higher_scored_members == 0
    assert read.mean_at_pick < max(mean for mean, _, _ in read.points)


def test_a_collapsed_population_is_visible_as_a_collapsed_population():
    """A population collapsed onto one mediocre region shows up in the numbers of the read.

    That is the failure the thesis draws beside the healthy panel: "the population has collapsed
    onto one mediocre region, its best member scores far below the attainable level".  The healthy
    population spans the plane, while the collapsed one is a handful of copies of one term.  Both
    hand the read the same pick, so what separates them is the population and nothing else.
    """
    terms = chain_terms(21)
    observed = [1, 5, 9, 7, 6]
    posterior = _LengthPosterior(_surrogate(observed, _IDEAL), terms)
    acquisition = ExpectedImprovement(
        posterior, incumbent=max(_objective(length) for length in observed)
    )
    candidates = [terms[length] for length in _LENGTHS if length not in observed]

    spread_out = read_frontier(AcquisitionRun(acquisition, terms[3], tuple(candidates)))
    collapsed = read_frontier(AcquisitionRun(acquisition, terms[3], (terms[0],) * 6))

    assert spread_out.distinct_members == len(candidates)
    assert collapsed.distinct_members == 1
    assert collapsed.mean_spread == 0.0
    assert collapsed.deviation_spread == 0.0
    assert spread_out.mean_spread > 0.0


@pytest.mark.parametrize(
    "observed", [[1, 5, 9], [1, 5, 9, 7], [1, 5, 9, 7, 6], [2, 3, 4], [1, 5, 9, 7, 6, 3]]
)
def test_both_frontier_readings_agree_on_the_chain(observed):
    """The two readings of the frontier, measured against each other on this space.

    The read keeps them apart because probability of improvement, the posterior mass above a
    threshold, can part them in principle: its score falls in the deviation once the mean exceeds
    the threshold, so a dominated candidate can outscore its dominator.  On the chain, over the
    states of the worked run of the thesis, that never happens for either acquisition, and the
    maximizer is undominated *and* highest-scoring in every one.

    Which is the point of measuring it rather than assuming it.  A space where a construction
    cannot go wrong is not evidence that the construction is safe, so the disagreement is
    exhibited where it can be constructed instead, in the unit tests, and here the two are only
    reported to agree.
    """
    terms = chain_terms(21)
    posterior = _LengthPosterior(_surrogate(observed, _IDEAL), terms)
    incumbent = max(_objective(length) for length in observed)
    candidates = [terms[length] for length in _LENGTHS if length not in observed]

    for acquisition in (
        ExpectedImprovement(posterior, incumbent=incumbent),
        ProbabilityOfImprovement(posterior, incumbent=incumbent),
    ):
        read = read_frontier(
            AcquisitionRun(acquisition, max(candidates, key=acquisition), tuple(candidates))
        )
        assert read.higher_scored_members == 0
        assert read.on_frontier


def test_the_chain_space_carries_exactly_the_terms_the_figures_assume():
    """The bridge between the two settings: one inhabitant per length, up to ten.

    Everything above that runs on ``chain_terms`` assumes it, and it is cosy that has to deliver
    it, so it is asked here rather than taken on trust.
    """
    query = generator_query(chain_space(), CHAIN)
    assert query is not None
    terms = chain_terms(21)
    assert len(terms) == len(set(terms)) == 11
