from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..acquisition_function import MarginalPosterior
from ._statistics import rank_correlation, spread


@dataclass(frozen=True)
class FitRead:
    """What the held-out predictions say about the surrogate's mean.

    Attributes:
        size (int): How many held-out terms were predicted.
        rank_correlation (float | None): Spearman's rank correlation of predicted against true
            values, the robust summary of the scatter, because for the loop the ranking of the
            predictions matters more than their values: acquisition maximization consumes
            comparisons.  ``None`` where one of the two sequences is constant.

            **Measured: it does not on its own separate a healthy surrogate from one whose kernel
            transfers nothing.**  Under a degenerate lengthscale the predictions are not constant,
            they are of the order of 1e-87, and their *ranking* survives intact: the rank
            correlation is 0.8 either way, the same number.  Ranks are invariant under any
            increasing map, and collapsing a spread of 0.9 onto a spread of 2e-87 is one.  So the
            robust summary is robust against the failure it is meant to summarize, and what
            separates the two cases is :attr:`prediction_spread` against
            :attr:`objective_spread`.  Read them together.
        predictions_constant (bool): Whether the surrogate answered one value at every held-out
            term, exactly.  The limiting case of a scatter collapsed to a horizontal cloud.  The
            practical form is a spread that has collapsed without reaching zero, which is why it
            is not the field the read turns on.
        prediction_spread (float): The range of the predictions.  This is the horizontal cloud,
            graded: the kernel transfers nothing, every prediction falls back toward the prior
            mean, and the acquisition then scores by uncertainty alone.
        objective_spread (float): The range of the true values, to compare it against.
        regression_slope (float | None): The slope of the least-squares line of predicted on true.
            One is the diagonal, and a slope far from it is a mis-scaled kernel or objective.
            ``None`` where the true values are constant and no line through them has a slope.

            It is a *tilt*, while a mis-scaled kernel or objective also shows as a systematic
            bend, which this cannot see: any curvature odd about the mean of the data leaves the
            slope at one.  A plot of the scatter is what shows a bend, and this field catches the
            half of that failure that is a change of scale.
        residual_root_mean_square (float): The root mean square of ``predicted - true``, the
            distance from the diagonal in the units of the objective.
        true_values (tuple[float, ...]): The scatter's x coordinates, for plotting.
        predicted_values (tuple[float, ...]): Its y coordinates.
    """

    size: int
    rank_correlation: float | None
    predictions_constant: bool
    prediction_spread: float
    objective_spread: float
    regression_slope: float | None
    residual_root_mean_square: float
    true_values: tuple[float, ...]
    predicted_values: tuple[float, ...]


def read_fit(
    surrogate: MarginalPosterior,
    terms: Sequence[object],
    true_values: Sequence[float],
) -> FitRead:
    """Compare a surrogate's predictions at held-out terms against their true objective values.

    The second acceptance check, and the first diagnostic of a surrogate implementation.  Points
    near the diagonal of the scatter mean that observed values transfer to unevaluated terms, and
    a horizontal cloud means the kernel transfers nothing.

    The terms have to be held out.  Conditioned on, a noise-free Gaussian process reproduces its
    training values exactly, and the scatter would sit on the diagonal whatever the kernel does.
    Which terms those are is the caller's choice and not a detail: conditioning on the terms of
    even length and predicting the odd ones is a split that keeps the training set spread over the
    space.  A split that leaves a whole region unobserved measures extrapolation instead, which is
    a different and harder question.

    Args:
        surrogate (MarginalPosterior): The fitted surrogate, conditioned on data that does **not**
            include ``terms``.
        terms (Sequence[object]): The held-out terms.
        true_values (Sequence[float]): Their objective values, in the same order.

    Returns:
        FitRead: The measurements.

    Raises:
        ValueError: If the two sequences differ in length, hold fewer than two entries, or hold
            a value that is not finite, or if the surrogate answers with the wrong number of
            predictions or with one that is not finite.
    """
    held_out = list(terms)
    observed = np.asarray(true_values, dtype=float).reshape(-1)
    if len(held_out) != observed.size:
        msg = (
            f"the fit read pairs terms with values by position: {len(held_out)} terms and "
            f"{observed.size} values"
        )
        raise ValueError(msg)
    if observed.size < 2:
        msg = f"a fit scatter needs at least two held-out terms, got {observed.size}"
        raise ValueError(msg)
    if not np.all(np.isfinite(observed)):
        msg = "the held-out objective values hold a value that is not finite"
        raise ValueError(msg)

    predicted = np.asarray(surrogate.predict(held_out), dtype=float).reshape(-1)
    if predicted.size != observed.size:
        msg = (
            f"the surrogate answered {predicted.size} predictions for {observed.size} held-out "
            f"terms"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(predicted)):
        msg = "the surrogate predicted a value that is not finite at a held-out term"
        raise ValueError(msg)

    residuals = predicted - observed
    objective_range = spread(observed)

    return FitRead(
        size=int(observed.size),
        rank_correlation=rank_correlation(observed, predicted),
        predictions_constant=bool(np.all(predicted == predicted[0])),
        prediction_spread=spread(predicted),
        objective_spread=objective_range,
        # np.polyfit would answer with a slope and a warning where the design matrix is
        # singular.  The singular case here is a constant x, and a constant x has no slope rather
        # than a badly conditioned one.
        regression_slope=(
            None
            if objective_range == 0.0
            else float(np.polyfit(observed, predicted, deg=1)[0])
        ),
        residual_root_mean_square=float(np.sqrt(np.mean(residuals**2))),
        true_values=tuple(float(value) for value in observed),
        predicted_values=tuple(float(value) for value in predicted),
    )


def kernel_objective_alignment(
    matrix: np.ndarray, values: Sequence[float]
) -> float | None:
    """Return the kernel-objective alignment of a kernel matrix against the objective values.

    The alignment is the Spearman rank correlation between ``k(t, t')`` and
    ``-|sigma(q(t)) - sigma(q(t'))|`` over the pairs of distinct terms.  A high value means that
    the kernel rates pairs as similar where the objective values lie close together, which is the
    property the fit read measures through a fitted surrogate and this one measures without
    fitting anything.

    That makes it the cheap comparison between kernels.  The publication this definition comes
    from selects among translations of a term with exactly this statistic, and it needs the
    objective values but no conditioning, no held-out split and no hyperparameters.

    Args:
        matrix (np.ndarray): The kernel matrix on a set of terms, square.  Whether it is
            normalized changes nothing: dividing every entry by the square roots of the two
            self-similarities is not monotone across pairs, so the two readings genuinely differ,
            and reporting which one was read is the caller's business.
        values (Sequence[float]): The objective value of each term, in the order the matrix covers
            them, already through the scalarization ``sigma`` that the definition is written with.

    Returns:
        float | None: The alignment, or ``None`` where one of the two sequences over the pairs is
            constant and the ranks do not exist.

    Raises:
        ValueError: If the matrix is not square, if it covers fewer than three terms, since two
            terms make one pair and one pair cannot be correlated with another, if the values do
            not match it in length, or if either holds a value that is not finite.
    """
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        msg = f"a kernel matrix is square; this one has shape {array.shape}"
        raise ValueError(msg)
    size = int(array.shape[0])
    if size < 3:
        msg = (
            f"the alignment correlates one pair of terms against another, and {size} terms make "
            f"at most one pair"
        )
        raise ValueError(msg)

    objective = np.asarray(values, dtype=float).reshape(-1)
    if objective.size != size:
        msg = (
            f"the matrix covers {size} terms and the objective {objective.size} values; the "
            f"alignment pairs them by position"
        )
        raise ValueError(msg)
    if not np.all(np.isfinite(array)) or not np.all(np.isfinite(objective)):
        msg = "the alignment needs finite kernel values and finite objective values"
        raise ValueError(msg)

    rows, columns = np.triu_indices(size, k=1)
    similarity = array[rows, columns]
    closeness = -np.abs(objective[rows] - objective[columns])
    return rank_correlation(similarity, closeness)
