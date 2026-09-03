from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_solve
from sklearn.gaussian_process import GaussianProcessRegressor


@dataclass(frozen=True)
class CalibrationRead:
    """What the standardized leave-one-out residuals say about the surrogate's uncertainty.

    A surrogate can predict well and still misreport its confidence, and only the standardized
    residuals expose it.  The fit read, which puts predicted against true values, cannot see this
    failure: under an overconfident surrogate the raw residuals stay at most 0.39 while the
    standardized ones reach 9.3.

    Attributes:
        size (int): How many training points the surrogate was conditioned on, one residual each.
        standardized_residuals (tuple[float, ...]): The values ``z_i = (y_i - m_{-i}) / s_{-i}``,
            in the order the surrogate holds its training data.  Calibrated, they scatter with
            both signs on the scale of plus and minus two.
        mean_absolute (float): Their mean absolute value, 1.35 for the calibrated reference
            surrogate.
        maximum_absolute (float): The largest of them: 2.29 calibrated, 9.30 overconfident.
        root_mean_square (float): Their spread **about zero**, which is the quantity the
            calibration rule is about: calibrated residuals are standard normal, and a standard
            normal is centered at zero rather than at whatever these happen to average.  Far above
            one means overconfident, and its symptom is an expected improvement near zero
            everywhere, so pure exploitation.  Far below one means underconfident, so pure
            exploration.
        standard_deviation (float): Their spread about their own mean.  Read it *beside* the one
            above, not instead of it: the difference between the two is bias, and residuals that
            all carry a single sign are the underconfident signature.  A surrogate whose residuals
            are all near +50 has a root mean square of 50 and a standard deviation of 0.34, and
            only the first of those numbers says what is wrong.
        outside_two (int): How many fall outside the band between minus two and plus two.
        positive_fraction (float): The share with a positive sign.  Zero or one is the
            underconfident signature: the residuals carry one sign because an untrained mean
            underestimates the whole objective.
        log_marginal_likelihood (float): The surrogate's log marginal likelihood.  It belongs to
            this read because it is the repair: on the reference data the marginal likelihood
            separates the calibrated surrogate from the overconfident and the underconfident one,
            and prefers the calibrated one.  Comparable only between surrogates over the same
            data.
        residuals (tuple[float, ...]): The raw residuals ``y_i - m_{-i}``, read beside the
            standardized ones to make the point that they stay moderate.
        predicted_deviations (tuple[float, ...]): The leave-one-out standard deviations they were
            divided by.
        targets_normalized (bool): Whether the regressor centered and scaled its targets
            (``normalize_y``).  Read it before anything else here, because it decides what these
            numbers are a statement about.

            The leave-one-out deviation ``s_{-i}`` is a function of the kernel alone, since a
            Gaussian process's posterior variance does not depend on the values it observed, while
            the residual is in the units of the objective.  **So a calibration is a statement
            about the objective's scale against the kernel's amplitude, and multiplying an
            objective by ten multiplies every standardized residual by ten.**  Measured on seven
            points: ``max |z|`` goes 1.64, 16.40, 164.0 as the objective is scaled by one, ten and
            a hundred under a kernel with a fixed amplitude.

            Two things fix the scale, and one of them has to be in place before a spread far above
            one means overconfidence rather than large numbers:

            * an amplitude the marginal likelihood can fit.  Model selection then absorbs the
              scale, and the same three objectives give ``max |z| = 2.11`` throughout.
            * ``normalize_y``, which sets the target variance to one and so meets a fixed
              amplitude halfway.  The same three give 5.56 throughout.

            Neither is free.  The calibrated and overconfident figures quoted above come from an
            uncentered fit whose objective already lives on the unit scale, so a run compared
            against them reads an uncentered surrogate.  And ``BayesianOptimization`` normalizes
            by default while its **default kernel declares no hyperparameters at all**, so on that
            configuration ``normalize_y`` is the only thing standing between the residuals and the
            units of the objective.  Model selection is off there for the same reason: a kernel
            that declares no hyperparameter offers an optimizer no amplitude to absorb the scale
            with.

            The raw fields carry the same units caveat: where the targets were normalized,
            :attr:`residuals`, :attr:`predicted_deviations` and :attr:`log_marginal_likelihood`
            are in the regressor's internal units rather than in those of the objective.
    """

    size: int
    standardized_residuals: tuple[float, ...]
    mean_absolute: float
    maximum_absolute: float
    root_mean_square: float
    standard_deviation: float
    outside_two: int
    positive_fraction: float
    log_marginal_likelihood: float
    residuals: tuple[float, ...]
    predicted_deviations: tuple[float, ...]
    targets_normalized: bool


def read_calibration(surrogate: GaussianProcessRegressor) -> CalibrationRead:
    """Measure how well a fitted surrogate's predicted uncertainty matches its own errors.

    The third acceptance check.  For each term, the surrogate is conditioned on the other
    evaluations and predicts the left-out one, a leave-one-out residual with a closed form for
    Gaussian processes (Rasmussen and Williams, Section 5.4.2).  With ``K`` the matrix the
    regressor was fitted with, including its diagonal term,

        ``m_{-i} = y_i - [K^-1 y]_i / [K^-1]_ii``    and    ``s_{-i}^2 = 1 / [K^-1]_ii``,

    so the standardized residual is ``[K^-1 y]_i / sqrt([K^-1]_ii)``.  The closed form is not an
    optimization: refitting term by term would refit the hyperparameters term by term too, and
    then the residuals would describe as many surrogates as there are terms rather than the one
    the loop uses.  It reads the fitted decomposition instead, because the regressor already holds
    the Cholesky factor and ``K^-1 y``.

    **This read conditions nothing itself, so what it reads is whatever the surrogate was fitted
    on.**  The surrogate a run returns is the one of its last ``suggest()``, which never saw the
    final evaluation.  :meth:`BayesianOptimization.surrogate_over_dataset` is the one over the
    whole dataset.

    Args:
        surrogate (GaussianProcessRegressor): A **fitted** regressor.

    Returns:
        CalibrationRead: The measurements.

    Raises:
        AttributeError: If the regressor has not been fitted.
        ValueError: If it was fitted on fewer than two points, where leaving one of one out
            leaves nothing to condition on, if it carries several targets, or if the inverse of
            its kernel matrix has a non-positive diagonal entry, which is a leave-one-out variance
            that is not a variance and means the matrix is numerically indefinite.
    """
    for attribute in ("L_", "alpha_", "y_train_"):
        if not hasattr(surrogate, attribute):
            msg = (
                f"the calibration read needs a fitted surrogate, and this one has no "
                f"{attribute!r}: call fit() before reading it"
            )
            raise AttributeError(msg)

    targets = np.asarray(surrogate.y_train_, dtype=float)
    if targets.ndim > 1 and targets.shape[1] != 1:
        msg = (
            f"the calibration read is written for one objective, and this surrogate carries "
            f"{targets.shape[1]}"
        )
        raise ValueError(msg)
    targets = targets.reshape(-1)

    size = int(targets.size)
    if size < 2:
        msg = (
            f"a leave-one-out residual conditions on the *other* observations, and there is no "
            f"other one among {size}"
        )
        raise ValueError(msg)

    # K^-1 y, which the regressor computed when it fitted, and the diagonal of K^-1, which it did
    # not.  Both from the same Cholesky factor, so both describe the same matrix: the kernel
    # with its fitted hyperparameters and the diagonal term the caller chose.
    weights = np.asarray(surrogate.alpha_, dtype=float).reshape(-1)
    inverse_diagonal = np.diag(cho_solve((surrogate.L_, True), np.eye(size)))
    # Finite *and* positive.  Positivity alone cannot fail here, because the inverse of a
    # Cholesky product is positive definite, so its diagonal is positive whenever it is a number
    # at all.  Testing only that left the case this actually breaks in unguarded: a factor with a
    # zero or near-zero pivot, which near-duplicate rows under too small a diagonal term produce,
    # comes back as ``inf`` or ``nan``.  Those pass ``> 0`` and turn into a deviation of zero and
    # a residual of ``nan``, which is a measurement this package does not hand on.
    if not np.all(np.isfinite(inverse_diagonal)) or np.any(inverse_diagonal <= 0.0):
        smallest = float(np.min(inverse_diagonal))
        msg = (
            f"the leave-one-out variance 1 / [K^-1]_ii is not a variance at every training point: "
            f"the diagonal of the inverse runs to {smallest} at its smallest and holds "
            f"{int(np.count_nonzero(~np.isfinite(inverse_diagonal)))} entries that are not finite."
            f"  The kernel matrix is numerically singular, which near-duplicate rows and a "
            f"diagonal term too small to separate them produce together."
        )
        raise ValueError(msg)

    deviations = 1.0 / np.sqrt(inverse_diagonal)
    standardized = weights / np.sqrt(inverse_diagonal)
    residuals = weights / inverse_diagonal

    return CalibrationRead(
        size=size,
        standardized_residuals=tuple(float(value) for value in standardized),
        mean_absolute=float(np.mean(np.abs(standardized))),
        maximum_absolute=float(np.max(np.abs(standardized))),
        root_mean_square=float(np.sqrt(np.mean(standardized**2))),
        standard_deviation=float(np.std(standardized)),
        outside_two=int(np.count_nonzero(np.abs(standardized) > 2.0)),
        positive_fraction=float(np.count_nonzero(standardized > 0.0) / size),
        log_marginal_likelihood=float(surrogate.log_marginal_likelihood_value_),
        residuals=tuple(float(value) for value in residuals),
        predicted_deviations=tuple(float(value) for value in deviations),
        targets_normalized=bool(surrogate.normalize_y),
    )
