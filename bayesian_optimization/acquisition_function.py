from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor

from cosy.core.tree import Tree


class AcquisitionFunction:
    """Base class for acquisition functions."""

    def __init__(
        self,
        gp: GaussianProcessRegressor,
        greater_is_better: bool = False,
        known_points: set[Any] | None = None,
        incumbent: float | None = None,
    ) -> None:
        self.gp = gp
        self.greater_is_better = greater_is_better
        self.known_points = known_points
        self.incumbent = incumbent

    def __call__(self, t: Any) -> float:
        raise NotImplementedError("Subclasses must implement __call__.")

    def evaluate_batch(self, trees: Sequence[Any]) -> dict[Any, float]:
        raise NotImplementedError("Subclasses must implement evaluate_batch.")


class ExpectedImprovement(AcquisitionFunction):
    """Expected Improvement acquisition function.

    Parameters
    ----------
    gp:
        Fitted ``GaussianProcessRegressor``.
    xi:
        Exploration parameter; larger values encourage more exploration.
    greater_is_better:
        Whether the objective is maximised (``True``) or minimised (``False``).
    known_points:
        Set of already-evaluated candidates; they receive EI = 0.
    incumbent:
        Current best value in the *transformed* GP space.
        If ``None``, the best value in ``gp.y_train_`` is used.
    """

    def __init__(
        self,
        gp: GaussianProcessRegressor,
        xi: float = 0.01,
        greater_is_better: bool = False,
        known_points: set[Any] | None = None,
        incumbent: float | None = None,
    ) -> None:
        super().__init__(gp, greater_is_better, known_points, incumbent)
        self.xi = xi

    def _loss_optimum(self) -> float:
        if self.incumbent is not None:
            return float(self.incumbent)
        y = np.asarray(self.gp.y_train_)
        return float(np.max(y) if self.greater_is_better else np.min(y))

    def evaluate_batch(self, trees: Sequence[Any]) -> dict[Tree, float]:
        """Evaluate EI for a batch of candidates.

        Parameters
        ----------
        trees:
            Sequence of candidate objects.  Non-``Tree`` entries and ``None``
            are silently skipped.

        Returns
        -------
        dict mapping each valid ``Tree`` to its EI value (always ≥ 0).
        """
        if not trees:
            return {}

        result: dict[Tree, float] = {}
        candidates: list[Tree] = []

        for item in trees:
            if item is None or not isinstance(item, Tree):
                continue
            if self.known_points is not None and item in self.known_points:
                result[item] = 0.0
                continue
            candidates.append(item)

        if not candidates:
            return result

        mu, sigma = self.gp.predict(candidates, return_std=True)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        sigma = np.asarray(sigma, dtype=float).reshape(-1)

        loss_optimum = self._loss_optimum()
        improve = (
            mu - loss_optimum - self.xi
            if self.greater_is_better
            else loss_optimum - mu - self.xi
        )

        ei = np.zeros_like(improve, dtype=float)
        positive = sigma > 0.0
        if np.any(positive):
            scaled = improve[positive] / sigma[positive]
            ei[positive] = improve[positive] * norm.cdf(scaled) + sigma[positive] * norm.pdf(scaled)

        for tree, value in zip(candidates, ei):
            result[tree] = float(max(value, 0.0))

        return result

    def __call__(self, t: Any) -> float:
        if t is None or not isinstance(t, Tree):
            return 0.0
        return float(self.evaluate_batch([t]).get(t, 0.0))


class UpperConfidenceBound(AcquisitionFunction):
    """Upper Confidence Bound acquisition function.

    Scores are always maximised: for minimisation ``score = -mu + kappa*sigma``
    (lower predicted mean → higher acquisition value).

    Parameters
    ----------
    gp:
        Fitted ``GaussianProcessRegressor``.
    greater_is_better:
        Whether the objective is maximised.
    known_points:
        Known candidates receive UCB = 0.
    incumbent:
        Unused in UCB; kept for API consistency.
    kappa:
        Exploration–exploitation trade-off; higher values favour exploration.
        Passed at construction time so both ``__call__`` and ``evaluate_batch``
        use a consistent value (fixes F14).
    """

    def __init__(
        self,
        gp: GaussianProcessRegressor,
        greater_is_better: bool = False,
        known_points: set[Any] | None = None,
        incumbent: float | None = None,
        kappa: float = 2.0,
    ) -> None:
        super().__init__(gp, greater_is_better, known_points, incumbent)
        self.kappa = kappa

    def evaluate_batch(self, trees: Sequence[Any]) -> dict[Tree, float]:
        if not trees:
            return {}

        result: dict[Tree, float] = {}
        candidates: list[Tree] = []

        for item in trees:
            if item is None or not isinstance(item, Tree):
                continue
            if self.known_points is not None and item in self.known_points:
                result[item] = 0.0
                continue
            candidates.append(item)

        if not candidates:
            return result

        mu, sigma = self.gp.predict(candidates, return_std=True)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        sigma = np.asarray(sigma, dtype=float).reshape(-1)

        if self.greater_is_better:
            ucb = mu + self.kappa * sigma
        else:
            ucb = -mu + self.kappa * sigma

        for tree, value in zip(candidates, ucb):
            result[tree] = float(value)

        return result

    def __call__(self, t: Any) -> float:
        if t is None or not isinstance(t, Tree):
            return 0.0
        return float(self.evaluate_batch([t]).get(t, 0.0))


class DiversityUCB(AcquisitionFunction):
    """UCB augmented with a diversity term.

    Score = mean_term + kappa(t) * sigma_norm + lambda_div * diversity_norm

    All three terms are normalised to comparable scales:

    - ``mean_term`` = ±mu (sign depends on ``greater_is_better``),
    - ``sigma_norm`` = sigma / max(sigma) ∈ [0, 1],
    - ``diversity_norm`` = min_dist / max(min_dist) ∈ [0, 1]  (normalized F16).

    ``kappa(t)`` = kappa0 * sqrt(log(iteration + 1)) grows with iteration count
    to promote exploration over time.

    Parameters
    ----------
    gp:
        Fitted ``GaussianProcessRegressor``.
    kernel:
        Structured kernel for computing candidate-to-training-set distances.
    greater_is_better:
        Whether the objective is maximised.
    known_points:
        Known candidates receive score = 0.
    incumbent:
        Unused; kept for API consistency.
    kappa0:
        Base exploration coefficient.
    lambda_div:
        Weight for the diversity term.
    diversity_window:
        Number of most-recent training points used for diversity computation.
    iteration:
        Current BO iteration (fixed at construction time; fixes F15).
    """

    def __init__(
        self,
        gp: GaussianProcessRegressor,
        kernel: Any,
        greater_is_better: bool = False,
        known_points: set[Any] | None = None,
        incumbent: float | None = None,
        kappa0: float = 2.0,
        lambda_div: float = 0.2,
        diversity_window: int = 20,
        iteration: int = 1,
    ) -> None:
        super().__init__(gp, greater_is_better, known_points, incumbent)
        self.kernel = kernel
        self.kappa0 = kappa0
        self.lambda_div = lambda_div
        self.diversity_window = diversity_window
        self.iteration = iteration

    def evaluate_batch(self, trees: Sequence[Any]) -> dict[Tree, float]:
        if not trees:
            return {}

        result: dict[Tree, float] = {}
        candidates: list[Tree] = []

        for item in trees:
            if item is None or not isinstance(item, Tree):
                continue
            if self.known_points is not None and item in self.known_points:
                result[item] = 0.0
                continue
            candidates.append(item)

        if not candidates:
            return result

        mu, sigma = self.gp.predict(candidates, return_std=True)
        mu = np.asarray(mu, dtype=float).reshape(-1)
        sigma = np.asarray(sigma, dtype=float).reshape(-1)

        sigma_norm = sigma / (np.max(sigma) + 1e-12)
        kappa = self.kappa0 * np.sqrt(np.log(self.iteration + 1))

        mean_term = mu if self.greater_is_better else -mu

        # Diversity: min distance to training set
        if self.gp.X_train_ is None or len(self.gp.X_train_) == 0:
            diversity = np.ones(len(candidates))
        else:
            X_train = self.gp.X_train_
            if self.diversity_window is not None:
                X_train = X_train[-self.diversity_window :]

            # Use kernel.diag() to avoid computing full n×n matrices (fixes F15)
            K_xt = self.kernel(candidates, X_train)
            K_xx = self.kernel.diag(candidates)   # shape (n_candidates,)
            K_tt = self.kernel.diag(X_train)       # shape (n_train,)

            d2 = K_xx[:, None] + K_tt[None, :] - 2.0 * K_xt
            d2 = np.maximum(d2, 0.0)
            distances = np.sqrt(d2)
            diversity = np.min(distances, axis=1)

        # Normalize diversity so all three terms are in comparable scales (fixes F16)
        diversity = diversity / (np.max(diversity) + 1e-12)

        scores = mean_term + kappa * sigma_norm + self.lambda_div * diversity

        for tree, value in zip(candidates, scores):
            result[tree] = float(value)

        return result

    def __call__(self, t: Any) -> float:
        if t is None or not isinstance(t, Tree):
            return 0.0
        return float(self.evaluate_batch([t]).get(t, 0.0))
