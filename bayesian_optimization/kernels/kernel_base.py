from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Hashable, Sequence
from typing import Any, Generic, TypeVar

import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
    GenericKernelMixin,
    Kernel,
    NormalizedKernelMixin,
    Hyperparameter,
)

T = TypeVar("T", bound=Hashable)


# ---------------------------------------------------------------------------
# Vendored from sklearn.gaussian_process.kernels to avoid importing a private
# symbol (_approx_fprime was removed from the public API in sklearn 1.x).
# Original: https://github.com/scikit-learn/scikit-learn
# License: BSD-3-Clause
# ---------------------------------------------------------------------------
def _approx_fprime(xk: np.ndarray, f, epsilon: float) -> np.ndarray:
    """Numerically approximate the Jacobian of f at xk."""
    f0 = f(xk)
    f0_arr = np.asarray(f0)
    grad = np.zeros((len(xk), *f0_arr.shape),
                    dtype=np.promote_types(float, f0_arr.dtype))
    ei = np.zeros(len(xk), dtype=float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[k, ...] = (np.asarray(f(xk + d)) - f0_arr) / epsilon
        ei[k] = 0.0
    return np.moveaxis(grad, 0, -1)


class StructuredKernelBase(GenericKernelMixin, NormalizedKernelMixin, Kernel, ABC, Generic[T]):
    """Common sklearn-compatible base class for non-vector structured kernels.

    Subclasses must implement ``_prepare_inputs`` and ``_kernel_matrix``.
    Gradient computation falls back to numerical finite differences via the
    vendored ``_approx_fprime``.
    """

    # The property below covers all access paths sklearn uses.  A bare class
    # attribute would conflict with it under mypy's no-redef rule.

    def __init__(
        self,
        epsilon: float = 1e-10,
        epsilon_bounds: tuple[float, float] = (1e-12, 1e-8),
    ) -> None:
        self.epsilon = epsilon
        self.epsilon_bounds = epsilon_bounds

    @property
    def requires_vector_input(self) -> bool:  # type: ignore[override]
        return False

    @abstractmethod
    def _prepare_inputs(self, X: Sequence[T]) -> Sequence[Any]:
        """Translate raw structured inputs into backend-specific objects."""

    @abstractmethod
    def _kernel_matrix(
        self, X_prepared: Sequence[Any], Y_prepared: Sequence[Any]
    ) -> np.ndarray:
        """Compute the (n_x × n_y) kernel matrix for prepared inputs."""

    def __call__(self, X: Any, Y: Any = None, eval_gradient: bool = False) -> Any:
        X_prepared = self._prepare_inputs(X)
        Y_prepared = X_prepared if Y is None else self._prepare_inputs(Y)
        K = np.asarray(self._kernel_matrix(X_prepared, Y_prepared))

        if not eval_gradient:
            return K

        if self.theta.size == 0:
            return K, np.empty((K.shape[0], K.shape[1], 0))

        def f(theta: np.ndarray) -> np.ndarray:
            return self.clone_with_theta(theta)(X, Y)

        grad = _approx_fprime(self.theta, f, self.epsilon)
        return K, grad

    def diag(self, X: Any) -> np.ndarray:
        return np.diag(self(X))

    def is_stationary(self) -> bool:
        return False
