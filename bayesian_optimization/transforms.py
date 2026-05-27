from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class YTransform(Protocol):
    """Bijective transformation applied to target values before GP fitting.

    Implementors must expose a ``name`` attribute and ``forward``/``inverse``
    methods that form a pair satisfying the round-trip contract:
    ``np.allclose(inverse(forward(y)), y, atol=1e-9)`` for valid inputs.
    """

    name: str

    def forward(self, y: np.ndarray) -> np.ndarray:
        """Map raw objective values y → transformed space."""
        ...

    def inverse(self, y_gp: np.ndarray) -> np.ndarray:
        """Map transformed values back → raw objective scale."""
        ...


class IdentityTransform:
    """No-op transform: y_gp = y."""

    name = "identity"

    def forward(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=float)

    def inverse(self, y_gp: np.ndarray) -> np.ndarray:
        return np.asarray(y_gp, dtype=float)


class Log1pTransform:
    """y_gp = log1p(y).

    Requires y > -1.  A small safety clamp guards against tiny
    negative numerical noise (values in (-clamp_tol, 0) are clamped to 0).

    Parameters
    ----------
    clamp_tol:
        Values < -1 + clamp_tol raise ValueError; values in [-clamp_tol, 0)
        are silently clamped to 0.
    """

    name = "log1p"

    def __init__(self, clamp_tol: float = 1e-12) -> None:
        self.clamp_tol = clamp_tol

    def forward(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        if np.any(y_arr < -1.0 + self.clamp_tol):
            raise ValueError(
                f"Log1pTransform requires y > -1; got min(y) = "
                f"{float(np.min(y_arr)):.6g}. "
                "Use SignedLogTransform or shift the objective."
            )
        return np.log1p(np.clip(y_arr, a_min=0.0, a_max=None))

    def inverse(self, y_gp: np.ndarray) -> np.ndarray:
        return np.expm1(np.asarray(y_gp, dtype=float))


class SignedLogTransform:
    """y_gp = sign(y) * log1p(|y|).

    Handles signed objectives without domain restrictions.
    """

    name = "signed_log"

    def forward(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        return np.sign(y_arr) * np.log1p(np.abs(y_arr))

    def inverse(self, y_gp: np.ndarray) -> np.ndarray:
        z = np.asarray(y_gp, dtype=float)
        return np.sign(z) * np.expm1(np.abs(z))


class StandardizeTransform:
    """y_gp = (y - mean) / std.

    Mean and std are estimated from the first call to ``forward()`` and
    frozen thereafter, so consistent scaling is maintained across the entire
    BO loop.
    """

    name = "standardize"

    def __init__(self) -> None:
        self._mean: float | None = None
        self._std: float | None = None

    def forward(self, y: np.ndarray) -> np.ndarray:
        y_arr = np.asarray(y, dtype=float)
        if self._mean is None:
            self._mean = float(np.mean(y_arr))
            self._std = float(np.std(y_arr)) or 1.0
        mean: float = self._mean  # type: ignore[assignment]
        std: float = self._std  # type: ignore[assignment]
        return (y_arr - mean) / std

    def inverse(self, y_gp: np.ndarray) -> np.ndarray:
        z = np.asarray(y_gp, dtype=float)
        if self._mean is None:
            return z
        mean: float = self._mean  # type: ignore[assignment]
        std: float = self._std  # type: ignore[assignment]
        return z * std + mean
