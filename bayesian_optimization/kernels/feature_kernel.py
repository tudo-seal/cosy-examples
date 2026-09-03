from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any, Generic, TypeVar

import numpy as np
from cosy.core.tree import Tree
from sklearn.gaussian_process.kernels import RBF, Hyperparameter

from .kernel_base import StructuredKernelBase

T = TypeVar("T", bound=Hashable)


class RBF_FeatureKernel(StructuredKernelBase[T]):
    """RBF kernel operating on algebraic feature vectors extracted from Trees.

    Each tree is mapped to a numeric vector via ``feature_algebra`` before the
    standard RBF kernel is applied.

    Parameters
    ----------
    feature_algebra:
        Callable that accepts a ``Tree`` and returns a 1-D numeric array.
    length_scale:
        Initial length scale of the RBF kernel.
    length_scale_bounds:
        Bounds for hyperparameter optimisation.
    normalize:
        Unused (kept for API compatibility with other structured kernels).
    """

    def __init__(
        self,
        feature_algebra: Any,
        length_scale: float = 1.0,
        length_scale_bounds: tuple[float, float] = (1e-5, 1e5),
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.feature_algebra = feature_algebra
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds
        self.normalize = normalize
        self._feature_cache: dict[Any, Any] = {}

    @property
    def hyperparameter_length_scale(self) -> Hyperparameter:
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds, fixed=False)

    def _feature_from_tree(self, tree: Tree[T]) -> Any:
        cached = self._feature_cache.get(tree)
        if cached is not None:
            return cached
        feature = tree.interpret(self.feature_algebra)
        self._feature_cache[tree] = feature
        return feature

    def _prepare_inputs(self, X: Sequence[T]) -> Sequence[Any]:
        return [self._feature_from_tree(t) for t in X]  # type: ignore[arg-type]

    def _kernel_matrix(
        self, X_prepared: Sequence[Any], Y_prepared: Sequence[Any]
    ) -> np.ndarray:
        kernel = RBF(
            length_scale=self.length_scale,
            length_scale_bounds=self.length_scale_bounds,
        )
        Xp = np.array(X_prepared)
        Yp = np.array(Y_prepared)
        return kernel(Xp, Yp, eval_gradient=False)

    def __call__(self, X: Any, Y: Any = None, eval_gradient: bool = False) -> Any:
        kernel = RBF(
            length_scale=self.length_scale,
            length_scale_bounds=self.length_scale_bounds,
        )
        X_prepared = self._prepare_inputs(X)
        Y_prepared = None if Y is None else self._prepare_inputs(Y)
        Xp = np.array(X_prepared)
        Yp = np.array(Y_prepared) if Y_prepared is not None else None
        return kernel(Xp, Yp, eval_gradient=eval_gradient)
