from __future__ import annotations

from abc import ABC
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Generic, TypeVar

import numpy as np
from cosy.core.tree import Tree
from grakel.kernels import VertexHistogram, WeisfeilerLehman
from sklearn.gaussian_process.kernels import Hyperparameter

from ..utils import to_grakel_graph as to_tree_graph
from .kernel_base import StructuredKernelBase

T = TypeVar("T", bound=Hashable)


def _materialize_graph(graph_source: Any) -> Any:
    if hasattr(graph_source, "__next__"):
        return next(graph_source)
    return graph_source


class _GrakelWeisfeilerLehmanBase(StructuredKernelBase[T], ABC):
    def __init__(
        self,
        base_graph_kernel: Any = VertexHistogram,
        normalize: bool = True,
        n_jobs: int | None = None,
        epsilon: float = 1e-10,
        epsilon_bounds: tuple[float, float] = (1e-12, 1e-8),
    ) -> None:
        super().__init__(epsilon=epsilon, epsilon_bounds=epsilon_bounds)
        self.base_graph_kernel = base_graph_kernel
        self.normalize = normalize
        self.n_jobs = n_jobs
        self._kernel_matrix_cache: dict[
            tuple[tuple[int, ...], tuple[int, ...], int], np.ndarray
        ] = {}

    def _build_kernel(self, n_iter: int) -> WeisfeilerLehman:
        return WeisfeilerLehman(
            n_iter=max(1, int(n_iter)),
            base_graph_kernel=self.base_graph_kernel,
            normalize=self.normalize,
            n_jobs=self.n_jobs,
        )

    def _kernel_from_graphs(
        self,
        X_graphs: Sequence[Any],
        Y_graphs: Sequence[Any],
        n_iter: int,
    ) -> np.ndarray:
        if len(X_graphs) == 0 or len(Y_graphs) == 0:
            return np.zeros((len(X_graphs), len(Y_graphs)), dtype=float)
        key = (
            tuple(id(g) for g in X_graphs),
            tuple(id(g) for g in Y_graphs),
            int(n_iter),
        )
        if key in self._kernel_matrix_cache:
            return self._kernel_matrix_cache[key]

        wl_kernel = self._build_kernel(n_iter)
        if X_graphs is Y_graphs:
            K = np.asarray(wl_kernel.fit_transform(X_graphs))
        else:
            wl_kernel.fit_transform(Y_graphs)
            K = np.asarray(wl_kernel.transform(X_graphs))

        try:
            self._kernel_matrix_cache[key] = K
        except Exception:
            pass
        return K


class WeisfeilerLehmanKernel(_GrakelWeisfeilerLehmanBase, Generic[T]):
    """Weisfeiler-Lehman kernel for trees via a graph backend."""

    def __init__(
        self,
        n_iter: float = 1.0,
        base_graph_kernel: Any = VertexHistogram,
        normalize: bool = True,
        to_grakel_graph: Callable[[Tree[T]], Any] = to_tree_graph,
        n_jobs: int | None = None,
        epsilon: float = 1e-10,
        epsilon_bounds: tuple[float, float] = (1e-12, 1e-8),
    ) -> None:
        super().__init__(
            base_graph_kernel=base_graph_kernel,
            normalize=normalize,
            n_jobs=n_jobs,
            epsilon=epsilon,
            epsilon_bounds=epsilon_bounds,
        )
        self.n_iter: float = 1.0 if n_iter < 1.0 else (n_iter if n_iter <= 5.0 else 5.0)
        self.to_grakel_graph = to_grakel_graph
        self._graph_cache: dict[Any, Any] = {}

    def _graph_from_tree(self, tree: Tree[T]) -> Any:
        cached = self._graph_cache.get(tree)
        if cached is not None:
            return cached
        graph = _materialize_graph(self.to_grakel_graph(tree))
        self._graph_cache[tree] = graph
        return graph

    def _prepare_inputs(self, X: Sequence[Tree[T]]) -> Sequence[Any]:
        return [self._graph_from_tree(t) for t in X]

    def _kernel_matrix(
        self, X_prepared: Sequence[Any], Y_prepared: Sequence[Any]
    ) -> np.ndarray:
        return self._kernel_from_graphs(X_prepared, Y_prepared, int(self.n_iter))
