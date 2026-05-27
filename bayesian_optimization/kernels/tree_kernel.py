from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Generic, TypeVar

import numpy as np
from scipy import sparse

from cosy.core.tree import Tree

from .kernel_base import StructuredKernelBase

T = TypeVar("T", bound=Hashable)


def _identity_tree_transformation(tree: Tree[T]) -> Tree[T]:
    return tree


class OrderedRootedSubtreeKernel(StructuredKernelBase[T], Generic[T]):
    """Kernel that counts identical ordered rooted subtrees.

    Let phi_s(t) be the number of occurrences of subtree-signature s in tree t.
    The unnormalized kernel is

        K(t_i, t_j) = sum_s phi_s(t_i) * phi_s(t_j)

    where signatures are computed recursively from the root label and the ordered
    list of child signatures.  Sibling order affects similarity.

    If ``normalize=True`` (default) the cosine-normalized form is used::

        K_norm(t_i, t_j) = K(t_i, t_j) / sqrt(K(t_i, t_i) * K(t_j, t_j))

    ``max_height`` truncates signatures at the given remaining depth. At depth 0
    only the node label is retained.
    """

    def __init__(
        self,
        max_height: int | None = None,
        normalize: bool = True,
        tree_transformation: Callable[[Tree[T]], Tree[T]] | None = None,
        epsilon: float = 1e-10,
        epsilon_bounds: tuple[float, float] = (1e-12, 1e-8),
    ) -> None:
        super().__init__(epsilon=epsilon, epsilon_bounds=epsilon_bounds)
        self.max_height = None if max_height is None else max(0, int(max_height))
        self.normalize = normalize
        self.tree_transformation = (
            tree_transformation
            if tree_transformation is not None
            else _identity_tree_transformation
        )
        self._feature_cache: dict[Any, Counter[Any]] = {}
        self._signature_cache: dict[tuple[Any, int | None], tuple[Any, Counter[Any]]] = {}

    def _prepare_inputs(self, X: Sequence[Tree[T]]) -> Sequence[Counter[Any]]:  # type: ignore[override]
        return [self._features(self.tree_transformation(tree)) for tree in X]

    def _kernel_matrix(
        self,
        X_prepared: Sequence[Counter[Any]],
        Y_prepared: Sequence[Counter[Any]],
    ) -> np.ndarray:
        signatures: dict[Any, int] = {}

        def collect(features_list: Sequence[Counter[Any]]) -> None:
            for features in features_list:
                for sig in features:
                    if sig not in signatures:
                        signatures[sig] = len(signatures)

        collect(X_prepared)
        collect(Y_prepared)

        n_x, n_y, n_feat = len(X_prepared), len(Y_prepared), len(signatures)
        if n_feat == 0:
            return np.zeros((n_x, n_y), dtype=float)

        def build_sparse(features_list: Sequence[Counter[Any]], n_rows: int):
            data, rows, cols = [], [], []
            for i, features in enumerate(features_list):
                for sig, count in features.items():
                    rows.append(i)
                    cols.append(signatures[sig])
                    data.append(count)
            return sparse.csr_matrix((data, (rows, cols)), shape=(n_rows, n_feat), dtype=float)

        X_mat = build_sparse(X_prepared, n_x)
        Y_mat = build_sparse(Y_prepared, n_y)
        K = X_mat.dot(Y_mat.T).toarray()

        if not self.normalize:
            return K

        x_norms = np.array(X_mat.multiply(X_mat).sum(axis=1)).reshape(-1)
        y_norms = np.array(Y_mat.multiply(Y_mat).sum(axis=1)).reshape(-1)
        denom = np.sqrt(np.asarray(np.outer(x_norms, y_norms), dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            K = np.divide(K, denom, out=np.zeros_like(K), where=denom > 0)
        return K

    def _features(self, tree: Tree[T]) -> Counter[Any]:
        cached = self._feature_cache.get(tree)
        if cached is not None:
            return cached
        _, features = self._signature_and_features(tree, self.max_height)
        self._feature_cache[tree] = features
        return features

    def _signature_and_features(
        self, tree: Tree[T], remaining_height: int | None
    ) -> tuple[Any, Counter[Any]]:
        cache_key = (tree, remaining_height)
        cached = self._signature_cache.get(cache_key)
        if cached is not None:
            return cached

        if remaining_height is not None and remaining_height <= 0:
            signature: Any = tree.root
            features: Counter[Any] = Counter({signature: 1})
            self._signature_cache[cache_key] = (signature, features)
            return signature, features

        child_results = [
            self._signature_and_features(
                child,
                None if remaining_height is None else remaining_height - 1,
            )
            for child in tree.children
        ]
        child_signatures = tuple(sig for sig, _ in child_results)
        signature = (tree.root, child_signatures)
        features = Counter({signature: 1})
        for _, child_features in child_results:
            features.update(child_features)

        self._signature_cache[cache_key] = (signature, features)
        return signature, features
