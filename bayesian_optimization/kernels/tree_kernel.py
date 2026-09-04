from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Generic, TypeVar

import numpy as np
from cosy.core.tree import Tree
from cosy.search.kernels import k_sst
from scipy import sparse

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

    which keeps large terms from dominating the scale, since the self-similarity of a counting
    kernel grows with the size of the term.  Every subtree of the term contributes.  The subtree
    kernel weights each shared subterm by a positive weight that depends on that subterm alone
    and truncates nothing at a height, so the former ``max_height`` parameter, which cut
    signatures off at a remaining depth and kept only the node label at 0, had no counterpart in
    the definition and is gone.
    """

    def __init__(
        self,
        normalize: bool = True,
        tree_transformation: Callable[[Tree[T]], Tree[T]] | None = None,
        epsilon: float = 1e-10,
        epsilon_bounds: tuple[float, float] = (1e-12, 1e-8),
    ) -> None:
        super().__init__(epsilon=epsilon, epsilon_bounds=epsilon_bounds)
        self.normalize = normalize
        self.tree_transformation = (
            tree_transformation
            if tree_transformation is not None
            else _identity_tree_transformation
        )
        self._feature_cache: dict[Any, Counter[Any]] = {}
        self._signature_cache: dict[Any, tuple[Any, Counter[Any]]] = {}

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
            return np.asarray(K, dtype=float)

        x_norms = np.array(X_mat.multiply(X_mat).sum(axis=1)).reshape(-1)
        y_norms = np.array(Y_mat.multiply(Y_mat).sum(axis=1)).reshape(-1)
        denom = np.sqrt(np.asarray(np.outer(x_norms, y_norms), dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            K = np.divide(K, denom, out=np.zeros_like(K), where=denom > 0)
        return np.asarray(K, dtype=float)

    def diag(self, X: Any) -> np.ndarray:
        """Return the self-similarities without building the full matrix.

        Args:
            X: The terms.

        Returns:
            np.ndarray: One value per element of ``X``.
        """
        if self.normalize:
            # Every term has at least the signature of its own root, so no self-similarity is
            # zero and the cosine normalization puts all of them at exactly 1.
            return np.ones(len(X), dtype=float)
        return np.array(
            [float(sum(count * count for count in features.values()))
             for features in self._prepare_inputs(list(X))],
            dtype=float,
        )

    def _features(self, tree: Tree[T]) -> Counter[Any]:
        cached = self._feature_cache.get(tree)
        if cached is not None:
            return cached
        _, features = self._signature_and_features(tree)
        self._feature_cache[tree] = features
        return features

    def _signature_and_features(self, tree: Tree[T]) -> tuple[Any, Counter[Any]]:
        cached = self._signature_cache.get(tree)
        if cached is not None:
            return cached

        child_results = [self._signature_and_features(child) for child in tree.children]
        child_signatures = tuple(sig for sig, _ in child_results)
        signature: Any = (tree.root, child_signatures)
        features: Counter[Any] = Counter({signature: 1})
        for _, child_features in child_results:
            features.update(child_features)

        self._signature_cache[tree] = (signature, features)
        return signature, features


class SubsetTreeKernel(StructuredKernelBase[T], Generic[T]):
    """Kernel that counts the subset trees two terms share, with decay one.

    Where :class:`OrderedRootedSubtreeKernel` counts *complete* subtrees, this one counts subset
    trees: fragments that may stop at any position, but keep all of a retained position's
    children.  The unit of comparison is therefore the production, a symbol together with the
    symbols of its children, and two positions with the same symbol over different children
    share nothing.

    A kernel for a surrogate is judged on two counts, validity and fit, and the two pull in
    different directions here, which is why both kernels are offered rather than one.  Both are
    valid, because a finite convolution of kernels is again a kernel by Haussler's theorem, and
    both count substructures that way.  Which one *fits* is not a formal question but a
    diagnostic one, and the answer depends on the space.

    The count itself is cosy's ``k_sst``: the same definition that scores the candidates of the
    kernel-diverse initialization, and cosy pins it there against a hand-counted branch on which
    filling a hole raises the score from zero to one.  Keeping one implementation means the
    surrogate and the initializer cannot drift apart on what a subset tree is.

    On terms of a hundred nodes the normalized Gram matrix of this kernel can already be
    numerically the identity.  A position contributes the product over its children of one plus
    their own contribution, so almost all of a term's self-similarity sits in the fragments that
    reach down from its top, and a balanced binary term of 127 nodes scores 2.1e11 against itself.
    A shared fragment has to match at every position it spans, so one differing production removes
    every fragment through it, and where two terms differ decides the entry as much as how large
    they are: on that same term a changed leaf symbol still leaves 0.62, while a changed symbol at
    the root leaves 8.8e-06.  Ten such terms that differ in the symbols nearest their root give an
    off-diagonal mean of 7e-05 with every eigenvalue within 2e-3 of one.  Both counts are whole
    numbers well inside what a double holds exactly, so this is the kernel's arithmetic and not a
    rounding effect, and a surrogate conditioned on such a matrix reproduces the terms it has
    observed and predicts the prior mean everywhere else.  See
    ``test_the_subset_tree_gram_matrix_collapses_on_large_terms``.

    Where a search space sits on that scale is read off its own terms, which is what
    :func:`~bayesian_optimization.diagnostics.read_gram` reports as ``off_diagonal_mean`` and
    ``condition_number``.  Neither parameter below reaches the mechanism.  ``normalize`` divides
    by the self-similarities and the collapse is that quotient, so switching it off leaves the raw
    scale in without moving the ratios.  A ``tree_transformation`` changes which terms are scored
    rather than how they are scored, and a fold that shrinks the terms can move where a space sits
    without changing that.  What damps the large fragments a self-similarity is built from is a
    decay below one, and cosy's ``k_sst`` fixes the decay at one.

    Parameters
    ----------
    normalize:
        Cosine normalization of the Gram matrix, which keeps large terms from dominating the
        scale.  A single node roots no subset tree, so its unnormalized self-similarity is zero,
        and those rows and columns stay zero under normalization rather than dividing by it.
    tree_transformation:
        An optional fold applied before scoring, the same freedom the WL kernel has.
    """

    def __init__(
        self,
        normalize: bool = True,
        tree_transformation: Callable[[Tree[T]], Tree[T]] | None = None,
        epsilon: float = 1e-10,
        epsilon_bounds: tuple[float, float] = (1e-12, 1e-8),
    ) -> None:
        super().__init__(epsilon=epsilon, epsilon_bounds=epsilon_bounds)
        self.normalize = normalize
        self.tree_transformation = (
            tree_transformation
            if tree_transformation is not None
            else _identity_tree_transformation
        )

    def _prepare_inputs(self, X: Sequence[Tree[T]]) -> Sequence[Tree[T]]:  # type: ignore[override]
        return [self.tree_transformation(tree) for tree in X]

    def _kernel_matrix(
        self,
        X_prepared: Sequence[Tree[T]],
        Y_prepared: Sequence[Tree[T]],
    ) -> np.ndarray:
        raw = np.array(
            [[k_sst(x, y) for y in Y_prepared] for x in X_prepared], dtype=float
        ).reshape(len(X_prepared), len(Y_prepared))
        if not self.normalize:
            return raw

        x_self = np.array([k_sst(x, x) for x in X_prepared], dtype=float)
        y_self = np.array([k_sst(y, y) for y in Y_prepared], dtype=float)
        denominator = np.sqrt(np.outer(x_self, y_self))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.divide(
                raw, denominator, out=np.zeros_like(raw), where=denominator > 0
            )

    def diag(self, X: Any) -> np.ndarray:
        """Return the self-similarities without building the full matrix.

        Args:
            X: The terms.

        Returns:
            np.ndarray: One value per element of ``X``.  Under normalization a term that roots no
                subset tree scores 0 rather than 1, because the normalization cannot divide by
                zero, and claiming similarity 1 for a term the kernel cannot see would be worse.
        """
        prepared = self._prepare_inputs(list(X))
        raw = np.array([k_sst(tree, tree) for tree in prepared], dtype=float)
        if not self.normalize:
            return raw
        return np.where(raw > 0.0, 1.0, 0.0)
