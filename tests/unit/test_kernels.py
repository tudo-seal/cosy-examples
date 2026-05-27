from __future__ import annotations

import numpy as np
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor

KERNEL_FACTORIES = [
    lambda: __import__("bayesian_optimization.kernels.tree_kernel", fromlist=["OrderedRootedSubtreeKernel"]).OrderedRootedSubtreeKernel(normalize=True),
    lambda: __import__("bayesian_optimization.kernels.tree_kernel", fromlist=["OrderedRootedSubtreeKernel"]).OrderedRootedSubtreeKernel(normalize=False),
    lambda: __import__("bayesian_optimization.kernels.tree_kernel", fromlist=["OrderedRootedSubtreeKernel"]).OrderedRootedSubtreeKernel(normalize=True, max_height=2),
    lambda: __import__("bayesian_optimization.kernels.graph_kernel", fromlist=["WeisfeilerLehmanKernel"]).WeisfeilerLehmanKernel(n_iter=2),
]

KERNEL_IDS = [
    "subtree_normalized",
    "subtree_unnormalized",
    "subtree_max_height2",
    "wl_n_iter2",
]


@pytest.mark.parametrize("kernel_factory", KERNEL_FACTORIES, ids=KERNEL_IDS)
def test_kernel_symmetry(kernel_factory, tree_corpus):
    kernel = kernel_factory()
    K = kernel(tree_corpus)
    assert np.allclose(K, K.T, atol=1e-8), "Kernel matrix is not symmetric"


@pytest.mark.parametrize("kernel_factory", KERNEL_FACTORIES, ids=KERNEL_IDS)
def test_kernel_positive_semidefinite(kernel_factory, tree_corpus):
    kernel = kernel_factory()
    K = kernel(tree_corpus)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-8), f"Kernel is not PSD; min eigenvalue = {eigvals.min():.2e}"


@pytest.mark.parametrize("kernel_factory", KERNEL_FACTORIES, ids=KERNEL_IDS)
def test_kernel_diagonal_correct(kernel_factory, tree_corpus):
    kernel = kernel_factory()
    K = kernel(tree_corpus)
    diag_direct = kernel.diag(tree_corpus)
    assert np.allclose(diag_direct, np.diag(K), atol=1e-8)


@pytest.mark.parametrize("kernel_factory", KERNEL_FACTORIES, ids=KERNEL_IDS)
def test_kernel_self_vs_cross_consistency(kernel_factory, tree_corpus):
    """K(X) must equal K(X, X)."""
    kernel = kernel_factory()
    K_self = kernel(tree_corpus)
    K_cross = kernel(tree_corpus, tree_corpus)
    assert np.allclose(K_self, K_cross, atol=1e-8)


@pytest.mark.parametrize("kernel_factory", KERNEL_FACTORIES, ids=KERNEL_IDS)
def test_kernel_works_with_gp(kernel_factory, tree_corpus):
    kernel = kernel_factory()
    X = np.asarray(tree_corpus, dtype=object)
    y = np.array([float(i) * 0.5 for i in range(len(tree_corpus))])
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False, optimizer=None)
    gp.fit(X, y)
    preds = gp.predict(X)
    assert preds.shape == (len(tree_corpus),)
    assert np.all(np.isfinite(preds))


def test_kernel_caching_consistency(tree_corpus):
    """Two separate kernel instances should produce the same matrix."""
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    k1 = OrderedRootedSubtreeKernel(normalize=True)
    k2 = OrderedRootedSubtreeKernel(normalize=True)
    K1 = k1(tree_corpus)
    K2 = k2(tree_corpus)
    assert np.allclose(K1, K2, atol=1e-10)


