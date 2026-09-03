from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.property._strategies import small_trees


@pytest.mark.property
@given(trees=st.lists(small_trees(), min_size=2, max_size=6))
@settings(max_examples=30, deadline=None)
def test_kernel_psd_for_arbitrary_trees(trees):
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    kernel = OrderedRootedSubtreeKernel(normalize=True)
    K = kernel(trees)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-7), f"Kernel not PSD; min eigenvalue: {eigvals.min():.2e}"


@pytest.mark.property
@given(trees=st.lists(small_trees(), min_size=2, max_size=6))
@settings(max_examples=30, deadline=None)
def test_normalized_kernel_diagonal_is_one(trees):
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    kernel = OrderedRootedSubtreeKernel(normalize=True)
    K = kernel(trees)
    diag = np.diag(K)
    assert np.allclose(diag, 1.0, atol=1e-8), f"Diagonal not all 1.0: {diag}"


@pytest.mark.property
@given(trees=st.lists(small_trees(), min_size=2, max_size=6))
@settings(max_examples=30, deadline=None)
def test_kernel_symmetry_for_arbitrary_trees(trees):
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    kernel = OrderedRootedSubtreeKernel(normalize=True)
    K = kernel(trees)
    assert np.allclose(K, K.T, atol=1e-8), "Kernel not symmetric"
