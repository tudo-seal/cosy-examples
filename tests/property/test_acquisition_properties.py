from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.gaussian_process import GaussianProcessRegressor

from tests.property._strategies import small_trees


def _make_gp(trees):
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel
    kernel = OrderedRootedSubtreeKernel(normalize=True)
    X = np.asarray(trees, dtype=object)
    y = np.arange(1, len(trees) + 1, dtype=float)
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5, normalize_y=True, optimizer=None)
    gp.fit(X, y)
    return gp


@pytest.mark.property
@given(
    train_trees=st.lists(small_trees(), min_size=2, max_size=5),
    query_tree=small_trees(),
)
@settings(max_examples=30, deadline=None)
def test_ei_always_non_negative(train_trees, query_tree):
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    gp = _make_gp(train_trees)
    ei = ExpectedImprovement(gp=gp, xi=0.01, greater_is_better=False)
    val = ei(query_tree)
    assert val >= -1e-10, f"EI={val} < 0"


@pytest.mark.property
@given(
    train_trees=st.lists(small_trees(), min_size=2, max_size=5),
    query_tree=small_trees(),
)
@settings(max_examples=30, deadline=None)
def test_ucb_finite_for_finite_inputs(train_trees, query_tree):
    from bayesian_optimization.acquisition_function import UpperConfidenceBound

    gp = _make_gp(train_trees)
    ucb = UpperConfidenceBound(gp=gp, greater_is_better=False, kappa=2.0)
    val = ucb(query_tree)
    assert np.isfinite(val), f"UCB={val} is not finite"


@pytest.mark.property
@given(
    train_trees=st.lists(small_trees(), min_size=2, max_size=5),
    query_tree=small_trees(),
)
@settings(max_examples=30, deadline=None)
def test_diversity_ucb_finite(train_trees, query_tree):
    from bayesian_optimization.acquisition_function import DiversityUCB
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    gp = _make_gp(train_trees)
    kernel = OrderedRootedSubtreeKernel(normalize=True)
    ducb = DiversityUCB(gp=gp, kernel=kernel, greater_is_better=False,
                        kappa0=2.0, lambda_div=0.3, iteration=1)
    val = ducb(query_tree)
    assert np.isfinite(val), f"DiversityUCB={val} is not finite"
