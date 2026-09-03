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
    return gp, float(y.max())


@pytest.mark.property
@given(
    train_trees=st.lists(small_trees(), min_size=2, max_size=5),
    query_tree=small_trees(),
)
@settings(max_examples=30, deadline=None)
def test_ei_always_non_negative(train_trees, query_tree):
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    gp, incumbent = _make_gp(train_trees)
    ei = ExpectedImprovement(gp=gp, incumbent=incumbent)
    val = ei(query_tree)
    # No tolerance: the closed form is non-negative mathematically, and the implementation clips
    # what cancellation leaves behind, so a negative value here is a defect, not a rounding.
    assert val >= 0.0, f"EI={val} < 0"


@pytest.mark.property
@given(
    train_trees=st.lists(small_trees(), min_size=2, max_size=5),
    query_tree=small_trees(),
)
@settings(max_examples=30, deadline=None)
def test_pi_is_a_probability(train_trees, query_tree):
    """Probability of improvement is a probability mass, so it lives in ``[0, 1]``.

    The score is the posterior probability that the value at the queried tree exceeds the
    threshold, and that holds whatever the posterior looks like.
    """
    from bayesian_optimization.acquisition_function import ProbabilityOfImprovement

    gp, incumbent = _make_gp(train_trees)
    pi = ProbabilityOfImprovement(gp=gp, incumbent=incumbent)
    val = pi(query_tree)
    assert 0.0 <= val <= 1.0, f"PI={val} is not a probability"


@pytest.mark.property
@given(
    train_trees=st.lists(small_trees(), min_size=2, max_size=5),
    query_tree=small_trees(),
)
@settings(max_examples=30, deadline=None)
def test_ucb_finite_for_finite_inputs(train_trees, query_tree):
    from bayesian_optimization.acquisition_function import UpperConfidenceBound

    gp, _ = _make_gp(train_trees)
    ucb = UpperConfidenceBound(gp=gp, beta=2.0)
    val = ucb(query_tree)
    assert np.isfinite(val), f"UCB={val} is not finite"


@pytest.mark.property
@given(
    train_trees=st.lists(small_trees(), min_size=2, max_size=5),
    query_tree=small_trees(),
)
@settings(max_examples=30, deadline=None)
def test_a_known_point_never_outranks_a_novel_one(train_trees, query_tree):
    """The floor of all three acquisitions has to hold for every batch, not just the ones we picked.

    The test that stood here covered DiversityUCB, which was dropped as related work: the method
    has exactly three acquisitions, expected improvement, probability of improvement, and upper
    confidence bound.  What is worth generating instead is the property the fixed sentinel used
    to violate.
    """
    from bayesian_optimization.acquisition_function import (
        ExpectedImprovement,
        ProbabilityOfImprovement,
        UpperConfidenceBound,
    )

    gp, incumbent = _make_gp(train_trees)
    known = set(train_trees)
    if query_tree in known:
        return

    for af in (
        ExpectedImprovement(gp=gp, incumbent=incumbent, known_points=known),
        ProbabilityOfImprovement(gp=gp, incumbent=incumbent, known_points=known),
        UpperConfidenceBound(gp=gp, beta=2.0, known_points=known),
    ):
        scores = af.evaluate_batch([*train_trees, query_tree])
        assert scores[query_tree] > max(scores[t] for t in known)
        assert all(np.isfinite(v) for v in scores.values())
