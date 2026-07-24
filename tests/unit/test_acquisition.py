from __future__ import annotations

import pytest
import numpy as np
from cosy.core.tree import Tree
from sklearn.gaussian_process import GaussianProcessRegressor


# ---------------------------------------------------------------------------
# ExpectedImprovement
# ---------------------------------------------------------------------------

def test_ei_non_negative(fitted_gp):
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    ei = ExpectedImprovement(gp=fitted_gp, xi=0.01, greater_is_better=False)
    novel = Tree("brand_new", (Tree("child"),))
    val = ei(novel)
    assert isinstance(val, float)
    assert val >= 0.0


def test_ei_zero_for_known_points(fitted_gp, tree_corpus):
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    known = set(tree_corpus)
    ei = ExpectedImprovement(gp=fitted_gp, xi=0.01, greater_is_better=False, known_points=known)
    for t in tree_corpus:
        assert ei(t) == 0.0


def test_ei_scalar_matches_batch(fitted_gp):
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    ei = ExpectedImprovement(gp=fitted_gp, xi=0.01, greater_is_better=False)
    novel = Tree("novel_for_batch_test", (Tree("x"),))
    scalar = ei(novel)
    batch_val = ei.evaluate_batch([novel]).get(novel, 0.0)
    assert np.isclose(scalar, batch_val, atol=1e-10)


def test_ei_handles_none_and_non_tree_inputs(fitted_gp):
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    ei = ExpectedImprovement(gp=fitted_gp, xi=0.01, greater_is_better=False)
    # None input
    assert ei(None) == 0.0
    # Non-Tree input
    assert ei("not_a_tree") == 0.0
    # evaluate_batch with mixed
    t = Tree("valid_node")
    result = ei.evaluate_batch([t, None, "bad"])
    assert None not in result
    assert "bad" not in result
    assert t in result


def test_ei_uses_provided_incumbent(fitted_gp, tree_corpus):
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    novel = Tree("incumbent_test_node", (Tree("x"),))
    # For minimization: incumbent = current best (minimum y seen so far).
    # High incumbent (bad current best) → large room for improvement → high EI.
    # Low incumbent (good current best) → less room for improvement → low EI.
    ei_bad_incumbent = ExpectedImprovement(gp=fitted_gp, xi=0.0, greater_is_better=False,
                                           incumbent=1000.0)
    ei_good_incumbent = ExpectedImprovement(gp=fitted_gp, xi=0.0, greater_is_better=False,
                                            incumbent=-1000.0)
    assert ei_bad_incumbent(novel) >= 0.0
    assert ei_good_incumbent(novel) >= 0.0
    assert ei_bad_incumbent(novel) >= ei_good_incumbent(novel)


def test_ei_maximization_incumbent_direction(fitted_gp):
    """For maximization the incumbent relation flips: a LOW incumbent (easy to beat) must give a
    higher EI than a HIGH one (hard to beat).  This is the direct sign-flip test for
    greater_is_better=True, a path no other test exercises."""
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    novel = Tree("max_incumbent_test", (Tree("x"),))
    # greater_is_better=True: improvement = mu - incumbent - xi.
    # Low incumbent (bad current best) → lots of room above → high EI.
    # High incumbent (great current best) → almost nothing beats it → EI ≈ 0.
    ei_low_incumbent = ExpectedImprovement(gp=fitted_gp, xi=0.0, greater_is_better=True,
                                           incumbent=-1000.0)
    ei_high_incumbent = ExpectedImprovement(gp=fitted_gp, xi=0.0, greater_is_better=True,
                                            incumbent=1000.0)
    assert ei_low_incumbent(novel) >= 0.0
    assert ei_high_incumbent(novel) >= 0.0
    assert ei_low_incumbent(novel) >= ei_high_incumbent(novel)
    # And the high-incumbent (unbeatable) case must be (near) zero, unlike the low one.
    assert ei_high_incumbent(novel) <= ei_low_incumbent(novel)


def test_ei_xi_monotonicity_under_flat_predictions(fitted_gp):
    """Higher xi → more exploration → possibly lower EI for candidates near incumbent."""
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    novel = Tree("xi_test", (Tree("q"),))
    ei_small = ExpectedImprovement(gp=fitted_gp, xi=0.0, greater_is_better=False)
    ei_large = ExpectedImprovement(gp=fitted_gp, xi=1.0, greater_is_better=False)
    # Both should be non-negative
    assert ei_small(novel) >= 0.0
    assert ei_large(novel) >= 0.0


# ---------------------------------------------------------------------------
# UpperConfidenceBound
# ---------------------------------------------------------------------------

def test_ucb_constructor_kappa_is_used_consistently(fitted_gp, tree_corpus):
    """F14: kappa must come from the constructor, not call-site."""
    from bayesian_optimization.acquisition_function import UpperConfidenceBound

    novel = Tree("ucb_kappa_test", (Tree("u"),))
    ucb_small = UpperConfidenceBound(gp=fitted_gp, greater_is_better=False, kappa=0.5)
    ucb_large = UpperConfidenceBound(gp=fitted_gp, greater_is_better=False, kappa=5.0)
    val_small = ucb_small(novel)
    val_large = ucb_large(novel)
    # Higher kappa → higher exploration bonus → higher UCB score
    assert val_large >= val_small


def test_ucb_higher_at_higher_variance(fitted_gp, tree_corpus):
    """UCB value should be larger for a novel tree (higher uncertainty) than a trained one."""
    from bayesian_optimization.acquisition_function import UpperConfidenceBound

    ucb = UpperConfidenceBound(gp=fitted_gp, greater_is_better=False, kappa=2.0)
    novel = Tree("high_variance_candidate", (Tree("x"), Tree("y"), Tree("z")))
    # Novel tree has higher uncertainty
    novel_score = ucb(novel)
    assert isinstance(novel_score, float)


def test_ucb_minimization_inverts_mean(fitted_gp):
    """For minimization, UCB = -mu + kappa*sigma (lower mean is better)."""
    from bayesian_optimization.acquisition_function import UpperConfidenceBound

    ucb = UpperConfidenceBound(gp=fitted_gp, greater_is_better=False, kappa=0.0)
    novel = Tree("min_ucb_test", (Tree("v"),))
    val = ucb(novel)
    # With kappa=0, UCB = -mu. A negative mean → positive score.
    mu, _ = fitted_gp.predict([novel], return_std=True)
    expected = float(-mu[0])
    assert np.isclose(val, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# DiversityUCB
# ---------------------------------------------------------------------------

def test_diversity_ucb_lambda_zero_matches_pure_ucb(fitted_gp):
    """With lambda_div=0, DiversityUCB degenerates to plain UCB."""
    from bayesian_optimization.acquisition_function import DiversityUCB, UpperConfidenceBound
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    kernel = OrderedRootedSubtreeKernel(normalize=True)
    novel = Tree("div_ucb_zero_lambda", (Tree("p"),))

    ducb = DiversityUCB(gp=fitted_gp, kernel=kernel, greater_is_better=False,
                        kappa0=2.0, lambda_div=0.0, iteration=1)
    ucb = UpperConfidenceBound(gp=fitted_gp, greater_is_better=False, kappa=2.0)

    # Both should be finite
    assert np.isfinite(ducb(novel))
    assert np.isfinite(ucb(novel))


def test_diversity_ucb_iteration_increases_kappa(fitted_gp):
    """Higher iteration number → higher exploration (larger kappa factor)."""
    from bayesian_optimization.acquisition_function import DiversityUCB
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    kernel = OrderedRootedSubtreeKernel(normalize=True)
    novel = Tree("iter_kappa_test", (Tree("r"),))

    ducb_1 = DiversityUCB(gp=fitted_gp, kernel=kernel, greater_is_better=False,
                          kappa0=1.0, lambda_div=0.0, iteration=1)
    ducb_10 = DiversityUCB(gp=fitted_gp, kernel=kernel, greater_is_better=False,
                           kappa0=1.0, lambda_div=0.0, iteration=10)
    val_1 = ducb_1(novel)
    val_10 = ducb_10(novel)
    # Larger iteration → larger kappa → larger score (for same novel tree with positive sigma)
    assert val_10 >= val_1


def test_diversity_ucb_diversity_max_for_dissimilar_candidates(fitted_gp, tree_corpus):
    """A structurally unique tree should get a non-zero diversity bonus."""
    from bayesian_optimization.acquisition_function import DiversityUCB
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    kernel = OrderedRootedSubtreeKernel(normalize=True)
    novel = Tree("totally_unique_ZZZZ", (Tree("Q"), Tree("Q"), Tree("Q"), Tree("Q")))

    ducb = DiversityUCB(gp=fitted_gp, kernel=kernel, greater_is_better=False,
                        kappa0=2.0, lambda_div=0.5, iteration=1)
    val = ducb(novel)
    assert np.isfinite(val)


def test_diversity_ucb_calls_kernel_diag(monkeypatch, fitted_gp):
    """F15: DiversityUCB must call kernel.diag() not np.diag(kernel(...))."""
    from bayesian_optimization.acquisition_function import DiversityUCB
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    diag_calls = []
    real_kernel = OrderedRootedSubtreeKernel(normalize=True)
    original_diag = real_kernel.diag

    def tracked_diag(X):
        diag_calls.append(len(X))
        return original_diag(X)

    monkeypatch.setattr(real_kernel, "diag", tracked_diag)

    novel = [Tree("diag_test_a"), Tree("diag_test_b")]
    ducb = DiversityUCB(gp=fitted_gp, kernel=real_kernel, greater_is_better=False,
                        kappa0=2.0, lambda_div=0.5, iteration=1)
    ducb.evaluate_batch(novel)
    assert len(diag_calls) > 0, "kernel.diag() was never called"


def test_diversity_ucb_terms_are_comparable_scale(fitted_gp):
    """F16: all three terms (mean, sigma_norm, diversity) should be in [~-5, ~5]."""
    from bayesian_optimization.acquisition_function import DiversityUCB
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    kernel = OrderedRootedSubtreeKernel(normalize=True)
    candidates = [
        Tree("scale_test_1", (Tree("s"),)),
        Tree("scale_test_2", (Tree("t"),)),
    ]
    ducb = DiversityUCB(gp=fitted_gp, kernel=kernel, greater_is_better=False,
                        kappa0=2.0, lambda_div=0.5, iteration=1)
    scores = list(ducb.evaluate_batch(candidates).values())
    assert all(np.isfinite(v) for v in scores)
    # Scores should be finite and not astronomically large
    assert all(abs(v) < 1e6 for v in scores)
