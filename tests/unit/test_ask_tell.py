from __future__ import annotations

import inspect

import numpy as np
import pytest
from cosy.core.tree import Tree


def test_initialize_suggest_observe_round_trip(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    assert s.candidate is not None
    bo.observe(s.candidate, 0.3)
    snap = bo.get_state_snapshot()
    assert snap["state"] == "OBSERVED"
    assert snap["x_list"][-1] == s.candidate
    assert snap["y_list"][-1] == pytest.approx(0.3)


def test_multiple_iterations_accumulate_data(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    for i, y_val in enumerate([0.4, 0.35, 0.3]):
        s = bo.suggest()
        bo.observe(s.candidate, y_val)
    snap = bo.get_state_snapshot()
    assert len(snap["x_list"]) == 6  # 3 initial + 3 observed
    assert len(snap["y_list"]) == 6


def test_y_list_contains_raw_values_after_each_observe(bo_factory, tree_corpus):
    bo = bo_factory()
    raw_values = [1.0, 2.0, 0.5]
    bo.initialize(x0=tree_corpus[:3], y0=raw_values)
    observed_y = 9.99
    s = bo.suggest()
    bo.observe(s.candidate, observed_y)
    snap = bo.get_state_snapshot()
    assert snap["y_list"][-1] == pytest.approx(observed_y)
    # Initial raw values preserved
    assert np.allclose(snap["y_list"][:3], raw_values)


def test_iteration_counter_increments_on_observe(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    assert bo.get_state_snapshot()["iteration"] == 0
    s = bo.suggest()
    bo.observe(s.candidate, 0.3)
    assert bo.get_state_snapshot()["iteration"] == 1
    s2 = bo.suggest()
    bo.observe(s2.candidate, 0.2)
    assert bo.get_state_snapshot()["iteration"] == 2


def test_suggestion_diagnostics_contain_all_required_fields(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    diag = s.diagnostics
    assert diag is not None
    required = {"timestamp", "incumbent_transformed", "incumbent_raw", "y_transform",
                "iteration", "fallback_used", "fallback_attempts"}
    assert required.issubset(diag.keys())


def test_duplicate_candidate_triggers_fallback(monkeypatch, bo_factory, tree_corpus):
    """If optimizer returns a known candidate, fallback sampling is triggered."""
    from bayesian_optimization import bo as bo_mod

    known = tree_corpus[3]  # a tree already in x0 would be a duplicate

    # Patch _sample_fallback_tree to return a fresh tree
    fresh = Tree("fallback_fresh")
    call_count = {"n": 0}

    def fake_fallback(initializer, seen):
        call_count["n"] += 1
        return fresh

    monkeypatch.setattr(bo_mod, "_sample_fallback_tree", fake_fallback)

    # Use a corpus tree as initial x0 AND have optimizer return it as candidate
    candidates = [tree_corpus[0]]  # this is in x0 → must trigger fallback
    bo = bo_factory(candidates=candidates)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    assert call_count["n"] >= 1
    assert s.diagnostics["fallback_used"] is True


def test_duplicate_fallback_respects_max_attempts(monkeypatch, bo_factory, tree_corpus):
    """If every fallback candidate is also a duplicate, RuntimeError after max_attempts."""
    from bayesian_optimization import bo as bo_mod

    always_known = tree_corpus[0]

    def fake_fallback(initializer, seen):
        return always_known  # always returns a known tree

    monkeypatch.setattr(bo_mod, "_sample_fallback_tree", fake_fallback)

    candidates = [tree_corpus[0]]  # in x0 → triggers fallback
    bo = bo_factory(candidates=candidates, max_duplicate_fallbacks=3)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    with pytest.raises(RuntimeError, match="Fallback"):
        bo.suggest()


def test_kernel_object_is_not_mutated_across_suggests(bo_factory, tree_corpus):
    """F7: suggest() must not assign back to self.kernel."""
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    kernel = OrderedRootedSubtreeKernel(normalize=True)
    original_id = id(kernel)
    original_params = kernel.get_params()

    bo = bo_factory(kernel=kernel)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    bo.observe(s.candidate, 0.3)
    bo.suggest()

    assert id(bo.kernel) == original_id
    assert bo.kernel.get_params() == original_params


def test_n_restarts_kernel_optimizer_is_not_decremented(bo_factory, tree_corpus):
    """F7: n_restarts_kernel_optimizer must stay constant across suggests."""
    bo = bo_factory(n_restarts_kernel_optimizer=5)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    bo.observe(s.candidate, 0.3)
    bo.suggest()
    assert bo.n_restarts_kernel_optimizer == 5


def test_bayesian_optimisation_wrapper_uses_user_x0_y0(bo_factory, tree_corpus):
    """F1: wrapper must pass x0/y0 to initialize(), not hardcode None."""
    bo = bo_factory()
    x0 = tree_corpus[:3]
    y0 = [1.0, 2.0, 0.5]
    result = bo.bayesian_optimisation(
        n_iters=1,
        obj_fun=lambda t: 0.3,
        x0=x0,
        y0=y0,
    )
    # The result should contain the initial x0 points
    all_x = list(result["x"])
    # x0 trees must appear in the result
    for t in x0:
        assert t in all_x, f"x0 tree {t} missing from result"


def test_search_space_none_no_special_path(bo_factory, tree_corpus):
    """F9: no special-case search_space=None branch; GP is always fitted."""
    bo = bo_factory()  # search_space=None
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    # acquisition_value is set only when GP was fitted
    assert s.acquisition_value is not None


def test_finalize_returns_complete_result_dict(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    bo.observe(s.candidate, 0.3)
    result = bo.finalize()
    assert "best_tree" in result
    assert "best_y" in result
    assert "x" in result
    assert "y" in result
    assert "gp_model" in result
    assert "y_transform" in result
    assert "iterations" in result
    assert result["iterations"] == 1


def test_best_returns_raw_scale_value(bo_factory, tree_corpus):
    bo = bo_factory()
    y0 = [5.0, 2.0, 8.0]
    bo.initialize(x0=tree_corpus[:3], y0=y0)
    best_tree, best_y = bo.best()
    assert best_y == pytest.approx(2.0)  # minimization default
