from __future__ import annotations

import pytest
from cosy.core.tree import Tree


def test_suggest_before_initialize_raises_runtime_error(bo_factory):
    bo = bo_factory()
    with pytest.raises(RuntimeError):
        bo.suggest()


def test_observe_before_suggest_raises_runtime_error(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    with pytest.raises(RuntimeError):
        bo.observe(tree_corpus[3], 1.0)


def test_double_suggest_raises_runtime_error(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    bo.suggest()
    with pytest.raises(RuntimeError):
        bo.suggest()


def test_double_initialize_raises_runtime_error(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    with pytest.raises(RuntimeError):
        bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])


def test_observed_to_suggested_is_allowed(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    bo.observe(s.candidate, 0.3)
    # OBSERVED → suggest() must be allowed
    s2 = bo.suggest()
    assert s2 is not None


def test_suggested_to_observed_with_matching_candidate(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    bo.observe(s.candidate, 0.5)
    snap = bo.get_state_snapshot()
    assert snap["state"] == "OBSERVED"


def test_observe_mismatch_raises_value_error(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    bo.suggest()
    with pytest.raises(ValueError):
        bo.observe(Tree("mismatch_candidate"), 0.5)


def test_reset_returns_to_uninitialized(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    bo.observe(s.candidate, 0.3)
    bo.reset()
    assert bo.get_state_snapshot()["state"] == "UNINITIALIZED"


def test_reset_clears_all_data(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    bo.reset()
    snap = bo.get_state_snapshot()
    assert snap["x_list"] == []
    assert snap["y_list"] == []


def test_finalize_from_observed(bo_factory, tree_corpus):
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


def test_finalize_from_initialized(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    result = bo.finalize()
    assert "best_tree" in result
    assert result["state"] if "state" in result else True


def test_finalize_transitions_to_finalized(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    bo.finalize()
    assert bo.get_state_snapshot()["state"] == "FINALIZED"


def test_initialize_with_unhashable_candidate_raises_type_error(bo_factory):
    bo = bo_factory()
    with pytest.raises(TypeError):
        bo.initialize(x0=[[1, 2, 3], [4, 5]], y0=[1.0, 2.0])
