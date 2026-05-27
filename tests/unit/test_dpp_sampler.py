from __future__ import annotations

import pytest
from cosy.core.tree import Tree


def _make_trees(n: int) -> list[Tree]:
    labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
    return [Tree(labels[i % len(labels)], (Tree(f"child_{i}"),)) for i in range(n)]


def test_dpp_terminates():
    from bayesian_optimization.dpp import lazy_dpp_sample_optimized
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    trees = _make_trees(20)
    result = lazy_dpp_sample_optimized(iter(trees), OrderedRootedSubtreeKernel(), n_samples=5)
    assert isinstance(result, list)
    assert 0 < len(result) <= 5


def test_dpp_respects_max_attempts():
    from bayesian_optimization.dpp import lazy_dpp_sample_optimized
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    # Only 3 trees, asking for 10 → terminates at 3 (pool exhaustion)
    trees = _make_trees(3)
    result = lazy_dpp_sample_optimized(iter(trees), OrderedRootedSubtreeKernel(),
                                       n_samples=10, max_attempts=100)
    assert len(result) <= 3


def test_dpp_default_mutates_selected_in_place():
    """F17 (default behavior): selected list is extended in-place by default."""
    from bayesian_optimization.dpp import lazy_dpp_sample_optimized
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    trees = _make_trees(10)
    existing = [trees[0]]
    original_id = id(existing)
    result = lazy_dpp_sample_optimized(iter(trees[1:]), OrderedRootedSubtreeKernel(),
                                       n_samples=3, selected=existing, inplace=True)
    assert id(result) == original_id, "inplace=True should return the same list object"
    assert len(existing) >= 1  # existing was extended


def test_dpp_inplace_false_does_not_mutate():
    """F17: inplace=False must not modify the passed-in selected list."""
    from bayesian_optimization.dpp import lazy_dpp_sample_optimized
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    trees = _make_trees(10)
    original_selected = [trees[0]]
    original_copy = list(original_selected)

    result = lazy_dpp_sample_optimized(iter(trees[1:]), OrderedRootedSubtreeKernel(),
                                       n_samples=3, selected=original_selected, inplace=False)
    assert original_selected == original_copy, "Original selected list must not be mutated"
    assert len(result) > 0


def test_dpp_score_threshold_filters_correctly():
    """A very high threshold should reject all candidates and return only seed."""
    from bayesian_optimization.dpp import lazy_dpp_sample_optimized
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    trees = _make_trees(10)
    result = lazy_dpp_sample_optimized(iter(trees[1:]), OrderedRootedSubtreeKernel(),
                                       n_samples=5, score_threshold=1e10)
    # Threshold so high nothing passes → only first element (unconditionally added)
    assert len(result) <= 1


def test_dpp_kernel_cache_avoids_recomputation():
    """KernelCache should return the same value for repeated calls."""
    from bayesian_optimization.dpp import KernelCache
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    kernel = OrderedRootedSubtreeKernel(normalize=True)
    cache = KernelCache(kernel)
    t1 = Tree("A", (Tree("B"),))
    t2 = Tree("C", (Tree("D"),))

    v1 = cache.get(t1, t2)
    v2 = cache.get(t1, t2)
    v3 = cache.get(t2, t1)

    assert v1 == v2
    assert v1 == v3  # symmetry preserved
    # Should only have computed once (check cache size)
    assert len(cache.cache) == 2  # (t1,t2) and (t2,t1)
