from __future__ import annotations

import pytest
import numpy as np
from cosy.core.tree import Tree


def test_initialize_with_explicit_x0_y0(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:4], y0=[1.0, 2.0, 0.5, 3.0])
    snap = bo.get_state_snapshot()
    assert snap["state"] == "INITIALIZED"
    assert len(snap["x_list"]) == 4
    assert np.allclose(snap["y_list"], [1.0, 2.0, 0.5, 3.0])


def test_initialize_with_obj_fun_evaluates_y0(bo_factory, tree_corpus):
    bo = bo_factory()
    obj_fun = lambda t: float(len(str(t.root)))
    bo.initialize(x0=tree_corpus[:3], obj_fun=obj_fun)
    snap = bo.get_state_snapshot()
    assert snap["state"] == "INITIALIZED"
    assert len(snap["y_list"]) == 3
    expected = [obj_fun(t) for t in tree_corpus[:3]]
    assert np.allclose(snap["y_list"], expected)


def test_initialize_length_mismatch_raises_value_error(bo_factory, tree_corpus):
    bo = bo_factory()
    with pytest.raises(ValueError):
        bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0])


def test_initialize_obj_fun_and_y0_both_none_raises_value_error(bo_factory, tree_corpus):
    bo = bo_factory()
    with pytest.raises(ValueError):
        bo.initialize(x0=tree_corpus[:3])


def test_initialize_sets_iteration_counter_to_zero(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    snap = bo.get_state_snapshot()
    assert snap["iteration"] == 0


def test_generate_unique_initial_samples_returns_tuple_and_unpacks_correctly():
    """Regression test for F2: result must unpack as (list, dict)."""
    from bayesian_optimization.initial_sampling import _generate_unique_initial_samples

    t1 = Tree("A", (Tree("B"),))
    t2 = Tree("C", (Tree("D"),))
    t3 = Tree("E")

    class FakeInit:
        def __init__(self, pool):
            self._pool = list(pool)
            self._idx = 0

        def initialize_population(self, n):
            batch = self._pool[self._idx: self._idx + n]
            self._idx += n
            return batch * (n // max(len(batch), 1) + 1)

    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel
    init = FakeInit([t1, t2, t3])
    result = _generate_unique_initial_samples(init, 2)
    # Must be a 2-tuple
    assert isinstance(result, tuple) and len(result) == 2
    samples, stats = result
    assert isinstance(samples, list)
    assert isinstance(stats, dict)
    assert len(samples) <= 2


def test_generate_unique_initial_samples_unique():
    """All returned samples are distinct."""
    from bayesian_optimization.initial_sampling import _generate_unique_initial_samples

    trees = [
        Tree("A", (Tree("B"),)),
        Tree("C", (Tree("D"),)),
        Tree("E"),
        Tree("F", (Tree("G"), Tree("H"))),
    ]

    class FakeInit:
        def __init__(self, pool):
            self._pool = pool

        def initialize_population(self, n):
            return list(self._pool) * 3

    init = FakeInit(trees)
    samples, _ = _generate_unique_initial_samples(init, 3)
    assert len(samples) == len(set(id(t) for t in samples)) or len(set(samples)) == len(samples)


def test_generate_unique_initial_samples_warns_when_insufficient():
    """A RuntimeWarning is raised when fewer unique samples than requested are available."""
    from bayesian_optimization.initial_sampling import _generate_unique_initial_samples

    t = Tree("only_one")

    class FakeInit:
        def initialize_population(self, n):
            return [t] * n

    init = FakeInit()
    with pytest.warns(RuntimeWarning):
        samples, stats = _generate_unique_initial_samples(init, 5)
    assert len(samples) < 5
