from __future__ import annotations

import numpy as np
import pytest
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
    bo.initialize(x0=tree_corpus[:3], objective=obj_fun)
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


class _StubSampler:
    """Sampler whose stream is a fixed list of trees."""

    def __init__(self, stream):
        self._stream = list(stream)
        self.streams_opened = 0

    def sample(self, query):
        self.streams_opened += 1
        return iter(self._stream)

    def at_least(self, query, count):
        return len(self._stream) >= count


def test_fallback_returns_the_first_unseen_inhabitant():
    from bayesian_optimization.initial_sampling import _sample_fallback_tree

    seen = [Tree("A"), Tree("B")]
    fresh = Tree("C", (Tree("c"),))
    sampler = _StubSampler([*seen, fresh, Tree("D")])

    assert _sample_fallback_tree(sampler, None, set(seen)) == fresh


def test_fallback_draws_one_stream_not_one_per_attempt():
    """Sampled initialization draws from a single stream, one requested element after the other.

    Re-posing the query per draw is what makes a draw expensive on a realistic space: measured
    at 20 minutes against 41 seconds for a prefix of one stream.
    """
    from bayesian_optimization.initial_sampling import _sample_fallback_tree

    seen = {Tree("A"), Tree("B"), Tree("C")}
    sampler = _StubSampler([Tree("A"), Tree("B"), Tree("C"), Tree("fresh")])

    _sample_fallback_tree(sampler, None, seen)

    assert sampler.streams_opened == 1


def test_fallback_raises_when_the_stream_runs_out():
    """An exhausted space is an error, not a substitute value."""
    from bayesian_optimization.initial_sampling import _sample_fallback_tree

    seen = {Tree("A"), Tree("B")}
    sampler = _StubSampler([Tree("A"), Tree("B")])

    with pytest.raises(RuntimeError, match="exhausted"):
        _sample_fallback_tree(sampler, None, seen)


def test_fallback_stops_a_sampler_that_draws_with_replacement():
    """Without the cap, a with-replacement sampler on an exhausted space would never return."""
    from bayesian_optimization.initial_sampling import _sample_fallback_tree

    known = Tree("only_one")

    class _Repeating:
        def sample(self, query):
            while True:
                yield known

    with pytest.raises(RuntimeError, match="already been evaluated"):
        _sample_fallback_tree(_Repeating(), None, {known}, max_draws=5)
