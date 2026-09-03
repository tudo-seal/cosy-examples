"""The loop's initial dataset is a set, and it may not inherit that from its default sampler.

A population is a finite multiset of individuals, so sampled initialization legitimately returns
repeats and the evolutionary components are fine with them. The Bayesian loop's dataset is not a
multiset: it is what the surrogate conditions on, and a repeated term is an evaluation of the
budget spent on an observation the previous one already carried.

The code used to state distinctness as a **consequence** of a sampling guarantee, that random
search under a size bound streams every inhabitant within the bound exactly once, so that any
prefix of that stream is free of repeats. It holds for the size-uniform stream and for no other.
The depth-bounded random sampler makes independent draws and so may repeat a term, and ``sampler``
is a constructor parameter, so the guarantee held only until someone used it. The USPS
configuration uses exactly that sampler, because its determinization is unaffordable.

These tests pin the repaired shape: the rejection runs whatever the sampler is, and the count it
reports is a check where the guarantee applies and the repair where it does not.
"""

from __future__ import annotations

import pytest
from cosy.core.tree import Tree

from bayesian_optimization.bo import _distinct_dataset
from bayesian_optimization.initial_sampling import distinct_prefix


class _ScriptedSampler:
    """Yields a fixed sequence, so a test can say exactly which repeats occur."""

    def __init__(self, script):
        self._script = list(script)
        self.streams = 0

    def sample(self, _query):
        self.streams += 1
        yield from self._script

    def __repr__(self):
        return "_ScriptedSampler"


def _tree(name):
    """A term whose structural identity is its name."""
    return Tree(name, (Tree("leaf"),))


# --- drawing a design from scratch ------------------------------------------------------------
def test_a_stream_without_repeats_rejects_nothing():
    """Size-uniform sampling streams each inhabitant once, so nothing here is rejected."""
    sampler = _ScriptedSampler([_tree("a"), _tree("b"), _tree("c")])
    drawn, rejected = distinct_prefix(sampler, object(), 3)
    assert rejected == 0
    assert [t.root for t in drawn] == ["a", "b", "c"]


def test_repeats_are_rejected_and_counted():
    """Repeats occur under depth-bounded random sampling, and the count goes into the run record."""
    sampler = _ScriptedSampler(
        [_tree("a"), _tree("a"), _tree("b"), _tree("a"), _tree("c")]
    )
    drawn, rejected = distinct_prefix(sampler, object(), 3)
    assert rejected == 2
    assert [t.root for t in drawn] == ["a", "b", "c"]


def test_the_comparison_is_structural_and_not_by_identity():
    """Two equal terms built separately are one term, and ``Tree.__eq__`` decides that."""
    sampler = _ScriptedSampler([_tree("a"), _tree("a"), _tree("b")])
    drawn, rejected = distinct_prefix(sampler, object(), 2)
    assert rejected == 1
    assert drawn[0] is not drawn[1]


def test_one_stream_is_opened_and_not_one_per_draw():
    """Each call re-poses the query, and on a real space that is what a draw costs."""
    sampler = _ScriptedSampler([_tree(name) for name in "abcde"])
    distinct_prefix(sampler, object(), 4)
    assert sampler.streams == 1


def test_a_short_design_is_an_error_and_not_a_short_design():
    """Topping it up with repeats is the thing this exists to prevent."""
    sampler = _ScriptedSampler([_tree("a"), _tree("a"), _tree("a")])
    with pytest.raises(RuntimeError, match="ended after 1 distinct"):
        distinct_prefix(sampler, object(), 3)


def test_a_sampler_that_never_ends_is_bounded_by_max_draws():
    """A depth-bounded sampler's stream ends only when a draw yields nothing.

    That may never happen, so the number of draws is capped.
    """

    class _Endless:
        def sample(self, _query):
            while True:
                yield _tree("always the same")

        def __repr__(self):
            return "_Endless"

    with pytest.raises(RuntimeError, match="appears too small"):
        distinct_prefix(_Endless(), object(), 2, max_draws=25)


def test_a_negative_design_size_is_refused():
    """A design holds a non-negative number of terms."""
    with pytest.raises(ValueError, match="non-negative"):
        distinct_prefix(_ScriptedSampler([]), object(), -1)


# --- repairing what an initializer returned ---------------------------------------------------
def test_the_initializers_own_terms_are_kept():
    """A kernel-diverse initializer picks its terms for their distance from each other.

    Redrawing the whole set discards that, so only the places a repeat occupied are filled again
    and an informed design stays the informed design rather than silently becoming the
    model-agnostic one.
    """
    chosen = [_tree("far"), _tree("near"), _tree("far")]
    sampler = _ScriptedSampler([_tree("far"), _tree("replacement")])
    kept, repeats = _distinct_dataset(chosen, sampler, object(), 3)
    assert repeats == 1
    assert [t.root for t in kept] == ["far", "near", "replacement"]


def test_a_design_without_repeats_is_returned_untouched():
    """The common case costs one scan and draws nothing."""
    chosen = [_tree("a"), _tree("b")]
    sampler = _ScriptedSampler([])
    kept, repeats = _distinct_dataset(chosen, sampler, object(), 2)
    assert repeats == 0
    assert kept == chosen
    assert sampler.streams == 0
