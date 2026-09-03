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
reports is a check where the guarantee applies and the repair where it does not. They also pin
that the count can be read at all, in every state a run passes through, because it belongs in the
run record and a record is written when the run ends rather than where the design is drawn.
"""

from __future__ import annotations

import random

import pytest
from cosy.core.tree import Tree
from cosy.search import DepthBoundedRandomSampler

from bayesian_optimization.bo import BayesianOptimization, _distinct_dataset
from bayesian_optimization.initial_sampling import distinct_prefix
from tests.spaces import EXPR, expression_space


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


# --- the count the run reports ----------------------------------------------------------------
def _drawing_loop():
    """A loop over the expression space whose sampler repeats terms.

    Depth-bounded random sampling draws independently, so a design of 8 costs a few redraws here.
    Seeded, so the tests below can pin an exact number instead of a nonzero one.
    """
    return BayesianOptimization(
        search_space=expression_space(),
        request=EXPR,
        sampler=DepthBoundedRandomSampler(4, random.Random(0)),
        seed=0,
    )


def _term_length(term):
    """A stand-in objective: how long the term prints.

    Cheap and finite, which is all these tests ask of an objective. They read the count the loop
    keeps about its initial design and never the values, and a value shared by two different terms
    is admissible anyway: the dataset refuses one term with two values, not one value on two terms.
    """
    return float(len(str(term)))


def test_the_count_answers_from_construction_through_a_finished_run(bo_factory, tree_corpus):
    """Every state of a run can be asked how many repeats the initial design cost.

    The count is what a run record reports about its own initial design, and a record is written
    at the end of a run, not at the initializer. So the answer has to exist wherever the caller
    stands, and before initialize() there is one: nothing was drawn, so nothing was redrawn.
    """
    bo = bo_factory()
    assert bo.initial_repeats_rejected == 0

    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    assert bo.initial_repeats_rejected == 0

    suggestion = bo.suggest()
    assert bo.initial_repeats_rejected == 0

    bo.observe(suggestion.candidate, 0.3)
    assert bo.initial_repeats_rejected == 0

    bo.finalize()
    assert bo.initial_repeats_rejected == 0


def test_a_closed_run_over_a_supplied_design_reports_no_repeats(bo_factory, tree_corpus):
    """A design the caller hands over was not drawn here, so this loop redrew nothing in it.

    This is the state the run record is written from, and the one the closed loop leaves behind.
    """
    bo = bo_factory()
    bo.optimize(objective=_term_length, budget=2, x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    assert bo.initial_repeats_rejected == 0


def test_a_drawn_design_reports_the_repeats_it_cost():
    """The count is a measurement and not a placeholder: under a repeating sampler it is nonzero.

    Pinned on the exact number rather than on nonzero, so that a change in what the sampler draws
    is visible here rather than absorbed. The design is distinct all the same.
    """
    bo = _drawing_loop()
    bo.initialize(objective=_term_length, initial_size=8)
    x_list = bo.get_state_snapshot()["x_list"]
    assert bo.initial_repeats_rejected == 3
    assert len(set(x_list)) == 8


def test_a_reset_puts_the_count_back_to_zero():
    """A second run reports its own initial design and not the one before it."""
    bo = _drawing_loop()
    bo.initialize(objective=_term_length, initial_size=8)
    assert bo.initial_repeats_rejected == 3
    bo.reset()
    assert bo.initial_repeats_rejected == 0


def test_an_abandoned_design_does_not_leave_its_count_behind(tree_corpus):
    """An initialization that fails after the draw leaves no count for the next one to report.

    The objective is evaluated after the design is repaired and the state becomes INITIALIZED only
    at the end, so a failing evaluation leaves the loop uninitialized and a second initialize() is
    allowed. The design the caller then supplies is a different design, and reporting the redraws
    of the abandoned one against it would be a number about nothing.
    """
    bo = _drawing_loop()
    with pytest.raises(ValueError, match="cannot be observed"):
        bo.initialize(objective=lambda _term: float("nan"), initial_size=8)
    assert bo.get_state_snapshot()["state"] == "UNINITIALIZED"
    assert bo.initial_repeats_rejected == 0

    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    assert bo.initial_repeats_rejected == 0
