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


def test_a_sampler_that_never_ends_is_bounded_per_term():
    """A depth-bounded sampler's stream ends only when a draw yields nothing.

    That may never happen, so a term is worth a fixed number of draws and no more.
    """

    class _Endless:
        def sample(self, _query):
            while True:
                yield _tree("always the same")

        def __repr__(self):
            return "_Endless"

    with pytest.raises(RuntimeError, match="appears too small"):
        distinct_prefix(_Endless(), object(), 2, max_draws_per_term=25)


def test_a_design_larger_than_the_draws_one_term_is_worth_is_drawable():
    """A design used to be refused for its size alone, in the name of the search space.

    The budget bounded the draws a whole design might take, and a design of ``count`` terms
    takes at least ``count`` draws, so every design past the budget was refused whatever the
    space held. The shipped budget is 100 and the design here is 101, on a stream that repeats
    every term once and can still serve it.
    """
    script = [term for i in range(101) for term in (_tree(f"t{i}"), _tree(f"t{i}"))]
    drawn, rejected = distinct_prefix(_ScriptedSampler(script), object(), 101)
    assert [t.root for t in drawn] == [f"t{i}" for i in range(101)]
    assert rejected == 100


def test_the_budget_starts_again_with_every_term_the_design_keeps():
    """Two terms are worth twice the budget, and the count restarts where one is kept.

    Both sides of the threshold are pinned here. Three repeats in a row are affordable under a
    budget of four and stay affordable when they occur again for the next term, while four in a
    row are one too many. A budget counted over the whole design would stop at the fourth
    repeat of the first case, with the design one term short.
    """

    def _script(repeats):
        first = [_tree("a")] * (repeats + 1)
        second = [_tree("b")] * (repeats + 1)
        return [*first, *second, _tree("c")]

    drawn, rejected = distinct_prefix(
        _ScriptedSampler(_script(3)), object(), 3, max_draws_per_term=4
    )
    assert [t.root for t in drawn] == ["a", "b", "c"]
    assert rejected == 6

    with pytest.raises(RuntimeError, match="4 times in a row"):
        distinct_prefix(_ScriptedSampler(_script(4)), object(), 3, max_draws_per_term=4)


def test_an_empty_design_is_answered_before_a_stream_is_opened():
    """A design of zero is complete before the first draw, and the loop cannot see that.

    The loop stops where a term has just been kept, so a design of zero would run to the end of
    the stream and, under a sampler whose stream never ends, past it.
    """
    sampler = _ScriptedSampler([_tree("a")])
    assert distinct_prefix(sampler, object(), 0) == ([], 0)
    assert sampler.streams == 0


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


# --- when the two novelty tests disagree ------------------------------------------------------
class _OneShot:
    """Offers one term, so a test can hand the fallback exactly what it must not accept."""

    def __init__(self, term):
        self.term = term

    def sample(self, _query):
        yield self.term

    def __repr__(self):
        return "_OneShot"


def test_a_label_that_hashes_against_its_own_equality_is_reported():
    """The fallback rejects a repeat with a set, and a set can only see what the labels hash by.

    Here two labels compare equal and hash apart, so the set does not recognize the term it is
    handed and this scan does. The design would gain a repeated training point, which is the one
    thing the function exists to prevent, so it stops instead.
    """

    class _ByName:
        def __init__(self, name):
            self.name = name

        def __eq__(self, other):
            return isinstance(other, _ByName) and self.name == other.name

        def __hash__(self):
            return id(self)

    kept, drawn = Tree(_ByName("x")), Tree(_ByName("x"))
    assert drawn == kept
    assert drawn not in {kept}
    with pytest.raises(RuntimeError, match="already in the initial design"):
        _distinct_dataset([kept], _OneShot(drawn), object(), 2)


def test_a_label_that_compares_asymmetrically_is_reported():
    """The two tests ask the same question in opposite directions.

    A set asks whether the term it holds equals the candidate, and this scan asks whether the
    candidate equals the term it holds. A label that answers the two differently reaches the
    branch with a hash that is identical for both terms and therefore beyond suspicion.
    """

    class _Asymmetric:
        def __init__(self, tag, accepts):
            self.tag = tag
            self.accepts = frozenset(accepts)

        def __eq__(self, other):
            return isinstance(other, _Asymmetric) and other.tag in self.accepts

        def __hash__(self):
            return 0

    strict = _Asymmetric("strict", {"strict"})
    lenient = _Asymmetric("lenient", {"strict", "lenient"})
    assert hash(strict) == hash(lenient)
    kept, drawn = Tree(strict), Tree(lenient)
    assert drawn == kept
    assert drawn not in {kept}
    with pytest.raises(RuntimeError, match="already in the initial design"):
        _distinct_dataset([kept], _OneShot(drawn), object(), 2)


def test_a_label_changed_after_its_term_was_built_is_reported():
    """A term takes its hash in its constructor and never again, so a label may outrun it.

    The label keeps the usual contract at every moment: it compares by its value and hashes by the
    same value. What goes stale is the term's hash, taken before the value changed, and that is
    enough for the set to miss a term this scan finds.
    """

    class _Knob:
        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            return isinstance(other, _Knob) and self.value == other.value

        def __hash__(self):
            return hash(self.value)

    knob = _Knob(1)
    kept = Tree(knob)
    knob.value = 2
    twin = _Knob(2)
    assert knob == twin
    assert hash(knob) == hash(twin)
    drawn = Tree(twin)
    assert drawn == kept
    assert drawn not in {kept}
    with pytest.raises(RuntimeError, match="already in the initial design"):
        _distinct_dataset([kept], _OneShot(drawn), object(), 2)


# --- a design that cannot become a set --------------------------------------------------------
class _FixedDesign:
    """An initializer that answers with whatever a test hands it, protocol or not."""

    def __init__(self, design):
        self.design = design

    def initialize(self, _query, _size):
        return list(self.design)


class _CountedObjective:
    """A quality measure that records how often it was asked."""

    def __init__(self):
        self.calls = 0

    def __call__(self, term):
        self.calls += 1
        return float(len(str(term)))


def _loop_over(initializer):
    """A loop on the expression space that draws its design from the given initializer."""
    return BayesianOptimization(
        search_space=expression_space(),
        request=EXPR,
        sampler=DepthBoundedRandomSampler(4, random.Random(0)),
        initializer=initializer,
        seed=0,
    )


@pytest.mark.parametrize(
    "design",
    [[["u"], ["v"], ["w"]], [["u"], ["v"], ["u"]]],
    ids=["distinct", "with a repeat"],
)
def test_a_design_the_initializer_could_not_hash_is_refused_in_the_loops_own_words(design):
    """The dataset is a set of terms, so a design that cannot be hashed cannot become one.

    Which message said so used to depend on whether the design happened to hold a repeat. A
    design without one reached the loop's own check, and a design with one met the set the
    fallback draws its replacement against first, where the interpreter answers with the type
    alone. Both are the same broken initializer and both are named the same way.
    """
    with pytest.raises(TypeError, match="the design the initializer returned must be hashable"):
        _loop_over(_FixedDesign(design)).initialize(objective=_CountedObjective(), initial_size=3)


def test_a_design_that_cannot_be_hashed_costs_no_evaluation(bo_factory):
    """A design that cannot enter the dataset is refused before an evaluation is spent on it.

    On the search this framework is built for, one evaluation trains a network. The check used
    to run after the values were collected, so a design that could never enter the dataset was
    paid for in full first.
    """
    supplied = _CountedObjective()
    with pytest.raises(TypeError, match="All candidates in x0 must be hashable"):
        bo_factory().initialize(objective=supplied, x0=[["u"], ["v"]])
    assert supplied.calls == 0

    drawn = _CountedObjective()
    with pytest.raises(TypeError, match="must be hashable"):
        _loop_over(_FixedDesign([["u"], ["v"], ["w"]])).initialize(objective=drawn, initial_size=3)
    assert drawn.calls == 0


def test_the_set_the_fallback_draws_against_grows_with_every_replacement():
    """Each replacement is drawn against the design as it stands, the earlier replacements too.

    The fallback opens a fresh stream per call, so a sampler that offers its terms in one order
    hands back the first replacement again at the second call. A set that had not learned that
    replacement would take it a second time, and the scan behind the fallback would stop the run
    instead of completing the design.
    """
    sampler = _ScriptedSampler([_tree("first"), _tree("second")])
    chosen = [_tree("a"), _tree("a"), _tree("a")]
    kept, repeats = _distinct_dataset(chosen, sampler, object(), 3)
    assert repeats == 2
    assert [t.root for t in kept] == ["a", "first", "second"]
    assert sampler.streams == 2


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
