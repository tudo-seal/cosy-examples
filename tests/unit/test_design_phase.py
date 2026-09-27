"""The initial design as the loop's first phase: drawn at once, handed out and taken back one term at a time.

``initialize()`` without an objective draws the design and evaluates nothing.  ``suggest()`` then hands
the design out in order, and ``observe()`` takes each value back, so a caller can write every
evaluation as it is measured and can take over values that already exist, without a second arm
drawing from the same stream.  After the last design value the passes begin exactly as they do after
a design handed over whole.

What the phase must not do is change the algorithm: the terms it hands out are the terms the closed
path evaluates, and the pass after it is the pass that would have followed ``initialize(x0, y0)``.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
from cosy.search import DepthBoundedRandomSampler, SizeUniformSampler

from bayesian_optimization.bo import BayesianOptimization
from bayesian_optimization.initial_sampling import distinct_prefix
from tests.spaces import EXPR, LIST, expression_space, list_space


def _loop(seed: int) -> BayesianOptimization:
    """A loop over the lists over ``{0, 1, 2}``, drawing size-uniformly under the bound 6."""
    return BayesianOptimization(
        search_space=list_space(),
        request=LIST,
        sampler=SizeUniformSampler(6, random.Random(seed)),
        seed=seed,
    )


class _RefusingOptimizer:
    """An evolutionary algorithm that must not be asked: the design phase maximizes nothing."""

    def evolutionary_best(self, *args, **kwargs):
        raise AssertionError("the design phase maximized an acquisition")


def _hand_out(bo, values):
    """Run the design phase: one suggestion per value, each taken back at once."""
    handed = []
    for value in values:
        suggestion = bo.suggest()
        handed.append(suggestion)
        bo.observe(suggestion.candidate, value)
    return handed


def test_an_initial_design_without_an_objective_is_drawn_and_not_evaluated():
    bo = _loop(seed=0)

    bo.initialize(initial_size=4)

    snapshot = bo.get_state_snapshot()
    assert snapshot["state"] == "DESIGN"
    assert len(bo.design) == 4
    assert len(set(bo.design)) == 4, "the design of a loop is a set of terms"
    assert snapshot["x_list"] == []
    assert snapshot["y_list"] == []


def test_the_drawn_design_is_the_design_the_closed_path_evaluates():
    """The phase moves the evaluation out of ``initialize()``; it must not move the draw."""
    phased = _loop(seed=3)
    phased.initialize(initial_size=5)

    closed = _loop(seed=3)
    closed.initialize(objective=lambda term: float(len(str(term))), initial_size=5)

    assert list(phased.design) == closed.get_state_snapshot()["x_list"]
    # Not equal because every draw is equal: another seed draws another design.
    other = _loop(seed=4)
    other.initialize(initial_size=5)
    assert list(other.design) != list(phased.design)


def test_the_drawn_design_is_the_prefix_the_paired_arm_draws():
    """One stream: the loop's own design is the head of what a paired random arm would draw."""
    bo = _loop(seed=5)
    bo.initialize(initial_size=4)

    reference = _loop(seed=5)
    prefix, repeats = distinct_prefix(reference.sampler, reference.query, 7)

    assert repeats == 0
    assert len(set(prefix)) == 7
    assert list(bo.design) == prefix[:4]


def test_the_design_phase_hands_the_terms_out_in_order_and_maximizes_nothing(
    bo_factory, tree_corpus
):
    bo = bo_factory(optimizer=_RefusingOptimizer())
    design = tree_corpus[:3]
    bo.initialize(design=design)

    handed = _hand_out(bo, [0.1, 0.2, 0.3])

    assert [suggestion.candidate for suggestion in handed] == design
    for index, suggestion in enumerate(handed):
        assert suggestion.acquisition_value is None
        assert suggestion.diagnostics is not None
        assert suggestion.diagnostics["phase"] == "design"
        assert suggestion.diagnostics["design_index"] == index
    assert bo.surrogate is None, "the design phase fitted a Gaussian process"
    snapshot = bo.get_state_snapshot()
    assert snapshot["state"] == "INITIALIZED"
    assert snapshot["iteration"] == 0
    assert snapshot["x_list"] == design
    assert snapshot["y_list"] == [0.1, 0.2, 0.3]
    assert bo.trace == [], "a design term is not a pass and gets no trace row"


def test_after_the_design_the_pass_is_the_pass_of_a_design_handed_over_whole(
    bo_factory, tree_corpus
):
    values = [1.0, 2.0, 0.5]
    phased = bo_factory()
    phased.initialize(design=tree_corpus[:3])
    _hand_out(phased, values)

    whole = bo_factory()
    whole.initialize(x0=tree_corpus[:3], y0=values)

    after_phase = phased.suggest()
    after_whole = whole.suggest()
    assert after_phase.candidate == after_whole.candidate
    assert after_phase.acquisition_value == pytest.approx(after_whole.acquisition_value)
    assert after_phase.diagnostics is not None and after_whole.diagnostics is not None
    assert after_phase.diagnostics["phase"] == after_whole.diagnostics["phase"] == "main"
    assert after_phase.diagnostics["iteration"] == after_whole.diagnostics["iteration"] == 0


def test_the_design_phase_takes_back_only_the_term_it_handed_out(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(design=tree_corpus[:2])
    bo.suggest()

    with pytest.raises(ValueError, match="does not match"):
        bo.observe(tree_corpus[1], 1.0)


def test_a_second_design_suggestion_waits_for_the_first_value(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(design=tree_corpus[:2])
    bo.suggest()

    with pytest.raises(RuntimeError):
        bo.suggest()


def test_a_run_finalized_during_its_design_keeps_what_it_observed_and_names_the_rest(
    bo_factory, tree_corpus
):
    """An interrupted design loses nothing that was measured, and says what it never measured."""
    bo = bo_factory()
    design = tree_corpus[:4]
    bo.initialize(design=design)
    first = bo.suggest()
    bo.observe(first.candidate, 0.7)
    bo.suggest()  # outstanding when the run is cut

    result = bo.finalize()

    assert result["best_tree"] == design[0]
    assert result["best_y"] == 0.7
    assert result["iterations"] == 0
    assert result["dropped_suggestion"] == design[1]
    assert result["design_remaining"] == design[1:]


def test_a_completed_design_leaves_nothing_remaining(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(design=tree_corpus[:2])
    _hand_out(bo, [0.2, 0.4])

    assert bo.finalize()["design_remaining"] == []


def test_a_design_with_no_value_yet_has_no_result(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(design=tree_corpus[:2])

    with pytest.raises(RuntimeError, match="No observations"):
        bo.finalize()


def test_a_design_given_as_terms_refuses_a_repeat(bo_factory, tree_corpus):
    """A repeated design term is an evaluation of the budget spent on an answer already known."""
    bo = bo_factory()
    with pytest.raises(ValueError, match="repeat"):
        bo.initialize(design=[tree_corpus[0], tree_corpus[1], tree_corpus[0]])


@pytest.mark.parametrize("extra", ["y0", "x0", "objective"])
def test_a_design_given_as_terms_carries_no_values_and_no_second_design(
    bo_factory, tree_corpus, extra
):
    """Values come back through ``observe()``; a second source of terms or values is a mix-up."""
    kwargs = {
        "y0": {"y0": [1.0, 2.0]},
        "x0": {"x0": tree_corpus[:2]},
        "objective": {"objective": lambda term: 1.0},
    }[extra]
    bo = bo_factory()
    with pytest.raises(ValueError, match="design"):
        bo.initialize(design=tree_corpus[:2], **kwargs)
    assert bo.get_state_snapshot()["state"] == "UNINITIALIZED"


def test_an_empty_design_is_a_complete_one(bo_factory):
    bo = bo_factory()
    bo.initialize(design=[])

    assert bo.get_state_snapshot()["state"] == "INITIALIZED"


def test_the_snapshot_shows_the_design_and_reset_forgets_it(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(design=tree_corpus[:3])
    _hand_out(bo, [1.0])

    snapshot = bo.get_state_snapshot()
    assert snapshot["state"] == "DESIGN"
    assert snapshot["design"] == tree_corpus[:3]
    assert snapshot["design_remaining"] == 2

    bo.reset()
    assert bo.design == ()
    assert bo.get_state_snapshot()["design"] == []


def test_a_design_needs_a_space_to_be_drawn_from(bo_factory):
    """Without terms and without a space there is nothing to hand out."""
    bo = bo_factory()
    with pytest.raises(NotImplementedError, match="requires a real search_space"):
        bo.initialize(initial_size=3)


def test_the_pass_configuration_is_checked_before_the_design_is_paid(bo_factory):
    """An ask/tell caller asks this before its design, as ``optimize()`` does for itself."""
    with pytest.raises(ValueError):
        bo_factory(acquisition_function="NoSuchScore").check_configuration()
    with pytest.raises(RuntimeError, match="optimizer is required"):
        bo_factory(optimizer=None).check_configuration()
    bo_factory().check_configuration()  # a configuration a pass can use passes


class _ProposesAKnownTerm:
    """An evolutionary algorithm whose maximization returns one fixed term, set once it is known."""

    def __init__(self) -> None:
        self.term = None

    def evolutionary_best(self, query, acquisition_objective, fitness_function_mode="batch"):
        return self.term


def _loop_proposing(seed: int):
    """A loop on a real space whose acquisition maximization always returns ``maximizer.term``."""
    # A stand-in for the evolutionary search, typed loosely as the conftest's DummyOptimizer is.
    maximizer: Any = _ProposesAKnownTerm()
    loop = BayesianOptimization(
        search_space=list_space(),
        request=LIST,
        sampler=SizeUniformSampler(6, random.Random(seed)),
        optimizer=maximizer,
        seed=seed,
    )
    return loop, maximizer


def test_a_design_term_proposed_again_is_replaced_as_after_a_design_handed_over_whole():
    """The duplicate fallback after the phase: the term is known, and a fresh one replaces it.

    Two things have to hold for that, whether the design was drawn or handed over: every design
    term is in the loop's duplicate index, and the loop has a sampler to draw the replacement from.
    On a loop without a search space neither can be seen, so this runs on a real one.
    """
    values = [0.1, 0.2, 0.3]
    drawn, drawn_maximizer = _loop_proposing(seed=7)
    drawn.initialize(initial_size=3)
    design = list(drawn.design)
    _hand_out(drawn, values)

    handed, handed_maximizer = _loop_proposing(seed=7)
    handed.initialize(design=design)
    _hand_out(handed, values)

    whole, whole_maximizer = _loop_proposing(seed=7)
    whole.initialize(x0=design, y0=values)

    for maximizer in (drawn_maximizer, handed_maximizer, whole_maximizer):
        maximizer.term = design[0]
    after_drawn, after_handed, after_whole = drawn.suggest(), handed.suggest(), whole.suggest()

    for suggestion in (after_drawn, after_handed, after_whole):
        assert suggestion.diagnostics is not None
        assert suggestion.diagnostics["fallback_used"] is True
        assert suggestion.candidate not in design
    # Fresh samplers of one seed draw one replacement; the drawn loop's sampler drew the design.
    assert after_handed.candidate == after_whole.candidate


def test_a_design_term_handed_out_still_awaits_its_value_in_the_snapshot(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(design=tree_corpus[:3])
    bo.suggest()  # handed out, not yet observed

    assert bo.get_state_snapshot()["design_remaining"] == 3


def test_a_drawn_design_counts_its_repeats_as_the_closed_path_does():
    """The count a run reports about its design does not depend on which path evaluated it."""

    def loop():
        return BayesianOptimization(
            search_space=expression_space(),
            request=EXPR,
            sampler=DepthBoundedRandomSampler(4, random.Random(0)),
            seed=0,
        )

    closed = loop()
    closed.initialize(objective=lambda term: float(len(str(term))), initial_size=8)
    phased = loop()
    phased.initialize(initial_size=8)

    assert closed.initial_repeats_rejected > 0, "this sampler has to repeat for the count to mean anything"
    assert phased.initial_repeats_rejected == closed.initial_repeats_rejected
    assert list(phased.design) == closed.get_state_snapshot()["x_list"]
