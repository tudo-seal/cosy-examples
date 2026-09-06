"""What the Ask/Tell entry points refuse, and in which words.

The state machine tests next door assert that a call out of order raises.  These assert the
sentence it raises with, because the sentence is what names the missing piece: a space to draw a
design from, an evolutionary algorithm to maximize the acquisition with, a candidate for that
maximization to answer, or a state the call can be made from at all.

Every path here is reached through the public Ask/Tell calls.  None of them reads or writes a
private attribute, so each one is a sequence a caller can actually produce.
"""
from __future__ import annotations

import pytest


def test_an_initial_design_without_terms_needs_a_space_to_draw_them_from(bo_factory):
    """``initialize()`` draws the design when no ``x0`` is given, and a draw needs a space.

    The fixture builds an optimization without a search space, which is the configuration of a
    caller who hands the pairs over ready-made.  Such a caller who then forgets ``x0`` is told
    which of the two ways of building a design is missing, rather than starting on an empty one.
    """
    bo = bo_factory()
    with pytest.raises(NotImplementedError, match="requires a real search_space"):
        bo.initialize()


def test_a_suggestion_without_an_evolutionary_algorithm_is_refused_by_name(
    bo_factory, tree_corpus
):
    """A suggestion is the maximization of an acquisition, and there is nothing to maximize with.

    The refusal names the parameter and the type it takes, because the optimization is
    constructible without one: an instance that only conditions a surrogate needs no search.
    """
    bo = bo_factory(optimizer=None)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    with pytest.raises(RuntimeError, match="An optimizer is required"):
        bo.suggest()


def test_a_maximization_that_answers_with_nothing_is_not_a_suggestion(bo_factory, tree_corpus):
    """An empty answer from the search is an error, not a candidate.

    The stand-in optimizer of the fixtures answers ``None`` once its list of candidates is used
    up.  Handing that on would put ``None`` into the dataset in the place of a term, and every
    later read of the dataset would carry it.
    """
    bo = bo_factory(candidates=[])
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    with pytest.raises(RuntimeError, match="did not return a candidate"):
        bo.suggest()


def test_finalizing_a_run_that_never_started_is_refused(bo_factory):
    """The answer of a run is a term of maximal observed value, and nothing was observed."""
    bo = bo_factory()
    with pytest.raises(RuntimeError, match="not allowed in state UNINITIALIZED"):
        bo.finalize()


def test_a_finalized_run_cannot_be_finalized_a_second_time(bo_factory, tree_corpus):
    """The second answer would differ from the first, and it is the first that describes the run.

    Finalizing clears the outstanding suggestion after naming it, so a second call would report
    nothing dropped where the first reported the term the run gave up on.
    """
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    suggestion = bo.suggest()

    result = bo.finalize()
    assert result["dropped_suggestion"] == suggestion.candidate

    with pytest.raises(RuntimeError, match="not allowed in state FINALIZED"):
        bo.finalize()
