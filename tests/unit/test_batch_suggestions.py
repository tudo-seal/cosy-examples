"""Batch suggestions: several suggestions outstanding at once, answered in any order.

``max_outstanding=k`` lets a caller hold up to k suggestions before it observes one, so that k
evaluations can run side by side.  The design is handed out k terms at a time, and the passes begin
only once every design term has its value.  A pass made while others are pending conditions its
surrogate on the real pairs and, for each pending pass, on the value the posterior expected at that
pick (the kriging believer): the pending terms are known points, and the loop never proposes one
twice.  Those assumed values reach that pass's surrogate and what it answers, never the dataset, a
trace row's observations or the answer.  At k = 1 the
loop is the one it always was.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
from cosy.search import SizeUniformSampler

from bayesian_optimization import BayesianOptimization, RandomSearch
from bayesian_optimization.acquisition_function import ExpectedImprovement
from bayesian_optimization.diagnostics import read_trace
from tests.spaces import LIST, list_space


def _recording(built: list[Any]) -> Any:
    """An acquisition factory that keeps what each pass hands it."""

    def factory(gp: Any, *, incumbent: float, known_points: set[Any]) -> Any:
        built.append({"gp": gp, "known_points": set(known_points)})
        return ExpectedImprovement(gp, incumbent=incumbent, known_points=known_points)

    return factory


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True])
def test_a_capacity_is_a_positive_count(bo_factory, capacity):
    with pytest.raises(ValueError, match="a positive count"):
        bo_factory(max_outstanding=capacity)
    with pytest.raises(ValueError, match="a positive count"):
        RandomSearch(None, max_outstanding=capacity)


def test_the_design_goes_out_k_terms_at_a_time_in_order(bo_factory, tree_corpus):
    bo = bo_factory(max_outstanding=2)
    bo.initialize(design=tree_corpus[:4])

    first, second = bo.suggest().candidate, bo.suggest().candidate
    assert [first, second] == tree_corpus[:2]
    with pytest.raises(RuntimeError, match="not allowed"):
        bo.suggest()
    assert bo.get_state_snapshot()["outstanding"] == tree_corpus[:2]

    bo.observe(tree_corpus[1], 2.0)
    assert bo.suggest().candidate == tree_corpus[2]
    bo.observe(tree_corpus[0], 1.0)
    assert bo.get_state_snapshot()["x_list"] == [tree_corpus[1], tree_corpus[0]], "observation order"


def test_no_pass_begins_while_a_design_value_is_pending(bo_factory, tree_corpus):
    bo = bo_factory(max_outstanding=3)
    bo.initialize(design=tree_corpus[:2])
    bo.suggest(), bo.suggest()

    with pytest.raises(RuntimeError, match="not allowed"):
        bo.suggest()
    bo.observe(tree_corpus[0], 1.0)
    with pytest.raises(RuntimeError, match="not allowed"):
        bo.suggest()
    bo.observe(tree_corpus[1], 2.0)
    assert bo.suggest().diagnostics["phase"] == "main"


def test_a_pending_pass_is_conditioned_on_as_believed_and_known(bo_factory, tree_corpus):
    """The second pass of a batch: its surrogate holds the real pairs and the first pass's pick at
    the mean the first pass's posterior expected there; its known points hold that pick."""
    built: list[Any] = []
    # the targets unnormalized, so that the regressor's y_train_ holds the values as conditioned on
    bo = bo_factory(max_outstanding=2, acquisition_function=_recording(built), gp_normalize_y=False)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])

    first = bo.suggest()
    second = bo.suggest()

    assert (first.diagnostics["iteration"], second.diagnostics["iteration"]) == (0, 1)
    assert first.candidate != second.candidate
    believed = built[1]["gp"]
    assert list(believed.X_train_) == [*tree_corpus[:3], first.candidate]
    assert list(believed.y_train_) == pytest.approx([1.0, 2.0, 0.5, first.diagnostics["mean_at_pick"]])
    assert first.candidate in built[1]["known_points"]
    assert len(bo.surrogate_over_dataset().X_train_) == 3, "the assumed value is no data"


def test_a_batch_is_answered_in_any_order_and_the_trace_reads_the_observations(
    bo_factory, tree_corpus
):
    bo = bo_factory(max_outstanding=2)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    first, second = bo.suggest(), bo.suggest()

    bo.observe(second.candidate, 5.0)
    bo.observe(first.candidate, 0.1)

    assert bo.get_state_snapshot()["x_list"][3:] == [second.candidate, first.candidate]
    assert [(row.iteration, row.incumbent, row.best) for row in bo.trace] == [
        (1, 2.0, 5.0), (0, 5.0, 5.0),
    ]
    assert read_trace(bo.trace).improvements == 1
    assert bo.get_state_snapshot()["iteration"] == 2
    assert bo.best() == (second.candidate, 5.0)


def test_every_suggestion_no_value_reached_is_named_at_the_end(bo_factory, tree_corpus):
    bo = bo_factory(max_outstanding=3)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    first, second, third = bo.suggest(), bo.suggest(), bo.suggest()
    bo.observe(second.candidate, 1.5)

    result = bo.finalize()
    assert result["dropped_suggestions"] == [first.candidate, third.candidate]
    assert result["dropped_suggestion"] == third.candidate


def test_at_capacity_one_the_list_is_there_and_says_what_the_one_says(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    assert bo.finalize()["dropped_suggestions"] == []

    open_one = bo_factory()
    open_one.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    candidate = open_one.suggest().candidate
    result = open_one.finalize()
    assert (result["dropped_suggestions"], result["dropped_suggestion"]) == ([candidate], candidate)


def test_a_random_search_never_proposes_a_pending_term():
    search = RandomSearch(list_space(), LIST, sampler=SizeUniformSampler(6, random.Random(2)),
                          seed=2, max_outstanding=3)
    search.initialize(initial_size=2)
    for _ in range(2):
        search.observe(search.suggest().candidate, 0.5)

    pending = [search.suggest().candidate for _ in range(3)]
    assert len(set(pending)) == 3
    assert not set(pending) & set(search.get_state_snapshot()["x_list"])


def test_a_bayesian_batch_never_proposes_a_pending_term(monkeypatch):
    """The maximization is made to answer the first pending pick; the second pass replaces it."""
    from bayesian_optimization.acquisition_optimizer import SampleMaximizer

    original = SampleMaximizer.maximize_with_population

    def the_pending_pick(self: Any, acquisition: Any, query: Any, **kwargs: Any) -> Any:
        _pick, population, generations = original(self, acquisition, query, **kwargs)
        return (picks[0] if picks else _pick), population, generations

    picks: list[Any] = []
    monkeypatch.setattr(SampleMaximizer, "maximize_with_population", the_pending_pick)
    loop = BayesianOptimization(
        list_space(), LIST, sampler=SizeUniformSampler(6, random.Random(4)), seed=4,
        maximizer=SampleMaximizer(SizeUniformSampler(6, random.Random(104)), 8),
        max_outstanding=2,
    )
    loop.initialize(initial_size=3)
    for _ in range(3):
        loop.observe(loop.suggest().candidate, 0.5)

    first = loop.suggest(record_population=True)
    picks.append(first.candidate)
    second = loop.suggest(record_population=True)
    assert second.diagnostics["fallback_used"] is True
    assert second.candidate != first.candidate
    assert second.candidate not in loop.get_state_snapshot()["x_list"]


def test_a_random_search_drawing_with_replacement_never_proposes_a_pending_term():
    """A stream that draws with replacement delivers a term again; while it is pending it is
    skipped, as a held one is."""
    from cosy.search import DepthBoundedRandomSampler

    # seed 7 delivers two of its four pending terms again (measured)
    search = RandomSearch(list_space(), LIST, sampler=DepthBoundedRandomSampler(3, random.Random(7)),
                          seed=7, max_outstanding=4)
    search.initialize(initial_size=1)
    search.observe(search.suggest().candidate, 0.5)

    pending = [search.suggest().candidate for _ in range(4)]
    assert len(set(pending)) == 4
    assert search.terms_skipped > 0, "the stream did deliver a term again"


def test_at_capacity_above_one_a_refused_suggestion_says_why(bo_factory, tree_corpus):
    full = bo_factory(max_outstanding=2)
    full.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    full.suggest(), full.suggest()
    with pytest.raises(RuntimeError, match="max_outstanding=2"):
        full.suggest()

    designing = bo_factory(max_outstanding=3)
    designing.initialize(design=tree_corpus[:2])
    designing.suggest(), designing.suggest()
    with pytest.raises(RuntimeError, match="design"):
        designing.suggest()


def test_every_dropped_suggestion_s_warning_says_where_the_result_names_it(
    bo_factory, tree_corpus, caplog
):
    bo = bo_factory(max_outstanding=2)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    bo.suggest(), bo.suggest()
    with caplog.at_level("WARNING"):
        bo.finalize()
    warnings = [record.getMessage() for record in caplog.records if "finalized" in record.getMessage()]
    assert len(warnings) == 2
    assert all("dropped_suggestions" in warning for warning in warnings)


def test_a_pending_term_the_maximization_returns_is_said_to_be_pending(caplog, monkeypatch):
    from bayesian_optimization.acquisition_optimizer import SampleMaximizer

    original = SampleMaximizer.maximize_with_population
    picks: list[Any] = []

    def the_pending_pick(self: Any, acquisition: Any, query: Any, **kwargs: Any) -> Any:
        pick, population, generations = original(self, acquisition, query, **kwargs)
        return (picks[0] if picks else pick), population, generations

    monkeypatch.setattr(SampleMaximizer, "maximize_with_population", the_pending_pick)
    loop = BayesianOptimization(
        list_space(), LIST, sampler=SizeUniformSampler(6, random.Random(4)), seed=4,
        maximizer=SampleMaximizer(SizeUniformSampler(6, random.Random(104)), 8), max_outstanding=2,
    )
    loop.initialize(initial_size=3)
    for _ in range(3):
        loop.observe(loop.suggest().candidate, 0.5)
    picks.append(loop.suggest().candidate)
    with caplog.at_level("WARNING"):
        loop.suggest()
    assert any("pending" in record.getMessage() for record in caplog.records)
    assert not any("already evaluated" in record.getMessage() for record in caplog.records)
