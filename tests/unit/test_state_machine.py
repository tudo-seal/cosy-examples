from __future__ import annotations

import logging

import pytest
from cosy.core.tree import Tree

LOGGER_NAME = "bayesian_optimization"


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
    # From the OBSERVED state, suggest() must be allowed again.
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


def test_reset_empties_the_shared_kernel_caches(bo_factory, tree_corpus):
    """The graph kernels cache on their module, and reset is what reaches that far.

    Their keys hold the terms, so after a run the terms of that run stay alive with nothing left
    to read them. reset drops every observation, and this drops what the observations left
    behind.
    """
    from bayesian_optimization.kernels import graph_kernel

    graph_kernel.clear_kernel_caches()
    bo = bo_factory(kernel=graph_kernel.WeisfeilerLehmanKernel(h=1))
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    suggestion = bo.suggest()
    bo.observe(suggestion.candidate, 0.4)
    assert len(graph_kernel._GRAPH_CACHE) > 0
    assert len(graph_kernel._MATRIX_CACHE) > 0

    bo.reset()

    assert len(graph_kernel._GRAPH_CACHE) == 0
    assert len(graph_kernel._MATRIX_CACHE) == 0


def test_reset_empties_caches_this_optimization_never_filled(bo_factory, tree_corpus):
    """Emptying the shared caches reaches every entry in the process, not only this run's.

    The caches carry the term and the translation an entry was built from and nothing about who
    asked for it, so there is no narrower set to drop. A run on the default kernel, which never
    converts a term to a graph, therefore empties what a graph kernel elsewhere put there, and
    that kernel pays for it in conversions.
    """
    from bayesian_optimization.kernels import graph_kernel
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    graph_kernel.clear_kernel_caches()
    graph_kernel.WeisfeilerLehmanKernel(h=1)._prepare_inputs(tree_corpus)
    assert len(graph_kernel._GRAPH_CACHE) == len(tree_corpus)
    bo = bo_factory()
    assert isinstance(bo.kernel, OrderedRootedSubtreeKernel)

    bo.reset()

    assert len(graph_kernel._GRAPH_CACHE) == 0


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
    assert "iterations" in result


def test_finalize_from_initialized(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    result = bo.finalize()
    assert "best_tree" in result
    assert result.get("state", True)


def test_finalize_transitions_to_finalized(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    bo.finalize()
    assert bo.get_state_snapshot()["state"] == "FINALIZED"


def test_initialize_with_unhashable_candidate_raises_type_error(bo_factory):
    bo = bo_factory()
    with pytest.raises(TypeError):
        bo.initialize(x0=[[1, 2, 3], [4, 5]], y0=[1.0, 2.0])


def _drop_warnings(caplog):
    """The warnings finalize() emits about a suggestion it gave up on, and no others."""
    return [
        record.getMessage()
        for record in caplog.records
        if record.levelno >= logging.WARNING and "dropped_suggestion" in record.getMessage()
    ]


def test_finalize_from_suggested_names_the_dropped_candidate(bo_factory, tree_corpus, caplog):
    """A run that finalizes with a suggestion still open gives that term up, and has to name it.

    An evaluation of the ask/tell layer may live outside this process, so a caller may hold a
    value for the outstanding term that only observe() can put into the dataset.  Finalizing
    stays allowed, because a run that was aborted still has to report what it collected, but the
    term it gives up on is named in the result and in the log rather than left to be noticed.
    """
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    first = bo.suggest()
    bo.observe(first.candidate, 0.3)
    outstanding = bo.suggest()

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = bo.finalize()

    assert result["dropped_suggestion"] == outstanding.candidate
    assert result["iterations"] == 1, "a suggestion without a value closes no pass"
    assert all(term != outstanding.candidate for term in result["x"]), (
        "the outstanding term has no value, so it cannot be in the dataset"
    )
    assert any(
        str(outstanding.candidate) in message for message in _drop_warnings(caplog)
    ), "finalize() gave the outstanding suggestion up without naming it in the log"

    snapshot = bo.get_state_snapshot()
    assert snapshot["state"] == "FINALIZED"
    assert snapshot["last_suggestion"] is None, "a finalized run still held a suggestion open"


def test_finalize_from_observed_drops_nothing(bo_factory, tree_corpus, caplog):
    """A run that observed its last suggestion before finalizing gives nothing up.

    Nothing is reported under ``dropped_suggestion`` and nothing is logged, so the report of the
    documented route stays quiet about a case that did not arise.
    """
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    bo.observe(s.candidate, 0.3)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = bo.finalize()

    assert result["dropped_suggestion"] is None
    assert _drop_warnings(caplog) == []
    assert bo.get_state_snapshot()["last_suggestion"] is None


def test_finalize_keeps_a_value_an_interrupted_observe_already_recorded(
    monkeypatch, bo_factory, tree_corpus, caplog
):
    """A term whose value the dataset already holds is not a dropped suggestion.

    observe() writes the term and its value before it moves the state, and an interrupt in
    between, which is what a stop signal during a run looks like, leaves the state at SUGGESTED
    with the value already recorded.  Reporting that term as given up would contradict the same
    result, which answers with it as the optimum of the run.
    """
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    outstanding = bo.suggest()

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt("stopped between the append and the state change")

    monkeypatch.setattr(bo, "_trace_record", interrupted)
    with pytest.raises(KeyboardInterrupt):
        bo.observe(outstanding.candidate, 42.0)

    snapshot = bo.get_state_snapshot()
    assert snapshot["state"] == "SUGGESTED"
    assert snapshot["y_list"][-1] == pytest.approx(42.0)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = bo.finalize()

    assert result["dropped_suggestion"] is None
    assert _drop_warnings(caplog) == []
    assert result["best_tree"] == outstanding.candidate
    assert result["best_y"] == pytest.approx(42.0)


def test_finalize_reports_a_term_the_dataset_never_paired_with_a_value(
    monkeypatch, bo_factory, tree_corpus, caplog
):
    """A term listed without a value beside it is still a dropped suggestion.

    observe() records the term in the duplicate index before it appends the value, so an
    interrupt in between leaves the index claiming a pass that never got one.  The report
    follows the pairing of the two dataset lists and not that index, which would otherwise let
    a half written record pass for a completed pass.
    """

    class _RefusingList(list):
        def append(self, item):
            raise KeyboardInterrupt("stopped before the value was appended")

    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    outstanding = bo.suggest()

    monkeypatch.setattr(bo, "_y_list", _RefusingList(bo.get_state_snapshot()["y_list"]))
    with pytest.raises(KeyboardInterrupt):
        bo.observe(outstanding.candidate, 42.0)

    snapshot = bo.get_state_snapshot()
    assert snapshot["state"] == "SUGGESTED"
    assert snapshot["x_list"][-1] == outstanding.candidate
    assert len(snapshot["y_list"]) == len(snapshot["x_list"]) - 1

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        result = bo.finalize()

    assert result["dropped_suggestion"] == outstanding.candidate
    assert result["best_tree"] != outstanding.candidate, (
        "a term without a value cannot be the optimum the same result reports"
    )
    assert any(
        str(outstanding.candidate) in message for message in _drop_warnings(caplog)
    ), "the term the dataset never valued was given up without a word"
