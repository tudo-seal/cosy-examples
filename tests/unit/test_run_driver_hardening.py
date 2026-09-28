"""The run layer's promises under the cases a review found unguarded.

A minimizing objective enters the loop negated and comes back in its own sign.  A caller's reading
reads every resumed record; without one, a kept value must be the run's objective's reading of the
record, and a resumed run keeps the values for the next resume.
An evaluation is on disk before the loop refuses its value, and before the run stops on a missing
objective.  Everything that can be refused is refused before a file is opened, so a corrected retry
under the same name runs; ``run_paired`` refuses a later strategy's mistakes before the first
strategy runs.  And the smaller promises: a legacy pool resumed is rewritten under the current tag,
the recorder's exclusive mode checks both of its files, a schema's column names are unique.
"""

from __future__ import annotations

import builtins
import csv
import math
import random

import pytest
from cosy.core.tree import Tree
from cosy.search import SizeUniformSampler

from bayesian_optimization import BayesianOptimization, RandomSearch
from bayesian_optimization.initial_sampling import distinct_prefix
from bayesian_optimization.runs import (
    FORMAT,
    MetricSchema,
    Objective,
    StepBudgets,
    TermRecord,
    build_acquisition_optimizer,
    read_term_pool,
    resume_term_pool,
)
from bayesian_optimization.runs import records as records_module
from bayesian_optimization.runs.driver import run_paired, run_search
from bayesian_optimization.runs.records import EvaluationRecorder
from bayesian_optimization.runs.resume import load_design_records
from bayesian_optimization.runs.term_pool import TermPoolWriter
from tests.spaces import LIST, list_space

SCHEMA = MetricSchema.for_metrics(Objective("score"), ["score", "length"])
LOSS = MetricSchema.for_metrics(Objective("loss", greater_is_better=False), ["loss"])


def _metrics(term):
    text = term.interpret({})
    return {"score": 1.0 / (1 + len(text)), "length": len(text)}


def _loss(term):
    return {"loss": float(len(term.interpret({})))}


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _sampler(seed):
    return SizeUniformSampler(6, random.Random(seed))


def _random(seed):
    return RandomSearch(list_space(), LIST, sampler=_sampler(seed), seed=seed)


def _bo(seed):
    return BayesianOptimization(
        list_space(), LIST, sampler=_sampler(seed), seed=seed,
        optimizer=build_acquisition_optimizer(population_size=6, generations=2, depth_bound=4,
                                              seed=seed),
    )


def _quiet(line):
    """An echo that prints nothing."""


def _run(strategy, evaluate, path, schema=SCHEMA, **kwargs):
    return run_search(strategy, evaluate, schema=schema, csv_path=str(path), pretty_algebra=dict,
                      echo=_quiet, **kwargs)


def _head(seed, count):
    terms, _ = distinct_prefix(_sampler(seed), _random(seed).query, count)
    return terms


# --- the sign of a minimizing objective ----------------------------------------------------------

@pytest.mark.parametrize("strategy", [_random, _bo], ids=["random", "bo"])
def test_a_minimizing_objective_enters_negated_and_reports_in_its_own_sign(tmp_path, strategy):
    outcome = _run(strategy(8), _loss, tmp_path / "run.csv", schema=LOSS, n_design=3, n_passes=2)

    losses = [float(row["loss"]) for row in _rows(tmp_path / "run.csv")]
    assert list(outcome.result["y"]) == [-loss for loss in losses]
    assert [float(row["loop_value"]) for row in _rows(tmp_path / "run.csv")] == [-v for v in losses]
    _header, records = read_term_pool(tmp_path / "run_terms.pickle")
    assert [record.loop_value for record in records] == [-loss for loss in losses]
    assert outcome.summary["best_objective_value"] == min(losses)
    assert outcome.summary["best_loop_value"] == -min(losses)


# --- resume --------------------------------------------------------------------------------------

def test_a_caller_s_reading_reads_every_resumed_record_and_a_disagreeing_kept_value_is_refused(
        tmp_path):
    """A kept loop value is the value the loop of the run that wrote the record was handed.

    A run that names its own reading resumes under it, whatever the records kept.  Without one, a
    kept value that this run's objective does not read off the record's metrics was handed to a
    loop under another objective or reading, and it is refused rather than mixed in.
    """
    terms = _head(seed=3, count=3)
    records = [TermRecord("pre_sample", i, term, {"score": 0.1 * (i + 1)}, loop_value=0.5 + i / 10)
               for i, term in enumerate(terms)]

    read = _run(_random(3), _metrics, tmp_path / "read.csv", resume=records, n_passes=0,
                resumed_value=lambda metrics: metrics["score"])
    assert list(read.result["y"]) == pytest.approx([0.1, 0.2, 0.3])
    _header, rewritten = read_term_pool(tmp_path / "read_terms.pickle")
    assert [record.loop_value for record in rewritten] == pytest.approx([0.1, 0.2, 0.3]), (
        "the run's own records keep the value this run was handed, for the next resume"
    )

    with pytest.raises(ValueError, match="kept"):
        _run(_random(3), _metrics, tmp_path / "kept.csv", resume=records, n_passes=0)
    assert not (tmp_path / "kept.csv").exists(), "refused before any file is opened"


def test_a_resumed_nan_is_refused_as_not_finite_whatever_it_kept(tmp_path):
    terms = _head(seed=3, count=2)
    records = [TermRecord("pre_sample", i, term, {"score": math.nan}, loop_value=math.nan)
               for i, term in enumerate(terms)]

    with pytest.raises(ValueError, match="finite"):
        _run(_random(3), _metrics, tmp_path / "run.csv", resume=records, n_passes=0)
    assert not (tmp_path / "run.csv").exists()


def test_a_pool_whose_values_another_reading_gave_does_not_resume_under_the_first_objective(
        tmp_path):
    """The chain a review found: a design measured under one objective, taken over under a second
    with a reading, then resumed under the first without one, which mixed the two quantities."""
    length = MetricSchema.for_metrics(Objective("length"), ["score", "length"])
    _run(_random(4), _metrics, tmp_path / "a.csv", n_design=3, n_passes=0)
    _header, from_a = read_term_pool(tmp_path / "a_terms.pickle")
    _run(_random(5), _metrics, tmp_path / "b.csv", schema=length, resume=from_a, n_passes=0,
         resumed_value=lambda metrics: float(metrics["length"]))
    _header, from_b = read_term_pool(tmp_path / "b_terms.pickle")
    design_b = [record for record in from_b if record.phase == "pre_sample"]
    assert [r.loop_value for r in design_b] == [float(r.metrics["length"]) for r in design_b]

    with pytest.raises(ValueError, match="resumed_value"):
        _run(_random(6), _metrics, tmp_path / "c.csv", resume=design_b, n_passes=1)
    assert not (tmp_path / "c.csv").exists()

    # under the objective whose values the records kept, they resume without a reading
    outcome = _run(_random(6), _metrics, tmp_path / "d.csv", schema=length, resume=design_b,
                   n_passes=0)
    assert list(outcome.result["y"]) == [record.loop_value for record in design_b]


def test_run_paired_hands_no_reading_on_for_a_design_its_first_arm_evaluated(tmp_path):
    """A caller may pass a reading on every run; for a drawn design it reads nothing, and the arms
    start from the values the first arm's loop was handed."""
    outcomes = run_paired(
        {"bo": _bo(5), "random": _random(5)}, _metrics, schema=SCHEMA, n_design=3, n_passes=1,
        resumed_value=lambda metrics: 100.0 + metrics["length"],
        csv_paths={"bo": str(tmp_path / "bo.csv"), "random": str(tmp_path / "random.csv")},
        pretty_algebra=dict, echo=_quiet,
    )

    design = list(outcomes["bo"].result["y"][:3])
    assert all(value < 1.0 for value in design), "scores, not the reading's values"
    assert list(outcomes["random"].result["y"][:3]) == design


def test_run_paired_leaves_the_refusal_to_a_caller_that_wrote_its_own_files_first(tmp_path):
    """A driver that refuses a taken name itself and then writes each run's configuration before
    anything trains, so that an interrupted run keeps its provenance, would be refused by its own
    files; run_search takes refuse_taken=False for it, and so does run_paired."""
    for name in ("bo", "random"):
        (tmp_path / f"{name}_config.json").write_text("{}")

    outcomes = run_paired(
        {"bo": _bo(6), "random": _random(6)}, _metrics, schema=SCHEMA, n_design=2, n_passes=1,
        csv_paths={"bo": str(tmp_path / "bo.csv"), "random": str(tmp_path / "random.csv")},
        pretty_algebra=dict, echo=_quiet, refuse_taken=False,
    )

    assert [len(outcome.result["y"]) for outcome in outcomes.values()] == [3, 3]
    with pytest.raises(FileExistsError, match="bo_config.json"):
        run_paired(
            {"bo": _bo(6), "random": _random(6)}, _metrics, schema=SCHEMA, n_design=2,
            n_passes=1, csv_paths={"bo": str(tmp_path / "b2.csv"), "random": str(tmp_path / "bo.csv")},
            pretty_algebra=dict, echo=_quiet,
        )


def test_run_paired_resumes_every_arm_under_the_caller_s_reading(tmp_path):
    terms = _head(seed=4, count=3)
    records = [TermRecord("pre_sample", i, term, {"score": 0.1 * (i + 1), "accuracy": 0.9 - i / 10})
               for i, term in enumerate(terms)]

    outcomes = run_paired(
        {"bo": _bo(4), "random": _random(4)}, _metrics, schema=SCHEMA, resume=records,
        resumed_value=lambda metrics: metrics["accuracy"], n_passes=1,
        csv_paths={"bo": str(tmp_path / "bo.csv"), "random": str(tmp_path / "random.csv")},
        pretty_algebra=dict, echo=_quiet,
    )

    for name in ("bo", "random"):
        assert list(outcomes[name].result["y"][:3]) == pytest.approx([0.9, 0.8, 0.7]), name


def test_a_resumed_run_keeps_the_loop_values_for_the_next_resume(tmp_path):
    _run(_random(4), _metrics, tmp_path / "a.csv", n_design=3, n_passes=0)
    _h, from_a = read_term_pool(tmp_path / "a_terms.pickle")

    _run(_random(5), _metrics, tmp_path / "b.csv", resume=from_a, n_passes=1)
    _h, from_b = read_term_pool(tmp_path / "b_terms.pickle")
    design_b = [record for record in from_b if record.phase == "pre_sample"]

    assert [r.loop_value for r in design_b] == [r.loop_value for r in from_a]
    assert all(row["loop_value"] != "" for row in _rows(tmp_path / "b.csv"))
    outcome = _run(_random(6), _metrics, tmp_path / "c.csv", resume=design_b, n_passes=0)
    assert outcome.summary["taken_over"] == 3


def test_the_design_records_of_a_checked_pool_resume_a_run(tmp_path):
    _run(_random(7), _metrics, tmp_path / "a.csv", n_design=3, n_passes=1,
         provenance={"space": "lists", "bound": 6})

    records = load_design_records(str(tmp_path / "a_terms.pickle"), expected={"bound": 6})
    assert [r.phase for r in records] == ["pre_sample"] * 3
    outcome = _run(_random(8), _metrics, tmp_path / "b.csv", resume=records, n_passes=0)
    assert outcome.summary["taken_over"] == 3
    with pytest.raises(ValueError, match="different configuration"):
        load_design_records(str(tmp_path / "a_terms.pickle"), expected={"bound": 7})


# --- a paid evaluation is on disk ----------------------------------------------------------------

@pytest.mark.parametrize("at", ["design", "pass"])
def test_a_non_finite_objective_is_on_disk_before_the_loop_refuses_it(tmp_path, at):
    calls = []

    def evaluate(term):
        calls.append(term)
        metrics = _metrics(term)
        if len(calls) == (2 if at == "design" else 4):
            metrics["score"] = math.nan
        return metrics

    with pytest.raises(ValueError, match="not finite"):
        _run(_random(1), evaluate, tmp_path / "run.csv", n_design=3, n_passes=2)

    assert len(_rows(tmp_path / "run.csv")) == len(calls)
    _h, records = read_term_pool(tmp_path / "run_terms.pickle")
    assert len(records) == len(calls)
    assert math.isnan(records[-1].loop_value)


def test_an_evaluation_without_its_objective_is_on_disk_before_the_run_stops(tmp_path):
    calls = []

    def evaluate(term):
        calls.append(term)
        return {"length": 1} if len(calls) == 2 else _metrics(term)

    with pytest.raises(KeyError, match="score"):
        _run(_random(1), evaluate, tmp_path / "run.csv", n_design=3, n_passes=0)

    rows = _rows(tmp_path / "run.csv")
    assert len(rows) == 2
    assert rows[-1]["loop_value"] == ""
    assert rows[-1]["taken_over"] == "False", "an evaluation this run made, if a failed one"
    _h, records = read_term_pool(tmp_path / "run_terms.pickle")
    assert records[-1].loop_value is None
    assert records[-1].taken_over is False


# --- refused before a file is opened, so the retry runs ------------------------------------------

def _initialized():
    strategy = _random(0)
    strategy.initialize(initial_size=1)
    return strategy


REFUSALS = {
    "negative design": (lambda: _random(0), {"n_design": -1, "n_passes": 1}, ValueError),
    "nothing evaluated": (lambda: _random(0), {"n_design": 0, "n_passes": 0}, ValueError),
    "repeated design": (lambda: _random(0), {"design": [Tree("a"), Tree("a")]}, ValueError),
    "unhashable design": (lambda: _random(0), {"design": [[1]]}, TypeError),
    "initialized strategy": (_initialized, {"n_design": 1, "n_passes": 1}, RuntimeError),
    "bo without a design": (lambda: _bo(0), {"n_design": 0, "n_passes": 1}, ValueError),
    "non-finite resumed value": (
        lambda: _random(0),
        {"resume": [TermRecord("pre_sample", 0, Tree("a"), {"score": math.inf},
                               loop_value=math.inf)]},
        ValueError,
    ),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_a_refused_run_opens_nothing_and_its_retry_runs(tmp_path, case):
    make, kwargs, error = REFUSALS[case]
    with pytest.raises(error):
        _run(make(), _metrics, tmp_path / "run.csv", **kwargs)
    assert list(tmp_path.iterdir()) == [], "a refused run left files that block its retry"

    outcome = _run(_random(0), _metrics, tmp_path / "run.csv", n_design=2, n_passes=1)
    assert outcome.summary["completed"]


def test_step_budgets_are_positive():
    with pytest.raises(ValueError, match="acquisition_warn"):
        StepBudgets(acquisition_warn=0)


# --- run_paired refuses a later strategy's mistake before the first runs --------------------------

def _paired(tmp_path, strategies, csv_paths=None, **kwargs):
    evaluated = []
    names = list(strategies)
    csv_paths = csv_paths or {name: str(tmp_path / f"{name}.csv") for name in names}
    with pytest.raises((ValueError, FileExistsError, RuntimeError)) as refused:
        run_paired(strategies, lambda term: evaluated.append(term) or _metrics(term), schema=SCHEMA,
                   csv_paths=csv_paths, pretty_algebra=dict, n_design=2, n_passes=1, echo=_quiet,
                   **kwargs)
    return evaluated, refused


def test_run_paired_refuses_a_later_arm_s_taken_file_before_the_first_runs(tmp_path):
    (tmp_path / "random_ea.csv").write_text("taken")
    evaluated, refused = _paired(tmp_path, {"bo": _bo(1), "random": _random(1)})
    assert refused.type is FileExistsError
    assert evaluated == []
    assert not (tmp_path / "bo.csv").exists()


def test_run_paired_refuses_a_later_arm_s_configuration_before_the_first_runs(tmp_path):
    later = _bo(2)
    later.acquisition_function = "NoSuchScore"
    evaluated, _refused = _paired(tmp_path, {"first": _random(2), "later": later})
    assert evaluated == []


@pytest.mark.parametrize("paths", [("x.csv", "x.csv"), ("y.csv", "y.tsv")], ids=["same", "siblings"])
def test_run_paired_refuses_runs_that_share_a_file(tmp_path, paths):
    csv_paths = {"a": str(tmp_path / paths[0]), "b": str(tmp_path / paths[1])}
    evaluated, refused = _paired(tmp_path, {"a": _random(3), "b": _random(3)}, csv_paths=csv_paths)
    assert refused.type is ValueError
    assert evaluated == []


def test_run_paired_refuses_one_strategy_under_two_names(tmp_path):
    shared = _random(4)
    evaluated, refused = _paired(tmp_path, {"a": shared, "b": shared})
    assert refused.type is ValueError
    assert evaluated == []


def test_run_paired_refuses_a_strategy_that_already_ran(tmp_path):
    evaluated, refused = _paired(tmp_path, {"a": _random(5), "b": _initialized()})
    assert refused.type is RuntimeError
    assert evaluated == []


# --- the smaller promises ------------------------------------------------------------------------

def test_a_run_without_passes_is_named_for_its_design(tmp_path):
    outcome = _run(_bo(6), _metrics, tmp_path / "run.csv", n_design=2, n_passes=0)
    assert outcome.summary["run_kind"] == "design_only"
    assert outcome.summary["strategy"] == "BayesianOptimization"


def test_the_default_echo_flushes_each_line(tmp_path, monkeypatch):
    flushed = []
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: flushed.append(kwargs.get("flush")))
    run_search(_random(0), _metrics, schema=SCHEMA, csv_path=str(tmp_path / "run.csv"),
               pretty_algebra=dict, n_design=2, n_passes=1)
    assert flushed and all(flushed), "a killed run's log must hold the lines before the kill"


def test_resuming_a_legacy_pool_rewrites_it_under_the_current_tag(tmp_path):
    import pickle

    path = tmp_path / "legacy_terms.pickle"
    with path.open("wb") as handle:
        pickle.dump({"format": "cnn_damg_term_pool", "version": 1, "provenance": {}}, handle)
        pickle.dump(TermRecord("pre_sample", 0, Tree("a"), {}), handle)

    header, _records, writer, _salvaged = resume_term_pool(str(path))
    writer.write("bo_step", 0, Tree("b"), {})
    writer.close()

    header_back, records_back = read_term_pool(path)
    assert header_back["format"] == FORMAT, "the records are re-pickled under the new path"
    assert len(records_back) == 2


def test_the_recorder_s_exclusive_mode_checks_the_term_file_too(tmp_path):
    (tmp_path / "run_terms.pickle").write_bytes(b"another run's records")
    # The refusal names the run it protects, which the bare error of an exclusive open would not.
    with pytest.raises(FileExistsError, match="run_terms.pickle; refused rather than truncated"):
        EvaluationRecorder(str(tmp_path / "run.csv"), dict, SCHEMA, mode="x")
    assert not (tmp_path / "run.csv").exists()


def test_the_recorder_s_exclusive_mode_leaves_nothing_when_the_term_file_fails(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("the disk is full")

    monkeypatch.setattr(records_module, "TermPoolWriter", refuse)
    with pytest.raises(OSError, match="disk is full"):
        EvaluationRecorder(str(tmp_path / "run.csv"), dict, SCHEMA, mode="x")
    assert list(tmp_path.iterdir()) == []
    assert TermPoolWriter is not refuse


def test_a_schema_refuses_two_columns_of_one_name():
    with pytest.raises(ValueError, match="phase"):
        MetricSchema.for_metrics(Objective("phase"), ["phase"])


def test_the_default_live_line_shows_the_reported_metrics():
    schema = MetricSchema.for_metrics(Objective("f1"), ["f1", "n", "absent"])
    assert schema.live_line({"f1": 0.123456, "n": 3}) == "f1=0.12346 n=3"


def test_a_run_that_may_overwrite_does(tmp_path):
    _run(_random(0), _metrics, tmp_path / "run.csv", n_design=3, n_passes=0)
    _run(_random(1), _metrics, tmp_path / "run.csv", n_design=2, n_passes=0, refuse_taken=False)
    assert len(_rows(tmp_path / "run.csv")) == 2
