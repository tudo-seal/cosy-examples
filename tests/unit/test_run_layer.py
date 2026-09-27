"""The run layer's records: a caller's own metrics, one set of artifacts, a neutral provenance.

The CIFAR driver wrote every run with the CIFAR example's columns and read the CIFAR metric keys
for its prints, so another caller had to carry ``accuracy``, ``n_params`` and ``train_seconds``
under those names and kept its own record in a second CSV.  A ``MetricSchema`` names the columns
and the objective instead; the CIFAR layout is one schema among others and writes exactly what it
wrote before.  ``RunArtifacts`` names every file a run writes, so a second start against a taken
name is refused before anything is truncated.  The provenance writer records the environment a
caller hands it, and no framework the run did not use.
"""

from __future__ import annotations

import csv
import json

import pytest
from cosy.core.tree import Tree

from bayesian_optimization.runs.artifacts import RunArtifacts
from bayesian_optimization.runs.budgets import StepBudgets
from bayesian_optimization.runs.metadata import write_run_metadata
from bayesian_optimization.runs.records import EvaluationRecorder
from bayesian_optimization.runs.schema import MetricSchema, Objective
from bayesian_optimization.runs.term_pool import read_term_pool
from bayesian_optimization.state import Suggestion

# The CIFAR driver's columns at e172cb4, in their order: the CIFAR schema must still write them.
CIFAR_COLUMNS_AT_E172CB4 = [
    "phase", "index", "structure", "objective_value", "accuracy", "test_accuracy", "term_size",
    "n_params", "train_seconds", "diverged", "epochs_completed", "acquisition_seconds",
    "acquisition_value", "fallback_used", "fallback_attempts", "timestamp", "n_repeats",
    "accuracy_runs", "accuracy_std", "diverged_runs",
]


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.reader(handle))


def test_the_loader_of_design_records_is_the_run_layer_s():
    """``run_search(resume=)`` takes records, and the loader that checks them is where it is."""
    from bayesian_optimization import runs
    from bayesian_optimization.runs import resume

    assert runs.load_design_records is resume.load_design_records
    assert "load_design_records" in runs.__all__


# --- the objective ------------------------------------------------------------------------------

def test_the_objective_reads_the_loop_s_value_off_the_metrics_in_its_direction():
    loss = Objective("loss", greater_is_better=False)
    assert loss.loop_value({"loss": 0.25}) == -0.25
    assert loss.as_reported(-0.25) == 0.25

    score = Objective("f1")
    assert score.loop_value({"f1": 0.7}) == 0.7
    assert score.as_reported(0.7) == 0.7


def test_an_objective_whose_key_is_missing_says_which_key():
    with pytest.raises(KeyError, match="val_f1"):
        Objective("val_f1").loop_value({"accuracy": 0.9})


# --- the schema and the recorder ----------------------------------------------------------------

def test_a_schema_of_one_s_own_metrics_writes_those_and_not_the_cifar_ones(tmp_path):
    schema = MetricSchema.for_metrics(Objective("f1"), ["f1", "n_params"])
    path = tmp_path / "run.csv"
    with EvaluationRecorder(str(path), dict, schema) as recorder:
        recorder.log("pre_sample", 0, Tree("a"), {"f1": 0.7, "n_params": 12}, loop_value=0.7)

    header, row = _rows(path)
    assert "accuracy" not in header
    assert header[:4] == ["phase", "index", "structure", "loop_value"]
    assert {"f1", "n_params", "term_size", "acquisition_value", "timestamp"} <= set(header)
    cells = dict(zip(header, row, strict=True))
    assert (cells["phase"], cells["index"], cells["structure"]) == ("pre_sample", "0", "a")
    assert (cells["loop_value"], cells["f1"], cells["n_params"]) == ("0.7", "0.7", "12")
    assert cells["acquisition_value"] == "", "a design row has no acquisition step"


def test_the_recorder_keeps_the_loop_s_value_in_the_term_record(tmp_path):
    schema = MetricSchema.for_metrics(Objective("loss", greater_is_better=False), ["loss"])
    path = tmp_path / "run.csv"
    with EvaluationRecorder(str(path), dict, schema) as recorder:
        recorder.log("pre_sample", 0, Tree("a"), {"loss": 0.4}, loop_value=-0.4)

    _header, records = read_term_pool(tmp_path / "run_terms.pickle")
    assert records[0].loop_value == -0.4
    assert records[0].metrics == {"loss": 0.4}


def test_a_pass_row_carries_its_acquisition_reading(tmp_path):
    schema = MetricSchema.for_metrics(Objective("f1"), ["f1"])
    path = tmp_path / "run.csv"
    suggestion = Suggestion(
        candidate=Tree("b"), acquisition_value=0.125,
        diagnostics={"iteration": 3, "fallback_used": False, "fallback_attempts": 0},
    )
    with EvaluationRecorder(str(path), dict, schema) as recorder:
        recorder.log("bo_step", 3, Tree("b"), {"f1": 0.5}, suggestion=suggestion,
                     acquisition_seconds=2.5, loop_value=0.5)

    header, row = _rows(path)
    cells = dict(zip(header, row, strict=True))
    assert (cells["acquisition_value"], cells["acquisition_seconds"]) == ("0.125", "2.5")
    assert cells["fallback_used"] == "False"


def test_the_cifar_schema_writes_the_cifar_driver_s_row_unchanged(tmp_path, monkeypatch):
    """The CIFAR layout is one schema: the same columns, the same cells, the same order."""
    from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_experiment_utils as utils
    from bayesian_optimization.runs import records

    assert utils.CSV_COLUMNS == CIFAR_COLUMNS_AT_E172CB4
    monkeypatch.setattr(records.time, "time", lambda: 1234.5)
    path = tmp_path / "run.csv"
    metrics = {
        "objective_value": 0.61, "accuracy": 0.8, "test_accuracy": None, "n_params": 99,
        "train_seconds": 3.5, "diverged": False, "epochs_completed": 2, "n_repeats": 2,
        "accuracy_runs": [0.79, 0.81], "accuracy_std": 0.01, "diverged_runs": [False, False],
    }
    suggestion = Suggestion(
        candidate=Tree("c"), acquisition_value=0.2,
        diagnostics={"fallback_used": True, "fallback_attempts": 1},
    )
    with utils.EvaluationLogger(str(path), dict) as logger:
        logger.log("bo_step", 4, Tree("c"), metrics, suggestion=suggestion, acquisition_seconds=7.0)

    header, row = _rows(path)
    assert header == CIFAR_COLUMNS_AT_E172CB4
    assert row == [
        "bo_step", "4", "c", "0.61", "0.8", "", "1", "99", "3.5", "False", "2", "7.0", "0.2",
        "True", "1", "1234.5", "2", "0.79 0.81", "0.01", "False False",
    ]


def test_a_recorder_asked_to_refuse_a_taken_run_refuses_it(tmp_path):
    path = tmp_path / "run.csv"
    path.write_text("a run that is still being written\n")
    schema = MetricSchema.for_metrics(Objective("f1"), ["f1"])

    with pytest.raises(FileExistsError):
        EvaluationRecorder(str(path), dict, schema, mode="x")
    assert path.read_text() == "a run that is still being written\n"


# --- the artifacts ------------------------------------------------------------------------------

def test_the_artifacts_of_a_run_are_named_after_its_csv():
    artifacts = RunArtifacts("out/run.csv")
    assert artifacts.paths() == {
        "csv": "out/run.csv",
        "terms": "out/run_terms.pickle",
        "ea": "out/run_ea.csv",
        "surrogate": "out/run_surrogate.csv",
        "config": "out/run_config.json",
        "diagnostics": "out/run_diagnostics.json",
        "trace": "out/run_trace.csv",
    }


def test_a_run_whose_name_is_taken_is_refused_before_anything_is_opened(tmp_path):
    artifacts = RunArtifacts(str(tmp_path / "run.csv"))
    assert artifacts.taken() == []
    artifacts.refuse_taken()  # nothing taken, nothing refused

    (tmp_path / "run_surrogate.csv").write_text("x")
    (tmp_path / "run_config.json").write_text("{}")
    assert sorted(artifacts.taken()) == sorted(
        [str(tmp_path / "run_surrogate.csv"), str(tmp_path / "run_config.json")]
    )
    with pytest.raises(FileExistsError, match="run_surrogate.csv") as refused:
        artifacts.refuse_taken()
    assert "run_config.json" in str(refused.value)


# --- the provenance -----------------------------------------------------------------------------

def test_the_provenance_records_the_environment_it_is_handed_and_no_other(tmp_path):
    csv_path = str(tmp_path / "run.csv")
    written = write_run_metadata(csv_path, {"seed": 7}, environment={"tensorflow": "2.21"})

    with open(written) as handle:
        record = json.load(handle)
    assert written == str(tmp_path / "run_config.json")
    assert record["seed"] == 7
    assert record["csv_path"] == csv_path
    assert record["tensorflow"] == "2.21"
    assert "python_version" in record and "written_at" in record
    assert not any(key.startswith("torch") for key in record), "a framework the run did not use"


# --- the budgets --------------------------------------------------------------------------------

def test_the_step_budgets_default_to_the_cifar_example_s_and_are_the_caller_s_to_set():
    from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_experiment_utils as utils

    budgets = StepBudgets()
    assert budgets.per_evaluation == utils.PER_EVALUATION_WARN_SECONDS
    assert budgets.acquisition_warn == utils.ACQUISITION_WARN_SECONDS
    assert budgets.acquisition_hard_limit == utils.ACQUISITION_HARD_LIMIT_SECONDS
    assert budgets.determinization == utils.DETERMINIZATION_WARN_SECONDS
    assert budgets.space_construction == utils.SPACE_CONSTRUCTION_WARN_SECONDS
    assert StepBudgets(per_evaluation=5).per_evaluation == 5
