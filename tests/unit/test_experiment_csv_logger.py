"""The result file must record whether a row came from the optimizer or from a fallback.

Part of plan step S0b.3.  The CSV is the only artifact that survives a run, so a degradation the
core now warns about is still invisible after the fact unless the columns carry it.  Both A30 runs
were read from these files alone.
"""

from __future__ import annotations

import csv

from cosy.core.tree import Tree

from bayesian_optimization.state import Suggestion


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_columns_cover_the_fallback_state():
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_experiment_utils import CSV_COLUMNS

    missing = {"acquisition_value", "fallback_used", "fallback_attempts"} - set(CSV_COLUMNS)
    assert not missing, f"the result file cannot show a degraded iteration: {sorted(missing)}"


def test_bo_step_records_acquisition_value_and_fallback(tmp_path):
    """A row from a fallback iteration must be distinguishable from a genuine BO step."""
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_experiment_utils import (
        ExperimentCSVLogger,
    )

    tree = Tree("arch")
    suggestion = Suggestion(
        candidate=tree,
        acquisition_value=0.75,
        diagnostics={"fallback_used": True, "fallback_attempts": 2, "iteration": 4},
    )

    path = tmp_path / "run.csv"
    # ``dict`` as the algebra factory yields an empty interpretation, so ``interpret`` falls back to
    # the combinator itself -- enough to exercise the writer without the CNN algebras.
    with ExperimentCSVLogger(str(path), dict) as logger:
        logger.log("bo_step", 0, tree, {"objective_value": 1.25}, suggestion=suggestion)

    (row,) = _read(path)
    assert row["acquisition_value"] == "0.75"
    assert row["fallback_used"] == "True"
    assert row["fallback_attempts"] == "2"


def test_row_without_a_suggestion_leaves_the_columns_empty(tmp_path):
    """Pre-sample rows have no acquisition step, so the cells stay empty rather than claim "False".

    Writing ``False`` would assert that no fallback happened in an iteration that never ran one --
    the same silent substitute value this phase removes elsewhere.
    """
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_experiment_utils import (
        ExperimentCSVLogger,
    )

    path = tmp_path / "run.csv"
    with ExperimentCSVLogger(str(path), dict) as logger:
        logger.log("pre_sample", 0, Tree("arch"), {"objective_value": 1.25})

    (row,) = _read(path)
    assert row["acquisition_value"] == ""
    assert row["fallback_used"] == ""
    assert row["fallback_attempts"] == ""
