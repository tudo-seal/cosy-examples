"""The result file must record whether a row came from the optimizer or from a fallback.

The CSV is the artifact a finished run is read from, so a degradation the loop warns about while it
runs is invisible afterwards unless the columns carry it.  A run whose every pass fell back to a
random sample was random search, and nothing else in the file says so.
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
        EvaluationLogger,
    )

    tree = Tree("arch")
    suggestion = Suggestion(
        candidate=tree,
        acquisition_value=0.75,
        diagnostics={"fallback_used": True, "fallback_attempts": 2, "iteration": 4},
    )

    path = tmp_path / "run.csv"
    # ``dict`` as the algebra factory yields an empty interpretation, so ``interpret`` falls back
    # to the combinator itself, which exercises the writer without the CNN algebras.
    with EvaluationLogger(str(path), dict) as logger:
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
        EvaluationLogger,
    )

    path = tmp_path / "run.csv"
    with EvaluationLogger(str(path), dict) as logger:
        logger.log("pre_sample", 0, Tree("arch"), {"objective_value": 1.25})

    (row,) = _read(path)
    assert row["acquisition_value"] == ""
    assert row["fallback_used"] == ""
    assert row["fallback_attempts"] == ""


def test_one_call_writes_both_the_row_and_the_term(tmp_path):
    """The term record cannot be forgotten, because it is not a separate call.

    A run whose CSV has rows but whose pickle has none cannot be analyzed afterwards and can only be
    repaired by retraining, which is what makes this an invariant of the logger rather than a
    convention its call sites are expected to keep.
    """
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_experiment_utils import (
        EvaluationLogger,
    )
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_pool import read_term_pool

    # Leafless, like the other tests here: ``dict`` as the algebra factory renders a combinator by
    # falling back to the combinator itself, which takes no arguments.
    tree = Tree("arch")
    path = tmp_path / "run.csv"
    with EvaluationLogger(str(path), dict, provenance={"dataset": "usps"}) as logger:
        logger.log("pre_sample", 0, tree, {"objective_value": 1.25, "accuracy": 0.94})

    header, records = read_term_pool(tmp_path / "run_terms.pickle")
    assert header["provenance"] == {"dataset": "usps"}
    (record,) = records
    assert record.term == tree, "the pickle must carry the term, not a rendering of it"
    assert record.phase == "pre_sample"
    assert record.metrics["accuracy"] == 0.94
    assert len(_read(path)) == 1, "and the CSV row is still written"
