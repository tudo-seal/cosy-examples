"""A run interrupted during initialization must keep the networks it already trained.

The ask/tell loop exists so that results survive an interruption, and that used to hold for the loop
passes and not for the initial design: the pre-samples were evaluated into a list and logged
afterwards, so an interruption anywhere in that phase left an empty CSV however many networks had
been trained.  With repeated measurements the phase is hours rather than minutes, and the same
silence makes "still running" and "stuck" indistinguishable from outside.

The test asserts the property directly rather than the code path: while the k-th pre-sample is being
measured, the file has to hold the k-1 before it already.
"""

from __future__ import annotations

import csv

import pytest
from cosy.core.tree import Tree

from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_experiment_utils as utils


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def paired_run(monkeypatch, bo_factory):
    """A ``run_ask_tell_search`` whose drawn design is fixed and whose training is instant."""
    # Leaves only: ``pretty_algebra=dict`` yields an empty interpretation, so a combinator
    # with children has nothing to apply and ``interpret`` raises.  The design under test is
    # the write order, not the rendering.
    design = [Tree(name) for name in ("a", "b", "c", "d")]

    def fake_draw(optimizer, count):
        assert count == len(design), "the paired design is one stream of draws"
        return list(design), 0

    monkeypatch.setattr(utils, "_draw_prefix", fake_draw)
    return design, bo_factory


def test_each_pre_sample_is_on_disk_before_the_next_one_is_trained(paired_run, tmp_path):
    """The property that makes an interrupted run readable."""
    design, bo_factory = paired_run
    path = tmp_path / "run.csv"
    seen_before_each = []

    def f_obj(tree):
        # What the file holds at the moment this candidate starts training.
        seen_before_each.append(len(_rows(path)) if path.exists() else 0)
        metrics_by_tree[tree] = {
            "objective_value": 1.0, "accuracy": 0.5, "n_params": 10,
            "train_seconds": 0.1, "diverged": False, "epochs_completed": 1,
        }
        return 0.5

    metrics_by_tree = {}
    utils.run_ask_tell_search(
        optimizer=bo_factory(),
        f_obj=f_obj,
        metrics_by_tree=metrics_by_tree,
        as_reported=lambda v: v,
        objective="accuracy",
        greater_is_better=True,
        n_pre_samples=len(design),
        n_iterations=0,
        csv_path=str(path),
        pretty_algebra=dict,
        baseline=True,
        verbose=False,
    )

    assert seen_before_each == [0, 1, 2, 3], (
        "the initial design was written after it was complete, not as it was measured: a run "
        f"interrupted during it would have lost every trained network (saw {seen_before_each})"
    )


def test_the_design_is_written_once_and_in_stream_order(paired_run, tmp_path):
    """The interleaved write must not duplicate the rows the later loop used to produce."""
    design, bo_factory = paired_run
    path = tmp_path / "run.csv"
    metrics_by_tree = {}

    def f_obj(tree):
        metrics_by_tree[tree] = {
            "objective_value": 1.0, "accuracy": 0.5 + 0.01 * len(metrics_by_tree), "n_params": 10,
            "train_seconds": 0.1, "diverged": False, "epochs_completed": 1,
        }
        return metrics_by_tree[tree]["accuracy"]

    utils.run_ask_tell_search(
        optimizer=bo_factory(),
        f_obj=f_obj,
        metrics_by_tree=metrics_by_tree,
        as_reported=lambda v: v,
        objective="accuracy",
        greater_is_better=True,
        n_pre_samples=len(design),
        n_iterations=0,
        csv_path=str(path),
        pretty_algebra=dict,
        baseline=True,
        verbose=False,
    )

    rows = _rows(path)
    assert [row["phase"] for row in rows] == ["pre_sample"] * len(design)
    assert [row["index"] for row in rows] == [str(i) for i in range(len(design))]
    # Rising accuracies in evaluation order: the file is the order the networks were measured in.
    assert [round(float(row["accuracy"]), 2) for row in rows] == [0.50, 0.51, 0.52, 0.53]


# The unpaired path has no test here on purpose.  It draws inside ``initialize()``, which needs a
# real search space, and a test that stubbed that out would assert the stub rather than the
# behavior.
