"""The driver over the design phase: refused before anything is paid, and nothing it measured lost.

``run_ask_tell_search`` evaluates the initial design term by term through the loop's design phase.
Three promises of that loop are checked here: a configuration no pass could use is refused before
a single term is drawn or trained, a value the loop refuses is on disk before it is refused, and a
value that arrives without its metrics is a stop rather than a row filled in.
"""

from __future__ import annotations

import csv

import pytest
from cosy.core.tree import Tree

from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_experiment_utils as utils


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _metrics(objective_value=1.0):
    return {
        "objective_value": objective_value, "accuracy": 0.5, "n_params": 10,
        "train_seconds": 0.1, "diverged": False, "epochs_completed": 1,
    }


def _run(optimizer, f_obj, metrics_by_tree, path, *, n_pre_samples, n_iterations=0):
    utils.run_ask_tell_search(
        optimizer=optimizer,
        f_obj=f_obj,
        metrics_by_tree=metrics_by_tree,
        as_reported=lambda v: v,
        objective="accuracy",
        greater_is_better=True,
        n_pre_samples=n_pre_samples,
        n_iterations=n_iterations,
        csv_path=str(path),
        pretty_algebra=dict,
        baseline=True,
        verbose=False,
    )


def test_a_configuration_no_pass_could_use_is_refused_before_anything_is_drawn(
    monkeypatch, bo_factory, tmp_path
):
    """A mistyped acquisition costs no draw and no training, as it costs ``optimize()`` none."""
    draws, trained = [], []

    def fake_draw(optimizer, count):
        draws.append(count)
        return [Tree("a"), Tree("b"), Tree("c")], 0

    def f_obj(tree):
        trained.append(tree)
        return 0.5

    monkeypatch.setattr(utils, "_draw_prefix", fake_draw)
    with pytest.raises(ValueError):
        _run(bo_factory(acquisition_function="NoSuchScore"), f_obj, {}, tmp_path / "run.csv",
             n_pre_samples=2, n_iterations=1)

    assert draws == [], "the design was drawn before the configuration was refused"
    assert trained == [], "a network was trained before the configuration was refused"


def test_a_non_finite_design_value_is_on_disk_before_the_loop_refuses_it(
    monkeypatch, bo_factory, tmp_path
):
    """The failing evaluation's row is evidence, and it is written before the refusal."""
    design = [Tree("a"), Tree("b"), Tree("c")]
    monkeypatch.setattr(utils, "_draw_prefix", lambda optimizer, count: (list(design), 0))
    path = tmp_path / "run.csv"
    metrics_by_tree = {}

    def f_obj(tree):
        diverged = tree == Tree("b")
        metrics_by_tree[tree] = _metrics(float("nan") if diverged else 1.0)
        return float("nan") if diverged else 0.5

    with pytest.raises(ValueError, match="not finite"):
        _run(bo_factory(), f_obj, metrics_by_tree, path, n_pre_samples=3)

    assert [row["index"] for row in _rows(path)] == ["0", "1"], (
        "the refused evaluation must be on disk, and nothing after it"
    )


def test_a_design_value_without_its_metrics_is_refused(monkeypatch, bo_factory, tmp_path):
    """No substitute for a missing measurement: a value without metrics is a stop."""
    monkeypatch.setattr(
        utils, "_draw_prefix", lambda optimizer, count: ([Tree("a"), Tree("b")], 0)
    )

    with pytest.raises(KeyError, match="recorded no metrics"):
        _run(bo_factory(), lambda tree: 0.5, {}, tmp_path / "run.csv", n_pre_samples=2)
