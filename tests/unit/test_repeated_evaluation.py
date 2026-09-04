"""Averaging a candidate over several trainings must average, not replace.

Two trainings of the same architecture do not reach the same accuracy, and the spread between them
can be wider than the difference a search is trying to read, so a single training cannot decide
between two candidates.  Repeating the measurement averages that spread down, and the whole point is
lost if the average silently drops what it averaged over, or if seeding the training moves the
search's own draws.

These tests fake the single training rather than run one.  What is under test is the aggregation and
the generator bookkeeping around it, and a real training would measure neither while costing
minutes.  ``test_the_search_draws_are_untouched`` is the one that would catch the subtle failure, a
run whose candidates change because the objective was repeated.
"""

from __future__ import annotations

import csv

import pytest
import torch

from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_experiment_utils as utils


def _fake_training(**overrides):
    """A stand-in for one training, returning a metric dict the aggregation can consume."""
    metrics = {
        "objective_value": 1.0,
        "accuracy": 0.5,
        "test_accuracy": 0.4,
        "n_params": 100,
        "train_seconds": 2.0,
        "diverged": False,
        "epochs_completed": 50,
    }
    metrics.update(overrides)
    return metrics


def _patch_trainings(monkeypatch, results):
    """Make ``_train_candidate_once`` hand back ``results`` in order, recording the seeds it saw."""
    seen_seeds = []
    remaining = list(results)

    def fake(tree, x, y, x_val, y_val, batch_size, protocol, x_test, y_test, seed):
        seen_seeds.append(seed)
        return remaining.pop(0)

    monkeypatch.setattr(utils, "_train_candidate_once", fake)
    return seen_seeds


def _evaluate(**kwargs):
    """Call the function under test with placeholders for everything it does not aggregate."""
    return utils.evaluate_candidate(
        tree=None, x=torch.zeros(1, 1), y=None, x_val=None, y_val=None, batch_size=1, **kwargs
    )


def test_a_single_repeat_is_the_measurement_it_always_was(monkeypatch):
    """One repetition has to be one training, or every run that asked for one has moved."""
    _patch_trainings(monkeypatch, [_fake_training(accuracy=0.61, objective_value=1.7)])

    metrics = _evaluate(repeats=1)

    assert metrics["accuracy"] == 0.61
    assert metrics["objective_value"] == 1.7
    assert metrics["train_seconds"] == 2.0
    assert metrics["n_repeats"] == 1
    # No spread was measured, so none is reported.  0.0 would claim the repetitions agreed.
    assert metrics["accuracy_std"] is None


def test_the_reported_value_is_the_mean_and_the_parts_survive(monkeypatch):
    """The loop optimizes the mean, and the CSV still has to show what the mean is a mean of."""
    _patch_trainings(monkeypatch, [
        _fake_training(accuracy=0.50, objective_value=1.0),
        _fake_training(accuracy=0.72, objective_value=2.0),
        _fake_training(accuracy=0.94, objective_value=3.0),
    ])

    metrics = _evaluate(repeats=3, training_seeds=(0, 1, 2))

    assert metrics["accuracy"] == pytest.approx(0.72)
    assert metrics["objective_value"] == pytest.approx(2.0)
    assert metrics["accuracy_runs"] == [0.50, 0.72, 0.94]
    assert metrics["accuracy_std"] == pytest.approx(0.22, abs=1e-9)
    # The sum, not the mean: this column is wall clock, and three trainings cost three trainings.
    assert metrics["train_seconds"] == pytest.approx(6.0)


def test_a_candidate_that_measured_the_same_thrice_is_not_one_that_scattered(monkeypatch):
    """The two cases share a mean and must not share a record."""
    _patch_trainings(monkeypatch, [_fake_training(accuracy=0.72) for _ in range(3)])
    steady = _evaluate(repeats=3, training_seeds=(0, 1, 2))

    _patch_trainings(monkeypatch, [
        _fake_training(accuracy=0.50), _fake_training(accuracy=0.72), _fake_training(accuracy=0.94),
    ])
    scattered = _evaluate(repeats=3, training_seeds=(0, 1, 2))

    assert steady["accuracy"] == pytest.approx(scattered["accuracy"])
    assert steady["accuracy_std"] == pytest.approx(0.0)
    assert scattered["accuracy_std"] > 0.2


def test_one_diverged_repetition_makes_the_candidate_diverged(monkeypatch):
    """Averaging a run whose numerics gave out with two healthy ones averages two different things.

    The flag is what says so, and the earliest stop is what says how far the worst one got.
    """
    _patch_trainings(monkeypatch, [
        _fake_training(diverged=False, epochs_completed=50),
        _fake_training(diverged=True, epochs_completed=3),
        _fake_training(diverged=False, epochs_completed=50),
    ])

    metrics = _evaluate(repeats=3, training_seeds=(0, 1, 2))

    assert metrics["diverged"] is True
    assert metrics["epochs_completed"] == 3
    assert metrics["diverged_runs"] == [False, True, False]


def test_the_seeds_reach_the_trainings_in_order(monkeypatch):
    """Each repetition gets its own seed, without which the repetitions are not reproducible."""
    seen = _patch_trainings(monkeypatch, [_fake_training() for _ in range(3)])

    _evaluate(repeats=3, training_seeds=(7, 8, 9))

    assert seen == [7, 8, 9]


def test_unseeded_stays_unseeded(monkeypatch):
    """Without ``training_seeds`` nothing is seeded, which is unseeded training."""
    seen = _patch_trainings(monkeypatch, [_fake_training() for _ in range(2)])

    metrics = _evaluate(repeats=2)

    assert seen == [None, None]
    assert metrics["training_seeds"] is None


def test_the_search_draws_are_untouched():
    """The one that matters: seeding the training must not move the global generator.

    The loop's own sampler and the acquisition-optimizing search draw from it, so a training that
    left it advanced, or worse reset it to a fixed seed, would make a run with three repetitions
    search differently than the same run with one.  The comparison between the two would then
    measure the averaging and a different search at once, and nothing would say which.

    Runs the real function with a fake training that itself consumes randomness, because a stub
    that draws nothing would pass this test without the restore being there at all.
    """
    def drawing_training(tree, x, y, x_val, y_val, batch_size, protocol, x_test, y_test, seed):
        if seed is not None:
            torch.manual_seed(seed)
        return _fake_training(accuracy=float(torch.rand(1).item()))

    original = utils._train_candidate_once
    utils._train_candidate_once = drawing_training
    try:
        torch.manual_seed(20260804)
        before = torch.rand(3)

        torch.manual_seed(20260804)
        utils.evaluate_candidate(
            tree=None, x=torch.zeros(1, 1), y=None, x_val=None, y_val=None, batch_size=1,
            repeats=3, training_seeds=(0, 1, 2),
        )
        after = torch.rand(3)
    finally:
        utils._train_candidate_once = original

    assert torch.equal(before, after), (
        "the training consumed or reset the generator the search draws from; a repeated run would "
        "see different candidates than an unrepeated one"
    )


def test_the_same_seeds_give_the_same_average():
    """Seeded repetitions must be reproducible, or the average is not a fixed quantity."""
    def drawing_training(tree, x, y, x_val, y_val, batch_size, protocol, x_test, y_test, seed):
        torch.manual_seed(seed)
        return _fake_training(accuracy=float(torch.rand(1).item()))

    original = utils._train_candidate_once
    utils._train_candidate_once = drawing_training
    try:
        first = utils.evaluate_candidate(
            tree=None, x=torch.zeros(1, 1), y=None, x_val=None, y_val=None, batch_size=1,
            repeats=3, training_seeds=(0, 1, 2),
        )
        second = utils.evaluate_candidate(
            tree=None, x=torch.zeros(1, 1), y=None, x_val=None, y_val=None, batch_size=1,
            repeats=3, training_seeds=(0, 1, 2),
        )
    finally:
        utils._train_candidate_once = original

    assert first["accuracy_runs"] == second["accuracy_runs"]


def test_a_seed_list_that_does_not_match_the_repetitions_is_refused(monkeypatch):
    _patch_trainings(monkeypatch, [_fake_training() for _ in range(3)])

    with pytest.raises(ValueError, match="one seed per repetition"):
        _evaluate(repeats=3, training_seeds=(0, 1))


def test_a_repeat_count_below_one_is_refused(monkeypatch):
    """Zero trainings is not a measurement, and a mean of nothing is not a value.

    Without the check the aggregation raises somewhere inside ``statistics``, which names the
    library rather than the argument that was wrong.
    """
    _patch_trainings(monkeypatch, [])

    with pytest.raises(ValueError, match="repeats must be at least 1"):
        _evaluate(repeats=0)


def test_repetitions_that_disagree_on_the_network_are_refused(monkeypatch):
    """The same term must interpret to the same network, and no average repairs two of them."""
    _patch_trainings(monkeypatch, [
        _fake_training(n_params=100), _fake_training(n_params=101),
    ])

    with pytest.raises(ValueError, match="different parameter counts"):
        _evaluate(repeats=2, training_seeds=(0, 1))


def test_the_csv_carries_the_repetitions(tmp_path):
    """The mean alone in the file would be a number nobody can check afterwards."""
    from cosy.core.tree import Tree

    missing = {"n_repeats", "accuracy_runs", "accuracy_std", "diverged_runs"} - set(
        utils.CSV_COLUMNS
    )
    assert not missing, f"a repeated measurement cannot be read back: {sorted(missing)}"

    path = tmp_path / "run.csv"
    with utils.EvaluationLogger(str(path), dict) as logger:
        logger.log("pre_sample", 0, Tree("arch"), {
            "objective_value": 1.0, "accuracy": 0.72, "n_repeats": 3,
            "accuracy_runs": [0.50, 0.72, 0.94], "accuracy_std": 0.22,
            "diverged_runs": [False, True, False],
        })

    with open(path, newline="") as handle:
        (row,) = list(csv.DictReader(handle))
    assert row["n_repeats"] == "3"
    assert row["accuracy_runs"] == "0.5 0.72 0.94"
    assert row["accuracy_std"] == "0.22"
    assert row["diverged_runs"] == "False True False"


def test_a_run_without_repetitions_leaves_the_columns_empty(tmp_path):
    """An unrepeated run must not claim a spread it never measured."""
    from cosy.core.tree import Tree

    path = tmp_path / "run.csv"
    with utils.EvaluationLogger(str(path), dict) as logger:
        logger.log("pre_sample", 0, Tree("arch"), {"objective_value": 1.0, "accuracy": 0.72})

    with open(path, newline="") as handle:
        (row,) = list(csv.DictReader(handle))
    assert row["accuracy_std"] == ""
    assert row["accuracy_runs"] == ""
    assert row["n_repeats"] == ""
