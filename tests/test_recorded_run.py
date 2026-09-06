"""The recorded VGGM run still says what it said when it was recorded.

``recorded_runs/`` holds one search of the CIFAR-10 example, VGGM over 20 initial terms and 30
loop passes with a paired random arm, together with the anchor the reference architecture reached.
The anchor file does not name the card it ran on, so its accuracies compare with the search and
its seconds do not. Nothing in the package reads these files, so without a test they are eight
files that no failure can reach. That is what this module is for.

It checks seven things. Only the column order is the file written out again, and it is written
out on purpose: a column that moves changes what every reader gets back.

The run is internally consistent. The summary fields of the configuration are maxima and counts
over the per-evaluation CSV, so a truncated, reordered or hand-edited CSV contradicts the
configuration beside it. The trace, the generation log and the surrogate log are held against the
same budget, and the acquisition value at each pick has to agree across the three files that
record it.

The term pool describes the same evaluations. Every record renders through
``pretty_term_algebra`` to the ``structure`` column of the row with the same phase and index, so
the pool of another run, or a change to how a term is rendered, no longer matches.

The anchor is the architecture the verifier demands. Its parameter count is held against
``cnn_damg_verify_reference.EXPECTED_PARAMETERS`` rather than written out again, so moving the
constant without a new anchor run is a failure and not a silent drift.

The three known defects of the record stay visible. The configuration describes the tutorial
search space although the run searched the VGG cell, its ``learning_rate_values`` is not the rate
any of the 80 evaluated terms carries, and nothing in the record says that the initial design was
taken over from an earlier attempt rather than trained. All three are pinned below, because the
files are evidence and correcting them would replace what happened with what should have happened.
All three are defects of the writer, in two places, and both places are repaired since: the
search-space block of the CIFAR-10 experiment is read off the repository now, and the metadata it
writes carries a ``resumed_from`` field. The record predates both.
"""

from __future__ import annotations

import collections
import csv
import json
import re
from pathlib import Path

import pytest

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_algebras import pretty_term_algebra
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_pool import FORMAT, read_term_pool
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_verify_reference import (
    EXPECTED_PARAMETERS,
)

RECORDED = Path(__file__).resolve().parents[1] / "recorded_runs"
RUN = "bo_VGGM_20260805_r3"

#: The seven artifacts one run writes: its CSV, and the six files that go beside it.
ARTIFACTS = (
    f"{RUN}.csv",
    f"{RUN}_config.json",
    f"{RUN}_diagnostics.json",
    f"{RUN}_ea.csv",
    f"{RUN}_surrogate.csv",
    f"{RUN}_trace.csv",
    f"{RUN}_terms.pickle",
)

#: The columns of the per-evaluation file, in order. A column that moves changes what every reader
#: of this file gets back, and a reader that goes by position gets it silently.
COLUMNS = (
    "phase", "index", "structure", "objective_value", "accuracy", "test_accuracy", "term_size",
    "n_params", "train_seconds", "diverged", "epochs_completed", "acquisition_seconds",
    "acquisition_value", "fallback_used", "fallback_attempts", "timestamp", "n_repeats",
    "accuracy_runs", "accuracy_std", "diverged_runs",
)

#: How many evaluations each phase of the run holds. The initial design is shared between the two
#: arms, so the file holds 80 rows for 50 searched and 50 baseline evaluations.
PHASE_SIZES = {"pre_sample": 20, "bo_step": 30, "random_sample": 30}

_LEARNING_RATE = re.compile(r"CNNrepository\.(?:SGD|Adam)\(learning_rate=([0-9.eE+-]+)")


def _rows():
    with (RECORDED / f"{RUN}.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def _config():
    return json.loads((RECORDED / f"{RUN}_config.json").read_text())


def test_the_run_left_all_seven_artifacts_behind():
    """A record missing one file answers fewer questions, and says nothing about which."""
    missing = [name for name in ARTIFACTS if not (RECORDED / name).is_file()]
    assert missing == [], f"the recorded run is incomplete: {missing}"


def test_the_per_evaluation_file_holds_the_columns_and_the_rows_of_the_recorded_budget():
    """80 rows over three phases, and the columns in the order the run wrote them."""
    rows = _rows()
    with (RECORDED / f"{RUN}.csv").open(newline="") as handle:
        assert tuple(next(csv.reader(handle))) == COLUMNS

    assert len(rows) == sum(PHASE_SIZES.values())
    counted = collections.Counter(row["phase"] for row in rows)
    assert dict(counted) == PHASE_SIZES

    for phase, size in PHASE_SIZES.items():
        indices = [int(row["index"]) for row in rows if row["phase"] == phase]
        assert indices == list(range(size)), phase


def test_the_configuration_is_the_summary_of_the_file_beside_it():
    """The summary numbers of the configuration are counts, maxima and one sum over the CSV.

    This is what makes the two files one record. A CSV that loses rows, or a configuration copied
    from a different run, contradicts its neighbor here rather than being read as the truth.
    """
    rows, config = _rows(), _config()
    accuracy = {(row["phase"], int(row["index"])): float(row["accuracy"]) for row in rows}

    def best_over(*phases):
        return max(value for (phase, _), value in accuracy.items() if phase in phases)

    assert config["target_cell"] == "VGGM"
    assert config["run_kind"] == "bayesian_optimization_with_paired_baseline"
    assert config["completed"] is True

    loop = config["bayesian_optimization"]
    assert loop["n_pre_samples"] == PHASE_SIZES["pre_sample"]
    assert loop["n_iterations"] == PHASE_SIZES["bo_step"]
    assert config["n_evaluations"] == loop["n_pre_samples"] + loop["n_iterations"]
    assert config["baseline_evaluations"] == config["n_evaluations"]
    assert config["shared_initial_evaluations"] == PHASE_SIZES["pre_sample"]

    # The searched arm and the random arm share the initial design, so each best is taken over the
    # design and one arm, and the two differences are what the paired comparison reports.
    assert config["best_objective_value"] == best_over("pre_sample", "bo_step")
    assert config["baseline_best_objective_value"] == best_over("pre_sample", "random_sample")
    assert config["shared_initial_best_objective_value"] == best_over("pre_sample")

    assert config["training_repeats"] == 3
    assert {int(row["n_repeats"]) for row in rows} == {config["training_repeats"]}
    assert config["epochs_per_candidate"] == 50
    assert {int(row["epochs_completed"]) for row in rows} == {config["epochs_per_candidate"]}

    # The training total is the sum over all 80 rows and not over the 50 the loop paid for, so it
    # counts the random arm and the taken-over design as well.
    assert config["total_training_seconds"] == pytest.approx(
        sum(float(row["train_seconds"]) for row in rows), rel=1e-12)


def test_the_recorded_run_drew_depth_bounded_and_never_determinized_the_space():
    """The run drew from the sampler that never counts, and the record says so twice.

    The command line defaults to ``--sampling size-uniform``, which determinizes the space and
    draws from the determinized program. This run passed ``--sampling depth-bounded`` instead, so
    a rerun that leaves the option out runs a different search rather than a slower one. The
    configuration says so twice, through the sampler it built and through the program it built it
    from, and the determinization timings a size-uniform run writes are absent.
    """
    config = _config()
    program = config["search_program"]
    assert program["sampling"] == "depth-bounded"
    assert config["bo_sampler"]["type"] == "DepthBoundedRandomSampler"
    assert config["bo_sampler"]["counting"] is None
    assert [key for key in program if key.startswith("determinization")] == []
    assert program["repository"] == "RecognizableCNNrepository"
    assert (program["coupled_nonterminals"], program["coupled_rules"]) == (641, 3912)


def test_the_configuration_describes_the_tutorial_space_although_the_run_searched_the_vgg_cell():
    """A recorded defect, kept as it was recorded.

    The driver writes the search-space block from its module constants rather than from the
    repository it built, and the VGG cells build a repository from a different set of constants.
    So this block names the tutorial geometry while the run searched the VGG one, and one field of
    the same block gives it away: ``num_feature_dimensions`` is read off the repository and is 48,
    the closure of the nine VGG feature sizes under parallel sums of width 2, where the twelve
    tutorial sizes close to 87.

    The numbers the run reports do not depend on this block. Repeating the run from it does.
    """
    space = _config()["search_space"]
    assert space["channel_dimensions"] == [3, 6, 16]
    assert space["kernel_dimensions"] == [[5, 5], [3, 3], [2, 2]]
    assert space["max_lin_layer_dim"] is None
    assert 65536 not in space["linear_feature_dimensions"]

    assert space["num_feature_dimensions"] == 48
    assert "65536" in _config()["target"]


def test_the_configuration_names_a_learning_rate_that_no_evaluated_term_carries():
    """The second half of the same defect, and the one a reader trips over first.

    The block says the rate is 0.001, which is the tutorial constant. All 80 evaluated terms carry
    0.1, which is the rate of the VGG recipe and the rate the VGG branch of the driver puts into
    its repository.
    """
    rates = collections.Counter()
    for row in _rows():
        found = _LEARNING_RATE.findall(row["structure"])
        assert len(found) == 1, f"{row['phase']}[{row['index']}] carries {len(found)} optimizers"
        rates[found[0]] += 1

    assert dict(rates) == {"0.1": sum(PHASE_SIZES.values())}
    assert _config()["search_space"]["learning_rate_values"] == [0.001]


def test_every_recorded_term_renders_to_the_structure_written_beside_it():
    """The term pool and the CSV are two views of the same 80 evaluations.

    The CSV keeps the rendering and the pool keeps the term, and only the pool can be fed back
    into a kernel. Whether they belong together cannot be read off their names, so it is checked:
    record and row agree in phase, index, rendering and accuracy, in the order both were written.
    """
    header, records = read_term_pool(RECORDED / f"{RUN}_terms.pickle")
    assert header["format"] == FORMAT
    assert header["provenance"]["target_cell"] == "VGGM"

    rows = _rows()
    assert len(records) == len(rows)

    algebra = pretty_term_algebra()
    for record, row in zip(records, rows):
        where = f"{record.phase}[{record.index}]"
        assert (record.phase, record.index) == (row["phase"], int(row["index"])), where
        assert record.term.interpret(algebra) == row["structure"], where
        assert record.metrics["accuracy"] == pytest.approx(float(row["accuracy"]), abs=0.0), where


def test_the_four_side_artifacts_have_the_shape_the_budget_gives_them():
    """The trace, the generations, the surrogate and the diagnostics all count the same run.

    Each of the four answers a different question, and each of them has a row count the budget
    fixes. A row count alone only catches a file that stopped early, since two runs at the same
    budget have the same shape, so the value at the pick ties three of them together as well: the
    best of the last generation of a pass is the acquisition value the trace records for that
    pass, and it is the one written beside the evaluated term in the per-evaluation file.
    """
    config = _config()
    loop = config["bayesian_optimization"]
    passes, generations = loop["n_iterations"], loop["evo_generations"]

    with (RECORDED / f"{RUN}_trace.csv").open(newline="") as handle:
        trace = list(csv.DictReader(handle))
    assert [int(row["iteration"]) for row in trace] == list(range(passes))

    # One row per generation per pass, and a pass writes the population it started from as well,
    # so a pass of 35 generations is 36 rows.
    with (RECORDED / f"{RUN}_ea.csv").open(newline="") as handle:
        evolution = list(csv.DictReader(handle))
    assert len(evolution) == passes * (generations + 1)
    seen = collections.Counter(int(row["bo_iteration"]) for row in evolution)
    assert set(seen) == set(range(passes))
    assert set(seen.values()) == {generations + 1}

    # What the inner search ended on is what the loop picked with, in all three files. The three
    # were formatted by three writers and differ in the last bits of the decimal they printed, so
    # they are compared as the numbers they are and not as the text they carry.
    final = {int(row["bo_iteration"]): row for row in evolution
             if int(row["generation"]) == generations}
    picked = [row for row in _rows() if row["phase"] == "bo_step"]
    for step in range(passes):
        at_pick = pytest.approx(float(trace[step]["acquisition"]), rel=1e-12)
        assert float(final[step]["best"]) == at_pick, step
        assert float(picked[step]["acquisition_value"]) == at_pick, step

    # The surrogate is fitted once per pass, on the design plus what the passes before it added.
    with (RECORDED / f"{RUN}_surrogate.csv").open(newline="") as handle:
        surrogate = list(csv.DictReader(handle))
    assert [int(row["n_train"]) for row in surrogate] == [
        loop["n_pre_samples"] + step for step in range(passes)
    ]

    diagnostics = json.loads((RECORDED / f"{RUN}_diagnostics.json").read_text())
    assert diagnostics["read_this_with"] == "bayesian_optimization.diagnostics"
    assert diagnostics["trace"]["passes"] == passes
    assert len(diagnostics["trace"]["best_trace"]) == passes
    assert diagnostics["calibration"]["size"] == config["n_evaluations"]
    assert diagnostics["frontier"]["size"] == loop["population_size"]


def test_the_initial_design_was_taken_over_and_the_record_does_not_say_from_where():
    """The third defect of this record, and the one that decides whether the run repeats.

    The 20 rows of the initial design were written within 0.04 s of each other while their
    trainings sum to more than four hours, which no run writes by training them. They were taken
    over from an earlier attempt with ``--resume-from``, and neither the configuration nor the term
    pool has a field that says so or names the file they came from. A reader who wants the same
    design has to be told outside the record.

    The rows themselves are sound. The design is 20 evaluated terms with their measurements, and
    the two arms share it, which is what makes the comparison paired.
    """
    rows = _rows()
    design = [row for row in rows if row["phase"] == "pre_sample"]
    stamps = {float(row["timestamp"]) for row in design}
    trained = sum(float(row["train_seconds"]) for row in design)

    assert max(stamps) - min(stamps) < 1.0
    assert trained > 4 * 3600, "the design was cheap enough to have been written as it was trained"

    config = _config()
    assert [key for key in config if "resum" in key] == []
    assert [key for key in config["bayesian_optimization"] if "resum" in key] == []
    header, _ = read_term_pool(RECORDED / f"{RUN}_terms.pickle")
    assert [key for key in header["provenance"] if "resum" in key] == []


def test_the_anchor_is_the_architecture_the_verifier_refuses_to_train_without():
    """The reference run is the number every later comparison is read against.

    Its parameter count is checked against the constant the verifier stops on rather than written
    out again, so the two cannot drift apart unnoticed. Three trainings, because the spread
    between them is the only thing that says whether a later difference is a difference.
    """
    anchor = json.loads((RECORDED / "vgg11_bn_reference.json").read_text())
    assert anchor["n_params"] == EXPECTED_PARAMETERS
    assert anchor["expected_parameters"] == EXPECTED_PARAMETERS

    assert [run["seed"] for run in anchor["runs"]] == [0, 1, 2]
    assert {run["n_params"] for run in anchor["runs"]} == {EXPECTED_PARAMETERS}
    assert all(run["diverged"] is False for run in anchor["runs"])
    assert {run["epochs_completed"] for run in anchor["runs"]} == {anchor["epochs"]}

    # The anchor and the search have to have been trained alike, or the search is read against a
    # network that was given a different budget.
    config = _config()
    assert anchor["epochs"] == config["epochs_per_candidate"]
    assert anchor["batch_size"] == config["batch_size"]
    assert anchor["learning_rate"] == 0.1
