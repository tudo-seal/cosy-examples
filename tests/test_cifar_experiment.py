"""What the CIFAR-10 experiment settles before any search runs, and what one run of it writes.

The module under test is a driver. It fixes a search space, resolves a target, and hands both to
the run layer. A real run cannot be a test: one candidate is a training run, the dataset is not in
this repository, and the experiment's own space takes minutes to build. Most tests here therefore
check what the driver decides before the first training.

Two decisions are worth a test. The first is the search space itself: with the cap on linear
feature sizes below the 3072 features of a raw CIFAR-10 image, no linear layer can consume the
image, so every candidate the head target admits has to begin with a convolution or a pooling
layer. The second is the wiring of the command line, where a mistake does not fail but runs a
different experiment than the one that was asked for.

The rest run the whole command line, on forty random images and the tutorial network's geometry,
where the program builds in about a second: the files a run writes, a paired run, a resumed design,
and the names a run takes and gives back.
"""

from __future__ import annotations

import csv
import inspect
import json

import pytest
import torch
from cosy.core import Synthesizer

from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_cifar_experiment as experiment
from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_experiment_utils as utils
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_experiment_utils import (
    CIFAR_SCHEMA,
    POSITION_TARGETS,
    TARGET_LENGTHS,
    describe_repository,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_reference_architectures import (
    cifar10_tutorial_repo,
    cifar10_vgg11_bn_repo,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_targets import make_cifar_head_target
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_algebras import pretty_term_algebra
from bayesian_optimization.runs import Objective, TermPoolWriter, read_term_pool, search_program


def _cnn_share(trees):
    pretty = [t.interpret(pretty_term_algebra()) for t in trees]
    return sum(1 for p in pretty if "Conv2d(" in p or "MaxPool2d(" in p), len(pretty)


@pytest.mark.slow
def test_cifar_head_target_forces_convolutional_front_end():
    """With the cap below the 3072-feature input, every candidate has to start convolutionally.

    This is the one property of the search space that no smaller repository shows, because it is
    the interaction of the cap with the feature list of this experiment, and it takes the full
    label set of the experiment to see it. That is what makes the test slow.
    """
    repo = CNNrepository(
        linear_feature_dimensions=experiment.LINEAR_FEATURE_DIMENSIONS,
        constant_values=experiment.CONSTANT_VALUES,
        learning_rate_values=experiment.LEARNING_RATE_VALUES,
        n_epoch_values=[20],
        channel_dimensions=experiment.CHANNEL_DIMENSIONS,
        height_width_dimensions=experiment.HEIGHT_WIDTH_DIMENSIONS,
        kernel_dimensions=experiment.KERNEL_DIMENSIONS,
        stride_values=experiment.STRIDE_VALUES,
        padding_values=experiment.PADDING_VALUES,
        max_parallel_width=experiment.MAX_PARALLEL_WIDTH,
        max_lin_layer_dim=1600,
    )
    target = make_cifar_head_target(epochs=20)
    search_space = Synthesizer(repo.specification(), {}).construct_solution_space(target).prune()
    trees = list(search_space.enumerate_trees(target, max_count=30))
    assert trees, "the CIFAR head target should synthesize"
    cnn_count, total = _cnn_share(trees)
    assert cnn_count == total, (
        f"expected every candidate to be convolutional, got {cnn_count}/{total}")


def test_the_command_line_offers_exactly_the_targets_the_experiment_can_resolve():
    """A name the parser accepts and the resolver does not is a run that dies after the parse."""
    target_action = next(action for action in experiment.build_parser()._actions
                         if action.dest == "target")
    assert set(target_action.choices) == set(TARGET_LENGTHS) | set(POSITION_TARGETS)


def test_every_named_vgg_cell_carries_the_geometry_its_positions_need():
    """A cell is a position list together with the kernels and paddings it was built for.

    Pairing a position list with a geometry that does not fit it produces an uninhabited target,
    which is an empty search space and not an error, so nothing later in the run says what went
    wrong.
    """
    for name, cell in experiment.VGG_CELLS.items():
        assert set(cell) == {"positions", "kernels", "paddings"}, name
        assert cell["positions"], name
        assert cell["kernels"], name
        assert cell["paddings"], name


def test_the_command_line_reaches_the_run_with_the_arguments_it_was_given(monkeypatch, tmp_path):
    """``main`` parses and forwards, and a swapped argument here runs a different experiment."""
    seen = {}

    def record(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "result", 0.0

    monkeypatch.setattr(experiment, "run_experiment", record)
    csv_path = str(tmp_path / "run.csv")
    experiment.main(["--target", "VGGM", "--epochs", "2", "--n-pre-samples", "3",
                     "--n-iterations", "4", "--population-size", "5", "--evo-generations", "6",
                     "--seed", "7", "--repeats", "2", "--csv-path", csv_path])

    assert seen["args"] == (3, 4, 5, 6, csv_path)
    assert seen["kwargs"]["target_cell"] == "VGGM"
    assert seen["kwargs"]["epochs"] == 2
    assert seen["kwargs"]["seed"] == 7
    assert seen["kwargs"]["repeats"] == 2


def test_resume_from_needs_no_baseline(monkeypatch, tmp_path):
    """The design is the loop's first phase on both paths, so a resume needs no random arm.

    The parser used to refuse ``--resume-from`` without ``--baseline``, because only the paired
    path drew its design before evaluating it.
    """
    seen = {}

    def record(*args, **kwargs):
        seen["kwargs"] = kwargs
        return "result", 0.0

    monkeypatch.setattr(experiment, "run_experiment", record)
    pool = str(tmp_path / "terms.pickle")
    experiment.main(["--resume-from", pool, "--csv-path", str(tmp_path / "run.csv")])

    assert seen["kwargs"]["resume_from"] == pool
    assert seen["kwargs"]["baseline"] is False


def test_load_cifar10_does_not_fetch_anything_by_default(tmp_path):
    """A missing dataset raises where the run starts, and leaves the directory as it found it.

    torchvision's own default is not to download either. This one had been turned around, which
    put a fetch of the whole archive on the path of every caller that did not think about it,
    including two that run for hours before they read the data.
    """
    with pytest.raises(RuntimeError):
        experiment.load_cifar10(str(tmp_path))

    assert list(tmp_path.iterdir()) == [], "a failed load left something behind"


def test_a_fetch_of_the_dataset_has_to_be_asked_for(monkeypatch, tmp_path):
    """The command line carries the flag through, and it is off unless it is given."""
    seen = {}
    monkeypatch.setattr(experiment, "run_experiment",
                        lambda *a, **k: (seen.update(k), ("result", 0.0))[1])

    experiment.main(["--csv-path", str(tmp_path / "a.csv")])
    assert seen["download"] is False

    experiment.main(["--csv-path", str(tmp_path / "b.csv"), "--download"])
    assert seen["download"] is True


def test_the_record_of_a_search_space_is_read_off_the_repository_that_builds_it():
    """The block a run records has to describe the repository the run searched, not a constant.

    The recorded VGGM run is the case: it built the VGG cell and wrote the dimensions of the
    non-VGG branch, which reads the module constants, beside a correct count of its own feature
    widths. Nine of the eleven fields came from those constants, one from the command line and one
    from the object. Reading all of them off the object is what keeps the halves of the block from
    describing different searches. The two repositories here are the two shipped reference
    architectures, which differ in more fields than the record's own pair does.
    """
    tutorial = cifar10_tutorial_repo()
    vgg = cifar10_vgg11_bn_repo()

    described = describe_repository(vgg)
    assert described["kernel_dimensions"] == [[3, 3]]
    assert described["learning_rate_values"] == [0.1]
    assert described["max_lin_layer_dim"] == 512
    assert described["num_feature_dimensions"] == 48

    # Every field the two repositories differ in has to differ in their records too, or the
    # record cannot tell one search from the other. num_feature_dimensions is the exception by
    # construction: it is a count over a field rather than the field.
    other = describe_repository(tutorial)
    for key in described:
        if key == "num_feature_dimensions":
            continue
        assert (described[key] != other[key]) == (getattr(vgg, key) != getattr(tutorial, key)), key
    assert described["num_feature_dimensions"] != other["num_feature_dimensions"]

    # And json.dump has to survive it, since that is where the block goes.
    assert json.loads(json.dumps(described)) == described


def test_the_metadata_of_a_run_carries_the_source_of_its_initial_design():
    """A resumed design and a trained one look alike in a record that does not say which it was.

    The field is written whether or not a run resumed, so its absence marks a record from before
    it was written rather than a run that trained its own design. This reads the source of the
    block rather than a record, because building one takes a dataset and a search.
    """
    source = inspect.getsource(experiment.run_experiment)
    assert '"resumed_from": resume_from' in source


# --- One run of the command line, on the tutorial network's geometry --------------------------------

#: The geometry of ``cifar10_tutorial_repo``, which the driver builds its repository from in place of
#: its own: the program of the target below builds in about a second, the experiment's own in minutes.
TUTORIAL_GEOMETRY = {
    "LINEAR_FEATURE_DIMENSIONS": [3072, 4704, 1176, 1600, 400, 120, 84, 10],
    "CHANNEL_DIMENSIONS": [3, 6, 16],
    "HEIGHT_WIDTH_DIMENSIONS": [(32, 32), (28, 28), (14, 14), (10, 10)],
    "KERNEL_DIMENSIONS": [(5, 5), (2, 2)],
    "STRIDE_VALUES": [1],
    "PADDING_VALUES": [0],
    "POOLING_KERNEL_DIMENSIONS": [(2, 2)],
}

#: Everything a run writes beside its CSV, the diagnostics' two among them.
RUN_FILES = ["run.csv", "run_config.json", "run_diagnostics.json", "run_ea.csv",
             "run_surrogate.csv", "run_terms.pickle", "run_trace.csv"]


def _small_cifar(monkeypatch):
    """The driver over the tutorial geometry and forty random images; returns the data loads made."""
    for name, value in TUTORIAL_GEOMETRY.items():
        monkeypatch.setattr(experiment, name, value)
    loads = []

    def images(data_dir, download=False):
        loads.append(data_dir)
        generator = torch.Generator().manual_seed(0)

        def split(size):
            return torch.utils.data.TensorDataset(
                torch.randn(size, 3, 32, 32, generator=generator),
                torch.randint(0, 10, (size,), generator=generator),
            )

        return split(40), split(8)

    monkeypatch.setattr(experiment, "load_cifar10", images)
    return loads


@pytest.fixture
def small_cifar(monkeypatch):
    """A run of the command line in seconds; the diagnostics, which refit the surrogate after the
    run, are left to the one test that reads their files and to the run layer's own tests."""
    monkeypatch.setattr(experiment, "write_run_diagnostics", lambda *args: None)
    return _small_cifar(monkeypatch)


def _run(directory, *arguments, name="run"):
    """Three design terms and one pass of one epoch each; a later argument overrides."""
    return experiment.main([
        "--target", "TUT1", "--size-bound", "400", "--epochs", "1", "--n-pre-samples", "3",
        "--n-iterations", "1", "--population-size", "4", "--evo-generations", "1", "--seed", "0",
        "--csv-path", str(directory / f"{name}.csv"), *arguments,
    ])


def _names(directory):
    return sorted(path.name for path in directory.iterdir())


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _config(path):
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def design_pool(tmp_path_factory):
    """The term pool of a run of the design alone, measured once for the tests that resume it."""
    directory = tmp_path_factory.mktemp("design")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(experiment, "write_run_diagnostics", lambda *args: None)
        _small_cifar(monkeypatch)
        _run(directory, "--n-iterations", "0", name="first")
    return directory / "first_terms.pickle"


@pytest.fixture
def trainings(monkeypatch):
    """Counts the trainings a run makes."""
    made = []
    train = utils._train_candidate_once

    def counted(*args, **kwargs):
        made.append(args[0])
        return train(*args, **kwargs)

    monkeypatch.setattr(utils, "_train_candidate_once", counted)
    return made


def test_a_run_writes_its_evaluations_through_the_run_layer(monkeypatch, tmp_path):
    _small_cifar(monkeypatch)
    result, seconds = _run(tmp_path)

    assert _names(tmp_path) == RUN_FILES
    rows = _rows(tmp_path / "run.csv")
    assert list(rows[0]) == CIFAR_SCHEMA.header, "the CIFAR layout of a row"
    assert [row["phase"] for row in rows] == ["pre_sample"] * 3 + ["bo_step"]
    config = _config(tmp_path / "run_config.json")
    assert (config["run_kind"], config["design_source"]) == ("bayesian_optimization", "drawn")
    assert (config["evaluated_here"], config["taken_over"]) == (4, 0)
    assert config["seconds"] == seconds
    assert config["resumed_from"] is None
    header, records = read_term_pool(str(tmp_path / "run_terms.pickle"))
    assert header["design_origin"] == "drawn"
    # the loop maximized the validation accuracy, and each record keeps what it was handed
    assert [record.loop_value for record in records] == [float(row["accuracy"]) for row in rows]
    assert list(result["y"]) == [record.loop_value for record in records]


def test_a_paired_run_writes_its_random_arm_beside_the_loop(small_cifar, tmp_path, monkeypatch):
    streams = []
    prefix = search_program.distinct_prefix

    def recorded(*args, **kwargs):
        terms, repeats = prefix(*args, **kwargs)
        streams.append(terms)
        return terms, repeats

    monkeypatch.setattr(search_program, "distinct_prefix", recorded)
    _run(tmp_path, "--baseline")

    assert [name for name in _names(tmp_path) if name.startswith("run_random")] == [
        "run_random.csv", "run_random_config.json", "run_random_terms.pickle"]
    loop_rows, arm_rows = _rows(tmp_path / "run.csv"), _rows(tmp_path / "run_random.csv")
    assert [row["phase"] for row in loop_rows] == ["pre_sample"] * 3 + ["bo_step"]
    assert [row["phase"] for row in arm_rows] == ["pre_sample"] * 3 + ["random_sample"]
    # one design, evaluated once: the arm took the loop's records over
    assert [row["structure"] for row in arm_rows[:3]] == [row["structure"] for row in loop_rows[:3]]
    _header, arm_records = read_term_pool(str(tmp_path / "run_random_terms.pickle"))
    assert [record.taken_over for record in arm_records] == [True] * 3 + [False]
    loop_config = _config(tmp_path / "run_config.json")
    arm_config = _config(tmp_path / "run_random_config.json")
    assert loop_config["paired_with"] == "run_random.csv"
    assert (arm_config["paired_with"], arm_config["design_taken_over_from"]) == ("run.csv", "run.csv")
    drawn = loop_config["design_drawn_up_front"]
    assert (drawn["design"], drawn["then"]) == (3, 1)
    assert "design_drawn_up_front" not in arm_config, "the draw is the loop's"
    assert (arm_config["run_kind"], arm_config["taken_over"]) == ("random_search", 3)
    # the arm's pass is the stream's next term past the design, drawn up front with it
    (stream,) = streams
    assert [record.term for record in arm_records] == stream


def _as_the_old_driver_kept_it(pool, old):
    """The pool rewritten as the driver before the run layer wrote one: no loop value kept, and no
    design origin in its header."""
    header, records = read_term_pool(str(pool))
    with TermPoolWriter(str(old), provenance=header["provenance"]) as writer:
        for record in records:
            writer.write(record.phase, record.index, record.term, record.metrics)
    return records


@pytest.mark.parametrize(("objective", "reading"), [
    ("accuracy", lambda metrics: metrics["accuracy"]),
    ("loss", lambda metrics: -metrics["objective_value"]),
])
def test_a_design_the_old_driver_measured_resumes_under_the_run_s_objective(
    small_cifar, design_pool, tmp_path, trainings, objective, reading
):
    """Its records kept no value, so the run's objective reads their metrics, as it always did."""
    records = _as_the_old_driver_kept_it(design_pool, tmp_path / "old_terms.pickle")

    result, _seconds = _run(tmp_path, "--resume-from", str(tmp_path / "old_terms.pickle"),
                            "--objective", objective, name="resumed")

    assert list(result["y"][:3]) == [reading(record.metrics) for record in records]
    assert len(trainings) == 1, "the pass is trained, the design is not"
    assert _config(tmp_path / "resumed_config.json")["resumed_from"] == str(tmp_path / "old_terms.pickle")


def test_a_run_the_run_layer_refuses_leaves_nothing_and_its_retry_runs(
    small_cifar, design_pool, tmp_path
):
    with pytest.raises(ValueError, match="n_design is 2"):
        _run(tmp_path, "--resume-from", str(design_pool), "--n-pre-samples", "2", name="retry")
    assert _names(tmp_path) == [], "the refused run left files behind"

    _run(tmp_path, "--resume-from", str(design_pool), "--n-iterations", "0", name="retry")
    assert (tmp_path / "retry.csv").is_file()


@pytest.mark.parametrize("changed", [("--batch-size", "64"), ("--repeats", "2")])
def test_a_design_measured_under_other_conditions_is_refused_before_anything_is_written(
    small_cifar, design_pool, tmp_path, changed
):
    with pytest.raises(ValueError, match="different configuration"):
        _run(tmp_path, "--resume-from", str(design_pool), *changed)
    assert _names(tmp_path) == []


@pytest.mark.parametrize("taken", ["run_config.json", "run_random.csv"])
def test_a_start_under_a_taken_name_is_refused_before_the_data_is_loaded(small_cifar, tmp_path, taken):
    (tmp_path / taken).write_text("another run's\n")

    with pytest.raises(FileExistsError, match=taken):
        _run(tmp_path, "--baseline")

    assert small_cifar == [], "the data was loaded for a run that cannot start"
    assert _names(tmp_path) == [taken]
    assert (tmp_path / taken).read_text() == "another run's\n"


def test_a_training_that_gives_out_keeps_what_the_run_wrote_and_its_names(
    small_cifar, tmp_path, monkeypatch
):
    def gives_out(*args, **kwargs):
        raise RuntimeError("the first training gave out")

    monkeypatch.setattr(utils, "_train_candidate_once", gives_out)
    with pytest.raises(RuntimeError, match="gave out"):
        _run(tmp_path)

    assert {"run.csv", "run_config.json", "run_terms.pickle"} <= set(_names(tmp_path))


def test_the_objective_a_run_maximizes_is_the_schema_s():
    assert utils.cifar_schema("accuracy").objective == Objective("accuracy")
    assert utils.cifar_schema("loss").objective == Objective("objective_value", greater_is_better=False)
    assert utils.cifar_schema("loss").columns == CIFAR_SCHEMA.columns, "one layout, either objective"
    with pytest.raises(ValueError, match="'accuracy' or 'loss'"):
        utils.cifar_schema("f1")


def test_the_line_of_an_evaluation_shows_what_the_driver_always_showed():
    metrics = {"accuracy": 0.5, "n_params": 10, "train_seconds": 1.26, "objective_value": 2.0}
    assert CIFAR_SCHEMA.live_line(metrics) == "accuracy=0.5000 params=10 train=1.3s"
    assert CIFAR_SCHEMA.live_line({"n_params": 10}) == "params=10", "what is not reported is left out"
