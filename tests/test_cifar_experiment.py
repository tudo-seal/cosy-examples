"""What the CIFAR-10 experiment settles before any search runs.

The module under test is a driver. It fixes a search space, resolves a target, and hands both to
the optimization loop, and none of that can be tested by running it: one candidate is a training
run, and the dataset is not in this repository. What can be tested is everything the driver decides
before the first training, and that is what these tests do.

Two decisions are worth a test. The first is the search space itself: with the cap on linear
feature sizes below the 3072 features of a raw CIFAR-10 image, no linear layer can consume the
image, so every candidate the head target admits has to begin with a convolution or a pooling
layer. The second is the wiring of the command line, where a mistake does not fail but runs a
different experiment than the one that was asked for.

No test here reads a dataset, and none of them calls ``run_experiment``.
"""

from __future__ import annotations

import inspect
import json

import pytest
from cosy.core import Synthesizer

from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_cifar_experiment as experiment
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_experiment_utils import (
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


def test_resume_from_without_baseline_is_rejected_before_the_search_is_built(monkeypatch, tmp_path):
    """The same condition is checked inside the run, but only after the space is determinized.

    On a large cell that is a long wait for an argument error, so the parser rejects it first.
    """
    called = []
    monkeypatch.setattr(experiment, "run_experiment", lambda *a, **k: called.append(k))

    with pytest.raises(SystemExit):
        experiment.main(["--resume-from", str(tmp_path / "terms.pickle"),
                         "--csv-path", str(tmp_path / "run.csv")])

    assert called == [], "the run started despite the rejected argument combination"


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
