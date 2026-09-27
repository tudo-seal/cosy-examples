"""A run that died after its initial design must not have to retrain it.

A run that spends hours of accelerator time on its initial design and then dies on its first loop
pass leaves those results complete and on disk, and restarting it repeats every one of those
trainings to arrive at the same numbers.

What is under test is the safety of taking them over rather than the plumbing.  A design measured
under other conditions, or paired with terms it was not measured on, is worse than no design at
all, because it produces a surrogate conditioned on values nobody can attribute to a network.
"""

from __future__ import annotations

import random

import pytest
from cosy.core.tree import Tree
from cosy.search import SizeUniformSampler

from bayesian_optimization.bo import BayesianOptimization
from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_experiment_utils as utils
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_pool import TermPoolWriter
from tests.spaces import LIST, list_space

PROVENANCE = {"target_cell": "VGGM", "epochs_per_candidate": 50, "training_repeats": 3}


def _pool(tmp_path, terms, provenance=PROVENANCE, phase="pre_sample"):
    """Write a term pool the way a run writes it."""
    path = tmp_path / "run_terms.pickle"
    with TermPoolWriter(str(path), provenance=dict(provenance)) as writer:
        for index, tree in enumerate(terms):
            writer.write(phase, index, tree, {
                "objective_value": 1.0, "accuracy": 0.5 + 0.1 * index, "n_params": 10,
                "train_seconds": 900.0, "diverged": False, "epochs_completed": 50,
            })
    return str(path)


def test_the_design_comes_back_with_its_measurements(tmp_path):
    terms = [Tree("a"), Tree("b"), Tree("c")]
    design = utils.load_initial_design(_pool(tmp_path, terms), expected=PROVENANCE)

    assert [tree for tree, _metrics in design] == terms
    assert [metrics["accuracy"] for _tree, metrics in design] == [0.5, 0.6, 0.7]


def test_a_design_measured_under_other_conditions_is_refused(tmp_path):
    """The one that matters: the values would be attached to candidates they do not describe."""
    path = _pool(tmp_path, [Tree("a")], provenance={**PROVENANCE, "epochs_per_candidate": 10})

    with pytest.raises(ValueError, match="different configuration"):
        utils.load_initial_design(path, expected=PROVENANCE)


def test_each_checked_field_is_actually_checked(tmp_path):
    """A check that passes whatever it is given is not a check."""
    for field, other in (("target_cell", "TUT1"), ("epochs_per_candidate", 2),
                         ("training_repeats", 1)):
        directory = tmp_path / field
        directory.mkdir()
        path = _pool(directory, [Tree("a")], provenance={**PROVENANCE, field: other})
        with pytest.raises(ValueError, match="different configuration"):
            utils.load_initial_design(path, expected=PROVENANCE)


def test_a_pool_without_the_requested_phase_says_so(tmp_path):
    path = _pool(tmp_path, [Tree("a")], phase="bo_step")

    with pytest.raises(ValueError, match="no 'pre_sample' records"):
        utils.load_initial_design(path, expected=PROVENANCE)


def test_a_design_of_the_wrong_length_is_refused():
    design = [(Tree("a"), {}), (Tree("b"), {})]

    with pytest.raises(ValueError, match="the size of the initial design must match"):
        utils._check_resumed_design(design, [Tree("a")])


def test_a_design_whose_terms_moved_is_refused():
    """The failure this guard exists for: value i would be paired with a network it is not from."""
    design = [(Tree("a"), {}), (Tree("b"), {})]

    with pytest.raises(ValueError, match="not the term this run drew"):
        utils._check_resumed_design(design, [Tree("a"), Tree("different")])


def test_a_matching_design_passes():
    design = [(Tree("a"), {}), (Tree("b"), {})]
    utils._check_resumed_design(design, [Tree("a"), Tree("b")])


def test_a_resumed_run_does_not_call_the_objective(monkeypatch, bo_factory, tmp_path):
    """The point of the whole thing: no network is trained for a value that already exists."""
    terms = [Tree("a"), Tree("b"), Tree("c")]
    design = utils.load_initial_design(_pool(tmp_path, terms), expected=PROVENANCE)

    monkeypatch.setattr(utils, "_draw_prefix", lambda optimizer, count: (list(terms), 0))
    trained = []

    def f_obj(tree):
        trained.append(tree)
        return 0.0

    metrics_by_tree = {}
    utils.run_ask_tell_search(
        optimizer=bo_factory(),
        f_obj=f_obj,
        metrics_by_tree=metrics_by_tree,
        as_reported=lambda v: v,
        objective="accuracy",
        greater_is_better=True,
        n_pre_samples=len(terms),
        n_iterations=0,
        csv_path=str(tmp_path / "resumed.csv"),
        pretty_algebra=dict,
        baseline=True,
        verbose=False,
        resume_design=design,
    )

    assert trained == [], "a resumed design must not retrain anything"
    # And the metrics are where everything downstream looks for them.
    assert {tree: metrics["accuracy"] for tree, metrics in metrics_by_tree.items()} == {
        Tree("a"): 0.5, Tree("b"): 0.6, Tree("c"): 0.7
    }


def test_a_resumed_value_is_read_the_way_the_caller_names(monkeypatch, bo_factory, tmp_path):
    """A resumed record seeds the surrogate with what the caller's objective would have returned.

    The loop cannot know how a caller's objective maps a metrics record to the value it
    maximizes: the CIFAR example maximizes ``accuracy``, another driver maximizes its own
    ``objective_value`` with ``greater_is_better=True``.  Left to the example's convention, a
    resumed design of that driver would condition the surrogate on validation accuracies while
    every row of the run says otherwise.  The pool here carries accuracies 0.5, 0.6, 0.7 and the
    objective 1.0 throughout, so the two readings cannot be confused.
    """
    terms = [Tree("a"), Tree("b"), Tree("c")]
    design = utils.load_initial_design(_pool(tmp_path, terms), expected=PROVENANCE)
    monkeypatch.setattr(utils, "_draw_prefix", lambda optimizer, count: (list(terms), 0))

    optimizer = bo_factory()
    utils.run_ask_tell_search(
        optimizer=optimizer,
        f_obj=lambda tree: 0.0,
        metrics_by_tree={},
        as_reported=lambda v: v,
        objective="objective_value",
        greater_is_better=True,
        n_pre_samples=len(terms),
        n_iterations=0,
        csv_path=str(tmp_path / "resumed_named.csv"),
        pretty_algebra=dict,
        baseline=True,
        verbose=False,
        resume_design=design,
        resumed_value=lambda metrics: metrics["objective_value"],
    )
    assert optimizer.get_state_snapshot()["y_list"] == [1.0, 1.0, 1.0]

    # the example's convention stays the default: greater is better reads the accuracy
    by_convention = bo_factory()
    utils.run_ask_tell_search(
        optimizer=by_convention,
        f_obj=lambda tree: 0.0,
        metrics_by_tree={},
        as_reported=lambda v: v,
        objective="accuracy",
        greater_is_better=True,
        n_pre_samples=len(terms),
        n_iterations=0,
        csv_path=str(tmp_path / "resumed_default.csv"),
        pretty_algebra=dict,
        baseline=True,
        verbose=False,
        resume_design=design,
    )
    assert by_convention.get_state_snapshot()["y_list"] == [0.5, 0.6, 0.7]


def _list_loop(seed):
    """A loop over the lists over ``{0, 1, 2}``: a real space, so the loop draws its own design."""
    return BayesianOptimization(
        search_space=list_space(),
        request=LIST,
        sampler=SizeUniformSampler(6, random.Random(seed)),
        seed=seed,
    )


def _design_of(seed, size):
    """The design a fresh loop of this seed draws, read off a twin through the closed path.

    The closed path rather than the design phase, so that this helper does not depend on the
    behaviour under test; ``tests/unit/test_design_phase.py`` holds that the two draw alike.
    """
    twin = _list_loop(seed)
    twin.initialize(objective=lambda tree: 0.0, initial_size=size)
    return list(twin.get_state_snapshot()["x_list"])


def _resume(optimizer, design, tmp_path, name, f_obj):
    utils.run_ask_tell_search(
        optimizer=optimizer,
        f_obj=f_obj,
        metrics_by_tree={},
        as_reported=lambda v: v,
        objective="accuracy",
        greater_is_better=True,
        n_pre_samples=len(design),
        n_iterations=0,
        csv_path=str(tmp_path / name),
        pretty_algebra=dict,
        baseline=False,
        verbose=False,
        resume_design=design,
    )


def test_a_design_resumes_without_a_random_arm(tmp_path):
    """Resume used to exist on the paired path only; the loop's own design is taken over as well."""
    terms = _design_of(seed=0, size=3)
    design = utils.load_initial_design(_pool(tmp_path, terms), expected=PROVENANCE)
    trained = []

    def f_obj(tree):
        trained.append(tree)
        return 0.0

    optimizer = _list_loop(0)
    _resume(optimizer, design, tmp_path, "resumed_unpaired.csv", f_obj)

    assert trained == [], "a resumed design must not retrain anything"
    snapshot = optimizer.get_state_snapshot()
    assert snapshot["x_list"] == terms
    assert snapshot["y_list"] == [0.5, 0.6, 0.7]


def test_without_a_random_arm_a_design_from_another_stream_is_still_refused(tmp_path):
    """The guard of the paired path holds on the unpaired one: values stay with their terms."""
    foreign = _design_of(seed=1, size=3)
    assert foreign != _design_of(seed=0, size=3), "the two seeds must draw different designs"
    design = utils.load_initial_design(_pool(tmp_path, foreign), expected=PROVENANCE)

    with pytest.raises(ValueError, match="not the term this run drew"):
        _resume(_list_loop(0), design, tmp_path, "refused.csv", lambda tree: 0.0)
