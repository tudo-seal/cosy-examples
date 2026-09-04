"""A run that died after its initial design must not have to retrain it.

A run that spends hours of accelerator time on its initial design and then dies on its first loop
pass leaves those results complete and on disk, and restarting it repeats every one of those
trainings to arrive at the same numbers.

What is under test is the safety of taking them over rather than the plumbing.  A design measured
under other conditions, or paired with terms it was not measured on, is worse than no design at
all, because it produces a surrogate conditioned on values nobody can attribute to a network.
"""

from __future__ import annotations

import pytest
from cosy.core.tree import Tree

from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_experiment_utils as utils
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_pool import TermPoolWriter

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
