"""The closed Bayesian optimization loop and the dataset it conditions on.

The loop is a closed procedure: an initial dataset from an initializer, then ``B`` passes of fit,
maximize, evaluate, and append, and a term of maximal observed value as the answer.  ``optimize()``
is that procedure.  The ask/tell methods below it are an engineering layer with no counterpart in
the algorithm.

Two clauses of the algorithm are asserted here rather than assumed.  Conditioning runs on the
**distinct** pairs of the dataset, which is what keeps a repeated evaluation from making the
noise-free Gram matrix singular, and the loop observes a *function*, so a value that is not finite
or a term that carries two different values is a contradiction rather than a datum.
"""
from __future__ import annotations

import math
from typing import Any

import pytest
from cosy.core.tree import Tree


def _objective_by_size(tree: Tree) -> float:
    """A cheap, deterministic objective: the number of nodes.

    Args:
        tree (Tree): The term to score.

    Returns:
        float: Its node count.
    """
    return 1.0 + sum(_objective_by_size(child) for child in tree.children)


# ---------------------------------------------------------------------------
# Conditioning on the distinct pairs
# ---------------------------------------------------------------------------


def _equal_but_separately_built() -> tuple[Tree, Tree]:
    """Two structurally equal terms that are not the same object.

    This is the shape a repeated pair actually arrives in.  The dataset a repeat can reach comes
    from outside: reloaded from a log, rebuilt by a caller, handed over by another process.  There
    the two copies are separate objects that compare and hash equal.  A test that passes one
    object twice cannot tell structural deduplication from deduplication by identity, and the
    latter would leave the duplicated row in the Gram matrix exactly where it matters.

    Returns:
        tuple[Tree, Tree]: The two terms.
    """
    return (
        Tree("A", (Tree("B"), Tree("C"))),
        Tree("A", (Tree("B"), Tree("C"))),
    )


def test_the_gp_is_conditioned_on_the_distinct_pairs(bo_factory, tree_corpus):
    """A repeated pair conditions the surrogate once, and the dataset still keeps both copies.

    The algorithm conditions the Gaussian process on the distinct pairs of the dataset, because an
    evaluation may repeat.  Under the rejection path this loop takes, a repeat can still arrive
    from outside through ``initialize(x0=..., y0=...)``, and a duplicated row makes the noise-free
    Gram matrix singular: two identical rows are linearly dependent, and only the jitter keeps the
    Cholesky factorization from failing outright.
    """
    bo = bo_factory()
    one, the_same = _equal_but_separately_built()
    assert one is not the_same and one == the_same
    bo.initialize(x0=[one, tree_corpus[1], the_same], y0=[1.0, 2.0, 1.0])

    bo.suggest()
    result = bo.finalize()

    assert result["gp_model"].X_train_.shape[0] == 2, (
        "the surrogate was conditioned on a repeated pair"
    )
    assert len(result["x"]) == 3, "the dataset itself keeps every evaluation"


def test_the_distinct_pairs_keep_the_order_of_first_appearance(bo_factory, tree_corpus):
    """Deduplication must not reorder the dataset.

    A seeded run is compared trajectory by trajectory.  The terms are handed over in an order that
    no sort would produce, so that this can tell first appearance from any ordering of its own.
    """
    bo = bo_factory()
    first, second, third = tree_corpus[2], tree_corpus[0], tree_corpus[1]
    assert [str(t) for t in (first, second, third)] != sorted(
        str(t) for t in (first, second, third)
    ), "an already sorted corpus would make this assertion vacuous"
    bo.initialize(x0=[first, second, first, third], y0=[1.0, 2.0, 1.0, 3.0])

    bo.suggest()
    conditioned = list(bo.finalize()["gp_model"].X_train_)

    assert conditioned == [first, second, third]


def test_one_term_with_two_values_is_refused(bo_factory):
    """The loop observes a function, so one term cannot carry two values.

    Distinct-pairs conditioning is stated for exact repetitions, an observation that repeats
    identically.  Two different values at one term are not a repetition: silently keeping one of
    them would pick a measurement for the caller, and keeping both is the singular matrix the
    clause exists to avoid.
    """
    bo = bo_factory()
    one, the_same = _equal_but_separately_built()

    with pytest.raises(ValueError, match="two different values"):
        bo.initialize(x0=[one, the_same], y0=[1.0, 2.0])


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_a_non_finite_observation_is_refused(bo_factory, tree_corpus, bad):
    """A failed evaluation stays visible and does not enter the dataset as a number.

    ``nan`` is the one that hides: it propagates through the fit, the posterior comes back as
    ``nan`` everywhere, and every acquisition value ties, which is a run that has stopped
    optimizing while still reporting suggestions.
    """
    bo = bo_factory()

    with pytest.raises(ValueError, match="finite"):
        bo.initialize(x0=tree_corpus[:2], y0=[1.0, bad])


def test_a_non_finite_value_is_refused_on_observe_too(bo_factory, tree_corpus):
    """The same rule on the tell side of the engineering layer."""
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    suggestion = bo.suggest()

    with pytest.raises(ValueError, match="finite"):
        bo.observe(suggestion.candidate, float("nan"))


def test_a_non_finite_value_is_refused_when_the_objective_produces_it(bo_factory, tree_corpus):
    """The path a real run takes: the value is not handed in, it is measured.

    A network that fails to train reaches the dataset through the objective, not through ``y0``,
    and that is the branch where a substitute value would do its damage unseen.
    """
    bo = bo_factory()

    with pytest.raises(ValueError, match="finite"):
        bo.initialize(objective=lambda t: float("nan"), x0=tree_corpus[:2])


@pytest.mark.parametrize(
    ("x_count", "y_count"), [(3, 2), (2, 3)]
)
def test_a_length_mismatch_is_refused_in_both_directions(bo_factory, tree_corpus, x_count, y_count):
    """Neither list may be truncated to the other: the pairing would be a guess.

    A ``y0`` longer than ``x0`` is the direction that used to pass: ``zip`` stops at the shorter
    one, and the values past the end simply vanish.
    """
    bo = bo_factory()

    with pytest.raises(ValueError, match=r"len\(x0\)"):
        bo.initialize(
            x0=tree_corpus[:x_count],
            y0=[float(i) for i in range(y_count)],
        )


# ---------------------------------------------------------------------------
# optimize(): the closed loop itself
# ---------------------------------------------------------------------------


def test_optimize_runs_budget_many_passes(bo_factory, tree_corpus):
    """``B`` passes, each appending exactly one evaluation to the dataset."""
    bo = bo_factory()
    result = bo.optimize(
        objective=_objective_by_size,
        budget=3,
        x0=tree_corpus[:3],
        y0=[1.0, 2.0, 0.5],
    )

    assert result["iterations"] == 3
    assert len(result["x"]) == 6
    assert len(result["y"]) == 6


def test_optimize_returns_a_term_of_maximal_observed_value(bo_factory, tree_corpus):
    """The terminal recommendation of the algorithm: the best point that was actually evaluated."""
    bo = bo_factory()
    result = bo.optimize(
        objective=_objective_by_size,
        budget=2,
        x0=tree_corpus[:3],
        y0=[1.0, 2.0, 0.5],
    )

    observed = list(zip(result["x"], result["y"], strict=True))
    assert result["best_y"] == max(value for _, value in observed)
    assert (result["best_tree"], result["best_y"]) in observed


def test_optimize_with_a_zero_budget_is_the_initial_design(bo_factory, tree_corpus):
    """A budget of zero runs no pass at all and answers from the initial dataset."""
    bo = bo_factory()
    result = bo.optimize(
        objective=_objective_by_size,
        budget=0,
        x0=tree_corpus[:3],
        y0=[1.0, 2.0, 0.5],
    )

    assert result["iterations"] == 0
    assert len(result["x"]) == 3
    assert result["best_y"] == 2.0


def test_optimize_refuses_a_negative_budget_before_it_evaluates_anything(bo_factory, tree_corpus):
    """A budget is a count of evaluations, and the check comes before the first of them.

    Checked after the initial design instead, a mistyped budget would cost the whole initial
    design in evaluations of the quality measure first, which on the search this framework is
    built for means training that many networks and then aborting.
    """
    bo = bo_factory()
    calls: list[Any] = []

    def counted(tree: Tree) -> float:
        """Count how often the objective was reached.

        Args:
            tree (Tree): The term to score.

        Returns:
            float: Its node count.
        """
        calls.append(tree)
        return _objective_by_size(tree)

    with pytest.raises(ValueError, match="budget"):
        bo.optimize(objective=counted, budget=-1, x0=tree_corpus[:3])

    assert calls == [], "the objective was called before the budget was checked"


@pytest.mark.parametrize(
    ("configuration", "run_arguments", "raised", "message"),
    [
        ({"acquisition_function": "ei"}, {}, ValueError, "ei"),
        (
            {"acquisition_function": "UpperConfidenceBound", "ucb_beta": 0.0},
            {},
            ValueError,
            "beta",
        ),
        (
            {"acquisition_function": "UpperConfidenceBound", "ucb_beta": math.nan},
            {},
            ValueError,
            "finite",
        ),
        (
            {"acquisition_function": "ProbabilityOfImprovement", "pi_margin": -1.0},
            {},
            ValueError,
            "margin",
        ),
        ({"optimizer": None}, {}, RuntimeError, "optimizer"),
        (
            {"acquisition_function": "UpperConfidenceBound"},
            {"acquisition_fitness_mode": "single"},
            ValueError,
            "batch",
        ),
        (
            {"acquisition_function": "UpperConfidenceBound"},
            {"acquisition_fitness_mode": "Batch"},
            ValueError,
            "batch",
        ),
        ({"acquisition_function": ["ExpectedImprovement"]}, {}, ValueError, "one of"),
    ],
)
def test_optimize_refuses_a_configuration_no_pass_could_use_before_it_evaluates_anything(
    bo_factory, tree_corpus, configuration, run_arguments, raised, message
):
    """Every pass reads the same acquisition, so a run can be refused before it draws a design.

    A misspelled acquisition, a parameter outside the range its acquisition admits, a missing
    evolutionary algorithm and a score that algorithm cannot be given one candidate at a time are
    all fixed before the first evaluation.  Found by the pass instead, each of them costs the whole
    initial design in evaluations of the quality measure first, which on the search this framework
    is built for means training that many networks and then aborting.  That is the reason the
    budget is checked here, and it does not stop at the budget.

    Two of the cases are near misses rather than plain typos.  ``"Batch"`` is not the batch mode,
    and the adapter scores every mode that is not ``"batch"`` one candidate at a time, so an upper
    confidence bound under it has to be refused exactly as under ``"single"``.  A name that is not
    even a string is refused for being an unknown name, not for being unhashable.
    """
    bo = bo_factory(**configuration)
    calls: list[Any] = []

    def counted(tree: Tree) -> float:
        """Count how often the objective was reached.

        Args:
            tree (Tree): The term to score.

        Returns:
            float: Its node count.
        """
        calls.append(tree)
        return _objective_by_size(tree)

    with pytest.raises(raised, match=message):
        bo.optimize(objective=counted, budget=2, x0=tree_corpus[:3], **run_arguments)

    assert calls == [], "the objective was called before the configuration was checked"


def test_a_zero_budget_runs_under_any_acquisition_setting(bo_factory, tree_corpus):
    """A run of no passes builds no acquisition, so an acquisition it never reads cannot fail it.

    The check above asks what the run will use, not what the object carries.  A budget of zero is
    the initial design on its own, and it answers from that design under any acquisition setting
    whatsoever, as it did before there was a check.
    """
    bo = bo_factory(acquisition_function="ei", ucb_beta=0.0, pi_margin=-1.0)

    result = bo.optimize(
        objective=_objective_by_size,
        budget=0,
        x0=tree_corpus[:3],
        y0=[1.0, 2.0, 0.5],
    )

    assert result["iterations"] == 0
    assert result["best_y"] == 2.0


def test_a_pass_still_refuses_an_acquisition_assigned_after_the_run_started(
    bo_factory, tree_corpus
):
    """The acquisition is a public attribute, so the pass keeps its own check.

    Read once at the start of the run, the configuration is only the configuration the run started
    with.  A caller who writes a new name onto the object afterwards reaches the pass without
    passing anything, and the pass has to be the one that says so.
    """
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    bo.acquisition_function = "ei"

    with pytest.raises(ValueError, match="ei"):
        bo.suggest()


def test_optimize_passes_gp_params_through(bo_factory, tree_corpus):
    """A parameter of the public signature that reaches nothing is a parameter that lies."""
    bo = bo_factory()
    result = bo.optimize(
        objective=_objective_by_size,
        budget=1,
        x0=tree_corpus[:3],
        y0=[1.0, 2.0, 0.5],
        gp_params={"normalize_y": False},
    )

    assert result["gp_model"].normalize_y is False


def test_optimize_passes_the_fitness_mode_through(bo_factory, tree_corpus):
    """The evolutionary algorithm has to hear which way it is meant to score its population."""
    seen: list[str] = []

    class ModeSpy:
        """A stand-in optimizer that records the mode it was called with."""

        def __init__(self, candidates: list[Tree]) -> None:
            """Remember the candidates to hand out.

            Args:
                candidates (list[Tree]): The candidates, one per call.
            """
            self._candidates = list(candidates)

        def evolutionary_best(
            self, query: Any, acquisition_objective: Any, fitness_function_mode: str = "batch"
        ) -> Tree:
            """Record the mode and return the next candidate.

            Args:
                query (Any): Ignored.
                acquisition_objective (Any): Ignored.
                fitness_function_mode (str): The mode under test. (Default value = 'batch')

            Returns:
                Tree: The next candidate.
            """
            seen.append(fitness_function_mode)
            return self._candidates.pop(0)

    bo = bo_factory()
    bo.optimizer = ModeSpy([Tree("fresh1"), Tree("fresh2")])
    bo.optimize(
        objective=_objective_by_size,
        budget=2,
        x0=tree_corpus[:3],
        y0=[1.0, 2.0, 0.5],
        acquisition_fitness_mode="single",
    )

    assert seen == ["single", "single"]


def test_optimize_uses_the_documented_jitter(bo_factory, tree_corpus):
    """One jitter for the whole class: the diagonal term that keeps the Gram matrix usable.

    A kernel matrix whose eigenvalues dip below zero at machine precision is repaired by adding a
    small constant to its diagonal.  The closed loop used to default to ``1e-10`` while
    ``initialize()`` defaulted to ``1e-6``, so the same configuration was conditioned four orders
    of magnitude apart depending on which entry point the caller took.
    """
    bo = bo_factory()
    result = bo.optimize(
        objective=_objective_by_size,
        budget=1,
        x0=tree_corpus[:3],
        y0=[1.0, 2.0, 0.5],
    )

    assert result["gp_model"].alpha == 1e-6


def test_the_ask_tell_layer_uses_the_same_jitter(bo_factory, tree_corpus):
    """Both entry points, or the two can drift apart again from the other side."""
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    bo.suggest()

    assert bo.finalize()["gp_model"].alpha == 1e-6


# ---------------------------------------------------------------------------
# No substitute values at the end of a run
# ---------------------------------------------------------------------------


def test_suggest_on_an_empty_dataset_raises(bo_factory):
    """A surrogate conditioned on nothing is the prior, and the loop has no initial design.

    Without this the sklearn regressor is handed a zero-row matrix and reports the shape rather
    than the cause.
    """
    bo = bo_factory()
    bo.initialize(x0=[], y0=[])

    with pytest.raises(RuntimeError, match="dataset is empty"):
        bo.suggest()


def test_finalize_without_observations_raises(bo_factory):
    """An empty dataset has no term of maximal observed value, and ``nan`` is not one.

    ``finalize()`` used to answer ``(None, nan)`` here, which reads downstream as a run that
    completed with a worthless optimum rather than as a run that evaluated nothing.
    """
    bo = bo_factory()
    bo.initialize(x0=[], y0=[])

    with pytest.raises(RuntimeError, match="No observations"):
        bo.finalize()


def test_finalize_reports_a_real_optimum_after_a_run(bo_factory, tree_corpus):
    """The counterpart: with observations the result carries no ``nan`` anywhere."""
    bo = bo_factory()
    result = bo.optimize(
        objective=_objective_by_size,
        budget=1,
        x0=tree_corpus[:2],
        y0=[1.0, 2.0],
    )

    assert result["best_tree"] is not None
    assert math.isfinite(result["best_y"])
