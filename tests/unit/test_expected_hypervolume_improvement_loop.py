"""The loop maximizes the expected hypervolume improvement, measured against its own dataset.

``acquisition_function="ExpectedHypervolumeImprovement"`` builds the score afresh for every pass
from the surrogate it fitted, the exact cost ``ehvi_cost`` and the reference point
``ehvi_reference``, against the pairs of value and cost the loop has observed, read where the
incumbent is read: the observed values, and under repeated measurements the posterior means.  The
cost and the reference point are refused before the design is paid.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter
from typing import Any

import numpy as np
import pytest
from cosy.core import Synthesizer
from cosy.search import SizeUniformSampler
from sklearn.gaussian_process.kernels import WhiteKernel

from bayesian_optimization import (
    BayesianOptimization,
    ExpectedHypervolumeImprovement,
    read_frontier,
)
from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel
from tests.integration.test_real_ea_loop import (
    _MAX_SIZE,
    _TARGET,
    _make_ea,
    _objective,
    _repository,
)

# Every term of the space has size 5 to 7 and a value of a few units, so the reference point bounds
# every observation from below and from above.
REFERENCE = (0.0, 8.0)


def _cost(term: Any) -> float:
    return float(term.size)


def _loop(**kwargs: Any) -> BayesianOptimization:
    space = Synthesizer(_repository(), {}).construct_solution_space(_TARGET).prune()
    settings: dict[str, Any] = {
        "acquisition_function": "ExpectedHypervolumeImprovement",
        "ehvi_cost": _cost,
        "ehvi_reference": REFERENCE,
    }
    settings.update(kwargs)
    return BayesianOptimization(
        space,
        _TARGET,
        optimizer=_make_ea(space, 7),
        seed=7,
        sampler=SizeUniformSampler(_MAX_SIZE, random.Random(7)),
        n_restarts_kernel_optimizer=0,
        **settings,
    )


def _design(bo: BayesianOptimization, size: int = 4) -> None:
    bo.initialize(initial_size=size)
    for _ in range(size):
        suggestion = bo.suggest()
        bo.observe(suggestion.candidate, _objective(suggestion.candidate))


def _paying(paid: list[Any]) -> Any:
    def objective(term: Any) -> float:
        paid.append(term)
        return _objective(term)

    return objective


def test_each_pass_measures_against_the_pairs_it_has_observed():
    """Every pass builds its own score, against every observed pair, the known points beneath."""
    bo = _loop()
    _design(bo)
    built = []
    for _step in range(3):
        snapshot = bo.get_state_snapshot()
        observed = [(float(y), _cost(x)) for x, y in zip(snapshot["x_list"], snapshot["y_list"])]
        suggestion = bo.suggest(record_population=True)
        assert bo.last_acquisition_run is not None
        acquisition = bo.last_acquisition_run.acquisition
        assert isinstance(acquisition, ExpectedHypervolumeImprovement)
        assert acquisition.front == observed
        assert acquisition.reference == REFERENCE
        assert acquisition.known_points == set(snapshot["x_list"])
        assert acquisition.gp is bo.surrogate
        built.append(acquisition)
        bo.observe(suggestion.candidate, _objective(suggestion.candidate))
    assert len({id(acquisition) for acquisition in built}) == 3, "a new score each pass"
    assert len(built[-1].front) == 6, "the control: the passes' picks joined the front"


def test_the_pick_scores_highest_in_its_population():
    """The maximization reads the score through the candidates, costs and all."""
    bo = _loop()
    _design(bo)
    bo.suggest(record_population=True)
    assert bo.last_acquisition_run is not None
    read = read_frontier(bo.last_acquisition_run)
    assert read.higher_scored_members == 0
    assert read.score_at_pick > 0.0, "the control: the pick improves on the front"


def test_under_repeated_measurements_the_front_holds_the_posterior_means():
    """The largest single observation is the one the noise inflated most, so, as for the
    incumbent, the front reads the posterior mean at each measured term."""
    bo = _loop(
        repeated_measurements=True,
        kernel=OrderedRootedSubtreeKernel(normalize=True) + WhiteKernel(noise_level=0.3),
    )
    _design(bo)
    bo.suggest(record_population=True)
    assert bo.last_acquisition_run is not None
    acquisition = bo.last_acquisition_run.acquisition
    snapshot = bo.get_state_snapshot()
    measured = list(dict.fromkeys(snapshot["x_list"]))
    means = np.asarray(bo.surrogate.predict(measured, return_std=True)[0], dtype=float)
    assert acquisition.front == [(float(m), _cost(x)) for x, m in zip(measured, means)]
    observed = [float(y) for y in snapshot["y_list"]]
    assert [value for value, _ in acquisition.front] != observed, "the control: means, not values"
    # And the score reads the latent function, as the three standard scores do: at a measured term
    # its deviation is the function's, below the observation's.
    _, latent = acquisition.marginal_posterior(measured[:1])
    _, noisy = bo.surrogate.predict(measured[:1], return_std=True)
    assert float(latent[0]) < float(noisy[0])


def test_an_outstanding_term_is_not_on_the_front_but_is_known():
    """A term still being evaluated has no value to stand on the front with."""
    bo = _loop(max_outstanding=2)
    _design(bo)
    first = bo.suggest(record_population=True)
    assert bo.last_acquisition_run is not None
    front = bo.last_acquisition_run.acquisition.front
    bo.suggest(record_population=True)
    second = bo.last_acquisition_run.acquisition
    assert second.front == front
    assert second.known_points is not None and first.candidate in second.known_points


@pytest.mark.parametrize(
    ("settings", "error", "match"),
    [
        ({"ehvi_cost": None}, TypeError, "the cost of a term"),
        ({"ehvi_cost": "size"}, TypeError, "the cost of a term"),
        ({"ehvi_reference": None}, ValueError, "the reference point"),
        ({"ehvi_reference": (math.nan, 8.0)}, ValueError, "the reference point"),
        ({"ehvi_reference": (0.0, math.inf)}, ValueError, "the reference point"),
    ],
)
def test_a_cost_or_reference_point_that_cannot_serve_is_refused_before_the_design(
    settings, error, match
):
    """Left to the pass, a missing cost or a reference without a finite bound would surface
    after the design had been drawn and evaluated.

    Args:
        settings (dict): The setting that cannot serve.
        error (type): The refusal.
        match (str): What it says.
    """
    paid: list[Any] = []
    with pytest.raises(error, match=match):
        _loop(**settings).optimize(_paying(paid), 2, initial_size=3)
    assert paid == [], "refused before anything was evaluated"


def test_the_cost_and_the_reference_are_read_under_this_acquisition_alone():
    """The parameters of an acquisition the run does not build are not the run's parameters."""
    bo = _loop(acquisition_function="ExpectedImprovement", ehvi_cost=None, ehvi_reference=None)
    bo.check_configuration()


def test_a_reference_cost_below_every_cost_is_named_rather_than_the_kernel(caplog):
    """Every term costs 5 to 7 and the reference cost is 4: nothing can add to the front's area.

    The score is then 0 everywhere for a reason the surrogate has no part in, and the warning says
    which setting it comes from instead of sending the reader to the kernel.
    """
    bo = _loop(ehvi_reference=(0.0, 4.0))
    with caplog.at_level(logging.WARNING, logger="bayesian_optimization"):
        bo.optimize(_objective, 2, initial_size=4)
    assert "ehvi_reference" in caplog.text
    assert "(0.0, 4.0)" in caplog.text
    assert "read_calibration" not in caplog.text


def test_an_observed_cost_that_is_not_finite_is_refused_naming_the_term_and_the_cost():
    """A cost is a measurement, and a failed one on the front is refused with what produced it."""
    bo = _loop(ehvi_cost=lambda term: math.inf if term.size == 5 else float(term.size))
    with pytest.raises(ValueError, match=r"ehvi_cost gave the observed term .* the cost inf"):
        bo.optimize(_objective, 2, initial_size=4)


def test_each_term_is_costed_once_per_run():
    """The cost is exact, so a term is costed once in a run however often a pass meets it; a reset,
    or another cost function, costs it afresh."""
    calls: Counter = Counter()

    def counting(term: Any) -> float:
        calls[term] += 1
        return float(term.size)

    bo = _loop(ehvi_cost=counting)
    bo.optimize(_objective, 3, initial_size=4)
    assert calls and max(calls.values()) == 1
    observed = list(bo.get_state_snapshot()["x_list"])
    values = [float(y) for y in bo.get_state_snapshot()["y_list"]]

    bo.reset()
    calls.clear()
    bo.initialize(x0=observed, y0=values)
    suggestion = bo.suggest()
    bo.observe(suggestion.candidate, _objective(suggestion.candidate))
    assert all(calls[term] == 1 for term in observed), "the reset forgot the costs"

    other: Counter = Counter()

    def recounting(term: Any) -> float:
        other[term] += 1
        return float(term.size)

    bo.ehvi_cost = recounting
    bo.suggest()
    assert all(other[term] == 1 for term in observed), "another cost function costs afresh"


def test_a_run_completes_and_answers_a_term_it_measured():
    """The closed loop, end to end, under the expected hypervolume improvement."""
    paid: list[Any] = []
    bo = _loop()
    result = bo.optimize(_paying(paid), 3, initial_size=4)
    assert len(paid) == 7
    assert result["best_tree"] in bo.get_state_snapshot()["x_list"]
