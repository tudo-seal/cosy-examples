"""An acquisition of the caller's own, built afresh for every pass by a factory.

``acquisition_function`` names one of the three scores this loop ships, or it is a callable that
builds an ``AcquisitionFunction`` from what a pass has: the surrogate it fitted, the incumbent, and
the points already observed, which the known-point floor keeps below every novel candidate.  The
loop builds its own three the same way, one new object per pass, so that a recorded acquisition
still says what its pass maximized once the next pass has run.  A callable that cannot be called
that way is refused before the design is paid; what it builds is checked at every pass, since only a
pass has a surrogate, an incumbent and points to build from.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pytest
from cosy.core import Synthesizer
from cosy.search import SizeUniformSampler

from bayesian_optimization import BayesianOptimization
from bayesian_optimization.acquisition_function import (
    AcquisitionFunction,
    UpperConfidenceBound,
    require_term,
)
from tests.integration.test_real_ea_loop import (
    _MAX_SIZE,
    _TARGET,
    _make_ea,
    _objective,
    _repository,
)


class Optimism(AcquisitionFunction):
    """The posterior mean plus half a deviation: a score of the caller's own, unbounded below."""

    lower_bound = None

    def score(self, mean: np.ndarray, deviation: np.ndarray) -> np.ndarray:
        scored: np.ndarray = mean + 0.5 * deviation
        return scored


class Elsewhere:
    """A posterior, and not the one a pass fitted."""

    def predict(self, X: Any, return_std: bool = False) -> Any:
        raise AssertionError("never asked")


class Recording:
    """A factory that builds :class:`Optimism` and keeps what each call was handed."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.built: list[AcquisitionFunction] = []

    def __call__(self, gp: Any, *, incumbent: float, known_points: set[Any]) -> AcquisitionFunction:
        self.calls.append({"gp": gp, "incumbent": incumbent, "known_points": set(known_points)})
        acquisition = Optimism(gp, known_points=known_points)
        self.built.append(acquisition)
        return acquisition


def _loop(factory: Any, **kwargs: Any) -> BayesianOptimization:
    space = Synthesizer(_repository(), {}).construct_solution_space(_TARGET).prune()
    return BayesianOptimization(
        space, _TARGET, acquisition_function=factory, optimizer=_make_ea(space, 7), seed=7,
        sampler=SizeUniformSampler(_MAX_SIZE, random.Random(7)), n_restarts_kernel_optimizer=0,
        **kwargs,
    )


def _design(bo: BayesianOptimization, size: int = 4) -> None:
    bo.initialize(initial_size=size)
    for _ in range(size):
        suggestion = bo.suggest()
        bo.observe(suggestion.candidate, _objective(suggestion.candidate))


def test_a_factory_builds_each_pass_s_acquisition_from_that_pass():
    factory = Recording()
    bo = _loop(factory)
    _design(bo)

    for step in range(3):
        observed = list(bo.get_state_snapshot()["x_list"])
        values = list(bo.get_state_snapshot()["y_list"])
        suggestion = bo.suggest(record_population=True)
        call = factory.calls[-1]
        assert len(factory.calls) == step + 1, "one build per pass"
        assert call["gp"] is bo.surrogate, "the surrogate this pass fitted"
        assert call["incumbent"] == max(values)
        assert call["known_points"] == set(observed)
        assert bo.last_acquisition_run is not None
        assert bo.last_acquisition_run.acquisition is factory.built[-1], "the one maximized"
        bo.observe(suggestion.candidate, _objective(suggestion.candidate))
    assert len({id(acquisition) for acquisition in factory.built}) == 3, "a new object each pass"


def _not_an_acquisition(gp: Any, *, incumbent: float, known_points: set[Any]) -> Any:
    return lambda term: 0.0


def _without_the_known_points(gp: Any, *, incumbent: float, known_points: set[Any]) -> Any:
    return Optimism(gp)


def _on_another_model(gp: Any, *, incumbent: float, known_points: set[Any]) -> Any:
    return Optimism(Elsewhere(), known_points=known_points)


def _wrong_signature(gp: Any) -> Any:
    return Optimism(gp)


def _paying(paid: list[Any]) -> Any:
    def objective(term: Any) -> float:
        paid.append(term)
        return _objective(term)

    return objective


@pytest.mark.parametrize(
    "setting",
    [
        _wrong_signature,
        UpperConfidenceBound,  # a class, but one that takes a beta and no incumbent
        len,
        lambda: None,
        Optimism(Elsewhere(), known_points=set()),  # an acquisition, not what builds one
    ],
)
def test_a_callable_that_cannot_be_called_as_a_factory_is_refused_before_the_design(setting):
    paid: list[Any] = []
    with pytest.raises(TypeError, match="cannot be called as"):
        _loop(setting).optimize(_paying(paid), 2, initial_size=3)
    assert paid == [], "refused before anything was evaluated"


@pytest.mark.parametrize(
    ("factory", "error", "match"),
    [
        (_not_an_acquisition, TypeError, "not an AcquisitionFunction"),
        (_without_the_known_points, ValueError, "known points"),
        (_on_another_model, ValueError, "surrogate"),
    ],
)
def test_what_a_factory_builds_is_refused_at_its_pass_before_its_pick_is_evaluated(
    factory, error, match
):
    """Only a pass has a surrogate, an incumbent and points to build from, so what a factory builds
    is checked there: the design is paid, the pass's pick is not."""
    paid: list[Any] = []
    with pytest.raises(error, match=match):
        _loop(factory).optimize(_paying(paid), 2, initial_size=3)
    assert len(paid) == 3, "the design, and nothing of the pass"


def test_an_acquisition_without_a_lower_bound_is_refused_at_its_pass_where_a_bound_is_needed():
    paid: list[Any] = []
    bo = _loop(Recording())
    bo.check_configuration(acquisition_fitness_mode="single")  # what it builds is not known yet
    with pytest.raises(ValueError, match="Optimism"):
        bo.optimize(_paying(paid), 2, initial_size=3, acquisition_fitness_mode="single")
    assert len(paid) == 3


def _predicting(gp: Any, *, incumbent: float, known_points: set[Any]) -> Any:
    gp.predict(list(known_points), return_std=True)
    return Optimism(gp, known_points=known_points)


def _reading_the_training_set(gp: Any, *, incumbent: float, known_points: set[Any]) -> Any:
    assert len(gp.X_train_) == len(known_points)
    return Optimism(gp, known_points=known_points)


def _scaled_by_the_incumbent(gp: Any, *, incumbent: float, known_points: set[Any]) -> Any:
    math.log(incumbent)  # every value of this objective is positive
    return Optimism(gp, known_points=known_points)


def _checking_its_points(gp: Any, *, incumbent: float, known_points: set[Any]) -> Any:
    for point in known_points:
        require_term(point)
    return Optimism(gp, known_points=known_points)


def _holding_a_pending_point_too(gp: Any, *, incumbent: float, known_points: set[Any]) -> Any:
    return Optimism(gp, known_points=set(known_points) | {"a term still being evaluated"})


@pytest.mark.parametrize(
    "factory",
    [
        _predicting,
        _reading_the_training_set,
        _scaled_by_the_incumbent,
        _checking_its_points,
        _holding_a_pending_point_too,
    ],
)
def test_a_factory_every_pass_serves_runs(factory):
    """Each of these uses what a real pass hands it -- a fitted surrogate, a positive incumbent,
    terms -- or holds more points than the pass's; none was ever unable to serve a pass."""
    result = _loop(factory).optimize(_objective, 2, initial_size=3)
    assert result["iterations"] == 2


def test_a_factory_is_called_once_per_pass_and_never_before_the_design():
    """A factory with a state of its own -- a schedule, a random draw -- runs the same passes
    through optimize() and through the ask/tell layer."""
    factory = Recording()
    _loop(factory).optimize(_objective, 3, initial_size=4)
    assert len(factory.calls) == 3


def test_the_log_names_the_class_the_factory_built(monkeypatch):
    import bayesian_optimization.bo as bo_module

    named: list[str] = []
    original = bo_module.warn_if_exploitation_stalls

    def spy(*args: Any, **kwargs: Any) -> bool:
        named.append(kwargs["acquisition_name"])
        warned: bool = original(*args, **kwargs)
        return warned

    monkeypatch.setattr(bo_module, "warn_if_exploitation_stalls", spy)
    for acquisition in (Recording(), "ExpectedImprovement"):
        bo = _loop(acquisition)
        _design(bo)
        bo.suggest()
    assert named == ["Optimism", "ExpectedImprovement"]


@pytest.mark.parametrize(
    "setting", ["ei", ["ExpectedImprovement"], 3],
)
def test_anything_else_is_refused_by_what_it_may_be(setting):
    with pytest.raises(ValueError, match="one of .* or a callable"):
        _loop(setting).check_configuration()
