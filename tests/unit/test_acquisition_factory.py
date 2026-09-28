"""An acquisition of the caller's own, built afresh for every pass by a factory.

``acquisition_function`` names one of the three scores this loop ships, or it is a callable that
builds an ``AcquisitionFunction`` from what a pass has: the surrogate it fitted, the incumbent, and
the points already observed, which the known-point floor keeps below every novel candidate.  The
loop builds its own three the same way, one new object per pass, so that a recorded acquisition
still says what its pass maximized once the next pass has run.  A factory that cannot serve a pass
is refused before the design is paid, by a dry run on a stand-in posterior.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
from cosy.core import Synthesizer
from cosy.search import SizeUniformSampler

from bayesian_optimization import BayesianOptimization
from bayesian_optimization.acquisition_function import AcquisitionFunction
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


@pytest.mark.parametrize(
    ("factory", "error", "match"),
    [
        (_not_an_acquisition, TypeError, "not an AcquisitionFunction"),
        (_without_the_known_points, ValueError, "known points"),
        (_on_another_model, ValueError, "surrogate"),
        (_wrong_signature, TypeError, "incumbent"),
    ],
)
def test_a_factory_that_cannot_serve_a_pass_is_refused_before_the_design(factory, error, match):
    paid: list[Any] = []

    def objective(term: Any) -> float:
        paid.append(term)
        return _objective(term)

    with pytest.raises(error, match=match):
        _loop(factory).optimize(objective, 2, initial_size=3)
    assert paid == [], "refused before anything was evaluated"


def test_an_acquisition_without_a_lower_bound_is_refused_where_a_bound_is_needed():
    bo = _loop(Recording())
    with pytest.raises(ValueError, match="Optimism"):
        bo.check_configuration(acquisition_fitness_mode="single")
    bo.check_configuration(acquisition_fitness_mode="batch")


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
