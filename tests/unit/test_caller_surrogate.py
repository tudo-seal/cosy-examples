"""A surrogate of the caller's own in place of the Gaussian process.

``surrogate_model`` takes a ``Surrogate``: an object whose ``fit(terms, values)`` answers a
posterior, anything with ``predict(X, return_std=...)``, a new one for every fit.  Every pass
conditions it on the distinct pairs of the dataset as it conditions the Gaussian process, and the
acquisition, the diagnostics' held-out fit and the result read that posterior.  The settings that
configure the Gaussian process reach nothing beside it and are refused, and the run layer leaves
the reads that only a Gaussian process answers -- its calibration, its marginal likelihood, its
kernel -- empty rather than failing after the design is paid.
"""

from __future__ import annotations

import csv
import json
import random
from typing import Any

import numpy as np
import pytest
from cosy.core import Synthesizer
from cosy.search import SizeUniformSampler

from bayesian_optimization import BayesianOptimization
from bayesian_optimization.runs import (
    build_acquisition_optimizer,
    run_search,
    write_run_diagnostics,
)
from tests.integration.test_real_ea_loop import (
    _MAX_SIZE,
    _TARGET,
    _make_ea,
    _objective,
    _repository,
)
from tests.spaces import LIST, list_space
from tests.unit.test_run_driver import SCHEMA, _metrics


class Remembering:
    """A posterior of the caller's own: the observed value at an observed term, the mean with a
    unit deviation anywhere else."""

    def __init__(self, terms: list[Any], values: list[float]) -> None:
        self.known = dict(zip(terms, values, strict=True))
        self.mean = float(np.mean(values))

    def predict(self, X: Any, return_std: bool = False) -> Any:
        mean = np.array([self.known.get(term, self.mean) for term in X], dtype=float)
        if not return_std:
            return mean
        deviation = np.array([0.0 if term in self.known else 1.0 for term in X], dtype=float)
        return mean, deviation


class Recording:
    """A surrogate that fits :class:`Remembering` and keeps what every fit was handed."""

    def __init__(self) -> None:
        self.fits: list[tuple[list[Any], list[float]]] = []
        self.fitted: list[Remembering] = []

    def fit(self, terms: Any, values: Any) -> Remembering:
        self.fits.append((list(terms), list(values)))
        self.fitted.append(Remembering(list(terms), list(values)))
        return self.fitted[-1]


class Reusing(Recording):
    """A surrogate whose fit answers the one posterior it built first, again and again."""

    def fit(self, terms: Any, values: Any) -> Remembering:
        if not self.fitted:
            return super().fit(terms, values)
        return self.fitted[0]


def _loop(surrogate: Any, **kwargs: Any) -> BayesianOptimization:
    space = Synthesizer(_repository(), {}).construct_solution_space(_TARGET).prune()
    return BayesianOptimization(
        space, _TARGET, seed=7, optimizer=_make_ea(space, 7),
        sampler=SizeUniformSampler(_MAX_SIZE, random.Random(7)), surrogate_model=surrogate,
        **kwargs,
    )


def _run(bo: BayesianOptimization, n_design: int = 4, n_passes: int = 2) -> list[Any]:
    bo.initialize(initial_size=n_design)
    for _ in range(n_design):
        suggestion = bo.suggest()
        bo.observe(suggestion.candidate, _objective(suggestion.candidate))
    runs = []
    for _ in range(n_passes):
        suggestion = bo.suggest(record_population=True)
        runs.append(bo.last_acquisition_run)
        bo.observe(suggestion.candidate, _objective(suggestion.candidate))
    return runs


def test_a_caller_s_surrogate_is_the_model_every_pass_maximizes_against():
    surrogate = Recording()
    bo = _loop(surrogate)
    runs = _run(bo)

    assert len(surrogate.fitted) == 2, "one fit per pass"
    assert [run.acquisition.gp for run in runs] == surrogate.fitted
    assert bo.surrogate is surrogate.fitted[-1]
    assert bo.finalize()["gp_model"] is surrogate.fitted[-1]


def test_it_is_fitted_on_the_distinct_pairs_in_the_order_they_first_appeared():
    surrogate = Recording()
    bo = _loop(surrogate)
    _run(bo, n_design=4, n_passes=1)

    terms, values = surrogate.fits[0]
    assert terms == list(bo.design)
    assert values == [_objective(term) for term in bo.design]


def test_a_fit_that_answers_a_posterior_it_answered_before_is_refused():
    """A recorded acquisition holds the posterior its pass maximized against; one fit that changes
    an earlier posterior in place would change what every earlier record says."""
    bo = _loop(Reusing())
    with pytest.raises(ValueError, match="new posterior"):
        _run(bo, n_design=4, n_passes=2)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("kernel", object()),
        ("kernel_optimizer", "fmin_l_bfgs_b"),
        ("n_restarts_kernel_optimizer", 5),
        ("gp_normalize_y", False),
    ],
)
def test_a_setting_of_the_gaussian_process_beside_a_caller_s_surrogate_is_refused(setting, value):
    with pytest.raises(ValueError, match=setting):
        _loop(Recording(), **{setting: value})


@pytest.mark.parametrize(("argument", "value"), [("gp_params", {"alpha": 1e-3}), ("alpha", 1e-3)])
def test_a_per_run_setting_of_the_gaussian_process_is_refused_before_the_design(argument, value):
    paid: list[Any] = []

    def objective(term: Any) -> float:
        paid.append(term)
        return _objective(term)

    with pytest.raises(ValueError, match=argument):
        _loop(Recording()).optimize(objective, 2, initial_size=3, **{argument: value})
    assert paid == []


def test_the_diagnostic_fits_go_through_the_caller_s_surrogate():
    surrogate = Recording()
    bo = _loop(surrogate)
    _run(bo)
    terms = list(bo.get_state_snapshot()["x_list"])
    values = list(bo.get_state_snapshot()["y_list"])

    held_out = bo.surrogate_over(terms[0::2], values[0::2])
    whole = bo.surrogate_over_dataset()

    assert held_out is surrogate.fitted[-2] and whole is surrogate.fitted[-1]
    assert surrogate.fits[-1] == (terms, values)


def test_a_run_with_a_caller_s_surrogate_leaves_the_gaussian_process_reads_empty(tmp_path):
    """On the list space the run layer's tests use.  The fit read predicts, which any posterior
    does; the calibration, the likelihood, the kernel and the Gram matrix of the constructor's
    kernel are the Gaussian process's, and a surrogate that is none has none of them."""
    bo = BayesianOptimization(
        list_space(), LIST, seed=3, sampler=SizeUniformSampler(6, random.Random(3)),
        optimizer=build_acquisition_optimizer(population_size=6, generations=2, depth_bound=4,
                                              seed=3),
        surrogate_model=Recording(),
    )
    csv_path = tmp_path / "run.csv"
    outcome = run_search(
        bo, _metrics, schema=SCHEMA, csv_path=str(csv_path), pretty_algebra=dict,
        echo=lambda line: None, n_design=3, n_passes=3,
    )
    diagnostics = write_run_diagnostics(str(csv_path), bo, outcome.result)

    with (tmp_path / "run_surrogate.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    gaussian_only = (
        "n_train", "log_marginal_likelihood", "calibration_root_mean_square",
        "calibration_standard_deviation", "calibration_maximum_absolute",
        "calibration_outside_two", "kernel_hyperparameters",
    )
    assert len(rows) == 3
    assert all(row[column] == "" for row in rows for column in gaussian_only)
    assert rows[-1]["fit_size"] != "", "the fit read predicts, which this surrogate does"
    assert diagnostics["gram"] is None and diagnostics["calibration"] is None
    assert diagnostics["fit"] is not None
    assert json.loads((tmp_path / "run_diagnostics.json").read_text())["calibration"] is None
