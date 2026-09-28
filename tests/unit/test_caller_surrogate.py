"""A surrogate of the caller's own in place of the Gaussian process.

``surrogate_model`` takes a ``Surrogate``: an object whose ``fit(terms, values)`` answers a
posterior, anything with ``predict(X, return_std=...)``.  Every pass conditions it on the distinct
pairs of the dataset as it conditions the Gaussian process, and keeps a copy of the posterior it
answered, so that nothing a later fit does -- a scikit-learn estimator refits itself -- changes what
an earlier pass recorded.  The diagnostics fit a copy of the surrogate, so that a surrogate with a
state of its own runs the same passes with and without them.  The settings that configure the
Gaussian process reach nothing beside it and are refused.  The run layer reads what only a Gaussian
process answers -- its calibration, its marginal likelihood, its kernel -- off any posterior that
has it, and leaves those cells empty for one that has not.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from typing import Any

import numpy as np
import pytest
from cosy.core import Synthesizer
from cosy.search import SizeUniformSampler
from sklearn.gaussian_process import GaussianProcessRegressor

from bayesian_optimization import BayesianOptimization
from bayesian_optimization.acquisition_optimizer import AcquisitionOptimizer
from bayesian_optimization.diagnostics.run_log import warn_if_exploitation_stalls
from bayesian_optimization.kernels import OrderedRootedSubtreeKernel
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


class InPlace(Remembering):
    """The shape of a scikit-learn estimator: ``fit`` refits the object itself and answers it."""

    def __init__(self) -> None:
        self.known, self.mean = {}, 0.0

    def fit(self, terms: Any, values: Any) -> InPlace:
        Remembering.__init__(self, list(terms), list(values))
        return self


class DoubleBuffered:
    """Two posteriors refitted in place in turn: the one answered two fits ago is changed."""

    def __init__(self) -> None:
        self.buffers = [InPlace(), InPlace()]
        self.count = 0

    def fit(self, terms: Any, values: Any) -> InPlace:
        self.count += 1
        return self.buffers[self.count % 2].fit(terms, values)


class Memoizing(Recording):
    """One posterior per set of pairs, answered again for the same pairs."""

    def __init__(self) -> None:
        super().__init__()
        self.answers: dict[tuple[Any, ...], Remembering] = {}

    def fit(self, terms: Any, values: Any) -> Remembering:
        key = tuple(zip(terms, values, strict=True))
        if key not in self.answers:
            self.answers[key] = super().fit(terms, values)
        return self.answers[key]


class Bagged:
    """A surrogate with a state of its own: each fit conditions a Gaussian process on a random half
    of the pairs, drawn from its own random source."""

    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.fits = 0

    def fit(self, terms: Any, values: Any) -> GaussianProcessRegressor:
        self.fits += 1
        chosen = sorted(self.rng.sample(range(len(terms)), max(1, len(terms) // 2)))
        gp = GaussianProcessRegressor(kernel=OrderedRootedSubtreeKernel(), alpha=1e-6,
                                      normalize_y=True)
        gp.fit(np.asarray([terms[i] for i in chosen], dtype=object),
               np.asarray([values[i] for i in chosen], dtype=float))
        return gp


class _Delegating:
    """A posterior that is not a ``GaussianProcessRegressor`` and answers everything one does."""

    def __init__(self, gp: GaussianProcessRegressor) -> None:
        self._gp = gp

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__") or name == "_gp":
            raise AttributeError(name)
        return getattr(self._gp, name)


class GaussianAlike:
    """A surrogate that fits a Gaussian process and answers it behind a delegating object."""

    def fit(self, terms: Any, values: Any) -> _Delegating:
        gp = GaussianProcessRegressor(kernel=OrderedRootedSubtreeKernel(), alpha=1e-6,
                                      normalize_y=True)
        gp.fit(np.asarray(terms, dtype=object), np.asarray(values, dtype=float))
        return _Delegating(gp)


class Uncopyable(Recording):
    def __deepcopy__(self, memo: Any) -> Any:
        raise TypeError("this surrogate cannot be copied")


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


def _list_loop(surrogate: Any) -> BayesianOptimization:
    return BayesianOptimization(
        list_space(), LIST, seed=3, sampler=SizeUniformSampler(6, random.Random(3)),
        optimizer=build_acquisition_optimizer(population_size=6, generations=2, depth_bound=4,
                                              seed=3),
        surrogate_model=surrogate,
    )


def test_a_caller_s_surrogate_is_the_model_every_pass_maximizes_against():
    surrogate = Recording()
    bo = _loop(surrogate)
    runs = _run(bo)

    assert len(surrogate.fitted) == 2, "one fit per pass"
    assert [run.acquisition.gp.known for run in runs] == [fit.known for fit in surrogate.fitted]
    assert bo.surrogate.known == surrogate.fitted[-1].known
    assert bo.finalize()["gp_model"].known == surrogate.fitted[-1].known


def test_it_is_fitted_on_the_distinct_pairs_in_the_order_they_first_appeared():
    surrogate = Recording()
    bo = _loop(surrogate)
    _run(bo, n_design=4, n_passes=1)

    terms, values = surrogate.fits[0]
    assert terms == list(bo.design)
    assert values == [_objective(term) for term in bo.design]


@pytest.mark.parametrize("surrogate", [InPlace, DoubleBuffered, Memoizing])
def test_every_pass_keeps_the_posterior_it_maximized_against_whatever_later_fits_do(surrogate):
    """A scikit-learn estimator passed as it is refits itself; a double buffer refits the posterior
    of two passes ago; a memo answers an old one.  Each pass keeps its own copy."""
    bo = _loop(surrogate())
    bo.initialize(initial_size=4)
    for _ in range(4):
        suggestion = bo.suggest()
        bo.observe(suggestion.candidate, _objective(suggestion.candidate))
    first = bo.suggest(record_population=True)
    recorded = bo.last_acquisition_run
    probe = [first.candidate]
    before = recorded.acquisition.gp.predict(probe, return_std=True)
    bo.observe(first.candidate, _objective(first.candidate))
    for _ in range(3):
        suggestion = bo.suggest()
        bo.observe(suggestion.candidate, _objective(suggestion.candidate))

    after = recorded.acquisition.gp.predict(probe, return_std=True)
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])


def test_a_retried_pass_may_answer_the_posterior_it_answered_before(monkeypatch):
    """A pass whose maximization failed is suggested again on the same pairs, and a memo answers
    the posterior it answered then: a correct answer, not a reused one."""
    bo = _loop(Memoizing())
    _run(bo, n_design=4, n_passes=0)
    original = AcquisitionOptimizer.maximize
    failures = []

    def failing_once(self: Any, *args: Any, **kwargs: Any) -> Any:
        if not failures:
            failures.append(True)
            raise RuntimeError("the maximization failed once")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(AcquisitionOptimizer, "maximize", failing_once)
    with pytest.raises(RuntimeError, match="failed once"):
        bo.suggest()
    suggestion = bo.suggest()
    bo.observe(suggestion.candidate, _objective(suggestion.candidate))


def test_the_diagnostics_fit_a_copy_so_a_surrogate_with_a_state_runs_the_same_passes(tmp_path):
    """The run layer's held-out read fits the surrogate after every pass; were it the loop's own,
    a surrogate that draws at each fit would draw differently under the run layer than alone."""
    plain = _list_loop(Bagged(5))
    plain.initialize(initial_size=4)
    for _ in range(4 + 4):
        suggestion = plain.suggest()
        plain.observe(suggestion.candidate, _metrics(suggestion.candidate)["score"])
    driven_surrogate = Bagged(5)
    driven = _list_loop(driven_surrogate)
    run_search(
        driven, _metrics, schema=SCHEMA, csv_path=str(tmp_path / "run.csv"), pretty_algebra=dict,
        echo=lambda line: None, n_design=4, n_passes=4,
    )

    assert driven.get_state_snapshot()["x_list"] == plain.get_state_snapshot()["x_list"]
    assert driven_surrogate.fits == 4, "the passes' fits, and the diagnostics' on copies"


def test_the_diagnostic_fits_answer_what_the_surrogate_answers_and_leave_it_as_it_was():
    surrogate = Recording()
    bo = _loop(surrogate)
    _run(bo)
    terms = list(bo.get_state_snapshot()["x_list"])
    values = list(bo.get_state_snapshot()["y_list"])

    held_out = bo.surrogate_over(terms[0::2], values[0::2])
    whole = bo.surrogate_over_dataset()

    assert held_out.known == dict(zip(terms[0::2], values[0::2], strict=True))
    assert whole.known == dict(zip(terms, values, strict=True))
    assert len(surrogate.fits) == 2, "only the passes fitted the surrogate itself"


def test_a_surrogate_that_cannot_be_copied_is_refused_before_the_design():
    paid: list[Any] = []

    def objective(term: Any) -> float:
        paid.append(term)
        return _objective(term)

    with pytest.raises(TypeError, match="copied"):
        _loop(Uncopyable()).optimize(objective, 2, initial_size=3)
    assert paid == []


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


def test_a_setting_or_a_surrogate_set_after_construction_is_refused_before_the_design():
    later_setting = _loop(Recording())
    later_setting.kernel_optimizer = "fmin_l_bfgs_b"
    with pytest.raises(ValueError, match="kernel_optimizer"):
        later_setting.check_configuration()

    later_surrogate = _loop(None, gp_normalize_y=False)
    later_surrogate.surrogate_model = Recording()
    with pytest.raises(ValueError, match="gp_normalize_y"):
        later_surrogate.check_configuration()


@pytest.mark.parametrize(
    ("argument", "value"),
    [("gp_params", {"alpha": 1e-3}), ("alpha", 1e-3), ("alpha", np.array([1e-3, 1e-3, 1e-3]))],
)
def test_a_per_run_setting_of_the_gaussian_process_is_refused_before_the_design(argument, value):
    paid: list[Any] = []

    def objective(term: Any) -> float:
        paid.append(term)
        return _objective(term)

    with pytest.raises(ValueError, match=argument):
        _loop(Recording()).optimize(objective, 2, initial_size=3, **{argument: value})
    assert paid == []


def test_a_run_with_a_caller_s_surrogate_leaves_the_gaussian_process_reads_empty(tmp_path):
    """On the list space the run layer's tests use.  The fit read predicts, which any posterior
    does, and the count of pairs a pass conditioned on is the loop's own; the calibration, the
    likelihood, the kernel and the Gram matrix of the constructor's kernel are the Gaussian
    process's, and a surrogate that is none has none of them."""
    bo = _list_loop(Recording())
    csv_path = tmp_path / "run.csv"
    outcome = run_search(
        bo, _metrics, schema=SCHEMA, csv_path=str(csv_path), pretty_algebra=dict,
        echo=lambda line: None, n_design=3, n_passes=3,
    )
    diagnostics = write_run_diagnostics(str(csv_path), bo, outcome.result)

    with (tmp_path / "run_surrogate.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    gaussian_only = (
        "log_marginal_likelihood", "calibration_root_mean_square",
        "calibration_standard_deviation", "calibration_maximum_absolute",
        "calibration_outside_two", "kernel_hyperparameters",
    )
    assert len(rows) == 3
    assert [int(row["n_train"]) for row in rows] == [3, 4, 5], "the pairs each pass conditioned on"
    assert all(row[column] == "" for row in rows for column in gaussian_only)
    assert rows[-1]["fit_size"] != "", "the fit read predicts, which this surrogate does"
    assert diagnostics["gram"] is None and diagnostics["calibration"] is None
    assert diagnostics["fit"] is not None
    assert json.loads((tmp_path / "run_diagnostics.json").read_text())["calibration"] is None


def test_a_posterior_that_answers_what_a_gaussian_process_answers_is_read_as_one_in_both_places(
    tmp_path,
):
    """Read off what the posterior has, not off its class, and alike in the per-pass log and the
    run's diagnostics; the constructor's kernel stays unread, since this surrogate did not use it."""
    bo = _list_loop(GaussianAlike())
    csv_path = tmp_path / "run.csv"
    outcome = run_search(
        bo, _metrics, schema=SCHEMA, csv_path=str(csv_path), pretty_algebra=dict,
        echo=lambda line: None, n_design=3, n_passes=3,
    )
    diagnostics = write_run_diagnostics(str(csv_path), bo, outcome.result)

    with (tmp_path / "run_surrogate.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(row["log_marginal_likelihood"] != "" for row in rows)
    assert all(row["calibration_root_mean_square"] != "" for row in rows)
    assert diagnostics["calibration"] is not None
    assert diagnostics["gram"] is None


def test_the_stall_warning_advises_the_gaussian_process_only_where_there_is_one(caplog):
    logger = logging.getLogger("bayesian_optimization.test_stall")
    with caplog.at_level(logging.WARNING, logger="bayesian_optimization.test_stall"):
        for gaussian_process in (True, False):
            warn_if_exploitation_stalls(
                logger, iteration=0, acquisition_value=0.0, lower_bound=0.0,
                acquisition_name="ExpectedImprovement", gaussian_process=gaussian_process,
            )
    with_gp, without = (record.getMessage() for record in caplog.records)
    assert "kernel_optimizer" in with_gp and "read_calibration" in with_gp
    assert "kernel_optimizer" not in without and "read_calibration" not in without
    assert "held-out" in without
