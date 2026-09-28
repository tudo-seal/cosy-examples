"""Repeated measurements, an option: a term measured again is a row of its own, under a noise model.

``BayesianOptimization(repeated_measurements=True)`` lets a pass propose a term the loop has measured
already, where the acquisition prefers it, and conditions the surrogate on every row rather than on
the distinct pairs.  A quality measure that varies between calls needs a model of that variation, a
``WhiteKernel`` in the kernel, and the acquisition scores the latent function rather than the noisy
observation of it: at a measured term the observation's deviation never falls below the noise, and
an acquisition that read it would measure the same term again and again.  Designs stay sets; a term
is measured again only by a pass.  Off, the loop is the one it always was.
"""

from __future__ import annotations

import csv
import random
from typing import Any

import numpy as np
import pytest
from cosy.search import SizeUniformSampler
from sklearn.gaussian_process.kernels import WhiteKernel

from bayesian_optimization import BayesianOptimization
from bayesian_optimization.acquisition_optimizer import SampleMaximizer
from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel
from bayesian_optimization.runs import MetricSchema, Objective, run_search, write_run_diagnostics
from tests.spaces import LIST, list_space

NOISE = 0.3


def _noisy() -> Any:
    return OrderedRootedSubtreeKernel(normalize=True) + WhiteKernel(noise_level=NOISE)


def _loop(bo_factory: Any, candidates: list[Any], **kwargs: Any) -> Any:
    return bo_factory(candidates=candidates, repeated_measurements=True, kernel=_noisy(), **kwargs)


def test_a_measured_term_the_maximization_prefers_is_measured_again(bo_factory, tree_corpus):
    bo = _loop(bo_factory, [tree_corpus[0]])
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])

    suggestion = bo.suggest()
    assert suggestion.candidate == tree_corpus[0]
    assert suggestion.diagnostics["fallback_used"] is False
    bo.observe(tree_corpus[0], 1.4)

    snapshot = bo.get_state_snapshot()
    assert snapshot["x_list"] == [*tree_corpus[:3], tree_corpus[0]]
    assert snapshot["y_list"] == [1.0, 2.0, 0.5, 1.4]


def test_every_row_conditions_the_surrogate(bo_factory, tree_corpus):
    bo = _loop(bo_factory, [tree_corpus[0], tree_corpus[3]])
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    bo.observe(bo.suggest().candidate, 1.4)

    bo.suggest()
    assert len(bo.surrogate.X_train_) == 4, "the pass conditioned on every row"
    assert len(bo.surrogate_over_dataset().X_train_) == 4, "and so does the whole-dataset read"


def test_the_measured_term_is_scored_not_floored(bo_factory, tree_corpus):
    """The known-point floor reads no measured term: the best term, proposed again, is scored by
    what measuring it again would tell, which is more than nothing."""
    bo = _loop(bo_factory, [tree_corpus[1]])
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])

    suggestion = bo.suggest()
    assert suggestion.candidate == tree_corpus[1]
    assert suggestion.acquisition_value > 0.0


def test_the_acquisition_scores_the_latent_function(bo_factory, tree_corpus):
    """At a measured term the deviation the acquisition reads is the function's: the observation's,
    less the noise the kernel fitted, in the units the targets were normalized by."""
    bo = _loop(bo_factory, [tree_corpus[0]])
    values = [1.0, 2.0, 0.5]
    bo.initialize(x0=tree_corpus[:3], y0=values)

    suggestion = bo.suggest()
    mean, observed = bo.surrogate.predict(np.asarray([tree_corpus[0]], dtype=object), return_std=True)
    latent = suggestion.diagnostics["deviation_at_pick"]
    assert latent**2 == pytest.approx(observed[0] ** 2 - NOISE * np.std(values) ** 2)
    assert latent < observed[0]
    assert suggestion.diagnostics["mean_at_pick"] == pytest.approx(mean[0])


def test_the_mode_without_a_noise_model_is_refused_before_the_design(bo_factory, tree_corpus):
    with pytest.raises(ValueError, match="WhiteKernel"):
        bo_factory(repeated_measurements=True)
    # the kernel is an attribute, read again before a run and before its design
    plain = _loop(bo_factory, [])
    plain.kernel = OrderedRootedSubtreeKernel(normalize=True)
    with pytest.raises(ValueError, match="WhiteKernel"):
        plain.check_configuration()
    with pytest.raises(ValueError, match="WhiteKernel"):
        plain.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    assert plain.get_state_snapshot()["x_list"] == []

    # the kernel a run hands over for the fit is the one read
    noisy = _loop(bo_factory, [])
    noisy.check_configuration()
    with pytest.raises(ValueError, match="WhiteKernel"):
        noisy.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5],
                         gp_params={"kernel": OrderedRootedSubtreeKernel(normalize=True)})
    assert noisy.get_state_snapshot()["x_list"] == []


def test_a_repeated_term_is_conditioned_on_only_in_the_mode(bo_factory, tree_corpus):
    rows = [tree_corpus[0], tree_corpus[0], tree_corpus[1]]
    with pytest.raises(ValueError, match="repeat"):
        bo_factory(kernel=_noisy()).surrogate_over(rows, [1.0, 1.2, 2.0])
    assert len(_loop(bo_factory, []).surrogate_over(rows, [1.0, 1.2, 2.0]).X_train_) == 3


def test_a_design_stays_a_set_in_the_mode(bo_factory, tree_corpus):
    with pytest.raises(ValueError, match="two different values"):
        _loop(bo_factory, []).initialize(x0=[tree_corpus[0], tree_corpus[0]], y0=[1.0, 1.5])


def test_off_the_loop_replaces_a_measured_term_as_it_always_did(bo_factory, tree_corpus):
    bo = bo_factory(candidates=[tree_corpus[0]], kernel=_noisy())
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    with pytest.raises(RuntimeError, match="no search space to draw a replacement from"):
        bo.suggest()


def test_a_run_measures_a_term_again_and_reads_its_diagnostics(tmp_path, monkeypatch):
    """Through the run layer: the term again as a row of its own, and the reads over the finished
    run answer rather than refuse a term with two values."""
    original = SampleMaximizer.maximize_with_population

    def the_first_measured(self: Any, acquisition: Any, query: Any, **kwargs: Any) -> Any:
        _pick, population, generations = original(self, acquisition, query, **kwargs)
        return loop.design[0], population, generations

    monkeypatch.setattr(SampleMaximizer, "maximize_with_population", the_first_measured)
    loop = BayesianOptimization(
        list_space(), LIST, sampler=SizeUniformSampler(6, random.Random(3)), seed=3,
        maximizer=SampleMaximizer(SizeUniformSampler(6, random.Random(103)), 8),
        kernel=_noisy(), repeated_measurements=True,
    )
    measured: list[Any] = []

    def evaluate(term: Any) -> dict[str, float]:
        measured.append(term)
        return {"score": 1.0 / (1 + len(str(term))) + 0.01 * len(measured)}

    path = tmp_path / "run.csv"
    outcome = run_search(
        loop, evaluate, schema=MetricSchema.for_metrics(Objective("score"), ["score"]),
        csv_path=str(path), pretty_algebra=dict, n_design=3, n_passes=2, echo=lambda line: None,
    )

    with open(path, newline="") as handle:
        structures = [row["structure"] for row in csv.DictReader(handle)]
    assert len(structures) == 5 and structures[3] == structures[4] == structures[0]
    write_run_diagnostics(str(path), loop, outcome.result)
    assert (tmp_path / "run_diagnostics.json").is_file()


def test_a_posterior_of_noise_alone_has_no_latent_variance(tree_corpus):
    from sklearn.gaussian_process import GaussianProcessRegressor

    from bayesian_optimization.bo import _latent

    terms = np.asarray(tree_corpus[:4], dtype=object)
    posterior = GaussianProcessRegressor(kernel=WhiteKernel(noise_level=NOISE), optimizer=None)
    posterior.fit(terms, [1.0, 2.0, 0.5, 1.5])
    _mean, observed = posterior.predict(terms, return_std=True)
    _mean, latent = _latent(posterior).predict(terms, return_std=True)
    assert np.all(observed > 0) and np.all(latent == 0)
    assert posterior.predict(terms, return_std=True)[1] == pytest.approx(observed), "left as it was"
