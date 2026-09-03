from __future__ import annotations

import logging

import numpy as np
import pytest


def test_the_gp_is_fitted_on_the_observations_themselves(bo_factory, tree_corpus):
    """No transformation sits between the caller's objective and the model.

    The layer that used to sit there applied ``log1p`` by default, the opposite direction to the
    ``exp`` of a scalarization, the order-preserving map from fitness values into the positive
    reals.  It has no counterpart in the thesis's Bayesian optimization at all.  ``normalize_y``
    is switched off here so the assertion is about the values and not about sklearn's internal
    centering.
    """
    bo = bo_factory(gp_normalize_y=False)
    y0 = [1.0, 2.0, 0.5]
    bo.initialize(x0=tree_corpus[:3], y0=y0)
    bo.suggest()

    assert bo._model is not None
    assert np.allclose(bo._model.y_train_, np.array(y0))


def test_gp_normalize_y_is_undone_on_predict(bo_factory, tree_corpus):
    """sklearn's ``normalize_y`` is a numerical convenience, not a change of scale.

    It centers and scales the targets for the fit and undoes it on predict, so predictions come
    back on the scale the observations were given in.  That is why it can stay while the
    transformation layer goes.
    """
    bo = bo_factory(gp_normalize_y=True)
    y0 = [10.0, 20.0, 5.0]
    bo.initialize(x0=tree_corpus[:3], y0=y0)
    bo.suggest()

    assert bo._model is not None
    predicted = bo._model.predict(np.asarray(tree_corpus[:3], dtype=object))
    assert np.allclose(predicted, np.array(y0), atol=1e-3)


def test_y_list_holds_what_the_caller_supplied(bo_factory, tree_corpus):
    bo = bo_factory()
    y0 = [10.0, 20.0, 5.0]
    bo.initialize(x0=tree_corpus[:3], y0=y0)
    s = bo.suggest()
    bo.observe(s.candidate, 99.0)
    snap = bo.get_state_snapshot()
    assert np.allclose(snap["y_list"][:3], y0)
    assert snap["y_list"][-1] == pytest.approx(99.0)


def test_incumbent_is_the_largest_observation(bo_factory, tree_corpus):
    """The incumbent is the largest observation, the ``y*`` of expected improvement."""
    bo = bo_factory()
    y0 = [3.0, 1.0, 5.0]
    bo.initialize(x0=tree_corpus[:3], y0=y0)
    s = bo.suggest()
    assert s.diagnostics is not None
    assert s.diagnostics["incumbent"] == pytest.approx(max(y0))


def test_finalize_best_y_is_the_maximum(bo_factory, tree_corpus):
    bo = bo_factory()
    y0 = [2.0, 5.0, 1.0]
    bo.initialize(x0=tree_corpus[:3], y0=y0)
    s = bo.suggest()
    bo.observe(s.candidate, 7.0)
    result = bo.finalize()
    assert result["best_y"] == pytest.approx(max([*y0, 7.0]))


def test_heavy_tailed_observations_still_fit(bo_factory, tree_corpus):
    """Smoke test: a wide range of objective values does not break the fit.

    It used to be the argument for the default ``log1p``.  What handles it now is the GP's own
    ``normalize_y``, and a caller who wants a log scale writes it into their objective.
    """
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 100.0, 10000.0])
    s = bo.suggest()
    assert s is not None
    assert s.acquisition_value is not None


def test_model_selection_on_a_kernel_with_nothing_to_fit_says_so(bo_factory, tree_corpus, caplog):
    """sklearn runs an optimizer over an empty parameter vector without a word.

    That silence is expensive: on a counting kernel whose amplitudes stay at their constructed
    values the GP is over-confident, EI underflows, the search stops exploring, and none of those
    symptoms points back at the kernel.  Model selection is the default now, so the case where it
    cannot do anything has to be audible.
    """
    from bayesian_optimization.kernels import OrderedRootedSubtreeKernel

    kernel = OrderedRootedSubtreeKernel()
    assert kernel.theta.size == 0, "this test is about a kernel with no hyperparameters"

    bo = bo_factory(kernel=kernel)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])

    with caplog.at_level(logging.WARNING, logger="bayesian_optimization"):
        bo.suggest()
        bo.observe(bo._last_suggestion.candidate, 0.3)
        bo.suggest()

    warnings = [r.getMessage() for r in caplog.records if "declares no hyperparameters" in r.getMessage()]
    assert len(warnings) == 1, f"expected exactly one warning per run, got {len(warnings)}"
    assert "OrderedRootedSubtreeKernel" in warnings[0]
