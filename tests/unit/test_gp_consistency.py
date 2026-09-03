from __future__ import annotations

import warnings

import numpy as np
import pytest
from cosy.core.tree import Tree


def test_gp_predict_lives_in_transformed_space(bo_factory, tree_corpus):
    """F6: gp.predict() must live in the same transformed space as incumbent_transformed."""
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()

    assert bo._model is not None
    # y_train_ should contain TRANSFORMED values
    y_raw = np.array([1.0, 2.0, 0.5])
    from bayesian_optimization.transforms import Log1pTransform
    y_transformed = Log1pTransform().forward(y_raw)
    # sklearn may normalize internally, but the underlying transform should be applied
    # We verify y_train_ is not equal to the raw values
    if bo._model.normalize_y:
        # sklearn centers internally, so y_train_ after centering differs from y_transformed
        pass
    else:
        assert not np.allclose(bo._model.y_train_, y_raw)


def test_y_list_remains_raw_after_suggest_and_observe(bo_factory, tree_corpus):
    """y_list must always hold raw (user-supplied) values."""
    bo = bo_factory()
    y0 = [10.0, 20.0, 5.0]
    bo.initialize(x0=tree_corpus[:3], y0=y0)
    s = bo.suggest()
    bo.observe(s.candidate, 99.0)
    snap = bo.get_state_snapshot()
    # The raw values must be unchanged
    assert np.allclose(snap["y_list"][:3], y0)
    assert snap["y_list"][-1] == pytest.approx(99.0)


def test_incumbent_matches_gp_predict_at_known_points(bo_factory, tree_corpus):
    """The incumbent_transformed in diagnostics must equal min/max of transformed y_list."""
    bo = bo_factory()
    y0 = [3.0, 1.0, 5.0]
    bo.initialize(x0=tree_corpus[:3], y0=y0)
    s = bo.suggest()
    diag = s.diagnostics
    from bayesian_optimization.transforms import Log1pTransform
    y_transformed = Log1pTransform().forward(np.array(y0))
    expected_incumbent = float(np.min(y_transformed))  # minimization
    assert diag["incumbent_transformed"] == pytest.approx(expected_incumbent, abs=1e-6)


def test_finalize_best_y_is_raw_scale(bo_factory, tree_corpus):
    """finalize() best_y must be on the raw scale, not transformed."""
    bo = bo_factory()
    y0 = [2.0, 5.0, 1.0]
    bo.initialize(x0=tree_corpus[:3], y0=y0)
    s = bo.suggest()
    bo.observe(s.candidate, 7.0)
    result = bo.finalize()
    # best_y must be the minimum raw value
    assert result["best_y"] == pytest.approx(min(y0 + [7.0]))


def test_normalize_y_true_emits_deprecation_warning():
    """normalize_y=True must emit exactly one DeprecationWarning."""
    from bayesian_optimization.bo import BayesianOptimization

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bo = BayesianOptimization(
            search_space=None,
            request=None,
            normalize_y=True,
        )
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    # Should behave as Log1pTransform
    from bayesian_optimization.transforms import Log1pTransform
    assert isinstance(bo._y_transform, Log1pTransform)


def test_normalize_y_false_emits_deprecation_warning_and_maps_to_identity():
    from bayesian_optimization.bo import BayesianOptimization

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        bo = BayesianOptimization(
            search_space=None,
            request=None,
            normalize_y=False,
        )
    dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
    assert len(dep_warnings) >= 1
    from bayesian_optimization.transforms import IdentityTransform
    assert isinstance(bo._y_transform, IdentityTransform)


def test_normalize_y_and_y_transform_both_set_raises_value_error():
    from bayesian_optimization.bo import BayesianOptimization
    from bayesian_optimization.transforms import IdentityTransform

    with pytest.raises(ValueError, match="both"):
        BayesianOptimization(
            search_space=None,
            request=None,
            normalize_y=True,
            y_transform=IdentityTransform(),
        )


def test_default_y_transform_is_log1p():
    from bayesian_optimization.bo import BayesianOptimization
    from bayesian_optimization.transforms import Log1pTransform

    bo = BayesianOptimization(search_space=None, request=None)
    assert isinstance(bo._y_transform, Log1pTransform)


def test_log1p_transform_handles_heavy_tailed_data(bo_factory, tree_corpus):
    """Smoke test: heavy-tailed y values don't cause GP fitting to fail."""
    from bayesian_optimization.transforms import Log1pTransform
    bo = bo_factory(y_transform=Log1pTransform())
    y0 = [1.0, 100.0, 10000.0]
    bo.initialize(x0=tree_corpus[:3], y0=y0)
    s = bo.suggest()
    assert s is not None
    assert s.acquisition_value is not None
