from __future__ import annotations

import numpy as np
import pytest

TRANSFORM_CASES = [
    ("IdentityTransform",    np.array([0.0, 1.5, -3.2, 100.0])),
    ("Log1pTransform",       np.array([0.0, 0.5, 10.0, 1e3])),
    ("SignedLogTransform",   np.array([-5.0, 0.0, 5.0, 1e3])),
    ("StandardizeTransform", np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
]


def _get_transform(name):
    import importlib
    mod = importlib.import_module("bayesian_optimization.transforms")
    return getattr(mod, name)


@pytest.mark.parametrize("name,y", TRANSFORM_CASES, ids=[c[0] for c in TRANSFORM_CASES])
def test_transform_round_trip(name, y):
    cls = _get_transform(name)
    t = cls()
    reconstructed = t.inverse(t.forward(y))
    assert np.allclose(reconstructed, y, atol=1e-9), (
        f"{name}: round-trip failed; expected {y}, got {reconstructed}"
    )


def test_log1p_rejects_y_below_minus_one():
    from bayesian_optimization.transforms import Log1pTransform
    t = Log1pTransform()
    with pytest.raises(ValueError, match="-1"):
        t.forward(np.array([-2.0, 1.0]))


def test_log1p_clamps_tiny_negative_noise_silently():
    """Tiny values just below 0 (but above -1+tol) should not raise."""
    from bayesian_optimization.transforms import Log1pTransform
    t = Log1pTransform(clamp_tol=1e-12)
    y = np.array([-1e-15, 0.0, 1.0])
    result = t.forward(y)
    assert np.all(np.isfinite(result))


def test_signed_log_preserves_sign():
    from bayesian_optimization.transforms import SignedLogTransform
    t = SignedLogTransform()
    y = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    fwd = t.forward(y)
    assert np.sign(fwd[0]) == np.sign(y[0])
    assert np.sign(fwd[-1]) == np.sign(y[-1])
    assert fwd[2] == pytest.approx(0.0)


def test_identity_is_noop():
    from bayesian_optimization.transforms import IdentityTransform
    t = IdentityTransform()
    y = np.array([1.0, -2.5, 0.0, 100.0])
    assert np.allclose(t.forward(y), y)
    assert np.allclose(t.inverse(y), y)


def test_standardize_freezes_mean_std_on_first_call():
    from bayesian_optimization.transforms import StandardizeTransform
    t = StandardizeTransform()
    y1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t.forward(y1)
    frozen_mean = t._mean
    frozen_std = t._std

    y2 = np.array([10.0, 20.0, 30.0])
    fwd2 = t.forward(y2)
    expected = (y2 - frozen_mean) / frozen_std
    assert np.allclose(fwd2, expected)


def test_standardize_inverse_recovers_original():
    from bayesian_optimization.transforms import StandardizeTransform
    t = StandardizeTransform()
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_t = t.forward(y)
    y_back = t.inverse(y_t)
    assert np.allclose(y_back, y, atol=1e-9)


def test_standardize_inverse_before_forward_is_identity():
    """Calling inverse on a fresh StandardizeTransform returns the input unchanged."""
    from bayesian_optimization.transforms import StandardizeTransform
    t = StandardizeTransform()
    y = np.array([1.0, 2.0, 3.0])
    assert np.allclose(t.inverse(y), y)


def test_transform_protocol_runtime_check():
    """All four transforms must satisfy the YTransform Protocol."""
    from bayesian_optimization.transforms import (
        YTransform,
        IdentityTransform,
        Log1pTransform,
        SignedLogTransform,
        StandardizeTransform,
    )
    for cls in [IdentityTransform, Log1pTransform, SignedLogTransform, StandardizeTransform]:
        instance = cls()
        assert isinstance(instance, YTransform), f"{cls.__name__} does not satisfy YTransform protocol"
