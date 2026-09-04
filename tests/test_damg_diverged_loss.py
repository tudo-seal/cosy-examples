"""A network whose training diverged reports the loss it measured, not a stand-in for it.

``learner`` used to map a non-finite loss onto the constant 1e6 before returning it.  That made a
diverged run indistinguishable from a run that finished, and it made it indistinguishable in the
favorable direction: a regression loss is unbounded above, so 1e6 undercuts every honestly
measured loss beyond it, and the optimization loop ranks the smaller number first.

These tests pin both halves of the fix.  ``learner`` hands the measured value on, and the
optimization loop refuses it, which is the failure the substitution used to hide.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from bayesian_optimization.bo import _finite_or_raise
from bayesian_optimization.examples.damg_nas.damg_repo_algebras import learner

#: The value the removed sanitizer substituted for a non-finite loss.
FORMER_PENALTY = 1e6


def _adam(model: nn.Module) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=1e-2)


def _run(loss_value: float) -> float:
    """Train a one-parameter model for three epochs against a loss that answers ``loss_value``.

    The loss is a stub rather than a diverging network, because what is under test is the way
    ``learner`` passes a value on and not the arithmetic that produces one.  It stays attached to
    the model output so that the finite case can take the gradient step the diverged case skips.
    """
    def loss_fn(prediction, _target):
        return prediction.sum() * 0.0 + loss_value

    x = torch.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    y = x.ravel()
    return float(learner(1, nn.Linear(1, 1), loss_fn, _adam, 3, x, y, x, y))


@pytest.mark.parametrize("loss_value", [math.inf, -math.inf, math.nan])
def test_a_non_finite_loss_is_returned_as_it_was_measured(loss_value):
    """No substitute value, and in particular not the 1e6 the sanitizer used to return."""
    observed = _run(loss_value)
    assert not math.isfinite(observed)
    assert observed != FORMER_PENALTY


def test_a_finite_loss_is_returned_unchanged():
    """The path that trained normally is the one the removal must leave alone."""
    assert _run(0.25) == pytest.approx(0.25)


@pytest.mark.parametrize("loss_value", [math.inf, math.nan])
def test_the_optimization_loop_refuses_what_learner_now_reports(loss_value):
    """The loop rejects the value, so a diverged evaluation stops the run where it happened."""
    observed = _run(loss_value)
    with pytest.raises(ValueError, match="cannot be observed"):
        _finite_or_raise(observed, "term")


def test_the_former_penalty_would_have_passed_the_same_check():
    """1e6 is finite, so the substitution let a diverged network into the surrogate silently."""
    assert _finite_or_raise(FORMER_PENALTY, "term") == FORMER_PENALTY
