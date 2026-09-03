from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, NotRequired, TypedDict


class BOState(Enum):
    UNINITIALIZED = "UNINITIALIZED"
    INITIALIZED   = "INITIALIZED"
    SUGGESTED     = "SUGGESTED"
    OBSERVED      = "OBSERVED"
    FINALIZED     = "FINALIZED"


class Diagnostics(TypedDict, total=False):
    timestamp: float
    incumbent: float
    iteration: int
    fallback_used: bool
    fallback_attempts: int
    # The posterior mean and the posterior standard deviation at the picked term, the two
    # coordinates every acquisition score weighs against each other.  They are the per-pass half
    # of the trace readings: the acquisition value alone cannot tell an exploiting run (deviation
    # at zero) from an exploring one (deviation at the prior maximum), and the trace has to expose
    # both of those degenerate cases.
    mean_at_pick: float
    deviation_at_pick: float
    # One phase is left: "reinit_presample" belonged to the pool-and-thin initial sampling that
    # has since been removed, and an alternative nothing can produce is a reader's dead end.
    phase: NotRequired[Literal["main"]]


@dataclass(frozen=True)
class Suggestion:
    """Immutable record returned by BayesianOptimization.suggest().

    ``acquisition_value`` is the score of ``candidate``: the score of whatever is actually
    returned, not of what the acquisition optimizer proposed.  Where
    ``diagnostics["fallback_used"]`` is true the optimizer's result was an already-evaluated
    candidate and was replaced by a random fallback sample, so the value describes that
    replacement.  Read the two together or not at all.
    """

    candidate: Any
    acquisition_value: float | None = None
    diagnostics: Diagnostics | None = None
