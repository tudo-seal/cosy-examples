from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, NotRequired, TypedDict


class BOState(Enum):
    UNINITIALIZED = "UNINITIALIZED"
    # The initial design is drawn, or handed over as terms, and some of it still awaits its
    # values: ``suggest()`` hands the next design term out and ``observe()`` takes its value back.
    DESIGN        = "DESIGN"
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
    # "main" is a pass of the loop; "design" is a term of the initial design handed out by the
    # design phase, which maximizes nothing and so carries no acquisition reading.
    phase: NotRequired[Literal["main", "design"]]
    # The position of a design term in the design, only on a suggestion of the design phase.
    design_index: NotRequired[int]


@dataclass(frozen=True)
class Suggestion:
    """Immutable record returned by BayesianOptimization.suggest().

    ``acquisition_value`` is the score of ``candidate``: the score of whatever is actually
    returned, not of what the acquisition optimizer proposed.  Where
    ``diagnostics["fallback_used"]`` is true the optimizer's result was an already-evaluated
    candidate and was replaced by a random fallback sample, so the value describes that
    replacement.  Read the two together or not at all.  A term of the design phase is scored by
    nothing: it carries ``None`` here and ``phase == "design"`` in its diagnostics.
    """

    candidate: Any
    acquisition_value: float | None = None
    diagnostics: Diagnostics | None = None
