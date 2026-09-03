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
    incumbent_transformed: float
    incumbent_raw: float
    y_transform: str
    iteration: int
    fallback_used: bool
    fallback_attempts: int
    phase: NotRequired[Literal["reinit_presample", "main"]]


@dataclass(frozen=True)
class Suggestion:
    """Immutable record returned by BayesianOptimization.suggest()."""

    candidate: Any
    acquisition_value: float | None = None
    diagnostics: Diagnostics | None = None
