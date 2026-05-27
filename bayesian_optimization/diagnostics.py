from __future__ import annotations

import logging

from .state import Suggestion

_LOGGER = logging.getLogger("bayesian_optimization")


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``bayesian_optimization``."""
    return logging.getLogger(f"bayesian_optimization.{name}")


def log_iteration(
    logger: logging.Logger,
    *,
    iteration: int,
    suggestion: Suggestion,
    y_observed: float,
    suggest_time: float,
    eval_time: float,
    observe_time: float,
) -> None:
    """Emit a structured INFO-level log line for one BO iteration."""
    logger.info(
        "iter=%d  acq=%.4f  y=%.6g  "
        "t_suggest=%.3fs  t_eval=%.3fs  t_observe=%.3fs",
        iteration,
        suggestion.acquisition_value if suggestion.acquisition_value is not None else float("nan"),
        y_observed,
        suggest_time,
        eval_time,
        observe_time,
    )
