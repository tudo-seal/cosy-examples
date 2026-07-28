from __future__ import annotations

import logging

from .state import Diagnostics, Suggestion

_LOGGER = logging.getLogger("bayesian_optimization")


def get_logger(name: str) -> logging.Logger:
    """Return a logger namespaced under ``bayesian_optimization``."""
    return logging.getLogger(f"bayesian_optimization.{name}")


def _has_reachable_handler(logger: logging.Logger) -> bool:
    """Whether a record logged here would reach any handler at all.

    Mirrors ``Logger.callHandlers``: an ancestor's handler counts, and ``propagate=False`` ends the
    walk.  Inspecting ``logger.handlers`` alone misses the first case, inspecting the root alone
    misses the second.
    """
    current: logging.Logger | None = logger
    while current is not None:
        if current.handlers:
            return True
        if not current.propagate:
            return False
        current = current.parent
    return False


def enable_verbose_logging() -> None:
    """Make ``verbose=True`` actually produce output.

    Raising the level is not enough.  With no handler anywhere on the chain, ``logging`` falls back
    to ``lastResort``, which is pinned at WARNING -- so every INFO line this package emits is
    dropped and ``verbose=True`` is silent.  A handler is therefore attached, but only when the
    application configured none itself; where it did, its configuration wins.
    """
    _LOGGER.setLevel(logging.INFO)
    if _has_reachable_handler(_LOGGER):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s  %(name)s  %(levelname)s  %(message)s"))
    _LOGGER.addHandler(handler)


def _fallback_field(diagnostics: Diagnostics | None) -> str:
    """Render the fallback state, keeping "none" and "not reported" apart.

    ``?`` rather than ``no`` when the information is absent: claiming that no fallback occurred, on
    a record that never carried the answer, is exactly the kind of silent substitute value this
    package refuses everywhere else.
    """
    if not diagnostics or "fallback_used" not in diagnostics:
        return "?"
    if not diagnostics["fallback_used"]:
        return "no"
    return f"yes({diagnostics.get('fallback_attempts', '?')})"


def _acquisition_field(suggestion: Suggestion) -> float:
    return (
        suggestion.acquisition_value
        if suggestion.acquisition_value is not None
        else float("nan")
    )


def log_suggestion(logger: logging.Logger, suggestion: Suggestion) -> None:
    """Emit a structured INFO-level log line for one ``suggest()`` call.

    The Ask/Tell interface has no loop of its own, so without this nothing reports what
    ``suggest()`` returned -- and Ask/Tell is the interface the experiment scripts drive.  The
    fallback state belongs on this line because it decides how ``acq`` must be read: after a
    fallback the value describes a random replacement, not the optimizer's result.
    """
    diagnostics = suggestion.diagnostics or {}
    incumbent = diagnostics.get("incumbent_raw")
    logger.info(
        "suggest  iter=%s  acq=%.4f  fallback=%s  incumbent_raw=%s",
        diagnostics.get("iteration", "?"),
        _acquisition_field(suggestion),
        _fallback_field(suggestion.diagnostics),
        "?" if incumbent is None else f"{incumbent:.6g}",
    )


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
        "iter=%d  acq=%.4f  y=%.6g  fallback=%s  "
        "t_suggest=%.3fs  t_eval=%.3fs  t_observe=%.3fs",
        iteration,
        _acquisition_field(suggestion),
        y_observed,
        _fallback_field(suggestion.diagnostics),
        suggest_time,
        eval_time,
        observe_time,
    )
