from __future__ import annotations

import logging

from ..state import Diagnostics, Suggestion

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
    to ``lastResort``, which is pinned at WARNING, so every INFO line this package emits is dropped
    and ``verbose=True`` is silent.  A handler is therefore attached, but only when the application
    configured none itself.  Where it did, its configuration wins.
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


def _posterior_field(diagnostics: Diagnostics | None, key: str) -> str:
    """Render one posterior coordinate at the pick, keeping absence visible."""
    if not diagnostics or key not in diagnostics:
        return "?"
    return f"{diagnostics[key]:.4g}"  # type: ignore[literal-required]


def log_suggestion(logger: logging.Logger, suggestion: Suggestion) -> None:
    """Emit a structured INFO-level log line for one ``suggest()`` call.

    The Ask/Tell interface has no loop of its own, so without this nothing reports what
    ``suggest()`` returned, and Ask/Tell is the interface the experiment scripts drive.  The
    fallback state belongs on this line because it decides how ``acq`` must be read: after a
    fallback the value describes a random replacement, not the optimizer's result.

    ``m`` and ``s`` are the pick's coordinates in the mean-deviation plane, where predicted value
    is weighed against uncertainty, and they are on the line because reading a run pass by pass
    needs them: an ``s`` pinned at zero and an ``s`` pinned at the prior maximum are the two
    degenerate runs, and neither is visible in the acquisition value alone.
    """
    diagnostics = suggestion.diagnostics or {}
    incumbent = diagnostics.get("incumbent")
    logger.info(
        "suggest  iter=%s  acq=%.4f  m=%s  s=%s  fallback=%s  incumbent=%s",
        diagnostics.get("iteration", "?"),
        _acquisition_field(suggestion),
        _posterior_field(suggestion.diagnostics, "mean_at_pick"),
        _posterior_field(suggestion.diagnostics, "deviation_at_pick"),
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
        "iter=%d  acq=%.4f  m=%s  s=%s  y=%.6g  fallback=%s  "
        "t_suggest=%.3fs  t_eval=%.3fs  t_observe=%.3fs",
        iteration,
        _acquisition_field(suggestion),
        _posterior_field(suggestion.diagnostics, "mean_at_pick"),
        _posterior_field(suggestion.diagnostics, "deviation_at_pick"),
        y_observed,
        _fallback_field(suggestion.diagnostics),
        suggest_time,
        eval_time,
        observe_time,
    )


def warn_if_exploitation_stalls(
    logger: logging.Logger,
    *,
    iteration: int,
    acquisition_value: float,
    lower_bound: float | None,
    acquisition_name: str,
) -> bool:
    """Say so when the maximization found nothing that promises anything.

    The symptom chain of an overconfident surrogate runs: the deviations it reports are too small,
    expected improvement vanishes everywhere, the loop repeats exploitative picks, the best
    observed value stalls.  Its cause is three steps away from where it shows, which is why it is
    reported here instead of repaired silently.  There is nothing here to repair, and a loop that
    quietly widened its own uncertainty to keep exploring would be reporting a surrogate it does
    not have.

    The condition is the acquisition's own lower bound, reached exactly.  Expected improvement and
    probability of improvement are bounded below by zero, so the maximum over the whole space
    being zero says that no term promises an improvement at all, a statement about the space and
    the surrogate, with no threshold chosen by anyone.  A value that is merely very small is the
    same failure in a graded form, and it is read off the trace instead
    (:class:`TraceRead.acquisition_trend`), where a scale is available to compare against.  The
    upper confidence bound has no lower bound and therefore no such moment.

    Args:
        logger (logging.Logger): Where to say it.
        iteration (int): Which pass.
        acquisition_value (float): The value at the pick.
        lower_bound (float | None): The acquisition's lower bound, where its definition fixes one.
        acquisition_name (str): Its name, for the message.

    Returns:
        bool: Whether the warning fired, so that a caller can hold it to once per run.
    """
    if lower_bound is None or acquisition_value > lower_bound:
        return False
    logger.warning(
        "iteration %d: the best %s over the whole search space is %.4g, its smallest possible "
        "value, so no candidate promises an improvement over the incumbent.  This is the "
        "overconfidence chain: posterior deviations too small, the acquisition flat at its floor, "
        "the loop exploiting and the best observed value stalling.  Read the standardized "
        "leave-one-out residuals (read_calibration): a spread far above one confirms it.  The "
        "repair is a kernel that carries its own scale together with a kernel_optimizer to fit "
        "it, because kernel_optimizer alone has nothing to move on a kernel that declares no "
        "hyperparameters.  On a finite space, exhaustion produces the same reading and is a "
        "legitimate end state.",
        iteration,
        acquisition_name,
        acquisition_value,
    )
    return True
