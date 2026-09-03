from __future__ import annotations

from collections.abc import Sequence
from dataclasses import astuple, dataclass, fields

import numpy as np

from ._statistics import rank_correlation, spread


@dataclass(frozen=True)
class TraceRecord:
    """One pass of the loop, as the columns a run log carries.

    Only numbers and flags: a trace is written per iteration and read as a table, and the terms it
    would otherwise carry belong to the dataset the run returns.

    Attributes:
        iteration (int): Which pass this was, counted from zero.
        acquisition (float): The acquisition value at the term the pass picked.  After a duplicate
            fallback it describes the replacement rather than the maximization's result, which is
            what :attr:`fallback_used` is beside it for.
        mean (float): ``m_D`` at the pick, from the surrogate of that pass.
        deviation (float): ``s_D`` at the pick.
        incumbent (float): The largest value observed *before* the pass, the ``y*`` the
            acquisition was built with.
        observed (float): The value the objective returned at the pick.
        best (float): The largest value observed after the pass.
        fallback_used (bool): Whether the maximization returned an already evaluated term and the
            loop replaced it.
        evaluations (int): How many evaluations the dataset holds after the pass, repetitions
            included, the initial design among them.
        distinct_observations (int): How many *distinct* terms it holds.  Both are read off the
            dataset after the pass rather than derived from the pass number, which is what lets
            them differ.
    """

    iteration: int
    acquisition: float
    mean: float
    deviation: float
    incumbent: float
    observed: float
    best: float
    fallback_used: bool
    evaluations: int
    distinct_observations: int


def trace_columns() -> tuple[str, ...]:
    """Return the column names of a trace table, in the order :func:`trace_rows` writes them.

    Returns:
        tuple[str, ...]: The names, taken from :class:`TraceRecord` so that the two cannot drift.
    """
    return tuple(field.name for field in fields(TraceRecord))


def trace_rows(records: Sequence[TraceRecord]) -> list[tuple[object, ...]]:
    """Return a trace as rows, ready for ``csv.writer``.

    Args:
        records (Sequence[TraceRecord]): The trace.

    Returns:
        list[tuple[object, ...]]: One row per record, matching :func:`trace_columns`.
    """
    return [astuple(record) for record in records]


@dataclass(frozen=True)
class TraceRead:
    """What a whole run says about itself, read off its best-value curve and its per-pass traces.

    Attributes:
        passes (int): How many passes the trace covers.
        acquisition_trend (float | None): The rank correlation of the acquisition value at the
            pick against the iteration number.  It should be negative: as the surrogate learns,
            the best attainable improvement shrinks.  A trend that never decays is the degenerate
            landscape, where the loop keeps finding the same promise and never collects it.
            ``None`` where the acquisition value never changed, which is that degeneration in its
            extreme form.
        acquisition_first (float): The value at the first pick, and
        acquisition_last (float): at the last.  An expected improvement of about zero at the pick
            from the first pass on is overconfidence (read :func:`read_calibration` next) or an
            incumbent that is not the maximum.
        acquisition_at_zero (int): How many passes picked a term whose acquisition value was zero.
            For expected improvement and probability of improvement, both bounded below by zero,
            that is a pass in which nothing in the space promised anything.
        deviation_spread (float): The range of ``s_D`` at the pick over the run.  Zero says the
            deviation was pinned, and there are two ways to pin it, each its own failure: at zero
            the loop only re-examines what it knows, at the prior maximum it explores and never
            exploits.
        deviation_minimum (float): The smallest ``s_D`` at a pick, and
        deviation_maximum (float): the largest, which says which of the two pins it was.
        fallbacks (int): How many passes ended in the duplicate fallback.  A run in which this
            fires at every iteration is random search rather than Bayesian optimization.
        distinct_fraction (float): Distinct terms in the dataset over evaluations spent on it,
            both counted after the last pass.  One means no evaluation was ever repeated.

            Under the rejection path of this implementation every term the *loop* proposes is
            novel, so a value below one comes from the dataset it was handed: an ``x0`` that
            repeats a term, or an ask/tell caller observing one twice.  That is the case worth
            seeing, and it is the case an earlier version of this field could not report: it
            derived the count from the pass number instead of from the dataset, which made it one
            by construction on every trace.
        improvements (int): How many passes raised the best observed value above the incumbent
            they started from.
        stalled_passes (int): How many passes have run since the last one that did, so the whole
            trace where none did.  A stalling best observed value is the downstream symptom of an
            overconfident surrogate, though on a finite space it is also what exhaustion looks
            like, and there it is a legitimate end state rather than a fault.
        best_trace (tuple[float, ...]): The best observed value after each pass, for the curve.
    """

    passes: int
    acquisition_trend: float | None
    acquisition_first: float
    acquisition_last: float
    acquisition_at_zero: int
    deviation_spread: float
    deviation_minimum: float
    deviation_maximum: float
    fallbacks: int
    distinct_fraction: float
    improvements: int
    stalled_passes: int
    best_trace: tuple[float, ...]


def read_trace(records: Sequence[TraceRecord]) -> TraceRead:
    """Read a finished run as a whole, from its best-value curve and its per-pass traces.

    The fifth reading, and the only one that needs a whole run rather than one pass of it.  It is
    the last of the checks in order of what it can conclude on its own: every degeneration it
    names has its cause in one of the four other reads, and this one says which run to take them
    to.

    Args:
        records (Sequence[TraceRecord]): The trace, one record per pass, in order.

    Returns:
        TraceRead: The measurements.

    Raises:
        ValueError: If the trace is empty, since a run with no pass has no reading, or if it holds
            a value that is not finite, which no finished pass produces: the loop refuses a
            non-finite observation where it is made.
    """
    if not records:
        msg = "a trace read summarizes the passes of a run, and this trace holds none"
        raise ValueError(msg)

    acquisitions = np.array([record.acquisition for record in records], dtype=float)
    deviations = np.array([record.deviation for record in records], dtype=float)
    best = np.array([record.best for record in records], dtype=float)
    # Every real column, not only the three this function goes on to average.  A non-finite
    # incumbent is the one that hides: `best > nan` is false, so the improvement count silently
    # drops the passes it touches instead of reporting that the trace is broken.
    for name in ("acquisition", "mean", "deviation", "incumbent", "observed", "best"):
        column = np.array([getattr(record, name) for record in records], dtype=float)
        if not np.all(np.isfinite(column)):
            msg = f"the trace holds a {name} that is not finite"
            raise ValueError(msg)
    if any(record.evaluations < 1 for record in records):
        msg = (
            "a completed pass has evaluated at least its own term, so a trace row cannot report "
            "fewer than one evaluation"
        )
        raise ValueError(msg)
    if any(
        record.distinct_observations > record.evaluations for record in records
    ):
        msg = (
            "a trace row reports more distinct terms than evaluations, which no dataset can hold"
        )
        raise ValueError(msg)

    passes = len(records)
    iterations = [record.iteration for record in records]

    # A pass improved when what it found beats what stood before it, which is the record's own
    # incumbent, not the previous row's best.  Comparing row against row would silently exempt the
    # first pass, and the first pass is the one most likely to beat a small initial design.
    improved = [
        index for index, record in enumerate(records) if record.best > record.incumbent
    ]
    last_improvement = improved[-1] if improved else -1

    return TraceRead(
        passes=passes,
        # One pass is a point, and a point has no trend.  Reported as absent rather than as zero:
        # zero is what a run that genuinely never decays produces, and the two are different runs.
        acquisition_trend=(
            rank_correlation(iterations, acquisitions) if passes > 1 else None
        ),
        acquisition_first=float(acquisitions[0]),
        acquisition_last=float(acquisitions[-1]),
        acquisition_at_zero=int(np.count_nonzero(acquisitions == 0.0)),
        deviation_spread=spread(deviations),
        deviation_minimum=float(np.min(deviations)),
        deviation_maximum=float(np.max(deviations)),
        fallbacks=sum(1 for record in records if record.fallback_used),
        distinct_fraction=records[-1].distinct_observations / records[-1].evaluations,
        improvements=len(improved),
        stalled_passes=passes - 1 - last_improvement,
        best_trace=tuple(float(value) for value in best),
    )
