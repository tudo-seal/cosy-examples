"""The front of a run over two objectives, its hypervolume, and the best value under a cost.

A run of the loop records every evaluation as a
:class:`~bayesian_optimization.runs.term_pool.TermRecord`, the term with the metrics it came back
with.  Over two objectives, a value that is maximized and a cost that is minimized, what the run
found is its front: the evaluations no other one dominates, where one pair dominates another if its
value is at least as large and its cost at most as large, and one of the two strictly.  This module
reads the pairs off the records and answers the three readings a comparison of searches over value
and cost reports: which evaluations are on the front, the area they dominate within a reference
point, and the best value found at a cost of at most a threshold.

A failed measurement is refused rather than read: a value or a cost that is not finite raises,
naming the record, since a front with a substitute on it is a front nobody measured.  An
evaluation repeated under repeated measurements is a pair of its own, as it is a row of its own in
the run.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Sequence
from typing import Any

from ..diagnostics import get_logger
from .term_pool import TermRecord

_LOG = get_logger("runs.fronts")

__all__ = ["best_value_under", "front_hypervolume", "nondominated", "run_pairs"]


def run_pairs(
    records: Sequence[TermRecord],
    *,
    value: str | Callable[[TermRecord], float],
    cost: str | Callable[[TermRecord], float],
    phases: Collection[str] | None = None,
) -> list[tuple[float, float]]:
    """Read the pairs ``(value, cost)`` off a run's records, one per record, in their order.

    Args:
        records (Sequence[TermRecord]): The run's records, as :func:`read_term_pool` answers them.
        value (str | Callable[[TermRecord], float]): The metric that is the value, by name, or a
            function of the record.  Larger is better.
        cost (str | Callable[[TermRecord], float]): The metric that is the cost, by name, or a
            function of the record, such as an exact cost read off its term.  Smaller is better.
        phases (Collection[str] | None): The phases to read, every phase where ``None``.  A file
            from a run that wrote the loop's passes and a random arm into one holds two searches,
            and read whole its front is their union; that is said in a warning, and ``phases``
            names the phases of one arm.

    Returns:
        list[tuple[float, float]]: One pair per record read.

    Raises:
        KeyError: If a record does not carry a named metric, naming it and the record.
        ValueError: If a value or a cost is not a number, or not finite, naming the record.
    """
    if phases is None:
        present = {record.phase for record in records}
        if "random_sample" in present and len(present) > 1:
            _LOG.warning(
                "these records hold the phases %s, random_sample among them: a run that wrote the "
                "loop's passes and a random arm into one file holds two searches, and read whole "
                "its front is their union.  Name the phases of one arm with phases=.",
                sorted(present),
            )
    pairs = []
    for record in records:
        if phases is not None and record.phase not in phases:
            continue
        pair = (_read(record, value, "value"), _read(record, cost, "cost"))
        pairs.append(pair)
    return pairs


def _read(record: TermRecord, source: str | Callable[[TermRecord], float], what: str) -> float:
    name = f"the record {record.phase} {record.index} (repeat {record.repeat})"
    if isinstance(source, str):
        if source not in record.metrics:
            msg = f"{name} carries no {source!r}, the {what}"
            raise KeyError(msg)
        raw = record.metrics[source]
    else:
        raw = source(record)
    try:
        number = float(raw)
    except (TypeError, ValueError):
        msg = (
            f"the {what} of {name} is not a number: {raw!r}; an evaluation that answered none is "
            "refused rather than placed on a front"
        )
        raise ValueError(msg) from None
    if not math.isfinite(number):
        msg = (
            f"the {what} of {name} is not finite: {number}; a failed measurement is refused rather "
            "than placed on a front"
        )
        raise ValueError(msg)
    return number


def _require_finite_pairs(pairs: Sequence[tuple[float, float]]) -> None:
    """Refuse a pair that is not two finite numbers, naming its position.

    The readings below compare and subtract, and a value that is not finite answers neither
    sensibly: it would be dropped, or decide the order it was read in.  ``run_pairs`` never
    produces one.

    Args:
        pairs (Sequence[tuple[float, float]]): The pairs ``(value, cost)``.

    Raises:
        ValueError: Naming the first pair that is not finite.
    """
    for position, (value, cost) in enumerate(pairs):
        if not (math.isfinite(value) and math.isfinite(cost)):
            msg = (
                f"pair {position} {(value, cost)!r} is not finite; a front is read from "
                "measured pairs"
            )
            raise ValueError(msg)


def nondominated(pairs: Sequence[tuple[float, float]]) -> list[int]:
    """Return the positions of the pairs no other pair dominates, in increasing order.

    A pair dominates another if its value is at least as large and its cost at most as large, and
    one of the two strictly.  Two equal pairs therefore do not dominate each other, and both are on
    the front.

    Args:
        pairs (Sequence[tuple[float, float]]): The pairs ``(value, cost)``.

    Returns:
        list[int]: The positions of the non-dominated pairs.

    Raises:
        ValueError: If a pair is not finite.
    """
    _require_finite_pairs(pairs)
    # By increasing cost, so that a pair can only be dominated by one read before it or by one of
    # its own cost; within a cost, by decreasing value, so that the group's best comes first.
    order = sorted(range(len(pairs)), key=lambda i: (pairs[i][1], -pairs[i][0]))
    front: list[int] = []
    best_cheaper = -math.inf
    position = 0
    while position < len(order):
        cost = pairs[order[position]][1]
        group = []
        while position < len(order) and pairs[order[position]][1] == cost:
            group.append(order[position])
            position += 1
        top = pairs[group[0]][0]
        # The group's best is dominated by a cheaper pair of at least its value; the rest of the
        # group by its best, which has their cost and a larger value.
        if top > best_cheaper:
            front.extend(index for index in group if pairs[index][0] == top)
            best_cheaper = top
    return sorted(front)


def front_hypervolume(
    pairs: Sequence[tuple[float, float]], reference: tuple[float, float]
) -> float:
    """Return the area the pairs dominate within a reference point.

    A pair dominates the values from the reference value up to its own and the costs from its own
    up to the reference cost; the hypervolume is the area of the union of those boxes.  A pair at
    or below the reference value, or at or past the reference cost, adds nothing.

    Args:
        pairs (Sequence[tuple[float, float]]): The pairs ``(value, cost)``.
        reference (tuple[float, float]): The reference point ``(value, cost)``.

    Returns:
        float: The hypervolume.

    Raises:
        ValueError: If a pair or the reference point is not finite, or if the area exceeds the
            floats, which only a reference point far beyond the pairs produces.
    """
    _require_finite_pairs(pairs)
    reference_value, reference_cost = reference
    if not (math.isfinite(reference_value) and math.isfinite(reference_cost)):
        msg = f"the reference point {reference!r} is not finite; it bounds the area being measured"
        raise ValueError(msg)
    inside = sorted(
        (cost, value) for value, cost in pairs if value > reference_value and cost < reference_cost
    )
    area = 0.0
    level = reference_value
    for (cost, value), (next_cost, _) in zip(inside, [*inside[1:], (reference_cost, 0.0)]):
        level = max(level, value)
        if next_cost > cost:
            area += (next_cost - cost) * (level - reference_value)
    if not math.isfinite(area):
        msg = (
            f"the area the pairs dominate within the reference point {reference!r} exceeds the "
            "floats; set the reference point near the pairs"
        )
        raise ValueError(msg)
    return area


def best_value_under(
    pairs: Sequence[tuple[float, float]], thresholds: Sequence[float]
) -> list[float | None]:
    """Return, per threshold, the largest value at a cost of at most that threshold.

    Args:
        pairs (Sequence[tuple[float, float]]): The pairs ``(value, cost)``.
        thresholds (Sequence[float]): The costs.

    Returns:
        list[float | None]: One value per threshold, ``None`` where no pair is that cheap.

    Raises:
        ValueError: If a pair is not finite, or a threshold is not a number; an infinite
            threshold is a bound that admits every cost or none.
    """
    _require_finite_pairs(pairs)
    if any(math.isnan(threshold) for threshold in thresholds):
        msg = f"a threshold that is not a number bounds nothing: {list(thresholds)!r}"
        raise ValueError(msg)
    return [
        max((value for value, cost in pairs if cost <= threshold), default=None)
        for threshold in thresholds
    ]
