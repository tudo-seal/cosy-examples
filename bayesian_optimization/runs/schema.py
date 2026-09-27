"""What a run records about each evaluation, and which of its numbers the loop maximizes.

An evaluation answers with a mapping of metrics, whatever the caller measures: an accuracy and a
loss on CIFAR, an F-score and a parameter count elsewhere.  :class:`Objective` names the one metric
the loop maximizes and which way is better, and turns it into the value the loop is handed and
back.  :class:`MetricSchema` names the columns of the run's CSV, in their order, each either a
field the loop knows about every evaluation or one of the caller's metrics.

The CIFAR driver's layout is one schema among others (``CIFAR_SCHEMA`` in its example); a caller
with other metrics writes those, and nothing forces the CIFAR keys on it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..state import Suggestion

#: The fields the loop knows about every evaluation, which a column can show instead of a metric.
LOOP_FIELDS: tuple[str, ...] = (
    "phase",
    "index",
    "structure",
    "term_size",
    "loop_value",
    "taken_over",
    "acquisition_seconds",
    "acquisition_value",
    "fallback_used",
    "fallback_attempts",
    "timestamp",
)


@dataclass(frozen=True)
class Objective:
    """The metric the loop maximizes, and which way is better.

    The loop maximizes throughout, so a metric that is better when smaller enters it negated; the
    one conversion lives here, in both directions, so that a number read off the loop and a number
    written into it cannot disagree about the sign.

    Attributes:
        key (str): The metric of an evaluation's mapping that is the objective.
        greater_is_better (bool): Whether larger values of it are better. (Default value = True)
    """

    key: str
    greater_is_better: bool = True

    def loop_value(self, metrics: Mapping[str, Any]) -> float:
        """The value the loop is handed for an evaluation with these metrics.

        Args:
            metrics (Mapping[str, Any]): What the evaluation answered.

        Returns:
            float: The objective, negated where smaller is better.

        Raises:
            KeyError: If the metrics do not carry the objective, naming its key.
        """
        if self.key not in metrics:
            msg = f"the evaluation reported no {self.key!r}, the metric this run maximizes"
            raise KeyError(msg)
        value = float(metrics[self.key])
        return value if self.greater_is_better else -value

    def as_reported(self, value: float) -> float:
        """The metric in its own sign, from the value the loop holds.

        Args:
            value (float): A value the loop was handed or reports.

        Returns:
            float: The metric as the evaluation measured it.
        """
        return value if self.greater_is_better else -value


@dataclass(frozen=True)
class Column:
    """One column of a run's CSV.

    Attributes:
        name (str): The header.
        field (str | None): The loop field the column shows, one of :data:`LOOP_FIELDS`; ``None``
            shows the caller's metric of the same name. (Default value = None)
        render (Callable[[Any], Any] | None): How a metric's value becomes a cell, given
            ``None`` where the evaluation did not report it; ``None`` writes the value as it is and
            an empty cell where it is missing. (Default value = None)
    """

    name: str
    field: str | None = None
    render: Callable[[Any], Any] | None = None

    def __post_init__(self) -> None:
        if self.field is not None and self.field not in LOOP_FIELDS:
            msg = f"{self.field!r} is not a loop field; the loop knows {', '.join(LOOP_FIELDS)}"
            raise ValueError(msg)


def _cell(value: Any) -> Any:
    return "" if value is None else value


@dataclass(frozen=True)
class MetricSchema:
    """The columns of a run's CSV in their order, the objective, and the live line of a run.

    Attributes:
        objective (Objective): The metric the loop maximizes.
        columns (tuple[Column, ...]): The CSV's columns, in order.
        live (Callable[[Mapping[str, Any]], str] | None): The text a driver prints after the
            objective for each evaluation; ``None`` prints every metric column. (Default value = None)
    """

    objective: Objective
    columns: tuple[Column, ...]
    live: Callable[[Mapping[str, Any]], str] | None = None

    def __post_init__(self) -> None:
        names = [column.name for column in self.columns]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            msg = f"a CSV header names each column once; {', '.join(repeated)} appear more than once"
            raise ValueError(msg)

    @classmethod
    def for_metrics(
        cls,
        objective: Objective,
        metrics: Sequence[str],
        *,
        live: Callable[[Mapping[str, Any]], str] | None = None,
    ) -> MetricSchema:
        """The default layout: where the evaluation stands, the loop's value, the metrics, the loop.

        Args:
            objective (Objective): The metric the loop maximizes.
            metrics (Sequence[str]): The caller's metrics, one column each, in this order.
            live (Callable | None): See the attribute. (Default value = None)

        Returns:
            MetricSchema: ``phase, index, structure, loop_value, taken_over``, the metrics, then
                ``term_size``, ``acquisition_seconds``, ``acquisition_value``, ``fallback_used``,
                ``timestamp``.
        """
        head = [
            Column(name, field=name)
            for name in ("phase", "index", "structure", "loop_value", "taken_over")
        ]
        body = [Column(name) for name in metrics]
        tail = [
            Column(name, field=name)
            for name in ("term_size", "acquisition_seconds", "acquisition_value", "fallback_used",
                         "timestamp")
        ]
        return cls(objective, tuple(head + body + tail), live)

    @property
    def header(self) -> list[str]:
        """The CSV's header row."""
        return [column.name for column in self.columns]

    def row(
        self,
        *,
        phase: str,
        index: Any,
        structure: Any,
        term_size: int,
        metrics: Mapping[str, Any],
        timestamp: float,
        loop_value: float | None = None,
        suggestion: Suggestion | None = None,
        acquisition_seconds: float | None = None,
        taken_over: bool = False,
    ) -> list[Any]:
        """One CSV row, in the order of the columns.

        A metric the evaluation did not report is an empty cell rather than an error, so a
        partially instrumented run still records its structures.  A row without a suggestion, a
        term of the design, leaves the acquisition cells empty: an empty cell says there was no
        acquisition step, where ``False`` would claim a fallback was ruled out that never was.
        ``taken_over`` says the row's value came from the record of an earlier run rather than
        from an evaluation this run made, so a resumed row is not read as a measurement of it.

        Returns:
            list[Any]: The cells.
        """
        diagnostics = (suggestion.diagnostics or {}) if suggestion is not None else {}
        fields: dict[str, Any] = {
            "phase": phase,
            "index": index,
            "structure": structure,
            "term_size": term_size,
            "loop_value": _cell(loop_value),
            "taken_over": taken_over,
            "acquisition_seconds": _cell(acquisition_seconds),
            "acquisition_value": "" if suggestion is None else _cell(suggestion.acquisition_value),
            "fallback_used": diagnostics.get("fallback_used", ""),
            "fallback_attempts": diagnostics.get("fallback_attempts", ""),
            "timestamp": timestamp,
        }
        cells = []
        for column in self.columns:
            if column.field is not None:
                cells.append(fields[column.field])
            elif column.render is not None:
                cells.append(column.render(metrics.get(column.name)))
            else:
                cells.append(metrics.get(column.name, ""))
        return cells

    def live_line(self, metrics: Mapping[str, Any]) -> str:
        """The text after the objective on a driver's line for one evaluation.

        Args:
            metrics (Mapping[str, Any]): What the evaluation answered.

        Returns:
            str: ``live(metrics)``, or every reported metric column as ``name=value``.
        """
        if self.live is not None:
            return self.live(metrics)
        parts = []
        for column in self.columns:
            if column.field is None and column.name in metrics:
                value = metrics[column.name]
                shown = f"{value:.5g}" if isinstance(value, float) and math.isfinite(value) else value
                parts.append(f"{column.name}={shown}")
        return " ".join(parts)
