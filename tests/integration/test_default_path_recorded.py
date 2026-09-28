"""The default path of the loop and of the run layer, held against a recording.

The trajectory tests compare two runs of the same code, so a change that moves every run the same
way passes them all.  This module records once what the default configurations compute -- every
term the loop is handed, the values it read at each pick, the surrogate it fitted, and the files the
run layer wrote -- and holds each later commit against that recording.  It exists for the changes
that add options to the core, whose defaults have to stay what the loop did before them.

Marked slow, so the CI jobs skip it: a run is exact on one machine, in-process and across processes,
but whether another machine's linear algebra returns the same floats has not been measured, and a
flipped pick fails every value after it.  Run it before and after each change to the core, exactly
with ``DEFAULT_PATH_EXACT=1``; without it the floats are held at a relative 1e-9.  The recording is
rewritten only on purpose: ``python -m tests.integration.test_default_path_recorded``.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Literal
from unittest import mock

import pytest
from cosy.core import Synthesizer
from cosy.core.tree import Tree

from bayesian_optimization import RandomSearch
from bayesian_optimization.acquisition_optimizer import AcquisitionOptimizer
from bayesian_optimization.runs import read_term_pool, run_paired, run_search, twin_sampler
from tests.integration.test_real_ea_loop import _TARGET, _make_bo, _objective, _repository
from tests.spaces import LIST, list_space
from tests.unit.test_run_driver import SCHEMA, _bo, _metrics

RECORDING = Path(__file__).with_name("default_path_recorded.json")

# What a run measures of the wall clock rather than computes.
_WALL_CLOCK = frozenset({"timestamp", "acquisition_seconds", "seconds", "created"})

Acquisition = Literal["ExpectedImprovement", "ProbabilityOfImprovement", "UpperConfidenceBound"]
_ACQUISITIONS: tuple[Acquisition, ...] = (
    "ExpectedImprovement", "ProbabilityOfImprovement", "UpperConfidenceBound",
)


def _term(tree: Tree[Any]) -> str:
    """A term by the names of its labels: a label that is a function renders by its name, not by
    the address ``repr`` gives it, which differs from one process to the next."""
    root = tree.root
    label = root if isinstance(root, str) else getattr(root, "__name__", None) or repr(root)
    if not tree.children:
        return label
    return f"{label}({', '.join(_term(child) for child in tree.children)})"


def _plain(value: Any) -> Any:
    """A JSON value for what a run hands back: a term by its labels, an array as a list."""
    if isinstance(value, Tree):
        return _term(value)
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items() if key not in _WALL_CLOCK}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(dataclasses.asdict(value))
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _cell(text: str) -> Any:
    """A CSV cell: a number where it reads as one, the text otherwise."""
    try:
        return float(text)
    except ValueError:
        return text


def _csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return [
            {key: _cell(text) for key, text in row.items() if key not in _WALL_CLOCK}
            for row in csv.DictReader(handle)
        ]


def _files(directory: Path) -> dict[str, Any]:
    """Every file a run left, read: the CSVs row by row, the pool's header and records."""
    read: dict[str, Any] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix == ".csv":
            read[path.name] = _csv(path)
        elif path.name.endswith("_terms.pickle"):
            header, records = read_term_pool(path)
            read[path.name] = {"header": _plain(header), "records": _plain(records)}
        else:
            read[path.name] = "unread"
    return read


def _loop(space: Any, acquisition: Acquisition) -> dict[str, Any]:
    """The ask/tell loop over the real space with the real evolutionary search, seed 7."""
    bo = _make_bo(space, acquisition, seed=7)
    bo.initialize(initial_size=5)
    steps = []
    for _ in range(5 + 4):
        suggestion = bo.suggest()
        step: dict[str, Any] = {"term": _term(suggestion.candidate)}
        if suggestion.acquisition_value is not None:
            step["acquisition_value"] = float(suggestion.acquisition_value)
        step.update(_plain(suggestion.diagnostics or {}))
        steps.append(step)
        bo.observe(suggestion.candidate, _objective(suggestion.candidate))
    model = bo.surrogate
    result = bo.finalize()
    return {
        "steps": steps,
        "alpha": _plain(model.alpha_),
        "log_marginal_likelihood": float(model.log_marginal_likelihood_value_),
        "theta": _plain(model.kernel_.theta),
        "best_tree": _term(result["best_tree"]),
        "best_y": float(result["best_y"]),
        "y": _plain(result["y"]),
        "iterations": result["iterations"],
        "trace": _plain(result["trace"]),
    }


def _loop_with_a_forced_fallback(space: Any) -> dict[str, Any]:
    """The duplicate's replacement, which no run above takes: on the second pass the maximization
    hands back a term the loop already observed, and the loop draws the replacement from its
    sampler's stream."""
    original = AcquisitionOptimizer.maximize
    picks: list[Any] = []

    def forced(self: AcquisitionOptimizer, acquisition: Any, query: Any, **kwargs: Any) -> Any:
        pick = original(self, acquisition, query, **kwargs)
        picks.append(pick)
        return min(acquisition.known_points, key=_term) if len(picks) == 2 else pick

    with mock.patch.object(AcquisitionOptimizer, "maximize", forced):
        return _loop(space, "ExpectedImprovement")


def _closed_loop(space: Any) -> dict[str, Any]:
    """The closed loop, ``optimize()``, under its defaults: expected improvement, seed 7."""
    bo = _make_bo(space, "ExpectedImprovement", seed=7)
    result = bo.optimize(_objective, 4, initial_size=5)
    return {
        "x": [_term(term) for term in result["x"]],
        "y": _plain(result["y"]),
        "best_tree": _term(result["best_tree"]),
        "iterations": result["iterations"],
        "trace": _plain(result["trace"]),
    }


def _quiet(line: str) -> None:
    """An echo that prints nothing."""


def _run_layer(directory: Path) -> dict[str, Any]:
    """``run_search`` under Bayesian optimization, and ``run_paired`` with a random arm on a twin."""
    alone, paired = directory / "alone", directory / "paired"
    alone.mkdir()
    paired.mkdir()
    common: dict[str, Any] = {
        "schema": SCHEMA, "pretty_algebra": dict, "echo": _quiet, "n_design": 3, "n_passes": 2,
    }
    single = run_search(_bo(1), _metrics, csv_path=str(alone / "bo.csv"), **common)
    loop = _bo(1)
    arm = RandomSearch(
        list_space(), LIST, sampler=twin_sampler(loop.sampler, loop.query, random.Random(1)), seed=1
    )
    outcomes = run_paired(
        {"bo": loop, "random": arm}, _metrics,
        csv_paths={"bo": str(paired / "bo.csv"), "random": str(paired / "random.csv")}, **common,
    )
    return {
        "search": {"summary": _plain(single.summary), "files": _files(alone)},
        "paired": {
            "summaries": {name: _plain(outcome.summary) for name, outcome in outcomes.items()},
            "files": _files(paired),
        },
    }


def characterize() -> dict[str, Any]:
    """Everything the recording holds, computed now."""
    space = Synthesizer(_repository(), {}).construct_solution_space(_TARGET).prune()
    with tempfile.TemporaryDirectory() as directory:
        run_layer = _run_layer(Path(directory))
    return {
        "loop": {acquisition: _loop(space, acquisition) for acquisition in _ACQUISITIONS},
        "loop_forced_fallback": _loop_with_a_forced_fallback(space),
        "closed_loop": _closed_loop(space),
        "run_layer": run_layer,
    }


def _differences(recorded: Any, computed: Any, rel: float, where: str = "") -> list[str]:
    """Where two characterizations part: keys and lengths and texts exactly, floats within ``rel``."""
    if isinstance(recorded, dict) and isinstance(computed, dict):
        if set(recorded) != set(computed):
            return [f"{where}: keys {sorted(set(recorded) ^ set(computed))} on one side only"]
        return [
            difference
            for key in recorded
            for difference in _differences(recorded[key], computed[key], rel, f"{where}/{key}")
        ]
    if isinstance(recorded, list) and isinstance(computed, list):
        if len(recorded) != len(computed):
            return [f"{where}: {len(recorded)} recorded, {len(computed)} now"]
        return [
            difference
            for index, (old, new) in enumerate(zip(recorded, computed, strict=True))
            for difference in _differences(old, new, rel, f"{where}[{index}]")
        ]
    floats = (float, int)
    if (
        isinstance(recorded, floats) and isinstance(computed, floats)
        and not isinstance(recorded, bool) and not isinstance(computed, bool)
    ):
        same = (
            recorded == computed
            or (math.isnan(recorded) and math.isnan(computed))
            or (math.isfinite(recorded) and math.isclose(recorded, computed, rel_tol=rel, abs_tol=0.0))
        )
        return [] if same else [f"{where}: {recorded!r} recorded, {computed!r} now"]
    return [] if recorded == computed else [f"{where}: {recorded!r} recorded, {computed!r} now"]


@pytest.mark.slow
@pytest.mark.integration
def test_the_default_path_computes_what_was_recorded():
    recorded = json.loads(RECORDING.read_text())
    computed = json.loads(json.dumps(characterize()))
    rel = 0.0 if os.environ.get("DEFAULT_PATH_EXACT") == "1" else 1e-9

    differences = _differences(recorded["characterization"], computed, rel)

    assert differences == [], f"{len(differences)} differences, the first: {differences[:5]}"
    # Not vacuous: the recording holds every pick of every run, the one replaced duplicate, and
    # every file of both arms.
    characterization = recorded["characterization"]
    assert sum(len(run["steps"]) for run in characterization["loop"].values()) == 3 * 9
    replaced = [step.get("fallback_used") for step in characterization["loop_forced_fallback"]["steps"]]
    assert replaced == [None] * 5 + [False, True, False, False]
    assert len(characterization["run_layer"]["paired"]["files"]) == 6


def test_the_comparison_sees_a_changed_float_a_changed_term_and_a_missing_row():
    """The comparison itself, on a recording altered three ways: each one is reported."""
    base = {"steps": [{"term": "a", "mean_at_pick": 0.5}, {"term": "b", "mean_at_pick": 0.25}]}
    moved = {"steps": [{"term": "a", "mean_at_pick": 0.5 + 1e-6}, {"term": "b", "mean_at_pick": 0.25}]}
    renamed = {"steps": [{"term": "a", "mean_at_pick": 0.5}, {"term": "c", "mean_at_pick": 0.25}]}
    short = {"steps": [{"term": "a", "mean_at_pick": 0.5}]}

    assert _differences(base, json.loads(json.dumps(base)), 0.0) == []
    assert _differences(base, moved, 1e-9) == [
        f"/steps[0]/mean_at_pick: 0.5 recorded, {0.5 + 1e-6!r} now"
    ]
    assert _differences(base, renamed, 1e-9) == ["/steps[1]/term: 'b' recorded, 'c' now"]
    assert _differences(base, short, 1e-9) == ["/steps: 2 recorded, 1 now"]
    assert _differences({"x": 1.0}, {"x": 1.0 + 1e-12}, 1e-9) == []
    assert _differences({"x": 1.0}, {"x": 1.0 + 1e-12}, 0.0) != []
    assert _differences({"x": math.nan}, {"x": math.nan}, 0.0) == []
    assert _differences({"x": math.nan}, {"x": 1.0}, 1e-9) != []


if __name__ == "__main__":
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
        cwd=Path(__file__).parent,
    ).stdout.strip()
    changed = subprocess.run(
        ["git", "status", "--porcelain", "--", "bayesian_optimization"], capture_output=True,
        text=True, check=True, cwd=Path(__file__).parents[2],
    ).stdout.strip()
    if changed:
        # the tree recorded is the commit the recording is committed with, not its parent
        commit += " with uncommitted changes to bayesian_optimization"
    RECORDING.write_text(
        json.dumps({"recorded_at": commit, "characterization": characterize()}, indent=1) + "\n"
    )
    print(f"recorded at {commit} into {RECORDING}", file=sys.stderr)
