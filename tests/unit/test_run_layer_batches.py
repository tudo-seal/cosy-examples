"""A run in rounds: several suggestions evaluated side by side, each written as it completes.

``run_search(..., batch_size=k)`` takes up to k suggestions of the strategy at a time -- the design
k terms at a time, then the passes -- and hands each round to ``evaluate_many``, which answers
``(term, metrics)`` pairs in the order the evaluations complete; every answer is written and
observed as it arrives.  Without ``evaluate_many`` the round is evaluated one term after the other
with ``evaluate``.  At k = 1 the run is the one it always was.
"""

from __future__ import annotations

import csv
import random
from typing import Any

import pytest
from cosy.search import SizeUniformSampler

from bayesian_optimization import BayesianOptimization, RandomSearch
from bayesian_optimization.acquisition_optimizer import SampleMaximizer
from bayesian_optimization.runs import read_term_pool, run_paired, run_search, twin_sampler
from tests.spaces import LIST, list_space
from tests.unit.test_run_driver import SCHEMA, _metrics


def _sampler(seed: int) -> SizeUniformSampler:
    return SizeUniformSampler(6, random.Random(seed))


def _bo(seed: int, capacity: int = 2) -> BayesianOptimization:
    return BayesianOptimization(
        list_space(), LIST, sampler=_sampler(seed), seed=seed, max_outstanding=capacity,
        maximizer=SampleMaximizer(_sampler(seed + 100), 8),
    )


def _rows(path: Any) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _quiet(line: str) -> None:
    """An echo that prints nothing."""


def _backwards(rounds: list[list[Any]]) -> Any:
    """An evaluation of a round that completes in reverse order, keeping the rounds it was asked."""

    def evaluate_many(terms: list[Any]) -> Any:
        rounds.append(list(terms))
        for term in reversed(terms):
            yield term, _metrics(term)

    return evaluate_many


def _run(path: Any, strategy: Any, **kwargs: Any) -> Any:
    return run_search(strategy, _metrics, schema=SCHEMA, csv_path=str(path), pretty_algebra=dict,
                      echo=_quiet, **kwargs)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, 3])
def test_a_round_larger_than_the_strategy_holds_is_refused_before_anything(tmp_path, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        _run(tmp_path / "run.csv", _bo(1, capacity=2), n_design=2, n_passes=2,
             batch_size=batch_size)
    assert list(tmp_path.iterdir()) == []


def test_a_run_in_rounds_writes_each_answer_as_it_completes(tmp_path):
    path = tmp_path / "run.csv"
    rounds: list[list[Any]] = []
    on_disk: list[int] = []
    backwards = _backwards(rounds)

    def evaluate_many(terms: list[Any]) -> Any:
        for answer in backwards(terms):
            on_disk.append(len(_rows(path)) if path.exists() else 0)
            yield answer

    outcome = _run(path, _bo(3), n_design=4, n_passes=4, batch_size=2, evaluate_many=evaluate_many)

    assert [len(terms) for terms in rounds] == [2, 2, 2, 2]
    assert on_disk == list(range(8)), "every answer on disk before the next one is taken"
    rows = _rows(path)
    assert [(row["phase"], row["index"]) for row in rows] == [
        ("pre_sample", "1"), ("pre_sample", "0"), ("pre_sample", "3"), ("pre_sample", "2"),
        ("bo_step", "1"), ("bo_step", "0"), ("bo_step", "3"), ("bo_step", "2"),
    ]
    _header, records = read_term_pool(str(tmp_path / "run_terms.pickle"))
    assert [record.term for record in records] == list(outcome.result["x"]), "the dataset's order"
    assert [record.term for record in records[:4]] == [
        rounds[0][1], rounds[0][0], rounds[1][1], rounds[1][0]]
    assert len(set(outcome.result["x"])) == 8


def test_without_an_evaluation_of_rounds_a_round_is_evaluated_term_by_term(tmp_path):
    evaluated: list[Any] = []

    def evaluate(term: Any) -> Any:
        evaluated.append(term)
        return _metrics(term)

    outcome = run_search(_bo(5), evaluate, schema=SCHEMA, csv_path=str(tmp_path / "run.csv"),
                         pretty_algebra=dict, echo=_quiet, n_design=3, n_passes=3, batch_size=2)
    assert evaluated == list(outcome.result["x"])
    assert [row["index"] for row in _rows(tmp_path / "run.csv")] == ["0", "1", "2", "0", "1", "2"]


@pytest.mark.parametrize("answer", ["a term not asked", "a term twice", "too few"])
def test_an_evaluation_of_a_round_answers_each_term_once(tmp_path, answer):
    def evaluate_many(terms: list[Any]) -> Any:
        if answer == "a term not asked":
            yield object(), {"score": 0.5, "length": 1}
        elif answer == "a term twice":
            yield terms[0], _metrics(terms[0])
            yield terms[0], _metrics(terms[0])
        else:
            yield terms[0], _metrics(terms[0])

    with pytest.raises(RuntimeError, match="evaluate_many"):
        _run(tmp_path / "run.csv", _bo(7), n_design=2, n_passes=0, batch_size=2,
             evaluate_many=evaluate_many)


def test_a_pair_runs_every_arm_in_rounds(tmp_path):
    rounds: list[list[Any]] = []
    loop = _bo(4)
    arm = RandomSearch(list_space(), LIST, sampler=twin_sampler(loop.sampler, loop.query,
                                                                random.Random(4)),
                       seed=4, max_outstanding=2)
    outcomes = run_paired(
        {"bo": loop, "random": arm}, _metrics, schema=SCHEMA, n_design=2, n_passes=4,
        csv_paths={"bo": str(tmp_path / "bo.csv"), "random": str(tmp_path / "random.csv")},
        pretty_algebra=dict, echo=_quiet, batch_size=2, evaluate_many=_backwards(rounds),
    )
    assert [len(terms) for terms in rounds] == [2, 2, 2, 2, 2]
    assert outcomes["random"].summary["taken_over"] == 2


def test_a_pair_refuses_a_round_one_of_its_arms_cannot_hold(tmp_path):
    loop = _bo(4, capacity=2)
    arm = RandomSearch(list_space(), LIST, sampler=twin_sampler(loop.sampler, loop.query,
                                                                random.Random(4)), seed=4)
    with pytest.raises(ValueError, match="batch_size"):
        run_paired(
            {"bo": loop, "random": arm}, _metrics, schema=SCHEMA, n_design=2, n_passes=2,
            csv_paths={"bo": str(tmp_path / "bo.csv"), "random": str(tmp_path / "random.csv")},
            pretty_algebra=dict, echo=_quiet, batch_size=2,
        )
    assert list(tmp_path.iterdir()) == []
