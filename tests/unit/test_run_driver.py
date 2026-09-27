"""One driver for every strategy: the design as the loop's first phase, the passes, a record of each.

``run_search`` runs any strategy of the ask/tell loop, Bayesian optimization or random search,
under a caller's metric schema.  Its design is drawn, given as terms, or resumed from the records of
an earlier run, and on every path each evaluation is on disk before the next one starts, with the
value the loop was handed.  ``run_paired`` runs several strategies from one design, evaluated once,
which is what the CIFAR driver's paired random-search arm did for one pair of strategies.
"""

from __future__ import annotations

import csv
import random

import pytest
from cosy.search import SizeUniformSampler

from bayesian_optimization import BayesianOptimization, RandomSearch
from bayesian_optimization.initial_sampling import distinct_prefix
from bayesian_optimization.runs import (
    MetricSchema,
    Objective,
    TermRecord,
    build_acquisition_optimizer,
    read_term_pool,
)
from bayesian_optimization.runs.driver import run_paired, run_search
from tests.spaces import LIST, list_space

SCHEMA = MetricSchema.for_metrics(Objective("score"), ["score", "length"])


def _metrics(term):
    """A stand-in evaluation: a finite score read off the list the term renders."""
    text = term.interpret({})
    return {"score": 1.0 / (1 + len(text)), "length": len(text)}


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _sampler(seed):
    return SizeUniformSampler(6, random.Random(seed))


def _random(seed):
    return RandomSearch(list_space(), LIST, sampler=_sampler(seed), seed=seed)


def _bo(seed):
    return BayesianOptimization(
        list_space(), LIST, sampler=_sampler(seed), seed=seed,
        optimizer=build_acquisition_optimizer(population_size=6, generations=2, depth_bound=4,
                                              seed=seed),
    )


def _quiet(line):
    """An echo that prints nothing."""


def _run(strategy, evaluate, path, **kwargs):
    return run_search(strategy, evaluate, schema=SCHEMA, csv_path=str(path), pretty_algebra=dict,
                      echo=_quiet, **kwargs)


def _stream_head(seed, count):
    terms, repeats = distinct_prefix(_sampler(seed), _random(seed).query, count)
    assert repeats == 0
    return terms


def test_a_random_search_runs_through_the_driver_and_every_row_is_written_as_measured(tmp_path):
    path = tmp_path / "run.csv"
    seen_before_each = []

    def evaluate(term):
        seen_before_each.append(len(_rows(path)) if path.exists() else 0)
        return _metrics(term)

    outcome = _run(_random(3), evaluate, path, n_design=2, n_passes=3)

    assert seen_before_each == [0, 1, 2, 3, 4]
    rows = _rows(path)
    assert [row["phase"] for row in rows] == ["pre_sample"] * 2 + ["random_sample"] * 3
    _header, records = read_term_pool(tmp_path / "run_terms.pickle")
    assert [record.loop_value for record in records] == [float(row["loop_value"]) for row in rows]
    assert outcome.summary["run_kind"] == "random_search"
    assert outcome.summary["design_source"] == "drawn"
    assert (outcome.summary["evaluated_here"], outcome.summary["taken_over"]) == (5, 0)


def test_a_bayesian_run_logs_its_passes_the_inner_search_and_the_surrogate(tmp_path):
    path = tmp_path / "run.csv"
    outcome = _run(_bo(1), _metrics, path, n_design=3, n_passes=2)

    rows = _rows(path)
    assert [row["phase"] for row in rows] == ["pre_sample"] * 3 + ["bo_step"] * 2
    assert all(row["acquisition_value"] == "" for row in rows[:3])
    assert all(row["acquisition_value"] != "" for row in rows[3:])
    assert len(_rows(tmp_path / "run_ea.csv")) > 0
    assert len(_rows(tmp_path / "run_surrogate.csv")) == 2
    assert outcome.summary["run_kind"] == "bayesian_optimization"
    assert outcome.result["iterations"] == 2


def test_a_design_given_as_terms_is_evaluated_as_given(tmp_path):
    terms = _stream_head(seed=9, count=3)
    outcome = _run(_random(9), _metrics, tmp_path / "run.csv", design=terms, n_passes=0)

    assert list(outcome.result["x"]) == terms
    assert outcome.summary["design_source"] == "terms"


def test_a_design_resumed_from_a_pool_is_taken_over_and_not_evaluated(tmp_path):
    _run(_random(5), _metrics, tmp_path / "first.csv", n_design=3, n_passes=0)
    _header, records = read_term_pool(tmp_path / "first_terms.pickle")
    evaluated = []

    def evaluate(term):
        evaluated.append(term)
        return _metrics(term)

    outcome = _run(_random(6), evaluate, tmp_path / "second.csv", resume=records, n_passes=2)

    assert len(evaluated) == 2, "only the passes are evaluated"
    assert not set(evaluated) & {record.term for record in records}
    assert list(outcome.result["x"][:3]) == [record.term for record in records]
    assert list(outcome.result["y"][:3]) == [record.loop_value for record in records]
    assert outcome.summary["design_source"] == "resumed"
    assert (outcome.summary["evaluated_here"], outcome.summary["taken_over"]) == (2, 3)


def test_a_record_without_the_loop_s_value_needs_the_caller_s_reading(tmp_path):
    terms = _stream_head(seed=2, count=3)
    legacy = [TermRecord("pre_sample", i, term, {"score": 0.5 + i / 10}) for i, term in enumerate(terms)]

    with pytest.raises(ValueError, match="resumed_value"):
        _run(_random(2), _metrics, tmp_path / "refused.csv", resume=legacy, n_passes=0)
    assert not (tmp_path / "refused.csv").exists(), "refused before any file is opened"

    outcome = _run(_random(2), _metrics, tmp_path / "read.csv", resume=legacy, n_passes=0,
                   resumed_value=lambda metrics: metrics["score"])
    assert list(outcome.result["y"]) == [0.5, 0.6, 0.7]


def test_a_run_whose_files_exist_is_refused_before_anything_is_evaluated(tmp_path):
    (tmp_path / "run_ea.csv").write_text("another run's generations\n")
    evaluated = []

    with pytest.raises(FileExistsError, match="run_ea.csv"):
        _run(_random(0), lambda term: evaluated.append(term) or _metrics(term), tmp_path / "run.csv",
             n_design=2, n_passes=1)
    assert evaluated == []
    assert not (tmp_path / "run.csv").exists()


def test_a_configuration_no_pass_could_use_is_refused_before_the_design(tmp_path):
    bo = _bo(0)
    bo.acquisition_function = "NoSuchScore"
    evaluated = []

    with pytest.raises(ValueError):
        _run(bo, lambda term: evaluated.append(term) or _metrics(term), tmp_path / "run.csv",
             n_design=2, n_passes=1)
    assert evaluated == []


def test_a_paired_comparison_shares_one_design_evaluated_once(tmp_path):
    """The CIFAR driver's paired baseline as a helper: one design, then each strategy's passes."""
    evaluated = []

    def evaluate(term):
        evaluated.append(term)
        return _metrics(term)

    outcomes = run_paired(
        {"bo": _bo(4), "random": _random(4)}, evaluate, schema=SCHEMA, n_design=3, n_passes=2,
        csv_paths={"bo": str(tmp_path / "bo.csv"), "random": str(tmp_path / "random.csv")},
        pretty_algebra=dict, echo=_quiet,
    )

    assert len(evaluated) == 3 + 2 + 2, "the shared design is evaluated once"
    head = _stream_head(seed=4, count=5)
    assert list(outcomes["bo"].result["x"][:3]) == head[:3]
    assert list(outcomes["random"].result["x"]) == head, (
        "the random arm takes the design over and continues the stream past it"
    )
    assert outcomes["random"].summary["taken_over"] == 3
    assert outcomes["bo"].summary["taken_over"] == 0
