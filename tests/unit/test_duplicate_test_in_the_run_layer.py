"""The run layer under an exchangeable duplicate test.

A design drawn up front is drawn pairwise distinct under the strategy's test, a design given as
terms that the test merges is refused as a repeated one is, and the strategies of a paired run,
which take one design and are compared on it, share one test. The test and the measurement are
those of :mod:`tests.unit.test_duplicate_test_in_the_loops`: one value, one evaluation.
"""

from __future__ import annotations

import random

import pytest
from cosy.search import DepthBoundedRandomSampler, SizeUniformSampler, generator_query

from bayesian_optimization import BayesianOptimization, RandomSearch
from bayesian_optimization.runs import MetricSchema, Objective, build_acquisition_optimizer
from bayesian_optimization.runs.driver import run_paired
from bayesian_optimization.runs.search_program import draw_design
from tests.spaces import EXPR, expression_space
from tests.unit.test_duplicate_test_in_the_loops import BOUND, SameValue, measure, value

SCHEMA = MetricSchema.for_metrics(Objective("score"), ["score"])


def _quiet(line):
    """An echo that prints nothing."""


def _query():
    return generator_query(expression_space(), EXPR)


def test_an_up_front_design_is_drawn_pairwise_distinct_under_the_test():
    head, repeats = draw_design(DepthBoundedRandomSampler(3, random.Random(0)), _query(), 5, test=SameValue())
    assert len({value(term) for term in head}) == 5
    assert repeats > 0, "the stream must offer a merged term, or the draw rejects nothing"
    plain, _ = draw_design(DepthBoundedRandomSampler(3, random.Random(0)), _query(), 5)
    assert len({value(term) for term in plain}) < 5


def test_a_given_design_the_test_merges_is_refused_as_a_repeated_one_is():
    drawn = [term for term, _ in zip(SizeUniformSampler(BOUND, random.Random(3)).sample(_query()), range(40))]
    a, b = next((x, y) for i, x in enumerate(drawn) for y in drawn[i + 1:] if value(x) == value(y))
    assert a != b
    with pytest.raises(ValueError, match="the duplicate test calls two of its terms one evaluation"):
        draw_design(SizeUniformSampler(BOUND, random.Random(4)), _query(), 2, design=[a, b], test=SameValue())
    head, _ = draw_design(SizeUniformSampler(BOUND, random.Random(4)), _query(), 2, design=[a, b])
    assert head == [a, b], "under term equality two distinct terms are a design"


def _strategies(seed: int, test) -> dict:
    space = expression_space()
    bo = BayesianOptimization(space, EXPR, sampler=SizeUniformSampler(BOUND, random.Random(seed)), seed=seed,
                              optimizer=build_acquisition_optimizer(population_size=6, generations=2, depth_bound=4,
                                                                    seed=seed),
                              duplicate_test=test)
    arm = RandomSearch(space, EXPR, sampler=SizeUniformSampler(BOUND, random.Random(seed)), seed=seed, duplicate_test=test)
    return {"bo": bo, "random": arm}


def _paired(tmp_path, strategies, n_design=4, n_passes=2):
    tmp_path.mkdir(exist_ok=True)
    return run_paired(strategies, lambda term: {"score": measure(term)}, schema=SCHEMA, n_design=n_design,
                      n_passes=n_passes, csv_paths={name: str(tmp_path / f"{name}.csv") for name in strategies},
                      pretty_algebra=dict, echo=_quiet)


def test_a_paired_run_draws_its_design_and_makes_its_passes_under_the_test(tmp_path):
    outcomes = _paired(tmp_path / "test", _strategies(2, SameValue()))
    for name, outcome in outcomes.items():
        terms = list(outcome.result["x"])
        assert len(terms) == 4 + 2
        assert len({value(term) for term in terms}) == len(terms), name
    plain = _paired(tmp_path / "plain", _strategies(2, None))
    assert any(len({value(term) for term in outcome.result["x"]}) < len(outcome.result["x"]) for outcome in plain.values()), (
        "under term equality some arm must measure two terms of one value, or nothing is merged")


def test_paired_strategies_under_two_duplicate_tests_are_refused_before_anything_is_drawn(tmp_path):
    strategies = _strategies(5, SameValue())
    strategies["random"] = RandomSearch(expression_space(), EXPR, sampler=SizeUniformSampler(BOUND, random.Random(5)), seed=5)
    with pytest.raises(ValueError, match="share one duplicate test"):
        _paired(tmp_path, strategies)
    assert not list(tmp_path.iterdir()), "the refusal comes before any file is written"
