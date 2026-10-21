"""The run layer under an exchangeable duplicate test.

A design drawn up front is drawn pairwise distinct under the strategy's test, a design given as
terms that the test merges is refused as a repeated one is, and the strategies of a paired run,
which take one design and are compared on it, share one test. The test and the measurement are
those of :mod:`tests.unit.test_duplicate_test_in_the_loops`: one value, one evaluation.
"""

from __future__ import annotations

import json
import random

import pytest
from cosy.search import DepthBoundedRandomSampler, SizeUniformSampler, generator_query

from bayesian_optimization import NSGA2, BayesianOptimization, RandomSearch
from bayesian_optimization.duplicates import DuplicateTest
from bayesian_optimization.runs import (
    MetricSchema,
    Objective,
    RunArtifacts,
    build_acquisition_optimizer,
    load_design_records,
    read_term_pool,
)
from bayesian_optimization.runs.driver import run_paired, run_search
from bayesian_optimization.runs.search_program import draw_design
from tests.spaces import EXPR, expression_space
from tests.unit.test_duplicate_test_in_the_loops import (
    BOUND,
    DEPTH,
    SameValue,
    measure,
    merged_pair,
    value,
)

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


def test_an_nsga2_run_names_a_member_the_test_merged_by_the_evaluation_it_was_answered_from(tmp_path):
    csv_path = str(tmp_path / "nsga2.csv")
    search = NSGA2(expression_space(), EXPR, cost=lambda term: float(term.size), depth_bound=DEPTH, seed=6,
                   sampler=SizeUniformSampler(BOUND, random.Random(6)), duplicate_test=SameValue())
    outcome = run_search(search, lambda term: {"score": measure(term)}, schema=SCHEMA, n_design=6, n_passes=5,
                         csv_path=csv_path, pretty_algebra=dict, echo=_quiet)
    evaluated = list(outcome.result["x"])
    merged = {member for formed in search.formed_generations for member in formed.members} - set(evaluated)
    assert merged, "no population holds a term the test merged with a measured one, so no label is borrowed"
    _header, records = read_term_pool(RunArtifacts(csv_path).path("terms"))
    term_of = {(record.phase, record.index): record.term for record in records}
    with open(RunArtifacts(csv_path).path("generations")) as handle:
        lines = [json.loads(line) for line in handle]
    assert len(lines) == len(search.formed_generations)
    for line, formed in zip(lines, search.formed_generations, strict=True):
        named = [term_of[tuple(label)] for label in line["members"]]
        assert named == [evaluated[row] for row in formed.rows]
        assert [value(term) for term in named] == [value(member) for member in formed.members]


def test_a_design_given_to_a_run_that_the_test_merges_is_refused_before_anything_is_opened(tmp_path):
    a, b = merged_pair()
    search = RandomSearch(expression_space(), EXPR, sampler=SizeUniformSampler(BOUND, random.Random(0)), seed=0,
                          duplicate_test=SameValue())
    with pytest.raises(ValueError, match="the duplicate test calls two terms of the design one evaluation"):
        run_search(search, lambda term: {"score": measure(term)}, schema=SCHEMA, design=[a, b], n_passes=1,
                   csv_path=str(tmp_path / "run.csv"), pretty_algebra=dict, echo=_quiet)
    assert not list(tmp_path.iterdir()), "the refusal comes before any file is written"


def test_a_design_the_loop_drew_under_the_test_resumes_against_the_head_of_its_stream(tmp_path):
    first = str(tmp_path / "first.csv")
    run_search(_strategies(4, SameValue())["bo"], lambda term: {"score": measure(term)}, schema=SCHEMA, n_design=6,
               csv_path=first, pretty_algebra=dict, echo=_quiet)
    records = load_design_records(RunArtifacts(first).path("terms"), {})
    assert records.origin == "drawn"
    stream = [term for term, _ in zip(SizeUniformSampler(BOUND, random.Random(4)).sample(_query()), range(6))]
    assert [record.term for record in records] != stream, "the design must skip a merged term, or it is the raw head"
    run_search(_strategies(4, SameValue())["bo"], lambda term: {"score": measure(term)}, schema=SCHEMA,
               resume=records, csv_path=str(tmp_path / "again.csv"), pretty_algebra=dict, echo=_quiet)


def test_paired_strategies_under_two_duplicate_tests_are_refused_before_anything_is_drawn(tmp_path):
    strategies = _strategies(5, SameValue())
    strategies["random"] = RandomSearch(expression_space(), EXPR, sampler=SizeUniformSampler(BOUND, random.Random(5)), seed=5)
    with pytest.raises(ValueError, match="share one duplicate test"):
        _paired(tmp_path, strategies)
    assert not list(tmp_path.iterdir()), "the refusal comes before any file is written"


def test_paired_strategies_under_two_tests_of_one_name_are_refused(tmp_path):
    # Finer than SameValue, so nothing downstream notices: a design distinct under one is under both.
    namesake = type(SameValue.__name__, (DuplicateTest,), {
        "key": lambda self, term: (value(term), term.size),
        "same": lambda self, a, b: (value(a), a.size) == (value(b), b.size),
    })
    assert namesake.__name__ == SameValue.__name__ and namesake is not SameValue
    strategies = _strategies(5, SameValue())
    strategies["random"] = RandomSearch(expression_space(), EXPR, sampler=SizeUniformSampler(BOUND, random.Random(5)),
                                        seed=5, duplicate_test=namesake())
    with pytest.raises(ValueError, match="share one duplicate test"):
        _paired(tmp_path, strategies)
    assert not list(tmp_path.iterdir()), "the refusal comes before any file is written"
