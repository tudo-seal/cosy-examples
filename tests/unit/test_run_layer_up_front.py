"""The run layer draws what a driver drew: a paired design up front, a resumed design's head.

A paired random arm draws, on a twin of the loop's sampler, the stream past the design.  The loop's
replacement for a duplicate opens a stream where its own sampler stands, so a sampler that never
drew hands the loop the arm's first term.  ``run_paired`` therefore draws the design and the other
arms' terms up front from the loop's sampler, as the CIFAR driver's paired path always did, under a
watchdog and into the first arm's record.  A resumed design is held against the head of the stream
it was drawn from and the sampler moved past it -- where the pool says it was drawn; a design given
as terms was never drawn, and resumes under any seed.
"""

from __future__ import annotations

import csv
import random
from typing import Any

import pytest
from cosy.search import SizeUniformSampler

from bayesian_optimization import BayesianOptimization, RandomSearch
from bayesian_optimization.acquisition_optimizer import SampleMaximizer
from bayesian_optimization.initial_sampling import _sample_fallback_tree, distinct_prefix
from bayesian_optimization.runs import (
    load_design_records,
    read_term_pool,
    run_paired,
    run_search,
    twin_sampler,
)
from bayesian_optimization.runs.term_pool import TermPoolWriter
from tests.spaces import LIST, list_space
from tests.unit.test_run_driver import SCHEMA, _metrics


def _sampler(seed: int) -> SizeUniformSampler:
    return SizeUniformSampler(6, random.Random(seed))


def _bo(seed: int) -> BayesianOptimization:
    """Maximized over a sample of a sampler of its own, which leaves the loop's stream alone."""
    return BayesianOptimization(
        list_space(), LIST, sampler=_sampler(seed), seed=seed,
        maximizer=SampleMaximizer(_sampler(seed + 100), 8),
    )


def _twin_of(loop: BayesianOptimization, seed: int) -> RandomSearch:
    return RandomSearch(
        list_space(), LIST, sampler=twin_sampler(loop.sampler, loop.query, random.Random(seed)),
        seed=seed,
    )


def _quiet(line: str) -> None:
    """An echo that prints nothing."""


def _rows(path: Any) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture
def every_pick_known(monkeypatch):
    """Every maximization answers a term the loop already holds, so every pass draws a
    replacement: the path a paired or resumed run has to get right."""
    original = SampleMaximizer.maximize_with_population

    def known(self: Any, acquisition: Any, query: Any, **kwargs: Any) -> Any:
        _pick, population, generations = original(self, acquisition, query, **kwargs)
        return min(acquisition.known_points, key=repr), population, generations

    monkeypatch.setattr(SampleMaximizer, "maximize_with_population", known)


def _pair(tmp_path: Any, seed: int = 4, **kwargs: Any) -> dict[str, Any]:
    loop = _bo(seed)
    return run_paired(
        {"bo": loop, "random": _twin_of(loop, seed)}, _metrics, schema=SCHEMA, n_passes=2,
        csv_paths={"bo": str(tmp_path / "bo.csv"), "random": str(tmp_path / "random.csv")},
        pretty_algebra=dict, echo=_quiet, **kwargs,
    )


def test_a_paired_loop_s_replacement_is_drawn_past_the_design_and_the_arm_s_terms(
    tmp_path, every_pick_known
):
    """As the paired path always drew: its replacement comes from where drawing the design and the
    random arm's terms leaves the loop's sampler, not from where the design alone leaves it."""
    _pair(tmp_path, n_design=3)

    loop = read_term_pool(str(tmp_path / "bo_terms.pickle"))[1]
    assert [row["fallback_used"] for row in _rows(tmp_path / "bo.csv")[3:]] == ["True", "True"]
    reference = _sampler(4)
    query = _bo(4).query
    distinct_prefix(reference, query, 3 + 2)
    design = {record.term for record in loop[:3]}
    assert loop[3].term == _sample_fallback_tree(reference, query, design)


def test_the_first_arm_s_record_says_the_design_was_drawn_up_front_and_what_it_cost(
    tmp_path, monkeypatch
):
    import time

    import bayesian_optimization.runs.driver as driver

    draw = driver.draw_design

    def slow(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.5)  # longer than the run itself, so that its seconds must hold the draw's
        return draw(*args, **kwargs)

    monkeypatch.setattr(driver, "draw_design", slow)
    outcomes = _pair(tmp_path, n_design=3)

    summary = outcomes["bo"].summary
    drawn = summary["design_drawn_up_front"]
    assert summary["design_source"] == "drawn"
    assert (drawn["design"], drawn["then"], drawn["repeats_skipped"]) == (3, 2, 0)
    assert summary["initial_repeats_rejected"] == drawn["repeats_skipped"]
    assert drawn["seconds"] >= 0.5
    assert summary["seconds"] >= drawn["seconds"] + 0.0 and outcomes["bo"].seconds == summary["seconds"]
    assert outcomes["random"].summary["design_taken_over_from"] == "bo.csv"


def test_the_up_front_draw_is_watched_and_announced(tmp_path, monkeypatch):
    import bayesian_optimization.runs.driver as driver

    open_budgets: list[str] = []
    during_the_draw: list[list[str]] = []
    budget, draw = driver.step_budget, driver.draw_design

    def watching(label: str, *args: Any, **kwargs: Any) -> Any:
        open_budgets.append(label)
        return budget(label, *args, **kwargs)

    def drawing(*args: Any, **kwargs: Any) -> Any:
        during_the_draw.append(list(open_budgets))
        return draw(*args, **kwargs)

    monkeypatch.setattr(driver, "step_budget", watching)
    monkeypatch.setattr(driver, "draw_design", drawing)
    said: list[str] = []
    loop = _bo(4)
    run_paired(
        {"bo": loop, "random": _twin_of(loop, 4)}, _metrics, schema=SCHEMA, n_passes=2,
        n_design=3, csv_paths={"bo": str(tmp_path / "bo.csv"), "random": str(tmp_path / "r.csv")},
        pretty_algebra=dict, echo=said.append,
    )

    assert len(during_the_draw) == 1
    assert any("up front" in label for label in during_the_draw[0])
    assert any("up front" in line for line in said)


def _measured(tmp_path: Any, name: str, strategy: Any, **kwargs: Any) -> str:
    path = tmp_path / f"{name}.csv"
    run_search(strategy, _metrics, schema=SCHEMA, csv_path=str(path), pretty_algebra=dict,
               echo=_quiet, **kwargs)
    return str(tmp_path / f"{name}_terms.pickle")


def test_a_resumed_bayesian_design_is_held_against_its_stream(tmp_path):
    pool = _measured(tmp_path, "first", _bo(1), n_design=3)
    records = load_design_records(pool, {})

    with pytest.raises(ValueError, match="different stream"):
        _measured(tmp_path, "other_seed", _bo(2), resume=records)
    _measured(tmp_path, "same_seed", _bo(1), resume=records)


def test_a_resumed_loop_s_replacement_is_the_one_a_fresh_run_draws(tmp_path, every_pick_known):
    """Its sampler moved past the design it took over, as a fresh run's is past the design it drew."""
    pool = _measured(tmp_path, "design", _bo(1), n_design=3)
    _measured(tmp_path, "fresh", _bo(1), n_design=3, n_passes=2)
    _measured(tmp_path, "resumed", _bo(1), resume=load_design_records(pool, {}), n_passes=2)

    fresh, resumed = _rows(tmp_path / "fresh.csv"), _rows(tmp_path / "resumed.csv")
    assert [row["fallback_used"] for row in resumed[3:]] == ["True", "True"]
    assert [row["structure"] for row in resumed] == [row["structure"] for row in fresh]


def test_a_design_given_as_terms_resumes_under_any_seed(tmp_path):
    head = [record.term for record in read_term_pool(_measured(tmp_path, "d", _bo(1), n_design=3))[1]]
    pool = _measured(tmp_path, "given", _bo(1), design=head)

    _measured(tmp_path, "resumed", _bo(7), resume=load_design_records(pool, {}), n_passes=1)


def test_the_pool_says_where_its_design_came_from(tmp_path):
    drawn = _measured(tmp_path, "drawn", _bo(1), n_design=3)
    head = [record.term for record in read_term_pool(drawn)[1]]
    given = _measured(tmp_path, "given", _bo(1), design=head)
    resumed_drawn = _measured(tmp_path, "rd", _bo(1), resume=load_design_records(drawn, {}))
    resumed_given = _measured(tmp_path, "rg", _bo(3), resume=load_design_records(given, {}))
    _pair(tmp_path, n_design=3)

    def origin(pool: str) -> str:
        recorded: str = read_term_pool(pool)[0]["design_origin"]
        return recorded

    assert [origin(pool) for pool in (drawn, given, resumed_drawn, resumed_given)] == [
        "drawn", "given", "drawn", "given",
    ]
    assert origin(str(tmp_path / "bo_terms.pickle")) == "drawn"
    assert origin(str(tmp_path / "random_terms.pickle")) == "drawn"
    assert load_design_records(given, {}).origin == "given"
    # a pair run on a design given as terms: given, for the arm that took it over as well
    (tmp_path / "g").mkdir()
    _pair(tmp_path / "g", design=head)
    assert origin(str(tmp_path / "g" / "bo_terms.pickle")) == "given"
    assert origin(str(tmp_path / "g" / "random_terms.pickle")) == "given"


def test_a_pool_written_before_origins_were_recorded_counts_as_drawn(tmp_path):
    path = tmp_path / "old_terms.pickle"
    term = next(iter(_sampler(1).sample(_bo(1).query)))
    with TermPoolWriter(str(path), provenance={}) as writer:
        writer.write("pre_sample", 0, term, {"score": 0.5}, loop_value=0.5)
    assert load_design_records(str(path), {}).origin == "drawn"


def test_a_random_search_resumes_on_its_own_stream(tmp_path):
    """It draws its design and its passes from one stream, and continues it past the held terms
    on its own: resumed, it makes the passes it made."""
    pool = _measured(tmp_path, "first", RandomSearch(list_space(), LIST, sampler=_sampler(3),
                                                    seed=3), n_design=2, n_passes=3)
    _measured(tmp_path, "again", RandomSearch(list_space(), LIST, sampler=_sampler(3), seed=3),
              resume=load_design_records(pool, {}), n_passes=3)

    assert [row["structure"] for row in _rows(tmp_path / "again.csv")] == [
        row["structure"] for row in _rows(tmp_path / "first.csv")
    ]


def test_the_other_arms_take_the_design_over_as_data(tmp_path):
    """A second Bayesian optimization on another seed does not hold the shared design against its
    own stream: the design is the first arm's."""
    first, second = _bo(4), _bo(9)
    run_paired(
        {"a": first, "b": second}, _metrics, schema=SCHEMA, n_design=3, n_passes=1,
        csv_paths={"a": str(tmp_path / "a.csv"), "b": str(tmp_path / "b.csv")},
        pretty_algebra=dict, echo=_quiet,
    )
    assert [row["structure"] for row in _rows(tmp_path / "b.csv")[:3]] == [
        row["structure"] for row in _rows(tmp_path / "a.csv")[:3]
    ]


def test_a_paired_resume_without_passes_holds_its_design_against_the_stream_as_one_run_would(
    tmp_path,
):
    """Without passes the pair draws nothing up front, and its first arm holds a drawn resumed
    design against the head of its stream, as it would run alone: a pool of another seed is
    refused, one of its own seed resumes."""
    pool = _measured(tmp_path, "first", _bo(1), n_design=3)

    def pair(seed: int, name: str) -> Any:
        loop = _bo(seed)
        return run_paired(
            {"bo": loop, "random": _twin_of(loop, seed)}, _metrics, schema=SCHEMA, n_passes=0,
            resume=load_design_records(pool, {}), pretty_algebra=dict, echo=_quiet,
            csv_paths={"bo": str(tmp_path / f"{name}.csv"),
                       "random": str(tmp_path / f"{name}_random.csv")},
        )

    with pytest.raises(ValueError, match="different stream"):
        pair(2, "other_seed")
    assert not (tmp_path / "other_seed.csv").exists(), "refused before anything was opened"
    outcomes = pair(1, "own_seed")
    assert outcomes["bo"].summary["taken_over"] == 3
