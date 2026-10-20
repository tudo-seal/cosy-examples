"""The front of a run over a value and a cost, its hypervolume, and the best value under a cost.

Each reading is held against its definition, computed the slow way: the front by checking every
pair against every other, the hypervolume as the area of a union of boxes over the grid of all
their coordinates, the best value under a cost by a scan of all pairs.
"""

from __future__ import annotations

import math
import random

import pytest
from cosy.core.tree import Tree

from bayesian_optimization.runs import (
    TermPoolWriter,
    TermRecord,
    best_value_under,
    front_hypervolume,
    nondominated,
    read_term_pool,
    run_pairs,
)
from tests.unit.test_expected_hypervolume_improvement import hypervolume as area_of_the_boxes


def dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """A value at least as large, a cost at most as large, and one of the two strictly."""
    return a[0] >= b[0] and a[1] <= b[1] and (a[0] > b[0] or a[1] < b[1])


def front_by_definition(pairs: list[tuple[float, float]]) -> list[int]:
    return [i for i in range(len(pairs)) if not any(dominates(other, pairs[i]) for other in pairs)]


def random_pairs(rng: random.Random, most: int = 15) -> list[tuple[float, float]]:
    """Pairs on a coarse grid, so that values and costs tie, a third of the time with a copy."""
    count = rng.randint(0, most)
    pairs = [(float(rng.randint(0, 6)), float(rng.randint(0, 6))) for _ in range(count)]
    if pairs and rng.random() < 1 / 3:
        pairs.append(pairs[rng.randrange(len(pairs))])
    return pairs


# ---------------------------------------------------------------------------
# The three readings
# ---------------------------------------------------------------------------


def test_the_front_is_the_set_of_pairs_no_other_pair_dominates():
    """Over random pairs with ties and copies; two equal pairs do not dominate each other."""
    rng = random.Random(23)
    copies_on_a_front = 0
    for _trial in range(500):
        pairs = random_pairs(rng)
        front = nondominated(pairs)
        assert front == front_by_definition(pairs), pairs
        copies_on_a_front += len({pairs[i] for i in front}) < len(front)
    # The control, counted from the definition alone (116 of 500 for this seed): copies standing
    # together on a front occur, so the tie rule is exercised.
    assert copies_on_a_front >= 100


def test_the_hypervolume_is_the_area_of_the_union_of_the_boxes():
    """Over random pairs, some of them outside the reference point."""
    rng = random.Random(29)
    positive = 0
    for _trial in range(300):
        pairs = random_pairs(rng)
        reference = (rng.choice([0.0, 1.5]), rng.choice([5.0, 7.0]))
        expected = area_of_the_boxes(pairs, reference)
        assert front_hypervolume(pairs, reference) == pytest.approx(expected), (pairs, reference)
        positive += expected > 0.0
    # The control, counted from the oracle alone (278 of 300 for this seed): most areas are not
    # empty.
    assert positive >= 250


def test_the_best_value_under_a_cost_is_the_largest_value_at_that_cost_or_below():
    """Thresholds below, on and between the costs; none where no pair is that cheap."""
    rng = random.Random(31)
    nones = 0
    for _trial in range(300):
        pairs = random_pairs(rng)
        thresholds = [rng.choice([-1.0, 0.0, 2.0, 3.5, 6.0, 9.0]) for _ in range(4)]
        expected = [
            max((value for value, cost in pairs if cost <= threshold), default=None)
            for threshold in thresholds
        ]
        assert best_value_under(pairs, thresholds) == expected
        nones += expected.count(None)
    # The control, counted from the scan alone (342 of 1 200 for this seed): thresholds below every
    # cost occur.
    assert nones >= 300


def test_the_front_of_no_pairs_is_empty_and_covers_nothing():
    assert nondominated([]) == []
    assert front_hypervolume([], (0.0, 1.0)) == 0.0
    assert best_value_under([], [1.0]) == [None]


# ---------------------------------------------------------------------------
# Reading the pairs off a run's records
# ---------------------------------------------------------------------------


def _records() -> list[TermRecord]:
    return [
        TermRecord("pre_sample", 0, Tree("a"), {"accuracy": 0.5, "params": 10.0}),
        TermRecord("pre_sample", 1, Tree("b", (Tree("c"),)), {"accuracy": 0.7, "params": 30.0}),
        TermRecord("bo_step", 0, Tree("d"), {"accuracy": 0.6, "params": 5.0}),
    ]


def test_the_pairs_are_read_by_metric_or_by_function_and_by_phase():
    records = _records()
    expected = [(0.5, 10.0), (0.7, 30.0), (0.6, 5.0)]
    assert run_pairs(records, value="accuracy", cost="params") == expected
    by_size = run_pairs(records, value="accuracy", cost=lambda record: float(record.term.size))
    assert by_size == [(0.5, 1.0), (0.7, 2.0), (0.6, 1.0)]
    assert run_pairs(records, value="accuracy", cost="params", phases={"bo_step"}) == [(0.6, 5.0)]


def test_the_pairs_survive_a_round_trip_through_the_term_pool(tmp_path):
    """The records a run writes are the records the readings read."""
    path = str(tmp_path / "run_terms.pickle")
    with TermPoolWriter(path) as writer:
        for record in _records():
            writer.write(record.phase, record.index, record.term, record.metrics)
    _header, records = read_term_pool(path)
    expected = [(0.5, 10.0), (0.7, 30.0), (0.6, 5.0)]
    assert run_pairs(records, value="accuracy", cost="params") == expected


def test_a_record_without_the_metric_is_refused_naming_it():
    records = [*_records(), TermRecord("bo_step", 1, Tree("e"), {"accuracy": 0.9})]
    with pytest.raises(KeyError, match="the record bo_step 1 carries no 'params'"):
        run_pairs(records, value="accuracy", cost="params")


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_a_value_or_a_cost_that_is_not_finite_is_refused_naming_the_record(bad):
    """A failed measurement is refused rather than placed on a front.

    Args:
        bad (float): The value that is not finite.
    """
    records = [*_records(), TermRecord("bo_step", 1, Tree("e"), {"accuracy": bad, "params": 1.0})]
    with pytest.raises(ValueError, match="the value of the record bo_step 1 is not finite"):
        run_pairs(records, value="accuracy", cost="params")
    records = [*_records(), TermRecord("bo_step", 1, Tree("e"), {"accuracy": 0.1, "params": bad})]
    with pytest.raises(ValueError, match="the cost of the record bo_step 1 is not finite"):
        run_pairs(records, value="accuracy", cost="params")
