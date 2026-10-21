"""The ask/tell loops under an exchangeable duplicate test: what the test merges is evaluated once.

The test here calls two expressions one evaluation when they have one value, and the measurement
depends on the value alone, so that what it merges would measure alike. Random search evaluates the
first terms of its stream the test keeps apart, Bayesian optimization proposes nothing the test
merges with a measured or an outstanding term, NSGA-II answers a merged offspring from the term it
measured, and a drawn design is pairwise distinct under the test. Under term equality, the default,
every loop runs exactly as it did.
"""

from __future__ import annotations

import hashlib
import random

import pytest
from cosy.search import DepthBoundedRandomSampler, SizeUniformSampler, generator_query

from bayesian_optimization import NSGA2, RandomSearch
from bayesian_optimization.bo import BayesianOptimization
from bayesian_optimization.duplicates import TERM_EQUALITY, DuplicateTest
from bayesian_optimization.initial_sampling import distinct_prefix
from tests.integration.test_real_ea_loop import _make_ea
from tests.spaces import EXPR, add, expression_space, lit, neg

#: The default mutation's depth bound on the infinite expression space, as tests/unit/test_nsga2.py sets it.
DEPTH = 5


def value(term) -> int:
    """An expression's value: ``lit`` is 1, ``neg`` negates, ``add`` adds."""
    if term.root is lit:
        return 1
    if term.root is neg:
        return -value(term.children[0])
    assert term.root is add
    return value(term.children[0]) + value(term.children[1])


class SameValue(DuplicateTest):
    """Two expressions with one value are one evaluation."""

    def key(self, term):
        return value(term)

    def same(self, a, b) -> bool:
        return value(a) == value(b)


class Parity(DuplicateTest):
    """Two expressions of one parity are one evaluation: a test that two terms exhaust."""

    def key(self, term):
        return value(term) % 2

    def same(self, a, b) -> bool:
        return value(a) % 2 == value(b) % 2


class Bounded:
    """A sampler that refuses to draw past a limit, so that a guard which never fires fails rather than spins."""

    def __init__(self, inner, limit: int) -> None:
        self.inner, self.limit, self.drawn = inner, limit, 0

    def sample(self, query):
        for term in self.inner.sample(query):
            self.drawn += 1
            if self.drawn > self.limit:
                raise AssertionError(f"drew {self.limit} terms and no guard fired")
            yield term

    def at_least(self, query, count):
        return self.inner.at_least(query, count)


def measure(term) -> float:
    """A measurement of the value alone, deterministic and spread over [0, 1)."""
    digest = hashlib.sha256(str(value(term)).encode()).digest()
    return int.from_bytes(digest[:6], "big") / 2**48


def run(loop, size: int, passes: int) -> list:
    """Draw a design of ``size``, then make ``passes`` passes; the terms in the order measured."""
    loop.initialize(initial_size=size)
    measured = []
    for _ in range(size + passes):
        suggestion = loop.suggest()
        measured.append(suggestion.candidate)
        loop.observe(suggestion.candidate, measure(suggestion.candidate))
    return measured


#: The size bound of the streams: under it the expression space holds terms of more than eight values.
BOUND = 9


def merged_pair():
    """Two distinct terms of one value, from the head of a stream."""
    drawn = [term for term, _ in zip(SizeUniformSampler(BOUND, random.Random(3)).sample(
        generator_query(expression_space(), EXPR)), range(40))]
    return next((x, y) for i, x in enumerate(drawn) for y in drawn[i + 1:] if value(x) == value(y))


def random_search(seed: int, test=None) -> RandomSearch:
    return RandomSearch(expression_space(), EXPR, sampler=SizeUniformSampler(BOUND, random.Random(seed)), seed=seed,
                        duplicate_test=test)


def bayesian(seed: int, test=None, sampler=None) -> BayesianOptimization:
    space = expression_space()
    return BayesianOptimization(search_space=space, request=EXPR,
                                sampler=sampler or SizeUniformSampler(BOUND, random.Random(seed)), seed=seed,
                                optimizer=_make_ea(space, seed), n_restarts_kernel_optimizer=0, duplicate_test=test)


def test_random_search_evaluates_the_first_terms_of_its_stream_the_test_keeps_apart():
    measured = run(random_search(3, SameValue()), 3, 5)
    expected, values = [], set()
    for term in SizeUniformSampler(BOUND, random.Random(3)).sample(generator_query(expression_space(), EXPR)):
        if value(term) not in values:
            values.add(value(term))
            expected.append(term)
        if len(expected) == len(measured):
            break
    assert measured == expected
    plain = run(random_search(3), 3, 5)
    assert len({value(term) for term in plain}) < len(plain), "the stream must repeat a value, or nothing is merged"


def test_random_search_counts_a_term_the_test_merges_with_a_held_one_against_its_draws():
    sampler = Bounded(DepthBoundedRandomSampler(8, random.Random(0)), 2000)
    search = RandomSearch(expression_space(), EXPR, sampler=sampler, seed=0, max_draws_per_term=20,
                          duplicate_test=Parity())
    run(search, 2, 0)
    drawn = sampler.drawn
    with pytest.raises(RuntimeError, match="the duplicate test merges with a held one"):
        search.suggest()
    assert sampler.drawn - drawn == 20, "the bound is the number of merged draws in a row, no more, no less"


def test_bayesian_optimization_never_proposes_what_the_test_merges():
    measured = run(bayesian(1, SameValue()), 3, 6)
    values = [value(term) for term in measured]
    assert len(set(values)) == len(values)
    plain = run(bayesian(1), 3, 6)
    assert len({value(term) for term in plain}) < len(plain), "the run must repeat a value, or nothing is merged"


def test_nsga2_answers_what_the_test_merges_from_the_term_it_measured():
    space = expression_space()
    design, _ = distinct_prefix(SizeUniformSampler(9, random.Random(0)), generator_query(space, EXPR), 6,
                                test=SameValue())
    search = NSGA2(space, EXPR, cost=lambda term: float(term.size), depth_bound=DEPTH, seed=0,
                   duplicate_test=SameValue())
    search.initialize(design=design)
    measured = []
    for _ in range(len(design) + 12):
        suggestion = search.suggest()
        measured.append(suggestion.candidate)
        search.observe(suggestion.candidate, measure(suggestion.candidate))
    values = [value(term) for term in measured]
    assert len(set(values)) == len(values)
    assert search.offspring_known > 0, "no offspring merged with a measured term, so nothing was answered"
    for formed in search.formed_generations:
        for member, row in zip(formed.members, formed.rows, strict=True):
            assert value(measured[row]) == value(member)


def test_nsga2_reads_a_merged_offsprings_cost_when_it_breeds_it_and_none_while_it_takes_a_value_back():
    space = expression_space()
    design, _ = distinct_prefix(SizeUniformSampler(9, random.Random(5)), generator_query(space, EXPR), 6,
                                test=SameValue())
    events: list[tuple[str, object]] = []
    where = ["initialize"]

    def cost(term) -> float:
        events.append((where[0], term))
        return float(term.size)

    search = NSGA2(space, EXPR, cost=cost, depth_bound=DEPTH, seed=5, duplicate_test=SameValue())
    search.initialize(design=design)
    # Three passes: the fourth generation of this seed breeds nothing new, which is convergence.
    for _ in range(len(design) + 3):
        where[0] = "suggest"
        suggestion = search.suggest()
        where[0] = "observe"
        search.observe(suggestion.candidate, measure(suggestion.candidate))
        events.append(("observed", suggestion.candidate))
    assert [term for place, term in events if place == "observe"] == []
    observed = {term for place, term in events if place == "observed"}
    measured_values: set[int] = set()
    early = 0
    for place, term in events:
        if place == "observed":
            measured_values.add(value(term))
        elif place == "suggest" and term not in observed and value(term) not in measured_values:
            early += 1
    assert early > 0, "no offspring was merged with one waiting in its generation, so no cost was read for it"


def test_a_drawn_design_is_pairwise_distinct_under_the_test():
    query = generator_query(expression_space(), EXPR)
    design, rejected = distinct_prefix(DepthBoundedRandomSampler(3, random.Random(0)), query, 5, test=SameValue())
    assert len({value(term) for term in design}) == 5
    assert rejected > 0, "the stream must offer a merged term, or the design rejects nothing"
    plain, _ = distinct_prefix(DepthBoundedRandomSampler(3, random.Random(0)), query, 5)
    assert len({value(term) for term in plain}) < 5, "under term equality the same stream's design repeats a value"
    bo = bayesian(0, SameValue(), sampler=DepthBoundedRandomSampler(3, random.Random(0)))
    bo.initialize(initial_size=5)
    assert len({value(term) for term in bo.design}) == 5
    plain_bo = bayesian(0, sampler=DepthBoundedRandomSampler(3, random.Random(0)))
    plain_bo.initialize(initial_size=5)
    assert len({value(term) for term in plain_bo.design}) < 5, "under term equality the loop's design repeats a value"


@pytest.mark.parametrize("make", [random_search, bayesian])
def test_a_design_handed_over_whose_terms_the_test_merges_is_refused_as_a_repeated_one_is(make):
    a, b = merged_pair()
    assert a != b
    with pytest.raises(ValueError, match="the duplicate test calls two terms of the design one evaluation"):
        make(0, SameValue()).initialize(design=[a, b])
    loop = make(0)
    loop.initialize(design=[a, b])
    assert loop.design == (a, b), "under term equality two distinct terms are a design"


def test_the_loops_own_design_under_the_test_is_the_head_of_its_stream_the_run_layer_draws():
    query = generator_query(expression_space(), EXPR)
    head, rejected = distinct_prefix(SizeUniformSampler(BOUND, random.Random(4)), query, 6, test=SameValue())
    assert rejected > 0, "the stream must offer a merged term within the design, or both draws agree by construction"
    loop = bayesian(4, SameValue())
    loop.initialize(initial_size=6)
    assert list(loop.design) == head
    assert loop.initial_repeats_rejected == rejected


@pytest.mark.parametrize("make", [random_search, bayesian])
def test_under_term_equality_a_loop_runs_as_it_did(make):
    default = run(make(7), 3, 4)
    for test in (TERM_EQUALITY, DuplicateTest()):
        assert run(make(7, test), 3, 4) == default


def test_something_that_is_no_duplicate_test_is_refused_by_name():
    with pytest.raises(TypeError, match="duplicate_test says when two terms are one evaluation"):
        RandomSearch(expression_space(), EXPR, seed=0, duplicate_test=lambda a, b: a == b)
