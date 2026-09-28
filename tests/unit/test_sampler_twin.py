"""A second sampler of one program that draws the same stream and counts nothing again.

A paired comparison runs its random arm on the stream the loop's design was drawn from, past the
design, as the paired arm always did. The arm is another strategy object and needs a sampler of its
own; building a second one would count the program again, which on the seven-position whistle
query of the accompanying project took 18 minutes. A twin shares the first one's counting table
and draws from its own random source.
"""

from __future__ import annotations

import random
from itertools import islice

import cosy.search.samplers as samplers_module
import pytest
from cosy.search import DepthBoundedRandomSampler, SizeUniformSampler
from cosy.search.determinize import determinize

from bayesian_optimization import RandomSearch
from bayesian_optimization.runs import DeterminizedSizeUniformSampler, twin_sampler
from tests.spaces import LIST, list_space


@pytest.fixture
def tables_built(monkeypatch):
    """Count the counting constructions built, table or tree, the step a twin must not repeat."""
    built = []
    for name in ("weighted_table", "weighted_tree"):
        original = getattr(samplers_module, name)

        def counting(*args, _original=original, **kwargs):
            built.append(args[0])
            return _original(*args, **kwargs)

        monkeypatch.setattr(samplers_module, name, counting)
    return built


def _query(space, sampler):
    return RandomSearch(space, LIST, sampler=sampler, seed=0).query


def test_a_twin_of_the_size_uniform_sampler_draws_the_same_stream_and_counts_once(tables_built):
    space = list_space()
    first = SizeUniformSampler(6, random.Random(7))
    query = _query(space, first)

    twin = twin_sampler(first, query, random.Random(7))
    head = list(islice(first.sample(query), 8))
    again = list(islice(twin.sample(query), 8))

    assert len(head) == 8 and again == head
    assert len(tables_built) == 1, "the twin shares the table built for the first"
    assert twin._weighted is first._weighted, "shared, not copied"
    other = twin_sampler(first, query, random.Random(8))
    assert list(islice(other.sample(query), 8)) != head, "another seed draws another stream"


def test_a_twin_of_the_determinized_sampler_shares_the_inner_table(tables_built):
    space = list_space()
    first = DeterminizedSizeUniformSampler(determinize(space, LIST), space, LIST, 6, random.Random(3))
    query = _query(space, first)

    twin = twin_sampler(first, query, random.Random(3))
    head = list(islice(first.sample(query), 6))
    again = list(islice(twin.sample(query), 6))

    assert len(head) == 6 and again == head
    assert len(tables_built) == 1
    assert twin._inner._weighted is first._inner._weighted, "shared, not copied"
    assert (twin.size_bound, twin.counting) == (first.size_bound, first.counting)


def test_the_first_sampler_draws_on_as_if_no_twin_were_made():
    space = list_space()
    alone = SizeUniformSampler(6, random.Random(5))
    query = _query(space, alone)
    expected = list(islice(alone.sample(query), 6))

    first = SizeUniformSampler(6, random.Random(5))
    twin_sampler(first, query, random.Random(99))
    assert list(islice(first.sample(query), 6)) == expected


def test_a_twin_of_the_depth_bounded_sampler_draws_the_same_stream():
    space = list_space()
    first = DepthBoundedRandomSampler(4, random.Random(11))
    query = _query(space, first)

    twin = twin_sampler(first, query, random.Random(11))

    assert list(islice(twin.sample(query), 6)) == list(islice(first.sample(query), 6))


def test_a_sampler_without_a_random_source_has_no_twin():
    class Fixed:
        def sample(self, query):
            yield from ()

    with pytest.raises(TypeError, match="rng"):
        twin_sampler(Fixed(), None, random.Random(0))
