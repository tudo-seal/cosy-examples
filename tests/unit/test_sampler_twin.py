"""A second sampler of one program that draws the same stream and counts nothing again.

A paired comparison runs its random arm on the stream the loop's design was drawn from, past the
design, as the paired arm always did. The arm is another strategy object and needs a sampler of its
own; building a second one would count the program again, which on a large program takes
minutes. A twin shares the first one's counting table and draws from its own random source.
"""

from __future__ import annotations

import random
from itertools import islice

import cosy.search.samplers as samplers_module
import pytest
from cosy.search import DepthBoundedRandomSampler, SizeUniformSampler
from cosy.search.determinize import determinize

from bayesian_optimization import RandomSearch
from bayesian_optimization.initial_sampling import _sample_fallback_tree, distinct_prefix
from bayesian_optimization.runs import (
    DeterminizedSizeUniformSampler,
    check_resumed_design,
    draw_design,
    twin_sampler,
)
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


def _head(sampler_of_seed, query, count):
    """The first ``count`` terms of a fresh sampler's stream, for a seed."""
    return list(islice(sampler_of_seed().sample(query), count))


@pytest.mark.parametrize("kind", ["size-uniform", "determinized", "depth-bounded"])
def test_a_twin_made_after_the_first_drew_draws_its_own_seed_s_stream_from_the_start(kind):
    """The twin's stream is the one its random source gives, not the first sampler's state."""
    space = list_space()

    def fresh(seed):
        if kind == "size-uniform":
            return SizeUniformSampler(6, random.Random(seed))
        if kind == "determinized":
            return DeterminizedSizeUniformSampler(
                determinize(space, LIST), space, LIST, 6, random.Random(seed)
            )
        return DepthBoundedRandomSampler(4, random.Random(seed))

    first = fresh(5)
    query = _query(space, first)
    list(islice(first.sample(query), 3))  # the first sampler has drawn

    twin = twin_sampler(first, query, random.Random(5))

    assert list(islice(twin.sample(query), 6)) == _head(lambda: fresh(5), query, 6)


def test_a_determinized_twin_leaves_the_first_sampler_s_stream_as_it_was():
    space = list_space()

    def fresh(seed):
        return DeterminizedSizeUniformSampler(
            determinize(space, LIST), space, LIST, 6, random.Random(seed)
        )

    first = fresh(2)
    query = _query(space, first)
    twin_sampler(first, query, random.Random(99))

    assert list(islice(first.sample(query), 6)) == _head(lambda: fresh(2), query, 6)


def test_a_twin_of_the_size_uniform_sampler_draws_the_same_stream_and_counts_once(tables_built):
    """For one query object, as the determinized sampler always has; two loops' queries do not."""
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


# --- the paired design, drawn up front --------------------------------------------------------------


def test_the_design_is_the_head_of_the_stream_and_the_sampler_is_left_past_the_paired_terms():
    """Drawn as the paired arm always drew it: the design and the arm's terms from one stream, so the
    loop's sampler lies past everything the arm draws; its fallback then opens a stream there."""
    space = list_space()
    sampler = SizeUniformSampler(6, random.Random(4))
    query = _query(space, sampler)

    design, repeats = draw_design(sampler, query, 3, then=2)

    reference = SizeUniformSampler(6, random.Random(4))
    stream, _ = distinct_prefix(reference, query, 5)
    assert (design, repeats) == (stream[:3], 0)
    assert sampler.rng.getstate() == reference.rng.getstate(), "where drawing all five left it"


def test_the_loop_s_fallback_after_the_paired_draw_is_not_the_random_arm_s_first_term():
    space = list_space()
    sampler = SizeUniformSampler(6, random.Random(9))
    query = _query(space, sampler)
    arm = twin_sampler(sampler, query, random.Random(9))

    design, _ = draw_design(sampler, query, 2, then=2)

    arms_first = next(term for term in arm.sample(query) if term not in design)
    assert _sample_fallback_tree(sampler, query, set(design)) != arms_first
    untouched = SizeUniformSampler(6, random.Random(9))
    assert _sample_fallback_tree(untouched, query, set(design)) == arms_first, (
        "the collision a sampler that never drew produces, which the draw exists to avoid"
    )


def test_a_given_design_heads_the_stream_and_only_the_paired_terms_move_the_sampler():
    space = list_space()
    other = SizeUniformSampler(6, random.Random(1))
    head, _ = distinct_prefix(other, _query(space, other), 2)
    fresh_state = random.Random(3).getstate()

    alone = SizeUniformSampler(6, random.Random(3))
    design, _ = draw_design(alone, _query(space, alone), 2, design=head)
    assert design == head
    assert alone.rng.getstate() == fresh_state, "the design came from the list, not the sampler"

    paired = SizeUniformSampler(6, random.Random(3))
    design, _ = draw_design(paired, _query(space, paired), 2, then=2, design=head)
    assert design == head
    assert paired.rng.getstate() != fresh_state, "the arm's two terms were drawn past it"


@pytest.mark.parametrize(
    ("size", "then", "given", "match"),
    [
        (2, -1, None, "past the design"),
        (-1, 2, None, "non-negative number of terms, not -1"),
        (2, 0, 3, "holds 3 terms"),
        (3, 0, 2, "holds 2 terms"),
        (3, 0, "repeat", "repeats a term"),
    ],
)
def test_a_draw_that_would_not_be_the_design_asked_for_is_refused(size, then, given, match):
    """Each of these returned a design, and not the one asked for: a negative count of paired terms
    shortened the stream under the design, a negative size sliced it from the end, a given design
    of another size was cut or topped up from the sampler, and a repeat in a given design was
    replaced by the sampler's next term.  Refused before the stream is opened."""
    space = list_space()
    other = SizeUniformSampler(6, random.Random(1))
    head, _ = distinct_prefix(other, _query(space, other), 3)
    design = None if given is None else [head[0], head[0], head[1]] if given == "repeat" else head[:given]
    sampler = SizeUniformSampler(6, random.Random(4))
    query = _query(space, sampler)
    state = sampler.rng.getstate()

    with pytest.raises(ValueError, match=match):
        draw_design(sampler, query, size, then=then, design=design)
    assert sampler.rng.getstate() == state, "refused before anything was drawn"


def test_a_resumed_design_is_held_against_the_head_of_the_stream():
    space = list_space()
    sampler = SizeUniformSampler(6, random.Random(4))
    query = _query(space, sampler)
    design, _ = draw_design(sampler, query, 3)
    records = [(term, {"score": 0.5}) for term in design]

    check_resumed_design(records, design)
    with pytest.raises(ValueError, match="different stream"):
        check_resumed_design(list(reversed(records)), design)
    with pytest.raises(ValueError, match="size of the initial design"):
        check_resumed_design(records[:2], design)
