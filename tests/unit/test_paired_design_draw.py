"""The paired baseline draws its design from the loop's own sampler, and takes no shortcut.

``_draw_prefix`` is the one production caller of ``distinct_prefix``.  What it has to get right is
small and easy to get wrong: it has to draw through the query the loop will use, so that the
sampler's counting construction is built once, and it has to refuse a loop that has nothing to draw
from rather than hand back a short design.

The terms here are bare ``Tree`` leaves and the sampler is a list.  What is under test is the
bookkeeping around the draw, not the drawing.
"""

from __future__ import annotations

import pytest
from cosy.core.tree import Tree

from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_experiment_utils as utils


class ScriptedSampler:
    """A sampler that yields a fixed sequence, and records which query it was asked."""

    def __init__(self, terms):
        self.terms = list(terms)
        self.queries = []

    def sample(self, query):
        """Yield the scripted terms.

        Args:
            query: The query the caller handed in, recorded for the assertions.

        Yields:
            Tree: The scripted terms, in order.
        """
        self.queries.append(query)
        yield from self.terms


class FakeLoop:
    """The two attributes ``_draw_prefix`` reads off a configured loop."""

    def __init__(self, sampler, query):
        self.sampler = sampler
        self.query = query


def test_the_design_is_drawn_through_the_loop_s_own_query():
    """One stream, one query, and the terms in the order the stream produced them.

    A second query object would make the sampler build its counting construction again, which is
    the cost the loop pays once per run, so the query has to be the loop's own.
    """
    sampler = ScriptedSampler([Tree("a"), Tree("b"), Tree("c"), Tree("d")])
    loop = FakeLoop(sampler, query="the loop's query")

    terms, rejected = utils._draw_prefix(loop, 3)

    assert terms == [Tree("a"), Tree("b"), Tree("c")]
    assert rejected == 0
    assert sampler.queries == ["the loop's query"], "one stream, opened on the loop's own query"


def test_a_stream_that_repeats_is_repaired_and_the_repair_is_counted():
    """A repeat is a training point that carries no observation the previous one did not.

    The size-uniform sampler repeats nothing, so the count comes back zero there and is a check
    rather than a repair.  A depth-bounded stream is a sequence of independent draws and may repeat,
    and then the number says how much the rejection had to do.  It goes into the run record, so it
    has to be the number of rejections and not a flag.
    """
    sampler = ScriptedSampler([Tree("a"), Tree("a"), Tree("b"), Tree("a"), Tree("c")])
    loop = FakeLoop(sampler, query="q")

    terms, rejected = utils._draw_prefix(loop, 3)

    assert terms == [Tree("a"), Tree("b"), Tree("c")]
    assert len(set(terms)) == 3
    assert rejected == 2


def test_a_loop_with_nothing_to_draw_from_is_refused():
    """A short design is not a design, so a missing sampler is an error and not an empty list.

    A loop that was never given a search space has no stream to pair a baseline against, and
    continuing would produce a baseline of whatever length happened to come out.
    """
    with pytest.raises(RuntimeError, match="no search space or no sampler"):
        utils._draw_prefix(FakeLoop(sampler=None, query="q"), 3)
    with pytest.raises(RuntimeError, match="no search space or no sampler"):
        utils._draw_prefix(FakeLoop(sampler=ScriptedSampler([]), query=None), 3)
