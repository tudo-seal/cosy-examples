"""When two terms are one evaluation: the loop's exchangeable duplicate test and the terms held under it.

The default is what every loop always did, term equality, so ``KnownTerms`` under it must hold exactly
what a set of terms holds. A test of a caller's own merges what it calls one evaluation: here, two
expressions with one value.
"""

from __future__ import annotations

import random

import pytest
from cosy.core.tree import Tree
from cosy.search import DepthBoundedRandomSampler, generator_query

from bayesian_optimization.duplicates import TERM_EQUALITY, DuplicateTest, KnownTerms
from tests.spaces import EXPR, add, expression_space, lit, neg


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


class OneBucket(DuplicateTest):
    """Every term in one bucket, told apart by equality: the key only narrows the search."""

    def key(self, term):
        return 0


def _drawn(count: int, seed: int = 0) -> list:
    """Terms from a sampler that draws with replacement, so that some repeat."""
    stream = DepthBoundedRandomSampler(4, random.Random(seed)).sample(generator_query(expression_space(), EXPR))
    return [next(stream) for _ in range(count)]


def test_under_the_default_the_terms_held_are_a_set_of_terms():
    terms = _drawn(300)
    held = set()
    known = KnownTerms(test=TERM_EQUALITY)
    for term in terms:
        assert (term in known) == (term in held)
        known.add(term)
        held.add(term)
    assert len(known) == len(held)
    assert set(known) == held
    assert len(terms) > len(held), "the draws must repeat a term, or this compares nothing"


def test_a_test_of_its_own_merges_what_it_calls_one_evaluation():
    one = _drawn(1)[0]
    terms = _drawn(400, seed=1)
    known = KnownTerms(test=SameValue())
    for term in terms:
        known.add(term)
    values = {value(term) for term in terms}
    assert len(known) == len(values)
    assert len(values) < len({*terms}), "two distinct terms must share a value, or nothing is merged"
    for term in terms:
        held = known.match(term)
        assert held is not None and value(held) == value(term)
        assert term in known
    absent = KnownTerms([t for t in terms if value(t) != value(one)], test=SameValue())
    assert one not in absent
    assert absent.match(one) is None


def test_a_shared_key_alone_merges_nothing():
    terms = _drawn(300, seed=2)
    known = KnownTerms(terms, test=OneBucket())
    assert len(known) == len(set(terms))
    assert all(term in known for term in terms)
    held, rest = terms[:150], terms[150:]
    partial = KnownTerms(held, test=OneBucket())
    outside = [term for term in rest if term not in set(held)]
    assert outside, "some draws must lie outside the held half, or this checks nothing"
    assert all(term not in partial and partial.match(term) is None for term in outside)


def test_a_union_holds_both_and_leaves_the_first_as_it_was():
    first, second = _drawn(40, seed=3), _drawn(40, seed=4)
    known = KnownTerms(first, test=TERM_EQUALITY)
    both = known.union(second)
    assert set(both) == set(first) | set(second)
    assert set(known) == set(first)
    assert both.test is known.test


def test_a_test_that_overrides_same_and_not_key_is_refused_where_it_is_defined():
    # Under the default key, the term itself, same() would be asked only of equal terms.
    with pytest.raises(TypeError, match="overrides same\\(\\) and not key\\(\\)"):

        class OnlySame(DuplicateTest):
            def same(self, a, b) -> bool:
                return value(a) == value(b)

    class KeyOnly(DuplicateTest):
        def key(self, term):
            return value(term)

    class BothFromTheBase(SameValue):
        def same(self, a, b) -> bool:
            return value(a) == value(b)

    assert Tree(add, (Tree(lit, ()), Tree(lit, ()))) not in KnownTerms([Tree(lit, ())], KeyOnly())
    assert len(KnownTerms([Tree(lit, ()), Tree(neg, (Tree(neg, (Tree(lit, ()),)),))], BothFromTheBase())) == 1
