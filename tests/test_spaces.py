"""The reference spaces are what the thesis says they are (the copy is pinned, not trusted).

:mod:`tests.spaces` restates the five spaces of cosy's ``tests/search_fixtures.py`` because
that test package is not installed and both repositories name theirs ``tests``.  A copy is a place
for two things to drift apart, so this file pins each space by the property it was built for (the
closed-form counts, the coupling, the ambiguity) rather than by identity with the original.
If a copy ever loses a clause, the experiments built on it stop measuring what they claim to.
"""

from __future__ import annotations

import pytest
from cosy.search import checker, term_size

from tests.oracles import cost_counts, inhabitants_within, render
from tests.spaces import (
    AMBIGUOUS_SIGNATURE,
    AMBIGUOUS_TARGET,
    AVL,
    EXPR,
    EXPR_SIGNATURE,
    LIST,
    LIST_SIGNATURE,
    LITERAL_SIGNATURE,
    PAIR,
    PAIR_SIGNATURE,
    TAGGED,
    ambiguous_space,
    avl_coupled_space,
    avl_space,
    avl_trees,
    constrained_space,
    coupled_cores_space,
    expression_space,
    list_space,
    literal_space,
    recognizable_pair_space,
)


@pytest.mark.parametrize("bound", [1, 2, 3, 4, 5])
def test_the_list_space_has_three_to_the_length_lists_of_each_size(bound):
    """The list space holds ``3^l`` lists of size ``l + 1``, and nothing else about it matters.

    The size-uniform draw built on this space is calibrated from that count, so every length
    carries the same total weight and the sampler spreads over the lengths instead of
    concentrating on the short lists.

    Args:
        bound (int): The size bound to check the counts at.
    """
    counts = cost_counts(
        inhabitants_within(list_space(), LIST, LIST_SIGNATURE, bound), term_size
    )
    assert counts == {size: 3 ** (size - 1) for size in range(1, bound + 1)}


def test_the_expression_space_counts_the_motzkin_numbers():
    """A unary and a binary combinator, so the size counts follow no power of a constant.

    The number of expressions of size ``n`` is the ``(n-1)``-th Motzkin number (1, 1, 2, 4, 9,
    21, 51), which is what makes this space a check on a counting recursion rather than on a
    formula that happens to be exponential.
    """
    counts = cost_counts(
        inhabitants_within(expression_space(), EXPR, EXPR_SIGNATURE, 7), term_size
    )
    assert counts == {1: 1, 2: 1, 3: 2, 4: 4, 5: 9, 6: 21, 7: 51}


def test_the_constrained_space_rejects_exactly_the_pairs_of_equal_words():
    """The external predicate is what couples the two holes, so it has to bite.

    Within size 5 the words run from ``zero`` and ``one`` up to ``w(w(one))``, and the pairs the
    predicate admits are the ordered pairs of *distinct* words.
    """
    inhabitants = inhabitants_within(constrained_space(), PAIR, PAIR_SIGNATURE, 5)
    rendered = {render(tree) for tree in inhabitants}
    assert "pair(zero, one)" in rendered
    assert "pair(zero, zero)" not in rendered
    assert "pair(wrap(zero), wrap(zero))" not in rendered
    assert all(tree.children[0] != tree.children[1] for tree in inhabitants)


def test_the_ambiguous_space_derives_one_inhabitant_twice_and_the_other_once():
    """``merge(base)`` has two derivations and ``merge(alt)`` one, so both cases are present.

    Checked through the branch counts against the brute-force set: the counts count success
    branches, so their total exceeds the number of inhabitants by exactly the ambiguity.  A space
    that had lost its intersection would show no excess and the ambiguity experiments would
    silently measure nothing.
    """
    from cosy.search import branch_counts, generator_query

    space = ambiguous_space()
    inhabitants = inhabitants_within(space, AMBIGUOUS_TARGET, AMBIGUOUS_SIGNATURE, 2)
    counted = branch_counts(generator_query(space, AMBIGUOUS_TARGET), 2, term_size)
    assert {render(tree) for tree in inhabitants} == {"merge(base)", "merge(alt)"}
    assert counted.total == 3


def test_the_literal_space_writes_two_symbols_per_clause():
    """``tag`` fixes a terminal and a constant argument, so the realized sizes have gaps.

    Sizes 1, 3, 5 with 1, 2, 4 inhabitants: the size-uniform distribution therefore has to spread
    its mass over a set of values with holes in it, and a bookkeeping that counted clause
    applications instead of symbols would report sizes 1, 2, 3 here and be right everywhere else.
    """
    counts = cost_counts(
        inhabitants_within(literal_space(), TAGGED, LITERAL_SIGNATURE, 5), term_size
    )
    assert counts == {1: 1, 3: 2, 5: 4}


# ---------------------------------------------------------------------------
# The contrast pairs: two ways of stating one condition have to state *one* condition
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bound", [3, 4, 5, 6])
def test_the_two_ways_of_stating_the_core_condition_admit_the_same_words(bound):
    """The coupled space and the recognizable one define the same language, size by size.

    This is the premise every measurement on the pair rests on: if they differed, a distribution
    drawn from the determinized program could match its own space perfectly and still say nothing
    about the coupled one.  The predicate reads the terms and the relation reads the states, and
    neither is written in terms of the other, so the agreement is a fact about the abstraction.

    Args:
        bound (int): The size bound to compare the languages at.
    """
    coupled = inhabitants_within(coupled_cores_space(), PAIR, PAIR_SIGNATURE, bound)
    recognizable = inhabitants_within(recognizable_pair_space(), PAIR, PAIR_SIGNATURE, bound)
    assert set(coupled) == set(recognizable)
    assert cost_counts(coupled, term_size) == {
        size: 2 * (size - 2) for size in range(3, bound + 1)
    }


def test_the_cores_condition_rejects_pairs_of_equal_cores_and_keeps_unequal_words():
    """The condition is coarser than term equality, which is what makes it recognizable.

    ``pair(w(zero), zero)`` has two *different* words with the *same* core and is rejected here,
    while ``constrained_space`` accepts it.  Without this the pair would be indistinguishable from
    the term-equality space, and the point of (REC) would be lost.
    """
    rendered = {
        render(tree)
        for tree in inhabitants_within(coupled_cores_space(), PAIR, PAIR_SIGNATURE, 4)
    }
    assert "pair(zero, one)" in rendered
    assert "pair(wrap(zero), one)" in rendered
    assert "pair(wrap(zero), zero)" not in rendered
    assert "pair(zero, zero)" not in rendered


def test_the_avl_oracle_enumerates_the_trees_the_plan_counted_by_hand():
    """``avl_trees`` is the oracle for a space too big to enumerate over its signature.

    At three keys the whole language is eleven trees: one leaf, three single nodes,
    ``C(3,2) * 2 = 6`` two-node trees and one balanced three-node tree.  Both AVL spaces accept
    exactly these, so the oracle and the spaces agree without either being read off the other.
    """
    keys = (0, 1, 2)
    trees = avl_trees(keys)
    assert cost_counts(trees, term_size) == {1: 1, 5: 3, 9: 6, 13: 1}
    assert len(set(trees)) == len(trees), "the enumeration produces each tree once"

    coupled, recognizable = avl_coupled_space(keys), avl_space(keys)
    assert all(checker(coupled, AVL, tree) for tree in trees)
    assert all(checker(recognizable, AVL, tree) for tree in trees)


def test_the_avl_condition_rejects_a_tree_with_a_wrong_cached_height():
    """The cached height is part of the condition, not decoration.

    Goldstein and Pierce (2022) name it as the reason their generator struggles ("the generator
    must guess the correct height to cache at each node"), so a space that ignored it would make
    the whole comparison meaningless.  The witness is a valid tree with its root height
    incremented.
    """
    from cosy.core.tree import Tree

    from tests.spaces import leaf, node

    valid = Tree(node, (Tree(0, ()), Tree(1, ()), Tree(leaf, ()), Tree(leaf, ())))
    wrong = Tree(node, (Tree(0, ()), Tree(2, ()), Tree(leaf, ()), Tree(leaf, ())))
    keys = (0, 1, 2)
    assert checker(avl_coupled_space(keys), AVL, valid)
    assert not checker(avl_coupled_space(keys), AVL, wrong)
    assert not checker(avl_space(keys), AVL, wrong)


def test_the_chain_holds_one_inhabitant_per_length_and_its_oracle_agrees():
    """The chain of lists over one element holds one list per length ``l``, of size ``2l + 1``.

    Both halves matter for what is built on this space.  The size formula is what makes the bound
    ``D = 21`` cut the chain at length ten, the setting in which the kernel-diverse draws are
    traced.  One inhabitant per size is what makes the "term axis" of the figures drawn on this
    chain a genuine order, and what lets a kernel between two inhabitants be written in closed
    form at all.

    The direct enumerator replaces the suite's usual generate-and-check oracle here, since
    ``cons_c`` is binary and the terms over the signature up to size 21 run into the millions.  It
    is therefore checked against the space rather than assumed: every term it produces is
    accepted, and the space produces no others.
    """
    from tests.spaces import CHAIN, chain_space, chain_terms

    space = chain_space()
    terms = chain_terms(21)

    assert [term_size(term) for term in terms] == list(range(1, 22, 2))
    assert len(set(terms)) == len(terms)
    assert all(checker(space, CHAIN, term) for term in terms)
    enumerated = space.enumerate_trees(CHAIN, max_count=3 * len(terms))
    assert {term for term in enumerated if term_size(term) <= 21} == set(terms), (
        "the space holds exactly the chains the oracle builds, and no others within the bound"
    )
