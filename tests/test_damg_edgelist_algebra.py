"""The edge-list algebra evaluates each continuation once, and composes it in a fixed order.

A clause of the algebra returns a continuation, a function from a layout position and a list of
inputs to the triple of edges, outputs and node positions.  Composition has to take the triple
apart, and taking it apart is where a continuation can end up being called once per component.
The continuations are built from the same combinators, so a repeated call multiplies through the
nesting of a term rather than adding up.

The first test counts the calls against the number of leaf combinators the term has, over every
target the module ships.  The second and third pin the composed result itself, edge order
included, since evaluating once has to give what evaluating three times gave.
"""

from __future__ import annotations

import itertools
from collections import Counter

import pytest
from cosy.core import Synthesizer
from cosy.core.tree import Tree
from cosy.core.types import Constructor, Literal

from bayesian_optimization.examples.damg_nas import damg_targets
from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository
from bayesian_optimization.examples.damg_nas.damg_repo_algebras import (
    edgelist_algebra,
    hierarchy_algebra,
)

#: The clauses that stand for a graph node or a wiring, which is where a continuation is built.
#: One set per granularity of the term, since a fold renames the clauses it keeps.
LEAF_CLAUSES = ("edges", "swap", "linear_layer", "sigmoid", "relu", "tanh", "sum", "product",
                "copy", "edges_h1", "swap_h1", "linear_layer_h1", "sigmoid_h1", "relu_h1",
                "tanh_h1", "sum_h1", "product_h1", "copy_h1", "node")

#: How many terms of each target are counted.
SAMPLE = 30

#: The names of the twelve targets, paired with the number of leaves every one of their terms has.
TARGETS_AND_LEAVES = [
    ("target_len_2", 2),
    ("target_len_3", 3),
    ("target_len_3_refined_1", 3),
    ("target_len_3_refined_2", 4),
    ("target_len_3_refined_3", 5),
    ("target_len_4", 4),
    ("target_len_4_refined_1", 4),
    ("target_len_5", 5),
    ("target_len_5_refined_1", 5),
    ("target_len_5_refined_2", 8),
    ("target_len_5_refined_3", 10),
    ("target_len_6", 6),
]


@pytest.fixture(scope="module")
def specification():
    """The repository in the configuration ``damg_example.py`` searches."""
    return DAMGrepository(
        linear_feature_dimensions=[1, 2, 3, 4, 5],
        constant_values=[0, 1, -1],
        learning_rate_values=[1e-2],
        n_epoch_values=[2000],
    ).specification()


def _terms(specification, target, count):
    space = Synthesizer(specification, {}).construct_solution_space(target).prune()
    return list(itertools.islice(space.enumerate_trees(target), count))


def _counting_algebra():
    """Return the algebra with every leaf continuation counted, and the counter it writes to."""
    algebra = edgelist_algebra(False)
    counts: Counter = Counter()

    def counting(name, clause):
        def wrapped(*arguments):
            continuation = clause(*arguments)

            def counted(id, inputs):
                counts[name] += 1
                return continuation(id, inputs)

            return counted

        return wrapped

    for name in LEAF_CLAUSES:
        algebra[name] = counting(name, algebra[name])
    return algebra, counts


def _leaves(term: Tree) -> int:
    return int(term.root in LEAF_CLAUSES) + sum(_leaves(child) for child in term.children)


@pytest.mark.parametrize(("name", "leaves"), TARGETS_AND_LEAVES)
def test_a_leaf_continuation_is_evaluated_once_per_leaf(specification, name, leaves):
    """One evaluation per leaf, so the interpretation stays linear in the size of the term.

    The term as it was synthesized is counted, and so are the three folds the kernels read: each
    granularity has clauses of its own, and a fold may drop a leaf, so each is counted against its
    own leaves.
    """
    for term in _terms(specification, getattr(damg_targets, name), SAMPLE):
        assert _leaves(term) == leaves
        folds = [term, *(term.interpret(hierarchy_algebra(level)) for level in (1, 2, 3))]
        for read in folds:
            algebra, counts = _counting_algebra()
            read.interpret(algebra)
            assert sum(counts.values()) == _leaves(read)


def _node(x, y):
    return f"node{(x, y)}"


def test_a_chain_is_composed_from_the_back(specification):
    """Sequential composition puts the edges of the second link in front of those of the first."""
    term = _terms(specification, damg_targets.target_len_2, 1)[0]
    edges, positions = term.interpret(hierarchy_algebra(2)).interpret(edgelist_algebra(True))
    first, second = _node(-5.5, -3.8), _node(-3.0, -3.8)
    assert edges == [
        (first, second),
        ("input", first),
        (second, "loss"),
        ("loss", "optimizer"),
        ("optimizer", "epochs(2000)"),
    ]
    assert positions[first] == (-5.5, -3.8)
    assert positions[second] == (-3.0, -3.8)


def test_a_parallel_pair_is_composed_from_the_front(specification):
    """Parallel composition keeps the left branch first and moves the right one down the page."""
    target = Constructor("Learner", Constructor("DAG",
                                                Constructor("input", Literal(2))
                                                & Constructor("output", Literal(2))
                                                & Constructor("structure",
                                                              Literal(((None, None),))))
                         & Constructor("Loss", Constructor("type", Literal(None)))
                         & Constructor("Optimizer", Constructor("type", Literal(None)))
                         & Constructor("epochs", Literal(2000)))
    term = _terms(specification, target, 1)[0]
    edges, positions = term.interpret(hierarchy_algebra(2)).interpret(edgelist_algebra(True))
    left, right = _node(-5.5, -3.8), _node(-5.5, -3.8 + 0.2)
    assert edges == [
        ("input", left),
        ("input", right),
        (left, "loss"),
        (right, "loss"),
        ("loss", "optimizer"),
        ("optimizer", "epochs(2000)"),
    ]
    assert positions[left] == (-5.5, -3.8)
    assert positions[right] == (-5.5, -3.8 + 0.2)


def _one_node_branch():
    """The granularity 3 spelling of a single node standing alone in a parallel composition."""
    return Tree("beside_singleton_h3", (Tree(1), Tree(1), Tree("node", (Tree(1), Tree(1)))))


def test_the_coarsest_parallel_clause_evaluates_each_branch_once():
    """``beside_cons_h3`` is written by hand here, because no shipped target reaches it.

    The granularity 3 fold merges two adjacent parallel nodes whenever they carry the same label,
    and at that granularity every node does, so the terms of the twelve targets never carry this
    clause.  It is still a clause of the algebra, and it composes the same way.
    """
    term = Tree("beside_cons_h3",
                (Tree(2), Tree(1), Tree(2), _one_node_branch(), _one_node_branch()))
    algebra, counts = _counting_algebra()
    edges, outputs, positions = term.interpret(algebra)((-5.5, -3.8), ["a", "b"])
    assert sum(counts.values()) == 2
    left, right = _node(-5.5, -3.8), _node(-5.5, -3.8 + 0.2)
    assert edges == [("a", left), ("b", right)]
    assert outputs == [left, right]
    assert positions == {left: (-5.5, -3.8), right: (-5.5, -3.8 + 0.2)}
