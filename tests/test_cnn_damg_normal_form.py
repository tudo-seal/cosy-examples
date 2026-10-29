"""A requested structure is rewritten where a law's pattern stands, not at its front.

A target names the structure it asks for as a tuple of layers, and the synthesizer sees that
structure only in normal form: ``CNNrepository.ParaTupleTuples.normalize`` rewrites it first.  An
edge before or after a layer is dropped, a crossing followed by its inverse becomes edges, and the
two pair laws and the triple law of the swaps each replace their left-hand side by their right-hand
side.  A rewrite puts its result where its pattern stood.

Each structure below holds one pattern behind a layer, so that the pattern does not stand at the
front.  The first holds an edge in second place, where a rewrite at the front is right as well; it
is the control for the expected values.  The last holds an edge in front of the triple, so that one
round of the rewriting finds two patterns.
"""

from __future__ import annotations

import pytest

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository


def _layer(label, i, o):
    """A layer of one component."""
    return ((label, i, o),)


def _edges(n):
    """`n` edges carried straight through."""
    return (("swap", 0, n), n, n)


def _swap(m, n):
    """The wiring that crosses `m` edges past `n` edges."""
    return (("swap", m, n), m + n, m + n)


def _linear(i, o, bias=True):
    return CNNrepository.Linear(in_features=i, out_features=o, bias=bias)


_RELU = CNNrepository.ReLu()
_SIGMOID = CNNrepository.Sigmoid()
_A = _layer(_linear(2, 2), 2, 2)
_B = _layer(_RELU, 2, 2)
_C = _layer(_linear(2, 2, bias=False), 2, 2)
_D = _layer(_SIGMOID, 2, 2)
#: The two components the triple law runs past each other.
_X = (_linear(2, 3), 2, 3)
_Y = (_linear(1, 4), 1, 4)

#: Per case, the requested structure and its normal form.
CASES = {
    "an_edge_in_second_place": (
        (_A, (_edges(2),), _B),
        (_A, _B),
    ),
    "an_edge_in_fourth_place": (
        (_A, _B, _C, (_edges(2),), _D),
        (_A, _B, _C, _D),
    ),
    "a_crossing_and_its_inverse_behind_a_layer": (
        (_layer(_linear(4, 4), 4, 4), (_swap(1, 3),), (_swap(3, 1),), _layer(_RELU, 4, 4)),
        (_layer(_linear(4, 4), 4, 4), _layer(_RELU, 4, 4)),
    ),
    "the_first_pair_law_behind_a_layer": (
        (_layer(_linear(6, 6), 6, 6), (_swap(1, 2), _edges(3)), (_edges(2), _swap(1, 3)),
         _layer(_RELU, 6, 6)),
        (_layer(_linear(6, 6), 6, 6), (_swap(1, 5),), _layer(_RELU, 6, 6)),
    ),
    "the_second_pair_law_behind_a_layer": (
        (_layer(_linear(6, 6), 6, 6), (_edges(1), _swap(2, 3)), (_swap(1, 3), _edges(2)),
         _layer(_RELU, 6, 6)),
        (_layer(_linear(6, 6), 6, 6), (_swap(3, 3),), _layer(_RELU, 6, 6)),
    ),
    "the_triple_law_behind_a_layer": (
        (_layer(_linear(3, 3), 3, 3), (_swap(1, 2),), (_X, _Y), (_swap(3, 4),), _layer(_RELU, 7, 7)),
        (_layer(_linear(3, 3), 3, 3), (_Y, _X), _layer(_RELU, 7, 7)),
    ),
    "an_edge_in_front_of_the_triple_law": (
        (_layer(_linear(3, 3), 3, 3), (_edges(3),), (_swap(1, 2),), (_X, _Y), (_swap(3, 4),),
         _layer(_RELU, 7, 7)),
        (_layer(_linear(3, 3), 3, 3), (_Y, _X), _layer(_RELU, 7, 7)),
    ),
}


@pytest.fixture(scope="module")
def normalizer():
    """The group a target's structure literal belongs to, over the labels the cases use."""
    labels = [_linear(2, 2), _linear(2, 2, bias=False), _linear(4, 4), _linear(6, 6),
              _linear(3, 3), _linear(2, 3), _linear(1, 4), _RELU, _SIGMOID]
    para = CNNrepository.Para(labels, range(1, 8))
    return CNNrepository.ParaTupleTuples(CNNrepository.ParaTuples(para, max_length=2))


@pytest.mark.parametrize("requested, normal_form", CASES.values(), ids=list(CASES))
def test_a_law_rewrites_the_structure_where_its_pattern_stands(normalizer, requested, normal_form):
    """The structure normalizes to the expected one, and the case is one a law applies to.

    Args:
        normalizer (Group): The structures' group, as the repository builds it.
        requested (tuple): The structure a target asks for.
        normal_form (tuple): The structure the laws turn it into.
    """
    assert not normalizer.normalform(requested), "no law applies to the requested structure"
    assert normalizer.normalform(normal_form), "the expected structure is not a normal form"
    assert normalizer.normalize(requested) == normal_form
