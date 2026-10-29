"""A requested structure is rewritten where a law's pattern stands, not at its front.

A target names the structure it asks for as a tuple of layers, and the synthesizer sees that
structure only in normal form: ``ParaTupleTuples.normalize`` rewrites it first.  An edge before or
after a layer is dropped, a crossing followed by its inverse becomes edges, and the two pair laws
and the triple law of the swaps each replace their left-hand side by their right-hand side.  A
rewrite puts its result where its pattern stood.

Each structure below holds one pattern behind a layer, so that the pattern does not stand at the
front.  The first holds an edge in second place, where a rewrite at the front is right as well; it
is the control for the expected values.  The last holds an edge in front of the triple law, so that
one round of the rewriting finds two patterns.

``CNNrepository`` and ``DAMGrepository`` each carry their own copy of the rewriting, so every case
runs against both, over the labels of the repository it runs against.
"""

from __future__ import annotations

import pytest

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository
from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository

REPOSITORIES = {"cnn": CNNrepository, "damg": DAMGrepository}


def _layer(label, i, o):
    """A layer of one component."""
    return ((label, i, o),)


def _edges(n):
    """`n` edges carried straight through."""
    return (("swap", 0, n), n, n)


def _swap(m, n):
    """The wiring that crosses `m` edges past `n` edges."""
    return (("swap", m, n), m + n, m + n)


def _labels(repository):
    """The component labels the cases use, as the repository's own classes.

    Args:
        repository (type): ``CNNrepository`` or ``DAMGrepository``.

    Returns:
        dict: The labels by name.
    """
    def linear(i, o, bias=True):
        return repository.Linear(in_features=i, out_features=o, bias=bias)

    return {
        "a": linear(2, 2),
        "c": linear(2, 2, bias=False),
        "linear_3": linear(3, 3),
        "linear_4": linear(4, 4),
        "linear_6": linear(6, 6),
        "x": linear(2, 3),
        "y": linear(1, 4),
        "relu": repository.ReLu(),
        "sigmoid": repository.Sigmoid(),
    }


def _cases(repository):
    """Per case, the requested structure and its normal form.

    Args:
        repository (type): The repository whose labels the layers carry.

    Returns:
        dict: The requested structure and its normal form, by the name of the case.
    """
    label = _labels(repository)
    a = _layer(label["a"], 2, 2)
    b = _layer(label["relu"], 2, 2)
    c = _layer(label["c"], 2, 2)
    d = _layer(label["sigmoid"], 2, 2)
    # The two components the triple law runs past each other.
    x = (label["x"], 2, 3)
    y = (label["y"], 1, 4)
    return {
        "an_edge_in_second_place": (
            (a, (_edges(2),), b),
            (a, b),
        ),
        "an_edge_in_fourth_place": (
            (a, b, c, (_edges(2),), d),
            (a, b, c, d),
        ),
        "a_crossing_and_its_inverse_behind_a_layer": (
            (_layer(label["linear_4"], 4, 4), (_swap(1, 3),), (_swap(3, 1),),
             _layer(label["relu"], 4, 4)),
            (_layer(label["linear_4"], 4, 4), _layer(label["relu"], 4, 4)),
        ),
        "the_first_pair_law_behind_a_layer": (
            (_layer(label["linear_6"], 6, 6), (_swap(1, 2), _edges(3)), (_edges(2), _swap(1, 3)),
             _layer(label["relu"], 6, 6)),
            (_layer(label["linear_6"], 6, 6), (_swap(1, 5),), _layer(label["relu"], 6, 6)),
        ),
        "the_second_pair_law_behind_a_layer": (
            (_layer(label["linear_6"], 6, 6), (_edges(1), _swap(2, 3)), (_swap(1, 3), _edges(2)),
             _layer(label["relu"], 6, 6)),
            (_layer(label["linear_6"], 6, 6), (_swap(3, 3),), _layer(label["relu"], 6, 6)),
        ),
        "the_triple_law_behind_a_layer": (
            (_layer(label["linear_3"], 3, 3), (_swap(1, 2),), (x, y), (_swap(3, 4),),
             _layer(label["relu"], 7, 7)),
            (_layer(label["linear_3"], 3, 3), (y, x), _layer(label["relu"], 7, 7)),
        ),
        "an_edge_in_front_of_the_triple_law": (
            (_layer(label["linear_3"], 3, 3), (_edges(3),), (_swap(1, 2),), (x, y), (_swap(3, 4),),
             _layer(label["relu"], 7, 7)),
            (_layer(label["linear_3"], 3, 3), (y, x), _layer(label["relu"], 7, 7)),
        ),
    }


#: The names of the cases, the same for both repositories.
CASE_NAMES = list(_cases(CNNrepository))


@pytest.fixture(scope="module", params=sorted(REPOSITORIES))
def repository(request):
    """One of the two repositories, by its key in ``REPOSITORIES``."""
    return REPOSITORIES[request.param]


@pytest.fixture(scope="module")
def normalizer(repository):
    """The group a target's structure literal belongs to, over the labels the cases use."""
    para = repository.Para(list(_labels(repository).values()), range(1, 8))
    return repository.ParaTupleTuples(repository.ParaTuples(para, max_length=2))


@pytest.mark.parametrize("case", CASE_NAMES)
def test_a_law_rewrites_the_structure_where_its_pattern_stands(repository, normalizer, case):
    """The structure normalizes to the expected one, and the case is one a law applies to.

    Args:
        repository (type): The repository the structure is requested from.
        normalizer (Group): The structures' group, as that repository builds it.
        case (str): The name of the case.
    """
    requested, normal_form = _cases(repository)[case]
    assert not normalizer.normalform(requested), "no law applies to the requested structure"
    assert normalizer.normalform(normal_form), "the expected structure is not a normal form"
    assert normalizer.normalize(requested) == normal_form
