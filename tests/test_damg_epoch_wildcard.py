"""A DAMG target may leave the epoch count open, and gets every count the repository offers.

Every slot of the ``learner`` suffix carries its value and the wildcard ``None``, which is what
lets a target pin the slot or leave it open.  The epoch count carried only its value, so a target
that did not name a number was uninhabited and answered with no term at all.

These tests enumerate the targets in full rather than drawing a sample, because the claim is a set
equality: the open target is the union of the pinned ones, and nothing else.
"""

from __future__ import annotations

import pytest
from cosy.core import Synthesizer
from cosy.core.types import Constructor, Literal

from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository

#: The epoch counts the repository under test enumerates.
EPOCH_VALUES = [10, 100, 2000]

#: How many terms one pinned epoch count has.  Measured on this repository and this structure.
TERMS_PER_COUNT = 736


def _target(epochs):
    return Constructor("Learner", Constructor("DAG",
                                              Constructor("input", Literal(1))
                                              & Constructor("output", Literal(1))
                                              & Constructor("structure", Literal((None, None))))
                       & Constructor("Loss", Constructor("type", Literal(None)))
                       & Constructor("Optimizer", Constructor("type", Literal(None)))
                       & Constructor("epochs", Literal(epochs)))


@pytest.fixture(scope="module")
def specification():
    repository = DAMGrepository(
        linear_feature_dimensions=[1, 2, 3, 4, 5],
        constant_values=[0, 1, -1],
        learning_rate_values=[1e-2],
        n_epoch_values=EPOCH_VALUES,
    )
    return repository.specification()


#: Position of the epoch count among the children of a ``learner`` node.  The parameters come in
#: the order the specification declares them: i, o, request, ls, epochs.
EPOCH_CHILD = 4


def _trees(specification, epochs):
    target = _target(epochs)
    space = Synthesizer(specification, {}).construct_solution_space(target).prune()
    return list(space.enumerate_trees(target))


def _terms(specification, epochs) -> set[str]:
    return {str(term) for term in _trees(specification, epochs)}


@pytest.mark.parametrize("epochs", EPOCH_VALUES)
def test_a_pinned_epoch_count_is_unaffected(specification, epochs):
    """The wildcard is added beside the value, so pinning a count answers as it always did."""
    assert len(_terms(specification, epochs)) == TERMS_PER_COUNT


def test_an_open_epoch_count_is_the_union_of_the_pinned_ones(specification):
    """Leaving the count open asks for all of them, and gets each of them exactly once."""
    open_terms = _terms(specification, None)
    pinned = [_terms(specification, epochs) for epochs in EPOCH_VALUES]
    assert open_terms == set().union(*pinned)
    assert len(open_terms) == len(EPOCH_VALUES) * TERMS_PER_COUNT


def test_an_open_epoch_count_yields_terms_at_all(specification):
    """The defect this pins is the empty answer, so the count matters less than that it is not 0."""
    assert len(_terms(specification, None)) > 0


def test_no_term_leaves_the_epoch_count_open(specification):
    """The wildcard opens the request, not the term.

    ``None`` may not reach ``n_epoch_values`` instead, because ``learner`` counts its epochs with
    ``range()`` and a term saying ``epochs=None`` would have no interpretation.
    """
    counts = {tree.children[EPOCH_CHILD].root for tree in _trees(specification, None)}
    assert counts == set(EPOCH_VALUES)
