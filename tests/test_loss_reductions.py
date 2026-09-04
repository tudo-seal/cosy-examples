"""Both repositories accept exactly the loss reductions they offer.

A target names its loss instead of drawing it out of the loss group, so the group's ``__contains__``
is what decides whether a named loss is answerable at all.  A reduction that is accepted there
without being yielded by ``__iter__`` gets no refusal: the target builds a solution space that holds
rules and derives no term, which reads like "nothing in the repository fits" rather than like "there
is no such loss".

The two claims below tell those two answers apart.  The first is the invariant, that the accepted
reductions are the offered ones.  The second is what it buys: a target on a reduction that is not
offered is now answered the way a target on a value that is no loss at all is answered, with a space
of no non-terminals.
"""

from __future__ import annotations

import pytest
from cosy.core import Synthesizer
from cosy.core.types import Constructor, Literal

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository
from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository

#: The reductions the loss groups build their values with.
OFFERED = {"mean", "sum"}

#: The reduction torch has beside those two.  No loss value of either repository carries it, and
#: the training loop could not read one that did, because it makes the loss a vector.
UNOFFERED = "none"

REPOSITORIES = {"cnn": CNNrepository, "damg": DAMGrepository}


@pytest.mark.parametrize("repository", sorted(REPOSITORIES))
def test_the_loss_group_offers_two_reductions(repository):
    """What the group yields is the measure the acceptance test is held against."""
    group = REPOSITORIES[repository].LossFunction()
    assert {value.reduction for value in group} == OFFERED


@pytest.mark.parametrize("repository", sorted(REPOSITORIES))
def test_every_offered_loss_is_accepted(repository):
    """A target may name any loss the group builds, which is the other half of the invariant."""
    group = REPOSITORIES[repository].LossFunction()
    assert [value for value in group if value not in group] == []


@pytest.mark.parametrize("repository", sorted(REPOSITORIES))
def test_no_loss_class_accepts_the_reduction_none_of_them_offers(repository):
    """Each loss class of the repository, asked with the reduction the group never builds."""
    group = REPOSITORIES[repository].LossFunction()
    classes = sorted({type(value) for value in group}, key=lambda cls: cls.__name__)
    assert classes, "the group yields nothing, so there is no class to ask"
    assert [cls.__name__ for cls in classes if cls(reduction=UNOFFERED) in group] == []


@pytest.fixture(scope="module")
def specification():
    """The smallest DAMG repository that still answers a two-position target."""
    return DAMGrepository(
        linear_feature_dimensions=[1, 2],
        constant_values=[0, 1],
        learning_rate_values=[1e-2],
        n_epoch_values=[10],
    ).specification()


def _target(loss):
    return Constructor("Learner", Constructor("DAG",
                                              Constructor("input", Literal(1))
                                              & Constructor("output", Literal(1))
                                              & Constructor("structure", Literal((None, None))))
                       & Constructor("Loss", Constructor("type", Literal(loss)))
                       & Constructor("Optimizer", Constructor("type", Literal(None)))
                       & Constructor("epochs", Literal(10)))


def _space(specification, loss):
    target = _target(loss)
    return Synthesizer(specification, {}).construct_solution_space(target).prune(), target


def test_an_offered_reduction_is_answered_with_terms(specification):
    """The contrast the two tests below are read against."""
    space, target = _space(specification, DAMGrepository.MSEloss(reduction="mean"))
    assert list(space.nonterminals())
    assert next(iter(space.enumerate_trees(target)), None) is not None


@pytest.mark.parametrize("loss", [DAMGrepository.MSEloss(reduction=UNOFFERED), "not a loss at all"],
                         ids=["a reduction that is not offered", "a value that is no loss"])
def test_a_loss_the_group_rejects_leaves_the_space_empty(specification, loss):
    """A rejected loss ends in an empty space, whatever makes it rejected.

    The empty space is the repository's own way of saying that the target names something it does
    not have.  Before the reduction was rejected, the same target built a space with rules in it
    and no term to derive from them, which is the answer an inhabited repository gives to a
    question it cannot satisfy.
    """
    space, target = _space(specification, loss)
    assert list(space.nonterminals()) == []
    assert next(iter(space.enumerate_trees(target)), None) is None
