"""The four swap laws are one statement, and the two repositories may share it.

``damg_nas/damg_repo.py`` and ``cnn_damg_nas/cnn_damg_repo.py`` build terms over different
alphabets and forbid the same four rewrite patterns.  Both inherit the predicates that do it from
``bayesian_optimization/examples/swap_laws.py``.  Sharing is worth having only while it is real:
two names bound to two equal copies read exactly like one name bound twice, and a repair reaches
only one of them.  The first test holds that the four laws are one object each, reached from both
repository classes.

The second holds the premise the sharing rests on.  The laws read a derivation tree by root name
and child position, and they check the number of children before they index it, so a law is right
about an alphabet only while the six combinators it reads carry the arities it assumes.  Those
arities are a property of two clause tables that nothing else ties together.  A parameter added to
``beside_singleton`` in one repository and not in the other would move every position the laws read
there, and the laws would raise rather than answer, but only for the terms that reach them.  Here
it is a failing test instead.
"""

from __future__ import annotations

import pytest
from cosy.core import Synthesizer

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_reference_architectures import (
    cifar10_vgg11_bn_repo,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_targets import (
    VGG11_BN_MERGED_POSITIONS,
    make_vgg11_bn_target,
)
from bayesian_optimization.examples.damg_nas import damg_targets
from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository
from bayesian_optimization.examples.swap_laws import SwapLaws

#: The names of the four laws, in the order the ``before_cons`` clause attaches them.
LAW_NAMES = ("swaplaw1", "swaplaw2", "swaplaw3", "swaplaw4")

#: The number of children each law requires of the six combinators it reads, as its calls to
#: ``_require_children_count`` state them.  A node with another arity raises there.
LAW_ARITIES = {
    "beside_singleton": 5,
    "before_singleton": 6,
    "before_cons": 9,
    "beside_cons": 11,
    "edges": 17,
    "swap": 19,
}

#: The merged VGG chain with one repeated position opened to two parallel components.  A chain of
#: single components reaches none of the wiring the laws are about, and this is the cheapest
#: opening that brings ``swap``, ``beside_cons`` and ``edges`` in together.
PARALLEL_POSITIONS = tuple(
    2 if index == 12 else pair for index, pair in enumerate(VGG11_BN_MERGED_POSITIONS)
)

#: A DAMG configuration with one feature dimension, which is the smallest space that still reaches
#: all six combinators.  Both 0 and 1 have to be among the constants, because ``DAMGrepository``
#: appends both of them as soon as either is missing and the duplicate makes the space ambiguous.
DAMG_PARAMETER_SETS = {
    "linear_feature_dimensions": [2],
    "constant_values": [0, 1],
    "learning_rate_values": [1e-2],
    "n_epoch_values": [2000],
}


def _arities(space):
    """Return the argument counts each terminal is used with in a synthesized program.

    Args:
        space (SolutionSpace): The pruned program.

    Returns:
        dict: Per terminal name, the sorted argument counts its rules carry.
    """
    found = {}
    for nonterminal in space.nonterminals():
        for rule in space.get(nonterminal) or ():
            found.setdefault(str(rule.terminal), set()).add(len(rule.arguments))
    return {terminal: sorted(counts) for terminal, counts in found.items()}


@pytest.fixture(scope="module")
def spaces():
    """Build one space per repository, both small and both reaching all six combinators.

    Returns:
        dict: Per repository name, the argument counts :func:`_arities` reads off its program.
    """
    damg = DAMGrepository(**DAMG_PARAMETER_SETS)
    cnn = cifar10_vgg11_bn_repo()
    return {
        "DAMGrepository": _arities(
            Synthesizer(damg.specification(), {})
            .construct_solution_space(damg_targets.target_len_3).prune()),
        "CNNrepository": _arities(
            Synthesizer(cnn.specification(), {})
            .construct_solution_space(make_vgg11_bn_target(PARALLEL_POSITIONS)).prune()),
    }


@pytest.mark.parametrize("law", LAW_NAMES)
def test_each_law_is_one_object_that_both_repositories_reach(law):
    """The name resolves to the same function through either repository class.

    This fails the moment one of the two files states a law of its own again, whether or not the
    two statements agree that day.

    Args:
        law (str): The name of the law.
    """
    shared = getattr(SwapLaws, law)
    assert getattr(DAMGrepository, law) is shared
    assert getattr(CNNrepository, law) is shared


def test_the_laws_arrive_through_the_shared_class():
    """Both repositories inherit ``SwapLaws``, rather than holding copies that happen to match."""
    assert SwapLaws in DAMGrepository.__mro__
    assert SwapLaws in CNNrepository.__mro__


@pytest.mark.parametrize("repository", ("DAMGrepository", "CNNrepository"))
def test_the_combinators_the_laws_read_have_the_arities_they_assume(spaces, repository):
    """The six combinators carry exactly the child counts the laws check for.

    Args:
        spaces (dict): The argument counts of both programs.
        repository (str): Which of the two to hold against the laws.
    """
    found = spaces[repository]
    assert {terminal: found.get(terminal) for terminal in LAW_ARITIES} == {
        terminal: [arity] for terminal, arity in LAW_ARITIES.items()
    }
