"""A DAMG target may name a structure triple that leaves slots open, and still get terms.

A target names the structure it asks for as a literal whose elements are triples of layer, input
dimension and output dimension.  Seven of the nine leaf combinators of ``DAMGrepository`` declare
seven such triples per component, the fully concrete one and six that leave one or two of the three
slots open, and ``edges`` and ``swap`` declare sixteen.  All but the fully concrete one are never
enumerated, they exist only as membership questions, so the group that answers those questions is
the only thing standing between such a target and its terms.

These tests ask the groups directly and then count what the synthesizer finds, one target per
variant.  The counts are capped, because the point is the difference between zero and not zero.
"""

from __future__ import annotations

import itertools

import pytest
from cosy.core import Synthesizer
from cosy.core.types import Constructor, Literal

from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository
from bayesian_optimization.examples.damg_nas.damg_repo_algebras import pretty_term_algebra

#: How many terms are drawn per target.  Every one of the seven has at least this many.
CAP = 4


@pytest.fixture(scope="module")
def repository() -> DAMGrepository:
    """The repository in the configuration ``damg_example.py`` searches."""
    return DAMGrepository(
        linear_feature_dimensions=[1, 2, 3, 4, 5],
        constant_values=[0, 1, -1],
        learning_rate_values=[1e-2],
        n_epoch_values=[2000],
    )


@pytest.fixture(scope="module")
def specification(repository: DAMGrepository):
    return repository.specification()


@pytest.fixture(scope="module")
def groups(repository: DAMGrepository):
    """Return the three nested groups the way ``specification`` builds them."""
    labels = repository.Label(
        repository.dimensions,
        repository.linear_feature_dimensions,
        repository.constant_values,
    )
    para = repository.Para(labels, repository.dimensions)
    para_tuples = repository.ParaTuples(para, max_length=max(repository.dimensions))
    return labels, para, para_tuples, repository.ParaTupleTuples(para_tuples)


@pytest.fixture(scope="module")
def variants(groups):
    """The seven structure triples the ``linear_layer`` combinator declares for one label."""
    labels = groups[0]
    layer = next(iter(labels.iter_linear()))
    i, o = layer.in_features, layer.out_features
    return {
        "para1": (layer, i, o),
        "para2": (layer, i, None),
        "para3": (layer, None, o),
        "para4": (None, i, o),
        "para5": (layer, None, None),
        "para6": (None, None, o),
        "para7": (None, i, None),
    }


def _target(structure):
    return Constructor("Learner", Constructor("DAG",
                                              Constructor("input", Literal(1))
                                              & Constructor("output", Literal(1))
                                              & Constructor("structure", Literal(structure)))
                       & Constructor("Loss", Constructor("type", Literal(None)))
                       & Constructor("Optimizer", Constructor("type", Literal(None)))
                       & Constructor("epochs", Literal(2000)))


def test_the_enumeration_holds_no_open_triple(groups, variants):
    """Only ``__contains__`` can admit an open triple, since ``__iter__`` never produces one."""
    _, _, para_tuples, _ = groups
    open_triples = [
        entry
        for entry in para_tuples.para
        if isinstance(entry, tuple) and any(slot is None for slot in entry)
    ]
    assert open_triples == []
    assert variants["para1"] in set(para_tuples.para)


@pytest.mark.parametrize("variant", ["para1", "para2", "para3", "para4", "para5", "para6", "para7"])
def test_every_declared_triple_is_a_member_of_every_group(groups, variants, variant):
    """What ``Para`` accepts, the two groups built on top of it accept as well."""
    _, para, para_tuples, para_tuple_tuples = groups
    triple = variants[variant]
    assert triple in para
    assert (triple,) in para_tuples
    assert ((triple,),) in para_tuple_tuples


@pytest.mark.parametrize("variant", ["para1", "para2", "para3", "para4", "para5", "para6", "para7"])
def test_a_target_naming_one_triple_has_terms(specification, variants, variant):
    """A target whose structure names one triple gets terms, open slots included."""
    target = _target(((variants[variant],),))
    space = Synthesizer(specification, {}).construct_solution_space(target).prune()
    terms = list(itertools.islice(space.enumerate_trees(target), CAP))
    assert len(terms) == CAP
    for term in terms:
        assert isinstance(term.interpret(pretty_term_algebra()), str)
