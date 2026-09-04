"""What a node of a DAMG carries as its label, and what it must not carry.

``as_DAMG`` turns a term into the multigraph of the network it denotes.  The edge list names a
node by its symbol plus its layout coordinate, because two instances of one layer have to be two
nodes, and the kernel needs the opposite: two instances of one layer sharing one label.  The
coordinate is therefore stripped again for the ``symbol`` attribute, and these tests pin what is
left standing after the strip, at each of the three granularities the kernels use.

The last test covers the one term shape that has no positioned node at all, a chain whose single
link is a swap, because the strip asks the edge-list algebra for positions unconditionally and
that shape is where the position map comes back empty.
"""

from __future__ import annotations

import itertools

import pytest
from cosy.core import Synthesizer
from cosy.core.types import Constructor, Literal

from bayesian_optimization.examples.damg_nas import damg_targets
from bayesian_optimization.examples.damg_nas.damg_kernels import as_DAMG, hierarchical_tree
from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository

#: How many terms of one target the label tests read.
SAMPLE = 8

#: The nodes every learner has beside its model, as they are named from granularity 2 on.  At
#: granularity 1 the loss and the optimizer are named by their full parameter lists instead.
FRAME = {"input", "loss", "optimizer", "epochs(2000)"}


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


def _terms(specification, target, count):
    space = Synthesizer(specification, {}).construct_solution_space(target).prune()
    return list(itertools.islice(space.enumerate_trees(target), count))


def _graph_of(term, level):
    """Return the position map and the relabeling ``as_DAMG`` builds at one granularity."""
    _, graph, positions, relabel = as_DAMG(hierarchical_tree(level)(term), verbose=True)
    return graph, positions, relabel


@pytest.fixture(scope="module")
def sample(specification):
    return _terms(specification, damg_targets.target_len_3, SAMPLE)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_every_node_of_the_graph_has_a_position(sample, level):
    """The strip is the suffix the algebra names, so every node has to be named there."""
    for term in sample:
        graph, positions, _ = _graph_of(term, level)
        assert [node for node in graph.nodes() if node not in positions] == []


@pytest.mark.parametrize("level", [2, 3])
def test_a_folded_node_is_labeled_by_its_kind_and_not_by_its_position(sample, level):
    """From granularity 2 every model node is called ``node``, which is all that is left of it."""
    for term in sample:
        graph, _, relabel = _graph_of(term, level)
        assert set(relabel.values()) == FRAME | {"node"}
        model_nodes = [node for node in graph.nodes() if node not in FRAME]
        assert model_nodes != []
        assert {relabel[node] for node in model_nodes} == {"node"}


def test_granularity_1_keeps_the_layer_with_its_dimensions(sample):
    """At granularity 1 the label is the layer triple, and it too carries no coordinate."""
    for term in sample:
        graph, positions, relabel = _graph_of(term, 1)
        for node in graph.nodes():
            assert relabel[node] == node.removesuffix(str(positions[node]))
            assert not relabel[node].endswith(str(positions[node]))
        assert any("DAMGrepository.Linear" in label for label in relabel.values())


def test_the_frame_nodes_keep_their_name(sample):
    """``input``, the loss, the optimizer and the epoch marker are in the map but carry no
    coordinate, so the strip has to leave them alone."""
    for term in sample:
        _, _, relabel = _graph_of(term, 2)
        for name in FRAME:
            assert relabel[name] == name


def _swap_only_target():
    """A target whose model is one link wide, so that a bare swap satisfies it."""
    return Constructor("Learner", Constructor("DAG",
                                              Constructor("input", Literal(2))
                                              & Constructor("output", Literal(2))
                                              & Constructor("structure", Literal(((None,),))))
                       & Constructor("Loss", Constructor("type", Literal(None)))
                       & Constructor("Optimizer", Constructor("type", Literal(None)))
                       & Constructor("epochs", Literal(2000)))


def test_a_model_without_a_positioned_node_still_converts(specification):
    """A chain whose only link is a swap places no node, and the conversion has to survive that."""
    target = _swap_only_target()
    terms = _terms(specification, target, 24)
    assert len(terms) == 24
    frames_only = 0
    for term in terms:
        for level in (1, 2, 3):
            _graph_of(term, level)
        graph, _, _ = _graph_of(term, 2)
        if set(graph.nodes()) == FRAME:
            frames_only += 1
    assert frames_only == 4
