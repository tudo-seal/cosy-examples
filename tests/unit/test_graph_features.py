"""The Weisfeiler-Lehman kernel's label counts as explicit features.

``WeisfeilerLehmanFeatures`` computes the vectors whose dot products the WL kernel sums, so its
normalized Gram is held to the kernel's own, over the same translation, at every round count.  The
vocabulary is the training terms': a term read later is mapped onto it, a label it alone has
dropped, which is what restricting a fit over both to the training columns gives.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from cosy.core.tree import Tree
from grakel import graph_from_networkx
from grakel.kernels import VertexHistogram
from scipy.sparse import issparse
from sklearn.exceptions import NotFittedError

from bayesian_optimization.kernels.graph_features import (
    ConcatenatedFeatures,
    WeisfeilerLehmanFeatures,
)
from bayesian_optimization.kernels.graph_kernel import WeisfeilerLehmanKernel
from bayesian_optimization.utils import to_grakel_graph, to_indexed_nx_digraph


def _lit() -> Tree:
    return Tree("lit", ())


def _neg(inner: Tree) -> Tree:
    return Tree("neg", (inner,))


def _add(left: Tree, right: Tree) -> Tree:
    return Tree("add", (left, right))


def _terms() -> list[Tree]:
    """Distinct expression terms of different shapes, depths and argument orders."""
    a = _lit()
    return [
        a,
        _neg(a),
        _neg(_neg(a)),
        _add(a, a),
        _add(_neg(a), a),
        _add(a, _neg(a)),
        _add(_add(a, a), a),
        _add(a, _add(a, a)),
        _neg(_add(a, _neg(a))),
        _add(_neg(_neg(a)), _add(a, _neg(a))),
        _add(_add(_neg(a), a), _add(a, a)),
        _neg(_neg(_add(a, a))),
    ]


class _Presence(VertexHistogram):
    """The vertex histogram over whether a label occurs: the kernel the presence features belong to.

    Inside the WL rounds grakel builds the base kernel with its default ``sparse="auto"``, so the
    counts may arrive as a sparse matrix.
    """

    def parse_input(self, X):  # noqa: N803 - grakel's name
        counts = super().parse_input(X)
        if issparse(counts):
            counts = counts.copy()
            counts.data = (counts.data > 0).astype(float)
            return counts
        return (np.asarray(counts) > 0).astype(float)


def _bare_graph(term: Tree):
    """The term's graph with the bare symbols as labels: ``add`` over two ``lit`` leaves has two
    neighbours of one label, which tells a multiset of neighbours from a set of them."""
    graph, symbols = to_indexed_nx_digraph(term)
    for node, symbol in symbols.items():
        graph.nodes[node]["symbol"] = str(symbol)
    return graph_from_networkx([graph.to_undirected()], node_labels_tag="symbol")


def _normalized_gram(features: np.ndarray) -> np.ndarray:
    gram = features @ features.T
    return np.asarray(gram / np.sqrt(np.outer(np.diag(gram), np.diag(gram))), dtype=float)


@pytest.mark.parametrize("h", [0, 1, 2])
@pytest.mark.parametrize("presence", [False, True])
@pytest.mark.parametrize("translation", [to_grakel_graph, _bare_graph], ids=["indexed", "bare"])
def test_the_features_give_the_kernel_s_gram(h, presence, translation):
    terms = _terms()
    kernel = WeisfeilerLehmanKernel(
        h=h,
        base_graph_kernel=_Presence if presence else VertexHistogram,
        normalize=True,
        to_grakel_graph=translation,
    )
    expected = np.asarray(kernel(np.array(terms, dtype=object)), dtype=float)
    features = (
        WeisfeilerLehmanFeatures(h=h, to_grakel_graph=translation, presence=presence)
        .fit(terms)
        .transform(terms)
    )

    assert features.shape[0] == len(terms) and features.shape[1] > 0
    # the kernel tells the terms apart, so an agreement is not one of two constant matrices
    assert np.ptp(expected[~np.eye(len(terms), dtype=bool)]) > 0.1
    assert np.max(np.abs(_normalized_gram(features) - expected)) <= 1e-12


def test_a_round_is_a_block_of_its_own_counting_the_nodes():
    terms = _terms()
    fitted = WeisfeilerLehmanFeatures(h=2).fit(terms)
    features = fitted.transform(terms)
    widths = [len(block) for block in fitted.vocabulary_]

    assert len(widths) == 3 and features.shape[1] == sum(widths)
    nodes = np.array([len(list(term.positions())) for term in terms], dtype=float)
    start = 0
    for width in widths:
        # every node carries exactly one label in every round
        assert np.array_equal(features[:, start : start + width].sum(axis=1), nodes)
        start += width


def test_a_held_out_term_reads_the_training_columns_of_a_fit_over_both():
    terms = _terms()
    training, held_out = terms[:6], terms[6:]
    alone = WeisfeilerLehmanFeatures(h=2).fit(training)
    both = WeisfeilerLehmanFeatures(h=2).fit(training + held_out)

    # the training terms come first, so their labels open each round's block in both fits
    columns = []
    start = 0
    for small, large in zip(alone.vocabulary_, both.vocabulary_, strict=True):
        assert list(large)[: len(small)] == list(small)
        columns.extend(range(start, start + len(small)))
        start += len(large)
    restricted = both.transform(held_out)[:, columns]
    read = alone.transform(held_out)

    assert np.array_equal(read, restricted)
    # the held-out terms carry labels the training terms lack, so the restriction drops something
    assert both.transform(held_out).sum() > read.sum()


def test_a_label_no_training_term_has_is_dropped_with_what_derives_from_it():
    a = _lit()
    fitted = WeisfeilerLehmanFeatures(h=1).fit([a, _neg(a), _add(a, a)])
    unseen = Tree("mul", (a, a))
    row = fitted.transform([unseen])[0]

    # round 0 sees the two ``lit`` leaves, a first and a second argument as in ``add``; round 1
    # sees nothing, since every node there either is the unseen ``mul`` or neighbours it
    assert row.sum() == 2.0
    assert fitted.transform([_add(a, a)])[0].sum() == 6.0


def test_presence_is_whether_a_label_occurs():
    terms = _terms()
    counts = WeisfeilerLehmanFeatures(h=1).fit(terms).transform(terms)
    presence = WeisfeilerLehmanFeatures(h=1, presence=True).fit(terms).transform(terms)

    assert counts.max() > 1.0
    assert np.array_equal(presence, (counts > 0).astype(float))


def test_concatenated_features_stand_side_by_side_in_the_order_given():
    terms = _terms()
    parts = [WeisfeilerLehmanFeatures(h=0), WeisfeilerLehmanFeatures(h=1, presence=True)]
    joined = ConcatenatedFeatures(parts).fit(terms).transform(terms)
    separate = [
        WeisfeilerLehmanFeatures(h=0).fit(terms).transform(terms),
        WeisfeilerLehmanFeatures(h=1, presence=True).fit(terms).transform(terms),
    ]

    assert np.array_equal(joined, np.hstack(separate))


def test_a_copy_of_a_fitted_map_reads_alike_and_leaves_the_original_alone():
    terms = _terms()
    fitted = WeisfeilerLehmanFeatures(h=2).fit(terms[:6])
    copied = copy.deepcopy(fitted)
    copied.fit(terms)

    assert copied.to_grakel_graph is fitted.to_grakel_graph
    assert sum(map(len, copied.vocabulary_)) > sum(map(len, fitted.vocabulary_))
    assert np.array_equal(copy.deepcopy(fitted).transform(terms), fitted.transform(terms))


def test_reading_before_a_fit_is_refused():
    with pytest.raises(NotFittedError, match="fit"):
        WeisfeilerLehmanFeatures().transform(_terms())


def test_a_fit_on_no_term_is_refused():
    with pytest.raises(ValueError, match="at least one term"):
        WeisfeilerLehmanFeatures().fit([])


@pytest.mark.parametrize("h", [-1, 1.0, True])
def test_a_round_count_that_is_not_a_natural_number_is_refused(h):
    with pytest.raises(ValueError, match="relabeling rounds"):
        WeisfeilerLehmanFeatures(h=h)


def test_concatenating_nothing_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        ConcatenatedFeatures([])
