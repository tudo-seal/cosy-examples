"""Explicit Weisfeiler-Lehman features of terms: the label counts the WL kernel compares.

:class:`~bayesian_optimization.kernels.graph_kernel.WeisfeilerLehmanKernel` compares two terms
through the graphs a translation makes of them.  With the vertex histogram as its base, ``k(t, t')``
is the sum over the rounds ``i = 0..h`` of the dot products of the two graphs' label counts in round
``i``.  Those counts are a feature map, and this module computes it, so that a model that needs
vectors instead of a kernel, a random forest for one, reads exactly what the kernel reads.

**The vocabulary is fixed by the fit.**  :meth:`WeisfeilerLehmanFeatures.fit` collects the labels of
the training terms, one column per label and round, and
:meth:`~WeisfeilerLehmanFeatures.transform` maps any term onto those columns.  A label no training
term has is dropped: it would be zero in every training row, so no model fitted on those rows could
use it.  A label of a later round that is built from a dropped one is dropped with it, since no
training node can carry it.

**It is grakel's relabeling, mirrored.**  A graph is read through grakel's own ``Graph``, its edge
dictionary and its node labels in the dictionary format grakel's ``WeisfeilerLehman`` converts to.
The label of a node in round ``i + 1`` is the pair of its label in round ``i`` and the sorted
multiset of its neighbours' labels in round ``i``, the credential grakel compresses.  The columns
of a round are numbered in the order their labels first occur, term by term and node by node,
which is the order grakel's ``VertexHistogram`` numbers them in.  The kernel's normalized Gram is
therefore the normalized Gram of these features, which the tests hold to the kernel's own.  ``h``
counts the rounds as the kernel does (the note on ``_GrakelWeisfeilerLehmanBase``), so ``h``
rounds give ``h + 1`` blocks of columns.

A round's labels are kept as integers, the column a label has in its round, so a credential is a
pair of small integers and not a nesting of every earlier round.  A label the fit has not seen
reads as ``-1`` while a term is transformed, and a credential holding ``-1`` is never one the fit
has seen, which is how a dropped label takes what derives from it along.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Protocol

import numpy as np
from cosy.core.tree import Tree
from grakel import Graph
from sklearn.exceptions import NotFittedError

from ..utils import to_grakel_graph as to_tree_graph
from .graph_kernel import _GRAPH_CACHE, _materialize_graph

#: What a label of a round reads as when the fit has not seen it.
_UNSEEN = -1


class TermFeatures(Protocol):
    """A feature map over terms whose columns a fit fixes: what a vector model reads a term by."""

    def fit(self, terms: Sequence[Any]) -> TermFeatures:
        """Fix the columns on ``terms`` and answer the map."""
        ...

    def transform(self, terms: Sequence[Any]) -> np.ndarray:
        """Return one row of features per term, over the columns the fit fixed."""
        ...


def _edges_and_labels(graph: Any) -> tuple[dict[Any, Any], dict[Any, Hashable]]:
    """Return the edge dictionary and the node labels of a graph, as grakel's WL reads them.

    A ``Graph`` is converted to the dictionary format in place, as grakel's ``WeisfeilerLehman``
    converts it; anything else is the iterable of an edge structure, node labels and, optionally,
    edge labels, from which grakel builds the same ``Graph``.

    Args:
        graph: A converted term.

    Returns:
        tuple[dict[Any, Any], dict[Any, Hashable]]: The neighbours of every node with any, and the
            label of every node.

    Raises:
        TypeError: If it is neither, as grakel refuses it.
    """
    if isinstance(graph, Graph):
        graph.desired_format("dictionary")
    else:
        try:
            parts = list(graph)
        except TypeError as error:
            msg = f"a converted term is a grakel Graph or an iterable of its parts, not {graph!r}"
            raise TypeError(msg) from error
        if len(parts) not in (2, 3):
            msg = (
                "a converted term given as parts is an edge structure, node labels and "
                f"optionally edge labels, not {len(parts)} parts"
            )
            raise TypeError(msg)
        graph = Graph(
            parts[0], parts[1], parts[2] if len(parts) == 3 else {}, graph_format="dictionary"
        )
    return graph.get_edge_dictionary(), graph.get_labels(purpose="dictionary")


class WeisfeilerLehmanFeatures:
    """The label counts of a term's graph in every Weisfeiler-Lehman round, one column per label.

    Parameters
    ----------
    h:
        Number of relabeling rounds, performed in addition to counting the initial labels, as
        :class:`~bayesian_optimization.kernels.graph_kernel.WeisfeilerLehmanKernel` counts them.
    to_grakel_graph:
        The translation from term to graph, the kernel's parameter of the same name.  A kernel
        and a feature map over the same translation share the converted graphs.
    presence:
        Whether a label occurs instead of how often: the features 0 or 1, the dot product of two
        rows the number of labels the two graphs share.

    Attributes
    ----------
    vocabulary_:
        After a fit, per round, the map from a label to its column in that round's block.  A
        label of round 0 is a node label; a label of a later round is the pair of a node's column
        in the round before and the sorted columns of its neighbours there.
    """

    def __init__(
        self,
        h: int = 1,
        to_grakel_graph: Callable[[Tree[Any]], Any] = to_tree_graph,
        presence: bool = False,
    ) -> None:
        if isinstance(h, bool) or not isinstance(h, int) or h < 0:
            msg = f"the number of relabeling rounds h is a non-negative integer, not {h!r}"
            raise ValueError(msg)
        self.h = h
        self.to_grakel_graph = to_grakel_graph
        self.presence = presence

    def _graph(self, term: Tree[Any]) -> Any:
        # the kernels' cache, under the kernels' key: the translation itself and the term
        key = (self.to_grakel_graph, term)
        cached = _GRAPH_CACHE.get(key)
        if cached is None:
            cached = _materialize_graph(self.to_grakel_graph(term))
            _GRAPH_CACHE[key] = cached
        return cached

    def _columns(
        self, term: Tree[Any], vocabulary: Sequence[dict[Hashable, int]], *, grow: bool
    ) -> list[list[int]]:
        """Return the column of every node's label in every round, the nodes in label-map order.

        Args:
            term: The term.
            vocabulary: Per round, the map from a label to its column.
            grow: Give a label not in the map the next column (a fit), or read it as unseen.

        Returns:
            list[list[int]]: Per round, one column per node, ``-1`` for a label not in the map.
        """
        edges, labels = _edges_and_labels(self._graph(term))
        nodes = list(labels)
        rounds: list[list[int]] = []
        previous: dict[Any, int] = {}
        for i in range(self.h + 1):
            columns = vocabulary[i]
            if i == 0:
                credentials: list[Hashable] = [labels[node] for node in nodes]
            else:
                credentials = [
                    (
                        previous[node],
                        tuple(sorted(previous[neighbour] for neighbour in edges.get(node, {}))),
                    )
                    for node in nodes
                ]
            read = []
            for credential in credentials:
                column = columns.get(credential)
                if column is None:
                    if grow:
                        column = len(columns)
                        columns[credential] = column
                    else:
                        column = _UNSEEN
                read.append(column)
            rounds.append(read)
            previous = dict(zip(nodes, read, strict=True))
        return rounds

    def fit(self, terms: Sequence[Tree[Any]]) -> WeisfeilerLehmanFeatures:
        """Fix the columns: every label the terms' graphs carry in every round.

        Args:
            terms: The training terms.

        Returns:
            WeisfeilerLehmanFeatures: This map, fitted.

        Raises:
            ValueError: If there is no term, since no column could be fixed.
        """
        terms = list(terms)
        if not terms:
            msg = "the features are fitted on at least one term"
            raise ValueError(msg)
        vocabulary: list[dict[Hashable, int]] = [{} for _ in range(self.h + 1)]
        for term in terms:
            self._columns(term, vocabulary, grow=True)
        self.vocabulary_ = tuple(vocabulary)
        return self

    def transform(self, terms: Sequence[Tree[Any]]) -> np.ndarray:
        """Return each term's label counts over the fitted columns, round 0's block first.

        Args:
            terms: The terms to read.

        Returns:
            np.ndarray: One row per term, one column per fitted label and round; under
                ``presence`` each entry 0 or 1.

        Raises:
            NotFittedError: If the map has not been fitted.
        """
        vocabulary = getattr(self, "vocabulary_", None)
        if vocabulary is None:
            msg = "the features read a term only over the columns a fit fixed: call fit first"
            raise NotFittedError(msg)
        terms = list(terms)
        widths = [len(columns) for columns in vocabulary]
        offsets = np.cumsum([0, *widths[:-1]])
        features = np.zeros((len(terms), sum(widths)), dtype=float)
        for row, term in enumerate(terms):
            read_rounds = self._columns(term, vocabulary, grow=False)
            for offset, read in zip(offsets, read_rounds, strict=True):
                for column, count in Counter(c for c in read if c != _UNSEEN).items():
                    features[row, offset + column] = count
        if self.presence:
            return (features > 0).astype(float)
        return features

    def fit_transform(self, terms: Sequence[Tree[Any]]) -> np.ndarray:
        """Fit on ``terms`` and read them.

        Args:
            terms: The training terms.

        Returns:
            np.ndarray: Their features.
        """
        terms = list(terms)
        return self.fit(terms).transform(terms)


class ConcatenatedFeatures:
    """Feature maps side by side: each fitted on the same terms, their columns in the order given.

    The explicit counterpart of a sum of kernels over several translations or round counts.  A
    sum weighs its summands, and a vector model that picks its own columns needs no weights.

    Parameters
    ----------
    parts:
        The feature maps, at least one.
    """

    def __init__(self, parts: Sequence[TermFeatures]) -> None:
        self.parts = list(parts)
        if not self.parts:
            msg = "a concatenation joins at least one feature map"
            raise ValueError(msg)

    def fit(self, terms: Sequence[Tree[Any]]) -> ConcatenatedFeatures:
        """Fit every part on ``terms``.

        Args:
            terms: The training terms.

        Returns:
            ConcatenatedFeatures: This map, fitted.
        """
        terms = list(terms)
        for part in self.parts:
            part.fit(terms)
        return self

    def transform(self, terms: Sequence[Tree[Any]]) -> np.ndarray:
        """Return every part's features of ``terms``, side by side.

        Args:
            terms: The terms to read.

        Returns:
            np.ndarray: One row per term, the parts' columns in the order of the parts.
        """
        terms = list(terms)
        return np.hstack([np.asarray(part.transform(terms), dtype=float) for part in self.parts])

    def fit_transform(self, terms: Sequence[Tree[Any]]) -> np.ndarray:
        """Fit on ``terms`` and read them.

        Args:
            terms: The training terms.

        Returns:
            np.ndarray: Their features.
        """
        terms = list(terms)
        return self.fit(terms).transform(terms)
