from __future__ import annotations

from collections.abc import Hashable
from typing import TypeVar

import grakel
import networkx as nx
from cosy.core.tree import Tree
from grakel.utils import graph_from_networkx

T = TypeVar("T", bound=Hashable)  # type of terminals

# The label a root carries in place of an argument index.  A root is not the argument of
# anything, and giving it an index would make it comparable to a position that is one.
ROOT_POSITION = "r"


def indexed_label(symbol: object, argument_index: int | None) -> str:
    """Return the node label of a term position: its symbol, plus where it sits in its parent.

    The graph of a term has one node per position, an edge from the position of every subterm to
    the positions of its immediate subterms, and the symbol at the position as the node label.
    That label alone forgets the order of the arguments, so ``f(a, b)`` and ``f(b, a)`` become the
    same graph.  The order is restored by extending every node label with the argument index of
    its position.  The index has to go into the *node* label rather than onto the edge, because
    the base kernel of the Weisfeiler-Lehman construction is a vertex histogram, which never looks
    at edge attributes, so writing it onto the edge, as this conversion did, discarded it
    silently.

    Args:
        symbol: The symbol at the position.
        argument_index: Which argument of its parent this position is, or ``None`` for a root.

    Returns:
        str: The label.  A string because grakel's label maps are string-keyed, whatever its
            documentation says about tuples.
    """
    position = ROOT_POSITION if argument_index is None else str(argument_index)
    return f"{symbol}@{position}"


def to_indexed_nx_digraph(
    tree: Tree[T], start_index: int = 0, argument_index: int | None = None
) -> tuple[nx.DiGraph, dict[int, T]]:
    """Build the graph of ``tree``: a node per position, an edge to each immediate subterm.

    The node labels carry the argument index, so the graph keeps the order of the arguments.

    Args:
        tree: The term.
        start_index: The node id to give the root of this subtree.
        argument_index: Which argument of its parent the root of this subtree is, or ``None`` at
            the top, where there is no parent.

    Returns:
        tuple[nx.DiGraph, dict[int, T]]: The graph, and the map from node id to the symbol at
            that position.
    """
    graph = nx.DiGraph()
    label = indexed_label(tree.root, argument_index)

    if not tree.children:
        graph.add_node(start_index, symbol=label, leaf=True)
        return graph, {start_index: tree.root}

    graph.add_node(start_index, symbol=label, leaf=False)
    index_mapping = {start_index: tree.root}
    current_index = start_index + 1
    for position, child in enumerate(tree.children):
        child_graph, child_mapping = to_indexed_nx_digraph(child, current_index, position)
        graph = nx.compose(graph, child_graph)
        graph.add_edge(start_index, current_index)
        index_mapping.update(child_mapping)
        current_index += len(child_mapping)
    return graph, index_mapping


def to_grakel_graph(tree: Tree[T]) -> grakel.Graph:
    """Convert a term into the undirected labeled graph the WL kernel runs on.

    Args:
        tree: The term.

    Returns:
        grakel.Graph: The graph, with the argument order carried in the node labels.
    """
    # grakel reads the labels off the ``symbol`` attribute, so the index mapping the conversion
    # also returns is not needed here.
    nx_graph, _index_mapping = to_indexed_nx_digraph(tree)
    return graph_from_networkx([nx_graph.to_undirected()], node_labels_tag="symbol")
