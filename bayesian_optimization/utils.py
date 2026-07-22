from __future__ import annotations

from cosy.core.tree import Tree
from cosy.core.solution_space import RHSRule, NonTerminalArgument, SolutionSpace

from collections.abc import Hashable
from typing import Generic, TypeVar, Union
import typing
import random
import networkx as nx
import grakel
from grakel.utils import graph_from_networkx

from copy import copy

T = TypeVar("T", bound=Hashable) # type of terminals


def canonical_tree(tree: Tree[T]) -> Tree[T]:
    """Rebuild a tree bottom-up so that ``size`` and ``_hash`` are consistent with its structure.

    Workaround for a defect in ``cosy.core.tree``: ``Tree.replace_subtree_at`` mutates
    ``current.children`` in place after copying, so every ancestor of the replacement point keeps
    the ``size`` and ``_hash`` it had *before* the replacement.  ``Tree.__eq__`` compares ``size``
    first, so trees produced by crossover compare unequal -- and hash differently -- to the
    structurally identical tree built directly.  Any ``set``/``dict`` keyed on trees therefore
    fails to recognise them.

    Rebuilding through ``Tree.__init__`` recomputes both fields, which restores the hash/eq
    contract.  The result is structurally identical to the input; only the cached fields change.

    Remove this once the defect is fixed upstream in cosy (see cosy-improvements.txt, section A1).
    """
    if not tree.children:
        return tree if tree.size == 1 else Tree(root=tree.root)
    return Tree(root=tree.root, children=tuple(canonical_tree(c) for c in tree.children))


def to_indexed_nx_digraph(tree: Tree[T], start_index: int = 0) -> tuple[nx.DiGraph, dict[int, T]]:

    G = nx.DiGraph()

    if not tree.children:
        G.add_node(start_index, symbol=str(tree.root), leaf=True)
        return G, {start_index: tree.root}
    else:
        G.add_node(start_index, symbol=str(tree.root), leaf=False)
        index_mapping = {start_index: tree.root}
        current_index = start_index + 1
        for child in tree.children:
            Gc, child_mapping = to_indexed_nx_digraph(child, current_index)
            G = nx.compose(G, Gc)
            G.add_edge(start_index, current_index, argument=child.root)
            index_mapping.update(child_mapping)
            current_index += len(child_mapping)
        return G, index_mapping

def to_grakel_graph(tree: Tree[T]) -> grakel.Graph:
    nx_graph, node_labels = to_indexed_nx_digraph(tree)
    gk_graph = graph_from_networkx([nx_graph.to_undirected()], node_labels_tag='symbol',
                                   edge_labels_tag='argument')
    # I hate grakel! Why do they write tuple in their docs when it actually must be a str?!
    return gk_graph

