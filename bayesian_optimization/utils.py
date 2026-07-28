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

