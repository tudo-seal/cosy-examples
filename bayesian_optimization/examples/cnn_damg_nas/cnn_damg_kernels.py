from typing import Callable

import re

from grakel.utils import graph_from_networkx
import networkx as nx

from sklearn.gaussian_process.kernels import Sum, Product, ConstantKernel, WhiteKernel

from cosy.core.tree import Tree

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo_algebras import (edgelist_algebra, hierarchy_algebra,
                                                                        operator_histogram_algebra)

from bayesian_optimization.kernels.graph_kernel import WeisfeilerLehmanKernel
from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

from bayesian_optimization.utils import to_grakel_graph, to_indexed_nx_digraph

# ============================================================================
# Kernel Setup
# ============================================================================

def as_DAMG(t: Tree, verbose=False):
    if verbose:
        edgelist, posA = t.interpret(edgelist_algebra(verbose))
    else:
        edgelist = t.interpret(edgelist_algebra(verbose))

    G = nx.MultiDiGraph()
    G.add_edges_from(edgelist)

    relabel = {n: re.sub(r"[)][(][-]*[0-9]*[.][0-9]*[,]\s[-]*[0-9]*[.][0-9]*[)]", ")", n)
               for n in G.nodes()}

    for n in G.nodes():
        G.nodes[n]['symbol'] = relabel[n]

    gk_graph = graph_from_networkx([G.to_undirected()], node_labels_tag='symbol')

    if verbose:
        return gk_graph, G, posA, relabel

    return gk_graph

def hierarchical_tree(n: int) -> Callable[[Tree], Tree]:
    def transform(tree: Tree) -> Tree:
        return tree.interpret(hierarchy_algebra(n))
    return transform

def as_hierarchical_tree_graph(n: int):
    def as_tree_graph(tree: Tree):
        h_tree = tree.interpret(hierarchy_algebra(n))
        return to_grakel_graph(h_tree)
    return as_tree_graph

def as_hierarchical_damg(n: int):
    def as_damg(tree: Tree):
        h_tree = tree.interpret(hierarchy_algebra(n))
        return as_DAMG(h_tree)
    return as_damg

"""
Hierarchy 0 for Tree is simply too much non-relevant information, we will not consider this and start from Hierarchy 1
for Trees, as it will remove all arguments with redundant information.
"""
tree_kernel_1 = OrderedRootedSubtreeKernel(tree_transformation=hierarchical_tree(1))
tree_kernel_2 = OrderedRootedSubtreeKernel(tree_transformation=hierarchical_tree(2))
tree_kernel_3 = OrderedRootedSubtreeKernel(tree_transformation=hierarchical_tree(3))

weighted_tree_kernel_1 = Product(ConstantKernel(0.01, constant_value_bounds=(1e-10, 1e10)), tree_kernel_1)
weighted_tree_kernel_2 = Product(ConstantKernel(0.01, constant_value_bounds=(1e-10, 1e10)), tree_kernel_2)
weighted_tree_kernel_3 = Product(ConstantKernel(0.01, constant_value_bounds=(1e-10, 1e10)), tree_kernel_3)

hierarchical_tree_kernel = Sum(weighted_tree_kernel_1, Sum(weighted_tree_kernel_2, weighted_tree_kernel_3))

hierarchical_tree_kernel_23 = Sum(weighted_tree_kernel_2, weighted_tree_kernel_3)

hierarchical_tree_kernel_12 = Sum(weighted_tree_kernel_1, weighted_tree_kernel_2)

hierarchical_tree_kernel_13 = Sum(weighted_tree_kernel_1, weighted_tree_kernel_3)

# We will only consider noisy hierarchical kernels, as training neural networks is always a "noisy" process...
noisy_hierarchical_tree_kernel = Sum(hierarchical_tree_kernel, WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-12, 1e2)))

noisy_hierarchical_tree_kernel_23 = Sum(hierarchical_tree_kernel_23, WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-12, 1e2)))

noisy_hierarchical_tree_kernel_12 = Sum(hierarchical_tree_kernel_12, WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-12, 1e2)))

noisy_hierarchical_tree_kernel_13 = Sum(hierarchical_tree_kernel_13, WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-12, 1e2)))


wl_kernel_1 = WeisfeilerLehmanKernel(n_iter=1.0, normalize=True,to_grakel_graph=as_hierarchical_tree_graph(1))
wl_kernel_2 = WeisfeilerLehmanKernel(n_iter=1.0, normalize=True, to_grakel_graph=as_hierarchical_tree_graph(2))
wl_kernel_3 = WeisfeilerLehmanKernel(n_iter=1.0, normalize=True, to_grakel_graph=as_hierarchical_tree_graph(3))

weighted_wl_kernel_1 = Product(ConstantKernel(0.01, constant_value_bounds=(1e-10, 1e10)), wl_kernel_1)
weighted_wl_kernel_2 = Product(ConstantKernel(0.01, constant_value_bounds=(1e-10, 1e10)), wl_kernel_2)
weighted_wl_kernel_3 = Product(ConstantKernel(0.01, constant_value_bounds=(1e-10, 1e10)), wl_kernel_3)

hierarchical_wl_kernel = Sum(weighted_wl_kernel_1, Sum(weighted_wl_kernel_2, weighted_wl_kernel_3))

hierarchical_wl_kernel_23 = Sum(weighted_wl_kernel_2, weighted_wl_kernel_3)

hierarchical_wl_kernel_12 = Sum(weighted_wl_kernel_1, weighted_wl_kernel_2)

hierarchical_wl_kernel_13 = Sum(weighted_wl_kernel_1, weighted_wl_kernel_3)

# We will only consider noisy hierarchical kernels, as training neural networks is always a "noisy" process...
noisy_hierarchical_wl_kernel = Sum(hierarchical_wl_kernel, WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-12, 1e2)))

noisy_hierarchical_wl_kernel_23 = Sum(hierarchical_wl_kernel_23, WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-12, 1e2)))

noisy_hierarchical_wl_kernel_12 = Sum(hierarchical_wl_kernel_12, WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-12, 1e2)))

noisy_hierarchical_wl_kernel_13 = Sum(hierarchical_wl_kernel_13, WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-12, 1e2)))

"""
Hierarchy 1 and 2 are basically just relabelings for DAMGs. Therefore, we don't consider the combination of 1 and 2 as
the structure doesn't change.
"""
damg_kernel_1 = WeisfeilerLehmanKernel(n_iter=1.0, normalize=True,to_grakel_graph=as_hierarchical_damg(1))
damg_kernel_2 = WeisfeilerLehmanKernel(n_iter=1.0, normalize=True, to_grakel_graph=as_hierarchical_damg(2))
damg_kernel_3 = WeisfeilerLehmanKernel(n_iter=1.0, normalize=True, to_grakel_graph=as_hierarchical_damg(3))

weighted_damg_kernel_1 = Product(ConstantKernel(0.01, constant_value_bounds=(1e-10, 1e10)),
                                 damg_kernel_1)
weighted_damg_kernel_2 = Product(ConstantKernel(0.01, constant_value_bounds=(1e-10, 1e10)),
                                 damg_kernel_2)
weighted_damg_kernel_3 = Product(ConstantKernel(0.01, constant_value_bounds=(1e-10, 1e10)),
                                 damg_kernel_3)

hierarchical_damg_kernel = Sum(weighted_damg_kernel_1, Sum(weighted_damg_kernel_2, weighted_damg_kernel_3))

hierarchical_damg_kernel_23 = Sum(weighted_damg_kernel_2, weighted_damg_kernel_3)

hierarchical_damg_kernel_13 = Sum(weighted_damg_kernel_1, weighted_damg_kernel_3)

# We will only consider noisy hierarchical kernels, as training neural networks is always a "noisy" process...
noisy_hierarchical_damg_kernel = Sum(hierarchical_damg_kernel, WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-12, 1e2)))

noisy_hierarchical_damg_kernel_23 = Sum(hierarchical_damg_kernel_23, WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-12, 1e2)))

noisy_hierarchical_damg_kernel_13 = Sum(hierarchical_damg_kernel_13, WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-12, 1e2)))

"""
Combining the different Tree-WL and DAMG-WL Kernel should be a good idea, right?
"""

# If this proves to behave good, we should test more combined kernels
combined_hierarchical_kernel = Sum(hierarchical_damg_kernel_13, hierarchical_wl_kernel)

noisy_combined_hierarchical_kernel = Sum(combined_hierarchical_kernel, WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-12, 1e2)))
