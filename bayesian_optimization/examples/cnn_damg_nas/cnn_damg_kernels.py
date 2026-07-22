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
#
# Amplitude and noise scale, and why they are what they are
# ---------------------------------------------------------
# The GP is fitted with normalize_y=True, so the targets it sees have unit variance.  The kernel's
# total prior variance must therefore be of order 1.  Each WeisfeilerLehman kernel below is built
# with normalize=True and hence has diagonal exactly 1.0, so the ConstantKernel factors ARE that
# prior variance: three of them at 0.3 plus a WhiteKernel at 0.1 sum to ~1.0.
#
# They used to start at 0.01 each with bounds (1e-10, 1e10), giving a total prior variance of 0.04
# against unit-variance data -- a ~25x mismatch that made the GP badly over-confident.  Measured
# consequence: sigma ~0.02-0.04, z-scores around -30, and Expected Improvement underflowing to
# ~1e-25..1e-47.  Since FitnessProportionalSelection used those raw EI values as roulette weights,
# a single individual received ~80% of the selection probability (see cnn_damg_usps_experiment.py,
# where the selection was changed to RankBasedSelection for the same reason).
#
# The bounds are now (1e-4, 1e2) instead of (1e-10, 1e10).  Restarts of the hyperparameter
# optimizer are drawn log-uniformly from the bounds, so 20 orders of magnitude meant almost every
# restart began somewhere absurd.  The narrower range still lets a hierarchy switch off (lower
# bound) or dominate (upper bound); fitted values observed in testing were 0.03-2.3.
#
# NOTE: these values only matter as a starting point when the hyperparameter optimizer runs.  The
# experiments pass kernel_optimizer="fmin_l_bfgs_b"; with kernel_optimizer=None they are frozen.
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

weighted_tree_kernel_1 = Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)), tree_kernel_1)
weighted_tree_kernel_2 = Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)), tree_kernel_2)
weighted_tree_kernel_3 = Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)), tree_kernel_3)

hierarchical_tree_kernel = Sum(weighted_tree_kernel_1, Sum(weighted_tree_kernel_2, weighted_tree_kernel_3))

hierarchical_tree_kernel_23 = Sum(weighted_tree_kernel_2, weighted_tree_kernel_3)

hierarchical_tree_kernel_12 = Sum(weighted_tree_kernel_1, weighted_tree_kernel_2)

hierarchical_tree_kernel_13 = Sum(weighted_tree_kernel_1, weighted_tree_kernel_3)

# We will only consider noisy hierarchical kernels, as training neural networks is always a "noisy" process...
noisy_hierarchical_tree_kernel = Sum(hierarchical_tree_kernel, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))

noisy_hierarchical_tree_kernel_23 = Sum(hierarchical_tree_kernel_23, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))

noisy_hierarchical_tree_kernel_12 = Sum(hierarchical_tree_kernel_12, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))

noisy_hierarchical_tree_kernel_13 = Sum(hierarchical_tree_kernel_13, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))


wl_kernel_1 = WeisfeilerLehmanKernel(n_iter=1.0, normalize=True,to_grakel_graph=as_hierarchical_tree_graph(1))
wl_kernel_2 = WeisfeilerLehmanKernel(n_iter=1.0, normalize=True, to_grakel_graph=as_hierarchical_tree_graph(2))
wl_kernel_3 = WeisfeilerLehmanKernel(n_iter=1.0, normalize=True, to_grakel_graph=as_hierarchical_tree_graph(3))

weighted_wl_kernel_1 = Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)), wl_kernel_1)
weighted_wl_kernel_2 = Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)), wl_kernel_2)
weighted_wl_kernel_3 = Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)), wl_kernel_3)

hierarchical_wl_kernel = Sum(weighted_wl_kernel_1, Sum(weighted_wl_kernel_2, weighted_wl_kernel_3))

hierarchical_wl_kernel_23 = Sum(weighted_wl_kernel_2, weighted_wl_kernel_3)

hierarchical_wl_kernel_12 = Sum(weighted_wl_kernel_1, weighted_wl_kernel_2)

hierarchical_wl_kernel_13 = Sum(weighted_wl_kernel_1, weighted_wl_kernel_3)

# We will only consider noisy hierarchical kernels, as training neural networks is always a "noisy" process...
noisy_hierarchical_wl_kernel = Sum(hierarchical_wl_kernel, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))

noisy_hierarchical_wl_kernel_23 = Sum(hierarchical_wl_kernel_23, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))

noisy_hierarchical_wl_kernel_12 = Sum(hierarchical_wl_kernel_12, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))

noisy_hierarchical_wl_kernel_13 = Sum(hierarchical_wl_kernel_13, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))

"""
Hierarchy 1 and 2 are basically just relabelings for DAMGs. Therefore, we don't consider the combination of 1 and 2 as
the structure doesn't change.
"""
damg_kernel_1 = WeisfeilerLehmanKernel(n_iter=1.0, normalize=True,to_grakel_graph=as_hierarchical_damg(1))
damg_kernel_2 = WeisfeilerLehmanKernel(n_iter=1.0, normalize=True, to_grakel_graph=as_hierarchical_damg(2))
damg_kernel_3 = WeisfeilerLehmanKernel(n_iter=1.0, normalize=True, to_grakel_graph=as_hierarchical_damg(3))

weighted_damg_kernel_1 = Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)),
                                 damg_kernel_1)
weighted_damg_kernel_2 = Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)),
                                 damg_kernel_2)
weighted_damg_kernel_3 = Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)),
                                 damg_kernel_3)

hierarchical_damg_kernel = Sum(weighted_damg_kernel_1, Sum(weighted_damg_kernel_2, weighted_damg_kernel_3))

hierarchical_damg_kernel_23 = Sum(weighted_damg_kernel_2, weighted_damg_kernel_3)

hierarchical_damg_kernel_13 = Sum(weighted_damg_kernel_1, weighted_damg_kernel_3)

# We will only consider noisy hierarchical kernels, as training neural networks is always a "noisy" process...
noisy_hierarchical_damg_kernel = Sum(hierarchical_damg_kernel, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))

noisy_hierarchical_damg_kernel_23 = Sum(hierarchical_damg_kernel_23, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))

noisy_hierarchical_damg_kernel_13 = Sum(hierarchical_damg_kernel_13, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))

"""
Combining the different Tree-WL and DAMG-WL Kernel should be a good idea, right?
"""

# If this proves to behave good, we should test more combined kernels
combined_hierarchical_kernel = Sum(hierarchical_damg_kernel_13, hierarchical_wl_kernel)

noisy_combined_hierarchical_kernel = Sum(combined_hierarchical_kernel, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))
