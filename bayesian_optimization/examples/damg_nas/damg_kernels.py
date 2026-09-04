from typing import Callable

import networkx as nx
from cosy.core.tree import Tree
from grakel.utils import graph_from_networkx
from sklearn.gaussian_process.kernels import ConstantKernel, Product, Sum, WhiteKernel

from bayesian_optimization.examples.damg_nas.damg_repo_algebras import (
    edgelist_algebra,
    hierarchy_algebra,
    operator_histogram_algebra,
)
from bayesian_optimization.kernels.graph_kernel import WeisfeilerLehmanKernel
from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel
from bayesian_optimization.utils import to_grakel_graph, to_indexed_nx_digraph

# ============================================================================
# Kernel Setup
# ============================================================================

def as_DAMG(t: Tree, verbose=False):
    """Convert a term into the labeled multigraph of the network it denotes.

    The edge-list algebra names a node by its symbol followed by its layout coordinate, because a
    name is what an edge list has instead of an identity: two instances of one layer have to be
    two nodes.  The kernel wants the opposite, two instances of one layer sharing one label, so
    the coordinate is stripped again for the ``symbol`` attribute, while the node keeps its full
    name as its identity.

    The strip used to be a regular expression matching a ``)`` followed by a pair of decimals.
    That pattern fits the hierarchy 1 naming, ``str((layer, i, o)) + str(id)``, and fits nothing
    else: from hierarchy 2 the algebra names a node ``"node" + str(id)``, with no ``)`` in front
    of the coordinate, so nothing was stripped and the label *was* the drawing position.  Two
    architectures of the same shape whose nodes sat at different positions then shared no label at
    all, and ``damg_kernel_2`` and ``damg_kernel_3`` compared drawings rather than node types.

    Nothing has to be guessed: the algebra returns the map from node name to coordinate, so the
    suffix to strip is the one it names there.  ``removesuffix`` and not a slice, because the
    frame nodes (``input``, the loss, the optimizer, the epoch marker) stand in that map with a
    position for plotting while carrying no coordinate in their name, and for them the strip has
    to be a no-op.

    Args:
        t (Tree): The term.
        verbose (bool): Also return the multigraph, the position map and the relabeling.
            (Default value = False)

    Returns:
        The grakel graph generator, and under ``verbose`` the multigraph, the positions and the
        relabeling beside it.
    """
    # The positions name the suffix, so the algebra is always asked for them.  ``verbose`` decides
    # what is handed back, not what is computed.
    edgelist, posA = t.interpret(edgelist_algebra(True))

    G = nx.MultiDiGraph()
    G.add_edges_from(edgelist)

    relabel = {n: (n.removesuffix(str(posA[n])) if n in posA else n) for n in G.nodes()}

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


# h is the number of relabeling rounds, counted on top of the comparison of the initial labels,
# and h=0 is a plain histogram over those labels.  A round costs a full relabeling of every graph
# and it is not a hyperparameter the marginal likelihood can adjust, so one is spent only where a
# histogram is blind to a difference that matters.
#
# These three read the folded term itself, where a label is a combinator symbol, and there the
# histogram still tells most terms apart.  Over the first forty terms of target_len_3 it leaves
# 17.9 percent of the pairs at exactly 1.0 on granularity 2 and 3, against 13.8 percent with one
# round, and none at all on granularity 1.  So no round.
wl_kernel_1 = WeisfeilerLehmanKernel(h=0, normalize=True,
                                     to_grakel_graph=as_hierarchical_tree_graph(1))
wl_kernel_2 = WeisfeilerLehmanKernel(h=0, normalize=True,
                                     to_grakel_graph=as_hierarchical_tree_graph(2))
wl_kernel_3 = WeisfeilerLehmanKernel(h=0, normalize=True,
                                     to_grakel_graph=as_hierarchical_tree_graph(3))

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
# Here the round count differs by granularity, because what a node label carries differs by
# granularity.  At granularity 1 the label is the layer with its dimensions, and the histogram
# alone already separates: the first forty terms of target_len_3 give forty different label
# multisets, and no off-diagonal entry of the Gram matrix reaches 1.0 without a round.  From
# granularity 2 the layer is folded away and every node is called node, so a histogram counts
# nodes and nothing else, and those same forty terms leave one single multiset.  Two
# architectures over two nodes, A before B against A beside B, score 1.0 without a round, 0.668
# with one and 0.502 with two.
#
# That the coarse granularities need a round is new, because it was never visible: while the
# strip missed them, see as_DAMG, their labels were drawing positions, and a histogram looked
# discriminating there because two graphs shared a label exactly when they placed a node at the
# same spot.
damg_kernel_1 = WeisfeilerLehmanKernel(h=0, normalize=True, to_grakel_graph=as_hierarchical_damg(1))
damg_kernel_2 = WeisfeilerLehmanKernel(h=1, normalize=True, to_grakel_graph=as_hierarchical_damg(2))
damg_kernel_3 = WeisfeilerLehmanKernel(h=1, normalize=True, to_grakel_graph=as_hierarchical_damg(3))

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
