from collections.abc import Callable

import networkx as nx
from cosy.core.tree import Tree
from grakel.utils import graph_from_networkx
from sklearn.gaussian_process.kernels import ConstantKernel, Product, Sum, WhiteKernel

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_algebras import (
    edgelist_algebra,
    hierarchy_algebra,
)
from bayesian_optimization.kernels.graph_kernel import WeisfeilerLehmanKernel
from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel
from bayesian_optimization.utils import to_grakel_graph

# ============================================================================
# Kernel Setup
#
# Amplitude and noise scale, and why they are what they are
# ---------------------------------------------------------
# The Gaussian process is fitted with normalize_y=True, so the targets it sees have unit variance,
# and the total prior variance of the kernel has to be of the same order.  Every Weisfeiler-Lehman
# kernel below is built with normalize=True and therefore has diagonal exactly 1.0, which makes the
# ConstantKernel factors that prior variance and nothing else: three of them at 0.3 plus a
# WhiteKernel at 0.1 add up to about 1.0.
#
# The sister example bayesian_optimization/examples/damg_nas/damg_kernels.py still carries the
# earlier numbers, three amplitudes of 0.01 and a noise level of 1e-2, for a total prior variance
# of 0.04 against unit-variance data.  A prior that narrow makes the posterior over-confident, and
# an over-confident posterior drives expected improvement toward zero on every candidate.  A
# selection that reads those raw acquisition values as weights then hands nearly all of its
# probability to whichever individual underflowed last.
#
# The bounds are (1e-4, 1e2) rather than the (1e-10, 1e10) of the sister file.  Restarts of the
# hyperparameter optimizer are drawn log-uniformly from the bounds, so twenty orders of magnitude
# put almost every restart somewhere useless.  The narrower range still lets one granularity switch
# off at the lower bound or dominate at the upper one.
#
# These values are a starting point only while the hyperparameter optimizer runs.  A run that
# passes kernel_optimizer=None freezes them as written.
# ============================================================================

def hierarchical_tree(n: int) -> Callable[[Tree], Tree]:
    """Return the granularity fold of level ``n``.

    Truncating a term to a level is a fold: it keeps every position above the level with its symbol
    and replaces every subterm rooted at the level by the constant of its root symbol.  A
    hierarchical kernel is an ordinary kernel with such a fold precomposed, and this is that fold.
    Naming it makes it something a caller can pass on rather than apply and forget, which is what
    :func:`as_DAMG` needs, because what a node is called depends on the level it was folded to.

    Args:
        n (int): The granularity level.

    Returns:
        Callable[[Tree], Tree]: The fold.
    """
    def transform(tree: Tree) -> Tree:
        transformed: Tree = tree.interpret(hierarchy_algebra(n))
        return transformed
    return transform


def no_fold(tree: Tree) -> Tree:
    """Return the term unchanged: the fold of a kernel that reads the term as it is.

    Args:
        tree (Tree): The term.

    Returns:
        Tree: The same term.
    """
    return tree


def as_DAMG(t: Tree, fold: Callable[[Tree], Tree] = no_fold, verbose=False):
    """Convert a term into the labeled multigraph of the network it denotes.

    The edge-list algebra names a node by its symbol followed by its layout coordinate, because a
    name is what an edge list has instead of an identity: two instances of one layer have to be two
    nodes.  The kernel wants the opposite, two instances of one layer sharing one label, so the
    coordinate is stripped again for the ``symbol`` attribute, while the node keeps its full name
    as its identity.

    The strip used to be a regular expression over the name, matching a ``)`` followed by a pair of
    decimals.  That pattern fits the granularity 1 naming, ``str((layer, i, o)) + str(id)``, and
    fits nothing else: from granularity 2 the algebra names a node ``"node" + str(id)``, with no
    ``)`` in front of the coordinate, so nothing was stripped and the label was the drawing
    position.  Two architectures of the same shape whose layers sat at different positions then
    shared no label at all, and ``damg_kernel_2`` and ``damg_kernel_3`` compared positions rather
    than node types.

    Nothing has to be guessed: the algebra returns the map from node name to coordinate, so the
    suffix to strip is the one it names there.  ``removesuffix`` and not a slice, because the frame
    nodes (``input``, the loss, the optimizer, the epoch marker) stand in that map with a position
    for plotting while carrying no coordinate in their name, and for them the strip has to be a
    no-op.

    Args:
        t (Tree): The term.
        fold (Callable[[Tree], Tree]): The granularity fold to apply first, see
            :func:`hierarchical_tree`.  Taking it here rather than applying it outside is what lets
            one conversion serve every granularity. (Default value = :func:`no_fold`)
        verbose (bool): Also return the intermediate graph, the positions and the relabeling.
            (Default value = False)

    Returns:
        The grakel graph generator, and under ``verbose`` the multigraph, the position map and the
        relabeling beside it.
    """
    # The positions name the suffix, so the algebra is always asked for them.  ``verbose`` decides
    # what is handed back, not what is computed.
    #
    # ``deduplicate=True`` asks for one edge per distinct source, not one per input feature.  The
    # width is real, see ``edgelist_algebra``, but the one caller here cannot read it, so it is not
    # built in the first place.  A layer contributes one tuple per input feature and the
    # compositions concatenate what they are handed, so the list a real architecture builds grows
    # with the tensor widths while the graph it produces does not.
    edgelist, positions = fold(t).interpret(edgelist_algebra(True, deduplicate=True))

    # And once more here, because ``deduplicate`` thins each layer's own edges while the
    # compositions still concatenate: two branches can contribute the same pair.  Cheap now that
    # the list is short, and it is what makes the graph independent of how the term was composed.
    #
    # Dropping the multiplicity is invisible to this kernel, and that is why it is allowed.
    # ``graph_from_networkx`` hands grakel an adjacency dictionary, so any number of parallel edges
    # between two nodes arrives there as ``{'A': {'C': 1.0}}``, the same entry a single edge
    # produces.  ``test_edge_multiplicity_never_reaches_grakel`` asserts that premise instead of
    # assuming it.  The width therefore never reached the kernel, it only reached the clock.  A
    # kernel that does read edge weights would have to take the multiplicity from the edge list
    # itself, which is unchanged.
    #
    # ``dict.fromkeys`` and not ``set``: the insertion order is kept, so the graph is built in the
    # order the algebra produced, one less thing that differs from the previous behavior.
    edgelist = list(dict.fromkeys(edgelist))

    G = nx.MultiDiGraph()
    G.add_edges_from(edgelist)

    relabel = {n: n.removesuffix(str(positions[n])) if n in positions else n for n in G.nodes()}

    for n in G.nodes():
        G.nodes[n]['symbol'] = relabel[n]

    gk_graph = graph_from_networkx([G.to_undirected()], node_labels_tag='symbol')

    if verbose:
        return gk_graph, G, positions, relabel

    return gk_graph


def as_hierarchical_tree_graph(n: int):
    def as_tree_graph(tree: Tree):
        h_tree = hierarchical_tree(n)(tree)
        return to_grakel_graph(h_tree)
    return as_tree_graph

def as_hierarchical_damg(n: int):
    def as_damg(tree: Tree):
        return as_DAMG(tree, fold=hierarchical_tree(n))
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


# h is the number of relabeling rounds, counted on top of the comparison of the initial labels, and
# h = 0 is a plain histogram over those labels.  A round costs a full relabeling of every graph and
# it is not a hyperparameter the marginal likelihood can adjust, so one is spent only where a
# histogram is blind to a difference that matters.
#
# These three read the folded term itself, where a label is a combinator symbol, and there the
# histogram still tells most terms apart.  Whether a round or two buys anything on this space is a
# measurement and belongs to the run that makes it.  What a round cannot do is revive a comparison:
# once two graphs share no label, they share none afterwards either, because a round replaces a
# label by an injective compression of that label together with the multiset of its neighbors'
# labels, so distinct labels stay distinct.
wl_kernel_1 = WeisfeilerLehmanKernel(h=0, normalize=True,
                                     to_grakel_graph=as_hierarchical_tree_graph(1))
wl_kernel_2 = WeisfeilerLehmanKernel(h=0, normalize=True,
                                     to_grakel_graph=as_hierarchical_tree_graph(2))
wl_kernel_3 = WeisfeilerLehmanKernel(h=0, normalize=True,
                                     to_grakel_graph=as_hierarchical_tree_graph(3))

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
For DAMGs, hierarchy 1 and 2 differ only in the node labels, the graph itself being the same, so
the combination of 1 and 2 is not considered.

What "only the labels" means is sharper than it reads: from hierarchy 2 the label is the same for
every node, so the two levels differ in how much they say, not in what they describe.  See as_DAMG
for why that was invisible until the relabel reached those levels at all.
"""
# Here the round count differs by granularity, because what a node label carries differs by
# granularity.  At granularity 1 the label is the layer with its dimensions, so the label multiset
# alone already separates architectures and a histogram says something.  From granularity 2 the
# layer type is folded away and every node is called ``node``, so a histogram counts nodes and
# nothing else, and the composition reaches the kernel only through a relabeling round.
# ``test_from_hierarchy_two_the_kernel_needs_a_relabeling_round`` builds two architectures over the
# same two nodes, ``A || B`` against ``A ; B``, and pins both halves of that.
#
# That the coarse granularities need a round is new, because it was never visible: while the strip
# missed them, see as_DAMG, their labels were drawing positions, and a histogram looked
# discriminating there because two graphs shared a label exactly when they placed a node at the
# same spot.
damg_kernel_1 = WeisfeilerLehmanKernel(h=0, normalize=True, to_grakel_graph=as_hierarchical_damg(1))
damg_kernel_2 = WeisfeilerLehmanKernel(h=1, normalize=True, to_grakel_graph=as_hierarchical_damg(2))
damg_kernel_3 = WeisfeilerLehmanKernel(h=1, normalize=True, to_grakel_graph=as_hierarchical_damg(3))

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

# The same three granularities at h = 2 throughout.  Not a second default but a second option: the
# round count is the one structural axis the marginal likelihood cannot reach, since the fit adjusts
# the amplitudes and the noise level and nothing else, so a run that wants a different round count
# has to name a different kernel.  Which of the two a search space wants is a measurement, and the
# sum stays the default because one fitted amplitude per granularity is what lets a fit answer that
# question without a change of kernel.
damg_kernel_1_h2 = WeisfeilerLehmanKernel(h=2, normalize=True, to_grakel_graph=as_hierarchical_damg(1))
damg_kernel_2_h2 = WeisfeilerLehmanKernel(h=2, normalize=True, to_grakel_graph=as_hierarchical_damg(2))
damg_kernel_3_h2 = WeisfeilerLehmanKernel(h=2, normalize=True, to_grakel_graph=as_hierarchical_damg(3))

weighted_damg_kernel_1_h2 = Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)),
                                    damg_kernel_1_h2)
weighted_damg_kernel_2_h2 = Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)),
                                    damg_kernel_2_h2)
weighted_damg_kernel_3_h2 = Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)),
                                    damg_kernel_3_h2)

hierarchical_damg_kernel_h2 = Sum(
    weighted_damg_kernel_1_h2,
    Sum(weighted_damg_kernel_2_h2, weighted_damg_kernel_3_h2),
)

noisy_hierarchical_damg_kernel_h2 = Sum(
    hierarchical_damg_kernel_h2, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
)

# The coarsest granularity on its own, at two rounds and at one.  A single granularity is what a run
# names when it has measured that the sum spends two graph conversions on summands it then fits to a
# negligible amplitude, and the conversion is what the acquisition spends its time on.
#
# The two two-level sums above differ in whether their summands run at the same round count.
# ``damg_13`` is staggered, h = 0 for the first level and h = 1 for the second, so its summands
# read different things.  ``damg_23`` runs both at h = 1, so its summands differ only in the
# granularity they fold to.
noisy_damg_kernel_3_h2 = Sum(
    weighted_damg_kernel_3_h2, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
)

noisy_damg_kernel_3 = Sum(
    weighted_damg_kernel_3, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
)

# The staggered sum, one round count per granularity, ascending with the granularity's coarseness.
# Its point is that the three summands stay different.  Granularity 1 keeps the layer with its
# dimensions and separates on the labels alone, so it is put where a histogram already works.
# Granularities 2 and 3 call every node ``node`` and reach the composition only through rounds, so
# each is put where it stops being the model that returns one value for every pair, and the coarser
# of the two gets the wider neighborhood.  The test that the three hierarchy kernels do not collapse
# onto each other is the guard on that.
hierarchical_damg_kernel_h012 = Sum(
    weighted_damg_kernel_1,          # level 1 at h = 0
    Sum(Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)), damg_kernel_2),
        Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)), damg_kernel_3_h2)),
)

noisy_hierarchical_damg_kernel_h012 = Sum(
    hierarchical_damg_kernel_h012, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
)

# ---------------------------------------------------------------------------
# The term graph at its two finest granularities, at two rounds
# ---------------------------------------------------------------------------
# Granularity 0 is the term as it was synthesized and granularity 1 the term with the arguments
# folded away that carry no structure.  Both read the term rather than the architecture graph, so a
# label is a combinator symbol, and two rounds buy the neighborhood that a histogram over those
# symbols cannot see.  They exist as separate bases so that a run can sum the two readings of a
# term with the reading of the network it denotes.
wl_kernel_0_h2 = WeisfeilerLehmanKernel(h=2, normalize=True,
                                        to_grakel_graph=as_hierarchical_tree_graph(0))
wl_kernel_1_h2 = WeisfeilerLehmanKernel(h=2, normalize=True,
                                        to_grakel_graph=as_hierarchical_tree_graph(1))

weighted_wl_kernel_0_h2 = Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)),
                                  wl_kernel_0_h2)
weighted_wl_kernel_1_h2 = Product(ConstantKernel(0.3, constant_value_bounds=(1e-4, 1e2)),
                                  wl_kernel_1_h2)

# The sum of the two term readings above and the finest reading of the architecture graph.
#
# A sum is not free.  It carries one more fitted amplitude per summand, so it asks more of the model
# selection than a single base does, and it pays for one graph conversion per summand on every
# acquisition step.  Whether that buys anything is a property of the objective, so this entry stands
# in the catalogue beside its three summands rather than in front of them.
top3_sum_kernel = Sum(
    weighted_wl_kernel_1_h2,
    Sum(weighted_wl_kernel_0_h2, weighted_damg_kernel_1),   # damg level 1 at h = 0
)

noisy_top3_sum_kernel = Sum(
    top3_sum_kernel, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1))
)

#: The name this sum was registered under first, kept so that a record naming it still resolves.
selected_kernel = top3_sum_kernel
noisy_selected_kernel = noisy_top3_sum_kernel


# The finest granularity of the architecture graph on its own.  A sum whose coarser summands are
# fitted to a negligible amplitude is its finest summand, and the graph conversion is still paid for
# all three, so a run that has measured that has to be able to name the summand.
noisy_damg_kernel_1 = Sum(
    weighted_damg_kernel_1,
    WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)),
)

#: The kernels a run may be asked for by name.  A run records which one it used, so two runs of one
#: script are told apart by their record rather than by whoever remembers what the file said that
#: day.
#:
#: The default is a sum rather than a single granularity.  Which granularity carries the prediction
#: is a property of the objective at hand and is not known before the run, so a sum with one fitted
#: amplitude per granularity is never blind, at the price of giving up a little on each.  The single
#: granularities stand beside it for a run that has measured which one it wants.

NAMED_KERNELS = {
    "damg": noisy_hierarchical_damg_kernel,
    "damg_1": noisy_damg_kernel_1,
    "damg@h2": noisy_hierarchical_damg_kernel_h2,
    "damg_13": noisy_hierarchical_damg_kernel_13,
    "damg_23": noisy_hierarchical_damg_kernel_23,
    "damg@h012": noisy_hierarchical_damg_kernel_h012,
    "damg_3": noisy_damg_kernel_3,
    "damg_3@h2": noisy_damg_kernel_3_h2,
    "tree": noisy_hierarchical_tree_kernel,
    "wl_tree": noisy_hierarchical_wl_kernel,
    # A name that says what the entry is: the sum of the three bases above.
    "top3_sum": noisy_top3_sum_kernel,
    # The name that sum was registered under first, kept so that a record naming it resolves.
    "selected": noisy_top3_sum_kernel,
}


def named_kernel(name):
    """Return the kernel a run asked for.

    Args:
        name (str): A key of :data:`NAMED_KERNELS`.

    Returns:
        The sklearn kernel.

    Raises:
        ValueError: If there is no such kernel.  Falling back to a default would make the run
            record say one thing and the fit do another.
    """
    if name not in NAMED_KERNELS:
        msg = f"no kernel named {name!r}; the ones a run may ask for are {sorted(NAMED_KERNELS)}"
        raise ValueError(msg)
    return NAMED_KERNELS[name]

"""
Combining the different Tree-WL and DAMG-WL Kernel should be a good idea, right?
"""

# If this proves to behave good, we should test more combined kernels
combined_hierarchical_kernel = Sum(hierarchical_damg_kernel_13, hierarchical_wl_kernel)

noisy_combined_hierarchical_kernel = Sum(combined_hierarchical_kernel, WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-6, 1e1)))
