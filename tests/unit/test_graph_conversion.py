"""The term-to-graph conversions, pinned value by value.

Three folds stand between a term and a kernel, and none of them had a test:

* ``to_indexed_nx_digraph`` and ``indexed_label`` build the graph of a term.  That graph has the
  positions of the term as its nodes, each labeled with the symbol at that position, and an edge
  between the position of every subterm and the positions of its immediate subterms.  A graph of
  that shape forgets the order among the arguments of a symbol, so every node label is extended by
  the argument index of its position, and the expectations below are read off that.
* ``edgelist_algebra`` interprets a term as the computation graph the network is, and
* ``as_DAMG`` turns that into a labeled graph, stripping the layout coordinate the edge list needs
  for identity back out of the label the kernel reads.

The last two have no specification, they are example code, so what is pinned is today's behavior,
and every case says which error it would catch.

The floating-point constants are written as expressions, ``Y0 + 0.2`` rather than digits.  The
coordinates end up inside node names, so the comparison is a string comparison over
``repr(float)``: ``-3.8 + 0.2`` is ``'-3.5999999999999996'`` and not ``'-3.6'``, and a test that
wrote the rounded digits would pass for the wrong reason and fail on a changed step size.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pytest
from cosy.core.tree import Tree
from grakel.kernels import VertexHistogram

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_kernels import (
    as_DAMG,
    damg_kernel_1,
    damg_kernel_2,
    damg_kernel_3,
    hierarchical_tree,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_algebras import edgelist_algebra
from bayesian_optimization.kernels import WeisfeilerLehmanKernel
from bayesian_optimization.utils import indexed_label, to_grakel_graph, to_indexed_nx_digraph

# The layout origin of edgelist_learner, and the two steps the compositions take.
X0, Y0 = -5.5, -3.8
X1, X2 = X0 + 2.5, X0 + 2.5 + 2.5
Y1, Y2 = Y0 + 0.2, Y0 + 0.2 + 0.2


def symbols(graph: nx.DiGraph) -> dict[int, str]:
    """Return the node labels of a term graph, keyed by node id.

    Args:
        graph (nx.DiGraph): The term graph.

    Returns:
        dict[int, str]: node id -> the ``symbol`` attribute, which holds the *label*.
    """
    return {node: data["symbol"] for node, data in graph.nodes(data=True)}


# ---------------------------------------------------------------------------
# The graph of a term and its argument indices, the conversion that has a specification
# ---------------------------------------------------------------------------


def test_the_root_carries_no_argument_index():
    """A root is the argument of nothing, so it must not be comparable to a position that is one.

    ``indexed_label`` is the whole of the instruction that every node label is extended by the
    argument index of its position, and the root is the case that instruction does not cover.
    Giving it index 0 would make the term ``a`` label-equal to ``a`` sitting in first position, and
    a vertex histogram would then count them together.
    """
    assert indexed_label("f", None) == "f@r"
    assert indexed_label("g", 1) == "g@1"
    assert indexed_label("a", 0) != indexed_label("a", None)


def test_a_single_node_term_is_one_labeled_node():
    """The base case, and the only one that takes the leaf branch of the recursion."""
    graph, index_mapping = to_indexed_nx_digraph(Tree("a"))

    assert symbols(graph) == {0: "a@r"}
    assert graph.nodes[0]["leaf"] is True
    assert set(graph.edges()) == set()
    # The mapping carries the raw symbol, not the label.  The two differ, and the attribute named
    # "symbol" is the one holding the label.
    assert index_mapping == {0: "a"}


def test_the_argument_index_is_relative_to_the_parent():
    """``a`` under ``g`` is ``a@0`` wherever ``g`` sits: the index is a position, not a path."""
    left, mapping_left = to_indexed_nx_digraph(Tree("f", (Tree("g", (Tree("a"),)), Tree("b"))))
    right, _ = to_indexed_nx_digraph(Tree("f", (Tree("a"), Tree("g", (Tree("b"),)))))

    assert symbols(left) == {0: "f@r", 1: "g@0", 2: "a@0", 3: "b@1"}
    assert set(left.edges()) == {(0, 1), (1, 2), (0, 3)}
    assert mapping_left == {0: "f", 1: "g", 2: "a", 3: "b"}

    assert symbols(right) == {0: "f@r", 1: "a@0", 2: "g@1", 3: "b@0"}
    assert set(right.edges()) == {(0, 1), (0, 2), (2, 3)}


def test_the_index_is_what_tells_two_argument_orders_apart():
    """The case the whole construction exists for.

    Without the index both terms are the labeled path ``b - f - g - a`` and are identical as
    labeled graphs.  No number of relabeling rounds separates them, and the kernel reports 1.0.
    The experiments run at ``h = 0``, where the label multiset is the only thing compared at all,
    so the index is not a refinement here.  It is the entire difference.
    """
    first = Tree("f", (Tree("g", (Tree("a"),)), Tree("b")))
    second = Tree("f", (Tree("b"), Tree("g", (Tree("a"),))))

    labels_first = set(symbols(to_indexed_nx_digraph(first)[0]).values())
    labels_second = set(symbols(to_indexed_nx_digraph(second)[0]).values())
    assert labels_first == {"f@r", "g@0", "a@0", "b@1"}
    assert labels_second == {"f@r", "b@0", "g@1", "a@0"}
    assert labels_first & labels_second == {"a@0", "f@r"}

    graphs = [next(to_grakel_graph(t)) for t in (first, second)]
    assert VertexHistogram(normalize=True).fit_transform(graphs)[0, 1] == pytest.approx(0.5)

    # The control: with the raw symbol as the label the two graphs are the same graph.
    raw = [
        _as_grakel_with_raw_symbols(t) for t in (first, second)
    ]
    assert VertexHistogram(normalize=True).fit_transform(raw)[0, 1] == pytest.approx(1.0)


def _as_grakel_with_raw_symbols(tree: Tree):
    """Convert a term the way the plain graph of a term would, without the argument index.

    The control for the test above.  It has to be built here rather than by switching a flag,
    because the production path has no such flag: the index is not optional.

    Args:
        tree (Tree): The term.

    Returns:
        list: The grakel triple.
    """
    graph, index_mapping = to_indexed_nx_digraph(tree)
    plain = nx.DiGraph()
    plain.add_nodes_from((node, {"symbol": str(index_mapping[node])}) for node in graph.nodes())
    plain.add_edges_from(graph.edges())
    from grakel.utils import graph_from_networkx

    return next(graph_from_networkx([plain.to_undirected()], node_labels_tag="symbol"))


def test_node_ids_are_preorder_and_gapless():
    """An asymmetric term is where the id arithmetic can be wrong without anything looking wrong.

    ``current_index += len(child_mapping)`` advances by the *subtree size*.  Two mutants:

    * advancing by ``1 + len(child.children)``, which counts one level, is accidentally right on
      ``f(g(a), b)`` and wrong here: ``b`` lands on id 3, ``nx.compose`` merges it with ``a``
      (the second graph's attributes win), a node disappears and an edge closes a cycle.  The
      symbol and edge assertions catch that.
    * advancing by ``len(child_mapping) + 1`` leaves a gap: the graph stays a correctly labeled
      tree, every kernel value is unchanged, and only the id set says so.  Hence the last two
      assertions, which are otherwise the sort a reviewer would call redundant.
    """
    term = Tree("f", (Tree("g", (Tree("h", (Tree("a"),)),)), Tree("b")))
    graph, index_mapping = to_indexed_nx_digraph(term)

    assert symbols(graph) == {0: "f@r", 1: "g@0", 2: "h@0", 3: "a@0", 4: "b@1"}
    assert set(graph.edges()) == {(0, 1), (1, 2), (2, 3), (0, 4)}
    assert sorted(graph.nodes()) == list(range(term.size))
    assert sorted(index_mapping) == list(range(term.size))


def test_the_conversion_starts_where_it_is_told_to():
    """``start_index`` and ``argument_index`` are public and the recursion's own hinges."""
    graph, index_mapping = to_indexed_nx_digraph(
        Tree("f", (Tree("a"),)), start_index=7, argument_index=2
    )
    assert symbols(graph) == {7: "f@2", 8: "a@0"}
    assert set(graph.edges()) == {(7, 8)}
    assert index_mapping == {7: "f", 8: "a"}


def test_to_grakel_graph_yields_one_undirected_graph():
    """The kernel gets an *undirected* graph, and it gets it out of a one-shot generator.

    Both halves are load-bearing and neither is visible in the signature.  Dropping
    ``.to_undirected()`` leaves the adjacency one-sided (``{0: {1, 3}, 1: {2}, 2: {}, 3: {}}``),
    which a vertex histogram survives silently and any edge-reading kernel does not.  And the
    return value is a generator whose element is a three-element list.  The annotation says
    ``grakel.Graph``, which it is not, and ``_materialize_graph`` in the kernel layer exists to
    absorb exactly that.  Pinned as today's behavior, not as a target.
    """
    result = to_grakel_graph(Tree("f", (Tree("g", (Tree("a"),)), Tree("b"))))
    assert hasattr(result, "__next__"), "a generator, whatever the annotation says"

    adjacency, labels, edge_labels = next(result)
    assert adjacency == {
        0: {1: 1.0, 3: 1.0},
        1: {0: 1.0, 2: 1.0},
        2: {1: 1.0},
        3: {0: 1.0},
    }
    assert labels == {0: "f@r", 1: "g@0", 2: "a@0", 3: "b@1"}
    assert edge_labels is None

    with pytest.raises(StopIteration):
        next(result)


# ---------------------------------------------------------------------------
# edgelist_algebra, the computation graph a term denotes
# ---------------------------------------------------------------------------


def layer(label: str, n_in: int, n_out: int) -> Tree:
    """Build a hierarchy-1 layer term.

    The ``_h1`` variants take three arguments where the full combinators take ten, and the seven
    the full ones drop are ignored by this algebra anyway.  ``as_hierarchical_damg(1)`` produces
    exactly this shape, so the test runs on the form the production path runs on.

    Args:
        label (str): The layer's label, standing in for a repository dataclass.
        n_in (int): Its input arity.
        n_out (int): Its output arity.

    Returns:
        Tree: The term.
    """
    return Tree("linear_layer_h1", (Tree(label), Tree(n_in), Tree(n_out)))


def name(label: str, n_in: int, n_out: int, position: tuple[float, float]) -> str:
    """Return the node name the algebra gives a layer.

    Args:
        label (str): The layer's label.
        n_in (int): Its input arity.
        n_out (int): Its output arity.
        position (tuple[float, float]): The layout coordinate it was interpreted at.

    Returns:
        str: ``str((label, n_in, n_out)) + str(position)``, carrying the ``)(`` seam the relabel
            sits on.
    """
    return str((label, n_in, n_out)) + str(position)


def parallel_then_sequential() -> Tree:
    """Build ``(A || B) ; C``: two layers side by side feeding one.

    The smallest term that exercises both compositions, so both coordinate shifts and the
    edge-list ordering of both helpers appear in one value.

    Returns:
        Tree: The term, wrapped in a learner.
    """
    row = Tree("beside_cons_h1", (
        Tree(2), Tree(1), Tree(1),          # i, i1, i2
        Tree(2), Tree(1), Tree(1),          # o, o1, o2
        layer("A", 1, 1),
        Tree("beside_singleton_h1", (Tree(1), Tree(1), layer("B", 1, 1))),
    ))
    model = Tree("before_cons_h1", (
        Tree(2), Tree(2), Tree(1),          # i, j, o
        row,
        Tree("before_singleton_h1", (
            Tree(2), Tree(1),
            Tree("beside_singleton_h1", (Tree(2), Tree(1), layer("C", 2, 1))),
        )),
    ))
    return Tree("learner_h1", (
        Tree(2), Tree(1), Tree(1),          # i, o, epochs
        Tree("loss"), Tree("optimizer"), Tree("scheduler"), model,
    ))


def test_the_step_size_is_not_a_round_number():
    """The guard for every coordinate below: these are float sums, and they read as such.

    Without this, a test written with ``-3.6`` would silently absorb a change of the step from 0.2
    to 0.25.  The coordinates end up inside node names, so the comparison is over ``repr(float)``.
    """
    assert str(Y1) == "-3.5999999999999996"
    assert Y1 != -3.6
    assert str(Y2) == "-3.3999999999999995"
    assert (str(X1), str(X2)) == ("-3.0", "-0.5")


def test_parallel_shifts_the_second_branch_down_and_sequential_shifts_right():
    """The layout is the identity of a node, so the two shifts are what keeps nodes apart.

    ``_beside_edgelists`` moves the right branch by +0.2 in y and leaves the left one alone.
    ``_before_edgelists`` moves the second stage by +2.5 in x, reading the unshifted y, so a
    parallel shift never leaks into the next stage.  Swapping the two axes, or letting the shift
    accumulate, would collapse two layers onto one name and silently merge them into one node.
    """
    edges = parallel_then_sequential().interpret(edgelist_algebra(False))
    node_a = name("A", 1, 1, (X0, Y0))
    node_b = name("B", 1, 1, (X0, Y1))
    node_c = name("C", 2, 1, (X1, Y0))

    assert edges == [
        (node_a, node_c),
        (node_b, node_c),
        ("input", node_a),
        ("input", node_b),
        (node_c, "loss"),
        ("loss", "optimizer"),
        # The schedule is its own node between optimizer and epochs.  The graph is what the graph
        # kernels read, so a term's schedule has to be visible in it.
        ("optimizer", "scheduler"),
        ("scheduler", "epochs(1)"),
    ]


def test_the_edge_list_comes_out_back_to_front():
    """The order is part of the value, and it is reverse-topological.

    ``_before_edgelists`` returns ``second[0] + first[0]``, the later stage's edges first, and
    ``_beside_edgelists`` returns ``left[0] + right[0]``.  Nothing downstream depends on the order
    today, since ``nx`` sorts it away, which is precisely why an accidental change would go
    unnoticed until something did.
    """
    edges = parallel_then_sequential().interpret(edgelist_algebra(False))
    targets = [target for _source, target in edges]
    assert targets.index(name("C", 2, 1, (X1, Y0))) < targets.index(name("A", 1, 1, (X0, Y0)))


def test_a_layer_with_several_outputs_produces_a_parallel_edge():
    """One layer feeding another twice is two edges in the edge list, and one in the graph.

    Reachable in the real repository, since ``sum`` and ``product`` take several inputs and
    ``copy`` produces several outputs.  The edge list keeps the multiplicity, because it models
    the data flow at the width of the tensors and that is what it is for.

    Where the multiplicity dies is what this test records.  It used to survive ``MultiDiGraph``
    and ``to_undirected`` and die at the grakel boundary, which keeps one edge of weight 1.0
    whatever arrived.  It now dies one step earlier, in ``as_DAMG``, which deduplicates before
    building the graph.

    Nothing downstream can tell, because the boundary discarded it anyway.
    ``test_edge_multiplicity_never_reaches_grakel`` asserts that premise, and
    ``test_deduplication_leaves_the_graph_of_a_real_term_alone`` asserts that nothing else goes
    with it.  What the earlier death buys is the clock, since a layer contributes one tuple per
    input feature and the compositions concatenate what they are handed.

    The last two assertions are the ones that matter for the kernel.  The middle one is the change.
    """
    model = Tree("before_cons_h1", (
        Tree(1), Tree(2), Tree(1),
        Tree("beside_singleton_h1", (Tree(1), Tree(2), layer("D", 1, 2))),
        Tree("before_singleton_h1", (
            Tree(2), Tree(1),
            Tree("beside_singleton_h1", (Tree(2), Tree(1), layer("E", 2, 1))),
        )),
    ))
    term = Tree("learner_h1", (
        Tree(1), Tree(1), Tree(1), Tree("loss"), Tree("optimizer"), Tree("scheduler"), model,
    ))
    node_d = name("D", 1, 2, (X0, Y0))
    node_e = name("E", 2, 1, (X1, Y0))

    # The algebra is unchanged: the width is still in the edge list, for whoever needs it.
    edges = term.interpret(edgelist_algebra(False))
    assert edges.count((node_d, node_e)) == 2

    # The graph keeps one edge per pair, since that is all the kernel can read.
    _graphs, multigraph, _positions, _relabel = as_DAMG(term, verbose=True)
    assert sorted(multigraph[node_d][node_e]) == [0]
    assert sorted(multigraph.to_undirected()[node_d][node_e]) == [0]

    # Unchanged, and the reason the change is invisible: this was 1.0 before it too.
    adjacency, _labels, _edge_labels = next(as_DAMG(term))
    assert adjacency[node_d][node_e] == 1.0


# ---------------------------------------------------------------------------
# as_DAMG, the relabel and what its regular expression does and does not reach
# ---------------------------------------------------------------------------

def strip(node_name: str, position) -> str:
    """Remove a node's layout coordinate the way ``as_DAMG`` does.

    Not a pattern over the name: the algebra returns the coordinate that was appended, so the
    suffix to remove is the one that was added.  ``removesuffix`` and not a slice, because the
    frame nodes appear in the position map without carrying a coordinate in their name.

    Args:
        node_name (str): The name.
        position: The coordinate the algebra recorded for it.

    Returns:
        str: The label.
    """
    return node_name.removesuffix(str(position))


def test_the_relabel_removes_the_coordinate_and_keeps_the_layer():
    """What the relabel is for: two instances of one layer must share a label, not a node.

    The identity of a node stays its full name with the coordinate, and only the ``symbol`` the
    kernel reads collapses.  That separation is the point: the graph keeps its shape while the
    histogram stops counting positions.
    """
    assert strip(name("A", 1, 1, (X0, Y0)), (X0, Y0)) == "('A', 1, 1)"
    assert strip(name("A", 1, 1, (X0, Y1)), (X0, Y1)) == "('A', 1, 1)"

    _graphs, multigraph, _positions, relabel = as_DAMG(parallel_then_sequential(), verbose=True)
    assert relabel[name("A", 1, 1, (X0, Y0))] == "('A', 1, 1)"
    assert len(multigraph.nodes()) == 8
    # Eight distinct labels, although A and B have the same arity: their names differ, which is
    # what hierarchy 1 keeps and what hierarchy 2 throws away.  Eight rather than seven because
    # the schedule is its own node in the chain.
    assert len(set(relabel.values())) == 8


def test_a_repr_with_its_own_brackets_survives_the_relabel():
    """Real labels are dataclass ``repr``s full of brackets, and only the trailing seam may go.

    The pattern needs a literal ``)`` immediately before ``(``, and needs both coordinates to
    carry a decimal point.  Inside a ``repr`` the tuples are integer tuples (``kernel_size=(3, 3)``)
    and they follow a ``=`` rather than a ``)``, which is the only reason a regular expression was
    safe here at all.
    """
    # A dataclass-like object, not a string: the production labels are repository dataclasses, and
    # ``str`` of a tuple quotes a string while leaving an object's repr as it is.  Using a string
    # here would test a name the algebra never builds.
    @dataclass(frozen=True)
    class Linear:
        """Stands in for a repository layer, repr and all."""

        in_features: int
        out_features: int
        bias: bool

        def __repr__(self) -> str:
            """Render the way a repository dataclass does.

            Returns:
                str: The rendering, brackets and all.
            """
            return (
                f"CNNrepository.Linear(in_features={self.in_features}, "
                f"out_features={self.out_features}, bias={self.bias})"
            )

    label = Linear(8, 4, True)
    term = Tree("learner_h1", (
        Tree(1), Tree(1), Tree(1), Tree("loss"), Tree("optimizer"), Tree("scheduler"),
        Tree("before_singleton_h1", (
            Tree(1), Tree(1),
            Tree("beside_singleton_h1", (Tree(1), Tree(1), layer(label, 8, 4))),
        )),
    ))
    _graphs, _multigraph, _positions, relabel = as_DAMG(term, verbose=True)
    expected = f"({label!r}, 8, 4)"
    assert "(" in repr(label), "the case is about a repr carrying its own brackets"
    assert expected in set(relabel.values())
    assert not any(str((X0, Y0)) in value for value in relabel.values())


def test_the_frame_nodes_keep_their_names():
    """``input``, the loss, the optimizer and the epoch marker carry no coordinate.

    They must not be touched, since they are the same node in every architecture, which is what
    makes them a frame.
    """
    _graphs, _multigraph, positions, relabel = as_DAMG(parallel_then_sequential(), verbose=True)
    for unchanged in ("input", "loss", "optimizer", "epochs(1)"):
        assert unchanged in positions, "the frame node must be in the map, or the case is empty"
        assert relabel[unchanged] == unchanged


def node_h(n_in: int, n_out: int) -> Tree:
    """Build a hierarchy-2/3 node term.

    From hierarchy 2 the layer type is folded away and every node is simply ``node``.

    Args:
        n_in (int): Its input arity.
        n_out (int): Its output arity.

    Returns:
        Tree: The term.
    """
    return Tree("node", (Tree(n_in), Tree(n_out)))


def folded_row_and_chain() -> tuple[Tree, Tree]:
    """Build ``A || B`` and ``A ; B`` over two hierarchy-2/3 nodes.

    Two architectures over the same two nodes, differing only in how they compose, which is the
    only thing hierarchy 2 and 3 can still see, since the layer type is gone.

    Returns:
        tuple[Tree, Tree]: The parallel one and the sequential one.
    """
    def alone(inner: Tree) -> Tree:
        return Tree("beside_singleton_h3", (Tree(1), Tree(1), inner))

    def learner(model: Tree) -> Tree:
        return Tree("learner_h2", (Tree(1), Tree("loss"), Tree("optimizer"), Tree("scheduler"), model))

    row = Tree("beside_cons_h3", (Tree(2), Tree(1), Tree(2), node_h(1, 1), alone(node_h(1, 1))))
    parallel = learner(Tree("before_singleton_h1", (Tree(2), Tree(2), row)))
    sequential = learner(Tree("before_cons_h1", (
        Tree(1), Tree(1), Tree(1), alone(node_h(1, 1)),
        Tree("before_singleton_h1", (Tree(1), Tree(1), alone(node_h(1, 1)))),
    )))
    return parallel, sequential


def test_the_relabel_reaches_every_hierarchy_level():
    """The coordinate goes, whatever the level names its nodes.

    Hierarchy 2 and 3 call a node ``"node" + str(id)``, with no ``)`` before the coordinate.  The
    relabel used to be a regular expression aimed at the hierarchy 1 shape and matched none of
    these, so from hierarchy 2 the labels were the layout coordinates, and ``damg_kernel_2`` and
    ``damg_kernel_3`` compared where a node had been drawn.  On the two terms below, ``A || B``
    carried ``node(-5.5, -3.8)`` and ``node(-5.5, -3.5999999999999996)`` while ``A ; B`` carried
    ``node(-5.5, -3.8)`` and ``node(-3.0, -3.8)``, so the one label they shared was the one node
    both happened to place at the origin.

    The algebra returns the map from node name to coordinate, so the suffix to remove is known
    rather than guessed, and it is known at every level.
    """
    parallel, sequential = folded_row_and_chain()
    for term in (parallel, sequential):
        _graphs, multigraph, _positions, relabel = as_DAMG(term, verbose=True)
        assert set(relabel.values()) == {"node", "input", "loss", "optimizer", "scheduler", "epochs(1)"}
        assert len(multigraph.nodes()) > len(set(relabel.values())), (
            "the nodes must stay apart even where their labels coincide"
        )


def test_a_frame_node_keeps_its_name_although_it_has_a_position():
    """The position map holds the frame nodes too, for plotting, and they carry no coordinate.

    That is why the strip is a suffix removal rather than a slice: ``input`` appears in the map
    with the position of the first layer, and cutting that many characters off it would leave a
    fragment.  A no-op is exactly right here.
    """
    parallel, _sequential = folded_row_and_chain()
    _graphs, _multigraph, positions, relabel = as_DAMG(parallel, verbose=True)
    assert "input" in positions
    assert relabel["input"] == "input"
    assert relabel["epochs(1)"] == "epochs(1)"


def test_from_hierarchy_two_the_kernel_needs_a_relabeling_round():
    """What the repaired relabel costs, and where it has to be paid.

    With every node called ``node``, a vertex histogram counts nodes and nothing else, so at
    ``h = 0`` two architectures over the same nodes are identical, however they compose.  The
    composition reaches the kernel through a relabeling round, which is why ``damg_kernel_2`` and
    ``damg_kernel_3`` run at ``h = 1`` while ``damg_kernel_1``, whose labels still carry the layer,
    runs at ``h = 0``.
    """
    parallel, sequential = folded_row_and_chain()
    pair = [parallel, sequential]

    at_zero = WeisfeilerLehmanKernel(h=0, normalize=True, to_grakel_graph=as_DAMG)(pair)[0, 1]
    at_one = WeisfeilerLehmanKernel(h=1, normalize=True, to_grakel_graph=as_DAMG)(pair)[0, 1]

    assert at_zero == pytest.approx(1.0), "a histogram over one label cannot see the composition"
    assert at_one < 1.0 - 1e-9


# ---------------------------------------------------------------------------
# The fold as a parameter, the case a hand-built term cannot reach
# ---------------------------------------------------------------------------

_IGNORED = Tree(0)  # a parameter edgelist_algebra does not read


def full_layer(label: str, n_in: int, n_out: int) -> Tree:
    """Build a layer in the *unfolded* form, the one synthesis produces.

    The tests above build hierarchy-1 and hierarchy-2/3 terms by hand, which is convenient and
    leaves one thing untested: they never hand ``as_DAMG`` a fold, because they are already folded.
    The full combinators take seven to sixteen parameters the edge-list algebra ignores, and they
    are written out here so a fold has something to fold.

    Args:
        label (str): The layer's label.
        n_in (int): Its input arity.
        n_out (int): Its output arity.

    Returns:
        Tree: The term.
    """
    return Tree("linear_layer", (Tree(label), Tree(n_in), Tree(n_out)) + (_IGNORED,) * 7)


def full_chain() -> Tree:
    """Build ``A ; B`` in the unfolded form, wrapped in a learner.

    Returns:
        Tree: The term.
    """
    def alone(n_in: int, n_out: int, inner: Tree) -> Tree:
        return Tree("beside_singleton", (Tree(n_in), Tree(n_out), _IGNORED, _IGNORED, inner))

    model = Tree("before_cons", (
        Tree(1), Tree(1), Tree(1), _IGNORED, _IGNORED, _IGNORED, _IGNORED,
        alone(1, 1, full_layer("A", 1, 1)),
        Tree("before_singleton", (
            Tree(1), Tree(1), _IGNORED, _IGNORED, _IGNORED, alone(1, 1, full_layer("B", 1, 1)),
        )),
    ))
    return Tree("learner", (
        Tree(1), Tree(1), _IGNORED, _IGNORED, Tree(1), _IGNORED, _IGNORED, _IGNORED,
        Tree("mse_loss", (Tree("loss"),)),
        Tree("no_scheduler", (
            _IGNORED, _IGNORED, Tree("adam_optimizer", (Tree("optimizer"),)),
        )),
        model,
    ))


def test_the_fold_decides_what_a_node_is_called():
    """``as_DAMG`` applies the fold itself, and the level decides the label alphabet.

    This is the change the parameter exists for, and no other test reaches it: the ones above hand
    in already-folded terms, so they run on the default ``no_fold`` throughout.  With the fold
    ignored, all three levels would read the same graph and levels 2 and 3 would collapse onto
    level 1, which is exactly what they must not do.
    """
    term = full_chain()

    at_one = as_DAMG(term, fold=hierarchical_tree(1), verbose=True)[3]
    at_two = as_DAMG(term, fold=hierarchical_tree(2), verbose=True)[3]
    at_three = as_DAMG(term, fold=hierarchical_tree(3), verbose=True)[3]

    # Level 1 keeps the layer with its dimensions.  From level 2 every node is just a node.
    assert {label for label in at_one.values() if label.startswith("(")} == {
        "('A', 1, 1)", "('B', 1, 1)",
    }
    for higher in (at_two, at_three):
        assert set(higher.values()) == {"node", "input", "loss", "optimizer", "scheduler", "epochs(1)"}

    assert set(at_one.values()) != set(at_two.values()), "the fold must change what is read"


def test_the_hierarchy_kernels_do_not_collapse_onto_each_other():
    """Three levels, three kernels, and they must not agree by accident.

    The sum ``damg_kernel_1 + damg_kernel_2 + damg_kernel_3`` is only worth its three terms if the
    levels see different things.  With the fold dropped they would all read level 0 and the sum
    would be three copies of one kernel.
    """
    def other_chain() -> Tree:
        def alone(inner: Tree) -> Tree:
            return Tree("beside_singleton", (Tree(1), Tree(1), _IGNORED, _IGNORED, inner))

        model = Tree("before_singleton", (
            Tree(1), Tree(1), _IGNORED, _IGNORED, _IGNORED, alone(full_layer("A", 1, 1)),
        ))
        return Tree("learner", (
            Tree(1), Tree(1), _IGNORED, _IGNORED, Tree(1), _IGNORED, _IGNORED, _IGNORED,
            Tree("mse_loss", (Tree("loss"),)),
            Tree("no_scheduler", (
                _IGNORED, _IGNORED, Tree("adam_optimizer", (Tree("optimizer"),)),
            )),
            model,
        ))

    pair = [full_chain(), other_chain()]
    values = [kernel(pair)[0, 1] for kernel in (damg_kernel_1, damg_kernel_2, damg_kernel_3)]

    assert all(np.isfinite(v) for v in values)
    assert len(set(np.round(values, 10))) > 1, f"the three levels agree: {values}"


# ---------------------------------------------------------------------------
# The multiplicity the kernel cannot read
# ---------------------------------------------------------------------------

def test_edge_multiplicity_never_reaches_grakel():
    """The premise the deduplication in ``as_DAMG`` rests on, asserted rather than assumed.

    ``graph_from_networkx`` hands grakel an adjacency dictionary, so parallel edges arrive as one
    entry of weight 1.0 however many there were.  If a future grakel ever changed that, or a
    switch to a weighted conversion did, deduplicating before the graph would silently change
    every kernel value in this project, and this test is what would say so first.
    """
    import networkx as nx
    import numpy as np
    from grakel.kernels import VertexHistogram, WeisfeilerLehman
    from grakel.utils import graph_from_networkx

    def build(edges):
        graph = nx.MultiDiGraph()
        graph.add_edges_from(edges)
        for node in graph.nodes():
            graph.nodes[node]["symbol"] = node
        return graph.to_undirected()

    same_shape_different_widths = [
        build([("A", "C"), ("B", "C")]),
        build([("A", "C")] * 5 + [("B", "C")]),
        build([("A", "C")] * 64 + [("B", "C")] * 16),
    ]

    adjacencies = [
        graph[0] if isinstance(graph, (list, tuple)) else graph
        for graph in graph_from_networkx(same_shape_different_widths, node_labels_tag="symbol")
    ]
    assert adjacencies[0] == adjacencies[1] == adjacencies[2], (
        "grakel now distinguishes edge multiplicities; as_DAMG must stop deduplicating"
    )

    for rounds in (1, 2, 3):
        gram = WeisfeilerLehman(
            n_iter=rounds, base_graph_kernel=VertexHistogram, normalize=True
        ).fit_transform(
            list(graph_from_networkx(same_shape_different_widths, node_labels_tag="symbol"))
        )
        assert np.allclose(gram, 1.0), (
            f"widths 1:1, 5:1 and 64:16 became distinguishable at n_iter={rounds}"
        )


def test_deduplication_leaves_the_graph_of_a_real_term_alone():
    """Nodes, edge set and labels of a term with a wide layer, with and without the copies.

    The unit above shows grakel cannot see the multiplicity.  This one shows ``as_DAMG`` removes
    nothing else while removing it.  It uses the same term as the parallel-edge test, whose
    ``copy`` layer is exactly the construct that produces the duplicates.
    """
    import networkx as nx

    model = Tree("before_cons_h1", (
        Tree(1), Tree(2), Tree(1),
        Tree("beside_singleton_h1", (Tree(1), Tree(2), layer("D", 1, 2))),
        Tree("before_singleton_h1", (
            Tree(2), Tree(1),
            Tree("beside_singleton_h1", (Tree(2), Tree(1), layer("E", 2, 1))),
        )),
    ))
    term = Tree("learner_h1", (
        Tree(1), Tree(1), Tree(1), Tree("loss"), Tree("optimizer"), Tree("scheduler"), model,
    ))

    edgelist, positions = term.interpret(edgelist_algebra(True))
    with_copies = nx.MultiDiGraph()
    with_copies.add_edges_from(edgelist)

    _graphs, deduplicated, _positions, _relabel = as_DAMG(term, verbose=True)

    assert set(deduplicated.nodes()) == set(with_copies.nodes())
    assert {(u, v) for u, v, _k in deduplicated.edges(keys=True)} == {
        (u, v) for u, v, _k in with_copies.edges(keys=True)
    }
    assert deduplicated.number_of_edges() < with_copies.number_of_edges(), (
        "this term is supposed to contain a duplicate; without one the test proves nothing"
    )
    assert positions == _positions
