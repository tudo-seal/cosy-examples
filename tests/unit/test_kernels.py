from __future__ import annotations

import numpy as np
import pytest
from cosy.core.tree import Tree
from grakel.kernels import VertexHistogram
from sklearn.gaussian_process import GaussianProcessRegressor

from bayesian_optimization.kernels.graph_kernel import (
    HierarchicalWLKernel,
    WeisfeilerLehmanKernel,
)
from bayesian_optimization.kernels.tree_kernel import (
    OrderedRootedSubtreeKernel,
    SubsetTreeKernel,
)
from bayesian_optimization.utils import to_grakel_graph


def _truncate(depth: int):
    """Return a fold that replaces everything below ``depth`` with a single node.

    Cutting a term at a fixed depth is the same coarsening as a repository's own granularity
    levels, as long as the symbols of a level sit at exactly that depth. The hierarchical
    kernel takes the truncation as a parameter precisely because what a level means is the
    repository's business, not the kernel's.
    """

    def fold(tree: Tree) -> Tree:
        if depth <= 0:
            return Tree(tree.root, ())
        return Tree(tree.root, tuple(_truncate(depth - 1)(child) for child in tree.children))

    return fold


KERNEL_FACTORIES = [
    lambda: OrderedRootedSubtreeKernel(normalize=True),
    lambda: OrderedRootedSubtreeKernel(normalize=False),
    lambda: SubsetTreeKernel(normalize=True),
    lambda: SubsetTreeKernel(normalize=False),
    lambda: WeisfeilerLehmanKernel(h=0),
    lambda: WeisfeilerLehmanKernel(h=2),
    lambda: HierarchicalWLKernel(truncations=[_truncate(1), _truncate(2)], h=1),
    lambda: HierarchicalWLKernel(
        truncations=[_truncate(1), _truncate(2)], h=1, normalize=False
    ),
]

KERNEL_IDS = [
    "subtree_normalized",
    "subtree_unnormalized",
    "subset_tree_normalized",
    "subset_tree_unnormalized",
    "wl_h0",
    "wl_h2",
    "hwl_normalized",
    "hwl_unnormalized",
]


@pytest.mark.parametrize("kernel_factory", KERNEL_FACTORIES, ids=KERNEL_IDS)
def test_kernel_symmetry(kernel_factory, tree_corpus):
    kernel = kernel_factory()
    K = kernel(tree_corpus)
    assert np.allclose(K, K.T, atol=1e-8), "Kernel matrix is not symmetric"


@pytest.mark.parametrize("kernel_factory", KERNEL_FACTORIES, ids=KERNEL_IDS)
def test_kernel_positive_semidefinite(kernel_factory, tree_corpus):
    """Every matrix of pairwise kernel values must be symmetric and positive semidefinite."""
    kernel = kernel_factory()
    K = kernel(tree_corpus)
    eigvals = np.linalg.eigvalsh(K)
    assert np.all(eigvals >= -1e-8), f"Kernel is not PSD; min eigenvalue = {eigvals.min():.2e}"


@pytest.mark.parametrize("kernel_factory", KERNEL_FACTORIES, ids=KERNEL_IDS)
def test_kernel_diagonal_correct(kernel_factory, tree_corpus):
    kernel = kernel_factory()
    K = kernel(tree_corpus)
    diag_direct = kernel.diag(tree_corpus)
    assert np.allclose(diag_direct, np.diag(K), atol=1e-8)


@pytest.mark.parametrize("kernel_factory", KERNEL_FACTORIES, ids=KERNEL_IDS)
def test_diag_does_not_build_the_full_matrix(kernel_factory, tree_corpus, monkeypatch):
    """``diag`` runs on every acquisition evaluation and must not cost a full matrix.

    sklearn asks for it on each ``predict(return_std=True)``, so a ``diag`` that computes the
    whole Gram matrix and keeps the diagonal turns every candidate scoring into an ``n x n``
    kernel evaluation.
    """
    kernel = kernel_factory()

    def refuse(*_args, **_kwargs):
        raise AssertionError("diag() built the full kernel matrix")

    monkeypatch.setattr(type(kernel), "_kernel_matrix", refuse, raising=True)
    values = kernel.diag(tree_corpus)
    assert values.shape == (len(tree_corpus),)
    assert np.all(np.isfinite(values))


@pytest.mark.parametrize("kernel_factory", KERNEL_FACTORIES, ids=KERNEL_IDS)
def test_kernel_self_vs_cross_consistency(kernel_factory, tree_corpus):
    """K(X) must equal K(X, X)."""
    kernel = kernel_factory()
    K_self = kernel(tree_corpus)
    K_cross = kernel(tree_corpus, tree_corpus)
    assert np.allclose(K_self, K_cross, atol=1e-8)


@pytest.mark.parametrize("kernel_factory", KERNEL_FACTORIES, ids=KERNEL_IDS)
def test_kernel_works_with_gp(kernel_factory, tree_corpus):
    kernel = kernel_factory()
    X = np.asarray(tree_corpus, dtype=object)
    y = np.array([float(i) * 0.5 for i in range(len(tree_corpus))])
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=False, optimizer=None)
    gp.fit(X, y)
    preds = gp.predict(X)
    assert preds.shape == (len(tree_corpus),)
    assert np.all(np.isfinite(preds))


def test_kernel_caching_consistency(tree_corpus):
    """Two separate kernel instances should produce the same matrix."""
    k1 = OrderedRootedSubtreeKernel(normalize=True)
    k2 = OrderedRootedSubtreeKernel(normalize=True)
    assert np.allclose(k1(tree_corpus), k2(tree_corpus), atol=1e-10)


# ---------------------------------------------------------------------------
# The properties the kernel definitions require of these implementations
# ---------------------------------------------------------------------------


def test_the_wl_kernel_sees_the_argument_order():
    """``f(a, b)`` and ``f(b, a)`` are different terms and must not score as identical.

    The graph of a term labels each node with the symbol at its position and links every
    position to its immediate subterms, which forgets the order among the arguments. Extending
    every node label with the argument index of its position restores it. Before that the order
    sat on the *edge*, which a vertex histogram never reads, so the two terms below had the same
    graph and the kernel could not tell any permutation of arguments apart.
    """
    first = Tree("f", (Tree("a"), Tree("b")))
    second = Tree("f", (Tree("b"), Tree("a")))

    kernel = WeisfeilerLehmanKernel(h=1, normalize=True)
    K = kernel([first, second])

    assert K[0, 1] < 1.0 - 1e-9, "the kernel cannot distinguish the two argument orders"


def test_h_is_the_number_of_relabeling_rounds():
    """``h`` counts rounds the way the kernel is defined, not the way grakel counts them.

    The Weisfeiler-Lehman kernel sums the base kernel over the rounds ``i = 0..h``, so ``h``
    rounds means ``h + 1`` summands. grakel runs ``n_iter + 1`` levels
    (``weisfeiler_lehman.py:109``: ``_n_iter = n_iter + 1``), so ``n_iter = h``, and ``h = 0``,
    the single summand made of the initial labels, is not expressible through grakel's WL at
    all, which rejects ``n_iter <= 0``. It is the base kernel by itself.

    This test states the round count as a *measurement* rather than as a parameter value,
    because the parameter value was where the off-by-one hid. With an unnormalized vertex
    histogram, the diagonal of a four-node term is the number of levels times four.
    """
    term = Tree("f", (Tree("g", (Tree("a"),)), Tree("b")))
    levels = {}
    for h in (0, 1, 2, 3):
        built = WeisfeilerLehmanKernel(h=h, normalize=False)._build_kernel(h)
        graphs = [next(to_grakel_graph(term))]
        levels[h] = built.fit_transform(graphs)[0, 0] / term.size

    assert levels == {0: 1.0, 1: 2.0, 2: 3.0, 3: 4.0}, (
        f"h rounds must give h+1 summands, measured {levels}"
    )

    with pytest.raises(ValueError, match="non-negative"):
        WeisfeilerLehmanKernel(h=-1)._build_kernel(-1)


def test_h_zero_is_the_vertex_histogram_itself():
    """``h = 0`` compares the initial labels and nothing else.

    The sum over the rounds has one summand there, the base kernel on the labels as they come,
    and that is the configuration the CNN experiments were meant to run under. They did not.
    ``n_iter = h + 1`` handed grakel a kernel that performs one relabeling round, so every
    ``h = 0`` kernel in this repository was an ``h = 1`` kernel. The two disagree measurably,
    0.5 against 0.25 on the pair below, so this is not a naming question.
    """
    first = Tree("f", (Tree("g", (Tree("a"),)), Tree("b")))
    second = Tree("f", (Tree("b"), Tree("g", (Tree("a"),))))
    graphs = [next(to_grakel_graph(t)) for t in (first, second)]

    expected = VertexHistogram(normalize=True).fit_transform(graphs)[0, 1]
    assert expected == pytest.approx(0.5)
    assert WeisfeilerLehmanKernel(h=0, normalize=True)([first, second])[0, 1] == pytest.approx(
        expected
    )


def test_more_rounds_can_separate_what_fewer_cannot(tree_corpus):
    """A round count that changes nothing would make the parameter meaningless."""
    without = WeisfeilerLehmanKernel(h=0)(tree_corpus)
    with_rounds = WeisfeilerLehmanKernel(h=2)(tree_corpus)
    assert not np.allclose(without, with_rounds, atol=1e-8)


def test_hierarchical_weights_are_fittable_hyperparameters():
    """The level weights of the hierarchical kernel are what the marginal likelihood adjusts.

    That kernel is a weighted sum of one plain Weisfeiler-Lehman kernel per granularity level,
    and the weights say how much each level contributes. Declaring them as hyperparameters is
    the reason the kernel exists as a class instead of as a sum of products with constant
    kernels. The sum widens what the model selection can adjust, from one round count to a
    weight per level. A kernel whose ``theta`` is empty is one sklearn optimizes without saying
    that it did nothing.
    """
    kernel = HierarchicalWLKernel(truncations=[_truncate(1), _truncate(2), _truncate(3)])

    assert kernel.theta.size == 3
    assert kernel.hyperparameter_lambdas.n_elements == 3

    moved = kernel.clone_with_theta(np.log([0.5, 1.0, 2.0]))
    assert np.allclose(moved.lambdas, [0.5, 1.0, 2.0])


def test_hierarchical_weights_change_the_matrix(tree_corpus):
    """A weight that does not reach the Gram matrix is not a hyperparameter of anything."""
    levels = [_truncate(1), _truncate(3)]
    first = HierarchicalWLKernel(truncations=levels, lambdas=[1.0, 0.0], normalize=False)
    second = HierarchicalWLKernel(truncations=levels, lambdas=[0.0, 1.0], normalize=False)
    assert not np.allclose(first(tree_corpus), second(tree_corpus), atol=1e-8)


def test_hierarchical_kernel_rejects_a_weight_per_nothing():
    with pytest.raises(ValueError, match="one weight per level"):
        HierarchicalWLKernel(truncations=[_truncate(1)], lambdas=[0.5, 0.5])
    with pytest.raises(ValueError, match="at least one level"):
        HierarchicalWLKernel(truncations=[])


def test_the_gradient_is_available_where_there_are_hyperparameters(tree_corpus):
    """sklearn needs a gradient to fit, and the base class approximates it numerically."""
    kernel = HierarchicalWLKernel(truncations=[_truncate(1), _truncate(2)])
    K, gradient = kernel(tree_corpus, eval_gradient=True)
    assert gradient.shape == (len(tree_corpus), len(tree_corpus), 2)
    assert np.all(np.isfinite(gradient))
    assert np.allclose(K, K.T, atol=1e-8)


def test_the_subset_tree_kernel_compares_productions_not_symbols():
    """The unit the subset-tree kernel compares is the production, not the symbol alone.

    A production is a symbol together with the symbols of its children, so two terms with the
    same root symbol but different children share no fragment there.
    """
    kernel = SubsetTreeKernel(normalize=False)
    shared = Tree("f", (Tree("a"), Tree("b")))
    same_symbol_other_children = Tree("f", (Tree("a"), Tree("c")))

    assert kernel([shared], [same_symbol_other_children])[0, 0] == 0.0
    assert kernel([shared], [shared])[0, 0] > 0.0


def test_the_two_tree_kernels_disagree_about_fragments():
    """Counting complete subtrees and counting subset trees are different questions.

    If they always agreed there would be no reason to offer both. Validity is a formal property
    both of them have, fit is not.
    """
    corpus = [
        Tree("f", (Tree("g", (Tree("a"),)), Tree("b"))),
        Tree("f", (Tree("g", (Tree("c"),)), Tree("b"))),
    ]
    subtree = OrderedRootedSubtreeKernel(normalize=True)(corpus)
    subset = SubsetTreeKernel(normalize=True)(corpus)
    assert not np.allclose(subtree, subset, atol=1e-8)


def test_a_leaf_roots_no_subset_tree():
    """Zero self-similarity cannot be normalized to one, and must not be faked into it."""
    kernel = SubsetTreeKernel(normalize=True)
    leaf = Tree("a")
    assert kernel.diag([leaf])[0] == 0.0
    assert kernel([leaf])[0, 0] == 0.0


def test_model_selection_moves_the_level_weights(tree_corpus):
    """End to end, the marginal likelihood is what sets the weights of the hierarchy levels.

    This is the point of declaring them as hyperparameters rather than fixing them by hand, and
    the only way to see that the numerical gradient, the bounds and sklearn's optimizer actually
    line up.
    """
    kernel = HierarchicalWLKernel(
        truncations=[_truncate(1), _truncate(3)], lambdas=[0.5, 0.5]
    )
    X = np.asarray(tree_corpus, dtype=object)
    y = np.array([float(len(str(tree))) for tree in tree_corpus])

    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=1e-6, normalize_y=True,
        optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0, random_state=0,
    )
    gp.fit(X, y)

    assert not np.allclose(gp.kernel_.lambdas, kernel.lambdas, atol=1e-6), (
        "the marginal likelihood left the level weights where they started"
    )
    assert np.all(np.isfinite(gp.kernel_.theta))


def test_h_zero_survives_a_dataset_that_makes_grakel_choose_sparse():
    """``h = 0`` must also hold up on the dataset a real run has, not just on two graphs.

    ``h = 0`` is the only path that hands grakel's *base* kernel to sklearn directly, and
    ``VertexHistogram`` decides at **fit** time whether to hold its features sparsely. It does
    when the fitted graphs share few labels, which is what a dataset of different architectures
    looks like. On that path the features are integer counts, so the kernel matrix is an integer
    array, and grakel normalizes it with an **in-place** division, which numpy refuses.

    Nothing smaller shows it. With a single graph in the fit the density is 1.0 and the dense
    path is taken, and with two similar graphs it is taken as well. It needs several graphs whose
    labels are mostly disjoint, and the cross evaluation (``X`` against ``Y``) that a GP does
    when it predicts.
    """
    def chain(prefix: str, length: int) -> Tree:
        term = Tree(f"{prefix}{length - 1}")
        for index in reversed(range(length - 1)):
            term = Tree(f"{prefix}{index}", (term,))
        return term

    train = [chain("a", 6), chain("b", 6), chain("c", 6)]
    query = [chain("d", 6)]

    kernel = WeisfeilerLehmanKernel(h=0, normalize=True)
    cross = kernel(query, train)

    assert cross.shape == (1, 3)
    assert np.all(np.isfinite(cross))
    # Normalized, so nothing may exceed a self-similarity of one.
    assert np.all(cross <= 1.0 + 1e-9)



def test_the_cache_keeps_the_translation_alive_so_its_identity_stays_meaningful():
    """The cache may not key on ``id()``, and this is the property that makes that safe.

    ``as_hierarchical_damg(level)`` returns a fresh closure per call. A caller that built one
    kernel per level without keeping the converters alive released each closure before making
    the next, and CPython handed out the **same id** three times, so the second and third
    kernels read the first one's matrix out of the cache. Measured before the fix on forty real
    terms, the Gram matrices of the DAMG hierarchies 1, 2 and 3 came out identical to every
    decimal, which is impossible for three different graph translations. The module's own
    comment forbade exactly this, and the code did it anyway.

    **Why this asserts a weak reference rather than two different matrices.** Reproducing the
    collision needs CPython to hand back the same address, and it does that only sometimes, so a
    test written that way passes with the defect in place whenever the allocator happens to
    move. What is deterministic is the fix's mechanism. The cache holds the callable itself, so
    the object cannot be collected while its entry lives, so its id cannot be reused. A cache
    that keys on ``id()`` keeps nothing alive, and the reference dies.
    """
    import gc
    import weakref

    from cosy.core.tree import Tree

    from bayesian_optimization.kernels.graph_kernel import WeisfeilerLehmanKernel
    from bayesian_optimization.utils import to_grakel_graph as to_graph

    terms = [Tree("f", (Tree("a"), Tree("b"))), Tree("g", (Tree("c"),))]

    def make_converter():
        """Return a fresh converter, so that dropping it can free its address."""
        def convert(tree):
            return to_graph(tree)
        return convert

    converter = make_converter()
    reference = weakref.ref(converter)
    kernel = WeisfeilerLehmanKernel(h=0, normalize=True, to_grakel_graph=converter)
    kernel(terms)
    del converter, kernel
    gc.collect()

    assert reference() is not None, (
        "the cache let the translation be collected, so a later closure can inherit its address "
        "and read a matrix computed for a different translation"
    )


def test_two_translations_alive_at_once_do_not_share_a_matrix():
    """The cache key has to *distinguish* translations, not merely hold them alive.

    Keeping the callables alive is what makes identity safe, and using them in the key is what
    makes it discriminating. One converter keeps the term, the other maps **every** term to the
    same graph, so their normalized matrices must differ. The second is all ones by
    construction.
    """
    import numpy as np
    from cosy.core.tree import Tree

    from bayesian_optimization.kernels.graph_kernel import WeisfeilerLehmanKernel
    from bayesian_optimization.utils import to_grakel_graph as to_graph

    terms = [
        Tree("f", (Tree("a"), Tree("b"))),
        Tree("g", (Tree("c"),)),
        Tree("h", (Tree("d"), Tree("e"), Tree("i"))),
    ]

    def keep(tree):
        """Convert the term as it is."""
        return to_graph(tree)

    def collapse(_tree):
        """Convert every term to one and the same graph."""
        return to_graph(Tree("x", ()))

    kept = np.asarray(
        WeisfeilerLehmanKernel(h=0, normalize=True, to_grakel_graph=keep)(terms), dtype=float
    )
    collapsed = np.asarray(
        WeisfeilerLehmanKernel(h=0, normalize=True, to_grakel_graph=collapse)(terms), dtype=float
    )

    assert np.allclose(collapsed, np.ones_like(collapsed))
    assert not np.allclose(kept, collapsed)
