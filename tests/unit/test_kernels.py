from __future__ import annotations

import time

import numpy as np
import pytest
from cosy.core.tree import Tree
from cosy.search.kernels import k_sst, k_st, normalized
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


def _chain(depth: int, leaf: str) -> Tree:
    """Return a unary chain of ``depth`` symbols above ``leaf``."""
    term: Tree = Tree(leaf)
    for index in reversed(range(depth)):
        term = Tree(f"c{index}", (term,))
    return term


def _subterm_corpus() -> list[Tree]:
    """Return one term per shape on which two counts of shared subterms could disagree.

    A single node is the smallest term there is, and the one term the subset-tree kernel scores
    zero, so the subtree kernel has to answer for it alone. A chain and a star are the two
    extremes of shape. The pairs one leaf apart are what a counting kernel has
    to separate, the repetitions are what a multiset counts and a set would not, the swapped
    arguments are what the word ordered in the name of this kernel means, and the last three
    terms carry roots that are not strings, because a signature is built from whatever the roots
    are.

    The chain stays far below the interpreter's recursion limit. The class recurses once per
    level where cosy's count iterates over a stack, so a deeper chain would measure that limit
    rather than the agreement of the two counts.
    """
    shared = Tree("f", (Tree("a"), Tree("b")))
    star = tuple(Tree(f"L{index}") for index in range(40))
    return [
        Tree("a"),
        Tree("a"),  # the same term once more, as a second object
        _chain(60, "end"),
        _chain(60, "other end"),  # the same chain, one leaf apart
        Tree("s", star),
        Tree("s", (*star[:-1], Tree("ZZ"))),  # the same star, one leaf apart
        Tree("s", tuple(Tree("L") for _ in star)),  # the same leaf forty times
        shared,
        Tree("f", (Tree("b"), Tree("a"))),  # the arguments swapped
        Tree("f", (Tree("a"), Tree("a"))),
        Tree("g", (shared, shared)),  # one subterm at two positions
        Tree("g", (shared, Tree("f", (Tree("a"), Tree("c"))))),
        Tree("h", (Tree("f", (Tree("a"),)),) * 3),
        Tree("u", (Tree("s", star[:10]),)),
        Tree(3, (Tree(4), Tree(5))),  # roots that are not strings
        Tree(3, (Tree(4), Tree(6))),
        Tree(("level", 1), (Tree(("level", 2)),)),
    ]


def test_the_subtree_kernel_counts_what_cosy_counts():
    """``OrderedRootedSubtreeKernel`` has to give cosy's ``k_st`` entry for entry.

    That holds for every form the class is called in, and the square matrix, the cross matrix and
    the diagonal are all asked for below.

    cosy already counts the subterms two terms share, and this class counts them a second time,
    from signatures and one sparse product, because a Gram matrix wants the count once per term
    rather than once per pair. What the second count may buy is the shape of the computation, not
    a second definition of what a subtree kernel counts. ``SubsetTreeKernel`` cannot drift from
    cosy, since it calls ``k_sst`` outright, and this one can, so the agreement is asserted.

    The values on cosy's chain terms are already held against a closed form by
    ``test_the_counting_kernel_shows_the_scale_spread_the_chapter_names``. Sorting the child
    signatures leaves every value of that closed form unchanged, so what it does not hold is the
    sibling order this kernel is named for.

    The comparison is exact rather than approximate. Both routes add up integer subterm counts,
    which a double carries exactly at these sizes, and both divide by the square root of the same
    product, so a difference in the last bit would already be a difference in what was counted.
    """
    corpus = _subterm_corpus()
    cosine = normalized(k_st)

    raw = OrderedRootedSubtreeKernel(normalize=False)(corpus)
    assert np.array_equal(raw, [[k_st(left, right) for right in corpus] for left in corpus])

    matrix = OrderedRootedSubtreeKernel(normalize=True)(corpus)
    assert np.array_equal(matrix, [[cosine(left, right) for right in corpus] for left in corpus])

    # A Gaussian process calls the cross form on every prediction, and it reaches the sparse
    # product with two separately prepared argument lists instead of one shared list.
    half = len(corpus) // 2
    cross = OrderedRootedSubtreeKernel(normalize=True)(corpus[:half], corpus[half:])
    assert np.array_equal(
        cross, [[cosine(left, right) for right in corpus[half:]] for left in corpus[:half]]
    )

    assert np.array_equal(
        OrderedRootedSubtreeKernel(normalize=False).diag(corpus),
        [k_st(term, term) for term in corpus],
    )


def test_the_subtree_kernel_sees_the_argument_order():
    """``f(a, b)`` and ``f(b, a)`` are different terms and must not score as identical.

    The signature of a position is its symbol together with the signatures of its children in
    order, and that order is what the name of this kernel promises. The two terms below share
    their two leaves and nothing else, so the count is two out of three. Dropping that order,
    by sorting the child signatures, would count the roots as shared as well and call the terms
    equal, and the rest of the suite stays green when it does.
    """
    first = Tree("f", (Tree("a"), Tree("b")))
    second = Tree("f", (Tree("b"), Tree("a")))

    assert k_st(first, second) == 2.0
    assert OrderedRootedSubtreeKernel(normalize=False)([first], [second])[0, 0] == 2.0
    assert OrderedRootedSubtreeKernel(normalize=True)([first], [second])[0, 0] == 2.0 / 3.0


def test_a_fold_moves_the_terms_and_not_the_count():
    """A ``tree_transformation`` decides which terms are scored, never how they are scored.

    The three tree kernels of the DAMG example each pass a fold, so this is not the unused half
    of the class. Under a fold the terms compared are the folded ones, and every entry still has
    to be what cosy gives for those.
    """
    corpus = _subterm_corpus()
    fold = _truncate(2)
    cosine = normalized(k_st)

    folded = OrderedRootedSubtreeKernel(normalize=True, tree_transformation=fold)(corpus)
    assert np.array_equal(
        folded, [[cosine(fold(left), fold(right)) for right in corpus] for left in corpus]
    )
    assert not np.array_equal(folded, OrderedRootedSubtreeKernel(normalize=True)(corpus))


def _shifted_term(index: int, depth: int = 6) -> Tree:
    """Return a binary term of ``2**depth`` leaves whose leaf symbols are shifted by ``index``.

    Each position above the leaves carries a symbol of its own, so few of the subterms repeat:
    the term returned for ``index`` zero has 95 distinct subterms among its 127 positions.
    Repetition is what flatters the batch form, since it shrinks the feature counters the sparse
    product runs on while a walk over both terms of a pair stays as long as the terms are. Little
    repetition is therefore the harder case for the measurement below, though not the hardest.
    """
    positions = iter(range(2 ** (depth + 1)))

    def build(level: int) -> Tree:
        position = next(positions)
        if level == 0:
            return Tree(f"leaf{(position + index) % 47}")
        return Tree(f"node{position}", (build(level - 1), build(level - 1)))

    return build(depth)


def test_the_batch_form_beats_one_cosy_call_per_pair():
    """Counting the subterms a second time is only worth it if a Gram matrix gets cheaper.

    ``OrderedRootedSubtreeKernel`` re-derives what cosy's ``k_st`` already counts, and the matrix
    is the whole reason: the class walks each term once and multiplies two sparse count matrices,
    while a pairwise route walks both terms of every pair. The extraction therefore grows with
    the number of terms and the pairwise work with its square.

    The route measured against is a careful pairwise one, raw ``k_st`` over the upper triangle
    with each self-similarity taken once. That is fewer calls than ``SubsetTreeKernel`` makes
    with ``k_sst`` for the same matrix, since that one fills the whole rectangle and adds a
    self-similarity list for each side, 960 calls against 495 on thirty terms. On the terms below
    the class came out about four times faster, and the assertion asks only that it win, since
    the margin depends on how much the terms repeat.
    """
    terms = [_shifted_term(index) for index in range(30)]
    assert terms[0].size == 127

    def batched() -> float:
        start = time.perf_counter()
        OrderedRootedSubtreeKernel(normalize=True)(terms)
        return time.perf_counter() - start

    def per_pair() -> float:
        start = time.perf_counter()
        size = len(terms)
        self_similarities = [k_st(term, term) for term in terms]
        matrix = np.empty((size, size))
        for row in range(size):
            for column in range(row, size):
                value = k_st(terms[row], terms[column])
                divisor = np.sqrt(self_similarities[row] * self_similarities[column])
                matrix[row, column] = matrix[column, row] = value / divisor
        return time.perf_counter() - start

    batch_seconds = min(batched() for _ in range(3))
    pair_seconds = min(per_pair() for _ in range(3))

    assert batch_seconds < pair_seconds, (
        f"the batch form took {batch_seconds:.3f} s where one call per pair took "
        f"{pair_seconds:.3f} s"
    )


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


def _balanced(depth: int, code: int = 0) -> Tree:
    """Return a balanced binary term whose internal symbols are the bits of ``code``.

    The bits are read in preorder, so bit zero is the root and the bits after it fall on its
    leftmost descendants. Small codes therefore differ where the subset trees are concentrated.
    """
    positions = iter(range(2**depth))

    def build(level: int) -> Tree:
        if level == 0:
            return Tree("a")
        symbol = "g" if (code >> next(positions)) & 1 else "f"
        return Tree(symbol, (build(level - 1), build(level - 1)))

    return build(depth)


def _change_the_leftmost_leaf(tree: Tree) -> Tree:
    """Return the term with its leftmost leaf carrying another symbol."""
    if not tree.children:
        return Tree("b")
    return Tree(tree.root, (_change_the_leftmost_leaf(tree.children[0]), *tree.children[1:]))


def test_the_subset_tree_gram_matrix_collapses_on_large_terms():
    """On terms of a hundred nodes the normalized subset-tree matrix is numerically the identity.

    A position scores the product over its children of one plus their own score, so the
    self-similarity of a term grows exponentially in the positions that have children, while two
    different terms share only the fragments that agree everywhere they reach. Each normalized
    entry is the quotient of those two counts, and at this size that quotient is 7e-05. A
    surrogate conditioned on such a matrix carries nothing from one observation to the next.
    """
    terms = [_balanced(6, code) for code in range(10)]
    assert len(set(terms)) == 10
    assert terms[0].size == 127

    matrix = SubsetTreeKernel(normalize=True)(terms)
    off_diagonal = matrix[~np.eye(len(terms), dtype=bool)]
    eigenvalues = np.linalg.eigvalsh((matrix + matrix.T) / 2)

    assert off_diagonal.mean() == pytest.approx(6.94e-05, rel=1e-2)
    assert off_diagonal.max() < 1e-2
    assert np.max(np.abs(eigenvalues - 1.0)) < 2e-3

    self_similarity = k_sst(terms[0], terms[0])
    shared = k_sst(terms[0], terms[1])
    assert self_similarity == pytest.approx(2.1e11, rel=1e-2)
    assert self_similarity.is_integer(), "the collapse is the ratio of two counts, not a rounding"
    assert shared.is_integer(), "the shared count is exact as well, so the quotient is too"


def test_where_two_terms_differ_decides_the_subset_tree_entry():
    """One changed symbol costs a term the fragments that ran through that symbol's position.

    The count at a position is the product over its children, so the fragments rooted near the
    top carry nearly all of a term's self-similarity, and a production that differs there removes
    all of them. Size alone therefore does not say where a space sits, which is why the two
    numbers below are nearly five decades apart on terms of one and the same size.
    """
    base = _balanced(6)
    kernel = SubsetTreeKernel(normalize=True)

    at_a_leaf = kernel([base], [_change_the_leftmost_leaf(base)])[0, 0]
    at_the_root = kernel([base], [Tree("g", base.children)])[0, 0]

    assert at_a_leaf == pytest.approx(0.616, rel=1e-2)
    assert at_the_root == pytest.approx(8.8e-06, rel=1e-2)


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


# ---------------------------------------------------------------------------
# The two module caches are bounded
# ---------------------------------------------------------------------------


def _distinct_terms(prefix: str, count: int) -> list[Tree]:
    """Return ``count`` structurally distinct terms of one and the same shape."""
    return [Tree(f"{prefix}{i}", (Tree("l"), Tree("r"))) for i in range(count)]


def test_the_graph_cache_stops_growing_at_its_bound():
    """A kernel fed one new term after another may not grow the shared cache without end.

    The key of a converted graph holds the term itself, so an unbounded cache keeps every term
    a run ever scored alive for the life of the process, and nothing a caller drops reaches that
    far. What the bound costs when it bites is one conversion per evicted term.
    """
    from bayesian_optimization.kernels import graph_kernel

    graph_kernel.clear_kernel_caches()
    kernel: WeisfeilerLehmanKernel = WeisfeilerLehmanKernel(h=1)
    bound = graph_kernel._GRAPH_CACHE_MAXSIZE
    kernel._prepare_inputs(_distinct_terms("graph_bound_", bound + 50))

    assert len(graph_kernel._GRAPH_CACHE) == bound


def test_the_matrix_cache_stops_growing_at_its_bound():
    """One batch after another of terms never seen before leaves a bounded number of matrices.

    The matrix key holds both term tuples, so it pins the terms exactly as the graph key does,
    and an unnormalized kernel adds one entry per term on top because ``diag`` asks for a
    one-by-one matrix per term.
    """
    from bayesian_optimization.kernels import graph_kernel

    graph_kernel.clear_kernel_caches()
    kernel: WeisfeilerLehmanKernel = WeisfeilerLehmanKernel(h=1)
    bound = graph_kernel._MATRIX_CACHE_MAXSIZE
    for index in range(bound + 20):
        kernel(_distinct_terms(f"matrix_bound_{index}_", 2))

    assert len(graph_kernel._MATRIX_CACHE) == bound


def test_the_entry_that_goes_is_the_one_read_longest_ago():
    """Eviction follows the reads, not the writes.

    A pass converts the batch and the training set and then asks for the matrix over the two, so
    the training set is both the oldest entry of the pass and the one the next pass reads first.
    Dropping the oldest written entry would drop exactly it.
    """
    from bayesian_optimization.kernels.graph_kernel import _BoundedCache

    cache: _BoundedCache[int] = _BoundedCache(2)
    cache["first"] = 1
    cache["second"] = 2
    assert cache.get("first") == 1

    cache["third"] = 3

    assert cache.get("first") == 1
    assert cache.get("third") == 3
    assert cache.get("second") is None


def test_a_cache_of_no_entries_is_refused():
    """A bound below one turns every write into a write and an eviction."""
    from bayesian_optimization.kernels.graph_kernel import _BoundedCache

    with pytest.raises(ValueError, match="at least one entry"):
        _BoundedCache(0)


def test_a_caller_writing_into_a_matrix_does_not_reach_the_cached_one():
    """What a caller does to the matrix it is handed may not survive in the cache.

    This is what a Gaussian process does to it. sklearn adds its jitter to the diagonal of the
    matrix the kernel returns, in place, once in fit and once in the marginal likelihood that fit
    evaluates, so a cached entry passed out directly drifts by twice the jitter on every fit and
    keeps whatever it has drifted to. Under a bound the damage is worse than under none, because
    whether a caller sees the drifted entry or a freshly computed one then depends on what was
    evicted in between.
    """
    from bayesian_optimization.kernels import graph_kernel

    terms = _distinct_terms("no_write_through_", 4)
    kernel: WeisfeilerLehmanKernel = WeisfeilerLehmanKernel(h=1)

    graph_kernel.clear_kernel_caches()
    first = np.asarray(kernel(terms), dtype=float)
    first[np.diag_indices_from(first)] += 1e-6
    second = np.asarray(kernel(terms), dtype=float)
    graph_kernel.clear_kernel_caches()
    cold = np.asarray(kernel(terms), dtype=float)

    assert np.array_equal(second, cold)
