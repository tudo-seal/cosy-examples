from __future__ import annotations

import inspect
from abc import ABC
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Generic, NamedTuple, TypeVar

import numpy as np
from cosy.core.tree import Tree
from grakel.kernels import VertexHistogram, WeisfeilerLehman
from sklearn.gaussian_process.kernels import Hyperparameter

from ..utils import to_grakel_graph as to_tree_graph
from .kernel_base import StructuredKernelBase

T = TypeVar("T", bound=Hashable)
V = TypeVar("V")


class _BoundedCache(Generic[V]):
    """A mapping of at most ``maxsize`` entries that drops the least recently used one first.

    The bound is what separates a cache from a leak here.  Both keys used below hold the terms
    themselves, so an entry keeps every term it was built from alive, and the caches sit on the
    module, where nothing a caller drops can reach them.

    Least recently used and not first in.  A pass over a batch converts the batch and the training
    set and then asks for the matrix over the two, and the training set is what the next pass asks
    for again.  Dropping the oldest entry would drop exactly those.

    Parameters
    ----------
    maxsize:
        How many entries are kept.  It has to exceed what a single pass reads, one entry per level
        and term of the batch and of the training set together, or a pass evicts what it is about
        to read again.
    """

    __slots__ = ("_entries", "maxsize")

    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            msg = f"a cache holds at least one entry, got maxsize={maxsize}"
            raise ValueError(msg)
        self.maxsize = maxsize
        self._entries: OrderedDict[Any, V] = OrderedDict()

    def get(self, key: Any) -> V | None:
        """Return the entry stored under ``key`` and make it the most recently used one.

        Args:
            key: What to look up.

        Returns:
            V | None: The entry, or None if there is none.
        """
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry

    def __setitem__(self, key: Any, value: V) -> None:
        """Store ``value`` under ``key``, dropping least recently used entries over the bound.

        Args:
            key: What to store it under.
            value: What to store.
        """
        self._entries[key] = value
        self._entries.move_to_end(key)
        while len(self._entries) > self.maxsize:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        """Return how many entries are held.

        Returns:
            int: The entry count, never above ``maxsize``.
        """
        return len(self._entries)

    def clear(self) -> None:
        """Drop every entry, and with it every term the keys keep alive."""
        self._entries.clear()


# Caches shared by every instance and keyed by what the value depends on, never by object
# identity.  sklearn clones a kernel on every fit, and ``clone_with_theta`` builds a fresh object
# for each step of the hyperparameter optimizer, so a cache held per instance is thrown away
# exactly when it would start paying off.  An ``id()``-keyed cache is worse than useless: ids are
# reused after collection, so a fresh graph can inherit a matrix computed for a different one.
#
# The translation is a **callable**, and a callable has no value key, because two closures over
# different levels are different translations and there is nothing in them to compare.  So the key
# holds the callable **itself** rather than its id.  It still hashes and compares by identity,
# which is the intended semantics.  What changes is that the cache now keeps a strong reference,
# so the id of a key can never be reused while its entry lives.
#
# This was a real collision and not a hypothetical one.  ``as_hierarchical_damg(level)`` returns a
# fresh closure per call, so a caller that builds one kernel per level without keeping the
# converters alive got the *same* id three times and the same matrix three times, measured on the
# DAMG hierarchies 1, 2 and 3, whose Gram matrices came out identical to every decimal.  The price
# of the fix is that the cache keeps those closures alive.  They are small, and the matrices
# beside them are not.
#
# Both bounds are measured and not guessed.  The loop was run over the expression space of
# ``tests/integration/test_real_ea_loop.py`` under a three-level hierarchical kernel, a population
# of 100 over 10 generations and 15 iterations, with every lookup recorded and the recorded
# sequence replayed under one bound after another.  The hit rates below come from that replay and
# are exact.  The miss costs are wall clock on one machine and are given as the spread of repeated
# runs.
#
# Graphs, 1024.  One pass holds a graph per level and term of the batch and of the training set at
# once, and below that working set a pass evicts what it is about to read again.  The replay shows
# where that edge is: 67.5 percent hits at 256 entries and 97.6 at 512 for the normalized kernel,
# 55.5 and 98.2 for the unnormalized one, and not one hit more unbounded in either.  A miss costs
# one conversion of a term into a grakel graph, 0.1 to 0.2 ms on that space and more on the deeper
# terms of a real repository.
#
# Matrices, 1024.  An unnormalized kernel reaches ``diag`` on every prediction that asks for a
# standard deviation, and ``diag`` asks for a 1x1 matrix per term and level, so there the cache
# grows per term rather than per batch.  Replayed unnormalized: 5.7 percent hits at 128 entries,
# 18.5 at 256, 86.3 at 512, and not one hit more unbounded.  Normalized it stays at 19.1 percent
# under every bound, since those keys carry the whole training tuple and rarely repeat.  A miss
# costs one grakel kernel over the two blocks, 0.2 to 0.6 ms for a single pair and 2.4 to 3.8 ms
# for a batch of 100 against 49 observations.
#
# Bounding the count bounds the memory too.  An entry is 8 bytes per pair drawn from the two
# blocks, so 1024 of the single-term matrices an evolutionary optimizer asks for against 50
# observations are 400 KiB, and 1024 matrices of a batch of 100 against 50 are 39 MiB.
_GRAPH_CACHE_MAXSIZE = 1024
_MATRIX_CACHE_MAXSIZE = 1024

_GRAPH_CACHE: _BoundedCache[Any] = _BoundedCache(_GRAPH_CACHE_MAXSIZE)
_MATRIX_CACHE: _BoundedCache[np.ndarray] = _BoundedCache(_MATRIX_CACHE_MAXSIZE)


def clear_kernel_caches() -> None:
    """Empty the two caches the graph kernels share.

    They live on the module, so they outlive every kernel, every model and every optimization
    built over them, and their keys hold the terms.  A caller who wants the terms of a finished
    run released says so here, and :meth:`BayesianOptimization.reset` does.  There is nothing
    narrower to empty: an entry records the term and the translation it was built from, not who
    asked for it.
    """
    _GRAPH_CACHE.clear()
    _MATRIX_CACHE.clear()


def _materialize_graph(graph_source: Any) -> Any:
    if hasattr(graph_source, "__next__"):
        return next(graph_source)
    return graph_source


class _Prepared(NamedTuple):
    """A converted term, carried together with the term it came from.

    The term is the cache key.  grakel's converted graphs are plain lists of adjacency and label
    maps, so they are neither hashable nor comparable in any useful way, while a term is both and
    determines the graph completely once the translation is fixed.
    """

    term: Any
    graph: Any


class _GrakelWeisfeilerLehmanBase(StructuredKernelBase[T], ABC):
    """Shared plumbing for the grakel-backed Weisfeiler-Lehman kernels.

    **The ``h`` convention.**  The Weisfeiler-Lehman kernel sums the base kernel over the
    relabeling rounds ``i = 0..h``, so ``h`` counts the rounds performed *in addition to*
    comparing the initial labels, and ``h`` rounds give ``h + 1`` summands.  grakel adds the same
    one itself, since ``weisfeiler_lehman.py`` sets ``_n_iter = n_iter + 1`` and builds one base
    kernel per level, so ``n_iter = h``.  The conversion happens here and nowhere else, and
    everything above this line uses ``h`` in the sense just given.

    ``h = 0`` is one summand, the initial labels, and grakel cannot express it: its WL rejects
    ``n_iter <= 0``.  It is the base kernel by itself, and that is what this builds.

    Getting this wrong is not a naming question and it is not loud.  ``n_iter = h + 1`` ran one
    relabeling round for every kernel that asked for none, which is the configuration the CNN
    experiments used throughout: measured on one pair of terms that differ only in argument order,
    0.25 instead of 0.5.  Both readings produce a valid kernel, a fitted GP and a finished run.
    """

    def __init__(
        self,
        base_graph_kernel: Any = VertexHistogram,
        normalize: bool = True,
        n_jobs: int | None = None,
        epsilon: float = 1e-10,
    ) -> None:
        super().__init__(epsilon=epsilon)
        self.base_graph_kernel = base_graph_kernel
        self.normalize = normalize
        self.n_jobs = n_jobs

    def _base_kernel_arguments(self) -> dict[str, Any]:
        """Return the arguments for building the base kernel on its own, at ``h = 0``.

        ``sparse=False`` where the base kernel understands it, and the reason was a defect in grakel
        rather than a preference.  ``VertexHistogram`` decides at *fit* time whether to hold its
        features sparsely, and it does when the fitted graphs share few labels, which is what a
        dataset of different architectures looks like.  Up to grakel 0.1.10 the features on that
        path are integer counts, and the normalization then divides an integer matrix **in place**,
        which numpy refuses outright: ``Cannot cast ufunc 'divide' output from float64 to int64``.
        The pinned 0.1.11 builds that matrix as ``float64`` and the division goes through, so the
        setting is no longer what stands between this kernel and that error.  It stays because the
        dense path is the one every measurement here was taken on, and turning it off is a change
        worth its own measurement rather than a side effect of a version bump.

        It only reaches us at ``h = 0``, because that is the only path handing the base kernel out
        directly.  For ``h >= 1`` grakel's own ``WeisfeilerLehman`` wraps it and accumulates in
        floats.  Nothing small reproduces it either: one graph in the fit gives density 1.0 and the
        dense path, so it takes several graphs with mostly disjoint labels *and* a cross
        evaluation, exactly what a GP does when it predicts, and exactly what no unit test did
        until one was written for it.

        The dense array costs ``n_graphs x n_labels`` floats, a few megabytes at the sizes here.
        grakel falls back to sparse on ``MemoryError``, which is the one path that still reaches
        the sparse branch, and under a grakel older than the pin it would bring the defect back.
        That fallback is louder than the alternative, which is why it is left alone rather than
        suppressed.

        Returns:
            dict[str, Any]: The keyword arguments the base kernel accepts.
        """
        arguments: dict[str, Any] = {"normalize": self.normalize, "n_jobs": self.n_jobs}
        # The base kernel is the caller's choice, and not every grakel kernel has ``sparse``.
        if "sparse" in inspect.signature(self.base_graph_kernel).parameters:
            arguments["sparse"] = False
        return arguments

    def _build_kernel(self, h: int) -> Any:
        """Build the grakel kernel for ``h`` relabeling rounds.

        Args:
            h: The number of relabeling rounds, counted in addition to the comparison of the
                initial labels.

        Returns:
            Any: The backend kernel, which is grakel's ``WeisfeilerLehman`` for ``h >= 1`` and the
                base kernel itself for ``h = 0``, the case of one summand that grakel's WL
                declines to build.

        Raises:
            ValueError: If ``h`` is negative.  Zero is meaningful, since it compares initial
                labels only, but a negative round count is not, and clamping it would hide the
                mistake.
        """
        if h < 0:
            msg = f"the number of relabeling rounds h must be non-negative, got {h}"
            raise ValueError(msg)
        if h == 0:
            return self.base_graph_kernel(**self._base_kernel_arguments())
        return WeisfeilerLehman(
            n_iter=h,
            base_graph_kernel=self.base_graph_kernel,
            normalize=self.normalize,
            n_jobs=self.n_jobs,
        )

    def _kernel_from_graphs(
        self,
        X_graphs: Sequence[Any],
        Y_graphs: Sequence[Any],
        h: int,
    ) -> np.ndarray:
        if len(X_graphs) == 0 or len(Y_graphs) == 0:
            return np.zeros((len(X_graphs), len(Y_graphs)), dtype=float)

        key = (
            tuple(item.term for item in X_graphs),
            tuple(item.term for item in Y_graphs),
            int(h),
            self.base_graph_kernel,
            bool(self.normalize),
            self._translation_key(),
        )
        # Every caller gets an array of its own.  A Gaussian process adds its jitter to the
        # diagonal of the matrix it is handed, in place, so an entry passed out directly drifts by
        # twice that jitter on every fit that reads it, and the drift accumulates for as long as
        # the entry lives.
        cached = _MATRIX_CACHE.get(key)
        if cached is not None:
            return cached.copy()

        x_raw = [item.graph for item in X_graphs]
        y_raw = [item.graph for item in Y_graphs]

        wl_kernel = self._build_kernel(h)
        if X_graphs is Y_graphs:
            matrix = np.asarray(wl_kernel.fit_transform(x_raw), dtype=float)
        else:
            wl_kernel.fit_transform(y_raw)
            matrix = np.asarray(wl_kernel.transform(x_raw), dtype=float)

        _MATRIX_CACHE[key] = matrix
        return matrix.copy()

    def _translation_key(self) -> Any:
        """Return what distinguishes this kernel's term-to-graph translation from another's.

        The callable itself, not its id: see the note above the caches.  Identity is still the
        comparison, since a function's ``__eq__`` is identity, but holding the object keeps that
        identity meaningful for as long as the cache entry lives.

        Returns:
            Any: The translation, as its own key.
        """
        return getattr(self, "to_grakel_graph", None)


class WeisfeilerLehmanKernel(_GrakelWeisfeilerLehmanBase, Generic[T]):
    """Weisfeiler-Lehman kernel for terms, computed through a graph backend.

    ``k(t, t') = sum over i = 0..h of k_base(G_i, G_i')``, where ``G_0`` is the graph of the term.
    That graph has the positions of the term as its nodes, each labeled with the symbol at that
    position, and an edge between the position of every subterm and the positions of its immediate
    subterms.  Each round then replaces every node label by the pair of its own label and the
    sorted multiset of its neighbors'.

    ``h`` is not a hyperparameter the marginal likelihood can adjust.  It is discrete, and once two
    graphs share no label they share none in any later round either, so the kernel is constant in
    ``h`` from that round on and on small terms there is often nothing for a model selection to
    choose between.  :class:`HierarchicalWLKernel` is the answer to that, turning one round count
    into a continuous weight per granularity level.

    Parameters
    ----------
    h:
        Number of relabeling rounds, performed in addition to comparing the initial labels.
        ``h = 0`` compares the initial labels only.
    base_graph_kernel:
        The grakel base kernel.  The standard choice is the vertex histogram, which counts pairs
        of equally labeled nodes.
    normalize:
        Cosine normalization of the Gram matrix, dividing each entry by the square roots of the
        two self-similarities, which removes the size bias of a counting kernel.
    to_grakel_graph:
        The translation from term to graph.  A kernel may run on a translation of the term rather
        than on the term itself, and such a translation is itself a fold, which is the hook the
        CNN example uses to score architectures as string diagrams.
    """

    def __init__(
        self,
        h: int = 1,
        base_graph_kernel: Any = VertexHistogram,
        normalize: bool = True,
        to_grakel_graph: Callable[[Tree[T]], Any] = to_tree_graph,
        n_jobs: int | None = None,
        epsilon: float = 1e-10,
    ) -> None:
        super().__init__(
            base_graph_kernel=base_graph_kernel,
            normalize=normalize,
            n_jobs=n_jobs,
            epsilon=epsilon,
        )
        self.h = h
        self.to_grakel_graph = to_grakel_graph

    def _graph_from_tree(self, tree: Tree[T]) -> _Prepared:
        # The converter itself, not its id.  See the note above the caches.
        key = (self.to_grakel_graph, tree)
        cached = _GRAPH_CACHE.get(key)
        if cached is None:
            cached = _materialize_graph(self.to_grakel_graph(tree))
            _GRAPH_CACHE[key] = cached
        return _Prepared(tree, cached)

    def _prepare_inputs(self, X: Sequence[Tree[T]]) -> Sequence[Any]:
        return [self._graph_from_tree(t) for t in X]

    def _kernel_matrix(
        self, X_prepared: Sequence[Any], Y_prepared: Sequence[Any]
    ) -> np.ndarray:
        return self._kernel_from_graphs(X_prepared, Y_prepared, int(self.h))

    def diag(self, X: Any) -> np.ndarray:
        """Return the self-similarities without building the full matrix.

        Args:
            X: The terms.

        Returns:
            np.ndarray: One value per element of ``X``.
        """
        if self.normalize:
            return np.ones(len(X), dtype=float)
        return np.array(
            [float(self._kernel_matrix([g], [g])[0, 0]) for g in self._prepare_inputs(X)],
            dtype=float,
        )


class HierarchicalWLKernel(_GrakelWeisfeilerLehmanBase, Generic[T]):
    """Weisfeiler-Lehman kernel summed over the granularity levels of a term, one weight each.

    ``k_hWL(t, t') = sum over l of lambda_l * k_WL^(h)(G_l, G_l')``, where ``G_l`` is the term
    graph of ``t`` truncated to granularity level ``l``.  The sum is what widens the model
    selection's reach: instead of choosing one round count it offers a weight per level, and each
    weight is a continuous hyperparameter the marginal likelihood can move once a caller passes a
    ``kernel_optimizer``, which no longer happens by default.  The weights also set the scale, as
    :meth:`diag` shows, so this kernel needs no constant factor around it.

    The construction starts its sum at level 2, because the graph of level 1 carries no edges and
    its relabeling rounds have nothing to propagate.  The truncation itself is a single fold, into
    an algebra whose carrier holds one truncation per level, which is why it enters as a function
    per level rather than as a method of this class: what "level" means belongs to the repository,
    not to the kernel.

    Parameters
    ----------
    truncations:
        One fold per level, in level order.  Each maps a term to its truncation.
    lambdas:
        The weight per level.  Fitted by the marginal likelihood unless bounds are "fixed".
    lambda_bounds:
        Bounds for the weights.  The default spans four orders of magnitude, which is enough for
        a level to switch off or dominate.  Restarts of the optimizer are drawn log-uniformly from
        these bounds, so a wider range mostly wastes them.
    h:
        Relabeling rounds, shared by all levels.
    """

    def __init__(
        self,
        truncations: Sequence[Callable[[Tree[T]], Tree[T]]],
        lambdas: Sequence[float] | None = None,
        lambda_bounds: tuple[float, float] = (1e-4, 1e2),
        h: int = 1,
        base_graph_kernel: Any = VertexHistogram,
        normalize: bool = True,
        to_grakel_graph: Callable[[Tree[T]], Any] = to_tree_graph,
        n_jobs: int | None = None,
        epsilon: float = 1e-10,
    ) -> None:
        super().__init__(
            base_graph_kernel=base_graph_kernel,
            normalize=normalize,
            n_jobs=n_jobs,
            epsilon=epsilon,
        )
        if not truncations:
            msg = "a hierarchical kernel needs at least one level"
            raise ValueError(msg)
        # Stored exactly as handed in, never copied: sklearn's ``clone`` rebuilds a kernel from
        # ``get_params()`` and rejects any constructor that does not return the same objects.  It
        # clones on every fit, so a copy here would make the kernel unfittable.
        self.truncations = truncations
        self.lambdas = (
            [1.0 / len(truncations)] * len(truncations) if lambdas is None else lambdas
        )
        if len(self.lambdas) != len(truncations):
            msg = (
                f"one weight per level: {len(truncations)} truncations against "
                f"{len(self.lambdas)} weights"
            )
            raise ValueError(msg)
        self.lambda_bounds = lambda_bounds
        self.h = h
        self.to_grakel_graph = to_grakel_graph

    @property
    def hyperparameter_lambdas(self) -> Hyperparameter:
        """The level weights, as one vector-valued hyperparameter."""
        return Hyperparameter(
            "lambdas", "numeric", self.lambda_bounds, len(self.truncations)
        )

    def _translation_key(self) -> Any:
        """Return what distinguishes this kernel's translation, converter and folds together.

        Returns:
            Any: The converter and the folds themselves, for the reason given above the caches.  A
                fold is a closure too, and ``hierarchical_tree(2)`` and ``hierarchical_tree(3)``
                differ in nothing a value comparison could see.
        """
        return (self.to_grakel_graph, tuple(self.truncations))

    def _graphs_per_level(self, X: Sequence[Tree[T]]) -> list[list[_Prepared]]:
        graphs = []
        for level, truncate in enumerate(self.truncations):
            per_level = []
            for tree in X:
                key = (self.to_grakel_graph, (level, truncate, tree))
                cached = _GRAPH_CACHE.get(key)
                if cached is None:
                    cached = _materialize_graph(self.to_grakel_graph(truncate(tree)))
                    _GRAPH_CACHE[key] = cached
                # The truncation is part of what identifies this graph, and the level index
                # stands for it: two levels of the same term are different graphs.
                per_level.append(_Prepared((level, tree), cached))
            graphs.append(per_level)
        return graphs

    def _prepare_inputs(self, X: Sequence[Tree[T]]) -> Sequence[Any]:
        return self._graphs_per_level(X)

    def _kernel_matrix(
        self, X_prepared: Sequence[Any], Y_prepared: Sequence[Any]
    ) -> np.ndarray:
        n_x = len(X_prepared[0]) if X_prepared else 0
        n_y = len(Y_prepared[0]) if Y_prepared else 0
        total = np.zeros((n_x, n_y), dtype=float)
        for weight, x_level, y_level in zip(
            self.lambdas, X_prepared, Y_prepared, strict=True
        ):
            total += float(weight) * self._kernel_from_graphs(x_level, y_level, int(self.h))
        return total

    def diag(self, X: Any) -> np.ndarray:
        """Return the self-similarities without building the full matrix.

        Args:
            X: The terms.

        Returns:
            np.ndarray: One value per element of ``X``.
        """
        if self.normalize:
            # Every level's normalized kernel has diagonal 1, so the sum has diagonal sum(lambda).
            return np.full(len(X), float(sum(self.lambdas)), dtype=float)
        levels = self._graphs_per_level(list(X))
        diagonal = np.zeros(len(X), dtype=float)
        for weight, per_level in zip(self.lambdas, levels, strict=True):
            for index, graph in enumerate(per_level):
                diagonal[index] += float(weight) * float(
                    self._kernel_from_graphs([graph], [graph], int(self.h))[0, 0]
                )
        return diagonal
