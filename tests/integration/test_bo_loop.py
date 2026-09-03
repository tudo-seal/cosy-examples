from __future__ import annotations

from typing import Any

import pytest
from cosy.core.tree import Tree

# ---------------------------------------------------------------------------
# Top-level imports tests
# ---------------------------------------------------------------------------


def test_new_top_level_imports_work():
    from bayesian_optimization import (  # noqa: F401
        BayesianOptimization,
        ExpectedImprovement,
        OrderedRootedSubtreeKernel,
        UpperConfidenceBound,
    )


# ---------------------------------------------------------------------------
# Stub evolutionary optimizer for integration tests
# ---------------------------------------------------------------------------

class StubEvolutionary:
    """Cycles through a fixed pool of candidates uniformly."""

    def __init__(self, pool: list[Tree], seed: int = 0):
        import random
        self._pool = list(pool)
        self._rng = random.Random(seed)

    def evolutionary_best(
        self,
        query: Any,
        acquisition_objective: Any,
        fitness_function_mode: str = "batch",
    ) -> Tree | None:
        """Stand in for ``EvolutionarySearch.evolutionary_best``.

        The signature is the one the driver has now: the search space and the quality measure
        are the *arguments* of a run, while population size and rates are parameters fixed on
        the object beforehand, so they do not appear here at all.  This stub ignores the query
        and draws from a pool of its own.
        """
        if not self._pool:
            return None
        candidates = list(self._rng.choices(self._pool, k=min(5, len(self._pool))))
        if fitness_function_mode == "batch":
            scores = acquisition_objective(candidates)
            if scores:
                best: Tree = max(scores, key=lambda t: scores[t])
                return best
            return candidates[0]
        else:
            scored = [(t, float(acquisition_objective(t))) for t in candidates]
            return max(scored, key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# Toy problem: objective is tree_depth (lower = better)
# ---------------------------------------------------------------------------

def _tree_depth(t: Tree) -> int:
    if not t.children:
        return 0
    return 1 + max(_tree_depth(c) for c in t.children)


def _toy_objective(t: Tree) -> float:
    """Return the depth of a tree, as a quantity to maximize."""
    return float(_tree_depth(t))


def _toy_objective_minimizing_depth(t: Tree) -> float:
    """Return the negated depth, which is how a minimizing caller states the same quantity.

    Negation is the whole of the conversion the loop asks a caller to make, and these tests are
    where it is exercised end to end: the loop climbs, the depth falls.
    """
    return -float(_tree_depth(t))


def _build_pool(n: int = 30) -> list[Tree]:
    """Build a fixed pool of trees with known objectives."""
    trees = [
        Tree("A"),  # depth 0 → objective 0.0 (optimal)
        Tree("B"),
        Tree("C"),
        Tree("D"),
        Tree("A", (Tree("B"),)),  # depth 1 → 1.0
        Tree("A", (Tree("B"), Tree("C"))),
        Tree("X", (Tree("Y"),)),
        Tree("M", (Tree("N"),)),
        Tree("A", (Tree("B", (Tree("C"),)),)),  # depth 2 → 2.0
        Tree("R", (Tree("S", (Tree("T"),)),)),
        Tree("P", (Tree("Q", (Tree("R"),)), Tree("S"))),
        Tree("E", (Tree("F", (Tree("G"),)),)),
    ]
    return trees


@pytest.mark.slow
@pytest.mark.integration
def test_bo_converges_on_toy_problem():
    """A minimizing run reaches a tree of depth at most one within a few iterations."""
    from bayesian_optimization.bo import BayesianOptimization

    pool = _build_pool()
    optimizer = StubEvolutionary(pool, seed=7)

    bo = BayesianOptimization(
        search_space=None,
        request=None,
        optimizer=optimizer,
        acquisition_function="ExpectedImprovement",
        seed=42,
        n_restarts_kernel_optimizer=0,
    )
    # Initialize with a subset (not including the depth-0 trees)
    x0 = [t for t in pool if _tree_depth(t) >= 1][:4]
    y0 = [_toy_objective_minimizing_depth(t) for t in x0]
    bo.initialize(x0=x0, y0=y0)

    for _ in range(6):
        s = bo.suggest()
        bo.observe(s.candidate, _toy_objective_minimizing_depth(s.candidate))

    best_tree, best_y = bo.best()
    assert _tree_depth(best_tree) <= 1, f"Expected convergence to depth<=1, got {best_y}"


@pytest.mark.slow
@pytest.mark.integration
def test_bo_diagnostics_iteration_monotone():
    """The iteration reported in the diagnostics never falls across ``suggest`` calls."""
    from bayesian_optimization.bo import BayesianOptimization

    pool = _build_pool()
    optimizer = StubEvolutionary(pool, seed=13)

    bo = BayesianOptimization(
        search_space=None,
        request=None,
        optimizer=optimizer,
        seed=42,
        n_restarts_kernel_optimizer=0,
    )
    x0 = pool[:4]
    y0 = [_toy_objective(t) for t in x0]
    bo.initialize(x0=x0, y0=y0)

    iterations = []
    for _ in range(4):
        s = bo.suggest()
        iterations.append(s.diagnostics["iteration"])
        bo.observe(s.candidate, _toy_objective(s.candidate))

    # The diagnostics report the iteration counter as it stood when suggest was called.
    for i in range(1, len(iterations)):
        assert iterations[i] >= iterations[i - 1]


@pytest.mark.slow
@pytest.mark.integration
def test_bo_incumbent_tracks_a_falling_depth_under_negation():
    """A minimizing caller negates, so the incumbent rises exactly as the depth falls."""
    from bayesian_optimization.bo import BayesianOptimization

    pool = _build_pool()
    optimizer = StubEvolutionary(pool, seed=99)

    bo = BayesianOptimization(
        search_space=None,
        request=None,
        optimizer=optimizer,
        seed=42,
        n_restarts_kernel_optimizer=0,
    )
    x0 = [t for t in pool if _tree_depth(t) >= 2][:3]
    y0 = [_toy_objective_minimizing_depth(t) for t in x0]
    bo.initialize(x0=x0, y0=y0)

    incumbents = []
    for _ in range(5):
        s = bo.suggest()
        incumbents.append(s.diagnostics["incumbent"])
        bo.observe(s.candidate, _toy_objective_minimizing_depth(s.candidate))

    for i in range(1, len(incumbents)):
        assert incumbents[i] >= incumbents[i - 1] - 1e-10
    # Read back as the quantity the caller cares about: the best depth never grows.
    best_depths = [-v for v in incumbents]
    for i in range(1, len(best_depths)):
        assert best_depths[i] <= best_depths[i - 1] + 1e-10


# ---------------------------------------------------------------------------
# Maximization is the only convention the loop has, and the tests above reach a minimum through
# it by negating.  These two use it directly, on the depth itself.
# Not marked `slow`: no network training, and a run takes about a second, so these run in the
# -m "not slow" suite.
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_bo_converges_on_toy_problem_maximization():
    """Under maximization the loop climbs toward the deepest tree, the objective being depth."""
    from bayesian_optimization.bo import BayesianOptimization

    pool = _build_pool()
    # search_space=None means there is no fallback sampler, so a re-suggested duplicate would
    # raise.  This seed drives the stub down a path that keeps suggesting novel candidates, as
    # the fixed seed of the minimization test does.
    optimizer = StubEvolutionary(pool, seed=0)

    bo = BayesianOptimization(
        search_space=None,
        request=None,
        optimizer=optimizer,
        acquisition_function="ExpectedImprovement",
        seed=42,
        n_restarts_kernel_optimizer=0,
    )
    # Start without the depth-2 trees, so the optimum has to be discovered by the search.
    x0 = [t for t in pool if _tree_depth(t) <= 1][:4]
    y0 = [_toy_objective(t) for t in x0]
    bo.initialize(x0=x0, y0=y0)

    for _ in range(6):
        s = bo.suggest()
        bo.observe(s.candidate, _toy_objective(s.candidate))

    _, best_y = bo.best()
    assert best_y >= 2.0, f"Expected convergence to depth>=2 under maximization, got {best_y}"


@pytest.mark.integration
def test_bo_incumbent_non_decreasing():
    """The incumbent is the best observation so far, so it never falls."""
    from bayesian_optimization.bo import BayesianOptimization

    pool = _build_pool()
    optimizer = StubEvolutionary(pool, seed=99)

    bo = BayesianOptimization(
        search_space=None,
        request=None,
        optimizer=optimizer,
        seed=42,
        n_restarts_kernel_optimizer=0,
    )
    # Start from the shallowest trees so the incumbent has room to rise.
    x0 = [t for t in pool if _tree_depth(t) <= 0][:3]
    y0 = [_toy_objective(t) for t in x0]
    bo.initialize(x0=x0, y0=y0)

    incumbents = []
    for _ in range(5):
        s = bo.suggest()
        incumbents.append(s.diagnostics["incumbent"])
        bo.observe(s.candidate, _toy_objective(s.candidate))

    for i in range(1, len(incumbents)):
        assert incumbents[i] >= incumbents[i - 1] - 1e-10
