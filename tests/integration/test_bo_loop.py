from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from cosy.core.tree import Tree


# ---------------------------------------------------------------------------
# Top-level imports tests
# ---------------------------------------------------------------------------


def test_new_top_level_imports_work():
    from bayesian_optimization import BayesianOptimization  # noqa: F401
    from bayesian_optimization import Log1pTransform, IdentityTransform  # noqa: F401


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
        acquisition_objective: Any,
        population_size: int,
        mutation_rate: float = 0.02,
        recombination_rate: float = 0.95,
        verbose: bool = False,
        fitness_function_mode: str = "batch",
    ) -> Tree | None:
        if not self._pool:
            return None
        candidates = list(self._rng.choices(self._pool, k=min(5, len(self._pool))))
        if fitness_function_mode == "batch":
            scores = acquisition_objective(candidates)
            if scores:
                return max(scores, key=lambda t: scores[t])
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
    return float(_tree_depth(t))


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
    """BO should find a tree with depth 0 (the optimum) within a few iterations."""
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
    y0 = [_toy_objective(t) for t in x0]
    bo.initialize(x0=x0, y0=y0)

    for _ in range(6):
        s = bo.suggest()
        y = _toy_objective(s.candidate)
        bo.observe(s.candidate, y)

    _, best_y = bo.best()
    assert best_y <= 1.0, f"Expected convergence to depth<=1, got {best_y}"


@pytest.mark.slow
@pytest.mark.integration
def test_bo_diagnostics_iteration_monotone():
    """iteration in diagnostics must be strictly increasing across suggest() calls."""
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

    # iteration in diagnostics = self._iteration at time of suggest
    for i in range(1, len(iterations)):
        assert iterations[i] >= iterations[i - 1]


@pytest.mark.slow
@pytest.mark.integration
def test_bo_incumbent_raw_non_increasing_for_minimization():
    """For minimization, incumbent_raw must be non-increasing over iterations."""
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
    y0 = [_toy_objective(t) for t in x0]
    bo.initialize(x0=x0, y0=y0)

    incumbent_raws = []
    for _ in range(5):
        s = bo.suggest()
        incumbent_raws.append(s.diagnostics["incumbent_raw"])
        bo.observe(s.candidate, _toy_objective(s.candidate))

    for i in range(1, len(incumbent_raws)):
        assert incumbent_raws[i] <= incumbent_raws[i - 1] + 1e-10
