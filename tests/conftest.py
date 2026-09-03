from __future__ import annotations

import random
from typing import Any

import numpy as np
import pytest
from cosy.core.tree import Tree
from sklearn.gaussian_process import GaussianProcessRegressor

# ---------------------------------------------------------------------------
# Shared Tree corpus
# ---------------------------------------------------------------------------

@pytest.fixture
def seeded_rng() -> random.Random:
    return random.Random(42)


@pytest.fixture
def tree_corpus() -> list[Tree]:
    """10 structurally diverse Tree[str] objects."""
    return [
        Tree("A", (Tree("B"), Tree("C"))),
        Tree("A", (Tree("B"), Tree("D"))),
        Tree("X", (Tree("Y", (Tree("Z"),)),)),
        Tree("R", (Tree("S"), Tree("T"), Tree("U"))),
        Tree("M", (Tree("N"),)),
        Tree("P", (Tree("Q", (Tree("R"),)), Tree("S"))),
        Tree("alpha"),
        Tree("beta", (Tree("gamma"), Tree("delta"))),
        Tree("E", (Tree("F", (Tree("G"), Tree("H"))), Tree("I"))),
        Tree("V", (Tree("W", (Tree("X2", (Tree("Y2"),)),)),)),
    ]


# ---------------------------------------------------------------------------
# DummyOptimizer
# ---------------------------------------------------------------------------

class DummyOptimizer:
    """Deterministic optimizer that pops candidates from a pre-supplied list."""

    def __init__(self, candidates: list[Any]):
        self._candidates = list(candidates)

    def evolutionary_best(
        self,
        acquisition_objective: Any,
        population_size: int,
        mutation_rate: float = 0.02,
        recombination_rate: float = 0.95,
        verbose: bool = False,
        fitness_function_mode: str = "batch",
    ) -> Any | None:
        if not self._candidates:
            return None
        return self._candidates.pop(0)


@pytest.fixture
def dummy_optimizer_factory():
    """Factory: dummy_optimizer_factory([t1, t2, ...]) -> DummyOptimizer."""
    def factory(candidates: list[Any]) -> DummyOptimizer:
        return DummyOptimizer(list(candidates))
    return factory


# ---------------------------------------------------------------------------
# Fitted GP fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_gp(tree_corpus: list[Tree]) -> GaussianProcessRegressor:
    """GP fitted on tree_corpus with a simple loss function."""
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    kernel = OrderedRootedSubtreeKernel(normalize=True)
    X = np.asarray(tree_corpus, dtype=object)
    y = np.array([float(i + 1) * 0.5 for i in range(len(tree_corpus))])
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, optimizer=None)
    gp.fit(X, y)
    return gp

