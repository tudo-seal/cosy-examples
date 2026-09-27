from __future__ import annotations

import ast
import importlib
import importlib.util
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from cosy.core.tree import Tree
from sklearn.gaussian_process import GaussianProcessRegressor

# ---------------------------------------------------------------------------
# The test modules a run without the examples' dependencies leaves out
# ---------------------------------------------------------------------------

#: What the examples need beyond the core and the run layer, installed with the ``cnn`` and
#: ``examples`` extras. A run where one of them is missing, the checks' job without torch, leaves
#: out every test module that needs it and says which.
OPTIONAL_DEPENDENCIES = ("torch", "torchvision", "pandas", "tqdm")
_ABSENT = frozenset(
    name for name in OPTIONAL_DEPENDENCIES if importlib.util.find_spec(name) is None
)
_LEFT_OUT: dict[str, str] = {}


def _missing_dependency_of(path: Path) -> str | None:
    """The absent dependency a test module's module-level imports need, or None.

    Decided by importing what the module imports at its top level, this repository's modules
    included, so a module that needs torch through the CIFAR example is found without a list of
    such modules to keep. An import inside a test is that test's own business: it skips itself.
    """
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            # the module, and each imported name that is a submodule of it
            names = [node.module, *(f"{node.module}.{alias.name}" for alias in node.names)]
        else:
            continue
        for name in names:
            top = name.split(".")[0]
            if top in _ABSENT:
                return top
            if top in ("bayesian_optimization", "tests"):
                try:
                    importlib.import_module(name)
                except ModuleNotFoundError as error:
                    missing = (error.name or "").split(".")[0]
                    if missing in _ABSENT:
                        return missing
                    if error.name != name:  # not merely a name that is no module
                        raise
    return None


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if not _ABSENT or collection_path.suffix != ".py" or not collection_path.name.startswith("test_"):
        return None
    missing = _missing_dependency_of(collection_path)
    if missing is None:
        return None
    _LEFT_OUT[str(collection_path.relative_to(config.rootpath))] = missing
    return True


def pytest_terminal_summary(terminalreporter: Any) -> None:
    if _LEFT_OUT:
        terminalreporter.write_sep("-", f"{len(_LEFT_OUT)} test modules left out, a dependency missing")
        for module, missing in sorted(_LEFT_OUT.items()):
            terminalreporter.write_line(f"{module}: needs {missing}")

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
        query: Any,
        acquisition_objective: Any,
        fitness_function_mode: str = "batch",
    ) -> Any | None:
        """Stand in for ``EvolutionarySearch.evolutionary_best``.

        The query and the quality measure are the arguments of a run. Population size and rates
        are parameters of the search object and no longer passed per call.
        """
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

    kernel: OrderedRootedSubtreeKernel[Any] = OrderedRootedSubtreeKernel(normalize=True)
    X = np.asarray(tree_corpus, dtype=object)
    y = np.array([float(i + 1) * 0.5 for i in range(len(tree_corpus))])
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True, optimizer=None)
    gp.fit(X, y)
    return gp


# ---------------------------------------------------------------------------
# BO factory fixture
# ---------------------------------------------------------------------------

# Candidates that are distinct from tree_corpus entries.
_DEFAULT_CANDIDATES: list[Tree] = [
    Tree("new1", (Tree("child1"),)),
    Tree("new2", (Tree("child2"), Tree("other"))),
    Tree("new3"),
    Tree("new4", (Tree("d"),)),
    Tree("new5", (Tree("e"), Tree("f"))),
    Tree("new6", (Tree("g", (Tree("h"),)),)),
    Tree("new7"),
    Tree("new8", (Tree("i"), Tree("j"), Tree("k"))),
]


@pytest.fixture
def bo_factory(dummy_optimizer_factory: Any) -> Any:
    """Factory that creates an *uninitialized* BayesianOptimization.

    Usage::

        def test_foo(bo_factory, tree_corpus):
            bo = bo_factory()
            bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
            ...
    """
    def factory(
        candidates: list[Any] | None = None,
        **bo_kwargs: Any,
    ) -> Any:
        from bayesian_optimization.bo import BayesianOptimization

        if candidates is None:
            candidates = list(_DEFAULT_CANDIDATES)

        optimizer = dummy_optimizer_factory(candidates)
        kwargs: dict[str, Any] = {
            "search_space": None,
            "request": None,
            "optimizer": optimizer,
            "seed": 42,
        }
        kwargs.update(bo_kwargs)
        return BayesianOptimization(**kwargs)

    return factory
