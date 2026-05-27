from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from cosy.core.tree import Tree

from .acquisition_function import AcquisitionFunction


def _make_acquisition_objective_batch(
    acquisition_function: AcquisitionFunction,
) -> Any:
    """Wrap an AcquisitionFunction as a batch fitness objective with caching."""
    cache: dict[Any, float] = {}

    def objective(sample: list[Any]) -> Mapping[Any, float]:
        missing = [t for t in sample if isinstance(t, Tree) and t not in cache]
        if missing:
            cache.update(acquisition_function.evaluate_batch(missing))
        return {t: cache.get(t, 0.0) for t in sample}

    return objective


def _make_acquisition_objective_single(
    acquisition_function: AcquisitionFunction,
) -> Any:
    """Wrap an AcquisitionFunction as a single-sample fitness objective with caching."""
    cache: dict[Any, float] = {}

    def objective(sample: Any) -> float:
        if isinstance(sample, Tree):
            cached = cache.get(sample)
            if cached is not None:
                return cached
            value = float(acquisition_function(sample))
            cache[sample] = value
            return value
        return float(acquisition_function(sample))

    return objective


class AcquisitionOptimizer:
    """Adapter that wraps an evolutionary optimizer to maximize an AcquisitionFunction.

    Parameters
    ----------
    evolutionary:
        Object with an ``evolutionary_best(objective, population_size, ...)`` method.
    population_size:
        EA population size.
    mutation_rate:
        EA mutation rate.
    recombination_rate:
        EA recombination rate.
    """

    def __init__(
        self,
        evolutionary: Any,
        *,
        population_size: int = 100,
        mutation_rate: float = 0.02,
        recombination_rate: float = 0.95,
    ) -> None:
        self.evolutionary = evolutionary
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.recombination_rate = recombination_rate

    def maximize(
        self,
        acquisition_fn: AcquisitionFunction,
        *,
        mode: Literal["single", "batch"] = "batch",
        verbose: bool = False,
    ) -> Any:
        """Run the EA to maximise ``acquisition_fn`` and return the best candidate.

        Parameters
        ----------
        acquisition_fn:
            The acquisition function to maximise.
        mode:
            Whether to evaluate the EA population in ``"batch"`` or
            ``"single"`` mode.
        verbose:
            Whether to pass verbosity flag to the evolutionary algorithm.

        Returns
        -------
        Any
            Best candidate found, or ``None`` if the optimizer returned nothing.
        """
        if mode == "batch":
            objective = _make_acquisition_objective_batch(acquisition_fn)
        else:
            objective = _make_acquisition_objective_single(acquisition_fn)

        return self.evolutionary.evolutionary_best(
            objective,
            self.population_size,
            mutation_rate=self.mutation_rate,
            recombination_rate=self.recombination_rate,
            verbose=verbose,
            fitness_function_mode=mode,
        )
