"""What maximizes the acquisition is the caller's choice: the evolutionary search, or a maximizer.

A pass hands its acquisition to an ``AcquisitionMaximizer``: ``maximize`` for the pick,
``maximize_with_population`` for the pick, the population it came from and a record per
generation.  The evolutionary search is one, through ``AcquisitionOptimizer``; ``SampleMaximizer`` is
the baseline it is measured against, the best of one sample drawn from a sampler and scored under
the same known-point floor.  A maximizer is given instead of the evolutionary search, never beside
it.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
from cosy.core import Synthesizer
from cosy.evolutionary_algorithms import (
    EvolutionarySearch,
    FitnessBasedReplacement,
    Generations,
    InitializationError,
    RankBasedSelection,
    ResolutionMutation,
    SampledInitialization,
    ScalarFitnessComparator,
    SubtreeSwap,
)
from cosy.search import SizeUniformSampler

from bayesian_optimization import BayesianOptimization
from bayesian_optimization.acquisition_function import (
    AcquisitionFunction,
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from bayesian_optimization.acquisition_optimizer import AcquisitionOptimizer, SampleMaximizer
from bayesian_optimization.diagnostics.frontier import GenerationRecord
from bayesian_optimization.runs import read_term_pool, run_search
from tests.integration.test_real_ea_loop import (
    _MAX_SIZE,
    _TARGET,
    _make_ea,
    _objective,
    _repository,
)
from tests.spaces import LIST, list_space
from tests.unit.test_run_driver import SCHEMA, _metrics


def _space() -> Any:
    return Synthesizer(_repository(), {}).construct_solution_space(_TARGET).prune()


def _sampler(seed: int) -> SizeUniformSampler:
    return SizeUniformSampler(_MAX_SIZE, random.Random(seed))


def _loop(space: Any, **kwargs: Any) -> BayesianOptimization:
    return BayesianOptimization(
        space, _TARGET, seed=7, sampler=_sampler(7), n_restarts_kernel_optimizer=0, **kwargs
    )


def _design(bo: BayesianOptimization, size: int = 4) -> None:
    bo.initialize(initial_size=size)
    for _ in range(size):
        suggestion = bo.suggest()
        bo.observe(suggestion.candidate, _objective(suggestion.candidate))


class Recording:
    """A maximizer that answers the first term of a fixed list, keeping what it was handed."""

    def __init__(self, terms: list[Any]) -> None:
        self.terms = terms
        self.handed: list[tuple[AcquisitionFunction, Any, str]] = []

    def maximize(self, acquisition_fn: AcquisitionFunction, query: Any, *, mode: str = "batch") -> Any:
        return self.maximize_with_population(acquisition_fn, query, mode=mode)[0]

    def maximize_with_population(
        self, acquisition_fn: AcquisitionFunction, query: Any, *, mode: str = "batch"
    ) -> tuple[Any, list[Any], list[GenerationRecord]]:
        self.handed.append((acquisition_fn, query, mode))
        record = GenerationRecord(
            generation=0, best=0.5, population_best=0.5, population_mean=0.25,
            population_worst=0.0, distinct_members=len(self.terms), last_improvement=0,
            offspring=0,
        )
        return self.terms[0], list(self.terms), [record]


def test_a_pass_hands_its_acquisition_and_query_to_the_maximizer_and_takes_its_pick():
    space = _space()
    query = _loop(space, optimizer=_make_ea(space, 1)).query
    novel = SampledInitialization(_sampler(99)).initialize(query, 12)
    maximizer = Recording(novel)
    bo = _loop(space, maximizer=maximizer)
    _design(bo)
    observed = set(bo.get_state_snapshot()["x_list"])
    maximizer.terms = [term for term in novel if term not in observed]

    suggestion = bo.suggest(record_population=True)

    acquisition, query, _mode = maximizer.handed[-1]
    assert suggestion.candidate == maximizer.terms[0]
    assert query == bo.query
    assert acquisition is bo.last_acquisition_run.acquisition
    assert bo.last_acquisition_run.population == tuple(maximizer.terms)
    assert [record.generation for record in bo.last_acquisition_run.generations] == [0]


def test_a_maximizer_is_given_instead_of_the_evolutionary_search_never_beside_it():
    space = _space()
    with pytest.raises(ValueError, match="either"):
        _loop(space, optimizer=_make_ea(space, 1), maximizer=Recording([]))
    with pytest.raises(RuntimeError, match="An optimizer is required"):
        _loop(space).check_configuration()


@pytest.mark.parametrize("acquisition", ["ExpectedImprovement", "ProbabilityOfImprovement", "UpperConfidenceBound"])
def test_the_sample_maximizer_never_picks_a_known_point_while_a_novel_one_is_drawn(acquisition):
    space = _space()
    bo = _loop(space, acquisition_function=acquisition, maximizer=SampleMaximizer(_sampler(3), 40))
    _design(bo, size=6)
    for _ in range(4):
        observed = set(bo.get_state_snapshot()["x_list"])
        suggestion = bo.suggest(record_population=True)
        sample = bo.last_acquisition_run.population
        if any(term not in observed for term in sample):
            assert bo.last_acquisition_run.pick not in observed
            assert not suggestion.diagnostics["fallback_used"]
        bo.observe(suggestion.candidate, _objective(suggestion.candidate))


def _generation_zero(space: Any, sampler: SizeUniformSampler, size: int) -> EvolutionarySearch:
    """The evolutionary search that stops after scoring its initial population: the same question."""
    return EvolutionarySearch(
        initializer=SampledInitialization(sampler),
        mutation=ResolutionMutation(_sampler(1), random.Random(2)),
        recombination=SubtreeSwap(random.Random(3), max_size=_MAX_SIZE),
        parent_selection=RankBasedSelection(1.7, rng=random.Random(4)),
        survivor_selection=FitnessBasedReplacement(),
        termination=Generations(0),
        population_size=size,
        crossover_rate=0.9,
        mutation_rate=0.1,
        rng=random.Random(5),
        comparator=ScalarFitnessComparator(greater_is_better=True),
    )


@pytest.mark.parametrize("acquisition", [ExpectedImprovement, ProbabilityOfImprovement, UpperConfidenceBound])
@pytest.mark.parametrize("seed", [11, 12, 13])
def test_the_sample_maximizer_picks_what_the_search_stopped_at_generation_zero_picks(acquisition, seed):
    """cosy's own driver, run for no generation past the first, maximizes over the same sample."""
    space = _space()
    bo = _loop(space, optimizer=_make_ea(space, 1))
    _design(bo, size=5)
    bo.suggest()
    model, incumbent = bo.surrogate, max(bo.get_state_snapshot()["y_list"])
    known = set(bo.get_state_snapshot()["x_list"])
    kwargs = {"beta": 2.0} if acquisition is UpperConfidenceBound else {"incumbent": incumbent}
    af = acquisition(model, known_points=known, **kwargs)

    pick, sample, generations = SampleMaximizer(_sampler(seed), 30).maximize_with_population(af, bo.query)
    oracle = AcquisitionOptimizer(_generation_zero(space, _sampler(seed), 30)).maximize(af, bo.query)

    assert sample == SampledInitialization(_sampler(seed)).initialize(bo.query, 30)
    assert pick == oracle
    assert [record.generation for record in generations] == [0]
    assert generations[0].distinct_members == len(set(sample))


def test_the_sample_maximizer_refuses_what_its_sampler_cannot_hold():
    with pytest.raises(ValueError, match="at least one"):
        SampleMaximizer(_sampler(1), 0)
    space = _space()
    bo = _loop(space, optimizer=_make_ea(space, 1))
    _design(bo)
    bo.suggest()
    af = ExpectedImprovement(bo.surrogate, incumbent=0.0, known_points=set())
    # the space holds 144 terms: a sample of a thousand is refused, not filled with repeats
    with pytest.raises(InitializationError, match="fewer than 1000 inhabitants"):
        SampleMaximizer(_sampler(1), 1000).maximize(af, bo.query)


def test_a_run_through_the_run_layer_logs_the_sample_as_its_one_generation(tmp_path):
    """On the list space the run layer's own tests use, whose terms render without an algebra."""
    bo = BayesianOptimization(
        list_space(), LIST, seed=5, sampler=SizeUniformSampler(6, random.Random(5)),
        maximizer=SampleMaximizer(SizeUniformSampler(6, random.Random(6)), 25),
    )

    run_search(
        bo, _metrics, schema=SCHEMA, csv_path=str(tmp_path / "run.csv"), pretty_algebra=dict,
        echo=lambda line: None, n_design=4, n_passes=3,
    )

    with (tmp_path / "run_ea.csv").open() as handle:
        rows = handle.read().splitlines()[1:]
    assert len(rows) == 3, "one generation per pass"
    _header, records = read_term_pool(tmp_path / "run_terms.pickle")
    assert len(records) == 7
