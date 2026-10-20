"""NSGA-II as a strategy of the ask/tell loop.

The oracle is cosy's own driver: ``EvolutionarySearch.evolutionary_stream`` with the same components,
the same seeds and the design as its initial population, its fitness function answering at once.
The ask/tell run, fed the same values one pass at a time, has to evaluate the same new offspring in
the same order and form the same populations, generation by generation.  Everything else here is
what the loop adds around that: the design as the first population, the dataset answering what it
holds, the cost refused unless it is a finite number, the state machine.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter

import pytest
from cosy.core import SpecificationBuilder, Synthesizer
from cosy.evolutionary_algorithms import (
    CrowdedTournamentSelection,
    CrowdedTruncation,
    EvolutionarySearch,
    Generations,
    ParetoFitnessComparator,
    ResolutionMutation,
    SubtreeSwap,
    dominance_fronts,
)
from cosy.search import DepthBoundedRandomSampler, SizeUniformSampler, generator_query

from bayesian_optimization import NSGA2, BOState
from bayesian_optimization import nsga2 as nsga2_module
from bayesian_optimization.initial_sampling import distinct_prefix
from bayesian_optimization.runs import acquisition as run_acquisition
from bayesian_optimization.runs import search_program
from tests.spaces import EXPR, LIST, expression_space, list_space, lit

SPACES = {"expressions": (expression_space, EXPR), "lists": (list_space, LIST)}
# The default mutation draws its subtrees depth-bounded; on the infinite expression space a draw
# bounded at the run layer's 1000 levels fails often enough to end variation at the attempt cap.
DEPTH = 5


def rendering(term) -> str:
    """The term by the names of its symbols: ``str`` of these spaces' terms carries the function
    objects' addresses, which differ from one process to the next."""
    root = getattr(term.root, "__name__", repr(term.root))
    return f"{root}({','.join(rendering(child) for child in term.children)})"


def value_of(term) -> float:
    """The stand-in for a measurement: deterministic, spread over [0, 1), unrelated to the cost."""
    digest = hashlib.sha256(rendering(term).encode()).digest()
    return int.from_bytes(digest[:6], "big") / 2**48


def cost_of(term) -> float:
    return float(term.size)


def components(seed: int) -> dict:
    """Fresh variation components, every generator derived from ``seed``: two calls build equal twins."""
    return {
        "mutation": ResolutionMutation(
            DepthBoundedRandomSampler(4, random.Random(seed + 1)), random.Random(seed + 2)
        ),
        "recombination": SubtreeSwap(random.Random(seed + 3)),
        "selection_rng": random.Random(seed + 4),
        "rng": random.Random(seed + 6),
    }


def design_of(space, request, size: int, seed: int) -> list:
    terms, repeats = distinct_prefix(
        SizeUniformSampler(9, random.Random(seed)), generator_query(space, request), size
    )
    assert repeats == 0
    return terms


class GivenPopulation:
    def __init__(self, terms) -> None:
        self.terms = list(terms)

    def initialize(self, query, size):
        assert size == len(self.terms)
        return list(self.terms)


def oracle(space, request, design, seed, generations, crossover_rate, mutation_rate):
    """cosy's driver on the same components: the populations, and the new offspring in evaluation order."""
    parts = components(seed)
    search = EvolutionarySearch(
        initializer=GivenPopulation(design),
        mutation=parts["mutation"],
        recombination=parts["recombination"],
        parent_selection=CrowdedTournamentSelection(2, parts["selection_rng"]),
        survivor_selection=CrowdedTruncation(),
        termination=Generations(generations),
        population_size=len(design),
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        rng=parts["rng"],
        comparator=ParetoFitnessComparator(maximize=(True, False)),
    )
    states = list(
        search.evolutionary_stream(
            generator_query(space, request), lambda term: (value_of(term), cost_of(term)), "single"
        )
    )
    evaluated = set(design)
    new_in_order = []
    counts = []  # per generation from 1: (offspring, distinct new terms, places of known terms)
    for state in states[1:]:
        known = sum(1 for term in state.offspring if term in evaluated)
        new = [term for term in dict.fromkeys(state.offspring) if term not in evaluated]
        evaluated.update(new)
        new_in_order.extend(new)
        counts.append((len(state.offspring), len(new), known))
    return [list(state.population) for state in states], new_in_order, counts


def ask_tell(space, request, design, seed, generations, crossover_rate, mutation_rate):
    """The same run through the loop: the design phase, then one pass per new offspring."""
    search = NSGA2(
        space, request, cost=cost_of, crossover_rate=crossover_rate, mutation_rate=mutation_rate,
        **components(seed),
    )
    search.initialize(design=design)
    passes = []
    for _ in design:
        suggestion = search.suggest()
        search.observe(suggestion.candidate, value_of(suggestion.candidate))
    while search.generation < generations:
        suggestion = search.suggest()
        passes.append(suggestion.candidate)
        search.observe(suggestion.candidate, value_of(suggestion.candidate))
    rows = search.get_state_snapshot()["x_list"]
    populations = [[rows[row] for row in formed.rows] for formed in search.formed_generations]
    return search, populations, passes


@pytest.mark.parametrize("space_name", sorted(SPACES))
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize(("mu", "crossover_rate", "mutation_rate"), [(4, 0.9, 0.3), (7, 0.5, 0.8)])
def test_nsga2_breeds_and_keeps_what_cosys_driver_does(space_name, seed, mu, crossover_rate, mutation_rate):
    make, request = SPACES[space_name]
    space = make()
    design = design_of(space, request, mu, seed)
    generations = 4

    populations, new_in_order, bred = oracle(space, request, design, seed, generations, crossover_rate, mutation_rate)
    search, asked_populations, passes = ask_tell(
        space, request, design, seed, generations, crossover_rate, mutation_rate
    )

    assert asked_populations[: generations + 1] == populations
    # The passes of the generations the oracle ran, and the first offspring of the next one.
    assert passes[: len(new_in_order)] == new_in_order
    assert [
        (formed.offspring, formed.evaluated, formed.known)
        for formed in search.formed_generations[1 : generations + 1]
    ] == bred
    # Not a comparison of two runs that never moved.
    assert new_in_order, "the run bred nothing new"
    assert any(populations[t] != populations[t - 1] for t in range(1, generations + 1))


def run(search: NSGA2, design, passes: int, value=value_of) -> list:
    search.initialize(design=design)
    for _ in design:
        suggestion = search.suggest()
        search.observe(suggestion.candidate, value(suggestion.candidate))
    made = []
    for _ in range(passes):
        suggestion = search.suggest()
        made.append(suggestion)
        search.observe(suggestion.candidate, value(suggestion.candidate))
    return made


def test_the_design_is_the_first_population_and_ranked_alone():
    space = expression_space()
    design = design_of(space, EXPR, 6, 0)
    search = NSGA2(space, EXPR, cost=cost_of, depth_bound=DEPTH, seed=0)
    run(search, design, 1)

    first = search.formed_generations[0]
    assert first.generation == 0
    assert first.rows == tuple(range(6))
    assert first.distances is None
    fitness = {term: (value_of(term), cost_of(term)) for term in design}
    expected = {}
    for rank, front in enumerate(dominance_fronts(design, fitness, ParetoFitnessComparator(maximize=(True, False)))):
        for member in front:
            expected.setdefault(member, rank)
    assert first.ranks == tuple(expected[term] for term in design)
    assert len(set(first.ranks)) > 1, "a design on one front tests no ranking"


def test_a_formed_population_carries_the_standing_its_truncation_gave_it():
    space = expression_space()
    design = design_of(space, EXPR, 6, 1)
    search = NSGA2(space, EXPR, cost=cost_of, depth_bound=DEPTH, seed=1, mutation_rate=0.3)
    run(search, design, 30)
    formed = search.formed_generations[1:]
    assert formed, "no population was formed"
    for record in formed:
        assert record.distances is not None
        assert len(record.distances) == len(record.rows) == 6
        assert any(math.isinf(distance) for distance in record.distances), "a front's boundary is infinite"


def test_an_offspring_the_dataset_holds_is_answered_and_never_handed_out():
    space = list_space()
    design = design_of(space, LIST, 5, 2)
    search = NSGA2(space, LIST, cost=cost_of, depth_bound=DEPTH, seed=2, crossover_rate=0.5, mutation_rate=0.2)
    made = run(search, design, 25)

    terms = [suggestion.candidate for suggestion in made]
    assert len(set(terms)) == len(terms)
    assert not set(terms) & set(design)
    assert search.offspring_known > 0, "every offspring was new, so nothing was answered from the dataset"
    passes_per_generation = Counter(suggestion.diagnostics["generation"] for suggestion in made)
    for record in search.formed_generations[1:]:
        assert record.evaluated == passes_per_generation[record.generation]
        assert record.known + record.evaluated <= record.offspring


def test_a_pass_is_an_offspring_of_its_generation():
    space = expression_space()
    design = design_of(space, EXPR, 4, 4)
    search = NSGA2(space, EXPR, cost=cost_of, depth_bound=DEPTH, seed=4, mutation_rate=0.3)
    search.initialize(design=design)
    for _ in design:
        suggestion = search.suggest()
        assert suggestion.diagnostics["phase"] == "design"
        search.observe(suggestion.candidate, value_of(suggestion.candidate))
    generations = []
    for _ in range(12):
        suggestion = search.suggest()
        assert suggestion.acquisition_value is None
        assert suggestion.diagnostics["phase"] == "main"
        # bred from the current population, for the next one
        assert suggestion.diagnostics["generation"] == search.generation + 1
        generations.append(suggestion.diagnostics["generation"])
        search.observe(suggestion.candidate, value_of(suggestion.candidate))
    assert generations[0] == 1
    assert len(set(generations)) > 1, "twelve passes of one generation test no numbering"
    assert NSGA2.PASS_PHASE == "nsga2_offspring"
    assert search.max_outstanding == 1


def test_a_second_run_after_reset_repeats_the_first():
    space = expression_space()
    design = design_of(space, EXPR, 5, 5)
    search = NSGA2(space, EXPR, cost=cost_of, depth_bound=DEPTH, seed=5, mutation_rate=0.3)
    first = [suggestion.candidate for suggestion in run(search, design, 12)]
    search.reset()
    assert search.formed_generations == ()
    assert search.population == ()
    second = [suggestion.candidate for suggestion in run(search, design, 12)]
    assert first == second

    other = NSGA2(space, EXPR, cost=cost_of, depth_bound=DEPTH, seed=6, mutation_rate=0.3)
    assert [suggestion.candidate for suggestion in run(other, design, 12)] != first


def test_a_converged_population_stops_the_run():
    """A space with one term: every offspring is the design's term, so no generation breeds anything new."""
    space = Synthesizer({lit: SpecificationBuilder().suffix(EXPR)}).construct_solution_space(EXPR)
    design = design_of(space, EXPR, 1, 0)
    search = NSGA2(space, EXPR, cost=cost_of, depth_bound=DEPTH, seed=0, idle_generations=3)
    search.initialize(design=design)
    suggestion = search.suggest()
    search.observe(suggestion.candidate, 0.5)
    with pytest.raises(RuntimeError, match="3 generations in a row without a term"):
        search.suggest()
    assert search.generation == 2, "the third idle generation is bred, not formed"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, "3", None, True])
def test_a_design_term_without_a_finite_cost_is_refused_before_it_is_measured(bad):
    space = expression_space()
    design = design_of(space, EXPR, 4, 8)
    victim = design[2]

    def cost(term):
        return bad if term == victim else cost_of(term)

    search = NSGA2(space, EXPR, cost=cost, depth_bound=DEPTH, seed=8)
    with pytest.raises(ValueError, match="cost returned .* NSGA-II minimizes an exact cost"):
        search.initialize(design=design)
    assert search.get_state_snapshot()["state"] == BOState.UNINITIALIZED.value


def test_an_offspring_without_a_finite_cost_is_refused_before_it_is_handed_out():
    space = expression_space()
    design = design_of(space, EXPR, 4, 9)

    def cost(term):
        return cost_of(term) if term in design else math.nan

    search = NSGA2(space, EXPR, cost=cost, depth_bound=DEPTH, seed=9, mutation_rate=0.3)
    search.initialize(design=design)
    for _ in design:
        suggestion = search.suggest()
        search.observe(suggestion.candidate, value_of(suggestion.candidate))
    with pytest.raises(ValueError, match="cost returned nan"):
        search.suggest()
    # The refused offspring stays the next one: a retry meets it again rather than skipping it.
    with pytest.raises(ValueError, match="cost returned nan"):
        search.suggest()
    snapshot = search.get_state_snapshot()
    assert snapshot["state"] == BOState.INITIALIZED.value
    assert snapshot["outstanding"] == []
    assert search.finalize()["iterations"] == 0


@pytest.mark.parametrize(
    "arguments",
    [
        {"crossover_rate": 1.5},
        {"crossover_rate": -0.1},
        {"crossover_rate": True},
        {"mutation_rate": 2},
        {"mutation_rate": "0.1"},
        {"tournament_size": 0},
        {"attempt_factor": 0},
        {"idle_generations": 0},
        {"depth_bound": 0},
        {"depth_bound": 2.5},
    ],
)
def test_a_configuration_outside_its_range_is_refused_at_construction(arguments):
    with pytest.raises(ValueError, match="is a p"):
        NSGA2(expression_space(), EXPR, cost=cost_of, **arguments)


def test_a_cost_that_is_not_callable_is_refused():
    with pytest.raises(TypeError, match="cost is the exact cost of a term"):
        NSGA2(expression_space(), EXPR, cost=3.0)


def test_a_loop_without_a_space_is_refused():
    with pytest.raises(ValueError, match="variation poses queries to the search space"):
        NSGA2(None, None, cost=cost_of)


def test_the_defaults_are_the_acquisition_optimizers_and_the_run_layers():
    assert nsga2_module.DEFAULT_CROSSOVER_RATE == run_acquisition.DEFAULT_CROSSOVER_RATE
    assert nsga2_module.DEFAULT_MUTATION_RATE == run_acquisition.DEFAULT_MUTATION_RATE
    assert nsga2_module.DEFAULT_DEPTH_BOUND == search_program.DEFAULT_DEPTH_BOUND
    assert nsga2_module.DEFAULT_TOURNAMENT_SIZE == 2
