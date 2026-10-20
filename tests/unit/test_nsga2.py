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
# The default mutation draws its subtrees depth-bounded, and on the infinite expression space a draw
# bounded at the run layer's 1000 levels can grow past any memory; these spaces get a bound that fits.
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


COMPARATOR_MAXIMIZE = (True, False)


def oracle(space, request, design, seed, generations, crossover_rate, mutation_rate, tournament_size=2):
    """cosy's driver on the same components, run ``generations`` generations.

    Returns, per generation from 0: the population, the standing the truncation formed it with
    (None for the design), and from generation 1 the counts (offspring, distinct new terms, places of
    known terms); and the new offspring over the whole run, in the order the driver evaluated them.
    """
    parts = components(seed)
    comparator = ParetoFitnessComparator(maximize=COMPARATOR_MAXIMIZE)
    search = EvolutionarySearch(
        initializer=GivenPopulation(design),
        mutation=parts["mutation"],
        recombination=parts["recombination"],
        parent_selection=CrowdedTournamentSelection(tournament_size, parts["selection_rng"]),
        survivor_selection=CrowdedTruncation(),
        termination=Generations(generations),
        population_size=len(design),
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        rng=parts["rng"],
        comparator=comparator,
    )
    states = list(
        search.evolutionary_stream(
            generator_query(space, request), lambda term: (value_of(term), cost_of(term)), "single"
        )
    )
    evaluated = set(design)
    new_in_order = []
    counts = [None]
    for state in states[1:]:
        known = sum(1 for term in state.offspring if term in evaluated)
        new = [term for term in dict.fromkeys(state.offspring) if term not in evaluated]
        evaluated.update(new)
        new_in_order.extend(new)
        counts.append((len(state.offspring), len(new), known))
    standings = [
        CrowdedTruncation.standing_of(state.population, state.fitness, comparator) for state in states
    ]
    return [list(state.population) for state in states], standings, counts, new_in_order


def ask_tell(space, request, design, seed, generations, crossover_rate, mutation_rate, tournament_size=2):
    """The same run through the loop: the design phase, then one pass per new offspring."""
    search = NSGA2(
        space, request, cost=cost_of, crossover_rate=crossover_rate, mutation_rate=mutation_rate,
        tournament_size=tournament_size, **components(seed),
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


def ranks_of(population) -> list:
    fitness = {term: (value_of(term), cost_of(term)) for term in population}
    rank_of: dict = {}
    comparator = ParetoFitnessComparator(maximize=COMPARATOR_MAXIMIZE)
    for rank, front in enumerate(dominance_fronts(population, fitness, comparator)):
        for member in front:
            rank_of.setdefault(member, rank)
    return [rank_of[term] for term in population]


@pytest.mark.parametrize("space_name", sorted(SPACES))
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize(
    ("mu", "crossover_rate", "mutation_rate", "tournament_size"),
    [(4, 0.9, 0.3, 2), (7, 0.5, 0.8, 2), (5, 0.9, 0.5, 3)],
)
def test_nsga2_breeds_and_keeps_what_cosys_driver_does(
    space_name, seed, mu, crossover_rate, mutation_rate, tournament_size
):
    make, request = SPACES[space_name]
    space = make()
    design = design_of(space, request, mu, seed)
    generations = 4

    search, asked, passes = ask_tell(
        space, request, design, seed, generations, crossover_rate, mutation_rate, tournament_size
    )
    # The ask/tell run stops inside the breeding of the generation after its current population;
    # the driver runs far enough to cover it.
    bred = search.generation + 1
    populations, standings, counts, new_in_order = oracle(
        space, request, design, seed, bred + 2, crossover_rate, mutation_rate, tournament_size
    )

    assert asked == populations[: search.generation + 1]
    # Every new offspring up to the one the last pass evaluated, in the driver's order.
    assert passes == new_in_order[: len(passes)]
    formed = search.formed_generations
    assert [(f.offspring, f.evaluated, f.known) for f in formed[1:]] == counts[1 : search.generation + 1]
    assert search.offspring_known == sum(known for _bred, _new, known in counts[1 : bred + 1])
    # The ranks and distances each population was formed with, the design's ranks alone.
    assert formed[0].ranks == tuple(ranks_of(design)) and formed[0].distances is None
    for record, standing in zip(formed[1:], standings[1 : search.generation + 1], strict=True):
        assert standing is not None
        assert record.ranks == tuple(rank for rank, _distance in standing)
        assert record.distances == tuple(distance for _rank, distance in standing)
    # Not a comparison of two runs that never moved.
    assert passes, "the run bred nothing new"
    assert any(populations[t] != populations[t - 1] for t in range(1, search.generation + 1))
    assert any(rank > 0 for record in formed for rank in record.ranks), "one front tests no ranking"


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


class ScriptedSwap:
    """A recombination whose children are scripted, one batch per call, and copies once the script ends."""

    def __init__(self, script) -> None:
        self.script = [list(batch) for batch in script]
        self.calls = 0

    def recombine(self, query, first, second):
        self.calls += 1
        return self.script.pop(0) if self.script else [first, second]


class NeverMutate:
    def mutate(self, query, term):
        raise AssertionError("mutation_rate 0 never mutates")


def scripted(space, request, design, script, **arguments):
    """NSGA-II on two design terms, one pass per generation, the children the script names."""
    return NSGA2(
        space, request, cost=cost_of, crossover_rate=1.0, mutation_rate=0.0, mutation=NeverMutate(),
        recombination=ScriptedSwap(script), seed=0, **arguments,
    )


def test_idle_generations_stop_a_run_only_in_a_row():
    space = list_space()
    terms = design_of(space, LIST, 5, 11)
    first, second, new = terms[0], terms[1], terms[2:]
    # idle, new, idle, new, idle, new, then idle twice
    script = [(first, second), (new[0], first), (second, first), (new[1], second), (first, first), (new[2], first),
              (first, second), (second, second)]
    search = scripted(space, LIST, [first, second], script, idle_generations=2)
    made = run(search, [first, second], 3)
    assert [suggestion.candidate for suggestion in made] == new
    # three idle generations so far, never two in a row: the run went on; two in a row stop it
    with pytest.raises(RuntimeError, match="2 generations in a row without a term"):
        search.suggest()
    # every bred place counted: 2 + 1 + 2 + 1 + 2 + 1 + 2 + 2
    assert search.offspring_known == 13


def test_the_attempt_cap_is_attempt_factor_passes_per_member():
    space = list_space()
    terms = design_of(space, LIST, 2, 12)
    search = scripted(space, LIST, terms, [()] * 50, attempt_factor=3)
    run(search, terms, 0)
    with pytest.raises(RuntimeError, match="produced 0 of 2 offspring in 6 passes"):
        search.suggest()
    assert search._given[1].calls == 6


def test_a_failed_breeding_leaves_the_population_and_a_retry_breeds_it_again():
    space = list_space()
    terms = design_of(space, LIST, 3, 13)
    new = design_of(space, LIST, 6, 13)[3:]
    # The second call fails: the first generation's breeding had one batch when it failed.
    script = [(new[0], terms[0]), "fail", (new[1], terms[1]), (new[2], terms[2])]

    class FailingOnce(ScriptedSwap):
        def recombine(self, query, first, second):
            if self.script and self.script[0] == list("fail"):
                self.script.pop(0)
                self.calls += 1
                raise KeyboardInterrupt
            return super().recombine(query, first, second)

    search = NSGA2(space, LIST, cost=cost_of, crossover_rate=1.0, mutation_rate=0.0, mutation=NeverMutate(),
                   recombination=FailingOnce(script), seed=0)
    run(search, terms, 0)
    with pytest.raises(KeyboardInterrupt):
        search.suggest()
    assert search.generation == 0
    assert [record.generation for record in search.formed_generations] == [0]
    assert search.get_state_snapshot()["state"] == BOState.INITIALIZED.value
    # The retry breeds the design again, from where the generators stand: no population was formed.
    suggestion = search.suggest()
    assert suggestion.diagnostics["generation"] == 1
    assert [record.generation for record in search.formed_generations] == [0]


def test_the_capacity_stays_one():
    search = NSGA2(list_space(), LIST, cost=cost_of, depth_bound=DEPTH, seed=14)
    assert search.max_outstanding == 1
    with pytest.raises(ValueError, match="one suggestion is outstanding at a time"):
        search.max_outstanding = 2
    search.max_outstanding = 1
    assert search.max_outstanding == 1


def test_finalize_forms_the_population_a_complete_generation_earned():
    space = list_space()
    terms = design_of(space, LIST, 2, 15)
    new = design_of(space, LIST, 4, 15)[2:]
    search = scripted(space, LIST, terms, [(new[0], terms[0]), (new[1], terms[1])])
    run(search, terms, 2)
    # Both generations' offspring have their values; the second population is formed by the next suggest.
    assert [record.generation for record in search.formed_generations] == [0, 1]
    search.finalize()
    assert [record.generation for record in search.formed_generations] == [0, 1, 2]
    assert search.formed_generations[2].evaluated == 1


def test_finalize_forms_nothing_while_a_generation_awaits_values():
    space = list_space()
    terms = design_of(space, LIST, 2, 16)
    new = design_of(space, LIST, 5, 16)[2:]
    # The first generation breeds two new terms, and the run ends after one of them.
    search = scripted(space, LIST, terms, [(new[0], new[1])])
    run(search, terms, 1)
    search.finalize()
    assert [record.generation for record in search.formed_generations] == [0]


def test_a_design_with_a_nan_cost_is_refused_before_the_objective_reads_it():
    space = list_space()
    terms = design_of(space, LIST, 5, 17)
    victim = terms[3]
    measured = []

    def objective(term):
        measured.append(term)
        return value_of(term)

    search = NSGA2(space, LIST, cost=lambda t: math.nan if t == victim else cost_of(t), depth_bound=DEPTH, seed=17)
    with pytest.raises(ValueError, match="cost returned nan"):
        search.initialize(objective=objective, x0=terms)
    assert measured == []
    assert search.get_state_snapshot()["state"] == BOState.UNINITIALIZED.value
    with pytest.raises(ValueError, match="cost returned nan"):
        search.initialize(x0=terms, y0=[value_of(term) for term in terms])
    assert search.get_state_snapshot()["state"] == BOState.UNINITIALIZED.value


def test_a_drawn_design_with_a_nan_cost_is_refused_before_the_objective_reads_it():
    space = list_space()
    sampler = SizeUniformSampler(6, random.Random(18))
    measured = []

    def objective(term):
        measured.append(term)
        return value_of(term)

    search = NSGA2(space, LIST, cost=lambda t: math.nan if t.size == 3 else cost_of(t), depth_bound=DEPTH,
                   sampler=sampler, seed=18)
    with pytest.raises(ValueError, match="cost returned nan"):
        search.initialize(objective=objective, initial_size=12)
    assert measured == []


def test_a_repeated_design_term_is_named_by_its_first_row():
    space = list_space()
    a, b, c = design_of(space, LIST, 3, 19)
    search = NSGA2(space, LIST, cost=cost_of, depth_bound=DEPTH, seed=19, mutation_rate=0.5)
    search.initialize(x0=[a, b, a, c], y0=[value_of(a), value_of(b), value_of(a), value_of(c)])
    search.suggest()
    assert search.formed_generations[0].rows == (0, 1, 3)
    assert search.population == (a, b, c)
