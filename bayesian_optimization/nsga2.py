"""NSGA-II as a strategy of the ask/tell loop: one offspring evaluated per pass.

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, "A fast and elitist multiobjective genetic algorithm:
NSGA-II", IEEE TEVC 6(2), 2002) keeps a population of ``mu``, breeds ``mu`` offspring from it by binary
tournaments under the crowded comparison, and forms the next population from parents and offspring
together, whole fronts first and the last front cut by crowding distance (§III-C).  cosy holds every
part of that: the two components :class:`~cosy.evolutionary_algorithms.CrowdedTournamentSelection`
and :class:`~cosy.evolutionary_algorithms.CrowdedTruncation`, the variation of its driver
:class:`~cosy.evolutionary_algorithms.EvolutionarySearch`, and the order
:class:`~cosy.evolutionary_algorithms.ParetoFitnessComparator`.  The driver evaluates a generation by
calling a fitness function; here an evaluation is a pass of the loop, a term handed out by
``suggest`` and its value taken back by ``observe``, so :class:`NSGA2` runs the driver's generation
step in that rhythm.  The variation is the driver's own method, so the operators and the order in
which they draw are cosy's code and not a copy; what is written here is the bookkeeping around it,
and a test holds it against the driver run on the same components, seeds and values.

**The two objectives.**  The loop's value, which it maximizes and which is measured, and an exact
cost ``cost(term)``, which is minimized and computed, never modelled.  A term's fitness is the pair
``(value, cost)`` under ``ParetoFitnessComparator(maximize=(True, False))``.

**The generations.**  The first population is the design: the dataset when the passes begin, every
distinct term in the order it was observed, so ``mu`` is the design's size.  Each generation's
offspring are bred as the driver breeds them.  A pass of its variation yields two offspring whether
it crosses or copies, so a generation holds ``mu`` of them for an even ``mu`` and ``mu + 1`` for an
odd one, where Deb et al. breed exactly ``N`` (§III-C, p. 186); the extra one competes in the
truncation like the others.  Those the dataset does not hold go out one per pass, in the order the
driver would have evaluated them; an offspring the dataset holds is answered from it at no cost, as
the driver answers it from its cache.  When every offspring has its value, the next population is
the truncation of parents and offspring, and the next generation's tournaments read the rank and the
distance that truncation formed it with.  A generation's boundary needs every value of its
offspring, so one suggestion is outstanding at a time.

**A failure leaves the run where it was.**  Breeding commits nothing until it has succeeded, so a
``suggest`` that raises inside it leaves the population and the queue as they were, and the next
``suggest`` breeds the same population again, from where the generators stand.

**What a run of it records.**  :attr:`formed_generations` holds one :class:`FormedGeneration` per
population, the first one included: its members, their dataset rows, and the rank and distance each
was formed with.
"""

from __future__ import annotations

import math
import numbers
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from cosy.core.solution_space import SolutionSpace
from cosy.evolutionary_algorithms import (
    CrowdedTournamentSelection,
    CrowdedTruncation,
    EvolutionarySearch,
    Generations,
    Initializer,
    Mutation,
    ParetoFitnessComparator,
    Recombination,
    ResolutionMutation,
    SubtreeSwap,
    dominance_fronts,
)
from cosy.search import DepthBoundedRandomSampler, Sampler

from .diagnostics import enable_verbose_logging, get_logger
from .loop import NT, AskTellLoop, G, T
from .state import Diagnostics, Suggestion

#: The crossover and the mutation rate, the acquisition optimizer's (``runs.acquisition``): the
#: operators and the terms are the same, so the reasons given there for the two numbers hold here.
DEFAULT_CROSSOVER_RATE = 0.9
DEFAULT_MUTATION_RATE = 0.03
#: The depth bound of the mutation's sampler, the run layer's (``runs.search_program``), chosen for
#: its finite network spaces.  On a recursive space whose terms can grow without bound, a draw
#: bounded at 1000 levels can grow past any memory: pass a bound that fits the space.
DEFAULT_DEPTH_BOUND = 1000
#: Deb et al. select parents by binary tournaments (§III-C, pp. 185 - 186).
DEFAULT_TOURNAMENT_SIZE = 2
#: How many generations in a row may breed no term the dataset does not hold before the run stops.
DEFAULT_IDLE_GENERATIONS = 20

# Per component an offset of the base seed, as ``runs.acquisition.build_acquisition_optimizer``
# derives them, so that one component's consumption shifts no other component's draws.
_MUTATION_SAMPLER, _MUTATION, _RECOMBINATION, _SELECTION, _RATES = 1, 2, 3, 4, 6


@dataclass(frozen=True)
class FormedGeneration:
    """One population of an NSGA-II run.

    Attributes:
        generation (int): The generation; 0 is the design.
        members (tuple[Any, ...]): The population, in its order.
        rows (tuple[int, ...]): Per place, the dataset row its member was observed in.
        ranks (tuple[int, ...]): Per place, the rank of its front, 0 for the first: among the fronts
            of parents and offspring together, as the truncation formed the population; for the
            design, among its own members.
        distances (tuple[float, ...] | None): Per place, the crowding distance it was formed with,
            within its whole front.  None for generation 0, which no truncation formed: its
            tournaments compare the rank alone (§III-C, "the usual binary tournament selection").
        offspring (int): How many offspring the previous population bred for this one; 0 for the
            design.
        evaluated (int): How many distinct terms among them the dataset did not hold, each one
            pass of the loop.
        known (int): How many of the offspring's places held a term the dataset did hold,
            answered at no cost; a term bred twice counts twice.
    """

    generation: int
    members: tuple[Any, ...]
    rows: tuple[int, ...]
    ranks: tuple[int, ...]
    distances: tuple[float, ...] | None
    offspring: int
    evaluated: int
    known: int


class _GivenPopulation:
    """The initializer the driver is built with; it is never asked, the design being the first population."""

    def __init__(self, population: Sequence[Any]) -> None:
        self._population = list(population)

    def initialize(self, query: Any, size: int) -> list[Any]:
        return list(self._population)


def _require_rate(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not 0.0 <= float(value) <= 1.0:
        msg = f"{name} is a probability, a number in [0, 1], not {value!r}"
        raise ValueError(msg)
    return float(value)


def _require_count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        msg = f"{name} is a positive count, not {value!r}"
        raise ValueError(msg)
    count: int = value
    return count


class NSGA2(AskTellLoop[NT, T, G]):
    """NSGA-II over a CoSy solution space, through the same ask/tell loop as BO and random search.

    Parameters
    ----------
    search_space, request, seed, sampler, initializer:
        As for :class:`~bayesian_optimization.loop.AskTellLoop`; the sampler and the initializer
        draw a design that is not handed over.  A search space is required: variation poses
        queries to it.
    cost:
        The exact cost of a term, minimized; a finite real number for every term the run meets.
    crossover_rate, mutation_rate:
        p_c and p_m of the driver: the crossover draw per pass, the mutation draw per offspring.
    depth_bound:
        The bound of the depth-bounded sampler the default mutation draws its subtrees from; see
        :data:`DEFAULT_DEPTH_BOUND` for a space whose terms can grow without bound.
    tournament_size:
        Participants per tournament of the crowded tournament selection.
    attempt_factor:
        The driver's cap on variation passes per generation, as a multiple of ``mu``.
    idle_generations:
        How many generations in a row may breed nothing the dataset does not hold before
        ``suggest`` raises: a population that only reproduces terms already measured has
        converged, and saying so is the result, where a substitute draw would hide it.
    mutation, recombination, rng, selection_rng:
        The variation's components and generators, each built from ``seed`` where it is not
        given: ``ResolutionMutation`` over a ``DepthBoundedRandomSampler``, ``SubtreeSwap``, the
        generator of the driver's two rate draws, and the tournament's.  A component handed over
        keeps the random state it reaches, as the loop's own initializer does across ``reset()``.
    """

    #: An evaluation of an offspring, in a run's rows and term records.
    PASS_PHASE: ClassVar[str] = "nsga2_offspring"

    def __init__(
        self,
        search_space: SolutionSpace[NT, T, G] | None,
        request: NT | None = None,
        *,
        cost: Callable[[Any], float],
        seed: int | None = None,
        sampler: Sampler | None = None,
        initializer: Initializer[NT, T, G] | None = None,
        crossover_rate: float = DEFAULT_CROSSOVER_RATE,
        mutation_rate: float = DEFAULT_MUTATION_RATE,
        depth_bound: int = DEFAULT_DEPTH_BOUND,
        tournament_size: int = DEFAULT_TOURNAMENT_SIZE,
        attempt_factor: int = 10,
        idle_generations: int = DEFAULT_IDLE_GENERATIONS,
        mutation: Mutation[NT, T, G] | None = None,
        recombination: Recombination[NT, T, G] | None = None,
        rng: random.Random | None = None,
        selection_rng: random.Random | None = None,
    ) -> None:
        if search_space is None:
            raise ValueError(
                "NSGA-II varies terms, and variation poses queries to the search space, so it "
                "needs one; the loop without a space serves strategies that draw nothing"
            )
        super().__init__(
            search_space, request, initializer=initializer, seed=seed, sampler=sampler,
            max_outstanding=1,
        )
        if not callable(cost):
            msg = f"cost is the exact cost of a term, a callable, not {cost!r}"
            raise TypeError(msg)
        self.cost = cost
        self.crossover_rate = _require_rate(crossover_rate, "crossover_rate")
        self.mutation_rate = _require_rate(mutation_rate, "mutation_rate")
        self.depth_bound = _require_count(depth_bound, "depth_bound")
        self.tournament_size = _require_count(tournament_size, "tournament_size")
        self.attempt_factor = _require_count(attempt_factor, "attempt_factor")
        self.idle_generations = _require_count(idle_generations, "idle_generations")
        self._given = (mutation, recombination, rng, selection_rng)
        self._comparator = ParetoFitnessComparator(maximize=(True, False))
        self._truncation: CrowdedTruncation[T] = CrowdedTruncation()
        self._nsga_logger = get_logger("nsga2")
        self._clear_run()

    @property
    def max_outstanding(self) -> int:
        """One: a generation's boundary needs the value of every offspring of the generation."""
        return 1

    @max_outstanding.setter
    def max_outstanding(self, value: int) -> None:
        if value != 1 or isinstance(value, bool):
            msg = (
                f"NSGA-II forms a population from the values of all of a generation's offspring, so "
                f"one suggestion is outstanding at a time, not {value!r}"
            )
            raise ValueError(msg)

    # -------------------------------------------------------------------------
    # The run's own state
    # -------------------------------------------------------------------------

    def _clear_run(self) -> None:
        self._search: EvolutionarySearch[NT, T, G] | None = None
        # One fitness tuple per term, built once: the truncation's standing is read back only for
        # the very value objects it was formed from, as the driver's cache hands them over.
        self._fitness: dict[Any, tuple[float, float]] = {}
        self._costs: dict[Any, float] = {}
        self._row_of: dict[Any, int] = {}
        self._population: list[Any] | None = None
        self._population_fitness: dict[Any, tuple[float, float]] = {}
        self._offspring: list[Any] = []
        self._waiting: list[Any] = []
        # Whether the current population has bred its offspring; a population is formed unbred.
        self._bred_current = False
        self._bred: tuple[int, int, int] = (0, 0, 0)
        # Generations bred in a row without a term the dataset does not hold.
        self._idle = 0
        self._generation = 0
        self._formed: list[FormedGeneration] = []
        self._known_total = 0

    def reset(self) -> None:
        """Reset to UNINITIALIZED and drop the run: its populations, values and costs.

        The default components are built again from ``seed`` at the next run's first pass, so a
        second run repeats the first; a component handed to the constructor keeps its state.
        """
        super().reset()
        self._clear_run()

    def _components(
        self,
    ) -> tuple[Mutation[NT, T, G], Recombination[NT, T, G], random.Random, CrowdedTournamentSelection[T]]:
        mutation, recombination, rng, selection_rng = self._given

        def derived(offset: int) -> random.Random:
            return random.Random(None if self.seed is None else self.seed * 100 + offset)

        if mutation is None:
            mutation = ResolutionMutation(
                DepthBoundedRandomSampler(self.depth_bound, derived(_MUTATION_SAMPLER)), derived(_MUTATION)
            )
        if recombination is None:
            recombination = SubtreeSwap(derived(_RECOMBINATION))
        selection: CrowdedTournamentSelection[T] = CrowdedTournamentSelection(
            self.tournament_size, selection_rng if selection_rng is not None else derived(_SELECTION)
        )
        return mutation, recombination, rng if rng is not None else derived(_RATES), selection

    # -------------------------------------------------------------------------
    # Read-only views of the run
    # -------------------------------------------------------------------------

    @property
    def generation(self) -> int:
        """The generation of the current population; 0 before and during the first generation's breeding."""
        return self._generation

    @property
    def population(self) -> tuple[Any, ...]:
        """The current population, empty before the first pass."""
        return tuple(self._population or ())

    @property
    def formed_generations(self) -> tuple[FormedGeneration, ...]:
        """One record per population the run formed, the design first."""
        return tuple(self._formed)

    @property
    def offspring_known(self) -> int:
        """How many bred offspring were answered from the dataset over the run, at no cost."""
        return self._known_total

    # -------------------------------------------------------------------------
    # The cost
    # -------------------------------------------------------------------------

    def _cost_of(self, term: Any) -> float:
        """The exact cost of a term, computed once and refused unless it is a finite number."""
        known = self._costs.get(term)
        if known is not None:
            return known
        value = self.cost(term)
        if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(float(value)):
            msg = (
                f"cost returned {value!r} for {term}: NSGA-II minimizes an exact cost, and a value "
                "that is not a finite number has no place in the order; check the function passed "
                "as cost="
            )
            raise ValueError(msg)
        self._costs[term] = float(value)
        return float(value)

    def check_design(self, terms: Sequence[Any]) -> None:
        """Refuse a design whose cost cannot be read, before a value of it is measured.

        Args:
            terms (Sequence[Any]): The design.

        Raises:
            ValueError: If a term's cost is not a finite number, naming the term.
        """
        for term in terms:
            self._cost_of(term)

    # -------------------------------------------------------------------------
    # The generations
    # -------------------------------------------------------------------------

    def _begin(self) -> None:
        """Make the design the first population, unbred."""
        terms, values = self._distinct_pairs()
        if not terms:
            raise RuntimeError(
                "NSGA-II breeds its first generation from the design, and the dataset holds no term"
            )
        for row, term in enumerate(self._x_list):
            self._row_of.setdefault(term, row)
        for term, value in zip(terms, values, strict=True):
            self._fitness[term] = (value, self._cost_of(term))
        mutation, recombination, rng, selection = self._components()
        self._search = EvolutionarySearch(
            initializer=_GivenPopulation(terms),
            mutation=mutation,
            recombination=recombination,
            parent_selection=selection,
            survivor_selection=self._truncation,
            termination=Generations(0),
            population_size=len(terms),
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            rng=rng,
            comparator=self._comparator,
            attempt_factor=self.attempt_factor,
        )
        self._population = list(terms)
        self._population_fitness = {term: self._fitness[term] for term in self._population}
        self._generation = 0
        rank_of: dict[Any, int] = {}
        for rank, front in enumerate(dominance_fronts(self._population, self._population_fitness, self._comparator)):
            for member in front:
                rank_of.setdefault(member, rank)
        self._record(tuple(rank_of[term] for term in self._population), None)

    def _breed(self) -> None:
        """Breed the next generation's offspring as the driver breeds them, and queue the new ones.

        Nothing of the run changes until the breeding has succeeded: a failure inside variation
        leaves the population unbred, and the next call breeds it again.
        """
        search = self._search
        assert search is not None and self._population is not None
        query = self._query()
        offspring: list[Any] = []
        attempts = 0
        cap = search.attempt_factor * search.population_size
        while len(offspring) < search.population_size:
            if attempts >= cap:
                msg = (
                    f"generation {self._generation + 1} produced {len(offspring)} of "
                    f"{search.population_size} offspring in {attempts} passes. Variation is "
                    "failing systematically. The parents may have grown past the sampler's bound, "
                    "or the recombination operator may find no acceptable pair."
                )
                raise RuntimeError(msg)
            attempts += 1
            offspring.extend(search._variation_pass(query, self._population, self._population_fitness))
        # In the order the driver's cache would ask for them: the first place of each term it lacks.
        waiting = [term for term in dict.fromkeys(offspring) if term not in self._fitness]
        known = sum(1 for term in offspring if term in self._fitness)
        self._offspring = offspring
        self._waiting = waiting
        self._bred = (len(offspring), len(waiting), known)
        self._known_total += known
        self._idle = 0 if waiting else self._idle + 1
        self._bred_current = True

    def _form_next_population(self) -> None:
        """Truncate parents and offspring to the next population, as the driver does."""
        search = self._search
        assert search is not None and self._population is not None
        parents, offspring = self._population, self._offspring
        pool_fitness = {term: self._fitness[term] for term in (*parents, *offspring)}
        population = self._truncation.select_survivors(
            parents, offspring, pool_fitness, self._comparator, search.population_size
        )
        if len(population) != search.population_size:
            msg = (
                f"CrowdedTruncation returned {len(population)} individuals for a population of "
                f"{search.population_size}"
            )
            raise ValueError(msg)
        if any(term not in pool_fitness for term in population):
            raise ValueError("CrowdedTruncation returned an individual it was not given")
        population_fitness = {term: self._fitness[term] for term in population}
        standing = CrowdedTruncation.standing_of(population, population_fitness, self._comparator)
        if standing is None:
            raise RuntimeError("the population CrowdedTruncation formed does not carry its standing")
        self._population = population
        self._population_fitness = population_fitness
        self._generation += 1
        self._bred_current = False
        self._record(tuple(rank for rank, _distance in standing), tuple(distance for _rank, distance in standing))

    def _record(self, ranks: tuple[int, ...], distances: tuple[float, ...] | None) -> None:
        population = self._population or []
        offspring, evaluated, known = self._bred if self._generation > 0 else (0, 0, 0)
        self._formed.append(
            FormedGeneration(
                generation=self._generation,
                members=tuple(population),
                rows=tuple(self._row_of[term] for term in population),
                ranks=ranks,
                distances=distances,
                offspring=offspring,
                evaluated=evaluated,
                known=known,
            )
        )

    def _propose(self, verbose: bool) -> Suggestion:
        """The next offspring the dataset does not hold, breeding generations until there is one.

        Raises:
            RuntimeError: If ``idle_generations`` generations in a row breed no new term, or if
                variation fails the driver's attempt cap.
            ValueError: If the offspring's cost is not a finite number, before it is handed out.
        """
        if verbose:
            enable_verbose_logging()
        if self._population is None:
            self._begin()
        while True:
            if not self._bred_current:
                self._breed()
            if self._waiting:
                break
            # One suggestion is outstanding at a time, so an empty queue means every offspring of
            # the generation has its value and the next population can be formed.
            if self._idle >= self.idle_generations:
                msg = (
                    f"NSGA-II bred {self.idle_generations} generations in a row without a term the "
                    f"dataset does not hold (generation {self._generation}): the population "
                    "reproduces what it has measured, which is convergence, not a pass to make"
                )
                raise RuntimeError(msg)
            self._form_next_population()
        term = self._waiting[0]
        self._cost_of(term)
        self._waiting.pop(0)
        diagnostics: Diagnostics = {
            "timestamp": time.time(),
            "iteration": self._passes_suggested,
            "phase": "main",
            "generation": self._generation + 1,
        }
        return Suggestion(candidate=term, acquisition_value=None, diagnostics=diagnostics)

    def _record_pass(self, suggestion: Suggestion, observed: float) -> None:
        """Take a measured offspring into the run: its fitness pair and its dataset row."""
        term = suggestion.candidate
        self._row_of.setdefault(term, len(self._x_list) - 1)
        self._fitness[term] = (observed, self._cost_of(term))
