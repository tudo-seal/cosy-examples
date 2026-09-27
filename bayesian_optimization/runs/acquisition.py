"""The evolutionary search that maximizes the acquisition function, and its default rates."""

import random
from typing import Any

from cosy.evolutionary_algorithms import (
    EvolutionarySearch,
    ExpScalarization,
    Generations,
    GenerousConservativeReplacement,
    RankBasedSelection,
    ResolutionMutation,
    SampledInitialization,
    ScalarFitnessComparator,
    SubtreeSwap,
)
from cosy.search import DepthBoundedRandomSampler

from bayesian_optimization.runs.search_program import DEFAULT_DEPTH_BOUND

# --- The acquisition optimizer --------------------------------------------------------------
# An evolutionary search converges to a best individual under five conditions, and four of them are
# decided by the components chosen below.  The fifth asks that the individuals a run can hold form a
# finite set, and that is a property of the space rather than of the components: the requested type
# carries structure literals over finite collections, so the set is finite.
#
# One of the four asks for an exhaustive sampler, which means that on every resolution query and for
# every completion within the sampler's bound, the first element of the stream is that completion
# with positive probability.  It is "every resolution query" that carries the weight here, because
# the mutation poses a residual query rather than a generator query, and a sampler that reaches only
# the generator queries would leave the mutation rate buying nothing.
#
# The position distribution is the other half, and it is not uniform over all positions: the
# operator draws among the non-leaves and the root.  A leaf holding a literal is a position where
# the residual query answers with the term already there, since the clause matcher pins a constant
# argument even at the opened position, so drawing one spends the mutation on the identity, and this
# repository is built from literal parameters.  Excluding the leaves keeps reachability, because the
# root is always a mutation point and the residual there is the generator query.
DEFAULT_SELECTION_PRESSURE = 1.7
# The convergence conditions ask for a crossover rate below 1 and a mutation rate above 0 and fix
# nothing else, so the two numbers are a choice, and this is the one place it is made.
#
# The mutation rate is per offspring and the operator is a macro-mutation.  ``ResolutionMutation``
# draws a position among the non-leaves and the root and replaces the whole subtree there with a
# fresh draw, so nothing here is analogous to flipping a bit.  How much of a term one mutation
# replaces therefore depends on how large the terms of the space are: on a large term the root is a
# rarer draw and an ordinary mutation replaces a smaller share, so the same rate shakes a large
# space less than a small one.  0.03 is chosen for the architectures this example searches, whose
# terms are large.  The two smaller example spaces of this repository sit lower still, at 0.02 in
# simple_nas and at 0.02 in the DAMG example.
#
# Convergence is untouched by the value.  The root is always a mutation point, and a mutation with
# an exhaustive sampler maps any individual to any other with positive probability through the root,
# so reachability holds for every positive rate.  The rate decides how hard the search is shaken,
# not what it can reach.  What the value is not is optimized: whether the inner search finds the
# argmax of the acquisition function is not measured here.
DEFAULT_CROSSOVER_RATE = 0.9
DEFAULT_MUTATION_RATE = 0.03
# Why the evolutionary operators stay depth-bounded while the loop's own sampler is size-uniform, on
# the very same program.  The two ask different questions.  The loop poses one generator query for
# the whole run, so a size-uniform sampler builds its counting construction once and every later
# draw reads the table it built.  The mutation poses a residual query at a fresh position of a fresh
# individual every time, and ``SizeUniformSampler`` keeps only the last construction it built, so a
# size-uniform mutation would pay that construction again per draw, and one pass of the loop makes
# as many draws as it makes mutations.  A depth-bounded draw never counts.
#
# Nothing in the convergence argument is given up by that.  The depth-bounded sampler is exhaustive
# on every resolution query, which is the condition the operators have to meet, and the finiteness
# condition comes from the space rather than from the bound.


def build_acquisition_optimizer(
    *,
    population_size: int,
    generations: int,
    depth_bound: int = DEFAULT_DEPTH_BOUND,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    selection_pressure: float = DEFAULT_SELECTION_PRESSURE,
    seed: int = 0,
) -> EvolutionarySearch[Any, Any, Any]:
    """Build the evolutionary search that maximizes the acquisition function.

    Each component gets its own generator, derived from ``seed``, so that changing one operator's
    consumption does not shift every later draw of the run.

    Rank-based rather than fitness-proportional parent selection.  Fitness-proportional weights
    would be the raw acquisition values, and expected improvement spans orders of magnitude on a
    confident surrogate, so a single individual takes nearly all of the selection probability.  Rank
    selection depends on the ordering alone.  It gives every member of the population a positive
    probability, which is one of the convergence conditions, as long as the pressure stays below 2:
    at 1.7 the worst individual keeps a share of 0.3 divided by the population size.

    Args:
        population_size (int): The population size.
        generations (int): The termination bound.
        depth_bound (int): The bound of the samplers, on the depth of a term.
        crossover_rate (float): The crossover rate, which the convergence conditions need below 1.
        mutation_rate (float): The mutation rate, which they need above 0.
        selection_pressure (float): The rank-selection pressure, below 2 so that every member keeps
            a positive probability.
        seed (int): The base seed.  Every component derives its own generator from it.

    Returns:
        EvolutionarySearch[Any, Any, Any]: The configured search.
    """
    def rng(offset: int) -> random.Random:
        """Derive one component's generator.

        Args:
            offset (int): The component's index.

        Returns:
            random.Random: Its generator.
        """
        return random.Random(seed * 100 + offset)

    return EvolutionarySearch(
        initializer=SampledInitialization(DepthBoundedRandomSampler(depth_bound, rng(0))),
        mutation=ResolutionMutation(DepthBoundedRandomSampler(depth_bound, rng(1)), rng(2)),
        # No max_size: the finiteness the convergence argument asks for comes from the space itself,
        # and a size cap on the acceptance test would be a second measure beside the sampler's depth
        # bound, which is the one confusion this module is written to avoid.
        recombination=SubtreeSwap(rng(3)),
        parent_selection=RankBasedSelection(selection_pressure, rng=rng(4)),
        # Generous and conservative, which is the condition on survivor selection and the component
        # that carries the convergence guarantee.  Truncation keeps a best member and is therefore
        # conservative, but it gives a worse member no chance of surviving, so it is not generous.
        survivor_selection=GenerousConservativeReplacement(ExpScalarization(), rng(5)),
        termination=Generations(generations),
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        rng=rng(6),
        comparator=ScalarFitnessComparator(True),  # the acquisition is always maximized
    )
