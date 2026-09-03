"""The BO loop against a real SolutionSpace and the real cosy EA.

This is the test that was missing, and its absence is why a chain of defects survived undetected.
Every other test in this repository avoids the real thing: ``conftest.py`` passes
``search_space=None``, ``DummyOptimizer`` pops from a list and ignores the objective,
``StubEvolutionary`` draws from a fixed pool.  **Nothing ever produced a crossover offspring**,
and crossover is where the corruption starts.

The search space here is a three-level expression grammar with 144 terms.  It is built through the
real ``Synthesizer`` and takes about 0.2 ms, five orders of magnitude below the CNN space that
makes ``test_cnn_damg_nas.py`` need ``max_seconds=240``.  Nothing here imports torch.

The three levels are not decoration.  A start symbol admitting a leaf-only tree used to make
cosy's crossover raise before it could do anything, and having both a unary and a binary
combinator at the middle level means swapped subtrees differ in *size*, which exposes a stale
``size`` on top of a stale ``_hash``.

**The evolutionary components all take part**: sampled initialization with hard failure clauses,
resolution mutation over the whole term including the root, the subtree swap with its batch
semantics, and the driver that assembles them into an evolutionary search over a synthesized
search space.  The mutation rate is positive here for the first time (see :func:`_make_ea`).
"""

from __future__ import annotations

import random
from typing import Literal

import numpy as np
import pytest
from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.core.tree import Tree
from cosy.evolutionary_algorithms import (
    EvolutionarySearch,
    FitnessBasedReplacement,
    Generations,
    RankBasedSelection,
    ResolutionMutation,
    SampledInitialization,
    ScalarFitnessComparator,
    SubtreeSwap,
)
from cosy.search import SizeUniformSampler, generator_query

_TARGET = Constructor("E2")
# A *size* bound, not a depth: the samplers used here bound the term they complete, and the
# size-uniform one counts function symbols.  Every term of this space has size 5 to 7, so 7
# admits all 144 of them.  A smaller bound would silently shrink the search space under test.
_MAX_SIZE = 7

# A weighted node count.  Deterministic, torch-free, and, unlike ``tree.size``, it does not read
# the very field the defect under test corrupts.  A missing symbol raises KeyError rather than
# returning a substitute value.
_WEIGHTS = {"top": 0.0, "add": 1.0, "neg": 2.0, "a": 0.5, "b": 1.5, "c": 2.5}


def _repository() -> dict:
    """Three leaves, a unary and a binary combinator, and a binary root.

    Returns:
        dict: Component specifications for ``Synthesizer``.  Every argument is a term argument,
            with no literal parameters, so the trees have no constant leaves.
    """
    return {
        "a": SpecificationBuilder().suffix(Constructor("E0")),
        "b": SpecificationBuilder().suffix(Constructor("E0")),
        "c": SpecificationBuilder().suffix(Constructor("E0")),
        "neg": SpecificationBuilder().argument("x", Constructor("E0")).suffix(Constructor("E1")),
        "add": SpecificationBuilder()
        .argument("l", Constructor("E0"))
        .argument("r", Constructor("E0"))
        .suffix(Constructor("E1")),
        "top": SpecificationBuilder()
        .argument("l", Constructor("E1"))
        .argument("r", Constructor("E1"))
        .suffix(Constructor("E2")),
    }


@pytest.fixture(scope="module")
def space():
    """Build the search space once, since it is only read and never mutated."""
    return Synthesizer(_repository(), {}).construct_solution_space(_TARGET).prune()


def _objective(tree: Tree[str]) -> float:
    """Weighted node count of a term."""
    return float(_WEIGHTS[tree.root] + sum(_objective(child) for child in tree.children))


def _rebuild(tree: Tree[str]) -> Tree[str]:
    """Rebuild the tree bottom-up through ``__init__``.

    Every ``size`` and ``_hash`` is then correct by construction.
    """
    return Tree(tree.root, tuple(_rebuild(child) for child in tree.children))


def _make_ea(
    space, seed: int, counting: Literal["tree", "table"] = "tree"
) -> EvolutionarySearch:
    """Build the real cosy EA, fully seeded.

    Every component carries its own generator, and nothing is distributed to them behind the
    caller's back.  The previous driver reassigned every ``rng`` attribute it could find unless
    ``distribute_rngs=False`` said otherwise.

    ``counting`` selects how the samplers count.  Both constructions produce the same stream from
    the same seed where the table applies, and this space has no predicate at all, so a run under
    ``"table"`` has to agree with a run under ``"tree"`` step for step, which is what
    :func:`test_the_two_counting_forms_run_the_same_evolution` asserts.  It is the one place the
    wiring of the lazy machinery is exercised end to end rather than in isolation.

    ``mutation_rate`` is **positive** here.  The rule that kept it at 0 rested on the mutation
    operator being defective: it excluded the root, sampled the residual by search order and
    retried other positions until one worked.  The operator now draws a non-leaf position of the
    individual uniformly at random, the root included, replaces the subterm there by a fresh
    variable, and takes the offspring from the sampler's stream on the resulting residual query.
    Convergence of a run to a fittest individual is guaranteed only for a positive mutation rate
    and a crossover rate below one.
    """
    return EvolutionarySearch(
        initializer=SampledInitialization(
            SizeUniformSampler(_MAX_SIZE, random.Random(seed), counting=counting)
        ),
        mutation=ResolutionMutation(
            SizeUniformSampler(_MAX_SIZE, random.Random(seed + 1), counting=counting),
            random.Random(seed + 2),
        ),
        recombination=SubtreeSwap(random.Random(seed + 3), max_size=_MAX_SIZE),
        parent_selection=RankBasedSelection(1.7, rng=random.Random(seed + 4)),
        survivor_selection=FitnessBasedReplacement(),
        termination=Generations(3),
        population_size=20,
        crossover_rate=0.9,
        mutation_rate=0.1,
        rng=random.Random(seed + 5),
        # The EA maximizes the acquisition value, which the optimizer always hands over as
        # "larger is better", independent of the objective's own direction.
        comparator=ScalarFitnessComparator(greater_is_better=True),
    )


def _make_bo(
    space,
    acquisition: Literal[
        "ExpectedImprovement", "ProbabilityOfImprovement", "UpperConfidenceBound"
    ],
    seed: int = 42,
    **kwargs,
):
    """Build a BayesianOptimization over the real space with the real EA."""
    from bayesian_optimization.bo import BayesianOptimization

    return BayesianOptimization(
        search_space=space,
        request=_TARGET,
        acquisition_function=acquisition,
        optimizer=_make_ea(space, seed),
        seed=seed,
        **kwargs,
        # One object for both roles: the initializer built in initialize() draws from it, and so
        # does the duplicate fallback.
        sampler=SizeUniformSampler(_MAX_SIZE, random.Random(seed)),
        n_restarts_kernel_optimizer=0,
    )


def _observed(bo) -> list[Tree[str]]:
    """The points observed so far, through the public snapshot rather than a private attribute."""
    return list(bo.get_state_snapshot()["x_list"])


@pytest.mark.integration
def test_the_two_counting_forms_run_the_same_evolution(space):
    """A whole EA run is identical under both counting forms, which pins the wiring end to end.

    cosy pins the two constructions against each other on a stream, and the validation suite
    measures each against the brute-force oracle.  Neither exercises what a caller actually
    assembles: a sampler inside an initialization inside a driver, with a mutation drawing
    residuals from the same construction generation after generation.  A table subtly misaligned
    with the tree would show here as a diverging population, at the first generation where a draw
    differs.

    The space has no predicate, so the table applies to it.  On a repository whose predicates read
    a hole this test could not be written, and ``weighted_table`` would refuse rather than count
    something else.
    """
    def run(counting: Literal["tree", "table"]) -> list[str]:
        ea = _make_ea(space, seed=3, counting=counting)
        return [
            f"{state.generation}: "
            + "|".join(sorted(str(tree) for tree in state.population))
            + f" -> {state.best}"
            for state in ea.evolutionary_stream(generator_query(space, _TARGET), _objective)
        ]

    by_tree, by_table = run("tree"), run("table")
    assert by_tree == by_table, (
        "the two counting forms produced different generations from the same seed"
    )
    assert len(by_tree) > 1, "a single generation would not exercise mutation at all"


@pytest.mark.integration
def test_crossover_offspring_satisfies_the_hash_contract(space):
    """An offspring must be indistinguishable from the same structure built directly.

    The parents have disjoint middle-level alphabets (``neg`` only against ``add`` only), so
    whichever crossover point pair is drawn, the swapped subtrees differ in symbol *and* size,
    and the failure does not depend on the seed.
    """
    leaf_a, leaf_b, leaf_c = Tree("a"), Tree("b"), Tree("c")
    primary = Tree("top", (Tree("neg", (leaf_a,)), Tree("neg", (leaf_a,))))
    secondary = Tree("top", (Tree("add", (leaf_b, leaf_c)), Tree("add", (leaf_c, leaf_b))))

    swap = SubtreeSwap(random.Random(0), max_size=_MAX_SIZE)
    offspring = swap.recombine(generator_query(space, _TARGET), primary, secondary)

    assert offspring, "the search space must admit crossover offspring at all"
    for child in offspring:
        rebuilt = _rebuild(child)
        assert child.size == rebuilt.size, (
            f"stale size: {child} reports {child.size}, actual {rebuilt.size}"
        )
        assert child == rebuilt
        assert hash(child) == hash(rebuilt)
        assert len({child, rebuilt}) == 1


@pytest.mark.integration
def test_ten_rounds_use_at_most_one_fallback(space):
    """Falling back should be an exception, not the mechanism.

    This is the end of the degradation chain, and it took two fixes to close: one so that an
    already-evaluated architecture is recognized at all, and one so that the acquisition function
    stops ranking known points above novel ones.  Before them the fallback fired from the second
    round onwards, and the run was random search wearing a BO label.

    The assertion sits *inside* the loop deliberately: if the fallback ever returns, the run also
    trips over the DPP defect below, and the loop would die with that ValueError before reaching an
    assertion placed after it.
    """
    bo = _make_bo(space, "UpperConfidenceBound")
    bo.initialize(objective=_objective, initial_size=5)

    flags: list[bool] = []
    for round_index in range(10):
        suggestion = bo.suggest()
        assert suggestion.diagnostics is not None
        flags.append(bool(suggestion.diagnostics["fallback_used"]))
        assert sum(flags) <= 1, (
            f"fallback used {sum(flags)} times in {round_index + 1} rounds: {flags}"
        )
        bo.observe(suggestion.candidate, _objective(suggestion.candidate))


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["batch", "single"])
@pytest.mark.parametrize("margin", [0.0, 0.25])
def test_probability_of_improvement_drives_the_loop(space, margin, mode):
    """Probability of improvement survives the whole path, not only a unit test.

    PI is the one score of the three that can saturate: on a run where the surrogate is confident
    everywhere, ``Phi`` underflows and a whole population scores exactly zero, which is a tie the
    known-point floor still has to sit below.  The margin makes it more likely, which is why both
    thresholds the score admits run here, at the incumbent and slightly above it.

    Both fitness modes run too.  The single-sample one had no coverage at all, and it is the one
    where the floor cannot be derived from a generation: PI is bounded below, so it is usable
    there, which is exactly what this pins.
    """
    bo = _make_bo(space, "ProbabilityOfImprovement", pi_margin=margin)
    bo.initialize(objective=_objective, initial_size=5)

    for _ in range(4):
        suggestion = bo.suggest(acquisition_fitness_mode=mode)
        assert suggestion.candidate not in _observed(bo)
        # The suggestion is novel, so its reported value is a genuine score rather than the
        # known-point floor, and a genuine PI score is a probability.
        assert 0.0 <= suggestion.acquisition_value <= 1.0
        bo.observe(suggestion.candidate, _objective(suggestion.candidate))

    assert len(_observed(bo)) == 9


@pytest.mark.integration
def test_ucb_never_prefers_a_known_point_over_a_novel_one(space):
    """No already-evaluated point may outscore a novel one.

    Nothing random takes part, with a fixed pool, a fixed kernel and ``optimizer=None``, so this
    is a deterministic statement about the acquisition function alone, with no EA involved.

    The direction used to be load-bearing here: under the minimizing branch UCB scored
    ``-mu + kappa*sigma``, which is negative in explored regions, and the fixed sentinel of 0.0
    then outranked every genuine candidate.  There is one direction now, and the floor is computed
    from the batch rather than fixed, so what this pins is the floor, not the branch.
    """
    from sklearn.gaussian_process import GaussianProcessRegressor

    from bayesian_optimization.acquisition_function import UpperConfidenceBound
    from bayesian_optimization.kernels import OrderedRootedSubtreeKernel

    pool = list(space.enumerate_trees(_TARGET, max_count=144))
    known = pool[:6]
    known_set = set(known)
    y = np.array([_objective(tree) for tree in known], dtype=float)

    gp = GaussianProcessRegressor(
        kernel=OrderedRootedSubtreeKernel(normalize=True),
        alpha=1e-6,
        normalize_y=True,
        optimizer=None,
        random_state=0,
    )
    gp.fit(np.asarray(known, dtype=object), y)

    ucb = UpperConfidenceBound(gp=gp, beta=2.0, known_points=known_set)

    novel = [tree for tree in pool if tree not in known_set]
    scores = ucb.evaluate_batch([*known, *novel])
    worst_novel = min(scores[tree] for tree in novel)
    best_known = max(scores[tree] for tree in known)

    assert worst_novel > best_known, (
        f"best known point scores {best_known}, worst novel point {worst_novel}, "
        f"the acquisition function prefers what has already been evaluated"
    )


@pytest.mark.integration
def test_no_suggestion_is_structurally_already_known(space):
    """A suggestion must never repeat a point that has already been evaluated.

    This assertion used to be carried by the ``canonical_tree`` workaround, which rebuilt every
    candidate before the duplicate check.  That workaround is gone, so what holds this up now is
    the tree itself: a crossover offspring hashes like the structurally identical tree, and the
    plain ``in`` test finds it.  No rebuilding anywhere, so if this ever fails again, the cached
    fields are stale again.
    """
    bo = _make_bo(space, "ExpectedImprovement")
    bo.initialize(objective=_objective, initial_size=5)

    seen = set(_observed(bo))
    for _ in range(10):
        suggestion = bo.suggest()
        candidate = suggestion.candidate
        assert candidate not in seen, f"re-suggested an already evaluated term: {candidate}"
        seen.add(candidate)
        bo.observe(candidate, _objective(candidate))


@pytest.mark.integration
def test_same_seed_yields_the_same_trajectory(space):
    """The same seed must produce the same run.

    This used to be impossible: the resolution engine collected its goals in a ``set[Goal]``
    whose element type defines neither ``__eq__`` nor ``__hash__``, so the enumeration order
    followed object addresses and the seeded shuffle in ``sample_tree`` operated on a differently
    ordered list every time.  The divergence showed up at the very first pre-sample, before a
    single EA generation, which is why the pre-samples stay part of the compared trajectory here.
    """

    def run() -> list[str]:
        bo = _make_bo(space, "ExpectedImprovement", seed=7)
        bo.initialize(objective=_objective, initial_size=5)
        trajectory = [str(tree) for tree in _observed(bo)]
        for _ in range(4):
            suggestion = bo.suggest()
            trajectory.append(str(suggestion.candidate))
            bo.observe(suggestion.candidate, _objective(suggestion.candidate))
        return trajectory

    assert run() == run()


@pytest.mark.integration
def test_the_fallback_does_not_depend_on_a_kernel_at_all(space):
    """Drawing a novel term is a question for the sampler, and for nothing else.

    The two terms below are distinct trees that the Weisfeiler-Lehman kernel scores at exactly
    1.0 against each other, because it builds an undirected, unordered graph, and that is the
    realistic case, since the caller hands the entire observed set to the fallback.  The greedy
    DPP that used to stand here read exactly that kernel: a rank-deficient seed set gave it a
    singular Gram matrix, and once the seed set spanned the kernel's rank (10 over all 108 terms
    of this space) *every* candidate had conditional variance 0 and it refused to draw at all,
    measured for 9 of 10 random 50-point seed sets.

    That device is gone.  A prefix of a size-uniform stream is a sample without replacement, so
    the first drawn term outside the observed set is novel by construction, whatever a kernel
    would have said about it.  The two terms stay in the test as the case that used to break.
    """
    from bayesian_optimization.initial_sampling import _sample_fallback_tree

    sampler = SizeUniformSampler(_MAX_SIZE, random.Random(0))
    leaf_a, leaf_b = Tree("a"), Tree("b")
    first = Tree("top", (Tree("add", (leaf_a, leaf_b)), Tree("neg", (leaf_a,))))
    second = Tree("top", (Tree("neg", (leaf_a,)), Tree("add", (leaf_b, leaf_a))))

    candidate = _sample_fallback_tree(
        sampler, generator_query(space, _TARGET), {first, second}
    )
    assert candidate not in {first, second}


@pytest.mark.integration
@pytest.mark.parametrize("seed", [5, 23, 101])
@pytest.mark.parametrize(
    "acquisition", ["ExpectedImprovement", "ProbabilityOfImprovement", "UpperConfidenceBound"]
)
@pytest.mark.parametrize("informed_initializer", [False, True])
def test_the_closed_loop_only_ever_evaluates_inhabitants(space, seed, acquisition, informed_initializer):
    """Every term the loop evaluates is an inhabitant, asserted over whole runs.

    The claim composes four closure claims, that the initializer streams inhabitants, that
    mutation and recombination return inhabitants, and that selection returns members of its
    input populations, and the composition is the part its proof marks as unverified.  Each claim
    has its own test in cosy, and what none of them covers is the loop that chains them, where a
    term travels from an initializer through several generations of an evolutionary run into the
    dataset.

    The parameters are the premises of the claim, not decoration.  Kernel-diverse initialization
    is one of the two initializers it admits, and that informed one reaches the dataset through a
    different construction entirely: a random search per member, each draw biased away from the
    members already drawn, rather than one stream.  The three acquisitions steer the evolutionary
    run differently, and three seeds make a rare escape less likely to pass unseen, since a
    composition defect need not fire on the first trajectory.

    The assertion runs over the whole dataset, so it covers terms of either origin, the
    evolutionary run's and, where the rejection path fired, the fallback sampler's.  It cannot
    *force* the fallback, which is what
    :func:`test_a_fallback_replacement_is_an_inhabitant_too` is for.

    Membership is decided by the checker of the resolution query, not by ``in`` on an enumeration:
    the point is derivability from the target, and an enumeration would additionally bound the
    size.
    """
    from cosy.search import checker, k_st

    from bayesian_optimization.initial_sampling import KernelDiverseInitializer

    initializer = (
        KernelDiverseInitializer(k_st, size_bound=_MAX_SIZE, rng=random.Random(seed))
        if informed_initializer
        else None
    )
    bo = _make_bo(space, acquisition, seed=seed, initializer=initializer)
    result = bo.optimize(_objective, budget=4, initial_size=5)

    evaluated = list(result["x"])
    assert len(evaluated) == 9, "five initial terms and one per pass"
    for tree in evaluated:
        assert checker(space, _TARGET, tree), (
            f"the loop evaluated {tree}, which is not an inhabitant of {_TARGET}"
        )


@pytest.mark.integration
def test_a_fallback_replacement_is_an_inhabitant_too(space):
    """A replacement drawn on the rejection path must be an inhabitant too.

    Everything the closure claim covers travels through the evolutionary run.  The replacement
    does not: it comes from the size-uniform sampler, which is a *different* closure claim, the
    one for collecting a population from a sampler's stream on the generator query, and it is
    evaluated by the expensive quality measure exactly like a proposal.  A stub optimizer that
    keeps returning an already evaluated term forces the path that a real run takes only
    occasionally.
    """
    from cosy.search import checker

    class RepeatsWhatIsKnown:
        """Returns a term from the observed set, so the rejection path has to fire."""

        def __init__(self, observed: list[Tree[str]]) -> None:
            """Remember what to repeat.

            Args:
                observed (list[Tree[str]]): The terms already evaluated.
            """
            self._observed = observed

        def evolutionary_best(
            self, query, fitness_function, fitness_function_mode: str = "batch"
        ) -> Tree[str]:
            """Return an already evaluated term.

            Args:
                query: Ignored.
                fitness_function: Ignored.
                fitness_function_mode (str): Ignored. (Default value = 'batch')

            Returns:
                Tree[str]: The first observed term.
            """
            return self._observed[0]

    bo = _make_bo(space, "ExpectedImprovement", seed=17)
    bo.initialize(objective=_objective, initial_size=4)
    bo.optimizer = RepeatsWhatIsKnown(_observed(bo))

    suggestion = bo.suggest()

    assert suggestion.diagnostics is not None
    assert suggestion.diagnostics["fallback_used"] is True, "the rejection path did not fire"
    assert suggestion.candidate not in _observed(bo), "the replacement repeats an evaluation"
    assert checker(space, _TARGET, suggestion.candidate), (
        f"the fallback produced {suggestion.candidate}, which is not an inhabitant of {_TARGET}"
    )


@pytest.mark.integration
def test_the_closed_loop_and_the_ask_tell_layer_agree(space):
    """The two entry points are one algorithm, so from one seed they run one trajectory.

    They are separate code paths, the closed loop calling the objective itself and the ask/tell
    layer having the value handed back in, and nothing else would notice them drifting apart,
    since each has its own tests.
    """
    closed = _make_bo(space, "ExpectedImprovement", seed=11)
    closed_result = closed.optimize(_objective, budget=3, initial_size=4)

    stepwise = _make_bo(space, "ExpectedImprovement", seed=11)
    stepwise.initialize(objective=_objective, initial_size=4)
    for _ in range(3):
        suggestion = stepwise.suggest()
        stepwise.observe(suggestion.candidate, _objective(suggestion.candidate))
    stepwise_result = stepwise.finalize()

    assert [str(tree) for tree in closed_result["x"]] == [
        str(tree) for tree in stepwise_result["x"]
    ]
    assert closed_result["best_y"] == stepwise_result["best_y"]


@pytest.mark.integration
def test_the_kernel_diverse_initializer_seeds_a_real_run(space):
    """The kernel-diverse initializer seeds a run through the loop rather than in isolation.

    Its unit tests drive ``initialize`` directly and its distribution is checked in the validation
    suite.  What neither touches is the one seam this package adds: an initializer handed to the
    constructor, drawing the initial dataset the GP is then conditioned on.  Everything downstream
    is indifferent to where the terms came from, which is exactly why a break here would be quiet.
    """
    from cosy.search import k_st

    from bayesian_optimization.initial_sampling import KernelDiverseInitializer

    bo = _make_bo(
        space,
        "ExpectedImprovement",
        initializer=KernelDiverseInitializer(
            k_st, size_bound=_MAX_SIZE, rng=random.Random(99)
        ),
    )
    bo.initialize(objective=_objective, initial_size=4)

    seeded = _observed(bo)
    assert len(set(seeded)) == 4, "the initial dataset holds distinct inhabitants"

    for _ in range(3):
        suggestion = bo.suggest()
        assert suggestion.candidate not in _observed(bo)
        bo.observe(suggestion.candidate, _objective(suggestion.candidate))

    assert _observed(bo)[:4] == seeded, "the loop keeps the population it was seeded with"
    assert len(_observed(bo)) == 7


@pytest.mark.integration
def test_the_acceptance_checks_run_on_a_real_run(space):
    """The diagnostic reads of the rebuilt loop run on the real space with the real EA.

    The reads have their formulas pinned against reference figures in the validation suite and
    their contracts in the unit tests, and both of those settings hand them data assembled for
    them.  Here they read a run: cosy's search space, cosy's evolutionary algorithm, this
    package's kernel over terms rather than over numbers, a population that came out of
    recombination, a surrogate conditioned on what the loop chose to evaluate.

    The one check with a verdict is the fourth, because it is the only one whose claim does not
    depend on how much data a run collected: the term an evolutionary run returns lies on the
    upper right frontier of its final population in the mean-deviation plane, and that is a
    statement about one maximization.  The others are asked to produce readable numbers on real
    inputs, and eight evaluations of a weighted node count over 144 terms is too little to demand
    a calibrated surrogate from, while a threshold invented here would be a threshold about this
    fixture.
    """
    from bayesian_optimization import (
        kernel_objective_alignment,
        read_calibration,
        read_fit,
        read_frontier,
        read_gram,
        read_trace,
    )
    from bayesian_optimization.diagnostics import spread

    bo = _make_bo(space, "ExpectedImprovement")
    result = bo.optimize(_objective, budget=4, initial_size=4, record_population=True)

    terms = list(result["x"])
    values = [float(value) for value in result["y"]]
    assert len(terms) == len(set(terms)) == 8

    # --- 1. The kernel, before any conditioning -----------------------------
    matrix = bo.kernel(terms)
    gram = read_gram(matrix, objective=values)
    assert gram.size == 8
    assert gram.symmetry_error < 1e-12
    assert gram.minimum_eigenvalue > -1e-8
    # The subtree kernel normalizes by default, so the scale spread a counting kernel shows, its
    # self-similarity growing with the term size until large terms dominate, is gone by
    # construction.  What is left to read is the structure, and it is between the extremes: the
    # matrix neither says everything is alike nor that nothing is.
    assert gram.diagonal_spread == pytest.approx(1.0)
    assert 0.05 < gram.off_diagonal_mean < 0.95
    assert gram.off_diagonal_spread > 0.1
    assert kernel_objective_alignment(matrix, values) is not None

    # --- 2. The mean, on terms the surrogate has not seen -------------------
    # Held out by conditioning on the run's first six evaluations and predicting its last two.
    fit = read_fit(bo._fit_surrogate(terms[:6], values[:6]), terms[6:], values[6:])
    assert fit.size == 2
    assert fit.prediction_spread > 0.0, "a surrogate that answered one value transfers nothing"
    assert np.isfinite(fit.residual_root_mean_square)

    # --- 3. The uncertainty, over the whole dataset -------------------------
    # The surrogate the run reports has not seen its final evaluation, but this one has.
    calibration = read_calibration(bo.surrogate_over_dataset())
    assert calibration.size == 8
    assert calibration.targets_normalized is True
    assert all(np.isfinite(value) for value in calibration.standardized_residuals)
    assert np.isfinite(calibration.log_marginal_likelihood)

    # --- 4. The inner evolutionary run --------------------------------------
    # Read as the run recorded it, with acquisition, returned term and population together.
    # Rebuilt from the surrogate over the whole dataset instead, the same pick comes out behind
    # seventeen of its twenty members, and the reading means nothing.
    run = bo.last_acquisition_run
    assert run is not None
    assert len(run.population) == _make_ea(space, 42).population_size
    assert not run.fallback_used

    frontier = read_frontier(run)
    assert frontier.on_frontier, "the returned term is dominated by its own final population"
    assert frontier.higher_scored_members == 0
    assert frontier.pick_in_population
    # A real final generation repeats itself, and the repetition is what the read reports.
    assert frontier.distinct_members < frontier.size
    assert frontier.mean_spread > 0.0
    # Nothing in this population had been evaluated when it was scored.  It is worth asserting
    # rather than assuming: the loop rejects duplicates, so a member that *is* a known point is a
    # member the maximization scored at the floor.  The count read three here for as long as the
    # recorded acquisition shared the loop's live set and kept growing after its own pass.
    assert frontier.known_members == 0

    # --- 5. The run itself ---------------------------------------------------
    trace = read_trace(result["trace"])
    assert trace.passes == 4
    assert trace.fallbacks == 0
    assert trace.best_trace[-1] == pytest.approx(max(values))
    # Counted off the dataset, so this is a statement and not an identity: four passes on a
    # four-term design, none of them repeating.
    assert trace.distinct_fraction == pytest.approx(1.0)
    assert result["trace"][-1].evaluations == 8
    # Neither pin of the deviation reading: the loop is not stuck at the data and not stuck at the
    # prior.
    assert trace.deviation_minimum > 0.0
    assert trace.deviation_spread == pytest.approx(
        spread([record.deviation for record in result["trace"]])
    )
