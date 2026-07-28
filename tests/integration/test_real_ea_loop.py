"""The BO loop against a real SolutionSpace and the real cosy EA (plan step S0b.4).

This is the test that was missing, and its absence is why a chain of defects survived two A30 runs.
Every other test in this repository avoids the real thing: ``conftest.py`` passes
``search_space=None``, ``DummyOptimizer`` pops from a list and ignores the objective,
``StubEvolutionary`` draws from a fixed pool.  **Nothing ever produced a crossover offspring** --
and crossover is where the corruption starts.

The search space here is a three-level expression grammar with 144 terms.  It is built through the
real ``Synthesizer`` and takes about 0.2 ms, five orders of magnitude below the CNN space that
makes ``test_cnn_damg_nas.py`` need ``max_seconds=240``.  Nothing here imports torch.

The three levels are not decoration.  A start symbol admitting a leaf-only tree makes cosy's
``Crossover`` raise ``ValueError`` in ``recombination.py:110-112`` before it can do anything, and
having both a unary and a binary combinator at the middle level means swapped subtrees differ in
*size*, which exposes a stale ``size`` on top of a stale ``_hash``.

Five of the six assertions fail against the unfixed code and are marked ``xfail(strict=True)``: the
suite stays green, the defect stays documented, and the moment a fix lands the resulting XPASS
forces the marker's removal.  The sixth passes today -- see its docstring, it is a regression
guard, not a proof.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.core.tree import Tree
from cosy.evolutionary_algorithms import (
    AgeBasedReplacement,
    Crossover,
    RandomLimitedDepthFirstInitialization,
    RankBasedSelection,
    ResolutionMutation,
    ScalarFitnessComparator,
    SimpleGeneticProgramming,
)

_TARGET = Constructor("E2")
_MAX_DEPTH = 3

# A weighted node count.  Deterministic, torch-free, and -- unlike ``tree.size`` -- it does not read
# the very field the defect under test corrupts.  A missing symbol raises KeyError rather than
# returning a substitute value.
_WEIGHTS = {"top": 0.0, "add": 1.0, "neg": 2.0, "a": 0.5, "b": 1.5, "c": 2.5}


def _repository() -> dict:
    """Three leaves, a unary and a binary combinator, and a binary root.

    Returns:
        dict: Component specifications for ``Synthesizer``.  Every argument is a term argument --
            no literal parameters, so the trees have no constant leaves.
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
    """Build the search space once; it is only read, never mutated."""
    return Synthesizer(_repository(), {}).construct_solution_space(_TARGET).prune()


def _objective(tree: Tree[str]) -> float:
    """Weighted node count of a term."""
    return float(_WEIGHTS[tree.root] + sum(_objective(child) for child in tree.children))


def _rebuild(tree: Tree[str]) -> Tree[str]:
    """Rebuild bottom-up through ``__init__``, so ``size`` and ``_hash`` are correct by construction."""
    return Tree(tree.root, tuple(_rebuild(child) for child in tree.children))


def _make_ea(space, seed: int) -> SimpleGeneticProgramming:
    """Build the real cosy EA, fully seeded.

    ``distribute_rngs=False`` is required: the default replaces every component's RNG with one
    drawn from an internal factory, discarding the seeds passed here.  ``mutation_rate`` is set to
    0 at the BO level per the project rule, so crossover is the only source of variation -- which
    is exactly the configuration the experiments run in.
    """
    return SimpleGeneticProgramming(
        space,
        _TARGET,
        lambda state: state.generation >= 3,
        RandomLimitedDepthFirstInitialization(
            space, _TARGET, max_depth=_MAX_DEPTH, rng=random.Random(seed)
        ),
        ResolutionMutation(space, _TARGET, max_depth=_MAX_DEPTH, rng=random.Random(seed + 1)),
        Crossover(space, _TARGET, max_depth=_MAX_DEPTH, rng=random.Random(seed + 2)),
        RankBasedSelection(1.7, rng=random.Random(seed + 3)),
        AgeBasedReplacement(),
        # The EA maximises the acquisition value, which the optimizer always hands over as
        # "larger is better" -- independent of the objective's own direction.
        ScalarFitnessComparator(greater_is_better=True),
        rng=random.Random(seed + 4),
        distribute_rngs=False,
    )


def _make_bo(space, acquisition: str, seed: int = 42):
    """Build a BayesianOptimization over the real space with the real EA."""
    from bayesian_optimization.bo import BayesianOptimization
    from bayesian_optimization.transforms import IdentityTransform

    return BayesianOptimization(
        search_space=space,
        request=_TARGET,
        acquisition_function=acquisition,
        optimizer=_make_ea(space, seed),
        optimizer_population_size=20,
        optimizer_mutation_rate=0.0,
        optimizer_recombination_rate=0.9,
        seed=seed,
        # Reaches the fallback initializer built in initialize(); the default of 100 would let it
        # sample far outside the intended depth.
        max_depth=_MAX_DEPTH,
        y_transform=IdentityTransform(),
        n_restarts_kernel_optimizer=0,
    )


def _observed(bo) -> list[Tree[str]]:
    """The points observed so far, through the public snapshot rather than a private attribute."""
    return list(bo.get_state_snapshot()["x_list"])


@pytest.mark.integration
def test_crossover_offspring_satisfies_the_hash_contract(space):
    """An offspring must be indistinguishable from the same structure built directly.

    The parents have disjoint middle-level alphabets (``neg`` only against ``add`` only), so
    whichever crossover point pair is drawn, the swapped subtrees differ in symbol *and* size --
    the failure does not depend on the seed.
    """
    leaf_a, leaf_b, leaf_c = Tree("a"), Tree("b"), Tree("c")
    primary = Tree("top", (Tree("neg", (leaf_a,)), Tree("neg", (leaf_a,))))
    secondary = Tree("top", (Tree("add", (leaf_b, leaf_c)), Tree("add", (leaf_c, leaf_b))))

    crossover = Crossover(space, _TARGET, max_depth=_MAX_DEPTH, rng=random.Random(0))
    offspring = crossover.recombine(primary, secondary)

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
@pytest.mark.xfail(
    strict=True,
    reason="BO-1/COSY-1: known points get the sentinel 0.0, which under minimisation beats every "
    "novel candidate, so the EA steers at them and suggest() falls back to a random tree",
)
def test_ten_rounds_use_at_most_one_fallback(space):
    """Falling back should be an exception, not the mechanism.

    The assertion sits *inside* the loop on purpose: once the fallback fires repeatedly the run
    also trips over the DPP defect below, and the loop would die with that ValueError before ever
    reaching an assertion placed after it.  Either failure is a real finding, so the ``xfail``
    covers both.
    """
    bo = _make_bo(space, "UpperConfidenceBound")
    bo.initialize(obj_fun=_objective, n_pre_samples=5, greater_is_better=False)

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
@pytest.mark.xfail(
    strict=True,
    reason="BO-1: the sentinel 0.0 for known points is only a floor for EI; for UCB under "
    "minimisation it is a ceiling over the whole novel population",
)
def test_ucb_never_prefers_a_known_point_over_a_novel_one(space):
    """No already-evaluated point may outscore a novel one.

    Nothing random takes part -- fixed pool, fixed kernel, ``optimizer=None`` -- so this is a
    deterministic statement about the acquisition function alone, with no EA involved.

    ``greater_is_better=False`` is load-bearing and must not be "simplified" away: it is the
    setting the CNN experiments run under and the default of ``initialize()``.  Under maximisation
    the same code passes, because the sentinel is then below the novel scores rather than above.
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

    ucb = UpperConfidenceBound(
        gp=gp,
        greater_is_better=False,
        known_points=known_set,
        incumbent=float(y.min()),
        kappa=2.0,
    )

    novel = [tree for tree in pool if tree not in known_set]
    scores = ucb.evaluate_batch([*known, *novel])
    worst_novel = min(scores[tree] for tree in novel)
    best_known = max(scores[tree] for tree in known)

    assert worst_novel > best_known, (
        f"best known point scores {best_known}, worst novel point {worst_novel} -- "
        f"the acquisition function prefers what has already been evaluated"
    )


@pytest.mark.integration
def test_no_suggestion_is_structurally_already_known(space):
    """A suggestion must never repeat a point that has already been evaluated.

    This assertion used to be carried by the ``canonical_tree`` workaround, which rebuilt every
    candidate before the duplicate check.  BO-2 removed it, so what holds this up now is COSY-1:
    a crossover offspring hashes like the structurally identical tree, and the plain ``in`` test
    finds it.  No rebuilding anywhere -- if this ever fails again, the cached fields are stale
    again.
    """
    bo = _make_bo(space, "ExpectedImprovement")
    bo.initialize(obj_fun=_objective, n_pre_samples=5, greater_is_better=False)

    seen = set(_observed(bo))
    for _ in range(10):
        suggestion = bo.suggest()
        candidate = suggestion.candidate
        assert candidate not in seen, f"re-suggested an already evaluated term: {candidate}"
        seen.add(candidate)
        bo.observe(candidate, _objective(candidate))


@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="COSY-3: new_goals is a set[Goal] and Goal defines neither __eq__ nor __hash__, so its "
    "iteration order follows object identity -- sample_tree shuffles a differently ordered list "
    "each run and no seed can make a run reproducible",
)
def test_same_seed_yields_the_same_trajectory(space):
    """The same seed must produce the same run.

    The pre-samples are part of the compared trajectory deliberately: the divergence appears at the
    very first one, before a single EA generation, which points straight at the enumeration order
    rather than at the EA.
    """

    def run() -> list[str]:
        bo = _make_bo(space, "ExpectedImprovement", seed=7)
        bo.initialize(obj_fun=_objective, n_pre_samples=5, greater_is_better=False)
        trajectory = [str(tree) for tree in _observed(bo)]
        for _ in range(4):
            suggestion = bo.suggest()
            trajectory.append(str(suggestion.candidate))
            bo.observe(suggestion.candidate, _objective(suggestion.candidate))
        return trajectory

    assert run() == run()


@pytest.mark.integration
@pytest.mark.xfail(
    strict=True,
    reason="BO-4: the DPP seed loop grows K_inv only when the score clears the threshold, so a "
    "WL-equivalent seed pair leaves K_inv one dimension short and the next matmul raises",
)
def test_sample_fallback_tree_handles_wl_equivalent_seen_points(space):
    """The fallback sampler must survive seed points the WL kernel cannot tell apart.

    Today it is only ever reached through a monkeypatch, so this defect never fired in a test.  The
    two terms below are distinct trees that the WL kernel scores at exactly 1.0 against each other,
    because it builds an undirected, unordered graph -- and that is the realistic case, since the
    real caller passes the entire set of observed points as seeds.
    """
    from bayesian_optimization.initial_sampling import _sample_fallback_tree

    initializer = RandomLimitedDepthFirstInitialization(
        space, _TARGET, max_depth=_MAX_DEPTH, rng=random.Random(0)
    )
    leaf_a, leaf_b = Tree("a"), Tree("b")
    first = Tree("top", (Tree("add", (leaf_a, leaf_b)), Tree("neg", (leaf_a,))))
    second = Tree("top", (Tree("neg", (leaf_a,)), Tree("add", (leaf_b, leaf_a))))

    candidate = _sample_fallback_tree(initializer, {first, second})
    assert candidate not in {first, second}
