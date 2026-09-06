"""Every algebra of the DAMG repository gives every combinator a meaning, on every shipped target.

An algebra is a dict from the name of a combinator to what that combinator stands for, and
``Tree.interpret`` looks a symbol up in it and calls what it finds with the node's children. Two
things can go wrong there and neither is caught where it is written. A combinator the algebra does
not name raises only when a term that uses it is interpreted, which is a term some targets produce
and others do not. And a clause whose parameter list is the wrong length raises only then as well,
because a dict of lambdas has no signature to check against.

So the file checks both, and it checks the second the only way there is: by interpreting real terms.
The key sets are held against the specification, and one term of every target in ``damg_targets`` is
interpreted under every algebra. That is twelve algebras against seventeen combinators, over
thirteen registered forms, and twelve targets to carry the terms.

``pretty_term_algebra`` and ``edgelist_algebra`` name twenty symbols beyond the seventeen, the ones
``hierarchy_algebra`` produces, so a folded term can be printed and drawn as well as an unfolded
one. That is a property of those two and of no other, and it is held here rather than assumed.
"""

from __future__ import annotations

import itertools
import time

import pytest
from cosy.core import Synthesizer

from bayesian_optimization.examples.damg_nas import damg_repo_algebras as algebras
from bayesian_optimization.examples.damg_nas import damg_targets
from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository

#: The seventeen combinators ``DAMGrepository.specification`` declares, written out so that a
#: combinator that disappears from the specification is a failure here and not a smaller number
#: compared against itself.
COMBINATORS = frozenset({
    "edges",
    "swap",
    "linear_layer",
    "sigmoid",
    "relu",
    "tanh",
    "sum",
    "product",
    "copy",
    "beside_singleton",
    "beside_cons",
    "before_singleton",
    "before_cons",
    "mse_loss",
    "l1loss",
    "adam_optimizer",
    "learner",
})

#: The twenty symbols ``hierarchy_algebra`` folds a term into. An algebra that reads folded terms
#: has to name these too, and one that reads unfolded terms alone must not be expected to.
FOLDED_SYMBOLS = frozenset({
    "edges_h1",
    "swap_h1",
    "linear_layer_h1",
    "sigmoid_h1",
    "relu_h1",
    "tanh_h1",
    "sum_h1",
    "product_h1",
    "copy_h1",
    "node",
    "beside_singleton_h1",
    "beside_singleton_h3",
    "beside_cons_h1",
    "beside_cons_h3",
    "before_singleton_h1",
    "before_cons_h1",
    "loss",
    "optimizer",
    "learner_h1",
    "learner_h2",
})

#: Every algebra of the module, by the name it is built under. ``hierarchy_algebra`` counts as four,
#: one per granularity, because the four share a function and nothing else: each names its own
#: symbols and folds its own way. ``edgelist_algebra`` is one algebra under two names, because its
#: argument decides what it returns and not what it names, which is why thirteen names stand for
#: twelve algebras.
ALGEBRA_FACTORIES = {
    "pretty_term": algebras.pretty_term_algebra,
    "edgelist": lambda: algebras.edgelist_algebra(False),
    "edgelist_verbose": lambda: algebras.edgelist_algebra(True),
    "pytorch_function": algebras.pytorch_function_algebra,
    "pytorch_model": algebras.pytorch_model_algebra,
    "hierarchy_0": lambda: algebras.hierarchy_algebra(0),
    "hierarchy_1": lambda: algebras.hierarchy_algebra(1),
    "hierarchy_2": lambda: algebras.hierarchy_algebra(2),
    "hierarchy_3": lambda: algebras.hierarchy_algebra(3),
    "request": algebras.request_algebra,
    "refinement_1": algebras.refinement_1_algebra,
    "refinement_2": algebras.refinement_2_algebra,
    "operator_histogram": algebras.operator_histogram_algebra,
}

#: The names under which the two fold-reading algebras are registered. ``edgelist_algebra``
#: appears under both of its arguments.
FOLD_READING_ALGEBRAS = frozenset({"pretty_term", "edgelist", "edgelist_verbose"})

#: The targets the module ships, by name. Read off the module rather than listed, so that a target
#: added there without a name in ``target_to_name`` fails the catalogue test below.
TARGET_NAMES = sorted(
    name
    for name in dir(damg_targets)
    if name.startswith("target_len_")
)


@pytest.fixture(scope="module")
def repository() -> DAMGrepository:
    """Build the repository in the configuration the example searches.

    Returns:
        DAMGrepository: The repository.
    """
    return DAMGrepository(
        linear_feature_dimensions=[1, 2, 3, 4, 5],
        constant_values=[0, 1, -1],
        learning_rate_values=[1e-2],
        n_epoch_values=[2000],
    )


@pytest.fixture(scope="module")
def specification(repository: DAMGrepository):
    """Build the specification once for the whole module.

    Args:
        repository (DAMGrepository): The repository.

    Returns:
        dict: The specification, a map from a combinator name to its clause.
    """
    return repository.specification()


def test_the_repository_declares_the_seventeen_combinators_and_no_others(specification):
    """The specification is the list every algebra is measured against, so it is pinned first."""
    assert set(specification) == set(COMBINATORS)
    assert len(COMBINATORS) == 17


def test_building_the_specification_costs_a_fraction_of_a_second(repository):
    """The specification is rebuilt per search space, so its cost has to stay negligible.

    It builds four nested groups and seventeen clauses and consults no space, and it takes about
    two milliseconds. The bound is far above that, because what a bound here can catch is one of
    the groups starting to enumerate itself, and that is orders of magnitude and not percent.
    """
    started = time.time()
    repository.specification()
    seconds = time.time() - started
    assert seconds < 0.5, f"building the specification took {seconds:.3f}s"


@pytest.mark.parametrize("name", sorted(ALGEBRA_FACTORIES))
def test_every_algebra_gives_every_combinator_a_meaning(name):
    """No algebra may leave a combinator out, whatever terms happen to be enumerated.

    A missing clause is invisible until a term that uses that combinator is interpreted, and which
    combinators a term uses depends on the target. This is what makes the gap a property of the
    algebra rather than of the run that happened to find it.
    """
    algebra = ALGEBRA_FACTORIES[name]()
    missing = COMBINATORS - set(algebra)
    assert missing == set(), f"{name} names no meaning for {sorted(missing)}"


@pytest.mark.parametrize("name", sorted(ALGEBRA_FACTORIES))
def test_only_the_printing_and_drawing_algebras_read_folded_terms(name):
    """``pretty_term`` and ``edgelist`` also name the twenty symbols a fold makes, the rest do not.

    That is what lets a folded term be printed and drawn beside the term it came from, which is how
    the granularity of a kernel is inspected. The other algebras interpret the unfolded term alone,
    and naming a folded symbol there would be a clause nothing can reach.
    """
    keys = set(ALGEBRA_FACTORIES[name]())
    if name in FOLD_READING_ALGEBRAS:
        assert FOLDED_SYMBOLS <= keys, (
            f"{name} reads folded terms and is missing {sorted(FOLDED_SYMBOLS - keys)}"
        )
        assert keys == COMBINATORS | FOLDED_SYMBOLS
    else:
        assert keys == COMBINATORS, f"{name} names {sorted(keys - COMBINATORS)} beyond the table"


@pytest.mark.parametrize("target_name", TARGET_NAMES)
def test_every_algebra_interprets_a_term_of_every_target(specification, target_name):
    """One term per target, interpreted under all thirteen algebra forms, has to come through.

    This is the arity half of the check and the only way to get at it: a clause is a lambda in a
    dict, so nothing compares its parameter list against the arity the combinator was declared
    with until ``interpret`` calls it. A term of a length target uses the compositions, the leaf
    combinators, the loss, the optimizer and the learner together, so one term exercises most of
    the table at once.
    """
    target = getattr(damg_targets, target_name)
    space = Synthesizer(specification, {}).construct_solution_space(target).prune()
    # islice, not max_count: an enumeration under max_count runs until the next new term for the
    # start symbol arrives, which on these spaces is far more work than the term that is wanted.
    terms = list(itertools.islice(space.enumerate_trees(target), 1))
    assert terms, f"{target_name} is uninhabited in the configuration the example searches"

    for name, factory in ALGEBRA_FACTORIES.items():
        try:
            terms[0].interpret(factory())
        except Exception as error:  # noqa: BLE001
            pytest.fail(
                f"{name} failed on a term of {target_name}: {type(error).__name__}: {error}")


def test_the_catalogue_holds_twelve_targets_and_names_every_one_of_them():
    """``target_to_name`` is the inverse of the module's own constants, and it has to stay total.

    A target added to the module without a line here answers "unknown", which is what a run record
    then says it searched. The count is asserted alongside, so that a target removed from the
    module rather than added shows up too.
    """
    assert len(TARGET_NAMES) == 12
    for name in TARGET_NAMES:
        target = getattr(damg_targets, name)
        assert damg_targets.target_to_name(target) == name

    assert damg_targets.target_to_name("not a target at all") == "unknown"


@pytest.mark.parametrize("target_name", TARGET_NAMES)
def test_building_a_search_space_costs_a_fraction_of_a_second_for_every_target(
    specification, target_name
):
    """Every DAMG space is built and pruned quickly, which is what makes these targets testable.

    On one laptop the slowest of the twelve is ``target_len_5_refined_2`` and the fastest
    ``target_len_2``, and both are fractions of a second. The bound sits well above the slowest,
    because what it catches is a configuration whose label product stops being enumerable, and that
    costs orders of magnitude rather than percent.
    """
    target = getattr(damg_targets, target_name)
    started = time.time()
    Synthesizer(specification, {}).construct_solution_space(target).prune()
    seconds = time.time() - started
    assert seconds < 3.0, f"building the {target_name} space took {seconds:.3f}s"
