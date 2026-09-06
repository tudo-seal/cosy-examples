"""The small NAS example loads its data, builds its space and closes its loop.

The example had no test at all, and it could not have had one: it read its table, built its search
space and configured its optimizer while it was being imported, with the epoch count and the batch
size written into the module. A test could neither ask for a cheaper configuration nor reach any
part on its own.

Each part is now a function, and each test below holds one of them against what it claims. Two of
them are about cost rather than about a result. The closed run trains for a single epoch, which is
what makes it a test instead of an experiment, and the import test is what keeps the file from
growing module-level work back.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
import torch
from cosy.core.solution_space import SolutionSpace
from cosy.evolutionary_algorithms import EvolutionarySearch

from bayesian_optimization.bo import BayesianOptimization
from bayesian_optimization.examples.simple_nas import simple_nas
from bayesian_optimization.examples.simple_nas.simple_nas import (
    Simple_DNN_Repository,
    build_acquisition_optimizer,
    build_repository,
    build_search_space,
    load_iris,
    main,
    make_objective,
    network_target,
    run_search,
)

# The two numbers every fixture below trains with. At batch size 16 one epoch costs a candidate
# about four milliseconds on one laptop, against about four hundred and sixty at the two hundred
# epochs the run this example ships with uses. The batch size is deliberately neither the 8 of that
# run nor a default of any function here, so that a batch size written into the repository instead
# of taken from the call shows up as a failure.
TEST_EPOCHS = 1
TEST_BATCH_SIZE = 16

# The combinators of the repository. Fourteen names, and the algebra interprets each of them.
EXPECTED_COMBINATORS = frozenset({
    "Adam", "CrossEntropy", "DataLoader", "ELU", "L1", "Layer", "MSE", "Model", "Model_cons",
    "ReLu", "SGD", "Sigmoid", "Softmax", "System",
})


@pytest.fixture(scope="module")
def iris() -> tuple[np.ndarray, np.ndarray]:
    """The shipped table, read once for the whole module.

    Returns:
        tuple[np.ndarray, np.ndarray]: Its features and its one-hot labels.
    """
    return load_iris()


@pytest.fixture(scope="module")
def repository(iris: tuple[np.ndarray, np.ndarray]) -> Simple_DNN_Repository:
    """The repository over the shipped table, at one epoch.

    Args:
        iris (tuple[np.ndarray, np.ndarray]): The features and the labels.

    Returns:
        Simple_DNN_Repository: The repository.
    """
    features, labels = iris
    return build_repository(features, labels, epochs=TEST_EPOCHS, batch_size=TEST_BATCH_SIZE)


@pytest.fixture(scope="module")
def target():
    """The target of the shipped table: four features in, three classes out.

    Returns:
        Type: The synthesis target.
    """
    return network_target(4, 3)


@pytest.fixture(scope="module")
def search_space(repository: Simple_DNN_Repository, target) -> SolutionSpace:
    """The space of that target over that repository, synthesized once for the whole module.

    Args:
        repository (Simple_DNN_Repository): The combinators.
        target (Type): The target.

    Returns:
        SolutionSpace: The space.
    """
    return build_search_space(repository, target)


def _rules_of(space: SolutionSpace, combinator: str) -> list:
    """Collect every rule of the space that applies the given combinator.

    Args:
        space (SolutionSpace): The space to read.
        combinator (str): The combinator's name.

    Returns:
        list: Its rules, over every nonterminal of the space.
    """
    return [rule
            for nonterminal in space.nonterminals()
            for rule in space[nonterminal]
            if rule.terminal == combinator]


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------

def test_the_iris_table_loads_as_150_rows_of_4_features_and_3_one_hot_classes(iris):
    """The loader returns the shape the target is written for.

    The target names four inputs and three outputs, and the space is built from the target and
    the repository alone. A loader that returned a different width would leave the space untouched
    and fail inside the first training call, with a shape error that names no column of the
    table.

    Args:
        iris (tuple[np.ndarray, np.ndarray]): The features and the labels.
    """
    features, labels = iris
    assert features.shape == (150, 4)
    assert labels.shape == (150, 3)
    # One-hot means exactly one entry per row, and the shipped table holds fifty of each class.
    assert np.array_equal(labels.sum(axis=1), np.ones(150))
    assert np.array_equal(labels.sum(axis=0), np.array([50.0, 50.0, 50.0]))


def test_the_class_of_a_row_is_read_from_its_class_column_and_not_from_its_position(tmp_path):
    """A shuffled table is labeled correctly, which a positional construction cannot manage.

    The loader used to build the labels as fifty zeros, fifty ones and fifty twos, which is right
    for the shipped file and wrong for every other ordering of it, without failing. The check below
    shuffles the rows and holds each label against the class name standing beside it.

    Args:
        tmp_path (Path): The directory the shuffled copy is written to.
    """
    shipped = pd.read_csv(str(simple_nas.DATA_PATH))
    shuffled = shipped.sample(frac=1.0, random_state=0).reset_index(drop=True)
    # A shuffle that left the blocks in place would prove nothing.
    assert list(shuffled[simple_nas.CLASS_COLUMN]) != list(shipped[simple_nas.CLASS_COLUMN])
    path = tmp_path / "shuffled_iris.csv"
    shuffled.to_csv(str(path), index=False)

    features, labels = load_iris(path)

    names = sorted(shipped[simple_nas.CLASS_COLUMN].unique())
    expected = np.array([names.index(name) for name in shuffled[simple_nas.CLASS_COLUMN]])
    assert np.array_equal(labels.argmax(axis=1), expected)
    assert np.array_equal(features, shuffled[list(simple_nas.FEATURE_COLUMNS)].to_numpy())


def test_the_table_is_found_from_a_working_directory_that_is_not_the_repository(tmp_path,
                                                                                monkeypatch):
    """The default path is resolved against the module, so any caller reads the same file.

    The table ships as package data. A path relative to the working directory would work for a
    caller standing in the repository root and fail for every other one, tests included.

    Args:
        tmp_path (Path): A directory outside the repository.
        monkeypatch (pytest.MonkeyPatch): The fixture that changes the working directory.
    """
    monkeypatch.chdir(tmp_path)
    features, labels = load_iris()
    assert features.shape == (150, 4)
    assert labels.shape == (150, 3)


# ---------------------------------------------------------------------------
# The repository
# ---------------------------------------------------------------------------

def test_the_repository_offers_the_fourteen_combinators_the_algebra_interprets(repository):
    """Every combinator has a specification and an interpretation, and no third one has either.

    A combinator that the algebra does not interpret raises only once the term that uses it is
    evaluated, which in a search is somewhere in the middle of a run.

    Args:
        repository (Simple_DNN_Repository): The repository.
    """
    assert set(repository.gamma()) == EXPECTED_COMBINATORS
    assert set(repository.torch_algebra()) == EXPECTED_COMBINATORS


def test_the_epoch_count_and_the_batch_size_of_the_call_reach_every_training_system(search_space):
    """Both numbers travel from the call into the literals of every ``System`` rule.

    They used to stand on module level, where the only way to change them was to edit the file.
    The space is the place to read them back, because that is where a run picks them up: the
    repository holds each as a one-element group, so a target that names neither still reaches
    exactly one epoch count and one batch size.

    Args:
        search_space (SolutionSpace): The space of the fixtures.
    """
    systems = _rules_of(search_space, "System")
    assert systems, "the space has no training system at all"
    assert {rule.literal_substitution["ep"] for rule in systems} == {TEST_EPOCHS}
    assert {rule.literal_substitution["bs"] for rule in systems} == {TEST_BATCH_SIZE}
    # The two dimensions come from the target and are pinned with them.
    assert {rule.literal_substitution["in"] for rule in systems} == {4}
    assert {rule.literal_substitution["out"] for rule in systems} == {3}


def test_a_stacked_model_has_exactly_one_more_hidden_layer_than_the_model_it_stacks(search_space):
    """``Model_cons`` couples its two hidden-layer parameters at ``m = n - 1``.

    This is the one coupling in the repository, and it is a dependent parameter choice rather than
    a predicate. It also settles the base case: ``n = 0`` would ask for ``m = -1``, which the group
    does not hold, so ``Model_cons`` cannot build a model without a hidden layer and ``Model`` is
    the only rule that does.

    Args:
        search_space (SolutionSpace): The space of the fixtures.
    """
    pairs = {(rule.literal_substitution["n"], rule.literal_substitution["m"])
             for rule in _rules_of(search_space, "Model_cons")}
    assert pairs == {(1, 0), (2, 1), (3, 2), (4, 3), (5, 4)}

    # And ``Model`` carries no hidden-layer parameter at all. Its suffix fixes the count instead.
    models = _rules_of(search_space, "Model")
    assert models
    assert all("n" not in rule.literal_substitution for rule in models)


def test_the_acquisition_optimizer_carries_the_configuration_of_the_published_run():
    """The rates and the population size of the example are what the defaults hand back.

    The two rates are the numbers the convergence conditions constrain, and the mutation rate of
    this example is quoted in the CNN example beside its own. A default that drifted would leave
    that comparison describing a run nobody makes.
    """
    search = build_acquisition_optimizer()
    assert search.population_size == 250
    assert search.crossover_rate == 0.9
    assert search.mutation_rate == 0.02
    # Below 1 and above 0 respectively, which is what almost sure convergence asks of them.
    assert 0.0 < search.mutation_rate
    assert search.crossover_rate < 1.0
    # The search's own generator is derived from the seed. Two searches built from one seed draw
    # the same number from it, and two from different seeds do not.
    assert build_acquisition_optimizer(seed=3).rng.random() == \
        build_acquisition_optimizer(seed=3).rng.random()
    assert build_acquisition_optimizer(seed=3).rng.random() != \
        build_acquisition_optimizer(seed=4).rng.random()


# ---------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------

def test_the_objective_reports_the_negated_test_loss(repository, search_space, target):
    """The objective is the negated loss, because the loop maximizes and the example minimizes.

    The old driver passed ``greater_is_better=False`` to the loop, which no longer takes it. The
    direction now lives in the objective alone, so this is the only place the sign is fixed, and a
    run that reported the accuracy or the loss unnegated would be maximizing the wrong quantity
    while looking exactly as healthy.

    Args:
        repository (Simple_DNN_Repository): The repository whose algebra scores a candidate.
        search_space (SolutionSpace): The space of the fixtures.
        target (Type): The start of the enumeration.
    """
    term = next(iter(search_space.enumerate_trees(target)))
    algebra = repository.torch_algebra()

    torch.manual_seed(0)
    accuracy, loss = term.interpret(algebra)
    torch.manual_seed(0)
    value = make_objective(repository)(term)

    assert value == pytest.approx(-loss)
    assert value < 0.0
    # The accuracy is the other half of the pair and is not what the loop reads.
    assert value != pytest.approx(-accuracy)


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def test_a_closed_run_of_one_epoch_observes_its_whole_budget_and_returns_a_trainable_candidate(
        repository, search_space, target):
    """The loop runs end to end at a cost a test can pay, and answers with a term of the space.

    The budget and the initial size are what a caller gets to choose, so both are read back off
    the result: the dataset holds one entry per initial term plus one per pass, and the best value
    is the largest of them.

    Args:
        repository (Simple_DNN_Repository): The repository whose algebra scores a candidate.
        search_space (SolutionSpace): The space of the fixtures.
        target (Type): The request the loop draws its own terms from.
    """
    budget, initial_size = 2, 4
    result = run_search(repository, search_space, target,
                        budget=budget, initial_size=initial_size,
                        optimizer=build_acquisition_optimizer(population_size=6, generations=2,
                                                              depth_bound=12),
                        depth_bound=12, seed=0)

    assert result["iterations"] == budget
    assert len(result["x"]) == initial_size + budget
    assert len(result["y"]) == initial_size + budget
    assert result["best_y"] == pytest.approx(max(result["y"]))
    # The best value is a negated loss, so it is negative, and the term it belongs to trains.
    assert result["best_y"] < 0.0
    accuracy, loss = result["best_tree"].interpret(repository.torch_algebra())
    assert 0.0 < accuracy <= 100.001
    assert np.isfinite(loss)


def test_main_runs_the_whole_example_and_reports_the_loss_of_its_best_candidate(capsys):
    """``main`` is the closed example, and every number it costs is one of its arguments.

    It used to be a module body with the epoch count written into it, so the only way to run it was
    to pay for two hundred epochs per candidate. What it prints is a loss rather than the negated
    loss the loop maximizes.

    Args:
        capsys (pytest.CaptureFixture): The fixture that reads the printed report.
    """
    result = main(epochs=1, batch_size=150, budget=1, initial_size=3,
                  population_size=6, generations=2, depth_bound=12, seed=1, verbose=False)

    assert len(result["x"]) == 4
    printed = capsys.readouterr().out
    assert f"with loss: {-result['best_y']}" in printed
    assert "accuracy:" in printed


# ---------------------------------------------------------------------------
# What the module does at import time
# ---------------------------------------------------------------------------

def test_importing_the_example_builds_no_repository_no_space_and_no_optimizer():
    """Importing the module costs nothing but the import of its dependencies.

    The module used to read the table, synthesize the space and configure the loop while it was
    being imported. Every importer paid for that, and no importer could change any of it. The
    check is on the objects rather than on a duration, so it names what came back instead of
    reporting that something got slower.
    """
    heavy = (Simple_DNN_Repository, SolutionSpace, BayesianOptimization, EvolutionarySearch,
             torch.utils.data.TensorDataset, np.ndarray)
    built = sorted(name for name, value in vars(simple_nas).items()
                   if isinstance(value, heavy))
    assert built == []


def test_the_example_reaches_the_package_without_writing_to_sys_path():
    """The module imports the package by name, with no path manipulation of its own.

    The file used to insert the repository root into ``sys.path`` before its own imports, which is
    why it carried a per-file exemption from the import-placement rule. The exemption is gone with
    the insertion, so this reads as a source check here and as a lint failure there.
    """
    source = simple_nas.__file__
    assert source is not None
    text = open(source, encoding="utf-8").read()
    assert "sys.path" not in text
    # And ``sys``, which the file imported for that insertion alone, is gone with it.
    assert "\nimport sys" not in text


# ---------------------------------------------------------------------------
# What the space hands out
# ---------------------------------------------------------------------------

def test_the_first_five_enumerated_terms_of_the_space_all_train(repository, search_space, target):
    """The first five terms the space hands out are training systems the algebra can run.

    A repository whose specifications and whose algebra disagree produces terms that synthesize and
    then fail to interpret, which a search discovers one evaluation at a time.

    Args:
        repository (Simple_DNN_Repository): The repository whose algebra runs a term.
        search_space (SolutionSpace): The space of the fixtures.
        target (Type): The start of the enumeration.
    """
    algebra = repository.torch_algebra()
    terms = list(itertools.islice(search_space.enumerate_trees(target), 5))
    assert len(terms) == 5
    for term in terms:
        assert term.root == "System"
        accuracy, loss = term.interpret(algebra)
        assert 0.0 < accuracy <= 100.001
        assert np.isfinite(loss)
