"""Search a fully connected classifier for the Iris table with Bayesian optimization.

The example is the small one of this repository. A repository of fourteen combinators describes
dense networks of up to five hidden layers, synthesis turns a target into a search space, and the
Bayesian optimization loop searches that space for the network with the lowest training loss.

Everything a run fixes is an argument of a function here, and nothing is built at import time.
The epoch count and the batch size decide what one candidate costs: training a candidate is one
pass over the table per epoch, and everything else it costs, the model construction and the one
scoring pass, is paid once whatever the epoch count is. The population size and the generation
count decide what one acquisition maximization costs, which is the larger of the two at the
settings below. The epoch count and the batch size used to stand on module level, where a caller
could not reach them, and so did the search space itself, which every import paid for.

The loop maximizes. The objective is therefore the negated loss of the scoring pass, and
``best_y`` comes back negated with it. :func:`main` prints the loss. The scoring pass reads the
data the candidate trained on, so this is a training loss and not a held out one.

Run it as a module from the repository root::

    python -m bayesian_optimization.examples.simple_nas.simple_nas
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from cosy.core import SpecificationBuilder, Synthesizer
from cosy.core.solution_space import SolutionSpace
from cosy.core.tree import Tree
from cosy.core.types import Constructor, DataGroup, Literal, Type, Var
from cosy.evolutionary_algorithms import (
    EvolutionarySearch,
    ExpScalarization,
    FitnessProportionalSelection,
    Generations,
    GenerousConservativeReplacement,
    ResolutionMutation,
    SampledInitialization,
    ScalarFitnessComparator,
    SubtreeSwap,
)
from cosy.search.samplers import DepthBoundedRandomSampler
from torch import nn

from bayesian_optimization import BayesianOptimization, WeisfeilerLehmanKernel

# The table ships with the package, and the path is resolved against this file so that a caller in
# any working directory reads the same file.
DATA_PATH = Path(__file__).resolve().parent / "data" / "iris.csv"
FEATURE_COLUMNS = ("sepal_length", "sepal_width", "petal_length", "petal_width")
CLASS_COLUMN = "class"

# The configuration this example has run with since it was written. The depth bound is the one
# the evolutionary operators were built with, the rates are the two numbers the convergence
# conditions constrain, and the population size and the generation count are what one acquisition
# maximization costs.
DEFAULT_DEPTH_BOUND = 100
DEFAULT_POPULATION_SIZE = 250
DEFAULT_GENERATIONS = 100
DEFAULT_CROSSOVER_RATE = 0.9
DEFAULT_MUTATION_RATE = 0.02

# The learning rates the two optimizers may carry, and the widths a layer may take. The widths
# also supply the input and the output width of a network, so a group without 4 and 3 leaves the
# Iris target with no rules at all.
DEFAULT_LEARNING_RATES = (0.01, 0.001, 0.0001)
DEFAULT_DIMENSIONS = (3, 4, *range(5, 30))
DEFAULT_MAX_HIDDEN = 5


def load_iris(path: str | Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Read the Iris table and return its features and its one-hot labels.

    The class of a row is read from the ``class`` column rather than from the row's position. The
    shipped file is grouped by class in blocks of fifty, so the two agree on it, but only the
    column keeps agreeing on a file that is shuffled or extended, and a label taken from a position
    is wrong without saying so.

    The class index is the rank of the class name among the sorted distinct names, so it does not
    depend on the order the rows arrive in either.

    Args:
        path (str | Path): The CSV file. (Default value = the table shipped with the package)

    Returns:
        tuple[np.ndarray, np.ndarray]: The features, one row per sample and one column per entry of
            ``FEATURE_COLUMNS``, and the labels, one row per sample and one column per class.
    """
    table = pd.read_csv(str(path))
    features = table[list(FEATURE_COLUMNS)].to_numpy()
    classes = sorted(table[CLASS_COLUMN].unique())
    indices = table[CLASS_COLUMN].map({name: i for i, name in enumerate(classes)}).to_numpy()
    labels = np.identity(len(classes))[indices]
    return features, labels


class Simple_DNN_Repository:
    """The fourteen combinators that build a dense classifier, and the algebra that runs them.

    A term of this repository is a training system: a model, an optimizer, a loss function and a
    data loader, together with the epoch count and the batch size to train it with. The model is a
    stack of dense layers, and the number of hidden layers is carried in the type, which is what
    lets a target ask for a network of a given depth.

    Attributes:
        learning_rates (list[float]): The learning rates the two optimizers may carry.
        dimensions (list[int]): The widths a layer may take.
        max_hidden (int): The largest number of hidden layers a model may have.
        batch_sizes (list[int]): The batch sizes the data loader may take.
        epochs (list[int]): The epoch counts a training system may run for.
        train_dataset (torch.utils.data.TensorDataset): The data every candidate trains on.
    """

    def __init__(self, learning_rates: Sequence[float],
                 dimensions: Sequence[int],
                 max_hidden: int,
                 batch_sizes: Sequence[int],
                 epochs: Sequence[int],
                 dataset: torch.utils.data.TensorDataset):
        """Fix what the repository offers.

        Args:
            learning_rates (Sequence[float]): The learning rates the two optimizers may carry.
            dimensions (Sequence[int]): The widths a layer may take.
            max_hidden (int): The largest number of hidden layers a model may have.
            batch_sizes (Sequence[int]): The batch sizes the data loader may take.
            epochs (Sequence[int]): The epoch counts a training system may run for.
            dataset (torch.utils.data.TensorDataset): The data every candidate trains on.
        """
        self.learning_rates = list(learning_rates)
        self.dimensions = list(dimensions)
        self.max_hidden = max_hidden
        self.batch_sizes = list(batch_sizes)
        self.epochs = list(epochs)
        self.train_dataset = dataset

    def gamma(self) -> dict[str, Any]:
        """Build the repository, one specification per combinator.

        ``Model_cons`` stacks a layer onto a model and is the rule that counts the hidden layers.
        Its two hidden-layer parameters are coupled: ``m`` is the count of the model it extends and
        ``n`` the count of the model it builds, and the candidate function fixes ``m`` at ``n - 1``.
        The coupling also settles the base case. ``n = 0`` would need ``m = -1``, which the group
        does not hold, so the rule cannot fire there and only ``Model`` builds a model with no
        hidden layer.

        Returns:
            dict[str, Any]: The specification of each combinator, keyed by its name.
        """
        dimension = DataGroup("dimension", self.dimensions)
        bias = DataGroup("bool", [True, False])
        learning_rate = DataGroup("learning_rate", self.learning_rates)
        hidden = DataGroup("hidden", list(range(0, self.max_hidden + 1, 1)))
        batch_size = DataGroup("batch_size", self.batch_sizes)
        epochs = DataGroup("epochs", self.epochs)
        return {
            "Layer": SpecificationBuilder()
            .parameter("n", dimension)
            .parameter("bias", bias)
            .argument("af", Constructor("activation_function"))
            .suffix(Constructor("layer", Var("n"))),

            "Model": SpecificationBuilder()
            .parameter("in", dimension)
            .parameter("out", dimension)
            .argument("l", Constructor("layer", Var("out")))
            .suffix(
                Constructor("model",
                            Constructor("input", Var("in"))
                            & Constructor("output", Var("out")))
                & Constructor("hidden", Literal(0))
            ),

            "Model_cons": SpecificationBuilder()
            .parameter("in", dimension)
            .parameter("out", dimension)
            .parameter("neurons", dimension)
            .parameter("n", hidden)
            .parameter("m", hidden, lambda vs: [vs["n"]-1])
            .argument("layer", Constructor("layer", Var("neurons")))
            .argument("model",
                 Constructor("model",
                             Constructor("input", Var("neurons"))
                             & Constructor("output", Var("out")))
                 & Constructor("hidden", Var("m"))
                 )
            .suffix(
                Constructor("model",
                            Constructor("input", Var("in"))
                            & Constructor("output", Var("out"))
                            )
                & Constructor("hidden", Var("n"))
            ),

            "ReLu": Constructor("activation_function"),

            "ELU": Constructor("activation_function"),

            "Sigmoid": Constructor("activation_function"),

            "Softmax": Constructor("activation_function"),

            "MSE": Constructor("loss_function"),

            "CrossEntropy": Constructor("loss_function"),

            "L1": Constructor("loss_function"),

            "DataLoader": SpecificationBuilder()
            .parameter("bs", batch_size)
            .suffix(Constructor("data", Constructor("batch_size", Var("bs")))),

            "Adam": SpecificationBuilder()
            .parameter("lr", learning_rate)
            .suffix(Constructor("optimizer", Constructor("learning_rate", Var("lr")))),

            "SGD": SpecificationBuilder()
            .parameter("lr", learning_rate)
            .suffix(Constructor("optimizer", Constructor("learning_rate", Var("lr")))),

            "System": SpecificationBuilder()
            .parameter("in", dimension)
            .parameter("out", dimension)
            .parameter("n", hidden)
            .parameter("lr", learning_rate)
            .parameter("ep", epochs)
            .parameter("bs", batch_size)
            .argument("data", Constructor("data", Constructor("batch_size", Var("bs"))))
            .argument("m", Constructor("model", Constructor("input", Var("in"))
                                       & Constructor("output", Var("out")))
                      & Constructor("hidden", Var("n")))
            .argument("opt", Constructor("optimizer", Constructor("learning_rate", Var("lr"))))
            .argument("l", Constructor("loss_function"))
            .suffix(
                Constructor("system",
                            Constructor("input_dim", Var("in"))
                            & Constructor("output_dim", Var("out"))
                            )
                & Constructor("learning_rate", Var("lr"))
                & Constructor("hidden_layer", Var("n"))
                & Constructor("epochs", Var("ep"))
                & Constructor("batch_size", Var("bs"))
            ),
        }

    @staticmethod
    def train_loop(dataloader, model, loss_fn, opti, batch_size) -> None:
        """Train the model for one pass over the data.

        Args:
            dataloader: The batches of one pass.
            model: The network to train.
            loss_fn: The loss to descend.
            opti: The optimizer, still waiting for the parameters to step.
            batch_size (int): The batch size, which the data loader has already applied. It is a
                parameter because the algebra passes it, and nothing here reads it.
        """
        model.train()
        for X, y in dataloader:
            X = X.float()
            y = y.float()
            pred = model(X)
            loss = loss_fn(pred, y)

            # build the optimizer
            optimizer = opti(model.parameters())

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    @staticmethod
    def test_loop(dataloader, model, loss_fn) -> tuple[float, float]:
        """Score the trained model on the data it trained on.

        The accuracy carries an offset of 1e-4, so it stays strictly positive even for a model
        that classifies nothing correctly. Nothing here reads it: the objective is built from the
        loss alone.

        Args:
            dataloader: The batches to score.
            model: The trained network.
            loss_fn: The loss to report.

        Returns:
            tuple[float, float]: The accuracy in percent, and the mean loss per batch.
        """
        # Evaluation mode matters for batch normalization and dropout, which this repository does
        # not build, and no_grad keeps the scoring pass from holding gradients it never reads.
        model.eval()
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        test_loss, correct = 0.0, 0.0

        with torch.no_grad():
            for X, y in dataloader:
                X = X.float()
                y = y.float()
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= size
        return 100 * correct + 0.0001, test_loss

    def system(self, dataloader, model, loss_fn, optimizer, batch_size, epochs
               ) -> tuple[float, float]:
        """Train the model for the given number of epochs, then score it.

        Args:
            dataloader: The batches of one epoch.
            model: The network to train.
            loss_fn: The loss to descend.
            optimizer: The optimizer, still waiting for the parameters to step.
            batch_size (int): The batch size, passed on to the training pass.
            epochs (int): How many passes over the data to train for.

        Returns:
            tuple[float, float]: The accuracy in percent, and the mean loss per batch.
        """
        for _epoch in range(epochs):
            self.train_loop(dataloader, model, loss_fn, optimizer, batch_size)
        return self.test_loop(dataloader, model, loss_fn)

    def torch_algebra(self) -> dict[str, Any]:
        """Build the algebra that turns a term into a trained network and its score.

        Returns:
            dict[str, Any]: One interpretation per combinator, keyed by its name.
        """
        return {
            "System": (lambda i, o, n, lr, ep, bs, data, m, opt, l:
                       self.system(data, m, l, opt, bs, ep)),
            "Layer": (lambda n, b, af: (b, af)),
            # The layer's activation function is dropped here on purpose: the output layer of a
            # classifier ends in the softmax that reads the class scores.
            "Model": (lambda i, o, l: nn.Sequential(nn.Linear(i, o, l[0]), nn.Softmax(dim=1))),
            "Model_cons": (lambda i, o, neurons, n, m, l, model:
                           nn.Sequential(nn.Linear(i, neurons, l[0]), l[1]).extend(model)),
            "ReLu": nn.ReLU(),
            "ELU": nn.ELU(),
            "Sigmoid": nn.Sigmoid(),
            "Softmax": nn.Softmax(dim=1),
            "MSE": nn.MSELoss(),
            "CrossEntropy": nn.CrossEntropyLoss(),
            "L1": nn.L1Loss(),
            "Adam": (lambda lr, params: torch.optim.Adam(params, lr=lr)),
            "SGD": (lambda lr, params: torch.optim.SGD(params, lr=lr)),
            "DataLoader": (lambda bs: torch.utils.data.DataLoader(self.train_dataset,
                                                                  batch_size=bs)),
        }


def build_repository(features: np.ndarray,
                     labels: np.ndarray,
                     *,
                     epochs: int,
                     batch_size: int,
                     learning_rates: Sequence[float] = DEFAULT_LEARNING_RATES,
                     dimensions: Sequence[int] = DEFAULT_DIMENSIONS,
                     max_hidden: int = DEFAULT_MAX_HIDDEN) -> Simple_DNN_Repository:
    """Wrap the data in a tensor dataset and build the repository over it.

    ``epochs`` and ``batch_size`` are single values rather than lists because a run trains every
    candidate the same way, and the two are what one candidate costs. The repository holds them as
    one-element groups, so neither of the two multiplies the space. What is still open above a
    model is the optimizer, its learning rate and the loss function, which is eighteen training
    systems per model.

    Args:
        features (np.ndarray): The features, one row per sample.
        labels (np.ndarray): The one-hot labels, one row per sample.
        epochs (int): How many passes over the data each candidate trains for.
        batch_size (int): The batch size each candidate trains with.
        learning_rates (Sequence[float]): The learning rates the two optimizers may carry.
        dimensions (Sequence[int]): The widths a layer may take.
        max_hidden (int): The largest number of hidden layers a model may have.

    Returns:
        Simple_DNN_Repository: The repository over that data.
    """
    dataset = torch.utils.data.TensorDataset(torch.tensor(features), torch.tensor(labels))
    return Simple_DNN_Repository(learning_rates, dimensions, max_hidden,
                                 [batch_size], [epochs], dataset)


def network_target(input_dim: int, output_dim: int) -> Type:
    """Build the target that asks for a training system of the given input and output width.

    Everything else stays open. The learning rate, the number of hidden layers, the epoch count and
    the batch size are conjuncts the suffix of ``System`` offers, and naming one of them here would
    pin it for the whole search instead of leaving it to the loop.

    Args:
        input_dim (int): The number of features a candidate reads.
        output_dim (int): The number of classes a candidate predicts.

    Returns:
        Type: The synthesis target.
    """
    return Constructor("system",
                       Constructor("input_dim", Literal(input_dim))
                       & Constructor("output_dim", Literal(output_dim)))


def build_search_space(repository: Simple_DNN_Repository, target: Type) -> SolutionSpace:
    """Synthesize the search space of the target over the repository.

    Args:
        repository (Simple_DNN_Repository): The combinators to search over.
        target (Type): What the terms of the space have to inhabit.

    Returns:
        SolutionSpace: The space, started at the target.
    """
    return Synthesizer(repository.gamma(), {}).construct_solution_space(target)


def build_acquisition_optimizer(
    *,
    population_size: int = DEFAULT_POPULATION_SIZE,
    generations: int = DEFAULT_GENERATIONS,
    depth_bound: int = DEFAULT_DEPTH_BOUND,
    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    seed: int = 0,
) -> EvolutionarySearch[Any, Any, Any]:
    """Build the evolutionary search that maximizes the acquisition function.

    Each component draws from its own generator, derived from ``seed``, so that changing what one
    operator consumes does not shift every later draw of the run.

    Survivor selection is generous and conservative: every individual survives with positive
    probability, and an individual of greatest scalarized fitness always survives. That is the one
    of the five conditions for almost sure convergence that falls to survivor selection. Fitness
    proportional selection carries the one on parent selection, and the two rates below carry the
    one on the rates. The example used to select survivors by age, which today's cosy no longer
    offers.

    Args:
        population_size (int): The population size.
        generations (int): The termination bound.
        depth_bound (int): The bound of the samplers, on the depth of a term.
        crossover_rate (float): The crossover rate, which the convergence conditions need below 1.
        mutation_rate (float): The mutation rate, which they need above 0.
        seed (int): The base seed. Every component derives its own generator from it.

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
        recombination=SubtreeSwap(rng(3)),
        parent_selection=FitnessProportionalSelection(ExpScalarization(), rng(4)),
        survivor_selection=GenerousConservativeReplacement(ExpScalarization(), rng(5)),
        termination=Generations(generations),
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        rng=rng(6),
        comparator=ScalarFitnessComparator(True),  # the acquisition is always maximized
    )


def make_objective(repository: Simple_DNN_Repository) -> Callable[[Tree[Any]], float]:
    """Build the objective that trains a candidate and reports its negated training loss.

    The loop maximizes, and the quantity this example searches for is a loss, so the objective
    negates it. A caller reading a run's ``best_y`` therefore reads a negated loss.

    Args:
        repository (Simple_DNN_Repository): The repository whose algebra runs the term.

    Returns:
        Callable[[Tree[Any]], float]: The objective.
    """
    algebra = repository.torch_algebra()

    def objective(term: Tree[Any]) -> float:
        """Train the candidate and report the negated training loss.

        Args:
            term (Tree[Any]): The candidate.

        Returns:
            float: The negated mean loss per batch.
        """
        _accuracy, loss = term.interpret(algebra)
        return -float(loss)

    return objective


def run_search(repository: Simple_DNN_Repository,
               search_space: SolutionSpace,
               target: Type,
               *,
               budget: int,
               initial_size: int,
               optimizer: EvolutionarySearch[Any, Any, Any] | None = None,
               depth_bound: int = DEFAULT_DEPTH_BOUND,
               seed: int = 0,
               verbose: bool = False) -> dict[str, Any]:
    """Run the closed Bayesian optimization loop over the space and return its result.

    The kernel is the Weisfeiler-Lehman kernel over the terms themselves, which is how the
    surrogate is fitted over terms rather than over vectors.

    Args:
        repository (Simple_DNN_Repository): The repository whose algebra scores a candidate.
        search_space (SolutionSpace): The space to search.
        target (Type): The query the loop draws its own terms from.
        budget (int): How many passes the loop runs after the initial design.
        initial_size (int): How many terms the initial design holds.
        optimizer (EvolutionarySearch[Any, Any, Any] | None): The acquisition optimizer.
            (Default value = None, meaning the configuration above)
        depth_bound (int): The bound of the loop's own sampler, on the depth of a term.
        seed (int): The base seed. The loop's sampler takes an offset of its own, outside the
            range the acquisition optimizer's components use, so the two draw different streams.
        verbose (bool): Log one line per pass.

    Returns:
        dict[str, Any]: The result of the loop. ``best_tree`` is the candidate it returns and
            ``best_y`` its negated training loss.
    """
    loop = BayesianOptimization(
        search_space=search_space,
        request=target,
        kernel=WeisfeilerLehmanKernel(),
        optimizer=(build_acquisition_optimizer(seed=seed) if optimizer is None else optimizer),
        seed=seed,
        # One object for both roles: the initial design draws from it, and so does the duplicate
        # fallback of a pass.
        sampler=DepthBoundedRandomSampler(depth_bound, random.Random(seed * 100 + 7)),
    )
    return loop.optimize(make_objective(repository), budget,
                         initial_size=initial_size, verbose=verbose)


def main(*,
         epochs: int = 200,
         batch_size: int = 8,
         budget: int = 5,
         initial_size: int = 10,
         population_size: int = DEFAULT_POPULATION_SIZE,
         generations: int = DEFAULT_GENERATIONS,
         depth_bound: int = DEFAULT_DEPTH_BOUND,
         seed: int = 0,
         verbose: bool = True) -> dict[str, Any]:
    """Load the data, synthesize the space, run the loop and report the best candidate.

    Every number a run costs is an argument here. The epoch count and the batch size decide what
    one candidate costs, and the population size and the generation count decide what one
    acquisition maximization costs, which is the larger of the two at the settings above.

    Args:
        epochs (int): How many passes over the data each candidate trains for.
        batch_size (int): The batch size each candidate trains with.
        budget (int): How many passes the loop runs after the initial design.
        initial_size (int): How many terms the initial design holds.
        population_size (int): The population of the acquisition optimizer.
        generations (int): The termination bound of the acquisition optimizer.
        depth_bound (int): The bound of every sampler, on the depth of a term.
        seed (int): The seed of the loop and of the acquisition optimizer.
        verbose (bool): Log one line per pass.

    Returns:
        dict[str, Any]: The result of the loop.
    """
    features, labels = load_iris()
    repository = build_repository(features, labels, epochs=epochs, batch_size=batch_size)
    target = network_target(features.shape[1], labels.shape[1])
    search_space = build_search_space(repository, target)

    optimizer = build_acquisition_optimizer(population_size=population_size,
                                            generations=generations,
                                            depth_bound=depth_bound, seed=seed)
    result = run_search(repository, search_space, target,
                        budget=budget, initial_size=initial_size, optimizer=optimizer,
                        depth_bound=depth_bound, seed=seed, verbose=verbose)

    best = result["best_tree"]
    print(f"Best tree found by Bayesian optimization:\n{best}\nwith loss: {-result['best_y']}")

    accuracy, loss = best.interpret(repository.torch_algebra())
    print(f"Evaluating best model once:\naccuracy: {accuracy}%, \nloss: {loss}")
    return result


if __name__ == "__main__":
    main()
