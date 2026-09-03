import time

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from cosy.core import Synthesizer
from cosy.evolutionary_algorithms import (
    AgeBasedReplacement,
    Crossover,
    FitnessProportionalSelection,
    RandomLimitedDepthFirstInitialization,
    ResolutionMutation,
    ScalarFitnessComparator,
    SimpleGeneticProgramming,
)

from bayesian_optimization import BayesianOptimization
from bayesian_optimization.examples.damg_nas.damg_kernels import noisy_hierarchical_damg_kernel
from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository
from bayesian_optimization.examples.damg_nas.damg_repo_algebras import (
    pretty_term_algebra,
    pytorch_function_algebra,
)
from bayesian_optimization.examples.damg_nas.damg_targets import target_len_3


def get_num_parameters(model):
    """
    Calculate the total number of parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to analyze

    Returns:
        int: Total number of parameters in the model

    Example:
        >>> model = nn.Linear(10, 5)
        >>> num_params = get_num_parameters(model)
        >>> print(f"Model has {num_params} parameters")
    """
    return sum(p.numel() for p in model.parameters())


def generate_data(true_model, n_samples, xmin=-10, xmax=10, eps=0.0):
    """
    Generate synthetic data by evaluating a true model on a grid of points and adding Gaussian noise.

    Note:
        model evaluation is not batched!

    Args:
        true_model (torch.nn.Module): The true model to evaluate
        n_samples (int): Number of samples to generate
        xmin (float, optional): Minimum x value. Defaults to -10.
        xmax (float, optional): Maximum x value. Defaults to 10.
        eps (float, optional): Variance of Gaussian noise to add. Defaults to 0.0.

    Returns:
        tuple: A tuple containing (x, y) where x is the input tensor and y is the output tensor with optional noise added

    This helper returns tensors shaped for the synthetic regression examples below.
    """
    x = torch.linspace(xmin, xmax, n_samples).view(-1, 1)
    y = true_model(x).detach().view(-1)

    if eps > 0:
        noise = torch.normal(mean=0.0, std=float(torch.sqrt(torch.tensor(eps))), size=y.shape)
        y = y + noise

    return x, y


def fit_model(model, x, y, n_epochs=2_000, verbose=True, name="model"):
    """
    Train a PyTorch model on given data using Adam optimizer and MSE loss.

    Note:
        Technically, this function implements Gradient Descent and not Stochastic Gradient Descent. There is no proper batching performed.

    Args:
        model (torch.nn.Module): The model to train
        x (torch.Tensor): Input tensor
        y (torch.Tensor): Target tensor
        n_epochs (int, optional): Number of training epochs. Defaults to 2000.
        verbose (bool, optional): Whether to show progress bar. Defaults to True.
        name (str, optional): Name of the model for progress bar display. Defaults to "model".

    Returns:
        torch.nn.Module: The trained model

    The function returns the trained model instance.
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    pbar = tqdm.tqdm(range(n_epochs), total=n_epochs, desc=f"Training {name}", disable=not verbose)

    for _ in pbar:
        optimizer.zero_grad()
        pred = model(x).ravel()
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    return model

class TrapezoidNetPure(nn.Module):
        def __init__(self, random_weights=False, sharpness=None):
            super().__init__()

            self.split = nn.Linear(1, 1, bias=True)
            self.left = nn.Linear(1, 1, bias=True)
            self.right = nn.Linear(1, 1, bias=True)
            self.sharpness = sharpness

            if not random_weights:
                with torch.no_grad():
                    # For left branch (x <= 0): we want output = x + 10
                    # So left(x) = x + 10 => weight = 1, bias = 10
                    self.left.weight.data.fill_(1.0)
                    self.left.bias.data.fill_(10.0)

                    # For right branch (x > 0): we want output = 10 - x
                    # So right(x) = 10 - x => weight = -1, bias = 10
                    self.right.weight.data.fill_(-1.0)
                    self.right.bias.data.fill_(10.0)

                self.split.weight.data.fill_(1.0)
                self.split.bias.data.fill_(0.0)

        def forward(self, x):
            if not self.sharpness:
                gate = (self.split(x) <= 0).float()
            else:
                gate = torch.sigmoid(-self.sharpness * self.split(x))

            left_out = self.left(x) * gate
            right_out = self.right(x) * (1 - gate)

            return left_out + right_out


if __name__ == "__main__":

    linear_feature_dimensions = [1, 2, 3, 4, 5, ]
    constant_values = [0, 1, -1]
    learning_rate_values = [1e-2, ]  # 5e-3, 1e-3]
    max_depth = 10000

    repo = DAMGrepository(linear_feature_dimensions=linear_feature_dimensions, constant_values=constant_values,
                         learning_rate_values=learning_rate_values,
                         n_epoch_values=[2000])

    target = target_len_3

    n_samples = 1_000
    train_xmin = -10
    train_xmax = 10
    test_xmin = -15
    test_xmax = 15
    eps = 1e-4

    x, y = generate_data(TrapezoidNetPure(), xmin=train_xmin, xmax=train_xmax, n_samples=n_samples, eps=eps)
    x_test, y_test = generate_data(TrapezoidNetPure(), xmin=test_xmin, xmax=test_xmax, n_samples=n_samples, eps=eps)

    def f_obj(t):
        learner = t.interpret(pytorch_function_algebra())
        return learner(x, y, x_test, y_test)

    print(f"Using target: {target}")

    synthesizer = Synthesizer(repo.specification(), {})

    start_time = time.time()
    search_space = synthesizer.construct_solution_space(target).prune()
    end_time = time.time()
    construction_time = end_time - start_time
    print(f"Search Space construction took {construction_time:.5f} seconds.")

    kernel = noisy_hierarchical_damg_kernel

    def termination(state) -> bool:
        return state.generation >= 100

    initialization = RandomLimitedDepthFirstInitialization(search_space, target, max_depth=max_depth)
    mutation = ResolutionMutation(search_space, target, max_depth=max_depth)
    recombination = Crossover(search_space, target, max_depth=max_depth)
    parent_selection = FitnessProportionalSelection()
    survivor_selection = AgeBasedReplacement()
    fitness_comparator = ScalarFitnessComparator(True)  # EI is always maximized in acquisition optimization.

    evo_alg = SimpleGeneticProgramming(  # type: ignore[arg-type]
        search_space,
        target,
        termination,
        initialization,
        mutation,
        recombination,
        parent_selection,
        survivor_selection,
        fitness_comparator,
    )

    optimizer = BayesianOptimization(search_space, target, kernel=kernel, optimizer=evo_alg,
                                     optimizer_population_size=100, acquisition_function="ExpectedImprovement",
                                     optimizer_mutation_rate=0.00, optimizer_recombination_rate=0.99)

    print("Starting Bayesian Optimization")
    start_time = time.time()
    result = optimizer.bayesian_optimisation(5, f_obj, n_pre_samples=10, greater_is_better=False, ei_xi=0.01,
                                             max_depth=100, verbose=True)
    end_time = time.time()
    bo_time = end_time - start_time
    print(f"Bayesian Optimization took {bo_time:.5f} seconds.")
    print(f"Best tree with loss {result['best_y']:.5f} found by bayesian optimization:\n{result['best_tree'].interpret(pretty_term_algebra())}")






