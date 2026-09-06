"""Bayesian optimization over the DAMG search space, on a one-dimensional regression problem.

This is the older of the two DAMG examples in this package. It searches directed acyclic
multigraphs of linear layers, activations and pointwise arithmetic for a network that fits a
trapezoid function, and the
CNN example next door is its successor: the same repository shape, the same loop, extended by
convolutions and run on CIFAR-10. This one stays because it is small enough to read in one sitting
and to run on a laptop, and it is kept in working order rather than modernized past recognition.

The four things a reader should take from it, in the order the script does them:

1. ``DAMGrepository.specification`` and one of the targets in ``damg_targets`` give a search space
   of networks whose structure has the requested length and whose dimensions chain up.
2. ``pytorch_function_algebra`` turns a term of that space into a training pipeline, so an
   objective value is an actual training run and not a surrogate cost.
3. :func:`build_acquisition_optimizer` assembles the evolutionary search that maximizes the
   acquisition function over that space.
4. ``BayesianOptimization.optimize`` runs the closed loop and returns the term of best observed
   value.

Two things here are easy to get wrong and neither says what it is, so both are written down where
they happen. The epoch count of the target has to be one the repository offers, or the space is
empty and the loop refuses the target as a request it has no rules for, without naming the epoch
count that caused it: :func:`epochs_of` is why this script cannot get that pair wrong. And which sampler the loop draws from decides what a run costs before a single
network is trained: see :func:`build_sampler`.
"""

import argparse
import math
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from cosy.core import Synthesizer
from cosy.core.types import Constructor, Intersection, Literal, Type
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
from cosy.search.samplers import DepthBoundedRandomSampler

from bayesian_optimization import BayesianOptimization
from bayesian_optimization.examples.damg_nas.damg_kernels import noisy_hierarchical_damg_kernel
from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository
from bayesian_optimization.examples.damg_nas.damg_repo_algebras import (
    pretty_term_algebra,
    pytorch_function_algebra,
)
from bayesian_optimization.examples.damg_nas.damg_targets import target_len_3

#: The bound on the depth of a term, for every sampler this example builds. Two hundred terms drawn
#: from the ``target_len_3`` space have depth 6 to 8 and from the ``target_len_6`` space 9 to 11, so
#: this bound constrains nothing the shipped targets ask for. It exists because a sampler is bounded
#: by definition, which is what makes an empty draw an observation rather than a hang.
DEPTH_BOUND = 100

#: The linear feature widths a layer of the searched networks may have.
LINEAR_FEATURE_DIMENSIONS = [1, 2, 3, 4, 5]

#: The constants a sum component may carry. A product component takes the same list without 0.
CONSTANT_VALUES = [0, 1, -1]

#: The learning rates the Adam optimizer may be built with.
LEARNING_RATE_VALUES = [1e-2]


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

def epochs_of(target: Type) -> int:
    """Read the epoch count a target asks for off the target itself.

    The repository enumerates the epoch counts it was given and the target names one of them, so
    the two have to agree. They used to be written twice, once as ``n_epoch_values=[2000]`` and
    once inside the target. A disagreement between them empties the space, and the loop then
    refuses the target as a request it has no rules for, in a message about the request that never
    names the epoch count that emptied it. Reading the count off the target removes the second
    place it could be written.

    Args:
        target (Type): The requested type, an intersection one of whose members is
            ``Constructor("epochs", Literal(n))``.

    Returns:
        int: The epoch count.

    Raises:
        ValueError: If the target names no epoch count, or more than one. Both are targets no
            repository can be built to match, so neither has a sensible default.
    """
    found: list[int] = []

    def collect(node: Type) -> None:
        """Collect the value of every ``epochs`` constructor below a type.

        Args:
            node (Type): The subtype to descend into.
        """
        if isinstance(node, Intersection):
            collect(node.left)
            collect(node.right)
        elif isinstance(node, Constructor):
            if node.name == "epochs" and isinstance(node.arg, Literal):
                found.append(node.arg.value)
            else:
                collect(node.arg)

    collect(target)
    if len(found) != 1:
        msg = (
            f"a target names exactly one epoch count and this one names {len(found)}: {found}. "
            f"The repository offers the counts it is given and the target picks one of them, so a "
            f"target without one is inhabited by nothing at all"
        )
        raise ValueError(msg)
    return int(found[0])


def build_repository(target: Type, *, max_parallel_width: int = 3) -> DAMGrepository:
    """Build the repository whose epoch set is the one the target asks for.

    Args:
        target (Type): The requested type, which names the epoch count.
        max_parallel_width (int): How many components may sit beside each other in one parallel
            composition. (Default value = 3)

    Returns:
        DAMGrepository: The repository.
    """
    return DAMGrepository(
        linear_feature_dimensions=LINEAR_FEATURE_DIMENSIONS,
        constant_values=CONSTANT_VALUES,
        learning_rate_values=LEARNING_RATE_VALUES,
        n_epoch_values=[epochs_of(target)],
        max_parallel_width=max_parallel_width,
    )


def build_search_space(repo: DAMGrepository, target: Type, *, verbose: bool = True):
    """Synthesize and prune the search space of a target, and report what it cost.

    Args:
        repo (DAMGrepository): The repository.
        target (Type): The requested type.
        verbose (bool): Print the construction time and the size of the program.
            (Default value = True)

    Returns:
        SolutionSpace: The pruned space, ready to be queried at ``target``.
    """
    started = time.time()
    space = Synthesizer(repo.specification(), {}).construct_solution_space(target).prune()
    seconds = time.time() - started
    if verbose:
        nonterminals = tuple(space.nonterminals())
        rules = sum(len(space.get(nt) or ()) for nt in nonterminals)
        print(
            f"Search space construction took {seconds:.5f} seconds "
            f"({len(nonterminals)} non-terminals, {rules} rules)."
        )
    return space


def build_sampler(seed: int) -> DepthBoundedRandomSampler:
    """Build the sampler the loop and the evolutionary operators draw their own terms from.

    Depth-bounded rather than size-uniform, and the difference is not a preference. A
    ``SizeUniformSampler`` counts the derivations of the space before its first draw, and
    ``DAMGrepository`` states its four swap laws as predicates over two sibling holes, so that
    counting has to walk the retained search tree. On the ``target_len_3`` space that walk took
    four orders of magnitude longer than three depth-bounded draws of the same space, 36 seconds
    against 0.004 on one laptop. The cost is paid once per query object and every further draw is
    free, so it is a fixed toll on a run rather than a per-draw cost, but it is a toll this example
    has no use for: nothing it does needs the uniform-over-sizes distribution that buys it. The CNN
    example next door does, and pays it on a determinized program where the count comes from a
    table instead of from the tree.

    Args:
        seed (int): The seed of the sampler's generator.

    Returns:
        DepthBoundedRandomSampler: The sampler, bounded by :data:`DEPTH_BOUND`.
    """
    return DepthBoundedRandomSampler(DEPTH_BOUND, random.Random(seed))


def build_acquisition_optimizer(
    *,
    population_size: int = 100,
    generations: int = 100,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.02,
    selection_pressure: float = 1.7,
    seed: int = 0,
) -> EvolutionarySearch:
    """Build the evolutionary search that maximizes the acquisition function over the space.

    Every component gets its own generator, derived from ``seed``, so that changing what one
    operator consumes does not shift every later draw of the run.

    The two rates are the ones the almost sure convergence of the search asks for: a crossover rate
    below 1, so that a pass can copy its parents unchanged, and a mutation rate above 0, so that
    mutation happens at all. The example used to pass 0.99 and 0.00, and the second of those is on
    the wrong side of the condition.

    Rank-based rather than fitness-proportional parent selection, which is what the example used to
    ask for. A proportional draw weighs an individual by its fitness read through a strictly
    positive map, and here the fitness is the acquisition value, which spans orders of magnitude on
    a confident surrogate. Exponentiating those values to get positive weights overflows: on a two-pass run over ``target_len_3`` with one training epoch it raised on a
    fitness of 1909, well past the 710 at which ``math.exp`` leaves the floats. Rank selection reads
    the ordering alone, so no magnitude reaches it, and it still gives every member of the
    population a positive probability as long as the pressure stays below 2, which is one of the
    convergence conditions.

    Args:
        population_size (int): The population size. (Default value = 100)
        generations (int): How many generations one acquisition maximization runs.
            (Default value = 100)
        crossover_rate (float): The crossover rate, below 1. (Default value = 0.9)
        mutation_rate (float): The mutation rate, above 0. (Default value = 0.02)
        selection_pressure (float): The rank-selection pressure, below 2 so that the worst
            individual keeps a positive share. (Default value = 1.7)
        seed (int): The base seed. Every component derives its own generator from it.

    Returns:
        EvolutionarySearch: The configured search.
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
        initializer=SampledInitialization(build_sampler(seed * 100)),
        mutation=ResolutionMutation(build_sampler(seed * 100 + 1), rng(2)),
        recombination=SubtreeSwap(rng(3)),
        parent_selection=RankBasedSelection(selection_pressure, rng=rng(4)),
        # Generous and conservative: every individual survives with positive probability and a
        # fittest one always survives, which is the condition on survivor selection that carries
        # the convergence argument.
        survivor_selection=GenerousConservativeReplacement(ExpScalarization(), rng(5)),
        termination=Generations(generations),
        population_size=population_size,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        rng=rng(6),
        comparator=ScalarFitnessComparator(True),  # the acquisition is always maximized
    )


def make_objective(x, y, x_test, y_test):
    """Build the objective the loop maximizes: the negated ``log1p`` of a trained term's test loss.

    Two transformations, and each is here for a reason the loop states rather than performs.

    The **negation** is because the loop maximizes throughout, while a loss is better when it is
    smaller. Passing a direction flag to the loop instead was the older arrangement and it is gone.

    The **logarithm** is because the raw loss on this problem is unusable as a fitness. Twelve terms
    drawn from a length-2 space and trained for one epoch spread their test loss from 5.4 to
    8.2e8, eight orders of magnitude, because a network whose dimensions chain up badly diverges
    while a workable one fits. A surrogate fitted on values that far apart returns acquisition
    values just as far apart, and the evolutionary run reads those through an exponential to get
    proportional weights, which overflows no matter what temperature is set: a scale small enough to tell two good networks apart puts a diverged one past the 710 at
    which ``math.exp`` leaves the range, and a scale large enough to hold it makes every draw
    uniform. ``log1p`` compresses the same twelve values into 1.9 to 20.5, where both the
    exponential and a Gaussian process on the values are well behaved. It is strictly decreasing in
    the loss, so the best term is the same term, and the loop's own contract puts a rescaling of
    the objective here rather than inside the loop.

    Args:
        x (torch.Tensor): The training inputs.
        y (torch.Tensor): The training targets.
        x_test (torch.Tensor): The test inputs.
        y_test (torch.Tensor): The test targets.

    Returns:
        Callable[[Tree], float]: The objective.
    """
    def objective(term) -> float:
        """Train the network a term denotes and score it.

        Args:
            term (Tree): The term.

        Returns:
            float: ``-log(1 + loss)`` on the test set.

        Raises:
            ValueError: If the loss is negative, which no loss this repository offers can be. Both
                of them are non-negative, and a negative one would leave the logarithm undefined,
                so it is a term that does not mean what it says rather than a bad candidate.
        """
        train = term.interpret(pytorch_function_algebra())
        loss = float(train(x, y, x_test, y_test))
        if loss < 0.0:
            msg = f"a loss cannot be negative and this term produced {loss}"
            raise ValueError(msg)
        # A non-finite loss stays non-finite here, and the loop refuses it rather than scoring it.
        return -math.log1p(loss)

    return objective


def main(
    target: Type = target_len_3,
    *,
    budget: int = 5,
    initial_size: int = 10,
    n_samples: int = 1_000,
    train_range: tuple[float, float] = (-10.0, 10.0),
    test_range: tuple[float, float] = (-15.0, 15.0),
    noise: float = 1e-4,
    population_size: int = 100,
    generations: int = 100,
    seed: int = 0,
    verbose: bool = True,
) -> dict:
    """Run the whole example: build the space, run the loop, print the best network found.

    Args:
        target (Type): The requested type. It names the epoch count, and the repository is built to
            offer that one. (Default value = ``target_len_3``)
        budget (int): How many passes the loop runs after the initial design. (Default value = 5)
        initial_size (int): How many terms the initial design holds. (Default value = 10)
        n_samples (int): How many points the training and the test set hold each.
            (Default value = 1000)
        train_range (tuple): The interval the training inputs cover.
        test_range (tuple): The interval the test inputs cover, wider than the training one so that
            the reported loss measures extrapolation too.
        noise (float): The variance of the Gaussian noise on the generated targets.
        population_size (int): The population of one acquisition maximization.
            (Default value = 100)
        generations (int): The generations of one acquisition maximization. (Default value = 100)
        seed (int): The seed every generator of the run derives from.
        verbose (bool): Print the progress of the run.

    Returns:
        dict: The result of ``BayesianOptimization.optimize``. ``best_y`` is what the loop
            maximized, so the test loss of the best term is ``expm1(-best_y)``.
    """
    torch.manual_seed(seed)
    repo = build_repository(target)
    if verbose:
        print(f"Using target: {target}")

    x, y = generate_data(
        TrapezoidNetPure(), xmin=train_range[0], xmax=train_range[1], n_samples=n_samples,
        eps=noise,
    )
    x_test, y_test = generate_data(
        TrapezoidNetPure(), xmin=test_range[0], xmax=test_range[1], n_samples=n_samples,
        eps=noise,
    )

    space = build_search_space(repo, target, verbose=verbose)

    optimizer = BayesianOptimization(
        space,
        target,
        kernel=noisy_hierarchical_damg_kernel,
        # Fit the kernel hyperparameters. The kernel sums three granularities, each weighted by a
        # ConstantKernel, and its theta holds those three weights plus the noise level of the
        # WhiteKernel around them. Without an optimizer all four stay at the 0.01 they were written
        # with for the whole run, and the loop says so out loud once per run. The CNN example next door does the same thing for the same reason.
        kernel_optimizer="fmin_l_bfgs_b",
        n_restarts_kernel_optimizer=20,
        optimizer=build_acquisition_optimizer(
            population_size=population_size, generations=generations, seed=seed,
        ),
        acquisition_function="ExpectedImprovement",
        # The loop draws its own terms, the initial design and the fallback for a duplicate
        # suggestion, and it draws them from this sampler rather than from the evolutionary
        # search's own. The offset keeps it off the seeds the search derives for its components,
        # which at seed 0 would otherwise be the same number and the same stream.
        sampler=build_sampler(seed * 100 + 7),
        seed=seed,
    )

    if verbose:
        print("Starting Bayesian Optimization")
    started = time.time()
    result = optimizer.optimize(
        make_objective(x, y, x_test, y_test),
        budget,
        initial_size=initial_size,
        verbose=verbose,
    )
    seconds = time.time() - started

    if verbose:
        print(f"Bayesian Optimization took {seconds:.5f} seconds.")
        best = result["best_tree"].interpret(pretty_term_algebra())
        loss = math.expm1(-result["best_y"])
        print(f"Best tree with test loss {loss:.5f}:\n{best}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--budget", type=int, default=5,
                        help="how many passes the loop runs after the initial design")
    parser.add_argument("--initial-size", type=int, default=10,
                        help="how many terms the initial design holds")
    parser.add_argument("--samples", type=int, default=1_000,
                        help="how many points the training and the test set hold each")
    parser.add_argument("--population-size", type=int, default=100,
                        help="the population of one acquisition maximization")
    parser.add_argument("--generations", type=int, default=100,
                        help="the generations of one acquisition maximization")
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed every generator of the run derives from")
    args = parser.parse_args()
    main(
        budget=args.budget,
        initial_size=args.initial_size,
        n_samples=args.samples,
        population_size=args.population_size,
        generations=args.generations,
        seed=args.seed,
    )
