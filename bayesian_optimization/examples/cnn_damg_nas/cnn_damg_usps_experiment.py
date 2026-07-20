"""First real NAS search experiment (not a single pinned reference architecture): a genuine
Bayesian-optimization search over a CNN search space on USPS.

Deliberately scoped as a *small validation run* - the goal is to confirm the full pipeline
(synthesis, training, GP kernel, evolutionary acquisition optimization) works correctly end to end
on real hardware (CPU here, the A30 GPU server next), not to find the best possible architecture.
Sized so it's practical to run to completion both locally (CPU) and on the A30, for a direct
wall-clock comparison.

Usage:
    python -m bayesian_optimization.examples.cnn_damg_nas.cnn_damg_usps_experiment
    python -m bayesian_optimization.examples.cnn_damg_nas.cnn_damg_usps_experiment --n-pre-samples 2 --n-iterations 2
"""

import argparse
import os
import time

import torch
import torchvision
from torchvision import transforms

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
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo_algebras import (
    pretty_term_algebra,
    pytorch_function_algebra,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_kernels import noisy_hierarchical_damg_kernel
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_targets import usps_experiment_target
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_experiment_utils import ExperimentCSVLogger

# Search space: richer than the single-architecture reference tests (more channel/kernel/spatial/
# feature choices), but still validated locally for construct_solution_space tractability
# (~70s one-time setup cost on CPU - see cnn_damg_nas tests/development notes).
LINEAR_FEATURE_DIMENSIONS = [256, 768, 400, 192, 64, 30, 16, 10]
CHANNEL_DIMENSIONS = [1, 4, 8, 12]
HEIGHT_WIDTH_DIMENSIONS = [(16, 16), (8, 8), (4, 4)]
KERNEL_DIMENSIONS = [(3, 3), (5, 5)]
STRIDE_VALUES = [1, 2]
PADDING_VALUES = [0, 1, 2]
MAX_PARALLEL_WIDTH = 2

CONSTANT_VALUES = [0, 1, -1]
LEARNING_RATE_VALUES = [1e-3]
N_EPOCH_VALUES = [50]  # must match usps_experiment_target's epochs
BATCH_SIZE = 64
MAX_DEPTH = 50

DATA_DIR = "./data"


def load_usps(data_dir: str = DATA_DIR):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.USPS(root=data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.USPS(root=data_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def dataset_to_tensors(dataset, device):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    return images.reshape(images.shape[0], -1).to(device), labels.to(device).long()


def run_experiment(n_pre_samples: int, n_iterations: int, population_size: int, evo_generations: int,
                    csv_path: str, data_dir: str = DATA_DIR, verbose: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_set, test_set = load_usps(data_dir)
    x, y = dataset_to_tensors(train_set, device)
    x_test, y_test = dataset_to_tensors(test_set, device)
    print(f"Loaded USPS: train={tuple(x.shape)}, test={tuple(x_test.shape)}")

    repo = CNNrepository(
        linear_feature_dimensions=LINEAR_FEATURE_DIMENSIONS,
        constant_values=CONSTANT_VALUES,
        learning_rate_values=LEARNING_RATE_VALUES,
        n_epoch_values=N_EPOCH_VALUES,
        channel_dimensions=CHANNEL_DIMENSIONS,
        height_width_dimensions=HEIGHT_WIDTH_DIMENSIONS,
        kernel_dimensions=KERNEL_DIMENSIONS,
        stride_values=STRIDE_VALUES,
        padding_values=PADDING_VALUES,
        max_parallel_width=MAX_PARALLEL_WIDTH,
    )
    target = usps_experiment_target

    def f_obj(t):
        learner_fn = t.interpret(pytorch_function_algebra())
        return learner_fn(x, y, x_test, y_test, batch_size=BATCH_SIZE)

    print(f"Using target: {target}")

    synthesizer = Synthesizer(repo.specification(), {})
    t0 = time.time()
    search_space = synthesizer.construct_solution_space(target).prune()
    print(f"Search space construction took {time.time() - t0:.2f}s")

    kernel = noisy_hierarchical_damg_kernel

    def termination(state) -> bool:
        return state.generation >= evo_generations

    initialization = RandomLimitedDepthFirstInitialization(search_space, target, max_depth=MAX_DEPTH)
    mutation = ResolutionMutation(search_space, target, max_depth=MAX_DEPTH)
    recombination = Crossover(search_space, target, max_depth=MAX_DEPTH)
    parent_selection = FitnessProportionalSelection()
    survivor_selection = AgeBasedReplacement()
    fitness_comparator = ScalarFitnessComparator(True)

    evo_alg = SimpleGeneticProgramming(  # type: ignore[arg-type]
        search_space, target, termination, initialization, mutation, recombination,
        parent_selection, survivor_selection, fitness_comparator,
    )

    optimizer = BayesianOptimization(search_space, target, kernel=kernel, optimizer=evo_alg,
                                     optimizer_population_size=population_size,
                                     acquisition_function="ExpectedImprovement",
                                     optimizer_mutation_rate=0.00, optimizer_recombination_rate=0.99,
                                     max_depth=MAX_DEPTH)

    print(f"Starting Bayesian Optimization: n_pre_samples={n_pre_samples}, n_iterations={n_iterations}, "
          f"population_size={population_size}, evo_generations={evo_generations}")
    print(f"Logging every evaluated structure to {csv_path}")

    t0 = time.time()
    with ExperimentCSVLogger(csv_path, pretty_term_algebra) as logger:
        # Ask/Tell interface (not the bayesian_optimisation() convenience wrapper): lets us log
        # every evaluated structure - pre-samples and BO steps alike - as we go, so results survive
        # a crash/interruption instead of only being available after a successful finalize().
        optimizer.initialize(obj_fun=f_obj, n_pre_samples=n_pre_samples, greater_is_better=False)
        snapshot = optimizer.get_state_snapshot()
        for idx, (tree, value) in enumerate(zip(snapshot["x_list"], snapshot["y_list"])):
            logger.log("pre_sample", idx, tree, value)
            print(f"  pre_sample[{idx}]: objective={value:.5f}")

        for _ in range(n_iterations):
            suggestion = optimizer.suggest(ei_xi=0.01, verbose=verbose)
            # NOTE: must not be named `y` - f_obj closes over the outer `y` (USPS label tensor),
            # and reassigning that name here would clobber it for every subsequent f_obj() call.
            objective_value = f_obj(suggestion.candidate)
            optimizer.observe(suggestion.candidate, objective_value)
            iteration = suggestion.diagnostics["iteration"] if suggestion.diagnostics else None
            logger.log("bo_step", iteration, suggestion.candidate, objective_value)
            print(f"  bo_step[{iteration}]: objective={objective_value:.5f}")

        result = optimizer.finalize()
    bo_time = time.time() - t0

    print(f"Bayesian Optimization took {bo_time:.2f}s "
          f"({n_pre_samples + n_iterations} network evaluations)")
    print(f"Best test loss: {result['best_y']:.5f}")
    print(f"Best tree:\n{result['best_tree'].interpret(pretty_term_algebra())}")
    return result, bo_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pre-samples", type=int, default=8)
    parser.add_argument("--n-iterations", type=int, default=12)
    parser.add_argument("--population-size", type=int, default=20)
    parser.add_argument("--evo-generations", type=int, default=15)
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Defaults to results/usps_experiment_<unix timestamp>.csv")
    args = parser.parse_args()

    csv_path = args.csv_path
    if csv_path is None:
        os.makedirs("results", exist_ok=True)
        csv_path = f"results/usps_experiment_{int(time.time())}.csv"

    run_experiment(args.n_pre_samples, args.n_iterations, args.population_size, args.evo_generations,
                   csv_path, data_dir=args.data_dir)
