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
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_targets import (
    usps_target_len_3,
    fmnist_target_len_3,
    cifar_target_len_3,
)


# Staged dataset strategy: start on USPS (small, CPU-tractable smoke test), move to FashionMNIST for
# the first real minimal CNN experiment, and finally to CIFAR-10 - the target experiment, meant to run
# on a server with an A30 GPU. Switching stages is exactly this one constant plus the matching
# CNNrepository config below.
DATASET = "usps"  # "usps" | "fmnist" | "mnist" | "cifar10"

_DATASET_CONFIG = {
    "usps": dict(
        loader=torchvision.datasets.USPS,
        channels=1, height=16, width=16,
        mean=(0.5,), std=(0.5,),
        target=usps_target_len_3,
    ),
    "fmnist": dict(
        loader=torchvision.datasets.FashionMNIST,
        channels=1, height=28, width=28,
        mean=(0.5,), std=(0.5,),
        target=fmnist_target_len_3,
    ),
    "mnist": dict(
        loader=torchvision.datasets.MNIST,
        channels=1, height=28, width=28,
        mean=(0.1307,), std=(0.3081,),
        target=fmnist_target_len_3,
    ),
    "cifar10": dict(
        loader=torchvision.datasets.CIFAR10,
        channels=3, height=32, width=32,
        mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616),
        target=cifar_target_len_3,
    ),
}

# A30 peripherals: keep batch size / worker count as config constants rather than hardcoded values,
# so a server run only needs to bump these, not touch the training loop itself.
BATCH_SIZE = 128
NUM_WORKERS = 2
DATA_DIR = "./data"

# Optional AMP placeholder for later performance tuning on the A30. Left off by default since it
# needs a CUDA device to have any effect.
USE_AMP = False


def load_dataset(name: str, data_dir: str = DATA_DIR):
    cfg = _DATASET_CONFIG[name]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    train_set = cfg["loader"](root=data_dir, train=True, download=True, transform=transform)
    test_set = cfg["loader"](root=data_dir, train=False, download=True, transform=transform)
    return train_set, test_set, cfg


def dataset_to_tensors(dataset, device, num_workers=0):
    # Flat-feature-vector convention: images are flattened to (N, C*H*W), matching the R^n -> R^m
    # convention the repository/algebras use everywhere else. Labels stay class indices (N,), long,
    # for cross_entropy_loss.
    # One-shot load of the whole split as a single batch - NUM_WORKERS (used for real mini-batch
    # training, see BATCH_SIZE below) would only add multiprocessing overhead here without benefit.
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False,
                                         num_workers=num_workers)
    images, labels = next(iter(loader))
    x = images.reshape(images.shape[0], -1).to(device)
    y = labels.to(device).long()
    return x, y


if __name__ == "__main__":
    # Device-agnostic: picks up a CUDA device automatically once run on the A30 target, falls back
    # to CPU for local smoke tests. The interpreted model then follows the data via model.to(x.device)
    # inside cnn_damg_repo_algebras.learner - no other code path needs to know about the device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set, test_set, cfg = load_dataset(DATASET)
    x, y = dataset_to_tensors(train_set, device)
    x_test, y_test = dataset_to_tensors(test_set, device)
    print(f"Loaded {DATASET}: train={tuple(x.shape)}, test={tuple(x_test.shape)}")

    linear_feature_dimensions = [cfg["channels"] * cfg["height"] * cfg["width"], 64, 10]
    constant_values = [0, 1, -1]
    learning_rate_values = [1e-3]
    max_depth = 10000

    repo = CNNrepository(
        linear_feature_dimensions=linear_feature_dimensions,
        constant_values=constant_values,
        learning_rate_values=learning_rate_values,
        n_epoch_values=[20],
        channel_dimensions=[cfg["channels"], 4],
        height_width_dimensions=[(cfg["height"], cfg["width"])],
        kernel_dimensions=[(3, 3)],
        stride_values=[1],
        padding_values=[0],
        max_parallel_width=2,
    )

    target = cfg["target"]

    def f_obj(t):
        learner_fn = t.interpret(pytorch_function_algebra())
        return learner_fn(x, y, x_test, y_test, batch_size=BATCH_SIZE)

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
