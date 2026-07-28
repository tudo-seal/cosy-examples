"""CIFAR-10 NAS search experiment: a genuine Bayesian-optimization search over a CNN search space.

The CIFAR-10 counterpart of `cnn_damg_usps_experiment.py` (same ask/tell + CSV-logging structure),
and the actual target experiment of the staged strategy - meant to run on the A30 GPU server.

Two things differ from the USPS script, both because CIFAR-10 training is far more expensive
(50k 3x32x32 images vs. 7.3k 1x16x16 ones):
  * `--epochs` is a CLI argument rather than a hard-coded constant, so a cheap smoke test can run a
    couple of epochs per candidate while the real run uses the full budget. The target is built to
    match (see `make_cifar_experiment_target`) - repository `n_epoch_values` and target epochs must
    always agree or nothing synthesizes.
  * `BATCH_SIZE` defaults to 128 (CIFAR-10 convention) instead of 64.

Usage:
    # cheap local smoke test (pipeline check + CPU timing reference)
    python -m bayesian_optimization.examples.cnn_damg_nas.cnn_damg_cifar_experiment \
        --epochs 2 --n-pre-samples 2 --n-iterations 1 --population-size 5 --evo-generations 3

    # real run (A30)
    python -m bayesian_optimization.examples.cnn_damg_nas.cnn_damg_cifar_experiment
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
    RankBasedSelection,
    RandomLimitedDepthFirstInitialization,
    ResolutionMutation,
    ScalarFitnessComparator,
    SimpleGeneticProgramming,
)

from bayesian_optimization import BayesianOptimization, IdentityTransform, Log1pTransform
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo_algebras import pretty_term_algebra
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_kernels import noisy_hierarchical_damg_kernel
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_targets import make_cifar_experiment_target
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_experiment_utils import (
    ExperimentCSVLogger,
    evaluate_candidate,
    write_run_metadata,
)

# Search space. Bootstrapped from the PyTorch-tutorial reference config in
# cnn_damg_reference_architectures.cifar10_tutorial_repo() (which is deliberately narrow - just wide
# enough for that one architecture) and widened into a genuine search space, the same way
# cnn_damg_usps_experiment.py widened usps_lecun1989_repo().
#
# Feature dimensions and channel/spatial choices have to be co-designed: a conv2d label is only
# generated when BOTH its flattened input (in_c*h*w) and output (out_c*h'*w') land in
# LINEAR_FEATURE_DIMENSIONS (see CNNrepository.Label.iter_conv2d). The values below are exactly the
# flattened sizes reachable from 3x32x32 through the listed channel/spatial combinations:
#   3*32*32=3072 (input)  6*28*28=4704  6*14*14=1176  16*10*10=1600  16*5*5=400
#   3*16*16=768           6*16*16=1536  16*8*8=1024   16*16*16=4096
# plus the small fully-connected sizes 120/84/10 the classifier head needs.
LINEAR_FEATURE_DIMENSIONS = [3072, 4704, 4096, 1600, 1536, 1176, 1024, 768, 400, 120, 84, 10]
CHANNEL_DIMENSIONS = [3, 6, 16]
HEIGHT_WIDTH_DIMENSIONS = [(32, 32), (28, 28), (16, 16), (14, 14), (10, 10), (8, 8), (5, 5)]
KERNEL_DIMENSIONS = [(5, 5), (3, 3), (2, 2)]
STRIDE_VALUES = [1, 2]
PADDING_VALUES = [0, 1]
MAX_PARALLEL_WIDTH = 2

CONSTANT_VALUES = [0, 1, -1]
LEARNING_RATE_VALUES = [1e-3]
BATCH_SIZE = 128
MAX_DEPTH = 1000

DEFAULT_EPOCHS = 50
DEFAULT_STRUCTURE_LENGTH = 5  # matches the agreed full-variance target (None,)*5

DATA_DIR = "./data"

# CIFAR-10 per-channel normalization statistics (standard values, same as used by the reference
# architecture tests in tests/integration/test_cnn_damg_nas.py).
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(data_dir: str = DATA_DIR):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def dataset_to_tensors(dataset, device):
    # Flat-feature-vector convention: (N, C*H*W), labels as class indices for cross_entropy_loss.
    # The whole split is materialized as one tensor (~614 MB for CIFAR-10 train in float32), which
    # keeps every candidate evaluation free of dataloader overhead - the same trade-off the USPS
    # script makes, and comfortably within the A30's 24 GB.
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
    images, labels = next(iter(loader))
    return images.reshape(images.shape[0], -1).to(device), labels.to(device).long()


def run_experiment(n_pre_samples: int, n_iterations: int, population_size: int, evo_generations: int,
                    csv_path: str, epochs: int = DEFAULT_EPOCHS,
                    structure_length: int = DEFAULT_STRUCTURE_LENGTH,
                    objective: str = "accuracy", max_lin_layer_dim: int | None = None,
                    data_dir: str = DATA_DIR, verbose: bool = True):
    # objective="accuracy" maximizes test accuracy; "loss" minimizes test cross-entropy (the former
    # default). They diverge sharply here (a 49% net had loss 17.9), so the direction is a real
    # choice - accuracy is what matters and is measured anyway in evaluate_candidate.
    if objective not in ("accuracy", "loss"):
        raise ValueError(f"objective must be 'accuracy' or 'loss', got {objective!r}")
    greater_is_better = objective == "accuracy"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_set, test_set = load_cifar10(data_dir)
    x, y = dataset_to_tensors(train_set, device)
    x_test, y_test = dataset_to_tensors(test_set, device)
    print(f"Loaded CIFAR-10: train={tuple(x.shape)}, test={tuple(x_test.shape)}")

    repo = CNNrepository(
        linear_feature_dimensions=LINEAR_FEATURE_DIMENSIONS,
        constant_values=CONSTANT_VALUES,
        learning_rate_values=LEARNING_RATE_VALUES,
        n_epoch_values=[epochs],  # must match the target's epochs literal
        channel_dimensions=CHANNEL_DIMENSIONS,
        height_width_dimensions=HEIGHT_WIDTH_DIMENSIONS,
        kernel_dimensions=KERNEL_DIMENSIONS,
        stride_values=STRIDE_VALUES,
        padding_values=PADDING_VALUES,
        max_parallel_width=MAX_PARALLEL_WIDTH,
        # When set below the 3072-feature raw image, no Linear label can consume the image, which
        # structurally forces a convolutional front end (see cnn_damg_repo; None = unconstrained).
        max_lin_layer_dim=max_lin_layer_dim,
    )
    target = make_cifar_experiment_target(epochs=epochs, length=structure_length)

    # Every candidate's full metric set, keyed by its term. The BO's initialize() evaluates the
    # pre-samples internally and only hands back objective values, so metrics measured during those
    # evaluations have to be stashed here to be logged afterwards.
    metrics_by_tree = {}

    def f_obj(t):
        metrics = evaluate_candidate(t, x, y, x_test, y_test, BATCH_SIZE)
        metrics_by_tree[t] = metrics
        # Both metrics are always recorded to the CSV; only which one the BO optimizes changes.
        return metrics["accuracy"] if objective == "accuracy" else metrics["objective_value"]

    print(f"Using target: {target}")

    synthesizer = Synthesizer(repo.specification(), {})
    t0 = time.time()
    search_space = synthesizer.construct_solution_space(target).prune()
    construction_time = time.time() - t0
    print(f"Search space construction took {construction_time:.2f}s")

    metadata = {
        "dataset": "cifar10",
        "device": str(device),
        "target": str(target),
        "structure_length": structure_length,
        "epochs_per_candidate": epochs,
        "batch_size": BATCH_SIZE,
        "max_depth": MAX_DEPTH,
        "train_samples": int(x.shape[0]),
        "test_samples": int(x_test.shape[0]),
        "search_space": {
            "linear_feature_dimensions": LINEAR_FEATURE_DIMENSIONS,
            "channel_dimensions": CHANNEL_DIMENSIONS,
            "height_width_dimensions": HEIGHT_WIDTH_DIMENSIONS,
            "kernel_dimensions": KERNEL_DIMENSIONS,
            "stride_values": STRIDE_VALUES,
            "padding_values": PADDING_VALUES,
            "max_parallel_width": MAX_PARALLEL_WIDTH,
            "max_lin_layer_dim": max_lin_layer_dim,
            "constant_values": CONSTANT_VALUES,
            "learning_rate_values": LEARNING_RATE_VALUES,
            "num_feature_dimensions": len(repo.feature_dimensions),
        },
        "bayesian_optimization": {
            "n_pre_samples": n_pre_samples,
            "n_iterations": n_iterations,
            "population_size": population_size,
            "evo_generations": evo_generations,
            "objective_metric": objective,          # "accuracy" (maximized) or "loss" (minimized)
            "greater_is_better": greater_is_better,
            "y_transform": "identity" if greater_is_better else "log1p",
            "acquisition_function": "ExpectedImprovement",
            "kernel_optimizer": "fmin_l_bfgs_b",
            "n_restarts_kernel_optimizer": 20,
            "ei_xi": 0.01,
            "parent_selection": "RankBasedSelection(selection_pressure=1.7)",
            "survivor_selection": "AgeBasedReplacement",
            # 0 is mandatory: mutation is currently buggy in CoSy
            "optimizer_mutation_rate": 0.0,
            "optimizer_recombination_rate": 0.99,
            "seed": None,  # deliberately unseeded - each run is an independent sample
        },
        "search_space_construction_seconds": construction_time,
    }
    # Written before training starts so an interrupted run still carries its provenance; rewritten
    # with the final timings once the run completes.
    print(f"Run configuration written to {write_run_metadata(csv_path, metadata)}")

    kernel = noisy_hierarchical_damg_kernel

    def termination(state) -> bool:
        return state.generation >= evo_generations

    initialization = RandomLimitedDepthFirstInitialization(search_space, target, max_depth=MAX_DEPTH)
    mutation = ResolutionMutation(search_space, target, max_depth=MAX_DEPTH)
    recombination = Crossover(search_space, target, max_depth=MAX_DEPTH)
    # RankBasedSelection, not FitnessProportionalSelection: the roulette weights are the raw EI
    # values, which span ~18 orders of magnitude here (the GP's fixed prior variance of 0.04
    # against normalize_y=True makes it over-confident, so EI underflows).  Measured on the real
    # USPS space: a single individual then receives 83.7% of the selection probability, and with
    # mutation_rate=0 there is no second source of diversity.  Rank-based selection depends only
    # on the ordering.  Measured over 40 generations: best EI 2.99e-18 vs 1.68e-23, still
    # improving at generation 28 vs stalling at generation 7.
    parent_selection = RankBasedSelection(1.7)
    survivor_selection = AgeBasedReplacement()
    fitness_comparator = ScalarFitnessComparator(True)

    evo_alg = SimpleGeneticProgramming(  # type: ignore[arg-type]
        search_space, target, termination, initialization, mutation, recombination,
        parent_selection, survivor_selection, fitness_comparator,
    )

    optimizer = BayesianOptimization(search_space, target, kernel=kernel, optimizer=evo_alg,
                                     optimizer_population_size=population_size,
                                     # Accuracy in [0,1] has no skew to correct, and Identity keeps
                                     # incumbent_raw directly readable as accuracy. Log1p (the
                                     # default) stays for the loss objective.
                                     y_transform=IdentityTransform() if greater_is_better else Log1pTransform(),
                                     acquisition_function="ExpectedImprovement",
                                     # Fit the kernel hyperparameters.  Without this the
                                     # amplitudes stay frozen at their initial values and the
                                     # GP is over-confident (see cnn_damg_kernels.py).
                                     # Measured cost at n_train=30: 1.0 s at 10 restarts vs
                                     # 0.6 s without -- negligible against a ~18 min BO step.
                                     kernel_optimizer="fmin_l_bfgs_b",
                                     n_restarts_kernel_optimizer=20,
                                     # mutation_rate MUST stay 0: the acquisition optimizer
                                     # explores by recombination only (deliberate choice,
                                     # carried over from the DNN experiments).
                                     optimizer_mutation_rate=0.00, optimizer_recombination_rate=0.99,
                                     max_depth=MAX_DEPTH)

    print(f"Starting Bayesian Optimization: n_pre_samples={n_pre_samples}, n_iterations={n_iterations}, "
          f"population_size={population_size}, evo_generations={evo_generations}, epochs={epochs}")
    print(f"Logging every evaluated structure to {csv_path}")

    t0 = time.time()
    with ExperimentCSVLogger(csv_path, pretty_term_algebra) as logger:
        # Ask/Tell interface (not the bayesian_optimisation() convenience wrapper): lets us log
        # every evaluated structure - pre-samples and BO steps alike - as we go, so results survive
        # a crash/interruption instead of only being available after a successful finalize().
        optimizer.initialize(obj_fun=f_obj, n_pre_samples=n_pre_samples,
                             greater_is_better=greater_is_better)
        snapshot = optimizer.get_state_snapshot()
        for idx, (tree, value) in enumerate(zip(snapshot["x_list"], snapshot["y_list"])):
            metrics = metrics_by_tree.get(tree, {"objective_value": value})
            logger.log("pre_sample", idx, tree, metrics)
            print(f"  pre_sample[{idx}]: objective={value:.5f} "
                  f"accuracy={metrics.get('accuracy', float('nan')):.4f} "
                  f"params={metrics.get('n_params', '?')} "
                  f"train={metrics.get('train_seconds', float('nan')):.1f}s", flush=True)

        for _ in range(n_iterations):
            suggestion = optimizer.suggest(ei_xi=0.01, verbose=verbose)
            # NOTE: must not be named `y` - f_obj closes over the outer `y` (CIFAR-10 label tensor),
            # and reassigning that name here would clobber it for every subsequent f_obj() call.
            objective_value = f_obj(suggestion.candidate)
            optimizer.observe(suggestion.candidate, objective_value)
            iteration = suggestion.diagnostics["iteration"] if suggestion.diagnostics else None
            metrics = metrics_by_tree[suggestion.candidate]
            logger.log("bo_step", iteration, suggestion.candidate, metrics, suggestion=suggestion)
            # `fallback` decides how this line reads: after a fallback the candidate is a random
            # sample, not the acquisition optimizer's choice. Seeing it live is the difference
            # between noticing a degraded run and reading it out of the CSV a day later.
            fallback = (suggestion.diagnostics or {}).get("fallback_used", "?")
            print(f"  bo_step[{iteration}]: objective={objective_value:.5f} "
                  f"accuracy={metrics['accuracy']:.4f} params={metrics['n_params']} "
                  f"train={metrics['train_seconds']:.1f}s fallback={fallback}", flush=True)

        result = optimizer.finalize()
    bo_time = time.time() - t0

    n_evaluations = n_pre_samples + n_iterations
    best_metrics = metrics_by_tree.get(result["best_tree"], {})
    print(f"Bayesian Optimization took {bo_time:.2f}s ({n_evaluations} network evaluations, "
          f"{bo_time / max(n_evaluations, 1):.2f}s per evaluation on average)")
    # result['best_y'] is the optimized objective on its raw scale: accuracy (maximized) or loss.
    print(f"Best {objective} ({'max' if greater_is_better else 'min'}): {result['best_y']:.5f}")
    if "accuracy" in best_metrics:
        print(f"Best tree's test accuracy: {best_metrics['accuracy'] * 100:.2f}%")
    if "objective_value" in best_metrics:
        print(f"Best tree's test loss: {best_metrics['objective_value']:.5f}")
    print(f"Best tree:\n{result['best_tree'].interpret(pretty_term_algebra())}")

    metadata.update({
        "bayesian_optimization_seconds": bo_time,
        "n_evaluations": n_evaluations,
        "mean_seconds_per_evaluation": bo_time / max(n_evaluations, 1),
        "total_training_seconds": sum(m["train_seconds"] for m in metrics_by_tree.values()),
        "best_objective_value": result["best_y"],
        "best_accuracy": best_metrics.get("accuracy"),
        "completed": True,
    })
    write_run_metadata(csv_path, metadata)
    return result, bo_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pre-samples", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=100)
    # 35: measured plateau of the acquisition EA under RankBasedSelection (last improvement
    # at generation 28 of 40).  Under the previous selection nothing improved past
    # generation 7, so the former default of 100 spent ~90% of its budget for nothing.
    parser.add_argument("--evo-generations", type=int, default=35)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Training epochs per candidate; the target is built to match.")
    parser.add_argument("--structure-length", type=int, default=DEFAULT_STRUCTURE_LENGTH,
                        help="Number of sequential components in the searched structure.")
    parser.add_argument("--objective", choices=["accuracy", "loss"], default="accuracy",
                        help="BO target: maximize test accuracy (default) or minimize test loss.")
    parser.add_argument("--max-lin-layer-dim", type=int, default=None,
                        help="Cap on Linear layer feature size. Below 3072 it forces a "
                             "convolutional front end (1600 is the tested value). None = off.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Defaults to results/cifar_experiment_<unix timestamp>.csv")
    args = parser.parse_args()

    csv_path = args.csv_path
    if csv_path is None:
        os.makedirs("results", exist_ok=True)
        csv_path = f"results/cifar_experiment_{int(time.time())}.csv"

    run_experiment(args.n_pre_samples, args.n_iterations, args.population_size, args.evo_generations,
                   csv_path, epochs=args.epochs, structure_length=args.structure_length,
                   objective=args.objective, max_lin_layer_dim=args.max_lin_layer_dim,
                   data_dir=args.data_dir)
