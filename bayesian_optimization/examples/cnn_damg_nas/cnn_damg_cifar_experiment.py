"""A Bayesian-optimization search for CIFAR-10 architectures over a synthesized CNN space.

This module is the driver.  It fixes the search space, picks the target, hands both to
``BayesianOptimization`` and writes the record of the run.  Everything it does not fix itself comes
from ``cnn_damg_experiment_utils``: the search program, the ask and tell loop, the evaluation of a
candidate and the files a run leaves behind.

Two things are settings of this driver rather than of the space it searches.

  * ``--epochs`` is an argument and not a constant, so a smoke test can spend two epochs per
    candidate where a real run spends fifty.  The repository and the target are built from the same
    value.  They have to agree: a target whose epoch count is not in the repository's
    ``n_epoch_values`` is uninhabited, and synthesis then returns nothing rather than an error.
  * ``BATCH_SIZE`` is 128, the usual CIFAR-10 batch, and it is not part of the search space.

A run needs the CIFAR-10 archive in ``--data-dir``.  Nothing else here reads from the network.

Usage:
    # smoke test: does the pipeline run at all, and what does one candidate cost on this machine
    python -m bayesian_optimization.examples.cnn_damg_nas.cnn_damg_cifar_experiment \
        --epochs 2 --n-pre-samples 2 --n-iterations 1 --population-size 5 --evo-generations 3

    # the full search
    python -m bayesian_optimization.examples.cnn_damg_nas.cnn_damg_cifar_experiment
"""

import argparse
import os
import time
import typing

import torch
import torchvision
from torchvision import transforms

from bayesian_optimization import BayesianOptimization
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_experiment_utils import (
    ACQUISITION_HARD_LIMIT_SECONDS,
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_DEPTH_BOUND,
    DEFAULT_MUTATION_RATE,
    DEFAULT_SIZE_BOUND,
    build_acquisition_optimizer,
    build_search,
    dataset_to_tensors,
    describe_sampler,
    describe_search,
    load_initial_design,
    make_objective,
    objective_direction,
    resolve_target_choice,
    run_ask_tell_search,
    split_train_validation,
    write_run_diagnostics,
    write_run_metadata,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_kernels import (
    NAMED_KERNELS,
    named_kernel,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_network_algebras import (
    P50,
    P50_CORRECTED_CIFAR,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_targets import (
    VGG11_BN_NAS_POSITIONS,
    VGG11_BN_VARIANCE_POSITIONS,
    make_cifar_experiment_target,
    make_cifar_head_target,
    make_cifar_tutorial_variance_target,
    make_vgg11_bn_target,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_algebras import pretty_term_algebra
from bayesian_optimization.examples.cnn_damg_nas.recognizable_cnn_damg_repo import (
    RecognizableCNNrepository,
)

# The search space. It starts from cnn_damg_reference_architectures.cifar10_tutorial_repo(), which
# is cut just wide enough for the one tutorial architecture, and widens every dimension of it into
# something worth searching.
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

# Pooling gets its own kernel sizes rather than sharing the convolution sizes, so that widening the
# convolution set does not widen the pooling set with it. On this configuration both spellings give
# the same five pooling labels, four of them at (2, 2) and one at (3, 3), against 45 convolution
# labels. (2, 2) is the size LeNet, AlexNet and VGG all use, and it is the one that halves the
# spatial extent.
# Which sizes contribute is geometry and not choice: a pooling label exists only when its output
# size is one of HEIGHT_WIDTH_DIMENSIONS and the flattened sizes on both ends are in
# LINEAR_FEATURE_DIMENSIONS. (4, 4) already satisfies that nowhere here, so a longer list buys
# nothing.
POOLING_KERNEL_DIMENSIONS = [(2, 2), (3, 3), (4, 4)]


CONSTANT_VALUES = [0, 1, -1]

# ---------------------------------------------------------------------------
# The VGG-11-BN geometry, for the two cells built on that reference architecture. It shares nothing
# with the tutorial geometry above: channels 64 to 512 instead of 3, 6 and 16, and features up to
# 65536. Mixing the two fails silently. A target whose dimensions the repository does not offer is
# uninhabited, which is an empty space and not a diagnostic.
# ---------------------------------------------------------------------------
VGG_LINEAR_FEATURE_DIMENSIONS = [3072, 65536, 32768, 16384, 8192, 4096, 2048, 512, 10]
VGG_CHANNEL_DIMENSIONS = [3, 64, 128, 256, 512]
VGG_HEIGHT_WIDTH_DIMENSIONS = [(32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)]
VGG_STRIDE_VALUES = [1]
VGG_POOLING_KERNEL_DIMENSIONS = [(2, 2)]
#: Without this cap the feature list contains 65536 and a single Linear over it would carry 4.3e9
#: weights, dwarfing every convolution in the space.
VGG_MAX_LIN_LAYER_DIM = 512
VGG_LEARNING_RATE_VALUES = [0.1]      # the SGD rate of the VGG/DARTS/NB201 recipe
VGG_WEIGHT_DECAY_VALUES = [5e-4]
VGG_MOMENTUM_VALUES = [0.9]

class VggCell(typing.TypedDict):
    """One VGG cell: which positions the target frees, and the geometry that variance needs.

    A cell is a pair and not two arguments, because a position list and a kernel set that were not
    built for each other produce an uninhabited target rather than an error.
    """

    positions: tuple
    kernels: list[tuple[int, int]]
    paddings: list[int]


#: The acquisition functions the loop can maximize. The command line and the type the loop expects
#: read the same list, so a new one is added in one place.
AcquisitionName = typing.Literal["ExpectedImprovement", "ProbabilityOfImprovement",
                                 "UpperConfidenceBound"]

#: The two VGG cells, each with the kernel set its variance is built for.
#:
#: VGGM is the merged chain, 15 positions of which 5 are free, at VGG's own single kernel size.
#: VGG-11 itself is not an element of it: a descriptor stands for one sequential stage, and the
#: merged position 12 admits one operator where VGG-11 has six.
#:
#: VGGN keeps all 30 positions and frees everything except the reduction skeleton and the
#: classifier, over the wider kernel set {1x1, 3x3, 5x5}. VGG-11 is an element of it, and that is
#: checked by synthesizing the concrete structure against this repository.
#:
#: VGGN is defined and it is not run. Its terms carry intermediate features up to 65536, and a
#: batch of 128 of them exhausted the memory of every GPU it was tried on, on the first term of the
#: initial design. Lowering the batch size did not help and capping the parallel width would have
#: taken the combinatorics away rather than bounded them. The definition stays because it is what
#: the wider search would ask for.
VGG_CELLS: dict[str, VggCell] = {
    "VGGM": {"positions": VGG11_BN_VARIANCE_POSITIONS,
             "kernels": [(3, 3)], "paddings": [1]},
    "VGGN": {"positions": VGG11_BN_NAS_POSITIONS,
             "kernels": [(1, 1), (3, 3), (5, 5)], "paddings": [0, 1, 2]},
}
LEARNING_RATE_VALUES = [1e-3]
BATCH_SIZE = 128
# The depth bound of the samplers that draw at random. A draw runs a depth-first search whose
# clause order is redrawn uniformly at each node, and it discards any partial term deeper than this.
MAX_DEPTH = DEFAULT_DEPTH_BOUND

DEFAULT_EPOCHS = 50
DEFAULT_STRUCTURE_LENGTH = 5  # matches the agreed full-variance target (None,)*5

# The hierarchical kernel: one Weisfeiler-Lehman kernel per level of the term, summed, with one
# fitted weight per level. A sum rather than the single level that measures best on any one cell,
# because which level carries the prediction is a property of the objective and is not known before
# the run. The fitted weights are what makes the choice, and a level that says nothing gets a weight
# near zero rather than a place in the model.
DEFAULT_KERNEL = "damg"

DATA_DIR = "./data"

# CIFAR-10 per-channel normalization statistics, the values the dataset is usually normalized with.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def load_cifar10(data_dir: str = DATA_DIR, download: bool = True):
    """Load the CIFAR-10 train and test splits.

    Args:
        data_dir (str): Where torchvision keeps the archive. (Default value = DATA_DIR)
        download (bool): Whether a missing archive may be fetched. (Default value = True)

    Returns:
        tuple: The train and test datasets.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=download,
                                             transform=transform)
    test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=download,
                                            transform=transform)
    return train_set, test_set


def run_experiment(n_pre_samples: int, n_iterations: int, population_size: int,
                    evo_generations: int, csv_path: str, epochs: int = DEFAULT_EPOCHS,
                    structure_length: int = DEFAULT_STRUCTURE_LENGTH,
                    target_cell: str | None = None, seed: int = 0,
                    objective: str = "accuracy", kernel: str = DEFAULT_KERNEL,
                    max_lin_layer_dim: int | None = None,
                    sampling: str = "size-uniform",
                    size_bound: int = DEFAULT_SIZE_BOUND, baseline: bool = False,
                    crossover_rate: float = DEFAULT_CROSSOVER_RATE,
                    mutation_rate: float = DEFAULT_MUTATION_RATE,
                    data_dir: str = DATA_DIR, verbose: bool = True,
                    protocol: str = "corrected", val_fraction: float = 0.1,
                    batch_size: int = BATCH_SIZE, repeats: int = 1,
                    acquisition_hard_limit: float = ACQUISITION_HARD_LIMIT_SECONDS,
                    resume_from: str | None = None,
                    acquisition: AcquisitionName = "ExpectedImprovement",
                    ucb_beta: float = 2.0):
    """Run the CIFAR-10 search end to end and write both artifacts.

    Args:
        n_pre_samples (int): mu_0, the size of the initial dataset.
        n_iterations (int): The budget B of loop passes.
        population_size (int): mu of the acquisition-optimizing search.
        evo_generations (int): Its termination bound.
        csv_path (str): Where the per-evaluation rows go.
        epochs (int): Training epochs per candidate. The target is built to match.
            (Default value = DEFAULT_EPOCHS)
        structure_length (int): Number of sequential components in the searched structure, or None
            with ``target_cell="HEAD"``, whose length its position list fixes.
            (Default value = DEFAULT_STRUCTURE_LENGTH)
        target_cell (str): Which named target this run asks for, or None for a length outside the
            grid. ``resolve_target_choice`` says which names there are. (Default value = None)
        seed (int): Seeds both sources of draws, the loop's own sampler and the
            acquisition-optimizing search. Different seeds are independent repetitions, the same
            seed measures the training noise. (Default value = 0)
        objective (str): "accuracy" (maximized) or "loss" (minimized). The two do not rank
            candidates alike, so the direction is a choice and not a formality.
            (Default value = "accuracy")
        max_lin_layer_dim (int | None): Cap on Linear layer feature size. Below 3072 it forces
            a convolutional front end. (Default value = None)
        sampling (str): How the loop draws its own terms. "size-uniform" determinizes the program
            and counts from it, "depth-bounded" searches the program as synthesized and never
            counts. ``build_search`` says which space admits which.
            (Default value = "size-uniform")
        size_bound (int): The bound D of the loop's size-uniform sampler, on the term size.
            (Default value = DEFAULT_SIZE_BOUND)
        baseline (bool): Also run a random search of the same budget from the same initial design,
            into the same artifacts.  Costs n_iterations extra trainings, not n_pre_samples +
            n_iterations: the initial design is evaluated once and shared. (Default value = False)
        data_dir (str): Where torchvision keeps the CIFAR-10 archive. (Default value = DATA_DIR)
        verbose (bool): Passed through to the loop. (Default value = True)
        repeats (int): How many trainings every candidate's objective value averages over. 1 is a
            single training. Above 1 the trainings are seeded 0 to repeats - 1, so the average is
            reproducible and two runs are comparable candidate by candidate, which an unseeded
            average would not be. Costs ``repeats`` times the training time and nothing extra in
            acquisition. (Default value = 1)
        acquisition_hard_limit (float): Seconds after which one acquisition maximization is given
            up on. It bounds a legitimate duration, so a cell whose acquisition is genuinely
            expensive needs it raised rather than left at the default.
            (Default value = ACQUISITION_HARD_LIMIT_SECONDS)

    Returns:
        tuple: The loop's result and its wall-clock duration.
    """
    greater_is_better, as_reported = objective_direction(objective)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    train_set, test_set = load_cifar10(data_dir)
    x_full, y_full = dataset_to_tensors(train_set, device)
    x_test, y_test = dataset_to_tensors(test_set, device)

    # The validation split is carved out of the training data. The test set is touched for the
    # record and by nothing that ranks a candidate, so what the search selects on and what it
    # reports are two different splits.
    x, y, x_val, y_val = split_train_validation(x_full, y_full, val_fraction=val_fraction)
    training_protocol = P50_CORRECTED_CIFAR if protocol == "corrected" else P50
    print(f"Loaded CIFAR-10: train={tuple(x.shape)}, val={tuple(x_val.shape)}, "
          f"test={tuple(x_test.shape)}")
    print(f"Training protocol: {protocol}, {training_protocol}", flush=True)

    # RecognizableCNNrepository, not CNNrepository: the same program with the same four swap laws,
    # stated so that they compile into the non-terminals instead of being decided on the terms.
    # That is what makes this space countable, and counting is what a size-uniform draw needs.
    # The VGG cells bring their own geometry, channels 64 to 512 and features up to 65536, none of
    # which the tutorial configuration above offers, so the repository follows the cell rather than
    # being fixed. Getting the pair wrong fails silently: a target whose dimensions the repository
    # does not offer is uninhabited, an empty space and no error.
    if target_cell in VGG_CELLS:
        repo = RecognizableCNNrepository(
            linear_feature_dimensions=VGG_LINEAR_FEATURE_DIMENSIONS,
            constant_values=CONSTANT_VALUES,
            learning_rate_values=VGG_LEARNING_RATE_VALUES,
            n_epoch_values=[epochs],
            channel_dimensions=VGG_CHANNEL_DIMENSIONS,
            height_width_dimensions=VGG_HEIGHT_WIDTH_DIMENSIONS,
            kernel_dimensions=VGG_CELLS[target_cell]["kernels"],
            stride_values=VGG_STRIDE_VALUES,
            padding_values=VGG_CELLS[target_cell]["paddings"],
            max_parallel_width=MAX_PARALLEL_WIDTH,
            pooling_kernel_dimensions=VGG_POOLING_KERNEL_DIMENSIONS,
            max_lin_layer_dim=VGG_MAX_LIN_LAYER_DIM,
            weight_decay_values=VGG_WEIGHT_DECAY_VALUES,
            momentum_values=VGG_MOMENTUM_VALUES,
        )
    else:
        repo = RecognizableCNNrepository(
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
            pooling_kernel_dimensions=POOLING_KERNEL_DIMENSIONS,
            # Set below the 3072 features of the raw image, no Linear label can consume the
            # image, which forces a convolutional front end. None leaves it unconstrained.
            max_lin_layer_dim=max_lin_layer_dim,
        )
    if target_cell in VGG_CELLS:
        target = make_vgg11_bn_target(VGG_CELLS[target_cell]["positions"], epochs=epochs)
    elif target_cell == "HEAD":
        target = make_cifar_head_target(epochs=epochs)
    elif target_cell in ("TUT1", "TUT2", "TUT3"):
        # The tutorial network's dimension chain with variance introduced where it is wanted, the
        # variant index saying how much structure stays pinned. These terms are larger than the
        # length targets', larger than the default --size-bound, so a run on them needs the bound
        # raised or the size-uniform sampler delivers nothing at all.
        target = make_cifar_tutorial_variance_target(epochs=epochs, variant=int(target_cell[-1]))
    else:
        target = make_cifar_experiment_target(epochs=epochs, length=structure_length)
    print(f"Using target ({target_cell or 'off-grid length'}): {target}", flush=True)

    # Fixed seeds rather than seeds derived from --seed: two runs that differ in their search seed
    # have to average over the same trainings, or a sweep over search seeds is a sweep over training
    # seeds as well and no difference between the runs can be attributed.
    training_seeds = tuple(range(repeats)) if repeats > 1 else None
    if repeats > 1:
        print(f"Averaging every objective value over {repeats} trainings, seeds {training_seeds}",
              flush=True)

    f_obj, metrics_by_tree = make_objective(
        x, y, x_val, y_val, batch_size, greater_is_better,
        protocol=training_protocol, x_test=x_test, y_test=y_test,
        repeats=repeats, training_seeds=training_seeds)

    # One call, one decision: the program and the sampler that fits it.
    program = build_search(repo, target, sampling=sampling, seed=seed,
                           size_bound=size_bound, depth_bound=MAX_DEPTH)

    evo_alg = build_acquisition_optimizer(
        population_size=population_size, generations=evo_generations, depth_bound=MAX_DEPTH,
        crossover_rate=crossover_rate, mutation_rate=mutation_rate, seed=seed,
    )

    optimizer = BayesianOptimization(
        program.space, program.request,
        kernel=named_kernel(kernel),
        optimizer=evo_alg,
        acquisition_function=acquisition,
        ucb_beta=ucb_beta,
        # The loop's own draws, the initial dataset and the fallback for a duplicate, come from the
        # sampler build_search chose together with the program. Passing the two separately is what
        # lets a program and a sampler that do not fit each other meet in the loop.
        sampler=program.sampler,
        # Fit the kernel hyperparameters.  Without this the amplitudes stay frozen at their
        # initial values and the GP is over-confident (see cnn_damg_kernels.py).
        kernel_optimizer="fmin_l_bfgs_b",
        n_restarts_kernel_optimizer=20,
    )

    metadata = {
        "dataset": "cifar10",
        "device": str(device),
        "target": str(target),
        # Which named cell this run is, beside the target's own text. The text says what was asked
        # for and the name says which of the prepared questions it was.
        "target_cell": target_cell,
        "structure_length": structure_length,
        "epochs_per_candidate": epochs,
        "batch_size": batch_size,
        # How many trainings one objective value averages over, and under which seeds. A record
        # with training_repeats above 1 reports means, and the spread they were taken over is in
        # the per-evaluation file beside it.
        "training_repeats": repeats,
        "training_seeds": list(training_seeds) if training_seeds is not None else None,
        "bo_sampler": describe_sampler(optimizer.sampler),
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
        # What the program cost to build and how large it came out, read off the objects.
        "search_program": program.provenance,
        "bayesian_optimization": {
            "n_pre_samples": n_pre_samples,
            "n_iterations": n_iterations,
            "evo_generations": evo_generations,
            "objective_metric": objective,          # "accuracy" (maximized) or "loss" (minimized)
            # The loop itself always maximizes, and a loss objective is negated on the way in.
            "greater_is_better": greater_is_better,
            "acquisition_function": acquisition,
            # Only read when the acquisition is UCB, but recorded either way: a file that omits it
            # cannot say whether the run took the default or was never asked.
            "ucb_beta": ucb_beta,
            # Which surrogate this run fitted.  It was hardwired until now, so a
            # record could not say which of two runs used which kernel.
            "kernel": kernel,
            "kernel_optimizer": "fmin_l_bfgs_b",
            "n_restarts_kernel_optimizer": 20,
            # Under which hard limit this run's acquisition ran. A run that gave up on a step
            # cannot be told from one that never had an expensive step without this field.
            "acquisition_hard_limit_seconds": acquisition_hard_limit,
            # Read off the object that will run, not written beside it.
            **describe_search(evo_alg),
            # Every draw the search makes is seeded, both the acquisition optimizer and the loop's
            # own dataset. Two runs at the same seed therefore see the same candidates and differ
            # only in what training makes of them, which is the sample this file records. At
            # --repeats 1 the training itself stays unseeded, its initialization and its batch
            # order are random, and above 1 it is seeded per repetition by `training_seeds`.
            "search_seed": seed,
        },
        # Also at the top level, under the name earlier records used, so that a reader can compare
        # this file with an older one field by field.
        "search_space_construction_seconds":
            program.provenance["search_space_construction_seconds"],
    }
    # Written before training starts, so that an interrupted run still carries its provenance,
    # and rewritten with the final timings once the run completes.
    print(f"Run configuration written to {write_run_metadata(csv_path, metadata)}")

    resume_design = None
    if resume_from is not None:
        if not baseline:
            # Without --baseline the loop draws its design inside initialize(), and there is no
            # point at which handed-in terms could be checked against the ones it drew.
            msg = "--resume-from needs --baseline: only the paired path draws its design up front"
            raise ValueError(msg)
        # The fields that change what a measured number means. Population size, budget and kernel
        # may differ, because they change what the run does with the design and not what the design
        # itself says.
        resume_design = load_initial_design(resume_from, expected={
            "target_cell": target_cell,
            "epochs_per_candidate": epochs,
            "batch_size": batch_size,
            "training_repeats": repeats,
        })
        print(f"Resuming the initial design of {resume_from}: {len(resume_design)} evaluated "
              f"terms taken over instead of retrained", flush=True)

    result, bo_time, summary = run_ask_tell_search(
        optimizer=optimizer,
        f_obj=f_obj,
        metrics_by_tree=metrics_by_tree,
        as_reported=as_reported,
        objective=objective,
        greater_is_better=greater_is_better,
        n_pre_samples=n_pre_samples,
        n_iterations=n_iterations,
        csv_path=csv_path,
        pretty_algebra=pretty_term_algebra,
        provenance=metadata,
        baseline=baseline,
        verbose=verbose,
        acquisition_hard_limit=acquisition_hard_limit,
        resume_design=resume_design,
    )

    metadata.update(summary)
    write_run_metadata(csv_path, metadata)
    # Last, and after the results are on disk: a read that raises then costs the diagnostics
    # rather than the run.
    write_run_diagnostics(csv_path, optimizer, result)
    return result, bo_time


def build_parser() -> argparse.ArgumentParser:
    """The command line of this experiment, as its own object so that it can be read and tested.

    Every argument here is a setting of the run and not of the search space, except ``--epochs``,
    ``--target``, ``--structure-length``, ``--size-bound`` and ``--max-lin-layer-dim``, which decide
    what is searched and are recorded in the run's configuration file for that reason.

    Returns:
        argparse.ArgumentParser: The parser ``main`` uses.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pre-samples", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=20)
    parser.add_argument("--population-size", type=int, default=100)
    parser.add_argument("--evo-generations", type=int, default=35)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="Training epochs per candidate. The target is built to match.")
    parser.add_argument("--structure-length", type=int, default=None,
                        help=f"Number of sequential components in the searched structure "
                             f"(default {DEFAULT_STRUCTURE_LENGTH}).  Say it through --target "
                             f"instead when the run is one of the named cells.")
    parser.add_argument("--target",
                        choices=["L3", "L5", "HEAD", "TUT1", "TUT2", "TUT3", "VGGM", "VGGN"],
                        default=None,
                        help="Which target to ask for.  L3 and L5 are the full-variance target at "
                             "that length, HEAD is the head target, and TUT1/2/3 build on the "
                             "PyTorch tutorial network's dimension chain with variance where it "
                             "is wanted, TUT1 pinning the most structure and TUT3 the least.  All "
                             "but L3/L5 fix their length through a position list and therefore "
                             "take no --structure-length.  TUT1/2/3 carry terms well past the "
                             "default --size-bound, so they need it raised or the size-uniform "
                             "sampler delivers nothing.  VGGM and VGGN are built on the VGG-11-BN "
                             "reference and bring their own geometry (channels 64 to 512, features "
                             "to 65536). VGGM is the merged 15-position chain with 5 free, VGGN "
                             "keeps all 30 positions, frees everything but the reduction skeleton, "
                             "and searches kernels {1x1, 3x3, 5x5}.  VGGN terms reach size 734, so "
                             "it needs --size-bound 800 or more.  The choice is recorded in the "
                             "run's configuration.")
    parser.add_argument("--objective", choices=["accuracy", "loss"], default="accuracy",
                        help="BO target: maximize test accuracy (default) or minimize test loss.")
    parser.add_argument("--max-lin-layer-dim", type=int, default=None,
                        help="Cap on Linear layer feature size. Below 3072 it forces a "
                             "convolutional front end (1600 is the tested value). None = off.")
    parser.add_argument("--crossover-rate", type=float, default=DEFAULT_CROSSOVER_RATE,
                        help="Crossover rate of the acquisition-optimizing search.  The "
                             "convergence conditions ask for a rate below 1.")
    parser.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION_RATE,
                        help="Mutation rate, PER OFFSPRING.  The operator replaces a whole subtree "
                             "rather than a single symbol, so this is not a per-gene rate.  The "
                             "convergence conditions ask for a rate above 0.")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run a random search of the same budget from the SAME initial "
                             "design, into the same artifacts (phase 'random_sample'). Costs "
                             "--n-iterations extra trainings, since the initial design is shared.")
    parser.add_argument("--sampling", choices=["size-uniform", "depth-bounded"],
                        default="size-uniform",
                        help="How the loop draws its own terms. 'size-uniform' determinizes the "
                             "search space and counts from the program, 'depth-bounded' searches "
                             "it as synthesized and never counts.")
    parser.add_argument("--size-bound", type=int, default=DEFAULT_SIZE_BOUND,
                        help="Bound D of the loop's size-uniform sampler, on the TERM SIZE. "
                             "Below the smallest term of the space it admits nothing.")
    parser.add_argument("--kernel", choices=sorted(NAMED_KERNELS), default=DEFAULT_KERNEL,
                        help="Which surrogate kernel to fit.  The run records it, so two "
                             "runs of this script are told apart by their record.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds both sources of draws, the loop's own sampler and the "
                             "acquisition-optimizing search.  Different seeds are independent "
                             "repetitions, the same seed measures the training noise.")
    parser.add_argument("--protocol", choices=["corrected", "p50"], default="corrected",
                        help="Which training recipe to use.  'corrected' is the default: He "
                             "initialization, gradient clipping at 5.0, random flip and crop-4. "
                             "'p50' is the older recipe (Adam at a constant 1e-3, no augmentation, "
                             "no initialization), for comparison runs against earlier studies.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Mini-batch size for training. A driver setting and not part of the "
                             "search space, but the one lever against a GPU running out of memory "
                             "on cells whose terms carry wide intermediate features.")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Share of the training data that becomes the validation set.  The "
                             "search is scored on it alone and the test split is only recorded.")
    parser.add_argument("--repeats", type=int, default=1,
                        help="How many trainings one objective value averages over.  1 is a single "
                             "training.  Training noise can be wider than the difference between "
                             "two candidates, in which case one training cannot decide between "
                             "them and the mean over repeats divides that noise by sqrt(repeats). "
                             "The single values and their spread are written to the CSV.  Costs "
                             "repeats times the training time and nothing extra in acquisition.")
    parser.add_argument("--acquisition", default="ExpectedImprovement",
                        choices=typing.get_args(AcquisitionName),
                        help="Which acquisition function the loop maximizes.  Which one is ahead "
                             "of which depends on the cell, so it is an argument and not a "
                             "constant.")
    parser.add_argument("--ucb-beta", type=float, default=2.0,
                        help="The exploration weight of the upper confidence bound, which scores a "
                             "term as its posterior mean plus beta times its posterior standard "
                             "deviation and has to be positive.  Read only with --acquisition "
                             "UpperConfidenceBound.")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Path to the <run>_terms.pickle of an interrupted run whose initial "
                             "design is taken over instead of being trained again.  Cell, epochs "
                             "and repeats have to agree, and the loaded terms have to be the ones "
                             "this run draws, and both are checked rather than assumed.  Only "
                             "with --baseline.")
    parser.add_argument("--acquisition-hard-limit", type=float,
                        default=ACQUISITION_HARD_LIMIT_SECONDS,
                        help="Seconds after which ONE acquisition maximization is given up on.  It "
                             "is a brake against an inner search that hangs, but it bounds a "
                             "legitimate duration too, so a cell whose acquisition is genuinely "
                             "expensive needs it raised rather than left at the default.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Defaults to results/cifar_experiment_<unix timestamp>.csv")
    return parser


def main(argv=None):
    """Run one search from the command line.

    Args:
        argv: The arguments to parse, or None to take them from the process.
            (Default value = None)

    Returns:
        tuple: The loop's result and its wall-clock duration, as ``run_experiment`` returns them.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # run_experiment checks this condition too, but only after the search space is built and
    # determinized, and on a large cell that is a long time to wait for an argument error. An
    # argument check belongs in front of the work. The one inside stays for programmatic callers.
    if args.resume_from is not None and not args.baseline:
        parser.error(
            "--resume-from needs --baseline: only the paired path draws its design up front, and "
            "only there can the terms handed in be checked against the terms drawn"
        )

    csv_path = args.csv_path
    if csv_path is None:
        os.makedirs("results", exist_ok=True)
        csv_path = f"results/cifar_experiment_{int(time.time())}.csv"

    target_cell, structure_length = resolve_target_choice(
        args.target, args.structure_length, DEFAULT_STRUCTURE_LENGTH
    )

    return run_experiment(
        args.n_pre_samples, args.n_iterations, args.population_size, args.evo_generations,
        csv_path, epochs=args.epochs, structure_length=structure_length,
        target_cell=target_cell, seed=args.seed,
        objective=args.objective, max_lin_layer_dim=args.max_lin_layer_dim,
        sampling=args.sampling, size_bound=args.size_bound,
        baseline=args.baseline, crossover_rate=args.crossover_rate,
        mutation_rate=args.mutation_rate, kernel=args.kernel,
        data_dir=args.data_dir, protocol=args.protocol,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size, repeats=args.repeats,
        acquisition_hard_limit=args.acquisition_hard_limit,
        resume_from=args.resume_from,
        acquisition=args.acquisition, ucb_beta=args.ucb_beta)


if __name__ == "__main__":
    main()
