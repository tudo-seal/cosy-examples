"""Train the VGG-11-BN reference architecture once, as the anchor for every later comparison.

Before any wildcard is opened, the search space is synthesized against exactly the chosen
reference learner, that one term is trained, and what it reaches is written down. Every number
measured against a target derived from this structure is then read against this run rather than
against a published figure produced by a different codebase on a different split.

What is pinned here, and why none of it is a search:

* The structure is fully concrete, 30 positions and no wildcard, so synthesis has nothing to
  explore and the term it returns is the reference architecture itself. The parameter count is
  checked against torchvision's ``vgg11_bn``, so a silent drift in the label set stops the run
  instead of producing a plausible wrong number.
* The optimizer is pinned to SGD at 0.1 with momentum 0.9 and weight decay 5e-4, and the schedule
  to CosineAnnealingLR. Both are stated in the target and not configured in the repository, so the
  query says which learner was asked for.
* The reported accuracy is the validation accuracy on the split carved out of the training data.
  The test accuracy is recorded beside it and used for nothing else.

No failure gets a substitute value. A diverged run is written out as diverged, with the epoch it
stopped at, and the number it reached is reported as measured.

This module needs the CIFAR-10 archive in ``--data-dir``. Nothing else here reads the network.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
from cosy.core import Synthesizer
from cosy.core.types import Constructor, Literal

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_cifar_experiment import (
    BATCH_SIZE,
    DATA_DIR,
    load_cifar10,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_experiment_utils import (
    dataset_to_tensors,
    evaluate_candidate,
    split_train_validation,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_network_algebras import (
    P50_CORRECTED_CIFAR,
    pytorch_model_algebra,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_reference_architectures import (
    cifar10_vgg11_bn_repo,
    cifar10_vgg11_bn_structure,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository

#: torchvision's vgg11_bn.features, minus the biases BatchNorm cancels, plus the CIFAR classifier.
#: Written out rather than computed, so that this module does not import torchvision.models to run
#: one comparison. The identity is held against torchvision itself by the test of this file.
EXPECTED_PARAMETERS = 9_225_984 - 2_752 + 5_130


def build_target(epochs: int, learning_rate: float, momentum: float, weight_decay: float):
    """The reference learner as a query: this structure, this optimizer, this schedule.

    Args:
        epochs (int): Epochs to train, which also becomes the schedule's T_max.
        learning_rate (float): The SGD rate.
        momentum (float): The SGD momentum.
        weight_decay (float): The SGD weight decay.

    Returns:
        The target type.
    """
    optimizer = CNNrepository.SGD(learning_rate=learning_rate, momentum=momentum,
                                  weight_decay=weight_decay)
    return Constructor("Learner", Constructor("DAG",
                                              Constructor("input", Literal(3072))
                                              & Constructor("output", Literal(10))
                                              & Constructor("structure",
                                                            Literal(cifar10_vgg11_bn_structure())))
                       & Constructor("Loss", Constructor("type", Literal(None)))
                       & Constructor("Optimizer", Constructor("type", Literal(optimizer)))
                       & Constructor("Scheduler", Constructor(
                           "type", Literal(CNNrepository.CosineAnnealingLR())))
                       & Constructor("epochs", Literal(epochs)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seeds", type=int, default=1,
                        help="How many independent trainings of the same term.  Their spread is "
                             "the only thing that says whether a later difference is a difference.")
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--json-path", default="results/vgg11_bn_reference.json")
    args = parser.parse_args()

    # The output directory is made here rather than at the first write, so that a wrong path fails
    # now and not after the training that was supposed to fill the file.
    directory = os.path.dirname(args.json_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    repo = cifar10_vgg11_bn_repo(epochs=args.epochs)
    target = build_target(args.epochs, args.learning_rate, args.momentum, args.weight_decay)

    started = time.time()
    space = Synthesizer(repo.specification(), {}).construct_solution_space(target).prune()
    tree = next(iter(space.enumerate_trees(target)))
    print(f"Synthesis: {time.time() - started:.1f}s", flush=True)

    n_params = sum(p.numel() for p in tree.interpret(pytorch_model_algebra()).parameters())
    if n_params != EXPECTED_PARAMETERS:
        raise SystemExit(
            f"the synthesized term has {n_params:,} parameters, not the {EXPECTED_PARAMETERS:,} of "
            f"vgg11_bn. The label set or the structure has drifted, and training this term would "
            f"measure a different architecture than the one this run claims to anchor")
    print(f"Term: {n_params:,} parameters, matches torchvision vgg11_bn", flush=True)

    train_set, test_set = load_cifar10(args.data_dir)
    x_full, y_full = dataset_to_tensors(train_set, device)
    x_test, y_test = dataset_to_tensors(test_set, device)
    x, y, x_val, y_val = split_train_validation(x_full, y_full, val_fraction=args.val_fraction)
    print(f"CIFAR-10: train={tuple(x.shape)}, val={tuple(x_val.shape)}, test={tuple(x_test.shape)}",
          flush=True)
    print(f"Protocol: {P50_CORRECTED_CIFAR}", flush=True)

    runs = []
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        started = time.time()
        result = evaluate_candidate(tree, x, y, x_val, y_val, args.batch_size,
                                    protocol=P50_CORRECTED_CIFAR, x_test=x_test, y_test=y_test)
        result["seed"] = seed
        result["wall_seconds"] = time.time() - started
        runs.append(result)
        print(f"seed {seed}: val={result['accuracy']:.4f} test={result['test_accuracy']:.4f} "
              f"diverged={result['diverged']} epochs={result['epochs_completed']}/{args.epochs} "
              f"({result['wall_seconds']:.0f}s)", flush=True)

        with open(args.json_path, "w") as handle:
            json.dump({"expected_parameters": EXPECTED_PARAMETERS, "n_params": n_params,
                       "epochs": args.epochs, "learning_rate": args.learning_rate,
                       "momentum": args.momentum, "weight_decay": args.weight_decay,
                       "batch_size": args.batch_size, "val_fraction": args.val_fraction,
                       "runs": runs}, handle, indent=2)

    validations = [r["accuracy"] for r in runs]
    print(f"\nvalidation accuracy: best={max(validations):.4f} "
          f"mean={sum(validations) / len(validations):.4f} over {len(runs)} run(s)")
    print(f"written to {args.json_path}")


if __name__ == "__main__":
    main()
