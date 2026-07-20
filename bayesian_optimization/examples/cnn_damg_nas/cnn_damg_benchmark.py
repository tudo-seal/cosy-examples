"""Device-agnostic CPU/A30 timing benchmark for the reference CNN architectures.

Meant to run *unchanged* on a plain CPU machine and on the A30 GPU server: picks up CUDA
automatically via `torch.cuda.is_available()` (see cnn_damg_repo_algebras.learner, which already
does `model = model.to(x.device)`), so the only thing that differs between runs is the reported
timing - the code path is identical.

Usage:
    python -m bayesian_optimization.examples.cnn_damg_nas.cnn_damg_benchmark --dataset usps
    python -m bayesian_optimization.examples.cnn_damg_nas.cnn_damg_benchmark --dataset cifar10
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo_algebras import (
    pytorch_model_algebra,
    learner as raw_learner,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_reference_architectures import (
    usps_lecun1989_repo,
    usps_lecun1989_structure,
    cifar10_tutorial_repo,
    cifar10_tutorial_structure,
)


_DATASETS = {
    "usps": dict(
        repo=usps_lecun1989_repo,
        structure=usps_lecun1989_structure,
        n_in=256,
        n_out=10,
        epochs=300,
        batch_size=64,
        torchvision_dataset="USPS",
        mean=(0.5,),
        std=(0.5,),
    ),
    "cifar10": dict(
        repo=cifar10_tutorial_repo,
        structure=cifar10_tutorial_structure,
        n_in=3072,
        n_out=10,
        epochs=20,
        batch_size=128,
        torchvision_dataset="CIFAR10",
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
    ),
}


def _synthesize(cfg):
    from cosy.core import Synthesizer
    from cosy.core.types import Constructor, Literal

    repo = cfg["repo"]()
    target = Constructor("Learner", Constructor("DAG",
                                                 Constructor("input", Literal(cfg["n_in"]))
                                                 & Constructor("output", Literal(cfg["n_out"]))
                                                 & Constructor("structure", Literal(cfg["structure"]())))
                         & Constructor("Loss", Constructor("type", Literal(None)))
                         & Constructor("Optimizer", Constructor("type", Literal(None)))
                         & Constructor("epochs", Literal(cfg["epochs"])))
    spec = repo.specification()
    search_space = Synthesizer(spec, {}).construct_solution_space(target).prune()
    trees = list(search_space.enumerate_trees(target, max_count=1))
    if not trees:
        raise RuntimeError(f"reference architecture for {cfg} did not synthesize - check the repo config")
    return trees[0]


def _load_dataset(name, cfg, data_dir="./data"):
    import torchvision
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    dataset_cls = getattr(torchvision.datasets, cfg["torchvision_dataset"])
    train_set = dataset_cls(root=data_dir, train=True, download=True, transform=transform)
    test_set = dataset_cls(root=data_dir, train=False, download=True, transform=transform)

    def to_tensors(dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        images, labels = next(iter(loader))
        return images.reshape(images.shape[0], -1), labels.long()

    return to_tensors(train_set), to_tensors(test_set)


def run_benchmark(dataset_name: str, epochs: int = None, batch_size: int = None, data_dir: str = "./data"):
    cfg = _DATASETS[dataset_name]
    epochs = epochs or cfg["epochs"]
    batch_size = batch_size or cfg["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"dataset={dataset_name} device={device} epochs={epochs} batch_size={batch_size}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    tree = _synthesize(cfg)
    model = tree.interpret(pytorch_model_algebra())
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params}")

    (x, y), (x_test, y_test) = _load_dataset(dataset_name, cfg, data_dir)
    x, y = x.to(device), y.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)
    print(f"train={tuple(x.shape)} test={tuple(x_test.shape)}")

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    t0 = time.time()
    test_loss = raw_learner(cfg["n_in"], model, loss_fn, lambda m: optim.Adam(m.parameters(), lr=1e-3),
                             epochs, x, y, x_test, y_test, batch_size=batch_size)
    dt = time.time() - t0

    with torch.inference_mode():
        preds = model(x_test).argmax(dim=-1)
        accuracy = (preds == y_test).float().mean().item()

    print(f"training_time_s={dt:.2f}")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_accuracy={accuracy * 100:.2f}%")
    return {"device": str(device), "n_params": n_params, "training_time_s": dt,
            "test_loss": test_loss, "test_accuracy": accuracy}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(_DATASETS), required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default="./data")
    args = parser.parse_args()
    run_benchmark(args.dataset, epochs=args.epochs, batch_size=args.batch_size, data_dir=args.data_dir)
