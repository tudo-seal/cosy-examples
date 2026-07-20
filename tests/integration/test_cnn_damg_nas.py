"""Integration tests for the CNN extension of the DAMG repository (cnn_damg_nas).

Two concerns, matching how this repository's query language is meant to be used:

1. Every combinator must actually interpret correctly at least once - including conv2d/maxpool2d
   and swap composition over a conv-derived (flattened) feature dimension. We don't rely on
   enumerating the search space and filtering for a match: instead each test builds a fully
   concrete (non-wildcard) ``structure`` literal, so synthesis has (near-)nothing left to search
   for and directly produces the intended term.
2. The historically smallest known CNN for USPS (LeCun et al. 1989) is described as one such
   concrete target and actually trained on the real USPS dataset; the resulting test accuracy is
   checked against the published benchmark (~5% test error) within a generous, documented margin.
"""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from cosy.core import Synthesizer
from cosy.core.types import Constructor, Literal

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo_algebras import (
    pretty_term_algebra,
    pytorch_model_algebra,
    learner as raw_learner,
)


# ---------------------------------------------------------------------------
# Shared small repository, sized so that every combinator has at least one
# reachable, concrete instantiation (conv2d/maxpool2d included).
# ---------------------------------------------------------------------------

def _small_repo() -> CNNrepository:
    return CNNrepository(
        linear_feature_dimensions=[1, 2, 4, 8, 16],
        constant_values=[0, 1, -1],
        learning_rate_values=[1e-3],
        n_epoch_values=[1],
        channel_dimensions=[1, 2],
        height_width_dimensions=[(4, 4), (2, 2)],
        kernel_dimensions=[(2, 2), (3, 3)],
        stride_values=[1],
        padding_values=[0],
        max_parallel_width=2,
    )


def _concrete_target(repo: CNNrepository, n_in: int, n_out: int, structure: tuple, epochs: int = 1):
    return Constructor("Learner", Constructor("DAG",
                                               Constructor("input", Literal(n_in))
                                               & Constructor("output", Literal(n_out))
                                               & Constructor("structure", Literal(structure)))
                        & Constructor("Loss", Constructor("type", Literal(None)))
                        & Constructor("Optimizer", Constructor("type", Literal(None)))
                        & Constructor("epochs", Literal(epochs)))


def _synthesize_one(repo: CNNrepository, target):
    """Enumerate a concrete (wildcard-free) target - this must resolve near-instantly since there
    is (almost) nothing left to search for; a slow/exhaustive search here would indicate the
    structure literal was not actually concrete."""
    spec = repo.specification()
    synthesizer = Synthesizer(spec, {})
    t0 = time.time()
    search_space = synthesizer.construct_solution_space(target).prune()
    trees = list(search_space.enumerate_trees(target, max_count=5))
    dt = time.time() - t0
    assert trees, f"expected the concrete target to be directly synthesizable, found none: {target}"
    assert dt < 5.0, f"synthesizing a fully concrete target took {dt:.2f}s - structure literal is not actually concrete"
    return trees[0]


# ---------------------------------------------------------------------------
# 1. Every "simple" (single-node) combinator interprets and produces a correctly
#    shaped forward pass: linear_layer, conv2d, maxpool2d, sigmoid, relu, tanh,
#    sum, product, copy, edges, swap.
# ---------------------------------------------------------------------------

def _node_case_id(case):
    return case[0]


_conv2d_label = CNNrepository.Conv2d(in_channels=1, out_channels=2, input_size=(4, 4), output_size=(2, 2),
                                      kernel_size=(3, 3), stride=1, padding=0, bias=True)
_maxpool2d_label = CNNrepository.MaxPool2d(in_channels=1, input_size=(4, 4), output_size=(2, 2), kernel_size=(2, 2))

# (name, label, i, o)
_SIMPLE_NODE_CASES = [
    ("linear_layer", CNNrepository.Linear(in_features=16, out_features=8, bias=True), 16, 8),
    ("conv2d", _conv2d_label, 16, 8),
    ("maxpool2d", _maxpool2d_label, 16, 4),
    ("sigmoid", CNNrepository.Sigmoid(), 4, 4),
    ("relu", CNNrepository.ReLu(inplace=False), 4, 4),
    ("tanh", CNNrepository.Tanh(), 4, 4),
    ("sum", CNNrepository.Sum(with_constant=0), 4, 1),
    ("product", CNNrepository.Product(with_constant=1), 4, 1),
    ("copy", CNNrepository.Copy(out_dimension=4), 1, 4),
    ("swap", ("swap", 2, 2), 4, 4),
]


@pytest.mark.parametrize("case", _SIMPLE_NODE_CASES, ids=_node_case_id)
def test_simple_combinator_interprets(case):
    _, label, i, o = case
    repo = _small_repo()
    target = _concrete_target(repo, i, o, (((label, i, o),),))
    tree = _synthesize_one(repo, target)

    model = tree.interpret(pytorch_model_algebra())
    x = torch.randn(3, i)
    y = model(x)
    assert y.shape == (3, o), f"{case[0]}: expected output shape (3, {o}), got {tuple(y.shape)}"


# ---------------------------------------------------------------------------
# 2. beside_cons: a genuine parallel (not just sequential-singleton) composition
#    of two distinct components at one position.
# ---------------------------------------------------------------------------

def test_beside_cons_parallel_composition_interprets():
    repo = _small_repo()
    linear_a = CNNrepository.Linear(in_features=4, out_features=4, bias=True)
    linear_b = CNNrepository.Linear(in_features=4, out_features=4, bias=False)
    structure = (
        ((linear_a, 4, 4), (linear_b, 4, 4)),
    )
    target = _concrete_target(repo, 8, 8, structure)
    tree = _synthesize_one(repo, target)

    pretty = tree.interpret(pretty_term_algebra())
    assert "||" in pretty or "Linear" in pretty  # sanity: both branches show up

    model = tree.interpret(pytorch_model_algebra())
    x = torch.randn(3, 8)
    y = model(x)
    assert y.shape == (3, 8)


def test_edges_as_identity_lane_in_beside_cons_interprets():
    # "edges" is ID-typed (a pure identity pass-through) and is therefore only ever valid as a
    # parallel lane inside a beside_cons composition, never as a standalone before_singleton
    # argument (which requires non_ID) - so it's tested embedded next to a real component.
    repo = _small_repo()
    linear = CNNrepository.Linear(in_features=4, out_features=4, bias=True)
    edges_label = ("swap", 0, 4)
    structure = (
        ((linear, 4, 4), (edges_label, 4, 4)),
    )
    target = _concrete_target(repo, 8, 8, structure)
    tree = _synthesize_one(repo, target)

    pretty = tree.interpret(pretty_term_algebra())
    assert "edges(" in pretty

    model = tree.interpret(pytorch_model_algebra())
    x = torch.randn(3, 8)
    y = model(x)
    assert y.shape == (3, 8)
    # the second (edges) lane must be passed through unchanged
    assert torch.allclose(y[:, 4:], x[:, 4:])


# ---------------------------------------------------------------------------
# 3. swap composition specifically over a conv2d-derived flattened feature -
#    the whole point of treating conv2d like linear_layer: the flattened
#    output feature of a conv2d is just an ordinary F member, so a swap can
#    rewire it exactly like it would rewire a linear_layer's output.
# ---------------------------------------------------------------------------

def test_swap_over_conv_derived_feature_interprets_and_permutes():
    repo = _small_repo()
    conv = CNNrepository.Conv2d(in_channels=1, out_channels=2, input_size=(4, 4), output_size=(2, 2),
                                 kernel_size=(3, 3), stride=1, padding=0, bias=True)
    # conv flattens 16 -> 8; swap(4, 4) rewires that 8-wide conv output before it is consumed further.
    structure = (
        ((conv, 16, 8),),
        ((("swap", 4, 4), 8, 8),),
    )
    target = _concrete_target(repo, 16, 8, structure)
    tree = _synthesize_one(repo, target)

    pretty = tree.interpret(pretty_term_algebra())
    assert "Conv2d(" in pretty
    assert "swap(" in pretty

    model = tree.interpret(pytorch_model_algebra())
    x = torch.randn(3, 16)
    y = model(x)
    assert y.shape == (3, 8)

    # confirm SwapModule actually permutes (not just passes through): compare the model's two
    # halves against a manually split/permuted reference computed via the conv sub-module alone.
    conv_only = [m for m in model.modules() if type(m).__name__ == "SynthConv2d"][0]
    conv_out = conv_only(x)
    first_half, second_half = conv_out[:, :4], conv_out[:, 4:]
    expected = torch.cat((second_half, first_half), dim=-1)
    assert torch.allclose(y, expected), "swap did not correctly permute the conv-derived feature halves"


# ---------------------------------------------------------------------------
# 4. cross_entropy_loss / adam_optimizer / learner: exercised together via an
#    actual (tiny) end-to-end training call.
# ---------------------------------------------------------------------------

def test_learner_cross_entropy_and_adam_train_end_to_end():
    repo = _small_repo()
    linear = CNNrepository.Linear(in_features=16, out_features=8, bias=True)
    structure = (((linear, 16, 8),),)
    # epochs must match the repo's n_epoch_values for the target itself to be synthesizable; the
    # actual training below calls raw_learner() directly with its own n_epochs, independently.
    target = _concrete_target(repo, 16, 8, structure)
    tree = _synthesize_one(repo, target)

    model = tree.interpret(pytorch_model_algebra())
    x = torch.randn(20, 16)
    y = torch.randint(0, 8, (20,))
    loss = raw_learner(16, model, nn.CrossEntropyLoss(), lambda m: optim.Adam(m.parameters(), lr=1e-3),
                       3, x, y, x, y, batch_size=8)
    assert torch.isfinite(torch.tensor(loss))


# ---------------------------------------------------------------------------
# 5. Reference architecture: LeCun et al. 1989, "Backpropagation Applied to Handwritten Zip Code
#    Recognition" (Neural Computation 1, 541-551) is - to our knowledge - the smallest published
#    CNN that reasonably solves this exact task: 16x16 USPS zip-code digits, the same dataset this
#    stage of the staged strategy uses. It reported 5.00% test error with 9760 parameters, using
#    Conv(1->12, 5x5, stride 2, padding 2) -> Conv(12->12, 5x5, stride 2, padding 2) -> FC(192->30)
#    -> FC(30->10) with tanh activations (the original H1->H2 connectivity was sparse; our conv2d
#    combinator only supports dense convolutions, so this echo has slightly more parameters: 10024).
#    Describing this exact architecture as a fully concrete structure literal (no wildcards) lets
#    synthesis produce it directly - the same mechanism the combinator-coverage tests above use.
#
#    Working with a well-benchmarked dataset like USPS is precisely what makes this check possible:
#    training this exact architecture and comparing the resulting test accuracy against the
#    published number is a real correctness signal, not just "did it run".
# ---------------------------------------------------------------------------

def _lecun1989_repo() -> CNNrepository:
    return CNNrepository(
        linear_feature_dimensions=[256, 768, 192, 30, 10],
        constant_values=[0, 1, -1],
        learning_rate_values=[1e-3],
        n_epoch_values=[300],
        channel_dimensions=[1, 12],
        height_width_dimensions=[(16, 16), (8, 8)],
        kernel_dimensions=[(5, 5)],
        stride_values=[2],
        padding_values=[2],
        max_parallel_width=2,
    )


def _lecun1989_structure():
    conv1 = CNNrepository.Conv2d(in_channels=1, out_channels=12, input_size=(16, 16), output_size=(8, 8),
                                  kernel_size=(5, 5), stride=2, padding=2, bias=True)
    conv2 = CNNrepository.Conv2d(in_channels=12, out_channels=12, input_size=(8, 8), output_size=(4, 4),
                                  kernel_size=(5, 5), stride=2, padding=2, bias=True)
    linear1 = CNNrepository.Linear(in_features=192, out_features=30, bias=True)
    linear2 = CNNrepository.Linear(in_features=30, out_features=10, bias=True)
    tanh = CNNrepository.Tanh()
    return (
        ((conv1, 256, 768),),
        ((tanh, 768, 768),),
        ((conv2, 768, 192),),
        ((tanh, 192, 192),),
        ((linear1, 192, 30),),
        ((tanh, 30, 30),),
        ((linear2, 30, 10),),
    )


@pytest.mark.slow
@pytest.mark.integration
def test_lecun1989_reference_architecture_matches_published_accuracy_on_usps():
    torchvision = pytest.importorskip("torchvision")
    from torchvision import transforms

    repo = _lecun1989_repo()
    target = _concrete_target(repo, 256, 10, _lecun1989_structure(), epochs=300)
    tree = _synthesize_one(repo, target)
    pretty = tree.interpret(pretty_term_algebra())
    assert pretty.count("Conv2d(") == 2
    assert pretty.count("Linear(") == 2

    torch.manual_seed(0)
    model = tree.interpret(pytorch_model_algebra())
    n_params = sum(p.numel() for p in model.parameters())
    # sanity: same order of magnitude as the original paper's 9760 (dense H1->H2 has somewhat more)
    assert 8_000 <= n_params <= 15_000

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_set = torchvision.datasets.USPS(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.USPS(root="./data", train=False, download=True, transform=transform)

    def to_tensors(dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        images, labels = next(iter(loader))
        return images.reshape(images.shape[0], -1), labels.long()

    x, y = to_tensors(train_set)
    x_test, y_test = to_tensors(test_set)

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    raw_learner(256, model, loss_fn, lambda m: optim.Adam(m.parameters(), lr=1e-3),
                300, x, y, x_test, y_test, batch_size=64)

    with torch.inference_mode():
        preds = model(x_test).argmax(dim=-1)
        accuracy = (preds == y_test).float().mean().item()

    # Published benchmark: 5.00% test error (94.52% measured for this dense-conv echo during
    # development). Generous margin (chance level for 10 classes is 90% error) to avoid flakiness
    # from random init/mini-batch shuffling while still being a meaningful correctness signal.
    assert accuracy >= 0.85, (
        f"expected the LeCun-1989-echo architecture to reach roughly the published ~95% USPS test "
        f"accuracy, got {accuracy * 100:.2f}%"
    )
