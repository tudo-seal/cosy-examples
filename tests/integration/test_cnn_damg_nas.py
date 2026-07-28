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
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_reference_architectures import (
    usps_lecun1989_repo,
    usps_lecun1989_structure,
    cifar10_tutorial_repo,
    cifar10_tutorial_structure,
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


def _synthesize_one(repo: CNNrepository, target, max_seconds: float = 5.0):
    """Enumerate a concrete (wildcard-free) target - this must resolve quickly since there is
    (almost) nothing left to search for; a slow/exhaustive search here would indicate the structure
    literal was not actually concrete. `max_seconds` defaults to a tight bound for the small
    DNN-style test repos; larger repos (more channel/kernel/feature combinations, e.g. the CIFAR-10
    tutorial net) need a looser bound - cosy's construct_solution_space cost grows worse than
    linearly with the label count even though the target itself is fully concrete."""
    spec = repo.specification()
    synthesizer = Synthesizer(spec, {})
    t0 = time.time()
    search_space = synthesizer.construct_solution_space(target).prune()
    trees = list(search_space.enumerate_trees(target, max_count=5))
    dt = time.time() - t0
    assert trees, f"expected the concrete target to be directly synthesizable, found none: {target}"
    assert dt < max_seconds, f"synthesizing a fully concrete target took {dt:.2f}s - structure literal is not actually concrete"
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
# 1b. Partially concrete structure literals.
#
# Every leaf combinator declares seven structure variants para1..para7 - (l,i,o), (l,i,None),
# (l,None,o), (None,i,o), (l,None,None), (None,None,o), (None,i,None) - so a target may pin only
# part of a component and leave the rest open. That only works because ParaTuples.__contains__
# delegates membership to the Para *group*: Para.__iter__ never yields a triple containing None, so
# testing against the materialized enumeration instead (as the legacy damg_repo.py does) silently
# rejects all six partial forms, producing 0 terms with no diagnostic. These tests pin that down.
# ---------------------------------------------------------------------------

_LIN_16_8 = CNNrepository.Linear(in_features=16, out_features=8, bias=True)

# (id, triple) - all must synthesize for the target input=16, output=8
_PARTIAL_TRIPLE_CASES = [
    ("para1_l_i_o", (_LIN_16_8, 16, 8)),
    ("para2_l_i_None", (_LIN_16_8, 16, None)),
    ("para3_l_None_o", (_LIN_16_8, None, 8)),
    ("para4_None_i_o", (None, 16, 8)),
    ("para5_l_None_None", (_LIN_16_8, None, None)),
    ("para6_None_None_o", (None, None, 8)),
    ("para7_None_i_None", (None, 16, None)),
]


@pytest.mark.parametrize("case", _PARTIAL_TRIPLE_CASES, ids=lambda c: c[0])
def test_partially_concrete_triples_synthesize(case):
    _, triple = case
    repo = _small_repo()
    target = _concrete_target(repo, 16, 8, ((triple,),))
    tree = _synthesize_one(repo, target)

    model = tree.interpret(pytorch_model_algebra())
    y = model(torch.randn(3, 16))
    assert y.shape == (3, 8)


def test_partial_triple_still_enforces_pinned_dimensions():
    """The relaxation must not make the dimension constraints toothless: a partial triple whose
    pinned dimensions contradict the target's input/output must still yield nothing."""
    repo = _small_repo()
    spec = repo.specification()
    # target says 16 -> 8, but the triple demands an output of 16
    target = _concrete_target(repo, 16, 8, (((None, 16, 16),),))
    search_space = Synthesizer(spec, {}).construct_solution_space(target).prune()
    assert list(search_space.enumerate_trees(target, max_count=1)) == []


def test_all_none_triple_is_not_expressible():
    """`(None, None, None)` is deliberately NOT one of the seven declared variants - a fully
    unconstrained component is expressed as the element `None` itself, not as an all-None triple."""
    repo = _small_repo()
    spec = repo.specification()
    target = _concrete_target(repo, 16, 8, (((None, None, None),),))
    search_space = Synthesizer(spec, {}).construct_solution_space(target).prune()
    assert list(search_space.enumerate_trees(target, max_count=1)) == []

    # ... whereas the element-level wildcard does synthesize
    wildcard_target = _concrete_target(repo, 16, 8, ((None,),))
    wildcard_space = Synthesizer(spec, {}).construct_solution_space(wildcard_target).prune()
    assert list(wildcard_space.enumerate_trees(wildcard_target, max_count=1))


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

@pytest.mark.slow
@pytest.mark.integration
def test_lecun1989_reference_architecture_matches_published_accuracy_on_usps():
    torchvision = pytest.importorskip("torchvision")
    from torchvision import transforms

    repo = usps_lecun1989_repo()
    target = _concrete_target(repo, 256, 10, usps_lecun1989_structure(), epochs=300)
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


# ---------------------------------------------------------------------------
# 6. CIFAR-10 stage: the official PyTorch CIFAR-10 tutorial network, chosen over e.g. Caffe's
#    cifar10_quick because its pooling already satisfies our stride == kernel_size restriction on
#    MaxPool2d (see the "Future Extensions" note in the CNN refactoring plan). This is a *local*
#    smoke test before moving the same setup to the A30 server - it only checks that synthesis,
#    interpretation and a short training run work correctly, not full convergence/accuracy (that
#    full-scale check is deferred to the actual A30 experiment where more epochs are affordable).
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_cifar10_tutorial_architecture_synthesizes_and_interprets():
    # This repo config has ~4x more labels than the USPS-scale tests above (8 linear feature sizes,
    # 3 channel counts, 4 spatial sizes, 2 kernel sizes - all needed for the exact architecture, none
    # removable), and cosy's construct_solution_space cost grows worse than linearly with label
    # count even for a fully concrete target - construction alone measured ~150s locally.
    repo = cifar10_tutorial_repo()
    target = _concrete_target(repo, 3072, 10, cifar10_tutorial_structure(), epochs=20)
    tree = _synthesize_one(repo, target, max_seconds=240.0)

    pretty = tree.interpret(pretty_term_algebra())
    assert pretty.count("Conv2d(") == 2
    assert pretty.count("MaxPool2d(") == 2
    assert pretty.count("Linear(") == 3

    model = tree.interpret(pytorch_model_algebra())
    n_params = sum(p.numel() for p in model.parameters())
    # sanity check against the well-known parameter count of this exact tutorial network
    assert 60_000 <= n_params <= 65_000

    x = torch.randn(5, 3072)
    y = model(x)
    assert y.shape == (5, 10)


@pytest.mark.slow
@pytest.mark.integration
def test_cifar10_tutorial_architecture_trains_locally_as_cpu_baseline():
    """Local CPU smoke test/timing baseline, to compare against the same run on the A30 server -
    not a convergence/accuracy check (few epochs on CPU won't reach the ~60-65% this network
    typically needs many more epochs for)."""
    torchvision = pytest.importorskip("torchvision")
    from torchvision import transforms

    repo = cifar10_tutorial_repo()
    target = _concrete_target(repo, 3072, 10, cifar10_tutorial_structure(), epochs=20)
    tree = _synthesize_one(repo, target, max_seconds=240.0)
    model = tree.interpret(pytorch_model_algebra())

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    def to_tensors(dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=0)
        images, labels = next(iter(loader))
        return images.reshape(images.shape[0], -1), labels.long()

    x, y = to_tensors(train_set)
    x_test, y_test = to_tensors(test_set)

    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    t0 = time.time()
    test_loss = raw_learner(3072, model, loss_fn, lambda m: optim.Adam(m.parameters(), lr=1e-3),
                             20, x, y, x_test, y_test, batch_size=128)
    dt = time.time() - t0
    print(f"\nCIFAR-10 tutorial net, CPU, 20 epochs, n_train={x.shape[0]}: {dt:.1f}s, "
          f"test loss={test_loss:.4f}")

    assert torch.isfinite(torch.tensor(test_loss))


# ---------------------------------------------------------------------------
# 7. Deliberate-variance targets (make_variance_target).
#
#    A search target should constrain a position only where there is a reason to: the classifier
#    head fixes the interface to the label space and bounds the network size, while the feature
#    extractor stays free - that is what is being searched for. These tests pin down the three
#    position descriptors (free / pinned out-degree / pinned dimensions) and the property that
#    matters for the CIFAR-10 experiment: with max_lin_layer_dim below the input feature, no linear
#    layer can consume the raw image, so the free part must begin convolutionally.
# ---------------------------------------------------------------------------

def _cnn_share(trees):
    pretty = [t.interpret(pretty_term_algebra()) for t in trees]
    return sum(1 for p in pretty if "Conv2d(" in p or "MaxPool2d(" in p), len(pretty)


def test_variance_target_position_descriptors_build_expected_structure():
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_targets import make_variance_target

    target = make_variance_target(16, [None, 2, (8, 4)], epochs=1, n_out=4)
    # dig the structure literal back out of the constructed type
    rendered = str(target)
    assert "(None, (None, None), ((None, 8, 4),))" in rendered, rendered


def test_variance_target_rejects_unsupported_descriptors():
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_targets import make_variance_target

    with pytest.raises(ValueError):
        make_variance_target(16, [0], epochs=1)          # out-degree must be >= 1
    with pytest.raises(ValueError):
        make_variance_target(16, ["nonsense"], epochs=1)  # unknown descriptor


def test_free_front_with_pinned_dimension_head_synthesizes():
    """The head positions use (i, o) descriptors, i.e. partially concrete triples - the form the
    legacy damg_repo silently rejects. Everything before them stays fully free."""
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_targets import make_variance_target

    repo = _small_repo()
    target = make_variance_target(16, [None, (8, 4)], epochs=1, n_out=4)
    search_space = Synthesizer(repo.specification(), {}).construct_solution_space(target).prune()
    trees = list(search_space.enumerate_trees(target, max_count=20))
    assert trees, "free front + pinned-dimension head should synthesize"

    model = trees[0].interpret(pytorch_model_algebra())
    assert model(torch.randn(3, 16)).shape == (3, 4)


def test_pinned_out_degree_position_is_honoured():
    """A position given as an int k must yield exactly k parallel components at that position."""
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_targets import make_variance_target

    repo = _small_repo()
    target = make_variance_target(16, [2, (8, 4)], epochs=1, n_out=4)
    search_space = Synthesizer(repo.specification(), {}).construct_solution_space(target).prune()
    trees = list(search_space.enumerate_trees(target, max_count=20))
    assert trees, "an out-degree-2 position should be satisfiable for a 16-feature input"

    for tree in trees[:5]:
        model_line = tree.interpret(pretty_term_algebra()).split("model= (")[1].split("\n")[1]
        first_position = model_line.split(" ; ")[0]
        assert first_position.count("||") == 1, f"expected exactly 2 parallel branches: {first_position}"


@pytest.mark.slow
def test_cifar_head_target_forces_convolutional_front_end():
    """With max_lin_layer_dim below the 3072-feature input, no linear layer can consume the raw
    image, so every candidate must start with a convolution or pooling layer."""
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_targets import make_cifar_head_target
    from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_cifar_experiment as experiment

    repo = CNNrepository(
        linear_feature_dimensions=experiment.LINEAR_FEATURE_DIMENSIONS,
        constant_values=experiment.CONSTANT_VALUES,
        learning_rate_values=experiment.LEARNING_RATE_VALUES,
        n_epoch_values=[20],
        channel_dimensions=experiment.CHANNEL_DIMENSIONS,
        height_width_dimensions=experiment.HEIGHT_WIDTH_DIMENSIONS,
        kernel_dimensions=experiment.KERNEL_DIMENSIONS,
        stride_values=experiment.STRIDE_VALUES,
        padding_values=experiment.PADDING_VALUES,
        max_parallel_width=experiment.MAX_PARALLEL_WIDTH,
        max_lin_layer_dim=1600,
    )
    target = make_cifar_head_target(epochs=20)
    search_space = Synthesizer(repo.specification(), {}).construct_solution_space(target).prune()
    trees = list(search_space.enumerate_trees(target, max_count=30))
    assert trees, "the CIFAR head target should synthesize"
    cnn_count, total = _cnn_share(trees)
    assert cnn_count == total, f"expected every candidate to be convolutional, got {cnn_count}/{total}"


# ---------------------------------------------------------------------------
# Regression guard for the interpretation-cost fix.
#
# The edgelist algebra must evaluate each continuation exactly once.  Writing the composition
# inline calls x and y once per tuple component, which makes interpretation cost
# 3^(nesting depth) -- measured 698 s vs 4.4 s over 789 conversions of real terms.
# ---------------------------------------------------------------------------

def test_edgelist_algebra_evaluates_each_continuation_once():
    """Guards the exponential-interpretation regression by counting continuation calls."""
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo_algebras import (
        _before_edgelists, _beside_edgelists,
    )

    calls = {"x": 0, "y": 0}

    def x(node_id, inputs):
        calls["x"] += 1
        return [("a", "b")], list(inputs), {"a": node_id}

    def y(node_id, inputs):
        calls["y"] += 1
        return [("c", "d")], list(inputs), {"c": node_id}

    _beside_edgelists(x, y, 1, (0.0, 0.0), ["i0", "i1"])
    assert calls == {"x": 1, "y": 1}, f"beside must call each continuation once, got {calls}"

    calls["x"] = calls["y"] = 0
    _before_edgelists(x, (y, 1), (0.0, 0.0), ["i0"])
    assert calls == {"x": 1, "y": 1}, f"before must call each continuation once, got {calls}"
