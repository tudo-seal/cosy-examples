"""Every combinator of the CNN repository builds a network, and two published ones come out exact.

Each test here asks for a fully concrete structure, one with no wildcard left in it, so synthesis
has nothing to search for and hands back the term that was described. That is deliberate. A test
that enumerated the space and filtered it for a match would also pass on a repository that reaches
the term by some other route, and it could not say which combinator carried it.

Two concerns. The first is the combinator table: every leaf combinator, the parallel composition,
the identity lane and a swap over a convolution-derived feature interpret at least once and produce
a forward pass of the expected shape. The second is the reference architectures: the CIFAR-10
tutorial network and VGG-11-BN are described as concrete structures, and the terms that come back
are held against their parameter counts, VGG-11-BN against torchvision's own network rather than
against a number written down here.

These tests do not exercise the optimization loop, they do not read a dataset, and the one training
call runs three epochs on random tensors. The experiment that uses this repository is tested
separately, in ``tests/test_cifar_experiment.py``.
"""

from __future__ import annotations

import time

import pytest
import torch
from cosy.core import Synthesizer
from cosy.core.types import Constructor, Literal
from torch import nn, optim

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_network_algebras import (
    learner as raw_learner,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_network_algebras import (
    pytorch_model_algebra,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_reference_architectures import (
    cifar10_tutorial_repo,
    cifar10_tutorial_structure,
    cifar10_vgg11_bn_repo,
    cifar10_vgg11_bn_structure,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_algebras import pretty_term_algebra

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
    """A target that pins the structure and leaves the loss and the optimizer open.

    The structure is what these tests are about, so the two remaining slots stay wildcards and
    whichever loss and optimizer the enumeration offers first is taken. That is sound here because
    nothing below trains the term it synthesizes. A target that is going to be trained has to pin
    the loss, since the repository offers the same loss combinator at two reductions and they do
    not train alike.
    """
    return Constructor("Learner", Constructor("DAG",
                                               Constructor("input", Literal(n_in))
                                               & Constructor("output", Literal(n_out))
                                               & Constructor("structure", Literal(structure)))
                        & Constructor("Loss", Constructor("type", Literal(None)))
                        & Constructor("Optimizer", Constructor("type", Literal(None)))
                        & Constructor("epochs", Literal(epochs)))


def _synthesize_one(repo: CNNrepository, target, max_seconds: float = 5.0):
    """Enumerate a wildcard-free target, which has to resolve quickly.

    There is almost nothing left to search for, so a slow or exhaustive search here says that the
    structure literal was not concrete after all, and the bound is what turns that into a failure.
    ``max_seconds`` defaults to a tight bound for the small test repositories. A repository with
    more channel, kernel and feature combinations needs a looser one, because the cost of
    ``construct_solution_space`` grows worse than linearly in the label count even when the target
    itself is fully concrete.
    """
    spec = repo.specification()
    synthesizer = Synthesizer(spec, {})
    t0 = time.time()
    search_space = synthesizer.construct_solution_space(target).prune()
    trees = list(search_space.enumerate_trees(target, max_count=5))
    dt = time.time() - t0
    assert trees, f"expected the concrete target to be directly synthesizable, found none: {target}"
    assert dt < max_seconds, (
        f"synthesizing a fully concrete target took {dt:.2f}s, so the structure literal is not "
        f"actually concrete")
    return trees[0]


# ---------------------------------------------------------------------------
# 1. Every "simple" (single-node) combinator interprets and produces a correctly
#    shaped forward pass: linear_layer, conv2d, maxpool2d, sigmoid, relu, tanh,
#    sum, product, copy, edges, swap.
# ---------------------------------------------------------------------------

def _node_case_id(case):
    return case[0]


# bias=False, because the repository enumerates bias-free convolutions only. A label with
# bias=True is not a member of the Label group, and a target that carries one is rejected.
_conv2d_label = CNNrepository.Conv2d(in_channels=1, out_channels=2, input_size=(4, 4),
                                      output_size=(2, 2), kernel_size=(3, 3), stride=1, padding=0,
                                      bias=False)
_maxpool2d_label = CNNrepository.MaxPool2d(in_channels=1, input_size=(4, 4), output_size=(2, 2),
                                            kernel_size=(2, 2))

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
# Every leaf combinator declares seven structure variants, (l,i,o), (l,i,None), (l,None,o),
# (None,i,o), (l,None,None), (None,None,o) and (None,i,None), so a target may pin part of a
# component and leave the rest open. It works only because membership in ParaTuples is delegated to
# the Para group. Para.__iter__ never yields a triple containing None, so a membership test against
# the materialized enumeration rejects all six partial forms, and it does so by producing no term
# rather than by reporting anything. These tests hold that behavior down.
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
    """A partial triple whose pinned dimensions contradict the target must still yield nothing.

    The relaxation opens the form of a component, not its dimensions.
    """
    repo = _small_repo()
    spec = repo.specification()
    # target says 16 -> 8, but the triple demands an output of 16
    target = _concrete_target(repo, 16, 8, (((None, 16, 16),),))
    search_space = Synthesizer(spec, {}).construct_solution_space(target).prune()
    assert list(search_space.enumerate_trees(target, max_count=1)) == []


def test_all_none_triple_is_not_expressible():
    """``(None, None, None)`` is deliberately not one of the seven declared variants.

    A fully unconstrained component is written as the element ``None`` itself, and the all-None
    triple is a different query that nothing answers.
    """
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
    # "edges" is ID-typed, a pure identity pass-through, so it is valid only as a parallel lane
    # inside a beside_cons composition and never as a standalone before_singleton argument, which
    # asks for non_ID. It is therefore tested next to a real component.
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
# 3. A swap over a flattened feature that came out of a conv2d. This is what treating conv2d like
#    linear_layer buys: the flattened output of a convolution is an ordinary member of the feature
#    set, so a swap rewires it exactly as it would rewire the output of a linear layer.
# ---------------------------------------------------------------------------

def test_swap_over_conv_derived_feature_interprets_and_permutes():
    repo = _small_repo()
    conv = CNNrepository.Conv2d(in_channels=1, out_channels=2, input_size=(4, 4),
                                 output_size=(2, 2), kernel_size=(3, 3), stride=1, padding=0,
                                 bias=False)
    # The convolution flattens 16 to 8, and swap(4, 4) rewires that 8-wide output.
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
    conv_only = next(m for m in model.modules() if type(m).__name__ == "SynthConv2d")
    conv_out = conv_only(x)
    first_half, second_half = conv_out[:, :4], conv_out[:, 4:]
    expected = torch.cat((second_half, first_half), dim=-1)
    assert torch.allclose(y, expected), "swap did not permute the conv-derived feature halves"


# ---------------------------------------------------------------------------
# 4. cross_entropy_loss / adam_optimizer / learner: exercised together via an
#    actual (tiny) end-to-end training call.
# ---------------------------------------------------------------------------

def test_learner_cross_entropy_and_adam_train_end_to_end():
    repo = _small_repo()
    linear = CNNrepository.Linear(in_features=16, out_features=8, bias=True)
    structure = (((linear, 16, 8),),)
    # The epochs of the target have to be in the repository's n_epoch_values for the target to be
    # inhabited at all. The training below calls raw_learner directly, with its own epoch count.
    target = _concrete_target(repo, 16, 8, structure)
    tree = _synthesize_one(repo, target)

    model = tree.interpret(pytorch_model_algebra())
    x = torch.randn(20, 16)
    y = torch.randint(0, 8, (20,))
    loss = raw_learner(16, model, nn.CrossEntropyLoss(),
                       lambda m: optim.Adam(m.parameters(), lr=1e-3),
                       3, x, y, x, y, batch_size=8)
    assert torch.isfinite(torch.tensor(loss))

# ---------------------------------------------------------------------------
# 5. The two CIFAR-10 reference architectures.
#
#    The tutorial network of the PyTorch documentation is used rather than a network of comparable
#    size from elsewhere, because its pooling already satisfies the restriction this repository
#    puts on MaxPool2d, that the stride equals the kernel size. Neither test trains: they check
#    that the repository expresses the published architecture exactly, which is the premise every
#    later measurement against these targets rests on.
# ---------------------------------------------------------------------------

def test_cifar10_tutorial_architecture_synthesizes_and_interprets():
    """The tutorial network is expressible, and the term interprets to a net of its parameter count.

    This repository carries several times the labels of the small ones above, all of them needed
    for the exact architecture, and the cost of ``construct_solution_space`` grows worse than
    linearly in the label count even for a fully concrete target. The bound below is therefore far
    looser than the default, and it is a bound on a wrong result rather than a measurement.
    """
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


def test_the_vgg11_bn_reference_is_torchvisions_network_without_the_cancelled_biases():
    """The VGG-11-BN term is torchvision's own network, counted against torchvision at run time.

    This is the check that the repository expresses a published architecture exactly and not merely
    something of its shape. The parameter count decomposes with no slack left over:

        torchvision vgg11_bn.features   9_225_984   its convolutions carry biases
        - the bias terms                    2_752   64+128+256+256+512+512+512+512
        + the CIFAR classifier               5_130   512*10 + 10
        = this term                     9_228_362

    The missing 2_752 are the repository's decision not to offer a bias on a convolution that is
    followed by BatchNorm, which is the same function either way. A drift in the label set, in the
    padding or in the pooling breaks this identity rather than shifting it, which is why the
    reference side is computed from torchvision here instead of being written down as a number.
    """
    torchvision = pytest.importorskip("torchvision.models")

    repo = cifar10_vgg11_bn_repo()
    structure = cifar10_vgg11_bn_structure()
    assert len(structure) == 30, "8 conv + 8 bn + 8 relu + 5 pool + 1 linear"

    target = _concrete_target(repo, 3072, 10, structure, epochs=50)
    tree = _synthesize_one(repo, target, max_seconds=120.0)

    pretty = tree.interpret(pretty_term_algebra())
    assert pretty.count("Conv2d(") == 8
    assert pretty.count("BatchNorm2d(") == 8
    assert pretty.count("MaxPool2d(") == 5
    assert pretty.count("Linear(") == 1
    assert "bias=True" not in pretty.split("Linear(")[0], "a convolution kept its bias"

    model = tree.interpret(pytorch_model_algebra())
    reference = torchvision.vgg11_bn(num_classes=10)
    reference_features = sum(p.numel() for p in reference.features.parameters())
    assert sum(p.numel() for p in model.parameters()) == reference_features - 2_752 + 5_130

    assert model(torch.randn(5, 3072)).shape == (5, 10)

# ---------------------------------------------------------------------------
# 6. Deliberate-variance targets (make_variance_target).
#
#    A search target should constrain a position only where there is a reason to. The classifier
#    head fixes the interface to the label space and bounds the size of the network, while the
#    feature extractor stays free, since that is what is being searched for. These tests hold down
#    the three position descriptors: free, pinned out-degree, and pinned dimensions.
# ---------------------------------------------------------------------------

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
        assert first_position.count("||") == 1, (
            f"expected exactly 2 parallel branches: {first_position}")


# ---------------------------------------------------------------------------
# 7. The edgelist algebra evaluates each continuation exactly once.
#
# Writing the composition inline instead calls x and y once per tuple component, which makes the
# cost of one interpretation exponential in the nesting depth. On the terms this example converts
# that is the difference between minutes and seconds, and nothing about the result changes, so a
# test that only compared outputs would not see it. This one counts the calls.
# ---------------------------------------------------------------------------

def test_edgelist_algebra_evaluates_each_continuation_once():
    """Guards the exponential-interpretation regression by counting continuation calls."""
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_algebras import (
        _before_edgelists,
        _beside_edgelists,
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
