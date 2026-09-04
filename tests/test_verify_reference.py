"""The reference tool asks for the architecture it claims to anchor, and it says so in numbers.

``cnn_damg_verify_reference`` trains one term and writes down what it reaches. Every later
measurement on a target derived from that structure is read against this run, so the run is worth
nothing if the term it trained was not the reference architecture.

Two things carry that. The tool's own guard compares the synthesized term against
``EXPECTED_PARAMETERS`` and refuses to train anything else, and that constant is a number written
into the source. Here the constant is held against torchvision's ``vgg11_bn`` and the guard is run
on the target the tool actually builds, so neither side is taken on faith.

Nothing here trains and nothing here reads a dataset.
"""

from __future__ import annotations

import pytest

from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_verify_reference as reference


def test_expected_parameters_is_torchvisions_count_without_the_cancelled_biases():
    """The constant in the source decomposes against torchvision, computed here rather than copied.

    The convolutions of ``vgg11_bn.features`` carry biases and this repository's do not, because a
    convolution followed by BatchNorm is the same function either way, so the 2752 bias terms come
    off. The 5130 that go on are the CIFAR classifier, 512 * 10 weights and 10 biases.
    """
    models = pytest.importorskip("torchvision.models")

    features = sum(p.numel() for p in models.vgg11_bn(num_classes=10).features.parameters())
    assert reference.EXPECTED_PARAMETERS == features - 2_752 + 5_130


def test_the_target_of_the_tool_synthesizes_a_term_of_that_size():
    """The guard inside the tool passes on the target the tool builds, so a run gets that far.

    This is the same identity as above read from the other side: the term the query returns is the
    reference architecture, not merely something the query admits.
    """
    from cosy.core import Synthesizer

    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_network_algebras import (
        pytorch_model_algebra,
    )
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_reference_architectures import (
        cifar10_vgg11_bn_repo,
    )

    repo = cifar10_vgg11_bn_repo(epochs=50)
    target = reference.build_target(50, 0.1, 0.9, 5e-4)
    space = Synthesizer(repo.specification(), {}).construct_solution_space(target).prune()
    tree = next(iter(space.enumerate_trees(target)))

    model = tree.interpret(pytorch_model_algebra())
    assert sum(p.numel() for p in model.parameters()) == reference.EXPECTED_PARAMETERS
