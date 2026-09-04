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


# ---------------------------------------------------------------------------
# What the three slots of the reference target admit.
#
# The structure is fully concrete, so each of these spaces is finite and small enough to enumerate
# completely. That is what makes the counts below say anything: a count over a prefix of an
# enumeration would depend on an order that ``enumerate_trees`` does not promise, while a closed
# count of a finite space depends on nothing.
# ---------------------------------------------------------------------------

def _closed_target(loss, optimizer, scheduler, epochs=50):
    """The reference structure with the three learner slots set to whatever is passed."""
    from cosy.core.types import Constructor, Literal

    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_reference_architectures import (
        cifar10_vgg11_bn_structure,
    )

    return Constructor("Learner", Constructor("DAG",
                                              Constructor("input", Literal(3072))
                                              & Constructor("output", Literal(10))
                                              & Constructor("structure",
                                                            Literal(cifar10_vgg11_bn_structure())))
                       & Constructor("Loss", Constructor("type", Literal(loss)))
                       & Constructor("Optimizer", Constructor("type", Literal(optimizer)))
                       & Constructor("Scheduler", Constructor("type", Literal(scheduler)))
                       & Constructor("epochs", Literal(epochs)))


def _inhabitants(target, epochs=50):
    from cosy.core import Synthesizer

    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_reference_architectures import (
        cifar10_vgg11_bn_repo,
    )

    repo = cifar10_vgg11_bn_repo(epochs=epochs)
    space = Synthesizer(repo.specification(), {}).construct_solution_space(target).prune()
    return list(space.enumerate_trees(target, max_count=20))


def test_the_target_of_the_tool_has_exactly_one_inhabitant():
    """The tool trains the term it describes, and there is no second term it could have trained."""
    trees = _inhabitants(reference.build_target(50, 0.1, 0.9, 5e-4))
    assert len(trees) == 1
    assert trees[0].size == 734


def test_an_open_loss_slot_would_admit_both_reductions_of_the_same_network():
    """This is what the pinned slot buys, and it is why the slot cannot stay open.

    Both terms describe the same network, same size and same parameter count, and they differ in
    one leaf. At a batch of 128 the sum reduction scales every gradient by 128, so the two do not
    train alike, and nothing in the run would say which one it got.
    """
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_network_algebras import (
        pytorch_components_algebra,
    )
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository

    optimizer = CNNrepository.SGD(learning_rate=0.1, momentum=0.9, weight_decay=5e-4)
    trees = _inhabitants(_closed_target(None, optimizer, CNNrepository.CosineAnnealingLR()))

    assert len(trees) == 2
    assert {tree.size for tree in trees} == {734}
    reductions = {tree.interpret(pytorch_components_algebra())[1].reduction for tree in trees}
    assert reductions == {"mean", "sum"}


def test_an_open_recipe_puts_one_term_per_recipe_into_the_space():
    """Four terms of one network, which is the count ``make_vgg11_bn_target`` documents.

    The optimizer slot and the schedule slot are independent, so leaving both open multiplies the
    terms of every network by the product of what the repository offers there. What this holds
    down is that product: two optimizers times two schedules on this structure. It falls when the
    repository gains or loses one of the four, and that number is what a search of this space
    spends three quarters of its draws on.
    """
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_algebras import (
        pretty_term_algebra,
    )

    loss = CNNrepository.CrossEntropyLoss(reduction="mean")
    trees = _inhabitants(_closed_target(loss, None, None))

    assert len(trees) == 4
    assert {tree.size for tree in trees} == {734}
    recipes = set()
    for tree in trees:
        pretty = tree.interpret(pretty_term_algebra())
        optimizer = "Adam" if "CNNrepository.Adam(" in pretty else "SGD"
        schedule = "none" if "CNNrepository.NoScheduler(" in pretty else "cosine"
        recipes.add((optimizer, schedule))
    assert recipes == {("Adam", "cosine"), ("Adam", "none"), ("SGD", "cosine"), ("SGD", "none")}
