from cosy.core.types import Constructor, Literal

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository

# Staged dataset strategy: N_in is the flattened input feature (channels*height*width) and drives the
# search-space size via the sparse feature set F (see CNNrepository.__init__). We therefore start small
# (USPS) before moving to the actual target datasets.
#
#   Stufe 1 - USPS:         1x16x16 -> N_in=256   (CPU smoke test / pipeline tractability check)
#   Stufe 2 - FashionMNIST: 1x28x28 -> N_in=784   (first real minimal CNN experiment, MNIST as an
#                                                   equally-shaped, more robust alternative)
#   Stufe 3 - CIFAR-10:     3x32x32 -> N_in=3072  (target experiment, meant to run on the A30)
#
# All three stages classify into 10 classes, so N_out=10 throughout.
N_OUT_CLASSES = 10


def _make_target(n_in: int, length: int, n_out: int = N_OUT_CLASSES, epochs: int = 2000, loss=None):
    # `loss=None` keeps the Loss slot a wildcard (Literal(None)) - synthesis is then free to pick
    # any cross_entropy_loss variant (any reduction). Passing a concrete loss instance (e.g.
    # CNNrepository.CrossEntropyLoss(reduction="mean")) instead pins the exact loss for this
    # target - the query language's own mechanism for eliminating variance at a given slot, no
    # repository/combinator change needed (the same idea already used for `structure` literals).
    # Note that the literal value in Constructor("epochs", Literal(...)) is currently not allowed to be None!
    return Constructor("Learner", Constructor("DAG",
                                               Constructor("input", Literal(n_in))
                                               & Constructor("output", Literal(n_out))
                                               & Constructor("structure", Literal((None,) * length)))
                        & Constructor("Loss", Constructor("type", Literal(loss)))
                        & Constructor("Optimizer", Constructor("type", Literal(None)))
                        & Constructor("epochs", Literal(epochs))
                        )


# Stufe 1: USPS (1x16x16, N_in=256)
usps_target_len_2 = _make_target(256, 2)
usps_target_len_3 = _make_target(256, 3)
usps_target_len_4 = _make_target(256, 4)

# First real NAS search experiment (not a single pinned architecture): wildcard length-4 target,
# few epochs per candidate (50, not 2000) to keep each BO evaluation fast during a small
# validation run (see cnn_damg_usps_experiment.py). Loss is pinned to CrossEntropyLoss(mean) -
# not a wildcard - so every evaluated structure's objective value is on the same scale and
# directly comparable (leaving it a wildcard would let synthesis also pick reduction="sum",
# whose values are on a completely different scale and not comparable to "mean" ones).
usps_experiment_target = _make_target(256, 4, epochs=50, loss=CNNrepository.CrossEntropyLoss(reduction="mean"))

# Historically smallest known CNN that reasonably solves this exact task: LeCun et al. 1989,
# "Backpropagation Applied to Handwritten Zip Code Recognition" (Neural Computation 1, 541-551) -
# the earliest real-world application of a backprop-trained neural net, applied directly to 16x16
# USPS zip-code digits (the same dataset this stage uses). Architecture (H1/H2/H3/output):
#   Conv(1->12, kernel 5x5, stride 2, padding 2)  : 1x16x16  -> 12x8x8  (flattened 256 -> 768)
#   Conv(12->12, kernel 5x5, stride 2, padding 2)  : 12x8x8   -> 12x4x4  (flattened 768 -> 192)
#   FC(192->30), FC(30->10)
# with tanh activations between layers - 7 sequential components in total (conv, tanh, conv, tanh,
# linear, tanh, linear). The original paper used *sparse* (not fully dense) connectivity between H1
# and H2 and reported 5.00% test error with 9760 parameters total; our conv2d combinator only
# supports dense convolutions, so the exact parameter count differs, but the per-layer feature-map
# shapes match exactly. structure stays a length-7 wildcard - the CNNrepository config used to
# synthesize this target is constrained tightly enough (see cnn_damg_nas tests) that this exact
# layer sequence is the (near-)only path through the search space.
usps_lecun1989_target = _make_target(256, 7, epochs=300)

# Stufe 2: FashionMNIST / MNIST (1x28x28, N_in=784)
fmnist_target_len_2 = _make_target(784, 2)
fmnist_target_len_3 = _make_target(784, 3)
fmnist_target_len_4 = _make_target(784, 4)

# Stufe 3: CIFAR-10 (3x32x32, N_in=3072)
cifar_target_len_2 = _make_target(3072, 2)
cifar_target_len_3 = _make_target(3072, 3)
cifar_target_len_4 = _make_target(3072, 4)

# Well-known small reference CNN, chosen (over e.g. Caffe's cifar10_quick, 75.33% test accuracy) because
# its pooling already satisfies our stride == kernel_size restriction on MaxPool2d - see the "Future
# Extensions" note in the CNN refactoring plan for what relaxing that restriction would take. This is
# the official PyTorch CIFAR-10 tutorial network:
#   Conv(3->6, 5x5) -> ReLU -> MaxPool(2x2) -> Conv(6->16, 5x5) -> ReLU -> MaxPool(2x2) ->
#   FC(400->120) -> ReLU -> FC(120->84) -> ReLU -> FC(84->10)
# 11 sequential components in total. structure stays a length-11 wildcard here; the exact concrete
# (non-wildcard) structure literal is built directly in tests/integration/test_cnn_damg_nas.py, the
# same way as for usps_lecun1989_target.
cifar10_pytorch_tutorial_target = _make_target(3072, 11, epochs=20)


_NAMED_TARGETS = {
    "usps_target_len_2": usps_target_len_2,
    "usps_target_len_3": usps_target_len_3,
    "usps_target_len_4": usps_target_len_4,
    "usps_lecun1989_target": usps_lecun1989_target,
    "usps_experiment_target": usps_experiment_target,
    "fmnist_target_len_2": fmnist_target_len_2,
    "fmnist_target_len_3": fmnist_target_len_3,
    "fmnist_target_len_4": fmnist_target_len_4,
    "cifar_target_len_2": cifar_target_len_2,
    "cifar_target_len_3": cifar_target_len_3,
    "cifar_target_len_4": cifar_target_len_4,
    "cifar10_pytorch_tutorial_target": cifar10_pytorch_tutorial_target,
}


def target_to_name(target):
    for name, value in _NAMED_TARGETS.items():
        if value == target:
            return name
    return "unknown"
