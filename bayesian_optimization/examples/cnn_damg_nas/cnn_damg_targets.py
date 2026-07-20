from cosy.core.types import Constructor, Literal

# Staged dataset strategy: N_in is the flattened input feature (channels*height*width) and drives the
# search-space size via the sparse feature set F (see CNNrepository.__init__). We therefore start small
# (USPS) before moving to the actual target datasets.
#
#   Stufe 1 - USPS:         1x16x16 -> N_in=256   (CPU smoke test / pipeline tractability check)
#   Stufe 2 - FashionMNIST: 1x28x28 -> N_in=784   (first real minimal CNN experiment, MNIST as an
#                                                   equally-shaped, more robust alternative)
#   Stufe 3 - CIFAR-10:     3x32x32 -> N_in=3072  (target experiment, meant to run on the A30)
#
# All three stages classify into 10 classes, so N_out=10 throughout. The Loss slot stays Literal(None)
# (wildcard), so synthesis is free to pick cross_entropy_loss.
N_OUT_CLASSES = 10


def _make_target(n_in: int, length: int, n_out: int = N_OUT_CLASSES, epochs: int = 2000):
    # Note that the literal value in Constructor("epochs", Literal(...)) is currently not allowed to be None!
    return Constructor("Learner", Constructor("DAG",
                                               Constructor("input", Literal(n_in))
                                               & Constructor("output", Literal(n_out))
                                               & Constructor("structure", Literal((None,) * length)))
                        & Constructor("Loss", Constructor("type", Literal(None)))
                        & Constructor("Optimizer", Constructor("type", Literal(None)))
                        & Constructor("epochs", Literal(epochs))
                        )


# Stufe 1: USPS (1x16x16, N_in=256)
usps_target_len_2 = _make_target(256, 2)
usps_target_len_3 = _make_target(256, 3)
usps_target_len_4 = _make_target(256, 4)

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


_NAMED_TARGETS = {
    "usps_target_len_2": usps_target_len_2,
    "usps_target_len_3": usps_target_len_3,
    "usps_target_len_4": usps_target_len_4,
    "usps_lecun1989_target": usps_lecun1989_target,
    "fmnist_target_len_2": fmnist_target_len_2,
    "fmnist_target_len_3": fmnist_target_len_3,
    "fmnist_target_len_4": fmnist_target_len_4,
    "cifar_target_len_2": cifar_target_len_2,
    "cifar_target_len_3": cifar_target_len_3,
    "cifar_target_len_4": cifar_target_len_4,
}


def target_to_name(target):
    for name, value in _NAMED_TARGETS.items():
        if value == target:
            return name
    return "unknown"
