from typing import Any

from cosy.core.types import Constructor, Literal

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository

# Staged dataset strategy: N_in is the flattened input feature (channels*height*width) and drives
# the search-space size through the sparse feature set F (see CNNrepository.__init__). The targets
# below therefore start small and grow.
#
#   stage 1, USPS:         1x16x16 -> N_in=256   (CPU smoke test and tractability check)
#   stage 2, FashionMNIST: 1x28x28 -> N_in=784   (first minimal CNN experiment, with MNIST as an
#                                                 equally shaped, more robust alternative)
#   stage 3, CIFAR-10:     3x32x32 -> N_in=3072  (the experiment the other two rehearse)
#
# All three stages classify into 10 classes, so N_out=10 throughout.
N_OUT_CLASSES = 10


def _make_target_from_structure(n_in: int, structure, n_out: int = N_OUT_CLASSES,
                                 epochs: int = 2000, loss=None):
    # `loss=None` keeps the Loss slot a wildcard (Literal(None)), so synthesis is free to pick any
    # cross_entropy_loss variant, that is any reduction. Passing a concrete loss instance such as
    # CNNrepository.CrossEntropyLoss(reduction="mean") pins the exact loss for this target instead.
    # That is the query language's own way of removing variance at one slot, and it needs no change
    # to the repository or to a combinator. The `structure` literals work the same way.
    #
    # `structure` is a tuple with one entry per sequential position. Forms known to synthesize:
    #   None                  any parallel width, any components (full variance)
    #   (None,)               exactly one component, otherwise unconstrained (arity control)
    #   ((label, i, o),)      exactly one fully concrete component
    #
    # Partially concrete triples such as `(None, i, o)`, with the dimensions pinned and the operator
    # free, are why `ParaTuples.__contains__` in cnn_damg_repo.py delegates to the Para group
    # instead of testing membership in a materialized tuple. `Para.__iter__` never yields a triple
    # containing None, while `Para.__contains__` accepts one. A target's structure literal is
    # inferred rather than enumerated, so it is validated exclusively through `__contains__`, and a
    # plain tuple test there rejects every partially concrete literal and yields 0 terms with no
    # diagnostic. damg_repo.py still tests against the materialized tuple, so the same literals do
    # not synthesize against it.
    #
    # `epochs=None` leaves the epoch count a wildcard, like every other slot here. That works
    # because the `learner` suffix in cnn_damg_repo.py carries Constructor("epochs", Var("epochs"))
    # and the matching Constructor("epochs", Literal(None)) beside it. Without the second conjunct a
    # target has to name the epoch count to be inhabited at all, "any epoch count" is
    # inexpressible, and a target whose count is not among the repository's n_epoch_values yields
    # 0 terms without saying so.
    return Constructor("Learner", Constructor("DAG",
                                               Constructor("input", Literal(n_in))
                                               & Constructor("output", Literal(n_out))
                                               & Constructor("structure", Literal(structure)))
                        & Constructor("Loss", Constructor("type", Literal(loss)))
                        & Constructor("Optimizer", Constructor("type", Literal(None)))
                        & Constructor("epochs", Literal(epochs))
                        )


def _make_target(n_in: int, length: int, n_out: int = N_OUT_CLASSES, epochs: int = 2000, loss=None):
    """Full-variance target: every one of `length` sequential positions is an unconstrained wildcard."""
    return _make_target_from_structure(n_in, (None,) * length, n_out, epochs, loss)


def make_full_variance_target(n_in: int, length: int, epochs: int, n_out: int = N_OUT_CLASSES,
                              loss=None):
    """Full-variance target with the epochs stated, for a caller that trains for its own budget.

    The epochs of a target and the repository's ``n_epoch_values`` must agree or nothing
    synthesizes. The target then asks for a literal the repository never offers, and the search
    space comes out empty.  It is a silent failure: construction succeeds, prunes to zero
    non-terminals, and the first thing that says so is the initializer, several steps later.
    The pinned targets below all carry the 2000-epoch default, so a caller training for fewer
    epochs needs its own target rather than a repository trimmed to fit. That is the rule
    everywhere here: the repository says what is expressible, the target says what is asked for.

    Args:
        n_in (int): Flattened input size.
        length (int): Number of sequential positions, each an unconstrained wildcard.
        epochs (int): Epochs per candidate, which the repository must offer.
        n_out (int): Number of classes. (Default value = N_OUT_CLASSES)
        loss: The loss to pin, or None to leave it a wildcard. (Default value = None)

    Returns:
        The requested type.
    """
    return _make_target(n_in, length, n_out, epochs, loss)


# Stage 1, USPS (1x16x16, N_in=256)
usps_target_len_2 = _make_target(256, 2)
usps_target_len_3 = _make_target(256, 3)
usps_target_len_4 = _make_target(256, 4)

def make_usps_experiment_target(epochs: int = 50, length: int = 5):
    """USPS NAS search target with a pinned loss, parameterized by epochs and structure length.

    The counterpart of `make_cifar_experiment_target`, and it exists for the same reason: the two
    numbers a cheap smoke test has to be able to lower are the epochs per candidate and the
    sequential depth of the searched structure, and both sit in the target rather than beside
    it. The repository's `n_epoch_values` must contain exactly this epoch count or nothing
    synthesizes, and it does so silently, since construction succeeds and prunes to zero.

    Loss is pinned to CrossEntropyLoss(mean) rather than left a wildcard, so every evaluated
    structure's objective value is on the same scale and directly comparable. A wildcard would let
    synthesis also pick reduction="sum", whose values are on a completely different scale.
    """
    return _make_target(256, length, epochs=epochs,
                        loss=CNNrepository.CrossEntropyLoss(reduction="mean"))


# A search target rather than a single pinned architecture: full variance over five positions, and
# few epochs per candidate (50 rather than 2000) so that one evaluation of the optimization loop
# stays cheap.  Defined by the function above rather than beside it, so the constant and the
# parameterized form cannot drift apart.
usps_experiment_target = make_usps_experiment_target(epochs=50, length=5)

# The smallest known CNN that reasonably solves this exact task: LeCun et al. 1989,
# "Backpropagation Applied to Handwritten Zip Code Recognition" (Neural Computation 1, 541-551).
# It is the earliest real-world application of a backprop-trained neural net, applied directly to
# the 16x16 USPS zip-code digits this stage uses. Architecture (H1/H2/H3/output):
#   Conv(1->12, kernel 5x5, stride 2, padding 2)   : 1x16x16  -> 12x8x8  (flattened 256 -> 768)
#   Conv(12->12, kernel 5x5, stride 2, padding 2)  : 12x8x8   -> 12x4x4  (flattened 768 -> 192)
#   FC(192->30), FC(30->10)
# with tanh activations between the layers, so 7 sequential components in total: conv, tanh, conv,
# tanh, linear, tanh, linear. The original paper used sparse rather than fully dense connectivity
# between H1 and H2 and reported 5.00% test error with 9760 parameters in total. The conv2d
# combinator here only supports dense convolutions, so the exact parameter count differs while the
# per-layer feature-map shapes match. structure stays a length-7 wildcard, because the repository
# this target is synthesized against (usps_lecun1989_repo in cnn_damg_reference_architectures.py)
# is tight enough that this layer sequence is very nearly the only path through the space.
usps_lecun1989_target = _make_target(256, 7, epochs=300)

# Stage 2, FashionMNIST and MNIST (1x28x28, N_in=784)
fmnist_target_len_2 = _make_target(784, 2)
fmnist_target_len_3 = _make_target(784, 3)
fmnist_target_len_4 = _make_target(784, 4)

# Stage 3, CIFAR-10 (3x32x32, N_in=3072)
cifar_target_len_2 = _make_target(3072, 2)
cifar_target_len_3 = _make_target(3072, 3)
cifar_target_len_4 = _make_target(3072, 4)

# A well-known small reference CNN, preferred over Caffe's cifar10_quick (75.33% test accuracy)
# because its pooling already satisfies the stride == kernel_size restriction that
# CNNrepository.Label.iter_maxpool2d imposes. This is the official PyTorch CIFAR-10 tutorial
# network:
#   Conv(3->6, 5x5) -> ReLU -> MaxPool(2x2) -> Conv(6->16, 5x5) -> ReLU -> MaxPool(2x2) ->
#   FC(400->120) -> ReLU -> FC(120->84) -> ReLU -> FC(84->10)
# 11 sequential components in total. structure stays a length-11 wildcard here. The concrete
# literal for the same network is cifar10_tutorial_structure in
# cnn_damg_reference_architectures.py.
cifar10_pytorch_tutorial_target = _make_target(3072, 11, epochs=20)


def make_cifar_experiment_target(epochs: int, length: int = 5):
    """CIFAR-10 NAS search target with a pinned loss, parameterized by epochs.

    Unlike `usps_experiment_target` (a fixed module constant), CIFAR-10 needs the epoch count to be
    adjustable: one candidate evaluation trains on 50k 3x32x32 images instead of 7.3k 16x16 ones, so
    a cheap smoke test must be able to run far fewer epochs than the real experiment. The repository's
    `n_epoch_values` must always contain exactly this value for the target to be synthesizable.

    Loss is pinned to CrossEntropyLoss(mean) for the same reason as in `usps_experiment_target`:
    leaving it a wildcard would let synthesis also pick reduction="sum", whose objective values live
    on a completely different scale and would not be comparable across the evaluations of one run.
    """
    return _make_target(3072, length, epochs=epochs,
                        loss=CNNrepository.CrossEntropyLoss(reduction="mean"))


# The CIFAR-10 counterpart of `usps_experiment_target`: a real search target, with a wildcard
# structure and a per-candidate epoch budget small enough to be practical.
cifar_experiment_target = make_cifar_experiment_target(epochs=50)


# ---------------------------------------------------------------------------
# Constructing search targets with *deliberate* variance.
#
# Guiding principle: constrain a position only where there is a substantive reason to. On the
# output side there is one, since the classifier fixes the interface to the label space and bounds
# the network's size. Inside the feature extractor there is none, and that is precisely what the
# search is for.
#
# Pinning the (in, out) dimensions of every position to a reference architecture's geometry
# collapses the question to "which operator realizes this fixed shape", which is not architecture
# search. None of the position lists below does that.
#
# A position may be described as:
#     None        fully free: any parallel width, any components.
#     k (int)     exactly k parallel components, each otherwise free, so (None,)*k. This pins the
#                 out-degree. It requires the position's in and out features to be decomposable
#                 into k summands from F, so k > 1 only works where the feature set admits the
#                 split.
#     (i, o)      arity 1 with pinned dimensions and a free operator, so ((None, i, o),). Which
#                 operators remain is a consequence of the dimensions: small pairs such as
#                 (120, 10) admit only linear layers, larger ones also admit convolutions. This is
#                 strictly more permissive than naming a concrete Linear label, and it bounds the
#                 layer's size just as effectively.
#
# An (i, o) position depends on partially concrete triples being synthesizable at all, which is why
# ParaTuples.__contains__ in cnn_damg_repo.py delegates to the Para group. The older damg_repo
# yields zero terms for such a position.
# ---------------------------------------------------------------------------

def _variance_structure(positions):
    """Turn a list of position descriptors into a ``structure`` literal.

    Args:
        positions: One descriptor per sequential position (see the module comment above).

    Returns:
        tuple: The structure literal.
    """
    structure: list[Any] = []
    for position in positions:
        if position is None:
            structure.append(None)
        elif isinstance(position, int):
            if position < 1:
                raise ValueError("a pinned out-degree must be at least 1")
            structure.append((None,) * position)
        elif isinstance(position, tuple) and len(position) == 2:
            in_features, out_features = position
            structure.append(((None, in_features, out_features),))
        else:
            raise ValueError(f"unsupported position descriptor: {position!r}")
    return tuple(structure)


def make_variance_target(n_in: int, positions, epochs: int, n_out: int = N_OUT_CLASSES):
    """Build a target from a list of position descriptors (see the module comment above)."""
    return _make_target_from_structure(n_in, _variance_structure(positions), n_out, epochs,
                                        CNNrepository.CrossEntropyLoss(reduction="mean"))


# Free feature extractor, pinned-dimension classifier head. The head both fixes how the network
# finishes and forces the free part to reduce the input down to the head's input width. On
# CIFAR-10, with max_lin_layer_dim set below 3072, no linear layer can do that reduction.
# Six positions, not five: with only three free ones plus the two head positions this target would
# be a strict subspace of the length-5 full-variance target, whose positions are unconstrained and
# can therefore realize any head, which would make the comparison between the two vacuous. The
# extra free position gives the head target a sequential depth the full-variance target does not
# reach.
USPS_HEAD_POSITIONS = (None, None, None, None, (64, 30), (30, 10))
CIFAR_HEAD_POSITIONS = (None, None, None, None, (400, 120), (120, 10))
# The same head as USPS, and deliberately so: 64, 30 and 10 all lie in the MNIST space's
# LINEAR_FEATURE_DIMENSIONS, so the head is expressible there without touching the repository, and
# keeping it identical means the HEAD cells of the two geometries differ in the geometry alone.
# What the head does differently here is that it has more to reduce, 784 flattened inputs against
# USPS's 256, and that is the point of the target rather than a side effect.
MNIST_HEAD_POSITIONS = (None, None, None, None, (64, 30), (30, 10))

usps_head_target = make_variance_target(256, USPS_HEAD_POSITIONS, epochs=50)
cifar_head_target = make_variance_target(3072, CIFAR_HEAD_POSITIONS, epochs=50)
mnist_head_target = make_variance_target(784, MNIST_HEAD_POSITIONS, epochs=50)


def make_cifar_head_target(epochs: int, positions=CIFAR_HEAD_POSITIONS):
    """CIFAR-10 head target, parameterized by epochs (see make_cifar_experiment_target)."""
    return make_variance_target(3072, positions, epochs=epochs)


def make_usps_head_target(epochs: int, positions=USPS_HEAD_POSITIONS):
    """USPS head target, parameterized by epochs (see make_cifar_head_target)."""
    return make_variance_target(256, positions, epochs=epochs)


def make_mnist_head_target(epochs: int, positions=MNIST_HEAD_POSITIONS):
    """MNIST/FashionMNIST head target, parameterized by epochs.

    One function for both datasets, like the space they share: the two cells must differ in the
    task and in nothing else, and a second target beside this one is how that stops being true.
    """
    return make_variance_target(784, positions, epochs=epochs)


# The PyTorch CIFAR-10 tutorial network as a starting point, with variance introduced where it is
# wanted and nowhere else.  Unlike the HEAD target, which pins a classifier head and leaves a free
# feature extractor, this one keeps the reference network's whole dimension chain and frees only
# the operators.
#
# The chain, from cifar10_tutorial_structure in cnn_damg_reference_architectures.py, on
# 3x32x32 = 3072 inputs:
#
#   pos  layer                       out shape      flattened
#   1    Conv(3->6, 5x5)             6 x 28 x 28    4704
#   2    ReLU                        6 x 28 x 28    4704
#   3    MaxPool(2x2)                6 x 14 x 14    1176
#   4    Conv(6->16, 5x5)           16 x 10 x 10    1600
#   5    ReLU                       16 x 10 x 10    1600
#   6    MaxPool(2x2)               16 x  5 x  5     400
#   7    FC(400->120)                                120
#   8    ReLU                                        120
#   9    FC(120->84)                                  84
#   10   ReLU                                         84
#   11   FC(84->10)                                   10
#
# Every one of 4704, 1176, 1600, 400, 120, 84, 10 lies in the linear feature dimensions of
# cifar10_tutorial_repo, so the chain is expressible without touching the repository. That is the
# rule here: steer variance through the query type, never by editing what exists.
#
# Where the variance goes:
#   * positions 7-11, the classifier tail: dimensions pinned, label free.  `(i, o)` becomes
#     `((None, i, o),)`, so synthesis picks the operator and nothing else.
#   * positions 3 and 6, the two pooling stages: likewise pinned in their dimensions.  These are the
#     two places where the spatial reduction happens, and letting them float would change what the
#     later dimensions even mean.
#   * position 1: `(None,)` rather than `None`, so exactly one component, otherwise unconstrained.
#     `None` would admit any parallel width at the very first position, where the input still carries
#     all 3072 features and the branching factor is largest.
#   * positions 2, 4, 5: fully free.  This is where the search may deviate from the reference.
#
# Three variants follow, by how much of the pooling chain stays pinned.  Freeing a pooling stage
# frees the dimension between its neighbors, so each step outwards multiplies what the positions
# around it may become.  Count the rules of a variant before spending training time on it.
CIFAR_TUTORIAL_POSITIONS_1 = (
    1,              # (None,), so one component with a free label
    None,           # free
    (4704, 1176),   # MaxPool stage 1, pinned
    None,           # free
    None,           # free
    (1600, 400),    # MaxPool stage 2, pinned
    (400, 120),
    (120, 120),
    (120, 84),
    (84, 84),
    (84, 10),
)

#: Variant 2: pooling stage 1 released.  Positions 2-3 may then realize any reduction that leaves
#: position 4 able to reach 1600, so the tutorial's 4704 -> 1176 becomes one option among many.
CIFAR_TUTORIAL_POSITIONS_2 = (
    1, None, None, None, None, (1600, 400),
    (400, 120), (120, 120), (120, 84), (84, 84), (84, 10),
)

#: Variant 3: both pooling stages released, and one position removed.  It carries four free
#: positions, as variant 2 does, but over a shorter chain of ten positions rather than eleven, and
#: with no pinned pooling stage at all.  The three variants therefore form a ladder in what is
#: held, not in how much is free: variant 1 pins the most structure, variant 2 loosens it, variant
#: 3 holds the least while offering variance no larger than variant 2's.
#:
#: The distinction matters because variance and structure are separate axes here.  Variant 2, with
#: five free positions, is the expensive one by a wide margin, so a third variant that simply
#: freed more would not have been runnable either.
CIFAR_TUTORIAL_POSITIONS_3 = (
    1, None, None, None, None,
    (400, 120), (120, 120), (120, 84), (84, 84), (84, 10),
)

CIFAR_TUTORIAL_VARIANTS = {
    1: CIFAR_TUTORIAL_POSITIONS_1,
    2: CIFAR_TUTORIAL_POSITIONS_2,
    3: CIFAR_TUTORIAL_POSITIONS_3,
}

cifar_tutorial_variance_target_1 = make_variance_target(3072, CIFAR_TUTORIAL_POSITIONS_1, epochs=50)
cifar_tutorial_variance_target_2 = make_variance_target(3072, CIFAR_TUTORIAL_POSITIONS_2, epochs=50)
cifar_tutorial_variance_target_3 = make_variance_target(3072, CIFAR_TUTORIAL_POSITIONS_3, epochs=50)


# ---------------------------------------------------------------------------
# VGG-11 with BatchNorm, the reference architecture of the training recipe.
#
# The 30 positions of ``cifar10_vgg11_bn_structure()``, each written as its ``(in, out)`` pair, so
# dimensions pinned and operator free. This is the base to edit, not a target to run: with every
# position pinned the question collapses to "which operator realizes this fixed shape", which is not
# architecture search (see the module comment above). Replace the positions that should carry the
# variance with ``None`` (fully free) or an integer (that many parallel components).
#
#   #   operator                     in ->    out
#   0   conv3x3 3->64   @32x32     3072 ->  65536
#   1   bn 64           @32x32    65536 ->  65536
#   2   relu                      65536 ->  65536
#   3   pool2x2 64      @32->16   65536 ->  16384
#   4   conv3x3 64->128 @16x16    16384 ->  32768
#   5   bn 128          @16x16    32768 ->  32768
#   6   relu                      32768 ->  32768
#   7   pool2x2 128     @16->8    32768 ->   8192
#   8   conv3x3 128->256 @8x8      8192 ->  16384
#   9   bn 256           @8x8     16384 ->  16384
#   10  relu                      16384 ->  16384
#   11  conv3x3 256->256 @8x8     16384 ->  16384
#   12  bn 256           @8x8     16384 ->  16384
#   13  relu                      16384 ->  16384
#   14  pool2x2 256      @8->4    16384 ->   4096
#   15  conv3x3 256->512 @4x4      4096 ->   8192
#   16  bn 512           @4x4      8192 ->   8192
#   17  relu                       8192 ->   8192
#   18  conv3x3 512->512 @4x4      8192 ->   8192
#   19  bn 512           @4x4      8192 ->   8192
#   20  relu                       8192 ->   8192
#   21  pool2x2 512      @4->2     8192 ->   2048
#   22  conv3x3 512->512 @2x2      2048 ->   2048
#   23  bn 512           @2x2      2048 ->   2048
#   24  relu                       2048 ->   2048
#   25  conv3x3 512->512 @2x2      2048 ->   2048
#   26  bn 512           @2x2      2048 ->   2048
#   27  relu                       2048 ->   2048
#   28  pool2x2 512      @2->1     2048 ->    512
#   29  linear 512->10              512 ->     10
#
# Freeing a position is not cheap, and the cost is not linear. On the 11-position tutorial chain,
# going from three free positions to five multiplies the rule count by roughly two orders of
# magnitude. This chain has 30 positions. Count the rules of a variant before spending training
# time on it, rather than assuming it is runnable.
#
# The five pooling positions (3, 7, 14, 21, 28) are the ones that set the spatial reduction: freeing
# one frees the dimension between its neighbors, which changes what every later position means.
VGG11_BN_POSITIONS = (
    (3072, 65536), (65536, 65536), (65536, 65536), (65536, 16384),      # 0-3   block 1 + pool
    (16384, 32768), (32768, 32768), (32768, 32768), (32768, 8192),      # 4-7   block 2 + pool
    (8192, 16384), (16384, 16384), (16384, 16384),                      # 8-10  block 3
    (16384, 16384), (16384, 16384), (16384, 16384), (16384, 4096),      # 11-14 block 4 + pool
    (4096, 8192), (8192, 8192), (8192, 8192),                           # 15-17 block 5
    (8192, 8192), (8192, 8192), (8192, 8192), (8192, 2048),             # 18-21 block 6 + pool
    (2048, 2048), (2048, 2048), (2048, 2048),                           # 22-24 block 7
    (2048, 2048), (2048, 2048), (2048, 2048), (2048, 512),              # 25-28 block 8 + pool
    (512, 10),                                                          # 29    classifier
)


#: The same chain with every run of consecutive positions of equal (in, out) merged into one
#: position: 30 positions become 15. What stays separate are the nine positions that change the
#: dimension, the five poolings and the four convolutions that raise the channel count.
#:
#: VGG-11 itself is not an element of any target over these 15 positions. A descriptor is one
#: sequential stage, so position 12 admits a single operator where VGG-11 has six. The published
#: accuracy of VGG-11-BN is therefore an external comparison mark for such a target, not the score
#: of one of its members.
VGG11_BN_MERGED_POSITIONS = (
    (3072, 65536),      #  0  conv 3->64
    (65536, 65536),     #  1  2x  bn, relu
    (65536, 16384),     #  2  pool
    (16384, 32768),     #  3  conv 64->128
    (32768, 32768),     #  4  2x  bn, relu
    (32768, 8192),      #  5  pool
    (8192, 16384),      #  6  conv 128->256
    (16384, 16384),     #  7  5x  bn, relu, conv, bn, relu
    (16384, 4096),      #  8  pool
    (4096, 8192),       #  9  conv 256->512
    (8192, 8192),       # 10  5x  bn, relu, conv, bn, relu
    (8192, 2048),       # 11  pool
    (2048, 2048),       # 12  6x  conv, bn, relu, conv, bn, relu
    (2048, 512),        # 13  pool
    (512, 10),          # 14  linear 512->10
)

#: Five of the fifteen merged positions freed: the first one and the four that repeat a shape.
#: Position 1 stays pinned because it is where the features are widest at 65536, which is the most
#: expensive place to open.
#:
#: At equal freedom this chain is far cheaper than the tutorial chain, because the configuration of
#: ``cifar10_vgg11_bn_repo`` is tighter: one kernel size instead of three, one stride, one padding,
#: and ``max_lin_layer_dim=512``, which leaves only the linear labels over features up to 512.
VGG11_BN_VARIANCE_POSITIONS = tuple(
    None if index in (0, 4, 7, 10, 12) else pair
    for index, pair in enumerate(VGG11_BN_MERGED_POSITIONS)
)


#: The NAS-style chain over all 30 positions: the reduction skeleton of five poolings and the
#: classifier stay pinned, and every operation between them is free. That split is
#: NAS-Bench-201's, whose macro skeleton is fixed while its cells are searched. It is also the
#: reason VGG-11 is an element of this space, unlike the merged chain above: all 30 positions are
#: still there, a pinned position admits any operator of those dimensions, and a free one any
#: operator at all.
#:
#: This is the most expensive of the three VGG chains by a wide margin. Twenty-four free positions
#: put it in the cost class of the five-free tutorial variant, so count its rules before spending
#: training time on it.
VGG11_BN_NAS_POSITIONS = tuple(
    pair if index in (3, 7, 14, 21, 28, 29) else None
    for index, pair in enumerate(VGG11_BN_POSITIONS)
)


def make_vgg11_bn_target(positions=VGG11_BN_POSITIONS, epochs: int = 50, optimizer=None,
                         scheduler=None):
    """A CIFAR-10 target over the VGG-11-BN chain, with the variance stated by ``positions``.

    Args:
        positions: One descriptor per sequential position, as in ``make_variance_target``:
            ``None`` free, an int for that many parallel components, an ``(in, out)`` pair for
            pinned dimensions with a free operator. (Default value = VGG11_BN_POSITIONS)
        epochs (int): Epochs per candidate. The repository must offer this value, since a target
            whose epoch count is not in ``n_epoch_values`` is uninhabited and says nothing.
            (Default value = 50)
        optimizer: A concrete ``CNNrepository.SGD``/``Adam`` to pin the optimizer, or None to leave
            the slot a wildcard. (Default value = None)
        scheduler: A concrete ``CNNrepository.CosineAnnealingLR``/``NoScheduler`` to pin the
            schedule, or None to leave it a wildcard. (Default value = None)

    Returns:
        The target type.

    Note:
        The default of both is None, and a None slot is a wildcard and not a default value. With
        the optimizer and the schedule left open, the space carries one term per recipe instead of
        one term per network. The repository offers two optimizers and two schedules, so on the
        pinned VGG-11-BN chain four terms describe the same network, and three of them train it
        with a recipe that was not asked for. Pinning both takes those terms out of the space and
        leaves every network in it, which makes it a decision about what is being searched for and
        not an optimization: a caller looking for an architecture under a fixed recipe passes
        both, and a caller looking for an architecture together with its recipe leaves them open.
    """
    # Built in one Constructor("Learner", ...) rather than by intersecting two of them: the target is
    # matched against the learner's suffix, and `C(A) & C(B)` is not the same query as `C(A & B)`.
    structure = _variance_structure(positions)
    return Constructor("Learner", Constructor("DAG",
                                              Constructor("input", Literal(3072))
                                              & Constructor("output", Literal(N_OUT_CLASSES))
                                              & Constructor("structure", Literal(structure)))
                       & Constructor("Loss", Constructor(
                           "type", Literal(CNNrepository.CrossEntropyLoss(reduction="mean"))))
                       & Constructor("Optimizer", Constructor("type", Literal(optimizer)))
                       & Constructor("Scheduler", Constructor("type", Literal(scheduler)))
                       & Constructor("epochs", Literal(epochs)))


def make_cifar_tutorial_variance_target(epochs: int, variant: int = 1):
    """CIFAR-10 target built on the tutorial network's dimension chain, parameterized by epochs.

    Args:
        epochs (int): The per-candidate epoch budget, pinned into the target like everywhere else.
        variant (int): Which of :data:`CIFAR_TUTORIAL_VARIANTS` to build. 1 pins both pooling
            stages, 2 releases the first, 3 releases both. (Default value = 1)

    Returns:
        The target.

    Raises:
        ValueError: If there is no such variant.  Falling back to a default would make the run
            record say one thing and the search do another.
    """
    if variant not in CIFAR_TUTORIAL_VARIANTS:
        msg = (f"no tutorial variance variant {variant!r}; "
               f"the ones a run may ask for are {sorted(CIFAR_TUTORIAL_VARIANTS)}")
        raise ValueError(msg)
    return make_variance_target(3072, CIFAR_TUTORIAL_VARIANTS[variant], epochs=epochs)


_NAMED_TARGETS = {
    "cifar_tutorial_variance_target_1": cifar_tutorial_variance_target_1,
    "cifar_tutorial_variance_target_2": cifar_tutorial_variance_target_2,
    "cifar_tutorial_variance_target_3": cifar_tutorial_variance_target_3,
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
    "cifar_experiment_target": cifar_experiment_target,
    "usps_head_target": usps_head_target,
    "cifar_head_target": cifar_head_target,
    "mnist_head_target": mnist_head_target,
}


def target_to_name(target):
    for name, value in _NAMED_TARGETS.items():
        if value == target:
            return name
    return "unknown"
