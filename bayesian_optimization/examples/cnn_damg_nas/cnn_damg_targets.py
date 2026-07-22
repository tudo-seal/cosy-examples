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


def _make_target_from_structure(n_in: int, structure, n_out: int = N_OUT_CLASSES,
                                 epochs: int = 2000, loss=None):
    # `loss=None` keeps the Loss slot a wildcard (Literal(None)) - synthesis is then free to pick
    # any cross_entropy_loss variant (any reduction). Passing a concrete loss instance (e.g.
    # CNNrepository.CrossEntropyLoss(reduction="mean")) instead pins the exact loss for this
    # target - the query language's own mechanism for eliminating variance at a given slot, no
    # repository/combinator change needed (the same idea already used for `structure` literals).
    #
    # `structure` is a tuple with one entry per sequential position. Forms verified to synthesize:
    #   None                  - any parallel width, any components (full variance)
    #   (None,)               - exactly one component, otherwise unconstrained (arity control)
    #   ((label, i, o),)      - exactly one fully concrete component
    #
    # Partially concrete triples such as `(None, i, o)` (dimensions pinned, operator free) currently
    # yield 0 terms, even though the leaf combinators explicitly declare them as the para1..para7
    # structure variants. This is a defect in the repositories, not a property of the query language:
    # `ParaTuples.__init__` does `self.para = tuple(para)`, snapshotting the Para group through its
    # __iter__ (which never yields a triple containing None), and `ParaTuples.__contains__` then does
    # a plain tuple membership test `v in self.para` instead of delegating to `Para.__contains__`
    # (which does accept those triples). The target's structure literal is inferred, not enumerated,
    # so it is validated exclusively through __contains__ - and gets rejected there, in
    # Synthesizer._enumerate_substitutions (cosy/core/synthesizer.py:183), on the `learner`
    # combinator's `request` parameter. Same defect in damg_repo.py (:380/:399) and
    # cnn_damg_repo.py (:577/:596).
    #
    # Note that the literal value in Constructor("epochs", Literal(...)) is currently not allowed to be None!
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


# Stufe 1: USPS (1x16x16, N_in=256)
usps_target_len_2 = _make_target(256, 2)
usps_target_len_3 = _make_target(256, 3)
usps_target_len_4 = _make_target(256, 4)

# First real NAS search experiment (not a single pinned architecture): full-variance length-5
# target, few epochs per candidate (50, not 2000) to keep each BO evaluation fast during a small
# validation run (see cnn_damg_usps_experiment.py). Loss is pinned to CrossEntropyLoss(mean) -
# not a wildcard - so every evaluated structure's objective value is on the same scale and
# directly comparable (leaving it a wildcard would let synthesis also pick reduction="sum",
# whose values are on a completely different scale and not comparable to "mean" ones).
usps_experiment_target = _make_target(256, 5, epochs=50, loss=CNNrepository.CrossEntropyLoss(reduction="mean"))

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


def make_cifar_experiment_target(epochs: int, length: int = 5):
    """CIFAR-10 NAS search target with a pinned loss, parameterized by epochs.

    Unlike `usps_experiment_target` (a fixed module constant), CIFAR-10 needs the epoch count to be
    adjustable: one candidate evaluation trains on 50k 3x32x32 images instead of 7.3k 16x16 ones, so
    a cheap smoke test must be able to run far fewer epochs than the real experiment. The repository's
    `n_epoch_values` must always contain exactly this value for the target to be synthesizable.

    Loss is pinned to CrossEntropyLoss(mean) for the same reason as in `usps_experiment_target`:
    leaving it a wildcard would let synthesis also pick reduction="sum", whose objective values live
    on a completely different scale and would not be comparable in the results CSV.
    """
    return _make_target(3072, length, epochs=epochs,
                        loss=CNNrepository.CrossEntropyLoss(reduction="mean"))


# The CIFAR-10 counterpart of `usps_experiment_target`: a genuine (wildcard-structure) search target
# for the real experiment on the A30, with a per-candidate epoch budget small enough to be practical.
cifar_experiment_target = make_cifar_experiment_target(epochs=50)


# ---------------------------------------------------------------------------
# Constructing search targets with *deliberate* variance.
#
# Guiding principle: constrain a position only where there is a substantive reason to. On the output
# side there is one - the classifier fixes the interface to the label space and bounds the network's
# size. Inside the feature extractor there is none: that is precisely what the search is for.
#
# (An earlier attempt pinned the (in, out) dimensions of EVERY position to a reference architecture's
# geometry. That collapses the question to "which operator realizes this fixed shape", which is not
# architecture search - do not reintroduce it.)
#
# A position may be described as:
#     None        - fully free: any parallel width, any components
#     k (int)     - exactly k parallel components, each otherwise free -> (None,)*k, i.e. pinned
#                   out-degree. Requires the position's in/out features to be decomposable into k
#                   summands from F, so k > 1 only works where the feature set admits the split.
#     (i, o)      - arity 1 with pinned dimensions, operator free -> ((None, i, o),). Which operators
#                   remain is then a consequence of the dimensions: small pairs such as (120, 10)
#                   admit only linear layers, larger ones also admit convolutions. This is the "lin"
#                   role - strictly more permissive than naming a concrete Linear label, while
#                   bounding the layer's size just as effectively.
#
# Note that (i, o) positions depend on partially concrete triples being synthesizable at all, i.e.
# on ParaTuples.__contains__ delegating to the Para group (see cnn_damg_repo.py) - the legacy
# damg_repo silently yields zero terms for them.
# ---------------------------------------------------------------------------

def make_variance_target(n_in: int, positions, epochs: int, n_out: int = N_OUT_CLASSES):
    """Build a target from a list of position descriptors (see the module comment above)."""
    structure = []
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
    return _make_target_from_structure(n_in, tuple(structure), n_out, epochs,
                                        CNNrepository.CrossEntropyLoss(reduction="mean"))


# Free feature extractor, pinned-dimension classifier head. The head both fixes how the network
# finishes and forces the free part to reduce the input down to the head's input width - which on
# CIFAR-10, with max_lin_layer_dim set below 3072, no linear layer can do.
# Six positions, not five: with only three free ones plus the two head positions this target would
# be a strict SUB-space of the length-5 full-variance target (whose positions are unconstrained and
# can therefore realize any head), making the comparison between the two vacuous. The extra free
# position gives the head target a sequential depth the full-variance target does not reach.
USPS_HEAD_POSITIONS = (None, None, None, None, (64, 30), (30, 10))
CIFAR_HEAD_POSITIONS = (None, None, None, None, (400, 120), (120, 10))

usps_head_target = make_variance_target(256, USPS_HEAD_POSITIONS, epochs=50)
cifar_head_target = make_variance_target(3072, CIFAR_HEAD_POSITIONS, epochs=50)


def make_cifar_head_target(epochs: int, positions=CIFAR_HEAD_POSITIONS):
    """CIFAR-10 head target, parameterized by epochs (see make_cifar_experiment_target)."""
    return make_variance_target(3072, positions, epochs=epochs)


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
    "cifar_experiment_target": cifar_experiment_target,
    "usps_head_target": usps_head_target,
    "cifar_head_target": cifar_head_target,
}


def target_to_name(target):
    for name, value in _NAMED_TARGETS.items():
        if value == target:
            return name
    return "unknown"
