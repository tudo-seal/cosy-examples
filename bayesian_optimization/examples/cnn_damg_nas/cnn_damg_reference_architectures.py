"""Reference architectures with no wildcard left in them.

Each ``_repo`` function returns the repository configuration one published network needs, and the
matching ``_structure`` function returns that network as a fully concrete ``structure`` literal. A
concrete literal leaves synthesis next to nothing to search for, so it produces the intended term
directly instead of enumerating the space and filtering for a match. That is what makes these
architectures usable as fixed points of comparison: every caller that asks for one gets the same
synthesized term.
"""

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository

# ---------------------------------------------------------------------------
# USPS (16x16x1, N_in=256): LeCun et al. 1989, "Backpropagation Applied to Handwritten Zip Code
# Recognition" (Neural Computation 1, 541-551). To our knowledge it is the smallest published CNN
# that reasonably solves this exact task, the same USPS zip-code digits. It reports 5.00% test
# error with 9760 parameters, using Conv(1->12, 5x5, stride 2, padding 2) -> Conv(12->12, 5x5,
# stride 2, padding 2) -> FC(192->30) -> FC(30->10) with tanh activations. The original H1->H2
# connectivity was sparse. The conv2d combinator here only supports dense convolutions, so this
# echo has slightly more parameters (10000), while the per-layer feature-map shapes match.
# ---------------------------------------------------------------------------

def usps_lecun1989_repo() -> CNNrepository:
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


def usps_lecun1989_structure():
    # bias=False on both convolutions is forced, not chosen. CNNrepository.Label.iter_conv2d yields
    # biasless convolutions only, because a BatchNorm placed right after a convolution cancels that
    # convolution's bias exactly and leaves it a dead search dimension. LeCun's network had biases,
    # so this echo is one step further from the original than it would otherwise be. The layer
    # shapes and the parameter count per layer are unaffected except for the 12 + 12 bias terms.
    conv1 = CNNrepository.Conv2d(in_channels=1, out_channels=12, input_size=(16, 16), output_size=(8, 8),
                                  kernel_size=(5, 5), stride=2, padding=2, bias=False)
    conv2 = CNNrepository.Conv2d(in_channels=12, out_channels=12, input_size=(8, 8), output_size=(4, 4),
                                  kernel_size=(5, 5), stride=2, padding=2, bias=False)
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


# ---------------------------------------------------------------------------
# CIFAR-10 (32x32x3, N_in=3072): the official PyTorch CIFAR-10 tutorial network. It is preferred
# over Caffe's cifar10_quick (75.33% test accuracy) because its pooling already satisfies the
# stride == kernel_size restriction that CNNrepository.Label.iter_maxpool2d imposes. Relaxing that
# restriction would take free stride and padding parameters there and an AvgPool2d combinator
# beside the pooling one.
#   Conv(3->6, 5x5) -> ReLU -> MaxPool(2x2) -> Conv(6->16, 5x5) -> ReLU -> MaxPool(2x2) ->
#   FC(400->120) -> ReLU -> FC(120->84) -> ReLU -> FC(84->10)
# ---------------------------------------------------------------------------

def cifar10_tutorial_repo() -> CNNrepository:
    return CNNrepository(
        linear_feature_dimensions=[3072, 4704, 1176, 1600, 400, 120, 84, 10],
        constant_values=[0, 1, -1],
        learning_rate_values=[1e-3],
        n_epoch_values=[20],
        channel_dimensions=[3, 6, 16],
        height_width_dimensions=[(32, 32), (28, 28), (14, 14), (10, 10)],
        kernel_dimensions=[(5, 5), (2, 2)],
        stride_values=[1],
        padding_values=[0],
        max_parallel_width=2,
    )


# ---------------------------------------------------------------------------
# CIFAR-10, VGG-11 with BatchNorm (Simonyan and Zisserman, ICLR 2015, configuration "A"). The
# BatchNorm variant is torchvision's ``vgg11_bn``. It is the reference for the training recipe
# below because it is the well-known architecture that is purely sequential. The combinator set of
# CNNrepository has no additive skip and no global average pooling, so NAS-Bench-201 and DARTS, the
# two papers the recipe itself is taken from, are not buildable here.
#
# This is the CIFAR adaptation, not the ImageNet original: the classifier is a single
# Linear(512, 10) rather than the 4096-4096-1000 stack, and there is no Dropout, which is not a
# combinator here. That adaptation is the one the commonly cited 92% CIFAR-10 accuracy belongs
# to.
#
# The features, layer by layer on a 32x32 input:
#   conv3x3(3->64)    64x32x32 = 65536   bn relu
#   pool2x2           64x16x16 = 16384
#   conv3x3(64->128) 128x16x16 = 32768   bn relu
#   pool2x2          128x 8x 8 =  8192
#   conv3x3(128->256)256x 8x 8 = 16384   bn relu
#   conv3x3(256->256)256x 8x 8 = 16384   bn relu
#   pool2x2          256x 4x 4 =  4096
#   conv3x3(256->512)512x 4x 4 =  8192   bn relu
#   conv3x3(512->512)512x 4x 4 =  8192   bn relu
#   pool2x2          512x 2x 2 =  2048
#   conv3x3(512->512)512x 2x 2 =  2048   bn relu
#   conv3x3(512->512)512x 2x 2 =  2048   bn relu
#   pool2x2          512x 1x 1 =   512
#   linear(512 -> 10)
#
# ``max_lin_layer_dim=512`` is not decoration. The feature list contains 65536, and a single
# Linear from that feature to itself would carry 65536 * 65536 weights, which dwarfs every
# convolution in the space.
# ---------------------------------------------------------------------------

def cifar10_vgg11_bn_repo(epochs: int = 50) -> CNNrepository:
    # `epochs` is a parameter because a target whose epoch count the repository does not offer is
    # simply uninhabited. Synthesis then returns nothing and says nothing. A cheap smoke run at 1 or
    # 2 epochs therefore has to build its own repository, which is what this argument is for.
    return CNNrepository(
        linear_feature_dimensions=[3072, 65536, 32768, 16384, 8192, 4096, 2048, 512, 10],
        constant_values=[0, 1, -1],
        learning_rate_values=[0.1],          # the SGD rate of the VGG/DARTS/NB201 recipe
        n_epoch_values=[epochs],
        channel_dimensions=[3, 64, 128, 256, 512],
        height_width_dimensions=[(32, 32), (16, 16), (8, 8), (4, 4), (2, 2), (1, 1)],
        kernel_dimensions=[(3, 3)],
        stride_values=[1],
        padding_values=[1],
        pooling_kernel_dimensions=[(2, 2)],
        max_parallel_width=2,
        max_lin_layer_dim=512,
        weight_decay_values=[5e-4],          # DARTS 3e-4, NB201 5e-4, VGG 5e-4
        momentum_values=[0.9],
    )


def cifar10_vgg11_bn_structure():
    def conv(in_c, out_c, size):
        return CNNrepository.Conv2d(in_channels=in_c, out_channels=out_c, input_size=size,
                                    output_size=size, kernel_size=(3, 3), stride=1, padding=1,
                                    bias=False)

    def norm(channels, size):
        return CNNrepository.BatchNorm2d(in_channels=channels, input_size=size)

    def pool(channels, size):
        return CNNrepository.MaxPool2d(in_channels=channels, input_size=size,
                                       output_size=(size[0] // 2, size[1] // 2), kernel_size=(2, 2))

    relu = CNNrepository.ReLu(inplace=False)
    classifier = CNNrepository.Linear(in_features=512, out_features=10, bias=True)

    # (in_channels, out_channels, spatial size, whether a pooling follows this block)
    blocks = [
        (3, 64, (32, 32), True),
        (64, 128, (16, 16), True),
        (128, 256, (8, 8), False),
        (256, 256, (8, 8), True),
        (256, 512, (4, 4), False),
        (512, 512, (4, 4), True),
        (512, 512, (2, 2), False),
        (512, 512, (2, 2), True),
    ]

    structure = []
    for in_c, out_c, size, pools in blocks:
        feature_in = in_c * size[0] * size[1]
        feature_out = out_c * size[0] * size[1]
        structure.append(((conv(in_c, out_c, size), feature_in, feature_out),))
        structure.append(((norm(out_c, size), feature_out, feature_out),))
        structure.append(((relu, feature_out, feature_out),))
        if pools:
            halved = out_c * (size[0] // 2) * (size[1] // 2)
            structure.append(((pool(out_c, size), feature_out, halved),))
    structure.append(((classifier, 512, 10),))
    return tuple(structure)


def cifar10_tutorial_structure():
    # bias=False for the same reason as in usps_lecun1989_structure above.
    conv1 = CNNrepository.Conv2d(in_channels=3, out_channels=6, input_size=(32, 32), output_size=(28, 28),
                                  kernel_size=(5, 5), stride=1, padding=0, bias=False)
    conv2 = CNNrepository.Conv2d(in_channels=6, out_channels=16, input_size=(14, 14), output_size=(10, 10),
                                  kernel_size=(5, 5), stride=1, padding=0, bias=False)
    pool1 = CNNrepository.MaxPool2d(in_channels=6, input_size=(28, 28), output_size=(14, 14), kernel_size=(2, 2))
    pool2 = CNNrepository.MaxPool2d(in_channels=16, input_size=(10, 10), output_size=(5, 5), kernel_size=(2, 2))
    linear1 = CNNrepository.Linear(in_features=400, out_features=120, bias=True)
    linear2 = CNNrepository.Linear(in_features=120, out_features=84, bias=True)
    linear3 = CNNrepository.Linear(in_features=84, out_features=10, bias=True)
    relu = CNNrepository.ReLu(inplace=False)
    return (
        ((conv1, 3072, 4704),),
        ((relu, 4704, 4704),),
        ((pool1, 4704, 1176),),
        ((conv2, 1176, 1600),),
        ((relu, 1600, 1600),),
        ((pool2, 1600, 400),),
        ((linear1, 400, 120),),
        ((relu, 120, 120),),
        ((linear2, 120, 84),),
        ((relu, 84, 84),),
        ((linear3, 84, 10),),
    )
