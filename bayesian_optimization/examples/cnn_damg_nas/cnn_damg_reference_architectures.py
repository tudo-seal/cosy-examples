"""Concrete (wildcard-free) reference architectures, shared between the test suite
(tests/integration/test_cnn_damg_nas.py) and the CPU/A30 benchmark script (cnn_damg_benchmark.py) so
both exercise the exact same synthesized term.

Each "structure" here is a fully concrete ``structure`` literal (no ``None`` wildcards): synthesis has
(near-)nothing left to search for and produces the intended term directly and quickly, instead of
enumerating the search space and filtering for a match.
"""

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository


# ---------------------------------------------------------------------------
# USPS (16x16x1, N_in=256): LeCun et al. 1989, "Backpropagation Applied to Handwritten Zip Code
# Recognition" (Neural Computation 1, 541-551) - to our knowledge the smallest published CNN that
# reasonably solves this exact task (the same USPS zip-code digits). Reported 5.00% test error with
# 9760 parameters, using Conv(1->12, 5x5, stride 2, padding 2) -> Conv(12->12, 5x5, stride 2,
# padding 2) -> FC(192->30) -> FC(30->10) with tanh activations. The original H1->H2 connectivity
# was sparse; our conv2d combinator only supports dense convolutions, so this echo has slightly more
# parameters (10024), but the per-layer feature-map shapes match exactly.
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


# ---------------------------------------------------------------------------
# CIFAR-10 (32x32x3, N_in=3072): the official PyTorch CIFAR-10 tutorial network. Chosen over e.g.
# Caffe's cifar10_quick (75.33% test accuracy) because its pooling already satisfies our
# stride == kernel_size restriction on MaxPool2d - see the "Future Extensions" note in the CNN
# refactoring plan for what relaxing that restriction (plus an AvgPool2d combinator) would take.
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


def cifar10_tutorial_structure():
    conv1 = CNNrepository.Conv2d(in_channels=3, out_channels=6, input_size=(32, 32), output_size=(28, 28),
                                  kernel_size=(5, 5), stride=1, padding=0, bias=True)
    conv2 = CNNrepository.Conv2d(in_channels=6, out_channels=16, input_size=(14, 14), output_size=(10, 10),
                                  kernel_size=(5, 5), stride=1, padding=0, bias=True)
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
