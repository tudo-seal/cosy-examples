"""The network a term denotes: one module per combinator, and the algebras that assemble them.

Each class below is what one combinator of the CNN repository stands for as a ``torch.nn.Module``,
so that composing them along a term gives the network the term encodes.  ``Tree.interpret`` reads
the three algebras at the end of this module, which differ only in what they do once the network
stands: ``pytorch_function_algebra`` returns a callable that trains it, ``pytorch_model_algebra``
returns the untrained module itself, and ``pytorch_components_algebra`` returns the parts a caller
needs in order to train it and keep it.  ``learner`` is the training loop all three name.

Every module here reads and writes a flat feature vector.  Conv2d, MaxPool2d and BatchNorm2d fold
the view into channels, height and width in and back out themselves, so the type system of the
repository sees the flattened width and nothing else.

The readings of a term that build no network live in ``cnn_damg_term_algebras``, which imports no
torch.  Nothing here imports from there, and nothing there imports from here.
"""

from dataclasses import dataclass

import torch
from torch import nn, optim


class EdgesModule(nn.Module):
    # nn.Identity? our forward is tuple of tensors -> tuple of tensors, so maybe not...
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def _as_tensor_input(x):
    # Accept plain tensors as well as simple tensor sequences produced by composed modules.
    if isinstance(x, (tuple, list)):
        if len(x) == 0:
            raise ValueError("Expected a non-empty tensor sequence.")
        if all(isinstance(part, torch.Tensor) for part in x):
            return torch.cat(tuple(x), dim=-1)
    return x


def _feature_dim(x) -> int:
    # Centralized feature-dimension lookup for modules that operate on the last axis.
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(x)!r}.")
    if x.ndim == 0:
        raise ValueError("Expected a tensor with at least one dimension.")
    return x.shape[-1]


class SwapModule(nn.Module):
    def __init__(self, n: int, m: int):
        super().__init__()
        self.n = n
        self.m = m

    def forward(self, x):
        x = _as_tensor_input(x)
        feature_dim = _feature_dim(x)
        if feature_dim != self.n + self.m:
            raise ValueError(
                f"Swap Module expected input dimensions ({self.n}, {self.m}), but got feature dimension {feature_dim}."
            )
        x1, x2 = torch.split(x, [self.n, self.m], dim=-1)
        return torch.cat((x2, x1), dim=-1)


class SynthLinear(nn.Module):
    def __init__(self, output_dim, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)
        self.o = output_dim
        self.in_features = in_features

    def forward(self, x):
        x = _as_tensor_input(x)
        if _feature_dim(x) != self.in_features:
            raise ValueError(f"Linear Layer expected {self.in_features} inputs, but got feature dimension {_feature_dim(x)}")
        y = self.linear(x)
        if _feature_dim(y) != self.o:
            raise ValueError(f"Linear Layer expected to produce {self.o} outputs, but got feature dimension {_feature_dim(y)}")
        return y


class SynthConv2d(nn.Module):
    # Conv2d is treated like Linear: it operates on a flat feature vector of
    # in_channels * in_h * in_w.  The view into an image before the convolution and the reshape
    # back after it are folded in here, so the type system of the repository never sees channels,
    # height and width, only the flattened scalar feature sizes.
    def __init__(self, in_channels, in_h, in_w, out_channels, out_h, out_w, kernel_size, stride, padding, bias):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=1, bias=bias)
        self.in_channels, self.in_h, self.in_w = in_channels, in_h, in_w
        self.out_channels, self.out_h, self.out_w = out_channels, out_h, out_w
        self.i = in_channels * in_h * in_w
        self.o = out_channels * out_h * out_w

    def forward(self, x):
        x = _as_tensor_input(x)
        if _feature_dim(x) != self.i:
            raise ValueError(f"Conv2d Layer expected {self.i} inputs, but got feature dimension {_feature_dim(x)}")
        batch_shape = x.shape[:-1]
        y = self.conv(x.reshape(-1, self.in_channels, self.in_h, self.in_w))
        y = y.reshape(*batch_shape, self.o)
        if _feature_dim(y) != self.o:
            raise ValueError(f"Conv2d Layer expected to produce {self.o} outputs, but got feature dimension {_feature_dim(y)}")
        return y


class SynthMaxPool2d(nn.Module):
    # MaxPool2d restricted to the usual case, stride equal to the kernel size and no padding.
    # It has no learned parameters.
    def __init__(self, in_channels, in_h, in_w, out_h, out_w, kernel_size):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size)
        self.in_channels, self.in_h, self.in_w = in_channels, in_h, in_w
        self.out_h, self.out_w = out_h, out_w
        self.i = in_channels * in_h * in_w
        self.o = in_channels * out_h * out_w

    def forward(self, x):
        x = _as_tensor_input(x)
        if _feature_dim(x) != self.i:
            raise ValueError(f"MaxPool2d Layer expected {self.i} inputs, but got feature dimension {_feature_dim(x)}")
        batch_shape = x.shape[:-1]
        y = self.pool(x.reshape(-1, self.in_channels, self.in_h, self.in_w))
        y = y.reshape(*batch_shape, self.o)
        if _feature_dim(y) != self.o:
            raise ValueError(f"MaxPool2d Layer expected to produce {self.o} outputs, but got feature dimension {_feature_dim(y)}")
        return y


class SynthBatchNorm2d(nn.Module):
    # Shape-preserving, and folded the way SynthConv2d and SynthMaxPool2d are.  The flat feature
    # vector is viewed as (C, H, W) here and reshaped back afterwards, so the type system never
    # sees the channel split.  The split matters all the same, because BatchNorm normalizes per
    # channel, and 4x8x8 and 16x4x4 are the same 256 features under two different normalizations.
    #
    # eps, momentum and affine keep torch's defaults.  The BatchNorm2d label of the repository
    # declares neither of the three, so none of them is a search dimension.
    def __init__(self, in_channels, in_h, in_w):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.in_channels, self.in_h, self.in_w = in_channels, in_h, in_w
        self.o = in_channels * in_h * in_w

    def forward(self, x):
        x = _as_tensor_input(x)
        if _feature_dim(x) != self.o:
            raise ValueError(f"BatchNorm2d expected {self.o} inputs, but got feature dimension {_feature_dim(x)}")
        batch_shape = x.shape[:-1]
        y = self.bn(x.reshape(-1, self.in_channels, self.in_h, self.in_w))
        y = y.reshape(*batch_shape, self.o)
        if _feature_dim(y) != self.o:
            raise ValueError(f"BatchNorm2d expected to produce {self.o} outputs, but got feature dimension {_feature_dim(y)}")
        return y


class SynthSigmoid(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.o = output_dim

    def forward(self, x):
        x = _as_tensor_input(x)
        if _feature_dim(x) != self.o:
            raise ValueError(f"Sigmoid expected {self.o} inputs, but got feature dimension {_feature_dim(x)}")
        y = self.sigmoid(x)
        if _feature_dim(y) != self.o:
            raise ValueError(f"Sigmoid expected to produce {self.o} outputs, but got feature dimension {_feature_dim(y)}")
        return y


class SynthReLU(nn.Module):
    def __init__(self, output_dim, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.o = output_dim

    def forward(self, x):
        x = _as_tensor_input(x)
        if _feature_dim(x) != self.o:
            raise ValueError(f"ReLU expected {self.o} inputs, but got feature dimension {_feature_dim(x)}")
        y = self.relu(x)
        if _feature_dim(y) != self.o:
            raise ValueError(f"ReLU expected to produce {self.o} outputs, but got feature dimension {_feature_dim(y)}")
        return y


class SynthTanh(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.tanh = nn.Tanh()
        self.o = output_dim

    def forward(self, x):
        x = _as_tensor_input(x)
        if _feature_dim(x) != self.o:
            raise ValueError(f"Tanh expected {self.o} inputs, but got feature dimension {_feature_dim(x)}")
        y = self.tanh(x)
        if _feature_dim(y) != self.o:
            raise ValueError(f"Tanh expected to produce {self.o} outputs, but got feature dimension {_feature_dim(y)}")
        return y


class SumModule(nn.Module):
    def __init__(self, output_dim, with_constant):
        super().__init__()
        self.with_constant = with_constant
        self.o = output_dim

    def forward(self, x):
        x = _as_tensor_input(x)
        x = torch.add(x, self.with_constant)
        x = torch.sum(x, dim=-1, keepdim=True)
        return x


class ProductModule(nn.Module):
    def __init__(self, output_dim, with_constant):
        super().__init__()
        self.with_constant = with_constant
        self.o = output_dim

    def forward(self, x):
        x = _as_tensor_input(x)
        x = torch.mul(x, self.with_constant)
        x = torch.prod(x, dim=-1, keepdim=True)
        return x


class BesideModule(nn.Module):
    def __init__(self, head, tail, i1: int):
        super().__init__()
        self.head = head
        self.tail = tail
        self.i = i1

    def forward(self, x):
        x = _as_tensor_input(x)
        feature_dim = _feature_dim(x)
        if self.i < 0 or self.i > feature_dim:
            raise ValueError(f"BesideModule expected a split index in [0, {feature_dim}], got {self.i}.")
        if self.i == feature_dim:
            if self.tail is not None:
                raise ValueError("BesideModule: tail is not None, but input dimension matches head dimension")
            return self.head(x)
        else:
            x1, x2 = torch.split(x, [self.i, feature_dim - self.i], dim=-1)
            head_out = self.head(x1)
            tail_out = self.tail(x2)
            output = torch.cat((head_out, tail_out), dim=-1)
            return output


class BeforeModule(nn.Module):
    # nn.Sequential basically, but the type of our forward is tuple of tensors -> tuple of tensors, I want to make sure nothing unforeseen happens
    def __init__(self, head, tail):
        super().__init__()
        self.head = head
        self.tail = tail if tail is not None else lambda x: x  # catch beside singleton

    def forward(self, x):
        x = _as_tensor_input(x)
        head_out = self.head(x)
        tail_out = self.tail(head_out)
        return tail_out


class CopyModule(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.o = output_dim

    def forward(self, x):
        x = _as_tensor_input(x)
        feature_dim = _feature_dim(x)
        if feature_dim != 1:
            raise ValueError(f"CopyModule expects a single input feature, got {feature_dim}.")
        return x.expand(*x.shape[:-1], self.o).clone()


# A non-finite training loss is reported as measured, and this file substitutes no number for it.
# A substitute would have to be a penalty, and no constant is one: a regression loss is unbounded
# above, so a network that trained badly can score worse than the constant, and the diverged
# network then ranks ahead of it.  The sign convention does not save it: the loop maximizes and a
# minimizing objective enters negated (see ``AcquisitionFunction`` in
# ``bayesian_optimization.acquisition_function``), and negating preserves the order.
#
# Letting the value propagate makes the failure a failure.  ``bayesian_optimization.bo`` refuses a
# non-finite observation wherever one can enter, so a diverged evaluation stops the run there
# instead of entering the surrogate, which a single non-finite observation makes undefined at
# every term rather than at one.


#: The mini-batch size the training loop below uses when a caller names none.  It is no part of
#: the type-level specification, because the batch size does not change which network a term
#: denotes, only how that network is trained.
_DEFAULT_BATCH_SIZE = 128


@dataclass(frozen=True)
class TrainingProtocol:
    """What the training does beyond reading the term, the part that is protocol and not search.

    Every field defaults to the training this module did before any of them existed, which is the
    protocol ``P50`` below, so an unset protocol changes nothing.  That is deliberate.  A switch to
    a corrected recipe has to be visible in the record of a run rather than inherited in silence.

    None of these fields changes which networks are expressible.  A term denotes the same network
    with or without gradient clipping, and such a network is trained differently rather than built
    differently, which is why the fields live here and not in the repository.  The choices that do
    change the network are combinators of the repository instead, among them BatchNorm, the
    optimizer and the learning-rate schedule.

    Attributes:
        grad_clip (float | None): Maximum gradient norm, clipped between ``backward()`` and
            ``step()``.  None disables the clipping.
        image_shape (tuple[int, int, int] | None): The (C, H, W) of one sample, needed to fold a
            flat feature vector back into an image for augmentation.  None disables augmentation
            whatever the two flags below say.
        random_flip (bool): Mirror horizontally with probability 0.5.  Right for photographs and
            wrong for handwritten digits, where a mirrored 2 is not a 2, which is why this is a
            field per dataset and not a global setting.
        crop_padding (int): Pad by this many pixels on each side and take a random crop of the
            original size.  0 disables the cropping.
        init (str | None): "he" applies Kaiming-normal initialization to every Conv2d and Linear
            before training.  None keeps the default of torch, which is too small for a ReLU
            network by a factor of six.  ``apply_he_initialisation`` carries the arithmetic.
    """

    grad_clip: float | None = None
    image_shape: tuple[int, int, int] | None = None
    random_flip: bool = False
    crop_padding: int = 0
    init: str | None = None

    @property
    def augments(self) -> bool:
        """Whether any augmentation is configured.

        Returns:
            bool: True if the training batches are to be transformed.
        """
        return self.image_shape is not None and (self.random_flip or self.crop_padding > 0)


#: The protocol this module trained with before any field of ``TrainingProtocol`` existed, named
#: so that the record of a run can say which one it used.
P50 = TrainingProtocol()

#: The corrected recipe for 32x32 color images.  The clip bound is the one DARTS and NAS-Bench-201
#: both use, and the augmentation is the usual CIFAR-10 protocol.
P50_CORRECTED_CIFAR = TrainingProtocol(
    grad_clip=5.0, image_shape=(3, 32, 32), random_flip=True, crop_padding=4, init="he",
)

#: The same for the digit datasets, without the flip, which would mirror the digits.
P50_CORRECTED_DIGITS = TrainingProtocol(
    grad_clip=5.0, image_shape=(1, 16, 16), random_flip=False, crop_padding=2, init="he",
)


def apply_he_initialisation(model):
    """Re-initialize every Conv2d and Linear with Kaiming-normal weights.

    The default of torch for both is ``kaiming_uniform_(a=sqrt(5))``.  Its gain is
    ``sqrt(2 / (1 + 5))``, its bound is ``gain * sqrt(3 / fan_in)``, and a uniform draw on
    ``(-b, b)`` has variance ``b ** 2 / 3``, so ``Var(w) * fan_in`` comes out at ``1 / 3`` where a
    ReLU network needs 2.  Over a stack of convolutions and ReLUs the activation scale then shrinks
    layer by layer until the biases carry the signal, and a ReLU unit that has stopped firing has
    exactly zero gradient, so it never starts again.

    ``nonlinearity="relu"`` is an assumption of the interpretation and not a fact of the term.  The
    activation is a sibling module that the term places freely, so a layer does not know what
    follows it.  It is the right assumption because the alternatives, sigmoid and tanh, want a
    smaller gain, and choosing too small is what the default already does.

    Args:
        model: The interpreted network.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def _augment(x_batch, protocol):
    """Apply flip and crop to one flat batch, returning it flat again.

    The flat vector is the image.  ``ToTensor`` gives (C, H, W), the loader stacks that to
    (N, C, H, W), and ``reshape(N, -1)`` is row-major, so ``view(C, H, W)`` gives the original back
    element for element.  ``SynthConv2d`` already relies on exactly this round trip.

    Both transforms preserve the shape, so nothing downstream sees a difference, neither the query
    type nor the feature widths the repository offers nor the set of combinators.

    Args:
        x_batch: The batch, of shape (B, C*H*W).
        protocol (TrainingProtocol): Which transforms to apply.

    Returns:
        The transformed batch, of shape (B, C*H*W).
    """
    channels, height, width = protocol.image_shape
    batch = x_batch.shape[0]
    images = x_batch.view(batch, channels, height, width)

    if protocol.random_flip:
        mirror = torch.rand(batch, device=x_batch.device) < 0.5
        images = torch.where(mirror[:, None, None, None], images.flip(-1), images)

    pad = protocol.crop_padding
    if pad > 0:
        # Padding with zeros in the normalized space means padding with the channel mean in the
        # original space, which is closer to the usual practice than padding with black.
        padded = nn.functional.pad(images, (pad, pad, pad, pad))
        top = torch.randint(0, 2 * pad + 1, (batch,), device=x_batch.device)
        left = torch.randint(0, 2 * pad + 1, (batch,), device=x_batch.device)
        rows = top[:, None] + torch.arange(height, device=x_batch.device)[None, :]
        cols = left[:, None] + torch.arange(width, device=x_batch.device)[None, :]
        index = torch.arange(batch, device=x_batch.device)[:, None, None]
        images = padded[index, :, rows[:, :, None], cols[:, None, :]]
        # The advanced indexing above yields (B, H, W, C), so put the channel axis back in front.
        images = images.permute(0, 3, 1, 2)

    return images.reshape(batch, -1)


def learner(i, open_model, loss_fn, optim, n_epochs, x, y, x_test, y_test, batch_size=None,
            report=None, protocol=None, scheduler=None):
    # The training loop for the synthesized model.
    #
    # ``report``, if it is given, is filled with ``diverged`` and ``epochs_completed`` before this
    # returns.  Divergence used to be invisible outside this function, because the flag died here,
    # and a network that stopped in the first of fifty epochs then looked exactly like one that
    # trained through and is merely bad.  Under an accuracy objective the comment above about
    # non-finite values propagating protects nothing: an aborted run still yields a finite
    # accuracy, and with non-finite parameters ``argmax`` over a row of NaN returns index 0, so the
    # accuracy becomes the frequency of class 0.  A caller reading that number alone cannot tell it
    # apart from architectural signal, although it depends on the initialization rather than on the
    # architecture.  The measurement itself is never substituted, neither by a default value nor by
    # a clipped one.  It is only named, and the name is what cannot be recovered after the run.
    model = open_model
    # Device-agnostic: the model follows wherever the data already lives, rather than this path
    # naming a device anywhere.
    model = model.to(x.device)
    # CrossEntropyLoss expects logits (N, num_classes) against class indices (N,).  Unlike the
    # regression losses it must not be raveled and reshaped onto the shape of the target.
    is_classification = isinstance(loss_fn, nn.CrossEntropyLoss)
    batch_size = batch_size or _DEFAULT_BATCH_SIZE
    protocol = protocol if protocol is not None else P50

    diverged = False
    epochs_completed = 0
    parameter_list = list(model.parameters())
    if len(parameter_list) > 0:

        # Training mode.  Neither train() nor eval() used to be called anywhere in this path,
        # which was harmless only while no module behaved differently between the two.  BatchNorm
        # does.  Without the eval() further down it would normalize the test set with the batch
        # statistics of the test set, which leaks information between test examples.
        model.train()

        # fit model, in shuffled mini-batches rather than one full-batch step per epoch
        optimizer = optim(model)
        # ``scheduler`` is a builder (optimizer, n_epochs) -> lr_scheduler, or None for a constant
        # rate.  It has to be built here rather than passed in ready-made, because a scheduler
        # binds to an optimizer instance and that instance is created one line above.
        lr_schedule = scheduler(optimizer, n_epochs) if scheduler is not None else None
        n = x.shape[0]

        # nn.BatchNorm2d raises in train() mode when it sees exactly one value per channel.  The
        # intuitive reading, a batch of one, is the wrong one: it normalizes over N*H*W, so a
        # single sample at 2x2 is four values and passes, and only N*H*W == 1 fails.  The condition
        # is therefore a batch of one together with a BatchNorm whose spatial extent is 1x1, which
        # is reachable here, since a 32x32 image halves to 1x1 after five poolings.
        #
        # A batch of one can only ever be the last of an epoch, when n % batch_size == 1, so the
        # answer is to drop that one sample, which is what drop_last=True does in a
        # DataLoader-based pipeline.  It requires n > batch_size, so a dataset smaller than a
        # single batch is never silently trained on nothing.  There the error torch raises is the
        # right outcome, since such a network genuinely cannot be trained on one sample.  And what
        # is dropped is one sample per epoch, not a measurement.  Nothing is substituted or
        # hidden.
        has_pointwise_batchnorm = any(isinstance(m, SynthBatchNorm2d) and m.in_h * m.in_w == 1
                                      for m in model.modules())
        drop_last_single = has_pointwise_batchnorm and n > batch_size and n % batch_size == 1

        for _ in range(n_epochs):
            permutation = torch.randperm(n, device=x.device)
            for start in range(0, n, batch_size):
                idx = permutation[start:start + batch_size]
                if drop_last_single and idx.shape[0] == 1:
                    continue
                x_batch, y_batch = x[idx], y[idx]
                if protocol.augments:
                    # Training batches only.  The evaluation further down is never augmented.
                    x_batch = _augment(x_batch, protocol)
                optimizer.zero_grad()
                if is_classification:
                    loss = loss_fn(model(x_batch), y_batch)
                else:
                    loss = loss_fn(model(x_batch).ravel().reshape_as(y_batch), y_batch)
                if not torch.isfinite(loss):
                    # The numerics have diverged.  Stop early, because further steps only spend
                    # time on parameters that are already non-finite.  The test-time evaluation
                    # below still runs and still reports what it measures.
                    diverged = True
                    break
                loss.backward()
                if protocol.grad_clip is not None:
                    # Between backward() and step() is the only place this can sit.
                    nn.utils.clip_grad_norm_(parameter_list, protocol.grad_clip)
                optimizer.step()
            if diverged:
                break
            # After the batches of the epoch, which is what the T_max of CosineAnnealingLR
            # counts.  Stepping once per batch instead would run through the whole cosine in the
            # first epoch.
            if lr_schedule is not None:
                lr_schedule.step()
            epochs_completed += 1

    if report is not None:
        report["diverged"] = diverged
        report["epochs_completed"] = epochs_completed

    # Evaluation mode: BatchNorm has to use its running statistics here, not the statistics of
    # the evaluation batch.  inference_mode() alone does not switch that over.
    model.eval()
    with torch.inference_mode():
        if is_classification:
            loss = loss_fn(model(x_test), y_test)
        else:
            loss = loss_fn(model(x_test).ravel().reshape_as(y_test), y_test)
        return loss.item()


def pytorch_function_algebra():
    # One entry per combinator, each returning the module that combinator stands for.  A
    # composition returns the module that composes the two it was handed, so folding a term with
    # this algebra assembles the network from the leaves up.  The ``learner`` entry closes over
    # the assembled network and returns the callable that trains it.
    return {
        "edges": (
            lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13,
                   para14, para15, para16: EdgesModule()),

        "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                        para13, para14, para15, para16: SwapModule(n, m)),

        "linear_layer": (
            lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthLinear(o, l.in_features,
                                                                                              l.out_features, l.bias)),

        "conv2d": (
            lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthConv2d(
                l.in_channels, l.input_size[0], l.input_size[1], l.out_channels, l.output_size[0], l.output_size[1],
                l.kernel_size, l.stride, l.padding, l.bias)),

        "maxpool2d": (
            lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthMaxPool2d(
                l.in_channels, l.input_size[0], l.input_size[1], l.output_size[0], l.output_size[1], l.kernel_size)),

        "batchnorm2d": (
            lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthBatchNorm2d(
                l.in_channels, l.input_size[0], l.input_size[1])),

        "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthSigmoid(o)),

        "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthReLU(o, l.inplace)),

        "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthTanh(o)),

        "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SumModule(o, l.with_constant)),

        "product": (
            lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: ProductModule(o, l.with_constant)),

        "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: CopyModule(o)),

        "beside_singleton": (lambda i, o, ls, para, x: BesideModule(x, None, i)),

        "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: BesideModule(x, y, i1)),

        "before_singleton": (lambda i, o, r, ls, ls1, x: BeforeModule(x, None)),

        "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: BeforeModule(x, y)),

        "mse_loss": (lambda l: nn.MSELoss(reduction=l.reduction)),

        "l1loss": (lambda l: nn.L1Loss(reduction=l.reduction)),

        "cross_entropy_loss": (lambda l: nn.CrossEntropyLoss(reduction=l.reduction)),

        "adam_optimizer": (lambda o: lambda m: optim.Adam(
            m.parameters(), lr=o.learning_rate, weight_decay=o.weight_decay)),

        "sgd_optimizer": (lambda o: lambda m: optim.SGD(
            m.parameters(), lr=o.learning_rate, momentum=o.momentum,
            dampening=o.dampening, weight_decay=o.weight_decay, nesterov=o.nesterov)),

        "cosine_annealing_lr": (lambda opti, sched, optimizer: (optimizer,
            lambda o, n_epochs: optim.lr_scheduler.CosineAnnealingLR(
                o, T_max=n_epochs, eta_min=sched.eta_min))),

        # None, and not ConstantLR, because torch.optim.lr_scheduler.ConstantLR multiplies the
        # rate by 1/3 for the first five epochs, which is a schedule and not the absence of one.
        "no_scheduler": (lambda opti, sched, optimizer: (optimizer, None)),

        "learner": (
            lambda i, o, r, ls, e, l, opt, sch, loss, scheduler, model: lambda x, y, x_test, y_test, batch_size=None: learner(
                i, model, loss, scheduler[0], e, x, y, x_test, y_test, batch_size=batch_size,
                scheduler=scheduler[1])),
    }


def pytorch_model_algebra():
    # The same modules as ``pytorch_function_algebra`` builds, and the same reasoning.  The two
    # differ in the ``learner`` entry alone, which here returns the assembled network instead of a
    # callable that trains it.
    return {
        "edges": (
            lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13,
                   para14, para15, para16: EdgesModule()),

        "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                        para13, para14, para15, para16: SwapModule(n, m)),

        "linear_layer": (
            lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthLinear(o, l.in_features,
                                                                                              l.out_features, l.bias)),

        "conv2d": (
            lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthConv2d(
                l.in_channels, l.input_size[0], l.input_size[1], l.out_channels, l.output_size[0], l.output_size[1],
                l.kernel_size, l.stride, l.padding, l.bias)),

        "maxpool2d": (
            lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthMaxPool2d(
                l.in_channels, l.input_size[0], l.input_size[1], l.output_size[0], l.output_size[1], l.kernel_size)),

        "batchnorm2d": (
            lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthBatchNorm2d(
                l.in_channels, l.input_size[0], l.input_size[1])),

        "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthSigmoid(o)),

        "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthReLU(o, l.inplace)),

        "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthTanh(o)),

        "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SumModule(o, l.with_constant)),

        "product": (
            lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: ProductModule(o, l.with_constant)),

        "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: CopyModule(o)),

        "beside_singleton": (lambda i, o, ls, para, x: BesideModule(x, None, i)),

        "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: BesideModule(x, y, i1)),

        "before_singleton": (lambda i, o, r, ls, ls1, x: BeforeModule(x, None)),

        "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: BeforeModule(x, y)),

        "mse_loss": (lambda l: nn.MSELoss(reduction=l.reduction)),

        "l1loss": (lambda l: nn.L1Loss(reduction=l.reduction)),

        "cross_entropy_loss": (lambda l: nn.CrossEntropyLoss(reduction=l.reduction)),

        "adam_optimizer": (lambda o: lambda m: optim.Adam(
            m.parameters(), lr=o.learning_rate, weight_decay=o.weight_decay)),

        "sgd_optimizer": (lambda o: lambda m: optim.SGD(
            m.parameters(), lr=o.learning_rate, momentum=o.momentum,
            dampening=o.dampening, weight_decay=o.weight_decay, nesterov=o.nesterov)),

        "cosine_annealing_lr": (lambda opti, sched, optimizer: (optimizer,
            lambda o, n_epochs: optim.lr_scheduler.CosineAnnealingLR(
                o, T_max=n_epochs, eta_min=sched.eta_min))),

        # None, and not ConstantLR, because torch.optim.lr_scheduler.ConstantLR multiplies the
        # rate by 1/3 for the first five epochs, which is a schedule and not the absence of one.
        "no_scheduler": (lambda opti, sched, optimizer: (optimizer, None)),

        "learner": (
            lambda i, o, r, ls, e, l, opt, sch, loss, scheduler, model: model),
    }


def pytorch_components_algebra():
    """Like ``pytorch_model_algebra``, but yields the parts of the learner instead of the model.

    ``pytorch_function_algebra`` hands back a training closure that builds and trains its model
    internally, so a caller never gets a handle on the trained module and cannot measure anything
    about it afterwards, its accuracy or its parameter count included.  Interpreting the same term
    a second time does not help either, because every interpretation constructs freshly initialized
    modules, so the model inspected would not be the model that was trained.

    This algebra is ``pytorch_model_algebra`` with one entry changed.  ``learner`` yields
    ``(model, loss_fn, optimizer_factory, n_epochs, scheduler_factory)``, and a caller trains that
    exact model through ``learner(...)`` and keeps it afterwards.

    ``scheduler_factory`` is ``(optimizer, n_epochs) -> lr_scheduler``, or None for a constant
    rate, and goes straight into ``learner(..., scheduler=...)``.  It cannot be a ready-made
    scheduler, because a scheduler binds to an optimizer instance and that instance does not exist
    until ``learner`` calls ``optimizer_factory``.
    """
    algebra = pytorch_model_algebra()
    algebra["learner"] = (
        lambda i, o, r, ls, e, l, opt, sch, loss, scheduler, model: (model, loss, scheduler[0], e, scheduler[1]))
    return algebra
