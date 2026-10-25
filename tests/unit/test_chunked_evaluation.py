"""The evaluation passes a held-out set through the network in chunks, not at once.

A pass over a whole held-out set at once needs memory in proportion to the set: a wide CIFAR network over the 10 000 test
images needed more than a 24 GB card holds, while its training at batch 128 needed a fraction of it. A network in eval
mode answers each example independently of the others in its pass, so the passes are cut into chunks and the outputs put
back together in order: the loss and the accuracies are what they were, up to the rounding of the kernels a device may
pick for another batch size.

The network here is a linear map that records the size of every pass it is given in eval mode; the training is the real
``learner``'s, one epoch, and for the candidate's own passes a stand-in.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn, optim

from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_experiment_utils as utils
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_network_algebras import learner

#: The most examples one evaluation pass may hold.
CHUNK = 500
FEATURES = 3 * 32 * 32


class _Recording(nn.Module):
    """A linear classifier over flattened images that keeps the size of every pass it gets in eval mode."""

    def __init__(self, classes: int = 10) -> None:
        super().__init__()
        self.linear = nn.Linear(FEATURES, classes)
        self.evaluated: list[int] = []

    def forward(self, x):
        if not self.training:
            self.evaluated.append(x.shape[0])
        return self.linear(x)


def _data(n: int, classes: int = 10):
    return torch.randn(n, FEATURES), torch.randint(0, classes, (n,))


def test_the_learner_s_final_loss_passes_the_held_out_set_in_chunks():
    torch.manual_seed(0)
    model = _Recording()
    x, y = _data(64)
    x_held, y_held = _data(1_234)
    loss = learner(None, model, nn.CrossEntropyLoss(), lambda m: optim.SGD(m.parameters(), lr=0.01), 1,
                   x, y, x_held, y_held, batch_size=32)
    assert model.evaluated, "the held-out set was never passed in eval mode"
    assert max(model.evaluated) <= CHUNK, model.evaluated
    assert sum(model.evaluated) == 1_234
    with torch.inference_mode():
        whole = nn.CrossEntropyLoss()(model.linear(x_held), y_held).item()
    assert loss == pytest.approx(whole, rel=1e-6)


def test_a_candidate_s_accuracies_pass_both_held_out_sets_in_chunks(monkeypatch):
    torch.manual_seed(1)
    model = _Recording()
    loss_fn = nn.CrossEntropyLoss()

    class _Tree:
        def interpret(self, _algebra):
            return model, loss_fn, (lambda m: optim.SGD(m.parameters(), lr=0.01)), 1, None

    def trained(i, open_model, loss_fn, optim_factory, n_epochs, x, y, x_test, y_test, **options):
        # as the real learner leaves it: trained, and in eval mode
        open_model.eval()
        options["report"].update(diverged=False, epochs_completed=n_epochs)
        return 0.25

    monkeypatch.setattr(utils, "raw_learner", trained)
    x, y = _data(32)
    x_val, y_val = _data(1_100)
    x_test, y_test = _data(1_300)
    metrics = utils._train_candidate_once(_Tree(), x, y, x_val, y_val, 32, None, x_test, y_test, seed=None)
    assert max(model.evaluated) <= CHUNK, model.evaluated
    assert sum(model.evaluated) == 1_100 + 1_300
    with torch.inference_mode():
        val = (model.linear(x_val).argmax(dim=-1) == y_val).float().mean().item()
        test = (model.linear(x_test).argmax(dim=-1) == y_test).float().mean().item()
    assert metrics["accuracy"] == val
    assert metrics["test_accuracy"] == test


def test_a_pass_in_chunks_is_the_whole_pass_in_order():
    from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_network_algebras import (
        EVALUATION_CHUNK,
        forward_in_chunks,
    )

    assert EVALUATION_CHUNK == CHUNK
    torch.manual_seed(2)
    model = _Recording().eval()
    x, _y = _data(1_234)
    with torch.inference_mode():
        whole = model.linear(x)
        chunked = forward_in_chunks(model, x)
    assert model.evaluated == [500, 500, 234]
    assert torch.allclose(chunked, whole, rtol=0, atol=1e-6)
    assert torch.equal(chunked.argmax(dim=-1), whole.argmax(dim=-1))
    with pytest.raises(ValueError):
        forward_in_chunks(model, x, chunk=0)
