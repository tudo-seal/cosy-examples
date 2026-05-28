# -*- coding: utf-8 -*-
"""
Algebren für ODErepository ausgelagert.
Die Funktionen hier nehmen das `repo`-Objekt entgegen und geben die jeweiligen Algebra-Dicts zurück.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from cosy.core.tree import Tree
from cosy.core.types import Constructor, Literal

import numpy as np
import math


# Interpretations of terms are algebras in my language

# The pretty_term_algebra interprets a Tree as easily readible string.
def pretty_term_algebra():
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: f"edges({io})"),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: f"swap({io}, {n}, {m})"),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: f"({str(l)}, {i}, {o})"),

            "beside_singleton": (lambda i, o, ls, para, x: f"{x})"),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: f"{x} || {y}"),

            "before_singleton": (lambda i, o, r, ls, ls1, x: f"({x}"),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: f"({x} ; {y}"),

            "mse_loss": (lambda l: str(l)),

            "l1loss": (lambda l: str(l)),

            "adam_optimizer": (lambda o: str(o)),

            "learner": (lambda i, o, r, ls, e, l, opt, loss, optimizer, model: f"""
Learner(
    model= (
        {model}
        ), 
    loss= {loss}, 
    optimizer= {optimizer}, 
    epochs= {e}
    )
"""),
            "edges_h1": (lambda io: f"edges({io})"),

            "swap_h1": (
                lambda io, n, m: f"swap({io}, {n}, {m})"),

            "linear_layer_h1": (lambda l, i, o: f"({str(l)}, {i}, {o})"),

            "sigmoid_h1": (lambda l, i, o: f"({str(l)}, {i}, {o})"),

            "relu_h1": (lambda l, i, o: f"({str(l)}, {i}, {o})"),

            "tanh_h1": (lambda l, i, o: f"({str(l)}, {i}, {o})"),

            "sum_h1": (lambda l, i, o: f"({str(l)}, {i}, {o})"),

            "product_h1": (lambda l, i, o: f"({str(l)}, {i}, {o})"),

            "copy_h1": (lambda l, i, o: f"({str(l)}, {i}, {o})"),

            "beside_singleton_h1": (lambda i, o, x: f"{x})"),

            "beside_cons_h1": (lambda i, i1, i2, o, o1, o2, x, y: f"{x} || {y}"),

            "before_singleton_h1": (lambda i, o, x: f"({x}"),

            "before_cons_h1": (lambda i, j, o, x, y: f"({x} ; {y}"),

            "learner_h1": (lambda i, o, e, loss, optimizer, model: f"""
        Learner(
            model= (
                {model}
                ), 
            loss= {loss}, 
            optimizer= {optimizer}, 
            epochs= {e}
            )
        """),
            "node": lambda i, o: "node",

            "loss": "loss",

            "optimizer": "optimizer",

            #"beside_singleton_h2": (lambda i, x: f"{x})"),

            #"beside_cons_h2": (lambda i1, x, y: f"{x} || {y}"),

            #"before_singleton_h2": (lambda i, x: f"({x}"),

            #"before_cons_h2": (lambda i, x, y: f"({x} ; {y}"),

            "learner_h2": (lambda e, loss, optimizer, model: f"""
                Learner(
                    model= (
                        {model}
                        ), 
                    loss= {loss}, 
                    optimizer= {optimizer}, 
                    epochs= {e}
                    )
                """),

            "beside_singleton_h3": (lambda i, o, x: f"{x})"),

            "beside_cons_h3": (lambda i, i1, o, x, y: f"{x} || {y}"),
        }

def edgelist_learner(model, loss, optimizer, epochs, verbose=False):
        f, inputs = model
        edgelist, to_outputs, pos_A = f((-5.5, -3.8), ["input" for _ in range(0, inputs)])
        edgelist = edgelist + [(o, loss) for o in to_outputs] + [(loss, optimizer)] + [(optimizer, f"epochs({epochs})")]
        if verbose:
            output_x = max([x for x, y in pos_A.values()]) + 2.5
            pos_A = pos_A | {"input": (-5.5, -3.8), "output": (output_x, -3.8),
                             loss: (output_x + 2.5, -3.8), optimizer: (output_x + 5.0, -3.8),
                             f"epochs({epochs})": (output_x + 5, -3.5)}
            return edgelist, pos_A
        return edgelist

# The edgelist_algebra interprets a Tree as a list of edges, where each edge is a tuple of two strings.
# The first string is the source node and the second string is the target node.
# The source and target nodes are labeled with the parameters of the corresponding constructor.
# This algebra allows us to translate a Tree into its corresponding directed acyclic multigraph, because for
# example nx.DiGraph takes an edgelist as input to construct such a graph.
# Therefore, composing the edgelist_algebra with a graph-constructor enables us to interpret a synthesized Tree as
# the directed acyclic multigraph it encodes and therefore as the computational graph of a neural network.
def edgelist_algebra(verbose=False):
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: lambda id, inputs: ([], inputs, {})),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: lambda id, inputs: ([], inputs[n:] + inputs[:n], {})),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "beside_singleton": (lambda i, o, ls, para, x: x),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: lambda id, inputs:
                (x(id, inputs[:i1])[0] + y((id[0], id[1] + 0.2), inputs[i1:])[0],
                 x(id, inputs[:i1])[1] + y((id[0], id[1] + 0.2), inputs[i1:])[1],
                 x(id, inputs[:i1])[2] | y((id[0], id[1] + 0.2), inputs[i1:])[2])),

            "before_singleton": (lambda i, o, r, ls, ls1, x: (x, i)),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: (lambda id, inputs:
                                                                      (
                                                                           y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[0] + x(id, inputs)[0],
                                                                           y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[1],
                                                                           y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[2] | x(id, inputs)[2]
                                                                       ),
                                                                       i)),

            "mse_loss": (lambda l: str(l)),

            "l1loss": (lambda l: str(l)),

            "adam_optimizer": (lambda o: str(o)),

            "learner": (lambda i, o, r, ls, e, l, opt, loss, optimizer, model: edgelist_learner(model, loss, optimizer, e, verbose=verbose)),

            "edges_h1": (lambda io: lambda id, inputs: ([], inputs, {})),

            "swap_h1": (
                lambda io, n, m: lambda id, inputs: ([], inputs[n:] + inputs[:n], {})),

            "linear_layer_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "sigmoid_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "relu_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "tanh_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "sum_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "product_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "copy_h1": (lambda l, i, o: lambda id, inputs: (
                [(x, str((l, i, o)) + str(id)) for x in inputs], [str((l, i, o)) + str(id) for _ in range(0, o)],
                {str((l, i, o)) + str(id): id})),

            "beside_singleton_h1": (lambda i, o, x: x),

            "beside_cons_h1": (lambda i, i1, i2, o, o1, o2, x, y: lambda id, inputs:
                (x(id, inputs[:i1])[0] + y((id[0], id[1] + 0.2), inputs[i1:])[0],
                 x(id, inputs[:i1])[1] + y((id[0], id[1] + 0.2), inputs[i1:])[1],
                 x(id, inputs[:i1])[2] | y((id[0], id[1] + 0.2), inputs[i1:])[2])),

            "before_singleton_h1": (lambda i, o, x: (x, i)),

            "before_cons_h1": (lambda i, j, o, x, y: (lambda id, inputs:
                                                                      (
                                                                           y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[0] + x(id, inputs)[0],
                                                                           y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[1],
                                                                           y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[2] | x(id, inputs)[2]
                                                                       ),
                                                                       i)),

            "learner_h1": (lambda i, o, e, loss, optimizer, model: edgelist_learner(model, loss, optimizer, e, verbose=verbose)),

            "node": (lambda i, o: lambda id, inputs: (
                [(x, "node" + str(id)) for x in inputs], ["node" + str(id) for _ in range(0, o)],
                {"node" + str(id): id})),

            "loss": "loss",

            "optimizer": "optimizer",

            #"beside_singleton_h2": (lambda i, x: x),

            #"beside_cons_h2": (lambda i1, x, y: lambda id, inputs:
            #    (x(id, inputs[:i1])[0] + y((id[0], id[1] + 0.2), inputs[i1:])[0],
            #     x(id, inputs[:i1])[1] + y((id[0], id[1] + 0.2), inputs[i1:])[1],
            #     x(id, inputs[:i1])[2] | y((id[0], id[1] + 0.2), inputs[i1:])[2])),

            #"before_singleton_h2": (lambda i, x: (x, i)),

            #"before_cons_h2": (lambda i, x, y: (lambda id, inputs:
            #                                                          (
            #                                                               y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[0] + x(id, inputs)[0],
            #                                                               y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[1],
            #                                                               y[0]((id[0] + 2.5, id[1]), x(id, inputs)[1])[2] | x(id, inputs)[2]
            #                                                           ),
            #                                                           i)),

            "learner_h2": (lambda e, loss, optimizer, model: edgelist_learner(model, loss, optimizer, e, verbose=verbose)),

            "beside_singleton_h3": (lambda i, o, x: x),

            "beside_cons_h3": (lambda i, i1, o, x, y: lambda id, inputs:
            (x(id, inputs[:i1])[0] + y((id[0], id[1] + 0.2), inputs[i1:])[0],
             x(id, inputs[:i1])[1] + y((id[0], id[1] + 0.2), inputs[i1:])[1],
             x(id, inputs[:i1])[2] | y((id[0], id[1] + 0.2), inputs[i1:])[2])),
        }

    # In the following we interpret the combinators as pytorch nn.modules


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


# Sinnvoller Strafwert: groß genug, um als "schlecht" erkennbar zu sein,
# klein genug, um Skalierungen (log1p, MinMax) nicht zu sprengen.
_NON_FINITE_LOSS_PENALTY = 1e6


def _sanitize_loss(value: float) -> float:
    """Map non-finite losses (NaN/Inf) to a finite penalty value.

    Reasons NaN/Inf can occur for nn.MSELoss:
      * Gradient explosion (e.g. ProductModule, high lr, deep nets)
      * NaN propagating through parameters after a single Inf gradient step
      * float32 overflow when predictions diverge
    Returning a non-finite value would break downstream consumers
    (sklearn GP-fit raises 'Input y contains infinity').
    """
    if not math.isfinite(value):
        return _NON_FINITE_LOSS_PENALTY
    return value


def learner(i, open_model, loss_fn, optim, n_epochs, x, y, x_test, y_test):
    # training loop for the synthesized model

    model = open_model
    parameter_list = list(model.parameters())
    if len(parameter_list) > 0:

        # fit model
        optimizer = optim(model)
        for _ in range(n_epochs):
            optimizer.zero_grad()
            pred = model(x).ravel()
            test = pred.reshape_as(y)
            loss = loss_fn(test, y)
            if not torch.isfinite(loss):
                # Training has diverged; stop early to keep params usable
                # for the test-time evaluation below (which will then likely
                # also produce a non-finite loss and trigger the sanitizer).
                break
            loss.backward()
            optimizer.step()

    with torch.inference_mode():
        y_pred = model(x_test).ravel()
        test = y_pred.reshape_as(y_test)
        loss = loss_fn(test, y_test)
        return _sanitize_loss(loss.item())

    # Having interpreted each combinator as its corresponding pytorch nn.module above allows the
    # pytorch_function_algebra to interpret a Tree directly as a callable learning pipeline for the
    # encoded neural network.


def pytorch_function_algebra():
    # Use nn.ModuleDict to store layers with unique names (that's why we abstract over an id) and implement
    # it as the pytorch algebra nn.ModuleDict then becomes the __init__ part and
    # the forward function is constructed accordingly
    return {
        "edges": (
            lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13,
                   para14, para15, para16: EdgesModule()),

        "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                        para13, para14, para15, para16: SwapModule(n, m)),

        "linear_layer": (
            lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthLinear(o, l.in_features,
                                                                                              l.out_features, l.bias)),

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

        "adam_optimizer": (lambda o: lambda m: optim.Adam(m.parameters(), lr=o.learning_rate)),

        "learner": (
            lambda i, o, r, ls, e, l, opt, loss, optimizer, model: lambda x, y, x_test, y_test: learner(i, model,
                                                                                                             loss,
                                                                                                             optimizer,
                                                                                                             e, x, y,
                                                                                                             x_test,
                                                                                                             y_test)),
    }

def pytorch_model_algebra():
    # Use nn.ModuleDict to store layers with unique names (that's why we abstract over an id) and implement
    # it as the pytorch algebra nn.ModuleDict then becomes the __init__ part and
    # the forward function is constructed accordingly
    return {
        "edges": (
            lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13,
                   para14, para15, para16: EdgesModule()),

        "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                        para13, para14, para15, para16: SwapModule(n, m)),

        "linear_layer": (
            lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: SynthLinear(o, l.in_features,
                                                                                              l.out_features, l.bias)),

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

        "adam_optimizer": (lambda o: lambda m: optim.Adam(m.parameters(), lr=o.learning_rate)),

        "learner": (
            lambda i, o, r, ls, e, l, opt, loss, optimizer, model: lambda x, y, x_test, y_test: model),
    }

def hierarchy_algebra(level: int):
    # Currently only level 0, 1, 2 and 3 exists. So if the level isn't 1, 2 or 3, level 0 is assumed.
    if level == 1:
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                             para13, para14, para15, para16: Tree("edges_h1", (Tree(io),))),

            "swap": (
                lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                       para13, para14, para15, para16: Tree("swap_h1",  (Tree(io), Tree(n), Tree(m),))),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("linear_layer_h1", (Tree(l), Tree(i), Tree(o),))),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("sigmoid_h1", (Tree(l), Tree(i), Tree(o),))),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("relu_h1", (Tree(l), Tree(i), Tree(o),))),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("tanh_h1", (Tree(l), Tree(i), Tree(o),))),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("sum_h1", (Tree(l), Tree(i), Tree(o),))),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("product_h1", (Tree(l), Tree(i), Tree(o),))),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7:
                             Tree("copy_h1", (Tree(l), Tree(i), Tree(o),))),

            "beside_singleton": (lambda i, o, ls, para, x: Tree("beside_singleton_h1", (Tree(i), Tree(o), x,))),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: Tree("beside_cons_h1", (Tree(i), Tree(i1), Tree(i2), Tree(o), Tree(o1), Tree(o2), x, y,))),

            "before_singleton": (lambda i, o, r, ls, ls1, x: Tree("before_singleton_h1",(Tree(i), Tree(o), x,))),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: Tree("before_cons_h1", (Tree(i), Tree(j), Tree(o), x, y,))),

             "mse_loss": (lambda l: Tree("mse_loss", (Tree(l),))),

            "l1loss": (lambda l: Tree("l1loss", (Tree(l),))),

            "adam_optimizer": (lambda o: Tree("adam_optimizer", (Tree(o),))),

            "learner": (lambda i, o, r, ls, e, l, opt, loss, optimizer, model: Tree("learner_h1",
                                                                                    (Tree(i), Tree(o), Tree(e), loss, optimizer, model,)))
        }
    if level == 2:
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                             para13, para14, para15, para16: Tree("edges_h1", (Tree(io),))),

            "swap": (
                lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                       para13, para14, para15, para16: Tree("swap_h1",  (Tree(io), Tree(n), Tree(m),))),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "beside_singleton": (lambda i, o, ls, para, x: Tree("beside_singleton_h1", (Tree(i), Tree(o), x,))),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: Tree("beside_cons_h1", (Tree(i), Tree(i1), Tree(i2), Tree(o), Tree(o1), Tree(o2), x, y,))),

            "before_singleton": (lambda i, o, r, ls, ls1, x: Tree("before_singleton_h1",(Tree(i), Tree(o), x,))),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: Tree("before_cons_h1", (Tree(i), Tree(j), Tree(o), x, y,))),

            "mse_loss": (lambda l: Tree("loss")),

            "l1loss": (lambda l: Tree("loss")),

            "adam_optimizer": (lambda o: Tree("optimizer")),

            "learner": (lambda i, o, r, ls, e, l, opt, loss, optimizer, model: Tree("learner_h2",
                                                                                    (Tree(e), loss, optimizer, model,)))
        }
    elif level == 3:
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                             para13, para14, para15, para16: Tree("edges_h1", (Tree(io),))),

            "swap": (
                lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                       para13, para14, para15, para16: Tree("swap_h1",  (Tree(io), Tree(n), Tree(m),))),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("node", (Tree(i), Tree(o),))),

            "beside_singleton": (lambda i, o, ls, para, x: Tree("beside_singleton_h3", (Tree(i), Tree(o), x,))),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y:
                            Tree("beside_cons_h3", (Tree(i), Tree(i1), Tree(o), x, y,))
                            if x.root == "edges_h1" or x.root == "swap_h1"
                            else (Tree("beside_singleton_h3", (Tree(i), Tree(o), y.children[2]))
                                  if y.root == "beside_singleton_h3"
                                  else Tree("beside_cons_h3",
                                            (Tree(i), Tree(i1), Tree(o), Tree("node", (Tree(i1), Tree(o1),)), y.children[4])))),

            "before_singleton": (lambda i, o, r, ls, ls1, x: Tree("before_singleton_h1",(Tree(i), Tree(o), x,))),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: Tree("before_cons_h1", (Tree(i), Tree(j), Tree(o), x, y,))),

            "mse_loss": (lambda l: Tree("loss")),

            "l1loss": (lambda l: Tree("loss")),

            "adam_optimizer": (lambda o: Tree("optimizer")),

            "learner": (lambda i, o, r, ls, e, l, opt, loss, optimizer, model: Tree("learner_h2",
                                                                                    (Tree(e), loss, optimizer, model,)))
        }
    else:
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                             para13, para14, para15, para16: Tree("edges",
                                                                  (
                                                                      Tree(io), Tree(para1), Tree(para2), Tree(para3),
                                                                      Tree(para4), Tree(para5), Tree(para6),
                                                                      Tree(para7),
                                                                      Tree(para8), Tree(para9), Tree(para10),
                                                                      Tree(para11), Tree(para12), Tree(para13),
                                                                      Tree(para14), Tree(para15), Tree(para16),
                                                                  ))),

            "swap": (
                lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                       para13, para14, para15, para16: Tree("swap",
                                                            (
                                                                Tree(io), Tree(n), Tree(m),
                                                                Tree(para1), Tree(para2), Tree(para3),
                                                                Tree(para4), Tree(para5), Tree(para6),
                                                                Tree(para7),
                                                                Tree(para8), Tree(para9), Tree(para10),
                                                                Tree(para11), Tree(para12), Tree(para13),
                                                                Tree(para14), Tree(para15), Tree(para16),
                                                            ))),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("linear_layer",
                                                                                                   (
                                                                                                       Tree(l), Tree(i),
                                                                                                       Tree(o),
                                                                                                       Tree(para1),
                                                                                                       Tree(para2),
                                                                                                       Tree(para3),
                                                                                                       Tree(para4),
                                                                                                       Tree(para5),
                                                                                                       Tree(para6),
                                                                                                       Tree(para7),
                                                                                                   ))),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("sigmoid",
                                                                                              (
                                                                                                  Tree(l), Tree(i),
                                                                                                  Tree(o),
                                                                                                  Tree(para1),
                                                                                                  Tree(para2),
                                                                                                  Tree(para3),
                                                                                                  Tree(para4),
                                                                                                  Tree(para5),
                                                                                                  Tree(para6),
                                                                                                  Tree(para7),
                                                                                              ))),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("relu",
                                                                                           (
                                                                                               Tree(l), Tree(i),
                                                                                               Tree(o),
                                                                                               Tree(para1), Tree(para2),
                                                                                               Tree(para3),
                                                                                               Tree(para4), Tree(para5),
                                                                                               Tree(para6),
                                                                                               Tree(para7),
                                                                                           ))),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("tanh",
                                                                                           (
                                                                                               Tree(l), Tree(i),
                                                                                               Tree(o),
                                                                                               Tree(para1), Tree(para2),
                                                                                               Tree(para3),
                                                                                               Tree(para4), Tree(para5),
                                                                                               Tree(para6),
                                                                                               Tree(para7),
                                                                                           ))),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("sum",
                                                                                          (
                                                                                              Tree(l), Tree(i), Tree(o),
                                                                                              Tree(para1), Tree(para2),
                                                                                              Tree(para3),
                                                                                              Tree(para4), Tree(para5),
                                                                                              Tree(para6),
                                                                                              Tree(para7),
                                                                                          ))),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("product",
                                                                                              (
                                                                                                  Tree(l), Tree(i),
                                                                                                  Tree(o),
                                                                                                  Tree(para1),
                                                                                                  Tree(para2),
                                                                                                  Tree(para3),
                                                                                                  Tree(para4),
                                                                                                  Tree(para5),
                                                                                                  Tree(para6),
                                                                                                  Tree(para7),
                                                                                              ))),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: Tree("copy",
                                                                                           (
                                                                                               Tree(l), Tree(i),
                                                                                               Tree(o),
                                                                                               Tree(para1), Tree(para2),
                                                                                               Tree(para3),
                                                                                               Tree(para4), Tree(para5),
                                                                                               Tree(para6),
                                                                                               Tree(para7),
                                                                                           ))),

            "beside_singleton": (lambda i, o, ls, para, x: Tree("beside_singleton",
                                                                (
                                                                    Tree(i), Tree(o),
                                                                    Tree(ls), Tree(para), x
                                                                ))),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: Tree("beside_cons",
                                                                                    (
                                                                                        Tree(i), Tree(i1), Tree(i2),
                                                                                        Tree(o), Tree(o1), Tree(o2),
                                                                                        Tree(ls), Tree(head),
                                                                                        Tree(tail),
                                                                                        x, y,
                                                                                    ))),

            "before_singleton": (lambda i, o, r, ls, ls1, x: Tree("before_singleton",
                                                                  (
                                                                      Tree(i), Tree(o), Tree(r),
                                                                      Tree(ls), Tree(ls1), x,
                                                                  ))),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: Tree("before_cons",
                                                                          (
                                                                              Tree(i), Tree(j), Tree(o),
                                                                              Tree(r), Tree(ls), Tree(head), Tree(tail),
                                                                              x, y,
                                                                          ))),

            "mse_loss": (lambda l: Tree("mse_loss", (Tree(l),))),

            "l1loss": (lambda l: Tree("l1loss", (Tree(l),))),

            "adam_optimizer": (lambda o: Tree("adam_optimizer", (Tree(o),))),

            "learner": (lambda i, o, r, ls, e, l, opt, loss, optimizer, model: Tree("learner",
                                                                                    (
                                                                                        Tree(i), Tree(o), Tree(r),
                                                                                        Tree(ls), Tree(e), Tree(l),
                                                                                        Tree(opt),
                                                                                        loss, optimizer, model,
                                                                                    )))
        }

def request_algebra():
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                             para13, para14, para15, para16: (("swap", 0, io), io, io)),

            "swap": (
                lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                       para13, para14, para15, para16: (("swap", n, m), io, io)),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (l, i, o)),

            "beside_singleton": (lambda i, o, ls, para, x: (x,)),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: (x,) + y),

            "before_singleton": (lambda i, o, r, ls, ls1, x: (x,)),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: (x,) + y),

            "mse_loss": (lambda l: Literal(l)),

            "l1loss": (lambda l: Literal(l)),

            "adam_optimizer": (lambda o: Literal(o)),

            "learner": (lambda i, o, r, ls, e, l, opt, loss, optimizer, model: Constructor("Learner", Constructor("DAG",
                          Constructor("input", Literal(i))
                          & Constructor("output", Literal(o))
                          & Constructor("structure", Literal(model)))
                                & Constructor("Loss", Constructor("type", loss))
                                & Constructor("Optimizer", Constructor("type", optimizer))
                                & Constructor("epochs", Literal(e))
                                ))
        }


def refinement_1_algebra():
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: None),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: None),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: None),

            "beside_singleton": (lambda i, o, ls, para, x: (x,)),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: (x,) + y),

            "before_singleton": (lambda i, o, r, ls, ls1, x: (x,)),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: (x,) + y),

            "mse_loss": (lambda l: Literal(None)),

            "l1loss": (lambda l: Literal(None)),

            "adam_optimizer": (lambda o: Literal(None)),

            "learner": (lambda i, o, r, ls, e, l, opt, loss, optimizer, model: Constructor("Learner", Constructor("DAG",
                          Constructor("input", Literal(i))
                          & Constructor("output", Literal(o))
                          & Constructor("structure", Literal(model)))
                                & Constructor("Loss", Constructor("type", loss))
                                & Constructor("Optimizer", Constructor("type", optimizer))
                                & Constructor("epochs", Literal(e))
                                ))
        }

# I have the feeling, that this one is to restrictive!
def refinement_2_algebra():
        return {
            "edges": (lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: (("swap", 0, io), io, io)),

            "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13, para14, para15, para16: (("swap", n, m), io, io)),

            "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: (None, i, o)),

            "beside_singleton": (lambda i, o, ls, para, x: (x,)),

            "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: (x,) + y),

            "before_singleton": (lambda i, o, r, ls, ls1, x: (x,)),

            "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: (x,) + y),

            "mse_loss": (lambda l: Literal(None)),

            "l1loss": (lambda l: Literal(None)),

            "adam_optimizer": (lambda o: Literal(None)),

            "learner": (lambda i, o, r, ls, e, l, opt, loss, optimizer, model: Constructor("Learner", Constructor("DAG",
                          Constructor("input", Literal(i))
                          & Constructor("output", Literal(o))
                          & Constructor("structure", Literal(model)))
                                & Constructor("Loss", Constructor("type", loss))
                                & Constructor("Optimizer", Constructor("type", optimizer))
                                & Constructor("epochs", Literal(e))
                                ))
        }


def operator_histogram(vec):
    if vec.sum() > 0:
        vec /= vec.sum()

    return vec

"""
Count occurences of combinators as vector (edges#, swap#, linear_layer#, sigmoid#, relu#, tanh#, sum#, product#, copy#) 
"""
def operator_histogram_algebra():
    return {
        "edges": (
            lambda io, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12, para13,
                   para14, para15, para16: np.array((1, 0, 0, 0, 0, 0, 0, 0, 0), dtype=float)),

        "swap": (lambda io, n, m, para1, para2, para3, para4, para5, para6, para7, para8, para9, para10, para11, para12,
                        para13, para14, para15, para16: np.array((0, 1, 0, 0, 0, 0, 0, 0, 0), dtype=float)),

        "linear_layer": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: np.array((0, 0, 1, 0, 0, 0, 0, 0, 0), dtype=float)),

        "sigmoid": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: np.array((0, 0, 0, 1, 0, 0, 0, 0, 0), dtype=float)),

        "relu": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: np.array((0, 0, 0, 0, 1, 0, 0, 0, 0), dtype=float)),

        "tanh": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: np.array((0, 0, 0, 0, 0, 1, 0, 0, 0), dtype=float)),

        "sum": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: np.array((0, 0, 0, 0, 0, 0, 1, 0, 0), dtype=float)),

        "product": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: np.array((0, 0, 0, 0, 0, 0, 0, 1, 0), dtype=float)),

        "copy": (lambda l, i, o, para1, para2, para3, para4, para5, para6, para7: np.array((0, 0, 0, 0, 0, 0, 0, 0, 1), dtype=float)),

        "beside_singleton": (lambda i, o, ls, para, x: x),

        "beside_cons": (lambda i, i1, i2, o, o1, o2, ls, head, tail, x, y: x + y),

        "before_singleton": (lambda i, o, r, ls, ls1, x: x),

        "before_cons": (lambda i, j, o, r, ls, head, tail, x, y: x + y),

        "mse_loss": (lambda l: ()),

        "l1loss": (lambda l: ()),

        "adam_optimizer": (lambda o: ()),

        "learner": (lambda i, o, r, ls, e, l, opt, loss, optimizer, model: operator_histogram(model))
    }

