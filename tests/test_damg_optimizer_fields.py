"""The optimizer a DAMG term is trained with is the one the term names.

An Adam label carries nine fields, and the synthesized term prints all nine.  The interpretation
used to read one of them, the learning rate, and build ``torch.optim.Adam`` with defaults for the
rest, so a term that says ``weight_decay=0.0005`` trained without weight decay and a term that
says ``amsgrad=True`` trained without the AMSGrad correction.

These tests pin the mapping field by field, pin that an enumerated optimizer is still built the
way it was, and check the round trip on a term the synthesizer actually produced.
"""

from __future__ import annotations

import itertools

import pytest
import torch.nn as nn
from cosy.core import Synthesizer
from cosy.core.types import Constructor, Literal

from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository
from bayesian_optimization.examples.damg_nas.damg_repo_algebras import (
    pretty_term_algebra,
    pytorch_function_algebra,
    pytorch_model_algebra,
)

#: One non-default value per field of ``DAMGrepository.Adam``, and the key it has to arrive under
#: in a parameter group of ``torch.optim.Adam``.  ``learning_rate`` is the only renamed field.
FIELDS = [
    ("learning_rate", "lr", 0.5),
    ("betas", "betas", (0.5, 0.75)),
    ("eps", "eps", 1e-3),
    ("weight_decay", "weight_decay", 5e-4),
    ("amsgrad", "amsgrad", True),
    ("maximize", "maximize", True),
    ("capturable", "capturable", True),
    ("differentiable", "differentiable", True),
    ("decoupled_weight_decay", "decoupled_weight_decay", True),
]

ALGEBRAS = {"function": pytorch_function_algebra, "model": pytorch_model_algebra}


def _build(label: DAMGrepository.Adam, algebra_name: str = "function") -> dict:
    """Interpret ``adam_optimizer`` on ``label`` and return the parameter group it produces."""
    builder = ALGEBRAS[algebra_name]()["adam_optimizer"](label)
    return dict(builder(nn.Linear(1, 1)).param_groups[0])


def test_the_label_declares_exactly_the_fields_under_test():
    """The table above is complete, so a tenth field cannot be added and silently go unchecked."""
    declared = [f.name for f in DAMGrepository.Adam.__dataclass_fields__.values()]
    assert declared == [name for name, _, _ in FIELDS]


@pytest.mark.parametrize("algebra_name", list(ALGEBRAS))
@pytest.mark.parametrize(("field", "key", "value"), FIELDS, ids=[f[0] for f in FIELDS])
def test_every_field_of_the_label_reaches_the_optimizer(algebra_name, field, key, value):
    """Each field arrives, and the value it arrives with is the one the label carried."""
    label = DAMGrepository.Adam(**{field: value})
    assert getattr(label, field) == value, "the chosen value has to differ from the default"
    assert _build(label, algebra_name)[key] == value


def test_an_enumerated_optimizer_is_built_the_way_it_was():
    """What the search varies is the learning rate, and those terms are built unchanged.

    The eight remaining defaults of the label are torch's own defaults, so passing them on rather
    than leaving them out gives the same optimizer.
    """
    label = next(iter(DAMGrepository.Optimizer([1e-2])))
    group = _build(label)
    assert group["lr"] == 1e-2
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-08
    assert group["weight_decay"] == 0.0
    assert not any(group[key] for key in
                   ["amsgrad", "maximize", "capturable", "differentiable",
                    "decoupled_weight_decay"])


def test_a_term_naming_weight_decay_is_trained_with_weight_decay():
    """The round trip: the target names a value, the term prints it, the optimizer carries it."""
    label = DAMGrepository.Adam(learning_rate=1e-2, weight_decay=5e-4)
    repository = DAMGrepository(
        linear_feature_dimensions=[1, 2, 3, 4, 5],
        constant_values=[0, 1, -1],
        learning_rate_values=[1e-2],
        n_epoch_values=[2000],
    )
    target = Constructor("Learner", Constructor("DAG",
                                                Constructor("input", Literal(1))
                                                & Constructor("output", Literal(1))
                                                & Constructor("structure", Literal((None, None))))
                         & Constructor("Loss", Constructor("type", Literal(None)))
                         & Constructor("Optimizer", Constructor("type", Literal(label)))
                         & Constructor("epochs", Literal(2000)))
    space = Synthesizer(repository.specification(), {}).construct_solution_space(target).prune()
    terms = list(itertools.islice(space.enumerate_trees(target), 2))
    assert len(terms) == 2
    for term in terms:
        assert "weight_decay=0.0005" in term.interpret(pretty_term_algebra())
    assert _build(label)["weight_decay"] == 5e-4
