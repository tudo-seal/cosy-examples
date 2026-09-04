"""The bound on a parallel composition counts components, not feature widths.

``specification`` built its ``ParaTuples`` group with ``max_length=max(self.dimensions)``.
``max_length`` is how many components may sit beside each other in one parallel composition, while
``self.dimensions`` holds feature widths, so the expression read a quantity of one kind as a
quantity of another.  It stayed harmless only because the two numbers are close at the sizes this
example runs at.

The bound now has a name of its own.  These tests pin what it counts, that it does not follow the
feature widths any more, and the one thing it does not do: membership in the group reads no
length, so the bound narrows the enumeration and not the space the search covers.
"""

from __future__ import annotations

import pytest

from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository

#: The configuration ``damg_example.py`` searches.  Its largest feature width is 5, and the widest
#: parallel composition any target in ``damg_targets.py`` names is 3, so the two numbers differ
#: here and a reverted bound is visible.
SHIPPED = dict(
    linear_feature_dimensions=[1, 2, 3, 4, 5],
    constant_values=[0, 1, -1],
    learning_rate_values=[1e-2],
    n_epoch_values=[2000],
)


def _para(repository: DAMGrepository):
    """The ``Para`` group the way ``specification`` builds it."""
    labels = repository.Label(
        repository.dimensions,
        repository.linear_feature_dimensions,
        repository.constant_values,
    )
    return repository.Para(labels, repository.dimensions)


@pytest.mark.parametrize("width", [2, 3, 4])
def test_specification_bounds_the_group_by_the_configured_width(monkeypatch, width):
    """``specification`` passes the repository's parallel width on, and nothing else."""
    seen = []

    class Recording(DAMGrepository.ParaTuples):
        def __init__(self, para, max_length=3):
            seen.append(max_length)
            super().__init__(para, max_length=max_length)

    monkeypatch.setattr(DAMGrepository, "ParaTuples", Recording)
    DAMGrepository(**SHIPPED, max_parallel_width=width).specification()
    assert seen == [width]


def test_the_default_bound_is_not_the_largest_feature_width():
    """The default is a component count, and in this configuration it is not ``max(dimensions)``."""
    repository = DAMGrepository(**SHIPPED)
    assert repository.max_parallel_width == 3
    assert max(repository.dimensions) == 5


@pytest.mark.parametrize("largest_feature", [5, 8, 16])
def test_the_bound_does_not_follow_the_feature_widths(largest_feature):
    """Widening the features widens ``dimensions`` and leaves the component count alone."""
    repository = DAMGrepository(
        linear_feature_dimensions=list(range(1, largest_feature + 1)),
        constant_values=[0, 1, -1],
        learning_rate_values=[1e-2],
        n_epoch_values=[2000],
    )
    assert max(repository.dimensions) == largest_feature
    assert repository.max_parallel_width == 3


@pytest.mark.parametrize("width", [1, 2, 3])
def test_the_enumeration_stops_at_the_bound(width):
    """What the bound counts is components, so no enumerated composition is longer than it.

    Enumerated on a one-dimensional repository, because the group grows as the number of Para
    values to the power of the bound.
    """
    repository = DAMGrepository([1], [0, 1], [1e-2], [10])
    compositions = list(repository.ParaTuples(_para(repository), max_length=width))
    assert max(len(entry) for entry in compositions) == width


def test_membership_reads_no_length():
    """A composition wider than the bound is still a member, which is the bound's limit.

    Membership is the only question the synthesizer puts to this group when a target names a
    structure, so the bound does not keep a wider composition out of the search.  Turning it into
    a bound on the space would mean answering False here, and that is a decision about what the
    search covers, not about what the bound is called.
    """
    repository = DAMGrepository([1], [0, 1], [1e-2], [10])
    para = _para(repository)
    component = next(entry for entry in para if entry is not None)
    narrow = repository.ParaTuples(para, max_length=1)
    assert (component,) * 4 in narrow
