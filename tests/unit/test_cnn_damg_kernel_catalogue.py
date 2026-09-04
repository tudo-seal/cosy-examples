"""The kernel catalogue of the CNN example, and the round counts its comments claim.

Every entry of ``NAMED_KERNELS`` is a kernel a run may ask for by name, and the name is what the
run records.  Two things have to hold for that record to mean anything.  The catalogue has to
resolve, so a name in a record still names a kernel, and the round count behind a name has to be
the one the file says it is, since ``h`` is the one axis of these kernels that no fit adjusts.

The round counts are the claim of the comments in ``cnn_damg_kernels``: the finest granularity of
the architecture graph reads labels that carry the layer with its dimensions, so a histogram
already separates and it runs at ``h = 0``, while granularities 2 and 3 call every node ``node``
and reach the composition only through a relabeling round.
"""

from __future__ import annotations

import pytest

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_kernels import (
    NAMED_KERNELS,
    damg_kernel_1,
    damg_kernel_1_h2,
    damg_kernel_2,
    damg_kernel_2_h2,
    damg_kernel_3,
    damg_kernel_3_h2,
    named_kernel,
    wl_kernel_0_h2,
    wl_kernel_1,
    wl_kernel_1_h2,
    wl_kernel_2,
    wl_kernel_3,
)
from bayesian_optimization.kernels import WeisfeilerLehmanKernel


def test_every_name_in_the_catalogue_resolves_to_a_kernel():
    """A run records the name it asked for, so a name that resolves to nothing is a lost record.

    The catalogue is built at import, and a typo in one of its values raises there rather than
    here.  What this pins is the other half: every key reaches its kernel through the accessor a
    caller uses, and the two aliases of one kernel reach the same object.
    """
    assert NAMED_KERNELS, "the catalogue must not be empty"
    for name in sorted(NAMED_KERNELS):
        assert named_kernel(name) is NAMED_KERNELS[name]
    assert named_kernel("selected") is named_kernel("top3_sum")


def test_an_unknown_name_is_refused_and_says_what_there_is():
    """Falling back to a default would make the record say one thing and the fit do another.

    The message carries the keys, because the caller that got the name wrong is a command line and
    the person reading the error is the one who typed it.
    """
    with pytest.raises(ValueError, match="no kernel named"):
        named_kernel("damg_4")
    with pytest.raises(ValueError, match="top3_sum"):
        named_kernel("")


def test_the_architecture_graph_runs_one_round_from_granularity_two():
    """The round count differs by granularity, because what a node label carries differs by it.

    At granularity 1 the label is the layer with its dimensions, so the label multiset alone
    separates architectures.  From granularity 2 the layer type is folded away and every node is
    called ``node``, so a histogram counts nodes and nothing else.  Dropping the round on those
    two would make ``damg_kernel_2`` and ``damg_kernel_3`` return one value for every pair of
    architectures over the same number of nodes, which is what
    ``test_from_hierarchy_two_the_kernel_needs_a_relabeling_round`` in the graph conversion tests
    measures on a pair.
    """
    assert damg_kernel_1.h == 0
    assert damg_kernel_2.h == 1
    assert damg_kernel_3.h == 1


def test_the_two_round_variants_carry_two_rounds_at_every_granularity():
    """The uniform variant exists so that a run can vary the one axis a fit cannot reach.

    It is a second option rather than a second default, and it is only that if all three of its
    granularities really run at the count its name announces.
    """
    for kernel in (damg_kernel_1_h2, damg_kernel_2_h2, damg_kernel_3_h2):
        assert kernel.h == 2
    for kernel in (wl_kernel_0_h2, wl_kernel_1_h2):
        assert kernel.h == 2


def test_the_term_readings_compare_labels_without_a_round():
    """These three read the folded term, where a label is a combinator symbol.

    A round costs a full relabeling of every graph, so it is spent only where a histogram is blind
    to a difference that matters, and over combinator symbols it is not.
    """
    for kernel in (wl_kernel_1, wl_kernel_2, wl_kernel_3):
        assert isinstance(kernel, WeisfeilerLehmanKernel)
        assert kernel.h == 0
        assert kernel.normalize is True
