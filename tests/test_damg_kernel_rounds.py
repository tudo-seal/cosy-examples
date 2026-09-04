"""How many relabeling rounds each DAMG kernel runs, and what each round is there for.

``h`` is the number of Weisfeiler-Lehman rounds a kernel runs on top of comparing the initial
labels, so ``h = 0`` is a plain histogram over those labels.  A round costs a full relabeling of
every graph and it is not a hyperparameter the marginal likelihood can adjust, so these tests pin
the two things the setting rests on: that a histogram alone separates where the label carries the
layer, and that it is blind where the label does not.

The numbers here are the ones the comments in ``damg_kernels.py`` name.
"""

from __future__ import annotations

import itertools
from collections import Counter

import numpy as np
import pytest
from cosy.core import Synthesizer
from cosy.core.types import Constructor, Literal

from bayesian_optimization.examples.damg_nas import damg_targets
from bayesian_optimization.examples.damg_nas.damg_kernels import (
    as_DAMG,
    as_hierarchical_damg,
    as_hierarchical_tree_graph,
    damg_kernel_1,
    damg_kernel_2,
    damg_kernel_3,
    hierarchical_tree,
    wl_kernel_1,
    wl_kernel_2,
    wl_kernel_3,
)
from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository
from bayesian_optimization.kernels.graph_kernel import WeisfeilerLehmanKernel

#: How many terms of ``target_len_3`` the Gram matrices below are built on.
SAMPLE = 40


@pytest.fixture(scope="module")
def specification():
    """The repository in the configuration ``damg_example.py`` searches."""
    return DAMGrepository(
        linear_feature_dimensions=[1, 2, 3, 4, 5],
        constant_values=[0, 1, -1],
        learning_rate_values=[1e-2],
        n_epoch_values=[2000],
    ).specification()


def _terms(specification, target, count):
    space = Synthesizer(specification, {}).construct_solution_space(target).prune()
    return list(itertools.islice(space.enumerate_trees(target), count))


@pytest.fixture(scope="module")
def sample(specification):
    return _terms(specification, damg_targets.target_len_3, SAMPLE)


def _off_diagonal(matrix):
    values = np.asarray(matrix)
    return values[~np.eye(values.shape[0], dtype=bool)]


def _label_multisets(terms, level):
    labelings = (as_DAMG(hierarchical_tree(level)(t), verbose=True)[3] for t in terms)
    return {tuple(sorted(Counter(labeling.values()).items())) for labeling in labelings}


def test_each_kernel_runs_the_rounds_its_granularity_needs():
    """One round from granularity 2 on, none where the label carries the layer."""
    assert damg_kernel_1.h == 0
    assert damg_kernel_2.h == 1
    assert damg_kernel_3.h == 1
    assert wl_kernel_1.h == 0
    assert wl_kernel_2.h == 0
    assert wl_kernel_3.h == 0


def test_granularity_1_separates_without_a_round(sample):
    """The layer with its dimensions is label enough: forty terms, forty label multisets."""
    assert len(_label_multisets(sample, 1)) == SAMPLE
    off_diagonal = _off_diagonal(damg_kernel_1(sample))
    assert off_diagonal.max() < 1.0


@pytest.mark.parametrize("level", [2, 3])
def test_the_folded_granularities_leave_one_label_multiset(sample, level):
    """Every node is called ``node`` there, so a histogram counts nodes and nothing else."""
    assert len(_label_multisets(sample, level)) == 1


def _target(structure, inputs, outputs):
    return Constructor("Learner", Constructor("DAG",
                                              Constructor("input", Literal(inputs))
                                              & Constructor("output", Literal(outputs))
                                              & Constructor("structure", Literal(structure)))
                       & Constructor("Loss", Constructor("type", Literal(None)))
                       & Constructor("Optimizer", Constructor("type", Literal(None)))
                       & Constructor("epochs", Literal(2000)))


def _two_node_term(specification, target):
    """Return the first term of ``target`` whose graph has exactly two model nodes."""
    for term in _terms(specification, target, 40):
        labels = Counter(as_DAMG(hierarchical_tree(2)(term), verbose=True)[3].values())
        if labels["node"] == 2 and sum(labels.values()) == 6:
            return term
    raise AssertionError("no two-node term in the sample")


@pytest.fixture(scope="module")
def series_and_parallel(specification):
    """Two architectures over two nodes: one behind the other, and the two side by side."""
    series = _two_node_term(specification, damg_targets.target_len_2)
    parallel = _two_node_term(specification, _target(((None, None),), 2, 2))
    return [series, parallel]


def test_a_histogram_cannot_tell_series_from_parallel(series_and_parallel):
    """Without a round the two score exactly 1.0, which is why granularity 2 keeps one."""
    without = WeisfeilerLehmanKernel(h=0, normalize=True,
                                     to_grakel_graph=as_hierarchical_damg(2))
    assert without(series_and_parallel)[0][1] == pytest.approx(1.0)
    assert damg_kernel_2(series_and_parallel)[0][1] == pytest.approx(0.668, abs=5e-4)
    with_two = WeisfeilerLehmanKernel(h=2, normalize=True,
                                      to_grakel_graph=as_hierarchical_damg(2))
    assert with_two(series_and_parallel)[0][1] == pytest.approx(0.502, abs=5e-4)


@pytest.mark.parametrize(("level", "kernel"), [(2, wl_kernel_2), (3, wl_kernel_3)])
def test_the_tree_graph_kernels_still_separate_without_a_round(sample, level, kernel):
    """A round buys them little: 17.9 percent of the pairs tie without one, 13.8 with one."""
    with_one = WeisfeilerLehmanKernel(h=1, normalize=True,
                                      to_grakel_graph=as_hierarchical_tree_graph(level))
    tied_without = (_off_diagonal(kernel(sample)) == 1.0).mean()
    tied_with = (_off_diagonal(with_one(sample)) == 1.0).mean()
    assert tied_without == pytest.approx(0.179, abs=5e-4)
    assert tied_with == pytest.approx(0.138, abs=5e-4)


def test_granularity_1_of_the_tree_graph_kernel_ties_nothing(sample):
    """On the finest granularity the histogram tells every pair of the forty apart."""
    assert (_off_diagonal(wl_kernel_1(sample)) == 1.0).sum() == 0
