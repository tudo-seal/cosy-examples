"""A random forest in place of the Gaussian process.

``RandomForestSurrogate`` fits scikit-learn's forest on explicit features of the terms and answers
a posterior whose mean is the average of the trees' predictions and whose deviation their spread.
Each fit answers a posterior of its own and leaves the surrogate as it was, so the copies the loop
and its diagnostics keep stay what they were.  A value that is not finite is refused.  A spread of
zero is an answer the forest gives, and the scores read it as a point mass.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from cosy.core.tree import Tree

from bayesian_optimization import (
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
)
from bayesian_optimization.kernels.graph_features import WeisfeilerLehmanFeatures
from bayesian_optimization.random_forest import RandomForestPosterior, RandomForestSurrogate
from tests.integration.test_real_ea_loop import _objective
from tests.unit.test_caller_surrogate import _loop, _run
from tests.unit.test_graph_features import _terms


def _symbols(term: Tree) -> list[str]:
    return [term.root, *(symbol for child in term.children for symbol in _symbols(child))]


def _values(terms: list[Tree]) -> list[float]:
    """A value the labels determine: the share of ``neg`` among a term's symbols, plus its size."""
    values = []
    for term in terms:
        symbols = _symbols(term)
        values.append(symbols.count("neg") / len(symbols) + 0.1 * len(symbols))
    return values


def _surrogate(n_estimators: int = 50) -> RandomForestSurrogate:
    return RandomForestSurrogate(WeisfeilerLehmanFeatures(h=1), n_estimators=n_estimators)


def test_a_fit_answers_a_posterior_the_acquisitions_can_read():
    terms = _terms()
    posterior = _surrogate().fit(terms, _values(terms))
    mean, deviation = posterior.predict(terms, return_std=True)

    assert isinstance(posterior, RandomForestPosterior)
    assert mean.shape == deviation.shape == (len(terms),)
    assert np.all(np.isfinite(mean)) and np.all(deviation >= 0.0)
    assert np.array_equal(posterior.predict(terms), mean)


def test_the_mean_and_the_deviation_are_those_of_the_trees():
    terms = _terms()
    posterior = _surrogate().fit(terms[:8], _values(terms[:8]))
    features = posterior.features.transform(terms)
    per_tree = np.stack([tree.predict(features) for tree in posterior.forest.estimators_])
    mean, deviation = posterior.predict(terms, return_std=True)

    assert len(posterior.forest.estimators_) == 50
    assert np.allclose(mean, posterior.forest.predict(features), rtol=0.0, atol=1e-12)
    assert np.array_equal(mean, per_tree.mean(axis=0))
    assert np.array_equal(deviation, per_tree.std(axis=0))
    # the trees disagree on terms they were not fitted on, so the spread is not a constant zero
    assert deviation[8:].max() > 0.0


def test_the_forest_reads_the_features_of_its_own_fit():
    terms = _terms()
    posterior = _surrogate().fit(terms[:6], _values(terms[:6]))

    expected = WeisfeilerLehmanFeatures(h=1).fit(terms[:6]).vocabulary_
    assert posterior.features.vocabulary_ == expected
    assert posterior.forest.n_features_in_ == sum(map(len, expected))


def test_each_fit_answers_a_posterior_of_its_own_and_leaves_the_surrogate_unfitted():
    terms = _terms()
    surrogate = _surrogate()
    first = surrogate.fit(terms[:6], _values(terms[:6]))
    before = first.predict(terms, return_std=True)
    second = surrogate.fit(terms, [-value for value in _values(terms)])
    after = first.predict(terms, return_std=True)

    assert second is not first and second.features is not first.features
    assert not hasattr(surrogate.features, "vocabulary_")
    assert np.array_equal(before[0], after[0]) and np.array_equal(before[1], after[1])


def test_the_same_seed_fits_the_same_forest():
    terms = _terms()
    values = _values(terms)
    one = _surrogate().fit(terms[:8], values[:8]).predict(terms, return_std=True)
    two = _surrogate().fit(terms[:8], values[:8]).predict(terms, return_std=True)
    other = RandomForestSurrogate(WeisfeilerLehmanFeatures(h=1), n_estimators=50, random_state=1)
    three = other.fit(terms[:8], values[:8]).predict(terms, return_std=True)

    assert np.array_equal(one[0], two[0]) and np.array_equal(one[1], two[1])
    assert not np.array_equal(one[1], three[1])


def test_the_surrogate_and_its_posterior_survive_a_deep_copy():
    terms = _terms()
    surrogate = _surrogate()
    posterior = copy.deepcopy(surrogate).fit(terms, _values(terms))
    copied = copy.deepcopy(posterior)

    assert np.array_equal(
        copied.predict(terms, return_std=True)[1], posterior.predict(terms, return_std=True)[1]
    )


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_value_that_is_not_finite_is_refused(bad):
    terms = _terms()
    values = _values(terms)
    values[3] = bad
    with pytest.raises(ValueError, match="not finite"):
        _surrogate().fit(terms, values)


def test_terms_and_values_of_different_counts_are_refused():
    terms = _terms()
    with pytest.raises(ValueError, match="one value per term"):
        _surrogate().fit(terms, _values(terms)[:-1])


def test_a_fit_on_no_term_is_refused():
    with pytest.raises(ValueError, match="at least one"):
        _surrogate().fit([], [])


@pytest.mark.parametrize("n_estimators", [0, -3, 2.5, True])
def test_a_forest_has_a_positive_whole_number_of_trees(n_estimators):
    with pytest.raises(ValueError, match="trees"):
        RandomForestSurrogate(WeisfeilerLehmanFeatures(), n_estimators=n_estimators)


def test_the_scores_read_a_spread_of_zero_as_a_point_mass():
    """Every tree of a forest fitted on one value predicts that value: the spread is exactly zero,
    and the three scores answer what they answer for a certain value."""
    terms = _terms()
    posterior = _surrogate().fit(terms, [0.5] * len(terms))
    mean, deviation = posterior.predict(terms, return_std=True)
    assert np.all(deviation == 0.0) and np.all(mean == 0.5)

    def scores(acquisition) -> list[float]:
        # the batch path the evolutionary search scores through
        return list(acquisition.evaluate_batch(terms).values())

    assert np.allclose(scores(ExpectedImprovement(posterior, incumbent=0.4)), 0.1)
    assert scores(ExpectedImprovement(posterior, incumbent=0.6)) == [0.0] * len(terms)
    assert scores(ProbabilityOfImprovement(posterior, incumbent=0.4)) == [1.0] * len(terms)
    assert scores(ProbabilityOfImprovement(posterior, incumbent=0.6)) == [0.0] * len(terms)
    assert scores(UpperConfidenceBound(posterior, beta=2.0)) == [0.5] * len(terms)


class _Counting(RandomForestSurrogate):
    """The forest, counting the pairs each fit is handed."""

    def __init__(self) -> None:
        super().__init__(WeisfeilerLehmanFeatures(h=1), n_estimators=20)
        self.sizes: list[int] = []

    def fit(self, terms, values):  # type: ignore[no-untyped-def]
        self.sizes.append(len(list(terms)))
        return super().fit(terms, values)


def test_a_loop_runs_its_passes_against_the_forest():
    surrogate = _Counting()
    bo = _loop(surrogate)
    runs = _run(bo, n_design=4, n_passes=2)

    # one fit per pass, on the design and then on the design with the first pass's term
    assert surrogate.sizes == [4, 5]
    assert [type(run.acquisition.gp) for run in runs] == [RandomForestPosterior] * 2
    assert isinstance(bo.surrogate, RandomForestPosterior)
    assert len(bo.surrogate.forest.estimators_) == 20
    assert all(isinstance(term, Tree) for term in bo.design)
    assert len({_objective(term) for term in bo.design}) >= 1
