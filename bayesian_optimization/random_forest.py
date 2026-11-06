"""A random forest in place of the Gaussian process: the mean of its trees and their spread.

:class:`~bayesian_optimization.bo.BayesianOptimization` takes any
:class:`~bayesian_optimization.acquisition_function.Surrogate` as its ``surrogate_model``.  This one
fits scikit-learn's ``RandomForestRegressor`` on explicit features of the terms, a
:class:`~bayesian_optimization.kernels.graph_features.TermFeatures` map such as the label counts a
Weisfeiler-Lehman kernel compares, and answers a posterior whose mean is the average of the trees'
predictions and whose deviation is the standard deviation of those predictions.  Reading the trees'
spread as the model's uncertainty is SMAC's (Hutter, Hoos and Leyton-Brown, 2011, "Sequential
Model-Based Optimization for General Algorithm Configuration").  It is not a Gaussian posterior,
and the scores read the pair ``(m, s)`` as if it were one, as SMAC does.

**A fit answers a posterior of its own.**  The feature map is copied before it is fitted, so its
columns are the training terms' of that fit, and the surrogate itself never holds a fitted state:
the copies the loop keeps of every posterior, and the copy of the surrogate a diagnostic fits, stay
what they were whatever a later fit does.

**A spread of zero is an answer.**  Every tree of a forest fitted on a single value predicts that
value, and trees that agree on a term agree exactly.  The deviation is then zero, and the scores
read a zero deviation as a point mass (``_expected_gain`` and
:class:`~bayesian_optimization.acquisition_function.ProbabilityOfImprovement`).

**A value that is not finite is refused**, as ``BayesianOptimization.observe`` refuses it: a forest
fitted on it would answer a mean that is a number without being a measurement.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np
from cosy.core.tree import Tree
from sklearn.ensemble import RandomForestRegressor

from .kernels.graph_features import TermFeatures


def _finite_features(features: Any) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if not np.all(np.isfinite(matrix)):
        msg = "the feature map answered a feature that is not finite"
        raise ValueError(msg)
    return matrix


class RandomForestPosterior:
    """What a fit of :class:`RandomForestSurrogate` answers: a forest over its fitted feature map.

    Parameters
    ----------
    features:
        The feature map, fitted on the forest's training terms.
    forest:
        The fitted forest.
    """

    def __init__(self, features: TermFeatures, forest: RandomForestRegressor) -> None:
        self.features = features
        self.forest = forest

    def predict(self, X: Any, return_std: bool = False) -> Any:  # noqa: N803 - sklearn's name
        """Return the mean of the trees' predictions at each term, with their spread.

        Args:
            X: The terms.
            return_std: Also return the standard deviation of the trees' predictions.

        Returns:
            The means, or the means and the deviations, one per term.
        """
        terms: list[Tree[Any]] = list(X)
        if not terms:
            empty = np.zeros(0, dtype=float)
            return (empty, empty.copy()) if return_std else empty
        features = _finite_features(self.features.transform(terms))
        # the trees read float32, as the forest's own predict hands them; converted once here
        # rather than once per tree
        prepared = np.ascontiguousarray(features, dtype=np.float32)
        per_tree = np.stack(
            [tree.predict(prepared, check_input=False) for tree in self.forest.estimators_]
        )
        mean = per_tree.mean(axis=0)
        if not return_std:
            return mean
        return mean, per_tree.std(axis=0)


class RandomForestSurrogate:
    """A random forest on explicit features of the terms, as a surrogate of the loop.

    Parameters
    ----------
    features:
        The feature map the forest reads the terms by.  It is copied and fitted anew at every fit,
        on that fit's terms.
    n_estimators:
        How many trees the forest grows.
    random_state:
        The seed of the forest: the same seed and the same pairs grow the same trees.
    n_jobs:
        How many processes grow the trees, scikit-learn's parameter.
    """

    def __init__(
        self,
        features: TermFeatures,
        n_estimators: int = 500,
        random_state: int = 0,
        n_jobs: int | None = 1,
    ) -> None:
        if isinstance(n_estimators, bool) or not isinstance(n_estimators, int) or n_estimators < 1:
            msg = f"a forest grows a positive whole number of trees, not {n_estimators!r}"
            raise ValueError(msg)
        self.features = features
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, terms: Sequence[Any], values: Sequence[float]) -> RandomForestPosterior:
        """Fit a copy of the feature map and a forest on the pairs, and answer the posterior.

        Args:
            terms: The terms.
            values: Their values, one per term.

        Returns:
            RandomForestPosterior: The fitted forest over its fitted feature map.

        Raises:
            ValueError: If there is no term, if the counts of terms and values differ, or if a
                value or a feature is not finite.
        """
        terms = list(terms)
        targets = np.asarray(values, dtype=float).reshape(-1)
        if not terms:
            msg = "a forest is fitted on at least one term"
            raise ValueError(msg)
        if len(terms) != len(targets):
            msg = f"a forest is fitted on one value per term, not {len(targets)} for {len(terms)}"
            raise ValueError(msg)
        if not np.all(np.isfinite(targets)):
            msg = "a value the forest is to be fitted on is not finite; it is refused, not replaced"
            raise ValueError(msg)
        features = copy.deepcopy(self.features).fit(terms)
        forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(_finite_features(features.transform(terms)), targets)
        return RandomForestPosterior(features, forest)
