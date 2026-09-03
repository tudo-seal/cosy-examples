from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from cosy.core.tree import Tree
from scipy.stats import norm


class MarginalPosterior(Protocol):
    """What an acquisition function needs from a surrogate, and all that it needs.

    All three scores consume a term through its marginal posterior alone, so they are defined
    wherever the surrogate is.  Naming that as a protocol rather than as
    ``GaussianProcessRegressor`` is not genericity for its own sake: it is the whole interface the
    scores use, and it keeps a test able to place a candidate at a chosen ``(m, s)`` instead of
    hoping a fitted model produces one there.
    """

    def predict(self, X: Any, return_std: bool = ...) -> Any:
        """Return the posterior mean at each element of ``X``, with the standard deviation."""
        ...


def require_term(item: Any) -> Tree[Any]:
    """Return a candidate as a term, or say that it is not one.

    Args:
        item (Any): The candidate.

    Returns:
        Tree[Any]: The candidate.

    Raises:
        TypeError: If it is not a ``Tree``.  The acquisition layer used to drop such an entry
            from a batch and answer ``0.0`` for it when asked alone.  That is a score an
            acquisition can genuinely produce, so the mistake reached the evolutionary algorithm
            as an ordinary fitness value.  Shared with the optimizer adapter so that both say the
            same thing.
    """
    if not isinstance(item, Tree):
        msg = f"an acquisition function scores terms, not {type(item).__name__}: {item!r}"
        raise TypeError(msg)
    return item


class AcquisitionFunction:
    """Base class for the three standard acquisition functions, all of them maximized.

    Every acquisition is **maximized**, and so is the objective behind it: the incumbent ``y*`` is
    the largest observed value, and an acquisition optimizer reads the scores as a fitness.  A
    caller who wants to minimize negates the objective on the way in and the result on the way
    out.  Carrying a ``greater_is_better`` flag through the class instead put the same decision in
    two places, the sign of the improvement term and the choice of incumbent, and one of the two
    was read by the fallback floor below, which is how an already-evaluated point could outrank
    every genuine candidate.

    A subclass supplies :meth:`score`, a map from the marginal posterior to a real number.
    Everything else is shared: asking the surrogate, checking what it answered, and keeping
    already-evaluated points out of the running.  The three definitions differ in that map and in
    nothing else.

    **The known-point floor is a deliberate deviation from the thesis.**  The optimization loop
    there allows a repeated evaluation and de-duplicates only when conditioning, which uses the
    distinct observed pairs because an evaluation may repeat, and the thesis marks its duplicate
    handling as an unverified design decision with rejection named as the alternative.  This
    implementation takes the rejection branch: an already-evaluated candidate is scored below
    every genuine score, so the acquisition maximization cannot return it, and ``suggest()``
    rejects it once more if it does.  The deviation is intentional and is to be resolved on the
    thesis side rather than here.

    Args:
        gp (MarginalPosterior): The fitted surrogate.
        known_points (set[Any] | None): Candidates the loop has already evaluated.  They are
            scored strictly below every genuine score, so they can never win the optimization.
    """

    lower_bound: float | None = None
    """The smallest score this acquisition can produce, where its definition fixes one.

    Expected improvement is clipped at zero and probability of improvement is a probability, so
    both are bounded below by 0.  The upper confidence bound is ``m + beta s`` with an unbounded
    posterior mean, so it has none, and its floor has to be derived from the scores it actually
    produced.  That is what makes the floor order-dependent, and why the single-sample objective
    of the optimizer adapter refuses to carry it.
    """

    def __init__(self, gp: MarginalPosterior, *, known_points: set[Any] | None = None) -> None:
        self.gp = gp
        self.known_points = known_points

    def score(self, mean: np.ndarray, deviation: np.ndarray) -> np.ndarray:
        """Score candidates from their posterior mean and standard deviation.

        Args:
            mean (np.ndarray): The posterior means ``m_D(t)``.
            deviation (np.ndarray): The posterior standard deviations ``s_D(t)``, non-negative.

        Returns:
            np.ndarray: One score per candidate.

        Raises:
            NotImplementedError: Always, because a subclass states the score.
        """
        raise NotImplementedError("Subclasses must implement score().")

    def evaluate_batch(self, trees: Sequence[Any]) -> dict[Tree[Any], float]:
        """Score a batch of candidates in one call to the surrogate.

        Args:
            trees (Sequence[Any]): The candidates.  Every entry must be a term.

        Returns:
            dict[Tree[Any], float]: The score of each candidate.  Already-evaluated candidates
                receive the floor of this batch rather than a score of their own.

        Raises:
            TypeError: If an entry is not a ``Tree``.
            ValueError: If the surrogate answers with a broken posterior.
        """
        if not trees:
            return {}

        result: dict[Tree[Any], float] = {}
        candidates, known = self._split_known(trees)

        if candidates:
            mean, deviation = self.marginal_posterior(candidates)
            scores = np.asarray(self.score(mean, deviation), dtype=float).reshape(-1)
            for tree, value in zip(candidates, scores, strict=True):
                result[tree] = float(value)

        # Known points are scored only once the genuine values are known, so the floor can sit
        # below them.  EI is clipped at zero and PI is a probability, so a novel candidate may
        # score exactly 0.0, and the previous sentinel of 0.0 then won the max() tie-break for
        # an already-evaluated point.
        if known:
            floor = self.known_point_floor([result[tree] for tree in candidates])
            for tree in known:
                result[tree] = floor

        return result

    def __call__(self, t: Any) -> float:
        """Score a single candidate.

        Args:
            t (Any): The candidate.

        Returns:
            float: Its score.  An already-evaluated candidate receives the floor of a batch that
                holds no genuine score, which is below zero.

        Raises:
            TypeError: If ``t`` is not a ``Tree``.
            ValueError: If the surrogate answers with a broken posterior.
        """
        return self.evaluate_batch([t])[t]

    def marginal_posterior(
        self, candidates: Sequence[Tree[Any]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Ask the surrogate for the marginal posterior, and check what comes back.

        A mean that is not a number, or a negative standard deviation, is a broken surrogate
        rather than a candidate that scores badly.  Scoring it anyway propagates a ``nan`` into
        the evolutionary run as an ordinary fitness value, where it competes for selection.

        It is public because the pair ``(m_D, s_D)`` is what the scores consume and what a plot of
        the candidates in the mean-deviation plane shows, so the frontier read asks for it here
        rather than calling the surrogate a second way and checking the answer a second time.

        Args:
            candidates (Sequence[Tree[Any]]): The candidates to predict at.

        Returns:
            tuple[np.ndarray, np.ndarray]: The means and the standard deviations.

        Raises:
            ValueError: If the surrogate returns the wrong number of values, a non-finite value,
                or a negative standard deviation.
        """
        mean_raw, deviation_raw = self.gp.predict(candidates, return_std=True)
        mean = np.asarray(mean_raw, dtype=float).reshape(-1)
        deviation = np.asarray(deviation_raw, dtype=float).reshape(-1)

        expected = (len(candidates),)
        if mean.shape != expected or deviation.shape != expected:
            msg = (
                f"the surrogate answered {mean.shape} means and {deviation.shape} deviations "
                f"for {len(candidates)} candidates"
            )
            raise ValueError(msg)
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(deviation)):
            msg = "the surrogate returned a posterior that is not finite"
            raise ValueError(msg)
        if np.any(deviation < 0.0):
            msg = "the surrogate returned a negative posterior standard deviation"
            raise ValueError(msg)
        return mean, deviation

    def _split_known(
        self, trees: Sequence[Any]
    ) -> tuple[list[Tree[Any]], list[Tree[Any]]]:
        """Split a batch into candidates to score and already-evaluated points.

        Args:
            trees (Sequence[Any]): The batch.

        Returns:
            tuple[list[Tree[Any]], list[Tree[Any]]]: The novel candidates and the known ones.

        Raises:
            TypeError: If an entry is not a ``Tree``.  See :func:`require_term`.
        """
        candidates: list[Tree[Any]] = []
        known: list[Tree[Any]] = []
        for entry in trees:
            item = require_term(entry)
            if self.known_points is not None and item in self.known_points:
                known.append(item)
            else:
                candidates.append(item)
        return candidates, known

    def known_point_floor(self, novel_scores: Sequence[float]) -> float:
        """Return the score an already-evaluated candidate gets.

        It must sit strictly below every genuine score, so that a known point can never win the
        optimization.  A fixed ``0.0`` achieved that only for expected improvement, which is
        clipped at zero.  Upper confidence bound scores are unbounded below, so ``0.0`` was the
        *maximizer* whenever the scores were negative, and the evolutionary algorithm steered
        straight at points that had already been evaluated.

        Where :attr:`lower_bound` is known, the floor is derived from it and is therefore the same
        number whatever else was scored.  That matters beyond tidiness: a floor computed from the
        scores seen *so far* changes with the order in which the evolutionary algorithm asks, and
        the first candidate of a run is asked with nothing to sit below.

        The floor is finite on purpose.  Proportional selection reads these scores through the
        exponential map, which is a scalarization because it is positive and order-preserving, and
        ``exp`` sends ``-inf`` to ``0.0``.  cosy's ``_proportional_weights`` rejects that: a
        scalarization maps into the *positive* reals, so a weight of zero is a broken component
        contract there, not a member that is never drawn.

        Args:
            novel_scores (Sequence[float]): The genuine scores available to compare against.  Used
                only where the acquisition has no lower bound of its own.  May be empty.

        Returns:
            float: A finite value strictly below all of them, and below zero when there are none,
                so a lone known point does not score zero either.
        """
        reference = (
            self.lower_bound if self.lower_bound is not None else min(novel_scores, default=0.0)
        )
        floor = reference - 1.0
        if not floor < reference:
            # Beyond float64's integer resolution, subtracting one is a no-op and "strictly below"
            # would become "equal to".  One ulp is a thin margin, but it is the widest one the
            # representation has left at that magnitude.
            floor = float(np.nextafter(reference, -np.inf))
        return floor


def _require_finite(name: str, value: float) -> float:
    """Return a parameter as a float, or say that it is not one.

    Args:
        name (str): The parameter's name, for the message.
        value (float): The value handed in.

    Returns:
        float: The value.

    Raises:
        ValueError: If the value is not finite.  ``nan`` in an incumbent or a threshold makes
            every comparison below it false, so a whole run would score zero without a symptom.
    """
    number = float(value)
    if not np.isfinite(number):
        msg = f"{name} must be finite: {value}"
        raise ValueError(msg)
    return number


class ExpectedImprovement(AcquisitionFunction):
    """Expected improvement: the expected gain over the incumbent, ``E[max(g(t) - y*, 0) | D]``.

    The closed form is ``(m - y*) Phi(z) + s phi(z)`` with ``z = (m - y*) / s``.  There is no
    exploration margin: the expectation already counts every outcome below the incumbent as zero,
    and the margin that does appear among the three scores belongs to
    :class:`ProbabilityOfImprovement`, where it is the threshold ``theta``.  A margin here shifts
    the improvement term without appearing in the standardization, and computes no quantity that
    either definition asks for.

    Args:
        gp (MarginalPosterior): The fitted surrogate.
        incumbent (float): The maximal observed value ``y*``.  It is required, and it is not read
            off the model: with ``normalize_y=True`` sklearn stores its targets centered and
            scaled while ``predict`` returns them in the caller's scale, so a default taken from
            ``gp.y_train_`` would compare the two across a change of units.  The incumbent belongs
            to the dataset, and the dataset belongs to the loop.
        known_points (set[Any] | None): Candidates the loop has already evaluated.
    """

    lower_bound = 0.0

    def __init__(
        self,
        gp: MarginalPosterior,
        *,
        incumbent: float,
        known_points: set[Any] | None = None,
    ) -> None:
        super().__init__(gp, known_points=known_points)
        self.incumbent = _require_finite("the incumbent", incumbent)

    def score(self, mean: np.ndarray, deviation: np.ndarray) -> np.ndarray:
        """Return the expected gain over the incumbent.

        Args:
            mean (np.ndarray): The posterior means.
            deviation (np.ndarray): The posterior standard deviations.

        Returns:
            np.ndarray: The expected improvement of each candidate, never negative.
        """
        improvement = mean - self.incumbent
        result = np.empty_like(improvement)

        # Where the value is certain, the expectation over a point mass is the gain itself.  The
        # previous implementation scored all of those 0.0, which is right only where the mean
        # fails to beat the incumbent as well.  Two routes reach ``s = 0`` in practice: a caller
        # who turns the numerical diagonal off (measured: at the default ``alpha=1e-6`` the
        # deviation at a training point is ~1.7e-3, at ``alpha=0`` it is exactly zero), and
        # sklearn clipping a numerically negative variance to zero on the way out.
        certain = deviation == 0.0
        result[certain] = np.maximum(improvement[certain], 0.0)

        uncertain = ~certain
        z = improvement[uncertain] / deviation[uncertain]
        result[uncertain] = (
            improvement[uncertain] * norm.cdf(z) + deviation[uncertain] * norm.pdf(z)
        )
        return np.maximum(result, 0.0)


class ProbabilityOfImprovement(AcquisitionFunction):
    """Probability of improvement: the posterior mass above a threshold, ``Pr[g(t) > theta | D]``.

    In closed form the score is ``Phi((m - theta) / s)``.  The threshold sits at the incumbent, or
    slightly above it, and ``margin`` is that lift.  It is the only exploration margin among the
    three scores.  Probability of improvement is more risk-averse than expected improvement, since
    it rewards a certain small gain over an uncertain large one, and the margin above the
    incumbent counters that.

    Args:
        gp (MarginalPosterior): The fitted surrogate.
        incumbent (float): The maximal observed value ``y*``.  See :class:`ExpectedImprovement`
            for why it is required rather than read off the model.
        known_points (set[Any] | None): Candidates the loop has already evaluated.
        margin (float): How far above the incumbent the threshold sits.  Zero puts it at the
            incumbent, which is the other reading the definition allows.
    """

    lower_bound = 0.0

    def __init__(
        self,
        gp: MarginalPosterior,
        *,
        incumbent: float,
        known_points: set[Any] | None = None,
        margin: float = 0.0,
    ) -> None:
        super().__init__(gp, known_points=known_points)
        self.incumbent = _require_finite("the incumbent", incumbent)
        self.margin = _require_finite("the margin", margin)
        if self.margin < 0.0:
            msg = (
                "the threshold sits at the incumbent or slightly above it, so the margin cannot "
                f"be negative: {margin}"
            )
            raise ValueError(msg)

    @property
    def threshold(self) -> float:
        """The threshold ``theta`` a value has to exceed, the incumbent lifted by the margin."""
        return self.incumbent + self.margin

    def score(self, mean: np.ndarray, deviation: np.ndarray) -> np.ndarray:
        """Return the posterior probability of exceeding the threshold.

        Args:
            mean (np.ndarray): The posterior means.
            deviation (np.ndarray): The posterior standard deviations.

        Returns:
            np.ndarray: One probability per candidate.
        """
        excess = mean - self.threshold
        result = np.empty_like(excess)

        # Improvement is a strict excess, so a value that is certain to land *on* the threshold
        # improves on it with probability zero.
        certain = deviation == 0.0
        result[certain] = np.where(excess[certain] > 0.0, 1.0, 0.0)

        uncertain = ~certain
        result[uncertain] = norm.cdf(excess[uncertain] / deviation[uncertain])
        return result


class UpperConfidenceBound(AcquisitionFunction):
    """Upper confidence bound: the posterior mean lifted by ``beta`` deviations, ``m + beta * s``.

    The score is an optimistic estimate: a quantile of the Gaussian value at ``t``, which the
    value exceeds only with a small probability tuned by ``beta``, so a larger ``beta`` rewards
    uncertainty more.  There is no incumbent in it.  The parameter used to be carried here "for
    API consistency", which is an invariant a caller has to be told rather than one the signature
    states.

    Args:
        gp (MarginalPosterior): The fitted surrogate.
        beta (float): The exploration parameter, strictly positive.
        known_points (set[Any] | None): Candidates the loop has already evaluated.
    """

    def __init__(
        self,
        gp: MarginalPosterior,
        *,
        beta: float = 2.0,
        known_points: set[Any] | None = None,
    ) -> None:
        super().__init__(gp, known_points=known_points)
        self.beta = _require_finite("beta", beta)
        if self.beta <= 0.0:
            msg = (
                "the exploration parameter beta must be strictly positive: at zero the score "
                "stops being an optimistic estimate, and below it the bound points the wrong "
                f"way: {beta}"
            )
            raise ValueError(msg)

    def score(self, mean: np.ndarray, deviation: np.ndarray) -> np.ndarray:
        """Return the upper confidence bound.

        Args:
            mean (np.ndarray): The posterior means.
            deviation (np.ndarray): The posterior standard deviations.

        Returns:
            np.ndarray: One bound per candidate.
        """
        return np.asarray(mean + self.beta * deviation, dtype=float)
