from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from cosy.core.tree import Tree
from scipy.stats import norm

# ---------------------------------------------------------------------------
# A posterior we can place by hand
# ---------------------------------------------------------------------------

class FixedPosterior:
    """A stand-in for the surrogate that reports a chosen ``(m, s)`` per term.

    All three scores consume ``t`` through the marginal posterior alone, so the geometry they
    describe, the level sets in the mean-deviation plane and the point mass at zero deviation, is
    a statement about that plane and not about any particular Gaussian process.  Placing points in
    the plane by hand is the only way to test it: a fitted GP hands out the pairs it happens to
    produce, and none of them sit on the boundaries that matter.
    """

    def __init__(self, posterior: dict[Any, tuple[float, float]]) -> None:
        self.posterior = posterior

    def predict(
        self, X: Any, return_std: bool = False
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        pairs = [self.posterior[t] for t in X]
        mean = np.array([m for m, _ in pairs], dtype=float)
        if not return_std:
            return mean
        return mean, np.array([s for _, s in pairs], dtype=float)


# ---------------------------------------------------------------------------
# The incumbent is data, not a property of the fitted model
# ---------------------------------------------------------------------------

def test_the_incumbent_cannot_be_read_off_the_fitted_gp(fitted_gp, tree_corpus):
    """``y*`` is the maximal observed value, and ``gp.y_train_`` is not it.

    ``normalize_y=True``, the default of this BO core, centers and scales the targets before
    storing them, while ``predict`` undoes that on the way out.  A default incumbent taken from
    ``gp.y_train_`` therefore compares a mean in the caller's scale against a threshold in the
    model's, and expected improvement is not linear in that difference, so the error is a change
    of ranking rather than an offset.  The parameter is required instead: the observations belong
    to the dataset ``D``, and the acquisition takes them from whoever holds it.
    """
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    stored = float(np.max(np.asarray(fitted_gp.y_train_)))
    observed = float(np.max(fitted_gp.predict(np.asarray(tree_corpus, dtype=object))))
    assert not np.isclose(stored, observed, atol=1e-6), (
        "the fixture must normalize for this test to bite"
    )

    with pytest.raises(TypeError):
        ExpectedImprovement(gp=fitted_gp)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# ExpectedImprovement
# ---------------------------------------------------------------------------

def test_ei_matches_its_closed_form(fitted_gp):
    """EI is ``(m - y*) Phi(z) + s phi(z)`` with ``z = (m - y*) / s``, and nothing else.

    Pinned against the definition, the expected gain over the incumbent ``E[max(g(t) - y*, 0)]``,
    rather than against the previous implementation, because what changed is exactly a term the
    definition does not have: the exploration margin xi used to be subtracted from the improvement
    without appearing in z, so this hand computation and the code disagreed by construction
    whenever xi was non-zero.
    """
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    novel = Tree("closed_form", (Tree("x"), Tree("y")))
    incumbent = 0.25
    ei = ExpectedImprovement(gp=fitted_gp, incumbent=incumbent)

    mu, sigma = fitted_gp.predict([novel], return_std=True)
    mu, sigma = float(mu[0]), float(sigma[0])
    assert sigma > 0.0, "the fixture must leave this candidate uncertain for the test to bite"
    improve = mu - incumbent
    z = improve / sigma
    expected = max(improve * norm.cdf(z) + sigma * norm.pdf(z), 0.0)

    assert np.isclose(ei(novel), expected, atol=1e-12)


def test_ei_at_zero_deviation_is_the_deterministic_gain():
    """``E[max(g(t) - y*, 0)]`` under a point mass is ``max(m - y*, 0)``, not zero.

    The previous implementation scored every candidate of zero posterior deviation at 0.0, which
    is right only where the mean also fails to beat the incumbent.  Clipping it there hides a
    certain improvement behind the same number an unpromising candidate gets.

    How a run reaches ``s = 0`` at all: the default jitter keeps it away (measured, at a training
    point: ~1.7e-3 for ``alpha=1e-6``), but a caller may pass ``alpha=0`` through ``gp_params``,
    and sklearn clips a numerically negative variance to exactly zero on the way out of
    ``predict``.
    """
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    above, below = Tree("certain_gain"), Tree("certain_loss")
    gp = FixedPosterior({above: (2.0, 0.0), below: (0.5, 0.0)})
    ei = ExpectedImprovement(gp=gp, incumbent=1.0)

    scores = ei.evaluate_batch([above, below])
    assert np.isclose(scores[above], 1.0, atol=1e-12)
    assert np.isclose(scores[below], 0.0, atol=1e-12)


def test_ei_ranks_known_points_below_every_novel_one(fitted_gp, tree_corpus):
    """An already-evaluated candidate must never outrank a novel one.

    This used to assert ``ei(t) == 0.0``, which pinned the defect: EI is clipped at zero, so a
    novel candidate may legitimately score exactly 0.0 too, and the ``max()`` tie-break then went
    to the point that had already been evaluated.  What matters is the ordering, not the number.
    """
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    known = set(tree_corpus)
    novel = [Tree("novel_a", (Tree("x"),)), Tree("novel_b", (Tree("y"), Tree("z")))]
    ei = ExpectedImprovement(gp=fitted_gp, incumbent=5.0, known_points=known)

    scores = ei.evaluate_batch([*tree_corpus, *novel])
    worst_novel = min(scores[t] for t in novel)
    best_known = max(scores[t] for t in tree_corpus)

    assert best_known < worst_novel
    assert all(np.isfinite(v) for v in scores.values())


def test_ei_scalar_matches_batch(fitted_gp):
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    ei = ExpectedImprovement(gp=fitted_gp, incumbent=5.0)
    novel = Tree("novel_for_batch_test", (Tree("x"),))
    assert np.isclose(ei(novel), ei.evaluate_batch([novel])[novel], atol=1e-10)


def test_ei_falls_as_the_incumbent_rises(fitted_gp):
    """Under maximization a low incumbent is easy to beat and a high one is not.

    The relation used to run the other way for the default configuration, because the class
    carried a ``greater_is_better`` flag that was false by default while the loop reported the
    result as a maximum.  With one convention there is one direction to test.
    """
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    novel = Tree("incumbent_direction", (Tree("x"),))
    easy = ExpectedImprovement(gp=fitted_gp, incumbent=-1000.0)
    hard = ExpectedImprovement(gp=fitted_gp, incumbent=1000.0)

    assert easy(novel) >= hard(novel)
    assert np.isclose(hard(novel), 0.0, atol=1e-9), "nothing beats an unbeatable incumbent"
    assert easy(novel) > 0.0


# ---------------------------------------------------------------------------
# ProbabilityOfImprovement
# ---------------------------------------------------------------------------

def test_pi_is_the_posterior_mass_above_the_threshold():
    """``Pr[g(t) > theta] = Phi((m - theta) / s)``, the standardized defining probability."""
    from bayesian_optimization.acquisition_function import ProbabilityOfImprovement

    t = Tree("pi_form")
    gp = FixedPosterior({t: (0.7, 0.4)})
    pi = ProbabilityOfImprovement(gp=gp, incumbent=0.5)

    assert np.isclose(pi(t), norm.cdf((0.7 - 0.5) / 0.4), atol=1e-12)


def test_pi_threshold_is_the_incumbent_plus_the_margin():
    """The threshold sits at the incumbent, or slightly above it, and the margin is that lift.

    This is where the exploration margin belongs.  It used to sit inside expected improvement,
    whose definition has no margin, and where subtracting it from the improvement without putting
    it into the standardization computes a quantity none of the three definitions name.
    """
    from bayesian_optimization.acquisition_function import ProbabilityOfImprovement

    t = Tree("pi_margin")
    gp = FixedPosterior({t: (0.7, 0.4)})

    at_incumbent = ProbabilityOfImprovement(gp=gp, incumbent=0.5)
    lifted = ProbabilityOfImprovement(gp=gp, incumbent=0.5, margin=0.1)
    shifted = ProbabilityOfImprovement(gp=gp, incumbent=0.6)

    assert np.isclose(at_incumbent(t), norm.cdf((0.7 - 0.5) / 0.4), atol=1e-12)
    assert np.isclose(lifted(t), norm.cdf((0.7 - 0.6) / 0.4), atol=1e-12)
    assert np.isclose(lifted(t), shifted(t), atol=1e-12)
    assert lifted(t) < at_incumbent(t), "a higher bar is harder to clear"


def test_pi_is_constant_on_rays_through_the_threshold():
    """In the mean-deviation plane, PI's level sets are the lines through ``(theta, 0)``.

    The argument of Phi is ``(m - theta) / s``, so two candidates whose offsets from the threshold
    stand in the same ratio to their deviations score identically, however far out they sit.
    """
    from bayesian_optimization.acquisition_function import ProbabilityOfImprovement

    near, far = Tree("near"), Tree("far")
    threshold = 1.0
    gp = FixedPosterior({near: (threshold + 0.2, 0.1), far: (threshold + 1.0, 0.5)})
    pi = ProbabilityOfImprovement(gp=gp, incumbent=threshold)

    scores = pi.evaluate_batch([near, far])
    assert np.isclose(scores[near], scores[far], atol=1e-12)


def test_pi_at_zero_deviation_is_a_point_mass():
    """A certain value is above the threshold or it is not, and ``Pr[g(t) > theta]`` is strict."""
    from bayesian_optimization.acquisition_function import ProbabilityOfImprovement

    above, at, below = Tree("above"), Tree("at"), Tree("below")
    gp = FixedPosterior({above: (1.5, 0.0), at: (1.0, 0.0), below: (0.5, 0.0)})
    pi = ProbabilityOfImprovement(gp=gp, incumbent=1.0)

    scores = pi.evaluate_batch([above, at, below])
    assert scores[above] == 1.0
    assert scores[at] == 0.0
    assert scores[below] == 0.0


def test_pi_rejects_a_margin_below_zero():
    """A threshold at or above the incumbent leaves no room for a negative margin."""
    from bayesian_optimization.acquisition_function import ProbabilityOfImprovement

    with pytest.raises(ValueError, match="margin"):
        ProbabilityOfImprovement(gp=FixedPosterior({}), incumbent=0.0, margin=-0.1)


def test_pi_ranks_known_points_below_every_novel_one(fitted_gp, tree_corpus):
    from bayesian_optimization.acquisition_function import ProbabilityOfImprovement

    known = set(tree_corpus)
    novel = [Tree("pi_novel_a", (Tree("x"),)), Tree("pi_novel_b", (Tree("y"),))]
    pi = ProbabilityOfImprovement(gp=fitted_gp, incumbent=5.0, known_points=known)

    scores = pi.evaluate_batch([*tree_corpus, *novel])
    assert max(scores[t] for t in tree_corpus) < min(scores[t] for t in novel)
    assert all(np.isfinite(v) for v in scores.values())


# ---------------------------------------------------------------------------
# UpperConfidenceBound
# ---------------------------------------------------------------------------

def test_ucb_is_mean_plus_beta_deviation(fitted_gp):
    """The score is ``m + beta s``, with beta the exploration parameter of the definition.

    Beta is not a kappa and it is not a call argument.  It is a parameter of the score, fixed
    where the score is built.
    """
    from bayesian_optimization.acquisition_function import UpperConfidenceBound

    novel = Tree("ucb_form", (Tree("v"),))
    mu, sigma = fitted_gp.predict([novel], return_std=True)
    mu, sigma = float(mu[0]), float(sigma[0])

    assert np.isclose(
        UpperConfidenceBound(gp=fitted_gp, beta=2.0)(novel), mu + 2.0 * sigma, atol=1e-9
    )
    assert np.isclose(
        UpperConfidenceBound(gp=fitted_gp, beta=0.5)(novel), mu + 0.5 * sigma, atol=1e-9
    )


def test_ucb_rejects_a_non_positive_beta(fitted_gp):
    """An upper confidence bound requires ``beta > 0``: at zero the score stops being an
    optimistic estimate, and below it the bound points the wrong way."""
    from bayesian_optimization.acquisition_function import UpperConfidenceBound

    for beta in (0.0, -1.0):
        with pytest.raises(ValueError, match="beta"):
            UpperConfidenceBound(gp=fitted_gp, beta=beta)


def test_ucb_is_constant_along_lines_of_slope_minus_one_over_beta():
    """In the mean-deviation plane, ``m + beta s = c`` is the line ``s = (c - m) / beta``."""
    from bayesian_optimization.acquisition_function import UpperConfidenceBound

    beta = 2.0
    low, high = Tree("low_mean"), Tree("high_mean")
    gp = FixedPosterior({low: (0.2, 0.4), high: (0.2 + beta * 0.3, 0.1)})
    ucb = UpperConfidenceBound(gp=gp, beta=beta)

    scores = ucb.evaluate_batch([low, high])
    assert np.isclose(scores[low], scores[high], atol=1e-12)


def test_ucb_ranks_known_points_below_every_novel_one(fitted_gp, tree_corpus):
    """The floor has to sit below scores that are unbounded below, and a fixed 0.0 did not."""
    from bayesian_optimization.acquisition_function import UpperConfidenceBound

    known = set(tree_corpus)
    novel = [Tree("ucb_novel_a", (Tree("x"),)), Tree("ucb_novel_b", (Tree("y"),))]
    ucb = UpperConfidenceBound(gp=fitted_gp, known_points=known, beta=2.0)

    scores = ucb.evaluate_batch([*tree_corpus, *novel])
    assert max(scores[t] for t in tree_corpus) < min(scores[t] for t in novel)
    assert all(np.isfinite(v) for v in scores.values())


def test_ucb_scalar_matches_batch(fitted_gp):
    from bayesian_optimization.acquisition_function import UpperConfidenceBound

    ucb = UpperConfidenceBound(gp=fitted_gp, beta=1.5)
    novel = Tree("ucb_batch", (Tree("x"),))
    assert np.isclose(ucb(novel), ucb.evaluate_batch([novel])[novel], atol=1e-10)


# ---------------------------------------------------------------------------
# What the three share
# ---------------------------------------------------------------------------

def _all_three(gp, known=None):
    from bayesian_optimization.acquisition_function import (
        ExpectedImprovement,
        ProbabilityOfImprovement,
        UpperConfidenceBound,
    )

    return (
        ExpectedImprovement(gp=gp, incumbent=1.0, known_points=known),
        ProbabilityOfImprovement(gp=gp, incumbent=1.0, known_points=known),
        UpperConfidenceBound(gp=gp, beta=2.0, known_points=known),
    )


def test_a_non_tree_is_rejected_rather_than_scored(fitted_gp):
    """Something that is not a term has no posterior, and it must not be given a number anyway.

    The previous behavior dropped ``None`` from a batch and answered 0.0 for it when asked alone.
    Zero is a score an acquisition can genuinely produce, so the mistake arrived at the
    evolutionary algorithm as an ordinary fitness value, the same failure mode the fixed
    known-point sentinel had, and the repository rule against substitute values exists for it.
    """
    for af in _all_three(fitted_gp):
        with pytest.raises(TypeError):
            af(None)
        with pytest.raises(TypeError):
            af("not_a_tree")
        with pytest.raises(TypeError):
            af.evaluate_batch([Tree("valid_node"), None])


def test_a_broken_posterior_is_not_scored(fitted_gp):
    """A deviation that is not a number, or below zero, is a broken surrogate and says so."""
    t = Tree("broken")

    for bad in ((0.5, float("nan")), (float("nan"), 0.5), (0.5, -1.0), (float("inf"), 0.5)):
        gp = FixedPosterior({t: bad})
        for af in _all_three(gp):
            with pytest.raises(ValueError):
                af(t)


def test_one_broken_entry_spoils_the_whole_batch(fitted_gp):
    """The check is over the batch, not over some of it.

    Asked one candidate at a time, ``all`` and ``any`` agree, so a check written with the wrong
    one passes every single-candidate test and then lets a ``nan`` through in the company of
    healthy candidates, where it reaches the evolutionary algorithm as an ordinary fitness.
    """
    healthy_a, broken, healthy_b = Tree("ok_a"), Tree("spoiled"), Tree("ok_b")

    for bad in ((float("nan"), 0.4), (0.5, float("nan")), (0.5, -1.0)):
        gp = FixedPosterior({healthy_a: (0.5, 0.4), broken: bad, healthy_b: (0.6, 0.3)})
        for af in _all_three(gp):
            with pytest.raises(ValueError):
                af.evaluate_batch([healthy_a, broken, healthy_b])


def test_a_surrogate_that_answers_the_wrong_number_of_values_is_rejected():
    """A mismatch between means and deviations must not be indexed into silently."""
    from bayesian_optimization.acquisition_function import ExpectedImprovement

    a, b = Tree("a"), Tree("b")

    class ShortDeviation(FixedPosterior):
        def predict(self, X, return_std=False):
            mean, deviation = super().predict(X, return_std=True)
            return mean, deviation[:-1]

    gp = ShortDeviation({a: (0.5, 0.4), b: (0.6, 0.3)})
    with pytest.raises(ValueError):
        ExpectedImprovement(gp=gp, incumbent=0.0).evaluate_batch([a, b])


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_a_parameter_that_is_not_finite_is_rejected(fitted_gp, bad):
    """An infinite incumbent or threshold makes every score ``nan``, and a nan makes them zero.

    Neither is a measurement that failed.  It is a parameter that was never a number, and the
    class says so at construction rather than handing out scores computed from it.
    """
    from bayesian_optimization.acquisition_function import (
        ExpectedImprovement,
        ProbabilityOfImprovement,
        UpperConfidenceBound,
    )

    with pytest.raises(ValueError):
        ExpectedImprovement(gp=fitted_gp, incumbent=bad)
    with pytest.raises(ValueError):
        ProbabilityOfImprovement(gp=fitted_gp, incumbent=bad)
    with pytest.raises(ValueError):
        ProbabilityOfImprovement(gp=fitted_gp, incumbent=0.0, margin=bad)
    with pytest.raises(ValueError):
        UpperConfidenceBound(gp=fitted_gp, beta=bad)


def test_an_empty_batch_scores_nothing(fitted_gp):
    for af in _all_three(fitted_gp):
        assert af.evaluate_batch([]) == {}


# ---------------------------------------------------------------------------
# The known-point floor
# ---------------------------------------------------------------------------

def test_the_floor_of_a_bounded_acquisition_does_not_move(fitted_gp):
    """EI is clipped at zero and PI is a probability, so their floor needs no comparison batch.

    A floor derived from the scores at hand is only as good as what happened to be scored, and at
    the first call there is nothing.  Where the definition bounds the score, the floor is fixed.
    """
    from bayesian_optimization.acquisition_function import (
        ExpectedImprovement,
        ProbabilityOfImprovement,
    )

    for af in (
        ExpectedImprovement(gp=fitted_gp, incumbent=1.0),
        ProbabilityOfImprovement(gp=fitted_gp, incumbent=1.0),
    ):
        assert af.lower_bound == 0.0
        # The comparison scores are deliberately away from zero: with a batch whose minimum is 0
        # the bounded and the relative floor coincide, and an implementation that ignored the
        # bound would pass unnoticed.
        assert af.known_point_floor(()) == af.known_point_floor((0.5, 5.0)) == -1.0


def test_the_floor_of_an_unbounded_acquisition_undercuts_the_scores_it_is_given(fitted_gp):
    """UCB has no lower bound of its own, so its floor comes from what was actually scored,
    including the empty case, which must still land below zero."""
    from bayesian_optimization.acquisition_function import UpperConfidenceBound

    ucb = UpperConfidenceBound(gp=fitted_gp, beta=2.0)
    assert ucb.lower_bound is None
    assert ucb.known_point_floor(()) < 0.0
    assert ucb.known_point_floor((-8.0, -3.0)) < -8.0
    assert ucb.known_point_floor((2.0, 7.0)) < 2.0


def test_the_floor_stays_strictly_below_even_where_subtracting_one_does_nothing(fitted_gp):
    """Past float64's integer resolution ``x - 1.0 == x``, and "strictly below" would be a lie."""
    from bayesian_optimization.acquisition_function import UpperConfidenceBound

    huge = -1e17
    floor = UpperConfidenceBound(gp=fitted_gp, beta=2.0).known_point_floor((huge,))
    assert floor < huge


# ---------------------------------------------------------------------------
# The one thing the base class does not supply
# ---------------------------------------------------------------------------

def test_an_acquisition_that_states_no_score_is_refused_where_it_would_be_used():
    """The base class carries everything but the map from the posterior to a number.

    Asking the surrogate, checking what it answered and flooring the known points are shared, and
    a subclass adds the score.  One that adds nothing has to say so at the first candidate.  A
    base returning zeros instead would give every candidate the same score, and the maximization
    would answer with whichever candidate its search happened to reach first.
    """
    from bayesian_optimization.acquisition_function import AcquisitionFunction

    term = Tree("t", ())
    incomplete = AcquisitionFunction(FixedPosterior({term: (1.0, 0.5)}))

    with pytest.raises(NotImplementedError, match="implement score"):
        incomplete.evaluate_batch([term])
    with pytest.raises(NotImplementedError, match="implement score"):
        incomplete(term)
