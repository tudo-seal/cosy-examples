from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..acquisition_function import AcquisitionFunction, require_term
from ._statistics import spread


@dataclass(frozen=True)
class GenerationRecord:
    """What one generation of the acquisition-optimizing search did.

    The frontier read looks at the *last* generation, which answers whether the returned term is
    the best of what the run ended with.  It cannot answer whether the run **improved**, and that
    is a separate claim, about the inner algorithm rather than about its answer.  A sequence of
    these is that claim's evidence: a survivor selection that always keeps a fittest member makes
    the best-so-far ``b`` monotone, and a run whose ``best`` is flat from generation 0 is one
    whose variation operators never produced anything the selection kept.

    All of it is read off the state the driver already yields, so recording it costs the driver
    nothing it was not doing anyway.

    Attributes:
        generation (int): The generation number.  The initial population is 0.
        best (float): The fitness of ``b``, the fittest individual of the whole run so far.  Under
            a total order this is monotone by construction, which is what makes a *drop* here a
            defect rather than a fluctuation.
        population_best (float): The fittest member of *this* generation, which may be worse than
            ``best``: the survivor selection is generous, so it keeps individuals a truncation
            would drop.
        population_mean (float): The mean fitness of this generation, the blunt measure of whether
            selection pressure is reaching the whole population.
        population_worst (float): Its least fit member.  Read beside the mean: a population whose
            worst equals its best has collapsed.
        distinct_members (int): How many of the ``mu`` members are distinct terms.
        last_improvement (int): The generation in which ``b`` was last replaced.  The distance
            from :attr:`generation` is how long the run has been stalled.
        offspring (int): How many individuals variation produced in this generation.  Zero says
            every pass was discarded: the acceptance test rejected everything the operators built,
            which is invisible in the fitness columns.
    """

    generation: int
    best: float
    population_best: float
    population_mean: float
    population_worst: float
    distinct_members: int
    last_improvement: int
    offspring: int


@dataclass(frozen=True)
class AcquisitionRun:
    """One acquisition maximization, kept whole.

    The frontier read needs three things that only make sense together, and taking them as three
    arguments invited the mistake this class exists to prevent: an acquisition is built fresh each
    pass, from that pass's surrogate, incumbent and known points, so pairing a population with any
    other acquisition scores it by a function no maximization ever ran.  Measured on a four-pass
    run over 144 terms: read against its own acquisition the returned term is outscored by nobody,
    read against one rebuilt from the surrogate over the whole dataset by seventeen of twenty.

    Attributes:
        acquisition (AcquisitionFunction): What the run maximized.  It must be the object the run
            used, and it must not be sharing mutable state with the loop that built it, or it
            stops describing that pass the moment the loop moves on.
        pick (Any): The term the maximization **returned**.  Not necessarily the term the loop
            went on to evaluate: where ``fallback_used`` is true the loop rejected this one as a
            duplicate and drew a replacement.  The frontier read diagnoses the maximization, so it
            reads this one either way.  The replacement was never scored against this population.
        population (tuple[Any, ...]): The final generation, in the order the driver held it.
        fallback_used (bool): Whether the loop replaced :attr:`pick` afterwards.  It changes
            nothing the read computes and everything about what a caller should conclude: a run
            that falls back every pass is random search, and the frontier of its maximizations is
            not what is wrong with it.
        generations (tuple[GenerationRecord, ...]): The run's trajectory, one entry per
            generation.  Empty where the maximization was not asked to record it, which is a
            different statement from a run of no generations: every run yields at least the
            zeroth.
    """

    acquisition: AcquisitionFunction
    pick: Any
    population: tuple[Any, ...]
    fallback_used: bool = False
    generations: tuple[GenerationRecord, ...] = ()


@dataclass(frozen=True)
class FrontierRead:
    """Where the returned term sits among the population that produced it.

    Attributes:
        size (int): How many individuals the final population holds.
        distinct_members (int): How many of them are distinct terms.  A collapsed population
            repeats itself, and the ratio to :attr:`size` is the bluntest measure of it.
        pick_in_population (bool): Whether the returned term is a member of the final population
            at all.  It need not be: cosy's driver carries the fittest individual of the **whole
            run** (``b``), so a term a later generation dropped is still the answer if nothing
            beat it.  Where this is false the frontier question below is still meaningful, since a
            point is on the frontier or dominated whether or not it is a member, but a false here
            plus a dominated pick says the run got worse, not that the selection is broken.
        known_members (int): How many members the acquisition holds as already evaluated.  Those
            were scored at the floor during the maximization, not at :attr:`points`' score, so
            their position in the plane does not explain their fate.
        mean_at_pick (float): ``m_D`` at the returned term.
        deviation_at_pick (float): ``s_D`` at it.
        score_at_pick (float): The acquisition value there, the genuine score, without the
            known-point floor, so that a pick and a member are compared by the same map.
        fallback_used (bool): Whether the loop rejected :attr:`AcquisitionRun.pick` as a duplicate
            afterwards.  Carried through so that a reader of this record alone can see it: every
            number here still describes the maximization, but a run that falls back has a problem
            the frontier does not diagnose.
        dominating_members (int): How many members lie up and to the right of the pick: mean at
            least as large, deviation at least as large, one of the two strictly.
        on_frontier (bool): Whether that count is zero.  The term an evolutionary run returns is
            expected to lie on the upper right frontier of its final population.  That is
            necessary and not sufficient: a member with a larger mean and a smaller deviation
            outscores the pick without dominating it.
        higher_scored_members (int): How many members score above the pick.  This is the question
            the maximization actually answers, and it does not follow from the one above.  See
            :func:`read_frontier` for which of the three acquisitions relates the two and how.
        mean_spread (float): The range of ``m_D`` over the population.
        deviation_spread (float): The range of ``s_D``.  Both near zero is the collapse to watch
            for: the population has converged onto one mediocre region.
        points (tuple[tuple[float, float, float], ...]): ``(m, s, score)`` per member, in the
            order the population was given, for the plane.
    """

    size: int
    distinct_members: int
    pick_in_population: bool
    fallback_used: bool
    known_members: int
    mean_at_pick: float
    deviation_at_pick: float
    score_at_pick: float
    dominating_members: int
    on_frontier: bool
    higher_scored_members: int
    mean_spread: float
    deviation_spread: float
    points: tuple[tuple[float, float, float], ...]


def read_frontier(run: AcquisitionRun) -> FrontierRead:
    """Place the term one acquisition maximization returned among its final population.

    The fourth acceptance check, and the one that exists only because the inner optimizer is an
    evolutionary algorithm: it debugs the evolutionary components and their use as the acquisition
    maximizer at once.  Every candidate becomes a point at ``(m_D(t), s_D(t))``, the acquisition
    shades the plane, and the returned term should sit on the upper right frontier of the final
    population, on the highest contour.  A population collapsed into one low-contour cluster is
    premature convergence, and its repair is in the selection parameters, on the other side of the
    interface.

    **Two frontier questions, and they are not the same one.**  Dominance in the plane is the
    reading that transfers to a picture, and scoring above every member is what the maximization
    actually promises.  Neither implies the other, in either direction, and both counts are
    reported for that reason:

    * A member with a **larger mean and a smaller deviation** dominates nothing and outscores the
      pick under every one of the three acquisitions.  So an undominated pick is not a
      well-optimized one, and ``on_frontier`` is a necessary condition rather than the check.
    * For expected improvement and the upper confidence bound, both increasing in the mean and in
      the deviation, the implication does hold one way: a dominating member always outscores, so
      ``dominating_members <= higher_scored_members``.
    * For probability of improvement not even that survives.  ``Phi((m - theta) / s)`` *falls* as
      the deviation grows once the mean is above the threshold, which in the loop is the normal
      case, with the threshold sitting at the incumbent, so a member up and to the right of the
      pick can score below it, and a dominated pick is then not a fault at all.

    A caller who wants one number reads :attr:`FrontierRead.higher_scored_members`.  The dominance
    count is what makes the picture of the plane legible.

    Args:
        run (AcquisitionRun): The maximization, whole: the acquisition it ran against, the term it
            returned, and its final population.  Taking them together is not tidiness: see
            :class:`AcquisitionRun`.

    Returns:
        FrontierRead: The measurements.

    Raises:
        ValueError: If the population is empty, or if the surrogate answers with a broken
            posterior.  See :meth:`AcquisitionFunction.marginal_posterior`.
        TypeError: If the pick or a member is not a term.  Checked here rather than left to the
            surrogate: a kernel handed a string may raise, or may hash it and answer a number, and
            a number that reached a diagnostic through that route is indistinguishable from a
            measurement.
    """
    members = [require_term(member) for member in run.population]
    pick = require_term(run.pick)
    acquisition = run.acquisition
    if not members:
        msg = (
            "the frontier read places the returned term among its final population, and this "
            "population is empty; a run that terminated has one"
        )
        raise ValueError(msg)

    # One call for everything, and the pick appended rather than assumed to be a member: the
    # driver returns the fittest individual of the whole run, which the last generation may no
    # longer hold.
    queried = [*members, pick]
    mean, deviation = acquisition.marginal_posterior(queried)
    scores = np.asarray(acquisition.score(mean, deviation), dtype=float).reshape(-1)

    member_mean, member_deviation, member_scores = mean[:-1], deviation[:-1], scores[:-1]
    pick_mean, pick_deviation, pick_score = (
        float(mean[-1]),
        float(deviation[-1]),
        float(scores[-1]),
    )

    at_least = (member_mean >= pick_mean) & (member_deviation >= pick_deviation)
    strictly = (member_mean > pick_mean) | (member_deviation > pick_deviation)
    dominating = int(np.count_nonzero(at_least & strictly))

    known = acquisition.known_points or ()

    return FrontierRead(
        size=len(members),
        distinct_members=len(set(members)),
        pick_in_population=pick in members,
        fallback_used=run.fallback_used,
        known_members=sum(1 for member in members if member in known),
        mean_at_pick=pick_mean,
        deviation_at_pick=pick_deviation,
        score_at_pick=pick_score,
        dominating_members=dominating,
        on_frontier=dominating == 0,
        higher_scored_members=int(np.count_nonzero(member_scores > pick_score)),
        mean_spread=spread(member_mean),
        deviation_spread=spread(member_deviation),
        points=tuple(
            (float(m), float(s), float(value))
            for m, s, value in zip(
                member_mean, member_deviation, member_scores, strict=True
            )
        ),
    )
