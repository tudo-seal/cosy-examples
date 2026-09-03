from __future__ import annotations

import logging
import math
import random
import time
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Generic, Literal, TypeVar

import numpy as np
from cosy.core.solution_space import SolutionSpace
from cosy.evolutionary_algorithms import (
    EvolutionarySearch,
    Initializer,
    SampledInitialization,
)
from cosy.search import Sampler, SizeUniformSampler, generator_query
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from .acquisition_function import (
    AcquisitionFunction,
    ExpectedImprovement,
    ProbabilityOfImprovement,
    UpperConfidenceBound,
    require_beta,
    require_margin,
)
from .acquisition_optimizer import AcquisitionOptimizer, unbounded_below_message
from .diagnostics import (
    AcquisitionRun,
    TraceRecord,
    enable_verbose_logging,
    get_logger,
    log_iteration,
    log_suggestion,
    warn_if_exploitation_stalls,
)
from .initial_sampling import _sample_fallback_tree
from .kernels.tree_kernel import OrderedRootedSubtreeKernel
from .state import BOState, Diagnostics, Suggestion

NT = TypeVar("NT", bound=Hashable)
T = TypeVar("T", bound=Hashable)
G = TypeVar("G", bound=Hashable)

# The bound of the sampler built when the caller names none.  A placeholder that fits toy
# spaces.  See the ``sampler`` parameter for why a real space needs its own.
DEFAULT_SIZE_BOUND = 100

_LOG = get_logger("bo")

# The small term added to the diagonal of the Gram matrix, and the one jitter of this class.
# The posterior equations the surrogate uses are the noise-free ones, because a fitness function
# is a function and the loop therefore observes exact values.  This constant is not a noise level.
# It is the numerical guard that keeps a Gram matrix of near-duplicate rows factorizable, the case
# in which eigenvalues dip below zero at machine precision.  A genuinely stochastic quality
# measure, a training run for instance, is the other case: that one adds a noise term to the
# diagonal, and it is spelled with a ``WhiteKernel`` in the kernel rather than with this constant.
_JITTER = 1e-6

# The three acquisitions this loop maximizes, under the names ``acquisition_function`` takes.
# One list rather than one written out per site: the check a run passes before it spends anything
# reads it, and so does the message that names the admitted values, so the message cannot name a
# set the check does not admit.  What it does not bind is the construction inside a pass, which
# builds each of the three from arguments of its own and names them again there.  A fourth entry
# added here alone is therefore admitted by the check and refused by the pass, after the design
# has been spent.  The value is the class, which is what tells whether a score can be given to an
# evolutionary run one candidate at a time.
_ACQUISITIONS: dict[str, type[AcquisitionFunction]] = {
    "ExpectedImprovement": ExpectedImprovement,
    "ProbabilityOfImprovement": ProbabilityOfImprovement,
    "UpperConfidenceBound": UpperConfidenceBound,
}

# An evolutionary algorithm is what a pass hands its acquisition to, so a run without one cannot
# make a pass.  Said once so that the check a run passes before it spends anything and the pass
# itself refuse its absence in the same words.
_NO_OPTIMIZER = "An optimizer is required.  Pass a EvolutionarySearch to the constructor."


def _unknown_acquisition(name: Any) -> str:
    """Return the message for an acquisition this loop does not know, naming the ones it does.

    The names come out of the table, so the message admits what the table admits.  A count written
    into the sentence by hand would be the one part of it that a fourth entry could contradict.

    Args:
        name (Any): The name that was configured.  Anything at all, since the caller this answers
            wrote down something that is not one of the names.

    Returns:
        str: The message.
    """
    known = ", ".join(repr(candidate) for candidate in _ACQUISITIONS)
    return f"Unknown acquisition_function {name!r}.  It has to be one of {known}."


def _finite_or_raise(value: Any, candidate: Any) -> float:
    """Return ``value`` as a float, refusing anything that is not finite.

    The objective is modeled as a total function on the terms, so the theory has no notion of a
    failed evaluation, and this framework does not invent one.  A network that fails to train, an
    interpretation that fails, a measurement that returns nothing: none of them get a substitute
    number.  The failure propagates out of the loop, visibly, at the pass that caused it.

    ``nan`` is the one worth naming.  It survives the fit, comes back out of the posterior at every
    term, ties every acquisition value, and leaves a run that has stopped optimizing while still
    producing suggestions.

    Args:
        value (Any): The observed value.
        candidate (Any): The term it was observed at, for the message.

    Returns:
        float: The value.

    Raises:
        ValueError: If the value is not a finite real number.
    """
    observed = float(value)
    if not math.isfinite(observed):
        msg = (
            f"the objective returned {observed} at {candidate}, and a value that is not finite "
            f"cannot be observed: it makes the posterior undefined everywhere rather than at one "
            f"term.  A failed evaluation is a failure, not a number, so let it propagate."
        )
        raise ValueError(msg)
    return observed


def _distinct_dataset(
    drawn: Sequence[Any], sampler: Sampler, query: Any, count: int
) -> tuple[list[Any], int]:
    """Make an initial design pairwise distinct, keeping what the initializer chose.

    Deduplicates in place rather than redrawing the whole design, because the initializer's terms
    are the initializer's answer.  The kernel-diverse initializer draws each member biased away
    from the members already drawn, so throwing the set away to draw a fresh one would silently
    replace the informed design with the model-agnostic one.  Only the places a repeat occupied
    are filled again.

    Structural comparison throughout (``Tree.__eq__``).  ``Tree`` caches its hash on the instance
    and the cached value survives a pickle, so a set is not a decision this project makes about
    term identity.  See :func:`~bayesian_optimization.initial_sampling.distinct_prefix`.

    Args:
        drawn (Sequence[Any]): What the initializer returned.
        sampler (Sampler): The loop's sampler, for the replacements.
        query (Any): The generator query.
        count (int): The design size that was asked for.

    Returns:
        tuple[list[Any], int]: The distinct design, and how many repeats were replaced.

    Raises:
        RuntimeError: If a replacement cannot be drawn.  ``_sample_fallback_tree`` raises on an
            exhausted space, and a design topped up with repeats is what this prevents.
    """
    kept: list[Any] = []
    repeats = 0
    for candidate in drawn:
        if any(candidate == other for other in kept):
            repeats += 1
        else:
            kept.append(candidate)
    while len(kept) < count:
        replacement = _sample_fallback_tree(sampler, query, set(kept))
        if any(replacement == other for other in kept):
            msg = (
                "the fallback returned a term already in the initial design.  Its novelty test "
                "is hash-based and Tree caches its hash, so the two disagreed about identity"
            )
            raise RuntimeError(msg)
        kept.append(replacement)
    return kept, repeats


def _check_request_against_space(search_space: Any, request: Any) -> None:
    """Refuse a request the search space has no rules for, before a query is built from the pair.

    A space queried at a non-terminal it has no rules for answers with the empty stream, and each
    consumer downstream then reports what it can see, which is the sampler.  The initial design
    and the evolutionary run both say that fewer inhabitants than were asked for lie inside the
    sampler's bound and offer to widen it, the duplicate fallback says the bounded space is
    exhausted, and the ``query`` property says nothing at all and hands back a query whose every
    draw is empty.  Widening the bound is the one repair those messages name, and it is the one
    that cannot help.

    Membership, not a test against ``None``.  A non-terminal of some other space, a mistyped
    target for instance, fails through those same messages, and a non-terminal is anything
    hashable, so a space whose non-terminal is ``None`` is queried at ``None`` and draws terms.
    The test is a lookup in the space's own rule table, ``SolutionSpace.__contains__``.

    Args:
        search_space (Any): The space, or None where there is none to query.
        request (Any): The non-terminal the space would be queried at.

    Raises:
        ValueError: If the space has no rules for the request.
    """
    if search_space is None or request in search_space:
        return
    msg = (
        f"the search space has no rules for the request {request}, so nothing can be drawn from "
        "it.  The request is the non-terminal the space is queried at, and every term the loop "
        "draws is an inhabitant of it: pass the non-terminal the space was synthesized for.  The "
        "default None belongs to the ask/tell engine, which runs with search_space=None and poses "
        "no query at all."
    )
    raise ValueError(msg)


# The diagnostics keys a trace row is read from.  ``Diagnostics`` declares three more, and no
# column of the row is built from any of them, so a row is complete without them.
_TRACE_DIAGNOSTICS_KEYS: tuple[str, ...] = (
    "iteration",
    "mean_at_pick",
    "deviation_at_pick",
    "incumbent",
    "fallback_used",
)


def _require_trace_diagnostics(suggestion: Suggestion) -> tuple[Diagnostics, float]:
    """Return the diagnostics and the acquisition value a trace row is read from.

    ``Diagnostics`` declares every key optional, so a mapping that holds none of them is a
    well-typed one, and a row is read from it by subscript.  A check against ``None`` alone
    therefore passes a partial mapping through to the first missing column, where it fails with a
    ``KeyError`` that names one key and never says which ones a row needs.

    Args:
        suggestion (Suggestion): The pass a row is to be written for.

    Returns:
        tuple[Diagnostics, float]: The diagnostics, and the acquisition value at the pick.

    Raises:
        ValueError: If the suggestion carries no diagnostics, no acquisition value, or
            diagnostics without every key a row is read from.  Every suggestion this class
            produces carries all of them.  One that does not came from somewhere else, and a row
            of substitute values would be a run description nobody measured.
    """
    diagnostics = suggestion.diagnostics
    if diagnostics is None:
        msg = (
            "this suggestion carries no diagnostics, so there is nothing to write a trace row "
            "from.  Every suggestion suggest() returns carries them."
        )
        raise ValueError(msg)
    if suggestion.acquisition_value is None:
        msg = (
            "this suggestion carries no acquisition value, so its trace row would have no score "
            "at the pick.  Every suggestion suggest() returns carries one."
        )
        raise ValueError(msg)
    missing = [key for key in _TRACE_DIAGNOSTICS_KEYS if key not in diagnostics]
    if missing:
        msg = (
            f"this suggestion's diagnostics are missing {', '.join(missing)}, so a trace row "
            "cannot be written from them.  Every suggestion suggest() returns carries all of "
            "them, and a row assembled around a gap would be a run description nobody measured."
        )
        raise ValueError(msg)
    return diagnostics, suggestion.acquisition_value


class BayesianOptimization(Generic[NT, T, G]):
    """Bayesian optimization over CoSy solution spaces, as a closed loop and as an ask/tell layer.

    :meth:`optimize` is the closed loop: it takes the initial dataset from the initializer, runs
    ``B`` passes of condition / maximize / evaluate / append, and answers with a term of maximal
    observed value, the standard terminal recommendation.  Everything the algorithm fixes before a
    run, the kernel ``k``, the acquisition ``alpha``, the evolutionary algorithm ``e`` and the
    initializer, is a constructor parameter.  The arguments of a run are the objective, the budget
    ``B``, and the initial size ``mu_0``.

    The ask/tell methods (:meth:`initialize`, :meth:`suggest`, :meth:`observe`, :meth:`finalize`)
    are the same loop turned inside out, and the algorithm has **no counterpart** for them.  They
    exist because an evaluation may live outside this process entirely, as a training run on
    another machine, a measurement, or a human, and a closed loop cannot wait for that.  They are
    an engineering layer, and where the two disagree the closed loop is the one to follow.

    The loop **maximizes** the objective throughout: the incumbent is the largest observed value,
    and every acquisition is maximized over the search space.  Minimize by negating the objective
    on the way in and the reported optimum on the way out.

    Objective values reach the Gaussian process as the caller supplied them.  There is no
    transformation layer.  The objective is a scalarization ``sigma`` composed with the quality
    measure ``q``, both chosen by the caller, and a scalarization is an order-preserving map from
    the fitness values into the positive reals, so a ``log1p`` applied behind the caller's back
    runs opposite to the ``exp`` such a map typically carries.  A caller who wants their objective
    on a log scale writes that into the objective.

    **The guarantees are the closure property and nothing more.**  Every term the loop evaluates
    is an inhabitant of the search space, and that holds by construction for every term this loop
    *produces*, since the initializers stream inhabitants and the variation operators are closed.
    It does **not** extend to terms handed in from outside through ``x0``.  Those are taken as
    given, and nothing here checks their membership, because the ask/tell engine also serves
    callers who have no search space at all.  The regret rates of the literature do *not*
    transfer.  They assume the objective is a sample path of the prior or lies in the reproducing
    kernel Hilbert space of the kernel, which is unsettled for a scalarized fitness over a
    synthesized space.  They assume an exact maximizer of the acquisition each pass, while this
    maximizes with an evolutionary run and inherits no exactness.  And their explicit rates
    concern standard covariance functions on Euclidean domains.  The construction does not depend
    on them.

    Workflow, closed::

        bo = BayesianOptimization(search_space, request, optimizer=ea)
        result = bo.optimize(objective, budget=50, initial_size=10)

    Workflow, ask/tell::

        bo = BayesianOptimization(search_space, request, optimizer=ea)
        bo.initialize(x0=..., y0=...)
        for _ in range(budget):
            suggestion = bo.suggest()
            y = objective(suggestion.candidate)
            bo.observe(suggestion.candidate, y)
        result = bo.finalize()

    Parameters
    ----------
    search_space:
        CoSy solution space.  May be ``None`` when ``x0`` is always supplied
        explicitly (e.g. in tests).
    request:
        The non-terminal the search space is queried at, and the root of the one query the loop
        poses.  Every term the loop draws is an inhabitant of it, so it has to be a non-terminal
        the space has rules for, usually the target the space was synthesized for.  A pair that
        has none is refused, here and again where the query is built.  Its default ``None`` is
        the setting for ``search_space=None``, where there is no query to pose.
    acquisition_function:
        Which of the three standard acquisition scores to maximize.
    ucb_beta:
        The exploration parameter of the upper confidence bound, strictly positive.  That score is
        the posterior mean plus ``beta`` times the posterior standard deviation, so a larger
        ``beta`` rewards uncertainty more.  It is a parameter of the score and is fixed for the
        run.  Passing it per ``suggest()`` call let two calls of the same run maximize two
        different functions without anything recording which.
    pi_margin:
        How far above the incumbent the threshold of the probability of improvement sits.  That
        score is the posterior probability that the value at a term exceeds the threshold, and
        zero puts the threshold at the incumbent, which is the other admissible setting.  It is
        the only margin among the three scores, and expected improvement has none.
    kernel:
        sklearn kernel for the GP.  Defaults to ``OrderedRootedSubtreeKernel()``.

        Three of the four kernels this package ships, ``OrderedRootedSubtreeKernel``,
        ``SubsetTreeKernel`` and ``WeisfeilerLehmanKernel``, declare no hyperparameter at all and
        normalize their Gram matrix, so their similarity carries the shape of a term and not its
        scale, and there is nothing for a kernel optimizer to fit.  Normalization puts
        ``k(t, t)`` at one wherever the unnormalized self-similarity is positive, and there is one
        term where it is not: ``SubsetTreeKernel`` scores a single node at zero, because a single
        node roots no subset tree.  ``HierarchicalWLKernel`` is the exception on both counts: it
        declares one weight per granularity level and sums one normalized kernel per level, so
        its diagonal is the sum of those weights, and model selection moves that sum along with
        them.

        A kernel whose diagonal is one fixes the prior variance of the GP at one.  Without a scale
        fitted to the observations the posterior deviation stays near what that leaves behind,
        and expected improvement, which lives on that deviation, is small for every candidate.  A
        caller who wants the scale fitted multiplies a ``ConstantKernel`` onto the kernel and
        passes ``kernel_optimizer`` along with it.  ``HierarchicalWLKernel`` needs no such factor,
        because its weights already are the scale.

        Two things to watch when model selection is on.  While every observed value is still the
        same, which is the state an initial design on a plateau of the objective leaves behind,
        the marginal likelihood has nothing to explain, and it drives the amplitude to the lower
        bound of the ``ConstantKernel`` instead of reading a scale off the data.  sklearn reports
        that as a ``ConvergenceWarning`` naming ``constant_value``.  And a ``WhiteKernel`` added
        to the sum gives the likelihood a way to call the whole spread observation noise, so the
        fit that results predicts a constant and can carry the higher likelihood of the two.  Read
        the standardized leave-one-out residuals
        (:func:`~bayesian_optimization.read_calibration`) before trusting a fitted kernel.
    kernel_optimizer:
        sklearn kernel hyperparameter optimizer.  Fixing a kernel's parameters by maximizing the
        marginal likelihood, the probability the model assigns to the observed values, is the
        standard model selection for a surrogate.  It is off by default here because the default
        kernel has nothing to select.  sklearn takes an optimizer for such a kernel and then
        skips its optimization step *without saying so*, which is a run spent on a surrogate
        whose scale never moved.  Pass ``"fmin_l_bfgs_b"`` together with a kernel that declares
        hyperparameters to switch model selection on.

        Both mismatches are said out loud, once per run.  An optimizer over a kernel whose
        ``theta`` is empty is the one, and a kernel that declares hyperparameters while this is
        ``None`` is the other, because that one freezes parameters the marginal likelihood could
        have fitted.
    n_restarts_kernel_optimizer:
        Number of random restarts for kernel optimization (never decremented).
    optimizer:
        A ``EvolutionarySearch`` that maximizes the acquisition function.  An
        evolutionary search takes the search space and the fitness function as its arguments,
        while its population size, rates and component choices are fixed before the run and are
        therefore its own parameters.  They used to be passed through this class, which had the
        split between arguments and parameters the wrong way round.
    initializer:
        Where the initial dataset comes from.  ``None`` selects sampled initialization with the
        size-uniform sampler, the model-agnostic default: it stratifies along the one canonical
        axis a search space has, its term size.  The informed alternative is
        ``KernelDiverseInitializer``, which spreads by the surrogate's own kernel instead.
        Neither is universally better, since the empirical recommendations diverge and an evenly
        spread design is not automatically an informative one, so both are offered and the
        model-agnostic one is the default.

        An initializer handed in is configured by its own constructor, so ``sampler`` and ``seed``
        do not reach it.  It carries its own, and its random state survives ``reset()``.  A caller
        who means one bound and one seed sets them in both places.

        ``sampler`` still applies: the duplicate fallback of ``suggest()`` draws from it whichever
        initializer is in use.  That is the reason to hand in a sampler the search space admits
        even when the initializer is replaced.  The two used to be separable, and a fallback the
        space could not run was the result.

        Neither a handed-in initializer nor a handed-in sampler is rebuilt by ``reset()``, and
        both carry their own random state across it, so a second run after a reset continues the
        first one's stream rather than repeating it.  Measured: with the default sampler two runs
        across ``reset()`` draw the same initial dataset, with a handed-in one they do not.  Hand
        in a freshly seeded object per run where that matters.
    seed:
        RNG seed.
    sampler:
        The loop's own source of terms: the default initializer draws its initial dataset from it,
        and the duplicate fallback of ``suggest()`` draws replacements from it.  One object, so a
        caller who knows their search space configures both at once, and cannot configure one of
        them and be left with the other.  That was the trap this parameter replaces: the bound and
        the counting construction were the *loop's*, while the initializer could be handed in
        separately, and a space that admits one sampler but not another then had a working
        initializer and a fallback that could not run on it.

        Only reached where there is a search space to draw from.  With ``search_space=None`` the
        loop has no query to sample and the parameter goes unused, which ``suggest()`` says if a
        duplicate ever sends it looking.

        ``None`` builds ``SizeUniformSampler(100, Random(seed))`` counting from a materialized
        search tree.  That is a placeholder for toy spaces and the first thing to replace on a real
        one, in two respects:

        * The bound is a term **size**, not a depth and not a dataset.  Size-uniform sampling
          draws a realized term size uniformly and then an inhabitant of that size uniformly, both
          within the bound, so what the bound admits is what the sampler counts.
        * The counting construction matters more than the bound.  ``counting="table"`` computes the
          same branch counts from the program instead of from a materialized search tree, seconds
          against hours on a realistic space, but it applies only where no predicate of the
          repository reads a hole, and raises where one does.  Where neither construction is
          affordable, the counting is what has to go: ``DepthBoundedRandomSampler`` never counts.
          Which of these a space admits is the repository's property and not this class's to
          decide, which is why the parameter is an object rather than two numbers.
    gp_normalize_y:
        Passed to sklearn's ``GaussianProcessRegressor(normalize_y=...)``.  This centers and
        scales the targets inside the GP for the duration of the fit and undoes it on predict, so
        it is a numerical convenience of the regressor, not a transformation of the objective.

        The theory does not mention it, and it is on by default here as a deliberate decision.
        The surrogate takes its mean function to be **zero**, as is customary, so an objective
        that lives around 0.9, an accuracy for instance, is read by the untouched prior as
        uniformly far above its mean, and the posterior spends its first evaluations discovering
        the offset.  Centering inside the fit is the standard remedy and leaves the model intact,
        because it is undone before any value leaves the regressor.
    """

    def __init__(
        self,
        search_space: SolutionSpace[NT, T, G] | None,
        request: NT | None = None,
        *,
        acquisition_function: Literal[
            "ExpectedImprovement", "ProbabilityOfImprovement", "UpperConfidenceBound"
        ] = "ExpectedImprovement",
        ucb_beta: float = 2.0,
        pi_margin: float = 0.0,
        kernel: Kernel | None = None,
        kernel_optimizer: str | None = None,
        n_restarts_kernel_optimizer: int = 20,
        optimizer: EvolutionarySearch[NT, T, G] | None = None,
        initializer: Initializer[NT, T, G] | None = None,
        seed: int | None = None,
        sampler: Sampler | None = None,
        gp_normalize_y: bool = True,
    ) -> None:
        # The pair is refused where the caller writes it down, so a mistyped target costs no
        # evaluation of the objective.  ``optimize`` refuses a negative budget in the same way and
        # for the same reason.  This reads the arguments, so it is not the guarantee on its own:
        # both attributes are public, and the pair is checked again in ``_query``, where it
        # becomes a query.
        _check_request_against_space(search_space, request)

        self.search_space = search_space
        self.request = request
        self.acquisition_function = acquisition_function
        self.ucb_beta = float(ucb_beta)
        self.pi_margin = float(pi_margin)
        self.kernel: Kernel = (
            OrderedRootedSubtreeKernel() if kernel is None else kernel
        )
        self.kernel_optimizer = kernel_optimizer
        self.n_restarts_kernel_optimizer = n_restarts_kernel_optimizer
        self.optimizer = optimizer
        self.initializer = initializer
        self.seed = seed
        self.sampler = sampler

        self._gp_normalize_y: bool = gp_normalize_y

        # --- Ask/Tell state ---------------------------------------------------
        self._bo_state: BOState = BOState.UNINITIALIZED
        self._x_list: list[Any] = []
        self._y_list: list[float] = []
        self._x_set: set[Any] = set()
        self._initial_repeats_rejected: int = 0
        self._last_suggestion: Suggestion | None = None
        self._model: GaussianProcessRegressor | None = None
        self._alpha: float = _JITTER
        self._gp_params: dict[str, Any] | None = None
        self._sampler: Sampler | None = None
        self._initializer: Initializer[NT, T, G] | None = None
        self._generator_query: Any = None
        self._last_optimized_kernel: Kernel | None = None
        self._iteration: int = 0
        self._warned_about_model_selection: bool = False
        self._warned_about_frozen_hyperparameters: bool = False
        self._warned_about_exploitation: bool = False
        self._logger: logging.Logger = _LOG
        self._trace: list[TraceRecord] = []

        self.last_acquisition_run: AcquisitionRun | None = None
        """The last acquisition maximization, whole, where one was recorded.

        Only the frontier check needs it, and that check reads the population as well as the pick:
        the term the evolutionary run returns has to lie on the upper right frontier of its final
        population in the mean-deviation plane.  Nothing else needs it, so it is filled only when
        :meth:`suggest` is asked to record it.  ``None`` says it was not asked, which is a
        different statement from an empty record and is kept apart from it.

        It is one object rather than three attributes because its three parts are only meaningful
        together, and a caller assembling them by hand gets it wrong in ways that read as
        findings.  See :class:`AcquisitionRun`.  Note in particular that its ``pick`` is what the
        maximization returned, which after a duplicate fallback is **not** the term the loop went
        on to evaluate.
        """

    def _query(self) -> Any:
        """Return the generator query naming the search space and the requested type.

        The one access path to a synthesized search space: every evolutionary component poses
        resolution queries, so this is what the initializer and the evolutionary search receive
        instead of the space itself.

        Without a search space there is no query to pose.  That configuration exists, because the
        ask/tell engine also drives optimizers that are not cosy's and the tests exercise it with
        stubs, and those callers do not read the query.  Every path that genuinely needs a search
        space checks for one before reaching this, and says so.

        **One object, and it is not a micro-optimization.**  The query is determined by the
        search space and the request, and the first call freezes that pair, so a second object would
        denote the same SLAD-tree, but ``SizeUniformSampler`` keys its counting construction by
        query *identity*, precisely because comparing a partial-term query structurally costs more
        than the lookup saves.  Minting a fresh query per call therefore made every caller pay the
        counting again: measured on the determinized CNN search space of the CIFAR-10 experiment
        at ``D = 200``, that is 93 s for the initial dataset and another 93 s for each duplicate
        fallback, against 0.04 s per draw once it is built.  Sharing the object is what makes the
        sampler's own cache reachable at all.

        Returns:
            Any: The generator query, or None if this optimizer has no search space.

        Raises:
            ValueError: If the search space has no rules for the request.
        """
        if self.search_space is None:
            return None
        if self._generator_query is None:
            # The pair the constructor read is not necessarily the pair the query is built from,
            # because both attributes are public and writable.  Checking again at the one place
            # the two are turned into a query is what makes the refusal hold for every query this
            # class hands out, and it is what makes the silent path loud.
            _check_request_against_space(self.search_space, self.request)
            self._generator_query = generator_query(self.search_space, self.request)
        return self._generator_query

    # -------------------------------------------------------------------------
    # Ask/Tell API
    # -------------------------------------------------------------------------

    def initialize(
        self,
        *,
        objective: Callable[[Any], float] | None = None,
        x0: Sequence[Any] | None = None,
        y0: Sequence[float] | None = None,
        initial_size: int = 10,
        gp_params: dict[str, Any] | None = None,
        alpha: float = _JITTER,
    ) -> None:
        """Build the initial dataset and transition to INITIALIZED state.

        This is the loop's first step, the dataset ``D <- ((t, sigma(q(t))) | t in init(mu_0))``
        of the terms the initializer draws paired with their objective values, with the
        engineering layer's addition that the caller may hand the pairs over ready-made.

        Parameters
        ----------
        objective:
            The objective ``sigma . q``, which is maximized.  Required when ``y0`` is
            ``None``.
        x0:
            Initial candidate trees.  When ``None`` the initializer draws them
            (requires a non-``None`` ``search_space``).
        y0:
            Initial objective values.  When ``None`` ``objective`` is called on each element
            of ``x0``.
        initial_size:
            The initial size ``mu_0``: how many terms the initializer draws when ``x0`` is
            ``None``.
        gp_params:
            Extra kwargs forwarded to ``GaussianProcessRegressor``.
        alpha:
            The numerical diagonal added to the Gram matrix.  See :data:`_JITTER`.
        """
        if self._bo_state != BOState.UNINITIALIZED:
            raise RuntimeError(
                f"Cannot re-initialize: current state is {self._bo_state.value}. "
                "Call reset() first."
            )

        # The sampler is built whenever there is a space to draw from: the duplicate fallback of
        # suggest() draws from it whichever initializer the caller chose.
        if self._sampler is None and self.search_space is not None:
            self._sampler = (
                SizeUniformSampler(DEFAULT_SIZE_BOUND, random.Random(self.seed))
                if self.sampler is None
                else self.sampler
            )
        if self._initializer is None and self._sampler is not None:
            self._initializer = (
                SampledInitialization(self._sampler)
                if self.initializer is None
                else self.initializer
            )

        if x0 is None:
            if self.search_space is None:
                raise NotImplementedError(
                    "initialize() without x0 requires a real search_space."
                )
            assert self._initializer is not None
            assert self._sampler is not None
            # The previous code drew a pool a hundred times the size and thinned it with a
            # greedy determinantal point process, which is related work rather than either of the
            # initializers this loop admits, and it warned and came back short where the stream
            # simply raises.  Both of those initializers raise rather than return a short
            # population.
            x0_list = list(self._initializer.initialize(self._query(), initial_size))
            # And then the dataset is made a *set*, which is not something the initializer can be
            # asked for.  Sampled initialization builds a population, and a population is a finite
            # multiset, so repeats are admissible there by definition.  The loop's dataset is not:
            # a repeated term is a training point carrying no observation the previous one did
            # not, one evaluation of the budget spent on nothing.
            #
            # The comment that used to stand here inherited the guarantee from the default
            # sampler, whose stream lists each inhabitant within the bound exactly once, so that
            # every prefix of it is a sample without replacement.  But ``sampler`` is a parameter
            # of this class, and the depth-bounded random sampler draws independently, so its
            # draws may repeat a term.  A guarantee that holds only until someone uses the
            # parameter is the kind of unwritten invariant this API must not have.
            x0_list, repeats = _distinct_dataset(
                x0_list, self._sampler, self._query(), initial_size
            )
            if repeats:
                self._logger.info(
                    "the initializer returned %d repeated term(s) in an initial design of %d.  "
                    "They were redrawn, as suggest() redraws a duplicate candidate.  The "
                    "size-uniform sampler lists each inhabitant within its bound exactly once, so "
                    "under it this number is 0",
                    repeats, initial_size,
                )
        else:
            x0_list = list(x0)
            # A design the caller hands over was not drawn here, so this loop redrew nothing in
            # it.  The count is about the repair, not about the design.
            repeats = 0

        # Resolve y0.  The lengths are compared before a single value is read, so that a mismatch
        # is reported as such rather than truncating the longer of the two.
        if y0 is None:
            if objective is None:
                raise ValueError(
                    "objective must be provided when y0 is None."
                )
            y0_list = [_finite_or_raise(objective(t), t) for t in x0_list]
        else:
            if len(x0_list) != len(y0):
                raise ValueError(
                    f"len(x0)={len(x0_list)} != len(y0)={len(y0)}."
                )
            y0_list = [_finite_or_raise(v, t) for t, v in zip(x0_list, y0, strict=True)]

        # Validate hashability
        for item in x0_list:
            try:
                hash(item)
            except TypeError as unhashable:
                raise TypeError(
                    f"All candidates in x0 must be hashable.  Got an "
                    f"unhashable item of type {type(item).__name__!r}."
                ) from unhashable

        self._x_list = x0_list
        self._y_list = y0_list
        # Checked here rather than at the fit: a dataset that contradicts itself does so the moment
        # it is handed over, and the first suggest() is far from the caller who assembled it.
        self._distinct_pairs()
        self._x_set = set(self._x_list)
        # The count is stored with the dataset it describes and not where it is computed.
        # Everything between the draw and the last line of this method can raise, the objective
        # above all, and a failure there leaves the state UNINITIALIZED, which permits a second
        # initialize().  Storing the count at the draw would carry the count of the abandoned
        # design into whatever design the caller supplies next.
        self._initial_repeats_rejected = repeats
        self._last_suggestion = None
        self._alpha = alpha
        self._gp_params = gp_params
        self._iteration = 0
        self._bo_state = BOState.INITIALIZED

    def suggest(
        self,
        *,
        acquisition_fitness_mode: Literal["single", "batch"] = "batch",
        verbose: bool = False,
        record_population: bool = False,
    ) -> Suggestion:
        """Fit the GP and return the next suggested candidate.

        One pass of the loop's first two lines: the surrogate is fitted anew on the distinct pairs
        of the dataset, including its kernel hyperparameters when model selection is on, and the
        evolutionary algorithm maximizes the current acquisition over the search space.  What it
        maximizes is the acquisition with the known-point floor beneath it, not the acquisition
        alone.  That floor is the first of the two mechanisms carrying the rejection of duplicates
        described below.

        Parameters
        ----------
        acquisition_fitness_mode:
            Whether the evolutionary algorithm scores its population one at a time or in batches.
        verbose:
            Log the suggestion.
        record_population:
            Keep the final population of the acquisition maximization in
            :attr:`last_acquisition_population`, for the frontier read of the acceptance checks.
            Off by default, and not because of what it costs, which is a list of references, but
            because of what it asks: recording it needs the optimizer's ``evolutionary_stream``,
            while maximizing needs only its ``evolutionary_best``, and the ask/tell engine also
            drives optimizers that are not cosy's.

        Returns
        -------
        Suggestion
            Frozen record with the candidate, acquisition value, and diagnostics.

        Raises
        ------
        RuntimeError
            If called in an invalid state, if no optimizer was configured, if the dataset is
            empty, if the optimizer returns ``None``, if it returns an already evaluated candidate
            and there is no search space to replace it from, or if the fallback sampler breaks its
            contract.  ``_sample_fallback_tree`` raises on its own account when the bounded space
            is exhausted.
        ValueError
            If the dataset gives one term two different values, or if the configured acquisition
            is not one of the three this class knows.
        """
        if self._bo_state not in (BOState.INITIALIZED, BOState.OBSERVED):
            raise RuntimeError(
                f"suggest() is not allowed in state {self._bo_state.value}."
            )
        if self.optimizer is None:
            raise RuntimeError(_NO_OPTIMIZER)

        if verbose:
            enable_verbose_logging()

        # --- Fit the GP on the distinct pairs of the dataset --------------------
        #
        # The dataset keeps every evaluation, and the conditioning sees each term once.  Under the
        # rejection path below the loop never proposes a term twice, so this fires only on a
        # dataset assembled from outside, and there it is what keeps the noise-free Gram matrix
        # invertible, since two identical rows are linearly dependent and only the jitter would
        # stand between that and a failed factorization.
        conditioned_x, conditioned_y = self._distinct_pairs()
        if not conditioned_x:
            raise RuntimeError(
                "there is nothing to condition the Gaussian process on: the dataset is empty.  "
                "A run starts from an initial design of mu_0 terms drawn by the initializer, not "
                "from no observation at all."
            )

        model = self._fit_surrogate(conditioned_x, conditioned_y)
        self._model = model
        # Store the optimized kernel for warm-start reporting only.  Never mutate self.kernel.
        self._last_optimized_kernel = getattr(model, "kernel_", None)

        # The incumbent the three scores compare against is the best observation, and best means
        # largest.  Read off the whole dataset rather than off the distinct pairs, which is the
        # same number, since removing repetitions removes no value, but says which of the two it
        # is a property of.
        incumbent = float(np.max(np.array(self._y_list, dtype=float)))

        # --- Build acquisition function ---------------------------------------
        #
        # A *copy* of the observed set.  Handing over the live one made the acquisition a view of
        # the loop rather than a record of this pass: the next observe() adds to it, and the same
        # object then answers the known-point floor at the very term it had just chosen.  While a
        # pass runs the two are equal, since nothing observes in between, so this changes no
        # decision.  It changes what an acquisition still means once the pass is over, which is
        # exactly what the diagnostics keep one for.
        known_points = set(self._x_set)

        af: AcquisitionFunction
        if self.acquisition_function == "ExpectedImprovement":
            af = ExpectedImprovement(
                gp=model,
                incumbent=incumbent,
                known_points=known_points,
            )
        elif self.acquisition_function == "ProbabilityOfImprovement":
            af = ProbabilityOfImprovement(
                gp=model,
                incumbent=incumbent,
                known_points=known_points,
                margin=self.pi_margin,
            )
        elif self.acquisition_function == "UpperConfidenceBound":
            af = UpperConfidenceBound(
                gp=model,
                beta=self.ucb_beta,
                known_points=known_points,
            )
        else:
            raise ValueError(_unknown_acquisition(self.acquisition_function))

        # --- Optimize acquisition function ------------------------------------
        acq_opt = AcquisitionOptimizer(self.optimizer)
        population: list[Any] | None = None
        generations: list[Any] = []
        if record_population:
            candidate, population, generations = acq_opt.maximize_with_population(
                af, self._query(), mode=acquisition_fitness_mode
            )
        else:
            candidate = acq_opt.maximize(af, self._query(), mode=acquisition_fitness_mode)

        if candidate is None:
            raise RuntimeError("Optimizer did not return a candidate.")

        # What the maximization returned, before the rejection path below may replace it.  The
        # frontier read diagnoses the maximization, and a replacement drawn at random was never
        # scored against this population, so read in its place it comes out behind every member.
        maximiser = candidate

        # --- Rejection of duplicates: a deliberate deviation from the algorithm -
        #
        # The algorithm as stated lets an evaluation repeat and removes duplicates only when
        # conditioning the surrogate, because an evaluation may repeat and an exact observation
        # repeats identically.  Rejecting a repeated proposal and adding a jitter to the diagonal
        # are the two alternatives named beside it.  This implementation takes rejection: a
        # repeated evaluation costs a call to the expensive quality measure and buys the surrogate
        # nothing, and on a finite search space a loop that is allowed to repeat can spend a whole
        # budget standing still.  Two mechanisms carry it.  The acquisition scores known points
        # below every genuine candidate, and this loop replaces one that gets through anyway.
        #
        # This is the one place where the code knowingly departs from the algorithm as stated, and
        # the resolution belongs on the specification side rather than here.
        #
        # One draw settles it.  ``_sample_fallback_tree`` returns an inhabitant outside the
        # observed set or raises, so a retry loop here would have nothing to retry.  The bounded
        # retry that used to stand in its place could not reach its second pass, and its error
        # message described a state the code cannot be in.
        fallback_used = candidate in self._x_set
        if fallback_used:
            _LOG.warning(
                "iteration %d: the acquisition optimizer returned an already evaluated "
                "candidate, so it is replaced with a random fallback sample.  The suggestion's "
                "acquisition_value then describes the replacement, not the optimizer's result.  A "
                "run in which this fires every iteration is random search, not BO.",
                self._iteration,
            )
            if self._sampler is None:
                given = "" if self.sampler is None else (
                    "  A sampler was handed in, but it went unused: there is no query to draw "
                    "from without a search space."
                )
                msg = (
                    "the acquisition optimizer returned an already evaluated candidate and "
                    f"there is no search space to draw a replacement from.{given}"
                )
                raise RuntimeError(msg)
            candidate = _sample_fallback_tree(self._sampler, self._query(), self._x_set)
            if candidate in self._x_set:
                raise RuntimeError(
                    f"the fallback sampler returned {candidate}, which has already been "
                    f"evaluated.  Drawing a novel inhabitant or raising is the whole of its "
                    f"contract, so this is a defect in the sampler rather than an exhausted "
                    f"search space.  Continuing would evaluate a duplicate, which is the one "
                    f"thing the rejection path exists to prevent."
                )

        if population is not None:
            self.last_acquisition_run = AcquisitionRun(
                acquisition=af,
                pick=maximiser,
                population=tuple(population),
                fallback_used=fallback_used,
                generations=tuple(generations),
            )

        acq_value = float(af(candidate))
        # The pick's place in the mean-deviation plane, for the trace readings.  One prediction
        # at one term, and it is asked of the acquisition rather than of the model so that a
        # broken posterior is reported here the same way it is reported everywhere else.
        mean_at_pick, deviation_at_pick = af.marginal_posterior([candidate])

        if not self._warned_about_exploitation:
            self._warned_about_exploitation = warn_if_exploitation_stalls(
                _LOG,
                iteration=self._iteration,
                acquisition_value=acq_value,
                # After a fallback the value describes a random replacement, and a replacement at
                # the floor says nothing about what the maximization found.
                lower_bound=None if fallback_used else af.lower_bound,
                acquisition_name=self.acquisition_function,
            )

        diagnostics: Diagnostics = {
            "timestamp": time.time(),
            "incumbent": incumbent,
            "iteration": self._iteration,
            "fallback_used": fallback_used,
            # Zero or one: a single draw decides, see above.  The column stays because the
            # experiment logs carry it.
            "fallback_attempts": int(fallback_used),
            "mean_at_pick": float(mean_at_pick[0]),
            "deviation_at_pick": float(deviation_at_pick[0]),
            "phase": "main",
        }

        suggestion = Suggestion(
            candidate=candidate,
            acquisition_value=acq_value,
            diagnostics=diagnostics,
        )
        self._last_suggestion = suggestion
        self._bo_state = BOState.SUGGESTED
        log_suggestion(_LOG, suggestion)
        return suggestion

    def _distinct_pairs(self) -> tuple[list[Any], list[float]]:
        """Return the distinct pairs of the dataset, in order of first appearance.

        The loop conditions the surrogate on these rather than on the whole dataset, because an
        evaluation may repeat and an exact observation repeats identically.  The order is the one
        the pairs first arrived in, so that two runs from the same seed condition on the same
        matrix rather than on a permutation of it.

        Returns:
            tuple[list[Any], list[float]]: The distinct terms and their values.

        Raises:
            ValueError: If one term carries two different values.  The loop observes a *function*,
                so that is not a repetition.  Keeping one of the two would pick a measurement on
                the caller's behalf, and keeping both is the singular Gram matrix this clause
                exists to prevent.  A quality measure that genuinely varies between calls is the
                stochastic case, and that one is modeled with a noise term on the diagonal rather
                than with two rows.
        """
        values: dict[Any, float] = {}
        terms: list[Any] = []
        observations: list[float] = []
        for candidate, value in zip(self._x_list, self._y_list, strict=True):
            if candidate in values:
                if values[candidate] != value:
                    raise ValueError(
                        f"the dataset gives {candidate} two different values, "
                        f"{values[candidate]} and {value}: the loop observes a function, so a "
                        f"repeated term repeats its value.  A quality measure that varies between "
                        f"calls is the stochastic case, which belongs on the diagonal of the "
                        f"kernel (WhiteKernel), not in the dataset."
                    )
                continue
            values[candidate] = value
            terms.append(candidate)
            observations.append(value)
        return terms, observations

    def _fit_surrogate(
        self, terms: Sequence[Any], values: Sequence[float]
    ) -> GaussianProcessRegressor:
        """Condition a Gaussian process on ``(terms, values)`` with this run's configuration.

        The one place a surrogate is built, so that the one a diagnostic reads is the one a pass
        maximized against: same kernel, same diagonal term, same model selection, same seed.

        Args:
            terms (Sequence[Any]): The terms to condition on, pairwise distinct.
            values (Sequence[float]): Their observed values.

        Returns:
            GaussianProcessRegressor: The fitted surrogate.
        """
        gp_kwargs: dict[str, Any] = dict(self._gp_params) if self._gp_params else {}
        gp_kwargs.setdefault("kernel", self.kernel)
        gp_kwargs.setdefault("alpha", self._alpha)
        gp_kwargs.setdefault("n_restarts_optimizer", self.n_restarts_kernel_optimizer)
        gp_kwargs.setdefault("optimizer", self.kernel_optimizer)
        gp_kwargs.setdefault("normalize_y", self._gp_normalize_y)
        gp_kwargs.setdefault("random_state", self.seed)

        self._warn_if_nothing_to_fit(gp_kwargs)
        self._warn_if_hyperparameters_go_unfitted(gp_kwargs)

        model = GaussianProcessRegressor(**gp_kwargs)
        model.fit(
            np.asarray(terms, dtype=object),
            np.asarray(values, dtype=float),
        )
        return model

    def surrogate_over_dataset(self) -> GaussianProcessRegressor:
        """Return a surrogate conditioned on the **whole** dataset, the final pair included.

        The model :meth:`finalize` reports is the one of the last :meth:`suggest`, and nothing is
        fitted after it, so that model has never seen what the run appended afterwards.  For the
        loop that is right: the fit exists to choose the next term, and after the last pass there
        is no next term.  For a diagnostic it is wrong, because the fit and the calibration reads
        of the acceptance checks are statements about the data the run collected, and one of the
        pairs would be missing from them.

        This conditions one on all of it, with the same configuration, and refits the kernel
        hyperparameters as any other pass would.

        Returns:
            GaussianProcessRegressor: The fitted surrogate.

        Raises:
            RuntimeError: If the dataset is empty.
            ValueError: If it gives one term two different values (see :meth:`_distinct_pairs`).
        """
        terms, values = self._distinct_pairs()
        if not terms:
            msg = (
                "there is nothing to condition a surrogate on: the dataset is empty.  A "
                "diagnostic over a run needs the run to have observed something."
            )
            raise RuntimeError(msg)
        return self._fit_surrogate(terms, values)

    @property
    def query(self) -> Any:
        """The generator query this loop poses, or ``None`` where there is no search space.

        The one object, so that a caller who wants to draw from :attr:`sampler` themselves, as a
        paired random-search baseline does, draws through the same query the loop uses, and the
        sampler's counting construction is built once for both.  Asking the sampler with a query
        of one's own would be correct and would pay the counting twice.

        Returns:
            Any: The query, or None.

        Raises:
            ValueError: If the search space has no rules for the request.  This is the path that
                used to report nothing and hand back a query every draw from which is empty.
        """
        return self._query()

    @property
    def surrogate(self) -> GaussianProcessRegressor | None:
        """The surrogate of the last :meth:`suggest`, or ``None`` before the first one.

        The model that pass actually maximized its acquisition against, kernel hyperparameters
        included, which is what a per-pass diagnostic is a statement about.  ``finalize()``
        reports the same object at the end of a run under ``gp_model``, and this exposes it while
        the run is still going, so a caller can record how the surrogate changed instead of only
        how it ended up.

        Returns:
            GaussianProcessRegressor | None: The model, or None if no pass has fitted one.
        """
        return self._model

    def surrogate_over(
        self, terms: Sequence[Any], values: Sequence[float]
    ) -> GaussianProcessRegressor:
        """Return a surrogate conditioned on a chosen subset, with this run's configuration.

        The held-out fit read of the acceptance checks needs exactly this: a surrogate that has
        **not** seen the terms it is asked to predict.  Conditioned on them, a noise-free Gaussian
        process reproduces its training values, and the scatter would sit on the diagonal whatever
        the kernel does, so :meth:`surrogate_over_dataset` cannot serve that read, and reaching
        past it into the private fitter is what every caller did instead.

        Which terms are held out is the caller's choice and not a detail.  The read plots
        predicted against true values after conditioning on the terms of even index and predicting
        the odd ones, a split that keeps the training set spread over the space.

        Args:
            terms (Sequence[Any]): The terms to condition on, pairwise distinct.
            values (Sequence[float]): Their observed values, in the same order.

        Returns:
            GaussianProcessRegressor: The fitted surrogate.

        Raises:
            ValueError: If the two sequences differ in length, if there is nothing to condition
                on, or if a term appears twice, which is the same rule the loop conditions under,
                and a repeated term with two values is the case :meth:`_distinct_pairs` refuses.
        """
        chosen = list(terms)
        observed = [float(value) for value in values]
        if len(chosen) != len(observed):
            msg = (
                f"a surrogate pairs terms with values by position: {len(chosen)} terms and "
                f"{len(observed)} values"
            )
            raise ValueError(msg)
        if not chosen:
            msg = (
                "there is nothing to condition a surrogate on: no terms were given.  A held-out "
                "fit needs a training half as well as a prediction half."
            )
            raise ValueError(msg)
        if len(set(chosen)) != len(chosen):
            msg = (
                "the terms to condition on repeat.  The loop conditions on the distinct pairs "
                "of the dataset, and a repeated term carries no second piece of information, only "
                "the risk of two different values for one term."
            )
            raise ValueError(msg)
        return self._fit_surrogate(chosen, observed)

    def _warn_if_nothing_to_fit(self, gp_kwargs: dict[str, Any]) -> None:
        """Say so when model selection is asked for and the kernel has nothing to select.

        sklearn takes an ``optimizer`` and a kernel whose ``theta`` is empty without complaining.
        It guards its whole optimization step with ``self.kernel_.n_dims > 0`` and skips it, so
        the amplitudes stay wherever they were constructed and the fit is bit for bit the one an
        unset optimizer would have produced.  On a counting kernel that is the difference between
        a calibrated GP and one whose sigma is an order of magnitude too small, and the symptom,
        an acquisition that underflows and a search that stops exploring, shows up nowhere near
        the cause.

        Args:
            gp_kwargs: The arguments the regressor is about to be built with.
        """
        if self._warned_about_model_selection or gp_kwargs.get("optimizer") is None:
            return
        kernel = gp_kwargs.get("kernel")
        if kernel is not None and getattr(kernel, "theta", np.empty(0)).size == 0:
            self._warned_about_model_selection = True
            _LOG.warning(
                "kernel_optimizer=%r is set, but %s declares no hyperparameters, so there is "
                "nothing for the marginal likelihood to fit and sklearn skips its optimization "
                "step without a word.  The kernel's scales stay at their constructed values for "
                "the whole run.",
                gp_kwargs.get("optimizer"),
                type(kernel).__name__,
            )

    def _warn_if_hyperparameters_go_unfitted(self, gp_kwargs: dict[str, Any]) -> None:
        """Say so when a kernel declares hyperparameters and no optimizer was set to fit them.

        This is the other half of the silence :meth:`_warn_if_nothing_to_fit` covers.  Model
        selection is off by default here because the kernel it defaults to declares nothing to
        select, so a caller who hands in one that does declare something gets it frozen at its
        constructed values unless they hand in an optimizer as well.  The symptom is the same as
        in the opposite case, a surrogate whose scale never moves, but the cause sits in the
        argument the caller left out rather than in the one they passed.

        Args:
            gp_kwargs: The arguments the regressor is about to be built with.
        """
        if self._warned_about_frozen_hyperparameters or gp_kwargs.get("optimizer") is not None:
            return
        kernel = gp_kwargs.get("kernel")
        theta = getattr(kernel, "theta", np.empty(0))
        if kernel is not None and theta.size > 0:
            self._warned_about_frozen_hyperparameters = True
            _LOG.warning(
                "kernel_optimizer is None, but %s declares hyperparameters (theta has %d "
                "entries), so the marginal likelihood never sees them and they stay at the values "
                "the kernel was constructed with for the whole run.  Pass "
                "kernel_optimizer='fmin_l_bfgs_b' to fit them.",
                type(kernel).__name__,
                theta.size,
            )

    def observe(self, candidate: Any, y: float) -> None:
        """Record the objective value for the last suggested candidate.

        Parameters
        ----------
        candidate:
            Must match the candidate from the last ``suggest()`` call.
        y:
            The objective value, exactly as measured.

        Raises
        ------
        RuntimeError
            If called outside the SUGGESTED state.
        ValueError
            If ``candidate`` does not match the last suggestion, if ``y`` is not finite, or if
            the last suggestion carries no diagnostics a trace row can be read from.
        """
        if self._bo_state != BOState.SUGGESTED:
            raise RuntimeError(
                f"observe() is not allowed in state {self._bo_state.value}.  "
                "Call suggest() first."
            )
        if self._last_suggestion is None:
            raise RuntimeError("Internal error: _last_suggestion is None in SUGGESTED state.")
        if candidate != self._last_suggestion.candidate:
            raise ValueError(
                "Observed candidate does not match the last suggested candidate."
            )

        # Record the tree suggest() produced, not the caller's argument.  The check above is
        # structural on purpose, since a caller may legitimately hand back a reconstructed tree,
        # but the observation set has to hold exactly what was suggested.
        recorded = self._last_suggestion.candidate
        value = _finite_or_raise(y, recorded)
        # Checked before the dataset grows, because the row is written after it has.  A suggestion
        # no row can be read from would otherwise leave a term and a value behind that no trace
        # row and no iteration count mention, and the state stays at SUGGESTED, so the very same
        # call is accepted again and appends them a second time.
        _require_trace_diagnostics(self._last_suggestion)
        self._x_list.append(recorded)
        self._x_set.add(recorded)
        self._y_list.append(value)
        self._trace.append(self._trace_record(self._last_suggestion, value))
        self._iteration += 1
        self._bo_state = BOState.OBSERVED

    def _trace_record(self, suggestion: Suggestion, observed: float) -> TraceRecord:
        """Close one pass into a row of the run trace.

        Written here rather than in :meth:`suggest` because a pass is only a pass once its value
        is in: the acquisition value at the pick and the value the objective answered are the two
        halves of every trace reading, and half a row would be a row that lies about the run's
        length.

        The two counts are read off the dataset that now exists, not predicted from the one that
        did.  ``suggest()`` used to write "distinct terms after this pass" as "distinct so far plus
        one", which is right for every term the loop itself proposes, since the rejection path
        sees to that, and makes the count *unable* to disagree with the number of passes.  The
        distinct fraction computed from it was then one by construction, on every trace, including
        traces of datasets that genuinely repeat.

        Args:
            suggestion (Suggestion): What the pass suggested.
            observed (float): What the objective answered, already checked for finiteness.

        Returns:
            TraceRecord: The row.

        Raises:
            ValueError: If no row can be read from the suggestion, as
                :func:`_require_trace_diagnostics` decides it.  ``observe()`` asks the same
                question before it records anything, so the answer here is a second reading of
                the values the row is then built from.
        """
        diagnostics, acquisition = _require_trace_diagnostics(suggestion)
        return TraceRecord(
            iteration=diagnostics["iteration"],
            acquisition=acquisition,
            mean=diagnostics["mean_at_pick"],
            deviation=diagnostics["deviation_at_pick"],
            incumbent=diagnostics["incumbent"],
            observed=observed,
            best=max(observed, diagnostics["incumbent"]),
            fallback_used=diagnostics["fallback_used"],
            evaluations=len(self._x_list),
            distinct_observations=len(self._x_set),
        )

    @property
    def trace(self) -> list[TraceRecord]:
        """The run trace, one row per completed pass.

        Read it with ``read_trace``, or write it out with ``trace_columns``/``trace_rows``.  It
        covers the passes of the loop and not the initial design: the design has no acquisition
        value and no pick, so a row for it would be a row of blanks.
        """
        return list(self._trace)

    @property
    def initial_repeats_rejected(self) -> int:
        """How many repeated terms the initializer produced and this loop redrew.

        Zero under the size-uniform sampler, whose stream lists each inhabitant within the bound
        exactly once, so that every prefix of it is a sample without replacement.  A nonzero value
        there says that guarantee did not hold, and under the depth-bounded random sampler, whose
        draws are independent and may repeat, it says how much the repair had to do.  Reported
        rather than absorbed: a run whose initial design needed redrawing is a run whose sampler
        does not supply what its dataset needs, and the record has to be able to say so.

        A design the caller hands to ``initialize()`` reports 0 whatever it contains, because the
        number counts what this loop redrew and it redrew nothing there.

        Returns:
            int: The count, 0 before ``initialize()`` and 0 again after ``reset()``.
        """
        return self._initial_repeats_rejected

    def reset(self) -> None:
        """Reset to UNINITIALIZED, clearing all observations.

        Constructor parameters are preserved.  The **default** initializer is rebuilt with it, so
        a second run repeats the first: its sampler is seeded from ``self.seed``, and dropping it
        here is what makes the stream start over rather than continue.

        An initializer handed to the constructor is a different matter.  It carries its own
        random state, which this cannot reach and does not reset.  A second run under one of those
        continues where the first left off, and two runs of an otherwise identical configuration
        draw different populations.  A caller who wants them to agree constructs a fresh
        initializer between runs.
        """
        self._bo_state = BOState.UNINITIALIZED
        self._x_list = []
        self._y_list = []
        self._x_set = set()
        self._initial_repeats_rejected = 0
        self._last_suggestion = None
        self._model = None
        self._iteration = 0
        self._last_optimized_kernel = None
        self._sampler = None
        self._initializer = None
        self._trace = []
        self._warned_about_exploitation = False
        # A second run fits a second surrogate, and a caller whose kernel and optimizer still do
        # not match has to be told a second time.  Both model selection flags go back with it.
        self._warned_about_model_selection = False
        self._warned_about_frozen_hyperparameters = False
        self.last_acquisition_run = None

    def best(self) -> tuple[Any, float]:
        """Return the best observation as ``(candidate, y)``.

        Best is the **largest** observed value: the loop maximises.

        Raises
        ------
        RuntimeError
            If called before any observations have been made.
        """
        if not self._y_list:
            raise RuntimeError("No observations available yet.")
        idx = int(np.argmax(np.array(self._y_list, dtype=float)))
        return self._x_list[idx], float(self._y_list[idx])

    def finalize(self) -> dict[str, Any]:
        """Transition to FINALIZED and return the optimization result.

        Returns
        -------
        dict with keys:
            ``best_tree``, ``best_y``, ``x``, ``y``, ``gp_model``, ``iterations``, ``trace``,
            ``dropped_suggestion``.

            ``gp_model`` is the surrogate the **last** :meth:`suggest` fitted, and nothing is
            fitted after it, so the model has seen nothing the run appended since.  A run that
            ends on an observation reports a model that never saw the pair it ended on.  A run
            that ends on an outstanding suggestion reports the model of that pass, which saw
            every pair the dataset held, because that pass appended none of its own.  Either way
            the fit is over the *distinct* pairs of what it was handed, which is fewer than the
            dataset holds whenever a term repeats in it.  A diagnostic that wants a surrogate
            over the whole dataset, as the fit scatter and the leave-one-out calibration of the
            acceptance checks do, asks :meth:`surrogate_over_dataset` for one rather than reading
            this.

            :attr:`last_acquisition_run` is left standing too, but it describes that same pass
            only where that :meth:`suggest` was asked to record a population.  Where it was not,
            the attribute still holds the most recent pass that was asked, which is an earlier
            one, or ``None`` if no pass was ever asked.

            ``iterations`` counts the passes :meth:`observe` closed.  A suggestion that never got
            a value is not one of them, and it is not a row of ``trace`` either.

            ``dropped_suggestion`` is the candidate of a suggestion that no value ever reached.
            It is ``None`` where no suggestion was open, and also where an open one already had
            its value in the dataset.  Finalizing with a suggestion open is allowed, and it is how
            an aborted run closes: a failing evaluation raises out of :meth:`optimize` between
            :meth:`suggest` and :meth:`observe`, and this call is what puts such a run into
            ``FINALIZED`` and names in one answer what it collected and which candidate it gave
            up on.  What such a candidate must not do is vanish.  An evaluation of the ask/tell
            layer may live outside this process and may already have been paid for, and
            :meth:`observe` is the only way to get it into the dataset, so the term is named
            here and logged instead.  The suggestion itself is
            cleared on every path, so a snapshot taken after this reports none outstanding.

            ``trace`` is the per-pass table of the trace readings.  See :attr:`trace`.

        Raises
        ------
        RuntimeError
            If called in an invalid state, or on an empty dataset.  The loop answers with a term
            of maximal observed value, and a run that observed nothing has none: reporting
            ``(None, nan)`` instead reads downstream as a completed run with a worthless optimum.
        """
        if self._bo_state not in (
            BOState.INITIALIZED, BOState.OBSERVED, BOState.SUGGESTED
        ):
            raise RuntimeError(
                f"finalize() is not allowed in state {self._bo_state.value}."
            )

        best_tree, best_y = self.best()

        # A suggestion was dropped when the dataset holds no value for its term.  The state does
        # not answer that question: observe() writes the term and its value before it moves the
        # state, so a run interrupted in between sits in SUGGESTED with the value already
        # recorded, and calling that term dropped would state the reverse of the truth.  The
        # dataset answers it exactly, because the two lists are appended in step and read by
        # position everywhere else, so a term has a value if and only if a y entry stands beside
        # it.  Membership in the duplicate index is not the same test: the index is written
        # between the two appends, so an interrupt there leaves it claiming a value the dataset
        # does not hold.  A suggestion cannot turn up paired here by accident either, because
        # suggest() replaces or refuses any candidate the observed set already holds.
        valued_terms = self._x_list[: len(self._y_list)]
        outstanding = self._last_suggestion
        dropped = None
        if outstanding is not None and outstanding.candidate not in valued_terms:
            dropped = outstanding.candidate
            _LOG.warning(
                "the run is finalized with the suggestion %s still outstanding.  No value ever "
                "reached that pass, so the dataset holds none for the term and the pass is not "
                "among the %d this result counts.  An evaluation that fails leaves the "
                "closed loop in exactly this state, and a caller who did measure the term hands "
                "it to observe() before finalizing.  The result names it under "
                "dropped_suggestion.",
                dropped,
                self._iteration,
            )
        self._last_suggestion = None

        self._bo_state = BOState.FINALIZED
        return {
            "best_tree": best_tree,
            "best_y": best_y,
            "x": np.asarray(self._x_list, dtype=object),
            "y": np.asarray(self._y_list, dtype=float),
            "gp_model": self._model,
            "iterations": self._iteration,
            "trace": self.trace,
            "dropped_suggestion": dropped,
        }

    def get_state_snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of the current Ask/Tell state."""
        return {
            "state": self._bo_state.value,
            "x_list": list(self._x_list),
            "y_list": list(self._y_list),
            "iteration": self._iteration,
            "last_suggestion": (
                None if self._last_suggestion is None
                else self._last_suggestion.candidate
            ),
        }

    # -------------------------------------------------------------------------
    # The closed loop
    # -------------------------------------------------------------------------

    def _check_pass_configuration(
        self, acquisition_fitness_mode: Literal["single", "batch"]
    ) -> None:
        """Refuse a configuration no pass of this run could use, before the design is drawn.

        A pass conditions the surrogate, builds one of the three acquisitions and hands it to the
        evolutionary algorithm, and what those steps read is fixed before the run starts.  Left to
        the pass, a name that is not one of the three, a parameter outside the range its
        acquisition admits, a missing evolutionary algorithm and a score that algorithm cannot be
        given one candidate at a time all surface after the initial design has been drawn and
        evaluated.  Those evaluations are what a run pays its budget for, and on the search this
        framework is built for one of them trains a network.  The budget itself is checked ahead
        of the design for that same reason.

        The acquisition and the algorithm are checked, and nothing else.  The parameter of an
        acquisition this run will not build is not this run's parameter, so the exploration
        parameter is read only under an upper confidence bound and the margin only under a
        probability of improvement.  The arguments this class forwards to
        ``GaussianProcessRegressor`` keep scikit-learn's rules and its messages rather than a copy
        of them here, which leaves the cost in place for them: a wrong ``alpha`` or a wrong entry
        in ``gp_params`` still surfaces at the first fit, with the design drawn and evaluated
        already.  ``kernel_optimizer`` is besides a default that ``gp_params`` overrides at the
        fit, so the value the constructor was given need not be the one a run uses.

        The checks inside :meth:`suggest` and inside the acquisitions themselves stay where they
        are.  Every value read here is a public attribute, and one assigned after construction
        reaches a pass without ever passing this.  A caller who reaches a pass through
        :meth:`initialize` and :meth:`suggest` rather than through :meth:`optimize` passes nothing
        here either, and pays the whole design as before.

        Args:
            acquisition_fitness_mode (Literal["single", "batch"]): How the evolutionary algorithm
                will be asked to score its population.

        Raises:
            RuntimeError: If no evolutionary algorithm was configured.
            ValueError: If the acquisition is not one of the three, if the parameter of the
                acquisition this run would build lies outside its range, or if that acquisition
                cannot be scored the way ``acquisition_fitness_mode`` asks.
        """
        if self.optimizer is None:
            raise RuntimeError(_NO_OPTIMIZER)

        # Only a string can name one of the three.  Looking a name up in the table before saying
        # so answers a list or a dict with a TypeError about hashing, where the caller was
        # promised a ValueError about the name.
        name = self.acquisition_function
        acquisition = _ACQUISITIONS.get(name) if isinstance(name, str) else None
        if acquisition is None:
            raise ValueError(_unknown_acquisition(name))

        if name == "UpperConfidenceBound":
            require_beta(self.ucb_beta)
        elif name == "ProbabilityOfImprovement":
            require_margin(self.pi_margin)

        # AcquisitionOptimizer._objective takes the batch path for the string "batch" and scores
        # everything else one candidate at a time.  So the mode is read here as that same
        # question, batch or not batch: asked the other way round, a mode of "Batch" or of "" or
        # of None would pass here and be scored one candidate at a time all the same.
        if acquisition_fitness_mode != "batch" and acquisition.lower_bound is None:
            raise ValueError(unbounded_below_message(name, "acquisition_fitness_mode"))

    def optimize(
        self,
        objective: Callable[[Any], float],
        budget: int,
        *,
        initial_size: int = 10,
        x0: Sequence[Any] | None = None,
        y0: Sequence[float] | None = None,
        gp_params: dict[str, Any] | None = None,
        alpha: float = _JITTER,
        verbose: bool = False,
        acquisition_fitness_mode: Literal["single", "batch"] = "batch",
        record_population: bool = False,
    ) -> dict[str, Any]:
        """Run the closed Bayesian optimization loop and return its result.

        The algorithm, line by line: the initial dataset comes from the initializer, which either
        collects terms from a sampler's stream or draws each member biased away from the ones
        already drawn, then each of the ``budget`` passes conditions the Gaussian process on the
        distinct pairs of the dataset, hands the current acquisition ``t -> alpha(t; D)`` to the
        evolutionary algorithm as its fitness function, evaluates the individual it returns, and
        appends the pair.  The answer is a term of maximal observed value.

        Everything the algorithm fixes before a run is a parameter of this object: the kernel, the
        acquisition, the evolutionary algorithm, the initializer.  What a run takes is here.

        There is no stopping criterion beyond the budget, and the acquisition's own parameters do
        not change over the run: the schedule that lowered the exploration margin for the last few
        iterations had no counterpart in the algorithm, and the margin it lowered is gone with it.

        Parameters
        ----------
        objective:
            The objective ``sigma . q``, which the loop **maximizes**.  A failing evaluation
            raises out of this call rather than becoming a value.  See :func:`_finite_or_raise`.
        budget:
            The evaluation budget ``B``: how many passes the loop runs.
        initial_size:
            The initial size ``mu_0``, passed to the initializer.
        x0, y0:
            An initial dataset supplied by the caller instead of drawn.  The algorithm has no
            counterpart for it and always draws, and the reason it exists is the same as for the
            ask/tell layer: evaluations may already have been paid for elsewhere.
        gp_params:
            Extra kwargs forwarded to ``GaussianProcessRegressor``.
        alpha:
            The numerical diagonal.  See :data:`_JITTER`.
        verbose:
            Log one line per pass.
        acquisition_fitness_mode:
            Whether the evolutionary algorithm scores its population one at a time or in batches.
        record_population:
            Keep the final population of each acquisition maximization.  See :meth:`suggest`.
            Over a whole loop only the last one survives, which is what the frontier read needs:
            the check asks whether *this* maximization returned a term on its own frontier, and
            every pass answers it the same way.

        Returns
        -------
        dict from :meth:`finalize`.  ``best_tree`` is the term the algorithm returns.  The rest,
        the dataset, the fitted surrogate, the pass count and the trace, is what a caller needs to
        report on the run and has no counterpart in the algorithm either.  Its
        ``dropped_suggestion`` is always ``None`` here, because this loop observes every term it
        suggests and an evaluation that fails raises out of this call instead of finalizing.

        Raises
        ------
        ValueError
            If ``budget`` is negative, or, where the run has passes to run, if the acquisition
            those passes would maximize is not one of the three, carries a parameter outside its
            range, or cannot be scored the way ``acquisition_fitness_mode`` asks.  All of these
            are checked before the initial design is drawn, so that a typo costs no evaluation of
            the objective.  Also for the dataset conditions of :meth:`initialize`.
        RuntimeError
            If this instance has already run.  One instance runs one loop, and a second run needs
            :meth:`reset` in between, which is also what restarts the default initializer's
            stream.  Also where the run has passes to run and no evolutionary algorithm was
            configured, checked ahead of the design as above.
        """
        if budget < 0:
            msg = f"a budget is a count of evaluations and cannot be negative: {budget}"
            raise ValueError(msg)

        # A run of no passes conditions nothing and maximizes nothing, so what a pass would have
        # read is not this run's configuration to refuse.  A zero budget is the initial design on
        # its own and stays that.
        if budget > 0:
            self._check_pass_configuration(acquisition_fitness_mode)

        if verbose:
            enable_verbose_logging()

        self.initialize(
            objective=objective,
            x0=x0,
            y0=y0,
            initial_size=initial_size,
            gp_params=gp_params,
            alpha=alpha,
        )

        for n in range(budget):
            t_suggest = time.time()
            suggestion = self.suggest(
                acquisition_fitness_mode=acquisition_fitness_mode,
                verbose=verbose,
                record_population=record_population,
            )
            t_suggest = time.time() - t_suggest

            t_eval = time.time()
            y_val = _finite_or_raise(objective(suggestion.candidate), suggestion.candidate)
            t_eval = time.time() - t_eval

            t_observe = time.time()
            self.observe(suggestion.candidate, y_val)
            t_observe = time.time() - t_observe

            if verbose:
                log_iteration(
                    self._logger,
                    iteration=n,
                    suggestion=suggestion,
                    y_observed=y_val,
                    suggest_time=t_suggest,
                    eval_time=t_eval,
                    observe_time=t_observe,
                )

        return self.finalize()
