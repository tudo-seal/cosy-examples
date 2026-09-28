from __future__ import annotations

import inspect
import logging
import math
import random
import time
from collections.abc import Callable, Hashable, Sequence
from typing import Any, ClassVar, Generic, Literal, TypeVar

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
    AcquisitionFactory,
    AcquisitionFunction,
    ExpectedImprovement,
    MarginalPosterior,
    ProbabilityOfImprovement,
    Surrogate,
    UpperConfidenceBound,
    require_beta,
    require_margin,
)
from .acquisition_optimizer import (
    AcquisitionMaximizer,
    AcquisitionOptimizer,
    resolve_fitness_mode,
    unbounded_below_message,
)
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
from .kernels.graph_kernel import clear_kernel_caches
from .kernels.tree_kernel import OrderedRootedSubtreeKernel
from .loop import (  # noqa: F401  (re-exported: callers and tests import them from here)
    DEFAULT_SIZE_BOUND,
    AskTellLoop,
    _check_request_against_space,
    _distinct_dataset,
    _finite_or_raise,
    _require_hashable,
)
from .state import BOState, Diagnostics, Suggestion

NT = TypeVar("NT", bound=Hashable)
T = TypeVar("T", bound=Hashable)
G = TypeVar("G", bound=Hashable)


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
_NO_OPTIMIZER = (
    "An optimizer is required.  Pass a EvolutionarySearch to the constructor, or an acquisition "
    "maximizer as maximizer=."
)


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
    return (
        f"Unknown acquisition_function {name!r}.  It has to be one of {known}, or a callable "
        "that builds an AcquisitionFunction for each pass."
    )


_BOTH_MAXIMIZERS = (
    "a pass maximizes its acquisition either with the evolutionary search (optimizer=) or with a "
    "maximizer (maximizer=), not both"
)


def _require_factory_signature(factory: Callable[..., Any]) -> None:
    """Refuse a callable that cannot be called as ``factory(gp, *, incumbent, known_points)``.

    What it builds needs what only a pass has, and is checked there (:func:`_built_by`); this is
    what can be refused before the design without calling it.  A callable whose signature Python
    cannot read is left to its first pass.

    Raises:
        TypeError: If the signature cannot take a pass's three arguments.
    """
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return
    try:
        signature.bind(None, incumbent=0.0, known_points=set())
    except TypeError as error:
        msg = (
            f"the acquisition_function {factory!r} cannot be called as factory(gp, *, incumbent, "
            f"known_points), which is how a pass builds its acquisition: {error}"
        )
        raise TypeError(msg) from None


def _built_by(
    factory: Callable[..., Any], gp: Any, *, incumbent: float, known_points: set[Any]
) -> AcquisitionFunction:
    """The acquisition a caller's factory builds for one pass, refused unless it can serve that pass.

    It has to be an :class:`AcquisitionFunction`, score with the surrogate the pass fitted, and hold
    at least the points the pass has observed: without them the known-point floor is gone, a pass
    may pick a point it has, and the duplicate's replacement becomes a random draw.  More points --
    terms still being evaluated elsewhere -- only floor more.

    Raises:
        TypeError: If the factory does not build an ``AcquisitionFunction``, or does not take the
            three arguments.
        ValueError: If the acquisition scores with another surrogate, or lacks the known points.
    """
    built = factory(gp, incumbent=incumbent, known_points=known_points)
    if not isinstance(built, AcquisitionFunction):
        msg = (
            f"the acquisition factory {factory!r} returned {type(built).__name__}, not an "
            "AcquisitionFunction"
        )
        raise TypeError(msg)
    if built.gp is not gp:
        msg = (
            f"the acquisition factory's {type(built).__name__} scores with another surrogate than "
            "the one the pass fitted and handed it"
        )
        raise ValueError(msg)
    held = built.known_points
    if held is None or not known_points <= set(held):
        msg = (
            f"the acquisition factory's {type(built).__name__} does not hold the known points the "
            "pass handed it: without them nothing keeps an observed term below a novel one"
        )
        raise ValueError(msg)
    return built


# The diagnostics keys a trace row is read from.  ``Diagnostics`` declares four more, and no
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
            diagnostics without every key a row is read from.  Every pass this class suggests
            carries all of them; a term of the design phase carries none, and :meth:`observe`
            takes it back without asking this.  One that does not came from somewhere else, and a row
            of substitute values would be a run description nobody measured.
    """
    diagnostics = suggestion.diagnostics
    if diagnostics is None:
        msg = (
            "this suggestion carries no diagnostics, so there is nothing to write a trace row "
            "from.  Every pass suggest() returns carries them."
        )
        raise ValueError(msg)
    if suggestion.acquisition_value is None:
        msg = (
            "this suggestion carries no acquisition value, so its trace row would have no score "
            "at the pick.  Every pass suggest() returns carries one."
        )
        raise ValueError(msg)
    missing = [key for key in _TRACE_DIAGNOSTICS_KEYS if key not in diagnostics]
    if missing:
        msg = (
            f"this suggestion's diagnostics are missing {', '.join(missing)}, so a trace row "
            "cannot be written from them.  Every pass suggest() returns carries all of "
            "them, and a row assembled around a gap would be a run description nobody measured."
        )
        raise ValueError(msg)
    return diagnostics, suggestion.acquisition_value


class BayesianOptimization(AskTellLoop[NT, T, G]):
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

    Workflow, ask/tell with the design as the loop's first phase::

        bo = BayesianOptimization(search_space, request, optimizer=ea)
        bo.check_configuration()          # refuse a pass configuration before anything is paid
        bo.initialize(initial_size=mu_0)  # draws the design, evaluates nothing; or design=terms
        for _ in range(len(bo.design) + budget):
            suggestion = bo.suggest()     # the design terms first, in order, then the passes
            y = objective(suggestion.candidate)
            bo.observe(suggestion.candidate, y)
        result = bo.finalize()

    The design phase moves the evaluation of the initial design out of :meth:`initialize` and
    changes nothing else: the terms it hands out are the terms the closed path evaluates, and the
    first pass after it is the pass that follows ``initialize(x0, y0)`` on the same pairs.  What it
    buys is the seam between drawing and evaluating.  A caller writes every value as it is measured,
    takes over values that already exist by observing them instead of measuring them again, and
    keeps every value it paid for when an evaluation fails, because each one is in the dataset the
    moment it is observed.

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
        Which of the three standard acquisition scores to maximize, by name, or an
        :class:`~bayesian_optimization.acquisition_function.AcquisitionFactory` that builds one of
        the caller's own for each pass from the surrogate it fitted, the incumbent and the points
        observed.  Any callable is taken for a factory, and a class with that signature, such as
        ``ExpectedImprovement``, is one.  A callable that cannot be called that way is refused
        before anything is paid; what it builds is checked at every pass, since only a pass has a
        surrogate to build from.
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
    maximizer:
        What maximizes the acquisition instead of the evolutionary search: an
        :class:`~bayesian_optimization.acquisition_optimizer.AcquisitionMaximizer`, such as
        :class:`~bayesian_optimization.acquisition_optimizer.SampleMaximizer`, the best of one
        sample.  Given instead of ``optimizer``, never beside it, since a pass has one maximization.
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
        search tree, and says so in a warning once per run.  That is a placeholder for toy spaces
        and the first thing to replace on a real one, in two respects:

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
    surrogate_model:
        A :class:`~bayesian_optimization.acquisition_function.Surrogate` of the caller's own in
        place of the Gaussian process: every pass conditions it on the distinct pairs of the
        dataset, and the acquisition, the diagnostics' fits and the result read the posterior its
        ``fit`` answers, a new one for every fit.  The settings that configure the Gaussian process
        -- ``kernel``, ``kernel_optimizer``, ``n_restarts_kernel_optimizer``, ``gp_normalize_y``,
        and a run's ``gp_params`` and ``alpha`` -- reach nothing beside it and are refused.  A run
        layer leaves the reads only a Gaussian process answers empty.
    """

    #: A pass of Bayesian optimization, as the CIFAR driver has always named it in its rows.
    PASS_PHASE: ClassVar[str] = "bo_step"

    def __init__(
        self,
        search_space: SolutionSpace[NT, T, G] | None,
        request: NT | None = None,
        *,
        acquisition_function: Literal[
            "ExpectedImprovement", "ProbabilityOfImprovement", "UpperConfidenceBound"
        ]
        | AcquisitionFactory = "ExpectedImprovement",
        ucb_beta: float = 2.0,
        pi_margin: float = 0.0,
        kernel: Kernel | None = None,
        kernel_optimizer: str | None = None,
        n_restarts_kernel_optimizer: int = 20,
        optimizer: EvolutionarySearch[NT, T, G] | None = None,
        maximizer: AcquisitionMaximizer | None = None,
        initializer: Initializer[NT, T, G] | None = None,
        seed: int | None = None,
        sampler: Sampler | None = None,
        gp_normalize_y: bool = True,
        surrogate_model: Surrogate | None = None,
    ) -> None:
        # The pair is refused where the caller writes it down, so a mistyped target costs no
        # evaluation of the objective.  ``optimize`` refuses a negative budget in the same way and
        # for the same reason.  This reads the arguments, so it is not the guarantee on its own:
        # both attributes are public, and the pair is checked again in ``_query``, where it
        # becomes a query.
        super().__init__(
            search_space, request, initializer=initializer, seed=seed, sampler=sampler
        )
        self.acquisition_function = acquisition_function
        self.ucb_beta = float(ucb_beta)
        self.pi_margin = float(pi_margin)
        self.kernel: Kernel = (
            OrderedRootedSubtreeKernel() if kernel is None else kernel
        )
        self.kernel_optimizer = kernel_optimizer
        self.n_restarts_kernel_optimizer = n_restarts_kernel_optimizer
        if optimizer is not None and maximizer is not None:
            raise ValueError(_BOTH_MAXIMIZERS)
        self.optimizer = optimizer
        self.maximizer = maximizer

        if surrogate_model is not None:
            # A parameter that reaches nothing is a parameter that lies: beside a caller's
            # surrogate the Gaussian process is never built, so its settings are refused where
            # they are written down, against the defaults of this signature.
            defaults = inspect.signature(BayesianOptimization.__init__).parameters
            given = {
                "kernel": kernel,
                "kernel_optimizer": kernel_optimizer,
                "n_restarts_kernel_optimizer": n_restarts_kernel_optimizer,
                "gp_normalize_y": gp_normalize_y,
            }
            configured = [name for name, value in given.items() if value != defaults[name].default]
            if configured:
                msg = (
                    f"{', '.join(configured)} configure the Gaussian process, which "
                    "surrogate_model replaces: beside it they would reach nothing"
                )
                raise ValueError(msg)
        self.surrogate_model = surrogate_model

        self._gp_normalize_y: bool = gp_normalize_y

        # --- The surrogate's state; the loop's own is AskTellLoop's -----------
        self._model: MarginalPosterior | None = None
        self._alpha: float = _JITTER
        self._gp_params: dict[str, Any] | None = None
        self._warned_about_model_selection: bool = False
        self._warned_about_frozen_hyperparameters: bool = False
        self._warned_about_exploitation: bool = False
        # The loop logs under this module's name, as it did before it was factored out.
        self._logger = _LOG
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

    def suggest(
        self,
        *,
        acquisition_fitness_mode: Literal["auto", "single", "batch"] = "batch",
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

        In the DESIGN state it hands out the next term of the design instead, in the design's
        order, and fits and maximizes nothing: the suggestion carries no acquisition value, its
        diagnostics name the phase ``"design"`` and the term's position, and none of the arguments
        below is read.  See :meth:`initialize`.

        Parameters
        ----------
        acquisition_fitness_mode:
            How the evolutionary algorithm is to score its population: ``"batch"`` a generation
            at a time, ``"single"`` one candidate at a time, ``"auto"`` whichever of the two the
            adapter would pick, which is ``"batch"``.  A fourth value is refused rather than
            scored one candidate at a time.  See
            :func:`~bayesian_optimization.acquisition_optimizer.resolve_fitness_mode`.
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
            If the dataset gives one term two different values, if the configured acquisition
            is not one of the three this class knows, if ``acquisition_fitness_mode`` is not one
            of the three modes, or if it asks for ``"single"`` and that acquisition has no lower
            bound to floor the terms already evaluated against.
        """
        return self._suggest(
            verbose,
            lambda: self._propose_pass(
                acquisition_fitness_mode=acquisition_fitness_mode,
                verbose=verbose,
                record_population=record_population,
            ),
        )

    def _propose(self, verbose: bool) -> Suggestion:
        """A pass under the default arguments of :meth:`suggest`.

        Args:
            verbose (bool): Log the suggestion.

        Returns:
            Suggestion: The pass.
        """
        return self._propose_pass(
            acquisition_fitness_mode="batch", verbose=verbose, record_population=False
        )

    def _propose_pass(
        self,
        *,
        acquisition_fitness_mode: Literal["auto", "single", "batch"],
        verbose: bool,
        record_population: bool,
    ) -> Suggestion:
        """One pass: fit the surrogate, maximize the acquisition, replace a duplicate.

        The body of :meth:`suggest`; the loop's state machine around it is
        :meth:`AskTellLoop._suggest`, the same for every strategy.

        Args:
            acquisition_fitness_mode (Literal["auto", "single", "batch"]): See :meth:`suggest`.
            verbose (bool): See :meth:`suggest`.
            record_population (bool): See :meth:`suggest`.

        Returns:
            Suggestion: The pass.
        """
        if self.optimizer is None and self.maximizer is None:
            raise RuntimeError(_NO_OPTIMIZER)
        if self.optimizer is not None and self.maximizer is not None:
            raise ValueError(_BOTH_MAXIMIZERS)

        # The mode is read here rather than at the maximization that uses it, which runs only
        # after the surrogate has been fitted on the distinct pairs of the dataset.  A mode
        # outside the three would otherwise be refused once that fit had already been paid.
        fitness_mode = resolve_fitness_mode(acquisition_fitness_mode)

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

        # The incumbent the three scores compare against is the best observation, and best means
        # largest.  Read off the whole dataset rather than off the distinct pairs, which is the
        # same number, since removing repetitions removes no value, but says which of the two it
        # is a property of.
        incumbent = float(np.max(np.array(self._y_list, dtype=float)))

        # --- Build acquisition function ---------------------------------------
        af = self._build_acquisition(model, incumbent)

        # --- Optimize acquisition function ------------------------------------
        acq_opt: AcquisitionMaximizer = (
            self.maximizer if self.maximizer is not None else AcquisitionOptimizer(self.optimizer)
        )
        population: list[Any] | None = None
        generations: list[Any] = []
        if record_population:
            candidate, population, generations = acq_opt.maximize_with_population(
                af, self._query(), mode=fitness_mode
            )
        else:
            candidate = acq_opt.maximize(af, self._query(), mode=fitness_mode)

        if candidate is None:
            raise RuntimeError("Optimizer did not return a candidate.")

        # What the maximization returned, before the rejection path below may replace it.  The
        # frontier read diagnoses the maximization, and a replacement drawn at random was never
        # scored against this population, so read in its place it comes out behind every member.
        maximiser = candidate

        # --- Rejection of duplicates: a deliberate deviation from the algorithm -
        #
        # Two mechanisms carry the rejection of duplicates.  The acquisition scores known points
        # below every genuine candidate, and a candidate that gets through anyway is replaced by a
        # fresh draw.  ``_replace_duplicate`` draws it, and says why the deviation is deliberate.
        fallback_used = candidate in self._x_set
        if fallback_used:
            candidate = self._replace_duplicate()

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
                acquisition_name=type(af).__name__,
            )

        diagnostics: Diagnostics = {
            "timestamp": time.time(),
            "incumbent": incumbent,
            "iteration": self._iteration,
            "fallback_used": fallback_used,
            # Zero or one: ``_replace_duplicate`` draws once and one draw settles it.  The
            # column stays because the experiment logs carry it.
            "fallback_attempts": int(fallback_used),
            "mean_at_pick": float(mean_at_pick[0]),
            "deviation_at_pick": float(deviation_at_pick[0]),
            "phase": "main",
        }

        return Suggestion(
            candidate=candidate,
            acquisition_value=acq_value,
            diagnostics=diagnostics,
        )

    def _build_acquisition(
        self, model: MarginalPosterior, incumbent: float
    ) -> AcquisitionFunction:
        """Build the acquisition of one pass, with the known-point floor beneath it.

        The name and the two parameters are public attributes, so a pass reads them for itself
        rather than trusting what the run started with.  A name assigned to a live object reaches
        its refusal here, and the check a run passes before it spends anything does not stand in
        for this one.

        Args:
            model (MarginalPosterior): The surrogate this pass fitted.
            incumbent (float): The largest observed value, which the two improvement scores
                measure a candidate against.

        Returns:
            AcquisitionFunction: The acquisition to maximize.

        Raises:
            ValueError: If ``acquisition_function`` is not one of the three.
        """
        # A *copy* of the observed set.  Handing over the live one made the acquisition a view of
        # the loop rather than a record of this pass: the next observe() adds to it, and the same
        # object then answers the known-point floor at the very term it had just chosen.  While a
        # pass runs the two are equal, since nothing observes in between, so this changes no
        # decision.  It changes what an acquisition still means once the pass is over, which is
        # exactly what the diagnostics keep one for.
        known_points = set(self._x_set)

        setting = self.acquisition_function
        if callable(setting) and not isinstance(setting, str):
            return _built_by(setting, model, incumbent=incumbent, known_points=known_points)
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
        return af

    def _replace_duplicate(self) -> Any:
        """Draw a term the loop has not evaluated yet, in place of one it already has.

        The algorithm as stated lets an evaluation repeat and removes duplicates only when
        conditioning the surrogate, because an evaluation may repeat and an exact observation
        repeats identically.  Rejecting a repeated proposal and adding a jitter to the diagonal
        are the two alternatives named beside it.  This implementation takes rejection: a repeated
        evaluation costs a call to the expensive quality measure and buys the surrogate nothing,
        and on a finite search space a loop that is allowed to repeat can spend a whole budget
        standing still.

        This is the one place where the code knowingly departs from the algorithm as stated, and
        the resolution belongs on the specification side rather than here.

        One draw settles it.  ``_sample_fallback_tree`` returns an inhabitant outside the observed
        set or raises, so a retry loop here would have nothing to retry.  The bounded retry that
        used to stand in its place could not reach its second pass, and its error message
        described a state the code cannot be in.

        Returns:
            Any: A term outside the observed set.

        Raises:
            RuntimeError: If there is no search space to draw a replacement from, or if the
                sampler returns a term that has already been evaluated.  ``_sample_fallback_tree``
                raises on its own account when the bounded space is exhausted.
        """
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
        return candidate

    def _fit_surrogate(
        self, terms: Sequence[Any], values: Sequence[float]
    ) -> MarginalPosterior:
        """Condition the surrogate on ``(terms, values)`` with this run's configuration.

        The one place a surrogate is built, so that the one a diagnostic reads is the one a pass
        maximized against: same kernel, same diagonal term, same model selection, same seed; or
        the caller's ``surrogate_model``, handed the same pairs.

        Args:
            terms (Sequence[Any]): The terms to condition on, pairwise distinct.
            values (Sequence[float]): Their observed values.

        Returns:
            MarginalPosterior: The fitted surrogate.

        Raises:
            ValueError: If the caller's surrogate answers the posterior the last pass fitted.
        """
        if self.surrogate_model is not None:
            fitted = self.surrogate_model.fit(list(terms), [float(value) for value in values])
            if self._model is not None and fitted is self._model:
                msg = (
                    "the surrogate's fit answered the posterior it answered before; every fit "
                    "has to answer a new posterior, since a recorded acquisition holds the one "
                    "its pass maximized against"
                )
                raise ValueError(msg)
            return fitted
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
        posterior: MarginalPosterior = model
        return posterior

    def surrogate_over_dataset(self) -> MarginalPosterior:
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
            MarginalPosterior: The fitted surrogate.

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
    def surrogate(self) -> MarginalPosterior | None:
        """The surrogate of the last :meth:`suggest`, or ``None`` before the first one.

        The model that pass actually maximized its acquisition against, kernel hyperparameters
        included, which is what a per-pass diagnostic is a statement about.  ``finalize()``
        reports the same object at the end of a run under ``gp_model``, and this exposes it while
        the run is still going, so a caller can record how the surrogate changed instead of only
        how it ended up.

        Returns:
            MarginalPosterior | None: The model, or None if no pass has fitted one.
        """
        return self._model

    def surrogate_over(
        self, terms: Sequence[Any], values: Sequence[float]
    ) -> MarginalPosterior:
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
            MarginalPosterior: The fitted surrogate.

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

    def initialize(
        self,
        *,
        objective: Callable[[Any], float] | None = None,
        x0: Sequence[Any] | None = None,
        y0: Sequence[float] | None = None,
        initial_size: int = 10,
        gp_params: dict[str, Any] | None = None,
        alpha: float = _JITTER,
        design: Sequence[Any] | None = None,
    ) -> None:
        """Build the initial dataset, or start the design phase that builds it.

        See :meth:`AskTellLoop.initialize` for ``objective``, ``x0``, ``y0``, ``initial_size`` and
        ``design``; this class adds the two arguments of the surrogate's fit.

        Parameters
        ----------
        gp_params:
            Extra kwargs forwarded to ``GaussianProcessRegressor``.
        alpha:
            The numerical diagonal added to the Gram matrix.  See :data:`_JITTER`.
        """
        if self.surrogate_model is not None and (gp_params or alpha != _JITTER):
            given = [
                name for name, set_here in (("gp_params", bool(gp_params)), ("alpha", alpha != _JITTER))
                if set_here
            ]
            msg = (
                f"{', '.join(given)} configure the Gaussian process, which surrogate_model "
                "replaces: beside it they would reach nothing"
            )
            raise ValueError(msg)
        super().initialize(
            objective=objective, x0=x0, y0=y0, initial_size=initial_size, design=design
        )
        # Stored once the dataset or the design exists, as before the loop was factored out: a
        # call that raises leaves the arguments of a run in progress untouched.
        self._alpha = alpha
        self._gp_params = gp_params

    def reset(self) -> None:
        """Reset to UNINITIALIZED, clearing all observations and the surrogate.

        See :meth:`AskTellLoop.reset` for what a reset keeps and what it rebuilds.

        Resetting empties the graph kernels' caches, and that reaches past this optimization.
        They live on their module and their keys hold the terms, so the terms of a finished run
        would otherwise stay alive with nothing left to read them.  Emptying takes the entries of
        every other kernel in the process with it, whatever kernel this optimization holds.  Their
        keys name the term and the translation and never the kernel that wrote the entry, so a
        dropped entry costs the conversion or the kernel evaluation again and nothing else.
        """
        super().reset()
        self._model = None
        self._warned_about_exploitation = False
        # A second run fits a second surrogate, and a caller whose kernel and optimizer still do
        # not match has to be told a second time.  Both model selection flags go back with it.
        self._warned_about_model_selection = False
        self._warned_about_frozen_hyperparameters = False
        self.last_acquisition_run = None
        clear_kernel_caches()

    def _check_pass_suggestion(self, suggestion: Suggestion) -> None:
        """A pass is refused before its value enters the dataset if no trace row can be read from it.

        Args:
            suggestion (Suggestion): The pass.
        """
        _require_trace_diagnostics(suggestion)

    def _record_pass(self, suggestion: Suggestion, observed: float) -> None:
        """A closed pass becomes a row of the run trace.

        Args:
            suggestion (Suggestion): The pass.
            observed (float): Its value.
        """
        self._trace.append(self._trace_record(suggestion, observed))

    def _result_model(self) -> MarginalPosterior | None:
        """The surrogate of the last pass, reported under ``gp_model``.

        Returns:
            MarginalPosterior | None: The model, or None before the first pass.
        """
        return self._model

    def check_configuration(
        self, acquisition_fitness_mode: Literal["auto", "single", "batch"] = "batch"
    ) -> None:
        """Refuse, before anything is evaluated, a configuration no pass of this run could use.

        :meth:`optimize` asks this for itself before it draws the design.  An ask/tell caller who
        will run passes asks it before :meth:`initialize`, so that a mistyped acquisition, a
        missing evolutionary algorithm or a score that algorithm cannot be given one candidate at
        a time is refused before the design is paid, not at the first :meth:`suggest` after it.
        A run of no passes maximizes nothing and has nothing to ask.

        Args:
            acquisition_fitness_mode (Literal["auto", "single", "batch"]): How the evolutionary
                algorithm will be asked to score its population, the value the passes will hand
                to :meth:`suggest`. (Default value = "batch")

        Raises:
            RuntimeError: If no evolutionary algorithm was configured.
            ValueError: As :meth:`optimize` refuses a configuration before its design.
        """
        self._check_pass_configuration(acquisition_fitness_mode)

    def _check_pass_configuration(
        self, acquisition_fitness_mode: Literal["auto", "single", "batch"]
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
        :meth:`initialize` and :meth:`suggest` rather than through :meth:`optimize` passes here only
        by calling :meth:`check_configuration` before :meth:`initialize`; one that does not pays the
        whole design before the first pass refuses the configuration.

        Args:
            acquisition_fitness_mode (Literal["auto", "single", "batch"]): How the evolutionary
                algorithm will be asked to score its population.

        Raises:
            RuntimeError: If no evolutionary algorithm was configured.
            ValueError: If the acquisition is not one of the three, if the parameter of the
                acquisition this run would build lies outside its range, if the mode is not one
                of the three modes, or if that acquisition cannot be scored the way
                ``acquisition_fitness_mode`` asks.
        """
        if self.optimizer is None and self.maximizer is None:
            raise RuntimeError(_NO_OPTIMIZER)
        if self.optimizer is not None and self.maximizer is not None:
            raise ValueError(_BOTH_MAXIMIZERS)

        # Only a string can name one of the three.  Looking a name up in the table before saying
        # so answers a list or a dict with a TypeError about hashing, where the caller was
        # promised a ValueError about the name.
        name = self.acquisition_function
        lower_bound: float | None = None
        label: str | None = None
        if callable(name) and not isinstance(name, str):
            # What a factory builds needs what only a pass has -- a fitted surrogate, the incumbent,
            # the observed points -- and is checked at every pass, its lower bound with it; what is
            # refused here is a callable that cannot be called the way a pass calls it.
            _require_factory_signature(name)
        else:
            acquisition = _ACQUISITIONS.get(name) if isinstance(name, str) else None
            if acquisition is None:
                raise ValueError(_unknown_acquisition(name))

            if name == "UpperConfidenceBound":
                require_beta(self.ucb_beta)
            elif name == "ProbabilityOfImprovement":
                require_margin(self.pi_margin)
            lower_bound, label = acquisition.lower_bound, name

        # The mode is resolved by the same function the maximization resolves it with, so that
        # what this refusal reads and what the run would use are one value.  Read as the plain
        # question "is this the string batch", it would refuse an upper confidence bound under
        # "auto", which is a mode that puts it on the batch path.
        mode = resolve_fitness_mode(acquisition_fitness_mode)
        if label is not None and mode != "batch" and lower_bound is None:
            raise ValueError(unbounded_below_message(label, "acquisition_fitness_mode"))

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
        acquisition_fitness_mode: Literal["auto", "single", "batch"] = "batch",
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
            How the evolutionary algorithm is to score its population.  See :meth:`suggest`.
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
            range, or cannot be scored the way ``acquisition_fitness_mode`` asks, or if that mode
            is not one of the three modes.  All of these are checked before the initial design is
            drawn, so that a typo costs no evaluation of the objective.  Also for the dataset
            conditions of :meth:`initialize`.
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
