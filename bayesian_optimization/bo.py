from __future__ import annotations

import logging
import random
import time
import warnings
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Generic, Literal, TypeVar

import numpy as np
from cosy.core.solution_space import SolutionSpace
from cosy.evolutionary_algorithms import Evolutionary, RandomLimitedDepthFirstInitialization
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from .acquisition_function import (
    AcquisitionFunction,
    DiversityUCB,
    ExpectedImprovement,
    UpperConfidenceBound,
)
from .acquisition_optimizer import AcquisitionOptimizer
from .diagnostics import get_logger, log_iteration
from .initial_sampling import (
    _generate_unique_initial_samples,
    _sample_fallback_tree,
)
from .kernels.kernel_base import StructuredKernelBase
from .kernels.tree_kernel import OrderedRootedSubtreeKernel
from .state import BOState, Diagnostics, Suggestion
from .transforms import IdentityTransform, Log1pTransform, YTransform
from .utils import canonical_tree

NT = TypeVar("NT", bound=Hashable)
T = TypeVar("T", bound=Hashable)
G = TypeVar("G", bound=Hashable)

_LOG = get_logger("bo")


class BayesianOptimization(Generic[NT, T, G]):
    """Bayesian Optimization over CoSy SolutionSpaces via an Ask/Tell interface.

    Workflow::

        bo = BayesianOptimization(search_space, request, optimizer=ea)
        bo.initialize(x0=..., y0=...)
        for _ in range(n_iters):
            suggestion = bo.suggest()
            y = obj_fun(suggestion.candidate)
            bo.observe(suggestion.candidate, y)
        result = bo.finalize()

    Parameters
    ----------
    search_space:
        CoSy solution space.  May be ``None`` when ``x0`` is always supplied
        explicitly (e.g. in tests).
    request:
        The non-terminal root request for the solution space.
    acquisition_function:
        Name of the acquisition function to use.
    kernel:
        sklearn kernel for the GP.  Defaults to ``OrderedRootedSubtreeKernel()``.
    kernel_optimizer:
        sklearn kernel hyperparameter optimizer (e.g. ``"fmin_l_bfgs_b"``).
    n_restarts_kernel_optimizer:
        Number of random restarts for kernel optimisation (never decremented).
    optimizer:
        Evolutionary algorithm adapter.
    optimizer_population_size / mutation_rate / recombination_rate:
        EA parameters.
    seed:
        RNG seed.
    max_depth:
        Maximum tree depth for the fallback initializer.
    y_transform:
        Bijective transformation applied to objective values before GP fitting.
        Defaults to ``Log1pTransform()``.
    gp_normalize_y:
        Passed to sklearn's ``GaussianProcessRegressor(normalize_y=...)``.
    max_duplicate_fallbacks:
        Hard cap on fallback sampling retries per ``suggest()`` call.
    normalize_y:
        Deprecated.  Use ``y_transform`` instead.
    """

    def __init__(
        self,
        search_space: SolutionSpace[NT, T, G] | None,
        request: NT | None = None,
        *,
        acquisition_function: Literal[
            "ExpectedImprovement", "UpperConfidenceBound", "DiversityUCB"
        ] = "ExpectedImprovement",
        kernel: Kernel | None = None,
        kernel_optimizer: str | None = None,
        n_restarts_kernel_optimizer: int = 20,
        optimizer: Evolutionary[NT, T, G] | None = None,
        optimizer_population_size: int = 100,
        optimizer_mutation_rate: float = 0.02,
        optimizer_recombination_rate: float = 0.95,
        seed: int | None = None,
        max_depth: int = 100,
        y_transform: YTransform | None = None,
        gp_normalize_y: bool = True,
        max_duplicate_fallbacks: int = 20,
        normalize_y: bool | None = None,
    ) -> None:
        # --- normalize_y → y_transform migration (F4 / F5) -------------------
        if normalize_y is not None and y_transform is not None:
            raise ValueError(
                "Set either normalize_y or y_transform, not both."
            )
        if normalize_y is not None:
            warnings.warn(
                "The 'normalize_y' parameter is deprecated.  "
                "Use 'y_transform=Log1pTransform()' or 'y_transform=IdentityTransform()' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            y_transform = Log1pTransform() if normalize_y else IdentityTransform()

        self.search_space = search_space
        self.request = request
        self.acquisition_function = acquisition_function
        self.kernel: Kernel = (
            OrderedRootedSubtreeKernel() if kernel is None else kernel
        )
        self.kernel_optimizer = kernel_optimizer
        self.n_restarts_kernel_optimizer = n_restarts_kernel_optimizer
        self.optimizer = optimizer
        self.optimizer_population_size = optimizer_population_size
        self.optimizer_mutation_rate = optimizer_mutation_rate
        self.optimizer_recombination_rate = optimizer_recombination_rate
        self.seed = seed
        self.max_depth = int(max_depth)

        self._y_transform: YTransform = (
            Log1pTransform() if y_transform is None else y_transform
        )
        self._gp_normalize_y: bool = gp_normalize_y
        self._max_duplicate_fallbacks: int = max_duplicate_fallbacks

        # --- Ask/Tell state ---------------------------------------------------
        self._bo_state: BOState = BOState.UNINITIALIZED
        self._x_list: list[Any] = []
        self._y_list: list[float] = []  # always raw user values
        self._x_set: set[Any] = set()
        self._last_suggestion: Suggestion | None = None
        self._model: GaussianProcessRegressor | None = None
        self._greater_is_better: bool = False
        self._alpha: float = 1e-6
        self._gp_params: dict[str, Any] | None = None
        self._initializer: RandomLimitedDepthFirstInitialization | None = None
        self._last_optimized_kernel: Kernel | None = None
        self._iteration: int = 0
        self._logger: logging.Logger = _LOG

    # -------------------------------------------------------------------------
    # Ask/Tell API
    # -------------------------------------------------------------------------

    def initialize(
        self,
        *,
        obj_fun: Callable[[Any], float] | None = None,
        x0: Sequence[Any] | None = None,
        y0: Sequence[float] | None = None,
        n_pre_samples: int = 10,
        gp_params: dict[str, Any] | None = None,
        alpha: float = 1e-6,
        greater_is_better: bool = False,
        initializer_kernel: StructuredKernelBase | None = None,
        sample_overapproximation_factor: int = 100,
    ) -> None:
        """Populate initial observations and transition to INITIALIZED state.

        Parameters
        ----------
        obj_fun:
            Objective function; required when ``y0`` is ``None``.
        x0:
            Initial candidate trees.  When ``None`` the fallback sampler is used
            (requires a non-``None`` ``search_space``).
        y0:
            Initial objective values (raw scale).  When ``None`` ``obj_fun``
            is called on each element of ``x0``.
        n_pre_samples:
            Number of samples to draw when ``x0`` is ``None``.
        gp_params:
            Extra kwargs forwarded to ``GaussianProcessRegressor``.
        alpha:
            GP noise level.
        greater_is_better:
            Whether the objective is maximised.
        initializer_kernel:
            Kernel used for DPP diversity during initial sampling.
        sample_overapproximation_factor:
            Over-sampling factor for the initial DPP pool.
        """
        if self._bo_state != BOState.UNINITIALIZED:
            raise RuntimeError(
                f"Cannot re-initialize: current state is {self._bo_state.value}. "
                "Call reset() first."
            )

        if initializer_kernel is None:
            initializer_kernel = OrderedRootedSubtreeKernel()

        # Create the initializer (needed for fallback sampling in suggest())
        if self._initializer is None and self.search_space is not None:
            self._initializer = RandomLimitedDepthFirstInitialization(
                self.search_space,
                self.request,
                max_depth=self.max_depth,
                rng=random.Random(self.seed),
            )

        # Resolve x0 (F2: unpack tuple correctly)
        if x0 is None:
            if self.search_space is None:
                raise NotImplementedError(
                    "initialize() without x0 requires a real search_space."
                )
            x0_list, _stats = _generate_unique_initial_samples(
                self._initializer,
                n_pre_samples,
                initializer_kernel=initializer_kernel,
                sample_overapproximation_factor=sample_overapproximation_factor,
            )
        else:
            x0_list = list(x0)

        # Resolve y0
        if y0 is None:
            if obj_fun is None:
                raise ValueError(
                    "obj_fun must be provided when y0 is None."
                )
            y0_list = [float(obj_fun(t)) for t in x0_list]
        else:
            y0_list = [float(v) for v in y0]

        if len(x0_list) != len(y0_list):
            raise ValueError(
                f"len(x0)={len(x0_list)} != len(y0)={len(y0_list)}."
            )

        # Validate hashability (F13)
        for item in x0_list:
            try:
                hash(item)
            except TypeError:
                raise TypeError(
                    f"All candidates in x0 must be hashable; "
                    f"got unhashable item of type {type(item).__name__!r}."
                )

        self._x_list = x0_list
        self._y_list = y0_list  # raw values
        self._x_set = set(self._x_list)
        self._last_suggestion = None
        self._greater_is_better = greater_is_better
        self._alpha = alpha
        self._gp_params = gp_params
        self._iteration = 0
        self._bo_state = BOState.INITIALIZED

    def suggest(
        self,
        *,
        ei_xi: float = 0.01,
        ucb_kappa: float = 2.0,
        acquisition_fitness_mode: Literal["single", "batch"] = "batch",
        diversity_ucb_kernel: StructuredKernelBase | None = None,
        verbose: bool = False,
    ) -> Suggestion:
        """Fit the GP and return the next suggested candidate.

        Returns
        -------
        Suggestion
            Frozen record with the candidate, acquisition value, and diagnostics.

        Raises
        ------
        RuntimeError
            If called in an invalid state or if the optimizer returns ``None``.
        """
        if self._bo_state not in (BOState.INITIALIZED, BOState.OBSERVED):
            raise RuntimeError(
                f"suggest() is not allowed in state {self._bo_state.value}."
            )
        if self.optimizer is None:
            raise RuntimeError(
                "An optimizer is required.  Pass an Evolutionary instance to the constructor."
            )

        # --- Transform y and fit GP -------------------------------------------
        yp_transformed = self._y_transform.forward(
            np.array(self._y_list, dtype=float)
        )

        gp_kwargs: dict[str, Any] = dict(self._gp_params) if self._gp_params else {}
        gp_kwargs.setdefault("kernel", self.kernel)
        gp_kwargs.setdefault("alpha", self._alpha)
        gp_kwargs.setdefault("n_restarts_optimizer", self.n_restarts_kernel_optimizer)
        gp_kwargs.setdefault("optimizer", self.kernel_optimizer)
        gp_kwargs.setdefault("normalize_y", self._gp_normalize_y)
        gp_kwargs.setdefault("random_state", self.seed)

        self._model = GaussianProcessRegressor(**gp_kwargs)
        self._model.fit(
            np.asarray(self._x_list, dtype=object), yp_transformed
        )
        model = self._model
        assert model is not None
        # Store optimized kernel for warm-start reporting only; never mutate self.kernel (F7)
        self._last_optimized_kernel = getattr(model, "kernel_", None)

        # Incumbent in transformed space (F6)
        incumbent_transformed = float(
            np.max(yp_transformed) if self._greater_is_better else np.min(yp_transformed)
        )

        # --- Build acquisition function ---------------------------------------
        af: AcquisitionFunction
        if self.acquisition_function == "ExpectedImprovement":
            af = ExpectedImprovement(
                gp=model,
                xi=ei_xi,
                greater_is_better=self._greater_is_better,
                known_points=self._x_set,
                incumbent=incumbent_transformed,
            )
        elif self.acquisition_function == "UpperConfidenceBound":
            af = UpperConfidenceBound(
                gp=model,
                greater_is_better=self._greater_is_better,
                known_points=self._x_set,
                incumbent=incumbent_transformed,
                kappa=ucb_kappa,
            )
        elif self.acquisition_function == "DiversityUCB":
            af = DiversityUCB(
                gp=model,
                kernel=diversity_ucb_kernel or OrderedRootedSubtreeKernel(),
                greater_is_better=self._greater_is_better,
                known_points=self._x_set,
                incumbent=incumbent_transformed,
                iteration=self._iteration,
            )
        else:
            raise ValueError(
                f"Unknown acquisition_function '{self.acquisition_function}'. "
                "Expected one of: 'ExpectedImprovement', 'UpperConfidenceBound', 'DiversityUCB'."
            )

        # --- Optimize acquisition function ------------------------------------
        acq_opt = AcquisitionOptimizer(
            self.optimizer,
            population_size=self.optimizer_population_size,
            mutation_rate=self.optimizer_mutation_rate,
            recombination_rate=self.optimizer_recombination_rate,
        )
        candidate = acq_opt.maximize(af, mode=acquisition_fitness_mode, verbose=verbose)

        if candidate is None:
            raise RuntimeError("Optimizer did not return a candidate.")

        # Candidates from crossover carry a stale ``size``/``_hash`` (see canonical_tree), which
        # would make the duplicate check below miss an architecture that was already evaluated --
        # and train it a second time.  Canonicalise before it reaches ``_x_set``.
        candidate = canonical_tree(candidate)

        # --- Fallback deduplication with hard limit (F18) ---------------------
        fallback_attempts = 0
        while candidate in self._x_set:
            if fallback_attempts >= self._max_duplicate_fallbacks:
                raise RuntimeError(
                    f"Fallback deduplication failed: could not find a novel candidate "
                    f"after {self._max_duplicate_fallbacks} attempts."
                )
            candidate = canonical_tree(_sample_fallback_tree(self._initializer, self._x_set))
            fallback_attempts += 1

        acq_value = float(af(candidate))

        incumbent_raw = float(
            self._y_transform.inverse(np.array([incumbent_transformed]))[0]
        )
        diagnostics: Diagnostics = {
            "timestamp": time.time(),
            "incumbent_transformed": incumbent_transformed,
            "incumbent_raw": incumbent_raw,
            "y_transform": self._y_transform.name,
            "iteration": self._iteration,
            "fallback_used": fallback_attempts > 0,
            "fallback_attempts": fallback_attempts,
            "phase": "main",
        }

        suggestion = Suggestion(
            candidate=candidate,
            acquisition_value=acq_value,
            diagnostics=diagnostics,
        )
        self._last_suggestion = suggestion
        self._bo_state = BOState.SUGGESTED
        return suggestion

    def observe(self, candidate: Any, y: float) -> None:
        """Record the objective value for the last suggested candidate.

        Parameters
        ----------
        candidate:
            Must match the candidate from the last ``suggest()`` call.
        y:
            Raw (untransformed) objective value.

        Raises
        ------
        RuntimeError
            If called outside the SUGGESTED state.
        ValueError
            If ``candidate`` does not match the last suggestion.
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

        self._x_list.append(candidate)
        self._x_set.add(candidate)
        self._y_list.append(float(y))
        self._iteration += 1
        self._bo_state = BOState.OBSERVED

    def reset(self) -> None:
        """Reset to UNINITIALIZED, clearing all observations.

        Constructor parameters are preserved.
        """
        self._bo_state = BOState.UNINITIALIZED
        self._x_list = []
        self._y_list = []
        self._x_set = set()
        self._last_suggestion = None
        self._model = None
        self._iteration = 0
        self._last_optimized_kernel = None
        self._initializer = None

    def best(self) -> tuple[Any, float]:
        """Return (best_candidate, best_y_raw) on the raw objective scale.

        Raises
        ------
        RuntimeError
            If called before any observations have been made.
        """
        if not self._y_list:
            raise RuntimeError("No observations available yet.")
        y_arr = np.array(self._y_list, dtype=float)
        idx = int(np.argmax(y_arr) if self._greater_is_better else np.argmin(y_arr))
        return self._x_list[idx], float(self._y_list[idx])

    def finalize(self) -> dict[str, Any]:
        """Transition to FINALIZED and return the optimization result.

        Returns
        -------
        dict with keys:
            ``best_tree``, ``best_y`` (raw), ``x``, ``y`` (raw),
            ``gp_model``, ``y_transform``, ``iterations``.
        """
        if self._bo_state not in (
            BOState.INITIALIZED, BOState.OBSERVED, BOState.SUGGESTED
        ):
            raise RuntimeError(
                f"finalize() is not allowed in state {self._bo_state.value}."
            )

        if self._y_list:
            best_tree, best_y_raw = self.best()
        else:
            best_tree, best_y_raw = None, float("nan")

        self._bo_state = BOState.FINALIZED
        return {
            "best_tree": best_tree,
            "best_y": best_y_raw,
            "x": np.asarray(self._x_list, dtype=object),
            "y": np.asarray(self._y_list, dtype=float),
            "gp_model": self._model,
            "y_transform": self._y_transform.name,
            "iterations": self._iteration,
        }

    def get_state_snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot of the current Ask/Tell state."""
        return {
            "state": self._bo_state.value,
            "x_list": list(self._x_list),
            "y_list": list(self._y_list),
            "iteration": self._iteration,
            "y_transform": self._y_transform.name,
            "last_suggestion": (
                None if self._last_suggestion is None
                else self._last_suggestion.candidate
            ),
        }

    # -------------------------------------------------------------------------
    # Convenience wrapper
    # -------------------------------------------------------------------------

    def bayesian_optimisation(
        self,
        n_iters: int,
        obj_fun: Callable[[Any], float],
        x0: Sequence[Any] | None = None,
        y0: Sequence[float] | None = None,
        n_pre_samples: int = 10,
        gp_params: dict[str, Any] | None = None,
        alpha: float = 1e-10,
        greater_is_better: bool = False,
        ei_xi: float = 0.01,
        max_depth: int = 100,
        exploit_last_iterations: int = 0,
        exploit_xi: float = 0.001,
        verbose: bool = False,
        acquisition_fitness_mode: Literal["single", "batch"] = "batch",
    ) -> dict[str, Any]:
        """Non-Ask/Tell convenience wrapper for full BO runs.

        Delegates to the Ask/Tell interface internally.  User-supplied ``x0``
        and ``y0`` are forwarded correctly (fixes F1).

        Returns
        -------
        dict from :meth:`finalize`.
        """
        if verbose:
            logging.getLogger("bayesian_optimization").setLevel(logging.INFO)

        # F1: pass real x0/y0 to initialize()
        self.initialize(
            obj_fun=obj_fun,
            x0=x0,
            y0=y0,
            n_pre_samples=n_pre_samples,
            gp_params=gp_params,
            alpha=alpha,
            greater_is_better=greater_is_better,
        )

        for n in range(n_iters):
            current_ei_xi = (
                exploit_xi if n >= n_iters - exploit_last_iterations else ei_xi
            )

            t_suggest = time.time()
            suggestion = self.suggest(
                ei_xi=current_ei_xi,
                acquisition_fitness_mode=acquisition_fitness_mode,
                verbose=verbose,
            )
            t_suggest = time.time() - t_suggest

            t_eval = time.time()
            y_val = float(obj_fun(suggestion.candidate))
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
