from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, Protocol

from cosy.evolutionary_algorithms import SampledInitialization
from cosy.search import Sampler

from .acquisition_function import AcquisitionFunction, require_term
from .diagnostics.frontier import GenerationRecord


def _generation_record(state: Any) -> GenerationRecord:
    """Summarize one generation of the driver's stream.

    The acquisition is a scalar, so the fitness values are floats and the summary is arithmetic.
    A search configured with a vector-valued quality measure would need an order rather than a
    mean, and this says so instead of averaging a tuple into a number that means nothing.

    Args:
        state (Any): The ``EAState`` the driver yielded.

    Returns:
        GenerationRecord: The record.

    Raises:
        TypeError: If a fitness is not a real number.
    """
    values = [state.fitness[individual] for individual in state.population]
    for value in (*values, state.best_fitness):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            msg = (
                f"a generation record summarises scalar fitness values and met {value!r}; an "
                f"acquisition is scalar, so a sequence here means the search was configured with "
                f"a different quality measure and this summary does not apply to it"
            )
            raise TypeError(msg)
    return GenerationRecord(
        generation=state.generation,
        best=float(state.best_fitness),
        population_best=max(float(value) for value in values),
        population_mean=sum(float(value) for value in values) / len(values),
        population_worst=min(float(value) for value in values),
        distinct_members=len(set(state.population)),
        last_improvement=state.last_improvement,
        offspring=len(state.offspring),
    )


def _known_points_of(acquisition_function: AcquisitionFunction) -> frozenset[Any]:
    """Return the acquisition function's known points as a set that is always safe to test."""
    known = getattr(acquisition_function, "known_points", None)
    return frozenset(known) if known else frozenset()


def _make_acquisition_objective_batch(
    acquisition_function: AcquisitionFunction,
) -> Any:
    """Wrap an AcquisitionFunction as a batch fitness objective with caching.

    Only genuine scores are cached.  The floor an already-evaluated candidate receives is not a
    score of its own, so caching it would carry a number from one generation into a later one
    where it no longer sits below everything, and a known point could win after all.  It is
    recomputed instead, and for an acquisition that is unbounded below it is recomputed from a
    running minimum over every genuine score seen so far, which keeps it consistent across the
    whole optimization rather than per generation.
    """
    cache: dict[Any, float] = {}

    def objective(sample: list[Any]) -> Mapping[Any, float]:
        known = _known_points_of(acquisition_function)
        # Checked before the cache is consulted: a non-term used to slip past the isinstance
        # filter here and receive the floor, which is a number the caller cannot tell from a
        # genuinely unpromising candidate.
        terms = [require_term(t) for t in sample]
        missing = [t for t in terms if t not in cache and t not in known]
        if missing:
            cache.update(acquisition_function.evaluate_batch(missing))

        floor = acquisition_function.known_point_floor(list(cache.values()))
        return {t: (floor if t in known else cache[t]) for t in terms}

    return objective


def unbounded_below_message(acquisition_name: str, mode_parameter: str) -> str:
    """Return the refusal an acquisition without a lower bound earns from single-sample scoring.

    The loop refuses the same pairing before it draws an initial design, and reads this text for
    it, so the early answer and the late one are one sentence and cannot drift apart.  Only the
    name of the parameter differs, because the two callers spell it differently.

    Args:
        acquisition_name (str): The acquisition that is unbounded below.
        mode_parameter (str): What the caller's own parameter for the mode is called.

    Returns:
        str: The message.
    """
    return (
        f"{acquisition_name} is unbounded below, so the score of an already-evaluated candidate "
        "can only be placed relative to the scores of the others in its generation.  Run it with "
        f'{mode_parameter}="batch", where every score of a generation is available before that '
        "floor is fixed."
    )


def _make_acquisition_objective_single(
    acquisition_function: AcquisitionFunction,
) -> Any:
    """Wrap an AcquisitionFunction as a single-sample fitness objective with caching.

    Known points are neither cached nor asked of the acquisition function: asked alone, there is
    no genuine score of the same batch for them to sit below.  They receive the floor, and here
    that floor has to come from the acquisition's own lower bound rather than from the scores seen
    so far.  A running minimum is empty at the first call, so the very first known point of a run
    would be scored at ``-1.0`` and beat every genuine candidate that scored below it.

    An acquisition without a lower bound cannot be carried this way, and this says so instead of
    scoring anyway.  Scoring an already-evaluated candidate at a floor, so that the search cannot
    hand it back, is a deliberate departure from the plain optimization loop, which evaluates
    whatever the run returns (see :meth:`AcquisitionFunction.known_point_floor`), and a departure
    that silently fails to do the one thing it was taken for is worse than the duplicate it was
    meant to prevent.

    Raises:
        ValueError: If the acquisition has known points but no lower bound.  Use ``mode="batch"``,
            where every score of the generation is available before the floor is fixed.
    """
    if _known_points_of(acquisition_function) and acquisition_function.lower_bound is None:
        raise ValueError(unbounded_below_message(type(acquisition_function).__name__, "mode"))

    cache: dict[Any, float] = {}

    def objective(sample: Any) -> float:
        require_term(sample)
        if sample in _known_points_of(acquisition_function):
            return acquisition_function.known_point_floor(())
        cached = cache.get(sample)
        if cached is not None:
            return cached
        value = float(acquisition_function(sample))
        cache[sample] = value
        return value

    return objective


def resolve_fitness_mode(mode: Any) -> Literal["single", "batch"]:
    """Answer which of the two scoring paths a requested fitness mode asks for.

    The driver knows three modes, and the one it defaults to is ``"auto"``: score a generation in
    one call where the annotation of the fitness function it was handed says that function takes
    a list, and one candidate at a time otherwise.  That question is settled here before it can
    be asked, because this module writes both of those functions, and the annotation selects the
    batch one.  So ``"auto"`` resolves to ``"batch"``, which is besides the only one of the two
    wrappers that carries an acquisition without a lower bound over known points.  The single one
    refuses that pairing, and takes an unbounded acquisition only where the one it is given has no
    known point to floor.

    Anything outside the three is refused rather than scored one candidate at a time.  The two
    wrappers answer alike, so a mistyped ``"batch"`` that fell through to single-sample scoring
    could not be told from a deliberate ``"single"`` by anything but the clock: the single path
    rebuilds the training side of the kernel matrix once per candidate, where the batch path
    builds it once for the whole generation.

    Args:
        mode (Any): The mode a caller asked for.  Typed as anything rather than as the three,
            because the values this has to answer for are the ones outside the type.

    Returns:
        Literal["single", "batch"]: The wrapper to build, and the mode to hand the driver.

    Raises:
        ValueError: If mode is none of the three.
    """
    if mode in ("auto", "batch"):
        return "batch"
    if mode == "single":
        return "single"
    msg = f'a fitness mode is one of "auto", "single" and "batch", and {mode!r} is none of them'
    raise ValueError(msg)


class AcquisitionMaximizer(Protocol):
    """What maximizes a pass's acquisition over the search space.

    ``maximize`` answers the pick; ``maximize_with_population`` answers the pick, the population it
    came from, and one :class:`~bayesian_optimization.diagnostics.frontier.GenerationRecord` per
    generation, which is what a run's frontier read and its search log are read from.  The
    evolutionary search is one, through :class:`AcquisitionOptimizer`; :class:`SampleMaximizer`,
    the best of one sample, is the baseline it is measured against.  A maximizer scores through
    the acquisition's known-point floor, as both of these do, or a pass may pick a point it has
    and its replacement becomes a random draw.
    """

    def maximize(
        self,
        acquisition_fn: AcquisitionFunction,
        query: Any,
        *,
        mode: Literal["auto", "single", "batch"] = "batch",
    ) -> Any: ...

    def maximize_with_population(
        self,
        acquisition_fn: AcquisitionFunction,
        query: Any,
        *,
        mode: Literal["auto", "single", "batch"] = "batch",
    ) -> tuple[Any, list[Any], list[GenerationRecord]]: ...


class AcquisitionOptimizer:
    """Adapter that wraps an evolutionary optimizer to maximize an AcquisitionFunction.

    The acquisition function enters the evolutionary search as a **fitness function**, not as a
    compositional measure folded over the term: it scores a candidate against every candidate
    evaluated before, so its value does not compose out of the values of the subterms.

    Its codomain is the reals, totally ordered, so any two fitness values are comparable and the
    selection methods of the field apply without adaptation.  One of them asks for more than an
    order: a proportional draw reads numbers, and it reads them through a scalarization into the
    *positive* reals.  ``ExpScalarization`` is that map, since ``exp`` is positive everywhere and
    preserves the order, and cosy's ``FitnessProportionalSelection`` takes it as a constructor
    argument, so a search configured with proportional selection already carries one.  Two
    consequences for acquisition values in particular: they are unbounded below, so no lift or
    shift makes them weights, and they can sit far enough below zero for ``exp`` to underflow,
    which is what the ``scale`` parameter of that scalarization is for.

    Population size and the two rates are no longer arguments here.  They are parameters of the
    search, fixed on the ``EvolutionarySearch`` before the run, together with its
    component choices.  A single run takes the search space and the quality measure and nothing
    else, so passing the parameters per call was the inversion the previous driver carried.

    Parameters
    ----------
    evolutionary:
        Object with an ``evolutionary_best(query, objective, mode)`` method, and, for
        :meth:`maximize_with_population`, an ``evolutionary_stream`` of the same signature.
    """

    def __init__(self, evolutionary: Any) -> None:
        self.evolutionary = evolutionary

    def maximize(
        self,
        acquisition_fn: AcquisitionFunction,
        query: Any,
        *,
        mode: Literal["auto", "single", "batch"] = "batch",
    ) -> Any:
        """Run the EA to maximize ``acquisition_fn`` and return the best candidate.

        Parameters
        ----------
        acquisition_fn:
            The acquisition function to maximize.
        query:
            The generator query naming the search space and the requested type.
        mode:
            How the EA population is to be scored: ``"batch"`` a generation at a time,
            ``"single"`` one candidate at a time, ``"auto"`` whichever of the two the annotation
            of the objective built here selects.  The driver is told the resolved mode rather
            than the word it was asked for, so its own reading of ``"auto"`` cannot disagree
            with the objective it is handed.  See :func:`resolve_fitness_mode`.

        Returns
        -------
        Any
            The fittest candidate encountered over the whole run, not the best of the final
            generation, which is what the previous driver returned.

        Raises
        ------
        ValueError
            If ``mode`` is none of the three, or if it asks for ``"single"`` and
            ``acquisition_fn`` has known points but no lower bound, which is the pairing the
            single-sample objective refuses.
        """
        resolved = resolve_fitness_mode(mode)
        return self.evolutionary.evolutionary_best(
            query, self._objective(acquisition_fn, resolved), resolved
        )

    def maximize_with_population(
        self,
        acquisition_fn: AcquisitionFunction,
        query: Any,
        *,
        mode: Literal["auto", "single", "batch"] = "batch",
    ) -> tuple[Any, list[Any], list[GenerationRecord]]:
        """Maximize, and keep both the population that produced the answer and how it got there.

        The frontier read places the candidates in the plane of posterior mean and posterior
        deviation and asks whether the returned term sits on the upper right frontier of its final
        generation, which is where a score rising in both coordinates has to put it.  A
        maximization that answers with the term alone cannot be read that way, because the
        population exists during the run and is dropped at the end of it.  The driver already
        yields it: ``evolutionary_best`` is the same stream with everything but the last ``best``
        thrown away.

        Keeping it costs a list of at most ``population_size`` references, and it is what makes
        the frontier check runnable on a real run rather than only on a fixture.  It is a
        separate method rather than the way :meth:`maximize` works, because it asks more of the
        optimizer: ``evolutionary_best`` is the whole of what a maximization needs, and a caller
        who does not read the frontier should not have to supply a stream to get one.

        Parameters
        ----------
        acquisition_fn:
            The acquisition function to maximize.
        query:
            The generator query naming the search space and the requested type.
        mode:
            How the EA population is to be scored.  See :meth:`maximize`.

        Returns
        -------
        tuple[Any, list[Any], list[GenerationRecord]]
            The fittest candidate of the whole run, the final population, and one record per
            generation.  The third is what says whether the run *improved*: the frontier read
            places the answer among the population it ended with, which is a statement about one
            generation and cannot distinguish a search that climbed from one that never moved.
            It is read off the states the stream already yields, so it costs nothing but the
            records themselves.

        Raises
        ------
        ValueError
            If ``mode`` is none of the three, or on the pairing of ``"single"`` with an
            acquisition that has known points but no lower bound.  See :meth:`maximize`.
        RuntimeError
            If the stream yields no generation at all.
        """
        resolved = resolve_fitness_mode(mode)
        final = None
        generations: list[GenerationRecord] = []
        for state in self.evolutionary.evolutionary_stream(
            query, self._objective(acquisition_fn, resolved), resolved
        ):
            final = state
            generations.append(_generation_record(state))
        if final is None:
            msg = (
                f"{type(self.evolutionary).__name__} yielded no generation at all; a run that "
                f"initialises its population yields at least the zeroth"
            )
            raise RuntimeError(msg)
        return final.best, list(final.population), generations

    @staticmethod
    def _objective(
        acquisition_fn: AcquisitionFunction, mode: Literal["single", "batch"]
    ) -> Any:
        """Wrap the acquisition as the fitness function the driver calls.

        Args:
            acquisition_fn (AcquisitionFunction): The acquisition to maximize.
            mode (Literal["single", "batch"]): How the driver will call it, resolved already.

        Returns:
            Any: The fitness function, with the caching and the known-point floor of the two
                wrappers above.
        """
        if mode == "batch":
            return _make_acquisition_objective_batch(acquisition_fn)
        return _make_acquisition_objective_single(acquisition_fn)


class SampleMaximizer:
    """Maximize an acquisition over one sample: draw it, score all of it, keep the best.

    The baseline an evolutionary search is measured against, and the maximization of the loops
    that search no further than their initial population.  The sample is what cosy's sampled
    initialization draws, one stream of the sampler, ``sample_size`` terms, and it is refused, not
    filled with repeats, where fewer inhabitants lie within the sampler's bound.  It is scored as
    the evolutionary search scores, through the known-point floor, and the first of the best terms
    in the order the stream delivered them is the pick, which is also what cosy's driver answers
    for a search stopped after its initial population.

    A sampler shared with the loop, the ``sampler=`` it draws its design and replacements from,
    draws both from one stream; a twin (:func:`~bayesian_optimization.runs.twin_sampler`) keeps
    them apart.

    Attributes:
        sampler (Sampler): What each sample is drawn from.
        sample_size (int): How many terms one sample holds.
    """

    def __init__(self, sampler: Sampler, sample_size: int) -> None:
        """Build the maximizer.

        Raises:
            ValueError: If ``sample_size`` is below one: a sample of nothing has no best term.
        """
        if sample_size < 1:
            msg = f"a sample to maximize over holds at least one term, not {sample_size}"
            raise ValueError(msg)
        self.sampler = sampler
        self.sample_size = sample_size

    def maximize(
        self,
        acquisition_fn: AcquisitionFunction,
        query: Any,
        *,
        mode: Literal["auto", "single", "batch"] = "batch",
    ) -> Any:
        """The best term of a new sample.  See :meth:`maximize_with_population`."""
        pick, _sample, _generations = self.maximize_with_population(acquisition_fn, query, mode=mode)
        return pick

    def maximize_with_population(
        self,
        acquisition_fn: AcquisitionFunction,
        query: Any,
        *,
        mode: Literal["auto", "single", "batch"] = "batch",
    ) -> tuple[Any, list[Any], list[GenerationRecord]]:
        """Draw a sample, score it, and answer its best term, the sample, and its one record.

        Args:
            acquisition_fn (AcquisitionFunction): The acquisition to maximize.
            query (Any): The generator query naming the search space and the requested type.
            mode (Literal["auto", "single", "batch"]): How the sample is scored, as for the
                evolutionary search. (Default value = "batch")

        Returns:
            tuple[Any, list[Any], list[GenerationRecord]]: The best term, the sample in stream
                order, and the record of its one generation.

        Raises:
            InitializationError: If fewer than ``sample_size`` inhabitants lie within the bound of
                the sampler, or its stream ends before the sample is full.
            ValueError: As the evolutionary search's objective refuses a mode, before anything is
                drawn; and for a score that is not a number, as cosy's driver refuses one.
        """
        resolved = resolve_fitness_mode(mode)
        # Built first: it refuses a mode the acquisition cannot be scored in, before anything is
        # drawn.
        objective = AcquisitionOptimizer._objective(acquisition_fn, resolved)
        sample: list[Any] = SampledInitialization(self.sampler).initialize(query, self.sample_size)
        if resolved == "batch":
            scored = objective(sample)
            values = [float(scored[term]) for term in sample]
        else:
            values = [float(objective(term)) for term in sample]
        for term, value in zip(sample, values, strict=True):
            if math.isnan(value):
                msg = (
                    f"the acquisition scored {term} as nan, not a number: every comparison with it "
                    "is false, so a maximization over it picks whatever it compares first, and "
                    "cosy's driver refuses it as this does"
                )
                raise ValueError(msg)
        best = 0
        for index in range(1, len(sample)):
            if values[index] > values[best]:
                best = index
        record = GenerationRecord(
            generation=0,
            best=values[best],
            population_best=values[best],
            population_mean=sum(values) / len(values),
            population_worst=min(values),
            distinct_members=len(set(sample)),
            last_improvement=0,
            offspring=0,
        )
        return sample[best], list(sample), [record]

