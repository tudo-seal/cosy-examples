"""One driver for every strategy of the ask/tell loop, and the paired comparison of several.

:func:`run_search` runs a strategy, Bayesian optimization or random search, under a caller's
:class:`~bayesian_optimization.runs.schema.MetricSchema`.  The design is the loop's first phase: it
is drawn by the strategy, given as terms, or resumed from the records of an earlier run, and on
every one of those paths each evaluation is written, row and term record, before the loop takes its
value and before the next evaluation starts.  A resumed design is taken over term by term, each
record with the value it kept, checked against the run's objective, or with the caller's reading of
its metrics where one is named.
Everything that can be refused is refused before anything is drawn, opened or paid: a taken run
name, a pass configuration no pass could use, a resumed record without a value.

:func:`run_paired` runs several strategies from one design at one budget.  The first evaluates the
design; the others take it over from the first one's records, so the design is paid once.  That is
what the CIFAR driver's paired random-search arm did for one pair of strategies, as a helper any
comparison can use, for instance of samplers at one budget.

The evaluation is the caller's: ``evaluate(term)`` answers with the mapping of metrics, and the
schema's objective reads the loop's value off it.
"""

from __future__ import annotations

import contextlib
import functools
import math
import os
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..bo import BayesianOptimization
from ..loop import AskTellLoop, _require_hashable
from ..random_search import RandomSearch
from ..state import BOState
from .artifacts import RunArtifacts
from .budgets import StepBudgets, step_budget
from .records import EAGenerationLogger, EvaluationRecorder, SurrogateLogger
from .resume import ResumedDesign, check_resumed_design
from .schema import MetricSchema, Objective
from .search_program import DesignDraw, draw_design
from .term_pool import TermRecord, read_term_pool

#: The phase of a design term in a run's rows and records, as the CIFAR driver has always named it.
DESIGN_PHASE = "pre_sample"


@dataclass
class RunOutcome:
    """What a run hands back beside its files.

    Attributes:
        result (dict[str, Any]): The strategy's ``finalize()``.
        seconds (float): The run's wall clock, the design included.
        summary (dict[str, Any]): The provenance of what ran: ``run_kind``, ``design_source``
            (``drawn``, ``terms`` or ``resumed``), the evaluations made here and taken over, the
            best value in the metric's own sign, and the strategy's repeat counts; where the design
            was drawn up front from the strategy's sampler, ``design_drawn_up_front`` (its size,
            the terms drawn past it, the repeats skipped, the seconds, which ``seconds`` holds);
            for the other arms of a pair, ``design_taken_over_from``.
    """

    result: dict[str, Any]
    seconds: float
    summary: dict[str, Any]


def _run_kind(strategy: AskTellLoop, n_passes: int) -> str:
    # Read off what ran rather than off the class alone: a run of no passes evaluated its design
    # and nothing else, whichever strategy would have made the passes.
    if n_passes == 0:
        return "design_only"
    if isinstance(strategy, BayesianOptimization):
        return "bayesian_optimization"
    if isinstance(strategy, RandomSearch):
        return "random_search"
    return type(strategy).__name__


def _resumed_values(
    records: Sequence[TermRecord],
    resumed_value: Callable[[Mapping[str, Any]], float] | None,
    objective: Objective,
) -> list[float]:
    """The value the loop is handed for each resumed record, refused where none can be named.

    The caller's reading, where it names one, reads every record.  Without one, a record hands the
    loop the value it kept, which is the value the loop of the run that WROTE it was handed, under
    that run's objective or under the reading it took its design over with; so it has to be this
    run's objective's reading of the record's metrics too, or it was handed under another, and the
    record is refused rather than mixed in.  A record that kept no value is refused as well: its
    metrics may name the objective's key and still mean another quantity, a caller's "objective
    value" holding whatever that run maximized.  Every refusal asks for the reading.
    """
    values = []
    for index, record in enumerate(records):
        if resumed_value is not None:
            value = float(resumed_value(record.metrics))
        elif record.loop_value is None:
            msg = (
                f"resumed record {index} kept no loop value, so nothing says what its metrics "
                "meant to the loop that wrote it; name how they become the value this run's loop "
                "maximizes with resumed_value"
            )
            raise ValueError(msg)
        else:
            try:
                value = objective.loop_value(record.metrics)
            except (KeyError, TypeError, ValueError) as error:
                msg = (
                    f"resumed record {index} cannot be read by this run's objective ({error}); "
                    "name how its metrics become the value the loop maximizes with resumed_value"
                )
                raise ValueError(msg) from None
            # A value that is not finite is refused below for what it is, not as a disagreement:
            # nan agrees with nothing, not even the nan it was kept as.
            if math.isfinite(value) and float(record.loop_value) != value:
                msg = (
                    f"resumed record {index} kept the loop value {record.loop_value}, and this "
                    f"run's objective reads {value} off its metrics: the record was handed to a "
                    "loop under another objective or reading; name the reading with resumed_value"
                )
                raise ValueError(msg)
        if not math.isfinite(value):
            msg = f"resumed record {index} hands the loop {value}, and the loop takes finite values"
            raise ValueError(msg)
        values.append(value)
    return values


def _loop_value_or_record(
    objective: Objective,
    recorder: EvaluationRecorder,
    phase: str,
    index: int,
    term: Any,
    metrics: Mapping[str, Any],
    **row: Any,
) -> float:
    """The loop's value of an evaluation; where it cannot be read, the evaluation is written first.

    A paid evaluation reaches the disk whatever its metrics hold: one that does not report the
    objective, or reports it as something that is not a number, is written with an empty loop
    value before the run stops on it.
    """
    try:
        return objective.loop_value(metrics)
    except Exception:
        recorder.log(phase, index, term, metrics, loop_value=None, **row)
        raise


def _require_uninitialized(strategy: AskTellLoop, name: str = "the strategy") -> None:
    state = strategy.get_state_snapshot()["state"]
    if state != BOState.UNINITIALIZED.value:
        msg = (
            f"{name} is in state {state}, not UNINITIALIZED: a run starts on a strategy that has "
            "not run; reset() it or construct a new one"
        )
        raise RuntimeError(msg)


def _draw_up_front(strategy, size, *, then, design, budgets, echo):
    """Draw a design, or head the stream with a given one, and ``then`` terms past it, up front.

    From the loop's own sampler, as the paired path always drew: the sampler is left where drawing
    all of it left it, so the loop's replacement for a duplicate -- a stream opened where the
    sampler stands -- draws past everything a paired arm draws on a twin.  Watched as that draw was,
    since the first draw builds the counting tables, which takes minutes on a large program and
    writes nothing, and announced through ``echo``.

    Returns:
        tuple[list, DesignDraw]: The design's terms, the head of the stream, and the draw's record.
    """
    if design is None:
        label = f"drawing {size} design terms"
    else:
        label = f"heading the stream with the {size} design terms given"
    if then:
        label += f" and the other arms' {then}"
    label += " up front"
    echo(f"{label[0].upper()}{label[1:]}, from the loop's sampler")
    with step_budget(label, budgets.acquisition_warn):
        started = time.time()
        head, repeats = draw_design(strategy.sampler, strategy.query, size, then=then, design=design)
        seconds = time.time() - started
    return head, DesignDraw(size, then, repeats, seconds, drawn=design is None)


def _design_of(strategy, *, n_design, n_passes, design, resume, resumed_value, objective):
    """The design a run starts from, refused where the run cannot start from it.

    Refused before anything is drawn, opened or paid, by :func:`run_search` for its own run and by
    :func:`run_paired` for its first run before it draws up front.

    Returns:
        tuple: The resumed records, their values (None unless resumed), the design's terms (None
            where the strategy draws them), the design's source and its size.

    Raises:
        ValueError, TypeError, RuntimeError: As :func:`run_search` says.
    """
    if design is not None and resume is not None:
        raise ValueError("a design is given either as terms or as resumed records, not both")
    if n_passes < 0:
        raise ValueError(f"a budget of passes is a count, not {n_passes}")
    values: list[float] | None = None
    records: list[TermRecord] = []
    terms: list[Any] | None = None
    if resume is not None:
        records = list(resume)
        values = _resumed_values(records, resumed_value, objective)
        terms = [record.term for record in records]
        source = "resumed"
    elif design is not None:
        terms = list(design)
        source = "terms"
    else:
        if n_design is None:
            raise ValueError("n_design is required when the strategy draws the design")
        source = "drawn"
    if terms is not None and n_design is not None and n_design != len(terms):
        msg = f"n_design is {n_design}, and the design handed over has {len(terms)} terms"
        raise ValueError(msg)
    size = len(terms) if terms is not None else n_design
    assert size is not None
    if size < 0:
        raise ValueError(f"a design holds a non-negative number of terms, not {size}")
    if size + n_passes == 0:
        raise ValueError("a run of no design and no passes evaluates nothing and has no result")
    if isinstance(strategy, BayesianOptimization) and size == 0 and n_passes > 0:
        raise ValueError(
            "Bayesian optimization conditions its first pass on the design, and a design of no "
            "terms leaves nothing to condition"
        )
    if terms is not None:
        _require_hashable(terms, "the design")
        if len(set(terms)) != len(terms):
            raise ValueError("the design repeats a term: a repeated design term is an evaluation "
                             "spent on a value the dataset already holds")
    _require_uninitialized(strategy)
    return records, values, terms, source, size


def _draws_from_its_stream(strategy: AskTellLoop) -> bool:
    """Whether a Bayesian loop's design is the head of its own sampler's stream.

    So it is with a sampler of its own and the default initializer, which collects the design from
    one stream of that sampler.  An initializer of the caller's draws the design otherwise, and a
    design drawn up front from the sampler would be another one; a loop without a sampler of its own
    draws from a placeholder the run layer does not hold.

    Args:
        strategy (AskTellLoop): The strategy.

    Returns:
        bool: Whether its design can be drawn up front, and a resumed one held against that head.
    """
    return (
        isinstance(strategy, BayesianOptimization)
        and getattr(strategy, "sampler", None) is not None
        and getattr(strategy, "initializer", None) is None
    )


def _require_batch_size(strategy: AskTellLoop, batch_size: Any, name: str = "the strategy") -> None:
    """Refuse a round the strategy cannot hold outstanding at once."""
    capacity = getattr(strategy, "max_outstanding", 1)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= capacity:
        msg = (
            f"batch_size is how many suggestions a round holds, a count from 1 to what {name} "
            f"holds outstanding at once (max_outstanding={capacity}), not {batch_size!r}"
        )
        raise ValueError(msg)


def _answers(round_terms, evaluate, evaluate_many):
    """The round's terms with their metrics, in the order the evaluations complete.

    ``evaluate_many`` answers each term of the round once, in any order; without it the round is
    evaluated one term after the other with ``evaluate``.

    Raises:
        RuntimeError: If ``evaluate_many`` answers a term the round did not ask, one twice, or
            leaves one unanswered.
    """
    if evaluate_many is None:
        for term in round_terms:
            yield term, evaluate(term)
        return
    waiting = list(round_terms)
    for term, metrics in evaluate_many(list(round_terms)):
        position = next((index for index, asked in enumerate(waiting) if asked == term), None)
        if position is None:
            msg = (
                f"evaluate_many answered {term!r}, which the round did not ask or had answered "
                "already: each term of a round is answered once"
            )
            raise RuntimeError(msg)
        yield waiting.pop(position), metrics
    if waiting:
        msg = f"evaluate_many left {len(waiting)} of the round's terms unanswered: {waiting!r}"
        raise RuntimeError(msg)


def _pairs(records):
    return [(record.term, record.metrics) for record in records]


def run_search(
    strategy: AskTellLoop,
    evaluate: Callable[[Any], Mapping[str, Any]],
    *,
    schema: MetricSchema,
    csv_path: str,
    pretty_algebra: Callable[[], Any],
    n_design: int | None = None,
    n_passes: int = 0,
    design: Sequence[Any] | None = None,
    resume: Sequence[TermRecord] | None = None,
    resumed_value: Callable[[Mapping[str, Any]], float] | None = None,
    provenance: Mapping[str, Any] | None = None,
    budgets: StepBudgets | None = None,
    refuse_taken: bool = True,
    verbose: bool = False,
    echo: Callable[[str], None] | None = None,
    drawn_up_front: DesignDraw | None = None,
    hold_resumed_design: bool = True,
    batch_size: int = 1,
    evaluate_many: Callable[[list[Any]], Iterable[tuple[Any, Mapping[str, Any]]]] | None = None,
) -> RunOutcome:
    """Run a strategy, writing every evaluation as it is measured.

    Args:
        strategy (AskTellLoop): The configured strategy, before ``initialize()``.
        evaluate (Callable): Maps a term to the mapping of its metrics; the objective is one of
            them.
        schema (MetricSchema): The run's columns and the objective.
        csv_path (str): The run's CSV; every other file is named after it.
        pretty_algebra (Callable): The algebra that renders a term for the CSV.
        n_design (int | None): The size of a drawn design; with ``design`` or ``resume`` it may be
            left out, and given it must match. (Default value = None)
        n_passes (int): The budget of passes after the design. (Default value = 0)
        design (Sequence | None): A design given as terms, evaluated as given. (Default value = None)
        resume (Sequence[TermRecord] | None): The design records of an earlier run of this
            configuration, taken over instead of evaluated: their terms are the design, and their
            values the loop values they kept, checked against the schema's objective, or
            ``resumed_value``'s reading of their metrics.  Which records belong to which
            configuration is the caller's check (see
            :func:`~bayesian_optimization.runs.resume.load_design_records`).  Under Bayesian
            optimization that draws its design from a sampler of its own -- the default
            initializer, which collects it from one stream of that sampler -- a design its pool
            says was drawn (``ResumedDesign.origin``, ``"drawn"`` for a plain list) is held against
            the head of the loop's stream, and the sampler moved past it, as the run it resumes had
            drawn it; a design given as terms is not, and resumes under any seed.  A loop with an
            initializer of the caller's, or with no sampler of its own, takes the design over as it
            stands. (Default value = None)
        resumed_value (Callable | None): How a resumed record's metrics become the value this
            run's loop is handed.  Given, it reads every resumed record.  Omitted, each record
            hands the loop the value it kept, which must be the schema's objective's reading of
            its metrics: a kept value is the value the loop of the run that wrote the record was
            handed, under that run's objective or reading, and one that disagrees, like a record
            that kept none, is refused. (Default value = None)
        provenance (Mapping | None): Written into the term pool's header. (Default value = None)
        budgets (StepBudgets | None): The watchdog's budgets; the CIFAR example's when omitted.
            (Default value = None)
        refuse_taken (bool): Refuse a run whose files exist, before anything is opened.
            (Default value = True)
        verbose (bool): Passed to ``suggest``. (Default value = False)
        echo (Callable[[str], None] | None): Where the per-evaluation lines go; ``None`` prints
            each line and flushes it, so that a killed run's log holds the lines before the kill.
            (Default value = None)
        drawn_up_front (DesignDraw | None): The draw a caller made of this strategy's design up
            front -- :func:`run_paired` makes it for its first strategy -- recorded as this run's:
            its seconds in the run's, its repeats as the design's. (Default value = None)
        hold_resumed_design (bool): Hold a drawn resumed design against the stream, as ``resume``
            says; :func:`run_paired` turns it off where it held the design itself or where the
            design is another strategy's. (Default value = True)
        batch_size (int): How many suggestions a round holds: the design goes out that many terms
            at a time, then the passes, and a round is evaluated before the next is suggested; at
            most the strategy's ``max_outstanding``.  A pass's row carries its label, the passes
            suggested before it. (Default value = 1)
        evaluate_many (Callable | None): Evaluates a round side by side: handed the round's terms,
            it answers ``(term, metrics)`` for each, once, in the order the evaluations complete,
            and each answer is written and observed as it arrives.  ``None`` evaluates a round one
            term after the other with ``evaluate``. (Default value = None)

    Returns:
        RunOutcome: The result, the wall clock and the summary.

    Raises:
        ValueError: For a design given both ways, a size that does not match it or is negative, a
            run that would evaluate nothing, a design that repeats a term, a resumed record
            without a finite value, Bayesian optimization without a design to condition its first
            pass on; and as ``check_configuration()`` refuses a pass configuration.
        TypeError: For a design term that cannot be hashed.
        RuntimeError: For a strategy that has already run.
        FileExistsError: If ``refuse_taken`` and a file of the run exists.
        All of these before anything is drawn, opened or evaluated, so a corrected retry under the
        same name runs.
    """
    budgets = budgets if budgets is not None else StepBudgets()
    echo = echo if echo is not None else functools.partial(print, flush=True)
    # --- Everything that can be refused is refused before anything is drawn, opened or paid.
    records, values, terms, source, size = _design_of(
        strategy, n_design=n_design, n_passes=n_passes, design=design, resume=resume,
        resumed_value=resumed_value, objective=schema.objective,
    )
    _require_batch_size(strategy, batch_size)
    artifacts = RunArtifacts(csv_path)
    if refuse_taken:
        artifacts.refuse_taken()
    if n_passes > 0:
        strategy.check_configuration()

    bayesian = isinstance(strategy, BayesianOptimization)
    # Where the design came from, recorded in the pool for a run that resumes it.
    if resume is not None:
        origin = getattr(resume, "origin", "drawn")
    elif design is not None and not (drawn_up_front is not None and drawn_up_front.drawn):
        origin = "given"
    else:
        origin = "drawn"
    if (
        resume is not None
        and hold_resumed_design
        and drawn_up_front is None
        and origin == "drawn"
        and _draws_from_its_stream(strategy)
    ):
        # The loop's replacement for a duplicate draws from where its sampler stands, so a loop
        # that takes a drawn design over has to stand where the run that drew it stood: past it.
        # A random search continues its design's stream past the held terms on its own.
        head, drawn_up_front = _draw_up_front(
            strategy, len(records), then=0, design=None, budgets=budgets, echo=echo
        )
        check_resumed_design(_pairs(records), head)
    phase = strategy.PASS_PHASE
    objective = schema.objective
    evaluated_here = 0
    taken_over = 0
    metrics: Mapping[str, Any]
    started = time.time()
    with contextlib.ExitStack() as stack:
        recorder = stack.enter_context(EvaluationRecorder(
            csv_path, pretty_algebra, schema, provenance=provenance,
            mode="x" if refuse_taken else "w", design_origin=origin,
        ))
        ea_logger = surrogate_logger = None
        if bayesian and n_passes > 0:
            ea_logger = stack.enter_context(EAGenerationLogger(artifacts.path("ea")))
            surrogate_logger = stack.enter_context(SurrogateLogger(artifacts.path("surrogate")))

        # --- The design, the loop's first phase: every term written before its value is taken.
        with step_budget(f"the design ({size} evaluations)", max(size, 1) * budgets.per_evaluation):
            if terms is None:
                strategy.initialize(initial_size=size)
            else:
                strategy.initialize(design=terms)
            design = list(strategy.design)
            # A design taken over is answered at once; one evaluated here goes out in rounds.
            width = 1 if values is not None else batch_size
            for start in range(0, len(design), width):
                positions: list[tuple[int, Any]] = []
                for index in range(start, min(start + width, len(design))):
                    suggestion = strategy.suggest(verbose=verbose)
                    term = suggestion.candidate
                    if term != design[index]:
                        msg = (
                            f"the loop handed out design term {index} out of order: {term} where "
                            f"the design holds {design[index]}"
                        )
                        raise RuntimeError(msg)
                    positions.append((index, term))
                if values is not None:
                    index, term = positions[0]
                    answers: Iterable[tuple[Any, Mapping[str, Any]]] = [
                        (term, records[index].metrics)
                    ]
                else:
                    answers = _answers([term for _index, term in positions], evaluate,
                                       evaluate_many)
                for term, metrics in answers:
                    index = next(index for index, asked in positions if asked is term)
                    if values is not None:
                        value = values[index]
                        taken_over += 1
                        note = " (resumed)"
                    else:
                        evaluated_here += 1
                        value = _loop_value_or_record(objective, recorder, DESIGN_PHASE, index,
                                                      term, metrics)
                        note = ""
                    # Written before the loop takes the value, so that a value the loop refuses,
                    # a non-finite one, is on disk with the metrics that explain it.
                    recorder.log(DESIGN_PHASE, index, term, metrics, loop_value=value,
                                 taken_over=values is not None)
                    echo(f"  {DESIGN_PHASE}[{index}]: objective={objective.as_reported(value):.5f} "
                         f"{schema.live_line(metrics)}{note}".rstrip())
                    strategy.observe(term, value)

        # --- The passes, in rounds of up to batch_size suggestions.
        for first in range(0, n_passes, batch_size):
            passes: list[tuple[int, Any, float]] = []
            for step in range(first, min(first + batch_size, n_passes)):
                proposal_started = time.time()
                with step_budget(
                    f"pass {step}: the proposal",
                    budgets.acquisition_warn,
                    hard_limit_seconds=budgets.acquisition_hard_limit if bayesian else None,
                ):
                    if bayesian:
                        # The frontier read and the generation log need the population as scored.
                        assert isinstance(strategy, BayesianOptimization)
                        suggestion = strategy.suggest(verbose=verbose, record_population=True)
                    else:
                        suggestion = strategy.suggest(verbose=verbose)
                acquisition_seconds = time.time() - proposal_started
                if ea_logger is not None and surrogate_logger is not None:
                    # Written before the evaluation, so an interrupted pass keeps its inner search;
                    # in a round, the surrogate of a later pass holds the earlier ones as assumed.
                    ea_logger.log(step, getattr(strategy, "last_acquisition_run", None))
                    surrogate_logger.log(step, strategy)
                passes.append((step, suggestion, acquisition_seconds))
            label = (
                f"pass {first}: the evaluation" if len(passes) == 1
                else f"passes {first} to {passes[-1][0]}: the evaluations"
            )
            # A round evaluated term by term takes as long as its terms together; one evaluated
            # side by side, as long as its slowest.
            budget = budgets.per_evaluation * (len(passes) if evaluate_many is None else 1)
            with step_budget(label, budget):
                for term, metrics in _answers(
                    [suggestion.candidate for _step, suggestion, _seconds in passes], evaluate,
                    evaluate_many,
                ):
                    step, suggestion, acquisition_seconds = next(
                        entry for entry in passes if entry[1].candidate is term
                    )
                    evaluated_here += 1
                    value = _loop_value_or_record(objective, recorder, phase, step, term, metrics,
                                                  suggestion=suggestion,
                                                  acquisition_seconds=acquisition_seconds)
                    recorder.log(phase, step, term, metrics, suggestion=suggestion,
                                 acquisition_seconds=acquisition_seconds, loop_value=value)
                    echo(f"  {phase}[{step}]: objective={objective.as_reported(value):.5f} "
                         f"{schema.live_line(metrics)}".rstrip())
                    strategy.observe(term, value)

    result = strategy.finalize()
    # A draw up front is the run's: its time in the run's.  Where it drew the run's design it is
    # the design's source and its repeats are the run's, the other arms' terms among them, as the
    # paired path always counted them; a design given or resumed was not the run's to draw, and the
    # draw that headed the stream with it or held it counts nothing of the run's.
    seconds = time.time() - started + (drawn_up_front.seconds if drawn_up_front else 0.0)
    if drawn_up_front is not None and drawn_up_front.drawn and resume is None:
        source = "drawn"
    summary: dict[str, Any] = {
        "run_kind": _run_kind(strategy, n_passes),
        "strategy": type(strategy).__name__,
        "design_source": source,
        "n_design": len(strategy.design),
        "n_passes": n_passes,
        "evaluated_here": evaluated_here,
        "taken_over": taken_over,
        "seconds": seconds,
        "best_objective_value": objective.as_reported(float(result["best_y"])),
        "best_loop_value": float(result["best_y"]),
        "initial_repeats_rejected": (
            drawn_up_front.repeats_skipped
            if drawn_up_front is not None and drawn_up_front.drawn and resume is None
            else strategy.initial_repeats_rejected
        ),
        "completed": True,
    }
    if drawn_up_front is not None:
        summary["design_drawn_up_front"] = drawn_up_front.record()
    if isinstance(strategy, RandomSearch):
        summary["terms_skipped"] = strategy.terms_skipped
    return RunOutcome(result, seconds, summary)


def run_paired(
    strategies: Mapping[str, AskTellLoop],
    evaluate: Callable[[Any], Mapping[str, Any]],
    *,
    schema: MetricSchema,
    csv_paths: Mapping[str, str],
    pretty_algebra: Callable[[], Any],
    n_passes: int,
    n_design: int | None = None,
    design: Sequence[Any] | None = None,
    resume: Sequence[TermRecord] | None = None,
    resumed_value: Callable[[Mapping[str, Any]], float] | None = None,
    provenance: Mapping[str, Any] | None = None,
    budgets: StepBudgets | None = None,
    refuse_taken: bool = True,
    verbose: bool = False,
    echo: Callable[[str], None] | None = None,
    batch_size: int = 1,
    evaluate_many: Callable[[list[Any]], Iterable[tuple[Any, Mapping[str, Any]]]] | None = None,
) -> dict[str, RunOutcome]:
    """Run several strategies from one design at one budget, the design evaluated once.

    The first strategy runs the design as :func:`run_search` would, drawn, given or resumed; every
    other one takes that design over from the first run's records, a resumed one under the same
    ``resumed_value``, and then makes its own passes.  Where the first strategy is Bayesian
    optimization that draws its design from a sampler of its own -- the default initializer -- its
    design and the other arms' terms, ``n_passes`` of them whatever the other arms are, are drawn up
    front from that sampler, as the paired path always drew them: an arm on a twin of the sampler
    (:func:`~bayesian_optimization.runs.twin_sampler`) draws the stream past the design, and the
    loop's replacement for a duplicate, a stream opened where its sampler stands, then draws past
    everything the arm draws.  A resumed design is held against that head where its pool says it
    was drawn; without passes nothing is drawn up front, and the first strategy holds a resumed
    design as it would run alone.  A first strategy with an initializer of the caller's draws its
    design as it would alone, and its replacement for a duplicate is not held apart from an arm's
    terms.
    Whether one strategy beats another is a question about the passes only if they start from the
    same place, and sharing the evaluated design is also what makes the comparison cost one extra
    budget of passes per strategy rather than one extra design.

    Args:
        strategies (Mapping[str, AskTellLoop]): The strategies by name; the first draws the design.
        csv_paths (Mapping[str, str]): One run CSV per name.
        n_passes (int): Every strategy's budget of passes.
        refuse_taken (bool): Refuse every run whose files exist, before the first run starts; a
            caller that refused them itself and wrote each run's configuration first passes
            False. (Default value = True)
        batch_size (int), evaluate_many (Callable | None): Every run's rounds, as for
            :func:`run_search`; a round larger than any strategy holds is refused before the first
            run starts. (Defaults: 1, None)
        (The other arguments as for :func:`run_search`.)

    Returns:
        dict[str, RunOutcome]: The outcome of each strategy's run, by name.

    Raises:
        ValueError: If the names of the strategies and the paths differ, if two runs would share
            a file, or if one strategy object appears under two names.
        RuntimeError: If a strategy has already run.
        FileExistsError: With ``refuse_taken``, if a file of any of the runs exists.
        All of these, and a pass configuration a later strategy could not use, before the first
        run starts, so that a later strategy's mistake costs no design.
    """
    names = list(strategies)
    if not names or set(names) != set(csv_paths):
        raise ValueError("one CSV path per strategy, under the same names")
    if len({id(strategy) for strategy in strategies.values()}) != len(names):
        raise ValueError("one strategy object under two names would run twice on one state")
    claimed: dict[str, str] = {}
    for name in names:
        for path in RunArtifacts(csv_paths[name]).paths().values():
            # Compared as the files they name, not as strings: ``sub/../a.csv`` is ``a.csv``, and
            # without the refusal of taken files nothing else would stop the second run
            # overwriting the first one's.
            file = os.path.realpath(path)
            if file in claimed:
                msg = f"the runs {claimed[file]!r} and {name!r} would both write {path}"
                raise ValueError(msg)
            claimed[file] = name
    for name in names:
        _require_uninitialized(strategies[name], f"the strategy {name!r}")
        _require_batch_size(strategies[name], batch_size, f"the strategy {name!r}")
        if refuse_taken:
            RunArtifacts(csv_paths[name]).refuse_taken()
    if n_passes > 0:
        for name in names:
            strategies[name].check_configuration()
    common: dict[str, Any] = {
        "schema": schema, "pretty_algebra": pretty_algebra, "n_passes": n_passes,
        "provenance": provenance, "budgets": budgets, "verbose": verbose, "echo": echo,
        "refuse_taken": refuse_taken, "batch_size": batch_size, "evaluate_many": evaluate_many,
    }
    first = names[0]
    leader = strategies[first]
    # The first run refuses its design before the draw up front, which on a large program costs
    # the counting tables, not after it.
    _design_of(
        leader, n_design=n_design, n_passes=n_passes, design=design, resume=resume,
        resumed_value=resumed_value, objective=schema.objective,
    )
    leader_design, drawn = design, None
    if n_passes > 0 and _draws_from_its_stream(leader):
        # The design and the other arms' terms from the leader's own sampler, up front, so that
        # its replacement for a duplicate draws past everything an arm on a twin draws.
        up_front = {
            "then": n_passes,
            "budgets": budgets if budgets is not None else StepBudgets(),
            "echo": echo if echo is not None else functools.partial(print, flush=True),
        }
        if resume is not None:
            given = [r.term for r in resume] if getattr(resume, "origin", "drawn") == "given" else None
            head, drawn = _draw_up_front(leader, len(resume), design=given, **up_front)
            if given is None:
                check_resumed_design(_pairs(resume), head)
        elif design is not None:
            _head, drawn = _draw_up_front(leader, len(design), design=list(design), **up_front)
        elif n_design is not None:
            leader_design, drawn = _draw_up_front(leader, n_design, design=None, **up_front)
    outcomes = {first: run_search(
        leader, evaluate, csv_path=csv_paths[first], n_design=n_design, design=leader_design,
        resume=resume, resumed_value=resumed_value, drawn_up_front=drawn,
        # held here where the pair drew up front; otherwise as the first strategy would alone
        hold_resumed_design=drawn is None, **common,
    )}
    header, records = read_term_pool(RunArtifacts(csv_paths[first]).path("terms"))
    shared = ResumedDesign(
        [record for record in records if record.phase == DESIGN_PHASE],
        header.get("design_origin", "drawn"),
    )
    # A design the first run took over under a reading carries that reading's values, which the
    # objective may not read off the metrics; the other runs take it over under the same reading.
    reading = resumed_value if resume is not None else None
    for name in names[1:]:
        # The design is the leader's, taken over as data: not held against this arm's stream.
        outcomes[name] = run_search(
            strategies[name], evaluate, csv_path=csv_paths[name], resume=shared,
            resumed_value=reading, hold_resumed_design=False, **common,
        )
        outcomes[name].summary["design_taken_over_from"] = os.path.basename(csv_paths[first])
    return outcomes
