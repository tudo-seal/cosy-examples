"""What the two cnn_damg_nas experiments share: the CIFAR layout of a run, and its driver.

The USPS and the CIFAR-10 script differ in their dataset, their search-space constants and their
command line.  Everything between them, the acquisition optimizer, the ask/tell loop, the logging
and the provenance record, was the same code in both files, character for character, and was then
kept here.  The parts that name no CIFAR metric and no CNN alphabet have since moved into the run
layer, :mod:`bayesian_optimization.runs`, and are re-exported from here: the search program and
its samplers, the watchdog, the acquisition-optimizer builder, the term pool, the recorder, the
run's files and its neutral provenance, and a driver for any strategy.  What stays is this
example's own: its column layout ``CIFAR_SCHEMA``, its evaluation, and ``run_ask_tell_search``.

The reason is not tidiness.  While it stood twice, every correction had to be made twice, and
statements had drifted from what the code did: both metadata blocks named a survivor selection the
run did not use, both called the mutation rate mandatory for a reason that had been repaired two
packages earlier, and both quoted a kernel prior variance that no longer existed.  A description
written by hand next to the thing it describes will do that.  So the description here is read off
the constructed objects, see :func:`describe_search`, rather than written beside them.

Seven artifacts per run, and they answer different questions:

* ``<run>.csv``, one row per objective-function evaluation: the phase, the index within that
  phase, the pretty-printed structure, the measured metrics and the two wall clocks, training and
  acquisition.  Flushed after each row, so an interrupted run keeps its results.  Whether the
  search finds better networks is read here.
* ``<run>_terms.pickle``, the same evaluations with the term rather than its rendering, one record
  each, written by the same call so that it cannot be omitted.  Whether this run can be analyzed
  afterwards is decided here and only here: a rendered structure cannot be fed back into a kernel,
  so without this file every offline question, which kernel orders the run or what a different
  round count would have predicted, costs a full retraining.  See :mod:`cnn_damg_term_pool`.
* ``<run>_config.json``, the run's provenance: dataset, full search-space configuration, loop
  parameters, device, timings, library versions.  Without it the CSVs of two runs are
  indistinguishable except by filename.
* ``<run>_trace.csv``, one row per pass of the outer loop: the acquisition value at the pick, the
  posterior mean and deviation there, the incumbent, the best so far, the fallback flag.  Whether
  the loop converges is read here.
* ``<run>_ea.csv``, one row per generation per pass of the inner evolutionary search.  Whether the
  acquisition optimizer optimizes is read here and nowhere else, because the loop keeps only the
  final population of the last pass.
* ``<run>_surrogate.csv``, one row per pass: the leave-one-out calibration, the held-out fit, the
  log marginal likelihood and the fitted kernel hyperparameters.  Whether the model gets better is
  read here.  The acceptance checks answer only whether it ended up usable.
* ``<run>_diagnostics.json``, the acceptance checks over the finished run.

A random-search baseline is the same script with ``--n-pre-samples 30 --n-iterations 0``.  The
initial dataset already comes from the size-uniform sampler, so a run with no passes evaluates
thirty drawn terms and nothing else, through the same code and into the same artifacts.  The
provenance names the run kind, read off ``n_iterations`` rather than off a flag beside it, so the
two cannot disagree.

That baseline is an independent sample at the same budget, not a paired one, and the difference
matters for how the comparison is read.  Two processes at the same seed do not draw the same terms,
because the synthesized program comes out in a different rule order per process and the stream
differs with it.  The distribution is unaffected, which is what makes the comparison sound.  What
is lost is the variance reduction a common initial design would have bought.  Pairing the two means
running both from one process, which is what the paired baseline of :func:`run_ask_tell_search`
does.

Training is not deterministic, since the weights are drawn and the mini-batches are shuffled, and
no seed is fixed unless the caller passes one.  The values recorded here are the reference to
compare against when a structure is retrained later, and small deviations on re-evaluation are
expected.
"""

import _thread
import contextlib
import csv
import json
import os
import random
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from cosy.core import Synthesizer
from cosy.evolutionary_algorithms import (
    EvolutionarySearch,
    ExpScalarization,
    Generations,
    GenerousConservativeReplacement,
    RankBasedSelection,
    ResolutionMutation,
    SampledInitialization,
    ScalarFitnessComparator,
    SubtreeSwap,
)
from cosy.search import DepthBoundedRandomSampler, SizeUniformSampler, generator_query
from cosy.search.determinize import determinize

from bayesian_optimization.diagnostics import (
    read_calibration,
    read_fit,
    read_frontier,
    read_gram,
    read_trace,
    trace_columns,
    trace_rows,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_network_algebras import (
    P50,
    P50_CORRECTED_CIFAR,
    P50_CORRECTED_DIGITS,
    TrainingProtocol,
    apply_he_initialisation,
    pytorch_components_algebra,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_network_algebras import (
    learner as raw_learner,
)
from bayesian_optimization.examples.cnn_damg_nas.recognizable_cnn_damg_repo import (
    check_alphabet,
)
from bayesian_optimization.initial_sampling import distinct_prefix

# The run machinery that no search space owns lives in :mod:`bayesian_optimization.runs`.  Every
# name of it that this module used to define is imported here, so it stays importable from here.
from bayesian_optimization.runs.acquisition import (
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_MUTATION_RATE,
    DEFAULT_SELECTION_PRESSURE,
    build_acquisition_optimizer,
)
from bayesian_optimization.runs.artifacts import _sibling_path, metadata_path_for
from bayesian_optimization.runs.budgets import step_budget
from bayesian_optimization.runs.metadata import write_run_metadata as _write_run_metadata
from bayesian_optimization.runs.records import (
    EA_CSV_COLUMNS,
    SURROGATE_CSV_COLUMNS,
    EAGenerationLogger,
    EvaluationRecorder,
    SurrogateLogger,
    kernel_hyperparameters,
)
from bayesian_optimization.runs.resume import _check_resumed_design, load_initial_design
from bayesian_optimization.runs.run_diagnostics import write_run_diagnostics
from bayesian_optimization.runs.schema import Column, MetricSchema, Objective
from bayesian_optimization.runs.search_program import (
    DEFAULT_DEPTH_BOUND,
    DEFAULT_SIZE_BOUND,
    DETERMINIZATION_STATE_LIMIT,
    DETERMINIZATION_WARN_SECONDS,
    SPACE_CONSTRUCTION_WARN_SECONDS,
    DeterminizedSizeUniformSampler,
    SearchProgram,
    describe_sampler,
    describe_search,
)
from bayesian_optimization.runs.search_program import build_search as _build_search
from bayesian_optimization.runs.term_pool import TermPoolWriter, read_term_pool

# --- Time budgets ---------------------------------------------------------------------------
# Not limits: thresholds past which a step reports that it is still running.  A run of this kind is
# long by design, so "still going" and "stuck" look identical from outside, and the difference
# otherwise only shows up hours later in a log that stops mid-sentence.
#
# The numbers are orders of magnitude, not expectations.  They are thresholds for reporting, so a
# threshold that has gone out of date makes a run noisier or quieter, never wrong.
PER_EVALUATION_WARN_SECONDS = 900
ACQUISITION_WARN_SECONDS = 600
# The one hard limit, and it is deliberate: an acquisition optimization that has run for an hour is
# not slow, it is stuck.  The draw cost of the depth-bounded sampler has a heavy tail on spaces
# whose parameters are literals, and with a positive mutation rate that tail sits in the loop rather
# than switched off.  Nothing is lost by stopping, because the evaluation log flushes every row as
# it is written.  Set to None to only report.
#: When to give up on one acquisition maximization.  A limit and not a warning, because a hanging
#: inner search is a known failure mode here rather than a slow day.
#:
#: It bounds a legitimate duration, so it has to be read against the space being searched.  One
#: acquisition step converts every offspring of every generation into a graph, so its cost is the
#: population times the generations times the conversion cost of one term, and that conversion cost
#: grows with the term.  A large architecture at a large population can exceed an hour with nothing
#: wrong, which is why a run may raise the limit rather than edit it here.
ACQUISITION_HARD_LIMIT_SECONDS = 3600


def build_search(
    repository,
    target,
    *,
    sampling="size-uniform",
    size_bound=DEFAULT_SIZE_BOUND,
    depth_bound=DEFAULT_DEPTH_BOUND,
    seed=0,
    state_limit=DETERMINIZATION_STATE_LIMIT,
):
    """Build the program a run searches together with the sampler that fits it, for this example.

    :func:`bayesian_optimization.runs.search_program.build_search`, with the alphabet check of
    :mod:`recognizable_cnn_damg_repo` as its ``check_alphabet``.  The arguments are that
    function's, and so is the account of why the program and the sampler are one decision.

    Returns:
        SearchProgram: The program, the symbol to query, the sampler, and the provenance.

    Raises:
        ValueError: If ``sampling`` is neither of the two names, or if the check refuses the
            program.
    """
    return _build_search(
        repository,
        target,
        check_alphabet=check_alphabet,
        sampling=sampling,
        size_bound=size_bound,
        depth_bound=depth_bound,
        seed=seed,
        state_limit=state_limit,
    )


def _as_list(values: Any) -> list | None:
    """Render an optional sequence for a record.

    Args:
        values (Any): The sequence, or None.

    Returns:
        list | None: The values as a list, or None.
    """
    return None if values is None else list(values)


def describe_repository(repo: Any) -> dict:
    """Read the search space of a run off the repository that will build it.

    The block this returns used to be written from the module constants of the experiment while
    the repository was built from a different set of them, so a run of the VGG cell recorded the
    tutorial cell's dimensions beside a correct count of its own non-terminals. One field of that
    block was read off the object, and it is the one that disagreed with the rest.

    Args:
        repo (Any): The repository the run searches over.

    Returns:
        dict: The parameters that decide which networks the space holds, as the repository has
            them. ``num_feature_dimensions`` is a count and not a parameter: it is what the widths
            close to under parallel sums, and it is what the space is large or small because of.
    """
    return {
        "linear_feature_dimensions": list(repo.linear_feature_dimensions),
        "channel_dimensions": list(repo.channel_dimensions),
        "height_width_dimensions": [list(pair) for pair in repo.height_width_dimensions],
        "kernel_dimensions": [list(pair) for pair in repo.kernel_dimensions],
        "pooling_kernel_dimensions": [list(pair) for pair in repo.pooling_kernel_dimensions],
        "stride_values": list(repo.stride_values),
        "padding_values": list(repo.padding_values),
        "max_parallel_width": repo.max_parallel_width,
        "max_lin_layer_dim": repo.max_lin_layer_dim,
        # Normalized rather than as given: the repository appends 0 and 1 if they are missing,
        # because they are the neutral values of the sum and the product component, and the space
        # was built from the normalized list.
        "constant_values": list(repo.constant_values),
        "learning_rate_values": list(repo.learning_rate_values),
        "n_epoch_values": list(repo.n_epoch_values),
        # None where the repository was given none, since the optimizer combinator then substitutes
        # its own single value and a list here would claim the caller chose it.
        "weight_decay_values": _as_list(repo.weight_decay_values),
        "momentum_values": _as_list(repo.momentum_values),
        "num_feature_dimensions": len(repo.feature_dimensions),
    }


def _joined(values):
    """Render a per-repetition list into one CSV cell, or an empty cell if there is none.

    Args:
        values (Sequence | None): The per-repetition measurements.

    Returns:
        str: The values separated by single spaces, or "" when nothing was measured.
    """
    if not values:
        return ""
    return " ".join(str(value) for value in values)


def _empty_if_none(value):
    """Render a value that may be missing into one CSV cell.

    Args:
        value (Any): The value, or None where nothing was measured.

    Returns:
        Any: The value, or "" for None.
    """
    return "" if value is None else value


CIFAR_SCHEMA = MetricSchema(
    objective=Objective("accuracy"),
    columns=(
        Column("phase", field="phase"),  # "pre_sample" | "bo_step" | "random_sample"
        Column("index", field="index"),  # position within the initial sample, or 0-indexed BO iteration
        Column("structure", field="structure"),  # pretty-printed term (architecture + loss + optimizer + epochs)
        Column("objective_value"),  # raw validation loss, exactly what the objective function returned
        # The validation accuracy of the same trained model, which is what the search maximizes.  Loss
        # and accuracy do not order the candidates the same way, and a lower loss can come with a lower
        # accuracy, so both are recorded.
        Column("accuracy"),
        # The held-out accuracy.  Never a search signal: ranking candidates by it selects on the split
        # that is meant to stay untouched, and the column sits in the same row as the validation number,
        # which makes it easy to read by accident.  Empty for runs without a test split.
        Column("test_accuracy", render=_empty_if_none),
        # The number of function symbols the term writes, which is the axis the loop's own sampler
        # stratifies along.  Without it a size-uniform initial design cannot be shown to be one.
        Column("term_size", field="term_size"),
        Column("n_params"),  # trainable parameter count, to see whether the loop drifts to large nets
        Column("train_seconds"),  # wall-clock training time
        # Whether the numerics gave out, and how far the training got before they did.  Without these a
        # network that stopped in its first epoch is indistinguishable from one that trained through and
        # is merely bad: the abort still yields a finite accuracy, and on logits that are all not a
        # number ``argmax`` returns class 0, so the accuracy becomes that class's frequency.  Divergence
        # depends on the initial weights rather than on the architecture, so it belongs in the noise
        # column and not in the variance between architectures, and it is the one property of an
        # evaluation that cannot be recovered once the run is over.
        Column("diverged"),
        Column("epochs_completed"),
        # Beside the training time, because the two are the run's whole wall clock and which of them
        # dominates decides where a run is spending itself: only the training uses the accelerator, so a
        # run that spends most of its time here is bound by the processor however fast the card is.
        # Empty for pre_sample rows, which have no acquisition step.
        Column("acquisition_seconds", field="acquisition_seconds"),
        # The three columns below decide how the row may be read at all.  When ``suggest()`` cannot
        # find a novel candidate it replaces the optimizer's result with a random fallback sample, and
        # ``acquisition_value`` then describes that replacement.  A run whose loop-pass rows all report
        # a fallback was random search, and without these columns that is indistinguishable in the
        # finished file from a run that was not.  Empty for pre_sample rows, which have no acquisition
        # step.
        Column("acquisition_value", field="acquisition_value"),
        Column("fallback_used", field="fallback_used"),
        Column("fallback_attempts", field="fallback_attempts"),
        Column("timestamp", field="timestamp"),
        # The four columns of a repeated measurement.  ``accuracy`` above is their mean, which is what
        # the search optimizes, and these say what it is a mean of: how many trainings, their individual
        # values, their spread, and whether any single one of them diverged.  A run that kept only the
        # mean could not afterwards tell a candidate that measured 0.72 three times from one that
        # measured 0.50, 0.72 and 0.94, and the whole reason for repeating is that those two are not the
        # same finding.  Empty for a run with one training per candidate, where no repetition happened
        # to describe.
        Column("n_repeats"),
        Column("accuracy_runs", render=_joined),
        Column("accuracy_std", render=_empty_if_none),
        Column("diverged_runs", render=_joined),
    ),
)
#: The CSV's columns, the CIFAR schema's header: this driver's layout is one schema among others.
CSV_COLUMNS = CIFAR_SCHEMA.header


def split_train_validation(x, y, val_fraction=0.1, seed=20260803):
    """Split a training set into a training and a validation part, deterministically.

    Deterministic on purpose, and with its own seed.  Two runs have to see the same split, or their
    numbers are not comparable, and the split must not move when the search seed moves, or a sweep
    over seeds would silently be a sweep over splits as well.

    The default keeps a tenth of the training set for validation.  There is no single standard for
    the share, and the trade is plain: a larger validation part measures an accuracy more precisely,
    while a larger training part reaches a higher one.  What decides the trade is which of the two
    error terms is larger, the sampling error of the validation set or the spread of the training
    itself, and on these datasets the training spread is the larger of the two, so the split favors
    training data.

    Args:
        x (torch.Tensor): Training features.
        y (torch.Tensor): Training labels.
        val_fraction (float): Share that becomes validation. (Default value = 0.1)
        seed (int): Seed of the permutation. (Default value = 20260803)

    Returns:
        tuple: ``(x_train, y_train, x_val, y_val)``.
    """
    n = x.shape[0]
    n_val = int(round(n * val_fraction))
    if not 0 < n_val < n:
        msg = f"val_fraction {val_fraction} yields {n_val} of {n} samples"
        raise ValueError(msg)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(n, generator=generator).to(x.device)
    val_idx, train_idx = permutation[:n_val], permutation[n_val:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def _train_candidate_once(tree, x, y, x_val, y_val, batch_size, protocol,
                          x_test, y_test, seed):
    """Interpret, initialize and train one candidate exactly once.

    Split out of :func:`evaluate_candidate` so that repeating a measurement repeats all of it.  A
    repetition that reused the interpreted model would train an already-trained network and measure
    something else entirely.  ``Tree.interpret`` builds a new one on every call, so the
    interpretation belongs inside the repeated part rather than before it.

    Args:
        seed (int | None): Seeds the weight initialization and the mini-batch shuffling of this one
            training.  ``None`` leaves the global generator alone, which is unseeded training.  The
            caller is responsible for restoring the generator state, see
            :func:`evaluate_candidate`.

    Returns:
        dict: The metrics of this single training.
    """
    if seed is not None:
        # Before ``interpret``, because the weights are drawn while the modules are constructed.
        # Seeding afterwards would leave the initialization as the one unseeded step.
        torch.manual_seed(seed)

    model, loss_fn, optimizer_factory, epochs, scheduler_factory = tree.interpret(
        pytorch_components_algebra())
    n_params = sum(p.numel() for p in model.parameters())
    input_features = x.shape[-1]

    # Initialization happens here rather than inside ``learner``, because this is the one point
    # where a freshly interpreted model is in hand, and ``Tree.interpret`` builds a new one on every
    # call, so there is no risk of initializing a model that already trained.
    if protocol is not None and protocol.init == "he":
        apply_he_initialisation(model)

    # ``learner`` fills this in, and its own comment says why the flag has to leave the function.
    # The measurement is reported exactly as taken either way: this names it, it does not replace
    # it.
    training = {}
    started = time.time()
    objective_value = raw_learner(input_features, model, loss_fn, optimizer_factory, epochs,
                                  x, y, x_val, y_val, batch_size=batch_size, report=training,
                                  protocol=protocol, scheduler=scheduler_factory)
    train_seconds = time.time() - started

    # `learner` already moved the model onto the data's device and trained it in place.
    with torch.inference_mode():
        predictions = model(x_val).argmax(dim=-1)
        accuracy = (predictions == y_val).float().mean().item()
        # Recorded, never optimized against, see the docstring.
        test_accuracy = None
        if x_test is not None and y_test is not None:
            test_predictions = model(x_test).argmax(dim=-1)
            test_accuracy = (test_predictions == y_test).float().mean().item()

    return {
        "objective_value": objective_value,
        "accuracy": accuracy,
        "test_accuracy": test_accuracy,
        "n_params": n_params,
        "train_seconds": train_seconds,
        "diverged": training["diverged"],
        "epochs_completed": training["epochs_completed"],
    }


def evaluate_candidate(tree, x, y, x_val, y_val, batch_size, protocol=None,
                       x_test=None, y_test=None, repeats=1, training_seeds=None):
    """Train one candidate and return all recorded metrics.

    Uses ``pytorch_components_algebra`` rather than ``pytorch_function_algebra`` so that the trained
    model stays accessible for the accuracy and parameter measurements.  The training itself goes
    through the very same ``learner`` routine the function algebra would have used, so the objective
    value is directly comparable to runs that used it.

    What the search sees is the validation split.  ``accuracy`` and ``objective_value`` are measured
    on ``x_val``, and that is the number the loop maximizes.  If a test split is passed as well, its
    accuracy is recorded alongside as ``test_accuracy``, for the record only.

    ``test_accuracy`` is not a search signal and must not become one.  It sits in the same row as
    the validation number, which makes it easy to read by accident, and a table that ranks
    candidates by it has selected on the split that was meant to stay untouched.  Ranking, model
    choice and every stopping rule read ``accuracy``.

    Repetitions average the training noise away and never hide it.  The same architecture trained
    twice does not reach the same accuracy, and the spread between two trainings can be wider than
    the difference a search is trying to read, so a single training cannot decide between two
    candidates.  ``repeats`` trains the candidate that many times and hands the mean to the search,
    which divides that noise by the square root of the count.  The individual measurements survive
    in ``accuracy_runs`` and their spread in ``accuracy_std``: the mean is what the loop optimizes,
    the spread is what says whether a difference is readable at all, and a run that reported only
    the mean would have thrown away the very number that justified the averaging.

    Repetitions are neither free nor a substitute for a repeated run.  They average one
    architecture's training noise, while a sweep over search seeds asks a different question, namely
    whether the search finds the same thing twice.

    Args:
        x_val (torch.Tensor): The features the search is scored on.
        y_val (torch.Tensor): Their labels.
        protocol (TrainingProtocol | None): What the training does beyond reading the term, such as
            clipping, weight initialization and augmentation.  None means the default protocol, so
            an unset protocol changes nothing about an existing run.
        x_test (torch.Tensor | None): Held-out features, recorded but never optimized against.
        y_test (torch.Tensor | None): Their labels.
        repeats (int): How many independent trainings the reported metrics average over.
            (Default value = 1)
        training_seeds (Sequence[int] | None): One seed per repetition, or ``None`` for unseeded
            training.  Seeded repetitions make a run reproducible and make two runs comparable
            candidate by candidate, which an unseeded average does not.  The global generator is
            restored afterwards, so seeding the training cannot move the search's own draws, which
            come from the sampler and the acquisition optimizer and have to stay where the run's own
            seed put them. (Default value = None)

    Returns:
        dict: The averaged metrics, plus the per-repetition measurements they average over.

    Raises:
        ValueError: If ``repeats`` is not positive, if ``training_seeds`` has a different length, or
            if the repetitions disagree on the parameter count.  The last one would mean the same
            term interpreted to two different networks, which no averaging can repair.
    """
    if repeats < 1:
        msg = f"repeats must be at least 1, got {repeats}"
        raise ValueError(msg)
    if training_seeds is not None and len(training_seeds) != repeats:
        msg = (
            f"training_seeds has {len(training_seeds)} entries for {repeats} repetitions; "
            f"one seed per repetition or None for unseeded training"
        )
        raise ValueError(msg)

    seeds = list(training_seeds) if training_seeds is not None else [None] * repeats

    # The training's generator state is borrowed, not taken.  The loop's own sampler and the
    # acquisition-optimizing search draw from the same global generator, and a run whose training
    # advanced it would search differently than the same run with one training per candidate.
    rng_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        runs = [
            _train_candidate_once(tree, x, y, x_val, y_val, batch_size, protocol,
                                  x_test, y_test, seed)
            for seed in seeds
        ]
    finally:
        torch.set_rng_state(rng_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    parameter_counts = {run["n_params"] for run in runs}
    if len(parameter_counts) != 1:
        msg = (
            f"the {repeats} repetitions of one term produced different parameter counts "
            f"{sorted(parameter_counts)}; the same term must interpret to the same network"
        )
        raise ValueError(msg)

    accuracies = [run["accuracy"] for run in runs]
    objective_values = [run["objective_value"] for run in runs]
    # A test split is either configured for the whole run or for none of it, so the repetitions
    # agree.  None stays None rather than becoming a number no measurement produced.
    test_accuracies = [run["test_accuracy"] for run in runs]
    measured_test = [value for value in test_accuracies if value is not None]

    return {
        "objective_value": statistics.fmean(objective_values),
        "accuracy": statistics.fmean(accuracies),
        "test_accuracy": statistics.fmean(measured_test) if measured_test else None,
        "n_params": parameter_counts.pop(),
        # The sum, not the mean: this is what the run actually spent, and the column is read as
        # wall clock.
        "train_seconds": sum(run["train_seconds"] for run in runs),
        # Any repetition that diverged makes the candidate one whose numerics gave out.  Averaging
        # a diverged training with two healthy ones would report the mean of two different things,
        # and the flag is what says so.
        "diverged": any(run["diverged"] for run in runs),
        # The earliest stop, for the same reason: it is the one that says how far the worst of the
        # repetitions got.
        "epochs_completed": min(run["epochs_completed"] for run in runs),
        "n_repeats": repeats,
        "training_seeds": None if training_seeds is None else list(training_seeds),
        "accuracy_runs": accuracies,
        # The sample standard deviation of the repetitions.  It is undefined for a single one, and
        # left undefined rather than reported as 0.0, which would claim that a spread was measured
        # to be zero when none was measured at all.
        "accuracy_std": statistics.stdev(accuracies) if repeats > 1 else None,
        "objective_value_runs": objective_values,
        "test_accuracy_runs": test_accuracies,
        "diverged_runs": [run["diverged"] for run in runs],
    }


class EvaluationLogger(EvaluationRecorder):
    """Writes one row and one term record per evaluation, flushed as they go, in the CIFAR layout.

    The generic recorder is :class:`bayesian_optimization.runs.records.EvaluationRecorder`; this is
    that recorder with :data:`CIFAR_SCHEMA`, under the name and the signature this module has always
    had.

    The files stay open for the whole run on purpose.  The point of this logger is that results
    survive a crash after hours of training, so every row is written and flushed when it happens
    rather than collected and dumped at the end.  That is what the class's own ``__enter__`` and
    ``__exit__`` are for: it is the context manager, so ``open`` here is not a leak.

    Both artifacts come from one call, because the term record is the one a caller would forget.
    The CSV keeps a rendering of the structure, which no kernel can read back, while the
    ``_terms.pickle`` beside it keeps the term itself, and every question asked of a finished run
    offline needs the second file.  Threading a separate writer through the three call sites would
    have let a fourth one omit it silently, and a run that omits it cannot be repaired afterwards.
    It has to be retrained.  See :mod:`cnn_damg_term_pool`.

    Args:
        path (str): The run's CSV.  The term records go beside it as ``<run>_terms.pickle``.
        pretty_algebra (Callable): The algebra that renders a term into the CSV's structure column.
        provenance (dict): Written into the term file's header, so the pickle identifies its own
            origin on a machine that received nothing else. (Default value = None)
    """

    def __init__(self, path, pretty_algebra, provenance=None):
        super().__init__(path, pretty_algebra, CIFAR_SCHEMA, provenance=provenance)


def dataset_to_tensors(dataset, device, num_workers=0):
    """Materialize a whole split as one pair of tensors on ``device``.

    Flat-feature-vector convention: images become ``(N, C*H*W)``, matching the R^n -> R^m convention
    the repository and the algebras use everywhere else.  Labels stay class indices, long, for
    ``cross_entropy_loss``.  The split is loaded in a single batch, which keeps every candidate
    evaluation free of dataloader overhead, at the price of holding the whole split in the device's
    memory at once.

    Args:
        dataset: A dataset the caller has already constructed.  Nothing is downloaded here.
        device (torch.device): Where the tensors go.
        num_workers (int): Workers for the one-shot load.  0 avoids the multiprocessing overhead
            that buys nothing for a single batch. (Default value = 0)

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The features and the labels.
    """
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=len(dataset), shuffle=False, num_workers=num_workers
    )
    images, labels = next(iter(loader))
    return images.reshape(images.shape[0], -1).to(device), labels.to(device).long()


def objective_direction(objective):
    """Turn the ``--objective`` choice into the two things the run needs from it.

    The loop maximizes, always.  A loss therefore enters it negated and comes back out negated, and
    every number read off the loop goes through the returned converter.  That the two scripts
    disagreed on one line of this, one reporting the loop's value and the other the metric's, is why
    it is one function now.

    Args:
        objective (str): ``"accuracy"`` (maximized) or ``"loss"`` (minimized).

    Returns:
        tuple[bool, Callable[[float], float]]: Whether greater is better, and the converter from
            what the loop reports back into the metric's own scale and sign.

    Raises:
        ValueError: If the objective is neither of the two.  The direction of a search is not
            something to guess a default for.
    """
    if objective not in ("accuracy", "loss"):
        msg = f"objective must be 'accuracy' or 'loss', got {objective!r}"
        raise ValueError(msg)
    greater_is_better = objective == "accuracy"

    def as_reported(value):
        """Turn a value the loop reports back into the metric's own scale and sign.

        Args:
            value (float): What the loop reported.

        Returns:
            float: The metric.
        """
        return float(value) if greater_is_better else -float(value)

    return greater_is_better, as_reported


#: The structure lengths the two length-named targets ask for.  Written out rather than imported
#: from the tooling that also states them: an example must not depend on the tooling, and a
#: dictionary of two entries is the cheaper duplicate.
TARGET_LENGTHS = {"L3": 3, "L5": 5}

#: The targets whose length is fixed by a position list rather than chosen.  A structure length is a
#: contradiction for every one of them, so they are refused together.  Each is built on a reference
#: architecture and carries its length in the positions it names.
POSITION_TARGETS = ("HEAD", "TUT1", "TUT2", "TUT3", "VGGM", "VGGN")


def resolve_target_choice(target, structure_length, default_length):
    """Turn ``--target`` and ``--structure-length`` into the one pair a run can be built from.

    The two arguments say the same thing in different vocabularies, since ``--target L3`` and
    ``--structure-length 3`` name one cell, and the position targets say something neither can:
    their length is fixed by their position list rather than chosen.  Resolving them in one place is
    what keeps the two experiment scripts from drifting apart on it.

    A contradiction raises rather than resolves.  Silently letting one argument win would produce
    a run whose CSV says one cell and whose target is another, and every number in it is plausible
    for the cell it claims to be.

    Args:
        target (str | None): ``"L3"``, ``"L5"``, ``"HEAD"``, or None to go by the length alone.
        structure_length (int | None): The explicit length, or None if it was not given.
        default_length (int): The script's default length, used when neither argument is given.

    Returns:
        tuple[str | None, int | None]: The cell label to record, and the structure length, which is
            None for a target whose length is its position list's.

    Raises:
        ValueError: If the two arguments contradict each other, or if a length is given alongside
            the head target, which has no length to set.
    """
    if target in POSITION_TARGETS:
        if structure_length is not None:
            msg = (
                f"--target {target} takes no --structure-length (got {structure_length}): this "
                f"target's length is fixed by its position list, which is what makes it more than "
                f"a sub-space of the length targets"
            )
            raise ValueError(msg)
        return target, None

    if target is not None:
        wanted = TARGET_LENGTHS[target]
        if structure_length is not None and structure_length != wanted:
            msg = (
                f"--target {target} is length {wanted}, but --structure-length says "
                f"{structure_length}; drop one of the two"
            )
            raise ValueError(msg)
        return target, wanted

    length = default_length if structure_length is None else structure_length
    # A free length that happens to be one of the two named ones is that cell and says so in the
    # record.  Any other length carries no cell label.
    label = next((name for name, size in TARGET_LENGTHS.items() if size == length), None)
    return label, length


def make_objective(x, y, x_val, y_val, batch_size, greater_is_better,
                   protocol=None, x_test=None, y_test=None, repeats=1, training_seeds=None):
    """Build the objective function and the store its measurements land in.

    Every candidate's full metric set is kept, keyed by its term: the loop takes one value per
    evaluation, while the run's CSV row and term pool need every metric that evaluation measured,
    so the metrics are stored here and read back by the driver right after each call.

    Args:
        x (torch.Tensor): Training features.
        y (torch.Tensor): Training labels.
        x_val (torch.Tensor): Validation features, which are what the loop is scored on.
        y_val (torch.Tensor): Validation labels.
        batch_size (int): Mini-batch size for training.
        protocol (TrainingProtocol | None): The training protocol, or None for the default.
        x_test (torch.Tensor | None): Held-out features, recorded but never optimized against.
        y_test (torch.Tensor | None): Their labels.
        greater_is_better (bool): Whether the loop maximizes the metric directly.
        repeats (int): How many trainings each candidate's value averages over, see
            :func:`evaluate_candidate`. (Default value = 1)
        training_seeds (Sequence[int] | None): One seed per repetition, or None for unseeded
            training. (Default value = None)

    Returns:
        tuple[Callable, dict]: The objective, and the measurement store it writes into.
    """
    metrics_by_tree = {}

    def f_obj(tree):
        """Train one candidate and return what the loop maximizes.

        Args:
            tree: The candidate term.

        Returns:
            float: The accuracy, or the negated loss, averaged over ``repeats`` trainings.
        """
        metrics = evaluate_candidate(tree, x, y, x_val, y_val, batch_size,
                                     protocol=protocol, x_test=x_test, y_test=y_test,
                                     repeats=repeats, training_seeds=training_seeds)
        metrics_by_tree[tree] = metrics
        # Both metrics always reach the CSV.  Only which one the loop maximizes changes.
        return metrics["accuracy"] if greater_is_better else -metrics["objective_value"]

    return f_obj, metrics_by_tree


def _draw_prefix(optimizer, count):
    """Take ``count`` pairwise distinct terms from one stream of the loop's own sampler.

    One stream and not ``count`` draws.  Calling ``sample()`` twice would start two streams from an
    advanced generator, and the second could repeat the first, which is exactly the property a
    paired baseline is built on.

    The distinctness is produced here, not inherited.  Every prefix of a size-uniform stream is a
    sample without replacement, so on that sampler the terms are distinct by construction.  The
    depth-bounded sampler says the opposite of itself: its stream is a sequence of independent
    draws, and independent draws may repeat a term.  A configuration whose determinization is
    unaffordable runs exactly that sampler, so the guarantee cannot be assumed for every run.

    The repeats are therefore rejected and counted.  Zero is the size-uniform case, where the
    guarantee holds.  A positive number is the repair where it does not, and it goes into the run's
    provenance rather than into a comment.

    Asked through ``optimizer.query`` rather than through a query of our own, so that the sampler's
    counting construction is the one the loop will use.  A second query object would build it again.

    Args:
        optimizer (BayesianOptimization): The configured loop, before ``initialize()``.
        count (int): How many terms to take.

    Returns:
        tuple[list, int]: The terms in stream order, and how many repeats were rejected.

    Raises:
        RuntimeError: If the loop has no search space to draw from, or if the stream cannot supply
            ``count`` distinct terms.  A short design is not a design, and filling it up with
            repeats would make the baseline evaluate the same network twice.
    """
    query = optimizer.query
    if query is None or optimizer.sampler is None:
        msg = (
            "a paired baseline draws its terms from the loop's own sampler, and this loop has "
            "no search space or no sampler to draw them from"
        )
        raise RuntimeError(msg)
    return distinct_prefix(optimizer.sampler, query, count)


def run_ask_tell_search(
    *,
    optimizer,
    f_obj,
    metrics_by_tree,
    as_reported,
    objective,
    greater_is_better,
    n_pre_samples,
    n_iterations,
    csv_path,
    pretty_algebra,
    provenance=None,
    baseline=False,
    verbose=True,
    acquisition_hard_limit=ACQUISITION_HARD_LIMIT_SECONDS,
    resume_design=None,
    resumed_value=None,
):
    """Run the loop through the ask/tell interface, logging every evaluation as it happens.

    Ask and tell rather than the closed ``optimize()``.  Every evaluated structure, pre-samples and
    loop passes alike, is written and flushed as it is produced, so results survive an interruption
    instead of existing only after a successful ``finalize()``.  That is the whole reason this is
    not four lines.

    Args:
        optimizer (BayesianOptimization): The configured loop.
        f_obj (Callable): The objective, as :func:`make_objective` returns it.
        metrics_by_tree (dict): Its measurement store.
        as_reported (Callable[[float], float]): The converter from :func:`objective_direction`.
        objective (str): The metric's name, for the printed summary.
        greater_is_better (bool): Whether the loop maximizes it directly.
        n_pre_samples (int): The size of the initial design, drawn before anything is evaluated
            and then evaluated term by term through the loop's design phase.
        n_iterations (int): The budget of loop passes.
        csv_path (str): Where the per-evaluation rows go.
        pretty_algebra (Callable): The algebra that renders a term for the CSV.
        provenance (dict): The run's provenance record, written into the header of
            ``<run>_terms.pickle``, so that the term file identifies its own origin when it travels
            without the JSON beside it, which is what happens when a pool is trained on one machine
            and analyzed on another. (Default value = None)
        baseline (bool): Also run a random search of the same budget from the same initial design.
            Whether the loop beats drawing at random is a question about the passes only if both
            start from the same place, so the two share their initial evaluations rather than each
            drawing their own, which is also why this costs one extra training per pass and not one
            per evaluation. (Default value = False)
        verbose (bool): Passed to ``suggest``. (Default value = True)
        acquisition_hard_limit (float): Seconds after which one acquisition maximization is given up
            on.  It bounds a legitimate duration, so it belongs to the cell being searched rather
            than to the code, and a large architecture at a large population can outlast the
            default. (Default value = ACQUISITION_HARD_LIMIT_SECONDS)
        resume_design (list | None): The measured design of an earlier run of this
            configuration, as :func:`load_initial_design` returns it.  Its terms are checked
            against the design this run draws and its values are taken over rather than measured
            again, on both paths. (Default value = None)
        resumed_value (Callable[[dict], float] | None): How a resumed record's metrics become
            the value the loop maximizes -- what ``f_obj`` would have returned for that term.
            ``None`` keeps the CIFAR example's convention, ``metrics["accuracy"]`` when greater
            is better and ``-metrics["objective_value"]`` otherwise; a caller whose objective
            maps a record differently names its own reading here, or its resumed design seeds
            the surrogate with a quantity the run's rows do not report.

    Returns:
        tuple[dict, float, dict]: The loop's result, its wall-clock duration, and the summary to
            merge into the provenance record.
    """
    print(f"Starting Bayesian Optimization: n_pre_samples={n_pre_samples}, "
          f"n_iterations={n_iterations}")
    print(f"Logging every evaluated structure to {csv_path}")

    started = time.time()
    # A pass configuration the loop could not use is refused before the design is drawn or paid,
    # as ``optimize()`` refuses it for itself.  A run without passes maximizes nothing.
    if n_iterations > 0:
        optimizer.check_configuration()
    # The paired baseline needs its terms before the loop starts, because the ones the loop
    # initializes on have to be the same objects.  One stream of as many draws as the two runs
    # together evaluate: on the size-uniform sampler every prefix of that stream is a sample without
    # replacement, so its first entries are a valid initial design on their own, and the two runs
    # share a starting point instead of each getting their own.
    drawn = None
    repeats_rejected = 0
    if baseline:
        with step_budget(
            f"drawing {n_pre_samples + n_iterations} terms for the paired baseline",
            ACQUISITION_WARN_SECONDS,
        ):
            drawn, repeats_rejected = _draw_prefix(
                optimizer, n_pre_samples + n_iterations
            )
            if repeats_rejected:
                print(
                    f"the sampler repeated {repeats_rejected} term(s) while drawing "
                    f"{n_pre_samples + n_iterations} for the paired design; they were rejected.  "
                    f"On the size-uniform sampler this number is 0, since every prefix of its "
                    f"stream is a sample without replacement",
                    flush=True,
                )

    with (
        EvaluationLogger(csv_path, pretty_algebra, provenance=provenance) as logger,
        EAGenerationLogger(_sibling_path(csv_path, "_ea.csv")) as ea_logger,
        SurrogateLogger(_sibling_path(csv_path, "_surrogate.csv")) as surrogate_logger,
    ):
        # The initial design is the loop's first phase on both paths: drawn before anything is
        # evaluated, handed out one term at a time, and written as it is measured.  The paired path
        # hands the loop the head of its own stream, the unpaired path lets the loop draw.  Either
        # way a run interrupted during its design leaves every network it trained on disk, and a
        # resumed design is taken over term by term instead of being measured again.
        #
        # This used to hold on the paired path only.  The unpaired path drew and evaluated its
        # design inside ``initialize()``, so its rows were written after the design was complete
        # and a resumed design had no point at which it could be checked and taken over: a run
        # that wanted either had to pay a random-search arm of the same budget for it.
        with step_budget(
            f"initialization ({n_pre_samples} networks)",
            n_pre_samples * PER_EVALUATION_WARN_SECONDS,
        ):
            if drawn is None:
                optimizer.initialize(initial_size=n_pre_samples)
            else:
                optimizer.initialize(design=drawn[:n_pre_samples])
            design = list(optimizer.design)
            if resume_design is not None:
                _check_resumed_design(resume_design, design)
            for idx, expected in enumerate(design):
                suggestion = optimizer.suggest(verbose=verbose)
                tree = suggestion.candidate
                if tree != expected:
                    msg = (
                        f"the loop handed out design term {idx} out of order: {tree} where the "
                        f"design holds {expected}, so a value would be written against the "
                        "wrong network"
                    )
                    raise RuntimeError(msg)
                if resume_design is not None:
                    # Measured already, in the run this one continues.  The value is taken
                    # rather than measured again, which is the whole point, and it goes into
                    # ``metrics_by_tree`` so that nothing downstream can tell the difference.
                    metrics = resume_design[idx][1]
                    metrics_by_tree[tree] = metrics
                    if resumed_value is not None:
                        value = float(resumed_value(metrics))
                    else:
                        value = (
                            metrics["accuracy"] if greater_is_better
                            else -metrics["objective_value"]
                        )
                    source = " (resumed)"
                else:
                    value = f_obj(tree)
                    # No substitute for a missing measurement: the objective records the metrics
                    # of every term it evaluates, and a value without them is worth stopping for
                    # rather than filling in.
                    if tree not in metrics_by_tree:
                        msg = (
                            f"pre-sample {idx} was evaluated, yet the objective recorded no "
                            f"metrics for it: {tree}"
                        )
                        raise KeyError(msg)
                    metrics = metrics_by_tree[tree]
                    source = ""
                # Written before the loop takes the value, so that a value the loop refuses, a
                # non-finite one, is on disk together with the metrics that explain it.
                logger.log("pre_sample", idx, tree, metrics)
                print(f"  pre_sample[{idx}]: objective={as_reported(value):.5f} "
                      f"accuracy={metrics['accuracy']:.4f} "
                      f"params={metrics['n_params']} "
                      f"train={metrics['train_seconds']:.1f}s{source}", flush=True)
                optimizer.observe(tree, value)

        for step in range(n_iterations):
            acquisition_started = time.time()
            with step_budget(
                f"BO step {step}: acquisition optimization",
                ACQUISITION_WARN_SECONDS,
                hard_limit_seconds=acquisition_hard_limit,
            ):
                # ``record_population``: the frontier read of the acceptance checks is a statement
                # about one maximization, and it cannot be rebuilt afterwards.  A surrogate refitted
                # on the whole dataset scores the same population differently, so members that lost
                # to the pick at the time can beat it later.  It has to be the population as it was
                # scored, and it is also what carries the per-generation records of the inner
                # search.
                suggestion = optimizer.suggest(verbose=verbose, record_population=True)
            acquisition_seconds = time.time() - acquisition_started
            # Written before the network trains: the inner search is finished at this point, and a
            # run interrupted during the training then still holds the generation it produced.
            ea_logger.log(step, optimizer.last_acquisition_run)
            surrogate_logger.log(step, optimizer)
            with step_budget(
                f"BO step {step}: training the suggested network",
                PER_EVALUATION_WARN_SECONDS,
            ):
                # Not named ``y``: ``f_obj`` closes over the label tensor of that name, and
                # rebinding it here would clobber it for every later call.
                objective_value = f_obj(suggestion.candidate)
            optimizer.observe(suggestion.candidate, objective_value)
            iteration = suggestion.diagnostics["iteration"] if suggestion.diagnostics else None
            metrics = metrics_by_tree[suggestion.candidate]
            logger.log("bo_step", iteration, suggestion.candidate, metrics, suggestion=suggestion,
                       acquisition_seconds=acquisition_seconds)
            # The fallback flag decides how this line reads: after a fallback the candidate is a
            # random sample rather than the acquisition optimizer's choice.  Seeing it live is the
            # difference between noticing a degraded run and reading it out of the CSV a day later.
            fallback = (suggestion.diagnostics or {}).get("fallback_used", "?")
            print(f"  bo_step[{iteration}]: objective={as_reported(objective_value):.5f} "
                  f"accuracy={metrics['accuracy']:.4f} params={metrics['n_params']} "
                  f"train={metrics['train_seconds']:.1f}s fallback={fallback}", flush=True)

        result = optimizer.finalize()
        bo_time = time.time() - started

        # The baseline's own evaluations, after the loop's, so that an interruption here leaves
        # the Bayesian run complete.  These terms come from the same stream as the shared prefix,
        # so together with it they are the draws a random search of this budget makes.
        baseline_rows = []
        if drawn is not None:
            for offset, tree in enumerate(drawn[n_pre_samples:]):
                with step_budget(
                    f"baseline sample {offset}: training the drawn network",
                    PER_EVALUATION_WARN_SECONDS,
                ):
                    value = f_obj(tree)
                metrics = metrics_by_tree[tree]
                logger.log("random_sample", offset, tree, metrics)
                baseline_rows.append(value)
                print(f"  random_sample[{offset}]: objective={as_reported(value):.5f} "
                      f"accuracy={metrics['accuracy']:.4f} params={metrics['n_params']} "
                      f"train={metrics['train_seconds']:.1f}s", flush=True)

    n_evaluations = n_pre_samples + n_iterations
    # The same rule as at the pre-samples: a term the loop reports must be one it was given a
    # value for, and a missing measurement is a fact rather than an empty dict.
    if result["best_tree"] not in metrics_by_tree:
        msg = f"the loop returned a best term that was never evaluated: {result['best_tree']}"
        raise KeyError(msg)
    best_metrics = metrics_by_tree[result["best_tree"]]
    print(f"Bayesian Optimization took {bo_time:.2f}s ({n_evaluations} network evaluations, "
          f"{bo_time / max(n_evaluations, 1):.2f}s per evaluation on average)")
    print(f"Best {objective} ({'max' if greater_is_better else 'min'}): "
          f"{as_reported(result['best_y']):.5f}")
    if "accuracy" in best_metrics:
        print(f"Best tree's validation accuracy: {best_metrics['accuracy'] * 100:.2f}%")
    if "objective_value" in best_metrics:
        print(f"Best tree's validation loss: {best_metrics['objective_value']:.5f}")
    print(f"Best tree:\n{result['best_tree'].interpret(pretty_algebra())}")

    summary = {
        "bayesian_optimization_seconds": bo_time,
        "n_evaluations": n_evaluations,
        "mean_seconds_per_evaluation": bo_time / max(n_evaluations, 1),
        "total_training_seconds": sum(m["train_seconds"] for m in metrics_by_tree.values()),
        # In the metric's own scale and sign, like every other number read out of the loop.  The
        # value the loop itself maximized stands beside it, named for what it is, because on a loss
        # objective the two differ and a record that shows only one is ambiguous.
        "best_objective_value": as_reported(result["best_y"]),
        "best_maximized_value": float(result["best_y"]),
        "best_accuracy": best_metrics.get("accuracy"),
        # Read off the budget rather than taken as a flag.  A loop with no passes evaluates its
        # initial dataset and stops, and that dataset is a size-uniform sample, so such a run is
        # random search over the same space, at the same budget, through the same code.  A separate
        # switch could disagree with the numbers, and this cannot.
        "run_kind": (
            "random_search_baseline" if n_iterations == 0
            else "bayesian_optimization_with_paired_baseline" if baseline
            else "bayesian_optimization"
        ),
        # How many repeats the sampler produced while the design was drawn, and how many the loop
        # itself had to redraw.  Zero on the size-uniform sampler, where every prefix is a sample
        # without replacement.  On the depth-bounded sampler the draws are independent and may
        # repeat, so the number says how much work the rejection had to do.  It is in the record
        # because a run whose initial design needed repair is a run whose sampler does not supply
        # what its dataset needs.
        "initial_repeats_rejected": (
            repeats_rejected if drawn is not None else optimizer.initial_repeats_rejected
        ),
        "completed": True,
    }
    if drawn is not None:
        # The two curves this run produces, at the same budget and from the same initial points.
        # Both in the metric's own scale and sign, like every other number here.
        shared = [optimizer.get_state_snapshot()["y_list"][index] for index in range(n_pre_samples)]
        summary["baseline_best_objective_value"] = as_reported(max([*shared, *baseline_rows]))
        summary["shared_initial_best_objective_value"] = as_reported(max(shared))
        summary["baseline_evaluations"] = n_pre_samples + n_iterations
        summary["shared_initial_evaluations"] = n_pre_samples
    return result, bo_time, summary


def write_run_metadata(csv_path, metadata):
    """Write the run's provenance next to its CSV and return the path used.

    The framework-neutral record of :func:`bayesian_optimization.runs.metadata.write_run_metadata`,
    with this example's stack as its environment: the torch it trained with and the accelerator.
    """
    return _write_run_metadata(csv_path, metadata, environment={
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
