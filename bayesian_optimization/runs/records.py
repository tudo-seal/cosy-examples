"""The records of a run: one row per evaluation, the inner search's generations, the surrogate's reads."""

import csv
import json
import os
import time

import numpy as np

from bayesian_optimization.diagnostics import read_calibration, read_fit

from .artifacts import _sibling_path
from .term_pool import TermPoolWriter

EA_CSV_COLUMNS = [
    "bo_iteration",      # which pass of the outer loop this inner run belongs to
    "generation",        # 0 is the initial population
    "best",              # b, the fittest individual of the whole inner run so far
    "population_best",   # the fittest member of THIS generation, which b may already beat
    "population_mean",
    "population_worst",
    "distinct_members",  # a collapsed population repeats itself
    # The generation the best individual was last replaced in.  The gap to ``generation`` is how
    # long the inner search has been stalled, which is what a termination bound is set against.
    "last_improvement",
    # Zero says that every offspring of this generation was discarded by the acceptance test, which
    # the fitness columns alone do not show.
    "offspring",
]

SURROGATE_CSV_COLUMNS = [
    "bo_iteration",
    # how many pairs the pass conditioned on, in a round the pending passes at their assumed values
    # among them
    "n_train",
    "log_marginal_likelihood",
    # The leave-one-out calibration, per pass rather than once at the end.  Read the two spreads
    # together: the spread about zero and the spread about their own mean differ by the bias, which
    # is the signature of a surrogate whose deviations stay at the prior.
    "calibration_root_mean_square",
    "calibration_standard_deviation",
    "calibration_maximum_absolute",
    "calibration_outside_two",
    # The held-out fit, per pass.  The rank correlation alone does not separate a good surrogate
    # from a collapsed one, since a surrogate that predicts one value for everything can still order
    # a few ties by chance.  The prediction spread against the objective spread is what does, so all
    # three are here.
    "fit_size",
    "fit_rank_correlation",
    "fit_prediction_spread",
    "fit_objective_spread",
    "fit_residual_root_mean_square",
    # What the model selection actually did.  A kernel whose amplitudes never move is one whose fit
    # had nothing to adjust, and nothing else in these artifacts would say so, because sklearn runs
    # its optimizer over an empty parameter vector without complaining.
    "kernel_hyperparameters",
]


#: What the run layer reads off a fitted Gaussian process, and only there.
GAUSSIAN_PROCESS_READS = ("L_", "alpha_", "y_train_", "X_train_", "log_marginal_likelihood_value_",
                          "kernel_")


def held_out_split(optimizer, terms, values):
    """Split a run's data into the half a held-out fit conditions on and the half it predicts.

    By the parity of the rows, which keeps the conditioning half spread over the whole run instead
    of over one region of it.  Under repeated measurements by the parity of the distinct terms, in
    the order they were first measured, every row of a term on its term's side: split by rows, a
    term measured again lands on both sides, or twice on the conditioning one, and the surrogate
    would have seen what it is asked to predict.

    Args:
        optimizer: The loop; its ``repeated_measurements``, where it has one, decides the split.
        terms (Sequence): The dataset's terms, one per row.
        values (Sequence[float]): Their values.

    Returns:
        tuple | None: The conditioning terms and values and the held-out terms and values, or None
            where no fit read is about the split: fewer than two held-out terms, an empty
            conditioning half, or, split by rows, a term twice on the conditioning side or on both.
    """
    terms, values = list(terms), list(values)
    if getattr(optimizer, "repeated_measurements", False):
        side = {term: index % 2 for index, term in enumerate(dict.fromkeys(terms))}
        conditioned = [row for row, term in enumerate(terms) if side[term] == 0]
        held_out = [row for row, term in enumerate(terms) if side[term] == 1]
        if len({terms[row] for row in held_out}) < 2 or not conditioned:
            return None
        return (
            [terms[row] for row in conditioned], [values[row] for row in conditioned],
            [terms[row] for row in held_out], [values[row] for row in held_out],
        )
    conditioned_terms, held_out_terms = terms[0::2], terms[1::2]
    if (
        len(held_out_terms) < 2
        or not conditioned_terms
        or len(set(conditioned_terms)) != len(conditioned_terms)
        or set(conditioned_terms) & set(held_out_terms)
    ):
        return None
    return conditioned_terms, values[0::2], held_out_terms, values[1::2]


def reads_as_a_gaussian_process(posterior):
    """Whether a posterior answers what the run layer reads off a fitted Gaussian process.

    Read off what it has, not off its class: a subclass's surrogate or a caller's may answer a
    posterior that delegates to one.

    Args:
        posterior: A fitted surrogate.

    Returns:
        bool: Whether it has every attribute of :data:`GAUSSIAN_PROCESS_READS`.
    """
    return all(hasattr(posterior, name) for name in GAUSSIAN_PROCESS_READS)


def kernel_hyperparameters(surrogate):
    """Return the fitted kernel's hyperparameters as ``{name: value}``.

    ``theta`` holds the logarithm of the value of every hyperparameter that lives on a log scale,
    which is all of the ones here, so it is exponentiated back into the amplitudes and noise levels
    a reader recognizes.  Walked with an explicit index rather than zipped, because a hyperparameter
    may hold several elements and a zip would then silently pair names with the wrong numbers.

    Args:
        surrogate: A fitted ``GaussianProcessRegressor``.

    Returns:
        dict: One entry per hyperparameter of the fitted kernel.
    """
    values = np.exp(surrogate.kernel_.theta)
    result = {}
    index = 0
    for hyperparameter in surrogate.kernel_.hyperparameters:
        count = hyperparameter.n_elements
        result[hyperparameter.name] = (
            float(values[index])
            if count == 1
            else [float(value) for value in values[index:index + count]]
        )
        index += count
    return result


class EAGenerationLogger:
    """Writes the trajectory of every acquisition maximization, one row per generation.

    The frontier read says where the answer sits in the population it came from.  This says whether
    the inner run got there by improving.  Convergence is a statement about the best individual over
    the generations, and without these rows a run holds no evidence for or against it, because the
    loop keeps only the last generation of the last pass.

    Flushed per pass for the same reason the evaluation log is: the inner run is finished before the
    network trains, and a run interrupted during the training should keep it.
    """

    def __init__(self, path):
        self._file = open(path, "w", newline="")  # noqa: SIM115, closed by __exit__
        self._writer = csv.writer(self._file)
        self._writer.writerow(EA_CSV_COLUMNS)
        self._file.flush()

    def log(self, bo_iteration, acquisition_run):
        """Append one inner run's generations.

        Args:
            bo_iteration (int): The pass of the outer loop.
            acquisition_run: The ``AcquisitionRun`` the pass recorded, or None.

        Raises:
            ValueError: If the run recorded no generations at all.  Every inner run yields at
                least its initial population, so an empty sequence means the recording was not
                asked for, and a silently empty file would read as if the search had done nothing.
        """
        if acquisition_run is None:
            return
        if not acquisition_run.generations:
            msg = (
                f"pass {bo_iteration} recorded an acquisition run with no generations; every run "
                f"yields at least the zeroth, so this is a maximization that was not asked to "
                f"record its trajectory rather than one that had none"
            )
            raise ValueError(msg)
        for record in acquisition_run.generations:
            self._writer.writerow([
                bo_iteration,
                record.generation,
                record.best,
                record.population_best,
                record.population_mean,
                record.population_worst,
                record.distinct_members,
                record.last_improvement,
                record.offspring,
            ])
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class SurrogateLogger:
    """Writes what the surrogate of each pass believed, and how well.

    The acceptance checks read the surrogate once, at the end.  That answers whether the model was
    usable and not whether it became usable, and the second question is the one a plot of a run is
    about: a loop whose calibration improves as evaluations arrive is working, and one whose
    calibration stays where it started is over-confident from the first pass to the last.

    The held-out fit costs one extra Gaussian-process fit per pass, which is seconds at these
    dataset sizes against a pass that trains a neural network.

    In a round a pass's surrogate holds the passes pending before it at their assumed values, and
    the reads of that surrogate count them: its ``n_train``, and its calibration, whose residual at
    an assumed value is the one the model was built to reproduce.  The held-out fit and the reads
    over the finished run condition on observations only.
    """

    def __init__(self, path):
        self._file = open(path, "w", newline="")  # noqa: SIM115, closed by __exit__
        self._writer = csv.writer(self._file)
        self._writer.writerow(SURROGATE_CSV_COLUMNS)
        self._file.flush()

    def log(self, bo_iteration, optimizer):
        """Append one pass's surrogate reads.

        Args:
            bo_iteration (int): The pass of the outer loop.
            optimizer (BayesianOptimization): The loop, just after ``suggest()``.
        """
        surrogate = optimizer.surrogate
        if surrogate is None:
            return
        # The reads only a Gaussian process answers -- its calibration, its training set, its
        # marginal likelihood, its fitted kernel -- read off any posterior that has them, whatever
        # its class.  One that has not leaves its cells empty rather than failing the run after its
        # design.
        gaussian = reads_as_a_gaussian_process(surrogate)
        # The leave-one-out read leaves each observation out against the others, so it needs two.
        calibration = (
            read_calibration(surrogate) if gaussian and len(surrogate.X_train_) >= 2 else None
        )

        # The same split by parity that the diagnostics use, over the data this pass saw.  The
        # cells stay empty where the split is not one a fit read is about:
        #
        # * fewer than two held-out terms, because a scatter of one point has no spread and no
        #   rank;
        # * a term on both sides, or twice on the conditioning side, because the surrogate would
        #   then have seen what it is asked to predict, and a noise-free Gaussian process
        #   reproduces its training values exactly, so the scatter would sit on the diagonal
        #   whatever the kernel does.  That is the one thing this read exists to rule out.
        #
        # The loop's rejection path makes the second case unreachable through ``suggest()``.  It is
        # guarded because a caller may seed the dataset with ``x0`` directly, and a run should not
        # die in its logger.
        snapshot = optimizer.get_state_snapshot()
        terms, values = snapshot["x_list"], snapshot["y_list"]
        # The loop's count of what the pass was handed, read just after its suggest: every row
        # under repeated measurements, else the distinct terms, and the passes pending before it,
        # at their assumed values, in a round.
        conditioned_on = (
            len(values) if getattr(optimizer, "repeated_measurements", False) else len(set(terms))
        ) + max(len(snapshot.get("outstanding", [])) - 1, 0)
        split = held_out_split(optimizer, terms, values)
        fit = None
        if split is not None:
            conditioned, conditioned_values, held_out, held_out_values = split
            fit = read_fit(
                optimizer.surrogate_over(conditioned, conditioned_values), held_out, held_out_values
            )

        self._writer.writerow([
            bo_iteration,
            # the pairs this pass conditioned on: the Gaussian process's own count, or the loop's
            len(surrogate.X_train_) if gaussian else conditioned_on,
            surrogate.log_marginal_likelihood_value_ if gaussian else "",
            "" if calibration is None else calibration.root_mean_square,
            "" if calibration is None else calibration.standard_deviation,
            "" if calibration is None else calibration.maximum_absolute,
            "" if calibration is None else calibration.outside_two,
            "" if fit is None else fit.size,
            "" if fit is None or fit.rank_correlation is None else fit.rank_correlation,
            "" if fit is None else fit.prediction_spread,
            "" if fit is None else fit.objective_spread,
            "" if fit is None else fit.residual_root_mean_square,
            json.dumps(kernel_hyperparameters(surrogate), sort_keys=True) if gaussian else "",
        ])
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class EvaluationRecorder:
    """Writes one CSV row and one term record per evaluation, flushed as they go.

    The files stay open for the whole run on purpose: the point is that results survive a crash
    after hours of evaluation, so every row is written and flushed when it happens rather than
    collected and dumped at the end.  The class is its own context manager, so the ``open`` below is
    not a leak.  Both artifacts come from one call, because the term record is the one a caller
    would forget, and a run that omits it cannot be repaired afterwards.

    The columns are the schema's (:class:`~bayesian_optimization.runs.schema.MetricSchema`), so a
    caller records its own metrics under its own names; the term record keeps the metrics whole,
    the value the loop was handed, which a resumed design is checked against, and whether the row
    was taken over.

    Args:
        path (str): The run's CSV.  The term records go beside it as ``<run>_terms.pickle``.
        pretty_algebra (Callable): The algebra that renders a term into the ``structure`` column.
        schema (MetricSchema): The columns, in order.
        provenance (dict): Written into the term file's header. (Default value = None)
        mode (str): ``"w"`` writes over whatever the two paths hold; ``"x"`` refuses a path that
            exists, both files checked before either is opened, so a second start against a run
            that is still being written truncates nothing. (Default value = "w")
    """

    def __init__(self, path, pretty_algebra, schema, provenance=None, *, mode="w",
                 design_origin=None):
        if mode not in ("w", "x"):
            msg = f"mode is 'w' or 'x', not {mode!r}"
            raise ValueError(msg)
        terms_path = _sibling_path(path, "_terms.pickle")
        if mode == "x":
            taken = [p for p in (path, terms_path) if os.path.exists(p)]
            if taken:
                msg = f"a run already writes to {', '.join(taken)}; refused rather than truncated"
                raise FileExistsError(msg)
        self._pretty_algebra = pretty_algebra
        self._schema = schema
        self._file = open(path, mode, newline="")  # noqa: SIM115, see the class docstring
        self._writer = csv.writer(self._file)
        self._writer.writerow(schema.header)
        self._file.flush()
        try:
            self._terms = TermPoolWriter(terms_path, provenance=provenance, mode=mode,
                                         design_origin=design_origin)
        except BaseException:
            # The CSV is already open here, and a caller that never got an instance back cannot
            # close it, since there is no ``__exit__`` for an object whose ``__init__`` raised.
            self._file.close()
            if mode == "x":
                # Created by this call and holding nothing but its header: removed, so that the
                # retry is not refused as a run that is still writing.
                os.remove(path)
            raise

    def log(self, phase, index, tree, metrics, suggestion=None, acquisition_seconds=None,
            loop_value=None, taken_over=False):
        """Append one evaluation, as one CSV row and one term record.

        A metric the evaluation did not report is written as an empty cell rather than raising, so
        a partially instrumented run still records its structures.  ``suggestion`` is the
        ``Suggestion`` that produced a pass and carries the acquisition value and the fallback
        state; a term of the design has none, and its cells stay empty.  ``loop_value`` is the
        value the loop was handed for this evaluation, kept in the term record.  ``taken_over``
        marks a row whose value came from an earlier run's record, in the schema's column of it and
        in the term record.
        """
        structure = tree.interpret(self._pretty_algebra())
        self._writer.writerow(self._schema.row(
            phase=phase,
            index=index,
            structure=structure,
            term_size=tree.size,
            metrics=metrics,
            timestamp=time.time(),
            loop_value=loop_value,
            suggestion=suggestion,
            acquisition_seconds=acquisition_seconds,
            taken_over=taken_over,
        ))
        self._file.flush()  # persist at once, so a crash mid-run loses no completed row
        self._terms.write(phase, index, tree, metrics, loop_value=loop_value,
                          taken_over=taken_over)

    def close(self):
        try:
            self._terms.close()
        finally:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
