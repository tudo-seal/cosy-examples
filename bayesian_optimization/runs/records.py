"""The per-pass records of a run: the inner search's generations and the surrogate's reads."""

import csv
import json

import numpy as np

from bayesian_optimization.diagnostics import read_calibration, read_fit

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
    "n_train",                        # how many pairs the pass conditioned on
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
        calibration = read_calibration(surrogate)

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
        conditioned, held_out = terms[0::2], terms[1::2]
        fit = None
        readable = (
            len(held_out) >= 2
            and conditioned
            and len(set(conditioned)) == len(conditioned)
            and not set(conditioned) & set(held_out)
        )
        if readable:
            fit = read_fit(
                optimizer.surrogate_over(conditioned, values[0::2]), held_out, values[1::2]
            )

        self._writer.writerow([
            bo_iteration,
            len(surrogate.X_train_),
            surrogate.log_marginal_likelihood_value_,
            calibration.root_mean_square,
            calibration.standard_deviation,
            calibration.maximum_absolute,
            calibration.outside_two,
            "" if fit is None else fit.size,
            "" if fit is None or fit.rank_correlation is None else fit.rank_correlation,
            "" if fit is None else fit.prediction_spread,
            "" if fit is None else fit.objective_spread,
            "" if fit is None else fit.residual_root_mean_square,
            json.dumps(kernel_hyperparameters(surrogate), sort_keys=True),
        ])
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
