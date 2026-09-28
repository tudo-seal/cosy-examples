"""The acceptance checks over a finished run, written beside its CSV."""

import csv
import json

from sklearn.gaussian_process import GaussianProcessRegressor

from bayesian_optimization.diagnostics import (
    read_calibration,
    read_fit,
    read_frontier,
    read_gram,
    read_trace,
    trace_columns,
    trace_rows,
)
from bayesian_optimization.runs.artifacts import _sibling_path


def write_run_diagnostics(csv_path, optimizer, result):
    """Run the acceptance checks over the finished run and write them out.

    A run that produced numbers has not thereby produced readable numbers, and these reads are what
    separates the two.  They are ordered by what they need, the kernel first, then the mean, then
    the uncertainty, then the inner evolutionary run, and a failure found early makes the later ones
    uninformative rather than wrong: an uninformative kernel gives a flat fit scatter, flat
    residuals and a flat acquisition landscape, and only the first of them says why.

    Every field here is a measurement and none is a verdict.  Each read names a failure mode without
    the threshold that separates it, because a threshold belongs to a search space and not to the
    method, and eleven terms of one chain and four hundred convolutional architectures do not share
    one.  So this writes the numbers and leaves the reading to whoever compares two runs.

    Two artifacts: ``<run>_diagnostics.json`` with the five reads, and ``<run>_trace.csv`` with the
    per-pass table behind the fifth.

    Called after the CSV and the metadata are on disk, so that a read which raises, a surrogate that
    will not fit for instance, costs the diagnostics and not the run.

    Each read states what it needs, and a read whose input is not there is written as ``null``.
    That is not a substitute value and not a swallowed failure: a smoke test of three evaluations
    has one held-out term, and the fit read is a statement about a scatter that needs two, so "there
    was not enough of a run to read this" is the true answer and it is what the file says.  A read
    that fails on data it was given still raises.

    Args:
        csv_path (str): The run's CSV.  The artifacts are written beside it.
        optimizer (BayesianOptimization): The loop, after ``finalize()``.
        result (dict): What ``finalize()`` returned.

    Returns:
        dict: The diagnostics, as they were written.
    """
    terms = list(result["x"])
    values = [float(value) for value in result["y"]]

    # --- 1. The kernel, before any conditioning ----------------------------------------------
    # ``optimizer.kernel`` and not the fitted one.  A kernel matrix betrays a broken kernel before
    # any Gaussian process is conditioned, and that reading is about the kernel as it was
    # constructed, before a fit chose its scales.
    # The constructor's kernel is the surrogate's only where the surrogate is the Gaussian process;
    # beside a caller's surrogate it conditioned nothing, and its matrix would describe no model.
    with_the_kernel = getattr(optimizer, "surrogate_model", None) is None
    gram = (
        read_gram(optimizer.kernel(terms), objective=values)
        if len(terms) >= 2 and with_the_kernel
        else None
    )

    # --- 2. The mean, on terms the surrogate has not seen -------------------------------------
    # Condition on the terms of even index and predict the odd ones.  Splitting by parity keeps the
    # conditioning half spread over the whole run instead of over one region of it, which would
    # measure extrapolation, a different and harder question.
    conditioned_terms, conditioned_values = terms[0::2], values[0::2]
    held_out_terms, held_out_values = terms[1::2], values[1::2]
    # The same three conditions ``SurrogateLogger`` states: two held-out terms to make a scatter, a
    # conditioning half without repeats, and no term on both sides.  A surrogate that has seen what
    # it predicts reproduces it, and the read would report a diagonal it did not earn.
    readable = (
        len(held_out_terms) >= 2
        and conditioned_terms
        and len(set(conditioned_terms)) == len(conditioned_terms)
        and not set(conditioned_terms) & set(held_out_terms)
    )
    fit = (
        read_fit(
            optimizer.surrogate_over(conditioned_terms, conditioned_values),
            held_out_terms,
            held_out_values,
        )
        if readable
        else None
    )

    # --- 3. The uncertainty, over the whole dataset -------------------------------------------
    # Not ``result["gp_model"]``, which is the surrogate of the last ``suggest()`` and never saw the
    # evaluation the run ended on.  A calibration is a statement about the data that was collected.
    # Only a Gaussian process has the factor and the weights the read takes apart; a caller's
    # surrogate has none, and the read is null.
    whole = optimizer.surrogate_over_dataset() if terms and with_the_kernel else None
    calibration = (
        read_calibration(whole) if isinstance(whole, GaussianProcessRegressor) else None
    )

    # --- 4. The inner evolutionary run --------------------------------------------------------
    run = optimizer.last_acquisition_run
    frontier = None if run is None else read_frontier(run)

    # --- 5. The loop itself -------------------------------------------------------------------
    trace = read_trace(result["trace"]) if result["trace"] else None

    diagnostics = {
        # null where the run was too short for the read, never a filled-in number.
        "gram": None if gram is None else {
            "size": gram.size,
            "symmetry_error": gram.symmetry_error,
            "minimum_eigenvalue": gram.minimum_eigenvalue,
            "condition_number": gram.condition_number,
            "diagonal_spread": gram.diagonal_spread,
            "off_diagonal_mean": gram.off_diagonal_mean,
            "off_diagonal_spread": gram.off_diagonal_spread,
            "seriation_neighbour_similarity": gram.seriation_neighbour_similarity,
            "coordinate_spread": gram.coordinate_spread,
            # The kernel's own coordinate against the objective.  A low value here is a statement
            # about this kernel on these terms and not a defect by itself, since the leading
            # principal component of a kernel need not be the direction the objective varies along.
            "objective_alignment": gram.objective_alignment,
        },
        "fit": None if fit is None else {
            "size": fit.size,
            "rank_correlation": fit.rank_correlation,
            "predictions_constant": fit.predictions_constant,
            "prediction_spread": fit.prediction_spread,
            "objective_spread": fit.objective_spread,
            "regression_slope": fit.regression_slope,
            "residual_root_mean_square": fit.residual_root_mean_square,
            "true_values": list(fit.true_values),
            "predicted_values": list(fit.predicted_values),
        },
        "calibration": None if calibration is None else {
            "size": calibration.size,
            "mean_absolute": calibration.mean_absolute,
            "maximum_absolute": calibration.maximum_absolute,
            # The spread about zero and the spread about their own mean, side by side.  The
            # difference between the two is bias, and residuals that carry a single sign with a
            # small spread are the signature of a surrogate whose deviations stayed at the prior.
            # Reading only one of the two hides it.
            "root_mean_square": calibration.root_mean_square,
            "standard_deviation": calibration.standard_deviation,
            "outside_two": calibration.outside_two,
            "positive_fraction": calibration.positive_fraction,
            "log_marginal_likelihood": calibration.log_marginal_likelihood,
            "targets_normalized": calibration.targets_normalized,
            "standardized_residuals": list(calibration.standardized_residuals),
        },
        # None rather than an empty record.  That no acquisition maximization was recorded is a
        # different statement from one that was recorded and held nothing.
        "frontier": None if frontier is None else {
            "size": frontier.size,
            "distinct_members": frontier.distinct_members,
            "pick_in_population": frontier.pick_in_population,
            "known_members": frontier.known_members,
            "mean_at_pick": frontier.mean_at_pick,
            "deviation_at_pick": frontier.deviation_at_pick,
            "score_at_pick": frontier.score_at_pick,
            "fallback_used": frontier.fallback_used,
            "dominating_members": frontier.dominating_members,
            "on_frontier": frontier.on_frontier,
            "higher_scored_members": frontier.higher_scored_members,
            "mean_spread": frontier.mean_spread,
            "deviation_spread": frontier.deviation_spread,
        },
        "trace": None if trace is None else {
            "passes": trace.passes,
            "acquisition_trend": trace.acquisition_trend,
            "acquisition_first": trace.acquisition_first,
            "acquisition_last": trace.acquisition_last,
            "acquisition_at_zero": trace.acquisition_at_zero,
            "deviation_spread": trace.deviation_spread,
            "deviation_minimum": trace.deviation_minimum,
            "deviation_maximum": trace.deviation_maximum,
            "fallbacks": trace.fallbacks,
            "distinct_fraction": trace.distinct_fraction,
            "improvements": trace.improvements,
            "stalled_passes": trace.stalled_passes,
            "best_trace": list(trace.best_trace),
        },
        # The reads carry no thresholds, and neither does this file.  What the fields mean is in
        # ``bayesian_optimization/diagnostics``, one remark per read.
        "read_this_with": "bayesian_optimization.diagnostics",
    }

    path = _sibling_path(csv_path, "_diagnostics.json")
    with open(path, "w") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True, default=float)
    print(f"Acceptance checks written to {path}", flush=True)

    if result["trace"]:
        trace_path = _sibling_path(csv_path, "_trace.csv")
        with open(trace_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(trace_columns())
            writer.writerows(trace_rows(result["trace"]))
        print(f"Run trace written to {trace_path}", flush=True)

    return diagnostics
