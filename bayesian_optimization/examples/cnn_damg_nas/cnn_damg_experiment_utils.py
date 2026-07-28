"""Shared experiment-result persistence for cnn_damg_nas experiments.

Two artifacts per run:

* ``<run>.csv`` - one row per objective-function evaluation: phase (pre_sample vs. bo_step), index
  within that phase, the pretty-printed structure (no term pickling needed), and the measured
  metrics. Flushed to disk immediately after each row, so an interrupted run keeps its results.
* ``<run>_config.json`` - the run's provenance: dataset, full search-space configuration, BO
  parameters, device, timings, library versions. Without it the CSVs of two runs are
  indistinguishable except by filename.

Training is non-deterministic (random init, mini-batch shuffling) and no seed is fixed, so the
values recorded here are the reference to compare against when a structure is retrained later -
expect small deviations on re-evaluation.
"""

import csv
import json
import os
import time

import torch

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo_algebras import (
    pytorch_components_algebra,
    learner as raw_learner,
)

CSV_COLUMNS = [
    "phase",            # "pre_sample" | "bo_step"
    "index",            # position within the initial sample, or 0-indexed BO iteration
    "structure",        # pretty-printed term (architecture + loss + optimizer + epochs)
    "objective_value",  # raw test loss, exactly what the objective function returned
    "accuracy",         # test accuracy of the same trained model (loss and accuracy do not
                        # correlate 1:1 - a lower loss can come with a lower accuracy)
    "n_params",         # trainable parameter count, to see whether BO drifts towards large nets
    "train_seconds",    # wall-clock training time, the basis for the CPU vs. A30 comparison
    # The three columns below decide how the row may be read at all. When `suggest()` cannot find a
    # novel candidate it replaces the optimizer's result with a random fallback sample, and
    # `acquisition_value` then describes that replacement. A run whose bo_step rows all say
    # fallback_used=True was random search; without these columns that is indistinguishable from BO
    # in the finished file. Empty for pre_sample rows, which have no acquisition step.
    "acquisition_value",
    "fallback_used",
    "fallback_attempts",
    "timestamp",
]


def evaluate_candidate(tree, x, y, x_test, y_test, batch_size):
    """Train one candidate and return all recorded metrics.

    Uses `pytorch_components_algebra` rather than `pytorch_function_algebra` so the trained model
    stays accessible for the accuracy/parameter measurements; the training itself goes through the
    very same `learner` routine the function algebra would have used, so the objective value is
    directly comparable to runs that used it.
    """
    model, loss_fn, optimizer_factory, epochs = tree.interpret(pytorch_components_algebra())
    n_params = sum(p.numel() for p in model.parameters())
    input_features = x.shape[-1]

    started = time.time()
    objective_value = raw_learner(input_features, model, loss_fn, optimizer_factory, epochs,
                                  x, y, x_test, y_test, batch_size=batch_size)
    train_seconds = time.time() - started

    # `learner` already moved the model onto the data's device and trained it in place.
    with torch.inference_mode():
        predictions = model(x_test).argmax(dim=-1)
        accuracy = (predictions == y_test).float().mean().item()

    return {
        "objective_value": objective_value,
        "accuracy": accuracy,
        "n_params": n_params,
        "train_seconds": train_seconds,
    }


class ExperimentCSVLogger:
    def __init__(self, path, pretty_algebra):
        self._pretty_algebra = pretty_algebra
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_COLUMNS)
        self._file.flush()

    def log(self, phase, index, tree, metrics, suggestion=None):
        """Append one evaluation. `metrics` is an `evaluate_candidate` result dict; missing keys are
        written as empty cells rather than failing, so a partially instrumented run still records
        its structures and objective values.

        `suggestion` is the `Suggestion` that produced this candidate, and carries the acquisition
        value and the fallback state. Pre-sample rows have none, and their cells stay empty: an
        empty cell says "there was no acquisition step here", whereas writing False would claim a
        fallback was ruled out that was never evaluated.
        """
        structure = tree.interpret(self._pretty_algebra())
        diagnostics = (suggestion.diagnostics or {}) if suggestion is not None else {}
        self._writer.writerow([
            phase,
            index,
            structure,
            metrics.get("objective_value", ""),
            metrics.get("accuracy", ""),
            metrics.get("n_params", ""),
            metrics.get("train_seconds", ""),
            "" if suggestion is None else suggestion.acquisition_value,
            diagnostics.get("fallback_used", ""),
            diagnostics.get("fallback_attempts", ""),
            time.time(),
        ])
        self._file.flush()  # persist immediately - a crash mid-run must not lose completed rows

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def metadata_path_for(csv_path):
    """`results/run.csv` -> `results/run_config.json`."""
    base, _ext = os.path.splitext(csv_path)
    return f"{base}_config.json"


def write_run_metadata(csv_path, metadata):
    """Write the run's provenance next to its CSV and return the path used."""
    path = metadata_path_for(csv_path)
    enriched = {
        **metadata,
        "csv_path": csv_path,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "written_at": time.time(),
    }
    with open(path, "w") as f:
        json.dump(enriched, f, indent=2, sort_keys=True)
    return path
