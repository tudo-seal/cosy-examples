"""Shared experiment-result persistence for cnn_damg_nas experiments.

Writes one CSV row per objective-function evaluation - phase (pre_sample vs. bo_step), index
within that phase, the pretty-printed structure (no term pickling needed), and the raw objective
value - flushed to disk immediately after each row. Training is non-deterministic (random init,
mini-batch shuffling), so the value recorded here is the reference to compare against when a
structure is retrained later.
"""

import csv
import time


class ExperimentCSVLogger:
    def __init__(self, path, pretty_algebra):
        self._pretty_algebra = pretty_algebra
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["phase", "index", "structure", "objective_value", "timestamp"])
        self._file.flush()

    def log(self, phase, index, tree, value):
        structure = tree.interpret(self._pretty_algebra())
        self._writer.writerow([phase, index, structure, value, time.time()])
        self._file.flush()  # persist immediately - a crash mid-run must not lose completed rows

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
