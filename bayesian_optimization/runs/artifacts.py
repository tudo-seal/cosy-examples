"""Where the artifacts of a run go: beside its CSV, named after it."""

import os


def metadata_path_for(csv_path):
    """Return the provenance path beside a run's CSV: ``run.csv`` becomes ``run_config.json``."""
    base, _ext = os.path.splitext(csv_path)
    return f"{base}_config.json"


def _sibling_path(csv_path, suffix):
    """Return a path beside a run's CSV: ``run.csv`` becomes ``run<suffix>``."""
    base, _ext = os.path.splitext(csv_path)
    return f"{base}{suffix}"
