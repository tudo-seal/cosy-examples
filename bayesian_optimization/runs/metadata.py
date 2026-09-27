"""A run's provenance record, written beside its CSV, with the environment its caller names."""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Mapping
from importlib import metadata as _distributions
from typing import Any

from .artifacts import metadata_path_for

#: The distributions whose versions every record carries: the two frameworks every run uses.
_FRAMEWORKS = ("combinatory-synthesizer", "cosy-examples")


def _version(distribution: str) -> str | None:
    try:
        return _distributions.version(distribution)
    except _distributions.PackageNotFoundError:
        return None


def write_run_metadata(
    csv_path: str,
    metadata: Mapping[str, Any],
    *,
    environment: Mapping[str, Any] | None = None,
) -> str:
    """Write the run's provenance next to its CSV and return the path used.

    The record is the caller's metadata, where the CSV is, the Python that ran the run, the
    versions of the two frameworks, and when it was written.  ``environment`` adds what the
    caller's own stack is, a deep-learning framework or an accelerator, at the record's top level.
    Nothing is written about a framework the caller did not name: a run that trained nothing with
    torch has no torch version in its record.

    Args:
        csv_path (str): The run's CSV; the record goes beside it as ``<run>_config.json``.
        metadata (Mapping[str, Any]): The caller's provenance.
        environment (Mapping[str, Any] | None): The caller's stack. (Default value = None)

    Returns:
        str: The path written.
    """
    path: str = metadata_path_for(csv_path)
    enriched = {
        **metadata,
        "csv_path": csv_path,
        "python_version": platform.python_version(),
        "package_versions": {name: _version(name) for name in _FRAMEWORKS},
        **(environment or {}),
        "written_at": time.time(),
    }
    with open(path, "w") as handle:
        json.dump(enriched, handle, indent=2, sort_keys=True)
    return path
