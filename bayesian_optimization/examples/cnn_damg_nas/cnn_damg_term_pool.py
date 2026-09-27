"""The term pool of a run, which lives in :mod:`bayesian_optimization.runs.term_pool`.

Every name this module defined is imported from there, so that its importers keep working and so
that a pickle written before the move still loads.  Such a pickle names
``bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_pool.TermRecord``, and that name
resolves here to the very class object the pool module defines.

``FORMAT`` is the tag a pool is written with now, ``term_pool``; the tag this module wrote,
``cnn_damg_term_pool``, is in ``LEGACY_FORMATS`` and still read.
"""

# The module's own imports as well, so that every name it offered is still offered.
from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any

from bayesian_optimization.runs.term_pool import (
    FORMAT,
    LEGACY_FORMATS,
    POOL_PHASE,
    VERSION,
    TermPoolWriter,
    TermRecord,
    TruncatedTermPool,
    as_pool,
    read_term_pool,
    resume_term_pool,
)

__all__ = [
    "FORMAT",
    "LEGACY_FORMATS",
    "POOL_PHASE",
    "VERSION",
    "TermPoolWriter",
    "TermRecord",
    "TruncatedTermPool",
    "as_pool",
    "read_term_pool",
    "resume_term_pool",
]
