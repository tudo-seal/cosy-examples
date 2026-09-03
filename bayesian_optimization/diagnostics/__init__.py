"""The four acceptance checks of the Bayesian optimization layer, and the run log beneath them.

Every diagnostic picture serves twice over: the picture that explains what the model believes is
also the instrument that shows the belief is broken.  Each of the four reads here measures one
such picture, and together they accept the layer:

========================  ==========================  ============================================
Read                      The picture                 What it catches
========================  ==========================  ============================================
:func:`read_gram`         the kernel matrix           the kernel discriminates nothing, or
                                                      transfers nothing
:func:`read_fit`          the fit scatter             the posterior mean is wrong
:func:`read_calibration`  standardized residuals      the mean is right and the uncertainty is not
:func:`read_frontier`     the mean-deviation plane    the inner evolutionary run fails
:func:`read_trace`        the run trace               the loop degenerates
========================  ==========================  ============================================

They are ordered by what they need: the first needs a kernel and terms, the second and third a
fitted surrogate, the fourth a finished acquisition maximization, the fifth a finished run.  A
failure found early makes the later reads unreadable rather than informative.  An uninformative
kernel produces a flat fit scatter, flat residuals and a flat acquisition landscape, and only the
first of the four says why.

Every read returns measurements and plot data, and none of them returns a verdict.  The failure
modes are named above, the thresholds that separate them are not, because a threshold belongs to a
search space rather than to the method.
"""
from __future__ import annotations

from ._statistics import rank_correlation, spread
from .calibration import CalibrationRead, read_calibration
from .fit import FitRead, kernel_objective_alignment, read_fit
from .frontier import AcquisitionRun, FrontierRead, GenerationRecord, read_frontier
from .gram import GramRead, read_gram
from .run_log import (
    enable_verbose_logging,
    get_logger,
    log_iteration,
    log_suggestion,
    warn_if_exploitation_stalls,
)
from .trace import TraceRead, TraceRecord, read_trace, trace_columns, trace_rows

__all__ = [
    "AcquisitionRun",
    "CalibrationRead",
    "FitRead",
    "FrontierRead",
    "GenerationRecord",
    "GramRead",
    "TraceRead",
    "TraceRecord",
    "enable_verbose_logging",
    "get_logger",
    "kernel_objective_alignment",
    "log_iteration",
    "log_suggestion",
    "rank_correlation",
    "read_calibration",
    "read_fit",
    "read_frontier",
    "read_gram",
    "read_trace",
    "spread",
    "trace_columns",
    "trace_rows",
    "warn_if_exploitation_stalls",
]
