"""The machinery of a run that no search space, dataset or training framework owns.

The program a run searches and its sampler, the acquisition optimizer, the step budgets, the term
pool, the records of a run under a caller's metric schema, the files a run writes, its neutral
provenance, the per-pass records, the acceptance checks and the resumption of an initial design.  None of
it imports a training framework, so a run over any space can use it.
"""

from .acquisition import (
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_MUTATION_RATE,
    DEFAULT_SELECTION_PRESSURE,
    build_acquisition_optimizer,
)
from .artifacts import RunArtifacts, metadata_path_for
from .budgets import StepBudgets, step_budget
from .driver import DESIGN_PHASE, RunOutcome, run_paired, run_search
from .metadata import write_run_metadata
from .records import (
    EA_CSV_COLUMNS,
    SURROGATE_CSV_COLUMNS,
    EAGenerationLogger,
    EvaluationRecorder,
    SurrogateLogger,
    kernel_hyperparameters,
)
from .resume import load_initial_design
from .run_diagnostics import write_run_diagnostics
from .schema import LOOP_FIELDS, Column, MetricSchema, Objective
from .search_program import (
    DEFAULT_DEPTH_BOUND,
    DEFAULT_SIZE_BOUND,
    DETERMINIZATION_STATE_LIMIT,
    DETERMINIZATION_WARN_SECONDS,
    SPACE_CONSTRUCTION_WARN_SECONDS,
    DeterminizedSizeUniformSampler,
    SearchProgram,
    build_search,
    describe_sampler,
    describe_search,
)
from .term_pool import (
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
    "DESIGN_PHASE",
    "DEFAULT_CROSSOVER_RATE",
    "DEFAULT_DEPTH_BOUND",
    "DEFAULT_MUTATION_RATE",
    "DEFAULT_SELECTION_PRESSURE",
    "DEFAULT_SIZE_BOUND",
    "DETERMINIZATION_STATE_LIMIT",
    "DETERMINIZATION_WARN_SECONDS",
    "EA_CSV_COLUMNS",
    "FORMAT",
    "LEGACY_FORMATS",
    "LOOP_FIELDS",
    "POOL_PHASE",
    "SPACE_CONSTRUCTION_WARN_SECONDS",
    "SURROGATE_CSV_COLUMNS",
    "VERSION",
    "Column",
    "DeterminizedSizeUniformSampler",
    "EAGenerationLogger",
    "EvaluationRecorder",
    "MetricSchema",
    "Objective",
    "RunArtifacts",
    "RunOutcome",
    "SearchProgram",
    "StepBudgets",
    "SurrogateLogger",
    "TermPoolWriter",
    "TermRecord",
    "TruncatedTermPool",
    "as_pool",
    "build_acquisition_optimizer",
    "build_search",
    "describe_sampler",
    "describe_search",
    "kernel_hyperparameters",
    "load_initial_design",
    "metadata_path_for",
    "read_term_pool",
    "resume_term_pool",
    "run_paired",
    "run_search",
    "step_budget",
    "write_run_diagnostics",
    "write_run_metadata",
]
