"""The machinery of a run that no search space, dataset or training framework owns.

The program a run searches and its sampler, the acquisition optimizer, the step budgets, the term
pool, the per-pass records, the acceptance checks and the resumption of an initial design.  None of
it imports a training framework, so a run over any space can use it.
"""

from .acquisition import (
    DEFAULT_CROSSOVER_RATE,
    DEFAULT_MUTATION_RATE,
    DEFAULT_SELECTION_PRESSURE,
    build_acquisition_optimizer,
)
from .artifacts import metadata_path_for
from .budgets import step_budget
from .records import (
    EA_CSV_COLUMNS,
    SURROGATE_CSV_COLUMNS,
    EAGenerationLogger,
    SurrogateLogger,
    kernel_hyperparameters,
)
from .resume import load_initial_design
from .run_diagnostics import write_run_diagnostics
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
    "DEFAULT_CROSSOVER_RATE",
    "DEFAULT_DEPTH_BOUND",
    "DEFAULT_MUTATION_RATE",
    "DEFAULT_SELECTION_PRESSURE",
    "DEFAULT_SIZE_BOUND",
    "DETERMINIZATION_STATE_LIMIT",
    "DETERMINIZATION_WARN_SECONDS",
    "EA_CSV_COLUMNS",
    "FORMAT",
    "POOL_PHASE",
    "SPACE_CONSTRUCTION_WARN_SECONDS",
    "SURROGATE_CSV_COLUMNS",
    "VERSION",
    "DeterminizedSizeUniformSampler",
    "EAGenerationLogger",
    "SearchProgram",
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
    "step_budget",
    "write_run_diagnostics",
]
