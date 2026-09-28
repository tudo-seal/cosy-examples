"""The core and the run layer import without the examples' dependencies.

The checks' job without torch installs no extra, and ``tests/conftest.py`` leaves out there every
test module that needs a missing dependency -- also one that needs it only because a module of the
core or the run layer came to import it, which would keep that job green over exactly the defect it
is there to catch. This test imports every module outside ``examples/`` itself and imports nothing
of this repository at its own top level, so it is never left out: in that job, a module outside the
examples that needs torch fails it.
"""

from __future__ import annotations

import importlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "bayesian_optimization"


def _modules_outside_the_examples() -> list[str]:
    names = []
    for path in sorted(PACKAGE.rglob("*.py")):
        parts = list(path.relative_to(PACKAGE.parent).with_suffix("").parts)
        if "examples" in parts:
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


def test_every_module_outside_the_examples_imports_without_their_dependencies():
    names = _modules_outside_the_examples()
    # the input set, counted: the core's 21 modules and the run layer's 12, and nothing of examples/
    assert len(names) >= 33, names
    assert {"bayesian_optimization", "bayesian_optimization.runs.driver"} <= set(names)
    assert not any(".examples" in name for name in names)

    for name in names:
        importlib.import_module(name)
