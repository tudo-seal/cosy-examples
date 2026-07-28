# cosy-examples

Bayesian optimization over type-inhabitation search spaces, built on the `cosy` framework, plus worked
examples from neural architecture search. Everything that needs dependencies lives here — `cosy` itself
stays dependency-free, which is why the Gaussian-process part cannot live there.

## Two layers, two different standards

**`bayesian_optimization/*.py` is a general, reusable BO core.** It combines cosy's resolution-based
enumeration and evolutionary algorithms with scikit-learn Gaussian processes. Future use cases must be able
to use this API **without prior knowledge and without unwritten invariants**.

**`bayesian_optimization/examples/**` are applications only.** An example must never paper over a defect in
the core or in cosy by handling the API cleverly. When a bug surfaces in an example, fix it at its root —
in the core, or in cosy — and remove the workaround here.

The sister repository `cosy` is at `~/PycharmProjects/cosy`, branch `felix_diss`, and is wired into this
session via `permissions.additionalDirectories`.

## Development setup

Use the venv for everything: `venv/bin/python`, `venv/bin/pytest`.

`cosy` must be installed **editable**, otherwise cosy fixes have no effect on anything run here:

```
venv/bin/pip install -e ~/PycharmProjects/cosy --no-deps
venv/bin/python -c "import cosy; print(cosy.__file__)"   # must point into the cosy source tree
```

`--no-deps` is mandatory (it protects the pinned numpy/scipy/torch). The git URL in `requirements.txt` and
`pyproject.toml` stays as it is — server, CI and fresh clones build from it.

## Rules that hold everywhere in this repo

- **`damg_nas/` is legacy** — both the one in the repo root and `bayesian_optimization/examples/damg_nas/`.
  Never modify them, never read them as a reference. The CNN work happens only in
  `bayesian_optimization/examples/cnn_damg_nas/`.
- **No default values for failures.** A network that fails to train or a term that fails to interpret does
  not get a substitute value; the failure stays visible. No `try/except` that swallows, no sentinel that
  silently replaces a measurement.
- **Steer search variance through the query type (the target), never by editing the repository.** The
  repository defines what is expressible, the target defines what is asked for. Mixing the two makes the
  repository experiment-specific and destroys comparability between experiments.
- **`mutation_rate` stays 0** in the acquisition-optimizing EA — mutation is currently defective in cosy.
  `population_size=100` is the deliberate compensation, so recombination is the only source of variation.

## Current work

Phase 1 fixes the legacy bugs in this repo and in cosy; phase 2 aligns both with the dissertation. The
plan, the constraints and the verified findings are in the auto-memory (`phase1-plan-status`,
`phase1-legacy-bugfix-scope`) and in `~/.claude/plans/tidy-riding-stonebraker.md`.

`CNN_REFACTORING_PLAN.md` and `cosy-improvements.txt` in the repo root are **superseded** — their line
numbers refer to an older installed cosy and mislead. `COSY_REWORK.md` remains valid as the phase-2 backlog.
