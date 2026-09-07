# Contributing

## Setup

Install the package with its dev extra into a virtual environment. The dev
extra carries the two checkers as well as the test tools, so one install
covers
everything below:

```
python -m pip install -e ".[dev]"
```

The `cosy` library comes along as the declared `combinatory-synthesizer`
dependency, and the tests import it directly. `pyproject.toml` puts the
repository root on `sys.path`, so `bayesian_optimization` is taken from the
checkout and not from an installed copy.

## Running the tests

```
pytest
```

The end-to-end tests of the optimization loop carry the `slow` marker, which
`pyproject.toml` declares along with `integration` and `property`. Skip them
for a quick run:

```
pytest -m "not slow"
```

Five of the 891 tests carry `slow`, and the other 886 finish in about 90
seconds in one process on the machine this was measured on.

## Measuring coverage

```
pytest --cov=bayesian_optimization
```

Name the package. A bare `--cov` reports the test files next to the code
under
test, and it lists only the modules that some test happened to import, so a
module no test touches drops out of the table instead of showing up at zero
percent. `coverage run --source=bayesian_optimization -m pytest` followed by
`coverage report` produces the same table for anyone who prefers the
coverage
command line.

## Checks

```
ruff check .
mypy bayesian_optimization tests
```

Both come with the `dev` extra and both read their configuration from
`pyproject.toml`. `mypy` covers 105 files over those two paths and reports no
error, only its twenty standing notes about unchecked bodies of untyped
functions, so a new error is one this change introduced.

`.github/workflows/checks.yml` runs these two and the fast half of the suite on
every push and every pull request, on both supported Python versions. It runs
the same commands, so a red run there fails for the same reason it fails here.
The slow tests are the end-to-end runs of the loop and are left to a local
`pytest` before a change goes out.

## Two layers, held to two standards

`bayesian_optimization/` outside `examples/` is a reusable core: 19 modules,
5670 lines, and 35 exported names. A future caller has to be able to use
that
API without prior knowledge and without unwritten invariants, so a change
there
carries its reason in the code rather than in a commit message.

`bayesian_optimization/examples/` is applications only: 24 modules, 13 418
lines. The dependency runs one way, and no module of
`bayesian_optimization/`
outside `examples/` imports anything from inside it. The tests do, 32 of
them. An example may never paper over a defect in the core
or in `cosy` by handling the API cleverly. When a bug surfaces in an
example,
fix it at its root and take the workaround out.

Inside an example, the repository says what is expressible and the target
says
what is asked for. Steering the variance of a search belongs in the target,
which is why `cnn_damg_targets.py` is a module of its own beside
`cnn_damg_repo.py`. Editing the repository per experiment makes it
experiment-specific and ends comparability between experiments.

## No substitute value for a failure

A failure is a failure, not a number. Nothing here catches an exception to
replace a measurement with a default.

`BayesianOptimization.observe` refuses an objective value that is not finite
and lets the failure propagate out of the pass that caused it, because a
value
that is not finite makes the posterior undefined at every term rather than
at
one. The candidate evaluation of the CNN experiment records a diverged
training
as diverged, with the epoch it stopped at, and reports the number it reached
as
measured. A quantity that was not measured stays absent: the standard
deviation
over repetitions is `None` for a single repetition rather than `0.0`, since
reporting zero would claim a spread was measured to be zero when none was
measured at all. The one `except Exception` in the package re-raises as
`TruncatedTermPool` with the records that decoded whole, so a caller who
wants
the salvage has to ask for it.

The same rule holds for the diagnostics. Every read raises on input it
cannot
read rather than returning a number that looks like a reading.

## `damg_nas` is the legacy precursor

`bayesian_optimization/examples/damg_nas/` came first and
`bayesian_optimization/examples/cnn_damg_nas/` replaced it. The two model
networks as string diagrams over different alphabets and inherit the same
four
swap laws from `bayesian_optimization/examples/swap_laws.py`, which states
them
as term predicates for both. `recognizable_swap_laws.py` beside it states
the
same four as an abstraction with a relation, which is the form the
determinization compiles into the non-terminals, and those two files are the
only places the laws are written down. Nothing in `cnn_damg_nas/` imports
anything from
`damg_nas/`.

New work goes into `cnn_damg_nas/`. `damg_nas/` and `simple_nas/` are kept
in
working order rather than modernized past recognition: both run against
today's
`cosy`, both are under test, and both take the settings a run costs as
arguments
rather than as module constants. What is legacy about them is the
evolutionary
search they were written against. Both selected survivors by age, which
`cosy`
no longer offers, and the DAMG example ran at a mutation rate of 0.00, which
is
on the wrong side of the condition the convergence of the search needs. Both
now
use `GenerousConservativeReplacement` and rates the conditions admit. A
change
there keeps them running, it does not turn them into a second CNN search.

## Why the Gaussian process half is here and not in `cosy`

`cosy` declares an empty dependency list, and that is a property to
preserve. A
Gaussian process surrogate needs scikit-learn, numpy and scipy, the graph
kernels need grakel and networkx, and the architecture searches need torch
and
torchvision. All of those are declared in this repository's
`pyproject.toml`,
none of them in cosy's.

So the split is not by topic. Everything that needs a runtime dependency
lives
here, whatever else it is, and cosy keeps the enumeration, the samplers and
the
evolutionary components that need none.
