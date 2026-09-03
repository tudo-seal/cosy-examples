# Contributing

## Setup

Install the package with its test extra into a virtual environment:

```
python -m pip install -e ".[test]"
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

## Measuring coverage

```
pytest --cov=bayesian_optimization
```

Name the package. A bare `--cov` reports the test files next to the code under
test, and it lists only the modules that some test happened to import, so a
module no test touches drops out of the table instead of showing up at zero
percent. `coverage run --source=bayesian_optimization -m pytest` followed by
`coverage report` produces the same table for anyone who prefers the coverage
command line.

## Checks

```
ruff check .
mypy bayesian_optimization
```

Both read their configuration from `pyproject.toml`.
