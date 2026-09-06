# cosy-examples

Bayesian optimization over CoSy solution spaces, and the neural architecture searches built on it.
A Gaussian process is fitted over the terms a synthesized search space derives, an evolutionary
search maximizes an acquisition score over that space, and the term it returns is the next
candidate the objective is spent on.

The `cosy` library itself declares no runtime dependency. This repository declares numpy, scipy,
scikit-learn, networkx, grakel, torch and torchvision, and pandas, matplotlib and tqdm beside
them, which is why the Gaussian process half of the method lives here and not there.

## Installation

Install the package with its dev extra into a virtual environment. The dev extra carries the two
checkers as well as the test tools, so one install covers everything the sections below run:

```
python -m pip install -e ".[dev]"
```

The `cosy` library comes along as the declared `combinatory-synthesizer` dependency. Python 3.11
or 3.12 is required, as `pyproject.toml` states.

The CNN examples read the CIFAR-10 archive from a directory the caller names. Nothing in this
repository downloads it unless a run is started with `--download`, and a missing archive stops a
run before it trains anything.

## Tests and checks

```
pytest
pytest -m "not slow"
ruff check .
mypy bayesian_optimization tests
```

The two checks need the `dev` extra, which carries `mypy` and `ruff`. The tests need `test`, and
`pip install -e ".[dev]"` covers both.

The suite holds 891 tests in 51 test modules. Five carry the `slow` marker, and the remaining 886
finish in about ninety seconds in one process on the machine this was measured on. `mypy` reports
nothing over the 105 files those two paths cover. `pyproject.toml` declares `slow`,
`integration` and `property`, and it puts the repository root on `sys.path`, so
`bayesian_optimization` is taken from the checkout rather than from an installed copy.
`CONTRIBUTING.md` covers coverage measurement and the conventions the two layers are held to.

## Quick start

The loop needs three things: a synthesized search space, the non-terminal to query it at, and an
evolutionary search that maximizes the acquisition score over that space. This example builds all
three over a three-level expression grammar, and it runs in a second or two on a laptop, imports
included.

```python
import random

from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.evolutionary_algorithms import (
    EvolutionarySearch,
    FitnessBasedReplacement,
    Generations,
    RankBasedSelection,
    ResolutionMutation,
    SampledInitialization,
    ScalarFitnessComparator,
    SubtreeSwap,
)
from cosy.search import SizeUniformSampler

from bayesian_optimization import BayesianOptimization

LEAF, INNER, ROOT = Constructor("E0"), Constructor("E1"), Constructor("E2")
MAX_SIZE = 7

repository = {
    "a": SpecificationBuilder().suffix(LEAF),
    "b": SpecificationBuilder().suffix(LEAF),
    "c": SpecificationBuilder().suffix(LEAF),
    "neg": SpecificationBuilder().argument("x", LEAF).suffix(INNER),
    "add": SpecificationBuilder().argument("l", LEAF).argument("r", LEAF).suffix(INNER),
    "top": SpecificationBuilder().argument("l", INNER).argument("r", INNER).suffix(ROOT),
}
space = Synthesizer(repository, {}).construct_solution_space(ROOT).prune()

WEIGHTS = {"top": 0.0, "add": 1.0, "neg": 2.0, "a": 0.5, "b": 1.5, "c": 2.5}


def objective(term):
    return float(WEIGHTS[term.root] + sum(objective(child) for child in term.children))


evolutionary = EvolutionarySearch(
    initializer=SampledInitialization(SizeUniformSampler(MAX_SIZE, random.Random(0))),
    mutation=ResolutionMutation(
        SizeUniformSampler(MAX_SIZE, random.Random(1)), random.Random(2)
    ),
    recombination=SubtreeSwap(random.Random(3), max_size=MAX_SIZE),
    parent_selection=RankBasedSelection(1.7, rng=random.Random(4)),
    survivor_selection=FitnessBasedReplacement(),
    termination=Generations(3),
    population_size=20,
    crossover_rate=0.9,
    mutation_rate=0.1,
    rng=random.Random(5),
    comparator=ScalarFitnessComparator(greater_is_better=True),
)

bo = BayesianOptimization(
    search_space=space,
    request=ROOT,
    optimizer=evolutionary,
    sampler=SizeUniformSampler(MAX_SIZE, random.Random(42)),
    seed=42,
)
result = bo.optimize(objective, budget=5, initial_size=8)
print(result["best_tree"], result["best_y"])
```

That run prints `top (add c c) (add c c) 12.0`. The loop maximizes throughout, so an objective that
is to be minimized is negated on the way in and the reported optimum on the way out.

## What the package holds

| Module | What it holds |
|---|---|
| `bo.py` | `BayesianOptimization`, the closed loop and the ask and tell engine under it |
| `acquisition_function.py` | the three scores, `ExpectedImprovement` and its two siblings |
| `acquisition_optimizer.py` | `AcquisitionOptimizer`, an acquisition as an evolutionary fitness |
| `initial_sampling.py` | `KernelDiverseInitializer` and the weight strategies it draws with |
| `kernels/` | the four structured kernels and their shared base class |
| `diagnostics/` | the five reads over a run, and the run log beneath them |
| `state.py` | `BOState`, `Suggestion`, `Diagnostics` |
| `utils.py` | the conversions from a term to a labeled graph the graph kernels read |

That is 19 modules of core, 5670 lines. `bayesian_optimization/examples/` holds 24 more modules and
13 418 lines, and every one of them is an application: an example may not work around a defect in
the core, it has to be fixed where it is. The dependency runs one way only, and no module outside
`examples/` imports anything from inside it.

`bayesian_optimization/__init__.py` exports 35 names, and nothing else is public API.

## The optimization loop

`optimize()` runs the loop closed. It takes the initial dataset from the initializer, then runs
`budget` passes of condition, maximize, evaluate, append, and answers with a term of maximal
observed value.

The ask and tell methods are the same loop turned inside out, for the case where an evaluation
lives outside the process, as a training run on another machine or as a measurement.

```python
bo.reset()
bo.initialize(objective=objective, initial_size=8)
for _ in range(5):
    suggestion = bo.suggest()
    bo.observe(suggestion.candidate, objective(suggestion.candidate))
result = bo.finalize()
```

The state runs `UNINITIALIZED`, `INITIALIZED`, then `SUGGESTED` and `OBSERVED` in alternation, and
`FINALIZED` at the end. Every method refuses a state it does not belong in with a `RuntimeError`
naming that state, and `reset()` returns the loop to `UNINITIALIZED`. `get_state_snapshot()`
reports the state, the observed terms and values, the pass count and any outstanding suggestion.

`finalize()` returns `best_tree`, `best_y`, `x`, `y`, `gp_model`, `iterations`, `trace` and
`dropped_suggestion`. The model it reports is the one the last `suggest()` fitted, so it has not
seen the pair the run ended on. A diagnostic that wants a surrogate over the whole dataset calls
`surrogate_over_dataset()` instead.

Objective values reach the Gaussian process as the caller supplied them. There is no transformation
layer, and no parameter selects one. `gp_normalize_y` is passed through to sklearn's
`GaussianProcessRegressor`, which centers and scales the targets for the duration of the fit and
undoes it on predict, so it changes no value the caller ever sees.

A failing evaluation is never a number. An objective that returns something not finite raises
where it was called, in the initial design as in a pass, rather than becoming an observation,
because a value that is not finite makes the posterior undefined at every term rather than at
one.

## Choosing a sampler

`sampler` is the one parameter that decides whether the loop runs on a real search space at all. It
is the loop's own source of terms, for the initial dataset and for the replacement drawn when the
acquisition maximization returns a term already evaluated, and those two cannot be configured
apart.
A space that admits one sampler and not another would otherwise get a working initializer and a
fallback that cannot run.

Passing `None` builds `SizeUniformSampler(100, Random(seed))` counting from a materialized search
tree. That is a placeholder for toy spaces, and it is the first thing to replace on a real one.

Three bounds are easy to confuse and are not the same quantity. The bound of `SizeUniformSampler`
is a term **size**, the number of symbols. The bound of `DepthBoundedRandomSampler` is a **depth**,
the longest path from the root to a leaf. Neither is cosy's engine parameter `max_depth`, which
bounds the open positions of a goal.

Size-uniform sampling counts, and what it can count is a property of the repository. Counting from
the program instead of from a materialized search tree, `counting="table"`, applies only where no
predicate of the repository reads a hole. `cnn_damg_repo.py` states four swap laws as predicates
over two sibling holes of one clause, so that form of the CNN repository does not qualify: at
structure length 2 with the USPS parameter sets, 8 of its clauses carry such a predicate, all of
them instances of `before_cons`. The form that does qualify is the recognizable one, under the CNN
search below.
Where neither counting construction is affordable, `DepthBoundedRandomSampler` is the answer,
because it never counts. That is the sampler the recorded run below drew from.

## The initial design

The default initial design is sampled initialization over the loop's sampler. With the size-uniform
sampler it stratifies along the one canonical axis a search space has, its term size.

`KernelDiverseInitializer` is the informed alternative. It draws one member at a time, each biased
away from the members already drawn, with the cost of a term being its summed kernel similarity to
the reference set. Neither design is universally better, so both are offered and the
model-agnostic one is the default.

Two properties of that initializer decide whether it is practical. Every draw recounts, because
each new member changes the cost function, and its costs are real numbers rather than integers, so
two terms whose scores agree mathematically may fall into two cost classes. An initializer handed
in carries its own random state and its own bound, and `sampler` and `seed` do not reach it.

## Kernels

Four structured kernels ship with the package.

| Kernel | Compares two terms by | Declared hyperparameters | `k(t, t)` |
|---|---|---|---|
| `OrderedRootedSubtreeKernel` | the complete ordered subtrees they share (default) | none | 1 |
| `SubsetTreeKernel` | the subset trees they share | none | 1, and 0 for a single node |
| `WeisfeilerLehmanKernel` | `h` rounds of relabeling on the term graph | none | 1 |
| `HierarchicalWLKernel` | the same, one level at a time | a weight per level | the weights summed |

The first three normalize their Gram matrix, so their similarity carries the shape of a term and
not its scale, and a kernel optimizer has nothing to move on them. `SubsetTreeKernel` scores a
single node at 0 rather than at 1, because a single node roots no subset tree and normalization
leaves that row and column alone instead of dividing by zero. `HierarchicalWLKernel` is the
exception on both counts: its weights are hyperparameters, and because it sums one normalized
kernel per level its diagonal is their sum, which model selection moves along with them.

`kernel_optimizer` is `None` by default for that reason. sklearn takes an optimizer for a kernel
with no hyperparameters and then skips its optimization step without a word, so the old default
`"fmin_l_bfgs_b"` asked for model selection on every run and got none of it on the kernel the
package ships with.

A caller who wants the scale fitted puts a `ConstantKernel` in front of the kernel and passes an
optimizer with it:

```python
from sklearn.gaussian_process.kernels import ConstantKernel

from bayesian_optimization import OrderedRootedSubtreeKernel

with_fitted_scale = BayesianOptimization(
    search_space=space,
    request=ROOT,
    optimizer=evolutionary,
    kernel=ConstantKernel(1.0) * OrderedRootedSubtreeKernel(),
    kernel_optimizer="fmin_l_bfgs_b",
)
```

What to watch once model selection is on. An initial design that lands on a plateau of the
objective leaves every observed value the same, and then the marginal likelihood has nothing to
explain: it drives the amplitude to the lower bound of the `ConstantKernel` instead of reading a
scale off the data. Measured on a chain of six terms with
`ConstantKernel(1.0, constant_value_bounds=(1e-4, 1e2))` in front of the default kernel: on an
objective that is the same value everywhere the fit lands on `0.01**2`, which is the lower bound,
and sklearn reports it as a `ConvergenceWarning` naming `constant_value`, while on an objective
that rises from 1.0 to 6.0 along the chain it lands on `1.66**2` and warns about nothing.
`HierarchicalWLKernel` needs no constant factor at all, since its weights are already the scale.

A `WhiteKernel` belongs in the sum when the quality measure is itself stochastic, a training run
for instance, and not otherwise. It is a noise level and not a numerical guard, and the loop's own
`_JITTER` of `1e-6` is the guard.

Read the standardized leave-one-out residuals (`read_calibration`) before trusting a fitted
kernel. A spread far above 1 is the overconfident surrogate the amplitude is meant to repair, and
a prediction that barely varies across candidates is the collapse above.

Migrating from the old default: a caller who passed no kernel, or one of the three without
hyperparameters, gets bit for bit the same fit as before, because the optimizer was never run for
them. A caller who passed a kernel that does declare hyperparameters and relied on the default to
fit them now has to pass `kernel_optimizer="fmin_l_bfgs_b"` as well. That case is not silent: a
kernel with a non-empty `theta` and no optimizer is reported once per run, as is an optimizer over
a kernel with an empty one.

## Acquisition functions

Three acquisition scores are available, named as strings on the constructor. Each is maximized over
the search space by the evolutionary search.

| `acquisition_function` | Score | Parameter |
|---|---|---|
| `"ExpectedImprovement"` | the expected gain over the incumbent, the default | none |
| `"ProbabilityOfImprovement"` | the mass above a threshold | `pi_margin`, over the incumbent, 0.0 |
| `"UpperConfidenceBound"` | the mean lifted by `ucb_beta` deviations | `ucb_beta`, positive, 2.0 |

A name outside those three is refused with a `ValueError` before the run spends an evaluation, in
a message that names the three the loop knows. That list is read off the same table the check
reads, so the message cannot name a set the check does not admit.

Both parameters are fixed for the run rather than passed per `suggest()` call, so two passes of one
run cannot maximize two different functions without anything recording which.

All three scores are defined where the posterior deviation is exactly zero, which the closed forms
are not. Expected improvement returns the gain itself there where the mean beats the incumbent and
zero where it does not, since the expectation over a point
mass is the gain, and probability of improvement returns 1 above the threshold and 0 on it or
below, since improvement is a strict excess. Deviation reaches zero when a caller turns the
numerical diagonal off, and when sklearn clips a numerically negative variance on the way out.

## Diagnostics

Five reads measure a run, and none of them returns a verdict. The failure modes are named, the
thresholds that separate them are not, because a threshold belongs to a search space and not to the
method.

| Read | Reads | What it catches |
|---|---|---|
| `read_gram` | a kernel matrix | the kernel discriminates nothing, or transfers nothing |
| `read_fit` | a surrogate against held-out terms | the posterior mean is wrong |
| `read_calibration` | standardized leave-one-out residuals | the mean is right, the spread is not |
| `read_frontier` | a finished acquisition maximization | the inner evolutionary run fails |
| `read_trace` | the per-pass trace of a run | the loop degenerates |

They are ordered by what they need, and a failure found early makes the later reads unreadable
rather than informative. An uninformative kernel produces a flat fit scatter, flat residuals and a
flat acquisition landscape, and only the first of the five says why.

Continuing the quick start above:

```python
from bayesian_optimization import read_calibration, read_fit, read_gram, read_trace

snapshot = bo.get_state_snapshot()
terms, values = snapshot["x_list"], snapshot["y_list"]

gram = read_gram(bo.kernel(terms), objective=values)
print(gram.size, gram.off_diagonal_mean, gram.minimum_eigenvalue, gram.objective_alignment)

kept, kept_values = terms[1::2], values[1::2]
held_out, held_values = terms[::2], values[::2]
fit = read_fit(bo.surrogate_over(kept, kept_values), held_out, held_values)
print(fit.size, fit.rank_correlation, fit.predictions_constant)

calibration = read_calibration(bo.surrogate_over_dataset())
print(calibration.size, calibration.standard_deviation, calibration.outside_two)

trace = read_trace(bo.trace)
print(trace.passes, trace.improvements, trace.fallbacks, trace.acquisition_at_zero)
```

The terms handed to `read_fit` have to be held out. Conditioned on them, a noise-free Gaussian
process reproduces its training values exactly, and the scatter would sit on the diagonal whatever
the kernel does. `surrogate_over()` fits one over a chosen subset with the run's own configuration,
which is what makes that split possible.

`read_frontier` needs a population as well as a pick, so `suggest(record_population=True)` or
`optimize(record_population=True)` has to ask for one. Where no pass was asked,
`last_acquisition_run` is `None`, which is a different statement from an empty record.

## The CNN architecture search

`bayesian_optimization/examples/cnn_damg_nas/` searches CIFAR-10 architectures over a synthesized
CNN space. Eleven modules carry it beside the package `__init__`, and they are split by what they
need rather than by topic.

| Module | What it holds |
|---|---|
| `cnn_damg_repo.py` | the combinators, their types and their four swap laws |
| `recognizable_cnn_damg_repo.py` | the same repository with the four laws stated compilably |
| `cnn_damg_targets.py` | the query types a run asks the space at |
| `cnn_damg_term_algebras.py` | the readings of a term that build no network, and import no torch |
| `cnn_damg_network_algebras.py` | a `torch.nn.Module` per combinator, and three algebras |
| `cnn_damg_kernels.py` | the named kernels a run may ask for |
| `cnn_damg_reference_architectures.py` | published networks as fully concrete literals |
| `cnn_damg_term_pool.py` | the evaluated terms themselves, beside the numbers of a run |
| `cnn_damg_experiment_utils.py` | the search, the persistence and the timing the drivers share |
| `cnn_damg_cifar_experiment.py` | the CIFAR-10 driver |
| `cnn_damg_verify_reference.py` | the reference training that anchors every comparison |

The repository declares 22 combinators. How often each of them reaches a drawn term is decided by
the literal sets rather than by any design intent: at structure length 2 with the USPS parameter
sets the pruned space has 307 non-terminals and 856 rules, and of those rules 579 belong to
`beside_cons` while `maxpool2d` has exactly one.

The four swap laws are stated once, in `bayesian_optimization/examples/swap_laws.py`, and inherited
by both the CNN and the older DAMG repository. What the laws read of a repository is the arities
of six combinators, and both give those six the same ones, which is why one statement serves two
alphabets. `recognizable_swap_laws.py` states the same four laws as an abstraction with a
relation on its values, which is the form `cosy.search.determinize` can compile into the
non-terminals. Compiling them is a trade rather than a gain: on that same space it turns 307
non-terminals and 856 rules into a product of 1044 states with 2605 rules, and buys a program that
can be counted from the program instead of from a materialized search tree.

A run picks one side of that trade and both halves of it together. `--sampling size-uniform`
determinizes and counts, `--sampling depth-bounded` searches the program as synthesized and draws
without counting. Asking for size-uniform sampling on the repository whose laws are still term
predicates raises rather than counting something else.

A smoke test says whether the pipeline runs and what one candidate costs on the machine at hand.
It has to name the sampler. Left out, the default determinizes the space first, and on the default
cell that costs far more than the two trainings the run is meant to time:

```
python -m bayesian_optimization.examples.cnn_damg_nas.cnn_damg_cifar_experiment \
    --epochs 2 --n-pre-samples 2 --n-iterations 1 --population-size 5 --evo-generations 3 \
    --sampling depth-bounded --data-dir ./data
```

A random-search baseline is the same script with `--n-pre-samples 30 --n-iterations 0`. The initial
dataset already comes from the sampler, so a run with no passes evaluates thirty drawn terms and
nothing else, through the same code and into the same artifacts. Which kind of run it was is read
off the pass count rather than off a flag beside it. `--baseline` is the other arrangement: it runs
a random arm of the same budget from the same initial design, in the same process, which is what
makes the comparison paired.

Every run writes seven artifacts, the CSV it is named after and six beside it: `<run>.csv` with
one row per evaluation, `<run>_terms.pickle` with the terms themselves, `<run>_config.json` with the
provenance, `<run>_trace.csv` with one row per pass, `<run>_ea.csv` with one row per generation of
the inner search, `<run>_surrogate.csv` with the per-pass model reads, and `<run>_diagnostics.json`
with five reads over the finished run and a note naming what to read them with. The CSV and the
term file are flushed as they are written, so an interrupted run keeps what it paid for.

## What a full search costs

A search costs what its evaluations cost, and on this space an evaluation is a training run. The
recorded VGG-scale run spent 75 286.94 seconds in the optimization loop on one NVIDIA A30, over 50
evaluations, an average of 1505.74 seconds each, with every evaluation training the same
architecture three times from three seeds for 50 epochs on CIFAR-10 at batch size 128.

Those 50 evaluations are 20 drawn terms and 30 loop passes, and the loop time divides very
unevenly between them. Of the 75 286.94 seconds, 58 669.84 went into the 30 acquisition
maximizations and 16 555.90 into training the 30 candidates they picked, so 78 percent of the loop
was the acquisition and a longer run buys more of that rather than more training. Building the
search space cost 34.09 seconds once.

The trainings of the 20 drawn terms are not in that total. They were taken over from an earlier
attempt, and they cost 15 778.99 seconds when they were paid for. The random arm cost a further
10 735.72 seconds after the loop had finished. All three together are the 43 070.60 seconds the
record reports as its training time.

Those numbers are the reason the smoke test above exists. Start there, read what one candidate
costs on the machine at hand, and multiply before starting a run with the defaults.

## The recorded run

`recorded_runs/` holds that search, so the numbers above can be read rather than believed. It is
the record of one run and not a fixture. Nothing in the package reads these files, so without a
test they would be eight artifacts no failure can reach, and `tests/test_recorded_run.py` is what
holds them against each other. The ninth file is the script the run was started with.

| File | What it holds |
|---|---|
| `bo_VGGM_20260805_r3.csv` | one row per evaluation, 80 of them over three phases |
| `bo_VGGM_20260805_r3_terms.pickle` | the 80 terms, in the order the CSV lists them |
| `bo_VGGM_20260805_r3_config.json` | what the run was told to do, and what it summed up to |
| `bo_VGGM_20260805_r3_trace.csv` | one row per loop pass |
| `bo_VGGM_20260805_r3_ea.csv` | the starting population and each generation, per maximization |
| `bo_VGGM_20260805_r3_surrogate.csv` | the per-pass model reads |
| `bo_VGGM_20260805_r3_diagnostics.json` | five reads over the finished run, gram to trace |
| `vgg11_bn_reference.json` | the anchor, three trainings of the reference architecture |
| `bo_20260805_r3.sh` | the script that drove the run |

The three phases of the CSV are 20 `pre_sample` rows, the initial design both arms share, 30
`bo_step` rows, what the loop picked, and 30 `random_sample` rows, the paired random arm. The
configuration is the summary of the file beside it: its counts and its maxima are counts and maxima
over those rows, which is what makes a truncated or a swapped CSV contradict its neighbor instead
of being read as the truth.

**The run drew depth-bounded, and never determinized the space.** The command line defaults to
`--sampling size-uniform`, which determinizes the space and draws from the determinized program,
so a rerun that leaves the option out is a different search rather than a slower one. The record
says so twice, through `bo_sampler.type` and through `search_program.sampling`, and the
determinization timings a size-uniform run writes are absent.

On this cell the default does not merely cost more, it draws nothing. Measured twice on one
laptop: the size-uniform build takes about a minute, roughly 24 seconds of synthesis and 34 of
determinization, turning 641 non-terminals and 3912 rules into 27 337 product states with 626 169
rules, and the first draw then counts for a further 58 seconds and yields an empty stream.

The reason is in the record itself. The 80 evaluated terms have sizes 486 to 795, and the
default `--size-bound` is 200, so the bound admits no term of this cell at all. A run that
wants size-uniform sampling here has to raise the bound first, and a run that leaves both
alone gets the `RuntimeError` that names how many distinct inhabitants it managed to collect,
which is none.

**Three defects of the record are pinned rather than corrected**, because the files are evidence
and repairing them would replace what happened with what should have happened.

1. The search-space block describes the tutorial geometry although the run searched the VGG cell.
   The driver wrote that block from its module constants rather than from the repository it
   built, which is repaired since. One field of the block gives it away, `num_feature_dimensions`
   is 48, which is the VGG closure and not the tutorial one.
2. The same block names `learning_rate_values` of 0.001, and all 80 evaluated terms carry 0.1,
   which is the rate of the VGG recipe.
3. The initial design was taken over from an earlier attempt, and nothing in the record says so or
   names the file it came from. The 20 rows were written within 0.04 seconds of each other while
   their trainings sum to 4.4 hours, which no run writes by training them.

The numbers the run reports do not depend on the first two. Repeating the run from the record does,
and so the three are stated here and asserted in the test rather than left for a reader to trip
over.

## The reference architecture

`cnn_damg_verify_reference.py` trains VGG-11-BN once, as the anchor every later comparison is read
against. The structure is a fully concrete literal, so synthesis has nothing to explore and returns
the reference architecture itself rather than a term that resembles it.

```
python -m bayesian_optimization.examples.cnn_damg_nas.cnn_damg_verify_reference \
    --epochs 50 --seeds 3 --data-dir ./data
```

Before the archive is read, the run synthesizes the term and counts its parameters. The count has
to be 9 228 362, which is the feature stack of torchvision's `vgg11_bn` without the convolution
biases that batch normalization cancels, plus a 10-class classifier. A mismatch stops the run
instead of producing a plausible wrong number. The run prints what that synthesis cost, 0.5 seconds
on the laptop this was checked on.

The number the run reports is the validation accuracy on a split carved out of the training data,
with the test accuracy recorded beside it and used for nothing else. A diverged training is written
out as diverged, with the epoch it stopped at, and gets no substitute value. `--seeds` is 1 by
default, and the anchor in `recorded_runs/` was made with 3.

## The older examples, and the known limits

Two examples predate the CNN work, and both run on a laptop.

`simple_nas/` fits small dense classifiers to the Iris table that ships beside it. Fourteen
combinators describe networks of up to five hidden layers, and the objective is the negated
training loss of the candidate, measured on the data it just trained on rather than on a held-out
part.

```
python -m bayesian_optimization.examples.simple_nas.simple_nas
```

`damg_nas/` is the legacy precursor of the CNN search. It models networks as string diagrams over a
different alphabet, inherits the same four swap laws, and fits a trapezoid function in one
dimension. The CNN modules import nothing from it.

```
python -m bayesian_optimization.examples.damg_nas.damg_example
```

What they cost. At their shipped defaults the whole of `simple_nas` took 225 seconds and the whole
of `damg_example` 163 seconds on one laptop CPU, so both are runs a reader can start and watch.
Most of that is the acquisition maximization and not the training. The DAMG example builds its
search space in under half a second, 88 non-terminals and 404 rules, and one candidate is a
2000-epoch fit
over 1000 points that costs a few tenths of a second, while the run maximizes the acquisition five
times
over 100 individuals for 100 generations. Both take every setting as an argument of `main`, so a
shorter run is a call and not an edit, and `damg_example` takes them on the command line as well.

What is legacy about them is what they were carried across from. Both were written against
`SimpleGeneticProgramming` with `AgeBasedReplacement`, neither of which today's `cosy` offers, and
both now run on `EvolutionarySearch` with `GenerousConservativeReplacement`. Each carried one
further defect of its own. The DAMG example ran at a mutation rate of 0.00 and a recombination rate
of 0.99, and the first of those is on the wrong side of the condition the convergence of the search
needs. The Iris example built its repository, its target, its search space and its whole optimizer
at module level, so every import paid for the space and no caller could change what it was built
with. Both are fixed, and neither is being modernized any further. New work goes into
`cnn_damg_nas/`.

`docs/` collects the limits that are properties of the method rather than defects to be fixed:
`docs/surrogate-limits.md` for the loop and the model, `docs/search-space-limits.md` for what a
search space has to admit before the loop can run on it, and `docs/cnn-example-limits.md` for the
CNN cell. `CONTRIBUTING.md` covers the setup, the checks and the rules the two layers are held to.
