# The CNN architecture search

What follows holds for `bayesian_optimization/examples/cnn_damg_nas/`. It is the largest example
here, and most of what surprises a reader about it is a property of the search space rather than of
the driver.

## The literal sets decide which combinators a draw sees

A combinator is not present in the synthesized program once. It is present as often as there are
valid literal assignments for it, and that number follows from the combinatorics of the literal
sets rather than from any design intent.

Measured on the length-2 space with the USPS parameter sets, which are written out in
`tests/test_recognizable_cnn_repo.py` because the experiment script that carried them is not part of
this repository: the repository declares 22 combinators, and the pruned space has 307 non-terminals
and 856 rules. Of those rules 579 belong to
`beside_cons`, 72 each to `linear_layer` and `conv2d`, 69 to `beside_singleton`, and `maxpool2d`,
`cross_entropy_loss`, `adam_optimizer`, `sgd_optimizer`, `cosine_annealing_lr` and `no_scheduler`
have exactly one each. Three of the 22 declared combinators reach no rule of that space at all,
`sum`, `product` and `copy`, because each of them needs a feature width of 1 that this parameter
set does not offer.

The ratio carries into the draws, though not one for one, because a rule count is not an occurrence
count. Measured over 300 terms drawn by the size-uniform sampler at a size bound of 83: 154
occurrences of `beside_cons` against one of `maxpool2d`, and half the drawn terms hold no
`beside_cons` at all. Nothing in the repository says that a component is disadvantaged. Changing a
literal set changes those ratios, and
with them what a search can find, which is why the parameter sets belong in the record of a run.

## Steer a run through the target, never through the repository

The repository says what is expressible and the target says what is asked for. `cnn_damg_targets.py`
is a module of its own for that reason. Narrowing a search by editing the repository makes it
experiment specific and ends comparability between experiments, while the same narrowing written
into the target leaves the repository where it was.

The epoch count is the one place where the two have to agree. A target whose epoch count is not in
the repository's `n_epoch_values` is uninhabited, and synthesis then returns nothing rather than an
error. The CIFAR driver builds both from its own `--epochs` argument, which is what keeps them in
agreement.

## The relabeling round count is not fitted, and the right one differs per level

The named kernel `damg`, which is the driver's default, is a sum of three Weisfeiler-Lehman kernels
over three granularities of the architecture graph the term denotes, built by
`as_hierarchical_damg(1)`,
`(2)` and `(3)`, each with a `ConstantKernel(0.3)` in front, plus a `WhiteKernel(0.1)`. Its round
counts are `h = 0` for the finest granularity and `h = 1` for the
other two, so the round count is already per level rather than shared.

The marginal likelihood cannot reach that number. A fit adjusts the amplitudes and the noise level
and nothing else, so a run that wants a different round count has to name a different kernel.
`NAMED_KERNELS` therefore carries `damg@h2` and `damg@h012` beside the default, twelve named
kernels in all, and which of them a search space wants is a measurement rather than a default.

The `WhiteKernel` is what makes this kernel right for this objective. Training a network is a
stochastic quality measure, so the same architecture evaluated twice gives two numbers, and that is
what the noise term is for.

## The variation rates are a choice, not a result

`DEFAULT_CROSSOVER_RATE` is 0.9 and `DEFAULT_MUTATION_RATE` is 0.03 in
`cnn_damg_experiment_utils.py`. Reachability of the inner search is untouched by any positive value:
the root
is always a mutation point, and a mutation with an exhaustive sampler maps any individual to any
other with positive probability through the root. A rate of 0.00 is the one value that breaks it,
which is why the older DAMG example was moved off it.
The rate decides how hard the search is shaken, not what it can reach.

What the value is not is optimized. Whether the inner search finds the maximum of the acquisition
function is not measured anywhere in this repository, so treat the default as a value to set
explicitly rather than as a result. The recorded VGG-scale run was made with `--mutation-rate 0.15`,
which is a different choice and not a correction of this one.

## The reported number is a validation accuracy

`split_train_validation` carves a validation part out of the training data, a tenth of it by
default, and the objective a run maximizes is measured there. The test accuracy is recorded beside
it and used for nothing else.

That split is deterministic and carries its own seed, so two runs see the same split and a sweep
over search seeds does not silently become a sweep over splits as well.

## A diverged training keeps its own number

A candidate is evaluated by training it, and a training can diverge. Nothing here substitutes a
value for that. The record carries `diverged` with the epoch the training stopped at, and the
accuracy it reached is reported as measured.

With repeats, any repetition that diverged makes the whole candidate diverged, and the epoch count
reported is the earliest stop, because averaging a diverged training with two healthy ones would
report the mean of two different things. The spread over repetitions is `None` for a single
repetition rather than `0.0`.

## A baseline run is not paired unless it runs in the same process

The same script with `--n-pre-samples 30 --n-iterations 0` is a random-search baseline: the initial
dataset already comes from the sampler, so a run with no passes evaluates thirty drawn terms and
nothing else, through the same code and into the same artifacts.

That baseline is an independent sample at the same budget and not a paired one, which changes how
the comparison is read. What is lost is the variance reduction a shared initial design would have
bought. Pairing the two means running both from one process, which is what `--baseline` does. The
recorded run took that route, and its 80 rows are 20 shared design terms plus 30 per arm.

## Two run artifacts answer questions the CSV cannot

A run writes seven files, and two of them exist because the others cannot answer everything.

`<run>_terms.pickle` holds the term itself rather than its rendering. A rendered structure cannot be
fed back into a kernel, so without that file every offline question, which kernel orders this run or
what a different round count would have predicted, costs a full retraining. It is written by the
same call that writes the CSV row, so it cannot be omitted by accident, and a truncated tail is
reported as `TruncatedTermPool` rather than silently shortened.

`<run>_ea.csv` holds one row per generation per pass of the inner evolutionary search, generation 0
being the population that pass started from. The rows are aggregates and not terms: best,
population best, mean, worst, distinct members, last improvement and offspring accepted. Whether the
acquisition optimizer optimizes is read there and nowhere else,
because the loop keeps only the final population of the last pass.
