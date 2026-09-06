# The loop, the model and the reads

What follows holds for `bayesian_optimization/` outside `examples/`. Each entry names the module it
can be read in.

## The three scores at zero posterior deviation

The closed forms of expected improvement and probability of improvement divide by the posterior
standard deviation, so neither is defined where that deviation is exactly zero. Both definitions
are, and `acquisition_function.py` follows the definitions rather than the forms.

Expected improvement returns the gain itself where the deviation is zero, since the expectation
over a point mass is the gain, and zero where the mean does not beat the incumbent. Probability of
improvement returns 1 strictly above the threshold and 0 on it or below, since improvement is a
strict excess and a value certain to land on the threshold improves on it with probability zero.

Zero is reached in practice on two separate routes. The first is a caller who passes `alpha=0` and
turns the numerical diagonal off. Measured on a chain of six terms with the default kernel,
`normalize_y=True` and values rising from 1.0 to 6.0: at the default `alpha=1e-6` the deviation at
a training point is 1.7e-3, and at `alpha=0` it is exactly zero. The second is sklearn clipping a
numerically negative variance to zero on its way out of `predict`, which it warns about when it
does. The two are not the same route, and the first does not produce the second's warning.

## What the incumbent term does to expected improvement

Expected improvement is `(m - y*) Phi(z) + s phi(z)` with `z = (m - y*) / s`. Wherever the posterior
mean at a candidate is below the incumbent, the first summand is negative and the whole score is
carried by the second, which is the deviation term. Measured at an incumbent of 1.0 and a mean of
0.5, the score is 0.0, 0.0417, 0.1978 and 0.5727 for deviations 0.1, 0.5, 1.0 and 2.0, so it rises
with the uncertainty and with nothing else. The score is then a measure of uncertainty rather than
of expected quality, and the loop is exploring whether that was intended or not.

Two states put the loop there. The first is a kernel with a diagonal of one and no scale fitted to
the data, which fixes the prior variance at one, so the posterior deviation stays near what that
leaves behind. The second is a fitted amplitude driven to the lower bound of its `ConstantKernel`.
Measured on the same chain of six terms with `ConstantKernel(1.0, constant_value_bounds=(1e-4,
1e2))` in front of the default kernel: on an objective that is the same value everywhere the fit
lands on `0.01**2`, which is the lower bound, and warns that it did, while on the objective rising
from 1.0 to 6.0 it lands on `1.19**2` and warns about nothing. `README.md` says how to fit the
scale and what to watch when doing so.

Upper confidence bound has no incumbent in it and rises in both the mean and the deviation, so it
is not subject to this.

## The loop conditions on the distinct pairs, and replaces duplicates

`bo.py` fits the surrogate on the distinct pairs of the dataset, which is fewer rows than the
dataset holds whenever a term repeats in it. Two further mechanisms keep repeats out. The
acquisition scores already evaluated candidates below every genuine one, and a candidate that gets
through anyway is replaced by a fresh draw from the sampler.

Those three answer different cases and are not interchangeable. Conditioning on the distinct pairs
handles an exact observation repeating identically. The acquisition floor handles the maximization
being pulled toward a point the surrogate is already certain about, before a duplicate is returned
at
all. The replacement handles the maximization returning one anyway. A fourth case none of the three
touches is a quality measure that is itself stochastic: that one needs a noise term on the diagonal,
and this loop adds none by default, and this loop adds none by default: `_JITTER` is `1e-6` and it
is
a numerical guard on a Gram matrix of near-duplicate rows, not a noise level. A stochastic quality
measure, a training run for instance, needs a `WhiteKernel` in the kernel. That
does not loosen the distinct pairs clause: `_distinct_pairs` still raises on a dataset that gives
one term two different values, whatever the kernel says. The repetitions are averaged into the one
value a term carries before the loop sees them, which is what the CNN driver's `--repeats` does.

## Reading `Suggestion` and `last_acquisition_run`

`Suggestion.acquisition_value` is the score of the candidate returned, not of the candidate the
maximization proposed. Where `diagnostics["fallback_used"]` is true, the proposal was a duplicate
and was replaced, so the value describes the replacement. The two fields are read together or not
at all.

`last_acquisition_run.pick` is the other way round. It is what the maximization returned, which
after a replacement is not the term the loop went on to evaluate. The attribute that can be absent
is `last_acquisition_run` itself: it is set only where a pass was
asked to record a population, and `None` there says it was not asked. `pick` is never `None` once
the record exists.

## What each read does not decide

The five reads in `diagnostics/` return measurements and no verdicts. That is deliberate, since a
threshold belongs to a search space and not to the method, and each of the four below has a case it
cannot separate on its own. `read_gram` is the fifth, and it
reports the shape of the kernel matrix itself rather than a reading of one run, so it is read
against a space and not against a threshold.

`read_fit` summarizes the fit scatter by a rank correlation, and a rank correlation alone does not
tell a transferring kernel from a flat one. `rank_correlation` returns `None` where either side is
constant rather than the `nan` that would compare false against every threshold in both directions,
and `FitRead.predictions_constant` reports the constancy in a field of its own. Read the two
together, and the prediction spread beside them.

`read_calibration` reports the spread of the standardized leave-one-out residuals, and a spread far
above one means an overconfident surrogate only once the scale of the objective is fixed. The
leave-one-out deviation is a function of the kernel alone, since the posterior variance of a
Gaussian process does not depend on the values it observed, while the residual is in the units of
the objective. `CalibrationRead.targets_normalized` says whether the regressor centered and scaled
its targets, and it decides what the other numbers are a statement about. Read it first.

`read_frontier` places the returned term in the mean and deviation plane of the final population
and asks whether it lies on the upper right frontier. That reading holds for expected improvement
and for the upper confidence bound, which rise in both coordinates. It does not hold for
probability of improvement above its threshold, where the score falls as the deviation rises:
measured at an incumbent of 1.0, a zero margin and a mean of 1.5, the score goes 1.0, 0.8413,
0.6915, 0.5987 for deviations 0.1, 0.5, 1.0 and 2.0. Below the threshold it rises again.

`read_frontier` also answers about one generation and not about a run. It reads the final
population of one acquisition maximization, so it says whether the answer was the best thing
standing at the end, not whether the inner run improved. `AcquisitionRun.generations` carries the
per-generation record that answers the second question, and it is filled only where a pass was
asked to record a population.

`read_trace` reports `acquisition_trend`, and a falling trend does not separate a surrogate that is
learning from one that has collapsed. `TraceRead` carries `improvements`, `stalled_passes` and
`deviation_minimum` beside it for that reason, and a run with a clean falling trend and zero
improvements is the case the trend alone would have passed.

## The kernel-diverse initial design

`KernelDiverseInitializer` in `initial_sampling.py` draws one member at a time, biased away from the
members already drawn. Two properties decide whether it is affordable and whether it draws what it
claims to.

Every draw recounts. Each new member changes the reference set and with it the cost function, so
the branch counts are built anew per member. That is the price of drawing exactly, and it is why
the choice of kernel decides whether the initializer is practical: the subtree kernel is the one
whose cost is a fold with finite support, the subset-tree kernel folds with unbounded support, and
the Weisfeiler-Lehman kernel is not a fold at all, since its round labels read the context above a
node. All three are admissible, because an exact draw asks only that the cost be computable, and
all three pay the tree form here.

Its costs are real numbers. The counting machinery groups inhabitants by cost value, and this is
the first consumer whose costs are not integers, so two terms whose scores agree mathematically may
differ in the last bit and count as two cost classes, which changes the count per value and with it
every weight. A kernel with weights that do not sum exactly is where that would show.

It also carries a hypothesis nothing here can check: the generator has to be unambiguous within the
bound. The stream is deliberately not deduplicated, but the initializer takes the first inhabitant
outside the reference set and thereby skips repeats, so an ambiguous generator yields a population
that looks correct, distinct inhabitants at the requested size, while drawn in proportion to
derivation counts rather than to the intended weight. `cosy.search.assert_unambiguous_within`
decides the question on a bounded space.
