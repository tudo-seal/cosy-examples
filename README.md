# cosy-examples

More elaborate examples for the CoSy framework, including a full
**Bayesian Optimization** package over CoSy `SolutionSpace`s.

---

## Bayesian Optimization (`bayesian_optimization/`)

A Gaussian-Process surrogate optimization loop for grammar-structured
search spaces.  The evolutionary algorithm (CoSy `Evolutionary`) explores
the solution space; the GP guides it via acquisition functions.

### Ask/Tell interface

```python
from bayesian_optimization import BayesianOptimization

bo = BayesianOptimization(
    search_space=my_space,
    request=start_symbol,
    optimizer=my_evolutionary,
    acquisition_function="ExpectedImprovement",
)

# --- Initialize ---
bo.initialize(x0=seed_trees, y0=seed_scores)

# --- Iterate ---
for _ in range(n_iters):
    suggestion = bo.suggest()            # fits GP, picks next candidate
    score = obj_fun(suggestion.candidate)
    bo.observe(suggestion.candidate, score)

result = bo.finalize()
print(result["best_tree"], result["best_y"])
```

State machine: `UNINITIALIZED → INITIALIZED → SUGGESTED → OBSERVED → …`
Invalid transitions raise `RuntimeError`.

### YTransform strategies

The GP is fitted on *transformed* y values for numerical stability.
Raw values (user scale) are stored in `bo._y_list` and returned by
`finalize()` / `best()`.

| Transform | Formula | Best used when |
|---|---|---|
| `Log1pTransform()` | `log1p(max(y, 0))` | Positive, heavy-tailed losses (default) |
| `IdentityTransform()` | `y` unchanged | Losses already well-scaled |
| `SignedLogTransform()` | `sign(y) * log1p(\|y\|)` | Signed objectives |
| `StandardizeTransform()` | `(y − μ) / σ` (frozen on first call) | General regression |

### Migration: `normalize_y` → `y_transform`

The old `normalize_y` parameter is deprecated.  Replace it:

| Old | New |
|---|---|
| `normalize_y=True` | `y_transform=Log1pTransform()` (default, no change needed) |
| `normalize_y=False` | `y_transform=IdentityTransform()` |

Passing both at once raises `ValueError`.

### Kernels

Four structured kernels ship with the package.

| Kernel | Compares two terms by | Declared hyperparameters | `k(t, t)` |
|---|---|---|---|
| `OrderedRootedSubtreeKernel` | the complete ordered subtrees they share (default) | none | 1 |
| `SubsetTreeKernel` | the subset trees they share | none | 1, and 0 for a single node |
| `WeisfeilerLehmanKernel` | `h` rounds of relabeling on the term graph | none | 1 |
| `HierarchicalWLKernel` | the same, one truncation level at a time | one weight per level | the sum of the weights |

The first three normalize their Gram matrix, so their similarity carries the shape of a term and
not its scale, and a kernel optimizer has nothing to move on them.  `SubsetTreeKernel` scores a
single node at 0 rather than at 1, because a single node roots no subset tree and normalization
leaves that row and column alone instead of dividing by zero.  `HierarchicalWLKernel` is the
exception on both counts: its weights are hyperparameters, and because it sums one normalized
kernel per level its diagonal is their sum, which model selection moves along with them.

`kernel_optimizer` is `None` by default for that reason.  sklearn takes an optimizer for a kernel
with no hyperparameters and then skips its optimization step without a word, so the old default
`"fmin_l_bfgs_b"` asked for model selection on every run and got none of it on the kernel the
package ships with.

A caller who wants the scale fitted puts a `ConstantKernel` in front of the kernel and passes an
optimizer with it:

```python
from sklearn.gaussian_process.kernels import ConstantKernel

from bayesian_optimization import OrderedRootedSubtreeKernel

bo = BayesianOptimization(
    ...,
    kernel=ConstantKernel(1.0) * OrderedRootedSubtreeKernel(),
    kernel_optimizer="fmin_l_bfgs_b",
)
```

Two things to watch once model selection is on.  While every observed value is still the same, the
state an initial design that lands on a plateau of the objective leaves behind, the marginal
likelihood has nothing to explain, and it drives the amplitude to the lower bound of the
`ConstantKernel` instead of reading a scale off the data.  sklearn reports that as a
`ConvergenceWarning` naming `constant_value`.  Adding a `WhiteKernel` to the sum makes the fit
worse rather than better: it gives the likelihood a way to call the whole spread observation
noise, and the fit that results predicts a constant and can carry the higher likelihood of the
two.  `HierarchicalWLKernel` needs no constant factor at all, since its weights are already the
scale.

Read the standardized leave-one-out residuals (`read_calibration`) before trusting a fitted
kernel.  A spread far above 1 is the overconfident surrogate the amplitude is meant to repair, and
a prediction that barely varies across candidates is one of the two collapses above.

Migrating from the old default: a caller who passed no kernel, or one of the three without
hyperparameters, gets bit for bit the same fit as before, because the optimizer was never run for
them.  A caller who passed a kernel that does declare hyperparameters and relied on the default to
fit them now has to pass `kernel_optimizer="fmin_l_bfgs_b"` as well.  That case is not silent: a
kernel with a non-empty `theta` and no optimizer is reported once per run, as is an optimizer over
a kernel with an empty one.

### Acquisition functions

| Name | Parameter | Notes |
|---|---|---|
| `"ExpectedImprovement"` | `ei_xi` (exploration) | Default |
| `"UpperConfidenceBound"` | `ucb_kappa` | `kappa` set at construction |
| `"DiversityUCB"` | `kappa0`, `lambda_div` | Encourages diverse exploration |

