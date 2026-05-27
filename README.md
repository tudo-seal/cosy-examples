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
from bayesian_optimization import BayesianOptimization, Log1pTransform
from bayesian_optimization import OrderedRootedSubtreeKernel

bo = BayesianOptimization(
    search_space=my_space,
    request=start_symbol,
    optimizer=my_evolutionary,
    acquisition_function="ExpectedImprovement",
    y_transform=Log1pTransform(),   # default
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

### Acquisition functions

| Name | Parameter | Notes |
|---|---|---|
| `"ExpectedImprovement"` | `ei_xi` (exploration) | Default |
| `"UpperConfidenceBound"` | `ucb_kappa` | `kappa` set at construction |
| `"DiversityUCB"` | `kappa0`, `lambda_div` | Encourages diverse exploration |

