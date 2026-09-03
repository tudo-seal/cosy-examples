from __future__ import annotations

import warnings
from typing import Any

from cosy.core.tree import Tree

from .dpp import lazy_dpp_sample_optimized
from .kernels.graph_kernel import WeisfeilerLehmanKernel
from .kernels.kernel_base import StructuredKernelBase
from .kernels.tree_kernel import OrderedRootedSubtreeKernel


def _dedupe_trees(samples: list[Tree[Any]]) -> tuple[list[Tree[Any]], int]:
    """Remove duplicate Tree objects, preserving insertion order.

    Returns
    -------
    (unique_samples, duplicates_removed)
    """
    unique: list[Tree[Any]] = []
    seen: set[Any] = set()
    duplicates_removed = 0
    for tree in samples:
        if tree in seen:
            duplicates_removed += 1
            continue
        seen.add(tree)
        unique.append(tree)
    return unique, duplicates_removed


def _dedupe_sample_pairs(
    x_samples: list[Tree[Any]],
    y_samples: list[Any] | None = None,
) -> tuple[list[Tree[Any]], list[Any] | None, int]:
    """Remove duplicate (x, y) pairs, keeping first occurrence of each x.

    Raises
    ------
    ValueError
        If ``y_samples`` is provided and its length differs from ``x_samples``.
    """
    if y_samples is not None and len(x_samples) != len(y_samples):
        raise ValueError("The length of x0 and y0 must be the same.")

    unique_x: list[Tree[Any]] = []
    unique_y: list[Any] | None = [] if y_samples is not None else None
    seen: set[Any] = set()
    duplicates_removed = 0

    for i, tree in enumerate(x_samples):
        if tree in seen:
            duplicates_removed += 1
            continue
        seen.add(tree)
        unique_x.append(tree)
        if unique_y is not None and y_samples is not None:
            unique_y.append(y_samples[i])

    return unique_x, unique_y, duplicates_removed


def _sample_fallback_tree(
    initializer: Any,
    seen: set[Any],
) -> Tree[Any]:
    """Sample a diverse tree from the initializer, avoiding already-seen trees.

    Uses DPP sampling for diversity.  Raises ``RuntimeError`` if the
    initializer consistently returns ``None`` or exhausts its pool.
    """
    max_attempts = 100
    sample_space = initializer.initialize_population(max_attempts)
    candidates = lazy_dpp_sample_optimized(
        iter(sample_space),
        WeisfeilerLehmanKernel(),
        n_samples=1,
        selected=list(seen),
        inplace=False,
    )
    for candidate in candidates:
        if candidate is not None and candidate not in seen:
            return candidate
    raise RuntimeError(
        f"_sample_fallback_tree failed to produce a novel tree after "
        f"{max_attempts} attempts.  Check the initializer configuration."
    )


def _generate_unique_initial_samples(
    initializer: Any,
    target_count: int,
    seed_samples: list[Tree[Any]] | None = None,
    max_attempts_factor: int = 5,
    initializer_kernel: StructuredKernelBase | None = None,
    sample_overapproximation_factor: int = 100,
) -> tuple[list[Tree[Any]], dict[str, int]]:
    """Generate ``target_count`` unique, diverse initial samples.

    The function uses DPP sampling to promote diversity within the initial
    set.  If fewer than ``target_count`` unique samples can be generated a
    ``RuntimeWarning`` is emitted.

    Parameters
    ----------
    initializer:
        Object with an ``initialize_population(n)`` method.
    target_count:
        Desired number of unique samples.
    seed_samples:
        Optional list of pre-existing samples to avoid duplicating.
    max_attempts_factor:
        Multiplier for maximum generator pulls relative to ``target_count``.
    initializer_kernel:
        Kernel used for DPP diversity sampling.  Defaults to
        ``OrderedRootedSubtreeKernel()``.
    sample_overapproximation_factor:
        Over-sampling factor per batch to feed the DPP.

    Returns
    -------
    (samples, stats)
        ``samples`` is a list of at most ``target_count`` unique trees.
        ``stats`` is a dict with keys ``requested``, ``generated``,
        ``attempts``, ``duplicates_removed``.
    """
    if initializer_kernel is None:
        initializer_kernel = OrderedRootedSubtreeKernel()

    unique_samples: list[Tree[Any]] = []
    seen: set[Any] = set()

    if seed_samples:
        unique_samples, duplicates_removed = _dedupe_trees(list(seed_samples))
        seen.update(unique_samples)
    else:
        duplicates_removed = 0

    attempts = 0

    while (
        len(unique_samples) < target_count
        and attempts < target_count * max_attempts_factor
    ):
        batch_size = max(target_count - len(unique_samples), 1)
        pre_batch = initializer.initialize_population(
            batch_size * sample_overapproximation_factor
        )
        batch = lazy_dpp_sample_optimized(
            iter(pre_batch),
            initializer_kernel,
            n_samples=batch_size,
            selected=list(seen),
            inplace=False,
        )
        attempts += len(batch)
        for tree in batch:
            if tree in seen:
                duplicates_removed += 1
                continue
            seen.add(tree)
            unique_samples.append(tree)
            if len(unique_samples) >= target_count:
                break

    if len(unique_samples) < target_count:
        warnings.warn(
            f"Only {len(unique_samples)} unique initial samples could be "
            f"generated after {attempts} attempts (requested {target_count}).",
            RuntimeWarning,
            stacklevel=2,
        )

    return unique_samples[:target_count], {
        "requested": target_count,
        "generated": len(unique_samples),
        "attempts": attempts,
        "duplicates_removed": duplicates_removed,
    }
