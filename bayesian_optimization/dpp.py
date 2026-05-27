from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import Any

import numpy as np


class KernelCache:
    """Cache for symmetric kernel evaluations.

    Stores ``k(x, y)`` results to avoid recomputation.  Symmetry is exploited:
    ``cache[(x, y)] == cache[(y, x)]`` are both populated on the first lookup.
    """

    def __init__(self, kernel: Any) -> None:
        self.kernel = kernel
        self.cache: dict[tuple[Any, Any], float] = {}

    def get(self, x: Hashable, y: Hashable) -> float:
        key = (x, y)
        if key in self.cache:
            return self.cache[key]
        value = float(self.kernel([x], [y])[0, 0])
        self.cache[key] = value
        self.cache[(y, x)] = value
        return value


def lazy_dpp_sample_optimized(
    generator: Iterable[Any],
    kernel: Any,
    n_samples: int,
    *,
    score_threshold: float = 1e-8,
    max_attempts: int = 10000,
    selected: list[Any] | None = None,
    inplace: bool = True,
) -> list[Any]:
    """Lazy greedy DPP sampler for structured-kernel diversity sampling.

    Draws up to ``n_samples`` items from ``generator`` that are diverse
    according to the DPP criterion defined by ``kernel``.

    Parameters
    ----------
    generator:
        Iterable yielding candidate structured objects (e.g. ``Tree``).
    kernel:
        Structured kernel with a ``__call__(X, Y)`` interface.
    n_samples:
        Desired number of selected items.
    score_threshold:
        Minimum DPP conditional variance for a candidate to be accepted.
    max_attempts:
        Maximum number of items drawn from ``generator`` before early exit.
    selected:
        Optional list of already-selected items to seed the DPP.
    inplace:
        If ``True`` (default) the ``selected`` list is extended in-place and
        returned.  If ``False`` a new list is returned and ``selected`` is not
        modified.

    Returns
    -------
    list
        The ``selected`` list (same object if ``inplace=True``, new if ``False``).
    """
    if selected is None:
        working: list[Any] = []
    elif inplace:
        working = selected
    else:
        working = list(selected)

    K_inv: np.ndarray | None = None if not working else None

    kernel_cache = KernelCache(kernel)
    attempts = 0

    # If there are pre-seeded items, we need to rebuild K_inv for them.
    # For simplicity, start fresh (DPP variance is only approximate for seeded items).
    if working:
        seed = working[0]
        kxx = kernel_cache.get(seed, seed)
        K_inv = np.array([[1.0 / max(kxx, 1e-12)]])
        for i in range(1, len(working)):
            item = working[i]
            k_vec = np.array([kernel_cache.get(item, s) for s in working[:i]], dtype=float)
            kxx = kernel_cache.get(item, item)
            score = float(max(kxx - k_vec @ K_inv @ k_vec, 0.0))
            if score > score_threshold:
                k_vec = k_vec.reshape(-1, 1)
                A, B = K_inv, k_vec
                C = k_vec.T
                D = np.array([[kxx]])
                schur = float(max(float(D[0, 0] - (C @ A @ B)[0, 0]), 1e-12))
                schur_inv = 1.0 / schur
                top_left = A + (A @ B @ C @ A) * schur_inv
                top_right = -(A @ B) * schur_inv
                bottom_left = -(C @ A) * schur_inv
                bottom_right = np.array([[schur_inv]])
                K_inv = np.block([[top_left, top_right], [bottom_left, bottom_right]])

    for candidate in generator:
        attempts += 1
        if attempts > max_attempts:
            break

        if not working:
            kxx = kernel_cache.get(candidate, candidate)
            working.append(candidate)
            K_inv = np.array([[1.0 / max(kxx, 1e-12)]])
            if len(working) >= n_samples:
                break
            continue

        k_vec = np.array([kernel_cache.get(candidate, s) for s in working], dtype=float)
        kxx = kernel_cache.get(candidate, candidate)
        score = float(max(kxx - k_vec @ K_inv @ k_vec, 0.0))

        if score <= score_threshold:
            continue

        working.append(candidate)

        k_col: np.ndarray = k_vec.reshape(-1, 1)  # type: ignore[assignment]
        A, B = K_inv, k_col  # type: ignore[assignment]
        C = k_col.T
        D = np.array([[kxx]])
        schur = float(max(float(D[0, 0] - (C @ A @ B)[0, 0]), 1e-12))
        schur_inv = 1.0 / schur
        top_left = A + (A @ B @ C @ A) * schur_inv
        top_right = -(A @ B) * schur_inv
        bottom_left = -(C @ A) * schur_inv
        bottom_right = np.array([[schur_inv]])
        K_inv = np.block([[top_left, top_right], [bottom_left, bottom_right]])

        if len(working) >= n_samples:
            break

    return working
