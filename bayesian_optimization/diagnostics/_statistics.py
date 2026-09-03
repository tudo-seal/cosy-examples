from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import spearmanr

Values = Sequence[float] | np.ndarray
"""What the reads hand these functions: a list of numbers or the array they measured it into."""


def rank_correlation(first: Values, second: Values) -> float | None:
    """Return Spearman's rank correlation, or ``None`` where the ranks do not exist.

    Three of the reads summarize a picture by a rank correlation: the fit scatter, whose ranking of
    the predictions matters more than their values because the acquisition maximization consumes
    comparisons, the kernel-objective alignment, which correlates the kernel value of a pair of
    terms with how close the two objective values lie, and the decay of the acquisition over a run.
    They share this function so that they share the answer to the one awkward case.

    That case is a constant sequence.  Spearman's coefficient divides by the spread of the ranks,
    and a sequence with no spread has none, so the statistic is undefined and scipy answers
    ``nan``.  ``nan`` is exactly what this package refuses to hand on as a measurement: it compares
    false against every threshold, so a check written as ``rho > 0.5`` reports a failure and a
    check written as ``rho < 0.5`` reports one too.  ``None`` says instead that there is no number
    here, and the caller has to say what it makes of that.

    It is not an error either.  A surrogate that predicts one value everywhere draws the horizontal
    cloud of a kernel that transfers nothing, which is the *finding* rather than a fault in
    measuring it, and the reads report the constancy in a field of their own beside this one.

    Args:
        first (Values): One sequence of values.
        second (Values): The other, of the same length.

    Returns:
        float | None: The rank correlation, or ``None`` if either sequence is constant.

    Raises:
        ValueError: If the sequences differ in length, hold fewer than two entries, or hold a
            value that is not finite.
    """
    left = np.asarray(first, dtype=float).reshape(-1)
    right = np.asarray(second, dtype=float).reshape(-1)
    if left.size != right.size:
        msg = f"a rank correlation needs two sequences of equal length: {left.size} and {right.size}"
        raise ValueError(msg)
    if left.size < 2:
        msg = f"a rank correlation needs at least two pairs, got {left.size}"
        raise ValueError(msg)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        msg = "a rank correlation needs finite values; one of the sequences is not finite"
        raise ValueError(msg)

    if _is_constant(left) or _is_constant(right):
        return None
    return float(spearmanr(left, right).statistic)


def _is_constant(values: np.ndarray) -> bool:
    """Return whether every entry equals the first one.

    Exact equality on purpose.  A tolerance here would decide for the caller when a spread is too
    small to rank, and the reads report the spread separately so that the caller can decide.

    Args:
        values (np.ndarray): The values.

    Returns:
        bool: Whether they are all equal.
    """
    return bool(np.all(values == values.flat[0]))


def spread(values: Values) -> float:
    """Return the range of a sequence, its maximum less its minimum.

    The reads use it wherever "does this vary at all" is the question: a near-constant kernel
    matrix, a horizontal cloud of predictions, a posterior deviation pinned at one value over a
    whole run.

    Args:
        values (Values): The values, which must not be empty.

    Returns:
        float: The range, zero for a constant sequence.

    Raises:
        ValueError: If the sequence is empty.
    """
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        msg = "the range of an empty sequence is not a number"
        raise ValueError(msg)
    return float(np.max(array) - np.min(array))
