"""Brute-force target distributions and the statistics that compare a sample against them.

A statistical test is only worth as much as the target it tests against, so nothing here is read
off the machinery under test.  The set of inhabitants within a bound comes from generate-and-check,
every term over the signature filtered by ``contains_tree``, and the weights are computed from
that set by the definition ``w(t) = pi(c(t)) / N_r(c(t))``: a cost value carries the probability
that ``pi`` gives it, spread uniformly over the ``N_r`` inhabitants that realize the value within
the bound.  The branch counts, which compute the same numbers inside the search, are never
consulted.

**Two statistics, and what each is for.**  The total variation distance is the quantity the
tolerances are stated in and the one the experiments report.  It is descriptive and has no
built-in notion of sample size.  Pearson's chi-square gives the decision, because it does: it says
whether a deviation of the observed size is what a correct sampler produces at this ``N``.  Every
assertion in the suite is therefore a p-value threshold, with the TV distance recorded next to it.

**A passing test is not yet evidence.**  A chi-square test against the right target passes just as
happily when the test is blunt, so every statistical claim in this suite is paired with a
*negative control*: the same sample, the same sample size, the same threshold, checked against a
deliberately wrong target that a defect would have produced.  The control has to fail for the
assertion to mean anything, and :func:`assert_rejects` is where that is written down.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import permutations, product
from typing import TYPE_CHECKING, Any

from cosy.core.tree import Tree
from cosy.search import checker

if TYPE_CHECKING:
    import random
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from cosy.core.solution_space import SolutionSpace
    from cosy.search.sampling import WeightedTree

__all__ = [
    "assert_rejects",
    "chi_square_pvalue",
    "cost_counts",
    "draw_first_elements",
    "draw_prefixes",
    "inhabitants_within",
    "plackett_luce_prefixes",
    "render",
    "target_weights",
    "terms_up_to",
    "total_variation",
]


# ---------------------------------------------------------------------------
# The brute-force oracle
# ---------------------------------------------------------------------------


def _compositions(total: int, parts: int) -> Iterable[tuple[int, ...]]:
    """Split a total into a fixed number of positive parts, in every way.

    Args:
        total (int): The number to split.
        parts (int): The number of parts.

    Yields:
        tuple[int, ...]: One composition per yield.
    """
    if parts == 1:
        if total >= 1:
            yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def terms_up_to(signature: Mapping[Any, int], bound: int) -> list[Tree]:
    """Build every term over a signature with at most ``bound`` symbol occurrences.

    Purely syntactic: no space is consulted, so the result is independent of every decision
    procedure the suite tests.

    Args:
        signature (Mapping[Any, int]): The function symbols with their arities.
        bound (int): The size bound ``D``.

    Returns:
        list[Tree]: All terms of size 1 to ``bound``.
    """
    by_size: dict[int, list[Tree]] = {size: [] for size in range(bound + 1)}
    for size in range(1, bound + 1):
        for symbol, arity in signature.items():
            if arity == 0:
                if size == 1:
                    by_size[size].append(Tree(symbol, ()))
                continue
            for split in _compositions(size - 1, arity):
                for children in product(*(by_size[part] for part in split)):
                    by_size[size].append(Tree(symbol, children))
    return [term for size in range(1, bound + 1) for term in by_size[size]]


def inhabitants_within(
    space: SolutionSpace, start: Any, signature: Mapping[Any, int], bound: int
) -> list[Tree]:
    """List every inhabitant of size at most ``bound``, by generate-and-check.

    Args:
        space (SolutionSpace): The space to test against.
        start (Any): The queried non-terminal.
        signature (Mapping[Any, int]): The function symbols with their arities.
        bound (int): The size bound ``D``.

    Returns:
        list[Tree]: The inhabitants of size at most ``bound``.
    """
    return [term for term in terms_up_to(signature, bound) if checker(space, start, term)]


def cost_counts(
    inhabitants: Sequence[Tree], cost: Callable[[Tree], Any]
) -> dict[Any, int]:
    """Count inhabitants per cost value.

    Args:
        inhabitants (Sequence[Tree]): The inhabitants within the bound.
        cost (Callable[[Tree], Any]): The cost function ``c``.

    Returns:
        dict[Any, int]: ``N_r``, per realized cost value.
    """
    counts: dict[Any, int] = {}
    for tree in inhabitants:
        value = cost(tree)
        counts[value] = counts.get(value, 0) + 1
    return counts


def target_weights(
    inhabitants: Sequence[Tree],
    cost: Callable[[Tree], Any],
    distribution: Callable[[Any], float],
) -> dict[Tree, float]:
    """Compute ``w(t) = pi(c(t)) / N_r(c(t))`` for every inhabitant, normalized to sum to one.

    The weight spreads the cost distribution over the inhabitants: a cost value carries the
    probability ``pi`` gives it, and within the value the inhabitants are uniform.  Here that
    definition is written out over the brute-force set.

    Args:
        inhabitants (Sequence[Tree]): The inhabitants within the bound.
        cost (Callable[[Tree], Any]): The cost function ``c``.
        distribution (Callable[[Any], float]): ``pi`` on the cost values, need not be normalized.

    Returns:
        dict[Tree, float]: The weight of each inhabitant.

    Raises:
        ValueError: If ``pi`` is not positive on a realized cost value.  That inhabitant could
            then never be drawn, and a target distribution with a hole in it would make every
            comparison against it meaningless.
    """
    counts = cost_counts(inhabitants, cost)
    probabilities: dict[Any, float] = {}
    for value in counts:
        probability = distribution(value)
        if probability <= 0.0 or not math.isfinite(probability):
            msg = f"pi must be positive on every realized cost value, but gives {probability} at {value!r}"
            raise ValueError(msg)
        probabilities[value] = probability
    total = sum(probabilities.values())
    return {
        tree: probabilities[cost(tree)] / total / counts[cost(tree)] for tree in inhabitants
    }


def plackett_luce_prefixes(
    weights: Mapping[Any, float], length: int
) -> dict[tuple[Any, ...], float]:
    """Compute the probability of every ordered prefix under sampling without replacement.

    A sample drawn without replacement takes the first element in proportion to ``w`` and then
    repeats on the remaining elements with the remaining weight, so a prefix ``(x_1, ..., x_k)``
    has probability ``prod_i w(x_i) / (W - sum_{j<i} w(x_j))``.  This is the Plackett-Luce form
    that random search is claimed to produce for every prefix of its stream.

    Args:
        weights (Mapping[Any, float]): The weight of each element, need not be normalized.
        length (int): The prefix length ``k``.

    Returns:
        dict[tuple[Any, ...], float]: The probability of each ordered prefix of that length.
    """
    total = sum(weights.values())
    result: dict[tuple[Any, ...], float] = {}
    for prefix in permutations(weights, length):
        probability = 1.0
        remaining = total
        for element in prefix:
            probability *= weights[element] / remaining
            remaining -= weights[element]
        result[prefix] = probability
    return result


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_first_elements(
    weighted: WeightedTree, rng: random.Random, draws: int
) -> Counter:
    """Draw the first element of ``draws`` independent streams.

    The counting is the expensive half of random search and does not depend on the randomness, so
    the caller builds the weighted tree once and this walks it once per draw, and only down to
    the first inhabitant, since the stream is lazy.

    Args:
        weighted (WeightedTree): The counted tree with its weights.
        rng (random.Random): The source of randomness.
        draws (int): The number of streams to start.

    Returns:
        Counter: How often each inhabitant came first.
    """
    observed: Counter = Counter()
    for _ in range(draws):
        observed[next(iter(weighted.stream(rng)))] += 1
    return observed


def draw_prefixes(
    weighted: WeightedTree, rng: random.Random, draws: int, length: int
) -> Counter:
    """Draw the first ``length`` elements of ``draws`` independent streams.

    Args:
        weighted (WeightedTree): The counted tree with its weights.
        rng (random.Random): The source of randomness.
        draws (int): The number of streams to start.
        length (int): The prefix length.

    Returns:
        Counter: How often each ordered prefix appeared.
    """
    observed: Counter = Counter()
    for _ in range(draws):
        stream = weighted.stream(rng)
        observed[tuple(next(stream) for _ in range(length))] += 1
    return observed


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def total_variation(observed: Mapping[Any, int], target: Mapping[Any, float]) -> float:
    """Return the total variation distance between an empirical and a target distribution.

    Args:
        observed (Mapping[Any, int]): The observed counts.
        target (Mapping[Any, float]): The target probabilities, normalized here if they are not.

    Returns:
        float: ``0.5 * sum_x |observed(x)/N - target(x)|``, over the union of both supports.
    """
    draws = sum(observed.values())
    total = sum(target.values())
    support = set(observed) | set(target)
    return 0.5 * sum(
        abs(observed.get(key, 0) / draws - target.get(key, 0.0) / total) for key in support
    )


def chi_square_pvalue(
    observed: Mapping[Any, int], target: Mapping[Any, float]
) -> tuple[float, float]:
    """Test observed counts against a target distribution with Pearson's chi-square.

    Args:
        observed (Mapping[Any, int]): The observed counts.
        target (Mapping[Any, float]): The target probabilities, normalized here if they are not.

    Returns:
        tuple[float, float]: The chi-square statistic and its p-value.

    Raises:
        ValueError: If something was observed that the target gives probability zero.  No
            p-value describes that: it is a categorical contradiction and has to be read as one.
    """
    from scipy import stats

    unexpected = [key for key in observed if target.get(key, 0.0) <= 0.0]
    if unexpected:
        msg = f"{len(unexpected)} observed outcome(s) carry probability zero under the target"
        raise ValueError(msg)
    draws = sum(observed.values())
    total = sum(target.values())
    keys = sorted(target, key=repr)
    counts = [observed.get(key, 0) for key in keys]
    expected = [target[key] / total * draws for key in keys]
    statistic, pvalue = stats.chisquare(f_obs=counts, f_exp=expected)
    return float(statistic), float(pvalue)


def assert_rejects(
    observed: Mapping[Any, int], wrong_target: Mapping[Any, float], threshold: float
) -> None:
    """Assert that the sample rejects a target the sampler would have to be broken to produce.

    The negative control of every statistical claim in the suite.  A chi-square test that accepts
    the truth proves nothing on its own, because at a small ``N`` it accepts almost anything, so
    each claim comes with a wrong distribution that a plausible defect would have produced, and
    this checks that the same sample at the same threshold turns it down.

    A control that puts probability zero on something that was observed is refuted outright, which
    is a *stronger* result than a small p-value, but only if it is a real alternative and not a
    distribution that simply forgot a category.  The two are indistinguishable from the outside, so
    this does not swallow the difference: a control must cover every outcome the sample produced,
    and one that does not is reported as a broken test rather than as a passing one.

    Args:
        observed (Mapping[Any, int]): The observed counts.
        wrong_target (Mapping[Any, float]): The distribution that must be rejected.
        threshold (float): The p-value threshold the positive claim was asserted at.

    Raises:
        AssertionError: If the wrong target survives the test, meaning the sample is too small or
            the statistic too blunt to support the positive claim either, or if the control does
            not cover the observed support, in which case it decides nothing.
    """
    uncovered = [key for key in observed if wrong_target.get(key, 0.0) <= 0.0]
    assert not uncovered, (
        f"the negative control gives probability zero to {len(uncovered)} observed outcome(s); a "
        f"control has to be a distribution the sampler could plausibly have produced, and one "
        f"missing a category rejects for the wrong reason"
    )
    _, pvalue = chi_square_pvalue(observed, wrong_target)
    assert pvalue < threshold, (
        f"the negative control survived at p={pvalue:.3g}: this sample cannot tell the claim "
        f"apart from a distribution it should exclude, so the positive assertion is vacuous"
    )


def render(tree: Tree) -> str:
    """Render a term as a readable string, for CSV output and failure messages.

    Args:
        tree (Tree): The term.

    Returns:
        str: ``root(child, ...)``, using the name of the combinator.
    """
    name = getattr(tree.root, "__name__", str(tree.root))
    if not tree.children:
        return name
    return f"{name}({', '.join(render(child) for child in tree.children)})"
