"""Initial designs for the Bayesian loop: the model-agnostic default and the informed alternative.

The literature offers three classes of initial design, random sampling, space filling, and warm
starting, and only two of them survive the move from a box with coordinates to a tree language.
Random sampling and space filling transfer through random search: sampled initialization with the
size-uniform sampler stratifies along the one canonical axis a search space has, its term size,
and cosy already carries it (``SampledInitialization`` over ``SizeUniformSampler``).  What this
module adds is the metric variant, which needs a metric that the term size does not give and the
surrogate's kernel does.  Warm starting stays outside: it presupposes a corpus of solved problems,
and the framework fixes one synthesis problem at a time.

Neither initializer is universally better.  The empirical recommendations diverge, and an evenly
spread design is not automatically an informative one.
"""
from __future__ import annotations

import itertools
import math
import random
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from cosy.core.tree import Tree
from cosy.evolutionary_algorithms import InitializationError
from cosy.search import (
    Sampler,
    k_st,
    normalized,
    reference_score,
    size_uniform,
    weighted_tree,
)

NT = TypeVar("NT")
T = TypeVar("T")
G = TypeVar("G")

# ``pi_R``: given the reference set, the distribution on the cost values it realizes.  There is one
# distribution per reference set, and the reference set is what changes from draw to draw.
PiStrategy = Callable[[Sequence[Tree[Any]]], Callable[[float], float]]

# One stream is long enough that this only ever fires on an exhausted space.  See the docstring
# below for why it is a count of draws and not of seconds.
_MAX_FALLBACK_DRAWS = 100


def _sample_fallback_tree(
    sampler: Sampler,
    query: Any,
    seen: set[Any],
    max_draws: int = _MAX_FALLBACK_DRAWS,
) -> Tree[Any]:
    """Draw the first inhabitant of ``query`` that this run has not evaluated yet.

    This is the rejection path: the Bayesian loop specifies no duplicate handling, and rather than
    inventing one this implementation redraws.  The draw comes from one stream of the sampler, not
    from ``max_draws`` separate calls, because each call re-poses the query, and that is what a
    draw costs on a realistic space.

    Under the size-uniform sampler the stream lists every inhabitant within the bound exactly once,
    so it enumerates the bounded space and ends, and the loop then falls through to the error.
    ``max_draws`` exists for the samplers that draw *with* replacement, where an exhausted space
    would otherwise spin forever.  It is deliberately not a fallback value: both exits raise.

    Args:
        sampler: The sampler to draw from.
        query: The generator query naming the search space and the requested type.
        seen: The candidates this run has already evaluated.
        max_draws: How many inhabitants to look at before giving up.

    Returns:
        Tree[Any]: The first drawn inhabitant that is not in ``seen``.

    Raises:
        RuntimeError: If the stream ended, or ``max_draws`` inhabitants were drawn, without a
            novel one among them.
    """
    for drawn, candidate in enumerate(sampler.sample(query), start=1):
        if candidate not in seen:
            return candidate
        if drawn >= max_draws:
            raise RuntimeError(
                f"the fallback drew {drawn} inhabitants from {sampler!r} and every one of them "
                f"had already been evaluated; the bounded search space appears exhausted"
            )
    raise RuntimeError(
        f"the stream of {sampler!r} ended without an inhabitant outside the "
        f"{len(seen)} already evaluated ones; the bounded search space is exhausted"
    )


def distinct_prefix(
    sampler: Sampler,
    query: Any,
    count: int,
    max_draws: int = _MAX_FALLBACK_DRAWS,
) -> tuple[list[Tree[Any]], int]:
    """Take ``count`` **pairwise distinct** inhabitants from one stream of ``sampler``.

    **Why the initial dataset needs this and a population does not.**  A population is a finite
    multiset of individuals, so sampled initialization adds whatever the stream delivers and
    repeats are admissible there.  The Bayesian loop's dataset is not a multiset.  It is the set
    of pairs the surrogate conditions on, and a repeated term is a training point that carries no
    observation the previous one did not, one evaluation of the budget spent on nothing.  With
    ``mu_0 = 10`` that is a tenth of the initial design.

    **Why it cannot be left to the sampler.**  Under size-uniform sampling every prefix of the
    stream is a sample **without replacement**, so ``count`` draws are already distinct and this
    function rejects nothing.  Under the depth-bounded random sampler they are not: its stream is
    a sequence of independent draws, which may repeat a term, and all that sampler guarantees is
    that every inhabitant within its bound can come first with positive probability.  A loop that
    takes its sampler as a parameter therefore cannot inherit the guarantee from its default,
    which is what the previous code did.

    So the rejection runs either way, and the count it returns is both things at once: **zero is a
    check** that the size-uniform stream really repeats nothing, and a positive number is the
    repair for a sampler that never promised it, recorded rather than absorbed.

    One stream and not ``count`` calls, for the reason ``_sample_fallback_tree`` gives: each call
    re-poses the query, and on a realistic space that is what a draw costs.

    Args:
        sampler: The sampler to draw from.
        query: The generator query naming the search space and the requested type.
        count: How many distinct inhabitants to collect.
        max_draws: How many inhabitants to look at in total before giving up.  Only reachable for
            samplers that draw with replacement, since the size-uniform stream ends on its own.

    Returns:
        tuple[list[Tree[Any]], int]: The terms in stream order, and how many repeats were rejected.

    Raises:
        ValueError: If ``count`` is negative.
        RuntimeError: If the stream ends or ``max_draws`` is reached before ``count`` distinct
            inhabitants are collected.  A short design is not a design, and topping it up with
            repeats is the thing this function exists to prevent.
    """
    if count < 0:
        msg = f"a design holds a non-negative number of terms, not {count}"
        raise ValueError(msg)
    drawn: list[Tree[Any]] = []
    rejected = 0
    for looked_at, candidate in enumerate(sampler.sample(query), start=1):
        # Structural comparison, not a set: a term hashes once when it is built and from its
        # labels alone, and nothing here requires a label to hash consistently with its equality
        # or to stay unchanged afterwards.  ``Tree.__eq__`` compares the size first, so the scan
        # short-circuits on almost every pair and the design is small either way.
        if any(candidate == kept for kept in drawn):
            rejected += 1
        else:
            drawn.append(candidate)
            if len(drawn) == count:
                return drawn, rejected
        if looked_at >= max_draws:
            msg = (
                f"{sampler!r} delivered {looked_at} inhabitants, {rejected} of them repeats, and "
                f"only {len(drawn)} distinct ones where {count} were asked for; the bounded search "
                f"space appears too small for this design"
            )
            raise RuntimeError(msg)
    msg = (
        f"the stream of {sampler!r} ended after {len(drawn)} distinct inhabitants "
        f"({rejected} repeats rejected) and the design asks for {count}; within its bound the "
        f"space holds fewer inhabitants than the run spends"
    )
    raise RuntimeError(msg)


def exponential_decay(sharpness: float = 8.0) -> PiStrategy:
    """Build the default weighting of cost values, ``value -> exp(-sharpness * value / |R|)``.

    The initializer needs a distribution ``pi_R`` on the cost values realized within the bound
    that is positive on all of them and decreasing in the value, and the shape is left open
    because the shape is what the caller tunes: a sharply decreasing ``pi_R`` concentrates the
    draw on the smallest realized values, a flat one spreads it over all of them evenly.

    The division by ``|R|`` is what makes one ``sharpness`` mean the same thing at every step.
    The cost is a *sum* of normalized similarities, so it ranges over ``[0, |R|]`` and its scale
    grows with the population drawn so far, and dividing brings the exponent back onto the mean
    similarity, in ``[0, 1]``.  Without it the same parameter is sharp at the first draw and flat
    at the tenth.

    A rank-based alternative, geometric decay over the sorted realized values, would be scale
    free without the division, but it has to know the whole set of realized values before it can
    score one of them, which costs a second counting pass over the search tree.  This form needs
    the value alone.

    Args:
        sharpness (float): How steeply the weight falls, per unit of mean similarity.  Must be
            positive: at zero the draw is uniform over the cost values, which no longer biases
            away from anything.

    Returns:
        PiStrategy: The strategy.

    Raises:
        ValueError: If ``sharpness`` is not strictly positive.
    """
    if not sharpness > 0.0:
        msg = f"the decay of pi_R must be strictly positive to be decreasing: {sharpness}"
        raise ValueError(msg)

    def pi_for(reference: Sequence[Tree[Any]]) -> Callable[[float], float]:
        """Return the distribution for one reference set.

        Args:
            reference (Sequence[Tree[Any]]): The terms drawn so far.

        Returns:
            Callable[[float], float]: The unnormalized weight of a cost value.
        """
        scale = float(len(reference)) or 1.0

        def pi(value: float) -> float:
            """Weight one realized cost value.

            Args:
                value (float): The cost value.

            Returns:
                float: Its unnormalized weight.
            """
            return math.exp(-sharpness * float(value) / scale)

        return pi

    return pi_for


class KernelDiverseInitializer:
    """Initializer that spreads its population apart under the surrogate's kernel.

    The population is drawn one member at a time, each draw biased away from the members already
    drawn.  ``t_1`` comes from the size-uniform stream, since with an empty reference set there is
    no similarity to be far from and the empty sum would make every term cost the same.  For
    ``i >= 1``, the cost function is ``c_R(t) = sum over r in R of k(t, r)``, the summed similarity
    of a term to the reference set, and ``t_{i+1}`` is the first inhabitant *outside* ``R_i`` that
    a random search for that cost and ``pi_{R_i}`` delivers.

    Informed search cannot do this: the induced distance is not monotone along the branches, so
    there is nothing for a best-first frontier to order.  Random search can, because it asks no
    monotonicity of its cost.

    **Two hypotheses the caller carries, because nothing here can check them.**

    *The generator must be unambiguous within the bound.*  Only then does a member follow the
    intended distribution, and this is the component where a violation hides best: the stream is
    deliberately not deduplicated (cosy leaves repeats visible), but "the first inhabitant
    *outside* R_i" skips them, so an ambiguous generator produces a population that looks
    perfectly correct, distinct inhabitants and the requested size, while being drawn in
    proportion to derivation counts rather than to the intended weight, which spreads the weight
    of a cost value evenly over the inhabitants that realize it.  Closure and termination hold
    either way, and only the distribution moves.  ``cosy.search.assert_unambiguous_within``
    decides the question on a bounded space.

    *Cost values are compared for equality.*  The counting machinery groups the inhabitants by
    cost value, and this is the first consumer whose costs are **reals** rather than integers: two
    terms whose scores agree mathematically may differ in the last bit and then count as two cost
    classes, which changes the count per value and with it every weight.  Measured on the
    expression space the float classes match the exact ones, but that is a measurement and not a
    guarantee.  A kernel with weights that do not sum exactly is where it would break.

    **Every draw recounts.**  Each step changes the reference set and with it the cost function,
    so the branch counts are built anew per member.  That is the price of drawing exactly, and it
    is why the choice of kernel decides whether the initializer is practical.  The subtree kernel
    is the one whose cost is a fold with finite support, hence the only one a table could ever
    count.  The subset-tree kernel folds with unbounded support, and the Weisfeiler-Lehman kernel
    is not a fold at all, since its round labels read the context *above* a node.  All three stay
    admissible, since an exact draw asks only that the cost be computable, but every one of them
    pays the tree form here, because cosy's table counts by size alone.

    Args:
        kernel (Callable[[Tree[Any], Tree[Any]], float]): The kernel ``k``.  The subtree kernel is
            the default because it is the one whose cost stays countable at scale.
        size_bound (int): The bound ``D`` on the term size.
        rng (random.Random): The source of randomness for both streams.
        normalize (bool): Whether to divide each kernel value by the square roots of the two
            self-similarities.  On by default because a counting kernel scores a large term high
            against itself, so an unnormalized score reads small terms as distant and the draws
            cluster at the short end.  Normalizing an already normalized kernel is the same map,
            since ``k_n(t, t) = 1``.
        pi (PiStrategy | None): The distributions ``pi_R``.  ``None`` selects
            :func:`exponential_decay`. (Default value = None)
    """

    def __init__(
        self,
        kernel: Callable[[Tree[Any], Tree[Any]], float] = k_st,
        *,
        size_bound: int,
        rng: random.Random,
        normalize: bool = True,
        pi: PiStrategy | None = None,
    ) -> None:
        self.kernel = normalized(kernel) if normalize else kernel
        self.size_bound = int(size_bound)
        self.rng = rng
        self.pi: PiStrategy = exponential_decay() if pi is None else pi

    def initialize(self, query: Any, size: int) -> list[Tree[Any]]:
        """Draw a population of ``size`` distinct inhabitants, each far from the ones before it.

        Args:
            query (Any): The generator query naming the search space and the requested type.
            size (int): The population size ``mu``.

        Returns:
            list[Tree[Any]]: Exactly ``size`` inhabitants, in the order they were drawn.

        Raises:
            ValueError: If ``size`` is negative, or if a ``pi_R`` rises with the value.
            InitializationError: If a stream delivers no further inhabitant before the population
                is full, which happens exactly when fewer than ``size`` inhabitants have size at
                most ``D``.
        """
        if size < 0:
            msg = f"a population size cannot be negative: {size}"
            raise ValueError(msg)
        if size == 0:
            return []

        first = next(iter(size_uniform(query, self.size_bound, self.rng)), None)
        if first is None:
            msg = (
                f"the size-uniform stream delivered no inhabitant of size at most "
                f"{self.size_bound}, so a population of {size} cannot be drawn"
            )
            raise InitializationError(msg)

        population: list[Tree[Any]] = [first]
        while len(population) < size:
            population.append(self._draw_outside(query, population, size))
        return population

    def _draw_outside(
        self, query: Any, reference: list[Tree[Any]], size: int
    ) -> Tree[Any]:
        """Draw the first inhabitant the random search delivers that is not in ``reference``.

        Args:
            query (Any): The generator query.
            reference (list[Tree[Any]]): The members drawn so far, ``R_i``.
            size (int): The requested population size, for the error message.

        Returns:
            Tree[Any]: The next member.

        Raises:
            ValueError: If ``pi_R`` rises with the cost value.
            InitializationError: If the stream ends without an inhabitant outside ``reference``.
        """
        drawn = set(reference)
        pi_r = self.pi(reference)
        weights: dict[float, float] = {}

        def cost(term: Tree[Any]) -> float:
            """Score a candidate against the reference set.

            Evaluated on ground terms alone.  A kernel score can also be read at a search *node*,
            over the partial inhabitant with its holes as fresh labels, and that reading is what
            an informed search would need.  The counting machinery evaluates the cost function
            only where a branch succeeds, so here a ground term is all that is ever passed.

            Args:
                term (Tree[Any]): The candidate.

            Returns:
                float: ``c_R(t)``.
            """
            return reference_score(term, reference, self.kernel)

        def recorded_pi(value: float) -> float:
            """Evaluate ``pi_R`` and remember what it said, so its shape can be checked.

            Args:
                value (float): A realized cost value.

            Returns:
                float: Its weight.
            """
            weight = float(pi_r(value))
            weights[float(value)] = weight
            return weight

        # Built rather than streamed directly, so that the shape of pi is checked against the
        # values this query actually realizes, before a single term is drawn under it.
        weighted = weighted_tree(query, self.size_bound, cost, recorded_pi)
        _decreasing_or_raise(weights)

        for candidate in weighted.stream(self.rng):
            if candidate not in drawn:
                return candidate
        msg = (
            f"the random search for a reference set of {len(reference)} terms delivered no "
            f"inhabitant outside it, so the population stops at {len(reference)} of {size}: "
            f"fewer than {size} inhabitants have size at most {self.size_bound}"
        )
        raise InitializationError(msg)


def _decreasing_or_raise(weights: dict[float, float]) -> None:
    """Check that ``pi_R`` does not rise with the cost value.

    A diverse draw needs a distribution that decreases in the cost value, and that is the whole
    mechanism: a rising one turns the initializer into its opposite, concentrating the draw on the
    terms most similar to those already drawn.  Nothing downstream would look wrong, since the
    population would still be distinct inhabitants, so the condition is checked here rather than
    left to the caller.

    Constant is admissible.  A flat distribution spreads the draw over all realized values evenly,
    the other end of the same dial.

    Args:
        weights (dict[float, float]): The weight of each realized cost value.

    Raises:
        ValueError: If a larger cost value carries a strictly larger weight.
    """
    ordered = sorted(weights.items())
    for (low_value, low_weight), (high_value, high_weight) in itertools.pairwise(ordered):
        if high_weight > low_weight:
            msg = (
                f"pi_R must be decreasing in the cost value, but it weights {high_value} at "
                f"{high_weight} above {low_value} at {low_weight}; a rising pi_R draws the terms "
                f"most similar to the ones already drawn"
            )
            raise ValueError(msg)
