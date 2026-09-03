"""The mechanics of kernel-diverse initialization.

A kernel-diverse initializer draws a population one member at a time, each draw biased away from
the members already drawn: the first member comes from a size-uniform stream, and every later one
is the first inhabitant outside the members so far that a random search on the summed kernel
similarity to them delivers.

What that fixes is tested here: where the first member comes from, that later members are drawn
from *outside* the reference set, that the reference set and with it the cost function change per
draw, and that a stream which runs dry is a failure rather than a short population. The
distributional half, that a later member is drawn with its cost following the chosen distribution
and uniformly within a cost value, is a claim about a distribution and is checked against the
oracles instead, as is the worked example.
"""
from __future__ import annotations

import random

import pytest
from cosy.core.tree import Tree
from cosy.evolutionary_algorithms import InitializationError
from cosy.search import generator_query, k_st, normalized, term_size

from tests.spaces import CHAIN, EXPR, chain_space, expression_space

_D = 21  # the bound that cuts the chain at length ten: a list of length l has size 2 * l + 1


@pytest.fixture(scope="module")
def chain():
    return chain_space()


@pytest.fixture(scope="module")
def chain_query(chain):
    return generator_query(chain, CHAIN)


def _length(term) -> int:
    """Return the list length of a chain term, which its size determines."""
    return (term_size(term) - 1) // 2


def _initializer(size_bound=_D, seed=1, **kwargs):
    from bayesian_optimization.initial_sampling import KernelDiverseInitializer

    return KernelDiverseInitializer(
        k_st, size_bound=size_bound, rng=random.Random(seed), **kwargs
    )


# ---------------------------------------------------------------------------
# What the definition fixes
# ---------------------------------------------------------------------------

def test_the_population_holds_distinct_inhabitants_of_the_space(chain_query, chain):
    """Every member is an inhabitant within the size bound, and no member repeats."""
    population = _initializer().initialize(chain_query, 5)

    assert len(population) == 5
    assert len(set(population)) == 5, "each member is drawn from outside the ones before it"
    inhabitants = set(chain.enumerate_trees(CHAIN, max_count=40))
    assert all(t in inhabitants for t in population)
    assert all(term_size(t) <= _D for t in population)


def test_the_first_member_comes_from_the_size_uniform_stream():
    """The first member is a size-uniform draw, not a kernel draw.

    With an empty reference set every cost is the empty sum, so the summed similarity is constant
    and a random search on it would draw by the distribution alone over a single cost value: that
    is, uniformly over the whole bounded space, not size-uniformly. Size-uniform sampling instead
    draws a realized size uniformly and then an inhabitant of that size uniformly, and the two
    differ wherever a size carries more than one inhabitant. That is why the first member is
    specified as a size-uniform draw.

    **This runs on the expression space, and the chain would make it vacuous.** The chain holds
    exactly one inhabitant per size, so the size-uniform weight is one over the number of terms
    within the bound there and the two distributions coincide: measured, identical first draws on
    20 of 20 seeds. On the expression space, where the sizes carry 1, 1, 2, 4, 9, 21 inhabitants,
    they agree on 2 of 20. The condition stated above is one the test space has to satisfy.
    """
    from cosy.search import size_uniform

    space = expression_space()
    query = generator_query(space, EXPR)
    bound = 9

    for seed in range(5):
        initializer = _initializer(size_bound=bound, seed=seed)
        first_of_population = initializer.initialize(query, 3)[0]
        first_of_stream = next(iter(size_uniform(query, bound, random.Random(seed))))
        assert first_of_population == first_of_stream


def test_every_later_member_is_drawn_from_outside_the_ones_before(chain_query):
    """Each later member is the first inhabitant outside those already drawn.

    A repeat is skipped, not returned.
    """
    population = _initializer(seed=7).initialize(chain_query, 8)
    assert len(set(population)) == 8


def test_the_reference_set_grows_with_every_draw(chain_query, monkeypatch):
    """Each step scores against *all* members drawn so far, so the cost function changes per draw.

    Scoring against the last member alone would still produce a spread-looking population, which
    is why this is pinned on the calls rather than on the outcome.
    """
    import bayesian_optimization.initial_sampling as module

    seen_reference_sizes: list[int] = []
    original = module.reference_score

    def recording(term, reference, kernel):
        reference = list(reference)
        seen_reference_sizes.append(len(reference))
        return original(term, reference, kernel)

    monkeypatch.setattr(module, "reference_score", recording)
    _initializer(seed=3).initialize(chain_query, 4)

    # Draw i scores against i reference terms, and the first member takes no kernel score at all.
    assert set(seen_reference_sizes) == {1, 2, 3}


# ---------------------------------------------------------------------------
# Failure semantics: when initialization fails, and when it must not
# ---------------------------------------------------------------------------

def test_a_population_larger_than_the_bounded_space_fails(chain_query):
    """Initialization fails when a request delivers no further inhabitant.

    The bound cuts the chain at length ten, so eleven inhabitants exist and a twelfth cannot be
    drawn. The population is not filled with repeats, and it is not returned short.
    """
    with pytest.raises(InitializationError, match="12"):
        _initializer().initialize(chain_query, 12)


def test_the_whole_bounded_space_is_still_reachable(chain_query):
    """Exactly 11 succeeds where 12 fails, so the failure clause is sharp, not conservative."""
    population = _initializer(seed=11).initialize(chain_query, 11)
    assert sorted(_length(t) for t in population) == list(range(11))


def test_an_empty_population_is_empty_rather_than_an_error(chain_query):
    assert _initializer().initialize(chain_query, 0) == []


def test_a_negative_population_size_is_rejected(chain_query):
    with pytest.raises(ValueError, match="negative"):
        _initializer().initialize(chain_query, -1)


# ---------------------------------------------------------------------------
# The two parameters
# ---------------------------------------------------------------------------

def test_the_kernel_is_normalized_unless_that_is_switched_off(chain_query):
    """Normalization is on by default, and applying it twice changes nothing.

    A counting kernel scores a large term high against itself, so large terms dominate the scale
    and an unnormalized score reads small terms as distant. Normalization divides each entry by
    the square roots of the two self-similarities, which puts every self-similarity at one, so a
    caller who hands in an already normalized kernel loses nothing: normalizing twice is the same
    map.
    """
    from bayesian_optimization.initial_sampling import KernelDiverseInitializer

    rng = random.Random(0)
    once = KernelDiverseInitializer(k_st, size_bound=_D, rng=rng)
    twice = KernelDiverseInitializer(normalized(k_st), size_bound=_D, rng=rng)
    raw = KernelDiverseInitializer(k_st, size_bound=_D, rng=rng, normalize=False)

    terms = list(chain_space().enumerate_trees(CHAIN, max_count=40))
    a, b = terms[0], terms[3]
    assert once.kernel(a, b) == pytest.approx(twice.kernel(a, b))
    assert once.kernel(a, b) != pytest.approx(raw.kernel(a, b))


def test_a_distribution_that_rises_with_the_value_is_rejected(chain_query):
    """The distribution over cost values has to decrease, which is what makes a draw diverse.

    A rising one turns the initializer into its own opposite, concentrating on the terms most
    similar to those already drawn, and nothing downstream would look wrong.
    """
    with pytest.raises(ValueError, match="decreasing"):
        _initializer(pi=lambda reference: (lambda value: float(value) + 1.0)).initialize(
            chain_query, 3
        )


def test_a_distribution_that_is_not_positive_is_rejected(chain_query):
    """The distribution has to be positive on every cost value the bounded space realizes.

    A value of zero drops realizable inhabitants from every draw.
    """
    with pytest.raises(ValueError, match="positive"):
        _initializer(pi=lambda reference: (lambda value: 0.0)).initialize(chain_query, 3)


def test_a_flat_distribution_is_admissible(chain_query):
    """A flat distribution is one end of the dial, not an error.

    A sharply decreasing distribution concentrates on the smallest realized cost values and a flat
    one spreads over all of them evenly, so the check for a decreasing distribution has to let the
    flat case through. Written as a strict decrease it would reject that second extreme, and the
    failure would look like a broken caller.
    """
    population = _initializer(
        pi=lambda reference: (lambda value: 1.0)
    ).initialize(chain_query, 4)
    assert len(set(population)) == 4


def test_an_exhausted_stream_at_the_first_member_is_an_error(chain_query):
    """The failure clause covers the first member too, and a bound of zero is where it bites.

    No term has size zero, so the size-uniform stream is empty before it starts. Returning the
    empty population there would be a substitute for a measurement that failed.
    """
    with pytest.raises(InitializationError, match="size at most"):
        _initializer(size_bound=0).initialize(chain_query, 1)


def test_a_sharpness_of_zero_is_rejected(chain_query):
    """At zero the decay is constant, so the draw is uniform over the cost values.

    That is admissible for the initializer, but no longer the *decreasing* distribution this
    strategy is, and an initializer built on it has silently stopped biasing away from anything.
    Zero is the boundary that matters: a guard written as ``sharpness < 0`` passes every negative
    test and lets exactly this through.
    """
    from bayesian_optimization.initial_sampling import exponential_decay

    with pytest.raises(ValueError, match="positive"):
        exponential_decay(0.0)


def test_the_default_sharpness_means_the_same_at_every_step():
    """The exponential decay divides by the reference-set size, so one sharpness holds throughout.

    It reads the *mean* similarity, ``exp(-s * v / |R|)`` for a summed similarity ``v`` over a
    reference set ``R``. Without the division the same parameter is sharp at the first draw and,
    at a population of 40, a deterministic argmin, because the cost is a sum over the reference
    set and its scale grows with the population. This is pinned on the strategy alone, since the
    distributional tests build their target with this same function, so a defect in it would move
    target and draw together and go unseen.
    """
    from bayesian_optimization.initial_sampling import exponential_decay

    strategy = exponential_decay(8.0)
    one_term = [Tree("r")]
    five_terms = [Tree("r")] * 5

    # Mean similarity 0.3 either way: 0.3 against one reference term, 1.5 against five.
    assert strategy(one_term)(0.3) == pytest.approx(strategy(five_terms)(1.5))
    assert strategy(one_term)(0.9) == pytest.approx(strategy(five_terms)(4.5))


def test_the_sharpness_of_the_default_is_a_parameter(chain_query):
    """Flat and sharp are both admissible, and they draw differently."""
    from bayesian_optimization.initial_sampling import exponential_decay

    flat = _initializer(seed=5, pi=exponential_decay(0.01)).initialize(chain_query, 4)
    sharp = _initializer(seed=5, pi=exponential_decay(30.0)).initialize(chain_query, 4)

    assert flat[0] == sharp[0], "the first member does not depend on pi"
    assert [_length(t) for t in flat] != [_length(t) for t in sharp]
