"""What the kernel-diverse initializer draws, and how often.

Two claims, two methods.  The worked example of three kernel-diverse draws on a chain of lists
carries numbers, so it is checked exactly, against the closed form of the subtree kernel on that
chain and against the values its figure plots.  The exactness claim, that the next member is drawn
in proportion to the cost-distribution weight among the inhabitants not yet drawn, is a statement
about a distribution, so it is checked statistically, against weights computed from the definition
by brute force, with a negative control that the same sample has to reject, as everything in this
suite does.
"""
from __future__ import annotations

import random
from collections import Counter

import pytest
from cosy.search import (
    generator_query,
    k_st,
    normalized,
    reference_score,
    term_size,
)

from bayesian_optimization.initial_sampling import (
    KernelDiverseInitializer,
    exponential_decay,
)
from tests.oracles import (
    assert_rejects,
    chi_square_pvalue,
    inhabitants_within,
    target_weights,
)
from tests.spaces import (
    CHAIN,
    EXPR,
    EXPR_SIGNATURE,
    chain_space,
    chain_terms,
    expression_space,
)

# A list of length l has size 2l + 1, so the bound 21 cuts the chain at length ten.
_D = 21
_THRESHOLD = 1e-3
_DRAWS = 2000


@pytest.fixture(scope="module")
def chain():
    return chain_space()


@pytest.fixture(scope="module")
def chain_query(chain):
    return generator_query(chain, CHAIN)


@pytest.fixture(scope="module")
def by_length():
    """The chain's inhabitants, indexed by list length."""
    return {(term_size(t) - 1) // 2: t for t in chain_terms(_D)}


def _length(term) -> int:
    return (term_size(term) - 1) // 2


# ---------------------------------------------------------------------------
# The worked example, exactly
# ---------------------------------------------------------------------------

def test_the_subtree_kernel_on_the_chain_has_the_closed_form_of_the_example(by_length):
    """On the chain, the subtree kernel of two lists of lengths a <= b counts a * b + a + 1.

    The z subterms match in a * b pairs, the ending nil in one, and the tails of lengths one to a
    in a more.  The example computes with this closed form rather than with the kernel, so if the
    two ever diverge every number below is about a different object than the example's.
    """
    for a in range(11):
        for b in range(a, 11):
            assert k_st(by_length[a], by_length[b]) == a * b + a + 1


def test_the_figure_values_of_the_summed_similarity_reproduce(by_length):
    """The summed similarity to the lists of lengths five and zero matches the plotted values.

    Both readings, normalized and unnormalized, to the four decimals the figure plots.  This is
    the pair the example turns on: the normalized one places the least similar list at the far end
    of the chain, the unnormalized one places it next to the empty list already drawn.
    """
    reference = [by_length[5], by_length[0]]
    knorm = normalized(k_st)
    candidates = [length for length in range(11) if length not in (0, 5)]

    plotted_normalized = {
        1: 1.3032, 2: 1.2605, 3: 1.2238, 4: 1.1980,
        6: 1.1385, 7: 1.1078, 8: 1.0840, 9: 1.0650, 10: 1.0496,
    }
    plotted_unnormalized = {1: 8, 2: 14, 3: 20, 4: 26, 6: 37, 7: 42, 8: 47, 9: 52, 10: 57}

    scored_normalized = {
        length: reference_score(by_length[length], reference, knorm)
        for length in candidates
    }
    scored_unnormalized = {
        length: reference_score(by_length[length], reference, k_st) for length in candidates
    }

    for length in candidates:
        assert scored_normalized[length] == pytest.approx(
            plotted_normalized[length], abs=5e-5
        )
        assert scored_unnormalized[length] == pytest.approx(plotted_unnormalized[length])

    assert min(scored_normalized, key=scored_normalized.get) == 10
    assert min(scored_unnormalized, key=scored_unnormalized.get) == 1


def test_against_a_single_draw_the_empty_list_is_the_least_similar(by_length):
    """Against a single drawn list of length five, the empty list is the least similar one.

    The normalized similarity peaks at the neighboring lengths six and four and is smallest for
    the empty list, which is what makes the empty list the likely second draw.
    """
    knorm = normalized(k_st)
    scored = {
        length: reference_score(by_length[length], [by_length[5]], knorm)
        for length in range(11)
        if length != 5
    }
    assert min(scored, key=scored.get) == 0
    assert max(scored, key=scored.get) == 6


@pytest.mark.parametrize(
    ("normalize", "sharpness", "trace"),
    [(True, 300.0, [5, 0, 10]), (False, 2.0, [5, 0, 1])],
)
def test_the_three_draws_of_the_example(chain_query, normalize, sharpness, trace):
    """Normalized, the three draws spread over the chain, and unnormalized they cluster instead.

    The example supposes the size-uniform first draw returns length five, and seed 26 is a run
    where it does.  The draw is random, so this pins the modal path and the sharpness is chosen to
    make it dominant: over the 400 seeds whose first draw is five, the traces below come up in 33
    of 34 runs normalized and 34 of 34 unnormalized.

    The two sharpnesses differ on purpose, and not to flatter the result.  The unnormalized cost
    runs up to 57 on this chain, so a sharpness that is ordinary on the normalized kernel drives
    ``exp`` out of the double range there, which the test below measures.  What the comparison
    turns on is which term the cost calls least similar, and that is a property of the kernel
    rather than of the sharpness: the argmin is 10 normalized and 1 unnormalized at every
    admissible sharpness, because ``pi`` is decreasing.
    """
    initializer = KernelDiverseInitializer(
        k_st,
        size_bound=_D,
        rng=random.Random(26),
        normalize=normalize,
        pi=exponential_decay(sharpness),
    )
    assert [_length(t) for t in initializer.initialize(chain_query, 3)] == trace


def test_an_unnormalized_sharpness_that_underflows_is_refused(chain_query):
    """A sharpness whose weights underflow to zero is refused rather than quietly accepted.

    The initializer asks for a ``pi`` positive on every cost value realized within the bound, and
    on an unnormalized counting kernel there is a sharpness above which floating point cannot
    deliver it.  The bound is arithmetic, not incidental.  The unnormalized cost reaches 56
    against a single reference term here, and ``exp(-s * 56)`` measured: it goes **subnormal** at
    ``s = 12.65`` and reaches exactly zero at ``s = 13.31``.  Only the second is refused.  Between
    the two the draw runs on weights with a handful of significant bits: at ``s = 13.3`` the
    weight is ``5e-324``, the smallest subnormal there is, so the ratios ``pi`` was meant to
    express are gone while nothing has failed.  Normalized, the same cost lies between zero and
    the number of reference terms, and the exponent stays above ``-sharpness``, so the default
    strategy cannot reach either regime.

    A sharpness of 30 is an ordinary choice on the normalized kernel, the one that makes the modal
    draw dominant in the example above, and it is past both bounds on the raw counts.  The refusal
    comes from the counting machinery, which will not spread a distribution that gives a realized
    cost value zero: those inhabitants could never be drawn, and a silent hole in the support is
    worse than a stopped run.
    """
    initializer = KernelDiverseInitializer(
        k_st,
        size_bound=_D,
        rng=random.Random(26),
        normalize=False,
        pi=exponential_decay(30.0),
    )
    with pytest.raises(ValueError, match="positive"):
        initializer.initialize(chain_query, 3)


# ---------------------------------------------------------------------------
# The exactness of the drawn distribution, statistically
# ---------------------------------------------------------------------------

def _observed_draws(query, reference, initializer, draws):
    """Draw the next population member repeatedly for a fixed reference set.

    The reference set is fixed on purpose: the exactness claim states the distribution of the next
    member *given* the members already drawn, and driving it through ``initialize`` would sample
    the first member as well and mix the conditional distributions of eleven different reference
    sets.

    Args:
        query: The generator query.
        reference (list): The members already drawn.
        initializer (KernelDiverseInitializer): The initializer under test.
        draws (int): How many draws to take.

    Returns:
        Counter: How often each inhabitant came out.
    """
    return Counter(
        initializer._draw_outside(query, list(reference), len(reference) + 1)
        for _ in range(draws)
    )


def test_v13_the_next_member_follows_the_cost_weight_on_the_chain(chain_query, by_length):
    """The next member is drawn in proportion to the cost weight, among the members not yet drawn.

    The weight of an inhabitant is ``pi`` at its cost value, divided by the number of inhabitants
    within the bound that realize that value.  On the chain every cost value is realized by
    exactly one term, so that number is one throughout and the target is ``pi`` restricted to the
    realized values and renormalized.  The control is the uniform draw: it is what an
    implementation that built the tree but ignored ``pi`` would produce, and the sharpness is
    picked so that every category still expects 180 draws or more.
    """
    reference = [by_length[5]]
    sharpness = 3.0
    knorm = normalized(k_st)
    inhabitants = chain_terms(_D)

    def cost(term):
        return reference_score(term, reference, knorm)

    weights = target_weights(inhabitants, cost, exponential_decay(sharpness)(reference))
    outside = {t: w for t, w in weights.items() if t not in set(reference)}

    initializer = KernelDiverseInitializer(
        k_st, size_bound=_D, rng=random.Random(20260731), pi=exponential_decay(sharpness)
    )
    observed = _observed_draws(chain_query, reference, initializer, _DRAWS)

    assert set(observed) <= set(outside), "a member of R_i was drawn"
    _, pvalue = chi_square_pvalue(observed, outside)
    assert pvalue > _THRESHOLD, f"the draw does not follow the cost weights (p={pvalue:.3g})"

    uniform = dict.fromkeys(outside, 1.0)
    assert_rejects(observed, uniform, _THRESHOLD)


def test_v13_the_division_by_the_cost_count_is_visible_where_terms_share_a_value():
    """The division by the number of terms sharing a cost value shows only where terms share one.

    The weight divides ``pi`` at a cost value by the number of inhabitants realizing it, and the
    chain holds one term per cost value, so that number is one there and an implementation that
    dropped the division would pass the test above unchanged.  The expression space shares cost
    values between up to seven terms, so the two distributions come apart, and the undivided one
    is exactly the control.
    """
    space = expression_space()
    query = generator_query(space, EXPR)
    bound = 6
    inhabitants = inhabitants_within(space, EXPR, EXPR_SIGNATURE, bound)
    reference = [inhabitants[0], inhabitants[4]]
    knorm = normalized(k_st)

    def cost(term):
        return reference_score(term, reference, knorm)

    # Written out here rather than taken from the module under test.  The chain test above builds
    # its target with the shipped `exponential_decay`, so target and implementation would move
    # together under a defect in it, and the suite's rule is that a target comes from the
    # definition by hand.  This one is decreasing and positive, which is all the kernel-diverse
    # initializer asks of its distribution.
    def handwritten_pi(value):
        return 2.0 ** -float(value)

    weights = target_weights(inhabitants, cost, handwritten_pi)
    outside = {t: w for t, w in weights.items() if t not in set(reference)}

    initializer = KernelDiverseInitializer(
        k_st,
        size_bound=bound,
        rng=random.Random(4711),
        pi=lambda _reference: handwritten_pi,
    )
    observed = _observed_draws(query, reference, initializer, _DRAWS)

    assert set(observed) <= set(outside)
    _, pvalue = chi_square_pvalue(observed, outside)
    assert pvalue > _THRESHOLD, f"the draw does not follow the cost weights (p={pvalue:.3g})"

    # pi without the division by the number of terms sharing a cost value: every such term would
    # carry the value's whole weight, so the values realized by many terms would be over-drawn.
    undivided = {t: handwritten_pi(cost(t)) for t in outside}
    assert_rejects(observed, undivided, _THRESHOLD)
