"""Random search as a strategy of its own, through the same ask/tell loop as Bayesian optimization.

The random-search arm used to exist only beside a Bayesian run: the paired baseline of the CIFAR
driver drew ``n + m`` terms from one stream of the loop's sampler, the first ``n`` shared as the
design and the other ``m`` evaluated after the loop.  ``RandomSearch`` is that arm as a strategy:
design and passes come from one stream of its sampler, so a run of ``n`` design terms and ``m``
passes evaluates exactly the first ``n + m`` distinct terms of that stream.  It fits no surrogate
and maximizes no acquisition.
"""

from __future__ import annotations

import random

import pytest
from cosy.core.tree import Tree
from cosy.search import DepthBoundedRandomSampler, SizeUniformSampler

from bayesian_optimization import RandomSearch
from bayesian_optimization.initial_sampling import distinct_prefix
from tests.spaces import EXPR, LIST, expression_space, list_space


def _sampler(seed, bound=6):
    return SizeUniformSampler(bound, random.Random(seed))


def _search(seed, bound=6):
    return RandomSearch(list_space(), LIST, sampler=_sampler(seed, bound), seed=seed)


def _run(search, n_design, n_passes):
    """The design phase, then the passes, every value the term's printed length."""
    evaluated = []
    search.initialize(initial_size=n_design)
    for _ in range(n_design + n_passes):
        suggestion = search.suggest()
        evaluated.append(suggestion.candidate)
        search.observe(suggestion.candidate, float(len(str(suggestion.candidate))))
    return evaluated


def _stream_head(seed, count, bound=6):
    """What the paired baseline drew: ``count`` distinct terms from one stream of a fresh sampler."""
    reference = _search(seed, bound)
    terms, repeats = distinct_prefix(_sampler(seed, bound), reference.query, count)
    assert repeats == 0
    return terms


def test_random_search_evaluates_the_first_distinct_terms_of_one_stream():
    """The equivalence the strategy is built on: design and passes are one sample."""
    evaluated = _run(_search(seed=11), n_design=3, n_passes=4)

    assert evaluated == _stream_head(seed=11, count=7)
    assert len(set(evaluated)) == 7
    assert evaluated != _run(_search(seed=12), n_design=3, n_passes=4), "the seed has to matter"


def test_random_search_fits_nothing_and_maximizes_nothing():
    search = _search(seed=2)
    search.initialize(initial_size=2)
    passes = []
    for index in range(4):
        suggestion = search.suggest()
        if index >= 2:
            passes.append(suggestion)
        search.observe(suggestion.candidate, 1.0)

    for suggestion in passes:
        assert suggestion.acquisition_value is None
        assert suggestion.diagnostics is not None
        assert suggestion.diagnostics["phase"] == "main"
    result = search.finalize()
    assert result["gp_model"] is None
    assert result["trace"] == []
    assert result["iterations"] == 2
    assert len(result["x"]) == 4


def test_the_passes_after_a_handed_over_design_continue_the_stream_past_it():
    """The shape of a paired comparison: the design is shared, and the passes skip what it holds."""
    head = _stream_head(seed=5, count=6)
    search = _search(seed=5)
    search.initialize(design=head[:3])
    for term in head[:3]:
        suggestion = search.suggest()
        assert suggestion.candidate == term
        search.observe(term, 1.0)

    passes = []
    for _ in range(3):
        suggestion = search.suggest()
        passes.append(suggestion.candidate)
        search.observe(suggestion.candidate, 1.0)

    assert passes == head[3:]
    assert search.terms_skipped == 3, "the three design terms at the head of the stream are skipped"


def test_a_sampler_that_repeats_is_skipped_and_counted():
    """Depth-bounded draws are independent and repeat; the search evaluates each term once."""
    search = RandomSearch(
        expression_space(), EXPR, sampler=DepthBoundedRandomSampler(4, random.Random(0)), seed=0
    )
    evaluated = _run(search, n_design=8, n_passes=8)

    assert len(set(evaluated)) == 16
    assert search.initial_repeats_rejected + search.terms_skipped > 0, (
        "this sampler has to repeat for the count to mean anything"
    )


def test_an_exhausted_space_is_a_stop_and_not_a_repeat():
    """The lists of length at most 1 are four terms; a fifth pass has nothing new to draw."""
    search = _search(seed=0, bound=2)
    evaluated = _run(search, n_design=2, n_passes=2)
    assert len(set(evaluated)) == 4

    with pytest.raises(RuntimeError, match="stream"):
        search.suggest()


def test_random_search_needs_a_space_to_draw_from():
    search = RandomSearch(None)
    with pytest.raises(NotImplementedError, match="requires a real search_space"):
        search.initialize(initial_size=2)


def test_the_closed_initialize_draws_its_design_from_the_same_stream():
    """``initialize(objective=...)`` evaluates the head of the stream, and the passes continue it."""
    search = _search(seed=13)
    search.initialize(objective=lambda term: 1.0, initial_size=3)
    passes = []
    for _ in range(2):
        suggestion = search.suggest()
        passes.append(suggestion.candidate)
        search.observe(suggestion.candidate, 1.0)

    head = _stream_head(seed=13, count=5)
    assert search.get_state_snapshot()["x_list"] == head
    assert passes == head[3:]


class _ScriptedSampler:
    """A sampler whose every stream plays one script from its start, counting the streams opened.

    ``repeat_last`` makes the script end in an endless repetition of its last term, the way a
    sampler that draws with replacement can keep returning a term that is already held.
    """

    def __init__(self, terms, repeat_last=False):
        self.terms = list(terms)
        self.repeat_last = repeat_last
        self.streams = 0
        self.drawn = 0

    def sample(self, query):
        self.streams += 1
        for term in self.terms:
            self.drawn += 1
            yield term
        while self.repeat_last:
            self.drawn += 1
            yield self.terms[-1]


def _scripted(terms, **kwargs):
    sampler = _ScriptedSampler(terms, **kwargs)
    return RandomSearch(list_space(), LIST, sampler=sampler), sampler


LEAVES = [Tree(name) for name in "abcdefghij"]


def test_reset_closes_the_stream_and_the_next_run_opens_a_new_one():
    # The script repeats a design term among the passes, so the run skips one term before reset.
    script = [LEAVES[0], LEAVES[1], LEAVES[2], LEAVES[2], LEAVES[3], LEAVES[4]]
    search, sampler = _scripted(script)
    first = _run(search, n_design=3, n_passes=1)
    assert search.terms_skipped == 1
    search.reset()

    assert search.terms_skipped == 0
    assert search.design == ()
    second = _run(search, n_design=3, n_passes=1)
    assert sampler.streams == 2, "the run after the reset did not open a stream of its own"
    assert second == first == LEAVES[:4]


def test_a_long_handed_over_design_is_not_mistaken_for_exhaustion():
    """The design terms at the head of the stream are skipped once each, which is not a repeat.

    A bound on repeats in a row that counted them would end a paired comparison with a design of
    a hundred terms or more the moment its passes begin.
    """
    head = _stream_head(seed=21, count=125)
    search = _search(seed=21)
    search.initialize(design=head[:120])
    for term in head[:120]:
        search.observe(search.suggest().candidate, 1.0)

    passes = []
    for _ in range(5):
        suggestion = search.suggest()
        passes.append(suggestion.candidate)
        search.observe(suggestion.candidate, 1.0)

    assert passes == head[120:]
    assert search.terms_skipped == 120


def test_the_repeat_bound_stops_a_stream_that_only_repeats():
    """A sampler that keeps returning a held term is given up on after the bound, not looped on."""
    sampler = _ScriptedSampler([Tree("a")], repeat_last=True)
    search = RandomSearch(list_space(), LIST, sampler=sampler, max_draws_per_term=5)
    search.initialize(initial_size=1)
    search.observe(search.suggest().candidate, 1.0)

    with pytest.raises(RuntimeError, match="repeated"):
        search.suggest()
    assert sampler.drawn == 1 + 5, "the bound is the number of repeats in a row, no more, no less"


def test_a_pass_never_repeats_an_earlier_pass():
    search, _ = _scripted([Tree("d0"), Tree("d1"), Tree("p0"), Tree("p0"), Tree("p1")])
    evaluated = _run(search, n_design=2, n_passes=2)

    assert evaluated == [Tree("d0"), Tree("d1"), Tree("p0"), Tree("p1")]
    assert search.terms_skipped == 1


def test_the_two_repeat_counters_are_kept_apart():
    """A repeat inside the design is the design's count; a repeat among the passes is theirs."""
    search, _ = _scripted([Tree("a"), Tree("a"), Tree("b"), Tree("c"), Tree("c"), Tree("d")])
    _run(search, n_design=2, n_passes=2)

    assert search.initial_repeats_rejected == 1
    assert search.terms_skipped == 1


def test_a_failed_closed_initialize_leaves_a_fresh_stream_for_the_retry():
    """The retry draws the head of a new stream, not what the failed attempt left of the old one."""
    search, sampler = _scripted(LEAVES)

    def fails_on_the_second(term):
        if term == LEAVES[1]:
            raise RuntimeError("the training diverged")
        return 1.0

    with pytest.raises(RuntimeError, match="diverged"):
        search.initialize(objective=fails_on_the_second, initial_size=3)
    search.initialize(initial_size=3)

    assert list(search.design) == LEAVES[:3]
    assert sampler.streams == 2


def test_a_negative_design_size_is_refused():
    search, _ = _scripted(LEAVES)
    with pytest.raises(ValueError, match="non-negative"):
        search.initialize(initial_size=-1)


def test_random_search_without_a_space_cannot_propose_a_pass():
    """A design handed over needs no space; the first pass does, since it has to draw."""
    search = RandomSearch(None)
    search.initialize(design=[Tree("a")])
    search.observe(search.suggest().candidate, 1.0)

    with pytest.raises(NotImplementedError, match="requires a real search_space"):
        search.suggest()
