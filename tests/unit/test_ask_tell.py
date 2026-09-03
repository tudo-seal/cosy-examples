from __future__ import annotations

import numpy as np
import pytest
from cosy.core.tree import Tree
from cosy.search import SizeUniformSampler


def test_initialize_suggest_observe_round_trip(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    assert s.candidate is not None
    bo.observe(s.candidate, 0.3)
    snap = bo.get_state_snapshot()
    assert snap["state"] == "OBSERVED"
    assert snap["x_list"][-1] == s.candidate
    assert snap["y_list"][-1] == pytest.approx(0.3)


def test_multiple_iterations_accumulate_data(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    for y_val in [0.4, 0.35, 0.3]:
        s = bo.suggest()
        bo.observe(s.candidate, y_val)
    snap = bo.get_state_snapshot()
    assert len(snap["x_list"]) == 6  # Three initial plus three observed.
    assert len(snap["y_list"]) == 6


def test_y_list_contains_raw_values_after_each_observe(bo_factory, tree_corpus):
    bo = bo_factory()
    raw_values = [1.0, 2.0, 0.5]
    bo.initialize(x0=tree_corpus[:3], y0=raw_values)
    observed_y = 9.99
    s = bo.suggest()
    bo.observe(s.candidate, observed_y)
    snap = bo.get_state_snapshot()
    assert snap["y_list"][-1] == pytest.approx(observed_y)
    # The initial raw values are preserved.
    assert np.allclose(snap["y_list"][:3], raw_values)


def test_iteration_counter_increments_on_observe(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    assert bo.get_state_snapshot()["iteration"] == 0
    s = bo.suggest()
    bo.observe(s.candidate, 0.3)
    assert bo.get_state_snapshot()["iteration"] == 1
    s2 = bo.suggest()
    bo.observe(s2.candidate, 0.2)
    assert bo.get_state_snapshot()["iteration"] == 2


def test_suggestion_diagnostics_contain_all_required_fields(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    diag = s.diagnostics
    assert diag is not None
    required = {"timestamp", "incumbent", "iteration", "fallback_used", "fallback_attempts"}
    assert required.issubset(diag.keys())


def test_duplicate_candidate_triggers_fallback(monkeypatch, bo_factory, tree_corpus):
    """An optimizer that returns an already evaluated candidate triggers the fallback draw."""
    from bayesian_optimization import bo as bo_mod

    # Patch the fallback draw to return a fresh tree.
    fresh = Tree("fallback_fresh")
    call_count = {"n": 0}

    def fake_fallback(sampler, query, seen):
        call_count["n"] += 1
        return fresh

    monkeypatch.setattr(bo_mod, "_sample_fallback_tree", fake_fallback)

    # Use a corpus tree as initial x0 and have the optimizer return it as candidate.
    candidates = [tree_corpus[0]]  # In x0, so the fallback must trigger.
    bo = bo_factory(candidates=candidates)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    # This fixture has no search space, so initialize() built no sampler.  The stub stands in
    # for the one a real run draws its replacement from.
    bo._sampler = object()
    s = bo.suggest()
    assert call_count["n"] >= 1
    assert s.diagnostics["fallback_used"] is True


def test_a_fallback_that_returns_a_known_term_is_a_contract_violation(
    monkeypatch, bo_factory, tree_corpus
):
    """The replacement draw is not retried, it is trusted, and the trust is checked once.

    ``_sample_fallback_tree`` returns an inhabitant outside the observed set or raises, so one
    draw settles the duplicate.  There used to be a bounded retry loop here, whose second pass was
    unreachable for exactly that reason and whose error message named a state the code cannot be
    in.  What remains is the check that the contract held: a sampler that breaks it would
    otherwise hand a duplicate to the expensive quality measure, which is the one thing the
    rejection path exists to prevent.
    """
    from bayesian_optimization import bo as bo_mod

    always_known = tree_corpus[0]

    def fake_fallback(sampler, query, seen):
        return always_known  # Breaks the contract on purpose.

    monkeypatch.setattr(bo_mod, "_sample_fallback_tree", fake_fallback)

    candidates = [tree_corpus[0]]  # In x0, so the fallback triggers.
    bo = bo_factory(candidates=candidates)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    bo._sampler = object()
    with pytest.raises(RuntimeError, match="already been evaluated"):
        bo.suggest()


def test_duplicate_without_a_search_space_says_so(bo_factory, tree_corpus):
    """Without a space there is nothing to redraw from, and that is an error, not a duplicate.

    The ask/tell engine also drives optimizers that are not cosy's, and those callers hand in
    their own candidates.  When one of them repeats an evaluated point there is no sampler to
    replace it with, so the loop says which of the two is missing instead of failing on a None.
    """
    bo = bo_factory(candidates=[tree_corpus[0]])
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    with pytest.raises(RuntimeError, match="no search space"):
        bo.suggest()


def test_kernel_object_is_not_mutated_across_suggests(bo_factory, tree_corpus):
    """The kernel object a caller hands in survives every suggest unchanged.

    ``suggest()`` fits a copy and must not assign the fitted result back to ``self.kernel``.
    """
    from bayesian_optimization.kernels.tree_kernel import OrderedRootedSubtreeKernel

    kernel = OrderedRootedSubtreeKernel(normalize=True)
    original_id = id(kernel)
    original_params = kernel.get_params()

    bo = bo_factory(kernel=kernel)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    bo.observe(s.candidate, 0.3)
    bo.suggest()

    assert id(bo.kernel) == original_id
    assert bo.kernel.get_params() == original_params


def test_n_restarts_kernel_optimizer_is_not_decremented(bo_factory, tree_corpus):
    """``n_restarts_kernel_optimizer`` stays constant across suggests."""
    bo = bo_factory(n_restarts_kernel_optimizer=5)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    bo.observe(s.candidate, 0.3)
    bo.suggest()
    assert bo.n_restarts_kernel_optimizer == 5


def test_the_closed_loop_uses_user_x0_y0(bo_factory, tree_corpus):
    """The closed loop passes the caller's ``x0`` and ``y0`` on, it does not initialize on None."""
    bo = bo_factory()
    x0 = tree_corpus[:3]
    y0 = [1.0, 2.0, 0.5]
    result = bo.optimize(
        objective=lambda t: 0.3,
        budget=1,
        x0=x0,
        y0=y0,
    )
    # The result must contain the initial x0 points.
    all_x = list(result["x"])
    # The x0 trees must appear in the result.
    for t in x0:
        assert t in all_x, f"x0 tree {t} missing from result"


def test_search_space_none_no_special_path(bo_factory, tree_corpus):
    """A missing search space takes no branch of its own: the surrogate is always fitted."""
    bo = bo_factory()  # No search space.
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    # The acquisition value is set only when the Gaussian process was fitted.
    assert s.acquisition_value is not None


def test_finalize_returns_complete_result_dict(bo_factory, tree_corpus):
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    s = bo.suggest()
    bo.observe(s.candidate, 0.3)
    result = bo.finalize()
    assert "best_tree" in result
    assert "best_y" in result
    assert "x" in result
    assert "y" in result
    assert "gp_model" in result
    assert "iterations" in result
    assert "trace" in result
    assert "dropped_suggestion" in result
    assert result["iterations"] == 1


def test_best_returns_the_largest_observation(bo_factory, tree_corpus):
    """The loop maximizes, and a caller who minimizes negates the objective outside it."""
    bo = bo_factory()
    y0 = [5.0, 2.0, 8.0]
    bo.initialize(x0=tree_corpus[:3], y0=y0)
    best_tree, best_y = bo.best()
    assert best_y == pytest.approx(8.0)
    assert best_tree == tree_corpus[2]


# ---------------------------------------------------------------------------
# State that survived a reset where it should not have
# ---------------------------------------------------------------------------


def test_reset_drops_the_sampler_with_the_initializer(bo_factory, tree_corpus):
    """A second run starts from a fresh stream, not from where the first one left off.

    The two tests that stood here pinned the lifecycle of a fitted y transform: it estimated its
    statistics on first use and froze them, so a reset that kept it scaled the next run by the
    previous run's numbers.  With the transformation layer gone, the state that must not survive
    a reset is the sampler, which carries the position in its stream.
    """
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[100.0, 200.0, 300.0])
    bo.suggest()

    bo.reset()

    assert bo._sampler is None
    assert bo._initializer is None
    assert bo.get_state_snapshot()["y_list"] == []


def test_observe_records_the_candidate_the_optimizer_produced(bo_factory, tree_corpus):
    """``observe`` must store the tree ``suggest`` returned, not the caller's argument.

    The equality check is structural on purpose, since a caller may legitimately pass a
    reconstructed tree, but what goes into the observation set has to be the object the core
    produced, so the set stays keyed on exactly what was suggested.
    """
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    suggestion = bo.suggest()

    equal_but_distinct = Tree(suggestion.candidate.root, suggestion.candidate.children)
    assert equal_but_distinct == suggestion.candidate
    assert equal_but_distinct is not suggestion.candidate

    bo.observe(equal_but_distinct, 0.3)

    assert bo._x_list[-1] is suggestion.candidate
    assert suggestion.candidate in bo._x_set


def test_observe_records_nothing_for_a_suggestion_it_cannot_trace(bo_factory, tree_corpus):
    """A pass no trace row can be written for must leave the dataset as it was.

    The row is built after the term and the value are appended, so a check inside it runs too
    late.  The term and the value stay in the dataset, no row and no iteration count mention
    them, the state is still SUGGESTED, and the same call is accepted again and appends them a
    second time.  ``finalize()`` then answers with an optimum the trace has never seen.
    """
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 5.0, 2.0])
    suggestion = bo.suggest()
    assert suggestion.diagnostics is not None
    suggestion.diagnostics.pop("iteration")

    before = bo.get_state_snapshot()
    with pytest.raises(ValueError, match="missing iteration"):
        bo.observe(suggestion.candidate, 42.0)

    after = bo.get_state_snapshot()
    assert len(after["x_list"]) == len(before["x_list"])
    assert len(after["y_list"]) == len(before["y_list"])
    assert len(bo._x_set) == 3
    assert after["iteration"] == before["iteration"]
    assert after["state"] == "SUGGESTED"
    assert bo.trace == []
    assert bo.finalize()["best_y"] == pytest.approx(5.0)


def test_partial_diagnostics_are_refused_by_name(bo_factory, tree_corpus):
    """Diagnostics that hold no key a row is read from must say which keys those are.

    Every key of ``Diagnostics`` is optional, so an empty mapping is a well-typed one and a check
    against ``None`` lets it through.  What followed was a ``KeyError`` on the first column, which
    names one key and leaves the other four to be found one rerun at a time.
    """
    from bayesian_optimization.state import Suggestion

    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 5.0, 2.0])

    with pytest.raises(ValueError) as refused:
        bo._trace_record(
            Suggestion(candidate=Tree("t"), acquisition_value=0.25, diagnostics={}), 1.0
        )

    message = str(refused.value)
    for key in ("iteration", "mean_at_pick", "deviation_at_pick", "incumbent", "fallback_used"):
        assert key in message, f"the refusal does not name {key}: {message!r}"


def test_the_checked_keys_are_the_keys_a_row_is_read_from(bo_factory, tree_corpus):
    """Diagnostics holding nothing but the checked keys must still produce a row.

    The check walks a list of names while the row is assembled by subscript, so the two can
    disagree, and a check that has fallen behind passes exactly the mapping that then fails.  A
    row built from the checked keys alone is what keeps them together.
    """
    from bayesian_optimization.bo import _TRACE_DIAGNOSTICS_KEYS
    from bayesian_optimization.state import Suggestion

    checked = {
        "iteration": 3,
        "mean_at_pick": 0.5,
        "deviation_at_pick": 0.25,
        "incumbent": 5.0,
        "fallback_used": False,
    }
    assert set(checked) == set(_TRACE_DIAGNOSTICS_KEYS)

    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 5.0, 2.0])
    row = bo._trace_record(
        Suggestion(candidate=Tree("t"), acquisition_value=0.25, diagnostics=checked), 7.0
    )

    assert row.iteration == 3
    assert row.acquisition == pytest.approx(0.25)
    assert row.observed == pytest.approx(7.0)
    assert row.best == pytest.approx(7.0)


def test_a_missing_score_is_not_reported_as_missing_diagnostics(bo_factory, tree_corpus):
    """The two halves of a row are refused under their own names.

    One message covered both, so a suggestion whose diagnostics are complete and whose
    acquisition value is absent was reported as carrying no diagnostics, which sends the reader
    to the half that is there.
    """
    from bayesian_optimization.state import Suggestion

    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 5.0, 2.0])
    complete = {
        "iteration": 1,
        "mean_at_pick": 0.5,
        "deviation_at_pick": 0.25,
        "incumbent": 5.0,
        "fallback_used": False,
    }

    with pytest.raises(ValueError, match="carries no diagnostics"):
        bo._trace_record(
            Suggestion(candidate=Tree("t"), acquisition_value=0.25, diagnostics=None), 1.0
        )
    with pytest.raises(ValueError, match="carries no acquisition value"):
        bo._trace_record(
            Suggestion(candidate=Tree("t"), acquisition_value=None, diagnostics=complete), 1.0
        )


def test_the_fallback_draws_from_the_sampler_the_caller_named(bo_factory, tree_corpus):
    """One sampler serves the initial dataset and the duplicate replacement.

    They used to be separable: the loop built a size-uniform sampler from ``size_bound`` and
    ``counting`` for the fallback, while ``initializer`` could be handed in on its own.  A search
    space that admits one sampler and not another (the CNN space admits no counting sampler at any
    bound) then had a working initializer and a fallback that could not run on it, and the failure
    surfaced only when a duplicate happened to come up, hours into a run.

    Handing a sampler in must therefore reach *both*.  The stand-in below is not a size-uniform
    sampler and would not be mistaken for one.
    """
    class _OneTermSampler:
        """A sampler that yields a single term, so it is recognizable in the result."""

        def __init__(self, term):
            self.term = term
            self.asked = 0

        def sample(self, _query):
            """Yield the one term.

            Args:
                _query: Ignored.

            Yields:
                Tree: The term.
            """
            self.asked += 1
            yield self.term

    replacement = Tree("only_this_one")
    sampler = _OneTermSampler(replacement)

    # The optimizer returns a term that is already in the dataset, which is what makes the loop
    # reach for a replacement at all.
    bo = bo_factory(candidates=[tree_corpus[0]], sampler=sampler)
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    bo._sampler = sampler  # This fixture has no search space, so initialize() built none.

    suggestion = bo.suggest()

    assert suggestion.diagnostics["fallback_used"] is True
    assert suggestion.candidate == replacement, "the replacement came from somewhere else"
    assert sampler.asked == 1


def test_a_handed_in_sampler_reaches_both_roles_on_a_real_search_space(dummy_optimizer_factory):
    """The wiring itself, not a stand-in for it.

    The test above sets ``bo._sampler`` by hand, because its fixture has no search space, and a
    test that assigns the private attribute cannot notice if the public parameter never arrives.
    A mutant that ignored ``sampler`` outright survived both suites for exactly that reason.

    So this one builds a real space and checks the parameter's two documented roles: the initial
    dataset is drawn from it, and so is the duplicate replacement.  The stand-in sampler yields
    terms no other source would produce, which is what makes both observable.
    """
    import random

    from cosy.core import Synthesizer
    from cosy.core.types import Constructor

    from bayesian_optimization.bo import BayesianOptimization

    start = Constructor("A")
    space = Synthesizer(
        {"a": start, "b": start, "h": Constructor("A") ** start}, {}
    ).construct_solution_space(start).prune()

    class _CountingSampler:
        """A size-uniform sampler that records how often it was asked."""

        def __init__(self):
            self.inner = SizeUniformSampler(4, random.Random(0))
            self.asked = 0

        def sample(self, query):
            """Stream the inner sampler's draws and count the request.

            Args:
                query: The resolution query.

            Yields:
                Tree: The drawn inhabitants.
            """
            self.asked += 1
            yield from self.inner.sample(query)

        def at_least(self, query, count):
            """Delegate the population-size check.

            Args:
                query: The resolution query.
                count (int): How many inhabitants are needed.

            Returns:
                bool: Whether the bounded space holds that many.
            """
            return self.inner.at_least(query, count)

    sampler = _CountingSampler()
    bo = BayesianOptimization(
        space, start, optimizer=dummy_optimizer_factory([]), sampler=sampler,
        n_restarts_kernel_optimizer=0,
    )
    bo.initialize(objective=lambda tree: float(tree.size), initial_size=3)

    assert bo._sampler is sampler, "the parameter must reach the loop's own draws"
    assert sampler.asked >= 1, "the initial dataset must come from it"
    assert len(bo.get_state_snapshot()["x_list"]) == 3


def test_the_loop_poses_one_query_object_for_the_whole_run(dummy_optimizer_factory):
    """The query is one object, and that is what makes the sampler's own cache reachable.

    ``SizeUniformSampler`` keys its counting construction by query *identity*, and it does so
    deliberately: comparing a partial-term query structurally would cost more than the lookup
    saves.  So a loop that mints a fresh query per call pays the counting again at every call
    site, and on the determinized CNN search space of the CIFAR-10 experiment that is 93 s a time
    against 0.04 s per draw once it is built.

    Nothing about the *value* of the query changes, which is why this is asserted on identity:
    two ``generator_query(space, request)`` results are equal and the failure would be invisible
    to any comparison but this one.
    """
    from cosy.core import Synthesizer
    from cosy.core.types import Constructor

    from bayesian_optimization.bo import BayesianOptimization

    start = Constructor("A")
    space = Synthesizer(
        {"a": start, "h": Constructor("A") ** start}, {}
    ).construct_solution_space(start).prune()

    bo = BayesianOptimization(space, start, optimizer=dummy_optimizer_factory([]))

    first = bo._query()
    assert first is bo._query(), "a second call minted a second object"
    assert first.solution_space is space
    assert first.start is start


def test_a_loop_without_a_search_space_still_has_no_query(bo_factory):
    """The caching must not turn "there is nothing to pose" into an object.

    The ask/tell engine also drives optimizers that are not cosy's, and those callers do not read
    the query.  ``None`` is what every path that needs a space checks for.
    """
    bo = bo_factory()
    assert bo.search_space is None
    assert bo._query() is None
    assert bo._query() is None


def test_surrogate_over_conditions_on_what_it_was_given(bo_factory, tree_corpus):
    """The held-out fit read needs a surrogate that has not seen the terms it predicts.

    Conditioned on them, a noise-free Gaussian process reproduces its training values, and the
    scatter of predicted against true values would sit on the diagonal of equality whatever the
    kernel does.  So this is the property that matters: the returned model was fitted on the given
    half and no more.
    """
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:6], y0=[1.0, 2.0, 0.5, 3.0, 1.5, 2.5])

    surrogate = bo.surrogate_over(tree_corpus[0:6:2], [1.0, 0.5, 1.5])

    assert len(surrogate.X_train_) == 3
    assert set(surrogate.X_train_.ravel().tolist()) == set(tree_corpus[0:6:2])
    # The whole dataset is still six pairs, and asking for a subset must not disturb the run.
    assert len(bo.get_state_snapshot()["x_list"]) == 6


def test_surrogate_over_refuses_the_three_ways_it_can_be_asked_wrong(bo_factory, tree_corpus):
    """Mismatched lengths, nothing to condition on, and a repeated term.

    The third is the one worth naming: the loop conditions the surrogate on the *distinct* pairs
    of its dataset, and a term handed in twice carries no second piece of information, only the
    risk of two different values for one term, which is the case the loop itself refuses.
    """
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])

    with pytest.raises(ValueError, match="by position"):
        bo.surrogate_over(tree_corpus[:2], [1.0])
    with pytest.raises(ValueError, match="nothing to condition"):
        bo.surrogate_over([], [])
    with pytest.raises(ValueError, match="repeat"):
        bo.surrogate_over([tree_corpus[0], tree_corpus[0]], [1.0, 2.0])


def test_a_search_space_the_request_does_not_name_is_refused():
    """A broken pair of space and request blames the sampler, so it is refused where it is given.

    The query is built from the two together, and a non-terminal the space has no rules for gives
    a query that is answered with the empty stream.  What the caller then sees is the initializer
    reporting fewer inhabitants than it asked for inside the bound of the sampler, which sends
    them to widen a bound that is not the problem.  The two ways to break the pair, the default
    ``None`` and a non-terminal of another space, fail through that same message, so the check is
    a membership test and catches both.
    """
    from cosy.core import Synthesizer
    from cosy.core.types import Constructor

    from bayesian_optimization.bo import BayesianOptimization

    start = Constructor("A")
    space = Synthesizer(
        {"a": start, "h": Constructor("A") ** start}, {}
    ).construct_solution_space(start).prune()

    with pytest.raises(ValueError, match="no rules for the request"):
        BayesianOptimization(space)
    with pytest.raises(ValueError, match="no rules for the request"):
        BayesianOptimization(space, Constructor("B"))


def test_the_request_is_checked_against_the_space_and_not_against_none():
    """Four configurations a check on the request has to leave standing.

    The ordinary one is a space together with the non-terminal it was synthesized for.  Without a
    search space the loop poses no query at all, so ``None`` is the setting there, and a request
    handed in beside no space is merely unused.  The fourth is the one that decides the shape of
    the check: a non-terminal is anything hashable, so a space whose non-terminal *is* ``None`` is
    queried at ``None`` and draws terms, and a check written as ``request is None`` would refuse
    it.
    """
    import random
    from collections import deque
    from itertools import islice

    from cosy.core import Synthesizer
    from cosy.core.solution_space import RHSRule, SolutionSpace
    from cosy.core.types import Constructor

    from bayesian_optimization.bo import BayesianOptimization

    start = Constructor("A")
    space = Synthesizer(
        {"a": start, "h": Constructor("A") ** start}, {}
    ).construct_solution_space(start).prune()

    assert BayesianOptimization(space, start).request is start
    assert BayesianOptimization(None).request is None
    assert BayesianOptimization(None, start).request is start

    queried_at_none = SolutionSpace(
        {None: deque([RHSRule(arguments=(), predicates=(), terminal="a")])}
    )
    bo = BayesianOptimization(queried_at_none, None)
    assert bo.query.start is None
    drawn = list(islice(SizeUniformSampler(3, random.Random(0)).sample(bo.query), 1))
    assert drawn, "the query names a non-terminal of the space and has to be live"


def test_a_space_assigned_after_construction_is_still_checked():
    """A search space and a request are checked again where they are turned into a query.

    Both are public attributes, so the pair the constructor read is not necessarily the pair the
    query is built from, and a check that only reads the constructor arguments is a courtesy
    and not a guarantee.  Assigning a real space to a loop that was built without one is the path
    that used to hand back a query naming no non-terminal, from which every draw is empty, and
    report nothing.  The repaired pair is pinned beside it, because a check placed here must let
    a query through once the two agree.
    """
    import random
    from itertools import islice

    from cosy.core import Synthesizer
    from cosy.core.types import Constructor

    from bayesian_optimization.bo import BayesianOptimization

    start = Constructor("A")
    space = (
        Synthesizer({"a": start, "h": Constructor("A") ** start}, {})
        .construct_solution_space(start)
        .prune()
    )

    bo = BayesianOptimization(None)
    bo.search_space = space
    with pytest.raises(ValueError, match="no rules for the request"):
        _ = bo.query

    bo.request = start
    drawn = list(islice(SizeUniformSampler(3, random.Random(0)).sample(bo.query), 1))
    assert drawn, "the pair agrees now, and the query has to be live"
