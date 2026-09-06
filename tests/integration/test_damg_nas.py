"""Every combinator of the DAMG repository builds a network, and the example that runs them works.

The file is built the way ``tests/test_cnn_damg_networks.py`` is built next door, for the older of
the two architecture search examples. Each test asks for a fully concrete structure, one with no
wildcard left in it, so synthesis has nothing to search for and hands back the term that was
described. A test that enumerated the space and filtered it for a match would also pass on a
repository that reaches the term by some other route, and it could not say which combinator carried
it.

Three concerns, in this order. The combinator table: every leaf combinator, the parallel
composition, the identity lane and the sequential composition interpret at least once and produce a
forward pass of the expected shape, and the loss, the optimizer and the learner are exercised
together by one short training call. The two silent failures the example can walk into: an epoch
count the repository does not offer, which empties the space and is then reported as a bound that is
too tight, and the sampler choice, which decides what a run costs before a single network is
trained. And the example itself, run end to end on a target small enough for a test.

The networks here are tiny and every training call runs on tensors drawn on the spot, so nothing in
this file reads a dataset. ``damg_example.main`` on ``target_len_3`` is the full example, and the
last test calls the same entry point on a one-lane target at one epoch, with a budget of two over
an initial design of three.
"""

from __future__ import annotations

import itertools
import math
import random
import time

import pytest
import torch
from cosy.core import Synthesizer
from cosy.core.types import Constructor, Literal
from cosy.evolutionary_algorithms import InitializationError, SampledInitialization
from cosy.search import samplers
from cosy.search.queries import generator_query
from cosy.search.samplers import DepthBoundedRandomSampler, SizeUniformSampler
from torch import nn, optim

from bayesian_optimization import BayesianOptimization
from bayesian_optimization.examples.damg_nas import damg_example
from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository
from bayesian_optimization.examples.damg_nas.damg_repo_algebras import (
    learner as raw_learner,
)
from bayesian_optimization.examples.damg_nas.damg_repo_algebras import (
    pretty_term_algebra,
    pytorch_function_algebra,
    pytorch_model_algebra,
)
from bayesian_optimization.examples.damg_nas.damg_targets import target_len_2, target_len_3

pytestmark = pytest.mark.integration

#: One epoch, so that a target of this file is inhabited by terms whose training call is short. The
#: shipped targets ask for two thousand, and the repository has to offer exactly what a target asks
#: for, which is what ``damg_example.epochs_of`` is for.
EPOCHS = 1


@pytest.fixture(scope="module")
def repository() -> DAMGrepository:
    """Build the repository of the example, with its epoch count reduced to one.

    Returns:
        DAMGrepository: The repository.
    """
    return DAMGrepository(
        linear_feature_dimensions=[1, 2, 3, 4, 5],
        constant_values=[0, 1, -1],
        learning_rate_values=[1e-2],
        n_epoch_values=[EPOCHS],
    )


@pytest.fixture(scope="module")
def specification(repository: DAMGrepository):
    """Build the specification once for the whole module.

    Args:
        repository (DAMGrepository): The repository.

    Returns:
        dict: The specification, a map from a combinator name to its clause.
    """
    return repository.specification()


def _target(n_in: int, n_out: int, structure: tuple, epochs: int = EPOCHS):
    """Build a target that pins the structure and leaves the loss and the optimizer open.

    The structure is what these tests are about, so the two remaining slots stay wildcards and
    whichever loss and optimizer the enumeration offers first is taken. That is sound here because
    only one test below trains the term it synthesizes, and that one reads the loss it got.

    Args:
        n_in (int): The width of the input.
        n_out (int): The width of the output.
        structure (tuple): The structure literal, a tuple of stages of triples.
        epochs (int): The epoch count, which the repository has to offer. (Default value = 1)

    Returns:
        Type: The target.
    """
    return Constructor("Learner", Constructor("DAG",
                                              Constructor("input", Literal(n_in))
                                              & Constructor("output", Literal(n_out))
                                              & Constructor("structure", Literal(structure)))
                       & Constructor("Loss", Constructor("type", Literal(None)))
                       & Constructor("Optimizer", Constructor("type", Literal(None)))
                       & Constructor("epochs", Literal(epochs)))


def _synthesize_one(specification: dict, target, max_seconds: float = 5.0):
    """Enumerate a wildcard-free target, which has to resolve quickly.

    There is almost nothing left to search for, so a slow search here says that the structure
    literal was not concrete after all, and the bound is what turns that into a failure.

    Args:
        specification (dict): The specification.
        target (Type): The requested type.
        max_seconds (float): The bound on the search. (Default value = 5.0)

    Returns:
        Tree: The first term of the space.
    """
    started = time.time()
    space = Synthesizer(specification, {}).construct_solution_space(target).prune()
    # islice and not max_count=1. Under max_count the enumeration stops when the *next* new term
    # for the start symbol arrives, so it always produces one more than it returns, and the bound
    # is passed on to the generation of each rule's combinations as well. On the target_len_3
    # space, max_count=1 does not finish in two minutes where islice returns the same term in
    # 0.02 seconds.
    terms = list(itertools.islice(space.enumerate_trees(target), 1))
    seconds = time.time() - started
    assert terms, f"expected the concrete target to be directly synthesizable, found none: {target}"
    assert seconds < max_seconds, (
        f"synthesizing a fully concrete target took {seconds:.2f}s, so the structure literal is "
        f"not actually concrete"
    )
    return terms[0]


# ---------------------------------------------------------------------------
# 1. The leaf combinators, each in a structure of one component.
# ---------------------------------------------------------------------------

#: (name, label, input width, output width). ``edges`` is missing on purpose and has its own test:
#: it is ID-typed, so it is a lane of a parallel composition and never a stage by itself.
_LEAF_CASES = [
    ("linear_layer", DAMGrepository.Linear(in_features=4, out_features=2, bias=True), 4, 2),
    ("sigmoid", DAMGrepository.Sigmoid(), 3, 3),
    ("relu", DAMGrepository.ReLu(inplace=False), 3, 3),
    ("tanh", DAMGrepository.Tanh(), 3, 3),
    ("sum", DAMGrepository.Sum(with_constant=0), 4, 1),
    ("product", DAMGrepository.Product(with_constant=1), 4, 1),
    ("copy", DAMGrepository.Copy(out_dimension=3), 1, 3),
    ("swap", ("swap", 2, 3), 5, 5),
]


@pytest.mark.parametrize("case", _LEAF_CASES, ids=lambda case: case[0])
def test_every_leaf_combinator_interprets_into_a_forward_pass_of_the_declared_width(
    specification, case
):
    """A structure of one component gives a network whose output has the width the target names."""
    name, label, n_in, n_out = case
    term = _synthesize_one(specification, _target(n_in, n_out, (((label, n_in, n_out),),)))

    model = term.interpret(pytorch_model_algebra())
    output = model(torch.randn(3, n_in))
    assert output.shape == (3, n_out), (
        f"{name}: expected an output of shape (3, {n_out}), got {tuple(output.shape)}"
    )


def test_a_swap_exchanges_the_two_blocks_of_its_input(specification):
    """``swap(n, m)`` moves the first ``n`` lanes behind the next ``m``, and is not a pass-through.

    The width alone cannot tell a swap from an identity, since a swap preserves it, so the test
    compares against the permutation itself.
    """
    term = _synthesize_one(specification, _target(5, 5, ((((("swap", 2, 3)), 5, 5),),)))

    model = term.interpret(pytorch_model_algebra())
    x = torch.randn(3, 5)
    y = model(x)
    expected = torch.cat((x[:, 2:], x[:, :2]), dim=-1)
    assert torch.allclose(y, expected), "the swap did not exchange the two blocks of its input"


def test_edges_is_an_identity_lane_and_not_a_stage_of_its_own(specification):
    """``edges`` passes its lanes through, and it is admissible only beside another component.

    It is ID-typed, so a stage consisting of it alone asks for a non-ID component and gets nothing.
    Both halves are checked: the standalone target is uninhabited, and beside a linear layer the
    lane comes out of the network untouched.
    """
    alone = _target(4, 4, (((("swap", 0, 4), 4, 4),),))
    space = Synthesizer(specification, {}).construct_solution_space(alone).prune()
    assert list(itertools.islice(space.enumerate_trees(alone), 1)) == [], (
        "edges is ID-typed, so a stage made of it alone must stay uninhabited"
    )

    linear = DAMGrepository.Linear(in_features=2, out_features=2, bias=True)
    term = _synthesize_one(
        specification, _target(4, 4, (((linear, 2, 2), (("swap", 0, 2), 2, 2)),))
    )
    assert "edges(" in term.interpret(pretty_term_algebra())

    model = term.interpret(pytorch_model_algebra())
    x = torch.randn(3, 4)
    y = model(x)
    assert y.shape == (3, 4)
    assert torch.allclose(y[:, 2:], x[:, 2:]), "the edges lane did not pass its input through"


def test_beside_cons_runs_its_two_lanes_on_their_own_halves_of_the_input(specification):
    """A parallel composition splits the input, runs one lane per half and concatenates.

    Two different labels, so the test can tell a genuine parallel composition from a single
    component that happens to have the right width: each half of the output has to equal that
    lane's own module applied to that half of the input.
    """
    left = DAMGrepository.Linear(in_features=2, out_features=2, bias=True)
    right = DAMGrepository.Sigmoid()
    term = _synthesize_one(specification, _target(4, 4, (((left, 2, 2), (right, 2, 2)),)))

    pretty = term.interpret(pretty_term_algebra())
    assert "||" in pretty, f"expected a parallel composition, got {pretty}"

    model = term.interpret(pytorch_model_algebra())
    x = torch.randn(3, 4)
    y = model(x)
    assert y.shape == (3, 4)
    # The right lane is a sigmoid, which has no parameters, so its half can be recomputed here.
    assert torch.allclose(y[:, 2:], torch.sigmoid(x[:, 2:])), (
        "the second lane did not receive the second half of the input"
    )


def test_before_cons_feeds_the_first_stage_into_the_second(specification):
    """A sequential composition of two stages chains their dimensions and their computation.

    The intermediate width, three here, appears in neither the input nor the output of the target,
    so a network that dropped one of the two stages could not have the shape this asserts.
    """
    first = DAMGrepository.Linear(in_features=4, out_features=3, bias=True)
    second = DAMGrepository.Linear(in_features=3, out_features=2, bias=True)
    term = _synthesize_one(
        specification, _target(4, 2, (((first, 4, 3),), ((second, 3, 2),)))
    )

    pretty = term.interpret(pretty_term_algebra())
    assert ";" in pretty, f"expected a sequential composition, got {pretty}"

    model = term.interpret(pytorch_model_algebra())
    y = model(torch.randn(3, 4))
    assert y.shape == (3, 2)

    linears = [m for m in model.modules() if type(m).__name__ == "SynthLinear"]
    assert len(linears) == 2, f"expected two linear layers in the composed network, got {linears}"


def test_the_learner_trains_the_network_and_reports_the_loss_it_reached(specification):
    """The loss, the optimizer and the learner together turn a term into a training run.

    ``pytorch_function_algebra`` is the algebra the example optimizes through, and this is the
    test that holds what it returns against a second reading of the same run, so it is the one that
    shows the training loop descends rather than only that it returns a number.

    Two calls rather than one, because a single finite number cannot tell training from a forward
    pass: the learner is called on the same network at zero epochs and at fifty, on data a linear
    layer can fit, and the second loss has to be the smaller one. Zero epochs is the same code path
    with an empty loop, so what the comparison isolates is the optimizer stepping.
    """
    linear = DAMGrepository.Linear(in_features=4, out_features=1, bias=True)
    term = _synthesize_one(specification, _target(4, 1, (((linear, 4, 1),),)))

    train = term.interpret(pytorch_function_algebra())
    torch.manual_seed(0)
    x = torch.randn(40, 4)
    y = x @ torch.tensor([1.0, -2.0, 0.5, 3.0])
    loss = train(x, y, x, y)
    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss)), f"the training run reported {loss}"
    assert loss >= 0.0, "both losses this repository offers are non-negative"

    torch.manual_seed(0)
    model = term.interpret(pytorch_model_algebra())
    untrained = raw_learner(4, model, nn.MSELoss(),
                            lambda m: optim.Adam(m.parameters(), lr=1e-2), 0, x, y, x, y)
    torch.manual_seed(0)
    model = term.interpret(pytorch_model_algebra())
    trained = raw_learner(4, model, nn.MSELoss(),
                          lambda m: optim.Adam(m.parameters(), lr=1e-2), 50, x, y, x, y)
    assert trained < untrained, (
        f"fifty epochs left the loss at {trained} against {untrained} untrained, so the optimizer "
        f"never stepped"
    )


# ---------------------------------------------------------------------------
# 2. The two silent failures.
# ---------------------------------------------------------------------------

def test_an_epoch_count_the_repository_does_not_offer_empties_the_space_without_saying_so():
    """A target and a repository that disagree about the epoch count leave nothing to search.

    This is the first of the two traps, and the point of the test is what the failure looks like
    rather than that it happens. The pruned space has no non-terminals at all, the sampler draws
    nothing, and the initializer of an evolutionary run reports fewer inhabitants than were asked
    for and offers to widen its bound, which is the one repair that cannot help. Nothing anywhere
    names the epoch count.
    """
    repo = DAMGrepository(
        linear_feature_dimensions=[1, 2, 3, 4, 5],
        constant_values=[0, 1, -1],
        learning_rate_values=[1e-2],
        n_epoch_values=[100],  # target_len_3 asks for 2000
    )
    space = Synthesizer(repo.specification(), {}).construct_solution_space(target_len_3).prune()

    assert tuple(space.nonterminals()) == ()
    assert target_len_3 not in space

    query = generator_query(space, target_len_3)
    sampler = DepthBoundedRandomSampler(100, random.Random(0))
    assert list(itertools.islice(sampler.sample(query), 3)) == []

    with pytest.raises(InitializationError) as raised:
        SampledInitialization(sampler).initialize(query, 3)
    message = str(raised.value)
    assert "fewer than 3 inhabitants" in message
    assert "Widen the bound" in message
    assert "epoch" not in message


def test_the_loop_refuses_the_mismatched_pair_and_the_example_cannot_build_one():
    """Two guards stand between the trap and a run, and both are checked here.

    ``BayesianOptimization`` refuses a request its space has no rules for, which turns the empty
    space into a message that names the request. And ``damg_example`` reads the epoch count off the
    target instead of being told it a second time, so the pair it builds always agrees.
    """
    repo = DAMGrepository(
        linear_feature_dimensions=[1, 2, 3, 4, 5],
        constant_values=[0, 1, -1],
        learning_rate_values=[1e-2],
        n_epoch_values=[100],
    )
    space = Synthesizer(repo.specification(), {}).construct_solution_space(target_len_3).prune()
    with pytest.raises(ValueError, match="no rules for the request"):
        BayesianOptimization(space, target_len_3)

    assert damg_example.epochs_of(target_len_3) == 2000
    assert damg_example.build_repository(target_len_3).n_epoch_values == [2000]
    small = _target(1, 1, ((None,),), epochs=7)
    assert damg_example.build_repository(small).n_epoch_values == [7]


def _sampler_costs(target):
    """Time three depth-bounded draws, three size-uniform draws and ten more of the latter.

    Args:
        target (Type): The requested type.

    Returns:
        tuple: The three durations in seconds, in that order.
    """
    repo = damg_example.build_repository(target)
    space = damg_example.build_search_space(repo, target, verbose=False)
    query = generator_query(space, target)

    depth_bounded = DepthBoundedRandomSampler(100, random.Random(0))
    started = time.time()
    assert len(list(itertools.islice(depth_bounded.sample(query), 3))) == 3
    depth_seconds = time.time() - started

    size_uniform = SizeUniformSampler(100, random.Random(0))
    started = time.time()
    assert len(list(itertools.islice(size_uniform.sample(query), 3))) == 3
    first_three_seconds = time.time() - started

    started = time.time()
    assert len(list(itertools.islice(size_uniform.sample(query), 10))) == 10
    next_ten_seconds = time.time() - started

    return depth_seconds, first_three_seconds, next_ten_seconds


def test_the_size_uniform_sampler_counts_the_space_once_and_the_other_never(monkeypatch):
    """The second trap: which sampler the loop draws from decides what a run costs.

    ``DAMGrepository`` states its four swap laws as predicates over two sibling holes, so a
    size-uniform draw cannot count from the program and has to walk the retained search tree
    instead. That walk is the whole cost of the sampler, and this counts the walks rather than
    timing them, so the assertion says what happens and not how fast this machine is.

    Three facts, and the third is the one a reader gets wrong. The depth-bounded sampler never
    counts. The size-uniform sampler counts on its first draw. And the construction is kept by the
    sampler, for the last query it was given, so ten further draws from that query build nothing, an
    initial design of ten costs what one of three costs, and ``forget`` is what gives the memory
    back at the price of counting again.

    Run on ``target_len_2``, whose terms have 58 symbols. On ``target_len_3``, which the example
    searches, that one walk takes tens of seconds, and the test below holds it.
    """
    walks = []
    original = samplers.weighted_tree

    def counting_weighted_tree(*args, **kwargs):
        """Record one walk of the retained search tree and perform it.

        Args:
            *args: Passed through.
            **kwargs: Passed through.

        Returns:
            The weighted tree.
        """
        walks.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(samplers, "weighted_tree", counting_weighted_tree)

    repo = damg_example.build_repository(target_len_2)
    space = damg_example.build_search_space(repo, target_len_2, verbose=False)
    query = generator_query(space, target_len_2)

    depth_bounded = DepthBoundedRandomSampler(100, random.Random(0))
    assert len(list(itertools.islice(depth_bounded.sample(query), 3))) == 3
    assert walks == [], "the depth-bounded sampler counts nothing, which is why it is the cheap one"

    size_uniform = SizeUniformSampler(100, random.Random(0))
    assert len(list(itertools.islice(size_uniform.sample(query), 3))) == 3
    assert len(walks) == 1, "the first size-uniform draw walks the retained search tree"

    assert len(list(itertools.islice(size_uniform.sample(query), 10))) == 10
    assert len(walks) == 1, "ten further draws from the same query must reuse that one walk"

    size_uniform.forget()
    assert len(list(itertools.islice(size_uniform.sample(query), 1))) == 1
    assert len(walks) == 2, "after forget the construction has to be built again"


@pytest.mark.slow
def test_the_counting_toll_on_the_target_the_example_searches_is_tens_of_seconds():
    """The same measurement on ``target_len_3``, which is what the example actually searches.

    Deselected from the default run because it costs what it measures. It is here because the
    number is the reason ``damg_example`` draws depth-bounded, and a reason that no test holds
    stops being true without anyone noticing.
    """
    depth, first_three, next_ten = _sampler_costs(target_len_3)

    assert first_three > 10.0, (
        f"three size-uniform draws on the target_len_3 space took {first_three:.2f}s, and the "
        f"sampler choice of damg_example is argued from tens of seconds"
    )
    assert first_three > 1000.0 * depth
    assert next_ten < 1.0


# ---------------------------------------------------------------------------
# 3. The example.
# ---------------------------------------------------------------------------

def test_the_acquisition_optimizer_keeps_both_convergence_rates():
    """The crossover rate stays below 1 and the mutation rate above 0.

    The example used to ask for 0.99 and 0.00, and the second is on the wrong side of the
    condition: with no mutation at all the search cannot reach an individual its population does
    not already contain a piece of.
    """
    search = damg_example.build_acquisition_optimizer(population_size=4, generations=2)
    assert 0.0 <= search.crossover_rate < 1.0
    assert 0.0 < search.mutation_rate <= 1.0
    assert search.population_size == 4


def test_the_objective_is_the_negated_log_of_the_test_loss():
    """The loop maximizes, so a loss reaches it negated, and on a log scale so it stays in range.

    The logarithm is not decoration. A term that diverges reports a loss eight orders of magnitude
    above one that fits, and an evolutionary run reads its fitness through an exponential.
    """
    x, y = damg_example.generate_data(damg_example.TrapezoidNetPure(), n_samples=32)

    repo = DAMGrepository(
        linear_feature_dimensions=[1, 2, 3, 4, 5],
        constant_values=[0, 1, -1],
        learning_rate_values=[1e-2],
        n_epoch_values=[EPOCHS],
    )
    term = _synthesize_one(
        repo.specification(),
        _target(1, 1,
                (((DAMGrepository.Linear(in_features=1, out_features=1, bias=True), 1, 1),),)),
    )

    # The same term, trained twice from the same seed, so the raw loss and the objective value are
    # two readings of one training run and the relation between them can be asserted exactly.
    torch.manual_seed(0)
    raw = term.interpret(pytorch_function_algebra())(x, y, x, y)
    torch.manual_seed(0)
    value = damg_example.make_objective(x, y, x, y)(term)

    assert value == pytest.approx(-math.log1p(raw))
    assert value <= 0.0, (
        "a loss is non-negative, so the negated logarithm of one plus it cannot be positive")


def test_the_example_runs_its_whole_loop_and_returns_a_term_of_the_space():
    """``main`` builds the space, runs the closed loop and answers with a term it evaluated.

    The budget is two passes over an initial design of three on a target of one training epoch,
    which is small enough for a test to pay for and still runs every stage: the initial design, the
    surrogate, the acquisition maximization and the answer of maximal observed value. The second
    pass is there so that at least one surrogate is fitted on a pair the loop itself produced.
    """
    small = _target(1, 1, (None, None), epochs=EPOCHS)
    result = damg_example.main(
        small, budget=2, initial_size=3, n_samples=64,
        population_size=10, generations=3, seed=0, verbose=False,
    )

    assert len(result["y"]) == 5
    assert result["best_y"] == max(result["y"])
    assert result["best_tree"] in result["x"]

    # The recommendation is an inhabitant of the space, which is what the loop's closure property
    # says, and it is the one guarantee the loop makes about the term it hands back.
    space = Synthesizer(
        damg_example.build_repository(small).specification(), {}
    ).construct_solution_space(small).prune()
    assert space.contains_tree(small, result["best_tree"]), (
        "the loop answered with a term that is not derivable from the space it searched"
    )
