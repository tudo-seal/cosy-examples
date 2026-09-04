"""The recognizable CNN repository states the same laws over the same clauses as the original.

``recognizable_cnn_damg_repo.py`` exists so that the CNN space can be determinized: its four swap
laws are stated as an abstraction with a relation instead of as term predicates, which is what lets
``cosy.search.determinize`` compile them into the non-terminals.  The two forms share the
``before_cons`` clause the laws hang on and differ in the one method that hangs them there.

These tests do not check that the one form *looks* like the other, they check that the two
programs *are* the same program:

* clause for clause, before any synthesis, so that the difference stays the four predicates,
* rule for rule, over every non-terminal, with the same terminals and the same arguments,
* predicate for predicate, on every pair of subterms the space realizes and on hand-built pairs
  that the laws were written to reject,
* clause for clause, by reading off which relations the synthesized rules actually carry,
* term for term, by counting both spaces and comparing the rows.

Structure length 2 throughout.  It is the only length whose space both forms can be counted on,
because the coupled form has to build the retained search tree, and removing that need is what the
recognizable form is for.  Length 2 has one blind spot, and the witnesses below are the answer to
it.  No law fires anywhere in this space: hand the repository four relations that admit everything
and the determinized program comes out with the same 2605 rules, so a relation that always returned
True would pass every other test here.
"""

from __future__ import annotations

import pytest
from cosy.core import Synthesizer
from cosy.core.recognizable import RecognizableConstraint, state_of
from cosy.core.solution_space import NonTerminalArgument
from cosy.core.tree import Tree
from cosy.core.types import Abstraction, Implication
from cosy.search import depth_first, generator_query, term_size
from cosy.search.counting import branch_counts, branch_multiplicities, size_table
from cosy.search.determinize import determinize, unabstracted_clauses

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_targets import (
    make_usps_experiment_target,
)
from bayesian_optimization.examples.cnn_damg_nas.recognizable_cnn_damg_repo import (
    OTHER_TERMINALS,
    R1,
    R2,
    R3,
    R4,
    RecognizableCNNrepository,
    alpha,
    check_alphabet,
)
from bayesian_optimization.examples.recognizable_swap_laws import TOKENS

LENGTH = 2
EPOCHS = 50

#: The configuration of the USPS stage of this example, which is what every number below was
#: measured on.  It is written out here rather than imported, because the experiment script that
#: carries it is not part of this repository.
PARAMETER_SETS = {
    "linear_feature_dimensions": [256, 768, 400, 192, 64, 30, 16, 10],
    "constant_values": [0, 1, -1],
    "learning_rate_values": [1e-3],
    "n_epoch_values": [EPOCHS],
    "channel_dimensions": [1, 4, 8, 12],
    "height_width_dimensions": [(16, 16), (8, 8), (4, 4)],
    "kernel_dimensions": [(3, 3), (5, 5)],
    "stride_values": [1, 2],
    "padding_values": [0, 1, 2],
    "max_parallel_width": 2,
}

#: The four laws paired with the relation that is meant to decide them, in law order.
LAWS = (
    ("swaplaw1", R1),
    ("swaplaw2", R2),
    ("swaplaw3", R3),
    ("swaplaw4", R4),
)


def build(repository_class):
    """Build the pruned space of one repository class at the fixed structure length.

    Args:
        repository_class (type): ``CNNrepository`` or ``RecognizableCNNrepository``.

    Returns:
        tuple: The space and the queried target.
    """
    repository = repository_class(**PARAMETER_SETS)
    synthesizer = Synthesizer(repository.specification(), {})
    target = make_usps_experiment_target(epochs=EPOCHS, length=LENGTH)
    return synthesizer.construct_solution_space(target).prune(), target


@pytest.fixture(scope="module")
def original():
    """Build the tracked repository's space once for the module.

    Returns:
        tuple: The space and the target.
    """
    return build(CNNrepository)


@pytest.fixture(scope="module")
def recognizable():
    """Build the recognizable repository's space once for the module.

    Returns:
        tuple: The space and the target.
    """
    return build(RecognizableCNNrepository)


@pytest.fixture(scope="module")
def determinization(recognizable):
    """Compile the four laws into the non-terminals, once for the module.

    Args:
        recognizable (tuple): The recognizable repository's space and target.

    Returns:
        Determinization: The product program, its start symbol and its states.
    """
    space, target = recognizable
    return determinize(space, target)


def shape_of(space):
    """Return the program's rules in a form that ignores the predicates.

    What a subclass that goes further than restating the laws can move is a parameter set, a
    constraint on the literals, or an argument type, all of which change which rules the synthesis
    produces.  Rendering the rules without their predicates isolates exactly that: what remains has
    to be identical, because the only intended difference between the two repositories is how the
    four laws are stated.

    Args:
        space (SolutionSpace): The synthesized program.

    Returns:
        dict[str, list[tuple]]: Per non-terminal, its rules as terminal plus argument descriptors.
    """
    shape = {}
    for nonterminal in space.nonterminals():
        rules = []
        for rule in space.get(nonterminal) or ():
            arguments = tuple(
                ("hole", str(argument.origin), argument.name)
                if isinstance(argument, NonTerminalArgument)
                else ("const", repr(argument.value), argument.name)
                for argument in rule.arguments
            )
            rules.append((str(rule.terminal), arguments))
        shape[str(nonterminal)] = sorted(rules)
    return shape


def clause_spine(clause):
    """Return what a clause introduces, in order, with its predicates named by kind.

    Args:
        clause (Specification): One combinator's specification, as the builder leaves it.

    Returns:
        list: One entry per parameter and per predicate, closed by the suffix type.
    """
    spine = []
    node = clause
    while isinstance(node, (Abstraction, Implication)):
        if isinstance(node, Abstraction):
            spine.append(("parameter", node.parameter.name))
        elif node.predicate.only_literals:
            spine.append(("constraint on the literals",))
        elif isinstance(node.predicate.constraint, RecognizableConstraint):
            spine.append(("law", "as a relation on states", node.predicate.constraint.abstraction))
        else:
            spine.append(("law", "as a predicate on terms", None))
        node = node.body
    spine.append(("suffix", str(node)))
    return spine


def test_the_two_forms_differ_in_the_four_laws_and_in_nothing_else():
    """The clause is one clause, and the override states its four laws the other way round.

    Read before any synthesis, so it says what the rule comparison cannot.  That one strips the
    predicates off the rules, and the four laws are the predicates.  Here the two clauses have to
    agree parameter for parameter, in the constraints on the literals and in the suffix, and each
    has to carry four laws of one kind: predicates on terms in the base class, relations on states
    in the override.  Dropping one, attaching a fifth, or stating a single law in the other form
    all leave a repository that builds, and the rule comparison sees none of the three.

    The four recognizable constraints have to name one and the same abstraction, because
    ``cosy.search.determinize`` builds the product over the distinct abstractions of a program and
    tells two of them apart with ``is`` or ``==``, which for a closure is identity.  Four closures
    over the same terminals would be four axes of that product where one does.
    """
    coupled = clause_spine(CNNrepository(**PARAMETER_SETS).specification()["before_cons"])
    restated = clause_spine(
        RecognizableCNNrepository(**PARAMETER_SETS).specification()["before_cons"]
    )

    def without_the_laws(spine):
        return [entry for entry in spine if entry[0] != "law"]

    assert without_the_laws(coupled) == without_the_laws(restated)
    assert [entry for entry in coupled if entry[0] == "law"] == [
        ("law", "as a predicate on terms", None)
    ] * 4
    assert [entry for entry in restated if entry[0] == "law"] == [
        ("law", "as a relation on states", alpha)
    ] * 4


def test_the_two_repositories_produce_the_same_rules(original, recognizable):
    """The two forms must agree: same non-terminals, same rules, same arguments.

    The clause is inherited, so what is left to catch is a subclass that goes further than
    restating the laws, and a change to the clause that the two forms do not survive equally.
    Neither would raise anything.  It would quietly synthesize a different space, and every number
    measured on it would be a number about the wrong program.

    Args:
        original (tuple): The tracked repository's space and target.
        recognizable (tuple): The recognizable repository's space and target.
    """
    original_space, _ = original
    recognizable_space, _ = recognizable
    assert shape_of(original_space) == shape_of(recognizable_space)


def test_only_the_recognizable_form_can_be_determinized(original, recognizable, determinization):
    """The point of the file: one program refuses determinization, the other does not.

    The eight offending clauses are the ``before_cons`` instances of this space.  Each carries the
    four laws as predicates over two of its holes, which is what no table indexed by the
    non-terminal can read.  What the compilation costs is the second half of the assertion: the
    product of 307 non-terminals with the reachable states of the abstraction, against 856 rules in
    the program it comes from.

    Args:
        original (tuple): The tracked repository's space and target.
        recognizable (tuple): The recognizable repository's space and target.
        determinization (Determinization): The compiled product program.
    """
    original_space, _ = original
    recognizable_space, _ = recognizable
    offenders = unabstracted_clauses(original_space)
    assert len(offenders) == 8, "the tracked repository's coupled clauses"
    assert all(clause.terminal == "before_cons" for clause in offenders)
    assert unabstracted_clauses(recognizable_space) == []

    assert len(list(recognizable_space.nonterminals())) == 307
    assert sum(len(recognizable_space.get(name) or ())
               for name in recognizable_space.nonterminals()) == 856
    assert determinization.state_count == 1044
    assert sum(len(determinization.space.get(name) or ())
               for name in determinization.space.nonterminals()) == 2605


def test_the_four_laws_are_compiled_over_one_abstraction(determinization):
    """One abstraction in the program is one axis of the product, which the count above assumes.

    Args:
        determinization (Determinization): The compiled product program.
    """
    assert determinization.abstractions == (alpha,)


def test_the_abstraction_covers_the_alphabet_the_program_uses(recognizable):
    """``alpha`` knows every terminal, and the laws' substring test agrees with its equality test.

    The laws match roots with ``token in root`` while the abstraction matches with ``root ==
    token``.  They agree only as long as no terminal name contains another as a substring, which is
    a property of this repository and not a law of nature, so it is checked rather than assumed.

    Args:
        recognizable (tuple): The recognizable repository's space and target.
    """
    recognizable_space, _ = recognizable
    found = check_alphabet(recognizable_space)
    assert found["substring_equals_equality"] is True
    # 19 of the 22 terminals the abstraction knows.  The three it does not reach here are sum,
    # product and copy, which need an input or an output feature of 1, and this configuration has
    # none.
    assert len(found["terminals"]) == 19


def test_the_abstraction_knows_every_combinator_of_the_repository():
    """The alphabet is held against the combinator table, not against a space.

    The check above sees only the terminals a space realizes, 19 of the 22 here, so a terminal that
    no configuration in use enumerates can fall out of the list without anything saying so.  That
    has happened, and the comment beside sum, product and copy in the module says how far it got.
    The combinator table carries all 22 whatever a configuration does with them.
    """
    combinators = frozenset(CNNrepository(**PARAMETER_SETS).specification())
    assert combinators == frozenset(TOKENS) | OTHER_TERMINALS


def test_the_relations_decide_every_realized_pair_as_the_laws_do(original):
    """``R`` after ``alpha`` and the four laws agree on every pair the space realizes.

    Args:
        original (tuple): The tracked repository's space and target.
    """
    original_space, target = original
    pairs = set()
    for tree in depth_first(generator_query(original_space, target), max_count=40):
        pending = [tree]
        while pending:
            node = pending.pop()
            if node.root == "before_cons" and len(node.children) == 9:
                pairs.add((node.children[7], node.children[8]))
            pending.extend(node.children)
    assert pairs, "no before_cons node in the first terms, so the harness measures nothing"

    for x, y in pairs:
        states = {"x": state_of(x, alpha), "y": state_of(y, alpha)}
        for name, relation in LAWS:
            law = getattr(CNNrepository, name)
            assert relation(states) == law(x, y), f"{name} disagrees with its relation"


# ------------------------------------------------------------------------- the witness corpus
#
# Every pair the test above sees is a pair some law admitted, because a term is only derived once
# every predicate on it has said True.  A relation that returned True on everything would therefore
# pass it.  The pairs a law rejects have to be built by hand, and these are they.
#
# Each of the four laws forbids the left-hand side of one rewrite rule and states that rule in its
# own docstring.  Read over the layers of a sequential composition rather than over the tree, three
# of the four are patterns of two consecutive layers and one is a pattern of three:
#
#   swaplaw2  [swap(a, b), edges(c)]  then  [edges(b), swap(a, c)]
#   swaplaw4  [edges(a), swap(b, c)]  then  [swap(a, c), edges(b)]
#   swaplaw3  [swap(a, b)]            then  [swap(b, a)]
#   swaplaw1  [swap(a, b)], then a layer whose first entry carries the edge numbers (b, c) and
#             whose remaining entries carry (a, d) between them, then [swap(c, d)]
#
# ``edges(x)`` is the degenerate crossing that carries x edges straight through, which is how the
# ``copy(x, edge())`` of the rewrite rules stands in a term.
#
# Each law gets the pair it rejects and, beside it, the same pair with the closing edge numbers
# changed so that the pattern does not close.  The second one has to be admitted, and it is what
# catches a relation that rejects more than its law does.


def _node(root, arity, **slots):
    """Build a tree with ``arity`` children, filling the named slots and leaving the rest at 0."""
    children = [Tree(0, ()) for _ in range(arity)]
    for slot, value in slots.items():
        children[int(slot[1:])] = value
    return Tree(root, tuple(children))


def _swap(m, n):
    """The wiring that crosses ``m`` edges past ``n`` edges."""
    return _node("swap", 19, _1=Tree(m, ()), _2=Tree(n, ()))


def _edges(n):
    """``n`` edges carried straight through, which is the degenerate crossing."""
    return _node("edges", 17, _0=Tree(n, ()))


def _beside_singleton(inner=None, i=None, o=None):
    """A layer with one entry."""
    slots = {}
    if inner is not None:
        slots["_4"] = inner
    if i is not None:
        slots["_0"] = Tree(i, ())
    if o is not None:
        slots["_1"] = Tree(o, ())
    return _node("beside_singleton", 5, **slots)


def _beside_cons(i, i1, o, o1, first, rest):
    """A layer with a first entry and a tail.

    The combinator binds the edge numbers of the first entry and, beside them, the sums over the
    remaining entries as ``i - i1`` and ``o - o1``.  A witness has to carry all four, since that is
    where the three-layer pattern reads the edge numbers of the entries after the first.
    """
    return _node("beside_cons", 11,
                 _0=Tree(i, ()), _1=Tree(i1, ()), _2=Tree(i - i1, ()),
                 _3=Tree(o, ()), _4=Tree(o1, ()), _5=Tree(o - o1, ()),
                 _9=first, _10=rest)


def _before_singleton(layer):
    """A sequential composition of exactly one layer."""
    return _node("before_singleton", 6, _5=layer)


def _before_cons(layer, rest):
    """A sequential composition of one layer and the rest."""
    return _node("before_cons", 9, _7=layer, _8=rest)


def _witnesses():
    """Build the corpus.

    Returns:
        list: One ``(law, case, verdict, head, tail)`` per witness, ``verdict`` being what the law
        named by ``law`` has to say about the pair.
    """
    m, n, p, q = 3, 4, 5, 6
    a, b, c = 3, 4, 5

    # The three-layer pattern: a layer that crosses m past n, then a layer whose first entry
    # carries (n, p) and whose remaining entries carry (m, q) between them, then a layer that
    # crosses p past q.  The three layers run two subterms past each other and cross back, which is
    # the same diagram as the two subterms side by side in the other order, so the space must not
    # hold both.
    first_layer = _beside_singleton(inner=_swap(m, n))
    third_layer = _beside_singleton(inner=_swap(p, q))
    third_layer_open = _beside_singleton(inner=_swap(p, q + 1))
    middle_layer = _beside_cons(i=n + m, i1=n, o=p + q, o1=p,
                                first=Tree("node", ()), rest=_beside_singleton(i=m, o=q))

    # The pair swaplaw3 forbids: crossing a past b and then b past a is the identity wiring.
    involution_left = _beside_singleton(inner=_swap(a, b))
    involution_right = _beside_singleton(inner=_swap(b, a))
    involution_open = _beside_singleton(inner=_swap(b, a + 1))

    # The pair swaplaw2 forbids: crossing a past b beside c straight through, then b straight
    # through beside a crossing past c, is one crossing of a past b + c.
    first_pair_left = _beside_cons(i=a + b + c, i1=a + b, o=a + b + c, o1=a + b,
                                   first=_swap(a, b), rest=_beside_singleton(inner=_edges(c)))
    first_pair_right = _beside_cons(i=a + b + c, i1=b, o=a + b + c, o1=b,
                                    first=_edges(b), rest=_beside_singleton(inner=_swap(a, c)))
    first_pair_open = _beside_cons(i=a + b + c, i1=b, o=a + b + c, o1=b,
                                   first=_edges(b), rest=_beside_singleton(inner=_swap(a + 1, c)))

    # The pair swaplaw4 forbids, the mirror of the one above: one crossing of a + b past c.
    second_pair_left = _beside_cons(i=a + b + c, i1=a, o=a + b + c, o1=a,
                                    first=_edges(a), rest=_beside_singleton(inner=_swap(b, c)))
    second_pair_right = _beside_cons(i=a + b + c, i1=a + c, o=a + b + c, o1=a + c,
                                     first=_swap(a, c), rest=_beside_singleton(inner=_edges(b)))
    second_pair_open = _beside_cons(i=a + b + c, i1=a + c, o=a + b + c, o1=a + c,
                                    first=_swap(a, c), rest=_beside_singleton(inner=_edges(b + 1)))

    return [
        ("swaplaw1", "the three layers close", False,
         first_layer, _before_cons(middle_layer, _before_singleton(third_layer))),
        ("swaplaw1", "the third layer crosses other edges", True,
         first_layer, _before_cons(middle_layer, _before_singleton(third_layer_open))),
        ("swaplaw2", "the two layers close", False,
         first_pair_left, _before_singleton(first_pair_right)),
        ("swaplaw2", "the second layer crosses other edges", True,
         first_pair_left, _before_singleton(first_pair_open)),
        ("swaplaw3", "the two layers close", False,
         involution_left, _before_singleton(involution_right)),
        ("swaplaw3", "the second layer crosses other edges", True,
         involution_left, _before_singleton(involution_open)),
        ("swaplaw4", "the two layers close", False,
         second_pair_left, _before_singleton(second_pair_right)),
        ("swaplaw4", "the second layer crosses other edges", True,
         second_pair_left, _before_singleton(second_pair_open)),
    ]


WITNESSES = _witnesses()


@pytest.mark.parametrize(("law", "case", "verdict", "head", "tail"), WITNESSES,
                         ids=[f"{law}-{case.replace(' ', '_')}" for law, case, _, _, _ in WITNESSES])
def test_the_relations_reject_what_the_laws_reject(law, case, verdict, head, tail):
    """A pair each law rejects and a near miss it admits, decided through the abstraction.

    Every relation is asked about every witness, not only the one the witness was built for.  A
    relation that fired on another law's pattern would reject terms its law admits, and the space
    would silently lose them.

    Args:
        law (str): The law the pair was built for.
        case (str): What the pair is, for the failure message.
        verdict (bool): What that law has to say about the pair.
        head (Tree): The first hole's subterm.
        tail (Tree): The second hole's subterm.
    """
    assert getattr(CNNrepository, law)(head, tail) is verdict, f"{law} does not see {case}"
    states = {"x": state_of(head, alpha), "y": state_of(tail, alpha)}
    for name, relation in LAWS:
        assert relation(states) is getattr(CNNrepository, name)(head, tail), (
            f"{name} and its relation disagree on the witness for {law}"
        )


def test_every_law_reaches_the_clause_once_and_in_its_own_slot(recognizable):
    """Each of the four relations is attached to ``before_cons``, once, in law order.

    A relation that is never attached is invisible to every other test in this file.  At structure
    length 2 no law fires, so dropping one of the clause's ``recognizable_constraint`` calls, or
    naming an earlier relation twice in place of a later one, leaves the same 1044 states and the
    same 2605 rules, leaves the counted row where it was, and leaves the witnesses deciding as
    their laws do.  One structure length up it is a different language: the determinized program
    has 419 818 rules with all four laws attached, 419 821 with the third one left off and 419 827
    with none of them.  Replace one entry of ``RELATIONS`` with a relation that admits everything
    and rebuild at length 3 to see it.  That build is what this test exists to avoid, so it reads
    the wiring off the length 2 program instead.

    The check reads the predicates off the synthesized rules, not off the clause that built them,
    so it goes on holding when the relations and the abstraction move to a module of their own.

    Args:
        recognizable (tuple): The recognizable repository's space and target.
    """
    recognizable_space, _ = recognizable
    in_law_order = tuple(relation for _, relation in LAWS)
    assert tuple(RecognizableCNNrepository.RELATIONS) == in_law_order, "RELATIONS is out of order"

    carriers = [
        rule
        for nonterminal in recognizable_space.nonterminals()
        for rule in recognizable_space.get(nonterminal) or ()
        if rule.predicates
    ]
    assert {str(rule.terminal) for rule in carriers} == {"before_cons"}
    assert len(carriers) == 8

    expected = tuple((alpha, relation) for relation in in_law_order)
    for rule in carriers:
        assert all(isinstance(predicate, RecognizableConstraint) for predicate in rule.predicates)
        attached = tuple(
            (predicate.abstraction, predicate.relation) for predicate in rule.predicates
        )
        assert attached == expected, "the clause does not attach the four laws once each, in order"


def test_the_determinized_space_counts_what_the_coupled_space_counts(original, determinization):
    """The invariant the whole construction rests on: same branches, counted two ways.

    ``branch_counts`` walks the retained search tree of the coupled program, ``size_table`` reads
    the determinized program.  They have to produce the same row, or the determinization has
    changed the language and every weight drawn from it would be wrong.

    Args:
        original (tuple): The tracked repository's space and target.
        determinization (Determinization): The compiled product program.
    """
    original_space, target = original
    # Three occupied sizes, which is enough to be a row and cheap enough for a test.
    bound = 83

    counted = branch_counts(generator_query(original_space, target), bound, term_size)
    assert branch_multiplicities(counted) == {}, "the coupled space derives some term twice"

    table = size_table(determinization.space, bound)
    row = {
        size: table.of(determinization.start, size)
        for size in range(bound + 1)
        if table.of(determinization.start, size)
    }
    assert row == dict(counted.counts)
    assert row == {62: 400, 71: 24, 83: 2312}
