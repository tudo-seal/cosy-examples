"""The recognizable DAMG repository states the same laws over the same clauses as the original.

``recognizable_damg_repo.py`` exists so that the DAMG space can be determinized: its four swap laws
are stated as an abstraction with a relation instead of as term predicates, which is what lets
``cosy.search.determinize`` compile them into the non-terminals.  The two forms share the
``before_cons`` clause the laws hang on and differ in the one method that hangs them there.

These tests do not check that the one form *looks* like the other, they check that the two programs
*are* the same program:

* clause for clause, before any synthesis, so that the difference stays the four predicates,
* rule for rule, over every non-terminal, with the same terminals and the same arguments,
* predicate for predicate, on every pair of subterms the space realizes and on hand-built pairs
  that the laws were written to reject,
* term for term, by counting both spaces and comparing the rows.

Three configurations and a corpus of pairs that belongs to none of them.  The first configuration is
the one ``damg_example.py`` searches, at ``target_len_3``, and it is what every number in the module
docstring was measured on.  On it the four laws forbid nothing at any size, so relations that admit
everything pass every test that only uses it.  The second is ``target_len_4`` with the feature
dimensions cut down to one value, which is the cheapest space here that the coupled walk reaches and
in which a law forbids something.  Only the third law does: with a relation that always says yes in
any of the other three slots, that comparison stays green.  The third configuration is the same
target with two feature dimensions, read through the table alone, and it reaches the second and the
fourth law as well.

The first law forbids nothing at any size of any of the three, so no count in this file pins it.
The witness corpus does.  It is a set of pairs each law was written to reject, decided through the
abstraction, and every relation is asked about every pair, so a slot holding a relation that always
says yes fails under the name of the law whose slot it is.
"""

from __future__ import annotations

import pytest
from cosy.core import Synthesizer
from cosy.core.recognizable import RecognizableConstraint, state_of
from cosy.search import depth_first, generator_query, term_size
from cosy.search.counting import branch_counts, branch_multiplicities, size_table
from cosy.search.determinize import determinize, unabstracted_clauses

from bayesian_optimization.examples.damg_nas import damg_targets
from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository
from bayesian_optimization.examples.damg_nas.recognizable_damg_repo import (
    OTHER_TERMINALS,
    R1,
    R2,
    R3,
    R4,
    RecognizableDAMGrepository,
    alpha,
    check_alphabet,
)
from bayesian_optimization.examples.recognizable_swap_laws import TOKENS
from tests.programs import clause_spine, shape_of
from tests.swap_law_witnesses import WITNESSES

#: The configuration of ``damg_example.py``, which is what the numbers below were measured on.
PARAMETER_SETS = {
    "linear_feature_dimensions": [1, 2, 3, 4, 5],
    "constant_values": [0, 1, -1],
    "learning_rate_values": [1e-2],
    "n_epoch_values": [2000],
}

#: The one feature dimension and the two constants that make ``target_len_4`` walkable.  The third
#: law first forbids a term at size 124 of that space, and the walk gets there in about two
#: seconds.  On the configuration above the same walk does not get there at all: it was given 200
#: seconds and did not finish.
#: Cutting the feature dimensions is not free, and this is the cut that costs the most: up to size
#: 200 the ``target_len_4`` space of the configuration above occupies 34 sizes and this one 9.  One
#: of the sizes that go is 180, the only one at which the second and the fourth law forbid
#: anything, so whole shapes leave and not only counts.  ``SHARP_PARAMETER_SETS`` below is the cut
#: that keeps them.
#: Both 0 and 1 have to be among the constants, because ``DAMGrepository`` appends both of them as
#: soon as either is missing, and the duplicate value that produces makes the space ambiguous.
REDUCED_PARAMETER_SETS = {
    "linear_feature_dimensions": [2],
    "constant_values": [0, 1],
    "learning_rate_values": [1e-2],
    "n_epoch_values": [2000],
}

#: The bound of the reduced space's comparison.  It is the first size at which a law forbids a term.
REDUCED_BOUND = 124

#: The names of the four laws, in the order the clause attaches them.
LAW_NAMES = ("swaplaw1", "swaplaw2", "swaplaw3", "swaplaw4")

#: Each law paired with the relation the repository puts in that law's slot.  Read off
#: ``RELATIONS`` rather than written out a second time, so that a slot carrying the wrong relation
#: fails under the name of the law whose slot it is, wherever this pairing is used.  That the four
#: are ``R1`` to ``R4`` in that order is a separate assertion, in
#: ``test_every_law_reaches_the_clause_once_and_in_its_own_slot``.
LAWS = tuple(zip(LAW_NAMES, RecognizableDAMGrepository.RELATIONS, strict=True))

#: The arity the abstraction assumes for each of the six terminals the laws read.  It carries the
#: arity in the state and refuses a node that has another one, so a clause that grows a parameter
#: turns every law that reads it into an error rather than into a wrong answer.
TOKEN_ARITIES = {
    "beside_singleton": 5,
    "beside_cons": 11,
    "before_singleton": 6,
    "before_cons": 9,
    "swap": 19,
    "edges": 17,
}


def admit_everything(substitution):
    """A relation in the shape of the four that forbids nothing.

    It is the worst a restatement of the laws could come out as, and it is what every counting test
    below runs against as a control.

    Args:
        substitution (dict): The states of the two holes, which it does not read.

    Returns:
        bool: True, whatever the pair.
    """
    return True


class AdmitEverything(RecognizableDAMGrepository):
    """The repository with all four laws replaced by the relation that forbids nothing."""

    RELATIONS = (admit_everything,) * 4


def build(repository_class, parameter_sets, target):
    """Build the pruned space of one repository class.

    Args:
        repository_class (type): ``DAMGrepository`` or a subclass that restates its laws.
        parameter_sets (dict): The configuration to build it with.
        target (Type): The queried type.

    Returns:
        SolutionSpace: The pruned space.
    """
    repository = repository_class(**parameter_sets)
    synthesizer = Synthesizer(repository.specification(), {})
    return synthesizer.construct_solution_space(target).prune()


def rule_count(space):
    """Return how many rules the program has over all of its non-terminals.

    Args:
        space (SolutionSpace): The synthesized program.

    Returns:
        int: The number of rules.
    """
    return sum(len(space.get(name) or ()) for name in space.nonterminals())


@pytest.fixture(scope="module")
def original():
    """Build the tracked repository's space once for the module.

    Returns:
        SolutionSpace: The coupled program at ``target_len_3``.
    """
    return build(DAMGrepository, PARAMETER_SETS, damg_targets.target_len_3)


@pytest.fixture(scope="module")
def recognizable():
    """Build the recognizable repository's space once for the module.

    Returns:
        SolutionSpace: The recognizable program at ``target_len_3``.
    """
    return build(RecognizableDAMGrepository, PARAMETER_SETS, damg_targets.target_len_3)


@pytest.fixture(scope="module")
def determinization(recognizable):
    """Compile the four laws into the non-terminals, once for the module.

    Args:
        recognizable (SolutionSpace): The recognizable program at ``target_len_3``.

    Returns:
        Determinization: The product program, its start symbol and its states.
    """
    return determinize(recognizable, damg_targets.target_len_3)


def test_the_two_forms_differ_in_the_four_laws_and_in_nothing_else():
    """The clause is one clause, and the override attaches four laws where four laws were.

    Read before any synthesis, so it says something the rule comparison cannot: where in the clause
    the laws sit, and how many there are.  Attaching a fifth, dropping one, or attaching them
    before the two holes are introduced all leave a repository that builds and synthesizes, and the
    first two of those change the language without changing a rule.

    The four recognizable constraints have to name one and the same abstraction, because
    ``cosy.search.determinize`` builds the product over the distinct abstractions of a program and
    tells two of them apart with ``is`` or ``==``, which for a closure is identity.  Four closures
    over the same terminals would be four axes of that product where one does.
    """
    coupled = clause_spine(DAMGrepository(**PARAMETER_SETS).specification()["before_cons"])
    restated = clause_spine(
        RecognizableDAMGrepository(**PARAMETER_SETS).specification()["before_cons"]
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


def test_before_cons_is_the_only_clause_with_a_predicate_over_a_hole():
    """The override reaches every law there is, because they all hang on the one clause.

    A law attached to a second clause would be inherited unchanged by the subclass and would keep
    the program uncountable, without any of the tests below noticing: they read the ``before_cons``
    clause and the rules, not the other clauses' predicates.
    """
    specification = DAMGrepository(**PARAMETER_SETS).specification()
    coupled = {
        terminal
        for terminal, clause in specification.items()
        for entry in clause_spine(clause)
        if entry[0] == "law"
    }
    assert coupled == {"before_cons"}


def test_the_two_repositories_produce_the_same_rules(original, recognizable):
    """The two forms must agree: same non-terminals, same rules, same arguments.

    The clause is inherited, so what is left to catch is a subclass that goes further than
    restating the laws, and a change to the clause that the two forms do not survive equally.
    Neither would raise anything.  It would quietly synthesize a different space, and every number
    measured on it would be a number about the wrong program.

    Args:
        original (SolutionSpace): The coupled program.
        recognizable (SolutionSpace): The recognizable program.
    """
    assert shape_of(original) == shape_of(recognizable)
    assert len(list(recognizable.nonterminals())) == 88
    assert rule_count(recognizable) == 404


def test_only_the_recognizable_form_can_be_determinized(original, recognizable, determinization):
    """The point of the file: one program refuses determinization, the other does not.

    The 30 offending clauses are the ``before_cons`` instances of this space.  Each carries the
    four laws as predicates over two of its holes, which is what no table indexed by the
    non-terminal can read.  What the compilation costs is the second half of the assertion: the
    product of 88 non-terminals with the reachable states of the abstraction, against 404 rules in
    the program it comes from.

    Args:
        original (SolutionSpace): The coupled program.
        recognizable (SolutionSpace): The recognizable program.
        determinization (Determinization): The compiled product program.
    """
    offenders = unabstracted_clauses(original)
    assert len(offenders) == 30, "the tracked repository's coupled clauses"
    assert all(clause.terminal == "before_cons" for clause in offenders)
    assert unabstracted_clauses(recognizable) == []

    assert determinization.state_count == 735
    assert rule_count(determinization.space) == 1486


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
        recognizable (SolutionSpace): The recognizable program.
    """
    found = check_alphabet(recognizable)
    assert found["substring_equals_equality"] is True
    assert len(found["terminals"]) == 17


def test_the_abstraction_knows_every_combinator_of_the_repository():
    """The alphabet is held against the combinator table, not against a space.

    The check above sees only the terminals a space realizes.  A terminal that no configuration in
    use enumerates can fall out of the hand-written list without anything saying so, and that has
    happened to the list of the CNN repository next door, where the comment beside ``sum``,
    ``product`` and ``copy`` says how far it got.  The combinator table carries all 17 whatever a
    configuration does with them.
    """
    combinators = frozenset(DAMGrepository(**PARAMETER_SETS).specification())
    assert combinators == frozenset(TOKENS) | OTHER_TERMINALS


def test_the_tokens_have_the_arities_the_abstraction_reads(original):
    """The abstraction reads fixed child positions, and this is what says they are those positions.

    ``alpha`` and the four relations are shared with the CNN repository, and what makes that
    possible is that the six terminals the laws look into have the same arity here.  A parameter
    added to one of these clauses would move every position the laws read.  The abstraction would
    then refuse the node rather than answer wrongly, but only where a law reaches it.

    Args:
        original (SolutionSpace): The coupled program.
    """
    arities = {}
    for nonterminal in original.nonterminals():
        for rule in original.get(nonterminal) or ():
            arities.setdefault(str(rule.terminal), set()).add(len(rule.arguments))
    assert {token: sorted(arities[token]) for token in TOKEN_ARITIES} == {
        token: [arity] for token, arity in TOKEN_ARITIES.items()
    }


def test_the_relations_decide_every_realized_pair_as_the_laws_do(original):
    """``R`` after ``alpha`` and the four laws agree on every pair the space realizes.

    Every pair this sees is a pair some law admitted, because a term is only derived once every
    predicate on it has said True, and on this target no law ever forbids anything.  Relations that
    admit everything therefore pass this too.  What answers that is the witness corpus below, which
    is built out of the pairs no space contains.

    Args:
        original (SolutionSpace): The coupled program.
    """
    pairs = set()
    for tree in depth_first(generator_query(original, damg_targets.target_len_3), max_count=200):
        pending = [tree]
        while pending:
            node = pending.pop()
            if node.root == "before_cons" and len(node.children) == 9:
                pairs.add((node.children[7], node.children[8]))
            pending.extend(node.children)
    assert len(pairs) == 384, "no before_cons node in the first terms, so this measures nothing"

    for x, y in pairs:
        states = {"x": state_of(x, alpha), "y": state_of(y, alpha)}
        for name, relation in LAWS:
            law = getattr(DAMGrepository, name)
            assert relation(states) == law(x, y), f"{name} disagrees with its relation"


@pytest.mark.parametrize(("law", "case", "verdict", "head", "tail"), WITNESSES,
                         ids=[f"{law}-{case.replace(' ', '_')}" for law, case, _, _, _ in WITNESSES])
def test_the_relations_reject_what_the_laws_reject(law, case, verdict, head, tail):
    """A pair each law rejects and a near miss it admits, decided through the abstraction.

    This is the only reading of the four laws in this file that does not go through a space.  It is
    needed because the spaces do not reach every law: two of the three forbid nothing at all, and
    the third reaches one law of four, so a relation that always said yes could sit in the first
    law's slot and no count would notice.

    Every relation is asked about every witness, not only about the one the witness was built for.
    A relation that fires on another law's pattern would reject terms its law admits, and the space
    would silently lose them.  The relations are the ones the repository has wired, so the failure
    names the law whose slot is wrong.

    Args:
        law (str): The law the pair was built for.
        case (str): What the pair is, for the failure message.
        verdict (bool): What that law has to say about the pair.
        head (Tree): The first hole's subterm.
        tail (Tree): The second hole's subterm.
    """
    assert getattr(DAMGrepository, law)(head, tail) is verdict, f"{law} does not see {case}"
    states = {"x": state_of(head, alpha), "y": state_of(tail, alpha)}
    for name, relation in LAWS:
        assert relation(states) is getattr(DAMGrepository, name)(head, tail), (
            f"{name} and its relation disagree on the witness for {law}"
        )


def test_every_law_reaches_the_clause_once_and_in_its_own_slot(recognizable):
    """Each of the four relations is attached to ``before_cons``, once, in law order.

    ``LAWS`` reads the four relations off ``RELATIONS``, so this is where ``RELATIONS`` itself is
    held against the four the shared module states, once each and in that order.  What the clause
    then does with them is read off the synthesized rules rather than off the clause that built
    them: naming an earlier relation twice in place of a later one leaves four constraints over one
    abstraction on the clause, which is all the spine comparison above can see.

    Args:
        recognizable (SolutionSpace): The recognizable program.
    """
    in_law_order = (R1, R2, R3, R4)
    assert tuple(RecognizableDAMGrepository.RELATIONS) == in_law_order, "RELATIONS is out of order"

    carriers = [
        rule
        for nonterminal in recognizable.nonterminals()
        for rule in recognizable.get(nonterminal) or ()
        if rule.predicates
    ]
    assert {str(rule.terminal) for rule in carriers} == {"before_cons"}
    assert len(carriers) == 30

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

    The bound is the first occupied size of this space and no more, because the walk is what the
    determinization is there to avoid: counting to 91 takes four times as long as counting to 82,
    and to 103 twelve times.  What the row says about the laws is nothing, since they forbid no
    term of this space at any size.  The reduced space below is where that is settled.

    Args:
        original (SolutionSpace): The coupled program.
        determinization (Determinization): The compiled product program.
    """
    bound = 82

    counted = branch_counts(generator_query(original, damg_targets.target_len_3), bound, term_size)
    assert branch_multiplicities(counted) == {}, "the coupled space derives some term twice"

    table = size_table(determinization.space, bound)
    row = {
        size: table.of(determinization.start, size)
        for size in range(bound + 1)
        if table.of(determinization.start, size)
    }
    assert row == dict(counted.counts)
    assert row == {82: 14416}


def test_the_whole_space_is_within_reach_of_the_table(determinization):
    """The gain, stated as the thing the coupled program cannot do at all.

    The table reads every size of the space at once, and it takes under a tenth of a second to do
    it.  Walking the coupled program to the largest of those sizes is not a slower way to the same
    answer, it is out of reach: the test above stops at the smallest occupied size because the
    walk to the second one already costs four times as much.

    Args:
        determinization (Determinization): The compiled product program.
    """
    bound = 200
    table = size_table(determinization.space, bound)
    row = {
        size: table.of(determinization.start, size)
        for size in range(bound + 1)
        if table.of(determinization.start, size)
    }
    assert len(row) == 25
    assert max(row) == 187
    assert sum(row.values()) == 27308788


@pytest.fixture(scope="module")
def reduced():
    """Build the three programs of the reduced configuration once for the module.

    Returns:
        tuple: The coupled program, the determinized recognizable one, and the determinized one
        whose relations admit everything.
    """
    target = damg_targets.target_len_4
    coupled = build(DAMGrepository, REDUCED_PARAMETER_SETS, target)
    recognizable = build(RecognizableDAMGrepository, REDUCED_PARAMETER_SETS, target)
    admitting = build(AdmitEverything, REDUCED_PARAMETER_SETS, target)
    return coupled, determinize(recognizable, target), determinize(admitting, target)


def reduced_row(determinization):
    """Read the sizes of a determinized reduced program up to the size the third law first reaches.

    Args:
        determinization (Determinization): The compiled product program.

    Returns:
        dict[int, int]: The occupied sizes and their counts.
    """
    table = size_table(determinization.space, REDUCED_BOUND)
    return {
        size: table.of(determinization.start, size)
        for size in range(REDUCED_BOUND + 1)
        if table.of(determinization.start, size)
    }


def test_the_two_forms_agree_where_the_laws_forbid_something(reduced):
    """The equality, on the one space the coupled walk reaches in which a law forbids something.

    It is the only place in this file where a count that is not empty of forbidden terms is held
    against the coupled walk.  The third program is the control: its relations admit everything, and
    it keeps 12 terms of size 124 that the laws throw out.  The coupled walk and the table agree
    with each other and both differ from it.

    Of the four laws only the third forbids anything of this space, so what this says of the
    restatement is that it forbids what the third law forbids and nothing besides.  Put a relation
    that always says yes in any of the other three slots and this test stays green.  The sharp
    space below is where the second and the fourth are reached, and the witnesses are where the
    first is.

    Args:
        reduced (tuple): The coupled program and the two determinized ones.
    """
    coupled, determinization, admitting = reduced
    counted = branch_counts(
        generator_query(coupled, damg_targets.target_len_4), REDUCED_BOUND, term_size
    )
    assert branch_multiplicities(counted) == {}, "the coupled space derives some term twice"

    assert dict(counted.counts) == reduced_row(determinization)
    assert dict(counted.counts) == {106: 7536, 115: 264}
    assert reduced_row(admitting) == {106: 7536, 115: 264, 124: 12}


def test_the_laws_forbid_nothing_of_the_searched_space(recognizable, determinization):
    """On ``target_len_3`` the two forms would agree even if the relations were wrong.

    This is not a property of the restatement, it is a property of the target, and it is the reason
    the test above exists.  Stated here so that it is a measured fact rather than a remark: the
    program whose relations admit everything determinizes to the same states and the same rules.

    Args:
        recognizable (SolutionSpace): The recognizable program.
        determinization (Determinization): The compiled product program.
    """
    admitting = determinize(
        build(AdmitEverything, PARAMETER_SETS, damg_targets.target_len_3),
        damg_targets.target_len_3,
    )
    assert admitting.state_count == determinization.state_count
    assert rule_count(admitting.space) == rule_count(determinization.space)


#: A second cut of ``target_len_4``, two feature dimensions rather than one.  It is the cheapest
#: space found here in which more than one law forbids anything: at size 180 the second and the
#: fourth forbid 12 terms each, which the one-dimension cut above never reaches because it has no
#: term of that size.  Only its tables are read and never the coupled walk, which is what makes it
#: affordable: walking it to 124 takes twelve seconds against two for the cut above.
SHARP_PARAMETER_SETS = {
    "linear_feature_dimensions": [2, 3],
    "constant_values": [0, 1],
    "learning_rate_values": [1e-2],
    "n_epoch_values": [2000],
}

#: The bound of the sharp space's comparison.  It is the largest size up to 200 at which any law of
#: this repository forbids a term.
SHARP_BOUND = 180

#: What each law forbids of the sharp space, as the terms that come back when a relation admitting
#: everything takes that law's slot.  The first law forbids nothing here, nothing on the reduced
#: space and nothing on the searched one, so no count in this file reaches it and the witnesses are
#: what pin it.  Its empty row says that, rather than hiding it.
FORBIDDEN_BY_LAW = {
    "swaplaw1": {},
    "swaplaw2": {180: 12},
    "swaplaw3": {124: 36},
    "swaplaw4": {180: 12},
}


def sharp_row(relations):
    """Read the occupied sizes of the sharp space, built with one given tuple of relations.

    Args:
        relations (tuple): The four relations to hand the four constraints, in law order.

    Returns:
        dict[int, int]: The occupied sizes up to ``SHARP_BOUND`` and their counts.
    """
    probe = type("Probe", (RecognizableDAMGrepository,), {"RELATIONS": relations})
    target = damg_targets.target_len_4
    determinization = determinize(build(probe, SHARP_PARAMETER_SETS, target), target)
    table = size_table(determinization.space, SHARP_BOUND)
    return {
        size: table.of(determinization.start, size)
        for size in range(SHARP_BOUND + 1)
        if table.of(determinization.start, size)
    }


@pytest.fixture(scope="module")
def sharp():
    """Read the sharp space once, as the four laws leave it.

    Returns:
        dict[int, int]: The occupied sizes and their counts.
    """
    return sharp_row(RecognizableDAMGrepository.RELATIONS)


@pytest.mark.parametrize("law", LAW_NAMES)
def test_each_law_forbids_of_the_sharp_space_what_it_is_meant_to(law, sharp):
    """Put a relation that always says yes in one law's slot and count what comes back.

    This is the sharpness axis, one law at a time, and it is the reason the file carries a third
    configuration.  The comparison on the reduced space replaces all four slots at once, and on that
    space only the third law forbids anything, so it cannot tell which of the four is doing the
    work.  Here each slot is answered for on its own: the third law is worth 36 terms of size 124,
    and the second and the fourth 12 terms of size 180 each.

    The first law is the one no space in this file reaches, and its row is empty.  That is a
    measured fact about these three spaces and not a gap left open: what pins the first law is
    ``test_the_relations_reject_what_the_laws_reject``.

    Args:
        law (str): The law whose slot is handed the relation that admits everything.
        sharp (dict): The occupied sizes of the space with all four laws in place.
    """
    relations = list(RecognizableDAMGrepository.RELATIONS)
    relations[LAW_NAMES.index(law)] = admit_everything
    without = sharp_row(tuple(relations))

    came_back = {
        size: count - sharp.get(size, 0)
        for size, count in without.items()
        if count != sharp.get(size)
    }
    assert came_back == FORBIDDEN_BY_LAW[law]
