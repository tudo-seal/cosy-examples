"""The CNN repository with its four swap laws stated so that they can be compiled away.

``cnn_damg_repo.py`` states the four swap laws as term predicates over two sibling holes
(``swaplaw1`` to ``swaplaw4``, attached to ``before_cons``).  That is what a user writes, and it is
what makes the synthesized program impossible to count from the program alone.  For several holes
the residual of a partial term need not be the product of the single-hole residuals: a predicate
that relates two holes leaves a subterm admissible at the one hole only for certain fillings of the
other, so the residual there is a genuine relation and no table indexed by the non-terminal can be
right about it.  ``cosy.search.counting.decomposable_or_raise`` names those clauses and refuses,
rather than returning counts that are quietly too large.

This module states the *same four laws* in the form ``cosy.core.recognizable`` calls **(REC)**: a
finite, bottom-up computable abstraction ``alpha`` from terms into a finite set, a relation ``R``
over its values, and the predicate *derived* as ``R`` after ``alpha`` instead of stated beside it.
That is the class of recognizable tree relations, and ``cosy.search.determinize.determinize`` pushes
a predicate of that class into the non-terminals.  What comes out carries no predicate over a hole
and is countable from the program.

**Why this is a second file rather than a change to cnn_damg_repo.py.**  ``cnn_damg_repo.py`` is
the repository the experiments of this example are stated against, and leaving it untouched keeps
their numbers comparable.  The price is the duplicated ``before_cons`` clause below, and that
price is paid with a test rather than with care: ``tests/test_recognizable_cnn_repo.py`` compares
the two programs rule for rule and fails on any drift.  Apart from the four constraint calls at its
end, the copy differs from its original in no line.

**What the second form buys.**  At structure length 2 the coupled program carries eight clauses no
table can read, and the recognizable one carries none.  Determinizing it costs 1044 states and 2605
rules, against 307 non-terminals and 856 rules in the program it comes from, and the two count the
same terms size for size.  ``tests/test_recognizable_cnn_repo.py`` holds all of that.

**Why compile the laws rather than delete them.**  At structure length 2 they forbid nothing, so
every number above reads the same with relations that admit everything in their place.  One length
further up they bite: the determinized program has 419 818 rules there where the admit-everything
relations give 419 827, and the smallest term the laws forbid has size 104.  That space takes about
a minute to build, which is why the tests stay at length 2 and pin the laws with pairs built by
hand instead.

**Determinize it, do not search it.**  The recognizable form is the slower of the two to enumerate,
because ``cosy.core.recognizable.state_of`` folds ``alpha`` over the whole sibling term again at
every check, four times per ``before_cons`` node.  Enumerate the first 20 000 terms of each of the
three forms with ``depth_first(generator_query(space, start))`` and time them: the recognizable form
runs four to five times as long as the coupled program, and the determinized program, which has no
predicate left to check, runs no longer than the coupled program.  Both comparisons move from run to
run, so the first is worth reading as an order of magnitude and the second only as a bound.  The
gain is in the counting, where the tree form has to build the retained search tree and the table
only reads the program.  What the determinized program does not keep is the order: the coupled and
the recognizable form list the same terms in the same order, the determinized one does not, so a
prefix of its stream is a different set of terms.

**The abstraction.**  ``alpha`` is the term truncated to the (suffix-closed) read set of the four
laws: every symbol they never test for collapses to one marker, every literal they never compare is
forgotten, and the arity of every inspected node is kept so that ``_require_children_count`` is
decided on the state exactly as the law decides it on the term.  The truncation is *slot-dependent*,
meaning that what a parent records of a child is only what a law can read below that particular
slot, and that is what keeps the carrier small enough to determinize.

``check_alphabet`` states the assumptions the abstraction rests on and refuses the space when they
do not hold: no unknown terminal, no disagreement between the laws' substring test
(``token in root``) and the abstraction's equality test, and no string literal colliding with a
terminal name.  Call it before determinizing a space this module has not seen.
"""

from __future__ import annotations

from cosy.core import SpecificationBuilder
from cosy.core.types import Constructor, DataGroup, Literal, Var

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository

MISS = ("?",)
OTHER = ("O",)
CUT = ("~",)

BS = "beside_singleton"
BC = "beside_cons"
BFS = "before_singleton"
BFC = "before_cons"
SW = "swap"
ED = "edges"

TOKENS = (BS, BC, BFS, BFC, SW, ED)

# The terminals of the CNN repository the laws never name.  Kept as an explicit set so that a
# symbol the abstraction has never seen is a loud failure and not a silent ``OTHER``.
OTHER_TERMINALS = frozenset(
    {
        "conv2d",
        "linear_layer",
        "maxpool2d",
        "batchnorm2d",
        # sum, product and copy were missing here for a while, although they have been in the
        # specification all along. They went unnoticed because no configuration in use enumerates
        # them: they need an input or an output feature of 1, and the feature sets in use start at
        # 10, so nothing ever built a term containing one. A configuration that did would have
        # failed here and only here, and only where something calls check_alphabet, so it would
        # have looked flaky rather than incomplete.
        "sum",
        "product",
        "copy",
        "relu",
        "sigmoid",
        "tanh",
        "learner",
        "cross_entropy_loss",
        "adam_optimizer",
        "sgd_optimizer",
        "cosine_annealing_lr",
        "no_scheduler",
    }
)

KNOWN_TERMINALS = frozenset(TOKENS) | OTHER_TERMINALS


def _at(states, index):
    """Return the state of child ``index``, or MISS when the node has no such child."""
    return states[index] if index < len(states) else MISS


# ----------------------------------------------------------------------------------------------
# slot restrictors: what a parent keeps of a child, per slot
# ----------------------------------------------------------------------------------------------
def _k_swap_or_edges(q):
    """beside_singleton[4] / beside_cons[9]: the laws test ``is swap`` and ``is edges`` there."""
    if q[0] in ("SW", "ED"):
        return q
    return CUT


def _k_swap_only(q):
    """A slot where only ``is swap`` is ever tested."""
    if q[0] == "SW":
        return q
    return CUT


def _bs_nolits(q):
    """A beside_singleton in a slot where no law reads its literals 0 and 1."""
    if q[0] == "BS":
        return ("BS", q[1], None, None, q[4])
    return CUT


def _bs_swaponly(q):
    """A beside_singleton in a slot where only ``[4] is swap`` is reachable."""
    if q[0] == "BS":
        return ("BS", q[1], None, None, _k_swap_only(q[4]))
    return CUT


def _k10(q):
    """beside_cons[10]: the whole beside_singleton is reachable there (law1 reads [0] and [1])."""
    if q[0] == "BS":
        return q
    return CUT


def _bc_nolits(q):
    """A beside_cons in a slot where no law reads its literals 1 and 4, nor [10][0], [10][1]."""
    if q[0] == "BC":
        return ("BC", q[1], None, None, q[4], _bs_nolits(q[5]))
    return CUT


def _k5(q):
    """before_singleton[5]: law2/law4 reach a beside_cons, law1/law3 a beside_singleton."""
    if q[0] == "BS":
        return _bs_swaponly(q)
    if q[0] == "BC":
        return _bc_nolits(q)
    return CUT


def _k7(q):
    """before_cons[7]: law1 reads the beside_cons whole, law3 only ``[4] is swap``."""
    if q[0] == "BC":
        return q
    if q[0] == "BS":
        return _bs_swaponly(q)
    return CUT


def _k8(q):
    """before_cons[8]: only law1 descends, and only into a before_singleton's beside_singleton."""
    if q[0] == "BFS":
        inner = q[2]
        return ("BFS", q[1], inner if inner[0] == "BS" else CUT)
    return CUT


# ----------------------------------------------------------------------------------------------
# alpha
# ----------------------------------------------------------------------------------------------
def alpha(symbol, child_states):
    """The Sigma-algebra: ``(symbol, states of the arguments) -> state``.

    Total over the whole alphabet: a literal value is a nullary symbol and receives ``()``.

    Args:
        symbol: The terminal name or the literal value.
        child_states: The states of the arguments, in clause order.

    Returns:
        The state.
    """
    if symbol == SW:
        return ("SW", len(child_states), _at(child_states, 1), _at(child_states, 2))
    if symbol == ED:
        return ("ED", len(child_states), _at(child_states, 0))
    if symbol == BS:
        return (
            "BS",
            len(child_states),
            _at(child_states, 0),
            _at(child_states, 1),
            _k_swap_or_edges(_at(child_states, 4)),
        )
    if symbol == BC:
        return (
            "BC",
            len(child_states),
            _at(child_states, 1),
            _at(child_states, 4),
            _k_swap_or_edges(_at(child_states, 9)),
            _k10(_at(child_states, 10)),
        )
    if symbol == BFS:
        return ("BFS", len(child_states), _k5(_at(child_states, 5)))
    if symbol == BFC:
        return ("BFC", len(child_states), _k7(_at(child_states, 7)), _k8(_at(child_states, 8)))
    if type(symbol) is str and symbol in OTHER_TERMINALS:
        return OTHER
    if child_states:
        # A symbol with arguments that is not a terminal this abstraction knows.  Silently
        # collapsing it would be exactly the kind of unrecorded assumption the plan forbids.
        msg = (
            f"alpha met the unknown {len(child_states)}-ary symbol {symbol!r}; the abstraction knows "
            f"{len(KNOWN_TERMINALS)} terminals and this is not one of them, so it is out of date"
        )
        raise ValueError(msg)
    return ("V", symbol)


# ----------------------------------------------------------------------------------------------
# R: one relation per law, a statement-for-statement mirror
# ----------------------------------------------------------------------------------------------
def _req(q, expected):
    """Mirror ``_require_children_count``: the arity is carried in the state."""
    if q[1] != expected:
        raise ValueError("Derivation trees have not the expected shape.")


def _eq(a, b):
    """Mirror ``==`` on two literal cells.

    Raises:
        ValueError: If a cell the law compares is not a literal.  The abstraction keeps literals
            exactly, so equality on the cells is equality on the terms.  Where a compared cell is
            not a literal, the state cannot answer, and it says so rather than guessing.
    """
    if a[0] != "V" or b[0] != "V":
        msg = (
            f"a compared cell is not a literal: {a!r} vs {b!r}; the abstraction keeps compared "
            f"positions exactly and cannot decide equality of two subterms"
        )
        raise ValueError(msg)
    return a[1] == b[1]


def R1(substitution):
    """``swaplaw1`` on states."""
    head = substitution["x"]
    tail = substitution["y"]
    if head[0] == "BS" and tail[0] == "BFC":
        _req(head, 5)
        _req(tail, 9)
        left_term = head[4]
        right_head = tail[2]
        right_tail = tail[3]
        if left_term[0] == "SW" and right_head[0] == "BC" and right_tail[0] == "BFS":
            _req(left_term, 19)
            _req(right_head, 11)
            _req(right_tail, 6)
            m = left_term[2]
            n = left_term[3]
            x_n = right_head[2]
            x_p = right_head[3]
            right_head_tail = right_head[5]
            right_tail_term = right_tail[2]
            if right_head_tail[0] == "BS" and right_tail_term[0] == "BS":
                _req(right_head_tail, 5)
                _req(right_tail_term, 5)
                y_m = right_head_tail[2]
                y_q = right_head_tail[3]
                right_swap = right_tail_term[4]
                if right_swap[0] == "SW":
                    _req(right_swap, 19)
                    p = right_swap[2]
                    q = right_swap[3]
                    if _eq(m, y_m) and _eq(n, x_n) and _eq(p, x_p) and _eq(q, y_q):
                        return False
        elif head[0] == "SW":
            # The second branch of swaplaw1 is dead code: its enclosing ``if`` already requires
            # "beside_singleton" in head.root, and "beside_singleton" in "swap" is False for every
            # string.  Reached, it would mean the guard analysis is wrong, so it says so.
            msg = "swaplaw1's second branch is unreachable by construction but was reached"
            raise AssertionError(msg)
    return True


def _law2_body(left_head, left_tail, right_term):
    """The shared inner body of ``swaplaw2``'s two branches."""
    if left_head[0] == "SW" and left_tail[0] == "BS" and right_term[0] == "BC":
        _req(left_head, 19)
        _req(left_tail, 5)
        _req(right_term, 11)
        m = left_head[2]
        n = left_head[3]
        left_tail_term = left_tail[4]
        right_head = right_term[4]
        right_tail = right_term[5]
        if left_tail_term[0] == "ED" and right_head[0] == "ED" and right_tail[0] == "BS":
            _req(left_tail_term, 17)
            _req(right_head, 17)
            _req(right_tail, 5)
            p = left_tail_term[2]
            right_n = right_head[2]
            right_tail_term = right_tail[4]
            if right_tail_term[0] == "SW" and _eq(n, right_n):
                _req(right_tail_term, 19)
                right_m = right_tail_term[2]
                right_p = right_tail_term[3]
                if _eq(m, right_m) and _eq(p, right_p):
                    return False
    return True


def R2(substitution):
    """``swaplaw2`` on states."""
    head = substitution["x"]
    tail = substitution["y"]
    if head[0] == "BC" and tail[0] == "BFS":
        _req(head, 11)
        _req(tail, 6)
        return _law2_body(head[4], head[5], tail[2])
    if head[0] == "BC" and tail[0] == "BFC":
        _req(head, 11)
        _req(tail, 9)
        return _law2_body(head[4], head[5], tail[2])
    return True


def _law3_body(left_term, right_term):
    """The shared inner body of ``swaplaw3``'s two branches."""
    if left_term[0] == "SW" and right_term[0] == "BS":
        _req(left_term, 19)
        _req(right_term, 5)
        m = left_term[2]
        n = left_term[3]
        right_beside = right_term[4]
        if right_beside[0] == "SW":
            _req(right_beside, 19)
            right_n = right_beside[2]
            right_m = right_beside[3]
            if _eq(m, right_m) and _eq(n, right_n):
                return False
    return True


def R3(substitution):
    """``swaplaw3`` on states."""
    head = substitution["x"]
    tail = substitution["y"]
    if head[0] == "BS" and tail[0] == "BFS":
        _req(head, 5)
        _req(tail, 6)
        return _law3_body(head[4], tail[2])
    if head[0] == "BS" and tail[0] == "BFC":
        _req(head, 5)
        _req(tail, 9)
        return _law3_body(head[4], tail[2])
    return True


def _law4_body(left_head, left_tail, right_term):
    """The shared inner body of ``swaplaw4``'s two branches."""
    if left_head[0] == "ED" and left_tail[0] == "BS" and right_term[0] == "BC":
        _req(left_head, 17)
        _req(left_tail, 5)
        _req(right_term, 11)
        m = left_head[2]
        left_tail_term = left_tail[4]
        right_head = right_term[4]
        right_tail = right_term[5]
        if left_tail_term[0] == "SW" and right_head[0] == "SW" and right_tail[0] == "BS":
            _req(left_tail_term, 19)
            _req(right_head, 19)
            _req(right_tail, 5)
            n = left_tail_term[2]
            p = left_tail_term[3]
            right_m = right_head[2]
            right_p = right_head[3]
            right_tail_term = right_tail[4]
            if right_tail_term[0] == "ED" and _eq(m, right_m) and _eq(p, right_p):
                _req(right_tail_term, 17)
                right_n = right_tail_term[2]
                if _eq(n, right_n):
                    return False
    return True


def R4(substitution):
    """``swaplaw4`` on states."""
    head = substitution["x"]
    tail = substitution["y"]
    if head[0] == "BC" and tail[0] == "BFS":
        _req(head, 11)
        _req(tail, 6)
        return _law4_body(head[4], head[5], tail[2])
    if head[0] == "BC" and tail[0] == "BFC":
        _req(head, 11)
        _req(tail, 9)
        return _law4_body(head[4], head[5], tail[2])
    return True


def check_alphabet(space):
    """Verify the abstraction covers the program's alphabet, and that the tokens are unambiguous.

    Args:
        space: The synthesized solution space.

    Returns:
        dict: What was found: the terminals, and the substring/equality check on the tokens.

    Raises:
        ValueError: If a terminal is unknown to the abstraction, or if the substring test
            ``token in root`` and the equality test ``root == token`` disagree anywhere.  The
            laws use the one test and the abstraction the other, and it identifies them.
    """
    terminals = set()
    literals = set()
    for nonterminal in space.nonterminals():
        for rule in space.get(nonterminal) or ():
            terminals.add(rule.terminal)
            for argument in rule.arguments:
                value = getattr(argument, "value", None)
                if type(value) is str:
                    literals.add(value)
    unknown = terminals - KNOWN_TERMINALS
    if unknown:
        msg = f"the abstraction does not know these terminals of the program: {sorted(unknown)}"
        raise ValueError(msg)
    disagreements = [
        (root, token)
        for root in terminals
        for token in TOKENS
        if (token in root) != (root == token)
    ]
    if disagreements:
        msg = (
            f"the laws' substring test and the abstraction's equality test disagree on "
            f"{disagreements}"
        )
        raise ValueError(msg)
    colliding = literals & KNOWN_TERMINALS
    if colliding:
        msg = f"a string literal of the program collides with a terminal name: {sorted(colliding)}"
        raise ValueError(msg)
    return {
        "terminals": sorted(terminals),
        "string_literals": sorted(literals),
        "substring_equals_equality": True,
    }


# ----------------------------------------------------------------------------------------------
# the repository
# ----------------------------------------------------------------------------------------------
class RecognizableCNNrepository(CNNrepository):
    """``CNNrepository`` with the four swap laws stated as recognizable constraints.

    The combinators, the parameter sets, the type structure and ``swaplaw1`` to ``swaplaw4``
    themselves are inherited unchanged.  Only the ``before_cons`` clause is rebuilt, because it is
    the only clause carrying a predicate over a hole, and the four ``.constraint`` calls at its end
    become four ``.recognizable_constraint`` calls.

    The rebuilt clause is a copy, so it can drift from the original.  That is what
    ``tests/test_recognizable_cnn_repo.py`` is for: it compares the two programs rule for rule and
    fails when they differ, which turns drift into a red test instead of a quietly different
    search space.

    Attributes:
        RELATIONS (tuple): The four relations, in law order.  A subclass may replace them, which is
            how one measures what the laws carve out of the space: hand over relations that admit
            everything and count the difference.  Replacing them is a *measurement* device and it
            changes the language.  The abstraction stays what it is.
    """

    RELATIONS = (R1, R2, R3, R4)

    def specification(self):
        """Return the specification with ``before_cons`` restated over the abstraction.

        Returns:
            dict: The combinator specifications, ready for ``Synthesizer``.
        """
        specifications = super().specification()
        specifications["before_cons"] = self._recognizable_before_cons()
        return specifications

    def _recognizable_before_cons(self):
        """Rebuild the ``before_cons`` clause with the laws stated as ``(alpha, R)``.

        A copy of the clause in ``CNNrepository.specification``, with its final four
        ``.constraint(lambda v: self.swaplawN(v["x"], v["y"]))`` replaced by
        ``.recognizable_constraint(alpha, R_N)``.  The parameter sets are rebuilt the same way the
        original builds them, since they are local to that method.

        Returns:
            Specification: The clause.
        """
        labels = self.Label(
            self.dimensions,
            self.linear_feature_dimensions,
            self.constant_values,
            self.channel_dimensions,
            self.height_width_dimensions,
            self.kernel_dimensions,
            self.stride_values,
            self.padding_values,
            self.max_lin_layer_dim,
            self.pooling_kernel_dimensions,
        )
        para_labels = self.Para(labels, self.dimensions)
        paratuples = self.ParaTuples(para_labels, max_length=self.max_parallel_width)
        paratupletuples = self.ParaTupleTuples(paratuples)
        dimension = DataGroup("dimension", self.dimensions)

        return (
            SpecificationBuilder()
            .parameter("i", dimension)
            .parameter("j", dimension)
            .parameter("o", dimension)
            .parameter("request", paratupletuples)
            .parameter("ls", paratupletuples, lambda v: [paratupletuples.normalize(v["request"])])
            .parameter_constraint(lambda v: v["ls"] is not None and len(v["ls"]) > 1)
            .parameter("head", paratuples, lambda v: [v["ls"][0]])
            .parameter_constraint(lambda v: v["head"] is None or
                                            (
                                                (v["i"] == sum([t[1] for t in v["head"]])
                                                 if None not in [t for t in v["head"]]
                                                    and None not in [t[1] for t in v["head"]]
                                                 else v["i"] > sum([t[1] for t in v["head"]
                                                                    if t is not None and t[1] is not None]))
                                                and (v["j"] == sum([t[2] for t in v["head"]])
                                                     if None not in [t for t in v["head"]]
                                                        and None not in [t[2] for t in v["head"]]
                                                     else v["j"] > sum([t[2] for t in v["head"]
                                                                        if t is not None and t[2] is not None]))
                                            )
                                  )
            .parameter("tail", paratupletuples, lambda v: [v["ls"][1:]])
            .parameter_constraint(lambda v: v["tail"] is None or
                                            (
                                                    (len(v["tail"]) > 0) and
                                                    (
                                                            v["tail"][0] is None or
                                                            (
                                                                v["j"] == sum([t[1] for t in v["tail"][0]])
                                                                if None not in [t for t in v["tail"][0]]
                                                                   and None not in [t[1] for t in v["tail"][0]]
                                                                else v["j"] > sum([t[1] for t in v["tail"][0]
                                                                                   if t is not None
                                                                                   and t[1] is not None])
                                                             )
                                                    ) and
                                                    (
                                                            v["tail"][-1] is None or
                                                            (v["o"] == sum([t[2] for t in v["tail"][-1]])
                                                             if None not in [t for t in v["tail"][-1]]
                                                                and None not in [t[2] for t in v["tail"][-1]]
                                                             else v["o"] > sum([t[2] for t in v["tail"][-1]
                                                                                if t is not None
                                                                                and t[2] is not None]))
                                                    )
                                            )
                                  )
            .argument("x", Constructor("DAG_parallel",
                                       Constructor("input", Var("i"))
                                       & Constructor("output", Var("j"))
                                       & Constructor("structure", Var("head"))) & Constructor("non_ID"))
            .argument("y", Constructor("DAG",
                                       Constructor("input", Var("j"))
                                       & Constructor("output", Var("o"))
                                       & Constructor("structure", Var("tail"))))
            .recognizable_constraint(alpha, self.RELATIONS[0])
            .recognizable_constraint(alpha, self.RELATIONS[1])
            .recognizable_constraint(alpha, self.RELATIONS[2])
            .recognizable_constraint(alpha, self.RELATIONS[3])
            .suffix(Constructor("DAG",
                                Constructor("input", Var("i"))
                                & Constructor("input", Literal(None))
                                & Constructor("output", Var("o"))
                                & Constructor("output", Literal(None))
                                & Constructor("structure", Var("request"))))
        )
