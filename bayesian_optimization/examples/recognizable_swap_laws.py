"""The four swap laws of the string diagram theory, as an abstraction and four relations on it.

``cnn_damg_nas/cnn_damg_repo.py`` and ``damg_nas/damg_repo.py`` both forbid the left-hand sides of
the rewrite rules of the theory their combinators build terms of, and four of those rules they
forbid with a term predicate over two sibling holes: ``swaplaw1`` to ``swaplaw4``, attached to
``before_cons``.  Both inherit those four from ``swap_laws.py``, so the arities they require and
the child positions they read are the same numbers on both sides, and only the alphabet around them
differs.  This module is the half that does not differ.

It holds the abstraction ``alpha``, the four relations ``R1`` to ``R4`` that mirror the four laws
statement for statement, and two factories that take the alphabet: ``make_alpha`` and
``make_alphabet_check``.  A repository writes down the terminals its own laws never name and calls
the factories with them.  Nothing here imports cosy: a relation reads states, which are tuples, and
the alphabet check reads a solution space through its published methods.

**Call each factory once per repository, at module level.**
:func:`cosy.search.determinize.determinize` collects the distinct abstractions a program states and
builds the product over them, and it tells two of them apart with ``is`` or ``==``, which for a
closure is identity.  Four constraints naming four separately built closures are then four axes of
the product where one would do.

**The abstraction.**  ``alpha`` is the term truncated to the (suffix-closed) read set of the four
laws: every symbol they never test for collapses to one marker, every literal they never compare is
forgotten, and the arity of every inspected node is kept so that ``_require_children_count`` is
decided on the state exactly as the law decides it on the term.  The truncation is *slot-dependent*,
meaning that what a parent records of a child is only what a law can read below that particular
slot, and that is what keeps the carrier small enough to determinize.

**The alphabet check.**  ``make_alphabet_check`` returns the guard that states the assumptions the
abstraction rests on and refuses the space when they do not hold: no unknown terminal, no
disagreement between the laws' substring test (``token in root``) and the abstraction's equality
test, and no string literal colliding with a terminal name.  Call it before determinizing a space
this module has not seen.
"""

from __future__ import annotations

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
def make_alpha(other_terminals):
    """Return the abstraction over one repository's alphabet.

    Args:
        other_terminals (frozenset): The terminals of that repository the four laws never name.
            Whatever their arity, they all collapse to one marker, since no law reads below one.
            They are listed rather than assumed so that a terminal the abstraction has never
            seen is a loud failure and not a silent marker.

    Returns:
        The abstraction, as ``(symbol, states of the arguments) -> state``.
    """
    known_terminals = frozenset(TOKENS) | other_terminals

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
        if type(symbol) is str and symbol in other_terminals:
            return OTHER
        if child_states:
            # A symbol with arguments that is not a terminal this abstraction knows.  Collapsing
            # it silently would leave the abstraction quietly wrong about a term it has never seen.
            msg = (
                f"alpha met the unknown {len(child_states)}-ary symbol {symbol!r}; the abstraction"
                f" knows {len(known_terminals)} terminals and this is not one of them, so it is"
                f" out of date"
            )
            raise ValueError(msg)
        return ("V", symbol)

    return alpha


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


def make_alphabet_check(other_terminals):
    """Return the guard that holds a space against one repository's alphabet.

    Args:
        other_terminals (frozenset): The set ``make_alpha`` was given for that repository.

    Returns:
        The guard, which takes a solution space and reports what it found in it.
    """
    known_terminals = frozenset(TOKENS) | other_terminals

    def check_alphabet(space):
        """Verify the abstraction covers the alphabet, and that the tokens are unambiguous.

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
        unknown = terminals - known_terminals
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
        colliding = literals & known_terminals
        if colliding:
            msg = (
                f"a string literal of the program collides with a terminal name:"
                f" {sorted(colliding)}"
            )
            raise ValueError(msg)
        return {
            "terminals": sorted(terminals),
            "string_literals": sorted(literals),
            "substring_equals_equality": True,
        }

    return check_alphabet
