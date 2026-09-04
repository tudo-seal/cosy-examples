"""The DAMG repository with its four swap laws stated so that they can be compiled away.

``damg_repo.py`` states the four swap laws as term predicates over two sibling holes
(``swaplaw1`` to ``swaplaw4``, attached to ``before_cons``).  That is what a user writes, and it is
what makes the synthesized program impossible to count from the program alone.  For several holes
the residual of a partial term need not be the product of the single-hole residuals: a predicate
that relates two holes leaves a subterm admissible at the one hole only for certain fillings of the
other, so the residual there is a genuine relation and no table indexed by the non-terminal can be
right about it.  ``cosy.search.counting.decomposable_or_raise`` names those clauses and refuses,
rather than returning counts that are quietly too large.

This module states the *same four laws* as a finite abstraction with a relation on its values.  The
predicate is then derived as the relation after the abstraction instead of stated beside it, which
is the condition ``cosy.core.recognizable`` calls **(REC)**, and under it
``cosy.search.determinize.determinize`` pushes the predicate into the non-terminals.  What comes out
carries no predicate over a hole and is countable from the program.  Three of the four relations
state their law as the coupled form states it.  ``R1`` states less, and ``recognizable_swap_laws``
says how much less.

**What of that is DAMG.**  The alphabet, and nothing else.  ``alpha`` and the four relations are the
same wherever these laws are stated, and they live in
``bayesian_optimization/examples/recognizable_swap_laws.py``, which
``cnn_damg_nas/recognizable_cnn_damg_repo.py`` reads too.  Both repositories inherit the four laws
from ``bayesian_optimization/examples/swap_laws.py``, so what is written here is the list of
terminals the laws never name, and a subclass of ``DAMGrepository`` that attaches the laws the other
way round.

**Why this is a second class rather than a change to DAMGrepository.**  The coupled form is the one
a user writes, and it is the form ``damg_example.py`` runs, so it stays.  What the two forms differ
in is one method: ``DAMGrepository._append_swap_laws`` puts the four laws on the ``before_cons``
clause as predicates, and the override below puts them there as recognizable constraints.  The
clause itself, with its parameter sets and the constraints on them, is written once and inherited,
so there is nothing left that could drift apart.

**What the second form buys.**  On the space of ``target_len_3``, the target ``damg_example.py``
searches, the coupled program carries 30 clauses no table can read and the recognizable one carries
none.  Determinizing it costs 735 states and 1486 rules, against 88 non-terminals and 404 rules in
the program it comes from, and takes about a tenth of a second.  The two count the same terms size
for size, and that is where the gain is.  The table reads all 25 occupied sizes of that space at
once, up to the largest term at 187 symbols and 27 308 788 terms in all, in under a tenth of a
second.  Walking the retained search tree of the coupled program is not a slower route to the same
answer: it counts up to size 82 in about 5 seconds, up to 103 in about a minute, and up to 131 in
more than the 200 seconds it was given.  ``tests/test_recognizable_damg_repo.py`` holds the counts,
and stops the walk at the first occupied size for exactly that reason.

**Why compile the laws rather than delete them.**  On ``target_len_2`` and ``target_len_3`` the four
laws forbid nothing at any size, so every number above reads the same with relations that admit
everything in their place, and no test on those two targets can tell a correct relation from one
that always says yes.  ``target_len_4`` is where three of the four bite: the third law forbids terms
of size 124, the second and the fourth forbid terms of size 180, and the first forbids nothing at
any size of any of those spaces.  ``tests/test_recognizable_damg_repo.py`` holds the three against a
count and the first against hand-built pairs, because a count cannot reach it.

**Determinize it, do not search it.**  The recognizable form is the slowest of the three to
enumerate, because ``cosy.core.recognizable.state_of`` folds ``alpha`` over the whole sibling term
again at every check, four times per ``before_cons`` node.  Over the first 20 000 terms of the
``target_len_3`` space it takes 8.5 s where the coupled program takes 1.3 s and the determinized one
1.2 s, best of three runs each.  Those three numbers move with the machine and their order does not.
What determinizing costs is the order of the stream: the coupled and the recognizable form
list the same terms in the same order, the determinized one does not, so a prefix of its stream is a
different set of terms.
"""

from __future__ import annotations

from bayesian_optimization.examples.damg_nas.damg_repo import DAMGrepository
from bayesian_optimization.examples.recognizable_swap_laws import (
    R1,
    R2,
    R3,
    R4,
    make_alpha,
    make_alphabet_check,
)

# The terminals of the DAMG repository the laws never name.  Kept as an explicit set so that a
# symbol the abstraction has never seen is a loud failure and not a silent ``OTHER``.
# tests/test_recognizable_damg_repo.py holds the list against the combinator table rather than
# against a space, because a terminal no configuration in use enumerates falls out of a list like
# this one without anything noticing.  That has happened to the CNN list next door.
OTHER_TERMINALS = frozenset(
    {
        "linear_layer",
        "sigmoid",
        "relu",
        "tanh",
        "sum",
        "product",
        "copy",
        "mse_loss",
        "l1loss",
        "adam_optimizer",
        "learner",
    }
)


#: The abstraction and the alphabet guard, each built once for this repository.  Once, because the
#: determinization keys its product on the abstraction and compares two of them by identity, so a
#: second closure over the same terminals would widen the product for nothing.
alpha = make_alpha(OTHER_TERMINALS)
check_alphabet = make_alphabet_check(OTHER_TERMINALS)


class RecognizableDAMGrepository(DAMGrepository):
    """``DAMGrepository`` with the four swap laws stated as recognizable constraints.

    Everything but the attachment of the laws is inherited: the combinators, the parameter sets,
    the type structure, the ``before_cons`` clause and ``swaplaw1`` to ``swaplaw4`` themselves.
    The four laws that the base class puts on the clause as predicates over two of its holes go on
    as ``(alpha, R)`` here instead.

    Attributes:
        RELATIONS (tuple): The four relations, in law order.  A subclass may replace them, which is
            how one measures what the laws carve out of the space: hand over relations that admit
            everything and count the difference.  Replacing them is a *measurement* device and it
            changes the language.  The abstraction stays what it is.
    """

    RELATIONS = (R1, R2, R3, R4)

    def _append_swap_laws(self, clause):
        """Attach the four laws as a relation on the abstraction of the two holes.

        Args:
            clause (SpecificationBuilder): The ``before_cons`` clause, both holes introduced.

        Returns:
            SpecificationBuilder: The clause with the four laws on it.
        """
        return (
            clause
            .recognizable_constraint(alpha, self.RELATIONS[0])
            .recognizable_constraint(alpha, self.RELATIONS[1])
            .recognizable_constraint(alpha, self.RELATIONS[2])
            .recognizable_constraint(alpha, self.RELATIONS[3])
        )
