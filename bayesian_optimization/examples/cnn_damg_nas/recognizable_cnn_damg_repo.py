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

**What of that is CNN.**  The alphabet, and nothing else.  ``alpha`` and the four relations are the
same wherever these laws are stated, and they live in
``bayesian_optimization/examples/recognizable_swap_laws.py``.  What stands here is the list of
terminals the laws never name, and a subclass of ``CNNrepository`` that attaches the laws the other
way round.

**Why this is a second class rather than a change to CNNrepository.**  The coupled form is the one
a user writes, and it is the form the experiments of this example are stated against, so it stays.
What the two forms differ in is one method: ``CNNrepository._append_swap_laws`` puts the four laws
on the ``before_cons`` clause as predicates, and the override below puts them there as recognizable
constraints.  The clause itself, with its parameter sets and the constraints on them, is written
once and inherited, so there is nothing left that could drift apart.
``tests/test_recognizable_cnn_repo.py`` holds the two programs against each other anyway, rule for
rule: an override is cheaper to get wrong than a copy is, and what is left to get wrong is the
number of laws it attaches and the form it states them in.

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
"""

from __future__ import annotations

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository
from bayesian_optimization.examples.recognizable_swap_laws import (
    R1,
    R2,
    R3,
    R4,
    make_alpha,
    make_alphabet_check,
)

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


#: The abstraction and the alphabet guard, each built once for this repository.  Once, because the
#: determinization keys its product on the abstraction and compares two of them by identity, so a
#: second closure over the same terminals would widen the product for nothing.
alpha = make_alpha(OTHER_TERMINALS)
check_alphabet = make_alphabet_check(OTHER_TERMINALS)


class RecognizableCNNrepository(CNNrepository):
    """``CNNrepository`` with the four swap laws stated as recognizable constraints.

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
