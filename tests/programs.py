"""Two readings of a repository, for holding a coupled form and a recognizable one against it.

``cnn_damg_nas`` and ``damg_nas`` each state their four swap laws twice: once as term predicates
over two holes of the ``before_cons`` clause, and once as relations on an abstraction of those two
subterms.  The second form is what can be determinized.  Both pairs are held together by the same
two questions, so the two readings live here rather than in one of the test files.

:func:`clause_spine` reads a clause before any synthesis and says what it introduces and in which
order, with its predicates named by kind but not by identity.  It is the reading that can see a law
that was dropped, added or attached in the wrong place, none of which changes a rule.

:func:`shape_of` reads a synthesized program and says which rules it has, without the predicates
they carry.  It is the reading that can see a subclass that changed more than the attachment of the
laws.
"""

from __future__ import annotations

from cosy.core.recognizable import RecognizableConstraint
from cosy.core.solution_space import NonTerminalArgument
from cosy.core.types import Abstraction, Implication


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


def shape_of(space):
    """Return the program's rules in a form that ignores the predicates.

    What a subclass can get wrong beyond the laws is a parameter set, a constraint on the literals,
    or an argument type, all of which change which rules the synthesis produces.  Rendering the
    rules without their predicates isolates exactly that: what remains has to be identical, because
    the only intended difference between the two forms is how the four laws are stated.

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
