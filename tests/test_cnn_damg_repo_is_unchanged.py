"""The CNN repository still builds the program it built when it arrived.

Nothing else in this repository runs ``cnn_damg_repo.py``.  The commits that follow this one
rewrite its prose, take the duplication out of its combinator table and repair one of its four swap
laws, and none of those touches has a test that would notice if it changed the synthesized program.
This module is that test.  It is scaffolding for the migration of the CNN example and goes away
with its last commit.

It compares against a record frozen at migration, in ``data/cnn_damg_repo_at_migration.txt``, so it
holds three things and nothing else.

The rules.  For three synthesized spaces it holds every non-terminal with its rules, and every rule
with its terminal and its argument list, a hole written as the non-terminal it comes from and a
constant as its ``repr``.  That is what moves when a parameter set, a parameter constraint or an
argument type changes.  Both sides are built in the same process against the same cosy, so an
upstream change to how a type prints moves the current side and the frozen side apart and has to be
regenerated, which is the price for a record that stays readable.

The clause spines.  A rule records what a target asked for, and all three targets below are pinned,
so a conjunct that only an open target would need is a conjunct no rule of theirs carries.  Turning
the ``output([None])`` of ``linear_layer`` into ``output([0])`` leaves all three spaces character
for character identical, and that is no equivalence: a target that leaves the output slot open no
longer reaches the combinator.  The second axis is therefore the clause itself, read off
``CNNrepository.specification()`` before any synthesis.  Per combinator it holds the parameters and
the predicates in the order the clause introduces them, closed by the suffix type, which carries
every conjunct verbatim.

The verdicts of the four swap laws.  A rule carries its predicates as closures, so the rule record
cannot look into them.  Switching a law off entirely leaves all three spaces character-identical.
The third axis is therefore a corpus of hand-built witnesses, each a pair of subterms that the law
either rejects or admits, with the verdict frozen beside it.

Three spaces, because each is blind where the others see.  The pinned VGG-11 chain is the space the
CNN experiment actually searches, and it is purely sequential: it reaches 16 of the 22 combinators
of ``CNNrepository.specification()`` and none of the parallel side.  The parallel chain frees one
merged position to width 2 and brings in ``swap``, ``beside_cons`` and ``edges``, the wiring
combinators the swap laws are about.  The small chain is a tiny configuration whose feature set
contains 1, which is the only way ``sum``, ``product`` and ``copy`` become inhabited, and it is the
only one of the three that reaches all 22.

Regenerate the frozen record with

    PYTHONPATH=. python tests/test_cnn_damg_repo_is_unchanged.py
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

import pytest
from cosy.core import Synthesizer
from cosy.core.solution_space import NonTerminalArgument
from cosy.core.tree import Tree

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_reference_architectures import (
    cifar10_vgg11_bn_repo,
)
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_targets import (
    VGG11_BN_MERGED_POSITIONS,
    VGG11_BN_POSITIONS,
    make_variance_target,
    make_vgg11_bn_target,
)
from tests.programs import clause_spine

FROZEN = Path(__file__).with_name("data") / "cnn_damg_repo_at_migration.txt"

#: The merged VGG chain with one repeated position opened to two parallel components.  One position
#: is enough: a single parallel composition puts ``beside_cons`` into the space, and with it the
#: wiring the swap laws constrain.  Position 12 is the cheapest place to open, and not because its
#: features are small.  Positions 13 and 14 are smaller in both features and reach no parallel
#: combinator at all.  Only eight of the fifteen bring ``swap``, ``beside_cons`` and ``edges`` in
#: together, and of those eight position 12 is by far the smallest space.  Opening another one in
#: :func:`build` shows it.
PARALLEL_POSITIONS = tuple(
    2 if index == 12 else pair for index, pair in enumerate(VGG11_BN_MERGED_POSITIONS)
)

#: A configuration small enough to free two whole positions and still build in a fraction of a
#: second.  The feature 1 is what matters: ``sum`` and ``product`` reduce to a single output feature
#: and ``copy`` fans out from a single input feature, so none of the three is inhabited unless 1 is
#: in the feature set.  The pinned chains have no such feature and never reach them.
SMALL_CONFIGURATION = {
    "linear_feature_dimensions": [1, 2, 4],
    "constant_values": [0, 1],
    "learning_rate_values": [1e-3],
    "n_epoch_values": [1],
    "channel_dimensions": [1, 2],
    "height_width_dimensions": [(2, 2), (1, 1)],
    "kernel_dimensions": [(2, 2)],
    "stride_values": [1],
    "padding_values": [0],
    "max_parallel_width": 2,
}

CELLS = ("vgg_chain", "parallel_chain", "small_free")

_BUILT: dict[str, dict] = {}


def build(cell):
    """Synthesize one of the three spaces and return its rule shape.

    Args:
        cell (str): One of :data:`CELLS`.

    Returns:
        dict: The shape, as :func:`shape_of` renders it.
    """
    if cell not in _BUILT:
        if cell == "small_free":
            repository = CNNrepository(**SMALL_CONFIGURATION)
            target = make_variance_target(4, (None, None), epochs=1, n_out=1)
        else:
            repository = cifar10_vgg11_bn_repo()
            positions = VGG11_BN_POSITIONS if cell == "vgg_chain" else PARALLEL_POSITIONS
            target = make_vgg11_bn_target(positions)
        space = Synthesizer(repository.specification(), {}).construct_solution_space(target).prune()
        _BUILT[cell] = shape_of(space)
    return _BUILT[cell]


def shape_of(space):
    """Return the synthesized program's rules in a form that ignores the predicates.

    A hole is written as the non-terminal it comes from and a constant as its ``repr``, so the two
    sides of the comparison do not have to be the same objects.  The rules of one non-terminal are
    sorted, because their order is not part of the program.

    Args:
        space (SolutionSpace): The pruned program.

    Returns:
        dict: Per non-terminal name, its rules as terminal, argument descriptors and predicate
        count.
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
            rules.append((str(rule.terminal), arguments, len(rule.predicates)))
        shape[str(nonterminal)] = sorted(rules)
    return shape


def spine_of(clause):
    """Return one combinator's clause as a tuple of entries, the suffix type last.

    :func:`tests.programs.clause_spine` walks the clause and yields a tuple per parameter, per
    predicate and one for the suffix.  Writing each of them as a single string is what lets the
    record hold a spine on one line and lets a failure point into the entry that moved.

    Args:
        clause (Specification): One combinator's specification, as the builder leaves it.

    Returns:
        tuple: One string per entry, closing with the rendered suffix type.
    """
    entries = []
    for entry in clause_spine(clause):
        kind = entry[0]
        if kind == "parameter":
            entries.append(f"parameter={entry[1]}")
        elif kind == "constraint on the literals":
            entries.append("literals")
        elif kind == "law":
            # A law stated as a relation on states carries the abstraction it was determinized
            # over, and two laws that differ only there are two different laws.
            entries.append("law=on-terms" if entry[2] is None else f"law=on-states:{entry[2]}")
        elif kind == "suffix":
            entries.append(entry[1])
        else:
            raise ValueError(f"a clause spine entry this record cannot hold: {entry!r}")
    return tuple(entries)


@cache
def _spines():
    """Read every clause of the repository, before any synthesis touches it.

    A spine holds parameter names, predicate kinds and the suffix type, and none of those carries a
    value from the configuration: both configurations of this module give the same spine for every
    combinator.  The reference repository therefore stands for both.

    Returns:
        dict: Per combinator name, its spine as :func:`spine_of` renders it.
    """
    return {name: spine_of(clause)
            for name, clause in cifar10_vgg11_bn_repo().specification().items()}


# --------------------------------------------------------------------------- the witness corpus
#
# Each of the four laws forbids the left-hand side of one rewrite rule, and each states that rule
# in its own docstring.  Read over the layers of a sequential composition rather than over the
# tree, three of the four are patterns of two consecutive layers and one is a pattern of three:
#
#   swaplaw2  [swap(a, b), edges(c)]  then  [edges(b), swap(a, c)]
#   swaplaw4  [edges(a), swap(b, c)]  then  [swap(a, c), edges(b)]
#   swaplaw3  [swap(a, b)]            then  [swap(b, a)]
#   swaplaw1  [swap(a, b)], then a layer whose first entry carries the edge numbers (b, c) and
#             whose remaining entries carry (a, d) between them, then [swap(c, d)]
#
# ``edges(x)`` is the degenerate crossing that carries x edges straight through, which is how the
# ``copy(x, edge())`` of the rewrite rules stands in a term.  In the triple, the clause that carries
# the weight is "between them": the entries after the first have to sum to (a, d), and there may be
# any number of them.
#
# The witnesses are built by hand rather than harvested from a space, because a pattern the laws
# reject cannot appear among the derived terms: a term is only derived once every predicate on it
# has said True.
#
# Each law gets a witness in both nestings the rest of the composition can take, because that is
# where the tracked triple pattern misses cases, and one more with the edge numbers changed so that
# the pattern does not close.  The last of the three has to be admitted, and it is what catches a
# law that rejects too much rather than too little.


def _node(root, arity, **slots):
    """Build a tree with `arity` children, filling the named slots and leaving the rest at 0."""
    children = [Tree(0, ()) for _ in range(arity)]
    for slot, value in slots.items():
        children[int(slot[1:])] = value
    return Tree(root, tuple(children))


def _swap(m, n):
    """The wiring that crosses `m` edges past `n` edges."""
    return _node("swap", 19, _1=Tree(m, ()), _2=Tree(n, ()))


def _edges(n):
    """`n` edges carried straight through, which is the degenerate crossing."""
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
    where the triple pattern reads the edge numbers of the entries after the first.
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
        list: One ``(law, key, purpose, head, tail)`` per witness.
    """
    trailer = Tree(0, ())
    m, n, p, q = 3, 4, 5, 6
    a, b, c = 3, 4, 5

    # The triple pattern: a layer that crosses m past n, then a layer whose first entry carries
    # (n, p) and whose remaining entries carry (m, q) between them, then a layer that crosses p
    # past q.  The three layers run two subterms past each other and cross back, which is the same
    # diagram as the two subterms side by side in the other order, so the space must not hold both.
    first_layer = _beside_singleton(inner=_swap(m, n))
    third_layer = _beside_singleton(inner=_swap(p, q))
    third_layer_open = _beside_singleton(inner=_swap(p, q + 1))
    two_entries = _beside_cons(i=n + m, i1=n, o=p + q, o1=p,
                               first=Tree("node", ()), rest=_beside_singleton(i=m, o=q))
    # The same layer with three entries.  What the pattern asks is that the entries after the
    # first sum to (m, q), not that there be exactly one of them.
    tail_of_three = _beside_cons(i=m, i1=1, o=q, o1=2,
                                 first=Tree("node", ()), rest=_beside_singleton(i=m - 1, o=q - 2))
    three_entries = _beside_cons(i=n + m, i1=n, o=p + q, o1=p,
                                 first=Tree("node", ()), rest=tail_of_three)

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
        ("swaplaw1", "triple_two_entries_nothing_after",
         "the triple, with two entries in the middle layer and nothing after the third",
         first_layer, _before_cons(two_entries, _before_singleton(third_layer))),
        ("swaplaw1", "triple_two_entries_more_after",
         "the same triple with the composition continuing past the third layer",
         first_layer, _before_cons(two_entries, _before_cons(third_layer, trailer))),
        ("swaplaw1", "triple_three_entries_nothing_after",
         "the triple with three entries in the middle layer, which the pattern covers too",
         first_layer, _before_cons(three_entries, _before_singleton(third_layer))),
        ("swaplaw1", "triple_three_entries_more_after",
         "three entries and a continuing composition, the two departures at once",
         first_layer, _before_cons(three_entries, _before_cons(third_layer, trailer))),
        ("swaplaw1", "triple_that_does_not_close",
         "the triple with the closing crossing on other edge numbers, which is no pattern",
         first_layer, _before_cons(two_entries, _before_singleton(third_layer_open))),
        ("swaplaw2", "first_pair_nothing_after",
         "the pair, with the second layer the last one",
         first_pair_left, _before_singleton(first_pair_right)),
        ("swaplaw2", "first_pair_more_after",
         "the pair, with the composition continuing past the second layer",
         first_pair_left, _before_cons(first_pair_right, trailer)),
        ("swaplaw2", "first_pair_that_does_not_close",
         "the pair on edge numbers that do not close it, which is no pattern",
         first_pair_left, _before_singleton(first_pair_open)),
        ("swaplaw3", "involution_nothing_after",
         "the pair, with the second layer the last one",
         involution_left, _before_singleton(involution_right)),
        ("swaplaw3", "involution_more_after",
         "the pair, with the composition continuing past the second layer",
         involution_left, _before_cons(involution_right, trailer)),
        ("swaplaw3", "involution_that_does_not_close",
         "the pair on edge numbers that do not close it, which is no pattern",
         involution_left, _before_singleton(involution_open)),
        ("swaplaw4", "second_pair_nothing_after",
         "the pair, with the second layer the last one",
         second_pair_left, _before_singleton(second_pair_right)),
        ("swaplaw4", "second_pair_more_after",
         "the pair, with the composition continuing past the second layer",
         second_pair_left, _before_cons(second_pair_right, trailer)),
        ("swaplaw4", "second_pair_that_does_not_close",
         "the pair on edge numbers that do not close it, which is no pattern",
         second_pair_left, _before_singleton(second_pair_open)),
    ]


WITNESSES = _witnesses()


# --------------------------------------------------------------------------- the frozen record

HEADER = """\
# The CNN repository as it was when it entered this repository.
#
# Written by tests/test_cnn_damg_repo_is_unchanged.py, which is also the only reader.  It holds
# three synthesized spaces rule by rule, the clause of every combinator, and the verdicts of the
# four swap laws on the witness corpus of that module.  The three spaces are the pinned VGG-11
# chain, the same chain with one merged position opened to two parallel components, and a small
# configuration with two free positions.  Together they reach every combinator of
# CNNrepository.specification().
#
# A rule is one line: the index of its non-terminal, its terminal, and one field per argument,
# `name=h<index>` for a hole into the non-terminal of that index and `name=c<index>` for a
# constant of that index, `-` for an argument that carries no name, and `p<count>` for the number
# of predicates the rule carries.  The non-terminals and the constants are listed once each, as
# JSON strings, so that a rule stays one readable line.
#
# A clause is one line too: the combinator, then one field per parameter and per predicate in the
# order the clause introduces them, closed by the suffix type as a JSON string.  It is read off
# the specification before any synthesis, which is what makes it see a conjunct that no target of
# the three spaces ever leaves open.
#
# When a commit changes this file legitimately, regenerate it with
#
#     PYTHONPATH=. python tests/test_cnn_damg_repo_is_unchanged.py
#
# and read the diff: it names every rule that appeared and every rule that went away.  A commit
# that meant to change nothing about the synthesized program leaves this file untouched.  The
# whole file goes away with the last commit of the migration, together with its module.
"""


def _render(shapes, spines, verdicts):
    """Render the frozen record.

    Args:
        shapes (dict): Per cell, the shape :func:`shape_of` returns.
        spines (dict): Per combinator, the spine :func:`spine_of` returns.
        verdicts (list): ``(law, key, verdict)`` per witness.

    Returns:
        str: The file contents.
    """
    lines = [HEADER.rstrip("\n")]
    for cell in CELLS:
        shape = shapes[cell]
        names = sorted(shape)
        name_index = {name: index for index, name in enumerate(names)}
        constants = sorted({
            argument[1]
            for rules in shape.values() for _, arguments, _ in rules
            for argument in arguments if argument[0] == "const"
        })
        constant_index = {constant: index for index, constant in enumerate(constants)}
        terminals = sorted({terminal for rules in shape.values() for terminal, _, _ in rules})
        lines.append("")
        lines.append(f"[cell {cell}]")
        lines.append(f"nonterminals {len(names)}")
        lines.append(f"rules {sum(len(rules) for rules in shape.values())}")
        lines.append("terminals " + " ".join(terminals))
        lines.append(f"[nonterminals {cell}]")
        lines.extend(f"{index} {json.dumps(name)}" for index, name in enumerate(names))
        lines.append(f"[constants {cell}]")
        lines.extend(f"{index} {json.dumps(constant)}" for index, constant in enumerate(constants))
        lines.append(f"[rules {cell}]")
        for name in names:
            for terminal, arguments, predicates in shape[name]:
                fields = [f"p{predicates}"]
                for kind, value, argument_name in arguments:
                    written = "-" if argument_name is None else str(argument_name)
                    # One rule is one line, so a name that carries a space or an '=' would make the
                    # line unreadable.  No combinator or parameter of the repository has one, and if
                    # one ever does, this stops rather than writing a record that cannot be read.
                    if any(character.isspace() for character in terminal + written) or "=" in written:
                        raise ValueError(
                            f"the record cannot hold {terminal!r} with an argument named "
                            f"{argument_name!r}: a rule has to fit on one line")
                    slot = ("h" if kind == "hole" else "c") + str(
                        name_index[value] if kind == "hole" else constant_index[value])
                    fields.append(f"{written}={slot}")
                lines.append(f"{name_index[name]} {terminal} " + " ".join(fields))
    lines.append("")
    lines.append(f"[spines {len(spines)}]")
    for name in sorted(spines):
        entries = spines[name]
        for entry in (name,) + entries[:-1]:
            # The suffix is the last entry and goes in as a JSON string, so it may hold anything.
            # The fields before it may not, since the line is read back by splitting on spaces and
            # on the first quote.  No combinator or parameter of the repository breaks that, and if
            # one ever does, this stops rather than writing a record that cannot be read.
            if any(character.isspace() for character in entry) or '"' in entry:
                raise ValueError(
                    f"the record cannot hold the clause of {name!r} with an entry {entry!r}: "
                    "a clause has to fit on one line")
        prefix = " ".join((name,) + entries[:-1])
        lines.append(f"{prefix} {json.dumps(entries[-1])}")
    lines.append("")
    lines.append("[witnesses]")
    lines.extend(f"{law} {key} {verdict}" for law, key, verdict in verdicts)
    lines.append("")
    return "\n".join(lines)


def _parse(text):
    """Read the frozen record back, and check it against its own header.

    The per-cell header states the number of non-terminals, the number of rules and the terminals
    reached, and the spine header states how many clauses follow.  Checking the body against them
    turns a truncated or hand-edited record into an error that names the file, rather than into a
    comparison that reports the damage as drift.

    Args:
        text (str): The file contents.

    Returns:
        tuple: The shapes per cell, the spines per combinator, and the verdicts as
        ``{(law, key): bool}``.
    """
    shapes = {cell: {} for cell in CELLS}
    declared = {}
    spines = {}
    declared_spines = None
    verdicts = {}
    names, constants, cell = [], [], None
    section = None
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            head = line[1:-1].split()
            section = head[0]
            if section in ("nonterminals", "constants", "rules", "cell"):
                cell = head[1]
            if section == "spines":
                declared_spines = int(head[1])
            if section == "nonterminals":
                names = []
            if section == "constants":
                constants = []
            continue
        if section == "cell":
            key, _, value = line.partition(" ")
            declared.setdefault(cell, {})[key] = value
        elif section == "nonterminals":
            names.append(json.loads(line.split(" ", 1)[1]))
        elif section == "constants":
            constants.append(json.loads(line.split(" ", 1)[1]))
        elif section == "rules":
            index, terminal, fields = line.split(" ", 2)
            fields = fields.split(" ")
            predicates = int(fields[0][1:])
            arguments = []
            for field in fields[1:]:
                argument_name, slot = field.rsplit("=", 1)
                arguments.append(
                    ("hole", names[int(slot[1:])], None if argument_name == "-" else argument_name)
                    if slot[0] == "h" else
                    ("const", constants[int(slot[1:])],
                     None if argument_name == "-" else argument_name))
            shapes[cell].setdefault(names[int(index)], []).append(
                (terminal, tuple(arguments), predicates))
        elif section == "spines":
            name, rest = line.split(" ", 1)
            fields, _, suffix = rest.partition('"')
            spines[name] = tuple(fields.split()) + (json.loads('"' + suffix),)
        elif section == "witnesses":
            law, key, verdict = line.split(" ")
            verdicts[(law, key)] = verdict == "True"
    for cell, cell_shape in shapes.items():
        for name in cell_shape:
            cell_shape[name] = sorted(cell_shape[name])
        stated = declared.get(cell)
        if stated is None:
            raise ValueError(f"{FROZEN} holds no record for {cell}")
        found = {
            "nonterminals": str(len(cell_shape)),
            "rules": str(sum(len(rules) for rules in cell_shape.values())),
            "terminals": " ".join(sorted({terminal for rules in cell_shape.values()
                                          for terminal, _, _ in rules})),
        }
        if stated != found:
            raise ValueError(
                f"{FROZEN} does not agree with its own header for {cell}: it states "
                f"{stated} and holds {found}")
    if declared_spines != len(spines):
        raise ValueError(
            f"{FROZEN} does not agree with its own header on the clauses: it states "
            f"{declared_spines} and holds {len(spines)}")
    return shapes, spines, verdicts


def _short(text, limit=90):
    """Cut a rendered name or value down to something a failure message can carry."""
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _fields(rule):
    """Take one rule apart into named fields, so that two rules can be compared field by field."""
    terminal, arguments, predicates = rule
    fields = {"terminal": terminal, "predicates": str(predicates)}
    for position, (kind, value, name) in enumerate(arguments):
        fields[f"argument {position} ({name})"] = f"{kind} {value}"
    return fields


def _as_text(rule):
    """Write one rule as a single line, with its long constants cut short."""
    terminal, arguments, predicates = rule
    parts = [f"{name}={kind}:{_short(value, 40)}" for kind, value, name in arguments]
    return f"{terminal}({', '.join(parts)}) with {predicates} predicates"


def _rule_difference(before, after):
    """List how two rules differ, naming only the fields that do.

    Returns:
        list: One line per differing field, or one line saying how many there are when the two
        rules have too little left in common for a field list to help.
    """
    left, right = _fields(before), _fields(after)
    keys = [key for key in left if left.get(key) != right.get(key)]
    keys += [key for key in right if key not in left]
    if not keys:
        return ["no difference"]
    if len(keys) > 3:
        return [f"{len(keys)} of its fields differ, starting with {keys[0]}"]
    return [f"{key}: {_short(left.get(key, 'absent'), 45)} -> "
            f"{_short(right.get(key, 'absent'), 45)}" for key in keys]


def _drift(frozen, current, cell):
    """Describe how the current space differs from the frozen one, or return the empty string."""
    lost = sorted(set(frozen) - set(current))
    gained = sorted(set(current) - set(frozen))
    changed = [name for name in sorted(set(frozen) & set(current))
               if frozen[name] != current[name]]
    if not (lost or gained or changed):
        return ""
    lines = [
        f"the {cell} space is not the one that was frozen",
        f"frozen: {len(frozen)} non-terminals with "
        f"{sum(len(rules) for rules in frozen.values())} rules",
        f"now:    {len(current)} non-terminals with "
        f"{sum(len(rules) for rules in current.values())} rules",
        f"{len(lost)} non-terminals gone, {len(gained)} new, "
        f"{len(changed)} kept their name and changed their rules",
    ]
    for name in lost[:3]:
        lines.append(f"  gone: {_short(name)}")
    for name in gained[:3]:
        lines.append(f"  new:  {_short(name)}")
    for name in changed[:3]:
        lines.append(f"  changed: {_short(name)}")
        before = [rule for rule in frozen[name] if rule not in current[name]]
        after = [rule for rule in current[name] if rule not in frozen[name]]
        if len(before) == len(after):
            # The rules line up, so name what moved inside them rather than reprinting them whole.
            for old_rule, new_rule in list(zip(before, after))[:3]:
                lines.extend(f"    {line}" for line in _rule_difference(old_rule, new_rule))
        else:
            for rule in before[:3]:
                lines.append(f"    lost: {_as_text(rule)}")
            for rule in after[:3]:
                lines.append(f"    new:  {_as_text(rule)}")
    return "\n".join(lines)


def _entry_difference(before, after, limit=45):
    """Say how two entries of a clause differ, and where.

    The suffix type of a clause runs to several hundred characters and two of them differ in a few,
    so a long entry is shown from twenty characters before the first one that moved rather than
    whole.

    Returns:
        str: The two entries, or the neighborhood of their first difference.
    """
    if len(before) <= limit and len(after) <= limit:
        return f"{before} -> {after}"
    common = 0
    while common < min(len(before), len(after)) and before[common] == after[common]:
        common += 1
    start = max(0, common - 20)
    return (f"from character {start}: {_short(before[start:], limit)} -> "
            f"{_short(after[start:], limit)}")


def _spine_drift(frozen, current, combinator):
    """Describe how the current clause differs from the frozen one, or return the empty string.

    It names the entries that moved and nothing else, so a clause whose suffix runs to several
    hundred characters does not arrive whole in the failure message.
    """
    lines = []
    for position in range(max(len(frozen), len(current))):
        before = frozen[position] if position < len(frozen) else None
        after = current[position] if position < len(current) else None
        if before == after:
            continue
        if before is None:
            lines.append(f"  entry {position} is new:  {_short(after, 45)}")
        elif after is None:
            lines.append(f"  entry {position} is gone: {_short(before, 45)}")
        else:
            lines.append(f"  entry {position}: {_entry_difference(before, after)}")
    if not lines:
        return ""
    if len(lines) > 3:
        lines = lines[:3] + [f"  and {len(lines) - 3} more entries differ"]
    return "\n".join([f"the clause of {combinator} is not the one that was frozen"] + lines)


@cache
def _frozen():
    """Read the frozen record once, or fail with the command that writes it."""
    if not FROZEN.exists():
        raise AssertionError(
            f"{FROZEN} is missing. Write it with "
            "'PYTHONPATH=. python tests/test_cnn_damg_repo_is_unchanged.py'")
    return _parse(FROZEN.read_text())


# --------------------------------------------------------------------------- the tests

@pytest.mark.parametrize("cell", CELLS)
def test_the_synthesized_rules_are_the_ones_that_were_frozen(cell):
    """Every non-terminal, every rule, every argument, in all three spaces.

    A changed parameter set, parameter constraint or argument type lands here.  A changed predicate
    body does not, which is what the witness corpus is for.

    Args:
        cell (str): The space to check.
    """
    frozen, _, _ = _frozen()
    drift = _drift(frozen[cell], build(cell), cell)
    assert not drift, drift


@pytest.mark.parametrize("combinator", sorted(_spines()))
def test_the_clause_of_every_combinator_is_the_one_that_was_frozen(combinator):
    """One clause, parameter by parameter and predicate by predicate, suffix type included.

    The three spaces build from pinned targets, so a conjunct that only an open target would need
    is in no rule of theirs, and changing it moves none of the three.  This axis reads the clause
    instead, before any synthesis, so it sees that conjunct.  A law that was dropped, added or
    attached to another combinator lands here too.

    A combinator the specification lost does not reach this test, since it is parametrized over
    what the specification declares.  The rules comparison catches it, because every declared
    combinator is reached by one of the three spaces, which is what the reach test keeps true.

    Args:
        combinator (str): The name of the combinator in ``CNNrepository.specification()``.
    """
    _, frozen, _ = _frozen()
    assert combinator in frozen, (
        f"{FROZEN} holds no clause for {combinator}, which the specification declares")
    drift = _spine_drift(frozen[combinator], _spines()[combinator], combinator)
    assert not drift, drift


def test_the_three_spaces_together_reach_every_combinator():
    """Every combinator of ``CNNrepository.specification()`` is inhabited in at least one space.

    This is why there are three spaces rather than one.  A combinator no space reaches is one whose
    rules the comparison cannot see, and the pinned VGG chain alone reaches neither the parallel
    side nor the scalar combinators.  If a later commit weakens one of the three spaces, this is the
    test that says which combinator went out of view.
    """
    declared = set(cifar10_vgg11_bn_repo().specification())
    reached = {terminal for cell in CELLS for rules in build(cell).values()
               for terminal, _, _ in rules}
    assert sorted(reached) == sorted(declared)


@pytest.mark.parametrize("law, key, purpose, head, tail", WITNESSES,
                         ids=[f"{law}-{key}" for law, key, _, _, _ in WITNESSES])
def test_the_swap_laws_decide_their_witnesses_the_way_they_did(law, key, purpose, head, tail):
    """One law on one witness, against the verdict frozen at migration.

    A frozen True on a witness that realizes one of the four patterns is not an endorsement.  It
    records that the law admits that case today, and the commit that repairs the law is the one
    that turns it into a False here.

    Args:
        law (str): The name of the law.
        key (str): The witness identifier in the frozen record.
        purpose (str): What the witness is.
        head (Tree): The layer the composition puts first.
        tail (Tree): The composition it is put in front of.
    """
    _, _, verdicts = _frozen()
    was = verdicts[(law, key)]
    now = getattr(CNNrepository, law)(head, tail)
    assert now == was, (
        f"{law} changed its mind about {purpose}: it said {was} and now says {now}")


if __name__ == "__main__":
    FROZEN.parent.mkdir(exist_ok=True)
    FROZEN.write_text(_render(
        {cell: build(cell) for cell in CELLS},
        _spines(),
        [(law, key, getattr(CNNrepository, law)(head, tail))
         for law, key, _, head, tail in WITNESSES]))
    print(f"wrote {FROZEN} ({FROZEN.stat().st_size} bytes)")
