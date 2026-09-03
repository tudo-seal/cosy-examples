"""The reference search spaces the tests share, with their signatures.

These are the same five spaces as ``tests/search_fixtures.py`` in cosy, restated rather than
imported: cosy ships ``src/cosy`` as its package and its ``tests`` package is not installed, and
both repositories name their test package ``tests``, so importing across would have to shadow one
with the other.  The copy is pinned instead of trusted: :mod:`tests.test_spaces` checks each
space against the closed forms for its inhabitant counts, so a divergence between the two copies
shows up as a failing test rather than as a silently different experiment.

Each space carries a case the others cannot:

* :func:`list_space`, lists over ``{0, 1, 2}``, the space on which clause-uniform sampling and
  size-uniform random search are compared.  ``3^l`` lists have size ``l + 1``, so the size counts
  are known in closed form and every target distribution over this space can be written down
  without running anything.
* :func:`expression_space`, ``E -> lit | neg(E) | add(E, E)``.  A unary and a binary combinator,
  so inhabitants of one size differ in shape and a cost function other than size has something to
  distinguish.
* :func:`constrained_space`, an external predicate couples the two holes of ``pair(_, _)``.  The
  residual there is a relation, not the product of its projections, so a sampler over it has to
  draw the two holes jointly.
* :func:`ambiguous_space`, one inhabitant on two success branches.  Exact counting instead needs
  every inhabitant within the bound to end exactly one success branch.  The ambiguity comes from
  an intersection of *arrows*.  On a nullary combinator every admissible subset of paths collapses
  to one clause, so no nullary intersection can produce it.
* :func:`literal_space`, constant arguments, the one case where a clause writes more than one
  symbol, and the shape the CNN search space is built from almost entirely.

Beside them stand two **contrast pairs**, each a coupled space and a recognizable one that define
the *same language*.  They are what the determinization can be measured on: the coupled member
states its condition as a predicate over the grounded subtrees and can only be counted from the
search tree, the recognizable member states the same condition as an abstraction and a relation on
its values, and after :func:`cosy.search.determinize.determinize` the table form applies to it.

* :func:`coupled_cores_space` / :func:`recognizable_pair_space`, ``pair(w1, w2)`` with differing
  innermost letters.  Small enough for the brute-force oracle, so the drawn distribution can be
  compared by hand against the weight random search realizes, the target probability of an
  inhabitant's cost divided by the number of inhabitants that share that cost.
* :func:`avl_coupled_space` / :func:`avl_space`, the benchmark of Goldstein and Pierce, keys and
  cached heights included, with :func:`avl_trees` as an oracle that enumerates the valid trees
  directly.  The signature has a 4-ary symbol, so terms-over-the-signature enumeration is hopeless
  at the bound the language needs, and the direct enumeration is what replaces it.

Only ``recognizable_pair_space`` and the two AVL spaces are restatements of cosy fixtures.
``coupled_cores_space`` is new here: cosy pairs ``recognizable_pair_space`` with
``constrained_space``, which couples over *term equality* and therefore defines a different
language.  That pairing shows that recognizability, the requirement that a finite algebra can
evaluate the constraint, is a real restriction, but it is no use for measuring whether the
determinization preserves a distribution, which needs the two members to agree.

The signatures are the alphabet the brute-force oracle enumerates over.  The literal space's
includes the literal values themselves, since they are leaves of the term.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cosy.core import Constructor, SpecificationBuilder, Synthesizer
from cosy.core.tree import Tree
from cosy.core.types import Arrow, DataGroup, Intersection

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from cosy.core.solution_space import SolutionSpace

__all__ = [
    "AMBIGUOUS_SIGNATURE",
    "AMBIGUOUS_TARGET",
    "AVL",
    "AVL_KEYS",
    "CHAIN",
    "CHAIN_SIGNATURE",
    "ELEM",
    "EXPR",
    "EXPR_SIGNATURE",
    "LEAF_STATE",
    "LIST",
    "LIST_SIGNATURE",
    "LITERAL_SIGNATURE",
    "PAIR",
    "PAIR_SIGNATURE",
    "TAGGED",
    "WORD",
    "X",
    "Y",
    "add",
    "alt",
    "ambiguous_space",
    "avl_coupled_space",
    "avl_relation",
    "avl_space",
    "avl_summary",
    "avl_trees",
    "avl_valid",
    "base",
    "chain_space",
    "chain_terms",
    "cons_0",
    "cons_1",
    "cons_2",
    "cons_c",
    "constrained_space",
    "core_of",
    "coupled_cores_space",
    "different",
    "different_cores",
    "different_cores_on_terms",
    "expression_space",
    "innermost",
    "leaf",
    "list_space",
    "lit",
    "literal_space",
    "merge",
    "neg",
    "nil",
    "nil_c",
    "node",
    "one",
    "pair",
    "recognizable_pair_space",
    "stop",
    "tag",
    "tree_summary",
    "wrap",
    "z",
    "zero",
]

# ---------------------------------------------------------------------------
# L, the lists over {0, 1, 2}:  List -> nil | cons_0(List) | cons_1(List) | cons_2(List)
# ---------------------------------------------------------------------------

LIST = Constructor("List")


def nil() -> str:
    """Build the empty list.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "[]"


def cons_0(rest: str) -> str:
    """Prepend the digit 0.

    Args:
        rest (str): The interpreted tail.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"0:{rest}"


def cons_1(rest: str) -> str:
    """Prepend the digit 1.

    Args:
        rest (str): The interpreted tail.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"1:{rest}"


def cons_2(rest: str) -> str:
    """Prepend the digit 2.

    Args:
        rest (str): The interpreted tail.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"2:{rest}"


def list_space() -> SolutionSpace:
    """Build the space of lists over ``{0, 1, 2}``.

    A list of length ``l`` is a term of size ``l + 1`` and ``3^l`` lists share that size, so the
    size-uniform weight of a list of length ``l`` under a bound ``D`` is ``1 / (D * 3^l)`` and
    every length carries total weight ``1/D``.  Random search therefore draws each length below
    the bound with probability ``1/D``.

    Returns:
        SolutionSpace: The space, started at ``List``.
    """
    specs = {
        nil: SpecificationBuilder().suffix(LIST),
        cons_0: SpecificationBuilder().argument("rest", LIST).suffix(LIST),
        cons_1: SpecificationBuilder().argument("rest", LIST).suffix(LIST),
        cons_2: SpecificationBuilder().argument("rest", LIST).suffix(LIST),
    }
    return Synthesizer(specs).construct_solution_space(LIST)


# ---------------------------------------------------------------------------
# C, the chain of single-element lists:  C -> nil_c | cons_c(Z, C),  Z -> z
# ---------------------------------------------------------------------------

CHAIN = Constructor("C")
ELEM = Constructor("Z")


def nil_c() -> str:
    """End the chain.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "[]"


def z() -> str:
    """Build the single element the chain is made of.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "z"


def cons_c(head: str, rest: str) -> str:
    """Prepend an element to the chain.

    Args:
        head (str): The interpreted element.
        rest (str): The interpreted tail.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"{head}:{rest}"


def chain_space() -> SolutionSpace:
    """Build the chain of single-element lists, which holds exactly one inhabitant per size.

    A term is therefore determined by its size, so the term axis of the surrogate and kernel plots
    is a genuine order, which is why the worked examples of Bayesian optimization run here.  The
    element sits in its own non-terminal, so the list of length ``l`` has ``l`` applications of
    ``cons_c``, ``l`` copies of ``z`` and one ``nil_c``.  Term size counts the occurrences of
    function symbols, so that list has size ``2l + 1``, and the bound ``D = 21`` of the
    kernel-diverse initialization cuts the chain at length ten.

    Under the unit-weight subtree kernel the space has a closed form: between the lists of lengths
    ``a <= b`` the ``z`` subterms match in ``a*b`` pairs, the ending ``nil_c`` in one, and the
    tails of lengths one to ``a`` in ``a`` more, so ``k(a, b) = a*b + a + 1``.

    Unlike the five spaces above, this one is not a restatement of a cosy fixture.  It carries the
    worked kernel examples, which the others cannot: they have several inhabitants per size, so no
    closed form for a kernel between two of them.

    Returns:
        SolutionSpace: The space, started at ``C``.
    """
    specs = {
        nil_c: SpecificationBuilder().suffix(CHAIN),
        z: SpecificationBuilder().suffix(ELEM),
        cons_c: SpecificationBuilder()
        .argument("head", ELEM)
        .argument("rest", CHAIN)
        .suffix(CHAIN),
    }
    return Synthesizer(specs).construct_solution_space(CHAIN)


def chain_terms(bound: int) -> list[Tree[Any]]:
    """Enumerate the chain's inhabitants of size at most ``bound``, the oracle for this space.

    The suite's usual oracle enumerates every term over the signature and filters with the checker,
    which does not survive this bound: ``cons_c`` is binary, so the terms over the signature up to
    size 21 number in the millions while the inhabitants number eleven.  A chain is determined by
    its length, so this builds them directly, and no part of cosy is consulted.

    Args:
        bound (int): The size bound ``D``.  A list of length ``l`` has size ``2l + 1``.

    Returns:
        list[Tree[Any]]: The inhabitants, shortest first.
    """
    terms: list[Tree[Any]] = []
    term: Tree[Any] = Tree(nil_c, ())
    while 2 * len(terms) + 1 <= bound:
        terms.append(term)
        term = Tree(cons_c, (Tree(z, ()), term))
    return terms


# ---------------------------------------------------------------------------
# E, the expressions:  E -> lit | neg(E) | add(E, E)
# ---------------------------------------------------------------------------

EXPR = Constructor("E")


def lit() -> str:
    """Build the literal.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "x"


def neg(inner: str) -> str:
    """Negate an expression.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"-{inner}"


def add(left: str, right: str) -> str:
    """Add two expressions.

    Args:
        left (str): The interpreted left operand.
        right (str): The interpreted right operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"({left}+{right})"


def expression_space() -> SolutionSpace:
    """Build the expression space.

    Returns:
        SolutionSpace: The space, started at ``E``.
    """
    specs = {
        lit: SpecificationBuilder().suffix(EXPR),
        neg: SpecificationBuilder().argument("inner", EXPR).suffix(EXPR),
        add: SpecificationBuilder()
        .argument("left", EXPR)
        .argument("right", EXPR)
        .suffix(EXPR),
    }
    return Synthesizer(specs).construct_solution_space(EXPR)


# ---------------------------------------------------------------------------
# P, coupled holes:  W -> zero | one | wrap(W) ;  Pair -> pair(W, W) with left != right
# ---------------------------------------------------------------------------

WORD = Constructor("W")
PAIR = Constructor("Pair")


def zero() -> str:
    """Build the first word.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "0"


def one() -> str:
    """Build the second word.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "1"


def wrap(inner: str) -> str:
    """Wrap a word, so that the sort is recursive and the bound bites.

    Args:
        inner (str): The interpreted operand.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"w({inner})"


def pair(left: str, right: str) -> str:
    """Pair two words.

    Args:
        left (str): The interpreted left word.
        right (str): The interpreted right word.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"<{left},{right}>"


def different(substitution: Mapping[str, Any]) -> bool:
    """Decide whether the two paired words differ.

    Args:
        substitution (Mapping[str, Any]): The grounded arguments of the clause, by variable name.

    Returns:
        bool: True when the two words are different terms.
    """
    return bool(substitution["left"] != substitution["right"])


def constrained_space() -> SolutionSpace:
    """Build the space whose external predicate couples two holes.

    Returns:
        SolutionSpace: The space, started at ``Pair``.
    """
    specs = {
        zero: SpecificationBuilder().suffix(WORD),
        one: SpecificationBuilder().suffix(WORD),
        wrap: SpecificationBuilder().argument("inner", WORD).suffix(WORD),
        pair: SpecificationBuilder()
        .argument("left", WORD)
        .argument("right", WORD)
        .constraint(different)
        .suffix(PAIR),
    }
    return Synthesizer(specs).construct_solution_space(PAIR)


# ---------------------------------------------------------------------------
# A, ambiguous:  base : X & Y ;  alt : X ;  merge : (X -> M) & (Y -> M) ;  target M
# ---------------------------------------------------------------------------

X = Constructor("X")
Y = Constructor("Y")
AMBIGUOUS_TARGET = Constructor("M")


def base() -> str:
    """Build the term that inhabits both argument sorts.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "b"


def alt() -> str:
    """Build the term that inhabits the first argument sort alone.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "a"


def merge(inner: str) -> str:
    """Turn an ``X`` or a ``Y`` into an ``M``.

    Args:
        inner (str): The interpreted argument.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"m({inner})"


def ambiguous_space() -> SolutionSpace:
    """Build a space in which one inhabitant ends more than one success branch.

    ``merge`` has two paths of arity one onto ``M``, so the inhabitation emits two clauses that
    share their terminal and differ in the sort they ask of their argument.  ``base`` inhabits
    both sorts, so ``merge(base)`` is derived twice and ``merge(alt)`` once.  The space carries
    both cases, which is what makes it a test rather than a demonstration.

    Returns:
        SolutionSpace: The space, started at ``M``.
    """
    specs = {
        base: SpecificationBuilder().suffix(Intersection(X, Y)),
        alt: SpecificationBuilder().suffix(X),
        merge: SpecificationBuilder().suffix(
            Intersection(Arrow(X, AMBIGUOUS_TARGET), Arrow(Y, AMBIGUOUS_TARGET))
        ),
    }
    return Synthesizer(specs).construct_solution_space(AMBIGUOUS_TARGET)


# ---------------------------------------------------------------------------
# D, literals:  Digit -> 0 | 1 ;  Tagged -> stop | tag(d: Digit, Tagged)
# ---------------------------------------------------------------------------

TAGGED = Constructor("Tagged")
DIGITS = DataGroup("digit", (0, 1))


def stop() -> str:
    """End a tagged chain.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "."


def tag(d: int, rest: str) -> str:
    """Prepend a literal digit to a tagged chain.

    Args:
        d (int): The literal argument.
        rest (str): The interpreted rest of the chain.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"{d}{rest}"


def literal_space() -> SolutionSpace:
    """Build the space whose clauses write two symbols each.

    ``tag`` fixes its terminal *and* its constant argument, so it grows a term by two symbols.
    Sizes are therefore odd throughout, which makes this the space on which size-uniform sampling,
    drawing evenly from the sizes the space actually realizes, meets a realized set with gaps in
    it.

    Returns:
        SolutionSpace: The space, started at ``Tagged``.
    """
    specs = {
        stop: SpecificationBuilder().suffix(TAGGED),
        tag: SpecificationBuilder()
        .parameter("d", DIGITS)
        .argument("rest", TAGGED)
        .suffix(TAGGED),
    }
    return Synthesizer(specs).construct_solution_space(TAGGED)


# ---------------------------------------------------------------------------
# R1, the contrast pair over words:  pair(w1, w2) with differing innermost letters
# ---------------------------------------------------------------------------


def core_of(term: Tree[Any]) -> int:
    """Return the innermost letter of a word, read off the term.

    Written without reference to :func:`innermost` on purpose: this is what the coupled space's
    predicate reads, and if the two agreed by construction the agreement of the two spaces would
    prove nothing about the abstraction.

    Args:
        term (Tree): A grounded word over ``zero``, ``one`` and ``wrap``.

    Returns:
        int: 0 or 1, the letter at the core.
    """
    while term.children:
        term = term.children[0]
    return 0 if term.root is zero else 1


def different_cores_on_terms(substitution: Mapping[str, Any]) -> bool:
    """Decide whether two paired words have different innermost letters, reading the terms.

    Args:
        substitution (dict): The grounded arguments of the clause, by variable name.

    Returns:
        bool: True when the two cores differ.
    """
    return core_of(substitution["left"]) != core_of(substitution["right"])


def innermost(symbol: object, states: tuple) -> int:
    """Abstract a word by the letter at its core, the abstraction of the recognizable form.

    Args:
        symbol (object): The function symbol.
        states (tuple): The abstractions of the arguments, empty on a letter.

    Returns:
        int: 0 or 1, the letter ``wrap`` was applied to.
    """
    return states[0] if states else (0 if symbol is zero else 1)


def different_cores(substitution: Mapping[str, Any]) -> bool:
    """Decide whether two abstracted cores differ, the relation of the recognizable form.

    Args:
        substitution (dict): The clause's substitution, with both holes carrying their abstraction.

    Returns:
        bool: True when the two letters differ.
    """
    return bool(substitution["left"] != substitution["right"])


def coupled_cores_space() -> SolutionSpace:
    """Build the space whose predicate reads both holes to compare their cores.

    A word of size ``s`` is ``wrap`` applied ``s - 1`` times to a letter, so there are two of each
    size, one per core.  A pair of size ``n`` splits its ``n - 1`` remaining symbols over the two
    words and admits two of the four core combinations, so ``N(n) = 2 (n - 2)``.

    Returns:
        SolutionSpace: The space, started at ``Pair``.
    """
    specs = {
        zero: SpecificationBuilder().suffix(WORD),
        one: SpecificationBuilder().suffix(WORD),
        wrap: SpecificationBuilder().argument("inner", WORD).suffix(WORD),
        pair: SpecificationBuilder()
        .argument("left", WORD)
        .argument("right", WORD)
        .constraint(different_cores_on_terms)
        .suffix(PAIR),
    }
    return Synthesizer(specs).construct_solution_space(PAIR)


def recognizable_pair_space() -> SolutionSpace:
    """Build the same language with the condition stated as an abstraction and a relation.

    Returns:
        SolutionSpace: The space, started at ``Pair``.
    """
    specs = {
        zero: SpecificationBuilder().suffix(WORD),
        one: SpecificationBuilder().suffix(WORD),
        wrap: SpecificationBuilder().argument("inner", WORD).suffix(WORD),
        pair: SpecificationBuilder()
        .argument("left", WORD)
        .argument("right", WORD)
        .recognizable_constraint(innermost, different_cores)
        .suffix(PAIR),
    }
    return Synthesizer(specs).construct_solution_space(PAIR)


# ---------------------------------------------------------------------------
# R2, the AVL trees of Goldstein and Pierce:  node(key, cached height, left, right) | leaf
# ---------------------------------------------------------------------------

AVL = Constructor("AVL")
AVL_KEYS = tuple(range(10))
"""goldstein2022, Table 1: "AVL trees with values 0-9"."""

LEAF_STATE = (0, None, None)
"""``alpha`` of the empty tree: height zero, no smallest key, no greatest key."""


def leaf() -> str:
    """Build the empty tree.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return "E"


def node(x: int, h: int, left: str, right: str) -> str:
    """Build an inner node carrying its key and its cached height.

    Args:
        x (int): The key.
        h (int): The cached height.
        left (str): The interpreted left subtree.
        right (str): The interpreted right subtree.

    Returns:
        str: Its rendering under ``interpret``.
    """
    return f"({left} {x}:{h} {right})"


def tree_summary(term: Tree[Any]) -> tuple:
    """Compute height and extreme keys of a grounded AVL term, without the abstraction.

    Args:
        term (Tree): A grounded term over ``leaf`` and ``node``.

    Returns:
        tuple: ``(height, smallest key, greatest key)``.
    """
    if not term.children:
        return LEAF_STATE
    key = term.children[0].root
    left = tree_summary(term.children[2])
    right = tree_summary(term.children[3])
    return (
        1 + max(left[0], right[0]),
        left[1] if left[1] is not None else key,
        right[2] if right[2] is not None else key,
    )


def _avl_condition(key: int, cached: int, left: tuple, right: tuple) -> bool:
    """Decide the validity condition of Goldstein and Pierce from the summaries and the literals.

    Args:
        key (int): The node's key.
        cached (int): The height the node caches.
        left (tuple): The left subtree's summary.
        right (tuple): The right subtree's summary.

    Returns:
        bool: True when the node is balanced, caches its height correctly and orders its keys.
    """
    if abs(left[0] - right[0]) > 1:  # (2) balance
        return False
    if cached != 1 + max(left[0], right[0]):  # (3) the cached height is the true one
        return False
    if left[2] is not None and left[2] >= key:  # (1) ordering, to the left
        return False
    return not (right[1] is not None and right[1] <= key)  # (1) ordering, to the right


def avl_valid(substitution: Mapping[str, Any]) -> bool:
    """Decide the AVL condition on the grounded subtrees, the coupled statement of it.

    Args:
        substitution (dict): Two grounded subtrees and two literals.

    Returns:
        bool: True when the node is a valid AVL node.
    """
    return _avl_condition(
        substitution["x"],
        substitution["h"],
        tree_summary(substitution["l"]),
        tree_summary(substitution["r"]),
    )


def avl_summary(symbol: object, states: tuple) -> object:
    """Abstract a tree by its height and its extreme keys, the abstraction for the AVL condition.

    Args:
        symbol (object): The function symbol, or a literal value.
        states (tuple): The abstractions of the arguments.

    Returns:
        object: ``(height, smallest key, greatest key)`` on a tree, the value itself on a literal.
    """
    if symbol is leaf:
        return LEAF_STATE
    if symbol is node:
        key, _cached, left, right = states
        return (
            1 + max(left[0], right[0]),
            left[1] if left[1] is not None else key,
            right[2] if right[2] is not None else key,
        )
    return symbol  # a literal is its own abstraction


def avl_relation(substitution: Mapping[str, Any]) -> bool:
    """Decide the AVL condition on the abstracted subtrees, the relation of the recognizable form.

    Args:
        substitution (dict): The clause's substitution, with both subtrees carrying their state.

    Returns:
        bool: True when the node is a valid AVL node.
    """
    return _avl_condition(
        substitution["x"], substitution["h"], substitution["l"], substitution["r"]
    )


def _avl_specification(keys: tuple, constrain: Callable[[Any], Any]) -> dict:
    """Build the AVL specification, with the constraint stated either way.

    Args:
        keys (tuple): The admissible key values.
        constrain (Callable[[Any], Any]): Applied to the ``node`` builder to attach the
            constraint.

    Returns:
        dict: The specification.
    """
    return {
        leaf: SpecificationBuilder().suffix(AVL),
        node: constrain(
            SpecificationBuilder()
            .parameter("x", DataGroup("key", tuple(keys)))
            .parameter("h", DataGroup("height", tuple(range(1, len(keys) + 1))))
            .argument("l", AVL)
            .argument("r", AVL)
        ).suffix(AVL),
    }


def avl_coupled_space(keys: tuple = AVL_KEYS) -> SolutionSpace:
    """Build the AVL space with the validity condition as a plain predicate over the holes.

    A node costs three symbols, the terminal and its two literals, and a leaf costs one, so a tree
    of ``k`` nodes has size ``4k + 1`` and the whole language lies below ``D = 4 |keys| + 1``.

    Args:
        keys (tuple): The admissible key values. (Default value = ``AVL_KEYS``)

    Returns:
        SolutionSpace: The space, started at ``AVL``.
    """
    specs = _avl_specification(keys, lambda builder: builder.constraint(avl_valid))
    return Synthesizer(specs).construct_solution_space(AVL).prune()


def avl_space(keys: tuple = AVL_KEYS) -> SolutionSpace:
    """Build the same language in the form the determinization consumes.

    Args:
        keys (tuple): The admissible key values. (Default value = ``AVL_KEYS``)

    Returns:
        SolutionSpace: The space, started at ``AVL``.
    """
    specs = _avl_specification(
        keys, lambda builder: builder.recognizable_constraint(avl_summary, avl_relation)
    )
    return Synthesizer(specs).construct_solution_space(AVL).prune()


def avl_trees(keys: tuple) -> list[Tree[Any]]:
    """Enumerate every valid AVL tree over a key set, as a term, the oracle for this space.

    The suite's usual oracle enumerates every term over the signature and filters with the checker,
    which is hopeless here: a 4-ary symbol at ``D = 13`` already gives some ``1e8`` terms.  This
    enumerates the *valid* trees directly instead.  Pick a root, split the remaining keys into the
    keys left of it and the keys right of it, recurse, and keep the combination when the heights
    balance.  The cached height is computed rather than guessed, so every tree produced satisfies
    all three parts of the condition by construction, and no part of cosy is consulted.

    Args:
        keys (tuple): The admissible key values.

    Returns:
        list[Tree[Any]]: The valid AVL trees, each exactly once.
    """
    memo: dict[tuple, list[Tree[Any]]] = {}

    def height(term: Tree[Any]) -> int:
        return 0 if not term.children else term.children[1].root

    def build(available: tuple) -> list[Tree[Any]]:
        if available in memo:
            return memo[available]
        trees: list[Tree[Any]] = [Tree(leaf, ())]
        for index, key in enumerate(available):
            for left in build(available[:index]):
                for right in build(available[index + 1 :]):
                    if abs(height(left) - height(right)) > 1:
                        continue
                    cached = 1 + max(height(left), height(right))
                    trees.append(
                        Tree(node, (Tree(key, ()), Tree(cached, ()), left, right))
                    )
        memo[available] = trees
        return trees

    return build(tuple(sorted(keys)))


# ---------------------------------------------------------------------------
# Signatures for the brute-force oracle
# ---------------------------------------------------------------------------

CHAIN_SIGNATURE = {nil_c: 0, z: 0, cons_c: 2}
LIST_SIGNATURE = {nil: 0, cons_0: 1, cons_1: 1, cons_2: 1}
EXPR_SIGNATURE = {lit: 0, neg: 1, add: 2}
PAIR_SIGNATURE = {zero: 0, one: 0, wrap: 1, pair: 2}
AMBIGUOUS_SIGNATURE = {base: 0, alt: 0, merge: 1}
LITERAL_SIGNATURE = {stop: 0, tag: 2, 0: 0, 1: 0}
