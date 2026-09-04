"""Hand-built pairs of subterms that the four swap laws reject, and near misses they admit.

``cnn_damg_nas`` and ``damg_nas`` inherit the same four swap laws from
``bayesian_optimization/examples/swap_laws.py``, as term predicates over two sibling holes of
``before_cons``.  A pair a law rejects never appears in a
synthesized space, because a term is only derived once every predicate on it has said True, so a
comparison that reads pairs off a space sees only pairs every law admitted.  A relation that
returned True on everything would pass such a comparison.  The pairs below are the answer to that,
and they are the only reading of the four laws that does not depend on a space at all.

They live here because both comparison files need them and the laws they are built for are the same
laws.  A witness is a pair ``(head, tail)`` with the verdict the law named beside it has to return.

The rewrite rules the four laws forbid the left-hand sides of, read over the layers of a sequential
composition rather than over the tree, are three patterns of two consecutive layers and one of
three:

* ``swaplaw2``: ``[swap(a, b), edges(c)]`` then ``[edges(b), swap(a, c)]``
* ``swaplaw4``: ``[edges(a), swap(b, c)]`` then ``[swap(a, c), edges(b)]``
* ``swaplaw3``: ``[swap(a, b)]`` then ``[swap(b, a)]``
* ``swaplaw1``: ``[swap(a, b)]``, then a layer whose first entry carries the edge numbers
  ``(b, c)`` and whose remaining entries carry ``(a, d)`` between them, then ``[swap(c, d)]``

``edges(x)`` is the degenerate crossing that carries x edges straight through, which is how the
``copy(x, edge())`` of the rewrite rules stands in a term.

Each law gets the pair it rejects and, beside it, the same pair with the closing edge numbers
changed so that the pattern does not close.  The second one has to be admitted, and it is what
catches a relation that rejects more than its law does.
"""

from __future__ import annotations

from cosy.core.tree import Tree


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


#: The corpus, as ``(law, case, verdict, head, tail)``.
WITNESSES = _witnesses()
