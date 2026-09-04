"""The four swap laws, stated once for every repository that builds terms of the same theory.

``damg_nas/damg_repo.py`` and ``cnn_damg_nas/cnn_damg_repo.py`` model neural networks as string
diagrams over different alphabets, and they normalize their terms against the same rewrite rules.
Most left-hand sides a clever typing of the combinators already rules out.  Four of them it cannot,
because they relate two sibling subterms rather than one subterm to its type, so those four are
stated here as predicates over the two holes of ``before_cons``, and a term is derived only once
all four have said True.

The laws read a derivation tree by root name and child position, so what they need of an alphabet
is the arities of the six combinators they index: ``beside_singleton`` 5, ``before_singleton`` 6,
``before_cons`` 9, ``beside_cons`` 11, ``edges`` 17 and ``swap`` 19.  Both repositories give those
six exactly these arities, and nothing else about their alphabets reaches the laws, which is why
one statement serves both.  ``tests/test_swap_laws_are_shared.py`` holds that premise, and a
repository that breaks it gets ``ValueError`` out of :func:`_require_children_count` rather than a
quiet wrong verdict.

A repository inherits :class:`SwapLaws` and attaches ``self.swaplaw1`` to ``self.swaplaw4`` to its
``before_cons`` clause.  ``recognizable_swap_laws.py`` next door states the same four laws in the
form that can be compiled into the non-terminals.
"""

from __future__ import annotations

from cosy.core.tree import Tree


def _tree_root_is(tree, root_name: str) -> bool:
    # Exact root comparison for Tree nodes.
    return tree.root == root_name


def _tree_root_contains(tree, token: str) -> bool:
    # Preserve the existing substring-based root matching used by the predicates.
    return token in tree.root


def _require_children_count(tree, expected: int):
    # Keep the shape checks and their error message in one place.
    if len(tree.children) != expected:
        raise ValueError("Derivation trees have not the expected shape.")
    return tree.children


def _tree_child(tree, index: int):
    # Short helper for indexing into a derivation tree.
    return tree.children[index]


class SwapLaws:
    """The four swap laws as term predicates, for a repository to inherit.

    A subclass gets four static methods, each of which takes the two subterms ``before_cons``
    composes and returns False on the left-hand side its own docstring names.  The class carries no
    state and no ``__init__``, so a repository inherits it beside whatever else it is.
    """

    @staticmethod
    def swaplaw1(head: Tree[str], tail: Tree[str]) -> bool:
        """
        before(swap(m+n, m, n), before(beside(x(n,p), y(m,q)), swap(p+q, p, q)))
        ->
        beside(y(m,q),x(n,p))

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """

        if _tree_root_contains(head, "beside_singleton") and _tree_root_contains(tail, "before_cons"):
            _require_children_count(head, 5)
            _require_children_count(tail, 9)

            left_term = _tree_child(head, 4)
            right_head = _tree_child(tail, 7)
            right_tail = _tree_child(tail, 8)

            if _tree_root_is(left_term, "swap") and _tree_root_contains(right_head, "beside_cons") and _tree_root_contains(right_tail, "before_singleton"):
                _require_children_count(left_term, 19)
                _require_children_count(right_head, 11)
                _require_children_count(right_tail, 6)

                m = left_term.children[1]
                n = left_term.children[2]
                x_n = right_head.children[1]
                x_p = right_head.children[4]

                right_head_tail = right_head.children[10]
                right_tail_term = right_tail.children[5]

                if _tree_root_contains(right_head_tail, "beside_singleton") and _tree_root_contains(right_tail_term, "beside_singleton"):
                    _require_children_count(right_head_tail, 5)
                    _require_children_count(right_tail_term, 5)

                    y_m = right_head_tail.children[0]
                    y_q = right_head_tail.children[1]
                    right_swap = right_tail_term.children[4]

                    if _tree_root_is(right_swap, "swap"):
                        _require_children_count(right_swap, 19)
                        p = right_swap.children[1]
                        q = right_swap.children[2]
                        if m == y_m and n == x_n and p == x_p and q == y_q:
                            return False

            elif _tree_root_is(head, "swap") and _tree_root_contains(right_head, "beside_cons") and _tree_root_contains(right_tail, "before_cons"):
                left_term = _tree_child(head, 4)
                _require_children_count(left_term, 19)
                _require_children_count(right_head, 11)
                _require_children_count(right_tail, 9)

                m = left_term.children[1]
                n = left_term.children[2]
                x_n = right_head.children[1]
                x_p = right_head.children[4]
                right_head_tail = right_head.children[10]
                right_tail_head = right_tail.children[7]

                if _tree_root_contains(right_head_tail, "beside_singleton") and _tree_root_contains(right_tail_head, "beside_singleton"):
                    _require_children_count(right_head_tail, 5)
                    _require_children_count(right_tail_head, 5)

                    y_m = right_head_tail.children[0]
                    y_q = right_head_tail.children[1]
                    right_swap = right_tail_head.children[4]

                    if _tree_root_is(right_swap, "swap"):
                        _require_children_count(right_swap, 19)
                        p = right_swap.children[1]
                        q = right_swap.children[2]
                        if m == y_m and n == x_n and p == x_p and q == y_q:
                            return False
        return True

    @staticmethod
    def swaplaw2(head: Tree[str], tail: Tree[str]) -> bool:
        """
        before(besides(swap(m+n, m, n), copy(p,edge())), besides(copy(n, edge()), swap(m+p, m, p)))
        ->
        swap(m + n + p, m, n+p)

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched

        """

        if _tree_root_contains(head, "beside_cons") and _tree_root_contains(tail, "before_singleton"):
            _require_children_count(head, 11)
            _require_children_count(tail, 6)

            left_head = _tree_child(head, 9)
            left_tail = _tree_child(head, 10)
            right_term = _tree_child(tail, 5)

            if _tree_root_is(left_head, "swap") and _tree_root_contains(left_tail, "beside_singleton") and _tree_root_contains(right_term, "beside_cons"):
                _require_children_count(left_head, 19)
                _require_children_count(left_tail, 5)
                _require_children_count(right_term, 11)

                m = left_head.children[1]
                n = left_head.children[2]
                left_tail_term = left_tail.children[4]  # swap(p, 0, p)
                right_head = right_term.children[9]  # swap(n, 0, n)
                right_tail = right_term.children[10]

                if _tree_root_is(left_tail_term, "edges") and _tree_root_is(right_head, "edges") and _tree_root_contains(right_tail, "beside_singleton"):
                    _require_children_count(left_tail_term, 17)
                    _require_children_count(right_head, 17)
                    _require_children_count(right_tail, 5)

                    p = left_tail_term.children[0]
                    right_n = right_head.children[0]
                    right_tail_term = right_tail.children[4]  # swap(m+p, m, p)

                    if _tree_root_is(right_tail_term, "swap") and n == right_n:
                        _require_children_count(right_tail_term, 19)
                        right_m = right_tail_term.children[1]
                        right_p = right_tail_term.children[2]
                        if m == right_m and p == right_p:
                            return False

        elif _tree_root_contains(head, "beside_cons") and _tree_root_contains(tail, "before_cons"):
            _require_children_count(head, 11)
            _require_children_count(tail, 9)

            left_head = _tree_child(head, 9)
            left_tail = _tree_child(head, 10)
            right_term = _tree_child(tail, 7)

            if _tree_root_is(left_head, "swap") and _tree_root_contains(left_tail, "beside_singleton") and _tree_root_contains(right_term, "beside_cons"):
                _require_children_count(left_head, 19)
                _require_children_count(left_tail, 5)
                _require_children_count(right_term, 11)

                m = left_head.children[1]
                n = left_head.children[2]
                left_tail_term = left_tail.children[4]  # swap(p, 0, p)
                right_head = right_term.children[9]  # swap(n, 0, n)
                right_tail = right_term.children[10]

                if _tree_root_is(left_tail_term, "edges") and _tree_root_is(right_head, "edges") and _tree_root_contains(right_tail, "beside_singleton"):
                    _require_children_count(left_tail_term, 17)
                    _require_children_count(right_head, 17)
                    _require_children_count(right_tail, 5)

                    p = left_tail_term.children[0]
                    right_n = right_head.children[0]
                    right_tail_term = right_tail.children[4]  # swap(m+p, m, p)

                    if _tree_root_is(right_tail_term, "swap") and n == right_n:
                        _require_children_count(right_tail_term, 19)
                        right_m = right_tail_term.children[1]
                        right_p = right_tail_term.children[2]
                        if m == right_m and p == right_p:
                            return False
        return True

    @staticmethod
    def swaplaw3(head: Tree[str], tail: Tree[str]) -> bool:
        """
        before(swap(m+n, m, n), swap(n+m, n, m))
        ->
        copy(m+n, edge())

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """
        if _tree_root_contains(head, "beside_singleton") and _tree_root_contains(tail, "before_singleton"):
            _require_children_count(head, 5)
            _require_children_count(tail, 6)

            left_term = _tree_child(head, 4)
            right_term = _tree_child(tail, 5)

            if _tree_root_is(left_term, "swap") and _tree_root_contains(right_term, "beside_singleton"):
                _require_children_count(left_term, 19)
                _require_children_count(right_term, 5)

                m = left_term.children[1]
                n = left_term.children[2]
                right_beside = right_term.children[4]

                if _tree_root_is(right_beside, "swap"):
                    _require_children_count(right_beside, 19)
                    right_n = right_beside.children[1]
                    right_m = right_beside.children[2]
                    if m == right_m and n == right_n:
                        return False

        elif _tree_root_contains(head, "beside_singleton") and _tree_root_contains(tail, "before_cons"):
            _require_children_count(head, 5)
            _require_children_count(tail, 9)

            left_term = _tree_child(head, 4)
            right_term = _tree_child(tail, 7)

            if _tree_root_is(left_term, "swap") and _tree_root_contains(right_term, "beside_singleton"):
                _require_children_count(left_term, 19)
                _require_children_count(right_term, 5)

                m = left_term.children[1]
                n = left_term.children[2]
                right_beside = right_term.children[4]

                if _tree_root_is(right_beside, "swap"):
                    _require_children_count(right_beside, 19)
                    right_n = right_beside.children[1]
                    right_m = right_beside.children[2]
                    if m == right_m and n == right_n:
                        return False
        return True

    @staticmethod
    def swaplaw4(head: Tree[str], tail: Tree[str]) -> bool:
        """
        before(besides(copy(m, edge()), swap(n+p, n, p)), besides(swap(m+p, m, p), copy(n,edge())))
        ->
        swap(m + n + p, m+n, p)

        forbid the pattern on the left-hand side of the rewrite rule by returning False if it is matched
        """
        if _tree_root_contains(head, "beside_cons") and _tree_root_contains(tail, "before_singleton"):
            _require_children_count(head, 11)
            _require_children_count(tail, 6)

            left_head = _tree_child(head, 9)
            left_tail = _tree_child(head, 10)
            right_term = _tree_child(tail, 5)

            if _tree_root_is(left_head, "edges") and _tree_root_contains(left_tail, "beside_singleton") and _tree_root_contains(right_term, "beside_cons"):
                _require_children_count(left_head, 17)
                _require_children_count(left_tail, 5)
                _require_children_count(right_term, 11)

                m = left_head.children[0]
                left_tail_term = left_tail.children[4]
                right_head = right_term.children[9]
                right_tail = right_term.children[10]

                if _tree_root_is(left_tail_term, "swap") and _tree_root_is(right_head, "swap") and _tree_root_contains(right_tail, "beside_singleton"):
                    _require_children_count(left_tail_term, 19)
                    _require_children_count(right_head, 19)
                    _require_children_count(right_tail, 5)

                    n = left_tail_term.children[1]
                    p = left_tail_term.children[2]
                    right_m = right_head.children[1]
                    right_p = right_head.children[2]
                    right_tail_term = right_tail.children[4]

                    if _tree_root_is(right_tail_term, "edges") and m == right_m and p == right_p:
                        _require_children_count(right_tail_term, 17)
                        right_n = right_tail_term.children[0]
                        if n == right_n:
                            return False

        elif _tree_root_contains(head, "beside_cons") and _tree_root_contains(tail, "before_cons"):
            _require_children_count(head, 11)
            _require_children_count(tail, 9)

            left_head = _tree_child(head, 9)
            left_tail = _tree_child(head, 10)
            right_term = _tree_child(tail, 7)

            if _tree_root_is(left_head, "edges") and _tree_root_contains(left_tail, "beside_singleton") and _tree_root_contains(right_term, "beside_cons"):
                _require_children_count(left_head, 17)
                _require_children_count(left_tail, 5)
                _require_children_count(right_term, 11)

                m = left_head.children[0]
                left_tail_term = left_tail.children[4]
                right_head = right_term.children[9]
                right_tail = right_term.children[10]

                if _tree_root_is(left_tail_term, "swap") and _tree_root_is(right_head, "swap") and _tree_root_contains(right_tail, "beside_singleton"):
                    _require_children_count(left_tail_term, 19)
                    _require_children_count(right_head, 19)
                    _require_children_count(right_tail, 5)

                    n = left_tail_term.children[1]
                    p = left_tail_term.children[2]
                    right_m = right_head.children[1]
                    right_p = right_head.children[2]
                    right_tail_term = right_tail.children[4]

                    if _tree_root_is(right_tail_term, "edges") and m == right_m and p == right_p:
                        _require_children_count(right_tail_term, 17)
                        right_n = right_tail_term.children[0]
                        if n == right_n:
                            return False
        return True
