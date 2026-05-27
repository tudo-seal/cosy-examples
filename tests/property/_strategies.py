from __future__ import annotations

from hypothesis import strategies as st
from cosy.core.tree import Tree

_LABELS = ["A", "B", "C", "D", "E", "X", "Y", "Z"]


def small_trees(max_depth: int = 3, max_branching: int = 3) -> st.SearchStrategy[Tree]:
    """Hypothesis strategy that generates small random Tree[str] objects."""
    base: st.SearchStrategy[Tree] = st.sampled_from(_LABELS).map(Tree)

    def extend(subtree_st: st.SearchStrategy[Tree]) -> st.SearchStrategy[Tree]:
        return st.builds(
            lambda label, children: Tree(label, tuple(children)),
            st.sampled_from(_LABELS),
            st.lists(subtree_st, min_size=1, max_size=max_branching),
        )

    return st.recursive(base, extend, max_leaves=2**max_depth)
