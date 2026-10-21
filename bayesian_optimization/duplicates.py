"""When two terms are one evaluation: the loop's duplicate test, exchangeable.

The ask/tell loop rejects on the proposal side: a candidate the dataset already holds, or one
still being evaluated, is not proposed again. *Which* candidates count as held is a test, and the
default is the one every loop of this package always used, term equality. A search that knows a
coarser equivalence -- two spellings of one network, which train as one -- hands the loops a test
of its own, and they reject, skip or answer from the dataset by that test wherever they compared
terms. What the surrogate conditions on is not touched: it still sees the distinct terms.

A test is a bucket ``key(term)`` and, inside a bucket, ``same(a, b)``. Two terms with different
keys are never one evaluation; two with one key are when ``same`` says so, so a key may be coarse
(a hash that collides) as long as ``same`` decides. The default's key is the term itself and its
``same`` is ``==``, so :class:`KnownTerms` under it holds exactly what a set of the terms holds.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator
from typing import Any

__all__ = ["TERM_EQUALITY", "DuplicateTest", "KnownTerms"]


class DuplicateTest:
    """When two terms are one evaluation; this class itself is term equality.

    A test of a caller's own overrides both methods, or ``key`` alone when equality inside a bucket
    is what it means. Both must be consistent: ``same(a, b)`` only for terms with one key.
    """

    def key(self, term: Any) -> Hashable:
        """The bucket a term falls in; two terms in different buckets are never one evaluation.

        Args:
            term: The term.

        Returns:
            Hashable: The term itself.
        """
        bucket: Hashable = term
        return bucket

    def same(self, a: Any, b: Any) -> bool:
        """Whether two terms of one bucket are one evaluation.

        Args:
            a: A term.
            b: A term with the same key.

        Returns:
            bool: ``a == b``.
        """
        return bool(a == b)


#: The default test of every loop: term equality, as before the test was exchangeable.
TERM_EQUALITY = DuplicateTest()


class KnownTerms:
    """Terms held under a duplicate test: ``term in known`` asks the test, not equality alone.

    It holds one term per class the test forms, the first one added, so its length counts
    evaluations rather than spellings. Under :data:`TERM_EQUALITY` it is a set of terms.

    Args:
        terms: The terms to hold from the start, in order.
        test: The duplicate test. (Default value = :data:`TERM_EQUALITY`)
    """

    def __init__(self, terms: Iterable[Any] = (), test: DuplicateTest = TERM_EQUALITY) -> None:
        self.test = test
        self._buckets: dict[Hashable, list[Any]] = {}
        self._count = 0
        for term in terms:
            self.add(term)

    def match(self, term: Any) -> Any | None:
        """The held term ``term`` is one evaluation with, or None.

        Args:
            term: The term to look up.

        Returns:
            Any | None: The first held term the test calls the same, or None.
        """
        for held in self._buckets.get(self.test.key(term), ()):
            if self.test.same(term, held):
                return held
        return None

    def add(self, term: Any) -> bool:
        """Hold a term unless the test calls it one evaluation with a held one.

        Args:
            term: The term.

        Returns:
            bool: Whether the term was new under the test.
        """
        bucket = self._buckets.setdefault(self.test.key(term), [])
        if any(self.test.same(term, held) for held in bucket):
            return False
        bucket.append(term)
        self._count += 1
        return True

    def union(self, terms: Iterable[Any]) -> KnownTerms:
        """A new holding of these terms and the given ones, under the same test; this one unchanged.

        Args:
            terms: The terms to add.

        Returns:
            KnownTerms: The union.
        """
        both = KnownTerms(self, self.test)
        for term in terms:
            both.add(term)
        return both

    def copy(self) -> KnownTerms:
        """A new holding of the same terms under the same test.

        Returns:
            KnownTerms: The copy.
        """
        return KnownTerms(self, self.test)

    def __contains__(self, term: Any) -> bool:
        return self.match(term) is not None

    def __iter__(self) -> Iterator[Any]:
        for bucket in self._buckets.values():
            yield from bucket

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return f"KnownTerms({self._count} held, test={type(self.test).__name__})"
