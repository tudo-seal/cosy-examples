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

__all__ = ["TERM_EQUALITY", "DuplicateTest", "KnownTerms", "one_test"]


class DuplicateTest:
    """When two terms are one evaluation; this class itself is term equality.

    A test of a caller's own overrides both methods, or ``key`` alone when equality inside a bucket
    is what it means. Both must be consistent: ``same(a, b)`` only for terms with one key. A class
    that overrides ``same`` and not ``key`` is refused where it is defined: under the default key,
    the term itself, ``same`` would be asked of equal terms only, and the test would be term
    equality whatever it says.

    ``same`` must be an equivalence on each bucket -- reflexive, symmetric and transitive -- and
    both methods functions of their terms. :class:`KnownTerms` holds the first term of each class
    and asks a new term against the held ones only, so under a relation that is not transitive
    what it holds depends on the order the terms arrive in. Neither property is checked.

    Where two tests must be one -- the strategies of a paired run, which are compared on one
    design -- they are one when :func:`one_test` says so: the same object, both term equality, or
    equal by the class's own ``==``. A test whose instances differ by a configuration is one object
    handed to each, or defines ``__eq__`` (and ``__hash__``) over that configuration.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.same is not DuplicateTest.same and cls.key is DuplicateTest.key:
            msg = (
                f"{cls.__name__} overrides same() and not key(): the default key is the term "
                "itself, so same() would be asked of equal terms only and the test would be term "
                "equality; give key() a bucket that the terms same() merges share"
            )
            raise TypeError(msg)

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


def one_test(a: DuplicateTest, b: DuplicateTest) -> bool:
    """Whether two duplicate tests are one.

    They are where they are the same object, where both are term equality (:class:`DuplicateTest`
    itself, however many instances), or where the class's own ``==`` says so; without an ``__eq__``
    of its own, two instances of one class are two tests, since a configuration may tell them apart.

    Args:
        a: A test.
        b: A test.

    Returns:
        bool: Whether they are one.
    """
    if a is b or (type(a) is DuplicateTest and type(b) is DuplicateTest):
        return True
    return type(a) is type(b) and bool(a == b)


class KnownTerms:
    """Terms held under a duplicate test: ``term in known`` asks the test, not equality alone.

    It holds one term per class the test forms, the first one added, so its length counts
    evaluations rather than spellings. Under plain term equality (:class:`DuplicateTest` itself) it
    keeps a set and asks only the set, so membership is the loops' former ``term in set`` exactly,
    down to which side of ``==`` is asked.

    Args:
        terms: The terms to hold from the start, in order.
        test: The duplicate test. (Default value = :data:`TERM_EQUALITY`)
    """

    def __init__(self, terms: Iterable[Any] = (), test: DuplicateTest = TERM_EQUALITY) -> None:
        self.test = test
        self._plain = type(test) is DuplicateTest
        self._set: set[Any] = set()
        self._order: list[Any] = []
        self._buckets: dict[Hashable, list[Any]] = {}
        self._count = 0
        for term in terms:
            self.add(term)

    @property
    def plain(self) -> bool:
        """Whether the test is plain term equality, :class:`DuplicateTest` itself.

        Returns:
            bool: Whether membership is a set's.
        """
        return self._plain

    def as_set(self) -> set[Any]:
        """Under plain term equality, a copy of the set membership asks.

        That set is built term by term in the order the terms were added, as the loops built their
        sets of terms, so its copy iterates as theirs did.

        Returns:
            set[Any]: The copy.

        Raises:
            TypeError: Under any other test, whose membership a set of the held terms would not
                answer.
        """
        if not self._plain:
            msg = (
                f"terms held under {type(self.test).__name__} are no set: a set of them would "
                "answer term equality"
            )
            raise TypeError(msg)
        return set(self._set)

    def match(self, term: Any) -> Any | None:
        """The held term ``term`` is one evaluation with, or None.

        Args:
            term: The term to look up.

        Returns:
            Any | None: The first held term the test calls the same, or None; under plain term
            equality the term itself when an equal one is held.
        """
        if self._plain:
            return term if term in self._set else None
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
        if self._plain:
            if term in self._set:
                return False
            self._set.add(term)
            self._order.append(term)
            self._count += 1
            return True
        bucket = self._buckets.setdefault(self.test.key(term), [])
        if any(self.test.same(term, held) for held in bucket):
            return False
        bucket.append(term)
        self._count += 1
        return True

    def union(self, terms: Iterable[Any]) -> KnownTerms:
        """A new holding of these terms and the given ones, under the same test; this one unchanged.

        The test is asked of the given terms only.

        Args:
            terms: The terms to add.

        Returns:
            KnownTerms: The union.
        """
        both = self.copy()
        for term in terms:
            both.add(term)
        return both

    def copy(self) -> KnownTerms:
        """A new holding of the same terms under the same test, which it does not ask again.

        The held terms are pairwise apart under the test already, so the copy takes their buckets as
        they stand: under a test as costly as an isomorphism, holding them anew would ask it of
        every pair in a bucket.

        Returns:
            KnownTerms: The copy.
        """
        clone = KnownTerms(test=self.test)
        clone._set = set(self._set)
        clone._order = list(self._order)
        clone._buckets = {key: list(bucket) for key, bucket in self._buckets.items()}
        clone._count = self._count
        return clone

    def __contains__(self, term: Any) -> bool:
        if self._plain:
            return term in self._set
        return self.match(term) is not None

    def __iter__(self) -> Iterator[Any]:
        if self._plain:
            yield from self._order
            return
        for bucket in self._buckets.values():
            yield from bucket

    def __len__(self) -> int:
        return self._count

    def __repr__(self) -> str:
        return f"KnownTerms({self._count} held, test={type(self.test).__name__})"
