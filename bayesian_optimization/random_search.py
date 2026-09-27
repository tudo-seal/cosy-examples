"""Random search as a strategy of the ask/tell loop.

Random search used to exist here only beside a Bayesian run: the CIFAR driver's paired baseline drew
``n + m`` terms from one stream of the loop's sampler, shared the first ``n`` as the design and
evaluated the other ``m`` after the loop.  :class:`RandomSearch` is that arm as a strategy of its
own, run through the same :class:`~bayesian_optimization.loop.AskTellLoop` as Bayesian optimization.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any, ClassVar

from cosy.core.solution_space import SolutionSpace
from cosy.search import Sampler

from .diagnostics import enable_verbose_logging, get_logger
from .initial_sampling import _MAX_DRAWS_PER_TERM
from .loop import NT, AskTellLoop, G, T, _require_hashable
from .state import Diagnostics, Suggestion


class RandomSearch(AskTellLoop[NT, T, G]):
    """Random search over a CoSy solution space, through the same ask/tell loop as BO.

    Every term the search draws, the design's and the passes', comes from ONE stream of its
    sampler, so a run of ``n`` design terms and ``m`` passes evaluates the first ``n + m``
    distinct terms of that stream.  Under the size-uniform sampler every prefix of the stream is
    a sample without replacement, and nothing is skipped; under a sampler that draws with
    replacement, such as the depth-bounded one, a repeated term is skipped and counted.  A design
    handed over with ``initialize(design=...)`` is evaluated as given, and the passes then draw
    from the start of the stream, skipping every term the dataset holds: on a sampler seeded as
    the one that drew that design, they continue exactly where the design ends, which is the shape
    of a paired comparison.

    There is no surrogate and no acquisition.  A pass carries no acquisition value, and
    ``finalize()`` reports ``gp_model`` as ``None`` and an empty trace.

    Parameters
    ----------
    search_space, request, seed, sampler:
        As for :class:`~bayesian_optimization.loop.AskTellLoop`.  There is no initializer: the
        design is the head of the same stream the passes continue.
    max_draws_per_term:
        How many draws in a row may repeat a term the stream already delivered before the search
        gives up on the next one, the bound ``distinct_prefix`` applies.  A held term the stream
        delivers for the first time, such as a term of a handed-over design at the head of the
        stream, is skipped without counting against it.  Reachable only for a sampler that draws
        with replacement.
    """

    #: A pass of random search, as the paired random-search arm was named in a run's rows.
    PASS_PHASE: ClassVar[str] = "random_sample"

    def __init__(
        self,
        search_space: SolutionSpace[NT, T, G] | None,
        request: NT | None = None,
        *,
        seed: int | None = None,
        sampler: Sampler | None = None,
        max_draws_per_term: int = _MAX_DRAWS_PER_TERM,
    ) -> None:
        super().__init__(search_space, request, seed=seed, sampler=sampler)
        self.max_draws_per_term = max_draws_per_term
        self._stream: Iterator[Any] | None = None
        # The terms the open stream has delivered, which is what tells a repeat of the stream from
        # a held term it delivers for the first time.
        self._delivered: set[Any] = set()
        # Skips of the draw in progress, so that a draw that raises still counts what it skipped.
        self._skips_pending: int = 0
        self._terms_skipped: int = 0
        self._logger = get_logger("random_search")

    @property
    def terms_skipped(self) -> int:
        """How many terms of the stream the passes skipped because the dataset already held them.

        Zero under the size-uniform sampler after a drawn design.  After a design handed over, it
        counts the design terms the stream delivers again; under a sampler that draws with
        replacement, it counts the repeats as well.  The design's own repeats are
        :attr:`initial_repeats_rejected`.

        Returns:
            int: The count, 0 again after ``reset()``.
        """
        return self._terms_skipped

    def reset(self) -> None:
        """Reset to UNINITIALIZED and close the stream.

        A second run opens a new stream on the sampler.  The default sampler is rebuilt by the
        reset, so its stream starts over; a sampler handed to the constructor is kept, with the
        random state it has reached, so its new stream is a new sample.
        """
        super().reset()
        self._close_stream()
        self._terms_skipped = 0

    def initialize(
        self,
        *,
        objective: Callable[[Any], float] | None = None,
        x0: Sequence[Any] | None = None,
        y0: Sequence[float] | None = None,
        initial_size: int = 10,
        design: Sequence[Any] | None = None,
    ) -> None:
        """Build the initial dataset, or start the design phase; see :meth:`AskTellLoop.initialize`.

        A call that raises closes the stream, so that a second ``initialize()`` draws the head of
        a new one rather than what the failed attempt left of the old.
        """
        try:
            super().initialize(
                objective=objective, x0=x0, y0=y0, initial_size=initial_size, design=design
            )
        except BaseException:
            self._close_stream()
            raise

    def _close_stream(self) -> None:
        self._stream = None
        self._delivered = set()
        self._skips_pending = 0

    def _next_new_term(self, held: set[Any]) -> Any:
        """Return the next term of the stream that ``held`` does not contain.

        Every term passed over on the way is added to ``_skips_pending``, which the caller reads,
        also when this raises.

        Args:
            held (set[Any]): The terms not to return again.

        Returns:
            Any: The term.

        Raises:
            NotImplementedError: If there is no search space to draw from.
            RuntimeError: If the stream ends, or ``max_draws_per_term`` draws in a row repeat a
                term the stream already delivered, before a new one comes.
        """
        if self._stream is None:
            self._resolve_sampling()
            if self._sampler is None:
                raise NotImplementedError("random search draws its terms and requires a real search_space.")
            self._stream = iter(self._sampler.sample(self._query()))
        repeats_in_a_row = 0
        for term in self._stream:
            _require_hashable([term], "the sampler's stream")
            repeat = term in self._delivered
            self._delivered.add(term)
            if term not in held:
                return term
            self._skips_pending += 1
            if not repeat:
                # Held, but new to this stream: a term of a design handed over, met once at the
                # head of the stream.  That is progress through the stream, not a repeat.
                repeats_in_a_row = 0
                continue
            repeats_in_a_row += 1
            if repeats_in_a_row >= self.max_draws_per_term:
                msg = (
                    f"the sampler's stream repeated a delivered term {repeats_in_a_row} times in "
                    "a row: under a sampler that draws with replacement, the space within its "
                    "bound is exhausted or nearly so"
                )
                raise RuntimeError(msg)
        msg = (
            f"the sampler's stream ended after {len(held)} held terms: every term within its "
            "bound has been drawn, and a repeat is not a new evaluation"
        )
        raise RuntimeError(msg)

    def _draw_design(self, size: int) -> tuple[list[Any], int]:
        """Take the design from the head of the one stream the passes continue.

        Args:
            size (int): The size of the design.

        Returns:
            tuple[list[Any], int]: The design, and how many repeats of the stream were skipped.
        """
        if size < 0:
            msg = f"a design holds a non-negative number of terms, not {size}"
            raise ValueError(msg)
        design: list[Any] = []
        held: set[Any] = set()
        self._skips_pending = 0
        for _ in range(size):
            term = self._next_new_term(held)
            design.append(term)
            held.add(term)
        repeats, self._skips_pending = self._skips_pending, 0
        return design, repeats

    def _propose(self, verbose: bool) -> Suggestion:
        """The next term of the stream the dataset does not hold.

        Args:
            verbose (bool): Switch the package's verbose logging on.

        Returns:
            Suggestion: The term, without an acquisition value.
        """
        if verbose:
            enable_verbose_logging()
        self._skips_pending = 0
        try:
            term = self._next_new_term(self._x_set)
        finally:
            self._terms_skipped += self._skips_pending
            self._skips_pending = 0
        diagnostics: Diagnostics = {
            "timestamp": time.time(),
            "iteration": self._iteration,
            "phase": "main",
        }
        return Suggestion(candidate=term, acquisition_value=None, diagnostics=diagnostics)
