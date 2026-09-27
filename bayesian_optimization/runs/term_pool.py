"""The terms themselves, beside the numbers that a CSV of an optimization run can carry.

A run's CSV records a rendering of each evaluated structure, and a rendering cannot be fed back
into a kernel. Every question that needs the term again is therefore unanswerable once the run has
finished: which kernel orders this run's architectures, what another kernel would have predicted,
whether the surrogate improves when it is refitted offline. The only way to ask one of them is to
train everything a second time.

This module writes the term. One record per evaluation, appended and flushed as it happens, in the
same discipline as the CSV beside it and for the same reason. The artifact exists to survive an
interruption, and a file that only became valid once the run finished would not.

One format serves a run and a pool, because they are the same thing measured twice. A run is a
stream of records whose ``phase`` says where each evaluation came from (``pre_sample``,
``bo_step``, ``random_sample``). A pool is a stream whose phase is ``pool`` and in which an
architecture may appear more than once, distinguished by ``repeat``. :func:`as_pool` groups the
second shape into the ``{terms, rows, repeats, retrained}`` mapping that a kernel study reads over
a pool. Two formats would have meant two readers, and the pool would have been the one nobody
wrote from an experiment.

A truncated tail is reported, never absorbed. Reading stops at the first record that does not
decode, and :class:`TruncatedTermPool` carries the ones that did. A caller that wants the salvage
has to catch that exception and therefore says so. The default is that a broken file is an error
rather than a short one, because a silently shortened pool reads exactly like a run that stopped
early on purpose.

A pool file is written once. :class:`TermPoolWriter` takes the open mode, and a pool driver passes
``"x"``, so that a second writer on a path which already carries a finished pool is an error and
not an overwrite. Continuing an interrupted pool goes through :func:`resume_term_pool` instead.
That is the only path which appends, and it drops a torn tail before it does, so the file it hands
back is one the reader can read to the end.
"""

from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "FORMAT",
    "POOL_PHASE",
    "VERSION",
    "TermPoolWriter",
    "TermRecord",
    "TruncatedTermPool",
    "as_pool",
    "read_term_pool",
    "resume_term_pool",
]

#: Written into the header so a file that is not one of these says so on the first read, rather
#: than unpickling into something shaped almost right.
FORMAT = "cnn_damg_term_pool"
VERSION = 1

#: The phase of a record that belongs to a training pool rather than to a run of the loop.
POOL_PHASE = "pool"


@dataclass
class TermRecord:
    """One evaluation: the term that was trained and what came back.

    Attributes:
        phase (str): Where the evaluation came from. ``pre_sample``, ``bo_step`` and
            ``random_sample`` for a run, ``pool`` for a training pool.
        index (int): The position within that phase. Together with ``phase`` and ``repeat`` this
            identifies the row of the run's CSV that the record belongs to.
        term: The structure itself, as the search produced it.
        metrics (dict): What the evaluation returned, such as the objective value, the accuracy,
            the parameter count and the training seconds.
        repeat (int): ``0`` for the measurement a run would have used, ``1..n`` for repeated
            trainings of the same architecture. Only a pool has these, and they are what a noise
            floor is measured from.
    """

    phase: str
    index: int
    term: Any
    metrics: dict = field(default_factory=dict)
    repeat: int = 0


class TruncatedTermPool(Exception):
    """The file ends inside a record, which is what an interrupted writer leaves behind.

    Attributes:
        header (dict): The file's header, which decoded.
        records (list[TermRecord]): The records that decoded whole, in order.
        path (str): The file this came from.
    """

    def __init__(self, path, header, records, cause):
        super().__init__(
            f"{path} ends inside a record after {len(records)} whole ones "
            f"({type(cause).__name__}: {cause}); the writer was interrupted"
        )
        self.path = path
        self.header = header
        self.records = records


class TermPoolWriter:
    """Appends one record per evaluation, flushed as it goes.

    The file stays open for the whole run rather than being reopened per record, because the point
    of flushing each record is that the file is readable at every moment of a run that may be
    killed at any of them. The class is its own context manager, so the ``open`` below is not a
    leak.

    Args:
        path (str): Where to write.
        provenance (dict): Anything that identifies what produced this file, such as the dataset,
            the target or the run's name. It goes into the header verbatim. This is not a second
            copy of the run's configuration file: that file stays the run's provenance record, and
            this is what makes the pickle recognizable on a machine which received only the pickle.
        mode (str): ``"w"`` overwrites an existing file, which is what a run does. Its CSV is
            overwritten in the same breath, so the two artifacts cannot disagree about which run
            they belong to. ``"x"`` refuses one, which is what a pool does. A pool is paid for in
            trainings, and a second writer on a finished pool's path would spend them again on top
            of the results. ``"a"`` appends without writing a second header and is only reachable
            through :func:`resume_term_pool`, which has already read and checked the first one.
            (Default value = "w")

    Raises:
        ValueError: If ``mode`` is not one of the three, or if ``"a"`` is given a provenance.
            Appending cannot restate what the file already says it is, and accepting a second
            provenance would let the caller believe that it had.
    """

    def __init__(self, path, provenance=None, *, mode="w"):
        if mode not in ("w", "x", "a"):
            msg = f"mode selects how the file is opened and is 'w', 'x' or 'a', not {mode!r}"
            raise ValueError(msg)
        if mode == "a" and provenance is not None:
            msg = (
                "appending cannot restate the provenance; the header is already on disk and this "
                "one would be silently dropped"
            )
            raise ValueError(msg)
        self._path = str(path)
        # Closed by close(), which the context manager calls. See the class docstring for why the
        # handle is held for the lifetime of the writer.
        self._file = open(path, mode + "b")  # noqa: SIM115
        if mode == "a":
            return
        pickle.dump(
            {
                "format": FORMAT,
                "version": VERSION,
                "created": time.time(),
                "provenance": dict(provenance or {}),
            },
            self._file,
        )
        self._file.flush()

    def write(self, phase, index, term, metrics, repeat=0):
        """Append one evaluation and flush it.

        Args:
            phase (str): See :class:`TermRecord`.
            index (int): See :class:`TermRecord`.
            term: The structure that was evaluated.
            metrics (dict): What the evaluation returned. Copied, so that a caller which reuses
                its dict does not rewrite records already on disk.
            repeat (int): See :class:`TermRecord`. (Default value = 0)
        """
        pickle.dump(
            TermRecord(
                phase=phase, index=index, term=term, metrics=dict(metrics), repeat=repeat
            ),
            self._file,
        )
        # Persist immediately. A crash mid-run must not lose completed records.
        self._file.flush()

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def read_term_pool(path):
    """Read a term pool back.

    Args:
        path (str): The file :class:`TermPoolWriter` wrote.

    Returns:
        tuple[dict, list[TermRecord]]: The header and the records, in the order they were written.

    Raises:
        ValueError: If the file is empty or its header is not this format. An empty file is an
            error rather than an empty pool: the writer emits the header before anything else, so
            a file without one was never written by it.
        TruncatedTermPool: If the file ends inside a record. The exception carries the records
            that decoded whole, so a caller that wants an interrupted run's results can take them,
            but has to ask.
    """
    with open(path, "rb") as handle:
        try:
            header = pickle.load(handle)
        except EOFError as exc:
            msg = f"{path} is empty; it carries no term pool header"
            raise ValueError(msg) from exc
        if not isinstance(header, dict) or header.get("format") != FORMAT:
            msg = (
                f"{path} is not a {FORMAT} file; its first object is "
                f"{type(header).__name__} {header!r:.80}"
            )
            raise ValueError(msg)

        records = []
        while True:
            try:
                records.append(pickle.load(handle))
            except EOFError:
                return header, records
            except Exception as exc:  # noqa: BLE001
                raise TruncatedTermPool(str(path), header, records, exc) from exc


def resume_term_pool(path):
    """Read an interrupted pool back and reopen it for appending.

    A pool costs one training per record and runs for hours, on machines that reboot unannounced,
    so continuing has to be possible and has to be safe. The file may end inside a record, and
    appending behind a torn tail would leave the bad bytes in the middle, where the next read stops
    at them and reports the records after them as missing. So the file is rewritten from the
    records that decoded whole, header and all, byte for byte the same header, and only then
    opened for appending.

    The rewrite goes to a sibling path and is moved into place, so an interruption during the
    rewrite itself leaves the original untouched. A leftover sibling is not cleaned up silently. It
    means a previous resume was interrupted mid-rewrite, the pool itself is still intact, and
    someone should look.

    Args:
        path (str): The pool file.

    Returns:
        tuple: The header, the records that survived, an appending :class:`TermPoolWriter`, and
        whether a torn tail was dropped.

    Raises:
        ValueError: If the file is not a term pool. See :func:`read_term_pool`.
        FileExistsError: If the rewrite's sibling path is already taken.
    """
    try:
        header, records = read_term_pool(path)
        salvaged = False
    except TruncatedTermPool as truncated:
        header, records = truncated.header, truncated.records
        salvaged = True

    rewritten = f"{path}.resuming"
    with open(rewritten, "xb") as handle:
        pickle.dump(header, handle)
        for record in records:
            pickle.dump(record, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(rewritten, path)
    return header, records, TermPoolWriter(path, mode="a"), salvaged


def as_pool(records, phase=POOL_PHASE):
    """Group pool records into the shape a kernel study reads.

    Args:
        records (list[TermRecord]): Records from :func:`read_term_pool`.
        phase (str): Which phase counts as the pool. Defaults to ``pool``. Pass a run's phase to
            treat that run's evaluations as a pool. (Default value = POOL_PHASE)

    Returns:
        dict: ``terms`` in index order, ``rows`` with one entry per first measurement, ``repeats``
        with one entry per repeated measurement, and ``retrained``, the positions that have
        repeats.

    Raises:
        ValueError: If an architecture has repeats but no first measurement, if two records claim
            the same ``(index, repeat)``, or if a repeat's term is not the term its first
            measurement trained. Each means the file does not describe what it appears to, and a
            pool assembled around the gap would put a term's metrics on another term's row.
    """
    selected = [record for record in records if record.phase == phase]
    seen = {}
    for record in selected:
        key = (record.index, record.repeat)
        if key in seen:
            msg = (
                f"two records claim index {record.index} repeat {record.repeat}; "
                f"the pool cannot say which measurement belongs to the term"
            )
            raise ValueError(msg)
        seen[key] = record

    firsts = {index: record for (index, repeat), record in seen.items() if repeat == 0}
    orphans = sorted(
        {index for (index, repeat) in seen if repeat != 0 and index not in firsts}
    )
    if orphans:
        msg = (
            f"architectures {orphans} have repeated measurements but no first one; "
            f"the pool has no value for them that a run would have used"
        )
        raise ValueError(msg)

    # A repeat is only a repeat if it trained the same architecture. The index alone does not say
    # so once records can come from two writes of the same pool, which is exactly what
    # :func:`resume_term_pool` makes possible, and a mismatch here would be read as training
    # noise. It would inflate the within-architecture variance a noise floor is built from and
    # fail the pool for the one reason that is not true of it.
    for (index, repeat), record in sorted(seen.items()):
        if repeat != 0 and record.term != firsts[index].term:
            msg = (
                f"repeat {repeat} of architecture {index} trained a different term than its first "
                f"measurement; the two cannot be measurements of one architecture, and reading "
                f"them as such would put architectural variance into the noise floor"
            )
            raise ValueError(msg)

    order = sorted(firsts)
    position = {index: place for place, index in enumerate(order)}
    terms = [firsts[index].term for index in order]
    rows = [
        {
            "architecture": position[index],
            "repeat": 0,
            "term_size": firsts[index].term.size,
            **firsts[index].metrics,
        }
        for index in order
    ]
    repeats = [
        {
            "architecture": position[record.index],
            "repeat": record.repeat,
            "term_size": record.term.size,
            **record.metrics,
        }
        for (index, repeat), record in sorted(seen.items())
        if repeat != 0
    ]
    retrained = sorted({position[index] for (index, repeat) in seen if repeat != 0})
    return {"terms": terms, "rows": rows, "repeats": repeats, "retrained": retrained}
