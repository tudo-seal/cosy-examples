"""What a pool file promises across a restart, and what a restart must not silently break.

A pool is paid for one training at a time, so the ways of losing one are worth naming and pinning.
Each of the three groups below pins one of them.

The writer refuses a second writer on a finished pool, so that a rerun cannot spend the trainings
again on top of the results. Resuming reads the file first and drops a torn tail before it appends,
so that no record ends up behind bad bytes where the next read stops, and it rewrites through a
sibling path, so that an interruption during the resume leaves the pool alone. The grouping
compares the terms at their positions, so that two different draws merged into one file cannot pass
as repeated trainings of one architecture.

These are tests over the file format alone. They write pools of a few records and never build a
search space or train anything.
"""

from __future__ import annotations

import pytest
from cosy.core.tree import Tree

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_pool import (
    POOL_PHASE,
    TermPoolWriter,
    TruncatedTermPool,
    as_pool,
    read_term_pool,
    resume_term_pool,
)


def _cell_identity():
    """The header fields that say which pool a file is, for one cell.

    A cell is one dataset paired with one requested type, and its identity is every field that a
    mismatch on makes the file's numbers wrong rather than merely different: what was searched, how
    it was drawn, and how many trainings that came to. Written out as a literal here rather than
    built by a driver, because what these tests need is a provenance a reader can compare field by
    field, not the code that assembles one.

    Returns:
        dict: One pool's identity.
    """
    return {
        "dataset": "usps",
        "target": "L5",
        "n_in": 256,
        "batch_size": 64,
        "sampling": "depth-bounded",
        "epochs": 50,
        "length": 5,
        "seed": 0,
        "pool_size": 150,
        "repeat_architectures": 8,
        "repeats": 4,
    }


# --- the writer -----------------------------------------------------------------------------
def test_a_second_writer_on_a_finished_pool_is_an_error(tmp_path):
    """A finished pool is one training per record, and overwriting it spends them all again."""
    path = tmp_path / "pool_usps_L5.pickle"
    TermPoolWriter(path, provenance={"cell": _cell_identity()}, mode="x").close()
    with pytest.raises(FileExistsError):
        TermPoolWriter(path, provenance={"cell": _cell_identity()}, mode="x")


def test_a_run_still_overwrites_its_own_file(tmp_path):
    """The run's CSV is overwritten in the same breath, so its pickle has to be too.

    Making ``x`` the default would leave the two artifacts able to disagree about which run they
    belong to, which is a different defect rather than the same one fixed.
    """
    path = tmp_path / "run_terms.pickle"
    TermPoolWriter(path, provenance={"run": "first"}).close()
    TermPoolWriter(path, provenance={"run": "second"}).close()
    header, _records = read_term_pool(path)
    assert header["provenance"] == {"run": "second"}


def test_appending_cannot_restate_the_provenance(tmp_path):
    """The header is already on disk, and a second one would be dropped without saying so."""
    path = tmp_path / "pool.pickle"
    TermPoolWriter(path, provenance={"cell": _cell_identity()}, mode="x").close()
    with pytest.raises(ValueError, match="cannot restate the provenance"):
        TermPoolWriter(path, provenance={"cell": _cell_identity()}, mode="a")


# --- resuming -------------------------------------------------------------------------------
def _write_pool(path, count, provenance=None):
    """Write ``count`` pool records with distinguishable terms."""
    with TermPoolWriter(path, provenance=provenance or {"cell": _cell_identity()},
                        mode="x") as writer:
        for index in range(count):
            writer.write(POOL_PHASE, index, Tree(f"arch{index}"), {"accuracy": 0.5 + index / 100})


def test_a_resumed_pool_keeps_its_records_and_its_header(tmp_path):
    """Continuing must not cost what was already paid for."""
    path = tmp_path / "pool.pickle"
    _write_pool(path, 3)
    header, records, writer, salvaged = resume_term_pool(path)
    writer.write(POOL_PHASE, 3, Tree("arch3"), {"accuracy": 0.53})
    writer.close()

    assert not salvaged
    assert len(records) == 3
    assert header["provenance"]["cell"]["dataset"] == "usps"
    header_again, records_again = read_term_pool(path)
    assert header_again == header
    assert [record.index for record in records_again] == [0, 1, 2, 3]


def test_a_torn_tail_is_dropped_before_anything_is_appended(tmp_path):
    """Appending behind broken bytes hides every record written after them.

    The reader stops at the tear, so the continued pool would look shorter than it is, and would
    look exactly like a run that stopped early on purpose.
    """
    path = tmp_path / "pool.pickle"
    _write_pool(path, 3)
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) - 20])
    with pytest.raises(TruncatedTermPool):
        read_term_pool(path)

    _header, records, writer, salvaged = resume_term_pool(path)
    writer.write(POOL_PHASE, 2, Tree("arch2"), {"accuracy": 0.52})
    writer.close()

    assert salvaged
    assert len(records) == 2
    _header_again, records_again = read_term_pool(path)
    assert [record.index for record in records_again] == [0, 1, 2]


def test_an_interrupted_rewrite_leaves_the_pool_alone(tmp_path):
    """The rewrite goes to a sibling and is moved into place.

    A leftover sibling means a previous resume was interrupted mid-rewrite. It is an error rather
    than something to clean up quietly, because the pool itself is still intact and someone should
    see that it happened.
    """
    path = tmp_path / "pool.pickle"
    _write_pool(path, 2)
    (tmp_path / "pool.pickle.resuming").write_bytes(b"leftover")
    with pytest.raises(FileExistsError):
        resume_term_pool(path)
    _header, records = read_term_pool(path)
    assert len(records) == 2


# --- grouping -------------------------------------------------------------------------------
def test_a_repeat_of_another_term_is_refused(tmp_path):
    """Two draws merged into one pool would read as training noise.

    The index alone stops identifying an architecture once records can come from two writes of the
    same pool, which is what resuming makes possible. A mismatch here would inflate the
    within-architecture variance and fail the pool for the one reason that is not true of it.
    """
    path = tmp_path / "pool.pickle"
    with TermPoolWriter(path, provenance={}, mode="x") as writer:
        writer.write(POOL_PHASE, 0, Tree("arch0"), {"accuracy": 0.5}, repeat=0)
        writer.write(POOL_PHASE, 0, Tree("somethingelse"), {"accuracy": 0.6}, repeat=1)
    _header, records = read_term_pool(path)
    with pytest.raises(ValueError, match="trained a different term"):
        as_pool(records)


def test_the_same_term_as_a_repeat_is_accepted(tmp_path):
    """Structural equality decides, not object identity. A resumed term is a different object."""
    path = tmp_path / "pool.pickle"
    with TermPoolWriter(path, provenance={}, mode="x") as writer:
        writer.write(POOL_PHASE, 0, Tree("arch0", (Tree("conv"),)), {"accuracy": 0.5}, repeat=0)
        writer.write(POOL_PHASE, 0, Tree("arch0", (Tree("conv"),)), {"accuracy": 0.6}, repeat=1)
    _header, records = read_term_pool(path)
    pool = as_pool(records)
    assert pool["retrained"] == [0]
    assert len(pool["repeats"]) == 1
