"""The term file has to survive the thing it exists for, which is an interrupted run.

A run's CSV keeps a rendering of each structure, and a rendering cannot be fed back into a kernel,
so the pickle beside it is the only thing that makes a finished run re-analyzable. The tests here
pin the two properties that decide whether it does. Every record written before an interruption is
readable afterwards, and a file that broke mid-record says so instead of passing as a shorter run.
"""

from __future__ import annotations

import pickle

import pytest
from cosy.core.tree import Tree

from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_pool import (
    FORMAT,
    TermPoolWriter,
    TruncatedTermPool,
    as_pool,
    read_term_pool,
)


def test_records_round_trip_with_the_term_itself(tmp_path):
    """The point of the file: what comes back is the term, not a rendering of it."""
    path = tmp_path / "run_terms.pickle"
    tree = Tree("arch", (Tree("conv"), Tree("linear")))

    with TermPoolWriter(path, provenance={"dataset": "usps"}) as writer:
        writer.write("pre_sample", 0, tree, {"accuracy": 0.94, "n_params": 12578})

    header, records = read_term_pool(path)
    assert header["format"] == FORMAT
    assert header["provenance"] == {"dataset": "usps"}
    (record,) = records
    assert record.term == tree
    assert record.term.size == tree.size
    assert record.metrics["accuracy"] == 0.94
    assert record.phase == "pre_sample"


def test_metrics_are_copied_so_a_reused_dict_cannot_rewrite_history(tmp_path):
    """The loop hands the same dict to the logger twice, and each record keeps its own values.

    What the record holds has to be fixed at the call, not at the moment it reaches the disk. A
    writer that both held the caller's dict and wrote its records later would put the second
    evaluation's accuracy on the first one's record.
    """
    path = tmp_path / "run_terms.pickle"
    metrics = {"accuracy": 0.1}

    with TermPoolWriter(path) as writer:
        writer.write("pre_sample", 0, Tree("a"), metrics)
        metrics["accuracy"] = 0.9
        writer.write("pre_sample", 1, Tree("b"), metrics)

    _header, records = read_term_pool(path)
    assert [record.metrics["accuracy"] for record in records] == [0.1, 0.9]


def test_records_are_readable_while_the_writer_is_still_open(tmp_path):
    """Every record is on disk as soon as it is written, not once the run has ended.

    This is the property the artifact exists for, and the only test here that can see it. The
    others close the writer before they read, so a writer that collected its records and wrote
    them all at close would pass them and still lose everything an interrupted run had done.
    """
    path = tmp_path / "run_terms.pickle"
    writer = TermPoolWriter(path, provenance={"dataset": "usps"})
    try:
        writer.write("bo_step", 0, Tree("arch0"), {"accuracy": 0.5})
        writer.write("bo_step", 1, Tree("arch1"), {"accuracy": 0.6})
        _header, records = read_term_pool(path)
    finally:
        writer.close()

    assert [record.index for record in records] == [0, 1]
    assert [record.metrics["accuracy"] for record in records] == [0.5, 0.6]


def test_everything_written_before_an_interruption_is_readable(tmp_path):
    """A run killed mid-training keeps every evaluation it completed.

    Simulated by truncating the file inside the last record, which is what a process killed between
    two flushes leaves behind.
    """
    path = tmp_path / "run_terms.pickle"
    with TermPoolWriter(path) as writer:
        for index in range(3):
            writer.write("bo_step", index, Tree(f"arch{index}"), {"accuracy": index / 10})

    whole = path.read_bytes()
    # Cut a few bytes off. The last record is now incomplete, the first two are untouched.
    path.write_bytes(whole[:-5])

    with pytest.raises(TruncatedTermPool) as raised:
        read_term_pool(path)
    assert [record.index for record in raised.value.records] == [0, 1]
    assert "interrupted" in str(raised.value)


def test_a_file_that_is_not_a_term_pool_says_so(tmp_path):
    """Reading the wrong pickle must fail loudly rather than yield an empty pool."""
    path = tmp_path / "other.pickle"
    with path.open("wb") as handle:
        pickle.dump({"terms": [], "rows": []}, handle)

    with pytest.raises(ValueError, match=FORMAT):
        read_term_pool(path)


def test_an_empty_file_is_an_error_not_an_empty_pool(tmp_path):
    path = tmp_path / "empty.pickle"
    path.write_bytes(b"")

    with pytest.raises(ValueError, match="empty"):
        read_term_pool(path)


def test_as_pool_groups_repeats_onto_their_architecture(tmp_path):
    """The noise floor needs first measurements and repeats apart, keyed to the same term."""
    path = tmp_path / "pool_terms.pickle"
    trees = [Tree(f"arch{index}") for index in range(3)]
    with TermPoolWriter(path) as writer:
        for index, tree in enumerate(trees):
            writer.write("pool", index, tree, {"accuracy": 0.9 + index / 100})
        for repeat in (1, 2):
            writer.write("pool", 1, trees[1], {"accuracy": 0.5}, repeat=repeat)

    _header, records = read_term_pool(path)
    pool = as_pool(records)

    assert pool["terms"] == trees
    assert [row["architecture"] for row in pool["rows"]] == [0, 1, 2]
    assert [row["repeat"] for row in pool["repeats"]] == [1, 2]
    assert {row["architecture"] for row in pool["repeats"]} == {1}
    assert pool["retrained"] == [1]


def test_as_pool_ignores_records_of_another_phase(tmp_path):
    """A run's evaluations and a pool's can share a file, and the grouping must not mix them."""
    path = tmp_path / "mixed_terms.pickle"
    with TermPoolWriter(path) as writer:
        writer.write("pool", 0, Tree("pooled"), {"accuracy": 0.9})
        writer.write("bo_step", 0, Tree("searched"), {"accuracy": 0.8})

    _header, records = read_term_pool(path)
    assert [term.root for term in as_pool(records)["terms"]] == ["pooled"]
    assert [term.root for term in as_pool(records, phase="bo_step")["terms"]] == ["searched"]


def test_as_pool_refuses_a_repeat_without_a_first_measurement(tmp_path):
    """A repeat whose architecture has no row would shift every later term onto a wrong row."""
    path = tmp_path / "orphan_terms.pickle"
    with TermPoolWriter(path) as writer:
        writer.write("pool", 0, Tree("a"), {"accuracy": 0.9})
        writer.write("pool", 7, Tree("b"), {"accuracy": 0.5}, repeat=1)

    _header, records = read_term_pool(path)
    with pytest.raises(ValueError, match="no first one"):
        as_pool(records)


def test_as_pool_refuses_two_records_for_the_same_measurement(tmp_path):
    path = tmp_path / "duplicate_terms.pickle"
    with TermPoolWriter(path) as writer:
        writer.write("pool", 0, Tree("a"), {"accuracy": 0.9})
        writer.write("pool", 0, Tree("b"), {"accuracy": 0.5})

    _header, records = read_term_pool(path)
    with pytest.raises(ValueError, match="which measurement"):
        as_pool(records)
