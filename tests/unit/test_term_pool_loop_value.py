"""A term record keeps the value the loop was handed, and a pool says which format wrote it.

A resumed design has to hand the loop the value it was handed when the design was measured, and a
record that kept only the metrics left the resuming caller to reconstruct that value from them: the
CIFAR driver read the accuracy, which was wrong for every objective that is not the accuracy.  The
record now keeps the value itself.

The pool also changed where its record class lives when the run machinery left the CIFAR example,
so a pool written now cannot be read by a checkout from before that move.  It says so on its
header: a new pool carries a format tag of its own, which an old reader refuses as "not a
cnn_damg_term_pool file", rather than failing on the first record and blaming an interrupted writer.
"""

from __future__ import annotations

import pickle

import pytest
from cosy.core.tree import Tree

from bayesian_optimization.runs.term_pool import (
    FORMAT,
    LEGACY_FORMATS,
    TermPoolWriter,
    TermRecord,
    read_term_pool,
)


def test_a_record_keeps_the_value_the_loop_was_handed(tmp_path):
    path = tmp_path / "run_terms.pickle"
    with TermPoolWriter(path) as writer:
        writer.write("pre_sample", 0, Tree("a"), {"objective_value": 0.3}, loop_value=-0.3)
        writer.write("bo_step", 0, Tree("b"), {"objective_value": 0.2})

    _header, records = read_term_pool(path)
    assert records[0].loop_value == -0.3
    assert records[1].loop_value is None, "a record written without a value says it has none"


def test_a_new_pool_carries_a_format_tag_of_its_own(tmp_path):
    path = tmp_path / "run_terms.pickle"
    with TermPoolWriter(path) as writer:
        writer.write("pre_sample", 0, Tree("a"), {})

    header, _records = read_term_pool(path)
    assert header["format"] == FORMAT
    assert FORMAT not in LEGACY_FORMATS
    assert "cnn_damg_term_pool" in LEGACY_FORMATS


def test_a_pool_written_before_the_move_still_reads_without_a_loop_value(tmp_path):
    """A legacy header, and records pickled before the field existed, read as they are."""
    path = tmp_path / "legacy_terms.pickle"
    record = TermRecord("pre_sample", 0, Tree("a"), {"accuracy": 0.5})
    # A record from before the field: its instance dictionary has no loop_value.
    del record.__dict__["loop_value"]
    with path.open("wb") as handle:
        pickle.dump({"format": "cnn_damg_term_pool", "version": 1, "provenance": {}}, handle)
        pickle.dump(record, handle)

    header, records = read_term_pool(path)
    assert header["format"] == "cnn_damg_term_pool"
    assert records[0].metrics == {"accuracy": 0.5}
    assert records[0].loop_value is None


def test_a_file_of_another_format_names_the_ones_this_reader_accepts(tmp_path):
    path = tmp_path / "other.pickle"
    with path.open("wb") as handle:
        pickle.dump({"format": "something_else", "version": 1}, handle)

    with pytest.raises(ValueError, match=FORMAT) as raised:
        read_term_pool(path)
    assert "cnn_damg_term_pool" in str(raised.value)
