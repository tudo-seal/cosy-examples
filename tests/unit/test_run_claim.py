"""A run's names, taken where it first writes and given back if it fails before measuring anything.

A driver that writes a run's configuration before the run, so that an interrupted run keeps its
provenance, writes before the run layer can refuse anything.  ``RunClaim`` takes the names there,
each configuration created exclusively, so that a second start under one of them is refused before
it writes; gives every file of the runs back if anything fails before the first evaluation begins,
so that a corrected retry under the same names runs; and catches the termination signals a setup
killed there receives -- ``tmux kill-session`` sends SIGHUP -- in that window only.  From the first
evaluation on nothing is removed.
"""

from __future__ import annotations

import errno
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Self

import pytest

from bayesian_optimization.runs import RunArtifacts, RunClaim, run_search, write_run_metadata
from bayesian_optimization.runs import claims as claims_module
from tests.unit.test_run_driver import SCHEMA, _metrics
from tests.unit.test_run_layer_up_front import _bo


def _names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def _run(csv_path: Path, evaluate: Any, **kwargs: Any) -> Any:
    return run_search(_bo(3), evaluate, schema=SCHEMA, csv_path=str(csv_path), pretty_algebra=dict,
                      echo=lambda line: None, refuse_taken=False, **kwargs)


def test_a_claim_takes_every_run_s_configuration_exclusively(tmp_path):
    with RunClaim([RunArtifacts(str(tmp_path / "x.csv")), RunArtifacts(str(tmp_path / "y.csv"))]):
        assert _names(tmp_path) == ["x_config.json", "y_config.json"]
        # a second start that claims a free name and a taken one takes neither
        with pytest.raises(FileExistsError, match="y_config.json"), RunClaim(
            [RunArtifacts(str(tmp_path / "z.csv")), RunArtifacts(str(tmp_path / "y.csv"))]
        ):
            pass
        assert _names(tmp_path) == ["x_config.json", "y_config.json"]


def test_a_name_another_run_took_since_the_start_is_refused_where_the_claim_is_taken(tmp_path):
    """``x_ea.csv`` is another run's CSV and this run's EA log."""
    (tmp_path / "x_ea.csv").write_text("another run's rows\n")
    with pytest.raises(FileExistsError, match="x_ea.csv"), RunClaim(
        [RunArtifacts(str(tmp_path / "x.csv"))]
    ):
        pass
    assert _names(tmp_path) == ["x_ea.csv"]


@pytest.mark.parametrize(("n_design", "n_passes"), [(0, 1), (2, -1), (0, 0)])
def test_a_run_the_run_layer_refuses_inside_the_claim_leaves_nothing(tmp_path, n_design, n_passes):
    csv_path = tmp_path / "x.csv"
    with pytest.raises(ValueError), RunClaim([RunArtifacts(str(csv_path))]) as claim:
        write_run_metadata(str(csv_path), {"a": "configuration"})
        _run(csv_path, claim.evaluate(_metrics), n_design=n_design, n_passes=n_passes)
    assert _names(tmp_path) == []
    # the effect: the corrected retry under the same name runs
    with RunClaim([RunArtifacts(str(csv_path))]) as claim:
        _run(csv_path, claim.evaluate(_metrics), n_design=2, n_passes=1)
    assert (tmp_path / "x.csv").is_file()


def test_a_run_whose_first_evaluation_fails_keeps_everything_and_its_names(tmp_path):
    csv_path = tmp_path / "x.csv"

    def gives_out(term: Any) -> Any:
        raise RuntimeError("the first training gave out")

    with pytest.raises(RuntimeError, match="gave out"), RunClaim(
        [RunArtifacts(str(csv_path))]
    ) as claim:
        write_run_metadata(str(csv_path), {"a": "configuration"})
        _run(csv_path, claim.evaluate(gives_out), n_design=2, n_passes=1)
    assert {"x.csv", "x_config.json", "x_terms.pickle"} <= set(_names(tmp_path))
    with pytest.raises(FileExistsError), RunClaim([RunArtifacts(str(csv_path))]):
        pass


def test_another_run_s_csv_named_like_one_of_this_run_s_files_is_not_given_back(tmp_path):
    with pytest.raises(RuntimeError), RunClaim([RunArtifacts(str(tmp_path / "x.csv"))]):
        # a run named x_ea.csv starts after this one took its names, and this one fails
        (tmp_path / "x_ea_config.json").write_text("{}\n")
        (tmp_path / "x_ea.csv").write_text("another run's rows\n")
        raise RuntimeError("this run fails before it measures")
    assert _names(tmp_path) == ["x_ea.csv", "x_ea_config.json"]


class _WithAStore(RunArtifacts):
    SUFFIXES = {**RunArtifacts.SUFFIXES, "store": "_store"}  # noqa: RUF012, as on its base


def test_a_directory_is_given_back_only_empty(tmp_path):
    """A directory of the run holds what the run judged worth keeping; a claim cannot judge it."""
    with pytest.raises(RuntimeError), RunClaim([_WithAStore(str(tmp_path / "x.csv"))]):
        (tmp_path / "x_store").mkdir()
        (tmp_path / "x_store" / "kept").write_text("something\n")
        raise RuntimeError("fails")
    assert _names(tmp_path) == ["x_store"]
    with pytest.raises(RuntimeError), RunClaim([_WithAStore(str(tmp_path / "y.csv"))]):
        (tmp_path / "y_store").mkdir()
        raise RuntimeError("fails")
    assert _names(tmp_path) == ["x_store"]


class _FullDisk:
    def __init__(self, handle: Any) -> None:
        self.handle = handle

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.handle.close()

    def write(self, text: str) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")


def test_a_claim_whose_placeholder_cannot_be_written_is_given_back(monkeypatch, tmp_path):
    real_open = open
    monkeypatch.setattr(
        claims_module, "open",
        lambda path, mode="r", *args, **kwargs: _FullDisk(real_open(path, mode, *args, **kwargs)),
        raising=False,
    )
    with pytest.raises(OSError, match="No space left"), RunClaim(
        [RunArtifacts(str(tmp_path / "a.csv")), RunArtifacts(str(tmp_path / "b.csv"))]
    ):
        pass
    assert _names(tmp_path) == []


def test_termination_is_caught_only_while_the_files_can_be_given_back(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    seen: dict[str, Any] = {}
    with RunClaim([RunArtifacts(str(tmp_path / "x.csv"))]) as claim:
        seen["in the window"] = signal.getsignal(signal.SIGTERM)

        def measure(term: Any) -> Any:
            seen.setdefault("evaluating", signal.getsignal(signal.SIGTERM))
            return _metrics(term)

        _run(tmp_path / "x.csv", claim.evaluate(measure), n_design=2, n_passes=1)
    assert seen["in the window"] not in (before, signal.SIG_DFL, signal.SIG_IGN)
    assert seen["evaluating"] == before
    assert signal.getsignal(signal.SIGTERM) == before


def test_a_run_that_evaluates_nothing_hands_the_signals_back_at_its_end(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    with RunClaim([RunArtifacts(str(tmp_path / "x.csv"))]):
        pass
    assert signal.getsignal(signal.SIGTERM) == before
    # whatever an earlier test left behind: not the claim's own handler
    assert signal.getsignal(signal.SIGTERM) is not claims_module._terminate


_CHILD = """
import signal, sys, time
signal.signal(signal.SIGHUP, signal.SIG_DFL)  # as in a terminal or a tmux session, not under nohup
from pathlib import Path
from bayesian_optimization.runs import RunArtifacts, RunClaim, write_run_metadata
with RunClaim([RunArtifacts(sys.argv[2])]):
    write_run_metadata(sys.argv[2], {"a": "configuration"})
    Path(sys.argv[1]).write_text("between the claim and the first evaluation")
    time.sleep(300)
"""


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP])
def test_a_run_terminated_before_its_first_evaluation_gives_everything_back(tmp_path, signum):
    ready, csv_path = tmp_path / "ready", tmp_path / "runs" / "x.csv"
    csv_path.parent.mkdir()
    root = Path(__file__).resolve().parents[2]
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(ready), str(csv_path)], cwd=root,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 120
        while not ready.exists():
            assert child.poll() is None, child.communicate()[1][-2000:]
            assert time.monotonic() < deadline, "the child never reached the window"
            time.sleep(0.05)
        assert (csv_path.parent / "x_config.json").is_file(), "the child holds its name"
        child.send_signal(signum)
        _out, err = child.communicate(timeout=60)
    finally:
        child.kill()
    assert child.returncode == -signum, err[-2000:]
    assert _names(csv_path.parent) == []


def test_a_handler_the_caller_installed_stays_in_the_window(tmp_path):
    """The claim catches a signal only where it would terminate the process; a caller who handles
    it keeps handling it, as it would without the claim."""

    def mine(signum: int, frame: Any) -> None:
        """The caller's own."""

    previous = signal.signal(signal.SIGTERM, mine)
    try:
        with RunClaim([RunArtifacts(str(tmp_path / "x.csv"))]):
            assert signal.getsignal(signal.SIGTERM) is mine
    finally:
        signal.signal(signal.SIGTERM, previous)


def test_the_termination_signals_wait_while_the_files_go_back(tmp_path, monkeypatch):
    """A second signal would cut the give-back short; it is held back until the files are gone and
    the handlers are the caller's again, and then it terminates as it would have."""
    held: list[set[int]] = []
    give_back = claims_module._give_back

    def giving_back(runs: Any) -> None:
        held.append(set(signal.pthread_sigmask(signal.SIG_BLOCK, [])))
        give_back(runs)

    monkeypatch.setattr(claims_module, "_give_back", giving_back)
    with pytest.raises(RuntimeError), RunClaim([RunArtifacts(str(tmp_path / "x.csv"))]):
        raise RuntimeError("fails before it measures")

    assert {signal.SIGTERM, signal.SIGHUP} <= held[0]
    assert not {signal.SIGTERM, signal.SIGHUP} & set(signal.pthread_sigmask(signal.SIG_BLOCK, []))


def test_a_first_evaluation_in_another_thread_leaves_the_handlers_to_the_end(tmp_path):
    import threading

    before = signal.getsignal(signal.SIGTERM)
    answers: list[Any] = []
    with RunClaim([RunArtifacts(str(tmp_path / "x.csv"))]) as claim:
        measure = claim.evaluate(lambda term: {"score": 1.0})
        worker = threading.Thread(target=lambda: answers.append(measure("a term")))
        worker.start()
        worker.join()
        assert answers == [{"score": 1.0}]
        assert claim.evaluations_begun
    assert signal.getsignal(signal.SIGTERM) == before


_SIGNALLED_WHILE_CLAIMING = """
import os, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_DFL)
from bayesian_optimization.runs import RunArtifacts, RunClaim
from bayesian_optimization.runs import claims
real = claims._claim_names

def claiming(*args):
    real(*args)
    os.kill(os.getpid(), signal.SIGTERM)  # the names taken, and the signal right after
    time.sleep(30)

claims._claim_names = claiming
with RunClaim([RunArtifacts(sys.argv[1]), RunArtifacts(sys.argv[2])]):
    time.sleep(30)
"""


def test_a_signal_while_the_names_are_taken_gives_them_back(tmp_path):
    root = Path(__file__).resolve().parents[2]
    child = subprocess.run(
        [sys.executable, "-c", _SIGNALLED_WHILE_CLAIMING, str(tmp_path / "x.csv"),
         str(tmp_path / "y.csv")],
        cwd=root, capture_output=True, text=True, timeout=120,
    )
    assert child.returncode == -signal.SIGTERM, child.stderr[-2000:]
    assert _names(tmp_path) == []


def test_a_name_taken_between_the_refusal_and_the_claim_stays_its_holder_s(tmp_path, monkeypatch):
    """Another start takes a name after this claim's refusal and before its exclusive create; the
    create refuses, and only what this claim created goes back, never the other run's."""
    claim_names = claims_module._claim_names

    def racing(paths: Any, taken: list[str]) -> None:
        (tmp_path / "y_config.json").write_text("another run's claim\n")
        claim_names(paths, taken)

    monkeypatch.setattr(claims_module, "_claim_names", racing)
    with pytest.raises(FileExistsError, match="y_config.json"), RunClaim(
        [RunArtifacts(str(tmp_path / "x.csv")), RunArtifacts(str(tmp_path / "y.csv"))]
    ):
        pass
    assert _names(tmp_path) == ["y_config.json"]
    assert (tmp_path / "y_config.json").read_text() == "another run's claim\n"
