"""A run's names, taken where it first writes and given back if it fails before measuring anything."""

from __future__ import annotations

import contextlib
import os
import signal
import threading
from collections.abc import Callable, Sequence
from typing import Any

from .artifacts import RunArtifacts, metadata_path_for

_PLACEHOLDER = '{"status": "claimed by a run that has not started"}\n'

#: What ``kill`` and ``tmux kill-session`` send.  SIGINT raises KeyboardInterrupt already, and
#: SIGKILL cannot be caught.
_TERMINATIONS: tuple[int, ...] = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)
)


class _Terminated(BaseException):
    """A termination signal, received while a run can still give its files back."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"terminated by signal {signum}")
        self.signum = signum


def _terminate(signum: int, _frame: Any) -> None:
    raise _Terminated(signum)


class RunClaim:
    """Take a run's names where it first writes, and give them back if it fails before measuring.

    A driver that writes a run's configuration before the run, so that an interrupted run keeps its
    provenance, writes before the run layer can refuse anything, and its setup -- the program, the
    data, the counting tables of a design drawn up front -- takes minutes on a large target.  So the
    names are refused again and taken where the run first writes, each configuration created
    exclusively: a second start under one of them, which passed its refusal at its own start, is
    refused there before it writes, and so is this run if another took one of its file names
    meanwhile (a run named like its EA log).

    Until the first evaluation begins, a failure gives every file of the runs back, so that a
    corrected retry under the same names runs: the run layer's own refusals come after the driver
    wrote its configurations.  The termination signals a setup killed there receives -- SIGTERM,
    and SIGHUP, which ``tmux kill-session`` sends -- are caught in that window only, where the
    main thread can catch them and the process does not ignore them; the files go back and the
    process is terminated by the same signal.  From the first evaluation on nothing is removed: a
    training that gives out keeps its files as evidence, and its names.

    What goes back: every file named after the runs, except one that is another run's CSV with
    that run's configuration beside it; a directory only if it is empty, since what a run keeps in
    one is not the claim's to judge.

    Use::

        with RunClaim([RunArtifacts(csv_path)]) as claim:
            write_run_metadata(csv_path, provenance)
            run_search(strategy, claim.evaluate(evaluate), csv_path=csv_path,
                       refuse_taken=False, ...)

    Attributes:
        runs (list[RunArtifacts]): The runs whose names this takes.
        evaluations_begun (bool): Whether the first evaluation has begun.
    """

    def __init__(self, runs: Sequence[RunArtifacts]) -> None:
        self.runs = list(runs)
        self.evaluations_begun = False
        self._caught: dict[int, Any] = {}

    def __enter__(self) -> RunClaim:
        for run in self.runs:
            run.refuse_taken()
        _claim_names([run.path("csv") for run in self.runs])
        self._caught = _catch_terminations()
        return self

    def evaluate(self, measure: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """Wrap a run's evaluation so that its first call marks the end of the window."""

        def evaluate(term: Any) -> Any:
            if not self.evaluations_begun:
                self.evaluations_begun = True
                # from the first evaluation on, a signal terminates the run as it always did
                _restore(self._caught)
            return measure(term)

        return evaluate

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        try:
            if exc is not None and not self.evaluations_begun:
                _give_back(self.runs)
        finally:
            _restore(self._caught)
        if isinstance(exc, _Terminated):
            # terminated by the signal after all, as without the handler
            os.kill(os.getpid(), exc.signum)


def _claim_names(csv_paths: Sequence[str]) -> None:
    """Create each run's configuration exclusively, a placeholder until the driver writes it.

    Raises:
        FileExistsError: If another run holds one of them.
        OSError: If a placeholder cannot be written.  Either way, those this call took are given
            back.
    """
    claimed: list[str] = []
    try:
        for path in csv_paths:
            config = metadata_path_for(path)
            with open(config, "x") as handle:
                # taken once created, before the placeholder is written: a write that fails
                # leaves the file, and it is given back with the others
                claimed.append(config)
                handle.write(_PLACEHOLDER)
    except BaseException:
        for config in claimed:
            with contextlib.suppress(FileNotFoundError):
                os.remove(config)
        raise


def _give_back(runs: Sequence[RunArtifacts]) -> None:
    """Remove the files a run wrote before its first evaluation; see :class:`RunClaim`."""
    for run in runs:
        for path in run.paths().values():
            if path != run.path("csv") and os.path.exists(metadata_path_for(path)):
                continue
            if os.path.isdir(path):
                with contextlib.suppress(OSError):
                    os.rmdir(path)
            else:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(path)


def _catch_terminations() -> dict[int, Any]:
    """Turn a termination signal into :class:`_Terminated`; returns the handlers it replaced."""
    if threading.current_thread() is not threading.main_thread():
        return {}
    replaced: dict[int, Any] = {}
    for signum in _TERMINATIONS:
        previous = signal.getsignal(signum)
        if previous is signal.SIG_DFL or callable(previous):
            replaced[signum] = signal.signal(signum, _terminate)
    return replaced


def _restore(replaced: dict[int, Any]) -> None:
    """Put the replaced handlers back, once."""
    for signum, handler in replaced.items():
        signal.signal(signum, handler)
    replaced.clear()
