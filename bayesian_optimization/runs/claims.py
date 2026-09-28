"""A run's names, taken where it first writes and given back if it fails before measuring anything."""

from __future__ import annotations

import contextlib
import os
import signal
import threading
from collections.abc import Callable, Iterator, Sequence
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
    and SIGHUP, which ``tmux kill-session`` sends -- are caught in that window only, from before
    the names are taken, where the main thread can catch them and they would terminate the process:
    a signal the process ignores stays ignored, and one the caller handles stays the caller's.  The
    files go back with the signals held, so that a second one cannot cut the give-back short, and
    the process is then terminated by the signal.  From the first evaluation on nothing is removed:
    a training that gives out keeps its files as evidence, and its names.  The handlers are the
    caller's again from the first evaluation, or, where it runs in another thread, from the end of
    the claim; a signal in between still terminates the run, its files kept.  A claim does not
    nest: an inner one's first evaluation does not end an outer one's window.

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
        # Caught before the names are taken, so that a signal between the two leaves nothing.
        self._caught = _catch_terminations()
        taken: list[str] = []
        try:
            _claim_names([run.path("csv") for run in self.runs], taken)
        except BaseException as failure:
            # Only the configurations this claim created go back: one another run holds, which
            # refused this claim, is that run's.
            with _signals_held():
                for config in taken:
                    with contextlib.suppress(FileNotFoundError):
                        os.remove(config)
                _restore(self._caught)
            if isinstance(failure, _Terminated):
                os.kill(os.getpid(), failure.signum)
            raise
        return self

    def evaluate(self, measure: Callable[[Any], Any]) -> Callable[[Any], Any]:
        """Wrap a run's evaluation so that its first call marks the end of the window.

        An evaluation of rounds, ``run_search``'s ``evaluate_many``, is wrapped the same way: its
        first call, the first round asked, ends the window.
        """

        def evaluate(term: Any) -> Any:
            if not self.evaluations_begun:
                self.evaluations_begun = True
                # From the first evaluation on, a signal terminates the run as it always did; only
                # the main thread can put a handler back, so another one leaves it to the end.
                if threading.current_thread() is threading.main_thread():
                    _restore(self._caught)
            return measure(term)

        return evaluate

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> None:
        # A second signal waits until the files are back and the handlers the caller's, and then
        # terminates as it would have.
        with _signals_held():
            try:
                if exc is not None and not self.evaluations_begun:
                    _give_back(self.runs)
            finally:
                _restore(self._caught)
        if isinstance(exc, _Terminated):
            # terminated by the signal after all, as without the handler
            os.kill(os.getpid(), exc.signum)


def _claim_names(csv_paths: Sequence[str], taken: list[str]) -> None:
    """Create each run's configuration exclusively, a placeholder until the driver writes it.

    Args:
        csv_paths (Sequence[str]): The runs' CSVs.
        taken (list[str]): Filled with every configuration this call creates, as it creates it,
            for :meth:`RunClaim.__enter__` to give back on a failure.

    Raises:
        FileExistsError: If another run holds one of them.
        OSError: If a placeholder cannot be written.
    """
    for path in csv_paths:
        config = metadata_path_for(path)
        with open(config, "x") as handle:
            # taken once created, before the placeholder is written: a write that fails leaves
            # the file, and it is given back with the others
            taken.append(config)
            handle.write(_PLACEHOLDER)


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
    """Turn a termination signal into :class:`_Terminated` where it would terminate the process.

    Only where Python can install a handler, the main thread, and only over the default action: a
    signal the process ignores stays ignored, and one the caller handles stays the caller's.

    Returns:
        dict[int, Any]: The handlers it replaced.
    """
    if threading.current_thread() is not threading.main_thread():
        return {}
    replaced: dict[int, Any] = {}
    for signum in _TERMINATIONS:
        if signal.getsignal(signum) is signal.SIG_DFL:
            replaced[signum] = signal.signal(signum, _terminate)
    return replaced


@contextlib.contextmanager
def _signals_held() -> Iterator[None]:
    """Hold the termination signals back, where the platform and the thread can; they arrive after."""
    if not hasattr(signal, "pthread_sigmask") or (
        threading.current_thread() is not threading.main_thread()
    ):
        yield
        return
    previous = signal.pthread_sigmask(signal.SIG_BLOCK, _TERMINATIONS)
    try:
        yield
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous)


def _restore(replaced: dict[int, Any]) -> None:
    """Put the replaced handlers back, once."""
    for signum, handler in replaced.items():
        signal.signal(signum, handler)
    replaced.clear()
