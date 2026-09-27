"""Report a step of a run that outruns its expected duration, while it is still running."""

import _thread
import contextlib
import threading
import time
from dataclasses import dataclass


@contextlib.contextmanager
def step_budget(label, expected_seconds, hard_limit_seconds=None):
    """Report a step that outruns its expected duration, while it is still running.

    A run of these experiments is long by design, since a single pass trains a network, so "it is
    still going" and "it is stuck" look alike from outside, and the difference only shows up hours
    later in a log that ends mid-sentence.  This makes the difference observable: a watchdog thread
    prints once the step passes the duration it was expected to take, and again at every doubling
    of that, so a step that runs 60x its budget says so 6 times instead of 60.

    It reports and it does not kill.  Aborting a step that trains a network for 15 minutes because
    it took 20 would destroy more than it saves.  ``hard_limit_seconds`` exists for the steps where
    hanging is a known failure mode rather than a slow day, such as sampling on a space whose draw
    cost has a heavy tail.  It raises ``KeyboardInterrupt`` in the main thread, so the surrounding
    ``with`` blocks still close their files.

    Args:
        label (str): What is being timed, as it should read in the log.
        expected_seconds (float): The duration beyond which the step is worth reporting.  Not a
            limit: exceeding it is normal on a slower machine, which is why the message says what
            was expected instead of claiming a failure.
        hard_limit_seconds (float | None): Abort the run past this duration, or None to only
            report. (Default value = None)

    Yields:
        None: The context in which the step runs.
    """
    if expected_seconds <= 0:
        # A threshold of zero makes the wait below return at once, every time, so the watchdog
        # prints as fast as the interpreter allows and the step it was watching gets no processor.
        # A caller computing this as a per-item budget times a count reaches zero the moment the
        # count is zero, so it is worth catching rather than documenting.
        msg = f"the expected duration of {label!r} must be positive, got {expected_seconds}"
        raise ValueError(msg)

    started = time.time()
    finished = threading.Event()

    def watch():
        """Report at the expected duration and at every doubling of it."""
        threshold = float(expected_seconds)
        while True:
            # The hard limit is its own deadline, not something checked when a report happens to
            # be due.  Waking only at the reporting thresholds made a limit of 3600 s fire at 4800,
            # the first doubling past it, and a limit below the first threshold never fire at all.
            deadlines = [threshold]
            if hard_limit_seconds is not None:
                deadlines.append(float(hard_limit_seconds))
            wake_at = min(d for d in deadlines if d > time.time() - started) \
                if any(d > time.time() - started for d in deadlines) else 0.0
            if finished.wait(timeout=max(wake_at - (time.time() - started), 0.0)):
                return
            elapsed = time.time() - started
            if hard_limit_seconds is not None and elapsed >= hard_limit_seconds:
                print(
                    f"[watchdog] {label}: past the hard limit of {hard_limit_seconds:.0f}s, "
                    "interrupting the run",
                    flush=True,
                )
                _thread.interrupt_main()
                return
            if elapsed < threshold:
                continue
            print(
                f"[watchdog] {label}: running for {elapsed:.0f}s, expected about "
                f"{expected_seconds:.0f}s",
                flush=True,
            )
            threshold *= 2

    watcher = threading.Thread(target=watch, name=f"watchdog:{label}", daemon=True)
    watcher.start()
    try:
        yield
    finally:
        finished.set()
        elapsed = time.time() - started
        if elapsed > expected_seconds:
            print(
                f"[watchdog] {label}: finished after {elapsed:.0f}s, "
                f"{elapsed / max(expected_seconds, 1e-9):.1f}x the expected duration",
                flush=True,
            )


@dataclass(frozen=True)
class StepBudgets:
    """How long each step of a run is expected to take, in seconds, and when one is given up on.

    The watchdog warns past a budget and does not stop the run, except for the acquisition's hard
    limit.  The defaults are the CIFAR example's, sized for fifteen-minute trainings; a run whose
    evaluations take seconds, or hours, sets its own.

    Attributes:
        space_construction (float): Building the search space. (Default value = 300)
        determinization (float): Determinizing it for the counting sampler. (Default value = 900)
        per_evaluation (float): One evaluation of the objective. (Default value = 900)
        acquisition_warn (float): One acquisition maximization, warned past. (Default value = 600)
        acquisition_hard_limit (float): One acquisition maximization, given up on.
            (Default value = 3600)
    """

    space_construction: float = 300
    determinization: float = 900
    per_evaluation: float = 900
    acquisition_warn: float = 600
    acquisition_hard_limit: float = 3600
