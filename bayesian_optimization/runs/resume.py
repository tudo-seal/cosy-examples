"""Continue an interrupted run from the initial design its term pool already holds."""

from bayesian_optimization.runs.term_pool import read_term_pool


def check_resumed_design(resume_design, drawn_prefix):
    """Refuse a resumed design whose terms are not the ones this run would have drawn.

    The resumed values describe those terms.  If the sampler now produces different ones, after a
    changed seed or a changed cell or with a sampler whose order is not reproducible, then pairing
    value ``i`` with term ``i`` pairs a measurement with a network it was not taken from, and every
    number after that is about nothing.  With a fixed seed the two streams agree, so this normally
    passes.  It exists for when it does not.

    Args:
        resume_design (list[tuple]): ``(term, metrics)`` as loaded.
        drawn_prefix (list): The terms this run drew.

    Raises:
        ValueError: On a different length, or on the first term that differs.
    """
    if len(resume_design) != len(drawn_prefix):
        msg = (
            f"the resumed design holds {len(resume_design)} terms but this run draws "
            f"{len(drawn_prefix)}; the size of the initial design must match the run being "
            f"continued"
        )
        raise ValueError(msg)
    for index, ((resumed, _metrics), drawn) in enumerate(zip(resume_design, drawn_prefix, strict=True)):
        if resumed != drawn:
            msg = (
                f"resumed term {index} is not the term this run drew at that position; the "
                f"sampler is producing a different stream, so the loaded values describe other "
                f"networks than the ones being paired with them"
            )
            raise ValueError(msg)


# The name the CIFAR driver and its tests import.
_check_resumed_design = check_resumed_design


def load_initial_design(path, expected, phase="pre_sample"):
    """Take a finished initial design out of an interrupted run's term pool.

    A run that dies after its initial design has spent hours of accelerator time on trainings whose
    results are complete and on disk, and restarting it repeats every one of them to arrive at the
    same numbers.  The terms and their metrics are in ``<run>_terms.pickle``, and this reads them
    back so that the next run can start where the last one got to.

    The provenance is checked, not assumed.  A design measured under different epochs, a different
    cell or a different number of repetitions is not this run's design, and silently conditioning a
    surrogate on it would produce a run whose dataset nobody can describe.  The fields compared are
    the ones that change what a number means, and everything else may differ.

    Args:
        path (str): The ``<run>_terms.pickle`` of the interrupted run.
        expected (dict): The fields the current run requires to match.
        phase (str): Which phase's records to take. (Default value = "pre_sample")

    Returns:
        list[tuple]: ``(term, metrics)`` in the order they were measured.

    Raises:
        ValueError: If the pool carries no matching provenance, or none of the requested phase.
    """
    return [(record.term, record.metrics) for record in load_design_records(path, expected, phase)]


def load_design_records(path, expected, phase="pre_sample"):
    """Load the design records of an earlier run, checked against this run's configuration.

    The records themselves rather than ``(term, metrics)`` pairs: a record carries the value the
    loop was handed, which :func:`~bayesian_optimization.runs.driver.run_search` checks against
    the resuming run's objective when it resumes the design (``resume=``).  The provenance check is
    :func:`load_initial_design`'s, which returns the pairs of these records.

    Args:
        path (str): The ``<run>_terms.pickle`` of the earlier run.
        expected (dict): The fields the current run requires to match.
        phase (str): Which phase's records to take. (Default value = "pre_sample")

    Returns:
        list[TermRecord]: The records of that phase, in the order they were measured.

    Raises:
        ValueError: If the pool carries no matching provenance, or none of the requested phase.
    """
    header, records = read_term_pool(path)
    provenance = header.get("provenance") or {}
    mismatched = {
        key: (provenance.get(key), value)
        for key, value in expected.items()
        if provenance.get(key) != value
    }
    if mismatched:
        detail = ", ".join(f"{k}: pool has {p!r}, run wants {w!r}" for k, (p, w) in mismatched.items())
        msg = (
            f"{path} was measured under a different configuration and its values do not describe "
            f"this run's candidates ({detail})"
        )
        raise ValueError(msg)

    design = [record for record in records if record.phase == phase]
    if not design:
        phases = sorted({record.phase for record in records})
        msg = f"{path} holds no {phase!r} records; it has {phases}"
        raise ValueError(msg)
    return design


