"""The BO core must report what it actually did.

Every assertion here fails against the unfixed core, and that is the point.  The degradation at
issue is *unobservable*: when the acquisition optimizer returns an already-evaluated candidate,
``suggest()`` silently swaps in a random fallback tree and reports the fallback's acquisition value
as if it were the optimizer's result.  Nothing in the logs, nothing on stdout, nothing in the result
file says so.

``verbose=True`` does not help, because it sets a level without ever attaching a handler, so
``logging.lastResort`` handles the records at WARNING and every INFO line is dropped.  And
``log_iteration`` is only reached from the convenience wrapper, while both experiment scripts drive
the Ask/Tell path.

Until this is fixed, no later fix is provable: a run cannot show whether it did Bayesian
optimization or random search.
"""

from __future__ import annotations

import logging

from cosy.core.tree import Tree

LOGGER_NAME = "bayesian_optimization"


def reachable_handlers(logger: logging.Logger) -> list[logging.Handler]:
    """Handlers a record would actually reach, walking the chain like ``Logger.callHandlers``.

    Checking ``logger.handlers`` alone is wrong (an ancestor's handler counts too) and checking the
    root alone is wrong (``propagate=False`` cuts the chain).
    """
    found: list[logging.Handler] = []
    current: logging.Logger | None = logger
    while current is not None:
        found.extend(current.handlers)
        if not current.propagate:
            break
        current = current.parent
    return found


def _fallback_bo(monkeypatch, bo_factory, tree_corpus):
    """A BO whose optimizer returns an already-known candidate, forcing the fallback path."""
    from bayesian_optimization import bo as bo_mod

    monkeypatch.setattr(bo_mod, "_sample_fallback_tree", lambda sampler, query, seen: Tree("fresh"))

    bo = bo_factory(candidates=[tree_corpus[0]])  # tree_corpus[0] is in x0, so it is a duplicate
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    # No search space in this fixture, so nothing built a sampler, and the stub is the source.
    bo._sampler = object()
    return bo


def test_fallback_substitution_emits_a_warning(monkeypatch, bo_factory, tree_corpus, caplog):
    """Replacing the optimizer's result with a random tree is not a detail: it must be announced.

    This is the visible end of the degradation chain: once it fires every generation, the run is
    random search wearing a BO label.
    """
    bo = _fallback_bo(monkeypatch, bo_factory, tree_corpus)

    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        suggestion = bo.suggest()

    assert suggestion.diagnostics is not None
    assert suggestion.diagnostics["fallback_used"] is True

    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warnings, (
        "suggest() replaced the optimizer's candidate with a random fallback sample and said "
        "nothing about it"
    )
    assert any("fallback" in r.getMessage().lower() for r in warnings)


def test_ask_tell_suggest_reports_the_iteration(bo_factory, tree_corpus, caplog):
    """The Ask/Tell path must log too, since it is the path both experiment scripts use.

    ``log_iteration`` exists but is only called from the convenience wrapper, so today a real
    experiment run produces no BO log output at all.
    """
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        bo.suggest()

    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    assert messages, "suggest() reported nothing on the Ask/Tell path"
    assert any("fallback" in m.lower() for m in messages), (
        "the iteration line does not say whether the candidate came from the optimizer or from a "
        "fallback sample, which is the one thing it has to say"
    )


def test_verbose_makes_output_reachable(monkeypatch, bo_factory, tree_corpus):
    """``verbose=True`` promises output, so it must attach a handler, not just set a level.

    The chain is cut with ``propagate=False`` to reproduce the situation of a caller who never ran
    ``logging.basicConfig()``, which is exactly how both experiment scripts run.
    """
    logger = logging.getLogger(LOGGER_NAME)
    monkeypatch.setattr(logger, "handlers", [])
    monkeypatch.setattr(logger, "propagate", False)

    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
    bo.suggest(verbose=True)

    assert reachable_handlers(logger), (
        "verbose=True set a level but attached no handler; logging.lastResort then drops every "
        "INFO record at WARNING, so the flag cannot produce output"
    )
    assert logger.isEnabledFor(logging.INFO)


def test_log_iteration_shows_the_fallback(caplog):
    """The per-iteration line must carry the fallback state, not only timings.

    A row that reports ``acq=...`` without saying the value belongs to a random replacement is
    actively misleading.
    """
    from bayesian_optimization.diagnostics import get_logger, log_iteration
    from bayesian_optimization.state import Suggestion

    suggestion = Suggestion(
        candidate=Tree("x"),
        acquisition_value=0.25,
        diagnostics={"fallback_used": True, "fallback_attempts": 3},
    )

    logger = get_logger("test")
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_iteration(
            logger,
            iteration=7,
            suggestion=suggestion,
            y_observed=1.5,
            suggest_time=0.1,
            eval_time=0.2,
            observe_time=0.3,
        )

    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "fallback" in line.lower(), "the iteration line hides that a fallback was used"
    assert "3" in line, "the number of fallback attempts is not reported"


def test_log_iteration_does_not_claim_absence_it_has_not_checked(caplog):
    """Missing diagnostics must read as unknown, never as 'no fallback'.

    Printing ``fallback=no`` for a suggestion that carries no diagnostics would assert something the
    line never verified, the same class of silent substitute value this package rules out.
    """
    from bayesian_optimization.diagnostics import get_logger, log_iteration
    from bayesian_optimization.state import Suggestion

    suggestion = Suggestion(candidate=Tree("x"), acquisition_value=0.25, diagnostics=None)

    logger = get_logger("test")
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_iteration(
            logger,
            iteration=1,
            suggestion=suggestion,
            y_observed=1.0,
            suggest_time=0.1,
            eval_time=0.1,
            observe_time=0.1,
        )

    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "fallback=?" in line, f"unknown fallback state not marked as unknown: {line!r}"


def test_suggest_logs_the_incumbent_it_actually_has(bo_factory, tree_corpus, caplog):
    """The log line must read a field the suggestion carries, not one it used to carry.

    ``log_suggestion`` looked up ``incumbent_raw``, the name from the era when the diagnostics
    reported the incumbent twice, once per side of a y transform.  A ``dict.get`` on a key that no
    longer exists does not fail.  It prints ``?``, and the one number that says whether the run is
    getting anywhere quietly disappears from every log line.
    """
    bo = bo_factory()
    bo.initialize(x0=tree_corpus[:3], y0=[1.0, 5.0, 2.0])

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        bo.suggest()

    line = "\n".join(r.getMessage() for r in caplog.records)
    assert "incumbent=5" in line, f"the incumbent is missing from the log line: {line!r}"


def test_an_absent_posterior_coordinate_is_not_reported_as_zero(caplog):
    """A posterior coordinate the record does not carry must read as unknown, never as zero.

    ``log_suggestion`` renders ``m`` and ``s`` for the trace readings, and a suggestion assembled
    elsewhere may not have them.  A ``0.0`` there is a coordinate, and a reader cannot tell it
    from one that was measured, which is the substitute value this package refuses everywhere
    else.  It is the same distinction ``_fallback_field`` draws between "no" and "not reported".
    """
    from bayesian_optimization.diagnostics import log_suggestion
    from bayesian_optimization.state import Suggestion

    logger = logging.getLogger(LOGGER_NAME)
    suggestion = Suggestion(
        candidate=Tree("t", ()),
        acquisition_value=0.25,
        diagnostics={"iteration": 3, "fallback_used": False, "incumbent": 1.0},
    )
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        log_suggestion(logger, suggestion)

    line = caplog.text
    assert "m=?" in line
    assert "s=?" in line
    assert "m=0" not in line and "s=0" not in line
