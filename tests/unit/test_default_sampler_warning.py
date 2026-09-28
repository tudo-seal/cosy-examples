"""A loop that draws from a sampler nobody chose says so, once per run.

Given a search space and no ``sampler=``, the loop builds its own:
``SizeUniformSampler(DEFAULT_SIZE_BOUND, Random(seed))``, terms of at most a hundred symbols, counted
over the derivation tree cosy's default construction materializes.  That suits the small spaces the
tests search and nothing larger, and the loop built it without a word.  Now it says so where it
builds it, which is once per run: ``reset()`` drops the sampler, and the next run builds it again.

The tests run on the three-level expression space, whose 144 terms all have size 5 to 7.  On the
list space the other tests use, the default's first draw had not come after a minute when this
module was written, where their bound of 6 draws in hundredths of a second: lists of up to a hundred
symbols over three letters, counted over a tree the sampler builds, are what the warning is for.
"""

from __future__ import annotations

import logging
import random

import pytest
from cosy.core import Synthesizer
from cosy.search import SizeUniformSampler

from bayesian_optimization import BayesianOptimization, RandomSearch
from bayesian_optimization.loop import DEFAULT_SIZE_BOUND
from bayesian_optimization.runs import build_acquisition_optimizer
from tests.integration.test_real_ea_loop import _MAX_SIZE, _TARGET, _objective, _repository

PHRASE = "no sampler was given"

# Phrases other tests count warnings by: a message carrying one would be counted by them too.
COUNTED_ELSEWHERE = (
    "fallback", "dropped_suggestion", "declares no hyperparameters", "theta has",
    "its smallest possible value",
)


def _said(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if PHRASE in record.getMessage()]


def _space():
    return Synthesizer(_repository(), {}).construct_solution_space(_TARGET).prune()


def _sampler() -> SizeUniformSampler:
    return SizeUniformSampler(_MAX_SIZE, random.Random(3))


def _random(**kwargs) -> RandomSearch:
    return RandomSearch(_space(), _TARGET, seed=3, **kwargs)


def _bayesian(**kwargs) -> BayesianOptimization:
    return BayesianOptimization(
        _space(), _TARGET, seed=3,
        optimizer=build_acquisition_optimizer(population_size=6, generations=2, depth_bound=4,
                                              seed=3),
        **kwargs,
    )


def _run(loop, n_design: int, n_passes: int) -> None:
    loop.initialize(initial_size=n_design)
    for _ in range(n_design + n_passes):
        suggestion = loop.suggest()
        loop.observe(suggestion.candidate, _objective(suggestion.candidate))


def test_a_bayesian_loop_without_a_sampler_says_once_what_it_draws_from(caplog):
    with caplog.at_level(logging.WARNING, logger="bayesian_optimization"):
        _run(_bayesian(), n_design=3, n_passes=2)

    said = _said(caplog)
    assert len(said) == 1, f"once per run, not {len(said)} times"
    message = said[0].getMessage()
    assert "SizeUniformSampler" in message and f"{DEFAULT_SIZE_BOUND}" in message
    assert "sampler=" in message, "the message names the way out"
    assert said[0].name == "bayesian_optimization.bo", "under the loop's own logger"
    assert not [phrase for phrase in COUNTED_ELSEWHERE if phrase in message]


def test_a_random_search_without_a_sampler_says_it_once_per_run(caplog):
    with caplog.at_level(logging.WARNING, logger="bayesian_optimization"):
        _run(_random(), n_design=2, n_passes=4)

    said = _said(caplog)
    assert len(said) == 1, f"once per run, not {len(said)} times"
    assert said[0].name == "bayesian_optimization.random_search"


def test_a_second_run_is_told_again(caplog):
    loop = _random()
    with caplog.at_level(logging.WARNING, logger="bayesian_optimization"):
        _run(loop, n_design=2, n_passes=1)
        loop.reset()
        _run(loop, n_design=2, n_passes=1)

    assert len(_said(caplog)) == 2


def test_a_loop_given_its_sampler_or_no_space_says_nothing(caplog, bo_factory, tree_corpus):
    with caplog.at_level(logging.WARNING, logger="bayesian_optimization"):
        _run(_bayesian(sampler=_sampler()), n_design=3, n_passes=2)
        _run(_random(sampler=_sampler()), n_design=2, n_passes=2)
        # no space: nothing to draw from, so nothing is built
        without_space = bo_factory()
        without_space.initialize(x0=tree_corpus[:3], y0=[1.0, 2.0, 0.5])
        without_space.suggest()

    assert _said(caplog) == []
