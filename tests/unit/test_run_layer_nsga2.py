"""NSGA-II through the run layer: its passes recorded under their phase, its populations written as they form.

``run_search`` runs NSGA-II as it runs every strategy of the loop.  What NSGA-II adds is a file:
one JSON line per population it forms, written before the offspring bred from that population is
evaluated, so that a long run says how its search moved while it is still running.  A member is
named by the phase and the index its evaluation was recorded under, which joins the line to the
CSV and the term pool.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random

import pytest
from cosy.search import SizeUniformSampler

from bayesian_optimization import NSGA2, RandomSearch
from bayesian_optimization.runs import MetricSchema, Objective, RunArtifacts, read_term_pool
from bayesian_optimization.runs.driver import DESIGN_PHASE, run_paired, run_search
from tests.spaces import LIST, list_space

SCHEMA = MetricSchema.for_metrics(Objective("score"), ["score", "length"])


def _metrics(term):
    """A stand-in evaluation: a score spread over [0, 1), read off the list the term renders."""
    text = term.interpret({})
    digest = hashlib.sha256(text.encode()).digest()
    return {"score": int.from_bytes(digest[:6], "big") / 2**48, "length": len(text)}


def _cost(term):
    return float(term.size)


def _quiet(line):
    """An echo that prints nothing."""


def _nsga2(seed):
    return NSGA2(
        list_space(), LIST, cost=_cost, seed=seed, sampler=SizeUniformSampler(6, random.Random(seed)),
        depth_bound=5, crossover_rate=0.5, mutation_rate=0.3,
    )


def _rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _lines(path):
    if not os.path.exists(path):
        return []
    with open(path) as handle:
        return [json.loads(line) for line in handle]


def _terms_by_label(path):
    _header, records = read_term_pool(RunArtifacts(str(path)).path("terms"))
    return {(record.phase, record.index): record.term for record in records}


def test_every_population_is_on_disk_before_an_offspring_bred_from_it_is_evaluated(tmp_path):
    path = tmp_path / "run.csv"
    generations = RunArtifacts(str(path)).path("generations")
    strategy = _nsga2(3)
    seen = []

    def evaluate(term):
        seen.append((len(_lines(generations)), len(strategy.formed_generations)))
        return _metrics(term)

    outcome = run_search(strategy, evaluate, schema=SCHEMA, csv_path=str(path), pretty_algebra=dict,
                         n_design=5, n_passes=15, echo=_quiet)

    rows = _rows(path)
    assert [row["phase"] for row in rows] == [DESIGN_PHASE] * 5 + ["nsga2_offspring"] * 15
    assert outcome.summary["run_kind"] == "nsga2"
    assert outcome.summary["generations_formed"] == len(strategy.formed_generations)
    assert outcome.summary["offspring_known"] == strategy.offspring_known
    # The design is evaluated before any population exists; every pass after the populations formed so far.
    assert seen[:5] == [(0, 0)] * 5
    assert all(written == formed for written, formed in seen[5:])
    assert len(strategy.formed_generations) > 2, "a run of one or two populations tests no log"

    lines = _lines(generations)
    assert [line["generation"] for line in lines] == list(range(len(strategy.formed_generations)))
    terms = _terms_by_label(path)
    for line, formed in zip(lines, strategy.formed_generations, strict=True):
        assert [terms[(phase, index)] for phase, index in line["members"]] == list(formed.members)
        assert line["ranks"] == list(formed.ranks)
        assert (line["offspring"], line["evaluated"], line["known"]) == (formed.offspring, formed.evaluated, formed.known)
    assert lines[0]["members"] == [[DESIGN_PHASE, index] for index in range(5)]
    assert lines[0]["distances"] is None
    for line, formed in zip(lines[1:], strategy.formed_generations[1:], strict=True):
        assert line["distances"] == [d if d != float("inf") else "inf" for d in formed.distances]
        assert "inf" in line["distances"], "a front's boundary is infinite"


def test_the_generation_log_alone_refuses_a_second_start(tmp_path):
    path = tmp_path / "run.csv"
    run_search(_nsga2(4), _metrics, schema=SCHEMA, csv_path=str(path), pretty_algebra=dict,
               n_design=4, n_passes=6, echo=_quiet)
    artifacts = RunArtifacts(str(path))
    for kind, file in artifacts.paths().items():
        if kind != "generations" and os.path.exists(file):
            os.remove(file)
    assert os.path.exists(artifacts.path("generations"))
    with pytest.raises(FileExistsError, match="_generations.jsonl"):
        run_search(_nsga2(4), _metrics, schema=SCHEMA, csv_path=str(path), pretty_algebra=dict,
                   n_design=4, n_passes=6, echo=_quiet)


def test_a_run_without_passes_writes_no_generation_log(tmp_path):
    path = tmp_path / "run.csv"
    outcome = run_search(_nsga2(5), _metrics, schema=SCHEMA, csv_path=str(path), pretty_algebra=dict,
                         n_design=4, n_passes=0, echo=_quiet)
    assert outcome.summary["run_kind"] == "design_only"
    assert not os.path.exists(RunArtifacts(str(path)).path("generations"))


def test_nsga2_breeds_from_a_design_it_takes_over_in_a_paired_run(tmp_path):
    paths = {"random": str(tmp_path / "random.csv"), "nsga2": str(tmp_path / "nsga2.csv")}
    strategies = {
        "random": RandomSearch(list_space(), LIST, sampler=SizeUniformSampler(6, random.Random(6)), seed=6),
        "nsga2": _nsga2(6),
    }
    outcomes = run_paired(strategies, _metrics, schema=SCHEMA, csv_paths=paths, pretty_algebra=dict,
                          n_design=4, n_passes=8, echo=_quiet)

    design = [row for row in _rows(paths["nsga2"]) if row["phase"] == DESIGN_PHASE]
    assert all(row["taken_over"] == "True" for row in design)
    assert outcomes["nsga2"].summary["run_kind"] == "nsga2"
    lines = _lines(RunArtifacts(paths["nsga2"]).path("generations"))
    assert lines[0]["members"] == [[DESIGN_PHASE, index] for index in range(4)]
    terms = _terms_by_label(paths["nsga2"])
    assert [terms[(phase, index)] for phase, index in lines[0]["members"]] == list(
        strategies["nsga2"].formed_generations[0].members
    )
