"""A pool trained on one machine has to load on the machine that analyzes it.

Training and analysis are split across machines because only the training profits from a GPU. The
split holds up exactly as far as a pickle written on the one can be read on the other, and a term
is not a plain value. It carries the repository's literals, which are nested dataclasses inside
:class:`CNNrepository`, so the file depends on their module path and on their names. Rename one and
every pool recorded before the rename becomes unreadable, together with the training time that
produced it.

So the dependency is measured rather than assumed. The first test below records exactly which
classes an unpickler resolves, and the round trip runs in a fresh interpreter that never built a
search space, which is the position the analyzing machine is in.
"""

from __future__ import annotations

import io
import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path

from cosy.core.tree import Tree

import bayesian_optimization
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo import CNNrepository
from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_pool import (
    TermPoolWriter,
    read_term_pool,
)

REPOSITORY_MODULE = "bayesian_optimization.examples.cnn_damg_nas.cnn_damg_repo"

#: The directory the package this process imported lives in. The child interpreter is given it, so
#: that the child reads the file back with the same code that wrote it.
PACKAGE_PATH = str(Path(bayesian_optimization.__file__).resolve().parents[1])


def _child_environment():
    """This process's environment with the package's directory on the import path.

    Appended rather than assigned, for two reasons. Assigning would drop whatever ``PYTHONPATH``
    already points the process at, which is where cosy comes from, and the child would then resolve
    a different cosy without saying so. And passing the directory at all is what makes the child
    independent of the working directory pytest was started in.

    Returns:
        dict: The environment to run the child interpreter with.
    """
    entries = [os.environ.get("PYTHONPATH", ""), PACKAGE_PATH]
    return {**os.environ, "PYTHONPATH": os.pathsep.join(entry for entry in entries if entry)}


def _term_with_repository_literals():
    """A term shaped like a synthesized one: string combinators over repository dataclasses.

    Not produced by the synthesizer, because building a search space costs a minute and what is
    under test is the persistence of the literals rather than the search that assembles them. The
    literals are the real classes, which is what the file's portability hangs on.
    """
    return Tree(
        "learner",
        (
            Tree(CNNrepository.Conv2d(1, 8, (16, 16), (14, 14), (3, 3), 1, 0)),
            Tree(CNNrepository.ReLu()),
            Tree(CNNrepository.Linear(1568, 10)),
            Tree(CNNrepository.CrossEntropyLoss()),
            Tree(CNNrepository.Adam(0.001)),
            Tree(50),
        ),
    )


def test_a_term_pickle_pulls_in_only_the_two_modules_it_should():
    """Record what a term file depends on, so a rename that breaks old pools is visible here.

    If this test fails after a refactoring, every pool file written before it is unreadable and has
    to be retrained. The failure is the warning, not a nuisance.
    """
    resolved = set()

    class Tracking(pickle.Unpickler):
        def find_class(self, module, name):
            resolved.add(f"{module}.{name}")
            return super().find_class(module, name)

    blob = pickle.dumps(_term_with_repository_literals())
    Tracking(io.BytesIO(blob)).load()

    assert "cosy.core.tree.Tree" in resolved
    repository_classes = {entry for entry in resolved if entry.startswith(REPOSITORY_MODULE)}
    assert repository_classes == {
        f"{REPOSITORY_MODULE}.CNNrepository.Conv2d",
        f"{REPOSITORY_MODULE}.CNNrepository.ReLu",
        f"{REPOSITORY_MODULE}.CNNrepository.Linear",
        f"{REPOSITORY_MODULE}.CNNrepository.CrossEntropyLoss",
        f"{REPOSITORY_MODULE}.CNNrepository.Adam",
    }, "the literals a pool depends on changed; pools written before this change cannot be read"
    unexpected = resolved - repository_classes - {"cosy.core.tree.Tree"}
    assert not unexpected, f"a term now drags in more than the repository and cosy: {unexpected}"


def test_a_pool_written_here_loads_in_an_interpreter_that_built_no_search_space(tmp_path):
    """The hand-off between the machines, minus the network: a fresh process, only the file."""
    path = tmp_path / "pool_terms.pickle"
    term = _term_with_repository_literals()
    with TermPoolWriter(path, provenance={"dataset": "usps", "target": "L5"}) as writer:
        writer.write("pool", 0, term, {"accuracy": 0.94, "n_params": 12578})

    script = textwrap.dedent(
        f"""
        from bayesian_optimization.examples.cnn_damg_nas.cnn_damg_term_pool import read_term_pool

        header, records = read_term_pool({str(path)!r})
        (record,) = records
        print(header["provenance"]["dataset"])
        print(record.term.size)
        print(type(record.term.children[0].root).__qualname__)
        print(record.term.children[0].root.out_channels)
        print(record.metrics["accuracy"])
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=_child_environment(),
    )
    assert completed.returncode == 0, completed.stderr
    dataset, size, literal, out_channels, accuracy = completed.stdout.split()
    assert dataset == "usps"
    assert int(size) == term.size
    assert literal == "CNNrepository.Conv2d"
    assert int(out_channels) == 8
    assert float(accuracy) == 0.94


def test_the_header_says_which_dataset_and_target_the_pool_came_from(tmp_path):
    """Pools for several datasets land in one directory, and one that cannot name its own is a
    guess."""
    path = tmp_path / "pool_terms.pickle"
    with TermPoolWriter(path, provenance={"dataset": "cifar10", "target": "HEAD"}) as writer:
        writer.write("pool", 0, _term_with_repository_literals(), {"accuracy": 0.5})

    header, _records = read_term_pool(path)
    assert header["provenance"]["dataset"] == "cifar10"
    assert header["provenance"]["target"] == "HEAD"
