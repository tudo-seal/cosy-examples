"""The readings of a CNN term that build no network are importable without torch.

The algebras of the CNN example used to sit in one module, and importing any one of them therefore
imported torch. Most of them do not need it. A term read as a string, as an edge list, as a coarser
term, as a request type or as an operator histogram is built from numpy and the cosy term type
alone, and only the reading that assembles a ``torch.nn.Module`` needs the library.

The two modules are therefore ``cnn_damg_term_algebras`` and ``cnn_damg_network_algebras``, and the
first one is meant to stay free of the second. Nothing enforces that on its own. An import added to
the term module for one convenience puts torch back on the path of every consumer of the graph
readings, and no other test in this repository would notice, because they all run in a process that
loads torch for some other reason anyway.

So the first test below runs a child interpreter in which torch cannot be imported at all, and
imports the term module there. The second one imports the network module in the same child and
requires it to fail, since a probe that blocks nothing would let the first test pass on any module.
The third holds what each half contains, so that a definition cannot quietly move between them or
be dropped in a rewrite.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import bayesian_optimization
from bayesian_optimization.examples.cnn_damg_nas import cnn_damg_term_algebras

PACKAGE = "bayesian_optimization.examples.cnn_damg_nas"

#: The directory the package this process imported lives in, handed to the child so that it reads
#: the same files rather than whatever its working directory happens to offer.
PACKAGE_PATH = str(Path(bayesian_optimization.__file__).resolve().parents[1])

#: The directory both halves live in.
ALGEBRA_DIR = Path(cnn_damg_term_algebras.__file__).resolve().parent

#: What the term half holds. Every name is a reading of a term that produces a description of it,
#: and none of them names a torch symbol.
TERM_DEFINITIONS = {
    "pretty_term_algebra",
    "edgelist_learner",
    "_beside_edgelists",
    "_before_edgelists",
    "edgelist_algebra",
    "hierarchy_algebra",
    "request_algebra",
    "refinement_1_algebra",
    "refinement_2_algebra",
    "operator_histogram",
    "_HISTOGRAM_OPERATORS",
    "_one_hot",
    "operator_histogram_algebra",
}

#: What the network half holds: the module each combinator stands for, the training loop, the
#: protocol that loop follows, and the three algebras that assemble a network from a term.
NETWORK_DEFINITIONS = {
    "EdgesModule",
    "_as_tensor_input",
    "_feature_dim",
    "SwapModule",
    "SynthLinear",
    "SynthConv2d",
    "SynthMaxPool2d",
    "SynthBatchNorm2d",
    "SynthSigmoid",
    "SynthReLU",
    "SynthTanh",
    "SumModule",
    "ProductModule",
    "BesideModule",
    "BeforeModule",
    "CopyModule",
    "_DEFAULT_BATCH_SIZE",
    "TrainingProtocol",
    "P50",
    "P50_CORRECTED_CIFAR",
    "P50_CORRECTED_DIGITS",
    "apply_he_initialisation",
    "_augment",
    "learner",
    "pytorch_function_algebra",
    "pytorch_model_algebra",
    "pytorch_components_algebra",
}

#: The child program. The finder raises rather than returning None, so an import of torch or of any
#: submodule of it stops there instead of falling through to the real one further down the path.
_PROBE = """
import sys


class NoTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is unimportable in this probe")
        return None


sys.meta_path.insert(0, NoTorch())
import {module}
print("imported, torch loaded:", "torch" in sys.modules)
"""


def _import_under_a_torch_ban(module):
    """Import one module in a child interpreter that cannot import torch.

    Args:
        module (str): The dotted name to import.

    Returns:
        subprocess.CompletedProcess: The finished child, with its output captured as text.
    """
    entries = [os.environ.get("PYTHONPATH", ""), PACKAGE_PATH]
    environment = {**os.environ,
                   "PYTHONPATH": os.pathsep.join(entry for entry in entries if entry)}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PROBE.format(module=module))],
        capture_output=True, text=True, env=environment, check=False,
    )


def _top_level_definitions(path):
    """The names a module binds at the top level, read off its source rather than by importing it.

    Args:
        path (pathlib.Path): The file to read.

    Returns:
        set: One name per function, class and plain assignment at the top level.
    """
    names = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(ast.unparse(target) for target in node.targets)
    return names


def test_the_term_algebras_import_with_torch_made_unimportable():
    """The point of the split: a consumer of the graph readings does not pay for torch."""
    finished = _import_under_a_torch_ban(f"{PACKAGE}.cnn_damg_term_algebras")
    assert finished.returncode == 0, finished.stderr
    assert "imported, torch loaded: False" in finished.stdout


def test_the_torch_ban_of_that_probe_really_bans_torch():
    """Without this, the test above would pass for a module that imports torch on every line."""
    finished = _import_under_a_torch_ban(f"{PACKAGE}.cnn_damg_network_algebras")
    assert finished.returncode != 0
    assert "torch is unimportable in this probe" in finished.stderr


def test_each_definition_sits_in_exactly_one_half():
    """The two halves cover the algebras of the example once each, and share nothing."""
    term = _top_level_definitions(ALGEBRA_DIR / "cnn_damg_term_algebras.py")
    network = _top_level_definitions(ALGEBRA_DIR / "cnn_damg_network_algebras.py")

    assert term == TERM_DEFINITIONS
    assert network == NETWORK_DEFINITIONS
    assert term & network == set()
