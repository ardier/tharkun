"""Adapters that RUN the other object-construction tools, not just read them.

``PynguinAstPass`` mines recipes from Pynguin test source, but nothing in the
portfolio ever invoked Pynguin -- the source had to arrive from somewhere else.
The same was true of Ghostwriter and Hypothesis' own type
resolution.  This module closes that: each backend is a pass that produces its
own evidence.

Two contracts hold for all of them.

*Nothing a backend says is believed.*  A backend proposes a candidate or some
test source; promotion to a validated recipe still comes from the ordinary
fresh-process replay.  A tool that reports a passing test whose object cannot
actually be rebuilt contributes a rejected candidate, not a result.

*Everything is availability-gated.*  A missing tool is a missing tool, not an
error: :func:`backend_status` says which are usable and why not, so a benchmark
row can record that a configuration was degraded rather than that it lost.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from tharkun.avengers.engine import Emission, RepeatPolicy
from tharkun.avengers.executor import IsolatedExecutor
from tharkun.avengers.model import (
    ConstructionRecipe,
    ObjectGoal,
    SourceArtifact,
    constructor_from_record,
)
from tharkun.avengers.passes import CANDIDATE, GOAL, SOURCE_ARTIFACT
from tharkun.knowledge import Evidence, KnowledgeBase
from tharkun.solve.environment import subprocess_environment

# Tools whose output is Python test source the recipe miner can read.
TEST_SOURCE_TOOLS = frozenset({"pynguin", "ghostwriter", "codamosa"})


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------


def _module_available(interpreter: str, module: str) -> bool:
    try:
        completed = subprocess.run(
            [interpreter, "-c", f"import importlib.util as u; raise SystemExit(0 if u.find_spec({module!r}) else 1)"],
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0


def pynguin_available(interpreter: str | None = None) -> bool:
    return _module_available(interpreter or sys.executable, "pynguin")


def hypothesis_available(interpreter: str | None = None) -> bool:
    return _module_available(interpreter or sys.executable, "hypothesis")


def ghostwriter_available(interpreter: str | None = None) -> bool:
    """Ghostwriter needs ``black`` to format what it writes.

    ``hypothesis`` alone is not enough, and the failure without ``black`` is an
    ImportError at generation time rather than at import time -- so probing for
    hypothesis would report a backend as available that cannot produce a line.
    """

    target = interpreter or sys.executable
    return _module_available(target, "hypothesis") and _module_available(target, "black")


def backend_status(interpreter: str | None = None) -> dict[str, str]:
    """Which backends this environment can actually run.

    ``"available"`` or a short reason.  A benchmark records this beside its
    numbers: a configuration that scored nothing because its tool was missing
    did not lose, and reporting the two the same way would be dishonest.
    """

    target = interpreter or sys.executable
    status: dict[str, str] = {}
    status["hypothesis"] = (
        "available" if _module_available(target, "hypothesis") else "hypothesis is not installed"
    )
    if ghostwriter_available(target):
        status["ghostwriter"] = "available"
    elif hypothesis_available(target):
        status["ghostwriter"] = "hypothesis is installed but black, which it formats with, is not"
    else:
        status["ghostwriter"] = "hypothesis is not installed"
    status["pynguin"] = (
        "available" if pynguin_available(target) else "pynguin is not installed"
    )
    # Probed in the TARGET interpreter, not this one: flatting happens inside
    # the worker, so what the parent can import says nothing about it.
    status["flatting"] = (
        "available"
        if _module_available(target, "py_obj_flatting")
        else "py_obj_flatting is not installed"
    )
    return status


# --------------------------------------------------------------------------
# Pynguin
# --------------------------------------------------------------------------


def _project_and_module(source_path: str) -> tuple[str, str] | None:
    """Split a file into the (project root, dotted module) Pynguin wants.

    The root is walked up past every directory holding an ``__init__.py``, so a
    module inside a package is named by its full dotted path and its relative
    imports still resolve.  The walk is ``solve/loader.py::project_root_and_module``
    -- one answer, shared with every tool that imports a project module for real
    (the Eregion inverter's forward replay included).
    """

    from tharkun.solve.loader import project_root_and_module

    return project_root_and_module(source_path)


@dataclass(frozen=True)
class PynguinBackendPass:
    """Run Pynguin over the target module and offer what it writes.

    Pynguin's output is Python test source, which the existing recipe miner
    already reads, so this pass emits ``SourceArtifact`` and lets that pass do
    the extraction -- one parser for every tool that writes tests.

    Pynguin is a search, so it answers differently every time, and repeating it
    with a fresh seed is worth doing.  It is NOT repeated by default, because
    ``--maximum-search-time`` bounds the search and not the run: assertion
    generation and crash-test re-execution happen afterwards and dominated the
    wall time by 10x when measured.  ``timeout_s`` is the hard bound; a caller
    that wants repetition asks for it, having budgeted for it.
    """

    interpreter: str = sys.executable
    search_seconds: int = 20
    algorithm: str = "DYNAMOSA"
    seed: int = 0
    timeout_s: float = 300.0
    include_crash_tests: bool = True
    # Avengers mines constructors and call sequences out of the source; it never
    # reads an assertion.  Pynguin's default generator runs a whole mutation
    # analysis to produce them, which is entirely wasted here.
    assertion_generation: str = "NONE"
    # None leaves Pynguin's own choice alone. True is the isolation model of
    # Berg et al. (arXiv:2603.06107), which keeps a C-level crash in the SUT
    # from taking the generator down with it.
    subprocess_mode: bool | None = None
    name: str = "pynguin"
    consumes: frozenset[str] = frozenset({GOAL})
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def run(
        self, knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        seen = {item.key for item in knowledge.evidence(SOURCE_ARTIFACT)}
        for item in delta:
            goal = item.value
            if not isinstance(goal, ObjectGoal):
                continue
            offset = len(knowledge.evidence(SOURCE_ARTIFACT))
            for artifact in self.generate(goal, seed=self.seed + offset):
                if artifact.key in seen:
                    continue
                seen.add(artifact.key)
                yield Emission(SOURCE_ARTIFACT, artifact.key, artifact, (item.identity,))

    files: tuple[str, ...] | None = None

    def generate(self, goal: ObjectGoal, *, seed: int = 0) -> tuple[SourceArtifact, ...]:
        """Invoke Pynguin once and read back every test module it wrote."""

        split = _project_and_module(goal.source_path)
        if split is None or not pynguin_available(self.interpreter):
            return ()
        project, module = split
        with tempfile.TemporaryDirectory(prefix="tharkun-pynguin-") as output:
            # Declared inputs are staged at scratch's root, separately from generated tests.
            test_output = (Path(tempfile.mkdtemp(prefix="tests-", dir=output))
                           if self.files is not None else Path(output))
            environment = subprocess_environment()
            # Pynguin refuses to run without this: it executes the module under
            # test, and it wants the caller to have said so out loud.
            environment["PYNGUIN_DANGER_AWARE"] = "1"
            command = [
                self.interpreter,
                "-m",
                "pynguin",
                "--project-path",
                project,
                "--module-name",
                module,
                "--output-path",
                str(test_output),
                "--algorithm",
                self.algorithm,
                "--maximum-search-time",
                str(self.search_seconds),
                "--seed",
                str(seed),
                "--assertion_generation",
                self.assertion_generation,
                # Pynguin writes a statistics directory relative to its cwd,
                # which is the project under analysis -- so without this it
                # drops pynguin-report/ into whatever repository it was aimed
                # at. It goes in the temporary output directory with the tests.
                "--report-dir",
                str(Path(output) / "report"),
            ]
            if self.files is not None:
                # The confined worker refuses secondary target processes, including
                # an explicit subprocess-mode request, before Pynguin imports the target.
                if self.subprocess_mode is True:
                    return ()
                command.append("--nosubprocess")
                request = Path(output) / "request.json"
                request.write_text(json.dumps({"files": self.files, "scratch": output,
                                               "argv": command[3:]}))
                command = [self.interpreter, "-m", "tharkun.avengers._pynguin_worker", str(request)]
            elif self.subprocess_mode is not None:
                command.append("--subprocess" if self.subprocess_mode else "--nosubprocess")
            try:
                subprocess.run(
                    command,
                    capture_output=True,
                    timeout=self.timeout_s,
                    env=environment,
                    cwd=project,
                )
            except (OSError, subprocess.SubprocessError):
                return ()
            return self._collect(test_output)

    def _collect(self, output: Path) -> tuple[SourceArtifact, ...]:
        # Only the top level: the report directory below it is Pynguin's own
        # bookkeeping, not test source.
        artifacts: list[SourceArtifact] = []
        for path in sorted(output.glob("*.py")):
            # A "crash test" is Pynguin's record of a sequence that killed the
            # interpreter -- exactly the boundary behaviour this search is for,
            # so it is ingested rather than filtered out.
            if path.name.startswith("crash_test_") and not self.include_crash_tests:
                continue
            try:
                artifacts.append(SourceArtifact("pynguin", path.read_text(encoding="utf-8")))
            except OSError:
                continue
        return tuple(artifacts)


# --------------------------------------------------------------------------
# Ghostwriter
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GhostwriterPass:
    """Ask Hypothesis' Ghostwriter to write tests for the target class.

    It runs in a disposable child because reading a module's signatures means
    importing it, and the recipe miner reads its output the same way it reads
    Pynguin's.
    """

    executor: IsolatedExecutor
    name: str = "ghostwriter"
    consumes: frozenset[str] = frozenset({GOAL})
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def run(
        self, _knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        from tharkun.avengers._wire import GeneratedSource

        for item in delta:
            goal = item.value
            if not isinstance(goal, ObjectGoal):
                continue
            response = self.executor.generate_source(goal, "ghostwriter")
            if isinstance(response, GeneratedSource) and response.source.strip():
                artifact = SourceArtifact("ghostwriter", response.source)
                yield Emission(SOURCE_ARTIFACT, artifact.key, artifact, (item.identity,))


# --------------------------------------------------------------------------
# Hypothesis from_type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HypothesisFromTypePass:
    """Draw constructor arguments through Hypothesis' own type resolution.

    The portfolio's other Hypothesis pass builds strategies from a hand-written
    annotation table.  ``st.from_type`` resolves the annotation the way
    Hypothesis itself does, which reaches dataclasses, enums, ``TypedDict`` and
    anything a user registered -- shapes the table does not know.

    It runs in a child because resolving an annotation means importing the
    module that declares it.
    """

    executor: IsolatedExecutor
    max_examples: int = 50
    name: str = "hypothesis-from-type"
    consumes: frozenset[str] = frozenset({GOAL})
    repeat: RepeatPolicy = field(default_factory=lambda: RepeatPolicy.exploratory(2, 2))

    def run(
        self, _knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        from tharkun.avengers._wire import ConstructorCandidates

        for item in delta:
            goal = item.value
            if not isinstance(goal, ObjectGoal):
                continue
            response = self.executor.search_constructors(
                goal, max_examples=self.max_examples
            )
            if not isinstance(response, ConstructorCandidates):
                continue
            for record in response.records:
                try:
                    call = constructor_from_record(record)
                except ValueError:
                    continue
                recipe = ConstructionRecipe(goal.key, call)
                yield Emission(CANDIDATE, recipe.key, recipe, (item.identity,))


# --------------------------------------------------------------------------
# Portfolios
# --------------------------------------------------------------------------


def backend_passes(
    names: Iterable[str],
    *,
    executor: IsolatedExecutor,
    interpreter: str | None = None,
    search_seconds: int = 20,
    max_examples: int = 50,
) -> tuple[Any, ...]:
    """Build the requested backend passes, skipping any that cannot run."""

    target = interpreter or sys.executable
    built: list[Any] = []
    for name in names:
        if name == "pynguin" and pynguin_available(target):
            built.append(
                PynguinBackendPass(interpreter=target, search_seconds=search_seconds, files=executor.files)
            )
        elif name == "ghostwriter" and ghostwriter_available(target):
            built.append(GhostwriterPass(executor))
        elif name == "hypothesis-from-type" and hypothesis_available(target):
            built.append(HypothesisFromTypePass(executor, max_examples=max_examples))
    return tuple(built)
