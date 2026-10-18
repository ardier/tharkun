from __future__ import annotations

import os
import sys

import pytest

from tharkun.avengers import IsolatedExecutor, MethodCall, ObjectGoal, synthesize_object
from tharkun.avengers.backends import (
    GhostwriterPass,
    HypothesisFromTypePass,
    PynguinBackendPass,
    _project_and_module,
    backend_passes,
    backend_status,
    ghostwriter_available,
    hypothesis_available,
    pynguin_available,
)
from tharkun.avengers.engine import SaturationBudget
from tharkun.avengers.model import SourceArtifact
from tharkun.avengers.passes import (
    ConstructorTemplatePass,
    PynguinAstPass,
    ReplayPass,
    SetupSequencePass,
    GeneratedSourcePass,
)
from tharkun.knowledge import KnowledgeBase


FIXTURE = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "fixtures", "avengers_target.py")
)


def _mine(executor):
    return [
        ConstructorTemplatePass(),
        GeneratedSourcePass(),
        SetupSequencePass(max_steps=0),
        ReplayPass(executor),
    ]


# -- the shared test-source ingester -----------------------------------------


def test_the_old_pynguin_pass_name_still_resolves():
    # A renamed public name stays a permanent alias in this package.
    assert PynguinAstPass is GeneratedSourcePass


def test_the_ingester_reads_every_test_writing_tool_not_only_pynguin():
    source = """
import target as module_0


def test_case_0():
    timer_like_0 = module_0.TimerLike()
    timer_like_0.start()
"""
    goal = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("stop"),))
    for tool in ("pynguin", "ghostwriter", "codamosa", "PYNGUIN"):
        kb = KnowledgeBase()
        kb.publish("object-goal", goal.key, goal, producer="initial")
        artifact = SourceArtifact(tool, source)
        kb.publish("source-artifact", artifact.key, artifact, producer="initial")
        emitted = list(GeneratedSourcePass().run(kb, ()))
        assert emitted, f"{tool} source should yield a candidate"
        assert any(item.value.setup == (MethodCall("start"),) for item in emitted)


def test_an_unknown_tools_source_is_ignored():
    goal = ObjectGoal(FIXTURE, "TimerLike")
    kb = KnowledgeBase()
    kb.publish("object-goal", goal.key, goal, producer="initial")
    artifact = SourceArtifact("some-llm", "x = 1\n")
    kb.publish("source-artifact", artifact.key, artifact, producer="initial")
    assert list(GeneratedSourcePass().run(kb, ())) == []


# -- availability ------------------------------------------------------------


def test_backend_status_names_every_backend_and_says_why_not():
    status = backend_status()
    assert set(status) == {
        "hypothesis",
        "ghostwriter",
        "pynguin",
        "flatting",
    }
    for name, value in status.items():
        assert value == "available" or len(value) > len("available")




def test_unavailable_backends_are_skipped_rather_than_failing():
    built = backend_passes(
        ["pynguin", "ghostwriter", "hypothesis-from-type", "nonsense"],
        executor=IsolatedExecutor(),
        interpreter="/definitely/not/an/interpreter",
    )
    assert built == ()


# -- project/module splitting ------------------------------------------------


def test_a_module_inside_a_package_is_named_by_its_dotted_path():
    project, module = _project_and_module(FIXTURE)
    assert module == "tests.fixtures.avengers_target"
    assert os.path.isdir(os.path.join(project, "tests", "fixtures"))


def test_a_standalone_module_is_named_by_its_stem(tmp_path):
    module = tmp_path / "loose.py"
    module.write_text("class A:\n    pass\n")
    project, name = _project_and_module(str(module))
    assert name == "loose"
    assert os.path.realpath(project) == os.path.realpath(str(tmp_path))


def test_something_that_is_not_a_python_file_has_no_split(tmp_path):
    other = tmp_path / "notes.txt"
    other.write_text("x")
    assert _project_and_module(str(other)) is None
    assert _project_and_module(str(tmp_path / "missing.py")) is None













# -- the live backends -------------------------------------------------------


@pytest.mark.skipif(not hypothesis_available(), reason="hypothesis is not installed")
def test_hypothesis_from_type_builds_a_validated_object():
    executor = IsolatedExecutor(timeout_s=30)
    report = synthesize_object(
        ObjectGoal(FIXTURE, "Box", (MethodCall("width"),)),
        passes=[HypothesisFromTypePass(executor, max_examples=100), *_mine(executor)],
        executor=executor,
        budget=SaturationBudget(max_seconds=60),
    )
    assert report.best is not None
    values = report.best.recipe.constructor.keyword_arguments
    assert values["lower"] < values["upper"]


@pytest.mark.skipif(not ghostwriter_available(), reason="ghostwriter needs hypothesis and black")
def test_ghostwriter_produces_source_for_the_target():
    executor = IsolatedExecutor(timeout_s=30)
    kb = KnowledgeBase()
    goal = ObjectGoal(FIXTURE, "Box")
    kb.publish("object-goal", goal.key, goal, producer="initial")
    _cursor, delta = kb.evidence_since(0, kinds=frozenset({"object-goal"}))
    emitted = list(GhostwriterPass(executor).run(kb, delta))
    assert emitted
    artifact = emitted[0].value
    assert artifact.tool == "ghostwriter"
    assert "Box" in artifact.source




# Pynguin is a randomized, time-boxed search: the gate asserts the wiring with a fixed generator, and only what holds for ANY
# outcome of a real search (docs/TESTING.md, "Randomized and wall-clock-boxed checks").  How often the real search yields a
# mineable object is a measured metric, scripts/pynguin_seed_hit_rate.py.
def test_a_backends_source_mines_into_a_validated_object_and_is_credited_to_it(monkeypatch):
    def generate(self, goal, *, seed=0):
        return (SourceArtifact("pynguin", "import target as module_0\n\n\ndef test_case_0():\n"
                                          "    box_0 = module_0.Box(1, 5)\n    box_0.width()\n"),)

    monkeypatch.setattr(PynguinBackendPass, "generate", generate)
    executor = IsolatedExecutor(timeout_s=30)
    report = synthesize_object(
        ObjectGoal(FIXTURE, "Box", (MethodCall("width"),)),
        passes=[
            PynguinBackendPass(search_seconds=15, timeout_s=180),
            *_mine(executor),
        ],
        executor=executor,
        budget=SaturationBudget(max_seconds=120),
    )
    assert report.best is not None
    assert report.metrics["pynguin"].novel > 0


@pytest.mark.skipif(not pynguin_available(), reason="pynguin is not installed")
def test_whatever_a_real_pynguin_run_writes_the_pipeline_holds_its_invariants():
    """Holds for ANY outcome of the search (even none): every validated object replays as a success, and a report exists."""
    executor = IsolatedExecutor(timeout_s=30)
    report = synthesize_object(
        ObjectGoal(FIXTURE, "Box", (MethodCall("width"),)),
        passes=[
            PynguinBackendPass(search_seconds=5, timeout_s=120),
            *_mine(executor),
        ],
        executor=executor,
        budget=SaturationBudget(max_seconds=120),
    )
    assert report.metrics["pynguin"].activations == 1          # the pass was dispatched (what a search finds is the nightly metric)
    assert all(item.outcome.succeeded for item in report.validated)
    assert report.best is None or report.best in report.validated


@pytest.mark.skipif(not hypothesis_available(), reason="hypothesis is not installed")
def test_backends_are_reachable_through_the_public_api():
    report = synthesize_object(
        ObjectGoal(FIXTURE, "Box", (MethodCall("width"),)),
        backends=("hypothesis-from-type",),
        max_setup_steps=0,
        hypothesis_examples=60,
        budget=SaturationBudget(max_seconds=60),
    )
    assert report.best is not None
    assert "hypothesis-from-type" in report.metrics
