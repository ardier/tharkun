from __future__ import annotations

import os

import pytest

from tharkun.avengers import (
    OBJECT,
    FunctionCall,
    MethodCall,
    ObjectGoal,
    SourceArtifact,
    synthesize_object,
)
from tharkun.avengers.engine import SaturationBudget
from tharkun.avengers.executor import IsolatedExecutor
from tharkun.knowledge import KnowledgeBase


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "avengers_target.py")


def test_hypothesis_bootstraps_a_validating_box_constructor():
    goal = ObjectGoal(FIXTURE, "Box", (MethodCall("width"),))
    knowledge = KnowledgeBase()
    report = synthesize_object(
        goal,
        knowledge=knowledge,
        hypothesis_examples=40,
        max_setup_steps=0,
        budget=SaturationBudget(max_seconds=15),
    )

    assert report.best is not None
    call = report.best.recipe.constructor
    values = call.keyword_arguments
    assert values["lower"] < values["upper"]
    assert report.best.outcome.succeeded
    assert knowledge.evidence_item("constructor-template", goal.key) is not None


def test_state_producing_method_is_discovered_before_stop():
    goal = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("stop"),))
    report = synthesize_object(
        goal,
        max_setup_steps=1,
        hypothesis_examples=10,
    )
    assert report.best is not None
    assert report.best.recipe.setup == (MethodCall("start"),)
    assert "object_0.start()" in report.best.program
    assert report.best.program.endswith("object_0.stop()")


def test_multiple_boundary_methods_are_verified_in_order():
    goal = ObjectGoal(
        FIXTURE,
        "TimerLike",
        (MethodCall("stop"), MethodCall("start")),
    )
    report = synthesize_object(
        goal,
        max_setup_steps=1,
        hypothesis_examples=10,
    )
    assert report.best is not None
    assert report.best.recipe.setup == (MethodCall("start"),)
    assert report.best.outcome.return_types == ("builtins.float", "builtins.NoneType")


def test_missing_calls_optimize_method_coverage_and_return_a_state_portfolio():
    report = synthesize_object(
        ObjectGoal(FIXTURE, "TimerLike"),
        # This probe matrix includes TimerLike.hang, so the ceiling has to be
        # real -- but a replay costs ~80ms idle and ~160ms under eight xdist
        # workers, so it also has to leave room for a loaded machine.
        executor=IsolatedExecutor(timeout_s=1.5),
        max_setup_steps=1,
        hypothesis_examples=10,
    )
    assert report.best is not None
    assert report.best.recipe.setup == ()
    assert {"start", "stop"} <= set(report.covered_methods)
    assert {item.recipe.setup for item in report.objects} == {
        (),
        (MethodCall("start"),),
    }


def test_search_exhaustion_is_unresolved_not_an_impossibility_claim():
    goal = ObjectGoal(FIXTURE, "Impossible")
    report = synthesize_object(
        goal,
        max_setup_steps=0,
        hypothesis_examples=5,
        budget=SaturationBudget(max_seconds=10),
    )
    assert report.best is None
    assert report.unresolved == (goal,)
    assert any(outcome.phase == "construct" for outcome in report.failures)


def test_pynguin_program_can_supply_the_missing_setup_sequence():
    generated = """
import target as module_0

def test_case_0():
    timer_like_0 = module_0.TimerLike()
    timer_like_0.start()
"""
    goal = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("stop"),))
    knowledge = KnowledgeBase()
    report = synthesize_object(
        goal,
        knowledge=knowledge,
        max_setup_steps=0,
        hypothesis_backends=(),
        source_artifacts=(SourceArtifact("pynguin", generated),),
    )
    assert report.best is not None
    assert report.best.recipe.setup == (MethodCall("start"),)
    item = knowledge.evidence_item("candidate-recipe", report.best.recipe.key)
    assert "test-source" in {derivation.producer for derivation in item.derivations}


def test_pynguin_program_preserves_module_level_setup_calls():
    generated = """
import target as module_0

def test_case_0():
    timer_like_0 = module_0.TimerLike()
    module_0.start_timer(timer_like_0)
"""
    goal = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("stop"),))
    report = synthesize_object(
        goal,
        max_setup_steps=0,
        hypothesis_backends=(),
        source_artifacts=(SourceArtifact("pynguin", generated),),
    )
    assert report.best is not None
    assert report.best.recipe.setup == (FunctionCall("start_timer", (OBJECT,)),)
    assert "start_timer(object_0)" in report.best.program


def test_an_unavailable_backend_fails_locally_and_the_portfolio_continues():
    report = synthesize_object(
        ObjectGoal(FIXTURE, "TimerLike"),
        max_setup_steps=0,
        hypothesis_backends=("not-a-provider",),
    )
    assert report.best is not None
    assert report.pass_runs["not-a-provider-constructor"] == 1


def test_method_probe_budget_is_validated():
    with pytest.raises(ValueError, match="max_method_probes"):
        synthesize_object(ObjectGoal(FIXTURE, "TimerLike"), max_method_probes=0)


@pytest.mark.parametrize("declared", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
def test_declared_files_preserve_api_owned_executor_defaults(monkeypatch, declared, explicit):
    from tharkun.avengers import api

    files = (FIXTURE,) if declared else None
    supplied, default = object(), object()
    constructed, used = [], []

    def factory(**kwargs):
        constructed.append(kwargs)
        return default

    def portfolio(**kwargs):
        used.append(kwargs["executor"])
        return ()

    monkeypatch.setattr(api, "IsolatedExecutor", factory)
    monkeypatch.setattr(api, "default_passes", portfolio)
    goal = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("stop"),))
    report = api.synthesize_object(goal, files=files, objectives=True,
                                   executor=supplied if explicit else None)
    assert not report.validated
    assert used == [supplied if explicit else default]
    assert constructed == ([] if explicit else [{"trace": True, "files": files}])


def test_a_successful_search_reports_when_it_first_built_an_object():
    goal = ObjectGoal(FIXTURE, "Box", (MethodCall("width"),))
    report = synthesize_object(
        goal,
        hypothesis_examples=40,
        max_setup_steps=0,
        budget=SaturationBudget(max_seconds=15),
    )
    assert report.best is not None
    assert report.time_to_first_object_s is not None
    assert report.time_to_first_object_s >= 0.0
    assert report.metrics["fresh-replay"].runs >= 1


def test_a_search_that_builds_nothing_reports_no_first_object_time():
    goal = ObjectGoal(FIXTURE, "Impossible", (MethodCall("value"),))
    report = synthesize_object(
        goal,
        hypothesis_examples=10,
        max_setup_steps=0,
        budget=SaturationBudget(max_seconds=10),
    )
    assert not report.validated
    assert report.time_to_first_object_s is None


def test_the_yield_scheduler_reaches_the_same_object_as_the_portfolio_order():
    from tharkun.avengers.engine import YieldOrder

    goal = ObjectGoal(FIXTURE, "Box", (MethodCall("width"),))
    ordered = synthesize_object(
        goal, hypothesis_examples=40, max_setup_steps=0,
        budget=SaturationBudget(max_seconds=15),
    )
    ranked = synthesize_object(
        goal, hypothesis_examples=40, max_setup_steps=0,
        budget=SaturationBudget(max_seconds=15), scheduler=YieldOrder(),
    )
    assert ordered.best is not None and ranked.best is not None
    assert ranked.best.outcome.succeeded
