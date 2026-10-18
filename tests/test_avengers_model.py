from __future__ import annotations

import inspect
import math
import os

import pytest

from tharkun.avengers.model import (
    AvengersReport,
    ConstructionRecipe,
    ConstructorCall,
    FieldBound,
    FunctionCall,
    MethodCall,
    MethodCoverage,
    OBJECT,
    ObjectGoal,
    ReplayOutcome,
    SourceArtifact,
    ValidatedRecipe,
    field_bounds_hold,
    materialize_recipe,
)


def test_recipe_identity_ignores_producer_by_not_storing_one(tmp_path):
    goal = ObjectGoal(str(tmp_path / "target.py"), "Box")
    first = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 0, "upper": 1}))
    second = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"upper": 1, "lower": 0}))
    assert first == second
    assert first.key == second.key


def test_literal_identity_preserves_special_float_and_container_shapes(tmp_path):
    goal = ObjectGoal(str(tmp_path / "target.py"), "Box")
    call = ConstructorCall(args=([float("nan")],), kwargs={"raw": b"\x00"})
    recipe = ConstructionRecipe(goal.key, call)
    assert recipe.key == ConstructionRecipe(goal.key, call).key
    assert math.isnan(call.args[0][0])


def test_nonliteral_values_are_rejected_before_crossing_process_boundary():
    with pytest.raises(TypeError, match="literal"):
        ConstructorCall(args=(object(),))


def test_validated_program_includes_setup_and_all_boundary_calls(tmp_path):
    goal = ObjectGoal(
        str(tmp_path / "target.py"),
        "Timer",
        (MethodCall("stop"), MethodCall("start")),
    )
    recipe = ConstructionRecipe(
        goal.key,
        ConstructorCall(kwargs={"logger": None}),
        (MethodCall("start"),),
    )
    validated = ValidatedRecipe(goal, recipe, ReplayOutcome("returned"))
    assert validated.program.splitlines() == [
        "object_0 = Timer(logger=None)",
        "object_0.start()",
        "object_0.stop()",
        "object_0.start()",
    ]


def test_report_best_prefers_the_shortest_validated_recipe(tmp_path):
    goal = ObjectGoal(str(tmp_path / "target.py"), "Timer")
    long = ValidatedRecipe(
        goal,
        ConstructionRecipe(goal.key, setup=(MethodCall("start"),)),
        ReplayOutcome("returned"),
    )
    short = ValidatedRecipe(goal, ConstructionRecipe(goal.key), ReplayOutcome("returned"))
    report = AvengersReport((long, short), "saturated", {}, 2)
    assert report.best == short


def test_source_artifact_identity_includes_tool_and_content():
    first = SourceArtifact("pynguin", "def test_a(): pass")
    assert first.key == SourceArtifact("pynguin", first.source).key
    assert first.key != SourceArtifact("ghostwriter", first.source).key


def test_function_call_renders_the_object_reference_in_nested_arguments():
    call = FunctionCall("classify", ([OBJECT],), {"subject": OBJECT})
    assert call.render() == "classify([object_0], subject=object_0)"


def test_object_goal_can_be_created_from_an_importable_class():
    goal = ObjectGoal.from_type(ObjectGoal, (MethodCall("key"),))
    assert goal.source_path == os.path.realpath(inspect.getsourcefile(ObjectGoal))
    assert goal.class_name == "ObjectGoal"
    assert goal.calls == (MethodCall("key"),)


def test_method_coverage_changes_best_recipe_and_builds_a_portfolio(tmp_path):
    goal = ObjectGoal(str(tmp_path / "target.py"), "Timer")
    idle = ValidatedRecipe(
        goal,
        ConstructionRecipe(goal.key),
        ReplayOutcome("returned"),
    )
    running = ValidatedRecipe(
        goal,
        ConstructionRecipe(goal.key, setup=(MethodCall("start"),)),
        ReplayOutcome("returned"),
    )
    returned = ReplayOutcome("returned")
    raised = ReplayOutcome("raised")
    report = AvengersReport(
        (idle, running),
        "saturated",
        {},
        0,
        method_coverage=(
            MethodCoverage(
                idle,
                ((MethodCall("start"), returned), (MethodCall("stop"), raised)),
            ),
            MethodCoverage(
                running,
                ((MethodCall("start"), raised), (MethodCall("stop"), returned)),
            ),
        ),
    )
    assert report.best == idle
    assert report.covered_methods == ("start", "stop")
    assert report.objects == (idle, running)


# FieldBound / field_bounds_hold / materialize_recipe ([internal ref] slice 3, [internal ref]) --
# moved here from test_avengers_service.py (2026-09-29, owner correction: these are
# tharkun.avengers's own public API, not a separate service's).

class _FieldBoundObj:
    def __init__(self, x, items):
        self.x = x
        self.items = items


def test_field_bounds_hold_and_fail():
    obj = _FieldBoundObj(5, [1])
    assert field_bounds_hold(obj, (FieldBound("x", "gt", 3), FieldBound("items", "truthy")))
    assert not field_bounds_hold(obj, (FieldBound("x", "gt", 10),))


def test_field_bounds_hold_missing_attribute_fails_closed():
    obj = _FieldBoundObj(5, [1])
    assert not field_bounds_hold(obj, (FieldBound("nope", "truthy"),))


def test_field_bounds_hold_type_mismatch_fails_closed_not_raises():
    obj = _FieldBoundObj("not a number", [1])
    assert not field_bounds_hold(obj, (FieldBound("x", "gt", 3),))


def test_materialize_recipe_bare_constructor_and_with_setup():
    class C:
        def __init__(self, x):
            self.x = x
            self.bumped = 0

        def bump(self):
            self.bumped += 1

    ns = {"C": C}
    bare = ConstructionRecipe("k", ConstructorCall(kwargs={"x": 3}), ())
    obj = materialize_recipe(bare, "C", ns)
    assert (obj.x, obj.bumped) == (3, 0)

    with_setup = ConstructionRecipe("k", ConstructorCall(kwargs={"x": 3}),
                                    (MethodCall("bump"),))
    obj2 = materialize_recipe(with_setup, "C", ns)
    assert (obj2.x, obj2.bumped) == (3, 1)
