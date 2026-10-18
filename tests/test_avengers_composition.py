from __future__ import annotations

import os

import pytest

from tharkun.avengers import (
    ConstructionRecipe,
    ConstructorCall,
    IsolatedExecutor,
    MethodCall,
    ObjectGoal,
    synthesize_object,
)
from tharkun.avengers.engine import SaturationBudget
from tharkun.avengers.model import (
    MAX_NESTING,
    NestedObject,
    _literal_record,
    _literal_value,
)
from tharkun.avengers.passes import NestedCandidatePass, resolve_annotation_class
from tharkun.knowledge import KnowledgeBase


FIXTURE = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "fixtures", "avengers_target.py")
)


def _box(lower: int = 1, upper: int = 5) -> NestedObject:
    return NestedObject(
        FIXTURE,
        "Box",
        ConstructionRecipe("inner", ConstructorCall(kwargs={"lower": lower, "upper": upper})),
    )


# -- the value object --------------------------------------------------------


def test_a_nested_object_survives_a_record_round_trip():
    nested = _box()
    assert _literal_value(_literal_record(nested)) == nested


def test_nesting_deeper_than_the_bound_is_refused():
    value = _box()
    for _ in range(MAX_NESTING):
        value = NestedObject(
            FIXTURE, "Box", ConstructionRecipe("inner", ConstructorCall((value,)))
        )
    assert value.depth > MAX_NESTING
    with pytest.raises(ValueError, match="nest"):
        _literal_record(value)


def test_a_composed_recipe_keys_on_its_whole_composition():
    goal = ObjectGoal(FIXTURE, "Shipment")
    first = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"box": _box(1, 5)}))
    second = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"box": _box(0, 9)}))
    assert first.key != second.key


# -- rendering ---------------------------------------------------------------


def test_a_composed_program_binds_each_nested_object_to_a_variable():
    goal = ObjectGoal(FIXTURE, "Shipment", (MethodCall("volume"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"box": _box(1, 5)}))
    program = recipe.render("Shipment")
    assert program.splitlines() == [
        "nested_0 = Box(lower=1, upper=5)",
        "object_0 = Shipment(box=nested_0)",
    ]


def test_a_composed_program_is_executable_python():
    goal = ObjectGoal(FIXTURE, "Shipment", (MethodCall("volume"),))
    recipe = ConstructionRecipe(
        goal.key,
        ConstructorCall(kwargs={"box": _box(1, 5), "label": "crate"}),
        (MethodCall("seal"),),
    )
    outcome = IsolatedExecutor().replay(goal, recipe)
    assert outcome.succeeded

    namespace: dict[str, object] = {}
    exec(compile(open(FIXTURE).read(), FIXTURE, "exec"), namespace)
    from tharkun.avengers.model import ValidatedRecipe

    program = ValidatedRecipe(goal, recipe, outcome).program
    exec(compile(program, "<program>", "exec"), namespace)
    assert namespace["object_0"].sealed is True


def test_the_same_sub_recipe_is_bound_once_and_shared():
    goal = ObjectGoal(FIXTURE, "Shipment")
    box = _box(1, 5)
    recipe = ConstructionRecipe(
        goal.key,
        ConstructorCall(kwargs={"box": box}),
        (MethodCall("seal", (), {"other": box}),),
    )
    program = recipe.render("Shipment")
    assert program.count("nested_0 = Box(") == 1
    assert program.count("nested_1") == 0


# -- worker materialisation --------------------------------------------------


def test_a_class_that_needs_an_object_is_built_from_a_nested_recipe():
    goal = ObjectGoal(FIXTURE, "Shipment", (MethodCall("volume"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"box": _box(1, 5)}))
    outcome = IsolatedExecutor().replay(goal, recipe)
    assert outcome.succeeded
    assert outcome.return_types == ("builtins.int",)


def test_a_nested_constructor_failure_is_reported_as_a_nested_phase():
    goal = ObjectGoal(FIXTURE, "Shipment", (MethodCall("volume"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"box": _box(5, 1)}))
    outcome = IsolatedExecutor().replay(goal, recipe)
    assert not outcome.succeeded
    assert outcome.phase == "construct/nested:construct"
    assert outcome.exception_type == "builtins.ValueError"
    assert "lower must be smaller" in (outcome.detail or "")


def test_the_receiver_cannot_appear_in_its_own_constructor_arguments():
    from tharkun.avengers import OBJECT

    goal = ObjectGoal(FIXTURE, "Box", (MethodCall("width"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": OBJECT, "upper": 2}))
    outcome = IsolatedExecutor().replay(goal, recipe)
    assert not outcome.succeeded
    assert outcome.phase == "construct"
    assert outcome.exception_type == "builtins.ValueError"


def test_a_nested_object_can_carry_its_own_setup():
    goal = ObjectGoal(FIXTURE, "Shipment", (MethodCall("volume"),))
    inner = ConstructionRecipe(
        "inner",
        ConstructorCall(kwargs={"lower": 1, "upper": 5}),
        (MethodCall("width"),),
    )
    recipe = ConstructionRecipe(
        goal.key, ConstructorCall(kwargs={"box": NestedObject(FIXTURE, "Box", inner)})
    )
    assert IsolatedExecutor().replay(goal, recipe).succeeded


# -- annotation resolution ---------------------------------------------------


def test_an_annotation_naming_a_module_local_class_resolves_to_it():
    assert resolve_annotation_class(FIXTURE, "Box") == (FIXTURE, "Box")
    assert resolve_annotation_class(FIXTURE, "Optional[Box]") == (FIXTURE, "Box")
    assert resolve_annotation_class(FIXTURE, "Box | None") == (FIXTURE, "Box")


def test_builtin_and_unknown_annotations_resolve_to_nothing():
    for annotation in ("int", "list[str]", "Mapping[str, int]", "NotAClass", None):
        assert resolve_annotation_class(FIXTURE, annotation) is None


# -- the pass ----------------------------------------------------------------


def test_the_pass_composes_a_recipe_from_evidence_already_in_the_base():
    from tharkun.avengers.model import ReplayOutcome, ValidatedRecipe
    from tharkun.solve.frontend.constructor import constructor_template

    goal = ObjectGoal(FIXTURE, "Shipment", (MethodCall("volume"),))
    box_goal = ObjectGoal(FIXTURE, "Box")
    box_recipe = ConstructionRecipe(
        box_goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 5})
    )

    kb = KnowledgeBase()
    kb.publish("object-goal", goal.key, goal, producer="initial")
    kb.publish(
        "validated-recipe",
        f"{box_goal.key}:{box_recipe.key}",
        ValidatedRecipe(box_goal, box_recipe, ReplayOutcome("returned")),
        producer="fresh-replay",
    )
    template = constructor_template(FIXTURE, "Shipment")
    kb.publish("constructor-template", goal.key, template, producer="constructor-template")
    _cursor, delta = kb.evidence_since(0, kinds=frozenset({"constructor-template"}))

    emitted = list(NestedCandidatePass().run(kb, delta))
    assert emitted
    composed = emitted[0].value.constructor.keyword_arguments["box"]
    assert isinstance(composed, NestedObject)
    assert composed.class_name == "Box"
    assert composed.recipe.constructor == box_recipe.constructor


def test_the_pass_stops_at_its_depth_bound():
    from tharkun.solve.frontend.constructor import constructor_template

    goal = ObjectGoal(FIXTURE, "Shipment")
    kb = KnowledgeBase()
    kb.publish("object-goal", goal.key, goal, producer="initial")
    kb.publish(
        "constructor-template",
        goal.key,
        constructor_template(FIXTURE, "Shipment"),
        producer="constructor-template",
    )
    _cursor, delta = kb.evidence_since(0, kinds=frozenset({"constructor-template"}))
    exhausted = NestedCandidatePass(max_depth=1, depth=1)
    assert list(exhausted.run(kb, delta)) == []


# -- end to end --------------------------------------------------------------


def test_a_class_needing_an_object_argument_is_synthesized_end_to_end():
    goal = ObjectGoal(FIXTURE, "Shipment", (MethodCall("volume"),))
    report = synthesize_object(
        goal,
        hypothesis_examples=30,
        max_setup_steps=0,
        budget=SaturationBudget(max_seconds=30),
    )
    assert report.best is not None, "Shipment needs a real Box and must still be built"
    box = report.best.recipe.constructor.keyword_arguments["box"]
    assert isinstance(box, NestedObject)
    assert "nested_0 = Box(" in report.best.program
    assert "object_0 = Shipment(box=nested_0)" in report.best.program


def test_composition_can_be_switched_off():
    goal = ObjectGoal(FIXTURE, "Shipment", (MethodCall("volume"),))
    report = synthesize_object(
        goal,
        hypothesis_examples=20,
        max_setup_steps=0,
        compose_depth=0,
        budget=SaturationBudget(max_seconds=15),
    )
    assert not report.validated
