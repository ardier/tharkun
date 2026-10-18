from __future__ import annotations

import json
import os

import pytest

from tharkun.avengers import (
    ConstructionRecipe,
    ConstructorCall,
    MethodCall,
    ObjectGoal,
    synthesize_object,
)
from tharkun.avengers.engine import SaturationBudget
from tharkun.avengers.model import (
    FunctionCall,
    OBJECT,
    ReplayOutcome,
    ValidatedRecipe,
    call_from_record,
    constructor_from_record,
    recipe_from_record,
)
from tharkun.avengers.store import (
    FORMAT,
    StoreSeedPass,
    TemplateStore,
)
from tharkun.knowledge import KnowledgeBase
from tests.avengers_rungs import passes_for


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "avengers_target.py")


def _goal(calls=()):
    return ObjectGoal(FIXTURE, "Box", calls)


def _validated(goal, recipe, digest="abc123"):
    return ValidatedRecipe(
        goal,
        recipe,
        ReplayOutcome("returned", phase="boundary", state_digest=digest),
    )


def _kb_with(goal, recipe, digest="abc123"):
    kb = KnowledgeBase()
    kb.publish(
        "validated-recipe",
        f"{goal.key}:{recipe.key}",
        _validated(goal, recipe, digest),
        producer="fresh-replay",
    )
    return kb


# -- record round-tripping ---------------------------------------------------


def test_every_recipe_literal_survives_a_record_round_trip():
    recipe = ConstructionRecipe(
        "g",
        ConstructorCall(
            (1, "two", b"\xff", (3, None)),
            {"flag": True, "ratio": 0.5, "items": [1, 2], "tags": {"a"}},
        ),
        (MethodCall("grow", (2,), {"by": -0.0}),),
    )
    assert recipe_from_record(recipe.record) == recipe


def test_a_stored_recipe_can_be_rekeyed_onto_a_different_goal():
    recipe = ConstructionRecipe("first-goal", ConstructorCall(kwargs={"lower": 1}))
    moved = recipe_from_record(recipe.record, goal_key="second-goal")
    assert moved.goal_key == "second-goal"
    assert moved.constructor == recipe.constructor


def test_a_function_call_record_round_trips_with_its_receiver_reference():
    call = FunctionCall("classify_box", (OBJECT,), {"minimum": 0})
    assert call_from_record(call.record) == call


def test_a_call_record_naming_both_a_method_and_a_function_is_refused():
    with pytest.raises(ValueError):
        call_from_record({"name": "grow", "target": "classify_box"})
    with pytest.raises(ValueError):
        call_from_record({"args": []})


def test_an_unknown_literal_tag_is_refused_rather_than_passed_through():
    with pytest.raises(ValueError):
        constructor_from_record({"args": [{"eval": "os.system('x')"}], "kwargs": {}})


def test_a_malformed_bytes_record_is_refused():
    with pytest.raises(ValueError):
        constructor_from_record({"args": [{"bytes": "not-hex"}], "kwargs": {}})


# -- store round-tripping ----------------------------------------------------


def test_saving_and_reloading_returns_the_same_recipe(tmp_path):
    goal = _goal((MethodCall("width"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 2}))
    store = TemplateStore(tmp_path)
    store.save(goal, _kb_with(goal, recipe))

    stored = store.recipes_for_goal(goal)
    assert len(stored) == 1
    assert stored[0].recipe == recipe
    assert stored[0].state_digest == "abc123"
    assert stored[0].class_name == "Box"


def test_a_reloaded_recipe_is_offered_as_a_candidate_not_as_a_result(tmp_path):
    goal = _goal((MethodCall("width"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 2}))
    store = TemplateStore(tmp_path)
    store.save(goal, _kb_with(goal, recipe))

    emissions = store.load(goal)
    assert [item.kind for item in emissions] == ["candidate-recipe"]
    assert emissions[0].value == recipe


def test_saving_twice_merges_rather_than_replaces(tmp_path):
    goal = _goal()
    first = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 2}))
    second = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 0, "upper": 9}))
    store = TemplateStore(tmp_path)
    store.save(goal, _kb_with(goal, first))
    store.save(goal, _kb_with(goal, second))

    keys = {item.recipe.key for item in store.recipes_for_goal(goal)}
    assert keys == {first.key, second.key}


def test_recipes_for_type_crosses_goals(tmp_path):
    store = TemplateStore(tmp_path)
    width = _goal((MethodCall("width"),))
    contains = _goal((MethodCall("contains", (1,)),))
    recipe = ConstructionRecipe(width.key, ConstructorCall(kwargs={"lower": 1, "upper": 2}))
    store.save(width, _kb_with(width, recipe))

    assert width.key != contains.key
    found = store.recipes_for_type(f"{FIXTURE}::Box")
    assert len(found) == 1
    assert found[0].recipe.constructor == recipe.constructor
    assert store.recipes_for_type(f"{FIXTURE}::Nothing") == ()


def test_the_store_is_sharded_by_the_goal_key(tmp_path):
    goal = _goal()
    store = TemplateStore(tmp_path)
    path = store.save(goal, KnowledgeBase())
    assert path.parent.name == goal.key[:2]
    assert path.name == f"{goal.key}.json"


def test_a_goal_key_with_a_path_separator_is_refused(tmp_path):
    store = TemplateStore(tmp_path)
    for hostile in ("../escape", "a/b", "", ".hidden"):
        with pytest.raises(ValueError):
            store._path(hostile)


# -- reading is defensive ----------------------------------------------------


def test_a_document_without_the_format_marker_is_ignored(tmp_path):
    goal = _goal()
    store = TemplateStore(tmp_path)
    path = store._path(goal.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"recipes": [{"recipe": {"goal": "g"}}]}))
    assert store.recipes_for_goal(goal) == ()


def test_a_corrupt_document_reads_as_empty_rather_than_raising(tmp_path):
    goal = _goal()
    store = TemplateStore(tmp_path)
    path = store._path(goal.key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json")
    assert store.recipes_for_goal(goal) == ()
    assert store.negatives_for_goal(goal) == ()
    assert store.templates_for_goal(goal) == ()


def test_one_unreadable_entry_does_not_lose_the_rest_of_the_file(tmp_path):
    goal = _goal()
    good = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 2}))
    store = TemplateStore(tmp_path)
    path = store.save(goal, _kb_with(goal, good))
    document = json.loads(path.read_text())
    document["recipes"].insert(0, {"recipe": {"goal": "g", "constructor": {"args": [{"eval": "x"}]}}})
    path.write_text(json.dumps(document))

    stored = store.recipes_for_goal(goal)
    assert len(stored) == 1
    assert stored[0].recipe == good


def test_an_oversized_document_is_refused(tmp_path, monkeypatch):
    goal = _goal()
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 2}))
    store = TemplateStore(tmp_path)
    store.save(goal, _kb_with(goal, recipe))
    monkeypatch.setattr("tharkun.avengers.store.MAX_DOCUMENT_BYTES", 4)
    assert store.recipes_for_goal(goal) == ()


def test_a_saved_document_declares_its_format_and_its_goal(tmp_path):
    goal = _goal((MethodCall("width"),))
    store = TemplateStore(tmp_path)
    document = json.loads(store.save(goal, KnowledgeBase()).read_text())
    assert document["format"] == FORMAT
    assert document["goal"]["class_name"] == "Box"
    assert document["goal"]["key"] == goal.key
    assert document["goal"]["calls"] == [MethodCall("width").record]


# -- the seed pass and the end-to-end loop -----------------------------------


def test_the_seed_pass_emits_candidates_for_the_goal_in_its_delta(tmp_path):
    goal = _goal((MethodCall("width"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 2}))
    store = TemplateStore(tmp_path)
    store.save(goal, _kb_with(goal, recipe))

    kb = KnowledgeBase()
    kb.publish("object-goal", goal.key, goal, producer="initial")
    _cursor, delta = kb.evidence_since(0, kinds=frozenset({"object-goal"}))
    emitted = list(StoreSeedPass(store).run(kb, delta))
    assert [item.value for item in emitted] == [recipe]


def test_a_second_search_reuses_the_first_search_result(tmp_path):
    goal = _goal((MethodCall("width"),))
    store = TemplateStore(tmp_path)

    first = synthesize_object(
        goal,
        store=store,
        hypothesis_examples=40,
        max_setup_steps=0,
        budget=SaturationBudget(max_seconds=15),
    )
    assert first.best is not None
    saved = {item.recipe.key for item in store.recipes_for_goal(goal)}
    assert first.best.recipe.key in saved

    # The stored recipe is re-verified, not trusted: it comes back through the
    # replay pass, so the second run still reports a real validation.
    second = synthesize_object(
        goal,
        store=store,
        passes=[StoreSeedPass(store)] + list(_replay_only()),
        budget=SaturationBudget(max_seconds=15),
    )
    assert second.best is not None
    assert second.best.recipe.key in saved


def _replay_only():
    from tharkun.avengers.executor import IsolatedExecutor
    from tharkun.avengers.passes import ReplayPass

    return (ReplayPass(IsolatedExecutor()),)


# -- what the objectives produce must survive too ----------------------------


def test_a_negative_sequence_is_stored_and_reloaded(tmp_path):
    from tharkun.avengers.objectives import NegativeSequence

    goal = _goal()
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 2}))
    calls = (MethodCall("width"), MethodCall("width"))
    negative = NegativeSequence(
        goal=ObjectGoal(FIXTURE, "Box", calls),
        recipe=recipe,
        calls=calls,
        outcome=ReplayOutcome(
            "raised",
            phase="boundary[1]",
            exception_type="fixtures.avengers_target.Boom",
            reproduced=True,
        ),
        shape="repeated",
    )
    kb = KnowledgeBase()
    kb.publish("negative-sequence", "n1", negative, producer="negative-sequences")

    store = TemplateStore(tmp_path)
    store.save(goal, kb)
    stored = store.negatives_for_goal(goal)
    assert len(stored) == 1
    assert stored[0].exception_type == "fixtures.avengers_target.Boom"
    assert stored[0].shape == "repeated"
    assert stored[0].reproduced is True
    assert [call.name for call in stored[0].calls] == ["width", "width"]
    assert stored[0].recipe == recipe


def test_everything_a_search_found_reaches_the_store(tmp_path):
    """The end-to-end shape: run with objectives, reopen the file, find them."""

    from tharkun.avengers.executor import IsolatedExecutor

    goal = ObjectGoal(FIXTURE, "TimerLike")
    store = TemplateStore(tmp_path)
    executor = IsolatedExecutor(timeout_s=3.0, trace=True)
    # the objective this test reads back is the negative sequences (tests/avengers_rungs.py: the states and branch objectives are
    # independent passes, and run in their own nodes elsewhere), so the search runs the construction portfolio and that one rung
    report = synthesize_object(
        goal,
        objectives=True,
        store=store,
        max_setup_steps=1,
        max_method_probes=8,
        executor=executor,
        passes=passes_for("negatives", executor=executor, store=store, max_setup_steps=1),
        budget=SaturationBudget(max_seconds=60),
    )
    assert report.negatives, "TimerLike rejects start(); start() on purpose"
    reopened = TemplateStore(tmp_path)
    assert len(reopened.recipes_for_goal(goal)) == len(report.validated)
    assert len(reopened.negatives_for_goal(goal)) == len(report.negatives)
