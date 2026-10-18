"""Boundary-directed goals: given a behaviour, produce the program that reaches it.

The default goal asks for calls that return normally. A directed goal inverts
that -- "find me an object whose ``stop()`` raises", "find me one that takes the
interpreter down" -- which is the question the object-synthesis work exists to
answer, and the one an ordinary search treats as a failure to be avoided.
"""

from __future__ import annotations

import os

import pytest

from tharkun.avengers import (
    ConstructionRecipe,
    ConstructorCall,
    IsolatedExecutor,
    MethodCall,
    ObjectGoal,
    ReplayOutcome,
    synthesize_object,
)
from tharkun.avengers.engine import SaturationBudget


FIXTURE = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "fixtures", "avengers_target.py")
)


def _goal(expect: str = "returns", calls=(MethodCall("stop"),)):
    return ObjectGoal(FIXTURE, "TimerLike", calls, expect)


# -- the vocabulary ----------------------------------------------------------


def test_an_unknown_expectation_is_refused():
    for bad in ("succeeds", "", "hangs", "returns:ValueError"):
        with pytest.raises((ValueError, TypeError)):
            _goal(bad)


def test_a_directed_goal_without_a_boundary_call_is_refused():
    # There is nothing to direct: the expectation is about the boundary.
    with pytest.raises(ValueError, match="boundary call"):
        ObjectGoal(FIXTURE, "TimerLike", (), "raises")


def test_the_default_goals_key_is_unchanged_by_the_new_field():
    # Keys mint the store's filenames and every recipe's identity. A goal
    # written before directed expectations existed must still name itself the
    # same way.
    plain = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("stop"),))
    assert plain.key == ObjectGoal(FIXTURE, "TimerLike", (MethodCall("stop"),), "returns").key
    assert plain.key != _goal("raises").key
    assert _goal("raises").key != _goal("raises:RuntimeError").key


# -- what counts as satisfying --------------------------------------------


def test_returning_satisfies_only_the_default_goal():
    outcome = ReplayOutcome("returned", phase="boundary")
    assert _goal("returns").satisfied(outcome)
    assert not _goal("raises").satisfied(outcome)
    assert not _goal("crashes").satisfied(outcome)


def test_raising_at_the_boundary_satisfies_a_raises_goal():
    outcome = ReplayOutcome(
        "raised", phase="boundary[0]", exception_type="builtins.RuntimeError"
    )
    assert _goal("raises").satisfied(outcome)
    assert _goal("raises:RuntimeError").satisfied(outcome)
    assert _goal("raises:builtins.RuntimeError").satisfied(outcome)
    assert not _goal("raises:ValueError").satisfied(outcome)
    assert not _goal("returns").satisfied(outcome)


def test_a_constructor_that_raises_does_not_satisfy_a_raises_goal():
    # It has not produced an object that misbehaves; it has failed to produce
    # an object, which is a different and uninteresting result.
    outcome = ReplayOutcome(
        "raised", phase="construct", exception_type="builtins.RuntimeError"
    )
    assert not _goal("raises").satisfied(outcome)
    assert not _goal("raises:RuntimeError").satisfied(outcome)


def test_a_setup_call_that_raises_does_not_satisfy_a_raises_goal():
    outcome = ReplayOutcome(
        "raised", phase="setup[0]", exception_type="builtins.RuntimeError"
    )
    assert not _goal("raises").satisfied(outcome)


def test_a_native_crash_satisfies_a_crashes_goal():
    assert _goal("crashes").satisfied(ReplayOutcome("crashed", phase="worker", signal=11))
    assert _goal("crashes").satisfied(ReplayOutcome("killed", phase="worker", signal=9))
    assert not _goal("crashes").satisfied(ReplayOutcome("timeout", phase="worker"))
    assert not _goal("raises").satisfied(ReplayOutcome("crashed", phase="worker", signal=11))


# -- end to end --------------------------------------------------------------


def test_a_search_finds_an_object_whose_boundary_raises():
    report = synthesize_object(
        _goal("raises:RuntimeError"),
        max_setup_steps=1,
        budget=SaturationBudget(max_seconds=45),
    )
    assert report.best is not None
    assert report.best.outcome.exception_type == "builtins.RuntimeError"
    assert report.best.program.endswith("object_0.stop()")
    # stop() without start() is precisely the program that does it.
    assert report.best.recipe.setup == ()


def test_a_search_finds_an_object_that_takes_the_interpreter_down():
    goal = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("hard_crash"),), "crashes")
    report = synthesize_object(
        goal, max_setup_steps=0, budget=SaturationBudget(max_seconds=45)
    )
    assert report.best is not None
    assert report.best.outcome.kind == "crashed"
    assert report.best.outcome.signal is not None


def test_a_directed_goals_own_answers_are_not_filed_as_its_failures():
    goal = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("hard_crash"),), "crashes")
    report = synthesize_object(
        goal, max_setup_steps=0, budget=SaturationBudget(max_seconds=45)
    )
    assert report.validated
    assert all(not goal.satisfied(item) for item in report.failures)


def test_a_behaviour_the_class_never_exhibits_is_unresolved_not_impossible():
    # Box.width() cannot be made to raise by any constructor. The report says
    # "not found", which is not the same as "cannot exist".
    goal = ObjectGoal(FIXTURE, "Box", (MethodCall("width"),), "raises")
    report = synthesize_object(
        goal,
        max_setup_steps=0,
        hypothesis_examples=20,
        budget=SaturationBudget(max_seconds=25),
    )
    assert not report.validated
    assert report.unresolved == (goal,)


def test_one_replay_can_answer_opposite_goals():
    """The worker reports what happened; the goal decides what it means."""

    wants_return = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("stop"),))
    wants_raise = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("stop"),), "raises")
    recipe = ConstructionRecipe(wants_return.key, ConstructorCall())
    outcome = IsolatedExecutor().replay(wants_return, recipe)

    assert not wants_return.satisfied(outcome)
    assert wants_raise.satisfied(outcome)
