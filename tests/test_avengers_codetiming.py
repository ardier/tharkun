"""The worked case study, as an executable claim.

``codetiming.Timer`` is the smallest real class with a genuine protocol:
``stop()`` requires a prior ``start()`` and ``start()`` refuses to run twice.
Everything the object-synthesis goal asks for shows up in one small class --
valid construction, method coverage, several receiver states, branch coverage,
and sequences that are wrong on purpose.
"""

from __future__ import annotations

import pytest

codetiming = pytest.importorskip("codetiming", reason="codetiming is a benchmark subject")

from tharkun.avengers import (  # noqa: E402
    IsolatedExecutor,
    MethodCall,
    ObjectGoal,
    synthesize_object,
)
from tharkun.avengers.engine import SaturationBudget  # noqa: E402
from tharkun.solve.frontend.constructor import constructor_template  # noqa: E402

TIMER = codetiming.Timer
BUDGET = SaturationBudget(max_seconds=120)


@pytest.fixture(scope="module")
def report():
    return synthesize_object(
        ObjectGoal.from_type(TIMER),
        objectives=True,
        max_setup_steps=2,
        max_method_probes=32,
        executor=IsolatedExecutor(timeout_s=5.0, trace=True),
        budget=BUDGET,
    )


# -- 1. valid construction ---------------------------------------------------


def test_the_dataclass_constructor_is_read_without_running_anything():
    # Timer is a @dataclass: there is no __init__ in its source, and reading
    # only __init__ reported "takes no arguments" -- wrong in exactly the way
    # that stops anything from varying its state.
    template = constructor_template(
        ObjectGoal.from_type(TIMER).source_path, "Timer"
    )
    assert template is not None
    names = [item.name for item in template.parameters]
    assert "name" in names and "logger" in names
    # A ClassVar is not a constructor parameter and neither is field(init=False).
    assert "timers" not in names
    assert "_start_time" not in names


def test_a_timer_that_can_cross_stop_is_discovered_from_the_boundary_alone():
    goal = ObjectGoal.from_type(TIMER, (MethodCall("stop"),))
    result = synthesize_object(goal, max_setup_steps=2, budget=BUDGET)
    assert result.best is not None
    # The start() nobody asked for is the whole point.
    assert result.best.recipe.setup == (MethodCall("start"),)
    assert result.best.program.splitlines() == [
        "object_0 = Timer()",
        "object_0.start()",
        "object_0.stop()",
    ]
    assert result.time_to_first_object_s is not None


# -- 2. method and state coverage -------------------------------------------


def test_both_public_methods_are_reached(report):
    assert report.covered_methods == ("start", "stop")
    assert set(report.method_universe) == {"start", "stop"}


def test_one_object_is_not_enough_so_a_portfolio_is_returned(report):
    # start() and stop() have incompatible preconditions: no single state
    # admits both, so "one object" is the wrong answer to this class.
    programs = {state.program for state in report.objects}
    assert len(programs) >= 2
    assert "object_0 = Timer()" in programs
    assert any("object_0.start()" in program for program in programs)


def test_several_distinct_receiver_states_are_found(report):
    assert len(report.states) >= 2
    assert len({state.digest for state in report.states}) == len(report.states)


# -- 3. branch coverage ------------------------------------------------------


def test_the_search_gets_well_past_the_constructor(report):
    assert report.branch_coverage is not None
    # Measured at 0.75 with the coverage objective varying one constructor
    # argument at a time; the floor guards the mechanism, not the exact number.
    assert report.branch_coverage >= 0.5
    assert report.line_coverage >= 0.5


def test_what_is_left_uncovered_is_left_for_a_stated_reason(report):
    assert report.coverage is not None
    source = open(report.coverage.goal.source_path, encoding="utf-8").read().splitlines()
    for start, _end in report.coverage.uncovered_arcs:
        guard = source[start - 1].strip()
        # Everything still uncovered is either a callable test -- which no
        # literal satisfies -- or a guard that needs two constructor arguments
        # set together, which this objective varies one at a time.
        assert "callable(" in guard or "self." in guard, guard


# -- 4. deliberately invalid sequences ---------------------------------------


def _sequences(report, shape=None):
    return {
        tuple(call.name for call in item.calls)
        for item in report.negatives
        if shape is None or item.shape == shape
    }


def test_starting_a_running_timer_is_reported_as_a_result(report):
    assert ("start", "start") in _sequences(report, "repeated")
    offending = next(
        item
        for item in report.negatives
        if tuple(call.name for call in item.calls) == ("start", "start")
    )
    assert offending.exception_type.endswith("TimerError")
    assert offending.outcome.reproduced is True
    assert offending.program.splitlines()[-2:] == [
        "object_0.start()",
        "object_0.start()",
    ]


def test_stopping_a_timer_that_never_started_is_reported_as_a_result(report):
    assert ("stop",) in _sequences(report, "unprepared") or ("stop", "stop") in _sequences(
        report, "repeated"
    )


def test_every_reported_sequence_raises_the_classes_own_error(report):
    assert report.negatives
    for item in report.negatives:
        # TimerError is defined in the module under test. A TypeError from
        # calling something with the wrong arity would not be a finding.
        assert item.exception_type.endswith("TimerError"), item.exception_type
        assert item.outcome.reproduced is True


def test_the_exception_is_named_by_its_real_module(report):
    # Not by the synthetic exec-target name a fallback loader would give it.
    assert all(
        item.exception_type.startswith("codetiming.") for item in report.negatives
    )
