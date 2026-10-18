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
from tharkun.avengers.model import ReplayOutcome, ValidatedRecipe
from tharkun.avengers.objectives import (
    COVERAGE,
    NEGATIVE,
    STATE,
    CoverageSetupPass,
    NegativeSequencePass,
    ReceiverState,
    StateDiversityPass,
    _truthy_value,
    class_branch_arcs,
    class_line_span,
    observe_coverage,
    target_exceptions,
)
from tharkun.knowledge import KnowledgeBase
from tests.avengers_rungs import RUNGS, passes_for


FIXTURE = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "fixtures", "avengers_target.py")
)
TRACING = IsolatedExecutor(trace=True)


def _validated(class_name, recipe, outcome=None):
    goal = ObjectGoal(FIXTURE, class_name)
    return ValidatedRecipe(goal, recipe, outcome or ReplayOutcome("returned"))


def _kb_with(*validated):
    kb = KnowledgeBase()
    for index, item in enumerate(validated):
        kb.publish(
            "validated-recipe",
            f"{item.goal.key}:{item.recipe.key}:{index}",
            item,
            producer="fresh-replay",
        )
    return kb


def _delta(kb, kind="validated-recipe"):
    _cursor, delta = kb.evidence_since(0, kinds=frozenset({kind}))
    return delta


# -- static denominators -----------------------------------------------------


def test_a_class_span_and_its_branch_arcs_are_found_statically():
    span = class_line_span(FIXTURE, "TimerLike")
    assert span is not None and span[0] < span[1]
    arcs = class_branch_arcs(FIXTURE, "TimerLike")
    # TimerLike guards on self.running in both start() and stop(); each guard
    # contributes the arc into its body and the arc that falls past it.
    assert len(arcs) == 4
    assert all(span[0] <= start <= span[1] for start, _end in arcs)


def test_a_class_that_does_not_exist_has_no_span():
    assert class_line_span(FIXTURE, "NotAClass") is None
    assert class_branch_arcs(FIXTURE, "NotAClass") == ()


# -- state coverage ----------------------------------------------------------


def test_two_recipes_reaching_one_state_are_published_once():
    goal = ObjectGoal(FIXTURE, "TimerLike")
    # start(); stop() returns TimerLike to the state the bare constructor
    # leaves it in, so these two programs are one result.
    bare = ConstructionRecipe(goal.key, ConstructorCall())
    round_trip = ConstructionRecipe(
        goal.key, ConstructorCall(), (MethodCall("start"), MethodCall("stop"))
    )
    first = TRACING.replay(goal, bare)
    second = TRACING.replay(goal, round_trip)
    assert first.state_digest == second.state_digest, "the premise of this test"

    kb = _kb_with(
        ValidatedRecipe(goal, bare, first), ValidatedRecipe(goal, round_trip, second)
    )
    states = [
        item
        for item in StateDiversityPass(TRACING, max_steps=0).run(kb, _delta(kb))
        if item.kind == STATE
    ]
    assert len(states) == 1
    assert isinstance(states[0].value, ReceiverState)
    assert states[0].value.stable is True


def test_a_replay_without_a_digest_is_not_counted_as_a_state():
    goal = ObjectGoal(FIXTURE, "TimerLike")
    without = _validated(
        "TimerLike",
        ConstructionRecipe(goal.key, ConstructorCall()),
        ReplayOutcome("returned", state_digest=None),
    )
    kb = _kb_with(without)
    emitted = list(StateDiversityPass(TRACING, max_steps=0).run(kb, _delta(kb)))
    assert [item for item in emitted if item.kind == STATE] == []


def test_the_pass_proposes_setups_that_reach_a_state_nobody_has_seen():
    goal = ObjectGoal(FIXTURE, "TimerLike")
    recipe = ConstructionRecipe(goal.key, ConstructorCall())
    base = ValidatedRecipe(goal, recipe, TRACING.replay(goal, recipe))
    kb = _kb_with(base)
    emitted = list(StateDiversityPass(TRACING, max_steps=1).run(kb, _delta(kb)))
    candidates = [item for item in emitted if item.kind == "candidate-recipe"]
    assert candidates, "start() reaches a state the bare constructor does not"
    assert any(
        recipe.value.setup == (MethodCall("start"),) for recipe in candidates
    )


# -- line and branch coverage ------------------------------------------------


def test_coverage_is_none_when_nothing_was_traced():
    goal = ObjectGoal(FIXTURE, "TimerLike")
    untraced = _validated("TimerLike", ConstructionRecipe(goal.key, ConstructorCall()))
    assert observe_coverage(goal, [untraced]) is None


def test_coverage_reports_what_the_validated_recipes_reached():
    goal = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("start"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall())
    outcome = TRACING.replay(goal, recipe)
    assert outcome.succeeded

    observed = observe_coverage(goal, [ValidatedRecipe(goal, recipe, outcome)])
    assert observed is not None
    assert 0.0 < observed.line_coverage <= 1.0
    assert 0.0 < observed.branch_coverage < 1.0
    assert observed.uncovered_arcs


def test_a_coverage_observation_is_keyed_by_what_it_covered():
    goal = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("start"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall())
    outcome = TRACING.replay(goal, recipe)
    narrow = observe_coverage(goal, [ValidatedRecipe(goal, recipe, outcome)])

    wider_outcome = ReplayOutcome(
        "returned", lines=(*outcome.lines, 99999), arcs=outcome.arcs
    )
    wider = observe_coverage(goal, [ValidatedRecipe(goal, recipe, wider_outcome)])
    # Coverage grows as recipes validate, so two observations of one goal are
    # two different facts and must not collide on one evidence identity.
    assert narrow.key != wider.key
    assert narrow.key.startswith(goal.key)


def test_a_truthy_literal_is_chosen_from_the_annotation():
    assert _truthy_value("str") == "x"
    assert _truthy_value("int") == 1
    assert _truthy_value("Optional[str]") == "x"
    assert _truthy_value("bool") is True
    assert _truthy_value("list[int]") == [0]


def test_no_literal_is_proposed_for_a_parameter_that_must_be_callable():
    from tharkun.avengers.objectives import NO_LITERAL

    # Satisfying `if self.logger:` with True would pass the guard and then
    # crash on `self.logger(text)` one line below it.
    assert _truthy_value("Optional[Callable[[str], None]]") is NO_LITERAL
    assert _truthy_value("Callable[[float], str]") is NO_LITERAL


def test_none_is_a_real_literal_not_an_absence_of_one():
    from tharkun.avengers.objectives import NO_LITERAL, _falsy_value

    # `logger=None` is exactly what the false side of `if self.logger:` needs.
    # Treating it as "no literal available" loses that branch.
    assert _falsy_value("Optional[Callable[[str], None]]") is None
    assert _falsy_value("Optional[str]") is None
    assert _falsy_value("str") == ""
    assert _falsy_value("int") == 0
    # A required callable has no falsy literal at all.
    assert _falsy_value("Callable[[float], str]") is NO_LITERAL


def test_the_coverage_pass_proposes_a_constructor_argument_for_an_uncovered_guard(tmp_path):
    module = tmp_path / "gated.py"
    module.write_text(
        "class Gated:\n"
        "    def __init__(self, label: str = ''):\n"
        "        self.label = label\n"
        "\n"
        "    def describe(self) -> str:\n"
        "        if self.label:\n"
        "            return self.label\n"
        "        return 'anonymous'\n"
    )
    path = os.path.realpath(str(module))
    goal = ObjectGoal(path, "Gated", (MethodCall("describe"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"label": ""}))
    outcome = TRACING.replay(goal, recipe)
    assert outcome.succeeded

    kb = _kb_with(ValidatedRecipe(goal, recipe, outcome))
    emitted = list(CoverageSetupPass(TRACING).run(kb, _delta(kb)))
    assert any(item.kind == COVERAGE for item in emitted)
    labels = {
        item.value.constructor.keyword_arguments.get("label")
        for item in emitted
        if item.kind == "candidate-recipe"
    }
    assert "x" in labels, "the truthy branch of `if self.label:` needs a truthy label"


# -- negative sequences ------------------------------------------------------


def test_the_target_modules_own_exception_names_are_collected():
    names = target_exceptions(FIXTURE)
    # RuntimeError is raised by name inside TimerLike, so it is a precondition
    # the class states, not an artefact of how it was probed.
    assert "RuntimeError" in names
    assert "ValueError" in names
    assert "KeyError" not in names


def test_repeating_a_state_transition_is_kept_as_a_result():
    goal = ObjectGoal(FIXTURE, "TimerLike")
    validated = _validated(
        "TimerLike",
        ConstructionRecipe(goal.key, ConstructorCall()),
        ReplayOutcome("returned", state_digest="d"),
    )
    kb = _kb_with(validated)
    emitted = list(NegativeSequencePass(TRACING).run(kb, _delta(kb)))
    negatives = [item.value for item in emitted if item.kind == NEGATIVE]
    assert negatives

    repeated = [item for item in negatives if item.shape == "repeated"]
    starts = [
        item
        for item in repeated
        if [call.name for call in item.calls] == ["start", "start"]
    ]
    assert starts, "start(); start() is exactly the sequence this objective is for"
    assert starts[0].exception_type == "builtins.RuntimeError"
    assert "object_0.start()\nobject_0.start()" in starts[0].program


def test_a_kept_negative_is_confirmed_in_its_own_child():
    goal = ObjectGoal(FIXTURE, "TimerLike")
    validated = _validated(
        "TimerLike",
        ConstructionRecipe(goal.key, ConstructorCall()),
        ReplayOutcome("returned", state_digest="d"),
    )
    kb = _kb_with(validated)
    negatives = [
        item.value
        for item in NegativeSequencePass(TRACING).run(kb, _delta(kb))
        if item.kind == NEGATIVE
    ]
    assert negatives
    assert all(item.outcome.reproduced is True for item in negatives)


def test_skipping_a_precondition_is_kept_as_a_result():
    goal = ObjectGoal(FIXTURE, "TimerLike")
    prepared = _validated(
        "TimerLike",
        ConstructionRecipe(goal.key, ConstructorCall(), (MethodCall("start"),)),
        ReplayOutcome("returned", state_digest="running"),
    )
    kb = _kb_with(prepared)
    negatives = [
        item.value
        for item in NegativeSequencePass(TRACING).run(kb, _delta(kb))
        if item.kind == NEGATIVE
    ]
    unprepared = [
        item
        for item in negatives
        if item.shape == "unprepared" and [c.name for c in item.calls] == ["stop"]
    ]
    assert unprepared, "stop() without start() is a documented precondition failure"
    assert unprepared[0].recipe.setup == ()


def test_an_exception_the_target_never_names_is_not_a_finding(tmp_path):
    module = tmp_path / "quiet.py"
    module.write_text(
        "class Quiet:\n"
        "    def __init__(self):\n"
        "        self.n = 0\n"
        "\n"
        "    def bump(self) -> int:\n"
        "        self.n += 1\n"
        "        return self.n\n"
    )
    path = os.path.realpath(str(module))
    goal = ObjectGoal(path, "Quiet")
    validated = ValidatedRecipe(
        goal,
        ConstructionRecipe(goal.key, ConstructorCall()),
        ReplayOutcome("returned", state_digest="d"),
    )
    kb = _kb_with(validated)
    emitted = list(NegativeSequencePass(TRACING).run(kb, _delta(kb)))
    assert [item for item in emitted if item.kind == NEGATIVE] == []


# -- end to end --------------------------------------------------------------


def test_objectives_are_off_unless_they_are_asked_for():
    report = synthesize_object(
        ObjectGoal(FIXTURE, "TimerLike"),
        max_setup_steps=0,
        hypothesis_examples=10,
        max_method_probes=8,
        budget=SaturationBudget(max_seconds=20),
    )
    assert report.states == ()
    assert report.negatives == ()
    assert report.coverage is None
    assert report.line_coverage is None


@pytest.mark.parametrize("rung", RUNGS)
def test_the_objectives_reach_states_branches_and_bad_sequences(rung):
    """One node per objective of the ladder (tests/avengers_rungs.py): each runs the production portfolio with that objective
    only and asserts its part of what the single search used to assert together."""
    executor = IsolatedExecutor(trace=True)
    report = synthesize_object(
        ObjectGoal(FIXTURE, "TimerLike"),
        objectives=True,
        executor=executor,
        passes=passes_for(rung, executor=executor, max_setup_steps=1, hypothesis_examples=10),
        max_setup_steps=1,
        hypothesis_examples=10,
        max_method_probes=16,
        budget=SaturationBudget(max_seconds=90),
    )
    if rung == "states":
        assert len(report.states) >= 2, "TimerLike() and TimerLike(); start() differ"
    elif rung == "coverage":
        assert report.covered_methods == ("start", "stop")
        assert report.branch_coverage == 1.0
    else:
        shapes = {item.shape for item in report.negatives}
        assert "repeated" in shapes
        assert any(
            [call.name for call in item.calls] == ["start", "start"]
            for item in report.negatives
        )


# -- states have to reproduce to be counted ----------------------------------


def test_a_state_whose_digest_does_not_reproduce_is_named_by_its_recipe():
    from tharkun.avengers.objectives import ReceiverState

    goal = ObjectGoal(FIXTURE, "TimerLike")
    recipe = ConstructionRecipe(goal.key, ConstructorCall())
    stable = ReceiverState(goal, recipe, "d1", stable=True)
    unstable = ReceiverState(goal, recipe, "d1", stable=False)
    assert stable.key == "d1"
    assert unstable.key == f"unstable:{recipe.key}"
    # Two recipes reaching one stable digest are ONE state; two recipes with
    # unreproducible digests are two, because nothing links them.
    other = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"logger": 1}))
    assert ReceiverState(goal, other, "d1", stable=True).key == stable.key
    assert ReceiverState(goal, other, "d1", stable=False).key != unstable.key


def test_a_receiver_holding_a_timestamp_is_counted_once_not_once_per_replay(tmp_path):
    module = tmp_path / "clocked.py"
    module.write_text(
        "import time\n"
        "\n"
        "\n"
        "class Clocked:\n"
        "    def __init__(self):\n"
        "        self.started = None\n"
        "\n"
        "    def start(self):\n"
        "        self.started = time.perf_counter()\n"
    )
    path = os.path.realpath(str(module))
    goal = ObjectGoal(path, "Clocked")
    recipe = ConstructionRecipe(goal.key, ConstructorCall(), (MethodCall("start"),))
    outcome = TRACING.replay(goal, recipe)
    assert outcome.succeeded and outcome.state_digest is not None
    # The premise: the same recipe really does digest differently each time.
    assert TRACING.replay(goal, recipe).state_digest != outcome.state_digest

    kb = _kb_with(ValidatedRecipe(goal, recipe, outcome))
    states = [
        item.value
        for item in StateDiversityPass(TRACING, max_steps=0).run(kb, _delta(kb))
        if item.kind == STATE
    ]
    assert len(states) == 1
    assert states[0].stable is False


def test_a_reproducible_state_is_reported_as_stable():
    goal = ObjectGoal(FIXTURE, "Box")
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 5}))
    outcome = TRACING.replay(goal, recipe)
    kb = _kb_with(ValidatedRecipe(goal, recipe, outcome))
    states = [
        item.value
        for item in StateDiversityPass(TRACING, max_steps=0).run(kb, _delta(kb))
        if item.kind == STATE
    ]
    assert len(states) == 1
    assert states[0].stable is True
    assert states[0].key == outcome.state_digest
