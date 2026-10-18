from __future__ import annotations

import os
import pickle
import signal
import sys
import time

from tharkun.avengers.executor import IsolatedExecutor
from tharkun.avengers.flatting import FlatObjectTemplate
from tharkun.avengers.model import (
    OBJECT,
    ConstructionRecipe,
    ConstructorCall,
    FunctionCall,
    MethodCall,
    ObjectGoal,
)


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "avengers_target.py")

UNPICKLED = False


def _mark_unpickled():
    global UNPICKLED
    UNPICKLED = True


class _PickleTripwire:
    def __reduce__(self):
        return (_mark_unpickled, ())


def test_valid_constructor_and_boundary_calls_return_normally():
    goal = ObjectGoal(FIXTURE, "Box", (MethodCall("width"),))
    recipe = ConstructionRecipe(
        goal.key,
        ConstructorCall(kwargs={"lower": 1, "upper": 2}),
    )
    outcome = IsolatedExecutor().replay(goal, recipe)
    assert outcome.succeeded
    assert outcome.return_types == ("builtins.int",)


def test_invalid_constructor_is_a_rejection_with_phase():
    goal = ObjectGoal(FIXTURE, "Box")
    recipe = ConstructionRecipe(
        goal.key,
        ConstructorCall(kwargs={"lower": 2, "upper": 1}),
    )
    outcome = IsolatedExecutor().replay(goal, recipe)
    assert outcome.kind == "raised"
    assert outcome.phase == "construct"
    assert outcome.exception_type == "builtins.ValueError"


def test_setup_can_prepare_a_stateful_object_for_a_boundary_call():
    goal = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("stop"),))
    unprepared = ConstructionRecipe(goal.key)
    assert IsolatedExecutor().replay(goal, unprepared).phase == "boundary[0]"

    prepared = ConstructionRecipe(goal.key, setup=(MethodCall("start"),))
    assert IsolatedExecutor().replay(goal, prepared).succeeded


def test_native_crash_is_contained_and_classified():
    goal = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("hard_crash"),))
    outcome = IsolatedExecutor().replay(goal, ConstructionRecipe(goal.key))
    assert outcome.kind == "crashed"
    assert outcome.signal == signal.SIGSEGV


def test_a_spoofed_success_file_cannot_conceal_a_later_native_crash():
    goal = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("spoof_then_crash"),))
    outcome = IsolatedExecutor().replay(goal, ConstructionRecipe(goal.key))

    assert outcome.kind == "crashed"
    assert outcome.signal == signal.SIGSEGV


def test_timeout_is_contained_and_classified():
    goal = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("hang"),))
    outcome = IsolatedExecutor(timeout_s=0.05).replay(goal, ConstructionRecipe(goal.key))
    assert outcome.kind == "timeout"


def test_target_stdout_cannot_corrupt_the_worker_protocol():
    goal = ObjectGoal(FIXTURE, "Noisy", (MethodCall("speak"),))
    assert IsolatedExecutor().replay(goal, ConstructionRecipe(goal.key)).succeeded


def test_free_function_boundary_receives_the_constructed_object_reference():
    goal = ObjectGoal(
        FIXTURE,
        "Box",
        (FunctionCall("classify_box", (OBJECT,), {"minimum": 0}),),
    )
    recipe = ConstructionRecipe(
        goal.key,
        ConstructorCall(kwargs={"lower": -1, "upper": 2}),
    )
    outcome = IsolatedExecutor().replay(goal, recipe)
    assert outcome.succeeded
    assert outcome.return_types == ("builtins.str",)


def test_worker_response_reader_never_unpickles_child_controlled_bytes(tmp_path):
    global UNPICKLED
    UNPICKLED = False
    response = tmp_path / "response.json"
    response.write_bytes(pickle.dumps(_PickleTripwire()))

    assert IsolatedExecutor._read_response(str(response)) is None
    assert UNPICKLED is False


def test_worker_response_reader_rejects_pathologically_nested_json(tmp_path):
    response = tmp_path / "response.json"
    response.write_bytes(b"[" * 2000 + b"]" * 2000)

    assert IsolatedExecutor._read_response(str(response)) is None


def test_data_only_wire_roundtrips_a_flat_template(tmp_path):
    from tharkun.avengers._wire import encode_response

    template = FlatObjectTemplate(
        recipe_blob=b"{}",
        recipe_sha="a" * 64,
        total_budget=2,
        n_leaves=1,
        baseline_data=b"\x00\x00",
        exemplar_data=b"\x01\x02",
        exemplar_json="{}",
        leaf_paths=("$.value",),
    )
    response = tmp_path / "response.json"
    response.write_bytes(encode_response(template))

    assert IsolatedExecutor._read_response(str(response)) == template


def test_mine_template_rejects_a_wire_valid_but_invalid_recipe(monkeypatch):
    template = FlatObjectTemplate(
        recipe_blob=b"{}",
        recipe_sha="a" * 64,
        total_budget=2,
        n_leaves=1,
        baseline_data=b"\x00\x00",
        exemplar_data=b"\x01\x02",
        exemplar_json="{}",
        leaf_paths=("$.value",),
    )
    monkeypatch.setattr(IsolatedExecutor, "_invoke", lambda *args: template)
    goal = ObjectGoal(FIXTURE, "Box")

    outcome = IsolatedExecutor().mine_template(goal, ConstructionRecipe(goal.key))

    assert outcome.kind == "worker-error"
    assert outcome.phase == "flatting"
    assert "invalid template response" in outcome.detail


# -- coverage across the boundary --------------------------------------------


def test_a_traced_replay_reports_the_lines_and_arcs_it_executed():
    goal = ObjectGoal(FIXTURE, "Box", (MethodCall("width"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 5}))
    outcome = IsolatedExecutor(trace=True).replay(goal, recipe)
    assert outcome.succeeded
    assert outcome.lines, "a traced replay reports the lines it ran"
    assert outcome.arcs, "a traced replay reports the transitions it took"
    assert all(isinstance(line, int) for line in outcome.lines)
    assert all(len(arc) == 2 for arc in outcome.arcs)


def test_tracing_is_off_unless_it_is_asked_for():
    goal = ObjectGoal(FIXTURE, "Box", (MethodCall("width"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 5}))
    outcome = IsolatedExecutor().replay(goal, recipe)
    assert outcome.succeeded
    assert outcome.lines == ()
    assert outcome.arcs == ()


def test_a_failing_recipe_still_reports_what_it_managed_to_execute():
    goal = ObjectGoal(FIXTURE, "Box", (MethodCall("width"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 5, "upper": 1}))
    outcome = IsolatedExecutor(trace=True).replay(goal, recipe)
    assert not outcome.succeeded
    # It reached the guard and took the raising side; that is how a caller
    # tells "raised on the way in" from "never got started".
    assert outcome.lines


def test_different_constructor_arguments_take_different_arcs():
    goal = ObjectGoal(FIXTURE, "Box", (MethodCall("width"),))
    executor = IsolatedExecutor(trace=True)
    good = executor.replay(
        goal, ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 5}))
    )
    bad = executor.replay(
        goal, ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 5, "upper": 1}))
    )
    assert set(good.arcs) != set(bad.arcs)


def test_a_crash_cause_groups_by_where_and_why():
    goal = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("hard_crash"),))
    outcome = IsolatedExecutor().replay(goal, ConstructionRecipe(goal.key))
    assert outcome.crash_cause == ("worker", "signal:11")

    raised = IsolatedExecutor().replay(
        ObjectGoal(FIXTURE, "Box", (MethodCall("width"),)),
        ConstructionRecipe("k", ConstructorCall(kwargs={"lower": 5, "upper": 1})),
    )
    assert raised.crash_cause == ("construct", "builtins.ValueError")


def test_a_normal_return_has_no_crash_cause():
    goal = ObjectGoal(FIXTURE, "Box", (MethodCall("width"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 5}))
    assert IsolatedExecutor().replay(goal, recipe).crash_cause is None


# -- batched replay ----------------------------------------------------------


def _box_items(count: int):
    goal = ObjectGoal(FIXTURE, "Box", (MethodCall("width"),))
    recipe = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 5}))
    return [(goal, recipe) for _ in range(count)]


def test_a_batch_returns_one_outcome_per_recipe_in_order():
    goal = ObjectGoal(FIXTURE, "Box", (MethodCall("width"),))
    good = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 1, "upper": 5}))
    bad = ConstructionRecipe(goal.key, ConstructorCall(kwargs={"lower": 5, "upper": 1}))
    outcomes = IsolatedExecutor().replay_many([(goal, good), (goal, bad), (goal, good)])
    assert [item.kind for item in outcomes] == ["returned", "raised", "returned"]


def test_an_empty_batch_is_an_empty_result():
    assert IsolatedExecutor().replay_many([]) == ()


def test_a_batch_of_one_is_a_plain_replay():
    outcomes = IsolatedExecutor().replay_many(_box_items(1))
    assert len(outcomes) == 1 and outcomes[0].succeeded


def test_a_crash_inside_a_batch_is_attributed_to_the_recipe_that_caused_it():
    live = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("start"),))
    crashing = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("hard_crash"),))
    # timeout_s is only a ceiling here: a crash is reported when the child dies; under load the default one turned the crash into a
    # 'timeout'
    outcomes = IsolatedExecutor(timeout_s=60.0).replay_many(
        [
            (live, ConstructionRecipe(live.key)),
            (crashing, ConstructionRecipe(crashing.key)),
            (live, ConstructionRecipe(live.key)),
        ]
    )
    # The crash takes the whole child down, so the batch cannot be believed.
    # Re-running it one child per recipe is what keeps the other two honest.
    assert [item.kind for item in outcomes] == ["returned", "crashed", "returned"]
    assert outcomes[1].signal == signal.SIGSEGV


def test_a_hang_inside_a_batch_is_attributed_to_the_recipe_that_caused_it():
    # [internal ref] part 3: was timeout_s=0.4. A hang always makes the shared batch
    # invocation time out (0.4 * 2 items = 0.8s, dwarfed by hang()'s 60s
    # sleep), so replay_many() always falls back to one fresh child per
    # recipe, each again budgeted at timeout_s alone. That fallback child for
    # "live" must fork/exec a whole Python interpreter, import tharkun and the
    # target module, unpickle, run .start(), and pickle+write the response,
    # all within timeout_s -- fine in isolation, but a real captured failure
    # under host contention (a busy merge-result campaign run) showed both
    # items independently timing out (['timeout', 'timeout'] != ['returned',
    # 'timeout']), not misattribution or a swap. 2.0s gives that fresh
    # interpreter real headroom while staying far below hang()'s 60s, so the
    # hang is still always correctly identified as "timeout".
    live = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("start"),))
    hanging = ObjectGoal(FIXTURE, "TimerLike", (MethodCall("hang"),))
    outcomes = IsolatedExecutor(timeout_s=2.0).replay_many(
        [(live, ConstructionRecipe(live.key)), (hanging, ConstructionRecipe(hanging.key))]
    )
    assert [item.kind for item in outcomes] == ["returned", "timeout"]


def test_a_batch_is_faster_than_one_child_per_recipe():
    items = _box_items(6)
    executor = IsolatedExecutor()

    start = time.monotonic()
    executor.replay_many(items)
    batched = time.monotonic() - start

    start = time.monotonic()
    for goal, recipe in items:
        executor.replay(goal, recipe)
    separate = time.monotonic() - start

    assert batched < separate, (batched, separate)


def test_a_traced_batch_reports_coverage_for_every_recipe():
    outcomes = IsolatedExecutor(trace=True).replay_many(_box_items(3))
    assert all(item.lines for item in outcomes)


def test_a_real_sv_task_file_loads_and_constructs(tmp_path):
    """A real SV-COMP task always does ``import _sv_verifier`` at module scope
    ([internal ref], W57 hit this): without a stub for it, the isolated executor's own
    module load fails before Avengers ever sees the target's classes at all.
    _worker.py::_load_class installs a zero-value stand-in (never the
    harness's real model, never a real draw -- Avengers only ever constructs
    and drives an object in isolation) so a goal built from a real SV task
    file can be tried like any other target."""
    from tharkun.avengers import ObjectGoal, SaturationBudget, synthesize_object

    target = tmp_path / "sv_task.py"
    target.write_text(
        "import _sv_verifier\n"
        "\n"
        "class Box:\n"
        "    def __init__(self, x: int):\n"
        "        self.x = x\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    n = _sv_verifier.nondet_int()\n"
        "    b = Box(n)\n"
        "    print(b.x)\n"
    )
    goal = ObjectGoal(str(target), "Box")
    report = synthesize_object(goal, budget=SaturationBudget(max_seconds=5.0))
    assert report.validated, "Avengers should construct Box despite the target's own _sv_verifier import"


class TestSvVerifierStubIsolation:
    """MG review (PR 104): does avengers/_worker.py's own _sv_verifier stub
    leak the way PR 101's sys.modules["_sv_verifier"] leak did?

    No -- a DIFFERENT, safe mechanism: _load_class installs a FILE on
    sys.path and lets the real `import _sv_verifier` machinery run (the
    target's own `import _sv_verifier` statement), so sys.modules gets a
    REAL, spec-bearing module object, never a raw sys.modules[...] = stub
    assignment the way sv/_child.py's/_probe.py's/_concolic_worker.py's
    pre-fix code did (PR 101's bug). It is also one-shot per subprocess in
    normal use (_worker.py::main handles exactly one request then exits),
    unlike this fixture-only direct-call test below, which drives the
    installer function itself to prove it is safe even called repeatedly
    in-process and even after a stale, spec-less leftover from some OTHER
    caller -- the same defensive case PR 101 added for
    sv/avengers_seed.py::_sv_verifier_stub_on_path."""

    def test_a_stale_specless_module_does_not_crash_the_stub_installer(self, monkeypatch):
        import types

        from tharkun.avengers import _worker

        monkeypatch.setattr(_worker, "_sv_verifier_stub_dir", None)
        stale = types.ModuleType("_sv_verifier")
        monkeypatch.setitem(sys.modules, "_sv_verifier", stale)
        _worker._sv_verifier_stub_on_path()  # must not raise
        sys.modules.pop("_sv_verifier", None)

    def test_repeated_in_process_calls_do_not_leave_a_specless_module(self, monkeypatch):
        from tharkun.avengers import _worker

        monkeypatch.setattr(_worker, "_sv_verifier_stub_dir", None)
        sys.modules.pop("_sv_verifier", None)
        _worker._sv_verifier_stub_on_path()
        _worker._sv_verifier_stub_on_path()  # cached path, must not raise or double-install
        # unlike the sv/_child.py-style leak, this installer never assigns a
        # raw sys.modules[...] entry itself -- nothing to restore here, and
        # nothing left behind for a later caller to trip over
        assert "_sv_verifier" not in sys.modules
        sys.modules.pop("_sv_verifier", None)
