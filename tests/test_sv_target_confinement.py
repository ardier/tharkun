"""Target execution routes share confinement, including solver handoffs."""
from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
import textwrap
import threading
from types import SimpleNamespace

import pytest

from tests import test_sv_fs_confinement as fs_controls
from tests import test_sv_spawn_confinement as spawn_controls
from tests import test_sv_hybrid_sandbox as write_controls
from tests.test_sv_fs_confinement import outside  # noqa: F401 -- shared victim fixture
from tharkun.hybrid.fdp_codec import atheris_available
from tharkun.sv import concolic, engine, prove


def program(tmp_path, source):
    path = tmp_path / "target.py"
    path.write_text(textwrap.dedent(source))
    return str(path)


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
@pytest.mark.parametrize("control", ["fs", "spawn", "fileio", "truncate"])
def test_original_hybrid_batteries_with_full_budget(tmp_path, outside, monkeypatch, control):
    # Call the original battery, including every assertion, with only its budgets changed.
    module, name = {
        "fs": (fs_controls, "test_the_hybrid_worker_cannot_mutate_the_file_system_outside_the_scratch_either"),
        "spawn": (spawn_controls, "test_the_hybrid_worker_cannot_spawn_processes_either"),
        "fileio": (write_controls, "test_a_program_that_truncates_a_file_it_does_not_own_leaves_it_alone"),
        "truncate": (write_controls, "test_os_level_and_pathlib_writes_outside_the_scratch_are_refused_too"),
    }[control]
    verify = module.verify
    calls = []

    def full_budget(*args, **kwargs):
        kwargs.update(budget_seconds=60, run_timeout=30)
        calls.append(kwargs)
        return verify(*args, **kwargs)

    monkeypatch.setattr(module, "verify", full_budget)
    args = (tmp_path, outside) if control in ("fs", "spawn") else (tmp_path,)
    getattr(module, name)(*args)
    assert len(calls) == 1


@pytest.mark.parametrize("effect", ["filesystem", "spawn", "thread"])
def test_direct_concolic_execution_confines_effects(tmp_path, effect):
    victim = tmp_path / "victim"
    victim.write_text("keep")
    action = {
        "filesystem": f"os.unlink({str(victim)!r})",
        "spawn": f"subprocess.run(['/bin/rm', {str(victim)!r}], check=True)",
        "thread": f"threading.Thread(target=lambda: os.unlink({str(victim)!r})).start()",
    }[effect]
    path = program(tmp_path, f"""
        import os, subprocess, threading
        import _sv_verifier
        if __name__ == '__main__':
            x = _sv_verifier.nondet_int()
            try:
                {action}
            except OSError:
                pass
            if x == 12345:
                raise ValueError
    """)
    solved = concolic.solve(path, ("ValueError",), budget_seconds=30, symbolic=False)
    assert victim.read_text() == "keep"
    assert solved.ran, solved
    assert set(solved.flags) & {"file_unmodelled", "threads", "import_state_unmodelled"}, solved
    assert not solved.closed
    for candidate in solved.candidates:
        confirmed, _ = engine.replay(path, {"vector": candidate.vector}, ("ValueError",), timeout=30)
        assert not confirmed


def test_concolic_positive_vector_replays_on_original(tmp_path):
    path = program(tmp_path, """
        import _sv_verifier
        if __name__ == '__main__':
            x = _sv_verifier.nondet_int()
            if x > 100000:
                raise KeyError
    """)
    solved = concolic.solve(path, ("KeyError",), budget_seconds=30, symbolic=False)
    assert solved.candidates, solved
    candidate = solved.candidates[0]
    assert candidate.vector[0]["value"] > 100000
    confirmed, out = engine.replay(path, {"vector": candidate.vector}, ("KeyError",), timeout=30)
    assert confirmed and out.violated(("KeyError",)) == "KeyError"


@pytest.mark.parametrize("case", ["valid", "changed-harness", "unsafe-original"])
def test_generated_driver_source_binding_keeps_original_admission(tmp_path, case):
    from tharkun.driver import drive_to_line
    from tharkun.solve.environment import confined

    source = "def f(x):\n    if x > 1:\n        return 42\n"
    original = program(tmp_path, source)
    generated = "import operator as generated_helper\n" + source
    harness = tmp_path / "generated.py"
    harness.write_text(generated + ("# changed\n" if case == "changed-harness" else ""))
    admitted = "import operator\n" + source if case == "unsafe-original" else source
    entered = []

    def confine():
        entered.append(True)
        return confined([original, str(harness)], threads=False)

    result = drive_to_line(str(harness), "f", 4, {"x": 2}, max_rounds=1,
                           confine=confine, _source_binding=(generated, original, admitted))
    if case == "valid":
        assert entered and result.status == "reached", result
    else:
        assert not entered, "refused source entered a target phase"
        assert result.status == "no_target", result
        assert ("changed after admission" if case == "changed-harness" else "reversible execution") in result.diagnostics


@pytest.mark.parametrize("collision", [
    "maker = __sv_op.attrgetter('__' + 'class' + '__')(int)",
    "__sv_op = None",
    "def __sv_draw(*args): return 0",
])
def test_concolic_refuses_original_scaffold_names_before_launch(tmp_path, monkeypatch, collision):
    path = program(tmp_path, "import _sv_verifier\n" + collision + "\n"
                   "if __name__ == '__main__':\n    x = _sv_verifier.nondet_int()\n"
                   "    if x > 100000:\n        raise KeyError\n")

    def launch(*args, **kwargs):
        pytest.fail("a scaffold collision must be refused before the worker starts")

    monkeypatch.setattr(concolic.subprocess, "run", launch)
    solved = concolic.solve(path, ("KeyError",), budget_seconds=30, symbolic=False)
    assert not solved.ran and "import_state_unmodelled" in solved.flags


@pytest.mark.parametrize("source, reserved", [
    ("__sv_user_value = 1\n", None),
    ("import math as __sv_op\n", "__sv_op"),
    ("def f(__sv_op): pass\n", "__sv_op"),
])
def test_harness_namespace_reserves_only_injected_globals(source, reserved):
    from tharkun.sv.sites import harness_namespace_refusal

    reason = harness_namespace_refusal(source)
    if reserved is None:
        assert reason is None
    else:
        assert reason and reserved in reason


def test_concolic_worker_refuses_scaffold_collision_before_target_phase(tmp_path, monkeypatch):
    from tharkun.sv import _concolic_worker

    original = "maker = __sv_op.attrgetter('__' + 'class' + '__')(int)\n"
    path = program(tmp_path, original)
    harness = tmp_path / "harness.py"
    harness.write_text("pass\n")
    output = tmp_path / "out.json"

    def target_phase(*args, **kwargs):
        pytest.fail("a scaffold collision entered a target phase")

    monkeypatch.setattr(_concolic_worker, "_stream_lens", target_phase)
    _concolic_worker._run_request(str(tmp_path / "request.json"), {
        "files": [path], "harness": str(harness), "scratch": str(tmp_path), "out": str(output),
        "source_binding": ["pass\n", path, original],
    })
    result = json.loads(output.read_text())
    assert result["flags"] == ["import_state_unmodelled"]
    assert "__sv_op" in result["refusal"]


@pytest.mark.parametrize("case", ["missing", "null", "scalar", "short", "non-string", "relative"])
def test_concolic_worker_refuses_invalid_binding_before_stub_installation(tmp_path, monkeypatch, case):
    from tharkun.sv import _concolic_worker

    path = program(tmp_path, "pass\n")
    output = tmp_path / "out.json"
    request = {"harness": path, "out": str(output), "program_dir": str(tmp_path)}
    invalid = {"null": None, "scalar": "abc", "short": ["pass\n", path],
               "non-string": ["pass\n", path, 7], "relative": ["pass\n", "target.py", "pass\n"]}
    if case != "missing":
        request["source_binding"] = invalid[case]
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))

    def target_phase(*args, **kwargs):
        pytest.fail("an invalid source binding reached stub or target setup")

    monkeypatch.setattr(_concolic_worker, "_stub", target_phase)
    monkeypatch.setattr(_concolic_worker, "_stream_lens", target_phase)
    _concolic_worker.main(str(request_path))
    result = json.loads(output.read_text())
    assert result["flags"] == ["import_state_unmodelled"]
    assert result["reached"] == [] and result["solver_calls"] == 0
    assert result["refusal"] == "missing or malformed source binding"


@pytest.mark.parametrize("starts_thread", [False, True])
def test_hybrid_plateau_enters_concolic_with_main_concolic_disabled(tmp_path, monkeypatch, starts_thread):
    from tharkun.sv import hybrid

    victim = tmp_path / "victim"
    victim.write_text("keep")
    path = program(tmp_path, """
        import _sv_verifier
        if __name__ == '__main__':
            x = _sv_verifier.nondet_int()
            assert x != 123456789
    """)
    if starts_thread:
        with open(path) as stream:
            source = stream.read()
        with open(path, "w") as stream:
            stream.write(source.replace("    assert", "    import os, threading\n"
                         "    try:\n"
                         f"        threading.Thread(target=lambda: os.unlink({str(victim)!r})).start()\n"
                         "    except OSError: pass\n    assert"))
    events = []

    class Plateau:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            events.append("fuzzer")

        def is_running(self):
            return False

        def cpu_seconds(self):
            return 0.0

        def coverage(self):
            return 0

        def stop(self):
            pass

    original = concolic.solve

    def solve(*args, **kwargs):
        events.append("concolic")
        assert kwargs["site_kind"] == "obligation"
        assert path in kwargs["files"]
        return original(*args, **kwargs)

    monkeypatch.setattr(hybrid, "atheris_available", lambda: True)
    monkeypatch.setattr(hybrid, "AtherisBackend", Plateau)
    monkeypatch.setattr(concolic, "solve", solve)
    result = engine.verify(path, ("AssertionError",), budget_seconds=60, run_timeout=30,
                           jobs=1, prove=False, concolic=False, stages=["hybrid"])
    assert events[:2] == ["fuzzer", "concolic"], events
    assert victim.read_text() == "keep"
    if starts_thread:
        assert result.verdict != "false", result
    else:
        assert result.verdict == "false", result
        assert engine.replay(path, result.witness, ("AssertionError",), timeout=30)[0]


@pytest.mark.parametrize("phase", ["import", "constructor", "bool_hook"])
def test_concolic_effects_before_guard_are_confined(tmp_path, phase):
    victim = tmp_path / "victim"
    victim.write_text("keep")
    effect = f"\n    try: os.unlink({str(victim)!r})\n    except OSError: pass\n"
    definitions = "import os\nimport _sv_verifier\ndef mutate():" + effect
    if phase == "import":
        definitions += "mutate()\n"
        call = "pass"
    elif phase == "constructor":
        definitions += "class Box:\n    def __init__(self): mutate()\n"
        call = "Box()"
    else:
        definitions += "class Box:\n    def __bool__(self):\n        mutate()\n        return True\n"
        call = "bool(Box())"
    path = program(tmp_path, definitions + "if __name__ == '__main__':\n"
                   "    x = _sv_verifier.nondet_int()\n" + f"    {call}\n"
                   "    if x == 12345: raise ValueError\n")
    solved = concolic.solve(path, ("ValueError",), budget_seconds=30, symbolic=False)
    assert victim.read_text() == "keep"
    assert solved.ran and solved.flags, solved
    assert not solved.closed


@pytest.mark.parametrize("effect", ["filesystem", "spawn", "thread"])
def test_probe_default_files_confine_target_and_stationary_confirmation(tmp_path, monkeypatch, effect):
    victim = tmp_path / "victim"
    victim.write_text("keep")
    action = {
        "filesystem": f"os.unlink({str(victim)!r})",
        "spawn": f"subprocess.run(['/bin/rm', {str(victim)!r}])",
        "thread": f"threading.Thread(target=lambda: os.unlink({str(victim)!r})).start()",
    }[effect]
    path = program(tmp_path, f"""
        import os, subprocess, threading
        try:
            {action}
        except OSError:
            pass
        raise AssertionError
    """)
    row = prove.run_probe(sys.executable, path, [], [], [], timeout=30)
    assert row["status"] == "raised", row
    assert "file_unmodelled" in row["flags"]
    assert victim.read_text() == "keep"
    # Force the confirmation branch independently of probe flags, then use a real child.
    monkeypatch.setattr(prove, "run_probe", lambda *args, **kwargs: {
        "status": "raised", "exc": "AssertionError", "mro": ["AssertionError"], "draws": [], "flags": [],
    })
    replayed = []
    original = engine.run_once

    def replay(*args, **kwargs):
        replayed.append(kwargs["files"])
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "run_once", replay)
    verdict = prove._stationary(path, ("AssertionError",),
                               SimpleNamespace(loops=[], draws=[], output_files=False),
                               python=sys.executable, remaining=lambda: 60, run_timeout=30,
                               memory_bytes=None, jobs=1, say=lambda message: None)
    assert replayed == [[path]]
    assert verdict is None
    assert victim.read_text() == "keep"


def test_nondet_drive_restores_host_and_allows_declared_helper_and_scratch(tmp_path):
    from tharkun.sv.nondet_harness import drive

    helper = tmp_path / "confinement_helper.py"
    helper.write_text("VALUE = 42\n")
    path = program(tmp_path, """
        import confinement_helper
        with open('scratch.txt', 'w') as stream:
            stream.write(str(confinement_helper.VALUE))
        with open('scratch.txt') as stream:
            assert stream.read() == '42'
        raise KeyError
    """)
    before = (os.getcwd(), builtins.open, os.unlink, subprocess.Popen, threading.Thread.start, sys.gettrace())
    out = drive(path, [], files=[path, str(helper)])
    assert out.exc == "KeyError", out
    assert not (set(out.flags) - {"file_effect"}), out
    assert (os.getcwd(), builtins.open, os.unlink, subprocess.Popen, threading.Thread.start, sys.gettrace()) == before
    assert not (tmp_path / "scratch.txt").exists()
    sys.modules.pop("confinement_helper", None)


@pytest.mark.parametrize("effect", ["filesystem", "spawn", "thread"])
def test_avengers_worker_and_parent_materialization_are_confined(tmp_path, monkeypatch, effect):
    from tharkun import avengers
    from tharkun.driver import _avengers_object_for_bounds
    from tharkun.solve.environment import confined

    victim = tmp_path / "victim"
    victim.write_text("keep")
    action = {
        "filesystem": f"os.unlink({str(victim)!r})",
        "spawn": f"subprocess.run(['/bin/rm', {str(victim)!r}])",
        "thread": f"threading.Thread(target=lambda: os.unlink({str(victim)!r})).start()",
    }[effect]
    path = program(tmp_path, f"""
        import os, subprocess, threading
        class Box:
            def __init__(self):
                try: {action}
                except OSError: pass
                self.value = 7
    """)
    goal = avengers.ObjectGoal(path, "Box")
    recipe = avengers.ConstructionRecipe(goal.key)
    executor = avengers.IsolatedExecutor(files=(path,), timeout_s=30)
    outcome = executor.replay(goal, recipe)
    assert outcome.kind == "unmodelled", outcome
    assert victim.read_text() == "keep"
    calls = []

    def synthesize(*args, **kwargs):
        calls.append(kwargs["executor"].files)
        return SimpleNamespace(validated=[SimpleNamespace(recipe=recipe)])

    monkeypatch.setattr(avengers, "synthesize_object", synthesize)
    obj = _avengers_object_for_bounds(path, "Box", (avengers.FieldBound("value", "eq", 7),),
                                      budget_seconds=1, confine=lambda: confined([path], threads=False))
    assert calls == [(path,)]
    assert obj is None
    assert victim.read_text() == "keep"


def test_avengers_guarded_worker_constructs_a_safe_declared_class(tmp_path):
    from tharkun import avengers

    path = program(tmp_path, """
        class Widget:
            def __init__(self, value: int):
                self.value = value
    """)
    goal = avengers.ObjectGoal(path, "Widget")
    recipe = avengers.ConstructionRecipe(goal.key, avengers.ConstructorCall(args=(7,)))
    outcome = avengers.IsolatedExecutor(files=(path,), timeout_s=30).replay(goal, recipe)
    assert outcome.succeeded, outcome


def test_pynguin_confined_bootstrap_and_subprocess_refusal(tmp_path, monkeypatch):
    from tharkun.avengers import ObjectGoal
    from tharkun.avengers import backends

    path = program(tmp_path, "class Box: pass\n")
    victim = tmp_path / "victim"
    victim.write_text("keep")
    requests = []
    monkeypatch.setattr(backends, "pynguin_available", lambda interpreter: True)
    fake = tmp_path / "pynguin.py"
    fake.write_text(textwrap.dedent(f"""
        import os, subprocess, threading
        refused = 0
        for action in (lambda: os.unlink({str(victim)!r}),
                       lambda: subprocess.run(['/bin/rm', {str(victim)!r}]),
                       lambda: threading.Thread(target=lambda: None).start()):
            try:
                action()
            except OSError:
                refused += 1
        assert refused == 3, refused
        print('CONFINED BOOTSTRAP')
    """))
    original_run = subprocess.run

    def launch(command, **kwargs):
        assert command[1:3] == ["-m", "tharkun.avengers._pynguin_worker"]
        with open(command[3]) as stream:
            request = json.load(stream)
        requests.append(request)
        assert "--nosubprocess" in request["argv"]
        kwargs["env"]["PYTHONPATH"] = os.pathsep.join([str(tmp_path), *sys.path])
        result = original_run(command, **kwargs)
        assert result.returncode == 0, result.stderr
        assert b"CONFINED BOOTSTRAP" in result.stdout
        return result

    monkeypatch.setattr(backends.subprocess, "run", launch)
    goal = ObjectGoal(path, "Box")
    assert backends.PynguinBackendPass(files=(path, str(fake)), timeout_s=30).generate(goal) == ()
    assert len(requests) == 1
    assert backends.PynguinBackendPass(files=(path,), subprocess_mode=True).generate(goal) == ()
    assert len(requests) == 1
    assert victim.read_text() == "keep"


def test_pynguin_collects_generated_tests_without_staged_inputs(tmp_path, monkeypatch):
    from pathlib import Path
    from tharkun.avengers import ObjectGoal
    from tharkun.avengers import backends

    path = program(tmp_path, "class Box: pass\n")
    generated_source = "def test_generated():\n    assert True\n"
    monkeypatch.setattr(backends, "pynguin_available", lambda interpreter: True)

    def launch(command, **kwargs):
        request = json.loads(Path(command[3]).read_text())
        scratch = Path(request["scratch"])
        output = Path(request["argv"][request["argv"].index("--output-path") + 1])
        assert output != scratch and output.parent == scratch
        (scratch / "staged_input.py").write_text("class Input: pass\n")
        (output / "test_generated.py").write_text(generated_source)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(backends.subprocess, "run", launch)
    artifacts = backends.PynguinBackendPass(files=(path,)).generate(ObjectGoal(path, "Box"))
    assert [(artifact.tool, artifact.source) for artifact in artifacts] == [("pynguin", generated_source)]


@pytest.mark.parametrize("effect", ["filesystem", "spawn", "thread"])
def test_in_process_drive_refuses_effects_and_restores_patches(tmp_path, effect):
    from tharkun.sv.nondet_harness import drive

    victim = tmp_path / "victim"
    victim.write_text("keep")
    action = {
        "filesystem": f"os.unlink({str(victim)!r})",
        "spawn": f"subprocess.run(['/bin/rm', {str(victim)!r}])",
        "thread": f"threading.Thread(target=lambda: os.unlink({str(victim)!r})).start()",
    }[effect]
    path = program(tmp_path, f"""
        import os, subprocess, threading
        try:
            {action}
        except OSError:
            pass
        raise KeyError
    """)
    before = (os.getcwd(), builtins.open, os.unlink, subprocess.Popen, threading.Thread.start)
    out = drive(path, [])
    assert out.exc == "KeyError", out
    assert "file_unmodelled" in out.flags
    assert out.violated(("KeyError",)) is None
    assert victim.read_text() == "keep"
    assert (os.getcwd(), builtins.open, os.unlink, subprocess.Popen, threading.Thread.start) == before


def test_avengers_parent_materialization_keeps_safe_objects_usable(tmp_path, monkeypatch):
    from tharkun import avengers
    from tharkun.driver import _avengers_object_for_bounds
    from tharkun.solve.environment import confined

    path = program(tmp_path, """
        class Box:
            def __init__(self, value):
                self.value = value
            def doubled(self):
                return self.value * 2
    """)
    goal = avengers.ObjectGoal(path, "Box")
    recipe = avengers.ConstructionRecipe(goal.key, avengers.ConstructorCall(args=(7,)))
    calls = []

    def synthesize(candidate_goal, **kwargs):
        calls.append((candidate_goal.class_name, kwargs["executor"].files))
        return SimpleNamespace(validated=[SimpleNamespace(recipe=recipe)])

    monkeypatch.setattr(avengers, "synthesize_object", synthesize)
    obj = _avengers_object_for_bounds(path, "Box", (avengers.FieldBound("value", "eq", 7),),
                                      budget_seconds=1, confine=lambda: confined([path], threads=False))
    assert calls == [("Box", (path,))]
    assert obj is not None and obj.value == 7
    assert obj.doubled() == 14


@pytest.mark.parametrize("status", ["raised", "timeout", "crashed"])
def test_refused_drive_fallback_preserves_observations_without_claiming_reach(tmp_path, monkeypatch, status):
    from tharkun.sv import nondet_harness

    path = program(tmp_path, "class Deferred:\n    def __del__(self): pass\n")
    helper = tmp_path / "helper.py"
    helper.write_text("VALUE = 7\n")
    vector = [{"kind": "int", "value": 7}]
    child = engine.Outcome(status, exc="KeyError" if status == "raised" else None,
                           mro=["KeyError", "Exception"] if status == "raised" else [],
                           draws=vector, flags=["file_effect"], invalid_draw=True,
                           site={"file": "target.py", "line": 2})
    calls = []

    def host_patch(*args, **kwargs):
        pytest.fail("refused source must fall back before patching the host")

    def run_once(python, target, draws, **kwargs):
        calls.append((python, target, draws, kwargs))
        return child

    monkeypatch.setattr(engine, "run_once", run_once)
    monkeypatch.setattr(nondet_harness._child, "_patch_random", host_patch)
    out = nondet_harness.drive(path, vector, target_line=2, files=[str(helper), path, str(helper)])
    assert len(calls) == 1
    python, target, draws, kwargs = calls[0]
    assert (python, target, draws) == (sys.executable, path, vector)
    assert list(kwargs["files"]) == [path, str(helper)]
    assert kwargs["timeout"] == 30
    assert (out.status, out.exc, out.mro, out.draws, out.invalid_draw) == (
        child.status, child.exc, child.mro, child.draws, child.invalid_draw)
    assert {"file_effect", "import_state_unmodelled", "stack_unmodelled"} <= set(out.flags)
    assert not out.reached
    assert out.violated(("KeyError",)) is None


def test_refused_drive_fallback_keeps_deferred_cleanup_confined(tmp_path):
    from tharkun.sv.nondet_harness import drive

    victim = tmp_path / "victim"
    victim.write_text("keep")
    path = program(tmp_path, f"""
        import os
        class Deferred:
            def __del__(self, os=os):
                try: os.unlink({str(victim)!r})
                except OSError: pass
        deferred = Deferred()
        raise KeyError
    """)
    out = drive(path, [], target_line=8)
    assert out.status == "raised" and out.exc == "KeyError", out
    assert {"import_state_unmodelled", "stack_unmodelled"} <= set(out.flags)
    assert not out.reached and out.violated(("KeyError",)) is None
    assert victim.read_text() == "keep"
