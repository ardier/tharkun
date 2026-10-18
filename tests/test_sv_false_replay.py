"""[internal ref]: every emitted violation must survive the validator's replay conditions."""
import os
import subprocess
import sys
import textwrap

import pytest

from tharkun.sv import engine, scheduler


def program(tmp_path, source):
    path = tmp_path / "program.py"
    path.write_text(textwrap.dedent(source))
    return str(path)


def test_replay_refuses_a_violation_that_only_the_inherited_environment_causes(tmp_path, monkeypatch):
    monkeypatch.setenv("THARKUN_FALSE_REPLAY_SECRET", "poison")
    path = program(tmp_path, """
        import os
        if os.environ.get("THARKUN_FALSE_REPLAY_SECRET") == "poison":
            raise IndexError("ambient-only")
    """)
    discovery = engine.run_once(sys.executable, path, [], timeout=30)
    assert discovery.violated(("IndexError",)) == "IndexError"
    confirmed, out = engine.replay(path, {"vector": []}, ("IndexError",))
    assert not confirmed and out.status == "clean"


def test_replay_uses_real_validator_boot_conditions(tmp_path, monkeypatch):
    monkeypatch.setenv("THARKUN_FALSE_REPLAY_SECRET", "poison")
    monkeypatch.setenv("PYTHONHASHSEED", "1")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "ambient-imports"))
    monkeypatch.setenv("PYTHONSTARTUP", str(tmp_path / "ambient-startup.py"))
    native_env = {"PATH": os.environ.get("PATH", os.defpath), "PYTHONHASHSEED": "0",
                  "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"}
    expected_hash = int(subprocess.check_output(
        [sys.executable, "-c", "print(hash('false-replay-check'))"], env=native_env, text=True))
    path = program(tmp_path, f"""
        import os, sys, site
        if os.environ.get("PYTHONHASHSEED") != "0" or hash("false-replay-check") != {expected_hash}:
            raise RuntimeError("wrong hash seed")
        if sys.stdin.read() != "" or site.ENABLE_USER_SITE:
            raise RuntimeError("wrong input or user site")
        if any(key in os.environ for key in ("THARKUN_FALSE_REPLAY_SECRET", "PYTHONPATH", "PYTHONSTARTUP")):
            raise RuntimeError("inherited environment")
        raise IndexError("validator control")
    """)
    confirmed, out = engine.replay(path, {"vector": []}, ("IndexError",))
    assert confirmed and out.exc == "IndexError"


def test_already_fresh_discovery_still_needs_validator_confirmation(tmp_path, monkeypatch):
    path = program(tmp_path, "[][0]\n")
    real_run_once = engine.run_once
    calls = []

    def run_once(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("minimal_env"):
            return engine.Outcome("clean")
        return real_run_once(*args, **kwargs)

    monkeypatch.setattr(engine, "run_once", run_once)
    verdict = engine.verify(path, ("IndexError",), budget_seconds=60, prove=False, concolic=False)
    assert verdict.verdict == "unknown" and verdict.witness is None and verdict.exception is None
    assert "validator replay" in verdict.reason
    assert len(calls) == 2 and calls[-1]["hashseed"] == 0 and calls[-1]["minimal_env"]


def test_hashseed_one_discovery_is_not_a_seed_zero_witness(tmp_path, monkeypatch):
    native_env = {"PATH": os.environ.get("PATH", os.defpath), "PYTHONHASHSEED": "1",
                  "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    seed_one_hash = int(subprocess.check_output(
        [sys.executable, "-c", "print(hash('false-replay-check'))"], env=native_env, text=True))
    path = program(tmp_path, f"if hash('false-replay-check') == {seed_one_hash}: raise IndexError\n")
    out = engine.run_once(sys.executable, path, [], timeout=30, hashseed=1)
    assert out.violated(("IndexError",)) == "IndexError"
    candidate = engine.Verdict("false", "IndexError", witness={"vector": []}, component="deterministic")
    monkeypatch.setattr(scheduler, "run", lambda *args, **kwargs: candidate)
    verdict = engine.verify(path, ("IndexError",), budget_seconds=60)
    assert verdict.verdict == "unknown" and verdict.witness is None


@pytest.mark.parametrize("component", ["deterministic", "symbolic:product", "proof:stationary-run",
                                       "concolic:z3", "enum:small_first"])
def test_no_discovery_component_can_bypass_validator_confirmation(tmp_path, monkeypatch, component):
    path = program(tmp_path, "pass\n")
    candidate = engine.Verdict("false", "IndexError", witness={"vector": []}, component=component,
                               detail={"unsound_proof": "candidate", "proof_reason": "unconfirmed"})
    monkeypatch.setattr(scheduler, "run", lambda *args, **kwargs: candidate)
    verdict = engine.verify(path, ("IndexError",), budget_seconds=60)
    assert verdict.verdict == "unknown" and verdict.witness is None
    assert verdict.component == "none" and "unsound_proof" not in verdict.detail
    assert "proof_reason" not in verdict.detail


@pytest.mark.parametrize("status", ["clean", "timeout", "crashed", "different-exception", "different-vector"])
def test_unconfirmed_replays_clear_the_exportable_witness(tmp_path, monkeypatch, status):
    path = program(tmp_path, "raise IndexError\n")
    candidate = engine.Verdict("false", "IndexError", witness={"vector": []})
    monkeypatch.setattr(scheduler, "run", lambda *args, **kwargs: candidate)
    out = (engine.Outcome("raised", "KeyError", ["KeyError"]) if status == "different-exception" else
           engine.Outcome("raised", "IndexError", ["IndexError"], draws=[{"kind": "int", "value": 0}])
           if status == "different-vector" else engine.Outcome(status))
    monkeypatch.setattr(engine, "run_once", lambda *args, **kwargs: out)
    verdict = engine.verify(path, ("IndexError", "KeyError"), budget_seconds=60)
    assert verdict.verdict == "unknown" and verdict.witness is None and verdict.exception is None


@pytest.mark.parametrize("cpu_used,wall_used", [(10, 0), (0, 10)])
def test_no_replay_is_started_after_the_whole_budget_is_spent(tmp_path, monkeypatch, cpu_used, wall_used):
    clock = {"cpu": 0, "wall": 0}
    monkeypatch.setattr(engine, "_cpu_seconds", lambda: clock["cpu"])
    monkeypatch.setattr(engine.time, "monotonic", lambda: clock["wall"])

    def run(*args, **kwargs):
        clock.update(cpu=cpu_used, wall=wall_used)
        return engine.Verdict("false", "IndexError", witness={"vector": []}, runs=2)

    monkeypatch.setattr(scheduler, "run", run)
    monkeypatch.setattr(engine, "replay", lambda *args, **kwargs: pytest.fail("budget exhausted"))
    verdict = engine.verify("unused.py", ("IndexError",), budget_seconds=10)
    assert verdict.verdict == "unknown" and verdict.witness is None and verdict.runs == 2
    assert "remaining budget" in verdict.reason


@pytest.mark.parametrize("cpu_only,expected_timeout", [(False, 3), (True, 8)])
def test_confirmation_shares_the_remaining_budget_and_child_memory(monkeypatch, cpu_only, expected_timeout):
    clock = {"cpu": 0, "wall": 0}
    monkeypatch.setattr(engine, "_cpu_seconds", lambda: clock["cpu"])
    monkeypatch.setattr(engine.time, "monotonic", lambda: clock["wall"])
    witness = {"vector": []}

    def run(*args, **kwargs):
        clock.update(cpu=2, wall=7)
        return engine.Verdict("false", "IndexError", witness=witness, runs=2)

    def replay(path, received, classes, **kwargs):
        assert received is witness and classes == ("IndexError",)
        assert kwargs["timeout"] == expected_timeout and kwargs["memory_bytes"] == 1024
        clock["wall"] += 0.5
        return True, engine.Outcome("raised", "IndexError", ["IndexError"])

    monkeypatch.setattr(scheduler, "run", run)
    monkeypatch.setattr(engine, "replay", replay)
    verdict = engine.verify("unused.py", ("IndexError", "KeyError"), budget_seconds=10, run_timeout=9,
                            run_memory_bytes=4096, jobs=4, cpu_time_only=cpu_only)
    assert verdict.verdict == "false" and verdict.witness is witness and verdict.runs == 3
    assert verdict.seconds == 7.5


@pytest.mark.parametrize("cpu_only,cpu_used,wall_used,expected", [
    (False, 8, 0, "unknown"), (False, 0, 8, "unknown"),
    (True, 8, 0, "unknown"), (True, 0, 8, "false"),
])
def test_replay_cannot_confirm_after_exhausting_the_task_budget(
        monkeypatch, cpu_only, cpu_used, wall_used, expected):
    clock = {"cpu": 0, "wall": 0}
    monkeypatch.setattr(engine, "_cpu_seconds", lambda: clock["cpu"])
    monkeypatch.setattr(engine.time, "monotonic", lambda: clock["wall"])
    witness = {"vector": []}

    def run(*args, **kwargs):
        clock.update(cpu=2, wall=2)
        return engine.Verdict("false", "IndexError", witness=witness, runs=2,
                              detail={"unsound_proof": "candidate", "proof_reason": "unconfirmed"})

    def replay(*args, **kwargs):
        assert kwargs["timeout"] == 8
        clock["cpu"] += cpu_used
        clock["wall"] += wall_used
        return True, engine.Outcome("raised", "IndexError", ["IndexError"])

    monkeypatch.setattr(scheduler, "run", run)
    monkeypatch.setattr(engine, "replay", replay)
    verdict = engine.verify("unused.py", ("IndexError",), budget_seconds=10, cpu_time_only=cpu_only)
    assert verdict.verdict == expected and verdict.runs == 3
    if expected == "unknown":
        assert verdict.witness is None and verdict.exception is None and verdict.component == "none"
        assert "exhausted task budget" in verdict.reason
        assert "unsound_proof" not in verdict.detail and "proof_reason" not in verdict.detail
    else:
        assert verdict.witness is witness and verdict.exception == "IndexError"


def test_product_model_and_controlled_draw_still_confirm(tmp_path):
    path = program(tmp_path, """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        assert x != 123
    """)
    verdict = engine.verify(path, ("AssertionError",), budget_seconds=900, canary_seconds=0)
    assert verdict.verdict == "false" and verdict.component == "symbolic:product"
    assert verdict.witness["format"] == "sv-py-witness/1" and verdict.witness["exception"] == "AssertionError"
    confirmed, out = engine.replay(path, verdict.witness, ("AssertionError",))
    assert confirmed and engine._replayed(verdict.witness["vector"], out)


def test_false_replay_does_not_change_true_or_unknown(monkeypatch):
    monkeypatch.setattr(engine, "replay", lambda *args, **kwargs: pytest.fail("no violation to confirm"))
    for kind in ("true", "unknown"):
        candidate = engine.Verdict(kind, reason="control", runs=2, seconds=1.0)
        monkeypatch.setattr(scheduler, "run", lambda *args, **kwargs: candidate)
        assert engine.verify("unused.py", ("IndexError",)) is candidate
