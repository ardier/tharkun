"""[internal ref]: a stage that runs dry never ends the run; the goal being met or the budget being spent does.

``sv/ledger.py::Goal`` / ``Ledger.stopped`` are the one stop condition every stage reads;
``sv/scheduler.py`` hands what enumeration left to the stages that can keep searching an unbounded draw
domain (``stages.Stage.deepen``), in rounds, under the same ledger.
"""
from __future__ import annotations

import sys
import textwrap
import time

import pytest

from tharkun.hybrid.fdp_codec import atheris_available
from tharkun.hybrid.backend import FakeBackend
from tharkun.sv import engine, stages
from tharkun.sv.engine import replay, verify
from tharkun.sv.hybrid import _LiveCpu
from tharkun.sv.ledger import Goal, Ledger

IE = ("IndexError",)


# --- the goal and the one stop condition --------------------------------------------------------

class Clock:
    def __init__(self):
        self.cpu, self.wall = 100.0, 5000.0


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(engine, "_cpu_seconds", lambda: c.cpu)
    monkeypatch.setattr(engine.time, "monotonic", lambda: c.wall)
    return c


def test_the_default_goal_is_one_confirmed_violation():
    g = Goal()
    assert g.wanted == 1 and not g.met
    g.record()
    assert g.met


def test_a_goal_of_several_is_met_only_by_that_many():
    g = Goal(wanted=3)
    g.record(2)
    assert not g.met
    g.record()
    assert g.met


def test_stopped_is_goal_met_or_budget_spent_or_cap_reached(clock):
    ledger = Ledger(100.0)
    assert not ledger.stopped() and ledger.stop_reason() == ""
    clock.cpu += 99.5
    assert not ledger.stopped() and ledger.stopped(reserve=1.0)       # a stage that needs a second asks for one
    assert ledger.stop_reason(reserve=1.0) == "budget exhausted"
    clock.cpu += 1.0
    assert ledger.stopped() and ledger.stop_reason() == "budget exhausted"


def test_stopped_when_the_goal_is_met_whatever_budget_remains(clock):
    ledger = Ledger(100.0)
    ledger.goal.record()
    assert ledger.stopped() and ledger.stop_reason() == "goal met" and ledger.remaining() == 100.0


def test_stopped_at_a_cap(clock):
    ledger = Ledger(100.0)
    ledger.capped(5.0, "canary")
    assert not ledger.stopped()
    clock.cpu += 5.0
    assert ledger.stopped() and ledger.stop_reason() == "search capped"


def test_a_running_fuzzers_cpu_is_on_the_ledger_until_it_is_reaped(clock):
    fuzzer = FakeBackend([1])
    ledger = Ledger(100.0)
    ledger.add_server(_LiveCpu(fuzzer))
    fuzzer.cpu = 12.0
    assert ledger.spent() == 12.0
    fuzzer.stop()                                  # reaped: ``os.times`` counts it from here on, not the view
    clock.cpu += 12.0
    assert ledger.spent() == 12.0


# --- the scheduler: enumeration running dry --------------------------------------------------------

NEEDLE = "q" * 37 + "!"       # in no candidate list: enumeration drains without it


def _prog(tmp_path, body):
    p = tmp_path / "t.py"
    p.write_text(textwrap.dedent(body))
    return str(p)


NEEDLE_PROGRAM = f"""
    import _sv_verifier
    if __name__ == "__main__":
        s = _sv_verifier.nondet_str()
        if s == {NEEDLE!r}:
            [][0]
"""


@pytest.fixture
def registry(monkeypatch):
    """Register a fake deepening stage; returns the list its calls are recorded in."""
    calls = []

    def install(deepen, name="needle"):
        stage = stages.Stage(name, f"test:{name}", deepen=deepen)
        monkeypatch.setattr(stages, "REGISTRY", stages.REGISTRY + (stage,))
        return calls

    return install


def _verify(program, names, **kw):
    kw.setdefault("budget_seconds", 60)
    return verify(program, IE, jobs=1, prove=False, concolic=False, constants=False, stages=names, **kw)


def test_without_a_deepening_stage_the_run_ends_when_enumeration_drains(tmp_path):
    v = _verify(_prog(tmp_path, NEEDLE_PROGRAM), [])
    assert v.verdict == "unknown" and "no stage left to try" in v.reason


def test_a_drained_enumeration_hands_the_rest_to_a_deepening_stage(tmp_path, registry):
    def deepen(ctx, budget, round_):
        yield stages.Candidate([{"kind": "str", "value": NEEDLE}], "test:needle")

    registry(deepen)
    v = _verify(_prog(tmp_path, NEEDLE_PROGRAM), ["needle"])
    assert v.verdict == "false" and v.component == "test:needle"
    assert replay(_prog(tmp_path, NEEDLE_PROGRAM), v.witness, IE)[0]


def test_the_goal_ends_the_deepening_at_the_first_confirmed_violation(tmp_path, registry):
    produced = []

    def deepen(ctx, budget, round_):
        for value in ("zz", NEEDLE, "never reached", "nor this"):
            produced.append(value)
            yield stages.Candidate([{"kind": "str", "value": value}], "test:needle")

    registry(deepen)
    v = _verify(_prog(tmp_path, NEEDLE_PROGRAM), ["needle"])
    assert v.verdict == "false"
    assert produced == ["zz", NEEDLE]               # the stream is closed, not drained, once the goal is met


def test_a_stage_that_keeps_proposing_is_stopped_by_the_budget_not_by_running_dry(tmp_path, registry, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(engine, "_cpu_seconds", lambda: clock.cpu)
    rounds = []

    def deepen(ctx, budget, round_):
        rounds.append(round_)
        clock.cpu += min(budget, 2.0)
        yield stages.Candidate([{"kind": "str", "value": f"x{round_}"}], "test:needle")

    registry(deepen)
    # Only the budget clock is synthetic; each candidate still gets a real replay.
    # Fresh replay keeps the forkserver's physical CPU from adding another clock.
    v = _verify(_prog(tmp_path, NEEDLE_PROGRAM), ["needle"], budget_seconds=10,
                cpu_time_only=True, deepen=False, lazy_vectors=100_000,
                lazy_seconds=float("inf"))
    assert v.verdict == "unknown" and "budget exhausted" in v.reason
    assert len(set(rounds)) > 1


def test_a_stage_with_nothing_left_ends_the_run_short_of_the_budget(tmp_path, registry):
    rounds = []

    def deepen(ctx, budget, round_):
        rounds.append(round_)
        return iter(())

    registry(deepen)
    t0 = time.monotonic()
    v = _verify(_prog(tmp_path, NEEDLE_PROGRAM), ["needle"], budget_seconds=60)
    assert v.verdict == "unknown" and "no stage left to try" in v.reason
    assert rounds == [1] and time.monotonic() - t0 < 30


def test_max_runs_is_honoured_by_the_remainder(tmp_path, registry):
    called = []

    def deepen(ctx, budget, round_):
        called.append(round_)
        return iter(())

    registry(deepen)
    v = _verify(_prog(tmp_path, NEEDLE_PROGRAM), ["needle"], max_runs=3)
    assert v.verdict == "unknown" and called == []


def test_a_proofs_canary_is_not_extended(tmp_path, registry):
    """After a proof the search is a canary of min(canary_seconds, budget/10): the remainder never runs."""
    called = []

    def deepen(ctx, budget, round_):
        called.append(round_)
        return iter(())

    registry(deepen)
    p = _prog(tmp_path, """
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            y = x * 0 + 1
            z = 10 // y
    """)
    v = verify(p, ("ZeroDivisionError",), jobs=1, budget_seconds=60, stages=["needle"])
    assert v.verdict == "true" and called == []


PROOF_PROGRAM = """
    import _sv_verifier
    if __name__ == "__main__":
        x = _sv_verifier.nondet_int()
        y = x * 0 + 1
        z = 10 // y
"""


def test_canary_seconds_zero_skips_the_canary(tmp_path):
    """The default keeps the canary (the search is capped after a proof and the program still runs); 0 is no
    canary: the proof's TRUE comes back at once, no run happened, and the ledger recorded no cap."""
    p = _prog(tmp_path, PROOF_PROGRAM)
    logs = []
    kept = verify(p, ("ZeroDivisionError",), jobs=1, budget_seconds=60, log=logs.append,
                  witness_dir=tmp_path / "proofs")
    assert kept.verdict == "true" and kept.runs > 0
    assert any("search capped" in m and "canary for" in m for m in logs)
    logs.clear()
    none = verify(p, ("ZeroDivisionError",), jobs=1, budget_seconds=60, canary_seconds=0, log=logs.append,
                  witness_dir=tmp_path / "proofs")
    assert none.verdict == "true" and none.component == kept.component
    assert none.runs == 0
    assert not any("search capped" in m for m in logs)
    assert any("no canary for" in m for m in logs)
    assert none.detail["canary"] == dict(attempted=False, completed=False, cut=False,
                                        reason="disabled")


@pytest.mark.parametrize("status,exception", [("timeout", None), ("crashed", None),
                                              ("raised", "MemoryError")])
def test_every_early_proof_return_records_a_cut_canary(tmp_path, monkeypatch, status, exception):
    p = _prog(tmp_path, "xs = [1]\nprint(xs[0])\n")
    monkeypatch.setattr(engine, "run_once", lambda *a, **k: engine.Outcome(
        status, exception, [exception] if exception else []))
    v = verify(p, IE, budget_seconds=900, concolic=False, stages=[])
    assert v.verdict == "true" and v.component.startswith("proof:")
    assert v.detail["sites"] and v.runs == 1
    assert v.detail["canary"] == dict(attempted=True, completed=False, cut=True,
                                     reason=exception or status)
    assert "canary cut:" in v.reason


def test_a_two_hash_proof_return_records_completed_canary_runs(tmp_path, monkeypatch):
    p = _prog(tmp_path, "xs = [1]\nprint(xs[0])\n")
    monkeypatch.setattr(engine, "run_once", lambda *a, **k: engine.Outcome("clean"))
    v = verify(p, IE, budget_seconds=900, concolic=False, stages=[])
    assert v.verdict == "true" and v.runs == 2 and v.detail["sites"]
    assert v.detail["canary"] == dict(attempted=True, completed=True, cut=False,
                                     reason="no violation")
    assert "canary: no violation" in v.reason


@pytest.mark.parametrize("draws", [[], [{"kind": "int", "value": 0}]])
@pytest.mark.parametrize("run_timeout,budget,expected", [(120, 10, 1.0), (0.25, 0.3, 0.25)])
def test_terminal_canary_samples_keep_the_run_timeout_and_task_limits(
        tmp_path, monkeypatch, draws, run_timeout, budget, expected):
    p = _prog(tmp_path, PROOF_PROGRAM if draws else "y = 1\nz = 10 // y\n")
    cpu = [0.0]
    monkeypatch.setattr(engine, "_cpu_seconds", lambda: cpu[0])
    timeouts = []

    def sample(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        cpu[0] += 0.1
        return engine.Outcome("clean", draws=draws)

    monkeypatch.setattr(engine, "run_once", sample)
    v = verify(p, ("ZeroDivisionError",), budget_seconds=budget, canary_seconds=0.05,
               run_timeout=run_timeout, jobs=1, stages=[], cpu_time_only=True)
    expected_timeouts = [expected] if draws else [expected, min(expected, budget - 0.1)]
    assert timeouts == pytest.approx(expected_timeouts)
    assert v.verdict == "true" and v.runs == len(expected_timeouts)
    assert v.detail["canary"] == dict(attempted=True, completed=True, cut=False,
                                     reason="no violation")


def test_a_hash_confirmation_after_the_canary_cap_can_still_refute_the_proof(tmp_path, monkeypatch):
    p = _prog(tmp_path, "xs = [1]\nprint(xs[0])\n")
    cpu, seeds = [0.0], []
    monkeypatch.setattr(engine, "_cpu_seconds", lambda: cpu[0])

    def sample(*args, **kwargs):
        seeds.append(kwargs["hashseed"])
        cpu[0] += 0.1
        if kwargs["hashseed"] == 1:
            return engine.Outcome("raised", "IndexError", ["IndexError"],
                                  site={"file": "t.py", "line": 2})
        return engine.Outcome("clean")

    monkeypatch.setattr(engine, "run_once", sample)
    from tharkun.sv.scheduler import run
    v = run(p, IE, budget_seconds=10, canary_seconds=0.05,
               jobs=1, stages=[], cpu_time_only=True)
    assert seeds == [0, 1]
    assert v.line() == "FALSE(IndexError)" and v.runs == 2
    assert v.detail["unsound_proof"].startswith("proof:")
    assert v.detail["canary"] == dict(attempted=True, completed=True, cut=False,
                                     reason="violation")


@pytest.mark.parametrize("remaining", [0.2, 0.0])
def test_canary_task_exhaustion_still_reports_a_cut(tmp_path, monkeypatch, remaining):
    p = _prog(tmp_path, "xs = [1]\nprint(xs[0])\n")
    cpu, timeouts = [0.0], []
    monkeypatch.setattr(engine, "_cpu_seconds", lambda: cpu[0])
    original = Ledger.capped

    def capped(self, seconds, why):
        original(self, seconds, why)
        cpu[0] = self.cpu_started + self.budget_seconds - remaining

    def sample(*args, **kwargs):
        timeouts.append(kwargs["timeout"])
        cpu[0] += remaining
        return engine.Outcome("timeout")

    monkeypatch.setattr(Ledger, "capped", capped)
    monkeypatch.setattr(engine, "run_once", sample)
    v = verify(p, IE, budget_seconds=10, jobs=1, stages=[], cpu_time_only=True)
    assert timeouts == pytest.approx([remaining] if remaining else [])
    assert v.verdict == "true" and "canary cut: timeout" in v.reason
    assert v.detail["canary"] == dict(attempted=bool(remaining), completed=False, cut=True,
                                     reason="timeout")


def test_every_admitted_canary_batch_slot_finishes_after_the_cap(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from threading import Event, Lock

    p = _prog(tmp_path, PROOF_PROGRAM)
    cpu = [0.0]
    monkeypatch.setattr(engine, "_cpu_seconds", lambda: cpu[0])
    finished, order_lock = Event(), Lock()
    reservations, timeouts = [], []
    original = Ledger.reserve

    @contextmanager
    def reserve(self, seconds, **kwargs):
        with order_lock:
            reservations.append(seconds)
            slot = len(reservations)
        if slot == 3:
            assert finished.wait(timeout=5)
        with original(self, seconds, **kwargs) as granted:
            yield granted

    def sample(python, program, vector, **kwargs):
        timeouts.append(kwargs["timeout"])
        if not vector:
            cpu[0] = 0.9
        else:
            cpu[0] += 0.2
            finished.set()
        return engine.Outcome("clean", draws=vector or [{"kind": "int", "value": 0}])

    monkeypatch.setattr(Ledger, "reserve", reserve)
    monkeypatch.setattr(engine, "run_once", sample)
    v = verify(p, ("ZeroDivisionError",), budget_seconds=10, canary_seconds=1,
               jobs=2, stages=[], cpu_time_only=True)
    assert len(reservations) == 3 and timeouts == [1.0, 1.0, 1.0]
    assert v.verdict == "true" and v.runs == 3
    assert v.detail["canary"] == dict(attempted=True, completed=True, cut=False,
                                     reason="no violation")


def test_native_violation_still_defeats_a_proof_and_records_canary(tmp_path, monkeypatch):
    p = _prog(tmp_path, "[][0]\n")
    monkeypatch.setattr(engine, "_static_proof", lambda *a, **k: engine.Verdict(
        "true", component="proof:stub"))
    v = verify(p, IE, budget_seconds=900, concolic=False, stages=[])
    assert v.line() == "FALSE(IndexError)" and v.detail["unsound_proof"] == "proof:stub"
    assert v.detail["canary"] == dict(attempted=True, completed=True, cut=False,
                                     reason="violation")


def test_the_cli_exposes_canary_seconds_with_the_shipped_default(tmp_path, monkeypatch):
    import tharkun.sv.__main__ as cli

    p = _prog(tmp_path, PROOF_PROGRAM)
    prp = tmp_path / "p.prp"
    prp.write_text("CHECK( init(main()), ! uncaught(ZeroDivisionError))\n")
    seen = {}

    def fake_verify(program, classes, **kw):
        seen.update(kw)
        from tharkun.sv.engine import Verdict
        return Verdict("unknown", reason="stub")

    monkeypatch.setattr(cli, "verify", fake_verify)
    cli.main([p, "--property", str(prp)])
    assert seen["canary_seconds"] == 60.0
    cli.main([p, "--property", str(prp), "--canary-seconds", "0"])
    assert seen["canary_seconds"] == 0.0


# --- the fuzzer as the deepening stage ------------------------------------------------------------

MAGIC_PROGRAM = """
    import _sv_verifier
    if __name__ == "__main__":
        s = _sv_verifier.nondet_str()
        if s.startswith("magic") and s[5:7] == "42" and len(s) > 9:
            [][0]
"""


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_the_hybrid_stage_finds_what_enumeration_cannot_reach(tmp_path):
    """End to end: the bounded search or the deepening round may be the one that gets there -- both are the
    fuzzer's -- and the witness replays."""
    p = _prog(tmp_path, MAGIC_PROGRAM)
    v = _verify(p, ["hybrid"], budget_seconds=60)
    assert v.verdict == "false" and v.component.startswith("hybrid:")
    assert replay(p, v.witness, IE)[0]


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_deepen_streams_the_fuzzers_find_and_the_find_replays(tmp_path):
    """The deepening generator itself: it yields a vector the engine's replay confirms, while the fuzzer runs,
    and its helper process and scratch directory go with the generator."""
    p = _prog(tmp_path, MAGIC_PROGRAM)
    ctx = stages.Context(p, IE, first=_FakeFirst([{"kind": "str", "value": ""}], []), log=lambda m: None)
    stream = stages_hybrid_deepen(ctx)
    try:
        cand = next(iter(stream))
    finally:
        stream.close()
        ctx.cleanup.close()
    out = engine.run_once(sys.executable, p, cand.vector, timeout=30)
    assert out.violated(IE) == "IndexError" and cand.component == "hybrid:atheris"


def stages_hybrid_deepen(ctx):
    from tharkun.sv.hybrid import deepen
    return deepen(ctx, 40.0, 1)


class _FakeFirst:
    def __init__(self, draws, flags):
        self.draws, self.flags = draws, flags


class _FakeCtx:
    def __init__(self, program, first):
        self.program, self.first, self.table, self.classes = program, first, None, IE
        self.ledger, self.python, self.memory_bytes, self.log = None, None, None, lambda m: None


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
@pytest.mark.parametrize("draws,flags", [([], []), ([{"kind": "str", "value": ""}], ["file_unmodelled"]),
                                         ([{"kind": "str", "value": ""}], ["orig_argv"])])
def test_the_fuzzer_does_not_deepen_a_program_that_drew_nothing_or_touched_the_environment(
        tmp_path, monkeypatch, draws, flags):
    """The worker runs the program FOR REAL, outside the engine's I/O model: it must not be started for a
    program with nothing to vary, nor for one whose first run flagged a file effect, argv or a clock."""
    from tharkun.sv import hybrid

    started = []
    monkeypatch.setattr(hybrid.AtherisBackend, "start", lambda self: started.append(self))
    p = _prog(tmp_path, NEEDLE_PROGRAM)
    assert list(hybrid.deepen(_FakeCtx(p, _FakeFirst(draws, flags)), 30.0, 1)) == []
    assert started == []
