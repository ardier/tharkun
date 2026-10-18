"""[internal ref] / [internal ref]: the child's model of unseeded ``random`` (``sv/_child.py::_patch_random``).

* [internal ref] (WRONG TRUE): ``random.randint(0, 2**100)`` / ``randrange(2**63)`` / ``randint(0, 2**63-1)`` / ``randint(0, 2**64)`` /
  ``randint(-2**70, 2**70)`` died with ``OverflowError`` in the child (``len(range(...))``) before any draw, and the scheduler
  answered TRUE "deterministic: one execution".  The index is counted arithmetically now, and an exception our model raises
  that CPython would not sets the ``model_error`` flag: never a TRUE (any flag blocks the deterministic and the proof
  paths) and never a witness (``Outcome.violated``).
* [internal ref] (WRONG FALSE / WRONG TRUE): ``random.getstate()``/``setstate()`` are not modelled by the draws.  A run that touches
  the module-level pair carries ``random_state_unmodelled``: never a FALSE witness, never a certified-deterministic TRUE.
  ``random.Random()`` instances are the real generator (their own getstate/setstate are exact); an OS-seeded one and
  ``SystemRandom`` are flagged like ``random.seed()``.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from tharkun.sv import verify
from tharkun.sv.engine import run_once

BIG = [
    ("e10 randint(0, 2**100)", "random.randint(0, 2**100)", 12345678901234567890),
    ("e11 randrange(2**63)", "random.randrange(2**63)", 12345),
    ("e12 randint(0, 2**63 - 1)", "random.randint(0, 2**63 - 1)", 12345),
    ("e18 randint(0, 2**64)", "random.randint(0, 2**64)", 7),
    ("g13 randint(-2**70, 2**70)", "random.randint(-2**70, 2**70)", -7),
]


def program(tmp_path, body: str, name="prog.py"):
    path = tmp_path / name
    path.write_text("import random\n" + textwrap.dedent(body))
    return str(path)


@pytest.mark.parametrize("label, call, target", BIG, ids=[b[0] for b in BIG])
def test_a_draw_over_a_huge_range_is_an_ordinary_rindex(tmp_path, label, call, target):
    path = program(tmp_path, f"x = {call}\nassert x != {target}\n")
    out = run_once(sys.executable, path, [{"kind": "rindex", "value": 5, "n": 1}], timeout=30)
    assert out.status == "clean" and out.flags == [], (out.status, out.exc, out.msg, out.flags)
    assert [d["kind"] for d in out.draws] == ["rindex"] and out.draws[0]["n"] > 2**62
    # the draw is decoded against the right range: the value drawn at index i is start + i
    low = {"random.randint(0, 2**100)": 0, "random.randrange(2**63)": 0, "random.randint(0, 2**63 - 1)": 0,
           "random.randint(0, 2**64)": 0, "random.randint(-2**70, 2**70)": -2**70}[call]
    out = run_once(sys.executable, path, [{"kind": "rindex", "value": target - low, "n": 1}], timeout=30)
    assert out.status == "raised" and out.exc == "AssertionError" and out.violated(("AssertionError",))


@pytest.mark.parametrize("label, call, target", BIG, ids=[b[0] for b in BIG])
def test_a_huge_random_range_is_never_a_true_verdict(tmp_path, label, call, target):
    verdict = verify(program(tmp_path, f"x = {call}\nassert x != {target}\n"), ("AssertionError",),
                     budget_seconds=15, jobs=1)
    assert verdict.verdict != "true", (label, verdict.verdict, verdict.reason)


@pytest.mark.parametrize("call, expected", [
    ("random.randrange(5, None, 2)", "TypeError"),
    ("random.randrange(5, None, True)", "TypeError"),
    ("random.randrange(5, step=2)", "TypeError"),
    ("random.randrange(5, None, 0)", "TypeError"),
    ("random.randrange(5, step=1.0)", "TypeError"),
    ("random.randrange(5, step=type('IntOne', (int,), {})(1))", "TypeError"),
    ("random.randrange(0, step=2)", "TypeError"),
    ("random.randrange(5, None, 1)", "ZeroDivisionError"),
    ("random.randrange(5)", "ZeroDivisionError"),
    ("random.randrange(0)", "ValueError"),
])
def test_missing_randrange_stop_preserves_native_step_identity(tmp_path, call, expected):
    path = program(tmp_path, call + "\n1 // 0\n")
    native = subprocess.run(
        [sys.executable, "-c",
         "import runpy,sys\n"
         "try: runpy.run_path(sys.argv[1], run_name='__main__')\n"
         "except BaseException as exc: print(type(exc).__name__)\n",
         path], capture_output=True, text=True, timeout=30, check=True)
    assert native.stdout.strip() == expected, native.stderr
    out = run_once(sys.executable, path, [], timeout=30)
    assert (out.status, out.exc, out.flags) == ("raised", expected, [])


def test_missing_randrange_stop_validates_start_before_step(tmp_path):
    path = program(tmp_path, """
        class Start:
            def __index__(self):
                raise RuntimeError("start before step")
        random.randrange(Start(), step=2)
    """)
    out = run_once(sys.executable, path, [], timeout=30)
    assert (out.status, out.exc, out.msg) == ("raised", "RuntimeError", "start before step")


def test_the_count_matches_len_where_len_works_and_empty_ranges_still_raise(tmp_path):
    from tharkun.sv import _child

    for args in [(5,), (2, 9), (2, 9, 3), (9, 2, -3), (9, 2, -1), (0, 0), (3, 2), (0, 10, 7), (-5, 5, 2)]:
        rng = range(*args)
        assert _child._range_count(rng) == len(rng), args
    out = run_once(sys.executable, program(tmp_path, "random.randrange(3, 3)\n"), [], timeout=30)
    assert out.status == "raised" and out.exc == "ValueError" and out.flags == []
    out = run_once(sys.executable, program(tmp_path, "random.randrange(1, 5, 0)\n"), [], timeout=30)
    assert out.status == "raised" and out.exc == "ValueError" and out.flags == []
    out = run_once(sys.executable, program(tmp_path, "random.randrange(1.5)\n"), [], timeout=30)
    assert out.status == "raised" and out.exc == "TypeError" and out.flags == []


def test_an_exception_our_model_raises_and_cpython_would_not_flags_the_run_and_is_no_witness(tmp_path, monkeypatch):
    from tharkun.sv import _child
    from tharkun.sv.nondet_harness import drive

    monkeypatch.setattr(_child, "_range_count", lambda rng: (_ for _ in ()).throw(OverflowError("model")))
    driven = drive(program(tmp_path, "random.randrange(10)\n"), [])
    assert driven.status == "raised" and driven.exc == "OverflowError"
    assert "model_error" in driven.flags and driven.violated(("OverflowError",)) is None
    from tharkun.sv.engine import Outcome

    out = Outcome("raised", "OverflowError", ["OverflowError", "ArithmeticError", "Exception"], flags=["model_error"])
    assert out.violated(("OverflowError",)) is None
    assert Outcome("raised", "OverflowError", ["OverflowError"]).violated(("OverflowError",)) == "OverflowError"


GETSTATE = """
    st = random.getstate()
    a = random.random()
    random.setstate(st)
    b = random.random()
    assert a == b
"""


def test_getstate_and_setstate_flag_the_run_and_it_is_never_a_witness(tmp_path):
    path = program(tmp_path, GETSTATE)
    out = run_once(sys.executable, path, [{"kind": "rfloat", "value": (0.25).hex()}, {"kind": "rfloat", "value": (0.5).hex()}],
                   timeout=30)
    assert out.status == "raised" and out.exc == "AssertionError"      # the two draws differ in the model...
    assert "random_state_unmodelled" in out.flags
    assert out.violated(("AssertionError",)) is None                  # ...but natively a == b: no witness


def test_a_getstate_program_is_never_false_and_never_true(tmp_path):
    verdict = verify(program(tmp_path, GETSTATE), ("AssertionError",), budget_seconds=15, jobs=1)
    assert verdict.verdict not in ("false", "true"), (verdict.verdict, verdict.reason)
    peek = verify(program(tmp_path, "assert random.getstate()[1][1] != 123456789\n", "peek.py"), ("AssertionError",),
                  budget_seconds=15, jobs=1)
    assert peek.verdict != "true", (peek.verdict, peek.reason)


def test_a_seeded_generator_and_instance_state_methods_stay_exact(tmp_path):
    path = program(tmp_path, """
        random.seed(7)
        st = random.getstate()
        a = random.random()
        random.setstate(st)
        b = random.random()
        assert a == b
        r = random.Random(3)
        s = r.getstate()
        c = r.random()
        r.setstate(s)
        assert c == r.random()
    """)
    out = run_once(sys.executable, path, [], timeout=30)
    assert out.status == "clean" and "random_state_unmodelled" not in out.flags


def test_an_os_seeded_instance_and_systemrandom_are_flagged_like_seed_none(tmp_path):
    for body in ("random.Random().random()\n", "random.SystemRandom().random()\n", "random.Random(None).getrandbits(8)\n"):
        out = run_once(sys.executable, program(tmp_path, body), [], timeout=30)
        assert "unseeded_random_unmodelled" in out.flags, (body, out.flags)
    out = run_once(sys.executable, program(tmp_path, "random.Random(5).random()\n"), [], timeout=30)
    assert out.flags == []


def test_getstate_native_read_refuses_determinism_but_static_index_proves_true(tmp_path):
    """MT's first word is the constant 0x80000000 after any seeding: ``st[1][0] != 2147483648`` fails natively on every run, so a
    FALSE witness replays; the run says nothing about other seeds, so no TRUE (flag ``random_state_read``)."""
    path = program(tmp_path, "st = random.getstate()\nassert st[1][0] != 2147483648\n")
    out = run_once(sys.executable, path, [], timeout=30)
    assert out.status == "raised" and out.flags == ["random_state_read"]
    assert out.violated(("AssertionError",)) == "AssertionError"          # a witness may stand
    verdict = verify(path, ("AssertionError",), budget_seconds=15, jobs=1)
    assert verdict.verdict == "false", (verdict.verdict, verdict.reason)
    true_path = program(tmp_path, "st = random.getstate()\nassert st[1][-1] != 0\n", "t.py")
    verdict = verify(true_path, ("AssertionError",), budget_seconds=15, jobs=1)
    assert verdict.verdict == "true" and verdict.component == "proof:interval", (verdict.verdict, verdict.reason)
    out = run_once(sys.executable, true_path, [], timeout=30)
    assert out.status == "clean" and out.flags == ["random_state_read"]  # the run still cannot certify determinism


def test_a_getstate_after_a_random_draw_or_any_setstate_is_never_a_witness(tmp_path):
    for body in ("random.random()\nst = random.getstate()\nassert st[1][-1] != 624\n",
                 "random.randint(0, 3)\nst = random.getstate()\nassert st[1][-1] != 624\n",
                 "st = random.getstate()\nrandom.setstate(st)\nassert False\n"):
        out = run_once(sys.executable, program(tmp_path, body), [], timeout=30)
        assert "random_state_unmodelled" in out.flags, body
        assert out.violated(("AssertionError",)) is None, body
