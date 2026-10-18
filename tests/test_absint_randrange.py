"""``random.randrange`` and ``random.getrandbits`` (the models of ``solve/absint/rand.py``): exact, or the named ``TypeError``.

``randrange`` is the draw ``go`` hashes its board with (``zobrist_strings = [random.randrange(2**63 - 1) ...]``); before this it was
contract-driven (no value, every class unproven: a program that called it was never TRUE for anything).  The model is the hull of
``Lib/random.py::Random.randrange`` over the bounds of its arguments (see ``Analyzer.m_random_randrange``).

Each case is a fixed program, run natively (the controls) and by absint; no generated programs.
"""

from __future__ import annotations

import io
import random
import sys
import textwrap
import types
import warnings

import pytest

from tharkun.solve.absint import prove

HEAD = "import math\nimport random\nimport _sv_verifier as sv\n"
SENTINEL = "reached = True\nc = 1 // 0\n"


def _stub() -> types.ModuleType:
    m = types.ModuleType("_sv_verifier")
    m.nondet_int = lambda: 7
    m.nondet_bool = lambda: True
    return m


def native(src: str):
    ns: dict = {"__name__": "__main__"}
    warnings.simplefilter("ignore")
    old_out, old_state, old_mod = sys.stdout, random.getstate(), sys.modules.get("_sv_verifier")
    sys.stdout = io.StringIO()
    sys.modules["_sv_verifier"] = _stub()
    try:
        exec(compile(src, "<program>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        reached = bool(ns.get("reached"))
        return reached, None if reached else {c.__name__ for c in type(e).__mro__}
    finally:
        sys.stdout = old_out
        random.setstate(old_state)
        if old_mod is None:
            sys.modules.pop("_sv_verifier", None)
        else:
            sys.modules["_sv_verifier"] = old_mod
    return False, None


def verdict(src: str, cls: str):
    return prove(src, [cls], filename="<program>", time_budget=20.0)


def never_true(consumer: str, cls: str | None = None) -> None:
    src = HEAD + textwrap.dedent(consumer) + "\n" + SENTINEL
    reached, raised = native(src)
    if cls is None:
        assert reached, ("the consumer must succeed natively", consumer, raised)
        cls = "ZeroDivisionError"
    else:
        assert not reached and cls in raised, ("native must raise " + cls, consumer, reached, raised)
    assert verdict(src, cls).verdict != "TRUE", src


# the hull of each call, with the native control: 400 draws all inside it, and for every value drawn, `assert x != v` is not proven
HULLS = [
    ("random.randrange(5)", 0, 4), ("random.randrange(1)", 0, 0), ("random.randrange(True)", 0, 0), ("random.randrange(2, 5)", 2, 4),
    ("random.randrange(-3, 3)", -3, 2), ("random.randrange(0, 10, 3)", 0, 9), ("random.randrange(10, 0, -3)", 1, 10),
    ("random.randrange(0, 10, 1)", 0, 9), ("random.randrange(start=2, stop=6, step=2)", 2, 5), ("random.randrange(5, None)", 0, 4),
    ("random.randrange(5, None, 1)", 0, 4), ("random.randrange(5, step=1)", 0, 4), ("random.randrange(stop=5, start=2)", 2, 4),
    ("random.randrange(7, 8)", 7, 7), ("random.randrange(-5, -2)", -5, -3), ("random.randrange(0, -6, -2)", -5, 0),
    ("random.randrange(0, 3, True)", 0, 2), ("random.randrange(False, 3)", 0, 2),
    ("random.getrandbits(4)", 0, 15), ("random.getrandbits(0)", 0, 0), ("random.getrandbits(1)", 0, 1), ("random.getrandbits(8)", 0, 255),
]


@pytest.mark.parametrize("call, lo, hi", HULLS)
def test_the_hull_contains_every_native_draw_and_is_proven(call, lo, hi):
    seen = {eval(call, {"random": random}) for _ in range(400)}  # noqa: S307 -- a fixed literal from the table above
    assert seen and min(seen) >= lo and max(seen) <= hi, (call, sorted(seen))
    assert verdict(HEAD + f"x = {call}\nassert {lo} <= x <= {hi}\n", "AssertionError").verdict == "TRUE", call
    for v in sorted(seen):
        assert verdict(HEAD + f"x = {call}\nassert x != {v}\n", "AssertionError").verdict != "TRUE", (call, v)


def test_the_hull_is_not_wider_than_the_arguments_allow():
    """One value past either end is proven out: the model is a hull, not 'any int'."""
    for call, lo, hi in (("random.randrange(5)", 0, 4), ("random.randrange(2, 5)", 2, 4), ("random.randrange(10, 0, -3)", 1, 10),
                         ("random.getrandbits(4)", 0, 15)):
        assert verdict(HEAD + f"x = {call}\nassert x >= {lo}\n", "AssertionError").verdict == "TRUE", call
        assert verdict(HEAD + f"x = {call}\nassert x <= {hi}\n", "AssertionError").verdict == "TRUE", call
        assert verdict(HEAD + f"x = {call}\nassert x >= {lo + 1}\n", "AssertionError").verdict != "TRUE", call
        assert verdict(HEAD + f"x = {call}\nassert x <= {hi - 1}\n", "AssertionError").verdict != "TRUE", call


def test_a_divisor_proven_non_zero_by_the_hull():
    assert verdict(HEAD + "x = random.randrange(1, 5)\nc = 10 // x\n", "ZeroDivisionError").verdict == "TRUE"
    assert verdict(HEAD + "x = random.randrange(0, 5)\nc = 10 // x\n", "ZeroDivisionError").verdict != "TRUE"
    assert verdict(HEAD + "x = random.randrange(5)\nc = 10 // (x + 1)\n", "ZeroDivisionError").verdict == "TRUE"
    assert verdict(HEAD + "x = random.randrange(5)\nc = 10 // (x - 4)\n", "ZeroDivisionError").verdict != "TRUE"
    assert verdict(HEAD + "x = random.randrange(5)\nc = 10 // (x - 5)\n", "ZeroDivisionError").verdict == "TRUE"


# a range with no value: a ValueError on every run (not a property class), so what follows is not reached
EMPTY = ["random.randrange(0)", "random.randrange(-5)", "random.randrange(5, 5)", "random.randrange(5, 3)", "random.randrange(3, 5, -1)",
         "random.randrange(0, 5, 0)", "random.randrange(5, 5, -1)", "random.randrange(False)", "random.getrandbits(-1)"]


@pytest.mark.parametrize("call", EMPTY)
def test_an_empty_range_ends_the_path(call):
    src = HEAD + f"x = {call}\nassert False\n"
    reached, raised = native(src)
    assert not reached and "ValueError" in raised, (call, reached, raised)
    assert verdict(src, "AssertionError").verdict == "TRUE", call


TYPE_ERRORS = [
    "random.randrange(5.0)", "random.randrange('a')", "random.randrange(None)", "random.randrange([1])", "random.randrange(5, 'a')",
    "random.randrange(5, 6.0)", "random.randrange(5, 6, 1.0)", "random.randrange(5, 6, None)", "random.randrange(5, None, 2)",
    "random.randrange(5, None, True)", "random.randrange(5, step=2)", "random.randrange(5, None, 0)", "random.randrange()",
    "random.randrange(1, 2, 3, 4)", "random.randrange(stop=5)", "random.randrange(1, start=1)", "random.randrange(1, step=1, step=2)"
    if False else "random.randrange(1, 2, step=1, stop=3)", "random.getrandbits(1.5)", "random.getrandbits('a')", "random.getrandbits(None)",
    "random.getrandbits(k=4)", "random.getrandbits()", "random.getrandbits(1, 2)",
]


@pytest.mark.parametrize("call", TYPE_ERRORS)
def test_a_wrong_kind_or_shape_is_a_recorded_type_error(call):
    never_true(f"x = {call}", "TypeError")
    src = HEAD + f"x = {call}\n" + SENTINEL
    assert verdict(src, "ZeroDivisionError").verdict == "TRUE", call  # certain: the sentinel is never reached


VALID = [
    "random.randrange(5)", "random.randrange(start=5)", "random.randrange(0, 5)", "random.randrange(0, stop=5)", "random.randrange(0, 5, 1)",
    "random.randrange(0, 5, step=2)", "random.randrange(start=0, stop=5, step=2)", "random.randrange(5, None)", "random.randrange(5, stop=None)",
    "random.randrange(5, step=1)", "random.randrange(5, None, 1)", "random.randrange(True)", "random.randrange(0, True + 4)",
    "random.randrange(10 ** 30)", "random.randrange(-10 ** 30, 10 ** 30, 7)", "random.getrandbits(3)", "random.getrandbits(0)",
]


@pytest.mark.parametrize("call", VALID)
def test_a_valid_shape_continues_the_path(call):
    never_true(f"x = {call}")


def test_an_unknown_bound_gives_the_half_line():
    """``n`` is any int: ``randrange(n)`` is ``[0, ...)`` when it succeeds (``n > 0``), and a ``ValueError`` otherwise."""
    src = HEAD + "n = sv.nondet_int()\nx = random.randrange(n)\nassert x >= 0\n"
    assert verdict(src, "AssertionError").verdict == "TRUE"
    src = HEAD + "n = sv.nondet_int()\nx = random.randrange(n)\nassert x < 5\n"
    assert verdict(src, "AssertionError").verdict != "TRUE"
    src = HEAD + "n = sv.nondet_int()\nx = random.randrange(0, n)\nassert x >= 0\n"
    assert verdict(src, "AssertionError").verdict == "TRUE"
    src = HEAD + "n = sv.nondet_int()\nx = random.randrange(n, 0, -1)\nassert x <= n\n"
    assert verdict(src, "AssertionError").verdict != "TRUE"  # a relation the interval cannot hold


def test_stop_may_be_none_or_a_number():
    """With ``stop`` a number or ``None`` the draw is ``[start, stop - 1]`` or ``[0, start - 1]``: joined, it starts at ``0``.  (The join
    of ``5`` and ``None`` keeps no bound on the number, so the upper end is open here: that is the join, not this model.)"""
    src = HEAD + "stop = 5 if sv.nondet_bool() else None\nx = random.randrange(3, stop)\nassert x >= 0\n"
    assert verdict(src, "AssertionError").verdict == "TRUE"
    src = HEAD + "stop = 5 if sv.nondet_bool() else None\nx = random.randrange(3, stop)\nassert x >= 3\n"
    assert verdict(src, "AssertionError").verdict != "TRUE"  # stop None draws from [0, 2]


def test_a_step_with_a_none_stop_is_the_type_error_only_unless_it_is_the_int_one():
    for call, ok in (("random.randrange(5, None, 1)", True), ("random.randrange(5, None, True)", False), ("random.randrange(5, None, 2)", False),
                     ("random.randrange(5, None, 1.0)", False)):
        src = HEAD + f"x = {call}\n" + SENTINEL
        reached, raised = native(src)
        assert reached == ok and (ok or "TypeError" in raised), (call, reached, raised)


def test_the_go_zobrist_shape():
    """``bm_go_nondet.py``: ``[[random.randrange(sys.maxsize) for _ in range(3)] for _ in range(4)]`` hashed with ``^=``."""
    src = HEAD + ("import sys\nz = [random.randrange(sys.maxsize) for _ in range(3)]\nh = 0\nfor v in z:\n    h ^= v\nassert h >= 0\n")
    assert verdict(src, "AssertionError").verdict in ("TRUE", "UNKNOWN")
    reached, raised = native(src)
    assert not raised
