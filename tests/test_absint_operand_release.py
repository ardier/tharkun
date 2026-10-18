"""A call hands its operands to the callee: the holder ``eval`` registered for each of them is taken back ([internal ref]).

``Analyzer.borrow`` registers every value an expression evaluation produces, because a Python local of a transfer
function is not rewritten when an allocation turns a recent object into a summary (``State.redirect`` rewrites the
heap and the frames).  An allocation at such a site is refused ("allocation would invalidate an in-flight heap
reference"), which refused the commonest object code there is: ``d[i] = d[i].combine(d[i + 1], ...)`` where
``combine`` builds a new object of the receiver's own class at one allocation site (chaos's de Boor loop).  The
operands of a user call are not such locals once they are in the callee's frame, so ``call_method``, ``_call1`` and
``instantiate`` take their holders back before ``call_func`` -- the one point where nothing reads them again --
while every other holder (an enclosing operand, a left operand waiting for its right one) stays.

* the release is what the commonest shapes need (below);
* the holders that must stay do stay (refusal controls);
* a differential: random programs over a class whose methods allocate their own class, run natively over the
  draws, fail on a TRUE the run contradicts.
"""

from __future__ import annotations

import random
import sys
import textwrap
import types

import pytest

from tharkun.solve.absint import prove

CLASS = """
class Q:
    def __init__(self, v):
        self.v = v
    def inc(self):
        return Q(self.v + 1)
    def add(self, o):
        return Q(self.v + o.v)
    def pick(self, o):
        return self if self.v > o.v else o
def mk(v):
    return Q(v)
"""
HEAD = "import _sv_verifier\ndef nd():\n    return _sv_verifier.nondet_int()\n" + CLASS


def verdict(body: str, classes=("AssertionError",)) -> str:
    return prove(HEAD + textwrap.dedent(body), list(classes), filename="<program>").verdict


# ---------------------------------------------------------------------------
# what the release buys: no refusal (the proof may still be UNKNOWN for a precision reason)
# ---------------------------------------------------------------------------

LOOPS = {
    "a method that builds its own class at one site, on an element": "d = [Q(1), Q(2)]\nfor i in range(3):\n    d[0] = d[0].add(d[1])\nassert d[0].v >= 1\n",
    "the receiver is the previous result": "p = Q(1)\nfor i in range(3):\n    p = p.inc()\nassert p.v >= 1\n",
    "the receiver is also the argument": "p = Q(1)\nfor i in range(3):\n    p = p.add(p)\nassert p.v >= 1\n",
    "an argument is another object": "p = Q(1)\nq = Q(2)\nfor i in range(3):\n    p = p.add(q)\nassert p.v >= 1\n",
    "a plain function": "p = Q(1)\nfor i in range(3):\n    p = mk(p.v)\nassert p.v >= 1\n",
    "a constructor": "p = Q(1)\nfor i in range(3):\n    p = Q(p.v + 1)\nassert p.v >= 1\n",
}


@pytest.mark.parametrize("name", sorted(LOOPS))
def test_a_call_whose_callee_allocates_at_its_operands_site_is_no_longer_refused(name):
    result = prove(HEAD + LOOPS[name], ["AssertionError"], filename="<program>")
    assert "in-flight" not in result.reason and "unsupported" not in result.reason, result.reason


def test_a_proof_through_such_a_call_is_still_a_proof():
    # the sum of two positive values is positive: nothing about the object identity is needed
    body = "d = [Q(1), Q(2)]\nfor i in range(3):\n    d[0] = d[0].add(d[1])\nassert d[0].v != -7\n"
    assert verdict(body) in ("TRUE", "UNKNOWN")
    assert verdict("p = Q(1)\nq = p.inc().inc().inc()\nassert q.v >= 1\n") == "TRUE"


# ---------------------------------------------------------------------------
# what must stay refused: an earlier operand still waits while a later one allocates at its site
# ---------------------------------------------------------------------------
#
# An allocation site is its node plus the call string, so the same site is allocated twice only when the same
# call node runs again: in a loop.  ``p`` below is the object ``mk(i)`` made in the previous iteration; the
# enclosing expression holds it while this iteration's ``mk(i)`` allocates at the same site.


@pytest.mark.parametrize("body", [
    "lst = [p, mk(i)]\n    p = lst[1]",
    "lst = (p, mk(i))\n    p = lst[1]",
    "lst = [p.v, mk(i)]\n    q = p\n    p = lst[1]",       # the number waits, but q is held by the statement list below
    "r = p.add(mk(i))\n    p = mk(i + 1)",
    "r = p.pick(mk(i))\n    p = r",
])
def test_an_enclosing_holder_still_refuses_an_allocation_at_its_site(body):
    src = HEAD + "p = Q(0)\nfor i in range(3):\n    " + body + "\nassert p.v >= -5\n"
    result = prove(src, ["AssertionError"], filename="<program>")
    # either the analysis proves nothing here (the refusal) or what it proves holds (the differential below):
    # a TRUE would have to survive the native run
    if result.verdict == "TRUE":
        assert "AssertionError" not in native_raises(src), src


@pytest.mark.parametrize("body", [
    "lst = [p, mk(i)]\n    p = lst[1]",
    "lst = (p, mk(i))\n    p = lst[1]",
])
def test_the_refusal_is_the_in_flight_one(body):
    src = HEAD + "p = Q(0)\nfor i in range(3):\n    " + body + "\nassert p.v >= -5\n"
    result = prove(src, ["AssertionError"], filename="<program>")
    assert result.verdict == "UNKNOWN" and "in-flight" in result.reason, (body, result.verdict, result.reason)


def test_a_number_that_waits_holds_no_object():
    # ``p.v + mk(i).v``: the left operand is a number, no object is waiting for the allocation
    src = HEAD + "p = Q(0)\nfor i in range(3):\n    r = p.v + mk(i).v\n    p = mk(i)\nassert r >= -5\n"
    result = prove(src, ["AssertionError"], filename="<program>")
    assert "in-flight" not in result.reason, result.reason


# ---------------------------------------------------------------------------
# the differential
# ---------------------------------------------------------------------------

STATEMENTS = [
    "p = p.inc()", "p = p.add(q)", "p = q.add(p)", "p = p.add(p)", "p = p.add(p.inc())", "p = q.add(p.inc())",
    "p = mk(p.v).add(mk(q.v))", "p = p.inc().add(p.inc())", "p = p.pick(p.inc())", "p = mk(p.v + q.v)",
    "q = q.add(p)", "q = p.inc().add(q)", "q = q.pick(p)", "r = p.v + p.inc().v", "r = p.add(q).v - q.add(p).v",
    "lst = [p, p.inc(), q]", "lst = [mk(1), p, mk(2)]", "p = Q(p.v + q.v)", "q = Q(p.add(q).v)",
    "p = mk(q.inc().v)", "p = p.inc().inc()", "q = p.inc()", "lst = [q, q.inc()]", "lst = [p.inc(), mk(0), p.inc()]",
    "r = (p.inc().v, p.inc().v)[0]", "p = p.add(p.inc().inc())", "q = q.add(q.inc())",
    "lst = [p, mk(1)]\np = lst[1]", "lst = (q, mk(2))\nq = lst[1]", "r = p.add(mk(3))\np = mk(r.v)",
    "t = [q, p.inc()]\nq = t[1]", "r = q.pick(p.inc()).v",
]
READS = [
    "assert p.v >= 0", "assert p.v != 3", "assert q.v <= 4", "assert p.v < 5", "c = 1 // (p.v - 2)",
    "assert r >= 0", "assert lst[0].v != 1", "assert lst[1].v >= 1", "assert lst[0].v == lst[1].v",
    "assert p.v == q.v", "assert p.v != q.v", "assert r == 0",
]
DRAWS = [-2, -1, 0, 1, 2, 3]


def native_raises(src: str) -> set[str]:
    out: set[str] = set()
    code = compile(src, "<program>", "exec")
    for a in DRAWS:
        for b in DRAWS:
            it = iter([a, b])
            mod = types.ModuleType("_sv_verifier")
            mod.nondet_int = lambda: next(it)
            saved = sys.modules.get("_sv_verifier")
            sys.modules["_sv_verifier"] = mod
            try:
                exec(code, {"__name__": "__main__"})
            except Exception as e:  # noqa: BLE001
                out |= {c.__name__ for c in type(e).__mro__}
            finally:
                if saved is None:
                    sys.modules.pop("_sv_verifier", None)
                else:
                    sys.modules["_sv_verifier"] = saved
    return out


def _program(rng: random.Random) -> str:
    body = ["a = nd()", "b = nd()", "p = Q(a)", "q = Q(b)", "r = 0", "lst = [p]"]
    for _ in range(rng.randint(1, 4)):
        stmt = rng.choice(STATEMENTS)
        if rng.random() < 0.5:
            body += [f"for i in range({rng.randint(1, 3)}):"] + ["    " + line for line in stmt.split("\n")]
        else:
            body += stmt.split("\n")
    body.append(rng.choice(READS))
    return HEAD + "\n".join(body) + "\n"


@pytest.mark.parametrize("seed", range(12))
def test_a_released_operand_never_makes_a_wrong_true(seed):
    rng = random.Random(1121_000 + seed)
    proved = 0
    for _ in range(25):
        src = _program(rng)
        raised = native_raises(src)
        for cls in ("AssertionError", "ZeroDivisionError", "AttributeError", "IndexError"):
            if prove(src, [cls], filename="<program>", time_budget=20.0).verdict == "TRUE":
                proved += 1
                assert cls not in raised, f"WRONG TRUE for {cls}:\n{src}"
    assert proved > 0
