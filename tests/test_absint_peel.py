"""A2 (docs/ABSINT.md "The first iteration on its own"): every loop is analysed once from its entry state, with
checks recorded, and the fixpoint then starts from where that iteration ended and describes iterations 2 and on.

What this file pins, worst failure first:

* a site is proven only if it is proven in BOTH the first-iteration pass and the final pass (AND, never
  overwrite): an unsafe first iteration or an unsafe later one keeps the site unproven;
* the witness's loop-head invariant holds at the first visit too (entry joined with the head) and is inductive;
* what the peeled iteration buys: a loop that runs once is proven as the straight-line code it is, while the same
  loop run twice is not;
* a loop whose first iteration adds nothing is not iterated at all;
* a pass over a loop starts where a statement starts (the fuzz programs the first version of this lost);
* the widening delay is a number of plain joins: more delay, more steps;
* a site unsafe in the first iteration only keeps its residual.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tharkun.solve.absint import interp, prove
from tharkun.solve.absint.interp import Analyzer

FIX = Path(__file__).parent / "fixtures" / "absint" / "peel"
AR = ("ZeroDivisionError", "FloatingPointError")
DL = ("KeyError", "IndexError")
DT = ("TypeError", "AttributeError")
AS = ("AssertionError",)


def _prove(source: str, classes, **kw):
    return prove(textwrap.dedent(source), classes, filename="<program>", time_budget=60, **kw)


# ------------------------------------------------ a site is proven in BOTH passes
@pytest.mark.parametrize("name, source", [
    ("for, unsafe in iteration 1 only", """
        x = 0
        for i in range(3):
            y = 1 // x
            x = 1
    """),
    ("while, unsafe in iteration 1 only", """
        x = 0
        i = 0
        while i < 3:
            y = 1 // x
            x = 1
            i += 1
    """),
    ("for, unsafe in later iterations only", """
        x = 1
        for i in range(3):
            y = 1 // x
            x = 0
    """),
    ("while, unsafe in later iterations only", """
        x = 1
        i = 0
        while i < 3:
            y = 1 // x
            x = 0
            i += 1
    """),
])
def test_a_site_unsafe_in_either_pass_stays_unproven(name, source):
    r = _prove(source, AR)
    assert r.verdict == "UNKNOWN" and "divide" in r.reason, (name, r.reason)


# ----------------------------------------------------- what the peeled iteration buys
@pytest.mark.parametrize("source", ["""
    xs = [0]
    for i in range(1):
        xs.pop()
""", """
    xs = [0]
    n = 0
    while n < 1:
        xs.pop()
        n += 1
""", """
    xs = [0, 1]
    for i in range(min(xs[1], 6)):
        xs.pop()
"""])
def test_a_loop_that_runs_once_is_proven_as_the_straight_line_code_it_is(source):
    assert _prove(source, DL).verdict == "TRUE"


def test_the_same_loop_run_twice_is_not_proven():
    r = _prove("""
        xs = [0]
        for i in range(2):
            xs.pop()
    """, DL)
    assert r.verdict == "UNKNOWN" and "pop" in r.reason


# ------------------------------------------------- a first iteration that adds nothing
def test_a_loop_whose_first_iteration_adds_nothing_is_not_iterated(monkeypatch):
    solved = []
    original = Analyzer._solve_fixpoint
    monkeypatch.setattr(Analyzer, "_solve_fixpoint",
                        lambda self, *a, **k: (solved.append(1), original(self, *a, **k))[1])
    r = _prove("""
        s = 0
        while True:
            s = 0
            break
        print(10 // (s + 1))
    """, AR)
    assert r.verdict == "TRUE" and not solved


def test_a_loop_that_changes_the_state_is_iterated(monkeypatch):
    solved = []
    original = Analyzer._solve_fixpoint
    monkeypatch.setattr(Analyzer, "_solve_fixpoint",
                        lambda self, *a, **k: (solved.append(1), original(self, *a, **k))[1])
    _prove("""
        s = 0
        while s < 3:
            s += 1
    """, AR)
    assert solved


# ---------------------------------------------------------- the witness invariant
def _loop_invariant(source: str, line: int):
    import ast
    from tharkun.solve.obligations import source_points
    source = textwrap.dedent(source).lstrip("\n")
    tree = ast.parse(source)
    locs = {loc for node, (_, loc) in source_points(tree, source, "t.py").items() if isinstance(node, ast.stmt)}
    r = prove(source, AR, filename="t.py", invariant_locations=locs)
    assert r.verdict == "TRUE", r.reason
    return [i for i in r.invariants if i.type == "loop_invariant" and i.line == line]


def test_the_loop_head_invariant_holds_at_the_first_visit_too():
    """x is 0 at the first visit and 5 at every later one: claiming ``x >= 5`` at the head (the head alone) is false
    at the first visit, and the independent checker would reject it."""
    (inv,) = _loop_invariant("""
        x = 0
        i = 0
        while i < 3:
            x = 5
            i += 1
        print(10 // (x + 1))
    """, 3)
    assert "x >= 0" in inv.value and "x <= 5" in inv.value
    assert "x >= 5" not in inv.value and "x >= 1" not in inv.value


def test_the_loop_head_invariant_of_a_counting_loop_is_still_confirmed_by_the_checker():
    # tests/test_loop_witness.py runs the full induction check on the same shape; this pins the value
    (inv,) = _loop_invariant("i = 1\nwhile i < 10:\n    7 // i\n    i += 1\n", 2)
    assert "i >= 1" in inv.value and "i <= 10" in inv.value


# ----------------------------------------------- a pass starts where a statement starts
@pytest.mark.parametrize("name, classes", [("p02002.py", AS), ("p02002.py", DT), ("p02580.py", AS)])
def test_a_pass_over_a_loop_does_not_inherit_the_scratch_of_the_pass_before(name, classes):
    """Fuzz programs the first version of A2 turned into 'one allocation site reached twice within a statement' and
    'allocation would invalidate an in-flight heap reference': the epoch and borrowed references an unfinished
    statement leaves behind belong to the pass before it."""
    assert prove((FIX / name).read_text(), classes, time_budget=60).verdict == "TRUE"


def test_fresh_pass_clears_the_statement_scratch():
    an = Analyzer("pass\n", "<p>", time_budget=60)
    an.epoch = {"site"}
    an.borrowed[-1].add("addr")
    an.fresh_pass()
    assert an.epoch == set() and an.borrowed[-1] == set()


# --------------------------------------------------------------- the widening delay
def test_the_default_widening_delay_is_zero():
    assert interp.WIDEN_DELAY == 0


def test_the_widening_delay_is_a_number_of_plain_joins_so_more_delay_is_more_steps(monkeypatch):
    def steps(delay):
        monkeypatch.setattr(interp, "WIDEN_DELAY", delay)
        counted = []
        original = Analyzer.exec_stmt
        monkeypatch.setattr(Analyzer, "exec_stmt", lambda self, *a, **k: (counted.append(1), original(self, *a, **k))[1])
        _prove("x = 0\nwhile x < 100:\n    x += 1\n", AR)
        monkeypatch.setattr(Analyzer, "exec_stmt", original)
        return len(counted)

    assert steps(0) < steps(1) < steps(3)


# ---------------------------------------------------------- the residual of a site
def test_a_site_unsafe_in_the_first_iteration_only_keeps_its_residual():
    """``cur`` is None at the first call and an object after it: the attribute site is unproven by AND, and the
    record keeps the condition under which it raises (the safe later visits say nothing about it)."""
    r = _prove("""
        class P:
            def m(self):
                return self
        def f(xs):
            cur = xs[0]
            for p in xs[1:]:
                cur = cur.m()
            return cur
        f([None, P(), P()])
    """, DT)
    assert r.verdict == "UNKNOWN"
    (rec,) = [x for x in r.records if x["kind"] == "attr" and x["exc"] == "AttributeError" and x["line"] == 8]
    assert rec["status"] == "may_raise" and rec["residual"] == "(cur) is None or not hasattr(cur, 'm')"
