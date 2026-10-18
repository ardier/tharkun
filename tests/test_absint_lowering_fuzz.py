"""[internal ref]: soundness fuzz of absint on the lowered program.

``prove`` answers TRUE only with a proof; a TRUE for a program that then raises one of the
classes is a wrong TRUE (32 points).  Random programs built from the shapes the lowering
touches (list, set and dict comprehensions with filters, nested comprehensions, ranges, indexing,
division, ``len``/``sum`` of a comprehension) are run for real; every program ``prove`` calls
TRUE must not raise.  The programs call ``main`` with concrete arguments so absint is as precise as
it gets: a TRUE is then the common case, and a wrong one has nowhere to hide.
"""

from __future__ import annotations

import random

import pytest

from tharkun.solve.absint import prove

CLASSES = ("ZeroDivisionError", "IndexError", "KeyError", "TypeError")


class _Gen:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def atom(self, vars_):
        r = self.rng
        pick = r.random()
        if pick < 0.30:
            return str(r.randint(-2, 4))
        if pick < 0.55 and vars_:
            return r.choice(vars_)
        if pick < 0.75:
            return f"xs[{r.randint(-1, 3)}]"
        return "n"

    def expr(self, vars_, depth=0):
        r = self.rng
        if depth >= 3:
            return self.atom(vars_)
        pick = r.random()
        if pick < 0.20:
            return self.atom(vars_)
        if pick < 0.40:
            op = r.choice(["+", "-", "*", "//", "%"])
            return f"({self.expr(vars_, depth + 1)} {op} {self.expr(vars_, depth + 1)})"
        if pick < 0.58:
            return f"len({self.comp(vars_, depth)})"
        if pick < 0.70:
            return f"sum({self.comp(vars_, depth)})"
        if pick < 0.78:
            return f"({self.expr(vars_, depth + 1)} if {self.expr(vars_, depth + 1)} else {self.expr(vars_, depth + 1)})"
        if pick < 0.86:
            return f"{self.comp(vars_, depth)}[{self.expr(vars_, depth + 1)}]"
        return f"max({self.expr(vars_, depth + 1)}, {self.expr(vars_, depth + 1)})"

    def comp(self, vars_, depth):
        r = self.rng
        name = r.choice(["a", "b", "x"])
        inner = [*vars_, name]
        source = r.choice(["xs", "range(3)", "range(n)", f"[{self.atom(vars_)}, {self.atom(vars_)}]"])
        cond = f" if {self.expr(inner, depth + 2)}" if r.random() < 0.35 else ""
        kind = r.random()
        if kind < 0.7:
            return f"[{self.expr(inner, depth + 1)} for {name} in {source}{cond}]"
        if kind < 0.85:
            return f"list({{{self.expr(inner, depth + 1)} for {name} in {source}{cond}}})"
        return f"list({{{name}: {self.expr(inner, depth + 1)} for {name} in {source}{cond}}}.values())"

    def program(self):
        r = self.rng
        a = self.expr(["u", "n"], 0)
        b = self.expr(["u", "n"], 0)
        shape = r.random()
        if shape < 0.4:
            body = f"    u = {a}\n    return u, {b}"
        elif shape < 0.7:
            body = f"    u = 1\n    w = {a}\n    return u, w, {b}"
        else:
            body = f"    u = 2\n    if {a}:\n        return {b}\n    return -1"
        args = r.choice(["([1, 2, 3], 2)", "([], 0)", "([0, 5], 3)", "([2], -1)", "([3, 1, 2, 0], 4)"])
        return f"def main(xs, n):\n{body}\nmain{args}\n"


def _raised(source: str):
    namespace: dict = {}
    try:
        exec(compile(source, "<fuzz>", "exec"), namespace)
    except BaseException as exc:  # noqa: BLE001 -- the point is to see what happens
        return type(exc).__name__
    return None


@pytest.mark.parametrize("seed", range(12))
def test_a_program_absint_proves_safe_does_not_raise(seed):
    gen = _Gen(random.Random(11070 + seed))
    proved = 0
    for _ in range(25):
        source = gen.program()
        for cls in CLASSES:
            result = prove(source, [cls], filename="<fuzz>", time_budget=20.0)
            if result.verdict != "TRUE":
                continue
            proved += 1
            assert _raised(source) != cls, f"WRONG TRUE for {cls}:\n{source}"
    assert proved > 0
