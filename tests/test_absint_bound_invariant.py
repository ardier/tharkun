"""A variable the zone does not have is named by no value (``State.rewrite_bounds_away`` / ``shift_bounds`` skip it).

The walk those two methods do over every value of every frame and object is skipped for a variable the zone does
not have (77% of the calls of a chaos run).  That is exact only if no bound and no float term ever names such a
variable.  This test runs the full walk anyway, on a copy, for every skipped call of a set of programs (the
absint fixtures and heap-heavy programs), and fails if the walk would have changed any value.  The walk is
``State._rewrite_bounds_away`` / ``State._shift_bounds``, the code the public methods run for a variable the zone has.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from tharkun.solve.absint import PROPERTY_CLASSES, prove
from tharkun.solve.absint import state as S

FIX = Path(__file__).parent / "fixtures" / "absint"

PROGRAMS = [
    textwrap.dedent("""
        class V:
            def __init__(self, x, y):
                self.x = x
                self.y = y
            def add(self, o):
                return V(self.x + o.x, self.y + o.y)
        d = [V(1, 2), V(3, 4), V(5, 6)]
        for i in range(2):
            for j in range(1, 3):
                d[j] = d[j].add(d[j - 1])
        n = 0
        for v in d:
            n += v.x
        assert n >= 0
    """),
    textwrap.dedent("""
        import _sv_verifier
        xs = [0] * 5
        i = 0
        while i < 5:
            xs[i] = i * 2
            i += 1
        t = [x for x in xs]
        s = 0.5
        for k in range(3):
            s = s * 2.0 + 1.0
        a = _sv_verifier.nondet_int()
        if 0 <= a < 5:
            assert xs[a] >= 0
    """),
    textwrap.dedent("""
        class A:
            def __init__(self, v):
                self.v = v
        def mk(n):
            return A(n + 1)
        p = A(0)
        q = [p, mk(1)]
        for i in range(3):
            q.append(mk(i))
            p = mk(p.v)
        assert len(q) >= 2
        assert p.v >= 0
    """),
]


def _fixture_sources():
    for sub in ("targets", "never_true", "never_true_ar", "never_true_dt"):
        for path in sorted((FIX / sub).glob("*.py")):
            yield path.name, path.read_text()


def _run_checked(src: str):
    violations: list = []
    skipped = [0]
    orig_rb, orig_sb = S.State.rewrite_bounds_away, S.State.shift_bounds

    def changed_by(self, method, *args):
        probe = self.copy()
        changed = [False]
        orig_map = S.State._map_values

        def mapping(st, fn):
            def tracked(v):
                r = fn(v)
                changed[0] = changed[0] or r is not v
                return r
            return orig_map(st, tracked)

        S.State._map_values = mapping
        try:
            method(probe, *args)
        finally:
            S.State._map_values = orig_map
        return changed[0]

    def rb(self, zv, ob=None):
        if zv not in self.zone.vars:
            skipped[0] += 1
            if changed_by(self, S.State._rewrite_bounds_away, zv, None):
                violations.append(("rewrite_bounds_away", zv))
        return orig_rb(self, zv, ob)

    def sb(self, zv, k, ob=None):
        if zv not in self.zone.vars:
            skipped[0] += 1
            if changed_by(self, S.State._shift_bounds, zv, k, None):
                violations.append(("shift_bounds", zv))
        return orig_sb(self, zv, k, ob)

    S.State.rewrite_bounds_away, S.State.shift_bounds = rb, sb
    try:
        for classes in PROPERTY_CLASSES.values():
            prove(src, classes, filename="<program>", time_budget=20.0)
    finally:
        S.State.rewrite_bounds_away, S.State.shift_bounds = orig_rb, orig_sb
    return violations, skipped[0]


@pytest.mark.parametrize("n", range(len(PROGRAMS)))
def test_an_absent_variable_is_named_by_no_value_in_a_heap_heavy_program(n):
    violations, skipped = _run_checked(PROGRAMS[n])
    assert not violations, violations[:3]
    assert skipped > 0


@pytest.mark.parametrize("name", [name for name, _src in _fixture_sources()])
def test_an_absent_variable_is_named_by_no_value_in_the_absint_fixtures(name):
    src = dict(_fixture_sources())[name]
    violations, _skipped = _run_checked(src)
    assert not violations, violations[:3]
