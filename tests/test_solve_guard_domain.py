"""solve/guard_domain: sound per-variable arm facts from a guard ([internal ref], slice 2).

The soundness check is brute force.  Random guards over ``x`` and ``y`` are
evaluated on a pool of concrete values (ints near the bounds, ±0.0, NaN, ±inf,
subnormals, bools, None, str, bytes and containers).  Every value that takes an
arm must be admitted by that arm's :class:`Domain`, and every atom from
:func:`atoms` must hold on it.
"""

from __future__ import annotations

import ast
import math
import random

import pytest

from tharkun.solve.fact_env import FactEnv, refine_all, to_value
from tharkun.solve.guard_domain import DOMAIN_KINDS, atoms, domain_of, domains, refine, refine_env
from tharkun.solve.values import IntFact, Kinds, StrFact, Value

NAN = float("nan")
POOL = [
    -(10 ** 20), -256, -11, -10, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 97, 255, 256,
    2 ** 53 + 1, 10 ** 400,
    -2.5, -1.0, -0.0, 0.0, 5e-324, 0.5, 1.0, 2.0, 2.5, 5.0, 9.999, 10.0, 1e300,
    math.inf, -math.inf, NAN,
    False, True, None,
    "", "a", "ab", "abc", "b", "m", "n", "z", "hello",
    b"", b"a", b"ab", b"\x01", b"z",
    [], [1], [1, 2], ["a"], (), (1, 2), (1,), {1}, frozenset({"a"}), {"a": 1},
]
CONSTS = [-10, -1, 0, 1, 2, 5, 10, 255, 2 ** 53 + 1, -0.0, 0.0, 2.5, 5.0, -2.5, 1e300,
          True, False, None, "", "a", "ab", "m", b"", b"a", b"\x01"]


def _lit(c) -> str:
    return repr(c)


def _atom(rng: random.Random, v: str) -> str:
    k = rng.randrange(14)
    c = _lit(rng.choice(CONSTS))
    op = rng.choice(["<", "<=", ">", ">=", "==", "!="])
    if k <= 3:
        return f"{v} {op} {c}" if rng.random() < 0.7 else f"{c} {op} {v}"
    if k == 4:
        items = ", ".join(_lit(rng.choice(CONSTS)) for _ in range(rng.randrange(0, 4)))
        coll = rng.choice([f"({items},)" if items else "()", f"[{items}]",
                           f"{{{items}}}" if items else "()"])
        return f"{v} {rng.choice(['in', 'not in'])} {coll}"
    if k == 5:
        coll = rng.choice(['"abc"', 'b"ab"', "range(-2, 5)", "range(10, 0, -3)", "range(0)"])
        return f"{v} {rng.choice(['in', 'not in'])} {coll}"
    if k == 6:
        return f"{c} {rng.choice(['in', 'not in'])} {v}"
    if k == 7:
        return f"len({v}) {op} {rng.choice(['0', '1', '2', '2.5', 'True'])}"
    if k == 8:
        return v
    if k == 9:
        return f"{v} {rng.choice(['is', 'is not'])} {rng.choice(['None', 'True', 'False'])}"
    if k == 10:
        a, b = sorted(rng.sample([-3, 0, 1, 2.5, 5, 10], 2))
        return f"{a} {rng.choice(['<', '<='])} {v} {rng.choice(['<', '<='])} {b}"
    if k == 11:
        return rng.choice([f"{v} % 2 == 0", f"{v} == y" if v == "x" else f"{v} < x",
                           f"abs({v}) > 1", f"{v} + 1 > 2"])
    if k == 12:
        return f'{v} {op} "{rng.choice(["a", "b", "m"])}"'
    return f"{v} {op} {c}"


def _guard(rng: random.Random, depth: int = 0) -> str:
    r = rng.random()
    if depth >= 3 or r < 0.4:
        return _atom(rng, "x" if rng.random() < 0.8 else "y")
    if r < 0.55:
        return f"not ({_guard(rng, depth + 1)})"
    op = rng.choice(["and", "or"])
    return " {} ".format(op).join(f"({_guard(rng, depth + 1)})" for _ in range(rng.randint(2, 3)))


def _eval(src: str, x, y):
    try:
        return bool(eval(src, {"__builtins__": {"len": len, "range": range, "abs": abs}},
                         {"x": x, "y": y}))
    except Exception:
        return None


def _check_guard(src: str) -> None:
    test = ast.parse(src, mode="eval").body
    dx_t, dx_f = domain_of(test, "x"), domain_of(test, "x", negate=True)
    dy_t, dy_f = domain_of(test, "y"), domain_of(test, "y", negate=True)
    at_t, at_f = atoms(test), atoms(test, negate=True)
    ys = POOL[::5] + [0, "a", None]
    for x in POOL:
        for y in ys:
            r = _eval(src, x, y)
            if r is None:
                continue
            dx, dy, got = (dx_t, dy_t, at_t) if r else (dx_f, dy_f, at_f)
            assert dx.admits(x), (src, "x", x, r, dx)
            assert dy.admits(y), (src, "y", y, r, dy)
            for var, op, val in got:
                _check_atom(src, var, op, val, {"x": x, "y": y}[var], r)


def _check_atom(src, var, op, val, v, r) -> None:
    """``(var, op, val)`` must hold for ``v`` (ordering atoms: on real ints, as
    ``fact_env.refine`` applies them)."""
    if op == "truthy":
        assert bool(v) == val, (src, var, v, r)
        return
    if op == "container_len_eq":
        assert len(v) == val, (src, var, v, r)
        return
    if op == "container_len_ne":
        assert len(v) != val, (src, var, v, r)
        return
    if op in ("<", "<=", ">", ">="):
        if type(v) is not int:
            return
    ok = {"==": lambda: v == val, "!=": lambda: v != val, "in": lambda: any(v == e for e in val),
          "not in": lambda: not any(v == e for e in val), "<": lambda: v < val, "<=": lambda: v <= val,
          ">": lambda: v > val, ">=": lambda: v >= val}[op]()
    assert ok, (src, var, op, val, v, r)


@pytest.mark.parametrize("seed", range(8))
def test_random_guards_sound(seed):
    rng = random.Random(1021_0000 + seed)
    for _ in range(150):
        _check_guard(_guard(rng))


@pytest.mark.parametrize("src", [
    "x == 0", "x != 0", "x == 1", "x == True", "x == -0.0", "x < 2.5", "x <= 2.5", "x > -0.0",
    "x >= 2 ** 53 + 1", "x < 1e300", "x in (0, 'a', None)", "x not in (1, 2)", "x in 'abc'",
    "x in b'ab'", "x in range(3)", "x in range(10, 0, -3)", "'a' in x", "1 not in x",
    "len(x) == 0", "len(x) > 1", "len(x) != 2", "x", "not x", "x is None", "x is not None",
    "x is True", "x is not False", "0 < x <= 10", "not (0 < x < 10)", "x < 'm'", "x >= b'a'",
    "x == 'ab' or x == 'abc'", "(x := 5) > 3 and x < 0", "x > 3 and f(x)",
    "x == (1, 2)", "x != (1, 2)",
])
def test_named_guards_sound(src):
    _check_guard(src)


def _d(src, var="x", **kw):
    return domain_of(src, var, **kw)


def test_equality_crosses_numeric_types():
    d = _d("x == 0")
    for v in (0, 0.0, -0.0, False):
        assert d.admits(v)
    for v in (1, True, "0", None, 0.5):
        assert not d.admits(v)
    assert d.kind == "int_set" and d.members == frozenset({0})
    f = _d("x == 0", negate=True)
    assert not any(f.admits(v) for v in (0, 0.0, -0.0, False))
    assert f.admits(None) and f.admits(1)


def test_open_world_keeps_other_types():
    t, f = refine(ast.parse("x == 5", mode="eval").body, "x")
    assert t.kinds == Kinds.top()               # a Fraction(5) takes the arm too
    assert t.i == IntFact(5, 5) and t.s.is_bottom() and t.bools == frozenset()
    closed, _ = refine(ast.parse("x == 5", mode="eval").body, "x", Value.only("int"))
    assert closed.kinds.names == frozenset({"int"}) and closed.i == IntFact(5, 5)


def test_float_bounds_are_exact():
    d = _d("x < 2.5")
    assert d.value.i == IntFact(None, 2)
    assert d.value.f.hi == math.nextafter(2.5, -math.inf) and not d.value.f.nan
    f = _d("x < 2.5", negate=True)
    assert f.value.i == IntFact(3, None) and f.value.f.nan and f.admits(NAN)
    assert _d("x > 9007199254740993").value.i == IntFact(2 ** 53 + 2, None)
    assert _d("x > 9007199254740993").value.f.lo == 2.0 ** 53 + 2
    assert not _d("x < 0.0").admits(-0.0) and _d("x <= 0.0").admits(-0.0)


def test_type_error_empties_both_arms():
    t, f = refine(ast.parse("x < 5", mode="eval").body, "x")
    for v in (t, f):
        assert v.s.is_bottom() and v.by.is_bottom() and not v.contains_value(None)
    t, f = refine(ast.parse("x < 'm'", mode="eval").body, "x")
    assert t.i.is_bottom() and f.i.is_bottom() and t.s == StrFact.top()


def test_lex_filters_finite_prior():
    prior = Value.only("str", s=StrFact(frozenset({"a", "m", "z"})))
    t, f = refine(ast.parse("x < 'm'", mode="eval").body, "x", prior)
    assert t.s.consts == frozenset({"a"}) and f.s.consts == frozenset({"m", "z"})
    assert t.length == IntFact(1, 1)
    assert _d("x < 'm'").kind == "lex_range"


def test_str_equality_is_the_chaos_shape():
    env = {"cmd": Value.only("str"), "n": Value.only("NoneType").join(Value.only("int"))}
    et, ef = refine_env(ast.parse("cmd in ('go', 'stop') and n is not None", mode="eval").body, env)
    assert et["cmd"].s.consts == frozenset({"go", "stop"}) and et["cmd"].length == IntFact(2, 4)
    assert et["n"].kinds.names == frozenset({"int"}) or not et["n"].contains_value(None)
    assert ef["cmd"] == env["cmd"]


def test_len_and_membership():
    d = _d("len(x) > 1")
    assert d.value.length == IntFact(2, None) and d.kind == "container_length"
    assert d.admits([1, 2]) and not d.admits([1]) and not d.admits(5)
    d = _d("'a' in x")
    assert d.kind == "set_membership" and d.value.contains == frozenset({"a"})
    assert d.admits(["a"]) and not d.admits(["b"])


def test_truthiness_and_none():
    assert _d("x", negate=True).members == frozenset({0, "", b"", None})
    assert not _d("x").admits(0.0) and _d("x").admits(NAN)
    assert _d("x is None").kind == "is_none"
    assert not _d("x is not None").admits(None)


def test_residual_and_rebinding():
    d = _d("x % 2 == 0 and x > 3")
    assert d.value.i == IntFact(4, None) and len(d.residual) == 1
    assert _d("x % 2 == 0").kind == "guard_only"
    assert _d("(x := 5) > 3 and x < 0").value == Value.top()
    assert _d("len(x) > 0 and f(x)").value.length == IntFact(0, None)   # a call may mutate
    assert _d("x > 3 and f()", local=False).value == Value.top()


def test_or_of_disjoint_intervals_joins():
    d = _d("x < 0 or x > 10", negate=True)
    assert d.value.i == IntFact(0, 10) and d.kind == "int_range_closed"
    d = _d("x == 1 or x == 5 or x == 9")
    assert d.members == frozenset({1, 5, 9}) and d.kind == "int_set"
    assert not d.admits(2)


def test_labels_are_eregion_kinds():
    for src in ("x == 0", "x < 5", "x >= 5", "x <= 5", "x != 3", "x == 2.5", "x > 1.5",
                "x == 'a'", "x != 'a'", "x == b'a'", "x != b'a'", "x < 'm'", "x is True",
                "x is None", "len(x) == 2", "'a' in x", "x % 2", "x == y"):
        assert _d(src).kind in DOMAIN_KINDS, src
    assert _d("y == 1").kind == "top"


def test_domains_and_atoms_feed_fact_env():
    ds = domains("0 <= i < n and s == 'ab'")
    assert set(ds) == {"i", "n", "s"} and ds["i"].value.i == IntFact(0, None)
    got = atoms("x in (1, 2) and len(s) == 3 and not (y is None) and 0 <= z < 4")
    assert ("s", "container_len_eq", 3) in got and ("y", "!=", None) in got
    env = refine_all(FactEnv(), got)
    assert env.possible and env.facts["x"].eq == {1, 2}
    assert to_value(env.facts["z"]).i == IntFact(0, 3)
    assert atoms("x == 1 or x == 2") == [("x", "in", {1, 2})]
    assert atoms("x is not True") == []          # 1 is not True, but 1 == True
