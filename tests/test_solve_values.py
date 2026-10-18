"""``tharkun.solve.values`` ([internal ref]): every transfer function and comparison refinement
is checked against CPython on random intervals -- a concrete result must lie in the
abstract result, or its exception must be in ``may_raise``."""

from __future__ import annotations

import math
import random
import zlib

import pytest

from tharkun.solve import values as V
from tharkun.solve.values import FloatFact, IntFact, Kinds, StrFact

SPECIAL = [0.0, -0.0, 1.0, -1.0, 0.5, 1e-300, -1e-300, 5e-324, 1e300, -1e300, math.inf, -math.inf,
           1.7976931348623157e308, 3.0, 7.25, -2.5]


def _rand_float(rng):
    r = rng.random()
    if r < 0.3:
        return rng.choice(SPECIAL)
    if r < 0.6:
        return rng.uniform(-10, 10)
    return rng.uniform(-1e6, 1e6) * rng.choice([1, 1e-200, 1e200])


def _float_fact(rng):
    a, b = sorted((_rand_float(rng), _rand_float(rng)))
    return FloatFact(a, b, rng.random() < 0.2)


def _int_fact(rng):
    if rng.random() < 0.15:
        return IntFact(rng.choice([None, -5]), rng.choice([None, 7]))
    a, b = sorted((rng.randint(-30, 30), rng.randint(-30, 30)))
    return IntFact(a, b)


def _sample_float(rng, f: FloatFact):
    out = []
    if f.nan:
        out.append(math.nan)
    if f.has_interval():
        out += [f.lo, f.hi]
        lo, hi = max(f.lo, -1.7976931348623157e308), min(f.hi, 1.7976931348623157e308)
        for _ in range(6):
            r = rng.random()
            v = min(max(lo * (1 - r) + hi * r, lo), hi)   # no overflow
            if f.contains(v):
                out.append(v)
        if f.lo <= 0.0 <= f.hi:
            out += [0.0, -0.0]
    return out


def _sample_int(rng, f: IntFact):
    lo = f.lo if f.lo is not None else -10**400
    hi = f.hi if f.hi is not None else 10**400
    out = [lo, hi] + [rng.randint(lo, hi) for _ in range(6)]
    if lo <= 0 <= hi:
        out.append(0)
    return out


def _check(res, raises, fn, xs, ys, what):
    for x in xs:
        for y in ys:
            try:
                z = fn(x, y)
            except Exception as e:  # noqa: BLE001
                assert type(e).__name__ in raises, (what, x, y, type(e).__name__, raises)
                continue
            if res is None:
                continue
            if isinstance(z, complex):
                raise AssertionError((what, x, y, z))
            if isinstance(res, IntFact) and isinstance(z, int):
                assert res.contains(z), (what, x, y, z, res)
            else:
                assert isinstance(res, FloatFact), (what, x, y, z, res)
                assert res.contains(float(z)), (what, x, y, z, res)


OPS = {"add": lambda x, y: x + y, "sub": lambda x, y: x - y, "mul": lambda x, y: x * y,
       "truediv": lambda x, y: x / y, "floordiv": lambda x, y: x // y,
       "mod": lambda x, y: x % y, "pow": lambda x, y: x ** y}


@pytest.mark.parametrize("op", sorted(OPS))
@pytest.mark.parametrize("kinds", ["ff", "ii", "if", "fi"])
def test_binops_are_sound_against_cpython(op, kinds):
    rng = random.Random(zlib.crc32(f"{op}{kinds}".encode()))
    for _ in range(1500):
        a = _float_fact(rng) if kinds[0] == "f" else _int_fact(rng)
        b = _float_fact(rng) if kinds[1] == "f" else _int_fact(rng)
        if op == "pow" and kinds == "ii":
            b = IntFact(*sorted((rng.randint(-3, 6), rng.randint(-3, 6))))
            a = IntFact(*sorted((rng.randint(-9, 9), rng.randint(-9, 9))))
        res, raises = V.binop(op, a, b)
        xs = _sample_float(rng, a) if kinds[0] == "f" else _sample_int(rng, a)
        ys = _sample_float(rng, b) if kinds[1] == "f" else _sample_int(rng, b)
        if op == "pow":
            xs, ys = xs[:5], [y for y in ys if not isinstance(y, int) or abs(y) < 50][:5]
        _check(res, raises, OPS[op], xs, ys, (op, a, b))


@pytest.mark.parametrize("fn, ref", [(V.sqrt, math.sqrt), (V.sin, math.sin), (V.cos, math.cos),
                                     (V.neg, lambda x: -x), (V.abs_, abs)])
def test_unary_ops_are_sound_against_cpython(fn, ref):
    rng = random.Random(7)
    for _ in range(3000):
        a = _float_fact(rng) if rng.random() < 0.7 else _int_fact(rng)
        res, raises = fn(a)
        xs = _sample_float(rng, a) if isinstance(a, FloatFact) else _sample_int(rng, a)
        _check(res, raises, lambda x, _y: ref(x), xs, [None], (fn.__name__, a))


def test_to_float_rounds_outward_and_reports_overflow():
    f, r = V.to_float(IntFact(2**53 + 1, 2**53 + 1))
    assert f.lo <= 2**53 + 1 <= f.hi
    assert V.to_float(IntFact(0, 10**400))[1] == {"OverflowError"}
    assert V.to_float(IntFact(0, 10))[1] == frozenset()


def test_zero_division_is_exact():
    assert V.truediv(FloatFact(1, 2, False), FloatFact(-1, 1, False))[1] == {"ZeroDivisionError"}
    assert V.truediv(FloatFact(1, 2, False), FloatFact(0.5, 1, False))[1] == frozenset()
    assert V.truediv(FloatFact(1, 2, False), FloatFact.const(math.nan))[1] == frozenset()
    assert "ZeroDivisionError" in V.mod(IntFact(1, 5), IntFact(0, 3))[1]
    assert "ZeroDivisionError" not in V.floordiv(IntFact(1, 5), IntFact(1, 3))[1]
    assert "ZeroDivisionError" in V.pow_(FloatFact(0, 1, False), IntFact(-2, -1))[1]


def test_nan_semantics():
    inf = FloatFact.const(math.inf)
    assert V.sub(inf, inf)[0].nan
    assert V.mul(FloatFact(0, 1, False), inf)[0].nan
    assert V.truediv(inf, inf)[0].nan
    assert not V.add(FloatFact(0, 1, False), FloatFact(2, 3, False))[0].nan
    assert V.sin(FloatFact(-math.inf, 0, False))[1] == {"ValueError"}
    s, r = V.sqrt(FloatFact(-1, 4, False))
    assert r == {"ValueError"} and s.lo == 0.0 and s.hi >= 2.0


def test_complex_pow_is_not_modelled():
    res, raises = V.pow_(FloatFact(-2, 2, False), FloatFact(0.5, 0.5, False))
    assert res is None


CMPS = {"<": lambda x, y: x < y, "<=": lambda x, y: x <= y, ">": lambda x, y: x > y,
        ">=": lambda x, y: x >= y, "==": lambda x, y: x == y, "!=": lambda x, y: x != y}


@pytest.mark.parametrize("op", sorted(CMPS))
@pytest.mark.parametrize("kinds", ["ff", "ii", "if"])
def test_refine_compare_keeps_every_pair_on_its_side(op, kinds):
    rng = random.Random(zlib.crc32(f"cmp{op}{kinds}".encode()))
    for _ in range(2000):
        a = _float_fact(rng) if kinds[0] == "f" else _int_fact(rng)
        b = _float_fact(rng) if kinds[1] == "f" else _int_fact(rng)
        at, bt, af, bf = V.refine_compare(op, a, b)
        xs = _sample_float(rng, a) if kinds[0] == "f" else _sample_int(rng, a)
        ys = _sample_float(rng, b) if kinds[1] == "f" else _sample_int(rng, b)
        for x in xs:
            for y in ys:
                ta, tb = (at, bt) if CMPS[op](x, y) else (af, bf)
                assert ta.contains(x) and tb.contains(y), (op, a, b, x, y, ta, tb)


def test_lattice_laws_on_samples():
    rng = random.Random(11)
    for _ in range(2000):
        for mk in (_float_fact, _int_fact):
            a, b = mk(rng), mk(rng)
            j, m = a.join(b), a.meet(b)
            assert a.leq(j) and b.leq(j) and m.leq(a) and m.leq(b)
            assert a.leq(a.widen(b)) and b.leq(a.widen(b))
    assert IntFact.bottom().is_bottom() and FloatFact.bottom().is_bottom()
    assert StrFact.const("a").join(StrFact.const("b")).consts == {"a", "b"}
    assert StrFact.const("a").meet(StrFact.const("b")).is_bottom()
    assert Kinds.of("int", "NoneType").may_be_none() and not Kinds.of("int").may_be_none()


def test_value_product():
    from tharkun.solve.values import Value
    five, s = Value.const(5), Value.const("ab")
    both = five.join(s)
    assert both.contains_value(5) and both.contains_value("ab") and not both.contains_value(6)
    assert not both.contains_value("abc") and not both.contains_value(None)
    assert five.meet(s).is_bottom() and Value.bottom().is_bottom() and not Value.top().is_bottom()
    rng = Value.of_int(IntFact(0, 10)).meet(Value(neq=frozenset({3})))
    assert rng.contains_value(2) and not rng.contains_value(3) and not rng.contains_value(11)
    assert five.leq(Value.of_int(IntFact(0, 10))) and not s.leq(Value.of_int(IntFact(0, 10)))
    assert Value.const(True).contains_value(True) and not Value.const(True).contains_value(False)
    none = Value.const(None)
    assert none.contains_value(None) and not none.contains_value(0)
    w = Value.of_int(IntFact(0, 1)).widen(Value.of_int(IntFact(0, 2)))
    assert w.i.hi is None and w.i.lo == 0


def test_equal_to_follows_python_numeric_equality():
    from tharkun.solve.values import Value
    z = Value.equal_to(0)
    assert z.contains_value(0) and z.contains_value(0.0) and z.contains_value(-0.0) and z.contains_value(False)
    assert not z.contains_value(True) and not z.contains_value("0")
    assert Value.equal_to(True).contains_value(1) and Value.equal_to(True).contains_value(1.0)
    assert Value.equal_to(2.0).contains_value(2) and not Value.equal_to(2.5).contains_value(2)
    assert Value.equal_to(10**400).contains_value(10**400)


@pytest.mark.parametrize("fn, ref", [(V.floor, math.floor), (V.ceil, math.ceil), (V.trunc, math.trunc),
                                     (V.round_, round), (V.exp, math.exp), (V.log, math.log),
                                     (V.atan, math.atan), (V.tanh, math.tanh)])
def test_more_unary_ops_are_sound_against_cpython(fn, ref):
    rng = random.Random(zlib.crc32(fn.__name__.encode()))
    for _ in range(2000):
        a = _float_fact(rng)
        res, raises = fn(a)
        _check(res, raises, lambda x, _y: ref(x), _sample_float(rng, a), [None], (fn.__name__, a))


@pytest.mark.parametrize("fn, ref", [(V.min_, min), (V.max_, max), (V.atan2, math.atan2),
                                     (V.hypot, math.hypot), (V.fsum, lambda x, y: math.fsum([x, y]))])
@pytest.mark.parametrize("kinds", ["ff", "ii", "if"])
def test_more_binary_ops_are_sound_against_cpython(fn, ref, kinds):
    if fn in (V.atan2, V.hypot, V.fsum) and kinds == "ii":
        kinds = "ff"
    if fn in (V.min_, V.max_) and kinds == "if":
        assert fn(IntFact(0, 1), FloatFact(0.5, 2.0, False))[0] is None
        return
    rng = random.Random(zlib.crc32(f"{fn.__name__}{kinds}".encode()))
    for _ in range(1000):
        a = _float_fact(rng) if kinds[0] == "f" else _int_fact(rng)
        b = _float_fact(rng) if kinds[1] == "f" else _int_fact(rng)
        res, raises = fn(a, b)
        xs = _sample_float(rng, a) if kinds[0] == "f" else _sample_int(rng, a)
        ys = _sample_float(rng, b) if kinds[1] == "f" else _sample_int(rng, b)
        _check(res, raises, ref, xs, ys, (fn.__name__, a, b))


@pytest.mark.parametrize("fn, ref", [(V.sq, lambda x: x * x), (V.fabs, math.fabs), (V.log10, math.log10),
                                     (V.round_n, None)])
def test_e_requested_unary_ops_are_sound(fn, ref):
    rng = random.Random(zlib.crc32(("e" + fn.__name__).encode()))
    for _ in range(2000):
        a = _float_fact(rng) if rng.random() < 0.8 else _int_fact(rng)
        if fn is V.round_n:
            n = rng.randint(-3, 5)
            res, raises = fn(a, n)
            f = (lambda x, _y, n=n: round(float(x), n))
        else:
            res, raises = fn(a)
            f = (lambda x, _y: ref(x))
        xs = _sample_float(rng, a) if isinstance(a, FloatFact) else _sample_int(rng, a)
        _check(res, raises, f, xs, [None], (fn.__name__, a))


def test_sums_and_squares_of_non_negatives_stay_non_negative():
    z = FloatFact(0.0, 3.0, False)
    assert V.add(z, z)[0].lo == 0.0 and V.mul(z, z)[0].lo == 0.0
    s = V.sq(FloatFact(-2.0, 3.0, False))[0]
    assert s.lo == 0.0 and s.hi == 9.0
    total = V.add(V.add(V.sq(FloatFact(-1, 1, False))[0], V.sq(FloatFact(-2, 2, False))[0])[0],
                  V.sq(FloatFact(-3, 3, False))[0])[0]
    assert total.lo == 0.0
    res, raises = V.pow_(total, FloatFact.const(-1.5))
    assert res is not None and "ZeroDivisionError" in raises


def test_draw_values_cover_the_reference_model():
    import random as _r
    import sys as _sys
    from tharkun.solve.values import Value, draw_value
    v, r = draw_value("_sv_verifier.nondet_float", [])
    for x in (math.inf, -math.inf, math.nan, -0.0, 1.5):
        assert v.contains_value(x)
    v, _ = draw_value("nondet_int", [])
    assert v.contains_value(10**400) and not v.contains_value(1.0)
    v, _ = draw_value("nondet_str", [])
    assert v.contains_value("") and v.contains_value("abc")
    v, _ = draw_value("random.random", [])
    assert v.contains_value(0.0) and not v.contains_value(1.0)
    rng = _r.Random(5)
    for a, b in [(0.0, 1024.0), (5.0, -3.0), (-_sys.float_info.max, _sys.float_info.max), (2.0, 2.0)]:
        v, _ = draw_value("nondet_float_range", [Value.const(a), Value.const(b)])
        for _ in range(2000):
            x = rng.uniform(a, b)
            assert v.contains_value(x), (a, b, x)
    v, r = draw_value("randint", [Value.const(3), Value.const(1)])
    assert "ValueError" in r
    v, r = draw_value("randint", [Value.const(1), Value.const(6)])
    assert r == frozenset() and v.contains_value(6) and not v.contains_value(7)
    v, r = draw_value("random.randrange", [Value.const(10)])
    assert r == frozenset() and v.contains_value(9) and not v.contains_value(10)
    v, r = draw_value("choice_index", [Value.of_int(IntFact(0, 5))])
    assert "IndexError" in r and v.contains_value(4) and not v.contains_value(5)
