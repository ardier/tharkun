"""Native/Python/SMT agreement and conservative implication boundaries."""
import math
import operator
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
from hypothesis import given, settings, strategies as st

from tharkun.solve import terms as t
from tharkun.solve import terms_z3 as tz


z3 = pytest.importorskip("z3")
BUILTINS = {"type", "int", "bool", "float"}


def test_optional_projection_import_is_lazy():
    code = "import sys; from tharkun.solve import terms_z3; assert 'z3' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_z3_refuses_without_a_false_yes(monkeypatch):
    monkeypatch.setitem(sys.modules, "z3", None)
    assert isinstance(tz.lower(t.literal(True)), t.Refused)
    assert isinstance(tz.implies([], t.literal(True)), t.Refused)


@pytest.mark.parametrize("op,native", [(t.eq, operator.eq), (t.ne, operator.ne),
    (t.lt, operator.lt), (t.le, operator.le), (t.gt, operator.gt), (t.ge, operator.ge)])
@settings(max_examples=30, deadline=None)
@given(st.integers(), st.integers())
def test_integer_comparisons_three_way(op, native, a, b):
    term = op(t.literal(a), t.literal(b))
    expected = native(a, b)
    assert t.eval_term(term, {}) is expected
    text = t.to_python(term)
    assert isinstance(text, str)
    assert eval(text) is expected
    projected = z3.simplify(tz.lower(term))
    assert z3.is_true(projected) if expected else z3.is_false(projected)


@pytest.mark.parametrize("op,native", [(t.add, operator.add), (t.sub, operator.sub),
    (t.mul, operator.mul)])
@settings(max_examples=30, deadline=None)
@given(st.integers(), st.integers())
def test_integer_arithmetic_three_way(op, native, a, b):
    value = op(t.literal(a), t.literal(b))
    expected = native(a, b)
    assert t.eval_term(value, {}) == expected
    assert z3.simplify(tz.lower(value)).as_long() == expected
    text = t.to_python(t.eq(value, t.literal(expected)))
    assert isinstance(text, str)
    assert eval(text) is True


@pytest.mark.parametrize("a,b,q,r", [(-7, 3, -3, 2), (7, -3, -3, -2),
    (-7, -3, 2, -1), (7, 3, 2, 1), (-(1 << 500), -13, (-(1 << 500)) // -13, (-(1 << 500)) % -13)])
def test_floor_mod_python_signs_native_and_z3(a, b, q, r):
    for op, expected in ((t.floordiv, q), (t.mod, r)):
        value = op(t.literal(a), t.literal(b))
        assert t.eval_term(value, {}) == expected
        assert z3.simplify(tz.lower(value)).as_long() == expected
        assert isinstance(t.to_python(t.eq(value, t.literal(expected))), t.Refused)


def test_floor_mod_symbolic_definedness_requires_an_independent_nonzero_guard():
    x, y = t.var("x", t.INT), t.var("y", t.INT)
    value = t.floordiv(x, y)
    assert isinstance(tz.lower(value), t.Refused)
    assert isinstance(tz.implies([], t.eq(value, t.literal(0))), t.Refused)
    guarded = t.and_(t.ne(y, t.literal(0)), t.eq(value, value))
    assert not isinstance(tz.lower(guarded), t.Refused)
    assert tz.implies([t.ne(y, t.literal(0))], t.eq(value, value)) == "yes"
    # A partial fact cannot use its own arbitrary SMT division-by-zero value.
    partial_fact = t.eq(value, t.literal(0))
    assert isinstance(tz.implies([partial_fact], partial_fact), t.Refused)
    assert tz.implies([partial_fact, t.eq(y, t.literal(2))], partial_fact) == "yes"
    for op in (t.floordiv, t.mod):
        bad = op(t.literal(1), t.literal(0))
        assert isinstance(tz.lower(bad), t.Refused)
        assert isinstance(tz.implies([t.literal(False)], t.eq(bad, t.literal(0))), t.Refused)


@pytest.mark.parametrize("op,native", [(t.eq, operator.eq), (t.ne, operator.ne),
    (t.lt, operator.lt), (t.le, operator.le), (t.gt, operator.gt), (t.ge, operator.ge)])
@settings(max_examples=20, deadline=None)
@given(st.floats(width=64), st.floats(width=64))
def test_float64_comparisons_native_python_and_ieee_z3(op, native, a, b):
    term, expected = op(t.literal(a), t.literal(b)), native(a, b)
    assert t.eval_term(term, {}) is expected
    projected = z3.simplify(tz.lower(term))
    assert z3.is_true(projected) if expected else z3.is_false(projected)
    text = t.to_python(term)
    if math.isfinite(a) and math.isfinite(b):
        assert isinstance(text, str)
        assert eval(text) is expected
    else:
        assert isinstance(text, t.Refused)


@pytest.mark.parametrize("op,native", [(t.add, operator.add), (t.sub, operator.sub),
    (t.mul, operator.mul)])
@settings(max_examples=20, deadline=None)
@given(st.floats(width=64), st.floats(width=64))
def test_float64_arithmetic_uses_rne_not_real(op, native, a, b):
    value, expected = op(t.literal(a), t.literal(b)), native(a, b)
    actual, projected = t.eval_term(value, {}), tz.lower(value)
    assert projected.sort() == z3.Float64()
    if math.isnan(expected):
        assert math.isnan(actual)
        assert z3.is_true(z3.simplify(z3.fpIsNaN(projected)))
    else:
        assert actual == expected
        assert z3.is_true(z3.simplify(z3.fpEQ(projected, tz.lower(t.literal(expected)))))
        if expected == 0.0:
            assert z3.is_true(z3.simplify(z3.fpIsNegative(projected))) == (math.copysign(1, expected) < 0)
    text = t.to_python(t.eq(value, t.literal(expected)))
    if math.isfinite(a) and math.isfinite(b) and math.isfinite(expected):
        assert isinstance(text, str)
        assert eval(text) is True
    else:
        assert isinstance(text, t.Refused)


def test_nan_nonreflexivity_and_signed_zero_comparison_are_not_structural_equality():
    nan = t.literal(float("nan"))
    assert tz.implies([], t.ne(nan, nan)) == "yes"
    assert tz.implies([], t.eq(nan, nan)) == "no"
    assert tz.implies([], t.eq(t.literal(-0.0), t.literal(0.0))) == "yes"
    assert z3.is_true(z3.simplify(z3.fpIsNegative(tz.lower(t.literal(-0.0)))))
    assert z3.is_false(z3.simplify(z3.fpIsNegative(tz.lower(t.literal(0.0)))))
    assert z3.is_true(z3.simplify(z3.fpIsNegative(tz.lower(t.neg(t.literal(0.0))))))
    x = t.var("x")
    xf, guard = t.as_f64(x), t.typeis(x, "float")
    assert tz.implies([guard], t.eq(xf, xf)) == "no"  # NaN remains possible.
    assert tz.implies([guard, t.ne(xf, xf)], t.not_(t.gt(xf, t.literal(0.0)))) == "yes"


def test_int_to_float_is_exact_when_concrete_and_refuses_unchecked_overflow_domain():
    for value in (0, -1, 3, 2**53 + 1):
        term = t.i2f(t.literal(value))
        assert z3.is_true(z3.simplify(z3.fpEQ(tz.lower(term), tz.lower(t.literal(float(value))))))
        assert isinstance(t.to_python(t.eq(term, t.literal(float(value)))), t.Refused)
    assert isinstance(tz.lower(t.i2f(t.literal(1 << 20000))), t.Refused)
    assert isinstance(tz.lower(t.i2f(t.var("x", t.INT))), t.Refused)


def test_known_exact_sorts_type_tests_and_explicit_bool_conversion():
    assert z3.is_true(z3.simplify(tz.lower(t.typeis(t.var("x", t.INT), "int"))))
    assert z3.is_false(z3.simplify(tz.lower(t.typeis(t.var("x", t.INT), "bool"))))
    assert z3.simplify(tz.lower(t.b2i(t.literal(True)))).as_long() == 1
    assert z3.simplify(tz.lower(t.neg(t.b2i(t.literal(True))))).as_long() == -1
    assert tz.implies([], t.eq(t.tag("int"), t.tag("bool"))) == "no"


def test_py_tag_universe_keeps_other_distinct_from_exact_object():
    x = t.var("x")
    tags = ["int", "bool", "float", "str", "bytes", "list", "tuple", "dict",
            "set", "frozenset", "NoneType", "object"]
    assert tz.implies([], t.or_(*(t.typeis(x, tag) for tag in tags))) == "no"
    assert tz.implies([t.typeis(x, "object")], t.not_(t.typeis(x, "int"))) == "yes"
    assert tz.implies([t.not_(t.typeis(x, "object"))], t.not_(t.typeis(x, "int"))) == "no"


def test_py_actual_values_link_across_record_facts_and_goal():
    x, y = t.var("x"), t.var("y")
    guards = [t.typeis(x, "int"), t.typeis(y, "int")]
    facts = [t.eq(t.as_int(x), t.literal(3)), t.eq(t.as_int(y), t.literal(-2)), *guards]
    goal = t.gt(t.add(t.as_int(x), t.as_int(y)), t.literal(0))
    assert tz.implies(facts, goal) == "yes"
    assert tz.implies(facts, t.not_(goal)) == "no"
    native = t.and_(*guards, goal)
    text = t.to_python(native, builtin_names=BUILTINS)
    assert isinstance(text, str)
    assert t.eval_term(native, {"x": 3, "y": -2}) is eval(text, {}, {"x": 3, "y": -2}) is True
    mixed_goal = t.gt(t.var("x", t.INT), t.literal(0))
    assert isinstance(tz.implies(facts, mixed_goal), t.Refused)


@pytest.mark.parametrize("make", [
    lambda g, p: p, lambda g, p: t.and_(p, g), lambda g, p: t.or_(g, p),
    lambda g, p: t.and_(t.not_(g), p),
    lambda g, p: t.and_(t.typeis(t.var("x"), "float"), p),
])
def test_missing_or_nondominating_cast_guards_refuse_without_invented_facts(make):
    g, p = t.typeis(t.var("x"), "int"), t.gt(t.as_int(t.var("x")), t.literal(0))
    term = make(g, p)
    assert isinstance(tz.lower(term), t.Refused)
    assert isinstance(tz.implies([], term), t.Refused)
    assert isinstance(tz.implies([term], p), t.Refused)


def test_unknown_solver_answer_stays_unknown_and_resource_policy_is_validated(monkeypatch):
    original = z3.Solver

    class UnknownSolver:
        def __init__(self, **kwargs): self.solver = original(**kwargs)
        def __getattr__(self, name): return getattr(self.solver, name)
        def check(self): return z3.unknown

    monkeypatch.setattr(z3, "Solver", UnknownSolver)
    assert tz.implies([], t.literal(True)) == "unknown"
    for limit in (0, -1, True, "10000"):
        assert isinstance(tz.implies([], t.literal(True), resource_limit=limit), t.Refused)


def test_unprojectable_or_nonboolean_input_refuses_not_true():
    for value in (t.Refused("unsupported"), None, t.var("K", t.INT), t.var("tag", t.TAG)):
        assert isinstance(tz.lower(value), t.Refused)
    assert isinstance(tz.implies(["True"], t.literal(True)), t.Refused)
    assert isinstance(tz.implies([], t.literal(1)), t.Refused)
    assert isinstance(tz.implies([t.literal(1)], t.literal(True)), t.Refused)
    assert isinstance(tz.implies([t.literal(True)] * 2048, t.literal(True)), t.Refused)


def test_concurrent_implications_own_contexts_and_same_name_query_bindings():
    def query(number):
        x = t.var("x")
        facts = [t.typeis(x, "int"), t.eq(t.as_int(x), t.literal(number))]
        closed_fp = t.eq(t.add(t.literal(float(number)), t.literal(1.0)), t.literal(float(number + 1)))
        goal = t.and_(t.gt(t.as_int(x), t.literal(0)), closed_fp)
        expected = "yes" if number > 0 else "no"
        for _ in range(4):
            assert tz.implies(facts, goal) == expected
        return expected

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(query, number) for number in (-7, -1, 0, 1, 2, 7, -3, 3)]
        # Simultaneously exercise an unrelated default-context legacy query.
        legacy = z3.Int("x")
        solver = z3.Solver()
        solver.add(legacy == 11)
        assert solver.check() == z3.sat
        assert solver.model().eval(legacy).as_long() == 11
        assert [future.result() for future in futures] == ["no", "no", "no", "yes", "yes", "yes", "no", "yes"]
