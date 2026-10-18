"""Native CPython semantics and refused witness boundaries for scalar terms."""
import math
import operator

import pytest
from hypothesis import given, settings, strategies as st

from tharkun.solve import terms as t
from tharkun.solve.predicates import check_predicate


BUILTINS = {"type", "int", "bool", "float", "str", "bytes", "list", "tuple", "dict", "set", "frozenset"}


def python_value(text, env):
    return eval(text, {}, env)


@pytest.mark.parametrize("op,native", [(t.eq, operator.eq), (t.ne, operator.ne),
    (t.lt, operator.lt), (t.le, operator.le), (t.gt, operator.gt), (t.ge, operator.ge)])
@settings(max_examples=40, deadline=None)
@given(st.integers(), st.integers())
def test_guarded_integer_comparisons_match_cpython_and_rendered_predicate(op, native, a, b):
    x, y = t.var("x"), t.var("y")
    term = t.and_(t.typeis(x, "int"), t.typeis(y, "int"), op(t.as_int(x), t.as_int(y)))
    text = t.to_python(term, builtin_names=BUILTINS)
    assert isinstance(text, str)
    check_predicate(text, {"x": "object", "y": "object"}, builtin_names=BUILTINS)
    expected = native(a, b)
    assert t.eval_term(term, {"x": a, "y": b}) is expected
    assert python_value(text, {"x": a, "y": b}) is expected


@pytest.mark.parametrize("op,native", [(t.add, operator.add), (t.sub, operator.sub), (t.mul, operator.mul)])
@settings(max_examples=40, deadline=None)
@given(st.integers(), st.integers())
def test_integer_arithmetic_matches_cpython_without_reordering(op, native, a, b):
    value = op(t.as_int(t.var("x")), t.as_int(t.var("y")))
    assert t.eval_term(value, {"x": a, "y": b}) == native(a, b)
    predicate = t.and_(t.typeis(t.var("x"), "int"), t.typeis(t.var("y"), "int"), t.eq(value, t.literal(native(a, b))))
    text = t.to_python(predicate, builtin_names=BUILTINS)
    assert isinstance(text, str)
    assert python_value(text, {"x": a, "y": b}) is True


@pytest.mark.parametrize("op,native", [(t.floordiv, operator.floordiv), (t.mod, operator.mod)])
@settings(max_examples=60, deadline=None)
@given(st.integers(), st.integers().filter(lambda n: n != 0))
def test_signed_integer_floor_mod_match_cpython(op, native, a, b):
    value = op(t.as_int(t.var("x")), t.as_int(t.var("y")))
    assert t.eval_term(value, {"x": a, "y": b}) == native(a, b)
    term = t.and_(t.typeis(t.var("x"), "int"), t.typeis(t.var("y"), "int"), t.eq(value, t.literal(0)))
    assert isinstance(t.to_python(term, builtin_names=BUILTINS), t.Refused)


@pytest.mark.parametrize("a,b,quotient,remainder", [(-7, 3, -3, 2),
    (7, -3, -3, -2), (-7, -3, 2, -1), (7, 3, 2, 1)])
def test_floor_mod_all_dividend_divisor_signs(a, b, quotient, remainder):
    assert t.eval_term(t.floordiv(t.literal(a), t.literal(b)), {}) == quotient == a // b
    assert t.eval_term(t.mod(t.literal(a), t.literal(b)), {}) == remainder == a % b


def test_partial_values_refuse_reachable_zero_or_overflow_not_dead_arms():
    zero = t.eq(t.floordiv(t.literal(1), t.literal(0)), t.literal(0))
    assert isinstance(t.eval_term(zero, {}), t.Refused)
    assert isinstance(t.eval_term(t.mod(t.literal(-7), t.literal(0)), {}), t.Refused)
    assert t.eval_term(t.and_(t.literal(False), zero), {}) is False
    assert t.eval_term(t.or_(t.literal(True), zero), {}) is True
    assert isinstance(t.eval_term(t.i2f(t.literal(1 << 20000)), {}), t.Refused)
    assert t.eval_term(t.i2f(t.literal(3)), {}) == 3.0
    assert isinstance(t.to_python(t.eq(t.i2f(t.literal(3)), t.literal(3.0))), t.Refused)


@pytest.mark.parametrize("source", ["type(x) is float and x == y", "type(x) is float and x != y",
    "type(x) is float and x < y", "type(x) is float and x <= y",
    "type(x) is float and x > y", "type(x) is float and x >= y",
    "type(x) is float and x + y != x", "type(x) is float and x - y != x",
    "type(x) is float and x * y != y",
    "type(x) is float and -x == y", "type(x) is float and not (x > y)"])
@settings(max_examples=50, deadline=None)
@given(st.floats(width=64), st.floats(width=64))
def test_float64_native_and_rendered_python_agree_including_nan(source, a, b):
    source = "type(y) is float and (" + source + ")"
    term = t.from_ast(source, {"x": "object", "y": "object"}, builtin_names=BUILTINS)
    # Nested type guards remain local to their ordered conjunctions.
    assert isinstance(term, t.Term)
    text = t.to_python(term, builtin_names=BUILTINS)
    assert isinstance(text, str)
    env = {"x": a, "y": b}
    assert t.eval_term(term, env) is python_value(source, env)
    assert t.eval_term(term, env) is python_value(text, env)


def test_float64_nan_signed_zero_and_nonfinite_literal_boundaries():
    nan = t.literal(float("nan"))
    assert t.eval_term(t.eq(nan, nan), {}) is False
    assert t.eval_term(t.ne(nan, nan), {}) is True
    assert t.eval_term(t.eq(t.literal(-0.0), t.literal(0.0)), {}) is True
    assert math.copysign(1, t.eval_term(t.neg(t.literal(0.0)), {})) == -1
    assert t.canon(t.literal(-0.0)) != t.canon(t.literal(0.0))
    for value in (float("inf"), -float("inf"), float("nan")):
        assert isinstance(t.to_python(t.eq(t.literal(value), t.literal(0.0))), t.Refused)


def test_bool_conversion_is_exact_numeric_and_not_an_implicit_type_fact():
    term = t.from_ast("type(x) is bool and x == 1", {"x": "object"}, builtin_names=BUILTINS)
    assert t.eval_term(term, {"x": True}) is True
    text = t.to_python(term, builtin_names=BUILTINS)
    assert isinstance(text, str)
    assert python_value(text, {"x": True}) is True
    assert t.eval_term(term, {"x": 1}) is False
    assert t.eval_term(t.from_ast("True + 2", {}), {}) == 3
    assert isinstance(t.to_python(t.eq(t.add(t.b2i(t.literal(True)), t.literal(2)), t.literal(3))), t.Refused)
    assert isinstance(t.to_python(t.typeis(t.b2i(t.literal(True)), "int"), builtin_names=BUILTINS), t.Refused)


@pytest.mark.parametrize("form", [
    lambda g, p: p, lambda g, p: t.and_(p, g), lambda g, p: t.or_(g, p),
    lambda g, p: t.and_(t.not_(g), p), lambda g, p: t.or_(t.and_(g, t.literal(False)), p),
    lambda g, p: t.and_(t.typeis(t.var("x"), "float"), p),
])
def test_cast_rendering_requires_a_matching_guard_on_every_evaluated_arm(form):
    guard = t.typeis(t.var("x"), "int")
    predicate = t.gt(t.as_int(t.var("x")), t.literal(0))
    assert isinstance(t.to_python(form(guard, predicate), types={"x": "int"}, builtin_names=BUILTINS), t.Refused)


def test_guarded_or_arms_are_independent_and_builtin_identity_is_required():
    x = t.var("x")
    term = t.or_(t.and_(t.typeis(x, "int"), t.gt(t.as_int(x), t.literal(0))),
                 t.and_(t.typeis(x, "bool"), t.as_bool(x)))
    text = t.to_python(term, builtin_names=BUILTINS)
    assert isinstance(text, str)
    for value in (1, -1, True, False, None, "x"):
        assert t.eval_term(term, {"x": value}) is python_value(text, {"x": value})
    assert isinstance(t.to_python(term), t.Refused)
    assert isinstance(t.to_python(term, types={"type": "object"}, builtin_names=BUILTINS), t.Refused)
    assert isinstance(t.to_python(term, types={"int": "object"}, builtin_names=BUILTINS), t.Refused)


def test_native_exact_type_tests_never_execute_user_protocols_or_conflate_object():
    class Poison:
        def __eq__(self, other): raise AssertionError("user equality")
        def __bool__(self): raise AssertionError("user truth")
        def __int__(self): raise AssertionError("user integer coercion")
        def __float__(self): raise AssertionError("user float coercion")

    class IntSubclass(int):
        pass

    x = t.var("x")
    value = Poison()
    assert t.eval_term(t.typeis(x, "object"), {"x": object()}) is True
    assert t.eval_term(t.typeis(x, "object"), {"x": value}) is False
    assert t.eval_term(t.typeis(x, "int"), {"x": IntSubclass(1)}) is False
    for cast in (t.as_int, t.as_bool, t.as_f64):
        assert isinstance(t.eval_term(cast(x), {"x": value}), t.Refused)
    assert t.eval_term(t.and_(t.typeis(x, "int"), t.gt(t.as_int(x), t.literal(0))), {"x": value}) is False
    assert isinstance(t.eval_term(t.var("x", t.INT), {"x": True}), t.Refused)
    assert isinstance(t.eval_term(t.var("missing", t.INT), {}), t.Refused)


def test_source_chains_bool_arithmetic_and_binding_current_values_match_native():
    term = t.from_ast("-3 < x <= 7", {"x": "int"})
    assert t.eval_term(term, {"x": 1}) is True
    assert t.eval_term(term, {"x": 8}) is False
    assert isinstance(t.eval_term(term, {"x": 1.0}), t.Refused)
    for source in ("-True", "+False", "True * 3", "True < 2", "False != 0"):
        assert t.eval_term(t.from_ast(source, {}), {}) == python_value(source, {})


@settings(max_examples=80, deadline=None)
@given(st.integers(), st.booleans(), st.floats(width=64), st.sampled_from(["x", "next_x", "δ"]))
def test_every_constructor_canonical_decoder_hypothesis_roundtrip(number, flag, floating, name):
    x, i, b, f = t.var(name), t.literal(number), t.literal(flag), t.literal(floating)
    terms = [x, i, b, f, t.tag("NoneType"), t.typeis(x, "int"),
             t.as_int(x), t.as_bool(x), t.as_f64(x), t.b2i(b), t.i2f(i),
             t.and_(t.typeis(x, "int"), t.gt(t.as_int(x), i)),
             t.or_(t.typeis(x, "float"), t.typeis(x, "NoneType")), t.not_(b), t.neg(i), t.neg(f)]
    terms += [op(i, t.literal(-3)) for op in (t.eq, t.ne, t.lt, t.le, t.gt, t.ge,
                                            t.add, t.sub, t.mul, t.floordiv, t.mod)]
    terms += [op(f, t.literal(-0.0)) for op in (t.eq, t.ne, t.lt, t.le, t.gt, t.ge,
                                              t.add, t.sub, t.mul)]
    for term in terms:
        assert t.from_canon(t.canon(term)) == term


def test_none_tests_refuse_with_specific_reason_even_under_matching_type_guard():
    x = t.var("x")
    none = t.typeis(x, "NoneType")
    for term in (none, t.and_(t.typeis(x, "int"), none)):
        result = t.to_python(term, types={"x": "int"}, builtin_names=BUILTINS)
        assert isinstance(result, t.Refused)
        assert result.reason == "none_test_outside_witness_grammar"
