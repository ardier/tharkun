"""Scalar term identity and construction boundaries; no solver semantics yet."""
from dataclasses import FrozenInstanceError

import pytest

from tharkun.solve import terms as t


def test_sorts_and_literals_do_not_collapse_python_tags():
    assert set(t.Sort) == {t.INT, t.BOOL, t.F64, t.TAG, t.PY, t.SEQ, t.MAP}
    assert t.literal(True).sort == t.BOOL
    assert t.literal(1).sort == t.INT
    assert t.canon(t.literal(True)) != t.canon(t.literal(1))
    assert t.canon(t.literal(1.0)) == '(term 1 (lit F64 "0x1.0000000000000p+0"))'


def test_canonical_identity_preserves_short_circuit_order_and_version():
    x = t.var("x")
    guard = t.typeis(x, "int")
    condition = t.gt(t.as_int(x), t.literal(0))
    assert t.canon(t.and_(guard, condition)) == (
        '(term 1 (and Bool (typeis Bool (var Py "x") (tag Tag "int")) '
        '(gt Bool (as_int Int (var Py "x")) (lit Int 0))))')
    assert t.canon(t.and_(guard, condition)) != t.canon(t.and_(condition, guard))
    assert t.canon(t.or_(guard, condition)) != t.canon(t.or_(condition, guard))


def test_canonicalization_does_not_reorder_arithmetic():
    a, b = t.var("a", t.INT), t.var("b", t.INT)
    assert t.canon(t.add(a, b)) != t.canon(t.add(b, a))


@pytest.mark.parametrize("value", [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 5e-324])
def test_f64_literals_use_hex_not_real_numbers(value):
    term = t.literal(value)
    assert term == t.Term("lit", t.F64, (value.hex(),))
    assert value.hex() in t.canon(term)
    assert "Real" not in t.canon(term)


def test_signed_zero_identity_is_preserved():
    assert t.canon(t.literal(0.0)) != t.canon(t.literal(-0.0))


@pytest.mark.parametrize("name", ["int", "bool", "float", "NoneType", "str", "bytes"])
def test_scalar_tags_need_no_string_or_container_value_sort(name):
    assert t.typeis(t.var("x"), name).sort == t.BOOL
    assert t.tag(name).sort == t.TAG


def test_unbounded_integer_serialization_avoids_runtime_digit_limit():
    value = 1 << 20000
    positive = t.canon(t.literal(value))
    negative = t.canon(t.literal(-value))
    assert positive.startswith("(term 1 (lit Int ")
    assert len(positive) > 6000
    assert negative == positive.replace("(lit Int ", "(lit Int -", 1)


def test_terms_are_frozen_and_hashable():
    term = t.typeis(t.var("x"), "bool")
    assert len({term, t.typeis(t.var("x"), "bool")}) == 1
    with pytest.raises(FrozenInstanceError):
        term.args = ()


@pytest.mark.parametrize("term", [
    lambda: t.literal("1"), lambda: t.literal(None), lambda: t.literal([1]),
    lambda: t.var("bad-name"), lambda: t.var("class"), lambda: t.tag("CustomClass"),
    lambda: t.add(t.literal(True), t.literal(1)),
    lambda: t.add(t.literal(1), t.literal(1.0)),
    lambda: t.eq(t.var("a"), t.var("b")),
    lambda: t.lt(t.literal(False), t.literal(True)),
    lambda: t.as_int(t.literal(True)), lambda: t.and_(), lambda: t.or_(),
    lambda: t.floordiv(t.literal(1.0), t.literal(2.0)),
])
def test_unsupported_construction_is_refused_not_true(term):
    result = term()
    assert isinstance(result, t.Refused)
    assert str(result).startswith("store_only:")
    assert t.canon(result) is result
    assert t.and_(t.literal(True), result) is result


def test_numeric_bool_conversion_is_explicit():
    assert t.add(t.b2i(t.literal(True)), t.literal(1)).sort == t.INT
    assert t.i2f(t.literal(1)).sort == t.F64


@pytest.mark.parametrize("op", [t.eq, t.ne, t.lt, t.le, t.gt, t.ge])
def test_numeric_comparison_constructor_shapes(op):
    assert op(t.literal(-1), t.literal(2)).sort == t.BOOL


@pytest.mark.parametrize("op", [t.add, t.sub, t.mul, t.floordiv, t.mod])
def test_integer_operation_constructor_shapes(op):
    assert op(t.literal(-7), t.literal(3)).sort == t.INT
    assert t.neg(t.literal(1)).sort == t.INT


@pytest.mark.parametrize("term", [t.literal(True), t.Refused("unsupported")])
def test_symbolic_values_have_no_implicit_truth_value(term):
    with pytest.raises(TypeError):
        bool(term)


@pytest.mark.parametrize("op,sort,args", [
    ("lit", t.INT, (True,)), ("lit", t.BOOL, (1,)),
    ("var", t.PY, ["x"]), ("mystery", t.BOOL, ()),
    ("tag", t.TAG, (["int"],)), ([], t.BOOL, ()),
    ("add", t.INT, (t.literal(1), t.literal(1.0))),
])
def test_direct_terms_cannot_smuggle_invalid_shapes(op, sort, args):
    with pytest.raises(ValueError):
        t.Term(op, sort, args)


def test_canonical_decoder_roundtrips_every_operation_and_record_binding():
    x, y = t.var("x"), t.var("y")
    i, j = t.as_int(x), t.literal(-3)
    terms = [x, t.literal(True), t.literal(1 << 20000), t.literal(-(1 << 20000)),
             t.literal(float("nan")), t.literal(-0.0), t.tag("NoneType"),
             t.typeis(x, "int"), t.as_bool(y), t.as_f64(y), t.b2i(t.literal(True)),
             t.i2f(j), t.and_(t.literal(True), t.literal(False)),
             t.or_(t.literal(False), t.literal(True)), t.not_(t.literal(False)), t.neg(j)]
    terms += [op(i, j) for op in (t.eq, t.ne, t.lt, t.le, t.gt, t.ge,
                                  t.add, t.sub, t.mul, t.floordiv, t.mod)]
    for term in terms:
        text = t.canon(term)
        assert t.from_canon(text) == term
        assert t.canon(t.from_canon(text)) == text
    binding = {"receiver": t.canon(x), "args": [t.canon(j)], "op": "Add"}
    assert t.from_canon(binding["receiver"]) == x
    assert t.from_canon(binding["args"][0]) == j


@pytest.mark.parametrize("text", [
    "", "true", "(term 2 (lit Bool true))", "(term true (lit Bool true))",
    "(term 1 (lit Int true))", "(term 1 (lit Bool 1))", "(term 1 (lit Int 01))",
    "(term 1 (lit Int -0))", "(term 1 (lit Int +1))", "(term 1 (lit F64 1))",
    '(term 1 (lit F64 "0x1p+0"))', '(term 1 (lit F64 "0x1p+999999"))',
    '(term 1 (var Py "x))', '(term 1 (var Py "\\q"))',
    "(term 1 (mystery Bool))", "(term 1 (lit Real 1))",
    "(term 1 (add Int (lit Int 1)))", "(term 1 (lit Int 1)) trailing",
    " (term 1 (lit Int 1))", "(term 1  (lit Int 1))", "(term 1 (lit Int 1))\n",
])
def test_canonical_decoder_refuses_malformed_noncanonical_or_unsupported(text):
    result = t.from_canon(text)
    assert isinstance(result, t.Refused)
    with pytest.raises(TypeError):
        bool(result)


def test_canonical_decoder_limits_depth_nodes_and_text():
    nested = t.literal(True)
    for _ in range(129):
        nested = t.not_(nested)
    assert isinstance(t.from_canon(t.canon(nested)), t.Refused)
    assert isinstance(t.from_canon(t.canon(t.and_(*([t.literal(True)] * 2049)))), t.Refused)
    assert isinstance(t.from_canon(" " * 65537), t.Refused)


def test_rename_preserves_operand_order_and_only_changes_names():
    term = t.and_(t.typeis(t.var("x"), "int"), t.gt(t.as_int(t.var("x")), t.literal(2)))
    renamed = t.rename(term, {"x": "next_x"})
    assert t.canon(renamed) == t.canon(term).replace('"x"', '"next_x"')
    assert t.eval_term(renamed, {"next_x": 3}) is True
    assert t.rename(term, {}) == term
    for mapping in ({"x": "class"}, {"x": "bad-name"}, {"x": "K"}):
        assert isinstance(t.rename(term, mapping), t.Refused)
    assert isinstance(t.rename(t.eq(t.var("x", t.INT), t.var("y", t.INT)), {"x": "y"}), t.Refused)


def test_translators_reject_nfkc_aliases_and_conflicting_same_name_sorts():
    unusual = t.var("K", t.INT)
    term = t.eq(unusual, t.literal(1))
    assert isinstance(t.to_python(term), t.Refused)
    assert isinstance(t.eval_term(term, {"K": 1, "K": 2}), t.Refused)
    assert isinstance(t.from_canon(t.canon(term)), t.Refused)
    mixed = t.and_(t.typeis(t.var("x"), "int"), t.gt(t.var("x", t.INT), t.literal(0)))
    assert isinstance(t.to_python(mixed, builtin_names={"type", "int"}), t.Refused)
    assert isinstance(t.eval_term(mixed, {"x": 1}), t.Refused)
    assert isinstance(t.eval_term(t.var("kind", t.TAG), {"kind": "int"}), t.Refused)


@pytest.mark.parametrize("source,types", [
    ("unknown > 0", {}), ("False and unknown > 0", {}),
    ("f()", {"f": "object"}), ("x.attr", {"x": "object"}),
    ("x[0]", {"x": "list"}), ("[1]", {}), ("'x' == 'x'", {}),
    ("x + 1.0", {"x": "int"}), ("x and True", {"x": "int"}),
    ("x == x", {"x": "object"}), ("x is y", {"x": "int", "y": "int"}),
    ("type(x) is int", {"x": "object", "type": "object"}),
    ("type(x) is int", {"x": "object", "int": "object"}),
    ("isinstance(x, int)", {"x": "object"}), ("x", {"x": "typing.Any"}),
])
def test_from_ast_requires_bound_pure_supported_values(source, types):
    assert isinstance(t.from_ast(source, types, builtin_names={"type", "int", "isinstance"}), t.Refused)


def test_from_ast_preserves_one_py_binding_and_requires_builtin_authentication():
    source = "type(x) is int and x > 0"
    assert isinstance(t.from_ast(source, {"x": "object"}), t.Refused)
    term = t.from_ast(source, {"x": "object"}, builtin_names={"type", "int"})
    assert term == t.and_(t.typeis(t.var("x"), "int"), t.gt(t.as_int(t.var("x")), t.literal(0)))
    assert t.eval_term(term, {"x": 5}) is True
    assert t.eval_term(term, {"x": 5.0}) is False
    typed = t.from_ast("x > 0", {"x": "int"})
    assert typed == t.gt(t.as_int(t.var("x")), t.literal(0))
    assert isinstance(t.to_python(typed, types={"x": "int"}), t.Refused)
    assert isinstance(t.eval_term(typed, {"x": True}), t.Refused)


def test_none_identity_source_is_exact_tag_fact_but_remains_store_only():
    term = t.from_ast("x is not None", {"x": "object"})
    assert term == t.not_(t.typeis(t.var("x"), "NoneType"))
    assert t.eval_term(term, {"x": None}) is False
    assert t.eval_term(term, {"x": 0}) is True
    assert isinstance(t.to_python(term, builtin_names={"type", "NoneType"}), t.Refused)


@pytest.mark.parametrize("node_type", ["Name", "Compare"])
def test_from_ast_refuses_malformed_ast_fields(node_type):
    import ast
    assert isinstance(t.from_ast(getattr(ast, node_type)(), {}), t.Refused)
