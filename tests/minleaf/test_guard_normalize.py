"""Phase 2 gate: 8 tests for guard_normalize.py."""

import ast

from tharkun.solve.normalize import (
    is_atom,
    normalize_guard,
)


def _parse_expr(code: str) -> ast.expr:
    return ast.parse(code, mode="eval").body


def test_chained_compare():
    expr = _parse_expr("0 <= x < 10")
    result = normalize_guard(expr)
    assert isinstance(result, ast.BoolOp)
    assert isinstance(result.op, ast.And)
    assert len(result.values) == 2
    assert isinstance(result.values[0], ast.Compare)
    assert isinstance(result.values[0].ops[0], ast.LtE)
    assert isinstance(result.values[1], ast.Compare)
    assert isinstance(result.values[1].ops[0], ast.Lt)


def test_not_in_to_notin():
    expr = _parse_expr("not (x in S)")
    result = normalize_guard(expr)
    assert isinstance(result, ast.Compare)
    assert isinstance(result.ops[0], ast.NotIn)


def test_x_not_in_preserved():
    expr = _parse_expr("x not in S")
    result = normalize_guard(expr)
    assert isinstance(result, ast.Compare)
    assert isinstance(result.ops[0], ast.NotIn)


def test_de_morgan_not_and():
    expr = _parse_expr("not (a and b)")
    result = normalize_guard(expr)
    assert isinstance(result, ast.BoolOp)
    assert isinstance(result.op, ast.Or)
    assert len(result.values) == 2
    for v in result.values:
        assert isinstance(v, ast.UnaryOp)
        assert isinstance(v.op, ast.Not)


def test_de_morgan_not_or():
    expr = _parse_expr("not (a or b)")
    result = normalize_guard(expr)
    assert isinstance(result, ast.BoolOp)
    assert isinstance(result.op, ast.And)
    assert len(result.values) == 2


def test_double_not():
    expr = _parse_expr("not (not x)")
    result = normalize_guard(expr)
    assert isinstance(result, ast.Name)
    assert result.id == "x"


def test_flatten_nested_and():
    expr = _parse_expr("a and (b and c)")
    result = normalize_guard(expr)
    assert isinstance(result, ast.BoolOp)
    assert isinstance(result.op, ast.And)
    assert len(result.values) == 3


def test_is_atom():
    assert is_atom(ast.Compare(
        left=ast.Name(id="x"), ops=[ast.Gt()],
        comparators=[ast.Constant(value=0)]
    )) is True
    assert is_atom(ast.BoolOp(op=ast.And(), values=[])) is False
    assert is_atom(ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="x"))) is False
    assert is_atom(ast.Call(
        func=ast.Name(id="foo"), args=[], keywords=[]
    )) is True
    assert is_atom(ast.Name(id="x")) is True
    assert is_atom(ast.Attribute(
        value=ast.Name(id="obj"), attr="x", ctx=ast.Load()
    )) is True
