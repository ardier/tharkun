"""Phase 3 gate: 9 tests for guard_trace.py."""

import ast

from tharkun.solve.guard_eval import trace_guard


def _parse_expr(code: str) -> ast.expr:
    return ast.parse(code, mode="eval").body


def test_single_true():
    traced = trace_guard(_parse_expr("x > 0"), {"x": 5}, desired_value=True)
    assert traced.trace.guard_value is True
    assert traced.trace.blocking_atom_ids == ()


def test_single_false():
    traced = trace_guard(_parse_expr("x > 0"), {"x": -1}, desired_value=True)
    assert traced.trace.guard_value is False
    assert "A0" in traced.trace.blocking_atom_ids


def test_and_short_circuit():
    traced = trace_guard(
        _parse_expr("a and b"), {"a": False, "b": True}, desired_value=True
    )
    assert "A0" in traced.trace.eval_order
    assert "A1" not in traced.trace.eval_order


def test_or_short_circuit():
    traced = trace_guard(
        _parse_expr("a or b"), {"a": True, "b": False}, desired_value=False
    )
    assert "A0" in traced.trace.eval_order
    assert "A1" not in traced.trace.eval_order


def test_and_false_cause():
    traced = trace_guard(
        _parse_expr("a and b"), {"a": True, "b": False}, desired_value=True
    )
    assert traced.trace.blocking_atom_ids == ("A1",)


def test_and_true_cause():
    traced = trace_guard(
        _parse_expr("a and b"), {"a": True, "b": True}, desired_value=False
    )
    assert set(traced.trace.blocking_atom_ids) == {"A0", "A1"}


def test_or_false_cause():
    traced = trace_guard(
        _parse_expr("a or b"), {"a": False, "b": False}, desired_value=True
    )
    assert set(traced.trace.blocking_atom_ids) == {"A0", "A1"}


def test_nested_cause():
    traced = trace_guard(
        _parse_expr("(a and b) or c"),
        {"a": False, "b": True, "c": False},
        desired_value=True,
    )
    # b skipped by short-circuit in And (a is False)
    assert "A1" not in traced.trace.eval_order
    assert set(traced.trace.blocking_atom_ids) == {"A0", "A2"}


def test_retrace_fresh():
    env = {"x": -1}
    t1 = trace_guard(_parse_expr("x > 0"), env, desired_value=True)
    env["x"] = 5
    t2 = trace_guard(_parse_expr("x > 0"), env, desired_value=True)
    assert t1.trace.guard_value is False
    assert t2.trace.guard_value is True
    # t1 is frozen, should still show False
    assert t1.trace.guard_value is False
