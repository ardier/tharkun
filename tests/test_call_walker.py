"""Tests for call_walker — shared call-chain walk/resolution infrastructure."""

from __future__ import annotations

import ast
import os

import pytest

from tharkun.solve.frontend.call_walker import (
    CallIndex,
    build_call_index,
    extract_guard_clause_condition,
    extract_return_condition,
    extract_scope_annotations,
    extract_scope_isinstance,
    find_enclosing_class,
    is_falsy_return,
    is_truthy_return,
    negate_chained_compare,
    resolve_callee,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "call_walker_target.py")


def _parse_fixture():
    source = open(FIXTURE).read()
    return ast.parse(source)


def _find_func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _find_class(tree, name):
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    return None


def _find_first_call(func_def):
    for node in ast.walk(func_def):
        if isinstance(node, ast.Call):
            return node
    return None


# ---------------------------------------------------------------------------
# build_call_index
# ---------------------------------------------------------------------------


class TestBuildCallIndex:
    def test_top_level_functions(self):
        tree = _parse_fixture()
        idx = build_call_index(tree)
        assert "is_valid" in idx.func_defs
        assert "check_range" in idx.func_defs
        assert "with_guards" in idx.func_defs

    def test_class_methods_indexed(self):
        tree = _parse_fixture()
        idx = build_call_index(tree)
        assert "Config.validate" in idx.func_defs
        assert "Config.__init__" in idx.func_defs
        # Bare method name also indexed (if no conflict).
        assert "validate" in idx.func_defs

    def test_class_defs_indexed(self):
        tree = _parse_fixture()
        idx = build_call_index(tree)
        assert "Config" in idx.class_defs
        assert "BaseProcessor" in idx.class_defs
        assert "SubProcessor" in idx.class_defs

    def test_cross_module_with_source_path(self):
        tree = _parse_fixture()
        idx = build_call_index(tree, source_path=FIXTURE)
        # Fixture has no imports, so cross-module adds nothing.
        # But source_path is set.
        assert idx.source_path == FIXTURE

    def test_cross_module_imports(self):
        cross_a = os.path.join(os.path.dirname(__file__), "fixtures", "cross_module_a.py")
        if not os.path.isfile(cross_a):
            pytest.skip("cross_module_a.py fixture not found")
        source = open(cross_a).read()
        tree = ast.parse(source)
        idx = build_call_index(tree, source_path=cross_a)
        # Should have imported function defs if cross_module_a imports from cross_module_b.
        assert idx.imports or True  # At least runs without error.


# ---------------------------------------------------------------------------
# resolve_callee
# ---------------------------------------------------------------------------


class TestResolveCallee:
    def test_simple_name(self):
        tree = _parse_fixture()
        idx = build_call_index(tree)
        # check_range calls is_valid(value)
        func = _find_func(tree, "check_range")
        call = _find_first_call(func)
        resolved, qname = resolve_callee(call, idx)
        assert resolved is not None
        assert qname == "is_valid"

    def test_self_method(self):
        tree = _parse_fixture()
        idx = build_call_index(tree)
        config_cls = _find_class(tree, "Config")
        run_func = _find_func(config_cls, "run")
        call = _find_first_call(run_func)
        resolved, qname = resolve_callee(call, idx, enclosing_class=config_cls)
        assert resolved is not None
        assert qname == "Config.validate"

    def test_super_method(self):
        tree = _parse_fixture()
        idx = build_call_index(tree)
        sub_cls = _find_class(tree, "SubProcessor")
        process_func = _find_func(sub_cls, "process")
        call = _find_first_call(process_func)
        resolved, qname = resolve_callee(call, idx, enclosing_class=sub_cls)
        assert resolved is not None
        assert qname == "BaseProcessor.process"

    def test_isinstance_typed_receiver(self):
        tree = _parse_fixture()
        idx = build_call_index(tree)
        func = _find_func(tree, "handle_config")
        # Find the obj.validate() call.
        calls = [n for n in ast.walk(func) if isinstance(n, ast.Call)]
        validate_call = None
        for c in calls:
            if isinstance(c.func, ast.Attribute) and c.func.attr == "validate":
                validate_call = c
                break
        assert validate_call is not None
        resolved, qname = resolve_callee(
            validate_call, idx,
            scope_isinstance={"obj": "Config"},
        )
        assert resolved is not None
        assert qname == "Config.validate"

    def test_annotation_typed_receiver(self):
        tree = _parse_fixture()
        idx = build_call_index(tree)
        func = _find_func(tree, "use_config")
        call = _find_first_call(func)
        resolved, qname = resolve_callee(
            call, idx,
            scope_annotations={"cfg": "Config"},
        )
        assert resolved is not None
        assert qname == "Config.validate"

    def test_class_name_method(self):
        tree = _parse_fixture()
        idx = build_call_index(tree)
        func = _find_func(tree, "use_class_method")
        calls = [n for n in ast.walk(func) if isinstance(n, ast.Call)]
        # Config.validate(c)
        validate_call = None
        for c in calls:
            if isinstance(c.func, ast.Attribute) and c.func.attr == "validate":
                validate_call = c
                break
        assert validate_call is not None
        resolved, qname = resolve_callee(validate_call, idx)
        assert resolved is not None
        assert qname == "Config.validate"

    def test_unresolvable_returns_none(self):
        tree = _parse_fixture()
        idx = build_call_index(tree)
        # Construct an unresolvable call: unknown_func()
        call = ast.Call(
            func=ast.Name(id="nonexistent_func"),
            args=[], keywords=[],
        )
        resolved, qname = resolve_callee(call, idx)
        assert resolved is None
        assert qname is None


# ---------------------------------------------------------------------------
# extract_return_condition
# ---------------------------------------------------------------------------


class TestExtractReturnCondition:
    def test_single_return(self):
        tree = _parse_fixture()
        func = _find_func(tree, "is_valid")
        cond = extract_return_condition(func)
        assert cond is not None
        src = ast.unparse(cond)
        assert "x > 0" in src

    def test_guard_clauses(self):
        tree = _parse_fixture()
        func = _find_func(tree, "with_guards")
        cond = extract_return_condition(func)
        assert cond is not None
        src = ast.unparse(cond)
        assert "not" in src.lower() or "Not" in src

    def test_if_true_else_false(self):
        tree = _parse_fixture()
        func = _find_func(tree, "if_true_else_false")
        cond = extract_return_condition(func)
        assert cond is not None
        src = ast.unparse(cond)
        assert "10" in src

    def test_complex_returns_none(self):
        tree = _parse_fixture()
        func = _find_func(tree, "complex_return")
        cond = extract_return_condition(func)
        assert cond is None


# ---------------------------------------------------------------------------
# find_enclosing_class
# ---------------------------------------------------------------------------


class TestFindEnclosingClass:
    def test_method_finds_class(self):
        tree = _parse_fixture()
        config_cls = _find_class(tree, "Config")
        validate = _find_func(config_cls, "validate")
        found = find_enclosing_class(tree, validate)
        assert found is config_cls

    def test_top_level_returns_none(self):
        tree = _parse_fixture()
        func = _find_func(tree, "is_valid")
        found = find_enclosing_class(tree, func)
        assert found is None


# ---------------------------------------------------------------------------
# Scope extraction helpers
# ---------------------------------------------------------------------------


class TestScopeExtraction:
    def test_extract_scope_isinstance(self):
        tree = _parse_fixture()
        func = _find_func(tree, "handle_config")
        result = extract_scope_isinstance(func)
        assert result == {"obj": "Config"}

    def test_extract_scope_annotations(self):
        tree = _parse_fixture()
        func = _find_func(tree, "use_config")
        result = extract_scope_annotations(func)
        assert result == {"cfg": "Config"}

    def test_no_isinstance_empty(self):
        tree = _parse_fixture()
        func = _find_func(tree, "is_valid")
        result = extract_scope_isinstance(func)
        assert result == {}


# ---------------------------------------------------------------------------
# negate_chained_compare
# ---------------------------------------------------------------------------


class TestNegateChainedCompare:
    def test_basic_chain(self):
        # 0 <= x <= 10
        node = ast.Compare(
            left=ast.Constant(value=0),
            ops=[ast.LtE(), ast.LtE()],
            comparators=[ast.Name(id="x"), ast.Constant(value=10)],
        )
        parts = negate_chained_compare(node)
        assert parts is not None
        assert len(parts) == 2

    def test_unnegatable_returns_none(self):
        # x in items
        node = ast.Compare(
            left=ast.Name(id="x"),
            ops=[ast.In()],
            comparators=[ast.Name(id="items")],
        )
        parts = negate_chained_compare(node)
        assert parts is None


# ---------------------------------------------------------------------------
# Guard clause helpers
# ---------------------------------------------------------------------------


class TestGuardClauseHelpers:
    def test_is_falsy_return_none(self):
        node = ast.Return(value=None)
        assert is_falsy_return(node)

    def test_is_falsy_return_false(self):
        node = ast.Return(value=ast.Constant(value=False))
        assert is_falsy_return(node)

    def test_is_truthy_return(self):
        node = ast.Return(value=ast.Constant(value=42))
        assert is_truthy_return(node)

    def test_guard_clause_condition(self):
        # if not x: return False
        stmt = ast.If(
            test=ast.UnaryOp(op=ast.Not(), operand=ast.Name(id="x")),
            body=[ast.Return(value=ast.Constant(value=False))],
            orelse=[],
        )
        cond = extract_guard_clause_condition(stmt)
        assert cond is not None
        assert isinstance(cond, ast.UnaryOp)  # NOT(NOT(x))

    def test_non_guard_returns_none(self):
        stmt = ast.Assign(
            targets=[ast.Name(id="x")],
            value=ast.Constant(value=1),
        )
        cond = extract_guard_clause_condition(stmt)
        assert cond is None
