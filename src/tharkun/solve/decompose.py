"""Recursive guard decomposition — traces into function calls to build
a tree of sub-conditions and identify what is solvable vs blocking.

Given a guard like ``if we_are_good(config):``, the decomposer resolves
``we_are_good`` to its definition, extracts the return condition, and
recursively decomposes it into solvable and opaque sub-parts.

Usage::

    from tharkun.solve.decompose import decompose_guard

    result = decompose_guard("module.py", 42)
    print(result.total_leaves, "conditions")
    print(result.solvable_leaves, "solvable")
    for b in result.blockers:
        print(f"  blocker: {b.expression} — {b.blocker_reason}")
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from tharkun.solve.frontend.call_walker import (
    CallIndex,
    build_call_index,
    extract_return_condition,
    extract_scope_annotations,
    extract_scope_isinstance,
    find_enclosing_class,
    negate_chained_compare,
    resolve_callee,
)
from tharkun.solve.condition import (
    VarConstraint,
    _extract_comparison_values,
    _extract_len_constraint,
    _extract_str_method_values,
    _free_names,
    _strategy_hint_for_type,
    _unparse,
    infer_var_type,
)
from tharkun.solve.constants import BUILTIN_NAMES


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class DecompNode:
    """One node in the decomposition tree.

    Attributes:
        expression:        Source text of this sub-condition.
        kind:              ``"and"`` | ``"or"`` | ``"not"`` | ``"call"``
                           | ``"leaf"``.
        solvable:          ``True`` if condition_solver can handle this leaf,
                           ``False`` if it's a known blocker, ``None`` if
                           it's a composite node whose solvability depends
                           on children.
        children:          Sub-conditions (empty for leaves).
        blocker_reason:    Why this node is unsolvable.
        resolved_function: Qualname of resolved function (for ``"call"`` nodes).
        var_constraint:    Leaf-level solver result.
        depth:             Depth in the tree (root = 0).
    """

    expression: str
    kind: str
    solvable: bool | None = None
    children: list["DecompNode"] = field(default_factory=list)
    blocker_reason: str | None = None
    resolved_function: str | None = None
    var_constraint: VarConstraint | None = None
    depth: int = 0


@dataclass
class DecompResult:
    """Complete decomposition of a guard expression."""

    guard_source: str
    root: DecompNode
    total_leaves: int = 0
    solvable_leaves: int = 0
    unsolvable_leaves: int = 0
    max_depth: int = 0
    blockers: list[DecompNode] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Expression decomposition
# ---------------------------------------------------------------------------



def _try_solve_leaf(
    expr: ast.expr,
    source_path: str,
    lineno: int,
) -> tuple[bool, VarConstraint | None]:
    """Attempt to solve a leaf expression using condition_solver.

    Returns ``(solvable, var_constraint)``.
    """
    free = _free_names(expr)
    var_names = sorted(free - BUILTIN_NAMES)
    if not var_names:
        # No variables — likely a constant expression.
        return True, None

    # Use the first variable as the representative.
    name = var_names[0]
    inferred_type, source_kind = infer_var_type(name, expr, source_path, lineno)

    values = _extract_comparison_values(name, expr)
    if not values and inferred_type == "str":
        synth = _extract_str_method_values(name, expr)
        if synth:
            values = [synth]
    if not values:
        len_min = _extract_len_constraint(name, expr)
        if len_min and len_min > 0:
            if inferred_type in ("str", "Any"):
                values = ["x" * len_min]
            elif inferred_type == "list":
                values = [list(range(len_min))]

    solvable = bool(values) or source_kind not in ("fallback", "self_attr")
    vc = VarConstraint(
        name=name,
        inferred_type=inferred_type,
        inference_source=source_kind,
        concrete_values=values,
        strategy_hint=_strategy_hint_for_type(inferred_type, values),
    )
    return solvable, vc


def _decompose_expr(
    expr: ast.expr,
    call_index: CallIndex,
    source_path: str,
    lineno: int,
    depth: int,
    max_depth: int,
    visited: frozenset[str],
    enclosing_class: ast.ClassDef | None = None,
    scope_isinstance: dict[str, str] | None = None,
    scope_annotations: dict[str, str] | None = None,
) -> DecompNode:
    """Recursively decompose an expression into a tree of sub-conditions."""
    src = _unparse(expr) or "<unknown>"

    if depth > max_depth:
        return DecompNode(
            expression=src, kind="leaf", solvable=False, depth=depth,
            blocker_reason="max_depth_exceeded",
        )

    # BoolOp: and / or
    if isinstance(expr, ast.BoolOp):
        kind = "and" if isinstance(expr.op, ast.And) else "or"
        children = [
            _decompose_expr(
                v, call_index, source_path, lineno, depth + 1, max_depth,
                visited, enclosing_class, scope_isinstance, scope_annotations,
            )
            for v in expr.values
        ]
        return DecompNode(expression=src, kind=kind, children=children, depth=depth)

    # UnaryOp(Not) — with special handling for negated chained comparisons.
    # not (A <= x <= B) → x < A or x > B
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        inner = expr.operand
        # Detect negated chained comparison: not (lo <= x <= hi)
        if (isinstance(inner, ast.Compare) and len(inner.ops) >= 2):
            negated_parts = negate_chained_compare(inner)
            if negated_parts:
                children = [
                    _decompose_expr(
                        p, call_index, source_path, lineno, depth + 1,
                        max_depth, visited, enclosing_class,
                        scope_isinstance, scope_annotations,
                    )
                    for p in negated_parts
                ]
                return DecompNode(
                    expression=src, kind="or", children=children, depth=depth,
                )
        child = _decompose_expr(
            inner, call_index, source_path, lineno, depth + 1, max_depth,
            visited, enclosing_class, scope_isinstance, scope_annotations,
        )
        return DecompNode(expression=src, kind="not", children=[child], depth=depth)

    # Function call — try to resolve and recurse.
    if isinstance(expr, ast.Call):
        resolved_def, func_name = resolve_callee(
            expr, call_index,
            enclosing_class=enclosing_class,
            scope_isinstance=scope_isinstance,
            scope_annotations=scope_annotations,
        )

        # Extract a display name even if resolution failed, so we can
        # report unresolved calls properly (not fall through to leaf).
        display_name = func_name
        if display_name is None:
            if isinstance(expr.func, ast.Name):
                display_name = expr.func.id
            elif isinstance(expr.func, ast.Attribute):
                if isinstance(expr.func.value, ast.Name):
                    display_name = f"{expr.func.value.id}.{expr.func.attr}"
                else:
                    display_name = expr.func.attr

        if display_name and display_name not in BUILTIN_NAMES:
            if resolved_def is not None and display_name not in visited:
                ret_cond = extract_return_condition(resolved_def)
                if ret_cond is not None:
                    child = _decompose_expr(
                        ret_cond, call_index, source_path, lineno,
                        depth + 1, max_depth, visited | {display_name},
                        enclosing_class, scope_isinstance, scope_annotations,
                    )
                    return DecompNode(
                        expression=src, kind="call", children=[child],
                        resolved_function=display_name, depth=depth,
                    )
                else:
                    return DecompNode(
                        expression=src, kind="call", solvable=False,
                        resolved_function=display_name, depth=depth,
                        blocker_reason="complex_return_logic",
                    )
            elif display_name in visited:
                return DecompNode(
                    expression=src, kind="call", solvable=False, depth=depth,
                    blocker_reason="recursive_call",
                )
            elif resolved_def is None:
                return DecompNode(
                    expression=src, kind="call", solvable=False, depth=depth,
                    blocker_reason="unresolved_function",
                )

    # Leaf expression — try to solve with condition_solver.
    solvable, vc = _try_solve_leaf(expr, source_path, lineno)
    reason = None if solvable else "unsolvable_leaf"
    return DecompNode(
        expression=src, kind="leaf", solvable=solvable,
        var_constraint=vc, depth=depth, blocker_reason=reason,
    )


# ---------------------------------------------------------------------------
# Summary computation
# ---------------------------------------------------------------------------


def _count_tree(node: DecompNode) -> tuple[int, int, int, int, list[DecompNode]]:
    """Return (total_leaves, solvable, unsolvable, max_depth, blockers)."""
    if not node.children:
        # Leaf node (or resolved opaque call).
        is_leaf = True
        is_solvable = node.solvable is True
        is_blocker = node.solvable is False
        blockers = [node] if is_blocker else []
        return (
            1 if is_leaf else 0,
            1 if is_solvable else 0,
            1 if is_blocker else 0,
            node.depth,
            blockers,
        )
    total = 0
    solvable = 0
    unsolvable = 0
    max_d = node.depth
    blockers: list[DecompNode] = []
    for child in node.children:
        t, s, u, d, b = _count_tree(child)
        total += t
        solvable += s
        unsolvable += u
        max_d = max(max_d, d)
        blockers.extend(b)
    return total, solvable, unsolvable, max_d, blockers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def decompose_guard(
    source_path: str,
    lineno: int,
    *,
    max_depth: int = 5,
) -> DecompResult:
    """Decompose a guard expression at *lineno* into a tree of sub-conditions.

    Recursively traces into called functions (same module and cross-module)
    to identify which sub-conditions are solvable and which are blockers.

    Args:
        source_path: Path to the Python source file.
        lineno:      Line number of the ``if``/``while``/``for`` statement.
        max_depth:   Maximum recursion depth for function tracing.

    Returns:
        A ``DecompResult`` with the decomposition tree and summary statistics.
    """
    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
    except (OSError, IOError):
        empty = DecompNode(expression="", kind="leaf", solvable=False,
                           blocker_reason="unreadable_file", depth=0)
        return DecompResult(guard_source="", root=empty, blockers=[empty],
                            unsolvable_leaves=1)

    try:
        tree = ast.parse(source)
    except SyntaxError:
        empty = DecompNode(expression="", kind="leaf", solvable=False,
                           blocker_reason="syntax_error", depth=0)
        return DecompResult(guard_source="", root=empty, blockers=[empty],
                            unsolvable_leaves=1)

    # Find the branch expression.
    test_expr = None
    for node in ast.walk(tree):
        ln = getattr(node, "lineno", None)
        if ln != lineno:
            continue
        if isinstance(node, (ast.If, ast.While)):
            test_expr = node.test
            break
        if isinstance(node, ast.For):
            test_expr = node.iter
            break

    if test_expr is None:
        empty = DecompNode(expression="", kind="leaf", solvable=False,
                           blocker_reason="no_branch_at_line", depth=0)
        return DecompResult(guard_source="", root=empty, blockers=[empty],
                            unsolvable_leaves=1)

    guard_source = _unparse(test_expr) or ""
    call_index = build_call_index(tree, source_path=rp)

    # Find enclosing class for method resolution (if the guard is inside a method).
    enclosing_class = None
    for node in ast.walk(tree):
        ln = getattr(node, "lineno", None)
        if ln != lineno:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            enclosing_class = find_enclosing_class(tree, node)
            break

    # Find enclosing function for scope analysis.
    enclosing_func = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.walk(node):
                if getattr(child, "lineno", None) == lineno:
                    enclosing_func = node
                    break
            if enclosing_func is not None:
                break

    scope_isinstance = extract_scope_isinstance(enclosing_func) if enclosing_func else None
    scope_annotations = extract_scope_annotations(enclosing_func) if enclosing_func else None

    root = _decompose_expr(
        test_expr, call_index, rp, lineno,
        depth=0, max_depth=max_depth, visited=frozenset(),
        enclosing_class=enclosing_class,
        scope_isinstance=scope_isinstance,
        scope_annotations=scope_annotations,
    )

    total, solvable, unsolvable, max_d, blockers = _count_tree(root)

    return DecompResult(
        guard_source=guard_source,
        root=root,
        total_leaves=total,
        solvable_leaves=solvable,
        unsolvable_leaves=unsolvable,
        max_depth=max_d,
        blockers=blockers,
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize_node(node: DecompNode) -> dict[str, Any]:
    d: dict[str, Any] = {
        "expression": node.expression,
        "kind": node.kind,
        "depth": node.depth,
    }
    if node.solvable is not None:
        d["solvable"] = node.solvable
    if node.blocker_reason:
        d["blocker_reason"] = node.blocker_reason
    if node.resolved_function:
        d["resolved_function"] = node.resolved_function
    if node.var_constraint:
        d["var_constraint"] = {
            "name": node.var_constraint.name,
            "inferred_type": node.var_constraint.inferred_type,
            "inference_source": node.var_constraint.inference_source,
            "concrete_values": [repr(v) for v in node.var_constraint.concrete_values],
            "strategy_hint": node.var_constraint.strategy_hint,
        }
    if node.children:
        d["children"] = [_serialize_node(c) for c in node.children]
    return d


def serialize_decomp(result: DecompResult) -> dict[str, Any]:
    """Serialize a ``DecompResult`` for JSON output."""
    return {
        "guard_source": result.guard_source,
        "total_leaves": result.total_leaves,
        "solvable_leaves": result.solvable_leaves,
        "unsolvable_leaves": result.unsolvable_leaves,
        "max_depth": result.max_depth,
        "blockers": [
            {"expression": b.expression, "reason": b.blocker_reason}
            for b in result.blockers
        ],
        "tree": _serialize_node(result.root),
    }
