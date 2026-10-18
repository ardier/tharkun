"""Guard evaluation: short-circuit tracing with cause tracking (minleaf phase 3).

Moved from ``tharkun.minleaf.guard_trace`` (#0606, plan #0600) together with the
``AtomEval`` / ``GuardTrace`` records, which ``tharkun.minleaf.types`` re-exports.

Evaluates a normalized guard via an explicit AST interpreter.
Determines which atoms are blocking the desired outcome.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from tharkun.solve.normalize import (
    is_atom,
    normalize_guard,
)

@dataclass(frozen=True)
class AtomEval:
    """Record of a single atom evaluation during guard tracing."""
    atom_id: str              # "A0", "A1", ... (stable pre-order)
    expr_src: str             # ast.unparse'd expression
    value: bool               # bool(eval(atom))
    ast_kind: str             # "Compare", "Call", "Name", etc.
    note: str | None = None   # exception info if eval failed


@dataclass(frozen=True)
class GuardTrace:
    guard_value: bool
    eval_order: tuple[str, ...]                  # atom_ids in evaluation order
    atoms: dict[str, AtomEval]                   # atom_id -> AtomEval
    blocking_atom_ids: tuple[str, ...]           # minimal cause set


@dataclass(frozen=True)
class TracedGuard:
    normalized_guard: ast.expr
    trace: GuardTrace
    atom_nodes: dict[str, ast.expr]    # atom_id -> AST node


def trace_guard(
    guard_expr: ast.expr,
    env: dict[str, object],
    desired_value: bool,
) -> TracedGuard:
    """Trace a guard expression with short-circuit semantics.

    Returns a TracedGuard with evaluation results, cause tracking, and blocking atoms.
    Completely stateless across calls.
    """
    normalized = normalize_guard(guard_expr)

    # Step 1: Assign atom IDs via pre-order traversal
    atom_nodes: dict[str, ast.expr] = {}
    _counter = [0]

    def _assign_ids(node: ast.expr) -> None:
        if is_atom(node):
            atom_id = f"A{_counter[0]}"
            _counter[0] += 1
            atom_nodes[atom_id] = node
        elif isinstance(node, ast.BoolOp):
            for v in node.values:
                _assign_ids(v)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            _assign_ids(node.operand)

    _assign_ids(normalized)

    # Build reverse map: id(node) -> atom_id
    node_to_id: dict[int, str] = {id(node): aid for aid, node in atom_nodes.items()}

    # Step 2: Evaluate with cause tracking
    eval_order: list[str] = []
    atoms: dict[str, AtomEval] = {}

    def _eval_node(node: ast.expr) -> tuple[bool, set[str]]:
        """Evaluate a node, returning (value, cause_set)."""
        if is_atom(node):
            aid = node_to_id[id(node)]
            eval_order.append(aid)
            value, note = _eval_atom(node, env)
            atoms[aid] = AtomEval(
                atom_id=aid,
                expr_src=ast.unparse(node),
                value=value,
                ast_kind=type(node).__name__,
                note=note,
            )
            return value, {aid}

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                # And: short-circuit on first False
                all_causes: set[str] = set()
                for child in node.values:
                    child_val, child_causes = _eval_node(child)
                    if not child_val:
                        return False, child_causes
                    all_causes |= child_causes
                return True, all_causes
            else:
                # Or: short-circuit on first True
                all_causes: set[str] = set()
                for child in node.values:
                    child_val, child_causes = _eval_node(child)
                    if child_val:
                        return True, child_causes
                    all_causes |= child_causes
                return False, all_causes

        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            child_val, child_causes = _eval_node(node.operand)
            return not child_val, child_causes

        # Should not reach here after normalization, but handle gracefully
        aid = node_to_id.get(id(node))
        if aid is not None:
            eval_order.append(aid)
            value, note = _eval_atom(node, env)
            atoms[aid] = AtomEval(
                atom_id=aid,
                expr_src=ast.unparse(node),
                value=value,
                ast_kind=type(node).__name__,
                note=note,
            )
            return value, {aid}
        return False, set()

    guard_value, top_causes = _eval_node(normalized)

    # Step 3: Determine blocking atoms
    if guard_value != desired_value:
        blocking_atom_ids = tuple(sorted(top_causes))
    else:
        blocking_atom_ids = ()

    trace = GuardTrace(
        guard_value=guard_value,
        eval_order=tuple(eval_order),
        atoms=atoms,
        blocking_atom_ids=blocking_atom_ids,
    )

    return TracedGuard(
        normalized_guard=normalized,
        trace=trace,
        atom_nodes=atom_nodes,
    )


def _eval_atom(node: ast.expr, env: dict[str, object]) -> tuple[bool, str | None]:
    """Evaluate a single atom expression. Returns (bool_value, note_or_none)."""
    try:
        code = compile(ast.Expression(body=node), "<guard>", "eval")
        raw = eval(code, env)
        return bool(raw), None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
