"""Compatibility and typed-template ownership for ``heapq.heappop``."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from tharkun.solve.contracts.adapters.common import raise_condition
from tharkun.solve.contracts.registry import get
from tharkun.solve.terms import Refused
from tharkun.solve.contracts.safety import Binding


@dataclass(frozen=True)
class LegacyCall:
    exc: str
    value: str | None
    reason: str | None


@dataclass(frozen=True)
class _LegacyTemplate:
    arity: int
    arg_index: int
    expr_template: str


# The old expression path remains exact and temporary: the M1 terms do not yet
# have Seq or len.  This is the sole owner of the applicability metadata.
_LEGACY = {"heapq.heappop": _LegacyTemplate(1, 0, "len({arg}) > 0")}
_ALLOWED_EXPR_NODES = (
    ast.Name, ast.Load, ast.Constant, ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not, ast.UAdd, ast.USub, ast.BinOp, ast.Add, ast.Sub,
    ast.Mult, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt,
    ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.Call,
)


def _is_python_expression(value: str) -> bool:
    try:
        tree = ast.parse(value, mode="eval").body
    except (SyntaxError, ValueError, RecursionError):
        return False
    return all(isinstance(node, _ALLOWED_EXPR_NODES) for node in ast.walk(tree))


def legacy_call(qualname: str, call_node: ast.Call) -> LegacyCall | None:
    """Render the pre-M1.4 heappop invariant with identical eligibility rules."""
    template = _LEGACY.get(qualname)
    if template is None:
        return None
    exc = "IndexError"
    if call_node.keywords or len(call_node.args) != template.arity or any(
        isinstance(arg, ast.Starred) for arg in call_node.args
    ):
        return LegacyCall(
            exc, None,
            "call shape does not match the template's checked arity (keywords/*args, or "
            f"{len(call_node.args)} positional args, not exactly {template.arity})",
        )
    contract = get(qualname)
    if contract is None or raise_condition(contract, exc) is None:
        return LegacyCall(exc, None, "contract missing, untested, or no longer claims a real condition for this class")
    try:
        arg_src = ast.unparse(call_node.args[template.arg_index])
    except Exception:  # noqa: BLE001 -- unparse is not total over every AST shape
        return LegacyCall(exc, None, "argument expression could not be unparsed")
    value = template.expr_template.format(arg=arg_src)
    if not _is_python_expression(value):
        return LegacyCall(
            exc, None,
            f"argument expression {arg_src!r} is outside the python_expression grammar "
            "(e.g. an attribute/subscript access)",
        )
    return LegacyCall(exc, value, None)


def heapq_cases(qualname: str, exc: str, binding: Binding):
    return Refused("M1 terms do not model heap length; use the preserved legacy expression")


CASES_BUILDERS = {("heapq.heappop", "IndexError"): heapq_cases}


__all__ = ["CASES_BUILDERS", "LegacyCall", "heapq_cases", "legacy_call"]
