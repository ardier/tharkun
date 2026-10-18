"""Phase 2: Guard normalization to canonical boolean form.

Transformations (applied in order):
1. Expand chained comparisons.
2. Normalize negated membership: not (x in S) -> Compare(x, NotIn, S).
3. Push Not inward (one level): De Morgan, comparison inversion, double negation.
4. Flatten associative And/Or.
5. Preserve everything else as-is (atoms).
"""

from __future__ import annotations

import ast

# Ordering comparisons: `not (a < b)` is NOT `a >= b` when either operand is NaN, since
# `a < b` is false for NaN so `not` is true while `a >= b` is false. Their negation needs
# the non-NaN disjunct when nan_safe is requested.
_ORDERING_OPS = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)

# Comparison operator inversion table
_CMP_INVERT: dict[type, type] = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.Lt: ast.GtE,
    ast.LtE: ast.Gt,
    ast.Gt: ast.LtE,
    ast.GtE: ast.Lt,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}


def normalize_guard(expr: ast.expr, *, nan_safe: bool = False) -> ast.expr:
    """Normalize a guard AST to canonical boolean form.

    ``nan_safe`` makes the negation of an ordering comparison faithful under NaN:
    ``not (x > 5)`` becomes ``(x <= 5) or (x != x)`` rather than the unsound ``x <= 5``.
    Off by default so minleaf and the static solver keep their integer-focused behaviour;
    the canonicalizer opts in.
    """
    result = expr
    result = _expand_chained_comparisons(result)
    result = _normalize_negation(result)
    result = _push_not_inward(result, nan_safe=nan_safe)
    result = _flatten_bool_ops(result)
    return result


def is_atom(expr: ast.expr) -> bool:
    """Return True if expr is an atom (not BoolOp, not UnaryOp(Not))."""
    if isinstance(expr, ast.BoolOp):
        return False
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return False
    return True


def _expand_chained_comparisons(expr: ast.expr) -> ast.expr:
    """Expand chained comparisons: 0 <= x < 10 -> And(0 <= x, x < 10)."""
    if isinstance(expr, ast.Compare) and len(expr.ops) > 1:
        # Chained comparison: a op1 b op2 c ... -> And(a op1 b, b op2 c, ...)
        parts: list[ast.expr] = []
        left = expr.left
        for i, (op, comparator) in enumerate(zip(expr.ops, expr.comparators)):
            part = ast.Compare(
                left=left,
                ops=[op],
                comparators=[comparator],
            )
            ast.copy_location(part, expr)
            parts.append(part)
            left = comparator
        if len(parts) == 1:
            return parts[0]
        result = ast.BoolOp(op=ast.And(), values=parts)
        ast.copy_location(result, expr)
        return result
    elif isinstance(expr, ast.BoolOp):
        new_values = [_expand_chained_comparisons(v) for v in expr.values]
        return ast.BoolOp(op=expr.op, values=new_values)
    elif isinstance(expr, ast.UnaryOp):
        new_operand = _expand_chained_comparisons(expr.operand)
        return ast.UnaryOp(op=expr.op, operand=new_operand)
    return expr


def _normalize_negation(expr: ast.expr) -> ast.expr:
    """Normalize negated membership: not (x in S) -> Compare(x, NotIn, S)."""
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        operand = _normalize_negation(expr.operand)
        if (isinstance(operand, ast.Compare) and len(operand.ops) == 1
                and isinstance(operand.ops[0], ast.In)):
            result = ast.Compare(
                left=operand.left,
                ops=[ast.NotIn()],
                comparators=operand.comparators,
            )
            ast.copy_location(result, expr)
            return result
        return ast.UnaryOp(op=expr.op, operand=operand)
    elif isinstance(expr, ast.BoolOp):
        new_values = [_normalize_negation(v) for v in expr.values]
        return ast.BoolOp(op=expr.op, values=new_values)
    return expr


def _push_not_inward(expr: ast.expr, *, nan_safe: bool = False) -> ast.expr:
    """Push Not inward: De Morgan, comparison inversion, double negation.

    Recurses into the De Morgan results, so a nested negation like
    ``not (a and (x < y))`` is fully pushed rather than left one level deep.
    """
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        operand = expr.operand
        # Double negation: not (not x) -> x
        if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.Not):
            return _push_not_inward(operand.operand, nan_safe=nan_safe)
        # De Morgan: not (a and b) -> or(not a, not b); recurse into each pushed child.
        if isinstance(operand, ast.BoolOp) and isinstance(operand.op, ast.And):
            new_values = [_push_not_inward(ast.copy_location(
                ast.UnaryOp(op=ast.Not(), operand=v), expr), nan_safe=nan_safe)
                for v in operand.values]
            return ast.copy_location(ast.BoolOp(op=ast.Or(), values=new_values), expr)
        # De Morgan: not (a or b) -> and(not a, not b); recurse likewise.
        if isinstance(operand, ast.BoolOp) and isinstance(operand.op, ast.Or):
            new_values = [_push_not_inward(ast.copy_location(
                ast.UnaryOp(op=ast.Not(), operand=v), expr), nan_safe=nan_safe)
                for v in operand.values]
            return ast.copy_location(ast.BoolOp(op=ast.And(), values=new_values), expr)
        # Comparison inversion: not (x < y) -> x >= y
        if isinstance(operand, ast.Compare) and len(operand.ops) == 1:
            op = operand.ops[0]
            inv_type = _CMP_INVERT.get(type(op))
            if inv_type is not None:
                inverted = ast.copy_location(ast.Compare(
                    left=operand.left, ops=[inv_type()],
                    comparators=operand.comparators), expr)
                if nan_safe and isinstance(op, _ORDERING_OPS):
                    return _with_nan_disjunct(inverted, operand.left,
                                              operand.comparators[0], expr)
                return inverted
        return expr
    elif isinstance(expr, ast.BoolOp):
        new_values = [_push_not_inward(v, nan_safe=nan_safe) for v in expr.values]
        return ast.BoolOp(op=expr.op, values=new_values)
    return expr


def _with_nan_disjunct(inverted: ast.expr, a: ast.expr, b: ast.expr,
                       loc: ast.expr) -> ast.expr:
    """``inverted or (a != a) or (b != b)`` — the NaN cases the inversion drops.

    ``not (a < b)`` is true when ``a >= b`` OR either operand is NaN, so the faithful form
    is a disjunction. A literal is never NaN, so its ``x != x`` disjunct is omitted (it is
    always false). This over-approximates the guard, which is the safe direction: the old
    under-approximation could make a sound decider call a reachable branch dead.
    """
    disjuncts = [inverted]
    for operand in (a, b):
        if not isinstance(operand, ast.Constant):
            disjuncts.append(ast.copy_location(ast.Compare(
                left=operand, ops=[ast.NotEq()], comparators=[operand]), loc))
    if len(disjuncts) == 1:
        return inverted
    return ast.copy_location(ast.BoolOp(op=ast.Or(), values=disjuncts), loc)


def _flatten_bool_ops(expr: ast.expr) -> ast.expr:
    """Flatten associative And/Or: And(a, And(b, c)) -> And(a, b, c)."""
    if isinstance(expr, ast.BoolOp):
        # First recursively flatten children
        flat_values: list[ast.expr] = []
        for v in expr.values:
            v_flat = _flatten_bool_ops(v)
            # If child is same BoolOp type, merge
            if isinstance(v_flat, ast.BoolOp) and type(v_flat.op) == type(expr.op):
                flat_values.extend(v_flat.values)
            else:
                flat_values.append(v_flat)
        return ast.BoolOp(op=expr.op, values=flat_values)
    elif isinstance(expr, ast.UnaryOp):
        return ast.UnaryOp(op=expr.op, operand=_flatten_bool_ops(expr.operand))
    return expr
