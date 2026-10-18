"""Checked, pure ``python_expression`` predicates for correctness witnesses.

This is syntax/type checking, NOT a proof that a predicate holds. ``types``
contains source-checked, definitely bound *exact builtin* types at the point;
container element types (e.g. ``dict[int]``) also require checked heap facts.
Builtin identities must be established by the caller. No target code is run.
"""
from __future__ import annotations

import ast
import math
from collections.abc import Mapping

BUILTINS = frozenset({"type", "isinstance", "len", "int", "float", "bool", "str", "bytes",
                      "list", "tuple", "dict", "set", "frozenset"})
TYPES = BUILTINS - {"type", "isinstance", "len"}
SCALARS = frozenset({"int", "float", "bool", "str", "bytes"})
SIZED = frozenset({"str", "bytes", "list", "tuple", "dict", "set", "frozenset"})


class InvalidPredicate(ValueError):
    """Unsupported, impure, partial, or non-Boolean expression."""


def parse_predicate(value: str) -> ast.expr:
    """Whitelist syntax; semantic purity/totality additionally needs ``check_predicate``."""
    if not isinstance(value, str) or not value or len(value) > 65536:
        raise InvalidPredicate("predicate must be a nonempty string of at most 65536 characters")
    try:
        tree = ast.parse(value, mode="eval").body
    except (SyntaxError, ValueError, RecursionError) as e:
        raise InvalidPredicate(f"invalid expression: {e}") from e
    allowed = (ast.Name, ast.Load, ast.Constant, ast.BoolOp, ast.And, ast.Or,
               ast.UnaryOp, ast.Not, ast.UAdd, ast.USub, ast.BinOp, ast.Add, ast.Sub,
               ast.Mult, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt,
               ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.Call)
    nodes = list(ast.walk(tree))
    if len(nodes) > 2048:
        raise InvalidPredicate("predicate too large")
    for n in nodes:
        if not isinstance(n, allowed):
            raise InvalidPredicate(f"unsupported syntax: {type(n).__name__}")
        if isinstance(n, ast.Constant) and (type(n.value) not in (bool, int, float, str, bytes)
                or (type(n.value) is float and not math.isfinite(n.value))):
            raise InvalidPredicate("unsupported constant")
        if isinstance(n, ast.Call) and (not isinstance(n.func, ast.Name)
                or n.func.id not in {"len", "type", "isinstance"} or n.keywords
                or len(n.args) != (2 if n.func.id == "isinstance" else 1)):
            raise InvalidPredicate("only len(x), type(x), isinstance(x, builtin) are allowed")
    return tree


def exact_test(n: ast.expr) -> tuple[str, str] | None:
    """An exact-type guard; isinstance(int) deliberately does not refine to int."""
    if (isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], ast.Is)
            and isinstance(n.left, ast.Call) and isinstance(n.left.func, ast.Name)
            and n.left.func.id == "type" and len(n.left.args) == 1
            and isinstance(n.left.args[0], ast.Name) and isinstance(n.comparators[0], ast.Name)
            and n.comparators[0].id in TYPES):
        return n.left.args[0].id, n.comparators[0].id
    return None


def check_predicate(value: str, types: Mapping[str, str], *,
                    builtin_names=BUILTINS) -> ast.expr:
    """Check total Boolean evaluation under verified bindings and builtin identities.

    ``object`` permits type tests only. Conjunctions can refine it by exact-type
    guards. Unknown overloads, attributes, indexing, division and mixed float/int
    arithmetic are refused. Membership needs primitive elements/keys: exact dict
    alone does not exclude a user-defined key's __eq__ or __hash__.
    """
    tree = parse_predicate(value)
    for name, kind in types.items():
        base = kind.split("[")[0]
        if base not in TYPES | {"object"} or ("[" in kind and (
                base not in SIZED or kind != f"{base}[{kind[len(base)+1:-1]}]"
                or kind[len(base)+1:-1] not in SCALARS)):
            raise InvalidPredicate(f"unsupported type for {name}: {kind}")

    def builtin(name):
        if name not in builtin_names or name in types:
            raise InvalidPredicate(f"unresolved builtin: {name}")

    def infer(n, env):
        if isinstance(n, ast.Constant):
            return type(n.value).__name__
        if isinstance(n, ast.Name):
            if n.id not in env:
                raise InvalidPredicate(f"name not definitely bound: {n.id}")
            return env[n.id]
        if isinstance(n, ast.Call):
            builtin(n.func.id)
            kind = infer(n.args[0], env)
            if n.func.id == "type":
                return "type"
            if n.func.id == "len" and kind.split("[")[0] in SIZED:
                return "int"
            if (n.func.id == "isinstance" and kind.split("[")[0] in TYPES
                    and isinstance(n.args[1], ast.Name) and n.args[1].id in TYPES):
                builtin(n.args[1].id)
                return "bool"
            raise InvalidPredicate("call is not total for its operands")
        if isinstance(n, ast.BoolOp):
            local = dict(env)
            for part in n.values:
                if infer(part, local) != "bool":
                    raise InvalidPredicate("and/or operands must be Boolean")
                guard = exact_test(part)
                if isinstance(n.op, ast.And) and guard:
                    name, kind = guard
                    # Preserve already established element facts, never manufacture them.
                    if local[name].split("[")[0] != kind:
                        local[name] = kind
            return "bool"
        if isinstance(n, ast.UnaryOp):
            k = infer(n.operand, env)
            if isinstance(n.op, ast.Not) and k == "bool":
                return "bool"
            if isinstance(n.op, (ast.UAdd, ast.USub)) and k in {"int", "float"}:
                return k
            raise InvalidPredicate("unsupported unary operand")
        if isinstance(n, ast.BinOp):
            a, b = infer(n.left, env), infer(n.right, env)
            if a == b and a in {"int", "float"}:
                return a
            raise InvalidPredicate("arithmetic requires same exact numeric types")
        if isinstance(n, ast.Compare):
            operands = [n.left, *n.comparators]
            for left, op, right in zip(operands, n.ops, operands[1:]):
                a = infer(left, env)
                if a == "type" and isinstance(op, (ast.Is, ast.IsNot)) and isinstance(right, ast.Name) and right.id in TYPES:
                    builtin(right.id)
                    continue
                b = infer(right, env)
                if isinstance(op, (ast.In, ast.NotIn)):
                    # Builtin primitive keys/elements have no user callbacks. Cross-kind
                    # primitive equality is safe; the backend may still decline it.
                    if a in SCALARS and "[" in b and b.split("[")[0] in SIZED - {"str", "bytes"}:
                        continue
                    if a == b == "str" or a == b == "bytes":
                        continue
                    raise InvalidPredicate("membership needs checked primitive elements/keys")
                if isinstance(op, (ast.Is, ast.IsNot)):
                    raise InvalidPredicate("identity is supported only for type tests")
                if a in SCALARS and b in SCALARS:
                    if isinstance(op, (ast.Eq, ast.NotEq)) or a == b or {a, b} <= {"int", "float", "bool"}:
                        continue
                raise InvalidPredicate("comparison may invoke user code or raise")
            return "bool"
        raise InvalidPredicate("unsupported expression")

    if infer(tree, dict(types)) != "bool":
        raise InvalidPredicate("predicate is not Boolean")
    return tree
