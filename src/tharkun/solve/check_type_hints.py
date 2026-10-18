"""Closed, non-executing syntax checks for verifier ``check_type`` hints.

This helper only recognizes names whose meaning the existing Resolver proves.
It never evaluates an annotation or calls a user object.  The returned effect
tags describe operations the verifier may perform while checking a value.
"""
from __future__ import annotations

import ast


_SCALAR_HINTS = frozenset({"int", "bool", "float", "str", "bytes", "complex"})
_CONTAINER_HINTS = frozenset({"list", "tuple", "dict", "set", "frozenset"})
def _other_binding(tree: ast.AST, local: str) -> bool:
    """Whether a spelling has any binding besides its one resolved import."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == local and isinstance(node.ctx, (ast.Store, ast.Del)):
            return True
        if isinstance(node, ast.arg) and node.arg == local:
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == local:
            return True
        if isinstance(node, ast.ExceptHandler) and node.name == local:
            return True
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == local:
            return True
        if isinstance(node, (ast.Global, ast.Nonlocal)) and local in node.names:
            return True
    return False


def _typing_name(node: ast.AST, resolver, attr: str) -> bool:
    if isinstance(node, ast.Name):
        imported = resolver.imports.names.get(node.id)
        if imported != ("typing", attr) or _other_binding(resolver.tree, node.id):
            return False
        return sum(1 for n in ast.walk(resolver.tree) if isinstance(n, ast.alias)
                   and (n.asname or n.name) == node.id and n.name == attr) == 1
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        alias = node.value.id
        if node.attr != attr or resolver.imports.modules.get(alias) != "typing":
            return False
        if _other_binding(resolver.tree, alias):
            return False
        return sum(1 for n in ast.walk(resolver.tree) if isinstance(n, ast.alias)
                   and n.name == "typing" and (n.asname or n.name) == alias) == 1
    return False


def closed_hint_effects(node: ast.AST, resolver) -> frozenset[str] | None:
    """Return verifier effects for a closed hint, or ``None`` when unproved.

    Effects are ``iterate`` for sequence hints and ``items`` for mapping hints.
    Only built-in scalar/container classes, ``None``, and the closed typing
    forms Any/Union/Optional are admitted.  Tuple- and mapping-generic arities
    match the verifier's supported runtime shapes.
    """
    if isinstance(node, ast.Constant) and node.value is None:
        return frozenset()
    if isinstance(node, ast.Name):
        if resolver.is_builtin(node) and node.id in _SCALAR_HINTS | _CONTAINER_HINTS:
            return frozenset()
        if node.id == "None" and resolver.is_builtin(node):
            return frozenset()
        if _typing_name(node, resolver, "Any"):
            return frozenset()
        return None
    if isinstance(node, ast.Attribute) and _typing_name(node, resolver, "Any"):
        return frozenset()
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = closed_hint_effects(node.left, resolver)
        right = closed_hint_effects(node.right, resolver)
        return None if left is None or right is None else left | right
    if isinstance(node, ast.Subscript):
        base = node.value
        if isinstance(base, ast.Name) and resolver.is_builtin(base) and base.id in _CONTAINER_HINTS:
            family = base.id
        elif _typing_name(base, resolver, "Union"):
            family = "Union"
        elif _typing_name(base, resolver, "Optional"):
            family = "Optional"
        else:
            return None
        args = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if family == "dict" and len(args) != 2:
            return None
        if family in {"list", "set", "frozenset"} and len(args) != 1:
            return None
        if family == "tuple":
            if not args or (len(args) == 2 and isinstance(args[1], ast.Constant)
                            and args[1].value is Ellipsis):
                args = args[:1]
            elif any(isinstance(a, ast.Constant) and a.value is Ellipsis for a in args):
                return None
        if family == "Optional" and len(args) != 1:
            return None
        if family == "Union" and len(args) < 2:
            return None
        effects: frozenset[str] = frozenset()
        for arg in args:
            child = closed_hint_effects(arg, resolver)
            if child is None:
                return None
            effects |= child
        if family in {"list", "tuple", "set", "frozenset"}:
            effects |= {"iterate"}
        elif family == "dict":
            effects |= {"items"}
        return effects
    return None
