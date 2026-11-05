"""Flow-insensitive EXACT types of a few expression forms (SOLVING, stdlib only; nothing is executed).

The census asks one question of a ``%`` or a slice: could the left operand / the sliced value be a ``str`` (``%``-formatting with a
mapping raises KeyError) or a ``dict`` (a hashable slice key raises KeyError)?  An answer needs an EXACT type, never a subclass: an int
subclass can define ``__mod__``.  This module answers ``"int"`` / ``"list"`` / ... or ``None`` (unknown, the census keeps the site).

Sound by construction, and deliberately small:

* an expression is an exact ``int`` if it is an int / bool / float constant... (floats are ``"float"``), ``_sv_verifier.nondet_int()``,
  ``len(..)``, ``abs(int)``, ``min`` / ``max`` of ints, unary ``+ - ~`` and ``+ - * // % & | ^ << >>`` of two ints, an ``a if c else b`` of two
  ints, or a NAME every binding of which (anywhere in the file) is a plain assignment / augmented assignment of an exact int or a
  ``for`` target over ``range(..)``;
* a name with any other binding (parameter, import, def / class, unpacking, ``with``, ``except``, comprehension / walrus / match target,
  ``global`` / ``nonlocal``, ``del``), or whose name is the attribute of a store anywhere in a scanned file (``mod.n = ..`` rebinds a
  module global from outside), has no type.  The greatest fixed point is taken over the names, so mutually recursive names stay exact
  only if every binding is.
"""

from __future__ import annotations

import ast
from typing import Callable

_INT_OPS = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)
_SEQ_CALLS = {"list": "list", "tuple": "tuple"}


def _bindings(tree: ast.AST):
    """name -> list of binding value nodes, or ``None`` when some binding of the name is opaque."""
    out: dict[str, list | None] = {}

    def opaque(name):
        out[name] = None

    def add(name, value):
        if out.get(name, []) is not None:
            out.setdefault(name, []).append(value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    add(t.id, node.value)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            add(node.target.id, ast.BinOp(left=ast.Name(id=node.target.id, ctx=ast.Load()), op=node.op, right=node.value))
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            add(node.target.id, ("for", node.iter))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            opaque(node.name)
        elif isinstance(node, ast.arg):
            opaque(node.arg)
        elif isinstance(node, ast.alias):
            opaque((node.asname or node.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            opaque(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for n in node.names:
                opaque(n)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            opaque(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            opaque(node.rest)
    assigned = {id(t) for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    assigned |= {id(n.target) for n in ast.walk(tree) if isinstance(n, (ast.AugAssign, ast.For)) and isinstance(n.target, ast.Name)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) and id(node) not in assigned:
            opaque(node.id)                      # unpacking, with / comprehension / walrus targets, del
    return out


class FixedTypes:
    """Exact-type answers for one parsed file.  ``callee(call)`` names a call's target (``"_sv_verifier.nondet_int"``) or ``None``;
    ``is_builtin(name_node)`` says a bare name is the real builtin; ``rebound`` is a set of attribute names stored anywhere."""

    def __init__(self, tree: ast.AST, callee: Callable[[ast.Call], str | None], is_builtin: Callable[[ast.Name], bool],
                 rebound: frozenset[str] = frozenset()):
        self.callee, self.is_builtin = callee, is_builtin
        binds = _bindings(tree)
        self.names: dict[str, str | None] = {}
        candidates = {n: v for n, v in binds.items() if v is not None and n not in rebound and v}
        kinds = {n: "int" for n in candidates}      # optimistic, then refine to the greatest fixed point
        self.names = dict(kinds)
        changed = True
        while changed:
            changed = False
            for name in list(self.names):
                got = {self._value_kind(v) for v in candidates[name]}
                if len(got) != 1 or None in got:
                    del self.names[name]
                    changed = True
                else:
                    self.names[name] = got.pop()
        self._tree = tree

    def _value_kind(self, v) -> str | None:
        if isinstance(v, tuple):                    # ("for", iter): a range() loop variable is an int
            it = v[1]
            if isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" and self.is_builtin(it.func) \
                    and not any(isinstance(a, ast.Starred) for a in it.args) and not it.keywords:
                return "int"
            return None
        return self.kind(v)

    def kind(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Constant):
            return {int: "int", bool: "int", float: "float", str: "str", bytes: "bytes"}.get(type(node.value))
        if isinstance(node, ast.Name):
            return self.names.get(node.id) if isinstance(node.ctx, ast.Load) else None
        if isinstance(node, (ast.List, ast.ListComp)):
            return "list"
        if isinstance(node, ast.Tuple):
            return "tuple"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)):
            return "int" if self.kind(node.operand) == "int" else None
        if isinstance(node, ast.BinOp) and isinstance(node.op, _INT_OPS):
            return "int" if self.kind(node.left) == "int" and self.kind(node.right) == "int" else None
        if isinstance(node, ast.IfExp):
            a, b = self.kind(node.body), self.kind(node.orelse)
            return a if a is not None and a == b else None
        if isinstance(node, ast.Call):
            if self.callee(node) == "_sv_verifier.nondet_int" and not node.args and not node.keywords:
                return "int"
            f = node.func
            if isinstance(f, ast.Name) and self.is_builtin(f) and not node.keywords and \
                    not any(isinstance(a, ast.Starred) for a in node.args):
                if f.id == "len" and len(node.args) == 1:
                    return "int"                     # len() always builds an exact int
                if f.id == "abs" and len(node.args) == 1:
                    return "int" if self.kind(node.args[0]) == "int" else None
                if f.id in ("min", "max") and len(node.args) >= 2:
                    return "int" if all(self.kind(a) == "int" for a in node.args) else None
                if f.id in _SEQ_CALLS and len(node.args) <= 1:
                    return _SEQ_CALLS[f.id]          # the constructors return the exact type
        return None
