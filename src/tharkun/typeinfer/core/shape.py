"""Recover an input's *structure* from how the code uses it, then build one (docs/TYPEINFER.md, ``Report.shape``).

Measured against the alternatives (2026-08): pytype and HiTyper both infer return
types but leave every unannotated **parameter** bare -- pytype emits `def f(cfg) -> Any`,
HiTyper emits `"type": []`. RightTyper is not on PyPI and Typify has no public
implementation. None recovers a nested ``TypedDict``-like shape, which is the thing
tharkun actually needs.

So this module does the usage-driven half itself, and it is tractable here for a reason
the general tools do not enjoy: **we only need the slice the guard touches.** Recovering
the full type of ``cfg`` is hard; recovering "``cfg`` is a mapping with key ``mode``
holding a str, and key ``tags`` holding a sequence of str with at least 3 elements" is
just reading the accesses. And unlike a static inferencer we can *check the answer* --
synthesize the shape, run it, and see whether the guard is reached.

Where it pays: the driver plateaus when a parameter has no live value to mutate (minleaf
needs concrete state, and nothing has been observed). A synthesized shape gives the loop
something to start from.

    shapes = infer_shapes(source_path, lineno, ["cfg"])
    seed = synthesize(shapes["cfg"])     # {"mode": "x", "tags": ["x", "x", "x"]}

``infer_params`` attaches the same recovery to every parameter it reports (``Report.shape``, through
:func:`shapes_of`).  Moved here from ``solve/frontend/structure_infer.py`` ([internal ref] A2).
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Method name -> the receiver type it implies. Kept small and unambiguous; a method
# shared across types (e.g. ``index``) is deliberately absent.
_METHOD_OWNER = {
    "upper": "str", "lower": "str", "title": "str", "strip": "str", "lstrip": "str",
    "rstrip": "str", "format": "str", "casefold": "str", "swapcase": "str",
    "isdigit": "str", "isalpha": "str", "splitlines": "str", "encode": "str",
    "decode": "bytes", "hex": "bytes",
    "keys": "dict", "values": "dict", "items": "dict", "get": "dict",
    "append": "list", "extend": "list", "sort": "list",
    "add": "set", "union": "set", "intersection": "set",
}
# Methods that work on both str and bytes; the literal argument disambiguates.
_STR_OR_BYTES = {"startswith", "endswith", "split", "join", "replace", "find", "count"}


@dataclass
class Shape:
    """A recovered structural description of one value."""

    kind: str = "unknown"          # scalar | mapping | sequence | object | unknown
    type_name: str | None = None   # for scalar: int/str/bytes/float/bool
    fields: dict[Any, "Shape"] = field(default_factory=dict)   # mapping keys / object attrs
    element: "Shape | None" = None                              # sequence element
    min_len: int = 0
    tuple_arity: int = 0           # >0 when iteration unpacks a fixed-width tuple

    def describe(self) -> str:
        if self.kind == "scalar":
            return self.type_name or "unknown"
        if self.kind == "mapping":
            inner = ", ".join(f"{k!r}: {v.describe()}" for k, v in sorted(
                self.fields.items(), key=lambda kv: repr(kv[0])))
            return "{" + inner + "}"
        if self.kind == "sequence":
            el = self.element.describe() if self.element else "unknown"
            n = f" (len>={self.min_len})" if self.min_len else ""
            return f"[{el}]{n}"
        if self.kind == "object":
            inner = ", ".join(f".{k}: {v.describe()}" for k, v in sorted(self.fields.items()))
            return "obj(" + inner + ")"
        return "unknown"


def _promote(shape: Shape, kind: str) -> None:
    """Move a shape to a container kind without discarding what is already known."""
    if shape.kind in ("unknown", kind):
        shape.kind = kind
    # A conflicting observation (used as both mapping and sequence) leaves the first
    # kind in place; synthesis then produces the more common one rather than guessing.


def _const_type(node: ast.expr) -> str | None:
    if isinstance(node, ast.Constant):
        v = node.value
        for t in (bool, int, float, str, bytes):
            if isinstance(v, t):
                return t.__name__
    return None


def _walk_target(root: Shape, node: ast.expr) -> Shape | None:
    """Resolve an access expression to the sub-shape it designates, creating as needed.

    ``cfg["limits"]["max"]`` walks root -> mapping['limits'] -> mapping['max'].
    Returns None when the expression is not rooted in the parameter.
    """
    if isinstance(node, ast.Name):
        return root
    if isinstance(node, ast.Subscript):
        parent = _walk_target(root, node.value)
        if parent is None:
            return None
        key = node.slice.value if isinstance(node.slice, ast.Constant) else None
        if isinstance(key, str):
            _promote(parent, "mapping")
            return parent.fields.setdefault(key, Shape())
        if isinstance(key, int):
            _promote(parent, "sequence")
            parent.min_len = max(parent.min_len, key + 1)
            if parent.element is None:
                parent.element = Shape()
            return parent.element
        return None
    if isinstance(node, ast.Attribute):
        parent = _walk_target(root, node.value)
        if parent is None:
            return None
        _promote(parent, "object")
        return parent.fields.setdefault(node.attr, Shape())
    return None


def _root_name(node: ast.expr) -> str | None:
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


class _UsageVisitor(ast.NodeVisitor):
    """Collect structural facts about *params* from every use in a subtree."""

    def __init__(self, shapes: dict[str, Shape]) -> None:
        self.shapes = shapes

    def _shape_for(self, node: ast.expr) -> Shape | None:
        name = _root_name(node)
        if name is None or name not in self.shapes:
            return None
        return _walk_target(self.shapes[name], node)

    def visit_Compare(self, node: ast.Compare) -> None:
        # p == <const>  /  p > <const>   -> scalar of the constant's type
        for side, other in ((node.left, node.comparators[0]),
                            (node.comparators[0], node.left)):
            sh = self._shape_for(side)
            t = _const_type(other)
            if sh is not None and t is not None and sh.kind in ("unknown", "scalar"):
                sh.kind, sh.type_name = "scalar", t
        # p in (…) -> p is a scalar matching the members
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, (ast.In, ast.NotIn)) and isinstance(
                    comp, (ast.Tuple, ast.List, ast.Set)):
                sh = self._shape_for(node.left)
                if sh is not None and comp.elts:
                    t = _const_type(comp.elts[0])
                    if t:
                        sh.kind, sh.type_name = "scalar", t
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # len(p) > n is handled by the enclosing Compare; here record sizedness.
        if isinstance(node.func, ast.Name) and node.func.id == "len" and node.args:
            sh = self._shape_for(node.args[0])
            if sh is not None and sh.kind == "unknown":
                sh.kind = "sequence"
        # p.method(...) -> receiver type from the method name
        if isinstance(node.func, ast.Attribute):
            sh = self._shape_for(node.func.value)
            if sh is not None:
                m = node.func.attr
                owner = _METHOD_OWNER.get(m)
                if owner is None and m in _STR_OR_BYTES and node.args:
                    at = _const_type(node.args[0])
                    owner = at if at in ("str", "bytes") else "str"
                if owner in ("str", "bytes"):
                    if sh.kind in ("unknown", "scalar"):
                        sh.kind, sh.type_name = "scalar", owner
                elif owner == "dict":
                    _promote(sh, "mapping")
                elif owner in ("list", "set"):
                    _promote(sh, "sequence")
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        # for a, b in p:  ->  p is a sequence of 2-tuples (the h11 headers shape)
        sh = self._shape_for(node.iter)
        if sh is not None:
            _promote(sh, "sequence")
            if isinstance(node.target, ast.Tuple):
                sh.tuple_arity = len(node.target.elts)
                sh.min_len = max(sh.min_len, 1)
                if sh.element is None:
                    sh.element = Shape(kind="sequence", tuple_arity=sh.tuple_arity)
                # Sub-shapes of the unpacked names are learned by a nested pass.
                names = {e.id: Shape() for e in node.target.elts if isinstance(e, ast.Name)}
                if names:
                    for stmt in node.body:
                        _collect(stmt, names)
                    sh.element.fields = {i: names[e.id]
                                         for i, e in enumerate(node.target.elts)
                                         if isinstance(e, ast.Name)}
        self.generic_visit(node)


def _collect(node: ast.AST, shapes: dict[str, Shape]) -> None:
    """Run both usage passes over *node* — accesses/methods, then ``len()`` bounds.

    Kept together so a nested scope (a ``for`` body, whose unpacked names get their own
    shapes) learns lengths as well, not only accesses.
    """
    _UsageVisitor(shapes).visit(node)
    _apply_len_bounds(node, shapes)


def _enclosing_func(tree: ast.Module, lineno: int):
    best = None
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(n, "end_lineno", n.lineno) or n.lineno
            if n.lineno <= lineno <= end and (best is None or n.lineno > best.lineno):
                best = n
    return best


def shapes_of(fn: ast.FunctionDef | ast.AsyncFunctionDef, params: Iterable[str]) -> dict[str, Shape]:
    """A structural shape per name of *params*, recovered from their uses in the body of *fn*."""
    shapes = {p: Shape() for p in params}
    for stmt in fn.body:
        _collect(stmt, shapes)
    return shapes


def infer_shapes(source_path: str, lineno: int, params: list[str]) -> dict[str, Shape]:
    """Recover a structural shape per parameter from its uses in the enclosing function."""
    try:
        source = Path(os.path.realpath(source_path)).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return {p: Shape() for p in params}
    fn = _enclosing_func(tree, lineno)
    if fn is None:
        return {p: Shape() for p in params}
    return shapes_of(fn, params)


def _apply_len_bounds(fn: ast.AST, shapes: dict[str, Shape]) -> None:
    for node in ast.walk(fn):
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            continue
        left, right, op = node.left, node.comparators[0], node.ops[0]
        for call, other, call_left in ((left, right, True), (right, left, False)):
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == "len" and call.args):
                continue
            n = other.value if isinstance(other, ast.Constant) else None
            if not isinstance(n, int) or isinstance(n, bool):
                continue
            name = _root_name(call.args[0])
            if name not in shapes:
                continue
            sh = _walk_target(shapes[name], call.args[0])
            if sh is None:
                continue
            need = n
            if (isinstance(op, ast.Gt) and call_left) or (isinstance(op, ast.Lt) and not call_left):
                need = n + 1
            elif (isinstance(op, ast.GtE) and call_left) or (isinstance(op, ast.LtE) and not call_left):
                need = n
            elif isinstance(op, ast.Eq):
                need = n
            else:
                continue
            sh.min_len = max(sh.min_len, need)
            if sh.kind == "unknown":
                sh.kind = "sequence"


_SCALAR_DEFAULT = {"int": 1, "float": 1.0, "bool": True, "str": "x", "bytes": b"x"}


def synthesize(shape: Shape) -> Any:
    """Build a concrete value matching *shape*, for use as a seed input."""
    if shape.kind == "scalar":
        return _SCALAR_DEFAULT.get(shape.type_name or "", "x")
    if shape.kind == "mapping":
        return {k: synthesize(v) for k, v in shape.fields.items()}
    if shape.kind == "sequence":
        if shape.tuple_arity and shape.element is not None:
            row = tuple(synthesize(shape.element.fields.get(i, Shape()))
                        for i in range(shape.tuple_arity))
            return [row for _ in range(max(shape.min_len, 1))]
        el = shape.element or Shape()
        return [synthesize(el) for _ in range(max(shape.min_len, 1))]
    if shape.kind == "object":
        from types import SimpleNamespace
        return SimpleNamespace(**{k: synthesize(v) for k, v in shape.fields.items()})
    return "x"
