from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class TypeHint:
    rendered: str
    source: str
    confidence: int


ANY = TypeHint("Any", "unknown", 0)


@dataclass(frozen=True)
class ParamInfo:
    name: str
    kind: str
    type_hint: TypeHint
    has_default: bool = False


@dataclass(frozen=True)
class CallableInfo:
    module_path: str
    module_key: str
    callable_id: str
    qualname: str
    name: str
    kind: str
    decorators: list[str] = field(default_factory=list)
    params: list[ParamInfo] = field(default_factory=list)
    returns: TypeHint = ANY
    skipped_reason: str | None = None
    lineno: int | None = None
    end_lineno: int | None = None
    is_async: bool = False


@dataclass(frozen=True)
class WrapperExportPlan:
    export_name: str
    target_qualname: str
    kind: str


@dataclass(frozen=True)
class WrapperPlan:
    module_path: str
    module_key: str
    wrapper_relpath: str
    exports: list[WrapperExportPlan] = field(default_factory=list)


def _unparse(expr: ast.AST) -> str | None:
    try:
        return ast.unparse(expr)
    except Exception:
        return None


def hint_from_annotation(expr: ast.expr | None) -> TypeHint:
    if expr is None:
        return ANY
    rendered = _unparse(expr)
    if rendered is None:
        return ANY
    return TypeHint(rendered, "explicit", 100)


def hint_from_default(expr: ast.expr | None) -> TypeHint:
    if expr is None:
        return ANY
    if isinstance(expr, ast.Constant):
        v = expr.value
        if v is None:
            return TypeHint("Optional[Any]", "default", 40)
        if isinstance(v, bool):
            return TypeHint("bool", "default", 80)
        if isinstance(v, int):
            return TypeHint("int", "default", 80)
        if isinstance(v, float):
            return TypeHint("float", "default", 80)
        if isinstance(v, str):
            return TypeHint("str", "default", 80)
        if isinstance(v, bytes):
            return TypeHint("bytes", "default", 80)
    if isinstance(expr, ast.List):
        return TypeHint("List[Any]", "default", 30)
    if isinstance(expr, ast.Dict):
        return TypeHint("Dict[Any, Any]", "default", 30)
    if isinstance(expr, ast.Tuple):
        return TypeHint("Tuple[Any, ...]", "default", 30)
    return ANY


def infer_usage_hints(fn: ast.AST, param_names: Iterable[str]) -> dict[str, TypeHint]:
    names = set(param_names)
    raw: dict[str, set[str]] = {}

    def add(name: str, hint: str) -> None:
        if name not in names:
            return
        raw.setdefault(name, set()).add(hint)

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Name) and node.func.id == "len" and node.args:
                a0 = node.args[0]
                if isinstance(a0, ast.Name):
                    add(a0.id, "Sequence[Any]")
            self.generic_visit(node)

        def visit_Compare(self, node: ast.Compare) -> None:
            if isinstance(node.left, ast.Name):
                for c in node.comparators:
                    if isinstance(c, ast.Constant) and c.value == "":
                        add(node.left.id, "str")
            self.generic_visit(node)

        def visit_BinOp(self, node: ast.BinOp) -> None:
            if isinstance(node.op, ast.Add) and isinstance(node.left, ast.Name):
                if isinstance(node.right, ast.Constant):
                    if isinstance(node.right.value, int):
                        add(node.left.id, "int")
                    if isinstance(node.right.value, float):
                        add(node.left.id, "float")
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.value, ast.Name) and node.attr == "encode":
                add(node.value.id, "str")
            if isinstance(node.value, ast.Name) and node.attr == "decode":
                add(node.value.id, "bytes")
            self.generic_visit(node)

    V().visit(fn)

    out: dict[str, TypeHint] = {}
    for name, opts in raw.items():
        if len(opts) == 1:
            out[name] = TypeHint(next(iter(opts)), "heuristic", 20)
        else:
            out[name] = ANY
    return out


def choose_hint(*candidates: TypeHint) -> TypeHint:
    best = ANY
    for h in candidates:
        if h.rendered != "Any" and h.confidence >= best.confidence:
            best = h
    return best
