"""``infer_params``: the parameters of one ``def``, each typed from numbered sources (docs/TYPEINFER.md).

Every source reads source text or data it is handed; none runs the target.  Sources are numbered by trust and
fused into one ranked list of candidates per parameter, each candidate a ``find_inputs`` kind
(``tharkun/typeinfer/core/kinds.py``).
"""

from __future__ import annotations

import ast
import heapq
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tharkun.solve.frontend.constructor import function_parameters
from tharkun.solve.frontend.import_resolver import (
    find_function_def,
    load_class_def,
    parse_imports,
    parse_with_type_comments,
)
from tharkun.typeinfer.core.kinds import (annotation_kind, drawable, kind_key, optional, str_or_int, typing_imports,
                                          value_kind)
from tharkun.typeinfer.core.shape import Shape, shapes_of
from tharkun.typeinfer.core.usage import usage_type

#: the sources, numbered by trust (docs/TYPEINFER.md "Sources"); 6 checkers, 7 docstring and 8 MonkeyType are
#: ruled, not built (A4)
SOURCES = {1: "annotation", 2: "default", 3: "usage", 4: "propagation", 5: "runtime", 9: "caller", 10: "fallback"}
#: a source's weight: its trust, 1/n
WEIGHT = {n: 1 / n for n in SOURCES}
_NO_VALUE = object()


@dataclass(frozen=True)
class Candidate:
    """One type a parameter may have.  ``source`` is ``"<n>:<name>"`` (``"1:type_comment"`` for a PEP 484 comment);
    ``weight`` is the summed trust of every source that gave the same kind, ``agree`` their other tags (and the
    source of a ``None`` arm it gained)."""

    kind: dict
    text: str
    source: str
    weight: float
    declared: bool
    agree: tuple[str, ...] = ()


@dataclass(frozen=True)
class Report:
    """A parameter's ranked candidates.  ``status``: ``established`` (a source 1-9 gave a kind ``find_inputs``
    draws), ``unavailable`` (an object parameter: sources named only a class, a callable or another kind it cannot
    draw; never a fallback), ``absent`` (no source named a type: nothing, ``Any``, or a ``None`` default alone; the
    ruling-4 fallback).  ``class_ref``: for the best class candidate, where the class is defined
    (``{"class", "path", "line"}``) or why it was not found (``{"class", "refused"}``).  ``shape``: the structure
    the body's uses give the parameter (a mapping's keys, a sequence's length, an object's attributes;
    ``core/shape.py``), or ``None`` when its uses say nothing of it (source 3 off, or ``declared_only``)."""

    status: str
    candidates: tuple[Candidate, ...]
    class_ref: dict | None = None
    shape: Shape | None = None


@dataclass(frozen=True)
class Evidence:
    """Data a caller hands in about one parameter: ``source`` 5 (runtime: a snapshot value, a probe child's type)
    or 9 (caller: e.g. an IR type id as ``label`` with the kind it means).  Give one of ``kind``, ``text`` (an
    annotation-like type name) or ``value`` (an observed value)."""

    param: str
    source: int
    label: str
    kind: dict | None = None
    text: str | None = None
    value: Any = _NO_VALUE
    declared: bool = False

    def as_kind(self) -> dict:
        if self.kind is not None:
            if not (isinstance(self.kind, dict) and isinstance(self.kind.get("kind"), str)):
                raise ValueError(f"evidence for {self.param!r}: kind must be a kind spec, got {self.kind!r}")
            return self.kind
        if self.text is not None:
            return annotation_kind(self.text)
        if self.value is not _NO_VALUE:
            return value_kind(self.value)
        raise ValueError(f"evidence for {self.param!r} gives no kind, text or value")


def infer_params(
    source_path: str,
    func: str | ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    evidence: Iterable[Evidence] = (),
    sources: Iterable[int] | None = None,
    declared_only: bool = False,
) -> dict[str, Report]:
    """A :class:`Report` for every parameter of ``func`` (a top-level function, ``method`` or ``Class.method``:
    ``import_resolver.py::find_function_def``) in ``source_path``, receiver and ``*args``/``**kwargs`` left out.
    ``func`` may also be a ``def`` node of that file that the caller found by a rule of its own (parsed with
    ``import_resolver.py::parse_with_type_comments``, or its type comments are not read); the receiver of a node
    is not dropped.

    ``sources`` picks the numbered sources (default: all built ones).  ``declared_only`` keeps only declared types
    (annotations, type comments, evidence marked ``declared``) and never falls back.  Raises ``LookupError`` when
    the file cannot be read or parsed or has no such ``def``."""
    wanted = set(SOURCES) if sources is None else set(sources)
    path = os.path.realpath(source_path)
    try:
        tree = parse_with_type_comments(Path(path).read_text(encoding="utf-8", errors="ignore"))
    except OSError as exc:
        raise LookupError(f"{source_path}: {exc}") from None
    if isinstance(func, str):
        node, owner = find_function_def(tree, func) if tree is not None else (None, None)
        if node is None:
            raise LookupError(f"{source_path}: no def {func!r}")
    else:
        node, owner, func = func, None, func.name
    receiver = owner is not None and not any(
        isinstance(d, ast.Name) and d.id == "staticmethod" for d in node.decorator_list)
    params = [p for p in function_parameters(node, drop_receiver=receiver)
              if p.kind not in ("var_positional", "var_keyword")]
    hinted = {a.arg for a in ast.walk(node.args) if isinstance(a, ast.arg) and a.annotation is not None}
    aliases = _aliases(path, tree, node)
    imports = typing_imports(tree)
    found: dict[str, list] = {p.name: [] for p in params}  # name -> [(n, tag, kind, text, declared)]

    for p in params:
        if 1 in wanted and p.annotation is not None:
            tag = "annotation" if p.name in hinted else "type_comment"
            found[p.name].append((1, tag, annotation_kind(p.annotation, aliases, imports), p.annotation, True))
        if 2 in wanted and p.default_is_literal and not declared_only:
            found[p.name].append((2, "default", value_kind(p.default), p.default_source or repr(p.default), False))
    usage = 3 in wanted and not declared_only
    if usage:
        for name, entries in _usage(node, found).items():
            found[name] += entries
    if 4 in wanted and not declared_only:
        from tharkun.solve.frontend.type_propagation import propagate_types

        for name, text in propagate_types(path, func).items():
            if name in found and name not in hinted:
                found[name].append((4, "propagation", annotation_kind(text, aliases, imports), text, False))
    for item in evidence:
        if item.source not in (5, 9):
            raise ValueError(f"evidence for {item.param!r}: source must be 5 (runtime) or 9 (caller)")
        if item.source in wanted and item.param in found and (item.declared or not declared_only):
            found[item.param].append((item.source, SOURCES[item.source], item.as_kind(), item.label, item.declared))
    if declared_only:
        found = {name: [e for e in entries if e[4]] for name, entries in found.items()}
    fallback = 10 in wanted and not declared_only
    shapes = shapes_of(node, found) if usage else {}
    return {name: _report(path, entries, fallback, shapes.get(name)) for name, entries in found.items()}


def _expressions(node: ast.FunctionDef | ast.AsyncFunctionDef) -> Iterable[ast.expr]:
    """The expressions of every statement of the body (a guard, an assigned value, a call statement ...)."""
    for stmt in node.body:
        for inner in ast.walk(stmt):
            if isinstance(inner, ast.stmt):
                yield from (c for c in ast.iter_child_nodes(inner) if isinstance(c, ast.expr))


def _rebinding(node: ast.FunctionDef | ast.AsyncFunctionDef, names: Iterable[str]) -> dict[str, tuple[int, int]]:
    """Where a parameter's original value dies: the end of the first assignment of the body's own level that gives
    it a FRESH value (``buf = []``; the new value does not mention the old one).  A value computed from the old one
    (``s = s.strip()``, ``n = int(n)``) is a coercion the original passes through, so its later uses still describe
    what the parameter may be; and after ``if c: s = []`` the original may still flow on."""
    wanted, ends = set(names), {}
    for stmt in node.body:
        if isinstance(stmt, ast.Assign) or (isinstance(stmt, ast.AnnAssign) and stmt.value is not None):
            mentioned = {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}
            for target in stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]:
                for inner in ast.walk(target):
                    if (isinstance(inner, ast.Name) and isinstance(inner.ctx, ast.Store) and inner.id in wanted
                            and inner.id not in mentioned):
                        ends.setdefault(inner.id, (stmt.end_lineno, stmt.end_col_offset))
    return ends


def _usage(node: ast.FunctionDef | ast.AsyncFunctionDef, found: dict) -> dict[str, list]:
    """Source 3: ``usage.py::usage_type`` over each expression that names a parameter, one entry per distinct kind
    (its text: the first rule and expression that gave it).  Expressions after the body gave the parameter a fresh
    value are not about what was passed in and are skipped (:func:`_rebinding`)."""
    dead = _rebinding(node, found)
    out: dict[str, list] = {}
    for expr in _expressions(node):
        names = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)} & set(found)
        for name in sorted(names - {"self", "cls"}):  # a receiver-named parameter has no usage type
            if name in dead and (expr.lineno, expr.col_offset) >= dead[name]:
                continue
            found_rule = usage_type(name, expr, parameter=True)
            if found_rule is None or found_rule[0] == "Any":
                continue
            type_text, rule = found_rule
            if rule == "isinstance" and type_text.split(".")[0] in found:
                continue  # A classinfo parameter's spelling does not name the input's class.
            kind = annotation_kind(type_text)
            entries = out.setdefault(name, [])
            if all(kind_key(kind) != kind_key(e[2]) for e in entries):
                entries.append((3, "usage", kind, f"{rule}: {ast.unparse(expr)[:120]}", False))
    return out


_ALIASED = (ast.Subscript, ast.Name, ast.Attribute)


def _own_aliases(tree: ast.Module) -> dict[str, ast.expr]:
    """The names a module binds to a type expression: ``X = Union[str, bytes]``, ``X: TypeAlias = ...``,
    ``type X = ...`` (the last binding wins)."""
    found: dict[str, ast.expr] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name, value = stmt.targets[0].id, stmt.value
        elif (isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None
              and ast.unparse(stmt.annotation).rsplit(".", 1)[-1] == "TypeAlias"):
            name, value = stmt.target.id, stmt.value
        elif isinstance(stmt, getattr(ast, "TypeAlias", ())):
            name, value = stmt.name.id, stmt.value
        else:
            continue
        if isinstance(value, _ALIASED) or (isinstance(value, ast.BinOp) and isinstance(value.op, ast.BitOr)):
            found[name] = value
    return found


def _aliases(path: str, tree: ast.Module, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, ast.expr]:
    """The type aliases the annotations of ``node`` may name: this module's, and for a name it imports from a project
    module (found statically, one hop, ``as`` renames followed) that module's alias and the aliases it names in turn."""
    # shortcut: an alias a package ``__init__`` re-exports is not followed, nor one named only in a type comment, and
    # nothing is cached (one parse per imported module per call); upgrade when A3/A5 call this per request.
    found = _own_aliases(tree)
    used = {n.id for a in ast.walk(node.args) if isinstance(a, ast.arg) and a.annotation is not None
            for n in ast.walk(a.annotation) if isinstance(n, ast.Name)}
    imported = {a.asname or a.name: a.name for s in ast.walk(tree) if isinstance(s, ast.ImportFrom) for a in s.names}
    wanted = {local for local in used & imported.keys() if local not in found}
    if not wanted:
        return found
    for local, module in parse_imports(path).items():
        if local not in wanted or module == path:
            continue
        try:
            other = parse_with_type_comments(Path(module).read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        theirs = _own_aliases(other) if other is not None else {}
        queue = [(imported[local], local)]  # (the name in that module, the name it goes by here)
        while queue:
            there, here = queue.pop()
            if there in theirs and here not in found:
                found[here] = theirs[there]
                queue += [(n.id, n.id) for n in ast.walk(theirs[there]) if isinstance(n, ast.Name)]
    return found


_NONE, _ANY = {"kind": "none"}, {"kind": "any"}


def _report(path: str, entries: list, fallback: bool, shape: Shape | None = None) -> Report:
    """The status (owner ruling 2026-10-20: ``None`` alone and ``Any`` name no type; a class or a callable is an
    object parameter), ruling 4's fallback for a parameter no source typed, the ``None`` arm a ``None`` default (or
    an observed ``None``) adds to every decided kind (``Optional[T]``), then the fusion: one candidate per distinct
    kind, its weight summed over the sources that gave it, ranked declared first, then by weight, then by how many
    observations gave it (several runtime values of one type), then by trust."""
    nones = [e for e in entries if e[2] == _NONE]
    named = [e for e in entries if e[2] != _NONE]
    decided = [e for e in named if e[2] != _ANY]
    status = ("established" if any(drawable(e[2]) for e in decided) else
              "unavailable" if decided else "absent")
    if fallback and status == "absent":
        named += [(10, "fallback", str_or_int(), "str | int", False),
                  (10, "fallback", {"kind": "list", "of": str_or_int()}, "list[str | int]", False)]
    none_tags: tuple = ()
    if nones and any(e[2] != _ANY for e in named):
        none_tags = tuple(dict.fromkeys(f"{e[0]}:{e[1]}" for e in nones))
        named = [e if e[2] == _ANY else (e[0], e[1], optional(e[2]), *e[3:]) for e in named]
        nones = []
    groups: dict[str, list] = {}
    for entry in named + nones:
        groups.setdefault(kind_key(entry[2]), []).append(entry)
    ranked = []
    for group in groups.values():
        group.sort(key=lambda e: e[0])
        best = group[0]
        tags = list(dict.fromkeys(f"{n}:{tag}" for n, tag, *_ in group))
        if best[2] != _ANY:
            tags += [t for t in none_tags if t not in tags]  # the source of the None arm
        weight = sum(WEIGHT[n] for n in {e[0] for e in group})
        candidate = Candidate(best[2], best[3], tags[0], weight, any(e[4] for e in group), tuple(tags[1:]))
        ranked.append((not candidate.declared, -weight, -len(group), best[0], candidate))
    candidates = [r[-1] for r in sorted(ranked, key=lambda r: r[:4])]  # a stable sort: ties keep source order
    cls = next(filter(None, map(_class_of, (c.kind for c in candidates))), None)
    return Report(status, tuple(candidates), _class_ref(path, cls) if cls else None,
                  shape if shape is not None and shape.kind != "unknown" else None)


def _class_of(kind: dict) -> str | None:
    """The class an ``object`` kind (or an arm of a union, ``Optional[Timer]``) names."""
    for arm in kind["of"] if kind["kind"] == "one_of" else [kind]:
        if arm["kind"] == "object" and "class" in arm:
            return arm["class"]
    return None


def _class_ref(path: str, text: str) -> dict:
    """Where the class ``text`` names is defined: this module, or a module it imports (static, never imported)."""
    name = text.split("[", 1)[0].strip()
    if name in ("object", "type", "SimpleNamespace"):
        return {"class": name, "refused": f"{name} is not a class of the program"}
    found = load_class_def(path, name)
    where = path
    if found is None:
        head, _, rest = name.partition(".")
        module = parse_imports(path).get(head)
        if module is not None:
            where = module
            found = load_class_def(module, rest or head)
    if found is None:
        return {"class": name, "refused": "no class definition in this module or a project module it imports"}
    return {"class": name, "path": where, "line": found.lineno}


def candidate_typings(reports: dict[str, Report], *, limit: int = 16, per_param: int = 4) -> list[dict[str, dict]]:
    """Joint typings of the parameters (``{param: kind}``), best first: the top candidate of every parameter, then
    every typing one rank further down, and so on (ranked by the sum of the candidates' ranks; among equal sums a
    later parameter moves down first).  A parameter with no candidate is left out; at most ``per_param`` candidates of each are combined."""
    names = [name for name, report in reports.items() if report.candidates]
    options = [[c.kind for c in reports[name].candidates[:per_param]] for name in names]
    start = (0,) * len(names)
    heap, seen, out = [(0, start)], {start}, []
    while heap and len(out) < limit:
        rank, index = heapq.heappop(heap)
        out.append({name: options[i][j] for i, (name, j) in enumerate(zip(names, index))})
        for i in range(len(names)):
            if index[i] + 1 < len(options[i]):
                step = index[:i] + (index[i] + 1,) + index[i + 1:]
                if step not in seen:
                    seen.add(step)
                    heapq.heappush(heap, (rank + 1, step))
    return out
