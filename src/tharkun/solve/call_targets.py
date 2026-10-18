"""The ``call_target`` fact: what every call site of the task program calls ([internal ref]).

SOLVING, by ``docs/LAYERS.md``'s rule: stdlib only (``ast``, ``builtins``), it executes nothing and imports
nothing above ``tharkun.solve``.  It is the second reader of the import-graph walker of
``solve/module_namespace.py`` (the first is ``module_constants``): a call site's callee is a NAME, and what a
name denotes is exactly the question that module answers, once, for the whole graph.

One record per call site of the ENTRY module (the file the verifier runs; the harness every consumer works on
is built from it), in source order::

    {"kind": "call_target", "module", "line", "col", "end_line", "end_col",
                                                           # the call expression, in the ORIGINAL source
     "scope", "callee",                                    # the lookup key, see below
     "target", "name", "def", "dispatch", "proven", "reason", "provenance"}

``target`` is what the callee resolves to:

* ``user_class`` / ``user_function`` -- a ``class`` / ``def`` of the program, undecorated; ``name`` is its name
  and ``def`` its definition site ``{"module", "line", "col"}`` (an imported one reports the module that
  defines it);
* ``user_method`` -- ``Cls.m(...)`` with ``Cls`` a user class, zero-argument ``super().m(...)`` inside a
  method of a user class with one user base, and ``recv.m(...)`` where ``recv`` is a variable whose ONE
  binding in its scope is ``recv = Cls(...)`` for a proven user class (the receiver's type, [internal ref]; the
  record then carries ``receiver``: ``{"name", "class", "line"}`` of that binding); ``name`` is
  ``"<defining class>.<method>"`` and ``def`` the ``def`` found by walking the single-inheritance chain;
* ``imported`` -- a name or a module attribute that comes from an ``import`` (``sqrt`` after ``from math
  import sqrt``; ``math.sqrt``); ``name`` is the dotted origin;
* ``builtin`` -- a builtin no module-level or local binding hides;
* ``unknown`` -- everything else, with the reason: a local binding hides the name, a method on a receiver
  whose type is not statically known (``dispatch`` is then ``"dynamic"``), a decorated ``class``/``def``, a
  binding by assignment, an expression that is not a name.

A receiver is typed only from a SINGLE binding (so every read of the variable sees an object that statement
made): exactly one binding statement in the variable's own scope (a function: parameters, assignments, ``for``
/ ``with`` / ``except`` / ``import`` / ``def`` / ``class`` targets, ``:=``, ``del`` all count, and no nested
scope declares it ``nonlocal``; the module: ``module_constants`` shows nothing but that one, possibly
conditional, binding -- ``if __name__ == "__main__":`` is the usual one -- and nothing writes it through
``global``), that statement is ``recv = Cls(...)`` with ``Cls`` resolved in the SAME scope to a proven user
class, and the module cannot write its own namespace.  It types the NAME; whether ``Cls(...)`` builds a
``Cls`` (``__new__``, a metaclass) and whether the instance carries an attribute that hides the method are
the consumer's checks, like every other class property (``solve/drawflow.py``, ``sv/_symexec_objects.py``).

``proven`` is true ONLY when the name's binding is proven by the module-namespace fact
(``solve/module_namespace.py::module_constants``: nothing in the import graph rebinds it AND the graph is
complete) and no local binding of the call's own scope chain hides it; a builtin is proven when nothing
binds the name and the graph is complete.  An unproven record keeps its declared ``target`` (a consumer can
say what it refused) and ``reason`` says what rebinds or hides the name.  ``proven`` is a fact about NAMES:
the object a proven name denotes can still be mutated (``Cls.m = f``), which a consumer that executes the
class checks against the live object (``sv/_symexec_objects.py``) or against the program's own attribute
stores (``solve/drawflow.py``).

``scope`` is ``"<main>"`` for module-level code (the main block included) and the dotted qualified name of
the enclosing def/class otherwise (``"Canvas.__init__"``, ``"f.g"``); ``callee`` is ``ast.unparse`` of the
callee expression.  Those two are what survives the harness (``sv/sites.py::build_harness`` rewrites line
numbers) and the analysis slice, so a consumer looks a call up by ``(scope, callee)`` through
:func:`call_target_index`; two sites of one scope that spell the callee alike but resolve differently (a
comprehension variable shadowing the name in one of them) make that key AMBIGUOUS, and the index answers
``unknown`` for it.

A nested call chain (``a.f().g()``) starts at one position for each of its calls, so a record is identified by
its whole extent: ``(module, line, col, end_line, end_col)``.
"""
from __future__ import annotations

import ast
import builtins
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from tharkun.solve.module_namespace import _Project, _rel

__all__ = ["call_targets", "call_target_index", "MAX_RECORDS"]

#: records published for one program (a larger program keeps the first ones; a call without a record is
#: simply unresolved for every consumer)
MAX_RECORDS = 5000

_MAIN = "<main>"
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
_COMPS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_BUILTIN_NAMES = frozenset(dir(builtins))
#: a name of these in the entry module can write its namespace or the builtins behind the analysis's back
#: (``globals()['len'] = f``, ``builtins.len = f``): no builtin is proven then
_NAMESPACE_WRITERS = frozenset({"globals", "vars", "locals", "exec", "eval", "setattr", "delattr", "compile",
                                "__import__", "__builtins__", "builtins"})


# ---------------------------------------------------------------------------
# names bound by one scope
# ---------------------------------------------------------------------------

def _store_names(target: ast.AST) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for e in target.elts:
            yield from _store_names(e)
    elif isinstance(target, ast.Starred):
        yield from _store_names(target.value)


def _bound_in(body: Sequence[ast.AST]) -> Set[str]:
    """Names the statements bind in THEIR scope: not inside a nested def / class / lambda / comprehension
    (those bind in their own), but their own names, decorators-free.  A ``:=`` anywhere inside binds here
    (a comprehension's walrus binds in the enclosing function), which only over-approximates."""
    out: Set[str] = set()
    stack: List[ast.AST] = list(body)
    while stack:
        n = stack.pop()
        if isinstance(n, _FUNCS + (ast.ClassDef,)):
            out.add(n.name)
            stack.extend(n.decorator_list)
            if isinstance(n, ast.ClassDef):
                stack.extend(n.bases)
                stack.extend(k.value for k in n.keywords)
            else:
                stack.extend(n.args.defaults)
                stack.extend(d for d in n.args.kw_defaults if d is not None)
            continue
        if isinstance(n, ast.Lambda):
            stack.extend(n.args.defaults)
            stack.extend(d for d in n.args.kw_defaults if d is not None)
            continue
        if isinstance(n, _COMPS):
            # the first iterable is evaluated in the enclosing scope; the rest is the comprehension's own
            stack.append(n.generators[0].iter)
            for g in n.generators[1:]:
                stack.append(g.iter)
            for sub in ast.walk(n):
                if isinstance(sub, ast.NamedExpr):
                    out.update(_store_names(sub.target))
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if a.name != "*":
                    out.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.MatchMapping) and n.rest:
            out.add(n.rest)
        stack.extend(ast.iter_child_nodes(n))
    return out


def _function_params(fn: ast.AST) -> Set[str]:
    a = fn.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    if a.vararg:
        names.add(a.vararg.arg)
    if a.kwarg:
        names.add(a.kwarg.arg)
    return names


def _binding_sites(body: Sequence[ast.AST], name: str) -> List[ast.AST]:
    """The nodes that bind ``name`` in the scope ``body`` makes (not inside a nested def / class / lambda /
    comprehension, whose bindings are their own): the statement for an assignment, ``for``, ``with``, ``def``,
    ``class``, ``import``, ``except`` or ``del``, the walrus, the match capture.  A ``:=`` anywhere (a
    comprehension's included) binds here."""
    out: List[ast.AST] = []
    stack: List[ast.AST] = list(body)
    while stack:
        n = stack.pop()
        if isinstance(n, _FUNCS + (ast.ClassDef,)):
            if n.name == name:
                out.append(n)
            stack.extend(n.decorator_list)
            continue
        if isinstance(n, ast.Lambda):
            stack.extend(n.args.defaults)
            continue
        if isinstance(n, _COMPS):
            for sub in ast.walk(n):
                if isinstance(sub, ast.NamedExpr) and name in _store_names(sub.target):
                    out.append(sub)
            stack.append(n.generators[0].iter)
            continue
        if isinstance(n, ast.Assign) and any(name in _store_names(t) for t in n.targets):
            out.append(n)
        elif isinstance(n, (ast.AugAssign, ast.AnnAssign)) and name in _store_names(n.target):
            out.append(n)
        elif isinstance(n, (ast.For, ast.AsyncFor)) and name in _store_names(n.target):
            out.append(n)
        elif isinstance(n, (ast.With, ast.AsyncWith)) and any(
                i.optional_vars is not None and name in _store_names(i.optional_vars) for i in n.items):
            out.append(n)
        elif isinstance(n, ast.Delete) and any(name in _store_names(t) for t in n.targets):
            out.append(n)
        elif isinstance(n, (ast.Import, ast.ImportFrom)) and any(
                a.name != "*" and (a.asname or a.name.split(".")[0]) == name for a in n.names):
            out.append(n)
        elif isinstance(n, ast.ExceptHandler) and n.name == name:
            out.append(n)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name == name:
            out.append(n)
        elif isinstance(n, ast.MatchMapping) and n.rest == name:
            out.append(n)
        elif isinstance(n, ast.NamedExpr) and name in _store_names(n.target):
            out.append(n)
        stack.extend(ast.iter_child_nodes(n))
    return out


def _nested_declarations(body: Sequence[ast.AST], name: str, kind) -> bool:
    """Does any scope nested in ``body`` declare ``name`` ``global`` / ``nonlocal`` (``kind``)?  Such a scope
    can rebind the variable of the enclosing one."""
    for st in body:
        for n in ast.walk(st):
            if isinstance(n, _FUNCS + (ast.ClassDef,)):
                if name in _declared(n.body, kind):
                    return True
    return False


def _declared(body: Sequence[ast.AST], kind) -> Set[str]:
    out: Set[str] = set()
    stack: List[ast.AST] = list(body)
    while stack:
        n = stack.pop()
        if isinstance(n, kind):
            out.update(n.names)
        if isinstance(n, _FUNCS + (ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(n))
    return out


class _Frame:
    """One scope between the module and a call site."""

    def __init__(self, kind: str, node: Optional[ast.AST], names: Set[str], globals_: Set[str],
                 nonlocals: Set[str]):
        self.kind = kind            # "module" | "function" | "class" | "lambda" | "comp"
        self.node = node
        self.names = names          # bound here
        self.globals_ = globals_    # declared ``global``
        self.nonlocals = nonlocals


def _function_names(fn: ast.AST) -> Set[str]:
    a = fn.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    if a.vararg:
        names.add(a.vararg.arg)
    if a.kwarg:
        names.add(a.kwarg.arg)
    if isinstance(fn, ast.Lambda):
        return names
    return names | _bound_in(fn.body)


def _comp_names(comp: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for g in comp.generators:
        names.update(_store_names(g.target))
    return names


# ---------------------------------------------------------------------------
# the producer
# ---------------------------------------------------------------------------

def call_targets(entry_path: str, constants: Sequence[Dict[str, Any]], roots: Sequence[str] = (), *,
                 max_modules: Optional[int] = None) -> List[Dict[str, Any]]:
    """One ``call_target`` record per call site of the entry module (see the module docstring).

    ``constants`` are the records ``solve/module_namespace.py::module_constants`` returned for the SAME
    ``entry_path`` / ``roots`` / ``max_modules``: ``proven`` of a name is read from them and from nothing
    else.  ``[]`` when the file cannot be read."""
    project = _Project(entry_path, roots, max_modules)
    entry = project.mods.get(project.entry)
    if entry is None or entry.tree is None:
        return []
    return _Producer(project, entry, constants).run()


class _Producer:
    def __init__(self, project: _Project, entry, constants: Sequence[Dict[str, Any]]):
        self.project = project
        self.entry = entry
        self.rel = _rel(project)
        self.module = self.rel(entry.path)
        self.complete = not project.incomplete
        self.namespace_writable = any(
            (isinstance(n, ast.Name) and n.id in _NAMESPACE_WRITERS)
            or (isinstance(n, ast.Attribute) and n.attr in _NAMESPACE_WRITERS)
            or (isinstance(n, ast.Import) and any(a.name.split(".")[0] in _NAMESPACE_WRITERS for a in n.names))
            or (isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in _NAMESPACE_WRITERS)
            for n in ast.walk(entry.tree))
        self.consts = {c["name"]: c for c in constants if c.get("module") == self.module}
        self.binder = project.binder(entry.path)
        self.records: List[Dict[str, Any]] = []
        self.frames: List[_Frame] = []
        self.qual: List[str] = []

    # -- walking -------------------------------------------------------------

    def run(self) -> List[Dict[str, Any]]:
        tree = self.entry.tree
        self.frames = [_Frame("module", tree, set(), set(), set())]
        self._body(tree.body)
        return self.records[:MAX_RECORDS]

    def _body(self, stmts: Sequence[ast.AST]) -> None:
        for st in stmts:
            self._visit(st)

    def _visit(self, node: ast.AST) -> None:
        if isinstance(node, _FUNCS):
            for d in node.decorator_list:
                self._visit(d)
            for d in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                self._visit(d)
            self._enter("function", node, _function_names(node), node.name, node.body)
        elif isinstance(node, ast.ClassDef):
            for d in node.decorator_list + node.bases + [k.value for k in node.keywords]:
                self._visit(d)
            self._enter("class", node, _bound_in(node.body), node.name, node.body)
        elif isinstance(node, ast.Lambda):
            for d in node.args.defaults + [x for x in node.args.kw_defaults if x is not None]:
                self._visit(d)
            self.frames.append(_Frame("lambda", node, _function_names(node), set(), set()))
            self._visit(node.body)
            self.frames.pop()
        elif isinstance(node, _COMPS):
            # the first iterable is evaluated outside the comprehension's own scope
            self._visit(node.generators[0].iter)
            self.frames.append(_Frame("comp", node, _comp_names(node), set(), set()))
            for i, g in enumerate(node.generators):
                if i:
                    self._visit(g.iter)
                for cond in g.ifs:
                    self._visit(cond)
            for part in ((node.key, node.value) if isinstance(node, ast.DictComp) else (node.elt,)):
                self._visit(part)
            self.frames.pop()
        else:
            if isinstance(node, ast.Call):
                self._record(node)
            for child in ast.iter_child_nodes(node):
                self._visit(child)

    def _enter(self, kind: str, node: ast.AST, names: Set[str], name: str, body: Sequence[ast.AST]) -> None:
        self.frames.append(_Frame(kind, node, names, _declared(body, ast.Global), _declared(body, ast.Nonlocal)))
        self.qual.append(name)
        self._body(body)
        self.qual.pop()
        self.frames.pop()

    # -- resolving one call --------------------------------------------------

    def _scope(self) -> str:
        return ".".join(self.qual) if self.qual else _MAIN

    def _local_binding(self, name: str) -> Optional[str]:
        """The kind of scope that binds ``name`` between the call and the module (``None`` when the module's
        namespace -- or the builtins -- answer).  Class scopes are visible only to code directly in their body."""
        for i, frame in enumerate(reversed(self.frames)):
            if frame.kind == "module":
                return None
            if frame.kind == "class" and i != 0:
                continue
            if name in frame.globals_:
                return None
            if name in frame.names or name in frame.nonlocals:
                return frame.kind
        return None

    def _binding_frame(self, name: str):
        """``(index, frame)`` of the scope that binds ``name`` for the call being resolved: a function scope, or
        ``(0, module frame)`` when the module's namespace answers; ``None`` when a lambda, comprehension or class
        scope binds it."""
        for i in range(len(self.frames) - 1, -1, -1):
            frame = self.frames[i]
            if frame.kind == "module":
                return i, frame
            if frame.kind == "class" and i != len(self.frames) - 1:
                continue
            if name in frame.globals_:
                return 0, self.frames[0]
            if name in frame.names or name in frame.nonlocals:
                return (i, frame) if frame.kind == "function" and name not in frame.nonlocals else None
        return 0, self.frames[0]

    def _record(self, call: ast.Call) -> None:
        target = self._resolve(call.func)
        rec = {
            "kind": "call_target", "module": self.module, "line": call.lineno, "col": call.col_offset,
            "end_line": call.end_lineno, "end_col": call.end_col_offset,
            "scope": self._scope(), "callee": ast.unparse(call.func),
            "target": target["target"], "name": target.get("name"), "def": target.get("def"),
            "dispatch": target.get("dispatch", "static"), "proven": bool(target.get("proven")),
            "reason": target["reason"],
            "provenance": {"producer": "tharkun", "method": "static-graph", "complete": self.complete},
        }
        if "receiver" in target:
            rec["receiver"] = target["receiver"]
        self.records.append(rec)

    def _unknown(self, reason: str, dispatch: str = "static") -> Dict[str, Any]:
        return {"target": "unknown", "reason": reason, "dispatch": dispatch, "proven": False}

    def _resolve(self, func: ast.expr) -> Dict[str, Any]:
        if isinstance(func, ast.Name):
            return self._resolve_name(func.id)
        if isinstance(func, ast.Attribute):
            base = func.value
            if isinstance(base, ast.Name):
                return self._resolve_attribute(base.id, func.attr)
            if (isinstance(base, ast.Call) and isinstance(base.func, ast.Name) and base.func.id == "super"
                    and not base.args and not base.keywords):
                return self._resolve_super(func.attr)
            return self._unknown("a method on a receiver whose type is not statically known", "dynamic")
        return self._unknown("the callee is not a name or an attribute", "dynamic")

    # a module-level name -------------------------------------------------

    def _resolve_name(self, name: str) -> Dict[str, Any]:
        local = self._local_binding(name)
        if local is not None:
            return self._unknown(f"{name} is bound in a {local} scope of the call")
        res = self._module_name(name)
        if res["target"] == "module":
            return self._unknown(f"{name} is a module: calling it raises")
        return res

    def _module_name(self, name: str) -> Dict[str, Any]:
        rec = self.consts.get(name)
        bound = self.binder is not None and name in self.binder.bindings
        if rec is None and not bound and not self._star_exports(name):
            if name in _BUILTIN_NAMES:
                proven = self.complete and not self.namespace_writable
                return {"target": "builtin", "name": name, "proven": proven,
                        "reason": "a builtin nothing binds" if proven
                        else "a builtin, but the import graph is incomplete or the module can write its "
                             "namespace (globals(), exec, setattr, builtins)"}
            return self._unknown(f"{name} is not bound at module level")
        if rec is None:
            return self._unknown(f"{name} has no module_constant record")
        proven = bool(rec.get("proven"))
        why = "proven constant" if proven else _why_not(rec)
        origin = rec.get("origin")
        kind, site, name_of = self._chase(self.entry.path, name, 0)
        if kind in ("class", "def"):
            node = site["node"]
            if node.decorator_list:
                return self._unknown(f"{name} is decorated: the name denotes the decorator's result")
            out = {"target": "user_class" if kind == "class" else "user_function", "name": node.name,
                   "def": {"module": site["module"], "line": node.lineno, "col": node.col_offset},
                   "proven": proven, "reason": why}
            if isinstance(node, ast.AsyncFunctionDef):
                return self._unknown(f"{name} is an async def")
            return out
        if kind == "external":
            return {"target": "imported", "name": name_of, "proven": proven, "reason": why}
        if kind == "module":
            return {"target": "module", "name": name_of, "proven": proven, "reason": why}
        return self._unknown(f"{name} is bound by {origin or 'an assignment'}: not a def, class or import")

    def _star_exports(self, name: str) -> bool:
        for star in self.entry.star_from:
            src = self.project.find(star)
            if src is None:
                continue
            binder = self.project.binder(src)
            if binder is not None and name in binder.bindings:
                return True
        return False

    def _chase(self, path: str, name: str, depth: int):
        """``(kind, site, dotted)`` of the binding ``path``'s module-level ``name`` ends up at, following
        ``from X import name`` through the graph: ``class`` / ``def`` (``site`` = {module, node}), ``external``
        (``dotted`` is ``module.name``), ``module`` (an ``import``), or ``other``."""
        mod = self.project.mods.get(path)
        binder = self.project.binder(path)
        if mod is None or binder is None or depth > 8:
            return "other", None, None
        items = binder.bindings.get(name)
        if not items:
            for star in mod.star_from:
                src = self.project.find(star)
                if src is not None and src != path and name in (self.project.binder(src).bindings
                                                                if self.project.binder(src) else {}):
                    return self._chase(src, name, depth + 1)
            return "other", None, None
        first = items[0]
        if first.kind in ("def", "class"):
            node = self._top_level(mod.tree, name, first.line)
            if node is None:
                return "other", None, None
            return first.kind, {"module": self.rel(path), "node": node}, None
        if first.kind == "import":
            return "module", None, first.source[0] if first.source else None
        if first.kind == "importfrom" and first.source is not None:
            module_ish, level, imported, _as = first.source
            base = self.project._resolve_relative(mod, level, module_ish) if level else module_ish
            src = self.project.find(base) if base else None
            if src is not None and src in self.project.mods:
                if imported in self.project.binder(src).bindings or self.project.mods[src].star_from:
                    return self._chase(src, imported, depth + 1)
                return "other", None, None
            if src is None and base is not None and self.project.find(f"{base}.{imported}") is not None:
                return "module", None, f"{base}.{imported}"
            return "external", None, f"{base}.{imported}" if base else imported
        return "other", None, None

    @staticmethod
    def _top_level(tree: ast.Module, name: str, line: int):
        for st in tree.body:
            if isinstance(st, _FUNCS + (ast.ClassDef,)) and st.name == name and st.lineno == line:
                return st
        return None

    # the type of a receiver variable ------------------------------------

    def _receiver_class(self, name: str):
        """``(user_class resolution, line of the binding)`` when ``name`` is bound by exactly one statement of its
        scope, ``name = Cls(...)`` with a proven user class (see the module docstring); else ``(None, why)``."""
        if self.namespace_writable:
            return None, "the module can write its own namespace (globals(), exec, setattr, builtins)"
        found = self._binding_frame(name)
        if found is None:
            return None, f"{name} is bound in a lambda, comprehension or class scope"
        index, frame = found
        if frame.kind == "module":
            rec = self.consts.get(name)
            if rec is None or rec.get("origin") != "assign":
                return None, f"{name} is not a module-level assignment"
            evidence = [r.get("kind") for r in (rec.get("rebinds") or [])]
            if any(kind != "conditional-binding" for kind in evidence):
                return None, f"{name} is rebound or stored into ({', '.join(sorted(set(map(str, evidence))))})"
            if not (rec.get("provenance") or {}).get("complete", True) or not self.complete:
                return None, "the import graph is incomplete"
            if self._star_exports(name):
                return None, f"a star import may bind {name}"
            sites = _binding_sites(frame.node.body, name)
            nested = _nested_declarations(frame.node.body, name, ast.Global)
        else:
            if name in _function_params(frame.node):
                return None, f"{name} is a parameter: its type is the caller's"
            sites = _binding_sites(frame.node.body, name)
            nested = _nested_declarations(frame.node.body, name, ast.Nonlocal)
        if nested:
            return None, f"a nested scope declares {name} {'global' if frame.kind == 'module' else 'nonlocal'}"
        if len(sites) != 1:
            return None, f"{name} has {len(sites)} bindings in its scope, not one"
        st = sites[0]
        if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                and isinstance(st.value, ast.Call) and isinstance(st.value.func, ast.Name)):
            return None, f"{name} is not bound by `{name} = Cls(...)`"
        # the class name is read where the binding runs: in the scope of that statement, not of the call
        saved = self.frames
        self.frames = saved[:index + 1]
        try:
            cls_res = self._resolve_name(st.value.func.id)
        finally:
            self.frames = saved
        if cls_res["target"] != "user_class":
            return None, f"{name} is bound by {st.value.func.id}(...), which is a {cls_res['target']}"
        return cls_res, st.lineno

    # an attribute call ---------------------------------------------------

    def _typed_method(self, base: str, attr: str) -> Optional[Dict[str, Any]]:
        """``base.attr(...)`` through the type of the receiver variable ``base``, or ``None`` when it has none."""
        cls_res, info = self._receiver_class(base)
        if cls_res is None:
            return None
        out = self._method_of(cls_res, attr, f"{base}.{attr}")
        if out["target"] == "user_method":
            out["receiver"] = {"name": base, "class": cls_res["name"], "line": info}
            out["reason"] = (f"a method of {cls_res['name']}, the class {base} is bound to by its one binding "
                             f"(line {info})")
        return out

    def _resolve_attribute(self, base: str, attr: str) -> Dict[str, Any]:
        local = self._local_binding(base)
        if local is not None:
            typed = self._typed_method(base, attr) if local == "function" else None
            if typed is not None:
                return typed
            return self._unknown(f"{base} is bound in a {local} scope of the call: a method on a value",
                                 "dynamic")
        res = self._module_name(base)
        if res["target"] == "user_class":
            return self._method_of(res, attr, f"{base}.{attr}")
        if res["target"] == "module":
            return {"target": "imported", "name": f"{res['name']}.{attr}" if res["name"] else attr,
                    "proven": res["proven"], "reason": "an attribute of an imported module"}
        if res["target"] == "imported":
            return {"target": "imported", "name": f"{res['name']}.{attr}", "proven": res["proven"],
                    "reason": "an attribute of an imported name"}
        typed = self._typed_method(base, attr)
        if typed is not None:
            return typed
        return self._unknown(f"{base}.{attr}: a method on a receiver whose type is not statically known",
                             "dynamic")

    def _method_of(self, cls_res: Dict[str, Any], attr: str, text: str) -> Dict[str, Any]:
        """``Cls.attr`` for a resolved user class: the ``def`` of ``attr`` along the single-inheritance chain."""
        site = cls_res["def"]
        mod_path = self._path_of(site["module"])
        mod = self.project.mods.get(mod_path) if mod_path else None
        if mod is None:
            return self._unknown(f"{text}: the defining module of {cls_res['name']} is not read")
        cls = self._top_level(mod.tree, cls_res["name"], site["line"])
        proven = cls_res["proven"]
        seen = 0
        while cls is not None and seen < 16:
            seen += 1
            found = _class_member(cls, attr)
            if found is _AMBIGUOUS:
                return self._unknown(f"{text}: {cls.name} binds {attr} more than once or not by a def")
            if found is not None:
                decos = [ast.unparse(d) for d in found.decorator_list]
                if any(d not in ("staticmethod", "classmethod") for d in decos):
                    return self._unknown(f"{text}: {attr} is decorated")
                if isinstance(found, ast.AsyncFunctionDef):
                    return self._unknown(f"{text}: {attr} is an async def")
                return {"target": "user_method", "name": f"{cls.name}.{attr}", "proven": proven,
                        "def": {"module": self.rel(mod.path), "line": found.lineno, "col": found.col_offset},
                        "reason": "a method found on the single-inheritance chain of a user class"}
            bases = [b for b in cls.bases if not (isinstance(b, ast.Name) and b.id == "object")]
            if not bases:
                return self._unknown(f"{text}: {attr} is not defined by the class chain")
            if len(bases) != 1 or not isinstance(bases[0], ast.Name) or cls.keywords:
                return self._unknown(f"{text}: {cls.name} has several bases, a metaclass or a base that is "
                                     f"not a name")
            base = self._module_name_in(mod_path, bases[0].id)
            if base is None:
                return self._unknown(f"{text}: the base {bases[0].id} of {cls.name} is not a user class")
            bmod, bcls, bproven = base
            proven = proven and bproven
            mod, cls = bmod, bcls
            mod_path = mod.path
        return self._unknown(f"{text}: the class chain is too deep")

    def _module_name_in(self, mod_path: str, name: str):
        """``(module, ClassDef, proven)`` of ``name`` in the module at ``mod_path`` when it is a user class.
        Proven only for the entry module (the only one ``constants`` covers)."""
        kind, site, _dotted = self._chase(mod_path, name, 0)
        if kind != "class":
            return None
        node = site["node"]
        defining = self._path_of(site["module"])
        mod = self.project.mods.get(defining) if defining else None
        if mod is None:
            return None
        proven = (mod_path == self.entry.path and defining == self.entry.path
                  and bool((self.consts.get(name) or {}).get("proven")))
        return mod, node, proven

    def _path_of(self, rel_module: str) -> Optional[str]:
        for path in self.project.mods:
            if self.rel(path) == rel_module:
                return path
        return None

    def _resolve_super(self, attr: str) -> Dict[str, Any]:
        """Zero-argument ``super().attr(...)`` directly in a method of a user class."""
        if self._local_binding("super") is not None or (
                self.binder is not None and "super" in self.binder.bindings):
            return self._unknown("super is bound by the program", "dynamic")
        frames = self.frames
        if len(frames) < 3 or frames[-1].kind != "function" or frames[-2].kind != "class":
            return self._unknown("super() outside a method of a class", "dynamic")
        cls = frames[-2].node
        if len(frames) != 3 or frames[-3].kind != "module":
            return self._unknown("super() in a nested class", "dynamic")
        bases = [b for b in cls.bases if not (isinstance(b, ast.Name) and b.id == "object")]
        if len(bases) != 1 or not isinstance(bases[0], ast.Name) or cls.keywords:
            return self._unknown("super() in a class without exactly one user base", "dynamic")
        res = self._resolve_name(bases[0].id)
        if res["target"] != "user_class":
            return self._unknown(f"super() base {bases[0].id} is not a user class", "dynamic")
        out = self._method_of(res, attr, f"super().{attr}")
        return out


_AMBIGUOUS = object()


def _class_member(cls: ast.ClassDef, name: str):
    """The ``def`` that binds ``name`` in ``cls``'s own body (``None`` when the body does not bind it,
    ``_AMBIGUOUS`` when it binds it more than once or other than by a ``def``)."""
    hits = []
    stack: List[ast.AST] = list(cls.body)
    while stack:
        n = stack.pop()
        if isinstance(n, _FUNCS + (ast.ClassDef,)):
            if n.name == name:
                hits.append(n)
            continue
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and n.id == name:
            hits.append(n)
        elif isinstance(n, (ast.Import, ast.ImportFrom)) and any(
                (a.asname or a.name.split(".")[0]) == name for a in n.names):
            hits.append(n)
        stack.extend(ast.iter_child_nodes(n))
    if not hits:
        return None
    if len(hits) != 1 or not isinstance(hits[0], _FUNCS):
        return _AMBIGUOUS
    return hits[0]


def _why_not(rec: Dict[str, Any]) -> str:
    rebinds = rec.get("rebinds") or []
    if rebinds:
        first = rebinds[0]
        extra = f" (+{len(rebinds) - 1} more)" if len(rebinds) > 1 else ""
        return f"not proven: {first.get('kind')} at line {first.get('line')}{extra}"
    if not (rec.get("provenance") or {}).get("complete", True):
        return "not proven: the import graph is incomplete"
    return "not proven"


# ---------------------------------------------------------------------------
# the consumers' door
# ---------------------------------------------------------------------------

def call_target_index(records: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """``{(scope, callee): record}`` -- the lookup every consumer uses.

    A key whose records disagree on ``(target, name, proven)`` (two sites of one scope that spell the callee
    alike and resolve differently) maps to an ``unknown`` record: a consumer must not pick one."""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for rec in records:
        key = (rec["scope"], rec["callee"])
        have = out.get(key)
        if have is None:
            out[key] = rec
        elif (have["target"], have.get("name"), have["proven"]) != (rec["target"], rec.get("name"), rec["proven"]):
            out[key] = {"kind": "call_target", "scope": key[0], "callee": key[1], "target": "unknown",
                        "name": None, "def": None, "dispatch": "dynamic", "proven": False,
                        "reason": "two call sites of this scope spell the callee alike and resolve differently"}
    return out
