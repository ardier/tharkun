"""Dead arms refuted from what the program can and cannot do (owner ruling 1365 style: the engine builds the candidate the arm asks
for and refutes it from a constraint; ``solve_condition`` answers ``infeasible`` with the refutation, the driver votes the same).

Two refutations, both CLOSED-WORLD over the source files given (the file's directory is the package; nothing outside it is assumed
to write a key, mutate a table or raise a program-defined exception):

* ``dict-key-never-written`` -- ``if 'k' in V`` where ``V`` is a local dict (``V = {..}``/``dict()``, or ``V = D[e]`` over
  ``D = defaultdict(dict)``) and the key set that can ever reach it, collected from EVERY occurrence of ``V``/``D`` in the function,
  is closed: only literal-key stores, literal-key reads and literal ``in`` tests.  The arm that needs ``'k'`` in ``V`` is infeasible
  when ``'k'`` is not a written key.  Refused (no refutation) on any other use: a dynamic key, ``.update``/``.setdefault``/``del``,
  ``V``/``D`` passed to a call, returned, iterated, rebound, captured by a nested scope or declared ``global``/``nonlocal``.
* ``callee-never-raises`` -- ``if not rets`` after ``rets = []; for f in T.values(): try: rets.append(f(x)) except E: ...`` where ``T``
  is a module-level dict literal, never mutated anywhere in the package, whose values are package functions.  ``rets`` can only be
  empty when EVERY ``f`` raised ``E`` (or ``T`` is empty); one value that cannot raise ``E`` (an exception-escape summary) and a
  non-empty ``T`` make the arm infeasible.  The summary is conservative: a raise of ``E`` or a subclass, a bare/variable re-raise
  under a handler that covers ``E``, and every call is followed (a package function by simple name over the whole package, a class's
  ``__init__``/``__post_init__``, a builtin or a stdlib module function or a builtin-type method is safe); an unresolved callee
  (a local variable called, a method name nothing defines) or a package dunder/property that may raise ``E`` means "may raise"."""
from __future__ import annotations

import ast
import builtins
import os
import re
from pathlib import Path
from typing import Any

_STDLIB_MODULES = frozenset({"re", "inspect", "itertools", "typing", "enum", "collections", "functools", "dataclasses", "json",
                             "string", "math", "copy", "textwrap", "operator", "abc", "sys", "os", "warnings", "datetime"})
_FUNC_SCOPES = (ast.Lambda, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)
_BUILTIN_METHODS = frozenset(name for t in (str, bytes, list, tuple, dict, set, frozenset, int, float, bool, complex, re.Pattern, re.Match,
                                            type((x for x in ())), type(iter(())), type({}.items()))
                             for name in dir(t) if not name.startswith("_"))      # methods of builtin and re objects: no program class inside
_UNSAFE_BUILTINS = frozenset({"exec", "eval", "compile", "open", "__import__", "input", "breakpoint", "setattr", "delattr"})


def _parents(root: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: parent for parent in ast.walk(root) for child in ast.iter_child_nodes(parent)}


def _innermost_function(tree: ast.Module, lineno: int):
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.lineno <= lineno <= (n.end_lineno or n.lineno)]
    return max(fns, key=lambda n: n.lineno) if fns else None


def _find_if(fn, lineno: int):
    return next((n for n in ast.walk(fn) if isinstance(n, ast.If) and n.lineno == lineno), None)


# ----------------------------------------------------------------------------------------------------------------------
# dict key never written
# ----------------------------------------------------------------------------------------------------------------------

def _binders(fn, name: str) -> list[ast.AST]:
    """Every place that (re)binds *name* inside *fn*, other than a plain Name store (those are returned as the Name nodes)."""
    out: list[ast.AST] = []
    args = fn.args
    for a in args.posonlyargs + args.args + args.kwonlyargs + [x for x in (args.vararg, args.kwarg) if x]:
        if a.arg == name:
            out.append(a)
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.append(n)
        elif isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names:
            out.append(n)
        elif isinstance(n, ast.ExceptHandler) and n.name == name:
            out.append(n)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not fn and n.name == name:
            out.append(n)
        elif isinstance(n, ast.alias) and (n.asname or n.name) == name:
            out.append(n)
    return out


def _literal_key(node: ast.expr):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def dict_key_refutation(tree: ast.Module, fn, node: ast.If, desired: str) -> dict | None:
    """The refutation of the arm of ``if 'k' in V`` / ``if 'k' not in V`` that needs ``'k'`` in ``V``, or ``None``."""
    test = node.test
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], (ast.In, ast.NotIn))
            and isinstance(test.comparators[0], ast.Name) and _literal_key(test.left) is not None):
        return None
    key, vname = _literal_key(test.left), test.comparators[0].id
    needs_key_arm = "then" if isinstance(test.ops[0], ast.In) else "else"
    if desired != needs_key_arm:
        return None
    parents = _parents(fn)
    stores = _binders(fn, vname)
    if len(stores) != 1 or not isinstance(stores[0], ast.Name):
        return None
    assign = parents.get(stores[0])
    if not (isinstance(assign, ast.Assign) and len(assign.targets) == 1 and assign.targets[0] is stores[0]):
        return None
    written: set[str] = set()
    outer: str | None = None
    value = assign.value
    if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):                   # V = D[e]; D = defaultdict(dict)
        outer = value.value.id
        d_stores = _binders(fn, outer)
        if len(d_stores) != 1 or not isinstance(d_stores[0], ast.Name):
            return None
        d_assign = parents.get(d_stores[0])
        if not (isinstance(d_assign, ast.Assign) and _is_defaultdict_dict(d_assign.value)):
            return None
    elif isinstance(value, ast.Dict) and all(k is not None and _literal_key(k) is not None for k in value.keys):
        written.update(_literal_key(k) for k in value.keys)                                       # V = {'a': ..}
    elif isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "dict" and not value.args \
            and all(k.arg for k in value.keywords):
        written.update(k.arg for k in value.keywords)
    else:
        return None
    # every occurrence of V (and D) must be a literal-key store, a literal-key read or a literal `in` test
    for name, is_outer in ((vname, False),) + (((outer, True),) if outer else ()):
        for n in ast.walk(fn):
            if not (isinstance(n, ast.Name) and n.id == name):
                continue
            ok, wrote = _use_is_closed(n, parents, is_outer, assign if not is_outer else None, stores[0])
            if not ok:
                return None
            written.update(wrote)
    for scope in ast.walk(fn):                          # a nested scope could write through a closure: not followed
        if scope is not fn and isinstance(scope, _FUNC_SCOPES):
            names = {x.id for x in ast.walk(scope) if isinstance(x, ast.Name)}
            if names & {vname, outer or vname}:
                return None
    if key in written:
        return None
    constraint = (f"the key {key!r} is never written to {vname}: the only keys that can reach it are {sorted(written)!r} "
                  f"(every occurrence of {vname}{'/' + outer if outer else ''} in the function is a literal-key store or read)")
    return {"kind": "dict-key-never-written", "constraint": constraint, "key": key, "variable": vname,
            "keys_ever_written": sorted(written),
            "candidate": {"key": key, "membership": False, "keys_ever_written": sorted(written)}}


def _is_defaultdict_dict(value: ast.expr) -> bool:
    return (isinstance(value, ast.Call) and len(value.args) == 1 and isinstance(value.args[0], ast.Name) and value.args[0].id == "dict"
            and not value.keywords and ((isinstance(value.func, ast.Name) and value.func.id == "defaultdict")
                                        or (isinstance(value.func, ast.Attribute) and value.func.attr == "defaultdict")))


def _use_is_closed(n: ast.Name, parents, is_outer: bool, alias_assign, alias_store) -> tuple[bool, set[str]]:
    p = parents.get(n)
    if not is_outer:                                                  # V
        if n is alias_store:
            return True, set()
        if isinstance(p, ast.Compare) and len(p.ops) == 1 and isinstance(p.ops[0], (ast.In, ast.NotIn)) \
                and p.comparators[0] is n and _literal_key(p.left) is not None:
            return True, set()
        if isinstance(p, ast.Subscript) and p.value is n and _literal_key(p.slice) is not None:
            if isinstance(p.ctx, ast.Load):
                return True, set()
            if isinstance(p.ctx, ast.Store) and isinstance(parents.get(p), (ast.Assign, ast.AugAssign)):
                return True, {_literal_key(p.slice)}
        return False, set()
    # D: `D = defaultdict(dict)`, `V = D[e]`, `D[e]['k'] = x`, reads `D[e]['k']` / `'k' in D[e]`
    if isinstance(p, ast.Assign) and n in p.targets:
        return True, set()
    if isinstance(p, ast.Subscript) and p.value is n and isinstance(p.ctx, ast.Load):
        gp = parents.get(p)
        if isinstance(gp, ast.Assign) and gp.value is p and gp is alias_assign_of(parents, alias_store):
            return True, set()
        if isinstance(gp, ast.Subscript) and gp.value is p and _literal_key(gp.slice) is not None:
            if isinstance(gp.ctx, ast.Load):
                return True, set()
            if isinstance(gp.ctx, ast.Store) and isinstance(parents.get(gp), (ast.Assign, ast.AugAssign)):
                return True, {_literal_key(gp.slice)}
        if isinstance(gp, ast.Compare) and len(gp.ops) == 1 and isinstance(gp.ops[0], (ast.In, ast.NotIn)) \
                and gp.comparators[0] is p and _literal_key(gp.left) is not None:
            return True, set()
    return False, set()


def alias_assign_of(parents, alias_store):
    return parents.get(alias_store)


# ----------------------------------------------------------------------------------------------------------------------
# callee never raises
# ----------------------------------------------------------------------------------------------------------------------

class _Package:
    """The ``.py`` files next to the source: functions and classes by simple name (class-hierarchy analysis by name)."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.trees: dict[str, ast.Module] = {}
        for path in sorted(directory.glob("*.py")):
            try:
                self.trees[path.stem] = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            except (OSError, SyntaxError):
                continue
        self.funcs: dict[str, list[tuple[str, str | None, ast.AST]]] = {}     # simple name -> (module, class, def)
        self.classes: dict[str, list[ast.ClassDef]] = {}
        self.stdlib_aliases: dict[str, set[str]] = {}                          # module -> names bound to stdlib modules
        for mod, tree in self.trees.items():
            aliases: set[str] = set()
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    aliases.update(a.asname or a.name.split(".")[0] for a in n.names if a.name.split(".")[0] in _STDLIB_MODULES)
                elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module and n.module.split(".")[0] in _STDLIB_MODULES:
                    aliases.update(a.asname or a.name for a in n.names)         # names imported from stdlib are stdlib callables
            self.stdlib_aliases[mod] = aliases
            for n in ast.walk(tree):
                if isinstance(n, ast.ClassDef):
                    self.classes.setdefault(n.name, []).append(n)
                    for item in n.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            self.funcs.setdefault(item.name, []).append((mod, n.name, item))
            for n in tree.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.funcs.setdefault(n.name, []).append((mod, None, n))
            for n in ast.walk(tree):                   # nested defs are reachable by name too
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not any(n is f[2] for fs in self.funcs.values() for f in fs):
                    self.funcs.setdefault(n.name, []).append((mod, None, n))

    def bases_of(self, class_name: str) -> set[str]:
        """The class and every base name reachable through package classes and the builtin exception hierarchy."""
        seen: set[str] = set()
        todo = [class_name]
        while todo:
            name = todo.pop()
            if name in seen:
                continue
            seen.add(name)
            for cd in self.classes.get(name, []):
                todo.extend(b.id if isinstance(b, ast.Name) else b.attr for b in cd.bases if isinstance(b, (ast.Name, ast.Attribute)))
            builtin = getattr(builtins, name, None)
            if isinstance(builtin, type) and issubclass(builtin, BaseException):
                seen.update(c.__name__ for c in builtin.__mro__)
        return seen


class _Escape:
    """May an exception of the family of ``target`` (a class name) escape a function?  ``True`` = may (or unknown).

    Resolution is closed-world and by name: a bare call ``f(..)`` goes to the caller module's module-level ``f`` (else every
    module-level ``f`` of the package), a method call ``x.m(..)`` to the methods ``m`` of the caller's module when it defines any
    (the receiver classes live there), else to every package method ``m``, a module call ``mod.f(..)`` to that module's ``f``."""

    def __init__(self, pkg: _Package, target: str):
        self.pkg, self.target = pkg, target
        #: the exception class is NOT defined by the program (a builtin such as ValueError, or a class a library defines): then any
        #: builtin, standard-library or builtin-method callee may raise it (``int("x")``), and nothing but a tested contract could say
        #: otherwise; for a class the program defines, only program code can raise it (a callback into it is followed).
        self.library_target = target not in pkg.classes
        builtin_target = getattr(builtins, target, None)
        builtin_family = ({n for n in dir(builtins) if isinstance(getattr(builtins, n), type)
                           and issubclass(getattr(builtins, n), BaseException) and issubclass(getattr(builtins, n), builtin_target)}
                          if isinstance(builtin_target, type) and issubclass(builtin_target, BaseException) else set())
        self.family = {target} | builtin_family | {n for n, cds in pkg.classes.items() if target in pkg.bases_of(n)}
        self.ancestors = pkg.bases_of(target)
        self.memo: dict[int, bool] = {}
        self.active: set[int] = set()
        self.mod_of = {id(d): m for defs in pkg.funcs.values() for m, _, d in defs}
        self.sibling_modules = set(pkg.trees)

    def covers(self, handler_types: list[str]) -> bool:
        """Does a handler for these class names catch a ``target``-family exception?"""
        return any(t in self.ancestors or t in self.family for t in handler_types)

    def function(self, fn: ast.AST) -> bool:
        key = id(fn)
        if key in self.memo:
            return self.memo[key]
        if key in self.active:
            return False                    # a cycle adds nothing new to a least fixpoint
        if self.library_target and not _trivial_function(fn):
            # a builtin / library exception can come from ANY operation (a subscript, an operator, a conversion, a call): without a
            # tested contract for each of them only a function that does nothing but return a name or a constant cannot raise it
            self.memo[key] = True
            return True
        self.active.add(key)
        try:
            result = self.stmts(fn.body, None, self.mod_of.get(key, ""))
        finally:
            self.active.discard(key)
        self.memo[key] = result
        return result

    def stmts(self, body: list[ast.stmt], handler, mod: str) -> bool:
        return any(self.stmt(s, handler, mod) for s in body)

    def stmt(self, s: ast.stmt, handler, mod: str) -> bool:
        if isinstance(s, ast.Raise):
            return self.raise_(s, handler) or any(self.expr(c, mod) for c in ast.iter_child_nodes(s))
        if isinstance(s, ast.Try):
            body = self.stmts(s.body, handler, mod)
            caught = any(self.covers(self._types(h)) for h in s.handlers)
            escapes = body and not caught
            for h in s.handlers:
                escapes = escapes or self.stmts(h.body, self._types(h), mod)
            return escapes or self.stmts(s.orelse, handler, mod) or self.stmts(s.finalbody, handler, mod)
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return False                    # a definition runs nothing; its body is followed where it is called
        if isinstance(s, ast.Match):
            return True
        header = [c for c in ast.iter_child_nodes(s) if not isinstance(c, ast.stmt)]
        inner = [x for f in ("body", "orelse") for x in (getattr(s, f, None) if isinstance(getattr(s, f, None), list) else [])]
        return any(self.expr(e, mod) for e in header) or self.stmts(inner, handler, mod)

    def _types(self, h: ast.ExceptHandler) -> list[str]:
        if h.type is None:
            return ["BaseException"]
        nodes = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
        return [n.id if isinstance(n, ast.Name) else n.attr if isinstance(n, ast.Attribute) else "?" for n in nodes]

    def raise_(self, s: ast.Raise, handler) -> bool:
        exc = s.exc
        if exc is None or (isinstance(exc, ast.Name) and not self._is_class(exc.id)):      # bare `raise` / `raise e`
            if handler is None:
                return True                 # outside any handler we cannot tell what is re-raised
            return self.covers(handler) or "?" in handler
        name = exc.func.id if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) else \
            exc.id if isinstance(exc, ast.Name) else exc.func.attr if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Attribute) else None
        if name is None or not self._is_class(name):
            return True
        return name in self.family

    def _is_class(self, name: str) -> bool:
        return name in self.pkg.classes or isinstance(getattr(builtins, name, None), type)

    def expr(self, e: ast.AST, mod: str) -> bool:
        called = {id(c.func) for c in ast.walk(e) if isinstance(c, ast.Call)}
        for n in ast.walk(e):
            if isinstance(n, ast.Call) and self.call(n, mod):
                return True
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and id(n) not in called:
                for m, cls, d in self.pkg.funcs.get(n.id, []):          # a package function used as a value may be called later
                    if cls is None and self.function(d):
                        return True
        return False

    def _any(self, defs) -> bool:
        return any(self.function(d) for d in defs)

    def call(self, c: ast.Call, mod: str) -> bool:
        f = c.func
        if isinstance(f, ast.Name):
            n = f.id
            if n in self.pkg.classes:
                return self.init(n)
            if n in self.pkg.funcs:
                module_level = [(m, d) for m, cls, d in self.pkg.funcs[n] if cls is None]
                same = [d for m, d in module_level if m == mod]
                if same or module_level:
                    return self._any(same or [d for _, d in module_level])
            if isinstance(getattr(builtins, n, None), type):
                return self.library_target                                          # builtin type / exception construction
            if callable(getattr(builtins, n, None)) and n not in _UNSAFE_BUILTINS:
                return self.library_target
            return n not in self._stdlib_names(mod) or self.library_target
        if isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Name):
                if f.value.id in self._stdlib_names(mod):
                    return self.library_target   # re.compile / inspect.cleandoc: stdlib cannot raise a PROGRAM class (callbacks are followed)
                if f.value.id in self.sibling_modules or f.value.id in _module_aliases(self.pkg, mod):
                    target = _module_aliases(self.pkg, mod).get(f.value.id, f.value.id)
                    defs = [d for m, cls, d in self.pkg.funcs.get(f.attr, []) if m == target and cls is None]
                    return self._any(defs) if defs else True
            if f.attr in self.pkg.classes:
                return self.init(f.attr)
            methods = [(m, d) for m, cls, d in self.pkg.funcs.get(f.attr, []) if cls is not None]
            if methods:
                same = [d for m, d in methods if m == mod]
                return self._any(same or [d for _, d in methods])
            return f.attr not in _BUILTIN_METHODS or self.library_target
        return True                         # called result, subscript, lambda variable: unknown callee

    def _stdlib_names(self, mod: str) -> set[str]:
        return self.pkg.stdlib_aliases.get(mod, set())

    def init(self, class_name: str) -> bool:
        for base in self.pkg.bases_of(class_name):
            for cd in self.pkg.classes.get(base, []):
                for item in cd.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name in ("__init__", "__post_init__", "__new__"):
                        if self.function(item):
                            return True
        return False


def _module_aliases(pkg: _Package, mod: str) -> dict[str, str]:
    """names bound in *mod* to sibling modules (``from . import a``, ``from pkg import a``, ``import pkg.a as a``)."""
    out: dict[str, str] = {}
    tree = pkg.trees.get(mod)
    if tree is None:
        return out
    for n in tree.body:
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name in pkg.trees and ((n.level == 1 and not n.module) or n.module == pkg.dir.name):
                    out[a.asname or a.name] = a.name
        elif isinstance(n, ast.Import):
            for a in n.names:
                parts = a.name.split(".")
                if len(parts) == 2 and parts[0] == pkg.dir.name and parts[1] in pkg.trees:
                    out[a.asname or parts[1]] = parts[1]
    return out


def _trivial_function(fn) -> bool:
    """A function whose body is only ``pass``, a docstring and ``return`` of a name, a constant or a display of them: it performs no
    operation that could raise a builtin or library exception (a name load can only raise NameError-family, never the classes asked)."""
    def safe(e) -> bool:
        if e is None or isinstance(e, ast.Constant):
            return True
        if isinstance(e, ast.Name):
            return isinstance(e.ctx, ast.Load)
        if isinstance(e, (ast.Tuple, ast.List)):
            return all(not isinstance(x, ast.Starred) and safe(x) for x in e.elts)
        return False
    for st in fn.body:
        if isinstance(st, ast.Pass) or (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)):
            continue
        if isinstance(st, ast.Return) and safe(st.value):
            continue
        return False
    return not fn.decorator_list


def _dunders_and_properties_safe(esc: _Escape) -> bool:
    """Implicit calls (operators, ``with``, iteration, attribute access) reach package dunders and properties: refuse the summary
    when any of them may raise the target."""
    for name, defs in esc.pkg.funcs.items():
        for _, cls, fn in defs:
            is_dunder = name.startswith("__") and name.endswith("__") and name not in ("__init__", "__post_init__", "__new__")
            is_property = any(isinstance(d, ast.Name) and d.id == "property" or isinstance(d, ast.Attribute) and d.attr in ("getter", "cached_property")
                              for d in getattr(fn, "decorator_list", []))
            if (is_dunder or is_property) and esc.function(fn):
                return False
    return True


def _resolve_table(pkg: _Package, mod: str, name: str):
    """The module-level dict literal named *name* visible in *mod* (defined there or imported from a package module)."""
    tree = pkg.trees[mod]
    for n in tree.body:
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            if any(isinstance(t, ast.Name) and t.id == name for t in targets) and isinstance(n.value, ast.Dict):
                return mod, n.value
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module:
            for a in n.names:
                if (a.asname or a.name) == name:
                    target = n.module.split(".")[-1] if n.level == 0 and n.module.split(".")[0] == pkg.dir.name else \
                        n.module if n.level == 1 else None
                    if target in pkg.trees and target != mod:
                        return _resolve_table(pkg, target, a.name)
    return None


def _table_is_closed(pkg: _Package, name: str) -> bool:
    """The table is only read (``.values()/.keys()/.items()``, ``T[k]``, ``in``), never written, passed or rebound, anywhere in the package."""
    for mod, tree in pkg.trees.items():
        parents = _parents(tree)
        assigns = 0
        for n in ast.walk(tree):
            if isinstance(n, ast.Name) and n.id == name:
                p = parents.get(n)
                if isinstance(n.ctx, ast.Store):
                    assigns += 1
                    if not isinstance(p, (ast.Assign, ast.AnnAssign)):
                        return False
                elif isinstance(p, ast.Attribute) and p.value is n and p.attr in ("values", "keys", "items", "get", "copy"):
                    pp = parents.get(p)
                    if not (isinstance(pp, ast.Call) and pp.func is p):
                        return False
                elif isinstance(p, ast.Subscript) and p.value is n and isinstance(p.ctx, ast.Load):
                    continue
                elif isinstance(p, ast.Compare) and any(isinstance(o, (ast.In, ast.NotIn)) for o in p.ops):
                    continue
                else:
                    return False
            elif isinstance(n, (ast.Global, ast.Nonlocal)) and name in n.names:
                return False
            elif isinstance(n, ast.alias) and n.asname == name:
                return False
        if assigns > 1:
            return False
    return True


def callee_raise_refutation(tree: ast.Module, rp: str, fn, node: ast.If, desired: str) -> dict | None:
    """Refute ``if not rets:`` (then) / ``if rets:`` (else) for the accumulate-or-catch loop described in the module docstring."""
    test = node.test
    negated = isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
    inner = test.operand if negated else test
    if not isinstance(inner, ast.Name):
        return None
    if desired != ("then" if negated else "else"):
        return None                                    # the arm that needs `rets` empty
    rets = inner.id
    parents = _parents(fn)
    binders = _binders(fn, rets)
    if len(binders) != 1 or not isinstance(binders[0], ast.Name):
        return None
    init = parents.get(binders[0])
    if not (isinstance(init, ast.Assign) and isinstance(init.value, ast.List) and not init.value.elts):
        return None
    appends = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Name) and n.id == rets and n is not binders[0] and n.lineno < node.lineno and n is not inner:
            p = parents.get(n)
            if isinstance(p, ast.Attribute) and p.attr == "append" and p.value is n and isinstance(parents.get(p), ast.Call):
                appends.append(parents[p])
            else:
                return None                            # aliased, passed, iterated, extended before the guard: not closed
    if len(appends) != 1 or len(appends[0].args) != 1 or appends[0].keywords:
        return None
    append = appends[0]
    # the append sits in a try body inside `for v in T.values()`, and its argument calls v
    try_node = next((a for a in _ancestors(append, parents) if isinstance(a, ast.Try)), None)
    loop = next((a for a in _ancestors(append, parents) if isinstance(a, ast.For)), None)
    if try_node is None or loop is None or not isinstance(loop.target, ast.Name) or loop.orelse:
        return None
    if not any(x is append for b in try_node.body for x in ast.walk(b)):           # the append is in the try BODY, not a handler
        return None
    if loop not in _ancestors(try_node, parents):
        return None
    call = append.args[0]
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == loop.target.id):
        return None
    if not try_node.handlers or any(h.type is None for h in try_node.handlers):
        return None
    handled = [t for h in try_node.handlers for t in ([n.id for n in (h.type.elts if isinstance(h.type, ast.Tuple) else [h.type])
                                                       if isinstance(n, ast.Name)])]
    if len(handled) != 1:
        return None
    target = handled[0]
    it = loop.iter
    if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Attribute) and it.func.attr == "values"
            and isinstance(it.func.value, ast.Name) and not it.args):
        return None
    pkg = _Package(Path(rp).parent)
    mod = Path(rp).stem
    if mod not in pkg.trees:
        return None
    table_name = it.func.value.id
    table = _resolve_table(pkg, mod, table_name)
    if table is None or not table[1].keys or not _table_is_closed(pkg, table_name):
        return None
    defining, literal = table
    names = _module_aliases(pkg, defining)
    esc = _Escape(pkg, target)
    if target not in esc.family or not _dunders_and_properties_safe(esc):
        return None
    summaries = []
    safe = None
    for value in literal.values:
        fdef = _function_of(value, pkg, defining, names)
        if fdef is None:
            return None                                # an entry that is not a package function: the table is not closed
        summaries.append((ast.unparse(value), esc.function(fdef)))
        if safe is None and not summaries[-1][1]:
            safe = ast.unparse(value)
    if safe is None:
        return None
    constraint = (f"{safe} cannot raise {target} (exception-escape summary over the package's call graph), and every other "
                  f"exception leaves the function before the guard; {table_name} is a non-empty module dict that nothing mutates, "
                  f"so the loop over {table_name}.values() appends at least once and {rets!r} is never empty")
    return {"kind": "callee-never-raises", "constraint": constraint, "exception": target, "callee": safe, "table": table_name,
            "summaries": {name: ("may raise" if may else "cannot raise") for name, may in summaries},
            "candidate": {"every_value_raised": False, "callee_that_cannot_raise": safe, "exception": target}}


def _ancestors(node: ast.AST, parents) -> list[ast.AST]:
    out = []
    while node in parents:
        node = parents[node]
        out.append(node)
    return out


def _function_of(value: ast.expr, pkg: _Package, mod: str, names: dict[str, str]):
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) and value.value.id in names:
        target_mod = names[value.value.id]
        for m, cls, fdef in pkg.funcs.get(value.attr, []):
            if m == target_mod and cls is None:
                return fdef
    if isinstance(value, ast.Name):
        for m, cls, fdef in pkg.funcs.get(value.id, []):
            if m == mod and cls is None:
                return fdef
    return None


def refute_arm(tree: ast.Module, rp: str, lineno: int, desired: str) -> dict | None:
    """The refutation (a dict with ``kind``, ``constraint``, ``candidate``...) of the *desired* arm of the ``if`` at *lineno*, or ``None``."""
    fn = _innermost_function(tree, lineno)
    if fn is None:
        return None
    node = _find_if(fn, lineno)
    if node is None:
        return None
    return dict_key_refutation(tree, fn, node, desired) or callee_raise_refutation(tree, rp, fn, node, desired)
