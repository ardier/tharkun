"""``Resolver`` -- resolve every name, attribute and import of a program against ALLOWLISTS ([internal ref]).

SOLVING, by ``docs/LAYERS.md``'s rule: the input is a program's AST, the output
is a claim about it computed without running it.  The claim is RESOLUTION:
every attribute the program touches and every module it imports is one the
analyses built on top know, so a table keyed by name (which calls may raise
``ZeroDivisionError``; which are deterministic) sees EVERYTHING the program
can do.  ``solve/census.py`` (``proof:no-site``) and ``solve/determinism.py``
(the determinism certificate) both stand on it.

What does not resolve is TOP -- the analyses refuse -- and a first cut of it
is reflection, which could reach any callable behind a table's back:

* the builtins ``eval``, ``exec``, ``compile``, ``__import__``, ``globals``,
  ``locals``, ``vars``, ``breakpoint`` and the name ``__builtins__``;
* ``getattr`` / ``setattr`` / ``delattr`` / ``hasattr`` unless the name is a
  string literal (then that name is resolved like an attribute);
* any DUNDER attribute (``__getattribute__``, ``__dict__``, ``__class__``,
  ``__mro__``, ``__floordiv__``, ...) except :data:`DUNDER_OK`;
* an attribute name that neither the program binds (``x.name = ...``, a class
  body, ``__slots__``, a constant ``setattr``) nor any allowlisted type or module
  has (:data:`VALUE_TYPES`, the classes of the allowed modules);
* a module used as a value, a module attribute that is private, missing or
  stored to; a relative or star import, an import outside the caller's allowlist.

Only unreferenced plain definitions directly in module/function bodies may be
dead. Class methods remain live unless a closed-world callback analysis proves
an ordinary unreferenced method unreachable. Dunders always remain live.
Callers use :meth:`Resolver.live`. Definition-time expressions still execute.
"""

from __future__ import annotations

import ast
import builtins
import os
import sys
import types
from dataclasses import dataclass

from tharkun.solve import extern

#: attribute dunders that reach nothing hidden: ``Base.__init__(self)`` runs program code
DUNDER_OK = frozenset({"__init__", "__name__", "__qualname__", "__doc__", "__module__"})

REFLECTIVE = frozenset({"eval", "exec", "compile", "__import__", "globals", "locals", "vars",
                        "breakpoint", "__builtins__"})
_ATTR_FUNCS = frozenset({"getattr", "setattr", "delattr", "hasattr"})

#: modules that resolve with ONLY the attributes listed ([internal ref]): ``types.ModuleType`` is a class whose instances print as ``<module 'x'>``
#: (no address), so ``isinstance(__builtins__, types.ModuleType)`` is a deterministic read; the rest of ``types`` (``FunctionType``,
#: ``MappingProxyType``, ``new_class``, ``coroutine``, ...) builds objects or runs code the tables here do not model
RESTRICTED_MODULE_ATTRS = {"types": frozenset({"ModuleType"})}

# CPython 3.14.4's heapq.__all__; keep star-import resolution pinned to the
# certified interpreter surface rather than following a different host version.
_HEAPQ_STAR_EXPORTS = (
    "heappush", "heappop", "heapify", "heapreplace", "heappushpop",
    "heappush_max", "heappop_max", "heapify_max", "heapreplace_max",
    "heappushpop_max", "nlargest", "nsmallest", "merge",
)

#: the value types whose (non-dunder) attributes are known
VALUE_TYPES = (int, float, complex, bool, str, bytes, bytearray, list, tuple, dict, set,
               frozenset, range, slice, type(None), BaseException)

#: comprehensions/genexps open their own implicit-function scope (PEP 289): a ``for`` target
#: or any name bound inside one is invisible outside it, and the reverse -- an outer name is
#: invisible inside one if the comprehension binds the same name itself. Only the ITERABLE of
#: the first ``for`` clause is an exception: it is evaluated eagerly in the enclosing scope,
#: before the comprehension's own frame exists, so it cannot see the comprehension's targets.
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _dunder(name: str) -> bool:
    return len(name) > 4 and name.startswith("__") and name.endswith("__")


def _public(obj) -> set[str]:
    return {n for n in dir(obj) if not n.startswith("_")}


def _module(name: str):
    """The stdlib module object for an allowlisted name (explicit imports: ``solve/`` uses no
    importlib), or None."""
    if name == "math":
        import math as m
    elif name == "cmath":
        import cmath as m
    elif name == "random":
        import random as m
    elif name == "re":
        import re as m
    elif name == "io":
        import io as m
    elif name == "array":
        import array as m
    elif name == "string":
        import string as m
    elif name == "itertools":
        import itertools as m
    elif name == "functools":
        import functools as m
    elif name == "collections":
        import collections as m
    elif name == "copy":
        import copy as m
    elif name == "heapq":
        import heapq as m
    elif name == "bisect":
        import bisect as m
    elif name == "typing":
        import typing as m
    elif name == "abc":
        import abc as m
    elif name == "dataclasses":
        import dataclasses as m
    elif name == "enum":
        import enum as m
    elif name == "fractions":
        import fractions as m
    elif name == "decimal":
        import decimal as m
    elif name == "numbers":
        import numbers as m
    elif name == "struct":
        import struct as m
    elif name == "textwrap":
        import textwrap as m
    elif name == "keyword":
        import keyword as m
    elif name == "json":
        import json as m
    elif name == "sys":
        import sys as m
    elif name == "os":
        import os as m
    elif name == "os.path":
        import os.path as m
    elif name == "types":
        import types as m
    else:
        # Keep module metadata in sync with the contract system's statically
        # imported native-module table (not its broader receiver-prefix index).
        from tharkun.solve.contracts.rules import native_module
        return native_module(name)
    return m


def type_attrs(modules) -> frozenset:
    """Non-dunder attribute names of the value types and of every class the modules export."""
    out: set[str] = set()
    for t in VALUE_TYPES:
        out |= {n for n in dir(t) if not _dunder(n)}
    for name in modules:
        mod = _module(name)
        if mod is None:
            continue
        for attr in RESTRICTED_MODULE_ATTRS.get(name, _public(mod)):
            obj = getattr(mod, attr, None)
            if isinstance(obj, type):
                out |= {n for n in dir(obj) if not _dunder(n)}
    return frozenset(out)


def module_attrs(name: str) -> frozenset | None:
    if name == "_sv_verifier":
        return frozenset({"nondet_bool", "nondet_int", "nondet_float", "nondet_float_range",
                          "nondet_str", "nondet_list", "nondet_dict", "check_type"})
    mod = _module(name)
    if name in RESTRICTED_MODULE_ATTRS:
        return RESTRICTED_MODULE_ATTRS[name]
    return frozenset(_public(mod)) if mod is not None else None


class Top(Exception):
    """Something the resolver cannot see through; the message says what and where."""


@dataclass
class Imports:
    modules: dict           # local alias -> module name
    names: dict             # local name -> (module name, attribute)


def _declared_sibling_modules(filename: str, files) -> dict[str, tuple[str, ...]]:
    """Index unique flat .py siblings for the primary or a declared sibling."""
    if not filename or filename.startswith("<") or not files:
        return {}
    try:
        primary = os.path.realpath(filename)
        declared_primary = os.path.realpath(os.fspath(files[0]))
    except (TypeError, OSError, ValueError):
        return {}
    directory = os.path.dirname(declared_primary)
    declared_paths = set()
    for item in files[1:]:
        try:
            path = os.path.realpath(os.fspath(item))
        except (TypeError, OSError, ValueError):
            continue
        if os.path.dirname(path) == directory and path.endswith(".py") and os.path.isfile(path):
            declared_paths.add(path)
    if primary != declared_primary and primary not in declared_paths:
        return {}
    candidates: dict[str, list[str]] = {}
    for item in files[1:]:
        try:
            path = os.path.realpath(os.fspath(item))
        except (TypeError, OSError, ValueError):
            continue
        if os.path.dirname(path) != directory or not path.endswith(".py"):
            continue
        name = os.path.basename(path)[:-3]
        if name == "__init__" or not name.isidentifier() or not os.path.isfile(path):
            continue
        candidates.setdefault(name, []).append(path)
    return {name: tuple(sorted(set(paths))) for name, paths in candidates.items()}


def _is_declared_sibling_file(filename: str, files) -> bool:
    if not filename or filename.startswith("<") or not files:
        return False
    try:
        primary_dir = os.path.dirname(os.path.realpath(os.fspath(files[0])))
        path = os.path.realpath(filename)
    except (TypeError, OSError, ValueError):
        return False
    return (os.path.dirname(path) == primary_dir and path.endswith(".py")
            and os.path.basename(path) != "__init__.py"
            and any(os.path.realpath(os.fspath(item)) == path for item in files[1:]))



class Resolver:
    """Resolution facts about one program (see the module docstring).

    Read-and-recurse ([internal ref], ``tharkun.solve.extern``): an import this Resolver cannot see
    through via ``allowed_modules`` alone is not an automatic refusal any more -- if it names
    code found via a directory THE PROGRAM ITSELF added to ``sys.path`` (a literal, static
    ``sys.path.insert(0, D)`` in the top-level program's own AST -- ``extern.root_snapshots``),
    a CHILD ``Resolver`` is built for that file (cached by resolved path, cycle-guarded) and
    its exposed names are folded into ``imports``/``known_attrs`` exactly like a stdlib
    allowlist entry. ``self.externs`` (local alias -> child ``Resolver``) is how a caller
    (``census.py``, ``determinism.py``) reaches those child trees to walk their OWN sites and
    effects too (condition (d): imported code is program code). See ``extern.py``'s module
    docstring for the full soundness argument and conditions (a)-(e).
    """

    def __init__(self, tree: ast.Module, allowed_modules, *, filename: str = "<program>",
                 _cache: dict | None = None, _pkg_dir: str | None = None, _stack: tuple = (),
                 _toplevel_roots: list | None = None, absent_imports=frozenset(),
                 _declared_files=(), _declared_sibling: bool = False):
        if not _stack:
            native_reason = extern.native_import_reason(tree, filename)
            if native_reason:
                raise Top(native_reason)
        self.tree = tree
        self.allowed = frozenset(allowed_modules)
        #: module names a run of the program OBSERVED to be absent (``import X`` raised ``ModuleNotFoundError`` and no finder
        #: on ``sys.meta_path`` had it: ``_child.py::watch_imports``): such an ``import X`` is an environment probe that
        #: raises, the same on every run of the same environment, so it resolves ([internal ref])
        self.absent_imports = frozenset(absent_imports)
        self.filename = filename
        self._declared_files = tuple(_declared_files)
        self._is_declared_sibling = (_declared_sibling or
                                     _is_declared_sibling_file(filename, self._declared_files))
        self._declared_modules = _declared_sibling_modules(filename, self._declared_files)
        self._cache = {} if _cache is None else _cache
        self._pkg_dir = _pkg_dir                 # None: the top-level program itself
        self._stack = _stack                     # resolved paths on the current recursion path
        self.externs: dict[str, "Resolver"] = {}  # local alias -> the child Resolver it came from
        # [internal ref]/MG's PR 86 review: THIS file's own package __init__.py chain (outermost to
        # innermost, non-inert only) -- set once, after construction, by whichever
        # _resolve_dotted call first resolves a dotted import ending at this file. A fact
        # about where this file lives on disk (its own path's package ancestry), not about
        # any one import statement, so it is intrinsic to the Resolver, not per-import.
        # absint/interp.py::_ensure_extern_loaded runs each of these BEFORE this file's own
        # top level, matching real import order -- (d): imported code is program code, and
        # a package's __init__.py runs whether or not any name is read FROM it.
        self.pkg_inits: list["Resolver"] = []
        if _pkg_dir is None:
            self._snapshots, self._final_roots = extern.root_snapshots(tree, filename)
            self._toplevel_roots = self._final_roots
        else:
            self._snapshots = {}
            self._final_roots = []
            self._toplevel_roots = _toplevel_roots or []
        self.parent: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(tree):
            for ch in ast.iter_child_nodes(node):
                self.parent[ch] = node
        self.bindings = self._bindings()
        self.imports = self._imports()
        self.user_attrs = self._user_attrs() | frozenset().union(
            *(sub.user_attrs for sub in self._unique_externs()), frozenset())
        self.known_attrs = type_attrs(self.allowed) | self.user_attrs
        self._mod_attrs = {m: module_attrs(m) for m in set(self.imports.modules.values())}
        self.dead = self._dead_defs()

    def _unique_externs(self):
        seen: set[int] = set()
        out = []
        for sub in self.externs.values():
            if id(sub) not in seen:
                seen.add(id(sub))
                out.append(sub)
        return out

    # -- scopes --------------------------------------------------------------------
    def _in_first_iter(self, node: ast.AST, comp: ast.AST) -> bool:
        """Is ``node`` (at or below ``comp``) inside ``comp``'s first ``for``'s iterable --
        the one piece of a comprehension that runs in the ENCLOSING scope?"""
        first_iter = comp.generators[0].iter
        n = node
        while n is not None and n is not comp:
            if n is first_iter:
                return True
            n = self.parent.get(n)
        return False

    def function_of(self, node: ast.AST) -> ast.AST:
        # A `:=` target binds in the nearest enclosing FUNCTION/module scope even when the
        # walrus sits inside one or more comprehensions (PEP 572) -- it never scopes to them.
        par = self.parent.get(node)
        walrus = isinstance(node, ast.Name) and isinstance(par, ast.NamedExpr) and par.target is node
        child, p = node, self.parent.get(node)
        while p is not None:
            if isinstance(p, COMPREHENSIONS):
                # Everything in a comprehension but its first iterable is the comprehension's
                # OWN scope; a walrus target always escapes it (PEP 572).
                if not walrus and not self._in_first_iter(node, p):
                    return p
            elif isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                body = child is p.body if isinstance(p, ast.Lambda) else child in p.body
                # Parameters bind in the function; decorators/defaults do not.
                # Annotations are lazy in 3.14 and use an enclosing annotation
                # scope, not the function's parameters or body locals.
                if body or isinstance(node, ast.arg):
                    return p
            elif isinstance(p, ast.Module):
                return p
            child, p = p, self.parent.get(p)
        return self.tree

    def _bindings(self) -> dict[str, set[int]]:
        """name -> ids of the scopes (def / lambda / comprehension / module) that bind it.  A
        class body's binding shadows nothing inside its methods, so it is not recorded as a
        scope."""
        out: dict[str, set[int]] = {}

        def owner(node):
            p = self.parent.get(node)
            while p is not None and not isinstance(p, (ast.Module, ast.FunctionDef,
                                                       ast.AsyncFunctionDef, ast.Lambda,
                                                       ast.ClassDef) + COMPREHENSIONS):
                p = self.parent.get(p)
            return p

        for node in ast.walk(self.tree):
            names: list[str] = []
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names = [node.name]
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                names = [node.id]
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [(a.asname or a.name).split(".")[0] for a in node.names]
            elif isinstance(node, ast.ExceptHandler) and node.name:
                names = [node.name]
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
                names = [node.name]
            elif isinstance(node, ast.arg):
                out.setdefault(node.arg, set()).add(id(self.function_of(node)))
                continue
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                for n in node.names:
                    out.setdefault(n, set()).add(id(self.tree))
                continue
            if not names:
                continue
            if isinstance(owner(node), ast.ClassDef):
                continue
            for n in names:
                out.setdefault(n, set()).add(id(self.function_of(node)))
        return out

    def is_builtin(self, node: ast.Name) -> bool:
        """Does this read of a name resolve to the builtins module?"""
        scopes = self.bindings.get(node.id, set())
        f = self.function_of(node)
        while True:
            if id(f) in scopes:
                if isinstance(f, ast.Module) and hasattr(builtins, node.id):
                    # Module stores occur in execution order: before a later
                    # definition/default, this read may still use the builtin.
                    raise Top(f"module shadows builtin {node.id}; binding order is not certified")
                return False
            if isinstance(f, ast.Module):
                return hasattr(builtins, node.id)
            f = self.function_of(f)

    def _imports(self) -> Imports:
        mods: dict = {}
        names: dict = {}
        star_bound: set = set()
        body_ids = {id(s) for s in self.tree.body}
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name in self.allowed:
                        self._refuse_declared_stdlib_collision(a.name, node)
                        mods[a.asname or a.name] = a.name
                        continue
                    sub = self._try_extern_import(node, body_ids, a.name)
                    if sub is None and a.name in self.absent_imports and "." not in a.name:
                        continue                 # observed absent: the statement raises, binds nothing, reads nothing
                    if sub is None:
                        raise Top(f"line {node.lineno}: imports {a.name}")
                    alias = a.asname or a.name
                    mods[alias] = a.name
                    self.externs[alias] = sub
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module in self.allowed:
                    self._refuse_declared_stdlib_collision(node.module, node)
                    attrs = module_attrs(node.module) or frozenset()
                    for a in node.names:
                        if a.name == "*":
                            if node.module != "heapq":
                                raise Top(f"line {node.lineno}: from {node.module} import *")
                            if not set(_HEAPQ_STAR_EXPORTS) <= attrs:
                                raise Top(f"line {node.lineno}: heapq star-import surface is unavailable")
                            for name in _HEAPQ_STAR_EXPORTS:
                                if name in names or name in mods:
                                    raise Top(f"line {node.lineno}: {name} rebound by star import")
                                names[name] = ("heapq", name)
                                star_bound.add(name)
                            continue
                        if a.name.startswith("_") or a.name not in attrs:
                            raise Top(f"line {node.lineno}: from {node.module} import {a.name}")
                        names[a.asname or a.name] = (node.module, a.name)
                    continue
                sub = self._try_extern_from(node, body_ids)
                if sub is None:
                    raise Top(f"line {node.lineno}: imports from {node.module!r}")
                public = sub.public_names()
                for a in node.names:
                    if a.name == "*":
                        for name in public:
                            if name in names or name in mods:
                                raise Top(f"line {node.lineno}: {name} rebound by star import")
                            names[name] = (f"$extern:{sub.filename}", name)
                            self.externs[name] = sub
                            star_bound.add(name)
                        continue
                    if a.name.startswith("_") or a.name not in public:
                        raise Top(f"line {node.lineno}: {node.module!r} has no public {a.name!r}")
                    alias = a.asname or a.name
                    names[alias] = (f"$extern:{sub.filename}", a.name)
                    self.externs[alias] = sub
        for alias in list(mods) + list(names):
            if alias in star_bound:
                # `*` never appears as `alias` in an ordinary `ast.Name`/def/other import, so
                # a nonzero ordinary binding count here means a REAL collision -- e.g. the
                # file also assigns this name itself -- and must still refuse.
                if self.bindings.get(alias, ()):
                    raise Top(f"the star-imported name {alias} is rebound")
                continue
            if len(self.bindings.get(alias, ())) != 1 or self._rebound(alias):
                raise Top(f"the imported name {alias} is rebound")
        return Imports(mods, names)

    # -- read-and-recurse ([internal ref]): conditions (a)-(e) in extern.py's module docstring ---------
    def _import_roots(self, node: ast.AST) -> list[str]:
        """Roots visible to ``node`` for an ABSOLUTE import: this file's own snapshot at
        ``node``'s position for the top-level program (condition (b)/(c) apply to ITS OWN
        ``sys.path.insert`` statements only); for a recursively-loaded file, the top-level
        program's OWN final roots (its inserts have all already run by the time any recursion
        can begin) -- a recursively-loaded file's own ``sys.path`` mutations, if any, are
        never honoured (it is not "the task's own tree" from that file's perspective)."""
        if self._pkg_dir is None:
            return self._snapshots.get(id(node), [])
        return self._toplevel_roots

    def _try_extern_import(self, node: ast.AST, body_ids: set, dotted: str):
        if id(node) not in body_ids:
            return None                      # (b): only a direct top-level statement is eligible
        roots = self._import_roots(node)
        resolved = self._resolve_dotted(dotted, roots, node) if roots else None
        return resolved if resolved is not None else self._resolve_declared_sibling(dotted, node)

    def _try_extern_from(self, node: ast.ImportFrom, body_ids: set):
        if id(node) not in body_ids:
            return None
        if node.level:
            if self._is_declared_sibling:
                return None                  # a declared flat sibling is not a package
            if self._pkg_dir is None or not node.module:
                return None                  # a top-level program's own relative import, or a
                                              # bare `from . import name`: refuse, unchanged
            pkg_dir = extern.package_dir_for(node.level, self.filename)
            if pkg_dir is None:
                raise Top(f"line {node.lineno}: relative import above the package root")
            return self._resolve_dotted(node.module, [pkg_dir], node)
        if not node.module:
            return None
        roots = self._import_roots(node)
        resolved = self._resolve_dotted(node.module, roots, node) if roots else None
        return resolved if resolved is not None else self._resolve_declared_sibling(node.module, node)

    def _refuse_declared_stdlib_collision(self, name: str, node: ast.AST) -> None:
        root = name.split(".", 1)[0]
        if root in self._declared_modules:
            raise Top(f"line {node.lineno}: declared sibling {root!r} collides with an allowed module")

    def _resolve_declared_sibling(self, dotted: str, node: ast.AST):
        if "." in dotted:
            return None
        paths = self._declared_modules.get(dotted)
        if not paths:
            return None
        if dotted in self.allowed or dotted in sys.stdlib_module_names:
            raise Top(f"line {node.lineno}: declared sibling {dotted!r} collides with a stdlib module")
        if len(paths) != 1:
            raise Top(f"line {node.lineno}: declared sibling {dotted!r} is ambiguous")
        try:
            found = extern.find_in_roots(dotted, [os.path.dirname(paths[0])])
        except extern.ExternTop as e:
            raise Top(f"line {node.lineno}: {e}") from e
        if found is None or os.path.realpath(found[0]) != paths[0] or found[1]:
            raise Top(f"line {node.lineno}: declared sibling {dotted!r} is shadowed by a package or other import")
        return self._load(paths[0], node, declared_sibling=True)

    def _resolve_dotted(self, dotted: str, roots: list[str], node: ast.AST):
        shadowed = extern.shadowed_stdlib_name(roots, self.allowed)
        if shadowed is not None:
            raise Top(f"line {node.lineno}: a program root shadows the stdlib name {shadowed!r}")
        try:
            found = extern.find_in_roots(dotted, roots)
        except extern.ExternTop as e:
            raise Top(f"line {node.lineno}: {e}") from e
        if found is None:
            return None
        leaf, inits = found
        init_resolvers = self._check_inits(inits, node)
        sub = self._load(leaf, node)
        if init_resolvers and not sub.pkg_inits:
            # Attach once (``_load`` caches by path, so a second resolution of the same
            # leaf file returns the SAME object -- its own package-init chain is a fact
            # about where it lives on disk, not about which import statement asked).
            sub.pkg_inits = init_resolvers
        return sub

    def _check_inits(self, inits: list[str], node: ast.AST) -> list["Resolver"]:
        """(e): a package import runs every intermediate ``__init__.py``'s top level too. An
        INERT one (``extern.toplevel_inert`` -- imports/defs/literal-assigns only) contributes
        nothing further anything downstream could observe and is skipped, unopened -- this is
        what makes a leaf submodule provable when the package's own ``__init__.py``
        re-exports OTHER siblings this program never actually loads a name from (docs/SV.md
        "Import boundary: the inert package-init shortcut"). A non-inert one must fully
        resolve exactly like any other recursively-read file. Returns the resolved,
        non-inert inits in outermost-to-innermost order (real import order) -- the caller
        attaches them to the leaf Resolver's ``pkg_inits`` so ``absint/interp.py`` executes
        them, in order, before the leaf's own top level runs (``_ensure_extern_loaded``)."""
        resolved: list["Resolver"] = []
        for init_path in inits:
            try:
                src = open(init_path, encoding="utf-8", errors="ignore").read()
                itree = ast.parse(src, init_path)
            except (OSError, SyntaxError, ValueError) as e:
                raise Top(f"line {node.lineno}: package init {init_path} does not parse: {e}")
            if extern.toplevel_inert(itree):
                continue
            sub = self._load(init_path, node)  # must fully resolve
            resolved.append(sub)
            # MG's [internal ref] import-boundary review (PR 86): the child Resolver used to be
            # discarded here once it resolved -- but census.py/determinism.py reach an
            # extern file ONLY through all_resolvers() -> self.externs, so a non-inert
            # __init__.py (a division, an assert, a call, an attribute access -- anything
            # CPython actually runs on import) was silently invisible to both, although
            # real Python always executes it. No ordinary Python identifier binds a
            # package __init__.py directly (nothing ever writes `import pkg.__init__`),
            # so a key no real alias can ever collide with is safe here; this entry
            # exists purely so all_resolvers() reaches the init's own top level, per
            # condition (d). (absint's OWN execution is separate -- see pkg_inits above
            # and interp.py::_ensure_extern_loaded: absint binds names lazily on demand,
            # so an init that no name is ever read FROM needs its own explicit hook.)
            self.externs[f"$init:{init_path}"] = sub
        return resolved

    def _load(self, path: str, node: ast.AST, *, declared_sibling: bool = False) -> "Resolver":
        if path in self._stack:
            raise Top(f"line {node.lineno}: import cycle at {path}")
        cached = self._cache.get(path, "$unset$")
        if cached is False:
            raise Top(f"line {node.lineno}: {path} was already refused")
        if cached != "$unset$":
            return cached
        try:
            src = open(path, encoding="utf-8", errors="ignore").read()
            tree = ast.parse(src, path)
        except (OSError, SyntaxError, ValueError) as e:
            self._cache[path] = False
            raise Top(f"line {node.lineno}: {path} does not parse: {e}")
        try:
            sub = Resolver(tree, self.allowed, filename=path, _cache=self._cache,
                           _pkg_dir=os.path.dirname(path), _stack=self._stack + (path,),
                           _toplevel_roots=self._toplevel_roots, absent_imports=self.absent_imports,
                           _declared_files=self._declared_files,
                           _declared_sibling=declared_sibling)
        except Top:
            self._cache[path] = False
            raise
        self._cache[path] = sub
        return sub

    def public_names(self) -> frozenset:
        return extern.toplevel_public_names(self.tree)

    def all_resolvers(self) -> list["Resolver"]:
        """``self`` plus every uniquely-reached extern child, transitively -- condition (d):
        a site/determinism census must walk each recursively-read file's OWN nodes too, not
        just resolve its imports. Order: this file first, then each child in first-use order
        (stable, not that it matters for soundness)."""
        seen: set[int] = {id(self)}
        out = [self]
        stack = list(self._unique_externs())
        while stack:
            sub = stack.pop(0)
            if id(sub) in seen:
                continue
            seen.add(id(sub))
            out.append(sub)
            stack.extend(sub._unique_externs())
        return out

    def _rebound(self, name: str) -> bool:
        count = 0
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                count += sum(1 for a in node.names if (a.asname or a.name).split(".")[0] == name)
            elif isinstance(node, ast.Name) and node.id == name and not isinstance(node.ctx, ast.Load):
                count += 1
            elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.arg)) and \
                    getattr(node, "name", getattr(node, "arg", None)) == name:
                count += 1
        return count != 1

    def _user_attrs(self) -> frozenset:
        out: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load):
                out.add(node.attr)
            elif isinstance(node, ast.ClassDef):
                for st in node.body:
                    if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        out.add(st.name)
                    for t in ast.walk(st) if isinstance(st, (ast.Assign, ast.AnnAssign)) else ():
                        if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                            out.add(t.id)
                        if isinstance(t, ast.Constant) and isinstance(t.value, str):
                            out.add(t.value)          # __slots__ entries
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and \
                    node.func.id == "setattr" and len(node.args) >= 2 and \
                    isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                out.add(node.args[1].value)
            elif isinstance(node, ast.keyword) and node.arg:
                out.add(node.arg)                     # dataclass-like keyword fields
        return frozenset(n for n in out if not _dunder(n) or n in DUNDER_OK)

    # -- live code ---------------------------------------------------------------
    def _closed_ordinary_hooks(self) -> set[str] | None:
        """Ordinary callbacks in a closed value universe; None means keep all methods.

        Values may be native builtins or plain source-defined objects/functions.
        Every external callable/attribute must preserve that universe. Receivers
        are NOT guessed from an attribute name: an unknown receiver includes every
        admitted native type AND every source class, whose named methods stay live.
        No external classes, subclasses, reflection or unmodelled protocols enter.
        """
        from tharkun.solve.contracts import closed_call_hooks

        try:
            self.check_all()
        except Top:
            return None
        # ** expansion is a language-level ordinary callback. Pattern matching,
        # generators and context managers are deferred rather than guessed.
        hooks = {"keys"}
        native_types = VALUE_TYPES + (object, type, types.FunctionType, types.MethodType,
                                     types.BuiltinFunctionType, types.BuiltinMethodType)
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom,
                                 ast.GeneratorExp, ast.With, ast.AsyncWith, ast.AsyncFor,
                                 ast.Match, ast.Try, ast.TryStar, ast.Raise)):
                return None
            if isinstance(node, ast.ClassDef) and (node.bases or node.keywords or
                                                   node.decorator_list or node.type_params):
                return None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    node.decorator_list or node.returns or node.type_params):
                return None
            if isinstance(node, ast.arg) and node.annotation is not None:
                return None
            if isinstance(node, ast.AnnAssign):
                return None
            names = []
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id == "__name__":
                    continue  # SV module entry supplies a string; source writes are closed
                if node.id.startswith("__"):
                    return None  # runtime-injected globals and implicit closure cells
                if node.id in self.imports.modules:
                    continue  # check_all requires direct, read-only module attributes
                if node.id in self.imports.names:
                    names = [".".join(self.imports.names[node.id])]
                elif self.is_builtin(node):
                    names = [node.id]
                elif node.id not in self.bindings and node.id != "__name__":
                    return None  # no implicit external globals such as __loader__
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                if isinstance(node.value, ast.Name) and node.value.id in self.imports.modules:
                    names = [self.imports.modules[node.value.id] + "." + node.attr]
                else:
                    # Source receivers remain possible; referenced source methods
                    # are retained by _dead_defs regardless of native contracts.
                    names = [t.__name__ + "." + node.attr for t in native_types
                             if hasattr(t, node.attr)]
            for name in names:
                callbacks = closed_call_hooks(name)
                if callbacks is None:
                    return None
                hooks.update(callbacks)
        return hooks

    def _dead_defs(self) -> set[int]:
        # A recursively loaded module may export any of its top-level definitions to
        # another file in the resolver graph. Its local AST cannot establish that an
        # otherwise unreferenced function is dead across that import boundary.
        if self._stack:
            return set()
        referenced: set[str] = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                referenced.add(node.id)
            elif isinstance(node, ast.Attribute):
                referenced.add(node.attr)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                referenced.add(node.value)
        hooks = self._closed_ordinary_hooks()
        if hooks is not None:
            referenced.update(hooks)
        # Unknown external calls keep every class method live. Only the closed
        # subset above can rule out an unreferenced ordinary method callback.
        parents = (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)
        if hooks is not None:
            parents += (ast.ClassDef,)
        return {id(n) for n in ast.walk(self.tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not _dunder(n.name)
                and isinstance(self.parent.get(n), parents)
                and n.name not in referenced and not n.decorator_list}

    def live(self, node: ast.AST) -> bool:
        child, p = None, node
        while p is not None:
            # An unused body is dead; its definition-time expressions still run.
            # Lazy annotations remain conservatively visible, since decorators
            # or library introspection can request them.
            if id(p) in self.dead and child in p.body:
                return False
            child, p = p, self.parent.get(p)
        return True

    # -- the resolution check ---------------------------------------------------------
    def _module_of(self, node: ast.AST) -> str | None:
        """The (possibly dotted) module name expression ``node`` denotes, or ``None`` --
        a plain ``Name`` bound by ``import X`` (or ``import X as Y``), or a ONE-LEVEL
        submodule access off one (``os.path`` off a plain ``import os``) when that dotted
        name is itself in :attr:`allowed` ([internal ref]: this is the one nesting depth ``os.path``
        needs; a program only ever gets here through an already-allowed top-level import,
        so this never widens what a program may import, only how ONE further attribute hop
        off it is validated)."""
        if isinstance(node, ast.Name):
            return self.imports.modules.get(node.id)
        if isinstance(node, ast.Attribute):
            base_mod = self._module_of(node.value)
            if base_mod is not None:
                dotted = f"{base_mod}.{node.attr}"
                if dotted in self.allowed:
                    return dotted
        return None

    def check(self, node: ast.AST) -> None:
        """Raise :class:`Top` if ``node`` itself does not resolve."""
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and getattr(node, "type_params", ())):
            raise Top(f"line {node.lineno}: type-parameter annotation scope is not modelled")
        if isinstance(node, ast.Name):
            if node.id in REFLECTIVE and (node.id == "__builtins__" or self.is_builtin(node)):
                if not self._builtins_type_probe(node):
                    raise Top(f"line {node.lineno}: uses {node.id}")
            if node.id in self.imports.modules:
                par = self.parent.get(node)
                if not (isinstance(par, ast.Attribute) and par.value is node):
                    raise Top(f"line {node.lineno}: module {node.id} used as a value")
            if node.id in _ATTR_FUNCS and self.is_builtin(node):
                call = self.parent.get(node)
                if not (isinstance(call, ast.Call) and call.func is node and len(call.args) >= 2
                        and isinstance(call.args[1], ast.Constant)
                        and isinstance(call.args[1].value, str)):
                    raise Top(f"line {node.lineno}: {node.id} with a computed name")
                self.check_attr_name(call.args[1].value, node.lineno)
        elif isinstance(node, ast.Attribute):
            base = node.value
            base_mod = self._module_of(base)
            if base_mod is not None:
                if (isinstance(base, ast.Name)
                        and base.id in self.imports.modules
                        and base.id in self.externs):
                    attrs = self.externs[base.id].public_names()
                else:
                    # [internal ref]: `module_attrs` is called fresh here (never `self._mod_attrs`, which
                # is only pre-populated for a DIRECTLY imported top-level name) so a ONE-LEVEL
                # submodule chain off an allowed module -- `os.path.NAME` off a plain
                # `import os` -- resolves against `os.path`'s OWN real attributes, not the
                # single-hop `self._mod_attrs["os"]` (which does not, and should not, list
                # `os.path`'s children) and not the coarse, flat `known_attrs` fallback below
                # (permissive only by accident, whenever some UNRELATED allowed type or
                # module happens to share the same attribute name, e.g. `str.join`).
                    attrs = module_attrs(base_mod) or frozenset()
                if not isinstance(node.ctx, ast.Load):
                    raise Top(f"line {node.lineno}: stores to module {base_mod}")
                if node.attr.startswith("_") or node.attr not in attrs:
                    raise Top(f"line {node.lineno}: {base_mod}.{node.attr} is not a public attribute")
                return
            self.check_attr_name(node.attr, node.lineno)

    def _builtins_type_probe(self, node: ast.Name) -> bool:
        """Is this ``__builtins__`` only asked WHAT IT IS ([internal ref]): ``isinstance(__builtins__, T)`` or ``type(__builtins__)`` in the
        top-level program, where it is the ``builtins`` module under ``python program.py`` (the child's ``native_main_module``;
        an imported module sees a dict), with ``isinstance``/``type`` the real builtins and ``__builtins__`` never bound by the
        program.  Anything else (an attribute, a subscript, a call, a store, an argument elsewhere) reaches ``eval`` and the like."""
        if node.id != "__builtins__" or self._pkg_dir is not None or "__builtins__" in self.bindings:
            return False
        call = self.parent.get(node)
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.args and call.args[0] is node
                and not call.keywords and not any(isinstance(a, ast.Starred) for a in call.args)):
            return False
        if call.func.id == "isinstance":
            ok_arity = len(call.args) == 2
        elif call.func.id == "type":
            ok_arity = len(call.args) == 1
        else:
            return False
        return ok_arity and self.is_builtin(call.func)

    def check_attr_name(self, name: str, line) -> None:
        if _dunder(name) and name not in DUNDER_OK:
            raise Top(f"line {line}: reflective attribute {name}")
        if name in DUNDER_OK:
            return                            # a name, a docstring, a module string: values with no hidden callable ([internal ref])
        if name not in self.known_attrs:
            raise Top(f"line {line}: attribute {name} resolves to nothing known")

    def check_all(self) -> None:
        for node in ast.walk(self.tree):
            self.check(node)

    # -- helpers for the analyses ---------------------------------------------------------
    def call_name(self, call: ast.Call) -> str | None:
        """``module.attr`` for a call of an imported module function (either spelling), the
        builtin's name for a builtin, None otherwise."""
        f = call.func
        if isinstance(f, ast.Name):
            if f.id in self.imports.names:
                m, a = self.imports.names[f.id]
                return f"{m}.{a}"
            if self.is_builtin(f):
                return f.id
            return None
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and \
                f.value.id in self.imports.modules:
            return f"{self.imports.modules[f.value.id]}.{f.attr}"
        return None
