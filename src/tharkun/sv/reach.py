"""The task's own code a program reaches, and what it says about the strings that steer it ([internal ref]).

A multi-file SV task (a ``.yml`` with ``required_files``, or a product program over a package the
task ships) runs code that is not in the file the verifier was handed.  The search stages used to see
only that file: the fuzzer instrumented it alone, and its seeds and dictionary came from its own AST,
so a product like ``assert f_orig(s) == f_mut(s)`` over two copies of a library gave the fuzzer no
coverage feedback from the library and no literal the library compares ``s`` with.

This module reads the program's imports and follows each one to a source file of the task, never
executing anything (stdlib only, ``ast`` and ``os``):

* :func:`reached` -- the task-local modules the program imports, transitively (its own directory,
  directories the program adds to ``sys.path`` with a literal ``sys.path.insert``/``append``, and the
  directories of a ``.yml``'s declared files).  The standard library and site-packages are NOT reached:
  an import that resolves to no task file is somebody else's code;
* :func:`called` -- the definitions of those modules the program actually CALLS (and the module-level
  names they read), by a static call graph through import aliases, ``from`` imports and package re-exports:
  a product over a library reaches every module the package imports, but calls only ``is_email`` and what
  it uses; a dictionary of every token of every module would drown the few tokens that matter;
* :func:`literals`, :func:`guard_tokens`, :func:`regex_patterns` -- what that code carries: its string
  constants, the arguments of the string methods and comparisons that guard it (``startswith``,
  ``endswith``, ``split``, ``in``, ``==``, ...), and the patterns it compiles (constant-folded through
  module-level names, ``str.format``, f-strings and ``+``);
* :func:`dictionary` -- the above as a libFuzzer dictionary: the tokens a mutation should splice in.

Everything here is a HINT for a search stage; a candidate built from it is replayed by the engine
before it can mean anything (docs/SV.md "3a": the fresh-child replay is the only FALSE source).
"""

from __future__ import annotations

import ast
import os
import re
import string
from dataclasses import dataclass
from typing import Iterable, Iterator

#: ``str`` methods whose FIRST argument is a literal the receiver is compared with or split on
STR_ARG_METHODS = frozenset({
    "startswith", "endswith", "split", "rsplit", "partition", "rpartition", "find", "rfind", "index",
    "rindex", "count", "replace", "strip", "lstrip", "rstrip", "removeprefix", "removesuffix",
    "__contains__",
})
#: ``re`` functions (and the methods of a compiled pattern) whose first argument is a pattern
_RE_FUNCS = frozenset({"compile", "match", "fullmatch", "search", "sub", "subn", "split", "findall",
                       "finditer"})
_RE_FLAGS = {"A": re.A, "ASCII": re.A, "I": re.I, "IGNORECASE": re.I, "L": re.L, "LOCALE": re.L,
             "M": re.M, "MULTILINE": re.M, "S": re.S, "DOTALL": re.S, "U": re.U, "UNICODE": re.U,
             "X": re.X, "VERBOSE": re.X}
_PRINTABLE = frozenset(string.printable)
MAX_MODULES = 48
MAX_TOKEN = 32


@dataclass(frozen=True)
class Module:
    name: str            # the dotted name it is imported under
    path: str            # its absolute source file
    tree: ast.Module


def _read(path: str) -> ast.Module | None:
    try:
        from tharkun.sv.engine import read_source
        return ast.parse(read_source(path), path)
    except (OSError, SyntaxError, ValueError, UnicodeDecodeError):
        return None


def _literal_path_args(tree: ast.Module, base: str) -> list[str]:
    """Directories the program adds to ``sys.path`` with a literal, module-level ``sys.path.insert(i, "D")``
    or ``sys.path.append("D")`` (a relative ``D`` is read against ``base``)."""
    out = []
    for st in tree.body:
        call = st.value if isinstance(st, ast.Expr) and isinstance(st.value, ast.Call) else None
        if call is None or not isinstance(call.func, ast.Attribute) or call.func.attr not in ("insert", "append"):
            continue
        owner = call.func.value
        if not (isinstance(owner, ast.Attribute) and owner.attr == "path" and isinstance(owner.value, ast.Name)
                and owner.value.id == "sys"):
            continue
        arg = call.args[-1] if call.args else None
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out.append(arg.value if os.path.isabs(arg.value) else os.path.join(base, arg.value))
    return out


def _roots(program: str, tree: ast.Module, task: str | None) -> list[str]:
    base = os.path.dirname(os.path.abspath(program))
    roots = [base] + _literal_path_args(tree, base)
    if task and task.endswith((".yml", ".yaml")):
        try:
            from tharkun.solve.environment import TaskEnvironment
            roots += [os.path.dirname(f) for f in TaskEnvironment.from_task(task).files]
        except (OSError, ValueError):
            pass
    seen, out = set(), []
    for r in roots:
        r = os.path.realpath(r)
        if r not in seen and os.path.isdir(r):
            seen.add(r)
            out.append(r)
    return out


def _find(name: str, roots: list[str]) -> tuple[str, bool] | None:
    """``(source path, is_package)`` of dotted ``name`` under the first root that has it."""
    parts = name.split(".")
    for root in roots:
        base = os.path.join(root, *parts)
        if os.path.isfile(base + ".py"):
            return base + ".py", False
        init = os.path.join(base, "__init__.py")
        if os.path.isfile(init):
            return init, True
    return None


def _imports(tree: ast.Module, package: str) -> Iterator[str]:
    """Every dotted module name ``tree`` may import (any depth: an import inside a function is reached too);
    ``package`` is the dotted package a relative import is read against (``""`` for a top-level module)."""
    parts = package.split(".") if package else []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                up = node.level - 1
                if up > len(parts):
                    continue                                   # beyond the top-level package: not importable
                anchor = parts[:len(parts) - up]
                base = ".".join(anchor + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            if base:
                yield base
                for a in node.names:
                    if a.name != "*":
                        yield base + "." + a.name         # ``from pkg import sub``: ``sub`` may be a module


def reached(program: str, task: str | None = None, tree: ast.Module | None = None) -> list[Module]:
    """The task-local modules ``program`` imports, transitively, in the order they are first met (the
    program's own imports first, a package's ancestors before it).  Never executes anything."""
    if tree is None:
        tree = _read(program)
    if tree is None:
        return []
    roots = _roots(program, tree, task)
    own = os.path.realpath(program)
    out: dict[str, Module] = {}
    pending: list[tuple[ast.Module, str]] = [(tree, "")]
    while pending and len(out) < MAX_MODULES:
        t, package = pending.pop(0)
        for name in _imports(t, package):
            parts = name.split(".")
            for i in range(1, len(parts) + 1):               # importing a.b.c imports a and a.b first
                dotted = ".".join(parts[:i])
                if dotted in out:
                    continue
                hit = _find(dotted, roots)
                if hit is None:
                    break
                path, is_pkg = hit
                if os.path.realpath(path) == own:
                    break
                sub = _read(path)
                if sub is None:
                    break
                out[dotted] = Module(dotted, path, sub)
                pending.append((sub, dotted if is_pkg else dotted.rpartition(".")[0]))
    return list(out.values())


def packages(modules: Iterable) -> list[str]:
    """The top-level package/module names to instrument (``atheris.instrument_imports(include=...)`` works
    on top-level names); ``modules`` are :class:`Module`\\ s or dotted names."""
    seen: dict[str, None] = {}
    for m in modules:
        seen.setdefault((m if isinstance(m, str) else m.name).split(".")[0], None)
    return list(seen)


# --- what the program calls -------------------------------------------------------------------------

@dataclass(frozen=True)
class Unit:
    """A piece of task code the program reaches: a function/class definition it calls, or the module-level
    assignment of a name that code reads (a compiled pattern, the string it is built from)."""
    module: str          # dotted module name ("" for the program's own file)
    node: ast.AST


class _Scope:
    """The names one module binds at the top level: its definitions, its assignments, its imports."""

    def __init__(self, tree: ast.Module, package: str):
        self.defs: dict[str, ast.AST] = {}
        self.assigns: dict[str, ast.AST] = {}
        #: local name -> ("module", dotted) | ("name", dotted module, original name)
        self.imports: dict[str, tuple] = {}
        #: modules this one star-imports (``from .validation import *``): what they bind, it re-exports
        self.stars: list[str] = []
        for st in tree.body:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.defs[st.name] = st
            elif isinstance(st, ast.Assign):
                for t in st.targets:
                    if isinstance(t, ast.Name):
                        self.assigns[t.id] = st
            elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
                self.assigns[st.target.id] = st
        parts = package.split(".") if package else []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    self.imports[a.asname or a.name.split(".")[0]] = ("module", a.name if a.asname else a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                up = node.level - 1 if node.level else 0
                if up > len(parts):
                    continue
                base = ".".join((parts[:len(parts) - up] if node.level else []) + ([node.module] if node.module else []))
                for a in node.names:
                    if a.name != "*":
                        self.imports[a.asname or a.name] = ("name", base, a.name)
                    elif base:
                        self.stars.append(base)


def called(program_tree: ast.Module, modules: list[Module]) -> list[Unit]:
    """The task-local definitions ``program_tree`` calls, transitively, and the module-level assignments of the
    names they read: the program itself is a unit (all of it runs), a module's top-level definition becomes
    one when something reachable names it (a class brings its methods).  Resolution follows ``import a.b as c``,
    ``from a import f``, dotted ``a.b.f`` chains and a package ``__init__`` re-exporting from its submodules;
    anything it cannot resolve (a method called on an object, a name bound at run time) is not followed -- the
    set is a hint for what to point a fuzzer at, so a miss only costs a search opportunity."""
    mods = {m.name: m for m in modules}
    scopes = {m.name: _Scope(m.tree, m.name if m.path.endswith("__init__.py") else m.name.rpartition(".")[0])
              for m in modules}
    top = _Scope(program_tree, "")
    units: dict[tuple, Unit] = {}
    work: list[tuple[str, ast.AST]] = [("", program_tree)]
    units[("", id(program_tree))] = Unit("", program_tree)

    def scope_of(name: str) -> _Scope | None:
        return top if name == "" else scopes.get(name)

    def lookup(module: str, name: str, depth: int = 0):
        """The ``(module, node)`` that binds ``name`` in ``module`` (following re-exports), or a submodule name."""
        sc = scope_of(module)
        if sc is None or depth > 8:
            return None
        if name in sc.defs:
            return module, sc.defs[name]
        if name in sc.assigns:
            return module, sc.assigns[name]
        imp = sc.imports.get(name)
        if imp is not None:
            if imp[0] == "name":
                return lookup(imp[1], imp[2], depth + 1) or (("module", f"{imp[1]}.{imp[2]}") if f"{imp[1]}.{imp[2]}" in mods else None)
            return ("module", imp[1]) if imp[1] in mods else None
        for star in sc.stars:
            hit = lookup(star, name, depth + 1)
            if hit is not None and hit[0] != "module":
                return hit
        sub = f"{module}.{name}" if module else name
        return ("module", sub) if sub in mods else None

    def add(module: str, node: ast.AST) -> None:
        key = (module, id(node))
        if key not in units:
            units[key] = Unit(module, node)
            work.append((module, node))

    def follow(module: str, chain: list[str]) -> None:
        """Resolve the dotted ``chain`` read in ``module`` and enqueue what it names."""
        hit = lookup(module, chain[0])
        i = 1
        while hit is not None:
            if hit[0] == "module":                      # a module: the next part is looked up inside it
                if i >= len(chain):
                    return
                hit = lookup(hit[1], chain[i])
                i += 1
                continue
            add(*hit)
            return

    while work and len(units) < 512:
        module, node = work.pop()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                follow(module, [sub.id])
            elif isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Load):
                chain, cur = [], sub
                while isinstance(cur, ast.Attribute):
                    chain.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    follow(module, [cur.id] + chain[::-1])
    return list(units.values())


def unit_modules(units: Iterable[Unit]) -> list[str]:
    """The dotted names of the task modules the units live in (not the program itself)."""
    seen: dict[str, None] = {}
    for u in units:
        if u.module:
            seen.setdefault(u.module, None)
    return list(seen)


# --- what the code carries -------------------------------------------------------------------------

def _ignored(tree: ast.AST) -> set[int]:
    """Docstrings and annotations: task metadata, never a value the code compares with."""
    skip: set[int] = set()
    for owner in ast.walk(tree):
        if isinstance(owner, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and owner.body:
            first = owner.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                skip.add(id(first.value))
        ann = (owner.annotation if isinstance(owner, (ast.AnnAssign, ast.arg)) else
               owner.returns if isinstance(owner, (ast.FunctionDef, ast.AsyncFunctionDef)) else None)
        if ann is not None:
            skip.update(id(n) for n in ast.walk(ann))
    return skip


def literals(tree: ast.AST, limit: int = 256) -> list[str]:
    """The string constants of ``tree`` (docstrings and annotations aside), shortest first within source
    order, deduplicated."""
    skip = _ignored(tree)
    seen: dict[str, None] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value \
                and id(node) not in skip:
            seen.setdefault(node.value, None)
    return list(seen)[:limit]


def _str_consts(node: ast.AST) -> Iterator[str]:
    """The literal strings an argument stands for: ``'x'`` or a tuple/list/set of them."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        yield node.value
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for e in node.elts:
            yield from _str_consts(e)


def guard_tokens(tree: ast.AST) -> list[str]:
    """The literals the code's guards compare a string with: the first argument of ``startswith``,
    ``endswith``, ``split``, ``partition``, ``find``, ``count``, ``replace``, ``strip``, ... (every
    arity form: a tuple of prefixes counts each prefix), the left side of ``'x' in s``, and a constant
    operand of ``==``/``!=``/``in`` over a non-constant."""
    skip = _ignored(tree)
    out: dict[str, None] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in STR_ARG_METHODS and node.args:
            for s in _str_consts(node.args[0]):
                out.setdefault(s, None)
        elif isinstance(node, ast.Compare):
            for op, right in zip(node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    for s in _str_consts(node.left):
                        out.setdefault(s, None)
                    if isinstance(right, (ast.Tuple, ast.List, ast.Set)):
                        for s in _str_consts(right):
                            out.setdefault(s, None)
                elif isinstance(op, (ast.Eq, ast.NotEq)):
                    if any(isinstance(x, ast.Name) and x.id == "__name__" for x in (node.left, right)):
                        continue                  # ``if __name__ == "__main__"`` guards the script, not a string
                    for side in (node.left, right):
                        if id(side) not in skip:
                            for s in _str_consts(side):
                                out.setdefault(s, None)
    return list(out)


class _Folder:
    """Constant-fold a string expression through module-level names, ``+``, f-strings and
    ``'...'.format(...)``; ``None`` for anything it cannot decide."""

    def __init__(self, tree: ast.Module):
        self.defs: dict[str, ast.expr] = {}
        for st in tree.body:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                self.defs[st.targets[0].id] = st.value
            elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
                self.defs[st.target.id] = st.value
        self._busy: set[str] = set()

    def fold(self, node: ast.expr, depth: int = 0):
        if depth > 12:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self._busy or node.id not in self.defs:
                return None
            self._busy.add(node.id)
            try:
                return self.fold(self.defs[node.id], depth + 1)
            finally:
                self._busy.discard(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a, b = self.fold(node.left, depth + 1), self.fold(node.right, depth + 1)
            return a + b if isinstance(a, str) and isinstance(b, str) else None
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts.append(v.value)
                elif isinstance(v, ast.FormattedValue) and v.conversion == -1 and v.format_spec is None:
                    f = self.fold(v.value, depth + 1)
                    if f is None:
                        return None
                    parts.append(str(f))
                else:
                    return None
            return "".join(parts)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format"
                and not node.keywords and not any(isinstance(a, ast.Starred) for a in node.args)):
            base = self.fold(node.func.value, depth + 1)
            args = [self.fold(a, depth + 1) for a in node.args]
            if isinstance(base, str) and all(a is not None for a in args):
                try:
                    return base.format(*args)
                except (IndexError, KeyError, ValueError):
                    return None
        return None


def _flags(node: ast.expr | None) -> int:
    if node is None:
        return 0
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "re":
        return int(_RE_FLAGS.get(node.attr, 0))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _flags(node.left) | _flags(node.right)
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    return 0


def regex_patterns(tree: ast.AST, folder: "_Folder | None" = None) -> list[tuple[str, int]]:
    """``(pattern, flags)`` for every ``re.compile/match/search/...(PATTERN[, flags])`` in ``tree`` whose pattern
    folds to a constant string, in source order, deduplicated.  ``folder`` resolves module-level names (default:
    those of ``tree`` itself, when it is a module).  A pattern built from anything else is missed (a missed
    pattern costs a search opportunity, never a wrong verdict)."""
    if folder is None:
        folder = _Folder(tree if isinstance(tree, ast.Module) else ast.Module(body=[], type_ignores=[]))
    out: dict[tuple[str, int], None] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in _RE_FUNCS and node.args):
            continue
        if not (isinstance(node.func.value, ast.Name) and node.func.value.id == "re"):
            continue
        pattern = folder.fold(node.args[0])
        if not isinstance(pattern, str):
            continue
        flag_arg = node.args[-1] if node.func.attr == "compile" and len(node.args) > 1 else None
        for kw in node.keywords:
            if kw.arg == "flags":
                flag_arg = kw.value
        out.setdefault((pattern, _flags(flag_arg)), None)
    return list(out)


def regex_tokens(pattern: str, flags: int = 0) -> list[str]:
    """The literal runs and class members of a pattern, as dictionary tokens: ``@``, ``://``, the
    characters a class lists (``!#$%&'*+-/=?^_`{|}~``), the end points of a range."""
    try:
        from re import _constants as c
        from re import _parser as parser
        parsed = parser.parse(pattern, flags)
    except (re.error, ImportError, RecursionError, AttributeError):
        return []
    out: dict[str, None] = {}

    def walk(items):
        run: list[str] = []
        for op, arg in list(items) + [(None, None)]:
            if op is c.LITERAL:
                run.append(chr(arg))
                continue
            if run:
                out.setdefault("".join(run), None)
                run = []
            if op is c.IN:
                for kind, val in arg:
                    if kind is c.LITERAL:
                        out.setdefault(chr(val), None)
                    elif kind is c.RANGE:
                        out.setdefault(chr(val[0]), None)
                        out.setdefault(chr(val[1]), None)
            elif op is c.SUBPATTERN:
                walk(arg[-1])
            elif op is c.BRANCH:
                for branch in arg[1]:
                    walk(branch)
            elif op in (c.MAX_REPEAT, c.MIN_REPEAT):
                walk(arg[2])
            elif op in (c.ASSERT, c.ASSERT_NOT):
                walk(arg[1])

    walk(parsed)
    return list(out)


def _fit(token: str) -> bool:
    return bool(token) and len(token) <= MAX_TOKEN and all(ch in _PRINTABLE for ch in token)


def folders(modules: Iterable[Module], program_tree: ast.Module | None = None) -> dict[str, "_Folder"]:
    """One constant folder per module (``""`` is the program's own file)."""
    out = {m.name: _Folder(m.tree) for m in modules}
    if program_tree is not None:
        out[""] = _Folder(program_tree)
    return out


def unit_patterns(units: Iterable[Unit], fold: dict[str, "_Folder"]) -> list[tuple[str, int]]:
    """The regexes the units compile or match with, folded in their own module's scope."""
    seen: dict[tuple[str, int], None] = {}
    for u in units:
        for pf in regex_patterns(u.node, fold.get(u.module)):
            seen.setdefault(pf, None)
    return list(seen)


def dictionary(units: Iterable[Unit], fold: dict[str, "_Folder"], limit: int = 64) -> list[str]:
    """Tokens for a libFuzzer dictionary, most informative first: what the reached code's guards compare a string
    with, the literal runs of the patterns it compiles, then its other short string constants.  The single
    characters a pattern's classes list are NOT tokens (a byte mutation finds them; they only dilute).  Only ``string.printable`` text (a ``nondet_str`` draw can hold nothing
    else), and few: libFuzzer picks an entry uniformly, so every token that does not matter dilutes the ones
    that do (measured on the m131 target: ~180 tokens never found the kill the 6 guard tokens found in seconds)."""
    units = list(units)
    guards: dict[str, None] = {}
    for u in units:
        for tok in guard_tokens(u.node):
            guards.setdefault(tok, None)
    runs: dict[str, None] = {}
    for pattern, flags in unit_patterns(units, fold):
        for tok in regex_tokens(pattern, flags):
            if len(tok) > 1:
                runs.setdefault(tok, None)
    lits: dict[str, None] = {}
    for u in units:
        if u.module == "" or not isinstance(u.node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue                       # the program's own literals are corpus seeds already; an
                                           # assignment's constants are patterns and format templates
        for lit in literals(u.node):
            lits.setdefault(lit, None)
    ordered: dict[str, None] = {}
    for group in (guards, runs, lits):
        for tok in group:
            ordered.setdefault(tok, None)
    return [tok for tok in ordered if _fit(tok)][:limit]


def render_dictionary(tokens: Iterable[str]) -> str:
    """The libFuzzer dictionary file text: one ``kN="..."`` entry per token, ``\\``, ``"`` and anything
    that is not a plain printable character written as ``\\xHH``."""
    lines = []
    for i, tok in enumerate(tokens):
        body = []
        for ch in tok:
            if ch == "\\":
                body.append("\\\\")
            elif ch == '"':
                body.append('\\"')
            elif 0x20 <= ord(ch) < 0x7f:
                body.append(ch)
            else:
                body.append(f"\\x{ord(ch):02x}")
        lines.append(f'k{i}="{"".join(body)}"')
    return "\n".join(lines) + ("\n" if lines else "")
