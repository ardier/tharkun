"""The static namespace of a module and of the import graph it runs in (#1132b, [internal ref], [internal ref]).

Two facts about a program's module-level names are asked for by different tools and need the SAME walker:

* ``regex_globals`` -- the compiled regexes the entry module can see, with their pattern, flags and where
  they are bound (python-string-utils builds ``EMAIL_RE`` in ``_regex.py`` as
  ``re.compile(r'^{}$'.format(RAW))`` and ``validation.py`` reaches it with ``from ._regex import *``);
* ``module_constants`` -- for every module-level name of the entry module: is it a CONSTANT, i.e. does
  NOTHING in the import graph rebind it?  A synthesis that bakes ``LIMIT = 50`` into an inverter is wrong the
  moment ``pkg/__init__.py`` does ``_c.LIMIT = 5`` or a function does ``global LIMIT; LIMIT += 1``.

Nothing here imports or executes anything: it reads source with ``ast`` and ``os`` (stdlib only, SOLVE layer).

Evaluation
----------
:func:`evaluate_module` folds a module's namespace the way the interpreter would, through a closed, pure subset:
literals and containers of them, arithmetic / bitwise / ``%``, f-strings and the pure ``str`` methods
(``format``, ``join``, ``replace`` ...), ``re.compile(pattern, flags)`` (checked to compile), ``re.escape``, the
``re`` flag constants and ``string`` constants, and imports of local modules (absolute, or relative level 1+,
including ``from .mod import *`` honouring ``__all__``).  Anything outside the subset makes THAT name unknown (it is
removed from the namespace, so a later read never sees a stale earlier binding); it never makes the whole module
unknown and it never guesses.  Evaluation is bounded (nodes, result size, import depth).

Rebinding
---------
A module-level name is PROVEN constant when all of the following hold over the import graph (the entry module,
the ``__init__`` chain of its package, and every local module those import, transitively):

* it has exactly one binding at module level, executed unconditionally (not under an ``if``/``try``/loop);
* no function or class declares it ``global`` and binds it;
* no module stores into it from outside: ``pkg.mod.NAME = v`` / ``del`` / ``+=``, ``setattr(pkg.mod, 'NAME', v)``,
  ``pkg.mod.__dict__['NAME'] = v``, ``vars(pkg.mod)[...]`` -- resolved through the module's import aliases; a store
  through a base it cannot resolve counts against the name when the target module ESCAPES (a bare reference to it
  is passed around, or ``sys.modules`` / ``importlib`` can hand it out);
* the module does not write its own namespace dynamically (``globals()[k] = v``, ``vars()``, ``exec``, ``eval``);
* a name imported ``from X import Y`` is proven only when ``X.Y`` is (the alias chain is followed to the defining
  module), so an ``__init__`` that runs ``_c.LIMIT = 5`` before ``mod`` does ``from ._c import LIMIT`` is seen;
* a MUTABLE foldable value (list / dict / set; a set literal folds to a frozenset and counts) is not mutated
  through any alias (``NAME.append(...)``, ``NAME.add(...)``, ``NAME[k] = v``, passed to a call, kept as a default
  argument, returned through a conditional or ``or`` expression), and a value that HOLDS a mutable member (``[[0]]``,
  ``([0],)``, ``{'a': [0]}``) is never proven: any read of the name can hand out the inner object, and no use is
  followed that far;
* the graph is COMPLETE: no dynamic import (``importlib``, ``__import__``, ``runpy``, ``pkgutil``), no unreadable or
  unresolvable relative import, it fits the module cap, and it holds no REFLECTION that could write a namespace the
  walker does not follow (``_Project._scan_reflection``): a call of ``exec`` or ``eval``; a reference to ``exec``,
  ``eval``, ``compile``, ``globals``, ``vars``, ``locals``, ``setattr``, ``delattr`` that is not the callee of a call
  (an alias, an argument, a default); ``builtins`` / ``__builtins__`` / ``inspect`` / ``gc`` / ``ctypes`` / ``code``;
  frame access (``sys._getframe``, ``f_globals``, ``__globals__``, ...); a ``__dict__`` or ``vars(x)`` that is
  handed on (assigned, passed, returned) instead of subscripted or called through a dict method.  Otherwise
  nothing is proven.  The reflection check is a conservative scan of names and attributes, not an alias analysis:
  a callable obtained by a computed name (``getattr(sys, 'x' + 'y')``) is the one thing it does not see.

Third-party and standard-library modules are outside the graph; the assumption, stated once, is that they do not
store into the project's own modules.  ``rebinds`` on every record is the EVIDENCE (empty exactly when proven and
complete), so a consumer can see WHY a name is not constant.
"""
from __future__ import annotations

import ast
import operator
import os
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

__all__ = [
    "RegexConst", "ModuleRef", "LocalModule", "evaluate_module", "regex_consts",
    "module_constants", "regex_globals", "MAX_MODULES",
]

MAX_MODULES = 200
_MAX_NODES = 20_000
_MAX_SIZE = 1_000_000
_MAX_DEPTH = 8
_MAX_RECORDS = 500


# ---------------------------------------------------------------------------
# values
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegexConst:
    """``re.compile(pattern, flags)`` with both arguments statically known.

    ``flags`` is the EXPLICIT flags argument (0 when none was given), so ``re.compile(pattern, flags)`` rebuilds
    the object; the implicit ``re.UNICODE`` bit of a str pattern is not folded in (see :attr:`compiled_flags`).
    ``module``/``line``/``col`` say where the defining ``re.compile`` call is; they are provenance, not identity.
    """

    pattern: str
    flags: int = 0
    module: str = field(default="", compare=False)
    line: int = field(default=0, compare=False)
    col: int = field(default=0, compare=False)

    @property
    def compiled_flags(self) -> int:
        return int(re.compile(self.pattern, self.flags).flags)


@dataclass(frozen=True)
class ModuleRef:
    """A stdlib module name the evaluator knows constants of (``re``, ``string``)."""

    name: str


@dataclass(frozen=True)
class LocalModule:
    """A local module's evaluated namespace."""

    path: str
    names: Tuple[Tuple[str, Any], ...]

    def get(self, name: str, default: Any = None) -> Any:
        for k, v in self.names:
            if k == name:
                return v
        return default


class _NoValue(Exception):
    """The expression is outside the evaluable subset."""


_BINOPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv, ast.BitOr: operator.or_, ast.BitAnd: operator.and_,
    ast.BitXor: operator.xor, ast.LShift: operator.lshift, ast.RShift: operator.rshift,
}
_CMPOPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge, ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_, ast.IsNot: operator.is_not,
}
#: pure, total-on-evaluable-input ``str`` methods
_STR_METHODS = frozenset({
    "format", "join", "replace", "strip", "lstrip", "rstrip", "lower", "upper", "title", "capitalize",
    "casefold", "swapcase", "zfill", "ljust", "rjust", "center", "removeprefix", "removesuffix",
    "split", "rsplit", "splitlines", "partition", "rpartition", "expandtabs",
})
_BUILTINS = {"str": str, "int": int, "len": len, "chr": chr, "ord": ord, "repr": repr, "bool": bool,
             "min": min, "max": max, "abs": abs, "tuple": tuple, "list": list}
_RE_FLAGS = {n: int(getattr(re, n)) for n in ("I", "IGNORECASE", "M", "MULTILINE", "S", "DOTALL", "X", "VERBOSE",
                                             "A", "ASCII", "U", "UNICODE", "L", "LOCALE")}
_STRING_CONSTS = {n: getattr(string, n) for n in (
    "ascii_letters", "ascii_lowercase", "ascii_uppercase", "digits", "hexdigits", "octdigits",
    "punctuation", "whitespace", "printable")}
#: ``frozenset`` is here because a set LITERAL folds to a frozenset (``_Set``): the object the program holds is a
#: mutable ``set`` (``X = {1}; X.add(2)``).  A genuine ``frozenset(...)`` is treated the same: only precision is lost.
_MUTABLE_TYPES = (list, dict, set, frozenset, bytearray)


def _holds_mutable(value: Any) -> bool:
    """A folded container with a mutable member at any depth (``[[0]]``, ``([0],)``, ``{'a': [0]}``): a read of the
    name can hand out the inner object, which is then mutated where the walker cannot follow it."""
    if isinstance(value, dict):
        members: Iterable[Any] = value.values()
    elif isinstance(value, (tuple, list, set, frozenset)):
        members = value
    else:
        return False
    return any(isinstance(m, _MUTABLE_TYPES) or _holds_mutable(m) for m in members)


def _check_size(v: Any) -> Any:
    if isinstance(v, (str, bytes, list, tuple, set, frozenset, dict)) and len(v) > _MAX_SIZE:
        raise _NoValue("result too large")
    return v


class _Evaluator:
    def __init__(self, ns: Dict[str, Any], module: str = ""):
        self.ns = ns
        self.module = module
        self.nodes = 0

    def ev(self, node: ast.AST) -> Any:
        self.nodes += 1
        if self.nodes > _MAX_NODES:
            raise _NoValue("node budget")
        meth = getattr(self, "_" + type(node).__name__, None)
        if meth is None:
            raise _NoValue(type(node).__name__)
        return _check_size(meth(node))

    def _Constant(self, n):  # noqa: N802
        if n.value is Ellipsis:
            raise _NoValue("ellipsis")
        return n.value

    def _Name(self, n):  # noqa: N802
        if n.id in self.ns:
            return self.ns[n.id]
        if n.id in _BUILTINS:
            raise _NoValue("builtin used as a value")
        raise _NoValue(f"unbound {n.id}")

    def _Tuple(self, n):  # noqa: N802
        return tuple(self.ev(e) for e in n.elts)

    def _List(self, n):  # noqa: N802
        return [self.ev(e) for e in n.elts]

    def _Set(self, n):  # noqa: N802
        try:
            return frozenset(self.ev(e) for e in n.elts)
        except TypeError:
            raise _NoValue("unhashable set element") from None

    def _Dict(self, n):  # noqa: N802
        if any(k is None for k in n.keys):
            raise _NoValue("dict unpacking")
        try:
            return {self.ev(k): self.ev(v) for k, v in zip(n.keys, n.values)}
        except TypeError:
            raise _NoValue("unhashable dict key") from None

    def _UnaryOp(self, n):  # noqa: N802
        v = self.ev(n.operand)
        try:
            if isinstance(n.op, ast.USub):
                return -v
            if isinstance(n.op, ast.UAdd):
                return +v
            if isinstance(n.op, ast.Invert):
                return ~v
            if isinstance(n.op, ast.Not):
                return not v
        except Exception as exc:  # noqa: BLE001
            raise _NoValue(type(exc).__name__) from None
        raise _NoValue("unary")

    def _BinOp(self, n):  # noqa: N802
        fn = _BINOPS.get(type(n.op))
        if fn is None:
            raise _NoValue("binop")
        left, right = self.ev(n.left), self.ev(n.right)
        if isinstance(n.op, ast.Mult):
            for a, b in ((left, right), (right, left)):
                if isinstance(a, (str, bytes, list, tuple)) and isinstance(b, int) and len(a) * max(b, 0) > _MAX_SIZE:
                    raise _NoValue("repeat too large")
        if isinstance(n.op, ast.LShift) and isinstance(right, int) and right > 4096:
            raise _NoValue("shift too large")
        try:
            return fn(left, right)
        except Exception as exc:  # noqa: BLE001 - TypeError, ValueError, ZeroDivisionError ...
            raise _NoValue(type(exc).__name__) from None

    def _BoolOp(self, n):  # noqa: N802
        is_and = isinstance(n.op, ast.And)
        val: Any = None
        for e in n.values:
            val = self.ev(e)
            if bool(val) is not is_and:
                return val
        return val

    def _IfExp(self, n):  # noqa: N802
        return self.ev(n.body) if self.ev(n.test) else self.ev(n.orelse)

    def _Compare(self, n):  # noqa: N802
        left = self.ev(n.left)
        for op, comp in zip(n.ops, n.comparators):
            fn = _CMPOPS.get(type(op))
            if fn is None:
                raise _NoValue("compare")
            right = self.ev(comp)
            try:
                if not fn(left, right):
                    return False
            except TypeError:
                raise _NoValue("compare types") from None
            left = right
        return True

    def _JoinedStr(self, n):  # noqa: N802
        out: List[str] = []
        for v in n.values:
            if isinstance(v, ast.Constant):
                out.append(v.value)
            elif isinstance(v, ast.FormattedValue):
                val = self.ev(v.value)
                conv = {-1: None, 115: str, 114: repr, 97: ascii}.get(v.conversion)
                if v.conversion not in (-1, 115, 114, 97):
                    raise _NoValue("conversion")
                if conv is not None:
                    val = conv(val)
                spec = self.ev(v.format_spec) if v.format_spec is not None else ""
                try:
                    out.append(format(val, spec))
                except Exception as exc:  # noqa: BLE001
                    raise _NoValue(type(exc).__name__) from None
            else:
                raise _NoValue("f-string part")
        return "".join(out)

    def _Subscript(self, n):  # noqa: N802
        base = self.ev(n.value)
        sl = n.slice
        if isinstance(sl, ast.Slice):
            parts = [None if p is None else self.ev(p) for p in (sl.lower, sl.upper, sl.step)]
            key: Any = slice(*parts)
        else:
            key = self.ev(sl)
        if isinstance(base, (LocalModule, ModuleRef, RegexConst)):
            raise _NoValue("subscript of a non-container")
        try:
            return base[key]
        except Exception as exc:  # noqa: BLE001
            raise _NoValue(type(exc).__name__) from None

    def _Attribute(self, n):  # noqa: N802
        base = self.ev(n.value)
        if isinstance(base, ModuleRef):
            if base.name == "re" and n.attr in _RE_FLAGS:
                return _RE_FLAGS[n.attr]
            if base.name == "string" and n.attr in _STRING_CONSTS:
                return _STRING_CONSTS[n.attr]
            raise _NoValue(f"{base.name}.{n.attr}")
        if isinstance(base, LocalModule):
            got = base.get(n.attr, _NoValue)
            if got is _NoValue:
                raise _NoValue(f"module attribute {n.attr}")
            return got
        if isinstance(base, RegexConst) and n.attr in ("pattern", "flags"):
            return getattr(base, n.attr)
        raise _NoValue("attribute")

    def _Call(self, n):  # noqa: N802
        if any(k.arg is None for k in n.keywords):
            raise _NoValue("kwargs unpacking")
        func = n.func
        if isinstance(func, ast.Name) and func.id in _BUILTINS and func.id not in self.ns:
            args = [self.ev(a) for a in n.args]
            kwargs = {k.arg: self.ev(k.value) for k in n.keywords}
            try:
                return _BUILTINS[func.id](*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                raise _NoValue(type(exc).__name__) from None
        if isinstance(func, ast.Attribute):
            recv = self.ev(func.value)
            args = [self.ev(a) for a in n.args]
            kwargs = {k.arg: self.ev(k.value) for k in n.keywords}
            if isinstance(recv, ModuleRef) and recv.name == "re":
                if func.attr == "compile":
                    return self._compile(args, kwargs, n)
                if func.attr == "escape" and len(args) == 1 and isinstance(args[0], str) and not kwargs:
                    return re.escape(args[0])
                raise _NoValue(f"re.{func.attr}")
            if isinstance(recv, str) and func.attr in _STR_METHODS:
                try:
                    return getattr(recv, func.attr)(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    raise _NoValue(type(exc).__name__) from None
        raise _NoValue("call")

    def _compile(self, args, kwargs, node):
        if len(args) > 2 or set(kwargs) - {"flags"}:
            raise _NoValue("re.compile signature")
        pattern = args[0] if args else None
        flags = kwargs.get("flags", args[1] if len(args) == 2 else 0)
        if isinstance(pattern, RegexConst):
            pattern, flags = pattern.pattern, (flags or pattern.flags)
        if not isinstance(pattern, str) or isinstance(flags, bool) or not isinstance(flags, int):
            raise _NoValue("re.compile arguments")
        try:
            re.compile(pattern, flags)
        except (re.error, ValueError, OverflowError, RecursionError):
            raise _NoValue("pattern does not compile") from None
        return RegexConst(pattern, int(flags), self.module, getattr(node, "lineno", 0), getattr(node, "col_offset", 0))


# ---------------------------------------------------------------------------
# locating modules
# ---------------------------------------------------------------------------

def _module_file(level: int, module: Optional[str], here: Path, roots: Sequence[Path]) -> Optional[Path]:
    parts = module.split(".") if module else []
    bases: List[Path] = []
    if level:
        base = here.parent
        for _ in range(level - 1):
            base = base.parent
        bases.append(base)
    else:
        bases.append(here.parent)
        bases.extend(roots)
    for base in bases:
        if not parts:
            cand = base / "__init__.py"
            if cand.is_file():
                return cand
            continue
        cand = base.joinpath(*parts).with_suffix(".py")
        if cand.is_file():
            return cand
        cand = base.joinpath(*parts) / "__init__.py"
        if cand.is_file():
            return cand
    return None


def _store_names(node: ast.AST) -> Iterable[str]:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
            yield sub.id
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield sub.name
        elif isinstance(sub, ast.alias):
            yield (sub.asname or sub.name).split(".")[0]


def evaluate_module(path: Path, roots: Sequence[Path] = (), *, _stack: Tuple[str, ...] = (),
                    _cache: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, Any]:
    """The statically known module-level namespace of *path* (name -> value).

    Names whose value is not statically known are absent.  *roots* are extra directories searched for absolute
    imports.  Pass one ``_cache`` dict to several calls to share the work AND the value objects (a
    :class:`RegexConst` reached through two modules is the same object)."""
    cache = _cache if _cache is not None else {}
    key = str(Path(path).resolve())
    if key in cache:
        return cache[key]
    if key in _stack or len(_stack) >= _MAX_DEPTH:
        return {}
    ns: Dict[str, Any] = {}
    try:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError, ValueError):
        cache[key] = ns
        return ns
    stack = _stack + (key,)
    ev = _Evaluator(ns, key)
    here = Path(path)

    def assign(target: ast.expr, value_node: ast.expr) -> None:
        if isinstance(target, ast.Name):
            try:
                ns[target.id] = ev.ev(value_node)
            except _NoValue:
                ns.pop(target.id, None)
        elif isinstance(target, (ast.Tuple, ast.List)) and all(isinstance(t, ast.Name) for t in target.elts):
            try:
                vals = ev.ev(value_node)
                if not isinstance(vals, (tuple, list)) or len(vals) != len(target.elts):
                    raise _NoValue("unpack")
                for t, v in zip(target.elts, vals):
                    ns[t.id] = v
            except _NoValue:
                for t in target.elts:
                    ns.pop(t.id, None)
        else:
            for name in _store_names(target):
                ns.pop(name, None)

    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                assign(tgt, stmt.value)
        elif isinstance(stmt, ast.AnnAssign):
            if stmt.value is not None:
                assign(stmt.target, stmt.value)
            else:
                for name in _store_names(stmt.target):
                    ns.pop(name, None)
        elif isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name):
            try:
                ns[stmt.target.id] = ev.ev(ast.BinOp(left=ast.Name(id=stmt.target.id, ctx=ast.Load()),
                                                     op=stmt.op, right=stmt.value))
            except _NoValue:
                ns.pop(stmt.target.id, None)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                bound = alias.asname or alias.name.split(".")[0]
                if alias.name in ("re", "string"):
                    ns[bound] = ModuleRef(alias.name)
                    continue
                mod_file = _module_file(0, alias.name, here, roots) if "." not in alias.name else None
                if mod_file is None:
                    ns.pop(bound, None)
                else:
                    sub = evaluate_module(mod_file, roots, _stack=stack, _cache=cache)
                    ns[bound] = LocalModule(str(mod_file), tuple(sorted(sub.items(), key=lambda kv: kv[0])))
        elif isinstance(stmt, ast.ImportFrom):
            _import_from(stmt, here, roots, ns, stack, cache)
        elif isinstance(stmt, (ast.Expr, ast.Pass)):
            continue  # docstrings and bare expressions bind nothing at module level
        else:
            for name in _store_names(stmt):
                ns.pop(name, None)
    cache[key] = ns
    return ns


def _import_from(stmt: ast.ImportFrom, here: Path, roots: Sequence[Path], ns: Dict[str, Any],
                 stack: Tuple[str, ...], cache: Dict[str, Dict[str, Any]]) -> None:
    mod_file = _module_file(stmt.level, stmt.module, here, roots)
    stdlib = stmt.level == 0 and stmt.module in ("re", "string")
    for alias in stmt.names:
        bound = alias.asname or alias.name
        if alias.name != "*":
            ns.pop(bound, None)
    if stdlib:
        for alias in stmt.names:
            if alias.name == "*":
                continue
            bound = alias.asname or alias.name
            if stmt.module == "re" and alias.name in _RE_FLAGS:
                ns[bound] = _RE_FLAGS[alias.name]
            elif stmt.module == "string" and alias.name in _STRING_CONSTS:
                ns[bound] = _STRING_CONSTS[alias.name]
        return
    if mod_file is None:
        return
    sub = evaluate_module(mod_file, roots, _stack=stack, _cache=cache)
    for alias in stmt.names:
        if alias.name == "*":
            exported = sub.get("__all__")
            if isinstance(exported, (list, tuple)) and all(isinstance(x, str) for x in exported):
                names = [x for x in exported if x in sub]
            else:
                names = [k for k in sub if not k.startswith("_")]
            for k in names:
                ns[k] = sub[k]
            continue
        bound = alias.asname or alias.name
        if alias.name in sub:
            ns[bound] = sub[alias.name]
        else:
            # ``from . import sibling``: a sibling module, not a name of the package
            sib = _module_file(stmt.level, alias.name, here, roots) if stmt.module is None and stmt.level else None
            if sib is not None:
                subns = evaluate_module(sib, roots, _stack=stack, _cache=cache)
                ns[bound] = LocalModule(str(sib), tuple(sorted(subns.items(), key=lambda kv: kv[0])))


def regex_consts(ns: Dict[str, Any]) -> Dict[str, RegexConst]:
    """The compiled-regex constants of a namespace, in name order."""
    return {k: ns[k] for k in sorted(ns) if isinstance(ns[k], RegexConst)}


# ---------------------------------------------------------------------------
# the import graph
# ---------------------------------------------------------------------------

@dataclass
class _Mod:
    path: str                  # realpath of the source file
    dotted: Optional[str]      # the name it is imported under, when known
    package: str               # dotted package a relative import is read against ("" at top level)
    is_package: bool
    tree: Optional[ast.Module]
    parents: Dict[ast.AST, ast.AST] = field(default_factory=dict)
    aliases: Dict[str, str] = field(default_factory=dict)       # local name -> dotted graph module
    star_from: List[str] = field(default_factory=list)          # dotted graph modules star-imported
    external_names: Set[str] = field(default_factory=set)       # names bound by importing a non-graph module


#: names whose use hands out arbitrary module objects or imports unknown modules
_DYNAMIC_IMPORT_MODULES = frozenset({"importlib", "runpy", "pkgutil", "imp"})
_DYNAMIC_IMPORT_FUNCS = frozenset({"import_module", "__import__", "reload", "run_module", "run_path",
                                   "find_loader", "load_module", "walk_packages", "iter_modules"})
_MUTATING_METHODS = frozenset({
    # list
    "append", "extend", "insert", "remove", "pop", "clear", "sort", "reverse",
    # dict
    "update", "popitem", "setdefault",
    # set
    "add", "discard", "difference_update", "intersection_update", "symmetric_difference_update",
    # bytearray
    "__setitem__", "__delitem__", "__iadd__", "__imul__",
})


def _json_value(value: Any) -> Tuple[Any, str]:
    """``(JSON-safe value or None, kind)`` for a folded value."""
    import math

    if value is None or isinstance(value, (bool, str)):
        return value, type(value).__name__
    if isinstance(value, int):
        return value, "int"
    if isinstance(value, float):
        return (value, "float") if math.isfinite(value) else (None, "float")
    if isinstance(value, bytes):
        return {"bytes": value.hex()}, "bytes"
    if isinstance(value, RegexConst):
        return {"regex": {"pattern": value.pattern, "flags": value.flags}}, "regex"
    if isinstance(value, tuple):
        items = [_json_value(v) for v in value]
        if any(k in ("module", "unknown") for _, k in items):
            return None, "tuple"
        return {"tuple": [v for v, _ in items]}, "tuple"
    if isinstance(value, list):
        items = [_json_value(v) for v in value]
        if any(k in ("module", "unknown") for _, k in items):
            return None, "list"
        return [v for v, _ in items], "list"
    if isinstance(value, (set, frozenset)):
        try:
            ordered = sorted(value, key=lambda x: (type(x).__name__, repr(x)))
        except TypeError:
            return None, "set"
        items = [_json_value(v) for v in ordered]
        if any(k in ("module", "unknown") for _, k in items):
            return None, "set"
        return {"set": [v for v, _ in items]}, "set"
    if isinstance(value, dict):
        if not all(isinstance(k, str) for k in value):
            return None, "dict"
        items = {k: _json_value(v) for k, v in value.items()}
        if any(kk in ("module", "unknown") for _, kk in items.values()):
            return None, "dict"
        return {k: v for k, (v, _) in items.items()}, "dict"
    if isinstance(value, (LocalModule, ModuleRef)):
        return None, "module"
    return None, "unknown"


#: callables that write a namespace the walker does not follow; a reference to one that is not the callee of a call
#: (an alias, an argument, a default) makes the graph incomplete.  ``exec`` and ``eval`` are incomplete even as callees.
_REFLECTIVE_NAMES = frozenset({"exec", "eval", "compile", "globals", "vars", "locals", "setattr", "delattr", "__import__"})
_REFLECTIVE_ATTRS = frozenset({
    "exec", "eval", "_getframe", "currentframe", "f_globals", "f_locals", "f_back", "f_builtins", "__globals__",
    "__builtins__", "gi_frame", "ag_frame", "cr_frame", "tb_frame", "__setattr__", "__delattr__",
    "get_referrers", "get_referents", "get_objects",
})
#: modules whose import (or use) hands out frames, builtins or arbitrary objects
_REFLECTIVE_MODULES = frozenset({"inspect", "builtins", "gc", "ctypes", "code", "codeop"})
_REFLECTIVE_MODULES_AS_NAMES = frozenset({"builtins", "__builtins__"})
_DICT_METHODS = frozenset({"get", "keys", "items", "values", "copy", "update", "setdefault", "pop", "popitem", "clear",
                           "__contains__", "__getitem__"})


def _namespace_use_ok(mod: "_Mod", node: ast.AST) -> bool:
    """A ``__dict__`` attribute or a ``vars(x)`` call is used in a way the walker follows: subscripted (a store is
    ``_Analysis.stores``' business), the receiver of a dict method, tested with ``in``, iterated, or the argument of a
    read-only wrapper.  Assigned, passed, returned, stored in a container or kept as a default it is an alias."""
    parent = mod.parents.get(node)
    if isinstance(parent, ast.Subscript) and parent.value is node:
        return True
    if isinstance(parent, ast.Attribute) and parent.value is node and parent.attr in _DICT_METHODS:
        grand = mod.parents.get(parent)
        return isinstance(grand, ast.Call) and grand.func is parent
    if isinstance(parent, ast.Compare) and node in parent.comparators and all(
            isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops):
        return True
    if isinstance(parent, (ast.For, ast.AsyncFor)) and parent.iter is node:
        return True
    if isinstance(parent, ast.comprehension) and parent.iter is node:
        return True
    if isinstance(parent, ast.Call) and parent.func is not node and isinstance(parent.func, ast.Name) \
            and parent.func.id in ("len", "sorted", "list", "dict", "set", "tuple", "frozenset", "iter") and node in parent.args:
        return True
    return False


class _Project:
    """The import graph of an entry file: its modules, aliases, bindings and every store into a module."""

    def __init__(self, entry: str, roots: Sequence[str] = (), max_modules: Optional[int] = None):
        from tharkun.solve.loader import project_root_and_module

        self.max_modules = MAX_MODULES if max_modules is None else max_modules
        self.entry = os.path.realpath(entry)
        split = project_root_and_module(self.entry)
        self.project_root = split[0] if split else os.path.dirname(self.entry)
        self.roots: List[str] = list(dict.fromkeys(
            [self.project_root, os.path.dirname(self.entry), *(os.path.realpath(r) for r in roots)]))
        self.mods: Dict[str, _Mod] = {}
        self.by_dotted: Dict[str, str] = {}
        self.incomplete: List[str] = []
        self.dynamic_import = False
        self.sys_modules_used = False
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._binders: Dict[str, "_Binder"] = {}
        self._build()

    def binder(self, path: str) -> Optional["_Binder"]:
        mod = self.mods.get(path)
        if mod is None or mod.tree is None:
            return None
        got = self._binders.get(path)
        if got is None:
            got = self._binders[path] = _Binder(mod.tree)
        return got

    # -- locating ----------------------------------------------------------

    def find(self, dotted: str) -> Optional[str]:
        parts = dotted.split(".")
        for root in self.roots:
            base = os.path.join(root, *parts)
            if os.path.isfile(base + ".py"):
                return os.path.realpath(base + ".py")
            init = os.path.join(base, "__init__.py")
            if os.path.isfile(init):
                return os.path.realpath(init)
        return None

    def _dotted_of(self, path: str) -> Tuple[Optional[str], str, bool]:
        from tharkun.solve.loader import project_root_and_module

        is_init = os.path.basename(path) == "__init__.py"
        for root in self.roots:
            if path.startswith(root + os.sep):
                rel = path[len(root) + 1:-3].split(os.sep)
                if rel and rel[-1] == "__init__":
                    rel = rel[:-1]
                if rel and all(p.isidentifier() for p in rel):
                    dotted = ".".join(rel)
                    package = dotted if is_init else dotted.rpartition(".")[0]
                    return dotted, package, is_init
        split = project_root_and_module(path)
        if split:
            dotted = split[1]
            package = dotted if is_init else dotted.rpartition(".")[0]
            return dotted, package, is_init
        return None, "", is_init

    # -- building ----------------------------------------------------------

    def _load(self, path: str) -> Optional[_Mod]:
        if path in self.mods:
            return self.mods[path]
        dotted, package, is_package = self._dotted_of(path)
        try:
            tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
        except (OSError, SyntaxError, UnicodeDecodeError, ValueError) as exc:
            self.incomplete.append(f"{path}: unreadable ({type(exc).__name__})")
            tree = None
        mod = _Mod(path, dotted, package, is_package, tree)
        if tree is not None:
            for parent in ast.walk(tree):
                for child in ast.iter_child_nodes(parent):
                    mod.parents[child] = parent
        self.mods[path] = mod
        if dotted:
            self.by_dotted.setdefault(dotted, path)
        return mod

    def _resolve_relative(self, mod: _Mod, level: int, module: Optional[str]) -> Optional[str]:
        """Dotted name a relative ``from {'.' * level}{module}`` import names, or None above the top package."""
        parts = mod.package.split(".") if mod.package else []
        up = level - 1
        if up > len(parts):
            return None
        anchor = parts[:len(parts) - up]
        return ".".join(anchor + ([module] if module else []))

    def _build(self) -> None:
        queue: List[str] = []
        queued: Set[str] = set()

        def add(path: Optional[str]) -> None:
            if not path or path in queued:
                return
            if len(queued) >= self.max_modules:
                if "module cap reached" not in self.incomplete:
                    self.incomplete.append("module cap reached")
                return
            queued.add(path)
            queue.append(path)

        entry_mod = self._load(self.entry)
        if entry_mod is None:
            return
        # the package chain runs before the entry: its __init__ files are part of the program
        if entry_mod.dotted:
            parts = entry_mod.dotted.split(".")
            for i in range(1, len(parts) if not entry_mod.is_package else len(parts) + 1):
                add(self.find(".".join(parts[:i])))
        add(self.entry)
        while queue:
            path = queue.pop(0)
            mod = self._load(path)
            if mod is None or mod.tree is None:
                continue
            for dotted_name, node, required in self._imports(mod):
                target = self.find(dotted_name)
                if target is None:
                    if required and isinstance(node, ast.ImportFrom) and node.level:
                        self.incomplete.append(f"{mod.path}:{node.lineno}: unresolved relative import "
                                               f"{'.' * node.level}{node.module or ''}")
                    continue
                add(target)
                # an imported module's own package chain runs first
                dotted_parts = dotted_name.split(".")
                for i in range(1, len(dotted_parts)):
                    add(self.find(".".join(dotted_parts[:i])))
            self._scan_dynamic(mod)
            self._scan_reflection(mod)
            self._bind_aliases(mod)

    def _imports(self, mod: _Mod) -> Iterator[Tuple[str, ast.AST, bool]]:
        """``(dotted name, node, required)``: ``required`` is False for the ``base.name`` candidates of a ``from
        base import name`` (``name`` may be an attribute, not a module)."""
        assert mod.tree is not None
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    yield a.name, node, False
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = self._resolve_relative(mod, node.level, node.module)
                    if base is None:
                        self.incomplete.append(f"{mod.path}:{node.lineno}: relative import beyond the top package")
                        continue
                else:
                    base = node.module or ""
                if base:
                    yield base, node, bool(node.level)
                    for a in node.names:
                        if a.name != "*":
                            yield base + "." + a.name, node, False
                elif node.level:
                    for a in node.names:
                        if a.name != "*":
                            yield a.name if not mod.package else f"{mod.package}.{a.name}", node, False

    def _scan_dynamic(self, mod: _Mod) -> None:
        assert mod.tree is not None
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in _DYNAMIC_IMPORT_MODULES:
                        self.dynamic_import = True
                        self.incomplete.append(f"{mod.path}:{node.lineno}: imports {a.name} (dynamic import)")
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in _DYNAMIC_IMPORT_MODULES:
                self.dynamic_import = True
                self.incomplete.append(f"{mod.path}:{node.lineno}: imports from {node.module} (dynamic import)")
            elif isinstance(node, ast.Name) and node.id == "__import__":
                self.dynamic_import = True
                self.incomplete.append(f"{mod.path}:{node.lineno}: uses __import__ (dynamic import)")
            elif isinstance(node, ast.Attribute) and node.attr in _DYNAMIC_IMPORT_FUNCS and isinstance(node.value, ast.Name) \
                    and node.value.id in _DYNAMIC_IMPORT_MODULES:
                self.dynamic_import = True
            elif isinstance(node, ast.Attribute) and node.attr == "modules" and isinstance(node.value, ast.Name) \
                    and node.value.id == "sys":
                self.sys_modules_used = True

    def _scan_reflection(self, mod: _Mod) -> None:
        """Make the graph INCOMPLETE (nothing proven) when a module can write a namespace the walker does not
        follow.  A direct call of ``globals()`` / ``vars()`` / ``setattr`` is followed (``_Analysis.stores``); what
        is not followed is the same callable reached under another name, handed on, or a frame / builtins route to
        it.  So: a call of ``exec`` / ``eval`` (the string can do anything, to any module); a REFERENCE to one of
        ``_REFLECTIVE_NAMES`` that is not the callee of a call; the modules in ``_REFLECTIVE_MODULES``; a frame or
        function-globals attribute; ``__setattr__`` / ``__delattr__`` not called on the spot; a ``__dict__`` or
        ``vars(x)`` that is used other than subscripted or called through a dict method; a constant attribute name
        of a ``getattr`` that is one of these."""
        assert mod.tree is not None

        def flag(node: ast.AST, what: str) -> None:
            self.incomplete.append(f"{mod.path}:{getattr(node, 'lineno', 0)}: {what} (reflection)")

        for node in ast.walk(mod.tree):
            parent = mod.parents.get(node)
            callee = isinstance(parent, ast.Call) and parent.func is node
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.split(".")[0] in _REFLECTIVE_MODULES:
                        flag(node, f"imports {a.name}")
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").split(".")[0] in _REFLECTIVE_MODULES:
                    flag(node, f"imports from {node.module}")
                for a in node.names:
                    if a.name in _REFLECTIVE_NAMES or a.name in _REFLECTIVE_ATTRS:
                        flag(node, f"imports {a.name}")
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in _REFLECTIVE_MODULES_AS_NAMES:
                    flag(node, f"uses {node.id}")
                elif node.id in _REFLECTIVE_NAMES and not callee:
                    flag(node, f"{node.id} used other than as a callee")
                elif node.id in ("exec", "eval"):
                    flag(node, f"calls {node.id}")
                elif node.id == "vars" and isinstance(parent, ast.Call) and parent.args \
                        and not _namespace_use_ok(mod, parent):
                    flag(node, "vars(x) handed on")
            elif isinstance(node, ast.Attribute):
                if node.attr in _REFLECTIVE_ATTRS:
                    if node.attr in ("__setattr__", "__delattr__") and callee:
                        continue                                # a direct call: ``_Analysis.stores`` follows it
                    flag(node, f"reaches .{node.attr}")
                elif node.attr == "__dict__" and not _namespace_use_ok(mod, node):
                    flag(node, "__dict__ handed on")
            elif isinstance(node, ast.Call):
                func = node.func
                fname = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
                if fname in ("getattr", "hasattr", "attrgetter", "__getattribute__"):
                    for arg in node.args[1:] if fname != "attrgetter" else node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and any(
                                part in _REFLECTIVE_NAMES or part in _REFLECTIVE_ATTRS or part == "__dict__"
                                for part in arg.value.split(".")):
                            flag(node, f"{fname} of {arg.value!r}")

    def _bind_aliases(self, mod: _Mod) -> None:
        assert mod.tree is not None
        for node in ast.walk(mod.tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.asname:
                        if self.find(a.name):
                            mod.aliases[a.asname] = a.name
                        else:
                            mod.external_names.add(a.asname)
                    else:
                        top = a.name.split(".")[0]
                        if self.find(top):
                            mod.aliases[top] = top
                        else:
                            mod.external_names.add(top)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = self._resolve_relative(mod, node.level, node.module)
                    if base is None:
                        continue
                else:
                    base = node.module or ""
                for a in node.names:
                    if a.name == "*":
                        if base and self.find(base):
                            mod.star_from.append(base)
                        continue
                    sub = f"{base}.{a.name}" if base else a.name
                    bound = a.asname or a.name
                    if self.find(sub) and self._is_module_name(base, a.name):
                        mod.aliases[bound] = sub
                    elif not (base and self.find(base)):
                        mod.external_names.add(bound)

    def _is_module_name(self, base: str, name: str) -> bool:
        """``from base import name`` names the SUBMODULE ``base.name`` (not an attribute of ``base``) when the
        submodule file exists and ``base`` does not bind ``name`` to something else (Python looks for the
        attribute first: ``from .impl import impl`` in ``base/__init__.py`` makes ``base.impl`` the function).
        A binding of ``name`` in ``base`` that is itself ``from base import name`` / ``from . import name`` is the
        submodule, not something else."""
        if self.find(f"{base}.{name}" if base else name) is None:
            return False
        path = self.find(base) if base else None
        binder = self.binder(path) if path else None
        if binder is None:
            return True
        base_mod = self.mods.get(path) if path else None
        for b in binder.bindings.get(name, []):
            if b.kind == "importfrom" and b.source is not None and base_mod is not None:
                module_ish, level, imported, _a = b.source
                resolved = self._resolve_relative(base_mod, level, module_ish) if level else module_ish
                if imported == name and resolved == base:
                    continue
            return False
        return True

    # -- resolution --------------------------------------------------------

    def resolve(self, mod: _Mod, node: ast.AST) -> Optional[str]:
        """The dotted GRAPH module the expression denotes in *mod*, or None."""
        if isinstance(node, ast.Name):
            return mod.aliases.get(node.id)
        if isinstance(node, ast.Attribute):
            base = self.resolve(mod, node.value)
            if base is not None:
                cand = f"{base}.{node.attr}"
                return cand if self.find(cand) and self._submodule(base, node.attr) else None
            return None
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) \
                and node.value.attr == "modules" and isinstance(node.value.value, ast.Name) \
                and node.value.value.id == "sys":
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and self.find(key.value):
                return key.value
        return None

    def _submodule(self, base: str, attr: str) -> bool:
        return self._is_module_name(base, attr)

    def namespace(self, mod: _Mod) -> Dict[str, Any]:
        return evaluate_module(Path(mod.path), [Path(r) for r in self.roots], _cache=self._cache)


# ---------------------------------------------------------------------------
# bindings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Binding:
    kind: str        # assign augassign annassign import importfrom star def class for with except walrus del match typealias
    line: int
    top: bool        # executed unconditionally (directly in the module body, or under ``with``)
    source: Optional[Tuple[str, int, str, Optional[str]]] = None   # importfrom: (module-ish, level, imported name, asname)


def _target_names(target: ast.AST) -> Iterator[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for e in target.elts:
            yield from _target_names(e)
    elif isinstance(target, ast.Starred):
        yield from _target_names(target.value)


def _walrus_names(node: ast.AST) -> Iterator[Tuple[str, int]]:
    """Names bound by ``:=`` in an expression (they bind in the enclosing scope, comprehensions included; a
    lambda body is its own scope)."""
    stack = [node]
    while stack:
        n = stack.pop()
        if isinstance(n, ast.Lambda):
            continue
        if isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name):
            yield n.target.id, n.lineno
        stack.extend(ast.iter_child_nodes(n))


class _Binder:
    """Every binding of a name in the MODULE scope, and every module-name store made from a function or class
    scope through ``global``."""

    def __init__(self, tree: ast.Module):
        self.bindings: Dict[str, List[_Binding]] = {}
        self.global_stores: Dict[str, List[int]] = {}
        self._scope(tree.body, top=True, module_scope=True, globals_=set(), sink=self.bindings)
        for scope in ast.walk(tree):
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                declared = self._global_names(scope.body)
                if declared:
                    local: Dict[str, List[_Binding]] = {}
                    self._scope(scope.body, top=True, module_scope=False, globals_=declared, sink=local)
                    for name, items in local.items():
                        if name in declared:
                            self.global_stores.setdefault(name, []).extend(b.line for b in items)

    @staticmethod
    def _global_names(body: Sequence[ast.stmt]) -> Set[str]:
        out: Set[str] = set()
        stack: List[ast.AST] = list(body)
        while stack:
            n = stack.pop()
            if isinstance(n, ast.Global):
                out.update(n.names)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                continue
            stack.extend(ast.iter_child_nodes(n))
        return out

    def _bind(self, sink, name: str, kind: str, line: int, top: bool, source=None) -> None:
        sink.setdefault(name, []).append(_Binding(kind, line, top, source))

    def _scope(self, body: Sequence[ast.stmt], *, top: bool, module_scope: bool, globals_: Set[str], sink) -> None:
        for st in body:
            self._stmt(st, top=top, module_scope=module_scope, globals_=globals_, sink=sink)

    def _stmt(self, st: ast.stmt, *, top: bool, module_scope: bool, globals_: Set[str], sink) -> None:
        def bind(name, kind, line=None, t=None, source=None):
            self._bind(sink, name, kind, line or st.lineno, top if t is None else t, source)

        def walrus(expr):
            for name, line in _walrus_names(expr):
                bind(name, "walrus", line)

        def sub(stmts, inner_top):
            self._scope(stmts, top=inner_top, module_scope=module_scope, globals_=globals_, sink=sink)

        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bind(st.name, "def")
            for d in st.decorator_list:
                walrus(d)
        elif isinstance(st, ast.ClassDef):
            bind(st.name, "class")
            for d in st.decorator_list:
                walrus(d)
        elif isinstance(st, ast.Assign):
            for t in st.targets:
                for name in _target_names(t):
                    bind(name, "assign")
            walrus(st.value)
        elif isinstance(st, ast.AugAssign):
            for name in _target_names(st.target):
                bind(name, "augassign")
            walrus(st.value)
        elif isinstance(st, ast.AnnAssign):
            if st.value is not None:
                for name in _target_names(st.target):
                    bind(name, "annassign")
                walrus(st.value)
        elif isinstance(st, (ast.For, ast.AsyncFor)):
            for name in _target_names(st.target):
                bind(name, "for", t=False)
            walrus(st.iter)
            sub(st.body, False)
            sub(st.orelse, False)
        elif isinstance(st, ast.While):
            walrus(st.test)
            sub(st.body, False)
            sub(st.orelse, False)
        elif isinstance(st, ast.If):
            walrus(st.test)
            sub(st.body, False)
            sub(st.orelse, False)
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            for item in st.items:
                walrus(item.context_expr)
                if item.optional_vars is not None:
                    for name in _target_names(item.optional_vars):
                        bind(name, "with")
            sub(st.body, top)
        elif isinstance(st, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            sub(st.body, False)
            for h in st.handlers:
                if h.name:
                    bind(h.name, "except", h.lineno, False)
                sub(h.body, False)
            sub(st.orelse, False)
            sub(st.finalbody, False)
        elif isinstance(st, ast.Match):
            walrus(st.subject)
            for case in st.cases:
                for pat in ast.walk(case.pattern):
                    name = getattr(pat, "name", None) if isinstance(pat, (ast.MatchAs, ast.MatchStar)) else None
                    if name:
                        bind(name, "match", st.lineno, False)
                    if isinstance(pat, ast.MatchMapping) and pat.rest:
                        bind(pat.rest, "match", st.lineno, False)
                sub(case.body, False)
        elif isinstance(st, ast.Import):
            for a in st.names:
                bind((a.asname or a.name.split(".")[0]), "import", source=(a.name, 0, "", a.asname))
        elif isinstance(st, ast.ImportFrom):
            for a in st.names:
                if a.name == "*":
                    bind("*", "star", source=(st.module or "", st.level, "*", None))
                else:
                    bind(a.asname or a.name, "importfrom", source=(st.module or "", st.level, a.name, a.asname))
        elif isinstance(st, ast.Delete):
            for t in st.targets:
                for name in _target_names(t):
                    bind(name, "del")
        elif isinstance(st, getattr(ast, "TypeAlias", ())):
            if isinstance(st.name, ast.Name):
                bind(st.name.id, "typealias")
        elif isinstance(st, (ast.Expr, ast.Return, ast.Raise, ast.Assert)):
            expr = getattr(st, "value", None) or getattr(st, "exc", None) or getattr(st, "test", None)
            if expr is not None:
                walrus(expr)


# ---------------------------------------------------------------------------
# evidence
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Rebind:
    kind: str
    module: str
    line: int
    detail: str = ""

    def as_dict(self, rel) -> Dict[str, Any]:
        out: Dict[str, Any] = {"kind": self.kind, "module": rel(self.module), "line": self.line}
        if self.detail:
            out["detail"] = self.detail
        return out


class _Analysis:
    """Rebinding evidence for the names of the modules of a :class:`_Project`."""

    def __init__(self, project: _Project):
        self.p = project
        self.binders: Dict[str, _Binder] = {}
        for path in project.mods:
            b = project.binder(path)
            if b is not None:
                self.binders[path] = b
        self._stores: Optional[List[Tuple[_Mod, int, str, Optional[str], Optional[str]]]] = None
        self._escaped: Optional[Set[str]] = None
        self._self_dynamic: Dict[str, List[Tuple[str, int]]] = {}
        self._memo: Dict[Tuple[str, str], Tuple[List[_Rebind], str]] = {}

    # -- stores ------------------------------------------------------------

    def stores(self) -> List[Tuple[_Mod, int, str, Optional[str], Optional[str]]]:
        """Every attribute store in the graph: ``(module, line, kind, target dotted or None, attr or None)``.
        ``target`` is the graph module the base resolves to (None when it does not resolve); ``attr`` is the
        stored name (None when dynamic)."""
        if self._stores is not None:
            return self._stores
        out: List[Tuple[_Mod, int, str, Optional[str], Optional[str]]] = []
        for mod in self.p.mods.values():
            if mod.tree is None:
                continue
            for node in ast.walk(mod.tree):
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    out.append((mod, node.lineno, "attr-store", self.p.resolve(mod, node.value), node.attr))
                elif isinstance(node, ast.Call):
                    func = node.func
                    fname = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
                    if fname in ("setattr", "delattr", "__setattr__", "__delattr__") and node.args:
                        base = node.args[0]
                        name_arg = node.args[1] if len(node.args) > 1 else None
                        if fname == "__setattr__" and isinstance(func, ast.Attribute) and not (
                                isinstance(func.value, ast.Name) and func.value.id == "object") and len(node.args) >= 1:
                            base, name_arg = func.value, node.args[0]
                        attr = name_arg.value if isinstance(name_arg, ast.Constant) and isinstance(name_arg.value, str) else None
                        out.append((mod, node.lineno, "setattr", self.p.resolve(mod, base), attr))
                    elif isinstance(func, ast.Attribute) and func.attr in ("update", "setdefault", "pop", "popitem", "clear") \
                            and isinstance(func.value, ast.Attribute) and func.value.attr == "__dict__":
                        out.append((mod, node.lineno, "dict-write", self.p.resolve(mod, func.value.value), None))
                    elif fname in ("exec", "eval"):
                        out.append((mod, node.lineno, "exec", mod.dotted, None))
                elif isinstance(node, ast.Subscript) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    val = node.value
                    if isinstance(val, ast.Attribute) and val.attr == "__dict__":
                        key = node.slice
                        attr = key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else None
                        out.append((mod, node.lineno, "dict-write", self.p.resolve(mod, val.value), attr))
                    elif isinstance(val, ast.Call) and isinstance(val.func, ast.Name) and val.func.id == "vars" and val.args:
                        key = node.slice
                        attr = key.value if isinstance(key, ast.Constant) and isinstance(key.value, str) else None
                        out.append((mod, node.lineno, "dict-write", self.p.resolve(mod, val.args[0]), attr))
            # globals()/vars()/locals() used other than as a read
            for node in ast.walk(mod.tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("globals", "vars", "locals") \
                        and not (node.func.id == "vars" and node.args):
                    if node.func.id == "locals" and not self._at_module_scope(mod, node):
                        continue
                    if not self._read_only_namespace_use(mod, node):
                        out.append((mod, node.lineno, "globals", mod.dotted, None))
        self._stores = out
        return out

    @staticmethod
    def _at_module_scope(mod: _Mod, node: ast.AST) -> bool:
        cur = mod.parents.get(node)
        while cur is not None:
            if isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                return False
            cur = mod.parents.get(cur)
        return True

    @staticmethod
    def _read_only_namespace_use(mod: _Mod, call: ast.Call) -> bool:
        """``globals()`` read as ``globals()[k]`` / ``.get`` / ``.keys`` / ``in`` / ``len`` -- never stored through,
        never handed to anyone who could."""
        parent = mod.parents.get(call)
        if isinstance(parent, ast.Subscript) and parent.value is call and isinstance(parent.ctx, ast.Load):
            return True
        if isinstance(parent, ast.Attribute) and parent.value is call and parent.attr in (
                "get", "keys", "items", "values", "copy", "__contains__", "__getitem__"):
            grand = mod.parents.get(parent)
            return isinstance(grand, ast.Call) and grand.func is parent
        if isinstance(parent, ast.Compare) and call in parent.comparators and all(
                isinstance(op, (ast.In, ast.NotIn)) for op in parent.ops):
            return True
        if isinstance(parent, ast.Call) and parent.func is not call and isinstance(parent.func, ast.Name) \
                and parent.func.id in ("len", "sorted", "list", "dict", "set", "tuple") and call in parent.args:
            return True
        return False

    # -- escape ------------------------------------------------------------

    def escaped(self) -> Set[str]:
        """Dotted graph modules whose module object can be reached other than by an attribute access or a
        ``from M import name``: passed, assigned, returned, stored, or handed out by ``sys.modules`` /
        ``importlib``."""
        if self._escaped is not None:
            return self._escaped
        out: Set[str] = set()
        if self.p.sys_modules_used or self.p.dynamic_import:
            out |= set(self.p.by_dotted)
        for mod in self.p.mods.values():
            if mod.tree is None:
                continue
            for node in ast.walk(mod.tree):
                if not isinstance(node, (ast.Name, ast.Attribute)):
                    continue
                parent = mod.parents.get(node)
                if isinstance(parent, ast.Attribute) and parent.value is node:
                    continue                                   # the base of a longer chain: not a bare use
                if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
                    continue
                if isinstance(parent, (ast.Import, ast.ImportFrom, ast.alias)):
                    continue
                if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
                    continue
                target = self.p.resolve(mod, node)
                if target is not None:
                    out.add(target)
        self._escaped = out
        return out

    # -- evidence for one name ---------------------------------------------

    def rebinds(self, module_path: str, name: str, seen: Optional[Set[Tuple[str, str]]] = None
                ) -> Tuple[List[_Rebind], str]:
        """``(evidence, origin)`` for ``module_path``'s module-level *name*; evidence is empty when nothing in
        the graph rebinds it (completeness of the graph is the caller's separate question)."""
        key = (module_path, name)
        if key in self._memo:
            return self._memo[key]
        seen = set() if seen is None else seen
        if key in seen:
            return [], "cycle"
        seen = seen | {key}
        mod = self.p.mods.get(module_path)
        binder = self.binders.get(module_path)
        if mod is None or binder is None:
            return [], "unknown"
        evidence: List[_Rebind] = []
        items = list(binder.bindings.get(name, []))
        star_origin: Optional[Tuple[str, str]] = None
        if not items:
            for star in mod.star_from:
                src = self.p.find(star)
                if src and self._exports(src, name):
                    items.append(_Binding("star", 0, True, (star, 0, name, None)))
                    star_origin = (src, name)
        origin = items[0].kind if items else "none"
        if not items:
            return [], "none"
        for b in items[1:]:
            evidence.append(_Rebind("module-assign", module_path, b.line, b.kind))
        if len(items) == 1 and not items[0].top:
            evidence.append(_Rebind("conditional-binding", module_path, items[0].line, items[0].kind))
        # a star import after the name's own binding can rebind it
        for star in mod.star_from:
            src = self.p.find(star)
            if src and items and items[0].kind != "star" and self._exports(src, name):
                evidence.append(_Rebind("module-assign", module_path, 0, "star import"))
        for line in binder.global_stores.get(name, []):
            evidence.append(_Rebind("global-store", module_path, line))
        # stores from anywhere in the graph
        escaped = self.p.mods[module_path].dotted in self.escaped() if mod.dotted else True
        for smod, line, kind, target, attr in self.stores():
            if kind in ("globals", "exec"):
                if smod.path == module_path:
                    evidence.append(_Rebind(kind, smod.path, line))
                continue
            if target is not None:
                if self.p.find(target) != module_path:
                    continue
                if attr is None or attr == name:
                    evidence.append(_Rebind(kind, smod.path, line, f"{smod.dotted or smod.path} stores into this module"))
            else:
                if attr is None or attr == name:
                    if escaped or smod.path == module_path or smod.star_from:
                        evidence.append(_Rebind("possible-" + kind, smod.path, line,
                                                "a store through a base that cannot be resolved to a module"))
        # the alias chain: from X import Y is stable only when X.Y is
        first = items[0]
        if first.kind in ("importfrom", "star") and first.source is not None:
            module_ish, level, imported, _asname = first.source
            if star_origin is not None:
                src_path, src_name = star_origin
            else:
                base = self.p._resolve_relative(mod, level, module_ish) if level else module_ish
                src_path = self.p.find(base) if base else None
                src_name = imported
                if src_path is None and base is not None:
                    # ``from . import sibling`` names a module, which is stable
                    sub = self.p.find(f"{base}.{imported}" if base else imported)
                    if sub is not None:
                        origin = "import"
                        src_path = None
                elif src_path is not None and not self.p._is_module_name(base, imported):
                    pass
                elif src_path is not None:
                    src_path = None            # ``from pkg import submodule``: the module object, stable
            if src_path is not None and src_path != module_path:
                sub_ev, sub_origin = self.rebinds(src_path, src_name, seen)
                for ev in sub_ev:
                    evidence.append(_Rebind(ev.kind, ev.module, ev.line,
                                            (ev.detail + "; " if ev.detail else "") + f"via {mod.dotted or module_path}.{name}"))
                origin = sub_origin if sub_origin not in ("none", "unknown", "cycle") else origin
        # a mutable folded value is not mutated through any alias
        value = self.p.namespace(mod).get(name, _NoValue)
        if _holds_mutable(value):
            evidence.append(_Rebind("mutation", module_path, items[0].line,
                                    "the value holds a mutable member; a read of the name can hand it out"))
        elif isinstance(value, _MUTABLE_TYPES):
            evidence.extend(self._mutations(module_path, name))
        out = (_dedupe(evidence), origin)
        self._memo[key] = out
        return out

    def _exports(self, src_path: str, name: str) -> bool:
        """Does ``from <src> import *`` bind *name*?"""
        mod = self.p.mods.get(src_path)
        if mod is None or mod.tree is None:
            return False
        ns = self.p.namespace(mod)
        exported = ns.get("__all__")
        if isinstance(exported, (list, tuple)) and all(isinstance(x, str) for x in exported):
            return name in exported
        if name.startswith("_"):
            return False
        binder = self.binders.get(src_path)
        return (binder is not None and name in binder.bindings) or name in ns

    def _mutations(self, module_path: str, name: str) -> List[_Rebind]:
        """Mutation or escape of the object bound to ``module_path.name`` through any of its aliases."""
        out: List[_Rebind] = []
        for mod in self.p.mods.values():
            if mod.tree is None:
                continue
            names = self._alias_names(mod, module_path, name)
            for node in ast.walk(mod.tree):
                ref = self._references(mod, node, names, module_path, name)
                if ref is None:
                    continue
                kind = self._use_kind(mod, ref)
                if kind:
                    out.append(_Rebind("mutation", mod.path, getattr(ref, "lineno", 0), kind))
        return out

    def _alias_names(self, mod: _Mod, module_path: str, name: str) -> Set[str]:
        """Names in *mod* bound to the object ``module_path.name`` (the defining binding, a ``from`` import of it,
        a star import that carries it)."""
        names: Set[str] = set()
        if mod.path == module_path:
            names.add(name)
        binder = self.binders.get(mod.path)
        if binder is None:
            return names
        for local, items in binder.bindings.items():
            for b in items:
                if b.kind == "importfrom" and b.source is not None:
                    module_ish, level, imported, _a = b.source
                    base = self.p._resolve_relative(mod, level, module_ish) if level else module_ish
                    src = self.p.find(base) if base else None
                    if src == module_path and imported == name:
                        names.add(local)
        for star in mod.star_from:
            if self.p.find(star) == module_path and self._exports(module_path, name):
                names.add(name)
        return names

    def _references(self, mod: _Mod, node: ast.AST, names: Set[str], module_path: str, name: str):
        if isinstance(node, ast.Name) and node.id in names and isinstance(node.ctx, ast.Load):
            return node
        if isinstance(node, ast.Attribute) and node.attr == name and isinstance(node.ctx, ast.Load):
            base = self.p.resolve(mod, node.value)
            if base is not None and self.p.find(base) == module_path:
                return node
        return None

    def _use_kind(self, mod: _Mod, ref: ast.AST) -> str:
        """What a use of the object does to it: ``"mutated ..."`` / ``"passed to a call"`` / ``"stored ..."`` /
        ``"returned"``; ``""`` for a plain read (subscript load, attribute read of a non-mutator, iteration,
        ``in``, ``len``)."""
        parent = mod.parents.get(ref)
        if isinstance(parent, ast.Attribute) and parent.value is ref:
            return f"mutated by .{parent.attr}()" if parent.attr in _MUTATING_METHODS else ""
        if isinstance(parent, ast.Subscript) and parent.value is ref:
            return "mutated by item assignment" if isinstance(parent.ctx, (ast.Store, ast.Del)) else ""
        if isinstance(parent, ast.AugAssign) and parent.target is ref:
            return "mutated by augmented assignment"
        call: Optional[ast.Call] = None
        if isinstance(parent, ast.Call) and ref in parent.args:
            call = parent
        elif isinstance(parent, ast.keyword) and isinstance(mod.parents.get(parent), ast.Call):
            call = mod.parents[parent]  # type: ignore[assignment]
        elif isinstance(parent, ast.Starred) and isinstance(mod.parents.get(parent), ast.Call):
            call = mod.parents[parent]  # type: ignore[assignment]
        if call is not None:
            fn = call.func
            if isinstance(fn, ast.Name) and fn.id in _READ_ONLY_CALLS:
                return ""
            return "passed to a call"
        if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) and getattr(parent, "value", None) is ref:
            return "stored under another name"
        if isinstance(parent, (ast.Return, ast.Yield, ast.YieldFrom)):
            return "returned"
        if isinstance(parent, (ast.Tuple, ast.List, ast.Set, ast.Dict)):
            return "stored in a container"
        if isinstance(parent, ast.arguments):
            return "kept as a default argument value"
        if isinstance(parent, ast.Lambda) and parent.body is ref:
            return "returned"
        if isinstance(parent, ast.IfExp) and ref is not parent.test:
            return "returned through a conditional expression"
        if isinstance(parent, ast.BoolOp) and not self._not_an_alias(mod, parent):
            return "returned through an and/or expression"
        return ""

    @staticmethod
    def _not_an_alias(mod: _Mod, node: ast.AST) -> bool:
        """A ``BoolOp`` whose value is only tested (an ``if`` / ``while`` / ``assert`` condition, a comprehension
        filter, under ``not`` or a comparison): the operand is read, never kept."""
        cur = node
        while True:
            parent = mod.parents.get(cur)
            if isinstance(parent, ast.BoolOp):
                cur = parent
                continue
            if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
                return True
            if isinstance(parent, ast.Compare):
                return True
            if isinstance(parent, (ast.If, ast.While, ast.Assert)):
                return parent.test is cur
            if isinstance(parent, ast.IfExp):
                return parent.test is cur
            if isinstance(parent, ast.comprehension):
                return cur in parent.ifs
            return False


_READ_ONLY_CALLS = frozenset({"len", "sorted", "list", "tuple", "set", "dict", "frozenset", "sum", "min", "max",
                              "any", "all", "repr", "str", "bool", "iter", "isinstance", "enumerate", "zip",
                              "reversed", "print", "hash", "id", "type"})


def _dedupe(items: List[_Rebind]) -> List[_Rebind]:
    seen: Set[Tuple[str, str, int, str]] = set()
    out: List[_Rebind] = []
    for r in items:
        k = (r.kind, r.module, r.line, r.detail)
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# public
# ---------------------------------------------------------------------------

def _rel(project: _Project):
    def rel(path: str) -> str:
        for root in project.roots:
            if path.startswith(root + os.sep):
                return path[len(root) + 1:]
        return path
    return rel


def module_constants(entry_path: str, roots: Sequence[str] = (), *,
                     max_modules: Optional[int] = None) -> List[Dict[str, Any]]:
    """One record per module-level name of the ENTRY module: is it a constant?

    ``{"kind": "module_constant", "module", "name", "proven", "value", "value_kind", "value_repr", "origin",
    "bindings", "rebinds", "provenance"}``.  ``proven`` is True only when ``rebinds`` is empty AND the graph is
    complete (``provenance.complete``); ``rebinds`` lists the evidence (``kind``, ``module``, ``line``,
    ``detail``) of everything in the import graph that rebinds, stores into or mutates the name.  ``value`` is
    the statically folded value as JSON (tuples ``{"tuple": [...]}``, sets ``{"set": [...]}``, bytes
    ``{"bytes": hex}``), ``None`` when it is not statically known -- a name can be proven stable without a
    known value (``K = int(os.environ['K'])``: read once, never rebound).  ``origin`` is how the name is bound
    (``assign``, ``importfrom``, ``import``, ``def``, ``class`` ...; an imported name reports the kind of its
    defining binding).  ``bindings`` lists the modules of the graph that bind the same object under some name.
    ``max_modules`` caps the import graph (default :data:`MAX_MODULES`); a graph over the cap is INCOMPLETE, so
    nothing is proven.  Executes nothing; ``[]`` when the file cannot be read."""
    project = _Project(entry_path, roots, max_modules)
    entry = project.mods.get(project.entry)
    if entry is None or entry.tree is None:
        return []
    analysis = _Analysis(project)
    rel = _rel(project)
    ns = project.namespace(entry)
    complete = not project.incomplete
    binder = analysis.binders[entry.path]
    names = set(binder.bindings)
    for star in entry.star_from:
        src = project.find(star)
        if src:
            names |= {n for n in project.namespace(project.mods[src]) if analysis._exports(src, n)}
            names |= {n for n in analysis.binders.get(src, _EMPTY_BINDER).bindings if analysis._exports(src, n)}
    records: List[Dict[str, Any]] = []
    for name in sorted(names):
        if name == "*" or (name.startswith("__") and name.endswith("__")):
            continue
        evidence, origin = analysis.rebinds(entry.path, name)
        value = ns.get(name, _NoValue)
        known = value is not _NoValue
        jv, vkind = _json_value(value) if known else (None, "unknown")
        proven = complete and not evidence
        records.append({
            "kind": "module_constant",
            "module": rel(entry.path),
            "name": name,
            "proven": proven,
            "value": jv if known else None,
            "value_kind": vkind,
            "value_repr": (repr(value)[:200] if known and vkind not in ("module",) else None),
            "origin": origin,
            "bindings": [{"module": rel(entry.path), "name": name}],
            "rebinds": [r.as_dict(rel) for r in evidence],
            "provenance": {"producer": "tharkun", "method": "static-graph", "complete": complete,
                           **({"incomplete": project.incomplete[:5]} if project.incomplete else {})},
        })
        if len(records) >= _MAX_RECORDS:
            break
    return records


class _EmptyBinder:
    bindings: Dict[str, List[_Binding]] = {}


_EMPTY_BINDER = _EmptyBinder()


def regex_globals(entry_path: str, roots: Sequence[str] = (), *,
                  max_modules: Optional[int] = None) -> List[Dict[str, Any]]:
    """One record per compiled regex the import graph of the entry module defines or uses.

    ``{"kind": "regex_global", "pattern", "flags", "compiled_flags", "module", "line", "col", "bindings",
    "proven", "provenance"}``: ``pattern`` is the fully folded source, ``flags`` the EXPLICIT flags argument
    (``re.compile(pattern, flags)`` rebuilds it), ``compiled_flags`` what the compiled object reports (implicit
    ``re.UNICODE`` and inline ``(?i)`` bits included), ``module``/``line``/``col`` the defining ``re.compile``
    call, ``bindings`` every ``{module, name}`` of the graph whose namespace binds it (a re-exporting package
    lists several; an inline ``re.match('lit', x)`` call has none), ``proven`` that every binding is a proven
    module constant (see :func:`module_constants`).  A name that is not statically known is absent.  ``max_modules``
    is as for :func:`module_constants`.  Executes nothing."""
    project = _Project(entry_path, roots, max_modules)
    if project.entry not in project.mods or project.mods[project.entry].tree is None:
        return []
    analysis = _Analysis(project)
    rel = _rel(project)
    complete = not project.incomplete
    found: Dict[int, Dict[str, Any]] = {}
    objs: Dict[int, RegexConst] = {}
    for path, mod in project.mods.items():
        if mod.tree is None:
            continue
        ns = project.namespace(mod)
        for name, value in regex_consts(ns).items():
            rec = found.get(id(value))
            if rec is None:
                rec = {
                    "kind": "regex_global", "pattern": value.pattern, "flags": value.flags,
                    "compiled_flags": value.compiled_flags,
                    "module": rel(value.module) if value.module else rel(path), "line": value.line, "col": value.col,
                    "bindings": [], "proven": True,
                    "provenance": {"producer": "tharkun", "method": "static-fold", "complete": complete},
                }
                found[id(value)] = rec
                objs[id(value)] = value
            rec["bindings"].append({"module": rel(path), "name": name})
            evidence, _origin = analysis.rebinds(path, name)
            if evidence or not complete:
                rec["proven"] = False
    # inline uses: re.match('lit', x) and friends with a folded pattern
    for path, mod in project.mods.items():
        if mod.tree is None:
            continue
        ns = project.namespace(mod)
        ev = _Evaluator(ns, path)
        for node in ast.walk(mod.tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.args
                    and node.func.attr in ("match", "fullmatch", "search", "sub", "subn", "split", "findall",
                                           "finditer")
                    and isinstance(node.func.value, ast.Name) and isinstance(ns.get(node.func.value.id), ModuleRef)
                    and ns[node.func.value.id].name == "re"):
                continue
            try:
                pattern = ev.ev(node.args[0])
                flags = 0
                for kw in node.keywords:
                    if kw.arg == "flags":
                        flags = ev.ev(kw.value)
                if node.func.attr in ("match", "fullmatch", "search", "split", "findall", "finditer"):
                    pos = 2 if node.func.attr in ("match", "fullmatch", "search", "findall", "finditer") else 3
                    if len(node.args) > pos:
                        flags = ev.ev(node.args[pos])
                if isinstance(pattern, RegexConst):
                    pattern = pattern.pattern
                if not isinstance(pattern, str) or isinstance(flags, bool) or not isinstance(flags, int):
                    continue
                const = RegexConst(pattern, int(flags), path, node.lineno, node.col_offset)
                compiled = const.compiled_flags
            except (_NoValue, re.error, ValueError, OverflowError, RecursionError):
                continue
            if any(r["pattern"] == pattern and r["flags"] == int(flags) and r["line"] == node.lineno
                   and r["module"] == rel(path) for r in found.values()):
                continue
            found[id(node)] = {
                "kind": "regex_global", "pattern": pattern, "flags": int(flags), "compiled_flags": compiled,
                "module": rel(path), "line": node.lineno, "col": node.col_offset, "bindings": [], "proven": True,
                "provenance": {"producer": "tharkun", "method": "static-fold", "complete": complete},
            }
    records = sorted(found.values(), key=lambda r: (r["module"], r["line"], r["col"], r["pattern"]))
    for r in records:
        r["bindings"].sort(key=lambda b: (b["module"], b["name"]))
    return records[:_MAX_RECORDS]
