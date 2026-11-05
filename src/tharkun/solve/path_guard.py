"""Path-sensitive guard solving through copy-propagated assignments (docstring_parser ``rest._build_meta``, numpydoc ``_parse_item``).

``solve_condition`` solves one guard over its own free variables.  A guard inside a function usually names locals
(``key = args[0]``, ``m = PAT.match(key)``, ``version, desc, *_ = text.split(...) + [None, None]``) and is only reached
under the enclosing ``if`` arms and after the early ``if C: return/raise`` guards, so the witness has to land on the ENTRY
inputs and satisfy the whole path.

Here the path from the function entry to the guard is replayed symbolically: straight-line ``name = expr`` and tuple
unpacking (with the length the unpack itself needs) are substituted forward (copy propagation), the ancestor ``if`` arms
and the early-exit guards become literals, and the target guard (with the requested polarity) is the last literal.  The
conjunction is put into a disjunctive normal form and solved per ENTRY term (a parameter, or ``param[i]`` of a list
parameter, whose length is solved from the ``len(param)`` literals).  A literal is solved by the preimage / member engines:
set membership and equality from the set, ``str`` predicates and ``in`` from ``solve/preimage``, regex (non-)membership
and regex groups from ``solve/preimage/regex_members`` (a group value is embedded at the group's span in a member), ``len``
of a str from its bound, ``X.split(sep, k)`` lengths from the split preimage, ``X is None`` over ``list + [None, ...]``
from the length, ``inspect.cleandoc(X)`` through the fixed points of cleandoc.  A literal outside these is NAMED
(:class:`Unsupported`) and the caller keeps its previous route; nothing is guessed.  Every witness is then checked by
evaluating the WHOLE path (every literal, with the module constants resolved from the source) over the entry values."""
from __future__ import annotations

import ast
import collections
import collections.abc
import copy
import dataclasses
import datetime
import decimal
import enum
import fractions
import itertools
import numbers
import operator
import os
import pathlib
import re
import typing
import uuid
from pathlib import Path
from typing import Any, Callable

from tharkun.solve.loop_guard import _subst

_MAX_ALTERNATIVES = 16
_MUTATORS = frozenset({"append", "extend", "insert", "pop", "remove", "clear", "sort", "reverse", "update", "add",
                       "discard", "setdefault", "popitem"})
_STR_PREDICATES = frozenset({"endswith", "startswith", "isdigit", "isalpha", "isalnum", "isspace", "islower", "isupper",
                             "istitle", "isnumeric", "isdecimal", "isidentifier", "isascii", "isprintable"})
_CMP = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}
_NEG = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}
_FLIP = {"==": "==", "!=": "!=", "<": ">", "<=": ">=", ">": "<", ">=": "<="}
_OPS = {"==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}


class Unsupported(Exception):
    """A literal or statement outside what is solved; the message names it."""


class Infeasible(Exception):
    """The requested arm is REFUTED: no execution takes it.  ``constraint`` is the fact that refutes it (the proof)."""

    def __init__(self, constraint: str, **detail):
        super().__init__(constraint)
        self.constraint, self.detail = constraint, detail


# -- module constants --------------------------------------------------------------------------------------------------

# the stdlib modules a program's names may be taken from (imported here, statically: the core never imports by name)
_STDLIB_MODULES = {"typing": typing, "collections": collections, "collections.abc": collections.abc, "datetime": datetime,
                   "uuid": uuid, "decimal": decimal, "enum": enum, "numbers": numbers, "fractions": fractions,
                   "pathlib": pathlib, "re": re, "dataclasses": dataclasses}
_STDLIB_IMPORTS = frozenset(_STDLIB_MODULES)
_BUILTIN_CLASSES = {n: getattr(__builtins__, n) if not isinstance(__builtins__, dict) else __builtins__[n]
                    for n in ("dict", "list", "str", "int", "float", "bool", "bytes", "tuple", "set", "frozenset", "object", "complex")}


class _Constants:
    """Module-level constants of the source file and of the siblings it imports from (relative imports, or absolute ones
    inside its own package), evaluated by AST (sets/lists/tuples/dicts/numbers/strings, ``A | B`` on sets,
    ``re.compile(lit, flags)``, ``typing`` aliases such as ``Union[dict, list]`` and their ``__args__``); names imported from a
    small allow-list of standard-library modules (typing, collections.abc, datetime, uuid, decimal, enum, dataclasses ...) are
    the real objects; nothing of the program under analysis is executed.  A function that is exactly ``isinstance`` behind a
    ``try/except Exception: return False`` (dataclasses_json ``_isinstance_safe``) resolves to ``isinstance``."""

    def __init__(self, path: str, depth: int = 0):
        self.path, self.depth, self.cache = path, depth, {}
        self.assigns: dict[str, ast.expr] = {}
        self.funcs: dict[str, ast.FunctionDef] = {}
        self.imports: dict[str, tuple[str, str]] = {}      # local name -> (module, name there)
        self.stars: list[str] = []
        try:
            tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError):
            return
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                self.assigns[node.targets[0].id] = node.value
            elif isinstance(node, ast.FunctionDef):
                self.funcs[node.name] = node
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = ("." * node.level) + node.module
                for alias in node.names:
                    if alias.name == "*":
                        self.stars.append(module)
                    else:
                        self.imports[alias.asname or alias.name] = (module, alias.name)

    def _sibling(self, module: str):
        if self.depth >= 3:
            return None
        here = os.path.dirname(self.path)
        if module.startswith("."):
            rel = module[1:]
        elif module.split(".")[0] == os.path.basename(here):          # absolute import inside the package
            rel = module.split(".", 1)[1] if "." in module else ""
        else:
            return None
        target = os.path.join(here, rel.replace(".", os.sep) + ".py")
        return _Constants(target, self.depth + 1) if rel and os.path.exists(target) else None

    def get(self, name: str):
        if name in self.cache:
            return self.cache[name]
        if name in self.assigns:
            value = self.eval(self.assigns[name])
        elif name in self.imports:
            module, there = self.imports[name]
            if module in _STDLIB_IMPORTS:
                try:
                    value = getattr(_STDLIB_MODULES[module], there)
                except AttributeError:
                    raise KeyError(name)
            else:
                sib = self._sibling(module)
                if sib is None:
                    raise KeyError(name)
                value = sib.get(there)
        elif name == "isinstance" or (name in self.funcs and _is_isinstance_wrapper(self.funcs[name])):
            value = isinstance
        elif name in _BUILTIN_CLASSES:
            value = _BUILTIN_CLASSES[name]
        else:
            for module in self.stars:
                sib = self._sibling(module)
                if sib is not None:
                    try:
                        value = sib.get(name)
                        break
                    except KeyError:
                        continue
            else:
                raise KeyError(name)
        self.cache[name] = value
        return value

    def eval(self, node: ast.expr):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return self.get(node.id)
        if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            items = [self.eval(e) for e in node.elts]
            return {ast.Set: set, ast.List: list, ast.Tuple: tuple}[type(node)](items)
        if isinstance(node, ast.Dict):
            return {self.eval(k): self.eval(v) for k, v in zip(node.keys, node.values) if k is not None}
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.BitOr, ast.BitAnd, ast.Sub, ast.Add)):
            a, b = self.eval(node.left), self.eval(node.right)
            return {ast.BitOr: operator.or_, ast.BitAnd: operator.and_, ast.Sub: operator.sub, ast.Add: operator.add}[type(node.op)](a, b)
        if isinstance(node, ast.Subscript):                   # typing special forms only: Union[dict, list]
            base = self.eval(node.value)
            if getattr(base, "__module__", None) == "typing" or type(base).__module__ == "typing":
                index = self.eval(node.slice)
                return base[index]
            raise KeyError("subscript")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "re":
                value = getattr(re, node.attr, None)
                if isinstance(value, re.RegexFlag):
                    return value
            base = self.eval(node.value)
            if node.attr == "__args__" and type(base).__module__ == "typing":
                return base.__args__
            raise KeyError("attribute")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "re" and node.func.attr == "compile":
            args = [self.eval(a) for a in node.args]
            kwargs = {k.arg: self.eval(k.value) for k in node.keywords if k.arg}
            return re.compile(*args, **kwargs)
        raise KeyError(ast.dump(node)[:60])


def _is_isinstance_wrapper(fn: ast.FunctionDef) -> bool:
    """``def f(o, t): try: result = isinstance(o, t) / except Exception: return False / else: return result`` (any spelling
    whose only effect is ``isinstance(o, t)`` with an exception answering False)."""
    params = [a.arg for a in fn.args.args]
    if len(params) != 2 or len(fn.body) != 1 or not isinstance(fn.body[0], ast.Try):
        return False
    t = fn.body[0]
    calls = [n for n in ast.walk(ast.Module(body=t.body, type_ignores=[])) if isinstance(n, ast.Call)]
    if len(calls) != 1 or not (isinstance(calls[0].func, ast.Name) and calls[0].func.id == "isinstance"
                               and [getattr(a, "id", None) for a in calls[0].args] == params):
        return False
    handler_ok = len(t.handlers) == 1 and isinstance(t.handlers[0].type, ast.Name) and t.handlers[0].type.id == "Exception" \
        and len(t.handlers[0].body) == 1 and isinstance(t.handlers[0].body[0], ast.Return) \
        and isinstance(t.handlers[0].body[0].value, ast.Constant) and t.handlers[0].body[0].value.value is False
    return handler_ok and not t.finalbody


# -- the path from the entry to the guard ------------------------------------------------------------------------------

def _stores(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del))}


def _ends_in_exit(stmts: list[ast.stmt]) -> bool:
    return bool(stmts) and isinstance(stmts[-1], (ast.Return, ast.Raise))


def _chain_to(stmts: list[ast.stmt], lineno: int, trail: list):
    """Blocks from *stmts* down to the ``if`` at *lineno*: [(stmts_before, owner_if_or_None, arm)] or None."""
    for i, stmt in enumerate(stmts):
        if isinstance(stmt, ast.If) and stmt.lineno == lineno:
            return trail + [(stmts[:i], stmt, None)]
        if isinstance(stmt, ast.If):
            for arm, body in (("then", stmt.body), ("else", stmt.orelse)):
                found = _chain_to(body, lineno, trail + [(stmts[:i], stmt, arm)])
                if found:
                    return found
        elif isinstance(stmt, (ast.For, ast.While, ast.AsyncFor)):
            if any(getattr(n, "lineno", None) == lineno for n in ast.walk(stmt)):
                if isinstance(stmt, ast.For) and isinstance(stmt.target, ast.Name) and _is_finditer(stmt.iter) and not stmt.orelse:
                    found = _chain_to(stmt.body, lineno, trail + [(stmts[:i], stmt, "for")])
                    if found:
                        return found
                raise Unsupported("the guard is inside a loop (loop_guard's shape, a re.finditer loop, or not solved)")
        elif isinstance(stmt, ast.Try):
            found = _chain_to(stmt.body, lineno, trail + [(stmts[:i], None, "try")])
            if found:
                return found
        elif isinstance(stmt, ast.With):
            found = _chain_to(stmt.body, lineno, trail + [(stmts[:i], None, "with")])
            if found:
                return found
    return None


def _is_finditer(expr: ast.expr) -> bool:
    return isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "finditer"


class _Path:
    """One symbolic state along the path: copy-propagated aliases, names rebound on branches the path did not fix, and the
    literals (boolean expressions over entry terms with the truth the path requires) collected so far."""

    def __init__(self, params: set[str]):
        self.params = params
        self.aliases: dict[str, ast.expr] = {}
        self.unknown: set[str] = set()
        self.literals: list[tuple[ast.expr, bool]] = []
        self.used_alias = False

    def copy(self) -> "_Path":
        other = _Path(self.params)
        other.aliases, other.unknown = dict(self.aliases), set(self.unknown)
        other.literals, other.used_alias = list(self.literals), self.used_alias
        return other

    def sub(self, expr: ast.expr) -> ast.expr:
        names = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}
        self.used_alias |= bool(names & set(self.aliases))
        bad = names & self.unknown
        if bad:
            raise Unsupported(f"{sorted(bad)[0]} is rebound on a branch the path does not fix")
        return _subst(expr, self.aliases)

    def bind(self, name: str, expr: ast.expr):
        value = self.sub(expr)
        self.unknown.discard(name)
        self.aliases[name] = value

    def mark(self, name: str):
        self.aliases.pop(name, None)
        self.unknown.add(name)

    def assign(self, target: ast.expr, value: ast.expr):
        if isinstance(target, ast.Name):
            self.bind(target.id, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            elts = target.elts
            star = [i for i, e in enumerate(elts) if isinstance(e, ast.Starred)]
            if len(star) > 1 or not all(isinstance(e, (ast.Name, ast.Starred)) for e in elts):
                raise Unsupported("unpacking target shape")
            if isinstance(value, (ast.Tuple, ast.List)) and not star and len(value.elts) == len(elts):
                values = [self.sub(v) for v in value.elts]       # the right side is evaluated before any binding
                for e, v in zip(elts, values):
                    self.unknown.discard(e.id)
                    self.aliases[e.id] = v
                return
            rhs = self.sub(value)
            n = len(elts)
            op, length = (ast.GtE(), n - 1) if star else (ast.Eq(), n)
            self.literals.append((ast.Compare(left=ast.Call(func=ast.Name(id="len", ctx=ast.Load()), args=[rhs], keywords=[]),
                                              ops=[op], comparators=[ast.Constant(length)]), True))
            for i, e in enumerate(elts):
                if isinstance(e, ast.Name):
                    idx = i if not star or i < star[0] else i - n
                    self.unknown.discard(e.id)
                    self.aliases[e.id] = ast.Subscript(value=copy.deepcopy(rhs), slice=ast.Constant(idx), ctx=ast.Load())
                elif isinstance(e.value, ast.Name):
                    self.mark(e.value.id)
        elif isinstance(target, (ast.Attribute, ast.Subscript)):
            base = target
            while isinstance(base, (ast.Attribute, ast.Subscript)):
                base = base.value
            if isinstance(base, ast.Name):
                self.mark(base.id)                  # an in-place store: the object is no longer the aliased value
        else:
            raise Unsupported("assignment target shape")


def _steps(states: list[_Path], stmts: list[ast.stmt], loads: list[tuple[int, str]] | None = None) -> list[_Path]:
    for stmt in stmts:
        out: list[_Path] = []
        for state in states:
            out.extend(_step(state, stmt, loads))
        states = out
        if len(states) > 8:
            raise Unsupported("too many paths")
    return states


def _step(state: _Path, stmt: ast.stmt, loads: list[tuple[int, str]] | None = None) -> list[_Path]:
    """The states after *stmt* ([] when the statement leaves the function: that path never reaches the guard)."""
    if isinstance(stmt, ast.Expr):
        call = stmt.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in _MUTATORS \
                and isinstance(call.func.value, ast.Name):
            state.mark(call.func.value.id)
        return [state]
    if isinstance(stmt, ast.Assign):
        values = stmt.value
        for target in stmt.targets:
            state.assign(target, values)
        return [state]
    if isinstance(stmt, ast.AnnAssign):
        if stmt.value is not None and isinstance(stmt.target, ast.Name):
            state.bind(stmt.target.id, stmt.value)
        return [state]
    if isinstance(stmt, (ast.Return, ast.Raise, ast.Continue, ast.Break)):
        return []
    if isinstance(stmt, ast.If):
        out: list[_Path] = []
        for truth, body in ((True, stmt.body), (False, stmt.orelse)):
            branch = state.copy()
            branch.literals.append((branch.sub(stmt.test), truth))
            out.extend(_steps([branch], body, loads))
        live = {n for ln, n in loads if ln > (stmt.end_lineno or stmt.lineno)} if loads is not None else None
        if len(out) == 2 and _same_bindings(out[0], out[1], live) and len(out[0].literals) == len(out[1].literals) == len(state.literals) + 1:
            joined = state.copy()                  # both arms leave the same bindings: the paths re-join, test or not test
            joined.unknown = out[0].unknown | out[1].unknown
            joined.aliases = {k: v for k, v in out[0].aliases.items()
                              if k not in joined.unknown and (live is None or k in live)}
            joined.used_alias = out[0].used_alias or out[1].used_alias
            return [joined]
        return out
    for name in _stores(stmt) | {t.target.id for t in ast.walk(stmt) if isinstance(t, ast.AugAssign) and isinstance(t.target, ast.Name)}:
        state.mark(name)
    return [state]


def _same_bindings(a: _Path, b: _Path, live: set[str] | None = None) -> bool:
    """Equal aliases once the names either arm left unknown (rebound or stored into), and the names nothing reads afterwards
    (*live* is None: all names count), are set aside."""
    unknown = a.unknown | b.unknown
    ka = {k for k in a.aliases.keys() - unknown if live is None or k in live}
    kb = {k for k in b.aliases.keys() - unknown if live is None or k in live}
    return ka == kb and all(ast.dump(a.aliases[k]) == ast.dump(b.aliases[k]) for k in ka)


def _build_paths(fn, lineno: int) -> tuple[list[_Path], ast.If]:
    chain = _chain_to(fn.body, lineno, [])
    if chain is None:
        raise Unsupported("no `if` at that line in a function")
    params = {a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    if fn.args.vararg or fn.args.kwarg:
        raise Unsupported("*args/**kwargs parameters")
    states = [_Path(params)]
    loads = [(n.lineno, n.id) for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
    for before, owner, arm in chain:
        states = _steps(states, [s for s in before if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))], loads)
        if isinstance(owner, ast.For):                           # `for m in re.finditer(..)`: the first iteration's match
            for state in states:
                state.unknown.discard(owner.target.id)
                state.aliases[owner.target.id] = ast.Call(func=ast.Name(id="next", ctx=ast.Load()),
                                                          args=[state.sub(owner.iter)], keywords=[])
        elif owner is not None and arm is not None:             # an ancestor `if`: the arm fixes its test
            for state in states:
                state.literals.append((state.sub(owner.test), arm == "then"))
    if not states:
        raise Unsupported("no path reaches the guard")
    return states, chain[-1][1]


_ENV_ATTRS = {"sys": {"version_info", "platform", "maxsize", "byteorder"}, "os": {"name", "sep", "linesep", "pathsep"}}
_ENV_VERSION_ATTRS = {"major", "minor", "micro", "releaselevel", "serial"}
_ENV_NODES = (ast.Compare, ast.BoolOp, ast.UnaryOp, ast.Constant, ast.Tuple, ast.Attribute, ast.Name, ast.Subscript, ast.Load,
              ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot)


def environment_modules(expr: ast.expr) -> set[str]:
    """The ``sys``/``os`` names an environment-constant guard reads."""
    return {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}


def environment_truth(expr: ast.expr) -> bool | None:
    """The truth of a guard that reads only interpreter constants (``sys.version_info`` and its fields, ``sys.platform``,
    ``sys.maxsize``, ``sys.byteorder``, ``os.name``, ``os.sep``, ``os.linesep``, ``os.pathsep``) and literals, evaluated on
    THIS interpreter; ``None`` when the guard reads anything else.  No call is made and nothing of the program runs."""
    names = environment_modules(expr)
    if not names or not names <= set(_ENV_ATTRS):
        return None
    for node in ast.walk(expr):
        if not isinstance(node, _ENV_NODES):
            return None
        if isinstance(node, ast.Attribute):
            base = node.value
            if isinstance(base, ast.Name):
                if node.attr not in _ENV_ATTRS[base.id]:
                    return None
            elif not (isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id == "sys"
                      and base.attr == "version_info" and node.attr in _ENV_VERSION_ATTRS):
                return None
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, (ast.Constant, ast.Slice)):
            return None
    import os as _os
    import sys as _sys
    try:
        return bool(eval(compile(ast.fix_missing_locations(ast.Expression(body=copy.deepcopy(expr))), "<env>", "eval"),
                         {"__builtins__": {}, "sys": _sys, "os": _os}))
    except Exception:
        return None


# -- literals -> constraints per entry term ----------------------------------------------------------------------------

def _key(expr: ast.AST) -> str:
    return ast.unparse(expr)


def _is_len(expr) -> ast.expr | None:
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "len" and len(expr.args) == 1 \
            and not expr.keywords:
        return expr.args[0]
    return None


def _regex_call(expr, consts: _Constants, methods=("match", "search", "fullmatch")):
    """``PAT.match|search|fullmatch(T)`` or ``re.match|search|fullmatch(lit, T[, flags])`` -> (pattern, flags, how, T)."""
    if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr in methods):
        return None
    how, base = expr.func.attr, expr.func.value
    try:
        if isinstance(base, ast.Name) and base.id == "re" and len(expr.args) >= 2:
            flags = 0
            if len(expr.args) > 2:
                flags = int(consts.eval(expr.args[2]))
            for kw in expr.keywords:
                if kw.arg == "flags":
                    flags = int(consts.eval(kw.value))
            pattern = consts.eval(expr.args[0])
            if not isinstance(pattern, str):
                return None
            return pattern, flags, how, expr.args[1]
        if isinstance(base, ast.Name) and len(expr.args) == 1 and not expr.keywords:
            compiled = consts.get(base.id)
            if isinstance(compiled, re.Pattern) and isinstance(compiled.pattern, str):
                return compiled.pattern, int(compiled.flags), how, expr.args[0]
    except (KeyError, ValueError, TypeError):
        return None
    return None


def _group_call(expr, consts):
    """``<regex call>.group(g)`` -> (regex tuple, g)."""
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "group" \
            and len(expr.args) == 1 and isinstance(expr.args[0], ast.Constant) and not expr.keywords:
        rc = _regex_call(expr.func.value, consts)
        if rc is not None:
            return rc, expr.args[0].value
    return None


def _finditer_group(expr, consts):
    """``next(<re.finditer call>).group(g)`` (the loop variable's value in the first iteration) -> (pattern, flags, X, g)."""
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "group" \
            and len(expr.args) == 1 and isinstance(expr.args[0], ast.Constant) and not expr.keywords:
        nxt = expr.func.value
        if isinstance(nxt, ast.Call) and isinstance(nxt.func, ast.Name) and nxt.func.id == "next" and len(nxt.args) == 1:
            rc = _regex_call(nxt.args[0], consts, ("finditer",))
            if rc is not None:
                return rc[0], rc[1], rc[3], expr.args[0].value
    return None


class _Cons:
    """One constraint: kind + term (an AST) + payload.  ``term`` is any expression; entry terms are resolved later."""

    def __init__(self, kind: str, term: ast.expr, **payload):
        self.kind, self.term, self.p = kind, term, payload

    def __repr__(self):
        return f"{self.kind}({_key(self.term)}, {self.p})"


def _literal(expr: ast.expr, truth: bool, consts: _Constants) -> list[list[_Cons]]:
    """Alternatives (each a conjunction of constraints) for ``bool(expr) == truth``."""
    env = environment_truth(expr)
    if env is not None:                                 # `sys.version_info.minor == 6`: decided by the interpreter, no input constraint
        return [[]] if env == truth else []
    # len(...) comparisons and comparisons in general
    if isinstance(expr, ast.Compare) and len(expr.ops) == 1:
        op, left, right = expr.ops[0], expr.left, expr.comparators[0]
        if isinstance(op, (ast.Is, ast.IsNot)) and isinstance(right, ast.Constant) and right.value is None:
            is_none = isinstance(op, ast.Is) == truth                 # the literal requires `left is None`
            return _none_literal(left, is_none, consts)
        if isinstance(op, (ast.In, ast.NotIn)):
            member = isinstance(op, ast.In) == truth
            try:
                container = consts.eval(right)
            except KeyError:
                container = None
            if isinstance(container, (set, frozenset, list, tuple)):
                return [[_Cons("in", left, values=frozenset(container), truth=member)]]
            if isinstance(left, ast.Constant) and isinstance(left.value, str):
                return [[_Cons("contains", right, sub=left.value, truth=member)]]
            raise Unsupported(f"membership {_key(expr)}")
        if type(op) in _CMP:
            sym = _CMP[type(op)]
            if isinstance(left, ast.Constant) and not isinstance(right, ast.Constant):
                left, right, sym = right, left, _FLIP[sym]
            if not truth:
                sym = _NEG[sym]
            if isinstance(right, ast.Constant):
                inner = _is_len(left)
                n = right.value
                if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Add) and isinstance(inner.right, (ast.List, ast.Tuple)) \
                        and isinstance(n, int):
                    inner, n = inner.left, n - len(inner.right.elts)       # len(D + [c, ...]) == len(D) + k
                if inner is not None and isinstance(n, int):
                    return [[_Cons("len", inner, op=sym, n=n)]]
                if sym in ("==", "!=") and not isinstance(right.value, bool):
                    return [[_Cons("eq", left, value=right.value, truth=(sym == "=="))]]
        raise Unsupported(f"comparison {_key(expr)}")
    if isinstance(expr, ast.BoolOp):                    # only reached through DNF; kept for completeness
        raise Unsupported("boolean operator")
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr in _STR_PREDICATES \
            and all(isinstance(a, ast.Constant) for a in expr.args) and not expr.keywords:
        return [[_Cons("pred", expr.func.value, method=expr.func.attr, args=[a.value for a in expr.args], truth=truth)]]
    rc = _regex_call(expr, consts)
    if rc is not None:                                  # `if m:` over a match object
        return [[_Cons("re", rc[3], pattern=rc[0], flags=rc[1], how=rc[2], truth=truth)]]
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and len(expr.args) == 2 and not expr.keywords:
        try:
            callee = consts.get(expr.func.id)
        except KeyError:
            callee = None
        if callee is isinstance:                         # isinstance(T, C) or the try-wrapped `_isinstance_safe(T, C)`
            try:
                classes = consts.eval(expr.args[1])
            except KeyError:
                raise Unsupported(f"classes of {_key(expr)} are not constants")
            return [[_Cons("isinst", expr.args[0], classes=_class_tuple(classes), truth=truth)]]
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and len(expr.args) == 1 and not expr.keywords:
        try:
            callee = consts.get(expr.func.id)
        except KeyError:
            callee = None
        if callable(callee) and getattr(callee, "__module__", "") in _STDLIB_IMPORTS:
            return [[_Cons("check", expr.args[0], expr=expr, truth=truth)]]     # a pure stdlib predicate: checked, never a source of values
    fg = _finditer_group(expr, consts)
    if fg is not None:                                  # the text a finditer match captured: a match exists, and it is (non-)empty
        pattern, flags, base, g = fg
        return [[_Cons("re", base, pattern=pattern, flags=flags, how="search", truth=True),
                 _Cons("mgroup", base, pattern=pattern, flags=flags, group=g, truth=truth)]]
    if isinstance(expr, (ast.Name, ast.Subscript, ast.Call)):       # bare truthiness of a str/list term
        return [[_Cons("falsy", expr, truth=truth)]]
    raise Unsupported(f"literal {_key(expr)}")


def _none_literal(term: ast.expr, is_none: bool, consts: _Constants) -> list[list[_Cons]]:
    rc = _regex_call(term, consts)
    if rc is not None:
        return [[_Cons("re", rc[3], pattern=rc[0], flags=rc[1], how=rc[2], truth=not is_none)]]
    gc = _group_call(term, consts)
    if gc is not None:
        (pattern, flags, how, base), g = gc
        matched = _Cons("re", base, pattern=pattern, flags=flags, how=how, truth=True)
        if not is_none:
            return [[matched, _Cons("group", base, pattern=pattern, flags=flags, how=how, group=g, present=True)]]
        return [[_Cons("re", base, pattern=pattern, flags=flags, how=how, truth=False)],
                [matched, _Cons("group", base, pattern=pattern, flags=flags, how=how, group=g, present=False)]]
    # `(D + [None, ...])[i] is None`  <=>  i >= len(D)
    if isinstance(term, ast.Subscript) and isinstance(term.slice, ast.Constant) and isinstance(term.slice.value, int) \
            and isinstance(term.value, ast.BinOp) and isinstance(term.value.op, ast.Add) \
            and isinstance(term.value.right, (ast.List, ast.Tuple)) \
            and all(isinstance(e, ast.Constant) and e.value is None for e in term.value.right.elts):
        i, pad = term.slice.value, len(term.value.right.elts)
        d = term.value.left
        if 0 <= i < len(term.value.right.elts) + 64:
            # element i is None iff len(D) <= i (when i < len(D) it is a str, never None)
            return [[_Cons("len", d, op="<=" if is_none else ">", n=i if is_none else i)]]
    if isinstance(term, (ast.Name, ast.Subscript)):
        return [[_Cons("none", term, is_none=is_none)]]
    raise Unsupported(f"`is None` over {_key(term)}")


def _dnf(expr: ast.expr, truth: bool, consts: _Constants) -> list[list[_Cons]]:
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return _dnf(expr.operand, not truth, consts)
    if isinstance(expr, ast.BoolOp):
        parts = [_dnf(v, truth, consts) for v in expr.values]
        conj = isinstance(expr.op, ast.And) == truth            # And positive / Or negative: conjunction of the parts
        if conj:
            out = [list(itertools.chain.from_iterable(c)) for c in itertools.product(*parts)]
        else:
            out = list(itertools.chain.from_iterable(parts))
        if len(out) > _MAX_ALTERNATIVES:
            raise Unsupported("too many alternatives")
        return out
    if isinstance(expr, ast.Compare) and len(expr.ops) > 1:
        pieces = []
        left = expr.left
        for op, right in zip(expr.ops, expr.comparators):
            pieces.append(ast.Compare(left=left, ops=[op], comparators=[right]))
            left = right
        return _dnf(ast.BoolOp(op=ast.And(), values=pieces), truth, consts)
    return _literal(expr, truth, consts)


def _class_tuple(classes) -> tuple:
    """A class, a typing alias of one, or a (nested) tuple of them -> a flat tuple of real classes (typing aliases by origin)."""
    items = classes if isinstance(classes, tuple) else (classes,)
    out: list = []
    for c in items:
        if isinstance(c, tuple):
            out.extend(_class_tuple(c))
            continue
        origin = typing.get_origin(c)
        if origin is typing.Union:
            out.extend(_class_tuple(typing.get_args(c)))
            continue
        c = origin or c
        if c is None or c is type(None):
            c = type(None)
        if not isinstance(c, type):
            raise Unsupported(f"{c!r} is not a class")
        out.append(c)
    return tuple(out)


def _exemplars(cls: type) -> list[Any]:
    """Canonical instances of one class (stdlib classes and the builtin types), simplest first; [] when none is known.
    A user class is never instantiated."""
    cabc = collections.abc
    table: list[tuple[type, list]] = [
        (bool, [True, False]), (int, [0, 1]), (float, [0.0]), (complex, [0j]), (str, ["a"]), (bytes, [b"a"]),
        (type(None), [None]), (list, [[]]), (tuple, [()]), (dict, [{}]), (set, [set()]), (frozenset, [frozenset()]),
        (datetime.datetime, [datetime.datetime(2000, 1, 1)]), (datetime.date, [datetime.date(2000, 1, 1)]),
        (uuid.UUID, [uuid.UUID(int=0)]), (decimal.Decimal, [decimal.Decimal(0)]),
        (enum.Enum, [enum.Enum("E", "a").a]),
    ]
    out: list[Any] = []
    for known, values in table:
        if issubclass(known, cls) or (cls is object):
            out.extend(values)
    if cls is object:
        return out[:1]
    return out[:3]


def _isinst_seeds(classes: tuple, truth: bool) -> list[Any]:
    pool = [0, "a", None, [], {}, 0.0, b"a", (), True]
    if truth:
        out: list[Any] = []
        for c in classes:
            out.extend(_exemplars(c))
        return [v for v in out if isinstance(v, classes)]
    return [v for v in pool if not isinstance(v, classes)]


# -- solving one entry term ---------------------------------------------------------------------------------------------

def _seeds(c: _Cons, solve_sub: Callable[[ast.expr, list[_Cons]], list[str]], consts: _Constants) -> list[Any]:
    from tharkun.solve.contracts.rules import Refused
    from tharkun.solve.preimage import Eq, HOLE, Truth, preimages
    from tharkun.solve.preimage import regex_members as RM
    k = c.kind
    if k == "in":
        if c.p["truth"]:
            return sorted(c.p["values"], key=lambda v: (len(str(v)), str(v)))
        n = 0
        while "a" * n in c.p["values"]:
            n += 1
        return ["a" * n]
    if k == "eq":
        if c.p["truth"]:
            return [c.p["value"]]
        v = c.p["value"]
        return [v + "a"] if isinstance(v, str) else []
    if k == "falsy":
        return [""] if not c.p["truth"] else ["a"]
    if k == "none":
        return [None] if c.p["is_none"] else [0, "a", []]
    if k == "isinst":
        return _isinst_seeds(c.p["classes"], c.p["truth"])
    if k == "check":
        return []
    if k == "len":
        op, n = c.p["op"], c.p["n"]
        for m in range(0, max(n, 0) + 3):
            if _OPS[op](m, n):
                return ["a" * m]
        return []
    if k == "pred":
        got = preimages("str." + c.p["method"], Truth(c.p["truth"]), (HOLE, *c.p["args"]), budget=4)
        return [] if isinstance(got, Refused) else [g for g in got if isinstance(g, str)]
    if k == "contains":
        got = preimages("str.__contains__", Truth(c.p["truth"]), (HOLE, c.p["sub"]), budget=4)
        return [] if isinstance(got, Refused) else [g for g in got if isinstance(g, str)]
    if k == "re":
        return _regex_seeds(c.p["pattern"], c.p["flags"], c.p["how"], c.p["truth"], [], solve_sub, c.term)
    raise Unsupported(f"constraint {k}")


def _regex_seeds(pattern: str, flags: int, how: str, truth: bool, groups: list[tuple[Any, bool, list[str] | None]],
                 solve_sub, term) -> list[str]:
    from tharkun.solve.preimage import regex_members as RM
    if RM.refusal(pattern, flags):
        raise Unsupported(f"regex {pattern!r}: {RM.refusal(pattern, flags)}")
    compiled = re.compile(pattern, flags)
    match = getattr(compiled, how)
    if truth:
        members = list(RM.members(pattern, flags, how="match" if how != "search" else "search", limit=12))
        out: list[str] = []
        for text in members:
            m = match(text)
            if m is None:
                continue
            ok = True
            for g, present, values in groups:
                if not present:
                    ok &= m.group(g) is None
                    continue
                if m.group(g) is None:
                    ok = False
                    break
            if ok:
                out.append(text)
        return out if not groups else _embed(compiled, match, members, groups)
    # a non-member: members of the same pattern under flags that differ ('.' takes '\n' under DOTALL, '$' sits before '\n',
    # case under IGNORECASE ...), kept when the real pattern does not match them
    out = []
    for text in RM.flag_variants(pattern, flags, flags | re.DOTALL | re.MULTILINE, limit=16):
        if match(text) is None:
            out.append(text)
    for text in RM.flag_variants(pattern, flags, flags ^ re.IGNORECASE, limit=16):
        if match(text) is None:
            out.append(text)
    return list(dict.fromkeys(out))


def _embed(compiled, match, members: list[str], groups) -> list[str]:
    """Texts whose group g holds each candidate value: a member where the group took part, with its span replaced."""
    out: list[str] = []
    for text in members:
        m = match(text)
        if m is None:
            continue
        for g, present, values in groups:
            if not present:
                continue
            if m.group(g) is None:
                break
        else:
            variants = [text]
            for g, present, values in groups:
                if not present or not values:
                    continue
                variants = [v[:mm.start(g)] + val + v[mm.end(g):] for v in variants for mm in [match(v)] if mm is not None
                            and mm.group(g) is not None for val in values]
            for v in variants:
                mm = match(v)
                if mm is not None and all((mm.group(g) is not None) == present and (not present or not values or mm.group(g) in values)
                                          for g, present, values in groups):
                    out.append(v)
    return list(dict.fromkeys(out))


# -- the driver ---------------------------------------------------------------------------------------------------------

_SAFE_BUILTINS = {n: __builtins__[n] if isinstance(__builtins__, dict) else getattr(__builtins__, n)
                  for n in ("len", "str", "int", "float", "bool", "list", "tuple", "set", "frozenset", "dict", "any", "all",
                            "min", "max", "sorted", "isinstance", "next", "None", "True", "False", "bytes", "abs")}


def _eval(expr: ast.expr, env: dict[str, Any], consts: _Constants) -> Any:
    """Evaluate a pure path expression over the entry values and the source's constants; failure -> raises."""
    scope: dict[str, Any] = {"re": re, "__builtins__": _SAFE_BUILTINS}
    import inspect as _inspect
    scope["inspect"] = _inspect
    for name in {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}:
        if name in env:
            scope[name] = env[name]
        elif name not in scope and name not in _SAFE_BUILTINS:
            scope[name] = consts.get(name)
    return eval(compile(ast.fix_missing_locations(ast.Expression(body=copy.deepcopy(expr))), "<path>", "eval"), scope)


def _entry_of(term: ast.expr, params: set[str]):
    if isinstance(term, ast.Name) and term.id in params:
        return (term.id, None)
    if isinstance(term, ast.Subscript) and isinstance(term.value, ast.Name) and term.value.id in params \
            and isinstance(term.slice, ast.Constant) and isinstance(term.slice.value, int) and term.slice.value >= 0:
        return (term.value.id, term.slice.value)
    return None


def _derived(term: ast.expr, consts: _Constants):
    """('group', regex, g, base) | ('cleandoc', base) | ('split', base, sep_args, kwargs) | ('slice', regex, base) | None"""
    gc = _group_call(term, consts)
    if gc is not None:
        return ("group", gc[0], gc[1], gc[0][3])
    if isinstance(term, ast.Subscript) and isinstance(term.slice, ast.Slice) and term.slice.upper is None \
            and term.slice.step is None and isinstance(term.slice.lower, ast.Call) \
            and isinstance(term.slice.lower.func, ast.Attribute) and term.slice.lower.func.attr == "start" \
            and not term.slice.lower.args:
        rc = _regex_call(term.slice.lower.func.value, consts)        # `Y[m.start():]` with m = a regex match over Y
        if rc is not None and _key(rc[3]) == _key(term.value):
            return ("slice", rc, term.value)
    if isinstance(term, ast.Call) and isinstance(term.func, ast.Attribute) and isinstance(term.func.value, ast.Name) \
            and term.func.value.id == "inspect" and term.func.attr == "cleandoc" and len(term.args) == 1 and not term.keywords:
        return ("cleandoc", term.args[0])
    if isinstance(term, ast.Call) and isinstance(term.func, ast.Attribute) and term.func.attr == "split" \
            and all(isinstance(a, ast.Constant) for a in term.args) \
            and all(k.arg and isinstance(k.value, ast.Constant) for k in term.keywords):
        return ("split", term.func.value, [a.value for a in term.args], {k.arg: k.value.value for k in term.keywords})
    return None


def _candidates(tkey: str, terms: dict[str, ast.expr], by_term: dict[str, list[_Cons]], consts: _Constants,
                depth: int = 0) -> list[Any]:
    """Seed values for the term *tkey* from its own constraints and from the constraints on terms derived from it."""
    if depth > 4:
        raise Unsupported("derived terms nest too deeply")
    from tharkun.solve.contracts.rules import Refused
    from tharkun.solve.preimage import Eq, HOLE, preimages
    term = terms[tkey]
    out: list[Any] = []
    children = [(k, _derived(t, consts)) for k, t in terms.items() if _derived(t, consts) is not None
                and _key(_derived(t, consts)[-1] if _derived(t, consts)[0] != "split" else _derived(t, consts)[1]) == tkey]
    def priority(c: _Cons) -> int:        # generators first: a value a literal demands before the pool values that merely avoid a class
        if c.kind in ("in", "eq", "re") or (c.kind == "isinst" and c.p["truth"]) or (c.kind == "none" and c.p["is_none"]):
            return 0
        return 1 if c.kind in ("pred", "contains") else 2
    # group children, per regex call
    for c in sorted(by_term.get(tkey, []), key=priority):
        if c.kind == "re":
            groups = []
            for ck, d in children:
                if d[0] == "group" and d[1][:3] == (c.p["pattern"], c.p["flags"], c.p["how"]):
                    present = any(x.kind == "group" and x.p["group"] == d[2] and x.p["present"] for x in by_term.get(tkey, []))
                    absent = any(x.kind == "group" and x.p["group"] == d[2] and not x.p["present"] for x in by_term.get(tkey, []))
                    values = _candidates(ck, terms, by_term, consts, depth + 1) if by_term.get(ck) else None
                    if absent and not present:
                        groups.append((d[2], False, None))
                    elif present or values:
                        groups.append((d[2], True, values))
            out.extend(_regex_seeds(c.p["pattern"], c.p["flags"], c.p["how"], c.p["truth"], groups, None, term)
                       if c.p["truth"] else _regex_seeds(c.p["pattern"], c.p["flags"], c.p["how"], False, [], None, term))
        elif c.kind not in ("group", "mgroup"):    # a group's presence only shapes the regex member that holds it
            out.extend(_seeds(c, None, consts))
    for ck, d in children:
        if d[0] == "cleandoc" and by_term.get(ck):
            import inspect as _inspect
            out.extend(v for v in _candidates(ck, terms, by_term, consts, depth + 1) if isinstance(v, str) and _inspect.cleandoc(v) == v)
        elif d[0] == "slice":                  # Y[m.start():] equals a candidate S when the match over S starts at 0
            matcher = getattr(re.compile(d[1][0], d[1][1]), d[1][2])
            for v in _candidates(ck, terms, by_term, consts, depth + 1):
                m = matcher(v) if isinstance(v, str) else None
                if m is not None and m.start() == 0:
                    out.append(v)
        elif d[0] == "split":
            for c in by_term.get(ck, []):
                if c.kind != "len":
                    raise Unsupported(f"split result constrained by {c.kind}")
                m = next((m for m in range(1, max(c.p["n"], 0) + 3) if _OPS[c.p["op"]](m, c.p["n"])), None)   # split gives >= 1 part
                if m is None:
                    continue
                value = [chr(ord("a") + i) for i in range(m)]
                got = preimages("str.split", Eq(value), (HOLE, *d[2]), d[3] or None, budget=4)
                if not isinstance(got, Refused):
                    out.extend(g for g in got if isinstance(g, str))
    seen: set[str] = set()
    unique = []
    for v in out:
        if repr(v) not in seen:
            seen.add(repr(v))
            unique.append(v)
    return unique[:24]


def solve_path_guard(fn, lineno: int, desired: str, source_path: str) -> tuple[list[dict[str, Any]], str]:
    """Witnesses (entry parameter dicts) for the ``if`` at *lineno* under *desired* ('then'|'else'), and the guard text.

    Raises :class:`Unsupported` (naming the literal) outside the solved shapes."""
    consts = _Constants(source_path)
    states, node = _build_paths(fn, lineno)
    params = states[0].params
    annotations = {a.arg: (ast.unparse(a.annotation) if a.annotation is not None else "")
                   for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    local_names = _stores(fn) - params
    results: list[dict[str, Any]] = []
    last: Unsupported | None = None
    refuted: list[Infeasible] = []
    text = ""
    for state in states:
        try:
            guard = state.sub(node.test)
            text = _key(guard)
            for lit, lit_truth in state.literals:             # a path condition decided by the interpreter and false: unreachable
                if (environment_truth(lit) is not None) and environment_truth(lit) != lit_truth:
                    raise Infeasible("a condition on the path to the guard reads only interpreter constants and is false on this "
                                     f"interpreter: {_key(lit)} is {environment_truth(lit)}, the path needs {lit_truth}",
                                     guard=text, arm=desired, kind="environment")
            target = _dnf(guard, desired == "then", consts)         # the requested arm on its own: refuted by a pattern?
            reasons = [_refutation(alt) for alt in target]
            if target and all(reasons):
                raise Infeasible(reasons[0][0], guard=text, arm=desired, **reasons[0][1])
            literals = state.literals + [(guard, desired == "then")]
            if not (state.used_alias or state.literals or _has_entry_subscript(guard, params) or _has_stdlib_isinstance(guard, consts)):
                raise Unsupported("the guard is over parameters and nothing precedes it")
            free = {n.id for lit, _ in literals for n in ast.walk(lit) if isinstance(n, ast.Name)}
            if free & local_names:
                raise Unsupported(f"unresolved local {sorted(free & local_names)[0]}")
            alternatives = [[]]
            for expr, truth in literals:
                options = _dnf(expr, truth, consts)
                alternatives = [a + o for a in alternatives for o in options]
                if len(alternatives) > _MAX_ALTERNATIVES:
                    raise Unsupported("too many alternatives")
            if not alternatives:                  # an interpreter-constant literal of the path is false: no execution gets here
                raise Infeasible("a condition on the path to the guard reads only interpreter constants and is false on this interpreter",
                                 guard=text, arm=desired, kind="environment")
            for conj in alternatives:
                try:
                    for w in _solve_conjunction(conj, params, annotations, literals, consts):
                        if w not in results:
                            results.append(w)
                except Unsupported as exc:
                    last = last or exc
        except Unsupported as exc:
            last = last or exc             # the first path's reason is the informative one
        except Infeasible as exc:
            refuted.append(exc)
    if not results:
        if refuted and len(refuted) == len(states):
            raise refuted[0]                   # every path to the guard is refuted: the arm is unreachable
        raise last or Unsupported("no witness satisfies the path")
    return results, text


def _refutation(conj: list[_Cons]) -> tuple[str, dict] | None:
    """A fact about the regexes in *conj* that makes it unsatisfiable: ``group(0)`` of a ``finditer`` match is empty, but every
    match of the pattern has length >= 1 (its parse tree's minimum length: a required first character, a mandatory repeat)."""
    from tharkun.solve.preimage import regex_members as RM
    for c in conj:
        if c.kind == "mgroup" and not c.p["truth"] and c.p["group"] == 0:
            lo = RM.min_match_length(c.p["pattern"], c.p["flags"])
            if lo is not None and lo >= 1:
                return (f"every match of {c.p['pattern']!r} has length >= {lo} (minimum length of its parse tree), "
                        f"so a match's group(0) is never empty", {"pattern": c.p["pattern"], "flags": int(c.p["flags"]), "min_length": lo})
    return None


def _has_entry_subscript(guard: ast.expr, params: set[str]) -> bool:
    return any(isinstance(n, ast.Subscript) and _entry_of(n, params) is not None for n in ast.walk(guard))


def _has_stdlib_isinstance(guard: ast.expr, consts: _Constants) -> bool:
    """An ``isinstance`` (or its try-wrapped twin) over a class outside the builtins (an ABC, ``datetime``, ``Enum``, a typing
    alias): the ordinary route has no value for those."""
    for n in ast.walk(guard):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and len(n.args) == 2:
            try:
                if consts.get(n.func.id) is isinstance and (not isinstance(n.args[1], (ast.Name, ast.Tuple)) or any(
                        c.__module__ != "builtins" for c in _class_tuple(consts.eval(n.args[1])))):
                    return True
            except (KeyError, Unsupported):
                continue
    return False


def _solve_conjunction(conj: list[_Cons], params: set[str], annotations: dict[str, str], literals, consts: _Constants) -> list[dict[str, Any]]:
    terms: dict[str, ast.expr] = {}
    by_term: dict[str, list[_Cons]] = {}

    def register(t: ast.expr):
        terms.setdefault(_key(t), t)
        d = _derived(t, consts)
        if d is not None:
            register(d[-1] if d[0] != "split" else d[1])

    for c in conj:
        register(c.term)
        by_term.setdefault(_key(c.term), []).append(c)
    entries: dict[str, ast.expr] = {}
    lengths: dict[str, list[_Cons]] = {}
    for k, t in terms.items():
        if _derived(t, consts) is not None:
            continue
        if _entry_of(t, params) is None:
            raise Unsupported(f"term {k} is not a parameter, parameter[i] or a derived value")
        entries[k] = t
    # list parameters: subscripted elements, or a `List`/`list`/`Sequence`/`Tuple`-annotated name
    lists: dict[str, dict] = {}
    for k, t in entries.items():
        name, idx = _entry_of(t, params)
        if idx is not None:
            lists.setdefault(name, {"n": 0, "cons": []})["n"] = max(lists[name]["n"] if name in lists else 0, idx + 1)
        elif re.search(r"\b(List|list|Sequence|Tuple|tuple)\b", annotations.get(name, "")):
            lists.setdefault(name, {"n": 0, "cons": []})
    for k, t in entries.items():
        name, idx = _entry_of(t, params)
        if idx is None and name in lists:
            for c in by_term.get(k, []):
                if c.kind == "len":
                    lists[name]["cons"].append(c)
                elif c.kind == "falsy":
                    lists[name]["cons"].append(_Cons("len", t, op="==" if not c.p["truth"] else ">", n=0))
                else:
                    raise Unsupported(f"list parameter {name} constrained by {c.kind}")
    # candidate values per entry term
    for k, t in entries.items():
        name, idx = _entry_of(t, params)
        if idx is not None and not re.search(r"\b(List|list|Sequence|Tuple|tuple)\b", annotations.get(name, "")):
            if any(not isinstance(v, str) for c in by_term.get(k, []) for v in (c.p.get("values") or [c.p.get("value", "")])):
                raise Unsupported(f"element type of {name} is not known to be str")
    domains: list[tuple[tuple[str, int | None], list[Any]]] = []
    for k, t in entries.items():
        name, idx = _entry_of(t, params)
        if name in lists and idx is None:
            continue
        cands = _candidates(k, terms, by_term, consts)
        if not cands:
            raise Unsupported(f"no value derived for {k}: " + "; ".join(map(repr, by_term.get(k, []))))
        domains.append(((name, idx), cands))
    # list lengths
    shapes: dict[str, list[int]] = {}
    for name, info in lists.items():
        sizes = []
        for n in range(info["n"], info["n"] + 6):
            if all(_OPS[c.p["op"]](n, c.p["n"]) for c in info["cons"] if c.kind == "len"):
                sizes.append(n)
        sizes = sizes[:2]
        if not sizes:
            raise Unsupported(f"no length for {name}")
        shapes[name] = sizes
    witnesses: list[dict[str, Any]] = []
    scalar = [(slot, cands) for slot, cands in domains if slot[1] is None]
    elems = [(slot, cands) for slot, cands in domains if slot[1] is not None]
    for sizes in itertools.product(*[shapes[n] for n in shapes]) if shapes else [()]:
        size_of = dict(zip(shapes, sizes))
        for pick in itertools.product(*[c for _, c in domains]) if domains else [()]:
            env: dict[str, Any] = {name: None for name in lists}
            for name, n in size_of.items():
                env[name] = ["a"] * n
            for (slot, _), value in zip(domains, pick):
                name, idx = slot
                if idx is None:
                    env[name] = copy.deepcopy(value)
                else:
                    env[name][idx] = copy.deepcopy(value)
            if _holds_path(literals, env, consts):
                witnesses.append(env)
                if len(witnesses) >= 3:
                    return witnesses
    if not witnesses:
        raise Unsupported("no combination of the derived values satisfies the whole path")
    return witnesses


def _holds_path(literals, env: dict[str, Any], consts: _Constants) -> bool:
    try:
        return all(bool(_eval(expr, env, consts)) == truth for expr, truth in literals)
    except Exception:
        return False
