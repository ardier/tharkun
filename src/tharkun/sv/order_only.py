"""The compared-values rule: an "order-only small model" TRUE for bounded list programs (docs/SV.md "Order-only small model").

The claim.  Take a harness whose draws are (1) lists of ``nondet_int`` elements, (2) scalars, and whose very first
statements after the draws are pure early-return guards that bound every list's length (``if not (len(arr) <= 5 ...): return``).
If every drawn element value -- and every drawn scalar that is itself compared with elements -- reaches ONLY order and
equality operations (``< <= > >= == !=``, ``in`` over a list/tuple, ``sorted``/``min``/``max`` without a key, copying,
returning, storing in a list/tuple), then the program's behaviour is a function of the values' relative order and ties:
two draw vectors whose values are order-isomorphic (same weak ordering) run through exactly the same control flow and
raise exactly the same exception class.  So the finitely many weak orderings of the (at most ``MAX_VALUES``) values, as
ranks ``0..k-1``, for every length vector the guards admit, cover EVERY input: no violation among them is a TRUE.  A
scalar that is an INDEX (kth's ``k``: used in arithmetic, as an index, in a ``range``) is not a value: it is enumerated
concretely over the range the guards admit, together with the lengths.

What decides this is a DATAFLOW check, never a name list (owner 1408: never key on spelling).  ``analyze`` types every
expression of the whole module with a small unification-based taint analysis:

* a type node is ``val`` (a drawn element value), ``seq`` (a list/tuple, with an element node), ``clean`` (anything that
  holds no drawn value) or unbound;
* ``val`` unified with ``clean`` -- or a ``seq`` whose elements are ``val`` unified with ``clean`` -- is a REFUSAL: a value
  that meets a constant, a length, an index, a dict key, ... has left the order-only fragment.  Unification is
  bidirectional, so aliasing (``b = a``, arguments, returns) can never hide a value in a container the analysis thinks is clean;
* operations that read a value as a number -- arithmetic, bit operations, ``abs``/``round``/``divmod``, ``int``/``str``/
  ``format``/``repr``/``hash``/``id``, a value as an index, a slice bound, a ``range`` argument, a dict or set key or
  element, a truthiness test of a value, ``is`` on a value, an unknown call, an f-string -- are *requirements* checked
  after every unification (``untainted`` / ``not_val``), so flow-insensitivity only ever refuses more;
* anything the analysis does not model (classes, lambdas, ``global``, ``with``, ``del``, generators, star arguments, a
  call through a variable, an import outside a small allow-list, ``eval``/``getattr``...) is a refusal, anywhere in the
  module, tainted or not.

Soundness of the bound.  The guards are accepted as a bound only when they sit in a contiguous prelude of pure
statements (draw assignments, ``n = len(arr)``-style linear aliases, ``if <pure test>: return``) -- nothing between the
draws and the guards can raise or branch on anything else -- and only their *necessary* atoms are used (the conjuncts that
must hold for the program to go on), so an atom that is not understood merely widens the enumerated set.  A list with no
upper bound, an index scalar with no finite range, or a plan above ``MAX_RUNS`` is a refusal (UNKNOWN), never a guess.

The enumeration sits behind ``OrderEnumerator`` (one method: ``vectors(plan)``); ``WeakOrderings`` is the default.  Runs go
through ``_order_child.py`` (one process, one fork per vector, a fresh module per run); the scheduler re-runs any violating
vector through its ordinary fresh child before FALSE, and checks the determinism certificate before TRUE
(``sv/scheduler.py``, stage "order-only small model").

Files: this module (analysis, plan, enumeration, batch runner), ``sv/_order_child.py`` (the batch child),
``sv/scheduler.py::run`` (the stage), ``sv/contracts.py::COMPONENTS`` (the contract row), ``docs/SV.md``.
"""
from __future__ import annotations

import ast
import builtins
import functools
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Iterator, Protocol

#: most values (list elements + value scalars) whose weak orderings are enumerated: 7 -> 47293 orderings
MAX_VALUES = 7
#: most runs one plan may ask for; a bigger plan is refused (UNKNOWN), not truncated
MAX_RUNS = 80_000
#: most values an index scalar may take for one length vector
MAX_INDEX_DOMAIN = 64
#: vectors per child process (a chunk that does not finish inside the allowance is lost: keep it small)
CHUNK = 500
#: CPU seconds one run may use before it counts as incomplete
RUN_CPU_SECONDS = 5

#: imports that cannot carry a drawn value anywhere but through an explicit call (and any such call is a refusal)
_ALLOWED_IMPORTS = frozenset({"typing", "math", "collections", "itertools", "functools", "string", "operator",
                              "dataclasses", "enum", "heapq", "bisect", "copy"})
_REFUSED_NAMES = frozenset({"eval", "exec", "compile", "globals", "locals", "vars", "getattr", "setattr", "delattr",
                            "__import__", "open", "input", "breakpoint", "dir", "memoryview", "super"})
_BUILTINS = frozenset(n for n in vars(builtins) if not n.startswith("_")) | {"__name__", "__file__", "__doc__"}


class Refused(Exception):
    """The program is outside the order-only fragment (or its bound is not provable): the reason, in words."""


# --------------------------------------------------------------------------------------- type nodes

class _T:
    __slots__ = ("parent", "tag", "elem")

    def __init__(self, tag=None, elem=None):
        self.parent, self.tag, self.elem = None, tag, elem


def _find(t: _T) -> _T:
    while t.parent is not None:
        if t.parent.parent is not None:
            t.parent = t.parent.parent
        t = t.parent
    return t


class _Graph:
    def __init__(self):
        self.reqs: list = []        # (kind, node, why), checked once unification is done
        self.deferred: list = []    # shape-polymorphic operations, resolved to a fixpoint

    def new(self):
        return _T()

    def clean(self):
        return _T("clean")

    def val(self):
        return _T("val")

    def seq(self, elem=None):
        return _T("seq", elem if elem is not None else _T())

    def unify(self, a: _T, b: _T, why: str = "") -> None:
        a, b = _find(a), _find(b)
        if a is b:
            return
        if a.tag is None:
            a.parent = b
            return
        if b.tag is None:
            b.parent = a
            return
        if a.tag == b.tag:
            b.parent = a
            if a.tag == "seq":
                self.unify(a.elem, b.elem, why)
            return
        tags = {a.tag, b.tag}
        if tags == {"val", "clean"} or tags == {"val", "seq"}:
            raise Refused(f"a drawn value meets something that is not a drawn value ({why or 'unification'})")
        # clean ~ seq: the shape is forgotten, so the elements must hold no drawn value
        clean, seq = (a, b) if a.tag == "clean" else (b, a)
        seq.parent = clean
        self.reqs.append(("untainted", seq.elem, why or "a list merged with a non-list"))

    def untainted(self, t: _T, why: str) -> None:
        self.reqs.append(("untainted", t, why))

    def not_val(self, t: _T, why: str) -> None:
        self.reqs.append(("not_val", t, why))


def _tainted(t: _T, seen=None) -> bool:
    t = _find(t)
    if t.tag == "val":
        return True
    if t.tag == "seq":
        seen = seen if seen is not None else set()
        if id(t) in seen:
            return False
        seen.add(id(t))
        return _tainted(t.elem, seen)
    return False


# --------------------------------------------------------------------------------------- the plan

@dataclass
class Draw:
    name: str
    kind: str                # "list" (nondet_list(nondet_int)) | "val" | "index" (a scalar int) | "bool"


@dataclass
class Plan:
    draws: list                                  # [Draw] in program order
    len_upper: dict                              # list name -> largest length
    index_atoms: dict                            # index scalar name -> [(coefs, const, op)], lengths-only atoms
    note: str = ""

    @property
    def lists(self):
        return [d for d in self.draws if d.kind == "list"]

    @property
    def value_scalars(self):
        return [d for d in self.draws if d.kind == "val"]


def _op_name(op) -> str | None:
    return {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}.get(type(op))


_NEGATE = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}


# --------------------------------------------------------------------------------------- the analyzer

@dataclass
class _Func:
    node: ast.FunctionDef
    params: list
    defaults: dict
    ret: _T
    locals: dict = field(default_factory=dict)


class _Analyzer:
    def __init__(self, tree: ast.Module, modes: dict[str, str]):
        self.tree, self.g = tree, _Graph()
        self.modes = modes                    # scalar draw name -> "val" | "index"
        self.sv_modules: set = set()          # local names bound to the _sv_verifier module
        self.sv_funcs: dict = {}              # local name -> nondet_* it names
        self.imports: set = set()             # local names bound to an allowed import
        self.module_vars: dict = {}
        self.funcs: dict[str, _Func] = {}
        self.func_stack: list = []
        self.comp_stack: list = []
        self.harness: list | None = None      # body holding the draws
        self.harness_func: str | None = None
        self.draws: list = []
        self.draw_stmts: set = set()
        self.draw_nodes: dict = {}            # id(stmt) -> Draw
        self.prelude: list = []

    # ------------------------------------------------------------------ top level
    def run(self):
        self._imports()
        self._functions()
        self._find_harness()
        for stmt in self.tree.body:
            self.stmt(stmt)
        self._fixpoint()
        for kind, node, why in self.g.reqs:
            if kind == "untainted" and _tainted(node):
                raise Refused(f"a drawn value reaches an operation that is not order-only: {why}")
            if kind == "not_val" and _find(node).tag == "val":
                raise Refused(f"a drawn value reaches an operation that is not order-only: {why}")

    def _imports(self):
        for node in self.tree.body:
            if isinstance(node, ast.Import):
                for a in node.names:
                    local = (a.asname or a.name).split(".")[0]
                    if a.name == "_sv_verifier":
                        self.sv_modules.add(a.asname or a.name)
                    elif a.name.split(".")[0] in _ALLOWED_IMPORTS:
                        self.imports.add(local)
                    else:
                        raise Refused(f"import of {a.name!r} is outside the order-only fragment")
            elif isinstance(node, ast.ImportFrom):
                if node.level or node.module is None:
                    raise Refused("relative import")
                for a in node.names:
                    if a.name == "*":
                        raise Refused("star import")
                    if node.module == "_sv_verifier":
                        self.sv_funcs[a.asname or a.name] = a.name
                    elif node.module.split(".")[0] in _ALLOWED_IMPORTS:
                        self.imports.add(a.asname or a.name)
                    else:
                        raise Refused(f"import from {node.module!r} is outside the order-only fragment")
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)) and node not in self.tree.body:
                raise Refused("import below module level")

    def _functions(self):
        stored = {t.id for n in self.tree.body for t in _store_names(n)}
        # module-level names: assignments (also under if/for/while/try at module level) and except-handler names
        self._module_stored = {t.id for n in self.tree.body if not isinstance(n, ast.FunctionDef)
                               for m in ast.walk(n) for t in _store_names(m)}
        self._module_stored |= {m.name for n in self.tree.body if not isinstance(n, ast.FunctionDef)
                                for m in ast.walk(n) if isinstance(m, ast.ExceptHandler) and m.name}
        for node in self.tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            a = node.args
            if node.decorator_list or a.vararg or a.kwarg or a.kwonlyargs or a.posonlyargs:
                raise Refused(f"function {node.name}: decorators, *args, keyword-only parameters")
            if node.name in stored or node.name in self.funcs:
                raise Refused(f"function {node.name} is rebound")
            params = [x.arg for x in a.args]
            defaults = dict(zip(params[len(params) - len(a.defaults):], a.defaults))
            self.funcs[node.name] = _Func(node, params, defaults, self.g.new(),
                                         {p: self.g.new() for p in params})
        for name, f in self.funcs.items():
            for n in ast.walk(f.node):
                if isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await, ast.Global, ast.Nonlocal, ast.Lambda,
                                  ast.ClassDef, ast.AsyncFunctionDef, ast.NamedExpr)):
                    raise Refused(f"function {name}: {type(n).__name__}")
                if isinstance(n, ast.FunctionDef) and n is not f.node:
                    raise Refused(f"function {name}: nested function")
            for t in (t for stmt in f.node.body for n in ast.walk(stmt) for t in _store_names(n)):
                f.locals.setdefault(t.id, self.g.new())
            for n in ast.walk(f.node):
                if isinstance(n, ast.ExceptHandler) and n.name:
                    f.locals.setdefault(n.name, self.g.new())

    # ------------------------------------------------------------------ the draw prelude
    def _sv_ref(self, node) -> str | None:
        """The ``nondet_*`` name an expression refers to, else None."""
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in self.sv_modules:
            return node.attr
        if isinstance(node, ast.Name) and node.id in self.sv_funcs:
            return self.sv_funcs[node.id]
        return None

    def _draw_of(self, stmt) -> Draw | None:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.Call) and not stmt.value.keywords):
            return None
        call, name = stmt.value, stmt.targets[0].id
        which = self._sv_ref(call.func)
        if which == "nondet_list" and len(call.args) == 1 and self._sv_ref(call.args[0]) == "nondet_int":
            return Draw(name, "list")
        if which == "nondet_int" and not call.args:
            return Draw(name, self.modes.get(name, "val"))
        if which == "nondet_bool" and not call.args:
            return Draw(name, "bool")
        return None

    def _find_harness(self):
        owner = None
        for node in [self.tree] + [f.node for f in self.funcs.values()]:
            body = node.body
            i = 0
            while i < len(body) and self._draw_of(body[i]) is None:
                if node is self.tree and not isinstance(body[i], (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Expr)) \
                        and not (isinstance(body[i], ast.Assign) and isinstance(body[i].value, ast.Constant)):
                    break
                if node is not self.tree and not (isinstance(body[i], ast.Expr) and isinstance(body[i].value, ast.Constant)):
                    break
                i += 1
            if i < len(body) and self._draw_of(body[i]) is not None:
                if owner is not None:
                    raise Refused("draws in more than one body")
                owner = (node, i)
        if owner is None:
            raise Refused("no draw prelude: the draws must be the first statements of the harness body")
        node, i = owner
        body = node.body
        j = i
        while j < len(body) and self._draw_of(body[j]) is not None:
            self.draws.append(self._draw_of(body[j]))
            self.draw_nodes[id(body[j])] = self.draws[-1]
            j += 1
        if len({d.name for d in self.draws}) != len(self.draws):
            raise Refused("a draw name is reused")
        self.harness, self.prelude_start, self.prelude_end = body, i, j
        if node is not self.tree:
            if node.args.args:
                raise Refused("the harness function takes parameters")
            self.harness_func = node.name
            calls = [n for n in ast.walk(self.tree) if isinstance(n, ast.Name) and n.id == node.name]
            if len(calls) != 1:
                raise Refused("the harness function is called other than once")
            top = [s for s in self.tree.body if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                   and isinstance(s.value.func, ast.Name) and s.value.func.id == node.name and not s.value.args]
            guarded = [s for t in self.tree.body if isinstance(t, ast.If) and _is_main_test(t.test)
                       for s in t.body if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
                       and isinstance(s.value.func, ast.Name) and s.value.func.id == node.name]
            if len(top) + len(guarded) != 1:
                raise Refused("the harness function is not called once, at module level")

    # ------------------------------------------------------------------ names
    def resolve(self, name: str):
        for scope in reversed(self.comp_stack):
            if name in scope:
                return ("var", scope[name])
        if self.func_stack and name in self.func_stack[-1].locals:
            return ("var", self.func_stack[-1].locals[name])
        if name in self._module_stored:
            return ("var", self.module_vars.setdefault(name, self.g.new()))
        if name in self.funcs:
            return ("func", self.funcs[name])
        if name in self.sv_modules or name in self.sv_funcs:
            return ("sv", None)
        if name in self.imports:
            return ("import", None)
        if name in _REFUSED_NAMES:
            raise Refused(f"use of {name!r}")
        if name in _BUILTINS:
            return ("builtin", name)
        raise Refused(f"undefined name {name!r}")

    def var(self, name: str) -> _T:
        kind, val = self.resolve(name)
        if kind == "var":
            return val
        if kind == "sv":
            raise Refused("a draw outside the harness prelude (or an unmodelled _sv_verifier name)")
        if kind == "func":
            raise Refused(f"function {name!r} used as a value")
        return self.g.clean()

    # ------------------------------------------------------------------ statements
    def stmt(self, s) -> None:
        g = self.g
        if isinstance(s, (ast.Import, ast.ImportFrom, ast.Pass, ast.Break, ast.Continue)):
            return
        if isinstance(s, ast.Expr):
            self.expr(s.value)
        elif isinstance(s, ast.FunctionDef):
            if s.name not in self.funcs or self.funcs[s.name].node is not s:
                raise Refused("nested or repeated function definition")
            f = self.funcs[s.name]
            self.func_stack.append(f)
            try:
                for p, d in f.defaults.items():
                    g.unify(f.locals[p], self.expr(d), "a default value")
                for t in s.body:
                    self.stmt(t)
                if not _terminates(s.body):
                    g.unify(f.ret, g.clean(), "a function that can fall off its end returns None")
            finally:
                self.func_stack.pop()
        elif isinstance(s, ast.Assign):
            draw = self.draw_nodes.get(id(s))
            if draw is not None and self.harness is not None and s in self.harness[self.prelude_start:self.prelude_end]:
                node = self.var(draw.name)
                g.unify(node, {"list": g.seq(g.val()), "val": g.val()}.get(draw.kind, g.clean()), f"draw {draw.name}")
                return
            if (len(s.targets) == 1 and isinstance(s.targets[0], (ast.Tuple, ast.List))
                    and isinstance(s.value, (ast.Tuple, ast.List)) and len(s.targets[0].elts) == len(s.value.elts)
                    and not any(isinstance(e, ast.Starred) for e in s.targets[0].elts + s.value.elts)):
                vals = [self.expr(v) for v in s.value.elts]
                for t, v in zip(s.targets[0].elts, vals):
                    self.store(t, v)
                return
            v = self.expr(s.value)
            for t in s.targets:
                self.store(t, v)
        elif isinstance(s, ast.AnnAssign):
            if not isinstance(s.target, ast.Name):
                raise Refused("annotated non-name target")
            if s.value is not None:
                self.store(s.target, self.expr(s.value))
        elif isinstance(s, ast.AugAssign):
            cur = self.expr(_as_load(s.target))
            self.store(s.target, self.binop(s.op, cur, self.expr(s.value)))
        elif isinstance(s, ast.If):
            g.not_val(self.expr(s.test), "an if-test reads a value's truthiness")
            for t in s.body + s.orelse:
                self.stmt(t)
        elif isinstance(s, ast.While):
            g.not_val(self.expr(s.test), "a while-test reads a value's truthiness")
            for t in s.body + s.orelse:
                self.stmt(t)
        elif isinstance(s, ast.For):
            self.iterate(s.iter, s.target)
            for t in s.body + s.orelse:
                self.stmt(t)
        elif isinstance(s, ast.Return):
            if not self.func_stack:
                raise Refused("return outside a function")
            g.unify(self.func_stack[-1].ret, self.expr(s.value) if s.value is not None else g.clean(),
                    "a function's return")
        elif isinstance(s, ast.Assert):
            g.not_val(self.expr(s.test), "an assert reads a value's truthiness")
            if s.msg is not None:
                g.untainted(self.expr(s.msg), "an assert message")
        elif isinstance(s, ast.Raise):
            for e in (s.exc, s.cause):
                if e is not None:
                    g.untainted(self.expr(e), "a raised exception")
        elif isinstance(s, ast.Try):
            for t in s.body:
                self.stmt(t)
            for h in s.handlers:
                if h.type is not None:
                    g.untainted(self.expr(h.type), "an except clause")
                if h.name:
                    self.store(ast.Name(h.name, ast.Store()), g.clean())
                for t in h.body:
                    self.stmt(t)
            for t in s.orelse + s.finalbody:
                self.stmt(t)
        else:
            raise Refused(f"unsupported statement {type(s).__name__}")

    def store(self, target, v: _T) -> None:
        g = self.g
        if isinstance(target, ast.Name):
            g.unify(self.var(target.id), v, f"assignment to {target.id}")
        elif isinstance(target, (ast.Tuple, ast.List)):
            e = g.new()
            g.unify(v, g.seq(e), "unpacking")
            for t in target.elts:
                if isinstance(t, ast.Starred):
                    raise Refused("starred unpacking")
                self.store(t, e)
        elif isinstance(target, ast.Subscript):
            base = self.expr(target.value)
            idx, is_slice = self.index(target.slice)
            g.deferred.append(["substore", base, idx, is_slice, v])
        elif isinstance(target, ast.Attribute):
            g.untainted(self.expr(target.value), "an attribute store")
            g.untainted(v, "an attribute store")
        else:
            raise Refused(f"unsupported assignment target {type(target).__name__}")

    def iterate(self, iter_node, target) -> None:
        g = self.g
        if isinstance(iter_node, ast.Call) and isinstance(iter_node.func, ast.Name) \
                and self.resolve(iter_node.func.id) == ("builtin", iter_node.func.id) \
                and iter_node.func.id in ("enumerate", "zip") and not iter_node.keywords:
            args = iter_node.args
            if not isinstance(target, ast.Tuple) or any(isinstance(a, ast.Starred) for a in args):
                raise Refused(f"{iter_node.func.id} must be unpacked in the loop target")
            if iter_node.func.id == "enumerate":
                if not (1 <= len(args) <= 2) or len(target.elts) != 2:
                    raise Refused("enumerate shape")
                if len(args) == 2:
                    g.untainted(self.expr(args[1]), "enumerate start")
                self.store(target.elts[0], g.clean())
                e = g.new()
                g.unify(self.expr(args[0]), g.seq(e), "enumerate")
                self.store(target.elts[1], e)
            else:
                if len(args) != len(target.elts) or not args:
                    raise Refused("zip shape")
                for a, t in zip(args, target.elts):
                    e = g.new()
                    g.unify(self.expr(a), g.seq(e), "zip")
                    self.store(t, e)
            return
        e = g.new()
        g.unify(self.expr(iter_node), g.seq(e), "iteration")
        self.store(target, e)

    # ------------------------------------------------------------------ expressions
    def index(self, sl):
        if isinstance(sl, ast.Slice):
            for part in (sl.lower, sl.upper, sl.step):
                if part is not None:
                    self.g.untainted(self.expr(part), "a slice bound")
            return self.g.clean(), True
        return self.expr(sl), False

    def expr(self, e) -> _T:
        g = self.g
        if isinstance(e, ast.Constant):
            return g.clean()
        if isinstance(e, ast.Name):
            return self.var(e.id)
        if isinstance(e, (ast.Tuple, ast.List)):
            el = g.new()
            for x in e.elts:
                if isinstance(x, ast.Starred):
                    raise Refused("starred display")
                g.unify(el, self.expr(x), "list display")
            return g.seq(el)
        if isinstance(e, (ast.Set, ast.Dict)):
            parts = e.elts if isinstance(e, ast.Set) else [k for k in e.keys] + list(e.values)
            for x in parts:
                if x is None:
                    raise Refused("dict unpacking")
                g.untainted(self.expr(x), "a set or dict display")
            return g.clean()
        if isinstance(e, ast.JoinedStr):
            for v in e.values:
                if isinstance(v, ast.FormattedValue):
                    g.untainted(self.expr(v.value), "an f-string")
                    if v.format_spec is not None:
                        self.expr(v.format_spec)
            return g.clean()
        if isinstance(e, ast.BoolOp):
            vals = [self.expr(v) for v in e.values]
            for v in vals:
                g.not_val(v, "and/or reads a value's truthiness")
            for v in vals[1:]:
                g.unify(vals[0], v, "and/or result")
            return vals[0]
        if isinstance(e, ast.IfExp):
            g.not_val(self.expr(e.test), "a conditional expression reads a value's truthiness")
            a, b = self.expr(e.body), self.expr(e.orelse)
            g.unify(a, b, "conditional expression")
            return a
        if isinstance(e, ast.UnaryOp):
            v = self.expr(e.operand)
            if isinstance(e.op, ast.Not):
                g.not_val(v, "`not` reads a value's truthiness")
            else:
                g.untainted(v, "unary arithmetic")
            return g.clean()
        if isinstance(e, ast.BinOp):
            return self.binop(e.op, self.expr(e.left), self.expr(e.right))
        if isinstance(e, ast.Compare):
            return self.compare(e)
        if isinstance(e, ast.Subscript):
            base = self.expr(e.value)
            idx, is_slice = self.index(e.slice)
            res = g.new()
            g.deferred.append(["sub", base, idx, is_slice, res])
            return res
        if isinstance(e, ast.Attribute):
            if isinstance(e.value, ast.Name) and self.resolve(e.value.id)[0] == "import":
                return g.clean()
            if e.attr.startswith("__") or e.attr in ("f_locals", "f_globals", "_getframe"):
                raise Refused(f"attribute {e.attr!r}")
            g.untainted(self.expr(e.value), "an attribute read")
            return g.clean()
        if isinstance(e, (ast.ListComp, ast.GeneratorExp)):
            self.comp_stack.append(self._comp_scope(e))
            try:
                for gen in e.generators:
                    if gen.is_async:
                        raise Refused("async comprehension")
                    self.iterate(gen.iter, gen.target)
                    for cond in gen.ifs:
                        g.not_val(self.expr(cond), "a comprehension condition reads a value's truthiness")
                return g.seq(self.expr(e.elt))
            finally:
                self.comp_stack.pop()
        if isinstance(e, (ast.SetComp, ast.DictComp)):
            self.comp_stack.append(self._comp_scope(e))
            try:
                for gen in e.generators:
                    if gen.is_async:
                        raise Refused("async comprehension")
                    self.iterate(gen.iter, gen.target)
                    for cond in gen.ifs:
                        g.not_val(self.expr(cond), "a comprehension condition")
                parts = [e.elt] if isinstance(e, ast.SetComp) else [e.key, e.value]
                for p in parts:
                    g.untainted(self.expr(p), "a set or dict comprehension")
                return g.clean()
            finally:
                self.comp_stack.pop()
        if isinstance(e, ast.Call):
            return self.call(e)
        raise Refused(f"unsupported expression {type(e).__name__}")

    def _comp_scope(self, comp) -> dict:
        """A comprehension's targets are its own variables (Python 3 scoping)."""
        return {n.id: self.g.new() for gen in comp.generators for n in ast.walk(gen.target)
                if isinstance(n, ast.Name)}

    def binop(self, op, left: _T, right: _T) -> _T:
        g, res = self.g, self.g.new()
        if isinstance(op, ast.Add):
            g.deferred.append(["add", left, right, res])
        elif isinstance(op, ast.Mult):
            g.deferred.append(["mul", left, right, res])
        else:
            g.untainted(left, "arithmetic or bit operation")
            g.untainted(right, "arithmetic or bit operation")
            g.unify(res, g.clean(), "arithmetic result")
        return res

    def compare(self, e: ast.Compare) -> _T:
        g = self.g
        operands = [self.expr(e.left)] + [self.expr(c) for c in e.comparators]
        for op, a, b in zip(e.ops, operands, operands[1:]):
            if isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
                g.unify(a, b, "comparison of a value with something that is not a value")
            elif isinstance(op, (ast.In, ast.NotIn)):
                g.unify(b, g.seq(a), "`in` over something that is not a list or tuple of values")
            elif isinstance(op, (ast.Is, ast.IsNot)):
                g.not_val(a, "`is` on a value")
                g.not_val(b, "`is` on a value")
            else:
                raise Refused("unknown comparison")
        return g.clean()

    def call(self, e: ast.Call) -> _T:
        g = self.g
        if any(isinstance(a, ast.Starred) for a in e.args) or any(k.arg is None for k in e.keywords):
            raise Refused("star arguments")
        f = e.func
        if isinstance(f, ast.Name):
            kind, val = self.resolve(f.id)
            if kind == "func":
                return self.user_call(val, e)
            if kind == "builtin":
                return self.builtin_call(f.id, e)
            if kind == "import":
                return self.generic_call(e)
            if kind == "sv":
                raise Refused("a draw outside the harness prelude")
            raise Refused(f"call through the variable {f.id!r}")
        if isinstance(f, ast.Attribute):
            if f.attr.startswith("__"):
                raise Refused(f"attribute {f.attr!r}")
            if isinstance(f.value, ast.Name) and self.resolve(f.value.id)[0] == "import":
                return self.generic_call(e)
            recv = self.expr(f.value)
            args = [self.expr(a) for a in e.args]
            kws = {k.arg: self.expr(k.value) for k in e.keywords}
            res = g.new()
            g.deferred.append(["method", recv, f.attr, args, kws, res])
            return res
        raise Refused("call of a computed callee")

    def generic_call(self, e: ast.Call) -> _T:
        for a in list(e.args) + [k.value for k in e.keywords]:
            self.g.untainted(self.expr(a), "an argument of a call the analysis does not model")
        return self.g.clean()

    def user_call(self, fn: _Func, e: ast.Call) -> _T:
        g = self.g
        if len(e.args) > len(fn.params):
            raise Refused(f"too many arguments to {fn.node.name}")
        bound = dict(zip(fn.params, e.args))
        for k in e.keywords:
            if k.arg not in fn.params or k.arg in bound:
                raise Refused(f"bad keyword {k.arg!r} for {fn.node.name}")
            bound[k.arg] = k.value
        for p in fn.params:
            if p in bound:
                g.unify(fn.locals[p], self.expr(bound[p]), f"argument {p} of {fn.node.name}")
            elif p not in fn.defaults:
                raise Refused(f"missing argument {p} for {fn.node.name}")
        return fn.ret

    def builtin_call(self, name: str, e: ast.Call) -> _T:
        g = self.g
        args = [self.expr(a) for a in e.args]
        kws = {k.arg: self.expr(k.value) for k in e.keywords}
        if name == "len" and len(args) == 1 and not kws:
            g.not_val(args[0], "len of a value")
            return g.clean()
        if name == "range" and not kws:
            for a in args:
                g.untainted(a, "a range argument")
            return g.clean()
        if name in ("sorted", "list", "tuple", "reversed") and len(args) <= 1:
            allowed = {"reverse"} if name == "sorted" else set()
            if set(kws) - allowed:
                raise Refused(f"{name} with a key or other keyword")
            for v in kws.values():
                g.untainted(v, "sorted(reverse=...)")
            el = g.new()
            if args:
                g.unify(args[0], g.seq(el), name)
            return g.seq(el)
        if name in ("min", "max") and args and not kws:
            if len(args) == 1:
                el = g.new()
                g.unify(args[0], g.seq(el), name)
                return el
            for a in args[1:]:
                g.unify(args[0], a, name)
            return args[0]
        if name in ("enumerate", "zip", "map", "filter", "iter", "next"):
            for a in list(args) + list(kws.values()):
                g.untainted(a, f"{name} outside a loop header")
            return g.clean()
        for a in list(args) + list(kws.values()):
            g.untainted(a, f"an argument of {name}()")
        return g.clean()

    # ------------------------------------------------------------------ shape-polymorphic operations
    def _fixpoint(self):
        g = self.g
        pending = g.deferred
        while pending:
            progress, rest = False, []
            for op in pending:
                if self._resolve_op(op, final=False):
                    progress = True
                else:
                    rest.append(op)
            pending = rest
            if not progress and pending:
                # a base nothing ever bound holds nothing: it is clean (one at a time, then the others may resolve)
                self._resolve_op(pending.pop(0), final=True)
        g.deferred = []

    def _resolve_op(self, op, final: bool) -> bool:
        g = self.g
        kind = op[0]
        key = {"add": 1, "mul": 1, "sub": 1, "method": 1, "substore": 1}[kind]
        subject = _find(op[key])
        if kind in ("add", "mul"):
            l, r = _find(op[1]), _find(op[2])
            if l.tag is None and r.tag is None and not final:
                return False
            if l.tag is None and r.tag != "seq" and not final:
                return False
            if r.tag is None and l.tag != "seq" and not final:
                return False
            res = op[3]
            if kind == "add" and "seq" in (l.tag, r.tag):
                g.unify(l, r, "list concatenation")
                g.unify(res, l, "list concatenation")
            elif kind == "mul" and l.tag == "seq":
                g.untainted(r, "a repeat count")
                g.unify(res, l, "list repetition")
            elif kind == "mul" and r.tag == "seq":
                g.untainted(l, "a repeat count")
                g.unify(res, r, "list repetition")
            else:
                g.untainted(l, "arithmetic")
                g.untainted(r, "arithmetic")
                g.unify(res, g.clean(), "arithmetic result")
            return True
        if subject.tag is None:
            if not final:
                return False
            g.unify(subject, g.clean(), "an unbound base")
            subject = _find(subject)
        if kind == "sub":
            _, base, idx, is_slice, res = op
            g.untainted(idx, "an index or slice bound")
            if subject.tag == "seq":
                g.unify(res, base if is_slice else subject.elem, "subscript")
            elif subject.tag == "val":
                raise Refused("a value is subscripted")
            else:
                g.unify(res, g.clean(), "subscript of a non-list")
            return True
        if kind == "substore":
            _, base, idx, is_slice, v = op
            g.untainted(idx, "an index")
            if subject.tag == "seq":
                g.unify(subject if is_slice else subject.elem, v, "subscript store")
            elif subject.tag == "val":
                raise Refused("a value is subscripted")
            else:
                g.untainted(v, "a store into a non-list")
            return True
        if kind == "method":
            _, recv, name, args, kws, res = op
            if subject.tag == "val":
                raise Refused(f"method .{name} called on a value")
            if subject.tag == "seq":
                self._list_method(subject, name, args, kws, res)
            else:
                for a in list(args) + list(kws.values()):
                    g.untainted(a, f"an argument of .{name}() on a non-list")
                g.unify(res, g.clean(), "method result")
            return True
        raise AssertionError(kind)

    def _list_method(self, seq, name, args, kws, res):
        g = self.g
        n = len(args)
        if name == "append" and n == 1 and not kws:
            g.unify(seq.elem, args[0], "append")
            g.unify(res, g.clean(), "append")
        elif name == "extend" and n == 1 and not kws:
            g.unify(args[0], g.seq(seq.elem), "extend")
            g.unify(res, g.clean(), "extend")
        elif name == "insert" and n == 2 and not kws:
            g.untainted(args[0], "an insert position")
            g.unify(seq.elem, args[1], "insert")
            g.unify(res, g.clean(), "insert")
        elif name == "pop" and n <= 1 and not kws:
            for a in args:
                g.untainted(a, "a pop position")
            g.unify(res, seq.elem, "pop")
        elif name in ("index", "count", "remove") and n == 1 and not kws:
            g.unify(seq.elem, args[0], name)
            g.unify(res, g.clean(), name)
        elif name in ("reverse", "clear") and n == 0 and not kws:
            g.unify(res, g.clean(), name)
        elif name == "copy" and n == 0 and not kws:
            g.unify(res, g.seq(seq.elem), "copy")
        elif name == "sort" and n == 0 and set(kws) <= {"reverse"}:
            for v in kws.values():
                g.untainted(v, "sort(reverse=...)")
            g.unify(res, g.clean(), "sort")
        else:
            g.untainted(seq, f"list method .{name}() the analysis does not model")
            for a in list(args) + list(kws.values()):
                g.untainted(a, f"an argument of .{name}()")
            g.unify(res, g.clean(), "method result")


def _store_names(node):
    """The ``Name`` nodes bound by ``node`` (shallow over statements, deep over targets)."""
    out = []
    if isinstance(node, (ast.Assign,)):
        targets = node.targets
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        targets = [node.target]
    elif isinstance(node, ast.For):
        targets = [node.target]
    elif isinstance(node, ast.NamedExpr):
        targets = [node.target]
    else:
        return out
    for t in targets:
        for n in ast.walk(t):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                out.append(n)
    return out


def _as_load(target):
    import copy
    t = copy.deepcopy(target)
    for n in ast.walk(t):
        if hasattr(n, "ctx"):
            n.ctx = ast.Load()
    return t


def _is_main_test(test) -> bool:
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__"
            and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.comparators[0], ast.Constant) and test.comparators[0].value == "__main__")


def _terminates(body) -> bool:
    if not body:
        return False
    last = body[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return True
    if isinstance(last, ast.If):
        return bool(last.orelse) and _terminates(last.body) and _terminates(last.orelse)
    return False


# --------------------------------------------------------------------------------------- the bound

def _linear(e, env, lens, scalars):
    """``e`` as ({var: coef}, const) over ``("len", name)`` and ``("sc", name)`` variables, or None."""
    if isinstance(e, ast.Constant) and type(e.value) is int:
        return {}, e.value
    if isinstance(e, ast.Name):
        if e.id in env:
            return env[e.id]
        if e.id in scalars:
            return {("sc", e.id): 1}, 0
        return None
    if isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "len" and len(e.args) == 1 \
            and not e.keywords and isinstance(e.args[0], ast.Name) and e.args[0].id in lens:
        return {("len", e.args[0].id): 1}, 0
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd)):
        x = _linear(e.operand, env, lens, scalars)
        if x is None:
            return None
        s = -1 if isinstance(e.op, ast.USub) else 1
        return {k: s * v for k, v in x[0].items()}, s * x[1]
    if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Sub)):
        a, b = _linear(e.left, env, lens, scalars), _linear(e.right, env, lens, scalars)
        if a is None or b is None:
            return None
        s = 1 if isinstance(e.op, ast.Add) else -1
        co = dict(a[0])
        for k, v in b[0].items():
            co[k] = co.get(k, 0) + s * v
        return {k: v for k, v in co.items() if v}, a[1] + s * b[1]
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Mult):
        a, b = _linear(e.left, env, lens, scalars), _linear(e.right, env, lens, scalars)
        if a is None or b is None:
            return None
        if not a[0]:
            return {k: a[1] * v for k, v in b[0].items()}, a[1] * b[1]
        if not b[0]:
            return {k: b[1] * v for k, v in a[0].items()}, a[1] * b[1]
    return None


def _pure(e, env, lens, scalars) -> bool:
    """A test that cannot raise: comparisons and and/or/not over linear integer expressions."""
    if isinstance(e, ast.BoolOp):
        return all(_pure(v, env, lens, scalars) for v in e.values)
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
        return _pure(e.operand, env, lens, scalars)
    if isinstance(e, ast.Compare):
        return all(_op_name(o) is not None for o in e.ops) and all(
            _linear(x, env, lens, scalars) is not None for x in [e.left] + list(e.comparators))
    return False


def _atoms(e, positive, env, lens, scalars):
    """Atoms (coefs, const, op) meaning ``sum(coefs*vars) + const op 0`` that MUST hold when ``e`` has truth value
    ``positive``.  Atoms that are not understood are simply left out: leaving one out only widens the enumeration."""
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
        return _atoms(e.operand, not positive, env, lens, scalars)
    if isinstance(e, ast.BoolOp):
        conj = isinstance(e.op, ast.And) == positive
        return [a for v in e.values for a in _atoms(v, positive, env, lens, scalars)] if conj else []
    if isinstance(e, ast.Compare):
        if len(e.ops) > 1 and not positive:
            return []
        out = []
        operands = [e.left] + list(e.comparators)
        for op, l, r in zip(e.ops, operands, operands[1:]):
            name = _op_name(op)
            a, b = _linear(l, env, lens, scalars), _linear(r, env, lens, scalars)
            if name is None or a is None or b is None:
                continue
            if not positive:
                name = _NEGATE[name]
            co = dict(a[0])
            for k, v in b[0].items():
                co[k] = co.get(k, 0) - v
            out.append(({k: v for k, v in co.items() if v}, a[1] - b[1], name))
        return out
    return []


def _bounds(atom, var, others: dict):
    """(lo, hi) on ``var`` from one atom once the other variables are fixed (None = no bound), or "empty"."""
    co, const, op = atom
    a = co.get(var, 0)
    if a == 0:
        return None, None
    r = const + sum(c * others[k] for k, c in co.items() if k != var)
    t = Fraction(-r, a)
    if op == "!=":
        return None, None
    if op == "==":
        return (math.ceil(t), math.floor(t)) if t.denominator == 1 else ("empty", "empty")
    if a < 0:
        op = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}[op]
    if op == "<":
        return None, math.ceil(t) - 1
    if op == "<=":
        return None, math.floor(t)
    if op == ">":
        return math.floor(t) + 1, None
    return math.ceil(t), None


def _plan(an: _Analyzer, modes) -> Plan:
    lists = {d.name for d in an.draws if d.kind == "list"}
    scalars = {d.name for d in an.draws if d.kind == "index"}
    body, env, atoms = an.harness, {}, []
    for s in body[an.prelude_end:]:
        if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) \
                and s.targets[0].id not in env and s.targets[0].id not in {d.name for d in an.draws}:
            lin = _linear(s.value, env, lists, scalars)
            if lin is None:
                break
            env[s.targets[0].id] = lin
            continue
        if isinstance(s, ast.If) and not s.orelse and len(s.body) == 1 and isinstance(s.body[0], ast.Return) \
                and (s.body[0].value is None or (isinstance(s.body[0].value, ast.Constant) and s.body[0].value.value is None)) \
                and _pure(s.test, env, lists, scalars):
            atoms.extend(_atoms(s.test, False, env, lists, scalars))
            continue
        break
    upper = {}
    for name in lists:
        hi = []
        for co, const, op in atoms:
            if set(co) == {("len", name)}:
                lo_, hi_ = _bounds((co, const, op), ("len", name), {})
                if hi_ == "empty":
                    hi.append(-1)
                elif hi_ is not None:
                    hi.append(hi_)
        if not hi:
            raise Refused(f"no early-return guard bounds len({name})")
        upper[name] = min(hi)
    index_atoms = {}
    for name in scalars:
        rows = [(co, const, op) for co, const, op in atoms if ("sc", name) in co
                and all(k[0] == "len" or k == ("sc", name) for k in co)]
        index_atoms[name] = rows
    return Plan(an.draws, upper, index_atoms)


def analyze(source: str, filename: str = "<program>") -> Plan:
    """The order-only plan of ``source``, or ``Refused`` naming why it is outside the fragment."""
    try:
        tree = ast.parse(source, filename)
    except (SyntaxError, ValueError) as e:
        raise Refused(f"unreadable: {e}")
    probe = _Analyzer(tree, {})
    probe._imports()
    probe._functions()
    probe._find_harness()
    int_scalars = [d.name for d in probe.draws if d.kind == "val"]
    if len(int_scalars) > 3:
        raise Refused("more than three scalar int draws")
    last: Refused | None = None
    for combo in itertools.product(("val", "index"), repeat=len(int_scalars)):
        modes = dict(zip(int_scalars, combo))
        an = _Analyzer(tree, modes)
        try:
            an.run()
            plan = _plan(an, modes)
            for d in plan.draws:
                if d.kind == "index" and not _index_bounded(plan, d.name):
                    raise Refused(f"no early-return guard bounds the index scalar {d.name}")
            return plan
        except Refused as r:
            last = r
    raise last or Refused("no plan")


def _index_bounded(plan: Plan, name: str) -> bool:
    lows = highs = False
    var = ("sc", name)
    for co, const, op in plan.index_atoms[name]:
        a = co[var]
        if op == "==":
            lows = highs = True
        elif op in ("<", "<="):
            highs, lows = highs or a > 0, lows or a < 0
        elif op in (">", ">="):
            lows, highs = lows or a > 0, highs or a < 0
    return lows and highs


# --------------------------------------------------------------------------------------- enumeration

@functools.lru_cache(maxsize=None)
def fubini(m: int) -> int:
    """Number of weak orderings of ``m`` items."""
    if m == 0:
        return 1
    return sum(math.comb(m, k) * fubini(m - k) for k in range(1, m + 1))


def weak_orderings(m: int) -> Iterator[tuple]:
    """Every weak ordering of ``m`` items as the tuple of their ranks ``0..k-1`` (``fubini(m)`` tuples)."""
    def rec(prefix, k):
        if len(prefix) == m:
            yield tuple(prefix)
            return
        for j in range(k):                                  # ties with an existing rank
            yield from rec(prefix + [j], k)
        for j in range(k + 1):                              # a new rank, inserted at position j
            yield from rec([r + 1 if r >= j else r for r in prefix] + [j], k + 1)
    yield from rec([], 0)


def _index_domain(plan: Plan, name: str, lens: dict, others: dict) -> range:
    lo, hi = -math.inf, math.inf
    var = ("sc", name)
    fixed = {("len", n): v for n, v in lens.items()}
    fixed.update({("sc", n): v for n, v in others.items()})
    for atom in plan.index_atoms[name]:
        l, h = _bounds(atom, var, fixed)
        if l == "empty":
            return range(0)
        if l is not None:
            lo = max(lo, l)
        if h is not None:
            hi = min(hi, h)
    return range(int(lo), int(hi) + 1) if lo <= hi else range(0)


def length_vectors(plan: Plan) -> list[dict]:
    names = [d.name for d in plan.lists]
    vecs = [dict(zip(names, c)) for c in itertools.product(*[range(plan.len_upper[n] + 1) for n in names])]
    vecs.sort(key=lambda v: (sum(v.values()), tuple(v.values())))
    return vecs


def plan_size(plan: Plan) -> int:
    """How many runs the plan needs (without generating them); raises ``Refused`` above the caps."""
    total = 0
    nvals = len(plan.value_scalars)
    for lens in length_vectors(plan):
        m = sum(lens.values()) + nvals
        if m > MAX_VALUES:
            raise Refused(f"{m} values to order exceeds {MAX_VALUES}")
        total += fubini(m) * _domain_product(plan, lens)
        if total > MAX_RUNS:
            raise Refused(f"the plan needs more than {MAX_RUNS} runs")
    return total


def _domain_product(plan: Plan, lens: dict) -> int:
    n = 1
    chosen: dict = {}
    for d in plan.draws:
        if d.kind == "bool":
            n *= 2
        elif d.kind == "index":
            dom = _index_domain(plan, d.name, lens, chosen)
            if len(dom) > MAX_INDEX_DOMAIN:
                raise Refused(f"index scalar {d.name} ranges over more than {MAX_INDEX_DOMAIN} values")
            n *= len(dom)
            chosen[d.name] = dom[0] if len(dom) else 0
            if not len(dom):
                return 0
    return n


class OrderEnumerator(Protocol):
    """The seam for the enumeration: ``WeakOrderings`` is the one in this file; a finite-domain enumerator
    (T178's) can replace it by implementing ``vectors`` over the same ``Plan``."""

    def vectors(self, plan: Plan) -> Iterator[list]: ...


class WeakOrderings:
    """Every length vector the guards admit x every weak ordering of the values x every admitted index value.

    A vector is a list of ``{"kind", "value"}`` draws in program order (``nondet_list`` as bool/int pairs and a closing
    False), small-first (shortest lists first)."""

    def vectors(self, plan: Plan) -> Iterator[list]:
        nvals = len(plan.value_scalars)
        for lens in length_vectors(plan):
            m = sum(lens.values()) + nvals
            for ranks in weak_orderings(m):
                yield from self._expand(plan, lens, ranks)

    def _expand(self, plan, lens, ranks):
        def go(i, pos, chosen, vec):
            if i == len(plan.draws):
                yield vec
                return
            d = plan.draws[i]
            if d.kind == "list":
                part = []
                for r in ranks[pos:pos + lens[d.name]]:
                    part += [{"kind": "bool", "value": True}, {"kind": "int", "value": r}]
                part.append({"kind": "bool", "value": False})
                yield from go(i + 1, pos + lens[d.name], chosen, vec + part)
            elif d.kind == "val":
                yield from go(i + 1, pos + 1, chosen, vec + [{"kind": "int", "value": ranks[pos]}])
            elif d.kind == "bool":
                for b in (False, True):
                    yield from go(i + 1, pos, chosen, vec + [{"kind": "bool", "value": b}])
            else:
                for v in _index_domain(plan, d.name, lens, chosen):
                    yield from go(i + 1, pos, {**chosen, d.name: v}, vec + [{"kind": "int", "value": v}])
        yield from go(0, 0, {}, [])


# --------------------------------------------------------------------------------------- the batch run

CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_order_child.py")


@dataclass
class BatchResult:
    ran: int = 0
    violation: dict | None = None        # {"vector": [...], "exc": name, "mro": [...]}
    anomaly: str | None = None           # a run that did not complete, or consumed other draws than it was given
    complete: bool = False               # every vector was run
    seconds: float = 0.0


def run_plan(plan: Plan, program: str, python: str, classes, *, allowance, memory_bytes=None,
             enumerator: OrderEnumerator | None = None) -> BatchResult:
    """Run every vector of ``plan`` natively, stopping at the first violation or anomaly.

    ``allowance()`` is the CPU seconds the caller can still spend; a plan that does not fit ends ``complete=False``."""
    enumerator = enumerator or WeakOrderings()
    result, t0 = BatchResult(), time.monotonic()
    stream = enumerator.vectors(plan)
    while True:
        chunk = list(itertools.islice(stream, CHUNK))
        if not chunk:
            result.complete = True
            break
        left = allowance()
        if left <= 1:
            result.anomaly = "the budget ran out before every ordering was run"
            break
        out = _run_chunk(chunk, program, python, classes, timeout=left, memory_bytes=memory_bytes)
        result.ran += out.get("ran", 0)
        if out.get("violation") is not None:
            v = out["violation"]
            result.violation = {"vector": chunk[v["index"]], "exc": v["exc"], "mro": v["mro"]}
            break
        if out.get("anomaly") or out.get("ran", 0) != len(chunk):
            result.anomaly = out.get("anomaly") or "the batch child ended early"
            break
    result.seconds = time.monotonic() - t0
    return result


def _run_chunk(chunk, program, python, classes, *, timeout, memory_bytes) -> dict:
    with tempfile.TemporaryDirectory(prefix="tharkun-order-") as tmp:
        req, out = os.path.join(tmp, "req.json"), os.path.join(tmp, "out.json")
        vectors = [["".join("b" if e["kind"] == "bool" else "i" for e in v), [int(e["value"]) for e in v]] for v in chunk]
        with open(req, "w") as fh:
            json.dump({"program": os.path.abspath(program), "vectors": vectors, "classes": list(classes), "out": out,
                       "memory": memory_bytes, "run_cpu": RUN_CPU_SECONDS}, fh)
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSTARTUP")}
        env.update(PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1")
        try:
            proc = subprocess.run([python, CHILD, req], cwd=tmp, env=env, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=max(1.0, timeout))
        except subprocess.TimeoutExpired:
            return {"anomaly": "the batch child timed out"}
        if proc.returncode != 0 or not os.path.exists(out):
            return {"anomaly": f"the batch child crashed: {(proc.stderr or '')[-200:]}"}
        try:
            with open(out) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {"anomaly": "the batch child left no readable result"}
