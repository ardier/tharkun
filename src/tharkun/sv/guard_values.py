"""``static:guard_values`` -- draw candidates solved from the program's own guards ([internal ref]).

A stage of :mod:`tharkun.sv.stages` that executes nothing.  It reads the task's
source, finds the variables bound directly to a draw (``x = nondet_int()``,
``_sv_verifier.nondet_str()``, ...) and every guard (``if`` / ``while`` test, and
the negation of an ``assert``), and SOLVES each guard, together with the tests
of the ``if`` statements that enclose it, for values of those variables:

* straight-line definitions are inlined first (``n = 1 + abs(x)`` makes the guard
  ``n % 7 == 3`` a guard over ``x``): a name assigned exactly once in its scope,
  outside every loop, before the guard;
* a guard over integer draws goes to the native integer solver
  (:func:`tharkun.solve.arith.solve_int_guard`), both polarities;
* any other guard whose free names are all draws goes to the static guard
  solver (:func:`tharkun.solve.condition.solve_condition`) with the draw kinds
  as ``var_types``.

The answers are CANDIDATES, not verdicts: :meth:`engine.Candidates` ranks them
right after each kind's head values, so the enumeration tries them early, at
every draw position of that kind -- including draws inside loops and helper
functions that the concolic harness cannot lift.  Every FALSE still comes from a
replayed run.  Cost: one parse plus a few milliseconds per guard, capped by
``max_guards`` and ``budget_seconds``.
"""

from __future__ import annotations

import ast
import copy
import time

from tharkun.solve.arith import free_names, solve_int_guard

#: ``_sv_verifier`` draw functions and the draw kind each binds
DRAWS = {"nondet_int": "int", "nondet_bool": "bool", "nondet_float": "float",
         "nondet_float_range": "float", "nondet_str": "str"}
_CAP_PER_KIND = 64
_INLINE_DEPTH = 8
_INLINE_NODES = 400


def _draw_kind(value: ast.expr) -> str | None:
    if isinstance(value, ast.Call):
        f = value.func
        name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
        return DRAWS.get(name)
    return None


def _rebound(target: ast.expr) -> list:
    """Names a store target REBINDS (``a[i] = v`` and ``a.f = v`` rebind nothing)."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        return [n for e in target.elts for n in _rebound(e)]
    if isinstance(target, ast.Starred):
        return _rebound(target.value)
    return []


class _Scope:
    """The names one function body (or the module) binds, and how."""

    def __init__(self, body: list):
        self.draws: dict[str, str] = {}
        self.defs: dict[str, tuple] = {}          # name -> (lineno, expr) for single straight-line defs
        counts: dict[str, int] = {}
        loop_bound: set = set()
        non_draw: set = set()

        def bind(name, lineno, value, in_loop):
            counts[name] = counts.get(name, 0) + 1
            kind = _draw_kind(value) if value is not None else None
            if kind:
                if self.draws.get(name, kind) != kind:
                    non_draw.add(name)
                self.draws.setdefault(name, kind)
            else:
                non_draw.add(name)
                if in_loop:
                    loop_bound.add(name)
                elif value is not None:
                    self.defs[name] = (lineno, value)

        def walk(stmts, in_loop):
            for st in stmts:
                if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    continue
                if isinstance(st, ast.Assign):
                    for t in st.targets:
                        if isinstance(t, ast.Name):
                            bind(t.id, st.lineno, st.value if len(st.targets) == 1 else None, in_loop)
                        else:
                            for n in _rebound(t):
                                bind(n, st.lineno, None, in_loop)
                elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
                    bind(st.target.id, st.lineno, st.value, in_loop)
                elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name):
                    bind(st.target.id, st.lineno, None, in_loop)
                elif isinstance(st, (ast.For, ast.AsyncFor)):
                    for n in _rebound(st.target):
                        bind(n, st.lineno, None, True)
                    walk(st.body, True)
                    walk(st.orelse, in_loop)
                    continue
                elif isinstance(st, ast.While):
                    walk(st.body, True)
                    walk(st.orelse, in_loop)
                    continue
                elif isinstance(st, (ast.With, ast.AsyncWith)):
                    for item in st.items:
                        if item.optional_vars is not None:
                            for n in _rebound(item.optional_vars):
                                bind(n, st.lineno, None, in_loop)
                # walrus and other nested bindings
                for n in ast.walk(st):
                    if isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name):
                        bind(n.target.id, st.lineno, None, in_loop)
                for field in ("body", "orelse", "finalbody"):
                    sub = getattr(st, field, None)
                    if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                        walk(sub, in_loop)
                for h in getattr(st, "handlers", []) or []:
                    if h.name:
                        bind(h.name, h.lineno, None, in_loop)
                    walk(h.body, in_loop)

        walk(body, False)
        for name in non_draw:
            self.draws.pop(name, None)
        for name, k in counts.items():
            if k != 1 or name in loop_bound:
                self.defs.pop(name, None)
        for name in self.draws:
            self.defs.pop(name, None)


class _Inline(ast.NodeTransformer):
    def __init__(self, scope: _Scope, before: int):
        self.scope, self.before, self.depth, self.nodes = scope, before, 0, 0

    def visit_Name(self, node: ast.Name):
        d = self.scope.defs.get(node.id)
        if d is None or d[0] >= self.before or self.depth >= _INLINE_DEPTH:
            return node
        self.depth += 1
        try:
            out = self.visit(copy.deepcopy(d[1]))
        finally:
            self.depth -= 1
        self.nodes += sum(1 for _ in ast.walk(out))
        if self.nodes > _INLINE_NODES:
            raise ValueError("inline too large")
        return out


def _guards(body: list):
    """``(lineno, [(test, polarity) ...])`` for every guard: the enclosing ``if`` tests
    plus the guard itself (``True`` means the test holds)."""
    out = []

    def walk(stmts, stack):
        for st in stmts:
            if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(st, (ast.If, ast.While)):
                out.append((st.lineno, stack + [(st.test, True)]))
                out.append((st.lineno, stack + [(st.test, False)]))
                walk(st.body, stack + [(st.test, True)])
                walk(st.orelse, stack + [(st.test, False)] if isinstance(st, ast.If) else stack)
                continue
            if isinstance(st, ast.Assert):
                out.append((st.lineno, stack + [(st.test, False)]))
            for field in ("body", "orelse", "finalbody"):
                sub = getattr(st, field, None)
                if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                    walk(sub, stack)
            for h in getattr(st, "handlers", []) or []:
                walk(h.body, stack)

    walk(body, [])
    return out


def _conj(parts: list) -> ast.expr:
    vals = [t if pol else ast.UnaryOp(op=ast.Not(), operand=t) for t, pol in parts]
    return vals[0] if len(vals) == 1 else ast.BoolOp(op=ast.And(), values=vals)


def _is_kind(kind: str, v) -> bool:
    if kind == "int":
        return isinstance(v, int) and not isinstance(v, bool)
    if kind == "float":
        return isinstance(v, float)
    return isinstance(v, str)


def _solve_line(program: str, lineno: int, polarity: bool, draws: dict) -> dict:
    """``solve_condition``'s first verified answer for the guard at ``lineno``, values decoded."""
    from tharkun.solve.condition import solve_condition
    try:
        sol = solve_condition(program, lineno, desired_branch="then" if polarity else "else",
                              var_types=dict(draws))
    except Exception:          # a static solver failure is a missing candidate, nothing more
        return {}
    if not sol.strategy.startswith(("direct", "arith")) or not sol.satisfying_inputs:
        return {}
    out = {}
    for name, v in sol.satisfying_inputs[0].items():
        if isinstance(v, str) and draws.get(name) != "str":
            try:
                v = ast.literal_eval(v)
            except (ValueError, SyntaxError):
                continue
        elif isinstance(v, str):
            try:
                lit = ast.literal_eval(v)
                if isinstance(lit, str):
                    v = lit
            except (ValueError, SyntaxError):
                pass
        out[name] = v
    return out


def seed_values(program: str, *, max_guards: int = 400, budget_seconds: float = 3.0,
                log=None) -> dict[str, list]:
    """Candidate draw values, per kind, solved from ``program``'s guards."""
    say = log or (lambda msg: None)
    try:
        from tharkun.sv.engine import read_source

        tree = ast.parse(read_source(program))
    except (OSError, SyntaxError, ValueError):
        return {}
    t0 = time.process_time()
    bodies = [tree.body] + [n.body for n in ast.walk(tree)
                            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    out: dict[str, list] = {}
    seen: set = set()
    solved = tried = 0

    def add(kind, v):
        key = (kind, repr(v))
        if key not in seen and len(out.get(kind, [])) < _CAP_PER_KIND:
            seen.add(key)
            out.setdefault(kind, []).append(v)

    for body in bodies:
        scope = _Scope(body)
        if not scope.draws:
            continue
        ints = {n for n, k in scope.draws.items() if k == "int"}
        for lineno, parts in _guards(body):
            if tried >= max_guards or time.process_time() - t0 > budget_seconds:
                say(f"guard_values: cap reached after {tried} guards")
                return out
            # inline each test; an enclosing test over names that are not draws cannot be
            # solved for, so it is dropped (the answers are only candidates)
            own_test, own_pol = parts[-1]
            usable = []
            for i, (test, pol) in enumerate(parts):
                try:
                    e = _Inline(scope, lineno).visit(copy.deepcopy(test))
                except (ValueError, RecursionError):
                    e = None
                ok = e is not None and free_names(e) and free_names(e) <= scope.draws.keys()
                if i == len(parts) - 1 and not ok:
                    usable = None
                    break
                if ok:
                    usable.append((e, pol))
            if usable is None:
                continue
            names = set().union(*(free_names(e) for e, _ in usable))
            tried += 1
            if names <= ints:
                # the whole conjunction first, then without one enclosing test at a time
                variants = [usable] + [usable[:i] + usable[i + 1:] for i in range(len(usable) - 1)]
                for variant in variants:
                    res = solve_int_guard(_conj(variant), set().union(*(free_names(e) for e, _ in variant)))
                    if res.status == "sat":
                        solved += 1
                        for v in res.witness.values():
                            add("int", v)
                        break
            elif len(parts) >= 1 and own_test is not None:
                # other kinds: the static guard solver, on the guard's own line and polarity
                sol = _solve_line(program, lineno, own_pol, scope.draws)
                if sol:
                    solved += 1
                    for name, v in sol.items():
                        kind = scope.draws.get(name)
                        if kind in ("int", "float", "str") and _is_kind(kind, v):
                            add(kind, v)
    say(f"guard_values: {tried} guards tried, {solved} solved, "
        f"{sum(len(v) for v in out.values())} candidates")
    return out
