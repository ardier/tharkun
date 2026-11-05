"""``proof:finite_domain`` -- TRUE for a program whose inputs are bounded by its own early-return guards (T178, lane 1518).

The shape (the SV-COMP ``quixbugs/*_2`` and ``ion`` tasks, ``docs/SV.md`` "Finite domain")::

    def main():
        n = _sv_verifier.nondet_int()                      # D: the draws, each ``name = nondet_int()``
        k = _sv_verifier.nondet_int()
        if not (1 <= n <= 1000 and 0 <= k <= 4):           # G: guards, each ``if COND: return``
            return
        ...                                                # B: the body, where the property can fail
    main()

The proof is one sound statement.  Let ``Box`` be the product of the intervals :func:`derive_box` reads off the guards.  Then

* **O1 the prefix cannot fail.**  Every input, inside ``Box`` or not, runs ``D; G`` first.  A guard here is built only from comparisons,
  ``and``/``or``/``not``, the draw names and integer literals: it cannot raise, so no input outside ``Box`` reaches a site before it returns.
* **O2 ``Box`` holds every input that passes the guards.**  Each bound is a NECESSARY condition of "every guard returned False"
  (``bounds`` below pushes negations through ``not``/``and``/``or``; a conjunct it cannot read gives no bound for its variables, which is
  weaker, never wrong).  An input the guards let through is therefore in ``Box``; an input outside it returns at a guard.
* **O3 the caller runs every vector of ``Box``** (fresh child processes, ``sweep``): a violation is a FALSE with its witness exactly as
  for any enumerated draw vector.
* **O4 determinism** of a run given its draws is certified by the caller (``engine._certify_determinism``), as for ``finite_tree``.
* **O5 these are all the draws**: the program names the verifier's ``nondet_*`` functions exactly ``len(D)`` times, each in ``D``.

The first implementations of "finite" in Tharkun (``engine.FINITE_KINDS = ("bool", "rindex")``, 2026-09-24, an AI session, Claude Opus 5.5)
took a draw's finiteness from its KIND; an int bounded by the program's own precondition was never finite, so every QuixBugs ``_2``
task searched for a violation until the budget ended (101 UNKNOWN pairs on the first official-set grader cycle).  Here the domain
comes from the value: the guards.
"""

from __future__ import annotations

import ast
import itertools
from dataclasses import dataclass

#: the most vectors one proof may run; the scheduler's ledger bounds the CPU actually spent (a sweep that does not finish is UNKNOWN)
BOX_CAP = 5000

#: the most vectors a batched sweep may cover (``sweep_batch``: one process runs thousands of vectors, so only the deadline limits it)
BATCH_CAP = 250_000

#: the share of the budget left when the stage starts that the sweep may use (the later stages keep the rest)
DEADLINE_SHARE = 0.9
#: runs measured before the rate is trusted to project the finish
PROJECT_AFTER = 8

_INF = None   # an unbounded side of an interval


@dataclass(frozen=True)
class Box:
    """The draws of ``D`` in execution order and their closed integer intervals."""
    names: tuple[str, ...]
    intervals: tuple[tuple[int, int], ...]

    @property
    def size(self) -> int:
        n = 1
        for lo, hi in self.intervals:
            n *= max(0, hi - lo + 1)
        return n

    def vectors(self):
        """Every draw vector of the box, as the entries the child replays (cheapest first: the lexicographic product)."""
        for combo in itertools.product(*(range(lo, hi + 1) for lo, hi in self.intervals)):
            yield [{"kind": "int", "value": v} for v in combo]


# ---------------------------------------------------------------------------------------------------------------- bounds
# An interval is (lo, hi) with ``None`` for an open side; a bound map is {name: interval}; a name missing from the map is unconstrained.

def _meet(a: dict, b: dict) -> dict:
    out = dict(a)
    for name, (lo, hi) in b.items():
        olo, ohi = out.get(name, (_INF, _INF))
        lo = olo if lo is None else lo if olo is None else max(lo, olo)
        hi = ohi if hi is None else hi if ohi is None else min(hi, ohi)
        out[name] = (lo, hi)
    return out


def _join(a: dict, b: dict) -> dict:
    """The hull of two alternatives: a name must be constrained in both, else it is not constrained."""
    out = {}
    for name in a.keys() & b.keys():
        (alo, ahi), (blo, bhi) = a[name], b[name]
        lo = None if alo is None or blo is None else min(alo, blo)
        hi = None if ahi is None or bhi is None else max(ahi, bhi)
        out[name] = (lo, hi)
    return out


def _int_const(node) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant) and type(node.operand.value) is int):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    return None


_NEGATE = {ast.Lt: ast.GtE, ast.LtE: ast.Gt, ast.Gt: ast.LtE, ast.GtE: ast.Lt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}
_FLIP = {ast.Lt: ast.Gt, ast.LtE: ast.GtE, ast.Gt: ast.Lt, ast.GtE: ast.LtE, ast.Eq: ast.Eq, ast.NotEq: ast.NotEq}


def _pair(left, op, right, names) -> dict:
    """The bound that ``left op right`` (both sides plain) implies, for a name against an integer literal."""
    if isinstance(left, ast.Name) and left.id in names and _int_const(right) is not None:
        name, k, op = left.id, _int_const(right), type(op)
    elif isinstance(right, ast.Name) and right.id in names and _int_const(left) is not None:
        name, k, op = right.id, _int_const(left), _FLIP[type(op)]
    else:
        return {}
    if op is ast.Lt:
        return {name: (_INF, k - 1)}
    if op is ast.LtE:
        return {name: (_INF, k)}
    if op is ast.Gt:
        return {name: (k + 1, _INF)}
    if op is ast.GtE:
        return {name: (k, _INF)}
    if op is ast.Eq:
        return {name: (k, k)}
    return {}                       # != gives no interval


def bounds(expr, positive: bool, names) -> dict:
    """The intervals of the draw names IMPLIED by ``expr`` being ``positive`` (true) or negative (false); a necessary condition."""
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return bounds(expr.operand, not positive, names)
    if isinstance(expr, ast.BoolOp):
        parts = [bounds(v, positive, names) for v in expr.values]
        conjunction = isinstance(expr.op, ast.And) == positive      # and (true) / or (false): every part holds
        out = parts[0]
        for p in parts[1:]:
            out = _meet(out, p) if conjunction else _join(out, p)
        return out
    if isinstance(expr, ast.Compare):
        if any(type(op) not in _NEGATE for op in expr.ops):
            return {}                                                   # ``is``, ``in``, ...: no information
        operands = [expr.left, *expr.comparators]
        links = [_pair(a, op if positive else _NEGATE[type(op)](), b, names)
                 for a, op, b in zip(operands, expr.ops, operands[1:])]
        out = links[0]
        for p in links[1:]:
            out = _meet(out, p) if positive else _join(out, p)          # a chain is a conjunction; its negation a disjunction
        return out
    return {}


# ---------------------------------------------------------------------------------------------------------------- the shape

def _pure_guard(expr, names) -> bool:
    """A guard that cannot raise: comparisons, boolean operators and ``not`` over the draw names and integer literals."""
    if isinstance(expr, ast.BoolOp):
        return all(_pure_guard(v, names) for v in expr.values)
    if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
        return _pure_guard(expr.operand, names)
    if isinstance(expr, ast.Compare):
        return (all(type(op) in _NEGATE for op in expr.ops)
                and all((isinstance(o, ast.Name) and o.id in names) or _int_const(o) is not None
                        for o in (expr.left, *expr.comparators)))
    return False


def _verifier_names(tree: ast.Module):
    """``(module aliases, {from-imported local name: real name})`` of ``_sv_verifier``, or None when a name is rebound."""
    modules, direct = set(), {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "_sv_verifier":
                    modules.add(a.asname or a.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "_sv_verifier" and node.level == 0:
            for a in node.names:
                direct[a.asname or a.name] = a.name
    return modules, direct


def _draw_kind(call, modules, direct) -> str | None:
    """The ``nondet_*`` function a no-argument call names, or None."""
    f = call.func
    if call.args or call.keywords:
        return None
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in modules:
        return f.attr
    if isinstance(f, ast.Name) and f.id in direct:
        return direct[f.id]
    return None


def derive_box(source: str, cap: int = BOX_CAP) -> Box | None:
    """The :class:`Box` of ``source`` or None when it does not have the shape ``D; G`` at the start of the function it ends by calling,
    or a draw is unbounded."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    modules, direct = _verifier_names(tree)
    if not modules and not direct:
        return None
    last = tree.body[-1] if tree.body else None
    if not (isinstance(last, ast.Expr) and isinstance(last.value, ast.Call) and isinstance(last.value.func, ast.Name)
            and not last.value.args and not last.value.keywords):
        return None
    entry = last.value.func.id
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == entry]
    if len(defs) != 1:
        return None
    fn = defs[0]
    a = fn.args
    if a.args or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg or fn.decorator_list:
        return None
    # the entry is called exactly once, by that last statement, and no name of the program is bound twice in a way that hides it
    if sum(1 for n in ast.walk(tree) if isinstance(n, ast.Name) and n.id == entry) != 1:
        return None
    if any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(tree)):
        return None
    # nothing else binds the entry's name (an import after the def would run code the proof never saw) or the verifier's names
    protected = {entry} | modules | set(direct)
    for n in ast.walk(tree):
        bound = ()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound = (n.name,)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            bound = tuple((a.asname or a.name).split(".")[0] for a in n.names if not (
                isinstance(n, ast.Import) and a.name == "_sv_verifier" or isinstance(n, ast.ImportFrom) and n.module == "_sv_verifier"))
        elif isinstance(n, ast.arg):
            bound = (n.arg,)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound = (n.name,)
        if n is not fn and any(b in protected for b in bound):
            return None
    if any(isinstance(n, (ast.Import, ast.ImportFrom)) and any(a.name == "*" for a in n.names) for n in ast.walk(tree)):
        return None
    # D
    draws, i = [], 0
    for st in fn.body:
        if (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                and isinstance(st.value, ast.Call) and _draw_kind(st.value, modules, direct) == "nondet_int"):
            draws.append(st.targets[0].id)
            i += 1
        else:
            break
    if not draws or len(set(draws)) != len(draws):
        return None
    # each draw name is bound exactly once in the function and the function does not rebind it before the guards
    names = set(draws)
    # O5: the verifier's nondet functions are named exactly len(D) times in the whole program
    uses = 0
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in modules and n.attr.startswith("nondet"):
            uses += 1
        elif isinstance(n, ast.Name) and n.id in direct and direct[n.id].startswith("nondet") and isinstance(n.ctx, ast.Load):
            uses += 1
    if uses != len(draws):
        return None
    for n in ast.walk(tree):                 # an alias of the module or a from-imported name must not be rebound or escape
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)) and (n.id in modules or n.id in direct):
            return None
    # G
    guards = []
    for st in fn.body[i:]:
        if (isinstance(st, ast.If) and not st.orelse and len(st.body) == 1 and isinstance(st.body[0], ast.Return)
                and st.body[0].value is None and _pure_guard(st.test, names)):
            guards.append(st.test)
        else:
            break
    if not guards:
        return None
    # no later statement of the function (or of the program) may rebind a draw name before... the guards read the draws as drawn, and
    # nothing between D and G runs code (D and G are consecutive statements), so no further check is needed for O2.
    got: dict = {}
    for g in guards:
        got = _meet(got, bounds(g, False, names))      # the guard returned False: execution fell through
    intervals = []
    for name in draws:
        lo, hi = got.get(name, (_INF, _INF))
        if lo is None or hi is None:
            return None
        intervals.append((lo, hi))
    box = Box(tuple(draws), tuple(intervals))
    if not 0 < box.size <= cap:
        return None
    return box


# ---------------------------------------------------------------------------------------------------------------- the sweep

@dataclass
class Sweep:
    """The outcome of running a list of vectors: ``violation`` (``vec``, ``out``, ``hit``), ``complete`` or ``incomplete`` (a run was
    cut, flagged, or the budget stopped)."""
    status: str
    runs: int = 0
    vec: list | None = None
    out: object = None
    hit: str | None = None
    why: str = ""


def sweep(vectors, run, classes, *, jobs: int, stopped, resource_exhausted, expected_draws: int | None = None,
          deadline: float | None = None, total: int | None = None) -> Sweep:
    """Run every vector in ``vectors`` with ``run`` (a fresh child per vector), ``jobs`` at a time.

    The entry point shared with the order-only rule: any finite candidate set can be swept.  The sweep covers nothing, and stops at the
    FIRST such run, when a vector's run is cut (timeout, crash, resource), carries a flag (uncontrolled nondeterminism, a fork), or draws
    more or fewer values than ``expected_draws`` (when given): ``incomplete``.  ``stopped()`` is the ledger's stop condition and
    ``deadline`` (a ``time.monotonic()`` value) the stage's own share of it, both checked before every batch: the first implementation
    (2fe973d1, an AI lane) checked only the ledger and kept running after a cut run, so a box of timing-out runs overran the budget."""
    import time
    from concurrent.futures import ThreadPoolExecutor
    started = time.monotonic()
    result = Sweep("complete")
    it = iter(vectors)
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        while True:
            batch = list(itertools.islice(it, max(1, jobs)))
            if not batch:
                return result
            if stopped():
                return Sweep("incomplete", result.runs, why="budget exhausted")
            if deadline is not None and time.monotonic() > deadline:
                return Sweep("incomplete", result.runs, why="the stage's share of the budget is spent")
            if deadline is not None and total and result.runs >= PROJECT_AFTER:
                # the measured rate must finish the box before the deadline, else stop now: a hopeless sweep must not burn the budget
                elapsed = time.monotonic() - started
                if elapsed + elapsed / result.runs * (total - result.runs) > deadline - started:
                    return Sweep("incomplete", result.runs, why=f"{result.runs} of {total} vectors in {elapsed:.0f}s projects past the deadline")
            outs = list(pool.map(lambda v: run(v, share=len(batch)), batch))
            result.runs += len(batch)
            for vec, out in zip(batch, outs):
                if out.status in ("timeout", "crashed") or resource_exhausted(out):
                    return Sweep("incomplete", result.runs, why=f"run {out.status}")
                hit = out.violated(classes)
                if hit:
                    return Sweep("violation", result.runs, vec, out, hit)
                if set(out.flags) - {"forked"}:        # "forked": the caller ran the vector in a fork of its warmed server (it certifies determinism first)
                    return Sweep("incomplete", result.runs, why=f"flags {sorted(out.flags)}")
                if expected_draws is not None and len(out.draws) != expected_draws:
                    # the run drew a different number of values than the box's draws: the box does not describe it
                    return Sweep("incomplete", result.runs, why=f"{len(out.draws)} draws, expected {expected_draws}")


# ---------------------------------------------------------------------------------------------------------------- the batched sweep

_BATCH_MODULES = frozenset({"_sv_verifier", "math", "string"})
_BATCH_NAMES = frozenset({
    "open", "input", "exec", "eval", "compile", "globals", "locals", "vars", "getattr", "setattr", "delattr", "__import__", "breakpoint",
    "id", "hash", "memoryview", "__builtins__", "builtins", "sys", "os", "time", "random", "setrecursionlimit"})


def batch_admissible(source: str) -> bool:
    """May ``sweep_batch`` vouch for this program's clean runs?  Only a program that imports ``_sv_verifier``, ``math`` and ``string``, uses nothing of the
    verifier but ``nondet_int``, and names no reflective, file, clock, random or interpreter-state facility: its runs then share nothing
    between vectors but the pure ``math`` module."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    modules, direct = _verifier_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(a.name not in _BATCH_MODULES for a in node.names):
                return False
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in _BATCH_MODULES or any(a.name == "*" for a in node.names):
                return False
            if node.module == "_sv_verifier" and any(a.name != "nondet_int" for a in node.names):
                return False
        elif isinstance(node, ast.Name) and node.id in _BATCH_NAMES:
            return False
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__") and node.attr not in ("__init__", "__name__"):
                return False
            if isinstance(node.value, ast.Name) and node.value.id in modules and node.attr != "nondet_int":
                return False
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom)):
            return False
    return True


def sweep_batch(box: Box, source: str, filename: str, python: str, run, classes, *, stopped, resource_exhausted,
                deadline: float | None = None, chunk: int = 4000, max_suspects: int = 200) -> Sweep:
    """Cover ``box`` with a few batched children instead of one child per vector (QuixBugs ``to_base``: 35,000 vectors).

    The batched child (``_sweep_child.py``) is trusted for one claim, "this vector ran to its end without raising"; every vector it lists as
    not clean goes back to ``run`` (the engine's own fresh child), which alone decides a violation, a cut run or a draw-count mismatch; the one exception
    is a plain ``Exception`` after all its draws none of whose classes (MRO names) is a property class, which cannot be a violation (QuixBugs
    ``to_base_1``, the buggy twin, fails its assertion on most vectors: not an arithmetic or data-lookup violation).
    The first, a middle and the last vector are also run by ``run`` and must be clean with no flag, so a batch environment that differs
    from the engine's is caught.  Over ``max_suspects`` suspects the sweep gives up (``incomplete``).  The caller admits the program
    (:func:`batch_admissible`) and certifies determinism first."""
    import json
    import os
    import subprocess
    import time
    expected = len(box.names)
    result = Sweep("complete")
    vectors = list(box.vectors())
    values = [[e["value"] for e in v] for v in vectors]

    def real(vec):
        result.runs += 1
        out = run(vec, share=1)
        if out.status in ("timeout", "crashed") or resource_exhausted(out):
            return Sweep("incomplete", result.runs, why=f"run {out.status}")
        hit = out.violated(classes)
        if hit:
            return Sweep("violation", result.runs, vec, out, hit)
        if set(out.flags) - {"forked"}:
            return Sweep("incomplete", result.runs, why=f"flags {sorted(out.flags)}")
        if len(out.draws) != expected:
            return Sweep("incomplete", result.runs, why=f"{len(out.draws)} draws, expected {expected}")
        return None

    for probe in (vectors[0], vectors[len(vectors) // 2], vectors[-1]):
        verdict = real(probe)
        if verdict is not None:
            return verdict
    child = os.path.join(os.path.dirname(__file__), "_sweep_child.py")
    suspects = 0
    for start in range(0, len(vectors), chunk):
        if stopped():
            return Sweep("incomplete", result.runs, why="budget exhausted")
        left = None if deadline is None else deadline - time.monotonic()
        if left is not None and left <= 0:
            return Sweep("incomplete", result.runs, why="the stage's share of the budget is spent")
        request = json.dumps({"source": source, "filename": filename, "vectors": values[start:start + chunk], "alarm": 5})
        try:
            done = subprocess.run([python, "-I", "-S", child], input=request, capture_output=True, text=True,
                                  timeout=min(120.0, left) if left is not None else 120.0)
            reply = json.loads(done.stdout)
        except (subprocess.TimeoutExpired, ValueError, OSError):
            return Sweep("incomplete", result.runs, why="the batched child did not answer")
        if done.returncode != 0 or reply.get("n") != len(values[start:start + chunk]):
            return Sweep("incomplete", result.runs, why="the batched child failed")
        for index, status, exception, used, mro, plain in reply["suspect"]:
            if plain and used == expected and not set(mro) & set(classes):
                continue        # an Exception none of whose classes is a property class, after all its draws: no violation, as the engine's child would rule
            suspects += 1
            if suspects > max_suspects:
                return Sweep("incomplete", result.runs, why=f"more than {max_suspects} vectors did not finish cleanly in the batch")
            verdict = real(vectors[start + index])
            if verdict is not None:
                return verdict
    result.runs += len(vectors)
    return result
