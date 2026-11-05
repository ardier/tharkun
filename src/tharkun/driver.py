"""Directed reachability driver: solve in a loop until a target line is reached.

A single guard flip almost never lands a deep target.  You flip the first
blocking guard, execute, and discover the *next* blocker that the first trace
never revealed — because execution didn't get that far.  The loop is the
product; a one-shot solve is its inner primitive.

``drive_to_line`` runs a SymCC-style concolic fixpoint over ordinary Python:

    trace a seed  →  find the blocking guard on the path to the target
                  →  flip it (snapshot, then static and search)
                  →  merge the flip into the inputs, re-trace
                  →  repeat until the target executes, or a budget is spent

The exit condition is *observed*, never claimed: every round re-executes and
checks whether the target line ran.  The tracer is the judge; a solver's output
is only a proposal, so an unsound (heuristic) proposal is harmless — a wrong
flip simply fails the next trace and the loop moves on.

Scope (v1): the driver controls **function parameters**.  A blocking guard whose
controlling variables are all locals computed from parameters needs backward
reasoning (the call_mapper / P6 path) and is reported as a plateau with
diagnostics rather than silently skipped.  [internal ref] slice 4 (``_flip_decompose``)
wires up one NARROW instance of that backward reasoning — a guard that is a
single bare call to a helper function, via ``solve/decompose.py`` +
``solve/frontend/call_mapper.py`` — everything else the sentence above describes
(compound conditions, multi-level call chains) is still a plateau.
"""

from __future__ import annotations

import ast
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from tharkun.trace_api import TracePolicy, Tracer

# Outcome tags for DriveResult.status.
REACHED = "reached"
PLATEAU = "plateau"            # no flippable frontier — stuck short of the target
BUDGET = "budget_exhausted"   # ran out of rounds
NO_TARGET = "no_target"       # target line is not inside any guard / bad input


@dataclass
class GuardFrame:
    """An ``if`` guard that encloses the target line."""

    branch_line: int              # the `if` line (the guard)
    desired_branch: str           # "then" | "else" — the side that contains the target
    then_span: tuple[int, int]
    else_span: tuple[int, int] | None
    depth: int                    # nesting depth (0 = outermost)


@dataclass
class DriveStep:
    """One round of the loop, for diagnostics/replay."""

    round: int
    inputs: dict[str, Any]
    blocker_line: int | None = None
    desired_branch: str | None = None
    technique: str | None = None      # which flipper produced the proposal
    note: str = ""


@dataclass
class DriveResult:
    status: str
    inputs: dict[str, Any]
    rounds: int = 0
    target_line: int = 0
    reached: bool = False
    history: list[DriveStep] = field(default_factory=list)
    diagnostics: str = ""
    # True when a randomized search was part of what failed. A deterministic portfolio
    # that comes up empty will come up empty again on identical state; a search will not
    # necessarily, so a caller deciding whether to write this frontier off needs to know
    # which kind of failure it was.
    used_stochastic: bool = False


# ---------------------------------------------------------------------------
# Frontier: which guards enclose the target
# ---------------------------------------------------------------------------


def _span(body: list[ast.stmt]) -> tuple[int, int] | None:
    if not body:
        return None
    lo = body[0].lineno
    hi = max(getattr(s, "end_lineno", s.lineno) or s.lineno for s in body)
    return (lo, hi)


def _in_span(line: int, span: tuple[int, int] | None) -> bool:
    return span is not None and span[0] <= line <= span[1]


def enclosing_guards(tree: ast.Module, target_line: int) -> list[GuardFrame]:
    """Guards whose then/else body contains *target_line*, outermost first.

    Nested ``elif`` is represented by CPython as an ``If`` in the ``orelse`` of
    its parent; each level is reported as its own frame with the polarity that
    holds the target.
    """
    frames: list[GuardFrame] = []

    def visit(node: ast.AST, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                then_span = _span(child.body)
                else_span = _span(child.orelse)
                desired: str | None = None
                if _in_span(target_line, then_span):
                    desired = "then"
                elif _in_span(target_line, else_span):
                    desired = "else"
                if desired is not None and target_line != child.lineno:
                    frames.append(
                        GuardFrame(
                            branch_line=child.lineno,
                            desired_branch=desired,
                            then_span=then_span or (child.lineno, child.lineno),
                            else_span=else_span,
                            depth=depth,
                        )
                    )
                    visit(child, depth + 1)
                    continue
            visit(child, depth)

    visit(tree, 0)
    frames.sort(key=lambda f: f.depth)
    return frames


# ---------------------------------------------------------------------------
# Trace execution
# ---------------------------------------------------------------------------


@dataclass
class _TraceOutcome:
    executed: set[int]
    snapshots: dict[int, list[dict[str, Any]]]
    ok: bool
    error: str = ""
    # Live (non-serialized) objects at each snapshot line, when capture_real was set.
    # Minleaf projects leaves off these, so it needs the real objects, not reprs.
    real: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    # Lines executed per file, for targets that live BELOW the entry function. A line in
    # a transitive callee is in the coverage report exactly like one in the entry
    # module, so it has to be observable to be driven to.
    by_path: dict[str, set[int]] = field(default_factory=dict)
    # Snapshots per file, serialized and live. A blocker in a CALLEE has its state in
    # that file, so reading only the entry file's leaves minleaf and redqueen with
    # nothing to work on — measured, they were invoked 30 times against the cheap tier's
    # 1158 and never once contributed.
    snapshots_by_path: dict[str, dict[int, list[dict[str, Any]]]] = field(default_factory=dict)
    real_by_path: dict[str, dict[int, list[dict[str, Any]]]] = field(default_factory=dict)


def _mangled(owner: str, name: str) -> str:
    """Python's private-name mangling: ``__m`` written in class ``C`` is stored as ``_C__m``."""
    stripped = owner.lstrip("_")
    return f"_{stripped}{name}" if name.startswith("__") and not name.endswith("__") and stripped else name


def _entry_callable(mod: Any, entry: str) -> Any:
    """The callable ``entry`` names in ``mod``: a top-level function, or ``Class.method`` ([internal ref], nested classes
    ``Outer.Inner.method`` too, a ``__private`` method under its mangled name).  A static or class method is bound as
    Python binds it; an instance method is the plain function, so a seed names its receiver as ``self``.  ``None``
    when ``mod`` has no such thing."""
    if mod is None:
        return None
    obj, parts = mod, entry.split(".")
    for i, part in enumerate(parts):
        obj = getattr(obj, _mangled(parts[i - 1], part) if i else part, None)
        if obj is None:
            return None
    return obj


def _def_node(tree: ast.AST, entry: str):
    """The ``def`` node of a top-level function or of ``Class.method`` (:func:`_entry_callable`'s names), else ``None``."""
    body, node = getattr(tree, "body", []), None
    for part in entry.split("."):
        found = [st for st in body if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                 and st.name == part]
        if not found:
            return None
        node = found[-1]
        body = node.body
    return node if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else None


def _run_traced(
    source_path: str,
    func_name: str,
    inputs: dict[str, Any],
    snapshot_lines: set[int],
    *,
    capture_real: bool = False,
    extra_paths: frozenset[str] = frozenset(),
    extra_snapshot_lines: dict[str, set[int]] | None = None,
) -> _TraceOutcome:
    """Execute ``func_name(**inputs)`` in *source_path* under the tracer.

    ``capture_real`` additionally retains the live objects at each snapshot line — needed
    by minleaf, which mutates real state rather than reasoning about a serialized copy.

    ``extra_paths`` widens the trace to files the entry function *calls*.  Without it the
    tracer sees one file, so a target line in a transitive callee is invisible and can
    never be driven to — even though it appears in the coverage report like any other.
    """
    from tharkun.branch_solver import _load_module_for_execution

    rp = os.path.realpath(source_path)
    mod, _ = _load_module_for_execution(rp)
    fn = _entry_callable(mod, func_name)
    if fn is None:
        return _TraceOutcome(set(), {}, ok=False, error=f"cannot load {func_name} from {rp}")

    watched = {rp} | {os.path.realpath(p) for p in extra_paths}
    snaps = {rp: set(snapshot_lines)}
    for path, lines in (extra_snapshot_lines or {}).items():
        snaps.setdefault(os.path.realpath(path), set()).update(lines)
    policy = TracePolicy(enabled=True, paths=watched, snapshot_lines=snaps,
                         capture_real=capture_real)
    error = ""
    try:
        with Tracer(policy) as tr:
            fn(**inputs)
    except Exception as exc:  # the subject raised — still a valid trace up to the raise
        error = f"{type(exc).__name__}: {exc}"

    return _TraceOutcome(
        set(tr.result.executed.get(rp, set())),
        dict(tr.result.snapshots.get(rp, {})),
        ok=True,
        error=error,
        real=dict(tr.result.snapshots_real.get(rp, {})),
        by_path={k: set(v) for k, v in tr.result.executed.items()},
        snapshots_by_path={k: dict(v) for k, v in tr.result.snapshots.items()},
        real_by_path={k: dict(v) for k, v in tr.result.snapshots_real.items()},
    )


def _iter_code_objects(root):
    """Every code object reachable from *root* (a module, class, or function), recursively.

    Descends into CLASSES so a target inside a method (des.py's ``DES.encrypt``,
    ``key56_to_key64``) is found, not just module-level functions; unwraps static/class
    methods; and walks ``co_consts`` so nested functions/comprehensions are reached."""
    import types as _types

    seen: set = set()
    stack: list = []
    scoped_seen: set = set()

    def add_members(container) -> None:
        try:
            members = list(vars(container).values())
        except TypeError:
            return
        for v in members:
            fn = v.__func__ if isinstance(v, (staticmethod, classmethod)) else v
            code = getattr(fn, "__code__", None)
            if isinstance(code, _types.CodeType):
                stack.append(code)
            elif isinstance(v, type) and id(v) not in scoped_seen:
                scoped_seen.add(id(v))
                add_members(v)                        # nested class

    if isinstance(root, (_types.ModuleType, type)):
        add_members(root)
    else:
        code = getattr(root, "__code__", root)
        if isinstance(code, _types.CodeType):
            stack.append(code)
    while stack:
        code = stack.pop()
        if id(code) in seen:
            continue
        seen.add(id(code))
        yield code
        for const in code.co_consts:
            if isinstance(const, _types.CodeType):
                stack.append(const)


def _code_for_line(root, rtp: str, target_line: int):
    """The code object in file *rtp* whose OWN lines include *target_line*, or None.

    A code object's own lines come from ``co_lines()`` — nested functions carry their own,
    so this picks the innermost enclosing scope, which is exactly what we want to
    instrument for a reach check (leaving heavy callees uninstrumented)."""
    import types as _types

    for code in _iter_code_objects(root):
        if os.path.realpath(getattr(code, "co_filename", "")) != rtp:
            continue
        try:
            for _start, _end, ln in code.co_lines():
                if ln == target_line:
                    return code
        except Exception:
            continue
    return None


def _target_reached_probe(source_path: str, func_name: str, inputs: dict[str, Any],
                          target_line: int, tp: str, extra_paths: frozenset[str]):
    """Fast "did *target_line* execute" check, or None to signal fall back to a full trace.

    Instruments ONLY the code object that contains the target line, via ``sys.monitoring``
    local events — so the heavy callees the target passes through run UNINSTRUMENTED. On a
    3-DES-per-call target that is ~4x faster than whole-program ``sys.settrace`` (measured),
    and the gap grows with callee weight. Returns None (→ caller uses the full trace) when
    monitoring is unavailable, the module or the enclosing code object can't be found, or a
    tool id can't be claimed, so correctness never depends on the fast path.
    """
    m = getattr(sys, "monitoring", None)
    if m is None:
        return None
    from tharkun.branch_solver import _load_module_for_execution

    rp = os.path.realpath(source_path)
    rtp = os.path.realpath(tp)
    mod, _ = _load_module_for_execution(rp)
    fn = _entry_callable(mod, func_name)
    if fn is None:
        return None
    tmod = mod if rtp == rp else _load_module_for_execution(rtp)[0]
    if tmod is None:
        return None
    target_code = _code_for_line(tmod, rtp, target_line)
    if target_code is None:
        return None
    tool = None
    for tid in (5, 4, 3, 2, 1, 0):
        try:
            m.use_tool_id(tid, "tharkun_reach")
            tool = tid
            break
        except ValueError:
            continue
    if tool is None:
        return None
    hit = [False]

    def on_line(code, lineno):
        if lineno == target_line:
            hit[0] = True
        return m.DISABLE                    # this location won't fire again this run

    try:
        m.register_callback(tool, m.events.LINE, on_line)
        m.set_local_events(tool, target_code, m.events.LINE)
        try:
            fn(**inputs)
        except Exception:
            pass                            # a raise mid-run is still a valid partial trace
    except Exception:
        return None
    finally:
        try:
            m.set_local_events(tool, target_code, 0)
            m.register_callback(tool, m.events.LINE, None)
        finally:
            m.free_tool_id(tool)
    return hit[0]


def _reaches_in_fresh_state(
    source_path: str, func_name: str, inputs: dict[str, Any], target_line: int,
    tp: str, extra_paths: frozenset[str], extra_snaps: dict[str, set[int]],
) -> bool:
    """Re-run *inputs* once with the target's PROJECT modules re-imported fresh; report
    whether *target_line* still executes.

    The driver runs the target in-process against modules cached in ``sys.modules``, so a
    module-level mutable counter accumulates across the drive's traces and can flip a
    counter guard to a false REACHED (a fresh single call with the same inputs would not
    reach). Popping the project's own modules — those whose file lives under the entry or
    target directory, never tharkun's or the stdlib's — forces a clean import so this call
    sees the state a first invocation would; the originals are restored afterward so the
    surrounding process keeps its continuity.

    Conservative: any failure to re-verify (a module without ``__file__``, a load error, an
    exception before the target) returns True — it never turns a real reach into a plateau
    on the strength of a re-verify that could not run.
    """
    import sys as _sys

    rp = os.path.realpath(source_path)
    rtp = os.path.realpath(tp)
    roots = {d for d in (os.path.dirname(rp), os.path.dirname(rtp)) if d}
    saved: dict[str, Any] = {}
    for modname, m in list(_sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        try:
            rf = os.path.realpath(f)
        except Exception:
            continue
        if any(rf == root or rf.startswith(root + os.sep) for root in roots):
            saved[modname] = m
    for modname in saved:
        del _sys.modules[modname]
    try:
        oc = _run_traced(source_path, func_name, inputs, {target_line},
                         extra_paths=extra_paths, extra_snapshot_lines=extra_snaps)
    except Exception:
        return True                       # cannot re-verify → do not override the reach
    finally:
        for modname, m in saved.items():
            _sys.modules[modname] = m
    if not oc.ok:
        return True
    executed = oc.by_path.get(rtp, set()) if rtp != rp else oc.executed
    return target_line in executed


# ---------------------------------------------------------------------------
# Flip a single blocking guard
# ---------------------------------------------------------------------------

# A flipper returns (inputs | None, technique, is_infeasible).  is_infeasible is
# True only when a *sound* procedure certified the guard-side dead.  Signature:
#   flip(source_path, branch_line, desired, snapshot, params, requires)
Flipper = Callable[..., "tuple[dict[str, Any] | None, str, bool]"]


_STORE: Any = None          # the run's KnowledgeBase, set by drive_to_line


def _new_store():
    """A fresh KnowledgeBase for a drive with no shared store threaded in."""
    from tharkun.knowledge import KnowledgeBase

    return KnowledgeBase()


def _flip_snapshot(
    source_path: str, branch_line: int, desired: str, snapshot: dict[str, Any],
    params: set[str], requires: str,
) -> tuple[dict[str, Any] | None, str, bool]:
    from tharkun.solve.condition import solve_with_snapshot

    sol = solve_with_snapshot(
        source_path, branch_line, snapshot, param_names=params, desired_branch=desired
    )
    for cand in sol.satisfying_inputs:
        applicable = {k: v for k, v in cand.items() if k in params}
        if applicable:
            return applicable, f"snapshot:{sol.strategy}", False
    return None, "snapshot", False


def _bank_static_facts(store, sol, params: set[str]) -> None:
    """Record what the static solver inferred, so later strategies can use it.

    Deliberately not certified: static inference is a best guess (an untyped parameter
    compared to a literal is *probably* that type), and the store keeps provenance so a
    consumer can prefer a sound source when one exists.
    """
    from tharkun.facts import Fact

    for vc in getattr(sol, "variables", None) or []:
        if vc.name not in params:
            continue
        if vc.inferred_type and vc.inferred_type not in ("Any", "object"):
            store.add(Fact("type", vc.name, vc.inferred_type,
                           source=f"static:{vc.inference_source}", confidence=0.7))
        for v in (vc.concrete_values or [])[:8]:
            store.add(Fact("candidate", vc.name, v,
                           source=f"static:{vc.inference_source}", confidence=0.7))


def _flip_static(
    source_path: str, branch_line: int, desired: str, snapshot: dict[str, Any],
    params: set[str], requires: str,
) -> tuple[dict[str, Any] | None, str, bool]:
    from tharkun.solve.condition import solve_condition

    from tharkun.facts import DecisionKey

    # The static solver is pure: identical queries share one cached answer.
    key = DecisionKey.of(source_path, branch_line, desired, requires)
    sol = (_STORE.lookup_or_run(key, "static",
                                lambda: solve_condition(source_path, branch_line,
                                                        desired_branch=desired))
           if _STORE is not None else
           solve_condition(source_path, branch_line, desired_branch=desired))
    # A REFUTED arm (owner 1365: every match of a finditer pattern is non-empty, so `if not chunk:` never holds) is a sound
    # infeasibility vote: the candidate the arm asks for ('' for chunk) is rejected by the pattern's own minimum length.
    if sol.strategy == "infeasible" and (sol.input_constraints or {}).get("refutation"):
        return None, "static:refuted", True
    # The static solver infers a type and a set of candidate values for every variable in
    # the guard, and until now that died with the call. Those are exactly what a search
    # needs to build a strategy from — a type to pick the generator and the guard's own
    # literals to seed its alphabet — so bank them whether or not the solve produced a
    # usable input. A solve that fails to produce inputs can still have inferred the type.
    if _STORE is not None and sol is not None:
        _bank_static_facts(_STORE, sol, params)
    # Snapshot types aim candidate generation only: an integer UNSAT is never a
    # dead-code proof for a Python parameter that may have another type.
    from tharkun.solve.arith import solve_int_guard

    ints = {name for name, value in snapshot.items() if type(value) is int}
    if ints and sol.guard_source:
        guard = f"({sol.guard_source})"
        if desired == "else":
            guard = f"not {guard}"
        result = solve_int_guard(f"({requires}) and ({guard})", ints)
        if result.status == "sat":
            applicable = {k: v for k, v in result.witness.items() if k in params}
            if applicable:
                return applicable, "static:arith", False
    for cand in sol.satisfying_inputs:
        applicable = {k: v for k, v in cand.items() if k in params}
        if applicable:
            return applicable, f"static:{sol.strategy}", False
    return None, "static", False


# ---------------------------------------------------------------------------
# [internal ref] slice 4: three more cheap-tier flippers, ported from branch_solver.py's
# per-branch toolset (tulkas's coverage solver) into drive_to_line's portfolio.
# Each targets a guard SHAPE ``_flip_snapshot``/``_flip_static`` cannot see past
# at all (a regex match, a ``self.xxx`` guard, a call to a helper function) rather
# than competing with them on shapes they already solve — so ordering them after
# the two general flippers is safe: those two simply abstain (return ``None``) on
# these shapes today, and continue to run first everywhere else. Every candidate
# these produce goes through the SAME ``_verified``/``_ok`` execution-replay gate
# every other flipper's candidate does (drive_to_line's own docstring: "a solver's
# output is only a proposal, so an unsound (heuristic) proposal is harmless"), so
# a wrong guess from any of the three costs one extra verify round, never a wrong
# commit — no new soundness surface, only new proposals.
# ---------------------------------------------------------------------------


def _flip_regex(
    source_path: str, branch_line: int, desired: str, snapshot: dict[str, Any],
    params: set[str], requires: str,
) -> tuple[dict[str, Any] | None, str, bool]:
    """Regex-guard flipper: branch_solver.py's regex-inversion tool
    (``solve/condition.py::_extract_regex_pattern`` / ``_generate_from_regex``),
    ported into the driver's cheap tier.

    When a guard matches a parameter against a string/bytes literal
    (``re.match(r"...", name)``) or a module-level ``re.compile(...)`` constant
    (``PAT.fullmatch(name)``), inverting the PATTERN (Hypothesis' ``from_regex``)
    finds a satisfying value directly instead of hoping a blind search stumbles
    onto one — the same kind of gap RedQueen closes for equality guards, applied
    to pattern-shaped ones instead, which no comparison-operand harvest ever sees.
    """
    from tharkun.solve.condition import (
        _extract_regex_pattern,
        _find_branch_at_line,
        _free_names,
        _generate_from_regex,
    )
    from tharkun.solve.constants import BUILTIN_NAMES

    try:
        source = Path(source_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None, "regex", False
    node = _find_branch_at_line(tree, branch_line)
    if node is None:
        return None, "regex", False

    for name in sorted(_free_names(node) & (params - BUILTIN_NAMES)):
        spec = _extract_regex_pattern(name, node, source_path)
        if spec is None:
            continue
        pattern, flags, is_bytes, _const_name = spec
        if desired == "then":
            val = _generate_from_regex(pattern, flags=flags, is_bytes=is_bytes)
        else:
            # "else": no pattern inversion for a NEGATIVE match is attempted (that
            # needs the pattern's complement, which ``from_regex`` cannot draw
            # from) — the empty string/bytes is not a match for any pattern that
            # requires at least one character, which covers the common case
            # cheaply. Verified like every other candidate; a pattern that DOES
            # accept "" just falls through to the next tier, no worse off.
            val = b"" if is_bytes else ""
        if val is not None:
            return {name: val}, "regex", False
    return None, "regex", False


def _extract_required_attr_value(guard_source: str, attr_full: str) -> Any:
    """The value a guard needs *attr_full* to equal — the comparison constant on
    the other side of ``self.attr == <const>`` (or ``<const> == self.attr``).

    A small local twin of ``branch_solver.py``'s own ``_extract_required_attr_value``
    (same logic, including its known imprecision: it does not check the comparison
    OPERATOR, so a ``!=`` reads the same as ``==``) — kept local rather than
    imported so this module's dependency surface stays inside ``tharkun.solve.*``
    (see the module docstring and LAYERS.md); ``branch_solver.py`` is a peer
    orchestrator built on top of this layer, not a layer driver.py sits above.
    A wrong extraction here is caught by execution-verify like every other
    flipper's candidate, so the imprecision is a precision limit, not a soundness
    one — unlike branch_solver.py's own ``state_sequence`` solver, which never
    verifies this value by execution at all.
    """
    from tharkun.solve.condition import _unparse

    try:
        expr = ast.parse(guard_source, mode="eval").body
    except SyntaxError:
        return None
    for node in ast.walk(expr):
        if not isinstance(node, ast.Compare):
            continue
        if _unparse(node.left) == attr_full:
            for comp in node.comparators:
                if isinstance(comp, ast.Constant):
                    return comp.value
        for comp in node.comparators:
            if _unparse(comp) == attr_full and isinstance(node.left, ast.Constant):
                return node.left.value
    return None


def _flip_attr(
    source_path: str, branch_line: int, desired: str, snapshot: dict[str, Any],
    params: set[str], requires: str,
) -> tuple[dict[str, Any] | None, str, bool]:
    """Attribute-guard flipper: ``solve/attrs.py``'s tracer
    (``extract_attr_targets`` / ``find_attr_assignments``), ported for the ONE
    shape drive_to_line's single-call model can act on directly — a guard on
    ``self.attr``/``cls.attr`` whose value flows straight from an ``__init__``
    parameter that shares a name with one of the entry's own controlled
    parameters (``self.attr = attr`` in ``__init__(self, attr, ...)``).

    branch_solver.py's own richer ``state_sequence`` solver (construct the
    object, call setup methods in sequence, THEN reach the guard) has nothing to
    bind to here: drive_to_line flips one call's parameters, it does not
    sequence multiple method calls — a real, documented scope limit (see the
    module docstring's "Scope (v1)"), not an oversight.
    """
    if desired != "then":
        # required-value extraction only reads an equality-shaped constant (see
        # _extract_required_attr_value); inverting to "make it NOT equal that
        # constant" is a different, unimplemented question, so this abstains
        # rather than propose the same value for the opposite branch.
        return None, "attr", False
    from tharkun.solve.attrs import extract_attr_targets, find_attr_assignments
    from tharkun.solve.condition import _find_branch_at_line, _unparse

    try:
        source = Path(source_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None, "attr", False
    node = _find_branch_at_line(tree, branch_line)
    if node is None:
        return None, "attr", False
    guard_source = _unparse(node) or ""

    for target in extract_attr_targets(node):
        for assignment in find_attr_assignments(source_path, target):
            if not assignment.is_init or assignment.rhs_source not in params:
                continue
            required = _extract_required_attr_value(guard_source, target.full)
            if required is None:
                continue
            return {assignment.rhs_source: required}, "attr", False
    return None, "attr", False


def _flip_decompose(
    source_path: str, branch_line: int, desired: str, snapshot: dict[str, Any],
    params: set[str], requires: str,
) -> tuple[dict[str, Any] | None, str, bool]:
    """Guard-decomposer flipper: branch_solver.py's step 2
    (``solve/decompose.py::decompose_guard``), ported as a cheap-tier flip
    strategy — the module docstring's own "backward reasoning (the call_mapper /
    P6 path)" for the narrow case below.

    ``solve_condition``/``_flip_static`` cannot see past ``if we_are_good(config):``
    — a call is opaque to a guard-AST-only solve. The decomposer resolves the
    callee and descends into its own ``return`` condition; this flipper supplies
    the missing last step — mapping the callee's OWN parameter names back to the
    caller's argument names (``solve/frontend/call_mapper.py::map_call_site``) so
    a leaf constraint found INSIDE the callee ends up proposed against one of the
    ENTRY's own controlled parameters, not a name that means nothing outside it.

    Scope, deliberately narrow for this first cut (a documented, honest gap, not
    a silent one): only a guard that IS a single, bare, plain-name, positional
    call (``if helper(x):``) — the decomposer's own headline example — with
    every leaf in the callee's return condition solvable and none of them under
    an ``or``/``not`` (De Morgan negation is not attempted: getting the polarity
    of a mixed and/or/not tree wrong is a real way to propose something that can
    never verify, so this abstains rather than guess). ``desired == "else"``
    abstains too: decompose_guard characterizes what makes the guard's THEN side
    true, not its negation. Compound top-level conditions
    (``if a and helper(x):``), keyword arguments, and call chains two levels
    deep are left to a later pass — not attempted, not guessed at.
    """
    if desired != "then":
        return None, "decompose", False
    from tharkun.solve.condition import _find_branch_at_line
    from tharkun.solve.decompose import decompose_guard
    from tharkun.solve.frontend.call_mapper import map_call_site

    try:
        source = Path(source_path).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None, "decompose", False
    node = _find_branch_at_line(tree, branch_line)
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
        return None, "decompose", False   # scope: a single bare positional call only

    mapping = map_call_site(source_path, branch_line, callee_name=node.func.id)
    if mapping is None or mapping.callee_func_def is None:
        return None, "decompose", False
    name_map = {m.callee_param: m.caller_name for m in mapping.args if m.is_simple_name}
    if not name_map:
        return None, "decompose", False

    try:
        decomp = decompose_guard(source_path, branch_line, max_depth=4)
    except Exception:
        return None, "decompose", False
    if decomp.solvable_leaves == 0 or decomp.unsolvable_leaves != 0:
        return None, "decompose", False

    candidate: dict[str, Any] = {}
    ok = True

    def _walk(n) -> None:
        nonlocal ok
        if not ok:
            return
        if n.kind in ("or", "not"):
            ok = False   # De Morgan not attempted — abstain, do not guess wrong.
            return
        if n.children:
            for c in n.children:
                _walk(c)
            return
        vc = n.var_constraint
        if vc is None or not vc.concrete_values:
            return       # a constant / value-less leaf — nothing to bind, no-op.
        caller_name = name_map.get(vc.name)
        if caller_name is None or caller_name not in params:
            ok = False   # the leaf's name lives in a scope this cut doesn't map.
            return
        candidate[caller_name] = vc.concrete_values[0]

    _walk(decomp.root)
    if not ok or not candidate:
        return None, "decompose", False
    return candidate, "decompose", False


# Live-state snapshots and static solving form the cheap tier. [internal ref] slice 4 adds
# three more cheap, static, guard-SHAPE-specific flippers (regex / attr / call
# decomposition) that the two general ones above abstain on entirely — see the
# block comment above ``_flip_regex`` for why appending them here is safe.
_DEFAULT_FLIPPERS: list[Flipper] = [
    _flip_snapshot, _flip_static, _flip_regex, _flip_attr, _flip_decompose,
]


# Heavy tier — property-based search, gated on Hypothesis availability.
# It runs only when the cheap tier produced
# no verifying candidate ("try everything, escalate lazily").
def _flip_hypothesis(
    source_path: str, branch_line: int, desired: str, func_name: str, params: set[str]
) -> tuple[dict[str, Any] | None, str]:
    from tharkun.hypothesis_adapter import hypothesis_available, hypothesis_solve_branch

    if not hypothesis_available():
        return None, "hypothesis"
    # A last-resort search: budget more examples than the per-branch chain's default.
    got = hypothesis_solve_branch(source_path, branch_line, func_name,
                                  desired_branch=desired, max_examples=500)
    if got:
        applicable = {k: v for k, v in got.items() if k in params}
        if applicable:
            return applicable, "hypothesis"
    return None, "hypothesis"


def _fact_literals(clauses: list[str]) -> tuple[list[int], list[bytes], list[str]]:
    """Constants appearing in the facts, to search over rather than hope to draw."""
    ints: list[int] = []
    bys: list[bytes] = []
    strs: list[str] = []
    for c in clauses:
        try:
            tree = ast.parse(c, mode="eval")
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant):
                v = n.value
                if isinstance(v, bool):
                    continue
                if isinstance(v, int) and v not in ints:
                    ints.append(v)
                elif isinstance(v, bytes) and v not in bys:
                    bys.append(v)
                elif isinstance(v, str) and v not in strs:
                    strs.append(v)
    return ints, bys, strs


_TYPE_SAMPLES: dict[str, Any] = {
    "int": 0, "float": 0.0, "bool": True, "str": "", "bytes": b"",
    "list": [], "tuple": (), "bytearray": bytearray(),
}


def _strategy_over_types(type_names: list[str], clauses: list[str] | None = None):
    """A strategy spanning every plausible type, not just the most likely one.

    Python is dynamically typed: a parameter can legitimately be bytes on one path and
    str on another, and the signals that disagree are usually each right about a path we
    have not taken yet. Generating over the union costs a wider search but cannot rule
    out the answer a priori, which betting on the mode can.
    """
    import hypothesis.strategies as st

    built = [_strategy_for(_TYPE_SAMPLES[t], clauses)
             for t in type_names if t in _TYPE_SAMPLES]
    built = [b for b in built if b is not None]
    if not built:
        return None
    return built[0] if len(built) == 1 else st.one_of(*built)


def _strategy_kind(value: Any, type_override: str | None) -> str | None:
    """The effective strategy kind: an explicit override outranks the live value's type.

    The override is a KnowledgeBase prior/observation ('int', 'str', 'bytes', …) — used
    when it is more trustworthy than the seed value's shape (a param seeded ``b''`` that the
    program actually runs as a str). The alphabet is kept either way; only the branch changes.
    """
    if type_override is not None:
        alias = {"bytearray": "bytes", "tuple": "list"}.get(type_override, type_override)
        return alias if alias in ("bool", "int", "float", "bytes", "str", "list") else None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, bytes):
        return "bytes"
    if isinstance(value, str):
        return "str"
    if isinstance(value, (list, tuple)):
        return "list"
    return None


def _strategy_for(value: Any, clauses: list[str] | None = None,
                  type_override: str | None = None):
    """A Hypothesis strategy matching the type the program is actually running with.

    Types come from the LIVE value, not from static inference — that is the point. Static
    inference reads `p[0] == 0x7F` as "p is subscriptable" and offers a list; the running
    program says `p` is bytes, and only bytes can satisfy it.

    ``type_override`` lets a caller impose a type the KnowledgeBase is MORE sure of than the
    seed's shape — the prior/observed type from the lattice — WITHOUT losing the alphabet
    below, which a bare type-only strategy would. That distinction is what the reach-search
    needs: the observed type picks the branch, the hint literals still fill the pool.

    The alphabet is drawn from the FACTS' own literals as well. Uniform ``binary()`` has
    to stumble onto ``0x7F`` at index 0 — about a 1-in-256 draw, measured at 4 reached
    runs out of 6 — while the byte the comparison demands is written in the comparison.
    Putting it in the pool is the same idea RedQueen uses, applied to the search's input
    distribution rather than to a single copy.
    """
    import hypothesis.strategies as st

    ints, bys, strs = _fact_literals(clauses or [])
    kind = _strategy_kind(value, type_override)

    if kind == "bool":
        return st.booleans()
    if kind == "int":
        pool = [i for i in ints if -(2 ** 63) < i < 2 ** 63]
        base = st.integers(min_value=-(2 ** 31), max_value=2 ** 31 - 1)
        # Neighbours too: a guard is as often `> k` as `== k`.
        near = [i + d for i in pool for d in (-1, 0, 1)]
        return st.one_of(st.sampled_from(near), base) if near else base
    if kind == "float":
        return st.floats(allow_nan=False, allow_infinity=False)
    if kind == "bytes":
        # A demanded integer reaches a bytes field as its ENCODING. Masking it to one
        # byte throws away everything above the low 8 bits: the observed operand 13987
        # became just b"\xa3", losing the b"\x36" that made it that number, so a search
        # aimed by a perfectly good fact still could not build the answer. Offer both
        # byte orders and the widths a header field actually uses.
        encodings: list[bytes] = []
        for i in ints:
            if i < 0:
                continue
            for width in (1, 2, 4, 8):
                if i >> (8 * width):
                    continue
                for order in ("big", "little"):
                    enc = i.to_bytes(width, order)
                    if enc not in encodings:
                        encodings.append(enc)
        alphabet = bytes(dict.fromkeys([i & 0xFF for i in ints] + [0, 0x41]))
        chunks = encodings + [b for b in bys] + [bytes([c]) for c in alphabet]
        return st.one_of(
            st.lists(st.sampled_from(chunks), max_size=32).map(b"".join),
            st.binary(max_size=64),
        )
    if kind == "str":
        seeds = [t for t in strs] + [chr(i) for i in ints if 0 <= i < 0x110000]
        base = st.text(max_size=64)
        return (st.one_of(st.lists(st.sampled_from(seeds), max_size=32).map("".join), base)
                if seeds else base)
    if kind == "list":
        return st.lists(st.integers(min_value=0, max_value=255), max_size=32)
    return None


def _flip_hypothesis_facts(
    facts: list[str], goal: str | None, live: dict[str, Any], params: set[str],
) -> tuple[dict[str, Any] | None, str]:
    """Search for one assignment satisfying the WHOLE accumulated fact set at once.

    The loop solves guards one at a time, so an answer to an early guard can make a later
    one unsatisfiable — `len(p) > 4` answered with a str kills `p[0] == 0x7F`. Backtracking
    re-solves the earlier guard under the later one's requirement, but a solver that only
    ever sees one guard can still walk into the same wall.

    This closes it from the other side: hand Hypothesis the CONJUNCTION — every path fact
    banked so far plus the guard we are trying to flip — with a strategy per parameter
    drawn from its live runtime type. There is no ordering to conflict with, because
    nothing is solved in sequence; one assignment either satisfies all of it or does not.
    """
    from tharkun.hypothesis_adapter import hypothesis_available

    if not hypothesis_available():
        return None, "hypothesis-facts"
    clauses = [*facts] + ([goal] if goal else [])
    if not clauses:
        return None, "hypothesis-facts"

    from tharkun.solve.condition import _free_names

    names: set[str] = set()
    for c in clauses:
        try:
            names |= _free_names(ast.parse(c, mode="eval").body)
        except SyntaxError:
            return None, "hypothesis-facts"
    names &= params & set(live)
    if not names:
        return None, "hypothesis-facts"

    strategies = {}
    for n in sorted(names):
        # Prefer the union of everything believed about this parameter; fall back to the
        # live value's own type when the store has nothing to say.
        candidates = _STORE.types_for(n) if _STORE is not None else []
        st_ = (_strategy_over_types(candidates, clauses) if candidates else None)
        if st_ is None:
            st_ = _strategy_for(live[n], clauses)
        if st_ is None:
            return None, "hypothesis-facts"      # an unmodelable parameter: abstain
        strategies[n] = st_

    predicate_src = " and ".join(clauses)
    try:
        code = compile(predicate_src, "<facts>", "eval")
    except SyntaxError:
        return None, "hypothesis-facts"
    # Names the clauses mention but we are not searching over (module constants, other
    # locals) keep their observed values, so the conjunction means what it meant.
    fixed = {k: v for k, v in live.items() if k not in strategies}

    def holds(assignment: dict[str, Any]) -> bool:
        try:
            return bool(eval(code, {"__builtins__": __builtins__}, {**fixed, **assignment}))
        except Exception:
            return False

    try:
        import hypothesis.strategies as st
        from hypothesis import HealthCheck, find, settings
        from tharkun.solve import seed as _seed

        got = find(st.fixed_dictionaries(strategies), holds,
                   settings=settings(max_examples=400, deadline=None, database=None,
                                     suppress_health_check=list(HealthCheck)),
                   random=_seed.hypothesis_random(f"driver:hypothesis-facts:{'|'.join(facts)}"))
    except Exception:
        return None, "hypothesis-facts"        # includes NoSuchExample
    return (dict(got) if got else None), "hypothesis-facts"


_HEAVY_FLIPPERS = [_flip_hypothesis]


def _flip_minleaf(
    source_path: str, branch_line: int, desired: str,
    real_snapshot: dict[str, Any], params: set[str],
) -> tuple[dict[str, Any] | None, str]:
    """Find the minimum-cost mutation of *live* state that flips the guard.

    This is the answer to the constructs an SMT engine cannot model — the measured
    majority of its abstentions.  Minleaf never translates the guard; it *evaluates* it
    on the real objects the tracer captured and searches for the smallest set of leaf
    values that flips it.  So an unmodeled container op, string method, or library call
    is a non-issue: the real Python runs.

    Its minimality objective also suits the caller — a patch close to the observed state
    re-encodes into a byte seed with the fewest perturbed fields.
    """
    from tharkun.minleaf.api import solve_minleaf_patch_for_guard

    roots = {k: v for k, v in real_snapshot.items() if k in params}
    if not roots:
        return None, "minleaf"
    # env must carry every root — the guard is *evaluated* against it — and for a
    # non-primitive root minleaf requires env and roots to be the same object (identity),
    # since it mutates leaves in place through that reference.
    # Try BOTH bytes policies. They expose different leaf sets — per_byte can flip a
    # single byte of a buffer, atomic can only replace the whole thing — so a guard one
    # cannot express the other often can. Whichever answers first wins; the driver
    # verifies either way.
    for policy in ("per_byte", "atomic"):
        # env must carry every root — the guard is *evaluated* against it — and for a
        # non-primitive root minleaf requires env and roots to reference the SAME live
        # objects, since it mutates leaves in place through them.
        env = dict(roots)
        fresh = roots
        try:
            result = solve_minleaf_patch_for_guard(
                source_path=os.path.realpath(source_path), target_line=branch_line,
                desired_branch=desired, env=env, roots=fresh, bytes_policy=policy,
            )
        except Exception:
            continue
        if result.success and result.patch:
            got = _apply_patch(fresh, result.patch.ops, params)
            if got:
                return got, f"minleaf:{policy}"
    return None, "minleaf"


def _apply_patch(roots: dict[str, Any], ops, params: set[str]) -> dict[str, Any] | None:
    """Materialise minleaf's patch into concrete inputs.

    A ``Patch`` *describes* the mutation rather than performing it, and each ``SetOp``
    carries a path from a root through attributes / keys to a leaf.  We deep-copy the
    roots first so the object the tracer captured is never corrupted, then write each
    leaf, following the path by attribute or subscript as the container allows.
    """
    import copy

    try:
        out = {k: copy.deepcopy(v) for k, v in roots.items()}
    except Exception:
        return None

    touched = False
    for op in ops:
        path = getattr(op, "path", ())
        if not path or not isinstance(path[0], str) or path[0] not in params:
            continue
        root = path[0]
        if len(path) == 1:
            out[root] = op.new
            touched = True
            continue
        # Walk to the parent of the leaf, remembering how we got there so an immutable
        # container can be rebuilt and written back into its own parent.
        try:
            trail: list[tuple[Any, Any]] = []      # (container, key) for each hop
            cur = out[root]
            for key in path[1:-1]:
                trail.append((cur, key))
                cur = getattr(cur, key) if isinstance(key, str) and hasattr(cur, key) else cur[key]
            last = path[-1]
            if isinstance(last, str) and hasattr(cur, last):
                setattr(cur, last, op.new)
            elif isinstance(cur, (bytes, bytearray, str, tuple)):
                # Immutable (or byte-typed) sequence: `cur[i] = v` raises, so build a new
                # one and store it back where cur came from. Without this a perfectly good
                # minleaf patch — e.g. SetOp(('data', 0), 0 -> 120) on a bytes parameter —
                # was silently dropped and the driver reported "no applicable input".
                rebuilt = _rebuild_sequence(cur, last, op.new)
                if rebuilt is None:
                    continue
                if trail:
                    parent, pkey = trail[-1]
                    if isinstance(pkey, str) and hasattr(parent, pkey):
                        setattr(parent, pkey, rebuilt)
                    else:
                        parent[pkey] = rebuilt
                else:
                    out[root] = rebuilt
            else:
                cur[last] = op.new
            touched = True
        except Exception:
            continue   # a path we cannot walk; other ops may still apply
    return out if touched else None


def _rebuild_sequence(seq: Any, index: Any, value: Any) -> Any | None:
    """A copy of *seq* with element *index* replaced — for immutable sequences."""
    if not isinstance(index, int):
        return None
    if isinstance(seq, (bytes, bytearray)):
        buf = bytearray(seq)
        if not 0 <= index < len(buf):
            return None
        try:
            buf[index] = int(value) & 0xFF
        except (TypeError, ValueError):
            return None
        return bytes(buf) if isinstance(seq, bytes) else buf
    if isinstance(seq, str):
        if not 0 <= index < len(seq):
            return None
        ch = value if isinstance(value, str) else chr(int(value))
        return seq[:index] + ch + seq[index + 1:]
    if isinstance(seq, tuple):
        if not 0 <= index < len(seq):
            return None
        lst = list(seq)
        lst[index] = value
        return tuple(lst)
    return None


_AVENGERS_CMP_OPS = {
    ast.Lt: "lt", ast.LtE: "le", ast.Gt: "gt", ast.GtE: "ge",
    ast.Eq: "eq", ast.NotEq: "ne",
}
_AVENGERS_FLIP_OP = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}
_AVENGERS_NEGATE_OP = {
    "lt": "ge", "le": "gt", "gt": "le", "ge": "lt", "eq": "ne", "ne": "eq",
    "truthy": "falsy", "falsy": "truthy", "none": "not_none", "not_none": "none",
}


def _avengers_field_bounds_from_guard(node: ast.AST, param_name: str) -> "tuple[Any, ...]":
    """Every ``<param_name>.<attr> <cmpop> <literal>`` / truthiness atom under *node*,
    as :class:`tharkun.avengers.FieldBound` -- this is guard analysis over THIS
    module's own AST (the program :func:`drive_to_line` is driving), not Avengers'
    concern, so it lives here rather than reaching into Avengers for it.

    Deliberately permissive about WHERE under the guard it looks (matches
    ``solve/guard_domain.py``'s own atom-extraction style: over-approximate, never
    under-approximate). A caller only ever uses this to narrow a SEARCH, so
    collecting a bound from a disjunct it does not strictly hold under just makes
    the search try an object it then discards -- it can never make a caller accept
    a wrong one, since every use here is followed by ``field_bounds_hold`` on the
    real constructed object and then, in the caller, independent re-execution
    (``_flip_verifies``).
    """
    from tharkun.avengers import FieldBound

    def is_attr(n: ast.AST) -> str | None:
        # ``param.attr``, and since [internal ref] a PATH of attributes and literal keys below it (``param.options.show_source``,
        # ``param.config['undefined'].value``): the text after the parameter's name, which ``FieldBound.attr`` reads as steps
        if not isinstance(n, (ast.Attribute, ast.Subscript)):
            return None
        inner = n
        while isinstance(inner, (ast.Attribute, ast.Subscript)):
            if isinstance(inner, ast.Subscript) and not (
                    isinstance(inner.slice, ast.Constant) and isinstance(inner.slice.value, (str, int))
                    and not isinstance(inner.slice.value, bool)):
                return None
            inner = inner.value
        if not (isinstance(inner, ast.Name) and inner.id == param_name):
            return None
        text = ast.unparse(n)[len(param_name):]
        return text[1:] if text.startswith(".") else text

    out: list[Any] = []
    # the WHOLE guard is the bare attribute (``if c.items:``) -- not reached by
    # the BoolOp case below, since there is no ``and``/``or`` to walk into.
    top_attr = is_attr(node)
    if top_attr is not None:
        out.append(FieldBound(top_attr, "truthy"))
    for n in ast.walk(node):
        if isinstance(n, ast.Compare) and len(n.ops) == 1 and len(n.comparators) == 1:
            if isinstance(n.ops[0], (ast.Is, ast.IsNot)):
                # ``p.attr is None`` / ``is not None`` ([internal ref]; the vocabulary's ``none`` / ``not_none``)
                attr = is_attr(n.left)
                other = n.comparators[0]
                if attr is not None and isinstance(other, ast.Constant) and other.value is None:
                    out.append(FieldBound(attr, "none" if isinstance(n.ops[0], ast.Is) else "not_none"))
                continue
            op = _AVENGERS_CMP_OPS.get(type(n.ops[0]))
            if op is None:
                continue
            attr = is_attr(n.left)
            other = n.comparators[0]
            if attr is None:
                attr = is_attr(other)
                if attr is not None:
                    op = _AVENGERS_FLIP_OP[op]
                    other = n.left
            if attr is not None and isinstance(other, ast.Constant):
                out.append(FieldBound(attr, op, other.value))
        elif isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            attr = is_attr(n.operand)
            if attr is not None:
                out.append(FieldBound(attr, "falsy"))
        elif isinstance(n, ast.BoolOp):
            for v in n.values:
                attr = is_attr(v)
                if attr is not None:
                    out.append(FieldBound(attr, "truthy"))
    return tuple(out)


def _avengers_negate_bounds(bounds: "tuple[Any, ...]") -> "tuple[Any, ...]":
    """Negate each bound independently (the ``else`` side of the guard).

    This is a SOUND under-approximation of the true negation of a conjunction
    (De Morgan's needs a disjunction), not the exact one: any object satisfying
    every independently-negated bound also satisfies the guard's real negation,
    so this never widens what a caller may accept -- it can only narrow the
    search, same as an unmet bound above.
    """
    from tharkun.avengers import FieldBound

    return tuple(FieldBound(b.attr, _AVENGERS_NEGATE_OP[b.op], b.value) for b in bounds)


def _avengers_classes_with_attrs(tree: ast.Module, wanted: set[str]) -> list[str]:
    """Every class in *tree* whose body assigns ``self.<attr>`` for every name in
    *wanted*, at least once anywhere in the class (a cheap, sound-for-search
    over-approximation -- a class that assigns the attribute only on some paths is
    still worth trying; the replay-and-check below is what actually decides)."""
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        assigned: set[str] = set()
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Attribute) and isinstance(inner.ctx, ast.Store)
                    and isinstance(inner.value, ast.Name) and inner.value.id == "self"):
                assigned.add(inner.attr)
        if wanted <= assigned:
            out.append(node.name)
    return out


def _avengers_object_for_bounds(
    source_path: str, class_name: str, bounds: "tuple[Any, ...]", *, budget_seconds: float, confine=None,
) -> Any | None:
    """Ask :mod:`tharkun.avengers` for a live object of *class_name* satisfying
    *bounds*, using its public API directly (``ObjectGoal``, ``SaturationBudget``,
    ``synthesize_object``, ``field_bounds_hold``, ``materialize_recipe`` -- no
    intermediate service module; [internal ref] slice 3, owner correction 2026-09-29).

    Tries validated recipes, simplest first, until one's real constructed object
    satisfies the bounds; a class that only ever constructs states outside the
    goal correctly returns None rather than a best-effort guess -- nothing here is
    a proposal the caller should trust without its own verification
    (``_flip_verifies`` re-executes the real program either way).
    """
    import time as _time

    from tharkun.avengers import (
        IsolatedExecutor, ObjectGoal, SaturationBudget, materialize_recipe, synthesize_object,
    )
    from tharkun.solve.loader import load_module_for_execution

    from contextlib import nullcontext

    if confine is not None:
        from tharkun.solve.extern import native_import_reason
        with open(source_path, "rb") as stream:
            if native_import_reason(stream.read(), source_path, reversible=True):
                return None
    with (confine or nullcontext)() as environment:
        module, mglobals = load_module_for_execution(source_path)
        if module is None or class_name not in mglobals:
            return None
        if environment is not None and (environment.flags - {"file_effect"}):
            return None
        files = environment.files if environment is not None else None
    t0 = _time.monotonic()
    per_attempt = max(1.0, min(budget_seconds, 3.0))
    while _time.monotonic() - t0 < budget_seconds:
        try:
            goal = ObjectGoal(source_path, class_name, field_bounds=bounds)
            report = synthesize_object(goal, budget=SaturationBudget(max_seconds=per_attempt),
                                       **({"executor": IsolatedExecutor(files=files)} if files is not None else {}))
        except Exception:
            break
        # the child that replayed each recipe already checked the bounds on the object it built (``ReplayOutcome.bounds_hold``,
        # ``avengers/_worker.py::_run``, C5); only a recipe it confirmed is materialized here, to hand the caller its object
        candidates = sorted(
            (item for item in report.validated if item.outcome.bounds_hold is True),
            key=lambda item: (
                len(item.recipe.setup),
                len(item.recipe.constructor.args) + len(item.recipe.constructor.kwargs),
                item.recipe.key,
            ),
        )[:8]
        for validated in candidates:
            with (confine or nullcontext)() as environment:
                try:
                    obj = materialize_recipe(validated.recipe, class_name, mglobals)
                except Exception:
                    obj = None
                    continue
                if environment is not None and (environment.flags - {"file_effect"}):
                    obj = None
                    return None
                if obj is not None:
                    return obj
                obj = None
    return None


def _avengers_present() -> bool:
    """Whether :mod:`tharkun.avengers` can be imported.

    Avengers is an optional component of the drive portfolio (owner ruling 2026-11-05: the SV-COMP
    development branch ships without it, and adding it back is a revert).  This is the one place the
    portfolio asks: a tier that needs it is skipped, never an error, so ``drive_to_line`` goes on to the
    next tier (the Hypothesis search) exactly as it does when Avengers finds nothing.  Not cached: an
    import of a package that is absent costs a path scan per blocked guard, which is nothing next to the
    guard's own budget.
    """
    try:
        # the names the tier uses, not just the package: a directory left behind by a deleted package (a ``__pycache__``) imports as an
        # empty namespace package, which is "absent" here
        from tharkun.avengers import (  # noqa: F401
            FieldBound, IsolatedExecutor, ObjectGoal, SaturationBudget, materialize_recipe, synthesize_object,
        )
    except ImportError:
        return False
    return True


def _flip_avengers(
    source_path: str, branch_line: int, desired: str,
    real_snapshot: dict[str, Any], params: set[str],
    *, budget_seconds: float = 10.0, confine=None,
) -> tuple[dict[str, Any] | None, str]:
    """Ask Avengers to CONSTRUCT an object satisfying the guard, for a
    parameter minleaf has nothing to mutate because no live value was ever
    observed.

    [internal ref] slice 3 (owner ruling, [internal ref]): Avengers as an on-demand
    service, called only for the one parameter and only the field bounds
    THIS guard states -- never an unconditional whole-file sweep the way
    ``sv/avengers_seed.py`` runs it. This is the fallback for exactly the
    gap ``docs/DESIGN.md`` names for minleaf: "when a parameter has never
    been observed with a meaningful value, it has nothing to mutate."

    Runs unconditionally, on demand only, the same way every other flipper
    in the portfolio does (no boolean gate) -- fires only for the one
    parameter and only the field bounds THIS guard states, never an
    unconditional whole-file sweep, and only once minleaf has nothing to
    mutate. CPU is bounded per call by ``budget_seconds`` (the driver's own
    ``avengers_budget_seconds``), spent from the blocked guard's own share
    of the round budget -- not a separate whole-task allowance.

    Calls :mod:`tharkun.avengers`'s existing public API directly (owner
    correction 2026-09-29: Avengers is already Tharkun's generic
    object-construction line; this is one caller of it, not a second layer on
    top of it) -- the guard-to-goal translation above is this module's own
    concern (parsing THIS driver's guard AST), not Avengers'.
    """
    if not _avengers_present():
        return None, "avengers"
    try:
        tree = ast.parse(Path(source_path).read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return None, "avengers"
    node = _find_branch_node(tree, branch_line)
    if node is None:
        return None, "avengers"
    guard_vars = _free_names_of(node)
    for pname in sorted(params & guard_vars):
        if real_snapshot.get(pname) is not None:
            continue  # minleaf already has something to mutate
        bounds = _avengers_field_bounds_from_guard(node, pname)
        if not bounds:
            continue
        if desired == "else":
            bounds = _avengers_negate_bounds(bounds)
        wanted_attrs = {b.attr for b in bounds}
        for class_name in _avengers_classes_with_attrs(tree, wanted_attrs):
            obj = _avengers_object_for_bounds(
                source_path, class_name, bounds, budget_seconds=budget_seconds,
                **({"confine": confine} if confine is not None else {}))
            if obj is not None:
                return {pname: obj}, f"avengers:{class_name}"
    return None, "avengers"


def _flip_verifies(
    source_path: str, func_name: str, blocker: GuardFrame,
    inputs: dict[str, Any], candidate: dict[str, Any],
    guard_path: str | None = None,
) -> bool:
    """True if applying *candidate* actually makes the blocker's target side execute.

    The tracer is the judge: a solver's answer is only a proposal until execution
    confirms it flips the guard.  This lets the portfolio discard a wrong candidate
    (including one behind a false 'infeasible' vote) and try the next.

    ``guard_path`` is the file the guard lives in, when that is not the entry function's
    — line numbers are per-file, so checking the blocker's span against the wrong file's
    executed lines would confirm a flip that never happened.
    """
    merged = {**inputs, **candidate}
    gp = os.path.realpath(guard_path) if guard_path else os.path.realpath(source_path)
    side = blocker.then_span if blocker.desired_branch == "then" else blocker.else_span
    if side is None:
        return False
    # Entering the target side executes its first body line, so the fast localized probe
    # (instruments only that function's code object, heavy callees run uninstrumented —
    # ~1000x faster than the whole-program trace on a DES-style callee) answers this. It
    # returns None when it cannot localize the line, so the full trace is the fallback.
    probed = _target_reached_probe(source_path, func_name, merged, side[0], gp,
                                   frozenset({gp}))
    if probed is not None:
        return probed
    outcome = _run_traced(source_path, func_name, merged, set(),
                          extra_paths=frozenset({gp}))
    if not outcome.ok:
        return False
    executed = (outcome.by_path.get(gp, set())
                if gp != os.path.realpath(source_path) else outcome.executed)
    return any(side[0] <= ln <= side[1] for ln in executed)


# ---------------------------------------------------------------------------
# The driver loop
# ---------------------------------------------------------------------------


def _guard_source(tree: ast.Module, branch_line: int) -> str | None:
    from tharkun.solve.condition import _find_branch_at_line

    node = _find_branch_at_line(tree, branch_line)
    return ast.unparse(node) if node is not None else None


def _prefix_clauses(
    tree: ast.Module, frames: list[GuardFrame], blocker: GuardFrame
) -> list[tuple[str, int, Any, str]]:
    """The lexically-enclosing guards, as ``(text, origin_line, node, polarity)`` folds.

    These must hold in their desired polarity while the blocker is solved — without them
    an independent per-guard solve oscillates (fixing an inner guard by breaking an outer
    one). Emitted with their program point so the flow-sensitivity filter can drop one
    that a reassignment on the path has since invalidated, rather than carrying it stale.
    """
    out: list[tuple[str, int, Any, str]] = []
    for f in frames:
        if f.depth >= blocker.depth:
            break
        node = _find_branch_node(tree, f.branch_line)
        if node is None:
            continue
        src = ast.unparse(node)
        text = f"({src})" if f.desired_branch == "then" else f"(not ({src}))"
        out.append((text, f.branch_line, node, f.desired_branch))
    return out


# --- flow sensitivity: which clauses a reassignment on the path has invalidated --------


def _target_names(node) -> set[str]:
    """The names a binding target rebinds. A subscript/attribute target kills its BASE
    (``a[0] = x`` and ``a.b = x`` both mutate ``a``)."""
    out: set[str] = set()
    if isinstance(node, ast.Name):
        out.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for e in node.elts:
            out |= _target_names(e)
    elif isinstance(node, ast.Starred):
        out |= _target_names(node.value)
    elif isinstance(node, (ast.Attribute, ast.Subscript)):
        base = node
        while isinstance(base, (ast.Attribute, ast.Subscript)):
            base = base.value
        if isinstance(base, ast.Name):
            out.add(base.id)
    return out


def _binding_index(tree: ast.Module) -> tuple[dict[int, frozenset[str]], list[tuple[int, int]],
                                              dict[int, frozenset[str]]]:
    """Every binding site in a file: ``(assigned_by_line, loop_spans, self_rhs_by_line)``.

    ``assigned_by_line`` maps a line to the names bound there; ``loop_spans`` is the
    ``(lo, hi)`` line range of every loop (the flow filter widens its window to a loop's
    span so a back-edge reassignment is not missed); ``self_rhs_by_line`` marks, for a
    plain ``v = rhs`` assignment, the names whose RHS *mentions the target itself*
    (``a = a + 1`` — not a severing assignment).

    Scope matters: a binding inside a NESTED function/lambda/class, or a comprehension's
    own iteration target, is a DIFFERENT variable than a same-named parameter of the
    enclosing function — recording it would wrongly evict facts (or falsely mark a
    parameter severed) whenever the nested line happens to fall in the flow window. So the
    walk records bindings only at the module level and directly inside a top-level
    function, and comprehension targets are skipped entirely (a walrus inside a
    comprehension, which DOES bind outward, is still recorded via ``NamedExpr``).
    """
    from tharkun.solve.condition import _free_names

    by_line: dict[int, set[str]] = {}
    loops: list[tuple[int, int]] = []
    self_rhs: dict[int, set[str]] = {}

    def add(line: int, names: set[str]) -> None:
        if names:
            by_line.setdefault(line, set()).update(names)

    def record(n) -> None:
        if isinstance(n, ast.Assign):
            names: set[str] = set()
            for t in n.targets:
                names |= _target_names(t)
            add(n.lineno, names)
            rhs = _free_names(n.value) if n.value is not None else set()
            keep = {nm for nm in names if nm in rhs}
            if keep:
                self_rhs.setdefault(n.lineno, set()).update(keep)
        elif isinstance(n, ast.AugAssign):
            names = _target_names(n.target)
            add(n.lineno, names)
            self_rhs.setdefault(n.lineno, set()).update(names)   # a += 1 reads a
        elif isinstance(n, ast.AnnAssign):
            if n.value is not None:
                add(n.lineno, _target_names(n.target))
        elif isinstance(n, ast.NamedExpr):
            add(n.target.lineno, _target_names(n.target))
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            add(n.target.lineno, _target_names(n.target))
            loops.append((n.lineno, getattr(n, "end_lineno", n.lineno) or n.lineno))
        elif isinstance(n, ast.While):
            loops.append((n.lineno, getattr(n, "end_lineno", n.lineno) or n.lineno))
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for item in n.items:
                if item.optional_vars is not None:
                    add(item.optional_vars.lineno, _target_names(item.optional_vars))
        elif isinstance(n, ast.Delete):
            names = set()
            for t in n.targets:
                names |= _target_names(t)
            add(n.lineno, names)
        elif isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) \
                and isinstance(n.value.func, ast.Attribute):
            # A bare ``a.append(b)`` / ``a.sort()`` mutates its receiver in place, which no
            # rebinding captures. Conservatively kill the receiver's base name: a clause
            # about ``a`` (its length, an element) may no longer hold. Over-kill — a pure
            # ``a.startswith(x)`` written as a statement also kills — but statements whose
            # result is discarded are almost always mutations, and dropping a fact only
            # loses precision, never soundness.
            add(n.lineno, _target_names(n.value.func.value))
        elif isinstance(n, ast.ExceptHandler) and n.name:
            add(n.lineno, {n.name})
        elif isinstance(n, getattr(ast, "MatchAs", ())) and getattr(n, "name", None):
            add(n.lineno, {n.name})

    # Record every binding site in the file. Ranging over the whole tree is deliberately
    # over-inclusive — a nested function's local, a nonlocal/global write, a method body's
    # assignment — because DROPPING a clause is sound (loses precision, never invents), and
    # a scoped walk got this exactly backwards: skipping deeper scopes made the reassignment
    # inside a *class method* (a def under a class) invisible, which revived the Witness B
    # false-dead one scope down. The one thing genuinely in a different scope that never
    # leaks is a comprehension's own iteration target, and that is handled by NOT giving
    # ast.comprehension a branch above (a walrus inside a comprehension, which does bind
    # outward, still registers via NamedExpr). Line numbers keep bindings in other functions
    # from colliding: the flow window is bounded by the blocker's own guards' lines.
    for n in ast.walk(tree):
        record(n)

    return ({ln: frozenset(v) for ln, v in by_line.items()}, loops,
            {ln: frozenset(v) for ln, v in self_rhs.items()})


def _killed_between(by_line: dict[int, frozenset[str]], loops: list[tuple[int, int]],
                    executed: set[int], line_a: int, line_b: int, names: set[str]) -> bool:
    """Was any of *names* reassigned by an EXECUTED statement between two program points?

    The window is the open-low, closed-high interval ``(min, max]`` of the two lines,
    widened to a whole loop's span when both endpoints sit inside one — a loop's back edge
    reruns the body, so a textual window would miss a reassignment that executes between
    two dynamic visits of the same line. Only executed lines count: an assignment on a
    branch the run did not take did not happen.
    """
    if not names:
        return False
    lo, hi = (line_a, line_b) if line_a <= line_b else (line_b, line_a)
    # A loop containing EITHER endpoint extends the window to that loop's whole body: the
    # loop re-enters, so an executed assignment anywhere in it happens between two visits of
    # a line inside it. Unioning every such loop (using the ORIGINAL endpoints for
    # membership) handles nested loops and the establish-outside / blocker-inside shape that
    # a single "both endpoints inside one loop" check silently misses.
    for ls, le in loops:
        if (ls <= line_a <= le) or (ls <= line_b <= le):
            lo, hi = min(lo, ls), max(hi, le)
    for la, bound in by_line.items():
        if la in executed and lo < la <= hi and (bound & names):
            return True
    return False


def _clause_is_stale(clause, blocker_line: int, blocker_path: str, executed: set[int],
                     by_line: dict[int, frozenset[str]], loops: list[tuple[int, int]]) -> bool:
    """True when a reassignment on the path has invalidated *clause* for the blocker.

    Only same-file clauses are decided: a clause whose origin is in another file cannot be
    line-compared against this file's assignments, so it is kept (indeterminate — the old,
    flow-insensitive behaviour, which is sound here because cross-file reassignment is not
    the bug being fixed)."""
    o = clause.origin
    if not o.path or os.path.realpath(o.path) != os.path.realpath(blocker_path):
        return False
    return _killed_between(by_line, loops, executed, o.lineno, blocker_line, set(clause.names))


def _type_preserving(seed: dict[str, Any], got: dict[str, Any]) -> dict[str, Any] | None:
    """*got* with any str/bytes value converted back to the type the program is running.

    A guard like ``len(p) > 4`` is happy with either, so a solver may answer the str
    ``'xxxxx'`` for a parameter the program has been passing as ``bytes``. That silently
    changes the type every LATER guard is solved against — ``p[0] == 0x7F`` yields an int
    for bytes and a str for str, so it becomes unsatisfiable and the loop blames the
    deeper guard for the earlier one's choice.

    Returns None when nothing needed converting. The variant is a *proposal* like any
    other: it only wins if execution confirms it flips the guard.
    """
    out = dict(got)
    changed = False
    for k, v in got.items():
        want = type(seed.get(k))
        if want is type(v) or k not in seed:
            continue
        try:
            if want is bytes and isinstance(v, str):
                out[k] = v.encode("latin-1")
            elif want is str and isinstance(v, (bytes, bytearray)):
                out[k] = bytes(v).decode("latin-1")
            else:
                continue
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
        changed = True
    return out if changed else None


#: How many raw correspondences to gather before a search stops looking. Only a few are
#: ever banked, and each harvest re-parses the guard's file.
_HARVEST_ENOUGH = 12


def _harvest_operands(guard_path: str, guard_line: int, outcome,
                      params: set[str], trial: dict[str, Any]) -> list:
    """What the blocking comparison demanded during ONE trial, un-banked.

    Returns RedQueen correspondences rather than writing them, so the caller can keep the
    best few across a whole search instead of banking every trial's.
    """
    from tharkun.redqueen import harvest

    gp = os.path.realpath(guard_path)
    snaps = (outcome.real_by_path.get(gp) or outcome.real or {}).get(guard_line) or []
    if not snaps:
        return []
    try:
        return harvest(guard_path, guard_line, {**trial, **snaps[-1]}, params)
    except Exception:
        return []


def _bank_operands(store, corrs, *, keep: int = 4) -> None:
    """Record the best correspondences a search observed — ONCE, not per trial.

    A search is not all-or-nothing: it runs the program hundreds of times, and each run
    observes the value the guard actually wanted. Banking those turns a failed search into
    an information source the next round can use.

    But it must be banked once, at the end. Called per trial it produced **37 facts from a
    single search** on a two-guard toy, differing only in the incidental filler bytes each
    trial happened to carry — ``b'6\xa3'``, ``b'6\xa3+\xdf\xbd'``, ``b'6\xa3\x00A'``.
    Worse than the volume: ``FactStore.add`` bumps the epoch on every new fact and
    ``is_exhausted`` is epoch-gated, so a chatty search re-opens every written-off frontier
    on every round — defeating the skip that took a toy from 26.7s to 7.7s.

    Deduplicated on ``(input_name, proposal)`` and capped, highest confidence first.
    """
    from tharkun.facts import Fact

    seen: set = set()
    for c in sorted(corrs, key=lambda c: -c.confidence):
        key = (c.input_name, repr(c.proposal))
        if key in seen:
            continue
        seen.add(key)
        # Bank the PROPOSAL, not the demanded value. RedQueen has already done the hard
        # part: `demanded` is the raw number the comparison wanted (13987), while
        # `proposal` is that number inverted back through the transform into something the
        # input can actually hold (b"6\xa3"). Storing the number left the next search to
        # rediscover the encoding by chance — measured, it did not.
        store.add(Fact("operand", c.input_name, c.proposal, source="reach-search",
                       confidence=c.confidence))
        # The raw comparison source is deliberately NOT banked alongside it: an offline
        # inversion needs the live env (RedQueen splices the demanded value into the live
        # buffer), so a banked shape string has no possible consumer — the fully-inverted
        # proposal above is the durable artifact. If a structural need ever appears, bank
        # the resolved (field, byteorder, width) decode spec, and only with its consumer.
        if len(seen) >= keep:
            return


def _reach_by_search(
    source_path: str, func_name: str, target_line: int, target_path: str,
    inputs: dict[str, Any], params: set[str], hints: list[Any],
    *, max_examples: int = 300, store=None,
    guard_path: str = "", guard_line: int = 0,
) -> dict[str, Any] | None:
    """Search the ENTRY parameters for an assignment that executes *target_line*.

    The last resort, and the one that does not need the dataflow inverted. A guard deep in
    a callee is stated in terms of that callee's locals — `block_length != 8 and pad`,
    where block_length is len(data[i:i+8]) and data is a slice of a caller argument. Every
    flipper works on entry parameters, so none can say "make block_length 3", and mapping
    the local back to the parameter needs call_mapper.

    Banked operands and static candidates tell us what the guard wants and aim
    a search: banked candidates supply the interesting values, the live types
    supply the shapes, and **execution decides**. The predicate is not the guard — which
    we cannot even evaluate out here — but "did the target line run", which is the only
    thing we actually care about and is never a heuristic.

    Length is searched as well as value, because for this class of guard the lever is
    usually how LONG a parameter is, not what is in it.
    """
    from tharkun.hypothesis_adapter import hypothesis_available

    if not hypothesis_available():
        return None
    searchable = {p for p in params if p in inputs and _strategy_for(inputs[p]) is not None}
    if not searchable:
        return None

    clauses = [repr(h) for h in hints]
    # What the KnowledgeBase already believes at this guard — the warmup's static types and
    # anything a prior run observed here, joined. When it types a parameter, that TYPE aims
    # the strategy better than the live value's shape can: a param seeded as b'' that the
    # guard needs to be a str gets a text strategy, not a bytes one. Falls back per-parameter
    # to the value-shape strategy whenever the state says nothing usable.
    prior = None
    if store is not None and guard_line and hasattr(store, "at"):
        try:
            from tharkun.knowledge import program_point

            prior = store.at(program_point(guard_path or source_path, guard_line))
        except Exception:
            prior = None

    def _strategy(p):
        # A prior/observed TYPE from the KB overrides the seed value's shape (a param seeded
        # b'' the program runs as str), but the hint alphabet is kept — a type alone gives a
        # needle-in-2^32 search, the demanded literal is what actually lands `magic == C`.
        # A prior MEMBERSHIP (values the analysis named) is folded into that alphabet too, so
        # both the KB's own candidates and the round's hints seed the pool.
        if prior is not None:
            from tharkun.strategies import constraints_from_state

            c = constraints_from_state(prior, p)
            if c is not None:
                extra = [repr(v) for v in c.get("values", [])]
                s = _strategy_for(inputs[p], clauses + extra, type_override=c["type"])
                if s is not None:
                    return s
        return _strategy_for(inputs[p], clauses)

    strategies = {p: _strategy(p) for p in sorted(searchable)}
    tp = os.path.realpath(target_path)
    # Snapshot the blocking guard so each trial's operands can be banked.
    harvested: list = []                 # collected per trial, banked once at the end
    watch_lines: set[int] = set()
    extra_watch: dict[str, set[int]] = {}
    if store is not None and guard_line:
        gp = os.path.realpath(guard_path or source_path)
        if gp == os.path.realpath(source_path):
            watch_lines = {guard_line}
        else:
            watch_lines = {guard_line}
            extra_watch = {gp: {guard_line}}

    def reaches(assignment: dict[str, Any]) -> bool:
        trial = {**inputs, **assignment}
        # Two regimes. While there is still operand harvesting to do, a trial needs the
        # FULL trace (snapshots at the guard line are RedQueen's raw material). Once enough
        # correspondences are banked, the predicate only needs "did the target line run" —
        # so take the fast localized probe, which instruments ONLY the target's own code
        # object and lets the heavy callees (e.g. 3x DES per call) run uninstrumented:
        # measured ~4x faster than whole-program settrace, and the gap grows with callee
        # weight. The probe returns None when it cannot localize the line, so the full
        # trace is always the correctness fallback.
        harvesting = store is not None and watch_lines and len(harvested) < _HARVEST_ENOUGH
        if not harvesting:
            probed = _target_reached_probe(source_path, func_name, trial, target_line, tp,
                                           frozenset({tp}))
            if probed is not None:
                return probed
        try:
            oc = _run_traced(source_path, func_name, trial, watch_lines,
                             capture_real=bool(watch_lines),
                             extra_paths=frozenset({tp}),
                             extra_snapshot_lines=extra_watch)
        except Exception:
            return False
        if not oc.ok:
            return False
        # Every trial EXECUTES the program, and every execution passes through the
        # blocking comparison. Those operands are exactly RedQueen's raw material, so a
        # search that fails still has to leave behind what it saw — otherwise a failed
        # search teaches nothing, the epoch never moves, and the frontier is written off
        # on the strength of one unlucky draw.
        # Stop harvesting once there is plenty to choose from. Only a handful of
        # correspondences are ever banked, and `harvest` re-parses the guard's file on
        # every call — running it for all 300 trials to keep 4 results cost 6.7s on a
        # two-guard toy (7.7s -> 14.4s) for observations that are near-duplicates anyway.
        if harvesting:
            harvested.extend(_harvest_operands(guard_path, guard_line, oc, params, trial))
        executed = (oc.by_path.get(tp, set())
                    if tp != os.path.realpath(source_path) else oc.executed)
        return target_line in executed

    try:
        import hypothesis.strategies as st
        from hypothesis import HealthCheck, find, settings
        from tharkun.solve import seed as _seed

        got = find(st.fixed_dictionaries(strategies), reaches,
                   settings=settings(max_examples=max_examples, deadline=None, database=None,
                                     suppress_health_check=list(HealthCheck)),
                   random=_seed.hypothesis_random(f"driver:reach-by-search:{func_name}:{target_path}:{target_line}"))
    except Exception:
        return None                      # includes NoSuchExample
    finally:
        # Flush on BOTH paths — a failed search is precisely the one whose observations
        # are worth keeping, since it leaves nothing else behind.
        if store is not None and harvested:
            _bank_operands(store, harvested)
    return dict(got) if got else None


def _guard_fact(tree: ast.Module, branch_line: int, desired: str) -> str | None:
    """The guard's condition as a requirement string for *desired*, or None."""
    node = _find_branch_node(tree, branch_line)
    if node is None:
        return None
    try:
        src = ast.unparse(node)
    except Exception:
        return None
    return f"({src})" if desired == "then" else f"(not ({src}))"


def _satisfied_clauses(tree: ast.Module, executed: set[int], blocker_line: int,
                       blocker_vars: set[str]) -> list[tuple[str, int, Any]]:
    """Guards already HOLDING on the executed path, as ``(text, origin_line, node)`` folds.

    ``_prefix_clauses`` only knows lexically-enclosing guards, so when the target sits
    below a call it does not preserve the guards in the CALLERS that got us there. The
    solver is then free to return a value that satisfies the local guard while breaking
    ``a == 100`` two frames up, the callee is never invoked again, and the flip fails
    verification for a reason the solver never saw.

    Any guard that ran AND entered its body is currently true, so emit it — with its
    program point, so the flow-sensitivity filter can drop one a later reassignment has
    invalidated. A guard is emitted only when EVERY free name it uses (builtins included)
    is one the blocker's probe also names: this walk ranges over the whole file, so a
    helper function's ``len(x) > 9`` on an unrelated ``x`` of the same name would otherwise
    be admitted by the render's builtin-blind name check and poison the precondition into
    a false infeasible.
    """
    from tharkun.solve.condition import _free_names

    out: list[tuple[str, int, Any]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or node.lineno not in executed:
            continue
        if node.lineno == blocker_line:
            continue
        span = _span(node.body)
        if span is None or not any(span[0] <= ln <= span[1] for ln in executed):
            continue                      # its body did not run: not currently true
        if span[0] == node.lineno:
            # A one-line body (`if g: stmt`) shares the guard's line, so "the body ran"
            # cannot be told from "the guard was evaluated". Do not claim it satisfied —
            # omitting a maybe-true clause is sound (fewer musts); asserting a false one is
            # not, and would over-constrain the solve into a false infeasible.
            continue
        if not (span[0] <= blocker_line <= span[1]):
            # A SIBLING guard that ran and entered its body on THIS trace is not a must for
            # reaching the blocker — a different input can skip it and still reach the
            # blocker. Only a guard whose body ENCLOSES the blocker line genuinely has to
            # hold; asserting a sibling over-constrains the solve into a false infeasible
            # (a certified-dead verdict on a reachable line).
            continue
        if not (_free_names(node.test) <= blocker_vars):
            continue                      # names a variable (or builtin) the probe lacks
        out.append((f"({ast.unparse(node.test)})", node.lineno, node.test))
    return out


def _prefix_vars(tree: ast.Module, frames: list[GuardFrame], blocker: GuardFrame) -> set[str]:
    """Free variables the outer (already-satisfied) guards depend on."""
    from tharkun.solve.condition import _free_names

    out: set[str] = set()
    for f in frames:
        if f.depth >= blocker.depth:
            break
        node = _find_branch_node(tree, f.branch_line)
        if node is not None:
            out |= _free_names(node)
    return out


def _find_branch_node(tree: ast.Module, branch_line: int):
    from tharkun.solve.condition import _find_branch_at_line

    return _find_branch_at_line(tree, branch_line)


def _free_names_of(node) -> set[str]:
    from tharkun.solve.condition import _free_names

    return _free_names(node) if node is not None else set()


def _text_names(text: str) -> set[str]:
    """Free names in a constraint string, or empty on a parse failure."""
    from tharkun.solve.condition import _free_names

    try:
        return _free_names(ast.parse(text, mode="eval").body)
    except SyntaxError:
        return set()


def _func_params(tree: ast.Module, func_name: str) -> set[str]:
    node = _def_node(tree, func_name) if "." in func_name else next(
        (n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func_name),
        None)
    if node is None:
        return set()
    a = node.args
    return {n for n in (p.arg for p in (a.posonlyargs + a.args + a.kwonlyargs)) if n not in ("self", "cls")}


def _enclosing_func_span(tree: ast.Module, line: int) -> tuple[int, int] | None:
    """The line range of the innermost function enclosing *line*, or None.

    Used to clamp entry-origin facts (context pins, bounds) to the function's start, so
    the flow filter's window covers every reassignment between entry and the blocker.
    """
    best: tuple[int, int] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lo, hi = node.lineno, getattr(node, "end_lineno", node.lineno) or node.lineno
            if lo <= line <= hi and (best is None or lo > best[0]):
                best = (lo, hi)
    return best


def _bounds_requires(bounds: dict[str, tuple[int, int]], blocker_vars: set[str]) -> str:
    """Range clauses ``lo <= v and v <= hi`` for bounded blocker vars.

    Feeds a field's actual value range (e.g. a ConsumeUInt(4) field is [0, 2**32-1])
    to the solver so it returns an *encodable* solution rather than an unbounded
    integer that cannot be written back to the fixed-width field.
    """
    return " and ".join(_bounds_clause_texts(bounds, blocker_vars))


def _bounds_clause_texts(bounds: dict[str, tuple[int, int]], blocker_vars: set[str]) -> list[str]:
    """One ``(lo <= v and v <= hi)`` clause text per bounded blocker var."""
    out = []
    for v in sorted(blocker_vars):
        if v in bounds:
            lo, hi = bounds[v]
            out.append(f"({lo} <= {v} and {v} <= {hi})")
    return out


def _call_chain_blockers(tree: ast.Module, executed: set[int], target_line: int) -> list[GuardFrame]:
    """Guards that RAN but whose body did not — candidate blockers anywhere in the file.

    ``enclosing_guards`` is lexical, so it only ever sees guards in the same function as
    the target.  When the target sits one or more CALLS below the entry point, the guard
    actually stopping us lives in a caller and is invisible to it — the driver then flips
    the right-looking guard in the callee, the callee is still never invoked, and the run
    plateaus with the seed untouched.

    This is the dynamic complement: whatever executed but did not enter its body is a
    live candidate, regardless of which function it is in.  Ordered by distance to the
    target so the nearest plausible blocker is tried first; correctness comes from the
    caller verifying that a flip actually makes progress.
    """
    out: list[GuardFrame] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or node.lineno not in executed:
            continue
        then_span = _span(node.body)
        else_span = _span(node.orelse)
        for desired, span in (("then", then_span), ("else", else_span)):
            if span is None:
                continue
            if any(span[0] <= ln <= span[1] for ln in executed):
                continue                      # that side already ran
            out.append(GuardFrame(branch_line=node.lineno, desired_branch=desired,
                                  then_span=then_span or (node.lineno, node.lineno),
                                  else_span=else_span, depth=0))
    out.sort(key=lambda f: abs(f.branch_line - target_line))
    return out


def _progress(executed: set[int], target_line: int) -> int:
    """How much of the path toward *target_line* has run — a hill-climbing score.

    Counting executed lines at or before the target is crude but monotone in the right
    direction: entering a callee that leads to the target strictly increases it, so a
    flip that opens the next layer is recognisable without any call-graph analysis.
    """
    return sum(1 for ln in executed if ln <= target_line)


def _func_name_containing(tree: ast.Module, line: int) -> str | None:
    """Name of the innermost function whose body contains *line*, or None."""
    best: tuple[int, str] | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lo, hi = node.lineno, getattr(node, "end_lineno", node.lineno) or node.lineno
            if lo <= line <= hi and (best is None or lo > best[0]):
                best = (lo, node.name)
    return best[1] if best else None


def _backward_propagate(
    source_path: str, entry_func: str, blocker_tree: ast.Module, blocker: GuardFrame,
    blocker_path: str, params: set[str], desired: str,
) -> dict[str, Any] | None:
    """Solve a guard on a callee PARAMETER by propagating it to the entry parameter.

    When the blocker is a guard on a callee's own parameter, the portfolio can solve the
    guard for that parameter but the driver discards the answer because the name is not an
    entry parameter. If ``call_mapper`` shows the callee is invoked with that parameter
    bound to a SIMPLE entry-parameter name (``validate(payload)`` — a pass-through, not a
    slice or a transform), the solved value applies to the entry parameter directly. This
    is the tractable half of the backward direction; a non-simple argument (a slice,
    ``int.from_bytes``, any expression) is left to the reach-search, per the expression-
    inversion non-goal.

    Returns an entry-parameter assignment to try (execution still judges it), or None.
    """
    blocker_node = _find_branch_node(blocker_tree, blocker.branch_line)
    if blocker_node is None:
        return None
    callee_name = _func_name_containing(blocker_tree, blocker.branch_line)
    if callee_name is None or callee_name == entry_func:
        return None                       # the guard is in the entry itself — portfolio's job
    guard_vars = _free_names_of(blocker_node)
    callee_params = _func_params(blocker_tree, callee_name)
    target_vars = guard_vars & callee_params
    if not target_vars or (target_vars & params):
        # No callee parameter to propagate, or a name that is ALREADY an entry parameter
        # (the portfolio handles that). A guard on a callee LOCAL (a slice result) has no
        # callee parameter here and correctly falls through to reach-search.
        return None

    from tharkun.solve.frontend.call_mapper import find_calls_to

    mapping: dict[str, str] = {}           # callee param -> entry parameter
    for site in find_calls_to(source_path, callee_name):
        cand: dict[str, str] = {}
        for var in target_vars:
            am = next((a for a in site.args if a.callee_param == var), None)
            if am is None or not am.is_simple_name or am.caller_name not in params:
                cand = {}
                break
            cand[var] = am.caller_name
        if len(cand) == len(target_vars):
            mapping = cand
            break
    if len(mapping) != len(target_vars):
        return None                        # a non-simple / non-entry argument → non-goal

    from tharkun.solve.condition import solve_condition

    sol = solve_condition(blocker_path, blocker.branch_line, desired_branch=desired)
    assign = sol.satisfying_inputs[0] if sol.satisfying_inputs else {}
    proposal: dict[str, Any] = {}
    for var, entry_param in mapping.items():
        if var in assign:
            proposal[entry_param] = assign[var]
    return proposal or None


# ---------------------------------------------------------------------------
# [internal ref] PR2: the preimage flipper -- backward propagation through str / format / len / set / re operations
# ---------------------------------------------------------------------------
#
# A guard over a LOCAL derived from an entry parameter (python-string-utils is_email:233,
# ``head.startswith('"') and head.endswith('"')`` where ``head, tail = input_string.split('@')`` then
# ``head = head.replace('\\ ', '')``) is exactly where every flipper above abstains: they solve the guard for names
# that are parameters.  ``_flip_preimage`` turns the guard into a demand (``solve/preimage/want.py``) on each local
# it reads, walks the function's straight-line definitions BACKWARD through the preimage engine
# (``solve/preimage``, a visitor over the contract rules) until only entry parameters are left, and proposes their
# values.  Every proposal goes through the same ``_verified`` execution gate as every other flipper's: a wrong
# candidate costs one verify run, never a commit.  Anything the walk cannot express (a loop or branch assigning a
# wanted local, a user call, an attribute, an unmodelled expression) makes the flipper abstain for that guard.

_PREIMAGE_PER_STEP = 8
_PREIMAGE_LIMIT = 16


def _function_node_containing(tree: ast.Module, line: int):
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lo, hi = node.lineno, getattr(node, "end_lineno", node.lineno) or node.lineno
            if lo <= line <= hi and (best is None or lo > best.lineno):
                best = node
    return best


def _stmts_before(body: list[ast.stmt], line: int) -> list[ast.stmt] | None:
    """The statements executed before the statement at *line* on the way to it, in order (straight-line; the
    compound statements on the path contribute the statements before the branch that holds *line*), or None when
    *line* is not in *body*."""
    out: list[ast.stmt] = []
    for st in body:
        lo, hi = st.lineno, getattr(st, "end_lineno", st.lineno) or st.lineno
        if not lo <= line <= hi:
            if hi < line:
                out.append(st)
            continue
        if lo == line or isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return out
        blocks = []
        if isinstance(st, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            blocks = [st.body, st.orelse]
        elif isinstance(st, (ast.With, ast.AsyncWith)):
            blocks = [st.body]
        elif isinstance(st, ast.Try):
            blocks = [st.body, *[h.body for h in st.handlers], st.orelse, st.finalbody]
        for b in blocks:
            inner = _stmts_before(b, line)
            if inner is not None:
                # a loop around the guard re-runs its own body: the walk stops at the loop (its targets are barriers)
                return out + ([st] if isinstance(st, (ast.For, ast.AsyncFor, ast.While)) else []) + inner
        return out
    return None


def _assigned_names(st: ast.stmt) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(st):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            out.add(node.id)
    return out


def _preimage_constant(node) -> tuple[bool, Any]:
    try:
        return True, ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return False, None


def _preimage_regexes(path: str) -> dict[str, Any]:
    """The module-level compiled regexes of *path* that are PROVEN constants (``solve/module_namespace.py``)."""
    import re as _re

    from tharkun.solve.module_namespace import regex_globals
    out: dict[str, Any] = {}
    try:
        recs = regex_globals(path)
    except Exception:  # noqa: BLE001 -- a hint: no regexes known
        return out
    here = os.path.realpath(path)
    for rec in recs:
        if not rec.get("proven") or type(rec.get("pattern")) is not str:
            continue
        for b in rec.get("bindings", ()):
            mod = b.get("module", "")
            if here.endswith(os.sep + mod) or os.path.basename(here) == os.path.basename(mod):
                try:
                    out[b["name"]] = _re.compile(rec["pattern"], int(rec.get("flags") or 0))
                except (_re.error, ValueError, TypeError):
                    pass
    return out


class _PreimageAbstain(Exception):
    """The backward walk cannot express this guard or definition (the flipper abstains)."""


class _PreimageChoices(Exception):
    """Separate parameter demands; projecting them to per-name unions would lose their correlation."""

    def __init__(self, choices):
        self.choices = choices


def _preimage_sub_matches(pattern, callback, subject, count, want, env):
    """Invert bounded copy/group/text templates, then check with native regex semantics (never the target lambda)."""
    import itertools
    import re

    from tharkun.solve.contracts.rules import Refused
    from tharkun.solve.preimage import Eq, HOLE, IsNone, preimages, regex_members as RM
    from tharkun.solve.preimage.visitor import _case_variants

    args = callback.args
    params = args.posonlyargs + args.args
    if count < 0 or len(params) != 1 or args.vararg or args.kwarg or args.kwonlyargs:
        return []
    match_name = params[0].arg

    def pieces(node):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return pieces(node.left) + pieces(node.right)
        if isinstance(node, ast.Constant) and type(node.value) is str:
            return [("literal", node.value)]
        if isinstance(node, ast.Name) and node.id != match_name:
            return [("name", node.id)]
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == match_name
                and node.func.attr == "group" and len(node.args) == 1 and not node.keywords):
            known, key = _preimage_constant(node.args[0])
            if known and ((type(key) is int and 0 <= key <= pattern.groups)
                          or (type(key) is str and key in pattern.groupindex)):
                return [("group", key)]
        raise _PreimageAbstain()

    try:
        atoms = pieces(callback.body)
    except _PreimageAbstain:
        return []
    if len(atoms) > 8 or not any(kind == "group" for kind, _ in atoms):
        return []

    def tokens(text):
        out, end = [], 0
        for index, match in enumerate(pattern.finditer(text)):
            if count and index >= count:
                break
            if match.start() == match.end():
                return []
            if match.start() > end:
                out.append(("source", (end, match.start())))
            for kind, value in atoms:
                if kind == "group":
                    span = match.span(value)
                    if span[0] < 0:
                        return []
                    out.append(("source", span))
                else:
                    out.append((kind, value))
            end = match.end()
        if end < len(text):
            out.append(("source", (end, len(text))))
        return out

    rows = []
    samples = RM.samples(((pattern.pattern, pattern.flags),))
    if not samples:
        # A trailing lookahead needs context; other proven patterns and literals can supply it.
        context = RM.samples(tuple((p.pattern, p.flags) for p in env["regexes"].values())) + list(env["pool"])
        samples = sorted({s for text in context for s in _case_variants(text)
                          if RM.native_member(pattern.pattern, pattern.flags, "search", s)}, key=lambda s: (len(s), s))[:8]
    # shortcut: eight sample repetitions and 256 case combinations; extend only with a measured missed template.
    for seed in itertools.islice(want.seeds(), _PREIMAGE_PER_STEP):
        if type(seed) is not str or len(seed) > 4096:
            continue
        for sample in samples:
            for repeat in range(1, min(8, len(seed)) + 1):
                exemplar = sample * repeat
                if RM.check_cost(pattern.pattern, pattern.flags, len(exemplar), "sub") > RM.WORK_LIMIT:
                    continue
                template = tokens(exemplar)
                if not template or len(template) > 16:
                    continue
                shape = "".join(re.escape(value) if kind == "literal" else
                                "(.*)" if kind == "name" or value[0] == value[1] else "(.+?)"
                                for kind, value in template)
                if RM.check_cost(shape, re.DOTALL, len(seed), "fullmatch") > RM.WORK_LIMIT:
                    continue
                split = re.fullmatch(shape, seed, re.DOTALL)
                if split is None:
                    continue
                captures = iter(split.groups())
                copied, names = {}, {}
                for kind, value in template:
                    if kind == "literal":
                        continue
                    part = next(captures)
                    target = copied if kind == "source" else names
                    if value in target and target[value] != part:
                        break
                    target[value] = part
                else:
                    spans = sorted(copied)
                    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
                        continue
                    variants = [tuple(dict.fromkeys(_case_variants(copied[span]))) for span in spans]
                    for parts in itertools.islice(itertools.product(*variants), 256):
                        text = exemplar
                        for (lo, hi), part in reversed(list(zip(spans, parts))):
                            text = text[:lo] + part + text[hi:]
                        if len(text) > 4096 or RM.check_cost(pattern.pattern, pattern.flags, len(text), "sub") > RM.WORK_LIMIT:
                            continue
                        checked = preimages("re.Pattern.search", IsNone(False), (pattern, HOLE),
                                            prior=Eq(text), pool=(text,), budget=1)
                        if isinstance(checked, Refused) or not tuple(checked):
                            continue
                        rendered = tokens(text)
                        if not rendered:
                            continue
                        result = "".join(text[value[0]:value[1]] if kind == "source" else
                                         names[value] if kind == "name" else value for kind, value in rendered)
                        if not want.holds(result):
                            continue
                        try:
                            back = _preimage_expr(subject, Eq(text), env)
                        except _PreimageAbstain:
                            continue
                        back = _merge_wants(back, {name: Eq(value) for name, value in names.items()})
                        if back not in rows:
                            rows.append(back)
                        if len(rows) >= _PREIMAGE_PER_STEP:
                            return rows
    return rows


def _preimage_expr(expr, want, env) -> dict[str, Any]:
    """``{name: Want}`` for the names *expr* reads such that the value of *expr* satisfies *want*."""
    import itertools

    from tharkun.solve.contracts.rules import Refused
    from tharkun.solve.preimage import HOLE, IsNone, OneOf, preimages

    if isinstance(expr, ast.Name):
        return {expr.id: want}
    pieces = _concat_pieces(expr)
    if pieces is not None and sum(1 for p in pieces if isinstance(p, ast.Name)) > 1:
        return _preimage_concat(pieces, want, env)
    call = None          # (qualname, operand nodes in call order, hole index, op)
    if isinstance(expr, ast.Call) and not any(isinstance(a, ast.Starred) for a in expr.args):
        f = expr.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in env["regexes"]:
            if not expr.args:
                raise _PreimageAbstain()
            call = (f"re.Pattern.{f.attr}", [env["regexes"][f.value.id], *expr.args], None, None)
        elif isinstance(f, ast.Attribute):
            call = (f"str.{f.attr}", [f.value, *expr.args], None, None)
        elif isinstance(f, ast.Name) and f.id in ("len", "set", "str", "ord", "chr", "repr", "ascii", "format"):
            call = (f.id, list(expr.args), None, None)
        if expr.keywords:
            raise _PreimageAbstain()
    elif isinstance(expr, ast.Subscript):
        key = expr.slice
        if isinstance(key, ast.Slice):
            parts = []
            for p in (key.lower, key.upper, key.step):
                ok, v = (True, None) if p is None else _preimage_constant(p)
                if not ok or not (v is None or type(v) is int):
                    raise _PreimageAbstain()
                parts.append(v)
            call = ("str.__getitem__", [expr.value, slice(*parts)], None, None)
        else:
            ok, v = _preimage_constant(key)
            if not ok or type(v) is not int:
                raise _PreimageAbstain()
            call = ("str.__getitem__", [expr.value, v], None, None)
    elif isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        call = ("protocol.binary_operator", [expr.left, expr.right], None, "+")
    elif isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mult):
        call = ("protocol.binary_operator", [expr.left, expr.right], None, "*")
    elif isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Mod):
        call = ("str.__mod__", [expr.left, expr.right], None, None)
    if call is None:
        raise _PreimageAbstain()
    qualname, operands, _, op = call
    if (qualname == "re.Pattern.sub" and len(operands) in (3, 4)
            and isinstance(operands[1], ast.Lambda) and not operands[1].args.defaults
            and not any(operands[1].args.kw_defaults)):
        if len(operands) == 4:
            known, count = _preimage_constant(operands[3])
            if not known or type(count) is not int:
                raise _PreimageAbstain()
        else:
            count = 0
        choices = _preimage_sub_matches(operands[0], operands[1], operands[2], count, want, env)
        # No match leaves the subject unchanged without executing the lambda.
        pre = preimages("re.Pattern.search", IsNone(True), (operands[0], HOLE), prior=want,
                        pool=tuple(itertools.islice(want.seeds(), _PREIMAGE_PER_STEP)) + env["pool"],
                        budget=_PREIMAGE_PER_STEP)
        got = () if isinstance(pre, Refused) else tuple(pre)
        if got:
            choices.append(_preimage_expr(operands[2], OneOf(got), env))
        if not choices:
            raise _PreimageAbstain()
        env.setdefault("rules", []).append("re.Pattern.search")
        raise _PreimageChoices(choices)
    values, hole_node = [], None
    for node in operands:
        if not isinstance(node, ast.AST):
            values.append(node)              # an already-resolved operand (a regex constant, a slice)
            continue
        ok, v = _preimage_constant(node)
        if ok:
            values.append(v)
        elif hole_node is None:
            hole_node = node
            values.append(HOLE)
        else:
            raise _PreimageAbstain()         # two unknown operands: not one chain
    if hole_node is None:
        raise _PreimageAbstain()
    if qualname == "str.__getitem__" and env["kinds"].get(_name_of(hole_node)) in (list, tuple):
        qualname = f"{env['kinds'][_name_of(hole_node)].__name__}.__getitem__"
    pre = preimages(qualname, want, tuple(values), op=op, pool=env["pool"], budget=_PREIMAGE_PER_STEP)
    if isinstance(pre, Refused):
        raise _PreimageAbstain()
    env.setdefault("rules", []).append(qualname)
    if pre.transfer is not None:
        back = _preimage_expr(hole_node, pre.transfer, env)
        if isinstance(hole_node, ast.Name) and not tuple(itertools.islice(pre.transfer.seeds(), 1)):
            got = tuple(itertools.islice(pre, _PREIMAGE_PER_STEP))
            if got:
                # Keep the exact demand and the rule's checked witnesses as distinct alternatives.
                raise _PreimageChoices([back, {hole_node.id: OneOf(got)}])
        return back
    got = tuple(itertools.islice(pre, _PREIMAGE_PER_STEP))
    if not got:
        raise _PreimageAbstain()
    return _preimage_expr(hole_node, OneOf(got), env)


def _concat_pieces(expr):
    """``a + '@' + b`` as its pieces in order (names and str constants), or None."""
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left, right = _concat_pieces(expr.left), _concat_pieces(expr.right)
        return None if left is None or right is None else left + right
    if isinstance(expr, ast.Name):
        return [expr]
    if isinstance(expr, ast.Constant) and type(expr.value) is str:
        return [expr.value]
    return None


def _preimage_concat(pieces, want, env) -> dict[str, Any]:
    """A concatenation of SEVERAL unknown names and str constants (``head + '@' + tail``): each seed of the want is
    split around the constants (a regex member ``a@aa.aa`` gives ``head='a'``, ``tail='aa.aa'``), one ``OneOf`` per
    name.  Every split is only a proposal: the drive (or the replay) judges the whole input."""
    import itertools
    import re as _re

    from tharkun.solve.preimage import Member, OneOf, regex_members
    names = [p.id for p in pieces if isinstance(p, ast.Name)]
    if len(set(names)) != len(names):
        raise _PreimageAbstain()
    rx = _re.compile("".join("(.*?)" if isinstance(p, ast.Name) else _re.escape(p) for p in pieces) + r"\Z", _re.DOTALL)
    values: dict[str, list] = {n: [] for n in names}
    seeds = []
    if (isinstance(want, Member) and len(pieces) == 3
            and isinstance(pieces[1], str) and len(pieces[1]) == 1):
        edges = [env.get("length_boundaries", {}).get(name, [None]) for name in names]
        for lengths in itertools.islice(itertools.product(*edges), 16):
            if all(value is None for value in lengths) or sum(value or 0 for value in lengths) + 1 > 4096:
                continue
            seeds.extend(seed for seed in regex_members.members(want.pattern, want.flags, how=want.how, limit=1,
                                                                  split=pieces[1], segments=lengths)
                         if len(seed) <= 4096)
    for seed in itertools.chain(seeds, itertools.islice(want.seeds(), 16)):
        m = rx.match(seed) if type(seed) is str else None
        if m is None:
            continue
        for n, v in zip(names, m.groups()):
            if v not in values[n]:
                values[n].append(v)
    if not all(values.values()):
        raise _PreimageAbstain()
    env.setdefault("rules", []).append("protocol.binary_operator")
    return {n: OneOf(tuple(vs[:_PREIMAGE_PER_STEP])) for n, vs in values.items()}


def _name_of(node) -> str | None:
    return node.id if isinstance(node, ast.Name) else None


def _merge_wants(a: dict, b: dict) -> dict:
    from tharkun.solve.preimage import All
    out = dict(a)
    for k, w in b.items():
        out[k] = All.of(out[k], w) if k in out else w
    return out


def _preimage_guard(test, flag: bool, env) -> list[dict]:
    """The alternatives (a disjunction) of ``{name: Want}`` under which *test* evaluates to *flag*."""
    from tharkun.solve.preimage import Compare, Eq, IsNone, Truth

    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return _preimage_guard(test.operand, not flag, env)
    if isinstance(test, ast.BoolOp):
        conj = isinstance(test.op, ast.And) == flag     # And wanted true / Or wanted false: every operand
        parts = [_preimage_guard(v, flag, env) for v in test.values]
        if conj:
            # an operand the walk cannot state constrains nothing it proposes (a user call in ``not ok(s) or ...``):
            # it is left to execution, which judges every proposal anyway; with no operand stated, nothing is
            alts = [{}]
            stated = False
            for p in parts:
                if not p:
                    continue
                stated = True
                alts = [_merge_wants(a, b) for a in alts for b in p][:8]
            return alts if stated else []
        return [a for p in parts for a in p][:8]
    try:
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            op, right = test.ops[0], test.comparators[0]
            ok_r, rv = _preimage_constant(right)
            ok_l, lv = _preimage_constant(test.left)
            if isinstance(op, (ast.In, ast.NotIn)) and ok_l and type(lv) is str:
                hit = isinstance(op, ast.In) == flag
                call = ast.Call(func=ast.Attribute(value=right, attr="__contains__", ctx=ast.Load()),
                                args=[test.left], keywords=[])
                return [_preimage_expr(call, Truth(hit), env)]
            if isinstance(op, (ast.Is, ast.IsNot)) and ok_r and rv is None:
                none = isinstance(op, ast.Is) == flag
                return [_preimage_expr(test.left, IsNone(none), env)]
            sym = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}.get(type(op))
            if sym and ok_r and not ok_l:
                if not flag:
                    sym = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}[sym]
                want = Eq(rv) if sym == "==" else Compare(sym, rv)
                return [_preimage_expr(test.left, want, env)]
            if sym and ok_l and not ok_r:
                if not flag:
                    sym = {"==": "!=", "!=": "==", "<": ">=", "<=": ">", ">": "<=", ">=": "<"}[sym]
                flipped = {"==": "==", "!=": "!=", "<": ">", "<=": ">=", ">": "<", ">=": "<="}[sym]
                want = Eq(lv) if flipped == "==" else Compare(flipped, lv)
                return [_preimage_expr(right, want, env)]
            return []
        return [_preimage_expr(test, Truth(flag), env)]
    except _PreimageChoices as choices:
        return choices.choices
    except _PreimageAbstain:
        return []


def _preimage_back(wants: dict, stmts: list[ast.stmt], env) -> list[dict]:
    """Move each correlated alternative back over definitions until only entry names remain."""
    from tharkun.solve.preimage import Eq, Items

    alts = [dict(wants)]
    for st in reversed(stmts):
        nxt = []
        for original in alts:
            wants = dict(original)
            if not _assigned_names(st) & set(wants):
                nxt.append(wants)
                continue
            try:
                if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                    w = wants.pop(st.targets[0].id)
                elif (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Tuple)
                      and all(isinstance(e, ast.Name) for e in st.targets[0].elts)):
                    names = [e.id for e in st.targets[0].elts]
                    items = tuple((i, wants.pop(n)) for i, n in enumerate(names) if n in wants)
                    env["kinds"][_name_of(st.value) or ""] = list
                    w = Items(Eq(len(names)), items)
                else:
                    continue
                choices = [_preimage_expr(st.value, w, env)]
            except _PreimageChoices as exc:
                choices = exc.choices
            except _PreimageAbstain:
                continue
            nxt.extend(_merge_wants(wants, back) for back in choices)
        alts = nxt[:_PATHS_MAX]
    return alts


def _flip_preimage(source_path: str, func_name: str, blocker_tree: ast.Module, blocker: GuardFrame,
                   blocker_path: str, params: set[str], inputs: dict[str, Any],
                   snapshot: dict[str, Any] | None = None, priors: dict[str, Any] | None = None,
                   rules_out: list | None = None) -> list[dict[str, Any]]:
    """Entry-parameter proposals that make the blocking guard take ``blocker.desired_branch``, by backward
    propagation through the str / format / len / set / re operations that derive the guard's locals ([internal ref]).
    Proposals only: every one is verified by execution before it is committed.

    ``priors`` ([internal ref] PR5) maps an entry parameter to a demand every admissible value satisfies
    (``solve/preimage/facts.py``: the drive's pins and bounds, absint / FactEnv facts): a proposal outside it is
    dropped before it is run.  ``rules_out`` receives the qualnames the walk went through (the rule_fact hints)."""
    import itertools

    from tharkun.solve.preimage import OneOf

    fn = _function_node_containing(blocker_tree, blocker.branch_line)
    node = _find_branch_node(blocker_tree, blocker.branch_line)
    if fn is None or node is None:
        return []
    test = node.test if isinstance(node, (ast.If, ast.While, ast.Assert)) else node   # the guard expression
    fparams = {p.arg for p in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    if os.path.realpath(blocker_path) == os.path.realpath(source_path) and fn.name == func_name.rsplit(".", 1)[-1]:
        mapping = {p: p for p in fparams & params}
    else:
        from tharkun.solve.frontend.call_mapper import find_calls_to
        mapping = {}
        for site in find_calls_to(source_path, fn.name):
            for am in site.args:
                if am.is_simple_name and am.caller_name in params and am.callee_param in fparams:
                    mapping.setdefault(am.callee_param, am.caller_name)
    if not mapping:
        return []
    pool = []
    for n in ast.walk(fn):
        if isinstance(n, ast.Constant) and type(n.value) is str and n.value and len(n.value) <= 32 \
                and n.value not in pool and not (isinstance(n, ast.Constant) and n is getattr(fn.body[0], "value", None)):
            pool.append(n.value)
    kinds = {k: type(v) for k, v in inputs.items() if type(v) in (list, tuple, str)}
    for name, value in (snapshot or {}).items():
        kind = ({"list": list, "tuple": tuple, "str": str}.get(value.get("__type__"))
                if type(value) is dict else type(value))
        if kind in (list, tuple, str):
            kinds[name] = kind
    env = {"regexes": _preimage_regexes(blocker_path), "pool": tuple(pool[:32]), "kinds": kinds}
    stmts = _stmts_before(fn.body, blocker.branch_line)
    if stmts is None:
        return []
    alts = [back for alt in _preimage_guard(test, blocker.desired_branch == "then", env)
            for back in _preimage_back(alt, stmts, env)]
    if not env.get("rules"):
        # no str / format / len / set / re step on the way: a comparison on a parameter (or a pass-through of one) is
        # the static flipper's and call_mapper's question, not a preimage
        return []
    out: list[dict[str, Any]] = []
    priors = priors or {}
    for back in alts:
        if not back or not set(back) <= set(mapping):
            continue
        per_param = []
        for local, w in back.items():
            vals = list(w.values) if isinstance(w, OneOf) else [v for v in itertools.islice(w.seeds(), 8)
                                                                  if w.holds(v)]
            prior = priors.get(mapping[local])
            if prior is not None:
                vals = [v for v in vals if prior.holds(v)]
            if not vals:
                break
            per_param.append([(mapping[local], v) for v in vals[:_PREIMAGE_PER_STEP]])
        else:
            for combo in itertools.product(*per_param):
                out.append(dict(combo))
                if len(out) >= _PREIMAGE_LIMIT:
                    break
    if rules_out is not None:
        rules_out.extend(dict.fromkeys(env.get("rules", ())))
    return out[:_PREIMAGE_LIMIT]


def _preimage_priors(context_facts: dict | None, param_bounds: dict | None) -> dict[str, Any]:
    """The demands a drive's own pins and bounds put on its entry parameters (``solve/preimage/facts.py``)."""
    from tharkun.solve.preimage.facts import prior_from_bounds, prior_from_pin
    from tharkun.solve.preimage.want import All
    out: dict[str, Any] = {}
    for k, v in (context_facts or {}).items():
        out[k] = prior_from_pin(v)
    for k, b in (param_bounds or {}).items():
        out[k] = All.of(out[k], prior_from_bounds(b)) if k in out else prior_from_bounds(b)
    return out


def _publish_preimage_hints(store, rules: list, path: str, line: int, want_text: str, proposal: dict) -> None:
    """One ``rule_fact`` hint per rule a committed preimage flip went through ([internal ref] PR5; never certified)."""
    if store is None or not hasattr(store, "publish"):
        return
    from tharkun.solve.preimage.facts import rule_fact
    site = f"{os.path.basename(path)}:{line}"
    for q in rules:
        key, value = rule_fact(q, site, want_text, proposal)
        try:
            store.publish("rule_fact", key, value, producer="driver:preimage")
        except ValueError:
            pass                       # the same rule at the same site already holds a (different) first hint


# ---------------------------------------------------------------------------
# [internal ref] PR6: preimage candidates for a demand on a function's RESULT (tharkun.inputs' preimage rung)
# ---------------------------------------------------------------------------

_PATHS_MAX = 16


def _exits(body: list[ast.stmt]) -> bool:
    """Does *body* always leave the function (its last statement a return or raise)?"""
    return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise))


def _path_items(body: list[ast.stmt], target: ast.stmt) -> list[list[tuple]] | None:
    """The execution paths from the start of *body* to *target* (a statement in it or nested in it), each a list
    of ``("stmt", st)`` / ``("cond", test, flag)`` in order; None when *target* is not under *body*.  A branch the
    path does not enter must not exit the function (so its guard takes the other value); a loop on the way, or
    one before the target that assigns, ends the enumeration (the walk does not unroll)."""
    paths: list[list[tuple]] = [[]]
    for st in body:
        if st is target:
            return paths
        lo, hi = st.lineno, getattr(st, "end_lineno", st.lineno) or st.lineno
        inside = lo <= target.lineno <= hi
        if isinstance(st, ast.If):
            if inside:
                for branch, flag in ((st.body, True), (st.orelse, False)):
                    sub = _path_items(branch, target)
                    if sub is not None:
                        return [p + [("cond", st.test, flag)] + q for p in paths for q in sub][:_PATHS_MAX]
                return None
            # a branch before the target: either side, unless it leaves the function
            sides = []
            for branch, flag in ((st.body, True), (st.orelse, False)):
                if _exits(branch):
                    continue
                flat = _straight(branch)
                if flat is None:
                    continue
                sides.append([("cond", st.test, flag)] + flat)
            if not sides:
                return None
            paths = [p + s for p in paths for s in sides][:_PATHS_MAX]
            continue
        if isinstance(st, ast.Try):
            if inside:
                sub = _path_items(st.body, target)
                if sub is None:
                    return None                 # the target is in a handler: an exception path, not walked
                return [p + q for p in paths for q in sub][:_PATHS_MAX]
            flat = _straight(st.body)
            if flat is None:
                return None
            paths = [p + flat for p in paths]
            continue
        if isinstance(st, (ast.With, ast.AsyncWith)) and inside:
            sub = _path_items(st.body, target)
            return None if sub is None else [p + q for p in paths for q in sub][:_PATHS_MAX]
        if isinstance(st, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Match)):
            if inside or _assigned_names(st):
                return None
            continue
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        paths = [p + [("stmt", st)] for p in paths]
    return None


def _straight(body: list[ast.stmt]) -> list[tuple] | None:
    """A block with no branching, as items (None when it branches or loops)."""
    out = []
    for st in body:
        if isinstance(st, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)):
            return None
        if isinstance(st, (ast.Return, ast.Raise)):
            return None
        out.append(("stmt", st))
    return out


def _world_proposals(tree, target_line, bindings, world, path):
    """Project existing path items onto file contents; unbound names stay unbound."""
    import copy
    from dataclasses import replace
    from tharkun.lattice import Origin
    from tharkun.solve.world import WorldRefused

    fn = _function_node_containing(tree, target_line) or tree
    target = next((n for n in ast.walk(fn) if isinstance(n, ast.stmt) and n.lineno == target_line), None)
    if target is None:
        return
    keys = list(world.objects)
    subjects = {key: f"world{i}" for i, key in enumerate(keys)}
    for items in _path_items(fn.body, target) or ():
        env = {name: ast.Constant(value) for name, value in bindings.items() if type(value) in (str, int, bool)}
        streams, reads = {}, {}
        proposed = {key: replace(obj) for key, obj in world.objects.items()}

        def opened(call):
            if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "open"
                    or not call.args or len(call.args) > 2 or call.keywords
                    or (len(call.args) == 2 and ast.literal_eval(call.args[1]) not in {"r", "rt"})):
                raise WorldRefused("world drive supports read-only builtin open")
            expr = project(call.args[0])
            value = ast.literal_eval(expr)
            for key in keys:
                if value == key or (isinstance(value, str) and os.path.abspath(os.path.join(os.path.dirname(path), value))
                                    == os.path.abspath(os.path.join(os.path.dirname(path), key))):
                    return key
            raise WorldRefused("world drive path is not declared in the request")

        class Project(ast.NodeTransformer):
            def visit_Name(self, node):
                return copy.deepcopy(env.get(node.id, node))

            def visit_Call(self, node):
                if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute)
                        and isinstance(node.func.value.value, ast.Name) and node.func.value.value.id == "sys"
                        and node.func.value.attr in {"stdin", "__stdin__"}
                        and "stdin" in proposed and proposed["stdin"].kind == "stream"):
                    streams["$stdin"] = "stdin"
                    node = copy.deepcopy(node)
                    node.func.value = ast.Name("$stdin", ast.Load())
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                        and node.func.value.id in streams:
                    handle = node.func.value.id
                    key = streams[handle]
                    count = reads.get(handle, 0)
                    if node.args or node.keywords or node.func.attr not in {"read", "readline"}:
                        raise WorldRefused("world drive supports unsized read/readline")
                    if node.func.attr == "read":
                        if count:
                            raise WorldRefused("world drive cannot project read after readline yet")
                        reads[handle] = -1
                        return ast.Name(subjects[key], ast.Load())
                    if count < 0:
                        raise WorldRefused("world drive cannot project readline after read yet")
                    reads[handle] = count + 1
                    return ast.Subscript(ast.Name(subjects[key] + "_lines", ast.Load()), ast.Constant(count), ast.Load())
                return self.generic_visit(node)

        def project(expr):
            return ast.fix_missing_locations(Project().visit(copy.deepcopy(expr)))

        def add(expr, lineno, polarity=""):
            names = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}
            affected = [key for key, subject in subjects.items() if names & {subject, subject + "_lines"}]
            if not affected:
                return
            if len(affected) != 1:
                raise WorldRefused("world drive cannot project a relation between files yet")
            key = affected[0]

            class Subject(ast.NodeTransformer):
                def visit_Name(self, node):
                    return ast.Name({subjects[key]: "content", subjects[key] + "_lines": "lines"}.get(node.id, node.id), ast.Load())

            fact = ast.unparse(ast.fix_missing_locations(Subject().visit(copy.deepcopy(expr))))
            proposed[key].add(fact, Origin(path, lineno, polarity=polarity))

        try:
            for node in ast.walk(fn):
                if isinstance(node, ast.With) and node.lineno < target_line <= node.end_lineno:
                    for item in node.items:
                        if not isinstance(item.optional_vars, ast.Name):
                            raise WorldRefused("world drive needs a named file handle")
                        streams[item.optional_vars.id] = opened(item.context_expr)
            for item in items:
                if item[0] == "cond":
                    expr = project(item[1])
                    if not item[2]:
                        expr = ast.UnaryOp(ast.Not(), expr)
                    add(expr, item[1].lineno, "then" if item[2] else "else")
                    continue
                node = item[1]
                if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                    if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "open":
                        streams[name] = opened(node.value)
                        continue
                    expr = project(node.value)
                    env[name] = expr
                    for call in ast.walk(expr):
                        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "int":
                            add(ast.Compare(call, [ast.Eq()], [copy.deepcopy(call)]), node.lineno)
            if any(obj.concrete != world.objects[key].concrete for key, obj in proposed.items()):
                yield proposed
        except (WorldRefused, ValueError, TypeError, SyntaxError) as exc:
            yield WorldRefused(str(exc))


def _preimage_walk(items: list[tuple], wants: dict, env) -> list[dict]:
    """Move *wants* back over a path's items (latest first); guards on the path add their demands."""
    alts = [dict(wants)]
    for item in reversed(items):
        nxt = []
        for w in alts:
            if item[0] == "cond":
                for g in _preimage_guard(item[1], item[2], env) or [{}]:
                    nxt.append(_merge_wants(w, g))
            else:
                nxt.extend(_preimage_back(w, [item[1]], env))
        alts = nxt[:_PATHS_MAX]
        if not alts:
            return []
    return alts


def _return_wants(expr, want, env) -> list[dict] | None:
    """The demands on locals under which ``return <expr>`` gives a value satisfying *want*: ``None`` for a constant
    that does not satisfy it; ``[{}]`` (the path's guards only) for an expression the walk cannot state, since
    reaching that return is still what a result-changing mutant on the path needs."""
    from tharkun.solve.preimage import Eq, Truth
    ok, value = _preimage_constant(expr)
    if ok:
        return [{}] if want.holds(value) else None
    flag = None
    if isinstance(want, Truth):
        flag = want.flag
    elif isinstance(want, Eq) and type(want.value) is bool:
        flag = want.value
    if flag is not None and isinstance(expr, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
        alts = _preimage_guard(expr, flag, env)
        return alts or [{}]
    try:
        return [_preimage_expr(expr, want, env)]
    except _PreimageChoices as choices:
        return choices.choices
    except _PreimageAbstain:
        return [{}]


def _passes(items: list[tuple], ret: ast.stmt, line: int) -> bool:
    """Does the path (its items, then ``ret``) run a statement or test the guard that spans *line*?"""
    for node in [ret] + [item[1] for item in items]:
        if node.lineno <= line <= (getattr(node, "end_lineno", None) or node.lineno):
            return True
    return False


def preimage_candidates(source_path: str, func_name: str, want, *, params: set[str] | None = None,
                        limit: int = 16, through: int | None = None) -> list[dict[str, Any]]:
    """Parameter values for which ``func_name`` (a top-level function of *source_path*, or ``Class.method``, [internal ref]) may RETURN a value that
    satisfies *want* (``solve/preimage/want.py``), by walking each ``return`` statement's paths backward through
    ``solve/preimage``: the returned expression's demand, the guards on the path, the definitions before it.
    Proposals only (``tharkun.inputs``' preimage rung replays every one in a fresh child); a path the walk cannot
    express contributes nothing.  [internal ref] PR6.

    ``through`` ([internal ref]) keeps only the paths that run the statement or test the guard on that line of
    *source_path* (``tharkun.inputs``' reach rung: the line a drive reached, so the walk proposes the propagation
    half of a forward kill -- the values that carry what changed at that line to the result)."""
    import itertools

    from tharkun.solve.preimage import OneOf
    try:
        tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return []
    fn = _def_node(tree, func_name)
    if not isinstance(fn, ast.FunctionDef):
        return []
    fparams = {p.arg for p in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    params = fparams if params is None else (set(params) & fparams)
    pool = [n.value for n in ast.walk(fn) if isinstance(n, ast.Constant) and type(n.value) is str
            and 0 < len(n.value) <= 32]
    # Boundary ordering is for anchored searches; unanchored callers keep their shortest-first proposals.
    boundaries = {}
    for cmp in (node for node in ast.walk(fn) if through is not None and isinstance(node, ast.Compare)):
        sides = [cmp.left, *cmp.comparators]
        for left, right in zip(sides, sides[1:]):
            for call, constant in ((left, right), (right, left)):
                if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
                        or call.func.id != "len" or len(call.args) != 1 or not isinstance(call.args[0], ast.Name)):
                    continue
                known, value = _preimage_constant(constant)
                if known and type(value) is int and 0 <= value <= 4096:
                    rows = boundaries.setdefault(call.args[0].id, [])
                    rows.extend(v for v in (value, value + 1, value - 1) if 0 <= v <= 4096 and v not in rows)
    base_env = {"regexes": _preimage_regexes(source_path), "pool": tuple(dict.fromkeys(pool))[:32]}

    def proposals(lengths):
        env = dict(base_env, kinds={}, length_boundaries=lengths)
        for ret in (n for n in ast.walk(fn) if isinstance(n, ast.Return) and n.value is not None):
            starts = _return_wants(ret.value, want, env)
            if starts is None:
                continue
            for items in _path_items(fn.body, ret) or []:
                if through is not None and not _passes(items, ret, through):
                    continue
                for back in (b for start in starts for b in _preimage_walk(items, start, env)):
                    if not back or not set(back) <= params:
                        continue
                    per = []
                    for name, w in back.items():
                        vals = list(w.values) if isinstance(w, OneOf) else [v for v in itertools.islice(w.seeds(), 8)
                                                                              if w.holds(v)]
                        if not vals:
                            break
                        per.append([(name, v) for v in vals[:_PREIMAGE_PER_STEP]])
                    else:
                        for combo in itertools.product(*per):
                            yield dict(combo)

    streams = (itertools.zip_longest(proposals({}), proposals(boundaries)) if boundaries
               else ((row,) for row in proposals({})))
    out: list[dict[str, Any]] = []
    seen: set = set()
    for group in streams:
        for row in group:
            if row is None:
                continue
            key = repr(sorted(row.items()))
            if key not in seen:
                seen.add(key)
                out.append(row)
                if len(out) >= limit:
                    return out
    return out


def drive_to_line(
    source_path: str,
    func_name: str,
    target_line: int,
    seed: dict[str, Any],
    *,
    max_rounds: int = 16,
    target_path: str | None = None,
    context_facts: dict[str, Any] | None = None,
    store: Any = None,
    flippers: list[Flipper] | None = None,
    param_bounds: dict[str, tuple[int, int]] | None = None,
    heavy: bool = True,
    minleaf: bool = True,
    avengers_budget_seconds: float = 10.0,
    confine=None,
    world=None,
    _source_binding: tuple[str, str, str] | None = None,
) -> DriveResult:
    """Drive execution of ``func_name`` toward *target_line*, starting from *seed*.

    Each blocking guard is solved by a **portfolio**: snapshot and static flippers
    run first and every applicable candidate is *verified by execution*
    before it is committed — the tracer is the judge, so a wrong candidate (or one behind
    a false 'infeasible' vote) is discarded.  When ``heavy`` is set (default) and the
    cheap tier finds nothing it escalates to ``minleaf`` — a minimal mutation of the live
    captured state, which needs no modeling of the guard and so covers the constructs the
    SMT lane abstains on — and then to hypothesis property search.

    Reach-search uses :func:`tharkun.redqueen.harvest` to observe live comparison
    operands and bank their inverted proposals for later searches.

    Avengers ([internal ref] slice 3, [internal ref]) runs unconditionally, on demand only —
    no boolean gate, same as every other tier above it — after ``minleaf`` for a guard on
    an attribute of a parameter minleaf found nothing to mutate for (no live value
    observed). It asks :mod:`tharkun.avengers` to CONSTRUCT an object satisfying the
    guard's own field bounds, for that one parameter only, never as an unconditional
    sweep of every class in the file — a task with no such guard spends 0 seconds here.
    ``avengers_budget_seconds`` caps the CPU time spent constructing per blocked guard,
    from the blocked guard's own share of the round budget.

    ``param_bounds`` gives an inclusive ``(lo, hi)`` range per parameter — the driver
    adds it to the solver's precondition so a flip returns a value inside the domain the
    caller can actually use (e.g. a fuzzer field's fixed byte width).  A parameter absent
    from the map is unconstrained.

    context_facts pins parameters whose values the callers fix, so fact-driven
    search preserves that context.

    ``target_path`` names the file *target_line* lives in when that is not the entry
    function's own file — a line one or more CALLS below the entry point.  Such a line is
    in the coverage report exactly like any other, so it is a legitimate target; what it
    needs is for the tracer to watch that file too, and for the blocker search to range
    over both the callee's guards and the caller's.  Defaults to *source_path*.

    Returns a :class:`DriveResult`.  ``status == REACHED`` means the target line
    actually executed with ``result.inputs``; ``PLATEAU`` means no flippable
    guard remained (diagnostics explain what blocked); ``BUDGET`` means the round
    cap was hit.

    ``_source_binding`` is the trusted generator's (generated text, original path,
    original text). Confined generated code is admitted against its original source
    only while its current bytes still match that exact generated text.
    """
    if confine is not None:
        from tharkun.solve.extern import native_import_reason
        with open(source_path, "rb") as stream:
            admitted_source = stream.read()
        admission_path = source_path
        if _source_binding is not None:
            generated, admission_path, original = _source_binding
            if admitted_source != generated.encode("utf-8"):
                return DriveResult(NO_TARGET, dict(seed), target_line=target_line,
                                   diagnostics="generated source changed after admission")
            admitted_source = original
        refusal = native_import_reason(admitted_source, admission_path, reversible=True)
        if refusal:
            return DriveResult(NO_TARGET, dict(seed), target_line=target_line, diagnostics=refusal)
    param_bounds = param_bounds or {}
    preimage_priors = _preimage_priors(context_facts, param_bounds)
    # Strategies read and write one knowledge base for the drive. Pass a shared store to
    # carry what was learned across drives — a verdict decided for one frontier is still
    # valid for the next one that asks the same question.
    global _STORE
    _prev_store, _STORE = _STORE, (store if store is not None else _new_store())
    # An explicit empty list disables the cheap tier (heavy-only); None means default.
    flippers = _DEFAULT_FLIPPERS if flippers is None else flippers
    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError) as exc:
        return DriveResult(NO_TARGET, dict(seed), target_line=target_line, diagnostics=str(exc))

    tp = os.path.realpath(target_path) if target_path else rp
    if tp != rp:
        try:
            target_tree = ast.parse(Path(tp).read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError) as exc:
            return DriveResult(NO_TARGET, dict(seed), target_line=target_line,
                               diagnostics=str(exc))
    else:
        target_tree = tree

    frames = enclosing_guards(target_tree, target_line)
    if not frames and tp == rp:
        return DriveResult(
            NO_TARGET, dict(seed), target_line=target_line,
            diagnostics=f"line {target_line} is not inside any if-guard",
        )
    params = _func_params(tree, func_name)
    guard_lines = {f.branch_line for f in frames}
    # Guards in the callee are snapshotted in the callee's file, not the entry's.
    if tp != rp:
        extra_paths = frozenset({tp})
        extra_snaps = {tp: guard_lines | {target_line}}
        snapshot_lines: set[int] = set()
    else:
        extra_paths = frozenset()
        extra_snaps = {}
        snapshot_lines = guard_lines | {target_line}
    if world is not None:
        observed_lines = {n.lineno for n in ast.walk(target_tree) if isinstance(n, ast.stmt) and n.lineno <= target_line}
        if tp == rp:
            snapshot_lines |= observed_lines
        else:
            extra_snaps[tp] |= observed_lines

    from tharkun.lattice import TOP, Origin

    inputs = dict(seed)
    history: list[DriveStep] = []
    seen: set[tuple] = set()
    # The accumulated path condition, as an abstract state rather than a list of strings.
    # Every guard we deliberately satisfy is MET into it and constrains every LATER solve.
    # Re-deriving facts from the current trace is not enough: once a flip moves execution
    # elsewhere, a constraint we already paid for stops being observable and would be
    # silently dropped — so the solver is free to undo it, and the loop can circle. Facts
    # are collected once and kept, each ANCHORED to the program point that established it,
    # so a reassignment on the path can invalidate exactly the ones it touches (a stale
    # `(a > 100)` carried past `a = b` is what turned a reachable line into a false dead).
    state = TOP
    # Binding sites per file, for the flow-sensitivity filter — computed once per drive.
    _binding_cache: dict[str, tuple] = {}

    def _bindings_for(path: str, atree: ast.Module) -> tuple:
        rpk = os.path.realpath(path)
        got = _binding_cache.get(rpk)
        if got is None:
            got = _binding_index(atree)
            _binding_cache[rpk] = got
        return got

    # A verification is a deterministic question WITHIN a drive — the same merged inputs
    # make the same guard side execute or they do not — so a candidate that already FAILED
    # is not re-executed for the remainder of the path. Deliberately drive-LOCAL, not
    # stored on the shared store: across drives the entry function, the module's imported
    # (and possibly stateful) code, and the guard span at a given file:line can all differ,
    # so a memoised verdict from one drive is not sound to reuse in another.
    verify_memo: dict = {}

    def _verified(blk: GuardFrame, cand: dict[str, Any], bpath: str) -> bool:
        merged = {**inputs, **cand}
        vkey = (os.path.realpath(bpath), blk.branch_line, blk.desired_branch,
                tuple(sorted((k, repr(v)) for k, v in merged.items())))
        if vkey in verify_memo:
            return verify_memo[vkey]
        ok = _flip_verifies(source_path, func_name, blk, inputs, cand, bpath)
        verify_memo[vkey] = ok
        return ok

    pruned: set[tuple[int, str]] = set()   # (branch_line, desired) certified dead
    # Committed flips, newest last: (inputs before the flip, the PRE-flip state, the guard
    # it satisfied). A guard deeper in is solved under whatever an earlier guard settled
    # on, and that choice can make the deeper one UNSATISFIABLE — `len(p) > 4` answered
    # with the str 'xxxxx' kills a later `p[0] == 0x7F`, which only holds for bytes.
    # Plateauing there blames the deeper guard for the earlier guard's choice. Instead the
    # deeper guard's requirement is added to the fact set and the earlier one re-solved
    # under both — and abandoning the fork restores the pre-flip STATE, so the earlier
    # guard's clause does not coexist with its own negation.
    trail: list[tuple[dict[str, Any], Any, GuardFrame | None, str]] = []
    used_stochastic = False
    backtracks = 0
    max_backtracks = 4

    from contextlib import ExitStack, nullcontext

    target_phase = ExitStack()
    try:
        target_phase.enter_context((confine or nullcontext)())
        for rnd in range(max_rounds):
            # Minleaf mutates the real objects captured at the guard.
            outcome = _run_traced(source_path, func_name, inputs, snapshot_lines,
                                  capture_real=minleaf or world is not None,
                                  extra_paths=extra_paths, extra_snapshot_lines=extra_snaps)
            if not outcome.ok:
                return DriveResult(
                    PLATEAU, inputs, rounds=rnd, target_line=target_line, history=history, used_stochastic=used_stochastic,
                    diagnostics=outcome.error,
                )
            # For a cross-file target the fallback must be EMPTY, never the entry file's
            # lines: line numbers are per-file, so defaulting to the entry's set makes an
            # unrelated line of the same number read as "the target executed".
            target_exec = (outcome.by_path.get(tp, set()) if tp != rp else outcome.executed)
            if target_line in target_exec:
                # Confirm the reach in a FRESH module state: an in-process drive accumulates
                # module-level globals across its traces, so a guard on a module counter can
                # read as reached when a first call with the same inputs would not. If the
                # fresh call does not reach, the target depends on non-parameter (module)
                # state — outside the driver's parameter-only scope — so report a plateau.
                if _reaches_in_fresh_state(source_path, func_name, inputs, target_line, tp,
                                           extra_paths, extra_snaps):
                    return DriveResult(
                        REACHED, inputs, rounds=rnd + 1, target_line=target_line,
                        reached=True, history=history, used_stochastic=used_stochastic,
                    )
                return DriveResult(
                    PLATEAU, inputs, rounds=rnd + 1, target_line=target_line, history=history,
                    used_stochastic=used_stochastic,
                    diagnostics="target reached only under accumulated in-process state; it "
                                "depends on non-parameter (module-level) state, not the inputs",
                )

            if world is not None:
                bindings = dict(inputs)
                snapshots = outcome.real_by_path.get(tp, {})
                for line in sorted(snapshots):
                    if snapshots[line]:
                        bindings.update(snapshots[line][-1])
                for proposed in _world_proposals(target_tree, target_line, bindings, world, tp):
                    from tharkun.solve.world import WorldRefused
                    if isinstance(proposed, WorldRefused):
                        history.append(DriveStep(rnd, dict(inputs), technique="world", note=str(proposed)))
                        continue
                    previous = {key: (obj.concrete, obj.facts, obj.mode) for key, obj in world.objects.items()}
                    for key, obj in proposed.items():
                        current = world.objects[key]
                        current.concrete, current.facts, current.mode = obj.concrete, obj.facts, obj.mode
                    accepted = False
                    try:
                        check = _run_traced(source_path, func_name, inputs, snapshot_lines,
                                            extra_paths=extra_paths, extra_snapshot_lines=extra_snaps)
                        executed = check.by_path.get(tp, set()) if tp != rp else check.executed
                        accepted = target_line in executed and _reaches_in_fresh_state(
                            source_path, func_name, inputs, target_line, tp, extra_paths, extra_snaps)
                        if accepted:
                            history.append(DriveStep(rnd, dict(inputs), technique="world",
                                                     note="file facts satisfied; target re-executed"))
                            return DriveResult(REACHED, inputs, rounds=rnd + 1, target_line=target_line,
                                               reached=True, history=history)
                    finally:
                        if not accepted:
                            for key, (concrete, facts, mode) in previous.items():
                                current = world.objects[key]
                                current.concrete, current.facts, current.mode = concrete, facts, mode

            # The blocker is the outermost enclosing guard whose target-side did not
            # execute — inner guards aren't even reached until it flips.
            #
            # For a cross-file target that means nothing at all while the callee has not run:
            # its guards have not been reached, so "did not enter" says only that we never
            # got there. The blocker is in the caller, and leaving it to the lexical frames
            # makes the driver flip an inner guard that changes nothing. Hand those rounds
            # straight to the caller-side search, which verifies by progress.
            blocker: GuardFrame | None = None
            for f in (frames if (tp == rp or target_exec) else []):
                target_side = f.then_span if f.desired_branch == "then" else f.else_span
                side_entered = target_side is not None and any(
                    target_side[0] <= ln <= target_side[1] for ln in target_exec
                )
                if not side_entered:
                    blocker = f
                    break

            # "No blocker among the frames" means all guards are satisfied only when those
            # frames were actually evaluated. For a cross-file target whose callee has not
            # run, it means the opposite — we never got there — and the search below is what
            # handles it.
            if blocker is None and frames and (tp == rp or target_exec):
                history.append(DriveStep(rnd, dict(inputs), note="target-side entered but line not hit"))
                return DriveResult(
                    PLATEAU, inputs, rounds=rnd + 1, target_line=target_line, history=history, used_stochastic=used_stochastic,
                    diagnostics="all enclosing guards satisfied but target line still not executed "
                                "(unmodeled control flow between guard and target)",
                )

            if blocker is not None and (blocker.branch_line, blocker.desired_branch) in pruned:
                history.append(DriveStep(rnd, dict(inputs), blocker.branch_line, blocker.desired_branch,
                                         note="blocker certified infeasible"))
                return DriveResult(
                    PLATEAU, inputs, rounds=rnd + 1, target_line=target_line, history=history, used_stochastic=used_stochastic,
                    diagnostics=f"guard at line {blocker.branch_line} ({blocker.desired_branch}) is "
                                f"provably unreachable — target line {target_line} is dead code",
                )

            # The blocking guard may live in the callee's file, not the entry's: snapshots,
            # the AST it is read from, and the file handed to a flipper all follow it.
            blocker_path = tp if (blocker is not None and tp != rp) else source_path
            blocker_tree = target_tree if (blocker is not None and tp != rp) else tree
            # Read state from the file the blocker is IN. Defaulting to the entry file's map
            # left every callee blocker with an empty snapshot, which silently disables the
            # live-state minleaf technique.
            b_snapshots = (outcome.snapshots_by_path.get(blocker_path, outcome.snapshots)
                           if blocker_path != source_path else outcome.snapshots)
            b_real = (outcome.real_by_path.get(blocker_path, {})
                      if blocker_path != source_path else outcome.real)

            # Snapshot at the blocker line (last hit), else empty.
            snaps = (b_snapshots.get(blocker.branch_line) or []) if blocker is not None else []
            snapshot = snaps[-1] if snaps else {}
            # The OBSERVED type of each parameter. Static inference is a guess and can be
            # flatly wrong — the subscript rule reads `token[:2]` as a list where the
            # program is running bytes — so an observed type must outrank an inferred one,
            # or feeding static facts to a search would make it worse rather than better.
            # Observation is certified in the only sense that matters here: it is what
            # actually happened.
            if _STORE is not None:
                from tharkun.facts import Fact
                from tharkun.lattice import TOP, path_class

                # From `inputs`, NOT from `snapshot`: the snapshot is serialized for
                # diagnostics, so its values are reprs and their types describe the
                # serialization rather than the program. `inputs` holds the actual
                # objects the driver is about to pass in.
                observed_state = TOP
                for _k, _v in inputs.items():
                    if _k in params and _v is not None:
                        _STORE.add(Fact("type", _k, type(_v).__name__,
                                        source="observed", certified=True, confidence=1.0))
                        observed_state = observed_state.with_types(_k, [type(_v).__name__])
                # Also record it into the lattice index at this program point, keyed by
                # the path class this execution took, so observations along the same path
                # aggregate and observations along different paths stay distinct (C2:
                # write-only — nothing reads the index yet, so no behaviour change).
                if not observed_state.is_top() and hasattr(_STORE, "observe"):
                    _STORE.observe((rp, target_line), path_class(outcome.by_path),
                                   observed_state)
            real_snaps = (b_real.get(blocker.branch_line) or []) if blocker is not None else []
            real_snapshot = real_snaps[-1] if real_snaps else {}

            # Express Boolean pins as comparisons shared by the fact solvers.
            def _pin(name: str, val: Any) -> str:
                if isinstance(val, bool):
                    return f"{name} != 0" if val else f"{name} == 0"
                return f"{name} == {val!r}"

            # The precondition: fold this round's transient facts (lexical prefix guards,
            # guards holding on the executed path, caller pins, field bounds) into the
            # threaded committed state at their own program points, then FILTER every
            # clause a reassignment on the path has since invalidated, then render.
            solve_state = state
            blocker_bindings: tuple = ((), [], {})
            if blocker is not None:
                blocker_node = _find_branch_node(blocker_tree, blocker.branch_line)
                blocker_vars = _free_names_of(blocker_node)
                bpath = os.path.realpath(blocker_path)
                blocker_bindings = _bindings_for(blocker_path, blocker_tree)
                fspan = _enclosing_func_span(blocker_tree, blocker.branch_line)
                fstart = fspan[0] if fspan else blocker.branch_line
                for text, line, node, pol in _prefix_clauses(blocker_tree, frames, blocker):
                    solve_state = solve_state.with_clause(text, Origin(bpath, line, polarity=pol),
                                                          node=node)
                for text, line, node in _satisfied_clauses(blocker_tree, target_exec,
                                                            blocker.branch_line, blocker_vars):
                    solve_state = solve_state.with_clause(text, Origin(bpath, line, polarity="then"),
                                                          node=node)
                for k, v in (context_facts or {}).items():
                    if k in blocker_vars:
                        solve_state = solve_state.with_clause(_pin(k, v), Origin(bpath, fstart))
                for text in _bounds_clause_texts(param_bounds, blocker_vars):
                    solve_state = solve_state.with_clause(text, Origin(bpath, fstart))
                by_line, loops, _self_rhs = blocker_bindings
                solve_state = solve_state.filtering_clauses(
                    lambda c: not _clause_is_stale(c, blocker.branch_line, bpath, target_exec,
                                                   by_line, loops))
                requires = solve_state.requires(blocker_vars)

                # Severed-parameter gate. A guard parameter is "severed" when the path
                # OVERWROTE it from something independent before the guard reads it
                # (`a = b`): the guard now sees the reassigned local, so changing the entry
                # parameter cannot flip it. If EVERY parameter the guard names is severed,
                # the per-guard candidate lanes (which can only set the guard's own
                # parameters) are doomed — skip straight to the conjunction and the
                # all-parameter reach-search.
                #
                # Gated on ``heavy``: the reach-search is the safety net for the one case a
                # per-guard candidate could STILL win — when the guard is not nested under
                # the reassignment, so setting the entry parameter reroutes control AROUND
                # it. That net only runs when ``heavy`` is set, so without it we take no
                # shortcut and run every lane, exactly as before.
                def _severed(v: str) -> bool:
                    assigned = False
                    for la, bound in by_line.items():
                        if la in target_exec and fstart < la < blocker.branch_line and v in bound:
                            assigned = True
                            if v in _self_rhs.get(la, frozenset()):
                                return False     # a += 1 / a = f(a): dependence survives
                    return assigned

                guard_params = blocker_vars & params
                severed_all = (heavy and bool(guard_params)
                               and all(_severed(v) for v in guard_params))
            else:
                blocker_node, blocker_vars, requires = None, set(), "True"
                severed_all = False

            # An isinstance/type test is the one guard whose answer legitimately changes a
            # parameter's type, so the type check below must not veto it.
            _asks_about_type = blocker_node is not None and any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id in ("isinstance", "type", "issubclass")
                for n in ast.walk(blocker_node)
            )

            def _variants(got: dict[str, Any] | None) -> list[dict[str, Any]]:
                """A candidate, plus its type-preserving twin — whichever executes wins.

                Ordered type-preserving-first: keeping the running type is what lets the
                guards after this one stay solvable.
                """
                if not got:
                    return [got] if got is not None else []
                same = _type_preserving(inputs, got)
                return [same, got] if same is not None else [got]

            def _ok(got: dict[str, Any] | None) -> bool:
                # Only structural sanity here. This used to also veto any candidate that
                # touched a variable an outer, already-satisfied guard reads — a backstop
                # from before execution-verification existed. It is now both redundant and
                # WRONG: an outer `len(c) >= 3` marks `c` locked, so a perfectly good patch
                # to `c[0]` was thrown away and the driver reported "no applicable input".
                # _flip_verifies re-runs the program and requires the blocker's target side
                # to execute, which can only happen if every outer guard still holds — so
                # breaking the prefix is caught by observation rather than by guesswork.
                if not got:
                    return False
                # One thing observation does NOT catch: a candidate that flips this guard by
                # changing a parameter's TYPE. `p[0] == 0x7F` is answered with `p = []` by the
                # subscript rule, which flips nothing and destroys the bytes every other guard
                # was solved against. Reject it here so the portfolio escalates to minleaf,
                # which mutates the live value in place. A guard that genuinely asks about the
                # type says so, and is exempt.
                if _asks_about_type:
                    return True
                return not any(k in inputs and inputs[k] is not None
                               and not isinstance(got[k], type(inputs[k]))
                               and not isinstance(inputs[k], type(got[k]))
                               for k in got)

            # ---- Portfolio: try everything, verify, escalate lazily ----
            # Cheap tier first (snapshot / static); gather every applicable
            # candidate, then accept the first one whose flip *actually reaches* the target
            # side when executed.  A wrong candidate — or one behind a false 'infeasible'
            # vote — is discarded, not committed.
            proposal: dict[str, Any] | None = None
            technique = None
            infeasible_votes = 0
            for flip in (flippers if blocker is not None else []):
                got, tech, infeasible = flip(
                    blocker_path, blocker.branch_line, blocker.desired_branch, snapshot, params, requires
                )
                if infeasible:
                    infeasible_votes += 1
                    continue
                # When every guard parameter is severed, a candidate that only sets those
                # parameters cannot flip the guard. The conjunction and reach-search
                # lanes handle parameters that the local was derived from.
                if severed_all and got and set(got) <= (blocker_vars & params):
                    continue
                for cand in _variants(got):
                    if _ok(cand) and _verified(blocker, cand, blocker_path):
                        proposal, technique = cand, tech
                        break
                if proposal is not None:
                    break

            # [internal ref] PR2: the preimage tier -- a guard over a local DERIVED from an entry parameter through str /
            # format / len / set / re operations is propagated back to the parameter (solve/preimage).  Cheap and
            # deterministic, so it runs before minleaf and the searches; every proposal is execution-verified.
            if proposal is None and blocker is not None:
                used_rules: list = []
                for cand in _flip_preimage(source_path, func_name, blocker_tree, blocker, blocker_path, params,
                                           inputs, snapshot, priors=preimage_priors, rules_out=used_rules):
                    if _ok(cand) and _verified(blocker, cand, blocker_path):
                        proposal, technique = cand, "preimage"
                        _publish_preimage_hints(_STORE, used_rules, blocker_path, blocker.branch_line,
                                                f"the {blocker.desired_branch} branch of the guard", cand)
                        break

            # Minleaf tier — concrete mutation of the live state.  Runs before the search
            # tier because it needs no modeling at all (it evaluates the real guard), which
            # is exactly where the SMT lane abstains, and it returns a minimal patch.
            # Skipped when every guard parameter is severed: minleaf mutates the guard's
            # own leaves, which the reassignment already made irrelevant to the entry.
            if proposal is None and minleaf and real_snapshot and blocker is not None \
                    and not severed_all:
                got, tech = _flip_minleaf(blocker_path, blocker.branch_line,
                                          blocker.desired_branch, real_snapshot, params)
                if _ok(got) and _verified(blocker, got, blocker_path):
                    proposal, technique = got, tech

            # Avengers tier — construct an object, for the case minleaf just fell through
            # on: a guard parameter with NO live value at all, so nothing to mutate.
            # Deliberately NOT gated on ``real_snapshot`` being non-empty (unlike every
            # tier above it): that is exactly the case this tier exists for -- a snapshot
            # is empty precisely when nothing meaningful was ever observed, which is
            # minleaf's own stated blind spot (``docs/DESIGN.md``), not a reason to skip.
            # [internal ref] slice 3 ([internal ref]): no boolean gate (owner ruling, no feature
            # flags) -- runs on demand only, exactly when a guard needs an object and
            # nothing cheaper found one, bounded by ``avengers_budget_seconds`` per guard.
            if proposal is None and blocker is not None and not severed_all:
                # Target objects stay guarded throughout the round. Only the
                # trusted worker-launch phase runs outside this context.
                target_phase.close()
                try:
                    got, tech = _flip_avengers(blocker_path, blocker.branch_line,
                                               blocker.desired_branch, real_snapshot, params,
                                               budget_seconds=avengers_budget_seconds,
                                               **({"confine": confine} if confine is not None else {}))
                finally:
                    target_phase.enter_context((confine or nullcontext)())
                if _ok(got) and _verified(blocker, got, blocker_path):
                    proposal, technique = got, tech

            # Heavy tier (property-based search) — only if nothing above found a flip.
            if proposal is None and heavy and blocker is not None:
                # Facts first: solving the conjunction cannot contradict an earlier answer,
                # which is the failure mode the per-guard flippers above run into. The
                # committed clauses are handed over FILTERED (a reassignment on the path
                # invalidates a fact for this search exactly as it does for the SMT lane).
                committed_texts = [
                    c.text for c in state.filtering_clauses(
                        lambda c: not _clause_is_stale(c, blocker.branch_line, bpath,
                                                       target_exec, by_line, loops)
                    ).clauses.clauses
                ]
                live = {**inputs, **{k: v for k, v in snapshot.items() if k in params}}
                used_stochastic = True
                got, tech = _flip_hypothesis_facts(
                    committed_texts, _guard_fact(blocker_tree, blocker.branch_line,
                                                 blocker.desired_branch), live, params)
                if _ok(got) and _verified(blocker, got, blocker_path):
                    proposal, technique = got, tech

            # Per-guard property search over the guard's own parameters. Doomed when they
            # are all severed — the reach-search below covers the derived-from parameters.
            if proposal is None and heavy and blocker is not None and not severed_all:
                used_stochastic = True
                for hflip in _HEAVY_FLIPPERS:
                    got, tech = hflip(blocker_path, blocker.branch_line, blocker.desired_branch,
                                      func_name, params)
                    if _ok(got) and _verified(blocker, got, blocker_path):
                        proposal, technique = got, tech
                        break

            # Only conclude dead-code if a *sound* procedure voted infeasible and nothing —
            # cheap or heavy — could produce a verifying flip.
            if proposal is None and infeasible_votes > 0 and blocker is not None:
                pruned.add((blocker.branch_line, blocker.desired_branch))
                history.append(DriveStep(rnd, dict(inputs), blocker.branch_line, blocker.desired_branch,
                                         "infeasible", "certified infeasible, unverified by any solver"))
                continue

            if proposal is None:
                # The lexical frame gave nothing. The real blocker may be in a CALLER, which
                # enclosing_guards cannot see. Try guards that ran without entering their
                # body, accepting any flip that strictly increases progress toward the target.
                # For a cross-file target, "progress" is measured in the CALLEE (how far
                # into the target file we got); the guards to flip are in the entry file.
                fallback: tuple[dict[str, Any], str, GuardFrame] | None = None
                base_progress = (len(target_exec) if tp != rp
                                 else _progress(outcome.executed, target_line))
                probe_line = target_line if tp == rp else max(outcome.executed, default=target_line)
                for cand in _call_chain_blockers(tree, outcome.executed, probe_line):
                    if (cand.branch_line, cand.desired_branch) in pruned:
                        continue
                    c_snaps = outcome.snapshots.get(cand.branch_line) or []
                    c_snap = c_snaps[-1] if c_snaps else {}
                    c_vars = _free_names_of(_find_branch_node(tree, cand.branch_line))
                    c_req = _bounds_requires(param_bounds, c_vars) or "True"
                    for flip in flippers:
                        got, tech, infeas = flip(source_path, cand.branch_line,
                                                 cand.desired_branch, c_snap, params, c_req)
                        if infeas or not got:
                            continue
                        same = _type_preserving(inputs, got)
                        got = same if same is not None else got
                        trial = {**inputs, **got}
                        oc2 = _run_traced(source_path, func_name, trial, snapshot_lines,
                                          capture_real=minleaf,
                                          extra_paths=extra_paths, extra_snapshot_lines=extra_snaps)
                        if not oc2.ok:
                            continue
                        oc2_target = (oc2.by_path.get(tp, set()) if tp != rp else oc2.executed)
                        advanced = (len(oc2_target) > base_progress if tp != rp
                                    else _progress(oc2.executed, target_line) > base_progress)
                        if (target_line in oc2_target or advanced) and _verified(
                                cand, got, source_path):
                            proposal, technique = got, f"{tech}@L{cand.branch_line}"
                            # Attribute the proposal to the guard it actually flipped.
                            # Keeping the lexical callee blocker here records (for
                            # example) ``c*c == 900`` after a proposal that only made
                            # ``a == 100`` true.  Besides making the history wrong, that
                            # poisons the committed path state: caller guards disappear
                            # and a later nonlinear solve may choose a model that breaks
                            # them.  The real caller guard is a durable path constraint.
                            blocker = cand
                            blocker_path = source_path
                            blocker_tree = tree
                            break
                        # No measurable progress — but flipping the blocking guard can still
                        # be the only move available. In an if/elif chain on flags, leaving
                        # the first arm lands on `return "none"`: no closer to the callee by
                        # any line count, yet nothing else makes the second arm reachable.
                        # Kept as a FALLBACK, never a peer of progress: promoting it to equal
                        # standing let the driver wander and cost ntlm_auth two lines.
                        if fallback is None and _verified(cand, got, source_path):
                            fallback = (got, f"{tech}@L{cand.branch_line}(flip-only)", cand)
                    if proposal is not None:
                        break
                if proposal is None and fallback is not None:
                    proposal, technique, blocker = fallback
                    blocker_path = source_path
                    blocker_tree = tree

            if proposal is None and blocker is not None and trail and backtracks < max_backtracks:
                # Backtrack: this guard is unsatisfiable under the assignment an EARLIER
                # guard chose. Bank what it demands as a lasting fact, restore the inputs AND
                # the state to before that flip (so the earlier guard's clause does not coexist
                # with its own negation), and re-solve the earlier guard under the stricter
                # fact set. The cycle check still applies — if the earlier guard can only
                # answer the same way, the next pass plateaus rather than spinning.
                conflict = _guard_fact(blocker_tree, blocker.branch_line, blocker.desired_branch)
                prior_inputs, prior_state, prior_guard, prior_rp = trail[-1]
                # But a conflict is only worth banking at the restored guard if it still holds
                # there. If a variable it names was reassigned on the path between that guard
                # and this blocker, the conflict describes a value that no longer exists there —
                # banking it would assert a false fact (Witness B, through the conflict lane).
                # Then do NOT backtrack (and do not spend the budget): fall through to the
                # reach-search, which searches the parameters the guard's local derived FROM.
                conflict_names = _text_names(conflict) if conflict else set()
                restored_line = prior_guard.branch_line if prior_guard is not None else fstart
                untransportable = bool(conflict) and (prior_rp == bpath) and _killed_between(
                    by_line, loops, target_exec, restored_line, blocker.branch_line, conflict_names)
                already = False
                if conflict:
                    from tharkun.lattice import Clause as _Clause
                    already = any(c.text == _Clause(conflict).text for c in state.clauses.clauses)
                if conflict and not untransportable and not already:
                    trail.pop()
                    inputs, state = prior_inputs, prior_state
                    node = _find_branch_node(blocker_tree, blocker.branch_line)
                    end = getattr(node, "end_lineno", blocker.branch_line) or blocker.branch_line
                    state = state.with_clause(conflict, Origin(bpath, blocker.branch_line, end,
                                                               blocker.desired_branch), node=node)
                    backtracks += 1
                    history.append(DriveStep(
                        rnd, dict(inputs), blocker.branch_line, blocker.desired_branch,
                        note=f"unsatisfiable under the current assignment; backtracking over "
                             f"L{prior_guard.branch_line if prior_guard else -1} with {conflict}"))
                    continue

            # Backward propagation (call_mapper): a guard on a callee PARAMETER bound to a
            # simple entry-parameter name is solved directly — propagate the guard's own
            # solution to the entry parameter — instead of blind-searching for it. Tried
            # before the reach-search because it is exact and cheap when it applies; a
            # non-simple argument (slice / int.from_bytes) is left to the search.
            if proposal is None and blocker is not None:
                back = _backward_propagate(source_path, func_name, blocker_tree, blocker,
                                           blocker_path, params, blocker.desired_branch)
                for cand in _variants(back):
                    if _ok(cand) and _verified(blocker, cand, blocker_path):
                        proposal, technique = cand, "call_mapper"
                        break

            if proposal is None and blocker is not None and heavy:
                # A search, so it belongs to the heavy tier and honours `heavy=False`.
                # Nothing could flip the guard on its own terms. Use banked operands and
                # static candidates to aim a search over the entry parameters.
                # Execution verifies reaches without inverting the callee dataflow.
                hints: list[Any] = []
                # Operands an EARLIER search observed at this guard. A failed search still
                # ran the program hundreds of times and saw what the comparison demanded;
                # that value exists only at runtime, so no static source can supply it.
                # Aiming the next search with it turns a failed round into progress.
                if _STORE is not None:
                    for pname in sorted(params):
                        # Operands an earlier search observed (the runtime inversion) AND
                        # the concrete values static inference proposed for this parameter
                        # — both aim the search. The static candidates were banked and
                        # read by nobody until now (C4).
                        for kind in ("operand", "candidate"):
                            for f in _STORE.get(kind, pname):
                                if f.value not in hints:
                                    hints.append(f.value)
                # The literals of the path's OWN history: every constant in a committed
                # clause is a value the path has already had to satisfy, so it belongs in
                # the search alphabet even though it names no parameter of THIS guard. This
                # is what makes a reach after a reassignment near-certain rather than a
                # lucky joint draw — `100` from a satisfied `a > 100` seeds the pool that
                # `b == 7` must also be found in.
                for lit_group in _fact_literals([c.text for c in state.clauses.clauses]):
                    for lit in lit_group:
                        if lit not in hints:
                            hints.append(lit)
                used_stochastic = True
                got = _reach_by_search(source_path, func_name, target_line, tp, inputs,
                                       params, hints, store=_STORE,
                                       guard_path=blocker_path,
                                       guard_line=blocker.branch_line)
                if got:
                    proposal, technique = got, "reach-search"

            if proposal is None:
                bl = blocker.branch_line if blocker is not None else -1
                bd = blocker.desired_branch if blocker is not None else ""
                history.append(DriveStep(rnd, dict(inputs), bl, bd,
                                         note="no flipper produced an applicable (parameter) input"))
                return DriveResult(
                    PLATEAU, inputs, rounds=rnd + 1, target_line=target_line, history=history, used_stochastic=used_stochastic,
                    diagnostics=(f"guard at line {bl} depends on non-parameter state; "
                                 f"backward propagation (call_mapper) needed") if blocker is not None
                                else f"no caller-side guard advanced execution toward {tp}:{target_line}",
                )

            new_inputs = {**inputs, **proposal}
            key = tuple(sorted((k, repr(v)) for k, v in new_inputs.items()))
            if key in seen:
                history.append(DriveStep(rnd, dict(inputs), blocker.branch_line if blocker else -1,
                                         blocker.desired_branch if blocker else "",
                                         technique, "proposal repeats a prior state"))
                return DriveResult(
                    PLATEAU, inputs, rounds=rnd + 1, target_line=target_line, history=history, used_stochastic=used_stochastic,
                    diagnostics=f"flip of line {blocker.branch_line if blocker else -1} did not change "
                                f"the reachable state (cycle) — likely a local-derived guard",
                )
            seen.add(key)
            # The trail carries the PRE-flip state, so abandoning this fork restores it.
            trail.append((dict(inputs), state, blocker, os.path.realpath(blocker_path)))
            # The flip verified, so this guard now holds on the path we are on. MEET it into
            # the state, anchored to the guard's program point (drop-then-add so a
            # re-established clause takes the fresh origin the flow filter keys on). A
            # caller-side flip found by the dynamic search has no lexical frame, so nothing
            # is banked for it — the same as before.
            if blocker is not None:
                fact_text = _guard_fact(blocker_tree, blocker.branch_line, blocker.desired_branch)
                if fact_text:
                    fnode = _find_branch_node(blocker_tree, blocker.branch_line)
                    fend = getattr(fnode, "end_lineno", blocker.branch_line) or blocker.branch_line
                    state = state.with_fresh_clause(
                        fact_text, Origin(os.path.realpath(blocker_path), blocker.branch_line,
                                          fend, blocker.desired_branch), node=fnode)
            history.append(DriveStep(rnd, dict(new_inputs),
                                     blocker.branch_line if blocker else -1,
                                     blocker.desired_branch if blocker else "",
                                     technique, "flipped"))
            inputs = new_inputs

        return DriveResult(
            BUDGET, inputs, rounds=max_rounds, target_line=target_line, history=history,
            diagnostics=f"target not reached within {max_rounds} rounds",
        )
    finally:
        target_phase.close()
        globals()['_STORE'] = _prev_store
