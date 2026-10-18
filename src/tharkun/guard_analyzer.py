"""Guard blocker analysis — combines static decomposition with runtime snapshots.

Orchestrates:
1. ``guard_decomposer.decompose_guard()`` → identifies blocker sub-conditions
2. Execution with ``TracePolicy(snapshot_lines=...)`` → captures ``f_locals`` at blocker lines
3. Snapshot analysis → determines why the branch is not taken

Usage::

    from tharkun.guard_analyzer import analyze_blocker

    result = analyze_blocker("module.py", 42)
    for diag in result.diagnostics:
        print(diag.expression, diag.suggestion)
"""

from __future__ import annotations

import ast
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from tharkun.solve.condition import (
    _free_names,
    _unparse,
    infer_var_type,
)
from tharkun.solve.constants import BUILTIN_NAMES
from tharkun.solve.decompose import (
    DecompNode,
    DecompResult,
    decompose_guard,
)
from tharkun.trace_api import TracePolicy, TraceResult, Tracer, _safe_snapshot


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class BlockerDiagnostic:
    """Diagnostic for one blocking sub-condition.

    Attributes:
        expression:       Source of the blocking condition.
        blocker_reason:   From decomposer (``"unresolved_function"``, etc.).
        line:             Source line number (if known).
        snapshot:         Captured ``f_locals`` at the blocker line (if executed).
        relevant_vars:    Subset of snapshot variables relevant to the guard.
        missing_vars:     Free variables expected but not present in snapshot.
        wrong_type_vars:  Variables whose type doesn't match inference.
        suggestion:       Human-readable fix suggestion.
    """

    expression: str
    blocker_reason: str | None = None
    line: int | None = None
    snapshot: dict[str, Any] | None = None
    relevant_vars: dict[str, Any] | None = None
    missing_vars: list[str] = field(default_factory=list)
    wrong_type_vars: list[str] = field(default_factory=list)
    suggestion: str | None = None


@dataclass
class AnalysisResult:
    """Complete blocker analysis combining decomposition and runtime snapshots."""

    guard_source: str
    decomposition: DecompResult
    diagnostics: list[BlockerDiagnostic] = field(default_factory=list)
    snapshot_count: int = 0
    executed: bool = False


# ---------------------------------------------------------------------------
# Snapshot capture via in-process tracer
# ---------------------------------------------------------------------------


def _capture_snapshots_inprocess(
    source_path: str,
    snapshot_lines: set[int],
    *,
    func_name: str | None = None,
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    timeout_s: float = 2.0,
) -> dict[int, list[dict[str, Any]]]:
    """Execute *source_path* with tracing and capture snapshots at *snapshot_lines*.

    This is a lightweight in-process execution — it imports the module and
    optionally calls *func_name*.  Returns ``{lineno: [locals_dict, ...]}``.
    """
    rp = os.path.realpath(source_path)
    policy = TracePolicy(
        enabled=True,
        snapshot_lines={rp: snapshot_lines},
        paths={rp},
    )

    import importlib.util

    try:
        spec = importlib.util.spec_from_file_location("_tulkas_snap_target", rp)
        if spec is None or spec.loader is None:
            return {}
        mod = importlib.util.module_from_spec(spec)

        with Tracer(policy) as tr:
            spec.loader.exec_module(mod)
            if func_name:
                fn = getattr(mod, func_name, None)
                if callable(fn):
                    try:
                        fn(*(args or ()), **(kwargs or {}))
                    except Exception:
                        pass  # We want the snapshot even if execution fails.
    except Exception as e:
        logger.debug("snapshot capture failed for %s: %s", source_path, e)
        return {}

    return tr.result.snapshots.get(rp, {})


# ---------------------------------------------------------------------------
# Diagnostic generation
# ---------------------------------------------------------------------------


_SUGGESTION_MAP = {
    "unresolved_function": "function is not defined in this module — consider mocking it or providing a stub",
    "complex_return_logic": "function has complex control flow — simplify or add a return annotation",
    "max_depth_exceeded": "call chain too deep — increase max_depth or refactor",
    "recursive_call": "recursive function detected — cannot statically decompose",
    "unsolvable_leaf": "expression uses patterns the solver cannot handle",
}


def _build_diagnostic(
    blocker: DecompNode,
    snapshots: dict[int, list[dict[str, Any]]],
    source_path: str,
) -> BlockerDiagnostic:
    """Build a diagnostic for a single blocker node."""
    # Try to find the line number from the blocker expression.
    line = None
    # If the blocker has a resolved function, look for its definition line.
    # For now, we use the line from the AST if available.

    # Find snapshot for this line (if any).
    snapshot: dict[str, Any] | None = None
    relevant_vars: dict[str, Any] | None = None
    missing_vars: list[str] = []

    # Try to parse the expression to find free variables.
    free_var_names: set[str] = set()
    try:
        expr = ast.parse(blocker.expression, mode="eval").body
        free_var_names = _free_names(expr)
    except SyntaxError:
        pass

    # Check all snapshots for relevant variables.
    # Prefer the snapshot with the BEST overlap (most free variables matched).
    best_overlap_count = 0
    for ln, snaps in snapshots.items():
        if not snaps:
            continue
        for snap in snaps:
            overlap = free_var_names & set(snap.keys())
            if len(overlap) > best_overlap_count:
                best_overlap_count = len(overlap)
                snapshot = snap
                line = ln
                relevant_vars = {k: snap[k] for k in sorted(overlap)}
                missing_vars = sorted(free_var_names - set(snap.keys()))

    if not snapshot and free_var_names:
        missing_vars = sorted(free_var_names)

    suggestion = _SUGGESTION_MAP.get(blocker.blocker_reason or "")

    # Enhance suggestion with snapshot data.
    if relevant_vars and not suggestion:
        parts = []
        for k, v in relevant_vars.items():
            parts.append(f"{k} = {_format_snap_value(v)}")
        suggestion = f"at runtime: {'; '.join(parts)}"

    return BlockerDiagnostic(
        expression=blocker.expression,
        blocker_reason=blocker.blocker_reason,
        line=line,
        snapshot=snapshot,
        relevant_vars=relevant_vars,
        missing_vars=missing_vars,
        suggestion=suggestion,
    )


def _format_snap_value(v: Any) -> str:
    """Format a snapshot value for human display."""
    if v is None:
        return "None"
    if isinstance(v, (bool, int, float)):
        return repr(v)
    if isinstance(v, str):
        if len(v) > 50:
            return repr(v[:50]) + "..."
        return repr(v)
    if isinstance(v, dict) and "__type__" in v:
        t = v.get("__type__", "")
        if t == "bytes":
            return f"bytes(len={v.get('len', '?')})"
        if t == "dict":
            return f"dict(len={v.get('len', '?')}, keys={v.get('keys', [])})"
        if t == "type":
            return f"<class {v.get('name', '?')!r}>"
        return v.get("repr", repr(v))
    return repr(v)[:100]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_blocker(
    source_path: str,
    lineno: int,
    *,
    execute: bool = True,
    func_name: str | None = None,
    func_args: tuple[Any, ...] = (),
    func_kwargs: dict[str, Any] | None = None,
    timeout_s: float = 2.0,
    max_depth: int = 5,
) -> AnalysisResult:
    """Analyze a guard blocker: decompose + optionally execute with snapshots.

    Args:
        source_path: Path to the Python source file.
        lineno:      Line number of the guard (``if``/``while``/``for``).
        execute:     If True, actually run the code to capture runtime snapshots.
        func_name:   Optional function to call after importing the module.
        func_args:   Positional args for *func_name*.
        func_kwargs: Keyword args for *func_name*.
        timeout_s:   Execution timeout in seconds.
        max_depth:   Max decomposition recursion depth.

    Returns:
        An ``AnalysisResult`` with decomposition tree and blocker diagnostics.
    """
    decomp = decompose_guard(source_path, lineno, max_depth=max_depth)

    if not decomp.blockers and not (execute and func_name):
        return AnalysisResult(
            guard_source=decomp.guard_source,
            decomposition=decomp,
            executed=False,
        )

    # Collect all lines where we need snapshots.
    # Use the guard line itself + enclosing lines of blocker expressions.
    rp = os.path.realpath(source_path)
    snapshot_line_set: set[int] = {lineno}

    # Also snapshot lines around the guard to capture variable state.
    for i in range(max(1, lineno - 2), lineno + 3):
        snapshot_line_set.add(i)

    snapshots: dict[int, list[dict[str, Any]]] = {}
    did_execute = False

    if execute:
        try:
            snapshots = _capture_snapshots_inprocess(
                source_path,
                snapshot_line_set,
                func_name=func_name,
                args=func_args,
                kwargs=func_kwargs,
                timeout_s=timeout_s,
            )
            did_execute = True
        except Exception:
            pass

    diagnostics: list[BlockerDiagnostic] = []
    for blocker in decomp.blockers:
        diag = _build_diagnostic(blocker, snapshots, source_path)
        diagnostics.append(diag)

    snap_count = sum(len(snaps) for snaps in snapshots.values())

    analysis = AnalysisResult(
        guard_source=decomp.guard_source,
        decomposition=decomp,
        diagnostics=diagnostics,
        snapshot_count=snap_count,
        executed=did_execute,
    )

    # Feed snapshot values back to improve diagnostics.
    if did_execute and diagnostics:
        re_solve_with_snapshots(analysis)

    return analysis


# ---------------------------------------------------------------------------
# Snapshot → re-solve feedback loop
# ---------------------------------------------------------------------------


def re_solve_with_snapshots(
    result: AnalysisResult,
) -> AnalysisResult:
    """Feed snapshot variable values back into diagnostics.

    For each blocker diagnostic that has runtime snapshot data:
    1. Extract actual variable types from the snapshot
    2. Compare expected vs actual types
    3. Identify value mismatches (e.g. "x = 42, needs > 100")
    4. Generate improved suggestions

    Mutates ``result.diagnostics`` in place and returns the same result.
    """
    import ast as _ast

    for diag in result.diagnostics:
        if not diag.relevant_vars:
            continue

        # Try to parse the blocker expression and check each variable.
        try:
            expr = _ast.parse(diag.expression, mode="eval").body
        except SyntaxError:
            continue

        free = _free_names(expr)
        parts: list[str] = []

        for var_name in sorted(free):
            if var_name in BUILTIN_NAMES:
                continue
            snap_val = diag.relevant_vars.get(var_name)
            if snap_val is None:
                continue

            # Infer what the guard expects.
            expected_type, source = infer_var_type(
                var_name, expr, "", 0,
            )

            actual_type = _snap_type_name(snap_val)
            val_repr = _format_snap_value(snap_val)

            if expected_type != "Any" and actual_type != expected_type:
                parts.append(
                    f"{var_name}: expected {expected_type}, got {actual_type} ({val_repr})"
                )
                if var_name not in diag.wrong_type_vars:
                    diag.wrong_type_vars.append(var_name)
            else:
                parts.append(f"{var_name} = {val_repr} ({actual_type})")

        if parts:
            diag.suggestion = "runtime state: " + "; ".join(parts)

    return result




def _snap_type_name(v: Any) -> str:
    """Return the type name from a snapshot value."""
    if v is None:
        return "NoneType"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, dict) and "__type__" in v:
        return v.get("__type__", "object")
    return type(v).__name__


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def serialize_diagnostic(diag: BlockerDiagnostic) -> dict[str, Any]:
    d: dict[str, Any] = {
        "expression": diag.expression,
        "blocker_reason": diag.blocker_reason,
    }
    if diag.line is not None:
        d["line"] = diag.line
    if diag.relevant_vars:
        d["relevant_vars"] = {
            k: _format_snap_value(v) for k, v in diag.relevant_vars.items()
        }
    if diag.missing_vars:
        d["missing_vars"] = diag.missing_vars
    if diag.suggestion:
        d["suggestion"] = diag.suggestion
    return d


def serialize_analysis(result: AnalysisResult) -> dict[str, Any]:
    from tharkun.solve.decompose import serialize_decomp

    return {
        "guard_source": result.guard_source,
        "executed": result.executed,
        "snapshot_count": result.snapshot_count,
        "diagnostics": [serialize_diagnostic(d) for d in result.diagnostics],
        "decomposition": serialize_decomp(result.decomposition),
    }
