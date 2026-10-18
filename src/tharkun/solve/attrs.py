"""Attribute tracer for ``self.xxx`` fuzz blocker resolution.

When a guard checks ``self.allow_unicode`` or ``self.md.references``, the
fuzzer controls input bytes, not object state.  This module bridges the gap:
it finds where attributes are assigned, traces their values at runtime, and
identifies which inputs produce attribute values that satisfy the guard.

Usage::

    from tharkun.solve.attrs import extract_attr_targets, find_attr_assignments

    targets = extract_attr_targets(guard_ast)
    for t in targets:
        assignments = find_attr_assignments(source_path, t)
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tharkun.solve.condition import _unparse


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttrTarget:
    """An attribute chain extracted from a guard expression.

    Attributes:
        base:  Root variable (``"self"`` or ``"cls"``).
        chain: Attribute path, e.g. ``("allow_unicode",)`` or ``("md", "references")``.
        full:  Full dotted path, e.g. ``"self.allow_unicode"``.
    """

    base: str
    chain: tuple[str, ...]
    full: str


@dataclass(frozen=True)
class AttrAssignment:
    """A source location where the target attribute is assigned.

    Attributes:
        file:        Source file path.
        lineno:      Line number of the assignment.
        rhs_source:  Source code of the right-hand side.
        method:      Qualname of the enclosing method.
        is_init:     True if the assignment is in ``__init__`` or ``__new__``.
    """

    file: str
    lineno: int
    rhs_source: str
    method: str
    is_init: bool


@dataclass(frozen=True)
class AttrObservation:
    """One runtime observation: input → attribute value."""

    input_repr: str
    attr_value: Any
    attr_type: str
    satisfies_guard: bool


@dataclass
class AttrTraceResult:
    """Complete result of attribute tracing for one blocker."""

    target: AttrTarget
    guard_source: str
    assignments: list[AttrAssignment] = field(default_factory=list)
    observations: list[AttrObservation] = field(default_factory=list)
    solving_input: dict[str, Any] | None = None
    strategy: str = "none"  # "observed" | "static" | "none"


# ---------------------------------------------------------------------------
# Step 1: Extract attribute targets from guard expression
# ---------------------------------------------------------------------------


def extract_attr_targets(expr: ast.expr) -> list[AttrTarget]:
    """Parse a guard expression to find ``self.xxx`` attribute chains.

    For ``self.allow_unicode and self.md.references``, returns::

        [
            AttrTarget(base="self", chain=("allow_unicode",), full="self.allow_unicode"),
            AttrTarget(base="self", chain=("md", "references"), full="self.md.references"),
        ]
    """
    raw: list[AttrTarget] = []
    seen: set[str] = set()

    for node in ast.walk(expr):
        if not isinstance(node, ast.Attribute):
            continue

        # Walk the attribute chain to find the root.
        chain: list[str] = [node.attr]
        cur = node.value
        while isinstance(cur, ast.Attribute):
            chain.append(cur.attr)
            cur = cur.value

        if isinstance(cur, ast.Name) and cur.id in ("self", "cls"):
            chain.reverse()
            full = f"{cur.id}.{'.'.join(chain)}"
            if full not in seen:
                seen.add(full)
                raw.append(AttrTarget(
                    base=cur.id,
                    chain=tuple(chain),
                    full=full,
                ))

    # Deduplicate: remove shorter chains that are prefixes of longer ones.
    # e.g. if we have self.md AND self.md.references, keep only the longer.
    targets: list[AttrTarget] = []
    fulls = {t.full for t in raw}
    for t in raw:
        is_prefix = any(
            other != t.full and other.startswith(t.full + ".")
            for other in fulls
        )
        if not is_prefix:
            targets.append(t)

    return targets


# ---------------------------------------------------------------------------
# Step 2: Find attribute assignments in source
# ---------------------------------------------------------------------------


def find_attr_assignments(
    source_path: str,
    target: AttrTarget,
) -> list[AttrAssignment]:
    """Find all assignments to *target* in the source file.

    Scans for patterns like:
    - ``self.attr = expr``
    - ``self.attr = expr`` inside ``__init__``
    """
    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []

    # The attribute we're looking for: self.chain[0] for simple attrs,
    # or self.chain[0] for the first level of nested attrs.
    attr_name = target.chain[0]
    assignments: list[AttrAssignment] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        method_name = node.name
        # Determine if this is __init__
        is_init = method_name in ("__init__", "__new__")

        # Find the enclosing class qualname.
        # Walk tree.body for class defs that contain this method.
        cls_name = ""
        for top in tree.body:
            if isinstance(top, ast.ClassDef):
                for child in top.body:
                    if child is node:
                        cls_name = top.name
                        break

        qualname = f"{cls_name}.{method_name}" if cls_name else method_name

        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if _matches_attr_target(tgt, target):
                        rhs = _unparse(stmt.value) or "<unknown>"
                        assignments.append(AttrAssignment(
                            file=rp,
                            lineno=getattr(stmt, "lineno", 0),
                            rhs_source=rhs,
                            method=qualname,
                            is_init=is_init,
                        ))
            elif isinstance(stmt, ast.AugAssign):
                if _matches_attr_target(stmt.target, target):
                    rhs = _unparse(stmt.value) or "<unknown>"
                    assignments.append(AttrAssignment(
                        file=rp,
                        lineno=getattr(stmt, "lineno", 0),
                        rhs_source=f"<augmented> {rhs}",
                        method=qualname,
                        is_init=is_init,
                    ))

    return assignments


def _matches_attr_target(node: ast.AST, target: AttrTarget) -> bool:
    """Check if an assignment target matches the AttrTarget."""
    if not isinstance(node, ast.Attribute):
        return False

    # Simple case: self.attr_name
    if len(target.chain) == 1:
        return (isinstance(node.value, ast.Name)
                and node.value.id == target.base
                and node.attr == target.chain[0])

    # Nested: self.a.b → check chain
    chain: list[str] = [node.attr]
    cur = node.value
    while isinstance(cur, ast.Attribute):
        chain.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name) and cur.id == target.base:
        chain.reverse()
        return tuple(chain) == target.chain

    return False


# ---------------------------------------------------------------------------
# Step 3: Analyze assignments statically
# ---------------------------------------------------------------------------


def analyze_attr_blocker(
    source_path: str,
    guard_expr: ast.expr,
) -> list[AttrTraceResult]:
    """Analyze attribute blockers in a guard expression.

    For each ``self.xxx`` in the guard, finds where the attribute is assigned
    and returns an ``AttrTraceResult`` with the assignment locations.

    This is the static-only analysis; runtime tracing would extend this
    with actual ``AttrObservation`` data from execution.
    """
    targets = extract_attr_targets(guard_expr)
    results: list[AttrTraceResult] = []

    guard_source = _unparse(guard_expr) or ""

    for target in targets:
        assignments = find_attr_assignments(source_path, target)
        strategy = "static" if assignments else "none"

        results.append(AttrTraceResult(
            target=target,
            guard_source=guard_source,
            assignments=assignments,
            strategy=strategy,
        ))

    return results


