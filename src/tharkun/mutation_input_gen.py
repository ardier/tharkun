"""Mutation-guided input generation — targeted seeds to kill surviving mutants.

Takes surviving mutants from ``feedback.py`` routing, extracts context from
mutation sites (what the mutation changed), and generates seed inputs
specifically designed to kill those mutants.

Pipeline::

    feedback.classify_and_route_survivors()
        → router.json (surviving mutants + routing)
    mutation_input_gen.generate_killing_seeds()
        → {callable_id: [seed_inputs]}
    concolic/proptest stages use these seeds as starting points
"""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tharkun.solve.condition import (
    _extract_comparison_values,
    _free_names,
    _try_intersect_int_constraints,
    _unparse,
    infer_var_type,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MutationContext:
    """Context extracted from a mutation site.

    Attributes:
        job_id:          Cosmic-ray job ID.
        callable_id:     The function containing the mutation.
        qualname:        Human-readable function name.
        operator_name:   Mutation operator (e.g. ``"NumberReplacer"``).
        mutant_kind:     Classified kind (``"boundary"``, ``"arithmetic"``, etc.).
        route:           Technique to route to (``"hypothesis"``, etc.).
        lineno:          Line number of the mutation.
        original_source: Source line in original code.
        guard_expr:      If mutation is in a guard, the guard expression AST.
    """

    job_id: str
    callable_id: str
    qualname: str
    operator_name: str
    mutant_kind: str
    route: str
    lineno: int
    original_source: str | None = None
    guard_expr: ast.expr | None = None


@dataclass
class SeedInput:
    """A generated seed input targeting a specific surviving mutant.

    Attributes:
        callable_id: Target function.
        kwargs:      Keyword arguments for the function call.
        source:      How the seed was generated.
        targets:     List of job_ids this seed is designed to kill.
    """

    callable_id: str
    kwargs: dict[str, Any]
    source: str  # "boundary_probe", "negation_flip", "arithmetic_zero", etc.
    targets: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Mutation context extraction
# ---------------------------------------------------------------------------


def _load_router_survivors(outp: Path) -> list[dict[str, Any]]:
    """Load routed survivors from ``mutants/router.json``."""
    router_path = outp / "mutants" / "router.json"
    if not router_path.is_file():
        return []
    try:
        obj = json.loads(router_path.read_text(encoding="utf-8"))
        return obj.get("rows", [])
    except Exception:
        return []


def _load_analysis_rows(outp: Path) -> dict[str, dict[str, Any]]:
    """Load mutation analysis rows, keyed by job_id."""
    analysis_path = outp / "mutants" / "mutants_analysis.json"
    if not analysis_path.is_file():
        return {}
    try:
        obj = json.loads(analysis_path.read_text(encoding="utf-8"))
        return {str(row.get("job_id", "")): row for row in obj.get("rows", []) if isinstance(row, dict)}
    except Exception:
        return {}


def _extract_guard_at_line(source_path: str, lineno: int) -> ast.expr | None:
    """Extract the guard expression at *lineno* if it's a branch."""
    try:
        source = Path(os.path.realpath(source_path)).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if getattr(node, "lineno", None) != lineno:
            continue
        if isinstance(node, (ast.If, ast.While)):
            return node.test
        if isinstance(node, ast.Compare):
            return node
    return None


def _extract_source_line(source_path: str, lineno: int) -> str | None:
    """Read a single source line."""
    try:
        lines = Path(os.path.realpath(source_path)).read_text(encoding="utf-8", errors="ignore").splitlines()
        if 1 <= lineno <= len(lines):
            return lines[lineno - 1].strip()
    except (OSError, IOError):
        pass
    return None


def extract_mutation_contexts(outp: Path) -> list[MutationContext]:
    """Extract mutation contexts from router + analysis data."""
    survivors = _load_router_survivors(outp)
    analysis = _load_analysis_rows(outp)

    contexts: list[MutationContext] = []
    for surv in survivors:
        job_id = str(surv.get("job_id", ""))
        row = analysis.get(job_id, {})

        start_pos = row.get("start_pos", [0, 0])
        lineno = int(start_pos[0]) if isinstance(start_pos, (list, tuple)) and start_pos else 0

        module_path = str(row.get("module_path_abs", row.get("module_path", "")))
        guard_expr = _extract_guard_at_line(module_path, lineno) if lineno > 0 else None
        original_source = _extract_source_line(module_path, lineno) if lineno > 0 else None

        contexts.append(MutationContext(
            job_id=job_id,
            callable_id=str(surv.get("primary_callable_id", "")),
            qualname=str(surv.get("primary_qualname", "")),
            operator_name=str(surv.get("operator_name", "")),
            mutant_kind=str(surv.get("mutant_kind", "unknown")),
            route=str(surv.get("route", "")),
            lineno=lineno,
            original_source=original_source,
            guard_expr=guard_expr,
        ))

    return contexts


# ---------------------------------------------------------------------------
# Seed generation strategies per mutant kind
# ---------------------------------------------------------------------------


def _boundary_seeds(ctx: MutationContext) -> list[SeedInput]:
    """Generate boundary-probing inputs for boundary/comparison mutants.

    For ``x >= 18`` mutated to ``x > 18``, the killing input is ``x = 18``.
    """
    seeds: list[SeedInput] = []
    if ctx.guard_expr is None:
        return seeds

    free = _free_names(ctx.guard_expr)
    builtins_filter = {"len", "isinstance", "type", "range", "int", "str", "float", "bool", "True", "False", "None"}
    var_names = sorted(free - builtins_filter)

    for name in var_names:
        # Extract comparison values — these are the boundary points.
        values = _extract_comparison_values(name, ctx.guard_expr)
        if not values:
            intersected = _try_intersect_int_constraints(name, ctx.guard_expr)
            if intersected:
                values = intersected

        # For each boundary value, probe around it: v-1, v, v+1
        boundary_vals: list[Any] = []
        for v in values:
            if isinstance(v, int):
                boundary_vals.extend([v - 1, v, v + 1])
            elif isinstance(v, float):
                boundary_vals.extend([v - 0.01, v, v + 0.01])
            elif isinstance(v, str):
                boundary_vals.extend([v, v + "x", v[:-1] if v else ""])

        for bv in boundary_vals:
            seeds.append(SeedInput(
                callable_id=ctx.callable_id,
                kwargs={name: bv},
                source="boundary_probe",
                targets=[ctx.job_id],
            ))

    return seeds


def _arithmetic_seeds(ctx: MutationContext) -> list[SeedInput]:
    """Generate inputs for arithmetic mutants (+ → -, * → /, etc.).

    Key insight: arithmetic mutants are killed when the two operands
    have specific relationships — e.g., ``a + b != a - b`` requires ``b != 0``.
    """
    seeds: list[SeedInput] = []
    if ctx.guard_expr is None:
        return seeds

    free = _free_names(ctx.guard_expr)
    builtins_filter = {"len", "isinstance", "type", "range", "int", "str", "float", "bool", "True", "False", "None"}
    var_names = sorted(free - builtins_filter)

    for name in var_names:
        t, _ = infer_var_type(name, ctx.guard_expr, "", 0)
        if t == "int":
            # Key values that distinguish arithmetic operators:
            # 0 (additive identity), 1 (multiplicative identity), -1, 2
            for v in [0, 1, -1, 2, 10]:
                seeds.append(SeedInput(
                    callable_id=ctx.callable_id,
                    kwargs={name: v},
                    source="arithmetic_identity",
                    targets=[ctx.job_id],
                ))
        elif t == "float":
            for v in [0.0, 1.0, -1.0, 0.5, 2.0]:
                seeds.append(SeedInput(
                    callable_id=ctx.callable_id,
                    kwargs={name: v},
                    source="arithmetic_identity",
                    targets=[ctx.job_id],
                ))

    return seeds


def _negation_seeds(ctx: MutationContext) -> list[SeedInput]:
    """Generate inputs for negation mutants (``not x`` ↔ ``x``).

    Need both truthy and falsy values for the negated variable.
    """
    seeds: list[SeedInput] = []
    if ctx.guard_expr is None:
        return seeds

    free = _free_names(ctx.guard_expr)
    builtins_filter = {"len", "isinstance", "type", "range", "int", "str", "float", "bool", "True", "False", "None"}
    var_names = sorted(free - builtins_filter)

    for name in var_names:
        t, _ = infer_var_type(name, ctx.guard_expr, "", 0)
        # Generate both truthy and falsy values.
        if t == "int":
            seeds.extend([
                SeedInput(callable_id=ctx.callable_id, kwargs={name: 0}, source="negation_falsy", targets=[ctx.job_id]),
                SeedInput(callable_id=ctx.callable_id, kwargs={name: 1}, source="negation_truthy", targets=[ctx.job_id]),
            ])
        elif t == "str":
            seeds.extend([
                SeedInput(callable_id=ctx.callable_id, kwargs={name: ""}, source="negation_falsy", targets=[ctx.job_id]),
                SeedInput(callable_id=ctx.callable_id, kwargs={name: "x"}, source="negation_truthy", targets=[ctx.job_id]),
            ])
        elif t == "bool":
            seeds.extend([
                SeedInput(callable_id=ctx.callable_id, kwargs={name: True}, source="negation_truthy", targets=[ctx.job_id]),
                SeedInput(callable_id=ctx.callable_id, kwargs={name: False}, source="negation_falsy", targets=[ctx.job_id]),
            ])
        else:
            seeds.extend([
                SeedInput(callable_id=ctx.callable_id, kwargs={name: None}, source="negation_falsy", targets=[ctx.job_id]),
                SeedInput(callable_id=ctx.callable_id, kwargs={name: 1}, source="negation_truthy", targets=[ctx.job_id]),
            ])

    return seeds


def _return_value_seeds(ctx: MutationContext) -> list[SeedInput]:
    """Generate inputs for return-value mutants.

    Return value mutants change what a function returns. To kill them,
    we need inputs that make the original and mutant return different values.
    This often means exercising different code paths.
    """
    seeds: list[SeedInput] = []
    if ctx.guard_expr is None:
        return seeds

    # Use the condition solver to find satisfying inputs for the guard.
    free = _free_names(ctx.guard_expr)
    builtins_filter = {"len", "isinstance", "type", "range", "int", "str", "float", "bool", "True", "False", "None"}
    var_names = sorted(free - builtins_filter)

    for name in var_names:
        values = _extract_comparison_values(name, ctx.guard_expr)
        if values:
            seeds.append(SeedInput(
                callable_id=ctx.callable_id,
                kwargs={name: values[0]},
                source="return_value_path",
                targets=[ctx.job_id],
            ))

    return seeds


_STRATEGY_MAP = {
    "boundary": _boundary_seeds,
    "constant_replacement": _boundary_seeds,
    "arithmetic": _arithmetic_seeds,
    "negation": _negation_seeds,
    "return_value": _return_value_seeds,
    "exception_removal": _negation_seeds,  # similar logic: truthy/falsy
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_killing_seeds(
    outp: Path,
) -> dict[str, list[SeedInput]]:
    """Generate seed inputs designed to kill surviving mutants.

    Reads ``mutants/router.json`` and ``mutants/mutants_analysis.json``
    from *outp*, then generates targeted inputs per callable.

    Returns ``{callable_id: [SeedInput, ...]}``.
    """
    contexts = extract_mutation_contexts(outp)
    seeds_by_callable: dict[str, list[SeedInput]] = {}

    for ctx in contexts:
        strategy = _STRATEGY_MAP.get(ctx.mutant_kind)
        if strategy is None:
            continue
        seeds = strategy(ctx)
        if seeds:
            seeds_by_callable.setdefault(ctx.callable_id, []).extend(seeds)

    # Deduplicate per callable (same kwargs may appear from multiple mutants).
    for cid in seeds_by_callable:
        seen: set[str] = set()
        deduped: list[SeedInput] = []
        for s in seeds_by_callable[cid]:
            key = repr(sorted(s.kwargs.items()))
            if key not in seen:
                seen.add(key)
                deduped.append(s)
        seeds_by_callable[cid] = deduped

    return seeds_by_callable


def serialize_seeds(seeds: dict[str, list[SeedInput]]) -> dict[str, Any]:
    """Serialize seeds for JSON output."""
    return {
        "summary": {
            "callables": len(seeds),
            "total_seeds": sum(len(s) for s in seeds.values()),
        },
        "by_callable": {
            cid: [
                {
                    "kwargs": s.kwargs,
                    "source": s.source,
                    "targets": s.targets,
                }
                for s in seed_list
            ]
            for cid, seed_list in sorted(seeds.items())
        },
    }


