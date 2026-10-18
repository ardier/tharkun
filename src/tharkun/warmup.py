"""Build a static type prior for entry functions and their direct callees."""

from __future__ import annotations

import ast
import os
from pathlib import Path

from tharkun.knowledge import KnowledgeBase, program_point
from tharkun.lattice import TOP, FactState


def _guards(tree: ast.AST) -> list[ast.stmt]:
    """Every ``if`` / ``while`` in *tree* — the points a prior is worth having."""
    return [n for n in ast.walk(tree) if isinstance(n, (ast.If, ast.While))]


def _static_state(path: str, guard_line: int) -> FactState:
    """Infer types without treating candidate values as exhaustive domains.

    Union types across guard facets; intersecting their alternative candidate values
    could falsely mark a live edge as unreachable."""
    from tharkun.solve.condition import solve_condition

    types_by_var: dict[str, set[str]] = {}
    try:
        sol = solve_condition(path, guard_line)
    except Exception:
        return TOP
    for vc in getattr(sol, "variables", None) or []:
        if vc.inferred_type and vc.inferred_type not in ("Any", "object"):
            types_by_var.setdefault(vc.name, set()).add(vc.inferred_type)

    state = TOP
    for name, tys in types_by_var.items():
        state = state.with_types(name, tys)
    return state


def build_prior(kb: KnowledgeBase, module_path: str, func_name: str) -> int:
    """Record static types at each guard; return the number of points priced."""
    rp = os.path.realpath(module_path)
    try:
        tree = ast.parse(Path(rp).read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return 0

    fn = next((n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func_name),
              None)
    if fn is None:
        return 0

    priced = 0
    for guard in _guards(fn):
        state = _static_state(rp, guard.lineno)
        if not state.is_top():
            kb.learn_prior(program_point(rp, guard.lineno), state)
            priced += 1
    return priced


def warm_targets(kb: KnowledgeBase, targets) -> int:
    """Warm every ``(module_path, func_name)`` in *targets*. Returns points priced.

    *targets* is what ``harness_map`` recovered — the functions the harness calls — plus,
    ideally, their direct callees. The caller decides the breadth; this just prices what it
    is handed.
    """
    total = 0
    for module_path, func_name in targets:
        total += build_prior(kb, module_path, func_name)
    return total
