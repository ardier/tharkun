"""Phase 9: API orchestration -- the public entry point.

Full orchestration of guard extraction, normalization, tracing,
state projection, fact conversion, dependency analysis, and patch search.
"""

from __future__ import annotations

import time
from typing import Any

from tharkun.minleaf.types import FactEnv

from tharkun.minleaf.atom_deps import atom_dependencies
from tharkun.solve.guard import (
    desired_guard_value,
    extract_frontier,
)
from tharkun.solve.normalize import normalize_guard
from tharkun.solve.guard_eval import trace_guard
from tharkun.minleaf.leaf_facts import build_leaf_facts_index
from tharkun.minleaf.patch_search import find_minleaf_patch
from tharkun.minleaf.state_project import build_state_index
from tharkun.minleaf.types import (
    Patch,
    SolveDiagnostics,
    SolveResult,
)

_PRIMITIVE_TYPES = (int, bool, float, str, bytes, type(None))


def solve_minleaf_patch_for_guard(
    *,
    source_path: str,
    target_line: int,
    desired_branch: str,              # "then" | "else"
    env: dict[str, object],           # evaluation environment
    roots: dict[str, object],         # named roots for projection
    facts: FactEnv | None = None,     # external analysis hints
    bytes_policy: str = "per_byte",   # "per_byte" | "atomic"
    max_leaf_changes: int = 3,        # BFS depth limit
    max_candidates_per_leaf: int = 8,
    retrace_max_rounds: int = 2,      # per-assignment retrace bound
    budget_ms: int = 2000,
    debug: bool = False,
) -> SolveResult:
    """Public API: solve for a minimal leaf patch to flip a guard branch."""
    start = time.monotonic()

    def _elapsed_ms() -> int:
        return int((time.monotonic() - start) * 1000)

    # Step 1: Validate env-root consistency (before any evaluation)
    for name, obj in roots.items():
        if name in env:
            if isinstance(obj, _PRIMITIVE_TYPES):
                if env[name] != roots[name]:
                    raise ValueError(
                        f"Primitive root {name!r}: env value {env[name]!r} "
                        f"differs from roots value {roots[name]!r}"
                    )
            else:
                if env[name] is not roots[name]:
                    raise ValueError(
                        f"Root {name!r}: env and roots must reference "
                        f"the same object (identity)"
                    )

    # Step 2: Extract guard
    frontier = extract_frontier(source_path, target_line)
    desired_value = desired_guard_value(frontier.stmt_kind, desired_branch)

    # Step 3: Normalize
    normalized = normalize_guard(frontier.guard_expr)

    # Step 4: Trace
    traced = trace_guard(normalized, env, desired_value)

    # Step 5: Already satisfied?
    if traced.trace.guard_value == desired_value:
        diags = SolveDiagnostics(
            guard_src=frontier.guard_src,
            guard_value_before=traced.trace.guard_value,
            trace=traced.trace,
            blocking_atoms=(),
            dependency_leaf_paths=(),
            dependency_container_paths=(),
            structural_blockers=(),
            facts_used=(facts is not None),
            facts_injected_candidates=0,
            facts_pruned_candidates=0,
            tried_patches=0,
            depth_reached=0,
            retrace_rounds_used=0,
            failure_reason=None,
        )
        return SolveResult(
            success=True,
            patch=Patch(ops=()),
            diagnostics=diags,
        )

    # Step 6: Project state
    state_index = build_state_index(roots, bytes_policy, env=env)

    # Step 7: Build leaf facts index
    leaf_facts_index = build_leaf_facts_index(facts, state_index)

    # Step 8: Compute deps, detect structural blockers
    blocking_atoms = tuple(
        (aid, traced.atom_nodes[aid])
        for aid in traced.trace.blocking_atom_ids
    )
    all_deps = [
        atom_dependencies(node, state_index)
        for _, node in blocking_atoms
    ]
    all_leaf_paths = sorted(
        set().union(*(d.leaf_paths for d in all_deps)) if all_deps else set()
    )
    all_container_paths = sorted(
        set().union(*(d.container_paths for d in all_deps)) if all_deps else set()
    )
    structural_ids = tuple(
        aid
        for (aid, _), d in zip(blocking_atoms, all_deps)
        if d.structural
    )

    # Step 9: Early-fail if no leaf deps
    if not all_leaf_paths:
        reason = "No patchable leaf dependencies."
        if all_container_paths:
            reason += " Blocking atoms depend on container structure (N6)."
        diags = _build_diags(
            frontier=frontier,
            traced=traced,
            tried=0,
            depth=0,
            retrace=0,
            all_leaf_paths=all_leaf_paths,
            all_container_paths=all_container_paths,
            structural_ids=structural_ids,
            facts_used=(facts is not None),
            facts_injected_candidates=0,
            facts_pruned_candidates=0,
            failure_reason=reason,
        )
        return SolveResult(success=False, patch=None, diagnostics=diags)

    # Step 10: Search
    remaining_ms = budget_ms - _elapsed_ms()
    patch, tried, depth, retrace, fi, fp = find_minleaf_patch(
        guard_expr=normalized,
        desired_value=desired_value,
        env=env,
        state_index=state_index,
        blocking_atoms=blocking_atoms,
        leaf_facts_index=leaf_facts_index,
        max_leaf_changes=max_leaf_changes,
        max_candidates_per_leaf=max_candidates_per_leaf,
        retrace_max_rounds=retrace_max_rounds,
        budget_ms=max(0, remaining_ms),
    )

    # Step 11: Assemble result
    diags = _build_diags(
        frontier=frontier,
        traced=traced,
        tried=tried,
        depth=depth,
        retrace=retrace,
        all_leaf_paths=all_leaf_paths,
        all_container_paths=all_container_paths,
        structural_ids=structural_ids,
        facts_used=(facts is not None),
        facts_injected_candidates=fi,
        facts_pruned_candidates=fp,
        failure_reason=None if patch else "Budget/depth exhausted",
    )
    return SolveResult(
        success=patch is not None,
        patch=patch,
        diagnostics=diags,
    )


def _build_diags(
    *,
    frontier,
    traced,
    tried: int,
    depth: int,
    retrace: int,
    all_leaf_paths,
    all_container_paths,
    structural_ids,
    facts_used: bool,
    facts_injected_candidates: int,
    facts_pruned_candidates: int,
    failure_reason: str | None,
) -> SolveDiagnostics:
    blocking_atom_evals = tuple(
        traced.trace.atoms[aid]
        for aid in traced.trace.blocking_atom_ids
        if aid in traced.trace.atoms
    )
    return SolveDiagnostics(
        guard_src=frontier.guard_src,
        guard_value_before=traced.trace.guard_value,
        trace=traced.trace,
        blocking_atoms=blocking_atom_evals,
        dependency_leaf_paths=tuple(all_leaf_paths),
        dependency_container_paths=tuple(all_container_paths),
        structural_blockers=structural_ids,
        facts_used=facts_used,
        facts_injected_candidates=facts_injected_candidates,
        facts_pruned_candidates=facts_pruned_candidates,
        tried_patches=tried,
        depth_reached=depth,
        retrace_rounds_used=retrace,
        failure_reason=failure_reason,
    )
