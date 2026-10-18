"""Phase 8: Patch search -- find a minimal-cost patch that flips the guard.

Uses BFS over k-combinations of leaf paths, with retrace-based expansion.
"""

from __future__ import annotations

import ast
import itertools
import time
from collections import deque
from typing import Any

from tharkun.minleaf.atom_deps import atom_dependencies
from tharkun.minleaf.candidates import (
    propose_combined_candidates,
    propose_leaf_candidates,
)
from tharkun.solve.guard_eval import trace_guard
from tharkun.minleaf.types import (
    LeafFactsIndex,
    Patch,
    Path,
    SetOp,
    StateIndex,
)

# Type alias
WorkItem = tuple[tuple[Path, Any], ...]

PRODUCT_CAP = 1000


def _tried_key(work_item) -> tuple:
    """Hashable identity for a work item.

    A candidate value can be an unhashable container (replacing a whole list or dict
    is how a ``len()`` or truthiness guard gets satisfied), so fall back to ``repr``
    for anything that will not hash.
    """
    out = []
    for path, value in work_item:
        try:
            hash(value)
            out.append((path, value))
        except TypeError:
            out.append((path, repr(value)))
    return tuple(out)


def find_minleaf_patch(
    *,
    guard_expr: ast.expr,
    desired_value: bool,
    env: dict[str, object],
    state_index: StateIndex,
    blocking_atoms: tuple[tuple[str, ast.expr], ...],
    leaf_facts_index: LeafFactsIndex | None,
    max_leaf_changes: int,
    max_candidates_per_leaf: int,
    retrace_max_rounds: int = 2,
    budget_ms: int,
) -> tuple[Patch | None, int, int, int, int, int]:
    """Find a minimal-cost patch that flips the guard.

    Returns (patch, tried_count, depth_reached, retrace_rounds_used,
             facts_injected, facts_pruned).
    """
    start = time.monotonic()

    def _budget_expired() -> bool:
        return (time.monotonic() - start) * 1000 >= budget_ms

    # Step 0: Global infeasibility check
    if leaf_facts_index is not None and not leaf_facts_index.global_possible:
        return (None, 0, 0, 0, 0, 0)

    # Compile guard once
    expr_wrapper = ast.Expression(body=guard_expr)
    ast.fix_missing_locations(expr_wrapper)
    guard_code = compile(expr_wrapper, "<guard>", "eval")
    baseline_value = bool(eval(guard_code, env))

    # Initial trace for atom values
    initial_trace = trace_guard(guard_expr, env, desired_value)

    # Step 1: Compute initial repair surface
    leaf_set: set[Path] = set()
    container_set: set[Path] = set()
    structural_blocker_ids: list[str] = []
    atom_dep_cache: dict[str, Any] = {}

    for atom_id, atom_ast in blocking_atoms:
        deps = atom_dependencies(atom_ast, state_index)
        atom_dep_cache[atom_id] = deps
        leaf_set |= deps.leaf_paths
        container_set |= deps.container_paths
        if deps.structural:
            structural_blocker_ids.append(atom_id)

    # Also include leaves from non-blocking atoms (skipped by short-circuit)
    # since they will need to be satisfied for the guard to flip.
    # Only do this when we have blocking atoms (guard needs fixing).
    if blocking_atoms:
        for aid, anode in initial_trace.atom_nodes.items():
            if aid not in atom_dep_cache:
                deps = atom_dependencies(anode, state_index)
                atom_dep_cache[aid] = deps
                leaf_set |= deps.leaf_paths

    S = sorted(leaf_set)
    C = sorted(container_set)

    if not S:
        return (None, 0, 0, 0, 0, 0)

    # Step 2: Rank S
    # For each path p in S, compute rank_key
    def _rank_key(p: Path) -> tuple:
        has_fact_eq = (leaf_facts_index is not None and leaf_facts_index.has_eq(p))
        has_direct = _has_direct_candidate(p, blocking_atoms, state_index, env)
        atom_count = sum(
            1 for aid, _ in blocking_atoms
            if p in atom_dep_cache[aid].leaf_paths
        )
        return (
            -int(has_fact_eq),
            -int(has_direct),
            -atom_count,
            p,
        )

    S.sort(key=_rank_key)

    # Step 3: Initialize
    # Keyed by a hashable projection: a candidate value may itself be a container
    # (a whole-list/dict replacement, which satisfies a len() or truthiness guard),
    # and those are unhashable.
    tried_global: set[tuple] = set()
    total_tried = 0
    max_depth_reached = 0
    total_retrace_used = 0
    global_retrace_cap = retrace_max_rounds * 50
    facts_injected_total = 0
    facts_pruned_total = 0

    # Step 4: Iterate k=1 to max_leaf_changes
    for k in range(1, max_leaf_changes + 1):
        max_depth_reached = k
        for T in itertools.combinations(S, k):
            if _budget_expired():
                return (None, total_tried, max_depth_reached,
                        total_retrace_used, facts_injected_total,
                        facts_pruned_total)

            # Build initial candidate pool for this subset
            candidates_for: dict[Path, list[Any]] = {}
            for p in T:
                pool: list[Any] = []
                # Candidates from blocking atoms
                for atom_id, atom_ast in blocking_atoms:
                    adeps = atom_dep_cache[atom_id]
                    if p in adeps.leaf_paths:
                        lc = propose_leaf_candidates(
                            atom_expr=atom_ast,
                            atom_value=initial_trace.trace.atoms[atom_id].value,
                            dependency_paths=adeps.leaf_paths,
                            state_index=state_index,
                            env=env,
                            leaf_facts_index=leaf_facts_index,
                            max_per_leaf=max_candidates_per_leaf,
                        )
                        if p in lc:
                            pool.extend(lc[p].candidates)
                            facts_injected_total += lc[p].facts_injected
                            facts_pruned_total += lc[p].facts_pruned
                # Candidates from non-blocking atoms (short-circuited)
                for aid, anode in initial_trace.atom_nodes.items():
                    if aid not in set(a[0] for a in blocking_atoms):
                        adeps = atom_dep_cache.get(aid)
                        if adeps and p in adeps.leaf_paths:
                            lc = propose_leaf_candidates(
                                atom_expr=anode,
                                atom_value=False,  # not yet evaluated, treat as False
                                dependency_paths=adeps.leaf_paths,
                                state_index=state_index,
                                env=env,
                                leaf_facts_index=leaf_facts_index,
                                max_per_leaf=max_candidates_per_leaf,
                            )
                            if p in lc:
                                pool.extend(lc[p].candidates)
                                facts_injected_total += lc[p].facts_injected
                                facts_pruned_total += lc[p].facts_pruned

                # E1: Combined constraint candidates
                # Use ALL atoms in the guard (not just blocking) since
                # short-circuit may hide constraints that share the same leaf.
                atom_exprs_for_p: list[tuple[ast.expr, bool]] = []
                for atom_id, atom_ast in blocking_atoms:
                    if p in atom_dep_cache[atom_id].leaf_paths:
                        atom_exprs_for_p.append(
                            (atom_ast, initial_trace.trace.atoms[atom_id].value)
                        )
                # Also include non-blocking atoms from the full guard
                for aid, anode in initial_trace.atom_nodes.items():
                    if aid not in set(a[0] for a in blocking_atoms):
                        adeps = atom_dependencies(anode, state_index)
                        if p in adeps.leaf_paths:
                            # Treat as "want True" (False value) since the
                            # overall guard needs all And-children True
                            atom_exprs_for_p.append((anode, False))
                if len(atom_exprs_for_p) > 1:
                    combined = propose_combined_candidates(
                        atom_exprs=atom_exprs_for_p,
                        target_path=p,
                        state_index=state_index,
                        env=env,
                    )
                    pool.extend(combined)

                # Stable dedup
                seen: set = set()
                deduped: list = []
                for v in pool:
                    try:
                        key = (type(v), v)
                        if key not in seen:
                            seen.add(key)
                            deduped.append(v)
                    except TypeError:
                        deduped.append(v)
                candidates_for[p] = deduped[:max_candidates_per_leaf]

            # Skip if any path has no candidates
            if any(len(candidates_for.get(p, [])) == 0 for p in T):
                continue

            # Build initial worklist
            worklist: deque[WorkItem] = deque()
            count = 0
            for combo in itertools.product(*(candidates_for[p] for p in T)):
                wi = tuple(sorted(zip(T, combo), key=lambda x: x[0]))
                if _tried_key(wi) not in tried_global:
                    worklist.append(wi)
                count += 1
                if len(worklist) >= PRODUCT_CAP:
                    break

            while worklist:
                if _budget_expired():
                    return (None, total_tried, max_depth_reached,
                            total_retrace_used, facts_injected_total,
                            facts_pruned_total)

                work_item = worklist.popleft()
                if _tried_key(work_item) in tried_global:
                    continue
                tried_global.add(_tried_key(work_item))
                total_tried += 1

                assignment = dict(work_item)
                saved = {p: state_index.leaves[p].getter() for p in T}

                try:
                    # Apply patch
                    for p in T:
                        state_index.leaves[p].setter(assignment[p])

                    v = bool(eval(guard_code, env))

                    if v == desired_value:
                        # SUCCESS
                        ops = tuple(sorted(
                            (SetOp(p, old=saved[p], new=assignment[p])
                             for p in T),
                            key=lambda op: op.path,
                        ))
                        return (Patch(ops=ops), total_tried,
                                max_depth_reached, total_retrace_used,
                                facts_injected_total, facts_pruned_total)

                    # Per-assignment re-tracing
                    should_retrace = (v != baseline_value)
                    if should_retrace and total_retrace_used < global_retrace_cap:
                        total_retrace_used += 1

                        traced2 = trace_guard(guard_expr, env, desired_value)
                        new_blocker_ids = traced2.trace.blocking_atom_ids

                        initial_ids = set(
                            aid for aid, _ in blocking_atoms
                        )
                        if set(new_blocker_ids) == initial_ids:
                            continue  # no new info

                        new_all_deps = [
                            atom_dependencies(
                                traced2.atom_nodes[aid], state_index
                            )
                            for aid in new_blocker_ids
                        ]
                        new_leaf_set = frozenset().union(
                            *(d.leaf_paths for d in new_all_deps)
                        )

                        if new_leaf_set - frozenset(T):
                            # Needs leaves outside T -> PRUNE this subset
                            # But add the new leaves to S for future k iterations
                            for new_p in new_leaf_set - frozenset(T):
                                if new_p not in leaf_set:
                                    leaf_set.add(new_p)
                                    S.append(new_p)
                            break

                        if new_leaf_set <= frozenset(T):
                            # Expand: generate new WorkItems from per-atom cands
                            for aid in new_blocker_ids:
                                adeps = atom_dependencies(
                                    traced2.atom_nodes[aid], state_index
                                )
                                new_cands = propose_leaf_candidates(
                                    atom_expr=traced2.atom_nodes[aid],
                                    atom_value=traced2.trace.atoms[aid].value,
                                    dependency_paths=adeps.leaf_paths,
                                    state_index=state_index,
                                    env=env,
                                    leaf_facts_index=leaf_facts_index,
                                    max_per_leaf=max_candidates_per_leaf,
                                )
                                for p in T:
                                    if p in new_cands:
                                        for cv in new_cands[p].candidates:
                                            expanded = tuple(sorted(
                                                ((q, cv if q == p else assignment[q])
                                                 for q in T),
                                                key=lambda x: x[0],
                                            ))
                                            if _tried_key(expanded) not in tried_global:
                                                worklist.append(expanded)

                            # E1: Combined constraint candidates from
                            # original + new blocking atoms for shared leaves
                            all_retrace_atoms = [
                                (traced2.atom_nodes[aid],
                                 traced2.trace.atoms[aid].value)
                                for aid in new_blocker_ids
                            ]
                            # Also include original blocking atom constraints
                            for orig_aid, orig_ast in blocking_atoms:
                                if orig_aid not in set(new_blocker_ids):
                                    av = initial_trace.trace.atoms[orig_aid].value
                                    all_retrace_atoms.append((orig_ast, av))

                            if len(all_retrace_atoms) > 1:
                                for p in T:
                                    combined = propose_combined_candidates(
                                        atom_exprs=all_retrace_atoms,
                                        target_path=p,
                                        state_index=state_index,
                                        env=env,
                                    )
                                    for cv in combined:
                                        expanded = tuple(sorted(
                                            ((q, cv if q == p else assignment[q])
                                             for q in T),
                                            key=lambda x: x[0],
                                        ))
                                        if _tried_key(expanded) not in tried_global:
                                            worklist.append(expanded)

                except Exception:
                    pass  # eval failure = candidate failure, continue
                finally:
                    # ALWAYS restore
                    for p in T:
                        state_index.leaves[p].setter(saved[p])

    return (None, total_tried, max_depth_reached, total_retrace_used,
            facts_injected_total, facts_pruned_total)


def _has_direct_candidate(
    p: Path,
    blocking_atoms: tuple[tuple[str, ast.expr], ...],
    state_index: StateIndex,
    env: dict[str, object],
) -> bool:
    """Check if path p has a direct pattern candidate from a Compare atom."""
    for aid, atom_ast in blocking_atoms:
        if isinstance(atom_ast, ast.Compare) and len(atom_ast.ops) == 1:
            # Check if this atom actually depends on path p
            deps = atom_dependencies(atom_ast, state_index)
            if p in deps.leaf_paths:
                return True
    return False
