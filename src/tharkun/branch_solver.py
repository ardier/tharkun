"""Per-branch aggressive coverage solver.

For each uncovered branch, applies the FULL toolset in sequence:
1. Condition solver → try to crack the guard statically
2. Guard decomposer → trace into called functions, find sub-conditions
3. Runtime snapshot → capture f_locals at the branch line
4. Re-solve with snapshots → feed runtime values back
5. Attr tracer → if self.xxx, find where attrs are set
6. Mock gen → if external calls, generate @patch specs
7. Regex inversion → if re.match pattern, generate from regex
8. Mutation seeds → if surviving mutants at this branch, use targeted seeds

This is tulkas's strategic advantage over Pynguin: instead of evolutionary
search (DynaMOSA), we apply deep per-branch analysis using debugger-level
access.  If we cover branches aggressively and smartly, population-based
search becomes orthogonal.

Usage::

    from tharkun.branch_solver import solve_uncovered_branches

    results = solve_uncovered_branches(source_path, by_mod, coverage_summary)
    # returns per-branch solutions with generated inputs
"""

from __future__ import annotations

import ast
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

from tharkun.solve.condition import (
    ConditionSolution,
    VarConstraint,
    _build_input_constraints,
    _extract_comparison_values,
    _extract_regex_pattern,
    _free_names,
    _generate_from_regex,
    _unparse,
    infer_var_type,
    solve_condition,
)
from tharkun.solve.constants import BUILTIN_NAMES
from tharkun.solve.decompose import DecompResult, decompose_guard
from tharkun.guard_analyzer import (
    AnalysisResult,
    analyze_blocker,
    re_solve_with_snapshots,
)
from tharkun.solve.attrs import (
    AttrTraceResult,
    analyze_attr_blocker,
    extract_attr_targets,
)
from tharkun.mock_gen import (
    ExternalCall,
    MockPlan,
    PatchSpec,
    detect_externals,
    generate_patches,
)
from tharkun.solve.frontend.type_propagation import propagate_types
from tharkun.solve.frontend.constructor import construct_instance
from tharkun.solve.frontend.proxy import RecordingProxy, infer_type_from_usage


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchTarget:
    """One edge of a branch statement — a (line, polarity) pair.

    Each ``if``/``while``/``for`` produces two targets: one for the ``then``
    edge (guard truthy / iterator non-empty) and one for the ``else`` edge.
    """
    lineno: int
    stmt_kind: str      # "if" | "while" | "for"
    desired_branch: str  # "then" | "else"


@dataclass
class NarrowedConstraint:
    """Constraint narrowed by minleaf diagnostics for heavy solvers.

    When minleaf fails, it knows which atoms block and what leaf paths
    they depend on. This narrows the domain for Hypothesis.
    """
    blocking_exprs: list[str] = field(default_factory=list)  # AST-unparsed blocking atoms
    leaf_paths: list[tuple] = field(default_factory=list)     # dependent leaf paths
    var_types: dict[str, str] = field(default_factory=dict)   # var → inferred type
    var_bounds: dict[str, tuple] = field(default_factory=dict) # var → (lo, hi)
    structural_reason: str | None = None                       # why it's hard


@dataclass
class BranchAttempt:
    """One technique's attempt at solving a branch."""

    technique: str  # "condition_solver", "decomposer", "snapshot", "mock", "regex", "attr"
    success: bool
    inputs: dict[str, Any] | None = None
    patches: list[PatchSpec] | None = None
    detail: str | None = None
    narrowed: NarrowedConstraint | None = None  # from minleaf on failure


@dataclass
class BranchSolution:
    """Complete solution for one uncovered branch.

    Attributes:
        file:              Source file path.
        lineno:            Branch line number.
        desired_branch:    Which edge was targeted ("then" or "else").
        stmt_kind:         Statement kind ("if", "while", or "for").
        guard_source:      The guard expression source.
        callable_id:       Enclosing callable (if known).
        solved:            True if at least one technique produced inputs.
        attempts:          All technique attempts (successful or not).
        best_inputs:       Best satisfying inputs from successful attempts.
        mock_patches:      @patch specs needed for external calls.
        decomposition:     Guard decomposition tree (if computed).
        attr_blockers:     Attribute blocker analysis (if any self.xxx).
    """

    file: str
    lineno: int
    desired_branch: str = "then"  # "then" | "else"
    stmt_kind: str = "if"         # "if" | "while" | "for"
    guard_source: str = ""
    callable_id: str | None = None
    solved: bool = False
    attempts: list[BranchAttempt] = field(default_factory=list)
    best_inputs: dict[str, Any] | None = None
    mock_patches: list[PatchSpec] = field(default_factory=list)
    decomposition: DecompResult | None = None
    attr_blockers: list[AttrTraceResult] = field(default_factory=list)
    blocker_info: dict | None = None                         # Structured blocker diagnostic for agents
    best_inputs_constraints: dict[str, dict] | None = None  # Structured constraints for the best inputs
    caller_inputs: dict[str, Any] | None = None              # best_inputs mapped back to caller variable names


@dataclass
class CoverageResult:
    """Result of aggressive branch solving across a module."""

    source_path: str
    total_branches: int = 0
    solved_branches: int = 0
    unsolved_branches: int = 0
    solutions: list[BranchSolution] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-branch solver
# ---------------------------------------------------------------------------


_STMT_KIND: dict[type, str] = {
    ast.If: "if",
    ast.While: "while",
    ast.For: "for",
}


def _find_branch_edges(source_path: str) -> list[BranchTarget]:
    """Return one BranchTarget per edge (then + else) per branch statement.

    Each ``if``/``while``/``for`` produces two targets.  Duplicate line
    numbers (e.g. from ``elif`` chains) are deduplicated; each unique
    (lineno, stmt_kind, desired_branch) triple appears once.
    """
    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []

    seen: set[tuple[int, str, str]] = set()
    targets: list[BranchTarget] = []
    for node in ast.walk(tree):
        kind = _STMT_KIND.get(type(node))
        if kind is None:
            continue
        ln = getattr(node, "lineno", None)
        if ln is None:
            continue
        for polarity in ("then", "else"):
            key = (ln, kind, polarity)
            if key not in seen:
                seen.add(key)
                targets.append(BranchTarget(ln, kind, polarity))

    return sorted(targets, key=lambda t: (t.lineno, t.desired_branch))


def _find_enclosing_callable(
    source_path: str,
    lineno: int,
) -> str | None:
    """Find the callable_id that encloses a line number."""
    try:
        source = Path(os.path.realpath(source_path)).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start + 1000)
            if start <= lineno <= end:
                return node.name
    return None


def solve_branch(
    source_path: str,
    lineno: int,
    *,
    covered_lines: set[int] | None = None,
    execute_snapshots: bool = True,
    max_depth: int = 5,
    facts: Any = None,
    desired_branch: str = "then",
    stmt_kind: str = "if",
    deterministic: bool = False,
) -> BranchSolution:
    """Apply the full toolset to solve one uncovered branch.

    Techniques are applied in priority order. Each technique that produces
    inputs is recorded.  The first technique that succeeds becomes the
    ``best_inputs``.

    Args:
        desired_branch: Which edge to target — ``"then"`` (guard truthy /
            iterator non-empty) or ``"else"`` (guard falsy / iterator empty).
        stmt_kind:      Statement kind detected by the caller (``"if"``,
            ``"while"``, or ``"for"``).
    """
    rp = os.path.realpath(source_path)
    sol = BranchSolution(
        file=rp, lineno=lineno,
        desired_branch=desired_branch, stmt_kind=stmt_kind,
    )
    sol.callable_id = _find_enclosing_callable(rp, lineno)

    # ---- Pre-step: Cross-function type propagation ----
    callable_info = None
    if sol.callable_id:
        try:
            propagated = propagate_types(rp, sol.callable_id)
            if propagated:
                from tharkun.solve.types import CallableInfo, ParamInfo, TypeHint
                params = [
                    ParamInfo(
                        name=name,
                        kind="arg",
                        type_hint=TypeHint(type_str, "propagated", 60),
                    )
                    for name, type_str in propagated.items()
                ]
                callable_info = CallableInfo(
                    module_path=rp,
                    module_key="",
                    callable_id=sol.callable_id,
                    qualname=sol.callable_id,
                    name=sol.callable_id,
                    kind="function",
                    params=params,
                )
        except Exception:
            pass

    # ---- Step 1: Condition solver (static, fast) ----
    _cond_sol_variables: list[VarConstraint] = []  # captured for best_inputs_constraints
    try:
        cond_sol = solve_condition(
            rp, lineno,
            callable_info=callable_info, facts=facts,
            desired_branch=desired_branch,
        )
        sol.guard_source = cond_sol.guard_source
        _cond_sol_variables = cond_sol.variables

        if cond_sol.satisfying_inputs:
            inputs = cond_sol.satisfying_inputs[0]
            # Verify: for non-"direct" strategies, check that the inputs
            # produce the desired edge outcome.
            verified = cond_sol.strategy == "direct"
            if not verified and cond_sol.guard_source:
                verified = _verify_inputs_satisfy_guard(
                    cond_sol.guard_source, inputs, desired_branch=desired_branch,
                )

            sol.attempts.append(BranchAttempt(
                technique="condition_solver",
                success=verified,
                inputs=inputs,
                detail=f"strategy={cond_sol.strategy}, confidence={cond_sol.confidence}"
                       f"{', UNVERIFIED' if not verified else ''}",
            ))
            if verified and not sol.best_inputs:
                sol.best_inputs = inputs
                sol.solved = True
        else:
            sol.attempts.append(BranchAttempt(
                technique="condition_solver",
                success=False,
                detail=f"strategy={cond_sol.strategy}, no satisfying inputs",
            ))
    except Exception as e:
        logger.debug("condition_solver failed for line %d: %s", lineno, e)
        sol.attempts.append(BranchAttempt(
            technique="condition_solver", success=False,
            detail=f"error: {e}",
        ))

    # ---- Step 2: Guard decomposer (traces into functions) ----
    try:
        decomp = decompose_guard(rp, lineno, max_depth=max_depth)
        sol.decomposition = decomp

        if decomp.solvable_leaves > 0 and decomp.unsolvable_leaves == 0:
            sol.attempts.append(BranchAttempt(
                technique="decomposer",
                success=True,
                detail=f"{decomp.solvable_leaves}/{decomp.total_leaves} solvable, 0 blockers",
            ))
        elif decomp.blockers:
            blocker_reasons = [b.blocker_reason for b in decomp.blockers]
            sol.attempts.append(BranchAttempt(
                technique="decomposer",
                success=False,
                detail=f"{decomp.unsolvable_leaves} blockers: {blocker_reasons}",
            ))
        else:
            sol.attempts.append(BranchAttempt(
                technique="decomposer",
                success=decomp.total_leaves > 0,
                detail=f"{decomp.solvable_leaves}/{decomp.total_leaves} solvable",
            ))
    except Exception as e:
        logger.debug("decomposer failed for line %d: %s", lineno, e)
        sol.attempts.append(BranchAttempt(
            technique="decomposer", success=False, detail=f"error: {e}",
        ))

    # ---- Step 3: Attr tracer (for self.xxx guards) ----
    guard_expr = None  # shared by steps 3 and 5
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if getattr(node, "lineno", None) == lineno and isinstance(node, (ast.If, ast.While)):
                guard_expr = node.test
                break

        if guard_expr:
            attr_results = analyze_attr_blocker(rp, guard_expr)
            sol.attr_blockers = attr_results
            if attr_results:
                has_assignments = any(r.assignments for r in attr_results)
                sol.attempts.append(BranchAttempt(
                    technique="attr_tracer",
                    success=has_assignments,
                    detail=f"{len(attr_results)} attr targets, assignments found: {has_assignments}",
                ))
    except Exception as e:
        logger.debug("attr_tracer failed for line %d: %s", lineno, e)
        sol.attempts.append(BranchAttempt(
            technique="attr_tracer", success=False, detail=f"error: {e}",
        ))

    # ---- Step 4: Mock detection (for external calls in the guard's callable) ----
    if sol.callable_id:
        try:
            externals = detect_externals(rp, sol.callable_id)
            if externals:
                patches = generate_patches(externals)
                sol.mock_patches = patches
                sol.attempts.append(BranchAttempt(
                    technique="mock_gen",
                    success=True,
                    patches=patches,
                    detail=f"{len(externals)} external calls detected",
                ))
        except Exception as e:
            logger.debug("mock_gen failed for line %d: %s", lineno, e)
            sol.attempts.append(BranchAttempt(
                technique="mock_gen", success=False, detail=f"error: {e}",
            ))

    # ---- Step 5: Regex inversion (for re.match guards) ----
    if guard_expr is not None:
        try:
            free = _free_names(guard_expr)
            var_names = sorted(free - BUILTIN_NAMES)
            for name in var_names:
                pat = _extract_regex_pattern(name, guard_expr)
                if pat:
                    val = _generate_from_regex(pat)
                    if val is not None:
                        sol.attempts.append(BranchAttempt(
                            technique="regex_inversion",
                            success=True,
                            inputs={name: val},
                            detail=f"pattern={pat!r}, generated={val!r}",
                        ))
                        if not sol.best_inputs:
                            sol.best_inputs = {name: val}
                            sol.solved = True
                        break
        except Exception:
            pass

    # ---- Step 6: Runtime snapshot + snapshot-informed re-solve ----
    if execute_snapshots and not sol.solved:
        try:
            # Build seed kwargs from partial solutions so the tracer
            # actually reaches the branch line inside the function.
            seed_kwargs = _build_seed_kwargs(sol, rp, callable_info)
            analysis = analyze_blocker(
                rp, lineno,
                execute=True,
                func_name=sol.callable_id,
                func_kwargs=seed_kwargs,
                timeout_s=2.0,
                max_depth=max_depth,
            )
            if analysis.executed and analysis.snapshot_count > 0:
                sol.attempts.append(BranchAttempt(
                    technique="runtime_snapshot",
                    success=True,
                    detail=f"{analysis.snapshot_count} snapshots captured",
                ))

                # Collect all snapshot f_locals at/near the guard line
                # and feed them back into the condition solver.
                snapshot_locals = _collect_snapshot_locals(analysis)

                if snapshot_locals:
                    from tharkun.solve.condition import solve_with_snapshot
                    snap_sol = solve_with_snapshot(
                        rp, lineno, snapshot_locals,
                        callable_info=callable_info,
                    )
                    if snap_sol.satisfying_inputs:
                        sol.attempts.append(BranchAttempt(
                            technique="snapshot_re_solve",
                            success=True,
                            inputs=snap_sol.satisfying_inputs[0],
                            detail=f"strategy={snap_sol.strategy}, "
                                   f"confidence={snap_sol.confidence}",
                        ))
                        if not sol.best_inputs:
                            sol.best_inputs = snap_sol.satisfying_inputs[0]
                            sol.solved = True
                    else:
                        sol.attempts.append(BranchAttempt(
                            technique="snapshot_re_solve",
                            success=False,
                            detail=f"snapshot had {len(snapshot_locals)} vars "
                                   f"but re-solve produced no inputs",
                        ))

                # Also record diagnostic info from blockers
                for diag in analysis.diagnostics:
                    if diag.relevant_vars and not sol.solved:
                        sol.attempts.append(BranchAttempt(
                            technique="snapshot_diagnostic",
                            success=False,
                            inputs=dict(diag.relevant_vars),
                            detail=diag.suggestion,
                        ))
            else:
                sol.attempts.append(BranchAttempt(
                    technique="runtime_snapshot",
                    success=False,
                    detail="no snapshots captured (branch not reached)",
                ))
        except Exception as e:
            logger.debug("runtime_snapshot failed for line %d: %s", lineno, e)
            sol.attempts.append(BranchAttempt(
                technique="runtime_snapshot", success=False, detail=f"error: {e}",
            ))

    # ---- Step 7: State sequence solver (for self.xxx guards) ----
    if not sol.solved and sol.attr_blockers:
        try:
            sol = _solve_state_guard(sol, rp)
        except Exception as e:
            logger.debug("state_sequence failed for line %d: %s", lineno, e)
            sol.attempts.append(BranchAttempt(
                technique="state_sequence", success=False,
                detail=f"error: {e}",
            ))

    # ---- Step 8: Minleaf runtime leaf-patch solver ----
    # Build FactEnv from condition_solver's VarConstraints (step 1 output)
    minleaf_facts = _build_fact_env_from_attempts(sol)
    minleaf_narrowed: NarrowedConstraint | None = None
    if not sol.solved and execute_snapshots and sol.callable_id:
        try:
            minleaf_inputs, minleaf_narrowed = _try_minleaf_solve(
                rp, lineno, sol, facts=minleaf_facts
            )
            if minleaf_inputs:
                sol.attempts.append(BranchAttempt(
                    technique="minleaf",
                    success=True,
                    inputs=minleaf_inputs,
                    detail="minimum leaf-patch via minleaf",
                ))
                if not sol.best_inputs:
                    sol.best_inputs = minleaf_inputs
                    sol.solved = True
            else:
                detail = "no patch found"
                if minleaf_narrowed and minleaf_narrowed.blocking_exprs:
                    detail += f"; blocking: {minleaf_narrowed.blocking_exprs[:3]}"
                sol.attempts.append(BranchAttempt(
                    technique="minleaf",
                    success=False,
                    detail=detail,
                    narrowed=minleaf_narrowed,
                ))
        except Exception as e:
            logger.debug("minleaf failed for line %d: %s", lineno, e)
            sol.attempts.append(BranchAttempt(
                technique="minleaf", success=False,
                detail=f"error: {e}",
            ))

    # ---- Step 9: Hypothesis property-based search (optional) ----
    if not sol.solved and sol.callable_id:
        try:
            from tharkun.hypothesis_adapter import hypothesis_solve_branch
            hyp_result = hypothesis_solve_branch(
                rp, lineno, sol.callable_id,
                narrowed=minleaf_narrowed,
                desired_branch=desired_branch,
                derandom=deterministic,
            )
            if hyp_result:
                sol.attempts.append(BranchAttempt(
                    technique="hypothesis",
                    success=True,
                    inputs=hyp_result,
                    detail="found via hypothesis.find()",
                ))
                if not sol.best_inputs:
                    sol.best_inputs = hyp_result
                    sol.solved = True
            else:
                sol.attempts.append(BranchAttempt(
                    technique="hypothesis",
                    success=False,
                    detail="no inputs found (or hypothesis not installed)",
                ))
        except Exception as e:
            logger.debug("hypothesis failed for line %d: %s", lineno, e)
            sol.attempts.append(BranchAttempt(
                technique="hypothesis", success=False,
                detail=f"error: {e}",
            ))

    # ---- Post-step A: Upgrade SimpleNamespace values via constructor ----
    _is_sentinel = bool(
        sol.best_inputs and (
            "__sequence__" in sol.best_inputs
            or "__iter_empty__" in sol.best_inputs
        )
    )
    if sol.solved and sol.best_inputs and not _is_sentinel:
        try:
            _upgrade_namespace_values(sol, rp)
        except Exception:
            pass

    # ---- Post-step B: Exception-driven feedback (verify + correct inputs) ----
    if sol.solved and sol.best_inputs and sol.callable_id and not _is_sentinel:
        try:
            from tharkun.solve.frontend.exception_feedback import try_execute_and_correct
            corrected = try_execute_and_correct(rp, sol.callable_id, sol.best_inputs)
            if corrected and corrected != sol.best_inputs:
                sol.attempts.append(BranchAttempt(
                    technique="exception_feedback",
                    success=True,
                    inputs=corrected,
                    detail="inputs corrected after TypeError",
                ))
                sol.best_inputs = corrected
        except Exception:
            pass  # Non-critical — original inputs remain

    # ---- Post-step C: Proxy-based type inference for unsolved branches ----
    if not sol.solved and sol.callable_id:
        try:
            proxy_result = _try_proxy_inference(rp, sol.callable_id, lineno)
            if proxy_result:
                sol.attempts.append(BranchAttempt(
                    technique="proxy_inference",
                    success=True,
                    inputs=proxy_result,
                    detail="types inferred via RecordingProxy execution",
                ))
                sol.best_inputs = proxy_result
                sol.solved = True
        except Exception:
            pass

    # ---- Post-step D: Populate agent-friendly fields ----
    if not sol.solved:
        sol.blocker_info = _compute_blocker_info(sol)
    if sol.solved and _cond_sol_variables:
        sol.best_inputs_constraints = _build_input_constraints(_cond_sol_variables)

    # ---- Post-step E: call_mapper backward propagation ----
    # Map best_inputs back to caller variable names so the solution is
    # actionable at the entry point, not just at the direct function.
    if sol.solved and sol.best_inputs and sol.callable_id:
        try:
            from tharkun.solve.frontend.call_mapper import find_calls_to
            callers = find_calls_to(rp, sol.callable_id)
            for mapping in callers:
                caller_inputs: dict[str, Any] = {}
                for arg_map in mapping.args:
                    param = arg_map.callee_param
                    if (param in sol.best_inputs
                            and arg_map.is_simple_name
                            and arg_map.caller_name):
                        caller_inputs[arg_map.caller_name] = sol.best_inputs[param]
                if caller_inputs:
                    sol.caller_inputs = caller_inputs
                    sol.attempts.append(BranchAttempt(
                        technique="call_mapper",
                        success=True,
                        inputs=caller_inputs,
                        detail=f"mapped back via {mapping.callee_name} at L{mapping.call_line}",
                    ))
                    break
        except Exception:
            pass

    return sol


# ---------------------------------------------------------------------------
# Fact extraction from prior attempts
# ---------------------------------------------------------------------------


def _build_fact_env_from_attempts(sol: BranchSolution) -> Any:
    """Build a FactEnv from condition_solver's VarConstraints.

    Extracts type/range info from step 1's VarConstraint list and
    converts to minleaf's FactEnv format for leaf-level fact injection.
    """
    try:
        from tharkun.minleaf.types import FactEnv, VarFacts
    except ImportError:
        return None

    facts: dict[str, Any] = {}
    for attempt in sol.attempts:
        if attempt.technique != "condition_solver":
            continue
        if not attempt.inputs:
            continue
        # Extract type info from the inputs
        for name, val in attempt.inputs.items():
            if isinstance(val, int):
                facts[name] = VarFacts(lo=val - 100, hi=val + 100)
            elif isinstance(val, float):
                facts[name] = VarFacts(lo=val - 100.0, hi=val + 100.0)
            elif isinstance(val, str) and val:
                facts[name] = VarFacts(eq={val})
            elif isinstance(val, bool):
                facts[name] = VarFacts(eq={val})

    if not facts:
        return None
    return FactEnv(facts=facts)


# ---------------------------------------------------------------------------
# Module loading for runtime execution
# ---------------------------------------------------------------------------


# #0607 (plan #0600): the loader lives in the solving core (the one place
# under solve/ that imports or executes a target).  These names stay for the
# fuzzing layer (hybrid/assist.py, hybrid/validity.py) and older callers.
from tharkun.solve.loader import (  # noqa: E402
    file_path_to_dotted_module as _file_path_to_dotted_module,
    load_module_for_execution as _load_module_for_execution,
)


# ---------------------------------------------------------------------------
# Method detection and object construction for OOP code
# ---------------------------------------------------------------------------


def _find_method_call_target(
    mod: Any,
    module_globals: dict[str, Any],
    source_path: str,
    lineno: int,
    callable_id: str,
    seed_kwargs: dict[str, Any],
) -> tuple[Any, dict[str, Any]] | None:
    """Find a method call target by constructing the enclosing class.

    For a method like `Detector.check`, this:
    1. Finds the enclosing class from AST
    2. Gets the class from the module
    3. Constructs an instance (with synthesized args)
    4. Calls setup methods (from state_sequence / attr_tracer)
    5. Returns (bound_method, kwargs_without_self)
    """
    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None

    # Find the enclosing class for this method
    enclosing_class_name = None
    for cls_node in ast.walk(tree):
        if not isinstance(cls_node, ast.ClassDef):
            continue
        for child in cls_node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if child.name == callable_id:
                    start = getattr(child, "lineno", 0)
                    end = getattr(child, "end_lineno", start + 1000)
                    if start <= lineno <= end:
                        enclosing_class_name = cls_node.name
                        break
        if enclosing_class_name:
            break

    if not enclosing_class_name:
        return None

    # Get the class from the module
    cls = getattr(mod, enclosing_class_name, None)
    if cls is None or not isinstance(cls, type):
        return None

    # Construct an instance
    obj = None
    # Strategy 1: no-arg constructor
    try:
        obj = cls()
    except (TypeError, Exception):
        pass

    # Strategy 2: use constructor.py to analyze __init__
    if obj is None:
        try:
            from tharkun.solve.frontend.constructor import construct_instance
            obj = construct_instance(rp, enclosing_class_name)
            if not isinstance(obj, cls):
                obj = None
        except Exception:
            pass

    # Strategy 3: __new__ without __init__
    if obj is None:
        try:
            obj = cls.__new__(cls)
        except (TypeError, Exception):
            pass

    if obj is None:
        return None

    # Get the bound method
    method = getattr(obj, callable_id, None)
    if method is None or not callable(method):
        return None

    # Filter out 'self' from seed_kwargs
    filtered_kwargs = {k: v for k, v in seed_kwargs.items() if k != "self"}

    return (method, filtered_kwargs)


# ---------------------------------------------------------------------------
# Minleaf integration
# ---------------------------------------------------------------------------


def _try_minleaf_solve(
    source_path: str,
    lineno: int,
    sol: BranchSolution,
    *,
    facts: Any = None,
) -> tuple[dict[str, Any] | None, NarrowedConstraint | None]:
    """Try to solve a branch using minleaf's runtime leaf-patch approach.

    Executes the enclosing function with seed kwargs, captures the runtime
    state at the branch line, then uses minleaf to find the minimum leaf
    mutations to flip the guard.

    Args:
        facts: Optional FactEnv from condition_solver's prior analysis.

    Returns (inputs_dict, None) on success, or (None, narrowed_constraint) on failure.
    """
    from tharkun.minleaf.api import solve_minleaf_patch_for_guard

    rp = os.path.realpath(source_path)

    # Build seed kwargs and execute to get runtime state
    seed_kwargs = _build_seed_kwargs(sol, rp) or {}
    if not seed_kwargs and not sol.callable_id:
        return None, None

    # Load the module — prefer package import (handles relative imports)
    mod, module_globals = _load_module_for_execution(rp)
    if mod is None:
        return None, None

    # Find the function — may be top-level or a method inside a class.
    func = getattr(mod, sol.callable_id, None)
    call_target = None  # (func, kwargs) or (bound_method, kwargs)

    if func is not None and callable(func):
        call_target = (func, seed_kwargs)
    else:
        # Try to find as a method: look for Class.method pattern
        call_target = _find_method_call_target(
            mod, module_globals, rp, lineno, sol.callable_id, seed_kwargs
        )

    if call_target is None:
        return None, None

    call_func, call_kwargs = call_target

    # Capture f_locals at the branch line via tracer.
    # Use capture_real=True so minleaf gets actual live objects for leaf projection.
    from tharkun.trace_api import TracePolicy, Tracer
    policy = TracePolicy(
        paths={rp},
        snapshot_lines={rp: {lineno}},
        capture_real=True,
    )
    tracer = Tracer(policy)
    try:
        with tracer:
            call_func(**call_kwargs)
    except Exception:
        pass

    # Prefer real objects for minleaf roots; fall back to safe snapshots for env.
    real_snaps = tracer.result.snapshots_real.get(rp, {}).get(lineno, [])
    safe_snaps = tracer.result.snapshots.get(rp, {}).get(lineno, [])

    if not real_snaps and not safe_snaps:
        return None, None

    # Build roots from real objects (for leaf projection)
    roots: dict[str, Any] = {}
    if real_snaps:
        roots = dict(real_snaps[0])

    # Add module-level globals (constants, imported names) that the guard
    # may reference but aren't in f_locals. Essential for guards like
    # `value < powers[0]` where `powers` is a module constant.
    for k, v in module_globals.items():
        if k.startswith("_") or k == "__builtins__":
            continue
        if k not in roots:
            roots[k] = v

    # Build env from roots — include primitives and containers for eval.
    env: dict[str, Any] = {}
    for k, v in roots.items():
        if isinstance(v, (bool, int, float, str, bytes, type(None),
                          list, tuple, dict, set, frozenset)):
            env[k] = v

    if not roots:
        return None, None

    # Run minleaf — pass through the caller's desired branch polarity.
    desired = sol.desired_branch  # "then" | "else"
    try:
        result = solve_minleaf_patch_for_guard(
            source_path=rp,
            target_line=lineno,
            desired_branch=desired,
            env=env,
            roots=roots,
            facts=facts,
        )
        if result.success and result.patch:
            # Convert patch ops to input dict.
            inputs = dict(env)
            for op in result.patch.ops:
                if len(op.path) >= 1:
                    root_name = op.path[0]
                    if isinstance(root_name, str):
                        if len(op.path) == 1:
                            inputs[root_name] = op.new
                        else:
                            if root_name in roots:
                                inputs[root_name] = roots[root_name]
            return inputs, None

        # Failure: extract diagnostics for downstream solvers
        diag = result.diagnostics
        narrowed = NarrowedConstraint(
            blocking_exprs=[a.expr_src for a in diag.blocking_atoms],
            leaf_paths=list(diag.dependency_leaf_paths),
            structural_reason=diag.structural_blockers[0] if diag.structural_blockers else None,
        )
        # Extract var types and bounds from leaf paths
        for path in diag.dependency_leaf_paths:
            if path and isinstance(path[0], str):
                var_name = path[0]
                leaf = roots.get(var_name)
                if leaf is not None:
                    narrowed.var_types[var_name] = type(leaf).__name__
                    if isinstance(leaf, (int, float)):
                        from tharkun.solve.config import config
                        w = config.narrowed_window
                        narrowed.var_bounds[var_name] = (leaf - w, leaf + w)
        return None, narrowed
    except Exception:
        pass

    return None, None


# ---------------------------------------------------------------------------
# Input verification
# ---------------------------------------------------------------------------


_EVAL_GLOBALS: dict[str, Any] = {
    "__builtins__": {
        "len": len, "isinstance": isinstance, "type": type,
        "int": int, "str": str, "float": float, "bool": bool,
        "list": list, "dict": dict, "set": set, "tuple": tuple,
        "hasattr": hasattr, "getattr": getattr, "callable": callable,
        "True": True, "False": False, "None": None,
        "abs": abs, "min": min, "max": max, "sum": sum,
        "any": any, "all": all,
    }
}


def _eval_guard(
    source_path: str,
    lineno: int,
    inputs: dict[str, Any],
) -> bool | None:
    """Evaluate the guard at *lineno* with *inputs* and return its boolean value.

    Returns ``None`` if the guard cannot be found or evaluated (e.g. missing
    imports, complex objects).  Callers should fall back to line-hit checks.
    """
    from tharkun.solve.condition import _find_branch_at_line
    try:
        source = Path(os.path.realpath(source_path)).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
        guard = _find_branch_at_line(tree, lineno)
        if guard is None:
            return None
        guard_src = ast.unparse(guard)
        result = eval(guard_src, _EVAL_GLOBALS, dict(inputs))  # noqa: S307
        return bool(result)
    except Exception:
        return None


def _verify_inputs_satisfy_guard(
    guard_source: str,
    inputs: dict[str, Any],
    desired_branch: str = "then",
) -> bool:
    """Check if *inputs* produce the desired edge outcome for *guard_source*.

    Returns True if evaluation matches *desired_branch* polarity, False if
    it doesn't, and True (optimistic) if evaluation fails (e.g. missing
    variables or type errors).
    """
    expected = desired_branch == "then"
    try:
        result = eval(guard_source, _EVAL_GLOBALS, dict(inputs))  # noqa: S307
        return bool(result) == expected
    except Exception:
        return True


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def _build_seed_kwargs(
    sol: BranchSolution,
    source_path: str,
    callable_info: Any = None,
) -> dict[str, Any] | None:
    """Build seed keyword arguments to call the enclosing function.

    Uses partial solutions from earlier steps, proxy objects, and default
    values to construct kwargs that will get execution as close to the
    branch line as possible — so the tracer can capture f_locals.

    Returns None if we can't build kwargs (e.g. no callable_id).
    """
    if not sol.callable_id:
        return None

    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None

    # Find the function definition to get its parameter list
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == sol.callable_id:
                func_def = node
                break
    if func_def is None:
        return None

    params = []
    for arg in func_def.args.args:
        if arg.arg in ("self", "cls"):
            continue
        params.append(arg.arg)

    if not params:
        return {}

    # Start with any partial inputs from earlier solving attempts
    kwargs: dict[str, Any] = {}

    # Pull from successful attempts (steps 1-5)
    for attempt in sol.attempts:
        if attempt.success and attempt.inputs:
            for name, val in attempt.inputs.items():
                if name in params and name not in kwargs:
                    kwargs[name] = val

    # For remaining params, synthesize defaults using type info
    from tharkun.solve.condition import _default_value_for_type

    if callable_info is not None:
        for p in getattr(callable_info, "params", []):
            if p.name in params and p.name not in kwargs:
                val = _default_value_for_type(p.type_hint.rendered)
                if val is not None:
                    kwargs[p.name] = val

    # Try type propagation for remaining unknowns
    try:
        propagated = propagate_types(rp, sol.callable_id)
        for name, type_str in propagated.items():
            if name in params and name not in kwargs:
                val = _default_value_for_type(type_str)
                if val is not None:
                    kwargs[name] = val
    except Exception:
        pass

    # Try to use annotations from the function def for remaining params
    if func_def:
        for arg in func_def.args.args:
            if arg.arg in params and arg.arg not in kwargs and arg.annotation:
                try:
                    ann = ast.unparse(arg.annotation)
                    val = _default_value_for_type(ann)
                    if val is not None:
                        kwargs[arg.arg] = val
                except Exception:
                    pass

    # Check for default values in the function signature.
    # Parameters WITH defaults that we can't resolve are OMITTED (not proxied),
    # so the real default applies at runtime. This is critical for identity
    # checks like `types is self.default` where the default must be the exact
    # sentinel object.
    params_with_defaults: set[str] = set()
    if func_def:
        n_defaults = len(func_def.args.defaults)
        n_args = len(func_def.args.args)
        default_start = n_args - n_defaults
        for i, arg in enumerate(func_def.args.args):
            if i >= default_start:
                params_with_defaults.add(arg.arg)
                if arg.arg in params and arg.arg not in kwargs:
                    default_node = func_def.args.defaults[i - default_start]
                    if isinstance(default_node, ast.Constant):
                        kwargs[arg.arg] = default_node.value
                    # Non-literal defaults: DON'T override — let the real default apply

    # For any still-missing params WITHOUT defaults, use RecordingProxy
    for name in params:
        if name not in kwargs and name not in params_with_defaults:
            kwargs[name] = RecordingProxy()

    return kwargs


def _collect_snapshot_locals(analysis: AnalysisResult) -> dict[str, Any]:
    """Merge all snapshot f_locals from an analysis into a single dict.

    When multiple snapshots exist (from different execution paths), merges
    them with later snapshots taking precedence.  Returns scalar values
    directly; complex values are kept as-is for type inference.
    """
    merged: dict[str, Any] = {}

    # The diagnostics have relevant_vars extracted from snapshot data
    for diag in analysis.diagnostics:
        if diag.relevant_vars:
            for name, val in diag.relevant_vars.items():
                merged[name] = val

    return merged


# ---------------------------------------------------------------------------
# Post-step helpers: upgrade inputs via constructor and proxy
# ---------------------------------------------------------------------------


def _upgrade_namespace_values(sol: BranchSolution, source_path: str) -> None:
    """Replace SimpleNamespace values in best_inputs with real objects.

    If the condition solver produced a SimpleNamespace for a parameter that
    maps to a real class in the codebase, try to construct a real instance.
    """
    from types import SimpleNamespace as _NS

    if not sol.best_inputs:
        return

    for name, value in list(sol.best_inputs.items()):
        if not isinstance(value, _NS):
            continue
        # Check if the guard mentions isinstance(name, SomeClass)
        # and SomeClass is defined in the source file
        if not sol.guard_source:
            continue
        import re
        m = re.search(rf"isinstance\s*\(\s*{re.escape(name)}\s*,\s*(\w+)", sol.guard_source)
        if not m:
            continue
        class_name = m.group(1)
        # Skip builtins
        if class_name in ("str", "int", "float", "bool", "list", "dict", "set", "tuple", "bytes"):
            continue
        real_obj = construct_instance(source_path, class_name)
        if not isinstance(real_obj, _NS):
            sol.best_inputs[name] = real_obj


def _try_proxy_inference(
    source_path: str,
    callable_id: str,
    lineno: int,
) -> dict[str, Any] | None:
    """Try to infer parameter types via RecordingProxy execution.

    Passes RecordingProxy objects as arguments, executes the function,
    then infers types from recorded attribute usage and re-solves.
    """
    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None

    # Find the function and its parameters
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == callable_id:
                func_def = node
                break
    if func_def is None:
        return None

    params = [arg.arg for arg in func_def.args.args if arg.arg not in ("self", "cls")]
    if not params:
        return None

    # Create proxies and execute
    proxies: dict[str, RecordingProxy] = {}
    for p in params:
        proxies[p] = RecordingProxy()

    try:
        mod, module_globals = _load_module_for_execution(rp)
        func = getattr(mod, callable_id, None) if mod else None
        if func is None or not callable(func):
            return None
        func(**proxies)
    except Exception:
        pass  # Expected — proxies won't satisfy all operations

    # Infer types from proxy usage
    from tharkun.solve.condition import _default_value_for_type
    inputs: dict[str, Any] = {}
    any_inferred = False
    for name, proxy in proxies.items():
        inferred = infer_type_from_usage(proxy)
        if inferred != "Any":
            val = _default_value_for_type(inferred)
            if val is not None:
                inputs[name] = val
                any_inferred = True
            else:
                inputs[name] = proxy
        else:
            inputs[name] = proxy

    if not any_inferred:
        return None

    # Re-solve with inferred types: try executing with real values
    # to see if the target branch is hit
    from tharkun.trace_api import TracePolicy, Tracer
    try:
        func = module_globals.get(callable_id)
        if func is None:
            return None
        policy = TracePolicy(paths={rp})
        tracer = Tracer(policy)
        with tracer:
            func(**inputs)
        executed = tracer.result.executed.get(rp, set())
        if lineno in executed:
            return inputs
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# State sequence solver: self.xxx guards
# ---------------------------------------------------------------------------


def _solve_state_guard(sol: BranchSolution, source_path: str) -> BranchSolution:
    """For guards on self.xxx, determine what method calls are needed to
    reach the required state, and generate a multi-step test plan.

    Uses attr_tracer assignments to find which methods SET the attribute,
    then builds a call sequence: construct → setup_methods → target_method.
    """
    import ast as _ast

    try:
        source = Path(source_path).read_text(encoding="utf-8", errors="ignore")
        tree = _ast.parse(source)
    except (OSError, SyntaxError):
        return sol

    for attr_result in sol.attr_blockers:
        if not attr_result.assignments:
            continue

        target = attr_result.target
        guard_source = sol.guard_source

        # Determine what value the guard NEEDS the attribute to have.
        # Parse the guard to find the comparison value.
        required_value = _extract_required_attr_value(guard_source, target.full)

        # Find which methods assign the attribute to the required value.
        setup_methods: list[str] = []
        for assignment in attr_result.assignments:
            if assignment.is_init:
                # Init sets a default — check if it matches.
                if required_value is not None:
                    # If init sets it to something else, we need a non-init method.
                    continue
            else:
                # Non-init method sets the attribute — this is a setup method.
                method_name = assignment.method.split(".")[-1] if "." in assignment.method else assignment.method
                if method_name not in setup_methods:
                    setup_methods.append(method_name)

        if setup_methods:
            # Find the class name from the enclosing callable.
            class_name = None
            for node in tree.body:
                if isinstance(node, _ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                            start = getattr(child, "lineno", 0)
                            end = getattr(child, "end_lineno", start + 100)
                            if start <= sol.lineno <= end:
                                class_name = node.name
                                break

            sol.attempts.append(BranchAttempt(
                technique="state_sequence",
                success=True,
                detail=f"need to call {setup_methods} on {class_name or '?'} before reaching line {sol.lineno}",
                inputs={"__sequence__": {
                    "class": class_name,
                    "setup_methods": setup_methods,
                    "target_line": sol.lineno,
                    "attr": target.full,
                    "required_value": required_value,
                }},
            ))
            sol.solved = True
            if not sol.best_inputs:
                sol.best_inputs = {"__sequence__": {
                    "class": class_name,
                    "setup_methods": setup_methods,
                    "target_line": sol.lineno,
                }}

    return sol


def _extract_required_attr_value(guard_source: str, attr_full: str) -> str | None:
    """From a guard like 'self.state != "disconnected"', extract the comparison value."""
    import ast as _ast
    try:
        expr = _ast.parse(guard_source, mode="eval").body
    except SyntaxError:
        return None

    for node in _ast.walk(expr):
        if not isinstance(node, _ast.Compare):
            continue
        # Check if left side is our attribute.
        left_src = _unparse(node.left)
        if left_src == attr_full:
            for comp in node.comparators:
                if isinstance(comp, _ast.Constant):
                    return comp.value
        # Check comparators too (flipped).
        for comp in node.comparators:
            comp_src = _unparse(comp)
            if comp_src == attr_full and isinstance(node.left, _ast.Constant):
                return node.left.value
    return None


# ---------------------------------------------------------------------------
# Module-level solver
# ---------------------------------------------------------------------------


def solve_uncovered_branches(
    source_path: str,
    covered_lines: set[int] | None = None,
    *,
    execute_snapshots: bool = True,
    max_depth: int = 5,
    facts: Any = None,
    deterministic: bool = False,
) -> CoverageResult:
    """Apply aggressive per-branch solving to all uncovered branch edges.

    For each branch edge (``then`` and ``else`` for every ``if``/``while``/
    ``for``) in *source_path* whose line is NOT in *covered_lines*, runs the
    full toolset to find satisfying inputs.

    Args:
        source_path:       Python source file.
        covered_lines:     Set of already-covered line numbers (from traces).
                          Both edges for a line are skipped when the line is
                          in this set.
        execute_snapshots: Whether to run the code for runtime snapshots.
        max_depth:         Max decomposition depth.
        facts:             Optional external ``FactEnv`` for constraint narrowing.
        deterministic:     If True, use a fixed seed for
                          Hypothesis, producing reproducible results.

    Returns:
        A ``CoverageResult`` with per-edge solutions.  ``total_branches`` now
        counts edges (2× the number of guard lines).
    """
    rp = os.path.realpath(source_path)
    all_edges = _find_branch_edges(rp)
    covered = covered_lines or set()

    uncovered = [t for t in all_edges if t.lineno not in covered]

    result = CoverageResult(source_path=rp, total_branches=len(all_edges))
    for target in uncovered:
        sol = solve_branch(
            rp, target.lineno,
            covered_lines=covered,
            execute_snapshots=execute_snapshots,
            max_depth=max_depth,
            facts=facts,
            desired_branch=target.desired_branch,
            stmt_kind=target.stmt_kind,
            deterministic=deterministic,
        )
        result.solutions.append(sol)
        if sol.solved:
            result.solved_branches += 1
        else:
            result.unsolved_branches += 1

    return result


def solve_branches_progressive(
    source_path: str,
    *,
    execute_snapshots: bool = False,
    max_depth: int = 5,
    deterministic: bool = False,
) -> Iterator[BranchSolution]:
    """Solve all branch edges in *source_path*, yielding each result immediately.

    Unlike ``solve_uncovered_branches()`` which collects all results before
    returning, this generator yields each ``BranchSolution`` as soon as it is
    computed.  Callers can stop iteration early (via ``break``) to limit the
    budget consumed.

    Usage::

        for sol in solve_branches_progressive("module.py"):
            if sol.solved:
                print(sol.lineno, sol.best_inputs)
            if budget_exhausted():
                break

    Args:
        source_path:       Python source file.
        execute_snapshots: Whether to run the code for runtime snapshots.
        max_depth:         Max decomposition depth.
        deterministic:     If True, seed Hypothesis for
                          reproducible results.

    Yields:
        ``BranchSolution`` for each branch edge, in line/polarity order.
    """
    rp = os.path.realpath(source_path)
    for target in _find_branch_edges(rp):
        yield solve_branch(
            rp, target.lineno,
            desired_branch=target.desired_branch,
            stmt_kind=target.stmt_kind,
            execute_snapshots=execute_snapshots,
            max_depth=max_depth,
            deterministic=deterministic,
        )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _safe_json_value(v: Any) -> Any:
    """Make a value JSON-serializable."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, bytes):
        return repr(v)
    if isinstance(v, (list, tuple)):
        return [_safe_json_value(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _safe_json_value(val) for k, val in v.items()}
    if isinstance(v, set):
        return [_safe_json_value(x) for x in sorted(v, key=repr)]
    # Anything else: repr it
    try:
        return repr(v)
    except Exception:
        return f"<{type(v).__name__}>"


def _compute_blocker_info(sol: BranchSolution) -> dict | None:
    """Derive a structured blocker diagnostic from a completed unsolved BranchSolution.

    Checks signals in priority order:
    1. infeasible — condition_solver explicitly said so
    2. external_call — mock_patches present
    3. attribute_unset — attr_blockers present
    4. decomposer_failed — decomposition.blockers present
    5. no_technique_succeeded — fallback
    """
    if sol.solved:
        return None
    for attempt in sol.attempts:
        if attempt.technique == "condition_solver" and "infeasible" in (attempt.detail or ""):
            return {
                "kind": "infeasible",
                "detail": f"Guard '{sol.guard_source}' is statically infeasible",
                "suggestion": "manual",
                "patch_target": None,
            }
    if sol.mock_patches:
        first = sol.mock_patches[0]
        return {
            "kind": "external_call",
            "detail": f"{first.target} ({first.category})",
            "suggestion": "add_mock_patch",
            "patch_target": first.target,
        }
    if sol.attr_blockers:
        full = sol.attr_blockers[0].target.full
        has_assignments = bool(sol.attr_blockers[0].assignments)
        detail = (
            f"'{full}' requires a setup method call" if has_assignments
            else f"'{full}' has no known assignments"
        )
        return {
            "kind": "attribute_unset",
            "detail": detail,
            "suggestion": "call_setup_method",
            "patch_target": None,
        }
    if sol.decomposition and sol.decomposition.blockers:
        b = sol.decomposition.blockers[0]
        return {
            "kind": "decomposer_failed",
            "detail": f"'{b.expression}': {b.blocker_reason}",
            "suggestion": "manual",
            "patch_target": None,
        }
    return {
        "kind": "no_technique_succeeded",
        "detail": f"Tried: {', '.join(a.technique for a in sol.attempts)}",
        "suggestion": "manual",
        "patch_target": None,
    }


def serialize_branch_solution(sol: BranchSolution) -> dict[str, Any]:
    return {
        "file": sol.file,
        "lineno": sol.lineno,
        "desired_branch": sol.desired_branch,
        "stmt_kind": sol.stmt_kind,
        "guard_source": sol.guard_source,
        "callable_id": sol.callable_id,
        "solved": sol.solved,
        "attempts": [
            {
                "technique": a.technique,
                "success": a.success,
                "detail": a.detail,
                "inputs": _safe_json_value(a.inputs),
            }
            for a in sol.attempts
        ],
        "best_inputs": _safe_json_value(sol.best_inputs),
        "best_inputs_constraints": sol.best_inputs_constraints,
        "mock_patches": [
            {"target": p.target, "category": p.category}
            for p in sol.mock_patches
        ],
        "attr_blockers": len(sol.attr_blockers),
        "blocker_info": sol.blocker_info,
    }


def serialize_coverage_result(result: CoverageResult) -> dict[str, Any]:
    return {
        "source_path": result.source_path,
        "total_branches": result.total_branches,
        "solved_branches": result.solved_branches,
        "unsolved_branches": result.unsolved_branches,
        "solutions": [serialize_branch_solution(s) for s in result.solutions],
    }
