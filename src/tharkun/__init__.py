"""Tharkun — Python branch guard solver and analysis toolkit.

Tharkun is a standalone package extracted from tulkas for solving Python
branch guard conditions. Given any ``if``/``while``/``for`` guard in Python
source code, tharkun can:

1. **Solve** the guard statically — infer variable types and produce concrete
   satisfying inputs (``condition_solver``)
2. **Decompose** compound guards — trace into called functions to identify
   solvable vs blocking sub-conditions (``guard_decomposer``)
3. **Analyze** blockers at runtime — capture variable state at branch lines
   via debugger-level ``sys.settrace`` hooks (``guard_analyzer``)
4. **Solve per-branch aggressively** — apply 9 techniques in sequence at each
   uncovered branch (``branch_solver``)
5. **Detect external calls** — identify HTTP, filesystem, database, checksum
   calls that need mocking (``mock_gen``)
6. **Trace attributes** — find where ``self.xxx`` is assigned and what methods
   must be called to reach a state (``attr_tracer``)
7. **Generate mutation seeds** — produce targeted inputs to kill surviving
   mutants (``mutation_input_gen``)

Usage::

    from tharkun import solve_condition, decompose_guard, solve_uncovered_branches

    # Solve a single guard
    solution = solve_condition("module.py", 42)

    # Decompose a compound guard
    result = decompose_guard("module.py", 42)

    # Aggressively solve all uncovered branches
    coverage = solve_uncovered_branches("module.py")

The package root is **lazy** (PEP 562, #0605 / plan #0600): every public name
below is imported from its home module on first access, and submodules are
loaded on attribute access.  ``import tharkun.condition_solver`` therefore
loads the static solver and nothing else -- no tracer, no driver, no fuzzing
layer -- which is what lets ``tharkun.solve`` stay a leaf that eregion can
import without pulling the observation layers in.  The name -> module table
is the public surface; ``tests/test_public_surface.py`` pins it.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.2.0"  # keep in sync with pyproject.toml (tests/test_version_sync.py)

# Public name -> home module.  Order mirrors the historical eager import
# block so ``__all__`` reads the same way the old module did.
_EXPORTS: dict[str, str] = {
    # condition_solver -- the static guard solver
    "ConditionSolution": "tharkun.solve.condition",
    "VarConstraint": "tharkun.solve.condition",
    "solve_condition": "tharkun.solve.condition",
    "solve_with_snapshot": "tharkun.solve.condition",
    "serialize_solution": "tharkun.solve.condition",
    "collect_loop_exit_facts": "tharkun.solve.loop_exit",
    "loop_exit_factenv": "tharkun.solve.loop_exit",
    # guard_decomposer
    "DecompNode": "tharkun.solve.decompose",
    "DecompResult": "tharkun.solve.decompose",
    "decompose_guard": "tharkun.solve.decompose",
    "serialize_decomp": "tharkun.solve.decompose",
    # guard_analyzer -- runtime blocker analysis (settrace)
    "AnalysisResult": "tharkun.guard_analyzer",
    "BlockerDiagnostic": "tharkun.guard_analyzer",
    "analyze_blocker": "tharkun.guard_analyzer",
    "serialize_analysis": "tharkun.guard_analyzer",
    # branch_solver -- per-branch aggressive solving
    "BranchTarget": "tharkun.branch_solver",
    "BranchSolution": "tharkun.branch_solver",
    "CoverageResult": "tharkun.branch_solver",
    "solve_branch": "tharkun.branch_solver",
    "solve_uncovered_branches": "tharkun.branch_solver",
    "solve_branches_progressive": "tharkun.branch_solver",
    "serialize_coverage_result": "tharkun.branch_solver",
    # mock_gen
    "ExternalCall": "tharkun.mock_gen",
    "MockPlan": "tharkun.mock_gen",
    "PatchSpec": "tharkun.mock_gen",
    "analyze_mocking_needs": "tharkun.mock_gen",
    "detect_externals": "tharkun.mock_gen",
    "generate_patches": "tharkun.mock_gen",
    # attr_tracer
    "AttrTarget": "tharkun.solve.attrs",
    "AttrTraceResult": "tharkun.solve.attrs",
    "analyze_attr_blocker": "tharkun.solve.attrs",
    "extract_attr_targets": "tharkun.solve.attrs",
    "find_attr_assignments": "tharkun.solve.attrs",
    # mutation_input_gen
    "MutationContext": "tharkun.mutation_input_gen",
    "SeedInput": "tharkun.mutation_input_gen",
    "generate_killing_seeds": "tharkun.mutation_input_gen",
    # emit
    "emit_branch_tests": "tharkun.emit",
    "emit_branch_tests_to_file": "tharkun.emit",
    # import_resolver
    "resolve_module_path": "tharkun.solve.frontend.import_resolver",
    "parse_imports": "tharkun.solve.frontend.import_resolver",
    "load_function_def": "tharkun.solve.frontend.import_resolver",
    # call_mapper
    "ArgMapping": "tharkun.solve.frontend.call_mapper",
    "CallSiteMapping": "tharkun.solve.frontend.call_mapper",
    "map_call_site": "tharkun.solve.frontend.call_mapper",
    "find_calls_to": "tharkun.solve.frontend.call_mapper",
    "map_call_chain": "tharkun.solve.frontend.call_mapper",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    home = _EXPORTS.get(name)
    if home is not None:
        value = getattr(importlib.import_module(home), name)
        globals()[name] = value  # cache: later lookups skip this hook
        return value
    if not name.startswith("_"):
        # ``tharkun.condition_solver`` without an explicit ``import``:
        # load the submodule on demand (import binds it on the package).
        try:
            return importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__name__}.{name}":
                raise  # a real import failure inside the submodule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
