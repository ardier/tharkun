"""``tharkun.solve`` -- the constraint-solving core (plan #0600, #0606).

A function belongs here iff its input is a predicate about named variables (a
guard AST, a ``FactEnv``, a source anchor) plus optional prior facts, and its
output is a claim about the variables' value set -- refined facts, a sound
domain, a witness, or satisfiable / infeasible / unknown -- computed **without
an observed output and without executing the target**.  Anything that runs the
target (``sys.settrace``, live ``f_locals``, hypothesis search, fuzzing) lives in
the layers above (``trace_api``, ``guard_analyzer``, ``branch_solver``,
``driver``, ``hybrid``).  ``docs/LAYERS.md`` states the rule with worked
examples; ``tests/test_layering.py`` enforces it: modules under ``solve/`` import
only ``solve/``.

The package root is lazy (PEP 562): ``from tharkun.solve import solve_condition``
loads the static solver and nothing else.  eregion's solver bridge imports only
these public names.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS: dict[str, str] = {
    # the static guard solver
    "solve_condition": "tharkun.solve.condition",
    "solve_with_snapshot": "tharkun.solve.condition",
    "serialize_solution": "tharkun.solve.condition",
    "infer_var_type": "tharkun.solve.condition",
    "ConditionSolution": "tharkun.solve.condition",
    "VarConstraint": "tharkun.solve.condition",
    # loop-exit reasoning (eregion's live call)
    "loop_exit_factenv": "tharkun.solve.loop_exit",
    "collect_loop_exit_facts": "tharkun.solve.loop_exit",
    # guard decomposition
    "decompose_guard": "tharkun.solve.decompose",
    "serialize_decomp": "tharkun.solve.decompose",
    "DecompNode": "tharkun.solve.decompose",
    "DecompResult": "tharkun.solve.decompose",
    # the fact carrier and its arithmetic
    "FactEnv": "tharkun.solve.fact_env",
    "VarFacts": "tharkun.solve.fact_env",
    "refine": "tharkun.solve.fact_env",
    "join": "tharkun.solve.fact_env",
    # sound value-set claims for a guard anchor (#0611)
    "guard_facts": "tharkun.solve.claims",
    "guard_domains": "tharkun.solve.claims",
    "GuardClaim": "tharkun.solve.claims",
    "IntDomain": "tharkun.solve.claims",
    "Provenance": "tharkun.solve.claims",
    # guard extraction / normalisation / evaluation
    "extract_frontier": "tharkun.solve.guard",
    "desired_guard_value": "tharkun.solve.guard",
    "Frontier": "tharkun.solve.guard",
    "find_branch_at_line": "tharkun.solve.guard",
    "free_names": "tharkun.solve.guard",
    "normalize_guard": "tharkun.solve.normalize",
    "trace_guard": "tharkun.solve.guard_eval",
    "canonicalize": "tharkun.solve.canonicalize",
    # proven affine bracket of an integer expression (#0911)
    "affine_envelope": "tharkun.solve.envelope",
    # a witness for an integer guard: linear, modular, polynomial ([internal ref])
    "solve_int_guard": "tharkun.solve.arith",
    "ArithResult": "tharkun.solve.arith",
    # non-relational value facts and sound transfer functions ([internal ref])
    "IntFact": "tharkun.solve.values",
    "FloatFact": "tharkun.solve.values",
    "StrFact": "tharkun.solve.values",
    "Kinds": "tharkun.solve.values",
    "Value": "tharkun.solve.values",
    "refine_compare": "tharkun.solve.values",
    # a Z3-free UNSAT proof for conditions over typed names ([internal ref], dead-path)
    "refute_stdlib": "tharkun.solve.refute",
    # sound per-variable arm facts from a guard AST: eregion reduce_domain port ([internal ref])
    "domain_of": "tharkun.solve.guard_domain",
    "refine_env": "tharkun.solve.guard_domain",
    "GuardDomain": "tharkun.solve.guard_domain",
    "AffineEnvelope": "tharkun.solve.envelope",
    # configuration and constants
    "config": "tharkun.solve.config",
    "Config": "tharkun.solve.config",
    "configure": "tharkun.solve.config",
    "reset": "tharkun.solve.config",
    "BUILTIN_NAMES": "tharkun.solve.constants",
    # the one target loader and its off switch
    "no_targets": "tharkun.solve.loader",
    "targets_disabled": "tharkun.solve.loader",
    # project import (#1132b): the real import of a package member, static split, isolated variant
    "project_root_and_module": "tharkun.solve.loader",
    "is_package_member": "tharkun.solve.loader",
    "load_project_module": "tharkun.solve.loader",
    "import_project_isolated": "tharkun.solve.loader",
    "ProjectImportError": "tharkun.solve.loader",
    # static module namespace across imports (#1132b, [internal ref]): proven module constants, compiled regexes
    "module_constants": "tharkun.solve.module_namespace",
    "regex_globals": "tharkun.solve.module_namespace",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    home = _EXPORTS.get(name)
    if home is not None:
        value = getattr(importlib.import_module(home), name)
        globals()[name] = value
        return value
    if not name.startswith("_"):
        try:
            return importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as exc:
            if exc.name != f"{__name__}.{name}":
                raise
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
