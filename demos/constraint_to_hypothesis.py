#!/usr/bin/env python3
"""Toy demo: tharkun tracks a guard's constraints, then Hypothesis produces a
valid value that satisfies them — verified by executing the real function.

This is a *demonstration of tharkun's existing capability*, not an engine /
eregion integration. It exercises two public pieces of tharkun:

  * ``condition_solver.solve_condition()`` — reads the (trivial) constraints
    off the guard AST and infers each variable's type. We print the tracked
    constraints. Crucially, condition_solver's *static* satisfying pick does
    **not** crack these guards (the modulo / length bound defeat its first
    guess) — so the value that finally satisfies the guard must come from the
    next stage.

  * ``hypothesis_adapter.hypothesis_solve_branch()`` — tharkun's Hypothesis
    (step-10) solver. We feed it a ``NarrowedConstraint`` *built from the
    constraints condition_solver tracked* (type + integer bounds), so the
    Hypothesis strategy is narrowed exactly the way the existing
    minleaf -> Hypothesis path narrows it. Hypothesis searches and returns a
    concrete value, and its predicate verifies the value by executing the real
    fixture function and checking — via tharkun's tracer — that the guard's
    true branch was taken.

The fixture under analysis lives in
``tests/fixtures/toy_constraint_hypothesis.py``.

Run it::

    cd /path/to/tharkun
    PYTHONPATH=src python demos/constraint_to_hypothesis.py

Exit code 0 means every demonstrated value was produced by Hypothesis and
verified to drive the guard's true branch through the real function.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

# Make tharkun importable when run as `python demos/constraint_to_hypothesis.py`
# from the repo root without PYTHONPATH (best-effort; PYTHONPATH=src also works).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tharkun import solve_condition  # noqa: E402
from tharkun.branch_solver import NarrowedConstraint  # noqa: E402
from tharkun.hypothesis_adapter import (  # noqa: E402
    hypothesis_available,
    hypothesis_solve_branch,
)

FIXTURE = os.path.join(
    _REPO_ROOT, "tests", "fixtures", "toy_constraint_hypothesis.py"
)


@dataclass
class Case:
    """One guard to demonstrate the constraint -> Hypothesis flow on."""

    func_name: str          # function in the fixture
    guard_line: int         # line of the `if` guard
    body_line: int          # line of the true-branch body (Hypothesis target)
    param: str              # the single parameter name
    expected_true: str      # return value when the guard's true branch is taken


CASES = [
    # x > 50 and x < 100 and x % 7 == 0   (int range + modulo)
    Case("classify", guard_line=24, body_line=25, param="x", expected_true="hit"),
    # s.startswith("v") and len(s) >= 4   (string prefix + length)
    Case("version_ok", guard_line=33, body_line=34, param="s", expected_true="valid"),
]


def _load_fixture_func(name: str) -> Callable[[Any], str]:
    """Import the real fixture function so we can execute it for verification."""
    spec = importlib.util.spec_from_file_location("toy_fixture", FIXTURE)
    assert spec and spec.loader, "could not load fixture spec"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, name)


def _narrowed_from_tracked(solution: Any, param: str) -> NarrowedConstraint:
    """Turn the constraints condition_solver tracked into a NarrowedConstraint.

    This is the same shape tharkun's minleaf step hands to the Hypothesis
    adapter: variable type + (for ints) integer bounds derived from the tracked
    concrete boundary values. For non-int params we still pass the tracked type
    so Hypothesis picks the right base strategy.
    """
    var = next((v for v in solution.variables if v.name == param), None)
    var_types: dict[str, str] = {}
    var_bounds: dict[str, tuple] = {}
    if var is not None:
        var_types[param] = var.inferred_type
        ints = [c for c in var.concrete_values if isinstance(c, int) and not isinstance(c, bool)]
        if var.inferred_type == "int" and len(ints) >= 2:
            var_bounds[param] = (min(ints), max(ints))
    return NarrowedConstraint(
        blocking_exprs=[solution.guard_source],
        var_types=var_types,
        var_bounds=var_bounds,
    )


def _fmt_constraints(solution: Any) -> str:
    parts = []
    for v in solution.variables:
        parts.append(
            f"{v.name}: {v.inferred_type} (via {v.inference_source}; "
            f"tracked values {v.concrete_values})"
        )
    return "; ".join(parts) if parts else "(none)"


def run_case(case: Case) -> bool:
    func = _load_fixture_func(case.func_name)

    print("=" * 72)
    print(f"GUARD  ({case.func_name}, line {case.guard_line}):")

    # --- Stage 1: tharkun tracks the guard's constraints (condition_solver) ---
    solution = solve_condition(FIXTURE, case.guard_line)
    print(f"    {solution.guard_source}")
    print()
    print("STAGE 1 — tharkun tracks constraints (condition_solver.solve_condition):")
    print(f"    tracked: {_fmt_constraints(solution)}")

    # Show that condition_solver's *static* pick does NOT satisfy the guard,
    # so the satisfying value genuinely has to come from Hypothesis.
    static_pick = None
    if solution.satisfying_inputs:
        static_pick = solution.satisfying_inputs[0].get(case.param)
    if static_pick is not None:
        static_result = func(static_pick)
        print(
            f"    condition_solver static pick: {case.param}={static_pick!r} -> "
            f"{case.func_name}({static_pick!r}) == {static_result!r} "
            f"{'(satisfies)' if static_result == case.expected_true else '(does NOT satisfy — modulo/length defeats the static pick)'}"
        )

    # --- Stage 2: Hypothesis produces a valid value, narrowed by the tracked
    #     constraints (hypothesis_adapter.hypothesis_solve_branch) ------------
    narrowed = _narrowed_from_tracked(solution, case.param)
    print()
    print("STAGE 2 — Hypothesis generates a valid value (hypothesis_adapter):")
    print(
        f"    narrowing strategy with tracked constraints -> "
        f"types={narrowed.var_types} bounds={narrowed.var_bounds or '{}'}"
    )
    produced = hypothesis_solve_branch(
        FIXTURE,
        case.body_line,            # target the guard's true-branch body line
        case.func_name,
        max_examples=2000,
        narrowed=narrowed,
        # [internal ref] part 3: matches tests/test_demo_constraint_hypothesis.py's
        # test_each_case_value_comes_from_hypothesis_and_verifies, which
        # already derandomizes for the same reason -- without it, this
        # search's random seed changes every run, so whether it finds a
        # satisfying example within max_examples is not fully reproducible.
        # A fixed seed is known to find one; ship the derandomization as
        # prevention for test_demo_runs_green (below main()), which wraps
        # this in the same deadline=None profile but was still an open flake
        # risk on the "does a random seed succeed" axis, independent of load.
        derandom=True,
    )
    if produced is None or case.param not in produced:
        print("    Hypothesis did NOT produce a value — FAIL")
        return False
    value = produced[case.param]
    print(f"    Hypothesis produced: {case.param}={value!r}")

    # --- Stage 3: verify by EXECUTING the real fixture function --------------
    result = func(value)
    ok = result == case.expected_true
    print()
    print("STAGE 3 — verify by executing the real function:")
    print(f"    {case.func_name}({value!r}) == {result!r}  "
          f"(want {case.expected_true!r})  {'PASS' if ok else 'FAIL'}")
    print()
    # The explicit point of the demo.
    print(
        f"  ==> tharkun tracked constraints {{{solution.guard_source}}} "
        f"-> Hypothesis generated {case.param}={value!r} "
        f"-> {case.func_name}({value!r})=={case.expected_true!r} "
        f"{'OK' if ok else 'FAILED'}"
    )

    # Hard assertion: the value must come from Hypothesis AND satisfy the guard
    # when run through the real function.
    assert ok, (
        f"{case.func_name}({value!r}) == {result!r}, expected {case.expected_true!r}"
    )
    return ok


def main() -> int:
    print("tharkun toy demo: guard constraints -> Hypothesis -> verified value")
    print()
    if not hypothesis_available():
        print("Hypothesis is not installed; this demo requires the `hypothesis` extra.")
        print("Install with:  pip install 'tharkun[hypothesis]'  (or `hypothesis`).")
        return 2

    all_ok = True
    for case in CASES:
        all_ok &= run_case(case)

    print("=" * 72)
    if all_ok:
        print("RESULT: all values were produced by Hypothesis and verified "
              "to drive the guard's true branch through the real function.")
        return 0
    print("RESULT: at least one case failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
