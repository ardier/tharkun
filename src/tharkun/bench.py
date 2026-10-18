"""Benchmark harness for tharkun.

Runs ``solve_uncovered_branches`` on a list of target files and produces
a solve-rate report.  Supports two measurement axes:

1. **Solve rate** — % of branches where the solver produces satisfying inputs.
2. **Branch coverage** — % of branches actually taken when executing with
   those inputs (requires ``--verify``).

Usage::

    from tharkun.bench import run_benchmark, format_report

    results = run_benchmark(["module_a.py", "module_b.py"])
    print(format_report(results))
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tharkun.branch_solver import solve_uncovered_branches, CoverageResult


@dataclass
class ModuleResult:
    """Benchmark result for a single module."""
    path: str
    name: str
    total_branches: int = 0
    solved_branches: int = 0
    verified_branches: int = 0  # Actually hit during execution
    blocked: list[tuple[int, str, str]] = field(default_factory=list)  # (lineno, guard_source, desired_branch)
    duration_s: float = 0.0
    error: str | None = None


@dataclass
class BenchmarkResult:
    """Aggregate benchmark result."""
    modules: list[ModuleResult] = field(default_factory=list)
    total_branches: int = 0
    total_solved: int = 0
    total_verified: int = 0
    total_duration_s: float = 0.0


def run_benchmark(
    targets: list[str],
    *,
    execute_snapshots: bool = False,
    verify: bool = False,
    max_depth: int = 5,
) -> BenchmarkResult:
    """Run the benchmark on a list of target files.

    Args:
        targets:           List of Python source file paths.
        execute_snapshots: Whether to run code for runtime snapshots.
        verify:            If True, execute functions with solved inputs and
                          check if branches are actually taken.
        max_depth:         Max decomposition depth.

    Returns:
        Aggregate benchmark result.
    """
    result = BenchmarkResult()

    for path in targets:
        rp = os.path.realpath(path)
        name = Path(rp).stem
        mod = ModuleResult(path=rp, name=name)

        t0 = time.time()
        try:
            coverage = solve_uncovered_branches(
                rp,
                execute_snapshots=execute_snapshots,
                max_depth=max_depth,
            )
            mod.total_branches = coverage.total_branches
            mod.solved_branches = coverage.solved_branches
            mod.duration_s = time.time() - t0

            for sol in coverage.solutions:
                if not sol.solved:
                    guard = (sol.guard_source or "")[:60]
                    mod.blocked.append((sol.lineno, guard, sol.desired_branch))

            if verify:
                mod.verified_branches = _verify_coverage(rp, coverage)

        except Exception as e:
            mod.error = str(e)
            mod.duration_s = time.time() - t0

        result.modules.append(mod)
        result.total_branches += mod.total_branches
        result.total_solved += mod.solved_branches
        result.total_verified += mod.verified_branches
        result.total_duration_s += mod.duration_s

    return result


def _verify_coverage(source_path: str, coverage: CoverageResult) -> int:
    """Execute functions with solved inputs and count branches actually taken."""
    from tharkun.trace_api import TracePolicy, Tracer

    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        module_globals: dict[str, Any] = {"__builtins__": __builtins__}
        exec(compile(source, rp, "exec"), module_globals)
    except Exception:
        return 0

    verified = 0
    for sol in coverage.solutions:
        if not sol.solved or not sol.best_inputs or not sol.callable_id:
            continue
        if "__sequence__" in sol.best_inputs or "__iter_empty__" in sol.best_inputs:
            continue

        func = module_globals.get(sol.callable_id)
        if func is None or not callable(func):
            continue

        # Edge-level verification: evaluate the guard with the inputs and
        # check the result against the desired branch polarity.
        from tharkun.branch_solver import _eval_guard
        guard_val = _eval_guard(rp, sol.lineno, sol.best_inputs)
        if guard_val is not None:
            # Guard evaluation succeeded — check polarity.
            if guard_val == (sol.desired_branch == "then"):
                verified += 1
        else:
            # Guard evaluation failed (complex objects, imports) —
            # fall back to line-hit check for this solution.
            policy = TracePolicy(paths={rp})
            tracer = Tracer(policy)
            try:
                with tracer:
                    func(**sol.best_inputs)
            except Exception:
                pass
            executed = tracer.result.executed.get(rp, set())
            if sol.lineno in executed:
                verified += 1

    return verified


def format_report(result: BenchmarkResult) -> str:
    """Format a benchmark result as a human-readable report."""
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("Tharkun Self-Benchmark Report")
    lines.append("=" * 70)
    lines.append("")

    # Per-module table (branches = edges: 2× guard lines)
    lines.append(f"{'Module':<25} {'Edges':>10} {'Solved':>10} {'Rate':>8} {'Time':>8}")
    lines.append("-" * 70)

    for mod in sorted(result.modules, key=lambda m: m.name):
        rate = 100 * mod.solved_branches / max(mod.total_branches, 1)
        time_str = f"{mod.duration_s:.2f}s"
        if mod.error:
            lines.append(f"{mod.name:<25} {'ERROR':>10} {mod.error[:30]:>30}")
        else:
            lines.append(
                f"{mod.name:<25} {mod.total_branches:>10} "
                f"{mod.solved_branches:>10} {rate:>7.0f}% {time_str:>8}"
            )

    lines.append("-" * 70)
    total_rate = 100 * result.total_solved / max(result.total_branches, 1)
    lines.append(
        f"{'TOTAL':<25} {result.total_branches:>10} "
        f"{result.total_solved:>10} {total_rate:>7.1f}% "
        f"{result.total_duration_s:>7.2f}s"
    )

    if result.total_verified > 0:
        verify_rate = 100 * result.total_verified / max(result.total_solved, 1)
        lines.append(f"{'VERIFIED':<25} {'':>10} {result.total_verified:>10} {verify_rate:>7.1f}%")

    lines.append("")

    # Blocked edges summary
    total_blocked = sum(len(m.blocked) for m in result.modules)
    if total_blocked > 0:
        lines.append(f"--- {total_blocked} Blocked Edges ---")
        for mod in result.modules:
            for lineno, guard, desired_branch in mod.blocked:
                lines.append(f"  {mod.name}:{lineno}({desired_branch})  {guard}")
        lines.append("")

    return "\n".join(lines)


def self_benchmark(*, verify: bool = False) -> BenchmarkResult:
    """Run benchmark on tharkun's own source modules."""
    src_dir = os.path.join(os.path.dirname(__file__))
    targets = sorted(
        str(p) for p in Path(src_dir).glob("*.py")
        if p.name != "__init__.py" and p.name != "__main__.py"
    )
    return run_benchmark(targets, verify=verify)
