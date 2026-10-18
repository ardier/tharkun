"""Tharkun CLI — solve, decompose, analyze, and mock Python branch guards.

Usage::

    tharkun solve <file> <line>        — solve a single guard
    tharkun decompose <file> <line>    — decompose into sub-conditions
    tharkun analyze <file> <line>      — runtime snapshot + diagnostics
    tharkun branches <file>            — solve ALL branches aggressively
    tharkun emit <file> [-o out.py]    — generate pytest tests for solved branches
    tharkun mocks <file> [function]    — detect external calls
    tharkun bench [files...]           — benchmark solve rate on targets
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def _cmd_solve(file: str, line: int, *, branch: str = "then") -> int:
    from tharkun.solve.condition import serialize_solution, solve_condition

    sol = solve_condition(file, line, desired_branch=branch)
    sys.stdout.write(json.dumps(serialize_solution(sol), indent=2, sort_keys=True) + "\n")
    # Human summary on stderr.
    sys.stderr.write(
        f"\nGuard: {sol.guard_source} [{sol.desired_branch} branch]\n"
        f"Strategy: {sol.strategy}, Confidence: {sol.confidence}\n"
        f"Variables: {len(sol.variables)}, Satisfying inputs: {len(sol.satisfying_inputs)}\n"
    )
    for v in sol.variables:
        vals = ", ".join(repr(x) for x in v.concrete_values[:3])
        sys.stderr.write(f"  {v.name}: {v.inferred_type} ({v.inference_source}) = [{vals}]\n")
    return 0


def _cmd_decompose(file: str, line: int, *, max_depth: int) -> int:
    from tharkun.solve.decompose import decompose_guard, serialize_decomp

    result = decompose_guard(file, line, max_depth=max_depth)
    sys.stdout.write(json.dumps(serialize_decomp(result), indent=2, sort_keys=True) + "\n")
    sys.stderr.write(
        f"\n{result.total_leaves} conditions: "
        f"{result.solvable_leaves} solvable, "
        f"{result.unsolvable_leaves} blocking\n"
    )
    for b in result.blockers:
        sys.stderr.write(f"  blocker: {b.expression} — {b.blocker_reason}\n")
    return 0


def _cmd_analyze(file: str, line: int, *, branch: str = "then") -> int:
    from tharkun.guard_analyzer import analyze_blocker, serialize_analysis

    result = analyze_blocker(file, line, execute=True)
    sys.stdout.write(json.dumps(serialize_analysis(result), indent=2, sort_keys=True) + "\n")
    decomp = result.decomposition
    sys.stderr.write(
        f"\nGuard: {result.guard_source}\n"
        f"Decomposition: {decomp.total_leaves} conditions, "
        f"{decomp.solvable_leaves} solvable, "
        f"{decomp.unsolvable_leaves} blocking\n"
    )
    if result.executed:
        sys.stderr.write(f"Runtime snapshots: {result.snapshot_count}\n")
    for diag in result.diagnostics:
        sys.stderr.write(f"\nBlocker: {diag.expression}")
        if diag.blocker_reason:
            sys.stderr.write(f" — {diag.blocker_reason}")
        sys.stderr.write("\n")
        if diag.relevant_vars:
            from tharkun.guard_analyzer import _format_snap_value
            sys.stderr.write("  Snapshot:\n")
            for k, v in diag.relevant_vars.items():
                sys.stderr.write(f"    {k} = {_format_snap_value(v)}\n")
        if diag.suggestion:
            sys.stderr.write(f"  Suggestion: {diag.suggestion}\n")
    return 0


def _cmd_branches(file: str, *, execute_snapshots: bool, max_depth: int) -> int:
    from tharkun.branch_solver import solve_uncovered_branches, serialize_coverage_result

    result = solve_uncovered_branches(
        file,
        execute_snapshots=execute_snapshots,
        max_depth=max_depth,
    )
    sys.stdout.write(json.dumps(serialize_coverage_result(result), indent=2, sort_keys=True) + "\n")
    sys.stderr.write(
        f"\n{result.solved_branches}/{result.total_branches} branches solved "
        f"({100 * result.solved_branches / max(result.total_branches, 1):.0f}%)\n"
    )
    for sol in result.solutions:
        status = "SOLVED" if sol.solved else "BLOCKED"
        techniques = [a.technique for a in sol.attempts if a.success]
        sys.stderr.write(
            f"  line {sol.lineno}({sol.desired_branch}): {status} [{sol.guard_source[:60]}]"
        )
        if techniques:
            sys.stderr.write(f" via {techniques}")
        sys.stderr.write("\n")
    return 0


def _cmd_emit(file: str, output: str | None, *, execute_snapshots: bool, max_depth: int, module_name: str | None) -> int:
    from tharkun.branch_solver import solve_uncovered_branches
    from tharkun.emit import emit_branch_tests, emit_branch_tests_to_file

    coverage = solve_uncovered_branches(
        file,
        execute_snapshots=execute_snapshots,
        max_depth=max_depth,
    )
    if output:
        emit_branch_tests_to_file(coverage, output, module_name=module_name)
        sys.stderr.write(f"\nWrote tests to {output}\n")
    else:
        source = emit_branch_tests(coverage, module_name=module_name)
        sys.stdout.write(source)

    solved = sum(1 for s in coverage.solutions if s.solved)
    sys.stderr.write(
        f"{solved}/{coverage.total_branches} branches → test functions\n"
    )
    return 0


def _cmd_mocks(file: str, function: str | None) -> int:
    from tharkun.mock_gen import analyze_mocking_needs, serialize_mock_plan

    fn_names = [function] if function else None
    plans = analyze_mocking_needs(file, fn_names)
    obj = {"plans": [serialize_mock_plan(p) for p in plans]}
    sys.stdout.write(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    sys.stderr.write(f"\n{len(plans)} functions with external calls\n")
    for p in plans:
        cats = {e.category for e in p.externals}
        sys.stderr.write(f"  {p.callable_qualname}: {', '.join(sorted(cats))}\n")
        for e in p.externals:
            sys.stderr.write(f"    @patch(\"{e.target}\")\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tharkun",
        description="Python branch guard solver and analysis toolkit.",
    )
    parser.add_argument("--version", action="version", version="tharkun 0.1.0")
    sub = parser.add_subparsers(dest="cmd")

    # solve
    sv = sub.add_parser("solve", help="Solve a single branch guard.")
    sv.add_argument("file", help="Path to Python source file.")
    sv.add_argument("line", type=int, help="Line number of the branch guard.")
    sv.add_argument("--branch", choices=["then", "else"], default="then",
                    help="Which edge to target: 'then' (guard truthy) or 'else' (guard falsy).")

    # decompose
    dc = sub.add_parser("decompose", help="Decompose a guard into sub-conditions.")
    dc.add_argument("file", help="Path to Python source file.")
    dc.add_argument("line", type=int, help="Line number of the branch guard.")
    dc.add_argument("--max-depth", type=int, default=5, help="Max recursion depth (default: 5).")

    # analyze
    an = sub.add_parser("analyze", help="Analyze a guard with runtime snapshots.")
    an.add_argument("file", help="Path to Python source file.")
    an.add_argument("line", type=int, help="Line number of the branch guard.")
    an.add_argument("--branch", choices=["then", "else"], default="then",
                    help="Which edge to target: 'then' (guard truthy) or 'else' (guard falsy).")

    # branches
    br = sub.add_parser("branches", help="Solve ALL branches in a file aggressively.")
    br.add_argument("file", help="Path to Python source file.")
    br.add_argument("--no-execute", action="store_true", help="Skip runtime snapshot capture.")
    br.add_argument("--max-depth", type=int, default=5, help="Max recursion depth (default: 5).")

    # emit
    em = sub.add_parser("emit", help="Generate pytest tests for all solvable branches.")
    em.add_argument("file", help="Path to Python source file.")
    em.add_argument("-o", "--output", default=None, help="Output file path (stdout if omitted).")
    em.add_argument("--no-execute", action="store_true", help="Skip runtime snapshot capture.")
    em.add_argument("--max-depth", type=int, default=5, help="Max recursion depth (default: 5).")
    em.add_argument("--module", default=None, help="Module name for imports (auto-detected if omitted).")

    # mocks
    mk = sub.add_parser("mocks", help="Detect external calls that need mocking.")
    mk.add_argument("file", help="Path to Python source file.")
    mk.add_argument("function", nargs="?", default=None, help="Specific function to analyze.")

    # bench
    bn = sub.add_parser("bench", help="Benchmark solve rate on target files.")
    bn.add_argument("files", nargs="*", help="Python files to benchmark (self-benchmark if omitted).")
    bn.add_argument("--verify", action="store_true", help="Verify by executing with solved inputs.")

    args = parser.parse_args(argv)

    if not args.cmd:
        parser.print_help()
        return 1

    if args.cmd == "solve":
        return _cmd_solve(args.file, args.line, branch=args.branch)
    if args.cmd == "decompose":
        return _cmd_decompose(args.file, args.line, max_depth=args.max_depth)
    if args.cmd == "analyze":
        return _cmd_analyze(args.file, args.line, branch=args.branch)
    if args.cmd == "branches":
        return _cmd_branches(
            args.file,
            execute_snapshots=not args.no_execute,
            max_depth=args.max_depth,
        )
    if args.cmd == "emit":
        return _cmd_emit(
            args.file, args.output,
            execute_snapshots=not args.no_execute,
            max_depth=args.max_depth,
            module_name=args.module,
        )
    if args.cmd == "mocks":
        return _cmd_mocks(args.file, args.function)
    if args.cmd == "bench":
        from tharkun.bench import run_benchmark, self_benchmark, format_report
        if args.files:
            result = run_benchmark(args.files, verify=args.verify)
        else:
            result = self_benchmark(verify=args.verify)
        sys.stderr.write(format_report(result) + "\n")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
