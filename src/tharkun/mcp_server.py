"""Tharkun MCP server — expose branch solving as MCP tools.

Powered by the ``mcp`` package's server class (optional dependency): mcp 2.x
renamed ``FastMCP`` to ``mcp.server.mcpserver.MCPServer``; we import
``MCPServer`` first and fall back to the mcp 1.x ``FastMCP`` name so this
module works unmodified against either major version. Both classes take the
same ``name``/``instructions=`` constructor args and expose the same
``tool()`` decorator and ``run()`` method, which is all this module uses.

Run as a stdio server::

    tharkun-mcp                       # after pip install tharkun[mcp]
    python -m tharkun.mcp_server

Configure in Claude Desktop / Claude Code::

    {"mcpServers": {"tharkun": {"command": "tharkun-mcp"}}}

Or without installing as a script::

    {"mcpServers": {"tharkun": {"command": "python", "args": ["-m", "tharkun.mcp_server"]}}}

The server communicates over stdio using JSON-RPC 2.0 (the MCP protocol).
Claude spawns it as a subprocess; tool calls arrive as requests, results go
back as responses.
"""
from __future__ import annotations

import os

try:
    # mcp>=2.0: FastMCP was renamed to MCPServer (mcp.server.fastmcp no
    # longer exists in 2.x).
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:
    try:
        # mcp<2.0: the original FastMCP name/location.
        from mcp.server.fastmcp import FastMCP
    except ImportError as _mcp_err:
        raise ImportError(
            "The 'mcp' package is required for the Tharkun MCP server. "
            "Install it with: pip install tharkun[mcp]"
        ) from _mcp_err

_server = FastMCP(
    "tharkun",
    instructions=(
        "Tharkun solves Python branch guard conditions. "
        "Use solve_guard to find inputs that satisfy or falsify a specific if/while/for guard. "
        "Use find_branch_edges to list all guards in a file. "
        "Use solve_all_branches for full-file solving. "
        "Use decompose_guard to understand compound guards. "
        "Use map_call_chain to trace how arguments flow to a guard. "
        "Use analyze_mocks to detect external calls that need mocking."
    ),
)


def _safe_call(fn):
    """Wrap a zero-arg callable; return {'error': str(e)} on any exception."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


@_server.tool()
def solve_guard(
    source_path: str,
    lineno: int,
    desired_branch: str = "then",
    execute_snapshots: bool = False,
) -> dict:
    """Solve a single branch guard at a specific line in a Python source file.

    Applies up to 10 techniques (static inference, runtime snapshots, minleaf
    mutation, Hypothesis search) and returns the first
    satisfying set of inputs.

    Args:
        source_path:       Absolute path to the Python source file.
        lineno:            Line number of the if/while/for guard.
        desired_branch:    "then" to satisfy the guard (make it True);
                           "else" to falsify it (make it False).
        execute_snapshots: If True, execute the code to capture runtime state.
                           Slower but more accurate for complex guards.

    Returns:
        JSON dict with:
        - solved (bool): whether satisfying inputs were found
        - best_inputs (dict|null): the satisfying inputs
        - best_inputs_constraints (dict|null): structured constraints on those inputs
        - guard_source (str): the guard expression
        - explanation (str): human-readable rationale (from condition_solver)
        - blocker_info (dict|null): structured blocker reason when unsolved
        - attempts (list): which techniques were tried and their outcomes
    """
    from tharkun.branch_solver import solve_branch, serialize_branch_solution
    return _safe_call(lambda: serialize_branch_solution(
        solve_branch(
            os.path.realpath(source_path),
            lineno,
            desired_branch=desired_branch,
            execute_snapshots=execute_snapshots,
        )
    ))


@_server.tool()
def find_branch_edges(source_path: str) -> list:
    """List all branch edges in a Python file.

    Returns two entries per guard line (one "then", one "else") for every
    if/while/for statement. This is the complete set of edges that
    solve_all_branches would target.

    Args:
        source_path: Absolute path to the Python source file.

    Returns:
        List of {lineno, stmt_kind, desired_branch} dicts.
    """
    from tharkun.branch_solver import _find_branch_edges
    return _safe_call(lambda: [
        {"lineno": e.lineno, "stmt_kind": e.stmt_kind, "desired_branch": e.desired_branch}
        for e in _find_branch_edges(os.path.realpath(source_path))
    ])


@_server.tool()
def solve_all_branches(
    source_path: str,
    execute_snapshots: bool = False,
    max_depth: int = 5,
) -> dict:
    """Solve every branch edge in a source file using all available techniques.

    Applies the full 10-technique chain to every if/while/for edge (both then
    and else) and returns aggregated results.  For large files this can be slow;
    prefer solve_guard for targeted solving.

    Args:
        source_path:       Absolute path to the Python source file.
        execute_snapshots: If True, run the code for runtime snapshots.
        max_depth:         Maximum guard decomposition depth (default 5).

    Returns:
        JSON dict with total_branches, solved_branches, unsolved_branches,
        and a solutions list (one entry per edge).
    """
    from tharkun.branch_solver import solve_uncovered_branches, serialize_coverage_result
    return _safe_call(lambda: serialize_coverage_result(
        solve_uncovered_branches(
            os.path.realpath(source_path),
            execute_snapshots=execute_snapshots,
            max_depth=max_depth,
        )
    ))


@_server.tool()
def decompose_guard(
    source_path: str,
    lineno: int,
    max_depth: int = 5,
) -> dict:
    """Decompose a compound guard into its atomic sub-conditions.

    Recursively traces into called functions to build a tree showing which
    sub-conditions are solvable and which are blocking (external calls,
    attribute dependencies, always-false constraints).

    Useful for understanding why a guard is hard to solve before attempting
    full solving.

    Args:
        source_path: Absolute path to the Python source file.
        lineno:      Line number of the guard.
        max_depth:   Maximum recursion depth into called functions (default 5).

    Returns:
        JSON dict with guard_source, solvable_leaves, unsolvable_leaves,
        blockers list (with expression and reason), and a recursive tree.
    """
    from tharkun.solve.decompose import decompose_guard as _dg, serialize_decomp
    return _safe_call(lambda: serialize_decomp(
        _dg(os.path.realpath(source_path), lineno, max_depth=max_depth)
    ))


@_server.tool()
def map_call_chain(
    source_path: str,
    target_line: int,
    entry_function: str | None = None,
    max_depth: int = 10,
) -> list:
    """Trace how arguments flow from a caller to a target line.

    Walks the call chain backward from the target line to find which caller
    parameters map to the guard's variables at each hop. Useful for backward
    constraint propagation: knowing which caller arguments end up in a guard
    allows solving the entry function's inputs instead.

    Args:
        source_path:    Absolute path to the Python source file.
        target_line:    Line number of the target guard or call site.
        entry_function: Outermost entry function name. None = auto-detect.
        max_depth:      Maximum chain depth (default 10).

    Returns:
        Ordered list of call-site mappings from outermost to innermost call,
        each with call_line, callee_name, callee_module, and args list.
        Each arg entry has callee_param, callee_position, caller_name
        (the variable name in the caller), caller_expr_source (the full
        expression), and is_simple_name (True when the arg is a bare variable).
    """
    from tharkun.solve.frontend.call_mapper import map_call_chain as _mc

    def _run() -> list:
        chain = _mc(
            os.path.realpath(source_path),
            target_line,
            entry_function=entry_function,
            max_depth=max_depth,
        )
        return [
            {
                "call_line": m.call_line,
                "callee_name": m.callee_name,
                "callee_module": m.callee_module,
                "args": [
                    {
                        "callee_param": a.callee_param,
                        "callee_position": a.callee_position,
                        "caller_name": a.caller_name,
                        "caller_expr_source": a.caller_expr_source,
                        "is_simple_name": a.is_simple_name,
                    }
                    for a in m.args
                ],
            }
            for m in chain
        ]

    return _safe_call(_run)


@_server.tool()
def analyze_mocks(
    source_path: str,
    function_name: str | None = None,
) -> dict:
    """Detect external calls that need mocking for isolated testing.

    Finds HTTP (requests, urllib, httpx), filesystem (open, os.path, shutil),
    database (sqlite3, psycopg2), and checksum (hashlib, hmac, zlib) calls,
    and returns suggested @patch decorator targets with proposed return values.

    Args:
        source_path:    Absolute path to the Python source file.
        function_name:  Specific function to analyze. None = analyze all functions.

    Returns:
        JSON dict with a "plans" list, one entry per function analyzed.
        Each plan has the function name, detected external calls, and
        a patches list with target strings ready for @patch.
    """
    from tharkun.mock_gen import analyze_mocking_needs, serialize_mock_plan

    def _run() -> dict:
        fn_names = [function_name] if function_name else None
        plans = analyze_mocking_needs(os.path.realpath(source_path), function_names=fn_names)
        return {"plans": [serialize_mock_plan(p) for p in plans]}

    return _safe_call(_run)


def main() -> None:
    """Entry point for the Tharkun MCP server (stdio transport)."""
    _server.run()


if __name__ == "__main__":
    main()
