"""Cross-function type propagation.

When ``f(x)`` calls ``g(x)`` and ``g`` has type annotations for that
parameter, we can propagate the type back to ``f``'s parameter even
if ``f`` has no annotations.

This module builds a lightweight callgraph from AST analysis and
propagates type information backwards through call edges.

Usage::

    from tharkun.solve.frontend.type_propagation import propagate_types

    types = propagate_types("module.py", "my_function")
    # {"x": "int", "data": "list"}
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any

from tharkun.solve.frontend.import_resolver import load_function_def, parse_imports


def _extract_annotations(func_def: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, str]:
    """Extract parameter annotations from a function definition."""
    result: dict[str, str] = {}
    for arg in func_def.args.args:
        if arg.arg in ("self", "cls"):
            continue
        if arg.annotation:
            try:
                result[arg.arg] = ast.unparse(arg.annotation)
            except Exception:
                pass
    return result


def _find_call_edges(
    func_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[str, dict[str, str]]]:
    """Find function calls inside func_def that pass parameters through.

    Returns a list of (callee_name, {callee_param_pos: caller_param_name}).
    For example, if f(x) calls g(x), returns [("g", {"0": "x"})].
    """
    edges: list[tuple[str, dict[str, str]]] = []

    # Collect parameter names
    param_names = {arg.arg for arg in func_def.args.args if arg.arg not in ("self", "cls")}

    for node in ast.walk(func_def):
        if not isinstance(node, ast.Call):
            continue

        # Resolve callee name
        callee_name = None
        if isinstance(node.func, ast.Name):
            callee_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            # Skip method calls on self/objects — too complex for now
            continue
        if callee_name is None:
            continue

        # Map positional args: which caller params flow into which callee positions
        arg_map: dict[str, str] = {}
        for i, arg in enumerate(node.args):
            if isinstance(arg, ast.Name) and arg.id in param_names:
                arg_map[str(i)] = arg.id

        # Map keyword args
        for kw in node.keywords:
            if kw.arg and isinstance(kw.value, ast.Name) and kw.value.id in param_names:
                arg_map[kw.arg] = kw.value.id

        if arg_map:
            edges.append((callee_name, arg_map))

    return edges


def propagate_types(
    source_path: str,
    function_name: str,
    *,
    max_depth: int = 3,
) -> dict[str, str]:
    """Propagate types from called functions back to the caller's parameters.

    For each parameter of ``function_name`` that lacks a type annotation,
    looks at functions called with that parameter and checks if the callee
    has annotations that can be propagated back.

    Args:
        source_path:    Path to the Python source file.
        function_name:  Name of the function to analyze.
        max_depth:      Maximum call chain depth to follow.

    Returns:
        Dict mapping parameter names to inferred type strings.
    """
    rp = os.path.realpath(source_path)

    # Load the target function
    func_def = load_function_def(rp, function_name)
    if func_def is None:
        return {}

    # Get existing annotations (these are already known)
    existing = _extract_annotations(func_def)

    # Find unannotated parameters
    unannotated = set()
    for arg in func_def.args.args:
        if arg.arg in ("self", "cls"):
            continue
        if arg.arg not in existing:
            unannotated.add(arg.arg)

    if not unannotated:
        return existing

    # Parse imports for cross-module resolution
    imports = parse_imports(rp)

    # Find call edges
    edges = _find_call_edges(func_def)

    propagated: dict[str, str] = dict(existing)

    for callee_name, arg_map in edges:
        # Try to find the callee's definition
        callee_def = load_function_def(rp, callee_name)

        # If not in same file, check imports
        if callee_def is None and callee_name in imports:
            callee_def = load_function_def(imports[callee_name], callee_name)

        if callee_def is None:
            continue

        callee_annotations = _extract_annotations(callee_def)
        callee_params = [
            arg.arg for arg in callee_def.args.args
            if arg.arg not in ("self", "cls")
        ]

        for key, caller_param in arg_map.items():
            if caller_param not in unannotated:
                continue
            if caller_param in propagated:
                continue

            # key is either a position string or a keyword name
            callee_param = None
            if key.isdigit():
                idx = int(key)
                if idx < len(callee_params):
                    callee_param = callee_params[idx]
            else:
                callee_param = key

            if callee_param and callee_param in callee_annotations:
                propagated[caller_param] = callee_annotations[callee_param]
                unannotated.discard(caller_param)

    return propagated
