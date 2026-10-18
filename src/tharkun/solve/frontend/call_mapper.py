"""Call-site argument mapping — extracts callee-param-to-caller-expression
mappings for backward constraint propagation.

Given a call site like ``process(data.strip())``, this module extracts:
- The callee's parameter name (``data``)
- The caller's expression (``data.strip()``)
- Whether the argument is a simple variable pass-through

Used by glamdring to propagate constraints backward through call chains.

Usage::

    from tharkun.solve.frontend.call_mapper import map_call_site, map_call_chain

    # Map a single call site
    mapping = map_call_site("file.py", 10)

    # Walk an entire call chain
    chain = map_call_chain("file.py", target_line=42)
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tharkun.solve.frontend.call_walker import (
    CallIndex,
    build_call_index,
    extract_scope_annotations,
    extract_scope_isinstance,
    find_enclosing_class,
    resolve_callee,
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class ArgMapping:
    """One callee parameter mapped to the caller's argument expression.

    Attributes:
        callee_param:       Parameter name in the callee function.
        callee_position:    Positional index (0-based).
        caller_expr:        The raw AST expression the caller passes.
        caller_expr_source: Unparsed source of the caller expression.
        is_simple_name:     True if the expression is just ``ast.Name``.
        caller_name:        If ``is_simple_name``, the variable name; else None.
    """
    callee_param: str
    callee_position: int
    caller_expr: ast.expr
    caller_expr_source: str
    is_simple_name: bool
    caller_name: str | None


@dataclass
class CallSiteMapping:
    """Complete mapping for one call site.

    Attributes:
        call_line:       Line number of the call expression.
        callee_name:     Resolved (or display) name of the called function.
        callee_module:   Resolved module path if cross-module, else None.
        callee_func_def: Resolved AST FunctionDef, or None if unresolvable.
        args:            Per-argument mappings.
    """
    call_line: int
    callee_name: str
    callee_module: str | None
    callee_func_def: ast.FunctionDef | ast.AsyncFunctionDef | None
    args: list[ArgMapping]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_callee_params(
    func_def: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[str]:
    """Extract parameter names, skipping ``self`` and ``cls``."""
    return [
        arg.arg for arg in func_def.args.args
        if arg.arg not in ("self", "cls")
    ]


def _build_arg_mappings(
    call_expr: ast.Call,
    callee_params: list[str],
) -> list[ArgMapping]:
    """Build ArgMapping list from a call expression and callee param names."""
    mappings: list[ArgMapping] = []

    # Positional arguments.
    for i, arg_expr in enumerate(call_expr.args):
        if i >= len(callee_params):
            break  # More args than params (e.g., *args).
        param_name = callee_params[i]
        is_simple = isinstance(arg_expr, ast.Name)
        mappings.append(ArgMapping(
            callee_param=param_name,
            callee_position=i,
            caller_expr=arg_expr,
            caller_expr_source=ast.unparse(arg_expr),
            is_simple_name=is_simple,
            caller_name=arg_expr.id if is_simple else None,
        ))

    # Keyword arguments.
    for kw in call_expr.keywords:
        if kw.arg is None:
            continue  # **kwargs — skip.
        if kw.arg not in callee_params:
            continue  # Unknown keyword.
        position = callee_params.index(kw.arg)
        is_simple = isinstance(kw.value, ast.Name)
        mappings.append(ArgMapping(
            callee_param=kw.arg,
            callee_position=position,
            caller_expr=kw.value,
            caller_expr_source=ast.unparse(kw.value),
            is_simple_name=is_simple,
            caller_name=kw.value.id if is_simple else None,
        ))

    return mappings


def _extract_call_name(call_expr: ast.Call) -> str | None:
    """Extract a display name from a call expression."""
    if isinstance(call_expr.func, ast.Name):
        return call_expr.func.id
    if isinstance(call_expr.func, ast.Attribute):
        if isinstance(call_expr.func.value, ast.Name):
            return f"{call_expr.func.value.id}.{call_expr.func.attr}"
        return call_expr.func.attr
    return None


def _find_calls_at_line(
    tree: ast.Module,
    line: int,
) -> list[ast.Call]:
    """Find all ast.Call nodes at the given line number."""
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node, "lineno", None) == line:
            calls.append(node)
    return calls


def _find_func_containing_line(
    tree: ast.Module,
    lineno: int,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Find the function definition that contains a given line."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = node.lineno
        end = getattr(node, "end_lineno", None) or start
        if start <= lineno <= end:
            # Check for a more specific (nested) function.
            for child in ast.walk(node):
                if child is node:
                    continue
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    cs = child.lineno
                    ce = getattr(child, "end_lineno", None) or cs
                    if cs <= lineno <= ce:
                        return child
            return node
    return None


def _parse_source(source_path: str) -> tuple[ast.Module | None, str]:
    """Parse a source file, return (tree, realpath)."""
    rp = os.path.realpath(source_path)
    try:
        source = Path(rp).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return None, rp
    return tree, rp


def _build_mapping_for_call(
    call_expr: ast.Call,
    index: CallIndex,
    tree: ast.Module,
    enclosing_func: ast.FunctionDef | ast.AsyncFunctionDef | None = None,
) -> CallSiteMapping | None:
    """Resolve a call expression and build a CallSiteMapping."""
    enclosing_class = None
    scope_isinstance = None
    scope_annotations = None

    if enclosing_func is not None:
        enclosing_class = find_enclosing_class(tree, enclosing_func)
        scope_isinstance = extract_scope_isinstance(enclosing_func)
        scope_annotations = extract_scope_annotations(enclosing_func)

    resolved_def, qname = resolve_callee(
        call_expr, index,
        enclosing_class=enclosing_class,
        scope_isinstance=scope_isinstance,
        scope_annotations=scope_annotations,
    )

    display_name = qname or _extract_call_name(call_expr)
    if display_name is None:
        return None

    # Determine callee module.
    callee_module = None
    if qname and index.imports:
        # Check if the resolved function came from an import.
        base_name = qname.split(".")[0]
        callee_module = index.imports.get(base_name) or index.imports.get(qname)

    # Build argument mappings.
    args: list[ArgMapping] = []
    if resolved_def is not None:
        callee_params = _extract_callee_params(resolved_def)
        args = _build_arg_mappings(call_expr, callee_params)

    return CallSiteMapping(
        call_line=getattr(call_expr, "lineno", 0),
        callee_name=display_name,
        callee_module=callee_module,
        callee_func_def=resolved_def,
        args=args,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_call_site(
    source_path: str,
    call_line: int,
    *,
    callee_name: str | None = None,
) -> CallSiteMapping | None:
    """Extract argument mapping for a call expression at the given line.

    Resolves the callee via ``call_walker`` and ``import_resolver``.
    Returns ``None`` if no call is found at the line or resolution fails.

    Args:
        source_path: Path to the Python source file.
        call_line:   Line number of the call expression.
        callee_name: Optional filter — if multiple calls on the same line,
                     only return the one matching this name.
    """
    tree, rp = _parse_source(source_path)
    if tree is None:
        return None

    index = build_call_index(tree, source_path=rp)
    calls = _find_calls_at_line(tree, call_line)
    if not calls:
        return None

    enclosing_func = _find_func_containing_line(tree, call_line)

    for call_expr in calls:
        if callee_name is not None:
            name = _extract_call_name(call_expr)
            if name != callee_name and not (name and name.endswith(f".{callee_name}")):
                continue

        mapping = _build_mapping_for_call(call_expr, index, tree, enclosing_func)
        if mapping is not None:
            return mapping

    return None


def find_calls_to(
    source_path: str,
    callee_name: str,
) -> list[CallSiteMapping]:
    """Find all call sites in *source_path* that call *callee_name*.

    Useful for backward walking: given a function, find all places it's
    called from, and extract the argument mappings at each call site.
    """
    tree, rp = _parse_source(source_path)
    if tree is None:
        return []

    index = build_call_index(tree, source_path=rp)
    results: list[CallSiteMapping] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _extract_call_name(node)
        if name != callee_name and not (name and name.endswith(f".{callee_name}")):
            continue

        enclosing_func = _find_func_containing_line(tree, getattr(node, "lineno", 0))
        mapping = _build_mapping_for_call(node, index, tree, enclosing_func)
        if mapping is not None:
            results.append(mapping)

    return results


def map_call_chain(
    source_path: str,
    target_line: int,
    *,
    entry_function: str | None = None,
    max_depth: int = 10,
) -> list[CallSiteMapping]:
    """Walk the call chain from *entry_function* down to *target_line*.

    Returns an ordered list of :class:`CallSiteMapping` from outermost
    call to innermost.  Each entry describes how arguments flow from
    caller to callee at that hop.

    If *entry_function* is None, uses the module-level function that
    contains the first call eventually reaching *target_line*.

    Uses ``import_resolver`` for cross-module call resolution.
    Cap at *max_depth* to prevent infinite recursion.
    """
    tree, rp = _parse_source(source_path)
    if tree is None:
        return []

    index = build_call_index(tree, source_path=rp)

    # Find the function containing target_line.
    target_func = _find_func_containing_line(tree, target_line)
    if target_func is None:
        return []
    target_func_name = target_func.name

    # If no entry_function, find a module-level function that calls
    # (transitively) the target function.
    if entry_function is None:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node is not target_func:
                    # Check if this function has any call in its body.
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            name = _extract_call_name(child)
                            if name == target_func_name or (
                                name and index.func_defs.get(name) is not None
                            ):
                                entry_function = node.name
                                break
                if entry_function:
                    break
        if entry_function is None:
            return []

    # DFS from entry function to target function.
    entry_def = index.func_defs.get(entry_function)
    if entry_def is None:
        return []

    chain: list[CallSiteMapping] = []
    visited: set[str] = set()

    def _dfs(
        func_def: ast.FunctionDef | ast.AsyncFunctionDef,
        func_name: str,
        current_tree: ast.Module,
        current_index: CallIndex,
        current_rp: str,
        depth: int,
    ) -> bool:
        """Return True if we found a path to target_func_name."""
        if depth > max_depth:
            return False
        if func_name in visited:
            return False
        visited.add(func_name)

        for node in ast.walk(func_def):
            if not isinstance(node, ast.Call):
                continue

            call_name = _extract_call_name(node)
            if call_name is None:
                continue

            mapping = _build_mapping_for_call(
                node, current_index, current_tree, func_def,
            )
            if mapping is None:
                continue

            # Did we reach the target?
            if call_name == target_func_name or mapping.callee_name == target_func_name:
                chain.append(mapping)
                return True

            # Recurse into the callee if resolved.
            if mapping.callee_func_def is not None:
                callee_name = mapping.callee_name
                chain.append(mapping)
                if _dfs(
                    mapping.callee_func_def, callee_name,
                    current_tree, current_index, current_rp,
                    depth + 1,
                ):
                    return True
                chain.pop()  # Backtrack.

        visited.discard(func_name)
        return False

    _dfs(entry_def, entry_function, tree, index, rp, 0)
    return chain
