"""Guard extraction and the shared guard-AST helpers.

One home for "find the guard at a line" (#0606, plan #0600): ``extract_frontier``
(the minleaf frontier extractor, formerly ``tharkun.minleaf.guard_extract``) and
the lighter ``find_branch_at_line`` / ``free_names`` / ``unparse`` /
``unwrap_named_expr`` / ``try_extract_constant`` helpers that lived as private
names in ``condition_solver``.  ``tharkun.solve.condition`` re-exports the latter
under their old underscore names for existing callers.

Supported statement types: ast.If, ast.While, ast.Assert.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Frontier:
    source_path: str
    target_line: int
    stmt_kind: str              # "if" | "while" | "assert"
    guard_expr: ast.expr
    guard_src: str
    lineno: int
    col_offset: int
    end_lineno: int | None
    end_col_offset: int | None


def extract_frontier(source_path: str, target_line: int) -> Frontier:
    """Extract the guard AST from a source file at a given line number.

    Raises ValueError if no supported statement is found at target_line.
    """
    with open(source_path, "r") as f:
        source_text = f.read()

    tree = ast.parse(source_text, filename=source_path)

    # Collect all If/While/Assert nodes with depth info
    candidates: list[tuple[int, ast.stmt, ast.expr, str]] = []
    # (depth, stmt_node, test_expr, stmt_kind)

    def _visit(node: ast.AST, depth: int) -> None:
        """Visit a single AST node, checking if it's a target statement."""
        if isinstance(node, ast.If):
            _handle_if(node, depth)
        elif isinstance(node, ast.While):
            if node.lineno == target_line:
                candidates.append((depth, node, node.test, "while"))
            for sub in node.body + node.orelse:
                _visit(sub, depth + 1)
        elif isinstance(node, ast.Assert):
            if node.lineno == target_line:
                candidates.append((depth, node, node.test, "assert"))
        else:
            for child in ast.iter_child_nodes(node):
                _visit(child, depth)

    def _handle_if(node: ast.If, depth: int) -> None:
        """Handle an If node, including elif chains."""
        # Check if this if/elif statement starts at the target line
        if node.lineno == target_line:
            candidates.append((depth, node, node.test, "if"))

        # Recurse into body
        for sub in node.body:
            _visit(sub, depth + 1)

        # Handle orelse: elif chain or else block
        for sub in node.orelse:
            if isinstance(sub, ast.If):
                # elif: same depth as the parent if
                _handle_if(sub, depth)
            else:
                _visit(sub, depth + 1)

    _visit(tree, 0)

    if not candidates:
        raise ValueError(
            f"No supported statement (if/while/assert) found at line {target_line} "
            f"in {source_path}"
        )

    # Select the innermost (highest depth) candidate; tiebreak by line number (latest)
    candidates.sort(key=lambda c: (c[0], c[1].lineno))
    best_depth, best_stmt, best_expr, best_kind = candidates[-1]

    # Recover source text
    guard_src = ast.get_source_segment(source_text, best_expr)
    if guard_src is None:
        guard_src = ast.unparse(best_expr)

    return Frontier(
        source_path=source_path,
        target_line=target_line,
        stmt_kind=best_kind,
        guard_expr=best_expr,
        guard_src=guard_src,
        lineno=best_expr.lineno,
        col_offset=best_expr.col_offset,
        end_lineno=getattr(best_expr, "end_lineno", None),
        end_col_offset=getattr(best_expr, "end_col_offset", None),
    )


def desired_guard_value(stmt_kind: str, desired_branch: str) -> bool:
    """Map (stmt_kind, desired_branch) to the desired boolean guard value.

    "then" always means True, "else" always means False.
    Raises ValueError for unknown desired_branch.
    """
    if desired_branch == "then":
        return True
    elif desired_branch == "else":
        return False
    else:
        raise ValueError(f"Unknown desired_branch: {desired_branch!r}")


# ---------------------------------------------------------------------------
# Shared guard-AST helpers (moved from condition_solver, #0606)
# ---------------------------------------------------------------------------


def unparse(node: ast.AST) -> str | None:
    try:
        return ast.unparse(node)
    except Exception:
        return None


def find_branch_at_line(tree: ast.Module, lineno: int) -> ast.expr | None:
    """Find the test expression of an if/while/for at *lineno*."""
    for node in ast.walk(tree):
        ln = getattr(node, "lineno", None)
        if ln != lineno:
            continue
        if isinstance(node, (ast.If, ast.While)):
            return node.test
        if isinstance(node, ast.For):
            return node.iter
    return None


def free_names(expr: ast.expr) -> set[str]:
    """Return all Name nodes in *expr* (the free variables).

    Also extracts targets from walrus operators (``NamedExpr``, Python 3.8+).
    Filters out comprehension-scoped loop variables (e.g. ``x`` in
    ``all(x > 0 for x in items)``).
    """
    names: set[str] = set()
    comp_vars: set[str] = set()
    _NamedExpr = getattr(ast, "NamedExpr", None)
    for node in ast.walk(expr):
        if isinstance(node, ast.Name):
            names.add(node.id)
        if _NamedExpr is not None and isinstance(node, _NamedExpr):
            target = getattr(node, "target", None)
            if isinstance(target, ast.Name):
                names.add(target.id)
        if isinstance(node, ast.comprehension):
            if isinstance(node.target, ast.Name):
                comp_vars.add(node.target.id)
    return names - comp_vars


def unwrap_named_expr(node: ast.AST) -> ast.AST:
    """If *node* is a ``NamedExpr`` (walrus ``:=``), return its ``.target``.

    Otherwise return *node* unchanged.  This lets comparison/value extraction
    see through walrus operators transparently.
    """
    _NamedExpr = getattr(ast, "NamedExpr", None)
    if _NamedExpr is not None and isinstance(node, _NamedExpr):
        return node.target
    return node


def try_extract_constant(node: ast.AST) -> tuple[Any, bool]:
    """Extract a constant value from an AST node, handling negative literals.

    ``-100`` in AST is ``UnaryOp(USub(), Constant(100))``, not ``Constant(-100)``.
    Returns ``(value, True)`` if extractable, ``(None, False)`` otherwise.
    """
    if isinstance(node, ast.Constant):
        return node.value, True
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))):
        return -node.operand.value, True
    folded, ok = _fold_pure_call(node)
    if ok:
        return folded, True
    return None, False


# Calls that only build a literal from literals.  A guard writes
# ``b == bytes.fromhex("00")`` as naturally as ``b == b"\x00"``, and refusing
# the first because it is syntactically a Call loses a fact that is right
# there.  The rule is deliberately narrow: the callee must be one of these
# exact names, every argument must be a literal, and evaluation happens in an
# empty namespace -- no target module is imported, no attribute of a user
# object is touched, nothing user-defined runs.
_PURE_METHODS = {
    ("bytes", "fromhex"): bytes.fromhex,
    ("bytearray", "fromhex"): bytearray.fromhex,
}
_PURE_BUILTINS = {"bytes": bytes, "bytearray": bytearray, "str": str,
                  "frozenset": frozenset, "tuple": tuple, "complex": complex}
# ``bytes(1_000_000_000)`` is a literal call that allocates a gigabyte, so a
# size argument is refused outright; the conversions we want take a literal
# sequence, never a length.
_MAX_FOLD_LEN = 4096


def _fold_pure_call(node: ast.AST) -> tuple[Any, bool]:
    """Evaluate a literal-only call to a known-pure constructor."""
    if not isinstance(node, ast.Call) or node.keywords:
        return None, False
    fn = None
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        fn = _PURE_METHODS.get((node.func.value.id, node.func.attr))
    elif isinstance(node.func, ast.Name):
        fn = _PURE_BUILTINS.get(node.func.id)
    if fn is None:
        return None, False
    args = []
    for a in node.args:
        try:
            v = ast.literal_eval(a)
        except Exception:
            return None, False
        if isinstance(v, int) and not isinstance(v, bool):
            return None, False        # a length, not a value: refuse to allocate
        if isinstance(v, (str, bytes, bytearray, list, tuple, set, frozenset)) and len(v) > _MAX_FOLD_LEN:
            return None, False
        args.append(v)
    try:
        out = fn(*args)
    except Exception:
        return None, False            # a malformed literal is not a fact
    if isinstance(out, bytearray):
        out = bytes(out)              # hashable, so it can live in a value set
    try:
        hash(out)
    except TypeError:
        return None, False
    return out, True
