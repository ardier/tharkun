"""Source-bound site identities and conservative statement pre-state bindings."""
from __future__ import annotations

import ast
import hashlib
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class SiteKey:
    source_sha256: str
    ast_path: str
    exception: str
    span: tuple[int, int, int, int]
    function: str
    statement_location: tuple[str, int, int, str]
    byte_column: int

    def row_subject(self) -> str:
        return f"{self.ast_path}:{self.exception}"

    def location(self) -> tuple[str, int, int, str]:
        return self.statement_location

    def point(self) -> tuple[str, int]:
        return os.path.realpath(self.statement_location[0]), self.statement_location[1]

    def record_key(self) -> tuple[str, str, str]:
        """Source-bound identity; raw absint coordinates alone can be ambiguous."""
        return self.source_sha256, self.ast_path, self.exception


def site_keys(tree: ast.AST, source: str, filename: str, exception: str = "") -> dict[ast.AST, SiteKey]:
    """Build once for a source AST; stamp per-check exceptions with ``dataclasses.replace``.

    Spans use one-based character columns and an exclusive end. Statement
    locations keep ``source_points``' original function/default/decorator rules.
    An exact AST node lookup distinguishes operations sharing legacy coordinates.
    """
    keys = {}
    lines = source.splitlines()
    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()

    def column(line, byte_column):
        return len(lines[line - 1].encode("utf-8")[:byte_column].decode("utf-8")) + 1

    def visit(node, path, function, statement):
        if isinstance(node, ast.stmt):
            statement = node
        if statement is not None:
            located = node if hasattr(node, "lineno") else statement
            line, col = located.lineno, located.col_offset
            end_line = getattr(located, "end_lineno", None) or line
            end_col = getattr(located, "end_col_offset", None)
            span = (line, column(line, col), end_line,
                    column(end_line, col if end_col is None else end_col))
            location = (filename, statement.lineno,
                        column(statement.lineno, statement.col_offset), function)
            keys[node] = SiteKey(sha, path, exception, span, function, location, col)
        child_function = function
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            child_function = node.name if function == "<module>" else f"{function}.{node.name}"
        for field, value in ast.iter_fields(node):
            children = enumerate(value) if isinstance(value, list) else [(None, value)]
            for i, child in children:
                if isinstance(child, ast.AST):
                    suffix = field if i is None else f"{field}[{i}]"
                    visit(child, f"{path}.{suffix}", child_function if field == "body" else function, statement)

    visit(tree, "$", "<module>", None)
    return keys


def prestate_info(node: ast.AST, parents: dict[ast.AST, ast.AST]):
    """Return ``(scope, statement, ordered_prefix)`` for a source-level atom.

    ``site`` is the historical direct-statement position. ``statement_entry``
    covers Boolean atoms with an effect-free prefix and the first operand of a
    direct comparison test; callers still prove any prefix's exact scalar types
    and safety.
    The structural classifier is shared so producers do not implement their
    own Boolean-ancestor or earlier-sibling grammar.
    """
    if isinstance(node, ast.BinOp):
        operands = (node.left, node.right)
    elif isinstance(node, ast.UnaryOp):
        operands = (node.operand,)
    elif isinstance(node, ast.Compare) and len(node.ops) == 1:
        operands = (node.left, *node.comparators)
    elif isinstance(node, ast.Subscript):
        operands = (node.value, node.slice)
    elif isinstance(node, ast.AugAssign):
        operands = (node.target, node.value)
    else:
        return None, None, ()
    if not all(isinstance(n, (ast.Name, ast.Constant)) for n in operands):
        return None, None, ()
    if isinstance(node, ast.AugAssign):
        if isinstance(node.target, ast.Name):
            return "site", parents.get(node), ()
        return None, None, ()
    parent = parents.get(node)
    if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr)):
        return ("site", parent, ()) if parent.value is node else (None, None, ())
    if isinstance(parent, (ast.If, ast.While, ast.Assert)):
        return ("site", parent, ()) if parent.test is node else (None, None, ())
    if isinstance(node, ast.BinOp) and isinstance(parent, ast.Compare) and parent.left is node:
        statement = parents.get(parent)
        if ((isinstance(statement, ast.Assert) and statement.test is parent)
                or (isinstance(statement, ast.If) and statement.test is parent)):
            return "statement_entry", statement, ()

    # Walk only through Boolean structure. Every earlier arm must itself have
    # the measured, side-effect-free scalar-comparison grammar; type/safety
    # proofs are deliberately left to the absint producer.
    cur = node
    blocks = []
    while True:
        parent = parents.get(cur)
        if isinstance(parent, ast.BoolOp) and cur in parent.values:
            blocks.append(parent.values[:parent.values.index(cur)])
            cur = parent
            continue
        if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not) and parent.operand is cur:
            cur = parent
            continue
        break
    if cur is node:
        return None, None, ()
    statement = parents.get(cur)
    if not ((isinstance(statement, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr))
             and statement.value is cur)
            or (isinstance(statement, (ast.If, ast.While, ast.Assert)) and statement.test is cur)):
        return None, None, ()

    def pure_prefix(expr):
        if (isinstance(expr, ast.Compare) and len(expr.ops) == 1
                and type(expr.ops[0]) in (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
            return all(type(part) in (ast.Name, ast.Constant) for part in (expr.left, *expr.comparators))
        if isinstance(expr, ast.BoolOp):
            return all(pure_prefix(part) for part in expr.values)
        if isinstance(expr, ast.UnaryOp) and isinstance(expr.op, ast.Not):
            return pure_prefix(expr.operand)
        return isinstance(expr, ast.Constant) and type(expr.value) is bool

    prefix = tuple(expr for block in reversed(blocks) for expr in block)
    if not all(pure_prefix(expr) for expr in prefix):
        return None, None, ()
    return "statement_entry", statement, prefix


def prestate_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str | None:
    """Compatibility wrapper returning only the state-scope classification."""
    return prestate_info(node, parents)[0]


def stable_at_prestate(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Whether operands denote the statement's entry values at a direct site.

    Nested Boolean atoms use :func:`prestate_scope`; this legacy predicate stays
    direct-only for consumers whose records have site-state semantics.
    """
    return prestate_scope(node, parents) == "site"
