"""Keep a final assignment expression in a while guard inside each iteration."""

from __future__ import annotations

import ast

from .provenance import stand_for
from .scopes import COMPREHENSIONS, Names


def lower_walrus_loops(lowered) -> None:
    """Extend an origin-aware lowering in place; unsupported expressions stay intact.

    ``while prefix and (name := value)`` becomes ``while prefix`` with the
    assignment and its truth test first in the body. No fresh name or scope is
    introduced. Only uncaptured function locals are rewritten: a global/cell
    store can invoke an old value's finalizer, which can rebind that name before
    the new guard reads it. While-else stays intact because a false assigned value must
    execute its else suite, whereas a break in the original body must not.
    """
    # STORE_FAST sees one fewer live RHS reference without NamedExpr's COPY.
    # A finalizer can observe that even when it never rebinds the target.
    if (Names(lowered.tree).reflective or any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == "__del__" for n in ast.walk(lowered.tree))):
        return

    class Lower(ast.NodeTransformer):
        blocked = None

        def visit_ClassDef(self, node):
            saved, self.blocked = self.blocked, None
            self.generic_visit(node)
            self.blocked = saved
            return node

        def visit_FunctionDef(self, node):
            saved = self.blocked
            self.blocked = set()
            scopes = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, *COMPREHENSIONS)
            for child in ast.walk(node):
                if isinstance(child, (ast.Global, ast.Nonlocal)):
                    self.blocked.update(child.names)
                elif child is not node and isinstance(child, scopes):
                    self.blocked.update(n.id for n in ast.walk(child) if isinstance(n, ast.Name))
            self.generic_visit(node)
            self.blocked = saved
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_While(self, node):
            self.generic_visit(node)
            test = node.test
            # A metaclass namespace may change a name when it is stored/read.
            if self.blocked is None or node.orelse:
                return node
            if isinstance(test, ast.NamedExpr):
                value = test
                prefix = stand_for(ast.Constant(True), test, "walrus-loop")
            elif (isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
                  and isinstance(test.values[-1], ast.NamedExpr)):
                value = test.values[-1]
                before = test.values[:-1]
                prefix = (before[0] if len(before) == 1 else
                          stand_for(ast.BoolOp(ast.And(), before), test, "walrus-prefix"))
            else:
                return node
            if value.target.id in self.blocked:
                return node
            bind = stand_for(ast.Assign(targets=[value.target], value=value.value), value, "walrus-bind")
            read = stand_for(ast.Name(value.target.id, ast.Load()), value, "walrus-read")
            stop = stand_for(ast.If(
                test=stand_for(ast.UnaryOp(ast.Not(), read), value, "walrus-test"),
                body=[stand_for(ast.Break(), value, "walrus-stop")], orelse=[]), value, "walrus-test")
            node.test = prefix
            node.body[:0] = [bind, stop]
            lowered.lowered += 1
            return node

    Lower().visit(lowered.tree)
    ast.fix_missing_locations(lowered.tree)
