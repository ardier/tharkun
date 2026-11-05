"""Must-identity facts for plain module copies; source syntax and reads stay intact."""
from __future__ import annotations

import ast
import time


MUST_IDENTITY_ATTR = '_lw_must_copy_identity'


def annotate_module_copies(tree, *, deadline):
    """Mark copies for closed absint execution, whose hook gates exclude target finalizers."""
    def block(statements, env):
        for statement in statements:
            if time.process_time() >= deadline:
                return {}
            if (isinstance(statement, ast.Assign)
                    and all(isinstance(t, ast.Name) for t in statement.targets)):
                value = statement.value
                token = env.get(value.id) if isinstance(value, ast.Name) else None
                if not isinstance(value, (ast.Name, ast.Constant)):
                    env.clear()
                if token is None:
                    token = object()
                for target in statement.targets:
                    env[target.id] = token
            elif isinstance(statement, ast.If):
                # Even truth conversion can run user code and delete a global binding.
                left = block(statement.body, {})
                right = block(statement.orelse, {})
                env = {name: object() for name in left.keys() & right.keys()}
            elif isinstance(statement, ast.Assert):
                test = statement.test
                if (isinstance(test, ast.Compare) and len(test.ops) == 1
                        and isinstance(test.ops[0], (ast.Is, ast.IsNot))
                        and isinstance(test.left, ast.Name) and isinstance(test.comparators[0], ast.Name)
                        and test.left.id in env
                        and env[test.left.id] is env.get(test.comparators[0].id)):
                    setattr(test, MUST_IDENTITY_ATTR, True)
                else:
                    env.clear()
            elif isinstance(statement, ast.Pass):
                continue
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, (ast.Name, ast.Constant)):
                continue
            else:
                # Calls, stores, deletes, loops, handlers and nested scopes may invalidate copies.
                env.clear()
        return env

    if isinstance(tree, ast.Module):
        block(tree.body, {})
