"""Fixed local source-object dispatch, using the inventory's lexical metadata.

These are possible source-call edges, not a normal-prefix/effects certificate.
The existing operation/constructor unknowns remain until their routes are proved.
"""
from __future__ import annotations

import ast

from tharkun.solve.source_arguments import bind_source_arguments


_BINARY = {"Add": "add", "Sub": "sub", "Mult": "mul", "MatMult": "matmul",
           "Div": "truediv", "FloorDiv": "floordiv", "Mod": "mod", "Pow": "pow",
           "LShift": "lshift", "RShift": "rshift", "BitAnd": "and", "BitOr": "or", "BitXor": "xor"}
_UNARY = {"UAdd": "__pos__", "USub": "__neg__", "Invert": "__invert__"}


def source_dispatch(index, scope, owner, node):
    """Return (method AST, bound operands, defaulted names, source-class ID) edges.

    A fresh no-argument default construction immediately precedes the operation.
    An annotation, isinstance test or inherited/custom constructor is not a type proof.
    """
    if node is None or index.opaque_bindings or index.opaque_class_identities:
        return []
    statement = index.parents.get(node)
    body = getattr(owner, "body", ())
    if (not isinstance(body, list) or not isinstance(statement, (ast.Return, ast.Assign, ast.AnnAssign, ast.Expr))
            or statement.value is not node or statement not in body or body.index(statement) == 0):
        return []
    previous = body[body.index(statement) - 1]
    candidates = []
    if isinstance(node, ast.BinOp) and (suffix := _BINARY.get(type(node.op).__name__)):
        candidates = [(node.left, "__" + suffix + "__", [node.right]),
                      (node.right, "__r" + suffix + "__", [node.left])]
    elif isinstance(node, ast.UnaryOp) and (method := _UNARY.get(type(node.op).__name__)):
        candidates = [(node.operand, method, [])]
    elif isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load) and not isinstance(node.slice, ast.Slice):
        candidates = [(node.value, "__getitem__", [node.slice])]
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            return []
        candidates = [(node.func.value, node.func.attr, node.args)]
    edges = []
    for receiver, method, arguments in candidates:
        if not isinstance(receiver, ast.Name) or method in index.assigned_attributes:
            continue
        bindings = index.bindings[scope["id"]].get(receiver.id, ())
        if (len(bindings) != 1 or not isinstance(previous, ast.Assign) or len(previous.targets) != 1
                or previous.targets[0] is not bindings[0] or not isinstance(previous.value, ast.Call)):
            continue
        constructor = previous.value
        if not isinstance(constructor.func, ast.Name) or constructor.args or constructor.keywords:
            continue
        cls = index.resolve(constructor.func.id, scope["id"])
        if (not isinstance(cls, ast.ClassDef) or cls not in body or cls.lineno >= previous.lineno
                or cls.bases or cls.keywords or cls.decorator_list or getattr(cls, "type_params", ())):
            continue
        methods = {part.name: part for part in cls.body if isinstance(part, ast.FunctionDef)}
        if len(methods) != sum(isinstance(part, ast.FunctionDef) for part in cls.body):
            continue
        if any(name in methods for name in ("__new__", "__init__")):
            continue
        if isinstance(node, ast.Call) and any(name in methods for name in ("__getattribute__", "__getattr__")):
            continue
        if any(not (isinstance(part, (ast.FunctionDef, ast.Pass)) or isinstance(part, ast.Expr)
                    and isinstance(part.value, ast.Constant) and isinstance(part.value.value, str)) for part in cls.body):
            continue
        if any(part.decorator_list or part.args.defaults or any(value is not None for value in part.args.kw_defaults)
               for part in methods.values()):
            continue
        target = methods.get(method)
        if target is None:
            continue
        bound, defaulted, reason = bind_source_arguments(target, [None, *arguments],
                                                        node.keywords if isinstance(node, ast.Call) else ())
        if reason is not None:
            continue
        class_id = index.definitions[cls].removesuffix(".<body>")
        edges.append((target, bound, defaulted, class_id))
    return edges
