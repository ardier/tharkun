"""Overapproximate scalar assertion goals by deleting independent float islands.

Only closed, single-assignment scalar computations are admitted. A deleted call
has no heap/global effects or property goals; its uncaught non-property exits
may disappear, adding paths. No class, handler, reflection or unknown call is
pruned. The caller must prove every deleted assertion on the original source.
"""
from __future__ import annotations

import ast
import time
from dataclasses import dataclass

from .sitekey import site_keys
from .slicing import slice_for


class _Refused(ValueError):
    pass


@dataclass(frozen=True)
class ProductSlice:
    source: str
    retained: frozenset[str]
    erased: frozenset[str]


def slice_product_goals(source: str, filename: str, safe_assertions: frozenset[str], *,
                        deadline: float | None = None) -> ProductSlice | None:
    """Return a scalar goal overapproximation, or refuse the entire projection.

    Original AST paths identify coverage, never the line numbers of the rendered
    slice. The exact prover must prove ALL assertions left in that slice.
    """
    deadline = time.monotonic() + 1.0 if deadline is None else deadline
    try:
        tree = ast.parse(source, filename)
        keys = site_keys(tree, source, filename)
        goals = {n: keys[n].ast_path for n in tree.body if isinstance(n, ast.Assert)}
        if not goals or any(isinstance(n, ast.Assert) and n not in goals for n in ast.walk(tree)):
            return None
        functions, aliases, direct, bound = {}, set(), {}, set()
        env, erased, retained, body = {}, set(), set(), []
        reserved = {"int", "bool", "float", "str", "type", "abs", "__builtins__"}

        def bind(name):
            if name in bound or name in reserved or name.startswith("__"):
                raise _Refused("rebound or reflective name")
            bound.add(name)

        def expression(node, local, stack=()):
            if time.monotonic() >= deadline:
                raise _Refused("projection budget")
            if isinstance(node, ast.Name):
                if node.id not in local:
                    raise _Refused("nonlocal or unknown value")
                return local[node.id]
            if isinstance(node, ast.Constant) and type(node.value) in (int, bool, float, str):
                kind = type(node.value).__name__
                return kind, kind == "float"
            if isinstance(node, ast.UnaryOp):
                kind, dep = expression(node.operand, local, stack)
                if isinstance(node.op, ast.Not):
                    return "bool", dep
                if isinstance(node.op, (ast.USub, ast.UAdd)) and kind in {"int", "bool", "float"}:
                    return ("float" if kind == "float" else "int"), dep
            if isinstance(node, ast.BinOp):
                left, ld = expression(node.left, local, stack)
                right, rd = expression(node.right, local, stack)
                if isinstance(node.op, ast.Add) and left == right == "str":
                    return "str", ld or rd
                if left in {"int", "bool", "float"} and right in {"int", "bool", "float"}:
                    if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Div)):
                        floating = "float" in (left, right) or isinstance(node.op, ast.Div)
                        return ("float" if floating else "int"), ld or rd or floating
            if isinstance(node, ast.Compare):
                values = [expression(x, local, stack) for x in (node.left, *node.comparators)]
                if all(isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops):
                    return "bool", any(dep for _, dep in values)
            if isinstance(node, ast.BoolOp):
                values = [expression(x, local, stack) for x in node.values]
                if len({kind for kind, _ in values}) == 1:
                    return values[0][0], any(dep for _, dep in values)
            if isinstance(node, ast.Call) and not node.keywords:
                helper = (node.func.attr if isinstance(node.func, ast.Attribute)
                          and isinstance(node.func.value, ast.Name) and node.func.value.id in aliases else
                          direct.get(node.func.id) if isinstance(node.func, ast.Name) else None)
                if helper in {"nondet_int", "nondet_bool", "nondet_str", "nondet_float"} and not node.args:
                    kind = helper.removeprefix("nondet_")
                    return kind, kind == "float"
                if helper == "check_type" and len(node.args) == 2 and isinstance(node.args[1], ast.Name):
                    kind, dep = expression(node.args[0], local, stack)
                    hint = node.args[1].id
                    if hint in reserved and (kind == hint or (kind == "bool" and hint == "int")):
                        return "NoneType", dep
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                    args = [expression(arg, local, stack) for arg in node.args]
                    if name == "abs" and len(args) == 1 and args[0][0] in {"int", "bool", "float"}:
                        return ("float" if args[0][0] == "float" else "int"), args[0][1]
                    if name in functions and name not in stack:
                        fn = functions[name]
                        params = [*fn.args.posonlyargs, *fn.args.args]
                        if len(params) != len(args):
                            raise _Refused("call arity")
                        frame = {param.arg: value for param, value in zip(params, args)}
                        statements = fn.body
                        if (statements and isinstance(statements[0], ast.Expr)
                                and isinstance(statements[0].value, ast.Constant)
                                and type(statements[0].value.value) is str):
                            statements = statements[1:]
                        for st in statements[:-1]:
                            if (not isinstance(st, ast.Assign) or len(st.targets) != 1
                                    or not isinstance(st.targets[0], ast.Name) or st.targets[0].id in frame):
                                raise _Refused("callee effect or rebinding")
                            frame[st.targets[0].id] = expression(st.value, frame, (*stack, name))
                        if statements and isinstance(statements[-1], ast.Return) and statements[-1].value is not None:
                            return expression(statements[-1].value, frame, (*stack, name))
            raise _Refused("unmodelled scalar effect")

        for st in tree.body:
            if isinstance(st, ast.Import) and all(a.name == "_sv_verifier" for a in st.names):
                for alias in st.names:
                    name = alias.asname or alias.name
                    bind(name)
                    aliases.add(name)
            elif isinstance(st, ast.ImportFrom) and st.module == "_sv_verifier" and not st.level:
                for alias in st.names:
                    if alias.name not in {"nondet_int", "nondet_bool", "nondet_str", "nondet_float", "check_type"}:
                        raise _Refused("foreign helper")
                    name = alias.asname or alias.name
                    bind(name)
                    direct[name] = alias.name
            elif isinstance(st, ast.FunctionDef):
                if (st.decorator_list or st.returns is not None or st.args.defaults or st.args.kw_defaults
                        or st.args.kwonlyargs or st.args.vararg or st.args.kwarg or getattr(st, "type_params", ())
                        or any(a.annotation is not None for a in (*st.args.posonlyargs, *st.args.args))):
                    raise _Refused("eager or unsupported definition")
                bind(st.name)
                functions[st.name] = st
            elif isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                name = st.targets[0].id
                bind(name)
                env[name] = expression(st.value, env)
                if env[name][1]:
                    continue
            elif isinstance(st, ast.Assert):
                _, floating = expression(st.test, env)
                if floating:
                    if goals[st] not in safe_assertions or st.msg is not None:
                        raise _Refused("unproved float goal")
                    erased.add(goals[st])
                    continue
                retained.add(goals[st])
            elif isinstance(st, ast.Expr):
                if expression(st.value, env)[1]:
                    continue
            else:
                raise _Refused("statement/control effect")
            body.append(st)
        # A useful projection must actually remove source, not retry ordinary product.
        if not retained or len(body) == len(tree.body):
            return None
        body = slice_for(body, set(), closed=True)
        return ProductSlice(ast.unparse(ast.Module(body=body, type_ignores=[])),
                            frozenset(retained), frozenset(erased))
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
