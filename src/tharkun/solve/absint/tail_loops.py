"""Proof-only loops for primitive, frame-unobservable direct tail recursion."""
import ast
import copy


DRAWS = {'nondet_bool', 'nondet_int', 'nondet_float', 'nondet_str'}


def prepare(function):
    """Return (body, globals, verifier modules), or None for an ordinary call."""
    if not isinstance(function, ast.FunctionDef):
        return None
    args = function.args
    if args.vararg or args.kwarg or args.kwonlyargs:
        return None
    parameters = [a.arg for a in args.posonlyargs + args.args]
    if function.name in parameters:
        return None
    allowed = (ast.Assert, ast.If, ast.Return, ast.Expr, ast.Pass)
    calls = set()
    modules = set()
    names = set()
    for node in ast.walk(ast.Module(function.body, [])):
        if isinstance(node, ast.stmt) and not isinstance(node, allowed):
            return None
        if isinstance(node, ast.Name):
            if not isinstance(node.ctx, ast.Load):
                return None
            names.add(node.id)
        if isinstance(node, (ast.Lambda, ast.NamedExpr, ast.ListComp, ast.SetComp, ast.DictComp,
                             ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom, ast.Starred,
                             ast.Subscript, ast.List, ast.Tuple, ast.Dict, ast.Set)):
            return None
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.attr in DRAWS):
                return None
            modules.add(node.value.id)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == function.name:
                if node.keywords or len(node.args) != len(parameters):
                    return None
                calls.add(id(node))
            elif (isinstance(node.func, ast.Attribute) and node.func.attr in DRAWS
                  and isinstance(node.func.value, ast.Name) and not node.args and not node.keywords):
                pass
            else:
                return None
    if not calls or not any(isinstance(n, ast.Return) and id(n.value) not in calls
                            for n in ast.walk(ast.Module(function.body, []))):
        return None
    consumed = set()

    def block(statements, tail=True):
        out = []
        for index, statement in enumerate(statements):
            last = tail and index == len(statements) - 1
            if isinstance(statement, (ast.Return, ast.Expr)) and id(statement.value) in calls:
                if isinstance(statement, ast.Expr) and not last:
                    raise ValueError
                call = statement.value
                # Expr tail calls discard the callee's result; only a None-returning body can loop.
                if isinstance(statement, ast.Expr) and any(
                        isinstance(n, ast.Return) and n.value is not None
                        and not (isinstance(n.value, ast.Constant) and n.value.value is None)
                        for n in ast.walk(ast.Module(function.body, []))):
                    raise ValueError
                consumed.add(id(call))
                if parameters:
                    assignment = ast.Assign([ast.Tuple([ast.Name(p, ast.Store()) for p in parameters], ast.Store())],
                                            ast.Tuple(copy.deepcopy(call.args), ast.Load()))
                    out.append(ast.copy_location(assignment, statement))
                out.append(ast.copy_location(ast.Continue(), statement))
            elif isinstance(statement, ast.If):
                new = copy.copy(statement)
                new.body, new.orelse = block(statement.body, last), block(statement.orelse, last)
                out.append(new)
            else:
                out.append(statement)
        return out

    try:
        body = block(function.body)
    except ValueError:
        return None
    if consumed != calls:
        return None
    # A nonrecursive fallthrough returns None, rather than starting another activation.
    body.append(ast.copy_location(ast.Return(ast.Constant(None)), function.body[-1]))
    loop = ast.copy_location(ast.While(ast.Constant(True), body, []), function.body[0])
    ast.fix_missing_locations(loop)
    return [loop], names - set(parameters) - {function.name}, modules


def primitive(value):
    return (not (value.top or value.refs or value.st) and not value.is_bot()
            and set(value.kinds()) <= {'int', 'bool', 'float', 'str', 'None'})
