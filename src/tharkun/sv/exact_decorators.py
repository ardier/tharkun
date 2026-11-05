"""Authenticate a closed forwarding decorator before treating it as inert.

Only calls of the wrapped function are observable in the exact subset: function
values, identity, attributes and arbitrary decorator applications stay refused.
The original callee body and source lines remain; nothing it executes is sliced.
"""
import ast


def _plain(fd):
    a = fd.args
    return (not fd.returns and not getattr(fd, "type_params", None)
            and not (a.posonlyargs or a.kwonlyargs or a.defaults or a.kw_defaults or a.vararg or a.kwarg)
            and not any(p.annotation for p in a.args)
            and len({p.arg for p in a.args}) == len(a.args))


def _forwarder(fd):
    if fd.decorator_list or not _plain(fd) or len(fd.args.args) != 1 or len(fd.body) != 2:
        return None
    wrapper, ret = fd.body
    if not (isinstance(wrapper, ast.FunctionDef) and not wrapper.decorator_list and _plain(wrapper)
            and isinstance(ret, ast.Return) and isinstance(ret.value, ast.Name)
            and ret.value.id == wrapper.name):
        return None
    captured = fd.args.args[0].arg
    params = [p.arg for p in wrapper.args.args]
    if captured in [wrapper.name, *params]:
        return None
    body = wrapper.body
    if len(body) == 1 and isinstance(body[0], ast.Return):
        call = body[0].value
    elif (len(body) == 2 and isinstance(body[0], ast.Assign) and len(body[0].targets) == 1
          and isinstance(body[0].targets[0], ast.Name) and isinstance(body[1], ast.Return)
          and isinstance(body[1].value, ast.Name) and body[1].value.id == body[0].targets[0].id
          and body[0].targets[0].id not in [captured, wrapper.name, *params]):
        call = body[0].value
    else:
        return None
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == captured
            and not call.keywords and len(call.args) == len(params)
            and all(isinstance(arg, ast.Name) and arg.id == param for arg, param in zip(call.args, params))):
        return None
    return len(params)


def lower_forwarding_decorators(tree):
    for index, fd in enumerate(tree.body):
        if not isinstance(fd, ast.FunctionDef):
            continue
        arity = _forwarder(fd)
        if arity is None:
            continue
        uses = [(i, node, node.decorator_list[0]) for i, node in enumerate(tree.body)
                if isinstance(node, ast.FunctionDef) and len(node.decorator_list) == 1
                and isinstance(node.decorator_list[0], ast.Name) and node.decorator_list[0].id == fd.name]
        references = {id(n) for n in ast.walk(tree) if isinstance(n, ast.Name)
                      and isinstance(n.ctx, ast.Load) and n.id == fd.name}
        if not uses or references != {id(dec) for _, _, dec in uses}:
            continue  # Calling dec(0), escaping it or stacking decorators is not forwarding authentication.
        if any(i <= index or len(node.args.args) != arity or not _plain(node) for i, node, _ in uses):
            continue
        for _, node, _ in uses:
            node.decorator_list = []
        # Keep the declaration/bindings visible to the ordinary whole-module name audit.
        fd.body = [ast.copy_location(ast.Return(value=ast.Name(id=fd.args.args[0].arg, ctx=ast.Load())), fd.body[-1])]
