"""Authenticate a returned additive local capture, then make its snapshot explicit.

Only direct calls can observe the result. The ordinary whole-program audit still
rejects heap captures, unsupported arguments and every unlowered function value.
"""
import ast
import copy

from tharkun.sv.exact_decorators import _plain
from tharkun.sv.exact_unroll import _bindings


def lower_returned_additive_closures(tree):
    if any(isinstance(n, (ast.List, ast.Tuple, ast.Dict, ast.Set, ast.Subscript,
                         ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)) for n in ast.walk(tree)):
        return  # newer container consumers must not broaden the original scalar-only capture domain
    counts = _bindings(tree)
    used = set(counts) | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    def fresh(name):
        while name in used:
            name += "_"
        used.add(name)
        return name

    for fd in list(tree.body):
        if not (isinstance(fd, ast.FunctionDef) and not fd.decorator_list and _plain(fd)
                and counts.get(fd.name) == 1 and len(fd.args.args) == 1 and len(fd.body) == 1
                and isinstance(fd.body[0], ast.Return) and isinstance(fd.body[0].value, ast.Lambda)):
            continue
        lam = fd.body[0].value
        a = lam.args
        if (len(a.args) != 1 or a.posonlyargs or a.kwonlyargs or a.defaults or a.kw_defaults
                or a.vararg or a.kwarg or a.args[0].annotation):
            continue
        capture, argument = fd.args.args[0].arg, a.args[0].arg
        body = lam.body
        if not (capture != argument and isinstance(body, ast.BinOp) and isinstance(body.op, ast.Add)
                and isinstance(body.left, ast.Name) and isinstance(body.right, ast.Name)
                and {body.left.id, body.right.id} == {capture, argument}):
            continue
        creators = [st for st in tree.body if isinstance(st, ast.Assign) and len(st.targets) == 1
                    and isinstance(st.targets[0], ast.Name) and isinstance(st.value, ast.Call)
                    and isinstance(st.value.func, ast.Name) and st.value.func.id == fd.name
                    and len(st.value.args) == 1 and not st.value.keywords
                    and not isinstance(st.value.args[0], ast.Starred)]
        # shortcut: one unaliased result per factory; extend when a measured task needs multiple closures.
        if len(creators) != 1:
            continue
        creator = creators[0]
        references = {id(n) for n in ast.walk(tree) if isinstance(n, ast.Name)
                      and isinstance(n.ctx, ast.Load) and n.id == fd.name}
        if references != {id(creator.value.func)} or tree.body.index(creator) <= tree.body.index(fd):
            continue
        target = creator.targets[0].id
        if counts.get(target) != 1:
            continue
        calls = [n for st in tree.body if not isinstance(st, ast.FunctionDef) for n in ast.walk(st)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == target]
        references = {id(n) for n in ast.walk(tree) if isinstance(n, ast.Name)
                      and isinstance(n.ctx, ast.Load) and n.id == target}
        if not calls or references != {id(n.func) for n in calls}:
            continue
        if any(n.keywords or len(n.args) != 1 or isinstance(n.args[0], ast.Starred)
               or n.lineno <= creator.end_lineno for n in calls):
            continue
        helper = copy.deepcopy(fd)
        helper.name = fresh("tharkun_closure_body")
        parameter = fresh("tharkun_closure_parameter")
        value = fresh("tharkun_closure_value")
        helper.args = copy.deepcopy(a)
        cap_arg = copy.deepcopy(fd.args.args[0])
        cap_arg.arg = parameter
        helper.args.args.append(cap_arg)
        expression = copy.deepcopy(body)
        for node in ast.walk(expression):
            if isinstance(node, ast.Name) and node.id == capture:
                node.id = parameter
        helper.body = [ast.copy_location(ast.Return(value=expression), fd.body[0])]
        for call in calls:
            call.func.id = helper.name
            call.args.append(ast.copy_location(ast.Name(id=value, ctx=ast.Load()), call))
        snapshot = ast.copy_location(ast.Assign(targets=[ast.Name(id=value, ctx=ast.Store())],
                                               value=creator.value.args[0]), creator)
        # Preserve both source bindings for the usual rebind/builtin audit; neither value may escape.
        creator.value = ast.copy_location(ast.Constant(value=0), creator.value)
        tree.body.insert(tree.body.index(creator), snapshot)
        fd.body = [ast.copy_location(ast.Return(value=ast.Name(id=capture, ctx=ast.Load())), fd.body[0])]
        tree.body.insert(tree.body.index(fd) + 1, helper)
    ast.fix_missing_locations(tree)
