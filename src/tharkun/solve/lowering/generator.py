"""Immediate generator consumers, retaining each yield/consumer boundary."""
from __future__ import annotations

import ast

from .comprehension import _Refuse
from .provenance import PATH_ATTR, stand_for
from .scopes import _own_scope_locals, rename_names, target_names


def consume(lowerer, gen, mode, *, limit=None):
    """Boolean consumers stop every loop; unpack reads at most arity + one.

    StopIteration inside the generator becomes RuntimeError. Consumer truth
    tests run outside those handlers, as they do after a native yield.
    """
    names = lowerer.names
    required = {"StopIteration", "RuntimeError"}
    if limit is not None:
        required.add("len")
    if mode == "unpack":
        required.add("iter")
    if "__del__" in names.bound or any(not names.stable_builtin(n) for n in required):
        raise _Refuse("generator: finalizer or rebound protocol builtin")
    if frame_sensitive_super(lowerer, gen):
        raise _Refuse("generator: implicit-frame super or an escaped super binding")
    if possibly_unbound_local(lowerer, gen):
        raise _Refuse("generator: possibly unbound captured local changes its exception class")
    if any(isinstance(n, (ast.NamedExpr, ast.Await, ast.Yield, ast.YieldFrom)) for n in ast.walk(gen)):
        raise _Refuse("generator: assignment/suspension in its body")
    if any(g.is_async or target_names(g.target) is None for g in gen.generators):
        raise _Refuse("generator: asynchronous or effectful target")
    for g in gen.generators[1:]:
        # GET_ITER of an inner iterable runs in the generator frame. Its
        # StopIteration must be wrapped; builtin containers cannot raise it.
        if not builtin_container(lowerer, g.iter):
            raise _Refuse("generator: nested iterable has an unknown __iter__")

    def mark(node, anchor=gen, role="generator"):
        return stand_for(node, anchor, role)

    def ref(name, anchor=gen):
        return lowerer._name(name, anchor, "generator")

    def step(body, anchor):
        error = names.fresh("stop")
        handler = ast.ExceptHandler(type=ref("StopIteration", anchor), name=error, body=[mark(ast.Raise(
            exc=mark(ast.Call(func=ref("RuntimeError", anchor),
                             args=[mark(ast.Constant("generator raised StopIteration"), anchor)], keywords=[]), anchor),
            cause=ref(error, anchor)), anchor)])
        return mark(ast.Try(body=body, handlers=[mark(handler, anchor)], orelse=[], finalbody=[]), anchor, "generator-step")

    mapping = {}
    for g in gen.generators:
        for name in sorted(target_names(g.target)):
            if name not in mapping:
                mapping[name] = names.fresh(name, renames=name, scope=getattr(gen, PATH_ATTR, ()))
    acc, item = names.fresh("acc"), names.fresh("item")
    boolean = mode in {"all", "any"}
    init = lowerer._assign(acc, mark(ast.Constant(mode == "all") if boolean else ast.List(elts=[], ctx=ast.Load())), gen, "generator")
    pre, value = lowerer.expr(rename_names(gen.elt, mapping))
    body = [step([*pre, lowerer._assign(item, value, gen.elt, "generator")], gen.elt)]
    if boolean:
        test = ref(item)
        if mode == "all":
            test = mark(ast.UnaryOp(op=ast.Not(), operand=test))
        body.append(mark(ast.If(test=test, body=[
            lowerer._assign(acc, mark(ast.Constant(mode == "any")), gen, "generator"),
            mark(ast.Break())], orelse=[])))
    else:
        body.append(mark(ast.Expr(value=mark(ast.Call(
            func=mark(ast.Attribute(value=ref(acc), attr="append", ctx=ast.Load())),
            args=[ref(item)], keywords=[])))))
        if limit is not None:
            body.append(mark(ast.If(test=mark(ast.Compare(
                left=mark(ast.Call(func=ref("len"), args=[ref(acc)], keywords=[])),
                ops=[ast.Gt()], comparators=[mark(ast.Constant(limit))])),
                body=[mark(ast.Break())], orelse=[])))

    def stopped():
        if boolean:
            test = ref(acc)
            return mark(ast.UnaryOp(op=ast.Not(), operand=test)) if mode == "all" else test
        return mark(ast.Compare(left=mark(ast.Call(func=ref("len"), args=[ref(acc)], keywords=[])),
                                ops=[ast.Gt()], comparators=[mark(ast.Constant(limit))]))

    for index in reversed(range(len(gen.generators))):
        g = gen.generators[index]
        for condition in reversed(g.ifs):
            cond = names.fresh("filter")
            pre, expression = lowerer.expr(rename_names(condition, mapping))
            # Keep Boolean context: bool(a and b) would truth-test a twice
            # when a is false, whereas a generator's filter tests it once.
            evaluate = mark(ast.If(test=expression, body=[lowerer._assign(cond,
                mark(ast.Constant(True), condition), condition, "generator")],
                orelse=[lowerer._assign(cond, mark(ast.Constant(False), condition), condition, "generator")]), condition)
            body = [step([*pre, evaluate], condition), mark(ast.If(test=ref(cond, condition), body=body, orelse=[]), condition)]
        target = rename_names(g.target, mapping)
        if not isinstance(target, ast.Name):
            temporary = names.fresh("unpack")
            body.insert(0, step([mark(ast.Assign(targets=[target], value=ref(temporary)))], g.target))
            target = ref(temporary)
            target.ctx = ast.Store()
        iterable = g.iter if index == 0 else rename_names(g.iter, mapping)
        pre = []
        if index:
            iterable_pre, value = lowerer.expr(iterable)
            saved = names.fresh("iterable")
            pre = [step([*iterable_pre, lowerer._assign(saved, value, g.iter, "generator")], g.iter)]
            iterable = ref(saved)
        loop = mark(ast.For(target=target, iter=iterable, body=body, orelse=[], type_comment=None), g.target)
        body = [*pre, loop]
        if index and (boolean or limit is not None):
            body.append(mark(ast.If(test=stopped(), body=[mark(ast.Break())], orelse=[])))
    lowerer.lowered += 1
    result = ref(acc)
    if mode == "unpack":
        # CPython's list fast path includes a length in excess-value errors;
        # a generator and a list iterator use the generic iterator message.
        result = mark(ast.Call(func=ref("iter"), args=[result], keywords=[]))
    return lowerer.block([init, *body], lowerer.scope), result


def builtin_container(lowerer, node):
    if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict, ast.ListComp, ast.SetComp, ast.DictComp)):
        return True
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id in {"list", "tuple", "set", "frozenset", "dict", "range"}
            and lowerer.names.stable_builtin(node.func.id))


def frame_sensitive_super(lowerer, gen):
    """An implicit super() observes the generator frame; aliases can do so too."""
    cached = getattr(lowerer, "_generator_super", None)
    if cached is None:
        root = lowerer.source_tree
        aliases = {"super"}
        aliases.update(a.asname or a.name for n in ast.walk(root) if isinstance(n, ast.ImportFrom)
                       and n.module == "builtins" for a in n.names if a.name == "super")
        parents = {c: n for n in ast.walk(root) for c in ast.iter_child_nodes(n)}
        escaped = any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id in aliases
                      and not (isinstance(parents.get(n), ast.Call) and parents[n].func is n)
                      for n in ast.walk(root))
        cached = lowerer._generator_super = aliases, escaped
    aliases, escaped = cached
    return escaped or any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in aliases
                          and (len(n.args) not in (1, 2) or n.keywords or any(isinstance(a, ast.Starred) for a in n.args))
                          for n in ast.walk(gen))


def possibly_unbound_local(lowerer, gen):
    """Removing a frame changes an unbound cell's NameError to UnboundLocalError."""
    if lowerer.scope != "function":
        return False
    path = getattr(gen, PATH_ATTR, ())
    node, function, statement_index = lowerer.source_tree, None, None
    try:
        for field, index in path:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and field == "body":
                function, statement_index = node, index
            value = getattr(node, field)
            node = value if index is None else value[index]
    except (AttributeError, IndexError, TypeError):
        return True
    if function is None or statement_index is None or not isinstance(node, ast.GeneratorExp):
        return True
    local = _own_scope_locals(function)
    arguments = function.args
    bound = {p.arg for p in [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                             arguments.vararg, arguments.kwarg] if p is not None}
    for statement in function.body[:statement_index]:
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            if isinstance(statement, ast.Assign) or statement.value is not None:
                bound.update(name for target in targets for name in (target_names(target) or ()))
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(statement.name)
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            bound.update(a.asname or a.name.split('.')[0] for a in statement.names)
    cleared = {n.id for n in ast.walk(function) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Del)}
    cleared.update(n.name for n in ast.walk(function) if isinstance(n, ast.ExceptHandler) and n.name)
    loop_names = set().union(*(target_names(g.target) or set() for g in gen.generators))
    reads = {n.id for n in ast.walk(gen) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return bool((reads - loop_names) & local & ((local - bound) | cleared))


def integer_generator(lowerer, gen):
    """Sum may materialize only exact, effect-free integer terms over ranges."""
    bound = set()
    for g in gen.generators:
        if (not isinstance(g.iter, ast.Call) or not isinstance(g.iter.func, ast.Name)
                or g.iter.func.id != "range" or not lowerer.names.stable_builtin("range")):
            return False
        bound.update(target_names(g.target) or ())
    def integer(node):
        if isinstance(node, ast.Constant):
            return type(node.value) in (int, bool)
        if isinstance(node, ast.Name):
            return node.id in bound
        if isinstance(node, ast.UnaryOp):
            return isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert, ast.Not)) and integer(node.operand)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                return (integer(node.left) and isinstance(node.right, ast.Constant)
                        and type(node.right.value) in (int, bool) and node.right.value >= 0)
            return (isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod,
                                        ast.LShift, ast.RShift, ast.BitOr, ast.BitAnd, ast.BitXor))
                    and integer(node.left) and integer(node.right))
        if isinstance(node, ast.Compare):
            return (all(isinstance(op, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)) for op in node.ops)
                    and all(integer(v) for v in (node.left, *node.comparators)))
        if isinstance(node, ast.BoolOp):
            return all(integer(v) for v in node.values)
        return False
    return integer(gen.elt)
