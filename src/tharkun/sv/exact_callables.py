"""Static closed-def flow for the scalar exact-product auditor; never runs source."""
import ast

from tharkun.sv.exact_unroll import _bindings


def _lambda_def(name, lam, at):
    return ast.copy_location(ast.FunctionDef(
        name=name, args=lam.args, decorator_list=[], returns=None, type_comment=None, type_params=[],
        body=[ast.copy_location(ast.Return(value=lam.body), lam.body)]), at)


def _plain_lambda(lam):
    """A lambda a def can stand for: no yield/await/walrus in the body, no defaults (a default is evaluated where the
    lambda is created, a def hoisted elsewhere would evaluate it elsewhere)."""
    a = lam.args
    return not (a.defaults or a.kw_defaults) and not any(
        isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await, ast.NamedExpr)) for n in ast.walk(lam.body))


def _own_names(fn):
    """Every name ``fn`` binds in its own scope (parameters and stores): what a lambda inside it could capture."""
    a = fn.args
    names = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
    names.update(x.arg for x in (a.vararg, a.kwarg) if x)
    for st in fn.body:
        for n in ast.walk(st):
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                names.add(n.id)
            elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(n.name)
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                names.update((x.asname or x.name).split(".")[0] for x in n.names)
            elif isinstance(n, ast.arg):
                names.add(n.arg)
    return names


def _closed_in(lam, captured):
    """``lam`` reads nothing of the enclosing function: its free names are globals or builtins, bound at call time exactly
    as the same lambda moved to module level would bind them."""
    params = {x.arg for x in lam.args.posonlyargs + lam.args.args + lam.args.kwonlyargs}
    params.update(x.arg for x in (lam.args.vararg, lam.args.kwarg) if x)
    free = {n.id for n in ast.walk(lam.body) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - params
    return not (free & captured)


def lift_named_lambdas(tree):
    """Give every lambda bound once to a name the def it stands for, in place.

    * ``f = lambda ...: e`` at module level, ``f`` bound exactly once in the program, becomes ``def f(...): return e``: the
      same function value at the same point, same parameters and late-bound free names.  They differ only in
      ``__name__``/``repr``, which the exact subset cannot observe (it refuses attributes and every use of a function as
      a value).
    * The same inside a module-level function, when the lambda reads nothing the function binds: the def is hoisted to module
      level under a fresh name and the local is bound to it (``f = _tharkun_lambda_N``).  Creating a closed lambda has no
      effect, so where it is created does not matter.
    * ``ops = {"a": lambda ..., "b": lambda ...}`` (distinct constant keys, closed lambdas, ``ops`` bound once) whose every
      other occurrence is the callee of ``ops["a"](...)`` with a key of the display: each such callee IS that lambda's
      hoisted def, and the display (which nothing else can read, mutate or pass on) goes.

    A name read at or before its own binding (recursion, use before binding) keeps the lambda, so a NameError at that
    point is never turned into a call.  Whatever the audit then refuses about a def it refuses here too.
    """
    count = _bindings(tree)
    used = {value for node in ast.walk(tree) for _, value in ast.iter_fields(node) if isinstance(value, str)}
    hoisted = []
    swap: dict = {}              # id(``ops[k]`` callee) -> the hoisted def standing for it

    def fresh():
        name = f"_tharkun_lambda_{len(hoisted)}"
        while name in used:
            name += "_"
        used.add(name)
        return name

    def nodes():                  # the program as it stands: what is left in the tree plus the defs already hoisted out of it
        yield from ast.walk(tree)
        for h in hoisted:
            yield from ast.walk(h)

    def read_early(name, st):
        return any(isinstance(n, ast.Name) and n.id == name and n.lineno <= st.end_lineno and not (
            isinstance(n.ctx, ast.Store) and n.lineno == st.lineno) for n in nodes())

    def scope(body, captured):
        for i, st in enumerate(body):
            if not (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)):
                continue
            name, value = st.targets[0].id, st.value
            if count.get(name) != 1 or read_early(name, st):
                continue
            if isinstance(value, ast.Lambda) and _plain_lambda(value):
                if captured is None:
                    body[i] = _lambda_def(name, value, st)
                elif _closed_in(value, captured):
                    h = fresh()
                    hoisted.append(_lambda_def(h, value, st))
                    body[i] = ast.copy_location(
                        ast.Assign(targets=st.targets, value=ast.copy_location(ast.Name(h, ast.Load()), value)), st)
            elif isinstance(value, ast.Dict) and value.keys:
                keys = [k.value if isinstance(k, ast.Constant) and isinstance(k.value, (str, int)) else None
                        for k in value.keys]
                if (None in keys or len(set(keys)) != len(keys) or len({type(k) for k in keys}) != 1
                        or not all(isinstance(v, ast.Lambda) and _plain_lambda(v) for v in value.values)
                        or (captured is not None and not all(_closed_in(v, captured) for v in value.values))):
                    continue
                callees = [c.func for c in nodes() if isinstance(c, ast.Call)
                           and isinstance(c.func, ast.Subscript) and isinstance(c.func.value, ast.Name)
                           and c.func.value.id == name and isinstance(c.func.slice, ast.Constant)
                           and c.func.slice.value in keys]
                loads = sum(1 for n in nodes() if isinstance(n, ast.Name) and n.id == name
                            and isinstance(n.ctx, ast.Load))
                if loads != len(callees) or any(type(c.slice.value) is not type(keys[0]) for c in callees):
                    continue
                defs = {}
                for k, lam in zip(keys, value.values):
                    defs[k] = fresh()
                    hoisted.append(_lambda_def(defs[k], lam, st))
                for c in callees:
                    swap[id(c)] = defs[c.slice.value]
                body[i] = ast.copy_location(ast.Pass(), st)

    scope(tree.body, None)
    for fn in tree.body[:]:
        if isinstance(fn, ast.FunctionDef):
            scope(fn.body, _own_names(fn))
    if swap:
        class Swap(ast.NodeTransformer):
            def visit_Subscript(self, node):
                if id(node) in swap:
                    return ast.copy_location(ast.Name(swap[id(node)], ast.Load()), node)
                return self.generic_visit(node)
        Swap().visit(tree)
        for h in hoisted:
            Swap().visit(h)
    tree.body[:0] = hoisted


def lift_scalar_lambdas(tree):
    """Lift closed literal callback arguments at once-only module sites into defs.

    Fresh def names are unobservable in the exact scalar subset: dynamic namespace
    access and callable metadata/escapes remain outside its existing audit.
    """
    used = {value for node in ast.walk(tree) for _, value in ast.iter_fields(node) if isinstance(value, str)}
    used.update(name for node in ast.walk(tree) if isinstance(node, (ast.Global, ast.Nonlocal)) for name in node.names)
    defs = []

    class Lift(ast.NodeTransformer):
        def visit_Lambda(self, node):
            return node

        def visit_Call(self, node):
            self.generic_visit(node)
            for i, value in enumerate(node.args):
                if not isinstance(value, ast.Lambda):
                    continue
                args = value.args
                if (len(args.args) != 1 or args.posonlyargs or args.kwonlyargs or args.defaults
                        or args.vararg or args.kwarg or args.args[0].annotation):
                    continue
                reads = {n.id for n in ast.walk(value.body) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                if (reads - {args.args[0].arg} or any(isinstance(n, (ast.Lambda, ast.NamedExpr, ast.Yield,
                        ast.YieldFrom, ast.Await)) for n in ast.walk(value.body))):
                    continue
                name = f"_tharkun_scalar_lambda_{len(defs)}"
                while name in used:
                    name += "_"
                used.add(name)
                fd = ast.FunctionDef(name=name, args=args,
                    body=[ast.copy_location(ast.Return(value=value.body), value.body)],
                    decorator_list=[], returns=None, type_comment=None, type_params=[])
                defs.append(ast.copy_location(fd, value))
                node.args[i] = ast.copy_location(ast.Name(id=name, ctx=ast.Load()), value)
            return node

    lift = Lift()

    def block(statements):
        for node in statements:
            if isinstance(node, (ast.Assign, ast.Expr, ast.Assert)):
                lift.visit(node)
            elif isinstance(node, ast.If):
                lift.visit(node.test)
                block(node.body)
                block(node.orelse)
    block(tree.body)
    tree.body[:0] = defs


def binding(fd, call):
    """Parameter positions in Python's argument evaluation order, or an unsupported binding."""
    a = fd.args
    names = [p.arg for p in a.args]
    if (a.defaults or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg
            or len(call.args) > len(names) or any(isinstance(v, ast.Starred) for v in call.args)):
        return None
    bound = dict(zip(names, range(len(call.args))))
    for i, keyword in enumerate(call.keywords, len(call.args)):
        if keyword.arg not in names or keyword.arg in bound:
            return None
        bound[keyword.arg] = i
    return [bound[name] for name in names] if len(bound) == len(names) else None


class ClosedCallables:
    """Possible source defs of aliases/parameters, used only to restrict admission.

    Execution retains each assignment's actual guarded value. This finite union
    analysis never substitutes an alias or equates versions/draws.
    """

    def __init__(self, tree):
        self.defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.bodies = {None: [n for n in tree.body if not isinstance(n, ast.FunctionDef)]}
        self.bodies.update({name: fd.body for name, fd in self.defs.items()})
        self.nodes = {owner: [n for st in body for n in ast.walk(st)] for owner, body in self.bodies.items()}
        self.locals = {owner: {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
                       for owner, nodes in self.nodes.items()}
        for name, fd in self.defs.items():
            self.locals[name].update(p.arg for p in fd.args.args)
        self.values = {owner: {} for owner in self.bodies}
        changed = True
        while changed:
            changed = False
            for owner, nodes in self.nodes.items():
                for n in nodes:
                    if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                        changed |= self._add(owner, n.targets[0].id, self.targets(n.value, owner))
                    elif isinstance(n, ast.Call):
                        arguments = [*n.args, *(k.value for k in n.keywords)]
                        for name in self.targets(n.func, owner):
                            fd = self.defs[name]
                            positions = binding(fd, n)
                            if positions is not None:
                                for param, index in zip(fd.args.args, positions):
                                    changed |= self._add(name, param.arg, self.targets(arguments[index], owner))
        self.active = any(self.values.values())

    def _add(self, owner, name, targets):
        old = self.values[owner].get(name, frozenset())
        new = old | targets
        if new == old:
            return False
        self.values[owner][name] = new
        return True

    def targets(self, node, owner):
        if isinstance(node, ast.Name):
            if owner is not None and node.id in self.locals[owner]:
                return self.values[owner].get(node.id, frozenset())
            if node.id in self.defs:
                return frozenset({node.id})
            return self.values[None].get(node.id, frozenset())
        if isinstance(node, ast.IfExp):
            left, right = self.targets(node.body, owner), self.targets(node.orelse, owner)
            return left | right if left and right else frozenset()
        return frozenset()

    def name_targets(self, name, owner):
        return self.targets(ast.Name(id=name, ctx=ast.Load()), owner)
