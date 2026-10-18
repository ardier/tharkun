"""Private, bounded call-row proof for fresh singleton-set ``pop`` receivers.

This is an AST analysis of a closed native subset, not an observation of a
target. Exact argument rows retain correlations across source calls. Native
list results retain an element-safety guarantee, not concrete contents; they
cannot supply call rows or decide branches. No result is published until all
reachable calls, defaults and effects have been checked.
"""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from inspect import Parameter, Signature


_MAX_STATES = 256
_MAX_FUNCTION_STATES = 64
_MAX_EDGES = 1000
_MAX_DEPTH = 64
_MAX_ITEMS = 64
_MAX_VALUE_DEPTH = 16
_MAX_SET = 8
_MAX_BITS = 64


class _Refuse(Exception):
    pass


@dataclass(frozen=True)
class _Value:
    kind: str
    data: object = None


@dataclass
class _Builder:
    # Builders never escape to aliases, parameters, nested containers or
    # globals. Every append/extend proves native elements before it succeeds.
    owner: object


_NONE = _Value("none")
_LIST_RESULT = _Value("list_result")
_UNKNOWN_BOOL = _Value("bool")


def _native(value):
    if not isinstance(value, _Value):
        return False
    if value.kind in {"int", "bool", "str", "none", "list_result"}:
        return True
    return value.kind in {"list", "tuple"} and all(_native(v) for v in value.data)


def _exact(value):
    if not _native(value) or value.kind == "list_result":
        return False
    if value.kind in {"list", "tuple"}:
        return all(_exact(v) for v in value.data)
    return value.kind != "bool" or value.data is not None


def _integer(value):
    if not isinstance(value, _Value) or value.kind != "int" or type(value.data) is not int:
        raise _Refuse
    return value.data


def _int(value):
    if value.bit_length() > _MAX_BITS:
        raise _Refuse
    return _Value("int", value)


def _truth(value):
    if not _exact(value):
        raise _Refuse
    return False if value.kind == "none" else bool(value.data)


def _plain(value):
    """Analysis-owned exact data, never a target object or custom protocol."""
    if not _exact(value):
        raise _Refuse
    if value.kind in {"list", "tuple"}:
        items = [_plain(v) for v in value.data]
        return items if value.kind == "list" else tuple(items)
    return value.data


def _size(value, depth=0):
    if depth > _MAX_VALUE_DEPTH:
        raise _Refuse
    size = 1
    if value.kind in {"list", "tuple"}:
        for item in value.data:
            size += _size(item, depth + 1)
            if size > _MAX_ITEMS:
                raise _Refuse
    return size


def _fresh(node):
    if isinstance(node, ast.Set):
        return not any(isinstance(e, ast.Starred) for e in node.elts)
    return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub)
            and _fresh(node.left) and _fresh(node.right))


def singleton_receivers(resolver):
    """Return only exact receiver AST identities whose entire proof closes."""
    try:
        return _Rows(resolver).prove()
    except (_Refuse, RecursionError):
        return frozenset()


class _Rows:
    def __init__(self, resolver):
        self.r = resolver
        self.tree = resolver.tree
        self.nodes = [n for n in ast.walk(self.tree) if resolver.live(n)]
        self.definitions = {n.name: n for n in self.tree.body if isinstance(n, ast.FunctionDef)}
        self.signatures = {}
        self.globals = {}
        self.memo = {}
        self.active = set()
        self.counts = Counter()
        self.edges = 0
        self.receivers = set()
        self.builders = set()

    def prepare(self):
        # These alternate bindings/effects cannot disappear in a branch, a
        # dead definition's header, or a call with rejected arity.
        forbidden = (ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda, ast.Import,
                     ast.ImportFrom, ast.Global, ast.Nonlocal, ast.NamedExpr,
                     ast.AugAssign, ast.Delete, ast.For, ast.AsyncFor, ast.While,
                     ast.With, ast.AsyncWith, ast.Try, ast.TryStar, ast.Raise,
                     ast.Match, ast.Yield, ast.YieldFrom, ast.Await,
                     ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        writes = Counter()
        for node in self.nodes:
            if isinstance(node, forbidden):
                raise _Refuse
            if isinstance(node, ast.FunctionDef):
                if (self.r.parent.get(node) is not self.tree or node.decorator_list
                        or getattr(node, "type_params", ()) or node.args.vararg or node.args.kwarg
                        or self.r._rebound(node.name)):
                    raise _Refuse
                if len(node.args.posonlyargs + node.args.args + node.args.kwonlyargs) > _MAX_ITEMS:
                    raise _Refuse
                if node.returns is not None and not (
                        isinstance(node.returns, ast.Constant) and node.returns.value is None):
                    raise _Refuse
                for arg in node.args.posonlyargs + node.args.args + node.args.kwonlyargs:
                    if arg.annotation is not None:
                        raise _Refuse
                    writes[node, arg.arg] += 1
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                if (isinstance(node, ast.AnnAssign) or len(node.targets) != 1
                        or not isinstance(node.targets[0], ast.Name)):
                    raise _Refuse
                writes[self.r.function_of(node), node.targets[0].id] += 1
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                if node.id in self.definitions:
                    par = self.r.parent.get(node)
                    if not (isinstance(par, ast.Call) and par.func is node):
                        raise _Refuse  # no callable aliases or callback escapes
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                receiver = node.func.value
                if node.func.attr in {"append", "extend"} and isinstance(receiver, ast.Name):
                    self.builders.add((self.r.function_of(receiver), receiver.id))
        if any(count != 1 for count in writes.values()):
            raise _Refuse  # includes parameter writes and later same-type writes

    def prove(self):
        self.prepare()
        self.block(self.tree.body, self.globals, self.tree, self.tree)
        if not self.receivers or self.active:
            raise _Refuse
        return frozenset(self.receivers)

    def lookup(self, node, env, scope):
        if node.id in env:
            return env[node.id]
        if id(scope) in self.r.bindings.get(node.id, set()):
            raise _Refuse  # a local read cannot borrow a later/global binding
        if node.id in self.globals:
            return self.globals[node.id]
        raise _Refuse

    def define(self, node, env):
        positional = node.args.posonlyargs + node.args.args
        defaults = dict(zip([p.arg for p in positional[len(positional) - len(node.args.defaults):]],
                            node.args.defaults))
        defaults.update({p.arg: d for p, d in zip(node.args.kwonlyargs, node.args.kw_defaults)
                         if d is not None})
        # Definition-time expressions run even when the body is unreferenced.
        values = {name: self.expr(expr, env, self.tree, self.tree)
                  for name, expr in defaults.items()}
        if not all(_exact(v) for v in values.values()):
            raise _Refuse
        params = []
        for kind, args in ((Parameter.POSITIONAL_ONLY, node.args.posonlyargs),
                           (Parameter.POSITIONAL_OR_KEYWORD, node.args.args),
                           (Parameter.KEYWORD_ONLY, node.args.kwonlyargs)):
            for arg in args:
                params.append(Parameter(arg.arg, kind, default=values.get(arg.arg, Parameter.empty)))
        self.signatures[node] = Signature(params)

    def block(self, statements, env, scope, owner):
        for stmt in statements:
            if isinstance(stmt, ast.FunctionDef) and scope is self.tree:
                self.define(stmt, env)
            elif isinstance(stmt, ast.Pass):
                pass
            elif isinstance(stmt, ast.Expr):
                self.expr(stmt.value, env, scope, owner)
            elif isinstance(stmt, ast.Assign):
                name = stmt.targets[0].id
                value = self.expr(stmt.value, env, scope, owner)
                if isinstance(value, _Builder):
                    raise _Refuse
                if (isinstance(stmt.value, ast.List) and not stmt.value.elts
                        and (scope, name) in self.builders):
                    value = _Builder(owner)
                env[name] = value
            elif isinstance(stmt, ast.If):
                condition = self.expr(stmt.test, env, scope, owner)
                branch = stmt.body if _truth(condition) else stmt.orelse
                returned, value = self.block(branch, env, scope, owner)
                if returned:
                    return True, value
            elif isinstance(stmt, ast.Assert):
                if not _native(self.expr(stmt.test, env, scope, owner)):
                    raise _Refuse
                if stmt.msg is not None:
                    if not isinstance(stmt.msg, ast.Constant) or type(stmt.msg.value) is not str:
                        raise _Refuse
                # Assertion failure may stop execution. Continuing is a safe
                # overapproximation, retaining every later incoming call.
            elif isinstance(stmt, ast.Return) and scope is not self.tree:
                value = self.expr(stmt.value, env, scope, owner)
                if isinstance(value, _Builder):
                    if value.owner is not owner:
                        raise _Refuse
                    value = _LIST_RESULT
                if not _native(value):
                    raise _Refuse
                return True, value
            else:
                raise _Refuse
        return False, _NONE

    def expr(self, node, env, scope, owner, depth=0):
        if depth > _MAX_VALUE_DEPTH:
            raise _Refuse
        if node is None:
            return _NONE
        if isinstance(node, ast.Constant):
            if type(node.value) is int:
                return _int(node.value)
            if type(node.value) is bool:
                return _Value("bool", node.value)
            if type(node.value) is str:
                return _Value("str", node.value)
            if node.value is None:
                return _NONE
            raise _Refuse
        if isinstance(node, ast.Name):
            return self.lookup(node, env, scope)
        if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            if len(node.elts) > _MAX_ITEMS:
                raise _Refuse
            values = tuple(self.expr(e, env, scope, owner, depth + 1) for e in node.elts)
            if isinstance(node, ast.Set):
                values = frozenset(_integer(v) for v in values)
                if len(values) > _MAX_SET:
                    raise _Refuse
                return _Value("set", values)
            if not all(_native(v) for v in values):
                raise _Refuse
            value = _Value("list" if isinstance(node, ast.List) else "tuple", values)
            _size(value)  # bound copied/hashed structure, including nested aliases
            return value
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand, env, scope, owner, depth + 1)
            if isinstance(node.op, ast.Not):
                return _Value("bool", not _truth(value))
            number = _integer(value)
            if isinstance(node.op, (ast.USub, ast.UAdd)):
                return _int(-number if isinstance(node.op, ast.USub) else number)
            raise _Refuse
        if isinstance(node, ast.BinOp):
            left = self.expr(node.left, env, scope, owner, depth + 1)
            right = self.expr(node.right, env, scope, owner, depth + 1)
            if (isinstance(node.op, ast.Sub) and isinstance(left, _Value)
                    and isinstance(right, _Value) and left.kind == right.kind == "set"):
                return _Value("set", left.data - right.data)
            a, b = _integer(left), _integer(right)
            if isinstance(node.op, ast.Add):
                return _int(a + b)
            if isinstance(node.op, ast.Sub):
                return _int(a - b)
            raise _Refuse
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left = self.expr(node.left, env, scope, owner, depth + 1)
            right = self.expr(node.comparators[0], env, scope, owner, depth + 1)
            op = node.ops[0]
            if isinstance(op, (ast.Eq, ast.NotEq)) and _native(left) and _native(right):
                if not _exact(left) or not _exact(right):
                    return _UNKNOWN_BOOL
                equal = _plain(left) == _plain(right)
                return _Value("bool", not equal if isinstance(op, ast.NotEq) else equal)
            a, b = _integer(left), _integer(right)
            operations = {ast.Lt: a < b, ast.LtE: a <= b, ast.Gt: a > b, ast.GtE: a >= b}
            if type(op) not in operations:
                raise _Refuse
            return _Value("bool", operations[type(op)])
        if isinstance(node, ast.Call):
            return self.call(node, env, scope, owner)
        raise _Refuse

    def call(self, node, env, scope, owner):
        receiver = (self.expr(node.func.value, env, scope, owner)
                    if isinstance(node.func, ast.Attribute) else None)
        # All arguments are checked before binding, even if arity will fail.
        args = []
        for arg in node.args:
            value = self.expr(arg.value if isinstance(arg, ast.Starred) else arg, env, scope, owner)
            if isinstance(arg, ast.Starred):
                if not _exact(value) or value.kind not in {"list", "tuple"}:
                    raise _Refuse
                if len(args) + len(value.data) > _MAX_ITEMS:
                    raise _Refuse
                args.extend(value.data)
            else:
                if len(args) >= _MAX_ITEMS:
                    raise _Refuse
                args.append(value)
        kwargs = {}
        for kw in node.keywords:
            value = self.expr(kw.value, env, scope, owner)
            if kw.arg is None or kw.arg in kwargs or len(args) + len(kwargs) >= _MAX_ITEMS:
                raise _Refuse
            kwargs[kw.arg] = value
        if isinstance(node.func, ast.Attribute):
            if any(isinstance(a, ast.Starred) for a in node.args) or kwargs:
                raise _Refuse
            if node.func.attr == "pop":
                if (args or not isinstance(node.func.value, ast.BinOp)
                        or not _fresh(node.func.value) or not isinstance(receiver, _Value)
                        or receiver.kind != "set"
                        or len(receiver.data) != 1):
                    raise _Refuse
                self.receivers.add(node.func.value)
                return _int(next(iter(receiver.data)))
            if (not isinstance(receiver, _Builder) or receiver.owner is not owner
                    or len(args) != 1):
                raise _Refuse
            if node.func.attr == "append" and _native(args[0]):
                return _NONE
            if (node.func.attr == "extend" and _native(args[0])
                    and args[0].kind in {"list", "list_result"}):
                return _NONE
            raise _Refuse
        if not isinstance(node.func, ast.Name) or node.func.id not in self.definitions:
            raise _Refuse
        function = self.definitions[node.func.id]
        if function not in self.signatures or not all(_exact(v) for v in args + list(kwargs.values())):
            raise _Refuse
        try:
            binding = self.signatures[function].bind(*args, **kwargs)
        except TypeError:
            raise _Refuse from None
        binding.apply_defaults()
        row = tuple(binding.arguments.items())
        key = function, row
        self.edges += 1
        if self.edges > _MAX_EDGES or key in self.active or len(self.active) >= _MAX_DEPTH:
            raise _Refuse
        if key in self.memo:
            return self.memo[key]
        self.counts[function] += 1
        if self.counts[function] > _MAX_FUNCTION_STATES or sum(self.counts.values()) > _MAX_STATES:
            raise _Refuse
        self.active.add(key)
        _, value = self.block(function.body, dict(row), function, key)
        self.active.remove(key)
        self.memo[key] = value
        return value
