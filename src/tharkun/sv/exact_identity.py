"""Alias origins for plain-class scalar identity proofs.

Copying preserves an origin; merges preserve its path guard. An operation whose
allocation/interning is unknown gets an unconstrained origin, never a guess that
equal values are the same object. These extra choices overapproximate native
identity: only UNSAT can prove a property, while SAT still needs native replay.
"""
import ast

import z3

from tharkun.sv._symexec import Ref, Union, Unsupported, _Dead, _Lit, _bool, _pytype
from tharkun.sv.exact_subset import type_identity_args


KEY = "\0identity:"
ABSENT = object()


def active(tree):
    return any(isinstance(n, ast.Compare) and len(n.ops) == 1 and isinstance(n.ops[0], (ast.Is, ast.IsNot))
               and type_identity_args(n) is None
               and not (isinstance(n.comparators[0], ast.Constant) and n.comparators[0].value is None)
               for n in ast.walk(tree))


class _Returned:
    def __init__(self, value, origin):
        self.value, self.origin = value, origin


class IdentityMixin:
    identity_enabled = True

    def __init__(self, tree, *args, **kwargs):
        super().__init__(tree, *args, **kwargs)
        self._origins = []
        self._origin_cache = {}
        self._known_origins = {}
        self._argument_origins = {}
        self._return_values = {id(n.value) for n in ast.walk(tree) if isinstance(n, ast.Return) and n.value is not None}
        for name, value in list(self.env.items()):
            self.env[KEY + name] = self.origin(value)

    def origin(self, value):
        if value is None:
            return z3.IntVal(-1)
        if _pytype(value) is bool:
            return z3.If(_bool(value), -2, -3)
        key = ("ref", value.oid) if isinstance(value, Ref) else ("class", value) if isinstance(value, type) else None
        if key is not None:
            if key not in self._known_origins:
                self._known_origins[key] = z3.IntVal(len(self._known_origins) + 1)
            return self._known_origins[key]
        return z3.FreshInt("identity origin")

    def set_origin(self, origin):
        if self._origins:
            self._origins[-1] = origin

    def ev(self, node):
        self._origins.append(None)
        try:
            value = super().ev(node)
            origin = self._origins[-1]
            if origin is None or value is None or isinstance(value, (Ref, type)) or _pytype(value) is bool:
                origin = self.origin(value)
        finally:
            self._origins.pop()
        self.last_origin = origin
        self._origin_cache[id(node), self.depth] = origin
        return _Returned(value, origin) if self.depth and id(node) in self._return_values else value

    def ev_Name(self, node):
        value = super().ev_Name(node)
        frame = self.env if node.id in self.env else self.frames[0] if self.frames and node.id in self.frames[0] else {}
        tag = frame.get(KEY + node.id)
        if tag is not None:
            self.set_origin(tag)
        return value

    def assign(self, target, value):
        tag = self.last_origin
        if isinstance(target, ast.Name):
            super().assign(target, value)
            self.env[KEY + target.id] = tag
        elif isinstance(target, ast.Attribute):
            self.pending_origin = tag
            try:
                super().assign(target, value)
            finally:
                del self.pending_origin
        else:
            raise Unsupported("identity origins for this assignment target")

    def st(self, stmt):
        if isinstance(stmt, ast.AugAssign):
            if not isinstance(stmt.target, (ast.Name, ast.Attribute)):
                raise Unsupported("identity origins for augmented subscript stores")
            import copy
            load = copy.copy(stmt.target)
            load.ctx = ast.Load()
            old = self.ev(load)
            value = self.ev_BinOp(ast.BinOp(_Lit(old), stmt.op, stmt.value))
            self.last_origin = self.origin(value)
            self.assign(stmt.target, value)
            return
        return super().st(stmt)

    def _inline(self, fd, args, call, **kwargs):
        tags = [self._origin_cache[id(a), self.depth] for a in call.args]
        if len(args) == len(tags) + 1 and isinstance(args[0], Ref):
            tags.insert(0, self.origin(args[0]))
        if len(args) != len(tags):
            raise Unsupported("identity origins for this call binding")
        depth = self.depth + 1
        self._argument_origins[depth] = {KEY + p.arg: tag for p, tag in zip(fd.args.args, tags)}
        try:
            result = super()._inline(fd, args, call, **kwargs)
        finally:
            self._argument_origins.pop(depth, None)
        if isinstance(result, _Returned):
            self.set_origin(result.origin)
            return result.value
        self.set_origin(self.origin(result))
        return result

    def _exec_fn(self, stmts, top=False):
        self.env.update(self._argument_origins.pop(self.depth, {}))
        return super()._exec_fn(stmts, top)

    def _merge_value(self, condition, a, b):
        if isinstance(a, _Returned) or isinstance(b, _Returned):
            a = a if isinstance(a, _Returned) else _Returned(a, self.origin(a))
            b = b if isinstance(b, _Returned) else _Returned(b, self.origin(b))
            return _Returned(super()._merge_value(condition, a.value, b.value),
                             z3.If(condition, a.origin, b.origin))
        return super()._merge_value(condition, a, b)

    def present(self, value):
        if not isinstance(value, Union):
            if value is ABSENT:
                raise _Dead()
            return value
        live = [(g, v) for g, v in self._work_members(value) if v is not ABSENT]
        if not live:
            raise _Dead()
        self._holds(z3.Or(*[_bool(g) for g, _ in live]))
        return live[0][1] if len(live) == 1 else Union(live)

    def ev_Compare(self, node):
        if (len(node.ops) == 1 and isinstance(node.ops[0], (ast.Is, ast.IsNot))
                and type_identity_args(node) is None
                and not (isinstance(node.comparators[0], ast.Constant) and node.comparators[0].value is None)):
            self.ev(node.left)
            left = self.last_origin
            self.ev(node.comparators[0])
            same = left == self.last_origin
            return z3.Not(same) if isinstance(node.ops[0], ast.IsNot) else same
        return super().ev_Compare(node)
