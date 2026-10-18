"""Demanded scalar and list-shape class invariants, checked independently of call contexts.

Candidate bounds are guesses. Constructor establishment, preservation by every
method, and the receiver precondition are static proof obligations.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

from .aval import BOT, V, int_iv
from .heap import InstObj, ListObj
from .state import Frame, State, iv_of_len, vleq


@dataclass
class ClassContract:
    fields: dict
    queries: dict


@dataclass
class ListShape:
    minimum: int
    element: V


def _holds(st, value, fact):
    if not isinstance(fact, ListShape):
        return vleq(value, st.zone, fact)
    if value.top or not value.val.kinds.is_bottom() or value.st or not value.refs:
        return False
    for addr in value.refs:
        obj = st.heap[addr]
        if type(obj) is not ListObj:
            return False
        lo, _ = iv_of_len(obj.length, st.zone)
        if lo is None or lo < fact.minimum or any(
                not vleq(v, st.zone, fact.element) for v in obj.values()):
            return False
    return True


def _methods(an, st, cls):
    if cls.bases:
        return None
    methods = {}
    for name, value in cls.attrs.items():
        fa = an._plain_method(st, value)
        if fa is None:
            return None
        fn = st.heap[fa].node
        args = fn.args
        if (len(args.posonlyargs + args.args) != 1 or args.kwonlyargs
                or args.vararg or args.kwarg or args.defaults or args.kw_defaults
                or fn.decorator_list):
            return None
        methods[name] = (fa, fn, (args.posonlyargs + args.args)[0].arg)
    return methods


def _scalar_body(fn, receiver, fields):
    # No external effects, global reads, receiver escapes, reflection or calls.
    parents = {child: node for node in ast.walk(fn) for child in ast.iter_child_nodes(node)}
    locals_ = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    for node in ast.walk(fn):
        if isinstance(node, (ast.Call, ast.Global, ast.Nonlocal, ast.Delete, ast.Yield,
                             ast.YieldFrom, ast.Await, ast.ClassDef, ast.Lambda,
                             ast.With, ast.Try, ast.TryStar, ast.Import, ast.ImportFrom, ast.Is, ast.IsNot)):
            return False
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
            return False
        if isinstance(node, ast.Attribute):
            if not (isinstance(node.value, ast.Name) and node.value.id == receiver
                    and node.attr in fields):
                return False
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id == receiver:
                parent = parents.get(node)
                if not isinstance(parent, ast.Attribute) or parent.value is not node:
                    return False
            elif node.id not in locals_:
                return False
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == receiver:
            return False
    return True


def infer(an, st, caddr):
    from .interp import Unsupported
    from .state import Incompatible

    cls = st.heap[caddr]
    methods = _methods(an, st, cls)
    if not methods or "__init__" not in methods:
        return None
    _, init, receiver = methods["__init__"]
    fields = {}
    for stmt in init.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
            continue
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Attribute)
                and isinstance(stmt.targets[0].value, ast.Name)
                and stmt.targets[0].value.id == receiver):
            return None
        try:
            value = ast.literal_eval(stmt.value)
        except (ValueError, TypeError):
            return None
        name = stmt.targets[0].attr
        # Sign and non-empty thresholds are candidates, not conclusions.
        if type(value) is int:
            fields[name] = int_iv(1, None) if value > 0 else int_iv(0, None) if value == 0 else int_iv(None, -1)
        elif type(value) is list and all(type(item) is int for item in value):
            fields[name] = ListShape(int(bool(value)), int_iv(None, None) if value else BOT)
        else:
            return None
    if not fields:
        return None
    hints = an.class_hints.get(cls.name, {}) if isinstance(an.class_hints, dict) else {}
    if not isinstance(hints, dict):
        hints = {}
    for name in fields.keys() & hints.keys():
        bounds = hints[name]
        if (not isinstance(fields[name], ListShape)
                and isinstance(bounds, (list, tuple)) and len(bounds) == 2
                and all(type(x) is int for x in bounds) and bounds[0] <= bounds[1]):
            lo, hi = bounds
            fields[name] = int_iv(lo, None) if lo >= 0 else int_iv(None, hi) if hi < 0 else int_iv(None, None)
    if any(not _scalar_body(fn, recv, fields) for _, fn, recv in methods.values()):
        return None

    def run(fa, fn, initial):
        checker = type(an)(an.source, an.filename)
        checker.deadline = an.deadline
        state = State()
        state.frames.append(Frame("g"))
        # Functions and class definitions are immutable in this restricted body.
        state.heap = {a: ob for a, ob in st.heap.items() if ob.kind in ("class", "func")}
        addr = (((), fn.lineno, fn.col_offset, "contract-receiver"), "r")
        attrs = {}
        for name, fact in initial.items():
            value = fact
            if isinstance(fact, ListShape):
                list_addr = (((), fn.lineno, fn.col_offset, "contract-list-" + name), "r")
                length = "contract-length-" + name
                state.zone.assign_interval(length, fact.minimum, None)
                state.heap[list_addr] = ListObj(length, fact.element)
                value = V(refs=frozenset({list_addr}))
            attrs[name] = (value, True, None)
        state.heap[addr] = InstObj(caddr, attrs)
        value = checker.call_func(state, fa, [V(refs=frozenset({addr}))], {}, fn)
        if state.bottom or any(not check.ok for check in checker.checks.values()):
            return None
        attrs = state.heap[addr].attrs
        if any(name not in attrs or not attrs[name][1]
               or not _holds(state, attrs[name][0], fact) for name, fact in fields.items()):
            return None
        if value.top or value.refs or value.st:
            return None
        if value.i is not None:
            value = value.replace(i=int_iv(*checker.ibounds(state, value.i)).i)
        return value, list(checker.checks.values())

    queries = {}
    try:
        for name, (fa, fn, recv) in methods.items():
            result = run(fa, fn, {} if name == "__init__" else fields)
            if result is None:
                return None  # failed induction drops the candidate
            # Mutation effects need a relational postcondition; keep the existing
            # transfer until an obligation demands that stronger contract.
            if name != "__init__" and not any(
                    isinstance(n, ast.AugAssign) or
                    isinstance(n, (ast.Attribute, ast.Subscript)) and isinstance(n.ctx, ast.Store)
                    for n in ast.walk(fn)):
                queries[fa] = result
    except (Unsupported, Incompatible):
        return None
    return ClassContract(fields, queries) if queries else None


def apply(an, st, addr, faddr, args, kwargs, node):
    if args or kwargs:
        return None
    obj = st.heap[addr]
    cls = st.heap[obj.cls]
    if cls in an.class_contracts:
        contract = an.class_contracts[cls]
    else:
        contract = infer(an, st, obj.cls)
        if not an._forced_draws:    # nothing is memoized while a case split's force is pending
            an.class_contracts[cls] = contract
    if contract is None or faddr not in contract.queries:
        return None
    if any(name not in obj.attrs or not obj.attrs[name][1]
           or not _holds(st, obj.attrs[name][0], fact)
           for name, fact in contract.fields.items()):
        return None  # aliases or external stores can invalidate the precondition
    result, checks = contract.queries[faddr]
    for check in checks:
        location = ast.copy_location(ast.Pass(), node)
        location.lineno, location.col_offset = check.line, check.col
        an.check(location, check.op, check.excs, True,
                 "inductive class invariant: " + "; ".join(check.facts))
    an.features.add("class-invariant")
    return result


def demands(source):
    """Field reads in receiver queries whose obligations may need a contract."""
    result = {}
    for cls in (n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.ClassDef)):
        names = set()
        for fn in cls.body:
            if not isinstance(fn, ast.FunctionDef) or fn.name == "__init__":
                continue
            args = fn.args.posonlyargs + fn.args.args
            if len(args) != 1:
                continue
            receiver = args[0].arg
            names.update(n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)
                         and isinstance(n.ctx, ast.Load) and isinstance(n.value, ast.Name)
                         and n.value.id == receiver)
        if names:
            result[cls.name] = sorted(names)
    return result
