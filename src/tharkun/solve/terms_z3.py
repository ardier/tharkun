"""Optional scalar projection and conservative pre-state container abstraction.

z3 is imported only by the two public functions. Source Py names share one tag
and one value per numeric sort. OTHER covers subclasses and custom instances;
it is distinct from the exact builtin object tag. Container lengths are unbounded
nonnegative Ints; primitive membership is an uninterpreted key set, without
ordering, cardinality or value claims. No cache or target execution.
"""
from __future__ import annotations

import struct

from . import terms as t


_TAGS = {name: index for index, name in enumerate(sorted(t._TAGS))}
_OTHER = len(_TAGS)
_SORT_TAG = {t.INT: "int", t.BOOL: "bool", t.F64: "float"}


class _Unknown(Exception):
    pass


def _check_terms(terms):
    names, count = {}, 0
    for term in terms:
        checked = t._variables(term)
        if isinstance(checked, t.Refused):
            return checked
        count += sum(1 for _ in t._term_nodes(term))
        if count > t._MAX_NODES:
            return t.Refused("combined facts and goal exceed the node limit")
        for name, sort in checked.items():
            if name in names and names[name] != sort:
                return t.Refused("one name has incompatible term sorts across facts and goal")
            names[name] = sort
    return names


def _guards(term):
    if term.op == "typeis" and term.args[0].op == "var" and term.args[0].sort == t.PY:
        return {(term.args[0].args[0], term.args[1].args[0])}
    if term.op == "and":
        result = set()
        for arg in term.args:
            result.update(_guards(arg))
        return result
    return set()


def _partial(term):
    return term.op in {"floordiv", "mod", "i2f"} or any(
        _partial(arg) for arg in term.args if isinstance(arg, t.Term))


class _Lowerer:
    def __init__(self, z3, resource_limit, context=None, types=None):
        self.z3 = z3
        self.context = context
        self.remaining = resource_limit
        self.solver = z3.Solver(ctx=context)
        self.variables, self.domains = {}, []
        self.types = types or {}
        self.lengths, self.keysets = {}, {}

    def true(self):
        return self.z3.BoolVal(True, ctx=self.context)

    def variable(self, name, sort):
        key = name, sort
        if key not in self.variables:
            z3 = self.z3
            constructors = {t.INT: lambda n: z3.Int(n, ctx=self.context),
                            t.BOOL: lambda n: z3.Bool(n, ctx=self.context),
                            t.F64: lambda n: z3.FP(n, z3.Float64(ctx=self.context), ctx=self.context),
                            t.TAG: lambda n: z3.Int(n, ctx=self.context)}
            value = constructors[sort](f"tharkun-term-{sort.value}:{name}")
            self.variables[key] = value
            if sort == t.TAG:
                self.domains.append(z3.And(value >= 0, value <= _OTHER))
        return self.variables[key]

    def check(self, condition):
        z3 = self.z3
        if self.remaining <= 0:
            raise _Unknown("resource limit exhausted")
        self.solver.set(rlimit=self.remaining)
        self.solver.push()
        try:
            self.solver.add(*self.domains, condition)
            before = self._count()
            result = self.solver.check()
            self.remaining -= max(0, self._count() - before)
            if result == z3.unknown:
                raise _Unknown("z3 answered unknown")
            return result
        finally:
            self.solver.pop()

    def _count(self):
        try:
            return self.solver.statistics().get_key_value("rlimit count")
        except self.z3.Z3Exception:
            return 0

    def defined(self, pc, condition):
        z3 = self.z3
        simple = z3.simplify(condition)
        if z3.is_true(simple):
            return
        if z3.is_false(simple) or self.check(z3.And(pc, z3.Not(condition))) != z3.unsat:
            raise ValueError("operation is not defined under its guards and facts")

    def fp_literal(self, value):
        # The IEEE bits retain signed zero; NaN comparisons still use fpEQ.
        z3 = self.z3
        bits = int.from_bytes(struct.pack(">d", value), "big")
        return z3.fpBVToFP(z3.BitVecVal(bits, 64, ctx=self.context),
                          z3.Float64(ctx=self.context), ctx=self.context)

    def view(self, term, guards, *, membership=False):
        name, kind, domain = t._view_info(term)
        if (name, kind) not in guards:
            raise ValueError("view lacks a dominating exact container guard or fact")
        if membership:
            t._checked_domain(term, self.types)
        self.variable(name, t.TAG)
        return name, kind, domain

    def container_length(self, name):
        if name not in self.lengths:
            value = self.z3.Int(f"tharkun-term-len:{name}", ctx=self.context)
            self.lengths[name] = value
            self.domains.append(value >= 0)
        return self.lengths[name]

    def key(self, term, pc, guards):
        z3 = self.z3
        if term.sort in (t.INT, t.BOOL):
            value = self.visit(term, pc, guards)
            if term.sort == t.BOOL:
                value = z3.If(value, z3.IntVal(1, ctx=self.context), z3.IntVal(0, ctx=self.context))
            return "numeric", value
        if term.op == "key":
            kind, text = term.args
            # ASCII encoding preserves all Python Unicode code points, including surrogates.
            encoded = text.encode("utf-8", "surrogatepass").hex() if kind == "str" else text
            return kind, z3.StringVal(encoded, ctx=self.context)
        if term.op == "var" and term.sort == t.PY:
            name = term.args[0]
            for kind in ("str", "bytes"):
                if (name, kind) in guards:
                    self.variable(name, t.TAG)
                    return kind, z3.String(f"tharkun-term-key-{kind}:{name}", ctx=self.context)
        raise ValueError("membership key lacks a checked exact primitive kind")

    def keyset(self, name, family):
        key = name, family
        if key not in self.keysets:
            sort = (self.z3.IntSort(ctx=self.context) if family == "numeric"
                    else self.z3.StringSort(ctx=self.context))
            self.keysets[key] = self.z3.Array(f"tharkun-term-keys-{family}:{name}", sort,
                                             self.z3.BoolSort(ctx=self.context))
        return self.keysets[key]

    def visit(self, term, pc, guards):
        z3, op, args = self.z3, term.op, term.args
        if op == "var":
            if term.sort == t.PY:
                raise ValueError("Py value requires an exact type test or guarded cast")
            return self.variable(args[0], term.sort)
        if op == "lit":
            if term.sort == t.F64:
                return self.fp_literal(float.fromhex(args[0]))
            return (z3.BoolVal(args[0], ctx=self.context) if term.sort == t.BOOL
                    else z3.IntVal(t._integer_text(args[0]), ctx=self.context))
        if op == "tag":
            return z3.IntVal(_TAGS[args[0]], ctx=self.context)
        if op == "typeis":
            value, tag = args
            if value.sort == t.PY:
                if value.op == "key":
                    return z3.BoolVal(value.args[0] == tag.args[0], ctx=self.context)
                if value.op != "var":
                    raise ValueError("unsupported Py type operand")
                return self.variable(value.args[0], t.TAG) == _TAGS[tag.args[0]]
            self.visit(value, pc, guards)
            return z3.BoolVal(_SORT_TAG[value.sort] == tag.args[0], ctx=self.context)
        if op == "len":
            value = args[0]
            if value.sort in (t.SEQ, t.MAP):
                name, _, _ = self.view(value, guards)
                return self.container_length(name)
            if value.op == "key":
                native = t.eval_term(term, {})
                if isinstance(native, t.Refused):
                    raise ValueError(native.reason)
                return z3.IntVal(native, ctx=self.context)
            if value.op == "var" and value.sort == t.PY:
                name = value.args[0]
                if any((name, kind) in guards for kind in ("str", "bytes")):
                    self.variable(name, t.TAG)
                    return self.container_length(name)
            raise ValueError("string length lacks a dominating exact str/bytes guard or fact")
        if op == "contains":
            name, _, domain = self.view(args[0], guards, membership=True)
            family, key = self.key(args[1], pc, guards)
            expected = "numeric" if domain in {"int", "bool"} else domain
            if family != expected:
                return z3.BoolVal(False, ctx=self.context)
            return z3.And(self.container_length(name) > 0,
                          z3.IsMember(key, self.keyset(name, family)))
        if op in ("as_seq", "as_map", "key"):
            raise ValueError("container/key views require len or contains projection")
        if op.startswith("as_"):
            name, kind = args[0].args[0], _SORT_TAG[term.sort]
            if (name, kind) not in guards:
                raise ValueError("cast lacks a dominating exact type guard or fact")
            self.variable(name, t.TAG)
            return self.variable(name, term.sort)
        if op in ("and", "or"):
            path, local, parts = pc, set(guards), []
            for arg in args:
                value = self.visit(arg, path, local)
                parts.append(value)
                path = z3.And(path, value if op == "and" else z3.Not(value))
                if op == "and":
                    local.update(_guards(arg))
            return z3.And(*parts) if op == "and" else z3.Or(*parts)
        if op == "i2f":
            if t._variables(args[0]):
                raise ValueError("symbolic int-to-float needs a checked overflow domain")
            value = t.eval_term(term, {})
            if isinstance(value, t.Refused):
                raise ValueError(value.reason)
            return self.fp_literal(value)
        values = [self.visit(arg, pc, guards) for arg in args]
        if op == "b2i":
            return z3.If(values[0], z3.IntVal(1, ctx=self.context), z3.IntVal(0, ctx=self.context))
        if op == "not":
            return z3.Not(values[0])
        if op == "neg":
            return z3.fpNeg(values[0], ctx=self.context) if term.sort == t.F64 else -values[0]
        x, y = values
        if op in ("floordiv", "mod"):
            self.defined(pc, y != 0)
            quotient = z3.If(y > 0, x / y, (-x) / (-y))
            return quotient if op == "floordiv" else x - y * quotient
        if op in ("add", "sub", "mul"):
            if term.sort == t.F64:
                fn = {"add": z3.fpAdd, "sub": z3.fpSub, "mul": z3.fpMul}[op]
                return fn(z3.RNE(ctx=self.context), x, y, ctx=self.context)
            return {"add": lambda: x + y, "sub": lambda: x - y, "mul": lambda: x * y}[op]()
        if args[0].sort == t.F64:
            equal = z3.fpEQ(x, y, ctx=self.context)
            return {"eq": lambda: equal, "ne": lambda: z3.Not(equal),
                    "lt": lambda: z3.fpLT(x, y, ctx=self.context),
                    "le": lambda: z3.fpLEQ(x, y, ctx=self.context),
                    "gt": lambda: z3.fpGT(x, y, ctx=self.context),
                    "ge": lambda: z3.fpGEQ(x, y, ctx=self.context)}[op]()
        return {"eq": lambda: x == y, "ne": lambda: x != y,
                "lt": lambda: x < y, "le": lambda: x <= y,
                "gt": lambda: x > y, "ge": lambda: x >= y}[op]()


def lower(term: t.Term | t.Refused, *, types=None):
    """Return a z3 AST or Refused; partial values require visible definedness.

    Exact typed free variables are supported for pure terms. Source names use Py
    and require matching typeis guards before casts. A value floor/mod needs a
    nonzero literal divisor; predicates can establish nonzero on an ordered arm.
    Symbolic i2f is refused until its OverflowError domain is explicitly modeled.
    Membership requires types to carry verified primitive heap domains, not
    annotations or constructor labels. Defaults refuse variable membership.
    """
    checked = _check_terms([term])
    if isinstance(checked, t.Refused):
        return checked
    context = t._binding_types({} if types is None else types)
    if isinstance(context, t.Refused):
        return context
    try:
        import z3
    except ImportError:
        return t.Refused("optional z3 is unavailable")
    try:
        return _Lowerer(z3, 10000, types=context).visit(term, z3.BoolVal(True), set())
    except (_Unknown, ValueError, TypeError, KeyError, OverflowError, RecursionError, z3.Z3Exception) as exc:
        return t.Refused(str(exc))


def implies(facts, goal: t.Term | t.Refused, *, resource_limit=10000, types=None):
    """Return 'yes', 'no', 'unknown' or Refused for defined implication.

    Only 'yes' certifies implication. Facts must be Boolean Terms, never strings
    or annotation hints. Direct exact-type facts can guard any fact/goal cast;
    within a term, ordered conjunction guards dominate. Total facts are lowered
    before partial facts so definedness cannot bootstrap from its own result.
    Unknown, missing z3 and unsupported terms never become 'yes'.
    types carries verified pre-state primitive heap domains for membership,
    never annotation hints; future memo keys must include this context. Container
    'no' may be a spurious abstract counterexample and is never a certification.
    """
    if type(facts) not in (list, tuple) or len(facts) > t._MAX_NODES:
        return t.Refused("facts must be a bounded list or tuple of Boolean Terms")
    if type(resource_limit) is not int or resource_limit <= 0:
        return t.Refused("resource limit must be a positive integer")
    checked = _check_terms([*facts, goal])
    if isinstance(checked, t.Refused):
        return checked
    if goal.sort != t.BOOL or any(fact.sort != t.BOOL for fact in facts):
        return t.Refused("implication requires Boolean facts and goal")
    context = t._binding_types({} if types is None else types)
    if isinstance(context, t.Refused):
        return context
    try:
        import z3
    except ImportError:
        return t.Refused("optional z3 is unavailable")
    try:
        # Each shadow query owns its context; the legacy checker may run in another thread.
        worker, guards, type_facts = _Lowerer(z3, resource_limit, z3.Context(), context), set(), []
        for fact in facts:
            if fact.op == "typeis" and fact.args[0].op == "var" and fact.args[0].sort == t.PY:
                guards.update(_guards(fact))
                type_facts.append(worker.visit(fact, worker.true(), set()))
        pc = z3.And(worker.true(), *type_facts)
        for fact in sorted(facts, key=_partial):
            value = worker.visit(fact, pc, guards)
            pc = z3.And(pc, value)
            guards.update(_guards(fact))
        value = worker.visit(goal, pc, guards)
        result = worker.check(z3.And(pc, z3.Not(value)))
        return "yes" if result == z3.unsat else "no"
    except _Unknown:
        return "unknown"
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, z3.Z3Exception) as exc:
        return t.Refused(str(exc))
