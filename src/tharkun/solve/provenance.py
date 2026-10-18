"""Flow-insensitive value provenance for the determinism text check.

Facts are unions over *all* writes, call arguments, returns and container
mutations. Nothing is inferred from a successful concrete run. Unknown calls
and unsupported binding forms stay unknown. This is a type/provenance analysis,
not a proof that an operation succeeds.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from inspect import Parameter, Signature

UNKNOWN = ("unknown",)
SCALARS = frozenset({"int", "float", "complex", "bool", "str", "bytes", "NoneType"})
_NUMERIC = frozenset({("int",), ("float-finite",), ("float-posinf",), ("float-neginf",)})
# Only resolved, non-rebound imports may use these immutable scalar constants.
STRING_CONSTANTS = frozenset({("string", "digits"), ("string", "ascii_uppercase")})
SCALAR_CALLS = {
    "int": "int", "float": "float", "complex": "complex", "bool": "bool",
    "str": "str", "repr": "str", "ascii": "str", "format": "str",
    "len": "int", "ord": "int", "chr": "str", "hex": "str", "oct": "str", "bin": "str",
}

_HEAPQ_CALLS = frozenset({
    "heapq.heapify", "heapq.heappush", "heapq.heappop", "heapq.heappushpop",
    "heapq.heapreplace", "heapq.nsmallest", "heapq.nlargest",
})


_COMPREHENSION_SCOPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef) + _COMPREHENSION_SCOPES


def _int_literal(node):
    """An exact ``int`` literal, optionally under one unary plus or minus (never a bool)."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return isinstance(node, ast.Constant) and type(node.value) is int


def _mentions(node, name):
    """Does ``node`` bind, rebind, declare or read the identifier ``name``?  Deliberately
    over-approximate: every spelling of a binding the grammar has, whichever scope owns it."""
    if isinstance(node, ast.Name):
        return node.id == name
    if isinstance(node, ast.arg):
        return node.arg == name
    if isinstance(node, ast.alias):
        return name in {node.name, node.asname, node.name.partition(".")[0]}
    if isinstance(node, ast.MatchMapping):
        return node.rest == name
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return name in node.names
    # def / class names, except-as, match captures, generic type parameters
    return getattr(node, "name", None) == name


def _terminates(body):
    """Control cannot fall off the end of ``body``: its last statement returns or raises on every path."""
    if not body:
        return False
    last = body[-1]
    if isinstance(last, (ast.Return, ast.Raise)):
        return True
    if isinstance(last, ast.If):
        return bool(last.orelse) and _terminates(last.body) and _terminates(last.orelse)
    return False


class Provenance:
    def __init__(self, resolver, *, class_only=False):
        self.r = resolver
        #: the census's class-only mode (``determinism_census``'s ``class_only_exceptions``): judge exceptions by CLASS alone (the caller
        #: has proven nothing can observe one in flight); see ``class_only_lookup`` and ``_heapq_order_key``
        self.class_only = class_only
        self._inert_lists = {}
        self.nodes = [n for n in ast.walk(resolver.tree) if resolver.live(n)]
        self.deque_calls = {n for n in self.nodes if isinstance(n, ast.Call)
                            and resolver.call_name(n) == "collections.deque"}
        self.lambda_defaultdicts = {
            n for n in self.nodes if isinstance(n, ast.Lambda)
            and isinstance(resolver.parent.get(n), ast.Call)
            and self._canonical_defaultdict_call(resolver.parent[n])
            and len(resolver.parent[n].args) == 1 and resolver.parent[n].args[0] is n
            and not resolver.parent[n].keywords
            and self._lambda_default_value(n) is not None
        }
        self.defaultdict_return_owners, self.defaultdict_return_origins = \
            self._defaultdict_return_owners()
        self.facts = defaultdict(set)
        self.changed = False
        self.slices = {}
        self.inexact = set()  # mutation/escape loses literal slots, even if element types stay equal
        self.literal_lists = set()
        for n in self.nodes:
            if isinstance(n, ast.List):
                try:
                    ast.literal_eval(n)
                except (ValueError, TypeError):
                    continue
                self.literal_lists.add(n)
        self.final = False
        self.classes = {n for n in self.nodes if isinstance(n, ast.ClassDef)}
        self.generators = {resolver.function_of(n) for n in self.nodes if isinstance(n, (ast.Yield, ast.YieldFrom))}
        self.methods = {}
        self.opaque = set()
        self.globals = defaultdict(set)
        # Which functions have ever been the target of invoke() (a real call,
        # even with unresolved arguments) -- see get()'s use below. A function
        # nobody calls never runs, so its never-bound parameters must stay
        # EMPTY (contribute nothing), not UNKNOWN (an address might reach
        # anywhere): the flat ("attr", name) namespace unions a write from
        # ANY method into every reader of that attribute name, so an uncalled
        # method's fabricated-unknown parameter would otherwise poison a
        # completely unrelated, address-free attribute of the same name.
        self.invoked = set()
        self.params = {("name", f, p.arg)
                        for f in self.nodes if isinstance(f, ast.FunctionDef)
                        for p in f.args.posonlyargs + f.args.args + f.args.kwonlyargs
                        + ([f.args.vararg] if f.args.vararg else [])
                        + ([f.args.kwarg] if f.args.kwarg else [])}
        self.unsafe = any(
            isinstance(n, (ast.Nonlocal, ast.NamedExpr, ast.Match, ast.AsyncFunctionDef))
            or isinstance(n, ast.Lambda) and n not in self.lambda_defaultdicts
            or isinstance(n, (ast.ExceptHandler, ast.Delete))
            or isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store) and n.attr.startswith("__")
            or isinstance(n, ast.ClassDef) and (n.keywords or n.decorator_list or any(
                not (isinstance(b, ast.Name) and (b.id in {c.name for c in self.classes} or
                    b.id == "object" and resolver.is_builtin(b)))
                for b in n.bases))
            or isinstance(n, ast.FunctionDef) and (n.decorator_list or isinstance(
                resolver.function_of(n), (ast.FunctionDef, ast.Lambda)))
            or isinstance(n, ast.Call) and resolver.call_name(n) in {"setattr", "delattr"}
            for n in self.nodes)
        for c in self.classes:
            self.methods[c] = {n.name: n for n in c.body if isinstance(n, ast.FunctionDef)}
            if any(name in self.methods[c] for name in (
                    "__getattr__", "__getattribute__", "__setattr__", "__new__",
                    "__get__", "__set__", "__init_subclass__")):
                self.unsafe = True
        if self.unsafe:
            return
        # Single inheritance has a unique lookup chain. Ambiguous bases remain
        # opaque rather than guessing a method resolution order.
        for c in self.classes:
            for st in c.body:
                if isinstance(st, ast.Assign):
                    for target in st.targets:
                        if isinstance(target, ast.Name) and target.id.startswith("__") and target.id not in {"__slots__", "__doc__", "__module__"}:
                            if isinstance(st.value, ast.Name) and st.value.id in self.methods[c]:
                                self.methods[c][target.id] = self.methods[c][st.value.id]
                            else:
                                self.opaque.add(c)
        for n in self.nodes:
            if isinstance(n, ast.Global):
                self.globals[self.r.function_of(n)].update(n.names)
        pending = set(self.classes)
        while pending:
            progressed = False
            for c in list(pending):
                bases = [b for b in c.bases if not (
                    isinstance(b, ast.Name) and b.id == "object" and resolver.is_builtin(b))]
                matches = [{k for k in self.classes if isinstance(b, ast.Name) and k.name == b.id}
                           for b in bases]
                if len(matches) > 1 or any(len(m) != 1 for m in matches):
                    self.opaque.add(c)
                elif matches:
                    base = next(iter(matches[0]))
                    # A lexical name is a base only when its class definition is
                    # the sole binding; assignments/parameters may replace it.
                    rebound = any((isinstance(n, ast.Name) and n.id == base.name and isinstance(n.ctx, (ast.Store, ast.Del)))
                                  or (isinstance(n, ast.arg) and n.arg == base.name)
                                  or (isinstance(n, ast.FunctionDef) and n.name == base.name)
                                  for n in self.nodes)
                    if rebound:
                        self.opaque.add(c)
                    if base in pending:
                        continue
                    self.methods[c] = self.methods[base] | self.methods[c]
                    if base in self.opaque:
                        self.opaque.add(c)
                pending.remove(c)
                progressed = True
            if not progressed:
                self.opaque.update(pending)
                break
        # Least fixpoint first; unresolved cycles then acquire UNKNOWN, never a
        # vacuous "all elements are safe" proof.
        for final in (False, True):
            self.final = final
            for _ in range(100):
                self.changed = False
                for n in self.nodes:
                    self.step(n)
                if not self.changed:
                    break
            else:
                self.unsafe = True
                return

    def add(self, key, values):
        old = self.facts[key]
        if not values <= old:
            old.update(values)
            self.changed = True

    def get(self, key):
        values = self.facts[key]
        if values:
            return set(values)
        if not self.final or key[0] in {"elem", "key"}:
            return set()
        if key in self.params and key[1] not in self.invoked:
            return set()  # uncalled function: no real argument facts exist, and none ever will
        return {UNKNOWN}

    def key(self, node):
        # A generator target and its first iterable have different scopes.
        # In particular, ``x for x in x.items`` must not overwrite the outer x.
        scope = self.r.function_of(node)
        if isinstance(scope, (ast.GeneratorExp, ast.ListComp, ast.SetComp, ast.DictComp)):
            while not isinstance(scope, ast.Module):
                if node.id in self.globals[scope]:
                    return ("name", self.r.tree, node.id)
                if id(scope) in self.r.bindings.get(node.id, set()):
                    return ("name", scope, node.id)
                scope = self.r.function_of(scope)
            return ("name", scope, node.id)
        child = node
        scope = self.r.parent.get(child)
        while scope is not None:
            if isinstance(scope, ast.ClassDef):
                return ("attr", node.id)
            if isinstance(scope, (ast.FunctionDef, ast.Lambda)):
                if child is not scope.args:
                    break
            if isinstance(scope, ast.Module):
                break
            child, scope = scope, self.r.parent.get(scope)
        scope = scope or self.r.tree
        while not isinstance(scope, ast.Module):
            if node.id in self.globals[scope]:
                return ("name", self.r.tree, node.id)
            if id(scope) in self.r.bindings.get(node.id, set()):
                return ("name", scope, node.id)
            scope = self.r.function_of(scope)
        return ("name", scope, node.id)

    def invalidate(self, values):
        origins = {v[1] for v in values if v[0] in {"seq", "dict", "iterator", "deque", "set"}}
        if not origins <= self.inexact:
            self.inexact.update(origins)
            self.changed = True

    def escape(self, values, seen=None):
        """Unknown callees may mutate reachable objects or invoke callbacks."""
        if not self.final:
            return
        seen = set() if seen is None else seen
        for v in values - seen:
            seen.add(v)
            if v[0] in {"seq", "dict", "iterator", "defaultdict", "deque", "set"}:
                self.invalidate({v})
                self.escape(self.get(("elem", v[1])) | self.get(("key", v[1])), seen)
                self.add(("elem", v[1]), {UNKNOWN})
                if v[0] in {"dict", "defaultdict"}:
                    self.add(("key", v[1]), {UNKNOWN})
            elif v[0] == "method":
                self.escape({v[1]}, seen)
            elif v[0] in {"inst", "class"}:
                for key in list(self.facts):
                    if key[0] == "attr":
                        self.escape(self.get(key), seen)
                        self.add(key, {UNKNOWN})
            elif v[0] in {"func", "bound"}:
                f = v[1]
                for p in f.args.posonlyargs + f.args.args + f.args.kwonlyargs:
                    self.add(("name", f, p.arg), {UNKNOWN})
                if f.args.vararg:
                    self.add(("name", f, f.args.vararg.arg), {UNKNOWN})
                if f.args.kwarg:
                    self.add(("name", f, f.args.kwarg.arg), {UNKNOWN})

    def scalar(self, values):
        return bool(values) and all(len(v) == 1 and v[0] in SCALARS for v in values)

    def closed_deques(self):
        """A closed transport/effect domain, checked after the all-write fixpoint.

        Carrying a plain object does not render it. Identity-only set membership
        does not observe its hash or the set's order. Neither permission extends
        to callbacks, arbitrary iterables, or default object/deque representation.
        """
        if self.unsafe or not self.deque_calls:
            return False
        records = {}
        for cls in self.classes:
            if cls.bases or cls.keywords or cls.decorator_list or self.r._rebound(cls.name):
                return False
            fields = set()
            for stmt in cls.body:
                if isinstance(stmt, ast.Pass) or (isinstance(stmt, ast.Expr) and
                        isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str)):
                    continue
                if not isinstance(stmt, ast.FunctionDef) or stmt.name != "__init__":
                    return False
                positional = stmt.args.posonlyargs + stmt.args.args
                params = positional + stmt.args.kwonlyargs
                if not positional or stmt.decorator_list:
                    return False
                receiver = positional[0].arg
                for write in stmt.body:
                    if isinstance(write, ast.Pass) or (isinstance(write, ast.Expr) and
                            isinstance(write.value, ast.Constant) and isinstance(write.value.value, str)):
                        continue
                    if not (isinstance(write, ast.Assign) and len(write.targets) == 1
                            and isinstance(write.targets[0], ast.Attribute)
                            and isinstance(write.targets[0].value, ast.Name)
                            and write.targets[0].value.id == receiver
                            and not write.targets[0].attr.startswith("__")
                            and isinstance(write.value, ast.Name)
                            and write.value.id in {p.arg for p in params}):
                        return False
                    fields.add(write.targets[0].attr)
            records[cls] = fields

        carried = {}

        def carry(values, seen=frozenset()):
            if not values or len(seen) >= 100:
                return False  # unresolved is not an empty container
            for value in values:
                if self.scalar({value}) or value[0] == "inst" and value[1] in records:
                    continue
                if value in carried:
                    if not carried[value]:
                        return False
                    continue
                if (value[0] != "seq" or not isinstance(value[1], (ast.List, ast.Tuple))
                        or value[1] in self.inexact or value in seen):
                    return False
                origin = value[1]
                # Empty is proved by the literal shape, not by an empty fact set.
                carried[value] = (all(not isinstance(e, ast.Starred)
                    and carry(self.value(e), seen | {value}) for e in origin.elts)
                    and all(carry({v}, seen | {value}) for v in self.get(("elem", origin))))
                if not carried[value]:
                    return False
            return True

        def exact(values, tag):
            return bool(values) and all(v[0] == tag and v[1] not in self.inexact for v in values)

        def reference(values):
            return bool(values) and all(carry({v}) or exact({v}, "deque") or exact({v}, "set")
                                        for v in values)

        def key_values(values):
            return bool(values) and all(self.scalar({v}) or v[0] == "inst" and v[1] in records
                                        for v in values)

        def ordered(values):
            return bool(values) and all(v[0] == "seq" and carry({v}) for v in values)

        def generator(node):
            if not isinstance(node, ast.GeneratorExp) or len(node.generators) != 1:
                return False
            clause = node.generators[0]
            if (clause.is_async or not isinstance(clause.target, ast.Name)
                    or not isinstance(node.elt, ast.Name)
                    or self.key(clause.target) != self.key(node.elt)
                    or not ordered(self.value(clause.iter)) or len(clause.ifs) > 1):
                return False
            for condition in clause.ifs:
                if not (isinstance(condition, ast.Compare) and len(condition.ops) == 1
                        and isinstance(condition.ops[0], (ast.In, ast.NotIn))
                        and isinstance(condition.left, ast.Name)
                        and self.key(condition.left) == self.key(clause.target)
                        and key_values(self.value(condition.left))
                        and exact(self.value(condition.comparators[0]), "set")):
                    return False
            parent = self.r.parent.get(node)
            return (isinstance(parent, ast.Call) and parent.args == [node] and not parent.keywords
                    and isinstance(parent.func, ast.Attribute) and parent.func.attr == "extend"
                    and exact(self.value(parent.func.value), "deque"))

        def call(node):
            name = self.r.call_name(node)
            args = [self.value(a) for a in node.args]
            if name == "collections.deque":
                return (len(args) <= 1 and not node.keywords and
                        (not args or ordered(args[0])))
            if name == "set":
                return not args and not node.keywords
            if name in {"print", "str", "repr", "ascii", "format", "bool", "len"}:
                return (not node.keywords and all(reference(v) for v in args)
                        and (name not in {"bool", "len"} or len(args) == 1))
            if isinstance(node.func, ast.Attribute):
                values = self.value(node.func.value)
                if node.keywords:
                    return False
                if exact(values, "deque"):
                    if node.func.attr == "popleft":
                        return not args
                    if len(args) != 1:
                        return False
                    if node.func.attr == "append":
                        return carry(args[0])
                    if node.func.attr == "extend":
                        return ordered(args[0]) or generator(node.args[0])
                if exact(values, "set") and len(args) == 1:
                    if node.func.attr == "add":
                        return key_values(args[0])
                    if node.func.attr == "update":
                        return ordered(args[0]) and all(key_values(self.value(e))
                            for v in args[0] for e in v[1].elts)
                return False
            callees = self.value(node.func)
            if not callees or any(v[0] not in {"func", "class"} for v in callees):
                return False
            # Reuse the real binder; unknown spreads never become one argument.
            variants = self.arguments(node)
            return (variants is not None and all(k.arg is not None for k in node.keywords)
                    and all(reference(v) for actual in variants for v in actual)
                    and all(reference(self.value(k.value)) for k in node.keywords))

        for node in self.nodes:
            if isinstance(node, (ast.ListComp, ast.Set, ast.SetComp, ast.Dict, ast.DictComp,
                                 ast.With, ast.AsyncWith, ast.Yield, ast.YieldFrom, ast.Await,
                                 ast.Try, ast.TryStar, ast.Raise, ast.AugAssign)):
                return False
            if isinstance(node, ast.Call) and not call(node):
                return False
            if isinstance(node, ast.GeneratorExp) and not generator(node):
                return False
            if isinstance(node, (ast.List, ast.Tuple)) and isinstance(node.ctx, ast.Load):
                if not carry(self.value(node)):
                    return False
            if isinstance(node, ast.For) and not ordered(self.value(node.iter)):
                return False
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if any(not isinstance(target, (ast.Name, ast.Attribute)) for target in targets):
                    return False  # unpacking itself iterates the assigned value
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                parent = self.r.parent.get(node)
                if isinstance(parent, ast.Call) and parent.func is node:
                    continue  # every callee is checked above
                if node.id in self.r.imports.modules:
                    continue  # Resolver requires a direct, read-only attribute
                if not reference(self.value(node)):
                    return False
            if isinstance(node, ast.Attribute):
                parent = self.r.parent.get(node)
                if isinstance(parent, ast.Call) and parent.func is node:
                    continue
                values = self.value(node.value)
                if not values or any(v[0] != "inst" or node.attr not in records.get(v[1], ())
                                     for v in values):
                    return False
                if not carry(self.get(("attr", node.attr))):
                    return False
            if isinstance(node, ast.Subscript):
                if not ordered(self.value(node.value)) or not isinstance(node.ctx, ast.Load):
                    return False
                if self.value(node.slice) - {("int",), ("bool",)} or not self.value(node.slice):
                    return False
            if isinstance(node, ast.Compare):
                values = [self.value(n) for n in [node.left] + node.comparators]
                if any(any(v[0] == "deque" for v in vs) for vs in values):
                    return False
                for left, op, right in zip(values, node.ops, values[1:]):
                    if any(v[0] == "set" for v in left | right):
                        if not (isinstance(op, (ast.In, ast.NotIn)) and key_values(left)
                                and exact(right, "set")):
                            return False
            if isinstance(node, (ast.BinOp, ast.UnaryOp)):
                operands = [node.left, node.right] if isinstance(node, ast.BinOp) else [node.operand]
                if not (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)):
                    if not all(carry(self.value(n)) for n in operands):
                        return False
        # Class names also must remain direct constructors, never dynamic aliases.
        for node in self.nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                classes = {v[1] for v in self.value(node) if v[0] == "class"}
                if classes and (len(classes) != 1 or node.id != next(iter(classes)).name
                                or not isinstance(self.r.parent.get(node), ast.Call)
                                or self.r.parent[node].func is not node):
                    return False
        return True

    def safe(self, node, mode="str"):
        if self.unsafe:
            return False
        values = self.value(node)
        return bool(values) and all(self.printable(v, set(), mode) for v in values)

    def printable(self, value, seen, mode):
        if mode == "any":
            return all(self.printable(value, seen, m) for m in ("str", "repr", "format"))
        if value in seen:
            return False
        seen = seen | {value}
        if value[0] == "type":
            return True  # type(x), with custom metaclasses excluded above
        if len(value) == 1 and value[0] in SCALARS:
            return True
        if value[0] == "inst":
            if value[1] in self.opaque:
                return False
            methods = self.methods[value[1]]
            f = methods.get("__repr__")
            if mode != "repr":
                f = methods.get("__str__", f)
            if mode == "format":
                f = methods.get("__format__", f)
            return f is not None and self.scalar(self.get(("return", f)))
        if value[0] in {"seq", "dict"}:
            elements = self.get(("elem", value[1]))
            if value[0] == "dict":
                elements = elements | self.get(("key", value[1]))
            return all(self.printable(v, seen, "repr") for v in elements)
        return False

    def elements(self, values):
        out = set()
        for v in values:
            if v[0] in {"seq", "iterator"}:
                out |= self.get(("elem", v[1]))
            elif v[0] in {"str", "bytes", "range"}:
                out.add(("str" if v[0] == "str" else "int",))
            elif v[0] == "dict":
                out |= self.get(("key", v[1]))
            elif v[0] not in SCALARS:
                out.add(UNKNOWN)
        return out

    def bind(self, target, values):
        if isinstance(target, ast.Name):
            self.add(self.key(target), values)
        elif isinstance(target, ast.Attribute):
            self.add(("attr", target.attr), values)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for i, t in enumerate(target.elts):
                items = set()
                for v in values:
                    if v[0] == "seq" and isinstance(v[1], ast.Tuple) and len(v[1].elts) == len(target.elts):
                        items |= self.get(("item", v[1], i))
                    else:
                        items |= self.elements({v})
                self.bind(t, items)
        elif isinstance(target, ast.Subscript):
            for ob in self.value(target.value):
                if ob[0] in {"seq", "dict", "defaultdict", "counter"}:
                    self.invalidate({ob})
                    self.add(("elem", ob[1]), values)
                    if ob[0] in {"dict", "defaultdict", "counter"}:
                        self.add(("key", ob[1]), self.value(target.slice))
                elif ob[0] == "inst":
                    self.method(ob[1], "__setitem__", [self.value(target.slice), values])

    def method(self, cls, name, args):
        if cls in self.opaque:
            return {UNKNOWN}
        f = self.methods[cls].get(name)
        if f is None:
            return {UNKNOWN}
        return self.invoke(f, [{("inst", cls)}] + args, {})

    def elements_mapping(self, values):
        return set().union(*(self.get(("elem", v[1])) if v[0] == "dict" else {UNKNOWN} for v in values))

    def arguments(self, call):
        variants = [[]]
        for arg in call.args:
            if isinstance(arg, ast.Starred):
                choices = []
                for v in self.value(arg.value):
                    if v[0] != "seq" or not isinstance(v[1], (ast.Tuple, ast.List)) or any(
                            isinstance(e, ast.Starred) for e in v[1].elts):
                        return None
                    if isinstance(v[1], ast.List) and (v[1] not in self.literal_lists or
                            any(n in self.inexact for n in ast.walk(v[1]))):
                        return None
                    choices.append([self.get(("item", v[1], i)) for i in range(len(v[1].elts))])
            else:
                choices = [[self.value(arg)]]
            variants = [prefix + tail for prefix in variants for tail in choices]
            if len(variants) > 100:
                return None  # ponytail: bounded argument products; refine only if needed
        return variants

    def invoke(self, f, args, kwargs):
        self.invoked.add(f)
        positional = f.args.posonlyargs + f.args.args
        defaults = dict(zip([p.arg for p in positional[len(positional) - len(f.args.defaults):]],
                            f.args.defaults))
        defaults.update({p.arg: d for p, d in zip(f.args.kwonlyargs, f.args.kw_defaults) if d})
        parameters = []
        for kind, nodes in (
                (Parameter.POSITIONAL_ONLY, f.args.posonlyargs),
                (Parameter.POSITIONAL_OR_KEYWORD, f.args.args),
                (Parameter.VAR_POSITIONAL, [f.args.vararg] if f.args.vararg else []),
                (Parameter.KEYWORD_ONLY, f.args.kwonlyargs),
                (Parameter.VAR_KEYWORD, [f.args.kwarg] if f.args.kwarg else [])):
            for p in nodes:
                default = self.value(defaults[p.arg]) if p.arg in defaults else Parameter.empty
                parameters.append(Parameter(p.arg, kind, default=default))
        if None in kwargs:
            # ponytail: unknown mapping keys lose argument/parameter correlation;
            # retain every value and default until exact mapping facts exist.
            spread = set().union(*args, *(self.elements_mapping(v) if k is None else v
                                         for k, v in kwargs.items()),
                                 *(p.default for p in parameters if p.default is not Parameter.empty))
            bound = {p.name: spread for p in parameters}
        else:
            try:
                binding = Signature(parameters).bind(*args, **kwargs)
            except TypeError:
                return set()  # Python rejects the call before entering its body.
            binding.apply_defaults()
            bound = dict(binding.arguments)
            if f.args.vararg:
                bound[f.args.vararg.arg] = set().union(*bound[f.args.vararg.arg])
            if f.args.kwarg:
                bound[f.args.kwarg.arg] = set().union(*bound[f.args.kwarg.arg].values())
        for p in parameters:
            values = bound[p.name]
            if p.kind in {Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD}:
                node = f.args.vararg if p.kind == Parameter.VAR_POSITIONAL else f.args.kwarg
                tag = "seq" if p.kind == Parameter.VAR_POSITIONAL else "dict"
                self.add(("elem", node), values)
                if tag == "dict":
                    self.add(("key", node), {("str",)})
                values = {(tag, node)}
            self.add(("name", f, p.name), values)
        return self.get(("return", f))

    def value(self, n):
        if self.unsafe:
            return {UNKNOWN}
        if n is None:
            return {("NoneType",)}
        if isinstance(n, ast.Constant):
            return {(type(n.value).__name__,)}
        if isinstance(n, ast.Name):
            if self.r.imports.names.get(n.id) in STRING_CONSTANTS:
                return {("str",)}
            if self.r.is_builtin(n):
                return {("builtin", n.id)}
            return self.get(self.key(n))
        if isinstance(n, (ast.List, ast.Tuple, ast.Set, ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            if hasattr(n, "elts"):
                for i, e in enumerate(n.elts):
                    values = self.elements(self.value(e.value)) if isinstance(e, ast.Starred) else self.value(e)
                    self.add(("elem", n), values)
                    if isinstance(n, (ast.Tuple, ast.List)):
                        self.add(("item", n, i), values)
            else:
                self.add(("elem", n), self.value(n.elt))
            return {("iterator" if isinstance(n, ast.GeneratorExp) else "seq", n)}
        if isinstance(n, ast.Dict):
            for k, v in zip(n.keys, n.values):
                self.add(("key", n), self.value(k) if k is not None else {UNKNOWN})
                self.add(("elem", n), self.value(v) if k is not None else {UNKNOWN})
            return {("dict", n)}
        if self.lambda_defaultdicts and isinstance(n, ast.DictComp):
            self.add(("key", n), self.value(n.key))
            self.add(("elem", n), self.value(n.value))
            return {("dict", n)}
        if isinstance(n, ast.Attribute):
            if isinstance(n.value, ast.Name) and (self.r.imports.modules.get(n.value.id), n.attr) in STRING_CONSTANTS:
                return {("str",)}
            out = set()
            for ob in self.value(n.value):
                if ob[0] == "inst":
                    if ob[1] in self.opaque:
                        out.add(UNKNOWN)
                        continue
                    f = self.methods[ob[1]].get(n.attr)
                    out |= {("bound", f, ob[1])} if f else self.facts.get(("attr", n.attr), set())
                    if f and ("attr", n.attr) in self.facts:
                        out |= self.facts[("attr", n.attr)]
                elif ob[0] == "class":
                    f = self.methods[ob[1]].get(n.attr)
                    out |= {("func", f)} if f else self.facts.get(("attr", n.attr), set())
                elif ob[0] == "defaultdict":
                    out.add(("method", ob, n.attr))
                elif ob[0] in {"str", "bytes", "seq", "dict", "range", "stream", "deque", "set", "counter"}:
                    out.add(("method", ob, n.attr))
                elif ob[0] in SCALARS:
                    # Exact builtin scalar types have no user-defined attributes.
                    if n.attr in {"real", "imag", "numerator", "denominator"}:
                        out.add(("int",))
                else:
                    out.add(UNKNOWN)
            return out
        if isinstance(n, ast.Subscript):
            out = set()
            for ob in self.value(n.value):
                if ob[0] == "seq" and isinstance(ob[1], ast.Tuple) and not any(
                        isinstance(e, ast.Starred) for e in ob[1].elts):
                    try:
                        index = (slice(*(ast.literal_eval(x) if x else None for x in
                                        (n.slice.lower, n.slice.upper, n.slice.step)))
                                 if isinstance(n.slice, ast.Slice) else ast.literal_eval(n.slice))
                        chosen = ob[1].elts[index]
                    except (ValueError, TypeError, IndexError):
                        out |= self.elements({ob})
                    else:
                        if isinstance(index, slice):
                            key = (n, ob[1])
                            if key not in self.slices:
                                self.slices[key] = ast.Tuple(elts=chosen, ctx=ast.Load())
                            out |= self.value(self.slices[key])
                        else:
                            out |= self.get(("item", ob[1], index % len(ob[1].elts)))
                elif ob[0] == "inst":
                    out |= self.method(ob[1], "__getitem__", [self.value(n.slice)])
                elif isinstance(n.slice, ast.Slice) and ob[0] == "seq":
                    self.add(("elem", n), self.elements({ob}))
                    out.add(("seq", n))
                elif isinstance(n.slice, ast.Slice) and ob[0] in {"str", "bytes"}:
                    out.add(ob)
                elif ob[0] == "defaultdict":
                    # A missing lookup inserts one factory-result summary.
                    self.add(("key", ob[1]), self.value(n.slice))
                    out |= self.get(("elem", ob[1]))
                elif ob[0] == "counter":
                    # Counter.__missing__ returns zero without inserting the key.
                    out.add(("int",))
                elif ob[0] == "dict":
                    out |= self.get(("elem", ob[1]))
                else:
                    out |= self.elements({ob})
            return out
        if isinstance(n, ast.IfExp):
            return self.value(n.body) | self.value(n.orelse)
        if isinstance(n, ast.BoolOp):
            return set().union(*(self.value(v) for v in n.values))
        if isinstance(n, (ast.JoinedStr,)):
            return {("str",)}
        if isinstance(n, (ast.Compare,)):
            # Comparisons can return arbitrary objects via user dunders.
            vals = self.value(n.left) | set().union(*(self.value(v) for v in n.comparators))
            return {("bool",)} if self.scalar(vals) else {UNKNOWN}
        if isinstance(n, ast.UnaryOp):
            vals = self.value(n.operand)
            return vals if not vals or self.scalar(vals) else {UNKNOWN}
        if isinstance(n, ast.BinOp):
            left, right = self.value(n.left), self.value(n.right)
            if any(v[0] == "inst" and v[1] in self.opaque for v in left | right):
                return {UNKNOWN}
            if (any(v[0] == "seq" for v in left | right) and
                    any(v[0] == "inst" for v in left | right)):
                return {UNKNOWN}  # sequence repetition can invoke __index__
            if isinstance(n.op, ast.Mod) and left == {("str",)}:
                return {("str",)}
            if self.scalar(left) and self.scalar(right):
                return left | right
            if isinstance(n.op, (ast.Add, ast.Mult)):
                seqs = {v for v in left | right if v[0] == "seq"}
                other = (left | right) - seqs
                if seqs and all(v[0] in {"int", "bool"} for v in other):
                    self.add(("elem", n), self.elements(seqs))
                    return {("seq", n)}
            names = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "truediv",
                     ast.FloorDiv: "floordiv", ast.Mod: "mod", ast.Pow: "pow",
                     ast.MatMult: "matmul", ast.BitOr: "or", ast.BitAnd: "and",
                     ast.BitXor: "xor", ast.LShift: "lshift", ast.RShift: "rshift"}
            suffix = names.get(type(n.op))
            out = set()
            for vals, other, prefix in ((left, right, "__"), (right, left, "__r")):
                for v in vals:
                    if v[0] == "inst" and suffix:
                        method = prefix + suffix + "__"
                        if method in self.methods[v[1]]:
                            out |= self.method(v[1], method, [other])
                    elif v == UNKNOWN:
                        out.add(UNKNOWN)
            return out
        if isinstance(n, ast.Call):
            return self.call(n)
        return {UNKNOWN}

    def _synthetic_pair(self, call):
        key = ("pair", call)
        if key not in self.slices:
            self.slices[key] = ast.Tuple(elts=[ast.Constant(0), ast.Constant(0)], ctx=ast.Load())
        return self.slices[key]

    def call(self, n):
        name = self.r.call_name(n)
        if self.empty_counter(n):
            return {("counter", n)}
        if self.counter_values_call(n):
            return {("counter-values", value[1]) for value in self.value(n.func.value)}
        if self.int_defaultdict(n) or self.lambda_defaultdict(n):
            default = (("int",) if self.int_defaultdict(n) else
                       ("float",) if self._lambda_default_value(n.args[0])[0].startswith("float-")
                       else ("int",))
            self.add(("elem", n), {default})
            return {("defaultdict", n)}
        args = [self.value(a) for a in n.args]
        # Unknown expansion length: every parameter may receive any expanded
        # element. Never shift later arguments as if the star occupied one slot.
        starred = any(isinstance(a, ast.Starred) for a in n.args)
        spread = set().union(*(self.elements(self.value(a.value)) if isinstance(a, ast.Starred)
                                 else self.value(a) for a in n.args))
        kwargs = defaultdict(set)
        for keyword in n.keywords:
            values = self.value(keyword.value)
            if self.final and keyword.arg is None and any(
                    v[0] not in SCALARS | {"seq", "dict", "range"} for v in values):
                # Exact non-mapping builtins simply raise TypeError; no hooks.
                # Mapping expansion invokes keys/__getitem__ before the callee,
                # including when its result and every parameter are ignored.
                self.unsafe = True
            kwargs[keyword.arg].update(values)
        if isinstance(n.func, ast.Attribute) and n.func.attr == "update":
            receivers = self.value(n.func.value)
            if receivers and all(v[0] == "defaultdict" for v in receivers):
                if (len(args) == 1 and not n.keywords
                        and not any(isinstance(a, ast.Starred) for a in n.args)
                        and args[0] and all(v[0] == "dict" for v in args[0])):
                    for receiver in receivers:
                        for mapping in args[0]:
                            self.add(("elem", receiver[1]), self.get(("elem", mapping[1])))
                            self.add(("key", receiver[1]), self.get(("key", mapping[1])))
                    return {("NoneType",)}
                self.escape(receivers | set().union(*args, *kwargs.values()))
                return {UNKNOWN}
        if self.final and any(isinstance(a, ast.Starred) and any(
                v[0] not in SCALARS | {"seq", "dict", "range"}
                for v in self.value(a.value)) for a in n.args):
            self.unsafe = True  # unmodelled iteration/length-hint effects
        if starred:
            args = [spread] * max((len(f.args.posonlyargs + f.args.args) for f in self.nodes
                                  if isinstance(f, ast.FunctionDef)), default=1)
        # These transfer facts are usable only with closed_deques()'s complete
        # effect check, including when no text conversion appears in the file.
        if name == "collections.deque" and len(args) <= 1 and not kwargs and not starred:
            self.add(("elem", n), self.elements(args[0]) if args else set())
            return {("deque", n)}
        if (self.deque_calls or self.class_only) and name == "set" and not args and not kwargs:
            return {("set", n)}
        if name in _HEAPQ_CALLS:
            return self._heapq_call(n, name, args, kwargs)
        if name == "type" and len(n.args) == 1 and not n.keywords:
            return {("type",)}
        if name in {"math.sqrt", "math.sin", "math.cos", "math.tan", "math.fabs",
                    "math.acos", "math.asin", "math.atan", "math.atan2", "math.exp", "math.log"}:
            return {("float",)}
        if name == "divmod":
            if all(self.scalar(v) for v in args):
                self.add(("elem", n), {("int",), ("float",)})
            elif self.final:
                self.add(("elem", n), {UNKNOWN})
            return {("seq", n)}
        if name == "print":
            for v in kwargs.get("file", set()):
                if v[0] == "inst":
                    self.method(v[1], "write", [{("str",)}])
                    self.method(v[1], "flush", [])
                elif v not in {("stream",), ("NoneType",)}:
                    self.escape({v})
            return {("NoneType",)}
        if name in {"isinstance", "issubclass", "callable"}:
            return {("bool",)}
        if name in SCALAR_CALLS:
            return {(SCALAR_CALLS[name],)}
        if name == "range":
            return {("range",)}
        if (self.class_only and name == "enumerate" and len(args) == 1 and not kwargs and not starred
                and args[0] and all(v[0] in {"seq", "str", "bytes", "range", "dict"} for v in args[0])):
            # iterating a builtin container runs no program code and mutates nothing; each step is a fresh
            # (position, element) pair, modelled by one synthetic tuple per call
            pair = self._synthetic_pair(n)
            self.add(("item", pair, 0), {("int",)})
            self.add(("item", pair, 1), self.elements(args[0]))
            self.add(("elem", pair), {("int",)} | self.elements(args[0]))
            self.add(("elem", n), {("seq", pair)})
            return {("iterator", n)}
        if name in {"list", "tuple", "sorted", "reversed", "iter"}:
            if kwargs:
                self.escape(set().union(*kwargs.values()))
            self.add(("elem", n), self.elements(args[0]) if args else set())
            return {("iterator" if name in {"iter", "reversed"} else "seq", n)}
        if name in {"sum", "min", "max", "abs", "round"}:
            if kwargs:
                self.escape(set().union(*kwargs.values()))
            vals = self.elements(args[0]) if name in {"sum", "min", "max"} and len(args) == 1 else set().union(*args)
            vals |= set().union(*kwargs.values())
            return vals if not vals or self.scalar(vals) else {UNKNOWN}
        if name in {"io.StringIO", "io.BytesIO", "open"}:
            return {("stream",)}
        if name in {"_sv_verifier.nondet_int", "_sv_verifier.nondet_bool",
                    "_sv_verifier.nondet_float", "_sv_verifier.nondet_str"}:
            return {(name.rsplit("_", 1)[1],)}
        out = set()
        variants = self.arguments(n)
        if variants is None:
            self.escape(self.value(n.func))
            variants = [args]
            out.add(UNKNOWN)
        for f in self.value(n.func):
            if f[0] == "func":
                for actual in variants:
                    out |= self.invoke(f[1], actual, kwargs)
            elif f[0] == "bound":
                for actual in variants:
                    out |= self.invoke(f[1], [{("inst", f[2])}] + actual, kwargs)
            elif f[0] == "class":
                if "__init__" in self.methods[f[1]]:
                    for actual in variants:
                        self.invoke(self.methods[f[1]]["__init__"], [{("inst", f[1])}] + actual, kwargs)
                out.add(("inst", f[1]))
            elif f[0] == "method":
                ob, method = f[1:]
                if (ob[0] in {"deque", "set"} and not kwargs and not starred and
                        ((ob[0] == "deque" and method == "popleft" and not args) or
                         (method in ({"append", "extend"} if ob[0] == "deque" else
                                     {"add", "update"}) and len(args) == 1)) and
                        # class-only: a set takes only members that hash and compare by identity or value
                        # (running no program code) and is otherwise treated as an unmodelled mutator
                        not (self.class_only and ob[0] == "set" and not (
                            method == "add" and self._plain_key_values(args[0])))):
                    if method == "popleft":
                        out |= self.get(("elem", ob[1]))
                    else:
                        self.add(("elem", ob[1]), self.elements(args[0])
                                 if method in {"extend", "update"} else args[0])
                        out.add(("NoneType",))
                elif ob[0] == "str":
                    if method in {"split", "splitlines", "rsplit"}:
                        self.add(("elem", n), {("str",)})
                        out.add(("seq", n))
                    elif method in {"strip", "rstrip", "lstrip", "replace", "join", "format",
                                    "lower", "upper", "capitalize", "casefold"}:
                        out.add(("str",))
                    else:
                        out.add(UNKNOWN)
                elif ob[0] == "seq":
                    if method not in {"count", "index", "copy"}:
                        self.invalidate({ob})
                    if kwargs:
                        self.escape(set().union(*kwargs.values()))
                    if method in {"append", "insert"} and args:
                        self.add(("elem", ob[1]), args[-1])
                    elif method == "extend" and args:
                        self.add(("elem", ob[1]), self.elements(args[0]))
                    elif method not in {"pop", "remove", "clear", "reverse", "sort", "count", "index", "copy"}:
                        self.add(("elem", ob[1]), {UNKNOWN})
                    if method == "pop":
                        out |= self.get(("elem", ob[1]))
                    elif method == "copy":
                        self.add(("elem", n), self.get(("elem", ob[1])))
                        out.add(("seq", n))
                    elif method in {"count", "index"}:
                        out.add(("int",))
                    else:
                        out.add(("NoneType",))
                elif ob[0] == "dict" and method in {"get", "pop", "setdefault"}:
                    if method != "get":
                        self.invalidate({ob})
                    out |= self.get(("elem", ob[1])) | (args[1] if len(args) > 1 else {("NoneType",)})
                    if method == "setdefault":
                        self.add(("elem", ob[1]), args[1] if len(args) > 1 else {("NoneType",)})
                        self.add(("key", ob[1]), args[0] if args else {UNKNOWN})
                elif ob[0] == "stream":
                    out.add(("str",) if method == "getvalue" else ("int",) if method == "write" else UNKNOWN)
                else:
                    # Unmodelled mutators (including dict.update) may change
                    # fields or nested elements through the receiver.
                    if ob[0] not in SCALARS:
                        self.escape({ob} | set().union(*args, *kwargs.values()))
                    out.add(UNKNOWN)
            elif f[0] in SCALARS or f[0] in {"seq", "dict", "stream", "range"}:
                pass  # These exact values are not callable; a call raises.
            else:
                out.add(UNKNOWN)
                if isinstance(n.func, ast.Attribute) and name is None:
                    self.escape(self.value(n.func.value))
                self.escape(set().union(*args, *kwargs.values()))
        return out

    def _heapq_shape(self, node, *, allow_tuple=False):
        """Whether an argument is an exact literal-backed builtin sequence."""
        values = self.value(node)
        if not values or any(v[0] != "seq" for v in values):
            return None

        def is_sequence_origin(origin, seen):
            if origin in seen:
                return False
            seen = seen | {origin}
            if isinstance(origin, ast.List):
                return True
            if allow_tuple and isinstance(origin, ast.Tuple):
                return True
            if isinstance(origin, ast.Subscript) and isinstance(origin.slice, ast.Slice):
                bases = self.value(origin.value)
                return bool(bases) and all(
                    base[0] == "seq" and is_sequence_origin(base[1], seen) for base in bases
                )
            return False

        if not all(is_sequence_origin(v[1], set()) for v in values):
            return None
        return values

    def _heapq_order_key(self, value, active=None):
        """A homogeneous, value-defined ordering key for heap elements."""
        active = set() if active is None else active
        if value in active:
            return None
        if self.class_only:
            # Numbers of every builtin kind order by value (NaN compares deterministically too).  Instances
            # of an identity-only class are never ordered: ``a < b`` raises TypeError -- text names the class
            # only -- and ``a == b`` is ``a is b``, so tuple comparison, which stops at the first slot that is
            # not identical, depends on object identity and never on an address.
            if len(value) == 1 and value[0] in {"int", "bool", "float"}:
                return ("number",)
            if value[0] == "inst" and self.identity_class(value[1]):
                return ("identity",)
        if value == ("int",) or value == ("bool",):
            return ("int",)
        if value == ("str",):
            return ("str",)
        if value[0] != "seq" or not isinstance(value[1], ast.Tuple):
            return None
        origin = value[1]
        if any(isinstance(item, ast.Starred) for item in origin.elts):
            return None
        active = active | {value}
        slots = []
        for i in range(len(origin.elts)):
            facts = self.get(("item", origin, i))
            if not facts:
                return None
            keys = {self._heapq_order_key(v, active) for v in facts}
            if None in keys or len(keys) != 1:
                return None
            slots.append(next(iter(keys)))
        return ("tuple", tuple(slots))

    def _heapq_values(self, values):
        if not values:
            return None
        keys = {self._heapq_order_key(value) for value in values}
        return None if None in keys or len(keys) != 1 else next(iter(keys))

    def heapq_call_safe(self, node):
        """Only direct, callback-free heapq calls with proven ordered values qualify."""
        name = self.r.call_name(node)
        if name not in _HEAPQ_CALLS or node.keywords or any(
                isinstance(arg, ast.Starred) for arg in node.args):
            return False
        if name in {"heapq.heapify", "heapq.heappop"}:
            if len(node.args) != 1:
                return False
            heap = self._heapq_shape(node.args[0])
            return heap is not None and self._heapq_values(self.elements(heap)) is not None
        if name in {"heapq.heappush", "heapq.heappushpop", "heapq.heapreplace"}:
            if len(node.args) != 2:
                return False
            heap = self._heapq_shape(node.args[0])
            if heap is None:
                return False
            values = self.elements(heap) | self.value(node.args[1])
            return self._heapq_values(values) is not None
        if len(node.args) != 2:
            return False
        n = self.value(node.args[0])
        if not n or any(v not in {("int",), ("bool",)} for v in n):
            return False
        source = self._heapq_shape(node.args[1], allow_tuple=True)
        return source is not None and self._heapq_values(self.elements(source)) is not None

    def _heapq_call(self, node, name, args, kwargs):
        """Track heap item types through recognized mutators; ordering is checked separately."""
        if node.keywords or any(isinstance(arg, ast.Starred) for arg in node.args):
            self.escape(set().union(*args, *kwargs.values()))
            return {UNKNOWN}
        arities = {
            "heapq.heapify": 1, "heapq.heappush": 2, "heapq.heappop": 1,
            "heapq.heappushpop": 2, "heapq.heapreplace": 2,
            "heapq.nsmallest": 2, "heapq.nlargest": 2,
        }
        if len(args) != arities[name]:
            self.escape(set().union(*args, *kwargs.values()))
            return {UNKNOWN}
        if name in {"heapq.nsmallest", "heapq.nlargest"}:
            self.add(("elem", node), self.elements(args[1]))
            return {("seq", node)}
        heap_values = args[0]
        if not heap_values or any(v[0] != "seq" for v in heap_values):
            self.escape(set().union(*args, *kwargs.values()))
            return {UNKNOWN}
        self.invalidate(heap_values)
        origin_items = set().union(*(self.get(("elem", v[1])) for v in heap_values))
        if name == "heapq.heapify":
            return {("NoneType",)}
        if name == "heapq.heappush":
            for heap in heap_values:
                self.add(("elem", heap[1]), args[1])
            return {("NoneType",)}
        if name in {"heapq.heappushpop", "heapq.heapreplace"}:
            for heap in heap_values:
                self.add(("elem", heap[1]), args[1])
        return origin_items | (args[1] if name == "heapq.heappushpop" else set())

    # -- exceptions judged by class alone (``class_only``) --------------------------------
    def identity_class(self, cls, seen=frozenset()):
        """Instances of ``cls`` hash, compare and print by identity, and nothing but ``__init__`` runs.

        A user class whose body is plain methods (no dunder but ``__init__``), plain non-dunder class
        attributes, a docstring or ``pass``; whose bases are ``object`` or other such classes.  Then
        ``==`` is ``is``, ``<`` raises TypeError, ``hash`` is the object's, ``repr`` is the default, and
        no protocol hook (``__getattr__``, ``__bool__``, ``__eq__``, ``__hash__``, ``__lt__`` ...) exists."""
        if cls in seen or cls in self.opaque or cls.keywords or cls.decorator_list:
            return False
        seen = seen | {cls}
        for base in cls.bases:
            if isinstance(base, ast.Name) and base.id == "object" and self.r.is_builtin(base):
                continue
            matches = [k for k in self.classes if isinstance(base, ast.Name) and k.name == base.id]
            if len(matches) != 1 or not self.identity_class(matches[0], seen):
                return False
        for st in cls.body:
            if isinstance(st, ast.FunctionDef):
                if st.decorator_list or (st.name.startswith("__") and st.name != "__init__"):
                    return False
            elif isinstance(st, ast.Pass) or (isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant)
                                              and isinstance(st.value.value, str)):
                continue
            elif isinstance(st, (ast.Assign, ast.AnnAssign)):
                targets = st.targets if isinstance(st, ast.Assign) else [st.target]
                if any(not isinstance(t, ast.Name) or t.id.startswith("__") for t in targets):
                    return False
            else:
                return False
        return all(not name.startswith("__") or name == "__init__" for name in self.methods.get(cls, {}))

    def _plain_key_values(self, values, active=frozenset()):
        """Every fact is a value that dict lookup hashes and compares without running program code:
        an int/bool/str/bytes/None, an identity-only instance, or a literal tuple of such."""
        if not values:
            return False
        for v in values:
            if v in {("int",), ("bool",), ("str",), ("bytes",), ("NoneType",)}:
                continue
            if v[0] == "inst" and self.identity_class(v[1]):
                continue
            if (v[0] == "seq" and isinstance(v[1], ast.Tuple) and v not in active
                    and not any(isinstance(e, ast.Starred) for e in v[1].elts)
                    and all(self._plain_key_values(self.get(("item", v[1], i)), active | {v})
                            for i in range(len(v[1].elts)))):
                continue
            return False
        return True

    def class_only_lookup(self, node):
        """A ``display[key]`` read whose only address-bearing effect is the text of its ``KeyError``.

        Only under ``class_only``, whose caller has shown that nothing can observe an exception in
        flight.  The receiver is a dict display; the probe and every stored key are plain (see
        ``_plain_key_values``), so hashing and equality run no program code and the exception is a
        ``KeyError`` on every path -- never the ``TypeError`` of an unhashable key or whatever a user
        ``__eq__`` raises."""
        if not self.class_only or self.unsafe:
            return False
        if not (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                and not isinstance(node.slice, ast.Slice)):
            return False
        receivers = self.value(node.value)
        if not receivers or any(v[0] != "dict" or not isinstance(v[1], ast.Dict) for v in receivers):
            return False
        # an empty key set is a dict nothing was stored into (every store, update and escape adds a fact)
        return (self._plain_key_values(self.value(node.slice))
                and all(not self.get(("key", v[1])) or self._plain_key_values(self.get(("key", v[1])))
                        for v in receivers))

    def int_defaultdict(self, node):
        return (isinstance(node, ast.Call)
                and self.r.call_name(node) == "collections.defaultdict"
                and len(node.args) == 1 and not node.keywords
                and isinstance(node.args[0], ast.Name) and node.args[0].id == "int"
                and self.r.is_builtin(node.args[0]))

    def empty_counter(self, node):
        if not (isinstance(node, ast.Call) and self.r.call_name(node) == "collections.Counter"
                and not node.args and not node.keywords):
            return False
        if isinstance(node.func, ast.Name):
            return (self.r.imports.names.get(node.func.id) == ("collections", "Counter")
                    and not self.r._rebound(node.func.id))
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            alias = node.func.value.id
            return (node.func.attr == "Counter" and self.r.imports.modules.get(alias) == "collections"
                    and not self.r._rebound(alias))
        return False

    def counter_values_call(self, node):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "values" and not node.args and not node.keywords
                and (values := self.value(node.func.value))
                and all(value[0] == "counter" for value in values))

    def counter_values_max_call(self, node):
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "max" and self.r.is_builtin(node.func)
                and len(node.args) == 1 and not node.keywords
                and not isinstance(node.args[0], ast.Starred)
                and self.counter_values_call(node.args[0]))

    def counter_reason(self, constructors):
        """Only local exact-key Counter reads/writes and max(values()) are order blind."""
        exact = _IntegerProvenance(self.r, counter=True)
        if self.unsafe or exact.unsafe:
            return "unsupported bindings in Counter provenance"
        for node in self.nodes:
            if not isinstance(node, ast.expr) or isinstance(getattr(node, "ctx", None), ast.Store):
                continue
            values = self.value(node)
            origins = {value[1] for value in values if value[0] == "counter"}
            views = {value[1] for value in values if value[0] == "counter-values"}
            if origins:
                if any(value[0] != "counter" for value in values) or any(
                        self.r.function_of(node) is not self.r.function_of(origin) for origin in origins):
                    return "Counter alias has mixed values or crosses a scope"
                parent = self.r.parent.get(node)
                if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node:
                    targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
                    if all(isinstance(target, ast.Name) and self.key(target)[:2] == (
                            "name", self.r.function_of(node)) for target in targets):
                        continue
                if isinstance(parent, ast.Subscript) and parent.value is node:
                    if not exact.counter_key(parent.slice):
                        return "Counter key is not an exact int, str, or flat tuple of ints"
                    if isinstance(parent.ctx, ast.Load):
                        continue
                    writer = self.r.parent.get(parent)
                    if isinstance(writer, (ast.Assign, ast.AnnAssign)) and exact.value(writer.value) == {("int",)}:
                        continue
                    if (isinstance(writer, ast.AugAssign) and isinstance(writer.op, ast.Add)
                            and exact.value(writer.value) == {("int",)}):
                        continue
                    return "Counter write is not a plain integer subscript assignment"
                if isinstance(parent, (ast.If, ast.IfExp)) and parent.test is node:
                    continue
                if isinstance(parent, ast.Attribute) and parent.value is node and parent.attr == "values":
                    call = self.r.parent.get(parent)
                    outer = self.r.parent.get(call) if isinstance(call, ast.Call) else None
                    if (isinstance(call, ast.Call) and call.func is parent and not call.args and not call.keywords
                            and isinstance(outer, ast.Call) and self.counter_values_max_call(outer)):
                        continue
                return "Counter escapes local subscript, truth, and max(values()) use"
            if views:
                if any(value[0] != "counter-values" for value in values) or any(
                        self.r.function_of(node) is not self.r.function_of(origin) for origin in views):
                    return "Counter values view has mixed values or crosses a scope"
                parent = self.r.parent.get(node)
                if isinstance(parent, ast.Call) and self.counter_values_max_call(parent):
                    continue
                return "Counter values view escapes direct max() use"
        if any(not exact.empty_counter(origin) for origin in constructors):
            return "Counter constructor is not an exact empty builtin Counter"
        if any(exact.get(("elem", origin)) - {("int",)} for origin in constructors):
            return "Counter values are not exact ints"
        return None

    def _canonical_defaultdict_call(self, node):
        if not isinstance(node, ast.Call) or self.r.call_name(node) != "collections.defaultdict":
            return False
        if any(isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
               and n.attr in {"defaultdict", "float"} for n in self.nodes):
            return False
        func = node.func
        if isinstance(func, ast.Name):
            return not self.r._rebound(func.id)
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            return not self.r._rebound(func.value.id)
        return False

    def _lambda_default_value(self, node):
        """Return a scalar tag only for a pure, zero-argument inline factory."""
        if (node.args.posonlyargs or node.args.args or node.args.kwonlyargs
                or node.args.vararg or node.args.kwarg or node.args.defaults
                or any(node.args.kw_defaults)):
            return None
        body = node.body
        sign = 1
        literal = body
        if isinstance(body, ast.UnaryOp) and isinstance(body.op, (ast.UAdd, ast.USub)):
            sign = -1 if isinstance(body.op, ast.USub) else 1
            literal = body.operand
        if isinstance(literal, ast.Constant) and type(literal.value) in {int, float}:
            value = sign * literal.value
            if type(value) is float and value != value:
                return None
            return ("int",) if type(value) is int else self._float_tag(value)
        if (isinstance(body, ast.Call) and isinstance(body.func, ast.Name)
                and body.func.id == "float" and self.r.is_builtin(body.func)
                and not any(isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                            and n.attr == "float" for n in self.nodes)
                and len(body.args) == 1
                and not body.keywords and isinstance(body.args[0], ast.Constant)
                and type(body.args[0].value) is str):
            try:
                value = float(body.args[0].value)
            except ValueError:
                return None
            if value != value:
                return None
            return self._float_tag(value)
        return None

    def _defaultdict_return_owners(self):
        owners = {}
        origins = set()
        for fn in self.nodes:
            if not isinstance(fn, ast.FunctionDef) or not fn.body:
                continue
            terminal = fn.body[-1]
            if not (isinstance(terminal, ast.Return) and isinstance(terminal.value, ast.Name)):
                continue
            name = terminal.value.id
            returns = [n for n in self.nodes if isinstance(n, ast.Return)
                       and self.r.function_of(n) is fn]
            if len(returns) != 1:
                continue
            writes = [n for n in self.nodes if isinstance(n, ast.Name) and n.id == name
                      and isinstance(n.ctx, (ast.Store, ast.Del)) and self.r.function_of(n) is fn]
            if len(writes) != 1:
                continue
            constructors = []
            for call in self.nodes:
                if not isinstance(call, ast.Call) or self.r.function_of(call) is not fn \
                        or not self.lambda_defaultdict(call):
                    continue
                assignment = self.r.parent.get(call)
                targets = (assignment.targets if isinstance(assignment, ast.Assign)
                           else [assignment.target] if isinstance(assignment, ast.AnnAssign) else [])
                if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                    constructors.append(call)
            if len(constructors) == 1:
                owners[fn] = name
                origins.add(constructors[0])
        return owners, origins

    @staticmethod
    def _float_tag(value):
        if value != value:
            return None
        if value == float("inf"):
            return ("float-posinf",)
        if value == float("-inf"):
            return ("float-neginf",)
        return ("float-finite",)

    def lambda_defaultdict(self, node):
        return (isinstance(node, ast.Call)
                and self._canonical_defaultdict_call(node)
                and len(node.args) == 1 and not node.keywords
                and isinstance(node.args[0], ast.Lambda)
                and node.args[0] in self.lambda_defaultdicts)

    def recognized_defaultdict(self, node):
        return self.int_defaultdict(node) or self.lambda_defaultdict(node)

    def defaultdict_candidate(self, node):
        return (isinstance(node, ast.Call) and self.r.call_name(node) == "collections.defaultdict"
                and bool(node.args) and isinstance(node.args[0], ast.Lambda))

    def defaultdict_reason(self, constructors):
        """Admit exact int factories and the separately closed numeric-lambda subset."""
        numeric = any(self.lambda_defaultdict(n) for n in constructors)
        if numeric and any(not self.lambda_defaultdict(n) for n in constructors):
            return "unsupported defaultdict lambda factory or constructor shape"
        if numeric and any(
                isinstance(n, ast.JoinedStr)
                or isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mod)
                   and isinstance(n.left, ast.Constant) and type(n.left.value) is str
                or isinstance(n, ast.Call) and (
                    self.r.call_name(n) in {"print", "str", "repr", "ascii", "format"}
                    or isinstance(n.func, ast.Attribute) and n.func.attr in {"format", "format_map"})
                for n in self.nodes):
            return "lambda-defaultdict program uses explicit text conversion"
        exact = _IntegerProvenance(self.r, numeric_defaultdicts=numeric)
        if self.unsafe or exact.unsafe:
            return "unsupported bindings in defaultdict provenance"
        for call in self.nodes:
            if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                    and call.func.attr == "update"):
                continue
            receivers = exact.value(call.func.value)
            origins = {v[1] for v in receivers if v[0] == "defaultdict"}
            if not origins:
                continue
            if not numeric:
                return "defaultdict update is outside the direct mapping subset"
            if (len(call.args) != 1 or call.keywords
                    or any(isinstance(a, ast.Starred) for a in call.args)):
                return "defaultdict update is outside the direct mapping subset"
            sources = exact.value(call.args[0])
            if not sources or any(v[0] != "dict" for v in sources):
                return "defaultdict update source is not a tracked mapping"
            if any(not exact.dict_source_ok(v[1]) for v in sources):
                return "defaultdict update mapping has unsupported keys or values"
        if numeric:
            for node in self.nodes:
                if isinstance(node, ast.BinOp):
                    values = exact.value(node)
                    if not values or not values <= _NUMERIC:
                        return "lambda-defaultdict arithmetic has unsupported numeric values"
                if isinstance(node, ast.Compare):
                    values = exact.value(node.left) | set().union(*(exact.value(v) for v in node.comparators))
                    mapping_compare = (len(node.ops) == 1
                                       and isinstance(node.ops[0], (ast.Eq, ast.NotEq))
                                       and any(v[0] == "defaultdict" for v in values))
                    if mapping_compare:
                        continue
                    if (not values or not values <= _NUMERIC
                            or not all(isinstance(op, (ast.Eq, ast.NotEq, ast.Lt,
                                                       ast.LtE, ast.Gt, ast.GtE))
                                       for op in node.ops)):
                        return "lambda-defaultdict comparison has unsupported numeric values"
        for node in self.nodes:
            if not isinstance(node, ast.expr) or isinstance(getattr(node, "ctx", None), ast.Store):
                continue
            values = self.value(node)
            origins = {v[1] for v in values if v[0] == "defaultdict"}
            if not origins:
                continue
            if any(v[0] != "defaultdict" for v in values) or any(
                    self.r.function_of(node) is not self.r.function_of(origin)
                    and not (origin in self.defaultdict_return_origins and
                             self._allowed_returned_use(node)) for origin in origins):
                return "defaultdict alias has mixed values or crosses a scope"
            parent = self.r.parent.get(node)
            if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node:
                targets = parent.targets if isinstance(parent, ast.Assign) else [parent.target]
                if all(isinstance(t, ast.Name) and self.key(t)[:2] == (
                        "name", self.r.function_of(node)) for t in targets):
                    continue
            if isinstance(parent, ast.Return) and parent.value is node:
                owner = self.r.function_of(node)
                if owner in self.defaultdict_return_owners and node.id == self.defaultdict_return_owners[owner]:
                    continue
            if isinstance(parent, ast.Compare) and len(parent.ops) == 1 and isinstance(
                    parent.ops[0], (ast.Eq, ast.NotEq)):
                other = parent.comparators[0] if parent.left is node else parent.left
                other_values = exact.value(other)
                if (isinstance(other, (ast.Dict, ast.DictComp, ast.Name)) and other_values
                        and all(v[0] == "dict" and exact.dict_source_ok(v[1]) for v in other_values)
                        and values <= {("defaultdict", origin) for origin in self.defaultdict_return_origins}):
                    continue
            if isinstance(parent, ast.Subscript) and parent.value is node:
                if not exact.integer_key(parent.slice):
                    return "defaultdict key is not an exact int or flat tuple of ints"
                if isinstance(parent.ctx, ast.Load):
                    continue
                writer = self.r.parent.get(parent)
                expected = _NUMERIC if numeric else {("int",)}
                if (isinstance(parent.ctx, ast.Store) and isinstance(writer, (ast.Assign, ast.AnnAssign))
                        and exact.value(writer.value) and exact.value(writer.value) <= expected):
                    continue
                return "defaultdict write is not a plain integer subscript assignment"
            if (isinstance(parent, ast.Attribute) and parent.value is node
                    and parent.attr == "update" and isinstance(self.r.parent.get(parent), ast.Call)
                    and self.r.parent[parent].func is parent):
                continue
            return "defaultdict escapes local subscript use"
        allowed = _NUMERIC if numeric else {("int",)}
        if any(not exact.get(("elem", origin)) or
               not exact.get(("elem", origin)) <= allowed for origin in constructors):
            return "defaultdict values are outside the exact numeric subset"
        return None

    def _allowed_returned_use(self, node):
        if (isinstance(node, ast.Name)
                and self.key(node)[:2] != ("name", self.r.function_of(node))):
            return False
        parent = self.r.parent.get(node)
        if isinstance(parent, ast.Subscript) and parent.value is node:
            return isinstance(parent.ctx, ast.Load)
        if isinstance(parent, ast.Compare) and len(parent.ops) == 1 and isinstance(
                parent.ops[0], (ast.Eq, ast.NotEq)):
            return True
        if isinstance(parent, (ast.Assign, ast.AnnAssign)) and parent.value is node and isinstance(node, ast.Call):
            return True
        return False

    # -- a read that cannot return ----------------------------------------------------
    def inert_empty_index(self, node):
        """Is ``node`` a Load subscript ``X[<int literal>]`` that ALWAYS raises the builtin
        ``IndexError`` instead of yielding a value?

        True for ``[][k]`` and for ``name[k]`` where ``name`` is a function-local list that
        is ``[]`` and provably stays ``[]`` (``_inert_empty_list``).  The read then raises
        ``IndexError("list index out of range")`` -- fixed text, no address -- before any
        enclosing expression sees a result, so a ``d[name[k]]`` lookup never computes the key
        whose ``KeyError`` text would otherwise be shown.  It does NOT say the read is
        unreachable or that the program is free of ``IndexError``: the run that observes the
        error still decides the verdict.  Syntactic and independent of the value facts, whose
        empty sets also mean "unmodelled"; the caller consults it only after the unsafe veto."""
        if self.unsafe:
            return False
        if not (isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load)
                and _int_literal(node.slice)):
            return False
        base = node.value
        if isinstance(base, ast.List):
            return not base.elts
        if not (isinstance(base, ast.Name) and isinstance(base.ctx, ast.Load)):
            return False
        scope = self.r.function_of(base)
        if not isinstance(scope, ast.FunctionDef):
            return False
        key = (scope, base.id)
        if key not in self._inert_lists:
            self._inert_lists[key] = self._inert_empty_list(scope, base.id)
        return self._inert_lists[key]

    def _inert_empty_list(self, fn, name):
        """``name`` is bound exactly once in ``fn`` -- by ``name = []`` as one of ``fn``'s own
        statements -- and every other mention of the identifier anywhere in ``fn`` (header,
        dead code and nested scopes included) is a read that cannot add an element."""
        first = [i for i, st in enumerate(fn.body)
                 if isinstance(st, ast.Assign) and len(st.targets) == 1
                 and isinstance(st.targets[0], ast.Name) and st.targets[0].id == name
                 and isinstance(st.value, ast.List) and not st.value.elts]
        if len(first) != 1:
            return False
        first = first[0]
        header = [fn.args, fn.returns, *fn.decorator_list, *getattr(fn, "type_params", ())]
        if any(_mentions(n, name) for part in header if part is not None for n in ast.walk(part)):
            return False
        for i, statement in enumerate(fn.body):
            for n in ast.walk(statement):
                if not _mentions(n, name):
                    continue
                if i == first and n is statement.targets[0]:
                    continue
                # before the binding it would be an unbound read; at it, a second mention
                if i <= first or not self._inert_read(n, fn):
                    return False
        return True

    def _inert_read(self, n, fn):
        """A Load of the empty local that is a truth test, ``name[<int literal>]`` or
        ``name.pop()`` -- the three reads that never put an element in, and never hand the
        list itself to anything else."""
        parent = self.r.parent
        if not (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)):
            return False
        scope = parent.get(n)
        while scope is not None and scope is not fn:
            if isinstance(scope, _NESTED_SCOPES):
                return False  # captured or re-scoped: another frame could write to it
            scope = parent.get(scope)
        if scope is None:
            return False
        up = parent[n]
        if isinstance(up, ast.Subscript) and up.value is n:
            return isinstance(up.ctx, ast.Load) and _int_literal(up.slice)
        if isinstance(up, ast.Attribute) and up.value is n:
            call = parent.get(up)
            return (up.attr == "pop" and isinstance(up.ctx, ast.Load) and isinstance(call, ast.Call)
                    and call.func is up and not call.args and not call.keywords)
        child = n
        while isinstance(up, ast.BoolOp):  # the value of ``a and name`` is only ever tested
            child, up = up, parent.get(up)
        if isinstance(up, ast.UnaryOp) and isinstance(up.op, ast.Not):
            return True
        return isinstance(up, (ast.If, ast.While, ast.IfExp, ast.Assert)) and up.test is child

    def step(self, n):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            values = self.value(n.value)
            if isinstance(n, ast.AugAssign):
                previous = self.value(n.target)
                if self.final and not (self.scalar(previous) and self.scalar(values)):
                    # In-place protocols may mutate aliases or replace the
                    # receiver with an arbitrary return value.
                    self.escape(previous | values)
                    values.add(UNKNOWN)
                values |= previous
            for t in n.targets if isinstance(n, ast.Assign) else [n.target]:
                self.bind(t, values)
        elif isinstance(n, (ast.For, ast.comprehension)):
            self.bind(n.target, self.elements(self.value(n.iter)))
        elif isinstance(n, ast.With):
            for item in n.items:
                if item.optional_vars:
                    self.bind(item.optional_vars, self.value(item.context_expr))
        elif isinstance(n, ast.FunctionDef):
            # shortcut: only the unique terminal direct return omits implicit None.
            falls_off = n not in self.defaultdict_return_owners and not (
                self.class_only and _terminates(n.body))
            if falls_off or n in self.generators:
                self.add(("return", n), {UNKNOWN} if n in self.generators else {("NoneType",)})
            par = self.r.parent.get(n)
            key = ("attr", n.name) if isinstance(par, ast.ClassDef) else (
                "name", self.r.function_of(n), n.name)
            if not isinstance(par, ast.ClassDef):
                self.add(key, {("func", n)})
        elif isinstance(n, ast.ClassDef):
            self.add(("name", self.r.function_of(n), n.name), {("class", n)})
            for f in self.methods[n].values():
                params = f.args.posonlyargs + f.args.args
                if params:
                    self.add(("name", f, params[0].arg), {("inst", n)})
        elif isinstance(n, ast.Return):
            self.add(("return", self.r.function_of(n)), self.value(n.value))
        elif isinstance(n, ast.Call):
            self.value(n)


class _IntegerProvenance(Provenance):
    """Narrow integer facts using the same all-writes fixpoint and Python binder.

    General address-free scalar unions are not exact types: division, powers,
    scalar attributes and in-place operations must not certify integer keys.
    """

    def __init__(self, resolver, *, counter=False, numeric_defaultdicts=False):
        self.counter_mode = counter
        self.numeric_defaultdicts = numeric_defaultdicts
        super().__init__(resolver)
        # This slice has no object-protocol model; unrelated callbacks must not
        # mutate an integer carrier behind the narrower producer transfers.
        self.unsafe |= bool(self.classes)

    def integer_key(self, node):
        values = self.value(node)
        return bool(values) and all(v == ("int",) or (
            v[0] == "seq" and isinstance(v[1], ast.Tuple)
            and not any(isinstance(e, ast.Starred) for e in v[1].elts)
            and all(self.get(("item", v[1], i)) == {("int",)} for i in range(len(v[1].elts))))
            for v in values)

    def counter_key(self, node):
        values = self.value(node)
        return bool(values) and all(v == ("int",) or v == ("str",) or (
            v[0] == "seq" and isinstance(v[1], ast.Tuple)
            and not any(isinstance(e, ast.Starred) for e in v[1].elts)
            and all(self.get(("item", v[1], i)) == {("int",)} for i in range(len(v[1].elts))))
            for v in values)

    def value(self, n):
        if self.numeric_defaultdicts and isinstance(n, ast.Constant) and type(n.value) is float:
            return {self._float_tag(n.value)}
        if self.numeric_defaultdicts and isinstance(n, ast.DictComp):
            self.add(("key", n), self.value(n.key))
            self.add(("elem", n), self.value(n.value))
            return {("dict", n)}
        if self.numeric_defaultdicts and isinstance(n, ast.Compare):
            values = self.value(n.left) | set().union(*(self.value(v) for v in n.comparators))
            if values and values <= _NUMERIC and all(isinstance(op, (ast.Eq, ast.NotEq, ast.Lt,
                    ast.LtE, ast.Gt, ast.GtE)) for op in n.ops):
                return {("bool",)}
        if isinstance(n, ast.BinOp):
            left, right = self.value(n.left), self.value(n.right)
            if self.numeric_defaultdicts and isinstance(n.op, ast.Add):
                if left and right and left <= _NUMERIC and right <= _NUMERIC:
                    results = set()
                    for a in left:
                        for b in right:
                            if a == b == ("int",):
                                results.add(("int",))
                            elif a == ("int",) and b in {("float-posinf",), ("float-neginf",)}:
                                results.add(b)
                            elif b == ("int",) and a in {("float-posinf",), ("float-neginf",)}:
                                results.add(a)
                            elif a == b and a in {("float-posinf",), ("float-neginf",)}:
                                results.add(a)
                            else:
                                return {UNKNOWN}
                    return results
                return set() if not self.final and (not left or not right) else {UNKNOWN}
            if isinstance(n.op, (ast.Add, ast.Sub)):
                if not left or not right:
                    return set() if not self.final else {UNKNOWN}
                if left == right == {("int",)}:
                    return {("int",)}
            return {UNKNOWN}
        if isinstance(n, ast.UnaryOp):
            values = self.value(n.operand)
            if self.numeric_defaultdicts and isinstance(n.op, ast.UAdd) and values and values <= _NUMERIC:
                return values
            if self.numeric_defaultdicts and isinstance(n.op, ast.USub) and values and values <= _NUMERIC:
                return {("float-neginf",) if v == ("float-posinf",) else
                        ("float-posinf",) if v == ("float-neginf",) else v for v in values}
            if isinstance(n.op, (ast.UAdd, ast.USub)) and values <= {("int",)}:
                return values
            return {UNKNOWN}
        if isinstance(n, ast.Attribute):
            return {UNKNOWN}
        if isinstance(n, ast.Subscript):
            values = self.value(n.value)
            if not values and not self.final:
                return set()
            if not values or not all(v[0] == "defaultdict" or self.counter_mode and v[0] == "counter" or (
                    v[0] == "seq" and isinstance(v[1], (ast.List, ast.Tuple))) for v in values):
                return {UNKNOWN}
            if any(v[0] == "counter" for v in values) and not self.counter_key(n.slice):
                return set() if not self.final else {UNKNOWN}
            if any(v[0] == "seq" for v in values) and self.value(n.slice) != {("int",)}:
                return set() if not self.final and not self.value(n.slice) else {UNKNOWN}
        if isinstance(n, (ast.Set, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.Compare)):
            return {UNKNOWN}
        return super().value(n)

    def call(self, n):
        if self.lambda_defaultdict(n):
            self.add(("elem", n), {self._lambda_default_value(n.args[0])})
            return {("defaultdict", n)}
        if self.int_defaultdict(n) or self.counter_mode and self.empty_counter(n):
            return super().call(n)
        name = self.r.call_name(n)
        args = [self.value(a) for a in n.args]
        if self.counter_mode and self.counter_values_call(n):
            return super().call(n)
        if self.counter_mode and self.counter_values_max_call(n):
            return {("int",)}
        if isinstance(n.func, ast.Attribute) and n.func.attr == "update":
            receivers = self.value(n.func.value)
            if receivers and all(v[0] == "defaultdict" for v in receivers):
                if (len(n.args) == 1 and not n.keywords
                        and not any(isinstance(a, ast.Starred) for a in n.args)
                        and args and args[0] and all(v[0] == "dict" for v in args[0])):
                    for receiver in receivers:
                        for mapping in args[0]:
                            self.add(("elem", receiver[1]), self.get(("elem", mapping[1])))
                            self.add(("key", receiver[1]), self.get(("key", mapping[1])))
                    return {("NoneType",)}
                if self.final:
                    self.unsafe = True
                return {UNKNOWN}
        parent = self.r.parent.get(n)
        if (self.numeric_defaultdicts and isinstance(parent, ast.Lambda)
                and parent.body is n and self._lambda_default_value(parent) in _NUMERIC):
            return {self._lambda_default_value(parent)}
        if (self.numeric_defaultdicts and name == "float" and len(n.args) == 1
                and not n.keywords and isinstance(n.args[0], ast.Constant)
                and type(n.args[0].value) is str and self.r.is_builtin(n.func)):
            try:
                parsed = float(n.args[0].value)
            except ValueError:
                return {UNKNOWN}
            tag = self._float_tag(parsed)
            return {tag} if tag else {UNKNOWN}
        if name in {"range", "len", "max"} and not n.keywords and not any(
                isinstance(a, ast.Starred) for a in n.args):
            if any(not values for values in args) and not self.final:
                return set()
            if name == "range" and 1 <= len(args) <= 3 and all(v == {("int",)} for v in args):
                return {("range",)}
            if name == "len" and len(args) == 1 and args[0] and all(
                    v[0] == "range" or v[0] == "seq" and isinstance(v[1], (ast.List, ast.Tuple))
                    for v in args[0]):
                return {("int",)}
            if (self.counter_mode and name == "len" and len(args) == 1
                    and args[0] == {("str",)}):
                return {("int",)}
            if name == "max" and len(args) >= 2 and all(v == {("int",)} for v in args):
                return {("int",)}
        if (self.numeric_defaultdicts and name in {"min", "max"} and len(args) == 2
                and not n.keywords and not any(isinstance(a, ast.Starred) for a in n.args)
                and all(v and v <= _NUMERIC for v in args)):
            return set().union(*args)
        if self.numeric_defaultdicts and name in {"min", "max"} and any(not values for values in args) and not self.final:
            return set()
        # Retain mutation/escape effects and the existing argument binder, but
        # never inherit an unsupported builtin/method's optimistic result tag.
        values = super().call(n)
        callees = self.value(n.func)
        if name is None and callees and all(v[0] == "func" for v in callees):
            return values
        if name == "print" and not n.keywords and all(self.scalar(v) for v in args):
            return values
        if self.final:
            self.unsafe = True
        return {UNKNOWN}

    def step(self, n):
        if isinstance(n, ast.AugAssign):
            if (self.counter_mode and isinstance(n.target, ast.Subscript) and isinstance(n.op, ast.Add)
                    and self.value(n.target.value)
                    and all(value[0] == "counter" for value in self.value(n.target.value))
                    and self.counter_key(n.target.slice) and self.value(n.target) == {("int",)}
                    and self.value(n.value) == {("int",)}):
                self.bind(n.target, {("int",)})
                return
            self.escape(self.value(n.target) | self.value(n.value))
            self.bind(n.target, {UNKNOWN})
        else:
            super().step(n)

    def dict_source_ok(self, origin):
        if not isinstance(origin, (ast.Dict, ast.DictComp)):
            return False
        if isinstance(origin, ast.Dict):
            pairs = zip(origin.keys, origin.values)
            for key, value in pairs:
                if key is None or not self.integer_key(key):
                    return False
                values = self.value(value)
                if not values or not values <= _NUMERIC:
                    return False
        else:
            if any(gen.ifs or not (self.value(gen.iter) and all(v[0] == "range" for v in self.value(gen.iter)))
                   for gen in origin.generators):
                return False
            if not self.integer_key(origin.key) or not self.value(origin.value) \
                    or not self.value(origin.value) <= _NUMERIC:
                return False
        keys = self.get(("key", origin))
        for key in keys:
            if key == ("int",):
                continue
            if (key[0] != "seq" or not isinstance(key[1], ast.Tuple)
                    or any(self.get(("item", key[1], i)) != {("int",)}
                           for i in range(len(key[1].elts)))):
                return False
        values = self.get(("elem", origin))
        return not values or values <= _NUMERIC
