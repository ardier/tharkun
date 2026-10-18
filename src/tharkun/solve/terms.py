"""Typed pure term construction and stable keys; no solver or target execution.

Boolean order is part of identity. Translators refuse unsupported or partial
evaluation; casts require actual exact types, never annotation coercion.
"""
from __future__ import annotations

import ast
import json
import keyword
import math
import operator
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

from .predicates import InvalidPredicate, check_predicate


class Sort(str, Enum):
    INT = "Int"
    BOOL = "Bool"
    F64 = "F64"
    TAG = "Tag"
    PY = "Py"
    SEQ = "Seq"
    MAP = "Map"


INT, BOOL, F64, TAG, PY, SEQ, MAP = Sort
VERSION = 1
CONTAINER_VERSION = 2
_PRIMITIVES = frozenset({"int", "bool", "str", "bytes"})
_V2_OPS = frozenset({"key", "as_seq", "as_map", "len", "contains"})
_TAGS = frozenset({"int", "bool", "float", "str", "bytes", "list", "tuple",
                   "dict", "set", "frozenset", "NoneType", "object"})


@dataclass(frozen=True)
class Refused:
    reason: str

    def __str__(self) -> str:
        return f"store_only:{self.reason}"

    def __bool__(self):
        raise TypeError("a refusal has no truth value")


@dataclass(frozen=True)
class Term:
    op: str
    sort: Sort
    args: tuple

    def __post_init__(self):
        if type(self.op) is not str or not isinstance(self.sort, Sort) or type(self.args) is not tuple:
            raise ValueError("term requires an operation, Sort and immutable tuple arguments")
        if self.op == "var":
            valid = (len(self.args) == 1 and type(self.args[0]) is str
                     and self.args[0].isidentifier() and not keyword.iskeyword(self.args[0]))
        elif self.op == "lit":
            valid = len(self.args) == 1 and (
                (self.sort == INT and type(self.args[0]) is int)
                or (self.sort == BOOL and type(self.args[0]) is bool)
                or (self.sort == F64 and type(self.args[0]) is str
                    and float.fromhex(self.args[0]).hex() == self.args[0]))
        elif self.op == "tag":
            valid = (self.sort == TAG and len(self.args) == 1
                     and type(self.args[0]) is str and self.args[0] in _TAGS)
        elif self.op == "key":
            valid = (self.sort == PY and len(self.args) == 2
                     and type(self.args[0]) is str and self.args[0] in {"str", "bytes"}
                     and type(self.args[1]) is str
                     and (self.args[0] == "str"
                          or bytes.fromhex(self.args[1]).hex() == self.args[1]))
        else:
            valid = all(isinstance(a, Term) for a in self.args) and _signature(self.op, self.sort, self.args)
        if not valid:
            raise ValueError(f"invalid {self.op!r} term of sort {self.sort.value}")

    def __bool__(self):
        raise TypeError("a symbolic term has no truth value")


def _signature(op, sort, args):
    if op in ("as_seq", "as_map"):
        required = 2 if op == "as_seq" else 1
        return (len(args) in (required, required + 1)
                and args[0].op == "var" and args[0].sort == PY
                and sort == (SEQ if op == "as_seq" else MAP)
                and (op == "as_map" or (args[1].op == "tag"
                                       and args[1].args[0] in {"list", "tuple"}))
                and (len(args) == required or (args[-1].op == "tag"
                                               and args[-1].args[0] in _PRIMITIVES)))
    if op == "len":
        return len(args) == 1 and args[0].sort in (SEQ, MAP, PY) and sort == INT
    if op == "contains":
        return (len(args) == 2 and args[0].sort in (SEQ, MAP)
                and args[1].sort in (INT, BOOL, PY) and sort == BOOL)
    if op == "typeis":
        return len(args) == 2 and args[0].sort in (PY, INT, BOOL, F64) and args[1].sort == TAG and sort == BOOL
    casts = {"as_int": (PY, INT), "as_bool": (PY, BOOL), "as_f64": (PY, F64),
             "b2i": (BOOL, INT), "i2f": (INT, F64)}
    if op in casts:
        return (len(args) == 1 and (args[0].sort, sort) == casts[op]
                and (not op.startswith("as_") or args[0].op == "var"))
    if op in ("and", "or"):
        return bool(args) and sort == BOOL and all(a.sort == BOOL for a in args)
    if op == "not":
        return len(args) == 1 and args[0].sort == sort == BOOL
    if op == "neg":
        return len(args) == 1 and args[0].sort == sort and sort in (INT, F64)
    if len(args) != 2 or args[0].sort != args[1].sort:
        return False
    operand = args[0].sort
    if op in ("eq", "ne"):
        return sort == BOOL and operand in (INT, BOOL, F64, TAG)
    if op in ("lt", "le", "gt", "ge"):
        return sort == BOOL and operand in (INT, F64)
    if op in ("add", "sub", "mul"):
        return sort == operand and sort in (INT, F64)
    if op in ("floordiv", "mod"):
        return sort == operand == INT
    return False


def _make(op, sort, *args):
    for arg in args:
        if isinstance(arg, Refused):
            return arg
    try:
        return Term(op, sort, tuple(args))
    except (ValueError, TypeError, OverflowError) as exc:
        return Refused(str(exc))


def var(name: str, sort: Sort = PY) -> Term | Refused:
    return _make("var", sort, name)


def literal(value) -> Term | Refused:
    if type(value) is bool:
        return _make("lit", BOOL, value)
    if type(value) is int:
        return _make("lit", INT, value)
    if type(value) is float:
        return _make("lit", F64, value.hex())
    return Refused("literal is not an exact int, bool or float")


def key_literal(value) -> Term | Refused:
    """Exact str/bytes key or length operand; not a general string value model."""
    if type(value) is not str and type(value) is not bytes:
        return Refused("key literal requires exact str or bytes")
    return _make("key", PY, type(value).__name__, value if type(value) is str else value.hex())


def as_seq(value, kind="list", *, primitive_domain=None):
    """Pre-state list/tuple view; a domain label is not proof of its contents."""
    args = [value, tag(kind)]
    if primitive_domain is not None:
        args.append(tag(primitive_domain))
    return _make("as_seq", SEQ, *args)


def as_map(value, *, primitive_domain=None):
    """Pre-state dict-key view; values are irrelevant and are never inspected."""
    args = [value]
    if primitive_domain is not None:
        args.append(tag(primitive_domain))
    return _make("as_map", MAP, *args)


def length(value): return _make("len", INT, value)
def contains(view, key): return _make("contains", BOOL, view, key)


def tag(name: str) -> Term | Refused:
    return _make("tag", TAG, name)


def typeis(value, type_name: str) -> Term | Refused:
    return _make("typeis", BOOL, value, tag(type_name))


def as_int(value): return _make("as_int", INT, value)
def as_bool(value): return _make("as_bool", BOOL, value)
def as_f64(value): return _make("as_f64", F64, value)
def b2i(value): return _make("b2i", INT, value)
def i2f(value): return _make("i2f", F64, value)
def and_(*values): return _make("and", BOOL, *values)
def or_(*values): return _make("or", BOOL, *values)
def not_(value): return _make("not", BOOL, value)
def eq(left, right): return _make("eq", BOOL, left, right)
def ne(left, right): return _make("ne", BOOL, left, right)
def lt(left, right): return _make("lt", BOOL, left, right)
def le(left, right): return _make("le", BOOL, left, right)
def gt(left, right): return _make("gt", BOOL, left, right)
def ge(left, right): return _make("ge", BOOL, left, right)


def _numeric(op, left, right=None):
    sort = left.sort if isinstance(left, Term) else INT
    return _make(op, sort, left) if right is None else _make(op, sort, left, right)


def add(left, right): return _numeric("add", left, right)
def sub(left, right): return _numeric("sub", left, right)
def mul(left, right): return _numeric("mul", left, right)
def neg(value): return _numeric("neg", value)
# These are partial values. Eval/render/lower must prove or guard right != 0.
def floordiv(left, right): return _make("floordiv", INT, left, right)
def mod(left, right): return _make("mod", INT, left, right)


def _integer_text(value):
    # Decimal chunks avoid Python's configurable int-to-string digit limit.
    sign = "-" if value < 0 else ""
    value = abs(value)
    chunks = []
    while value >= 10**9:
        value, rest = divmod(value, 10**9)
        chunks.append(f"{rest:09d}")
    return sign + str(value) + "".join(reversed(chunks))


def canon(term: Term | Refused) -> str | Refused:
    """Versioned typed s-expression; preserves argument order and exact sorts."""
    if isinstance(term, Refused):
        return term
    if not isinstance(term, Term):
        return Refused("canonicalization requires a Term")

    version = VERSION

    def emit(value):
        nonlocal version
        if isinstance(value, Term):
            if value.op in _V2_OPS or value.sort in (SEQ, MAP):
                version = CONTAINER_VERSION
            return f"({value.op} {value.sort.value} {' '.join(emit(a) for a in value.args)})"
        if type(value) is int:
            return _integer_text(value)
        return json.dumps(value, ensure_ascii=True, allow_nan=False)

    try:
        body = emit(term)
        return f"(term {version} {body})"
    except RecursionError:
        return Refused("term nesting exceeds serialization capacity")


_MAX_TEXT, _MAX_NODES, _MAX_DEPTH = 65536, 2048, 128
_KINDS = {INT: int, BOOL: bool, F64: float}
_TYPE_TAGS = {int: "int", bool: "bool", float: "float", str: "str", bytes: "bytes",
              list: "list", tuple: "tuple", dict: "dict", set: "set",
              frozenset: "frozenset", type(None): "NoneType", object: "object"}
_TYPE_BY_NAME = {name: kind for kind, name in _TYPE_TAGS.items()}
_OPERATORS = {"eq": operator.eq, "ne": operator.ne, "lt": operator.lt,
              "le": operator.le, "gt": operator.gt, "ge": operator.ge,
              "add": operator.add, "sub": operator.sub, "mul": operator.mul,
              "floordiv": operator.floordiv, "mod": operator.mod}


def _name(name):
    return (type(name) is str and name.isidentifier() and not keyword.iskeyword(name)
            and unicodedata.normalize("NFKC", name) == name)


def _variables(term):
    """Validate bounded input and reject aliases with incompatible representations."""
    if isinstance(term, Refused):
        return term
    pending, names, count = [(term, 0)], {}, 0
    try:
        while pending:
            value, depth = pending.pop()
            count += 1
            if count > _MAX_NODES or depth > _MAX_DEPTH:
                return Refused("term exceeds node or depth limit")
            if type(value) is not Term:
                return Refused("translation requires a Term")
            Term(value.op, value.sort, value.args)
            if value.op == "var":
                name = value.args[0]
                if not _name(name) or value.sort in (TAG, SEQ, MAP):
                    return Refused("unsupported or non-normalized variable")
                if name in names and names[name] != value.sort:
                    return Refused("one name has incompatible term sorts")
                names[name] = value.sort
            pending.extend((arg, depth + 1) for arg in value.args if isinstance(arg, Term))
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        return Refused(str(exc))
    return names


def from_canon(text: str) -> Term | Refused:
    """Decode only our bounded, versioned grammar; require exact canonical round trip."""
    if type(text) is not str or not text or len(text) > _MAX_TEXT:
        return Refused("canonical text must be nonempty and at most 65536 characters")
    tokens, end = [], 0
    pattern = r'\s+|[()]|"(?:\\.|[^"\\])*"|[^\s()"]+'
    for match in re.finditer(pattern, text):
        if match.start() != end:
            return Refused("invalid canonical token")
        end = match.end()
        if not match.group().isspace():
            tokens.append(match.group())
    if end != len(text):
        return Refused("invalid canonical token")
    position, count = 0, 0

    def take():
        nonlocal position
        if position >= len(tokens):
            raise ValueError("truncated canonical text")
        token = tokens[position]
        position += 1
        return token

    def body(depth):
        nonlocal count
        count += 1
        if count > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("canonical term exceeds node or depth limit")
        if take() != "(":
            raise ValueError("expected term body")
        op, sort, args = take(), Sort(take()), []
        while position < len(tokens) and tokens[position] != ")":
            if tokens[position] == "(":
                args.append(body(depth + 1))
            else:
                token = take()
                if token.startswith('"') or token in ("true", "false"):
                    args.append(json.loads(token))
                elif re.fullmatch(r"-?[0-9]+", token):
                    digits = token.lstrip("-")
                    value = 0
                    for start in range(0, len(digits), 9):
                        chunk = digits[start:start + 9]
                        value = value * 10**len(chunk) + int(chunk)
                    args.append(-value if token.startswith("-") else value)
                else:
                    raise ValueError("unsupported canonical atom")
        if take() != ")":
            raise ValueError("unterminated term body")
        return Term(op, sort, tuple(args))

    version = "unknown"
    try:
        opening, envelope, version = take(), take(), take()
        if (opening, envelope) != ("(", "term") or version not in ("1", "2"):
            return Refused(f"unsupported canonical envelope or version {version}")
        term = body(0)
        if take() != ")" or position != len(tokens):
            return Refused(f"canonical version {version}: trailing content")
        checked = _variables(term)
        if isinstance(checked, Refused):
            return Refused(f"canonical version {version}: {checked.reason}")
        if canon(term) != text:
            return Refused(f"canonical version {version}: text is not canonical for that version")
        return term
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        return Refused(f"canonical version {version}: {exc}")


def _view_info(view):
    """Validated views always refer to the same Py source binding."""
    if view.op == "as_seq":
        return (view.args[0].args[0], view.args[1].args[0],
                view.args[2].args[0] if len(view.args) == 3 else None)
    if view.op == "as_map":
        return (view.args[0].args[0], "dict",
                view.args[1].args[0] if len(view.args) == 2 else None)
    raise ValueError("container operand requires a source-bound view")


@dataclass(frozen=True)
class _Snapshot:
    items: tuple
    domain: str | None


def eval_term(term: Term | Refused, env: dict):
    """Interpret pure terms without target calls; undefined evaluations refuse.

    ``Py`` may hold any object, inspected only by exact ``type``. User coercions,
    equality, truth and arithmetic methods are never called. Unvisited ordered
    Boolean arms are not evaluated. Int floor/mod use Python's sign rule.
    Exact list/tuple items or dict keys are snapped per evaluation. Membership
    checks actual primitive contents independently; views imply no post-state,
    annotations or mutation model, and dict values are never inspected.
    """
    checked = _variables(term)
    if isinstance(checked, Refused):
        return checked
    if type(env) is not dict:
        return Refused("evaluation requires a plain binding dictionary")

    def evaluate(value):
        op, args = value.op, value.args
        if op == "var":
            name = args[0]
            if name not in env:
                return Refused(f"unbound name: {name}")
            result = env[name]
            if value.sort != PY and type(result) is not _KINDS[value.sort]:
                return Refused(f"wrong actual kind for {name}")
            return result
        if op == "lit":
            return float.fromhex(args[0]) if value.sort == F64 else args[0]
        if op == "key":
            return args[1] if args[0] == "str" else bytes.fromhex(args[1])
        if op == "tag":
            return next(kind for kind, name in _TYPE_TAGS.items() if name == args[0])
        if op in ("as_seq", "as_map"):
            name, kind, domain = _view_info(value)
            actual = evaluate(args[0])
            if isinstance(actual, Refused):
                return actual
            if type(actual) is not _TYPE_BY_NAME[kind]:
                return Refused("view has wrong actual container kind")
            # Exact builtin iteration takes a snapshot without equality/hash callbacks.
            return _Snapshot(tuple(actual), domain)
        if op == "len":
            actual = evaluate(args[0])
            if isinstance(actual, Refused):
                return actual
            if args[0].sort in (SEQ, MAP) and type(actual) is _Snapshot:
                return len(actual.items)
            if type(actual) is str or type(actual) is bytes:
                return len(actual)
            return Refused("length requires an exact container view, str or bytes")
        if op == "contains":
            snapshot, key = evaluate(args[0]), evaluate(args[1])
            if isinstance(snapshot, Refused):
                return snapshot
            if isinstance(key, Refused):
                return key
            if (type(snapshot) is not _Snapshot or snapshot.domain not in _PRIMITIVES
                    or not any(type(key) is kind for kind in (int, bool, str, bytes))
                    or any(type(item) is not _TYPE_BY_NAME[snapshot.domain] for item in snapshot.items)):
                return Refused("membership requires checked exact primitive contents and key")
            return key in snapshot.items
        if op in ("and", "or"):
            for arg in args:
                result = evaluate(arg)
                if isinstance(result, Refused):
                    return result
                if type(result) is not bool:
                    return Refused("Boolean operand has wrong actual kind")
                if result == (op == "or"):
                    return result
            return op == "and"
        values = []
        for arg in args:
            result = evaluate(arg)
            if isinstance(result, Refused):
                return result
            values.append(result)
        if op == "typeis":
            return type(values[0]) is values[1]
        if op.startswith("as_"):
            return values[0] if type(values[0]) is _KINDS[value.sort] else Refused("cast has wrong actual kind")
        if op == "b2i":
            return 1 if values[0] else 0
        if op == "i2f":
            return float(values[0])
        if op == "not":
            return not values[0]
        if op == "neg":
            return -values[0]
        if op in ("floordiv", "mod") and values[1] == 0:
            return Refused("zero divisor")
        if args[0].sort == TAG:
            return (values[0] is values[1]) == (op == "eq")
        return _OPERATORS[op](*values)

    try:
        return evaluate(term)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError, RuntimeError, RecursionError) as exc:
        return Refused(str(exc))


def rename(term: Term | Refused, mapping: dict[str, str]) -> Term | Refused:
    """Simultaneous, capture-free name replacement; literals, tags and order survive."""
    names = _variables(term)
    if isinstance(names, Refused):
        return names
    if type(mapping) is not dict or any(not _name(a) or not _name(b) for a, b in mapping.items()):
        return Refused("rename requires normalized identifier pairs")
    renamed = [mapping.get(name, name) for name in names]
    if len(set(renamed)) != len(renamed):
        return Refused("rename would merge distinct source bindings")

    def replace(value):
        if value.op == "var":
            return var(mapping.get(value.args[0], value.args[0]), value.sort)
        return Term(value.op, value.sort, tuple(replace(a) if isinstance(a, Term) else a for a in value.args))

    try:
        return replace(term)
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        return Refused(str(exc))


def _binding_types(types):
    if type(types) is not dict or any(not _name(name) or type(kind) is not str
                                    or (kind not in _TAGS and _container_type(kind) is None)
                                    for name, kind in types.items()):
        return Refused("bindings require normalized names and checked exact builtin kinds")
    return dict(types)


def _container_type(kind):
    if type(kind) is not str:
        return None
    match = re.fullmatch(r"(list|tuple|dict)\[(int|bool|str|bytes)\]", kind)
    return match.groups() if match else None


def _checked_domain(view, types):
    name, kind, domain = _view_info(view)
    if domain is None or types.get(name) != f"{kind}[{domain}]":
        raise ValueError("membership lacks verified primitive heap context")
    return name, kind, domain


def _cast(value, kind):
    container = _container_type(kind)
    base, domain = container if container else (kind, None)
    if base in {"list", "tuple"}:
        return as_seq(value, base, primitive_domain=domain)
    if base == "dict":
        return as_map(value, primitive_domain=domain)
    return {"int": as_int, "bool": as_bool, "float": as_f64}.get(kind, lambda v: v)(value)


def _numeric_pair(left, right, *, ordering=False):
    if isinstance(left, Refused):
        return left, left
    if isinstance(right, Refused):
        return right, right
    if {left.sort, right.sort} <= {INT, BOOL} and (ordering or left.sort != right.sort):
        return (b2i(left) if left.sort == BOOL else left,
                b2i(right) if right.sort == BOOL else right)
    return left, right


def from_ast(node_or_text, types: dict[str, str], *, builtin_names=()) -> Term | Refused:
    """Convert a bounded pure subset under source-checked must-bindings.

    Every source name remains ``var(name, PY)``. Supplied types permit casts but
    are not embedded proof facts: a consumer must supply guards before using
    such casts. Annotation hints and unresolved builtin identities are refused.
    Partial Int floor/mod values retain their runtime definedness obligation.
    Container bracket types must be verified heap domains, never annotations.
    Only exact list/tuple/dict views, len and primitive membership are admitted.
    """
    env = _binding_types(types)
    if isinstance(env, Refused):
        return env
    try:
        if type(node_or_text) is str:
            if not node_or_text or len(node_or_text) > _MAX_TEXT:
                return Refused("source expression exceeds text limit")
            tree = ast.parse(node_or_text, mode="eval").body
        elif type(node_or_text) is ast.Expression:
            tree = node_or_text.body
        elif isinstance(node_or_text, ast.expr):
            tree = node_or_text
        else:
            return Refused("expected expression text or AST")
        allowed = {ast.Name, ast.Load, ast.Constant, ast.BoolOp, ast.And, ast.Or,
                   ast.UnaryOp, ast.Not, ast.UAdd, ast.USub, ast.BinOp, ast.Add,
                   ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod, ast.Compare, ast.Eq,
                   ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
                   ast.Call, ast.In, ast.NotIn}
        pending, count = [(tree, 0)], 0
        while pending:
            node, depth = pending.pop()
            count += 1
            if count > _MAX_NODES or depth > _MAX_DEPTH or type(node) not in allowed:
                return Refused("unsupported or oversized source syntax")
            if isinstance(node, ast.Name) and (not _name(node.id)
                    or (node.id not in env and node.id not in builtin_names)):
                return Refused("unbound or non-normalized source name")
            if isinstance(node, ast.Call):
                if (type(node.func) is not ast.Name or node.func.id not in {"type", "len"}
                        or node.func.id not in builtin_names or node.func.id in env
                        or len(node.args) != 1 or node.keywords):
                    return Refused("only authenticated type/len calls are pure")
                operand = node.args[0]
                bound = type(operand) is ast.Name and operand.id in env
                string = (type(operand) is ast.Constant
                          and (type(operand.value) is str or type(operand.value) is bytes))
                if not bound and not (node.func.id == "len" and string):
                    return Refused("type/len requires a bound name or exact string length operand")
            pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))

        def guard(node):
            if type(node) is ast.Compare and len(node.ops) == 1:
                left, right, op = node.left, node.comparators[0], node.ops[0]
                if (type(left) is ast.Call and left.func.id == "type" and type(right) is ast.Name
                        and right.id in _TAGS and right.id in builtin_names and right.id not in env
                        and isinstance(op, (ast.Is, ast.IsNot))):
                    return left.args[0].id, right.id, type(op) is ast.Is
                if (type(left) is ast.Name and left.id in env and type(right) is ast.Constant
                        and right.value is None and isinstance(op, (ast.Is, ast.IsNot))):
                    return left.id, "NoneType", type(op) is ast.Is
            return None

        def convert(node, local):
            g = guard(node)
            if g:
                result = typeis(var(g[0]), g[1])
                return result if g[2] else not_(result)
            if type(node) is ast.Constant:
                return (key_literal(node.value) if type(node.value) is str or type(node.value) is bytes
                        else literal(node.value))
            if type(node) is ast.Name:
                if node.id not in local:
                    return Refused(f"name not bound as a value: {node.id}")
                return _cast(var(node.id), local[node.id])
            if type(node) is ast.BoolOp:
                context, parts = dict(local), []
                for part in node.values:
                    parts.append(convert(part, context))
                    g = guard(part)
                    if type(node.op) is ast.And and g and g[2]:
                        if context[g[0]].split("[")[0] != g[1]:
                            context[g[0]] = g[1]
                return and_(*parts) if type(node.op) is ast.And else or_(*parts)
            if type(node) is ast.UnaryOp:
                value = convert(node.operand, local)
                if type(node.op) is ast.Not:
                    return not_(value)
                if isinstance(value, Term) and value.sort == BOOL:
                    value = b2i(value)
                if type(node.op) is ast.USub:
                    return neg(value)
                return value if isinstance(value, Term) and value.sort in (INT, F64) else Refused("invalid unary plus")
            if type(node) is ast.BinOp:
                left, right = _numeric_pair(convert(node.left, local), convert(node.right, local), ordering=True)
                op = {ast.Add: add, ast.Sub: sub, ast.Mult: mul, ast.FloorDiv: floordiv, ast.Mod: mod}[type(node.op)]
                return op(left, right)
            if type(node) is ast.Call and node.func.id == "len":
                return length(convert(node.args[0], local))
            if type(node) is ast.Compare:
                operands = [convert(node.left, local), *(convert(n, local) for n in node.comparators)]
                parts = []
                for left, op, right in zip(operands, node.ops, operands[1:]):
                    if type(op) in {ast.In, ast.NotIn}:
                        member = contains(right, left)
                        parts.append(not_(member) if type(op) is ast.NotIn else member)
                        continue
                    if type(op) not in {ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE}:
                        return Refused("identity supported only for exact type or None tests")
                    left, right = _numeric_pair(left, right, ordering=type(op) not in {ast.Eq, ast.NotEq})
                    fn = {ast.Eq: eq, ast.NotEq: ne, ast.Lt: lt, ast.LtE: le, ast.Gt: gt, ast.GtE: ge}[type(op)]
                    parts.append(fn(left, right))
                return parts[0] if len(parts) == 1 else and_(*parts)
            return Refused("unsupported value expression")

        result = convert(tree, env)
        checked = _variables(result)
        return checked if isinstance(checked, Refused) else result
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, SyntaxError, OverflowError, RecursionError) as exc:
        return Refused(str(exc))


def to_python(term: Term | Refused, *, types=None, builtin_names=()) -> str | Refused:
    """Render a total Boolean predicate in the unchanged witness grammar.

    Builtin names must be verified in the source scope by the caller. Every cast
    is dominated by its exact type guard in the ordered conjunction; supplied
    binding hints do not discharge this obligation. Unsupported value terms
    stay store-only, including floor/mod, i2f and nonfinite float literals.
    Views require dominating exact kind guards. Membership also requires checked
    heap domains in types (e.g. dict[int]), never annotation-as-proof. Verified
    builtin_names must include len and the list/tuple/dict/str/bytes guard names.
    Literal type tests fold only in this projection; their true AND and false OR
    operands drop. Canonical terms and whole-term refusal checks stay unchanged.
    """
    names = _variables(term)
    if isinstance(names, Refused):
        return names
    if term.sort != BOOL:
        return Refused("witness predicate must be Boolean")
    env = {name: _KINDS[sort].__name__ if sort != PY else "object" for name, sort in names.items()}
    if types is not None:
        provided = _binding_types(types)
        if isinstance(provided, Refused):
            return provided
        for name, sort in names.items():
            if sort != PY and name in provided and provided[name] != env[name]:
                return Refused("provided binding contradicts term sort")
        env.update(provided)

    def render(value, guards):
        op, args = value.op, value.args
        if op == "var":
            return args[0]
        if op == "lit":
            if value.sort == F64:
                number = float.fromhex(args[0])
                if not math.isfinite(number):
                    raise ValueError("nonfinite literal is outside witness grammar")
                return repr(number)
            return _integer_text(args[0]) if value.sort == INT else repr(args[0])
        if op == "key":
            return repr(args[1] if args[0] == "str" else bytes.fromhex(args[1]))
        if op in ("as_seq", "as_map"):
            name, kind, _ = _view_info(value)
            if (name, kind) not in guards:
                raise ValueError("view lacks dominating exact container guard")
            return render(args[0], guards)
        if op == "len":
            if "len" not in builtin_names or "len" in env:
                raise ValueError("unverified or shadowed builtin len")
            if args[0].sort == PY and args[0].op != "key":
                if args[0].op != "var" or not any((args[0].args[0], k) in guards for k in ("str", "bytes")):
                    raise ValueError("string length lacks dominating exact str/bytes guard")
            return f"len({render(args[0], guards)})"
        if op == "contains":
            _checked_domain(args[0], env)
            key = args[1]
            if key.sort == PY and key.op != "key":
                if key.op != "var" or not any((key.args[0], k) in guards for k in ("str", "bytes")):
                    raise ValueError("membership key lacks exact primitive guard")
            return f"({render(key, guards)} in {render(args[0], guards)})"
        if op == "typeis":
            name = args[1].args[0]
            if args[0].op == "lit":
                return repr(_KINDS[args[0].sort].__name__ == name)
            if name == "NoneType":
                raise ValueError("none_test_outside_witness_grammar")
            if name not in builtin_names or "type" not in builtin_names or name in env or "type" in env:
                raise ValueError("unverified or shadowed builtin type test")
            if any(n.op in {"b2i", "i2f"} for n in _term_nodes(args[0])):
                raise ValueError("converted value type has no faithful witness spelling")
            return f"(type({render(args[0], guards)}) is {name})"
        if op.startswith("as_"):
            name, kind = args[0].args[0], _KINDS[value.sort].__name__
            if (name, kind) not in guards:
                raise ValueError("cast lacks dominating exact type guard")
            return render(args[0], guards)
        if op == "b2i":
            return render(args[0], guards)
        if op in {"tag", "i2f", "floordiv", "mod"}:
            raise ValueError(f"{op} has no supported witness spelling")
        if op in ("and", "or"):
            def operands(group):
                # Boolean operands are exact bools; same-op grouping preserves order/definedness.
                for child in group.args:
                    if child.op == op:
                        yield from operands(child)
                    else:
                        yield child

            local, parts, folded, dropped = set(guards), [], [], False
            for arg in operands(value):
                text = render(arg, local)
                # Only literal type tests (or groups reduced by them) produce bare bool text here.
                is_folded = arg.op != "lit" and text in ("True", "False")
                folded.append(is_folded)
                if is_folded and text == ("True" if op == "and" else "False"):
                    dropped = True
                else:
                    parts.append(text)
                if op == "and" and arg.op == "typeis" and arg.args[0].op == "var":
                    local.add((arg.args[0].args[0], arg.args[1].args[0]))
            if all(folded):
                return ("False" if op == "and" else "True") if parts else ("True" if op == "and" else "False")
            if dropped and len(parts) == 1:
                return parts[0]
            return "(" + f" {op} ".join(parts) + ")"
        if op in ("not", "neg"):
            return f"({'not ' if op == 'not' else '-'}{render(args[0], guards)})"
        symbol = {"eq": "==", "ne": "!=", "lt": "<", "le": "<=", "gt": ">", "ge": ">=",
                  "add": "+", "sub": "-", "mul": "*"}[op]
        return f"({render(args[0], guards)} {symbol} {render(args[1], guards)})"

    try:
        text = render(term, set())
        check_predicate(text, env, builtin_names=builtin_names)
        return text
    except (ValueError, TypeError, KeyError, OverflowError, RecursionError, InvalidPredicate) as exc:
        return Refused(str(exc))


def _term_nodes(term):
    yield term
    for arg in term.args:
        if isinstance(arg, Term):
            yield from _term_nodes(arg)
