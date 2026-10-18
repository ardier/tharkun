"""Sound per-variable value facts from a guard ([internal ref], eregion facts port, slice 2).

SOLVING, by ``docs/LAYERS.md``'s rule.  The input is a guard: a Python
expression AST, as it appears in an ``if`` / ``while`` / ``assert`` test.  The
output is, for a variable named in it, a :class:`tharkun.solve.values.Value` for
each arm.  The arm facts contain EVERY value the variable can hold when the guard
evaluates true (resp. false).  The source is only read; nothing is run.

This is eregion's static value-fact front end, moved onto the Python AST:

* ``eregion_ir.program_extract.domain_capture.reduce_domain``, the sound
  over-approximating accumulator over IR expr dicts, with its 17-kind
  ``domain_kind`` taxonomy (kept here as :attr:`Domain.kind`, a label);
* ``eregion_commons.fact_repr.apply_constraint`` / ``merge_env``, the meet /
  join, which here are ``Value.meet`` / ``Value.join``;
* ``eregion_forward_passes...emit_bw_assume_facts._extract_constraint``, the AST
  -> ``(var, op, value)`` atoms, ported as :func:`atoms` for
  ``tharkun.solve.fact_env.refine_all``.

SOUNDNESS BAR (eregion's hard rule, kept): the arm fact must CONTAIN every value
that reaches the arm.  Over-approximation is allowed; under-approximation is not.
A part of the guard this module does not model leaves the fact unchanged, and it
is kept in :attr:`Domain.residual` so a caller can still check it.

How a guard is read
-------------------
:func:`refine` walks the test and returns ``(on_true, on_false)``:

* ``A and B``: the true arm is ``T(A) & T(B)``, the false arm is ``F(A) | F(B)``;
  ``A or B`` is the dual; ``not A`` swaps the arms.  Chained compares are the
  ``and`` of their pairs.  This is more general than eregion's reducer, which
  only folds an ``or`` of ``==`` / ``in`` into a set and otherwise gives up.
* The atoms, where ``x`` is the variable and ``c`` a literal (on either side):

  - ``x <op> c`` for ``< <= > >=`` with a number: exact integer and float bounds
    (Python compares int and float exactly), NaN on the false arm of a float
    compare, and ``bool`` treated as the int it is.  With a str / bytes constant it
    is a lexical bound (:attr:`Domain.lex`), which also filters a finite prior;
  - ``x == c`` / ``x != c`` for a scalar ``c``: :meth:`Value.equal_to`, so
    ``x == 0`` admits ``0``, ``0.0``, ``-0.0`` and ``False``;
  - ``x in C`` / ``x not in C`` for a literal tuple / list / set of scalars, a str
    or bytes literal (substrings), or ``range(consts)``;
  - ``c in x`` / ``c not in x`` (``x`` is the container): ``Value.contains`` /
    ``Value.lacks``;
  - ``x is None``, ``x is True`` / ``False`` and their ``is not`` forms;
  - bare truthiness ``x`` / ``not x``;
  - ``len(x) <op> K``: ``Value.length``.

The world is OPEN
-----------------
A variable may hold a value of a type the guard never names: a ``Fraction``, a
numpy scalar, or an object whose ``__eq__`` says yes to anything.  So an atom
constrains the builtin scalar types (``int float bool str bytes NoneType``)
and leaves every other type alone: ``x == 5`` yields ``Kinds.top()`` with
``i = {5}``, ``f = {5.0}``, ``bools = {}``, ``s`` and ``by`` empty, and ``None``
excluded through ``neq``.  A caller that knows the type meets the result with
its own ``Value``.  Once the kinds are closed, :func:`refine` also derives the
length of a finite str / bytes set.  A comparison that raises ``TypeError``
for a type (``"a" < 5``) reaches NEITHER arm, so that type is empty on both.

Assumptions (the ones eregion makes too, written down):

* Comparison and ``__bool__`` / ``__len__`` / ``__contains__`` dunders don't
  mutate state.  The length, contains and lacks facts are dropped anyway when the
  test holds a call other than ``len`` / ``range`` / ``isinstance``, because a
  call may mutate a container between two atoms.
* ``var`` is not rebound inside the test.  A walrus to ``var`` anywhere in the
  test drops every fact.  A call that rebinds a global ``var`` is excluded by
  ``local=True`` (the default); pass ``local=False`` for a global, and any such
  call then drops every fact.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Tuple

from tharkun.solve.values import FloatFact, IntFact, Kinds, StrFact, Value

__all__ = [
    "DOMAIN_KINDS", "MAX_CONST_SET", "Domain", "GuardDomain", "Lex", "atoms", "domain_of", "domains",
    "refine", "refine_env",
]

#: eregion's closed ``domain_kind`` set (``domain_capture.DOMAIN_KINDS``).
DOMAIN_KINDS = (
    "int_range_closed", "int_range_low_open", "int_range_high_open",
    "int_set", "int_neq",
    "float_range", "float_set",
    "str_set", "str_neq", "bytes_set", "bytes_neq", "lex_range",
    "bool_set", "is_none", "container_length", "set_membership",
    "guard_only", "top",
)

#: eregion's cap for an ``or`` of equalities folded into ``in`` (``MAX_CONST_SET``).
MAX_CONST_SET = 8
#: the most members / substrings / range elements kept as an exact set.
_MAX_MEMBERS = 256

_INF = math.inf
_SCALAR_KINDS = frozenset({"int", "float", "bool", "str", "bytes", "NoneType"})
_NO_NONE = frozenset({None})
_PURE_CALLS = frozenset({"len", "range", "isinstance"})

_NEG = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "==",
        "in": "not in", "not in": "in", "is": "is not", "is not": "is"}
_FLIP = {"<": ">", "<=": ">=", ">": "<", ">=": "<=", "==": "==", "!=": "!="}
_OPS = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==",
        ast.NotEq: "!=", ast.In: "in", ast.NotIn: "not in", ast.Is: "is",
        ast.IsNot: "is not"}


def _cmp(op: str, a: Any, b: Any) -> bool:
    """``a <op> b``, concretely (raises what Python raises)."""
    if op == "<":
        return a < b
    if op == "<=":
        return a <= b
    if op == ">":
        return a > b
    if op == ">=":
        return a >= b
    if op == "==":
        return a == b
    if op == "!=":
        return a != b
    if op == "in":
        return a in b
    if op == "not in":
        return a not in b
    raise ValueError(op)


# --------------------------------------------------------------------------- #
# literals
# --------------------------------------------------------------------------- #

_MISSING = object()


def _const(node: ast.AST) -> Any:
    """The scalar literal ``node`` denotes (``-5`` included), else ``_MISSING``."""
    if isinstance(node, ast.Constant):
        v = node.value
        if v is None or isinstance(v, (bool, int, float, str, bytes)):
            return v
        return _MISSING
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _const(node.operand)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v if isinstance(node.op, ast.USub) else v
    return _MISSING


def _const_items(node: ast.AST) -> Optional[list]:
    """The scalars of a literal tuple / list / set, else ``None``."""
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return None
    out = []
    for elt in node.elts:
        v = _const(elt)
        if v is _MISSING:
            return None
        out.append(v)
    return out


def _const_range(node: ast.AST) -> Optional[range]:
    """``range(a[, b[, s]])`` with int literals, else ``None``."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "range" and not node.keywords and 1 <= len(node.args) <= 3):
        return None
    args = [_const(a) for a in node.args]
    if any(type(a) is not int for a in args):
        return None
    try:
        return range(*args)
    except ValueError:              # step 0
        return None


def _is_var(node: ast.AST, var: str) -> bool:
    return isinstance(node, ast.Name) and node.id == var


def _is_len_of(node: ast.AST, var: str) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "len" and len(node.args) == 1 and not node.keywords
            and _is_var(node.args[0], var))


def _mentions(node: ast.AST, var: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == var for n in ast.walk(node))


def _is_nan(v: Any) -> bool:
    return isinstance(v, float) and math.isnan(v)


def _tag(v: Any) -> str:
    if v is None:
        return "none"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, bytes):
        return "bytes"
    if isinstance(v, str):
        return "str"
    return "other"


# --------------------------------------------------------------------------- #
# exact bounds for a comparison with a number
# --------------------------------------------------------------------------- #

def _int_true(op: str, c) -> IntFact:
    """The ints ``n`` with ``n <op> c`` for an ordering ``op`` and a number ``c``."""
    if _is_nan(c):
        return IntFact.bottom()
    if isinstance(c, float):
        if math.isinf(c):
            up = (c > 0) == (op in ("<", "<="))     # every int is below +inf
            return IntFact.top() if up else IntFact.bottom()
        lo_x, hi_x = math.ceil(c), math.floor(c)    # the ints just above / below c
        return {"<": IntFact(None, lo_x - 1), "<=": IntFact(None, hi_x),
                ">": IntFact(hi_x + 1, None), ">=": IntFact(lo_x, None)}[op]
    c = int(c)
    return {"<": IntFact(None, c - 1), "<=": IntFact(None, c),
            ">": IntFact(c + 1, None), ">=": IntFact(c, None)}[op]


def _float_hi(c, strict: bool) -> Optional[float]:
    """The largest float ``y`` with ``y < c`` (``strict``) or ``y <= c``; ``None`` if none.
    Python compares a float with an int exactly, so this is exact for either."""
    ok = (lambda y: y < c) if strict else (lambda y: y <= c)
    if isinstance(c, float):
        y = c
    else:
        try:
            y = float(c)
        except OverflowError:
            y = _INF if c > 0 else -_INF
    while not ok(y):
        if y == -_INF:
            return None
        y = math.nextafter(y, -_INF)
    while True:
        n = math.nextafter(y, _INF)
        if n == y or not ok(n):
            return y
        y = n


def _float_lo(c, strict: bool) -> Optional[float]:
    """The smallest float ``y`` with ``y > c`` (``strict``) or ``y >= c``; ``None`` if none."""
    ok = (lambda y: y > c) if strict else (lambda y: y >= c)
    if isinstance(c, float):
        y = c
    else:
        try:
            y = float(c)
        except OverflowError:
            y = _INF if c > 0 else -_INF
    while not ok(y):
        if y == _INF:
            return None
        y = math.nextafter(y, _INF)
    while True:
        n = math.nextafter(y, -_INF)
        if n == y or not ok(n):
            return y
        y = n


def _float_true(op: str, c) -> FloatFact:
    """The floats ``y`` (NaN excluded) with ``y <op> c`` for an ordering ``op``."""
    if _is_nan(c):
        return FloatFact.bottom()
    if op in ("<", "<="):
        hi = _float_hi(c, op == "<")
        return FloatFact.bottom() if hi is None else FloatFact(-_INF, hi, False)
    lo = _float_lo(c, op == ">")
    return FloatFact.bottom() if lo is None else FloatFact(lo, _INF, False)


def _bools_true(op: str, c) -> frozenset:
    out = set()
    for b in (False, True):
        try:
            if _cmp(op, b, c):
                out.add(b)
        except TypeError:
            pass
    return frozenset(out)


# --------------------------------------------------------------------------- #
# the arm fact: a Value plus the exact finite set and the lexical bounds
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Lex:
    """Lexical bounds on the str and bytes values of a variable (eregion's
    ``lex_range``).  ``None`` is unbounded; ``*_open`` marks a strict bound.  The
    bounds of a str and of a bytes variable are kept apart (they never compare)."""

    lo: Any = None
    lo_open: bool = False
    hi: Any = None
    hi_open: bool = False

    def admits(self, v: Any) -> bool:
        for b, is_open, low in ((self.lo, self.lo_open, True), (self.hi, self.hi_open, False)):
            if b is None or type(b) is not type(v):
                continue
            if low and (v < b or (is_open and v == b)):
                return False
            if not low and (v > b or (is_open and v == b)):
                return False
        return True


_LEX_TOP: Tuple[Lex, Lex] = (Lex(), Lex())         # (str bounds, bytes bounds)


def _lex_meet(a: Lex, b: Lex) -> Lex:
    lo, lo_open = a.lo, a.lo_open
    if b.lo is not None and (lo is None or b.lo > lo or (b.lo == lo and b.lo_open)):
        lo, lo_open = b.lo, b.lo_open
    hi, hi_open = a.hi, a.hi_open
    if b.hi is not None and (hi is None or b.hi < hi or (b.hi == hi and b.hi_open)):
        hi, hi_open = b.hi, b.hi_open
    return Lex(lo, lo_open, hi, hi_open)


def _lex_join(a: Lex, b: Lex) -> Lex:
    if a.lo is None or b.lo is None:
        lo, lo_open = None, False
    elif a.lo < b.lo or (a.lo == b.lo and not a.lo_open):
        lo, lo_open = a.lo, a.lo_open
    else:
        lo, lo_open = b.lo, b.lo_open
    if a.hi is None or b.hi is None:
        hi, hi_open = None, False
    elif a.hi > b.hi or (a.hi == b.hi and not a.hi_open):
        hi, hi_open = a.hi, a.hi_open
    else:
        hi, hi_open = b.hi, b.hi_open
    return Lex(lo, lo_open, hi, hi_open)


@dataclass(frozen=True)
class _Arm:
    """One arm's fact: ``value`` over-approximates the set; ``members`` (when not
    ``None``) is a finite set such that every builtin-scalar value in the arm is
    ``==`` to one of them; ``lex`` bounds its str / bytes values."""

    value: Value = Value()
    members: Optional[frozenset] = None
    lex: Tuple[Lex, Lex] = _LEX_TOP

    def meet(self, o: "_Arm") -> "_Arm":
        if self.members is None:
            m = o.members
        elif o.members is None:
            m = self.members
        else:
            m = frozenset(a for a in self.members if any(_same(a, b) for b in o.members))
        return _Arm(self.value.meet(o.value), m,
                    (_lex_meet(self.lex[0], o.lex[0]), _lex_meet(self.lex[1], o.lex[1])))

    def join(self, o: "_Arm") -> "_Arm":
        m = None if self.members is None or o.members is None else self.members | o.members
        if m is not None and len(m) > _MAX_MEMBERS:
            m = None
        return _Arm(self.value.join(o.value), m,
                    (_lex_join(self.lex[0], o.lex[0]), _lex_join(self.lex[1], o.lex[1])))


def _same(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:
        return True


_TOP_ARM = _Arm()


def _open(*, i: IntFact = IntFact.bottom(), f: FloatFact = FloatFact.bottom(),
          s: StrFact = StrFact.bottom(), by: StrFact = StrFact.bottom(),
          bools: Optional[frozenset] = frozenset(), none: bool = False,
          length: IntFact = IntFact(0, None), contains: frozenset = frozenset(),
          lacks: frozenset = frozenset()) -> Value:
    """Any value of a type outside the builtin scalars, plus the given parts of the
    builtin scalar types (``None`` only when ``none``).  The OPEN-world constructor
    every atom uses."""
    return Value(kinds=Kinds.top(), i=i, f=f, s=s, by=by, bools=bools,
                 neq=frozenset() if none else _NO_NONE, length=length,
                 contains=contains, lacks=lacks)


def _equal_to(c: Any) -> Value:
    """Open world: every value ``== c`` among the builtin scalars, plus every other type."""
    if _is_nan(c):
        return _open()
    e = Value.equal_to(c)
    return _open(i=e.i, f=e.f, s=e.s, by=e.by,
                 bools=e.bools if e.bools is not None else None, none=c is None)


def _excluding(vals: Iterable[Any]) -> Value:
    """Every value except those ``==`` to one of ``vals`` (NaN never excludes anything)."""
    try:
        neq = frozenset(v for v in vals if not _is_nan(v))
    except TypeError:
        return Value.top()
    return Value(neq=neq)


def _substrings(c: Any) -> Optional[frozenset]:
    n = len(c)
    if n * (n + 1) // 2 + 1 > _MAX_MEMBERS:
        return None
    return frozenset(c[i:j] for i in range(n + 1) for j in range(i, n + 1))


# --------------------------------------------------------------------------- #
# atoms
# --------------------------------------------------------------------------- #

class _Ctx:
    """What the walk records besides the arms (for the label and the residual)."""

    def __init__(self) -> None:
        self.residual: List[ast.expr] = []
        self.tags: set = set()
        self.len_seen = False
        self.mentioned = False


def _order_atom(op: str, c: Any, ctx: _Ctx) -> Tuple[_Arm, _Arm]:
    """``x <op> c`` for an ordering ``op``."""
    ctx.tags.add(_tag(c))
    if isinstance(c, (bool, int, float)):
        neg = _NEG[op]
        t = _open(i=_int_true(op, c), f=_float_true(op, c), bools=_bools_true(op, c))
        if _is_nan(c):
            f = _open(i=IntFact.top(), f=FloatFact.top(), bools=frozenset({False, True}))
        else:
            ff = _float_true(neg, c).join(FloatFact(_INF, -_INF, True))   # NaN: false
            f = _open(i=_int_true(neg, c), f=ff, bools=_bools_true(neg, c))
        return _Arm(t), _Arm(f)
    if isinstance(c, (str, bytes)):
        strict = op in ("<", ">")
        lo_side = op in (">", ">=")
        pos = Lex(c, strict, None, False) if lo_side else Lex(None, False, c, strict)
        negl = Lex(None, False, c, not strict) if lo_side else Lex(c, not strict, None, False)
        is_str = isinstance(c, str)
        parts = dict(s=StrFact.top()) if is_str else dict(by=StrFact.top())
        t_lex = (pos, Lex()) if is_str else (Lex(), pos)
        f_lex = (negl, Lex()) if is_str else (Lex(), negl)
        return _Arm(_open(**parts), None, t_lex), _Arm(_open(**parts), None, f_lex)
    # None (or anything else): every builtin scalar raises TypeError
    return _Arm(_open()), _Arm(_open())


def _eq_atom(op: str, c: Any, ctx: _Ctx) -> Tuple[_Arm, _Arm]:
    ctx.tags.add(_tag(c))
    t = _Arm(_equal_to(c), frozenset() if _is_nan(c) else frozenset({c}))
    f = _Arm(_excluding([c]))
    return (t, f) if op == "==" else (f, t)


def _in_atom(op: str, coll: ast.AST, ctx: _Ctx) -> Optional[Tuple[_Arm, _Arm]]:
    """``x in <literal>``; ``None`` when the collection is not a literal we read."""
    items = _const_items(coll)
    if items is not None:
        for v in items:
            ctx.tags.add(_tag(v))
        t = _Arm(Value.bottom(), frozenset())
        for v in items:
            t = t.join(_Arm(_equal_to(v), frozenset() if _is_nan(v) else frozenset({v})))
        if not items:
            t = _Arm(Value.bottom(), frozenset())
        f = _Arm(_excluding(items))
        return (t, f) if op == "in" else (f, t)
    c = _const(coll)
    if isinstance(c, (str, bytes)):
        ctx.tags.add(_tag(c))
        subs = _substrings(c)
        part = StrFact(subs) if subs is not None else StrFact.top()
        if isinstance(c, str):
            t = _Arm(_open(s=part), subs)
            f = _Arm(_open(s=StrFact.top()).meet(_excluding(subs or ())))
        else:
            ints = frozenset(c)
            ib = IntFact(min(ints), max(ints)) if ints else IntFact.bottom()
            t = _Arm(_open(by=part, i=ib, bools=frozenset(b for b in (False, True) if b in c)),
                     None if subs is None else subs | ints)
            f = _Arm(_open(by=StrFact.top(), i=IntFact(0, 255),
                           bools=frozenset(b for b in (False, True) if b not in c))
                     .meet(_excluding((subs or frozenset()) | ints)))
        return (t, f) if op == "in" else (f, t)
    r = _const_range(coll)
    if r is not None:
        ctx.tags.add("int")
        if len(r):
            lo, hi = min(r[0], r[-1]), max(r[0], r[-1])
            t = _Arm(_open(i=IntFact(lo, hi), f=FloatFact(float(lo), float(hi), False)
                           if abs(lo) <= 2 ** 53 and abs(hi) <= 2 ** 53
                           else FloatFact(-_INF, _INF, False),
                           bools=frozenset(b for b in (False, True) if b in r)),
                     frozenset(r) if len(r) <= _MAX_MEMBERS else None)
            f = _Arm(_excluding(r) if len(r) <= _MAX_MEMBERS else Value.top())
        else:
            t, f = _Arm(Value.bottom(), frozenset()), _TOP_ARM
        return (t, f) if op == "in" else (f, t)
    return None


def _contains_atom(op: str, c: Any, ctx: _Ctx) -> Tuple[_Arm, _Arm]:
    """``c in x``: ``x`` is a container (an int / float / bool / None raises)."""
    ctx.tags.add("container")
    s = StrFact.top() if isinstance(c, str) else StrFact.bottom()
    byte_ok = isinstance(c, bytes) or (isinstance(c, int) and 0 <= c <= 255)
    by = StrFact.top() if byte_ok else StrFact.bottom()
    try:
        hash(c)
        t = _Arm(_open(s=s, by=by, contains=frozenset({c})))
        f = _Arm(_open(s=s, by=by, lacks=frozenset({c})))
    except TypeError:
        t = f = _Arm(_open(s=s, by=by))
    return (t, f) if op == "in" else (f, t)


def _len_atom(op: str, k: Any, ctx: _Ctx) -> Tuple[_Arm, _Arm]:
    """``len(x) <op> k``: ``x`` is sized (a number / bool / None raises)."""
    ctx.tags.add("container")
    ctx.len_seen = True
    nat = IntFact(0, None)
    sized = dict(s=StrFact.top(), by=StrFact.top())
    if op in ("==", "!="):
        if isinstance(k, (bool, int)) or (isinstance(k, float) and k.is_integer()):
            eq = nat.meet(IntFact.const(int(k)))
        else:
            eq = IntFact.bottom()                      # no length equals 2.5 / nan
        ne = nat
        if not eq.is_bottom() and eq.lo == 0:
            ne = IntFact(1, None)
        t, f = _Arm(_open(length=eq, **sized)), _Arm(_open(length=ne, **sized))
        return (t, f) if op == "==" else (f, t)
    if not isinstance(k, (bool, int, float)):
        return _Arm(_open()), _Arm(_open())              # len(x) < "a": TypeError
    t = nat.meet(_int_true(op, k))
    f = nat if _is_nan(k) else nat.meet(_int_true(_NEG[op], k))
    return _Arm(_open(length=t, **sized)), _Arm(_open(length=f, **sized))


def _is_atom(op: str, c: Any, ctx: _Ctx) -> Tuple[_Arm, _Arm]:
    """``x is c``: identity implies the exact type and value."""
    ctx.tags.add(_tag(c))
    if c is None:
        t, f = _Arm(Value.only("NoneType"), frozenset({None})), _Arm(_excluding([None]))
    elif isinstance(c, bool):
        t = _Arm(Value.const(c), frozenset({c}))
        f = _Arm(Value(bools=frozenset({not c})))
    else:
        t = _Arm(Value.const(c) if not _is_nan(c) else Value.only("float", f=FloatFact(_INF, -_INF, True)))
        f = _TOP_ARM
    return (t, f) if op == "is" else (f, t)


_FALSY = frozenset({0, "", b"", None})


def _truth_atom(ctx: _Ctx) -> Tuple[_Arm, _Arm]:
    t = _Arm(_excluding(_FALSY))
    f = _Arm(_open(i=IntFact.const(0), f=FloatFact(0.0, 0.0, False), s=StrFact.const(""),
                   by=StrFact.const(b""), bools=frozenset({False}), none=True), _FALSY)
    return t, f


def _compare_pair(left: ast.expr, op: str, right: ast.expr, var: str,
                  ctx: _Ctx) -> Optional[Tuple[_Arm, _Arm]]:
    """One ``left <op> right``; ``None`` when not modelled."""
    if _is_var(left, var) and not _mentions(right, var):
        if op in ("in", "not in"):
            return _in_atom(op, right, ctx)
        c = _const(right)
        if c is _MISSING:
            if op in ("==", "!=") and _const_items(right) is not None:
                # x == (1, 2): no builtin scalar equals a container literal
                t = _Arm(_open(), frozenset())
                return (t, _TOP_ARM) if op == "==" else (_TOP_ARM, t)
            return None
        if op in ("is", "is not"):
            return _is_atom(op, c, ctx)
        if op in ("==", "!="):
            return _eq_atom(op, c, ctx)
        return _order_atom(op, c, ctx)
    if _is_var(right, var) and not _mentions(left, var):
        c = _const(left)
        if c is _MISSING:
            return None
        if op in ("in", "not in"):
            return _contains_atom(op, c, ctx)
        if op in ("is", "is not"):
            return _is_atom(op, c, ctx)
        if op in ("==", "!="):
            return _eq_atom(op, c, ctx)
        return _order_atom(_FLIP[op], c, ctx)
    if op in ("in", "not in", "is", "is not"):
        return None
    if _is_len_of(left, var) and not _mentions(right, var):
        k = _const(right)
        return None if k is _MISSING else _len_atom(op, k, ctx)
    if _is_len_of(right, var) and not _mentions(left, var):
        k = _const(left)
        return None if k is _MISSING else _len_atom(_FLIP[op], k, ctx)
    return None


def _walk(node: ast.expr, var: str, ctx: _Ctx) -> Tuple[_Arm, _Arm]:
    """``(true arm, false arm)`` for ``var`` under the test ``node``."""
    if not _mentions(node, var):
        return _TOP_ARM, _TOP_ARM
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        t, f = _walk(node.operand, var, ctx)
        return f, t
    if isinstance(node, ast.BoolOp):
        parts = [_walk(v, var, ctx) for v in node.values]
        t, f = parts[0]
        for pt, pf in parts[1:]:
            if isinstance(node.op, ast.And):
                t, f = t.meet(pt), f.join(pf)
            else:
                t, f = t.join(pt), f.meet(pf)
        return t, f
    if isinstance(node, ast.Name):
        return _truth_atom(ctx)
    if isinstance(node, ast.Compare) and node.ops:
        operands = [node.left, *node.comparators]
        t, f = _TOP_ARM, None
        for i, op_node in enumerate(node.ops):
            op = _OPS.get(type(op_node))
            left, right = operands[i], operands[i + 1]
            pair = _compare_pair(left, op, right, var, ctx) if op else None
            if pair is None:
                if _mentions(left, var) or _mentions(right, var):
                    ctx.residual.append(ast.Compare(left=left, ops=[op_node], comparators=[right]))
                pair = (_TOP_ARM, _TOP_ARM)
            pt, pf = pair
            t = t.meet(pt)
            f = pf if f is None else f.join(pf)
        return t, f
    ctx.residual.append(node)
    return _TOP_ARM, _TOP_ARM


def _impure(test: ast.expr, var: str, local: bool) -> Tuple[bool, bool]:
    """``(rebinds var, may mutate or rebind through a call)``."""
    rebinds, calls = False, False
    for n in ast.walk(test):
        if isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name) and n.target.id == var:
            rebinds = True
        elif isinstance(n, (ast.Await, ast.Yield, ast.YieldFrom)):
            calls = True
        elif isinstance(n, ast.Call):
            if not (isinstance(n.func, ast.Name) and n.func.id in _PURE_CALLS):
                calls = True
    return rebinds or (calls and not local), calls


def _drop_mutable(a: _Arm) -> _Arm:
    v = a.value
    return _Arm(Value(v.kinds, v.i, v.f, v.s, v.by, v.bools, v.neq, IntFact(0, None)), a.members, a.lex)


def _tighten(v: Value, lex: Tuple[Lex, Lex] = _LEX_TOP) -> Value:
    """Facts that follow once the kinds are closed: a finite str / bytes set filtered by
    the lexical bounds and the length, and the length of a finite str / bytes set."""
    s, by = v.s, v.by
    if s.consts is not None:
        s = StrFact(frozenset(x for x in s.consts if lex[0].admits(x) and v.length.contains(len(x))))
    if by.consts is not None:
        by = StrFact(frozenset(x for x in by.consts if lex[1].admits(x) and v.length.contains(len(x))))
    v = Value(v.kinds, v.i, v.f, s, by, v.bools, v.neq, v.length, v.contains, v.lacks)
    names = v.kinds.names
    if names is None or not names <= {"str", "bytes"}:
        return v
    lens = set()
    for k in names:
        part = v.s if k == "str" else v.by
        if part.consts is None:
            return v
        lens.update(len(x) for x in part.consts)
    if not lens:
        return v
    return Value(v.kinds, v.i, v.f, v.s, v.by, v.bools, v.neq,
                 v.length.meet(IntFact(min(lens), max(lens))), v.contains, v.lacks)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Domain:
    """The sound fact about ``var`` on one arm of a guard.

    * ``value`` -- a :class:`Value` containing every value ``var`` can hold on the
      arm (open world, met with the prior);
    * ``members`` -- when not ``None``, a finite set such that every builtin
      scalar value on the arm is ``==`` to one of them (the exact ``in`` / ``==``
      set eregion calls ``values``; ``Value.i`` is only its interval hull);
    * ``lex`` -- lexical bounds on the arm's str and bytes values;
    * ``kind`` -- eregion's ``domain_kind`` label (:data:`DOMAIN_KINDS`), chosen
      from the builtin-scalar view of the fact; a label, not a further claim;
    * ``residual`` -- the sub-expressions mentioning ``var`` that did not narrow
      the fact (eregion's ``guards``), for a caller to check.
    """

    var: str
    value: Value
    members: Optional[frozenset] = None
    lex: Tuple[Lex, Lex] = _LEX_TOP
    kind: str = "top"
    residual: Tuple[ast.expr, ...] = ()

    def admits(self, v: Any) -> bool:
        """Whether the concrete value ``v`` is in this domain.  The check is exact for
        builtin scalars; for a container it checks length and contains / lacks."""
        val = self.value
        if not val.contains_value(v):
            return False
        if self.members is not None and _tag(v) != "other":
            if not any(_same(v, m) for m in self.members):
                return False
        if isinstance(v, str) and not self.lex[0].admits(v):
            return False
        if isinstance(v, bytes) and not self.lex[1].admits(v):
            return False
        if _tag(v) == "other":
            try:
                n = len(v)
            except TypeError:
                n = None
            if n is not None and not val.length.contains(n):
                return False
            try:
                if any(c not in v for c in val.contains) or any(c in v for c in val.lacks):
                    return False
            except TypeError:
                pass
        return True


#: the facade name (tharkun.solve.GuardDomain); Domain inside this module.
GuardDomain = Domain


def _arms(test: ast.expr, var: str, prior: Value, local: bool) -> Tuple[_Arm, _Arm, _Ctx]:
    ctx = _Ctx()
    rebinds, calls = _impure(test, var, local)
    if rebinds:
        ctx.residual.append(test)
        return _Arm(prior), _Arm(prior), ctx
    ctx.mentioned = _mentions(test, var)
    t, f = _walk(test, var, ctx)
    if calls:
        t, f = _drop_mutable(t), _drop_mutable(f)
    p = _Arm(prior)
    t, f = t.meet(p), f.meet(p)
    return (_Arm(_tighten(t.value, t.lex), t.members, t.lex),
            _Arm(_tighten(f.value, f.lex), f.members, f.lex), ctx)


def refine(test: ast.expr, var: str, prior: Optional[Value] = None, *,
           local: bool = True) -> Tuple[Value, Value]:
    """``(on_true, on_false)``: ``prior`` narrowed by ``test`` evaluating true / false.

    Each result contains every value of ``var`` that satisfies ``prior`` and takes
    that arm.  A part of ``test`` this module does not read leaves ``prior``
    unchanged.  ``local=False`` says ``var`` is a global a call could rebind."""
    t, f, _ = _arms(test, var, Value.top() if prior is None else prior, local)
    return t.value, f.value


def refine_env(test: ast.expr, env: Mapping[str, Value], *,
               local: bool = True) -> Tuple[dict, dict]:
    """:func:`refine` for every name ``test`` reads: ``(env_true, env_false)``, each
    mapping a name to its narrowed ``Value`` (a name missing from ``env`` starts at
    top).  Names the test does not read are not in the result."""
    names = sorted({n.id for n in ast.walk(test) if isinstance(n, ast.Name)})
    et, ef = {}, {}
    for name in names:
        et[name], ef[name] = refine(test, name, env.get(name), local=local)
    return et, ef


def _parse(pred: Any) -> ast.expr:
    if isinstance(pred, str):
        return ast.parse(pred, mode="eval").body
    if isinstance(pred, ast.Expression):
        return pred.body
    return pred


def domain_of(preds: Any, var: str, *, negate: bool = False, prior: Optional[Value] = None,
              type_hint: Optional[str] = None, local: bool = True) -> Domain:
    """The :class:`Domain` of ``var`` where every predicate in ``preds`` holds (their
    conjunction; ``negate`` takes the arm where their conjunction is false).  A
    predicate is an ``ast.expr`` or source text; a single one may be passed bare.
    ``type_hint`` (``int float str bytes bool none list tuple set dict``) only
    steers the label, as eregion's ``var_type_hint`` does."""
    if isinstance(preds, (str, ast.AST)):
        preds = [preds]
    exprs = [_parse(p) for p in preds]
    if not exprs:
        test: ast.expr = ast.Constant(True)
    elif len(exprs) == 1:
        test = exprs[0]
    else:
        test = ast.BoolOp(op=ast.And(), values=exprs)
    t, f, ctx = _arms(test, var, Value.top() if prior is None else prior, local)
    arm = f if negate else t
    kind = _label(arm, ctx, type_hint)
    return Domain(var, arm.value, arm.members, arm.lex, kind, tuple(ctx.residual))


def domains(preds: Any, *, negate: bool = False, prior: Optional[Mapping[str, Value]] = None,
            local: bool = True) -> dict:
    """:func:`domain_of` for every name the predicates read."""
    if isinstance(preds, (str, ast.AST)):
        preds = [preds]
    exprs = [_parse(p) for p in preds]
    names = sorted({n.id for e in exprs for n in ast.walk(e) if isinstance(n, ast.Name)})
    prior = prior or {}
    return {n: domain_of(exprs, n, negate=negate, prior=prior.get(n), local=local) for n in names}


# --------------------------------------------------------------------------- #
# the eregion label
# --------------------------------------------------------------------------- #

_CLOSED = Value(kinds=Kinds(_SCALAR_KINDS))


def _label(arm: _Arm, ctx: _Ctx, hint: Optional[str]) -> str:
    """eregion's ``domain_kind`` for this arm (``_classify`` in ``domain_capture``)."""
    v = arm.value
    if v.contains or v.lacks:
        return "set_membership"
    container_hint = hint in ("list", "tuple", "set", "dict", "container")
    if container_hint or (ctx.len_seen and v.length != IntFact(0, None)):
        return "container_length"
    closed = _tighten(v.meet(_CLOSED), arm.lex)
    live = closed._live_kinds() or frozenset()
    if live == {"NoneType"}:
        return "is_none"
    if arm.members is not None:
        ms = [m for m in arm.members if closed.contains_value(m)] or list(arm.members)
        tags = {_tag(m) for m in ms}
        if not ms:
            return "guard_only"
        if tags == {"bool"}:
            return "bool_set"
        if tags == {"none"}:
            return "is_none"
        if tags <= {"int", "bool"}:
            return "int_set"
        if tags <= {"int", "float", "bool"}:
            return "float_set"
        if tags == {"str"}:
            return "str_set"
        if tags == {"bytes"}:
            return "bytes_set"
        return "guard_only"
    tags = set(ctx.tags)
    if hint in ("int", "float", "str", "bytes", "bool", "none"):
        tags = {hint}
    if "float" in tags and (v.f.lo != -_INF or v.f.hi != _INF):
        return "float_range"
    if tags & {"str", "bytes"} and (arm.lex[0] != Lex() or arm.lex[1] != Lex()):
        return "lex_range"
    if not v.i.is_bottom() and (v.i.lo is not None or v.i.hi is not None) and not tags & {"str", "bytes"}:
        if v.i.lo is not None and v.i.hi is not None:
            return "int_range_closed"
        return "int_range_low_open" if v.i.lo is not None else "int_range_high_open"
    neq = {x for x in v.neq if x is not None}
    if neq:
        nt = {_tag(x) for x in neq}
        if nt == {"str"}:
            return "str_neq"
        if nt == {"bytes"}:
            return "bytes_neq"
        if nt <= {"int", "float", "bool"}:
            return "int_neq"
    if ctx.residual or ctx.mentioned or v != Value.top():
        return "guard_only"
    return "top"


# --------------------------------------------------------------------------- #
# atoms: eregion's _extract_constraint, for fact_env.refine_all
# --------------------------------------------------------------------------- #

def atoms(test: Any, negate: bool = False) -> List[Tuple[str, str, Any]]:
    """``(var, op, value)`` atoms implied by ``test`` (true, or false when ``negate``),
    in the op vocabulary of ``tharkun.solve.fact_env.refine``.  The port of eregion's
    ``emit_bw_assume_facts._extract_constraint`` with its default-on flags (truthiness,
    ``is None``, container-literal ``==``) and chained compares.

    Every atom returned holds on the arm (a conjunction).  Where eregion's version
    would claim more than holds, this one returns nothing for that part:
    ``x is not True`` is not ``x != True`` (``1`` is not ``True`` but ``1 == True``),
    and ``x is 5`` is dropped.  ``len(x) == K`` / ``!= K`` become
    ``container_len_eq`` / ``container_len_ne``."""
    node = _parse(test)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return atoms(node.operand, not negate)
    if isinstance(node, ast.Name):
        return [(node.id, "truthy", not negate)]
    if isinstance(node, ast.BoolOp):
        conj = isinstance(node.op, ast.And) != negate
        if conj:
            return [a for part in node.values for a in atoms(part, negate)]
        if negate:                      # not (A and B): a disjunction
            return []
        var, values = None, set()
        for part in node.values:
            got = atoms(part, False)
            if len(got) != 1 or got[0][1] not in ("==", "in"):
                return []
            v, op, val = got[0]
            if var is None:
                var = v
            if v != var:
                return []
            if op == "==":
                values.add(val)
            else:
                values.update(val)
            if len(values) > MAX_CONST_SET:
                return []
        return [] if var is None else [(var, "in", values)]
    if isinstance(node, ast.Compare) and len(node.ops) > 1:
        if negate:
            return []
        ops = [node.left, *node.comparators]
        return [a for i, o in enumerate(node.ops)
                for a in atoms(ast.Compare(left=ops[i], ops=[o], comparators=[ops[i + 1]]))]
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
        return []
    op = _OPS.get(type(node.ops[0]))
    left, right = node.left, node.comparators[0]
    for side, other, flip in ((left, right, False), (right, left, True)):
        if isinstance(side, ast.Call) and isinstance(side.func, ast.Name) and side.func.id == "len" \
                and len(side.args) == 1 and isinstance(side.args[0], ast.Name) and op in ("==", "!="):
            k = _const(other)
            if type(k) is int and k >= 0:
                eff = op if not negate else _NEG[op]
                return [(side.args[0].id, "container_len_eq" if eff == "==" else "container_len_ne", k)]
            return []
    if isinstance(left, ast.Name):
        var, val_node, eff = left.id, right, op
    elif isinstance(right, ast.Name) and op in _FLIP:
        var, val_node, eff = right.id, left, _FLIP[op]
    else:
        return []
    if eff in ("is", "is not"):
        c = _const(val_node)
        if c is None:
            e = eff if not negate else _NEG[eff]
            return [(var, "==" if e == "is" else "!=", None)]
        if isinstance(c, bool) and (eff == "is") != negate:
            return [(var, "==", c)]
        return []
    if eff in ("in", "not in"):
        items = _const_items(val_node)
        if items is None:
            return []
        try:
            vals = set(items)
        except TypeError:
            return []
        return [(var, eff if not negate else _NEG[eff], vals)]
    c = _const(val_node)
    if c is _MISSING:
        if eff == "==" and not negate and isinstance(val_node, (ast.Tuple, ast.List, ast.Set)):
            items = _const_items(val_node)
            if items is not None:
                lit = frozenset(items) if isinstance(val_node, ast.Set) else tuple(items)
                return [(var, "==", lit)]
        return []
    if c is None or _is_nan(c):
        return []
    return [(var, eff if not negate else _NEG[eff], c)]
