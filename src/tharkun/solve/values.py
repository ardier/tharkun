"""Non-relational value facts and their sound transfer functions ([internal ref], from eregion).

SOLVING, by ``docs/LAYERS.md``'s rule: a claim about the set of values a variable
may hold, computed without running anything.  Every fact here is an
OVER-approximation: it contains every value the variable can really take, and
every operation keeps that true.  A fact may be wider than necessary; it is
never narrower.

The facts (all frozen; ``None`` bounds mean unbounded):

* :class:`IntFact` ``(lo, hi)`` -- Python ints in ``[lo, hi]``.  Ints are
  unbounded, so ``+ - *`` are exact on the bounds.
* :class:`FloatFact` ``(lo, hi, nan)`` -- the floats ``x`` with ``lo <= x <= hi``
  (``-inf`` / ``inf`` allowed as values and as bounds; ``-0.0`` and ``0.0``
  compare equal and are both inside any interval containing ``0``), plus NaN
  when ``nan``.  Results are rounded OUTWARD (``math.nextafter``), so the true
  float result of an operation is always inside.
* :class:`StrFact` ``(consts)`` -- a finite set of strings, or ``None`` for any.
* :class:`Kinds` ``(names)`` -- the Python type names a value may have
  (``"int"``, ``"float"``, ``"str"``, ``"bool"``, ``"NoneType"``, ...), or
  ``None`` for any type.
* :class:`Value` -- the reduced product: one fact per type, exclusions, a length
  and container membership.  It is the per-variable carrier of the guard front end.

**Equality crosses the numeric types.**  ``0 == 0.0 == -0.0 == False`` and
``1 == 1.0 == True`` in Python, so a guard ``x == 0`` (or ``x in {0, ...}``) admits
all of them.  :meth:`Value.const` is the ONE value given; :meth:`Value.equal_to` is
every value equal to it -- the one a guard front end must use for ``==`` / ``in``.
Exclusions (``Value.neq``) already follow ``==`` because they are tested with ``in``.

Each has ``join`` (union), ``meet`` (intersection), ``widen``, ``leq`` (inclusion)
and ``is_bottom`` (the empty set), and ``top()`` / ``bottom()`` / ``const(v)``.

Transfer functions return ``(result, may_raise)``: the fact for the result of
the operation over every pair of inputs that does NOT raise, and the frozenset
of exception class names some pair may raise.  ``ZeroDivisionError`` is exact:
it is present iff the divisor may be ``0`` / ``0.0`` / ``-0.0`` (a NaN divisor
does not raise; it yields NaN).  ``OverflowError`` and ``ValueError`` (a math
domain error) are reported the same way, so a caller can tell them from its
property classes.  A result of ``None`` means "not a value this module models"
(e.g. a negative float to a fractional power is a ``complex``): the caller must
treat it as any value of any type.

:func:`refine_compare` narrows two facts by a comparison, for both outcomes.

The first consumer is the abstract interpreter (lane E, ``solve/absint``); the
eregion value facts (``eregion_ir.program_extract.domain_capture.reduce_domain``,
``eregion_commons.fact_repr``) move onto these facts in the next slices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

__all__ = [
    "IntFact", "FloatFact", "StrFact", "Kinds", "Value", "Fact",
    "add", "sub", "mul", "truediv", "floordiv", "mod", "pow_", "neg", "abs_", "to_float",
    "sqrt", "sin", "cos", "binop", "refine_compare",
    "floor", "ceil", "trunc", "round_", "round_n", "min_", "max_", "exp", "log", "log10", "atan",
    "atan2", "tanh", "sq", "fabs", "hypot", "fsum", "draw_value",
]

_INF = math.inf
_NONE: frozenset = frozenset()
_ZDE = frozenset({"ZeroDivisionError"})
_OVF = frozenset({"OverflowError"})
_VAL = frozenset({"ValueError"})
_FLOAT_MAX_INT = int(1.7976931348623157e308)
_FLOAT_MAX = 1.7976931348623157e308


def _down(x: float) -> float:
    return x if math.isinf(x) else math.nextafter(x, -_INF)


def _up(x: float) -> float:
    return x if math.isinf(x) else math.nextafter(x, _INF)


# --------------------------------------------------------------------------- #
# IntFact
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class IntFact:
    """Python ints in ``[lo, hi]`` (``None``: unbounded on that side)."""

    lo: Optional[int] = None
    hi: Optional[int] = None
    empty: bool = False

    @staticmethod
    def top() -> "IntFact":
        return IntFact()

    @staticmethod
    def bottom() -> "IntFact":
        return IntFact(empty=True)

    @staticmethod
    def const(v: int) -> "IntFact":
        return IntFact(v, v)

    @staticmethod
    def range(lo: Optional[int], hi: Optional[int]) -> "IntFact":
        if lo is not None and hi is not None and lo > hi:
            return IntFact.bottom()
        return IntFact(lo, hi)

    def is_bottom(self) -> bool:
        return self.empty

    def contains(self, v: int) -> bool:
        return (not self.empty and (self.lo is None or v >= self.lo)
                and (self.hi is None or v <= self.hi))

    def leq(self, other: "IntFact") -> bool:
        if self.empty:
            return True
        if other.empty:
            return False
        return ((other.lo is None or (self.lo is not None and self.lo >= other.lo))
                and (other.hi is None or (self.hi is not None and self.hi <= other.hi)))

    def join(self, other: "IntFact") -> "IntFact":
        if self.empty:
            return other
        if other.empty:
            return self
        lo = None if self.lo is None or other.lo is None else min(self.lo, other.lo)
        hi = None if self.hi is None or other.hi is None else max(self.hi, other.hi)
        return IntFact(lo, hi)

    def meet(self, other: "IntFact") -> "IntFact":
        if self.empty or other.empty:
            return IntFact.bottom()
        lo = other.lo if self.lo is None else self.lo if other.lo is None else max(self.lo, other.lo)
        hi = other.hi if self.hi is None else self.hi if other.hi is None else min(self.hi, other.hi)
        return IntFact.range(lo, hi)

    def widen(self, other: "IntFact") -> "IntFact":
        """``self`` widened by the next iterate ``other``: a bound that grew goes to infinity."""
        if self.empty:
            return other
        if other.empty:
            return self
        lo = self.lo if (other.lo is not None and self.lo is not None and other.lo >= self.lo) else None
        hi = self.hi if (other.hi is not None and self.hi is not None and other.hi <= self.hi) else None
        return IntFact(lo, hi)

    def has_zero(self) -> bool:
        return self.contains(0)

    def as_float(self) -> "FloatFact":
        """The floats these ints convert to (ignoring the ints ``float()`` rejects)."""
        return to_float(self)[0]


# --------------------------------------------------------------------------- #
# FloatFact
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FloatFact:
    """Floats in ``[lo, hi]`` (infinities included), plus NaN when ``nan``.

    The empty interval is ``lo > hi``; ``FloatFact.bottom()`` is the empty set."""

    lo: float = -_INF
    hi: float = _INF
    nan: bool = True

    @staticmethod
    def top() -> "FloatFact":
        return FloatFact()

    @staticmethod
    def bottom() -> "FloatFact":
        return FloatFact(_INF, -_INF, False)

    @staticmethod
    def const(v: float) -> "FloatFact":
        v = float(v)
        if math.isnan(v):
            return FloatFact(_INF, -_INF, True)
        return FloatFact(v, v, False)

    @staticmethod
    def range(lo: float, hi: float, nan: bool = False) -> "FloatFact":
        return FloatFact(float(lo), float(hi), nan)

    def has_interval(self) -> bool:
        return self.lo <= self.hi

    def is_bottom(self) -> bool:
        return not self.has_interval() and not self.nan

    def contains(self, v: float) -> bool:
        if math.isnan(v):
            return self.nan
        return self.lo <= v <= self.hi

    def leq(self, other: "FloatFact") -> bool:
        if self.nan and not other.nan:
            return False
        if not self.has_interval():
            return True
        return other.has_interval() and other.lo <= self.lo and self.hi <= other.hi

    def join(self, other: "FloatFact") -> "FloatFact":
        if not self.has_interval():
            return FloatFact(other.lo, other.hi, self.nan or other.nan)
        if not other.has_interval():
            return FloatFact(self.lo, self.hi, self.nan or other.nan)
        return FloatFact(min(self.lo, other.lo), max(self.hi, other.hi), self.nan or other.nan)

    def meet(self, other: "FloatFact") -> "FloatFact":
        lo, hi = max(self.lo, other.lo), min(self.hi, other.hi)
        if lo > hi:
            lo, hi = _INF, -_INF
        return FloatFact(lo, hi, self.nan and other.nan)

    def widen(self, other: "FloatFact") -> "FloatFact":
        if not self.has_interval():
            return other
        if not other.has_interval():
            return FloatFact(self.lo, self.hi, self.nan or other.nan)
        lo = self.lo if other.lo >= self.lo else -_INF
        hi = self.hi if other.hi <= self.hi else _INF
        return FloatFact(lo, hi, self.nan or other.nan)

    def has_zero(self) -> bool:
        return self.has_interval() and self.lo <= 0.0 <= self.hi

    def has_inf(self) -> bool:
        return self.has_interval() and (math.isinf(self.lo) or math.isinf(self.hi))


# --------------------------------------------------------------------------- #
# StrFact, Kinds
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class StrFact:
    """A finite set of strings (``consts``), or any string (``consts is None``)."""

    consts: Optional[frozenset] = None

    @staticmethod
    def top() -> "StrFact":
        return StrFact(None)

    @staticmethod
    def bottom() -> "StrFact":
        return StrFact(frozenset())

    @staticmethod
    def const(v: str) -> "StrFact":
        return StrFact(frozenset({v}))

    def is_bottom(self) -> bool:
        return self.consts is not None and not self.consts

    def contains(self, v: str) -> bool:
        return self.consts is None or v in self.consts

    def leq(self, other: "StrFact") -> bool:
        if other.consts is None:
            return True
        return self.consts is not None and self.consts <= other.consts

    def join(self, other: "StrFact") -> "StrFact":
        if self.consts is None or other.consts is None:
            return StrFact(None)
        return StrFact(self.consts | other.consts)

    def meet(self, other: "StrFact") -> "StrFact":
        if self.consts is None:
            return other
        if other.consts is None:
            return self
        return StrFact(self.consts & other.consts)

    def widen(self, other: "StrFact", limit: int = 16) -> "StrFact":
        j = self.join(other)
        return StrFact(None) if j.consts is not None and len(j.consts) > limit else j


@dataclass(frozen=True)
class Kinds:
    """The Python type names a value may have (``None``: any type)."""

    names: Optional[frozenset] = None

    @staticmethod
    def top() -> "Kinds":
        return Kinds(None)

    @staticmethod
    def bottom() -> "Kinds":
        return Kinds(frozenset())

    @staticmethod
    def const(v) -> "Kinds":
        return Kinds(frozenset({type(v).__name__}))

    @staticmethod
    def of(*names: str) -> "Kinds":
        return Kinds(frozenset(names))

    def is_bottom(self) -> bool:
        return self.names is not None and not self.names

    def may_be(self, name: str) -> bool:
        return self.names is None or name in self.names

    def may_be_none(self) -> bool:
        return self.may_be("NoneType")

    def leq(self, other: "Kinds") -> bool:
        return other.names is None or (self.names is not None and self.names <= other.names)

    def join(self, other: "Kinds") -> "Kinds":
        if self.names is None or other.names is None:
            return Kinds(None)
        return Kinds(self.names | other.names)

    def meet(self, other: "Kinds") -> "Kinds":
        if self.names is None:
            return other
        if other.names is None:
            return self
        return Kinds(self.names & other.names)

    def widen(self, other: "Kinds") -> "Kinds":
        return self.join(other)


Fact = Union[IntFact, FloatFact]


# --------------------------------------------------------------------------- #
# Value: the reduced product, one fact per Python type a variable may hold
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Value:
    """Every value a variable may hold, as one fact per Python type.

    The set a ``Value`` stands for is the union, over the type names in ``kinds``,
    of the values of that type its component allows:

    * ``int`` -> ``i`` (an :class:`IntFact`), ``float`` -> ``f`` (a :class:`FloatFact`),
      ``str`` -> ``s`` and ``bytes`` -> ``by`` (a :class:`StrFact` of str / of bytes),
      ``bool`` -> ``bools`` (a subset of ``{False, True}``, or ``None`` for both),
      ``NoneType`` -> ``None``; any other type name is unconstrained;
    * minus every value in ``neq`` (``x != v`` / ``x not in {...}``);
    * a sized value (str, bytes, container) also has ``len(v)`` in ``length``;
    * a container also holds every element of ``contains`` and none of ``lacks``
      (``k in s`` / ``k not in s``).

    ``kinds=Kinds.top()`` means any type.  All operations are componentwise, so a
    ``Value`` is exactly as precise as its parts; ``meet`` is the conjunction of two
    facts about one variable and ``join`` the disjunction.  This is the carrier the
    guard front end (``reduce_domain`` / ``_extract_constraint`` ported from eregion,
    ``solve/guard_domain``) produces per variable.
    """

    kinds: Kinds = Kinds(None)
    i: IntFact = IntFact()
    f: FloatFact = FloatFact()
    s: StrFact = StrFact(None)
    by: StrFact = StrFact(None)
    bools: Optional[frozenset] = None
    neq: frozenset = frozenset()
    length: IntFact = IntFact(0, None)
    contains: frozenset = frozenset()
    lacks: frozenset = frozenset()

    @staticmethod
    def top() -> "Value":
        return Value()

    @staticmethod
    def bottom() -> "Value":
        return Value(kinds=Kinds.bottom())

    @staticmethod
    def only(kind: str, **parts) -> "Value":
        """A value of the single type ``kind``: every other type's component is empty, so
        joins stay precise; ``parts`` sets this type's components."""
        base = dict(i=IntFact.bottom(), f=FloatFact.bottom(), s=StrFact.bottom(),
                    by=StrFact.bottom(), bools=frozenset(), length=IntFact.bottom())
        if kind in ("str", "bytes"):
            base["length"] = IntFact(0, None)
        own = {"int": {"i": IntFact()}, "float": {"f": FloatFact()}, "str": {"s": StrFact()},
               "bytes": {"by": StrFact()}, "bool": {"bools": None}}.get(kind, {})
        base.update(own)
        base.update(parts)
        return Value(kinds=Kinds.of(kind), **base)

    @staticmethod
    def const(v) -> "Value":
        """The single value ``v`` (a scalar: int, float, str, bytes, bool or None)."""
        if v is None:
            return Value.only("NoneType")
        if isinstance(v, bool):
            return Value.only("bool", bools=frozenset({v}))
        if isinstance(v, int):
            return Value.only("int", i=IntFact.const(v))
        if isinstance(v, float):
            return Value.only("float", f=FloatFact.const(v))
        if isinstance(v, str):
            return Value.only("str", s=StrFact.const(v), length=IntFact.const(len(v)))
        if isinstance(v, bytes):
            return Value.only("bytes", by=StrFact.const(v), length=IntFact.const(len(v)))
        return Value.only(type(v).__name__)

    @staticmethod
    def equal_to(v) -> "Value":
        """Every scalar ``x`` with ``x == v`` -- what a guard ``x == v`` (or ``x in {v}``)
        admits.  Python's numbers compare across types (``0 == 0.0 == False``), so an
        int, float or bool ``v`` also admits its equals of the other two kinds."""
        out = Value.const(v)
        if isinstance(v, (bool, int, float)) and not (isinstance(v, float) and math.isnan(v)):
            if not isinstance(v, float):
                try:
                    if float(v) == v:
                        out = out.join(Value.const(float(v)))
                except OverflowError:
                    pass
            if not isinstance(v, int) or isinstance(v, bool):
                if isinstance(v, bool) or (math.isfinite(v) and float(v).is_integer()):
                    out = out.join(Value.const(int(v)))
            if not isinstance(v, bool) and v in (0, 1):
                out = out.join(Value.const(bool(v)))
            if isinstance(v, bool):
                out = out.join(Value.const(float(v)))
        return out

    @staticmethod
    def of_int(i: IntFact) -> "Value":
        return Value.only("int", i=i)

    @staticmethod
    def of_float(f: FloatFact) -> "Value":
        return Value.only("float", f=f)

    @staticmethod
    def of_str(s: StrFact) -> "Value":
        return Value.only("str", s=s)

    def _live_kinds(self) -> Optional[frozenset]:
        """The type names whose component is not empty (``None``: any type)."""
        if self.kinds.names is None:
            return None
        empty = {"int": self.i.is_bottom(), "float": self.f.is_bottom(),
                 "str": self.s.is_bottom() or self.length.is_bottom(),
                 "bytes": self.by.is_bottom() or self.length.is_bottom(),
                 "bool": self.bools is not None and not self.bools}
        return frozenset(k for k in self.kinds.names if not empty.get(k, False))

    def is_bottom(self) -> bool:
        live = self._live_kinds()
        return live is not None and not live

    def contains_value(self, v) -> bool:
        """Whether the concrete scalar ``v`` is in this set (used by tests and checks)."""
        try:
            if v in self.neq:
                return False
        except TypeError:
            pass
        name = type(v).__name__
        if not self.kinds.may_be(name):
            return False
        if isinstance(v, bool):
            return self.bools is None or v in self.bools
        if isinstance(v, int):
            return self.i.contains(v)
        if isinstance(v, float):
            return self.f.contains(v)
        if isinstance(v, (str, bytes)):
            part = self.s if isinstance(v, str) else self.by
            return part.contains(v) and self.length.contains(len(v))
        return True

    def leq(self, other: "Value") -> bool:
        if self.is_bottom():
            return True
        return (self.kinds.leq(other.kinds) and self.i.leq(other.i) and self.f.leq(other.f)
                and self.s.leq(other.s) and self.by.leq(other.by)
                and (other.bools is None or (self.bools is not None and self.bools <= other.bools))
                and other.neq <= self.neq and self.length.leq(other.length)
                and other.contains <= self.contains and other.lacks <= self.lacks)

    def join(self, other: "Value") -> "Value":
        if self.is_bottom():
            return other
        if other.is_bottom():
            return self
        return Value(self.kinds.join(other.kinds), self.i.join(other.i), self.f.join(other.f),
                     self.s.join(other.s), self.by.join(other.by),
                     None if self.bools is None or other.bools is None else self.bools | other.bools,
                     self.neq & other.neq, self.length.join(other.length),
                     self.contains & other.contains, self.lacks & other.lacks)

    def meet(self, other: "Value") -> "Value":
        bools = other.bools if self.bools is None else self.bools if other.bools is None \
            else self.bools & other.bools
        return Value(self.kinds.meet(other.kinds), self.i.meet(other.i), self.f.meet(other.f),
                     self.s.meet(other.s), self.by.meet(other.by), bools,
                     self.neq | other.neq, self.length.meet(other.length),
                     self.contains | other.contains, self.lacks | other.lacks)

    def widen(self, other: "Value") -> "Value":
        j = self.join(other)
        return Value(j.kinds, self.i.widen(other.i), self.f.widen(other.f),
                     self.s.widen(other.s), self.by.widen(other.by), j.bools, j.neq,
                     self.length.widen(other.length), j.contains, j.lacks)


# --------------------------------------------------------------------------- #
# float transfer functions
# --------------------------------------------------------------------------- #

def _f(x: Fact) -> Tuple[FloatFact, frozenset]:
    return to_float(x) if isinstance(x, IntFact) else (x, _NONE)


def _corners(a: FloatFact, b: FloatFact, op) -> Tuple[float, float, bool]:
    """min / max of ``op`` over the interval corners, NaN corners dropped (reported)."""
    vals, nan = [], False
    for x in (a.lo, a.hi):
        for y in (b.lo, b.hi):
            try:
                v = op(x, y)
            except (OverflowError, ZeroDivisionError):
                continue
            if math.isnan(v):
                nan = True
            else:
                vals.append(v)
    if not vals:
        return _INF, -_INF, nan
    return min(vals), max(vals), nan


def _fadd(a: FloatFact, b: FloatFact, sign: int) -> FloatFact:
    if not a.has_interval() or not b.has_interval():
        return FloatFact(_INF, -_INF, a.nan or b.nan)
    blo, bhi = (b.lo, b.hi) if sign > 0 else (-b.hi, -b.lo)
    nan = a.nan or b.nan
    # inf + -inf is nan
    if (a.hi == _INF and blo == -_INF) or (a.lo == -_INF and bhi == _INF):
        nan = True
    # IEEE addition is correctly rounded, so monotone in each operand: the float sums of
    # the corners bound every float sum exactly (no outward step, which would push a
    # bound of 0.0 below zero and lose the sign)
    lo = a.lo + blo
    hi = a.hi + bhi
    if math.isnan(lo):
        lo = -_INF
    if math.isnan(hi):
        hi = _INF
    return FloatFact(lo, hi, nan)


def _fmul(a: FloatFact, b: FloatFact) -> FloatFact:
    if not a.has_interval() or not b.has_interval():
        return FloatFact(_INF, -_INF, a.nan or b.nan)
    nan = a.nan or b.nan or (a.has_zero() and b.has_inf()) or (b.has_zero() and a.has_inf())
    lo, hi, corner_nan = _corners(a, b, lambda x, y: x * y)
    if corner_nan:                      # 0 * inf corner: the products near it reach 0
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    # correctly rounded, hence monotone within each sign region: the corners are exact bounds
    return FloatFact(lo, hi, nan)


def _nonzero_parts(b: FloatFact) -> list:
    """``b``'s interval without zero, as closed sub-intervals (zero replaced by the
    smallest subnormal of the right sign)."""
    if not b.has_interval():
        return []
    tiny = 5e-324
    parts = []
    if b.lo < 0.0:
        parts.append((b.lo, min(b.hi, -tiny)))
    if b.hi > 0.0:
        parts.append((max(b.lo, tiny), b.hi))
    return [p for p in parts if p[0] <= p[1]]


def _fdiv(a: FloatFact, b: FloatFact) -> Tuple[FloatFact, frozenset]:
    raises = _ZDE if b.has_zero() else _NONE
    nan = a.nan or b.nan or (a.has_inf() and b.has_inf())
    out = FloatFact(_INF, -_INF, nan)
    if not a.has_interval():
        return out, raises
    for lo_b, hi_b in _nonzero_parts(b):
        lo, hi, cn = _corners(a, FloatFact(lo_b, hi_b, False), lambda x, y: x / y)
        if cn:
            lo, hi = min(lo, -_INF if a.lo < 0 else 0.0), max(hi, _INF if a.hi > 0 else 0.0)
        out = out.join(FloatFact(lo, hi, False))   # correctly rounded: corners are exact
    return out, raises


def add(a: Fact, b: Fact):
    """``a + b``."""
    if isinstance(a, IntFact) and isinstance(b, IntFact):
        return _iadd(a, b, 1), _NONE
    (fa, ra), (fb, rb) = _f(a), _f(b)
    return _fadd(fa, fb, 1), ra | rb


def sub(a: Fact, b: Fact):
    """``a - b``."""
    if isinstance(a, IntFact) and isinstance(b, IntFact):
        return _iadd(a, b, -1), _NONE
    (fa, ra), (fb, rb) = _f(a), _f(b)
    return _fadd(fa, fb, -1), ra | rb


def mul(a: Fact, b: Fact):
    """``a * b``."""
    if isinstance(a, IntFact) and isinstance(b, IntFact):
        return _imul(a, b), _NONE
    (fa, ra), (fb, rb) = _f(a), _f(b)
    return _fmul(fa, fb), ra | rb


def truediv(a: Fact, b: Fact):
    """``a / b`` (always a float; ``ZeroDivisionError`` iff ``b`` may be zero)."""
    (fa, ra), (fb, rb) = _f(a), _f(b)
    if isinstance(a, IntFact) and isinstance(b, IntFact):
        # int / int is correctly rounded from the exact quotient; the float
        # conversions above can only widen, and large ints raise OverflowError
        # only when the QUOTIENT overflows -- report it conservatively
        ra = rb = _NONE
        if _unbounded_or_huge(a) or _unbounded_or_huge(b):
            ra = _OVF
    res, rz = _fdiv(fa, fb)
    return res, ra | rb | rz


def _unbounded_or_huge(a: IntFact) -> bool:
    return a.lo is None or a.hi is None or abs(a.lo) > _FLOAT_MAX_INT or abs(a.hi) > _FLOAT_MAX_INT


def floordiv(a: Fact, b: Fact):
    """``a // b``."""
    if isinstance(a, IntFact) and isinstance(b, IntFact):
        return _ifloordiv(a, b)
    (fa, ra), (fb, rb) = _f(a), _f(b)
    q, rz = _fdiv(fa, fb)
    # CPython computes a // b exactly (from fmod), not as floor(a / b) in floats: the
    # rounded quotient can be off by one either way, so widen by one
    nan = q.nan or fa.has_inf()          # inf // finite is nan
    if not q.has_interval():
        return FloatFact(_INF, -_INF, nan), ra | rb | rz
    lo = float(math.floor(q.lo)) - 1.0 if math.isfinite(q.lo) else q.lo
    hi = float(math.floor(q.hi)) + 1.0 if math.isfinite(q.hi) else q.hi
    return FloatFact(_down(lo), _up(hi), nan), ra | rb | rz


def mod(a: Fact, b: Fact):
    """``a % b`` (the result has the divisor's sign)."""
    if isinstance(a, IntFact) and isinstance(b, IntFact):
        return _imod(a, b)
    (fa, ra), (fb, rb) = _f(a), _f(b)
    raises = ra | rb | (_ZDE if fb.has_zero() else _NONE)
    nan = fa.nan or fb.nan or fa.has_inf()
    if not fb.has_interval():
        return FloatFact(_INF, -_INF, nan), raises
    m = max(abs(fb.lo), abs(fb.hi))
    lo = -m if fb.lo < 0 else 0.0
    hi = m if fb.hi > 0 else 0.0
    return FloatFact(_down(lo), _up(hi), nan), raises


def neg(a: Fact):
    if isinstance(a, IntFact):
        if a.empty:
            return a, _NONE
        return IntFact(None if a.hi is None else -a.hi, None if a.lo is None else -a.lo), _NONE
    if not a.has_interval():
        return a, _NONE
    return FloatFact(-a.hi, -a.lo, a.nan), _NONE


def abs_(a: Fact):
    if isinstance(a, IntFact):
        if a.empty:
            return a, _NONE
        if a.lo is not None and a.lo >= 0:
            return a, _NONE
        if a.hi is not None and a.hi <= 0:
            return neg(a)[0], _NONE
        hi = None if a.lo is None or a.hi is None else max(-a.lo, a.hi)
        return IntFact(0, hi), _NONE
    if not a.has_interval():
        return a, _NONE
    if a.lo >= 0.0:
        return FloatFact(0.0 if a.lo == 0.0 else a.lo, a.hi, a.nan), _NONE
    if a.hi <= 0.0:
        return FloatFact(-a.hi, -a.lo, a.nan), _NONE
    return FloatFact(0.0, max(-a.lo, a.hi), a.nan), _NONE


def to_float(a: IntFact):
    """``float(a)``: rounded outward; ``OverflowError`` for ints beyond the float range."""
    if a.empty:
        return FloatFact.bottom(), _NONE
    raises = _NONE
    big = float(_FLOAT_MAX_INT)
    if a.lo is None or a.lo < -_FLOAT_MAX_INT:
        lo, raises = -big, _OVF
    else:
        lo = float(a.lo)
        lo = _down(lo) if lo > a.lo else lo
    if a.hi is None or a.hi > _FLOAT_MAX_INT:
        hi, raises = big, _OVF
    else:
        hi = float(a.hi)
        hi = _up(hi) if hi < a.hi else hi
    return FloatFact(max(lo, -big), min(hi, big), False), raises


def pow_(a: Fact, b: Fact):
    """``a ** b``.  ``None`` result: a complex result is possible (negative base,
    fractional exponent), which this module does not model."""
    if isinstance(a, IntFact) and isinstance(b, IntFact):
        return _ipow(a, b)
    (fa, ra), (fb, rb) = _f(a), _f(b)
    raises = ra | rb
    if fa.has_zero() and fb.has_interval() and fb.lo < 0:
        raises |= _ZDE
    integral_exp = fb.has_interval() and fb.lo == fb.hi and float(fb.lo).is_integer()
    if fa.has_interval() and fa.lo < 0 and not integral_exp:
        return None, raises | _OVF
    raises |= _OVF                      # float ** can overflow (OverflowError)
    nan = fa.nan or fb.nan
    if fa.has_interval() and fa.lo >= 1.0 and fb.has_interval() and fb.lo >= 0.0:
        lo, hi = _corners(fa, fb, _safe_pow)[:2]
        return FloatFact(_down(lo), _up(hi), nan), raises
    if fa.has_interval() and fa.lo >= 0.0:
        return FloatFact(0.0, _INF, nan), raises
    return FloatFact(-_INF, _INF, nan), raises


def _safe_pow(x: float, y: float) -> float:
    try:
        return x ** y
    except OverflowError:
        return _INF


def sqrt(a: Fact):
    """``math.sqrt(a)``: ``ValueError`` for a negative argument (``-0.0`` is fine)."""
    fa, raises = _f(a)
    if fa.has_interval() and fa.lo < 0.0:
        raises |= _VAL
    if not fa.has_interval() or fa.hi < 0.0:
        return FloatFact(_INF, -_INF, fa.nan), raises
    lo = math.sqrt(max(fa.lo, 0.0))
    hi = math.sqrt(fa.hi)
    return FloatFact(max(0.0, _down(lo)) if fa.lo > 0 else 0.0, _up(hi), fa.nan), raises


def _trig(a: Fact, fn):
    fa, raises = _f(a)
    if fa.has_inf():
        raises |= _VAL                  # math.sin(inf): ValueError (math domain error)
    if not fa.has_interval() or (fa.lo == fa.hi and math.isinf(fa.lo)):
        return FloatFact(_INF, -_INF, fa.nan), raises
    if fa.lo == fa.hi:
        v = fn(fa.lo)
        return FloatFact(max(-1.0, _down(v)), min(1.0, _up(v)), fa.nan), raises
    return FloatFact(-1.0, 1.0, fa.nan), raises


def sin(a: Fact):
    """``math.sin(a)`` in ``[-1, 1]``; ``ValueError`` on an infinity."""
    return _trig(a, math.sin)


def cos(a: Fact):
    """``math.cos(a)`` in ``[-1, 1]``; ``ValueError`` on an infinity."""
    return _trig(a, math.cos)


def _to_int_monotone(a: Fact, fn):
    """``fn(a)`` for a monotone float -> int conversion (floor, ceil, trunc, round):
    ``ValueError`` on NaN, ``OverflowError`` on an infinity."""
    if isinstance(a, IntFact):
        return a, _NONE
    raises = (_VAL if a.nan else _NONE) | (_OVF if a.has_inf() else _NONE)
    if not a.has_interval() or (a.lo == a.hi and math.isinf(a.lo)):
        return IntFact.bottom(), raises
    lo = None if math.isinf(a.lo) else fn(a.lo)
    hi = None if math.isinf(a.hi) else fn(a.hi)
    return IntFact(lo, hi), raises


def floor(a: Fact):
    """``math.floor(a)`` (an int)."""
    return _to_int_monotone(a, math.floor)


def ceil(a: Fact):
    """``math.ceil(a)`` (an int)."""
    return _to_int_monotone(a, math.ceil)


def trunc(a: Fact):
    """``int(a)`` / ``math.trunc(a)`` of a float (toward zero)."""
    return _to_int_monotone(a, math.trunc)


def round_(a: Fact):
    """``round(a)`` with no digits (round half to even, an int)."""
    return _to_int_monotone(a, round)


def _minmax(a: Fact, b: Fact, pick_min: bool):
    if isinstance(a, IntFact) and isinstance(b, IntFact):
        if a.empty or b.empty:
            return (b if a.empty else a), _NONE
        if pick_min:
            lo = None if a.lo is None or b.lo is None else min(a.lo, b.lo)
            hi = b.hi if a.hi is None else a.hi if b.hi is None else min(a.hi, b.hi)
        else:
            lo = b.lo if a.lo is None else a.lo if b.lo is None else max(a.lo, b.lo)
            hi = None if a.hi is None or b.hi is None else max(a.hi, b.hi)
        return IntFact(lo, hi), _NONE
    if isinstance(a, IntFact) != isinstance(b, IntFact):
        # min(2, 2.5) is the int 2 and min(3, 2.5) the float 2.5: the result is one of the
        # arguments, of either kind -- the caller joins the arguments' Values
        return None, _NONE
    (fa, ra), (fb, rb) = _f(a), _f(b)
    if fa.nan or fb.nan:
        # with a NaN the result depends on the argument order, but it is always one of
        # the arguments: it lies in their union
        return fa.join(fb), ra | rb
    if not fa.has_interval() or not fb.has_interval():
        return FloatFact.bottom(), ra | rb
    f = min if pick_min else max
    return FloatFact(f(fa.lo, fb.lo), f(fa.hi, fb.hi), False), ra | rb


def min_(a: Fact, b: Fact):
    """``min(a, b)`` (``None`` for an int and a float: the result keeps its own kind)."""
    return _minmax(a, b, True)


def max_(a: Fact, b: Fact):
    """``max(a, b)`` (``None`` for an int and a float: the result keeps its own kind)."""
    return _minmax(a, b, False)


def sq(a: Fact):
    """``a * a`` of ONE value (not two independent ones): never negative."""
    if isinstance(a, IntFact):
        if a.empty:
            return a, _NONE
        m = abs_(a)[0]
        return IntFact(m.lo * m.lo if m.lo is not None else None,
                       m.hi * m.hi if m.hi is not None else None), _NONE
    if not a.has_interval():
        return FloatFact(_INF, -_INF, a.nan), _NONE
    m = abs_(a)[0]
    return FloatFact(m.lo * m.lo, m.hi * m.hi, a.nan), _NONE


def fabs(a: Fact):
    """``math.fabs(a)`` (a float; ``OverflowError`` for an int beyond the float range)."""
    fa, raises = _f(a)
    return abs_(fa)[0], raises


def log10(a: Fact):
    """``math.log10(a)``: monotone; ``ValueError`` for ``a <= 0``.  An int of any size is
    taken exactly (``math.log10`` does not convert it to a float first)."""
    fa, raises = (_cmp_view(a), _NONE) if isinstance(a, IntFact) else (a, _NONE)
    if fa.has_interval() and fa.lo <= 0.0:
        raises |= _VAL
    if not fa.has_interval() or fa.hi <= 0.0:
        return FloatFact(_INF, -_INF, fa.nan), raises
    lo = -_INF if fa.lo <= 0.0 else _down(math.log10(fa.lo))
    return FloatFact(lo, _up(math.log10(fa.hi)), fa.nan), raises


def hypot(*args: Fact):
    """``math.hypot(*args)``: at least every ``|arg|`` and at most their sum; an infinite
    argument gives ``inf`` even beside a NaN."""
    facts, raises = [], _NONE
    for x in args:
        fx, r = _f(x)
        facts.append(abs_(fx)[0])
        raises |= r
    if any(not f.has_interval() and not f.nan for f in facts):
        return FloatFact.bottom(), raises
    nan = any(f.nan for f in facts)
    lo = max((f.lo for f in facts if f.has_interval()), default=0.0)
    hi = 0.0
    for f in facts:
        hi = hi + f.hi if f.has_interval() else _INF
    return FloatFact(max(0.0, _down(lo)), _up(hi), nan), raises


def fsum(*args: Fact):
    """``math.fsum(args)``: the correctly rounded exact sum -- bounded by the corner sums,
    widened a step; ``OverflowError`` when a finite sum overflows, ``ValueError`` for
    ``inf + -inf``."""
    facts, raises = [], _NONE
    for x in args:
        fx, r = _f(x)
        facts.append(fx)
        raises |= r
    if any(f.is_bottom() for f in facts):
        return FloatFact.bottom(), raises
    nan = any(f.nan for f in facts)
    if any(f.has_interval() and f.hi == _INF for f in facts) and \
            any(f.has_interval() and f.lo == -_INF for f in facts):
        raises |= _VAL
    raises |= _OVF
    lo = hi = 0.0
    for f in facts:
        if f.has_interval():
            lo, hi = lo + f.lo, hi + f.hi
    lo = -_INF if math.isnan(lo) else lo
    hi = _INF if math.isnan(hi) else hi
    return FloatFact(_down(lo), _up(hi), nan), raises


def round_n(a: Fact, n: int):
    """``round(a, n)`` of a float (a float): within half a unit of the n-th digit,
    widened a step; ``OverflowError`` cannot happen, NaN / inf pass through."""
    fa, raises = _f(a)
    if not fa.has_interval():
        return FloatFact(_INF, -_INF, fa.nan), raises
    half = 0.5 * 10.0 ** (-n) if -300 < n < 300 else (0.0 if n >= 300 else _INF)
    lo = fa.lo - half if math.isfinite(fa.lo) else fa.lo
    hi = fa.hi + half if math.isfinite(fa.hi) else fa.hi
    return FloatFact(_down(lo), _up(hi), fa.nan), raises


def _exp_or_inf(x: float) -> float:
    try:
        return math.exp(x)
    except OverflowError:
        return _INF


def exp(a: Fact):
    """``math.exp(a)``: monotone; ``OverflowError`` past about 709.78."""
    fa, raises = _f(a)
    if not fa.has_interval():
        return FloatFact(_INF, -_INF, fa.nan), raises
    if fa.hi > 709.0:
        raises |= _OVF
    lo, hi = _exp_or_inf(fa.lo), _exp_or_inf(fa.hi)
    return FloatFact(max(0.0, _down(lo)), _up(hi), fa.nan), raises


def log(a: Fact):
    """``math.log(a)``: monotone; ``ValueError`` for ``a <= 0``.  An int of any size is
    taken exactly (``math.log`` does not convert it to a float first)."""
    fa, raises = (_cmp_view(a), _NONE) if isinstance(a, IntFact) else (a, _NONE)
    if fa.has_interval() and fa.lo <= 0.0:
        raises |= _VAL
    if not fa.has_interval() or fa.hi <= 0.0:
        return FloatFact(_INF, -_INF, fa.nan), raises
    lo = -_INF if fa.lo <= 0.0 else _down(math.log(fa.lo))
    return FloatFact(lo, _up(math.log(fa.hi)), fa.nan), raises


def atan(a: Fact):
    """``math.atan(a)`` in ``[-pi/2, pi/2]`` (monotone)."""
    fa, raises = _f(a)
    if not fa.has_interval():
        return FloatFact(_INF, -_INF, fa.nan), raises
    return FloatFact(max(-1.5707963267948968, _down(math.atan(fa.lo))),
                     min(1.5707963267948968, _up(math.atan(fa.hi))), fa.nan), raises


def atan2(y: Fact, x: Fact):
    """``math.atan2(y, x)`` in ``[-pi, pi]``."""
    (fy, ry), (fx, rx) = _f(y), _f(x)
    nan = fy.nan or fx.nan
    if not fy.has_interval() or not fx.has_interval():
        return FloatFact(_INF, -_INF, nan), ry | rx
    return FloatFact(-3.1415926535897936, 3.1415926535897936, nan), ry | rx


def tanh(a: Fact):
    """``math.tanh(a)`` in ``[-1, 1]`` (monotone)."""
    fa, raises = _f(a)
    if not fa.has_interval():
        return FloatFact(_INF, -_INF, fa.nan), raises
    return FloatFact(max(-1.0, _down(math.tanh(fa.lo))), min(1.0, _up(math.tanh(fa.hi))),
                     fa.nan), raises


# --------------------------------------------------------------------------- #
# int transfer functions (Python ints never overflow)
# --------------------------------------------------------------------------- #

def _iadd(a: IntFact, b: IntFact, sign: int) -> IntFact:
    if a.empty or b.empty:
        return IntFact.bottom()
    blo, bhi = (b.lo, b.hi) if sign > 0 else (None if b.hi is None else -b.hi,
                                              None if b.lo is None else -b.lo)
    lo = None if a.lo is None or blo is None else a.lo + blo
    hi = None if a.hi is None or bhi is None else a.hi + bhi
    return IntFact(lo, hi)


def _imul(a: IntFact, b: IntFact) -> IntFact:
    if a.empty or b.empty:
        return IntFact.bottom()
    if a.lo is None or a.hi is None or b.lo is None or b.hi is None:
        if a == IntFact.const(0) or b == IntFact.const(0):
            return IntFact.const(0)
        return IntFact.top()
    ps = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
    return IntFact(min(ps), max(ps))


def _ifloordiv(a: IntFact, b: IntFact):
    raises = _ZDE if b.has_zero() else _NONE
    if a.empty or b.empty:
        return IntFact.bottom(), raises
    parts = []
    if b.lo is None or b.lo < 0:
        parts.append((b.lo, -1 if b.hi is None or b.hi >= 0 else b.hi))
    if b.hi is None or b.hi > 0:
        parts.append((1 if b.lo is None or b.lo <= 0 else b.lo, b.hi))
    out = IntFact.bottom()
    for lo_b, hi_b in parts:
        if a.lo is None or a.hi is None or lo_b is None or hi_b is None:
            # |a // b| <= |a| for |b| >= 1: bounded iff a is bounded
            if a.lo is not None and a.hi is not None:
                m = max(abs(a.lo), abs(a.hi))
                out = out.join(IntFact(-m - 1, m))
            else:
                return IntFact.top(), raises
            continue
        qs = [x // y for x in (a.lo, a.hi) for y in (lo_b, hi_b)]
        out = out.join(IntFact(min(qs), max(qs)))
    return out, raises


def _imod(a: IntFact, b: IntFact):
    raises = _ZDE if b.has_zero() else _NONE
    if a.empty or b.empty:
        return IntFact.bottom(), raises
    lo, hi = 0, 0
    if b.hi is None or b.hi > 0:
        hi = None if b.hi is None else b.hi - 1
        # a already inside [0, b.lo - 1]: a % b == a
        if b.lo is not None and b.lo > 0 and a.lo is not None and a.hi is not None \
                and 0 <= a.lo and a.hi < b.lo:
            return a, raises
    if b.lo is None or b.lo < 0:
        lo = None if b.lo is None else b.lo + 1
    return IntFact(lo, hi), raises


def _ipow(a: IntFact, b: IntFact):
    raises = _NONE
    if a.empty or b.empty:
        return IntFact.bottom(), raises
    if b.lo is not None and b.lo >= 0:
        if a.lo is not None and a.lo >= 0 and a.hi is not None and b.hi is not None \
                and b.hi <= 64 and a.hi.bit_length() * max(b.hi, 1) <= 4096:
            # a non-negative int base is 0 (0**0 == 1, else 0) or >= 1 (monotone in
            # both): the extremes are at the corners
            ps = [x ** y for x in (a.lo, a.hi) for y in (b.lo, b.hi)]
            return IntFact(min(ps), max(ps)), raises
        return IntFact.top(), raises
    # a negative exponent gives a float, and 0 ** negative raises
    if a.has_zero():
        raises |= _ZDE
    return FloatFact(-_INF, _INF, False), raises | _OVF


_OPS = {"add": add, "sub": sub, "mul": mul, "truediv": truediv, "floordiv": floordiv,
        "mod": mod, "pow": pow_}


def binop(op: str, a: Fact, b: Fact):
    """``_OPS[op](a, b)`` for ``op`` in ``add sub mul truediv floordiv mod pow``."""
    return _OPS[op](a, b)


# --------------------------------------------------------------------------- #
# draws: what a nondeterministic call may return (SV-COMP _sv_verifier, random)
# --------------------------------------------------------------------------- #

_IDX = frozenset({"IndexError"})


def _numeric(v: "Value"):
    """A numeric view of a ``Value`` argument: an ``IntFact`` when it can only be an int
    or a bool, else a ``FloatFact`` covering its ints, bools and floats; ``None`` when
    it may be something else (a non-numeric argument: nothing is claimed)."""
    names = v.kinds.names
    if names is None or not names or not names <= {"int", "bool", "float"}:
        return None
    i = v.i if "int" in names else IntFact.bottom()
    if "bool" in names:
        bools = v.bools if v.bools is not None else frozenset({False, True})
        for b in bools:
            i = i.join(IntFact.const(int(b)))
    if "float" not in names:
        return i
    f = v.f
    if not i.is_bottom():
        f = f.join(_cmp_view(i))
    return f


def _uniform(a, b):
    """``random.uniform(a, b)`` is ``a + (b - a) * random()``: between a and b (either
    order), give or take the last bits of rounding; NaN / inf when ``b - a`` overflows
    or an argument is not finite."""
    fa, fb = (_cmp_view(a) if isinstance(a, IntFact) else a), (_cmp_view(b) if isinstance(b, IntFact) else b)
    if not fa.has_interval() or not fb.has_interval():
        return FloatFact(_INF, -_INF, fa.nan or fb.nan)
    lo, hi = min(fa.lo, fb.lo), max(fa.hi, fb.hi)
    span = sub(FloatFact(fb.lo, fb.hi, False), FloatFact(fa.lo, fa.hi, False))[0]
    wild = span.has_inf() or span.nan or fa.nan or fb.nan
    if wild:
        return FloatFact(-_INF, _INF, True)
    for _ in range(4):
        lo, hi = _down(lo), _up(hi)
    return FloatFact(lo, hi, False)


_PRINTABLE_LEN = IntFact(0, None)


def draw_value(func: str, args: list) -> tuple:
    """``(Value, may_raise)`` for what the nondeterministic call ``func(*args)`` may
    return, ``args`` being :class:`Value` s.  ``func`` is a name of the SV-COMP
    ``_sv_verifier`` model (``nondet_int``, ``nondet_bool``, ``nondet_float``,
    ``nondet_float_range``, ``nondet_str``) or of ``random`` (``random``, ``uniform``,
    ``randint``, ``randrange``, ``choice_index``, ``getrandbits``), with or without its
    module prefix.  An unknown ``func`` is ``(Value.top(), frozenset())``.

    Faithful to the reference ``_sv_verifier.py``: ``nondet_int`` is any int (unbounded),
    ``nondet_float`` any float including ``inf``, ``-inf``, ``nan`` and ``-0.0``,
    ``nondet_str`` any string of printable characters, and ``nondet_float_range(a, b)``
    is ``random.uniform(a, b)``."""
    name = func.rsplit(".", 1)[-1]
    nums = [_numeric(a) if isinstance(a, Value) else a for a in args]
    if name == "nondet_int":
        return Value.of_int(IntFact.top()), _NONE
    if name == "nondet_bool":
        return Value.only("bool"), _NONE
    if name == "nondet_float":
        return Value.of_float(FloatFact.top()), _NONE
    if name == "nondet_str":
        return Value.only("str", length=_PRINTABLE_LEN), _NONE
    if name == "random":
        return Value.of_float(FloatFact(0.0, math.nextafter(1.0, 0.0), False)), _NONE
    if name in ("uniform", "nondet_float_range"):
        if len(nums) != 2 or any(n is None for n in nums):
            return Value.of_float(FloatFact.top()), _NONE
        return Value.of_float(_uniform(*nums)), _NONE
    if name == "randint":
        if len(nums) != 2 or any(not isinstance(n, IntFact) for n in nums):
            return Value.of_int(IntFact.top()), _VAL | frozenset({"TypeError"})
        a, b = nums
        raises = _VAL if (a.hi is None or b.lo is None or a.hi > b.lo) else _NONE
        return Value.of_int(IntFact.range(a.lo, b.hi)), raises
    if name == "randrange":
        if not nums or len(nums) > 3 or any(not isinstance(n, IntFact) for n in nums):
            return Value.of_int(IntFact.top()), _VAL | frozenset({"TypeError"})
        if len(nums) == 1:
            start, stop, step = IntFact.const(0), nums[0], IntFact.const(1)
        else:
            start, stop = nums[0], nums[1]
            step = nums[2] if len(nums) == 3 else IntFact.const(1)
        raises = _VAL if (step.has_zero() or stop.lo is None or start.hi is None
                          or not (step.lo is not None and step.lo > 0 and start.hi < stop.lo)) else _NONE
        if step.lo is not None and step.lo > 0:
            hi = None if stop.hi is None else stop.hi - 1
            return Value.of_int(IntFact.range(start.lo, hi)), raises
        return Value.of_int(start.join(stop)), raises
    if name == "choice_index":
        # the index random.choice(seq) picks, given len(seq)
        n = nums[0] if nums and isinstance(nums[0], IntFact) else IntFact(0, None)
        raises = _IDX if (n.lo is None or n.lo <= 0) else _NONE
        return Value.of_int(IntFact(0, None if n.hi is None else n.hi - 1)), raises
    if name == "getrandbits":
        k = nums[0] if nums and isinstance(nums[0], IntFact) else IntFact(None, None)
        raises = _VAL if (k.lo is None or k.lo < 0) else _NONE
        hi = None if k.hi is None or k.hi > 4096 else 2 ** max(k.hi, 0) - 1
        return Value.of_int(IntFact(0, hi)), raises
    return Value.top(), _NONE


# --------------------------------------------------------------------------- #
# comparisons
# --------------------------------------------------------------------------- #

def _int_cmp(op: str, a: IntFact, b: IntFact):
    """(a_true, b_true, a_false, b_false) for ints."""
    def lt(x, y):          # x < y: x.hi <= y.hi - 1, y.lo >= x.lo + 1
        return (x.meet(IntFact(None, None if y.hi is None else y.hi - 1)),
                y.meet(IntFact(None if x.lo is None else x.lo + 1, None)))

    def le(x, y):
        return (x.meet(IntFact(None, y.hi)), y.meet(IntFact(x.lo, None)))

    if op == "<":
        at, bt = lt(a, b)
        bf, af = le(b, a)
        return at, bt, af, bf
    if op == "<=":
        at, bt = le(a, b)
        bf, af = lt(b, a)
        return at, bt, af, bf
    if op in (">", ">="):
        bt, at, bf, af = _int_cmp("<" if op == ">" else "<=", b, a)
        return at, bt, af, bf
    if op in ("==", "!="):
        m = a.meet(b)
        af, bf = a, b
        if b.lo is not None and b.lo == b.hi:
            af = _int_exclude(a, b.lo)
        if a.lo is not None and a.lo == a.hi:
            bf = _int_exclude(b, a.lo)
        return (m, m, af, bf) if op == "==" else (af, bf, m, m)
    raise ValueError(op)


def _int_exclude(a: IntFact, v: int) -> IntFact:
    if a.empty:
        return a
    if a.lo == v and a.hi == v:
        return IntFact.bottom()
    if a.lo == v:
        return IntFact(v + 1, a.hi)
    if a.hi == v:
        return IntFact(a.lo, v - 1)
    return a


def _float_cmp(op: str, a: FloatFact, b: FloatFact):
    """(a_true, b_true, a_false, b_false) for floats: NaN makes every ordered
    comparison and ``==`` false and ``!=`` true."""
    def upto(x: FloatFact, hi: float, strict: bool) -> FloatFact:
        if strict:
            hi = math.nextafter(hi, -_INF) if not math.isinf(hi) or hi > 0 else hi
        return x.meet(FloatFact(-_INF, hi, False))

    def from_(x: FloatFact, lo: float, strict: bool) -> FloatFact:
        if strict:
            lo = math.nextafter(lo, _INF) if not math.isinf(lo) or lo < 0 else lo
        return x.meet(FloatFact(lo, _INF, False))

    no_nan_a = FloatFact(a.lo, a.hi, False)
    no_nan_b = FloatFact(b.lo, b.hi, False)
    if op in ("<", "<="):
        strict = op == "<"
        if not a.has_interval() or not b.has_interval():
            at, bt = FloatFact.bottom(), FloatFact.bottom()
        else:
            at = upto(no_nan_a, b.hi, strict)
            bt = from_(no_nan_b, a.lo, strict)
        # false: a >= b (resp. a > b) with both numbers, or either is NaN
        if not a.has_interval() or not b.has_interval():
            af_num, bf_num = FloatFact.bottom(), FloatFact.bottom()
        else:
            af_num = from_(no_nan_a, b.lo, not strict)
            bf_num = upto(no_nan_b, a.hi, not strict)
        af = af_num.join(FloatFact(_INF, -_INF, a.nan))
        bf = bf_num.join(FloatFact(_INF, -_INF, b.nan))
        if b.nan:                       # b NaN: the comparison is false for every a
            af = af.join(a)
        if a.nan:
            bf = bf.join(b)
        return at, bt, af, bf
    if op in (">", ">="):
        bt, at, bf, af = _float_cmp("<" if op == ">" else "<=", b, a)
        return at, bt, af, bf
    if op in ("==", "!="):
        m = no_nan_a.meet(no_nan_b)
        af, bf = a, b
        if b.has_interval() and b.lo == b.hi and not b.nan:
            af = _float_exclude(a, b.lo)
        if a.has_interval() and a.lo == a.hi and not a.nan:
            bf = _float_exclude(b, a.lo)
        return (m, m, af, bf) if op == "==" else (af, bf, m, m)
    raise ValueError(op)


def _float_exclude(a: FloatFact, v: float) -> FloatFact:
    if not a.has_interval():
        return a
    if a.lo == v and a.hi == v:
        return FloatFact(_INF, -_INF, a.nan)
    if a.lo == v:
        return FloatFact(math.nextafter(v, _INF), a.hi, a.nan)
    if a.hi == v:
        return FloatFact(a.lo, math.nextafter(v, -_INF), a.nan)
    return a


def refine_compare(op: str, a: Fact, b: Fact):
    """``(a_true, b_true, a_false, b_false)``: ``a`` and ``b`` narrowed to the pairs for
    which ``a <op> b`` is true, and to those for which it is false.  ``op`` is one of
    ``< <= > >= == !=``.  Mixed int / float operands are compared as Python does
    (exactly); the int side is narrowed to the ints inside the float bound."""
    if isinstance(a, IntFact) and isinstance(b, IntFact):
        return _int_cmp(op, a, b)
    if isinstance(a, FloatFact) and isinstance(b, FloatFact):
        return _float_cmp(op, a, b)
    fa = a if isinstance(a, FloatFact) else _cmp_view(a)
    fb = b if isinstance(b, FloatFact) else _cmp_view(b)
    at, bt, af, bf = _float_cmp(op, fa, fb)
    return (_back(a, at), _back(b, bt), _back(a, af), _back(b, bf))


def _cmp_view(a: IntFact) -> FloatFact:
    """The ints of ``a`` placed on the float line for an exact comparison, outward: the low end is the largest float
    that is at most the smallest int, the high end the smallest float that is at least the largest int.  An int beyond the
    float range is above every FINITE float and below ``inf`` (``inf > 10 ** 400`` is True in Python), so a huge positive
    int sits in ``[largest finite float, inf]`` -- never at the infinity itself, which would prune ``x > n`` for ``x = inf``
    -- and a huge negative one in ``[-inf, -largest finite float]``.  The ends are never computed with ``float(n)`` for an
    int out of range (``OverflowError``)."""
    if a.empty:
        return FloatFact.bottom()
    if a.lo is None or a.lo < -_FLOAT_MAX_INT:
        lo = -_INF
    elif a.lo > _FLOAT_MAX_INT:
        lo = _FLOAT_MAX
    else:
        lo = float(a.lo)
        if lo > a.lo:
            lo = _down(lo)
    if a.hi is None or a.hi > _FLOAT_MAX_INT:
        hi = _INF
    elif a.hi < -_FLOAT_MAX_INT:
        hi = -_FLOAT_MAX
    else:
        hi = float(a.hi)
        if hi < a.hi:
            hi = _up(hi)
    return FloatFact(lo, hi, False)


def _back(orig: Fact, f: FloatFact) -> Fact:
    """A float refinement of an operand, returned in the operand's own kind."""
    if isinstance(orig, FloatFact):
        return f
    if not f.has_interval():
        return IntFact.bottom()
    big = float(_FLOAT_MAX_INT)
    # a bound at the float range's edge came from clamping a huge int: no bound at all
    lo = None if f.lo <= -big else math.ceil(f.lo)
    hi = None if f.hi >= big else math.floor(f.hi)
    # float bounds were rounded outward from ints: widen by one to stay sound
    lo = None if lo is None else lo - 1
    hi = None if hi is None else hi + 1
    return orig.meet(IntFact(lo, hi))
