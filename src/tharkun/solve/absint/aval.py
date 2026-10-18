"""The absint abstract value: lane D's shared :class:`~tharkun.solve.values.Value`
plus what a per-variable fact cannot hold (docs/ABSINT.md).

One fact structure, owned by Tharkun (``solve/values.py``), carries every
*scalar* fact of a value:

* ``Value.kinds`` -- the scalar types it may have (``int``, ``bool``, ``float``,
  ``str``, ``NoneType``);
* ``Value.i`` -- the constant bounds of its int part (:class:`IntFact`);
* ``Value.f`` -- its float part (:class:`FloatFact`: outward-rounded, NaN flag);
* ``Value.s`` -- its string constants (:class:`StrFact`);
* ``Value.bools`` -- its possible truth values.

absint adds only what needs the whole-program state:

* ``lin`` / ``slo`` / ``shi`` -- the int part's link to the zone (an exact
  linear form, and bounds on a zone variable: relational facts);
* ``term`` / ``why`` -- the float part's value term and the proof text of a
  bound a lemma produced (``lemmas.py``);
* ``refs`` -- heap objects it may be (``heap.py``); ``st`` -- static objects
  (builtins, modules, exception classes); ``top`` -- anything (any use aborts).

The attribute view ``v.i`` / ``v.b`` / ``v.f`` / ``v.s`` / ``v.n`` is what the
interpreter reads; it is derived from the ``Value`` on every access.
"""

from __future__ import annotations

from ..values import FloatFact, IntFact, Kinds, StrFact, Value
from .linear import IntInfo, Lin  # noqa: F401  (re-exported for the interpreter)
from .zone import ZERO

ANY_STR = "ANY"

_NO_INT = IntFact.bottom()
_NO_FLOAT = FloatFact.bottom()
_NO_STR = StrFact.bottom()


class FloatInfo:
    """The float view of a value: the :class:`FloatFact`, its value term, and the
    proof text behind a lemma bound (``why`` is not part of equality)."""

    __slots__ = ("fact", "term", "why")

    def __init__(self, fact: FloatFact | None = None, term=None, why: str = ""):
        self.fact = fact if fact is not None else FloatFact.top()
        self.term = term
        self.why = why

    def __eq__(self, o: object) -> bool:
        return isinstance(o, FloatInfo) and self.fact == o.fact and self.term == o.term

    def __hash__(self) -> int:
        return hash((self.fact, self.term))

    def __repr__(self) -> str:
        f = self.fact
        t = f" term={self.term}" if self.term is not None else ""
        return f"Float([{f.lo}, {f.hi}]{' nan' if f.nan else ''}{t})"


def _merge_why(a: str, b: str) -> str:
    if a == b or not b:
        return a
    if not a:
        return b
    parts = list(dict.fromkeys([p.strip() for p in (a + "; " + b).split(";") if p.strip()]))
    return "; ".join(parts[:6])


def f_join(a, b):
    if not a:
        return b or False
    if not b:
        return a
    if a == b:
        return a
    return FloatInfo(a.fact.join(b.fact), a.term if a.term == b.term else None, _merge_why(a.why, b.why))


def f_leq(a, b) -> bool:
    if not a:
        return True
    if not b:
        return False
    if b.term is not None and a.term != b.term:
        return False
    return a.fact.leq(b.fact)


def f_widen(a, b):
    if not a:
        return b or False
    if not b:
        return a
    j = f_join(a, b)
    return FloatInfo(a.fact.widen(j.fact), j.term, j.why)


class V:
    """An abstract value.  Construct it from the view components (``i`` an
    :class:`IntInfo`, ``b`` a set of bools, ``f`` a :class:`FloatInfo`, ``s`` a set
    of str constants or ``ANY_STR``, ``n`` None-ness); they are stored in a
    :class:`~tharkun.solve.values.Value`."""

    __slots__ = ("val", "lin", "slo", "shi", "term", "why", "refs", "st", "top")

    def __init__(self, i=None, b=None, f=False, s=None, n=False, refs=frozenset(), st=frozenset(), top=False):
        kinds = []
        ifact, lin, slo, shi = _NO_INT, None, None, None
        if i is not None:
            kinds.append("int")
            lin = i.lin
            clo = chi = None
            if i.lo is not None:
                if i.lo[0] == ZERO:
                    clo = i.lo[1]
                else:
                    slo = i.lo
            if i.hi is not None:
                if i.hi[0] == ZERO:
                    chi = i.hi[1]
                else:
                    shi = i.hi
            ifact = IntFact(clo, chi)
            if clo is not None and chi is not None and clo > chi:
                ifact = IntFact.bottom()
        bools = frozenset()
        if b:
            kinds.append("bool")
            bools = frozenset(b)
        ffact, term, why = _NO_FLOAT, None, ""
        if f:
            kinds.append("float")
            ffact, term, why = f.fact, f.term, f.why
        sfact = _NO_STR
        length = IntFact.bottom()
        if s is not None:
            kinds.append("str")
            sfact = StrFact(None) if s == ANY_STR else StrFact(frozenset(s))
            length = IntFact(0, None)
        if n:
            kinds.append("NoneType")
        self.val = Value(Kinds(frozenset(kinds)), ifact, ffact, sfact, _NO_STR, bools, frozenset(), length)
        self.lin = lin
        self.slo = slo
        self.shi = shi
        self.term = term
        self.why = why
        self.refs = refs
        self.st = st
        self.top = top

    # ------------------------------------------------------------------ views
    def _has(self, kind: str) -> bool:
        return kind in self.val.kinds.names

    @property
    def i(self) -> IntInfo | None:
        if not self._has("int"):
            return None
        c = self.val.i
        lo = self.slo if self.slo is not None else (None if c.lo is None else (ZERO, c.lo))
        hi = self.shi if self.shi is not None else (None if c.hi is None else (ZERO, c.hi))
        return IntInfo(self.lin, lo, hi)

    @property
    def b(self):
        if not self._has("bool"):
            return None
        return self.val.bools if self.val.bools is not None else frozenset({True, False})

    @property
    def f(self):
        if not self._has("float"):
            return False
        return FloatInfo(self.val.f, self.term, self.why)

    @property
    def s(self):
        if not self._has("str"):
            return None
        return ANY_STR if self.val.s.consts is None else self.val.s.consts

    @property
    def n(self) -> bool:
        return self._has("NoneType")

    def replace(self, **kw) -> "V":
        d = {k: kw[k] if k in kw else getattr(self, k) for k in ("i", "b", "f", "s", "n", "refs", "st", "top")}
        return V(**d)

    # ------------------------------------------------------------------ queries
    def is_bot(self) -> bool:
        return not self.val.kinds.names and not self.refs and not self.st and not self.top

    def only_int(self) -> bool:
        return self.val.kinds.names == frozenset({"int"}) and not self.refs and not self.st and not self.top

    def kinds(self) -> list[str]:
        k = []
        if self.top:
            k.append("top")
        for name, label in (("int", "int"), ("bool", "bool"), ("float", "float"), ("str", "str"),
                            ("NoneType", "None")):
            if self._has(name):
                k.append(label)
        if self.refs:
            k.append("obj")
        if self.st:
            k.append("static")
        return k

    def _key(self):
        return (self.val, self.lin, self.slo, self.shi, self.term, self.refs, self.st, self.top)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, V) and self._key() == o._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        parts = []
        if self.top:
            parts.append("TOP")
        if self.i is not None:
            parts.append(repr(self.i))
        if self.b:
            parts.append(f"bool{sorted(self.b)}")
        if self.f:
            parts.append(repr(self.f))
        if self.s is not None:
            parts.append("str" if self.s == ANY_STR else f"str{sorted(self.s)}")
        if self.n:
            parts.append("None")
        if self.refs:
            parts.append(f"refs{sorted(map(str, self.refs))}")
        if self.st:
            parts.append(f"st{sorted(map(str, self.st))}")
        return "V(" + " | ".join(parts) + ")" if parts else "V(bot)"


BOT = V()
NONE = V(n=True)
FLOAT = V(f=FloatInfo())
ANYSTR = V(s=ANY_STR)
BOOL = V(b=frozenset({True, False}))
TOP = V(top=True)


def const_bool(x: bool) -> V:
    return V(b=frozenset({x}))


def const_int(c: int) -> V:
    return V(i=IntInfo.interval(c, c))


def int_iv(lo: int | None, hi: int | None) -> V:
    return V(i=IntInfo.interval(lo, hi))


def const_str(s: str) -> V:
    return V(s=frozenset({s}))
