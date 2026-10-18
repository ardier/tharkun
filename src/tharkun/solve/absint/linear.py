"""Linear forms and the int component of an absint value (docs/ABSINT.md).

``Lin`` is an exact integer linear form over zone variables.  ``IntInfo`` is the
view of a value's int part: an exact form (``lin``) and symbolic bounds
``(var, c)`` meaning ``var + c`` (``var`` is ``ZERO`` for a constant bound).  The
*storage* of the constant bounds is lane D's :class:`~tharkun.solve.values.IntFact`
inside the value's :class:`~tharkun.solve.values.Value` (``aval.py``); only the
relational parts (``lin``, bounds on a zone variable) live here.
"""

from __future__ import annotations

from .zone import ZERO, Zone


class Lin:
    """An integer linear form ``sum(coef * var) + c`` over zone variables."""

    __slots__ = ("terms", "c")

    def __init__(self, terms: dict[str, int] | None = None, c: int = 0) -> None:
        t = {k: v for k, v in (terms or {}).items() if v != 0 and k != ZERO}
        self.terms = tuple(sorted(t.items()))
        self.c = c

    @staticmethod
    def var(v: str, c: int = 0) -> "Lin":
        return Lin({v: 1}, c)

    @staticmethod
    def const(c: int) -> "Lin":
        return Lin(None, c)

    def is_const(self) -> bool:
        return not self.terms

    def single(self) -> tuple[str, int] | None:
        """``(var, c)`` when the form is ``var + c``."""
        if not self.terms:
            return (ZERO, self.c)
        if len(self.terms) == 1 and self.terms[0][1] == 1:
            return (self.terms[0][0], self.c)
        return None

    def vars(self) -> set[str]:
        return {k for k, _ in self.terms}

    def _d(self) -> dict[str, int]:
        return dict(self.terms)

    def add(self, o: "Lin") -> "Lin":
        d = self._d()
        for k, v in o.terms:
            d[k] = d.get(k, 0) + v
        return Lin(d, self.c + o.c)

    def neg(self) -> "Lin":
        return Lin({k: -v for k, v in self.terms}, -self.c)

    def sub(self, o: "Lin") -> "Lin":
        return self.add(o.neg())

    def scale(self, k: int) -> "Lin":
        return Lin({a: b * k for a, b in self.terms}, self.c * k)

    def shift(self, k: int) -> "Lin":
        return Lin(dict(self.terms), self.c + k)

    def subst_consts(self, z: Zone) -> "Lin":
        """Replace every variable the zone pins to one value by that value."""
        d = {}
        c = self.c
        for k, v in self.terms:
            lo, hi = z.bounds(k)
            if lo is not None and lo == hi:
                c += v * lo
            else:
                d[k] = v
        return Lin(d, c)

    def norm(self, z: Zone) -> "Lin":
        """``subst_consts`` plus elimination of ``x - y`` pairs whose difference the
        zone pins to one value."""
        lin = self.subst_consts(z)
        changed = True
        while changed and len(lin.terms) >= 2:
            changed = False
            d = dict(lin.terms)
            pos = [k for k, v in d.items() if v == 1]
            neg = [k for k, v in d.items() if v == -1]
            for x in pos:
                for y in neg:
                    up = z.get(x, y)
                    dn = z.get(y, x)
                    if up is not None and dn is not None and up == -dn:
                        del d[x]
                        del d[y]
                        lin = Lin(d, lin.c + up)
                        changed = True
                        break
                if changed:
                    break
        return lin

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Lin) and self.terms == o.terms and self.c == o.c

    def __hash__(self) -> int:
        return hash((self.terms, self.c))

    def __repr__(self) -> str:
        s = " + ".join(f"{'' if v == 1 else v}{k}" for k, v in self.terms)
        if self.c or not s:
            s = f"{s} + {self.c}" if s else str(self.c)
        return s


def lin_bounds(lin: Lin, z: Zone) -> tuple[int | None, int | None]:
    """Constant bounds of a linear form under a zone (DBM-exact for ``x - y + c``)."""
    if z.bottom:
        return (0, 0)
    t = lin.terms
    if not t:
        return (lin.c, lin.c)
    if len(t) == 2 and {t[0][1], t[1][1]} == {1, -1}:
        (x, _), (y, _) = (t[0], t[1]) if t[0][1] == 1 else (t[1], t[0])
        hi = z.get(x, y)
        lo = z.get(y, x)
        return (None if lo is None else lin.c - lo, None if hi is None else lin.c + hi)
    lo: int | None = lin.c
    hi: int | None = lin.c
    for k, v in t:
        klo, khi = z.bounds(k)
        if v > 0:
            lo = None if lo is None or klo is None else lo + v * klo
            hi = None if hi is None or khi is None else hi + v * khi
        else:
            lo = None if lo is None or khi is None else lo + v * khi
            hi = None if hi is None or klo is None else hi + v * klo
    return (lo, hi)


def lin_entails_le(z: Zone, a: Lin, b: Lin, k: int = 0) -> bool:
    """Does the zone prove ``a <= b + k``?"""
    if z.bottom:
        return True
    d = a.sub(b).norm(z)
    lo, hi = lin_bounds(d, z)
    return hi is not None and hi <= k


# ----------------------------------------------------------------------- intinfo
SymB = tuple  # (var, c): the bound var + c; var ZERO for a constant bound.


class IntInfo:
    """The int component of a value.

    ``lin``: the exact value as a linear form (in flight, or the variable's own
    zone variable in an environment); ``None`` when not linear.  ``lo`` / ``hi``:
    symbolic bounds ``(var, c)``, ``None`` for infinite.  Bounds are always
    sound; ``lin`` is exact.
    """

    __slots__ = ("lin", "lo", "hi")

    def __init__(self, lin: Lin | None = None, lo: SymB | None = None, hi: SymB | None = None):
        self.lin = lin
        self.lo = lo
        self.hi = hi

    @staticmethod
    def interval(lo: int | None, hi: int | None) -> "IntInfo":
        return IntInfo(
            Lin.const(lo) if lo is not None and lo == hi else None,
            None if lo is None else (ZERO, lo),
            None if hi is None else (ZERO, hi),
        )

    def __eq__(self, o: object) -> bool:
        return isinstance(o, IntInfo) and self.lin == o.lin and self.lo == o.lo and self.hi == o.hi

    def __hash__(self) -> int:
        return hash((self.lin, self.lo, self.hi))

    def __repr__(self) -> str:
        return f"Int(lin={self.lin}, lo={self.lo}, hi={self.hi})"

    def vars(self) -> set[str]:
        s = set(self.lin.vars()) if self.lin else set()
        for b in (self.lo, self.hi):
            if b is not None and b[0] != ZERO:
                s.add(b[0])
        return s


# ---------------------------------------------------------------------------- V
