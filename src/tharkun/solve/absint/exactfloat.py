"""Exact dyadic float terms (owner DECISIONS 1471, capability V_TERMS).

A float value is an *exact dyadic term* when it is known to be the double nearest to nothing: the real number
``(c + sum(k_i * x_i)) / 2**s`` itself, with ``x_i`` zone variables (the int variables the value was converted from), and that real
number is representable in binary64, i.e. ``|value| * 2**s < 2**53`` (and ``s`` is small).  IEEE-754 ``+``, ``-`` and ``*`` are correctly rounded,
so on exact operands whose exact result is representable they return the exact result: ``float(i) + float(j)`` IS ``float(i + j)`` for ``|i|, |j|,
|i + j| < 2**53``, and ``float(i) + 0.5`` is ``(2*i + 1) / 2``.  Two exact terms that are the same real number are the same double (no NaN, no
infinity; the sign of a zero is not observable by ``==``), so ``==`` between them is decided, and the exact result interval of a product of exact
operands is not rounded outward.

The term is ``("xd", s, c, ((("iv", zone_var), k), ...))`` (the ``("iv", zv)`` parts are what ``state.py`` strips when ``zv`` can change, the same
convention as the trig terms of ``lemmas.py``).  Anything not exact (a division, a float of unknown rounding, an int of unbounded range, a
product of two non-constant terms) has no term: nothing here is a lemma about rounding, it is arithmetic on values that need none.
"""

from __future__ import annotations

from fractions import Fraction

TWO53 = 2 ** 53
MAX_SCALE = 24            # denominators up to 2**24: literals like 0.5, 0.25, 1.5 ... (a literal like 0.1 has no term)


def _norm(s: int, c: int, terms: dict):
    terms = {v: k for v, k in terms.items() if k != 0}
    while s > 0 and c % 2 == 0 and all(k % 2 == 0 for k in terms.values()):
        s -= 1
        c //= 2
        terms = {v: k // 2 for v, k in terms.items()}
    return ("xd", s, c, tuple(sorted(terms.items(), key=repr)))


def of_lin(lin):
    """The term of an int with exact linear form ``lin`` (the caller has checked its bounds are within +-2**53)."""
    if lin is None:
        return None
    return _norm(0, lin.c, {("iv", v): k for v, k in lin.terms})


def of_atom(n: int):
    """a fresh exact value no other term names: the float conversion of an int with bounds within +-2**53 but no linear form (``i % 1000`` of a
    draw).  The value is immutable, so the atom needs no invalidation; two conversions are two atoms, never the same."""
    return ("xd", 0, 0, ((("at", n), 1),))


def of_float_const(x: float):
    if x != x or x in (float("inf"), float("-inf")):
        return None
    num, den = x.as_integer_ratio()
    s = den.bit_length() - 1
    if s > MAX_SCALE or abs(num) >= TWO53:
        return None
    return _norm(s, num, {})


def is_xd(t) -> bool:
    return isinstance(t, tuple) and len(t) == 4 and t[0] == "xd"


def _parts(t):
    return t[1], t[2], dict(t[3])


def add(a, b, sign: int = 1):
    sa, ca, ta = _parts(a)
    sb, cb, tb = _parts(b)
    s = max(sa, sb)
    fa, fb = 2 ** (s - sa), 2 ** (s - sb) * sign
    terms = {}
    for v, k in ta.items():
        terms[v] = terms.get(v, 0) + k * fa
    for v, k in tb.items():
        terms[v] = terms.get(v, 0) + k * fb
    return _norm(s, ca * fa + cb * fb, terms)


def neg(a):
    s, c, t = _parts(a)
    return _norm(s, -c, {v: -k for v, k in t.items()})


def scale_by_const(a, const):
    """``a * const`` where ``const`` is a term without variables (a literal): still linear."""
    sa, ca, ta = _parts(a)
    sc, cc, tc = _parts(const)
    if tc:
        return None
    return _norm(sa + sc, ca * cc, {v: k * cc for v, k in ta.items()})


def is_const(t) -> bool:
    return not t[3]


def const_value(t) -> Fraction:
    return Fraction(t[2], 2 ** t[1])


def scaled_diff(a, b):
    """``a - b`` as ``(s, c, terms)`` with a common scale: zero iff the two terms are the same real number"""
    d = add(a, b, -1)
    return d


def representable(t, lo: float, hi: float) -> bool:
    """the exact value of ``t`` lies in [lo, hi] (the result's fact interval, outward rounded) and ``|value| * 2**s < 2**53``"""
    if lo != lo or hi != hi or lo in (float("-inf"), float("inf")) or hi in (float("-inf"), float("inf")):
        return False
    return max(abs(lo), abs(hi)) * (2 ** t[1]) < TWO53


def exact_interval(op: str, alo: float, ahi: float, blo: float, bhi: float, square: bool = False):
    """(lo, hi) of the REAL ``a op b`` over the operand intervals, as floats (exact: the caller proved them representable)"""
    A, B = [Fraction(alo), Fraction(ahi)], [Fraction(blo), Fraction(bhi)]
    if op == "add":
        c = [x + y for x in A for y in B]
    elif op == "sub":
        c = [x - y for x in A for y in B]
    elif op == "mul":
        c = [x * y for x in A for y in B]
        if square and A[0] <= 0 <= A[1]:
            c.append(Fraction(0))
    else:
        return None
    lo, hi = min(c), max(c)
    if square:
        lo = Fraction(0) if A[0] <= 0 <= A[1] else lo
    if float(lo) != lo or float(hi) != hi:
        return None
    return float(lo), float(hi)
