"""``solve_int_guard`` -- a native integer arithmetic solver for guards ([internal ref]).

SOLVING, by ``docs/LAYERS.md``'s rule: the input is a guard AST over named
INTEGER variables; the output is a witness (an assignment that makes the guard
true), a refutation, or unknown.  Nothing is executed; the stdlib only.

``solve_condition`` falls back to this when its own inference leaves a guard
unsolved and the guard's free names are known to be integers (``var_types``,
an ``int`` annotation, or integer facts).  The static solver used to give up on
``x * 7 - 11 == 3489`` (it returned ``x = 0``, which does not satisfy it);
this module answers ``x = 500``.

What it handles, exactly (no search window):

* polynomials over the variables with ``+``, ``-``, ``*``, unary ``-`` and
  ``**`` by a small constant exponent;
* ``abs(e)`` as an auxiliary ``a`` with ``a >= 0`` and ``a * a == e * e``;
* ``e << k``, ``e >> k`` and ``e & (2**k - 1)`` for a constant ``k`` (a scale, a
  floor division and a remainder);
* ``e // m`` and ``e % m`` for a nonzero constant ``m``: each introduces an
  auxiliary integer ``q`` with ``e // m == q`` encoded as the linear bracket
  ``m*q <= e <= m*q + m - 1`` (``m > 0``; mirrored for ``m < 0``) and
  ``e % m == e - m*q`` -- so modular guards become linear ones;
* comparisons ``== != < <= > >=`` (chains are expanded by ``normalize_guard``),
  ``and`` / ``or`` (disjunctive normal form, a bounded number of branches),
  ``not`` over a comparison, and a bare integer's truthiness (``e != 0``).

How it solves one conjunction:

1. **Linear equalities are eliminated exactly.**  An equality with a unit
   coefficient substitutes that variable.  Otherwise the two smallest
   coefficients ``a x + b y`` are replaced by ``g w`` (``g = gcd(a, b)``) through
   the complete integer parametrisation ``x = u w + (b/g) t``,
   ``y = v w - (a/g) t`` (``u a + v b = g``), which lowers the equation's
   variable count by one.  A constant ``c`` not divisible by the coefficients'
   gcd refutes the branch -- this is the only way a branch is REFUTED.
2. **A univariate polynomial equality** is solved for its integer roots
   (exactly for degree <= 2 via ``isqrt``; for higher degree by locating the
   real roots numerically and checking the neighbouring integers exactly).
3. **The rest is a bounded depth-first assignment**: each remaining variable
   takes its exact root set or its bounds from the constraints that are
   univariate in it at that point, then small values; every constraint is
   checked exactly as soon as it is ground.

Every witness is re-checked against the ORIGINAL guard before it is returned,
so a witness is always a true one.  ``unknown`` is not a claim.
"""

from __future__ import annotations

import ast
import cmath
import copy
import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional

__all__ = ["ArithResult", "solve_int_guard", "int_roots", "free_names"]

#: bounds on work -- never claims
_MAX_EXP = 8
_MAX_DEGREE = 12
_MAX_BRANCHES = 32
_MAX_NODES = 20000
_MAX_DEPTH = 64
_MAX_DIVISOR_TRIAL = 10**5

Mono = tuple  # sorted tuple of (var, exponent)
Poly = dict  # Mono -> int


class _Decline(Exception):
    pass


@dataclass
class ArithResult:
    """``status`` is ``sat`` (``witness`` makes the guard true), ``unsat`` (every
    branch was refuted by exact linear elimination) or ``unknown``."""

    status: str
    witness: dict = field(default_factory=dict)
    explanation: str = ""


# --- polynomial arithmetic ----------------------------------------------------

def _const(k: int) -> Poly:
    return {(): k} if k else {}


def _var(name: str) -> Poly:
    return {((name, 1),): 1}


def _add(p: Poly, q: Poly, sign: int = 1) -> Poly:
    out = dict(p)
    for m, c in q.items():
        v = out.get(m, 0) + sign * c
        if v:
            out[m] = v
        else:
            out.pop(m, None)
    return out


def _scale(p: Poly, k: int) -> Poly:
    return {m: c * k for m, c in p.items()} if k else {}


def _mono_mul(a: Mono, b: Mono) -> Mono:
    d: dict = {}
    for v, e in a + b:
        d[v] = d.get(v, 0) + e
    return tuple(sorted(d.items()))


def _mul(p: Poly, q: Poly) -> Poly:
    out: Poly = {}
    for m1, c1 in p.items():
        for m2, c2 in q.items():
            m = _mono_mul(m1, m2)
            v = out.get(m, 0) + c1 * c2
            if v:
                out[m] = v
            else:
                out.pop(m, None)
    return out


def _degree(p: Poly) -> int:
    return max((sum(e for _, e in m) for m in p), default=0)


def _vars(p: Poly) -> set:
    return {v for m in p for v, _ in m}


def _const_of(p: Poly) -> Optional[int]:
    if not p:
        return 0
    if set(p) == {()}:
        return p[()]
    return None


def _subst(p: Poly, var: str, q: Poly) -> Poly:
    """``p`` with ``var`` replaced by the polynomial ``q``."""
    if var not in _vars(p):
        return p
    out: Poly = {}
    powers = {0: _const(1)}
    for m, c in p.items():
        e = dict(m).get(var, 0)
        rest = tuple((v, k) for v, k in m if v != var)
        if e not in powers:
            base = powers[max(powers)]
            for i in range(max(powers) + 1, e + 1):
                base = _mul(base, q)
                powers[i] = base
        out = _add(out, _mul({rest: c}, powers[e]))
    return out


def _eval(p: Poly, env: dict) -> int:
    total = 0
    for m, c in p.items():
        t = c
        for v, e in m:
            t *= env[v] ** e
        total += t
    return total


# --- AST -> constraints -------------------------------------------------------

@dataclass
class _Con:
    poly: Poly
    op: str  # "==" | "!=" | "<=" : ``poly op 0``


def _poly_key(p: Poly) -> tuple:
    return tuple(sorted(p.items()))


class _Translator:
    def __init__(self, int_vars: set):
        self.int_vars = int_vars
        self.memo: dict = {}         # shared auxiliaries: (op, argument) -> variable
        self._n = 0

    def fresh(self, stem: str) -> str:
        self._n += 1
        return f"__{stem}{self._n}"

    def term(self, node: ast.expr) -> tuple:
        """``(poly, side_constraints)`` for an integer term."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise _Decline("non-int constant")
            return _const(node.value), []
        if isinstance(node, ast.Name):
            if node.id not in self.int_vars:
                raise _Decline(f"{node.id} is not a known integer")
            return _var(node.id), []
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            p, side = self.term(node.operand)
            return (_scale(p, -1) if isinstance(node.op, ast.USub) else p), side
        if isinstance(node, ast.BinOp):
            lp, ls = self.term(node.left)
            if isinstance(node.op, ast.Pow):
                k = _const_of(self.term(node.right)[0])
                base = _const_of(lp)
                if base is not None and k is not None and 0 <= k <= 256 and abs(base) <= 2**64:
                    return _const(base ** k), ls
                if k is None or not 0 <= k <= _MAX_EXP:
                    raise _Decline("pow")
                out = _const(1)
                for _ in range(k):
                    out = _mul(out, lp)
                return out, ls
            rp, rs = self.term(node.right)
            side = ls + rs
            if isinstance(node.op, ast.Add):
                return _add(lp, rp), side
            if isinstance(node.op, ast.Sub):
                return _add(lp, rp, -1), side
            if isinstance(node.op, ast.Mult):
                return _mul(lp, rp), side
            if isinstance(node.op, (ast.LShift, ast.RShift)):
                k = _const_of(rp)
                if k is None or not 0 <= k <= 256:
                    raise _Decline("shift")
                if isinstance(node.op, ast.LShift):
                    return _scale(lp, 2 ** k), side
                return self._divmod(lp, 2 ** k, side, floor=True)
            if isinstance(node.op, ast.BitAnd):
                # e & (2**k - 1) == e % 2**k (for every Python int)
                m = _const_of(rp) if _const_of(lp) is None else _const_of(lp)
                other = lp if _const_of(lp) is None else rp
                if m is None or m < 0 or (m + 1) & m:
                    raise _Decline("bitand")
                return self._divmod(other, m + 1, side, floor=False)
            if isinstance(node.op, (ast.FloorDiv, ast.Mod)):
                m = _const_of(rp)
                if not m:
                    raise _Decline("division by a non-constant or zero")
                return self._divmod(lp, m, side, floor=isinstance(node.op, ast.FloorDiv))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "abs"
                and len(node.args) == 1 and not node.keywords):
            # a == abs(e)  <=>  a >= 0 and a*a == e*e
            p, side = self.term(node.args[0])
            key = ("abs", _poly_key(p))
            if key in self.memo:            # one auxiliary per distinct argument
                return self.memo[key], side
            a = _var(self.fresh("a"))
            self.memo[key] = a
            return a, side + [_Con(_scale(a, -1), "<="), _Con(_add(_mul(a, a), _mul(p, p), -1), "==")]
        raise _Decline(type(node).__name__)

    def _divmod(self, lp: Poly, m: int, side: list, floor: bool) -> tuple:
        """``lp // m`` (``floor``) or ``lp % m`` through an auxiliary quotient ``q``."""
        if _const_of(lp) is not None:
            k = _const_of(lp)
            return _const(k // m if floor else k % m), side
        key = ("divmod", _poly_key(lp), m)
        if key in self.memo:                # one quotient per distinct (dividend, divisor)
            qp = self.memo[key]
            return (qp if floor else _add(lp, _scale(qp, m), -1)), side
        q = self.fresh("q")
        qp = _var(q)
        self.memo[key] = qp
        # lp - m*q is the remainder r:  0 <= r <= m-1 (m > 0),  m+1 <= r <= 0 (m < 0)
        r = _add(lp, _scale(qp, m), -1)
        if m > 0:
            side = side + [_Con(_scale(r, -1), "<="), _Con(_add(r, _const(-(m - 1))), "<=")]
        else:
            side = side + [_Con(r, "<="), _Con(_add(_scale(r, -1), _const(m + 1)), "<=")]
        return (qp if floor else r), side

    def compare(self, left: ast.expr, op: ast.cmpop, right: ast.expr, negate: bool) -> list:
        lp, ls = self.term(left)
        rp, rs = self.term(right)
        d = _add(lp, rp, -1)       # left - right
        flip = {ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Lt: ast.GtE, ast.GtE: ast.Lt,
                ast.Gt: ast.LtE, ast.LtE: ast.Gt}
        kind = type(op)
        if negate:
            if kind not in flip:
                raise _Decline("cmpop")
            kind = flip[kind]
        if kind is ast.Eq:
            c = _Con(d, "==")
        elif kind is ast.NotEq:
            c = _Con(d, "!=")
        elif kind is ast.LtE:
            c = _Con(d, "<=")
        elif kind is ast.Lt:
            c = _Con(_add(d, _const(1)), "<=")
        elif kind is ast.GtE:
            c = _Con(_scale(d, -1), "<=")
        elif kind is ast.Gt:
            c = _Con(_add(_scale(d, -1), _const(1)), "<=")
        else:
            raise _Decline("cmpop")
        return ls + rs + [c]

    def dnf(self, node: ast.expr, negate: bool = False) -> list:
        """A list of conjunctions (each a list of constraints)."""
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self.dnf(node.operand, not negate)
        if isinstance(node, ast.BoolOp):
            is_and = isinstance(node.op, ast.And) != negate
            parts = [self.dnf(v, negate) for v in node.values]
            if is_and:
                out = [[]]
                for p in parts:
                    out = [a + b for a in out for b in p]
                    if len(out) > _MAX_BRANCHES:
                        raise _Decline("too many branches")
                return out
            out = [c for p in parts for c in p]
            if len(out) > _MAX_BRANCHES:
                raise _Decline("too many branches")
            return out
        if isinstance(node, ast.Compare):
            if len(node.ops) == 1:
                return [self.compare(node.left, node.ops[0], node.comparators[0], negate)]
            pieces = []
            left = node.left
            for op, right in zip(node.ops, node.comparators):
                pieces.append(ast.Compare(left=left, ops=[op], comparators=[right]))
                left = right
            return self.dnf(ast.BoolOp(op=ast.And(), values=pieces), negate)
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return [[]] if node.value != negate else []
        # truthiness of an integer term
        p, side = self.term(node)
        return [side + [_Con(p, "==" if negate else "!=")]]


# --- integer roots of a univariate polynomial --------------------------------

def _coeffs(p: Poly, var: str) -> list:
    """Dense coefficients, lowest degree first."""
    deg = _degree(p)
    out = [0] * (deg + 1)
    for m, c in p.items():
        out[dict(m).get(var, 0)] += c
    while len(out) > 1 and out[-1] == 0:
        out.pop()
    return out


def _horner(cs: list, x) -> int:
    acc = 0
    for c in reversed(cs):
        acc = acc * x + c
    return acc


def int_roots(cs: list) -> Optional[list]:
    """The integer roots of ``sum(cs[i] * x**i)``, sorted.  Exact for degree <= 2;
    for higher degree, every root the numeric root finder isolates (each checked
    exactly).  ``None`` for the zero polynomial (every integer is a root)."""
    cs = list(cs)
    while len(cs) > 1 and cs[-1] == 0:
        cs.pop()
    if all(c == 0 for c in cs):
        return None
    roots: set = set()
    # factor out x**k
    k = 0
    while cs[k] == 0:
        k += 1
    if k:
        roots.add(0)
        cs = cs[k:]
    deg = len(cs) - 1
    if deg == 0:
        pass
    elif deg == 1:
        b, a = cs
        if b % a == 0:
            roots.add(-b // a)
    elif deg == 2:
        c, b, a = cs
        disc = b * b - 4 * a * c
        if disc >= 0:
            s = math.isqrt(disc)
            if s * s == disc:
                for num in (-b + s, -b - s):
                    if num % (2 * a) == 0:
                        roots.add(num // (2 * a))
    elif deg <= _MAX_DEGREE:
        for z in _approx_roots(cs):
            if abs(z.imag) > 1e-6 * max(1.0, abs(z)):
                continue
            r = z.real
            if not math.isfinite(r):
                continue
            base = int(round(r))
            spread = max(2, int(abs(r) * 1e-12) + 2)
            for x in range(base - spread, base + spread + 1):
                if _horner(cs, x) == 0:
                    roots.add(x)
        # small constant term: the rational root theorem is exact
        c0 = abs(cs[0])
        if c0 and c0 <= 10**7:
            for d in range(1, math.isqrt(c0) + 1):
                if c0 % d == 0:
                    for x in (d, -d, c0 // d, -(c0 // d)):
                        if _horner(cs, x) == 0:
                            roots.add(x)
    return sorted(roots)


def _approx_roots(cs: list) -> list:
    """Durand-Kerner on the monic polynomial (floats); good enough to isolate."""
    an = cs[-1]
    try:
        mon = [complex(c / an) for c in cs]
    except OverflowError:
        return []
    n = len(cs) - 1
    bound = 1 + max(abs(c) for c in mon[:-1])
    zs = [bound * cmath.exp(2j * math.pi * (i + 0.25) / n) for i in range(n)]

    def f(z):
        acc = 0j
        for c in reversed(mon):
            acc = acc * z + c
        return acc

    for _ in range(500):
        moved = 0.0
        new = []
        for i, z in enumerate(zs):
            den = 1 + 0j
            for j, w in enumerate(zs):
                if i != j:
                    den *= (z - w) or 1e-12
            try:
                nz = z - f(z) / den
            except (OverflowError, ZeroDivisionError):
                return zs
            moved = max(moved, abs(nz - z))
            new.append(nz)
        zs = new
        if moved < 1e-12 * bound:
            break
    return zs


# --- solving ------------------------------------------------------------------

def _linear_coefs(p: Poly) -> Optional[tuple]:
    """``({var: coef}, const)`` if ``p`` is linear, else ``None``."""
    coefs, c = {}, 0
    for m, k in p.items():
        if not m:
            c = k
        elif len(m) == 1 and m[0][1] == 1:
            coefs[m[0][0]] = k
        else:
            return None
    return coefs, c


def _ext_gcd(a: int, b: int) -> tuple:
    """``(g, u, v)`` with ``u*a + v*b == g == gcd(a, b) > 0``."""
    old_r, r, old_s, s, old_t, t = a, b, 1, 0, 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return old_r, old_s, old_t


class _Refuted(Exception):
    pass


class _Solver:
    def __init__(self, fresh):
        self.fresh = fresh
        self.nodes = 0
        self.complete = True

    def eliminate(self, cons: list, subst: dict) -> tuple:
        """Eliminate every linear equality exactly.  Returns ``(cons, subst)``."""
        cons = list(cons)
        while True:
            pick = None
            for i, c in enumerate(cons):
                k = _const_of(c.poly)
                if k is not None:
                    ok = (k == 0) if c.op == "==" else (k != 0) if c.op == "!=" else (k <= 0)
                    if not ok:
                        raise _Refuted()
                    pick = ("drop", i)
                    break
                if c.op == "==" and _linear_coefs(c.poly) is not None:
                    pick = ("lin", i)
                    break
            if pick is None:
                return cons, subst
            kind, i = pick
            if kind == "drop":
                cons.pop(i)
                continue
            coefs, c0 = _linear_coefs(cons[i].poly)
            g = 0
            for a in coefs.values():
                g = math.gcd(g, a)
            if c0 % g:
                raise _Refuted()
            unit = next((v for v, a in sorted(coefs.items()) if abs(a) == g), None)
            if unit is not None:
                a = coefs[unit]
                # a*unit + rest + c0 == 0  ->  unit = -(rest + c0) / a, a == +-g, exact after /g
                rest = {m: k for m, k in cons[i].poly.items() if m != ((unit, 1),)}
                val = {m: (-k) // a for m, k in rest.items()}
                binding = {m: k for m, k in val.items() if k}
                cons.pop(i)
                cons, subst = self._apply(cons, subst, unit, binding)
                continue
            # two smallest coefficients: a x + b y -> g2 w
            (x, a), (y, b) = sorted(coefs.items(), key=lambda kv: (abs(kv[1]), kv[0]))[:2]
            g2, u, v = _ext_gcd(a, b)
            w, t = self.fresh("w"), self.fresh("t")
            xb = _add(_scale(_var(w), u), _scale(_var(t), b // g2))
            yb = _add(_scale(_var(w), v), _scale(_var(t), -(a // g2)))
            cons, subst = self._apply(cons, subst, x, xb)
            cons, subst = self._apply(cons, subst, y, yb)

    @staticmethod
    def _apply(cons, subst, var, poly):
        cons = [_Con(_subst(c.poly, var, poly), c.op) for c in cons]
        subst = {k: _subst(p, var, poly) for k, p in subst.items()}
        subst[var] = poly
        return cons, subst

    def search(self, cons: list, env: dict, depth: int = 0) -> Optional[dict]:
        self.nodes += 1
        if self.nodes > _MAX_NODES or depth > _MAX_DEPTH:
            self.complete = False
            return None
        live = []
        for c in cons:
            p = c.poly
            if _vars(p) & env.keys():
                p = {}
                for m, k in c.poly.items():
                    t, rest = k, []
                    for v, e in m:
                        if v in env:
                            t *= env[v] ** e
                        else:
                            rest.append((v, e))
                    p = _add(p, {tuple(rest): t})
            k = _const_of(p)
            if k is not None:
                ok = (k == 0) if c.op == "==" else (k != 0) if c.op == "!=" else (k <= 0)
                if not ok:
                    return None
                continue
            live.append(_Con(p, c.op))
        if not live:
            return env
        # the variable with the most exact information: in an equality first, then most used
        counts: dict = {}
        for c in live:
            vs = _vars(c.poly)
            weight = (100 if c.op == "==" else 10) if len(vs) == 1 else 1
            for v in vs:
                counts[v] = counts.get(v, 0) + weight
        # most exact information first; on a tie, a program variable before an auxiliary
        # quotient (which the program variables determine)
        var = max(sorted(counts), key=lambda v: (counts[v], not v.startswith(("__q", "__a"))))
        for val in self._candidates(var, live):
            got = self.search(live, {**env, var: val}, depth + 1)
            if got is not None:
                return got
            if self.nodes > _MAX_NODES:
                break
        return None

    def _candidates(self, var: str, live: list) -> Iterable[int]:
        lo, hi = None, None
        exact: Optional[set] = None
        avoid: set = set()
        hints: list = []
        for c in live:
            if _vars(c.poly) != {var}:
                if c.op == "==" and var in _vars(c.poly):
                    hints.extend(_divisor_hints(c.poly))
                continue
            cs = _coeffs(c.poly, var)
            if c.op == "==":
                roots = int_roots(cs)
                if roots is None:
                    continue
                if len(cs) > 3:
                    self.complete = False
                exact = set(roots) if exact is None else exact & set(roots)
            elif c.op == "!=":
                if len(cs) == 2 and cs[0] % cs[1] == 0:
                    avoid.add(-cs[0] // cs[1])
            elif len(cs) == 2:
                b, a = cs                   # a*x + b <= 0
                if a > 0:
                    bound = (-b) // a
                    hi = bound if hi is None else min(hi, bound)
                else:                       # a < 0:  x >= b / -a
                    bound = _ceil_div(b, -a)
                    lo = bound if lo is None else max(lo, bound)
        if exact is not None:
            out = sorted(exact, key=lambda x: (abs(x), x))
            return [x for x in out if (lo is None or x >= lo) and (hi is None or x <= hi)
                    and x not in avoid]
        if lo is not None and hi is not None and lo > hi:
            return []
        self.complete = False
        seeds = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 7, -7, 8, -8, 10, -10, 16, -16] + hints
        if lo is not None:
            seeds += [lo, lo + 1, lo + 2]
        if hi is not None:
            seeds += [hi, hi - 1, hi - 2]
        out, seen = [], set()
        for x in seeds:
            if x in seen or x in avoid:
                continue
            if (lo is not None and x < lo) or (hi is not None and x > hi):
                continue
            seen.add(x)
            out.append(x)
        if not out and lo is not None:
            out = [x for x in range(lo, lo + len(avoid) + 2) if x not in avoid and (hi is None or x <= hi)]
        return out


def _divisor_hints(p: Poly) -> list:
    """Divisors of an equality's constant term (both signs): for ``x * y == c`` the
    values of ``x`` that leave ``y`` an exact root."""
    c = abs(p.get((), 0))
    if not c or c > 10**10:
        return []
    out = []
    for d in range(2, min(math.isqrt(c), _MAX_DIVISOR_TRIAL) + 1):
        if c % d == 0:
            out.extend((d, -d, c // d, -(c // d)))
    out.extend((c, -c))
    return out


def _fm_infeasible(cons: list, max_vars: int = 8, max_rows: int = 256) -> bool:
    """Fourier-Motzkin elimination over the rationals on the LINEAR ``<=`` constraints:
    ``True`` only when they have no real solution -- hence no integer one.  Bounded
    work; ``False`` means "not shown"."""
    from fractions import Fraction
    rows = []
    for c in cons:
        if c.op != "<=":
            continue
        lin = _linear_coefs(c.poly)
        if lin is None:
            continue
        coefs, k = lin
        rows.append(({v: Fraction(a) for v, a in coefs.items() if a}, Fraction(k)))
    names = sorted({v for r, _ in rows for v in r})
    if not rows or len(names) > max_vars:
        return False
    for v in names:
        if any(not r and k > 0 for r, k in rows):
            return True
        pos = [(r, k) for r, k in rows if r.get(v, 0) > 0]
        neg = [(r, k) for r, k in rows if r.get(v, 0) < 0]
        rest = [(r, k) for r, k in rows if not r.get(v, 0)]
        for rp, kp in pos:
            for rn, kn in neg:
                mp, mn = -rn[v], rp[v]                # both positive
                row = {}
                for w in set(rp) | set(rn):
                    a = rp.get(w, 0) * mp + rn.get(w, 0) * mn
                    if a and w != v:
                        row[w] = a
                rest.append((row, kp * mp + kn * mn))
                if len(rest) > max_rows:
                    return False
        rows = rest
    return any(not r and k > 0 for r, k in rows)


def _ceil_div(a: int, b: int) -> int:
    return -((-a) // b)


def _check(expr: ast.expr, witness: dict) -> bool:
    try:
        tree = ast.fix_missing_locations(ast.Expression(body=copy.deepcopy(expr)))
        code = compile(tree, "<guard>", "eval")
        return bool(eval(code, {"__builtins__": {"abs": abs}}, dict(witness)))  # noqa: S307 -- int-only AST
    except Exception:
        return False


def free_names(expr: ast.expr) -> set:
    """Names read by ``expr``, except the ``abs`` it calls (the only call handled)."""
    callee = {id(n.func) for n in ast.walk(expr) if isinstance(n, ast.Call)}
    return {n.id for n in ast.walk(expr) if isinstance(n, ast.Name) and id(n) not in callee}


def solve_int_guard(expr, int_vars: Iterable[str]) -> ArithResult:
    """Find integers for ``int_vars`` that make the guard ``expr`` (an AST or source
    text) true.  See the module docstring for what is handled."""
    if isinstance(expr, str):
        try:
            expr = ast.parse(expr, mode="eval").body
        except SyntaxError:
            return ArithResult("unknown", explanation="syntax")
    names = set(int_vars)
    free = free_names(expr)
    if not free <= names:
        return ArithResult("unknown", explanation=f"non-integer names {sorted(free - names)}")
    tr = _Translator(names)
    try:
        branches = tr.dnf(expr)
    except (_Decline, RecursionError) as exc:
        return ArithResult("unknown", explanation=f"declined: {exc}")
    counter = itertools.count()

    def fresh(stem):
        return f"__{stem}{next(counter)}_"

    all_refuted = True
    for conj in branches:
        s = _Solver(fresh)
        try:
            cons, subst = s.eliminate(conj, {})
        except _Refuted:
            continue
        if _fm_infeasible(cons):
            continue                        # the linear inequalities have no real solution
        all_refuted = False
        env = s.search(cons, {})
        if env is None:
            continue
        witness = {}
        for v in sorted(free):
            p = subst.get(v, _var(v))
            witness[v] = _eval(p, {k: env.get(k, 0) for k in _vars(p)})
        if _check(expr, witness):
            return ArithResult("sat", witness, "linear elimination + bounded assignment")
    if all_refuted:
        return ArithResult("unsat", explanation="every branch refuted by exact linear elimination")
    return ArithResult("unknown", explanation="no witness found")
