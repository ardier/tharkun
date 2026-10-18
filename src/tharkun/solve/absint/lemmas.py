"""Named numeric lemmas the absint float domain applies (docs/ABSINT.md).

Each lemma is a pure function from *value terms* (and intervals) to a bound,
and every proof fact that uses one names it (``lemma TRIG_PAIR``, ...), so an
auditor can check the lemma once and every proof that cites it.

Value terms are hashable tuples describing how a float value was computed:

* ``('trig', f, key, k)``   -- ``fl(k * f(x))`` with ``f`` in ``{'sin', 'cos'}``,
  ``k`` a finite constant, and ``key`` the identity of the argument ``x``
  (``('iv', zone_var)``: the current value of an int variable);
* ``('rd', addr, attr, ver)`` -- the value of attribute ``attr`` of the single
  heap object ``addr`` at attribute version ``ver``;
* ``('sumsq', bases)``      -- ``fl(sum of fl(b * b) for b in bases)``, bases a sorted tuple;
* ``('sqrt', s, lo, hi)``   -- ``math.sqrt`` of the value ``s``, whose interval was ``[lo, hi]``;
* ``('div', a, b)``         -- ``fl(a / b)``.

Two values with the same term are the same float: the interpreter drops a term
as soon as anything it names can change (an int variable is reassigned, an
object is refreshed, focused, blurred or demoted).

Rounding: every bound below keeps a margin far larger than the accumulated
rounding error of the few operations involved (at most ~10 ulp relative).
"""

from __future__ import annotations

import math

EPS = 2.0 ** -52

#: TRIG_PAIR: for every finite float x, max(|sin x|, |cos x|) >= sqrt(2)/2 - 1e-9.
#: Mathematically max(|sin|, |cos|) >= sqrt(2)/2 for every real; the libm results
#: (glibc, macOS libm) are within 1 ulp (< 2.3e-16 on [-1, 1]) of the true values
#: for every finite argument (full argument reduction), far inside the 1e-9 margin.
#: Hence a*sin(x)^2 + b*cos(x)^2 >= min(a, b) * (sqrt(2)/2 - 1e-9)^2 >= 0.4999 * min(a, b).
TRIG_PAIR_C = 0.4999

#: NORMALIZE: for finite floats a_1..a_m (m <= 8), S = fl(sum fl(a_i^2)) with
#: 2^-500 <= S <= 2^500 and N = fl(sqrt(S)): sum over i of fl(a_i / N)^2 lies in
#: [1 - 1e-12, 1 + 1e-12] (real arithmetic on the float results).  Relative error:
#: squares and sum <= (m + 1) ulp, sqrt 1 ulp, quotient 1 ulp, squared again 2 ulp:
#: under 30 ulp ~ 7e-15; a quotient that underflows contributes < 2^-1000 absolute.
NORMALIZE_TOL = 1e-12
NORMALIZE_S_LO = 2.0 ** -500
NORMALIZE_S_HI = 2.0 ** 500

#: Widening thresholds for object invariants: every bound a lemma has produced in
#: this analysis (plus 0 and inf).  Widening an invariant bound jumps to the next
#: landmark instead of straight to 0 / inf, so a lemma's bound survives a loop
#: whose first iteration was more precise (e.g. i = 0).  Finite, so termination holds.
LANDMARKS_LO: set = {0.0}
LANDMARKS_HI: set = {math.inf}


LANDMARK_CAP = 64


def add_landmarks(lo: float, hi: float) -> None:
    """Record an invariant bound as a widening threshold (bounded set: termination)."""
    if len(LANDMARKS_LO) < LANDMARK_CAP:
        LANDMARKS_LO.add(lo)
    if len(LANDMARKS_HI) < LANDMARK_CAP:
        LANDMARKS_HI.add(hi)


def reset_landmarks() -> None:
    LANDMARKS_LO.clear()
    LANDMARKS_LO.add(0.0)
    LANDMARKS_HI.clear()
    LANDMARKS_HI.add(math.inf)


def landmark_key() -> tuple[frozenset, frozenset]:
    """The landmark sets as they are now.  They are module state that grows during an analysis
    and steers float widening, so a loop's fixpoint is a function of them
    (``interp.py::Analyzer._fixpoint`` keys its reuse on this)."""
    return frozenset(LANDMARKS_LO), frozenset(LANDMARKS_HI)


def merge_landmarks(key: tuple[frozenset, frozenset]) -> None:
    """Add the landmarks of ``landmark_key()`` taken after a computation: what re-running it would have added."""
    LANDMARKS_LO.update(key[0])
    LANDMARKS_HI.update(key[1])


def widen_lo(old: float, new: float) -> float:
    if new >= old:
        return old
    return max(x for x in LANDMARKS_LO if x <= new)


def widen_hi(old: float, new: float) -> float:
    if new <= old:
        return old
    return min(x for x in LANDMARKS_HI if x >= new)


def sq_interval(lo: float, hi: float) -> tuple[float, float]:
    """Real bounds of x*x for x in [lo, hi] (hi rounded up)."""
    if lo <= 0.0 <= hi:
        mn = 0.0
    else:
        m = min(abs(lo), abs(hi))
        mn = m * m * (1 - 4 * EPS)
    mx = max(abs(lo), abs(hi))
    mx = math.inf if math.isinf(mx) else math.nextafter(mx * mx * (1 + 4 * EPS), math.inf)
    return mn, mx


def trig_pair_lower(terms) -> tuple[float, str] | None:
    """Lower bound of sum t^2 over ``terms`` from TRIG_PAIR, or None.

    ``terms`` are the value terms of the summands' bases; only ``('trig', ...)``
    terms contribute, grouped by argument key."""
    by_key: dict = {}
    for t in terms:
        if isinstance(t, tuple) and t and t[0] == "trig":
            _, f, key, k = t
            d = by_key.setdefault(key, {})
            d[f] = d.get(f, 0.0) + k * k
    best = 0.0
    used = []
    for key, d in by_key.items():
        if d.get("sin", 0.0) > 0 and d.get("cos", 0.0) > 0:
            best += min(d["sin"], d["cos"]) * TRIG_PAIR_C
            used.append(key)
    if not used:
        return None
    n = len(terms)
    lo = best * (1 - 8 * (n + 1) * EPS)
    LANDMARKS_LO.add(lo)
    return lo, f"lemma TRIG_PAIR over sin/cos of {len(used)} argument(s)"


def normalize_bounds(terms) -> tuple[float, float, str] | None:
    """NORMALIZE: the attribute terms are ``a_i / N`` with ``N = sqrt(sum a_i^2)``."""
    if not terms or len(terms) > 8:
        return None
    ns = set()
    num = []
    for t in terms:
        if not (isinstance(t, tuple) and len(t) == 3 and t[0] == "div"):
            return None
        num.append(t[1])
        ns.add(t[2])
    if len(ns) != 1:
        return None
    n = next(iter(ns))
    if not (isinstance(n, tuple) and len(n) == 4 and n[0] == "sqrt"):
        return None
    s, slo, shi = n[1], n[2], n[3]
    if not (isinstance(s, tuple) and len(s) == 2 and s[0] == "sumsq"):
        return None
    if sorted(map(repr, s[1])) != sorted(map(repr, num)):
        return None
    if not (slo >= NORMALIZE_S_LO and shi <= NORMALIZE_S_HI):
        return None  # the precondition is a checked fact: no underflow / overflow
    LANDMARKS_LO.add(1 - NORMALIZE_TOL)
    LANDMARKS_HI.add(1 + NORMALIZE_TOL)
    return (1 - NORMALIZE_TOL, 1 + NORMALIZE_TOL,
            f"lemma NORMALIZE (precondition checked: S in [{slo:.4g}, {shi:.4g}])")
