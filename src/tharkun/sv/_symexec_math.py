"""The ``math`` module in the symbolic executor ([internal ref] batch 5).

``sv/_symexec.py`` evaluated no call into ``math`` at all: every ``math.sqrt(x)`` made the whole
statement, and everything it assigned, unmodelled.  This module is its consumer of the ``math``
contracts (``solve/contracts/rules_math.py``) and of the table every engine shares
(``solve/contracts/adapters/math_model.py``, also read by absint, here through ``adapters/concolic.py``): a call is

* **folded natively** when every argument is a concrete builtin number (or a list/tuple of them):
  ``math`` is a pure function of its arguments, so the executor just runs it.  A call that raises
  (``sqrt(-1)``, ``exp(710)``, a wrong argument count) is REFUSED with the exception named, not
  silently dropped: the executor does not decide what the rest of a path that ends here means;
* **encoded in z3** when it is exact: ``floor``/``ceil``/``trunc``/``fabs``/``copysign`` on ints and
  reals, ``fmod``/``remainder``, ``fma``, ``isclose`` (default tolerances), ``modf``, ``pow`` with
  a small concrete integer exponent, ``ldexp`` with a concrete shift, ``isqrt`` and ``gcd`` where a
  definition exists, ``fsum``/``prod``/``sumprod`` over a list with symbolic elements, and the
  predicates on ints.  "Exact" is relative to the executor's own model: a float draw is a z3
  ``Real``, with no infinity, nan, signed zero or rounding (every candidate is replayed on the
  real program before any verdict, ``sv/concolic.py``);
* **refused with a reason** otherwise (:data:`REFUSED`): a transcendental or irrational function of a
  symbolic value has no exact encoding, and the refusal is the ``unsupported`` reason the engine
  reports for the sites behind it.

Partial functions of a symbolic argument (``fmod`` by zero, ``isqrt`` of a negative) end the path
exactly as CPython's ``ValueError`` does: the argument's domain becomes a path condition, which
only the whole-harness executor (``_Runner._holds``) has.
"""

from __future__ import annotations

import fractions
import math

import z3

from tharkun.solve.contracts.adapters import concolic as _contracts
from tharkun.sv._symexec import SymList, Unsupported, _is_sym, _num

#: why a symbolic argument has no exact encoding, per function: the reason the engine reports
REFUSED = {
    **{n: "transcendental: no exact real encoding" for n in (
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "exp", "exp2", "expm1", "log", "log10", "log2", "log1p", "gamma", "lgamma", "erf", "erfc")},
    **{n: "algebraic irrational: a square or cube root has no exact encoding in a rational model" for n in (
        "sqrt", "cbrt", "hypot", "dist")},
    **{n: "a multiple of an irrational constant" for n in ("degrees", "radians")},
    **{n: "IEEE-754 bit-level: not representable over the reals" for n in ("ulp", "nextafter", "frexp")},
    **{n: "an integer function with no linear or polynomial encoding" for n in ("factorial", "comb", "perm", "lcm")},
}

#: the functions :func:`_encode` has a z3 encoding for (``gcd`` only with at most one unknown argument and
#: the others 0); together with :data:`REFUSED` they are every function of the module, and no other
ENCODED = frozenset({
    "floor", "ceil", "trunc", "fabs", "isnan", "isinf", "isfinite", "copysign", "fmod", "remainder", "fma",
    "isclose", "ldexp", "pow", "modf", "isqrt", "gcd", "fsum", "prod", "sumprod",
})

#: arguments larger than this are not folded: factorial/comb/perm cost grows with them
_FOLD_LIMIT = 5000
_SHIFT_LIMIT = 1000
_POW_LIMIT = 8


def _symbolic(v) -> bool:
    if isinstance(v, (list, tuple)):
        return any(_symbolic(x) for x in v)
    return _is_sym(v) or isinstance(v, SymList)


def _is_number(v) -> bool:
    return type(v) in (int, float, bool)


def call(exe, name: str, n):
    """The value of the call node ``n`` to ``math.<name>``, or ``Unsupported`` (with the reason)."""
    model = _contracts.math_function(name)
    if model is None:
        raise Unsupported(f"math.{name} has no tested contract")
    shape = model.shape(len(n.args), [k.arg for k in n.keywords])
    if shape == "unmodelled":
        raise Unsupported(f"math.{name} with keyword arguments")
    if shape == "typeerror":
        raise Unsupported(f"math.{name}: CPython raises TypeError for this call shape")
    args = [exe._seq(exe.ev(a)) for a in n.args]
    if not any(_symbolic(a) for a in args):
        return _fold(name, model, args)
    return _encode(exe, name, model, args)


# --------------------------------------------------------------------------- concrete
def _fold(name: str, model, args):
    for i, a in enumerate(args):
        kind = model.kind_of(i)
        if kind == "iterable":
            ok = type(a) in (list, tuple) and all(_is_number(x) for x in a)
        elif kind == "real":
            ok = _is_number(a)
        elif kind == "int":
            ok = type(a) in (int, bool)
        else:   # intopt
            ok = type(a) in (int, bool) or a is None
        if not ok:
            raise Unsupported(f"math.{name} of an argument that is not an exact builtin number")
        if name in ("factorial", "comb", "perm") and type(a) in (int, bool) and abs(a) > _FOLD_LIMIT:
            raise Unsupported(f"math.{name} of {a}: too large to fold")
    try:
        return getattr(math, name)(*args)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as e:
        raise Unsupported(f"math.{name} raises {type(e).__name__} on these arguments") from None


# --------------------------------------------------------------------------- symbolic
def _is_int(t) -> bool:
    return isinstance(t, int) or (_is_sym(t) and t.sort() == z3.IntSort())


def _term(v):
    """A z3-compatible arithmetic term: ints stay ints, a finite float becomes its exact rational."""
    v = _num(v)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise Unsupported("an infinite or nan constant has no real encoding")
        return z3.RealVal(fractions.Fraction(v))
    return v


def _real(t):
    if _is_sym(t):
        return z3.ToReal(t) if t.sort() == z3.IntSort() else t
    return z3.RealVal(t)


def _abs(t):
    return z3.If(t >= 0, t, -t)


def _trunc(t):
    return t if _is_int(t) else z3.If(t >= 0, z3.ToInt(t), -z3.ToInt(-t))


def _require(exe, cond, name):
    """The rest of the path runs only when ``cond`` holds (CPython raises ValueError otherwise)."""
    hold = getattr(exe, "_holds", None)
    if hold is None:
        raise Unsupported(f"math.{name} of a symbolic value ends the path on a domain error: "
                          "only the whole-harness executor tracks path conditions")
    hold(cond)


def _fresh_int(exe, name):
    seq = getattr(exe, "_contract_var_name", None)
    if seq is None:
        raise Unsupported(f"math.{name} needs a fresh variable: only the whole-harness executor has them")
    return z3.Int(seq(f"math.{name}"))


def _reals(args):
    out = []
    for a in args:
        if not (_is_number(a) or _is_sym(a)):
            raise Unsupported("an argument that is not a number")
        out.append(_term(a))
    return out


def _encode(exe, name: str, model, args):
    reason = REFUSED.get(name)
    if reason is not None or name not in ENCODED:
        raise Unsupported(f"math.{name} of a symbolic value has no exact encoding ({reason or 'not encoded'})")
    if name in ("floor", "ceil", "trunc"):
        x, = _reals(args)
        if _is_int(x):
            return x
        if name == "floor":
            return z3.ToInt(x)
        if name == "ceil":
            return -z3.ToInt(-x)
        return _trunc(x)
    if name == "fabs":
        x, = _reals(args)
        return _real(_abs(x))
    if name in ("isnan", "isinf", "isfinite"):
        x, = _reals(args)
        if not _is_int(x):
            raise Unsupported(f"math.{name} of a symbolic float: a z3 Real has no nan or infinity")
        return name == "isfinite"
    if name == "copysign":
        x, y = _reals(args)
        if not _is_int(y):
            raise Unsupported("math.copysign of a symbolic float sign: a z3 Real has no signed zero")
        mag = _abs(x)
        return _real(z3.If(y >= 0, mag, -mag))
    if name in ("fmod", "remainder"):
        x, y = _reals(args)
        _require(exe, y != 0, name)
        q = _real(x) / _real(y)
        if name == "fmod":
            t = _trunc(q)
        else:   # round half to even
            f = z3.ToInt(q)
            d = q - z3.ToReal(f)
            t = z3.If(d > z3.RealVal(fractions.Fraction(1, 2)), f + 1,
                      z3.If(d < z3.RealVal(fractions.Fraction(1, 2)), f, z3.If(f % 2 == 0, f, f + 1)))
        return _real(x) - _real(y) * z3.ToReal(t)
    if name == "fma":
        x, y, z = _reals(args)
        return _real(x) * _real(y) + _real(z)
    if name == "isclose":
        a, b = _reals(args)
        a, b = _real(a), _real(b)
        rel = z3.RealVal(fractions.Fraction(1e-09))
        diff, big = _abs(a - b), z3.If(_abs(a) >= _abs(b), _abs(a), _abs(b))
        return z3.Or(a == b, diff <= rel * big)
    if name == "ldexp":
        x, i = args
        if _symbolic(i) or type(i) not in (int, bool) or abs(i) > _SHIFT_LIMIT:
            raise Unsupported("math.ldexp with a symbolic or huge shift")
        return _real(_reals([x])[0]) * z3.RealVal(fractions.Fraction(2) ** int(i))
    if name == "pow":
        x, y = args
        if _symbolic(y) or not _is_number(y) or float(y) != int(y) or abs(y) > _POW_LIMIT:
            raise Unsupported("math.pow with a symbolic, fractional or large exponent")
        (base,) = _reals([x])
        base, k = _real(base), int(y)
        if k < 0:
            _require(exe, base != 0, name)
        out = z3.RealVal(1)
        for _ in range(abs(k)):
            out = out * base
        return out if k >= 0 else 1 / out
    if name == "modf":
        x, = _reals(args)
        if _is_int(x):
            return (z3.RealVal(0), _real(x))
        t = _trunc(x)
        return (x - z3.ToReal(t), z3.ToReal(t))
    if name == "isqrt":
        n_, = _reals(args)
        if not _is_int(n_):
            raise Unsupported("math.isqrt of a float: TypeError")
        r = _fresh_int(exe, name)
        _require(exe, z3.And(n_ >= 0, r >= 0, r * r <= n_, n_ < (r + 1) * (r + 1)), name)
        return r
    if name == "gcd":
        symbolic = [a for a in args if _symbolic(a)]
        rest = [a for a in args if not _symbolic(a)]
        if len(symbolic) == 1 and all(type(a) in (int, bool) and a == 0 for a in rest):
            x = _term(symbolic[0])
            if _is_int(x):
                return _abs(x)
        raise Unsupported("math.gcd of two unknown values has no linear encoding")
    if name in ("fsum", "prod", "sumprod"):
        return _encode_iterable(name, args)
    raise Unsupported(f"math.{name} of a symbolic value is not encoded")


def _encode_iterable(name, args):
    if name == "sumprod":
        p, q = args
        if not isinstance(p, (list, tuple)) or not isinstance(q, (list, tuple)):
            raise Unsupported("math.sumprod of a symbolic-length sequence")
        if len(p) != len(q):
            raise Unsupported("math.sumprod raises ValueError: the inputs differ in length")
        total = 0
        for a, b in zip(p, q):
            total = total + _term(a) * _term(b)
        return total
    seq, = args
    if not isinstance(seq, (list, tuple)):
        raise Unsupported(f"math.{name} of a symbolic-length sequence")
    terms = _reals(list(seq))
    if name == "fsum":
        total = z3.RealVal(0)
        for t in terms:
            total = total + _real(t)
        return total
    out = 1
    for t in terms:
        out = out * t
    return out
