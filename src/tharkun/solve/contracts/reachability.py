"""Audited call reachability for closed programs on CPython 3.14.4.

A present row has TWO claims: on an exact native receiver (for methods),
calling it introduces no values outside builtin values, its inputs, and values
returned by their Python hooks; and it invokes no ordinary user attributes
except the listed names. All dunder hooks remain possible. This is NOT an
exception, purity, or determinism contract. Missing rows mean unknown.

The caller must establish that receivers cannot be external classes or builtin
subclasses before using these rows. In particular, a method name alone cannot
establish the native receiver contract.

Evidence: CPython v3.14.4 Objects/{list,dict,set,bytes,bytearray,unicode}object.c
(native collection methods), Objects/{abstract,float,long,range}object.c and
Python/bltinmodule.c (constructors and numeric/representation protocols),
Modules/mathmodule.c, Modules/_randommodule.c and Lib/random.py.
https://github.com/python/cpython/tree/v3.14.4

The deliberately small surface is exercised in tests/test_pyresolve_precision.py.
Extend it only with an audited callback AND returned-value provenance claim.
"""
from __future__ import annotations

#: every function and constant of ``math`` ([internal ref] batch 5): none looks up an ordinary attribute of its argument
#: (``__float__``, ``__index__``, ``__trunc__``, ``__iter__``, ``__mul__`` ... are dunders, which stay live) and
#: what comes back is a builtin number, a tuple of them, or -- for ``prod``/``sumprod`` -- a value a dunder
#: returned. Swept against a hostile argument that logs every ordinary attribute lookup:
#: ``tests/contracts/test_engine_audit_consumed.py::test_a_closed_hooks_row_holds_against_a_hostile_argument``.
_MATH = (
    "acos", "acosh", "asin", "asinh", "atan", "atan2", "atanh", "cbrt", "ceil", "comb", "copysign", "cos", "cosh",
    "degrees", "dist", "erf", "erfc", "exp", "exp2", "expm1", "fabs", "factorial", "floor", "fma", "fmod", "frexp",
    "fsum", "gamma", "gcd", "hypot", "isclose", "isfinite", "isinf", "isnan", "isqrt", "lcm", "ldexp", "lgamma",
    "log", "log10", "log1p", "log2", "modf", "nextafter", "perm", "pow", "prod", "radians", "remainder", "sin",
    "sinh", "sqrt", "sumprod", "tan", "tanh", "trunc", "ulp",
    "pi", "e", "tau", "inf", "nan",
)

# No ordinary callbacks; dunders, including iter/hash/eq/index/float/repr, remain live.
_CLOSED = frozenset({
    "abs", "divmod", "float", "int", "len", "list", "range", "repr", "set",
    *(f"math.{name}" for name in _MATH),
    "random.random", "random.randrange",
    "_sv_verifier.nondet_int",  # harness primitive: returns an exact Python int
    "list.append", "list.clear", "list.copy", "list.pop", "list.remove",
    "dict.clear", "dict.copy", "dict.pop",
    "set.add", "set.clear", "set.copy", "set.pop", "set.remove", "set.update",
    "frozenset.copy",
    "str.join", "str.find", "bytes.join", "bytes.find",
    "bytearray.append", "bytearray.clear", "bytearray.copy", "bytearray.pop",
    "bytearray.remove", "bytearray.join", "bytearray.find",
})


def closed_call_hooks(qualname: str) -> frozenset[str] | None:
    """The census visitor's audited hooks, or unknown."""
    from tharkun.solve.contracts.rules import Refused, get_rule
    from tharkun.solve.contracts.visitors import CensusVisitor

    rule = get_rule(qualname)
    if rule is None:
        # Reachability also audits native calls which have no exception contract.
        return _closed_call_hooks_model(qualname)
    out = rule.accept(CensusVisitor())
    return None if isinstance(out, Refused) else out


def _closed_call_hooks_model(qualname: str) -> frozenset[str] | None:
    """Complete ordinary hooks AND closed return provenance, or unknown (None)."""
    if qualname in _CLOSED:
        return frozenset()
    if qualname == "random.seed":
        # isinstance observes user __class__; do not assume its str/bytes branch
        # has an exact native argument when it calls encode/decode (Lib/random.py).
        return frozenset({"encode", "decode"})
    if qualname in {"dict", "dict.update"}:
        return frozenset({"keys"})
    return None
