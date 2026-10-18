"""The ``math`` module as the engines see it: ONE hand-verified table, read by absint
(``solve/absint/interp.py::Analyzer._math_call``) and the symbolic executor
(``sv/_symexec.py``, through ``adapters/concolic.py``) so both reason the same way about a call
(2026-10-05 owner rule: a contract is consumed by at least absint AND the symbolic engine).

The contracts (``solve/contracts/rules_math.py``) say WHICH property classes a call can raise; this
table adds what an engine needs to act on them for the one call shape it understands: the
accepted argument count and keywords, the kind each argument needs, the kind of the result, and
the closed range of a float result.  Every entry is pinned against CPython 3.14.4 by
``tests/contracts/test_math_module.py`` (arity, keywords, kinds, ranges, the classes an
exact-builtin-number call can raise), and an entry is usable only while the registry still holds a
TESTED contract for the name (:func:`math_function`): withdrawing the contract withdraws the model.

Argument kinds: ``real`` is an exact builtin ``int``/``bool``/``float`` (an engine never passes
a user object, whose ``__float__``/``__index__`` could run any code), ``int`` an exact ``int``/
``bool`` (a float, even ``2.0``, is a ``TypeError``), ``intopt`` an ``int``/``bool`` or ``None``
(``perm``'s ``k``), ``iterable`` a builtin ``list``/``tuple``.
Result kinds: ``float``, ``bool``, ``int``, ``float,int`` and ``float,float`` (pairs),
``number`` (``prod`` / ``sumprod`` over ints and floats).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from tharkun.solve.contracts.adapters.common import resolve

_INF = math.inf
#: pi and pi/2 rounded UP, so a range built from them contains every libm result
_PI_UP = 3.1415926535897936
_HALF_PI_UP = 1.5707963267948968


@dataclass(frozen=True)
class MathFn:
    name: str
    args: tuple[str, ...]            # the kind of each REQUIRED-OR-OPTIONAL positional argument
    min_args: int
    max_args: int | None             # None: any number of further arguments of kind ``args[-1]``
    result: str
    keywords: frozenset = frozenset()    # keyword names the real function accepts
    float_range: tuple | None = None     # closed range of a float result (a nan argument gives nan)
    int_min: int | None = None           # lower bound of an int result
    nan_from_numbers: bool = False       # can a float result be nan although no argument is nan?

    def shape(self, arg_count: int, keywords) -> str:
        """How a call with this many positional arguments and these keyword names relates to the
        verified shape: ``"ok"``; ``"typeerror"`` -- CPython raises TypeError before anything else
        runs (a wrong argument count, or a keyword the function does not take: most take none at
        all); ``"unmodelled"`` -- a keyword the function really takes (``isclose(rel_tol=...)``),
        whose binding the model does not follow."""
        keywords = set(keywords)
        if keywords - self.keywords:
            return "typeerror"
        if keywords:
            return "unmodelled"      # a keyword may fill a positional (isclose(1, b=2) is valid)
        if arg_count < self.min_args or (self.max_args is not None and arg_count > self.max_args):
            return "typeerror"
        return "ok"

    def kind_of(self, index: int) -> str:
        return self.args[min(index, len(self.args) - 1)]


def _unary(name, result="float", rng=None, int_min=None):
    return MathFn(name, ("real",), 1, 1, result, float_range=rng, int_min=int_min)


def _binary(name, result="float", rng=None, kinds=("real", "real")):
    return MathFn(name, kinds, 2, 2, result, float_range=rng)


_MODEL: dict[str, MathFn] = {fn.name: fn for fn in (
    _unary("acos", rng=(0.0, _PI_UP)),
    _unary("acosh", rng=(0.0, _INF)),
    _unary("asin", rng=(-_HALF_PI_UP, _HALF_PI_UP)),
    _unary("asinh"),
    _unary("atan", rng=(-_HALF_PI_UP, _HALF_PI_UP)),
    _unary("atanh"),
    _unary("cbrt"),
    _unary("cos", rng=(-1.0, 1.0)),
    _unary("cosh", rng=(1.0, _INF)),
    _unary("degrees"),
    _unary("erf", rng=(-1.0, 1.0)),
    _unary("erfc", rng=(0.0, 2.0)),
    _unary("exp", rng=(0.0, _INF)),
    _unary("exp2", rng=(0.0, _INF)),
    _unary("expm1", rng=(-1.0, _INF)),
    _unary("fabs", rng=(0.0, _INF)),
    _unary("gamma"),
    _unary("lgamma"),
    _unary("log10"),
    _unary("log1p"),
    _unary("log2"),
    _unary("radians"),
    _unary("sin", rng=(-1.0, 1.0)),
    _unary("sinh"),
    _unary("sqrt", rng=(0.0, _INF)),
    _unary("tan"),
    _unary("tanh", rng=(-1.0, 1.0)),
    _unary("ulp", rng=(5e-324, _INF)),
    _unary("isnan", "bool"),
    _unary("isinf", "bool"),
    _unary("isfinite", "bool"),
    _unary("floor", "int"),
    _unary("ceil", "int"),
    _unary("trunc", "int"),
    _unary("frexp", "float,int"),
    _unary("modf", "float,float"),
    _binary("atan2", rng=(-_PI_UP, _PI_UP)),
    _binary("copysign"),
    _binary("fmod"),
    _binary("remainder"),
    _binary("pow"),
    _binary("ldexp", kinds=("real", "int")),
    MathFn("fma", ("real", "real", "real"), 3, 3, "float"),
    MathFn("hypot", ("real",), 0, None, "float", float_range=(0.0, _INF)),
    MathFn("log", ("real", "real"), 1, 2, "float", nan_from_numbers=True),   # log(inf, inf) is nan
    MathFn("isclose", ("real", "real"), 2, 2, "bool", keywords=frozenset({"a", "b", "rel_tol", "abs_tol"})),
    MathFn("nextafter", ("real", "real"), 2, 2, "float", keywords=frozenset({"steps"})),
    MathFn("factorial", ("int",), 1, 1, "int", int_min=1),
    MathFn("gcd", ("int",), 0, None, "int", int_min=0),
    MathFn("lcm", ("int",), 0, None, "int", int_min=0),
    MathFn("isqrt", ("int",), 1, 1, "int", int_min=0),
    MathFn("comb", ("int", "int"), 2, 2, "int", int_min=0),
    MathFn("perm", ("int", "intopt"), 1, 2, "int", int_min=0),
    MathFn("fsum", ("iterable",), 1, 1, "float"),
    MathFn("prod", ("iterable",), 1, 1, "number", keywords=frozenset({"start"}), nan_from_numbers=True),
    MathFn("dist", ("iterable", "iterable"), 2, 2, "float", float_range=(0.0, _INF), nan_from_numbers=True),
    MathFn("sumprod", ("iterable", "iterable"), 2, 2, "number", nan_from_numbers=True),
)}

#: the constants, by name: reading one raises nothing and runs nothing
CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf, "nan": math.nan}

#: the property classes an exact-builtin-number call of the VERIFIED shape (right argument count and
#: kinds) can raise natively: none, except ``math.log(x, base)`` with ``base == 1``. ``ValueError``
#: and ``OverflowError`` are not property classes, ``FloatingPointError`` never happens. A claim about
#: ONE shape, like the tables in ``common.py``: it says nothing about a user object argument.
SCALAR_SHAPE_RAISES: dict[str, frozenset] = {"log": frozenset({"ZeroDivisionError"})}


def math_function(name: str) -> MathFn | None:
    """The model of ``math.<name>``, or ``None`` when there is none or the registry no longer
    holds a TESTED contract for it."""
    fn = _MODEL.get(name)
    if fn is None:
        return None
    contract = resolve(f"math.{name}")
    return fn if contract is not None and contract.tested else None


def modelled_names() -> frozenset:
    """Every ``math`` function with a model (whether or not its contract is still tested)."""
    return frozenset(_MODEL)


def scalar_shape_raises(name: str) -> frozenset:
    """The property classes an exact-builtin-number call of ``math.<name>`` can raise, given
    the right argument count and kinds (:data:`SCALAR_SHAPE_RAISES`)."""
    return SCALAR_SHAPE_RAISES.get(name, frozenset())


def constant(name: str) -> float | None:
    """The value of ``math.<name>`` when it is a constant with a TESTED contract."""
    if name not in CONSTANTS:
        return None
    contract = resolve(f"math.{name}")
    return CONSTANTS[name] if contract is not None and contract.tested else None


__all__ = ["MathFn", "CONSTANTS", "SCALAR_SHAPE_RAISES", "math_function", "modelled_names",
           "scalar_shape_raises", "constant"]
