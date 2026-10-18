"""The rules of the whole ``math`` module ([internal ref], [internal ref]): every function CPython 3.14 defines, and the
constants, as two :class:`~tharkun.solve.contracts.rules.Rule` classes (``MathFunction``: the 57 functions, one
visitor kind, ``math_function``; ``MathConstant``: the 5 constants, ``math_constant``).  The ``Contract`` of each
qualname is exactly the one this module held as plain data before the migration (``tests/contracts/test_registry_digest.py``
proves it byte-identical); ``register_rule`` puts it in the registry, so ``get("math.sqrt")`` and every adapter
(``adapters/math_model.py`` included) work unchanged.

Batch 1 wrote twelve of them; the owner's rule ("strive to support all of Python: when a missing
contract unblocks progress, add all its siblings at once") completes the module.  ``bm_raytrace``
needs ``math.tan``, ``bm_chaos``/``bm_scimark`` use ``sin``/``cos``/``sqrt``/``log``/``pi``.

Cross-checked against Mopsa's ``share/mopsa/stubs/python/math.py`` (LGPL-3.0, see
``docs/contracts/THIRD_PARTY_REFERENCES.md``): its ``sin``/``cos`` stubs raise a plain
``TypeError`` for a non-``int``/``float`` argument, matching the "must be real number" family of
TypeErrors tested here for every function in this module.

What every function shares, all measured on CPython 3.14.4 (``tests/contracts/test_math_.py``,
``tests/contracts/test_math_module.py``):

* **Argument conversion.**  A real-number argument is anything with ``__float__`` or ``__index__``
  (an ``int``, ``bool``, ``float``, ``Fraction``, a user class with either): ``__float__`` is
  tried first, an exception it raises propagates unchanged, and a ``__float__`` that returns a
  non-float is a ``TypeError``.  The integer functions (``factorial``, ``gcd``, ``lcm``,
  ``isqrt``, ``comb``, ``perm``) need ``__index__`` ONLY: a whole-valued
  ``float`` like ``5.0`` is a ``TypeError`` (older Pythons accepted it with a deprecation
  warning).  ``floor``/``ceil`` ask the argument's ``__floor__``/``__ceil__`` first, ``trunc`` asks
  ``__trunc__`` and nothing else (a ``__float__`` alone is not enough).
* **Call shape.**  Every function except ``log`` takes its arguments positionally only: a keyword
  is a ``TypeError``, so is a wrong argument count.  The exceptions are ``isclose`` (``a``, ``b``
  and the keyword-only ``rel_tol``/``abs_tol``), ``nextafter`` (``steps=``) and ``prod``
  (``start=``).
* **What a number argument can raise.**  For a builtin ``int``/``float``/``bool`` argument
  the ONLY property class any function here raises is ``TypeError`` (a wrong argument kind), with one
  exception: ``math.log(x, base)`` raises ``ZeroDivisionError`` when ``base`` is exactly 1.  Domain
  errors (``sqrt(-1)``, ``log(0)``, ``acos(2)``, ``fmod(1, 0)``, ``tan(inf)``, ...) are
  ``ValueError`` and overflows (``exp(710)``, an ``int`` too large for a float, ...) are
  ``OverflowError``: neither is a property class, which is why ``may_raise_other`` is True.
  ``FloatingPointError`` is NEVER raised (``fpectl`` was removed; ``solve/census.py``'s own
  docstring notes this).  Every other class reaches the caller only through a user object's
  ``__float__``/``__index__``/``__iter__`` and the like, which is what the ``propagates uncaught``
  claims say; ``adapters/common.py::SCALAR_NATIVE_RAISES`` is the hand-verified table of the
  classes an exact-builtin-number call can raise, and the tests sweep it.
* **No callbacks for a builtin number.**  ``may_execute_callback`` is True because a user object
  may be passed; with exact ``int``/``float``/``bool`` arguments no user code runs.
"""

from __future__ import annotations

import math

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, Shape, register_rule
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/math.html"
_REAL = "the argument's __float__ (tried first) or __index__"
_INDEX = "the argument's __index__ ONLY (a float is rejected, whole-valued or not)"

_UNARY_TYPEERROR = (
    "the call is not exactly one positional argument (a keyword is rejected too), or the argument "
    "has neither __float__ nor __index__ (None, a str, a list, a complex...), or its __float__ "
    "returns a non-float"
)


def _contract(qualname, anchor, returns, *, dunder, type_error, extra_native=None, notes="", source_extra=()):
    native = {"TypeError": type_error}
    if extra_native:
        native.update(extra_native)
    return Contract(
        qualname=qualname,
        raises=dunder_propagation_claims(dunder, native=native, tested_native=frozenset(native)),
        may_raise_other=True,   # ValueError (domain), OverflowError (range / int too large for a float), a user dunder's own
        returns=returns,
        effects=Effects(deterministic=True),
        notes=notes,
        source=(f"{_DOCS}{anchor}", "CPython Modules/mathmodule.c", *source_extra),
        tested=True,
    )


def _real(qualname, shape="", ret="float", *, notes="", type_error=_UNARY_TYPEERROR, extra_native=None):
    return _contract(qualname, f"#math.{qualname.split('.')[1]}", ReturnSpec(ret, shape, tested=True),
                     dunder=_REAL, type_error=type_error, notes=notes, extra_native=extra_native)


_PI = "pi"

#: the single-argument float functions: name -> (shape of the result, the non-property exceptions)
_UNARY_FLOAT = {
    "acos": ("in [0.0, pi]; nan for nan", "ValueError when |x| > 1"),
    "acosh": ("in [0.0, inf]; nan for nan", "ValueError when x < 1"),
    "asin": ("in [-pi/2, pi/2]; nan for nan", "ValueError when |x| > 1"),
    "asinh": ("", "OverflowError only for an int too large for a float"),
    "atan": ("in [-pi/2, pi/2]; nan for nan", "OverflowError only for an int too large for a float"),
    "atanh": ("", "ValueError when |x| >= 1"),
    "cbrt": ("", "OverflowError only for an int too large for a float"),
    "cos": ("in [-1.0, 1.0]; nan for nan", "ValueError for an infinity"),
    "cosh": ("in [1.0, inf]; nan for nan", "OverflowError when the result exceeds the float range"),
    "degrees": ("", "OverflowError only for an int too large for a float"),
    "erf": ("in [-1.0, 1.0]; nan for nan", "OverflowError only for an int too large for a float"),
    "erfc": ("in [0.0, 2.0]; nan for nan", "OverflowError only for an int too large for a float"),
    "exp": ("in [0.0, inf]; nan for nan", "OverflowError when the result exceeds the float range"),
    "exp2": ("in [0.0, inf]; nan for nan", "OverflowError when the result exceeds the float range"),
    "expm1": ("in [-1.0, inf]; nan for nan", "OverflowError when the result exceeds the float range"),
    "fabs": ("in [0.0, inf]; ALWAYS a float, even for an int argument", "OverflowError only for an int too large for a float"),
    "gamma": ("", "ValueError for 0, a negative integer and -inf; OverflowError past about 171.6"),
    "lgamma": ("", "ValueError for 0, a negative integer; OverflowError for a huge argument"),
    "log10": ("", "ValueError when x <= 0"),
    "log1p": ("", "ValueError when x <= -1"),
    "log2": ("", "ValueError when x <= 0"),
    "radians": ("", "OverflowError only for an int too large for a float"),
    "sin": ("in [-1.0, 1.0]; nan for nan", "ValueError for an infinity"),
    "sinh": ("", "OverflowError when the result exceeds the float range"),
    "sqrt": ("result >= 0.0, or nan for a nan input", "ValueError for a negative argument"),
    "tan": ("", "ValueError for an infinity"),
    "tanh": ("in [-1.0, 1.0]; nan for nan", "OverflowError only for an int too large for a float"),
    "ulp": ("in [5e-324, inf]; nan for nan", "OverflowError only for an int too large for a float"),
}

#: the two- and three-argument float functions: name -> (arity text, shape, non-property exceptions)
_MULTI_FLOAT = {
    "atan2": ("exactly two positional arguments (y, x)", "in [-pi, pi]; nan for a nan argument",
              "OverflowError only for an int too large for a float"),
    "copysign": ("exactly two positional arguments", "", "OverflowError only for an int too large for a float"),
    "fmod": ("exactly two positional arguments", "", "ValueError for fmod(inf, y) and fmod(x, 0)"),
    "remainder": ("exactly two positional arguments", "", "ValueError for remainder(inf, y) and remainder(x, 0)"),
    "fma": ("exactly three positional arguments", "", "ValueError for an invalid operation (inf * 0, inf - inf), OverflowError when the exact result overflows"),
}


def _contracts():
    out = []
    for name, (shape, other) in _UNARY_FLOAT.items():
        out.append(_real(f"math.{name}", shape, notes=f"non-property exceptions: {other}."))
    for name, (arity, shape, other) in _MULTI_FLOAT.items():
        out.append(_real(
            f"math.{name}", shape, notes=f"non-property exceptions: {other}.",
            type_error=f"the call is not {arity} (a keyword is rejected too), or an argument has "
                       f"neither __float__ nor __index__, or its __float__ returns a non-float"))
    return out


_CONTRACTS = (
    *_contracts(),
    # ---- the predicates and the rounding functions
    _real("math.isnan", ret="bool", notes="OverflowError only for an int too large for a float."),
    _real("math.isinf", ret="bool", notes="OverflowError only for an int too large for a float."),
    _real("math.isfinite", ret="bool", notes="OverflowError only for an int too large for a float."),
    _contract(
        "math.floor", "#math.floor", ReturnSpec("int", "for an int, float or bool argument", tested=True),
        dunder="the argument's __floor__ (tried first), else its __float__ or __index__",
        type_error=_UNARY_TYPEERROR,
        notes="ValueError for nan, OverflowError for an infinity. Tries __floor__ FIRST (so Decimal/Fraction-like "
              "objects use their own rounding); falls back to __index__/__float__ only if __floor__ is absent. "
              "A user __floor__ may return any object."),
    _contract(
        "math.ceil", "#math.ceil", ReturnSpec("int", "for an int, float or bool argument", tested=True),
        dunder="the argument's __ceil__ (tried first), else its __float__ or __index__",
        type_error=_UNARY_TYPEERROR,
        notes="ValueError for nan, OverflowError for an infinity. Tries __ceil__ FIRST, same fallback order as floor()."),
    _contract(
        "math.trunc", "#math.trunc", ReturnSpec("int", "for an int, float or bool argument", tested=True),
        dunder="the argument's __trunc__ ONLY",
        type_error="the call is not exactly one positional argument, or the argument has no __trunc__ "
                   "(a __float__ or __index__ alone is NOT enough: only int, bool and float define __trunc__ "
                   "among the builtin numbers; a str, None or list has none)",
        notes="ValueError for nan, OverflowError for an infinity. A user __trunc__ may return any object."),
    _real("math.frexp", "a (float, int) pair (m, e) with x == m * 2**e, 0.5 <= |m| < 1 (or m is 0.0, inf or nan)",
          ret="tuple[float, int]", notes="OverflowError only for an int too large for a float; never raises for an infinity or nan."),
    _real("math.modf", "a (fractional, integral) float pair with the sign of x; (nan, nan) for nan, (0.0, inf) for inf",
          ret="tuple[float, float]", notes="OverflowError only for an int too large for a float."),
    _contract(
        "math.ldexp", "#math.ldexp", ReturnSpec("float", "x * 2**i", tested=True),
        dunder="the first argument's __float__ or __index__ (the second argument runs no user code: it must BE an int)",
        type_error="the call is not exactly two positional arguments, or the first argument is not real-number-like, or the "
                   "second argument is not an int (an int subclass or bool is fine; a float, even 2.0, and an object with "
                   "only __index__ are rejected: unlike the other integer functions ldexp does not call __index__)",
        notes="OverflowError when the result exceeds the float range (an int shift of any size is accepted when x is 0)."),
    _contract(
        "math.hypot", "#math.hypot", ReturnSpec("float", "in [0.0, inf]; hypot() is 0.0; inf if any coordinate is inf, "
                                                "even next to a nan", tested=True),
        dunder=_REAL + ", for each coordinate",
        type_error="a coordinate has neither __float__ nor __index__, or a keyword argument is given (any number of "
                   "positional coordinates is fine, none included)",
        notes="OverflowError for an int too large for a float, or when the result exceeds the float range."),
    _contract(
        "math.isclose", "#math.isclose", ReturnSpec("bool", tested=True),
        dunder=_REAL + ", for a, b, rel_tol and abs_tol",
        type_error="a or b is missing or extra positionals are given, an unknown keyword is given, or an argument is "
                   "not real-number-like (a and b may be passed by keyword; rel_tol and abs_tol are keyword-only)",
        notes="ValueError for a negative rel_tol or abs_tol; OverflowError for an int too large for a float. "
              "isclose(nan, nan) is False, isclose(inf, inf) is True."),
    _contract(
        "math.nextafter", "#math.nextafter", ReturnSpec("float", "the float that is `steps` representable values after x, towards y",
                                                       tested=True),
        dunder=_REAL + " for x and y, and the keyword steps' __index__",
        type_error="the call is not two positional arguments plus the optional keyword steps, an argument is not "
                   "real-number-like, or steps is neither None nor an integer (a float is rejected)",
        notes="ValueError for a negative steps."),
    _contract(
        "math.log", "#math.log", ReturnSpec("float", tested=True),
        dunder=_REAL + ", for x and for base",
        type_error="the call is not one or two positional arguments (log takes no keyword at all), or an argument has "
                   "neither __float__ nor __index__, or its __float__ returns a non-float",
        extra_native={
            "ZeroDivisionError": (
                "the two-argument form with a base of exactly 1 (an int 1, True, or the float 1.0) and x positive, inf or "
                "nan: log(x) / log(1) divides by zero. It is NOT ValueError, unlike every other domain error here; "
                "an x <= 0 is checked first and is ValueError. Also propagates if an argument's __float__ raises it"),
        },
        notes="ValueError when x <= 0 or base <= 0. An int argument is taken exactly, with no conversion to float "
              "(log(10**400) works). The one function of the module that raises ZeroDivisionError on its own."),
    _contract(
        "math.pow", "#math.pow", ReturnSpec("float", "ALWAYS float, even for two int arguments (unlike the ** operator/builtin pow())",
                                            tested=True),
        dunder=_REAL,
        type_error="the call is not exactly two positional arguments, or an argument has neither __float__ nor __index__",
        notes="ValueError for a domain error (a negative base with a fractional exponent, 0 ** negative -- it is NOT "
              "ZeroDivisionError here); OverflowError when the result exceeds the float range."),
    # ---- the integer functions
    _contract(
        "math.factorial", "#math.factorial", ReturnSpec("int", "result >= 1", tested=True),
        dunder=_INDEX,
        type_error="the argument has no __index__ -- includes float, even a whole value like 5.0 (confirmed on 3.14.4; "
                   "accepted with a deprecation warning in older Python, now a hard TypeError)",
        notes="ValueError for a negative argument; OverflowError above sys.maxsize."),
    _contract(
        "math.gcd", "#math.gcd", ReturnSpec("int", "gcd() with no arguments returns 0; result >= 0", tested=True),
        dunder="each argument's __index__ ONLY (same as factorial -- float is rejected)",
        type_error="an argument has no __index__ (includes float)"),
    _contract(
        "math.lcm", "#math.lcm", ReturnSpec("int", "lcm() with no arguments returns 1; result >= 0, and 0 if any argument is 0",
                                            tested=True),
        dunder="each argument's __index__ ONLY",
        type_error="an argument has no __index__ (includes float), or a keyword argument is given"),
    _contract(
        "math.isqrt", "#math.isqrt", ReturnSpec("int", "result >= 0 with result**2 <= n < (result+1)**2", tested=True),
        dunder=_INDEX,
        type_error="the call is not exactly one positional argument, or the argument has no __index__ (includes float)",
        notes="ValueError for a negative argument."),
    _contract(
        "math.comb", "#math.comb", ReturnSpec("int", "result >= 0; 0 when k > n", tested=True),
        dunder="each argument's __index__ ONLY",
        type_error="the call is not exactly two positional arguments, or an argument has no __index__ (includes float)",
        notes="ValueError for a negative n or k; OverflowError for k above sys.maxsize."),
    _contract(
        "math.perm", "#math.perm", ReturnSpec("int", "result >= 0; 0 when k > n; perm(n) is n!", tested=True),
        dunder="each argument's __index__ ONLY (k may be None)",
        type_error="the call is not one or two positional arguments, or an argument has no __index__ (includes float)",
        notes="ValueError for a negative n or k; OverflowError for k above sys.maxsize."),
    # ---- the functions over an iterable
    Contract(
        qualname="math.fsum",
        raises=dunder_propagation_claims(
            "the iterable's __iter__/__next__, and each yielded element's __float__/__index__",
            native={
                "TypeError": "the argument is not iterable, or yields something with neither __float__ nor __index__",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("float", "an exact, order-independent sum -- unlike sum(), immune to float-addition rounding drift", tested=True),
        effects=Effects(deterministic=True),
        notes="ValueError for -inf + inf; OverflowError for an intermediate overflow or an int too large for a float.",
        source=(f"{_DOCS}#math.fsum", "CPython Modules/mathmodule.c::math_fsum"),
        tested=True,
    ),
    Contract(
        qualname="math.prod",
        raises=dunder_propagation_claims(
            "the iterable's __iter__/__next__, and the elements' (and start's) __mul__/__rmul__",
            native={
                "TypeError": "the argument is not iterable, a second positional argument is given (start is keyword-only), "
                             "an unknown keyword is given, or two elements do not support *",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("object", "an int for ints, a float once a float is met; ANY object for elements with their own "
                                     "__mul__ (prod(['a'], start=2) is 'aa')", tested=True),
        effects=Effects(deterministic=True),
        notes="prod is a generic reduction by *, not a numeric function: a str, list or user object is accepted when it "
              "supports * with the running product.",
        source=(f"{_DOCS}#math.prod", "CPython Modules/mathmodule.c::math_prod_impl"),
        tested=True,
    ),
    Contract(
        qualname="math.dist",
        raises=dunder_propagation_claims(
            "the iterables' __iter__/__next__, and each element's __float__/__index__",
            native={
                "TypeError": "the call is not exactly two positional arguments, an argument is not iterable, or an element has "
                             "neither __float__ nor __index__",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("float", "the Euclidean distance, in [0.0, inf]", tested=True),
        effects=Effects(deterministic=True),
        notes="ValueError when the points have a different number of dimensions; OverflowError for an int too large for a float. "
              "Any iterable of reals is a point (a tuple, list, range...).",
        source=(f"{_DOCS}#math.dist", "CPython Modules/mathmodule.c::math_dist_impl"),
        tested=True,
    ),
    Contract(
        qualname="math.sumprod",
        raises=dunder_propagation_claims(
            "the iterables' __iter__/__next__, and the elements' __mul__/__add__ (and reflected forms)",
            native={
                "TypeError": "the call is not exactly two positional arguments, an argument is not iterable, or two elements "
                             "do not support * or +",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("object", "an int for ints, a float once a float is met, else whatever + and * give", tested=True),
        effects=Effects(deterministic=True),
        notes="ValueError when the inputs differ in length.",
        source=(f"{_DOCS}#math.sumprod", "CPython Modules/mathmodule.c::math_sumprod_impl"),
        tested=True,
    ),
    # ---- the constants
    *(
        Contract(
            qualname=f"math.{name}",
            raises=never_claims(tested=frozenset({"AssertionError", "AttributeError", "FloatingPointError", "IndexError",
                                                  "KeyError", "TypeError", "ZeroDivisionError"})),
            may_raise_other=False,
            returns=ReturnSpec("float", shape, tested=True),
            effects=Effects(deterministic=True, may_execute_callback=False,
                            notes="a module attribute read: it can be rebound by the program (math.pi = 3), which no contract "
                                  "can see; a consumer must treat an assignment to a math attribute as unmodelled."),
            notes="a constant, not a callable: reading it raises nothing and runs no code.",
            source=(f"{_DOCS}#math.{name}", "CPython Modules/mathmodule.c::math_exec"),
            tested=True,
        )
        for name, shape in (
            ("pi", "exactly 3.141592653589793"),
            ("e", "exactly 2.718281828459045"),
            ("tau", "exactly 6.283185307179586"),
            ("inf", "positive infinity"),
            ("nan", "a quiet nan (nan != nan)"),
        )
    ),
)

_BY_QUALNAME: dict[str, Contract] = {c.qualname: c for c in _CONTRACTS}
assert len(_BY_QUALNAME) == len(_CONTRACTS), "a qualname is contracted twice"
_CONSTANTS = tuple(f"math.{name}" for name in ("pi", "e", "tau", "inf", "nan"))
_FUNCTIONS = tuple(sorted(q for q in _BY_QUALNAME if q not in _CONSTANTS))


@register_rule
class MathFunction(Rule):
    """One of the 57 functions of ``math`` (``sin`` ... ``sumprod``).

    The visitor kind is ONE for the family because every engine treats them alike: it reads the per-function model
    table (``adapters/math_model.py``: argument kinds, result kind, closed float range) for the call it understands
    and the contract for the property classes.  :meth:`native` is CPython's own function (the default resolution
    over ``math``); :meth:`shape_for` is the call shape the contract's claims are valid for, read from that table.
    """

    SAMPLES = {'math.acos': 'import math\nz = math.acos(0.5)\n', 'math.acosh': 'import math\nz = math.acosh(2.0)\n', 'math.asin': 'import math\nz = math.asin(0.5)\n', 'math.asinh': 'import math\nz = math.asinh(0.5)\n', 'math.atan': 'import math\nz = math.atan(0.5)\n', 'math.atan2': 'import math\nz = math.atan2(1.0, 2.0)\n', 'math.atanh': 'import math\nz = math.atanh(0.5)\n', 'math.cbrt': 'import math\nz = math.cbrt(8.0)\n', 'math.ceil': 'import math\nz = math.ceil(2.5)\n', 'math.comb': 'import math\nz = math.comb(5, 2)\n', 'math.copysign': 'import math\nz = math.copysign(1.0, -2.0)\n', 'math.cos': 'import math\nz = math.cos(0.5)\n', 'math.cosh': 'import math\nz = math.cosh(0.5)\n', 'math.degrees': 'import math\nz = math.degrees(0.5)\n', 'math.dist': 'import math\nz = math.dist([0.0, 0.0], [3.0, 4.0])\n', 'math.erf': 'import math\nz = math.erf(0.5)\n', 'math.erfc': 'import math\nz = math.erfc(0.5)\n', 'math.exp': 'import math\nz = math.exp(0.5)\n', 'math.exp2': 'import math\nz = math.exp2(0.5)\n', 'math.expm1': 'import math\nz = math.expm1(0.5)\n', 'math.fabs': 'import math\nz = math.fabs(-0.5)\n', 'math.factorial': 'import math\nz = math.factorial(5)\n', 'math.floor': 'import math\nz = math.floor(2.5)\n', 'math.fma': 'import math\nz = math.fma(2.0, 3.0, 4.0)\n', 'math.fmod': 'import math\nz = math.fmod(5.0, 3.0)\n', 'math.frexp': 'import math\nz = math.frexp(8.0)\n', 'math.fsum': 'import math\nz = math.fsum([0.5, 0.25])\n', 'math.gamma': 'import math\nz = math.gamma(2.5)\n', 'math.gcd': 'import math\nz = math.gcd(12, 18)\n', 'math.hypot': 'import math\nz = math.hypot(3.0, 4.0)\n', 'math.isclose': 'import math\nz = math.isclose(1.0, 1.0)\n', 'math.isfinite': 'import math\nz = math.isfinite(0.5)\n', 'math.isinf': 'import math\nz = math.isinf(0.5)\n', 'math.isnan': 'import math\nz = math.isnan(0.5)\n', 'math.isqrt': 'import math\nz = math.isqrt(17)\n', 'math.lcm': 'import math\nz = math.lcm(4, 6)\n', 'math.ldexp': 'import math\nz = math.ldexp(1.5, 3)\n', 'math.lgamma': 'import math\nz = math.lgamma(2.5)\n', 'math.log': 'import math\nz = math.log(8.0, 2.0)\n', 'math.log10': 'import math\nz = math.log10(2.0)\n', 'math.log1p': 'import math\nz = math.log1p(0.5)\n', 'math.log2': 'import math\nz = math.log2(2.0)\n', 'math.modf': 'import math\nz = math.modf(2.5)\n', 'math.nextafter': 'import math\nz = math.nextafter(1.0, 2.0)\n', 'math.perm': 'import math\nz = math.perm(5, 2)\n', 'math.pow': 'import math\nz = math.pow(2.0, 3.0)\n', 'math.prod': 'import math\nz = math.prod([2, 3])\n', 'math.radians': 'import math\nz = math.radians(0.5)\n', 'math.remainder': 'import math\nz = math.remainder(5.0, 3.0)\n', 'math.sin': 'import math\nz = math.sin(0.5)\n', 'math.sinh': 'import math\nz = math.sinh(0.5)\n', 'math.sqrt': 'import math\nz = math.sqrt(4.0)\n', 'math.sumprod': 'import math\nz = math.sumprod([1, 2], [3, 4])\n', 'math.tan': 'import math\nz = math.tan(0.5)\n', 'math.tanh': 'import math\nz = math.tanh(0.5)\n', 'math.trunc': 'import math\nz = math.trunc(2.5)\n', 'math.ulp': 'import math\nz = math.ulp(1.0)\n'}
    TWINS = {'math.acos': 'import math\nz = math.acos()\n', 'math.acosh': 'import math\nz = math.acosh()\n', 'math.asin': 'import math\nz = math.asin()\n', 'math.asinh': 'import math\nz = math.asinh()\n', 'math.atan': 'import math\nz = math.atan()\n', 'math.atan2': 'import math\nz = math.atan2()\n', 'math.atanh': 'import math\nz = math.atanh()\n', 'math.cbrt': 'import math\nz = math.cbrt()\n', 'math.ceil': 'import math\nz = math.ceil()\n', 'math.comb': 'import math\nz = math.comb()\n', 'math.copysign': 'import math\nz = math.copysign()\n', 'math.cos': 'import math\nz = math.cos()\n', 'math.cosh': 'import math\nz = math.cosh()\n', 'math.degrees': 'import math\nz = math.degrees()\n', 'math.dist': 'import math\nz = math.dist()\n', 'math.erf': 'import math\nz = math.erf()\n', 'math.erfc': 'import math\nz = math.erfc()\n', 'math.exp': 'import math\nz = math.exp()\n', 'math.exp2': 'import math\nz = math.exp2()\n', 'math.expm1': 'import math\nz = math.expm1()\n', 'math.fabs': 'import math\nz = math.fabs()\n', 'math.factorial': 'import math\nz = math.factorial()\n', 'math.floor': 'import math\nz = math.floor()\n', 'math.fma': 'import math\nz = math.fma()\n', 'math.fmod': 'import math\nz = math.fmod()\n', 'math.frexp': 'import math\nz = math.frexp()\n', 'math.fsum': 'import math\nz = math.fsum()\n', 'math.gamma': 'import math\nz = math.gamma()\n', 'math.isclose': 'import math\nz = math.isclose()\n', 'math.isfinite': 'import math\nz = math.isfinite()\n', 'math.isinf': 'import math\nz = math.isinf()\n', 'math.isnan': 'import math\nz = math.isnan()\n', 'math.isqrt': 'import math\nz = math.isqrt()\n', 'math.ldexp': 'import math\nz = math.ldexp()\n', 'math.lgamma': 'import math\nz = math.lgamma()\n', 'math.log': 'import math\nz = math.log()\n', 'math.log10': 'import math\nz = math.log10()\n', 'math.log1p': 'import math\nz = math.log1p()\n', 'math.log2': 'import math\nz = math.log2()\n', 'math.modf': 'import math\nz = math.modf()\n', 'math.nextafter': 'import math\nz = math.nextafter()\n', 'math.perm': 'import math\nz = math.perm()\n', 'math.pow': 'import math\nz = math.pow()\n', 'math.prod': 'import math\nz = math.prod()\n', 'math.radians': 'import math\nz = math.radians()\n', 'math.remainder': 'import math\nz = math.remainder()\n', 'math.sin': 'import math\nz = math.sin()\n', 'math.sinh': 'import math\nz = math.sinh()\n', 'math.sqrt': 'import math\nz = math.sqrt()\n', 'math.sumprod': 'import math\nz = math.sumprod()\n', 'math.tan': 'import math\nz = math.tan()\n', 'math.tanh': 'import math\nz = math.tanh()\n', 'math.trunc': 'import math\nz = math.trunc()\n', 'math.ulp': 'import math\nz = math.ulp()\n'}

    names = _FUNCTIONS
    kind = "math_function"

    @classmethod
    def contract_for(cls, qualname: str) -> Contract:
        return _BY_QUALNAME[qualname]

    @classmethod
    def shape_for(cls, qualname: str) -> Shape:
        from tharkun.solve.contracts.adapters.math_model import _MODEL

        fn = _MODEL[qualname.split(".", 1)[1]]
        return Shape(min_args=fn.min_args, max_args=fn.max_args, keywords_ok=bool(fn.keywords))


@register_rule
class MathConstant(Rule):
    """``math.pi``, ``e``, ``tau``, ``inf``, ``nan``: a module attribute read (no call), so :meth:`native` returns
    the value and the shape accepts no argument."""

    SAMPLES = {'math.e': 'import math\nz = math.e\n', 'math.inf': 'import math\nz = math.inf\n', 'math.nan': 'import math\nz = math.nan\n', 'math.pi': 'import math\nz = math.pi\n', 'math.tau': 'import math\nz = math.tau\n'}
    TWINS = {}

    names = _CONSTANTS
    kind = "math_constant"
    SHAPE = Shape(min_args=0, max_args=0)

    @classmethod
    def contract_for(cls, qualname: str) -> Contract:
        return _BY_QUALNAME[qualname]

    def native(self, *args, **kwargs):
        if args or kwargs:
            raise TypeError(f"{self.qualname} is a constant, not a callable")
        return getattr(math, self.qualname.split(".", 1)[1])


__all__ = ["MathFunction", "MathConstant"]
