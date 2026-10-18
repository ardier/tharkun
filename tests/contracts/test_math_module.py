"""The whole ``math`` module, pinned against CPython 3.14.4 ([internal ref]).

``tharkun.solve.contracts.rules_math`` holds one contract per function and constant;
``tharkun.solve.contracts.adapters.math_model`` is the table the engines act on.  This module
checks every claim of both against the interpreter:

* completeness: the contracts are exactly the public names of ``math``;
* the call shape of the model (argument count, keywords, the kind each argument needs, the kind of the result);
* the TypeError conditions, and that every other property class reaches the caller ONLY through a user
  object's dunder (``__float__``, ``__index__``, ``__iter__``, ``__trunc__``) -- all seven classes swept;
* what an exact-builtin-number call can raise (:data:`SCALAR_SHAPE_RAISES`): nothing but ``ValueError`` and
  ``OverflowError``, plus ``ZeroDivisionError`` for ``log(x, 1)``, swept over a grid and by Hypothesis;
* the closed ranges and lower bounds of the results.
"""

from __future__ import annotations

import itertools
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import all_contracts, for_module, get
from tharkun.solve.contracts.adapters import math_model as mm
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES

INF = math.inf
NAN = math.nan

#: one valid argument tuple per function (the domain of every function holds for it)
VALID = {
    "acos": (0.5,), "acosh": (2.0,), "asin": (0.5,), "asinh": (0.5,), "atan": (0.5,), "atanh": (0.5,),
    "cbrt": (8.0,), "cos": (0.5,), "cosh": (0.5,), "degrees": (0.5,), "erf": (0.5,), "erfc": (0.5,),
    "exp": (0.5,), "exp2": (0.5,), "expm1": (0.5,), "fabs": (-0.5,), "gamma": (2.5,), "lgamma": (2.5,),
    "log10": (2.0,), "log1p": (0.5,), "log2": (2.0,), "radians": (0.5,), "sin": (0.5,), "sinh": (0.5,),
    "sqrt": (4.0,), "tan": (0.5,), "tanh": (0.5,), "ulp": (1.0,),
    "isnan": (0.5,), "isinf": (0.5,), "isfinite": (0.5,),
    "floor": (2.5,), "ceil": (2.5,), "trunc": (2.5,), "frexp": (8.0,), "modf": (2.5,),
    "atan2": (1.0, 2.0), "copysign": (1.0, -2.0), "fmod": (5.0, 3.0), "remainder": (5.0, 3.0),
    "pow": (2.0, 3.0), "ldexp": (1.5, 3), "fma": (2.0, 3.0, 4.0), "hypot": (3.0, 4.0), "log": (8.0, 2.0),
    "isclose": (1.0, 1.0), "nextafter": (1.0, 2.0),
    "factorial": (5,), "gcd": (12, 18), "lcm": (4, 6), "isqrt": (17,), "comb": (5, 2), "perm": (5, 2),
    "fsum": ([0.5, 0.25],), "prod": ([2, 3],), "dist": ((0.0, 0.0), (3.0, 4.0)), "sumprod": ([1, 2], [3, 4]),
}
RESULT_TYPES = {
    "float": (float,), "bool": (bool,), "int": (int,), "float,int": (tuple,), "float,float": (tuple,),
    "number": (int, float),
}
#: the dunder a user object must define, per argument kind, for the call to reach user code
DUNDER = {"real": "__float__", "int": "__index__", "intopt": "__index__", "iterable": "__iter__"}
#: functions whose real arguments are NOT converted through __float__ (trunc asks __trunc__ only)
SPECIAL_DUNDER = {"trunc": "__trunc__"}
#: positions whose kind the model states loosely: perm's k may be None, prod/sumprod iterate any objects
LOOSE = set()
#: positions that run no user code at all: ldexp's shift must BE an int, __index__ is not asked
NO_DUNDER = {("ldexp", 1)}


def _model_functions():
    return sorted(mm.modelled_names())


# --------------------------------------------------------------------------- completeness
def test_the_contracts_are_exactly_the_public_names_of_math():
    public = {n for n in dir(math) if not n.startswith("_")}
    registered = {c.qualname.split(".", 1)[1] for c in for_module("math")}
    assert registered == public


def test_the_model_covers_every_function_and_constant():
    functions = {n for n in dir(math) if callable(getattr(math, n)) and not n.startswith("_")}
    assert mm.modelled_names() == functions
    assert set(mm.CONSTANTS) == {n for n in dir(math) if not n.startswith("_")} - functions


def test_every_math_contract_is_tested_and_total():
    for c in for_module("math"):
        assert c.tested, c.qualname
        assert {r.exc for r in c.raises} == set(ALL_PROPERTY_CLASSES), c.qualname


def test_every_function_has_a_valid_sample():
    assert set(VALID) == mm.modelled_names()


@pytest.mark.parametrize("name", _model_functions())
def test_the_model_needs_the_tested_contract(name, monkeypatch):
    assert mm.math_function(name) is not None
    from tharkun.solve.contracts.adapters import math_model

    monkeypatch.setattr(math_model, "resolve", lambda q: None)
    assert mm.math_function(name) is None


# --------------------------------------------------------------------------- call shape
@pytest.mark.parametrize("name", _model_functions())
class TestCallShape:
    def test_a_valid_call_returns_the_modelled_kind(self, name):
        fn = mm.math_function(name)
        out = getattr(math, name)(*VALID[name])
        assert type(out) in RESULT_TYPES[fn.result] or isinstance(out, RESULT_TYPES[fn.result])
        if fn.result == "float,int":
            assert type(out[0]) is float and type(out[1]) is int
        if fn.result == "float,float":
            assert type(out[0]) is float and type(out[1]) is float
        if fn.result == "int":
            assert type(out) is int
        if fn.result == "bool":
            assert type(out) is bool

    def test_the_argument_count_range(self, name):
        fn = mm.math_function(name)
        f = getattr(math, name)
        sample = VALID[name]
        if fn.min_args > 0:
            for n in range(fn.min_args):
                args = (sample * 3)[:n] if sample else ()
                with pytest.raises(TypeError):
                    f(*args)
                assert fn.shape(n, ()) == "typeerror"
        if fn.max_args is not None:
            args = (sample * 6)[:fn.max_args + 1]
            if name in ("fsum", "prod", "dist", "sumprod"):
                args = tuple([sample[0]] * (fn.max_args + 1))
            with pytest.raises(TypeError):
                f(*args)
            assert fn.shape(fn.max_args + 1, ()) == "typeerror"
        for n in range(fn.min_args, (fn.max_args if fn.max_args is not None else fn.min_args + 3) + 1):
            assert fn.shape(n, ()) == "ok"

    def test_unknown_keywords_are_a_typeerror(self, name):
        fn = mm.math_function(name)
        with pytest.raises(TypeError):
            getattr(math, name)(*VALID[name], no_such_keyword=1)
        assert fn.shape(len(VALID[name]), ("no_such_keyword",)) == "typeerror"

    def test_keyword_calls_are_unmodelled_exactly_for_the_functions_that_take_keywords(self, name):
        fn = mm.math_function(name)
        sample = VALID[name]
        accepted = {"isclose": dict(rel_tol=0.1), "nextafter": dict(steps=2), "prod": dict(start=2)}
        if name in accepted:
            getattr(math, name)(*sample, **accepted[name])   # valid on 3.14.4
            assert fn.shape(len(sample), tuple(accepted[name])) == "unmodelled"
        else:
            # a function with no keyword at all rejects every keyword; so does the model
            assert not fn.keywords
            assert fn.shape(len(sample), ("x",)) == "typeerror"

    def test_each_argument_kind(self, name):
        fn = mm.math_function(name)
        f = getattr(math, name)
        sample = VALID[name]
        for i in range(len(sample)):
            kind = fn.kind_of(i)
            bad = {"real": ["x", None, [], 1j, object()], "int": ["x", None, 2.0, 2.5, 1j],
                   "intopt": ["x", 2.0, 2.5, 1j], "iterable": [5, 2.5, None]}[kind]
            if name == "trunc":
                bad = ["x", None, [], 1j]
            for b in bad:
                if (name, i) in LOOSE and b is None:
                    continue
                args = list(sample)
                args[i] = b
                with pytest.raises(TypeError):
                    f(*args)
            ok = {"real": [True, 2, 0.5], "int": [True, 2], "intopt": [True, 2, None], "iterable": [(), []]}[kind]
            for g in ok:
                if name == "trunc" or (name in ("factorial",) and g == 0.5):
                    continue
                args = list(sample)
                args[i] = g
                try:
                    f(*args)
                except (ValueError, OverflowError):
                    pass   # a domain error is not a kind error
                except ZeroDivisionError:
                    assert name == "log"   # base True is 1


# --------------------------------------------------------------------------- propagation
def _positions(name):
    fn = mm.math_function(name)
    n = max(fn.min_args, 1) if fn.max_args is None else fn.max_args
    return [(i, fn.kind_of(i)) for i in range(min(n, len(VALID[name])))]


def _propagation_cases():
    for name in _model_functions():
        fn = mm.math_function(name)
        for i, kind in _positions(name):
            dunder = SPECIAL_DUNDER.get(name) if kind == "real" else None
            dunder = dunder or DUNDER[kind]
            if (name, i) in NO_DUNDER:
                continue
            yield pytest.param(name, i, dunder, id=f"{name}[{i}]:{dunder}")


@pytest.mark.parametrize("name,index,dunder", list(_propagation_cases()))
@pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
def test_a_user_dunder_exception_propagates_unchanged(name, index, dunder, exc_name):
    """The ONLY way a class other than TypeError (and log's ZeroDivisionError) leaves a ``math``
    call is through a user object: every one of the seven classes passes through unchanged."""
    args = list(VALID[name])
    args[index] = raiser(dunder, exc_name)()
    with pytest.raises(PROPERTY_CLASSES[exc_name]):
        getattr(math, name)(*args)


@pytest.mark.parametrize("name", [n for n in _model_functions()
                                  if "real" in mm.math_function(n).args and n not in ("trunc", "prod", "sumprod")])
def test_a_float_dunder_returning_a_non_float_is_a_typeerror(name):
    class NotAFloat:
        def __float__(self):
            return "x"

    fn = mm.math_function(name)
    args = list(VALID[name])
    i = next(i for i in range(len(args)) if fn.kind_of(i) == "real")
    args[i] = NotAFloat()
    with pytest.raises(TypeError):
        getattr(math, name)(*args)


@pytest.mark.parametrize("name", ["floor", "ceil"])
def test_floor_and_ceil_ask_their_own_dunder_first(name):
    class Own:
        def __floor__(self):
            return "own-floor"

        def __ceil__(self):
            return "own-ceil"

        def __float__(self):
            raise KeyError("never reached")

    assert getattr(math, name)(Own()) == f"own-{name}"


def test_trunc_needs_its_own_dunder_not_a_float():
    class OnlyFloat:
        def __float__(self):
            return 2.5

    with pytest.raises(TypeError):
        math.trunc(OnlyFloat())

    class Own:
        def __trunc__(self):
            return 7

    assert math.trunc(Own()) == 7


# --------------------------------------------------------------------------- the non-property exceptions
#: (function, arguments, the exception CPython raises): the domain errors and overflows the contracts name
OTHER = [
    ("acos", (2.0,), ValueError), ("acosh", (0.5,), ValueError), ("asin", (-2.0,), ValueError),
    ("atanh", (1.0,), ValueError), ("cos", (INF,), ValueError), ("sin", (-INF,), ValueError),
    ("tan", (INF,), ValueError), ("cosh", (711.0,), OverflowError), ("sinh", (711.0,), OverflowError),
    ("exp", (710.0,), OverflowError), ("exp2", (1024.0,), OverflowError), ("expm1", (710.0,), OverflowError),
    ("gamma", (0.0,), ValueError), ("gamma", (-1.0,), ValueError), ("gamma", (172.0,), OverflowError),
    ("lgamma", (0.0,), ValueError), ("lgamma", (1e308,), OverflowError),
    ("log", (0,), ValueError), ("log", (-1.0,), ValueError), ("log", (2, 0), ValueError), ("log", (2, -1), ValueError),
    ("log10", (0,), ValueError), ("log2", (-1.0,), ValueError), ("log1p", (-1.0,), ValueError),
    ("sqrt", (-1.0,), ValueError), ("sqrt", (-INF,), ValueError), ("sqrt", (10 ** 400,), OverflowError),
    ("fabs", (10 ** 400,), OverflowError), ("isnan", (10 ** 400,), OverflowError),
    ("pow", (0.0, -1.0), ValueError), ("pow", (-8.0, 1 / 3), ValueError), ("pow", (10.0, 400.0), OverflowError),
    ("fmod", (1.0, 0.0), ValueError), ("fmod", (INF, 1.0), ValueError),
    ("remainder", (1.0, 0.0), ValueError), ("remainder", (INF, 1.0), ValueError),
    ("floor", (NAN,), ValueError), ("floor", (INF,), OverflowError), ("ceil", (NAN,), ValueError),
    ("trunc", (INF,), OverflowError), ("ldexp", (1.0, 2000), OverflowError),
    ("fma", (INF, 0.0, 1.0), ValueError), ("fma", (1e308, 10.0, -1e308), OverflowError),
    ("isclose", (1.0, 1.0, ), None),
    ("factorial", (-1,), ValueError), ("factorial", (10 ** 30,), OverflowError),
    ("isqrt", (-1,), ValueError), ("comb", (-1, 1), ValueError), ("comb", (5, -1), ValueError),
    ("perm", (-1, 1), ValueError), ("perm", (10 ** 30, 10 ** 30), OverflowError),
    ("fsum", ([INF, -INF],), ValueError), ("fsum", ([1e308, 1e308],), OverflowError),
    ("dist", ([0.0], [0.0, 1.0]), ValueError), ("sumprod", ([1], [2, 3]), ValueError),
]


@pytest.mark.parametrize("name,args,exc", [c for c in OTHER if c[2] is not None],
                         ids=lambda v: v.__name__ if isinstance(v, type) else None)
def test_domain_errors_and_overflows_are_not_property_classes(name, args, exc):
    with pytest.raises(exc) as info:
        getattr(math, name)(*args)
    assert type(info.value) is exc
    assert exc.__name__ not in ALL_PROPERTY_CLASSES
    assert get(f"math.{name}").may_raise_other


class TestLog:
    """``log`` is the one function that raises ZeroDivisionError on its own."""

    @pytest.mark.parametrize("x", [2, 0.5, True, 10 ** 400, INF, NAN, 1])
    @pytest.mark.parametrize("base", [1, 1.0, True])
    def test_base_one_is_zero_division(self, x, base):
        with pytest.raises(ZeroDivisionError):
            math.log(x, base)

    @pytest.mark.parametrize("x,base", [(0, 1), (-1, 1), (0.0, 1.0)])
    def test_a_non_positive_x_is_checked_first_and_is_value_error(self, x, base):
        with pytest.raises(ValueError):
            math.log(x, base)

    def test_a_base_next_to_one_is_not_zero_division(self):
        assert math.log(2, 1 + 2 ** -52) > 1e15
        assert math.log(2, 1 - 2 ** -53) < -1e15

    @given(st.one_of(st.integers(), st.floats()), st.one_of(st.integers(), st.floats()))
    @settings(max_examples=300, deadline=None)
    def test_zero_division_iff_base_is_one(self, x, base):
        try:
            math.log(x, base)
            raised = None
        except (ValueError, OverflowError, ZeroDivisionError) as e:
            raised = type(e)
        if raised is ZeroDivisionError:
            assert base == 1 and not (isinstance(x, (int, float)) and not math.isnan(x) and x <= 0)

    def test_the_claim_names_the_condition(self):
        claim = get("math.log").raise_claim("ZeroDivisionError")
        assert claim.tested and "base" in claim.condition and "exactly 1" in claim.condition

    def test_log_takes_no_keyword_at_all(self):
        with pytest.raises(TypeError):
            math.log(2, base=3)
        with pytest.raises(TypeError):
            math.log(x=2)


# --------------------------------------------------------------------------- what a builtin-number call raises
INT_GRID = [0, 1, -1, 2, 10, -10, 255, 10 ** 15, -(10 ** 15), 10 ** 30, -(10 ** 30), 10 ** 400, True, False]
FLOAT_GRID = [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.5, 1e-300, 1e300, -1e300, 710.0, -745.0, INF, -INF, NAN]
SMALL_INT_GRID = [0, 1, 2, 5, 20, True, False, -1, -5]


def _grid(kind):
    return {"real": INT_GRID + FLOAT_GRID, "int": SMALL_INT_GRID, "intopt": SMALL_INT_GRID + [None]}[kind]


def _scalar_functions():
    return [n for n in _model_functions() if mm.math_function(n).args[0] in ("real", "int")]


def test_only_log_dist_prod_and_sumprod_produce_nan_from_numbers():
    """A nan result needs a nan argument everywhere else (swept over a grid with no nan in it)."""
    grid = [0, 1, -1, 2, 0.0, -0.0, 0.5, -0.5, 1.0, 2.5, 1e300, -1e300, 1e-300, 710.0, INF, -INF, 10 ** 30, True]
    produced = set()
    for name in _scalar_functions():
        fn = mm.math_function(name)
        if any(k == "int" or k == "intopt" for k in (fn.kind_of(i) for i in range(len(VALID[name])))):
            continue
        arity = range(fn.min_args, (fn.max_args if fn.max_args is not None else 3) + 1)
        for k in arity:
            pool = grid if k <= 2 else [0, 1, 2.5, INF, -INF, 1e300]
            for args in itertools.product(*[pool] * k):
                try:
                    r = getattr(math, name)(*args)
                except (ValueError, OverflowError, ZeroDivisionError):
                    continue
                if any(isinstance(v, float) and math.isnan(v) for v in (r if isinstance(r, tuple) else (r,))):
                    produced.add(name)
    assert produced == {n for n in mm.modelled_names() if mm.math_function(n).nan_from_numbers and n == "log"} == {"log"}
    assert {n for n in mm.modelled_names() if mm.math_function(n).nan_from_numbers} == {"log", "dist", "sumprod", "prod"}
    assert math.isnan(math.dist([INF], [INF])) and math.isnan(math.sumprod([INF], [0])) and math.isnan(math.prod([INF, 0]))


def _exceptions(name, args):
    try:
        getattr(math, name)(*args)
    except BaseException as e:  # noqa: BLE001
        return type(e)
    return None


@pytest.mark.parametrize("name", _scalar_functions())
def test_exact_number_arguments_raise_only_value_overflow_and_the_tabled_classes(name):
    """The table: nothing outside ValueError/OverflowError (and MemoryError for the integer functions that
    can ask for the moon) for ANY exact int/bool/float of the right kinds, and never a property class
    except what :data:`SCALAR_SHAPE_RAISES` names."""
    fn = mm.math_function(name)
    allowed = {ValueError, OverflowError, MemoryError, None} | {PROPERTY_CLASSES[c] for c in mm.scalar_shape_raises(name)}
    kinds = [fn.kind_of(i) for i in range(len(VALID[name]))]
    grids = [_grid(k) for k in kinds]
    if len(grids) > 2:
        grids = [g[::3] for g in grids]
    seen = set()
    for args in itertools.product(*grids):
        if name in ("factorial", "comb", "perm") and any(isinstance(a, int) and a > 5000 for a in args):
            continue   # a huge factorial/combination takes forever; its OverflowError/MemoryError is not a property class
        seen.add(_exceptions(name, args))
    assert seen <= allowed, (name, seen)


@pytest.mark.parametrize("name", [n for n in _scalar_functions() if mm.math_function(n).max_args is None])
def test_variable_arity_functions_never_raise_a_property_class_for_numbers(name):
    for n in range(0, 5):
        for args in itertools.product(*[(INT_GRID if mm.math_function(name).args[0] == "int" else INT_GRID + FLOAT_GRID)[::4]] * n):
            e = _exceptions(name, args)
            assert e in (None, ValueError, OverflowError), (name, args, e)


@given(st.floats(allow_nan=True, allow_infinity=True))
@settings(max_examples=200, deadline=None)
def test_no_unary_float_function_raises_a_property_class(x):
    for name in _scalar_functions():
        fn = mm.math_function(name)
        if fn.min_args != 1 or fn.max_args != 1 or fn.args != ("real",):
            continue
        e = _exceptions(name, (x,))
        assert e in (None, ValueError, OverflowError), (name, x, e)


def test_the_table_of_scalar_shape_raises_is_log_only():
    assert mm.SCALAR_SHAPE_RAISES == {"log": frozenset({"ZeroDivisionError"})}
    assert all(mm.scalar_shape_raises(n) == frozenset() for n in mm.modelled_names() - {"log"})


# --------------------------------------------------------------------------- results
def _range_cases():
    return [n for n in _model_functions() if mm.math_function(n).float_range is not None]


@pytest.mark.parametrize("name", _range_cases())
def test_a_float_result_stays_in_the_stated_range(name):
    fn = mm.math_function(name)
    lo, hi = fn.float_range
    arity = len(VALID[name])
    pool = INT_GRID[:10] + FLOAT_GRID
    for args in itertools.product(*[pool] * min(arity, 2)):
        args = args + VALID[name][len(args):]
        if name == "dist":
            args = ((args[0],), (args[1],))
        try:
            r = getattr(math, name)(*args)
        except (ValueError, OverflowError):
            continue
        assert math.isnan(r) or lo <= r <= hi, (name, args, r)


@given(st.floats(allow_nan=True, allow_infinity=True), st.floats(allow_nan=True, allow_infinity=True))
@settings(max_examples=200, deadline=None)
def test_ranges_hold_for_arbitrary_floats(x, y):
    for name in _range_cases():
        fn = mm.math_function(name)
        arity = len(VALID[name])
        args = (x, y)[:arity] if arity <= 2 else (x, y, 1.0)[:arity]
        if name == "hypot":
            args = (x, y)
        if name == "dist":
            args = ((x,), (y,))
        try:
            r = getattr(math, name)(*args)
        except (ValueError, OverflowError):
            continue
        lo, hi = fn.float_range
        assert math.isnan(r) or lo <= r <= hi, (name, args, r)


@pytest.mark.parametrize("name", [n for n in _model_functions() if mm.math_function(n).int_min is not None])
def test_an_int_result_respects_its_lower_bound(name):
    fn = mm.math_function(name)
    pool = [0, 1, 2, 3, 6, 10, 12, True, False]
    for args in itertools.product(*[pool] * max(len(VALID[name]), 1)):
        args = args[:max(len(VALID[name]), 1)] if fn.max_args is None else args[:fn.max_args]
        if name == "factorial" and args[0] > 500:
            continue
        try:
            r = getattr(math, name)(*args)
        except (ValueError, OverflowError):
            continue
        assert r >= fn.int_min, (name, args, r)
    if fn.min_args == 0:
        assert getattr(math, name)() >= fn.int_min


# --------------------------------------------------------------------------- shapes of tuples, sign and zero
def test_frexp_and_modf_results():
    for x in [0.0, -0.0, 8.0, -2.5, 1e-310, 1e300, INF, -INF]:
        m, e = math.frexp(x)
        assert isinstance(m, float) and isinstance(e, int)
        assert m == 0.0 or math.isinf(m) or 0.5 <= abs(m) < 1.0
        f, i = math.modf(x)
        assert isinstance(f, float) and isinstance(i, float)
    assert math.isnan(math.frexp(NAN)[0]) and all(math.isnan(v) for v in math.modf(NAN))


def test_copysign_distinguishes_negative_zero():
    assert math.copysign(1, -0.0) == -1.0 and math.copysign(1, 0.0) == 1.0
    assert math.copysign(3, -5) == -3.0 and math.copysign(-3, 5) == 3.0


def test_fabs_of_an_int_is_a_float():
    assert type(math.fabs(-3)) is float and math.fabs(-3) == 3.0


def test_pow_zero_to_a_negative_power_is_value_error_not_zero_division():
    with pytest.raises(ValueError):
        math.pow(0, -1)


# --------------------------------------------------------------------------- the iterable functions
class TestIterables:
    def test_fsum_is_exact(self):
        assert math.fsum([0.1] * 10) == 1.0

    def test_prod_is_a_generic_reduction(self):
        assert math.prod([2, 3.0]) == 6.0
        assert math.prod(["a"], start=2) == "aa"
        assert math.prod([[1], 2]) == [1, 1]

    def test_prod_start_is_keyword_only(self):
        with pytest.raises(TypeError):
            math.prod([1], 2)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_prod_element_dunder_propagates(self, exc_name):
        # the running product starts at 1, so the first multiplication is int * element: the
        # element's REFLECTED __rmul__ answers it
        Bad = raiser("__rmul__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            math.prod([Bad(), 2])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_sumprod_element_dunder_propagates(self, exc_name):
        Bad = raiser("__mul__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            math.sumprod([Bad()], [2])

    def test_dist_accepts_any_iterable_of_reals(self):
        assert math.dist(range(2), range(2)) == 0.0
        assert math.dist((0,), (3,)) == 3.0

    def test_sumprod_of_unsupported_operands_is_typeerror(self):
        with pytest.raises(TypeError):
            math.sumprod(["a"], [2])


# --------------------------------------------------------------------------- the constants
@pytest.mark.parametrize("name", sorted(mm.CONSTANTS))
class TestConstants:
    def test_the_contract_is_total_and_never(self, name):
        c = get(f"math.{name}")
        assert c.tested and not c.may_raise_other and not c.effects.may_execute_callback
        assert all(r.condition == "never" and r.tested for r in c.raises)

    def test_the_value(self, name):
        want = mm.constant(name)
        got = getattr(math, name)
        assert (math.isnan(got) and math.isnan(want)) or got == want

    def test_the_constant_is_a_float(self, name):
        assert type(getattr(math, name)) is float


def test_ldexp_shift_must_be_an_int_not_merely_indexable():
    class Index:
        def __index__(self):
            return 3

    class MyInt(int):
        pass

    with pytest.raises(TypeError):
        math.ldexp(1.0, Index())
    assert math.ldexp(1.0, MyInt(3)) == 8.0 and math.ldexp(1.0, True) == 2.0
    assert math.ldexp(1.0, -(10 ** 30)) == 0.0
