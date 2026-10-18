"""Engine 2 (the static analyser) consumption of the ``math`` contracts ([internal ref] batch 5).

Before this batch ``interp.py::MODULES`` modelled three functions of ``math`` (``sin``, ``cos``,
``sqrt``); ``math.tan`` (bm_raytrace), ``math.log`` and every other call fell to the contract
fallback (``_contract_call``), which records unproven checks and returns an unknown value that
aborts the next use.  ``Analyzer._math_call`` now models EVERY function of the module, driven by
the contract and the shared table ``contracts/adapters/math_model.py``:

* the contract supplies the seven classes, the model proves what it can for the abstract arguments
  (``TypeError`` once the argument count and kinds are verified, ``ZeroDivisionError`` of ``log``
  once the base provably is not 1, everything else because exact numbers run no user code);
* the result is an interval (``solve/values.py``'s transfer where one exists, else the closed range
  the table states, else any float), an int with its documented lower bound, a bool or a pair;
* a call that CPython rejects (argument count, a keyword, an argument that is not a number)
  stays an unproven ``TypeError`` -- never TRUE for the dynamic-typing property.

Pinned against CPython in ``test_math_module.py``; here against the analyser.
"""

from __future__ import annotations

import dataclasses
import itertools
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.contracts.test_math_module import VALID
from tharkun.solve import values as fv
from tharkun.solve.absint import prove
from tharkun.solve.absint.interp import MODULES
from tharkun.solve.contracts.adapters import absint as _absint
from tharkun.solve.contracts.adapters import math_model as mm
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES

ALL = tuple(ALL_PROPERTY_CLASSES)
#: a float LITERAL is the any-float value in absint (``Analyzer.e_Constant``): log's base is given as an int
SAMPLE = {**VALID, "log": (8.0, 2)}
AR = ("ZeroDivisionError", "FloatingPointError")
DL = ("KeyError", "IndexError")
DT = ("TypeError", "AttributeError")
AS = ("AssertionError",)
INF = math.inf


def _lit(v):
    if isinstance(v, float):
        return "float('inf')" if v == INF else "float('-inf')" if v == -INF else repr(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_lit(x) for x in v) + "]"
    return repr(v)


def _call(name, args):
    return f"math.{name}({', '.join(_lit(a) for a in args)})"


def _true(src, classes=ALL):
    r = prove(src, classes)
    assert r.verdict == "TRUE", (src, r.reason)
    return r


def _never_true(src, classes):
    r = prove(src, classes)
    assert r.verdict == "UNKNOWN", (src, r.explain())
    return r


# --------------------------------------------------------------------------- the gate
class TestGate:
    def test_every_function_of_math_is_modelled_and_every_constant_is_known(self):
        assert MODULES["math"]["funcs"] == mm.modelled_names()
        assert set(MODULES["math"]["consts"]) == set(mm.CONSTANTS)

    def test_the_exact_constants_are_the_floats(self):
        """The values are the contract's, read through the rule (``AbsintVisitor.visit_math_constant``, [internal ref] step 2): the
        analyzer proves ``math.<name> == <CPython's float>`` and nothing else."""
        for name in ("pi", "e", "tau"):
            _true(f"import math\nassert math.{name} == {getattr(math, name)!r}\n", ("AssertionError",))
            assert prove(f"import math\nassert math.{name} == {getattr(math, name) + 0.5!r}\n",
                         ("AssertionError",)).verdict != "TRUE"

    @pytest.mark.parametrize("name", sorted(mm.modelled_names()))
    def test_a_tested_contract_and_a_model_exist(self, name):
        assert _absint.contract_for_module_func("math", name).tested
        assert _absint.math_function(name) is not None

    @pytest.mark.parametrize("name", sorted(mm.modelled_names()))
    def test_withdrawing_the_contract_withdraws_the_model(self, monkeypatch, name):
        load_all()
        monkeypatch.delitem(REGISTRY, f"math.{name}")
        r = prove(f"import math\nz = {_call(name, SAMPLE[name])}\n", ALL)
        assert r.verdict == "UNKNOWN" and "math" in r.reason and name in r.reason or "unsupported" in r.reason

    @pytest.mark.parametrize("name", ["sin", "tan", "log", "floor", "gcd"])
    def test_an_untested_contract_withdraws_the_model_too(self, monkeypatch, name):
        load_all()
        c = REGISTRY[f"math.{name}"]
        monkeypatch.setitem(REGISTRY, f"math.{name}", dataclasses.replace(c, tested=False))
        r = prove(f"import math\nz = {_call(name, SAMPLE[name])}\n", ALL)
        assert r.verdict == "UNKNOWN" and "unsupported" in r.reason


# --------------------------------------------------------------------------- every function is modelled
ITERABLE_FUNCTIONS = {"fsum", "prod", "dist", "sumprod"}


@pytest.mark.parametrize("name", sorted(mm.modelled_names() - ITERABLE_FUNCTIONS))
class TestEveryFunction:
    def test_a_valid_call_proves_every_property_class(self, name):
        _true(f"import math\nz = {_call(name, SAMPLE[name])}\n")

    def test_every_property_class_is_recorded_at_the_call(self, name):
        r = prove(f"import math\nz = {_call(name, SAMPLE[name])}\n", ALL)
        assert {s[2] for s in r.sites if s[2].startswith(f"math.{name}:")} == {f"math.{name}:{e}" for e in ALL}

    def test_the_call_works_through_from_import_and_alias(self, name):
        _true(f"from math import {name} as f\nz = f({', '.join(_lit(a) for a in SAMPLE[name])})\n")

    def test_the_result_is_used_without_aborting(self, name):
        fn = mm.math_function(name)
        use = {"float": "z + 1.0", "bool": "z or False", "int": "z + 1", "float,int": "z[0] + 1.0",
               "float,float": "z[0] + z[1]"}[fn.result]
        _true(f"import math\nz = {_call(name, SAMPLE[name])}\nw = {use}\n")


@pytest.mark.parametrize("name", sorted(ITERABLE_FUNCTIONS))
def test_a_call_over_an_iterable_is_refused_not_guessed(name):
    r = prove(f"import math\nz = {_call(name, SAMPLE[name])}\n", ALL)
    assert r.verdict == "UNKNOWN" and "iterable" in r.reason and name in r.reason


# --------------------------------------------------------------------------- never TRUE
class TestRejectedCalls:
    """What CPython rejects is a TypeError, whatever the engine knows about the value."""

    @pytest.mark.parametrize("src", [
        "import math\nz = math.sqrt()\n",
        "import math\nz = math.sqrt(1.0, 2.0)\n",
        "import math\nz = math.sin(x=1.0)\n",
        "import math\nz = math.atan2(1.0)\n",
        "import math\nz = math.log(1.0, 2.0, 3.0)\n",
        "import math\nz = math.sqrt('x')\n",
        "import math\nz = math.sqrt(None)\n",
        "import math\nz = math.sqrt([1.0])\n",
        "import math\nz = math.factorial(5.0)\n",
        "import math\nz = math.isqrt(2.5)\n",
        "import math\nz = math.gcd(2.0, 4)\n",
        "import math\nz = math.ldexp(1.0, 2.0)\n",
        "import math\nz = math.comb(5, 2.0)\n",
        "import math\nz = math.trunc('x')\n",
    ])
    def test_never_true_for_dynamic_typing(self, src):
        _never_true(src, DT)

    @pytest.mark.parametrize("src", [
        "import _sv_verifier\nimport math\nx = _sv_verifier.nondet_bool()\nz = math.sqrt('x' if x else 1.0)\n",
        "import _sv_verifier\nimport math\nx = _sv_verifier.nondet_bool()\nz = math.factorial(5.0 if x else 5)\n",
    ])
    def test_a_branch_that_raises_keeps_the_site_open(self, src):
        _never_true(src, DT)

    def test_a_certain_typeerror_ends_the_path_for_the_other_properties(self):
        """sqrt('x') raises TypeError for certain: the division after it never runs (a clean run)."""
        _true("import math\nz = math.sqrt('x')\nw = 1 // 0\n", AR)

    def test_a_certain_typeerror_is_not_clean_for_dynamic_typing(self):
        _never_true("import math\nz = math.sqrt('x')\nw = 1 // 0\n", DT)

    @pytest.mark.parametrize("kw", ["rel_tol=0.1", "abs_tol=0.5"])
    def test_a_keyword_the_function_takes_is_not_modelled(self, kw):
        r = prove(f"import math\nz = math.isclose(1.0, 1.0, {kw})\n", ALL)
        assert r.verdict == "UNKNOWN" and "keyword" in r.reason

    def test_prod_start_is_a_keyword_the_function_takes(self):
        r = prove("import math\nz = math.prod([1, 2], start=3)\n", ALL)
        assert r.verdict == "UNKNOWN"


class TestLogZeroDivision:
    """``math.log(x, base)`` divides by ``log(base)``: ZeroDivisionError exactly when base is 1."""

    @pytest.mark.parametrize("src", [
        "import math\nz = math.log(2, 1)\n",
        "import math\nz = math.log(2.0, 1.0)\n",
        "import math\nz = math.log(2.0, True)\n",
        "import _sv_verifier\nimport math\nb = _sv_verifier.nondet_int()\nz = math.log(5.0, b)\n",
        "import _sv_verifier\nimport math\nb = _sv_verifier.nondet_int()\nif b >= 1:\n    z = math.log(5.0, b)\n",
        "import _sv_verifier\nimport math\nb = _sv_verifier.nondet_float()\nz = math.log(5.0, b)\n",
    ])
    def test_never_true_when_the_base_can_be_one(self, src):
        _never_true(src, AR)

    @pytest.mark.parametrize("src", [
        "import math\nz = math.log(2, 2)\n",
        "import math\nz = math.log(2.0, 10)\n",
        "import math\nz = math.log(2.0)\n",
        "import _sv_verifier\nimport math\nb = _sv_verifier.nondet_int()\nif b > 1:\n    z = math.log(5.0, b)\n",
        "import _sv_verifier\nimport math\nb = _sv_verifier.nondet_int()\nz = math.log(5.0, abs(b) + 2)\n",
    ])
    def test_proved_when_the_base_excludes_one(self, src):
        _true(src, AR)

    def test_the_other_classes_of_log_stay_proved(self):
        _true("import math\nz = math.log(2.0, 3)\n", ALL)


class TestFloorAndFriends:
    def test_floor_may_be_zero(self):
        _never_true("import _sv_verifier\nimport math\nx = _sv_verifier.nondet_float()\nz = 1 // math.floor(x)\n", AR)

    def test_floor_of_an_int_is_the_int_itself_so_it_can_index(self):
        _true("import _sv_verifier\nimport math\nn = abs(_sv_verifier.nondet_int()) % 10\n"
              "xs = [0] * 10\nv = xs[math.floor(n)]\n", DL)

    def test_floor_of_a_float_cannot_index_without_a_bound(self):
        _never_true("import _sv_verifier\nimport math\nx = _sv_verifier.nondet_float()\n"
                    "xs = [0] * 10\nv = xs[math.floor(x)]\n", DL)


class TestDivisionByAModelledResult:
    def test_fabs_alone_can_be_zero(self):
        _never_true("import _sv_verifier\nimport math\nx = _sv_verifier.nondet_float()\n"
                    "y = 1.0 / math.fabs(x)\n", AR)

    def test_exp_is_positive_but_may_underflow_to_zero(self):
        _never_true("import _sv_verifier\nimport math\nx = _sv_verifier.nondet_float()\n"
                    "y = 1.0 / math.exp(x)\n", AR)

    def test_cosh_is_at_least_one(self):
        _true("import _sv_verifier\nimport math\nx = _sv_verifier.nondet_float()\ny = 1.0 / math.cosh(x)\n", AR)

    def test_isqrt_plus_one_is_positive(self):
        _true("import _sv_verifier\nimport math\nn = _sv_verifier.nondet_int()\n"
              "if n >= 0:\n    z = 1 // (math.isqrt(n) + 1)\n", AR)

    def test_gcd_may_be_zero(self):
        _never_true("import math\nz = 1 // math.gcd(0, 0)\n", AR)

    def test_factorial_is_positive(self):
        _true("import math\nz = 1 // math.factorial(5)\n", AR)

    def test_the_constants_do_not_raise(self):
        _true("import math\nz = 1 / math.tau\nw = math.pi + math.e\nv = math.inf - math.nan\n", ALL)


# --------------------------------------------------------------------------- the closed ranges
def _abstract(name, *values):
    """The abstract result of ``math.<name>(*values)`` and the state it left, straight from the analyser."""
    import ast

    from tharkun.solve.absint.interp import Analyzer
    from tharkun.solve.absint.state import Frame, State

    an = Analyzer("pass", "<t>")
    st = State()
    st.frames.append(Frame("g"))
    node = ast.parse(f"math.{name}()").body[0].value
    return an, st, an._math_call(st, list(values), {}, node, name)


def _real_v(lo, hi, nan):
    from tharkun.solve.absint.aval import FloatInfo, V

    return V(f=FloatInfo(fv.FloatFact(lo, hi, nan)))


def _int_v(lo, hi):
    from tharkun.solve.absint.aval import V
    from tharkun.solve.absint.linear import IntInfo

    return V(i=IntInfo.interval(lo, hi))


def _holds(an, st, v, native) -> bool:
    """Does the abstract value ``v`` contain the native result?"""
    if isinstance(native, tuple):
        return True   # the pair's two items are checked through their own tests
    if isinstance(native, bool):
        return bool(v.b) and native in v.b or v.i is not None
    if isinstance(native, int):
        ii = an._as_int(st, v)
        if ii is None:
            return False
        lo, hi = an.ibounds(st, ii)
        return (lo is None or lo <= native) and (hi is None or native <= hi)
    return v.f is not None and v.f.fact.contains(native)


SAMPLE_POINTS = [0.0, -0.0, 0.5, -0.5, 1.0, -1.0, 2.0, 2.5, 3.0, 5.0, -5.0, 10.0, 100.0, 709.0, 710.0, -745.0,
                 1e-300, 1e300, -1e300, INF, -INF, math.nan]


@pytest.mark.parametrize("name", sorted(n for n in mm.modelled_names() - ITERABLE_FUNCTIONS
                                        if mm.math_function(n).args[0] == "real"))
class TestTheAbstractResultContainsTheNativeOne:
    """The soundness test of the model itself: for every function and every sample inside the abstract
    arguments, CPython's answer lies in the abstract answer (or CPython raises ValueError/OverflowError,
    which are not property classes and end the path)."""

    @given(data=st.data())
    @settings(max_examples=120, deadline=None)
    def test_over_intervals(self, name, data):
        fn = mm.math_function(name)
        arity = len(VALID[name])
        ranges, picks = [], []
        for i in range(arity):
            if fn.kind_of(i) == "int":
                lo = data.draw(st.integers(-20, 20))
                hi = lo + data.draw(st.integers(0, 20))
                ranges.append(_int_v(lo, hi))
                picks.append(data.draw(st.integers(lo, hi)))
                continue
            a = data.draw(st.floats(allow_nan=False, allow_infinity=True))
            b = data.draw(st.floats(allow_nan=False, allow_infinity=True))
            lo, hi = min(a, b), max(a, b)
            ranges.append(_real_v(lo, hi, False))
            picks.append(data.draw(st.one_of(st.floats(min_value=lo, max_value=hi, allow_nan=False), st.just(lo), st.just(hi))))
        an, state, v = _abstract(name, *ranges)
        try:
            native = getattr(math, name)(*picks)
        except (ValueError, OverflowError, ZeroDivisionError):
            return
        if state.bottom:
            pytest.fail(f"math.{name}{tuple(picks)} = {native!r} but the abstract call ended the path")
        assert _holds(an, state, v, native), (name, picks, native, v)

    def test_over_special_points_including_nan(self, name):
        fn = mm.math_function(name)
        arity = len(VALID[name])
        pool = SAMPLE_POINTS if arity <= 2 else SAMPLE_POINTS[::3]
        for picks in itertools.product(*[pool] * arity):
            if any(fn.kind_of(i) == "int" for i in range(arity)):
                picks = tuple(int(p) if fn.kind_of(i) == "int" and math.isfinite(p) and abs(p) < 1e6 else
                              (3 if fn.kind_of(i) == "int" else p) for i, p in enumerate(picks))
            vs = [_int_v(p, p) if fn.kind_of(i) == "int" else _real_v(p, p, math.isnan(p)) if not math.isnan(p)
                  else _real_v(INF, -INF, True) for i, p in enumerate(picks)]
            an, state, v = _abstract(name, *vs)
            try:
                native = getattr(math, name)(*picks)
            except (ValueError, OverflowError, ZeroDivisionError):
                continue
            assert not state.bottom, (name, picks, native)
            assert _holds(an, state, v, native), (name, picks, native, v)


@pytest.mark.parametrize("name", sorted(n for n in mm.modelled_names() - ITERABLE_FUNCTIONS
                                        if mm.math_function(n).args[0] == "int"))
class TestTheIntegerFunctionsContainTheNativeResult:
    @given(st.lists(st.integers(0, 12), min_size=1, max_size=3))
    @settings(max_examples=80, deadline=None)
    def test_over_small_ints(self, name, ns):
        fn = mm.math_function(name)
        arity = len(VALID[name]) if fn.max_args is not None else len(ns)
        picks = (ns * 3)[:arity]
        an, state, v = _abstract(name, *[_int_v(p, p) for p in picks])
        native = getattr(math, name)(*picks)
        assert not state.bottom and _holds(an, state, v, native), (name, picks, native)


@pytest.mark.parametrize("name", sorted(n for n in mm.modelled_names()
                                        if mm.math_function(n).int_min is not None))
def test_an_int_result_has_its_lower_bound(name):
    fn = mm.math_function(name)
    args = ", ".join(["abs(n) % 5"] * max(len(VALID[name]), 1)) if fn.max_args is None or fn.max_args > 1 else "abs(n) % 5"
    if name in ("perm", "comb"):
        args = "abs(n) % 5, abs(n) % 3"
    _true(f"import _sv_verifier\nimport math\nn = _sv_verifier.nondet_int()\nr = math.{name}({args})\n"
          f"assert r >= {fn.int_min}\n", AS)


def test_a_pair_result_unpacks():
    _true("import math\nm, e = math.frexp(8.0)\nf, i = math.modf(2.5)\nz = m + f + i\nw = e + 1\n")


def test_a_pair_of_the_wrong_length_ends_the_path_with_a_value_error():
    _true("import math\na, b, c = math.modf(2.5)\nassert False\n", AS)
    _never_true("import math\na, b = math.modf(2.5)\nassert False\n", AS)


# --------------------------------------------------------------------------- the transfer functions
def _contained(fact: fv.FloatFact, x: float) -> bool:
    return fact.contains(x)


FLOATS = st.floats(allow_nan=True, allow_infinity=True, width=64)


@given(lo=st.floats(allow_nan=False, allow_infinity=False, min_value=-800, max_value=800),
       width=st.floats(min_value=0, max_value=50), data=st.data())
@settings(max_examples=150, deadline=None)
@pytest.mark.parametrize("name", ["exp", "log", "log10", "atan", "tanh", "fabs"])
def test_the_unary_transfer_contains_the_native_result(name, lo, width, data):
    hi = lo + width
    x = data.draw(st.floats(min_value=lo, max_value=hi, allow_nan=False))
    res, raises = getattr(fv, name)(fv.FloatFact(lo, hi, False))
    try:
        r = getattr(math, name)(x)
    except ValueError:
        assert "ValueError" in raises or res.is_bottom(), (name, x)
        return
    except OverflowError:
        assert "OverflowError" in raises or res.is_bottom(), (name, x)
        return
    assert res.contains(r), (name, lo, hi, x, r, res)


@given(st.floats(min_value=-1e6, max_value=1e6), st.floats(min_value=0, max_value=1e3), st.data())
@settings(max_examples=150, deadline=None)
def test_the_binary_transfers_contain_the_native_result(lo, width, data):
    hi = lo + width
    x = data.draw(st.floats(min_value=lo, max_value=hi, allow_nan=False))
    y = data.draw(st.floats(min_value=lo, max_value=hi, allow_nan=False))
    a = fv.FloatFact(lo, hi, False)
    assert fv.atan2(a, a)[0].contains(math.atan2(x, y))
    assert fv.hypot(a, a)[0].contains(math.hypot(x, y))
    assert fv.hypot(a, a, a)[0].contains(math.hypot(x, y, x))


@given(st.floats(min_value=-1e15, max_value=1e15), st.floats(min_value=0, max_value=1e3), st.data())
@settings(max_examples=150, deadline=None)
def test_the_rounding_transfers_contain_the_native_result(lo, width, data):
    hi = lo + width
    x = data.draw(st.floats(min_value=lo, max_value=hi, allow_nan=False))
    a = fv.FloatFact(lo, hi, False)
    for name in ("floor", "ceil", "trunc"):
        res, _ = getattr(fv, name)(a)
        r = getattr(math, name)(x)
        assert (res.lo is None or res.lo <= r) and (res.hi is None or r <= res.hi), (name, lo, hi, x, r, res)
