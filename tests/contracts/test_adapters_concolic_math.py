"""Engine 3 (the symbolic executor) consumption of the ``math`` contracts ([internal ref] batch 5).

``sv/_symexec.py`` evaluated no call into ``math``: the statement and everything it assigned became
unmodelled.  ``sv/_symexec_math.py`` now reads the contracts through the table absint reads too
(``contracts/adapters/math_model.py``, via ``adapters/concolic.py``):

* every call on concrete numbers is FOLDED natively (``math`` is pure) -- pinned against CPython for
  every function; a call that raises is refused with the exception named, never dropped;
* the functions with an exact z3 encoding (:data:`ENCODED`) are encoded -- pinned against CPython on
  exactly representable samples;
* every other function of a symbolic argument (:data:`REFUSED`) says why it cannot be encoded, and the
  reason is what the engine reports for the sites behind it;
* the two sets together are every function of the module, and nothing else.
"""

from __future__ import annotations

import ast
import dataclasses
import fractions
import math
import sys
import textwrap

import pytest

from tests.contracts.test_math_module import VALID
from tharkun.solve.contracts.adapters import concolic as _concolic
from tharkun.solve.contracts.adapters import math_model as mm
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.sv.sites import build_harness

z3 = pytest.importorskip("z3")

from tharkun.sv import _symexec, _symexec_math  # noqa: E402

PRE = ("import _sv_verifier\nimport math\nfrom math import floor, tan\n\n"
       "if __name__ == '__main__':\n    n = _sv_verifier.nondet_int()\n    x = _sv_verifier.nondet_float()\n")


def _run(body, classes=("IndexError",)):
    src = PRE + textwrap.indent(textwrap.dedent(body), "    ")
    h = build_harness(src, classes, None)
    mod = {"__name__": "sv_harness_test"}
    sys.modules.setdefault("_sv_verifier", type(sys)("_sv_verifier"))
    exec(compile(h.source, "sv_harness.py", "exec"), mod)
    return list(_symexec.solve_targets(h.source, mod, "__sv_harness", [s["line"] for s in h.sites],
                                       h.params, h.kinds, {}).values())


def _sat(outs):
    return [m for o, m in outs if o == "sat"]


def _reasons(outs):
    return [m for o, m in outs if o == "unsupported"]


# --------------------------------------------------------------------------- the partition
class TestPartition:
    def test_encoded_and_refused_are_every_function_and_nothing_else(self):
        assert _symexec_math.ENCODED | set(_symexec_math.REFUSED) == mm.modelled_names()
        assert not _symexec_math.ENCODED & set(_symexec_math.REFUSED)

    def test_every_refusal_has_a_reason(self):
        assert all(len(r) > 10 for r in _symexec_math.REFUSED.values())

    @pytest.mark.parametrize("name", sorted(mm.modelled_names()))
    def test_the_adapter_gate(self, name):
        assert _concolic.math_function(name) is not None

    def test_the_constants(self):
        for name, value in mm.CONSTANTS.items():
            got = _concolic.math_constant(name)
            assert (math.isnan(got) and math.isnan(value)) or got == value

    @pytest.mark.parametrize("name", sorted(mm.modelled_names()))
    def test_withdrawing_the_contract_withdraws_the_model(self, monkeypatch, name):
        load_all()
        monkeypatch.delitem(REGISTRY, f"math.{name}")
        assert _concolic.math_function(name) is None

    def test_an_untested_contract_withdraws_the_model(self, monkeypatch):
        load_all()
        monkeypatch.setitem(REGISTRY, "math.floor", dataclasses.replace(REGISTRY["math.floor"], tested=False))
        assert _concolic.math_function("floor") is None
        assert _concolic.math_constant("pi") is not None


# --------------------------------------------------------------------------- concrete folds
def _fold_cases():
    for name in sorted(mm.modelled_names()):
        yield pytest.param(name, id=name)


@pytest.mark.parametrize("name", list(_fold_cases()))
class TestFold:
    def test_a_call_on_concrete_arguments_is_evaluated_like_cpython(self, name):
        native = getattr(math, name)(*VALID[name])
        if isinstance(native, tuple):
            first, second = native
            cond = f"a == {first!r} and b == {second!r}"
            pre = f"    a, b = math.{name}({', '.join(repr(a) for a in VALID[name])})\n"
        else:
            if isinstance(native, float) and not math.isfinite(native):
                pytest.skip("a non-finite result has no literal")
            cond = f"r == {native!r}"
            pre = f"    r = math.{name}({', '.join(repr(a) for a in VALID[name])})\n"
        sat = _sat(_run(pre + f"    if {cond} and n == 3:\n        raise IndexError('hit')\n"))
        assert sat and sat[0]["n"] == 3, (name, native)

    def test_a_wrong_answer_is_not_reached(self, name):
        native = getattr(math, name)(*VALID[name])
        if isinstance(native, tuple) or isinstance(native, float) and not math.isfinite(native):
            pytest.skip("covered by the right-answer case")
        wrong = (native + 1) if not isinstance(native, bool) else (not native)
        out = _run(f"    r = math.{name}({', '.join(repr(a) for a in VALID[name])})\n"
                   f"    if r == {wrong!r} and n == 3:\n        raise IndexError('hit')\n")
        assert not _sat(out)


class TestFoldEdges:
    def test_a_name_bound_by_from_import_folds(self):
        assert _sat(_run("    if floor(2.5) == 2 and tan(0.0) == 0.0 and n == 3:\n        raise IndexError\n"))

    def test_the_constants(self):
        assert _sat(_run("    if math.floor(math.pi * 100) == 314 and math.tau == 2 * math.pi and n == 3:\n"
                         "        raise IndexError\n"))
        assert not _sat(_run("    if math.e == 3.0 and n == 3:\n        raise IndexError\n"))

    @pytest.mark.parametrize("call,exc", [
        ("math.sqrt(-1.0)", "ValueError"), ("math.exp(710.0)", "OverflowError"),
        ("math.log(0.0)", "ValueError"), ("math.factorial(-1)", "ValueError"),
        ("math.log(2, 1)", "ZeroDivisionError"), ("math.acos(2.0)", "ValueError"),
    ])
    def test_a_call_that_raises_is_refused_with_the_exception_named(self, call, exc):
        out = _run(f"    if {call} > 0:\n        raise IndexError\n")
        assert not _sat(out)
        assert any(exc in r for r in _reasons(out)), out

    def test_a_wrong_argument_count_is_refused(self):
        out = _run("    if math.sqrt(1.0, 2.0) > 0:\n        raise IndexError\n")
        assert not _sat(out) and any("TypeError" in r for r in _reasons(out))

    def test_a_keyword_the_function_takes_is_refused_as_a_keyword(self):
        out = _run("    if math.isclose(1.0, 1.0, rel_tol=0.1):\n        raise IndexError\n")
        assert not _sat(out) and any("keyword" in r for r in _reasons(out))

    def test_a_huge_factorial_is_not_computed_in_the_solver(self):
        out = _run("    if math.factorial(10 ** 7) > 0:\n        raise IndexError\n")
        assert not _sat(out) and any("too large" in r for r in _reasons(out))

    def test_a_non_number_argument_is_refused(self):
        out = _run("    if math.sqrt('4') > 0:\n        raise IndexError\n")
        assert not _sat(out) and _reasons(out)


class TestWhatCountsAsMath:
    """The callee is resolved by identity in the harness module's globals."""

    def _exe(self, g, env=None):
        return _symexec._Exec(g, env or {}, {})

    def _callee(self, src):
        return ast.parse(src, mode="eval").body.func

    def test_the_module_attribute_and_the_imported_name(self):
        exe = self._exe({"math": math, "floor": math.floor, "alias": math.tan})
        assert exe._math_function(self._callee("math.sqrt(1)")) == "sqrt"
        assert exe._math_function(self._callee("floor(1)")) == "floor"
        assert exe._math_function(self._callee("alias(1)")) == "tan"

    def test_a_local_or_a_program_function_is_never_math(self):
        exe = self._exe({"math": math, "floor": math.floor}, env={"floor": 1, "math": 2})
        assert exe._math_function(self._callee("floor(1)")) is None
        assert exe._math_function(self._callee("math.sqrt(1)")) is None
        mine = self._exe({"sqrt": lambda v: v, "math": object()})
        assert mine._math_function(self._callee("sqrt(1)")) is None
        assert mine._math_function(self._callee("math.sqrt(1)")) is None

    def test_a_function_with_the_name_of_a_math_one_is_not_math(self):
        def floor(v):
            return v

        assert self._exe({"floor": floor})._math_function(self._callee("floor(1)")) is None

    def test_a_constant_of_the_module(self):
        exe = self._exe({"math": math})
        assert exe.ev(ast.parse("math.pi", mode="eval").body) == math.pi
        with pytest.raises(_symexec.Unsupported):
            exe.ev(ast.parse("math.nosuch", mode="eval").body)
        with pytest.raises(_symexec.Unsupported):
            self._exe({}).ev(ast.parse("math.pi", mode="eval").body)


# --------------------------------------------------------------------------- the z3 encodings
class _Exe:
    """The part of ``_Runner`` the encodings use: path conditions and fresh variables."""

    def __init__(self):
        self.conditions = []
        self.seq = 0

    def _holds(self, cond):
        self.conditions.append(cond)

    def _contract_var_name(self, qualname):
        self.seq += 1
        return f"_v_{self.seq}"


def _value_of(term):
    if isinstance(term, (bool, int, float)):
        return term
    t = z3.simplify(term)
    if z3.is_true(t):
        return True
    if z3.is_false(t):
        return False
    if z3.is_int_value(t):
        return t.as_long()
    if z3.is_rational_value(t):
        return fractions.Fraction(t.numerator_as_long(), t.denominator_as_long())
    raise AssertionError(f"not a numeral: {t}")


def _sym(v):
    """A z3 constant for a Python number: ints stay Int, floats are exact rationals."""
    if isinstance(v, bool):
        return z3.IntVal(int(v))
    if isinstance(v, int):
        return z3.IntVal(v)
    return z3.RealVal(fractions.Fraction(v))


DYADIC = [fractions.Fraction(k, 8) for k in (-37, -16, -9, -8, -1, 0, 1, 3, 8, 11, 16, 29, 40)]
INTS = [-9, -4, -1, 0, 1, 2, 5, 12]


def _equal(got, want):
    if isinstance(want, bool):
        return got == want
    return fractions.Fraction(got) == fractions.Fraction(want)


class TestEncodingsAgreeWithCpython:
    """Each exact encoding, evaluated on constants, equals CPython's answer."""

    @pytest.mark.parametrize("name", ["floor", "ceil", "trunc"])
    @pytest.mark.parametrize("x", DYADIC + INTS)
    def test_rounding(self, name, x):
        want = getattr(math, name)(float(x) if not isinstance(x, int) else x)
        arg = z3.RealVal(x) if isinstance(x, fractions.Fraction) else _sym(x)
        got = _value_of(_symexec_math._encode(_Exe(), name, mm.math_function(name), [arg]))
        assert got == want

    @pytest.mark.parametrize("x", DYADIC + INTS)
    def test_fabs(self, x):
        want = math.fabs(float(x))
        arg = z3.RealVal(x) if isinstance(x, fractions.Fraction) else _sym(x)
        assert _equal(_value_of(_symexec_math._encode(_Exe(), "fabs", mm.math_function("fabs"), [arg])), want)

    @pytest.mark.parametrize("name,want", [("isnan", False), ("isinf", False), ("isfinite", True)])
    def test_the_predicates_of_an_int(self, name, want):
        assert _value_of(_symexec_math._encode(_Exe(), name, mm.math_function(name), [z3.IntVal(5)])) is want

    @pytest.mark.parametrize("name", ["isnan", "isinf", "isfinite", "copysign"])
    def test_a_real_has_no_nan_infinity_or_signed_zero(self, name):
        args = [z3.RealVal(1)] if name != "copysign" else [z3.RealVal(1), z3.RealVal(0)]
        with pytest.raises(_symexec.Unsupported):
            _symexec_math._encode(_Exe(), name, mm.math_function(name), args)

    @pytest.mark.parametrize("x", DYADIC + INTS)
    @pytest.mark.parametrize("y", [-5, -2, -1, 1, 3, 7])
    def test_copysign_with_an_int_sign(self, x, y):
        arg = z3.RealVal(x) if isinstance(x, fractions.Fraction) else _sym(x)
        got = _value_of(_symexec_math._encode(_Exe(), "copysign", mm.math_function("copysign"), [arg, _sym(y)]))
        assert _equal(got, math.copysign(float(x), y))

    @pytest.mark.parametrize("name", ["fmod", "remainder"])
    @pytest.mark.parametrize("x", DYADIC + INTS)
    @pytest.mark.parametrize("y", [fractions.Fraction(3, 2), fractions.Fraction(-5, 4), 2, -3, 7, fractions.Fraction(1, 8)])
    def test_fmod_and_remainder(self, name, x, y):
        ax = z3.RealVal(x) if isinstance(x, fractions.Fraction) else _sym(x)
        ay = z3.RealVal(y) if isinstance(y, fractions.Fraction) else _sym(y)
        got = _value_of(_symexec_math._encode(_Exe(), name, mm.math_function(name), [ax, ay]))
        assert _equal(got, getattr(math, name)(float(x), float(y)))

    @pytest.mark.parametrize("name", ["fmod", "remainder"])
    def test_a_zero_divisor_is_a_domain_error_that_ends_the_path(self, name):
        exe = _Exe()
        _symexec_math._encode(exe, name, mm.math_function(name), [z3.RealVal(1), z3.RealVal(0)])
        s = z3.Solver()
        s.add(*exe.conditions)
        assert s.check() == z3.unsat

    @pytest.mark.parametrize("x,y,w", [(1, 2, 3), (fractions.Fraction(3, 2), fractions.Fraction(-5, 4), 7),
                                       (-4, 3, fractions.Fraction(1, 8))])
    def test_fma(self, x, y, w):
        args = [z3.RealVal(v) if isinstance(v, fractions.Fraction) else _sym(v) for v in (x, y, w)]
        got = _value_of(_symexec_math._encode(_Exe(), "fma", mm.math_function("fma"), args))
        assert _equal(got, math.fma(float(x), float(y), float(w)))

    @pytest.mark.parametrize("a,b", [(1.0, 1.0), (1.0, 1.0000000005), (1.0, 1.000000002), (0.0, 1e-12), (100.0, 100.00000005),
                                     (5, 5), (3, 4), (-1.0, -1.0000000001)])
    def test_isclose_with_the_default_tolerances(self, a, b):
        got = _value_of(_symexec_math._encode(_Exe(), "isclose", mm.math_function("isclose"), [_sym(a), _sym(b)]))
        assert got is math.isclose(a, b)

    @pytest.mark.parametrize("x", DYADIC + INTS)
    def test_modf(self, x):
        arg = z3.RealVal(x) if isinstance(x, fractions.Fraction) else _sym(x)
        frac, whole = _symexec_math._encode(_Exe(), "modf", mm.math_function("modf"), [arg])
        want = math.modf(float(x))
        assert _equal(_value_of(frac), want[0]) and _equal(_value_of(whole), want[1])

    @pytest.mark.parametrize("x", DYADIC + [3, -2, 0])
    @pytest.mark.parametrize("y", [0, 1, 2, 3, 8, -1, -2])
    def test_pow_with_a_small_integer_exponent(self, x, y):
        if x == 0 and y < 0:
            return
        arg = z3.RealVal(x) if isinstance(x, fractions.Fraction) else _sym(x)
        got = _value_of(_symexec_math._encode(_Exe(), "pow", mm.math_function("pow"), [arg, y]))
        want = math.pow(float(x), y)
        if y >= 0:
            assert _equal(got, want)       # exact: only products of dyadics
        else:
            assert math.isclose(float(got), want, rel_tol=1e-12)   # a reciprocal is rounded by the float

    @pytest.mark.parametrize("name,args", [("pow", [3.0, z3.RealVal(0)]), ("pow", [z3.RealVal(2), 0.5]),
                                           ("pow", [z3.RealVal(2), 99]), ("ldexp", [z3.RealVal(1), z3.IntVal(1)]),
                                           ("ldexp", [z3.RealVal(1), 5000])])
    def test_pow_and_ldexp_refuse_a_symbolic_fractional_or_huge_exponent(self, name, args):
        with pytest.raises(_symexec.Unsupported):
            _symexec_math._encode(_Exe(), name, mm.math_function(name), args)

    @pytest.mark.parametrize("x", DYADIC)
    @pytest.mark.parametrize("i", [-3, 0, 1, 10])
    def test_ldexp_with_a_concrete_shift(self, x, i):
        got = _value_of(_symexec_math._encode(_Exe(), "ldexp", mm.math_function("ldexp"), [z3.RealVal(x), i]))
        assert _equal(got, math.ldexp(float(x), i))

    @pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 8, 15, 16, 17, 24, 25, 99, 100, 10 ** 6])
    def test_isqrt_is_defined_by_its_bounds(self, n):
        exe = _Exe()
        r = _symexec_math._encode(exe, "isqrt", mm.math_function("isqrt"), [z3.IntVal(n)])
        s = z3.Solver()
        s.add(*exe.conditions)
        assert s.check() == z3.sat
        assert s.model().eval(r).as_long() == math.isqrt(n)
        s.add(r != math.isqrt(n))
        assert s.check() == z3.unsat      # unique

    def test_isqrt_of_a_negative_ends_the_path(self):
        exe = _Exe()
        _symexec_math._encode(exe, "isqrt", mm.math_function("isqrt"), [z3.IntVal(-1)])
        s = z3.Solver()
        s.add(*exe.conditions)
        assert s.check() == z3.unsat

    def test_gcd_with_zeros_and_at_most_one_unknown(self):
        assert _value_of(_symexec_math._encode(_Exe(), "gcd", mm.math_function("gcd"), [z3.IntVal(-6), 0])) == 6
        with pytest.raises(_symexec.Unsupported):
            _symexec_math._encode(_Exe(), "gcd", mm.math_function("gcd"), [z3.IntVal(6), 4])
        with pytest.raises(_symexec.Unsupported):
            _symexec_math._encode(_Exe(), "gcd", mm.math_function("gcd"), [z3.Int("a"), z3.Int("b")])

    def test_fsum_prod_and_sumprod_over_a_list_with_a_symbolic_element(self):
        a = z3.Real("a")
        s = z3.Solver()
        s.add(a * 2 == 3)
        fs = _symexec_math._encode(_Exe(), "fsum", mm.math_function("fsum"), [[a, 1.0, 2]])
        pr = _symexec_math._encode(_Exe(), "prod", mm.math_function("prod"), [[a, 4]])
        sp = _symexec_math._encode(_Exe(), "sumprod", mm.math_function("sumprod"), [[a, 1], [2, 3]])
        assert s.check() == z3.sat
        m = s.model()
        assert _equal(_value_of(m.eval(fs, model_completion=True)), math.fsum([1.5, 1.0, 2]))
        assert _equal(_value_of(m.eval(pr, model_completion=True)), math.prod([1.5, 4]))
        assert _equal(_value_of(m.eval(sp, model_completion=True)), math.sumprod([1.5, 1], [2, 3]))

    def test_sumprod_of_unequal_lengths_is_refused(self):
        with pytest.raises(_symexec.Unsupported):
            _symexec_math._encode(_Exe(), "sumprod", mm.math_function("sumprod"), [[z3.Int("a")], [1, 2]])

    @pytest.mark.parametrize("name", ["fmod", "isqrt"])
    def test_a_partial_function_needs_the_whole_harness_executor(self, name):
        class Slice:   # the slice executor has no path conditions
            pass

        args = [z3.Real("a"), z3.Real("b")] if name == "fmod" else [z3.Int("a")]
        with pytest.raises(_symexec.Unsupported):
            _symexec_math._encode(Slice(), name, mm.math_function(name), args)


# --------------------------------------------------------------------------- the engine end to end
class TestSolved:
    @pytest.mark.parametrize("body,check", [
        ("if math.floor(x) == 7:\n    raise IndexError\n", lambda m: math.floor(m["x"]) == 7),
        ("if math.ceil(x) == 7:\n    raise IndexError\n", lambda m: math.ceil(m["x"]) == 7),
        ("if math.trunc(x) == -3:\n    raise IndexError\n", lambda m: math.trunc(m["x"]) == -3),
        ("if math.fabs(x) > 3.5 and x < 0:\n    raise IndexError\n", lambda m: m["x"] < -3.5),
        ("if math.isqrt(n) == 12:\n    raise IndexError\n", lambda m: math.isqrt(m["n"]) == 12),
        ("if math.fmod(x, 3.0) == 1.0 and x > 10:\n    raise IndexError\n", lambda m: math.fmod(m["x"], 3.0) == 1.0),
        ("if math.remainder(x, 4.0) == -1.0 and x > 10:\n    raise IndexError\n", lambda m: math.remainder(m["x"], 4.0) == -1.0),
        ("if math.copysign(2.0, n) < 0:\n    raise IndexError\n", lambda m: m["n"] < 0),
        ("if math.isclose(x, 1.0) and x != 1.0:\n    raise IndexError\n", lambda m: math.isclose(m["x"], 1.0) and m["x"] != 1.0),
        ("f, i = math.modf(x)\nif i == 5.0 and f > 0.5:\n    raise IndexError\n", lambda m: 5.5 < m["x"] < 6.0),
        ("if math.pow(n, 2) == 49.0:\n    raise IndexError\n", lambda m: m["n"] in (7, -7)),
        ("if math.ldexp(x, 3) == 16.0:\n    raise IndexError\n", lambda m: m["x"] == 2.0),
        ("if math.fma(x, 2.0, 1.0) == 7.0:\n    raise IndexError\n", lambda m: m["x"] == 3.0),
        ("if math.gcd(0, n) == 7:\n    raise IndexError\n", lambda m: abs(m["n"]) == 7),
        ("if math.fsum([x, 1.0]) == 3.0:\n    raise IndexError\n", lambda m: m["x"] == 2.0),
        ("if math.prod([n, 3]) == 12:\n    raise IndexError\n", lambda m: m["n"] == 4),
    ])
    def test_a_guard_on_the_result_is_solved_and_the_witness_holds_natively(self, body, check):
        sat = _sat(_run(body))
        assert sat and check(sat[0]), sat

    @pytest.mark.parametrize("name", sorted(_symexec_math.REFUSED))
    def test_a_refused_function_reports_its_reason_for_the_site_behind_it(self, name):
        fn = mm.math_function(name)
        arity = max(fn.min_args, 1) if fn.max_args is None else fn.max_args
        if "iterable" in fn.args:
            args = ", ".join(["[x, 1.0]"] * arity)
        else:
            args = ", ".join(("n" if fn.kind_of(i) in ("int", "intopt") else "x") for i in range(arity))
        use = {"float": "> 0", "bool": "", "int": "> 0", "float,int": "[0] > 0", "float,float": "[0] > 0",
               "number": "> 0"}[fn.result]
        out = _run(f"if math.{name}({args}){use}:\n    raise IndexError\n")
        assert not _sat(out)
        assert any(f"math.{name}" in r and "no exact encoding" in r for r in _reasons(out)), (name, out)

    def test_a_site_behind_a_refusal_is_reported_unsupported_not_unsat(self):
        out = _run("if math.sqrt(x) > 3.5:\n    raise IndexError\n")
        assert [o for o, _ in out] == ["unsupported"]

    def test_a_site_that_needs_the_refused_value_is_not_closed(self):
        from tharkun.sv.concolic import Solved

        assert not Solved(ran=True, sites={5: "unsupported"}).closed
