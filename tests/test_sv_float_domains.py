"""The float draw domains of the child model ([internal ref]): ``float_range_producible`` / ``float_producible`` / ``snap_float_range``.

The sv-benchmarks helper documents ``nondet_float_range(a, b)`` as "a float in [a, b]" and implements it as ``random.uniform(a, b)`` =
``a + (b - a) * random()``, with ``random()`` = ``k * 2**-53`` for an integer ``0 <= k < 2**53``.  Owner ruling (2026-10-06): a TRUE is
proven over the larger (documented) reading, a FALSE is confirmed only with a value the CODE can produce.  ``_child.float_range_producible``
decides "the code can produce ``v``" EXACTLY, in IEEE double arithmetic, by searching the monotone value set; these tests hold it to the
native ``random.Random.uniform`` itself (a subclass whose ``random()`` returns a chosen ``k * 2**-53``), to a brute-force reference on small
monotone functions, and (when the grading harness is around) to the harness shim's copy.  No generated programs: tables and seeded draws.
"""

from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

import pytest

from tharkun.sv import _child as C

R = 2.0 ** -53
TOP = 1.0 - R


class Draw(random.Random):
    """``random()`` is a chosen ``k * 2**-53``: ``uniform`` is then CPython's own, so its answer is a value the helper returns."""

    def __init__(self, k):
        super().__init__(0)
        self.k = k

    def random(self):
        return self.k * R


def native(a, b, k):
    return Draw(k).uniform(a, b)


BOUNDS = [(0.0, 1.0), (1.0, 0.0), (5, 10), (10, 5), (-1.0, 1.0), (-3.7, 123456.789), (1e10, 1e10 + 1.0), (0.1, 0.7), (-1.5, 2.5), (2.5, 2.5),
          (0, 0), (-0.0, 0.0), (1e-5, 3e-5), (-1e300, 1e300), (1e308, 1.7e308), (True, False), (0.0, True), (5e-324, 1e-323), (-5e-324, 5e-324),
          (1e16, 1e16 + 2.0), (-1e-300, 1e-300), (123456789012345678, 123456789012345679)]
KS = [0, 1, 2, 3, 2 ** 20, 2 ** 52 - 1, 2 ** 52, 2 ** 52 + 1, 2 ** 53 - 2, 2 ** 53 - 1, 987654321987, 4503599627370497, 3 * 2 ** 50]


@pytest.mark.parametrize("a, b", BOUNDS)
def test_every_value_the_native_uniform_returns_is_producible(a, b):
    """Completeness: the control is ``random.uniform`` itself."""
    for k in KS:
        v = native(a, b, k)
        assert C.float_range_producible(a, b, v), (a, b, k, v)


def test_seeded_random_draws_are_all_producible():
    rng = random.Random(1159)
    for _ in range(3000):
        a = rng.choice([0.0, -1.0, 1e-5, -3.7, 1e10, 2.5, 123456.789, -2.0 ** 60, 7])
        b = a + rng.choice([1.0, 7.25, 1e-3, -2.0, 1e12, 0.0, -1e-9])
        k = rng.randrange(1 << 53) if rng.random() < 0.7 else rng.choice(KS)
        assert C.float_range_producible(a, b, native(a, b, k)), (a, b, k)


NOT_PRODUCIBLE = [
    (0.0, 1.0, 1.0), (0.0, 1.0, -0.0), (0.0, 1.0, 1e-300), (0.0, 1.0, 5e-324), (0.0, 1.0, 0.1), (0.0, 1.0, 0.3), (0.0, 1.0, 2.0),
    (0.0, 1.0, -1e-300), (0.0, 1.0, math.nan), (0.0, 1.0, math.inf), (0.0, 1.0, 1 + 2 ** -52), (0.0, 1.0, 2.0 ** -54),
    (5, 10, 4.999999999999999), (5, 10, 10.000000000000002), (5, 10, 0.0), (-1.0, 1.0, 1.0), (-1.0, 1.0, -1.0 - 2 ** -52),
    (1e308, 1.7e308, 1.8e308), (-1e300, 1e300, 1e300), (-1e308, 1e308, 1e308), (-1e308, 1e308, 0.0), (-1e308, 1e308, -math.inf),
    (2.5, 2.5, 2.5000000000000004), (2.5, 2.5, -2.5),
]


@pytest.mark.parametrize("a, b, v", NOT_PRODUCIBLE)
def test_a_value_the_helper_cannot_return_is_not_producible(a, b, v):
    assert not C.float_range_producible(a, b, v), (a, b, v)


def test_the_bound_b_is_producible_exactly_when_the_rounding_reaches_it():
    """``uniform(0.0, 1.0)`` never returns ``1.0`` (its largest value is ``1 - 2**-53``), ``uniform(5, 10)`` does return ``10.0``
    (``5 + 5 * (1 - 2**-53)`` rounds to it): the Python docs say the end point "may or may not be included depending on rounding"."""
    assert native(0.0, 1.0, 2 ** 53 - 1) == TOP and not C.float_range_producible(0.0, 1.0, 1.0)
    assert native(5, 10, 2 ** 53 - 1) == 10.0 and C.float_range_producible(5, 10, 10.0)
    assert not C.float_range_producible(0.0, 1.0, 1.0) and C.float_range_producible(0.0, 1.0, TOP)


def test_an_overflowing_or_nan_span_produces_only_nan_and_inf():
    """``b - a`` overflows: ``nan`` for ``k == 0`` (``inf * 0.0``) and ``inf`` / ``-inf`` / ``nan`` for every other ``k``."""
    for a, b, expect in ((-1e308, 1e308, {"nan", "inf"}), (1e308, -1e308, {"nan", "-inf"}), (-sys.float_info.max, sys.float_info.max, {"nan", "inf"}),
                         (-math.inf, math.inf, {"nan"}), (math.inf, math.inf, {"nan"}), (0.0, math.nan, {"nan"}), (math.nan, 1.0, {"nan"}),
                         (0.0, math.inf, {"nan", "inf"}), (-math.inf, 0.0, {"nan"})):
        got = {repr(native(a, b, k)) for k in (0, 1, 2 ** 52, 2 ** 53 - 1)}
        assert got == expect, (a, b, got)
        for v in (math.inf, -math.inf, math.nan, 0.0, 1.0, 1e308):
            produced = repr(v) in expect
            assert C.float_range_producible(a, b, v) == produced, (a, b, v)


def test_a_zero_span_returns_a_and_signed_zeros_are_distinct():
    assert C.float_range_producible(2.5, 2.5, 2.5) and not C.float_range_producible(2.5, 2.5, 2.4999999999999996)
    # -0.0 - -0.0 is +0.0, and -0.0 + 0.0 is +0.0: a zero span returns +0.0 even for a = -0.0
    assert native(-0.0, -0.0, 5) == 0.0 and math.copysign(1.0, native(-0.0, -0.0, 5)) > 0
    assert C.float_range_producible(-0.0, -0.0, 0.0) and not C.float_range_producible(-0.0, -0.0, -0.0)
    assert native(0.0, 0.0, 5) == 0.0 and C.float_range_producible(0.0, 0.0, 0.0) and not C.float_range_producible(0.0, 0.0, -0.0)
    assert not C.float_range_producible(0.0, 1.0, -0.0)           # 0.0 + 1.0 * 0.0 is +0.0
    # a = -0.0 and a negative span: -0.0 + (-5e-324 * 0.0 = -0.0) is -0.0 (k == 0), then the tiny negative values
    assert math.copysign(1.0, native(-0.0, -5e-324, 0)) < 0 and C.float_range_producible(-0.0, -5e-324, -0.0)
    assert not C.float_range_producible(-0.0, -5e-324, 0.0)


def test_an_int_beyond_the_float_range_is_an_overflow_error_in_the_helper_so_no_value():
    with pytest.raises(OverflowError):
        random.uniform(1, 10 ** 400)
    assert not C.float_range_producible(1, 10 ** 400, 5.0)
    assert not C.float_range_producible(10 ** 400, 1, 5.0)


def test_plain_numbers_only_user_bounds_keep_the_documented_interval():
    class B:
        def __init__(self, v):
            self.v = v

        def __lt__(self, o):
            return self.v < o.v

        def __gt__(self, o):
            return self.v > o.v

    assert C.float_range_valid(0.0, 1.0, 0.5) and not C.float_range_valid(0.0, 1.0, 1.0)
    # an object bound: the documented closed interval (the formula would run user arithmetic dozens of times)
    assert C.float_range_valid(0, 1, 1.0) is False
    assert not C.float_range_valid(True, 2.5, math.nan)


# ------------------------------------------------------------------ the search itself, against brute force

def _brute(f, nk):
    return {(math.copysign(1.0, f(k)), f(k)) if f(k) == 0 else (None, f(k)) for k in range(nk)}


@pytest.mark.parametrize("nondecreasing", [True, False])
def test_the_monotone_search_agrees_with_enumeration_on_small_monotone_functions(nondecreasing):
    rng = random.Random(7 if nondecreasing else 8)
    for _ in range(300):
        nk = rng.randrange(1, 40)
        steps = sorted(rng.choice([-3.0, -1.5, 0.0, 0.0, 0.5, 2.0, 2.0, 7.25]) for _ in range(nk))
        values = steps if nondecreasing else steps[::-1]
        f = values.__getitem__
        image = set(values)
        for v in set(values) | {-10.0, 10.0, 0.25, 1.75, 2.0000001}:
            assert C._monotone_image_contains(f, nk, v, nondecreasing) == (v in image), (values, v, nondecreasing)


def test_the_monotone_search_reads_zero_signs_exactly():
    """The zero of ``a + (b - a) * r`` has one sign for a whole ``uniform`` call (it is ``-0.0`` only for ``a = -0.0`` and a negative
    span, where every product is ``-0.0`` or negative, else ``+0.0``: ``x + -x`` is ``+0.0``), so a monotone ``f`` never mixes the two."""
    f = [0.0, 0.0, 0.0, 1.0].__getitem__
    assert C._monotone_image_contains(f, 4, 0.0, True) and not C._monotone_image_contains(f, 4, -0.0, True)
    g = [-0.0, -0.0, -0.0, -1.0][::-1].__getitem__
    assert C._monotone_image_contains(g, 4, -0.0, True) and not C._monotone_image_contains(g, 4, 0.0, True)


# ------------------------------------------------------------------ nondet_float

def test_nondet_float_takes_only_the_four_values_of_the_helper():
    for v in (math.inf, -math.inf, math.nan, -0.0):
        assert C.float_producible(v)
    for v in (0.0, 1.0, -1.0, 1e300, -1e300, 5e-324, sys.float_info.max, -sys.float_info.max):
        assert not C.float_producible(v)


def test_the_second_branch_of_nondet_float_is_the_overflowing_uniform():
    """``random.uniform(-max, max)`` is ``inf`` or ``nan`` natively, which is the ``float_range_producible`` set of that call."""
    m = sys.float_info.max
    assert {repr(native(-m, m, k)) for k in KS} == {"nan", "inf"}
    for v in (math.inf, math.nan):
        assert C.float_range_producible(-m, m, v) and C.float_producible(v)
    assert not C.float_range_producible(-m, m, 0.0) and not C.float_range_producible(-m, m, -math.inf)


# ------------------------------------------------------------------ snap_float_range

def test_a_snapped_proposal_is_always_producible():
    rng = random.Random(3)
    for _ in range(600):
        a = rng.choice([0.0, -1.0, 2.5, 1e-5, 123.456, 5, 0])
        b = rng.choice([1.0, 7.25, -3.0, 1e3, 124.0, 10])
        for v in (rng.uniform(-10, 10), 1e300, -1e300, rng.random(), math.nan, math.inf, -math.inf, 0.0, a, b):
            s = C.snap_float_range(v, a, b)
            assert C.float_range_producible(a, b, s), (a, b, v, s)


def test_a_huge_proposal_snaps_to_the_largest_value_not_to_the_bound():
    assert C.snap_float_range(4.15e300, 0.0, 1.0) == TOP
    assert C.snap_float_range(-4.15e300, 0.0, 1.0) == 0.0
    assert C.snap_float_range(0.5, 0.0, 1.0) == 0.5
    assert C.snap_float_range(math.nan, 0.0, 1.0) == 0.0
    assert C.snap_float_range(1e300, 1.0, 0.0) == 1.0                  # a > b: the domain runs from 1.0 down
    assert C.snap_float_range(-1e300, 1.0, 0.0) == native(1.0, 0.0, 2 ** 53 - 1)
    assert C.snap_float_range(1.0, 0.0, 1.0) == TOP


def test_a_snap_is_the_first_producible_value_at_or_past_the_proposal():
    for v in (0.1, 0.3, 0.7, 1e-17, 0.99):
        s = C.snap_float_range(v, 0.0, 1.0)
        assert s >= v and s - v < 2.0 ** -52 and C.float_range_producible(0.0, 1.0, s)
        assert not C.float_range_producible(0.0, 1.0, math.nextafter(v, 0.0)) or s == v or math.nextafter(v, 0.0) < v <= s


def test_an_overflowing_span_snaps_to_inf_or_nan():
    assert C.snap_float_range(1e300, -1e308, 1e308) == math.inf
    assert C.snap_float_range(math.nan, -1e308, 1e308) != C.snap_float_range(math.nan, -1e308, 1e308)   # nan stays nan
    assert C.snap_float_range(math.inf, -1e308, 1e308) == math.inf
    assert C.snap_float_range(0.5, -math.inf, math.inf) != 0.5                                                # the only value is nan


def test_user_object_bounds_are_refused_by_the_proposal_generator():
    class B(float):
        pass

    with pytest.raises(ValueError, match="bounds are not plain numbers"):
        C.snap_float_range(5.0, B(0.0), B(1.0))


# ------------------------------------------------------------------ the child and the harness shim agree

def _harness_import_hooks():
    import _thread
    import random as current_random
    import threading

    from tharkun.sv.nondet_harness import _RANDOM_NAMES

    for obj, names in ((current_random, _RANDOM_NAMES),
                       (current_random.Random, ("seed",)),
                       (current_random.SystemRandom, ("random", "getrandbits", "randbytes")),
                       (threading.Thread, ("start",)),
                       (_thread, C._THREAD_STARTERS),
                       (threading, C._THREAD_STARTERS)):
        for name in names:
            if hasattr(obj, name):
                yield obj, name, getattr(obj, name)


def _harness_shim():
    here = Path(__file__).resolve()
    for c in (os.environ.get("EREGION_SV_SRC"), str(here.parents[2] / "eregion-sv" / "src")):
        if c and (Path(c) / "eregion_sv" / "shim" / "_sv_verifier.py").is_file():
            # The replay shim installs process-wide hooks on import; this comparison
            # needs only its predicates, so preserve the collection process exactly.
            with pytest.MonkeyPatch.context() as restore:
                for obj, name, original in _harness_import_hooks():
                    restore.setattr(obj, name, original)
                restore.setattr(sys, "path", [c, *sys.path])
                from eregion_sv.shim import _sv_verifier as shim
                return shim
    return None


SHIM = _harness_shim()


@pytest.mark.skipif(SHIM is None, reason="needs the grading harness")
def test_importing_the_harness_predicates_restores_all_process_hooks(monkeypatch):
    saved = list(_harness_import_hooks())
    # Exercise a fresh import even though collection already loaded the predicates.
    monkeypatch.delitem(sys.modules, "eregion_sv.shim._sv_verifier")
    monkeypatch.delattr(sys.modules["eregion_sv.shim"], "_sv_verifier")
    assert _harness_shim() is not SHIM
    for obj, name, original in saved:
        assert getattr(obj, name) is original, name


@pytest.mark.skipif(SHIM is None, reason="needs the grading harness (eregion_sv): EREGION_SV_SRC or the umbrella's repos/eregion-sv")
def test_the_child_and_the_harness_shim_have_the_same_float_domains():
    """The child is the judge of every FALSE Tharkun reports and the shim is the judge of the grader's replay: their predicates must agree."""
    rng = random.Random(42)
    values = [0.0, -0.0, 1.0, -1.0, 0.5, 0.1, TOP, R, 1e-300, 5e-324, 10.0, 7.0, 4.999999999999999, math.inf, -math.inf, math.nan, 1e308, -1e308,
              2.5, 1e16 + 2.0, 61726.5445]
    for a, b in BOUNDS:
        vs = values + [native(a, b, k) for k in KS] + [rng.uniform(-10, 10) for _ in range(20)]
        for v in vs:
            assert C.float_range_producible(a, b, v) == SHIM.float_range_producible(a, b, v), (a, b, v)
            assert C.float_range_valid(a, b, v) == SHIM.float_range_valid(a, b, v), (a, b, v)
    for v in values:
        assert C.float_producible(v) == SHIM.float_producible(v), v
    assert C._FLOAT_DEFAULT == SHIM._FLOAT_DEFAULT and math.copysign(1.0, C._FLOAT_DEFAULT) < 0


# ------------------------------------------------------------------ bounds that are not exactly int / float / bool ([internal ref] follow-up, MG's review)

import decimal
import fractions
import struct


class MyInt(int):
    pass


class MyFloat(float):
    pass


class UserBound:
    """arithmetic a predicate cannot decide: the helper would run it"""

    def __init__(self, v):
        self.v = v

    def __sub__(self, o):
        return UserBound(self.v - o.v)

    def __mul__(self, k):
        return UserBound(self.v * k)

    def __add__(self, o):
        return self.v + o.v

    def __lt__(self, o):
        return self.v < o.v

    def __gt__(self, o):
        return self.v > o.v


NON_PLAIN = [
    ("an int subclass", MyInt(0), MyInt(1)),
    ("a float subclass", MyFloat(0.0), MyFloat(1.0)),
    ("Fraction", fractions.Fraction(0), fractions.Fraction(1)),
    ("Decimal", decimal.Decimal(0), decimal.Decimal(1)),
    ("a user object", UserBound(0.0), UserBound(1.0)),
    ("a plain lower bound and a subclass upper bound", 0, MyInt(1)),
    ("a subclass lower bound and a plain upper bound", MyFloat(0.0), 1.0),
]


@pytest.mark.parametrize("label, a, b", NON_PLAIN, ids=[n[0] for n in NON_PLAIN])
def test_a_bound_that_is_not_exactly_int_float_or_bool_has_no_value_in_the_domain(label, a, b):
    """The closed interval used as the fallback held ``1.0`` for ``(0, 1)``, a value ``uniform`` never returns: a FALSE with it was a
    wrong FALSE.  Every value is refused now, so no run with such a bound can be a witness."""
    for v in (0.0, -0.0, 0.25, 0.5, TOP, 1.0, 2.0, -1.0, math.inf, -math.inf, math.nan):
        assert C.float_range_valid(a, b, v) is False, (label, v)
    # the same call with plain bounds is a real domain: the refusal is about the TYPE, not the numbers
    assert C.float_range_valid(0, 1, 0.5) and C.float_range_valid(0.0, 1.0, TOP) and not C.float_range_valid(0, 1, 1.0)
    assert C.float_range_valid(True, 2, 1.5) and C.float_range_valid(False, True, 0.5)


def _prog(tmp_path, bound_a, bound_b, guard, imports="import fractions, decimal"):
    p = tmp_path / "prog.py"
    p.write_text(f"""import _sv_verifier
{imports}
class MyInt(int):
    pass
class MyFloat(float):
    pass
x = _sv_verifier.nondet_float_range({bound_a}, {bound_b})
{guard}
""")
    return str(p)


E2E = [
    ("MyInt(0), MyInt(1)", "MyInt(0)", "MyInt(1)"),
    ("MyFloat(0.0), MyFloat(1.0)", "MyFloat(0.0)", "MyFloat(1.0)"),
    ("Fraction(0), Fraction(1)", "fractions.Fraction(0)", "fractions.Fraction(1)"),
    ("Decimal(0), Decimal(1)", "decimal.Decimal(0)", "decimal.Decimal(1)"),
]


@pytest.mark.parametrize("label, a, b", E2E, ids=[e[0] for e in E2E])
def test_a_draw_with_such_bounds_is_refused_and_never_a_witness(tmp_path, label, a, b):
    """MG's probes (#175 review): ``nondet_float_range(MyInt(0), MyInt(1)); assert x < 1.0`` was FALSE (witness 1.0) and natively safe."""
    from tharkun.sv import verify
    from tharkun.sv.engine import run_once

    path = _prog(tmp_path, a, b, "assert x < 1.0")
    for v in (1.0, 0.5, 0.0):
        out = run_once(sys.executable, path, [{"kind": "float", "value": float(v).hex()}], timeout=30)
        assert out.status == "raised" and out.exc == "ValueError" and out.invalid_draw, (label, v, out.status, out.exc)
        assert out.violated(("AssertionError", "ValueError")) is None, (label, v)
    verdict = verify(path, ("AssertionError",), budget_seconds=8, jobs=1)
    assert verdict.verdict != "false", (label, verdict.verdict, verdict.reason)
    # the plain twin of the same program is the control: it reaches a violation the code can produce (x == 0.9999999999999999 is not
    # < 0.5) and the witness is a producible value
    plain = _prog(tmp_path, "0", "1", "assert x < 0.5", "")
    verdict = verify(plain, ("AssertionError",), budget_seconds=60, jobs=1)
    assert verdict.verdict == "false", (verdict.verdict, verdict.reason)
    assert C.float_range_producible(0, 1, float.fromhex(verdict.witness["vector"][0]["value"]))


def test_the_non_plain_bound_tests_catch_the_old_closed_interval_rule(monkeypatch):
    def old(a, b, v):
        if type(a) in (int, float, bool) and type(b) in (int, float, bool):
            return C.float_range_producible(a, b, v)
        return not (math.isnan(v) or not min(a, b) <= v <= max(a, b))

    monkeypatch.setattr(C, "float_range_valid", old)
    with pytest.raises(AssertionError):
        test_a_bound_that_is_not_exactly_int_float_or_bool_has_no_value_in_the_domain("an int subclass", MyInt(0), MyInt(1))


@pytest.mark.skipif(SHIM is None, reason="needs the grading harness (eregion_sv): EREGION_SV_SRC or the umbrella's repos/eregion-sv")
@pytest.mark.parametrize("label, a, b", NON_PLAIN, ids=[n[0] for n in NON_PLAIN])
def test_the_harness_shim_refuses_the_same_bounds(label, a, b):
    for v in (0.0, 0.5, 1.0, math.nan):
        assert SHIM.float_range_valid(a, b, v) is False and C.float_range_valid(a, b, v) is False, (label, v)


# ------------------------------------------------------------------ the predicate against BRUTE FORCE at a reduced resolution (MG's review of #53)

_EDGE = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1e-300, -1e-300, 5e-324, -5e-324, 1e-310, 2.2250738585072014e-308, 1e308, -1e308,
         sys.float_info.max, -sys.float_info.max, math.inf, -math.inf, math.nan, 1.5, 3.0, 0.1, 0.7, -3.7, 123456.789, 2.5, 1 - 2 ** -53,
         1 + 2 ** -52, 2 ** 53, -2 ** 53, 2 ** 53 + 2, 1e16, 1e17, 0, 1, -1, 5, 10, 2 ** 60, -2 ** 60, 2 ** 1023, 2 ** 1024, True, False, 3, 7, -2]


def _random_bound(rng):
    c = rng.random()
    if c < 0.25:
        return rng.choice(_EDGE)
    if c < 0.45:
        return rng.uniform(-10, 10)
    if c < 0.6:
        return rng.randint(-20, 20)
    if c < 0.75:
        return struct.unpack("d", struct.pack("Q", rng.getrandbits(64)))[0]
    if c < 0.85:
        return 5e-324 * rng.randint(-2000, 2000)
    if c < 0.95:
        return rng.choice([1e308, -1e308, 1e307, 1.2e308]) * rng.random()
    return float(rng.randint(-2 ** 62, 2 ** 62))


def _brute_image(module, a, b, nk, r):
    """The values (float.hex, ``"nan"`` for any nan) ``a + (b - a) * (k * r)`` takes over EVERY ``k < nk``; None when the helper raises."""
    out = set()
    try:
        for k in range(nk):
            v = a + (b - a) * (k * r)
            out.add("nan" if v != v else float(v).hex())
    except OverflowError:
        return None
    return out


def _candidates(a, b, image):
    vs = {0.0, -0.0, math.inf, -math.inf, math.nan, 1.0, -1.0}
    for x in (a, b):
        try:
            fx = float(x)
        except OverflowError:
            continue
        vs.add(fx)
        if math.isfinite(fx):
            vs.update((math.nextafter(fx, math.inf), math.nextafter(fx, -math.inf)))
    for h in list(image or [])[:40]:
        if h == "nan":
            continue
        v = float.fromhex(h)
        vs.add(v)
        if math.isfinite(v):
            vs.update((math.nextafter(v, math.inf), math.nextafter(v, -math.inf)))
    return vs


def _predicate_vs_brute_force(module, bits, pairs, monkeypatch):
    """``module.float_range_producible`` at ``k * 2**-bits`` (the module reads ``_NK`` and ``_R53`` at call time) against the set built
    by enumerating every ``k``: the binary search over the monotone image must give the enumeration's answer for every candidate."""
    monkeypatch.setattr(module, "_NK", 1 << bits)
    monkeypatch.setattr(module, "_R53", 2.0 ** -bits)
    bad, checked, producible = [], 0, 0
    for a, b in pairs:
        image = _brute_image(module, a, b, 1 << bits, 2.0 ** -bits)
        for v in _candidates(a, b, image):
            want = image is not None and ("nan" if v != v else float(v).hex()) in image
            got = module.float_range_producible(a, b, v)
            checked += 1
            producible += want
            if got != want:
                bad.append((a, b, v, want, got))
    return bad, checked, producible


def _pairs(rng, n_random):
    import itertools

    return list(itertools.product(_EDGE, _EDGE)) + [(_random_bound(rng), _random_bound(rng)) for _ in range(n_random)]


@pytest.mark.parametrize("bits, n_random", [(3, 600), (6, 600), (10, 150)])
def test_the_child_predicate_equals_brute_force_over_every_k_at_a_reduced_resolution(monkeypatch, bits, n_random):
    pairs = _pairs(random.Random(1159 + bits), n_random)
    bad, checked, producible = _predicate_vs_brute_force(C, bits, pairs, monkeypatch)
    assert not bad, bad[:5]
    assert checked > 50_000 and producible > 5_000        # a comparison that checks nothing proves nothing


@pytest.mark.skipif(SHIM is None, reason="needs the grading harness (eregion_sv): EREGION_SV_SRC or the umbrella's repos/eregion-sv")
def test_the_shim_predicate_equals_brute_force_over_every_k_at_a_reduced_resolution(monkeypatch):
    bad, checked, producible = _predicate_vs_brute_force(SHIM, 6, _pairs(random.Random(53), 400), monkeypatch)
    assert not bad, bad[:5]
    assert checked > 50_000 and producible > 5_000


@pytest.mark.parametrize("mutant", ["inverted direction", "top k lost", "no successor test", "zero sign ignored"])
def test_the_brute_force_check_catches_a_broken_search(monkeypatch, mutant):
    real = C._monotone_image_contains

    def inverted(f, nk, v, nondecreasing):
        return real(f, nk, v, not nondecreasing)

    def top_lost(f, nk, v, nondecreasing):
        return real(f, nk - 1, v, nondecreasing)

    def no_successor(f, nk, v, nondecreasing):
        if v != v:
            return False
        first, last = f(0), f(nk - 1)
        if (v < first or v > last) if nondecreasing else (v > first or v < last):
            return False
        return C._same_float(f(C._first_reaching(f, nk, v, nondecreasing)), v)

    def zero_sign_ignored(f, nk, v, nondecreasing):
        return real(f, nk, abs(v) if v == 0.0 else v, nondecreasing) or real(f, nk, -abs(v) if v == 0.0 else v, nondecreasing)

    monkeypatch.setattr(C, "_monotone_image_contains", {"inverted direction": inverted, "top k lost": top_lost,
                                                        "no successor test": no_successor, "zero sign ignored": zero_sign_ignored}[mutant])
    bad, checked, _ = _predicate_vs_brute_force(C, 6, _pairs(random.Random(7), 300), monkeypatch)
    if mutant == "no successor test":
        pytest.skip("MG: no data of the brute force distinguishes dropping the successor test (it looks equivalent)")
    assert bad, mutant


@pytest.mark.parametrize("base", [int, float])
@pytest.mark.parametrize("upper", [False, True])
def test_metaclass_equality_cannot_admit_a_bound_or_run_its_callbacks(base, upper, monkeypatch):
    calls = []

    class SpoofMeta(type):
        def __eq__(cls, other):
            calls.append("type equality")
            return True

    class Bound(base, metaclass=SpoofMeta):
        def __add__(self, other):
            calls.append("addition")
            # A fresh native helper returns 0.0, but repeated predicate calls used to reach 0.5.
            return 0.0 if calls.count("addition") == 1 else 0.5

        def __sub__(self, other):
            calls.append("subtraction")
            return super().__sub__(other)

        def __float__(self):
            calls.append("conversion")
            return super().__float__()

    a, b = (0, Bound(1)) if upper else (Bound(0), 1)
    assert C.float_range_valid(a, b, 0.5) is False
    with pytest.raises(ValueError, match="bounds are not plain numbers"):
        C.snap_float_range(0.5, a, b)
    monkeypatch.setattr(C._State, "ranges", [])
    C._note_range(a, b)
    assert C._State.ranges == []
    assert calls == []


@pytest.mark.parametrize("base", ["int", "float"])
@pytest.mark.parametrize("upper", [False, True])
def test_a_metaclass_spoof_draw_is_invalid_and_cannot_be_a_witness(tmp_path, base, upper):
    from tharkun.sv.engine import run_once

    a, b = ("0", "Bound(1)") if upper else ("Bound(0)", "1")
    path = tmp_path / "spoof.py"
    path.write_text(f"""import _sv_verifier
calls = []
class SpoofMeta(type):
    def __eq__(cls, other):
        calls.append("type equality")
        return True
class Bound({base}, metaclass=SpoofMeta):
    def __float__(self):
        calls.append("conversion")
        return super().__float__()
    def __add__(self, other):
        calls.append("addition")
        return super().__add__(other)
    def __sub__(self, other):
        calls.append("subtraction")
        return super().__sub__(other)
try:
    _sv_verifier.nondet_float_range({a}, {b})
except ValueError:
    pass
assert calls == []
""")
    out = run_once(sys.executable, str(path), [{"kind": "float", "value": (0.5).hex()}], timeout=30)
    assert out.invalid_draw and out.exc is None, (out.status, out.exc)
    assert out.violated(("AssertionError", "ValueError")) is None
    assert out.ranges == []
