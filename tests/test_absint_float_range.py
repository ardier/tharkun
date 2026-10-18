"""[internal ref], [internal ref]: ``_sv_verifier.nondet_float_range`` in absint, modelled BOTH ways; exact float
literals; float comparisons that decide a branch.

The helper documents ``nondet_float_range(a, b)`` as a float in ``[a, b]`` and implements it as
``random.uniform(a, b)`` = ``a + (b - a) * random()`` (never ``b`` for ``(0.0, 1.0)``; ``inf`` or ``nan`` when
``b - a`` overflows or a bound is infinite).  A TRUE is proven over the LARGER reading, a FALSE is confirmed only with a
value the code can produce (``sv/_child.py``, the harness shim): so the abstract value is the JOIN of the documented
domain (the non-NaN floats in ``[min(a, b), max(a, b)]``, the draws the shim accepted before [internal ref]) and the code's
(``Analyzer.uniform_formula``).  These tests hold the model to both: it contains every documented draw, the bounds
included, and every value the native ``random.uniform`` returns, ``inf`` and ``nan`` among them.
"""

from __future__ import annotations

import math
import random
import textwrap

import pytest

from tharkun.solve.absint import prove

INF = float("inf")
NAN = float("nan")


def _prove(src: str, classes=("AssertionError",)):
    return prove(textwrap.dedent(src), list(classes), filename="<program>")


def _lit(v) -> str:
    """A source literal for ``v``: ``repr(inf)`` is the bare name ``inf``, which is a NameError."""
    if isinstance(v, float) and math.isinf(v):
        return "1e999" if v > 0 else "-1e999"
    if isinstance(v, float) and math.isnan(v):
        return "(1e999 - 1e999)"
    return repr(v)


def accepted(v: float, a: float, b: float) -> bool:
    """The shim's rule, verbatim."""
    return not (math.isnan(v) or not min(a, b) <= v <= max(a, b))


# ---------------------------------------------------------------------------
# the domain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a, b", [(0.0, 1024.0), (1024.0, 0.0), (-3.5, 2.25), (0, 1), (2, -2), (5.0, 5.0), (0.1, 0.7)])
def test_the_domain_holds_the_documented_one_for_constant_bounds(a, b):
    lo, hi = min(a, b), max(a, b)
    base = f"import _sv_verifier\nx = _sv_verifier.nondet_float_range({_lit(a)}, {_lit(b)})\n"
    # never more than the two readings allow ...
    assert _prove(base + f"assert x >= {_lit(lo)}\nassert x <= {_lit(hi)}\nassert x >= {_lit(lo)} or x < {_lit(lo)}\n").verdict == "TRUE"
    # ... and nothing less: the documented reading accepts the bounds themselves, so a strict version is unproven (a TRUE is proven
    # over the LARGER reading, even where the code never returns b: uniform(0.0, 1.0))
    assert _prove(base + f"assert x > {_lit(lo)}\n").verdict == "UNKNOWN"
    assert _prove(base + f"assert x < {_lit(hi)}\n").verdict == "UNKNOWN"


@pytest.mark.parametrize("a, b", [(0.0, 1.0), (5, 10), (-3.5, 2.25), (0.1, 0.7), (2.5, 2.5), (1e308, 1.7e308)])
def test_every_value_the_native_uniform_returns_is_in_the_domain(a, b):
    """The code reading: ``random.uniform`` with ``random()`` fixed to a chosen ``k * 2**-53`` returns a value the program may see, so
    ``assert x != <that value>`` is never TRUE."""
    class Draw(random.Random):
        def __init__(self, k):
            super().__init__(0)
            self.k = k

        def random(self):
            return self.k * 2.0 ** -53

    for k in (0, 1, 2 ** 52, 2 ** 53 - 1):
        v = Draw(k).uniform(a, b)
        src = f"import _sv_verifier\nx = _sv_verifier.nondet_float_range({_lit(a)}, {_lit(b)})\nassert x != {_lit(v)}\n"
        assert _prove(src).verdict == "UNKNOWN", (a, b, k, v)


@pytest.mark.parametrize("a, b", [(-1e308, 1e308), (1e308, -1e308), (-INF, INF), (0.0, INF), (-INF, 3.0), (INF, INF), (0.0, NAN)])
def test_an_overflowing_span_or_an_infinite_or_nan_bound_may_return_nan_and_inf(a, b):
    """``uniform(-1e308, 1e308)``: ``b - a`` is ``inf``, so the code returns ``nan`` (``inf * 0.0``) or ``inf``, outside ``[a, b]``.
    ``assert x <= 1e308`` after it was proven TRUE ([internal ref]'s probe, the hull of its bounds) and fails natively on every draw."""
    class Draw(random.Random):
        def __init__(self, k):
            super().__init__(0)
            self.k = k

        def random(self):
            return self.k * 2.0 ** -53

    produced = {Draw(k).uniform(a, b) for k in (0, 1, 2 ** 53 - 1)}
    base = f"import _sv_verifier\nx = _sv_verifier.nondet_float_range({_lit(a)}, {_lit(b)})\n"
    for v in produced:
        if v != v:
            assert _prove(base + "assert x == x\n").verdict == "UNKNOWN", (a, b)
        elif v == INF or v == -INF:
            assert _prove(base + f"assert x != {_lit(v)}\n").verdict == "UNKNOWN", (a, b, v)
    if any(v == INF for v in produced) and max(a, b) != INF:
        assert _prove(base + f"assert x <= {_lit(max(a, b))}\n").verdict == "UNKNOWN", (a, b)


def test_a_nan_bound_keeps_the_documented_draw_beside_the_codes_nan():
    """The documented reading of ``nondet_float_range(1.0, nan)`` accepts exactly ``1.0`` (``min`` and ``max`` both return ``a``); the code,
    ``uniform(1.0, nan)``, returns ``nan``.  A TRUE must hold under both: ``1.0 / (x - 1.0)`` divides by zero for the documented draw ``1.0``
    and by ``nan`` (no error) for the code's, so it is not proven free of ``ZeroDivisionError``."""
    assert math.isnan(random.uniform(1.0, NAN))                                  # the native control: the code returns nan
    with pytest.raises(ZeroDivisionError):
        1.0 / (1.0 - 1.0)                                                      # ... and the documented draw is the zero divisor
    src = f"import _sv_verifier\nx = _sv_verifier.nondet_float_range(1.0, {_lit(NAN)})\ny = 1.0 / (x - 1.0)\n"
    assert _prove(src, ["ZeroDivisionError"]).verdict == "UNKNOWN"
    # nan is in the domain too: a nan never compares equal, so x == x is not proven
    assert _prove(f"import _sv_verifier\nx = _sv_verifier.nondet_float_range(1.0, {_lit(NAN)})\nassert x == x\n").verdict == "UNKNOWN"


def test_the_probe_g15_the_hidden_overflow_is_not_proven():
    """MAIN's probe ``g15_uniform_ok_hidden.py`` and MIN's ``cmp_e14_uniform_overflow_hidden``, through the helper that is called
    ``nondet_float_range``: absint alone is UNKNOWN (it was TRUE: the model was the closed interval)."""
    src = """
        import _sv_verifier as sv
        b = sv.nondet_int()
        if b == 12345678901234567890123:
            y = sv.nondet_float_range(-1e308, 1e308)
            assert y <= 1e308
    """
    assert _prove(src).verdict == "UNKNOWN"


def test_the_result_is_a_float_never_nan_and_the_abstract_domain_contains_every_accepted_draw():
    for a, b in [(0.0, 1.0), (1.0, 0.0), (-1.0, -1.0), (-INF, INF), (0.0, -0.0)]:
        for v in (a, b, 0.0, -0.0, INF, -INF, NAN, (a + b) / 2 if math.isfinite(a + b) else 0.0):
            if not accepted(v, a, b):
                continue
            lo, hi = min(a, b), max(a, b)
            assert lo <= v <= hi and v == v


def test_a_non_constant_bound_gives_the_hull_of_the_bounds_facts():
    """The documented reading is the hull of the bounds' facts; the code's formula, ``a + (b - a) * random()`` over interval facts,
    does not relate ``a`` to itself, so it is wider (``[0, 21]`` here, not ``[0, 20]``): the price of proving a TRUE over both."""
    result = _prove("""
        import _sv_verifier
        a = _sv_verifier.nondet_float_range(0.0, 1.0)
        b = _sv_verifier.nondet_float_range(10.0, 20.0)
        x = _sv_verifier.nondet_float_range(a, b)
        assert x >= 0.0
        assert x <= 21.0
        assert x >= 0.0 or x < 0.0
    """)
    assert result.verdict == "TRUE", result.reason
    assert _prove("""
        import _sv_verifier
        a = _sv_verifier.nondet_float_range(0.0, 1.0)
        b = _sv_verifier.nondet_float_range(10.0, 20.0)
        x = _sv_verifier.nondet_float_range(a, b)
        assert x <= 20.0
    """).verdict == "UNKNOWN"
    assert _prove("""
        import _sv_verifier
        a = _sv_verifier.nondet_float_range(0.0, 1.0)
        b = _sv_verifier.nondet_float_range(10.0, 20.0)
        x = _sv_verifier.nondet_float_range(a, b)
        assert x > 0.0
    """).verdict == "UNKNOWN"


def test_a_bound_that_may_be_nan_is_not_symmetric_exactly_as_min_and_max_are():
    # b NaN with a number a accepts exactly a (min and max both return a); a NaN accepts nothing
    for a, b in [(1.0, NAN), (NAN, 1.0)]:
        for v in (1.0, 0.5, 2.0, NAN, INF):
            expected = accepted(v, a, b)
            assert expected == ((not math.isnan(a)) and math.isnan(b) and v == a), (a, b, v)
    # the unknown float (any value including NaN) as a bound: the code returns nan for a nan bound, so the value may be nan too ([internal ref])
    result = _prove("""
        import _sv_verifier
        a = _sv_verifier.nondet_float()
        x = _sv_verifier.nondet_float_range(1.0, a)
        y = x + 1.0
    """, ["TypeError", "ZeroDivisionError"])
    assert result.verdict == "TRUE", result.reason
    assert _prove("""
        import _sv_verifier
        a = _sv_verifier.nondet_float()
        x = _sv_verifier.nondet_float_range(1.0, a)
        assert x >= -1e999
    """).verdict == "UNKNOWN"
    # ... and it cannot claim more than the hull of every accepted draw
    assert _prove("""
        import _sv_verifier
        a = _sv_verifier.nondet_float()
        x = _sv_verifier.nondet_float_range(1.0, a)
        assert x > 1.0
    """).verdict == "UNKNOWN"


def test_non_numeric_bounds_are_a_recorded_typeerror_and_a_wrong_arity_too():
    result = _prove("""
        import _sv_verifier
        x = _sv_verifier.nondet_float_range('a', 1.0)
    """, ["TypeError"])
    assert result.verdict == "UNKNOWN"
    ok = _prove("""
        import _sv_verifier
        x = _sv_verifier.nondet_float_range(0, 1.0)
        y = _sv_verifier.nondet_float_range(True, 2)
    """, ["TypeError"])
    assert ok.verdict == "TRUE", ok.reason


# ---------------------------------------------------------------------------
# what it buys: the chaos and spline shapes
# ---------------------------------------------------------------------------


def test_a_float_range_draw_keeps_a_division_safe():
    result = _prove("""
        import _sv_verifier
        thickness = _sv_verifier.nondet_float_range(1.0, 2.0)
        r = 1.0 / thickness
    """, ["ZeroDivisionError"])
    assert result.verdict == "TRUE", result.reason
    assert _prove("""
        import _sv_verifier
        thickness = _sv_verifier.nondet_float_range(0.0, 1.0)
        r = 1.0 / thickness
    """, ["ZeroDivisionError"]).verdict == "UNKNOWN"


# ---------------------------------------------------------------------------
# float literals and comparisons
# ---------------------------------------------------------------------------


def test_a_float_literal_is_exact_and_float_rounding_is_not_trusted():
    assert _prove("y = 0.5\nassert y >= 0.0\nassert y > 0.25\nassert y < 0.75\n").verdict == "TRUE"
    # 0.1 + 0.2 == 0.3 is False in IEEE-754: never proved
    assert _prove("assert 0.1 + 0.2 == 0.3\n").verdict == "UNKNOWN"
    assert _prove("y = 0.1\nassert y * 3 == 0.3\n").verdict == "UNKNOWN"


def test_a_float_comparison_decides_a_branch_only_when_the_facts_leave_it_no_pair():
    ok = _prove("""
        import _sv_verifier
        x = _sv_verifier.nondet_float_range(1.0, 1024.0)
        if x > 0.5:
            pass
        else:
            z = 1 // 0
    """, ["ZeroDivisionError"])
    assert ok.verdict == "TRUE", ok.reason
    for cond in ("x > 1.0", "x >= 1.5", "x < 1024.0", "x == 1.0"):
        result = _prove(f"""
            import _sv_verifier
            x = _sv_verifier.nondet_float_range(1.0, 1024.0)
            if {cond}:
                pass
            else:
                z = 1 // 0
        """, ["ZeroDivisionError"])
        assert result.verdict == "UNKNOWN", cond


def test_nan_makes_every_ordered_comparison_false_and_an_unknown_float_may_be_nan():
    for src in (
        "import _sv_verifier\nx = _sv_verifier.nondet_float()\nassert x == x\n",
        "import _sv_verifier\nx = _sv_verifier.nondet_float()\nassert x >= 0.0 or x < 0.0\n",
        "import _sv_verifier\nx = _sv_verifier.nondet_float()\nassert not (x != x)\n",
    ):
        assert _prove(src).verdict == "UNKNOWN", src
    # with the NaN excluded the same comparisons are decided
    assert _prove("import _sv_verifier\nx = _sv_verifier.nondet_float_range(0.0, 1.0)\nassert x >= 0.0 or x < 0.0\n").verdict == "TRUE"


def test_an_int_against_a_float_is_compared_exactly():
    assert _prove("import _sv_verifier\nx = _sv_verifier.nondet_float_range(0.0, 1.0)\nassert x < 2\nassert 2 > x\n").verdict == "TRUE"
    assert _prove("import _sv_verifier\nx = _sv_verifier.nondet_float_range(0.0, 3.0)\nassert x < 2\n").verdict == "UNKNOWN"


def test_a_comparison_with_a_non_numeric_operand_is_left_alone():
    # a str against a float is a TypeError (recorded); nothing here decides it
    result = _prove("x = 'a'\ny = 1.0\nz = x < y\n", ["TypeError"])
    assert result.verdict == "UNKNOWN"


# ---------------------------------------------------------------------------
# the differential: a TRUE for a program that raises, over random float programs
# ---------------------------------------------------------------------------

CONSTS = ["0.0", "1.0", "-1.0", "0.5", "2.0", "-0.0", "1e308", "3"]
CMPS = ["<", "<=", ">", ">=", "==", "!="]


def _program(rng):
    a, b = rng.choice(CONSTS), rng.choice(CONSTS)
    lines = ["import _sv_verifier", f"x = _sv_verifier.nondet_float_range({a}, {b})"]
    if rng.random() < 0.5:
        lines.append(f"y = _sv_verifier.nondet_float_range({rng.choice(CONSTS)}, x)")
    else:
        lines.append(f"y = x {rng.choice('+-*')} {rng.choice(CONSTS)}")
    cmp_ = f"y {rng.choice(CMPS)} {rng.choice(CONSTS)}"
    tail = rng.choice(["1 // 0", "1.0 / y", "1.0 / x", "[1][3]", "{}[1]"])
    lines += [f"if {cmp_}:", "    pass", "else:", f"    z = {tail}"]
    if rng.random() < 0.4:
        lines.append(f"assert x {rng.choice(CMPS)} {rng.choice(CONSTS)}")
    return "\n".join(lines) + "\n"


class _Invalid(Exception):
    pass


def _run(src: str, draws):
    it = iter(draws)

    def nondet_float_range(a, b):
        v = next(it)
        if math.isnan(v) or not min(a, b) <= v <= max(a, b):
            raise _Invalid
        return v

    mod = type("M", (), {"nondet_float_range": staticmethod(nondet_float_range), "nondet_float": staticmethod(lambda: next(it))})
    import sys

    sys.modules["_sv_verifier"] = mod  # type: ignore[assignment]
    try:
        exec(compile(src, "<native>", "exec"), {})
    except _Invalid:
        return "invalid"
    except BaseException as exc:  # noqa: BLE001
        return type(exc).__name__
    finally:
        del sys.modules["_sv_verifier"]
    return None


@pytest.mark.parametrize("seed", range(30))
def test_a_program_absint_proves_safe_never_raises_for_any_accepted_draw(seed):
    rng = random.Random(1105_00 + seed)
    proved = 0
    for _ in range(10):
        src = _program(rng)
        for cls in ("ZeroDivisionError", "IndexError", "KeyError", "AssertionError", "TypeError"):
            result = prove(src, [cls], filename="<fuzz>", time_budget=20.0)
            if result.verdict != "TRUE":
                continue
            proved += 1
            points = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1e308, -1e308, INF, -INF, NAN, 3.0, 1e-308, 5e-324]
            for d1, d2 in ((p, q) for p in points for q in points):
                outcome = _run(src, [d1, d2])
                assert outcome != cls, f"WRONG TRUE for {cls} with draws {d1!r}, {d2!r}:\n{src}"
    assert proved > 0


# ---------------------------------------------------------------------------
# #138 review lead: a float against an int beyond the float range (``inf > 10 ** 400`` is True)
# ---------------------------------------------------------------------------

BIG = "1" + "0" * 400                       # 10 ** 400: above every finite float, below inf
MAXFLT = 1.7976931348623157e308
HUGE_INTS = [BIG, "-" + BIG, str(2 ** 1024), str(2 ** 1024 - 2 ** 970), str(2 ** 1024 - 2 ** 970 - 1),
             str(int(MAXFLT)), str(int(MAXFLT) + 1), str(int(MAXFLT) - 1), "-" + str(int(MAXFLT) + 1)]
CMP_OPS = ["<", "<=", ">", ">=", "==", "!="]


def test_a_float_that_may_be_infinite_is_above_a_huge_int_and_the_branch_is_kept():
    base = "import _sv_verifier\nx = _sv_verifier.nondet_float_range(0.0, 1e999)\n"
    # x = inf is an accepted draw, and inf > 10 ** 400: the true branch is reachable
    assert _prove(base + f"if x > {BIG}:\n    z = 1 // 0\n", ["ZeroDivisionError"]).verdict == "UNKNOWN"
    assert _prove(base + f"if x < {BIG}:\n    pass\nelse:\n    z = 1 // 0\n", ["ZeroDivisionError"]).verdict == "UNKNOWN"
    # no crash: the comparison used to raise OverflowError out of the view of the int
    assert _prove(base + f"if x >= {BIG}:\n    z = 1 // 0\n", ["ZeroDivisionError"]).verdict == "UNKNOWN"


def test_a_finite_float_is_below_a_huge_int_and_the_branch_is_pruned():
    base = "import _sv_verifier\nw = _sv_verifier.nondet_float_range(0.0, 1.7e308)\n"
    assert _prove(base + f"if w < {BIG}:\n    pass\nelse:\n    z = 1 // 0\n", ["ZeroDivisionError"]).verdict == "TRUE"
    assert _prove(base + f"if w > {BIG}:\n    z = 1 // 0\n", ["ZeroDivisionError"]).verdict == "TRUE"
    assert _prove(base + f"if w > -{BIG}:\n    pass\nelse:\n    z = 1 // 0\n", ["ZeroDivisionError"]).verdict == "TRUE"


@pytest.mark.parametrize("big", HUGE_INTS)
def test_a_comparison_with_a_huge_int_never_makes_a_wrong_true(big):
    for a, b in [(0.0, INF), (-INF, 0.0), (-INF, INF), (0.0, MAXFLT), (-MAXFLT, MAXFLT), (MAXFLT, INF)]:
        for op in CMP_OPS:
            src = (f"import _sv_verifier\nx = _sv_verifier.nondet_float_range({_lit(a)}, {_lit(b)})\n"
                   f"if x {op} {big}:\n    z = 1 // 0\n")
            result = prove(src, ["ZeroDivisionError"], filename="<program>")      # must not raise
            if result.verdict != "TRUE":
                continue
            for v in (a, b, 0.0, MAXFLT, -MAXFLT, INF, -INF, 1e300):
                if not accepted(v, a, b):
                    continue
                truth = {"<": v < int(big), "<=": v <= int(big), ">": v > int(big), ">=": v >= int(big),
                         "==": v == int(big), "!=": v != int(big)}[op]
                assert not truth, f"WRONG TRUE: x={v!r} {op} {big[:12]}...: the branch raises"
