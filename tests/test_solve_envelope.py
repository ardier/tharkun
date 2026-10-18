"""``tharkun.solve.affine_envelope`` -- one test per rule, the declines, and a
soundness sweep that checks every claim against the real Python values (#0911).
"""

from __future__ import annotations

import ast
from fractions import Fraction

import pytest

from tharkun.solve import AffineEnvelope, affine_envelope

_XS = range(-300, 301)


def _env(expr: str) -> AffineEnvelope:
    env = affine_envelope(expr, "x")
    assert env is not None, f"{expr!r} declined"
    return env


def _holds(expr: str, env: AffineEnvelope) -> None:
    """The claim is a bound on the REAL value at every sampled integer."""
    code = compile(ast.parse(expr, mode="eval"), "<expr>", "eval")
    for x in _XS:
        value = eval(code, {"__builtins__": {"min": min, "max": max}}, {"x": x})  # noqa: S307
        assert env.a * x + env.lo <= value <= env.a * x + env.hi, (expr, x, value, env)


def _claim(expr: str, a, lo, hi) -> None:
    env = _env(expr)
    assert (env.a, env.lo, env.hi) == (Fraction(a), Fraction(lo), Fraction(hi)), (expr, env)
    _holds(expr, env)


# -- one test per rule ------------------------------------------------------


def test_literal_and_variable() -> None:
    _claim("7", 0, 7, 7)
    _claim("x", 1, 0, 0)


def test_add_and_sub() -> None:
    _claim("x + 1", 1, 1, 1)
    _claim("x - 3", 1, -3, -3)
    _claim("(x + 1) - (x - 2)", 0, 3, 3)


def test_unary() -> None:
    _claim("-x", -1, 0, 0)
    _claim("+x", 1, 0, 0)
    _claim("~x", -1, -1, -1)


def test_mul_by_constant() -> None:
    _claim("x * 2", 2, 0, 0)
    _claim("-3 * (x + 1)", -3, -3, -3)


def test_mul_of_two_bounded_values() -> None:
    _claim("(x % 4) * (x % 3)", 0, 0, 6)


def test_xor_with_nonnegative_constant() -> None:
    # x ^ c - x in [-c, c]: x ^ c == x + c - 2 * (x & c) and 0 <= x & c <= c.
    _claim("x ^ 5", 1, -5, 5)
    _claim("5 ^ x", 1, -5, 5)


def test_xor_with_negative_constant() -> None:
    # x ^ c == -(x ^ ~c) - 1 with ~c >= 0.
    _claim("x ^ -6", -1, -6, 4)


def test_and_with_constant() -> None:
    _claim("x & 12", 0, 0, 12)
    _claim("x & -4", 1, -3, 0)


def test_or_with_constant() -> None:
    _claim("x | 12", 1, 0, 12)
    _claim("x | -4", 0, -4, -1)


def test_mod_by_constant() -> None:
    _claim("x % 7", 0, 0, 6)
    _claim("x % -7", 0, -6, 0)


def test_floordiv_by_constant() -> None:
    _claim("x // 4", Fraction(1, 4), Fraction(-3, 4), 0)
    _claim("x // -4", Fraction(-1, 4), Fraction(-3, 4), 0)


def test_shifts() -> None:
    _claim("x << 3", 8, 0, 0)
    _claim("x >> 2", Fraction(1, 4), Fraction(-3, 4), 0)


def test_literal_tuple_subscript() -> None:
    _claim("(x + 1, x * 2, x ^ 5)[2]", 1, -5, 5)
    _claim("[x, 3][-1]", 0, 3, 3)


def test_min_max_with_one_slope() -> None:
    _claim("min(x, x + 3)", 1, 0, 0)
    _claim("max(x ^ 1, x + 2)", 1, 2, 2)


def test_the_0545_relation() -> None:
    # f(x) = (x + 1, x * 2, x ^ 5); y = a + c: 2x + 1 + [-5, 5].
    _claim("(x + 1, x * 2, x ^ 5)[0] + (x + 1, x * 2, x ^ 5)[2]", 2, -4, 6)


# -- declines ----------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "x * x",          # a product of two non-constant values
        "x ** 2",         # power
        "x / 2",          # true division leaves the integers
        "x + y",          # another free name
        "x + 1.5",        # a non-int literal
        "x + True",       # bool is not taken for int
        "x ^ x",          # bitwise between two non-constant operands
        "x % x",          # modulus by a non-constant
        "x % 0",          # modulo by zero
        "x >> -1",        # negative shift count
        "abs(x)",         # a call outside min / max
        "min(x, 5)",      # min over different slopes
        "(x, 1)[x]",      # a non-literal index
        "x < 3",          # a comparison is not an integer expression
        "x +",            # not an expression
    ],
)
def test_declines(expr: str) -> None:
    assert affine_envelope(expr, "x") is None


def test_rejects_a_bad_variable_name() -> None:
    assert affine_envelope("x + 1", "not a name") is None


def test_accepts_an_ast() -> None:
    assert affine_envelope(ast.parse("x ^ 5", mode="eval"), "x") == _env("x ^ 5")
    assert affine_envelope(ast.parse("x ^ 5", mode="eval").body, "x") == _env("x ^ 5")


def test_as_record_is_plain_data() -> None:
    assert _env("x // 4").as_record() == {"var": "x", "a": "1/4", "lo": "-3/4", "hi": "0"}


# -- soundness sweep over compositions ----------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "((x ^ 13) + (x & 7)) * 3 - (x | 2)",
        "((x // 3) ^ 6) + (x % 5) * 4",
        "(x << 2) - ((x >> 1) ^ -9)",
        "~(x ^ -3) + (x & -8)",
        "max((x ^ 3) * 2, (x | 1) * 2 - 5) - min(x // 2, (x + 7) // 2)",
        "((x * 5 + 3) // -7) | 11",
    ],
)
def test_composed_claims_hold_everywhere(expr: str) -> None:
    _holds(expr, _env(expr))
