"""``tharkun.solve.refute.refute_stdlib`` ([internal ref]): the Z3-free dead-path procedure.
It may only say UNSAT when no assignment satisfies the conditions."""

from __future__ import annotations

import itertools
import math
import random

import pytest

from tharkun.solve.refute import refute_stdlib

T = {"x": "int", "y": "int", "z": "float", "w": "float", "b": "bool"}


@pytest.mark.parametrize("residual, scope", [
    ("x == 0", ["x > 3"]),
    ("2 * x + 4 * y == 7", []),                                  # gcd
    ("x % 7 == 3", ["x % 7 == 4"]),
    ("x // 2 == 5", ["x < 10"]),
    ("z > 1.0", ["z < 0.5"]),
    ("z == 0.0", ["z > 0.0 or z < 0.0"]),
    ("not b", ["b"]),
    ("b and x > 2", ["b == False"]),
    ("x < y", ["y < x"]),
    ("z * z < 0.0", []),                                         # a square is never negative
    ("x + 1 <= 0", ["x >= 0"]),
    ("w == w", ["w != w"]),                                      # NaN: w != w iff NaN, then w == w fails
])
def test_refutes_contradictions(residual, scope):
    ok, cert = refute_stdlib(residual, scope, T)
    assert ok, cert
    assert "unsat" in cert


@pytest.mark.parametrize("residual, scope", [
    ("x == 0", ["x < 3"]),
    ("not (z < 1.0)", ["z < 1.0 or z > 2.0"]),                   # NaN or z > 2
    ("not (z >= 0.0)", ["not (z < 0.0)"]),                      # NaN satisfies both
    ("z + 1.0 == z", []),                                        # huge z, inf
    ("s == 'a'", []),                                            # untyped: no claim
    ("x // y == 3", ["y == 0 or y != 0"]),
])
def test_does_not_refute_satisfiable_formulas(residual, scope):
    assert refute_stdlib(residual, scope, T)[0] is False


FLOATS = [0.0, -0.0, 1.0, -1.0, 0.5, 2.5, 3.0, 1e300, -1e300, math.inf, -math.inf, math.nan]


def _term(rng, d):
    if d == 0 or rng.random() < 0.35:
        return rng.choice(["x", "y", "z", "b", str(rng.randint(-4, 4)), rng.choice(["0.5", "2.0", "-1.5"])])
    op = rng.choice(["+", "-", "*", "//", "%"])
    return f"({_term(rng, d - 1)} {op} {_term(rng, d - 1)})"


def _cond(rng):
    c = f"{_term(rng, 2)} {rng.choice(['<', '<=', '>', '>=', '==', '!='])} {_term(rng, 2)}"
    return f"not ({c})" if rng.random() < 0.2 else c


def test_random_formulas_are_never_wrongly_refuted():
    rng = random.Random(1021)
    samples = [dict(x=x, y=y, z=z, b=b) for x, y, z, b in
               itertools.product(range(-6, 7), range(-6, 7), FLOATS, (False, True))]
    refuted = 0
    for _ in range(400):
        parts = [_cond(rng) for _ in range(rng.randint(1, 3))]
        residual, scope = parts[0], parts[1:]
        ok, _cert = refute_stdlib(residual, scope, T)
        if not ok:
            continue
        refuted += 1
        for env in samples:
            try:
                sat = all(eval(p, {}, dict(env)) for p in parts)
            except (ZeroDivisionError, OverflowError, ValueError):
                continue
            assert not sat, (parts, env)
    assert refuted > 20
