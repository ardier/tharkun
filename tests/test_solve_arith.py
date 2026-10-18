"""The native integer guard solver (``tharkun.solve.arith``, [internal ref]) and its use as
``solve_condition``'s fallback for integer guards the inference leaves unsolved."""

from __future__ import annotations

import itertools
import random
import textwrap

import pytest

from tharkun.solve.arith import int_roots, solve_int_guard
from tharkun.solve.condition import solve_condition


@pytest.mark.parametrize("guard, names", [
    ("x * 7 - 11 == 3489", ["x"]),                                  # affine
    ("(x * 3 + 1) % 5 == 2 and x > 100", ["x"]),                    # modular + bound
    ("x * x - 3 * x == 1519054", ["x"]),                            # quadratic
    ("x ** 3 - 2 * x == 999998000", ["x"]),                         # cubic
    ("2 * x + 3 * y == 17 and x - y == 1", ["x", "y"]),             # linear system
    ("6 * x + 10 * y + 15 * z == 7 and z > 3", ["x", "y", "z"]),    # no unit coefficient
    ("x * y == 391 and x > 1 and y > 1", ["x", "y"]),               # factoring
    ("x // 10 == 42 and x % 10 == 7", ["x"]),
    ("-7 <= x // -3 < -5", ["x"]),                                  # negative divisor
    ("x % 1000003 == 999999 and x > 10**12", ["x"]),                # far from zero
    ("x == 3 or x * 5 == 2", ["x"]),
    ("not (x < 1000000)", ["x"]),
    ("x * x + y * y == 25 and x > 0 and y > 0", ["x", "y"]),
    ("1 + abs(x) > 5000 and (1 + abs(x)) % 7 == 3", ["x"]),       # abs
    ("abs(x - 10) == 3 and x < 10", ["x"]),
    ("(x & 0xFFFF) == 4660 and x >> 16 == 3", ["x"]),               # masks and shifts
    ("x * 8 << 2 == 64", ["x"]),
    ("1 + abs(x) > 10**6 and (1 + abs(x)) % 9973 == 4321 and (1 + abs(x)) % 16 == 5", ["x"]),
])
def test_finds_a_witness_that_satisfies_the_guard(guard, names):
    res = solve_int_guard(guard, names)
    assert res.status == "sat", res
    assert eval(guard, {}, dict(res.witness))


@pytest.mark.parametrize("guard", ["6 * x + 10 * y == 7", "x != 0 and x * 2 == 0", "2 * x == 1"])
def test_refutes_only_by_exact_linear_elimination(guard):
    assert solve_int_guard(guard, ["x", "y"]).status == "unsat"


def test_declines_non_integer_names_and_operations():
    assert solve_int_guard("x * 2 == y", ["x"]).status == "unknown"        # y's type unknown
    assert solve_int_guard("x / 2 == 3", ["x"]).status == "unknown"        # true division
    assert solve_int_guard("f(x) == 3", ["x"]).status == "unknown"
    assert solve_int_guard("x == 1.5", ["x"]).status == "unknown"


def test_int_roots():
    assert int_roots([-6, 1, 1]) == [-3, 2]                  # x^2 + x - 6
    assert int_roots([0, 0, 1]) == [0]
    assert int_roots([1, 0, 1]) == []                        # x^2 + 1
    assert int_roots([-8, 0, 0, 1]) == [2]                   # x^3 - 8
    assert int_roots([0]) is None


def _rand_term(rng, depth):
    if depth == 0 or rng.random() < 0.3:
        return rng.choice(["x", "y", str(rng.randint(-20, 20))])
    op = rng.choice(["+", "-", "*", "//", "%"])
    if op in ("//", "%"):
        return f"({_rand_term(rng, depth - 1)} {op} {rng.choice([2, 3, 5, -3, 7])})"
    return f"({_rand_term(rng, depth - 1)} {op} {_rand_term(rng, depth - 1)})"


def test_random_guards_never_wrong():
    """Every ``sat`` witness satisfies the guard; no ``unsat`` guard has a small solution."""
    rng = random.Random(1021)
    for _ in range(300):
        parts = [f"{_rand_term(rng, 2)} {rng.choice(['==', '!=', '<', '<=', '>', '>='])} "
                 f"{_rand_term(rng, 2)}" for _ in range(rng.randint(1, 3))]
        guard = rng.choice([" and ", " or "]).join(parts)
        res = solve_int_guard(guard, ["x", "y"])
        if res.status == "sat":
            assert eval(guard, {}, dict(res.witness)), guard
        elif res.status == "unsat":
            for x, y in itertools.product(range(-15, 16), repeat=2):
                assert not eval(guard, {}, {"x": x, "y": y}), (guard, x, y)


def _write(tmp_path, src):
    p = tmp_path / "t.py"
    p.write_text(textwrap.dedent(src))
    return str(p)


def test_solve_condition_uses_it_for_annotated_ints(tmp_path):
    path = _write(tmp_path, """\
        def f(x: int, y: int):
            if x * 7 - 11 == 3489:
                pass
            if 2 * x + 3 * y == 17 and x - y == 1:
                pass
    """)
    sol = solve_condition(path, 2)
    assert sol.strategy == "arith" and sol.satisfying_inputs == [{"x": 500}]
    sol = solve_condition(path, 4)
    assert sol.strategy == "arith" and sol.satisfying_inputs == [{"x": 4, "y": 3}]


def test_solve_condition_takes_var_types_and_polarity(tmp_path):
    path = _write(tmp_path, """\
        def main():
            x = draw()
            if x * x - 3 * x == 1519054:
                pass
            if x * 7 - 11 != 3489:
                pass
    """)
    assert solve_condition(path, 3).strategy != "arith"                  # untyped: unchanged
    sol = solve_condition(path, 3, var_types={"x": "int"})
    assert sol.strategy == "arith"
    (x,) = sol.satisfying_inputs[0].values()
    assert x * x - 3 * x == 1519054
    sol = solve_condition(path, 5, var_types={"x": "int"}, desired_branch="else")
    assert sol.satisfying_inputs and all(v["x"] * 7 - 11 == 3489 for v in sol.satisfying_inputs)


def test_infeasible_only_for_vouched_types(tmp_path):
    path = _write(tmp_path, """\
        def f(x: int, y):
            if 6 * x + 10 * y == 7:
                pass
    """)
    # y's type is not vouched for: no claim
    assert solve_condition(path, 2).strategy != "infeasible"
    assert solve_condition(path, 2, var_types={"y": "int"}).strategy == "infeasible"


@pytest.mark.parametrize("guard", ["x < y and y < x", "x // 2 == 5 and x < 10",
                                   "x + y > 10 and x < 3 and y < 3"])
def test_linear_inequalities_are_refuted_by_fourier_motzkin(guard):
    assert solve_int_guard(guard, ["x", "y"]).status == "unsat"
