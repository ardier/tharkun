"""Tests for native directed reachability and execution-verified search."""
from __future__ import annotations

import ast
import os
import tempfile
import textwrap
import unittest

from tharkun.driver import (
    BUDGET,
    NO_TARGET,
    PLATEAU,
    REACHED,
    drive_to_line,
    enclosing_guards,
)
from tharkun.hypothesis_adapter import hypothesis_available

requires_hypothesis = unittest.skipUnless(
    hypothesis_available(), "hypothesis not installed"
)


def _write(src: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "mod.py")
    with open(p, "w") as f:
        f.write(textwrap.dedent(src))
    return p


class TestEnclosingGuards(unittest.TestCase):
    def test_nested_then(self) -> None:
        src = textwrap.dedent("""
            def f(a, b):
                if a > 0:
                    if b > 0:
                        return 1
                return 0
        """)
        frames = enclosing_guards(ast.parse(src), 5)  # the `return 1`
        self.assertEqual([(g.branch_line, g.desired_branch) for g in frames],
                         [(3, "then"), (4, "then")])

    def test_else_polarity(self) -> None:
        src = textwrap.dedent("""
            def f(a):
                if a > 0:
                    return 1
                else:
                    return 2
        """)
        frames = enclosing_guards(ast.parse(src), 6)  # the `return 2`
        self.assertEqual([(g.branch_line, g.desired_branch) for g in frames], [(3, "else")])

    def test_line_not_in_guard(self) -> None:
        src = "def f(a):\n    x = 1\n    return x\n"
        self.assertEqual(enclosing_guards(ast.parse(src), 3), [])


class TestNoTarget(unittest.TestCase):
    def test_target_not_in_any_guard(self) -> None:
        p = _write("def f(a):\n    return a\n")
        r = drive_to_line(p, "f", 2, seed={"a": 0})
        self.assertEqual(r.status, NO_TARGET)


class TestReachLinear(unittest.TestCase):
    """Reachability that the heuristic flippers alone can drive."""

    def test_nested_linear_guards(self) -> None:
        p = _write("""
            def f(a, b):
                if a > 100:
                    if b == 5:
                        return "hit"
                return "miss"
        """)
        r = drive_to_line(p, "f", 5, seed={"a": 0, "b": 0}, max_rounds=8)
        self.assertEqual(r.status, REACHED)
        self.assertTrue(r.reached)
        self.assertGreater(r.inputs["a"], 100)
        self.assertEqual(r.inputs["b"], 5)

    def test_already_reached_seed(self) -> None:
        p = _write("""
            def f(a):
                if a > 0:
                    return "hit"
                return "miss"
        """)
        r = drive_to_line(p, "f", 4, seed={"a": 5})
        self.assertEqual(r.status, REACHED)
        self.assertEqual(r.rounds, 1)  # reached on the first trace, no flips


class TestReachSound(unittest.TestCase):
    def test_nonlinear_nested(self) -> None:
        p = _write("""
            def f(a, b, c):
                if a > 100:
                    if b == a + 1:
                        if c * c == 49 and c > 0:
                            return "TARGET"
                return "miss"
        """)
        r = drive_to_line(p, "f", 6, seed={"a": 0, "b": 0, "c": 0}, max_rounds=10)
        self.assertEqual(r.status, REACHED)
        self.assertGreater(r.inputs["a"], 100)
        self.assertEqual(r.inputs["b"], r.inputs["a"] + 1)
        self.assertEqual(r.inputs["c"], 7)



class TestParamBounds(unittest.TestCase):
    def test_bounds_keep_solution_in_range(self) -> None:
        # A modular guard has solutions spaced ~M apart; without a bound the solver may
        # return one outside a field's domain. param_bounds must pin it inside [0, 2^16).
        p = _write("""
            def f(y):
                if (y * 2654435761 + 1) % 65521 == 12345:
                    return "hit"
                return "miss"
        """)
        r = drive_to_line(p, "f", 4, seed={"y": 0}, max_rounds=6,
                          param_bounds={"y": (0, 65535)})
        self.assertEqual(r.status, REACHED)
        self.assertTrue(0 <= r.inputs["y"] <= 65535)


class TestPortfolio(unittest.TestCase):
    """The 'try everything' portfolio: escalate to hypothesis, verify by execution."""

    RANGE_SRC = """
        def g(x: int):
            if 100 <= x <= 110:
                return "hit"
            return "miss"
    """

    @requires_hypothesis
    def test_heavy_tier_solves_when_cheap_disabled(self) -> None:
        # flippers=[] disables the cheap tier; the heavy tier (hypothesis) must solve it.
        p = _write(self.RANGE_SRC)
        r = drive_to_line(p, "g", 4, seed={"x": 0}, max_rounds=4, flippers=[],
                          heavy=True, minleaf=False)
        self.assertEqual(r.status, REACHED)
        self.assertTrue(100 <= r.inputs["x"] <= 110)
        # Either heavy search may claim it — the fact-driven one solves the conjunction
        # of the banked path facts, the plain one solves this guard alone. Which wins is
        # a scheduling detail; that the HEAVY tier is what solved it is the point.
        self.assertTrue(r.history[-1].technique.startswith("hypothesis"),
                        f"heavy tier did not solve it: {r.history[-1].technique}")

    def test_heavy_disabled_plateaus_when_cheap_disabled(self) -> None:
        # no cheap flippers and heavy off → nothing can solve it.
        p = _write(self.RANGE_SRC)
        r = drive_to_line(p, "g", 4, seed={"x": 0}, max_rounds=4, flippers=[],
                          heavy=False, minleaf=False)
        self.assertEqual(r.status, PLATEAU)

    def test_verification_rejects_a_non_reaching_candidate(self) -> None:
        # A flipper that proposes a value which does NOT flip the guard must be discarded
        # (the tracer is the judge), leaving the driver to plateau rather than commit it.
        def _bad_flip(source, line, desired, snapshot, params, requires):
            return {"x": -999}, "bad", False   # -999 does not satisfy 100<=x<=110
        p = _write(self.RANGE_SRC)
        r = drive_to_line(p, "g", 4, seed={"x": 0}, max_rounds=3,
                          flippers=[_bad_flip], heavy=False, minleaf=False)
        self.assertEqual(r.status, PLATEAU)
        self.assertNotEqual(r.inputs.get("x"), -999)   # the bad candidate was not committed


class TestMinleafTier(unittest.TestCase):
    """Minleaf covers what the SMT lane cannot model: live object/container state."""

    SRC = """
        from dataclasses import dataclass

        @dataclass
        class Profile:
            age: int
            country: str

        @dataclass
        class User:
            profile: Profile
            is_admin: bool

        def check(user):
            if user.profile.age >= 18 and user.profile.country == "DE":
                return "GRANTED"
            return "denied"
    """

    def setUp(self) -> None:
        import sys
        self.path = _write(self.SRC)
        sys.path.insert(0, os.path.dirname(self.path))
        import importlib.util
        spec = importlib.util.spec_from_file_location("_ml_fixture", self.path)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        tree = ast.parse(open(self.path).read())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "check")
        self.target = next(n for n in ast.walk(fn) if isinstance(n, ast.If)).body[0].lineno

    def _user(self):
        return self.mod.User(profile=self.mod.Profile(age=16, country="FR"), is_admin=False)

    def test_minleaf_reaches_an_attribute_chain_guard(self) -> None:
        r = drive_to_line(self.path, "check", self.target, seed={"user": self._user()},
                          max_rounds=3, minleaf=True)
        self.assertEqual(r.status, REACHED)
        self.assertTrue(r.history[-1].technique.startswith("minleaf"))
        self.assertGreaterEqual(r.inputs["user"].profile.age, 18)
        self.assertEqual(r.inputs["user"].profile.country, "DE")

    def test_without_minleaf_the_same_guard_plateaus(self) -> None:
        r = drive_to_line(self.path, "check", self.target, seed={"user": self._user()},
                          max_rounds=3, minleaf=False)
        self.assertEqual(r.status, PLATEAU)

    def test_patch_does_not_corrupt_the_traced_object(self) -> None:
        # The applier deep-copies, so the caller's seed object is untouched.
        u = self._user()
        drive_to_line(self.path, "check", self.target, seed={"user": u},
                      max_rounds=3, minleaf=True)
        self.assertEqual(u.profile.age, 16)
        self.assertEqual(u.profile.country, "FR")


class TestPlateauNonParam(unittest.TestCase):
    def test_local_derived_guard_plateaus(self) -> None:
        # The guard depends on a local computed from an external call the driver
        # cannot invert by changing a parameter — it should plateau, not spin.
        p = _write("""
            def f(n):
                k = len(str(n)) - len(str(n))   # always 0, not controllable to 5
                if k == 5:
                    return "hit"
                return "miss"
        """)
        r = drive_to_line(p, "f", 5, seed={"n": 1}, max_rounds=4)
        self.assertIn(r.status, (PLATEAU, BUDGET))
        self.assertFalse(r.reached)


if __name__ == "__main__":
    unittest.main()


def test_native_integer_candidates_respect_polarity_and_never_certify_types(tmp_path):
    from tharkun.driver import _flip_static

    path = tmp_path / "guard.py"
    path.write_text("def f(x):\n    if 2 * x == 6:\n        return True\n")
    args = (str(path), 2)
    got, technique, dead = _flip_static(*args, "then", {"x": 0}, {"x"}, "0 <= x <= 5")
    assert got == {"x": 3} and technique == "static:arith" and not dead
    got, _, dead = _flip_static(*args, "else", {"x": 0}, {"x"}, "x == 4")
    assert got == {"x": 4} and not dead
    path.write_text("def f(x):\n    if 2 * x == 1:\n        return True\n")
    # No integer solution, but x=0.5 reaches: an observed int is not a type invariant.
    _, _, dead = _flip_static(*args, "then", {"x": 0}, {"x"}, "True")
    assert not dead
