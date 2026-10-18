"""Tests for OSS-Fuzz blocker patterns: split/partition, type coercion,
dict key shape, and byte interior constraints.
"""

from __future__ import annotations

import os
import unittest

from tharkun.solve.condition import solve_condition

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "condition_guards.py")


def _line_of(text):
    with open(FIXTURE) as f:
        for i, line in enumerate(f, 1):
            if text in line:
                return i
    raise ValueError(f"text {text!r} not found in {FIXTURE}")


# ---------------------------------------------------------------------------
# P1: Split result constraints
# ---------------------------------------------------------------------------


class TestSplitEqTwo(unittest.TestCase):
    """len(line.split("=")) == 2 → line must contain exactly one '='."""

    def test_split_eq_two(self):
        sol = solve_condition(FIXTURE, _line_of('len(line.split("=")) == 2'))
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["line"]
        self.assertEqual(len(val.split("=")), 2)


class TestSplitGte(unittest.TestCase):
    """len(data.split(",")) >= 3 → data must contain at least two ','."""

    def test_split_gte_three(self):
        sol = solve_condition(FIXTURE, _line_of('len(data.split(",")) >= 3'))
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["data"]
        self.assertGreaterEqual(len(val.split(",")), 3)


class TestSplitMaxsplit(unittest.TestCase):
    """len(line.split(":", 1)) == 2 → line must contain ':'."""

    def test_split_maxsplit(self):
        sol = solve_condition(FIXTURE, _line_of('len(line.split(":", 1)) == 2'))
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["line"]
        self.assertIn(":", val)


# ---------------------------------------------------------------------------
# P2: Type coercion guards
# ---------------------------------------------------------------------------


class TestFloatCoercion(unittest.TestCase):
    """float(quality_str) > 0.5 → quality_str is a numeric string > 0.5."""

    def test_float_coercion_gt(self):
        sol = solve_condition(FIXTURE, _line_of("float(quality_str) > 0.5"))
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["quality_str"]
        self.assertIsInstance(val, str)
        self.assertGreater(float(val), 0.5)


class TestIntCoercion(unittest.TestCase):
    """int(week_str) > 10 → week_str is a numeric string > 10."""

    def test_int_coercion_gt(self):
        sol = solve_condition(FIXTURE, _line_of("int(week_str) > 10"))
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["week_str"]
        self.assertIsInstance(val, str)
        self.assertGreater(int(val), 10)


class TestFloatCoercionEq(unittest.TestCase):
    """float(s) == 1.0 → s = "1.0"."""

    def test_float_coercion_eq(self):
        sol = solve_condition(FIXTURE, _line_of("float(s) == 1.0"))
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["s"]
        self.assertIsInstance(val, str)
        self.assertEqual(float(val), 1.0)


# ---------------------------------------------------------------------------
# P3: Dict key shape synthesis
# ---------------------------------------------------------------------------


class TestDictSubscriptGuard(unittest.TestCase):
    """data["name"] and data["age"] > 0 → dict with both keys."""

    def test_dict_subscript(self):
        sol = solve_condition(FIXTURE, _line_of('data["name"] and data["age"] > 0'))
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["data"]
        self.assertIsInstance(val, dict)
        self.assertIn("name", val)
        self.assertIn("age", val)
        self.assertGreater(val["age"], 0)


class TestDictGetGuard(unittest.TestCase):
    """config.get("host") and config.get("port") → dict with both keys."""

    def test_dict_get(self):
        sol = solve_condition(FIXTURE, _line_of('config.get("host") and config.get("port")'))
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["config"]
        self.assertIsInstance(val, dict)
        self.assertIn("host", val)
        self.assertIn("port", val)


# ---------------------------------------------------------------------------
# P4: Byte interior containment
# ---------------------------------------------------------------------------


class TestByteInteriorContainment(unittest.TestCase):
    r"""b"\x1b" in data → data contains the escape byte."""

    def test_escape_byte(self):
        sol = solve_condition(FIXTURE, _line_of('b"\\x1b" in data'))
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["data"]
        self.assertIsInstance(val, bytes)
        self.assertIn(b"\x1b", val)


class TestByteInteriorNull(unittest.TestCase):
    r"""b"\x00" in data and len(data) > 4 → bytes with null and length > 4."""

    def test_null_byte_with_length(self):
        sol = solve_condition(FIXTURE, _line_of('b"\\x00" in data and len(data) > 4'))
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["data"]
        self.assertIsInstance(val, bytes)
        self.assertIn(b"\x00", val)
        self.assertGreater(len(val), 4)
