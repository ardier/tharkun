"""Tests for [internal ref] slice 4: branch_solver.py's tools ported into drive_to_line's
cheap-tier portfolio (``_flip_regex``, ``_flip_attr``, ``_flip_decompose``).

Each technique gets three kinds of coverage:

* an ISOLATED unit test calling the flipper function directly, so a failure points
  straight at the flipper's own logic;
* an END-TO-END ``drive_to_line`` test with ``flippers=[that_one]`` and
  ``heavy=False, minleaf=False`` — nothing else in the portfolio can reach the
  target, so REACHED with ``history`` attributing the named technique is proof the
  new flipper (not some other tier) did the work; a companion CONTROL test with
  ``flippers=[]`` confirms the target is genuinely unreachable without it, not
  trivially reachable by the seed alone;
* the documented SCOPE BOUNDARIES (desired == "else", an "or" tree, a compound
  top-level condition, a guard with no pattern/attr/call at all) abstain cleanly —
  ``(None, tech, False)`` — rather than propose something wrong. This is what
  keeps a wrong guess free: every candidate any flipper returns is re-executed and
  verified by ``drive_to_line`` before it is ever committed (see driver.py's own
  module docstring), so these tests are about not WASTING a round on a guess that
  can never verify, not about soundness — an unverified wrong guess was never
  possible here to begin with.
"""
from __future__ import annotations

import os
import tempfile
import textwrap
import unittest

from tharkun.driver import PLATEAU, REACHED, _flip_attr, _flip_decompose, _flip_regex, drive_to_line


def _write(src: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "mod.py")
    with open(p, "w") as f:
        f.write(textwrap.dedent(src))
    return p


class TestFlipRegex(unittest.TestCase):
    SRC = """
        import re

        def target(s):
            if re.fullmatch(r"[a-f0-9]{8}", s):
                return "hex"
            return "no"
    """

    def test_isolated_generates_matching_value(self) -> None:
        p = _write(self.SRC)
        got, tech, infeasible = _flip_regex(p, 5, "then", {}, {"s"}, "True")
        self.assertEqual(tech, "regex")
        self.assertFalse(infeasible)
        self.assertIsNotNone(got)
        import re
        self.assertRegex(got["s"], r"^[a-f0-9]{8}$")

    def test_end_to_end_reached_by_regex_alone(self) -> None:
        p = _write(self.SRC)
        r = drive_to_line(p, "target", 6, {"s": "zzz"},
                          flippers=[_flip_regex], heavy=False, minleaf=False)
        self.assertEqual(r.status, REACHED)
        self.assertEqual([s.technique for s in r.history], ["regex"])

    def test_control_unreachable_without_it(self) -> None:
        p = _write(self.SRC)
        r = drive_to_line(p, "target", 6, {"s": "zzz"},
                          flippers=[], heavy=False, minleaf=False)
        self.assertEqual(r.status, PLATEAU)

    def test_abstains_when_guard_has_no_pattern(self) -> None:
        p = _write("""
            def f(x):
                if x > 5:
                    return "hit"
                return "miss"
        """)
        got, tech, infeasible = _flip_regex(p, 3, "then", {}, {"x"}, "True")
        self.assertIsNone(got)
        self.assertEqual(tech, "regex")
        self.assertFalse(infeasible)


class TestFlipAttr(unittest.TestCase):
    SRC = """
        class Widget:
            def __init__(self, count):
                self.count = count

            def check(self):
                if self.count == 42:
                    return "hit"
                return "miss"

        def target(count):
            w = Widget(count)
            return w.check()
    """

    def test_isolated_finds_init_passthrough(self) -> None:
        p = _write(self.SRC)
        got, tech, infeasible = _flip_attr(p, 7, "then", {}, {"count"}, "True")
        self.assertEqual((got, tech, infeasible), ({"count": 42}, "attr", False))

    def test_end_to_end_reached_by_attr_alone(self) -> None:
        p = _write(self.SRC)
        r = drive_to_line(p, "target", 8, {"count": 0},
                          flippers=[_flip_attr], heavy=False, minleaf=False)
        self.assertEqual(r.status, REACHED)
        self.assertEqual([s.technique for s in r.history], ["attr"])

    def test_control_unreachable_without_it(self) -> None:
        p = _write(self.SRC)
        r = drive_to_line(p, "target", 8, {"count": 0},
                          flippers=[], heavy=False, minleaf=False)
        self.assertEqual(r.status, PLATEAU)

    def test_abstains_on_else_branch(self) -> None:
        p = _write(self.SRC)
        got, tech, infeasible = _flip_attr(p, 7, "else", {}, {"count"}, "True")
        self.assertIsNone(got)
        self.assertEqual(tech, "attr")
        self.assertFalse(infeasible)


class TestFlipDecompose(unittest.TestCase):
    SRC = """
        def we_are_good(config):
            return config > 0 and config < 100

        def target(x):
            if we_are_good(x):
                return "good"
            return "bad"
    """

    def test_isolated_solves_through_the_call(self) -> None:
        p = _write(self.SRC)
        got, tech, infeasible = _flip_decompose(p, 6, "then", {}, {"x"}, "True")
        self.assertEqual(tech, "decompose")
        self.assertFalse(infeasible)
        self.assertIsNotNone(got)
        self.assertIn("x", got)
        self.assertTrue(0 < got["x"] < 100)

    def test_end_to_end_reached_by_decompose_alone(self) -> None:
        p = _write(self.SRC)
        r = drive_to_line(p, "target", 7, {"x": -5},
                          flippers=[_flip_decompose], heavy=False, minleaf=False)
        self.assertEqual(r.status, REACHED)
        self.assertEqual([s.technique for s in r.history], ["decompose"])

    def test_control_unreachable_without_it(self) -> None:
        p = _write(self.SRC)
        r = drive_to_line(p, "target", 7, {"x": -5},
                          flippers=[], heavy=False, minleaf=False)
        self.assertEqual(r.status, PLATEAU)

    def test_abstains_on_else_branch(self) -> None:
        p = _write(self.SRC)
        got, tech, infeasible = _flip_decompose(p, 6, "else", {}, {"x"}, "True")
        self.assertIsNone(got)
        self.assertEqual(tech, "decompose")
        self.assertFalse(infeasible)

    def test_abstains_on_or_tree(self) -> None:
        p = _write("""
            def helper(x):
                return x > 100 or x < 0

            def f(x):
                if helper(x):
                    return "hit"
                return "miss"
        """)
        got, tech, infeasible = _flip_decompose(p, 6, "then", {}, {"x"}, "True")
        self.assertIsNone(got)
        self.assertEqual(tech, "decompose")
        self.assertFalse(infeasible)

    def test_abstains_on_compound_top_level_condition(self) -> None:
        p = _write("""
            def helper(x):
                return x > 0 and x < 100

            def f(a, x):
                if a and helper(x):
                    return "hit"
                return "miss"
        """)
        got, tech, infeasible = _flip_decompose(p, 6, "then", {}, {"a", "x"}, "True")
        self.assertIsNone(got)
        self.assertEqual(tech, "decompose")
        self.assertFalse(infeasible)


class TestDefaultPortfolioIncludesAll(unittest.TestCase):
    """The default portfolio (flippers=None) still reaches each target when the
    new flippers are mixed in with the pre-existing ones — a plain integration
    check that adding them did not disturb ordering or the existing cheap tier."""

    def test_regex_reachable_by_default_portfolio(self) -> None:
        p = _write(TestFlipRegex.SRC)
        r = drive_to_line(p, "target", 6, {"s": "zzz"})
        self.assertEqual(r.status, REACHED)

    def test_attr_reachable_by_default_portfolio(self) -> None:
        p = _write(TestFlipAttr.SRC)
        r = drive_to_line(p, "target", 8, {"count": 0})
        self.assertEqual(r.status, REACHED)

    def test_decompose_reachable_by_default_portfolio(self) -> None:
        p = _write(TestFlipDecompose.SRC)
        r = drive_to_line(p, "target", 7, {"x": -5})
        self.assertEqual(r.status, REACHED)


if __name__ == "__main__":
    unittest.main()
