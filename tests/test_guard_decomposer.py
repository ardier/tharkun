"""Tests for the guard decomposer module."""
from __future__ import annotations

import os
import unittest

from tharkun.solve.decompose import (
    DecompNode,
    DecompResult,
    decompose_guard,
    serialize_decomp,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "decompose_target.py")


class TestSimpleBooleanFunction(unittest.TestCase):
    """is_valid(x) → x > 0 and x < 100 — should decompose into 2 solvable leaves."""

    def test_resolves_and_decomposes(self) -> None:
        result = decompose_guard(FIXTURE, 10)  # if is_valid(x):
        self.assertIn("is_valid", result.guard_source)
        self.assertEqual(result.root.kind, "call")
        self.assertEqual(result.root.resolved_function, "is_valid")
        # The return is x > 0 and x < 100 → "and" node with 2 leaves
        self.assertEqual(len(result.root.children), 1)
        and_node = result.root.children[0]
        self.assertEqual(and_node.kind, "and")
        self.assertEqual(len(and_node.children), 2)

    def test_both_leaves_solvable(self) -> None:
        result = decompose_guard(FIXTURE, 10)
        self.assertEqual(result.total_leaves, 2)
        self.assertEqual(result.solvable_leaves, 2)
        self.assertEqual(result.unsolvable_leaves, 0)
        self.assertEqual(result.blockers, [])


class TestGuardClauseFunction(unittest.TestCase):
    """check_data(data) → guard clauses + return True."""

    def test_resolves_guard_clauses(self) -> None:
        result = decompose_guard(FIXTURE, 24)  # if check_data(data):
        self.assertEqual(result.root.kind, "call")
        self.assertEqual(result.root.resolved_function, "check_data")
        # Should have children from the decomposed guard conditions
        self.assertGreater(result.total_leaves, 0)


class TestNestedCalls(unittest.TestCase):
    """outer(x) → inner1(x) and inner2(x) — two levels deep."""

    def test_resolves_two_levels(self) -> None:
        result = decompose_guard(FIXTURE, 42)  # if outer(x):
        self.assertEqual(result.root.kind, "call")
        self.assertEqual(result.root.resolved_function, "outer")
        # outer returns inner1(x) and inner2(x)
        and_node = result.root.children[0]
        self.assertEqual(and_node.kind, "and")
        # Each child should be a resolved call
        self.assertEqual(len(and_node.children), 2)
        for child in and_node.children:
            self.assertEqual(child.kind, "call")
            self.assertIsNotNone(child.resolved_function)

    def test_all_solvable(self) -> None:
        result = decompose_guard(FIXTURE, 42)
        self.assertGreater(result.solvable_leaves, 0)
        self.assertEqual(result.unsolvable_leaves, 0)


class TestOpaqueExternalCall(unittest.TestCase):
    """external_lib.validate(x) — unresolved function → blocker."""

    def test_external_call_is_blocker(self) -> None:
        result = decompose_guard(FIXTURE, 48)  # if external_lib.validate(x):
        self.assertGreater(result.unsolvable_leaves, 0)
        self.assertGreater(len(result.blockers), 0)
        blocker = result.blockers[0]
        self.assertIn("unresolved", blocker.blocker_reason or "")


class TestMixedSolvability(unittest.TestCase):
    """partial_check: x > 0 and len(data) >= 3 and external_validate(data) → 2 solvable, 1 opaque."""

    def test_partial_solvability(self) -> None:
        result = decompose_guard(FIXTURE, 58)  # if partial_check(x, data):
        self.assertGreater(result.solvable_leaves, 0)
        self.assertGreater(result.unsolvable_leaves, 0)
        # Should identify external_validate as the blocker
        self.assertGreater(len(result.blockers), 0)

    def test_two_solvable_one_blocker(self) -> None:
        result = decompose_guard(FIXTURE, 58)
        self.assertGreaterEqual(result.solvable_leaves, 2)
        self.assertGreaterEqual(result.unsolvable_leaves, 1)


class TestDirectGuard(unittest.TestCase):
    """No function call — x > 10 and name == 'admin' — direct decomposition."""

    def test_decomposes_and(self) -> None:
        result = decompose_guard(FIXTURE, 86)  # if x > 10 and name == "admin":
        self.assertEqual(result.root.kind, "and")
        self.assertEqual(len(result.root.children), 2)
        self.assertEqual(result.total_leaves, 2)
        self.assertEqual(result.solvable_leaves, 2)


class TestDeepNesting(unittest.TestCase):
    """level1 → level2 → level3 — three levels of call resolution."""

    def test_three_levels_deep(self) -> None:
        result = decompose_guard(FIXTURE, 104)  # if level1(x):
        self.assertEqual(result.root.kind, "call")
        self.assertEqual(result.root.resolved_function, "level1")
        self.assertGreater(result.max_depth, 2)
        # x != 0 is solvable; x > -100 may not be (negative literal is
        # UnaryOp in AST, not Constant — known condition_solver limitation).
        self.assertGreater(result.solvable_leaves, 0)
        self.assertGreaterEqual(result.total_leaves, 2)


class TestOrReturnFunction(unittest.TestCase):
    """accepts_many → isinstance(x, int) or isinstance(x, str)."""

    def test_or_decomposition(self) -> None:
        result = decompose_guard(FIXTURE, 114)  # if accepts_many(x):
        self.assertEqual(result.root.kind, "call")
        or_node = result.root.children[0]
        self.assertEqual(or_node.kind, "or")
        self.assertEqual(len(or_node.children), 2)


class TestIfTrueReturnFalse(unittest.TestCase):
    """is_positive: if n > 0: return True; return False → extracts n > 0."""

    def test_extracts_condition(self) -> None:
        result = decompose_guard(FIXTURE, 80)  # if is_positive(n):
        self.assertEqual(result.root.kind, "call")
        self.assertEqual(result.root.resolved_function, "is_positive")
        self.assertGreater(result.solvable_leaves, 0)


class TestGuardWithRaise(unittest.TestCase):
    """strict_check: raise on None/negative, return True — guard clauses with raise."""

    def test_raise_guard_clauses(self) -> None:
        result = decompose_guard(FIXTURE, 128)  # if strict_check(x):
        self.assertEqual(result.root.kind, "call")
        self.assertEqual(result.root.resolved_function, "strict_check")
        # Should decompose the guard clauses
        self.assertGreater(result.total_leaves, 0)


class TestNonexistentFile(unittest.TestCase):
    def test_returns_empty_result(self) -> None:
        result = decompose_guard("/nonexistent/path.py", 1)
        self.assertEqual(result.unsolvable_leaves, 1)
        self.assertEqual(result.blockers[0].blocker_reason, "unreadable_file")


class TestNoGuardAtLine(unittest.TestCase):
    def test_returns_empty_result(self) -> None:
        result = decompose_guard(FIXTURE, 5)  # def is_valid — not a guard
        self.assertEqual(result.unsolvable_leaves, 1)
        self.assertEqual(result.blockers[0].blocker_reason, "no_branch_at_line")


class TestSerializeDecomp(unittest.TestCase):
    def test_serializes_to_dict(self) -> None:
        result = decompose_guard(FIXTURE, 10)
        d = serialize_decomp(result)
        self.assertIn("guard_source", d)
        self.assertIn("tree", d)
        self.assertIn("total_leaves", d)
        self.assertIn("blockers", d)
        self.assertIsInstance(d["tree"], dict)
        self.assertIn("kind", d["tree"])


if __name__ == "__main__":
    unittest.main()
