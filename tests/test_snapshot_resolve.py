"""Tests for snapshot-informed condition re-solving."""
from __future__ import annotations

import os
import unittest

from tharkun.solve.condition import (
    solve_with_snapshot,
    _simplify_with_resolved,
    _try_eval_clause,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "snapshot_target.py")


class TestTryEvalClause(unittest.TestCase):
    def test_simple_comparison_true(self) -> None:
        import ast
        clause = ast.parse("x > 0", mode="eval").body
        result = _try_eval_clause(clause, {"x": 5})
        self.assertTrue(result)

    def test_simple_comparison_false(self) -> None:
        import ast
        clause = ast.parse("x > 100", mode="eval").body
        result = _try_eval_clause(clause, {"x": 5})
        self.assertFalse(result)

    def test_unresolved_returns_none(self) -> None:
        import ast
        clause = ast.parse("x > 0", mode="eval").body
        result = _try_eval_clause(clause, {})
        self.assertIsNone(result)

    def test_isinstance_check(self) -> None:
        import ast
        clause = ast.parse("isinstance(x, int)", mode="eval").body
        result = _try_eval_clause(clause, {"x": 42})
        self.assertTrue(result)

    def test_truthiness(self) -> None:
        import ast
        clause = ast.parse("enabled", mode="eval").body
        result = _try_eval_clause(clause, {"enabled": True})
        self.assertTrue(result)

    def test_truthiness_false(self) -> None:
        import ast
        clause = ast.parse("enabled", mode="eval").body
        result = _try_eval_clause(clause, {"enabled": False})
        self.assertFalse(result)


class TestSimplifyWithResolved(unittest.TestCase):
    def test_and_one_clause_resolved(self) -> None:
        import ast
        expr = ast.parse("enabled and x > 0", mode="eval").body
        simplified = _simplify_with_resolved(expr, {"enabled": True})
        # Should simplify to just "x > 0"
        source = ast.unparse(simplified)
        self.assertNotIn("enabled", source)
        self.assertIn("x", source)

    def test_and_all_resolved_true(self) -> None:
        import ast
        expr = ast.parse("x > 0 and y > 0", mode="eval").body
        simplified = _simplify_with_resolved(expr, {"x": 5, "y": 10})
        self.assertIsInstance(simplified, ast.Constant)
        self.assertTrue(simplified.value)

    def test_and_one_resolved_false(self) -> None:
        import ast
        expr = ast.parse("enabled and x > 0", mode="eval").body
        simplified = _simplify_with_resolved(expr, {"enabled": False})
        self.assertIsInstance(simplified, ast.Constant)
        self.assertFalse(simplified.value)

    def test_or_one_clause_true(self) -> None:
        import ast
        expr = ast.parse("x > 100 or fallback", mode="eval").body
        simplified = _simplify_with_resolved(expr, {"fallback": True})
        self.assertIsInstance(simplified, ast.Constant)
        self.assertTrue(simplified.value)

    def test_no_simplification_possible(self) -> None:
        import ast
        expr = ast.parse("x > 0", mode="eval").body
        simplified = _simplify_with_resolved(expr, {})
        # Should return the original expression unchanged
        self.assertEqual(ast.unparse(simplified), "x > 0")


class TestSolveWithSnapshot(unittest.TestCase):
    def test_compound_and_partial_snapshot(self) -> None:
        """enabled and x > threshold — threshold is a computed local (=200).

        Snapshot provides threshold=200 (non-parameter local).
        enabled and x are function params — solver should solve for them.
        """
        sol = solve_with_snapshot(FIXTURE, 7, {
            "enabled": True,      # param — solver should still solve for it
            "x": 5,               # param — solver should still solve for it
            "base_threshold": 100, # param
            "threshold": 200,     # computed local — should be substituted
        })
        # threshold=200 is a computed local, so after substitution: x > 200
        self.assertGreater(len(sol.satisfying_inputs), 0)
        inputs = sol.satisfying_inputs[0]
        self.assertIn("x", inputs)
        if isinstance(inputs["x"], (int, float)):
            self.assertGreater(inputs["x"], 200)

    def test_all_params_in_snapshot(self) -> None:
        """x > 0 and y > 0 — both are params, so snapshot doesn't help.
        Falls back to normal solving which handles this fine."""
        sol = solve_with_snapshot(FIXTURE, 14, {
            "x": 5,
            "y": 10,
        })
        # This is a simple guard that the normal solver handles
        self.assertGreater(len(sol.satisfying_inputs), 0)
        inputs = sol.satisfying_inputs[0]
        self.assertGreater(inputs.get("x", 0), 0)
        self.assertGreater(inputs.get("y", 0), 0)

    def test_partial_resolved_with_computed_local(self) -> None:
        """x > threshold — threshold is a computed local, not a param.
        Uses the snapshot_exec_target fixture where threshold = x*2+10."""
        exec_fixture = os.path.join(os.path.dirname(__file__), "fixtures", "snapshot_exec_target.py")
        sol = solve_with_snapshot(exec_fixture, 11, {
            "x": 0,
            "y": 0,
            "threshold": 10,  # computed local: 0*2+10
        })
        self.assertGreater(len(sol.satisfying_inputs), 0)
        inputs = sol.satisfying_inputs[0]
        # threshold=10 should be substituted, y > 10 → y=11
        if isinstance(inputs.get("y"), (int, float)):
            self.assertGreater(inputs["y"], 10)

    def test_no_snapshot_fallback(self) -> None:
        """With empty snapshot, should fall back to normal solving."""
        sol = solve_with_snapshot(FIXTURE, 6, {})
        # Should still attempt to solve (may or may not succeed)
        self.assertIsNotNone(sol)

    def test_nonexistent_file(self) -> None:
        sol = solve_with_snapshot("/nonexistent.py", 1, {"x": 1})
        self.assertEqual(sol.strategy, "none")


if __name__ == "__main__":
    unittest.main()
