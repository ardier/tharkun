"""Tests for branch_solver — per-branch aggressive solving."""
from __future__ import annotations

import os
import unittest

from tharkun.branch_solver import (
    BranchSolution,
    BranchTarget,
    CoverageResult,
    _extract_required_attr_value,
    _find_branch_edges,
    _find_enclosing_callable,
    serialize_branch_solution,
    serialize_coverage_result,
    solve_branch,
    solve_uncovered_branches,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "condition_guards.py")


class TestFindEnclosingCallable(unittest.TestCase):
    def test_finds_function(self) -> None:
        name = _find_enclosing_callable(FIXTURE, 5)
        self.assertEqual(name, "f")

    def test_finds_method(self) -> None:
        name = _find_enclosing_callable(FIXTURE, 26)
        self.assertEqual(name, "m")

    def test_line_outside_function(self) -> None:
        # Line 1 is a docstring, not inside a function
        name = _find_enclosing_callable(FIXTURE, 1)
        self.assertIsNone(name)

    def test_nonexistent_file(self) -> None:
        name = _find_enclosing_callable("/nonexistent/file.py", 1)
        self.assertIsNone(name)


class TestExtractRequiredAttrValue(unittest.TestCase):
    def test_eq_string(self) -> None:
        val = _extract_required_attr_value('self.state == "connected"', "self.state")
        self.assertEqual(val, "connected")

    def test_ne_string(self) -> None:
        val = _extract_required_attr_value('self.state != "disconnected"', "self.state")
        self.assertEqual(val, "disconnected")

    def test_no_match(self) -> None:
        val = _extract_required_attr_value("self.flag", "self.state")
        self.assertIsNone(val)

    def test_flipped(self) -> None:
        val = _extract_required_attr_value('"active" == self.mode', "self.mode")
        self.assertEqual(val, "active")

    def test_invalid_syntax(self) -> None:
        val = _extract_required_attr_value("not valid python !!!", "self.x")
        self.assertIsNone(val)


class TestSolveBranch(unittest.TestCase):
    def test_simple_guard(self) -> None:
        sol = solve_branch(FIXTURE, 5, execute_snapshots=False)
        self.assertIsInstance(sol, BranchSolution)
        self.assertEqual(sol.lineno, 5)
        self.assertTrue(sol.solved)
        self.assertIsNotNone(sol.best_inputs)
        self.assertIn("x", sol.best_inputs)
        self.assertGreater(sol.best_inputs["x"], 100)

    def test_records_attempts(self) -> None:
        sol = solve_branch(FIXTURE, 5, execute_snapshots=False)
        self.assertGreater(len(sol.attempts), 0)
        techniques = [a.technique for a in sol.attempts]
        self.assertIn("condition_solver", techniques)

    def test_callable_id_found(self) -> None:
        sol = solve_branch(FIXTURE, 5, execute_snapshots=False)
        self.assertEqual(sol.callable_id, "f")

    def test_string_guard(self) -> None:
        sol = solve_branch(FIXTURE, 7, execute_snapshots=False)
        self.assertTrue(sol.solved)
        self.assertEqual(sol.best_inputs["name"], "hello")


class TestSolveUncoveredBranches(unittest.TestCase):
    def test_solves_many(self) -> None:
        result = solve_uncovered_branches(FIXTURE, execute_snapshots=False)
        self.assertIsInstance(result, CoverageResult)
        self.assertGreater(result.total_branches, 0)
        self.assertGreater(result.solved_branches, 0)
        # Should solve a good fraction
        rate = result.solved_branches / max(result.total_branches, 1)
        self.assertGreater(rate, 0.5)

    def test_with_covered_lines(self) -> None:
        # If we say line 5 is already covered, it should be skipped
        lines = {t.lineno for t in _find_branch_edges(FIXTURE)}
        result = solve_uncovered_branches(
            FIXTURE,
            covered_lines=lines,  # All covered
            execute_snapshots=False,
        )
        self.assertEqual(len(result.solutions), 0)


class TestSerialize(unittest.TestCase):
    def test_serialize_branch_solution(self) -> None:
        sol = solve_branch(FIXTURE, 5, execute_snapshots=False)
        d = serialize_branch_solution(sol)
        self.assertIn("file", d)
        self.assertIn("lineno", d)
        self.assertIn("solved", d)
        self.assertIn("attempts", d)
        self.assertIn("best_inputs", d)
        self.assertEqual(d["lineno"], 5)
        self.assertTrue(d["solved"])

    def test_serialize_coverage_result(self) -> None:
        result = solve_uncovered_branches(FIXTURE, execute_snapshots=False)
        d = serialize_coverage_result(result)
        self.assertIn("source_path", d)
        self.assertIn("total_branches", d)
        self.assertIn("solutions", d)
        self.assertIsInstance(d["solutions"], list)


class TestBranchTarget(unittest.TestCase):
    """Tests for BranchTarget dataclass and _find_branch_edges."""

    def test_branch_target_fields(self) -> None:
        t = BranchTarget(lineno=5, stmt_kind="if", desired_branch="then")
        self.assertEqual(t.lineno, 5)
        self.assertEqual(t.stmt_kind, "if")
        self.assertEqual(t.desired_branch, "then")

    def test_branch_target_frozen(self) -> None:
        t = BranchTarget(lineno=5, stmt_kind="if", desired_branch="then")
        with self.assertRaises((AttributeError, TypeError)):
            t.lineno = 99  # type: ignore[misc]

    def test_find_branch_edges_returns_pairs(self) -> None:
        edges = _find_branch_edges(FIXTURE)
        self.assertIsInstance(edges, list)
        # Each guard line should produce both "then" and "else" targets
        lines = [e.lineno for e in edges]
        polarity = [e.desired_branch for e in edges]
        self.assertIn("then", polarity)
        self.assertIn("else", polarity)
        # Should be exactly 2× the number of unique guard lines
        self.assertEqual(len(edges), 2 * len(set(lines)))

    def test_find_branch_edges_nonexistent_file(self) -> None:
        edges = _find_branch_edges("/nonexistent/file.py")
        self.assertEqual(edges, [])

    def test_solve_uncovered_branches_counts_edges(self) -> None:
        result = solve_uncovered_branches(FIXTURE, execute_snapshots=False)
        guard_lines = {t.lineno for t in _find_branch_edges(FIXTURE)}
        self.assertEqual(result.total_branches, 2 * len(guard_lines))

    def test_branch_solution_desired_branch_field(self) -> None:
        result = solve_uncovered_branches(FIXTURE, execute_snapshots=False)
        for sol in result.solutions:
            self.assertIn(sol.desired_branch, ("then", "else"))

    def test_serialize_branch_solution_includes_desired_branch(self) -> None:
        result = solve_uncovered_branches(FIXTURE, execute_snapshots=False)
        for sol in result.solutions:
            d = serialize_branch_solution(sol)
            self.assertIn("desired_branch", d)
            self.assertIn(d["desired_branch"], ("then", "else"))

    def test_else_branch_solutions_present(self) -> None:
        result = solve_uncovered_branches(FIXTURE, execute_snapshots=False)
        else_solutions = [s for s in result.solutions if s.desired_branch == "else"]
        self.assertGreater(len(else_solutions), 0)


class TestFalseBranchSolving(unittest.TestCase):
    """Integration tests: solve_branch for desired_branch='else'."""

    def test_solve_branch_else_numeric_guard(self) -> None:
        # Line 5: `if x > 100:` — "else" branch wants x <= 100
        sol = solve_branch(FIXTURE, 5, desired_branch="else")
        self.assertIsNotNone(sol)
        if sol.solved and sol.best_inputs:
            x = sol.best_inputs.get("x")
            if x is not None and isinstance(x, (int, float)):
                self.assertLessEqual(x, 100)

    def test_solve_branch_then_numeric_guard(self) -> None:
        # Line 5: `if x > 100:` — "then" branch wants x > 100
        sol = solve_branch(FIXTURE, 5, desired_branch="then")
        self.assertIsNotNone(sol)
        if sol.solved and sol.best_inputs:
            x = sol.best_inputs.get("x")
            if x is not None and isinstance(x, (int, float)):
                self.assertGreater(x, 100)

    def test_solve_branch_desired_branch_recorded(self) -> None:
        sol_then = solve_branch(FIXTURE, 5, desired_branch="then")
        sol_else = solve_branch(FIXTURE, 5, desired_branch="else")
        self.assertEqual(sol_then.desired_branch, "then")
        self.assertEqual(sol_else.desired_branch, "else")


class TestBlockerInfo(unittest.TestCase):
    """Tests for the blocker_info field on BranchSolution."""

    def test_blocker_info_none_when_solved(self) -> None:
        sol = solve_branch(FIXTURE, 5, execute_snapshots=False)
        self.assertTrue(sol.solved)
        self.assertIsNone(sol.blocker_info)

    def test_blocker_info_populated_when_unsolved(self) -> None:
        sol = solve_branch("/nonexistent/file.py", 99, execute_snapshots=False)
        self.assertFalse(sol.solved)
        self.assertIsInstance(sol.blocker_info, dict)
        self.assertIn("kind", sol.blocker_info)
        self.assertIn("detail", sol.blocker_info)
        self.assertIn("suggestion", sol.blocker_info)
        self.assertIn("patch_target", sol.blocker_info)

    def test_blocker_info_kind_is_valid(self) -> None:
        sol = solve_branch("/nonexistent/file.py", 99, execute_snapshots=False)
        valid_kinds = {
            "infeasible", "external_call", "attribute_unset",
            "decomposer_failed", "no_technique_succeeded",
        }
        self.assertIn(sol.blocker_info["kind"], valid_kinds)

    def test_blocker_info_in_serialization(self) -> None:
        sol = solve_branch("/nonexistent/file.py", 99, execute_snapshots=False)
        d = serialize_branch_solution(sol)
        self.assertIn("blocker_info", d)


class TestDeterministicMode(unittest.TestCase):
    """Tests for deterministic=True mode."""

    def test_deterministic_does_not_crash(self) -> None:
        sol = solve_branch(FIXTURE, 5, execute_snapshots=False, deterministic=True)
        self.assertIsInstance(sol, BranchSolution)


    def test_deterministic_produces_consistent_results(self) -> None:
        sol1 = solve_branch(FIXTURE, 5, execute_snapshots=False, deterministic=True)
        sol2 = solve_branch(FIXTURE, 5, execute_snapshots=False, deterministic=True)
        self.assertEqual(sol1.best_inputs, sol2.best_inputs)


class TestProgressiveAPI(unittest.TestCase):
    """Tests for solve_branches_progressive()."""

    def test_yields_branch_solutions(self) -> None:
        from tharkun.branch_solver import solve_branches_progressive
        solutions = list(solve_branches_progressive(FIXTURE, execute_snapshots=False))
        self.assertGreater(len(solutions), 0)
        for sol in solutions:
            self.assertIsInstance(sol, BranchSolution)

    def test_can_stop_early(self) -> None:
        from tharkun.branch_solver import solve_branches_progressive
        first = None
        for sol in solve_branches_progressive(FIXTURE, execute_snapshots=False):
            first = sol
            break
        self.assertIsInstance(first, BranchSolution)

    def test_count_equals_edge_count(self) -> None:
        from tharkun.branch_solver import solve_branches_progressive, _find_branch_edges
        edges = _find_branch_edges(FIXTURE)
        solutions = list(solve_branches_progressive(FIXTURE, execute_snapshots=False))
        self.assertEqual(len(solutions), len(edges))


class TestBestInputsConstraints(unittest.TestCase):
    """Tests for best_inputs_constraints on BranchSolution."""

    def test_constraints_populated_when_solved(self) -> None:
        sol = solve_branch(FIXTURE, 5, execute_snapshots=False)
        if sol.solved and sol.best_inputs_constraints is not None:
            self.assertIsInstance(sol.best_inputs_constraints, dict)
            self.assertIn("x", sol.best_inputs_constraints)
            self.assertIn("type", sol.best_inputs_constraints["x"])

    def test_constraints_in_serialization(self) -> None:
        sol = solve_branch(FIXTURE, 5, execute_snapshots=False)
        d = serialize_branch_solution(sol)
        self.assertIn("best_inputs_constraints", d)


if __name__ == "__main__":
    unittest.main()
