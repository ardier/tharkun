"""Tests for the emit module — BranchSolution → pytest file rendering."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from tharkun.branch_solver import BranchAttempt, BranchSolution, CoverageResult
from tharkun.emit import (
    _render_value,
    _sanitize_name,
    emit_branch_tests,
)
from tharkun.mock_gen import PatchSpec


class TestRenderValue(unittest.TestCase):
    def test_none(self) -> None:
        self.assertEqual(_render_value(None), "None")

    def test_bool(self) -> None:
        self.assertEqual(_render_value(True), "True")
        self.assertEqual(_render_value(False), "False")

    def test_int(self) -> None:
        self.assertEqual(_render_value(42), "42")

    def test_float(self) -> None:
        self.assertEqual(_render_value(3.14), "3.14")

    def test_str(self) -> None:
        self.assertEqual(_render_value("hello"), "'hello'")

    def test_bytes(self) -> None:
        self.assertEqual(_render_value(b"\x00"), "b'\\x00'")

    def test_list(self) -> None:
        self.assertEqual(_render_value([1, 2]), "[1, 2]")

    def test_dict(self) -> None:
        result = _render_value({"a": 1})
        self.assertIn("'a'", result)
        self.assertIn("1", result)

    def test_tuple(self) -> None:
        self.assertEqual(_render_value((1, 2)), "(1, 2)")
        self.assertEqual(_render_value((1,)), "(1,)")

    def test_empty_set(self) -> None:
        self.assertEqual(_render_value(set()), "set()")

    def test_simple_namespace(self) -> None:
        ns = SimpleNamespace(kind="method")
        result = _render_value(ns)
        self.assertIn("SimpleNamespace", result)
        self.assertIn("kind", result)
        self.assertIn("method", result)


class TestSanitizeName(unittest.TestCase):
    def test_clean_name(self) -> None:
        self.assertEqual(_sanitize_name("foo_bar"), "foo_bar")

    def test_special_chars(self) -> None:
        self.assertEqual(_sanitize_name("foo.bar-baz"), "foo_bar_baz")

    def test_empty(self) -> None:
        self.assertEqual(_sanitize_name(""), "unknown")


class TestEmitBranchTests(unittest.TestCase):
    def _make_coverage(self, solutions: list[BranchSolution]) -> CoverageResult:
        return CoverageResult(
            source_path="/tmp/example.py",
            total_branches=len(solutions),
            solved_branches=sum(1 for s in solutions if s.solved),
            unsolved_branches=sum(1 for s in solutions if not s.solved),
            solutions=solutions,
        )

    def test_no_solved_branches(self) -> None:
        cov = self._make_coverage([
            BranchSolution(file="/tmp/example.py", lineno=5, solved=False),
        ])
        result = emit_branch_tests(cov, module_name="example")
        self.assertIn("No solved branches", result)

    def test_simple_solved_branch(self) -> None:
        sol = BranchSolution(
            file="/tmp/example.py",
            lineno=5,
            guard_source="x > 100",
            callable_id="check_value",
            solved=True,
            best_inputs={"x": 101},
        )
        cov = self._make_coverage([sol])
        result = emit_branch_tests(cov, module_name="example")

        self.assertIn("import unittest", result)
        self.assertIn("from example import *", result)
        self.assertIn("def test_check_value_line5_0", result)
        self.assertIn("check_value(x=101)", result)
        self.assertIn("class Test", result)

    def test_mock_patches_rendered(self) -> None:
        sol = BranchSolution(
            file="/tmp/example.py",
            lineno=10,
            guard_source="response.ok",
            callable_id="fetch_data",
            solved=True,
            best_inputs={"url": "http://test"},
            mock_patches=[
                PatchSpec(
                    target="requests.get",
                    return_value='Mock(status_code=200)',
                    category="http",
                ),
            ],
        )
        cov = self._make_coverage([sol])
        result = emit_branch_tests(cov, module_name="example")

        self.assertIn("from unittest.mock import", result)
        self.assertIn('@patch("requests.get"', result)
        self.assertIn("mock_requests_get", result)

    def test_state_sequence_rendered(self) -> None:
        sol = BranchSolution(
            file="/tmp/example.py",
            lineno=20,
            guard_source="self.state == 'connected'",
            callable_id="send",
            solved=True,
            best_inputs={"__sequence__": {
                "class": "Connection",
                "setup_methods": ["connect"],
                "target_line": 20,
            }},
        )
        cov = self._make_coverage([sol])
        result = emit_branch_tests(cov, module_name="example")

        self.assertIn("obj = Connection()", result)
        self.assertIn("obj.connect()", result)
        self.assertIn("obj.send()", result)

    def test_multiple_solutions(self) -> None:
        sol1 = BranchSolution(
            file="/tmp/example.py", lineno=5,
            guard_source="x > 0", callable_id="f",
            solved=True, best_inputs={"x": 1},
        )
        sol2 = BranchSolution(
            file="/tmp/example.py", lineno=10,
            guard_source="y == 'hello'", callable_id="g",
            solved=True, best_inputs={"y": "hello"},
        )
        unsolved = BranchSolution(
            file="/tmp/example.py", lineno=15,
            solved=False,
        )
        cov = self._make_coverage([sol1, sol2, unsolved])
        result = emit_branch_tests(cov, module_name="example")

        self.assertIn("test_f_line5_0", result)
        self.assertIn("test_g_line10_1", result)
        # Unsolved branch should not produce a test
        self.assertNotIn("line15", result)

    def test_custom_class_name(self) -> None:
        sol = BranchSolution(
            file="/tmp/example.py", lineno=5,
            callable_id="f", solved=True, best_inputs={"x": 1},
        )
        cov = self._make_coverage([sol])
        result = emit_branch_tests(cov, module_name="example", class_name="MyTests")
        self.assertIn("class MyTests(unittest.TestCase):", result)

    def test_mock_open_rendered(self) -> None:
        sol = BranchSolution(
            file="/tmp/example.py", lineno=5,
            callable_id="read_file", solved=True,
            best_inputs={"path": "/tmp/f"},
            mock_patches=[
                PatchSpec(
                    target="builtins.open",
                    return_value=None,
                    setup_lines=['mock_open_obj = mock_open(read_data=b"")'],
                    category="filesystem",
                ),
            ],
        )
        cov = self._make_coverage([sol])
        result = emit_branch_tests(cov, module_name="example")
        self.assertIn("mock_open", result)

    def test_output_is_valid_python(self) -> None:
        sol = BranchSolution(
            file="/tmp/example.py", lineno=5,
            guard_source="x > 0", callable_id="f",
            solved=True, best_inputs={"x": 1},
        )
        cov = self._make_coverage([sol])
        result = emit_branch_tests(cov, module_name="example")
        # Should compile without SyntaxError
        compile(result, "<test>", "exec")


if __name__ == "__main__":
    unittest.main()
