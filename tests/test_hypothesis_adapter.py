"""Tests for the Hypothesis property-based search adapter."""
from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from tharkun.hypothesis_adapter import (
    _infer_param_types,
    _strategy_for_type,
    hypothesis_available,
    hypothesis_solve_branch,
)


FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "condition_guards.py")


class TestHypothesisAvailable(unittest.TestCase):
    def test_returns_bool(self) -> None:
        result = hypothesis_available()
        self.assertIsInstance(result, bool)


class TestStrategyForType(unittest.TestCase):
    def test_int(self) -> None:
        strat = _strategy_for_type("int")
        if hypothesis_available():
            self.assertIsNotNone(strat)
        else:
            self.assertIsNone(strat)

    def test_str(self) -> None:
        strat = _strategy_for_type("str")
        if hypothesis_available():
            self.assertIsNotNone(strat)

    def test_unknown(self) -> None:
        strat = _strategy_for_type("SomeCustomType")
        self.assertIsNone(strat)


class TestInferParamTypes(unittest.TestCase):
    def test_typed_function(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def foo(x: int, y: str) -> bool:\n    return x > 0\n")
            path = f.name
        try:
            types = _infer_param_types(path, "foo")
            self.assertEqual(types["x"], "int")
            self.assertEqual(types["y"], "str")
        finally:
            os.unlink(path)

    def test_untyped_function(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def bar(a, b):\n    return a + b\n")
            path = f.name
        try:
            types = _infer_param_types(path, "bar")
            self.assertEqual(types["a"], "Any")
            self.assertEqual(types["b"], "Any")
        finally:
            os.unlink(path)

    def test_self_excluded(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("class C:\n    def m(self, x: int):\n        pass\n")
            path = f.name
        try:
            types = _infer_param_types(path, "m")
            self.assertNotIn("self", types)
            self.assertIn("x", types)
        finally:
            os.unlink(path)

    def test_nonexistent_function(self) -> None:
        types = _infer_param_types(FIXTURE, "does_not_exist")
        self.assertEqual(types, {})

    def test_nonexistent_file(self) -> None:
        types = _infer_param_types("/nonexistent/file.py", "foo")
        self.assertEqual(types, {})


class TestHypothesisSolveBranch(unittest.TestCase):
    @unittest.skipUnless(hypothesis_available(), "hypothesis not installed")
    def test_solves_simple_guard(self) -> None:
        # Target line 3 (the body `return True`) which requires x > 100
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def check(x: int):\n    if x > 100:\n        return True\n    return False\n")
            path = f.name
        try:
            result = hypothesis_solve_branch(path, 3, "check", max_examples=500)
            self.assertIsNotNone(result)
            self.assertIn("x", result)
            self.assertGreater(result["x"], 100)
        finally:
            os.unlink(path)

    @unittest.skipUnless(hypothesis_available(), "hypothesis not installed")
    def test_returns_none_for_impossible(self) -> None:
        # Target line 3 (the body of `if False:`) which is truly unreachable
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def impossible(x: int):\n    if False:\n        return True\n    return False\n")
            path = f.name
        try:
            result = hypothesis_solve_branch(path, 3, "impossible", max_examples=50)
            self.assertIsNone(result)
        finally:
            os.unlink(path)


class TestHypothesisSolveBranchDeadline(unittest.TestCase):
    """[internal ref] part 2: hypothesis_solve_branch() never set a deadline, so it
    inherited Hypothesis's 200 ms per-example default.  Under BenchExec host
    load a perfectly fine example can take longer than that, Hypothesis raises
    DeadlineExceeded, and the old bare `except Exception: return None` turned
    that into an indistinguishable "no value found" -- silently losing a FALSE
    the search would otherwise have caught.
    """

    @unittest.skipUnless(hypothesis_available(), "hypothesis not installed")
    def test_slow_predicate_still_finds_a_value(self) -> None:
        # Every call sleeps well past the 200 ms default deadline. Before the
        # fix (no deadline=None), the first slow example would raise
        # DeadlineExceeded and the adapter would swallow it as None.
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(
                "def slow_check(x: int):\n"
                "    import time\n"
                "    time.sleep(0.25)\n"
                "    if x > 0:\n"
                "        return True\n"
                "    return False\n"
            )
            path = f.name
        try:
            result = hypothesis_solve_branch(path, 5, "slow_check", max_examples=15)
            self.assertIsNotNone(result)
            self.assertIn("x", result)
            self.assertGreater(result["x"], 0)
        finally:
            os.unlink(path)

    @unittest.skipUnless(hypothesis_available(), "hypothesis not installed")
    def test_non_notfound_error_is_not_swallowed_as_none(self) -> None:
        # A genuine internal Hypothesis error (anything but NoSuchExample) must
        # propagate rather than collapse into the same None a real "not found"
        # returns -- a caller (e.g. branch_solver's infeasibility vote) needs to
        # tell "the search broke" apart from "the search ran and found nothing".
        from hypothesis.errors import InvalidArgument

        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def check(x: int):\n    if x > 0:\n        return True\n    return False\n")
            path = f.name
        try:
            with mock.patch("hypothesis.find", side_effect=InvalidArgument("boom")):
                with self.assertRaises(InvalidArgument):
                    hypothesis_solve_branch(path, 3, "check", max_examples=10)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
