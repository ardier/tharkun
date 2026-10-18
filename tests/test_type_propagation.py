"""Tests for cross-function type propagation."""
from __future__ import annotations

import os
import unittest

from tharkun.solve.frontend.type_propagation import propagate_types

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "propagation_target.py")


class TestPropagateTypes(unittest.TestCase):
    def test_untyped_caller_gets_types(self) -> None:
        """untyped_caller(a, b) calls typed_callee(x: int, name: str)."""
        types = propagate_types(FIXTURE, "untyped_caller")
        self.assertEqual(types.get("a"), "int")
        self.assertEqual(types.get("b"), "str")

    def test_partial_caller(self) -> None:
        """partial_caller(a, b: str) — a should get int from typed_callee."""
        types = propagate_types(FIXTURE, "partial_caller")
        self.assertEqual(types.get("a"), "int")
        self.assertEqual(types.get("b"), "str")  # already annotated

    def test_keyword_caller(self) -> None:
        """keyword_caller(val, label) calls typed_callee(x=val, name=label)."""
        types = propagate_types(FIXTURE, "keyword_caller")
        self.assertEqual(types.get("val"), "int")
        self.assertEqual(types.get("label"), "str")

    def test_no_propagation(self) -> None:
        """no_propagation calls an undefined function — no types to propagate."""
        types = propagate_types(FIXTURE, "no_propagation")
        self.assertNotIn("a", types)
        self.assertNotIn("b", types)

    def test_fully_typed(self) -> None:
        """fully_typed already has annotations — returns them as-is."""
        types = propagate_types(FIXTURE, "fully_typed")
        self.assertEqual(types.get("x"), "int")
        self.assertEqual(types.get("y"), "float")

    def test_nonexistent_function(self) -> None:
        types = propagate_types(FIXTURE, "does_not_exist")
        self.assertEqual(types, {})

    def test_nonexistent_file(self) -> None:
        types = propagate_types("/nonexistent/file.py", "foo")
        self.assertEqual(types, {})


if __name__ == "__main__":
    unittest.main()
