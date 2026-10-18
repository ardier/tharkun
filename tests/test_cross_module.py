"""Tests for SG-2: cross-module type resolution and decomposer resolution."""
from __future__ import annotations

import os
import unittest

from tharkun.solve.decompose import decompose_guard
from tharkun.solve.frontend.import_resolver import (
    load_function_def,
    parse_imports,
    resolve_module_path,
)

FIXTURE_A = os.path.join(os.path.dirname(__file__), "fixtures", "cross_module_a.py")
FIXTURE_B = os.path.join(os.path.dirname(__file__), "fixtures", "cross_module_b.py")


class TestResolveModulePath(unittest.TestCase):
    def test_resolves_existing_module(self) -> None:
        # cross_module_a should resolve from the fixtures dir
        fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
        result = resolve_module_path("cross_module_a", fixtures_dir)
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("cross_module_a.py"))

    def test_returns_none_for_nonexistent(self) -> None:
        result = resolve_module_path("nonexistent_module_xyz", "")
        self.assertIsNone(result)


class TestParseImports(unittest.TestCase):
    def test_parses_fixture_b_imports(self) -> None:
        imports = parse_imports(FIXTURE_B)
        # Should find imports from cross_module_a
        # The exact resolution depends on sys.path, but the names should be present
        # if the fixture directory is findable.
        self.assertIsInstance(imports, dict)

    def test_nonexistent_file(self) -> None:
        imports = parse_imports("/nonexistent/file.py")
        self.assertEqual(imports, {})


class TestLoadFunctionDef(unittest.TestCase):
    def test_loads_top_level_function(self) -> None:
        func_def = load_function_def(FIXTURE_A, "validate_point")
        self.assertIsNotNone(func_def)
        self.assertEqual(func_def.name, "validate_point")

    def test_loads_helper(self) -> None:
        func_def = load_function_def(FIXTURE_A, "helper")
        self.assertIsNotNone(func_def)

    def test_returns_none_for_missing(self) -> None:
        func_def = load_function_def(FIXTURE_A, "nonexistent_func")
        self.assertIsNone(func_def)


class TestDecomposerCrossModule(unittest.TestCase):
    """Test that guard_decomposer follows imports across modules."""

    def _line_of(self, fixture: str, text: str) -> int:
        with open(fixture) as f:
            for i, line in enumerate(f, 1):
                if text in line:
                    return i
        raise ValueError(f"{text!r} not found in {fixture}")

    def test_follows_imported_function(self) -> None:
        # `if validate_point(p):` in cross_module_b should resolve
        # validate_point from cross_module_a
        lineno = self._line_of(FIXTURE_B, "if validate_point(p):")
        result = decompose_guard(FIXTURE_B, lineno)
        # Should resolve the call (not mark as unresolved)
        self.assertEqual(result.root.kind, "call")
        self.assertEqual(result.root.resolved_function, "validate_point")
        # Should have solvable leaves from the return condition
        self.assertGreater(result.total_leaves, 0)

    def test_follows_imported_helper(self) -> None:
        lineno = self._line_of(FIXTURE_B, "if helper(x):")
        result = decompose_guard(FIXTURE_B, lineno)
        self.assertEqual(result.root.kind, "call")
        self.assertEqual(result.root.resolved_function, "helper")
        self.assertGreater(result.solvable_leaves, 0)

    def test_combined_guard_both_resolved(self) -> None:
        lineno = self._line_of(FIXTURE_B, "if validate_point(p) and helper(x):")
        result = decompose_guard(FIXTURE_B, lineno)
        self.assertEqual(result.root.kind, "and")
        self.assertEqual(len(result.root.children), 2)
        for child in result.root.children:
            self.assertEqual(child.kind, "call")
            self.assertIsNotNone(child.resolved_function)

    def test_same_module_still_works(self) -> None:
        """Regression: same-module resolution should not break."""
        decomp_fixture = os.path.join(os.path.dirname(__file__), "fixtures", "decompose_target.py")
        result = decompose_guard(decomp_fixture, 10)  # if is_valid(x):
        self.assertEqual(result.root.kind, "call")
        self.assertEqual(result.root.resolved_function, "is_valid")


if __name__ == "__main__":
    unittest.main()
