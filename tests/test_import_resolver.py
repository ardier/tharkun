"""Tests for import_resolver — cross-module resolution."""
from __future__ import annotations

import os
import tempfile
import unittest

from tharkun.solve.frontend.import_resolver import (
    load_class_def,
    load_function_def,
    parse_imports,
    resolve_module_path,
)


class TestResolveModulePath(unittest.TestCase):
    def test_resolve_existing_module(self) -> None:
        # tharkun itself should be resolvable from src/
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
        result = resolve_module_path("tharkun.solve.condition", src_dir)
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("condition.py"))

    def test_resolve_package_init(self) -> None:
        src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
        result = resolve_module_path("tharkun", src_dir)
        self.assertIsNotNone(result)
        self.assertTrue(result.endswith("__init__.py"))

    def test_resolve_nonexistent(self) -> None:
        result = resolve_module_path("nonexistent.module.xyz", "/tmp")
        self.assertIsNone(result)

    def test_resolve_simple_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "mymod.py")
            with open(path, "w") as f:
                f.write("x = 1\n")
            result = resolve_module_path("mymod", tmpdir)
            self.assertIsNotNone(result)
            self.assertTrue(result.endswith("mymod.py"))


class TestParseImports(unittest.TestCase):
    def test_parse_fixture_imports(self) -> None:
        fixture = os.path.join(os.path.dirname(__file__), "fixtures", "cross_module_a.py")
        if not os.path.exists(fixture):
            self.skipTest("cross_module_a.py fixture not found")
        result = parse_imports(fixture)
        self.assertIsInstance(result, dict)

    def test_parse_nonexistent(self) -> None:
        result = parse_imports("/nonexistent/file.py")
        self.assertEqual(result, {})

    def test_parse_simple_imports(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("import os\nimport sys\n")
            path = f.name
        try:
            result = parse_imports(path)
            # os and sys are stdlib — resolve_module_path may or may not find them
            self.assertIsInstance(result, dict)
        finally:
            os.unlink(path)


class TestLoadFunctionDef(unittest.TestCase):
    def test_load_top_level_function(self) -> None:
        fixture = os.path.join(os.path.dirname(__file__), "fixtures", "condition_guards.py")
        func = load_function_def(fixture, "f")
        self.assertIsNotNone(func)
        self.assertEqual(func.name, "f")

    def test_load_method(self) -> None:
        fixture = os.path.join(os.path.dirname(__file__), "fixtures", "condition_guards.py")
        func = load_function_def(fixture, "C.m")
        self.assertIsNotNone(func)
        self.assertEqual(func.name, "m")

    def test_load_nonexistent_function(self) -> None:
        fixture = os.path.join(os.path.dirname(__file__), "fixtures", "condition_guards.py")
        func = load_function_def(fixture, "does_not_exist")
        self.assertIsNone(func)

    def test_load_from_nonexistent_file(self) -> None:
        func = load_function_def("/nonexistent/file.py", "f")
        self.assertIsNone(func)


class TestLoadClassDef(unittest.TestCase):
    def test_load_class(self) -> None:
        fixture = os.path.join(os.path.dirname(__file__), "fixtures", "condition_guards.py")
        cls = load_class_def(fixture, "C")
        self.assertIsNotNone(cls)
        self.assertEqual(cls.name, "C")

    def test_load_nonexistent_class(self) -> None:
        fixture = os.path.join(os.path.dirname(__file__), "fixtures", "condition_guards.py")
        cls = load_class_def(fixture, "DoesNotExist")
        self.assertIsNone(cls)

    def test_load_from_nonexistent_file(self) -> None:
        cls = load_class_def("/nonexistent/file.py", "C")
        self.assertIsNone(cls)


if __name__ == "__main__":
    unittest.main()
