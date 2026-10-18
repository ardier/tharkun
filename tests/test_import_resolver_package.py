"""Absolute intra-package imports.

A module inside a package imports its siblings by the package's absolute name —
`ntlm_auth/compute_keys.py` says `from ntlm_auth.constants import NegotiateFlags`. That
resolves against the package's *parent*, but resolve_module_path only searched the
module's own directory and sys.path, so on any package not installed on the path these
imports silently vanished from `parse_imports` — and with them every cross-module guard
that depended on a name they bound.
"""
from __future__ import annotations

import os
import unittest

from tharkun.solve.frontend.import_resolver import _package_roots, parse_imports, resolve_module_path

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "flagpkg")


class TestPackageRoots(unittest.TestCase):
    def test_package_dir_yields_its_parent(self) -> None:
        roots = _package_roots(FIX)
        self.assertIn(os.path.realpath(os.path.dirname(FIX)), roots)

    def test_non_package_dir_yields_nothing(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:      # no __init__.py anywhere above
            self.assertEqual(_package_roots(d), [])

    def test_empty_dir_is_safe(self) -> None:
        self.assertEqual(_package_roots(""), [])


class TestAbsoluteIntraPackageImport(unittest.TestCase):
    def test_resolves_sibling_by_absolute_package_name(self) -> None:
        got = resolve_module_path("flagpkg.constants", FIX)
        self.assertIsNotNone(got)
        self.assertEqual(os.path.basename(got), "constants.py")

    def test_parse_imports_binds_the_imported_name(self) -> None:
        imports = parse_imports(os.path.join(FIX, "reader.py"))
        self.assertIn("Flags", imports)
        self.assertEqual(os.path.basename(imports["Flags"]), "constants.py")

    def test_unknown_module_still_returns_none(self) -> None:
        self.assertIsNone(resolve_module_path("flagpkg.nosuchmodule", FIX))


if __name__ == "__main__":
    unittest.main()
