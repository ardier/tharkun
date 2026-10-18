"""Tests for SG-5: mock generation — external call detection + @patch generation."""
from __future__ import annotations

import os
import unittest

from tharkun.mock_gen import (
    ExternalCall,
    MockPlan,
    PatchSpec,
    analyze_mocking_needs,
    detect_externals,
    generate_patches,
    render_mock_params,
    render_patch_decorators,
    serialize_mock_plan,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mock_targets.py")


class TestDetectHTTP(unittest.TestCase):
    """Scenario 1: HTTP library detection."""

    def test_requests_get_detected(self) -> None:
        exts = detect_externals(FIXTURE, "fetch_user")
        targets = {e.target for e in exts}
        self.assertIn("requests.get", targets)

    def test_requests_get_category_is_http(self) -> None:
        exts = detect_externals(FIXTURE, "fetch_user")
        http = [e for e in exts if e.target == "requests.get"]
        self.assertEqual(http[0].category, "http")

    def test_requests_post_detected(self) -> None:
        exts = detect_externals(FIXTURE, "post_data")
        targets = {e.target for e in exts}
        self.assertIn("requests.post", targets)


class TestDetectFilesystem(unittest.TestCase):
    """Scenario 2: Filesystem operation detection."""

    def test_open_detected(self) -> None:
        exts = detect_externals(FIXTURE, "read_config")
        targets = {e.target for e in exts}
        self.assertIn("builtins.open", targets)

    def test_os_path_exists_detected(self) -> None:
        exts = detect_externals(FIXTURE, "check_and_read")
        targets = {e.target for e in exts}
        self.assertIn("os.path.exists", targets)

    def test_pathlib_read_text_detected(self) -> None:
        exts = detect_externals(FIXTURE, "check_and_read")
        targets = {e.target for e in exts}
        self.assertIn("pathlib.Path.read_text", targets)


class TestDetectDatabase(unittest.TestCase):
    """Scenario 3: Database call detection."""

    def test_sqlite_connect_detected(self) -> None:
        exts = detect_externals(FIXTURE, "get_user_count")
        targets = {e.target for e in exts}
        self.assertIn("sqlite3.connect", targets)

    def test_database_category(self) -> None:
        exts = detect_externals(FIXTURE, "get_user_count")
        db = [e for e in exts if e.target == "sqlite3.connect"]
        self.assertEqual(db[0].category, "database")


class TestDetectChecksums(unittest.TestCase):
    """Scenario 4: Checksum/hash detection."""

    def test_crc32_detected(self) -> None:
        exts = detect_externals(FIXTURE, "verify_checksum")
        targets = {e.target for e in exts}
        self.assertIn("zlib.crc32", targets)

    def test_sha256_detected(self) -> None:
        exts = detect_externals(FIXTURE, "compute_hash")
        targets = {e.target for e in exts}
        self.assertIn("hashlib.sha256", targets)

    def test_hmac_detected(self) -> None:
        exts = detect_externals(FIXTURE, "sign_message")
        targets = {e.target for e in exts}
        self.assertIn("hmac.new", targets)

    def test_checksum_category(self) -> None:
        exts = detect_externals(FIXTURE, "verify_checksum")
        self.assertTrue(all(e.category == "checksum" for e in exts))


class TestDetectMixed(unittest.TestCase):
    """Scenario 5: Mixed HTTP + filesystem in one function."""

    def test_detects_multiple_categories(self) -> None:
        exts = detect_externals(FIXTURE, "download_and_cache")
        categories = {e.category for e in exts}
        self.assertIn("http", categories)
        self.assertIn("filesystem", categories)

    def test_all_externals_found(self) -> None:
        exts = detect_externals(FIXTURE, "download_and_cache")
        targets = {e.target for e in exts}
        self.assertIn("requests.get", targets)
        # Path.exists, Path.read_text, Path.write_text
        self.assertTrue(any("Path" in t for t in targets))


class TestDetectNested(unittest.TestCase):
    """Scenario 6: Nested helper calls.

    Note: mock_gen detects calls in the IMMEDIATE function body only.
    _fetch_raw has requests.get; get_enriched calls _fetch_raw but does not
    directly call requests.get — so get_enriched will NOT show requests.get.
    """

    def test_direct_call_detected(self) -> None:
        exts = detect_externals(FIXTURE, "_fetch_raw")
        targets = {e.target for e in exts}
        self.assertIn("requests.get", targets)

    def test_indirect_call_not_in_immediate(self) -> None:
        exts = detect_externals(FIXTURE, "get_enriched")
        targets = {e.target for e in exts}
        # get_enriched calls _fetch_raw, not requests.get directly
        self.assertNotIn("requests.get", targets)


class TestDetectAttributeChain(unittest.TestCase):
    """Scenario 7: self.session.get() on a class method."""

    def test_session_get_detected(self) -> None:
        exts = detect_externals(FIXTURE, "ApiClient.get_resource")
        targets = {e.target for e in exts}
        self.assertIn("requests.Session.get", targets)

    def test_session_in_constructor(self) -> None:
        exts = detect_externals(FIXTURE, "ApiClient.__init__")
        targets = {e.target for e in exts}
        self.assertIn("requests.Session", targets)


class TestNothingToMock(unittest.TestCase):
    """Scenario 8: Pure function — no externals detected."""

    def test_pure_function_empty(self) -> None:
        exts = detect_externals(FIXTURE, "pure_function")
        self.assertEqual(exts, [])


class TestGeneratePatches(unittest.TestCase):
    """Test patch spec generation from detected externals."""

    def test_http_patch(self) -> None:
        exts = [ExternalCall("requests.get", "http", 10)]
        patches = generate_patches(exts)
        self.assertEqual(len(patches), 1)
        self.assertEqual(patches[0].target, "requests.get")
        self.assertIsNotNone(patches[0].return_value)

    def test_open_uses_mock_open(self) -> None:
        exts = [ExternalCall("builtins.open", "filesystem", 10)]
        patches = generate_patches(exts)
        self.assertEqual(patches[0].target, "builtins.open")
        self.assertIn("mock_open", patches[0].setup_lines[0])

    def test_checksum_patch_has_return(self) -> None:
        exts = [ExternalCall("zlib.crc32", "checksum", 10)]
        patches = generate_patches(exts)
        self.assertEqual(patches[0].return_value, "0")


class TestRenderPatchDecorators(unittest.TestCase):
    def test_renders_decorator_lines(self) -> None:
        patches = [
            PatchSpec(target="requests.get", return_value="Mock()", category="http"),
            PatchSpec(target="os.path.exists", return_value="True", category="filesystem"),
        ]
        lines = render_patch_decorators(patches)
        self.assertEqual(len(lines), 2)
        self.assertTrue(any("requests.get" in l for l in lines))
        self.assertTrue(any("os.path.exists" in l for l in lines))

    def test_mock_params(self) -> None:
        patches = [
            PatchSpec(target="requests.get", category="http"),
        ]
        params = render_mock_params(patches)
        self.assertEqual(params, ["mock_requests_get"])


class TestAnalyzeMockingNeeds(unittest.TestCase):
    """Full analysis: detect all functions with externals."""

    def test_returns_plans_for_external_functions(self) -> None:
        plans = analyze_mocking_needs(FIXTURE)
        # Should have plans for functions with externals, not for pure_function
        qualnames = {p.callable_qualname for p in plans}
        self.assertIn("fetch_user", qualnames)
        self.assertIn("get_user_count", qualnames)
        self.assertNotIn("pure_function", qualnames)

    def test_specific_function(self) -> None:
        plans = analyze_mocking_needs(FIXTURE, ["verify_checksum"])
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0].callable_qualname, "verify_checksum")

    def test_nonexistent_file(self) -> None:
        plans = analyze_mocking_needs("/nonexistent.py")
        self.assertEqual(plans, [])


class TestSerializeMockPlan(unittest.TestCase):
    def test_serializes(self) -> None:
        plan = MockPlan(
            callable_qualname="fn",
            source_path="/fake.py",
            externals=[ExternalCall("requests.get", "http", 10)],
            patches=[PatchSpec(target="requests.get", return_value="Mock()", category="http")],
        )
        d = serialize_mock_plan(plan)
        self.assertIn("callable", d)
        self.assertIn("externals", d)
        self.assertIn("patches", d)


if __name__ == "__main__":
    unittest.main()
