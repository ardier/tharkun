"""Offline checks for the release helper's archive and draft boundaries."""

import contextlib
import copy
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("zenodo_draft", ROOT / "scripts/zenodo_draft.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ArchiveTests(unittest.TestCase):
    def check(self, packaged=b"answer = 42\n", expected=None):
        source = io.BytesIO()
        with tarfile.open(fileobj=source, mode="w") as tar:
            member = tarfile.TarInfo("src/tharkun/example.py")
            member.size = len(b"answer = 42\n")
            tar.addfile(member, io.BytesIO(b"answer = 42\n"))
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "tharkun.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("tharkun/VERSION", "0.2.2\n")
                bundle.writestr("tharkun/lib/tharkun/example.py", packaged)
                bundle.writestr("tharkun/lib/tharkun/__pycache__/example.pyc", b"cache")
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            with patch.object(MODULE.subprocess, "check_output", side_effect=["b" * 40, source.getvalue()]) as git:
                result = MODULE.check_archive(archive, expected or digest, "svcomp27-0.2.2")
                self.assertEqual(git.call_args_list[0].args[0],
                                 ["git", "rev-parse", "refs/tags/svcomp27-0.2.2^{commit}"])
                self.assertEqual(result, ("0.2.2", hashlib.md5(archive.read_bytes()).hexdigest(), "b" * 40))

    def test_exact_source_match_ignores_generated_bytecode(self):
        self.check()

    def test_changed_package_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "differ from the public release tag"):
            self.check(packaged=b"answer = 0\n")

    def test_checksum_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "validated SHA256"):
            self.check(expected="0" * 64)


class DraftTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.archive = Path(self.folder.name) / "tharkun.zip"
        self.archive.write_bytes(b"validated archive")
        self.md5 = hashlib.md5(self.archive.read_bytes()).hexdigest()
        self.sha256 = hashlib.sha256(self.archive.read_bytes()).hexdigest()
        self.deposit_url = f"{MODULE.API}/deposit/depositions/{MODULE.RECORD}"
        self.draft_url = f"{MODULE.API}/deposit/depositions/99999999"
        self.record = {
            "id": int(MODULE.RECORD), "conceptrecid": MODULE.CONCEPT,
            "metadata": {"version": "0.2.1", "doi": "10.5281/zenodo.23495266"},
            "files": [],
        }
        self.draft = {
            "id": 99999999, "conceptrecid": MODULE.CONCEPT,
            "submitted": False, "state": "unsubmitted",
            "metadata": {"version": "0.2.1", "title": "Tharkun",
                         "description": "This archive contains version 0.2.1"},
            "links": {"bucket": f"{MODULE.API}/files/test-bucket"},
            "files": [{"id": "old-file", "filename": "tharkun.zip",
                       "checksum": "0" * 32, "filesize": 1}],
        }
        self.existing_draft = False
        self.bad_upload = False
        self.calls = []

    def matching_file(self):
        return {"filename": "tharkun.zip", "checksum": self.md5,
                "filesize": self.archive.stat().st_size}

    def request(self, method, url, **kwargs):
        self.calls.append((method, url))
        if (method, url) == ("GET", f"{MODULE.API}/records/{MODULE.RECORD}/versions/latest"):
            return copy.deepcopy(self.record)
        if (method, url) == ("GET", self.deposit_url):
            return {"links": {"latest_draft": self.draft_url if self.existing_draft else self.deposit_url}}
        if (method, url) == ("POST", self.deposit_url + "/actions/newversion"):
            return {"links": {"latest_draft": self.draft_url}}
        if (method, url) == ("GET", self.draft_url):
            return copy.deepcopy(self.draft)
        if (method, url) == ("PUT", self.draft_url):
            self.draft["metadata"] = copy.deepcopy(kwargs["payload"]["metadata"])
            self.draft["metadata"]["prereserve_doi"] = {"doi": "10.5281/zenodo.99999999"}
            return copy.deepcopy(self.draft)
        if (method, url) == ("DELETE", self.draft_url + "/files/old-file"):
            self.draft["files"] = []
            return None
        if (method, url) == ("PUT", self.draft["links"]["bucket"] + "/tharkun.zip"):
            self.assertEqual(kwargs["upload"].read(), self.archive.read_bytes())
            self.draft["files"] = [self.matching_file()]
            return {"checksum": "md5:" + ("0" * 32 if self.bad_upload else self.md5),
                    "size": self.archive.stat().st_size}
        self.fail(f"Unexpected API operation: {method} {url}")

    def run_helper(self, version="0.2.2", archive_error=None, tag="svcomp27-0.2.2"):
        argv = ["zenodo_draft.py", "--archive", str(self.archive),
                "--tag", tag, "--sha256", self.sha256]
        with patch.object(sys, "argv", argv), \
             patch.dict(os.environ, {"ZENODO_TOKEN": "offline-test-token"}, clear=True), \
             patch.object(MODULE, "check_archive", return_value=(version, self.md5, "b" * 40),
                          side_effect=archive_error), \
             patch.object(MODULE, "request", side_effect=self.request), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            MODULE.main()
        return output.getvalue()

    def assert_read_only(self):
        self.assertTrue(all(method == "GET" for method, _ in self.calls))

    def test_new_draft_uploads_and_reserves_doi_without_publishing(self):
        output = self.run_helper()
        self.assertIn("https://zenodo.org/uploads/99999999", output)
        self.assertIn("10.5281/zenodo.99999999", output)
        self.assertEqual(self.draft["metadata"]["version"], "0.2.2")
        self.assertIn("version 0.2.2", self.draft["metadata"]["description"])
        self.assertEqual([url for method, url in self.calls if method == "POST"],
                         [self.deposit_url + "/actions/newversion"])

    def test_rerun_resumes_matching_draft_without_reupload(self):
        self.run_helper()
        self.calls.clear()
        self.existing_draft = True
        self.run_helper()
        self.assertFalse(any(method in ("POST", "DELETE") for method, _ in self.calls))
        self.assertNotIn(("PUT", self.draft["links"]["bucket"] + "/tharkun.zip"), self.calls)

    def test_unrelated_existing_draft_is_not_changed(self):
        self.existing_draft = True
        with self.assertRaisesRegex(ValueError, "Another Zenodo draft already exists"):
            self.run_helper()
        self.assert_read_only()

    def test_already_published_identical_archive_is_not_changed(self):
        self.record["files"] = [{"key": "tharkun.zip", "checksum": "md5:" + self.md5,
                                 "size": self.archive.stat().st_size}]
        self.assertIn("Already published", self.run_helper(version="0.2.1"))
        self.assert_read_only()

    def test_wrong_record_series_is_rejected(self):
        self.record["conceptrecid"] = "other"
        with self.assertRaisesRegex(ValueError, "existing Tharkun DOI series"):
            self.run_helper()
        self.assert_read_only()

    def test_already_published_version_with_other_bytes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "already published with different bytes"):
            self.run_helper(version="0.2.1")
        self.assert_read_only()

    def test_older_release_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "increase major.minor.patch"):
            self.run_helper(version="0.2.0")
        self.assert_read_only()

    def test_published_or_wrong_series_draft_is_not_modified(self):
        for changes, message in (({"submitted": True}, "published record"),
                                 ({"submitted": None}, "published record"),
                                 ({"state": "inprogress"}, "published record"),
                                 ({"id": MODULE.RECORD}, "published record"),
                                 ({"conceptrecid": "other"}, "existing Tharkun DOI series")):
            with self.subTest(changes=changes):
                original = copy.deepcopy(self.draft)
                self.draft.update(changes)
                with self.assertRaisesRegex(ValueError, message):
                    self.run_helper()
                self.assertFalse(any(method in ("PUT", "DELETE") for method, _ in self.calls))
                self.draft = original
                self.calls.clear()

    def test_unexpected_attachment_stops_before_changing_metadata(self):
        self.draft["files"].append({"filename": "manual-notes.txt"})
        with self.assertRaisesRegex(ValueError, "Unexpected draft attachment"):
            self.run_helper()
        self.assertFalse(any(method in ("PUT", "DELETE") for method, _ in self.calls))

    def test_archive_failure_happens_before_any_api_call(self):
        with self.assertRaisesRegex(ValueError, "bad archive"):
            self.run_helper(archive_error=ValueError("bad archive"))
        self.assertEqual(self.calls, [])

    def test_upload_verification_failure_is_reported(self):
        self.bad_upload = True
        with self.assertRaisesRegex(ValueError, "Uploaded file checksum/size mismatch"):
            self.run_helper()


class RequestBoundaryTests(unittest.TestCase):
    def test_foreign_or_insecure_urls_and_publication_actions_are_rejected(self):
        urls = ["http://zenodo.org/api/deposit/depositions/1",
                "https://example.org/api/deposit/depositions/1",
                "https://zenodo.org/api/deposit/depositions/1/actions/publish",
                "https://zenodo.org/api/deposit/depositions/1/actions/edit"]
        with patch.object(MODULE.urllib.request, "build_opener") as opener:
            for url in urls:
                with self.subTest(url=url), self.assertRaises(ValueError):
                    MODULE.request("POST", url, token="offline-test-token")
            opener.assert_not_called()

    def test_cross_origin_redirect_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "another origin"):
            MODULE.ZenodoRedirects().redirect_request(None, None, 302, "Found", {},
                                                      "https://example.org/api/files/key")


if __name__ == "__main__":
    unittest.main()
