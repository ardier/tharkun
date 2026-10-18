"""Real-world validation: the hybrid loop on gunicorn's parse_address.

Plain Atheris stalls on the fd://, tcp:// string-prefix branches — the whole-buffer
unicode FDP field hides the byte->string mapping value-profile needs.  The assist
solves the prefixes and the variable-width codec recodes them into byte seeds that
reach the branches.  These tests pin that end to end (atheris-gated).
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

from tharkun.hybrid.assist import AssistEngine, HarnessSpec
from tharkun.hybrid.fdp_codec import atheris_available

requires_atheris = unittest.skipUnless(atheris_available(), "atheris not installed")

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures")
HARNESS = os.path.join(FIXDIR, "gunicorn_harness.py")
TARGET = os.path.join(FIXDIR, "gunicorn_util.py")


@requires_atheris
class TestGunicornAssist(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, FIXDIR)
        import gunicorn_util  # noqa: F401
        self.mod = gunicorn_util
        self.corpus = tempfile.mkdtemp()
        # seeds atheris can find on its own — none reach fd://, tcp://, unix:
        for i, s in enumerate([b"x", b"\x9f:", b";\xdb]"]):
            with open(os.path.join(self.corpus, f"s{i}"), "wb") as f:
                f.write(s)
        self.spec = HarnessSpec(
            harness_path=HARNESS, target_path=TARGET,
            target_func="parse_address", corpus_dir=self.corpus,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.corpus, ignore_errors=True)

    def _netloc_of(self, data: bytes) -> str:
        import atheris
        return atheris.FuzzedDataProvider(data).ConsumeUnicodeNoSurrogates(sys.maxsize)

    def test_assist_recodes_prefix_seeds(self) -> None:
        res = AssistEngine(self.spec).assist_once()
        self.assertGreater(len(res.seeds_written), 0)
        netlocs = [self._netloc_of(open(p, "rb").read()) for p in res.seeds_written]
        # the recoded seeds decode to the solved string prefixes atheris could not reach
        self.assertTrue(any(n.startswith("fd://") for n in netlocs),
                        f"expected an fd:// seed, got {netlocs}")

    def test_recoded_seed_reaches_new_branch(self) -> None:
        res = AssistEngine(self.spec).assist_once()
        # at least one recoded seed must drive parse_address into the fd:// path,
        # which raises RuntimeError on a non-int descriptor (line 6-10).
        reached_fd = False
        for p in res.seeds_written:
            netloc = self._netloc_of(open(p, "rb").read())
            if netloc.startswith("fd://"):
                reached_fd = True
                try:
                    self.mod.parse_address(netloc)
                except RuntimeError:
                    pass  # expected for fd:// with an empty/non-int descriptor
        self.assertTrue(reached_fd)


if __name__ == "__main__":
    unittest.main()
