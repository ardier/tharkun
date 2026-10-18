"""Measured: the hybrid loop unlocks gunicorn's regex request blockers; plain Atheris can't.

gunicorn/http/message.py gates the request body on three regexes — the method token, the
HTTP version (``HTTP/(\\d)\\.(\\d)``), and the header-name token. Behind a fixed-width fuzzer
field these are sparse *and* obscured (a regex fullmatch offers value-profile nothing to
copy), which per the design is exactly where the concolic assist beats random mutation.

Two pins:
  * ``test_keeps_unlocking_every_regex_blocker`` — no fuzzer timing: drive the assist loop
    directly and watch the corpus depth climb 0 -> 3 as each regex blocker is detected,
    solved, and recoded, with the ledger showing three SOLVED. This is the "detect the
    blockers, keep unlocking them" behaviour, on real regexes.
  * the plain-vs-hybrid A/B (same harness, same Atheris budget both arms; the hybrid arm reaches strictly deeper) is a
    RESULT of a randomized, wall-clock-boxed fuzzer, so it is a measured metric (``scripts/hybrid_bench_metric.py``, the
    nightly), not a gate assertion (docs/TESTING.md, "Randomized and wall-clock-boxed checks"); the gate keeps what holds
    whatever the fuzzer finds (``test_benchmark_runs_both_arms``).
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile
import unittest

from tharkun.hybrid.assist import AssistEngine, HarnessSpec
from tharkun.hybrid.fdp_codec import atheris_available

requires_atheris = unittest.skipUnless(atheris_available(), "atheris not installed")

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures")
sys.path.insert(0, FIXDIR)
RHARNESS = os.path.join(FIXDIR, "gunicorn_request_harness.py")
RTARGET = os.path.join(FIXDIR, "gunicorn_request.py")

# Seeds that pass NO guard — every field is non-token bytes, so depth starts at 0 and each
# regex blocker is a genuine blocker (not one a lucky seed already satisfies).
BLOCKING_SEEDS = [b"\x00" * 32, b"\xff" * 32, bytes((i * 13) % 256 for i in range(32))]


def _depth(data: bytes) -> int:
    """How far into the nested regex chain this seed's fields reach (0..3)."""
    import atheris

    import gunicorn_request as gr

    fdp = atheris.FuzzedDataProvider(data)
    method = fdp.ConsumeBytes(8)
    name = fdp.ConsumeBytes(8)
    version = fdp.ConsumeBytes(8)
    d = 0
    if gr.TOKEN8_RE.fullmatch(method):          # blocker 1
        d = 1
        if gr.VERSION_RE.fullmatch(version):    # blocker 2
            d = 2
            if gr.TOKEN8_RE.fullmatch(name):    # blocker 3
                d = 3
    return d


@requires_atheris
class TestGunicornRequestUnlock(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = tempfile.mkdtemp()
        for i, s in enumerate(BLOCKING_SEEDS):
            with open(os.path.join(self.corpus, f"s{i}"), "wb") as f:
                f.write(s)
        self.spec = HarnessSpec(harness_path=RHARNESS, target_path=RTARGET,
                                target_func="parse_request", corpus_dir=self.corpus)

    def tearDown(self) -> None:
        shutil.rmtree(self.corpus, ignore_errors=True)

    def _max_depth(self) -> int:
        best = 0
        for p in glob.glob(self.corpus + "/*"):
            best = max(best, _depth(open(p, "rb").read()))
        return best

    def test_keeps_unlocking_every_regex_blocker(self) -> None:
        eng = AssistEngine(self.spec, max_frontiers=4)
        self.assertEqual(self._max_depth(), 0, "blocking seeds should pass no guard")
        for _ in range(8):
            eng.assist_once()
            if self._max_depth() >= 3:
                break
        self.assertEqual(self._max_depth(), 3,
                         "assist did not unlock all three regex blockers")
        summ = eng.ledger.summary()
        # Three distinct regex frontiers, each moved to SOLVED in the worklist.
        self.assertGreaterEqual(summ.solved, 3, f"ledger: {summ}")


if __name__ == "__main__":
    unittest.main()
