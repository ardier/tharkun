"""Real-gunicorn coverage: driving the parsers covers far more than OSS-Fuzz's util-only job.

OSS-Fuzz ships one harness (fuzz_util.py) so its report only ever sees gunicorn/util.py; the
HTTP and ASGI request parsers sit at 0% because nothing imports them. gunicorn_cov.py drives
those parsers and measures coverage.py statement coverage per module (the OSS-Fuzz units).

These tests are gunicorn-gated (skipped when gunicorn is not importable). The coverage
comparison runs gunicorn_cov.py in subprocesses (a fresh process per suite so module-level
statements are counted once, not lost to import caching). A second, atheris-gated test shows
the hybrid loop solving the real parse_address prefix guards that random mutation misses.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
COV_TOOL = os.path.join(HERE, "gunicorn_cov.py")
FIXDIR = os.path.join(HERE, "fixtures")


def _has_gunicorn() -> bool:
    try:
        import gunicorn  # noqa: F401
        return True
    except Exception:
        return False


def _run_suite(suite: str):
    r = subprocess.run([sys.executable, COV_TOOL, suite],
                       capture_output=True, text=True, timeout=300)
    if r.returncode == 2:            # gunicorn unavailable inside the subprocess
        return None
    if r.returncode != 0 or not r.stdout.strip():
        raise AssertionError(f"gunicorn_cov {suite} failed: {r.returncode}\n{r.stderr[-800:]}")
    return json.loads(r.stdout)


@unittest.skipUnless(_has_gunicorn(), "gunicorn not installed")
class TestGunicornRealCoverage(unittest.TestCase):
    def test_multi_harness_far_exceeds_util_only(self) -> None:
        util = _run_suite("util")
        allc = _run_suite("all")
        if util is None or allc is None:
            self.skipTest("gunicorn unavailable in subprocess")
        uc = util["_total"][0]
        ac = allc["_total"][0]
        # Driving the HTTP + ASGI parsers covers hundreds of statements the util-only job
        # never reaches — the whole point of adding harnesses for the unfuzzed surface.
        self.assertGreater(ac, uc + 400, f"util-only covered={uc}, all covered={ac}")

    def test_parser_modules_light_up(self) -> None:
        allc = _run_suite("all")
        if allc is None:
            self.skipTest("gunicorn unavailable in subprocess")
        # Modules OSS-Fuzz's harness never imports reach real coverage under a driving harness.
        for mod, floor in (("http/message.py", 0.4), ("http/body.py", 0.4),
                           ("asgi/parser.py", 0.4)):
            if mod not in allc:
                continue
            c, l = allc[mod]
            self.assertGreater(c / max(l, 1), floor, f"{mod}: {c}/{l}")


def _has_atheris() -> bool:
    try:
        from tharkun.hybrid.fdp_codec import atheris_available
        return atheris_available()
    except Exception:
        return False


@unittest.skipUnless(_has_gunicorn() and _has_atheris(), "needs gunicorn + atheris")
class TestHybridLiftsRealUtil(unittest.TestCase):
    def test_hybrid_solves_real_parse_address_prefixes(self) -> None:
        """The hybrid assist recodes seeds reaching gunicorn.util.parse_address's fd:// /
        tcp:// / unix: branches — the sparse prefix guards random ConsumeString never lands."""
        import glob
        import shutil
        import tempfile

        import atheris

        from gunicorn import util
        from tharkun.hybrid.assist import AssistEngine, HarnessSpec

        corpus = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, corpus, ignore_errors=True)
        for i, s in enumerate([b"x", b"host:80", b"[::1]:9", b"a.b:1"]):
            with open(os.path.join(corpus, f"s{i}"), "wb") as f:
                f.write(s)

        spec = HarnessSpec(
            harness_path=os.path.join(FIXDIR, "fuzz_util_real.py"),
            target_path=util.__file__, target_func="parse_address", corpus_dir=corpus,
        )
        eng = AssistEngine(spec, max_frontiers=8)
        for _ in range(6):
            eng.assist_once()

        reached = set()
        for p in glob.glob(corpus + "/*"):
            netloc = atheris.FuzzedDataProvider(open(p, "rb").read()).ConsumeString(100)
            for pre in ("fd://", "tcp://", "unix:"):
                if netloc.startswith(pre):
                    reached.add(pre)
        self.assertTrue(reached, f"assist solved no prefix; ledger={eng.ledger.summary()}")
        # At least the fd:// branch — the clearest sparse-and-obscured guard — must fall.
        self.assertIn("fd://", reached, f"reached={reached}, ledger={eng.ledger.summary()}")

    def test_hybrid_covers_parse_address_prefixes_mid_buffer(self) -> None:
        """The FULL fuzz_util.py harness puts parse_address at field 3, behind three other
        variable-length ConsumeString fields. A solved "fd://" recodes into that mid-buffer
        field as "fd://<leftover>" — not an exact match, but it still satisfies startswith, so
        the execution-verified recode accepts it. This pins that end to end."""
        import glob
        import shutil
        import tempfile

        import atheris

        from gunicorn import util
        from tharkun.hybrid.assist import AssistEngine, HarnessSpec

        corpus = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, corpus, ignore_errors=True)
        # Wide seeds so field 3 (parse_address's netloc) is addressable, not starved.
        for i in range(3):
            with open(os.path.join(corpus, f"s{i}"), "wb") as f:
                f.write(bytes((i * 7 + j) % 256 for j in range(500)))

        spec = HarnessSpec(
            harness_path=os.path.join(FIXDIR, "fuzz_util_full.py"),
            target_path=util.__file__, target_func="parse_address",
            corpus_dir=corpus, package_dir=os.path.dirname(util.__file__),
        )
        eng = AssistEngine(spec, max_frontiers=12)
        reached: set = set()
        for _ in range(4):
            eng.assist_once()
            reached.clear()
            for p in glob.glob(corpus + "/*"):
                fdp = atheris.FuzzedDataProvider(open(p, "rb").read())
                for _ in range(3):           # skip is_ipv6 / warn / split_request_uri fields
                    fdp.ConsumeString(100)
                netloc = fdp.ConsumeString(100)   # field 3 -> parse_address.netloc
                for pre in ("fd://", "tcp://", "unix:"):
                    if netloc.startswith(pre):
                        reached.add(pre)
            if "fd://" in reached:
                break
        # The mid-buffer field 3 recode must land at least the fd:// branch that OSS-Fuzz's
        # own report shows uncovered — the whole point of the execution-verified recode.
        self.assertIn("fd://", reached, f"reached={reached}, ledger={eng.ledger.summary()}")


if __name__ == "__main__":
    unittest.main()
