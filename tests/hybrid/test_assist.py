"""Tests for the concolic assist engine.

find_frontiers is pure (source + coverage set) and runs everywhere.  The
solve→recode→converge tests drive the real atheris FDP codec and skip without it.
"""
from __future__ import annotations

import glob
import os
import shutil
import tempfile
import textwrap
import unittest

from tharkun.hybrid.assist import AssistEngine, HarnessSpec, find_frontiers
from tharkun.hybrid.fdp_codec import atheris_available

requires_atheris = unittest.skipUnless(atheris_available(), "atheris not installed")

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures")
HARNESS = os.path.join(FIXDIR, "deep_harness.py")
TARGET = os.path.join(FIXDIR, "deep_target.py")


class TestFindFrontiers(unittest.TestCase):
    SRC = textwrap.dedent("""
        def f(a, b):
            if a > 0:            # line 3
                if b > 0:        # line 4
                    return 1     # line 5
            return 0
    """)

    def test_uncovered_edge_on_border_is_a_frontier(self) -> None:
        # We reach the outer guard (line 3) but never enter its body (lines 4-5).
        fronts = find_frontiers(self.SRC, covered_lines={3, 6})
        self.assertTrue(any(f.branch_line == 3 and f.desired_branch == "then" for f in fronts))

    def test_fully_covered_has_no_frontier(self) -> None:
        self.assertEqual(find_frontiers(self.SRC, covered_lines={3, 4, 5, 6}), [])

    def test_unreached_guard_is_not_a_frontier(self) -> None:
        # If the guard line itself never executed, it is not on the border.
        self.assertEqual(find_frontiers(self.SRC, covered_lines={6}), [])

    def test_ranked_by_block_size(self) -> None:
        src = textwrap.dedent("""
            def f(a):
                if a > 0:
                    x = 1
                    y = 2
                    z = 3
                if a < 0:
                    q = 1
                return 0
        """)
        fronts = find_frontiers(src, covered_lines={3, 7, 9})
        # the 3-line block ranks above the 1-line block
        self.assertGreater(fronts[0].block_size, fronts[-1].block_size)


@requires_atheris
class TestAssistConverges(unittest.TestCase):
    def setUp(self) -> None:
        self.corpus = tempfile.mkdtemp()
        # a bland seed that hits none of the deep guards
        with open(os.path.join(self.corpus, "seed0"), "wb") as f:
            f.write(b"\x00\x00\x00\x00\x02\x00\x00\x00\x05")
        self.spec = HarnessSpec(
            harness_path=HARNESS, target_path=TARGET,
            target_func="parse", corpus_dir=self.corpus,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.corpus, ignore_errors=True)

    def _reaches_deep(self) -> bool:
        import atheris
        import sys
        sys.path.insert(0, FIXDIR)
        import deep_target
        for p in glob.glob(os.path.join(self.corpus, "*")):
            data = open(p, "rb").read()
            fdp = atheris.FuzzedDataProvider(data)
            m, k, s = fdp.ConsumeInt(4), fdp.ConsumeIntInRange(0, 20), fdp.ConsumeInt(4)
            if deep_target.parse(m, k, s) == "DEEP":
                return True
        return False

    def test_first_round_solves_the_magic_number(self) -> None:
        eng = AssistEngine(self.spec)
        res = eng.assist_once()
        self.assertEqual(res.frontiers_found, 1)
        self.assertEqual(len(res.seeds_written), 1)
        self.assertEqual(res.solved[0]["magic"], 1094795585)  # 0x41414141

    def test_iterative_assist_reaches_deep_target(self) -> None:
        eng = AssistEngine(self.spec)
        for _ in range(6):
            eng.assist_once()
            if self._reaches_deep():
                break
        self.assertTrue(self._reaches_deep(), "assist did not converge to DEEP")

    def test_written_seed_decodes_through_real_harness(self) -> None:
        import atheris
        import sys
        sys.path.insert(0, FIXDIR)
        import deep_target
        eng = AssistEngine(self.spec)
        res = eng.assist_once()
        data = open(res.seeds_written[0], "rb").read()
        fdp = atheris.FuzzedDataProvider(data)
        m = fdp.ConsumeInt(4)
        self.assertEqual(m, 1094795585)  # the recoded magic really decodes back


if __name__ == "__main__":
    unittest.main()
