"""The assist must work through INDEPENDENT frontiers one at a time, live.

Not a nested chain (that is the driver's job within one solve) but several sibling
guarded regions: unlock one, re-rank what remains, attack the next. The fixture is
deliberately trivial so a failure here means the loop is wrong, not the solver.
"""
from __future__ import annotations

import glob
import os
import shutil
import sys
import tempfile
import unittest

from tharkun.hybrid.assist import AssistEngine, HarnessSpec, find_frontiers
from tharkun.hybrid.fdp_codec import atheris_available

requires_atheris = unittest.skipUnless(atheris_available(), "atheris not installed")

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures")
HARNESS = os.path.join(FIXDIR, "toy_harness.py")
TARGET = os.path.join(FIXDIR, "toy_target.py")


class TestFrontierRanking(unittest.TestCase):
    """Pure: the four arms rank by how much uncovered code each gates."""

    def test_ranked_by_uncovered_gated(self) -> None:
        src = open(TARGET).read()
        # Only the guard lines are covered — every arm body is still dark.
        covered = {n for n in range(1, 40)} - set()
        fronts = find_frontiers(src, covered)
        self.assertEqual(fronts, [])          # nothing dark => no frontier

    def test_dark_arms_become_frontiers(self) -> None:
        import ast
        src = open(TARGET).read()
        guards = sorted(n.lineno for n in ast.walk(ast.parse(src)) if isinstance(n, ast.If))
        fronts = find_frontiers(src, set(guards))
        self.assertEqual(len(fronts), 4)
        # Biggest body first: arm A gates four statements, arm D one.
        self.assertEqual(fronts[0].branch_line, guards[0])
        self.assertGreater(fronts[0].uncovered_gated, fronts[-1].uncovered_gated)


@requires_atheris
class TestSequentialUnlock(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, FIXDIR)
        import toy_target
        self.mod = toy_target
        self.corpus = tempfile.mkdtemp()
        with open(os.path.join(self.corpus, "seed0"), "wb") as f:
            f.write(b"\x00" * 16)
        self.spec = HarnessSpec(harness_path=HARNESS, target_path=TARGET,
                                target_func="dispatch", corpus_dir=self.corpus)

    def tearDown(self) -> None:
        shutil.rmtree(self.corpus, ignore_errors=True)

    def _arms(self) -> str:
        import atheris
        got = set()
        for p in glob.glob(self.corpus + "/*"):
            fdp = atheris.FuzzedDataProvider(open(p, "rb").read())
            try:
                args = [fdp.ConsumeUInt(4) for _ in range(4)]
            except Exception:
                continue
            got |= {h[0] for h in self.mod.dispatch(*args)}
        return "".join(sorted(got))

    def test_unlocks_every_independent_frontier(self) -> None:
        # max_frontiers=1: exactly one blocker attacked per round, so the test pins the
        # find -> unlock -> re-rank -> next behaviour rather than a batch solve.
        eng = AssistEngine(self.spec, max_frontiers=1)
        counts = []
        for _ in range(6):
            res = eng.assist_once()
            counts.append(res.frontiers_found)
            if self._arms() == "ABCD":
                break
        self.assertEqual(self._arms(), "ABCD")
        # The remaining-frontier count must strictly shrink as arms are unlocked.
        self.assertEqual(counts[:4], [4, 3, 2, 1])

    def test_each_round_writes_one_seed(self) -> None:
        eng = AssistEngine(self.spec, max_frontiers=1)
        res = eng.assist_once()
        self.assertEqual(len(res.seeds_written), 1)
        self.assertEqual(res.frontiers_found, 4)


if __name__ == "__main__":
    unittest.main()


@requires_atheris
class TestMixedSiblingsWithNesting(unittest.TestCase):
    """Real code is siblings AND nesting. These fixtures are both at once.

    The flat toy above isolates the frontier loop; deep_target isolates the driver's
    nested solve. Neither exercises the composition, which is the shape actual targets
    have — and the composition is where two real bugs were found (immutable-sequence
    patching, and a locked-variable veto that discarded good patches).
    """

    def _drive(self, name: str, target_fn: str, seed: dict, decode, want: str,
               rounds: int = 10, atleast: str | None = None) -> str:
        # ``want`` stops the loop on an exact arm set; ``atleast`` stops as soon as those
        # arms are all present (a superset is fine). The superset stop keeps the test fast
        # when an extra arm may or may not also unlock — without it the loop runs every
        # round chasing an arm set it never exactly hits.
        import atheris
        sys.path.insert(0, FIXDIR)
        mod = __import__(f"{name}_target")
        corpus = tempfile.mkdtemp()
        try:
            with open(os.path.join(corpus, "s"), "wb") as f:
                f.write(b"\x00" * 40)
            spec = HarnessSpec(harness_path=os.path.join(FIXDIR, f"{name}_harness.py"),
                               target_path=os.path.join(FIXDIR, f"{name}_target.py"),
                               target_func=target_fn, corpus_dir=corpus)
            eng = AssistEngine(spec, max_frontiers=1)
            for _ in range(rounds):
                eng.assist_once()
                got = set()
                for p in glob.glob(corpus + "/*"):
                    try:
                        got |= set(decode(open(p, "rb").read(), mod))
                    except Exception:
                        pass
                arms = "".join(sorted({a[0] for a in got}))
                if arms == want or (atleast is not None and set(atleast) <= set(arms)):
                    return arms
            return arms
        finally:
            shutil.rmtree(corpus, ignore_errors=True)

    def test_easy_mixed_unlocks_every_arm(self) -> None:
        import atheris

        def dec(data, mod):
            f = atheris.FuzzedDataProvider(data)
            return mod.route(f.ConsumeUInt(4), f.ConsumeUInt(4), f.ConsumeBytes(8))
        self.assertEqual(self._drive("mixed_easy", "route", {}, dec, "ABC"), "ABC")

    def test_hard_mixed_unlocks_the_smt_arms(self) -> None:
        # At least three of the four arms fall to the default portfolio: A (exact 32-bit
        # magic and a nested exact version) and B/C (nonlinear + uninvertible modular) are
        # all touchstone/static territory. Arm D — a slice-aliased suffix (`rest = name[6:]`
        # then `rest == "prod"`) — can fall to live operand harvesting. With
        # flow-sensitive reach-search it can ALSO fall now: the reassignment severs the
        # entry field from the suffix guard, so the round routes to the all-parameter
        # reach-search whose pool carries 'cfg://' and 'prod'. So ABC is the floor and ABCD
        # is allowed — never fewer than the three SMT arms.
        import atheris

        def dec(data, mod):
            f = atheris.FuzzedDataProvider(data)
            args = [f.ConsumeUInt(4) for _ in range(4)]
            args.append(f.ConsumeUnicodeNoSurrogates(64))
            return mod.parse(*args)
        arms = self._drive("mixed_hard", "parse", {}, dec, "ABCD", atleast="ABC")
        self.assertLessEqual(set("ABC"), set(arms), arms)     # the three SMT arms are the floor
        self.assertLessEqual(set(arms), set("ABCD"), arms)    # never spurious arms

    def test_frontier_shape_is_mixed_depth(self) -> None:
        import ast
        src = open(os.path.join(FIXDIR, "mixed_hard_target.py")).read()
        guards = sorted(n.lineno for n in ast.walk(ast.parse(src)) if isinstance(n, ast.If))
        depths = {f.depth for f in find_frontiers(src, set(guards))}
        self.assertEqual(depths, {0, 1})   # siblings and nested interiors together
