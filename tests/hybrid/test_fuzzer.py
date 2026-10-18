"""Tests for the hybrid orchestrator and backends.

The orchestration is driven deterministically with a FakeBackend and an injected
clock/sleep — no real subprocess, no real waiting.  A separate atheris-gated test
checks AtherisBackend actually launches and reports.
"""
from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest

from tharkun.hybrid.assist import HarnessSpec
from tharkun.hybrid.backend import AtherisBackend, FakeBackend
from tharkun.hybrid.fdp_codec import atheris_available
from tharkun.hybrid.fuzzer import HybridFuzzer, should_assist

requires_atheris = unittest.skipUnless(atheris_available(), "atheris not installed")

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures")
HARNESS = os.path.join(FIXDIR, "deep_harness.py")
TARGET = os.path.join(FIXDIR, "deep_target.py")


class _Clock:
    """A manual clock: each call advances by `step`."""

    def __init__(self, step: float = 1.0) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


class TestShouldAssist(unittest.TestCase):
    def test_assists_on_plateau(self) -> None:
        self.assertTrue(should_assist(cov_now=10, best_cov=10, secs_since_gain=6, plateau_s=5))

    def test_no_assist_when_coverage_advances(self) -> None:
        self.assertFalse(should_assist(cov_now=12, best_cov=10, secs_since_gain=6, plateau_s=5))

    def test_no_assist_before_plateau_window(self) -> None:
        self.assertFalse(should_assist(cov_now=10, best_cov=10, secs_since_gain=2, plateau_s=5))


class TestAtherisBackendCommand(unittest.TestCase):
    def test_command_includes_corpus_and_reload(self) -> None:
        b = AtherisBackend("/tmp/h.py", "/tmp/corpus", python="python3")
        cmd = b.command()
        self.assertEqual(cmd[0], "python3")
        self.assertIn("-reload=1", cmd)
        self.assertTrue(cmd[1].endswith("h.py"))

    def test_reload_can_be_disabled(self) -> None:
        b = AtherisBackend("/tmp/h.py", "/tmp/corpus", reload=False)
        self.assertNotIn("-reload=1", b.command())


class TestOrchestrationDeterministic(unittest.TestCase):
    """Drive HybridFuzzer with a FakeBackend + fake clock — no atheris, no waiting."""

    def setUp(self) -> None:
        self.corpus = tempfile.mkdtemp()
        self.spec = HarnessSpec(
            harness_path=HARNESS, target_path=TARGET,
            target_func="parse", corpus_dir=self.corpus,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.corpus, ignore_errors=True)

    def test_assist_fires_on_plateau(self) -> None:
        # coverage never advances → plateau → assist should fire at least once.
        backend = FakeBackend([5, 5, 5, 5, 5, 5, 5, 5])
        # assist_reserve_s=0: this test is about plateau detection, not budget bounding
        fuzzer = HybridFuzzer(self.spec, backend=backend, plateau_s=2.0,
                              assist_cooldown_s=1.0, assist_reserve_s=0.0)
        # stub the assist so this test is independent of atheris/solving
        fired = {"n": 0}
        from tharkun.hybrid.assist import AssistResult
        fuzzer.assist.assist_once = lambda: (fired.__setitem__("n", fired["n"] + 1)
                                             or AssistResult(seeds_written=["s"]))
        stats = fuzzer.run(duration_s=6, poll_s=1.0, clock=_Clock(1.0), sleep=lambda s: None)
        self.assertGreaterEqual(stats.assists, 1)
        self.assertTrue(backend.started and backend.stopped)

    def test_no_assist_when_coverage_climbs(self) -> None:
        # coverage rises every poll → never plateaus → no assist.
        backend = FakeBackend([1, 2, 3, 4, 5, 6, 7, 8])
        fuzzer = HybridFuzzer(self.spec, backend=backend, plateau_s=2.0,
                              assist_cooldown_s=1.0, assist_reserve_s=0.0)
        fuzzer.assist.assist_once = lambda: (_ for _ in ()).throw(AssertionError("should not assist"))
        stats = fuzzer.run(duration_s=6, poll_s=1.0, clock=_Clock(1.0), sleep=lambda s: None)
        self.assertEqual(stats.assists, 0)


@requires_atheris
class TestAtherisBackendLaunch(unittest.TestCase):
    def test_launches_and_stops(self) -> None:
        import sys
        corpus = tempfile.mkdtemp()
        try:
            with open(os.path.join(corpus, "seed0"), "wb") as f:
                f.write(b"\x00\x00\x00\x00")
            # bound the run hard so the test is quick and self-terminating
            b = AtherisBackend(
                os.path.join(FIXDIR, "fuzz_harness.py"), corpus,
                python=sys.executable,
                extra_args=["-runs=200", "-max_total_time=2"],
            )
            b.start()
            self.assertTrue(b.started if hasattr(b, "started") else True)
            b.stop()
            self.assertFalse(b.is_running())
        finally:
            shutil.rmtree(corpus, ignore_errors=True)

    def test_cpu_seconds_advances_and_survives_stop(self) -> None:
        """[internal ref] part 2: cpu_seconds() must report real process CPU time while running,
        and keep reporting it (not silently reset to 0) once /proc has disappeared."""
        import sys
        corpus = tempfile.mkdtemp()
        try:
            with open(os.path.join(corpus, "seed0"), "wb") as f:
                f.write(b"\x00\x00\x00\x00")
            b = AtherisBackend(
                os.path.join(FIXDIR, "fuzz_harness.py"), corpus,
                python=sys.executable,
                extra_args=["-runs=100000", "-max_total_time=2"],
            )
            self.assertEqual(b.cpu_seconds(), 0.0)  # before start()
            b.start()
            deadline = time.monotonic() + 5
            seen_positive = False
            while time.monotonic() < deadline and b.is_running():
                if b.cpu_seconds() > 0.0:
                    seen_positive = True
                    break
            self.assertTrue(seen_positive, "cpu_seconds() never advanced above 0 while running")
            last_live = b.cpu_seconds()
            b.stop()
            # /proc/<pid>/stat is gone now (reaped) -- must return the last live reading,
            # not silently drop back to 0.
            self.assertGreaterEqual(b.cpu_seconds(), last_live)
        finally:
            shutil.rmtree(corpus, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


class TestBudgetIsRespected(unittest.TestCase):
    """An assist round must not be STARTED if it cannot finish inside the budget.

    The old loop only checked the clock at the top, so a run could overshoot duration_s
    by the length of an assist — which silently makes any timed comparison unfair (the
    hybrid arm simply got more wall clock than the baseline).
    """

    def setUp(self) -> None:
        self.corpus = tempfile.mkdtemp()
        self.spec = HarnessSpec(harness_path=HARNESS, target_path=TARGET,
                                target_func="parse_address", corpus_dir=self.corpus)

    def tearDown(self) -> None:
        shutil.rmtree(self.corpus, ignore_errors=True)

    def test_no_assist_starts_without_room_to_finish(self) -> None:
        backend = FakeBackend([5] * 12)          # permanently plateaued
        fuzzer = HybridFuzzer(self.spec, backend=backend, plateau_s=2.0,
                              assist_cooldown_s=1.0, assist_reserve_s=5.0)
        fuzzer.assist.assist_once = lambda: (_ for _ in ()).throw(
            AssertionError("started an assist with no room in the budget"))
        # 4s budget, 5s reserve: every candidate round is refused.
        stats = fuzzer.run(duration_s=4, poll_s=1.0, clock=_Clock(1.0), sleep=lambda s: None)
        self.assertEqual(stats.assists, 0)

    def test_elapsed_is_reported(self) -> None:
        backend = FakeBackend([1, 2, 3, 4, 5, 6])
        fuzzer = HybridFuzzer(self.spec, backend=backend, plateau_s=2.0)
        stats = fuzzer.run(duration_s=3, poll_s=1.0, clock=_Clock(1.0), sleep=lambda s: None)
        self.assertGreater(stats.elapsed_s, 0)
