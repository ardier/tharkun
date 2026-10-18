"""Tests for the guard analyzer module (runtime snapshot + decomposition)."""
from __future__ import annotations

import os
import unittest

from tharkun.guard_analyzer import (
    AnalysisResult,
    BlockerDiagnostic,
    analyze_blocker,
    serialize_analysis,
    _capture_snapshots_inprocess,
)
from tharkun.trace_api import TracePolicy, TraceResult, Tracer

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "analyze_target.py")
DECOMP_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "decompose_target.py")


class TestTraceSnapshot(unittest.TestCase):
    """Test the core snapshot mechanism in trace_api."""

    def test_snapshot_captures_locals(self) -> None:
        """Verify that f_locals are captured at snapshot lines."""
        rp = os.path.realpath(FIXTURE)
        # Line 15 is `if x > 100:` in simple_guard
        policy = TracePolicy(
            enabled=True,
            snapshot_lines={rp: {15}},
            paths={rp},
        )
        import importlib.util
        spec = importlib.util.spec_from_file_location("_snap_test", rp)
        mod = importlib.util.module_from_spec(spec)
        with Tracer(policy) as tr:
            spec.loader.exec_module(mod)
            mod.simple_guard(42)

        snaps = tr.result.snapshots.get(rp, {})
        self.assertIn(15, snaps)
        self.assertGreater(len(snaps[15]), 0)
        # x should be in the captured locals
        first_snap = snaps[15][0]
        self.assertIn("x", first_snap)
        self.assertEqual(first_snap["x"], 42)

    def test_snapshot_multiple_calls(self) -> None:
        """Multiple calls should produce multiple snapshots at the same line."""
        rp = os.path.realpath(FIXTURE)
        policy = TracePolicy(
            enabled=True,
            snapshot_lines={rp: {15}},
            paths={rp},
        )
        import importlib.util
        spec = importlib.util.spec_from_file_location("_snap_test2", rp)
        mod = importlib.util.module_from_spec(spec)
        with Tracer(policy) as tr:
            spec.loader.exec_module(mod)
            mod.simple_guard(10)
            mod.simple_guard(200)

        snaps = tr.result.snapshots.get(rp, {}).get(15, [])
        self.assertGreaterEqual(len(snaps), 2)

    def test_no_snapshot_without_policy(self) -> None:
        """Without snapshot_lines, no snapshots are captured."""
        rp = os.path.realpath(FIXTURE)
        policy = TracePolicy(enabled=True, paths={rp})
        import importlib.util
        spec = importlib.util.spec_from_file_location("_snap_test3", rp)
        mod = importlib.util.module_from_spec(spec)
        with Tracer(policy) as tr:
            spec.loader.exec_module(mod)
            mod.simple_guard(42)

        self.assertEqual(tr.result.snapshots, {})


class TestCaptureSnapshotsInprocess(unittest.TestCase):
    def test_captures_at_guard_line(self) -> None:
        snaps = _capture_snapshots_inprocess(
            FIXTURE, {15},
            func_name="simple_guard",
            kwargs={"x": 42},
        )
        self.assertIn(15, snaps)
        self.assertIn("x", snaps[15][0])

    def test_nonexistent_file(self) -> None:
        snaps = _capture_snapshots_inprocess("/nonexistent.py", {1})
        self.assertEqual(snaps, {})


class TestAnalyzeBlocker(unittest.TestCase):
    def test_analyze_with_execution(self) -> None:
        # Use decompose_target.py line 58: if partial_check(x, data): — has an opaque blocker
        result = analyze_blocker(DECOMP_FIXTURE, 58, execute=True)
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.executed, True)
        self.assertIn("partial_check", result.guard_source)
        self.assertGreater(len(result.diagnostics), 0)

    def test_analyze_without_execution(self) -> None:
        result = analyze_blocker(DECOMP_FIXTURE, 58, execute=False)
        self.assertIsInstance(result, AnalysisResult)
        self.assertEqual(result.executed, False)
        self.assertIsNotNone(result.decomposition)
        self.assertGreater(len(result.diagnostics), 0)

    def test_no_blockers_returns_empty_diagnostics(self) -> None:
        # Line 15: simple guard x > 100 — fully solvable
        result = analyze_blocker(FIXTURE, 15, execute=False)
        self.assertEqual(result.diagnostics, [])

    def test_opaque_call_diagnostic(self) -> None:
        # Line 48 in decompose_target.py: if external_lib.validate(x):
        result = analyze_blocker(DECOMP_FIXTURE, 48, execute=False)
        self.assertGreater(len(result.diagnostics), 0)
        diag = result.diagnostics[0]
        self.assertIn("unresolved", diag.blocker_reason or "")
        self.assertIsNotNone(diag.suggestion)


class TestSerializeAnalysis(unittest.TestCase):
    def test_serializes(self) -> None:
        result = analyze_blocker(DECOMP_FIXTURE, 48, execute=False)
        d = serialize_analysis(result)
        self.assertIn("guard_source", d)
        self.assertIn("diagnostics", d)
        self.assertIn("decomposition", d)
        self.assertIsInstance(d["diagnostics"], list)


class TestSafeSnapshot(unittest.TestCase):
    """Test _safe_snapshot handles edge cases."""

    def test_primitives(self) -> None:
        from tharkun.trace_api import _safe_snapshot
        self.assertEqual(_safe_snapshot(42), 42)
        self.assertEqual(_safe_snapshot(3.14), 3.14)
        self.assertEqual(_safe_snapshot(True), True)
        self.assertIsNone(_safe_snapshot(None))

    def test_long_string_truncated(self) -> None:
        from tharkun.trace_api import _safe_snapshot
        long_s = "x" * 1000
        result = _safe_snapshot(long_s)
        self.assertEqual(len(result), 256)

    def test_bytes_captured_safely(self) -> None:
        from tharkun.trace_api import _safe_snapshot
        result = _safe_snapshot(b"\x00" * 500)
        self.assertIsInstance(result, dict)
        self.assertEqual(result["__type__"], "bytes")
        self.assertEqual(result["len"], 500)

    def test_dict_captured_safely(self) -> None:
        from tharkun.trace_api import _safe_snapshot
        big_dict = {f"key_{i}": i for i in range(100)}
        result = _safe_snapshot(big_dict)
        self.assertEqual(result["__type__"], "dict")
        self.assertEqual(result["len"], 100)

    def test_object_captured_safely(self) -> None:
        from tharkun.trace_api import _safe_snapshot
        class Foo:
            pass
        result = _safe_snapshot(Foo())
        self.assertIsInstance(result, dict)
        self.assertEqual(result["__type__"], "Foo")


if __name__ == "__main__":
    unittest.main()
