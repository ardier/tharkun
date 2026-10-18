"""Tests for trace_api — sys.settrace-based variable capture."""
from __future__ import annotations

import os
import tempfile
import textwrap
import unittest

from tharkun.trace_api import TracePolicy, Tracer, _safe_snapshot


class TestSafeSnapshot(unittest.TestCase):
    def test_none(self) -> None:
        self.assertIsNone(_safe_snapshot(None))

    def test_primitives(self) -> None:
        self.assertEqual(_safe_snapshot(42), 42)
        self.assertEqual(_safe_snapshot(3.14), 3.14)
        self.assertEqual(_safe_snapshot(True), True)

    def test_str_short(self) -> None:
        self.assertEqual(_safe_snapshot("hello"), "hello")

    def test_str_long_truncated(self) -> None:
        long_str = "x" * 300
        result = _safe_snapshot(long_str)
        self.assertEqual(len(result), 256)

    def test_bytes(self) -> None:
        result = _safe_snapshot(b"\x00\x01")
        self.assertEqual(result["__type__"], "bytes")
        self.assertIn("len", result)

    def test_list(self) -> None:
        result = _safe_snapshot([1, 2, 3])
        self.assertEqual(result["__type__"], "list")
        self.assertEqual(result["len"], 3)

    def test_dict(self) -> None:
        result = _safe_snapshot({"a": 1, "b": 2})
        self.assertEqual(result["__type__"], "dict")
        self.assertEqual(result["len"], 2)

    def test_set(self) -> None:
        result = _safe_snapshot({1, 2})
        self.assertEqual(result["__type__"], "set")

    def test_type_object(self) -> None:
        result = _safe_snapshot(int)
        self.assertEqual(result["__type__"], "type")
        self.assertEqual(result["name"], "int")

    def test_arbitrary_object(self) -> None:
        class Foo:
            pass
        result = _safe_snapshot(Foo())
        self.assertEqual(result["__type__"], "Foo")


class TestTracerLineCapture(unittest.TestCase):
    def _write_temp(self, code: str) -> str:
        f = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
        f.write(code)
        f.close()
        return os.path.realpath(f.name)

    def test_basic_line_tracing(self) -> None:
        path = self._write_temp("x = 1\ny = 2\nz = x + y\n")
        rp = os.path.realpath(path)
        policy = TracePolicy(paths={rp})
        tracer = Tracer(policy)
        with tracer:
            exec(compile(open(path).read(), path, "exec"))
        self.assertIn(rp, tracer.result.executed)
        self.assertIn(1, tracer.result.executed[rp])
        os.unlink(path)

    def test_snapshot_capture(self) -> None:
        code = "x = 42\ny = x + 1\n"
        path = self._write_temp(code)
        rp = os.path.realpath(path)
        policy = TracePolicy(
            paths={rp},
            snapshot_lines={rp: {2}},
        )
        tracer = Tracer(policy)
        with tracer:
            exec(compile(open(path).read(), path, "exec"))
        # Should have captured locals at line 2
        self.assertIn(rp, tracer.result.snapshots)
        self.assertIn(2, tracer.result.snapshots[rp])
        snaps = tracer.result.snapshots[rp][2]
        self.assertGreater(len(snaps), 0)
        # x should be 42 at line 2
        self.assertEqual(snaps[0].get("x"), 42)
        os.unlink(path)

    def test_capture_real_objects(self) -> None:
        code = "data = [1, 2, 3]\nresult = sum(data)\n"
        path = self._write_temp(code)
        rp = os.path.realpath(path)
        policy = TracePolicy(
            paths={rp},
            snapshot_lines={rp: {2}},
            capture_real=True,
        )
        tracer = Tracer(policy)
        with tracer:
            exec(compile(open(path).read(), path, "exec"))
        # Safe snapshot: data is wrapped
        safe_snaps = tracer.result.snapshots[rp][2]
        self.assertGreater(len(safe_snaps), 0)
        # data in safe snapshot is a dict with __type__
        safe_data = safe_snaps[0].get("data")
        self.assertIsInstance(safe_data, dict)
        self.assertEqual(safe_data.get("__type__"), "list")
        # Real snapshot: data is the actual list
        real_snaps = tracer.result.snapshots_real[rp][2]
        self.assertGreater(len(real_snaps), 0)
        real_data = real_snaps[0].get("data")
        self.assertIsInstance(real_data, list)
        self.assertEqual(real_data, [1, 2, 3])
        os.unlink(path)

    def test_capture_real_not_set_by_default(self) -> None:
        code = "x = 42\ny = x + 1\n"
        path = self._write_temp(code)
        rp = os.path.realpath(path)
        policy = TracePolicy(
            paths={rp},
            snapshot_lines={rp: {2}},
            # capture_real defaults to False
        )
        tracer = Tracer(policy)
        with tracer:
            exec(compile(open(path).read(), path, "exec"))
        # snapshots_real should be empty
        self.assertEqual(tracer.result.snapshots_real, {})
        os.unlink(path)

    def test_disabled_policy(self) -> None:
        policy = TracePolicy(enabled=False)
        tracer = Tracer(policy)
        with tracer:
            x = 1 + 1  # noqa: F841
        # No lines should be recorded
        self.assertEqual(tracer.result.executed, {})

    def test_duration_recorded(self) -> None:
        policy = TracePolicy()
        tracer = Tracer(policy)
        with tracer:
            _ = sum(range(100))
        self.assertGreater(tracer.result.duration_s, 0.0)

    def test_file_filter(self) -> None:
        code = "a = 1\n"
        path = self._write_temp(code)
        rp = os.path.realpath(path)
        # Filter to a different file — should not capture
        policy = TracePolicy(paths={"/nonexistent/file.py"})
        tracer = Tracer(policy)
        with tracer:
            exec(compile(open(path).read(), path, "exec"))
        self.assertNotIn(rp, tracer.result.executed)
        os.unlink(path)


if __name__ == "__main__":
    unittest.main()


# -- arcs (branch coverage) --------------------------------------------------


def _module(tmp_path, body: str):
    import os

    # Strip the leading newline the triple-quoted literal starts with, so the
    # first "def" really is line 1 and the asserted arcs read like the source.
    source = textwrap.dedent(body).strip() + "\n"
    path = tmp_path / "arced.py"
    path.write_text(source)
    namespace: dict = {}
    exec(compile(source, str(path), "exec"), namespace)
    return os.path.realpath(str(path)), namespace


def test_arcs_are_not_recorded_unless_they_are_asked_for(tmp_path):
    path, namespace = _module(tmp_path, """
        def f(x):
            if x > 0:
                return "pos"
            return "neg"
    """)
    tracer = Tracer(TracePolicy(paths={path}))
    with tracer:
        namespace["f"](1)
    assert tracer.result.executed[path]
    assert tracer.result.arcs == {}


def test_each_branch_polarity_records_a_different_arc(tmp_path):
    path, namespace = _module(tmp_path, """
        def f(x):
            if x > 0:
                return "pos"
            return "neg"
    """)

    def arcs_for(value):
        tracer = Tracer(TracePolicy(paths={path}, arcs=True))
        with tracer:
            namespace["f"](value)
        return tracer.result.arcs[path]

    taken = arcs_for(1)
    missed = arcs_for(-1)
    assert (2, 3) in taken and (2, 4) not in taken
    assert (2, 4) in missed and (2, 3) not in missed


def test_a_frame_entry_is_recorded_as_an_arc_from_zero(tmp_path):
    path, namespace = _module(tmp_path, """
        def f():
            return 1
    """)
    tracer = Tracer(TracePolicy(paths={path}, arcs=True))
    with tracer:
        namespace["f"]()
    assert (0, 2) in tracer.result.arcs[path]


def test_recursion_does_not_invent_an_arc_between_separate_frames(tmp_path):
    path, namespace = _module(tmp_path, """
        def fact(n):
            if n <= 1:
                return 1
            return n * fact(n - 1)
    """)
    tracer = Tracer(TracePolicy(paths={path}, arcs=True))
    with tracer:
        assert namespace["fact"](4) == 24

    arcs = tracer.result.arcs[path]
    # The recursive call on line 4 enters a NEW frame, which starts at (0, 2).
    # A per-code-object "previous line" would have recorded (4, 2) instead --
    # an arc the program never took.
    assert (0, 2) in arcs
    assert (4, 2) not in arcs
    assert {(2, 3), (2, 4)} <= arcs


def test_a_loop_records_the_arc_back_to_its_header(tmp_path):
    path, namespace = _module(tmp_path, """
        def total(items):
            result = 0
            for item in items:
                result += item
            return result
    """)
    tracer = Tracer(TracePolicy(paths={path}, arcs=True))
    with tracer:
        assert namespace["total"]([1, 2, 3]) == 6

    arcs = tracer.result.arcs[path]
    assert (3, 4) in arcs, "loop body entered"
    assert (4, 3) in arcs, "loop repeated"
    assert (3, 5) in arcs, "loop exhausted"


def test_arcs_respect_the_path_allowlist(tmp_path):
    path, namespace = _module(tmp_path, """
        def f():
            return 1
    """)
    tracer = Tracer(TracePolicy(paths={"/nowhere/at/all.py"}, arcs=True))
    with tracer:
        namespace["f"]()
    assert tracer.result.arcs == {}
