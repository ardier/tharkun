"""Stress tests for guard decomposer + runtime snapshot analyzer.

10 real-world scenarios covering deep recursion, classmethods, lambdas,
decorators, *args/**kwargs, large locals, circular refs, loop snapshots,
exception snapshots, and complex types.
"""
from __future__ import annotations

import os
import unittest

from tharkun.solve.decompose import decompose_guard
from tharkun.guard_analyzer import analyze_blocker
from tharkun.trace_api import TracePolicy, Tracer, _safe_snapshot

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "stress_guards.py")


# ===== DECOMPOSER STRESS TESTS =====


class TestDeepRecursionChain(unittest.TestCase):
    """Scenario 1: 3-level recursive validators."""

    def test_resolves_three_levels(self) -> None:
        result = decompose_guard(FIXTURE, 29)  # if level1_top(...)
        self.assertEqual(result.root.kind, "call")
        self.assertEqual(result.root.resolved_function, "level1_top")
        self.assertGreaterEqual(result.max_depth, 2)

    def test_has_solvable_leaves(self) -> None:
        result = decompose_guard(FIXTURE, 29)
        self.assertGreater(result.solvable_leaves, 0)
        # Some conditions may be unsolvable (negated compound guards produce
        # UnaryOp(Not) that leaf solver can't fully handle), but the simple
        # comparisons like value > 0, value < 1000 should be solvable.
        self.assertGreater(result.total_leaves, 3)


class TestClassmethodStaticmethod(unittest.TestCase):
    """Scenario 2: classmethod + staticmethod validators."""

    def test_resolves_guard(self) -> None:
        result = decompose_guard(FIXTURE, 53)  # if guard_class_validators(...)
        self.assertEqual(result.root.kind, "call")
        self.assertEqual(result.root.resolved_function, "guard_class_validators")

    def test_and_has_two_children(self) -> None:
        result = decompose_guard(FIXTURE, 53)
        and_node = result.root.children[0]
        self.assertEqual(and_node.kind, "and")
        self.assertEqual(len(and_node.children), 2)

    def test_both_are_calls(self) -> None:
        result = decompose_guard(FIXTURE, 53)
        and_node = result.root.children[0]
        for child in and_node.children:
            self.assertEqual(child.kind, "call")


class TestLambdaAndHigherOrder(unittest.TestCase):
    """Scenario 3: lambda + higher-order function validators."""

    def test_lambda_has_blockers(self) -> None:
        result = decompose_guard(FIXTURE, 75)  # if guard_with_lambda(x):
        self.assertEqual(result.root.resolved_function, "guard_with_lambda")
        # even_check is a lambda → unresolvable
        self.assertGreater(result.unsolvable_leaves, 0)

    def test_lambda_has_solvable_leaf_too(self) -> None:
        result = decompose_guard(FIXTURE, 75)
        # x > 0 should be solvable
        self.assertGreater(result.solvable_leaves, 0)

    def test_higher_order_has_blockers(self) -> None:
        result = decompose_guard(FIXTURE, 79)  # if guard_higher_order(x):
        self.assertEqual(result.root.resolved_function, "guard_higher_order")
        # apply_validator calls validator(value) which is a parameter → unsolvable
        self.assertGreater(result.unsolvable_leaves, 0)


class TestDecoratedFunctions(unittest.TestCase):
    """Scenario 4: @my_cache decorated functions."""

    def test_decorated_guard_resolves(self) -> None:
        result = decompose_guard(FIXTURE, 99)  # if guard_decorated(x):
        self.assertEqual(result.root.kind, "call")
        self.assertEqual(result.root.resolved_function, "guard_decorated")

    def test_has_leaves(self) -> None:
        result = decompose_guard(FIXTURE, 99)
        # Whether cached_is_valid is resolved or not, we should have some analysis
        self.assertGreater(result.total_leaves, 0)


class TestVarargsKwargs(unittest.TestCase):
    """Scenario 5: *args/**kwargs in validators."""

    def test_varargs_resolves(self) -> None:
        result = decompose_guard(FIXTURE, 122)  # if guard_varargs(x, y, z):
        self.assertEqual(result.root.resolved_function, "guard_varargs")
        self.assertGreater(result.total_leaves, 0)

    def test_kwargs_resolves(self) -> None:
        result = decompose_guard(FIXTURE, 126)  # if guard_kwargs(field):
        self.assertEqual(result.root.resolved_function, "guard_kwargs")
        self.assertGreater(result.total_leaves, 0)


# ===== SNAPSHOT STRESS TESTS =====


class TestSnapshotLargeLocals(unittest.TestCase):
    """Scenario 6: large dict (1000+ keys) in locals."""

    def test_snapshot_does_not_oom(self) -> None:
        rp = os.path.realpath(FIXTURE)
        # Line 139 is the `if not all(...)` guard inside validate_large_config
        policy = TracePolicy(enabled=True, snapshot_lines={rp: {136, 139}}, paths={rp})
        import importlib.util
        spec = importlib.util.spec_from_file_location("_snap_large", rp)
        mod = importlib.util.module_from_spec(spec)
        with Tracer(policy) as tr:
            spec.loader.exec_module(mod)
            mod.uses_large_locals({"wrong": "key"})
        snaps = tr.result.snapshots.get(rp, {})
        # Should have captured something without OOM
        self.assertGreater(len(snaps), 0)

    def test_large_dict_is_shallow(self) -> None:
        big = {f"k{i}": i for i in range(1000)}
        snap = _safe_snapshot(big)
        self.assertIsInstance(snap, dict)
        self.assertEqual(snap["__type__"], "dict")
        self.assertEqual(snap["len"], 1000)
        # Keys should be truncated to 20
        self.assertLessEqual(len(snap["keys"]), 20)


class TestSnapshotCircularRefs(unittest.TestCase):
    """Scenario 7: circular references in locals."""

    def test_no_infinite_recursion(self) -> None:
        rp = os.path.realpath(FIXTURE)
        policy = TracePolicy(enabled=True, snapshot_lines={rp: {156, 160}}, paths={rp})
        import importlib.util
        spec = importlib.util.spec_from_file_location("_snap_circ", rp)
        mod = importlib.util.module_from_spec(spec)

        from tests.fixtures.stress_guards import Node
        node = Node(50)
        node.parent = node  # circular

        with Tracer(policy) as tr:
            spec.loader.exec_module(mod)
            mod.uses_circular(node)
        # Should complete without hanging
        self.assertIsNotNone(tr.result)

    def test_circular_object_safe_snapshot(self) -> None:
        from tests.fixtures.stress_guards import Node
        node = Node(42)
        node.parent = node
        snap = _safe_snapshot(node)
        # Should be a dict with __type__ and repr, not infinite
        self.assertIsInstance(snap, dict)
        self.assertEqual(snap["__type__"], "Node")


class TestSnapshotLoopIterations(unittest.TestCase):
    """Scenario 8: multiple snapshots at same line (loop)."""

    def test_accumulates_snapshots(self) -> None:
        rp = os.path.realpath(FIXTURE)
        # Line 171: if item < 0
        policy = TracePolicy(enabled=True, snapshot_lines={rp: {171}}, paths={rp})
        import importlib.util
        spec = importlib.util.spec_from_file_location("_snap_loop", rp)
        mod = importlib.util.module_from_spec(spec)
        with Tracer(policy) as tr:
            spec.loader.exec_module(mod)
            mod.uses_loop([100, 200, -1, 500])
        snaps = tr.result.snapshots.get(rp, {}).get(171, [])
        # Should have 3 snapshots (iterations 0, 1, 2 — blocker at item=-1)
        self.assertGreaterEqual(len(snaps), 3)

    def test_iteration_values_differ(self) -> None:
        rp = os.path.realpath(FIXTURE)
        policy = TracePolicy(enabled=True, snapshot_lines={rp: {171}}, paths={rp})
        import importlib.util
        spec = importlib.util.spec_from_file_location("_snap_loop2", rp)
        mod = importlib.util.module_from_spec(spec)
        with Tracer(policy) as tr:
            spec.loader.exec_module(mod)
            mod.uses_loop([10, 20, 30])
        snaps = tr.result.snapshots.get(rp, {}).get(171, [])
        items = [s.get("item") for s in snaps if "item" in s]
        # Items should differ across iterations
        self.assertGreater(len(set(items)), 1)


class TestSnapshotAtException(unittest.TestCase):
    """Scenario 9: snapshot capture at/before exception."""

    def test_captures_state_before_exception(self) -> None:
        rp = os.path.realpath(FIXTURE)
        # Line 188: if mode == "strict" and value == 0 (before raise)
        policy = TracePolicy(enabled=True, snapshot_lines={rp: {188}}, paths={rp})
        import importlib.util
        spec = importlib.util.spec_from_file_location("_snap_exc", rp)
        mod = importlib.util.module_from_spec(spec)
        with Tracer(policy) as tr:
            spec.loader.exec_module(mod)
            mod.uses_exception(0, "strict")
        snaps = tr.result.snapshots.get(rp, {}).get(188, [])
        self.assertGreater(len(snaps), 0)
        self.assertEqual(snaps[0].get("value"), 0)
        self.assertEqual(snaps[0].get("mode"), "strict")


class TestSnapshotComplexTypes(unittest.TestCase):
    """Scenario 10: dataclass + enum + nested types in snapshot."""

    def test_captures_complex_types(self) -> None:
        rp = os.path.realpath(FIXTURE)
        # Line 225: if timeout < 0
        policy = TracePolicy(enabled=True, snapshot_lines={rp: {225}}, paths={rp})
        import importlib.util
        spec = importlib.util.spec_from_file_location("_snap_complex", rp)
        mod = importlib.util.module_from_spec(spec)

        from tests.fixtures.stress_guards import Request, Status, AppConfig
        req = Request(id=1, status=Status.PENDING, payload={})
        config = AppConfig(version="1.0", requests=[req], settings={})

        with Tracer(policy) as tr:
            spec.loader.exec_module(mod)
            mod.uses_complex_types(config, -1.0)
        snaps = tr.result.snapshots.get(rp, {}).get(225, [])
        self.assertGreater(len(snaps), 0)
        snap = snaps[0]
        self.assertEqual(snap.get("timeout"), -1.0)
        # config should be captured as object repr
        self.assertIn("config", snap)

    def test_enum_safe_snapshot(self) -> None:
        from tests.fixtures.stress_guards import Status
        snap = _safe_snapshot(Status.RUNNING)
        self.assertIsInstance(snap, dict)
        self.assertIn("repr", snap)

    def test_dataclass_safe_snapshot(self) -> None:
        from tests.fixtures.stress_guards import Request, Status
        req = Request(id=1, status=Status.PENDING, payload={"k": "v"})
        snap = _safe_snapshot(req)
        self.assertIsInstance(snap, dict)
        self.assertEqual(snap["__type__"], "Request")


if __name__ == "__main__":
    unittest.main()
