"""Tests for the RecordingProxy type inference system."""
from __future__ import annotations

import unittest

from tharkun.solve.frontend.proxy import RecordingProxy, get_accessed_attrs, infer_type_from_usage


class TestRecordingProxy(unittest.TestCase):
    def test_records_attr_access(self) -> None:
        p = RecordingProxy()
        _ = p.foo
        _ = p.bar
        attrs = get_accessed_attrs(p)
        self.assertEqual(attrs, {"foo", "bar"})

    def test_records_subscript(self) -> None:
        p = RecordingProxy()
        _ = p[0]
        self.assertTrue(object.__getattribute__(p, "_subscript_used"))

    def test_records_iter(self) -> None:
        p = RecordingProxy()
        for _ in p:
            pass
        self.assertTrue(object.__getattribute__(p, "_iter_used"))

    def test_records_len(self) -> None:
        p = RecordingProxy()
        _ = len(p)
        self.assertTrue(object.__getattribute__(p, "_len_used"))

    def test_records_contains(self) -> None:
        p = RecordingProxy()
        _ = "x" in p
        self.assertTrue(object.__getattribute__(p, "_contains_used"))

    def test_is_truthy(self) -> None:
        p = RecordingProxy()
        self.assertTrue(bool(p))

    def test_callable_returns_proxy(self) -> None:
        p = RecordingProxy()
        result = p.method()
        self.assertIsInstance(result, RecordingProxy)

    def test_chained_access(self) -> None:
        p = RecordingProxy()
        _ = p.foo.bar
        attrs = get_accessed_attrs(p)
        self.assertIn("foo", attrs)


class TestInferTypeFromUsage(unittest.TestCase):
    def test_str_methods(self) -> None:
        p = RecordingProxy()
        _ = p.startswith
        _ = p.upper
        self.assertEqual(infer_type_from_usage(p), "str")

    def test_list_methods(self) -> None:
        p = RecordingProxy()
        _ = p.append
        _ = p.sort
        self.assertEqual(infer_type_from_usage(p), "list")

    def test_dict_methods(self) -> None:
        p = RecordingProxy()
        _ = p.keys
        _ = p.get
        self.assertEqual(infer_type_from_usage(p), "dict")

    def test_set_methods(self) -> None:
        p = RecordingProxy()
        _ = p.add
        _ = p.discard
        self.assertEqual(infer_type_from_usage(p), "set")

    def test_subscript_suggests_list_or_dict(self) -> None:
        p = RecordingProxy()
        _ = p[0]
        result = infer_type_from_usage(p)
        self.assertIn(result, ("list", "dict"))

    def test_get_plus_subscript_suggests_dict(self) -> None:
        p = RecordingProxy()
        _ = p.get
        _ = p["key"]
        self.assertEqual(infer_type_from_usage(p), "dict")

    def test_unknown_attrs_suggest_object(self) -> None:
        p = RecordingProxy()
        _ = p.some_custom_attr
        _ = p.another_one
        self.assertEqual(infer_type_from_usage(p), "object")

    def test_no_usage_returns_any(self) -> None:
        p = RecordingProxy()
        self.assertEqual(infer_type_from_usage(p), "Any")

    def test_real_world_str_usage(self) -> None:
        """Simulate how a function might use a string parameter."""
        p = RecordingProxy()
        try:
            if p.startswith("http"):
                pass
        except Exception:
            pass
        self.assertEqual(infer_type_from_usage(p), "str")


if __name__ == "__main__":
    unittest.main()
