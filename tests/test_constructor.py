"""Tests for the real object constructor module."""
from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

from tharkun.solve.frontend.constructor import (
    ConstructorParameter,
    ConstructorTemplate,
    _extract_init_params,
    _synthesize_arg,
    construct_instance,
    constructor_template,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "constructor_target.py")


class TestSynthesizeArg(unittest.TestCase):
    def test_int(self) -> None:
        self.assertEqual(_synthesize_arg("int"), 0)

    def test_str(self) -> None:
        self.assertEqual(_synthesize_arg("str"), "")

    def test_optional(self) -> None:
        self.assertIsNone(_synthesize_arg("Optional[str]"))

    def test_union_none(self) -> None:
        self.assertIsNone(_synthesize_arg("str | None"))

    def test_list_generic(self) -> None:
        self.assertEqual(_synthesize_arg("list[int]"), [])

    def test_dict_generic(self) -> None:
        self.assertEqual(_synthesize_arg("dict[str, int]"), {})

    def test_none_annotation(self) -> None:
        self.assertIsNone(_synthesize_arg(None))

    def test_with_default(self) -> None:
        _MISSING = object()
        self.assertEqual(_synthesize_arg("int", 42), 42)


class TestExtractInitParams(unittest.TestCase):
    def test_simple_class(self) -> None:
        from tharkun.solve.frontend.import_resolver import load_class_def
        cls_def = load_class_def(FIXTURE, "SimpleClass")
        params = _extract_init_params(cls_def)
        # SimpleClass.__init__ only has self
        self.assertEqual(params, [])

    def test_with_args(self) -> None:
        from tharkun.solve.frontend.import_resolver import load_class_def
        cls_def = load_class_def(FIXTURE, "WithArgs")
        params = _extract_init_params(cls_def)
        self.assertEqual(len(params), 2)
        self.assertEqual(params[0][0], "name")  # name
        self.assertEqual(params[0][1], "str")   # annotation
        self.assertEqual(params[1][0], "count")
        self.assertEqual(params[1][2], 5)  # default

    def test_no_init(self) -> None:
        from tharkun.solve.frontend.import_resolver import load_class_def
        cls_def = load_class_def(FIXTURE, "NoInit")
        params = _extract_init_params(cls_def)
        self.assertEqual(params, [])


class TestConstructorTemplate(unittest.TestCase):
    def test_records_all_parameter_kinds_and_literal_defaults(self) -> None:
        template = constructor_template(FIXTURE, "SignatureKinds")

        self.assertIsInstance(template, ConstructorTemplate)
        assert template is not None
        self.assertEqual(
            [(parameter.name, parameter.kind) for parameter in template.parameters],
            [
                ("token", "positional_only"),
                ("count", "positional_or_keyword"),
                ("labels", "var_positional"),
                ("enabled", "keyword_only"),
                ("metadata", "keyword_only"),
                ("extras", "var_keyword"),
            ],
        )
        metadata = template.parameters[4]
        self.assertTrue(metadata.has_default)
        self.assertTrue(metadata.default_is_literal)
        self.assertEqual(metadata.default, {"answer": 42})
        self.assertEqual(metadata.default_source, "{'answer': 42}")

    def test_non_literal_default_is_distinguished_from_no_default(self) -> None:
        template = constructor_template(FIXTURE, "WithNonLiteralDefault")

        assert template is not None
        value = template.parameters[0]
        self.assertTrue(value.has_default)
        self.assertFalse(value.default_is_literal)
        self.assertIsNone(value.default)
        self.assertEqual(value.default_source, "DEFAULT_VALUE")

    def test_missing_class_returns_none(self) -> None:
        self.assertIsNone(constructor_template(FIXTURE, "DoesNotExist"))

    def test_records_are_frozen(self) -> None:
        parameter = ConstructorParameter("value", "int", "keyword_only")
        with self.assertRaises(FrozenInstanceError):
            parameter.name = "other"  # type: ignore[misc]

    def test_analysis_does_not_execute_target_module(self) -> None:
        source = (
            "raise RuntimeError('must not run')\n"
            "class Safe:\n"
            "    def __init__(self, x: int = 3): pass\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py") as target:
            target.write(source)
            target.flush()
            template = constructor_template(target.name, "Safe")

        assert template is not None
        self.assertEqual(template.parameters[0].default, 3)


class TestConstructInstance(unittest.TestCase):
    def test_simple_class(self) -> None:
        obj = construct_instance(FIXTURE, "SimpleClass")
        self.assertNotIsInstance(obj, SimpleNamespace)
        self.assertEqual(obj.value, 0)

    def test_with_defaults(self) -> None:
        obj = construct_instance(FIXTURE, "WithDefaults")
        self.assertNotIsInstance(obj, SimpleNamespace)
        self.assertEqual(obj.x, 10)
        self.assertEqual(obj.y, 3.14)
        self.assertEqual(obj.label, "default")

    def test_with_args(self) -> None:
        obj = construct_instance(FIXTURE, "WithArgs")
        self.assertNotIsInstance(obj, SimpleNamespace)
        # name should be "" (synthesized from str annotation), count should be 5 (default)
        self.assertEqual(obj.name, "")
        self.assertEqual(obj.count, 5)

    def test_no_init(self) -> None:
        obj = construct_instance(FIXTURE, "NoInit")
        # Should construct successfully — no args needed
        self.assertNotIsInstance(obj, SimpleNamespace)

    def test_positional_only_and_keyword_only_args(self) -> None:
        obj = construct_instance(FIXTURE, "SignatureKinds")
        self.assertNotIsInstance(obj, SimpleNamespace)
        self.assertEqual(obj.token, "")
        self.assertEqual(obj.count, 2)
        self.assertEqual(obj.labels, ())
        self.assertTrue(obj.enabled)
        self.assertEqual(obj.metadata, {"answer": 42})

    def test_static_fallback_keeps_positional_only_attributes(self) -> None:
        from tharkun.solve.loader import no_targets

        with no_targets():
            obj = construct_instance(
                FIXTURE,
                "SignatureKinds",
                overrides={"token": "supplied"},
            )

        self.assertIsInstance(obj, SimpleNamespace)
        self.assertEqual(obj.token, "supplied")
        self.assertEqual(obj.count, 2)

    def test_nonexistent_class(self) -> None:
        obj = construct_instance(FIXTURE, "DoesNotExist")
        self.assertIsInstance(obj, SimpleNamespace)

    def test_nonexistent_file(self) -> None:
        obj = construct_instance("/nonexistent/file.py", "Foo")
        self.assertIsInstance(obj, SimpleNamespace)


if __name__ == "__main__":
    unittest.main()
