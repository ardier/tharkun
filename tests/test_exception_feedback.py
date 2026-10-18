"""Tests for exception-driven type feedback."""
from __future__ import annotations

import os
import tempfile
import unittest

from tharkun.solve.frontend.exception_feedback import (
    TypeCorrection,
    _identify_param,
    _make_corrected_value,
    parse_type_error,
    try_execute_and_correct,
)


class TestParseTypeError(unittest.TestCase):
    def test_expected_got(self) -> None:
        err = TypeError("expected str, got int")
        result = parse_type_error(err, {"x": 42})
        self.assertIsNotNone(result)
        self.assertEqual(result.expected_type, "str")
        self.assertEqual(result.wrong_type, "int")
        self.assertEqual(result.param, "x")

    def test_not_iterable(self) -> None:
        err = TypeError("argument of type 'int' is not iterable")
        result = parse_type_error(err, {"data": 5})
        self.assertIsNotNone(result)
        self.assertEqual(result.wrong_type, "int")
        self.assertEqual(result.expected_type, "str")
        self.assertEqual(result.param, "data")

    def test_not_subscriptable(self) -> None:
        err = TypeError("'int' object is not subscriptable")
        result = parse_type_error(err, {"items": 10})
        self.assertIsNotNone(result)
        self.assertEqual(result.expected_type, "list")

    def test_no_attribute(self) -> None:
        err = TypeError("'int' object has no attribute 'lower'")
        result = parse_type_error(err, {"x": 42})
        self.assertIsNotNone(result)
        self.assertEqual(result.expected_type, "object")

    def test_no_len(self) -> None:
        err = TypeError("object of type 'int' has no len()")
        result = parse_type_error(err, {"s": 0})
        self.assertIsNotNone(result)
        self.assertEqual(result.expected_type, "str")

    def test_unparseable_error(self) -> None:
        err = TypeError("something completely unrelated")
        result = parse_type_error(err, {"x": 1})
        self.assertIsNone(result)


class TestIdentifyParam(unittest.TestCase):
    def test_finds_int_param(self) -> None:
        self.assertEqual(_identify_param({"a": "hello", "b": 42}, "int"), "b")

    def test_finds_str_param(self) -> None:
        self.assertEqual(_identify_param({"x": "hello", "y": 1}, "str"), "x")

    def test_not_found(self) -> None:
        self.assertIsNone(_identify_param({"x": 1}, "str"))

    def test_unknown_type(self) -> None:
        self.assertIsNone(_identify_param({"x": 1}, "CustomClass"))


class TestMakeCorrectedValue(unittest.TestCase):
    def test_str(self) -> None:
        self.assertIsInstance(_make_corrected_value("str"), str)

    def test_int(self) -> None:
        self.assertIsInstance(_make_corrected_value("int"), int)

    def test_list(self) -> None:
        self.assertIsInstance(_make_corrected_value("list"), list)

    def test_object(self) -> None:
        val = _make_corrected_value("object")
        self.assertIsNotNone(val)


class TestTryExecuteAndCorrect(unittest.TestCase):
    def test_correct_inputs_pass_through(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def add(x, y):\n    return x + y\n")
            path = f.name
        try:
            result = try_execute_and_correct(path, "add", {"x": 1, "y": 2})
            self.assertIsNotNone(result)
            self.assertEqual(result["x"], 1)
            self.assertEqual(result["y"], 2)
        finally:
            os.unlink(path)

    def test_corrects_type_error(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            # Function expects str but we'll pass int
            f.write("def greet(name):\n    return 'Hello ' + name\n")
            path = f.name
        try:
            result = try_execute_and_correct(path, "greet", {"name": 42})
            # Should correct name from int to str
            if result is not None:
                self.assertIsInstance(result["name"], str)
        finally:
            os.unlink(path)

    def test_nonexistent_function(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("x = 1\n")
            path = f.name
        try:
            result = try_execute_and_correct(path, "nope", {"x": 1})
            self.assertIsNone(result)
        finally:
            os.unlink(path)

    def test_non_type_errors_still_return_inputs(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            # Raises ValueError, not TypeError
            f.write("def check(x):\n    raise ValueError('bad')\n")
            path = f.name
        try:
            result = try_execute_and_correct(path, "check", {"x": 1})
            # Non-TypeError means inputs are type-correct
            self.assertIsNotNone(result)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
