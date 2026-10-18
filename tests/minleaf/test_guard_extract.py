"""Phase 1 gate: 7 tests for guard_extract.py."""

import ast
import tempfile
import os

import pytest

from tharkun.solve.guard import (
    Frontier,
    desired_guard_value,
    extract_frontier,
)


def _write_tmp(code: str) -> str:
    """Write code to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(code)
    return path


def test_extract_if():
    path = _write_tmp("if x > 0:\n    pass\n")
    try:
        result = extract_frontier(path, 1)
        assert result.stmt_kind == "if"
        assert isinstance(result.guard_expr, ast.Compare)
        assert "x" in result.guard_src and "0" in result.guard_src
    finally:
        os.unlink(path)


def test_extract_while():
    path = _write_tmp("while n < 10:\n    pass\n")
    try:
        result = extract_frontier(path, 1)
        assert result.stmt_kind == "while"
    finally:
        os.unlink(path)


def test_extract_assert():
    path = _write_tmp("assert valid\n")
    try:
        result = extract_frontier(path, 1)
        assert result.stmt_kind == "assert"
    finally:
        os.unlink(path)


def test_extract_elif():
    path = _write_tmp("if a:\n  pass\nelif b > 5:\n  pass\n")
    try:
        result = extract_frontier(path, 3)
        assert "b" in result.guard_src
        assert "a" not in result.guard_src
    finally:
        os.unlink(path)


def test_extract_nested_innermost():
    path = _write_tmp("if a:\n  if b:\n    pass\n")
    try:
        result = extract_frontier(path, 2)
        assert "b" in result.guard_src
    finally:
        os.unlink(path)


def test_extract_missing_raises():
    path = _write_tmp("x = 5\n")
    try:
        with pytest.raises(ValueError):
            extract_frontier(path, 1)
    finally:
        os.unlink(path)


def test_desired_guard_value():
    assert desired_guard_value("if", "then") is True
    assert desired_guard_value("if", "else") is False
    assert desired_guard_value("assert", "then") is True
    assert desired_guard_value("assert", "else") is False
    assert desired_guard_value("while", "then") is True
    with pytest.raises(ValueError):
        desired_guard_value("if", "banana")
