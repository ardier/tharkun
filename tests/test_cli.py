"""Tests for the tharkun CLI (cli.py)."""

from __future__ import annotations

import json
import os
import sys
import unittest
from io import StringIO
from unittest.mock import patch

from tharkun.cli import main

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "condition_guards.py")


class TestCLISolve(unittest.TestCase):
    """Test the 'solve' subcommand."""

    def test_solve_returns_json(self) -> None:
        with patch("sys.stdout", new_callable=StringIO) as mock_out, \
             patch("sys.stderr", new_callable=StringIO):
            rc = main(["solve", FIXTURE, "5"])
        self.assertEqual(rc, 0)
        data = json.loads(mock_out.getvalue())
        self.assertIn("guard_source", data)
        self.assertIn("satisfying_inputs", data)

    def test_solve_nonexistent_file(self) -> None:
        with patch("sys.stdout", new_callable=StringIO), \
             patch("sys.stderr", new_callable=StringIO):
            # Should not crash — returns valid JSON even for missing file
            rc = main(["solve", "/nonexistent.py", "1"])
        self.assertEqual(rc, 0)


class TestCLIDecompose(unittest.TestCase):
    """Test the 'decompose' subcommand."""

    def test_decompose_returns_json(self) -> None:
        with patch("sys.stdout", new_callable=StringIO) as mock_out, \
             patch("sys.stderr", new_callable=StringIO):
            rc = main(["decompose", FIXTURE, "5"])
        self.assertEqual(rc, 0)
        data = json.loads(mock_out.getvalue())
        self.assertIn("total_leaves", data)

    def test_decompose_max_depth(self) -> None:
        with patch("sys.stdout", new_callable=StringIO) as mock_out, \
             patch("sys.stderr", new_callable=StringIO):
            rc = main(["decompose", FIXTURE, "5", "--max-depth", "2"])
        self.assertEqual(rc, 0)
        json.loads(mock_out.getvalue())  # Should be valid JSON


class TestCLIMocks(unittest.TestCase):
    """Test the 'mocks' subcommand."""

    def test_mocks_returns_json(self) -> None:
        mock_fixture = os.path.join(
            os.path.dirname(__file__), "fixtures", "mock_targets.py"
        )
        with patch("sys.stdout", new_callable=StringIO) as mock_out, \
             patch("sys.stderr", new_callable=StringIO):
            rc = main(["mocks", mock_fixture])
        self.assertEqual(rc, 0)
        data = json.loads(mock_out.getvalue())
        self.assertIn("plans", data)


class TestCLINoCommand(unittest.TestCase):
    """Test behavior with no subcommand."""

    def test_no_command_returns_1(self) -> None:
        with patch("sys.stdout", new_callable=StringIO), \
             patch("sys.stderr", new_callable=StringIO):
            rc = main([])
        self.assertEqual(rc, 1)

    def test_version(self) -> None:
        with self.assertRaises(SystemExit) as ctx:
            main(["--version"])
        self.assertEqual(ctx.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
