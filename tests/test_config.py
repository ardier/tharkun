"""Tests for the tharkun configuration module (config.py)."""

from __future__ import annotations

import unittest

from tharkun.solve.config import Config, config, configure, reset


class TestConfig(unittest.TestCase):
    """Test configuration management."""

    def setUp(self) -> None:
        reset()

    def tearDown(self) -> None:
        reset()

    def test_defaults(self) -> None:
        self.assertEqual(config.snapshot_timeout_s, 2.0)
        self.assertEqual(config.max_decompose_depth, 5)
        self.assertEqual(config.int_range_default, (-10000, 10000))

    def test_configure(self) -> None:
        configure(snapshot_timeout_s=5.0)
        self.assertEqual(config.snapshot_timeout_s, 5.0)

    def test_configure_unknown_key(self) -> None:
        with self.assertRaises(ValueError):
            configure(nonexistent_key=42)

    def test_reset(self) -> None:
        configure(snapshot_timeout_s=99.0)
        self.assertEqual(config.snapshot_timeout_s, 99.0)
        reset()
        self.assertEqual(config.snapshot_timeout_s, 2.0)

    def test_multiple_overrides(self) -> None:
        configure(
            snapshot_timeout_s=10.0,
            max_decompose_depth=10,
        )
        self.assertEqual(config.snapshot_timeout_s, 10.0)
        self.assertEqual(config.max_decompose_depth, 10)

    def test_new_config_fields(self) -> None:
        """Verify the new fields added for magic number consolidation."""
        self.assertEqual(config.hypothesis_int_range, (-1000, 1000))
        self.assertEqual(config.narrowed_window, 1000)


if __name__ == "__main__":
    unittest.main()
