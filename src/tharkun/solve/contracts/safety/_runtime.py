"""Runtime guards for CPython-version-pinned safety templates."""

from __future__ import annotations

import sys


def supports_cpython_314() -> bool:
    """Whether the running interpreter matches the measured 3.14 template domain."""
    return sys.implementation.name == "cpython" and sys.version_info[:2] == (3, 14)


__all__ = ["supports_cpython_314"]
