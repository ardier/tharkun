"""Fixture for testing step 6 execution with seed kwargs.

These functions have guards that are hard to solve statically but
easy once you know runtime variable values.
"""


def compute_and_check(x: int, y: int):
    """Guard depends on a computed intermediate value."""
    threshold = x * 2 + 10
    if y > threshold:
        return "above"
    return "below"


def config_driven(data: str, multiplier: int = 3):
    """Guard depends on len(data) * multiplier — a compound computation."""
    adjusted_len = len(data) * multiplier
    if adjusted_len > 20:
        return "long"
    return "short"


def multi_path(flag: bool, value: int = 0):
    """Multiple branches — flag controls which path."""
    if flag:
        computed = value + 100
    else:
        computed = value - 100
    if computed > 50:
        return "high"
    return "low"
