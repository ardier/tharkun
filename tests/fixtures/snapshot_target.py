"""Fixture for snapshot-informed re-solve tests."""


def compound_and(x, base_threshold, enabled):
    """Guard: enabled and x > threshold — threshold is a computed local."""
    threshold = base_threshold * 2
    if enabled and x > threshold:
        return True
    return False


def all_resolved(x, y):
    """Guard: x > 0 and y > 0 — snapshot has both x=5, y=10."""
    if x > 0 and y > 0:
        return True
    return False


def partial_resolved(x, limit):
    """Guard: x > limit — snapshot has limit=50 but not x."""
    if x > limit:
        return True
    return False


def string_guard(name, prefix):
    """Guard: name.startswith(prefix) — snapshot has prefix='http'."""
    if name.startswith(prefix):
        return True
    return False


def simple_truthiness(data):
    """Guard: data — just a truthiness check."""
    if data:
        return True
    return False
