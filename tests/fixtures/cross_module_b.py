"""Fixture B: imports from cross_module_a and uses them in guards."""

from tests.fixtures.cross_module_a import Point, validate_point, helper


def make_point(x: int, y: int) -> Point:
    return Point(x, y)


def process_point(p: Point) -> str:
    if validate_point(p):
        return "valid"
    return "invalid"


def uses_helper(x):
    if helper(x):
        return "ok"
    return "fail"


def combined(p: Point, x):
    if validate_point(p) and helper(x):
        return "both"
    return "neither"
