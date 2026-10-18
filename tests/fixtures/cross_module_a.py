"""Fixture A: defines classes and validators imported by cross_module_b."""


class Point:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


def validate_point(p):
    return p.x > 0 and p.y > 0


def helper(x):
    return x > 0 and x < 100
