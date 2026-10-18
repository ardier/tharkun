"""Deepest level in the call chain fixture."""


def validate(value):
    if len(value) > 3 and value.startswith("x"):
        return True
    return False
