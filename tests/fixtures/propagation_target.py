"""Fixture for cross-function type propagation tests."""


def typed_callee(x: int, name: str) -> bool:
    return x > 0 and len(name) > 0


def untyped_caller(a, b):
    """Calls typed_callee(a, b) — should propagate int, str back."""
    return typed_callee(a, b)


def partial_caller(a, b: str):
    """a is untyped but passed to typed_callee positionally."""
    return typed_callee(a, b)


def keyword_caller(val, label):
    """Uses keyword args to call typed_callee."""
    return typed_callee(x=val, name=label)


def no_propagation(a, b):
    """Calls a function that doesn't exist in this file."""
    return some_external(a, b)


def fully_typed(x: int, y: float) -> str:
    return str(x + y)
