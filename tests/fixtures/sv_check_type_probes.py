"""Named native calls whose MR1793 outcomes are recorded in a fixture."""

from __future__ import annotations

import builtins


PROPERTY_EXCEPTIONS = (
    "AssertionError",
    "AttributeError",
    "IndexError",
    "KeyError",
    "TypeError",
    "FloatingPointError",
    "ZeroDivisionError",
)


def check_type_probe_calls():
    """Return stable-label calls; this describes inputs, never checker behavior."""
    calls = {
        "match:int": lambda check: check(1, int),
        "match:bool-as-int": lambda check: check(True, int),
        "match:bool": lambda check: check(True, bool),
        "match:float": lambda check: check(1.5, float),
        "match:str": lambda check: check("x", str),
        "match:list": lambda check: check([], list),
        "match:dict": lambda check: check({}, dict),
        "mismatch:int-as-str": lambda check: check(1, str),
        "binding:extra-positional": lambda check: check(1, int, str),
        "binding:duplicate-value": lambda check: check(1, int, value=2),
        "binding:missing-value": lambda check: check(hint=int),
    }

    for exception in PROPERTY_EXCEPTIONS:
        error = getattr(builtins, exception)

        class Meta(type):
            def __instancecheck__(cls, value, _error=error):
                raise _error("type checker hook")

        class CustomHint(metaclass=Meta):
            pass

        class CustomValue:
            @property
            def __class__(self, _error=error):
                raise _error("value class hook")

        calls[f"hook:metaclass:{exception}"] = (
            lambda check, hint=CustomHint: check(object(), hint)
        )
        calls[f"hook:class-attribute:{exception}"] = (
            lambda check, value=CustomValue(): check(value, int)
        )

    class BadRepr:
        def __repr__(self):
            raise AssertionError("mismatch diagnostic repr")

    class BadTupleRepr:
        def __repr__(self):
            raise AssertionError("tuple mismatch diagnostic repr")

    calls["repr:list-mismatch"] = lambda check: check([BadRepr()], str)
    calls["repr:tuple-mismatch"] = lambda check: check((BadTupleRepr(),), int)
    return calls
