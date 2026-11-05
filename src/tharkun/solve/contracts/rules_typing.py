"""Contracts for ``typing.TypeVar``, ``get_origin`` and ``get_args`` (standing batch 14, lane 1493).

``TypeVar`` stores constraints / bound / default structurally, but truth-tests its variance flags.
``get_origin`` / ``get_args`` consult class and alias attributes, which accepted user objects can override.
These protocols can propagate arbitrary exceptions, and open alias attributes can return arbitrary objects.
Exact on CPython 3.14.4 (``tests/contracts/test_typing_.py``).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec

@register_rule
class TypingTypeVar(Rule):
    """The rule of ``typing.TypeVar(name, *constraints, bound=None, ...)``."""

    SAMPLES = {"typing.TypeVar": 'import typing\nT = typing.TypeVar("T")\n'}
    TWINS = {"typing.TypeVar": "import typing\nT = typing.TypeVar()\n"}

    names = ("typing.TypeVar",)
    CONTRACT = Contract(
        qualname="typing.TypeVar",
        raises=dunder_propagation_claims("variance flags' __bool__/__len__",
            native={"TypeError": "name is not a str; exactly one constraint is given (need zero or at least two); constraints and bound are both given; an unknown keyword; the wrong argument count; or propagates uncaught from a variance flag's truth protocol"},
            tested_native=frozenset({"TypeError"})),
        may_raise_other=True,
        returns=ReturnSpec("typing.TypeVar", "a distinct type variable named name; covariant / contravariant / infer_variance only set flags on it", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/typing.html#typing.TypeVar", "CPython Objects/typevarobject.c"),
        tested=True,
    )


@register_rule
class TypingGetOrigin(Rule):
    """The rule of ``typing.get_origin(tp)``."""

    SAMPLES = {"typing.get_origin": 'import typing\nx = typing.get_origin(typing.List[int])\n'}
    TWINS = {"typing.get_origin": "import typing\nx = typing.get_origin()\n"}

    names = ("typing.get_origin",)
    CONTRACT = Contract(
        qualname="typing.get_origin",
        raises=dunder_propagation_claims("class/alias attribute lookup, including an open __origin__",
            native={"TypeError": "wrong argument count or an unknown keyword; or propagates uncaught from class/alias attribute lookup"}, tested_native=frozenset({"TypeError"})),
        may_raise_other=True,
        returns=ReturnSpec("object", "canonical aliases return their origin (a class, typing special form or parameter), plain values return None; overridden alias attributes may return any object", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/typing.html#typing.get_origin", "CPython Lib/typing.py::get_origin"),
        tested=True,
    )


@register_rule
class TypingGetArgs(Rule):
    """The rule of ``typing.get_args(tp)``."""

    SAMPLES = {"typing.get_args": 'import typing\nx = typing.get_args(typing.List[int])\n'}
    TWINS = {"typing.get_args": "import typing\nx = typing.get_args()\n"}

    names = ("typing.get_args",)
    CONTRACT = Contract(
        qualname="typing.get_args",
        raises=dunder_propagation_claims("class/alias attribute lookup, including open __args__/__origin__/__metadata__",
            native={"TypeError": "wrong argument count or an unknown keyword; or propagates uncaught from class/alias attribute lookup or metadata operations"}, tested_native=frozenset({"TypeError"})),
        may_raise_other=True,
        returns=ReturnSpec("object", "canonical aliases return an argument tuple, plain values return (); overridden alias attributes and metadata operations may return any object", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/typing.html#typing.get_args", "CPython Lib/typing.py::get_args"),
        tested=True,
    )


__all__ = ["TypingTypeVar", "TypingGetOrigin", "TypingGetArgs"]
