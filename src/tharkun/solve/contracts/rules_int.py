"""Callback-free integer queries, checked against CPython 3.14.4."""
from __future__ import annotations

from tharkun.solve.contracts._helpers import never_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec


def _contract(method, result, shape):
    return Contract(
        qualname=f"int.{method}",
        raises=never_claims(native={"TypeError": (
            "receiver is not an int (bool and int subclasses are accepted), or arguments/keywords are supplied; "
            f"CPython 3.14.4 args: ('int.{method}() takes no arguments (1 given)',) for one extra positional; "
            f"('int.{method}() takes no keyword arguments',) for keywords; "
            f"(\"descriptor '{method}' for 'int' objects doesn't apply to a 'str' object\",) for a str receiver; "
            f"('unbound method int.{method}() needs an argument',) for a missing receiver"
        )}, tested=frozenset(ALL_PROPERTY_CLASSES)),
        may_raise_other=False,
        returns=ReturnSpec(result, shape, tested=True),
        effects=Effects(deterministic=True, may_execute_callback=False),
        source=("https://docs.python.org/3.14/library/stdtypes.html#int." + method,
                "CPython Objects/longobject.c::int_" + method + "_impl"),
        tested=True,
    )


@register_rule
class IntBitLength(Rule):
    names = ("int.bit_length",)
    SAMPLES = {"int.bit_length": "x = (255).bit_length()\n"}
    TWINS = {"int.bit_length": "x = (255).bit_length(1)\n"}
    CONTRACT = _contract("bit_length", "int", "bit length of abs(self); zero iff self == 0")


@register_rule
class IntBitCount(Rule):
    names = ("int.bit_count",)
    SAMPLES = {"int.bit_count": "x = (255).bit_count()\n"}
    TWINS = {"int.bit_count": "x = (255).bit_count(1)\n"}
    CONTRACT = _contract("bit_count", "int", "population count of abs(self); zero iff self == 0")


@register_rule
class IntConjugate(Rule):
    names = ("int.conjugate",)
    SAMPLES = {"int.conjugate": "x = (-255).conjugate()\n"}
    TWINS = {"int.conjugate": "x = (-255).conjugate(1)\n"}
    CONTRACT = _contract("conjugate", "int", "same numeric value as self; bool becomes int")


@register_rule
class IntIsInteger(Rule):
    names = ("int.is_integer",)
    SAMPLES = {"int.is_integer": "x = (-255).is_integer()\n"}
    TWINS = {"int.is_integer": "x = (-255).is_integer(1)\n"}
    CONTRACT = _contract("is_integer", "bool", "always True, including zero and bool receivers")


@register_rule
class IntAsIntegerRatio(Rule):
    names = ("int.as_integer_ratio",)
    SAMPLES = {"int.as_integer_ratio": "x = (-255).as_integer_ratio()\n"}
    TWINS = {"int.as_integer_ratio": "x = (-255).as_integer_ratio(1)\n"}
    CONTRACT = _contract("as_integer_ratio", "tuple[int,int]", "(numeric value of self, 1); bool numerator becomes int")
