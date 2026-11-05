"""Native float identity, integrality and ratio queries (CPython 3.14.4)."""
from __future__ import annotations

from tharkun.solve.contracts._helpers import never_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec


def _contract(method, result, shape):
    return Contract(
        qualname="float." + method,
        raises=never_claims(native={"TypeError": (
            "receiver is not a float (float subclasses are accepted), or arguments/keywords are supplied; "
            f"CPython 3.14.4 args: ('float.{method}() takes no arguments (1 given)',) for one extra positional; "
            f"('float.{method}() takes no keyword arguments',) for keywords; "
            f"(\"descriptor '{method}' for 'float' objects doesn't apply to a 'int' object\",) for int receiver; "
            f"('unbound method float.{method}() needs an argument',) for missing receiver"
        )}, tested=frozenset(ALL_PROPERTY_CLASSES)),
        may_raise_other=method == "as_integer_ratio",
        returns=ReturnSpec(result, shape, tested=True),
        effects=Effects(deterministic=True, may_execute_callback=False),
        source=("https://docs.python.org/3.14/library/stdtypes.html#float." + method,
                "CPython Objects/floatobject.c"),
        notes=("as_integer_ratio: ValueError args ('cannot convert NaN to integer ratio',) for NaN; "
               "OverflowError args ('cannot convert Infinity to integer ratio',) for either infinity"
               if method == "as_integer_ratio" else "NaN and infinities are accepted without raising"),
        tested=True,
    )


@register_rule
class FloatConjugate(Rule):
    names = ("float.conjugate",)
    SAMPLES = {"float.conjugate": "x = (-1.5).conjugate()\n"}
    TWINS = {"float.conjugate": "x = (-1.5).conjugate(1)\n"}
    CONTRACT = _contract("conjugate", "float", "same float value, including infinities, signed zero and NaN")


@register_rule
class FloatIsInteger(Rule):
    names = ("float.is_integer",)
    SAMPLES = {"float.is_integer": "x = (-1.5).is_integer()\n"}
    TWINS = {"float.is_integer": "x = (-1.5).is_integer(1)\n"}
    CONTRACT = _contract("is_integer", "bool", "True iff finite and integral; False for NaN and infinities")


@register_rule
class FloatAsIntegerRatio(Rule):
    names = ("float.as_integer_ratio",)
    SAMPLES = {"float.as_integer_ratio": "x = (-1.5).as_integer_ratio()\n"}
    TWINS = {"float.as_integer_ratio": "x = (-1.5).as_integer_ratio(1)\n"}
    CONTRACT = _contract("as_integer_ratio", "tuple[int,int]", "exact reduced numerator/denominator; denominator positive")
