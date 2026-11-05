"""Range search descriptors: native integer fast paths and equality callback boundaries."""
from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec


def _contract(method, shape):
    return Contract(
        qualname="range." + method,
        raises=dunder_propagation_claims(
            "needle's __eq__ and its result's __bool__ on the non-exact-int/bool path (class and args propagate unchanged)",
            native={"TypeError": (
                "wrong receiver, arity or keywords, or equality callback raises TypeError; "
                f"CPython 3.14.4 args ('range.{method}() takes exactly one argument (0 given)',) for missing needle; "
                f"('range.{method}() takes no keyword arguments',) for keywords; "
                f"(\"descriptor '{method}' for 'range' objects doesn't apply to a 'list' object\",) for list receiver; "
                f"('unbound method range.{method}() needs an argument',) for missing receiver"
            )}, tested_native=frozenset({"TypeError"})),
        may_raise_other=True,
        returns=ReturnSpec("int", shape, tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=("https://docs.python.org/3.14/library/stdtypes.html#range",
                "CPython Objects/rangeobject.c::range_" + method),
        notes=("Exact int/bool needles use arithmetic without equality callbacks. Other needles use iteration and "
               "may call user equality; count may then exceed one. Missing index: ValueError args "
               "('range.index(x): x not in range',) on the exact int/bool path, "
               "('sequence.index(x): x not in sequence',) on the general comparison path."),
        tested=True,
    )


@register_rule
class RangeCount(Rule):
    names = ("range.count",)
    SAMPLES = {"range.count": "r=range(5)\nx=r.count(2)\n"}
    TWINS = {"range.count": "r=range(5)\nx=r.count()\n"}
    CONTRACT = _contract("count", "0 <= result <= range length; result in {0,1} for exact int/bool needles")


@register_rule
class RangeIndex(Rule):
    names = ("range.index",)
    SAMPLES = {"range.index": "r=range(5)\nx=r.index(2)\n"}
    TWINS = {"range.index": "r=range(5)\nx=r.index()\n"}
    CONTRACT = _contract("index", "0 <= result < range length on successful paths")
