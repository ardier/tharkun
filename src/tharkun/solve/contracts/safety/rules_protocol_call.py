"""Narrow source-term cases for the generic call protocol."""

from __future__ import annotations

from tharkun.solve.contracts.safety import Binding, Case
from tharkun.solve.contracts.safety.rules_protocols import _source_operand
from tharkun.solve.terms import PY, Refused, Term, literal, typeis

_NONCALLABLE_SCALARS = ("int", "bool", "float", "str", "NoneType")
_PROPAGATION_GAP = (
    "Terms cannot express callable-target tp_call slots, class/metaclass call "
    "routing, or constructor callback evidence"
)


def _source_stable(term: Term) -> bool:
    return (term.op == "key" and term.sort == PY) or _source_operand(term)


def protocol_call_cases(qualname: str, exception: str, binding: Binding):
    if qualname != "protocol.call":
        return Refused("call template requested for a different rule")
    if exception != "TypeError":
        return Refused(_PROPAGATION_GAP)
    if binding.op != "Call" or binding.receiver is None:
        return Refused("call cases require a source-level Call binding")
    if not _source_stable(binding.receiver) or any(
        not _source_stable(arg) for arg in binding.args
    ):
        return Refused("call receiver and arguments must be source variables or exact literals")
    return tuple(
        Case(typeis(binding.receiver, tag), literal(True))
        for tag in _NONCALLABLE_SCALARS
    )


CASES_BUILDERS = {
    ("protocol.call", "TypeError"): protocol_call_cases,
    **{
        ("protocol.call", exception): protocol_call_cases
        for exception in (
            "AssertionError", "AttributeError", "FloatingPointError", "IndexError",
            "KeyError", "ZeroDivisionError",
        )
    },
}

__all__ = ["CASES_BUILDERS", "protocol_call_cases"]
