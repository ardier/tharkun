"""CPython-tested safety cases for selected builtin functions."""

from __future__ import annotations

from tharkun.solve.terms import Refused, literal, typeis
from tharkun.solve.contracts.safety import Binding, Case
from tharkun.solve.contracts.safety.rules_protocols import _source_operand


_LEN_TAGS = (
    "int", "bool", "float", "str", "NoneType", "list", "tuple", "bytes",
    "dict", "set", "frozenset",
)
_LEN_TYPEERROR_TAGS = frozenset({"int", "bool", "float", "NoneType"})


def len_cases(qualname: str, exception: str, binding: Binding):
    """Partition len's tested TypeError claim by exact builtin receiver type."""
    if qualname != "len" or exception != "TypeError":
        return Refused("builtin len cases cover TypeError only")
    receiver = binding.receiver
    if (receiver is None or binding.args or binding.op is not None
            or not _source_operand(receiver)):
        return Refused("len binding requires one source value, no args, and no operator")
    result = []
    for tag in _LEN_TAGS:
        domain = typeis(receiver, tag)
        if isinstance(domain, Refused):
            return domain
        raises = True if tag in _LEN_TYPEERROR_TAGS else None
        result.append(Case(domain, literal(raises) if raises is not None else None))
    return tuple(result)


CASES_BUILDERS = {("len", "TypeError"): len_cases}
