"""CPython-tested safety cases for selected builtin functions."""

from __future__ import annotations

from tharkun.solve.terms import PY, Refused, and_, eq, length, literal, or_, typeis
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


def int_cases(qualname: str, exception: str, binding: Binding):
    """Build exact-builtin cases for one-argument int() ValueError."""
    if qualname != "int" or exception != "ValueError":
        return Refused("builtin int cases cover ValueError only")
    receiver = binding.receiver
    if (receiver is None or receiver.op != "var" or receiver.sort != PY
            or binding.args or binding.op is not None):
        return Refused("int binding requires one source receiver, no args, and no operator")
    return (
        Case(typeis(receiver, "int"), None),
        Case(typeis(receiver, "bool"), None),
        Case(and_(typeis(receiver, "str"), eq(length(receiver), literal(0))), literal(True)),
        Case(and_(typeis(receiver, "bytes"), eq(length(receiver), literal(0))), literal(True)),
    )


def classcheck_cases(qualname: str, exception: str, binding: Binding):
    """Invalid exact-builtin operands of isinstance/issubclass; no user hooks or classinfo tuples."""
    if qualname not in ("isinstance", "issubclass") or exception != "TypeError":
        return Refused("class checks cover TypeError only")
    if (binding.receiver is None or len(binding.args) != 1 or binding.op is not None
            or any(not _source_operand(value) for value in (binding.receiver, *binding.args))):
        return Refused("class check binding requires two source operands and no operator")
    # These exact types have neither class-check hooks nor a classinfo interpretation.
    tags = ("int", "bool", "float", "str")
    domain = and_(*(or_(*(typeis(value, tag) for tag in tags))
                    for value in (binding.receiver, *binding.args)))
    return (Case(domain, literal(True)),)


CASES_BUILDERS = {
    ("len", "TypeError"): len_cases,
    ("int", "ValueError"): int_cases,
    ("isinstance", "TypeError"): classcheck_cases,
    ("issubclass", "TypeError"): classcheck_cases,
}
