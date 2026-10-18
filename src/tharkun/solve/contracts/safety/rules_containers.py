"""CPython-tested exact-builtin sequence subscription cases."""

from __future__ import annotations

from tharkun.solve.terms import (
    BOOL, F64, INT, PY, Refused, Term, and_, as_bool, as_int, as_seq, b2i,
    eq, ge, length, literal, lt, neg, or_, typeis,
)
from tharkun.solve.contracts.safety import Binding, Case
from tharkun.solve.contracts.safety.rules_protocols import _source_operand


_KEY_TAGS = ("int", "bool", "float", "str", "NoneType")
_SEQUENCES = {
    "list.__getitem__": "list",
    "tuple.__getitem__": "tuple",
    "str.__getitem__": "str",
}


def _index_value(key: Term, key_tag: str) -> Term | None:
    literal_tag = {INT: "int", BOOL: "bool", F64: "float"}.get(key.sort)
    if key.op == "lit" and literal_tag != key_tag:
        return None
    if key_tag == "int":
        return key if key.op == "lit" else as_int(key)
    if key_tag == "bool":
        return b2i(key) if key.op == "lit" else b2i(as_bool(key))
    return None


def sequence_getitem_cases(qualname: str, exception: str, binding: Binding):
    """Build full scalar-key partitions for exact list/tuple/str subscriptions."""
    kind = _SEQUENCES.get(qualname)
    if kind is None or exception not in {"IndexError", "TypeError"}:
        return Refused("sequence subscription cases cover IndexError and TypeError only")
    if (binding.receiver is None or binding.receiver.op != "var"
            or binding.receiver.sort != PY or len(binding.args) != 1
            or binding.op is not None or not _source_operand(binding.args[0])):
        return Refused("subscription binding requires a source receiver, one stable key, and no operator")

    receiver, key = binding.receiver, binding.args[0]
    receiver_domain = typeis(receiver, kind)
    if isinstance(receiver_domain, Refused):
        return receiver_domain
    if kind == "str":
        size = length(receiver)
    else:
        size = length(as_seq(receiver, kind))

    result = []
    for key_tag in _KEY_TAGS:
        domain = and_(receiver_domain, typeis(key, key_tag))
        if exception == "TypeError":
            raises = literal(True) if key_tag not in {"int", "bool"} else None
        elif key_tag in {"int", "bool"}:
            index = _index_value(key, key_tag)
            if index is None:
                raises = None
            else:
                raises = or_(lt(index, neg(size)), ge(index, size))
        else:
            raises = None
        result.append(Case(domain, raises))
    return tuple(result)


def list_pop_cases(qualname: str, exception: str, binding: Binding):
    """Build the exact-list, zero-argument pop exception case."""
    if qualname != "list.pop" or exception not in {"IndexError", "TypeError"}:
        return Refused("list.pop cases cover IndexError and TypeError only")
    receiver = binding.receiver
    if (receiver is None or receiver.op != "var" or receiver.sort != PY
            or binding.args or binding.op is not None):
        return Refused("list.pop binding requires one source receiver, no args, and no operator")
    domain = typeis(receiver, "list")
    if isinstance(domain, Refused):
        return domain
    if exception == "TypeError":
        raises = None
    else:
        empty = eq(length(as_seq(receiver, "list")), literal(0))
        if isinstance(empty, Refused):
            return empty
        raises = empty
    return (Case(domain, raises),)


CASES_BUILDERS = {
    (qualname, exception): sequence_getitem_cases
    for qualname in _SEQUENCES
    for exception in ("IndexError", "TypeError")
}
CASES_BUILDERS.update({
    ("list.pop", exception): list_pop_cases
    for exception in ("IndexError", "TypeError")
})
