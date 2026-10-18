"""Explicit M1-term refusal for callback-bearing collections constructors."""

from __future__ import annotations

from tharkun.solve.contracts.safety import Binding
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES
from tharkun.solve.terms import Refused


def defaultdict_cases(qualname: str, exception: str, binding: Binding):
    """Refuse until terms can name the builtin factory class and initializer hooks."""
    if qualname != "collections.defaultdict":
        return Refused("defaultdict safety case received the wrong qualname")
    if not isinstance(binding, Binding):
        return Refused("defaultdict safety case requires a typed Binding")
    return Refused(
        f"collections.defaultdict {exception}: M1 terms cannot represent the exact builtin "
        "default_factory class and initializer callback domain"
    )


CASES_BUILDERS = {
    ("collections.defaultdict", exception): defaultdict_cases
    for exception in ALL_PROPERTY_CLASSES
}


__all__ = ["CASES_BUILDERS", "defaultdict_cases"]
