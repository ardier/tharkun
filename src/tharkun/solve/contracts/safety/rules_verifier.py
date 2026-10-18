"""Specific M1-term refusals for nondeterministic verifier containers."""

from __future__ import annotations

from tharkun.solve.contracts.safety import Binding
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES
from tharkun.solve.terms import Refused

_QUALNAMES = {"_sv_verifier.nondet_list", "_sv_verifier.nondet_dict"}
_CHECK_TYPE = "_sv_verifier.check_type"


def verifier_container_cases(qualname: str, exception: str, binding: Binding):
    if qualname not in _QUALNAMES:
        return Refused("verifier-container safety case received the wrong qualname")
    if not isinstance(binding, Binding):
        return Refused("verifier-container safety case requires a typed Binding")
    return Refused(
        f"{qualname} {exception}: M1 terms cannot represent a fresh nondeterministic "
        "container with callback-derived element/key/value summaries"
    )


def verifier_check_type_cases(qualname: str, exception: str, binding: Binding):
    if qualname != _CHECK_TYPE:
        return Refused("verifier check_type safety case received the wrong qualname")
    if not isinstance(binding, Binding):
        return Refused("verifier check_type safety case requires a typed Binding")
    return Refused(
        f"{qualname} {exception}: M1 terms cannot express exact builtin class identity "
        "for the hint argument"
    )


CASES_BUILDERS = {
    (qualname, exception): verifier_container_cases
    for qualname in _QUALNAMES
    for exception in ALL_PROPERTY_CLASSES
}
CASES_BUILDERS.update({
    (_CHECK_TYPE, exception): verifier_check_type_cases
    for exception in ALL_PROPERTY_CLASSES
})


__all__ = ["CASES_BUILDERS", "verifier_check_type_cases", "verifier_container_cases"]
