"""Typed, TESTED contract safety cases and their public projections."""

from __future__ import annotations

from dataclasses import dataclass

from tharkun.solve.terms import BOOL, Refused, Term, and_, from_canon, literal, not_, or_
from tharkun.solve.contracts.registry import get


@dataclass(frozen=True)
class Binding:
    """Source-level operands for one original operation node.

    receiver is the left operand for a binary operator and the value for a
    unary operator. args is the right operand for binary operators and empty
    for unary operators. op is the AST operator class name (for example Add,
    Lt, or UAdd).
    """

    receiver: Term | None
    args: tuple[Term, ...]
    op: str | None

    def __post_init__(self) -> None:
        if self.receiver is not None and not isinstance(self.receiver, Term):
            raise TypeError("Binding.receiver must be a Term or None")
        if type(self.args) is not tuple or any(not isinstance(arg, Term) for arg in self.args):
            raise TypeError("Binding.args must be a tuple of Terms")
        if self.op is not None and type(self.op) is not str:
            raise TypeError("Binding.op must be a string or None")


@dataclass(frozen=True)
class Case:
    """One disjoint domain and its in-domain raise condition, if any."""

    domain: Term
    raises: Term | None

    def __post_init__(self) -> None:
        if not isinstance(self.domain, Term) or self.domain.sort != BOOL:
            raise TypeError("Case.domain must be a Bool Term")
        if self.raises is not None and (
            not isinstance(self.raises, Term) or self.raises.sort != BOOL
        ):
            raise TypeError("Case.raises must be a Bool Term or None")


def binding_from_record(record_binding: dict) -> Binding | Refused:
    """Decode the record-schema binding through the single canonical decoder."""
    if type(record_binding) is not dict or set(record_binding) != {"receiver", "args", "op"}:
        return Refused("record binding must contain exactly receiver, args, and op")
    receiver_text = record_binding["receiver"]
    args_text = record_binding["args"]
    op = record_binding["op"]
    if receiver_text is not None and type(receiver_text) is not str:
        return Refused("record receiver must be canonical text or null")
    if type(args_text) is not list or any(type(text) is not str for text in args_text):
        return Refused("record args must be a list of canonical texts")
    if op is not None and type(op) is not str:
        return Refused("record op must be text or null")
    receiver = from_canon(receiver_text) if receiver_text is not None else None
    if isinstance(receiver, Refused):
        return receiver
    args = tuple(from_canon(text) for text in args_text)
    refusal = next((value for value in args if isinstance(value, Refused)), None)
    if refusal is not None:
        return refusal
    try:
        return Binding(receiver, args, op)
    except (TypeError, ValueError) as exc:
        return Refused(f"invalid record binding: {exc}")


def _case_builders():
    from tharkun.solve.contracts.safety.heapq import CASES_BUILDERS as HEAPQ_BUILDERS
    from tharkun.solve.contracts.safety.rules_collections import CASES_BUILDERS as COLLECTIONS_BUILDERS
    from tharkun.solve.contracts.safety.rules_containers import CASES_BUILDERS as CONTAINER_BUILDERS
    from tharkun.solve.contracts.safety.rules_builtins import CASES_BUILDERS as BUILTIN_BUILDERS
    from tharkun.solve.contracts.safety.rules_protocols import CASES_BUILDERS as PROTOCOL_BUILDERS
    from tharkun.solve.contracts.safety.rules_protocol_call import CASES_BUILDERS as CALL_BUILDERS
    from tharkun.solve.contracts.safety.rules_verifier import CASES_BUILDERS as VERIFIER_BUILDERS
    builder_modules = (
        PROTOCOL_BUILDERS,
        HEAPQ_BUILDERS,
        COLLECTIONS_BUILDERS,
        CALL_BUILDERS,
        CONTAINER_BUILDERS,
        BUILTIN_BUILDERS,
        VERIFIER_BUILDERS,
    )
    merged = {}
    for builders in builder_modules:
        merged.update(builders)
    return merged


def _checked_claim(qualname: str, exception: str):
    contract = get(qualname)
    if contract is None:
        return None
    try:
        claim = contract.raise_claim(exception)
    except KeyError:
        return None
    return claim if claim.tested and claim.condition != "unknown" else None


def cases(qualname: str, exception: str, binding: Binding):
    """Return the template's fixed-order, pairwise-disjoint semantic cases."""
    if not isinstance(binding, Binding):
        return Refused("safety binding has the wrong type")
    claim = _checked_claim(qualname, exception)
    if claim is None:
        return Refused("missing, untested, or non-specific raise claim")
    builder = _case_builders().get((qualname, exception))
    if builder is None:
        return Refused("no semantic cases for this qualname and exception")
    result = builder(qualname, exception, binding)
    if isinstance(result, Refused) or claim.condition != "never":
        return result
    return tuple(Case(item.domain, None) for item in result)


def _validated_cases(qualname: str, exception: str, binding: Binding):
    result = cases(qualname, exception, binding)
    if isinstance(result, Refused):
        return result
    if not isinstance(result, tuple) or not result or any(not isinstance(item, Case) for item in result):
        return Refused("template returned an invalid or empty case set")
    return result


def case_safety(case: Case) -> Term | Refused:
    """Return the safe region for one case without weakening its domain."""
    if not isinstance(case, Case):
        return Refused("safety case has the wrong type")
    return case.domain if case.raises is None else and_(case.domain, not_(case.raises))


def admission(qualname: str, exception: str, binding: Binding) -> Term | Refused:
    """Return the ordered union of the cases' domains."""
    result = _validated_cases(qualname, exception, binding)
    if isinstance(result, Refused):
        return result
    return or_(*(item.domain for item in result))


def raise_condition(qualname: str, exception: str, binding: Binding) -> Term | Refused:
    """Return the ordered union of raise conditions within their case domains."""
    result = _validated_cases(qualname, exception, binding)
    if isinstance(result, Refused):
        return result
    raising = [and_(item.domain, item.raises) for item in result if item.raises is not None]
    return or_(*raising) if raising else literal(False)


def safety(qualname: str, exception: str, binding: Binding) -> Term | Refused:
    """Return the union of case-specific safe regions; never a weakened term."""
    result = _validated_cases(qualname, exception, binding)
    if isinstance(result, Refused):
        return result
    safe_cases = [case_safety(item) for item in result]
    refusal = next((item for item in safe_cases if isinstance(item, Refused)), None)
    return refusal if refusal is not None else or_(*safe_cases)


__all__ = [
    "Binding", "Case", "admission", "binding_from_record", "case_safety", "cases",
    "raise_condition", "safety",
]
