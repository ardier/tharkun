"""Static, typed exception conditions projected from the existing contract cases.

No English claim is interpreted as a predicate. A condition is exact only inside
its complete case domain; outside that domain the caller still owes coverage.
"""
from __future__ import annotations

from tharkun.solve.contracts.registry import get
from tharkun.solve.contracts.safety import Binding, admission, cases
from tharkun.solve.terms import PY, Refused, Term, and_, canon, literal, not_, to_python, var


EXCLUDED_EXCEPTIONS = ("KeyboardInterrupt", "MemoryError", "RecursionError")
_PREDICATE_BUILTINS = ("type", "int", "bool", "float", "len", "list", "tuple", "str", "bytes", "dict")


def _operand_types(domain: Term, names: tuple[str, ...]) -> dict[str, str | None]:
    """Read positive exact-type assumptions; retain all other guards in domain."""
    result = dict.fromkeys(names)
    pending = [domain]
    while pending:
        term = pending.pop()
        if term.op == "and":
            pending.extend(term.args)
        elif term.op == "typeis" and term.args[0].op == "var":
            name, tag = term.args[0].args[0], term.args[1].args[0]
            if name in result:
                result[name] = tag
    return result


def unknown_case(qualname: str | None, operands: tuple[str, ...], reason: str,
                 *, domain: Term | None = None, excluded_classes=()) -> dict:
    """Unknown coverage is conditional data, not a weakened raise predicate."""
    domain = literal(True) if domain is None else domain
    return {"contract": qualname, "domain": canon(domain), "raises": None,
            "condition": None, "operand_types": _operand_types(domain, operands),
            "predicate": None, "predicate_kind": "unavailable",
            "projection_refusal": reason, "unknown": True,
            "excluded_classes": list(excluded_classes)}


def condition_table(qualname: str, operands: tuple[str, ...], op: str | None = None) -> dict:
    """Return JSON-safe cases over named operation operands, without solving them.

    The first operand occupies Binding.receiver, as in the existing safety API;
    subsequent operands occupy Binding.args. An unavailable machine condition is
    an explicit gap. False-raising cases are retained: their type/domain guards
    matter to consumers deciding which fixed-builtin case applies.
    """
    if (type(operands) is not tuple or not operands or len(set(operands)) != len(operands)
            or any(type(name) is not str or not name.isidentifier() for name in operands)):
        raise ValueError("operands must be distinct identifier names in a nonempty tuple")
    values = tuple(var(name, PY) for name in operands)
    refusal = next((value for value in values if isinstance(value, Refused)), None)
    if refusal is not None:
        raise ValueError(refusal.reason)
    binding = Binding(values[0], values[1:], op)
    contract = get(qualname)
    result = {"contract": qualname, "conditions": {}, "gaps": [], "open_world": True}
    if contract is None:
        result["gaps"].append({"contract": qualname, "class": "*", "reason": "missing contract"})
        result["conditions"]["*"] = [unknown_case(qualname, operands, "missing contract")]
        return result
    result["open_world"] = contract.may_raise_other
    claims = {claim.exc: claim for claim in (*contract.raises, *contract.other_raises)
              if claim.exc not in EXCLUDED_EXCEPTIONS}
    for exception, claim in claims.items():
        if claim.condition == "never" and claim.tested:
            continue
        found = cases(qualname, exception, binding)
        if isinstance(found, Refused):
            result["gaps"].append({"contract": qualname, "class": exception, "reason": found.reason})
            result["conditions"][exception] = [unknown_case(qualname, operands, found.reason)]
            continue
        rows = []
        for case in found:
            raises = literal(False) if case.raises is None else case.raises
            condition = and_(case.domain, raises)
            predicate = to_python(condition, builtin_names=_PREDICATE_BUILTINS)
            reason = predicate.reason if isinstance(predicate, Refused) else None
            rows.append({"contract": qualname, "domain": canon(case.domain),
                         "raises": canon(raises), "condition": canon(condition),
                         "operand_types": _operand_types(case.domain, operands),
                         "predicate": None if reason is not None else predicate,
                         "predicate_kind": "unavailable" if reason is not None else "exact",
                         "projection_refusal": reason})
            if reason is not None:
                result["gaps"].append({"contract": qualname, "class": exception,
                                       "reason": reason})
        admitted = admission(qualname, exception, binding)
        reason = admitted.reason if isinstance(admitted, Refused) else "outside admitted typed/domain cases"
        outside = literal(True) if isinstance(admitted, Refused) else not_(admitted)
        rows.append(unknown_case(qualname, operands, reason, domain=outside))
        result["gaps"].append({"contract": qualname, "class": exception, "reason": reason,
                               "domain": canon(outside)})
        result["conditions"][exception] = rows
    return result
