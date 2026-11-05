"""Facts about the hole as a preimage's PRIOR, and preimage results as rule-keyed HINTS ([internal ref] PR5).

A preimage candidate that contradicts a sound forward fact about its own operand cannot occur at that site, so it is
filtered before anything runs it.  :func:`prior_from_fact` states such a fact in the demand vocabulary
(``solve/preimage/want.py``):

* absint's value facts (``solve/values.py``): ``StrFact`` (a finite set of strings) and ``IntFact`` (an interval);
* the fact carrier (``solve/fact_env.py::VarFacts``): ``eq`` / ``neq`` / ``lo`` / ``hi`` / ``container_len`` /
  ``container_len_ne``;
* a drive's own parameter pins and bounds (``driver.py::drive_to_line``'s ``context_facts`` / ``param_bounds``,
  :func:`prior_from_pin`, :func:`prior_from_bounds`).

Every translation is a SUPERSET of the fact's value set (a fact the vocabulary cannot state becomes ``ANY``), so the
prior only ever removes candidates the fact excludes: a filter, never a claim.

:func:`rule_fact` is the record a consumer publishes for a preimage it used, in the shape docs/contracts/FORMAT.md
"Rules publish facts to the store" ruled: kind ``rule_fact`` (evidence, never certifiable), key
``<qualname>@<site>#preimage``, value ``{"kind", "claim", "data", "assumes"}`` -- a HINT; nothing certifies through it.
Its first writer is the drive's preimage tier, into the drive's own ``KnowledgeBase``; registering the kind in
``sv/contracts.py::KINDS`` waits for an SV component that writes it (that registry lists what ``tharkun/sv`` writes).  SOLVING by docs/LAYERS.md: stdlib and ``tharkun.solve`` only.
"""

from __future__ import annotations

from typing import Any

from tharkun.solve.preimage.want import ANY, All, Compare, Length, Not, OneOf, Want

__all__ = ["prior_from_fact", "prior_from_pin", "prior_from_bounds", "rule_fact"]


def prior_from_fact(fact: Any) -> Want:
    """A demand every value of ``fact`` satisfies (``ANY`` for a fact the vocabulary cannot state)."""
    from tharkun.solve.fact_env import VarFacts
    from tharkun.solve.values import IntFact, StrFact
    if isinstance(fact, StrFact):
        if fact.consts is None:
            return ANY
        return OneOf(tuple(sorted(fact.consts)))
    if isinstance(fact, IntFact):
        if fact.empty:
            return OneOf(())
        parts = ([Compare(">=", fact.lo)] if fact.lo is not None else []) + \
                ([Compare("<=", fact.hi)] if fact.hi is not None else [])
        return All.of(*parts)
    if isinstance(fact, VarFacts):
        parts: list[Want] = []
        if fact.eq is not None:
            parts.append(OneOf(tuple(fact.eq)))
        if fact.neq:
            parts.append(Not(OneOf(tuple(fact.neq))))
        if fact.lo is not None:
            parts.append(Compare(">=", fact.lo))
        if fact.hi is not None:
            parts.append(Compare("<=", fact.hi))
        if fact.container_len is not None:
            parts.append(Length(fact.container_len, fact.container_len))
        for n in sorted(fact.container_len_ne):
            parts.append(Not(Length(n, n)))
        return All.of(*parts)
    return ANY


def prior_from_pin(value: Any) -> Want:
    """A parameter the callers fix (``context_facts``): only that value."""
    return OneOf((value,))


def prior_from_bounds(bounds: tuple) -> Want:
    """An inclusive ``(lo, hi)`` range a drive keeps a parameter in (``param_bounds``)."""
    lo, hi = bounds
    return All.of(*([Compare(">=", lo)] if lo is not None else []), *([Compare("<=", hi)] if hi is not None else []))


def rule_fact(qualname: str, site: str, want: Want, candidate: Any, *, kind: str = "", technique: str = "preimage"
              ) -> tuple[str, dict]:
    """``(key, value)`` of the ``rule_fact`` hint for a preimage a consumer used at ``site`` (``file:line``)."""
    from tharkun.solve.contracts.rules import get_rule
    rule = get_rule(qualname)
    contract = rule.contract if rule is not None else None
    value = {
        "kind": kind or (rule.kind if rule is not None else ""),
        "claim": "preimage",
        "data": {"want": repr(want), "candidate": repr(candidate)[:256], "technique": technique},
        "assumes": [[qualname, bool(contract.tested) if contract is not None else False]],
    }
    return f"{qualname}@{site}#preimage", value
