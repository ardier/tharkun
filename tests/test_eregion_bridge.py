"""[internal ref] JOB 2 ([internal ref]): the producer -> fact-store bridge.

Owner ruling (2026-09-29): the bridge is an extension of the KnowledgeBase's
own Evidence/Derivation mechanism, not a separate translator.  This pins that
``tharkun.sv.eregion_bridge`` turns a producer's plain-dict input (the shapes
``eregion_codegen.objects.tharkun_facts`` builds on the Eregion side; not
imported here -- Tharkun does not depend on Eregion) into canonical Evidence,
keeps candidates and certified facts in different kinds, and that only
``publish_certified`` writes the certified kind.
"""
from __future__ import annotations

import pytest

from tharkun.knowledge import KnowledgeBase
from tharkun.solve.obligations import Invariant
from tharkun.sv.eregion_bridge import (
    CERTIFIED_INVARIANT,
    INVARIANT_CANDIDATE,
    invariant_key,
    publish_certified,
    publish_fact,
    publish_invariant_candidate,
)

_FACT = {
    "kind": "invariant_candidate",
    "site": {"file": "t.py", "line": 6, "column": 5, "function": "<module>"},
    "value": "d != 0",
    "evidence": {"engine": "eregion", "solutions": 0},
}


def test_publishes_a_candidate_as_canonical_evidence() -> None:
    kb = KnowledgeBase()
    publish_invariant_candidate(kb, _FACT, producer="eregion")

    items = kb.evidence(kind=INVARIANT_CANDIDATE)
    assert len(items) == 1
    inv = items[0].value["invariant"]
    assert inv == Invariant("t.py", 6, 5, "<module>", "d != 0")
    assert items[0].key == invariant_key(inv)
    assert items[0].value["evidence"] == {"engine": "eregion", "solutions": 0}
    assert items[0].derivations[0].producer == "eregion"
    # a candidate is a proposal: nothing certified exists yet
    assert kb.evidence(kind=CERTIFIED_INVARIANT) == ()


def test_the_same_claim_twice_is_one_item_with_two_derivations() -> None:
    kb = KnowledgeBase()
    publish_invariant_candidate(kb, _FACT, producer="eregion.a")
    publish_invariant_candidate(kb, _FACT, producer="eregion.b")

    (item,) = kb.evidence(kind=INVARIANT_CANDIDATE)
    assert {d.producer for d in item.derivations} == {"eregion.a", "eregion.b"}


def test_two_claims_at_one_statement_do_not_collide() -> None:
    kb = KnowledgeBase()
    other = dict(_FACT, value="d > 0")
    publish_invariant_candidate(kb, _FACT, producer="eregion")
    publish_invariant_candidate(kb, other, producer="eregion")

    assert {i.value["invariant"].value for i in kb.evidence(kind=INVARIANT_CANDIDATE)} == {"d != 0", "d > 0"}


@pytest.mark.parametrize("broken", [
    {"kind": "invariant_candidate", "value": "d != 0"},                                  # no site
    dict(_FACT, site={"file": "t.py", "line": 0, "column": 5, "function": "<module>"}),  # no such line
    dict(_FACT, value="__import__('os')"),                                               # not a predicate
    dict(_FACT, type="not_a_type"),
])
def test_a_malformed_fact_fails_loudly_instead_of_publishing_half_a_candidate(broken) -> None:
    kb = KnowledgeBase()
    with pytest.raises((KeyError, ValueError)):
        publish_fact(kb, broken, producer="eregion")
    assert kb.evidence() == ()


def test_publish_fact_dispatches_on_kind_and_rejects_an_unknown_one() -> None:
    kb = KnowledgeBase()
    publish_fact(kb, _FACT, producer="eregion")
    assert len(kb.evidence(kind=INVARIANT_CANDIDATE)) == 1
    with pytest.raises(ValueError):
        publish_fact(kb, {"kind": "family_intersection_candidate"}, producer="eregion")  # ruled, not built
    with pytest.raises(ValueError):
        publish_fact(kb, {"kind": "certified_invariant"}, producer="eregion")  # a producer cannot certify


def test_certified_is_its_own_kind_derived_from_the_candidate() -> None:
    kb = KnowledgeBase()
    publish_invariant_candidate(kb, _FACT, producer="eregion")
    inv = kb.evidence(kind=INVARIANT_CANDIDATE)[0].value["invariant"]

    publish_certified(kb, inv, checker="sv.validate.check_candidates_individually", producer="fact_proof")

    (item,) = kb.evidence(kind=CERTIFIED_INVARIANT)
    assert item.value["invariant"] == inv
    assert item.derivations[0].producer == "fact_proof"
    assert item.derivations[0].parents == ((INVARIANT_CANDIDATE, invariant_key(inv)),)
