"""The call-shape twin of every contract, against every engine that can say TRUE ([internal ref]).

A twin is the sample with the contracted call given the wrong number of arguments (none, else one too many),
whichever CPython rejects with TypeError.  Whatever an engine knows about a contract, a call CPython rejects is a
violation of the dynamic-typing property: TRUE on a twin is a wrong TRUE (a model that checked the arguments'
kinds but not their count, or trusted a contract for a call shape it was never verified for).
"""

from __future__ import annotations

import pytest

from tests.contracts import _engine_audit as A
from tests.contracts._engine_audit import SAMPLES
from tests.contracts._native import peer_runtime_check_type, run_native
from tharkun.solve.absint import prove
from tharkun.solve.census import site_census

DT = ("TypeError", "AttributeError")
SKIPPED = {"open", "print"}  # print() with no argument is valid; open('/dev/null') shapes vary
TWINS = {q: A.arity_twin(SAMPLES[q], q) for q in sorted(SAMPLES) if q not in SKIPPED and not q.startswith("random.")}
TWINS = {q: t for q, t in TWINS.items() if t is not None}


def test_most_contracts_have_a_call_shape_twin():
    assert len(TWINS) > 120, len(TWINS)


@pytest.mark.parametrize("qualname", sorted(TWINS))
def test_the_twin_is_a_typeerror_on_cpython(qualname):
    oracle = peer_runtime_check_type() if qualname == "_sv_verifier.check_type" else None
    assert run_native(TWINS[qualname], check_type_oracle=oracle) == "TypeError"


@pytest.mark.parametrize("qualname", sorted(TWINS))
def test_absint_never_proves_a_twin(qualname):
    r = prove(TWINS[qualname], DT, time_budget=30.0)
    assert r.verdict != "TRUE", (qualname, TWINS[qualname], r.explain())


@pytest.mark.parametrize("qualname", sorted(TWINS))
def test_the_census_never_calls_a_twin_free_of_sites(qualname):
    assert not site_census(TWINS[qualname], DT, "<twin>").no_site
