"""Tharkun's engines over the protocol-swallowing catalog ([internal ref] batch 5, [internal ref]).

Each case of ``_swallowing_cases.py`` is a small program with a pinned native outcome
(``test_swallowing_native.py``).  Built so that a WRONG model gives a wrong TRUE -- the violation is
reachable only because CPython swallows, or only because it does not -- each is handed to the engines that
can say TRUE or that a TRUE-capable engine leans on:

* ``census`` (``proof:no-site``), ``absint`` (``proof:interval`` / ``proof:shape``), ``product``
  (``symbolic:product``) and ``stationary`` (``proof:stationary``) must NEVER say TRUE for a case whose
  native run violates the property;
* the concolic stage must never report every site closed ("no run fires it") for such a case: that stops
  the search early;
* the engine's own child run (``sv/_child.py``) must classify the run exactly as CPython does.

A refusal (or an absint "unproven check", which models the program and finds the site) is the sound,
incomplete answer.  The cases where an engine models the swallowing exactly enough to prove a CLEAN program
TRUE are pinned too, so a model that regresses to refusing, or worse, is noticed.
"""

from __future__ import annotations

import sys

import pytest

from tests.contracts import _swallowing as S
from tests.contracts._swallowing_cases import CASES

PY = sys.executable
VIOLATING = [c for c in CASES if c.violates]


def _id(c):
    return c.id


# the clean cases an engine proves TRUE (it models the swallowing, or the path, exactly)
ABSINT_PROVES = {
    "in_range_non_int_is_false", "lh_extend_attributeerror_propagates", "lh_extend_valueerror_propagates",
    "slot_class_variable_conflict", "lh_extend_conditional_typeerror_clean_after", "ix_huge_slice_bound_clamps",
    "assert_message_not_evaluated_when_true",
    # [internal ref]: a try statement is modelled; a clause naming a non-exception is a TypeError only when an exception
    # is matched (lazily, as CPython), so the clean lazy case is proven ...
    "except_non_exception_class_lazy",
    # ... and the explicit TypeError raise inside __len__ is an exception in flight that call_func hands to the
    # length probe instead of the caller (CPython's hint swallows it): no check is recorded for it any more,
    # and the clean program is proven exactly (before: an imprecise UNKNOWN from a failed "raise" check).
    "lh_typeerror_swallowed_dt",
    # [internal ref]: absint has a set heap kind; `{1} in {frozenset({1})}` converts the set key to a frozenset (no TypeError), which
    # tests/test_absint_sets.py::test_membership pins against CPython
    "set_contains_set_key_converted",
}
CENSUS_PROVES = {"it_dict_star_attributeerror_becomes_typeerror", "delattr_missing_is_attributeerror",
                 "it_generator_stopiteration_runtimeerror"}
STATIONARY_PROVES = {
    "lh_length_hint_negative", "lh_extend_conditional_typeerror_clean_after", "lh_attributeerror_propagates",
    "truth_len_negative_valueerror", "lh_len_negative_valueerror", "lh_extend_attributeerror_propagates",
    "lh_valueerror_propagates", "lh_len_too_big_overflowerror", "lh_extend_valueerror_propagates",
}


@pytest.mark.parametrize("case", VIOLATING, ids=_id)
class TestNeverTrueOnAViolatingProgram:
    def test_census(self, case):
        assert S.census(case.src, case.classes)[0] != S.TRUE

    def test_absint(self, case):
        outcome, reason = S.absint(case.src, case.classes)
        assert outcome != S.TRUE, (case.id, case.mechanism)

    def test_product(self, case):
        assert S.product(case.src, case.classes, PY)[0] != S.TRUE

    def test_stationary(self, case):
        assert S.stationary(case.src, case.classes, PY)[0] != S.TRUE

    def test_concolic_never_closes_a_site_a_run_can_fire(self, case):
        assert S.concolic(case.src, case.classes, PY, budget=15.0)[0] != S.TRUE


@pytest.mark.parametrize("case", CASES, ids=_id)
def test_the_engines_own_child_run_agrees_with_cpython(case):
    exc, status = S.child(case.src, case.classes, case.vector, PY)
    assert exc == case.raises, (case.mechanism, status)


@pytest.mark.parametrize("case_id", sorted(ABSINT_PROVES))
def test_absint_proves_these_clean_programs_exactly(case_id):
    case = next(c for c in CASES if c.id == case_id)
    assert not case.violates
    assert S.absint(case.src, case.classes)[0] == S.TRUE


@pytest.mark.parametrize("case_id", sorted(CENSUS_PROVES))
def test_the_census_proves_these_clean_programs(case_id):
    case = next(c for c in CASES if c.id == case_id)
    assert not case.violates
    assert S.census(case.src, case.classes)[0] == S.TRUE


@pytest.mark.parametrize("case_id", sorted(STATIONARY_PROVES))
def test_stationary_proves_these_clean_programs(case_id):
    case = next(c for c in CASES if c.id == case_id)
    assert not case.violates
    assert S.stationary(case.src, case.classes, PY)[0] == S.TRUE


def test_no_clean_case_is_proved_that_is_not_pinned():
    """Every TRUE an engine gives on a clean case is listed above (a new TRUE is good news, but it must be
    looked at: it may be a model that stopped being exact)."""
    proved = {c.id for c in CASES if not c.violates and S.absint(c.src, c.classes)[0] == S.TRUE}
    assert proved == ABSINT_PROVES
    proved = {c.id for c in CASES if not c.violates and S.census(c.src, c.classes)[0] == S.TRUE}
    assert proved == CENSUS_PROVES
