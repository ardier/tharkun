"""The fact a ``**`` records for its ``ZeroDivisionError`` check names the BASE and the EXPONENT ([internal ref] PR 3, ``bm_nbody_nondet``).

``0.0 ** negative`` is the ``ZeroDivisionError`` of ``**``: it needs the base to be able to be ``0`` and the exponent to be negative.
The site fact used to print the exponent as a divisor (``divisor in [-1.5, -1.5] may be 0`` for ``(dx*dx + dy*dy + dz*dz) ** (-1.5)``),
which reads as a division by a constant that can be zero.  The check itself was right and is unchanged; so is every verdict.

``bm_nbody_nondet`` stays UNKNOWN for ``arithmetic``, and both unproven checks are real: ``advance`` computes
``(dx*dx + dy*dy + dz*dz) ** -1.5`` and ``report_energy`` divides by ``(dx*dx + dy*dy + dz*dz) ** 0.5``, and two bodies at the same
point make that sum ``0.0`` (a ``ZeroDivisionError`` natively, see the controls below).  Nothing in the program says the bodies never
meet, so no sound analysis proves it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tharkun.solve.absint import PROPERTY_CLASSES, prove

NBODY = Path(__file__).parent / "fixtures" / "absint" / "targets" / "bm_nbody_nondet.py"
HEAD = "import _sv_verifier as sv\n"


def divide_facts(src: str):
    r = prove(src, ["ZeroDivisionError"], filename="<program>", time_budget=20.0)
    return r, [(line, fact) for line, _col, op, _exc, fact in r.sites if op == "divide"]


def test_a_pow_that_may_raise_names_its_base_and_its_exponent():
    r, facts = divide_facts(HEAD + "x = sv.nondet_float()\ny = (x * x) ** -1.5\n")
    assert r.verdict == "UNKNOWN"
    (line, fact), = facts
    assert fact.startswith("UNPROVEN: base in [") and "exponent in [-1.5, -1.5]" in fact and "the base may be 0 and the exponent negative" in fact
    assert "divisor" not in fact


def test_a_pow_that_cannot_raise_says_why():
    r, facts = divide_facts(HEAD + "x = sv.nondet_float()\ny = (x * x + 1.0) ** -1.5\n")
    assert r.verdict == "TRUE"
    (line, fact), = facts
    assert "base in [1, inf]" in fact and "0 ** negative is excluded" in fact


def test_a_nonnegative_exponent_cannot_raise():
    r, facts = divide_facts(HEAD + "x = sv.nondet_float()\ny = (x * x) ** 0.5\n")
    assert r.verdict == "TRUE"
    (line, fact), = facts
    assert "exponent in [0.5, 0.5]" in fact and "0 ** negative is excluded" in fact


def test_division_keeps_its_divisor_fact():
    r, facts = divide_facts(HEAD + "x = sv.nondet_float()\ny = 1.0 / x\n")
    (line, fact), = facts
    assert fact.startswith("UNPROVEN: divisor in [") and "may be 0" in fact


def test_the_controls_are_native_zero_division_errors():
    with pytest.raises(ZeroDivisionError):
        0.0 ** -1.5
    with pytest.raises(ZeroDivisionError):
        1.0 / (0.0 ** 0.5)
    assert (1.0 + 0.0) ** -1.5 == 1.0


def test_bm_nbody_stays_unknown_and_its_two_unproven_checks_are_the_coincident_bodies():
    src = NBODY.read_text()
    r = prove(src, PROPERTY_CLASSES["arithmetic"], filename="bm_nbody_nondet.py", time_budget=60.0)
    assert r.verdict == "UNKNOWN"
    unproven = {(line, op): fact for line, _col, op, _exc, fact in r.sites if fact.startswith("UNPROVEN")}
    assert sorted(unproven) == [(92, "divide"), (113, "divide")], unproven
    assert "the base may be 0 and the exponent negative" in unproven[(92, "divide")] and "exponent in [-1.5, -1.5]" in unproven[(92, "divide")]
    assert "divisor in [0, inf] may be 0" in unproven[(113, "divide")]
    lines = src.splitlines()
    assert "** (-1.5)" in lines[91] and "** 0.5" in lines[112]
