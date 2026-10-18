"""absint reads the contracts of the ``math`` constants ([internal ref] step 2): ``pi``, ``e``, ``tau``, ``inf``, ``nan``.

``MODULES["math"]["consts"]`` held a hand value for each (``inf`` and ``nan`` the any-float).  ``Analyzer.module_attr`` now asks the rule
(``visit_rule("math.<name>")`` -> ``AbsintVisitor.visit_math_constant``), whose value is the contract's own
(``adapters/math_model.py::constant``: CPython's float, while the contract is TESTED); withdrawn or untested, the read is a named refusal.

Pinned against CPython 3.14.4: every program of the table below is TRUE for ``AssertionError`` exactly when CPython runs it clean, in
both directions where absint has the precision (the exact value), and never TRUE where CPython raises; withdrawal; a NEGATIVE CONTROL: a
wrong value in the table (``pi == 3.0``) proves the program that raises natively, which the never-TRUE direction forbids.
"""

from __future__ import annotations

import pytest

from tests.contracts._native import run_native
from tharkun.solve.absint import prove
from tharkun.solve.contracts.adapters import math_model
from tharkun.solve.contracts.registry import REGISTRY, load_all

CONSTANTS = ("pi", "e", "tau", "inf", "nan")
#: (program, does it run clean on CPython)
PROGRAMS = [
    ("import math\nassert math.pi > 3.14 and math.pi < 3.15\n", True),
    ("import math\nassert math.pi < 3.14\n", False),
    ("import math\nassert math.e > 2.71 and math.e < 2.72\n", True),
    ("import math\nassert math.e > 2.72\n", False),
    ("import math\nassert math.tau > 6.28 and math.tau < 6.29\n", True),
    ("import math\nassert math.tau < 6.28\n", False),
    ("import math\nassert math.inf > 1e308\n", True),
    ("import math\nassert math.inf < 1e308\n", False),
    ("import math\nassert -math.inf < -1e308\n", True),
    ("import math\nx = math.nan\nassert x != x\n", True),
    ("import math\nx = math.nan\nassert x == x\n", False),
    ("import math\nx = math.pi\nassert x == 3.141592653589793\n", True),
    ("import math\nx = math.pi\nassert x == 3.0\n", False),
]
IDS = [f"{i}" for i in range(len(PROGRAMS))]


@pytest.mark.parametrize("src,clean", PROGRAMS, ids=IDS)
def test_the_table_is_native(src, clean):
    assert (run_native(src) == "") is clean, src


@pytest.mark.parametrize("src,clean", PROGRAMS, ids=IDS)
def test_never_true_where_cpython_raises(src, clean):
    verdict = prove(src, ("AssertionError",), time_budget=30.0).verdict
    if not clean:
        assert verdict != "TRUE", src


@pytest.mark.parametrize("src,clean", [p for p in PROGRAMS if p[1]], ids=[i for p, i in zip(PROGRAMS, IDS) if p[1]])
def test_true_where_cpython_runs_clean_and_the_value_is_exact(src, clean):
    r = prove(src, ("AssertionError",), time_budget=30.0)
    assert r.verdict == "TRUE", (src, r.reason)


@pytest.mark.parametrize("name", CONSTANTS)
def test_the_value_is_cpythons_own(name):
    import math

    value = math_model.constant(name)
    expected = getattr(math, name)
    assert (value != value and expected != expected) or value == expected


@pytest.mark.parametrize("name", CONSTANTS)
def test_withdrawing_the_contract_is_a_named_refusal_not_the_table_of_old(name, monkeypatch):
    load_all()
    src = f"import math\nx = math.{name}\n"
    assert prove(src, ("AssertionError",), time_budget=30.0).verdict == "TRUE"
    monkeypatch.delitem(REGISTRY, f"math.{name}")
    r = prove(src, ("AssertionError",), time_budget=30.0)
    assert r.verdict == "UNKNOWN" and f"math.{name}" in r.reason and "no tested contract" in r.reason, (name, r.reason)


def test_negative_control_a_wrong_value_in_the_table_proves_a_program_that_raises(monkeypatch):
    """The control: with ``pi == 3.0`` the table proves ``assert math.pi < 3.1``, which CPython raises: the never-TRUE test fails
    for such a table."""
    monkeypatch.setitem(math_model.CONSTANTS, "pi", 3.0)
    src = "import math\nassert math.pi < 3.1\n"
    assert run_native(src) == "AssertionError"
    assert prove(src, ("AssertionError",), time_budget=30.0).verdict == "TRUE"
