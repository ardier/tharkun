"""absint reads the contracts of the builtins it modelled by hand ([internal ref] step 2, PR "absint builtins").

``abs``, ``len``, ``range``, ``int``, ``float``, ``list``, ``print`` and ``isinstance`` had a transfer function
(``Analyzer.b_<name>`` / ``t_<name>``) that never read the contract of the callable (the audit's ``M`` cell).  They now run
through their rule: ``AbsintVisitor.visit_<kind>`` -> ``Analyzer._contract_builtin_call`` -> ``c_<name>`` -> the unchanged
model under ``Analyzer._modelled``, which records the SEVEN classes as the contract states them and upgrades a class only on an
argument the model made.

What this file pins, against CPython 3.14.4:

* every program of the table below: for each of the seven classes, absint is never TRUE for a class the program raises
  natively (the soundness direction), and is TRUE for every class of a clean program (the precision the hand model had);
* the call site records the contract's seven ops (``<name>:<class>``); where the call raises ``TypeError`` natively, the
  ``TypeError`` op is UNPROVEN and the other six are proven (no user hook can run: the classes the contract admits only through
  a dunder);
* the contract is what admits the model: withdrawn, the call is a named refusal, not the old hand model;
* NEGATIVE CONTROLS: a ``_modelled`` that proves ``TypeError`` whatever the model checked is wrong exactly where this table looks.
"""

from __future__ import annotations

import pytest

from tests.contracts._native import run_native
from tharkun.solve.absint import prove
from tharkun.solve.absint.interp import Analyzer
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES

ALL = tuple(ALL_PROPERTY_CLASSES)
HAND_MODELLED = ("abs", "len", "range", "int", "float", "list", "print", "isinstance")

#: (builtin, program, the class CPython raises ("" = clean))
CASES = [
    ("abs", "x = abs(-3)\n", ""),
    ("abs", "x = abs(2.5)\n", ""),
    ("abs", "x = abs('a')\n", "TypeError"),
    ("abs", "x = abs(None)\n", "TypeError"),
    ("abs", "x = abs([1])\n", "TypeError"),
    ("abs", "x = abs()\n", "TypeError"),
    ("len", "x = len([1, 2])\n", ""),
    ("len", "x = len('ab')\n", ""),
    ("len", "x = len({1: 2})\n", ""),
    ("len", "class A:\n    def __len__(self):\n        return 3\nx = len(A())\n", ""),
    ("len", "x = len(5)\n", "TypeError"),
    ("len", "x = len(None)\n", "TypeError"),
    ("len", "x = len()\n", "TypeError"),
    ("range", "r = range(3)\n", ""),
    ("range", "r = range(1, 8, 2)\n", ""),
    ("range", "r = range(1.5)\n", "TypeError"),
    ("range", "r = range('a')\n", "TypeError"),
    ("range", "r = range()\n", "TypeError"),
    ("range", "r = range(1, 2, 3, 4)\n", "TypeError"),
    ("int", "x = int('5')\n", ""),
    ("int", "x = int(2.7)\n", ""),
    ("int", "x = int()\n", ""),
    ("int", "x = int(None)\n", "TypeError"),
    ("int", "x = int([1])\n", "TypeError"),
    ("float", "x = float(3)\n", ""),
    ("float", "x = float('2.5')\n", ""),
    ("float", "x = float(None)\n", "TypeError"),
    ("float", "x = float(3, 0)\n", "TypeError"),
    ("list", "x = list(range(3))\n", ""),
    ("list", "x = list([1, 2])\n", ""),
    ("list", "x = list()\n", ""),
    ("list", "x = list(5)\n", "TypeError"),
    ("list", "x = list(1, 2)\n", "TypeError"),
    ("print", "print(1)\n", ""),
    ("print", "print(1, 'a', sep='-', end='')\n", ""),
    ("print", "print(1, sep=1)\n", "TypeError"),
    ("print", "print(1, end=2)\n", "TypeError"),
    ("isinstance", "x = isinstance(3, int)\n", ""),
    ("isinstance", "x = isinstance('a', (int, str))\n", ""),
    ("isinstance", "x = isinstance(3)\n", "TypeError"),
    ("isinstance", "x = isinstance(3, int, int)\n", "TypeError"),
]
IDS = [f"{name}-{i}" for i, (name, _s, _n) in enumerate(CASES)]


@pytest.mark.parametrize("name,src,native", CASES, ids=IDS)
def test_the_table_is_native(name, src, native):
    assert run_native(src) == native, (name, src)


@pytest.mark.parametrize("name,src,native", CASES, ids=IDS)
def test_never_true_for_a_class_cpython_raises_and_true_for_every_class_of_a_clean_program(name, src, native):
    for exc in ALL:
        verdict = prove(src, (exc,), time_budget=30.0).verdict
        if exc == native:
            assert verdict != "TRUE", (name, src, exc)
        elif native == "":
            assert verdict == "TRUE", (name, src, exc)


def _status(src: str, name: str) -> dict:
    """``{class: "proven" | "UNPROVEN"}`` of the ``<name>:<class>`` ops recorded at the call.  A call the model refuses
    (``list(5)``: "several possible iterables") records nothing and is an UNKNOWN: ``None`` then, which the callers accept
    only where the table says the call raises natively."""
    r = prove(src, ALL, time_budget=30.0)
    if r.verdict == "UNKNOWN" and r.reason.startswith("unsupported:"):
        return None
    return {s[3]: ("UNPROVEN" if s[4].startswith("UNPROVEN") else "proven") for s in r.sites if s[2] == f"{name}:{s[3]}"}


@pytest.mark.parametrize("name,src,native", [c for c in CASES if c[2] == "TypeError"],
                         ids=[i for c, i in zip(CASES, IDS) if c[2] == "TypeError"])
def test_a_native_typeerror_is_unproven_and_the_other_six_are_proven(name, src, native):
    status = _status(src, name)
    if status is None:      # refused by name: no op was recorded, nothing is proven
        assert run_native(src) == "TypeError"
        return
    assert set(status) == set(ALL), (name, src, status)
    assert status["TypeError"] == "UNPROVEN", (name, src, status)
    assert all(v == "proven" for k, v in status.items() if k != "TypeError"), (name, src, status)


@pytest.mark.parametrize("name,src,native", [c for c in CASES if c[2] == ""], ids=[i for c, i in zip(CASES, IDS) if c[2] == ""])
def test_a_clean_call_records_all_seven_classes_proven(name, src, native):
    status = _status(src, name)
    assert status is not None and set(status) == set(ALL), (name, src, status)
    assert set(status.values()) == {"proven"}, (name, src, status)


@pytest.mark.parametrize("name", HAND_MODELLED)
def test_withdrawing_the_contract_withdraws_the_model(name, monkeypatch):
    """The contract is what admits the model: with it gone the call is a named refusal, not the hand model of old."""
    load_all()
    src = next(s for n, s, native in CASES if n == name and native == "")
    assert prove(src, ALL, time_budget=30.0).verdict == "TRUE"
    monkeypatch.delitem(REGISTRY, name)
    r = prove(src, ALL, time_budget=30.0)
    assert r.verdict == "UNKNOWN" and f"{name}() has no tested contract" in r.reason, (name, r.reason)


# --------------------------------------------------------------------------- negative controls
def _lax(real):
    def modelled(self, st, node, qualname, model, args, kwargs, hooks, **kw):
        result, proven = real(self, st, node, qualname, model, args, kwargs, hooks, **kw)
        proven["TypeError"] = "lax: proven whatever the model checked"
        return result, proven
    return modelled


@pytest.mark.parametrize("name,src,native", [c for c in CASES if c[2] == "TypeError"],
                         ids=[i for c, i in zip(CASES, IDS) if c[2] == "TypeError"])
def test_negative_control_a_modelled_that_proves_typeerror_unconditionally_is_caught(name, src, native, monkeypatch):
    """The control: with ``_modelled`` proving ``TypeError`` whatever the model recorded, the op of a call that raises it
    natively comes out proven -- the very thing ``test_a_native_typeerror_is_unproven_...`` forbids."""
    monkeypatch.setattr(Analyzer, "_modelled", _lax(Analyzer._modelled))
    status = _status(src, name)
    if status is None:
        pytest.skip("the model refuses this call by name: there is no op to prove")
    assert status.get("TypeError") == "proven", (name, src, status)


def test_the_length_of_a_truth_value_is_still_the_model_not_the_call():
    """``if A():`` and the length probe of ``list(A())`` ask for a length that is not a call the program writes: they run
    the model (``_len_model``), so no ``len:<class>`` op appears at a line where the program never calls ``len``."""
    src = ("class A:\n    def __len__(self):\n        return 2\n    def __iter__(self):\n        return iter([1, 2])\n"
           "x = list(A())\nif A():\n    y = 1\n")
    r = prove(src, ALL, time_budget=30.0)
    assert not [s for s in r.sites if s[2].startswith("len:")], r.sites
    assert run_native(src) == ""
