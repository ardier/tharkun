"""absint reads the contracts of the ``random`` functions it modelled by hand ([internal ref] step 2, PR "absint random").

``random.random``, ``randint``, ``randrange``, ``choice``, ``uniform`` and ``seed`` had a transfer function (``rand.py::RandomMixin.m_random_*``)
that never read the contract of the callable (the audit's ``M`` cell).  They now run through their rule: ``AbsintVisitor.visit_random_<name>``
-> ``Analyzer._contract_module_model`` -> the unchanged model under ``Analyzer._modelled``, which records the SEVEN classes as the contract
states them (``random.<name>:<class>``) and upgrades a class only on an argument the model made.  (``random.getrandbits`` has no contract:
it stays a model of its own; ``random.sample`` and ``random.shuffle`` have no hand model: they were already the generic contract-driven call.)

What this file pins, against CPython 3.14.4:

* every program of the table: for each of the seven classes, absint is never TRUE for a class the program raises natively (the soundness
  direction), and is TRUE for every class of a clean program (the precision the hand model had);
* the call site records the contract's seven ops; where the call raises natively, the class CPython raises is UNPROVEN and the other six are
  proven (no user hook can run: the model refuses a user object by name);
* the contract is what admits the model: untested or withdrawn, the call is a named refusal, not the model of old;
* NEGATIVE CONTROLS: a ``_modelled`` that proves the class whatever the model checked is wrong exactly where this table looks.
"""

from __future__ import annotations

import pytest

from tests.contracts._native import run_native
from tests.contracts.test_engine_audit_consumed import untested
from tharkun.solve.absint import prove
from tharkun.solve.absint.interp import Analyzer
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES

ALL = tuple(ALL_PROPERTY_CLASSES)
RULES = ("random", "randint", "randrange", "choice", "uniform", "seed")
HEAD = "import random\n"

#: (function, the call, the class CPython raises ("" = clean))
CASES = [
    ("random", "x = random.random()\n", ""),
    ("random", "x = random.random(1)\n", "TypeError"),
    ("random", "x = random.random(a=1)\n", "TypeError"),
    ("randint", "x = random.randint(1, 6)\n", ""),
    ("randint", "x = random.randint(a=1, b=6)\n", ""),
    ("randint", "x = random.randint(-3, 3)\n", ""),
    ("randint", "x = random.randint(1.5, 6)\n", "TypeError"),
    ("randint", "x = random.randint('a', 6)\n", "TypeError"),
    ("randint", "x = random.randint(None, 6)\n", "TypeError"),
    ("randint", "x = random.randint()\n", "TypeError"),
    ("randint", "x = random.randint(1)\n", "TypeError"),
    ("randint", "x = random.randint(1, 2, 3)\n", "TypeError"),
    ("randrange", "x = random.randrange(5)\n", ""),
    ("randrange", "x = random.randrange(1, 10, 2)\n", ""),
    ("randrange", "x = random.randrange(10, 0, -1)\n", ""),
    ("randrange", "x = random.randrange(1.5)\n", "TypeError"),
    ("randrange", "x = random.randrange('a')\n", "TypeError"),
    ("randrange", "x = random.randrange(1, None, 2)\n", "TypeError"),
    ("randrange", "x = random.randrange()\n", "TypeError"),
    ("choice", "x = random.choice([1, 2, 3])\n", ""),
    ("choice", "x = random.choice((4, 5))\n", ""),
    ("choice", "x = random.choice([])\n", "IndexError"),
    ("choice", "x = random.choice(())\n", "IndexError"),
    ("choice", "x = random.choice(5)\n", "TypeError"),
    ("choice", "x = random.choice(None)\n", "TypeError"),
    ("choice", "x = random.choice()\n", "TypeError"),
    ("choice", "x = random.choice([1], [2])\n", "TypeError"),
    ("uniform", "x = random.uniform(0, 1)\n", ""),
    ("uniform", "x = random.uniform(a=0.5, b=2)\n", ""),
    ("uniform", "x = random.uniform(3, 1)\n", ""),
    ("uniform", "x = random.uniform('a', 1)\n", "TypeError"),
    ("uniform", "x = random.uniform(None, 1)\n", "TypeError"),
    ("uniform", "x = random.uniform(1)\n", "TypeError"),
    ("seed", "random.seed(1)\n", ""),
    ("seed", "random.seed()\n", ""),
    ("seed", "random.seed(None)\n", ""),
    ("seed", "random.seed('abc')\n", ""),
    ("seed", "random.seed(1.5)\n", ""),
    ("seed", "random.seed(1, 2)\n", ""),
    ("seed", "random.seed([1])\n", "TypeError"),
    ("seed", "random.seed(1, 2, 3)\n", "TypeError"),
]
IDS = [f"{name}-{i}" for i, (name, _s, _n) in enumerate(CASES)]
RAISING = [c for c in CASES if c[2] != ""]
CLEAN = [c for c in CASES if c[2] == ""]


def _ids(cases):
    return [i for c, i in zip(CASES, IDS) if c in cases]


@pytest.mark.parametrize("name,call,native", CASES, ids=IDS)
def test_the_table_is_native(name, call, native):
    assert run_native(HEAD + call) == native, (name, call)


@pytest.mark.parametrize("name,call,native", CASES, ids=IDS)
def test_never_true_for_a_class_cpython_raises_and_true_for_every_class_of_a_clean_program(name, call, native):
    src = HEAD + call
    for exc in ALL:
        verdict = prove(src, (exc,), time_budget=30.0).verdict
        if exc == native:
            assert verdict != "TRUE", (name, call, exc)
        elif native == "":
            assert verdict == "TRUE", (name, call, exc)


def _status(src: str, name: str):
    """``{class: "proven" | "UNPROVEN"}`` of the ``random.<name>:<class>`` ops recorded at the call, or ``None`` when the model refuses the
    call by name (it records nothing)."""
    r = prove(src, ALL, time_budget=30.0)
    if r.verdict == "UNKNOWN" and r.reason.startswith("unsupported:"):
        return None
    return {s[3]: ("UNPROVEN" if s[4].startswith("UNPROVEN") else "proven") for s in r.sites if s[2] == f"random.{name}:{s[3]}"}


@pytest.mark.parametrize("name,call,native", RAISING, ids=_ids(RAISING))
def test_the_class_cpython_raises_is_unproven_and_the_other_six_are_proven(name, call, native):
    status = _status(HEAD + call, name)
    if status is None:      # refused by name: no op was recorded, nothing is proven
        assert run_native(HEAD + call) == native
        return
    assert set(status) == set(ALL), (name, call, status)
    assert status[native] == "UNPROVEN", (name, call, status)
    assert all(v == "proven" for k, v in status.items() if k != native), (name, call, status)


@pytest.mark.parametrize("name,call,native", CLEAN, ids=_ids(CLEAN))
def test_a_clean_call_records_all_seven_classes_proven(name, call, native):
    status = _status(HEAD + call, name)
    assert status is not None and set(status) == set(ALL), (name, call, status)
    assert set(status.values()) == {"proven"}, (name, call, status)


def test_a_sequence_that_may_be_empty_leaves_only_indexerror_unproven():
    """``choice(xs)`` of a list whose length the analyzer knows nothing about: ``IndexError`` stays unproven, ``TypeError`` is proven."""
    src = (HEAD + "import _sv_verifier as sv\nn = sv.nondet_int()\nxs = [0] * (n % 3)\nx = random.choice(xs)\n")
    status = _status(src, "choice")
    assert status is not None and status["IndexError"] == "UNPROVEN" and status["TypeError"] == "proven", status


@pytest.mark.parametrize("name", RULES)
def test_withdrawing_the_contract_withdraws_the_model(name, monkeypatch):
    """The contract is what admits the model: with it gone (or untested) the call is a named refusal, not the hand model of old."""
    load_all()
    qual = f"random.{name}"
    src = HEAD + next(c for n, c, native in CASES if n == name and native == "")
    assert prove(src, ALL, time_budget=30.0).verdict == "TRUE"
    monkeypatch.setitem(REGISTRY, qual, untested(REGISTRY[qual]))
    r = prove(src, ALL, time_budget=30.0)
    assert r.verdict == "UNKNOWN" and f"{qual} has no tested contract" in r.reason, (name, r.reason)
    monkeypatch.undo()
    load_all()
    monkeypatch.delitem(REGISTRY, qual)
    r = prove(src, ALL, time_budget=30.0)
    assert r.verdict == "UNKNOWN" and f"{qual} has no tested contract" in r.reason, (name, r.reason)


def test_a_function_without_a_contract_keeps_its_model():
    """``random.getrandbits`` has no contract (nothing to read): its model answers as before."""
    assert "random.getrandbits" not in REGISTRY
    assert prove(HEAD + "x = random.getrandbits(8)\n", ALL, time_budget=30.0).verdict == "TRUE"


# --------------------------------------------------------------------------- negative controls
def _lax(real, exc):
    def modelled(self, st, node, qualname, model, args, kwargs, hooks, **kw):
        result, proven = real(self, st, node, qualname, model, args, kwargs, hooks, **kw)
        proven[exc] = "lax: proven whatever the model checked"
        return result, proven
    return modelled


@pytest.mark.parametrize("name,call,native", RAISING, ids=_ids(RAISING))
def test_negative_control_a_modelled_that_proves_the_class_unconditionally_is_caught(name, call, native, monkeypatch):
    """The control: with ``_modelled`` proving the class CPython raises whatever the model recorded, the op of the call comes out proven --
    the very thing ``test_the_class_cpython_raises_is_unproven_...`` forbids."""
    monkeypatch.setattr(Analyzer, "_modelled", _lax(Analyzer._modelled, native))
    status = _status(HEAD + call, name)
    if status is None:
        pytest.skip("the model refuses this call by name: there is no op to prove")
    assert status.get(native) == "proven", (name, call, status)
