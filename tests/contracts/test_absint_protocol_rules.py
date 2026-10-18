"""absint reads the contracts of the object protocols it dispatches by hand ([internal ref] step 2, PR "absint protocols").

The operators, the comparisons, ``__getitem__``, ``len``, iteration and the truth value of a USER INSTANCE are dispatched by
``Analyzer.binary_special`` / ``special`` / ``iterable`` / ``_len_model`` / ``truth_split`` -- a hand implementation of what
``solve/contracts/rules_protocols.py`` states as tested rules, which never read them (the audit's ``M`` cell).  Each dispatch now runs through
its rule: ``Analyzer.visit_protocol`` -> ``Rule.accept(AbsintVisitor)`` -> ``AbsintVisitor.visit_protocol_<name>`` ->
``Analyzer._through_protocol`` -> the unchanged dispatch under a tap, and the seven classes are recorded at the node as
``protocol.<name>:<class>``, as the contract states them.

What this file pins, against CPython 3.14.4:

* every program of the table: for each of the seven classes, absint is never TRUE for a class the program raises natively (the soundness
  direction), and is TRUE for every class of a clean program (the precision the hand dispatch had);
* the ops of the dispatch are recorded; where the dispatch has no applicable method (``TypeError`` of the contract's native condition) that
  class is UNPROVEN and the other six are proven;
* a class that propagates out of the dunder (``ZeroDivisionError`` from ``__add__``) is proven AT THE DISPATCH because the body is analysed:
  the program is not TRUE for it, because the body's own site is unproven;
* the contract is what admits the dispatch: untested or withdrawn, it is a named refusal, not the dispatch of old;
* NEGATIVE CONTROLS: a ``_proven_classes`` that proves every class is wrong exactly where this table looks.
"""

from __future__ import annotations

import pytest

from tests.contracts._native import run_native
from tests.contracts.test_engine_audit_consumed import untested
from tharkun.solve.absint import prove
from tharkun.solve.absint.interp import Analyzer
from tharkun.solve.absint.visitor import PROTOCOL_DISPATCHES
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES

ALL = tuple(ALL_PROPERTY_CLASSES)

_A_ADD = "class A:\n    def __add__(self, o):\n        return 1\n"
_ITER = "class A:\n    def __iter__(self):\n        return iter([1, 2])\n"
_LEN = "class A:\n    def __len__(self):\n        return {}\n"

#: (rule, program, the class CPython raises ("" = clean))
CASES = [
    ("protocol.binary_operator", _A_ADD + "x = A() + A()\n", ""),
    ("protocol.binary_operator", _A_ADD + "x = A() + 1\n", ""),
    ("protocol.binary_operator", "class A:\n    def __radd__(self, o):\n        return 2\nx = 1 + A()\n", ""),
    ("protocol.binary_operator", "class A:\n    def __sub__(self, o):\n        return 3\nx = A() - 1\n", ""),
    ("protocol.binary_operator", _A_ADD + "class B(A):\n    def __radd__(self, o):\n        return 2\nx = A() + B()\n", ""),
    ("protocol.binary_operator", "class A:\n    pass\nx = A() + 1\n", "TypeError"),
    ("protocol.binary_operator", "class A:\n    pass\nx = 1 + A()\n", "TypeError"),
    ("protocol.binary_operator", "class A:\n    pass\nx = A() - A()\n", "TypeError"),
    ("protocol.binary_operator", "class A:\n    def __add__(self):\n        return 1\nx = A() + A()\n", "TypeError"),
    ("protocol.inplace_operator", "class A:\n    def __iadd__(self, o):\n        return self\na = A()\na += 1\n", ""),
    ("protocol.inplace_operator", _A_ADD + "a = A()\na += 1\n", ""),
    ("protocol.inplace_operator", "class A:\n    pass\na = A()\na += 1\n", "TypeError"),
    ("protocol.comparison", "class A:\n    def __lt__(self, o):\n        return True\nx = A() < A()\n", ""),
    ("protocol.comparison", "class A:\n    def __eq__(self, o):\n        return True\nx = A() == A()\n", ""),
    ("protocol.comparison", "class A:\n    def __gt__(self, o):\n        return True\nx = 1 < A()\n", ""),
    ("protocol.comparison", "class A:\n    pass\nx = A() == A()\n", ""),
    ("protocol.comparison", "class A:\n    pass\nx = A() != 1\n", ""),
    ("protocol.comparison", "class A:\n    pass\nx = A() < A()\n", "TypeError"),
    ("protocol.comparison", "class A:\n    pass\nx = A() >= 1\n", "TypeError"),
    ("protocol.container_getitem", "class A:\n    def __getitem__(self, i):\n        return i\nx = A()[0]\n", ""),
    ("protocol.container_getitem", "class A:\n    def __getitem__(self, i):\n        return 5\nx = A()[1] + A()[2]\n", ""),
    ("protocol.container_getitem", "class A:\n    pass\nx = A()[0]\n", "TypeError"),
    ("protocol.container_len", _LEN.format("2") + "x = len(A())\n", ""),
    ("protocol.container_len", _LEN.format("0") + "x = len(A())\n", ""),
    ("protocol.container_len", _LEN.format("'a'") + "x = len(A())\n", "TypeError"),
    ("protocol.container_len", "class A:\n    pass\nx = len(A())\n", "TypeError"),
    ("protocol.iteration", _ITER + "for v in A():\n    pass\n", ""),
    ("protocol.iteration", _ITER + "x = list(A())\n", ""),
    ("protocol.iteration", "class A:\n    pass\nfor v in A():\n    pass\n", "TypeError"),
    ("protocol.iteration", "class A:\n    def __iter__(self):\n        return 1\nfor v in A():\n    pass\n", "TypeError"),
    ("protocol.truthiness", _LEN.format("1") + "if A():\n    pass\n", ""),
    ("protocol.truthiness", _LEN.format("0") + "if A():\n    pass\nelse:\n    pass\n", ""),
    ("protocol.truthiness", "class A:\n    pass\nif A():\n    pass\n", ""),
    ("protocol.truthiness", "class A:\n    pass\nx = not A()\n", ""),
    ("protocol.truthiness", _LEN.format("'a'") + "if A():\n    pass\n", "TypeError"),
    ("protocol.attribute_access", "class A:\n    def __init__(self):\n        self.x = 1\na = A()\ny = a.x\n", ""),
    ("protocol.attribute_access", "class A:\n    z = 5\na = A()\ny = a.z\n", ""),
    ("protocol.attribute_access", "class A:\n    def m(self):\n        return 1\nx = A().m()\n", ""),
    ("protocol.attribute_access", "class A:\n    pass\na = A()\ny = a.x\n", "AttributeError"),
    ("protocol.attribute_access", "class A:\n    def __init__(self):\n        self.x = 1\na = A()\ny = a.w\n", "AttributeError"),
]
#: the read of an attribute is on the hot path of every program with an object: only the classes left unproven are recorded (``lean``)
LEAN = frozenset({"protocol.attribute_access"})
IDS = [f"{name.split('.', 1)[1]}-{i}" for i, (name, _s, _n) in enumerate(CASES)]
CLEAN = [c for c in CASES if c[2] == "" and c[0] not in LEAN]
RAISING = [c for c in CASES if c[2] != "" and c[0] not in LEAN]
LEAN_CASES = [c for c in CASES if c[0] in LEAN]


def _ids(cases):
    return [i for c, i in zip(CASES, IDS) if c in cases]


@pytest.mark.parametrize("rule,src,native", CASES, ids=IDS)
def test_the_table_is_native(rule, src, native):
    assert run_native(src) == native, (rule, src)


@pytest.mark.parametrize("rule,src,native", CASES, ids=IDS)
def test_never_true_for_a_class_cpython_raises_and_true_for_every_class_of_a_clean_program(rule, src, native):
    for exc in ALL:
        verdict = prove(src, (exc,), time_budget=30.0).verdict
        if exc == native:
            assert verdict != "TRUE", (rule, src, exc)
        elif native == "":
            assert verdict == "TRUE", (rule, src, exc)


def _status(src: str, rule: str):
    """``{class: "proven" | "UNPROVEN"}`` of the ``<rule>:<class>`` ops recorded at the dispatch, or ``None`` when the analysis refuses the
    program by name (it records nothing for the dispatch)."""
    r = prove(src, ALL, time_budget=30.0)
    if r.verdict == "UNKNOWN" and r.reason.startswith("unsupported:"):
        return None
    return {s[3]: ("UNPROVEN" if s[4].startswith("UNPROVEN") else "proven") for s in r.sites if s[2] == f"{rule}:{s[3]}"}


@pytest.mark.parametrize("rule,src,native", CLEAN, ids=_ids(CLEAN))
def test_a_clean_dispatch_records_all_seven_classes_proven(rule, src, native):
    status = _status(src, rule)
    assert status is not None and set(status) == set(ALL), (rule, src, status)
    assert set(status.values()) == {"proven"}, (rule, src, status)


@pytest.mark.parametrize("rule,src,native", RAISING, ids=_ids(RAISING))
def test_a_dispatch_with_no_applicable_method_leaves_only_the_native_typeerror_unproven(rule, src, native):
    status = _status(src, rule)
    if status is None:      # refused by name: nothing is recorded, nothing is proven
        assert run_native(src) == native
        return
    assert set(status) == set(ALL), (rule, src, status)
    assert status[native] == "UNPROVEN", (rule, src, status)
    assert all(v == "proven" for k, v in status.items() if k != native), (rule, src, status)


#: a class that propagates out of the dunder: proven at the dispatch (the body is analysed), not TRUE for the program (its own site is not)
_SV = "import _sv_verifier as sv\n"
PROPAGATION = [
    ("protocol.binary_operator", _SV + "class A:\n    def __add__(self, o):\n        return 1 // sv.nondet_int()\nx = A() + A()\n", "ZeroDivisionError"),
    ("protocol.inplace_operator", _SV + "class A:\n    def __iadd__(self, o):\n        x = 1 // sv.nondet_int()\n        return self\na = A()\na += 1\n",
     "ZeroDivisionError"),
    ("protocol.comparison", _SV + "class A:\n    def __lt__(self, o):\n        x = 1 // sv.nondet_int()\n        return True\nx = A() < A()\n",
     "ZeroDivisionError"),
    ("protocol.container_getitem", _SV + "class A:\n    def __getitem__(self, i):\n        return 1 // sv.nondet_int()\nx = A()[0]\n",
     "ZeroDivisionError"),
    ("protocol.container_getitem", "class A:\n    def __getitem__(self, i):\n        raise IndexError('x')\nx = A()[0]\n", "IndexError"),
    ("protocol.container_len", _SV + "class A:\n    def __len__(self):\n        x = 1 // sv.nondet_int()\n        return 2\nn = len(A())\n",
     "ZeroDivisionError"),
    ("protocol.truthiness", _SV + "class A:\n    def __len__(self):\n        x = 1 // sv.nondet_int()\n        return 1\nif A():\n    pass\n",
     "ZeroDivisionError"),
]


@pytest.mark.parametrize("rule,src,native", PROPAGATION, ids=[f"{r.split('.', 1)[1]}-{n}-{i}" for i, (r, _s, n) in enumerate(PROPAGATION)])
def test_a_class_that_propagates_from_the_dunder_is_proven_at_the_dispatch_and_not_true_for_the_program(rule, src, native):
    assert run_native(src) == native
    assert prove(src, (native,), time_budget=30.0).verdict != "TRUE", (rule, src)
    status = _status(src, rule)
    assert status is not None and status[native] == "proven", (rule, src, status)


@pytest.mark.parametrize("rule,src,native", LEAN_CASES, ids=_ids(LEAN_CASES))
def test_a_lean_protocol_records_only_the_classes_it_leaves_unproven(rule, src, native):
    """A clean read of an attribute records no ``protocol.attribute_access`` op at all; a missing attribute records exactly the native
    ``AttributeError``, unproven (the other six are proven: no user ``__getattribute__`` / ``__getattr__`` can run)."""
    status = _status(src, rule)
    assert status is not None, (rule, src)
    if native == "":
        assert status == {}, (rule, src, status)
    else:
        assert status == {native: "UNPROVEN"}, (rule, src, status)


# --------------------------------------------------------------------------- the contract admits the dispatch
@pytest.mark.parametrize("rule", PROTOCOL_DISPATCHES)
def test_withdrawing_or_untesting_the_contract_withdraws_the_dispatch(rule, monkeypatch):
    """With the contract gone (or untested) the dispatch of a user instance is a named refusal, not the hand dispatch of old."""
    load_all()
    src = next(s for r, s, native in CASES if r == rule and native == "")
    assert prove(src, ALL, time_budget=30.0).verdict == "TRUE"
    monkeypatch.setitem(REGISTRY, rule, untested(REGISTRY[rule]))
    r = prove(src, ALL, time_budget=30.0)
    assert r.verdict == "UNKNOWN" and f"{rule} has no tested contract" in r.reason, (rule, r.reason)
    monkeypatch.undo()
    load_all()
    monkeypatch.delitem(REGISTRY, rule)
    r = prove(src, ALL, time_budget=30.0)
    assert r.verdict == "UNKNOWN" and f"{rule} has no tested contract" in r.reason, (rule, r.reason)


def test_a_builtin_operand_does_not_ask_for_a_protocol():
    """``1 + 2``, ``xs[0]``, ``len(xs)``, ``for v in xs`` and ``if xs`` are the builtins' own: no ``protocol.*`` op is recorded."""
    src = "xs = [1, 2]\nx = 1 + 2\ny = xs[0]\nn = len(xs)\nfor v in xs:\n    pass\nif xs:\n    pass\nz = 1 < 2\n"
    r = prove(src, ALL, time_budget=30.0)
    assert r.verdict == "TRUE"
    assert not [s for s in r.sites if s[2].startswith("protocol.")], r.sites


# --------------------------------------------------------------------------- negative controls
@pytest.mark.parametrize("rule,src,native", RAISING, ids=_ids(RAISING))
def test_negative_control_a_proven_classes_that_proves_everything_is_caught(rule, src, native, monkeypatch):
    """The control: with every class proven whatever the dispatch recorded, the native ``TypeError`` of a dispatch with no applicable
    method comes out proven -- the very thing the test above forbids."""
    monkeypatch.setattr(Analyzer, "_proven_classes", staticmethod(lambda contract, seen, hooks, native=None, type_free=None: {
        exc: "lax: proven whatever the dispatch recorded" for exc in ALL}))
    status = _status(src, rule)
    if status is None:
        pytest.skip("the analysis refuses this program by name: there is no op to prove")
    assert status.get(native) == "proven", (rule, src, status)
