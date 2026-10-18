"""The exact product proof reads the contracts of ``abs``, ``int``, ``min`` and ``max`` ([internal ref] step 2).

``sv/exact_subset.py`` admitted ``abs(x)``, ``int(x)``, ``min(a, b)`` and ``max(a, b)`` from a table of names (the audit's ``M`` cell for the product
engine).  The allowlist now asks the rule: ``solve/contracts/visitors.py::ProductVisitor`` answers the audited arity while the contract
of the builtin is TESTED, so an untested or withdrawn contract takes the builtin out of the subset (the proof then says UNKNOWN).

Pinned: the audited shapes are admitted, every other shape and arity is refused; withdrawn / untested, the call is refused; and the
proof over the admitted shapes still answers as CPython does (a fixed pair per builtin: the equivalent program is TRUE, the violating one
is found).  NEGATIVE CONTROL: a visitor that admits ``sum`` is caught by the table of refused shapes.
"""

from __future__ import annotations

import ast
import dataclasses
import sys
import textwrap

import pytest

from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.solve.contracts.rules import get_rule
from tharkun.solve.contracts.visitors import ProductVisitor
from tharkun.sv import exact_subset, product_proof

AR = ("AssertionError",)

ADMITTED = [
    "x = abs(-3)\n", "x = int(3)\n", "x = min(3, 1)\n", "x = max(3, 1)\n",
    "def f(a, b):\n    return min(a, b)\n",
]
REFUSED = [
    "x = abs()\n", "x = abs(1, 2)\n", "x = int()\n", "x = int(1, 2)\n", "x = min(1)\n", "x = min(1, 2, 3)\n", "x = max(1)\n", "x = max(1, 2, 3)\n",
    "x = len(3)\n", "x = divmod(3, 2)\n", "x = pow(2, 3)\n", "x = sum(3)\n",
]


@pytest.mark.parametrize("src", ADMITTED)
def test_the_audited_shapes_are_admitted(src):
    assert exact_subset.audit(ast.parse(src), set()) is None, src


@pytest.mark.parametrize("src", REFUSED)
def test_every_other_shape_and_builtin_is_refused(src):
    assert exact_subset.audit(ast.parse(src), set()) is not None, src


@pytest.mark.parametrize("name", ["abs", "int", "min", "max"])
def test_the_visit_is_the_audited_arity_and_a_withdrawn_or_untested_contract_refuses(name, monkeypatch):
    load_all()
    rule = get_rule(name)
    assert rule.accept(ProductVisitor()) == ProductVisitor.EXACT_ARITY[name]
    src = {"abs": "x = abs(-3)\n", "int": "x = int(3)\n", "min": "x = min(3, 1)\n", "max": "x = max(3, 1)\n"}[name]
    assert exact_subset.audit(ast.parse(src), set()) is None
    monkeypatch.setitem(REGISTRY, name, dataclasses.replace(REGISTRY[name], tested=False))
    assert exact_subset.audit(ast.parse(src), set()) is not None, "untested: refused"
    monkeypatch.undo()
    load_all()
    monkeypatch.delitem(REGISTRY, name)
    assert exact_subset.audit(ast.parse(src), set()) is not None, "withdrawn: refused"


@pytest.mark.parametrize("name", ["len", "divmod", "pow", "sum", "float", "range", "sorted", "print", "isinstance"])
def test_no_other_builtin_has_a_product_visit(name):
    assert name not in ProductVisitor.implements() and "visit_" + name not in dir(ProductVisitor)


def test_negative_control_a_visitor_that_admits_sum_is_caught(monkeypatch):
    """The control: with ``visit_sum`` the allowlist would admit ``sum(3)``, which ``_symexec`` does not model exactly: the table of
    refused shapes above fails for such a visitor."""
    assert exact_subset.audit(ast.parse("x = sum(3)\n"), set()) is not None
    visits = []

    def admit_sum(self, rule, ctx):
        visits.append("sum")
        return 1

    monkeypatch.setattr(ProductVisitor, "visit_sum", admit_sum, raising=False)
    assert exact_subset.audit(ast.parse("x = sum(3)\n"), set()) is None
    assert visits, "the injected visitor must be reached"


_EQUIVALENT = """
    import _sv_verifier
    a = _sv_verifier.nondet_int()
    b = _sv_verifier.nondet_int()
    assert min(a, b) <= max(a, b)
    assert abs(a) >= 0
"""
_VIOLATING = """
    import _sv_verifier
    a = _sv_verifier.nondet_int()
    assert abs(a) > 0
"""


def _prove(src, tmp_path):
    p = tmp_path / "prog.py"
    p.write_text(textwrap.dedent(src))
    return product_proof.prove(p.read_text(), AR, program=str(p), python=sys.executable)


def test_the_proof_over_the_admitted_shapes_still_answers_as_cpython_does(tmp_path):
    ok = _prove(_EQUIVALENT, tmp_path)
    assert ok is not None and ok.verdict == "true", ok
    bad = _prove(_VIOLATING, tmp_path)
    assert bad is not None and bad.verdict == "false", bad
