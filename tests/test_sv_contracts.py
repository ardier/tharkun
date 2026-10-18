"""The component contracts are data, and the code and the design doc are held to them
([internal ref] step 3a, ``sv/contracts.py``; docs/SV.md "Component contracts").

* the doc's table IS ``contracts.render_table()`` (a finding either way: fix the stale side in the
  same commit);
* every ``file::symbol`` a contract names exists;
* every kind of row a component writes into the store is registered, and every registered kind is
  one the code can write;
* ``certified=True`` is written by exactly one module, ``sv/obligation_table.py`` -- MG's review bar
  ("no third writer"): a seed, a solve, a fuzz find or a producer's claim can never certify.
Pure AST scans and imports: nothing runs a task.
"""
from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

import pytest

from tharkun.sv import contracts
from tharkun.sv.contracts import COMPONENTS, KIND_NAMES, KINDS, render_table

SRC = Path(contracts.__file__).resolve().parents[1]           # .../tharkun
SV = SRC / "sv"
DOC = SRC.parents[1] / "docs" / "SV.md"
BEGIN = "<!-- BEGIN CONTRACT TABLE (generated: tharkun.sv.contracts.render_table; tests/test_sv_contracts.py) -->"
END = "<!-- END CONTRACT TABLE -->"


def test_the_design_docs_table_is_the_registrys_table():
    text = DOC.read_text()
    assert BEGIN in text and END in text, "docs/SV.md lost the generated contract-table markers"
    in_doc = text.split(BEGIN, 1)[1].split(END, 1)[0].strip()
    assert in_doc == render_table(), (
        "docs/SV.md's component-contract table differs from tharkun.sv.contracts.COMPONENTS: "
        "regenerate it with `python -c 'from tharkun.sv.contracts import render_table; print(render_table())'`")


@pytest.mark.parametrize("component", COMPONENTS, ids=lambda c: c.id)
def test_every_named_symbol_exists(component):
    for where in component.where.split(";"):
        path, _, symbol = where.partition("::")
        assert (SRC / path).is_file(), f"{component.id}: {path} does not exist"
        module = importlib.import_module("tharkun." + path.removesuffix(".py").replace("/", "."))
        assert hasattr(module, symbol), f"{component.id}: {path} has no {symbol}"


def test_component_ids_are_unique_and_every_kind_has_writers_or_is_documented_as_unread():
    ids = [c.id for c in COMPONENTS]
    assert len(ids) == len(set(ids))
    for kind in KINDS.values():
        assert kind.writers, f"{kind.name} has no writer"
        assert kind.store in ("fact", "evidence")


def test_only_the_obligation_kind_is_certifiable():
    assert {k.name for k in KINDS.values() if k.certifiable} == {"obligation"}


def _py_files():
    return [p for p in sorted(SV.rglob("*.py")) if "__pycache__" not in p.parts]


def _kind_literals(path):
    """String kinds a module hands to ``Fact(kind=...)``, ``.publish(<kind>, ...)`` or
    ``evidence(kind=...)``, plus its module-level ``NAME = "kind"`` constants for the two kind-owning
    modules."""
    tree = ast.parse(path.read_text())
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        if name == "Fact":
            for kw in node.keywords:
                if kw.arg == "kind" and isinstance(kw.value, ast.Constant):
                    found.add(kw.value.value)
        if name == "publish" and node.args and isinstance(node.args[0], ast.Constant):
            found.add(node.args[0].value)
    if path.name in ("obligation_table.py", "eregion_bridge.py"):
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id.isupper() and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str) and node.targets[0].id not in ("OPEN", "DISCHARGED", "VIOLATED")):
                found.add(node.value.value)
    return found


def test_every_kind_the_code_writes_is_registered_and_every_registered_kind_can_be_written():
    written = set()
    for path in _py_files():
        written |= _kind_literals(path)
    assert written <= KIND_NAMES, f"unregistered kind(s) in tharkun/sv: {sorted(written - KIND_NAMES)}"
    # ... and a registered kind nothing can write is a stale registry entry
    assert KIND_NAMES <= written, f"registered but never written: {sorted(KIND_NAMES - written)}"


def test_the_bridges_kind_constants_are_the_registered_names():
    from tharkun.sv import eregion_bridge, obligation_table
    assert (eregion_bridge.INVARIANT_CANDIDATE in KINDS and eregion_bridge.CERTIFIED_INVARIANT in KINDS
            and eregion_bridge.DRAW_VECTOR in KINDS)
    for name in ("OBLIGATION", "SITE_RECORD", "SITE_OUTCOME", "STAGE_RAN", "CLASS_HINTS"):
        assert getattr(obligation_table, name) in KINDS


def test_certified_true_is_written_only_by_the_obligation_table():
    """MG's review bar, as a scan that fails closed: outside ``obligation_table.py`` no module of
    tharkun.sv passes ``certified=`` anything but the literal ``False`` (a variable, an expression or
    ``True`` all count), builds a ``Fact`` with ``certified`` as a positional argument (its sixth
    field), or assigns ``.certified`` on an object."""
    offenders = []
    for path in _py_files():
        if path.name == "obligation_table.py":
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            where = f"{path.relative_to(SRC)}:{getattr(node, 'lineno', '?')}"
            if isinstance(node, ast.keyword) and node.arg == "certified" \
                    and not (isinstance(node.value, ast.Constant) and node.value.value is False):
                offenders.append(f"{path.relative_to(SRC)}:{node.value.lineno}")
            elif isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", None)) == "Fact" \
                    and len(node.args) >= 6:
                offenders.append(where)
            elif isinstance(node, ast.Attribute) and node.attr == "certified" and isinstance(node.ctx, ast.Store):
                offenders.append(where)
    assert offenders == [], f"certified written outside sv/obligation_table.py: {offenders}"


def test_the_two_certifying_methods_are_the_only_certified_true_writers_in_the_table():
    tree = ast.parse((SV / "obligation_table.py").read_text())
    writers = set()
    for cls in (n for n in tree.body if isinstance(n, ast.ClassDef)):
        for fn in (n for n in cls.body if isinstance(n, ast.FunctionDef)):
            for node in ast.walk(fn):
                if isinstance(node, ast.keyword) and node.arg == "certified" and not (
                        isinstance(node.value, ast.Constant) and node.value.value is False):
                    writers.add(fn.name)
    assert writers == {"discharge", "violate"}


def test_kind_names_in_a_contract_cell_are_registered_when_they_look_like_kinds():
    """A contract cell names store kinds in backticks; the words that are kinds must be REGISTERED
    ones (a typo or a rename must fail here, not silently drift)."""
    kind_like = {"obligation", "site_record", "site_outcome", "stage_ran", "class_hints", "candidate_invariant",
                 "input_sensitivity", "loop_trip_count", "invariant_candidate", "certified_invariant",
                 "candidate_vector", "call_target", "regex_pattern", "module_constant"}
    for c in COMPONENTS:
        for cell in (c.reads, c.writes):
            for token in re.findall(r"`([a-z_]+)`", cell):
                if token in kind_like:
                    assert token in KIND_NAMES, f"{c.id}: `{token}` is not a registered kind"
