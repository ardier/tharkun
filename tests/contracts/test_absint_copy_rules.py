"""absint's exact-builtin-list copy transfer reuses the tested ``list.copy`` model."""

from __future__ import annotations

import pytest

from tests.contracts._native import run_native
from tests.contracts.test_engine_audit_consumed import untested
from tharkun.solve.absint import PROPERTY_CLASSES, prove
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.solve.contracts.rules import RULES
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES

ALL = tuple(ALL_PROPERTY_CLASSES)


@pytest.mark.parametrize("head,call", [
    ("import copy\n", "copied = copy.copy(outer)"),
    ("import copy as c\n", "copied = c.copy(outer)"),
    ("from copy import copy as shallow\n", "copied = shallow(outer)"),
])
def test_builtin_list_copy_has_a_fresh_outer_list_and_shared_children(head, call):
    src = head + "outer = [[1]]\n" + call + "\ncopied[0].clear()\nassert len(outer[0]) == 0\n"
    assert run_native(src) == "", src
    result = prove(src, ("AssertionError",), time_budget=15)
    assert result.verdict == "TRUE", (result.verdict, result.reason)


@pytest.mark.parametrize("head,call", [
    ("import copy\n", "copied = copy.copy(outer)"),
    ("import copy as c\n", "copied = c.copy(outer)"),
    ("from copy import copy as shallow\n", "copied = shallow(outer)"),
])
@pytest.mark.parametrize("mutation,unchanged", [
    ("copied.append([2])", "outer"),
    ("outer.append([2])", "copied"),
])
def test_builtin_list_copy_outer_mutations_are_independent(head, call, mutation, unchanged):
    src = head + "outer = [[1]]\n" + call + "\n" + mutation + f"\nassert len({unchanged}) == 1\n"
    assert run_native(src) == "", src
    result = prove(src, ("AssertionError",), time_budget=15)
    assert result.verdict == "TRUE", (result.verdict, result.reason)


def test_exact_list_copy_records_all_contract_classes_as_proven():
    src = "import copy\ncopy.copy([[1]])\n"
    result = prove(src, ALL, time_budget=15)
    status = {site[3]: site[4] for site in result.sites if site[2].startswith("copy.copy:")}
    assert set(status) == set(ALL), (result.reason, result.sites)
    assert all("UNPROVEN" not in fact for fact in status.values()), status


def test_list_subclass_copy_hook_is_not_given_the_builtin_list_transfer():
    src = (
        "import copy\n"
        "class HookList(list):\n"
        "    def __copy__(self):\n"
        "        raise AssertionError('custom hook')\n"
        "copy.copy(HookList([1]))\n"
    )
    assert run_native(src) == "AssertionError"
    result = prove(src, ("AssertionError",), time_budget=15)
    assert result.verdict != "TRUE", (result.verdict, result.reason)


def test_non_list_builtin_and_deepcopy_keep_the_generic_path():
    for call in ("copy.copy({1: 2})", "copy.deepcopy([[1]])"):
        result = prove("import copy\n" + call + "\n", ("AssertionError",), time_budget=15)
        assert result.verdict != "TRUE", (call, result.verdict, result.reason)


def test_withdrawing_or_untesting_copy_copy_withdraws_the_list_transfer(monkeypatch):
    load_all()
    src = "import copy\ncopy.copy([[1]])\n"
    assert prove(src, ALL, time_budget=15).verdict == "TRUE"
    monkeypatch.setitem(REGISTRY, "copy.copy", untested(REGISTRY["copy.copy"]))
    result = prove(src, ALL, time_budget=15)
    assert result.verdict == "UNKNOWN" and "copy.copy has no tested contract" in result.reason, result.reason


def test_missing_copy_copy_rule_withdraws_the_list_transfer(monkeypatch):
    load_all()
    src = "import copy\ncopy.copy([[1]])\n"
    monkeypatch.delitem(RULES, "copy.copy")
    result = prove(src, ALL, time_budget=15)
    assert result.verdict == "UNKNOWN" and "copy.copy has no rule" in result.reason, result.reason
