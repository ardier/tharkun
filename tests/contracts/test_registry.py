"""Tests for the contract loader API ([internal ref])."""

from __future__ import annotations

from tharkun.solve.contracts import all_contracts, for_module, for_receiver, get
from tharkun.solve.contracts.registry import REGISTRY, load_all
from tharkun.solve.contracts.rules import RULES, Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseClaim


def test_get_returns_none_for_unwritten_contract():
    assert get("no.such.callable") is None


def test_get_returns_a_written_contract():
    c = get("len")
    assert c is not None
    assert c.qualname == "len"


def test_all_contracts_is_qualname_sorted_and_nonempty():
    contracts = all_contracts()
    names = [c.qualname for c in contracts]
    assert names == sorted(names)
    assert "len" in names


def test_for_module_returns_only_that_modules_entries():
    # builtins are bare names, not "module.name" -- for_module("math") should not
    # accidentally pick up unrelated entries.
    for c in for_module("math"):
        assert c.qualname.startswith("math.")


def test_for_receiver_prefix_matches():
    load_all()
    fake = Contract(
        "____test_type__.method",
        tuple(RaiseClaim(e, "never") for e in ALL_PROPERTY_CLASSES),
        True, None, Effects(), source=("test",),
    )
    class TestRule(Rule):
        names = (fake.qualname,)
        CONTRACT = fake

    register_rule(TestRule)
    try:
        assert fake in for_receiver("____test_type__")
        assert fake not in for_receiver("____test_type")
    finally:
        del REGISTRY[fake.qualname]
        del RULES[fake.qualname]


def test_register_rule_rejects_duplicate_qualname():
    load_all()
    existing = get("len")
    import pytest

    with pytest.raises(ValueError, match="duplicate"):
        class Duplicate(Rule):
            names = ("len",)
            CONTRACT = existing

        register_rule(Duplicate)
