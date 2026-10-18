"""The rules of the whole registry ([internal ref], [internal ref]): the structure every engine's visitor is written against."""

from __future__ import annotations

from collections import Counter

from tharkun.solve.contracts import all_contracts, get
from tharkun.solve.contracts.adapters import common
from tharkun.solve.contracts.rules import all_rules, get_rule

def test_every_contract_is_a_rule_and_every_rule_a_contract():
    contracts = {c.qualname for c in all_contracts()}
    assert contracts == {r.qualname for r in all_rules()}


def test_a_rule_holds_the_contract_the_registry_serves():
    for rule in all_rules():
        assert rule.contract is get(rule.qualname)


def test_a_kind_names_one_class():
    """The visitor dispatch key is unique to a class: two classes sharing one would be one rule to every engine."""
    by_kind = {}
    for rule in all_rules():
        by_kind.setdefault(rule.kind, set()).add(type(rule))
    clash = {k: v for k, v in by_kind.items() if len(v) > 1}
    assert not clash, clash


def test_the_kinds_are_the_snake_case_of_the_class_names():
    for rule in all_rules():
        cls = type(rule)
        assert cls.kind == "".join(("_" + c.lower()) if c.isupper() else c for c in cls.__name__).lstrip("_") \
            or "kind" in cls.__dict__


def test_a_family_member_is_one_instance_per_qualname():
    counts = Counter(r.qualname for r in all_rules())
    assert set(counts.values()) == {1}


def test_the_hand_verified_call_shapes_are_the_rules_shapes():
    """``adapters/common.py::CALL_SHAPE`` (exactly N positional arguments, no keyword) is the engines' copy of the
    shape of heapq's and bisect's rules."""
    for qualname, n in common.CALL_SHAPE.items():
        rule = get_rule(qualname)
        assert rule.shape is not None and (rule.shape.min_args, rule.shape.max_args) == (n, n), qualname
        assert rule.applicable(n, False) and not rule.applicable(n, True) and not rule.applicable(n + 1, False)


def test_a_rule_without_a_verified_shape_applies_to_no_call():
    rule = get_rule("list.append")
    assert rule.shape is None and not rule.applicable(2, False) and not rule.applicable(1, False)
