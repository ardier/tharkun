"""``solve/contracts/rules.py``: one class per rule, one visitor per engine ([internal ref])."""
from __future__ import annotations

import pytest

from tharkun.solve.contracts import all_contracts, for_receiver, get, registry
from tharkun.solve.contracts import rules
from tharkun.solve.contracts._helpers import never_claims
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec


def toy_contract(qualname, tested=True):
    return Contract(qualname=qualname, raises=never_claims(tested=frozenset({
        "AssertionError", "AttributeError", "FloatingPointError", "IndexError", "KeyError", "TypeError",
        "ZeroDivisionError"}) if tested else frozenset()),
        may_raise_other=True, returns=ReturnSpec("int", tested=tested), effects=Effects(deterministic=True),
        source=("test",), tested=tested)


@pytest.fixture
def toys():
    """Register a two-member family and a single rule under ``toy.*``; remove them afterwards."""
    registry.load_all()                       # the registry is lazy: capture it AFTER it is filled
    before_rules, before_contracts = dict(rules.RULES), dict(registry.REGISTRY)
    registry.REGISTRY["toy.plain"] = toy_contract("toy.plain")      # a contract that is not (yet) a rule

    class ToyArith(rules.Rule):
        names = ("toy.add", "toy.sub")
        SHAPE = rules.Shape(2, 2)
        _OPS = {"toy.add": lambda a, b: a + b, "toy.sub": lambda a, b: a - b}

        @classmethod
        def contract_for(cls, qualname):
            return toy_contract(qualname)

        def native(self, *args, **kwargs):
            return self._OPS[self.qualname](*args)

        @classmethod
        def sample(cls, qualname):
            return f"{qualname}(1, 2)"

    class ToyLen(rules.Rule):
        names = ("toy.len",)
        kind = "toy_length"
        CONTRACT = toy_contract("toy.len", tested=False)
        SHAPE = rules.Shape(1, 1)

    rules.register_rule(ToyArith)
    rules.register_rule(ToyLen)
    yield ToyArith, ToyLen
    rules.RULES.clear(); rules.RULES.update(before_rules)
    registry.REGISTRY.clear(); registry.REGISTRY.update(before_contracts)


class EngineA(rules.Visitor):
    engine = "engine-a"
    not_applicable = {"toy_length": "has no length to speak of"}

    def visit_toy_arith(self, rule, ctx):
        return ("arith", rule.qualname, ctx)


class EngineB(rules.Visitor):
    engine = "engine-b"


def test_a_family_registers_one_rule_per_qualname_and_a_kind_per_class(toys):
    arith, length = toys
    assert arith.kind == "toy_arith" and length.kind == "toy_length"          # snake_case, or set explicitly
    got = {r.qualname: r for r in rules.all_rules() if r.qualname.startswith("toy.")}
    assert sorted(got) == ["toy.add", "toy.len", "toy.sub"]
    assert got["toy.add"].kind == got["toy.sub"].kind == "toy_arith"
    assert type(got["toy.add"]) is type(got["toy.sub"]) is arith
    assert [r.qualname for r in rules.all_rules()] == sorted(r.qualname for r in rules.all_rules())


def test_the_registry_facade_keeps_working_unchanged(toys):
    rule = rules.get_rule("toy.add")
    assert get("toy.add") is not None and get("toy.add").qualname == "toy.add"
    assert get("toy.add").tested and rule.tested and not rules.get_rule("toy.len").tested     # claim-level flags kept
    assert {c.qualname for c in for_receiver("toy")} == {"toy.add", "toy.sub", "toy.len", "toy.plain"}
    assert get("toy.add") in all_contracts()
    assert rules.get_rule("toy.plain") is None and get("toy.plain") is not None       # a plain contract is untouched


def test_a_qualname_cannot_be_registered_twice_as_a_rule_or_as_a_plain_contract(toys):
    class Again(rules.Rule):
        names = ("toy.add",)
        CONTRACT = toy_contract("toy.add")

    class OverPlain(rules.Rule):
        names = ("toy.plain",)                 # a plain contract already owns it
        CONTRACT = toy_contract("toy.plain")

    with pytest.raises(ValueError, match="duplicate"):
        rules.register_rule(Again)
    with pytest.raises(ValueError, match="duplicate"):
        rules.register_rule(OverPlain)
    assert get("toy.plain").qualname == "toy.plain" and "toy.plain" not in rules.RULES


def test_a_contract_that_names_another_qualname_is_refused(toys):
    class Wrong(rules.Rule):
        names = ("toy.wrong",)
        CONTRACT = toy_contract("toy.elsewhere")

    with pytest.raises(ValueError, match="names"):
        rules.register_rule(Wrong)
    assert "toy.wrong" not in rules.RULES and get("toy.wrong") is None


def test_a_rule_covers_only_its_names(toys):
    arith, _ = toys
    with pytest.raises(ValueError):
        arith("toy.mul")
    with pytest.raises(ValueError, match="covers no qualname"):
        class Empty(rules.Rule):
            pass
        rules.register_rule(Empty)


def test_the_visitor_dispatches_on_kind_and_refuses_by_name_otherwise(toys):
    add, length = rules.get_rule("toy.add"), rules.get_rule("toy.len")
    assert add.accept(EngineA(), ctx=7) == ("arith", "toy.add", 7)
    refused = add.accept(EngineB())
    assert refused == rules.Refused("engine-b", "toy.add", "engine-b has no visit for rule kind 'toy_arith'")
    assert "toy.add" in str(refused) and "engine-b" in str(refused)
    na = length.accept(EngineA())                       # declared not applicable: the refusal carries the reason
    assert isinstance(na, rules.Refused) and na.reason == "has no length to speak of"


def test_which_engine_consumes_which_rule_is_static(toys):
    add, sub, length = (rules.get_rule(q) for q in ("toy.add", "toy.sub", "toy.len"))
    assert EngineA.implements() == frozenset({"toy_arith"}) and EngineB.implements() == frozenset()
    assert [EngineA.cell(r) for r in (add, sub, length)] == ["C", "C", "-"]
    assert [EngineB.cell(r) for r in (add, sub, length)] == ["R", "R", "R"]
    assert rules.Visitor.implements() == frozenset()


def test_a_rule_without_a_verified_shape_applies_to_no_call(toys):
    """The CALL_SHAPE discipline: absent from the table = none of the precise facts, not "any arity"."""
    class Unverified(rules.Rule):
        names = ("toy.unverified",)
        CONTRACT = toy_contract("toy.unverified")

    rules.register_rule(Unverified)
    rule = rules.get_rule("toy.unverified")
    assert rules.Rule.SHAPE is None and rule.shape is None
    assert not any(rule.applicable(n, kw) for n in range(0, 6) for kw in (False, True))
    assert rules.get_rule("toy.add").applicable(2)                    # a rule that names a shape still applies


def test_shape_and_applicability(toys):
    add = rules.get_rule("toy.add")
    assert add.applicable(2) and not add.applicable(1) and not add.applicable(3) and not add.applicable(2, True)
    assert rules.Shape(1, None, keywords_ok=True).accepts(5, True)


def test_the_claims_are_read_through_the_tested_gate(toys):
    add, length = rules.get_rule("toy.add"), rules.get_rule("toy.len")
    assert add.never_raises("IndexError") and not add.may_raise("IndexError")
    assert length.may_raise("IndexError") and length.raise_condition("IndexError") is None   # untested: no claim


def test_native_semantics_and_test_support(toys):
    add, sub = rules.get_rule("toy.add"), rules.get_rule("toy.sub")
    assert (add.native(5, 3), sub.native(5, 3)) == (8, 2)
    assert type(add).sample("toy.add") == "toy.add(1, 2)" and type(add).twin("toy.add") is None


@pytest.mark.parametrize("qualname,args,expected", [
    ("len", ([1, 2, 3],), 3), ("str.split", ("a,b", ","), ["a", "b"]), ("math.sqrt", (9.0,), 3.0),
    ("re.match", ("a", "ab"), None),
])
def test_the_default_native_is_cpython_itself(qualname, args, expected):
    out = rules.resolve_native(qualname)(*args)
    assert (out.group() == "a") if qualname == "re.match" else out == expected


def test_an_unreachable_qualname_has_no_native():
    for q in ("nonesuch", "str.nonesuch", "nomodule_xyz.f", "protocol.length_hint"):
        with pytest.raises(KeyError):
            rules.resolve_native(q)


def test_the_kind_vocabulary_lists_what_a_visitor_can_implement(toys):
    assert {"toy_arith", "toy_length"} <= set(rules.kinds())
