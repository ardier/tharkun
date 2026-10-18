"""Tests for the contract data model itself ([internal ref]): the soundness invariants that
``Contract.__post_init__`` and friends enforce, independent of any one callable.
"""

from __future__ import annotations

import pytest

from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseClaim, ReturnSpec


def _claims(**overrides):
    base = {exc: "never" for exc in ALL_PROPERTY_CLASSES}
    base.update(overrides)
    return tuple(RaiseClaim(exc, cond) for exc, cond in base.items())


def test_all_property_classes_is_the_sv_comp_seven():
    assert set(ALL_PROPERTY_CLASSES) == {
        "TypeError", "AttributeError", "IndexError", "KeyError",
        "ZeroDivisionError", "FloatingPointError", "AssertionError",
    }


def test_raise_claim_rejects_non_property_class():
    with pytest.raises(ValueError):
        RaiseClaim("ValueError", "always")


def test_raise_claim_rejects_empty_condition():
    with pytest.raises(ValueError):
        RaiseClaim("TypeError", "")


def test_raise_claim_unknown_cannot_be_tested():
    with pytest.raises(ValueError):
        RaiseClaim("TypeError", "unknown", tested=True)


def test_contract_requires_every_property_class():
    incomplete = tuple(c for c in _claims() if c.exc != "KeyError")
    with pytest.raises(ValueError, match="missing raise claims"):
        Contract("x", incomplete, True, None, Effects(), source=("doc",))


def test_contract_rejects_duplicate_claims():
    dup = _claims() + (RaiseClaim("TypeError", "never"),)
    with pytest.raises(ValueError, match="duplicate"):
        Contract("x", dup, True, None, Effects(), source=("doc",))


def test_contract_requires_source():
    with pytest.raises(ValueError, match="source"):
        Contract("x", _claims(), True, None, Effects(), source=())


def test_contract_tested_true_requires_every_non_unknown_claim_tested():
    claims = _claims(TypeError="raised when x is bad")  # tested defaults False
    with pytest.raises(ValueError, match="untested"):
        Contract("x", claims, True, None, Effects(), source=("doc",), tested=True)


def test_contract_tested_true_ok_when_claims_and_return_are_tested():
    claims = tuple(
        RaiseClaim(c.exc, c.condition, tested=True) for c in _claims(TypeError="raised when x is bad")
    )
    contract = Contract(
        "x", claims, True, ReturnSpec("int", tested=True), Effects(), source=("doc",), tested=True,
    )
    assert contract.tested


def test_contract_tested_true_exempts_unknown_claims():
    claims = tuple(
        RaiseClaim(c.exc, c.condition, tested=(c.exc != "AttributeError"))
        for c in _claims(AttributeError="unknown")
    )
    # AttributeError is "unknown" and untested; every OTHER claim is tested="never"
    # claims default to condition "never" with tested=True below except AttributeError.
    claims = tuple(
        RaiseClaim(c.exc, c.condition, tested=(c.condition != "unknown")) for c in claims
    )
    contract = Contract("x", claims, True, None, Effects(), source=("doc",), tested=True)
    assert contract.tested


def test_contract_raise_claim_lookup():
    contract = Contract("x", _claims(), True, None, Effects(), source=("doc",))
    assert contract.raise_claim("TypeError").condition == "never"
    with pytest.raises(KeyError):
        contract.raise_claim("ValueError")


def test_returnspec_and_effects_defaults():
    r = ReturnSpec("int")
    assert r.shape == "" and r.tested is False
    e = Effects()
    assert e.mutates_receiver is False and e.mutates_args == () and e.deterministic is True
