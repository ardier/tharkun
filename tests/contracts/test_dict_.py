"""Tests backing every claim in ``tharkun.solve.contracts.rules_dict`` ([internal ref])."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


def _hash_raiser(exc_name):
    """A hashable-looking key whose __hash__ raises."""
    return raiser("__hash__", exc_name)


class TestGetItem:
    contract = get("dict.__getitem__")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.dictionaries(st.text(), st.integers(), min_size=1))
    def test_present_key_returns_value(self, d):
        k = next(iter(d))
        assert d[k] == d[k]

    def test_missing_key_raises_keyerror(self):
        with pytest.raises(KeyError):
            {"a": 1}["x"]

    def test_unhashable_key_raises_typeerror(self):
        with pytest.raises(TypeError):
            {}[[]]
        with pytest.raises(TypeError):
            {}[{}]

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_hash_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            {"a": 1}[_hash_raiser(exc_name)()]

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_eq_propagation_on_collision(self, exc_name):
        Bad = raiser("__eq__", exc_name, nargs=1)
        Bad.__hash__ = lambda self: hash("a")  # force a collision with the real key "a"
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            {"a": 1}[Bad()]


class TestSetItem:
    contract = get("dict.__setitem__")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_sets_and_overwrites(self):
        d = {}
        d["a"] = 1
        d["a"] = 2
        assert d == {"a": 2}

    def test_unhashable_key_raises_typeerror(self):
        with pytest.raises(TypeError):
            {}.__setitem__([], 1)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_hash_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            {}[_hash_raiser(exc_name)()] = 1


class TestGet:
    contract = get("dict.get")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_present_key(self):
        assert {"a": 1}.get("a") == 1

    def test_missing_key_returns_none_or_default(self):
        assert {"a": 1}.get("x") is None
        assert {"a": 1}.get("x", 5) == 5

    def test_never_raises_keyerror(self):
        {}.get("anything")  # must not raise

    def test_unhashable_key_raises_typeerror(self):
        with pytest.raises(TypeError):
            {}.get([])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_hash_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            {}.get(_hash_raiser(exc_name)())


class TestPop:
    contract = get("dict.pop")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_removes_and_returns(self):
        d = {"a": 1}
        assert d.pop("a") == 1
        assert d == {}

    def test_missing_key_no_default_raises_keyerror(self):
        with pytest.raises(KeyError):
            {}.pop("x")

    def test_missing_key_with_default_returns_default(self):
        assert {}.pop("x", 5) == 5

    def test_unhashable_key_raises_typeerror_when_receiver_nonempty(self):
        with pytest.raises(TypeError):
            {"a": 1}.pop([])
        with pytest.raises(TypeError):
            {"a": 1}.pop([], 5)  # a default does not suppress the hashing attempt

    def test_empty_receiver_skips_hashing_entirely(self):
        # SURPRISING (confirmed on 3.14.4): pop() on an EMPTY dict never hashes the
        # key, so even an unhashable key hits the "not found" path, not TypeError.
        with pytest.raises(KeyError):
            {}.pop([])
        assert {}.pop([], 5) == 5

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_hash_propagation_when_receiver_nonempty(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            {"a": 1}.pop(_hash_raiser(exc_name)())


class TestPopitem:
    contract = get("dict.popitem")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_pops_last_inserted_lifo(self):
        d = {"a": 1, "b": 2}
        assert d.popitem() == ("b", 2)
        assert d == {"a": 1}

    def test_empty_raises_keyerror(self):
        with pytest.raises(KeyError):
            {}.popitem()


class TestSetdefault:
    contract = get("dict.setdefault")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_existing_key_unchanged(self):
        d = {"a": 1}
        assert d.setdefault("a", 5) == 1
        assert d == {"a": 1}

    def test_missing_key_inserts_default(self):
        d = {}
        assert d.setdefault("x", 5) == 5
        assert d == {"x": 5}

    def test_missing_key_no_default_uses_none(self):
        d = {}
        assert d.setdefault("x") is None
        assert d == {"x": None}

    def test_unhashable_key_raises_typeerror(self):
        with pytest.raises(TypeError):
            {}.setdefault([], 1)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_hash_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            {}.setdefault(_hash_raiser(exc_name)(), 1)


class TestUpdate:
    contract = get("dict.update")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_from_mapping(self):
        d = {"a": 1}
        d.update({"b": 2})
        assert d == {"a": 1, "b": 2}

    def test_from_pairs(self):
        d = {}
        d.update([("a", 1), ("b", 2)])
        assert d == {"a": 1, "b": 2}

    def test_from_kwargs(self):
        d = {}
        d.update(a=1)
        assert d == {"a": 1}

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            {}.update(5)

    def test_wrong_pair_length_raises_valueerror(self):
        with pytest.raises(ValueError):
            {}.update([("a",)])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_keys_method_propagation(self, exc_name):
        class MapLike:
            def keys(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            {}.update(MapLike())


class TestKeysValuesItems:
    def test_contracts_are_tested(self):
        assert get("dict.keys").tested
        assert get("dict.values").tested
        assert get("dict.items").tested

    def test_views_reflect_receiver(self):
        d = {"a": 1, "b": 2}
        assert list(d.keys()) == ["a", "b"]
        assert list(d.values()) == [1, 2]
        assert list(d.items()) == [("a", 1), ("b", 2)]

    def test_views_are_live(self):
        d = {"a": 1}
        keys = d.keys()
        d["b"] = 2
        assert list(keys) == ["a", "b"]

    def test_empty_dict_views_never_raise(self):
        {}.keys()
        {}.values()
        {}.items()


class TestClearCopy:
    def test_contracts_are_tested(self):
        assert get("dict.clear").tested
        assert get("dict.copy").tested

    def test_clear_empties(self):
        d = {"a": 1}
        assert d.clear() is None
        assert d == {}

    def test_copy_is_shallow(self):
        inner = [1]
        d = {"a": inner}
        d2 = d.copy()
        assert d2 == d
        assert d2 is not d
        assert d2["a"] is inner
