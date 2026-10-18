"""Tests backing every claim in ``tharkun.solve.contracts.rules_operator`` ([internal ref] batch 3)."""

from __future__ import annotations

import operator

import pytest

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


class TestItemgetter:
    contract = get("operator.itemgetter")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_single_index_returns_bare_value(self):
        assert operator.itemgetter(0)([10, 20]) == 10

    def test_multiple_indices_returns_tuple(self):
        assert operator.itemgetter(0, 2)([10, 20, 30]) == (10, 30)

    def test_no_args_raises_typeerror_at_build_time(self):
        with pytest.raises(TypeError):
            operator.itemgetter()

    def test_dict_missing_key_raises_keyerror(self):
        with pytest.raises(KeyError):
            operator.itemgetter("x")({})

    def test_list_out_of_range_raises_indexerror(self):
        with pytest.raises(IndexError):
            operator.itemgetter(5)([1, 2])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_getitem_propagation(self, exc_name):
        Bad = raiser("__getitem__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            operator.itemgetter(0)(Bad())

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: the target's __getitem__ is arbitrary user code -- it can
        # raise ANY exception, not just one of the 7 property classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_getitem_propagates_non_property_exception(self):
        class Bad:
            def __getitem__(self, key):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            operator.itemgetter(0)(Bad())


class TestAttrgetter:
    contract = get("operator.attrgetter")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_single_name(self):
        class O:
            x = 5

        assert operator.attrgetter("x")(O()) == 5

    def test_dotted_path(self):
        class Inner:
            b = 42

        class Outer:
            a = Inner()

        assert operator.attrgetter("a.b")(Outer()) == 42

    def test_multiple_names_returns_tuple(self):
        class O:
            x, y = 1, 2

        assert operator.attrgetter("x", "y")(O()) == (1, 2)

    def test_no_args_raises_typeerror_at_build_time(self):
        with pytest.raises(TypeError):
            operator.attrgetter()

    def test_missing_attribute_raises_attributeerror(self):
        with pytest.raises(AttributeError):
            operator.attrgetter("x")(object())

    def test_missing_attribute_in_dotted_path_raises_attributeerror(self):
        class Outer:
            a = object()

        with pytest.raises(AttributeError):
            operator.attrgetter("a.b")(Outer())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_getattr_propagation(self, exc_name):
        class Bad:
            def __getattr__(self, name):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            operator.attrgetter("x")(Bad())

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_getattr_propagates_non_property_exception(self):
        class Bad:
            def __getattr__(self, name):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            operator.attrgetter("x")(Bad())


class TestAdd:
    contract = get("operator.add")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_matches_plus_operator(self):
        assert operator.add(1, 2) == 1 + 2

    def test_unsupported_types_raise_typeerror(self):
        with pytest.raises(TypeError):
            operator.add("a", 1)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_add_dunder_propagation(self, exc_name):
        Bad = raiser("__add__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            operator.add(Bad(), 1)

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_add_dunder_propagates_non_property_exception(self):
        class Bad:
            def __add__(self, other):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            operator.add(Bad(), 1)
