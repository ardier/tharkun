"""Tests backing every claim in ``tharkun.solve.contracts.rules_tuple`` ([internal ref])."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


def _eq_raiser(exc_name):
    return raiser("__eq__", exc_name, nargs=1)


class TestIndex:
    contract = get("tuple.index")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.integers(), min_size=1))
    def test_finds_first_occurrence(self, xs):
        t = tuple(xs)
        x = t[0]
        idx = t.index(x)
        assert t[idx] == x

    def test_not_found_raises_valueerror(self):
        with pytest.raises(ValueError):
            (1, 2).index(5)

    def test_bad_start_stop_raises_typeerror(self):
        with pytest.raises(TypeError):
            (1, 2, 3).index(1, "x")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_eq_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            (1, 2).index(_eq_raiser(exc_name)())


class TestCount:
    contract = get("tuple.count")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @given(st.lists(st.integers(min_value=0, max_value=3)))
    def test_matches_manual_count(self, xs):
        t = tuple(xs)
        for v in range(4):
            assert t.count(v) == sum(1 for x in t if x == v)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_eq_propagation(self, exc_name):
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            (1, 2).count(_eq_raiser(exc_name)())


class TestGetItem:
    contract = get("tuple.__getitem__")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_positive_and_negative_index(self):
        t = (1, 2, 3)
        assert t[0] == 1
        assert t[-1] == 3

    def test_out_of_range_raises_indexerror(self):
        with pytest.raises(IndexError):
            (1, 2)[5]

    def test_bad_key_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            (1, 2)["x"]

    def test_slice_out_of_range_never_raises(self):
        assert (1, 2)[10:20] == ()
        assert (1, 2, 3)[-100:100] == (1, 2, 3)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_index_dunder_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            (1, 2)[Bad()]
