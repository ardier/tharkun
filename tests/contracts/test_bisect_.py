"""Tests backing every claim in ``tharkun.solve.contracts.rules_bisect`` ([internal ref] batch 3)."""

from __future__ import annotations

import bisect

import pytest

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


class TestBisectLeft:
    contract = get("bisect.bisect_left")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_basic(self):
        assert bisect.bisect_left([1, 3, 5], 3) == 1

    def test_inserts_before_equal_run(self):
        assert bisect.bisect_left([1, 3, 3, 3, 5], 3) == 1

    def test_differs_from_bisect_right_on_equal_run(self):
        a = [1, 3, 3, 3, 5]
        assert bisect.bisect_left(a, 3) != bisect.bisect_right(a, 3)
        assert bisect.bisect_right(a, 3) == 4

    def test_no_len_raises_typeerror(self):
        with pytest.raises(TypeError):
            bisect.bisect_left(5, 1)

    def test_bad_lo_hi_raises_typeerror(self):
        with pytest.raises(TypeError):
            bisect.bisect_left([1, 2, 3], 1, lo="x")

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_lt_propagation(self, exc_name):
        Bad = raiser("__lt__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            bisect.bisect_left([Bad(), Bad()], Bad())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_lo_index_propagation(self, exc_name):
        Bad = raiser("__index__", exc_name)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            bisect.bisect_left([1, 2, 3], 2, Bad())

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: __lt__ is arbitrary user code -- it can raise ANY exception,
        # not just one of the 7 property classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_lt_propagates_non_property_exception(self):
        class BadLt:
            def __lt__(self, other):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            bisect.bisect_left([BadLt(), BadLt()], BadLt())


class TestInsort:
    contract = get("bisect.insort")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_inserts_in_sorted_position(self):
        a = [1, 3, 5]
        bisect.insort(a, 4)
        assert a == [1, 3, 4, 5]

    def test_is_insort_right_inserts_after_equal_run(self):
        a = [1, 3, 3, 5]
        bisect.insort(a, 3)
        assert a == [1, 3, 3, 3, 5]
        assert a.index(3) == 1  # unchanged existing 3s stay first

    def test_returns_none_mutates_in_place(self):
        a = [1, 2]
        assert bisect.insort(a, 0) is None
        assert a == [0, 1, 2]

    def test_no_len_raises_typeerror(self):
        with pytest.raises(TypeError):
            bisect.insort(5, 1)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_lt_propagation(self, exc_name):
        Bad = raiser("__lt__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            bisect.insort([Bad(), Bad()], Bad())

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_lt_propagates_non_property_exception(self):
        class BadLt:
            def __lt__(self, other):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            bisect.insort([BadLt(), BadLt()], BadLt())
