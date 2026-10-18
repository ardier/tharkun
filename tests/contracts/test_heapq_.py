"""Tests backing every claim in ``tharkun.solve.contracts.rules_heapq`` ([internal ref] batch 3)."""

from __future__ import annotations

import heapq

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.contracts._propagation import PROPERTY_CLASSES, raiser
from tharkun.solve.contracts import get


class TestHeappush:
    contract = get("heapq.heappush")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_maintains_heap_invariant(self):
        h = []
        for x in [3, 1, 4, 1, 5]:
            heapq.heappush(h, x)
        assert h[0] == min(h)

    def test_single_element_push_never_compares(self):
        # Pushing onto an empty heap needs no comparison at all.
        Bad = raiser("__lt__", "KeyError", nargs=1)
        h = []
        heapq.heappush(h, Bad())  # must not raise

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_lt_propagation_when_comparison_needed(self, exc_name):
        Bad = raiser("__lt__", exc_name, nargs=1)
        h = [Bad(), Bad()]
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            heapq.heappush(h, Bad())

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

        h = [BadLt(), BadLt()]
        with pytest.raises(ValueError):
            heapq.heappush(h, BadLt())


class TestHeappop:
    contract = get("heapq.heappop")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_pops_smallest(self):
        h = [3, 1, 4, 1, 5]
        heapq.heapify(h)
        assert heapq.heappop(h) == 1

    def test_empty_raises_indexerror(self):
        with pytest.raises(IndexError):
            heapq.heappop([])

    @given(st.lists(st.integers(), min_size=1))
    def test_pops_in_ascending_order(self, xs):
        h = list(xs)
        heapq.heapify(h)
        popped = [heapq.heappop(h) for _ in range(len(xs))]
        assert popped == sorted(xs)

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_lt_propagation_when_comparison_needed(self, exc_name):
        Bad = raiser("__lt__", exc_name, nargs=1)
        h = [Bad(), Bad(), Bad()]
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            heapq.heappop(h)

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_lt_propagates_non_property_exception(self):
        class BadLt:
            def __lt__(self, other):
                raise ValueError("boom")

        h = [BadLt(), BadLt(), BadLt()]
        with pytest.raises(ValueError):
            heapq.heappop(h)


class TestHeapify:
    contract = get("heapq.heapify")

    def test_contract_is_tested(self):
        assert self.contract.tested

    def test_returns_none_mutates_in_place(self):
        h = [3, 1, 2]
        assert heapq.heapify(h) is None
        assert h[0] == 1

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_lt_propagation(self, exc_name):
        Bad = raiser("__lt__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            heapq.heapify([Bad(), Bad()])

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_lt_propagates_non_property_exception(self):
        class BadLt:
            def __lt__(self, other):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            heapq.heapify([BadLt(), BadLt()])


class TestNlargestNsmallest:
    def test_contracts_are_tested(self):
        assert get("heapq.nlargest").tested
        assert get("heapq.nsmallest").tested

    def test_basic(self):
        assert heapq.nlargest(2, [1, 2, 3]) == [3, 2]
        assert heapq.nsmallest(2, [1, 2, 3]) == [1, 2]

    def test_does_not_require_heap_order(self):
        assert heapq.nlargest(2, [5, 1, 4, 2]) == [5, 4]

    def test_non_iterable_raises_typeerror(self):
        with pytest.raises(TypeError):
            heapq.nlargest(2, 5)
        with pytest.raises(TypeError):
            heapq.nsmallest(2, 5)

    def test_bad_n_raises_typeerror(self):
        with pytest.raises(TypeError):
            heapq.nlargest("x", [1, 2, 3])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_lt_propagation(self, exc_name):
        Bad = raiser("__lt__", exc_name, nargs=1)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            heapq.nlargest(1, [Bad(), Bad()])
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            heapq.nsmallest(1, [Bad(), Bad()])

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_key_propagation(self, exc_name):
        def bad_key(x):
            raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            heapq.nlargest(1, [1, 2], key=bad_key)
        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            heapq.nsmallest(1, [1, 2], key=bad_key)

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: the key function and __lt__ are arbitrary user code -- either
        # can raise ANY exception, not just one of the 7 property classes.
        assert get("heapq.nlargest").may_raise_other is True
        assert get("heapq.nsmallest").may_raise_other is True

    def test_effects_flag_may_execute_callback(self):
        assert get("heapq.nlargest").effects.may_execute_callback is True
        assert get("heapq.nsmallest").effects.may_execute_callback is True

    def test_key_propagates_non_property_exception(self):
        def bad_key(x):
            raise ValueError("boom")

        with pytest.raises(ValueError):
            heapq.nlargest(1, [1, 2], key=bad_key)
        with pytest.raises(ValueError):
            heapq.nsmallest(1, [1, 2], key=bad_key)
