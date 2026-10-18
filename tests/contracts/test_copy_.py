"""Tests backing every claim in ``tharkun.solve.contracts.rules_copy`` ([internal ref] batch 3)."""

from __future__ import annotations

import copy
import threading

import pytest

from tests.contracts._propagation import PROPERTY_CLASSES
from tharkun.solve.contracts import get


class TestCopy:
    contract = get("copy.copy")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @pytest.mark.parametrize("x", [5, "s", (1, 2), range(3), frozenset({1}), None])
    def test_atomic_types_returned_unchanged(self, x):
        assert copy.copy(x) is x

    def test_list_dict_set_use_their_own_copy(self):
        xs = [1, 2]
        ys = copy.copy(xs)
        assert ys == xs
        assert ys is not xs

    def test_plain_object_is_copied_not_identical(self):
        class Plain:
            def __init__(self):
                self.x = 5

        p = Plain()
        p2 = copy.copy(p)
        assert p2 is not p
        assert p2.x == 5

    def test_shallow_copy_shares_nested_mutables(self):
        inner = [1]
        outer = [inner]
        outer2 = copy.copy(outer)
        assert outer2 is not outer
        assert outer2[0] is inner  # shallow: nested list is SHARED

    def test_uses_dunder_copy_when_present(self):
        class HasCopy:
            def __copy__(self):
                return "custom-copy-result"

        assert copy.copy(HasCopy()) == "custom-copy-result"

    def test_unpicklable_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            copy.copy(threading.Lock())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_dunder_copy_propagation(self, exc_name):
        class Bad:
            def __copy__(self):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            copy.copy(Bad())

    def test_may_raise_other_is_true(self):
        # REVIEW FIX: __copy__/__reduce_ex__/__reduce__ is arbitrary user code -- it
        # can raise ANY exception, not just one of the 7 property classes.
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_dunder_copy_propagates_non_property_exception(self):
        class Bad:
            def __copy__(self):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            copy.copy(Bad())


class TestDeepcopy:
    contract = get("copy.deepcopy")

    def test_contract_is_tested(self):
        assert self.contract.tested

    @pytest.mark.parametrize("x", [5, "s", (1, 2), range(3), None])
    def test_atomic_types_returned_unchanged(self, x):
        assert copy.deepcopy(x) is x

    def test_nested_mutables_are_recursively_copied(self):
        inner = [1]
        outer = [inner]
        outer2 = copy.deepcopy(outer)
        assert outer2 is not outer
        assert outer2[0] is not inner  # DEEP: nested list is its own copy
        assert outer2[0] == inner

    def test_self_cycle_does_not_infinite_recurse(self):
        class Node:
            pass

        n = Node()
        n.next = n
        n2 = copy.deepcopy(n)
        assert n2.next is n2  # the copy's cycle points to ITSELF, not the original

    def test_shared_reference_stays_shared_in_the_copy(self):
        shared = [1]
        pair = [shared, shared]
        pair2 = copy.deepcopy(pair)
        assert pair2[0] is pair2[1]  # memo preserves the sharing structure

    def test_uses_dunder_deepcopy_when_present(self):
        class HasDeepcopy:
            def __deepcopy__(self, memo):
                return "custom-deepcopy-result"

        assert copy.deepcopy(HasDeepcopy()) == "custom-deepcopy-result"

    def test_unpicklable_type_raises_typeerror(self):
        with pytest.raises(TypeError):
            copy.deepcopy(threading.Lock())

    @pytest.mark.parametrize("exc_name", sorted(PROPERTY_CLASSES))
    def test_dunder_deepcopy_propagation(self, exc_name):
        class Bad:
            def __deepcopy__(self, memo):
                raise PROPERTY_CLASSES[exc_name]("boom")

        with pytest.raises(PROPERTY_CLASSES[exc_name]):
            copy.deepcopy(Bad())

    def test_may_raise_other_is_true(self):
        assert self.contract.may_raise_other is True

    def test_effects_flags_may_execute_callback(self):
        assert self.contract.effects.may_execute_callback is True

    def test_dunder_deepcopy_propagates_non_property_exception(self):
        class Bad:
            def __deepcopy__(self, memo):
                raise ValueError("boom")

        with pytest.raises(ValueError):
            copy.deepcopy(Bad())
