"""Tests for the Engine 4 (fuzzer/draws) adapter ([internal ref] step 2)."""

from __future__ import annotations

import heapq

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import find, strategies as st  # noqa: E402

from tharkun.hypothesis_adapter import _strategy_for_type  # noqa: E402
from tharkun.solve.contracts.adapters.draws import (  # noqa: E402
    boundary_value_for_type, heap_ordered, heap_ordered_supported, range_kwargs,
)


class TestBoundaryValueForType:
    def test_list_worked_example(self):
        # list.pop is TESTED to raise IndexError when empty -- real evidence.
        assert boundary_value_for_type("list") == []

    def test_dict_is_backed_by_evidence_too(self):
        assert boundary_value_for_type("dict") == {}

    def test_a_type_with_no_receiver_contracts_yields_none(self):
        assert boundary_value_for_type("complex") is None

    def test_a_type_not_in_the_empty_value_table_yields_none(self):
        assert boundary_value_for_type("int") is None


class TestConstructorBackedBoundary:
    """[internal ref] batch 4: the empty ``list``/``tuple``/``set`` is built by calling the contracted
    constructor with no argument, so it is offered only while that contract is TESTED."""

    @pytest.mark.parametrize("name,empty", [("list", []), ("tuple", ()), ("set", set())])
    def test_the_empty_container_comes_from_the_constructor(self, name, empty):
        got = boundary_value_for_type(name)
        assert got == empty and type(got) is type(empty)

    @pytest.mark.parametrize("name", ["list", "set"])
    def test_every_call_returns_a_new_container(self, name):
        a, b = boundary_value_for_type(name), boundary_value_for_type(name)
        assert a is not b

    @pytest.mark.parametrize("name", ["list", "tuple", "set"])
    def test_withdrawing_or_untesting_the_constructor_contract_withdraws_the_value(self, monkeypatch, name):
        import dataclasses

        from tharkun.solve.contracts.registry import REGISTRY, load_all

        load_all()
        monkeypatch.setitem(REGISTRY, name, dataclasses.replace(REGISTRY[name], tested=False))
        assert boundary_value_for_type(name) is None
        monkeypatch.delitem(REGISTRY, name)
        assert boundary_value_for_type(name) is None

    def test_a_type_without_a_constructor_contract_is_unaffected(self, monkeypatch):
        from tharkun.solve.contracts.registry import REGISTRY, load_all

        load_all()
        monkeypatch.delitem(REGISTRY, "list")
        monkeypatch.delitem(REGISTRY, "set")
        assert boundary_value_for_type("dict") == {}


class TestHeapOrdered:
    def test_worked_example_is_supported(self):
        assert heap_ordered_supported() is True

    def test_heap_ordered_actually_uses_the_real_heapq(self):
        out = heap_ordered([5, 3, 8, 1, 9, 2])
        assert out[0] == min(out)
        # popping every element in order recovers the heap invariant fully.
        h = list(out)
        popped = [heapq.heappop(h) for _ in range(len(h))]
        assert popped == sorted([5, 3, 8, 1, 9, 2])


class TestRangeKwargs:
    def test_random_random_worked_example(self):
        assert range_kwargs("random.random") == (0.0, 1.0, False)

    @pytest.mark.parametrize("qualname", ["list", "set", "tuple", "heapq.heapify", "list.__getitem__", "dict.__getitem__", "dict.pop", "dict.popitem", "list.pop", "tuple.__getitem__", "str.__getitem__", "str.index", "str.rindex", "bytes.__getitem__", "set.pop"])
    def test_non_range_rules_do_not_leak_boolean_visit_results(self, qualname):
        assert range_kwargs(qualname) is None

    def test_unknown_qualname_is_none(self):
        assert range_kwargs("random.randint") is None


class TestHypothesisAdapterWiring:
    """2026-09-28 owner direction: no feature flag -- this engine is always
    on (it only ever ADDS a candidate value to a strategy's domain, never
    trusted as a proof premise, see ``draws.boundary_value_for_type``'s own
    docstring)."""

    def test_boundary_value_is_reliably_reachable(self):
        strat = _strategy_for_type("list")
        # st.just([]) inside the one_of means find() must succeed immediately.
        result = find(strat, lambda v: v == [])
        assert result == []

    def test_negative_control_a_type_without_evidence_is_not_augmented(self):
        # "int" has no boundary_value_for_type entry -- the hook must be a no-op,
        # so the strategy must be unaffected (still produces plain ints, not
        # wrapped in an unrelated one_of that could change its shape).
        strat = _strategy_for_type("int")
        v = find(strat, lambda _: True)
        assert isinstance(v, int)
