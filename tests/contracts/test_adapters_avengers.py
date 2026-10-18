"""Tests for the Engine 5 (Avengers/object construction) adapter ([internal ref] step 2)."""

from __future__ import annotations

import os

import pytest

from tharkun.avengers import MethodCall, ObjectGoal, synthesize_object
from tharkun.solve.contracts.adapters.avengers import nonempty_seed_for_type

FIXTURE = os.path.realpath(
    os.path.join(os.path.dirname(__file__), "..", "fixtures", "avengers_heap_target.py")
)


class TestNonemptySeedForType:
    def test_list_worked_example(self):
        # list.pop is TESTED to raise IndexError when empty -- real evidence.
        assert nonempty_seed_for_type("list") == [0]

    def test_dict_is_backed_by_evidence_too(self):
        assert nonempty_seed_for_type("dict") == {0: 0}

    def test_a_type_not_in_the_seed_table_yields_none(self):
        assert nonempty_seed_for_type("complex") is None

    def test_negative_control_a_tabled_type_with_no_contract_evidence_yields_none(self):
        # "bytearray" IS in the seed table, but has ZERO contracts registered
        # (batch 1-3 never contracted it) -- so there is no TESTED evidence its
        # methods ever raise on empty, and none must be manufactured.
        assert nonempty_seed_for_type("bytearray") is None


class TestConstructorBackedSeed:
    """[internal ref] batch 4: the one-element ``list``/``tuple``/``set`` seed is built by the contracted
    constructor, so it is offered only while that contract is TESTED."""

    @pytest.mark.parametrize("name,seed", [("list", [0]), ("tuple", (0,)), ("set", {0})])
    def test_the_seed_comes_from_the_constructor(self, name, seed):
        got = nonempty_seed_for_type(name)
        assert got == seed and type(got) is type(seed)

    @pytest.mark.parametrize("name", ["list", "set"])
    def test_every_call_returns_a_new_container(self, name):
        assert nonempty_seed_for_type(name) is not nonempty_seed_for_type(name)

    @pytest.mark.parametrize("name", ["list", "tuple", "set"])
    def test_withdrawing_or_untesting_the_constructor_contract_withdraws_the_seed(self, monkeypatch, name):
        import dataclasses

        from tharkun.solve.contracts.registry import REGISTRY, load_all

        load_all()
        monkeypatch.setitem(REGISTRY, name, dataclasses.replace(REGISTRY[name], tested=False))
        assert nonempty_seed_for_type(name) is None
        monkeypatch.delitem(REGISTRY, name)
        assert nonempty_seed_for_type(name) is None


class TestAvengersWiring:
    """``MinHeap.pop()`` (tests/fixtures/avengers_heap_target.py) is backed
    directly by ``heapq.heappop`` -- the worked example, end to end through
    REAL fresh-process replay (``synthesize_object``), not just a proposed
    recipe."""

    def test_a_contract_backed_seed_lets_pop_succeed(self):
        goal = ObjectGoal(FIXTURE, "MinHeap", (MethodCall("pop"),))
        report = synthesize_object(goal, max_setup_steps=1)
        assert report.best is not None
        assert dict(report.best.recipe.constructor.kwargs).get("data") == [0]
        assert report.best.outcome.succeeded

    def test_negative_control_the_baseline_candidate_is_still_offered(self):
        # The contract-backed seed must ADD a candidate, never remove the
        # pre-existing empty baseline: a goal with NO calls (construction
        # alone) is satisfied by the plain empty-list baseline either way.
        goal = ObjectGoal(FIXTURE, "MinHeap")
        report = synthesize_object(goal, max_setup_steps=1)
        assert report.best is not None
