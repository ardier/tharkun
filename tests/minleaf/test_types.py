"""Phase 0 gate: 6 tests for types.py."""

from tharkun.minleaf.types import (
    LeafFacts,
    LeafFactsIndex,
    Patch,
    SetOp,
)


def test_patch_empty():
    p = Patch(ops=())
    assert p.cost() == 0
    assert p.touched_paths() == frozenset()


def test_patch_cost_dedup():
    p = Patch(ops=(SetOp(("x",), 0, 1), SetOp(("x",), 1, 2)))
    assert p.cost() == 1


def test_patch_cost_distinct():
    p = Patch(ops=(SetOp(("a",), 0, 1), SetOp(("b",), 0, 1), SetOp(("c",), 0, 1)))
    assert p.cost() == 3


def test_leaf_facts_index_missing():
    idx = LeafFactsIndex(facts={})
    assert idx.get(("x",)) is None
    assert idx.is_excluded(("x",), 42) is False
    assert idx.in_bounds(("x",), 999) is True


def test_leaf_facts_index_neq():
    idx = LeafFactsIndex(facts={("x",): LeafFacts(neq=frozenset({42, 99}))})
    assert idx.is_excluded(("x",), 42) is True
    assert idx.is_excluded(("x",), 43) is False


def test_leaf_facts_index_bounds():
    idx = LeafFactsIndex(facts={("x",): LeafFacts(lo=0, hi=100)})
    assert idx.in_bounds(("x",), 50) is True
    assert idx.in_bounds(("x",), -1) is False
    assert idx.in_bounds(("x",), 101) is False
    assert idx.in_bounds(("x",), 0) is True    # lo inclusive
    assert idx.in_bounds(("x",), 100) is True   # hi inclusive
