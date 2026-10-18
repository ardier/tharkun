"""Phase 6 gate: 10 tests for leaf_facts.py."""

from tharkun.minleaf.types import FactEnv, VarFacts

from tharkun.minleaf.leaf_facts import (
    build_leaf_facts_index,
    merge_leaf_facts,
)
from tharkun.minleaf.types import LeafFacts, LeafRef, StateIndex


def _noop_getter():
    return None


def _noop_setter(v):
    pass


def _make_si(leaf_paths):
    leaves = {}
    for p in leaf_paths:
        leaves[p] = LeafRef(path=p, typ=int, getter=_noop_getter, setter=_noop_setter)
    return StateIndex(leaves=leaves, bytes_policy="per_byte", roots={})


def test_F1_eq():
    fe = FactEnv(facts={"x": VarFacts(eq={42})})
    si = _make_si([("x",)])
    idx = build_leaf_facts_index(fe, si)
    assert idx.get(("x",)).eq == frozenset({42})


def test_F1_bounds():
    fe = FactEnv(facts={"x": VarFacts(lo=0, hi=100)})
    si = _make_si([("x",)])
    idx = build_leaf_facts_index(fe, si)
    assert idx.get(("x",)).lo == 0
    assert idx.get(("x",)).hi == 100


def test_F1_neq():
    fe = FactEnv(facts={"x": VarFacts(neq={7, 13})})
    si = _make_si([("x",)])
    idx = build_leaf_facts_index(fe, si)
    assert idx.get(("x",)).neq == frozenset({7, 13})


def test_F2_dotted():
    fe = FactEnv(facts={"user.profile.age": VarFacts(lo=0, hi=100)})
    si = _make_si([("user", "profile", "age")])
    idx = build_leaf_facts_index(fe, si)
    assert idx.get(("user", "profile", "age")).lo == 0


def test_F2_int_segment():
    fe = FactEnv(facts={"data.3": VarFacts(eq={255})})
    si = _make_si([("data", 3)])
    idx = build_leaf_facts_index(fe, si)
    assert idx.get(("data", 3)).eq == frozenset({255})


def test_F3_container_dropped():
    # "user" is container root (not a leaf); ("user","profile","age") is a leaf
    fe = FactEnv(facts={"user": VarFacts(lo=0)})
    si = _make_si([("user", "profile", "age")])
    idx = build_leaf_facts_index(fe, si)
    assert idx.get(("user", "profile", "age")) is None


def test_F3_container_eq_kept():
    # "role" is a leaf (not just a container root)
    fe = FactEnv(facts={"role": VarFacts(eq={"admin"})})
    si = _make_si([("role",)])
    idx = build_leaf_facts_index(fe, si)
    assert idx.get(("role",)).eq == frozenset({"admin"})


def test_F4_infeasible():
    fe = FactEnv(possible=False)
    si = _make_si([])
    idx = build_leaf_facts_index(fe, si)
    assert idx.global_possible is False


def test_F5_none():
    si = _make_si([])
    idx = build_leaf_facts_index(None, si)
    assert idx.facts == {}
    assert idx.global_possible is True


def test_merge():
    a = LeafFacts(eq=frozenset({1, 2, 3}), neq=frozenset({10}), lo=0)
    b = LeafFacts(eq=frozenset({2, 3, 4}), neq=frozenset({20}), hi=100)
    merged = merge_leaf_facts(a, b)
    assert merged.eq == frozenset({2, 3})
    assert merged.neq == frozenset({10, 20})
    assert merged.lo == 0
    assert merged.hi == 100
