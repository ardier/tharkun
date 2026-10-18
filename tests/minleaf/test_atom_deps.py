"""Phase 5 gate: 12 tests for atom_deps.py."""

import ast

from tharkun.minleaf.atom_deps import atom_dependencies
from tharkun.minleaf.types import LeafRef, StateIndex


def _parse_expr(code: str) -> ast.expr:
    return ast.parse(code, mode="eval").body


def _noop_getter():
    return None


def _noop_setter(v):
    pass


def _make_si(leaf_paths, roots=None):
    """Build a minimal StateIndex with given leaf paths."""
    leaves = {}
    for p in leaf_paths:
        leaves[p] = LeafRef(
            path=p, typ=int, getter=_noop_getter, setter=_noop_setter
        )
    return StateIndex(leaves=leaves, bytes_policy="per_byte", roots=roots or {})


def test_simple_name_leaf():
    si = _make_si([("x",)])
    deps = atom_dependencies(_parse_expr("x > 0"), si)
    assert deps.leaf_paths == frozenset({("x",)})
    assert deps.structural is False


def test_nested_attr_leaf():
    si = _make_si([("user", "profile", "age")])
    deps = atom_dependencies(_parse_expr("user.profile.age >= 18"), si)
    assert deps.leaf_paths == frozenset({("user", "profile", "age")})


def test_subscript_leaf():
    si = _make_si([("data", 3)])
    deps = atom_dependencies(_parse_expr("data[3] == 0xFF"), si)
    assert deps.leaf_paths == frozenset({("data", 3)})


def test_binop_multi():
    si = _make_si([("x",), ("y",)])
    deps = atom_dependencies(_parse_expr("x + y > 10"), si)
    assert deps.leaf_paths == frozenset({("x",), ("y",)})


def test_call_arg():
    si = _make_si([("x",)])
    deps = atom_dependencies(_parse_expr("foo(x)"), si)
    assert deps.leaf_paths == frozenset({("x",)})
    assert deps.structural is False


def test_C1_truthiness():
    # ("flags",) is NOT a leaf; ("flags",0) and ("flags",1) are leaves
    si = _make_si([("flags", 0), ("flags", 1)])
    deps = atom_dependencies(_parse_expr("flags"), si)
    assert deps.leaf_paths == frozenset()
    assert deps.container_paths == frozenset({("flags",)})
    assert deps.structural is True
    assert deps.structural_reason == "truthiness"


def test_C2_len():
    # The full atom is "len(xs) > 3" (a Compare).
    # ("xs",) is NOT a leaf; ("xs",0)..("xs",2) are leaves.
    si = _make_si([("xs", 0), ("xs", 1), ("xs", 2)])
    deps = atom_dependencies(_parse_expr("len(xs) > 3"), si)
    assert deps.container_paths == frozenset({("xs",)})
    assert deps.structural_reason == "len"


def test_C3_membership():
    # x is a leaf, big_set has children (is a reachable container)
    si = _make_si([("x",), ("big_set", 0)])
    deps = atom_dependencies(_parse_expr("x in big_set"), si)
    assert deps.leaf_paths == frozenset({("x",)})
    assert deps.container_paths == frozenset({("big_set",)})
    assert deps.structural_reason == "membership_container"


def test_C4_dynamic_index():
    # data[i] where i is a Name (non-constant), data has children
    si = _make_si([("data", 0), ("data", 1), ("data", 2), ("data", 3)])
    deps = atom_dependencies(_parse_expr("data[i] == 0xFF"), si)
    assert deps.container_paths == frozenset({("data",)})
    assert deps.structural_reason == "dynamic_index"


def test_C5_any():
    si = _make_si([("flags", 0), ("flags", 1), ("flags", 2)])
    deps = atom_dependencies(_parse_expr("any(flags)"), si)
    assert deps.container_paths == frozenset({("flags",)})
    assert deps.structural_reason == "iteration_helper"


def test_unresolved_not_container():
    # "user.profile" is not a leaf and not in a C1-C5 context
    # (it's an argument to a non-C5 function call)
    si = _make_si([("user", "profile", "age")])
    deps = atom_dependencies(_parse_expr("foo(user.profile)"), si)
    assert deps.container_paths == frozenset()


def test_no_state_match():
    si = _make_si([])
    deps = atom_dependencies(_parse_expr("z > 0"), si)
    assert deps.leaf_paths == frozenset()
    assert deps.container_paths == frozenset()
    assert deps.structural is False
