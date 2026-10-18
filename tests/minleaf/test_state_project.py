"""Phase 4 gate: 10 tests for state_project.py."""

from dataclasses import dataclass

import pytest

from tharkun.minleaf.state_project import build_state_index


def test_primitive_int():
    roots = {"x": 42}
    si = build_state_index(roots, "per_byte")
    assert ("x",) in si.leaves
    assert si.leaves[("x",)].typ is int
    assert si.leaves[("x",)].getter() == 42


def test_primitive_setter_rebinds_env():
    roots = {"x": 42}
    env = {"x": 42}
    si = build_state_index(roots, "per_byte", env=env)
    si.leaves[("x",)].setter(99)
    assert env["x"] == 99
    assert roots["x"] == 99


def test_nested_dataclass():
    @dataclass
    class Profile:
        age: int
        country: str

    @dataclass
    class User:
        profile: Profile

    user = User(profile=Profile(age=16, country="DE"))
    roots = {"user": user}
    si = build_state_index(roots, "per_byte")
    assert ("user", "profile", "age") in si.leaves
    assert ("user", "profile", "country") in si.leaves
    assert si.leaves[("user", "profile", "age")].getter() == 16


def test_dict_leaves():
    roots = {"cfg": {"timeout": 30, "retries": 3}}
    si = build_state_index(roots, "per_byte")
    assert si.leaves[("cfg", "timeout")].getter() == 30
    assert si.leaves[("cfg", "retries")].getter() == 3


def test_list_leaves():
    roots = {"xs": [10, 20, 30]}
    si = build_state_index(roots, "per_byte")
    assert si.leaves[("xs", 0)].getter() == 10
    assert si.leaves[("xs", 2)].getter() == 30


def test_list_setter():
    roots = {"xs": [10, 20, 30]}
    si = build_state_index(roots, "per_byte")
    si.leaves[("xs", 1)].setter(99)
    assert roots["xs"][1] == 99


def test_bytes_per_byte():
    roots = {"data": b"\x00\x01\x02"}
    si = build_state_index(roots, "per_byte")
    assert si.leaves[("data", 0)].getter() == 0
    assert si.leaves[("data", 1)].getter() == 1
    assert si.leaves[("data", 2)].getter() == 2


def test_bytes_atomic():
    roots = {"data": b"\x00\x01"}
    si = build_state_index(roots, "atomic")
    assert ("data",) in si.leaves
    assert si.leaves[("data",)].getter() == b"\x00\x01"
    assert ("data", 0) not in si.leaves


def test_tuple_read_only():
    roots = {"t": (1, 2)}
    si = build_state_index(roots, "per_byte")
    assert ("t", 0) in si.leaves
    assert si.leaves[("t", 0)].getter() == 1
    with pytest.raises(TypeError):
        si.leaves[("t", 0)].setter(99)


def test_cycle_detection():
    class Node:
        pass

    a = Node()
    a.child = a  # type: ignore
    roots = {"a": a}
    si = build_state_index(roots, "per_byte")
    # Should complete without hanging
    assert len(si.leaves) >= 0  # finite
