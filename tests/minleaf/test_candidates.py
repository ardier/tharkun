"""Phase 7 gate: 14 tests for candidates.py."""

import ast

from tharkun.minleaf.candidates import (
    LeafCandidates,
    propose_combined_candidates,
    propose_leaf_candidates,
)
from tharkun.minleaf.types import (
    LeafFacts,
    LeafFactsIndex,
    LeafRef,
    Path,
    StateIndex,
)


def _parse_expr(code: str) -> ast.expr:
    return ast.parse(code, mode="eval").body


def _make_si_with_values(leaf_data):
    """Build a StateIndex with given leaf paths and values.

    leaf_data: list of (path, value) tuples
    """
    leaves = {}
    store = {}
    for p, v in leaf_data:
        store[p] = v

        def _make_accessors(path, st):
            def getter():
                return st[path]
            def setter(val):
                st[path] = val
            return getter, setter

        g, s = _make_accessors(p, store)
        leaves[p] = LeafRef(path=p, typ=type(v), getter=g, setter=s)
    return StateIndex(leaves=leaves, bytes_policy="per_byte", roots={})


def test_eq_pattern():
    # atom "x == 42" is False, want True, x=0
    si = _make_si_with_values([(("x",), 0)])
    result = propose_leaf_candidates(
        atom_expr=_parse_expr("x == 42"),
        atom_value=False,
        dependency_paths=frozenset({("x",)}),
        state_index=si,
        env={"x": 0},
        leaf_facts_index=None,
        max_per_leaf=8,
    )
    assert 42 in result[("x",)].candidates


def test_lt_pattern():
    # atom "x < 10" is False, want True, x=20
    si = _make_si_with_values([(("x",), 20)])
    result = propose_leaf_candidates(
        atom_expr=_parse_expr("x < 10"),
        atom_value=False,
        dependency_paths=frozenset({("x",)}),
        state_index=si,
        env={"x": 20},
        leaf_facts_index=None,
        max_per_leaf=8,
    )
    assert 9 in result[("x",)].candidates


def test_gt_pattern():
    # atom "x > 100" is False, want True, x=50
    si = _make_si_with_values([(("x",), 50)])
    result = propose_leaf_candidates(
        atom_expr=_parse_expr("x > 100"),
        atom_value=False,
        dependency_paths=frozenset({("x",)}),
        state_index=si,
        env={"x": 50},
        leaf_facts_index=None,
        max_per_leaf=8,
    )
    assert 101 in result[("x",)].candidates


def test_gte_pattern():
    # atom "x >= 100" is False, want True, x=50
    si = _make_si_with_values([(("x",), 50)])
    result = propose_leaf_candidates(
        atom_expr=_parse_expr("x >= 100"),
        atom_value=False,
        dependency_paths=frozenset({("x",)}),
        state_index=si,
        env={"x": 50},
        leaf_facts_index=None,
        max_per_leaf=8,
    )
    assert 100 in result[("x",)].candidates


def test_in_pattern():
    # atom "x in (1,2,3)" is False, want True, x=5
    si = _make_si_with_values([(("x",), 5)])
    result = propose_leaf_candidates(
        atom_expr=_parse_expr("x in (1, 2, 3)"),
        atom_value=False,
        dependency_paths=frozenset({("x",)}),
        state_index=si,
        env={"x": 5},
        leaf_facts_index=None,
        max_per_leaf=8,
    )
    assert {1, 2, 3} <= set(result[("x",)].candidates)


def test_notin_pattern():
    # atom "x not in (1,2,3)" is False, want True, x=2
    si = _make_si_with_values([(("x",), 2)])
    result = propose_leaf_candidates(
        atom_expr=_parse_expr("x not in (1, 2, 3)"),
        atom_value=False,
        dependency_paths=frozenset({("x",)}),
        state_index=si,
        env={"x": 2},
        leaf_facts_index=None,
        max_per_leaf=8,
    )
    candidates = result[("x",)].candidates
    assert all(c not in (1, 2, 3) for c in candidates)


def test_modulo_pattern():
    # atom "(x % 7) == 3" is False, want True, x=0
    si = _make_si_with_values([(("x",), 0)])
    result = propose_leaf_candidates(
        atom_expr=_parse_expr("x % 7 == 3"),
        atom_value=False,
        dependency_paths=frozenset({("x",)}),
        state_index=si,
        env={"x": 0},
        leaf_facts_index=None,
        max_per_leaf=8,
    )
    assert 3 in result[("x",)].candidates


def test_E1_combined():
    # Two blocking atoms: "x > 1000" and "x % 7 == 3" sharing leaf ("x",)
    cands = propose_combined_candidates(
        atom_exprs=[
            (_parse_expr("x > 1000"), False),
            (_parse_expr("x % 7 == 3"), False),
        ],
        target_path=("x",),
        state_index=_make_si_with_values([(("x",), 0)]),
        env={"x": 0},
    )
    # Smallest x >= 1001 (from x > 1000) where x % 7 == 3 is 1004
    # (1001 % 7 == 0, so 1001 + 3 = 1004, and 1004 % 7 == 3)
    assert 1004 in cands


def test_fallback_int():
    # atom "foo(x)" (no pattern), x=5, leaf is int
    si = _make_si_with_values([(("x",), 5)])
    result = propose_leaf_candidates(
        atom_expr=_parse_expr("foo(x)"),
        atom_value=False,
        dependency_paths=frozenset({("x",)}),
        state_index=si,
        env={"x": 5, "foo": lambda x: x},
        leaf_facts_index=None,
        max_per_leaf=8,
    )
    cands = set(result[("x",)].candidates)
    assert {0, 1, -1, 6, 4} <= cands


def test_fallback_bool():
    si = _make_si_with_values([(("x",), True)])
    result = propose_leaf_candidates(
        atom_expr=_parse_expr("foo(x)"),
        atom_value=False,
        dependency_paths=frozenset({("x",)}),
        state_index=si,
        env={"x": True, "foo": lambda x: x},
        leaf_facts_index=None,
        max_per_leaf=8,
    )
    assert False in result[("x",)].candidates


def test_fact_injected_first():
    # atom "x == CONST" (Name, not Constant), x=0, facts eq={99}
    si = _make_si_with_values([(("x",), 0)])
    idx = LeafFactsIndex(facts={("x",): LeafFacts(eq=frozenset({99}))})
    result = propose_leaf_candidates(
        atom_expr=_parse_expr("x == CONST"),
        atom_value=False,
        dependency_paths=frozenset({("x",)}),
        state_index=si,
        env={"x": 0},
        leaf_facts_index=idx,
        max_per_leaf=8,
    )
    assert result[("x",)].candidates[0] == 99


def test_fact_neq_prune():
    # atom "x > 0" is False, want True, x=-1, facts neq={1}
    si = _make_si_with_values([(("x",), -1)])
    idx = LeafFactsIndex(facts={("x",): LeafFacts(neq=frozenset({1}))})
    result = propose_leaf_candidates(
        atom_expr=_parse_expr("x > 0"),
        atom_value=False,
        dependency_paths=frozenset({("x",)}),
        state_index=si,
        env={"x": -1},
        leaf_facts_index=idx,
        max_per_leaf=8,
    )
    assert 1 not in result[("x",)].candidates


def test_fact_bounds_clamp():
    # atom "x > 500" is False, want True, x=0, facts hi=50
    si = _make_si_with_values([(("x",), 0)])
    idx = LeafFactsIndex(facts={("x",): LeafFacts(hi=50)})
    result = propose_leaf_candidates(
        atom_expr=_parse_expr("x > 500"),
        atom_value=False,
        dependency_paths=frozenset({("x",)}),
        state_index=si,
        env={"x": 0},
        leaf_facts_index=idx,
        max_per_leaf=8,
    )
    cands = result[("x",)].candidates
    assert 501 not in cands
    assert 50 in cands


def test_dedup_and_exclusion():
    # raw candidates = [42, 42, 5], current_value = 5
    # After dedup: [42, 5]. After exclusion of current (5): [42]
    # We simulate this via an eq-pattern atom
    si = _make_si_with_values([(("x",), 5)])

    # Construct a situation that generates [42, 42, 5]:
    # facts eq={42} (gives 42 from Stage 1)
    # pattern eq 42 (gives 42 from Stage 2)
    # plus some fallback won't trigger since we have pattern matches
    # Let's just use facts with eq={42, 5}
    idx = LeafFactsIndex(facts={("x",): LeafFacts(eq=frozenset({42, 5}))})
    result = propose_leaf_candidates(
        atom_expr=_parse_expr("x == CONST"),
        atom_value=False,
        dependency_paths=frozenset({("x",)}),
        state_index=si,
        env={"x": 5},
        leaf_facts_index=idx,
        max_per_leaf=8,
    )
    cands = result[("x",)].candidates
    assert 42 in cands
    assert 5 not in cands  # current value excluded
    # Check that 42 appears only once
    assert cands.count(42) == 1


# ---------------------------------------------------------------------------
# len() / structural candidates
# ---------------------------------------------------------------------------


def _solve(src, roots, desired="then"):
    import os
    import tempfile
    from tharkun.minleaf.api import solve_minleaf_patch_for_guard
    p = os.path.join(tempfile.mkdtemp(), "g.py")
    with open(p, "w") as f:
        f.write(src)
    return solve_minleaf_patch_for_guard(
        source_path=p, target_line=1, desired_branch=desired,
        env=dict(roots), roots=dict(roots),
    )


def test_len_list_grows_to_satisfy_gt():
    r = _solve("if len(tags) > 2:\n    pass\n", {"tags": ["a"]})
    assert r.success is True
    assert len(r.patch.ops[0].new) == 3


def test_len_list_exact_eq():
    r = _solve("if len(xs) == 3:\n    pass\n", {"xs": []})
    assert r.success is True
    assert len(r.patch.ops[0].new) == 3


def test_len_reversed_operand_order():
    # 5 < len(xs) — the length is on the right; the comparison must be flipped.
    r = _solve("if 5 < len(xs):\n    pass\n", {"xs": [1]})
    assert r.success is True
    assert len(r.patch.ops[0].new) > 5


def test_len_str_pads_preserving_type():
    r = _solve("if len(pw) >= 8:\n    pass\n", {"pw": "ab"})
    assert r.success is True
    assert isinstance(r.patch.ops[0].new, str)
    assert len(r.patch.ops[0].new) >= 8


def test_len_grow_reuses_existing_elements():
    # Padding cycles the existing elements, so the element type stays what the
    # program expects rather than becoming a foreign filler.
    r = _solve("if len(xs) > 2:\n    pass\n", {"xs": ["tag"]})
    assert r.success is True
    assert all(isinstance(v, str) for v in r.patch.ops[0].new)


def test_membership_and_len_solved_together():
    r = _solve(
        'if cfg["mode"] in ("fast","turbo") and len(cfg["tags"]) > 2:\n    pass\n',
        {"cfg": {"mode": "slow", "tags": ["a"]}},
    )
    assert r.success is True
    paths = {op.path for op in r.patch.ops}
    assert ("cfg", "mode") in paths
    assert ("cfg", "tags") in paths


def test_container_is_patchable_for_truthiness():
    # Making a non-empty list falsy is a structural change; the container leaf
    # makes it expressible.
    r = _solve("if flags:\n    pass\n", {"flags": [0, 0, 0]}, desired="else")
    assert r.success is True
    assert not r.patch.ops[0].new


def test_bytes_whole_value_solvable_under_per_byte():
    # b"AB" projects to two int leaves under the per_byte policy; no per-byte edit makes
    # a 2-byte value equal a 5-byte one. The bytes object is now a leaf in its own right.
    r = _solve('if token == b"MAGIC":\n    pass\n', {"token": b"AB"})
    assert r.success is True
    assert r.patch.ops[0].new == b"MAGIC"


def test_per_byte_edit_still_preferred_when_it_suffices():
    # Registering the container must not make minleaf replace wholesale when a single
    # byte would do — the cost model (distinct paths touched) keeps both at 1, and the
    # search still finds the element edit.
    r = _solve("if token[0] == 88:\n    pass\n", {"token": b"AB"})
    assert r.success is True
    assert r.patch.ops[0].path == ("token", 0)
