"""The FactState -> make_strategy bridge: a prior's type aims the search strategy."""

from __future__ import annotations

import pytest

from tharkun.lattice import TOP, BOTTOM
from tharkun.strategies import constraints_from_state, strategy_from_state


def _hypothesis() -> bool:
    try:
        import hypothesis  # noqa: F401

        return True
    except ImportError:
        return False


def test_type_only_selects_that_strategy_type():
    assert constraints_from_state(TOP.with_types("x", ["str"]), "x") == {"type": "str"}
    assert constraints_from_state(TOP.with_types("x", ["bytes"]), "x") == {"type": "bytes"}


def test_int_interval_becomes_value_range():
    s = TOP.with_types("n", ["int"]).with_interval("n", lo=1001, hi=1009)
    assert constraints_from_state(s, "n") == {"type": "int", "value_range": [1001, 1009]}


def test_int_values_membership_dominates_interval():
    s = TOP.with_types("c", ["int"]).with_values("c", [4321]).with_interval("c", lo=0, hi=9999)
    got = constraints_from_state(s, "c")
    assert got == {"type": "int", "values": [4321]}


def test_type_aliases_fold_to_vocabulary():
    assert constraints_from_state(TOP.with_types("x", ["List"]), "x")["type"] == "list"
    assert constraints_from_state(TOP.with_types("x", ["bytearray"]), "x")["type"] == "bytes"
    assert constraints_from_state(TOP.with_types("x", ["bool"]), "x")["type"] == "int"


def test_ambiguous_type_falls_back_to_none():
    assert constraints_from_state(TOP.with_types("u", ["int", "str"]), "u") is None


def test_unknown_subject_is_none():
    assert constraints_from_state(TOP, "z") is None


def test_unmappable_type_is_none():
    assert constraints_from_state(TOP.with_types("o", ["MyClass"]), "o") is None


def test_bottom_state_subject_is_none():
    # BOTTOM has no vars; asking about any subject is a clean None, never an exception.
    assert constraints_from_state(BOTTOM, "anything") is None


@pytest.mark.skipif(not _hypothesis(), reason="hypothesis not installed")
def test_strategy_from_state_builds_a_real_strategy():
    from hypothesis.strategies import SearchStrategy

    s = strategy_from_state(TOP.with_types("x", ["str"]), "x")
    assert isinstance(s, SearchStrategy)
    assert isinstance(s.example(), str)


@pytest.mark.skipif(not _hypothesis(), reason="hypothesis not installed")
def test_strategy_from_state_int_range_draws_in_range():
    s = strategy_from_state(TOP.with_types("n", ["int"]).with_interval("n", lo=5, hi=9), "n")
    assert s is not None
    for _ in range(50):
        v = s.example()
        assert 5 <= v <= 9


def test_strategy_from_state_none_when_nothing_usable():
    assert strategy_from_state(TOP, "z") is None
    assert strategy_from_state(TOP.with_types("u", ["int", "str"]), "u") is None


# --- integration: the prior actually aims the reach-search -----------------------

import os

REACH = os.path.join(os.path.dirname(__file__), "fixtures", "reach_prior.py")




@pytest.mark.skipif(not _hypothesis(), reason="hypothesis not installed")
def test_prior_type_flips_reach_search_from_impossible_to_solved():
    """Same seed, same target: the value-shape strategy cannot reach it; the prior can.

    ``f(x)`` needs ``x == 7`` (an int). Seeded with a str, the value-shape strategy searches
    the text domain and never produces 7, so the target line stays unreached. A prior that
    types ``x: int`` with the value {7} makes the search draw 7 immediately. This is the
    'warmup helps the search' wire, proven end to end through ``_reach_by_search``.
    """
    from tharkun.driver import _reach_by_search
    from tharkun.knowledge import KnowledgeBase, program_point
    from tharkun.lattice import TOP as _TOP

    rp = os.path.realpath(REACH)

    # Without a prior: a str seed gives a text strategy that can never satisfy x == 7.
    empty = KnowledgeBase()
    got_none = _reach_by_search(rp, "f", 3, rp, {"x": "abc"}, {"x"}, hints=[],
                                store=empty, guard_path=rp, guard_line=2, max_examples=200)
    assert got_none is None

    # With a prior typing x:int and naming the value, the search lands it at once.
    kb = KnowledgeBase()
    kb.learn_prior(program_point(rp, 2),
                   _TOP.with_types("x", ["int"]).with_values("x", [7]))
    got = _reach_by_search(rp, "f", 3, rp, {"x": "abc"}, {"x"}, hints=[],
                           store=kb, guard_path=rp, guard_line=2, max_examples=200)
    assert got is not None and got.get("x") == 7
