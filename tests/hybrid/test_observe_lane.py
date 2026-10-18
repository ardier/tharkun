"""The Atheris lane feeds the KnowledgeBase: corpus seeds -> observed types at guards.

The bidirectional half of the shared pool. Atheris (or the assist) writes seeds; tracing
each one records the RUNTIME types at every guard it reaches into the lattice, so the
assist's own search is typed by what actually flowed through the program — not by a static
guess or the shape of whatever value a seed happened to carry.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from tharkun.hybrid.assist import AssistEngine, HarnessSpec
from tharkun.knowledge import program_point

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _atheris() -> bool:
    from tharkun.hybrid.fdp_codec import atheris_available

    return atheris_available()


def _engine(corpus_dir: str) -> AssistEngine:
    return AssistEngine(HarnessSpec(
        harness_path=os.path.join(FIXDIR, "deep_harness.py"),
        target_path=os.path.join(FIXDIR, "deep_target.py"),
        target_func="parse", corpus_dir=corpus_dir,
    ))


def test_observe_records_runtime_types_at_a_guard():
    """A seed run through parse(magic, kind, size) observes int types at the guard lines."""
    eng = _engine(tempfile.mkdtemp())
    target = os.path.join(FIXDIR, "deep_target.py")
    n = eng._observe_seed_into_kb(
        b"", target, "parse",
        {"magic": 1094795585, "kind": 7, "size": 1},
        [target],
    )
    assert n >= 1
    # deep_target guard on line 2 uses `magic`; the observed type is int.
    state = eng.facts.at(program_point(target, 2))
    assert "int" in (state.types_for("magic") or set())


def test_observe_is_types_only_never_failing_values():
    """The observe records types, not the concrete values a seed carried.

    A membership fact built from a seed's values at an uncovered guard would be exactly the
    values that FAILED it — trapping the search. So no value set is recorded, only the type.
    """
    from tharkun.strategies import constraints_from_state

    eng = _engine(tempfile.mkdtemp())
    target = os.path.join(FIXDIR, "deep_target.py")
    eng._observe_seed_into_kb(b"", target, "parse",
                              {"magic": 5, "kind": 0, "size": 0}, [target])
    state = eng.facts.at(program_point(target, 2))
    got = constraints_from_state(state, "magic")
    assert got == {"type": "int"}          # a type, and no "values" membership


def test_observe_dedups_by_seed_content():
    eng = _engine(tempfile.mkdtemp())
    target = os.path.join(FIXDIR, "deep_target.py")
    seeds = [{"data": b"abc", "params": {"magic": 1, "kind": 2, "size": 3}}]
    eng._observe_new_seeds(seeds, target, "parse", lambda s: s["params"], [target])
    first = dict(eng.facts.index.observed)
    # Re-running with the same seed content changes nothing (idempotent-guarded).
    eng._observe_new_seeds(seeds, target, "parse", lambda s: s["params"], [target])
    assert dict(eng.facts.index.observed).keys() == first.keys()


def test_observe_survives_a_bad_seed():
    """A seed whose params make the target raise must not break the observe (returns 0)."""
    eng = _engine(tempfile.mkdtemp())
    target = os.path.join(FIXDIR, "deep_target.py")
    # Wrong arity / types: parse takes (magic, kind, size); give it something that raises.
    n = eng._observe_seed_into_kb(b"", target, "parse", {"magic": object()}, [target])
    assert n == 0


@pytest.mark.skipif(not _atheris(), reason="atheris not installed")
def test_observed_type_aims_a_later_search():
    """End to end: the type the lane observed types the reach-search that follows.

    parse's `magic` is an int. After the lane observes that, a reach-search over `magic`
    (seeded, misleadingly, with a str) builds an int strategy from the observed type and
    lands the magic value — where the str value-shape strategy never could.
    """
    from tharkun.driver import _reach_by_search

    eng = _engine(tempfile.mkdtemp())
    target = os.path.realpath(os.path.join(FIXDIR, "deep_target.py"))
    # Observe a run so the KB learns magic:int (and the value 1094795585 is NOT recorded).
    eng._observe_seed_into_kb(b"", target, "parse",
                              {"magic": 1094795585, "kind": 7, "size": 1}, [target])
    # Seed the search with a str for magic; only the observed int type makes it searchable
    # in the right domain. Aim with the magic constant as a hint (RedQueen would supply it).
    got = _reach_by_search(target, "parse", 3, target,
                           {"magic": "xxx", "kind": 7, "size": 1}, {"magic"},
                           hints=[1094795585], store=eng.facts,
                           guard_path=target, guard_line=2, max_examples=200)
    assert got is not None and got.get("magic") == 1094795585
