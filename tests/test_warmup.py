"""The BEFORE warmup: static type hints, per point."""

from __future__ import annotations

import os

import pytest

from tharkun.knowledge import KnowledgeBase, program_point
from tharkun.lattice import BOTTOM
from tharkun.warmup import build_prior, warm_targets

TOYS = os.path.join(os.path.dirname(__file__), "fixtures", "technique_toys.py")
DEEP = os.path.join(os.path.dirname(__file__), "hybrid", "fixtures", "deep_target.py")




def test_prior_records_static_types_at_guard():
    """Static infers `n: int` at `by_dead_guard`'s guard; the prior carries that typing."""
    kb = KnowledgeBase()
    priced = build_prior(kb, TOYS, "by_dead_guard")
    assert priced >= 1
    # The guard is on line 32 of the fixture (`if n > 10 and n < 5:`).
    state = kb.at(program_point(TOYS, 32))
    assert "int" in state.types_for("n")




def test_live_edge_is_not_proved_dead():
    """A satisfiable guard's bodies are never a dead certificate.

    ``at`` returns BOTTOM for an unrecorded point too, so the meaningful check is
    ``proved_dead`` — recorded unreachable — which must be False for every live edge.
    """
    kb = KnowledgeBase()
    build_prior(kb, TOYS, "by_static")
    for node_line in range(6, 13):
        assert not kb.proved_dead(program_point(TOYS, node_line))


def test_warm_is_idempotent_on_the_index():
    """Re-warming the same target does not invent new dead points."""
    kb = KnowledgeBase()
    build_prior(kb, TOYS, "by_dead_guard")
    before = len(kb.index.prior)
    build_prior(kb, TOYS, "by_dead_guard")
    assert len(kb.index.prior) == before


def test_warm_targets_prices_multiple():
    """`warm_targets` accepts the `(module, func)` list `harness_map` hands back."""
    kb = KnowledgeBase()
    total = warm_targets(kb, [(TOYS, "by_dead_guard"), (TOYS, "by_static")])
    assert total >= 2


def test_missing_function_prices_nothing():
    kb = KnowledgeBase()
    assert build_prior(kb, TOYS, "does_not_exist") == 0


def test_compound_guard_on_one_var_is_not_false_dead():
    """A variable named in two facets of a live guard must not forge a dead certificate.

    ``kind * kind == 49 and kind > 0`` yields two VarConstraints on ``kind`` whose candidate
    value-sets are disjoint; recording them by MEET would empty the set and mark the guard's
    body BOTTOM. The prior records types only, so every live edge here stays reachable.
    """
    kb = KnowledgeBase()
    build_prior(kb, DEEP, "parse")
    for st in kb.index.prior.values():
        assert not st.is_bottom()




def test_unwarmed_frontier_is_not_pruned():
    """The trap that shipping ``at().is_bottom()`` would have caused: a body line the
    warmup never recorded must NOT read as dead, or the assist prunes everything.
    """
    kb = KnowledgeBase()
    build_prior(kb, TOYS, "by_dead_guard")
    # A deep callee line no warmup touched — unrecorded, must not be a dead certificate.
    assert not kb.proved_dead(program_point(TOYS, 9999))
    # ...even though at() returns BOTTOM for it by the forward-analysis convention.
    assert kb.at(program_point(TOYS, 9999)).is_bottom()
