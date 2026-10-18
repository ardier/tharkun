"""Phase 6: Leaf facts conversion.

Converts upstream FactEnv/VarFacts (keyed by variable name) to
LeafFactsIndex (keyed by leaf Path).
"""

from __future__ import annotations

from typing import Any

from tharkun.minleaf.types import FactEnv, VarFacts

from tharkun.minleaf.types import (
    LeafFacts,
    LeafFactsIndex,
    Path,
    StateIndex,
)


def build_leaf_facts_index(
    fact_env: FactEnv | None,
    state_index: StateIndex,
    *,
    source_tag: str = "static",
) -> LeafFactsIndex:
    """Convert upstream FactEnv to a LeafFactsIndex keyed by leaf Path."""
    # F5: No facts
    if fact_env is None:
        return LeafFactsIndex(facts={}, global_possible=True)

    # F4: Global infeasibility
    global_possible = fact_env.possible

    facts: dict[Path, LeafFacts] = {}

    for var_name, vf in fact_env.facts.items():
        # F2: Dotted name expansion
        if "." in var_name:
            segments = var_name.split(".")
            path_parts: list[str | int] = []
            for seg in segments:
                try:
                    path_parts.append(int(seg))
                except ValueError:
                    path_parts.append(seg)
            path = tuple(path_parts)
        else:
            path = (var_name,)

        # Check if path is a leaf
        if path in state_index.leaves:
            # F1 / F2: Direct primitive match
            lf = _vf_to_leaf_facts(vf, source_tag)
            if path in facts:
                facts[path] = merge_leaf_facts(facts[path], lf)
            else:
                facts[path] = lf
        else:
            # F3: Container root, no leaf match
            # Only record if VarFacts.eq is non-None
            if vf.eq is not None:
                lf = _vf_to_leaf_facts(vf, source_tag)
                # But only if some path derived from this var is a leaf
                # Actually per spec: just skip unless eq is non-None
                # and the path itself happens to be a leaf (already handled above)
                pass
            # Do NOT propagate facts to child leaves

    return LeafFactsIndex(facts=facts, global_possible=global_possible)


def _vf_to_leaf_facts(vf: VarFacts, source_tag: str) -> LeafFacts:
    """Convert a VarFacts to a LeafFacts."""
    eq = frozenset(vf.eq) if vf.eq is not None else None
    neq = frozenset(vf.neq)

    # Infer type from eq set or bounds presence
    typ: type | None = None
    if eq is not None and len(eq) > 0:
        first = next(iter(eq))
        typ = type(first)
    elif vf.lo is not None or vf.hi is not None:
        typ = int

    return LeafFacts(
        eq=eq,
        neq=neq,
        lo=vf.lo,
        hi=vf.hi,
        typ=typ,
        source=source_tag,
    )


def merge_leaf_facts(a: LeafFacts, b: LeafFacts) -> LeafFacts:
    """Merge two LeafFacts for the same path.

    - eq: intersection if both non-None; else whichever is non-None.
    - neq: union.
    - lo: max of both.
    - hi: min of both.
    - source: concatenate with "+".
    """
    # eq
    if a.eq is not None and b.eq is not None:
        eq = a.eq & b.eq
    elif a.eq is not None:
        eq = a.eq
    elif b.eq is not None:
        eq = b.eq
    else:
        eq = None

    # neq
    neq = a.neq | b.neq

    # lo
    if a.lo is not None and b.lo is not None:
        lo = max(a.lo, b.lo)
    elif a.lo is not None:
        lo = a.lo
    else:
        lo = b.lo

    # hi
    if a.hi is not None and b.hi is not None:
        hi = min(a.hi, b.hi)
    elif a.hi is not None:
        hi = a.hi
    else:
        hi = b.hi

    # typ: pick whichever is non-None (prefer a)
    typ = a.typ if a.typ is not None else b.typ

    # source
    source = a.source + "+" + b.source

    return LeafFacts(eq=eq, neq=neq, lo=lo, hi=hi, typ=typ, source=source)
