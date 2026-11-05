"""Does the egglog shadow earn its dependency? Measure it against the authority.

:mod:`tharkun.egraph` is a shadow: it runs alongside the orientation normalizer
(:mod:`tharkun.canonicalize`) and its only justification for ever being promoted is
evidence that, on real clauses, it merges constraints orientation keeps apart — without
merging constraints that are genuinely different. This module produces that evidence as
two numbers per corpus:

* **reaches_further** — pairs egglog canonicalizes together that orientation does not. The
  payoff column: equivalences the cheap engine misses.
* **over_merges** — pairs egglog merges that are KNOWN distinct (drawn from the catalog's
  ``distinct`` pairs). The danger column: a canonicalizer that invents an equivalence is
  worse than one that misses it, because a merged-but-unequal clause asserts a fact no path
  established. This MUST stay zero.

The measured result, recorded so the promotion stays a decision and not a guess:

* On tharkun's own ~1.6k guard texts, egglog reaches **0** pairs further than orientation —
  the two engines partition real single-source guards identically.
* On a synthetic probe of egglog's home turf, its only wins over orientation WERE
  **commutative ``BinOp`` ordering** (``a + b`` ≡ ``b + a``, ``x * 2`` ≡ ``2 * x``) and
  **boolean idempotence** (``a and a`` ≡ ``a``). Both were cheap, dependency-free rules, so
  the orientation engine ABSORBED them — ``probe_gap()["egglog_only"]`` is now empty. This
  is the intended arc: measure where the shadow reaches further, then close the gap in the
  authority, keeping egglog a shadow rather than a load-bearing dependency. The probe pairs
  stay in :data:`DEEP_PROBE` as a regression witness — if orientation ever loses one, it
  reappears as an egglog-only win.
* Neither engine collapses associativity-across-parens or algebraic cancellation — egglog's
  AC ruleset is deliberately off (it explodes). That is the standing residue an e-graph
  would still reach; nothing in the current corpus needs it.

Availability-gated: every entry point returns an empty/gated report when ``egglog`` is not
installed, so importing this module never fails.
"""

from __future__ import annotations

import ast
import collections
import glob
import itertools
import os
from dataclasses import dataclass, field


# The synthetic probe of egglog's home turf: truly-equivalent pairs that need reasoning the
# orientation engine deliberately does not do. Each is (a, b, why). Kept here, not in the
# catalog, because the catalog is the CONTRACT both engines must satisfy — these are cases
# only egglog satisfies today, so asserting them as contract would (correctly) fail
# orientation. This is the measurement of the gap, not a requirement on it.
DEEP_PROBE: list[tuple[str, str, str]] = [
    ("a + b == 10", "b + a == 10", "additive commutativity in a term"),
    ("x * 2 == y", "2 * x == y", "multiplicative commutativity"),
    ("2 + a == n", "a + 2 == n", "const-left additive commutativity"),
    ("a and a", "a", "boolean idempotence (and)"),
    ("a or a", "a", "boolean idempotence (or)"),
]

# Pairs NEITHER engine reaches today — the honest ceiling, so nobody claims egglog gets them.
UNREACHED_BY_BOTH: list[tuple[str, str, str]] = [
    ("(a + b) + c == 0", "a + (b + c) == 0", "associativity across parens (egglog AC is off)"),
    ("a + 2 + 3 == n", "a + 5 == n", "fold chain with a symbolic term"),
    ("x - x == 0", "True", "algebraic cancellation"),
]


@dataclass
class ShadowBehavior:
    """The promotion evidence for one corpus."""

    available: bool = True
    guards: int = 0
    egglog_canonicalized: int = 0
    orient_classes: int = 0
    egglog_classes: int = 0
    reaches_further: list[tuple[str, str, str]] = field(default_factory=list)
    over_merges: list[tuple[str, str, str]] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "available": self.available,
            "guards": self.guards,
            "egglog_canonicalized": self.egglog_canonicalized,
            "orient_classes": self.orient_classes,
            "egglog_classes": self.egglog_classes,
            "reaches_further": len(self.reaches_further),
            "over_merges": len(self.over_merges),
        }


def _harvest_guards(source_globs, max_len: int = 80) -> list[str]:
    """Every distinct ``if``/``while`` test text under *source_globs*, short ones only.

    Short-guard filter keeps the pairwise comparison tractable and focuses on the atoms a
    canonicalizer actually has to unify; a 200-char compound guard rarely recurs verbatim.
    """
    guards: set[str] = set()
    for pattern in source_globs:
        for f in glob.glob(pattern, recursive=True):
            try:
                tree = ast.parse(open(f, encoding="utf-8", errors="ignore").read())
            except (OSError, SyntaxError):
                continue
            for n in ast.walk(tree):
                if isinstance(n, (ast.If, ast.While)):
                    try:
                        text = ast.unparse(n.test)
                    except Exception:
                        continue
                    if len(text) <= max_len:
                        guards.add(text)
    return sorted(guards)


def _classes(canon_by_src: dict[str, str]) -> dict[str, list[str]]:
    m: dict[str, list[str]] = collections.defaultdict(list)
    for src, can in canon_by_src.items():
        m[can].append(src)
    return {k: v for k, v in m.items() if len(v) > 1}


def measure_corpus(source_globs) -> ShadowBehavior:
    """Run both engines over the guards under *source_globs* and count further/over-merge."""
    from tharkun import egraph
    from tharkun.solve.canonicalize import canonicalize as orient

    if not egraph.egraph_available():
        return ShadowBehavior(available=False)

    guards = _harvest_guards(source_globs)
    o = {g: orient(g) for g in guards}
    e = {}
    for g in guards:
        ce = egraph.canonicalize(g)
        if ce is not None:
            e[g] = ce

    og, eg = _classes(o), _classes(e)
    further: list[tuple[str, str, str]] = []
    for can, members in eg.items():
        for a, b in itertools.combinations(sorted(members), 2):
            if a in o and b in o and o[a] != o[b]:
                further.append((a, b, can))

    # Over-merge is measured against KNOWN-distinct pairs, not this corpus (we cannot know
    # which corpus pairs are truly distinct). Sourced from the catalog in check_soundness().
    return ShadowBehavior(
        available=True,
        guards=len(guards),
        egglog_canonicalized=len(e),
        orient_classes=len(og),
        egglog_classes=len(eg),
        reaches_further=further,
    )


def check_soundness() -> list[tuple[str, str, str]]:
    """Every catalog ``distinct`` pair egglog wrongly MERGES — the danger column. Want: [].

    A shadow that over-merges is not a slower authority, it is an unsound one: it would
    assert an equivalence no path established. This is the one egglog invariant that gates
    promotion regardless of how far it reaches.
    """
    from tharkun import egraph
    from tharkun.solve.rewrite_catalog import CATALOG

    if not egraph.egraph_available():
        return []
    bad: list[tuple[str, str, str]] = []
    for group in CATALOG:
        for a, b in group.distinct:
            ca, cb = egraph.canonicalize(a), egraph.canonicalize(b)
            if ca is not None and cb is not None and ca == cb:
                bad.append((a, b, group.name))
    return bad


def probe_gap() -> dict[str, list[tuple[str, str, str]]]:
    """Where egglog beats orientation on the DEEP_PROBE, and where both still fall short.

    Returns ``{"egglog_only": [...], "neither": [...], "orientation_regressed": [...]}``.
    ``orientation_regressed`` should stay empty — a probe pair the authority merges but the
    shadow does not would mean the shadow lost ground, worth knowing.
    """
    from tharkun import egraph
    from tharkun.solve.canonicalize import canonicalize as orient

    out = {"egglog_only": [], "neither": [], "orientation_regressed": []}
    if not egraph.egraph_available():
        return out
    for a, b, why in DEEP_PROBE:
        oa, ob = orient(a), orient(b)
        ea, eb = egraph.canonicalize(a), egraph.canonicalize(b)
        o_merge = oa == ob
        e_merge = ea is not None and eb is not None and ea == eb
        if e_merge and not o_merge:
            out["egglog_only"].append((a, b, why))
        elif o_merge and not e_merge:
            out["orientation_regressed"].append((a, b, why))
    for a, b, why in UNREACHED_BY_BOTH:
        oa, ob = orient(a), orient(b)
        ea, eb = egraph.canonicalize(a), egraph.canonicalize(b)
        o_merge = oa == ob
        e_merge = ea is not None and eb is not None and ea == eb
        if not o_merge and not e_merge:
            out["neither"].append((a, b, why))
    return out


def format_report(source_globs=None) -> str:
    """A human-readable promotion report for the given corpus (defaults to tharkun's src)."""
    from tharkun import egraph

    if not egraph.egraph_available():
        return "egglog not installed — shadow behavior unmeasurable."
    if source_globs is None:
        here = os.path.dirname(__file__)
        source_globs = [os.path.join(here, "**", "*.py")]

    beh = measure_corpus(source_globs)
    bad = check_soundness()
    gap = probe_gap()
    lines = [
        "egglog shadow behavior vs the orientation authority",
        f"  corpus guards               : {beh.guards}",
        f"  egglog canonicalized        : {beh.egglog_canonicalized}",
        f"  orientation merge-classes   : {beh.orient_classes}",
        f"  egglog merge-classes        : {beh.egglog_classes}",
        f"  egglog reaches further      : {len(beh.reaches_further)} pairs",
        f"  egglog over-merges (UNSOUND): {len(bad)}  (must be 0)",
        "  deep-probe wins egglog-only :",
    ]
    for a, b, why in gap["egglog_only"]:
        lines.append(f"      {a!r} == {b!r}  ({why})")
    lines.append("  reached by neither engine   :")
    for a, b, why in gap["neither"]:
        lines.append(f"      {a!r} == {b!r}  ({why})")
    return "\n".join(lines)
