"""egglog canonicalizer — the SHADOW engine, never the authority.

:mod:`tharkun.canonicalize` is the authority: a fast, dependency-free orientation
normalizer that runs under the core interpreter. This module reaches further — arithmetic
rearrangement, constant folding, conditional subsumption — by equality saturation over an
e-graph, but it depends on the Rust-backed ``egglog`` (Python >=3.11), so it can never be
load-bearing. It runs *alongside* orientation and its only job is to show, on real clauses,
where it merges something orientation kept apart. That disagreement log is the data that
would justify promoting it; nothing consumes its output otherwise.

The egglog term language and rules live in :mod:`tharkun._egraph_rules`, which egglog
requires at module scope. This module imports it lazily behind :func:`egraph_available`, so
``import tharkun.egraph`` succeeds when egglog is absent.

Lifted from a standalone study that answered the three questions that decide usability:

* **Extraction is reproducible but not canonical** under egglog's default integer cost —
  fixed by a total-order ``(size, canonical_string)`` cost whose per-e-class minimum is
  unique. Verified stable across processes.
* **AC rewrites explode** in egglog (8 conjuncts -> 650 ms), so associativity and
  commutativity are done in the Python frontend in linear time, never as egglog rules.
* **Four negation rules are unsound for NaN** — dropped in ``_egraph_rules``; the NaN-safe
  forms live in the orientation engine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def egraph_available() -> bool:
    """True when the optional ``egglog`` dependency can be imported and the rules load."""
    try:
        from tharkun import _egraph_rules  # noqa: F401

        return True
    except Exception:
        return False


def shadow_enabled() -> bool:
    """True when the shadow should run — gated by env AND availability.

    Off by default: the whole point of a shadow is that it costs nothing and depends on
    nothing until someone deliberately turns it on to gather the disagreement data.
    """
    return bool(os.environ.get("THARKUN_EGRAPH_SHADOW")) and egraph_available()


def canonicalize(text: str, *, deep: bool = True) -> str | None:
    """A canonical string for *text* via equality saturation, or None on any failure.

    Never raises — the shadow must not be able to break a caller, so a missing dependency,
    an unrepresentable guard, or an egglog error all return None. ``deep`` runs the
    subsumption ruleset as well, which is where the e-graph reaches past orientation.
    """
    try:
        from tharkun._egraph_rules import canonicalize as _canon

        return _canon(text, deep=deep)
    except Exception:
        return None


@dataclass
class ShadowLog:
    """Where egglog and orientation disagree, so promotion is a measured decision.

    ``egglog_further`` — two clause texts orientation kept distinct that egglog merged.
    Each is a case the authority misses and the e-graph reaches: the payoff column.
    ``divergent`` — orientation merged something egglog did not, to investigate, since the
    authority should be the more conservative of the two.
    """

    comparisons: int = 0
    egglog_further: list[tuple[str, str]] = field(default_factory=list)
    divergent: list[tuple[str, str, str]] = field(default_factory=list)

    def observe(self, a_text: str, b_text: str, a_orient: str, b_orient: str) -> None:
        """Record whether egglog merges (a, b) that orientation left apart."""
        self.comparisons += 1
        ea, eb = canonicalize(a_text), canonicalize(b_text)
        if ea is None or eb is None:
            return
        orient_equal = a_orient == b_orient
        egglog_equal = ea == eb
        if egglog_equal and not orient_equal:
            self.egglog_further.append((a_text, b_text))
        elif orient_equal and not egglog_equal:
            self.divergent.append((a_text, b_text, "orientation-merged-egglog-did-not"))

    def report(self) -> dict:
        return {"comparisons": self.comparisons,
                "egglog_reached_further": len(self.egglog_further),
                "divergent": len(self.divergent)}


#: The process-wide shadow log, populated when the shadow runs.
LOG = ShadowLog()
