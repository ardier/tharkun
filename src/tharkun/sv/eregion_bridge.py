"""Eregion -> Tharkun candidate-fact bridge ([internal ref]/JOB 2, [internal ref]).

Owner ruling (2026-09-29): this bridge is an EXTENSION OF THE FACT STORE, not
a separate translator -- Eregion is one more fact producer into
``KnowledgeBase``'s ``Evidence``/``Derivation`` mechanism (the same structure
Avengers already uses for its own fixpoint-pipeline artifacts).  This module
only PACKAGES what a producer already emits (plain, JSON-safe dicts) into
canonical ``Evidence``; the store's own machinery and
``sv.validate.check_candidates_individually`` decide what is true.

A producer's live objects (``eregion_codegen.objects.wrapper.AnswerSet`` and the
like) are NEVER imported here -- Tharkun does not depend on Eregion.  The
producer side emits plain dicts (``eregion_codegen.objects.tharkun_facts`` in the
eregion-codegen repo) and :func:`publish_fact` is the one place that turns them
into ``solve.obligations.Invariant`` objects and publishes them as ``Evidence``.
``sv/fact_producers.py`` is how the dicts get here (the generic plugin seam);
``sv/fact_proof.py`` is the consumer that reads these kinds back out, certifies
them and may discharge obligations with the certified ones.

Evidence kinds, and who may write each:

* :data:`INVARIANT_CANDIDATE` -- a producer's claim that one ``Invariant`` holds
  at one source location.  Never trusted: it is a proposal.
* :data:`DRAW_VECTOR` -- a producer's proposed draw vector: values for the task's own ``_sv_verifier`` draws, in
  draw order, that the producer thinks reach a violation.  Never trusted and never certified: it is replayed
  natively on the unmodified program by the ``producer`` stage (``sv/stages.py``), and only that replay can
  decide anything (docs/SV.md "Draw vectors from a producer").
* Periodic-family intersection (``sv.validate.FamilyIntersectionCandidate``) is
  RULED, NOT BUILT here: no producer can emit it honestly today (the engine's
  ``AnswerSet`` exposes flags, not the raw bases and period; the failing pre-state
  of a method is not derivable by the engine; and the verifier body would need the
  float-soundness lemma only absint carries, ``solve/absint/lemmas.py::TRIG_PAIR``),
  so a bridge shape for it would be dead code.  docs/SV.md "3a" records the blockers.
* :data:`CERTIFIED_INVARIANT` -- written ONLY by :func:`publish_certified`, which
  ``sv/fact_proof.py`` calls for an ``Invariant`` the independent checker proved
  on its own.  A reader that needs a fact it can rely on reads this kind and no
  other (the owner's "certified-only" read for proof components).
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from tharkun.knowledge import KnowledgeBase
from tharkun.solve.obligations import Invariant

INVARIANT_CANDIDATE = "invariant_candidate"
CERTIFIED_INVARIANT = "certified_invariant"
DRAW_VECTOR = "draw_vector"
#: a proposed vector longer than this, or one entry bigger than these, is dropped (a runaway producer)
MAX_VECTOR_ENTRIES = 256
MAX_INT_BITS = 4096
MAX_STR_CHARS = 4096


def invariant_key(inv: Invariant) -> str:
    """The canonical key of an ``Invariant`` in the store: its location plus a
    digest of its predicate (several different claims may sit at one statement,
    and the same claim published twice is one item with two derivations)."""
    digest = hashlib.sha256(f"{inv.type}|{inv.value}".encode("utf-8")).hexdigest()[:12]
    return f"{inv.file_name}:{inv.line}:{inv.column}:{inv.function}::{digest}"


def publish_invariant_candidate(kb: KnowledgeBase, fact: Mapping[str, Any], *, producer: str) -> None:
    """Publish one ``invariant_candidate`` fact (``fact["site"]`` is
    ``{file, line, column, function}``, ``fact["value"]`` a ``python_expression``
    predicate, ``fact["evidence"]`` optional JSON-safe provenance such as the
    producer's own engine flags -- raw data, never a verdict).

    ``Invariant``'s own constructor validates the location and the predicate's
    syntax: a malformed fact raises ``ValueError``/``KeyError`` here rather
    than publishing half a candidate."""
    site = fact["site"]
    inv = Invariant(str(site["file"]), site["line"], site["column"], str(site["function"]),
                    str(fact["value"]), str(fact.get("type", "location_invariant")))
    kb.publish(INVARIANT_CANDIDATE, invariant_key(inv),
               {"invariant": inv, "evidence": dict(fact.get("evidence") or {})},
               producer=producer)


def publish_certified(kb: KnowledgeBase, inv: Invariant, *, checker: str, producer: str) -> None:
    """Record that *inv* was proved by an independent checker (``checker`` names
    it, e.g. ``"sv.validate.check_candidates_individually"``).

    Not callable from a producer's data: only the certifier in ``sv/fact_proof.py``
    reaches it, and only for a candidate whose per-candidate check returned
    ``True``.  Derived from the candidate it certifies."""
    key = invariant_key(inv)
    kb.publish(CERTIFIED_INVARIANT, key, {"invariant": inv, "checker": checker}, producer=producer,
               parents=((INVARIANT_CANDIDATE, key),))


def _draw_vector_entries(fact: Mapping[str, Any]) -> list:
    """The validated entries of ``fact["vector"]``, each ``{"kind", "value"}`` and nothing else: a ``bool``, an
    ``int`` of at most :data:`MAX_INT_BITS` bits, a ``float`` (a ``float.hex`` string), or a ``str`` inside
    ``string.printable`` of at most :data:`MAX_STR_CHARS` characters.  ``ValueError`` for anything else (the
    caller drops the one fact); a vector that holds a draw the child would refuse is not worth publishing."""
    from tharkun.sv.engine import VECTOR_KINDS, admissible

    vector = fact.get("vector")
    if not isinstance(vector, list) or not 1 <= len(vector) <= MAX_VECTOR_ENTRIES:
        raise ValueError("a draw vector is a list of 1 to %d entries" % MAX_VECTOR_ENTRIES)
    entries = []
    for entry in vector:
        if not isinstance(entry, dict) or set(entry) != {"kind", "value"} or entry["kind"] not in VECTOR_KINDS:
            raise ValueError("a draw vector entry is {kind, value} of kind %s" % "/".join(VECTOR_KINDS))
        kind, value = entry["kind"], entry["value"]
        if kind == "bool":
            ok = type(value) is bool
        elif kind == "int":
            ok = type(value) is int and value.bit_length() <= MAX_INT_BITS
        elif kind == "float":
            ok = isinstance(value, str)
            if ok:
                try:
                    float.fromhex(value)
                except (ValueError, OverflowError):
                    ok = False
        else:
            ok = isinstance(value, str) and len(value) <= MAX_STR_CHARS and admissible("str", value)
        if not ok:
            raise ValueError("a draw vector entry holds a value its kind does not admit")
        entries.append({"kind": kind, "value": value})
    return entries


def publish_draw_vector(kb: KnowledgeBase, fact: Mapping[str, Any], *, producer: str) -> None:
    """Publish one ``draw_vector`` fact (``fact["vector"]``: the entries, see :func:`_draw_vector_entries`;
    ``fact["site"]`` optional ``{file, line, column, function}`` of the obligation it was proposed for;
    ``fact["evidence"]`` optional provenance, raw data).  Keyed by a digest of the entries: the same vector
    proposed twice is one item with two derivations."""
    entries = _draw_vector_entries(fact)
    site = fact.get("site")
    if site is not None:
        site = {"file": str(site["file"]), "line": int(site["line"]), "column": int(site["column"]),
                "function": str(site["function"])}
    key = hashlib.sha256(json.dumps(entries, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    kb.publish(DRAW_VECTOR, key, {"vector": entries, "site": site, "evidence": dict(fact.get("evidence") or {})},
               producer=producer)


#: a producer fact's ``kind`` -> the publisher that turns it into Evidence; an unknown
#: kind is rejected, never guessed at
_PUBLISHERS = {"invariant_candidate": publish_invariant_candidate, "draw_vector": publish_draw_vector}


def publish_fact(kb: KnowledgeBase, fact: Mapping[str, Any], *, producer: str) -> None:
    """Publish one producer fact, dispatching on ``fact["kind"]``.  Raises
    ``ValueError`` for an unknown kind (the caller drops that one fact and
    carries on: an unrecognised claim proves nothing)."""
    publisher = _PUBLISHERS.get(fact.get("kind"))
    if publisher is None:
        raise ValueError(f"unknown producer fact kind {fact.get('kind')!r}")
    publisher(kb, fact, producer=producer)
