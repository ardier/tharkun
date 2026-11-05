"""One door onto everything the pipeline knows and everything it has decided.

The solving components were talking to two separate structures — ``FactStore`` (a
memoizing scheduler: decision cache, epoch, exhaustion) and, in principle, ``FactIndex`` (a
lattice: what is known at each program point, joined and canonicalized). ``FactIndex`` had
no production caller at all, so the lattice, ``FactState``, and — transitively —
canonicalization were dead in production. And a component wanting both had to hold two
objects and know which did what.

``KnowledgeBase`` is the single object every component holds. It keeps the two disciplines
that genuinely differ rather than forcing them into one mechanism — a memo is per-run, a
fact is per-program-point, and their merges mean different things — but presents one
surface:

* **scheduling** (inherited from ``FactStore``): ``lookup_or_run`` for a memoized decision,
  ``mark_exhausted`` / ``is_exhausted`` for the epoch-gated skip, ``bump`` when something is
  learned, ``add`` / ``get`` / ``types_for`` for the flat fact path still in use.
* **knowledge** (delegated to ``FactIndex``): ``learn_prior`` for the BEFORE warmup,
  ``observe`` for a traced execution, ``at`` for the combined read at a point.

Subclassing ``FactStore`` rather than wrapping it keeps every existing caller working
unchanged — the driver's ``_STORE.add(...)`` and the assist's ``self.facts.bump(...)`` are
the same calls — so the base can be introduced without touching a single call site beyond
the two constructors.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable

from tharkun.facts import FactStore
from tharkun.lattice import BOTTOM, FactState, FactIndex


EvidenceKey = tuple[str, str]


@dataclass(frozen=True)
class Derivation:
    """One explanation for a canonical evidence item.

    The producer and its parent facts are provenance, not part of the evidence
    identity.  Two tools may therefore discover the same recipe without making
    downstream passes repeat the same work.
    """

    producer: str
    parents: tuple[EvidenceKey, ...] = ()


@dataclass(frozen=True)
class Evidence:
    """A canonical, replayable item learned by a multi-strategy pipeline."""

    kind: str
    key: str
    value: Any
    derivations: tuple[Derivation, ...] = ()

    @property
    def identity(self) -> EvidenceKey:
        return (self.kind, self.key)


class KnowledgeBase(FactStore):
    """The scheduler and the lattice behind one door. A drop-in ``FactStore``."""

    def __init__(self) -> None:
        super().__init__()
        # The knowledge half: a two-tier lattice index (prior by point, observed by
        # (point, path class), joined on read).
        self.index = FactIndex()
        # General evidence for fixpoint drivers such as ``tharkun.avengers``.
        # ``FactStore`` remains the compact subject-oriented store used by the
        # branch driver.  This index adds explicit semantic identity and full
        # derivations for artifacts (templates, recipes, replay outcomes) whose
        # producers must not be part of equality.
        self._evidence: dict[EvidenceKey, Evidence] = {}
        self._evidence_log: list[EvidenceKey] = []

    # --- knowledge: read/write the lattice --------------------------------------
    def learn_prior(self, point: tuple, state: FactState) -> None:
        """Record what static analysis worked out BEFORE the run, at *point*."""
        self.index.learn_prior(point, state)
        # A prior is genuinely new information — re-open anything written off under a
        # thinner picture. Cheap, and keeps the exhaustion gate honest.
        self.bump("prior learned")

    def observe(self, point: tuple, path_class: str, state: FactState) -> None:
        """Record what an execution showed at *point* along one path class."""
        self.index.observe(point, path_class, state)

    def at(self, point: tuple) -> FactState:
        """Everything believed at *point*: the static prior joined with what ran."""
        return self.index.at(point)

    def knows(self, point: tuple) -> bool:
        """True when anything at all is recorded at *point* (prior or observed)."""
        if point in self.index.prior:
            return True
        return any(p == point for (p, _c) in self.index.observed)

    def proved_dead(self, point: tuple) -> bool:
        """True only when *point* was explicitly recorded unreachable — a dead-edge certificate.

        ``at`` returns ``BOTTOM`` for an unrecorded point too (a forward join analysis starts
        every point at ``BOTTOM``), so ``at(point).is_bottom()`` alone cannot tell "proved
        dead" from "never reached". Callers that PRUNE work on the strength of a dead verdict
        must use this: it requires the point to be in the prior *and* for nothing observed to
        have contradicted the certificate by actually reaching it.
        """
        return point in self.index.prior and self.at(point).is_bottom()

    # --- canonical evidence for fixpoint pipelines -----------------------------
    def publish(
        self,
        kind: str,
        key: str,
        value: Any,
        *,
        producer: str,
        parents: tuple[EvidenceKey, ...] = (),
    ) -> bool:
        """Publish evidence and return whether its semantic identity is new.

        ``(kind, key)`` is the canonical identity.  A second producer retains a
        second :class:`Derivation`, while the evidence log and epoch move only
        for a new semantic item.  A producer reusing an identity for a different
        value is a programming error: silently choosing either value would make
        replay and provenance disagree.
        """

        identity = (kind, key)
        derivation = Derivation(producer, tuple(parents))
        current = self._evidence.get(identity)
        if current is None:
            self._evidence[identity] = Evidence(kind, key, value, (derivation,))
            self._evidence_log.append(identity)
            self.bump("evidence published")
            return True
        if current.value != value:
            raise ValueError(
                f"evidence identity {identity!r} already names a different value"
            )
        if derivation not in current.derivations:
            self._evidence[identity] = Evidence(
                current.kind,
                current.key,
                current.value,
                current.derivations + (derivation,),
            )
        return False

    def evidence(self, kind: str | None = None) -> tuple[Evidence, ...]:
        """Return canonical evidence in first-publication order."""

        return tuple(
            self._evidence[identity]
            for identity in self._evidence_log
            if kind is None or identity[0] == kind
        )

    def evidence_item(self, kind: str, key: str) -> Evidence | None:
        """Return one canonical item, including every known derivation."""

        return self._evidence.get((kind, key))

    def evidence_since(
        self,
        cursor: int = 0,
        *,
        kinds: set[str] | frozenset[str] | None = None,
    ) -> tuple[int, tuple[Evidence, ...]]:
        """Return semantic evidence published after *cursor* and a new cursor.

        The cursor always advances across the full log even when ``kinds``
        filters the returned items.  This lets independent passes keep cheap,
        stable cursors without scanning the whole knowledge base each round.
        """

        if cursor < 0 or cursor > len(self._evidence_log):
            raise ValueError(f"invalid evidence cursor {cursor}")
        identities = self._evidence_log[cursor:]
        selected = tuple(
            self._evidence[identity]
            for identity in identities
            if kinds is None or identity[0] in kinds
        )
        return len(self._evidence_log), selected

    # --- the persisted log ([internal ref]) ------------------------------------------------
    def export_evidence(
        self, kinds: set[str] | frozenset[str] | None = None
    ) -> list[dict[str, Any]]:
        """The evidence list in first-publication order, as plain JSON records.

        This is the persisted log of the bootstrap loop (docs/BOOTSTRAP.md "Determinism and replay"): one
        ``{"kind", "key", "value", "derivations": [{"producer", "parents": [[kind, key], ...]}]}`` record per
        canonical item.  A value that is not JSON (a live Avengers recipe object) raises ``TypeError`` naming
        the item: a log that silently dropped items could not be replayed.  Filter with ``kinds`` to export
        only the JSON-valued kinds of a mixed knowledge base.
        """

        records: list[dict[str, Any]] = []
        for item in self.evidence():
            if kinds is not None and item.kind not in kinds:
                continue
            try:
                json.dumps(item.value, allow_nan=False)
            except (TypeError, ValueError) as error:
                raise TypeError(f"evidence {item.identity!r} is not JSON and cannot be logged: {error}") from error
            records.append({
                "kind": item.kind,
                "key": item.key,
                "value": item.value,
                "derivations": [
                    {"producer": d.producer, "parents": [list(parent) for parent in d.parents]}
                    for d in item.derivations
                ],
            })
        return records

    def import_evidence(self, records: Iterable[dict[str, Any]]) -> int:
        """Publish exported records in order (every derivation kept); the number of new items.

        Importing into a knowledge base that already holds an item with another value raises, as
        :meth:`publish` does: a log is replayed into an empty hub, not merged into a divergent one.
        """

        added = 0
        for record in records:
            derivations = record.get("derivations") or [{"producer": "import", "parents": []}]
            for derivation in derivations:
                parents = tuple((str(kind), str(key)) for kind, key in derivation.get("parents", ()))
                added += self.publish(
                    record["kind"],
                    record["key"],
                    record["value"],
                    producer=derivation["producer"],
                    parents=parents,
                )
        return added

    # --- diagnostics ------------------------------------------------------------
    def knowledge_report(self) -> dict:
        """Size and clause-pressure of the knowledge half, for the phase-2 decision."""
        report = self.index.clause_pressure()
        report["evidence"] = len(self._evidence)
        return report

    def __repr__(self) -> str:                      # pragma: no cover - debugging aid
        sched = super().report()
        return (f"<KnowledgeBase points={len(self.index.prior)} "
                f"observed={len(self.index.observed)} strategies={list(sched)}>")


def program_point(path: str, lineno: int) -> tuple:
    """The canonical key for a program point — a file and a line.

    Coarser than a lattice ``Origin`` (which also names a branch edge): the prior and the
    observed index are keyed by *where in the program*, and both engines agree on that.
    """
    import os

    return (os.path.realpath(path), lineno)
