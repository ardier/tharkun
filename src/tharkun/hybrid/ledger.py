"""BlockerLedger — the persistent worklist of blockers the hybrid loop is working.

Every uncovered frontier the assist detects is a *blocker*: a guard whose body the
fuzzer has not entered. The loop's job, per the design, is to move each blocker to one
of two terminal verdicts —

  * **SOLVED** — a seed was recoded that takes the branch, or
  * **IMPOSSIBLE** — a sound decider certified the branch dead —

or to hold it as work still to do (**PENDING**, never attacked yet; **EXHAUSTED**, the
portfolio resisted but a new fact may re-open it).

The ledger does not *decide* any of this; it *records* the decisions the engine already
makes (``find_frontiers`` detects, ``_frontier_dead`` proves impossible, ``drive_to_line``
solves or exhausts), so a run can answer "what are we solving for, and what have we proved
impossible?" in one place. It is held on the ``AssistEngine`` and persists across rounds —
the accumulating worklist, not a per-round snapshot.

Keyed by the same scope tuple the package path already uses for exhaustion gating:
``(realpath, branch_line, desired_branch, entry_func)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

Scope = tuple  # (realpath: str, branch_line: int, desired_branch: str, entry_func: str)


class BlockerStatus(str, Enum):
    PENDING = "pending"        # detected, uncovered, not yet resolved
    SOLVED = "solved"          # WE recoded a seed that takes the branch
    IMPOSSIBLE = "impossible"  # a sound decider certified the branch dead
    EXHAUSTED = "exhausted"    # the portfolio resisted; epoch-gated, may re-open
    RETIRED = "retired"        # its body got covered by ANOTHER tactic (Atheris, an
                               # incidental reach) before we solved it — off the worklist


# Terminal verdicts never downgrade on re-detection: a witness or a proof, once had, holds.
_TERMINAL = frozenset({BlockerStatus.SOLVED, BlockerStatus.IMPOSSIBLE})
# Statuses that reconcile-against-coverage may move to RETIRED (uncovered, still open work).
_OPEN = frozenset({BlockerStatus.PENDING, BlockerStatus.EXHAUSTED})


@dataclass
class BlockerEntry:
    scope: Scope
    status: BlockerStatus = BlockerStatus.PENDING
    label: str = ""              # human-readable "file Lnn/side via func"
    value: int = 0               # uncovered_gated at detection (worklist priority)
    body_line: int = 0           # first line of the uncovered side — coverage retires it
    seed_hash: str | None = None
    inputs: dict[str, Any] | None = None
    retriable: bool = False      # for EXHAUSTED: a randomized search may re-open it

    @property
    def file(self) -> str:
        return self.scope[0]


@dataclass
class LedgerSummary:
    detected: int = 0
    solved: int = 0
    impossible: int = 0
    pending: int = 0
    exhausted: int = 0
    retired: int = 0
    solved_labels: list[str] = field(default_factory=list)
    impossible_labels: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"{self.detected} detected, {self.solved} solved, "
                f"{self.impossible} impossible, {self.retired} retired (other tactic), "
                f"{self.exhausted} exhausted, {self.pending} pending")


@dataclass
class BlockerLedger:
    _entries: dict[Scope, BlockerEntry] = field(default_factory=dict)

    def _get(self, scope: Scope, label: str = "", value: int = 0) -> BlockerEntry:
        e = self._entries.get(scope)
        if e is None:
            e = BlockerEntry(scope=scope, label=label, value=value)
            self._entries[scope] = e
        return e

    def note_detected(self, scope: Scope, *, label: str = "", value: int = 0,
                      body_line: int = 0) -> None:
        """A frontier was found (its body is uncovered, or it would not be a frontier). New →
        PENDING; terminal or EXHAUSTED → left alone."""
        e = self._get(scope, label, value)
        if label:
            e.label = label
        if body_line:
            e.body_line = body_line
        e.value = max(e.value, value)
        # Being enumerable again means the body is uncovered again — but do not disturb a
        # terminal verdict, and do not reset EXHAUSTED to PENDING (the KB's epoch decides when
        # an exhausted scope re-opens). A RETIRED entry re-detected IS uncovered again → PENDING.
        if e.status not in _TERMINAL and e.status is not BlockerStatus.EXHAUSTED:
            e.status = BlockerStatus.PENDING

    def reconcile_covered(self, covered_by_file: dict[str, set]) -> int:
        """Retire open blockers whose body another tactic has since covered.

        Called each round with live coverage (per realpath'd file). Any PENDING/EXHAUSTED
        blocker whose body line is now covered — by Atheris's own background progress, a
        prior drive's seed, or an incidental reach — is moved to RETIRED and drops off the
        active worklist, so neither a sound decider nor static is spent on an already-open branch.
        SOLVED/IMPOSSIBLE are terminal and untouched. Returns how many were retired."""
        n = 0
        for e in self._entries.values():
            if e.status in _OPEN and e.body_line and e.body_line in covered_by_file.get(e.file, ()):
                e.status = BlockerStatus.RETIRED
                n += 1
        return n

    def mark_solved(self, scope: Scope, *, seed_hash: str | None = None,
                    inputs: dict[str, Any] | None = None, label: str = "") -> None:
        e = self._get(scope, label)
        e.status = BlockerStatus.SOLVED
        e.seed_hash = seed_hash
        e.inputs = dict(inputs) if inputs else None

    def mark_impossible(self, scope: Scope, *, label: str = "") -> None:
        """Record a a sound decider dead-proof. Never overwrites a SOLVED entry (a proved-dead
        frontier is pruned before it can be driven, so a SOLVED/IMPOSSIBLE clash cannot
        legitimately arise — guarded defensively so a witness always wins if it did)."""
        e = self._get(scope, label)
        if e.status is not BlockerStatus.SOLVED:
            e.status = BlockerStatus.IMPOSSIBLE

    def mark_exhausted(self, scope: Scope, *, retriable: bool = False, label: str = "") -> None:
        e = self._get(scope, label)
        if e.status in _TERMINAL:
            return
        e.status = BlockerStatus.EXHAUSTED
        e.retriable = retriable

    # --- reporting ---------------------------------------------------------

    def counts(self) -> dict[str, int]:
        out = {s.value: 0 for s in BlockerStatus}
        for e in self._entries.values():
            out[e.status.value] += 1
        return out

    def entries(self) -> list[BlockerEntry]:
        return list(self._entries.values())

    def summary(self) -> LedgerSummary:
        c = self.counts()
        return LedgerSummary(
            detected=len(self._entries),
            solved=c["solved"], impossible=c["impossible"],
            pending=c["pending"], exhausted=c["exhausted"], retired=c["retired"],
            solved_labels=[e.label for e in self._entries.values()
                           if e.status is BlockerStatus.SOLVED],
            impossible_labels=[e.label for e in self._entries.values()
                               if e.status is BlockerStatus.IMPOSSIBLE],
        )
