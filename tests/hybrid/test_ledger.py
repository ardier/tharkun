"""BlockerLedger: the persistent worklist records solve/prove-impossible/exhaust verdicts.

The ledger does not decide anything — it records what the engine already decided — so the
tests pin the state machine: terminal verdicts (SOLVED / IMPOSSIBLE) stick, EXHAUSTED is
not silently reset to PENDING on re-detection, and the summary counts add up.
"""
from __future__ import annotations

import unittest

from tharkun.hybrid.ledger import BlockerLedger, BlockerStatus

A = ("/pkg/a.py", 10, "then", "entry")
B = ("/pkg/b.py", 22, "else", "callee")
C = ("/pkg/c.py", 5, "then", "entry")


class TestBlockerLedger(unittest.TestCase):
    def test_detect_is_pending(self) -> None:
        led = BlockerLedger()
        led.note_detected(A, label="a", value=3)
        self.assertEqual(led.counts()["pending"], 1)
        self.assertEqual(led.summary().detected, 1)

    def test_solved_is_terminal(self) -> None:
        led = BlockerLedger()
        led.note_detected(A, label="a")
        led.mark_solved(A, seed_hash="abc", inputs={"x": 1})
        # re-detection must not downgrade a solved blocker
        led.note_detected(A, label="a")
        e = led.entries()[0]
        self.assertIs(e.status, BlockerStatus.SOLVED)
        self.assertEqual(e.seed_hash, "abc")
        self.assertEqual(led.summary().solved, 1)

    def test_impossible_never_overwrites_solved(self) -> None:
        led = BlockerLedger()
        led.mark_solved(A, seed_hash="abc")
        led.mark_impossible(A)               # a witness must win over a (spurious) dead-proof
        self.assertIs(led.entries()[0].status, BlockerStatus.SOLVED)

    def test_exhausted_not_reset_by_redetect(self) -> None:
        led = BlockerLedger()
        led.note_detected(B, label="b")
        led.mark_exhausted(B, retriable=True)
        led.note_detected(B, label="b")      # still enumerable next round, still exhausted
        self.assertIs(led.entries()[0].status, BlockerStatus.EXHAUSTED)
        self.assertTrue(led.entries()[0].retriable)

    def test_exhausted_can_be_solved_later(self) -> None:
        led = BlockerLedger()
        led.note_detected(B)
        led.mark_exhausted(B)
        led.mark_solved(B, seed_hash="z")    # KB re-opened it; a later round cracked it
        self.assertIs(led.entries()[0].status, BlockerStatus.SOLVED)

    def test_summary_counts_add_up(self) -> None:
        led = BlockerLedger()
        led.note_detected(A); led.mark_solved(A)
        led.note_detected(B); led.mark_impossible(B)
        led.note_detected(C)                 # stays pending
        s = led.summary()
        self.assertEqual(s.detected, 3)
        self.assertEqual((s.solved, s.impossible, s.pending, s.exhausted), (1, 1, 1, 0))
        self.assertEqual(s.solved_labels + s.impossible_labels, ["", ""])  # no labels given

    def test_value_takes_the_max(self) -> None:
        led = BlockerLedger()
        led.note_detected(A, value=2)
        led.note_detected(A, value=7)
        led.note_detected(A, value=4)
        self.assertEqual(led.entries()[0].value, 7)

    def test_reconcile_retires_a_covered_pending_blocker(self) -> None:
        led = BlockerLedger()
        led.note_detected(A, label="a", body_line=10)   # A = ("/pkg/a.py", 10, ...)
        n = led.reconcile_covered({"/pkg/a.py": {10, 11}})
        self.assertEqual(n, 1)
        self.assertIs(led.entries()[0].status, BlockerStatus.RETIRED)
        self.assertEqual(led.summary().retired, 1)

    def test_reconcile_retires_an_exhausted_blocker(self) -> None:
        led = BlockerLedger()
        led.note_detected(B, body_line=22)
        led.mark_exhausted(B)
        led.reconcile_covered({"/pkg/b.py": {22}})       # Atheris covered it after we gave up
        self.assertIs(led.entries()[0].status, BlockerStatus.RETIRED)

    def test_reconcile_leaves_solved_and_uncovered_alone(self) -> None:
        led = BlockerLedger()
        led.note_detected(A, body_line=10); led.mark_solved(A, seed_hash="z")
        led.note_detected(C, body_line=5)                # C body NOT in the covered set
        led.reconcile_covered({"/pkg/a.py": {10}, "/pkg/c.py": {99}})
        by = {e.scope: e.status for e in led.entries()}
        self.assertIs(by[A], BlockerStatus.SOLVED)       # a witness we wrote stays SOLVED
        self.assertIs(by[C], BlockerStatus.PENDING)      # uncovered stays pending

    def test_redetect_after_retire_reopens_to_pending(self) -> None:
        led = BlockerLedger()
        led.note_detected(A, body_line=10)
        led.reconcile_covered({"/pkg/a.py": {10}})
        self.assertIs(led.entries()[0].status, BlockerStatus.RETIRED)
        led.note_detected(A, body_line=10)               # enumerable again ⇒ uncovered again
        self.assertIs(led.entries()[0].status, BlockerStatus.PENDING)


if __name__ == "__main__":
    unittest.main()
