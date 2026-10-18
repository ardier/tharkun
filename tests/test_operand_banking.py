"""A failed search must leave behind what it saw — as the INVERSION, not the raw value.

`_reach_by_search` runs the program hundreds of times and every run observes what the
blocking comparison demanded. Those observations are the only thing a failed search
produces, so throwing them away makes the failure total.

Two bugs this pins, both measured:

1. The banking stored `c.demanded` — the raw number 13987 — while `Correspondence` also
   carried `c.proposal`, the same number already inverted back through `int.from_bytes`
   into `b'6\\xa3'`. RedQueen had done the hard part and the answer was discarded, leaving
   the next search to rediscover the encoding by chance. It did not.

2. Banking ran per trial, producing **37 facts from a single search** whose proposals
   differed only in each trial's incidental filler. Worse than the volume: `FactStore.add`
   bumps the epoch and `is_exhausted` is epoch-gated, so a chatty search re-opens every
   written-off frontier on every round — defeating the skip measured at 26.7s -> 7.7s.
"""
from __future__ import annotations

import ast
import os
import sys
import unittest

from tharkun.facts import FactStore

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
MOD = os.path.join(FIX, "operand_bank.py")

try:
    import hypothesis  # noqa: F401
    HAVE = True
except ImportError:                                   # pragma: no cover
    HAVE = False

requires_hypothesis = unittest.skipUnless(HAVE, "hypothesis not installed")


def setUpModule() -> None:
    if FIX not in sys.path:
        sys.path.insert(0, FIX)


def _guards() -> list:
    return [n for n in ast.walk(ast.parse(open(MOD).read())) if isinstance(n, ast.If)]


def _search(store, hints, target_line, max_examples=60):
    import tharkun.driver as DRV

    outer = _guards()[0]
    DRV._STORE = store
    return DRV._reach_by_search(MOD, "gate", target_line, MOD,
                                {"token": b"\x00\x00", "other": 0},
                                {"token", "other"}, hints,
                                max_examples=max_examples, store=store,
                                guard_path=MOD, guard_line=outer.lineno)


def _banked(store, name="token"):
    return [f.value for f in store.get("operand", name)]


@requires_hypothesis
class TestTheInversionIsBankedNotTheRawValue(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FactStore()
        self.target = _guards()[0].body[0].lineno      # inside the token gate

    def test_a_failed_search_banks_something(self) -> None:
        _search(self.store, [], self.target)
        self.assertTrue(_banked(self.store), "a failed search left nothing behind")

    def test_what_is_banked_is_the_encoded_form(self) -> None:
        # 13987 big-endian is b'\x36\xa3'. Banking the number instead would leave the
        # next consumer to rediscover the byte order and width by chance.
        _search(self.store, [], self.target)
        self.assertTrue(any(isinstance(v, bytes) and v.startswith(b"\x36\xa3")
                            for v in _banked(self.store)),
                        f"expected the inversion, got {_banked(self.store)!r}")

    def test_the_raw_demanded_number_is_not_what_lands(self) -> None:
        _search(self.store, [], self.target)
        self.assertNotIn(13987, _banked(self.store))


@requires_hypothesis
class TestBankingIsBoundedPerSearch(unittest.TestCase):
    """Per-trial banking produced 37 facts from one search and moved the epoch 38 times."""

    def setUp(self) -> None:
        self.store = FactStore()
        self.target = _guards()[0].body[0].lineno

    def test_a_single_search_banks_only_a_handful(self) -> None:
        _search(self.store, [], self.target, max_examples=120)
        self.assertLessEqual(len(_banked(self.store)), 4)

    def test_the_epoch_barely_moves(self) -> None:
        # The epoch gates frontier exhaustion; a search that moves it dozens of times
        # re-opens every written-off frontier on every round.
        _search(self.store, [], self.target, max_examples=120)
        self.assertLessEqual(self.store.epoch, 10)

    def test_duplicate_proposals_are_not_banked_twice(self) -> None:
        _search(self.store, [], self.target, max_examples=120)
        vals = [repr(v) for v in _banked(self.store)]
        self.assertEqual(len(vals), len(set(vals)))


@requires_hypothesis
class TestABankedInversionAimsTheNextSearch(unittest.TestCase):
    """The point of banking: round 2 succeeds where round 1 failed."""

    def test_round_two_solves_what_round_one_could_not(self) -> None:
        import operand_bank

        store = FactStore()
        target = _guards()[0].body[0].lineno

        cold = _search(store, [], target)
        self.assertIsNone(cold, "the cold search was supposed to fail")

        hints = _banked(store)
        self.assertTrue(hints)

        warm = _search(store, hints, target)
        self.assertIsNotNone(warm, "the banked inversion did not aim the next search")

        # Execution is the judge, not the search's own claim.
        merged = {"token": b"\x00\x00", "other": 0, **warm}
        self.assertEqual(operand_bank.gate(**merged), "FIRST")


if __name__ == "__main__":
    unittest.main()
