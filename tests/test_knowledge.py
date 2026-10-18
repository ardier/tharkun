"""One door onto what the pipeline knows and what it has decided.

The scheduler (FactStore) and the lattice (FactIndex) were separate, and the lattice had no
production caller — so FactState and canonicalization were dead in production. KnowledgeBase
is the single object every component holds; it must be a drop-in FactStore (every existing
scheduling call keeps working) while adding the knowledge door.
"""
from __future__ import annotations

import unittest

from tharkun.knowledge import KnowledgeBase, program_point
from tharkun.lattice import TOP, Origin
from tharkun.facts import DecisionKey, Fact, FactStore


class TestItIsADropInScheduler(unittest.TestCase):
    """Every FactStore method a live caller uses must work unchanged on the facade."""

    def setUp(self) -> None:
        self.kb = KnowledgeBase()

    def test_is_a_factstore(self) -> None:
        self.assertIsInstance(self.kb, FactStore)

    def test_decision_memo_still_works(self) -> None:
        key = DecisionKey.of("/m.py", 10, "then", "True")
        runs = []
        for _ in range(3):
            self.kb.lookup_or_run(key, "static", lambda: runs.append(1) or "sat")
        self.assertEqual(len(runs), 1)

    def test_facts_and_types_still_work(self) -> None:
        self.kb.add(Fact("type", "x", "int", source="static"))
        self.assertEqual(self.kb.types_for("x"), ["int"])

    def test_exhaustion_gate_still_works(self) -> None:
        scope = ("/m.py", 5, "then", "f")
        self.kb.mark_exhausted(scope)
        self.assertTrue(self.kb.is_exhausted(scope))
        self.kb.bump("learned")
        self.assertFalse(self.kb.is_exhausted(scope))


class TestTheKnowledgeDoor(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = KnowledgeBase()
        self.p = program_point("/m.py", 10)

    def test_prior_and_observed_join_on_read(self) -> None:
        self.kb.learn_prior(self.p, TOP.with_types("token", ["list"]))
        self.kb.observe(self.p, "cls1", TOP.with_types("token", ["bytes"]))
        self.assertEqual(self.kb.at(self.p).types_for("token"),
                         frozenset({"list", "bytes"}))

    def test_an_unseen_point_is_bottom(self) -> None:
        self.assertTrue(self.kb.at(program_point("/never.py", 1)).is_bottom())

    def test_learning_a_prior_bumps_the_epoch(self) -> None:
        scope = ("/m.py", 5, "then", "f")
        self.kb.mark_exhausted(scope)
        self.kb.learn_prior(self.p, TOP.with_types("x", ["int"]))
        self.assertFalse(self.kb.is_exhausted(scope))

    def test_program_point_is_stable(self) -> None:
        self.assertEqual(program_point("/a/../m.py", 10), program_point("/m.py", 10))


class TestCanonicalEvidence(unittest.TestCase):
    def setUp(self) -> None:
        self.kb = KnowledgeBase()

    def test_identity_excludes_producer_but_retains_both_derivations(self) -> None:
        self.assertTrue(
            self.kb.publish("recipe", "Box(0, 1)", {"lower": 0, "upper": 1},
                            producer="hypothesis")
        )
        epoch = self.kb.epoch
        self.assertFalse(
            self.kb.publish("recipe", "Box(0, 1)", {"lower": 0, "upper": 1},
                            producer="pynguin")
        )
        item = self.kb.evidence_item("recipe", "Box(0, 1)")
        self.assertEqual([d.producer for d in item.derivations],
                         ["hypothesis", "pynguin"])
        self.assertEqual(self.kb.epoch, epoch,
                         "provenance alone must not wake semantic consumers")

    def test_delta_cursor_returns_each_semantic_item_once(self) -> None:
        self.kb.publish("goal", "g", "goal", producer="user")
        cursor, first = self.kb.evidence_since()
        self.assertEqual([(x.kind, x.key) for x in first], [("goal", "g")])
        self.kb.publish("recipe", "r", "recipe", producer="search")
        cursor, second = self.kb.evidence_since(cursor, kinds={"recipe"})
        self.assertEqual([(x.kind, x.key) for x in second], [("recipe", "r")])
        self.assertEqual(self.kb.evidence_since(cursor)[1], ())

    def test_same_identity_cannot_silently_change_value(self) -> None:
        self.kb.publish("recipe", "r", 1, producer="a")
        with self.assertRaises(ValueError):
            self.kb.publish("recipe", "r", 2, producer="b")


if __name__ == "__main__":
    unittest.main()


class TestDriverObservesIntoTheIndex(unittest.TestCase):
    """C2: every drive records observed types into the lattice index, write-only.

    Nothing reads the index yet, so this changes no behaviour — it is the tier-2 write
    side of the shared base. Observations on the same path aggregate (join); on different
    paths they stay distinct until the widening budget.
    """

    def _drive(self, kb):
        import ast
        import os

        from tharkun.driver import drive_to_line

        fix = os.path.join(os.path.dirname(__file__), "fixtures", "operand_bank.py")
        ifs = [n for n in ast.walk(ast.parse(open(fix).read())) if isinstance(n, ast.If)]
        drive_to_line(fix, "gate", ifs[0].body[0].lineno,
                      {"token": b"\x00\x00", "other": 0}, max_rounds=4, store=kb)

    def test_a_drive_populates_the_observed_index(self) -> None:
        kb = KnowledgeBase()
        self._drive(kb)
        self.assertGreater(len(kb.index.observed), 0)

    def test_observed_types_are_the_real_runtime_types(self) -> None:
        kb = KnowledgeBase()
        self._drive(kb)
        seen = set()
        for state in kb.index.observed.values():
            for name, var in state.vars.items():
                if var.types.types:
                    seen |= {(name, t) for t in var.types.types}
        self.assertIn(("token", "bytes"), seen)   # not "list" (static) — what actually ran
        self.assertIn(("other", "int"), seen)
