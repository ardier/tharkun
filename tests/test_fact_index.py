"""Aggregating facts as fuzzing runs, without the state exploding.

Atheris produces thousands of inputs. Keeping a state per input explodes; the lattice is
what prevents that, because two states sharing a key become one. So the whole design is
"what is the key" and "what happens at budget", and these tests pin both.

Two things here are easy to get wrong in ways nothing downstream would catch:

* Combining the static prior with an observation must be a JOIN. Static inference is a
  heuristic, not a sound over-approximation — meeting a wrong prior with a real
  observation yields BOTTOM, i.e. "this point is unreachable", about a path that
  demonstrably just executed.
* The bound must come from WIDENING, not truncation. Dropping entries loses facts
  silently; joining them loses precision visibly and keeps soundness.
"""
from __future__ import annotations

import unittest

from tharkun.lattice import (
    BOTTOM,
    TOP,
    FactIndex,
    Origin,
    path_class,
)


class TestPathClassIsDerivableFromWhatTheTracerActuallyRecords(unittest.TestCase):
    """There is no branch-decision sequence anywhere; the executed-line set is what exists."""

    def test_the_same_lines_give_the_same_class(self) -> None:
        self.assertEqual(path_class({1, 2, 3}), path_class({3, 2, 1}))

    def test_different_lines_give_different_classes(self) -> None:
        self.assertNotEqual(path_class({1, 2, 3}), path_class({1, 2, 4}))

    def test_it_works_on_the_per_file_shape_the_driver_produces(self) -> None:
        by_path = {"a.py": {1, 2}, "b.py": {7}}
        self.assertEqual(path_class(by_path), path_class({"b.py": {7}, "a.py": {2, 1}}))

    def test_a_file_boundary_is_not_ignored(self) -> None:
        self.assertNotEqual(path_class({"a.py": {1}}), path_class({"b.py": {1}}))

    def test_it_is_stable_across_processes(self) -> None:
        # A content hash, not hash() — which is salted per process.
        import subprocess
        import sys

        code = ("import sys; sys.path.insert(0, 'src');"
                "from tharkun.lattice import path_class;"
                "print(path_class({1, 2, 3}))")
        runs = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True).stdout.strip() for _ in range(2)}
        self.assertEqual(len(runs), 1)


class TestTheTwoTiersCombineByJoin(unittest.TestCase):
    """The correction that matters: a heuristic prior must not veto a real observation."""

    def setUp(self) -> None:
        self.ix = FactIndex()
        self.p = ("m.py", 10)

    def test_an_observation_contradicting_the_prior_widens_rather_than_kills(self) -> None:
        # Static inference reading `token[:2]` says list; the run says bytes. `list` is
        # incomplete, not false — and meeting would declare the point unreachable.
        self.ix.learn_prior(self.p, TOP.with_types("token", ["list"]))
        self.ix.observe(self.p, path_class({1}), TOP.with_types("token", ["bytes"]))
        self.assertEqual(self.ix.at(self.p).types_for("token"), frozenset({"list", "bytes"}))
        self.assertFalse(self.ix.at(self.p).is_bottom())

    def test_a_point_with_nothing_recorded_is_bottom_not_a_crash(self) -> None:
        self.assertTrue(self.ix.at(("never.py", 1)).is_bottom())

    def test_several_analyses_may_speak_to_the_same_point(self) -> None:
        self.ix.learn_prior(self.p, TOP.with_types("x", ["int"]))
        self.ix.learn_prior(self.p, TOP.with_types("x", ["float"]))
        self.assertEqual(self.ix.prior[self.p].types_for("x"), frozenset({"int", "float"}))


class TestAggregationAlongTheSamePath(unittest.TestCase):
    """Same path class ⇒ joined. That IS the aggregation."""

    def setUp(self) -> None:
        self.ix = FactIndex()
        self.p = ("m.py", 10)

    def test_two_inputs_on_the_same_path_become_one_entry(self) -> None:
        cls = path_class({1, 2, 3})
        self.ix.observe(self.p, cls, TOP.with_types("x", ["int"]))
        self.ix.observe(self.p, cls, TOP.with_types("x", ["str"]))
        self.assertEqual(len(self.ix.observed), 1)
        self.assertEqual(self.ix.at(self.p).types_for("x"), frozenset({"int", "str"}))

    def test_different_paths_stay_separate_until_budget(self) -> None:
        self.ix.observe(self.p, path_class({1}), TOP.with_types("x", ["int"]))
        self.ix.observe(self.p, path_class({2}), TOP.with_types("x", ["str"]))
        self.assertEqual(len(self.ix.observed), 2)

    def test_one_path_class_can_be_read_on_its_own(self) -> None:
        a, b = path_class({1}), path_class({2})
        self.ix.observe(self.p, a, TOP.with_types("x", ["int"]))
        self.ix.observe(self.p, b, TOP.with_types("x", ["str"]))
        self.assertEqual(self.ix.at(self.p, cls=a).types_for("x"), frozenset({"int"}))


class TestBoundednessComesFromWideningNotTruncation(unittest.TestCase):
    """The claim this design actually makes."""

    def setUp(self) -> None:
        self.ix = FactIndex(budget=4)
        self.p = ("m.py", 10)

    def _flood(self, n: int) -> None:
        for i in range(n):
            self.ix.observe(self.p, path_class({1, i + 100}), TOP.with_types("x", ["int"]))

    def test_entries_stay_bounded_however_many_paths_arrive(self) -> None:
        self._flood(200)
        self.assertLessEqual(len(self.ix.observed), self.ix.budget)

    def test_widening_joins_rather_than_dropping(self) -> None:
        # Every type seen must survive. Truncation would silently lose whichever entries
        # were evicted; joining loses precision instead, which is the sound trade.
        for i, t in enumerate(["int", "str", "bytes", "list", "tuple", "float"]):
            self.ix.observe(self.p, path_class({1, i + 100}), TOP.with_types("x", [t]))
        self.assertEqual(self.ix.at(self.p).types_for("x"),
                         frozenset({"int", "str", "bytes", "list", "tuple", "float"}))

    def test_a_widened_point_stops_distinguishing_paths(self) -> None:
        self._flood(50)
        self.assertIn(self.p, self.ix.widened)
        self.assertEqual(sum(1 for k in self.ix.observed if k[0] == self.p), 1)

    def test_widening_one_point_leaves_others_alone(self) -> None:
        other = ("m.py", 99)
        self._flood(50)
        self.ix.observe(other, path_class({7}), TOP.with_types("y", ["int"]))
        self.assertNotIn(other, self.ix.widened)
        self.assertEqual(self.ix.at(other).types_for("y"), frozenset({"int"}))

    def test_precision_degrades_but_does_not_collapse(self) -> None:
        self._flood(200)
        self.assertFalse(self.ix.at(self.p).is_top(),
                         "widening threw away everything instead of merging it")


class TestMustFactsSurviveAggregationCorrectly(unittest.TestCase):
    def test_a_constraint_on_every_path_survives(self) -> None:
        ix = FactIndex()
        p = ("m.py", 10)
        for i in range(3):
            ix.observe(p, path_class({i}),
                       TOP.with_clause("(pad != 0)", Origin("m.py", 10 + i)))
        self.assertIn("(pad != 0)", ix.at(p).requires())

    def test_a_constraint_on_only_one_path_does_not(self) -> None:
        ix = FactIndex()
        p = ("m.py", 10)
        ix.observe(p, path_class({1}), TOP.with_clause("(x > 5)", Origin("m.py", 10)))
        ix.observe(p, path_class({2}), TOP.with_clause("(y < 3)", Origin("m.py", 11)))
        self.assertEqual(ix.at(p).requires(), "True")


class TestClausePressureInstrumentation(unittest.TestCase):
    """Data for the phase-2 decision, so it is not a guess."""

    def test_it_reports_the_duplication_ratio(self) -> None:
        ix = FactIndex()
        p = ("m.py", 10)
        # Two clauses canonicalization keeps DISTINCT — dividing through a coefficient is
        # beyond orientation/fold (only additive constants move) — but a full decision
        # procedure would collapse `x*2 == 6` and `x == 3`. That residue is what
        # clause_pressure measures for the phase-2 decision.
        from tharkun.solve.canonicalize import canonicalize as _cz
        assert _cz("x * 2 == 6") != _cz("x == 3")       # guard: still distinct today
        ix.observe(p, path_class({1}),
                   TOP.with_clause("x * 2 == 6", Origin("m.py", 1))
                      .with_clause("x == 3", Origin("m.py", 2)))
        got = ix.clause_pressure()
        self.assertEqual(got["distinct"], 2, "two syntactically distinct clauses")

    def test_it_reports_the_structural_counts(self) -> None:
        ix = FactIndex(budget=2)
        p = ("m.py", 10)
        ix.learn_prior(p, TOP.with_types("x", ["int"]))
        for i in range(5):
            ix.observe(p, path_class({i}), TOP.with_types("x", ["int"]))
        got = ix.clause_pressure()
        self.assertEqual(got["points"], 1)
        self.assertGreaterEqual(got["widened"], 1)


if __name__ == "__main__":
    unittest.main()
