"""The abstract state's algebra, checked against the laws rather than by example.

A join that is merely plausible is worthless: every solver downstream takes the path
condition as given, so a merge that quietly asserts too much is a silent unsoundness
rather than a wrong answer someone notices. These tests check the laws — commutativity,
associativity, idempotence, the identities, and the partial order — because those are what
make the operations mean what their names say.

The two directions are the thing most worth pinning. May-facts (types, values) UNION at a
merge; must-facts (clauses) INTERSECT. Getting that backwards for clauses would claim a
guard holds when only one predecessor established it, and nothing downstream would catch
it.

Motivating bug: `drive_to_line` keeps its path condition as a list of guard strings that
survives backtracking, so after abandoning a path it can hold `(x > 5)` and
`(not (x > 5))` at the same time.
"""
from __future__ import annotations

import unittest

from tharkun.lattice import (
    BOTTOM,
    TOP,
    Clause,
    Clauses,
    FactState,
    Interval,
    Origin,
    TypeSet,
    ValueSet,
    VarState,
)


def _c(text: str, line: int = 0, polarity: str = "") -> Clauses:
    return Clauses(frozenset({Clause(text, Origin("m.py", line, polarity=polarity))}))


class TestLawsHoldForEveryComponent(unittest.TestCase):
    """Commutative, associative, idempotent — for join and meet, on every component."""

    def samples(self):
        return [
            (TypeSet(frozenset({"int"})), TypeSet(frozenset({"str"})),
             TypeSet(frozenset({"int", "bytes"}))),
            (ValueSet(frozenset({1, 2})), ValueSet(frozenset({2, 3})),
             ValueSet(frozenset({9}))),
            (Interval(0, 10), Interval(5, 20), Interval(-5, 3)),
            (_c("a"), _c("b"), _c("a")),
            (VarState(types=TypeSet(frozenset({"int"}))),
             VarState(interval=Interval(0, 5)),
             VarState(values=ValueSet(frozenset({7})))),
        ]

    def test_join_is_commutative(self) -> None:
        for a, b, _ in self.samples():
            self.assertEqual(a.join(b), b.join(a), type(a).__name__)

    def test_meet_is_commutative(self) -> None:
        for a, b, _ in self.samples():
            self.assertEqual(a.meet(b), b.meet(a), type(a).__name__)

    def test_join_is_associative(self) -> None:
        for a, b, c in self.samples():
            self.assertEqual(a.join(b).join(c), a.join(b.join(c)), type(a).__name__)

    def test_meet_is_associative(self) -> None:
        for a, b, c in self.samples():
            self.assertEqual(a.meet(b).meet(c), a.meet(b.meet(c)), type(a).__name__)

    def test_both_are_idempotent(self) -> None:
        for a, _, _ in self.samples():
            self.assertEqual(a.join(a), a, type(a).__name__)
            self.assertEqual(a.meet(a), a, type(a).__name__)

    def test_join_over_approximates_both_inputs(self) -> None:
        # x ⊑ join(x, y): a merge may only ever lose precision, never gain it.
        for a, b, _ in self.samples():
            self.assertTrue(a.leq(a.join(b)), type(a).__name__)
            self.assertTrue(b.leq(a.join(b)), type(a).__name__)


class TestMayFactsUnion(unittest.TestCase):
    """Types and values are what a variable COULD be, so a merge keeps both."""

    def test_a_type_from_one_branch_survives_the_merge(self) -> None:
        merged = TypeSet(frozenset({"bytes"})).join(TypeSet(frozenset({"list"})))
        self.assertEqual(merged.types, frozenset({"bytes", "list"}))

    def test_unknown_on_either_side_admits_anything(self) -> None:
        self.assertTrue(TypeSet(frozenset({"int"})).join(TypeSet(None)).is_top())

    def test_meet_narrows_to_the_overlap(self) -> None:
        both = TypeSet(frozenset({"int", "str"})).meet(TypeSet(frozenset({"str", "bytes"})))
        self.assertEqual(both.types, frozenset({"str"}))

    def test_a_disjoint_meet_is_bottom(self) -> None:
        self.assertTrue(TypeSet(frozenset({"int"})).meet(TypeSet(frozenset({"str"}))).is_bottom())

    def test_values_widen_past_the_cap_rather_than_grow_forever(self) -> None:
        from tharkun.lattice import VALUE_SET_CAP

        big = ValueSet(frozenset(range(VALUE_SET_CAP)))
        self.assertTrue(big.join(ValueSet(frozenset({-1, -2}))).is_top())


class TestMustFactsIntersect(unittest.TestCase):
    """The direction that is expensive to get wrong."""

    def test_a_clause_from_only_one_branch_does_NOT_survive(self) -> None:
        # Union here would assert the guard holds when only one path established it.
        merged = _c("(x > 5)").join(_c("(y < 3)"))
        self.assertEqual(merged.clauses, frozenset())
        self.assertEqual(merged.render(), "True")

    def test_a_clause_on_both_branches_does_survive(self) -> None:
        merged = (_c("(x > 5)").meet(_c("(a)"))).join(_c("(x > 5)").meet(_c("(b)")))
        self.assertEqual({c.text for c in merged.clauses}, {"(x > 5)"})

    def test_meet_conjoins_because_refining_only_adds(self) -> None:
        both = _c("(x > 5)").meet(_c("(y < 3)"))
        self.assertEqual({c.text for c in both.clauses}, {"(x > 5)", "(y < 3)"})


class TestIdentities(unittest.TestCase):
    def test_top_is_the_identity_of_meet(self) -> None:
        s = TOP.with_types("x", ["int"]).with_clause("(x > 5)")
        self.assertEqual(s.meet(TOP), s)

    def test_bottom_is_the_identity_of_join(self) -> None:
        # An unreachable edge contributes nothing to a merge — which is exactly what a
        # dead-branch proof means.
        s = TOP.with_types("x", ["int"]).with_clause("(x > 5)")
        self.assertEqual(s.join(BOTTOM), s)
        self.assertEqual(BOTTOM.join(s), s)

    def test_bottom_absorbs_meet(self) -> None:
        self.assertTrue(TOP.with_types("x", ["int"]).meet(BOTTOM).is_bottom())

    def test_top_is_top_and_bottom_is_bottom(self) -> None:
        self.assertTrue(TOP.is_top())
        self.assertTrue(BOTTOM.is_bottom())
        self.assertFalse(TOP.is_bottom())


class TestBottomIsDetectedNotReturnedSilently(unittest.TestCase):
    """merge_leaf_facts can yield eq=∅ or lo>hi silently; this must not."""

    def test_an_empty_interval_is_bottom(self) -> None:
        self.assertTrue(Interval(10, 20).meet(Interval(0, 5)).is_bottom())

    def test_an_impossible_variable_makes_the_whole_state_unreachable(self) -> None:
        s = TOP.with_types("x", ["int"]).with_types("x", ["str"])
        self.assertTrue(s.is_bottom(), "a variable with no possible type left the state alive")

    def test_an_empty_value_set_is_bottom(self) -> None:
        self.assertTrue(ValueSet(frozenset({1})).meet(ValueSet(frozenset({2}))).is_bottom())


class TestIntervals(unittest.TestCase):
    def test_meet_raises_the_floor_and_lowers_the_ceiling(self) -> None:
        got = Interval(0, 100).meet(Interval(10, 50))
        self.assertEqual((got.lo, got.hi), (10, 50))

    def test_join_takes_the_hull(self) -> None:
        got = Interval(0, 10).join(Interval(50, 60))
        self.assertEqual((got.lo, got.hi), (0, 60))

    def test_an_unbounded_side_stays_unbounded_after_a_join(self) -> None:
        self.assertIsNone(Interval(0, None).join(Interval(5, 10)).hi)

    def test_exclusions_conjoin_on_meet(self) -> None:
        got = Interval(excluded=frozenset({1})).meet(Interval(excluded=frozenset({2})))
        self.assertEqual(got.excluded, frozenset({1, 2}))

    def test_an_exclusion_from_one_branch_is_DROPPED_on_join(self) -> None:
        # A value excluded on one branch is perfectly reachable via the other; carrying
        # the exclusion through would be unsound.
        got = Interval(excluded=frozenset({1, 2})).join(Interval(excluded=frozenset({2})))
        self.assertEqual(got.excluded, frozenset({2}))


class TestOriginAnchorsFactsToProgramPoints(unittest.TestCase):
    """The anchor a flat store cannot express."""

    def test_the_same_text_on_different_edges_is_a_distinct_RECORD(self) -> None:
        # As stored facts they stay distinct — the origin is real provenance and the
        # store keys on it. What must NOT depend on the origin is whether a merge
        # considers them the same constraint; see TestJoinMatchesOnMeaning.
        then = Clause("(x > 5)", Origin("m.py", 12, polarity="then"))
        els = Clause("(x > 5)", Origin("m.py", 12, polarity="else"))
        self.assertNotEqual(then, els)
        self.assertEqual(len({then, els}), 2)

    def test_the_same_text_at_the_same_edge_deduplicates(self) -> None:
        o = Origin("m.py", 12, polarity="then")
        self.assertEqual(len({Clause("(x > 5)", o), Clause("(x > 5)", o)}), 1)

    def test_the_cached_node_is_not_part_of_identity(self) -> None:
        # Two parses of the same source are distinct objects with no __eq__; including
        # the node would defeat the deduplication that makes Clauses a set.
        import ast

        o = Origin("m.py", 12)
        a = Clause("x > 5", o, ast.parse("x > 5", mode="eval").body)
        b = Clause("x > 5", o, ast.parse("x > 5", mode="eval").body)
        self.assertEqual(a, b)
        self.assertEqual(len({a, b}), 1)

    def test_origin_is_hashable_and_serializable(self) -> None:
        import json

        o = Origin("m.py", 12, 14, "then")
        hash(o)
        json.dumps({"path": o.path, "lineno": o.lineno, "polarity": o.polarity})

    def test_names_come_from_the_cached_node_when_present(self) -> None:
        import ast

        c = Clause("int.from_bytes(token[:2], 'big') == 13987", Origin(),
                   ast.parse("int.from_bytes(token[:2], 'big') == 13987", mode="eval").body)
        self.assertEqual(c.names, frozenset({"token"}))   # `int` is a builtin, not bindable

    def test_names_are_recovered_by_parsing_when_the_node_is_absent(self) -> None:
        self.assertEqual(Clause("len(key) != 7").names, frozenset({"key"}))

    def test_unparseable_text_yields_no_names_rather_than_raising(self) -> None:
        self.assertEqual(Clause("this is not ( python").names, frozenset())


class TestJoinMatchesOnMeaning(unittest.TestCase):
    """A merge asks whether the same CONSTRAINT holds, not where it was learned.

    Set intersection over Clause compares origins too, so the same constraint established
    at two different program points vanished at every merge — sound, since facts were lost
    rather than invented, but so lossy the join returned nothing in the common case.
    """

    def test_a_constraint_learned_at_two_points_survives_the_merge(self) -> None:
        a = TOP.with_clause("(x > 5)", Origin("m.py", 10, polarity="then"))
        b = TOP.with_clause("(x > 5)", Origin("m.py", 20, polarity="then"))
        self.assertIn("(x > 5)", a.join(b).requires())

    def test_it_survives_across_opposite_polarities_of_different_guards(self) -> None:
        # Two callers reach here having each pinned pad, one via the then edge of its
        # own guard and one via the else edge of another. The constraint still holds.
        a = TOP.with_clause("(pad != 0)", Origin("a.py", 3, polarity="then"))
        b = TOP.with_clause("(pad != 0)", Origin("b.py", 9, polarity="else"))
        self.assertIn("(pad != 0)", a.join(b).requires())

    def test_a_genuinely_one_sided_constraint_still_does_not_survive(self) -> None:
        # The must-direction is unchanged: matching on meaning must not weaken it.
        a = TOP.with_clause("(x > 5)", Origin("m.py", 10))
        b = TOP.with_clause("(y < 3)", Origin("m.py", 20))
        self.assertEqual(a.join(b).requires(), "True")

    def test_the_surviving_clause_keeps_a_real_origin(self) -> None:
        # One of the two actual program points, never an invented or blank one.
        a = TOP.with_clause("(x > 5)", Origin("m.py", 10, polarity="then"))
        b = TOP.with_clause("(x > 5)", Origin("m.py", 20, polarity="then"))
        surviving = next(iter(a.join(b).clauses.clauses))
        self.assertIn(surviving.origin.lineno, (10, 20))

    def test_meet_deduplicates_by_meaning_too(self) -> None:
        # Re-learning a constraint further along a path adds nothing, and letting it in
        # twice would grow the rendered precondition without bound.
        s = (TOP.with_clause("(x > 5)", Origin("m.py", 10))
                .with_clause("(x > 5)", Origin("m.py", 40)))
        self.assertEqual(len(s.clauses.clauses), 1)
        self.assertEqual(s.requires(), "(x > 5)")


class TestCanonicalizationMakesSpellingsMerge(unittest.TestCase):
    """The three measured failures that canonical clause text closes."""

    def test_a_constraint_written_two_ways_survives_a_join(self) -> None:
        # `Clauses.join` is intersection; without canonicalization (x > 5) and (5 < x)
        # are different clauses and the merge rendered `True`, losing a fact that holds.
        a = TOP.with_clause("(x > 5)", Origin("m.py", 10, polarity="then"))
        b = TOP.with_clause("(5 < x)", Origin("m.py", 20, polarity="then"))
        self.assertIn("(x > 5)", a.join(b).requires())

    def test_the_two_spellings_do_not_both_survive_a_meet(self) -> None:
        d = (TOP.with_clause("(x > 5)", Origin("m.py", 1))
                .with_clause("(5 < x)", Origin("m.py", 2)))
        self.assertEqual(d.requires(), "(x > 5)")

    def test_a_semantic_negation_is_detected_as_a_contradiction(self) -> None:
        # (x > 5) and its literal negation — canonicalization makes this semantic, so it
        # is caught even though the two are not textual `not (...)` twins.
        c = (TOP.with_clause("(x > 5)", Origin("m.py", 1))
                .with_clause("(not (x > 5))", Origin("m.py", 2)))
        self.assertEqual(len(c.clauses.contradictions()), 1)

    def test_ordering_pair_is_NOT_a_contradiction_under_nan(self) -> None:
        # (x > 5) and (x <= 5) are both false when x is NaN, so they do NOT contradict.
        # NaN-safe canonicalization is what stops this being wrongly flagged (and wrongly
        # pruned) — the safe direction.
        c = (TOP.with_clause("(x > 5)", Origin("m.py", 1))
                .with_clause("(x <= 5)", Origin("m.py", 2)))
        self.assertEqual(c.clauses.contradictions(), [])


class TestContradictionDetection(unittest.TestCase):
    """What a list of strings could never notice."""

    def test_a_guard_and_its_negation_are_reported(self) -> None:
        both = _c("(x > 5)").meet(_c("(not (x > 5))"))
        self.assertEqual(len(both.contradictions()), 1)

    def test_a_consistent_set_reports_nothing(self) -> None:
        self.assertEqual(_c("(x > 5)").meet(_c("(y < 3)")).contradictions(), [])


class TestRendering(unittest.TestCase):
    def test_only_clauses_the_consumer_can_bind_are_rendered(self) -> None:
        # An unbound name does not make a probe imprecise, it makes it ill-formed.
        s = TOP.with_clause("(x > 5)").with_clause("(y < 3)")
        self.assertEqual(s.requires(frozenset({"x"})), "(x > 5)")

    def test_a_dropped_clause_stays_and_applies_later(self) -> None:
        s = TOP.with_clause("(x > 5)").with_clause("(y < 3)")
        self.assertEqual(s.requires(frozenset({"x"})), "(x > 5)")
        self.assertIn("(y < 3)", s.requires(frozenset({"x", "y"})))

    def test_no_bindable_clause_renders_the_trivial_precondition(self) -> None:
        self.assertEqual(TOP.with_clause("(x > 5)").requires(frozenset({"z"})), "True")

    def test_rendering_is_deterministic(self) -> None:
        a = TOP.with_clause("(b)", Origin("m.py", 2)).with_clause("(a)", Origin("m.py", 1))
        b = TOP.with_clause("(a)", Origin("m.py", 1)).with_clause("(b)", Origin("m.py", 2))
        self.assertEqual(a.requires(), b.requires())


class TestFactStateBehavesLikeAPathCondition(unittest.TestCase):
    def test_a_variable_known_on_one_branch_only_is_unconstrained_after(self) -> None:
        # The other path reached here without establishing anything about it.
        merged = TOP.with_types("x", ["int"]).join(TOP.with_types("y", ["str"]))
        self.assertIsNone(merged.types_for("x"))
        self.assertIsNone(merged.types_for("y"))

    def test_refining_the_same_variable_twice_narrows(self) -> None:
        s = TOP.with_interval("n", lo=0).with_interval("n", hi=10)
        self.assertEqual((s.vars["n"].interval.lo, s.vars["n"].interval.hi), (0, 10))

    def test_a_type_set_survives_a_merge_of_two_seeds(self) -> None:
        # token was bytes under one seed and list under another: both are real.
        merged = TOP.with_types("token", ["bytes"]).join(TOP.with_types("token", ["list"]))
        self.assertEqual(merged.types_for("token"), frozenset({"bytes", "list"}))

    def test_going_back_to_a_prior_state_drops_what_was_learned_after_it(self) -> None:
        # This is the whole point: the state is a value, so abandoning a path is just
        # using the earlier value. A fact cannot outlive the path that produced it.
        before = TOP.with_clause("(x > 5)", Origin("m.py", 10, polarity="then"))
        after = before.with_clause("(y == 1)", Origin("m.py", 20, polarity="then"))
        self.assertIn("(y == 1)", after.requires())
        self.assertNotIn("(y == 1)", before.requires())

    def test_the_two_edges_of_a_branch_are_incompatible(self) -> None:
        then = TOP.with_clause("(x > 5)", Origin("m.py", 12, polarity="then"))
        els = TOP.with_clause("(not (x > 5))", Origin("m.py", 12, polarity="else"))
        self.assertEqual(then.join(els).requires(), "True")     # nothing common
        self.assertEqual(len(then.meet(els).clauses.contradictions()), 1)


class TestDropForgetsReassignedVariables(unittest.TestCase):
    """``drop`` is the flow-sensitivity primitive: a reassigned variable's facts go.

    Origin cannot distinguish a clause learned before a reassignment from one learned
    after — ``Clause`` identity is its canonical text — so staleness is enforced by
    matching on the names a clause mentions, which is what ``drop`` does.
    """

    def test_a_clause_naming_the_dropped_variable_is_gone(self) -> None:
        s = TOP.with_clause("(a > 100)", Origin("m.py", 2, polarity="then"))
        self.assertEqual(s.drop({"a"}).requires(), "True")

    def test_a_clause_not_naming_it_survives(self) -> None:
        s = (TOP.with_clause("(a > 100)", Origin("m.py", 2))
                .with_clause("(c == 3)", Origin("m.py", 4)))
        after = s.drop({"a"})
        self.assertNotIn("(a > 100)", after.requires())
        self.assertIn("(c == 3)", after.requires())

    def test_the_dropped_variables_varstate_is_forgotten_too(self) -> None:
        s = TOP.with_types("a", ["int"]).with_interval("a", lo=101)
        self.assertIsNone(s.drop({"a"}).types_for("a"))
        self.assertNotIn("a", s.drop({"a"}).vars)

    def test_over_kill_drops_a_clause_mentioning_a_live_and_a_killed_var(self) -> None:
        # A clause over both a killed and a surviving variable goes whole — precision is
        # lost (the surviving half is forgotten), soundness is not (nothing is invented).
        s = TOP.with_clause("(a + c > 5)", Origin("m.py", 2))
        self.assertEqual(s.drop({"a"}).requires(), "True")

    def test_dropping_an_absent_name_is_a_no_op(self) -> None:
        s = TOP.with_clause("(c == 3)", Origin("m.py", 4)).with_types("c", ["int"])
        self.assertEqual(s.drop({"a"}), s)

    def test_dropping_nothing_is_identity(self) -> None:
        s = TOP.with_clause("(a > 100)", Origin("m.py", 2))
        self.assertIs(s.drop(set()), s)

    def test_drop_matches_on_the_canonical_form(self) -> None:
        # canonicalize may reorder operands; the name match must see the canonical text,
        # which is what Clause.names parses. `100 < a` canonicalizes but still names `a`.
        s = TOP.with_clause("(100 < a)", Origin("m.py", 2))
        self.assertEqual(s.drop({"a"}).requires(), "True")

    def test_dropping_never_resurrects_an_unreachable_state(self) -> None:
        # A state made bottom by a real contradiction stays bottom — drop can only forget,
        # never make a dead path live.
        dead = TOP.with_types("a", ["int"]).with_types("a", ["str"])
        self.assertTrue(dead.is_bottom())
        self.assertTrue(dead.drop({"a"}).is_bottom())


if __name__ == "__main__":
    unittest.main()
