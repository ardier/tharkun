"""The shared knowledge base, and what may and may not be remembered.

Prompted by a direct observation: static can only tell us something new at a new
frontier, or when the facts constraining it change. Measured on ntlm_auth's harness it
was invoked 1158 times across **39 distinct** questions — a 96.6% repeat rate, and 11.4
of its 11.8 seconds spent re-deciding what it had already decided.

That is not tuning. A decision procedure given the same guard and the same precondition
necessarily returns the same verdict. What matters is drawing the line correctly: the
techniques that read live program state are asked a genuinely different question every
round, and serving one of those from a store would hand back an answer about state that
has since moved on.
"""
from __future__ import annotations

import unittest

from tharkun.facts import (
    DETERMINISTIC,
    STATEFUL,
    STOCHASTIC,
    Constraint,
    DecisionKey,
    Fact,
    FactStore,
    is_memoizable,
)


class TestMemoizationBoundary(unittest.TestCase):
    def test_decision_procedures_are_memoizable(self) -> None:
        self.assertTrue(is_memoizable("static"))
        self.assertTrue(is_memoizable("static"))

    def test_live_state_techniques_are_not(self) -> None:
        # These read the program's state, which differs every round.
        for s in ("snapshot", "minleaf", "redqueen"):
            self.assertFalse(is_memoizable(s), s)

    def test_search_is_not_memoizable(self) -> None:
        # A repeat may succeed where the last attempt failed.
        for s in ("hypothesis", "hypothesis-facts", "reach-search"):
            self.assertFalse(is_memoizable(s), s)

    def test_qualified_names_resolve_to_their_family(self) -> None:
        self.assertTrue(is_memoizable("static:direct"))
        self.assertFalse(is_memoizable("minleaf:per_byte"))

    def test_unknown_strategy_defaults_to_not_memoizable(self) -> None:
        # A wrong "yes" serves a stale answer; a wrong "no" only costs time.
        self.assertFalse(is_memoizable("something-new"))


class TestDecisionIdentity(unittest.TestCase):
    def test_precondition_is_part_of_the_question(self) -> None:
        # The case that makes this matter: a guard satisfiable in isolation can be
        # infeasible once the callers' pinned values are folded in.
        a = DecisionKey.of("/m.py", 10, "then", "True")
        b = DecisionKey.of("/m.py", 10, "then", "pad != 0")
        self.assertNotEqual(a, b)

    def test_typing_is_part_of_the_question(self) -> None:
        a = DecisionKey.of("/m.py", 10, "then", "True", {"x": "int"})
        b = DecisionKey.of("/m.py", 10, "then", "True", {"x": "str"})
        self.assertNotEqual(a, b)

    def test_typing_order_does_not_matter(self) -> None:
        a = DecisionKey.of("/m.py", 10, "then", "True", {"x": "int", "y": "str"})
        b = DecisionKey.of("/m.py", 10, "then", "True", {"y": "str", "x": "int"})
        self.assertEqual(a, b)

    def test_polarity_is_part_of_the_question(self) -> None:
        self.assertNotEqual(DecisionKey.of("/m.py", 10, "then", "True"),
                            DecisionKey.of("/m.py", 10, "else", "True"))


class TestStoreServesAndRuns(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FactStore()
        self.key = DecisionKey.of("/m.py", 10, "then", "True")

    def test_second_ask_is_served_not_run(self) -> None:
        runs = []
        for _ in range(5):
            self.store.lookup_or_run(self.key, "static",
                                     lambda: runs.append(1) or "satisfiable")
        self.assertEqual(len(runs), 1, "a decision procedure was re-run")
        self.assertEqual(self.store.report()["static"],
                         {"served": 4, "ran": 1, "committed": 0})

    def test_stateful_strategy_is_always_re_run(self) -> None:
        runs = []
        for _ in range(5):
            self.store.lookup_or_run(self.key, "minleaf",
                                     lambda: runs.append(1) or "patch")
        self.assertEqual(len(runs), 5, "live-state technique was served a stale answer")

    def test_two_strategies_do_not_share_a_key_space(self) -> None:
        # They return different SHAPES of answer, so a shared key space hands one
        # strategy the other's verdict.
        self.store.lookup_or_run(self.key, "static:predicate", lambda: "STATIC-VERDICT")
        got = self.store.lookup_or_run(self.key, "static", lambda: "STATIC-SOLUTION")
        self.assertEqual(got, "STATIC-SOLUTION")

    def test_a_stronger_precondition_is_asked_again(self) -> None:
        runs = []
        self.store.lookup_or_run(self.key, "static", lambda: runs.append(1))
        stronger = DecisionKey.of("/m.py", 10, "then", "pad != 0")
        self.store.lookup_or_run(stronger, "static", lambda: runs.append(1))
        self.assertEqual(len(runs), 2)


class TestFacts(unittest.TestCase):
    def setUp(self) -> None:
        self.store = FactStore()

    def test_certified_facts_outrank_inferred_ones(self) -> None:
        self.store.add(Fact("type", "x", "int", source="guess", confidence=0.9))
        self.store.add(Fact("type", "x", "bytes", source="static", certified=True))
        self.assertEqual(self.store.best("type", "x").value, "bytes")

    def test_confidence_orders_uncertified_facts(self) -> None:
        self.store.add(Fact("type", "y", "str", confidence=0.3))
        self.store.add(Fact("type", "y", "int", confidence=0.8))
        self.assertEqual(self.store.best("type", "y").value, "int")

    def test_duplicate_facts_are_not_stored_twice(self) -> None:
        f = Fact("width", "key", 0, source="length_requirements")
        self.store.add(f)
        self.store.add(f)
        self.assertEqual(len(self.store.get("width", "key")), 1)

    def test_missing_subject_is_empty_not_an_error(self) -> None:
        self.assertEqual(self.store.get("type", "nope"), [])
        self.assertIsNone(self.store.best("type", "nope"))

    def test_subjects_lists_what_is_known_of_a_kind(self) -> None:
        self.store.add(Fact("width", "a", 0))
        self.store.add(Fact("width", "b", 7))
        self.store.add(Fact("type", "c", "int"))
        self.assertEqual(self.store.subjects("width"), ["a", "b"])


class TestEpochGatedExhaustion(unittest.TestCase):
    """Don't re-attack a frontier when nothing has been learned since it last resisted.

    Memoizing the individual queries is not enough: the ROUND still pays for the stateful
    and search tiers. Measured on a toy whose one unsolvable guard keeps a frontier alive,
    five such rounds cost 26.7s and produced no coverage at all; skipping them took the
    same run to 7.7s with an identical line set.

    Soundness rests entirely on the gate being epoch-based rather than permanent.
    """

    def setUp(self) -> None:
        self.store = FactStore()
        self.scope = ("/m.py", 42, "then", "f")

    def test_a_scope_that_resisted_is_skipped(self) -> None:
        self.store.mark_exhausted(self.scope)
        self.assertTrue(self.store.is_exhausted(self.scope))

    def test_an_untried_scope_is_never_skipped(self) -> None:
        self.assertFalse(self.store.is_exhausted(("/other.py", 1, "then", "g")))

    def test_a_new_fact_re_opens_everything(self) -> None:
        self.store.mark_exhausted(self.scope)
        self.store.add(Fact("width", "key", 0))
        self.assertFalse(self.store.is_exhausted(self.scope),
                         "a learned fact must make a written-off frontier live again")

    def test_a_new_decision_re_opens_everything(self) -> None:
        self.store.mark_exhausted(self.scope)
        self.store.lookup_or_run(DecisionKey.of("/m.py", 9, "then", "True"),
                                 "static", lambda: "satisfiable")
        self.assertFalse(self.store.is_exhausted(self.scope))

    def test_an_explicit_bump_re_opens_everything(self) -> None:
        # New coverage is not a Fact, but it is new information: a frontier can fall from
        # a base seed that did not exist before.
        self.store.mark_exhausted(self.scope)
        self.store.bump("new seed")
        self.assertFalse(self.store.is_exhausted(self.scope))

    def test_a_served_repeat_does_not_re_open(self) -> None:
        # Being handed a remembered verdict teaches nothing, so it must NOT count.
        key = DecisionKey.of("/m.py", 9, "then", "True")
        self.store.lookup_or_run(key, "static", lambda: "satisfiable")
        self.store.mark_exhausted(self.scope)
        self.store.lookup_or_run(key, "static", lambda: "satisfiable")
        self.assertTrue(self.store.is_exhausted(self.scope))

    def test_re_recording_the_same_decision_does_not_re_open(self) -> None:
        key = DecisionKey.of("/m.py", 9, "then", "True")
        self.store.record_decision(key, "satisfiable", source="static")
        self.store.mark_exhausted(self.scope)
        self.store.record_decision(key, "satisfiable", source="static")
        self.assertTrue(self.store.is_exhausted(self.scope))

    def test_a_duplicate_fact_does_not_re_open(self) -> None:
        f = Fact("width", "key", 0)
        self.store.add(f)
        self.store.mark_exhausted(self.scope)
        self.store.add(f)
        self.assertTrue(self.store.is_exhausted(self.scope))

    def test_scopes_are_independent(self) -> None:
        other = ("/m.py", 99, "else", "f")
        self.store.mark_exhausted(self.scope)
        self.assertFalse(self.store.is_exhausted(other))


class TestYieldAccounting(unittest.TestCase):
    """What a power schedule allocates on: committed answers, not invocations."""

    def test_credit_is_recorded_per_strategy_family(self) -> None:
        store = FactStore()
        store.credit("minleaf:per_byte")
        store.credit("static:direct@L38")
        store.credit("minleaf:atomic")
        self.assertEqual(store.yields["minleaf"], 2)
        self.assertEqual(store.yields["static"], 1)

    def test_report_shows_invocations_against_payoff(self) -> None:
        # The asymmetry the measurement found: 1158 invocations, 0 committed answers.
        store = FactStore()
        key = DecisionKey.of("/m.py", 1, "then", "True")
        for _ in range(10):
            store.lookup_or_run(key, "static", lambda: None)
        rep = store.report()["static"]
        self.assertEqual(rep["committed"], 0)
        self.assertEqual(rep["ran"] + rep["served"], 10)


if __name__ == "__main__":
    unittest.main()


class TestConstraintCarriesTheExpression(unittest.TestCase):
    """A constraint stores what must HOLD, not the value that satisfies it.

    Storing the value loses the knowledge. A search that observes
    `int.from_bytes(token[:2], "big") == 13987` knows far more than the number: it knows
    which bytes, in which order, produce it. Measured — a search aimed with the bare
    number still failed, because it had to rediscover the encoding by chance, while the
    consumer that could invert the expression never received it.
    """

    GUARD = "int.from_bytes(token[:2], 'big') == 13987"

    def test_text_is_the_stored_form(self) -> None:
        c = Constraint("token", self.GUARD)
        self.assertEqual(c.text, self.GUARD)
        self.assertEqual(str(c), self.GUARD)

    def test_node_is_available_for_structural_consumers(self) -> None:
        import ast

        c = Constraint("token", self.GUARD)
        self.assertIsInstance(c.node, ast.Compare)

    def test_node_is_derived_not_stored_so_it_cannot_drift(self) -> None:
        c = Constraint("token", self.GUARD)
        self.assertEqual(ast_unparse(c.node), ast_unparse(Constraint("token", c.text).node))

    def test_builtins_are_not_bindable_names(self) -> None:
        # `int` is mentioned but no probe binds it; counting it made the clause look
        # unbindable and silently dropped it from every projection.
        self.assertEqual(Constraint("token", self.GUARD).names, {"token"})

    def test_unparseable_text_degrades_quietly(self) -> None:
        c = Constraint("x", "this is not ( python")
        self.assertIsNone(c.node)
        self.assertEqual(c.names, set())

    def test_can_be_built_from_a_node(self) -> None:
        import ast

        node = ast.parse("len(key) != 7", mode="eval").body
        c = Constraint.of_node("key", node, source="length_requirements")
        self.assertEqual(c.text, "len(key) != 7")
        self.assertEqual(c.subject, "key")


def ast_unparse(node):
    import ast

    return ast.unparse(node)


class TestProjections(unittest.TestCase):
    """One store, several consumer languages. Each projection is a total function of it."""

    def setUp(self) -> None:
        self.store = FactStore()
        self.store.require(Constraint("token", "int.from_bytes(token[:2], 'big') == 13987",
                                      source="reach-search"))
        self.store.require(Constraint("key", "len(key) != 7",
                                      source="length_requirements", certified=True))

    def test_requires_string_for_the_smt_lane(self) -> None:
        self.assertEqual(self.store.requires_str({"key"}), "len(key) != 7")

    def test_projection_conjoins_everything_bindable(self) -> None:
        got = self.store.requires_str({"token", "key"})
        self.assertIn("len(key) != 7", got)
        self.assertIn("13987", got)
        self.assertIn(" and ", got)

    def test_a_clause_naming_an_unbindable_variable_is_dropped(self) -> None:
        # Not an optimisation: for a probe such a clause is ill-formed.
        self.assertEqual(self.store.requires_str({"key"}).count("token"), 0)

    def test_a_dropped_clause_is_kept_and_applies_later(self) -> None:
        self.assertEqual(self.store.requires_str({"key"}), "len(key) != 7")
        self.assertIn("13987", self.store.requires_str({"key", "token"}))

    def test_no_bindable_clause_yields_the_trivial_precondition(self) -> None:
        self.assertEqual(self.store.requires_str({"unrelated"}), "True")

    def test_a_caller_supplied_base_is_conjoined_first(self) -> None:
        got = self.store.requires_str({"key"}, base="0 <= key")
        self.assertTrue(got.startswith("0 <= key and "))

    def test_certified_constraints_come_first(self) -> None:
        order = [c.source for c in self.store.constraints({"token", "key"})]
        self.assertEqual(order[0], "length_requirements")

    def test_redqueen_projection_returns_expressions_for_one_input(self) -> None:
        # The consumer a bare value cannot serve: it needs the SHAPE to invert.
        goals = self.store.goals_for("token")
        self.assertEqual(len(goals), 1)
        self.assertIn("from_bytes", goals[0].text)
        self.assertEqual(self.store.goals_for("key")[0].text, "len(key) != 7")

    def test_duplicates_are_ignored_and_do_not_move_the_epoch(self) -> None:
        c = Constraint("key", "len(key) != 7", source="length_requirements", certified=True)
        before = self.store.epoch
        self.store.require(c)
        self.assertEqual(self.store.epoch, before)

    def test_a_new_constraint_re_opens_exhausted_scopes(self) -> None:
        scope = ("/m.py", 1, "then", "f")
        self.store.mark_exhausted(scope)
        self.store.require(Constraint("z", "z > 0", source="static"))
        self.assertFalse(self.store.is_exhausted(scope))


class TestTypesAreASetNotAWinner(unittest.TestCase):
    """Python is dynamically typed, so one parameter can genuinely be several things.

    The signals that disagree are usually each right about something. Static inference
    reading `token[:2]` says list; the run that happened says bytes. `list` is not wrong,
    it is INCOMPLETE — the guard admits bytes, list, str and tuple alike — and the next
    run may take a different path with a different type. Collapsing to a single winner
    points a generator at one shape and can rule out the answer a priori.

    The rest of the codebase already knows this: type_hints fuses signals into a
    TypeDistribution and candidate_typings enumerates joint typings so static can
    solve each one.
    """

    def setUp(self) -> None:
        self.store = FactStore()
        self.store.add(Fact("type", "token", "list",
                            source="static:subscript", confidence=0.7))
        self.store.add(Fact("type", "token", "bytes",
                            source="observed", certified=True))
        self.store.add(Fact("type", "token", "str",
                            source="static:method_call", confidence=0.5))

    def test_every_plausible_type_is_offered(self) -> None:
        self.assertEqual(set(self.store.types_for("token")), {"bytes", "list", "str"})

    def test_observed_comes_first(self) -> None:
        # A consumer that can only take one must still take the best one.
        self.assertEqual(self.store.types_for("token")[0], "bytes")

    def test_a_wrong_static_guess_is_kept_not_discarded(self) -> None:
        # `list` is what the subscript rule inferred. It is incomplete, not false, and a
        # later path may well pass a list.
        self.assertIn("list", self.store.types_for("token"))

    def test_duplicates_from_several_sources_collapse(self) -> None:
        self.store.add(Fact("type", "token", "bytes",
                            source="static:comparison", confidence=0.6))
        self.assertEqual(self.store.types_for("token").count("bytes"), 1)

    def test_the_set_is_capped(self) -> None:
        for i, t in enumerate(["int", "float", "dict", "tuple", "set"]):
            self.store.add(Fact("type", "token", t, confidence=0.1 * i))
        self.assertLessEqual(len(self.store.types_for("token", max_types=4)), 4)

    def test_unknown_subject_is_empty(self) -> None:
        self.assertEqual(self.store.types_for("nope"), [])


class TestStrategyOverTypes(unittest.TestCase):
    """A strategy spanning the union, not a bet on the mode."""

    def test_union_of_types_yields_a_strategy(self) -> None:
        try:
            import hypothesis  # noqa: F401
        except ImportError:
            self.skipTest("hypothesis not installed")
        from tharkun.driver import _strategy_over_types

        self.assertIsNotNone(_strategy_over_types(["bytes", "list"]))

    def test_a_single_type_needs_no_union(self) -> None:
        try:
            import hypothesis  # noqa: F401
        except ImportError:
            self.skipTest("hypothesis not installed")
        from tharkun.driver import _strategy_over_types

        self.assertIsNotNone(_strategy_over_types(["int"]))

    def test_unmodelable_types_yield_nothing_rather_than_a_wrong_guess(self) -> None:
        from tharkun.driver import _strategy_over_types

        self.assertIsNone(_strategy_over_types(["SomeUserClass", "Socket"]))
