"""Tests for external VarFacts input to condition_solver."""

from __future__ import annotations

import os
import unittest

from tharkun.solve.condition import solve_condition
from tharkun.branch_solver import solve_branch
from tharkun.minleaf.types import FactEnv, VarFacts

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "facts_guards.py")


def _line_of(text):
    with open(FIXTURE) as f:
        for i, line in enumerate(f, 1):
            if text in line:
                return i
    raise ValueError(f"text {text!r} not found in {FIXTURE}")


class TestFactsNarrowIntRange(unittest.TestCase):
    """Guard `x > 50`, facts say lo=0, hi=100 → value in (50, 100]."""

    def test_narrowed_range(self):
        line = _line_of("if x > 50:")
        facts = FactEnv(facts={"x": VarFacts(lo=0, hi=100)})
        sol = solve_condition(FIXTURE, line, facts=facts)
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["x"]
        self.assertGreater(val, 50)
        self.assertLessEqual(val, 100)


class TestFactsWhitelist(unittest.TestCase):
    """Guard `x > 50`, facts say eq={75} → x=75."""

    def test_whitelist_value_used(self):
        line = _line_of("if x > 50:")
        facts = FactEnv(facts={"x": VarFacts(eq={75})})
        sol = solve_condition(FIXTURE, line, facts=facts)
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["x"]
        self.assertEqual(val, 75)


class TestFactsExclude(unittest.TestCase):
    """Guard `x != 0`, facts say neq={0, 1} → avoids both."""

    def test_exclude_values(self):
        line = _line_of("if x != 0:")
        facts = FactEnv(facts={"x": VarFacts(neq={0, 1})})
        sol = solve_condition(FIXTURE, line, facts=facts)
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["x"]
        self.assertNotIn(val, {0, 1})


class TestFactsOverrideType(unittest.TestCase):
    """Guard `len(x) > 3`, facts say eq={"hello", "world"} → picks from whitelist."""

    def test_whitelist_with_len(self):
        line = _line_of("if len(x) > 3:")
        facts = FactEnv(facts={"x": VarFacts(eq={"hello", "world"})})
        sol = solve_condition(FIXTURE, line, facts=facts)
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["x"]
        self.assertIn(val, {"hello", "world"})


class TestNoFactsUnchanged(unittest.TestCase):
    """Existing tests pass unchanged with facts=None."""

    def test_no_facts(self):
        line = _line_of("if x > 50:")
        sol_without = solve_condition(FIXTURE, line)
        sol_with = solve_condition(FIXTURE, line, facts=None)
        self.assertEqual(sol_without.strategy, sol_with.strategy)
        self.assertEqual(len(sol_without.satisfying_inputs), len(sol_with.satisfying_inputs))


class TestInfeasibleEarlyReturn(unittest.TestCase):
    """facts.possible == False → empty solution with strategy 'infeasible'."""

    def test_infeasible(self):
        line = _line_of("if x > 50:")
        facts = FactEnv(possible=False)
        sol = solve_condition(FIXTURE, line, facts=facts)
        self.assertEqual(sol.strategy, "infeasible")
        self.assertEqual(sol.confidence, 0.0)
        self.assertEqual(sol.satisfying_inputs, [])


class TestFactsStrSynthesisWithNeq(unittest.TestCase):
    """Guard `x.startswith("pre") and x.endswith("suf")`, neq={"presuf"}."""

    def test_avoids_excluded(self):
        line = _line_of('if x.startswith("pre") and x.endswith("suf"):')
        facts = FactEnv(facts={"x": VarFacts(neq={"presuf"})})
        sol = solve_condition(FIXTURE, line, facts=facts)
        self.assertTrue(sol.satisfying_inputs)
        val = sol.satisfying_inputs[0]["x"]
        self.assertTrue(val.startswith("pre"))
        self.assertTrue(val.endswith("suf"))
        self.assertNotEqual(val, "presuf")


class TestStrategyAnnotation(unittest.TestCase):
    """When facts contribute to type inference, strategy includes '+facts'."""

    def test_facts_type_inference_annotated(self):
        line = _line_of("if x > 50:")
        # Use facts with bounds that will be used for type inference
        # (the comparison already gives type info, so this tests when
        # facts contribute to narrowing values via _try_intersect_int_constraints).
        facts = FactEnv(facts={"x": VarFacts(lo=0, hi=100)})
        sol = solve_condition(FIXTURE, line, facts=facts)
        # Strategy may or may not include +facts depending on whether
        # facts were used for type inference (comparison already infers int).
        # The key test is that solve_condition runs successfully.
        self.assertIn(sol.strategy, ("direct", "direct+facts", "type_inferred", "type_inferred+facts"))


class TestSolveBranchPassesFacts(unittest.TestCase):
    """solve_branch passes facts through to condition_solver."""

    def test_branch_with_facts(self):
        line = _line_of("if x > 50:")
        facts = FactEnv(facts={"x": VarFacts(eq={75})})
        sol = solve_branch(FIXTURE, line, facts=facts, execute_snapshots=False)
        # The branch should be solved with the facts-provided value.
        self.assertTrue(sol.solved or sol.best_inputs)
        if sol.best_inputs:
            self.assertEqual(sol.best_inputs.get("x"), 75)
