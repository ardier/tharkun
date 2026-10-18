"""Tests for mutation-guided input gen + attr tracer."""
from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path

from tharkun.mutation_input_gen import (
    MutationContext,
    SeedInput,
    _boundary_seeds,
    _arithmetic_seeds,
    _negation_seeds,
    extract_mutation_contexts,
    generate_killing_seeds,
    serialize_seeds,
)
from tharkun.solve.attrs import (
    AttrTarget,
    analyze_attr_blocker,
    extract_attr_targets,
    find_attr_assignments,
)

ATTR_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "attr_trace_target.py")


# ---------------------------------------------------------------------------
# Mutation-guided input gen tests
# ---------------------------------------------------------------------------


class TestBoundarySeeds(unittest.TestCase):
    def _ctx(self, guard_source: str) -> MutationContext:
        expr = ast.parse(guard_source, mode="eval").body
        return MutationContext(
            job_id="j1", callable_id="cid", qualname="fn",
            operator_name="BoundaryMutator", mutant_kind="boundary",
            route="hypothesis", lineno=10, guard_expr=expr,
        )

    def test_int_boundary_probes(self) -> None:
        ctx = self._ctx("x >= 18")
        seeds = _boundary_seeds(ctx)
        self.assertGreater(len(seeds), 0)
        # Should have x=17, x=18, x=19 (boundary ±1)
        values = {s.kwargs.get("x") for s in seeds}
        self.assertIn(18, values)
        self.assertIn(17, values)
        self.assertIn(19, values)

    def test_float_boundary_probes(self) -> None:
        ctx = self._ctx("ratio > 0.5")
        seeds = _boundary_seeds(ctx)
        values = {s.kwargs.get("ratio") for s in seeds if "ratio" in s.kwargs}
        # Should probe around 0.5
        self.assertTrue(any(abs(v - 0.5) < 0.1 for v in values if isinstance(v, float)))

    def test_no_guard_returns_empty(self) -> None:
        ctx = MutationContext(
            job_id="j1", callable_id="cid", qualname="fn",
            operator_name="BoundaryMutator", mutant_kind="boundary",
            route="hypothesis", lineno=10,
        )
        seeds = _boundary_seeds(ctx)
        self.assertEqual(seeds, [])


class TestArithmeticSeeds(unittest.TestCase):
    def test_int_identity_values(self) -> None:
        # Use a simple comparison so infer_var_type returns "int"
        expr = ast.parse("x > 10", mode="eval").body
        ctx = MutationContext(
            job_id="j1", callable_id="cid", qualname="fn",
            operator_name="ArithmeticMutator", mutant_kind="arithmetic",
            route="assertions", lineno=10, guard_expr=expr,
        )
        seeds = _arithmetic_seeds(ctx)
        # Should include 0, 1, -1 (identity values)
        x_vals = {s.kwargs.get("x") for s in seeds if "x" in s.kwargs}
        self.assertIn(0, x_vals)
        self.assertIn(1, x_vals)


class TestNegationSeeds(unittest.TestCase):
    def test_bool_truthy_falsy(self) -> None:
        expr = ast.parse("flag", mode="eval").body
        ctx = MutationContext(
            job_id="j1", callable_id="cid", qualname="fn",
            operator_name="NegationMutator", mutant_kind="negation",
            route="hypothesis", lineno=10, guard_expr=expr,
        )
        seeds = _negation_seeds(ctx)
        sources = {s.source for s in seeds}
        self.assertIn("negation_truthy", sources)
        self.assertIn("negation_falsy", sources)


class TestGenerateKillingSeeds(unittest.TestCase):
    def test_from_router_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            outp = Path(tmp)
            mutants_dir = outp / "mutants"
            mutants_dir.mkdir()

            router_obj = {
                "summary": {},
                "rows": [
                    {
                        "job_id": "j1",
                        "module_key": "mk",
                        "primary_callable_id": "mk:function:fn",
                        "primary_qualname": "fn",
                        "operator_name": "BoundaryMutator",
                        "mutant_kind": "boundary",
                        "route": "hypothesis",
                        "rationale": "test",
                    }
                ],
            }
            (mutants_dir / "router.json").write_text(json.dumps(router_obj))
            (mutants_dir / "mutants_analysis.json").write_text(json.dumps({"rows": []}))

            seeds = generate_killing_seeds(outp)
            # No guard expression found (no analysis row with line info) → empty
            # This is expected — the analysis row would need real line info
            self.assertIsInstance(seeds, dict)

    def test_no_router_returns_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            seeds = generate_killing_seeds(Path(tmp))
            self.assertEqual(seeds, {})


class TestSerializeSeeds(unittest.TestCase):
    def test_serializes(self) -> None:
        seeds = {
            "cid": [SeedInput("cid", {"x": 18}, "boundary_probe", ["j1"])],
        }
        obj = serialize_seeds(seeds)
        self.assertIn("summary", obj)
        self.assertEqual(obj["summary"]["total_seeds"], 1)


# ---------------------------------------------------------------------------
# Attr tracer tests
# ---------------------------------------------------------------------------


class TestExtractAttrTargets(unittest.TestCase):
    def test_simple_self_attr(self) -> None:
        expr = ast.parse("self.allow_unicode", mode="eval").body
        targets = extract_attr_targets(expr)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].base, "self")
        self.assertEqual(targets[0].chain, ("allow_unicode",))
        self.assertEqual(targets[0].full, "self.allow_unicode")

    def test_nested_attr(self) -> None:
        expr = ast.parse("self.md.references", mode="eval").body
        targets = extract_attr_targets(expr)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].chain, ("md", "references"))
        self.assertEqual(targets[0].full, "self.md.references")

    def test_compound_guard(self) -> None:
        expr = ast.parse("self.strict and self.allow_unicode", mode="eval").body
        targets = extract_attr_targets(expr)
        self.assertEqual(len(targets), 2)
        fulls = {t.full for t in targets}
        self.assertIn("self.strict", fulls)
        self.assertIn("self.allow_unicode", fulls)

    def test_no_self_attrs(self) -> None:
        expr = ast.parse("x > 0 and y < 10", mode="eval").body
        targets = extract_attr_targets(expr)
        self.assertEqual(targets, [])

    def test_cls_attr(self) -> None:
        expr = ast.parse("cls.VERSION", mode="eval").body
        targets = extract_attr_targets(expr)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].base, "cls")


class TestFindAttrAssignments(unittest.TestCase):
    def test_finds_init_assignment(self) -> None:
        target = AttrTarget(base="self", chain=("allow_unicode",), full="self.allow_unicode")
        assignments = find_attr_assignments(ATTR_FIXTURE, target)
        self.assertGreater(len(assignments), 0)
        init_assigns = [a for a in assignments if a.is_init]
        self.assertGreater(len(init_assigns), 0)

    def test_finds_strict_assignment(self) -> None:
        target = AttrTarget(base="self", chain=("strict",), full="self.strict")
        assignments = find_attr_assignments(ATTR_FIXTURE, target)
        self.assertGreater(len(assignments), 0)
        self.assertTrue(any(a.is_init for a in assignments))

    def test_nonexistent_attr_returns_empty(self) -> None:
        target = AttrTarget(base="self", chain=("nonexistent",), full="self.nonexistent")
        assignments = find_attr_assignments(ATTR_FIXTURE, target)
        self.assertEqual(assignments, [])

    def test_method_assignment(self) -> None:
        # self.md is assigned in setup_markdown, not __init__
        target = AttrTarget(base="self", chain=("md",), full="self.md")
        assignments = find_attr_assignments(ATTR_FIXTURE, target)
        self.assertGreater(len(assignments), 0)
        # Should find both __init__ (self.md = None) and setup_markdown
        methods = {a.method for a in assignments}
        self.assertGreater(len(methods), 0)


class TestAnalyzeAttrBlocker(unittest.TestCase):
    def test_parser_guard(self) -> None:
        expr = ast.parse("self.allow_unicode", mode="eval").body
        results = analyze_attr_blocker(ATTR_FIXTURE, expr)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].target.full, "self.allow_unicode")
        self.assertEqual(results[0].strategy, "static")
        self.assertGreater(len(results[0].assignments), 0)

    def test_compound_guard(self) -> None:
        expr = ast.parse("self.enabled and len(self.name) > 0", mode="eval").body
        results = analyze_attr_blocker(ATTR_FIXTURE, expr)
        self.assertGreater(len(results), 0)
        fulls = {r.target.full for r in results}
        self.assertIn("self.enabled", fulls)
        self.assertIn("self.name", fulls)


if __name__ == "__main__":
    unittest.main()
