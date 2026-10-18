"""Tests for the condition solver module (IB-3)."""
from __future__ import annotations

import ast
import os
import textwrap
import unittest

from tharkun.solve.condition import (
    ConditionSolution,
    VarConstraint,
    infer_var_type,
    serialize_solution,
    solve_condition,
)
from tharkun.solve.types import CallableInfo, ParamInfo, TypeHint

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "condition_guards.py")


class TestInferVarType(unittest.TestCase):
    """Test individual type inference rules against the fixture AST."""

    def _parse_guard(self, lineno: int):
        import ast
        source = open(FIXTURE, encoding="utf-8").read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if getattr(node, "lineno", None) == lineno:
                if isinstance(node, (ast.If, ast.While)):
                    return node.test
        return None

    def test_rule1_comparison_int(self) -> None:
        expr = self._parse_guard(5)  # x > 100
        self.assertIsNotNone(expr)
        t, src = infer_var_type("x", expr, FIXTURE, 5)
        self.assertEqual(t, "int")
        self.assertEqual(src, "comparison_literal")

    def test_rule1_comparison_str(self) -> None:
        expr = self._parse_guard(7)  # name == "hello"
        t, src = infer_var_type("name", expr, FIXTURE, 7)
        self.assertEqual(t, "str")
        self.assertEqual(src, "comparison_literal")

    def test_rule2_isinstance(self) -> None:
        expr = self._parse_guard(11)  # isinstance(items, list) and len(items) > 3
        t, src = infer_var_type("items", expr, FIXTURE, 11)
        self.assertEqual(t, "list")
        self.assertEqual(src, "isinstance")

    def test_rule3_str_method(self) -> None:
        expr = self._parse_guard(43)  # s.startswith("prefix")
        t, src = infer_var_type("s", expr, FIXTURE, 43)
        self.assertEqual(t, "str")
        self.assertEqual(src, "method_call")

    def test_rule3_bytes_method(self) -> None:
        expr = self._parse_guard(48)  # data.startswith(b"\xff\xfe")
        t, src = infer_var_type("data", expr, FIXTURE, 48)
        self.assertEqual(t, "bytes")
        self.assertEqual(src, "method_call")

    def test_rule3_dict_method(self) -> None:
        expr = self._parse_guard(53)  # d.get("key")
        t, src = infer_var_type("d", expr, FIXTURE, 53)
        self.assertEqual(t, "dict")
        self.assertEqual(src, "method_call")

    def test_rule4_subscript(self) -> None:
        expr = self._parse_guard(13)  # lines[1].startswith("=")
        t, src = infer_var_type("lines", expr, FIXTURE, 13)
        self.assertEqual(t, "list")
        self.assertEqual(src, "subscript")

    def test_rule5_param_hint(self) -> None:
        import ast
        source = open(FIXTURE, encoding="utf-8").read()
        tree = ast.parse(source)
        expr = None
        for node in ast.walk(tree):
            if getattr(node, "lineno", None) == 38:
                if isinstance(node, ast.If):
                    expr = node.test
        self.assertIsNotNone(expr)
        ci = CallableInfo(
            module_path=FIXTURE,
            module_key="mk",
            callable_id="mk:function:typed_params",
            qualname="typed_params",
            name="typed_params",
            kind="function",
            params=[
                ParamInfo(name="x", kind="arg", type_hint=TypeHint("int", "explicit", 100)),
                ParamInfo(name="name", kind="arg", type_hint=TypeHint("str", "explicit", 100)),
            ],
        )
        # x > 50 — comparison_literal takes precedence over param_hint
        t, src = infer_var_type("x", expr, FIXTURE, 38, ci)
        self.assertEqual(t, "int")
        self.assertIn(src, ("comparison_literal", "param_hint"))

    def test_rule6_self(self) -> None:
        expr = self._parse_guard(26)  # self.allow_unicode
        t, src = infer_var_type("self", expr, FIXTURE, 26)
        self.assertEqual(t, "object")
        self.assertEqual(src, "self_attr")

    def test_rule7_fallback(self) -> None:
        import ast
        # A synthetic expression with no type signals
        expr = ast.parse("unknown_var", mode="eval").body
        t, src = infer_var_type("unknown_var", expr, FIXTURE, 0)
        self.assertEqual(t, "Any")
        self.assertEqual(src, "fallback")


class TestSolveCondition(unittest.TestCase):
    def test_gt_int(self) -> None:
        sol = solve_condition(FIXTURE, 5)  # x > 100
        self.assertIn("x > 100", sol.guard_source)
        self.assertEqual(sol.strategy, "direct")
        self.assertGreater(sol.confidence, 0.0)
        # Should have x with value > 100
        x_var = next((v for v in sol.variables if v.name == "x"), None)
        self.assertIsNotNone(x_var)
        self.assertEqual(x_var.inferred_type, "int")
        self.assertTrue(any(v > 100 for v in x_var.concrete_values))

    def test_eq_str(self) -> None:
        sol = solve_condition(FIXTURE, 7)  # name == "hello"
        self.assertIn("hello", sol.guard_source)
        name_var = next((v for v in sol.variables if v.name == "name"), None)
        self.assertIsNotNone(name_var)
        self.assertEqual(name_var.inferred_type, "str")
        self.assertIn("hello", name_var.concrete_values)

    def test_isinstance_compound(self) -> None:
        sol = solve_condition(FIXTURE, 11)  # isinstance(items, list) and len(items) > 3
        items_var = next((v for v in sol.variables if v.name == "items"), None)
        self.assertIsNotNone(items_var)
        self.assertEqual(items_var.inferred_type, "list")

    def test_compound_int_constraints(self) -> None:
        sol = solve_condition(FIXTURE, 15)  # x > 0 and x < 10 and x != 5
        x_var = next((v for v in sol.variables if v.name == "x"), None)
        self.assertIsNotNone(x_var)
        self.assertEqual(x_var.inferred_type, "int")
        # All values should satisfy x > 0, x < 10, x != 5
        for v in x_var.concrete_values:
            self.assertGreater(v, 0)
            self.assertLess(v, 10)
            self.assertNotEqual(v, 5)

    def test_in_tuple(self) -> None:
        sol = solve_condition(FIXTURE, 33)  # x in (1, 2, 3)
        x_var = next((v for v in sol.variables if v.name == "x"), None)
        self.assertIsNotNone(x_var)
        self.assertTrue(set(x_var.concrete_values).issubset({1, 2, 3}))

    def test_self_attr_solved_via_truthiness(self) -> None:
        sol = solve_condition(FIXTURE, 26)  # self.allow_unicode
        self_var = next((v for v in sol.variables if v.name == "self"), None)
        self.assertIsNotNone(self_var)
        self.assertEqual(self_var.inference_source, "attr_truthiness")
        self.assertGreater(len(self_var.concrete_values), 0)
        # Should produce SimpleNamespace(allow_unicode=True) for truthiness
        self.assertTrue(self_var.concrete_values[0].allow_unicode)

    def test_nonexistent_file(self) -> None:
        sol = solve_condition("/nonexistent/path.py", 1)
        self.assertEqual(sol.strategy, "none")
        self.assertEqual(sol.confidence, 0.0)

    def test_line_without_branch(self) -> None:
        sol = solve_condition(FIXTURE, 6)  # pass
        self.assertEqual(sol.strategy, "none")

    def test_bytes_subscript(self) -> None:
        sol = solve_condition(FIXTURE, 9)  # data[:2] == b"\xfe\xff"
        data_var = next((v for v in sol.variables if v.name == "data"), None)
        self.assertIsNotNone(data_var)
        self.assertEqual(data_var.inferred_type, "bytes")


class TestSerializeSolution(unittest.TestCase):
    def test_roundtrip(self) -> None:
        sol = solve_condition(FIXTURE, 5)
        d = serialize_solution(sol)
        self.assertIn("guard_source", d)
        self.assertIn("variables", d)
        self.assertIn("satisfying_inputs", d)
        self.assertIsInstance(d["variables"], list)
        self.assertGreater(len(d["variables"]), 0)


# ---------------------------------------------------------------------------
# Stress tests for real-world IB-3 gaps
# ---------------------------------------------------------------------------


class TestFlippedInequalities(unittest.TestCase):
    """Bug fix: flipped comparisons like 0.0 < ratio should produce values."""

    def test_flipped_float_produces_value(self) -> None:
        sol = solve_condition(FIXTURE, 61)  # 0.0 < ratio <= 1.0
        ratio_var = next((v for v in sol.variables if v.name == "ratio"), None)
        self.assertIsNotNone(ratio_var)
        self.assertEqual(ratio_var.inferred_type, "float")
        # Must have at least one concrete value
        self.assertGreater(len(ratio_var.concrete_values), 0)
        # The value must satisfy 0.0 < ratio
        for v in ratio_var.concrete_values:
            self.assertGreater(v, 0.0)

    def test_flipped_int_produces_value(self) -> None:
        sol = solve_condition(FIXTURE, 66)  # 10 > x
        x_var = next((v for v in sol.variables if v.name == "x"), None)
        self.assertIsNotNone(x_var)
        # Should have a value < 10
        self.assertGreater(len(x_var.concrete_values), 0)
        for v in x_var.concrete_values:
            self.assertLess(v, 10)


class TestLenConstraint(unittest.TestCase):
    """Bug fix: len(x) >= N should produce a sized value."""

    def test_len_password_ge_8(self) -> None:
        sol = solve_condition(FIXTURE, 71)  # len(password) >= 8
        pw_var = next((v for v in sol.variables if v.name == "password"), None)
        self.assertIsNotNone(pw_var)
        # Should have a value with len >= 8
        self.assertGreater(len(pw_var.concrete_values), 0)
        for v in pw_var.concrete_values:
            self.assertGreaterEqual(len(v), 8)

    def test_len_equals_zero_is_the_empty_value(self) -> None:
        # `len(x) == 0` is satisfied by the EMPTY value, a real answer — not "no
        # constraint". Surfaced running tharkun on idna's check_label / a TOML parser.
        import os
        import tempfile

        p = os.path.join(tempfile.mkdtemp(), "m.py")
        with open(p, "w") as f:
            f.write("def f(label):\n    if len(label) == 0:\n        return 1\n    return 0\n")
        sol = solve_condition(p, 2, desired_branch="then")
        self.assertEqual(sol.strategy, "direct")
        self.assertEqual(len(sol.satisfying_inputs[0]["label"]), 0)


class TestStartswithEndswith(unittest.TestCase):
    """Bug fix: startswith + endswith should synthesise a satisfying string."""

    def test_prefix_and_suffix(self) -> None:
        sol = solve_condition(FIXTURE, 76)  # x.startswith("prefix") and x.endswith("suffix")
        x_var = next((v for v in sol.variables if v.name == "x"), None)
        self.assertIsNotNone(x_var)
        self.assertEqual(x_var.inferred_type, "str")
        self.assertGreater(len(x_var.concrete_values), 0)
        val = x_var.concrete_values[0]
        self.assertTrue(val.startswith("prefix"), f"{val!r} should start with 'prefix'")
        self.assertTrue(val.endswith("suffix"), f"{val!r} should end with 'suffix'")

    def test_startswith_only(self) -> None:
        sol = solve_condition(FIXTURE, 81)  # s.startswith("http://")
        s_var = next((v for v in sol.variables if v.name == "s"), None)
        self.assertIsNotNone(s_var)
        self.assertGreater(len(s_var.concrete_values), 0)
        self.assertTrue(s_var.concrete_values[0].startswith("http://"))


class TestSubscriptTypeConflict(unittest.TestCase):
    """Bug fix: data[0] == 'header' should infer data as list, not str."""

    def test_data_subscript_is_list(self) -> None:
        sol = solve_condition(FIXTURE, 86)  # data[0] == "header" and data[-1] == "footer"
        data_var = next((v for v in sol.variables if v.name == "data"), None)
        self.assertIsNotNone(data_var)
        self.assertEqual(data_var.inferred_type, "list")


class TestMultiVarCompound(unittest.TestCase):
    """Multi-variable guard: username and len(password) >= 8 and role in (...)."""

    def test_role_solved(self) -> None:
        sol = solve_condition(FIXTURE, 96)
        role_var = next((v for v in sol.variables if v.name == "role"), None)
        self.assertIsNotNone(role_var)
        self.assertIn(role_var.concrete_values[0], ("admin", "editor"))

    def test_password_has_length(self) -> None:
        sol = solve_condition(FIXTURE, 96)
        pw_var = next((v for v in sol.variables if v.name == "password"), None)
        self.assertIsNotNone(pw_var)
        # len constraint should be propagated
        if pw_var.concrete_values:
            self.assertGreaterEqual(len(pw_var.concrete_values[0]), 8)


class TestNegatedIsinstance(unittest.TestCase):
    """Fix #1: not isinstance(x, str) should produce a non-str value."""

    def test_negated_isinstance_produces_non_str(self) -> None:
        sol = solve_condition(FIXTURE, 101)  # not isinstance(x, str)
        x_var = next((v for v in sol.variables if v.name == "x"), None)
        self.assertIsNotNone(x_var)
        self.assertEqual(x_var.inference_source, "negated_isinstance")
        self.assertGreater(len(x_var.concrete_values), 0)
        # Value must NOT be a string
        self.assertNotIsInstance(x_var.concrete_values[0], str)


class TestIsNoneIdentity(unittest.TestCase):
    """Fix #3: is None should produce None, is not None should produce non-None."""

    def test_is_none_produces_none(self) -> None:
        sol = solve_condition(FIXTURE, 106)  # x is None
        x_var = next((v for v in sol.variables if v.name == "x"), None)
        self.assertIsNotNone(x_var)
        self.assertIn(None, x_var.concrete_values)

    def test_is_not_none_produces_non_none(self) -> None:
        sol = solve_condition(FIXTURE, 111)  # x is not None
        x_var = next((v for v in sol.variables if v.name == "x"), None)
        self.assertIsNotNone(x_var)
        self.assertGreater(len(x_var.concrete_values), 0)
        self.assertIsNotNone(x_var.concrete_values[0])


class TestMethodChain(unittest.TestCase):
    """Fix #4: response.json().get(...) should detect response as object, not fallback."""

    def test_method_chain_infers_object(self) -> None:
        sol = solve_condition(FIXTURE, 116)  # response.json().get("status") == "ok"
        resp_var = next((v for v in sol.variables if v.name == "response"), None)
        self.assertIsNotNone(resp_var)
        self.assertEqual(resp_var.inferred_type, "object")
        self.assertEqual(resp_var.inference_source, "method_chain")


class TestOrBranches(unittest.TestCase):
    """Fix #10: or-branches should produce one satisfying_inputs per branch."""

    def test_or_produces_multiple_inputs(self) -> None:
        sol = solve_condition(FIXTURE, 121)  # x < 0 or isinstance(y, str) or z is None
        # Should have multiple satisfying_inputs (one per or branch)
        self.assertGreater(len(sol.satisfying_inputs), 1,
                           f"Expected >1 satisfying inputs, got {len(sol.satisfying_inputs)}")

    def test_or_first_branch_has_negative_x(self) -> None:
        sol = solve_condition(FIXTURE, 121)
        # First branch: x < 0
        first = sol.satisfying_inputs[0]
        self.assertIn("x", first)
        self.assertLess(first["x"], 0)

    def test_or_last_branch_has_none_z(self) -> None:
        sol = solve_condition(FIXTURE, 121)
        # Third branch: z is None
        z_branch = next((inp for inp in sol.satisfying_inputs if inp.get("z") is None), None)
        self.assertIsNotNone(z_branch, "Should have a branch where z is None")

    def test_or_simple(self) -> None:
        sol = solve_condition(FIXTURE, 126)  # a > 10 or b == "yes"
        self.assertGreaterEqual(len(sol.satisfying_inputs), 2)
        # Branch 1: a > 10
        a_branch = sol.satisfying_inputs[0]
        self.assertIn("a", a_branch)
        self.assertGreater(a_branch["a"], 10)
        # Branch 2: b == "yes"
        b_branch = sol.satisfying_inputs[1]
        self.assertIn("b", b_branch)
        self.assertEqual(b_branch["b"], "yes")


class TestAbcStrategyHints(unittest.TestCase):
    """Fix #8: common ABC types should have strategy hints."""

    def test_pathlike_has_hint(self) -> None:
        from tharkun.solve.condition import _strategy_hint_for_type
        hint = _strategy_hint_for_type("PathLike", [])
        self.assertIsNotNone(hint)
        self.assertIn("pathlib", hint)

    def test_sequence_has_hint(self) -> None:
        from tharkun.solve.condition import _strategy_hint_for_type
        hint = _strategy_hint_for_type("Sequence", [])
        self.assertIsNotNone(hint)
        self.assertIn("st.lists", hint)

    def test_callable_has_hint(self) -> None:
        from tharkun.solve.condition import _strategy_hint_for_type
        hint = _strategy_hint_for_type("Callable", [])
        self.assertIsNotNone(hint)


class TestWalrusOperator(unittest.TestCase):
    """Fix #2: walrus operator (n := expr) should be visible to the solver.

    Note: walrus requires Python 3.8+. The fixture uses a manually constructed
    AST to avoid syntax errors on older versions.
    """

    def test_walrus_comparison_detected(self) -> None:
        """(n := len(data)) > 0 — n should be detected and typed as int."""
        import sys
        if sys.version_info < (3, 8):
            self.skipTest("walrus operator requires Python 3.8+")

        # Construct AST for: (n := len(data)) > 0
        named = ast.NamedExpr(
            target=ast.Name(id="n", ctx=ast.Store()),
            value=ast.Call(
                func=ast.Name(id="len", ctx=ast.Load()),
                args=[ast.Name(id="data", ctx=ast.Load())],
                keywords=[],
            ),
        )
        compare = ast.Compare(
            left=named,
            ops=[ast.Gt()],
            comparators=[ast.Constant(value=0)],
        )
        ast.fix_missing_locations(compare)

        t, src = infer_var_type("n", compare, FIXTURE, 0)
        self.assertEqual(t, "int")
        self.assertEqual(src, "comparison_literal")

    def test_walrus_value_extraction(self) -> None:
        """(n := len(data)) > 0 — should extract n > 0 → value 1."""
        import sys
        if sys.version_info < (3, 8):
            self.skipTest("walrus operator requires Python 3.8+")

        from tharkun.solve.condition import _extract_comparison_values

        named = ast.NamedExpr(
            target=ast.Name(id="n", ctx=ast.Store()),
            value=ast.Call(
                func=ast.Name(id="len", ctx=ast.Load()),
                args=[ast.Name(id="data", ctx=ast.Load())],
                keywords=[],
            ),
        )
        compare = ast.Compare(
            left=named,
            ops=[ast.Gt()],
            comparators=[ast.Constant(value=0)],
        )
        ast.fix_missing_locations(compare)

        values = _extract_comparison_values("n", compare)
        self.assertGreater(len(values), 0)
        self.assertGreater(values[0], 0)


class TestNegativeLiterals(unittest.TestCase):
    """IB-3 patch: x > -100 should work (negative literal is UnaryOp in AST)."""

    def test_negative_literal_type_inference(self) -> None:
        sol = solve_condition(FIXTURE, 131)  # x > -100 and x < -10
        x_var = next((v for v in sol.variables if v.name == "x"), None)
        self.assertIsNotNone(x_var)
        self.assertEqual(x_var.inferred_type, "int")

    def test_negative_literal_value_extraction(self) -> None:
        sol = solve_condition(FIXTURE, 131)
        x_var = next((v for v in sol.variables if v.name == "x"), None)
        self.assertIsNotNone(x_var)
        self.assertGreater(len(x_var.concrete_values), 0)
        for v in x_var.concrete_values:
            self.assertGreater(v, -100)
            self.assertLess(v, -10)


class TestRegexInversion(unittest.TestCase):
    """IB-3 patch: re.match(pattern, var) should generate a matching string."""

    def test_regex_type_inference(self) -> None:
        sol = solve_condition(FIXTURE, 139)  # re.match(r"^\d{3}-\d{2}-\d{4}$", ssn)
        ssn_var = next((v for v in sol.variables if v.name == "ssn"), None)
        self.assertIsNotNone(ssn_var)
        # Should infer str type from regex context
        self.assertIn(ssn_var.inferred_type, ("str", "Any"))

    def test_regex_value_matches_pattern(self) -> None:
        import re
        sol = solve_condition(FIXTURE, 139)
        ssn_var = next((v for v in sol.variables if v.name == "ssn"), None)
        self.assertIsNotNone(ssn_var)
        if ssn_var.concrete_values:
            # If hypothesis is available, should generate a matching value
            val = ssn_var.concrete_values[0]
            self.assertIsNotNone(re.match(r"^\d{3}-\d{2}-\d{4}$", val),
                                 f"Generated value {val!r} doesn't match SSN pattern")


class TestRegexCompiledPatterns(unittest.TestCase):
    """Beyond inline re.match(literal): compiled module-level patterns and bytes.

    Real parsers compile a pattern once (`VERSION_RE = re.compile(...)`) and use it via a
    bound method (`VERSION_RE.fullmatch(x)`), sometimes over bytes, sometimes with flags,
    sometimes with a COMPUTED pattern (`re.compile(r"[%s]+" % ...)`). tharkun must invert
    all of these into a genuinely matching value.
    """

    def _solve(self, src: str, guard_needle: str, var: str, desired="then"):
        import os
        import tempfile

        p = os.path.join(tempfile.mkdtemp(), "m.py")
        with open(p, "w") as f:
            f.write(textwrap.dedent(src))
        ln = next(i for i, l in enumerate(open(p), 1) if guard_needle in l)
        sol = solve_condition(p, ln, desired_branch=desired)
        return sol.satisfying_inputs[0].get(var) if sol.satisfying_inputs else None

    def test_compiled_str_pattern(self) -> None:
        import re

        v = self._solve('''
            import re
            VERSION_RE = re.compile(r"HTTP/(\\d)\\.(\\d)")
            def f(v):
                if VERSION_RE.fullmatch(v):
                    return 1
                return 0
        ''', "VERSION_RE.fullmatch(v)", "v")
        self.assertIsInstance(v, str)
        self.assertIsNotNone(re.compile(r"HTTP/(\d)\.(\d)").fullmatch(v), v)

    def test_compiled_bytes_pattern(self) -> None:
        import re

        v = self._solve('''
            import re
            SIG = re.compile(b"GET|POST|PUT")
            def f(data):
                if SIG.fullmatch(data):
                    return 1
                return 0
        ''', "SIG.fullmatch(data)", "data")
        self.assertIsInstance(v, bytes)
        self.assertIsNotNone(re.compile(b"GET|POST|PUT").fullmatch(v), v)

    def test_compiled_pattern_with_flags(self) -> None:
        import re

        v = self._solve('''
            import re
            HDR = re.compile(r"content-length", re.IGNORECASE)
            def f(name):
                if HDR.fullmatch(name):
                    return 1
                return 0
        ''', "HDR.fullmatch(name)", "name")
        self.assertIsInstance(v, str)
        self.assertIsNotNone(re.compile(r"content-length", re.I).fullmatch(v), v)

    def test_computed_pattern_resolved_by_import(self) -> None:
        # The pattern is not an AST literal (a % format over another constant), so it is
        # read from the actually-compiled Pattern by importing the module.
        import re

        v = self._solve('''
            import re
            SPECIALS = "!#$"
            TOKEN_RE = re.compile(r"[%s0-9a-zA-Z]+" % re.escape(SPECIALS))
            def f(name):
                if not TOKEN_RE.fullmatch(name):
                    raise ValueError
                return name
        ''', "if not TOKEN_RE.fullmatch(name)", "name", desired="else")
        self.assertIsInstance(v, str)
        self.assertIsNotNone(re.compile(r"[%s0-9a-zA-Z]+" % re.escape("!#$")).fullmatch(v), v)

    def test_inline_literal_still_works(self) -> None:
        import re

        v = self._solve('''
            import re
            def f(x):
                if re.match(r"^\\d{3}-\\d{2}$", x):
                    return 1
                return 0
        ''', "re.match(", "x")
        self.assertIsInstance(v, str)
        self.assertIsNotNone(re.compile(r"\d{3}-\d{2}").match(v), v)


class TestTruthinessAndConstraint(unittest.TestCase):
    """Fix: x and len(y) > 3 should produce a truthy x."""

    def test_truthiness_and_constraint(self) -> None:
        sol = solve_condition(FIXTURE, 144)  # x and len(y) > 3
        x_var = next((v for v in sol.variables if v.name == "x"), None)
        self.assertIsNotNone(x_var)
        # x must be truthy (not 0, not "", not [], etc.)
        self.assertGreater(len(x_var.concrete_values), 0)
        self.assertTrue(bool(x_var.concrete_values[0]),
                        f"x should be truthy, got {x_var.concrete_values[0]!r}")
        # The satisfying inputs should also have truthy x
        self.assertGreater(len(sol.satisfying_inputs), 0)
        self.assertTrue(bool(sol.satisfying_inputs[0]["x"]),
                        f"satisfying x should be truthy, got {sol.satisfying_inputs[0]['x']!r}")


class TestIsinstanceAstCall(unittest.TestCase):
    """Fix: isinstance(node, ast.Call) should produce a satisfying ast.Call value."""

    def test_isinstance_ast_call(self) -> None:
        import ast as _ast
        sol = solve_condition(FIXTURE, 149)  # isinstance(node, _ast.Call)
        node_var = next((v for v in sol.variables if v.name == "node"), None)
        self.assertIsNotNone(node_var)
        self.assertIn(node_var.inferred_type, ("_ast.Call", "ast.Call"))
        # Should have a satisfying value that is actually an ast.Call
        self.assertGreater(len(sol.satisfying_inputs), 0)
        val = sol.satisfying_inputs[0]["node"]
        self.assertIsInstance(val, _ast.Call,
                              f"Expected ast.Call instance, got {type(val).__name__}")


class TestAttrComparisonInSet(unittest.TestCase):
    """Fix: c.kind in {"method", "classmethod"} should produce a satisfying input."""

    def test_attr_comparison_in_set(self) -> None:
        sol = solve_condition(FIXTURE, 157)  # c.kind in {"method", "classmethod"}
        c_var = next((v for v in sol.variables if v.name == "c"), None)
        self.assertIsNotNone(c_var)
        self.assertEqual(c_var.inferred_type, "object")
        self.assertEqual(c_var.inference_source, "attr_comparison")
        # Should have at least one concrete value (a SimpleNamespace)
        self.assertGreater(len(c_var.concrete_values), 0)
        val = c_var.concrete_values[0]
        self.assertIn(val.kind, {"method", "classmethod"})
        # satisfying_inputs should contain the value
        self.assertGreater(len(sol.satisfying_inputs), 0)
        self.assertIn(sol.satisfying_inputs[0]["c"].kind, {"method", "classmethod"})


class TestAttrComparisonEq(unittest.TestCase):
    """Fix: p.has_default == True should produce a satisfying input."""

    def test_attr_comparison_eq(self) -> None:
        sol = solve_condition(FIXTURE, 162)  # p.has_default == True
        p_var = next((v for v in sol.variables if v.name == "p"), None)
        self.assertIsNotNone(p_var)
        self.assertEqual(p_var.inferred_type, "object")
        self.assertEqual(p_var.inference_source, "attr_comparison")
        self.assertGreater(len(p_var.concrete_values), 0)
        val = p_var.concrete_values[0]
        self.assertEqual(val.has_default, True)


class TestNestedAttr(unittest.TestCase):
    """Fix: node.value.id == "self" should at least not crash."""

    def test_nested_attr(self) -> None:
        sol = solve_condition(FIXTURE, 167)  # node.value.id == "self"
        node_var = next((v for v in sol.variables if v.name == "node"), None)
        self.assertIsNotNone(node_var)
        self.assertEqual(node_var.inferred_type, "object")
        self.assertEqual(node_var.inference_source, "attr_comparison")
        # Should produce a nested SimpleNamespace
        self.assertGreater(len(node_var.concrete_values), 0)
        val = node_var.concrete_values[0]
        self.assertEqual(val.value.id, "self")


class TestComprehensionFilter(unittest.TestCase):
    """Fix B: comprehension loop vars filtered from free names."""

    def test_items_is_free_but_x_is_not(self) -> None:
        sol = solve_condition(FIXTURE, 172)  # all(x > 0 for x in items)
        var_names = {v.name for v in sol.variables}
        self.assertIn("items", var_names)
        self.assertNotIn("x", var_names)


class TestHasattrRule(unittest.TestCase):
    """Fix C: hasattr(obj, 'ready') produces SimpleNamespace."""

    def test_hasattr_produces_namespace(self) -> None:
        sol = solve_condition(FIXTURE, 177)  # hasattr(obj, "ready")
        obj_var = next((v for v in sol.variables if v.name == "obj"), None)
        self.assertIsNotNone(obj_var)
        self.assertEqual(obj_var.inference_source, "hasattr")
        self.assertGreater(len(obj_var.concrete_values), 0)
        self.assertTrue(hasattr(obj_var.concrete_values[0], "ready"))


class TestChainedIsinstance(unittest.TestCase):
    """Fix E: isinstance(func.value, str) resolves root to func."""

    def test_chained_isinstance_resolves(self) -> None:
        sol = solve_condition(FIXTURE, 182)  # isinstance(func.value, str)
        func_var = next((v for v in sol.variables if v.name == "func"), None)
        self.assertIsNotNone(func_var)
        # Should infer type from isinstance — str
        self.assertEqual(func_var.inferred_type, "str")
        self.assertEqual(func_var.inference_source, "isinstance")


class TestDeepAttrTruthiness(unittest.TestCase):
    """Fix: bare attribute chain truthiness like frame.eval_ctx.volatile."""

    def test_deep_attr_builds_nested_namespace(self) -> None:
        sol = solve_condition(FIXTURE, 190)  # frame.eval_ctx.volatile
        frame_var = next((v for v in sol.variables if v.name == "frame"), None)
        self.assertIsNotNone(frame_var)
        self.assertEqual(frame_var.inference_source, "attr_truthiness")
        self.assertGreater(len(frame_var.concrete_values), 0)
        val = frame_var.concrete_values[0]
        self.assertTrue(val.eval_ctx.volatile)

    def test_negated_attr_builds_falsy(self) -> None:
        sol = solve_condition(FIXTURE, 195)  # not obj.allow_unicode
        obj_var = next((v for v in sol.variables if v.name == "obj"), None)
        self.assertIsNotNone(obj_var)
        self.assertEqual(obj_var.inference_source, "attr_truthiness_negated")
        val = obj_var.concrete_values[0]
        self.assertFalse(val.allow_unicode)

    def test_triple_deep_chain(self) -> None:
        sol = solve_condition(FIXTURE, 200)  # ctx.md.htmlStash.html_counter
        ctx_var = next((v for v in sol.variables if v.name == "ctx"), None)
        self.assertIsNotNone(ctx_var)
        val = ctx_var.concrete_values[0]
        self.assertTrue(val.md.htmlStash.html_counter)

    def test_strategy_is_direct(self) -> None:
        sol = solve_condition(FIXTURE, 190)
        self.assertEqual(sol.strategy, "direct")


class TestMathPredicates(unittest.TestCase):
    """Math module predicate semantics."""

    def test_isnan(self) -> None:
        import math
        sol = solve_condition(FIXTURE, 209)
        self.assertEqual(sol.strategy, "direct")
        val = sol.satisfying_inputs[0]["value"]
        self.assertTrue(math.isnan(val))

    def test_isinf_and_negative(self) -> None:
        import math
        sol = solve_condition(FIXTURE, 214)
        self.assertEqual(sol.strategy, "direct")
        val = sol.satisfying_inputs[0]["value"]
        self.assertTrue(math.isinf(val))
        self.assertLess(val, 0)

    def test_not_isfinite(self) -> None:
        import math
        sol = solve_condition(FIXTURE, 219)
        self.assertEqual(sol.strategy, "direct")
        val = sol.satisfying_inputs[0]["value"]
        self.assertFalse(math.isfinite(val))


class TestSpecialPatterns(unittest.TestCase):
    """Special guard patterns: any/all generators, modular, callable, containment."""

    def test_any_startswith(self) -> None:
        sol = solve_condition(FIXTURE, 226)
        self.assertEqual(sol.strategy, "direct")
        val = sol.satisfying_inputs[0]["value"]
        self.assertTrue(any(val.startswith(p) for p in {"10.", "192.168.", "127."}))

    def test_modular_membership(self) -> None:
        sol = solve_condition(FIXTURE, 231)
        self.assertEqual(sol.strategy, "direct")
        val = sol.satisfying_inputs[0]["value"]
        self.assertIn(val % 100, (11, 12, 13))

    def test_callable(self) -> None:
        sol = solve_condition(FIXTURE, 236)
        self.assertEqual(sol.strategy, "direct")
        val = sol.satisfying_inputs[0]["fmt"]
        self.assertTrue(callable(val))

    def test_string_containment(self) -> None:
        sol = solve_condition(FIXTURE, 241)
        self.assertEqual(sol.strategy, "direct")
        val = sol.satisfying_inputs[0]["value"]
        self.assertIn("x", val)


class TestSelfCompoundPatterns(unittest.TestCase):
    """Self.xxx compound patterns: is not None, method calls, len()."""

    def test_self_is_not_none(self) -> None:
        sol = solve_condition(FIXTURE, 249)
        self.assertEqual(sol.strategy, "direct")
        self_obj = sol.satisfying_inputs[0]["self"]
        self.assertIsNotNone(self_obj.encoding)

    def test_self_compound_method(self) -> None:
        sol = solve_condition(FIXTURE, 253)
        self.assertEqual(sol.strategy, "direct")
        self_obj = sol.satisfying_inputs[0]["self"]
        self.assertIsNotNone(self_obj.encoding)
        self.assertTrue(callable(self_obj._usable))

    def test_self_len_contents(self) -> None:
        sol = solve_condition(FIXTURE, 257)
        self.assertEqual(sol.strategy, "direct")
        self_obj = sol.satisfying_inputs[0]["self"]
        self.assertNotEqual(len(self_obj.contents), 1)

    def test_self_method_truthiness(self) -> None:
        sol = solve_condition(FIXTURE, 261)
        self.assertEqual(sol.strategy, "direct")
        self_obj = sol.satisfying_inputs[0]["self"]
        self.assertTrue(callable(self_obj._usable))


class TestLiteralTrue(unittest.TestCase):
    def test_while_true(self) -> None:
        sol = solve_condition(FIXTURE, 268)
        self.assertEqual(sol.strategy, "direct")
        self.assertEqual(sol.confidence, 1.0)


class TestFalseBranchSolving(unittest.TestCase):
    """Tests for desired_branch='else' (false-branch) solving."""

    def test_else_numeric_gt_produces_non_satisfying_value(self) -> None:
        # Line 5: `if x > 100:` — "else" wants x <= 100
        sol = solve_condition(FIXTURE, 5, desired_branch="else")
        self.assertIsNotNone(sol)
        self.assertTrue(len(sol.satisfying_inputs) > 0)
        x = sol.satisfying_inputs[0].get("x")
        if x is not None and isinstance(x, (int, float)):
            self.assertLessEqual(x, 100)

    def test_then_and_else_produce_opposite_values(self) -> None:
        # Line 5: `if x > 100:`
        sol_then = solve_condition(FIXTURE, 5, desired_branch="then")
        sol_else = solve_condition(FIXTURE, 5, desired_branch="else")
        x_then = sol_then.satisfying_inputs[0].get("x") if sol_then.satisfying_inputs else None
        x_else = sol_else.satisfying_inputs[0].get("x") if sol_else.satisfying_inputs else None
        if x_then is not None and x_else is not None:
            # then: x > 100, else: x <= 100
            self.assertGreater(x_then, 100)
            self.assertLessEqual(x_else, 100)

    def test_else_string_eq_guard(self) -> None:
        # Line 7: `if name == "hello":` — "else" wants name != "hello"
        sol = solve_condition(FIXTURE, 7, desired_branch="else")
        self.assertIsNotNone(sol)
        self.assertTrue(len(sol.satisfying_inputs) > 0)
        name = sol.satisfying_inputs[0].get("name")
        if name is not None:
            self.assertNotEqual(name, "hello")

    def test_desired_branch_recorded_in_solution(self) -> None:
        sol_then = solve_condition(FIXTURE, 5, desired_branch="then")
        sol_else = solve_condition(FIXTURE, 5, desired_branch="else")
        self.assertEqual(sol_then.desired_branch, "then")
        self.assertEqual(sol_else.desired_branch, "else")

    def test_serialize_includes_desired_branch(self) -> None:
        sol = solve_condition(FIXTURE, 5, desired_branch="else")
        d = serialize_solution(sol)
        self.assertIn("desired_branch", d)
        self.assertEqual(d["desired_branch"], "else")

    def test_for_else_sentinel(self) -> None:
        """For-loop 'else' branch returns __iter_empty__ sentinel."""
        import tempfile, os
        src = "def f(items):\n    for x in items:\n        pass\n"
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
            tmp.write(src)
            path = tmp.name
        try:
            sol = solve_condition(path, 2, desired_branch="else")
            self.assertIsNotNone(sol)
            self.assertTrue(len(sol.satisfying_inputs) > 0)
            self.assertIn("__iter_empty__", sol.satisfying_inputs[0])
        finally:
            os.unlink(path)

    def test_while_else_via_negation(self) -> None:
        """While guard false-branch: condition is negated to find inputs."""
        import tempfile, os
        src = "def f(n):\n    while n > 0:\n        break\n"
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
            tmp.write(src)
            path = tmp.name
        try:
            sol = solve_condition(path, 2, desired_branch="else")
            self.assertIsNotNone(sol)
            if sol.satisfying_inputs:
                n = sol.satisfying_inputs[0].get("n")
                if n is not None and isinstance(n, (int, float)):
                    self.assertLessEqual(n, 0)
        finally:
            os.unlink(path)


class TestInputConstraints(unittest.TestCase):
    """Tests for input_constraints field on ConditionSolution."""

    def test_int_comparison_produces_gte(self) -> None:
        # Line 5: `if x > 100:`
        sol = solve_condition(FIXTURE, 5)
        self.assertIsNotNone(sol.input_constraints)
        self.assertIn("x", sol.input_constraints)
        c = sol.input_constraints["x"]
        self.assertEqual(c["type"], "int")
        self.assertIn("gte", c)
        self.assertGreater(c["gte"], 100)

    def test_str_comparison_produces_eq(self) -> None:
        # Line 7: `if name == "hello":`
        sol = solve_condition(FIXTURE, 7)
        if sol.input_constraints and "name" in sol.input_constraints:
            c = sol.input_constraints["name"]
            self.assertEqual(c["type"], "str")
            self.assertIn("eq", c)
            self.assertEqual(c["eq"], "hello")

    def test_input_constraints_in_serialization(self) -> None:
        sol = solve_condition(FIXTURE, 5)
        d = serialize_solution(sol)
        self.assertIn("input_constraints", d)

    def test_unsolved_guard_has_no_crash(self) -> None:
        sol = solve_condition("/nonexistent/file.py", 1)
        # Should not raise — input_constraints is None for unsolvable
        self.assertIsNone(sol.input_constraints)


class TestStrategyExplanations(unittest.TestCase):
    """Tests for explanation field on ConditionSolution."""

    def test_explanation_is_string(self) -> None:
        sol = solve_condition(FIXTURE, 5)
        self.assertIsInstance(sol.explanation, str)
        self.assertGreater(len(sol.explanation), 0)

    def test_explanation_references_guard(self) -> None:
        sol = solve_condition(FIXTURE, 5)
        # The explanation should reference the guard expression or the strategy.
        explanation_lower = sol.explanation.lower()
        self.assertTrue(
            "x > 100" in explanation_lower or "direct" in explanation_lower
            or "guard" in explanation_lower or "solved" in explanation_lower,
            f"Unexpected explanation: {sol.explanation!r}"
        )

    def test_for_else_explanation_mentions_iterable(self) -> None:
        import tempfile, os
        src = "def f(items):\n    for x in items:\n        pass\n"
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
            tmp.write(src)
            path = tmp.name
        try:
            sol = solve_condition(path, 2, desired_branch="else")
            self.assertIn("iterable", sol.explanation.lower())
        finally:
            os.unlink(path)

    def test_explanation_in_serialization(self) -> None:
        sol = solve_condition(FIXTURE, 5)
        d = serialize_solution(sol)
        self.assertIn("explanation", d)
        self.assertIsInstance(d["explanation"], str)

    def test_infeasible_explanation(self) -> None:
        try:
            from tharkun.minleaf.types import FactEnv

            class _Infeasible:
                possible = False
                facts: dict = {}

            sol = solve_condition(FIXTURE, 5, facts=_Infeasible())
            self.assertIn("infeasible", sol.explanation.lower())
        except ImportError:
            self.skipTest("minleaf not available")


if __name__ == "__main__":
    unittest.main()
