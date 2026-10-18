"""Bitmask flag guards — `if flags & MASK:`.

The comparison extractor walks `ast.Compare` nodes, so a bare bitwise truthiness test
produced nothing at all: measured on ntlm_auth, three of five reachable frontiers were
this one shape, and every one came back `strategy: none`. The guard is trivially
invertible once the mask is a number, which is the actual work — protocol code names its
masks (`Flags.EXTENDED_SESSIONSECURITY`), usually in a sibling module.
"""
from __future__ import annotations

import ast
import os
import unittest

from tharkun import solve_condition
from tharkun.solve.condition import (
    _extract_bitmask_values,
    _find_branch_at_line,
    _resolve_int_constant,
)

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "flagpkg")
READER = os.path.join(FIX, "reader.py")


def _test_expr(path: str, lineno: int) -> ast.expr:
    return _find_branch_at_line(ast.parse(open(path).read()), lineno)


def _line_of(path: str, needle: str) -> int:
    """Line of the `if`/`elif` containing *needle* — never the import that names it."""
    for i, line in enumerate(open(path), start=1):
        stripped = line.strip()
        if needle in line and (stripped.startswith("if ") or stripped.startswith("elif ")):
            return i
    raise AssertionError(f"no if/elif line containing {needle!r}")


class TestMaskConstantResolution(unittest.TestCase):
    def test_resolves_class_attribute_across_modules(self) -> None:
        expr = _test_expr(READER, _line_of(READER, "Flags.EXTENDED_SESSIONSECURITY"))
        self.assertEqual(_resolve_int_constant(expr.right, READER), 0x00080000)

    def test_resolves_module_level_constant(self) -> None:
        expr = _test_expr(READER, _line_of(READER, "MODULE_LEVEL_MASK"))
        self.assertEqual(_resolve_int_constant(expr.right, READER), 0x40)

    def test_literal_mask_needs_no_resolution(self) -> None:
        self.assertEqual(_resolve_int_constant(ast.parse("8", mode="eval").body, ""), 8)

    def test_unresolvable_name_is_none(self) -> None:
        node = ast.parse("Nowhere.NOPE", mode="eval").body
        self.assertIsNone(_resolve_int_constant(node, READER))

    def test_owner_class_scopes_the_lookup(self) -> None:
        # MODULE_LEVEL_MASK exists at module level, but not on Flags — so a dotted
        # reference to it must not silently pick up the module-level one.
        node = ast.parse("Flags.MODULE_LEVEL_MASK", mode="eval").body
        self.assertIsNone(_resolve_int_constant(node, os.path.join(FIX, "constants.py")))


class TestBitmaskExtraction(unittest.TestCase):
    def test_truthiness_yields_the_mask(self) -> None:
        expr = _test_expr(READER, _line_of(READER, "Flags.EXTENDED_SESSIONSECURITY"))
        self.assertEqual(_extract_bitmask_values("negotiate_flags", expr, READER),
                         [0x00080000])

    def test_not_equal_zero_yields_the_mask(self) -> None:
        expr = _test_expr(READER, _line_of(READER, "!= 0"))
        self.assertEqual(_extract_bitmask_values("flags", expr, READER), [0x1])

    def test_equals_mask_yields_the_mask(self) -> None:
        expr = _test_expr(READER, _line_of(READER, "== Flags.NEGOTIATE_SEAL"))
        self.assertEqual(_extract_bitmask_values("flags", expr, READER), [0x20])

    def test_equals_zero_yields_zero(self) -> None:
        expr = _test_expr(READER, _line_of(READER, "Flags.NEGOTIATE_SEAL == 0"))
        self.assertEqual(_extract_bitmask_values("flags", expr, READER), [0])

    def test_reversed_operand_order(self) -> None:
        expr = ast.parse("0x20 & flags", mode="eval").body
        self.assertEqual(_extract_bitmask_values("flags", expr, ""), [0x20])

    def test_compared_operand_is_not_also_read_as_truthiness(self) -> None:
        # `flags & M == 0` is satisfied by CLEARING the bits. Walking into the inner
        # BinOp and offering M as well would hand back a value that makes it false.
        expr = _test_expr(READER, _line_of(READER, "Flags.NEGOTIATE_SEAL == 0"))
        self.assertEqual(_extract_bitmask_values("flags", expr, READER), [0])

    def test_zero_mask_is_not_offered(self) -> None:
        # `flags & 0` is falsy for every input; offering 0 would be a wrong answer.
        expr = ast.parse("flags & 0", mode="eval").body
        self.assertEqual(_extract_bitmask_values("flags", expr, ""), [])

    def test_other_variable_is_not_claimed(self) -> None:
        expr = _test_expr(READER, _line_of(READER, "Flags.EXTENDED_SESSIONSECURITY"))
        self.assertEqual(_extract_bitmask_values("other", expr, READER), [])


class TestEndToEndSolve(unittest.TestCase):
    def test_each_branch_solves_to_a_passing_value(self) -> None:
        import sys
        sys.path.insert(0, os.path.dirname(FIX))
        from flagpkg.reader import read

        for needle, expected in (("Flags.EXTENDED_SESSIONSECURITY", "extended"),
                                 ("Flags.NEGOTIATE_SEAL:", "seal"),
                                 ("MODULE_LEVEL_MASK", "module-mask")):
            line = _line_of(READER, needle)
            sol = solve_condition(READER, line)
            self.assertTrue(sol.satisfying_inputs, f"no inputs for L{line} ({needle})")
            flags = sol.satisfying_inputs[0]["negotiate_flags"]
            # The real judge: run it. A mask that does not actually take the branch is
            # not a solution, whatever the solver called it.
            self.assertEqual(read(flags), expected)

    def test_bitwise_operand_infers_int(self) -> None:
        sol = solve_condition(READER, _line_of(READER, "Flags.EXTENDED_SESSIONSECURITY"))
        var = {v.name: v for v in sol.variables}["negotiate_flags"]
        self.assertEqual(var.inferred_type, "int")


if __name__ == "__main__":
    unittest.main()
