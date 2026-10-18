"""[internal ref] slice 3 ([internal ref]): driver.py's on-demand Avengers object flip.

Two layers: the pure guard-to-goal AST helpers driver.py owns (fast, no
Avengers invoked -- this is analysis of the program drive_to_line is driving,
not Avengers' own concern, corrected 2026-09-29: driver.py calls
tharkun.avengers's existing public API directly, no intermediate service
module), and one end-to-end toy proving the actual capability gain -- a guard
on an attribute of a parameter with NO live value, which minleaf's own
documented blind spot ("nothing to mutate") leaves drive_to_line unable to
flip otherwise. The tier runs unconditionally, on demand only (owner ruling,
no feature flags) -- see the design note on [internal ref].
"""
from __future__ import annotations

import ast
import os
import tempfile
import textwrap
import unittest

from tharkun.avengers import FieldBound
from tharkun.driver import (
    PLATEAU,
    REACHED,
    _avengers_classes_with_attrs,
    _avengers_field_bounds_from_guard,
    _avengers_negate_bounds,
    drive_to_line,
)


def _guard(src: str, lineno: int) -> ast.AST:
    tree = ast.parse(textwrap.dedent(src))
    from tharkun.solve.condition import _find_branch_at_line

    node = _find_branch_at_line(tree, lineno)
    assert node is not None
    return node


class TestFieldBoundsFromGuard(unittest.TestCase):
    def test_comparison_both_orders(self) -> None:
        node = _guard("if c.x > 3:\n    pass\n", 1)
        self.assertEqual(_avengers_field_bounds_from_guard(node, "c"), (FieldBound("x", "gt", 3),))
        # the literal on the left flips the operator
        node = _guard("if 3 < c.x:\n    pass\n", 1)
        self.assertEqual(_avengers_field_bounds_from_guard(node, "c"), (FieldBound("x", "gt", 3),))

    def test_truthy_and_falsy(self) -> None:
        node = _guard("if c.items:\n    pass\n", 1)
        self.assertEqual(_avengers_field_bounds_from_guard(node, "c"), (FieldBound("items", "truthy"),))
        node = _guard("if not c.items:\n    pass\n", 1)
        self.assertEqual(_avengers_field_bounds_from_guard(node, "c"), (FieldBound("items", "falsy"),))

    def test_conjunction_collects_every_atom(self) -> None:
        node = _guard("if c.x > 3 and c.items:\n    pass\n", 1)
        got = set(_avengers_field_bounds_from_guard(node, "c"))
        self.assertEqual(got, {FieldBound("x", "gt", 3), FieldBound("items", "truthy")})

    def test_ignores_a_different_parameter(self) -> None:
        node = _guard("if d.x > 3:\n    pass\n", 1)
        self.assertEqual(_avengers_field_bounds_from_guard(node, "c"), ())

    def test_non_literal_comparator_is_skipped(self) -> None:
        # a container literal is not an ast.Constant -- documented gap, not silently wrong
        node = _guard("if c.items == []:\n    pass\n", 1)
        self.assertEqual(_avengers_field_bounds_from_guard(node, "c"), ())


class TestNegateBounds(unittest.TestCase):
    def test_negation_table(self) -> None:
        self.assertEqual(_avengers_negate_bounds((FieldBound("x", "gt", 3),)),
                         (FieldBound("x", "le", 3),))
        self.assertEqual(_avengers_negate_bounds((FieldBound("items", "truthy"),)),
                         (FieldBound("items", "falsy"),))


class TestClassesWithAttrs(unittest.TestCase):
    def test_finds_the_class_that_assigns_every_attr(self) -> None:
        tree = ast.parse(textwrap.dedent("""
            class A:
                def __init__(self):
                    self.x = 1

            class B:
                def __init__(self, x, items):
                    self.x = x
                    self.items = items
        """))
        self.assertEqual(_avengers_classes_with_attrs(tree, {"x"}), ["A", "B"])
        self.assertEqual(_avengers_classes_with_attrs(tree, {"x", "items"}), ["B"])
        self.assertEqual(_avengers_classes_with_attrs(tree, {"nope"}), [])


class TestDriveToLineEndToEnd(unittest.TestCase):
    """The capability this slice adds: reaching a guard drive_to_line could
    not reach before, because the parameter it needs is an OBJECT with no
    live value at all -- minleaf's documented blind spot."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp(prefix="tharkun-driver-avengers-test-")
        self.toy = os.path.join(self.tmp, "toy.py")
        with open(self.toy, "w") as f:
            # the owner's own example ([internal ref]): "a C whose .x > 3 and
            # .items is non-empty" -- a combined bound the search could NOT
            # find before the ObjectGoal.field_bounds fix (measured: 0/15
            # independent fresh-search attempts, ~130s), and now finds
            # reliably in a few seconds (HypothesisCandidatePass's own search
            # targets the bounds directly).
            f.write(textwrap.dedent("""
                class C:
                    def __init__(self, x: int, items: list):
                        self.x = x
                        self.items = items


                def f(c):
                    if c is not None and c.x > 3 and c.items:
                        return 1          # the target
                    return 0
            """))
        with open(self.toy) as f:
            for i, line in enumerate(f.readlines(), 1):
                if "the target" in line:
                    self.target_line = i
                    break

    def test_default_reaches_it(self) -> None:
        """No flag: the tier is unconditionally in the default portfolio (owner
        ruling, no feature flags, [internal ref])."""
        r = drive_to_line(self.toy, "f", self.target_line, {"c": None}, max_rounds=4,
                          avengers_budget_seconds=15.0)
        self.assertEqual(r.status, REACHED)
        c = r.inputs["c"]
        self.assertGreater(c.x, 3)
        self.assertTrue(c.items)

    def test_no_attribute_bound_stays_correct(self) -> None:
        """A plain scalar guard has no attribute bound to extract -- avengers must
        stay a no-op there (never interferes with a case it was not built for),
        and the existing cheap tier still solves it exactly as before."""
        toy2 = os.path.join(self.tmp, "toy_scalar.py")
        with open(toy2, "w") as f:
            f.write(textwrap.dedent("""
                def f(n):
                    if n > 1000000:
                        return 1          # the target
                    return 0
            """))
        with open(toy2) as f:
            for i, line in enumerate(f.readlines(), 1):
                if "the target" in line:
                    target_line = i
                    break
        r = drive_to_line(toy2, "f", target_line, {"n": 0}, max_rounds=4)
        # solved by the existing cheap tier (static/snapshot); avengers never engages
        self.assertEqual(r.status, REACHED)

    def test_unsatisfiable_bound_never_reaches(self) -> None:
        """A guard whose bounds are genuinely UNSATISFIABLE by any real
        construction must stay PLATEAU, never a false anything -- Avengers'
        own search-until-milestone loop simply exhausts its budget and
        _flip_verifies never gets a candidate to (mis)confirm."""
        toy3 = os.path.join(self.tmp, "toy_unsat.py")
        with open(toy3, "w") as f:
            f.write(textwrap.dedent("""
                class C:
                    def __init__(self, x: int):
                        self.x = x


                def f(c):
                    if c is not None and c.x > 3 and c.x < 0:
                        return 1          # the target
                    return 0
            """))
        with open(toy3) as f:
            for i, line in enumerate(f.readlines(), 1):
                if "the target" in line:
                    target_line = i
                    break
        r = drive_to_line(toy3, "f", target_line, {"c": None}, max_rounds=4,
                          avengers_budget_seconds=3.0)
        self.assertEqual(r.status, PLATEAU)


if __name__ == "__main__":
    unittest.main()
