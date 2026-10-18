"""Reaching guards that live several CALLS below the entry point.

The harness only ever calls the entry function, but the interesting guards are one and
two calls down. Two things were needed and both were missing:

  * blocker finding was purely lexical, so a guard in a CALLER — the thing actually
    stopping us — was invisible, and the driver kept flipping the right-looking guard in
    a callee that was never invoked;
  * the path prefix was also lexical, so the solver could satisfy a local guard while
    breaking a caller's, and the flip failed verification for a reason it never saw.
"""
from __future__ import annotations

import ast
import os
import unittest

from tharkun.driver import PLATEAU, REACHED, drive_to_line


TARGET = os.path.join(os.path.dirname(__file__), "hybrid", "fixtures", "layers_target.py")


def _body_of(guard_line: int) -> int:
    tree = ast.parse(open(TARGET).read())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.If) and n.lineno == guard_line)
    return node.body[0].lineno


class TestCallLayers(unittest.TestCase):
    SEED = {"a": 0, "b": 0, "c": 0}

    def _drive(self, guard_line: int):
        return drive_to_line(TARGET, "top", _body_of(guard_line),
                             seed=dict(self.SEED), max_rounds=6)

    def test_entry_function_guard(self) -> None:
        r = self._drive(7)
        self.assertEqual(r.status, REACHED)
        self.assertEqual(r.inputs["a"], 100)

    def test_one_call_down(self) -> None:
        # b == 200 lives in mid(), which top() only calls when a == 100.
        r = self._drive(15)
        self.assertEqual(r.status, REACHED)
        self.assertEqual((r.inputs["a"], r.inputs["b"]), (100, 200))

    def test_two_calls_down_nonlinear(self) -> None:
        r = self._drive(23)
        self.assertEqual(r.status, REACHED)
        self.assertEqual(r.inputs["c"] * r.inputs["c"], 900)

    def test_two_calls_down_nested_needs_caller_guards_preserved(self) -> None:
        # c*c == 900 admits -30 and +30; only +30 also satisfies a+b+c == 330 with the
        # caller guards (a == 100, b == 200) still holding. Without those in the prefix
        # the solver picks -30 and the nested guard becomes unsatisfiable.
        r = self._drive(25)
        self.assertEqual(r.status, REACHED)
        self.assertEqual(r.inputs, {"a": 100, "b": 200, "c": 30})
        # Caller-side proposals must be recorded against the guards they actually
        # flipped.  Otherwise the path state loses these constraints and the nonlinear
        # solve can nondeterministically choose c == -30.
        self.assertEqual([step.blocker_line for step in r.history[:2]], [7, 15])
