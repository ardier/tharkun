"""call_mapper backward propagation: solving a guard on a callee PARAMETER.

When a blocking guard reads a callee's own parameter that the caller binds to a SIMPLE
entry-parameter name (`validate(payload)` — a pass-through, not a slice or a transform),
the driver propagates the guard's solution to the entry parameter directly, via the
`call_mapper` lane, instead of blind-searching for it. A non-simple argument (a slice,
`int.from_bytes`, any expression) is left to the reach-search — the expression-inversion
non-goal SPEC_constraint_propagation.md declares out of scope for the driver.
"""
from __future__ import annotations

import os
import tempfile
import textwrap
import unittest

from tharkun.driver import REACHED, drive_to_line


def _toy(src: str, func: str, target: int, seed: dict, **kw):
    p = os.path.join(tempfile.mkdtemp(), "m.py")
    with open(p, "w") as f:
        f.write(textwrap.dedent(src))
    return drive_to_line(p, func, target, seed, max_rounds=8, **kw)


class TestBackwardPropagation(unittest.TestCase):
    def _tech(self, r) -> str:
        return r.history[-1].technique if r.history else ""

    def test_a_length_guard_on_a_passed_through_parameter(self) -> None:
        r = _toy('''
            def validate(key):
                if len(key) != 8:
                    return "bad"
                return "ok"

            def entry(payload):
                return validate(payload)
        ''', "entry", 4, {"payload": b"12345678"})
        self.assertEqual(r.status, REACHED, r.diagnostics)
        self.assertEqual(self._tech(r), "call_mapper")
        self.assertNotEqual(len(r.inputs["payload"]), 8)         # the guard is satisfied
        self.assertIsInstance(r.inputs["payload"], bytes)        # coerced to the entry type

    def test_a_magic_value_guard_on_a_passed_through_parameter(self) -> None:
        r = _toy('''
            def check(code):
                if code == 48879:
                    return "hit"
                return "no"

            def entry(x):
                return check(x)
        ''', "entry", 4, {"x": 0})
        self.assertEqual(r.status, REACHED, r.diagnostics)
        self.assertEqual(self._tech(r), "call_mapper")
        self.assertEqual(r.inputs["x"], 48879)

    def test_a_keyword_argument_pass_through(self) -> None:
        r = _toy('''
            def check(code):
                if code == 7:
                    return "hit"
                return "no"

            def entry(x):
                return check(code=x)
        ''', "entry", 4, {"x": 0})
        self.assertEqual(r.status, REACHED, r.diagnostics)
        self.assertEqual(self._tech(r), "call_mapper")
        self.assertEqual(r.inputs["x"], 7)

    def test_a_slice_argument_is_left_to_the_reach_search(self) -> None:
        # NON-GOAL boundary: the callee parameter is bound to a slice of the entry
        # parameter, which needs expression inversion the driver does not do — so the
        # backward lane must NOT claim it; the reach-search handles it (or plateaus).
        r = _toy('''
            def validate(key):
                if len(key) != 7:
                    return "hit"
                return "no"

            def entry(payload):
                return validate(payload[:7])
        ''', "entry", 4, {"payload": b"abcdefg"})
        self.assertNotEqual(self._tech(r), "call_mapper")

    def test_a_local_derived_guard_is_not_claimed(self) -> None:
        # The guard reads a callee LOCAL (a slice result), not a parameter — no simple
        # backward mapping exists, so the lane abstains.
        r = _toy('''
            def validate(data):
                head = data[:4]
                if len(head) != 4:
                    return "hit"
                return "no"

            def entry(payload):
                return validate(payload)
        ''', "entry", 5, {"payload": b"abcd"})
        self.assertNotEqual(self._tech(r), "call_mapper")


if __name__ == "__main__":
    unittest.main()
