"""Is every technique in the chain actually able to win something?

Asked directly, and the answer was no. Instrumenting a real ntlm_auth run showed the
full portfolio being *invoked* but only two members ever producing a committed flip:

    invoked   touchstone 1158  snapshot 1158  static 929  hypothesis 344  redqueen 30
    committed snapshot 309     static 174     everything else 0

Two distinct problems hid behind that. RedQueen was invoked 30 times against the cheap
tier's 1158 because cross-file targeting read live objects only from the ENTRY file, so a
blocker inside a callee had no state for the techniques that need it. And RedQueen could
not invert `int.from_bytes` at all — the shape binary parsers are made of, and exactly
where it is supposed to beat a static solver, because the demanded value exists only at
runtime.

One toy per technique, each shaped so that technique should be the one to crack it. The
point is not that a particular member wins — the portfolio is free to find a cheaper
route — but that each is *capable* of winning when it is the only one that can.
"""
from __future__ import annotations

import ast
import os
import sys
import unittest

from tharkun.driver import PLATEAU, REACHED, drive_to_line

FIX = os.path.join(os.path.dirname(__file__), "fixtures")
MOD = os.path.join(FIX, "technique_toys.py")


def setUpModule() -> None:
    if FIX not in sys.path:
        sys.path.insert(0, FIX)


def _target(func: str) -> int:
    """First line of the first if-body in *func* — the drive target."""
    tree = ast.parse(open(MOD).read())
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == func:
            for x in ast.walk(n):
                if isinstance(x, ast.If):
                    return x.body[0].lineno
    raise AssertionError(func)


class TestEachTechniqueCanWin(unittest.TestCase):
    def _drive(self, func: str, seed: dict, **kw):
        return drive_to_line(MOD, func, _target(func), seed, max_rounds=10, **kw)

    def test_static_literal_comparison(self) -> None:
        import technique_toys

        r = self._drive("by_static", {"code": 0})
        self.assertEqual(r.status, REACHED)
        self.assertEqual(technique_toys.by_static(**r.inputs), "STATIC")

    def test_minleaf_flips_one_leaf_of_live_state(self) -> None:
        import technique_toys

        r = self._drive("by_minleaf", {"cfg": {"limits": [1, 2, 3]}})
        self.assertEqual(r.status, REACHED)
        self.assertEqual(technique_toys.by_minleaf(**r.inputs), "MINLEAF")

    def test_minleaf_is_the_one_that_wins_it(self) -> None:
        # No other member can: the value is a leaf deep in live state, and the demanded
        # change is a single element rather than a rebuilt structure.
        r = self._drive("by_minleaf", {"cfg": {"limits": [1, 2, 3]}})
        self.assertTrue(r.history[-1].technique.startswith("minleaf"),
                        f"minleaf did not win it: {r.history[-1].technique}")


class TestRedqueenIntDecodeShapes(unittest.TestCase):
    """The matcher itself, across the encodings a binary parser actually uses."""

    def _match(self, src: str):
        from tharkun.redqueen import _match_int_decode

        return _match_int_decode(ast.parse(src, mode="eval").body, "p")

    def test_big_endian_slice(self) -> None:
        # `p[:2]` has no explicit lower bound; None means "from the start".
        self.assertEqual(self._match('int.from_bytes(p[:2], "big")'), (None, 2, "big", False))
        self.assertEqual(self._match('int.from_bytes(p[0:2], "big")'), (0, 2, "big", False))

    def test_little_endian_keyword(self) -> None:
        self.assertEqual(self._match('int.from_bytes(p[4:8], byteorder="little")'),
                         (4, 8, "little", False))

    def test_signed_keyword_is_carried(self) -> None:
        self.assertEqual(self._match('int.from_bytes(p[0:4], "big", signed=True)'),
                         (0, 4, "big", True))

    def test_whole_parameter_with_no_slice(self) -> None:
        self.assertEqual(self._match('int.from_bytes(p, "big")'), (0, None, "big", False))

    def test_struct_unpack_byte_order_from_format(self) -> None:
        self.assertEqual(self._match('struct.unpack(">H", p[0:2])[0]'), (0, 2, "big", False))
        self.assertEqual(self._match('struct.unpack("<I", p[0:4])[0]'), (0, 4, "little", False))

    def test_unrelated_call_is_not_matched(self) -> None:
        self.assertIsNone(self._match("len(p)"))
        self.assertIsNone(self._match('int.from_bytes(q[:2], "big")'))   # other param


if __name__ == "__main__":
    unittest.main()
