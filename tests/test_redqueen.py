"""Tests for live input-to-state correspondence harvesting used by reach-search."""
from __future__ import annotations

import ast
import os
import tempfile
import textwrap
import unittest

from tharkun.redqueen import Correspondence, harvest


def _write(src: str) -> str:
    d = tempfile.mkdtemp()
    p = os.path.join(d, "mod.py")
    with open(p, "w") as f:
        f.write(textwrap.dedent(src))
    return p


def _guard_line(path: str) -> int:
    """First ``if`` line in the file."""
    tree = ast.parse(open(path).read())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.If)).lineno


# ---------------------------------------------------------------------------
# harvest(): pure-logic correspondence detection (no optional deps)
# ---------------------------------------------------------------------------


class TestHarvestIdentity(unittest.TestCase):
    def test_pure_identity_reads_the_demanded_literal(self) -> None:
        p = _write("""
            def f(x):
                if x == 4242:
                    return "hit"
                return "miss"
        """)
        corrs = harvest(p, _guard_line(p), {"x": 0}, {"x"})
        self.assertEqual(len(corrs), 1)
        c = corrs[0]
        self.assertEqual((c.input_name, c.kind, c.demanded, c.proposal), ("x", "identity", 4242, 4242))
        self.assertGreaterEqual(c.confidence, 0.9)

    def test_identity_reads_a_runtime_constant_not_a_literal(self) -> None:
        # The demanded value is computed at runtime; no literal appears in the source.
        # RedQueen *observes* it by evaluating the operand against the live env.
        p = _write("""
            def secret():
                return sum([10, 10, 2])
            def f(x):
                if x == secret():
                    return "hit"
                return "miss"
        """)
        corrs = harvest(p, _guard_line(p), {"x": 0, "secret": lambda: 22}, {"x"})
        self.assertEqual([(c.kind, c.demanded, c.proposal) for c in corrs], [("identity", 22, 22)])

    def test_identity_through_value_identity_transform(self) -> None:
        # token.swapcase().swapcase() is value-identity on the seed → a strong (not top)
        # confidence identity correspondence, proposing the demanded value.
        p = _write("""
            def f(token):
                if token.swapcase().swapcase() == "OPEN":
                    return "hit"
                return "miss"
        """)
        corrs = harvest(p, _guard_line(p), {"token": "seed"}, {"token"})
        self.assertTrue(corrs)
        c = corrs[0]
        self.assertEqual((c.input_name, c.kind, c.proposal), ("token", "identity", "OPEN"))
        # Value-identity transform sits below a bare-name identity in confidence.
        self.assertLess(c.confidence, 0.95)

    def test_speculative_identity_when_types_line_up(self) -> None:
        # Transform is NOT value-identity on the seed ("seed".upper() != "seed"), but the
        # types match (str→str), so a low-confidence speculative copy is offered.  The
        # driver's execution check is what decides whether it was right.
        p = _write("""
            def f(token):
                if token.upper() == "LOUD":
                    return "hit"
                return "miss"
        """)
        corrs = harvest(p, _guard_line(p), {"token": "seed"}, {"token"})
        self.assertTrue(corrs)
        self.assertEqual(corrs[0].proposal, "LOUD")
        self.assertLessEqual(corrs[0].confidence, 0.4)


class TestHarvestOffset(unittest.TestCase):
    def test_add_offset(self) -> None:
        p = _write("""
            def f(n):
                if n + 4 == 100:
                    return "hit"
                return "miss"
        """)
        (c,) = harvest(p, _guard_line(p), {"n": 0}, {"n"})
        self.assertEqual((c.kind, c.demanded, c.proposal), ("offset", 100, 96))

    def test_sub_offset(self) -> None:
        p = _write("""
            def f(n):
                if n - 3 == 10:
                    return "hit"
                return "miss"
        """)
        (c,) = harvest(p, _guard_line(p), {"n": 0}, {"n"})
        self.assertEqual(c.proposal, 13)

    def test_reversed_add_offset(self) -> None:
        p = _write("""
            def f(n):
                if 5 + n == 20:
                    return "hit"
                return "miss"
        """)
        (c,) = harvest(p, _guard_line(p), {"n": 0}, {"n"})
        self.assertEqual((c.kind, c.proposal), ("offset", 15))


class TestHarvestSubstring(unittest.TestCase):
    def test_prefix_slice(self) -> None:
        p = _write("""
            def f(s):
                if s[:5] == "MAGIC":
                    return "hit"
                return "miss"
        """)
        (c,) = harvest(p, _guard_line(p), {"s": "zz"}, {"s"})
        self.assertEqual(c.kind, "substring")
        self.assertEqual(c.proposal[:5], "MAGIC")

    def test_mid_slice_splices_into_window(self) -> None:
        p = _write("""
            def f(s):
                if s[2:5] == "XYZ":
                    return "hit"
                return "miss"
        """)
        (c,) = harvest(p, _guard_line(p), {"s": "abcdefgh"}, {"s"})
        self.assertEqual(c.proposal[2:5], "XYZ")
        self.assertEqual(c.proposal[:2], "ab")   # bytes before the window are preserved


class TestHarvestNegatives(unittest.TestCase):
    def test_inequality_yields_no_correspondence(self) -> None:
        # RedQueen copies a single demanded value; an inequality names no single value.
        p = _write("""
            def f(x):
                if x > 100:
                    return "hit"
                return "miss"
        """)
        self.assertEqual(harvest(p, _guard_line(p), {"x": 0}, {"x"}), [])

    def test_both_sides_param_dependent_yields_nothing(self) -> None:
        # If the "demanded" side also depends on a parameter, the value it wants is itself
        # a moving target — no fixed value to copy.
        p = _write("""
            def f(a, b):
                if a == b:
                    return "hit"
                return "miss"
        """)
        self.assertEqual(harvest(p, _guard_line(p), {"a": 1, "b": 2}, {"a", "b"}), [])

    def test_unevaluable_operand_is_skipped_not_crashed(self) -> None:
        # cfg["mode"] on an empty dict raises KeyError; harvest must swallow it and, since
        # the transform can't be observed, offer nothing (a structural insert is minleaf's
        # job, not a value copy).
        p = _write("""
            def f(cfg):
                if cfg["mode"] == "fast":
                    return "hit"
                return "miss"
        """)
        self.assertEqual(harvest(p, _guard_line(p), {"cfg": {}}, {"cfg"}), [])


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Value-level correspondence: no syntactic link between operand and parameter
# ---------------------------------------------------------------------------


class TestValueLevelCorrespondence(unittest.TestCase):
    """RedQueen's actual mechanism: the link is an observed value, not a name.

    A guard like ``head == expected()`` where ``head = token[:5]`` ran earlier mentions
    no parameter at all, so a syntactic search finds nothing — yet at runtime ``head``
    holds the parameter's value. That is the correspondence, and finding it without
    knowing *how* the value got there is exactly what survives unmodeled code.
    """

    SRC = """
        def _expected():
            return "ALPHA"

        def check(token):
            head = token[:5]
            if head == _expected():
                return "HIT"
            return "miss"
    """

    def _setup(self):
        p = _write(self.SRC)
        tree = ast.parse(open(p).read())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "check")
        guard = next(n for n in ast.walk(fn) if isinstance(n, ast.If))
        return p, guard.lineno, guard.body[0].lineno

    def test_harvest_finds_local_carrying_the_param_value(self) -> None:
        from tharkun.redqueen import harvest
        p, guard_line, _ = self._setup()
        # 'head' is a local; it shares the parameter's value at the guard.
        env = {"token": "xxxxx", "head": "xxxxx", "_expected": lambda: "ALPHA"}
        corrs = harvest(p, guard_line, env, {"token"})
        self.assertTrue(corrs, "value-level correspondence should be found")
        self.assertEqual(corrs[0].input_name, "token")
        self.assertEqual(corrs[0].proposal, "ALPHA")

    def test_demanded_side_depending_on_the_param_is_rejected(self) -> None:
        # If the "demanded" side moves with the input it is not a fixed target to copy,
        # so no correspondence is claimed in either direction.
        from tharkun.redqueen import harvest
        p = _write("""
            def check(token):
                head = token[:5]
                if head == token.upper():
                    return "HIT"
                return "miss"
        """)
        tree = ast.parse(open(p).read())
        guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If))
        env = {"token": "xxxxx", "head": "xxxxx"}
        self.assertEqual(harvest(p, guard.lineno, env, {"token"}), [])


# ---------------------------------------------------------------------------
# Slice aliasing: a local that IS a window of a parameter
# ---------------------------------------------------------------------------


class TestSliceAliasing(unittest.TestCase):
    """Parsers split the input up front and guard on the pieces.

    ``body = data[4:]`` then ``body == expected()``: the guard mentions no parameter,
    and when ``body`` is empty its value carries no positional information either — so
    value-level matching cannot place it and the input can never be grown. Reading the
    assignment recovers the window, and splicing there may legitimately lengthen the
    input.
    """

    SRC = """
        def _head():
            return "ALPHA"

        def _tail(h):
            return h.lower() + "-omega"

        def check(token):
            head = token[:5]
            if head == _head():
                deeper = token[5:]
                if deeper == _tail(head):
                    return "DEEP"
                return "stage1"
            return "miss"
    """

    def _paths(self):
        p = _write(self.SRC)
        tree = ast.parse(open(p).read())
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == "check")
        ifs = [n for n in ast.walk(fn) if isinstance(n, ast.If)]
        return p, ifs[1].body[0].lineno

    def test_aliases_are_recovered(self) -> None:
        from tharkun.redqueen import slice_aliases
        p, _ = self._paths()
        al = slice_aliases(p, 8, {"token"})
        self.assertEqual(al["head"], ("token", None, 5))
        self.assertEqual(al["deeper"], ("token", 5, None))

    def test_reassigned_local_is_not_aliased(self) -> None:
        # An alias only holds if the local is bound once; a rebinding makes it a lie.
        from tharkun.redqueen import slice_aliases
        p = _write("""
            def f(data):
                part = data[:4]
                part = "clobbered"
                if part == "ABCD":
                    return 1
                return 0
        """)
        self.assertEqual(slice_aliases(p, 5, {"data"}), {})
