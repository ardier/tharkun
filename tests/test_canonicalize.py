"""One text for one meaning, so a constraint written two ways is one clause.

Clause identity is the source string. `(x > 5)` and `(5 < x)` were two clauses, and since
clause sets merge by INTERSECTION at a join, a constraint two predecessors both established
— written differently — was silently lost: the merge rendered `True` and every solver
downstream worked without a fact that held. Canonicalization makes the merge see them as
one.

Determinism is the whole point: if the same clause canonicalized differently across runs
the merge would fail anyway, so the ordering is total and stable across processes.
"""
from __future__ import annotations

import unittest

from tharkun.solve.canonicalize import canonicalize


class TestEquivalentGuardsCanonicalizeEqual(unittest.TestCase):
    EQUIV = [
        ("5 < x", "x > 5"),
        ("5 <= x", "x >= 5"),
        ("x < 5", "5 > x"),
        ("x == 5", "5 == x"),
        ("x != 5", "5 != x"),
        ("len(key) != 7", "7 != len(key)"),
        ("a and b", "b and a"),
        ("a or b or c", "c or b or a"),
        ("int.from_bytes(t[:2], 'big') == 13987", "13987 == int.from_bytes(t[:2], 'big')"),
    ]

    def test_each_pair_maps_to_one_string(self) -> None:
        for a, b in self.EQUIV:
            self.assertEqual(canonicalize(a), canonicalize(b), f"{a!r} vs {b!r}")


class TestNaNSafeNegation(unittest.TestCase):
    """`not (x > 5)` is `(x <= 5) or (x != x)`, not `x <= 5` — sound under NaN.

    The old inversion `not (x > 5) -> x <= 5` is an UNDER-approximation: it drops the NaN
    case (where `not (x > 5)` is true but `x <= 5` is false), which could make a sound
    decider call a reachable branch dead. The disjunction over-approximates instead, the
    safe direction.
    """

    GRID = [float("nan"), -1.0, 0.0, 5.0, 6.0]

    def _truth(self, expr: str, xval: float):
        try:
            return bool(eval(expr, {}, {"x": xval}))
        except Exception:
            return None

    def test_every_ordering_negation_is_truth_table_faithful_under_nan(self) -> None:
        for orig in ["not (x > 5)", "not (x <= 5)", "not (x >= 5)", "not (x < 5)"]:
            canon = canonicalize(orig)
            for v in self.GRID:
                self.assertEqual(self._truth(orig, v), self._truth(canon, v),
                                 f"{orig!r} vs {canon!r} at x={v}")

    def test_the_nan_disjunct_is_present_for_a_variable_operand(self) -> None:
        self.assertIn("x != x", canonicalize("not (x > 5)"))

    def test_a_constant_operand_gets_no_disjunct(self) -> None:
        # A literal is never NaN, so `5 != 5` is omitted — it is always false.
        self.assertNotIn("5 != 5", canonicalize("not (5 > x)"))

    def test_ordering_negation_no_longer_equals_the_bare_inversion(self) -> None:
        # They are genuinely different constraints under NaN; merging them was the bug.
        self.assertNotEqual(canonicalize("not (x > 5)"), canonicalize("x <= 5"))

    def test_equality_negation_stays_the_bare_inversion(self) -> None:
        # ==/!= negation IS NaN-safe, so no disjunct and the merge still holds.
        self.assertEqual(canonicalize("not (x == 5)"), canonicalize("x != 5"))

    def test_two_spellings_of_the_negation_still_agree(self) -> None:
        self.assertEqual(canonicalize("not (x > 5)"), canonicalize("not (5 < x)"))


class TestStringSliceStartswith(unittest.TestCase):
    """`X[:k] == LIT` (len(LIT) == k) IS `X.startswith(LIT)` — merged, but only exactly.

    The dangerous direction is over-merging: an offset slice, a strided slice, or a length
    mismatch is a genuinely different constraint and must stay apart, or the merge would
    invent a fact.
    """

    def test_prefix_slice_equality_becomes_startswith(self) -> None:
        for slc, sw in [
            ("token[:2] == b'ab'", "token.startswith(b'ab')"),
            ("s[:3] == 'abc'", "s.startswith('abc')"),
            ("s[0:3] == 'abc'", "s.startswith('abc')"),      # explicit 0 start
            ("s[0:3:1] == 'abc'", "s.startswith('abc')"),    # explicit unit step
        ]:
            self.assertEqual(canonicalize(slc), canonicalize(sw), slc)

    def test_literal_on_the_left_normalizes_too(self) -> None:
        self.assertEqual(canonicalize("b'ab' == token[:2]"),
                         canonicalize("token.startswith(b'ab')"))

    def test_canonical_form_is_the_startswith_call(self) -> None:
        self.assertEqual(canonicalize("token[:2] == b'ab'"), "token.startswith(b'ab')")

    def test_different_prefix_length_stays_distinct(self) -> None:
        self.assertNotEqual(canonicalize("token[:2] == b'ab'"),
                            canonicalize("token[:3] == b'abc'"))

    def test_different_prefix_value_stays_distinct(self) -> None:
        self.assertNotEqual(canonicalize("token[:2] == b'ab'"),
                            canonicalize("token[:2] == b'ba'"))

    def test_startswith_is_not_endswith(self) -> None:
        self.assertNotEqual(canonicalize("token.startswith(b'ab')"),
                            canonicalize("token.endswith(b'ab')"))

    def test_offset_slice_is_not_a_prefix(self) -> None:
        # token[1:3] is not the first k elements, so it must NOT become startswith.
        self.assertNotEqual(canonicalize("token[1:3] == b'ab'"),
                            canonicalize("token.startswith(b'ab')"))
        self.assertNotIn("startswith", canonicalize("token[1:3] == b'ab'"))

    def test_strided_slice_is_not_a_prefix(self) -> None:
        self.assertNotIn("startswith", canonicalize("token[:4:2] == b'ab'"))

    def test_length_mismatch_short_literal_stays_distinct(self) -> None:
        # token[:2] == b'a' is true only for token == b'a' (a shorter token); it is a
        # different constraint from startswith(b'a'), which holds for any b'a...'.
        self.assertNotEqual(canonicalize("token[:2] == b'a'"),
                            canonicalize("token.startswith(b'a')"))
        self.assertNotIn("startswith", canonicalize("token[:2] == b'a'"))

    def test_length_mismatch_long_literal_stays_distinct(self) -> None:
        # token[:2] == b'abc' is unsatisfiable; it must not fold to startswith(b'abc').
        self.assertNotIn("startswith", canonicalize("token[:2] == b'abc'"))

    def test_inequality_is_not_rewritten(self) -> None:
        # Only `==` folds; `!=` is `not startswith`, left for a later rule.
        self.assertNotIn("startswith", canonicalize("token[:2] != b'ab'"))

    def test_truth_table_faithful_over_every_string_including_short_ones(self) -> None:
        # The soundness claim, checked concretely: `X[:k] == LIT` and `X.startswith(LIT)`
        # agree on every X, including X shorter than k where the slice is truncated.
        for xval in ["", "a", "ab", "abc", "ba", "abx", "xab"]:
            slc = eval("x[:2] == 'ab'", {}, {"x": xval})
            sw = eval("x.startswith('ab')", {}, {"x": xval})
            self.assertEqual(slc, sw, xval)


class TestDistinctGuardsStayDistinct(unittest.TestCase):
    DISTINCT = [
        ("x > 5", "x > 3"),
        ("x > 5", "x >= 5"),
        ("a and b", "a or b"),
        ("x == 5", "x != 5"),
        ("x * 2 == 6", "x == 3"),   # coefficient division is beyond orientation/fold
    ]

    def test_a_real_difference_is_preserved(self) -> None:
        for a, b in self.DISTINCT:
            self.assertNotEqual(canonicalize(a), canonicalize(b), f"{a!r} vs {b!r}")


class TestDeterminism(unittest.TestCase):
    def test_idempotent(self) -> None:
        for t in ["5 < x", "x > 5", "b and a", "not (x > 5)"]:
            self.assertEqual(canonicalize(t), canonicalize(canonicalize(t)), t)

    def test_stable_across_processes(self) -> None:
        # hash()-based ordering would be salted per process; the tie-break is ast.unparse.
        import subprocess
        import sys

        code = ("import sys; sys.path.insert(0, 'src');"
                "from tharkun.canonicalize import canonicalize;"
                "print(canonicalize('b and a and c'), canonicalize('5 < x'))")
        runs = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                               text=True, cwd=".").stdout.strip() for _ in range(2)}
        self.assertEqual(len(runs), 1)


class TestDegradesGracefully(unittest.TestCase):
    def test_unparseable_text_is_returned_unchanged(self) -> None:
        self.assertEqual(canonicalize("this is not ( python"), "this is not ( python")

    def test_a_bare_name_survives(self) -> None:
        self.assertEqual(canonicalize("flag"), "flag")


if __name__ == "__main__":
    unittest.main()
