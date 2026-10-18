"""Constraint dicts to strategies — narrow constructively, do not filter.

The driver's search tiers built strategies from a type name alone, so everything the
static solver had already established — the bounds a comparison implies, the literal set
a membership test names, the mask a bitwise guard demands — was discarded and
rediscovered by sampling. A guard saying `x > 1000 and x < 1010` deserves
st.integers(1001, 1009), not st.integers() filtered nine million ways.

Two cases are the pattern worth keeping: a CONTIGUOUS bitmask needs no filter at all
(every value in [0, mask] already satisfies it), and a left shift is enumerated over the
quotient so every draw is a multiple by construction rather than by rejection.
"""
from __future__ import annotations

import unittest

from tharkun.strategies import (
    UnsatisfiableConstraints,
    make_strategy,
    strategy_for_type,
)

try:
    import hypothesis  # noqa: F401
    HAVE = True
except ImportError:                                   # pragma: no cover
    HAVE = False

requires_hypothesis = unittest.skipUnless(HAVE, "hypothesis not installed")


def draw(strategy, n: int = 30) -> list:
    """Concrete values from a strategy, for asserting on what it can actually produce."""
    return [strategy.example() for _ in range(n)]


@requires_hypothesis
class TestIntNarrowing(unittest.TestCase):
    def test_range_is_respected(self) -> None:
        got = draw(make_strategy({"type": "int", "value_range": [1001, 1009]}))
        self.assertTrue(all(1001 <= v <= 1009 for v in got), got)

    def test_membership_dominates_and_only_yields_named_values(self) -> None:
        got = draw(make_strategy({"type": "int", "values": [3, 7, 11]}))
        self.assertTrue(set(got) <= {3, 7, 11}, got)

    def test_membership_beats_a_contradicting_range(self) -> None:
        # A named set is the sharpest fact available; nothing narrows it further.
        got = draw(make_strategy({"type": "int", "values": [5], "value_range": [900, 1000]}))
        self.assertEqual(set(got), {5})

    def test_one_open_side_still_bounds_the_other(self) -> None:
        got = draw(make_strategy({"type": "int", "value_range": [None, -5]}))
        self.assertTrue(all(v <= -5 for v in got), got)

    def test_booleans_are_not_ints_here(self) -> None:
        # bool is an int subclass; letting it through produces True where 1 was meant.
        with self.assertRaises(UnsatisfiableConstraints):
            make_strategy({"type": "int", "values": [True, False]})

    def test_an_empty_range_is_raised_not_returned(self) -> None:
        # An empty domain is a fact worth surfacing, not a search to run.
        with self.assertRaises(UnsatisfiableConstraints):
            make_strategy({"type": "int", "value_range": [10, 5]})


@requires_hypothesis
class TestBitmask(unittest.TestCase):
    def test_only_masked_bits_are_ever_set(self) -> None:
        mask = 0b1010
        got = draw(make_strategy({"type": "int", "bitmask_mask": mask}))
        self.assertTrue(all((v & ~mask) == 0 for v in got), got)

    def test_a_contiguous_mask_needs_no_filter(self) -> None:
        # 0b111: every value in [0, 7] already satisfies it, so filtering would reject
        # nothing and cost everything. The strategy must be the bare bounded integers.
        strat = make_strategy({"type": "int", "bitmask_mask": 0b111})
        self.assertNotIn("filter", repr(strat).lower())
        self.assertTrue(all(0 <= v <= 7 for v in draw(strat)))

    def test_a_sparse_mask_still_only_yields_valid_values(self) -> None:
        mask = 0b100001
        self.assertTrue(all((v & ~mask) == 0 for v in draw(make_strategy(
            {"type": "int", "bitmask_mask": mask}))))

    def test_a_zero_mask_admits_only_zero(self) -> None:
        self.assertEqual(set(draw(make_strategy({"type": "int", "bitmask_mask": 0}))), {0})

    def test_a_mask_that_excludes_the_range_is_unsatisfiable(self) -> None:
        with self.assertRaises(UnsatisfiableConstraints):
            make_strategy({"type": "int", "bitmask_mask": 0b11, "value_range": [100, 200]})


@requires_hypothesis
class TestShift(unittest.TestCase):
    def test_left_shift_yields_multiples_by_construction(self) -> None:
        # Enumerating the quotient and multiplying back means every draw is valid;
        # drawing integers and filtering for multiples would throw most of them away.
        got = draw(make_strategy({"type": "int", "shift_direction": "left",
                                  "shift_bits": 3, "value_range": [0, 1000]}))
        self.assertTrue(all(v % 8 == 0 for v in got), got)

    def test_right_shift_recovers_nothing_and_says_so(self) -> None:
        # The low bits are gone, so the source is unconstrained within the range.
        got = draw(make_strategy({"type": "int", "shift_direction": "right",
                                  "shift_bits": 3, "value_range": [0, 50]}))
        self.assertTrue(all(0 <= v <= 50 for v in got))

    def test_a_step_with_no_multiple_in_range_is_unsatisfiable(self) -> None:
        with self.assertRaises(UnsatisfiableConstraints):
            make_strategy({"type": "int", "shift_direction": "left",
                           "shift_bits": 8, "value_range": [1, 100]})


@requires_hypothesis
class TestStringsAndBytes(unittest.TestCase):
    def test_regex_samples_the_language_rather_than_filtering_for_it(self) -> None:
        # No amount of sampling st.text() produces this; the pattern must be sampled.
        import re

        pat = r"^[0-9a-f]{8}:[0-9a-f]{8}$"
        got = draw(make_strategy({"type": "str", "regex": pat}), n=10)
        self.assertTrue(all(re.fullmatch(pat, v) for v in got), got)

    def test_length_bounds_are_respected(self) -> None:
        got = draw(make_strategy({"type": "str", "length": {"min": 4, "max": 6}}))
        self.assertTrue(all(4 <= len(v) <= 6 for v in got), got)

    def test_bytes_length_bounds(self) -> None:
        got = draw(make_strategy({"type": "bytes", "length": {"min": 2, "max": 2}}))
        self.assertTrue(all(len(v) == 2 and isinstance(v, bytes) for v in got))

    def test_string_membership(self) -> None:
        got = draw(make_strategy({"type": "str", "values": ["GET", "POST"]}))
        self.assertTrue(set(got) <= {"GET", "POST"})

    def test_contradictory_length_is_unsatisfiable(self) -> None:
        with self.assertRaises(UnsatisfiableConstraints):
            make_strategy({"type": "str", "length": {"min": 9, "max": 2}})


@requires_hypothesis
class TestStructured(unittest.TestCase):
    def test_element_constraints_recurse(self) -> None:
        got = draw(make_strategy({
            "type": "list", "length": {"min": 2, "max": 3},
            "element_constraints": {"type": "int", "value_range": [10, 12]},
        }))
        self.assertTrue(all(2 <= len(v) <= 3 and all(10 <= e <= 12 for e in v) for v in got), got)

    def test_tuple_is_a_bounded_length_not_a_fixed_arity(self) -> None:
        got = draw(make_strategy({"type": "tuple", "length": {"min": 1, "max": 2},
                                  "element_constraints": {"type": "int", "values": [5]}}))
        self.assertTrue(all(isinstance(v, tuple) and set(v) == {5} for v in got), got)

    def test_dict_key_and_value_constraints_recurse(self) -> None:
        got = draw(make_strategy({
            "type": "dict", "length": {"min": 1, "max": 2},
            "key_constraints": {"type": "str", "values": ["a", "b"]},
            "value_constraints": {"type": "int", "value_range": [0, 3]},
        }))
        for d in got:
            self.assertTrue(set(d) <= {"a", "b"})
            self.assertTrue(all(0 <= x <= 3 for x in d.values()))

    def test_an_unsatisfiable_element_propagates(self) -> None:
        with self.assertRaises(UnsatisfiableConstraints):
            make_strategy({"type": "list",
                           "element_constraints": {"type": "int", "value_range": [9, 1]}})


@requires_hypothesis
class TestFloats(unittest.TestCase):
    def test_bounds_are_respected_and_nan_excluded(self) -> None:
        import math

        got = draw(make_strategy({"type": "float", "value_range": [1.5, 9.5]}))
        self.assertTrue(all(1.5 <= v <= 9.5 and not math.isnan(v) for v in got), got)

    def test_an_open_side_is_bounded_rather_than_left_open(self) -> None:
        import math

        got = draw(make_strategy({"type": "float", "value_range": [0.0, None]}))
        self.assertTrue(all(math.isfinite(v) for v in got))


@requires_hypothesis
class TestSharedTypeFallback(unittest.TestCase):
    def test_configured_integer_range_is_respected(self) -> None:
        got = draw(strategy_for_type("int", int_range=(7, 9)))
        self.assertTrue(all(7 <= value <= 9 for value in got))

    def test_bool_is_not_accidentally_treated_as_integer(self) -> None:
        got = draw(strategy_for_type("bool"))
        self.assertTrue(all(type(value) is bool for value in got))

    def test_branch_adapter_container_shapes_remain_available(self) -> None:
        self.assertTrue(all(isinstance(value, set) for value in draw(strategy_for_type("set"))))
        self.assertTrue(all(isinstance(value, tuple) for value in draw(strategy_for_type("tuple"))))

    def test_unknown_custom_type_abstains(self) -> None:
        self.assertIsNone(strategy_for_type("package.Custom"))


class TestRejections(unittest.TestCase):
    """Bad input is refused loudly rather than producing a strategy that cannot help."""

    def test_unknown_type(self) -> None:
        with self.assertRaises(UnsatisfiableConstraints):
            make_strategy({"type": "complex"})

    def test_missing_type(self) -> None:
        with self.assertRaises(UnsatisfiableConstraints):
            make_strategy({"value_range": [1, 2]})

    def test_not_a_dict(self) -> None:
        with self.assertRaises(UnsatisfiableConstraints):
            make_strategy("int")


if __name__ == "__main__":
    unittest.main()
