"""Tests for the Atheris FDP codec — validated against the real atheris decoder.

The whole point of the codec is byte-identity with Atheris, so these tests require
atheris and verify every re-encode by feeding it back through the real FDP.  They
skip cleanly where atheris is absent (tested with 3.1.0 on CPython 3.14.4).
"""
from __future__ import annotations

import sys
import unittest

from tharkun.hybrid.fdp_codec import (
    FieldRecord,
    atheris_available,
    encode_overrides,
    field_bounds,
    record_schedule,
    replay_values,
)

requires_atheris = unittest.skipUnless(atheris_available(), "atheris not installed")

if atheris_available():
    import atheris

    def _harness(data):
        fdp = atheris.FuzzedDataProvider(data)
        magic = fdp.ConsumeInt(4)
        name = fdp.ConsumeBytes(3)
        kind = fdp.ConsumeIntInRange(0, 10)
        flag = fdp.ConsumeBool()
        return magic, name, kind, flag

    SEED = b"\x01\x00\x00\x00abc\x05\x01"


@requires_atheris
class TestRecord(unittest.TestCase):
    def test_schedule_captures_every_consume(self) -> None:
        sched = record_schedule(_harness, SEED)
        self.assertEqual([r.method for r in sched],
                         ["ConsumeInt", "ConsumeBytes", "ConsumeIntInRange", "ConsumeBool"])

    def test_front_back_classification(self) -> None:
        sched = record_schedule(_harness, SEED)
        ends = {r.method: r.end for r in sched}
        self.assertEqual(ends["ConsumeInt"], "front")
        self.assertEqual(ends["ConsumeBytes"], "front")
        self.assertEqual(ends["ConsumeIntInRange"], "back")
        self.assertEqual(ends["ConsumeBool"], "back")

    def test_recorded_values_match_real_fdp(self) -> None:
        sched = record_schedule(_harness, SEED)
        self.assertEqual([r.value for r in sched], list(replay_values(SEED, sched)))


@requires_atheris
class TestEncodeRoundTrip(unittest.TestCase):
    def test_override_decodes_through_real_harness(self) -> None:
        sched = record_schedule(_harness, SEED)
        new = encode_overrides(SEED, sched, {0: 1000, 2: 7, 3: True})
        self.assertIsNotNone(new)
        magic, name, kind, flag = _harness(new)  # the REAL harness
        self.assertEqual(magic, 1000)
        self.assertEqual(kind, 7)
        self.assertTrue(flag)
        self.assertEqual(name, b"abc")           # untouched field preserved

    def test_int_override_little_endian(self) -> None:
        sched = record_schedule(_harness, SEED)
        new = encode_overrides(SEED, sched, {0: 0x04030201})
        self.assertEqual(_harness(new)[0], 0x04030201)

    def test_out_of_range_override_rejected(self) -> None:
        sched = record_schedule(_harness, SEED)
        self.assertIsNone(encode_overrides(SEED, sched, {2: 999}))  # kind is 0..10

    def test_variable_width_string_resizes_whole_buffer_field(self) -> None:
        # A whole-buffer string field (ConsumeUnicodeNoSurrogates(maxsize)) can be
        # re-encoded to a longer string — the buffer is rebuilt, verified vs real FDP.
        def uni_harness(data):
            fdp = atheris.FuzzedDataProvider(data)
            return fdp.ConsumeUnicodeNoSurrogates(sys.maxsize)
        seed = b"x"                          # decodes to a short/empty string
        sched = record_schedule(uni_harness, seed)
        new = encode_overrides(seed, sched, {0: "fd://5"})
        self.assertIsNotNone(new)
        self.assertEqual(uni_harness(new), "fd://5")

    def test_no_override_is_identity(self) -> None:
        sched = record_schedule(_harness, SEED)
        self.assertEqual(encode_overrides(SEED, sched, {}), SEED)


class TestFieldBounds(unittest.TestCase):
    """field_bounds is pure — runs without atheris."""

    def _rec(self, method, args, nbytes) -> FieldRecord:
        return FieldRecord(0, method, args, None, "front", nbytes, (0, nbytes))

    def test_uint_bounds(self) -> None:
        self.assertEqual(field_bounds(self._rec("ConsumeUInt", (4,), 4)), (0, 2**32 - 1))
        self.assertEqual(field_bounds(self._rec("ConsumeUInt", (2,), 2)), (0, 65535))

    def test_int_bounds_signed(self) -> None:
        self.assertEqual(field_bounds(self._rec("ConsumeInt", (4,), 4)), (-(2**31), 2**31 - 1))

    def test_intinrange_bounds(self) -> None:
        self.assertEqual(field_bounds(self._rec("ConsumeIntInRange", (0, 50), 1)), (0, 50))

    def test_bool_bounds(self) -> None:
        self.assertEqual(field_bounds(self._rec("ConsumeBool", (), 1)), (0, 1))

    def test_unbounded_returns_none(self) -> None:
        self.assertIsNone(field_bounds(self._rec("ConsumeUnicodeNoSurrogates", (3,), 4)))


@requires_atheris
class TestConsumeUIntIsFront(unittest.TestCase):
    def test_uint_consumes_from_front(self) -> None:
        def h(data):
            fdp = atheris.FuzzedDataProvider(data)
            return fdp.ConsumeBytes(2), fdp.ConsumeUInt(2)
        sched = record_schedule(h, bytes(range(8)))
        ends = {r.method: r.end for r in sched}
        self.assertEqual(ends["ConsumeUInt"], "front")


@requires_atheris
class TestRejectionLogging(unittest.TestCase):
    """A refusal must say *why*, and whether an encoder could fix it."""

    @staticmethod
    def _harness(data):
        fdp = atheris.FuzzedDataProvider(data)
        word = fdp.ConsumeUnicodeNoSurrogates(6)
        ratio = fdp.ConsumeProbability()
        n = fdp.ConsumeUInt(2)
        return word, ratio, n

    SEED = b"\x01hello\x40\x00\x00"

    def _reject(self, overrides):
        sched = record_schedule(self._harness, self.SEED)
        rej = []
        out = encode_overrides(self.SEED, sched, overrides, rej)
        self.assertIsNone(out)
        self.assertTrue(rej, "a refusal must record a reason")
        return rej[0]

    def test_surrogate_is_a_hard_limit(self) -> None:
        # ConsumeUnicodeNoSurrogates cannot produce one, so no encoding exists.
        r = self._reject({0: "\ud800abc"})
        self.assertFalse(r.mechanically_possible)
        self.assertIn("surrogate", r.reason)

    def test_range_violation_is_a_hard_limit(self) -> None:
        # The harness's own ConsumeIntInRange bounds exclude the value — no encoding helps.
        def h(data):
            fdp = atheris.FuzzedDataProvider(data)
            return fdp.ConsumeIntInRange(0, 10)
        sched = record_schedule(h, b"\x05")
        rej = []
        self.assertIsNone(encode_overrides(b"\x05", sched, {0: 999}, rej))
        self.assertFalse(rej[0].mechanically_possible)

    def test_no_rejections_list_still_works(self) -> None:
        # The parameter is optional; callers that don't care keep the old signature.
        # Use a genuinely impossible value (a surrogate) rather than a float, which is
        # now encodable.
        sched = record_schedule(self._harness, self.SEED)
        self.assertIsNone(encode_overrides(self.SEED, sched, {0: "\ud800x"}))


@requires_atheris
class TestStringEncoding(unittest.TestCase):
    @staticmethod
    def _str_harness(data):
        fdp = atheris.FuzzedDataProvider(data)
        word = fdp.ConsumeUnicodeNoSurrogates(3)
        n = fdp.ConsumeInt(4)
        return word, n

    SEED = b"\x01GET*\x00\x00\x00"

    def test_ascii_string_override_round_trips(self) -> None:
        sched = record_schedule(self._str_harness, self.SEED)
        new = encode_overrides(self.SEED, sched, {0: "PUT"})
        self.assertIsNotNone(new)
        word, n = self._str_harness(new)
        self.assertEqual(word, "PUT")
        self.assertEqual(n, 42)                  # untouched int field preserved

    def test_wrong_length_string_rejected(self) -> None:
        # Width is preserved; a differently-sized target does not fit the field.
        sched = record_schedule(self._str_harness, self.SEED)
        self.assertIsNone(encode_overrides(self.SEED, sched, {0: "HELLO"}))

    def test_non_latin1_string_rejected(self) -> None:
        sched = record_schedule(self._str_harness, self.SEED)
        self.assertIsNone(encode_overrides(self.SEED, sched, {0: "€uro"}))


@requires_atheris
class TestWideUnicodeEncoding(unittest.TestCase):
    """Beyond latin-1: the wide mode round-trips every scalar value.

    An even mode byte with (mode & 3) == 0 makes atheris read four bytes per codepoint,
    little-endian, masked to 21 bits — so CJK, emoji and the top of the range all encode.
    """

    @staticmethod
    def _harness(data):
        return atheris.FuzzedDataProvider(data).ConsumeUnicodeNoSurrogates(sys.maxsize)

    def _round_trip(self, target: str) -> None:
        seed = b"x"
        sched = record_schedule(self._harness, seed)
        out = encode_overrides(seed, sched, {0: target})
        self.assertIsNotNone(out, f"could not encode {target!r}")
        self.assertEqual(self._harness(out), target)

    def test_accented(self) -> None:
        self._round_trip("café")

    def test_emoji(self) -> None:
        self._round_trip("café☕")

    def test_cjk(self) -> None:
        self._round_trip("日本語")

    def test_top_of_range(self) -> None:
        self._round_trip("\U0010fff0")

    def test_ascii_still_uses_the_compact_mode(self) -> None:
        # ASCII must not regress to 4 bytes per character.
        seed = b"x"
        sched = record_schedule(self._harness, seed)
        out = encode_overrides(seed, sched, {0: "fd://5"})
        self.assertEqual(self._harness(out), "fd://5")
        self.assertEqual(len(out), 1 + len("fd://5"))   # 1:1 latin-1 packing


@requires_atheris
class TestFloatEncoding(unittest.TestCase):
    """ConsumeProbability / ConsumeFloatInRange are a little-endian u64 lerp."""

    @staticmethod
    def _prob_harness(data):
        fdp = atheris.FuzzedDataProvider(data)
        fdp.ConsumeBytes(4)                 # keep the back field 8 bytes wide
        return fdp.ConsumeProbability()

    @staticmethod
    def _range_harness(data):
        fdp = atheris.FuzzedDataProvider(data)
        fdp.ConsumeBytes(4)
        return fdp.ConsumeFloatInRange(0.0, 100.0)

    def _rt(self, harness, value, idx=1):
        seed = b"abcd" + b"\x00" * 8
        sched = record_schedule(harness, seed)
        out = encode_overrides(seed, sched, {idx: value})
        self.assertIsNotNone(out, f"could not encode {value!r}")
        self.assertAlmostEqual(harness(out), value, places=12)

    def test_probability_bounds_and_middle(self) -> None:
        for v in (0.0, 0.25, 0.5, 0.75, 1.0):
            self._rt(self._prob_harness, v)

    def test_float_in_range(self) -> None:
        for v in (0.0, 25.0, 42.5, 100.0):
            self._rt(self._range_harness, v)

    def test_value_outside_the_fields_range_is_a_hard_limit(self) -> None:
        seed = b"abcd" + b"\x00" * 8
        sched = record_schedule(self._range_harness, seed)
        rej = []
        self.assertIsNone(encode_overrides(seed, sched, {1: 500.0}, rej))
        self.assertFalse(rej[0].mechanically_possible)

    def test_back_field_grows_when_the_seed_is_too_narrow(self) -> None:
        # The back region is rebuilt from its field bytes, so a back-consumed field can
        # widen (here 1 byte -> the 8 a float needs) instead of refusing.
        seed = b"abcd\x00"          # only one spare byte for the back field
        sched = record_schedule(self._prob_harness, seed)
        out = encode_overrides(seed, sched, {1: 0.75})
        self.assertIsNotNone(out)
        self.assertAlmostEqual(self._prob_harness(out), 0.75, places=12)
        self.assertGreater(len(out), len(seed))       # the buffer grew at the back


@requires_atheris
class TestMultipleProviders(unittest.TestCase):
    """A harness may build several FuzzedDataProviders over the SAME buffer.

    The real ntlm_auth OSS-Fuzz harness builds three. Each one restarts at byte 0, so
    their fields overlap — a shared, partially-consumed recorder reported wrong values
    and wrong byte ranges, and replay_values shared the blind spot so verification
    could not catch it.
    """

    @staticmethod
    def _two_providers(data):
        a = atheris.FuzzedDataProvider(data).ConsumeUInt(4)
        b = atheris.FuzzedDataProvider(data).ConsumeUInt(4)
        return a, b

    SEED = bytes([1, 0, 0, 0, 2, 0, 0, 0])

    def test_each_provider_restarts_at_byte_zero(self) -> None:
        sched = record_schedule(self._two_providers, self.SEED)
        self.assertEqual(len(sched), 2)
        # Ground truth: both read the same first four bytes.
        self.assertEqual(self._two_providers(self.SEED), (1, 1))
        self.assertEqual([r.value for r in sched], [1, 1])
        self.assertEqual([r.byte_range for r in sched], [(0, 4), (0, 4)])

    def test_records_are_tagged_by_provider(self) -> None:
        sched = record_schedule(self._two_providers, self.SEED)
        self.assertEqual([r.provider for r in sched], [0, 1])

    def test_replay_matches_the_real_harness(self) -> None:
        sched = record_schedule(self._two_providers, self.SEED)
        self.assertEqual(replay_values(self.SEED, sched), [1, 1])

    def test_overlapping_resize_targets_the_owning_provider(self) -> None:
        # Two providers read the same bytes. A resize is laid out for the provider that
        # owns the changed field; the others simply read different values, which is a
        # different input rather than a false one. Only the targeted field must hold.
        def h(data):
            a = atheris.FuzzedDataProvider(data).ConsumeUnicodeNoSurrogates(sys.maxsize)
            b = atheris.FuzzedDataProvider(data).ConsumeUInt(2)
            return a, b
        seed = b"\x01ab"
        sched = record_schedule(h, seed)
        out = encode_overrides(seed, sched, {0: "abcdefgh"}, verify_only_targeted=True)
        self.assertIsNotNone(out)
        self.assertEqual(h(out)[0], "abcdefgh")     # the field we asked for holds

    def test_overlapping_resize_still_strict_by_default(self) -> None:
        # Without verify_only_targeted the strict contract stands: every field must
        # decode as recorded, which an overlapping resize cannot guarantee.
        def h(data):
            a = atheris.FuzzedDataProvider(data).ConsumeUnicodeNoSurrogates(sys.maxsize)
            b = atheris.FuzzedDataProvider(data).ConsumeUInt(2)
            return a, b
        seed = b"\x01ab"
        sched = record_schedule(h, seed)
        # "z..." changes byte 1, which provider 1's ConsumeUInt(2) also reads, so the
        # strict contract (every recorded field must still decode as before) is violated.
        self.assertIsNone(encode_overrides(seed, sched, {0: "zbcdefgh"}))


@requires_atheris
class TestOracleGuidedFloatInversion(unittest.TestCase):
    """ConsumeFloat/ConsumeRegularFloat are inverted by querying the real provider.

    They are not a single lerp — trailing bytes select a range and the leading eight
    interpolate within it, with special cases at the extremes. Rather than hard-code a
    piecewise mapping that could be wrong (and would rot on an atheris upgrade), the
    encoder brackets and bisects using the actual decoder.
    """

    @staticmethod
    def _harness(data):
        fdp = atheris.FuzzedDataProvider(data)
        fdp.ConsumeBytes(4)
        return fdp.ConsumeRegularFloat()

    SEED = b"abcd" + b"\x00" * 9

    def test_reachable_value_round_trips(self) -> None:
        sched = record_schedule(self._harness, self.SEED)
        out = encode_overrides(self.SEED, sched, {1: 0.0})
        self.assertIsNotNone(out)
        self.assertEqual(self._harness(out), 0.0)

    def test_identity_target_round_trips(self) -> None:
        sched = record_schedule(self._harness, self.SEED)
        base = self._harness(self.SEED)
        out = encode_overrides(self.SEED, sched, {1: base})
        self.assertIsNotNone(out)
        self.assertEqual(self._harness(out), base)

    def test_far_unreachable_value_is_refused_with_the_nearest(self) -> None:
        # The field produces 2**64 points spread across the double range, so the gaps at
        # small magnitudes are enormous. 42.5 has no near neighbour — refuse rather than
        # substitute something absurd (its true "nearest" is 0.0).
        sched = record_schedule(self._harness, self.SEED)
        rej = []
        self.assertIsNone(encode_overrides(self.SEED, sched, {1: 42.5}, rej))
        self.assertIn("cannot represent", rej[0].reason)
        self.assertIn("nearest", rej[0].reason)

    def test_close_value_is_substituted_and_recorded(self) -> None:
        # Within tolerance the nearest reachable value is used instead of refusing —
        # sound because the caller verifies by running the guard, not by equality — and
        # the substitution is always recorded.
        sched = record_schedule(self._harness, self.SEED)
        rej = []
        out = encode_overrides(self.SEED, sched, {1: 1e300}, rej)
        self.assertIsNotNone(out)
        self.assertTrue(rej and "substituted" in rej[0].reason)
        self.assertAlmostEqual(self._harness(out) / 1e300, 1.0, places=5)


@requires_atheris
class TestRecorderDelegatesEveryReader(unittest.TestCase):
    """An unlisted Consume* method must not silently truncate the schedule."""

    def test_unlisted_reader_is_recorded(self) -> None:
        def h(data):
            fdp = atheris.FuzzedDataProvider(data)
            return fdp.ConsumeBytes(2), fdp.ConsumeRegularFloat()
        sched = record_schedule(h, b"ab" + b"\x00" * 9)
        # Before generic delegation this raised inside the harness, record_schedule
        # swallowed it, and the float field vanished from the schedule entirely.
        self.assertEqual([r.method for r in sched], ["ConsumeBytes", "ConsumeRegularFloat"])

    def test_override_of_an_absent_field_is_not_silent_success(self) -> None:
        def h(data):
            return atheris.FuzzedDataProvider(data).ConsumeBytes(2)
        sched = record_schedule(h, b"ab")
        rej = []
        self.assertIsNone(encode_overrides(b"ab", sched, {5: 1}, rej))
        self.assertIn("no recorded field", rej[0].reason)


@requires_atheris
class TestTruncationIsReported(unittest.TestCase):
    def test_harness_exception_is_recorded(self) -> None:
        # A schedule cut short by a crash looks identical to a legitimately short one;
        # acting on it models the harness wrongly, so the fact is recorded.
        def boom(data):
            fdp = atheris.FuzzedDataProvider(data)
            fdp.ConsumeBytes(2)
            raise ValueError("harness blew up")
        record_schedule(boom, b"ab")
        self.assertIsInstance(record_schedule.last_exception, ValueError)

    def test_clean_run_records_no_exception(self) -> None:
        def fine(data):
            return atheris.FuzzedDataProvider(data).ConsumeBytes(2)
        record_schedule(fine, b"ab")
        self.assertIsNone(record_schedule.last_exception)


class TestWidenSeed(unittest.TestCase):
    """A seed too small for the harness starves its later fields.

    An FDP string field is greedy — `ConsumeUnicodeNoSurrogates(256)` takes everything
    until its own count binds — so on a short seed the first field swallows the buffer
    and every later one consumes zero bytes. A zero-width field cannot be written to at
    all, which reads as a codec gap but is really a seed-size problem.
    """

    @staticmethod
    def _harness(data):
        import atheris
        fdp = atheris.FuzzedDataProvider(data)
        return (fdp.ConsumeUnicodeNoSurrogates(8),
                fdp.ConsumeUnicodeNoSurrogates(8),
                fdp.ConsumeUnicodeNoSurrogates(8))

    @requires_atheris
    def test_short_seed_starves_later_fields(self) -> None:
        from tharkun.hybrid.fdp_codec import record_schedule

        sched = record_schedule(self._harness, b"\x01ab")
        self.assertGreater(sched[0].nbytes, 0)
        self.assertEqual([r.nbytes for r in sched[1:]], [0, 0])

    @requires_atheris
    def test_widening_binds_the_starved_fields(self) -> None:
        from tharkun.hybrid.fdp_codec import record_schedule, widen_seed

        grown = widen_seed(self._harness, b"\x01ab")
        sched = record_schedule(self._harness, grown)
        self.assertTrue(all(r.nbytes > 0 for r in sched),
                        f"still starved: {[r.nbytes for r in sched]}")

    @requires_atheris
    def test_already_saturated_seed_is_returned_unchanged(self) -> None:
        from tharkun.hybrid.fdp_codec import widen_seed

        data = bytes(range(1, 200))
        self.assertEqual(widen_seed(data=data, harness_fn=self._harness), data)

    @requires_atheris
    def test_widening_is_deterministic(self) -> None:
        from tharkun.hybrid.fdp_codec import widen_seed

        self.assertEqual(widen_seed(self._harness, b"\x01ab"),
                         widen_seed(self._harness, b"\x01ab"))

    @requires_atheris
    def test_respects_the_size_cap(self) -> None:
        from tharkun.hybrid.fdp_codec import widen_seed

        # A harness no amount of widening can saturate: the last field is unreachable
        # because the greedy one before it has no count limit to bind.
        def greedy(data):
            import atheris
            fdp = atheris.FuzzedDataProvider(data)
            return fdp.ConsumeBytes(1 << 30), fdp.ConsumeBytes(4)

        self.assertLessEqual(len(widen_seed(greedy, b"\x00" * 8, max_size=512)), 512)


@requires_atheris
class TestExecutionVerifiedRecode(unittest.TestCase):
    """encode_overrides' `accept` hook: a recode whose mid-buffer field re-decodes to the
    solved value PLUS trailing bytes is rejected by the strict exact-match contract, but
    accepted when the caller execution-verifies the seed still reaches the target — the
    fix that lets a solved "fd://" survive as a field the following fields spill into."""

    @staticmethod
    def _two_strings(data):
        fdp = atheris.FuzzedDataProvider(data)
        fdp.ConsumeUnicodeNoSurrogates(20)
        fdp.ConsumeUnicodeNoSurrogates(20)

    SEED = b"\x01hello world padding here xx" + b"\x01second field data here yy"

    def _sched(self):
        return record_schedule(self._two_strings, self.SEED)

    def test_strict_rejects_prefix_recode(self) -> None:
        # A short override into a mid-buffer string field re-decodes as "fd://<leftover>",
        # which is not an exact match — the default (accept=None) contract rejects it.
        sched = self._sched()
        self.assertIsNone(
            encode_overrides(self.SEED, sched, {0: "fd://"}, verify_only_targeted=True))

    def test_accept_takes_a_prefix_satisfying_recode(self) -> None:
        sched = self._sched()
        taken = encode_overrides(self.SEED, sched, {0: "fd://"},
                                 verify_only_targeted=True, accept=lambda nb: True)
        self.assertIsNotNone(taken)
        # what it actually re-decodes to still satisfies startswith — the whole point.
        self.assertTrue(replay_values(taken, sched)[0].startswith("fd://"))

    def test_accept_false_still_rejects(self) -> None:
        # accept only ever ACCEPTS what it approves; a False verdict keeps the strict result.
        sched = self._sched()
        self.assertIsNone(
            encode_overrides(self.SEED, sched, {0: "fd://"},
                             verify_only_targeted=True, accept=lambda nb: False))


if __name__ == "__main__":
    unittest.main()
