"""Length guards are decided by a field's WIDTH, which is not a value.

`if len(key) != 7` on `lm_hash[:7]` is satisfied for free when lm_hash is starved to
b'', and unsatisfiable while it is 124 bytes. No flipper can express that, because width
follows from how much buffer is left when the harness reaches the field — not from
anything that can be written *into* the field. So the requirement is read off the guard
and a base with that width is built, with the real provider as the oracle.

Found by hand first: encoding negotiate_flags=128 into the original 64-byte ntlm_auth
seed reaches des.py:212-213, which the engine had been calling a plateau. This is that
step made mechanical.
"""
from __future__ import annotations

import os
import unittest

from tharkun.hybrid.assist import length_requirements
from tharkun.hybrid.fdp_codec import atheris_available, record_schedule, seed_with_field_width

requires_atheris = unittest.skipUnless(atheris_available(), "atheris not installed")

FIXDIR = os.path.join(os.path.dirname(__file__), "fixtures")


def _write(tmp: str, body: str) -> str:
    path = os.path.join(tmp, "lenmod.py")
    with open(path, "w") as f:
        f.write(body)
    return path


SRC = '''
def check(key, data):
    if len(key) != 7:
        return "not-7"
    if len(data) > 4:
        return "long"
    if len(key) == 3:
        return "three"
    return "ok"
'''


class TestLengthRequirementInference(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.mod = _write(self.tmp, SRC)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_not_equal_wants_any_other_width(self) -> None:
        # len(key) != 7 -> 0 is the easiest width to arrange, and always available.
        self.assertEqual(length_requirements(self.mod, 3, "then"), {"key": 0})

    def test_not_equal_else_side_wants_the_literal(self) -> None:
        self.assertEqual(length_requirements(self.mod, 3, "else"), {"key": 7})

    def test_greater_than_wants_one_more(self) -> None:
        self.assertEqual(length_requirements(self.mod, 5, "then"), {"data": 5})

    def test_greater_than_else_side_wants_one_less(self) -> None:
        self.assertEqual(length_requirements(self.mod, 5, "else"), {"data": 4})

    def test_equality_wants_the_literal(self) -> None:
        self.assertEqual(length_requirements(self.mod, 7, "then"), {"key": 3})

    def test_no_length_test_yields_nothing(self) -> None:
        other = _write(self.tmp, "def f(x):\n    if x > 3:\n        return 1\n    return 0\n")
        self.assertEqual(length_requirements(other, 2, "then"), {})

    def test_slice_resolves_to_its_base_name(self) -> None:
        # The guard reads len(key) but the caller passed lm_hash[:7]; it is the BASE
        # whose width has to change, so a slice must not hide it.
        sliced = _write(self.tmp, "def f(buf):\n    if len(buf[:7]) != 7:\n        return 1\n    return 0\n")
        self.assertEqual(length_requirements(sliced, 2, "then"), {"buf": 0})


@requires_atheris
class TestSeedWithFieldWidth(unittest.TestCase):
    """The acting half: build a buffer whose field really has the wanted width."""

    def setUp(self) -> None:
        import sys
        sys.path.insert(0, FIXDIR)
        import xmod_harness
        self.harness = xmod_harness.TestOneInput

    def _width(self, buf: bytes, idx: int) -> int:
        sched = record_schedule(self.harness, buf)
        return sched[idx].nbytes if idx < len(sched) else -1

    def test_builds_a_buffer_that_starves_a_field(self) -> None:
        built = seed_with_field_width(self.harness, b"\xAA" * 512, field_index=3, want=0)
        self.assertIsNotNone(built)
        self.assertEqual(self._width(built, 3), 0)

    def test_builds_a_buffer_where_a_field_binds(self) -> None:
        built = seed_with_field_width(self.harness, b"\xAA" * 16, field_index=3, want=8)
        self.assertIsNotNone(built)
        self.assertGreater(self._width(built, 3), 0)

    def test_result_is_a_real_input_the_harness_accepts(self) -> None:
        built = seed_with_field_width(self.harness, b"\xAA" * 256, field_index=1, want=0)
        self.assertIsNotNone(built)
        self.harness(built)          # must not raise: it is an ordinary fuzzer input

    def test_out_of_range_field_still_returns_something_usable(self) -> None:
        built = seed_with_field_width(self.harness, b"\xAA" * 64, field_index=99, want=4)
        self.assertTrue(built is None or isinstance(built, bytes))


if __name__ == "__main__":
    unittest.main()
