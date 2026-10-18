"""Tests for byte-input value generation in condition_solver.

Covers the *forward* solving slice of the int.from_bytes seam: given a guard
that tests ``int.from_bytes(data[i:j], order)`` or ``data[k]`` byte indexing,
the solver must synthesize a concrete ``bytes`` value for ``data`` that
*actually satisfies* the guard. Each test executes the fixture function with
the generated input and checks the truthy branch is taken.

Out of scope (and NOT tested here): backward inversion of a deep constraint on
the lifted integer onto raw bytes — the SPEC defers expression inversion to
glamdring.
"""

from __future__ import annotations

import importlib.util
import os
import unittest

from tharkun.solve.condition import solve_condition

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "byte_input_guards.py")


def _line_of(text):
    with open(FIXTURE) as f:
        for i, line in enumerate(f, 1):
            if text in line:
                return i
    raise ValueError(f"text {text!r} not found in {FIXTURE}")


def _load_fixture_module():
    spec = importlib.util.spec_from_file_location("byte_input_guards", FIXTURE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_MOD = _load_fixture_module()


def _first_data(sol):
    """Return the ``data`` value from the first satisfying input, or None."""
    if not sol.satisfying_inputs:
        return None
    return sol.satisfying_inputs[0].get("data")


class TestByteIndexEquality(unittest.TestCase):
    """data[0] == 0x30 → bytes whose first byte is 0x30."""

    def test_asn1_tag_eq_is_bytes(self):
        line = _line_of("GUARD_index_eq")
        sol = solve_condition(FIXTURE, line)
        data = _first_data(sol)
        self.assertIsInstance(data, (bytes, bytearray))
        self.assertGreater(len(data), 0)
        self.assertEqual(data[0], 0x30)

    def test_asn1_tag_eq_takes_branch(self):
        line = _line_of("GUARD_index_eq")
        sol = solve_condition(FIXTURE, line)
        data = _first_data(sol)
        self.assertEqual(_MOD.asn1_tag(data), "sequence")


class TestByteIndexInequality(unittest.TestCase):
    """data[0] >= 0x20 → first byte >= 0x20."""

    def test_asn1_tag_ge_takes_branch(self):
        line = _line_of("GUARD_index_ge")
        sol = solve_condition(FIXTURE, line)
        data = _first_data(sol)
        self.assertIsInstance(data, (bytes, bytearray))
        self.assertEqual(_MOD.asn1_tag_ge(data), "constructed-ish")


class TestByteIndexCompound(unittest.TestCase):
    """data[3] == 0xFF and data[0] == 0x7F → both byte positions set."""

    def test_magic_index_and_takes_branch(self):
        line = _line_of("GUARD_index_and")
        sol = solve_condition(FIXTURE, line)
        data = _first_data(sol)
        self.assertIsInstance(data, (bytes, bytearray))
        self.assertGreaterEqual(len(data), 4)
        self.assertEqual(_MOD.magic_index(data), "magic")


class TestFromBytesGreater(unittest.TestCase):
    """int.from_bytes(data[0:2], 'big') > 1000 → bytes lifting to > 1000."""

    def test_length_prefix_big_is_bytes(self):
        line = _line_of("GUARD_fb_gt")
        sol = solve_condition(FIXTURE, line)
        data = _first_data(sol)
        self.assertIsInstance(data, (bytes, bytearray))

    def test_length_prefix_big_takes_branch(self):
        line = _line_of("GUARD_fb_gt")
        sol = solve_condition(FIXTURE, line)
        data = _first_data(sol)
        self.assertEqual(_MOD.length_prefix_big(data), "long")


class TestFromBytesEqual(unittest.TestCase):
    """int.from_bytes(data[2:4], 'big') == 256 → exact value at the slice."""

    def test_length_prefix_eq_takes_branch(self):
        line = _line_of("GUARD_fb_eq")
        sol = solve_condition(FIXTURE, line)
        data = _first_data(sol)
        self.assertIsInstance(data, (bytes, bytearray))
        self.assertEqual(_MOD.length_prefix_eq(data), "exact")


class TestFromBytesLittle(unittest.TestCase):
    """int.from_bytes(data[0:4], 'little') >= 65536 → little-endian aware."""

    def test_length_prefix_little_takes_branch(self):
        line = _line_of("GUARD_fb_little")
        sol = solve_condition(FIXTURE, line)
        data = _first_data(sol)
        self.assertIsInstance(data, (bytes, bytearray))
        self.assertEqual(_MOD.length_prefix_little(data), "wide")


class TestFromBytesKwarg(unittest.TestCase):
    """int.from_bytes(data[1:3], byteorder='big') < 5 → keyword byteorder."""

    def test_from_bytes_kwarg_takes_branch(self):
        line = _line_of("GUARD_fb_kwarg")
        sol = solve_condition(FIXTURE, line)
        data = _first_data(sol)
        self.assertIsInstance(data, (bytes, bytearray))
        self.assertEqual(_MOD.from_bytes_kwarg(data), "tiny")


class TestFromBytesSingleByte(unittest.TestCase):
    """int.from_bytes(data[0:1], 'big') == 9 → single-byte slice."""

    def test_from_bytes_single_byte_takes_branch(self):
        line = _line_of("GUARD_fb_single")
        sol = solve_condition(FIXTURE, line)
        data = _first_data(sol)
        self.assertIsInstance(data, (bytes, bytearray))
        self.assertEqual(_MOD.from_bytes_no_order(data), "nine")


class TestByteSynthesisSoundness(unittest.TestCase):
    """The synthesis must never report a `bytes` that fails the guard as satisfying.

    Regression for the adversarial-audit findings: contradictory atoms merged last-write-
    wins, signed decode treated as unsigned, negative-literal targets dropped, and a valid
    value discarded because the verifier lacked a builtin.
    """

    def _solve(self, guard):
        import tempfile

        src = f"def f(data):\n    if {guard}:\n        return True\n    return False\n"
        p = tempfile.mktemp(suffix=".py")
        with open(p, "w") as fh:
            fh.write(src)
        try:
            return solve_condition(p, 2)
        finally:
            os.remove(p)

    def _holds(self, guard, value):
        ns = {"int": int, "len": len, "hex": hex, "bin": bin}
        exec(f"def _g(data):\n    return bool({guard})", ns)
        try:
            return bool(ns["_g"](value))
        except Exception:
            return False

    def test_valid_byte_guards_synthesize_a_satisfying_value(self):
        for guard in ("data[0] == 48",
                      "data[0] == 200",                                    # was a neg-literal fallback bug
                      'int.from_bytes(data[0:2], "big") == 258',
                      'int.from_bytes(data[0:4], "little") == 5',
                      'int.from_bytes(data[0:2], "big", signed=True) == -25536',
                      'int.from_bytes(data[0:1], "big", signed=True) == -1'):
            sol = self._solve(guard)
            self.assertEqual(sol.strategy, "direct", guard)
            self.assertTrue(self._holds(guard, sol.satisfying_inputs[0]["data"]), guard)

    def test_unsatisfiable_byte_guards_return_none(self):
        for guard in ('data[0] == 1 and int.from_bytes(data[0:2], "big") == 2',
                      "data[0] == 5 and data[0] != 5",
                      'int.from_bytes(data[0:2], "big", signed=True) == 40000'):  # > signed max
            sol = self._solve(guard)
            self.assertEqual(sol.strategy, "none", f"{guard} -> {sol.strategy}")
            self.assertEqual(sol.satisfying_inputs, [], guard)

    def test_a_satisfiable_non_byte_disjunct_is_not_declared_unsat(self):
        # A guard whose byte atom is out of range but whose OTHER leaf (a startswith, an
        # endswith, or a `!=` on the whole value) can satisfy it must NOT return "none".
        # `_byte_atoms_unsatisfiable` reasons only about byte atoms, so it must bail on any
        # rescuing structure (a false-UNSAT regression the adversarial audit caught).
        for guard in (r"(data[0] > 300 or data.startswith(b'AB')) and data != b'\x00'",
                      r"(int.from_bytes(data[0:2],'big') > 70000 or data.startswith(b'PK'))"
                      r" and data != b''",
                      r"(data[0] < 0 or data.endswith(b'ZZ')) and data != b''"):
            sol = self._solve(guard)
            self.assertNotEqual(sol.strategy, "none", guard)

    def test_a_never_returns_a_direct_false_sat(self):
        # Whatever the strategy, a returned `direct` value must actually satisfy the guard.
        for guard in ('data[0] == 1 and int.from_bytes(data[0:2], "big") == 2',
                      'int.from_bytes(data[0:2], "big", signed=True) == 40000',
                      'int.from_bytes(data[0:2], "big") == 4660 and hex(data[0]) != "0x00"'):
            sol = self._solve(guard)
            if sol.strategy == "direct" and sol.satisfying_inputs:
                self.assertTrue(self._holds(guard, sol.satisfying_inputs[0]["data"]),
                                f"direct false-SAT: {guard}")

    def test_a_valid_value_using_hex_is_not_over_rejected(self):
        # The verifier used to lack hex(), discarding a genuinely satisfying value.
        guard = 'int.from_bytes(data[0:2], "big") == 4660 and hex(data[0]) != "0x00"'
        sol = self._solve(guard)
        self.assertEqual(sol.strategy, "direct")
        self.assertTrue(self._holds(guard, sol.satisfying_inputs[0]["data"]))


if __name__ == "__main__":
    unittest.main()
