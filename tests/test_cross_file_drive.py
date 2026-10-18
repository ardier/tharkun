"""Driving to a line that lives BELOW the entry function, in another file.

A line in a transitive callee appears in the coverage report exactly like one in the
entry module, so it is a legitimate target — but the tracer watched a single file, so it
was invisible and could never be driven to. Worse, line numbers are per-file: falling
back to the entry file's executed set makes an unrelated line of the same number read as
"the target ran".

Reaching `inner.py` also needs two things the loop did not do:

* the blocker is in the CALLER while the callee has not run, so its lexical guards say
  only "we never got there";
* `len(payload) > 4` can be answered with a str, which makes the deeper `payload[0] ==
  0x7F` unsatisfiable — the fact set has to iterate rather than blame the deeper guard.

Every assertion here is checked against a real trace, never against the driver's own
claim: a reachability engine that grades its own homework is worth nothing.
"""
from __future__ import annotations

import os
import sys
import unittest

from tharkun.driver import PLATEAU, REACHED, drive_to_line

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "xfilepkg")
OUTER = os.path.join(FIX, "outer.py")
INNER = os.path.join(FIX, "inner.py")


def setUpModule() -> None:
    # outer.py does `from xfilepkg import inner`, so the fixtures dir must be importable
    # BEFORE any test drives it — otherwise the load fails and the drive plateaus for a
    # reason that has nothing to do with the driver.
    d = os.path.dirname(FIX)
    if d not in sys.path:
        sys.path.insert(0, d)


def _really_runs(line: int, kwargs: dict) -> bool:
    """Ground truth: execute the entry point and watch inner.py for *line*."""
    from xfilepkg.outer import entry

    target = os.path.realpath(INNER)
    hit: set[int] = set()

    def tr(frame, event, arg):
        if event == "line" and os.path.realpath(frame.f_code.co_filename) == target:
            hit.add(frame.f_lineno)
        return tr

    sys.settrace(tr)
    try:
        entry(**kwargs)
    except Exception:
        pass
    finally:
        sys.settrace(None)
    return line in hit


class TestCrossFileTarget(unittest.TestCase):
    def _drive(self, line: int):
        return drive_to_line(OUTER, "entry", line, {"mode": 0, "payload": b""},
                             target_path=INNER, max_rounds=16)

    def test_reaches_a_line_gated_only_by_the_caller(self) -> None:
        r = self._drive(9)                       # return "INNER-SHORT"
        self.assertEqual(r.status, REACHED)
        self.assertTrue(_really_runs(9, r.inputs))

    def test_reaches_a_line_behind_a_callee_guard(self) -> None:
        r = self._drive(8)                       # return "INNER-SHALLOW"
        self.assertEqual(r.status, REACHED)
        self.assertTrue(_really_runs(8, r.inputs))

    def test_reaches_through_caller_guard_plus_two_callee_guards(self) -> None:
        r = self._drive(7)                       # return "INNER-DEEP"
        self.assertEqual(r.status, REACHED, r.diagnostics)
        self.assertTrue(_really_runs(7, r.inputs))

    def test_deep_target_needs_the_types_to_stay_consistent(self) -> None:
        # The only way to satisfy len(payload) > 4 AND payload[0] == 0x7F at once.
        r = self._drive(7)
        self.assertIsInstance(r.inputs["payload"], bytes)
        self.assertGreater(len(r.inputs["payload"]), 4)
        self.assertEqual(r.inputs["payload"][0], 0x7F)
        self.assertEqual(r.inputs["mode"], 1234)

    def test_every_claim_matches_a_real_trace(self) -> None:
        for line in (7, 8, 9):
            r = self._drive(line)
            self.assertEqual(r.reached, _really_runs(line, r.inputs),
                             f"inner.py:{line}: claimed {r.reached}, trace disagrees")

    def test_line_numbers_are_not_confused_across_files(self) -> None:
        # outer.py has a line 7 too. Targeting inner.py:7 with a seed that never enters
        # inner must NOT read outer.py's line 7 as a hit.
        r = drive_to_line(OUTER, "entry", 7, {"mode": 0, "payload": b""},
                          target_path=INNER, max_rounds=0)
        self.assertFalse(r.reached)


class TestSameFileUnaffected(unittest.TestCase):
    """target_path defaults to source_path; existing single-file behaviour is unchanged."""

    def test_same_file_target_still_works(self) -> None:
        r = drive_to_line(OUTER, "entry", 6, {"mode": 0, "payload": b""}, max_rounds=8)
        self.assertEqual(r.status, REACHED)
        self.assertEqual(r.inputs["mode"], 1234)

    def test_unguarded_line_is_still_no_target(self) -> None:
        from tharkun.driver import NO_TARGET

        r = drive_to_line(OUTER, "entry", 4, {"mode": 0, "payload": b""}, max_rounds=4)
        self.assertEqual(r.status, NO_TARGET)


if __name__ == "__main__":
    unittest.main()
