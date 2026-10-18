"""The concolic worker's stage deadline is ENFORCED, also against a program that swallows the alarm (#155 review note 2).

``sv/_concolic_worker.py`` arms ``signal.alarm`` around everything that runs the program under test; the alarm raises
``_SiteTimeout`` inside the program.  When that was an ``Exception``, a broad ``except Exception`` in the program (or in
``driver.drive_to_line``, which treats an exception of the subject as a valid trace) swallowed the one-shot alarm and the worker
ran on until the parent killed it at ``budget + 5`` s, losing its result and overrunning the stage (bm_go_nondet at the 9 s
triage budget: 11.4 s).  The tests give a program that spins forever behind such a handler and assert the worker FINISHED (its
record exists: ``Solved.ran``), which a parent kill never leaves; the stage budget is 3 s, the parent's kill at 8 s.  (The symbolic stage reaches the site, and the worker then replays the
solved vector on the real program to read the draws it consumes: that replay ran under a fixed 10 s alarm, longer than the stage
and than the parent's kill; it is now bounded by the stage deadline.)
"""
from __future__ import annotations

import textwrap

import pytest

pytest.importorskip("z3")

from tharkun.sv import concolic  # noqa: E402

_SPIN = """
    import _sv_verifier

    def spin():
        while True:
            try:
                for i in range(100000):
                    pass
            except {handler}:
                pass

    x = _sv_verifier.nondet_int()
    spin()
    assert x != 7
"""


def test_a_program_that_swallows_the_alarm_does_not_outlive_the_worker_deadline(tmp_path):
    p = tmp_path / "spin.py"
    p.write_text(textwrap.dedent(_SPIN.format(handler="Exception")))
    solved = concolic.solve(str(p), ("AssertionError",), budget_seconds=3.0, drive=True)
    assert solved.ran, "the worker was killed by the parent: its deadline was not enforced"
