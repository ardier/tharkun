#!/usr/bin/env python3
"""The worked case study: synthesize a real ``codetiming.Timer``.

``Timer`` is a small class with a real precondition. ``stop()`` raises unless
``start()`` ran first, and ``start()`` raises if it already did -- so the
class cannot be exercised by calling methods at random, and a search that only
knows how to CONSTRUCT objects never reaches ``stop()`` at all.

Run from the repository root::

    python -m pip install 'tharkun[bench]'
    python demos/avengers_codetiming.py

Only the class is named. Everything below -- the setup call ``stop()`` needs,
the second receiver state, the sequences the class rejects on purpose, and how
far into the class they get -- is discovered.
"""

from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from codetiming import Timer  # noqa: E402

from tharkun.avengers import (  # noqa: E402
    IsolatedExecutor,
    MethodCall,
    ObjectGoal,
    synthesize_object,
)
from tharkun.avengers.engine import SaturationBudget  # noqa: E402


def _section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def main() -> int:
    _section("1. A Timer that can cross stop()")
    # The boundary is named; the start() that has to happen first is not.
    goal = ObjectGoal.from_type(Timer, (MethodCall("stop"),))
    directed = synthesize_object(goal, max_setup_steps=2)
    if directed.best is None:
        print("No validated recipe found within the search budget.")
        return 1
    print(directed.best.program)
    print(f"# first object after {directed.time_to_first_object_s:.2f}s")

    _section("2. Every objective, from the class name alone")
    report = synthesize_object(
        ObjectGoal.from_type(Timer),
        objectives=True,
        max_setup_steps=2,
        max_method_probes=32,
        executor=IsolatedExecutor(timeout_s=5.0, trace=True),
        budget=SaturationBudget(max_seconds=120),
    )
    print(
        f"{len(report.validated)} valid objects, "
        f"{len(report.states)} distinct receiver states, "
        f"{len(report.covered_methods)}/{len(report.method_universe)} methods reached"
    )
    if report.branch_coverage is not None:
        print(
            f"line {report.line_coverage:.0%}, branch {report.branch_coverage:.0%} "
            "of the Timer class body"
        )

    _section("3. The state portfolio")
    # One object is not enough: start() and stop() have incompatible
    # preconditions, so no single state admits both.
    for state in report.objects:
        admits = next(
            (
                item.returned
                for item in report.method_coverage
                if item.validated.recipe.key == state.recipe.key
            ),
            (),
        )
        print(f"  # admits {', '.join(call.name for call in admits) or 'nothing'}")
        for line in state.program.splitlines():
            print(f"  {line}")

    _section("4. Sequences the class rejects on purpose")
    print("These are results, not failures: each is a precondition, demonstrated.")
    for negative in report.negatives:
        confirmed = "reproduced" if negative.outcome.reproduced else "not reproduced"
        print(f"\n  # {negative.shape} -> {negative.exception_type} ({confirmed})")
        for line in negative.program.splitlines():
            print(f"  {line}")

    _section("5. What is still out of reach")
    if report.coverage is not None and report.coverage.uncovered_arcs:
        print("Uncovered branches, by guard line:")
        for start, end in report.coverage.uncovered_arcs:
            print(f"  line {start} -> {end}")
        print(
            "\nTwo reasons, and they are different reasons.\n"
            "\n"
            "  callable(self.text) has no literal witness at all: only a function\n"
            "  satisfies it. The coverage objective proposes nothing there rather\n"
            "  than a value that passes the guard and crashes on the line below.\n"
            "\n"
            "  The branches inside start()'s initial_text handling need TWO\n"
            "  constructor arguments set together. The objective varies one\n"
            "  parameter at a time, so it reaches each guard and not their\n"
            "  conjunction."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
