"""[internal ref]: absint runs on the exact comprehension lowering and reports on the ORIGINAL program.

The lowering turns comprehensions into loops (``solve/lowering``); the analyzer interprets that
tree.  What this file pins is the seam: a comprehension is no longer an abort, a finding sits at
the original comprehension's line and column, the fresh names of the lowering never leak into a
fact, a predicate or an invariant, and a lowered loop variable never makes a statement outside
the comprehension look safe (a wrong TRUE costs 32 points).
"""

from __future__ import annotations

import re
import textwrap

import pytest

from tharkun.solve.absint import prove

FRESH = re.compile(r"_lw\d+_")


def _prove(source: str, classes=("ZeroDivisionError",), **kw):
    return prove(textwrap.dedent(source), list(classes), filename="<program>", **kw)


def _no_fresh_names(result) -> None:
    blobs = [result.reason, *(str(site) for site in result.sites)]
    for rec in result.records:
        blobs.extend(str(v) for v in rec.values())
    for inv in result.invariants:
        blobs.append(str(inv))
    for blob in blobs:
        assert not FRESH.search(blob), blob


def test_a_program_with_a_comprehension_is_proved_instead_of_aborted():
    result = _prove("""
        def main(xs):
            ys = [10 // x for x in xs if x > 0]
            return ys
        main([1, 2])
    """)
    assert result.verdict == "TRUE", result.reason
    assert result.complete
    _no_fresh_names(result)


def test_an_unsafe_site_inside_a_comprehension_is_reported_at_the_comprehensions_position():
    result = _prove("""
        def main(xs):
            ys = [10 // x for x in xs]
            return ys
        main([0, 1])
    """)
    assert result.verdict == "UNKNOWN"
    divide = [r for r in result.records if r["exc"] == "ZeroDivisionError" and r["status"] == "may_raise"]
    assert [(r["line"], r["col"]) for r in divide] == [(3, 10)]  # the original `10 // x`
    assert divide[0]["status"] == "may_raise"
    assert divide[0]["residual"] == "(x) == 0"  # the loop variable under the name the source gave it
    _no_fresh_names(result)


def test_the_loop_variable_does_not_leak_into_the_enclosing_scope():
    # original: x stays 2, so the divisor is 0.  Were the comprehension variable to leak (the
    # last element, 3), the divisor would be 1 and the program would look safe: a wrong TRUE.
    result = _prove("""
        def main():
            x = 2
            ys = [x for x in [1, 3]]
            return 10 // (x - 2)
        main()
    """)
    assert result.verdict == "UNKNOWN"
    _no_fresh_names(result)


def test_a_fact_about_the_outer_variable_of_the_same_name_is_not_taken_from_the_loop_variable():
    result = _prove("""
        def main(xs):
            x = 0
            ys = [x + 1 for x in xs if x > 0]
            return 10 // x
        main([5])
    """)
    assert result.verdict == "UNKNOWN"
    # the UNSAFE sites: a call of a contracted callable (the comprehension's ``append``) records its seven classes too, as safe
    divide = [r for r in result.records if r["exc"] == "ZeroDivisionError" and r["status"] == "may_raise"]
    assert [(r["line"], r["col"]) for r in divide] == [(5, 11)]
    assert all("_lw" not in str(v) for r in divide for v in r.values())


def test_a_site_after_the_comprehension_says_nothing_about_the_fresh_variable():
    result = _prove("""
        def main(xs):
            ys = [c for c in xs if c > 0]
            z = 3
            return 10 // z
        main([1])
    """)
    assert result.verdict == "TRUE"
    _no_fresh_names(result)


def test_a_nested_comprehension_and_a_dict_comprehension_are_proved():
    result = _prove("""
        def main(n):
            grid = [[i * j + 1 for j in range(3)] for i in range(2)]
            table = {i: 100 // (i + 1) for i in range(4)}
            return grid, table
        main(3)
    """)
    assert result.verdict == "TRUE", result.reason
    _no_fresh_names(result)


def test_a_refused_comprehension_still_aborts_the_analysis_as_before():
    # a generator expression consumed by `any` is lazy: the lowering refuses it, absint aborts
    result = _prove("""
        def main(xs):
            return any(10 // x > 1 for x in xs)
        main([1, 2])
    """)
    assert result.verdict == "UNKNOWN"
    assert "unsupported" in result.reason
    assert not result.complete


def test_no_invariant_names_a_variable_of_the_lowering():
    result = _prove("""
        def main(xs):
            ys = [c + 1 for c in xs]
            n = 4
            m = n + 1
            return 10 // m
        main([1, 2])
    """, invariant_locations=[("<program>", 5, 5, "main"), ("<program>", 6, 5, "main")])
    assert result.verdict == "TRUE"
    _no_fresh_names(result)
