"""[internal ref]: a ``randrange`` draw over a range too wide to enumerate is searched by the literals the program compares with.

``random.randint(0, 2**100)`` is a ``rindex`` draw of ``2**100 + 1`` values.  The enumeration tries the first 32 indices and
``n - 1`` (``engine.py::_RINDEX_CAP``); a violation at ``x == 12345678901234567890`` was never reached, so the pairs MIN labels
FALSE (``ch_randint_huge_false``, ``ch_randint_neg_big_false``) stayed UNKNOWN.  The child now logs the arithmetic of a randrange
draw (``start``, ``step``: ``range[i] = start + step * i``, ``_child.py::_draw``) and ``engine.py::Candidates`` proposes, after
those, the index at which the call returns a literal of the program (``harvest_constants`` -> ``rindex_values``,
``rindex_hits``).  A hit is only a CANDIDATE: the child runs the vector, and a witness is what that run did.
"""
from __future__ import annotations

import sys
import textwrap

import pytest

from tharkun.sv import engine, verify
from tharkun.sv.engine import Candidates, harvest_constants, run_once


def program(tmp_path, body: str, name="prog.py"):
    path = tmp_path / name
    path.write_text("import random\n" + textwrap.dedent(body))
    return str(path)


# ---------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("values, n, base, expected", [
    ([12345678901234567890], 2**100 + 1, (0, 1), [12345678901234567890]),
    ([-123456789012345678901234567890], 2**100 + 1, (-2**100, 1), [2**100 - 123456789012345678901234567890]),
    ([2**65 * 7], 2**70, (0, 7), [2**65]),                                 # a step: range(0, .., 7)[2**65] = 7 * 2**65
    ([2**65 * 7 + 1], 2**70, (0, 7), []),                                  # not on the step: no index returns it
    ([5], 2**70, (10, 1), []),                                             # before the start
    ([10 + 2**70], 2**70, (10, 1), []),                                    # at index n: out of the range
    ([10 + 2**70 - 1], 2**70, (10, 1), [2**70 - 1]),                       # the last element
    ([100], 2**70, (200, -1), [100]),                                      # a negative step: range(200, ..., -1)[100] = 100
    ([100, 100, 101], 2**70, (200, -1), [100, 99]),                        # each index once
    ([7], 2**70, None, []),                                                # random.choice has no arithmetic
    ([7], 2**70, (0, 0), []),                                              # a zero step is a ValueError before any draw
])
def test_rindex_hits_is_the_inverse_of_the_range_arithmetic(values, n, base, expected):
    got = engine.rindex_hits(values, n, base)
    assert got == expected
    for i in got:                                                          # and it IS the inverse: the range returns the value
        assert range(base[0], base[0] + base[1] * n, base[1])[i] in values


def test_the_hits_come_after_the_small_indices_and_n_minus_1_and_only_for_a_wide_range():
    c = Candidates({"rindex_values": [12345678901234567890]}, deepen=False)
    n, base = 2**100 + 1, (0, 1)
    seq = [c.value("rindex", r, n, base) for r in range(40)]
    assert seq[:32] == list(range(32)) and seq[32] == n - 1 and seq[33] == 12345678901234567890 and seq[34] is None
    # a range the enumeration covers whole needs no hint: every index is already a candidate, once
    seq = [c.value("rindex", r, 20, (0, 1)) for r in range(25)]
    assert seq[:20] == list(range(20)) and seq[20] is None
    # without the arithmetic (random.choice) the old sequence is unchanged
    assert [c.value("rindex", r, n) for r in range(35)] == list(range(32)) + [n - 1, None, None]
    # a literal that is already a small index is not proposed twice
    c = Candidates({"rindex_values": [3, 40]}, deepen=False)
    assert [c.value("rindex", r, 1000, (0, 1)) for r in range(35)] == list(range(32)) + [999, 40, None]


def test_the_source_of_a_hit_is_a_harvested_constant():
    c = Candidates({"rindex_values": [5000]}, deepen=False)
    assert c.source({"kind": "rindex", "value": 5000, "n": 2**40, "start": 0, "step": 1}) == "constants"
    assert c.source({"kind": "rindex", "value": 4999, "n": 2**40, "start": 0, "step": 1}) == "small_first"
    assert c.source({"kind": "rindex", "value": 3, "n": 2**40, "start": 0, "step": 1}) == "small_first"


def test_harvest_constants_keeps_every_int_literal_at_any_size_with_its_neighbours(tmp_path):
    path = program(tmp_path, '''
        """a docstring with 777"""
        x = random.randint(0, 2**100)
        assert x != 12345678901234567890 and x != -123456789012345678901234567890 and x != 31
    ''')
    got = harvest_constants(path)["rindex_values"]
    for c in (12345678901234567890, 123456789012345678901234567890, 31):
        assert {c, c + 1, c - 1, -c, -c + 1, -c - 1} <= set(got)
    assert 777 not in got                                                  # a docstring is never a hint ([internal ref])
    assert len(got) <= engine._RINDEX_VALUES_CAP


# ---------------------------------------------------------------------------
# the child logs the arithmetic and the witness does not carry it
# ---------------------------------------------------------------------------

def test_the_child_logs_start_and_step_of_a_randrange_draw_and_none_for_choice(tmp_path):
    path = program(tmp_path, "random.randrange(10, 100, 5)\nrandom.randint(-3, 3)\nrandom.randrange(7)\nrandom.choice([4, 5])\n")
    out = run_once(sys.executable, path, [], timeout=30)
    assert out.status == "clean", (out.exc, out.msg, out.flags)
    assert [(d.get("start"), d.get("step"), d["n"]) for d in out.draws] == [(10, 5, 18), (-3, 1, 7), (0, 1, 7), (None, None, 2)]


def test_a_witness_vector_is_kind_value_and_n_only(tmp_path):
    path = program(tmp_path, "x = random.randint(0, 2**100)\nassert x != 5000\n")
    v = verify(path, ("AssertionError",), budget_seconds=20, jobs=1)
    assert v.verdict == "false", (v.verdict, v.reason)
    assert v.witness["vector"] == [{"kind": "rindex", "value": 5000, "n": 2**100 + 1}]


# ---------------------------------------------------------------------------
# end to end: the pairs, natively controlled
# ---------------------------------------------------------------------------

CASES = [
    ("ch_randint_huge_false", "x = random.randint(0, 2**100)\nassert x != 12345678901234567890\n", 12345678901234567890),
    ("ch_randint_neg_big_false", "x = random.randint(-(2**100), 0)\nassert x != -123456789012345678901234567890\n",
     2**100 - 123456789012345678901234567890),
    ("a stepped randrange", f"x = random.randrange(0, 2**90, 7)\nassert x != {7 * 2**80}\n", 2**80),
    ("a randrange from an offset", f"x = random.randrange({10**30}, {10**31})\nassert x != {5 * 10**30 + 17}\n", 4 * 10**30 + 17),
    ("randrange(2**63)", "x = random.randrange(2**63)\nassert x != 12345\n", 12345),
    ("a literal in a comparison, another class", f"x = random.randint(0, {10**26})\ny = 1 // (x - {10**25 + 3})\n", 10**25 + 3),
]


@pytest.mark.parametrize("label, body, index", CASES, ids=[c[0] for c in CASES])
def test_a_violation_at_a_literal_of_a_huge_range_is_found_and_replays(tmp_path, label, body, index):
    path = program(tmp_path, body)
    classes = ("ZeroDivisionError",) if "//" in body else ("AssertionError",)
    v = verify(path, classes, budget_seconds=20, jobs=1)
    assert v.verdict == "false", (label, v.verdict, v.reason)
    assert v.component == "enum:constants", v.component
    # the witness replays: a fresh child run of exactly its vector raises the class at the same place
    out = run_once(sys.executable, path, v.witness["vector"], timeout=30)
    assert out.status == "raised" and out.violated(classes) and out.flags == [], (out.status, out.exc, out.flags)
    if index is not None:
        assert v.witness["vector"][0]["value"] == index


@pytest.mark.parametrize("label, body", [
    ("a literal outside the range", f"x = random.randint(0, 2**100)\nassert x != {2**100 + 5}\n"),
    ("a literal off the step", f"x = random.randrange(0, 2**90, 7)\nassert x != {7 * 2**80 + 1}\n"),
    ("a literal before the start", "x = random.randint(100, 2**100)\nassert x != 99\n"),
    ("a property that holds", "x = random.randint(0, 2**100)\nassert x >= 0\n"),
], ids=lambda p: p if isinstance(p, str) and len(p) < 40 else None)
def test_a_literal_no_index_returns_is_never_invented_into_a_violation(tmp_path, label, body):
    v = verify(program(tmp_path, body), ("AssertionError",), budget_seconds=12, jobs=1)
    assert v.verdict != "false", (label, v.verdict, v.reason)         # TRUE or UNKNOWN, whatever the proofs say: never a witness


# ---------------------------------------------------------------------------
# negative controls: each rule undone, the tests above must fail
# ---------------------------------------------------------------------------

def test_the_arithmetic_tests_catch_a_hit_that_ignores_the_step(monkeypatch):
    real = engine.rindex_hits

    def broken(values, n, base):
        return real(values, n, (base[0], 1) if base else base)

    monkeypatch.setattr(engine, "rindex_hits", broken)
    with pytest.raises(AssertionError):
        test_rindex_hits_is_the_inverse_of_the_range_arithmetic([2**65 * 7], 2**70, (0, 7), [2**65])


def test_the_arithmetic_tests_catch_a_hit_that_ignores_the_range_end(monkeypatch):
    def broken(values, n, base):
        return [(v - base[0]) // base[1] for v in values] if base else []

    monkeypatch.setattr(engine, "rindex_hits", broken)
    with pytest.raises(AssertionError):
        test_rindex_hits_is_the_inverse_of_the_range_arithmetic([10 + 2**70], 2**70, (10, 1), [])


def test_the_end_to_end_tests_catch_a_search_that_ignores_the_logged_arithmetic(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "entry_base", lambda entry: None)
    path = program(tmp_path, CASES[0][1])
    v = verify(path, ("AssertionError",), budget_seconds=8, jobs=1)
    assert v.verdict != "false", (v.verdict, v.reason)                  # no arithmetic, no hint: the old UNKNOWN
