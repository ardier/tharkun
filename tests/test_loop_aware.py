"""Loop-aware static resolution: break-fact collection, assert-sensitivity,
natural-exit negation, the in-order reaching definition (value carried PAST a
loop to a post-loop point), soundness, and the FactEnv bridge.

See docs/LOOP_AWARE_RESOLUTION.md.
"""
from __future__ import annotations

from tharkun import collect_loop_exit_facts, loop_exit_factenv
from tharkun.solve.condition import _resolve_scope_variables


def _write(tmp_path, src: str):
    p = tmp_path / "m.py"
    p.write_text(src, encoding="utf-8")
    return str(p)


# A while-True state machine whose every break sets q = 2, with a bw_assume assert.
WHILE_TRUE = """\
def f(data):
    q = 0
    i = 0
    while True:
        if data[i] == 1:
            assert not (data[i] < 0)
            q = 2
            break
        i = i + 1
    return (q, True)
"""

# A loop with a *natural* exit (the condition can fall through).
NATURAL = """\
def f(data):
    q = 0
    i = 0
    while i < len(data):
        if data[i] == 1:
            q = 2
            break
        i = i + 1
    return (q, True)
"""

# Two terminal states: q = 2 or q = 3, no natural fall-through.
MULTI_TERMINAL = """\
def f(data):
    q = 0
    while True:
        if data[0]:
            q = 2
            break
        if data[1]:
            q = 3
            break
    return q
"""


def test_break_fact_collection(tmp_path):
    facts = collect_loop_exit_facts(_write(tmp_path, WHILE_TRUE), lineno=9)  # the return
    breaks = [f for f in facts if f["kind"] == "break"]
    assert breaks, "expected a break exit fact"
    assert all(b["assignments"].get("q") == 2 for b in breaks)


def test_assert_sensitive(tmp_path):
    facts = collect_loop_exit_facts(_write(tmp_path, WHILE_TRUE), lineno=9)
    breaks = [f for f in facts if f["kind"] == "break"]
    assert any(b["asserts"] for b in breaks), "the bw_assume assert must be collected"


def test_natural_exit_negates_condition(tmp_path):
    facts = collect_loop_exit_facts(_write(tmp_path, NATURAL), lineno=8)
    natural = [f for f in facts if f["kind"] == "natural"]
    assert natural, "expected a natural exit fact for a `while <cond>` loop"
    assert natural[0]["conditions"] == ["not (i < len(data))"]


def test_while_true_has_no_natural_exit(tmp_path):
    facts = collect_loop_exit_facts(_write(tmp_path, WHILE_TRUE), lineno=9)
    assert not any(f["kind"] == "natural" for f in facts)


def test_value_carried_to_return(tmp_path):
    # The whole point: q is 2 AT THE RETURN, not the pre-loop init 0.
    resolved = _resolve_scope_variables(["q"], _write(tmp_path, WHILE_TRUE), lineno=9)
    assert resolved == {"q": 2}


def test_natural_exit_is_conservative(tmp_path):
    # A natural fall-through means q could be the break value OR the carried value;
    # the resolver must NOT claim a single value.
    resolved = _resolve_scope_variables(["q"], _write(tmp_path, NATURAL), lineno=8)
    assert "q" not in resolved


def test_factenv_single_value(tmp_path):
    fe = loop_exit_factenv(_write(tmp_path, WHILE_TRUE), lineno=9)
    assert "q" in fe.facts
    assert fe.facts["q"].eq == {2}


def test_factenv_multi_terminal_value_set(tmp_path):
    fe = loop_exit_factenv(_write(tmp_path, MULTI_TERMINAL), lineno=9)
    assert fe.facts["q"].eq == {2, 3}


def test_factenv_natural_exit_unknown(tmp_path):
    fe = loop_exit_factenv(_write(tmp_path, NATURAL), lineno=8)
    assert "q" not in fe.facts  # natural fall-through -> unknown, never guessed


def test_post_loop_reassignment_wins(tmp_path):
    # A constant reassignment AFTER the loop overrides the loop-exit value.
    src = (
        "def f(data):\n"
        "    q = 0\n"
        "    while True:\n"
        "        if data[0]:\n"
        "            q = 2\n"
        "            break\n"
        "    q = 7\n"
        "    return q\n"
    )
    resolved = _resolve_scope_variables(["q"], _write(tmp_path, src), lineno=8)
    assert resolved == {"q": 7}
