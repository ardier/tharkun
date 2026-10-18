"""Fixpoint reuse (A4, docs/ABSINT.md "Fixpoint reuse"): ``Analyzer._fixpoint`` hands back the loop head of an
earlier solve when the same loop is entered again, in the same call string and analyzer scratch, from a
state ``State.equiv`` calls equal.

What this file pins, in order of how bad the failure would be:

* reuse changes NOTHING a result says (verdict, component, sites, reason, lemmas, records, invariants), on
  programs built to make the same inner loop be solved again and again, and on the real-world programs the
  differential harness measured;
* it never serves the recording pass, a loop inside a length probe (``_probe_length``), or a run whose key differs
  (call string, float landmarks, learned globals, entry state);
* a hit leaves the analyzer exactly as the skipped solve would have (temporary counter, epoch, borrowed set,
  landmarks), so every later step sees what it would have seen;
* it solves less: a nest of loops asks for far fewer fixpoints than it requests.
"""

from __future__ import annotations

import ast
import textwrap
from pathlib import Path

import pytest

from tharkun.solve.absint import prove
from tharkun.solve.absint import lemmas
from tharkun.solve.absint.aval import FloatInfo, V, int_iv
from tharkun.solve.absint.interp import FIX_REUSE_WAYS, Analyzer
from tharkun.solve.absint.state import Frame, State
from tharkun.solve.values import FloatFact

FIX = Path(__file__).parent / "fixtures" / "absint"
CLASSES = {"AR": ("ZeroDivisionError", "FloatingPointError"), "DL": ("KeyError", "IndexError"),
           "DT": ("TypeError", "AttributeError"), "AS": ("AssertionError",)}

PROGRAMS = {
    "triple_nest": """
        total = 0
        for i in range(5):
            for j in range(3):
                for k in range(3):
                    total += 1
        print(10 // (total + 1))
    """,
    "while_nest": """
        i = 0
        s = 0
        while i < 4:
            j = 0
            while j < 3:
                s += j
                j += 1
            i += 1
        print(10 // (s + 1))
    """,
    "call_in_loop": """
        def inner(m):
            t = 0
            for q in range(m):
                t += q
            return t
        acc = 0
        for r in range(4):
            acc += inner(3)
        print(10 // (acc + 1))
    """,
    "unproven_division_in_the_inner_loop": """
        s = 0
        for i in range(4):
            for j in range(3):
                s += 10 // (j - 1)
        print(s)
    """,
    "break_and_continue": """
        xs = [1, 2, 3, 4]
        n = 0
        for a in range(3):
            for b in range(len(xs)):
                if xs[b] == 2:
                    continue
                if xs[b] == 4:
                    break
                n += xs[b]
        print(100 // (n + 1))
    """,
    "float_state_nest": """
        x = 1.0
        for i in range(3):
            for j in range(4):
                x = x * 0.5 + 1.0
        print(1.0 / x)
    """,
    "list_growth_nest": """
        xs = []
        for i in range(3):
            for j in range(3):
                xs.append(j)
        print(xs[0], xs[len(xs) - 1], 7 // len(xs))
    """,
    "sibling_loops_in_a_function_called_twice": """
        def f(n):
            a = 0
            for i in range(n):
                a += 1
            b = 0
            for i in range(n):
                b += 2
            return a + b
        print(8 // (f(3) + f(3) + 1))
    """,
}


def _observed(r) -> dict:
    """Everything a result says, the time aside."""
    return {"verdict": r.verdict, "component": r.component, "sites": r.sites, "reason": r.reason,
            "lemmas": r.lemmas, "checks_total": r.checks_total, "records": r.records,
            "complete": r.complete, "invariants": [repr(i) for i in r.invariants]}


@pytest.fixture
def no_reuse(monkeypatch):
    """A switch for the TEST only: the analyzer with every fixpoint solved."""
    def off():
        monkeypatch.setattr(Analyzer, "_fixpoint", Analyzer._solve_fixpoint)
    return off


@pytest.fixture(autouse=True)
def _landmarks():
    lemmas.reset_landmarks()
    yield
    lemmas.reset_landmarks()


# ------------------------------------------------------- results do not change
@pytest.mark.parametrize("classes", sorted(CLASSES))
@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_reuse_changes_nothing_a_result_says(name, classes, monkeypatch):
    source = textwrap.dedent(PROGRAMS[name])
    with_reuse = _observed(prove(source, CLASSES[classes], time_budget=60))
    monkeypatch.setattr(Analyzer, "_fixpoint", Analyzer._solve_fixpoint)
    assert _observed(prove(source, CLASSES[classes], time_budget=60)) == with_reuse


@pytest.mark.parametrize("task, classes", [("bm_nbody_nondet.py", "AR"), ("bm_nbody_nondet.py", "DT"),
                                           ("bm_fannkuch_nondet.py", "DL")])
def test_reuse_changes_nothing_on_a_real_world_program(task, classes, monkeypatch):
    source = (FIX / "targets" / task).read_text()
    with_reuse = _observed(prove(source, CLASSES[classes], time_budget=60))
    assert with_reuse["complete"]
    monkeypatch.setattr(Analyzer, "_fixpoint", Analyzer._solve_fixpoint)
    assert _observed(prove(source, CLASSES[classes], time_budget=60)) == with_reuse


# ------------------------------------------------------------- it does reuse
def _count(monkeypatch):
    seen = {"requested": 0, "solved": 0, "recording_requested": 0, "recording_solved": 0}
    fixpoint, solve = Analyzer._fixpoint, Analyzer._solve_fixpoint

    def counting_fixpoint(self, *a, **k):
        seen["requested"] += 1
        seen["recording_requested"] += bool(self.record)
        return fixpoint(self, *a, **k)

    def counting_solve(self, *a, **k):
        seen["solved"] += 1
        seen["recording_solved"] += bool(self.record)
        return solve(self, *a, **k)

    monkeypatch.setattr(Analyzer, "_fixpoint", counting_fixpoint)
    monkeypatch.setattr(Analyzer, "_solve_fixpoint", counting_solve)
    return seen


def test_a_nest_of_loops_is_solved_far_less_often_than_it_is_asked(monkeypatch):
    seen = _count(monkeypatch)
    assert prove(textwrap.dedent(PROGRAMS["triple_nest"]), CLASSES["AR"], time_budget=60).verdict == "TRUE"
    assert seen["solved"] < seen["requested"] * 0.75, seen


def test_the_recording_pass_always_solves_its_own_loops(monkeypatch):
    seen = _count(monkeypatch)
    prove(textwrap.dedent(PROGRAMS["triple_nest"]), CLASSES["AR"], time_budget=60)
    assert seen["recording_requested"] > 0
    assert seen["recording_solved"] == seen["recording_requested"], seen
    assert seen["solved"] < seen["requested"], seen   # while the non-recording ones do hit


# ------------------------------------------- one run's key: when a hit must not happen
def _analyzer(landmarks=True):
    an = Analyzer("while x:\n    pass\n", "<p>", time_budget=60)
    an.record = False
    calls = []

    def solve(entry, step, fctx, node):
        calls.append(1)
        an.tmp += 3
        an.epoch = {"epoch-site"}
        an.borrowed[-1].add("borrowed-addr")
        if landmarks:
            lemmas.add_landmarks(2.5, 9.5)
        return entry.copy()

    an._solve_fixpoint = solve
    return an, calls, an.tree.body[0]


def _state(**env) -> State:
    st = State()
    st.frames.append(Frame("g"))
    for k, v in env.items():
        st.frame.env[k] = v
    return st


def _entry(lo=0, hi=3) -> State:
    return _state(x=int_iv(lo, hi))


def test_the_same_entry_is_solved_once_and_the_hit_applies_what_the_solve_changed():
    an, calls, node = _analyzer()
    head = an._fixpoint(_entry(), None, None, node)
    assert (an.tmp, an.epoch, an.borrowed[-1]) == (3, {"epoch-site"}, {"borrowed-addr"})
    # what a second solve would have started from: the scratch as the first one found it
    an.epoch, an.tmp = set(), 100
    an.borrowed[-1].clear()
    lemmas.reset_landmarks()
    key = lemmas.landmark_key()
    again = an._fixpoint(_entry(), None, None, node)
    assert len(calls) == 1                                    # not solved again
    assert again is not head and again.equiv(head)           # a copy, not the cached object
    assert an.tmp == 103                                      # the counter advances as a solve would
    assert an.epoch == {"epoch-site"} and an.borrowed[-1] == {"borrowed-addr"}
    assert lemmas.landmark_key() != key and 2.5 in lemmas.LANDMARKS_LO and 9.5 in lemmas.LANDMARKS_HI


def test_a_different_entry_state_is_solved_again():
    an, calls, node = _analyzer()
    an._fixpoint(_entry(0, 3), None, None, node)
    an.epoch, an.tmp = set(), 0
    an.borrowed[-1].clear()
    an._fixpoint(_entry(0, 4), None, None, node)
    assert len(calls) == 2


@pytest.mark.parametrize("change", ["record", "catch", "callstring", "landmarks", "globals", "epoch", "borrowed",
                                    "extern", "widen_delay"])
def test_a_run_whose_key_differs_is_solved_again(change, monkeypatch):
    an, calls, node = _analyzer()
    an._fixpoint(_entry(), None, None, node)
    an.epoch, an.tmp = set(), 0
    an.borrowed[-1].clear()
    if change == "record":
        an.record = True
    elif change == "catch":
        an._probe_scopes.append(object())
    elif change == "callstring":
        an.callstring.append((7, 7))
    elif change == "landmarks":
        lemmas.add_landmarks(0.125, 64.0)
    elif change == "globals":
        an.global_written.add("g:counter")
    elif change == "epoch":
        an.epoch = {"another-site"}
    elif change == "borrowed":
        an.borrowed[-1].add("another-addr")
    elif change == "extern":
        an._ext_failed.add("/tmp/other.py")
    elif change == "widen_delay":
        # solve state the loop design added (A2): a head solved under one delay is not the head under another
        from tharkun.solve.absint import interp
        monkeypatch.setattr(interp, "WIDEN_DELAY", interp.WIDEN_DELAY + 1)
    an._fixpoint(_entry(), None, None, node)
    assert len(calls) == 2


def test_a_loop_under_a_length_probe_is_not_even_stored():
    an, calls, node = _analyzer()
    an._probe_scopes.append(object())
    an._fixpoint(_entry(), None, None, node)
    an._probe_scopes.pop()
    an.epoch, an.tmp = set(), 0
    an.borrowed[-1].clear()
    an._fixpoint(_entry(), None, None, node)
    assert len(calls) == 2                                    # its raises were noted in that scope: no skipping


def test_a_bottom_entry_is_never_reused():
    an, calls, node = _analyzer()
    bottom = _entry()
    bottom.set_bottom()
    an._fixpoint(bottom, None, None, node)
    an._fixpoint(bottom, None, None, node)
    assert len(calls) == 2


def test_a_solve_that_learns_something_is_not_kept():
    an, calls, node = _analyzer()
    solve = an._solve_fixpoint

    def learning(entry, step, fctx, n):
        an.global_written.add("g:learned")
        return solve(entry, step, fctx, n)

    an._solve_fixpoint = learning
    an._fixpoint(_entry(), None, None, node)
    an.epoch, an.tmp = set(), 0
    an.borrowed[-1].clear()
    an._fixpoint(_entry(), None, None, node)
    assert len(calls) == 2    # the second run starts from different knowledge, so it is a different run


def test_only_the_last_few_runs_of_a_key_are_kept():
    an, calls, node = _analyzer(landmarks=False)
    for hi in range(FIX_REUSE_WAYS + 3):
        an.epoch, an.tmp = set(), 0
        an.borrowed[-1].clear()
        an._fixpoint(_entry(0, hi), None, None, node)
    assert sum(len(runs) for runs in an._fix_cache.values()) == FIX_REUSE_WAYS
    an.epoch, an.tmp = set(), 0
    an.borrowed[-1].clear()
    an._fixpoint(_entry(0, 0), None, None, node)               # the oldest went first: solved again
    assert len(calls) == FIX_REUSE_WAYS + 4


# ---------------------------------------------------------- State.equiv itself
def test_equal_states_are_equiv_and_a_copy_is_too():
    a = _entry()
    assert a.equiv(a.copy()) and a.copy().equiv(a)


def test_a_state_that_includes_the_other_but_is_not_equal_is_not_equiv():
    wide, narrow = _entry(0, 4), _entry(0, 3)
    assert narrow.leq(wide) and not wide.leq(narrow)
    assert not wide.equiv(narrow) and not narrow.equiv(wide)


def test_a_different_variable_set_is_not_equiv():
    assert not _entry().equiv(_state(x=int_iv(0, 3), y=int_iv(0, 0)))


def test_a_different_frame_stack_is_not_equiv():
    a, b = _entry(), _entry()
    b.frames.append(Frame("c1"))
    assert not a.equiv(b)
    c = _entry()
    c.frames[0] = Frame("h", dict(c.frames[0].env))
    assert not a.equiv(c)


def test_different_pins_are_not_equiv():
    a, b = _entry(), _entry()
    b.pins = (int_iv(1, 1),)
    assert not a.equiv(b)


def test_one_sided_frame_binding_is_joined_as_maybe_unbound():
    bound = _state(x=int_iv(1, 1))
    absent = _state()
    joined = bound.join(absent)
    assert joined.frame.env["x"] == int_iv(1, 1)
    assert joined.frame.maybe_unbound == {"x"}
    assert bound.leq(joined) and absent.leq(joined)
    assert not joined.leq(bound) and not joined.leq(absent)
    assert not absent.leq(bound)
    assert joined.equiv(joined.copy())

    widened = bound.widen(absent)
    assert widened.frame.maybe_unbound == {"x"}
    assert joined.equiv(widened)


def test_the_proof_text_of_a_value_is_part_of_equiv_though_leq_and_equality_ignore_it():
    fact = FloatFact.top()
    a = _state(f=V(f=FloatInfo(fact, None, "lemma A")))
    b = _state(f=V(f=FloatInfo(fact, None, "lemma B")))
    assert a.leq(b) and b.leq(a) and a.frames[0].env["f"] == b.frames[0].env["f"]
    assert not a.equiv(b) and a.equiv(a.copy())


def test_a_bottom_state_equals_nothing_not_even_itself():
    a = _entry()
    a.set_bottom()
    assert not a.equiv(a.copy())


def test_landmark_key_and_merge_round_trip():
    lemmas.add_landmarks(-1.5, 3.5)
    key = lemmas.landmark_key()
    lemmas.reset_landmarks()
    assert lemmas.landmark_key() != key
    lemmas.merge_landmarks(key)
    assert lemmas.landmark_key() == key


def test_the_corpus_programs_are_valid_python():
    for src in PROGRAMS.values():
        ast.parse(textwrap.dedent(src))
