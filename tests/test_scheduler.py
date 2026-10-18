"""sv/scheduler.py::run -- step 1's order/budget logic, moved from engine.py::verify
unchanged ([internal ref], docs/SV.md "3a").

These are the targeted regression tests the bar for step 1 names explicitly: the two
"hidden" branches inside "execution" that the docstring's five-stage list does not show
(a crashed/timed-out first run ends everything immediately; a zero-draw, no-flag first
run takes its own exclusive return path), and the double concolic_share spend (stage 2's
own symbolic call, and the portfolio's narrower retry, each drawn from a DIFFERENT
``remaining()`` snapshot). Each is a real behavior-preservation risk called out in
docs/SV.md's "3a" -> "What step 1 must not change" -- a scheduler that merges either
hidden branch into a generic loop, or that computes both concolic shares from one
snapshot, changes what today's engine.py::verify does.
"""
from __future__ import annotations

import textwrap

import pytest

from tharkun.facts import Fact
from tharkun.knowledge import KnowledgeBase
from tharkun.sv import concolic as sv_concolic
from tharkun.sv import engine
from tharkun.sv import prove as sv_prove
from tharkun.sv import scheduler
from tharkun.sv.concolic import Solved


def _prog(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


# --- hidden branch 1: a crashed/timed-out first run ends verification immediately ------

def test_a_crashed_first_run_never_reaches_a_second_run(tmp_path, monkeypatch):
    p = _prog(tmp_path, 'crash.py', "pass\n")
    calls = []

    def fake_run_once(python, program, vector, **kwargs):
        calls.append(vector)
        return engine.Outcome('crashed', msg='synthetic crash')

    monkeypatch.setattr(engine, 'run_once', fake_run_once)
    v = engine.verify(p, ('KeyError',), prove=False, concolic=False, budget_seconds=60)
    assert v.verdict == 'unknown' and 'first run crashed' in v.reason
    # exactly the first run (vector []) -- no second hash-seed run, no portfolio run,
    # no enumeration run
    assert calls == [[]]


def test_a_timed_out_first_run_never_reaches_a_second_run(tmp_path, monkeypatch):
    p = _prog(tmp_path, 'slow.py', "pass\n")
    calls = []

    def fake_run_once(python, program, vector, **kwargs):
        calls.append(vector)
        return engine.Outcome('timeout')

    monkeypatch.setattr(engine, 'run_once', fake_run_once)
    v = engine.verify(p, ('KeyError',), prove=False, concolic=False, budget_seconds=60)
    assert v.verdict == 'unknown' and 'first run timeout' in v.reason
    assert calls == [[]]


# --- hidden branch 2: zero draws / no flags is an exclusive early return ---------------

def test_zero_draws_never_reaches_prove_dynamic_or_the_portfolio_or_enumeration(tmp_path, monkeypatch):
    p = _prog(tmp_path, 'det.py', """
        xs = [1, 2, 3]
        if __name__ == "__main__":
            print(xs[2])
    """)

    def fail_prove_dynamic(*a, **k):
        pytest.fail('prove_dynamic must not run when the first run drew nothing')

    def fail_thread_pool(*a, **k):
        pytest.fail('the enumeration (ThreadPoolExecutor) must not run when the first run drew nothing')

    monkeypatch.setattr(sv_prove, 'prove_dynamic', fail_prove_dynamic)
    monkeypatch.setattr(engine, 'ThreadPoolExecutor', fail_thread_pool)
    v = engine.verify(p, ('KeyError', 'IndexError'), budget_seconds=60)
    assert v.verdict == 'true' and v.runs == 2


def test_zero_draws_with_uncertified_determinism_stops_at_two_runs(tmp_path, monkeypatch):
    """Same early-return branch, the UNKNOWN exit: still exactly 2 runs, no later stage."""
    p = _prog(tmp_path, 'det_unknown.py', """
        import array
        a = array.array('i', [1, 2, 3])
        if __name__ == "__main__":
            print(a.buffer_info())
    """)

    def fail_prove_dynamic(*a, **k):
        pytest.fail('prove_dynamic must not run when the first run drew nothing')

    def fail_thread_pool(*a, **k):
        pytest.fail('the enumeration (ThreadPoolExecutor) must not run when the first run drew nothing')

    monkeypatch.setattr(sv_prove, 'prove_dynamic', fail_prove_dynamic)
    monkeypatch.setattr(engine, 'ThreadPoolExecutor', fail_thread_pool)
    v = engine.verify(p, ('KeyError', 'IndexError'), budget_seconds=60, prove=False)
    assert v.verdict == 'unknown' and v.runs == 2


# --- the double concolic_share spend ----------------------------------------------------

def test_concolic_share_is_spent_twice_from_two_different_snapshots(tmp_path, monkeypatch):
    """Stage 2 (the whole-harness symbolic solve) and the portfolio's narrowed retry each
    call ``concolic.solve`` once, each with its OWN ``budget_seconds``/``budget`` computed
    from ``remaining()`` AT THAT CALL -- not a single upfront split of the original
    budget. Mocking ``concolic.solve`` to leave a site "open" makes both calls fire;
    the mock advances the ledger's CPU clock (``engine._cpu_seconds``, with ``cpu_time_only``, the
    [internal ref] seam) by a fixed amount, so the second ``remaining()`` is smaller than the first by
    exactly that much: no sleep, no wall clock, and the hybrid stage (which also calls
    ``concolic.solve`` when the fuzzer gets time, which a loaded host decides) is not in the run."""
    p = _prog(tmp_path, 'nd.py', """
        from _sv_verifier import nondet_int
        if __name__ == "__main__":
            x = abs(nondet_int())
            if x == 999999:
                raise KeyError()
    """)
    calls = []
    clock = [0.0]
    monkeypatch.setattr(engine, '_cpu_seconds', lambda: clock[0])

    def fake_solve(program, classes, *, budget_seconds, **kwargs):
        calls.append(budget_seconds)
        clock[0] += 0.2
        return Solved(candidates=[], sites={f'nd.py:{len(calls)}': 'open'})

    monkeypatch.setattr(sv_concolic, 'solve', fake_solve)
    v = engine.verify(p, ('KeyError',), budget_seconds=15, prove=False, concolic_share=0.4,
                      stages=['concolic'], cpu_time_only=True)
    assert len(calls) == 2, f"expected exactly 2 concolic.solve calls, got {len(calls)}: {calls}"
    assert calls[0] != calls[1], (
        "both concolic budgets were identical -- looks like a single upfront split, "
        "not two independent remaining() snapshots"
    )
    assert calls[1] < calls[0], "the second (later) snapshot must see LESS remaining budget"


# --- the obligation table: a KnowledgeBase, not a new structure (docs/SV.md "3a") -------
# (the table's own behavior -- rows, decisions, hints, reads -- is tests/test_obligation_table.py)

def test_run_keeps_its_obligation_table_in_a_knowledgebase_with_one_open_row_per_obligation(tmp_path, monkeypatch):
    p = _prog(tmp_path, 'obligations.py', """
        d = {}
        if __name__ == "__main__":
            print(d['missing'])
    """)
    seen = []
    original = scheduler.ObligationTable

    def spy(*a, **k):
        t = original(*a, **k)
        seen.append(t)
        return t

    monkeypatch.setattr(scheduler, 'ObligationTable', spy)
    engine.verify(p, ('KeyError',), budget_seconds=60)
    (table,) = seen
    assert isinstance(table.kb, KnowledgeBase)
    rows = [f for f in table.kb if f.kind == 'obligation']
    assert rows and all(not f.subject.startswith('$coverage:') for f in rows)
    assert {f.value for f in rows if not f.certified} == {'open'}      # the census's own rows stay as written


def test_verify_decides_the_same_whatever_a_fresh_table_already_holds(tmp_path):
    """A program with nothing to hand over decides the same verdict and proof component
    with each fresh table. Independent calls may perform different numbers of executions
    within their timing budgets."""
    p = _prog(tmp_path, 'kb_inert.py', """
        d = {'a': 1}
        if __name__ == "__main__":
            print(d['a'])
    """)
    v1 = engine.verify(p, ('KeyError',), budget_seconds=60)
    v2 = engine.verify(p, ('KeyError',), budget_seconds=60)
    assert v1.line() == v2.line() and v1.component == v2.component


# --- [internal ref] step 2: the store's write path ([internal ref]'s step-2 design note) -----

def test_a_true_verdict_writes_a_certified_discharge_for_every_open_obligation(tmp_path, monkeypatch):
    """docs/SV.md "3a" / step 2: certified=True is written ONLY from a whole-program
    proof (here) or a confirmed replay (the next test) -- never by anything else."""
    p = _prog(tmp_path, 'true.py', """
        d = {'a': 1}
        if __name__ == "__main__":
            print(d['a'])
    """)
    calls = []
    original = scheduler.ObligationTable.discharge

    def spy(self, classes, component):
        calls.append((classes, component))
        return original(self, classes, component)

    monkeypatch.setattr(scheduler.ObligationTable, 'discharge', spy)
    v = engine.verify(p, ('KeyError',), budget_seconds=60)
    assert v.verdict == 'true'
    assert calls and calls[0][0] == ('KeyError',) and calls[0][1] == v.component


def test_a_false_verdict_writes_a_certified_violation_at_the_witness_line(tmp_path, monkeypatch):
    p = _prog(tmp_path, 'false.py', """
        d = {}
        if __name__ == "__main__":
            print(d['missing'])
    """)
    calls = []
    original = scheduler.ObligationTable.violate

    def spy(self, exception, line, component, column=None):
        calls.append((exception, line, component, column))
        return original(self, exception, line, component, column=column)

    monkeypatch.setattr(scheduler.ObligationTable, 'violate', spy)
    # This concrete first run checks bookkeeping independently of host wall time.
    v = engine.verify(p, ('KeyError',), budget_seconds=60, run_timeout=30,
                      jobs=1, cpu_time_only=True)
    assert v.verdict == 'false'
    assert calls and calls[0][0] == 'KeyError' and calls[0][2] == v.component
    assert calls[0][1] == v.witness['site']['line']


def test_step2_writes_do_not_change_the_verdict(tmp_path, monkeypatch):
    """Discharge and violation writes record decisions already made: disabling both writers
    preserves the verdict and proof component. Independent calls need not perform the
    same number of executions within their timing budgets."""
    p = _prog(tmp_path, 'nochange.py', """
        d = {'a': 1}
        if __name__ == "__main__":
            print(d['a'])
    """)
    v1 = engine.verify(p, ('KeyError',), budget_seconds=60)
    monkeypatch.setattr(scheduler.ObligationTable, 'discharge', lambda *a, **k: None)
    monkeypatch.setattr(scheduler.ObligationTable, 'violate', lambda *a, **k: None)
    v2 = engine.verify(p, ('KeyError',), budget_seconds=60)
    assert v1.line() == v2.line() and v1.component == v2.component


def test_a_true_verdict_writes_uncertified_candidate_invariants(tmp_path, monkeypatch):
    """[internal ref] step 2's fact-kind taxonomy addendum, priority 1: finish() already builds
    the candidate list before checking it; writing one uncertified Fact per candidate
    there records what was PROPOSED, independent of whether it was proved."""
    p = _prog(tmp_path, 'candidates.py', """
        d = {'a': 1}
        if __name__ == "__main__":
            print(d['a'])
    """)
    calls = []
    from tharkun.sv import proof as sv_proof
    original = sv_proof.finish

    def spy(*a, **k):
        v = original(*a, **k)
        calls.append(k.get('knowledge'))
        return v

    monkeypatch.setattr(sv_proof, 'finish', spy)
    v = engine.verify(p, ('KeyError',), budget_seconds=60)
    assert v.verdict == 'true'
    assert calls and calls[0] is not None
    facts = [f for f in calls[0] if f.kind == 'candidate_invariant']
    assert facts and all(f.certified is False for f in facts)


# --- [internal ref]-followup: "input_sensitivity" -- which draw position(s) move which guard ----

def test_flip_draw_produces_a_different_same_kind_value():
    assert scheduler._flip_draw({'kind': 'bool', 'value': False}) == {'kind': 'bool', 'value': True}
    assert scheduler._flip_draw({'kind': 'bool', 'value': True}) == {'kind': 'bool', 'value': False}
    assert scheduler._flip_draw({'kind': 'int', 'value': 5}) == {'kind': 'int', 'value': 6}
    flipped = scheduler._flip_draw({'kind': 'float', 'value': (1.5).hex()})
    assert flipped['kind'] == 'float' and float.fromhex(flipped['value']) == 2.5
    assert scheduler._flip_draw({'kind': 'rindex', 'value': 0, 'n': 3}) == {'kind': 'rindex', 'value': 1, 'n': 3}
    # wraps rather than going out of [0, n)
    assert scheduler._flip_draw({'kind': 'rindex', 'value': 2, 'n': 3}) == {'kind': 'rindex', 'value': 0, 'n': 3}
    assert scheduler._flip_draw({'kind': 'str', 'value': 'ab'}) == {'kind': 'str', 'value': 'abx'}


def test_write_input_sensitivity_records_a_hint_only_when_the_outcome_moves():
    kb = KnowledgeBase()
    first = engine.Outcome('clean', draws=[{'kind': 'bool', 'value': False, 'site': 'p.py:2'},
                                            {'kind': 'int', 'value': 0, 'site': 'p.py:3'}])
    # flipping position 0 (bool) changes the outcome; flipping position 1 (int) does not
    outcomes = {0: engine.Outcome('raised', exc='KeyError', mro=['KeyError'], site={'line': 4}),
                1: engine.Outcome('clean')}

    def fake_run(vector):
        # the LAST entry is the one _write_input_sensitivity flipped
        return outcomes[len(vector) - 1]

    scheduler._write_input_sensitivity(kb, 'p.py', 'python3', first, fake_run)
    facts = [f for f in kb if f.kind == 'input_sensitivity']
    assert len(facts) == 1
    assert facts[0].certified is False
    assert facts[0].subject == 'draw0@p.py:2'


def test_write_input_sensitivity_does_nothing_without_draws():
    kb = KnowledgeBase()
    scheduler._write_input_sensitivity(kb, 'p.py', 'python3', engine.Outcome('clean'),
                                       lambda vector: pytest.fail('must not run without draws'))
    assert not list(kb)


def test_a_draw_that_actually_moves_the_outcome_is_recorded_end_to_end(tmp_path, monkeypatch):
    """Real program, real replay (no mocking run_once): the first run draws the bool
    default (False) and stays clean; flipping it to True is the ONLY way to reach the
    raise, so the differencing pass must record exactly that position as sensitive."""
    p = _prog(tmp_path, 'sensitive.py', """
        from _sv_verifier import nondet_bool
        if __name__ == "__main__":
            flag = nondet_bool()
            if flag:
                raise KeyError()
    """)
    seen_kbs = []
    original = scheduler._write_input_sensitivity

    def spy(kb, program, python, out, run, **kwargs):
        result = original(kb, program, python, out, run, **kwargs)
        seen_kbs.append(list(kb))
        return result

    monkeypatch.setattr(scheduler, '_write_input_sensitivity', spy)
    engine.verify(p, ('KeyError',), budget_seconds=60, run_timeout=30, prove=False, concolic=False)
    assert seen_kbs, "the differencing pass must run on the first (non-terminal) outcome"
    sensitivity = [f for f in seen_kbs[0] if f.kind == 'input_sensitivity']
    assert sensitivity, "flipping the only draw is the sole way to reach the raise -- must be recorded"
    assert all(f.certified is False for f in sensitivity)
