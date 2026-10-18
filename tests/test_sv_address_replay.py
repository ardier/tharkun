"""[internal ref] ([internal ref]): a FALSE witness must not be reported for a violation whose
outcome depends on a memory address unless it survives repeated fresh replay.

The motivating case (lane C1's step-5 sweep, [internal ref] notes): ``rc_id_in_list.py``
(``funcs = [id]; x = funcs[0](object()) % 3; print([0, 0][x])``) reproduces its IndexError on
a fresh interpreter only 6 times in 20 independent replays -- the default ``id()``/
``object.__hash__`` value is ASLR/allocator-dependent, and CPython gives no portable,
recordable control over it (unlike ``PYTHONHASHSEED`` for string/bytes hashing). A witness
only pins what the engine controls (the draw vector); reporting FALSE here handed a validator
a witness that, most of the time, would NOT replay for them.

``sv/scheduler.py::run``'s ``found()`` now gates every FALSE candidate the [internal ref] determinism
census (``solve/determinism.py::determinism_census``) cannot certify address-independent
behind ``sv/engine.py::_REPLAY_CONFIRM_K`` (20) consecutive fresh replays that must ALL
reproduce the same exception -- see that constant's docstring for the k=20 justification. A
program the census DOES certify skips the k-of-k loop, but still needs the universal
validator replay added by [internal ref].

Folded in ([internal ref], same area): every replay here, and ``sv/engine.py::replay()``
itself, must REJECT a witness whose vector length differs from the number of draws the
replayed run actually consumes -- ``sv/engine.py::_replayed`` is the same length/kind/value
check ``eregion_sv.replay.validate`` (the real grader) makes, so tharkun's own self-check is
never weaker than what a validator would accept. It is not live on any known task today
(witnesses are built from the draws a run actually made, so they roundtrip by construction);
the tests below cover it directly with a hand-built short and long vector.

Also folded in (MG, PR 97 review): the k-of-k loop must vary ``hashseed`` across its own
replays instead of repeating the default 0. A violation whose value depends on STRING/BYTES
hash order (``next(iter(a_str_set))``, ``list(a_str_set)``) is perfectly reproducible k/k
times under a FIXED hashseed, even though a validator inherits whatever hashseed its own
environment gives it -- usually not 0. MG's two attacks (``h01``/``h04`` below, PYTHONHASHSEED-
dependent set-iteration-order violations) reproduced 20/20 under the unfixed loop despite
failing under most real hashseeds; both are covered directly below.
"""
from __future__ import annotations

import re
import textwrap

from tharkun.sv import engine
from tharkun.sv import replay as sv_replay

ADDR_ID_SHAPE = """
    funcs = [id]
    x = funcs[0](object()) % 3
    if __name__ == "__main__":
        print([0, 0][x])
"""


def _prog(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


# --- the fast path skips k-of-k but still pays for validator replay -------------------------

def test_census_clean_false_needs_validator_replay(tmp_path, monkeypatch):
    p = _prog(tmp_path, 'det.py', """
        xs = [1, 2, 3]
        if __name__ == "__main__":
            print(xs[5])
    """)
    calls = []
    real_run_once = engine.run_once

    def counting_run_once(*a, **k):
        calls.append(1)
        return real_run_once(*a, **k)

    monkeypatch.setattr(engine, 'run_once', counting_run_once)
    v = engine.verify(p, ('IndexError',), budget_seconds=60)
    assert v.line() == 'FALSE(IndexError)'
    assert len(calls) == 2, f"expected discovery and validator replay, got {len(calls)}"


# --- the gate itself: deterministic via monkeypatch, never flaky ---------------------------

def test_address_dependent_violation_that_fails_one_replay_is_unknown_not_false(tmp_path, monkeypatch):
    """A synthetic FALSE candidate whose replay does NOT reproduce a second time: the gate
    must refuse it (UNKNOWN), and never report the unconfirmed FALSE."""
    p = _prog(tmp_path, 'addr.py', ADDR_ID_SHAPE)
    calls = []

    def fake_run_once(python, program, vector, **kwargs):
        calls.append(vector)
        if len(calls) == 1:
            return engine.Outcome('raised', 'IndexError', ['IndexError'], site={'file': 'addr.py'})
        return engine.Outcome('clean')

    monkeypatch.setattr(engine, 'run_once', fake_run_once)
    v = engine.verify(p, ('IndexError',), budget_seconds=60, prove=False, concolic=False)
    assert v.verdict == 'unknown'
    assert 'not proven address-independent' in v.reason
    assert 'replay did not reproduce it' in v.reason
    # gives up on the FIRST failing replay -- the discovery run (1) plus one failed
    # confirmation (2), not all the way to k
    assert len(calls) == 2, f"expected the gate to stop after the first failed replay, got {len(calls)} calls"


def test_address_dependent_violation_confirmed_k_of_k_is_false(tmp_path, monkeypatch):
    """The same shape, but EVERY replay happens to reproduce: accepted, and exactly
    ``_REPLAY_CONFIRM_K`` address trials plus the mandatory validator replay."""
    p = _prog(tmp_path, 'addr.py', ADDR_ID_SHAPE)
    calls = []

    def fake_run_once(python, program, vector, **kwargs):
        calls.append(vector)
        return engine.Outcome('raised', 'IndexError', ['IndexError'], site={'file': 'addr.py'})

    monkeypatch.setattr(engine, 'run_once', fake_run_once)
    v = engine.verify(p, ('IndexError',), budget_seconds=60, prove=False, concolic=False)
    assert v.line() == 'FALSE(IndexError)'
    assert len(calls) == engine._REPLAY_CONFIRM_K + 1


def test_a_different_exception_on_replay_is_not_a_confirmation(tmp_path, monkeypatch):
    """A replay that raises but NOT the same property class must count as a failed
    confirmation, not a pass-through."""
    p = _prog(tmp_path, 'addr.py', ADDR_ID_SHAPE)
    calls = []

    def fake_run_once(python, program, vector, **kwargs):
        calls.append(vector)
        if len(calls) == 1:
            return engine.Outcome('raised', 'IndexError', ['IndexError'], site={'file': 'addr.py'})
        return engine.Outcome('raised', 'KeyError', ['KeyError'], site={'file': 'addr.py'})

    monkeypatch.setattr(engine, 'run_once', fake_run_once)
    v = engine.verify(p, ('IndexError',), budget_seconds=60, prove=False, concolic=False)
    assert v.verdict == 'unknown'
    assert len(calls) == 2


def test_budget_exhaustion_during_confirmation_is_unknown_not_false(tmp_path, monkeypatch):
    """A replay loop that would need more CPU than remains concedes UNKNOWN -- it must never
    guess FALSE from a partial run of confirmations."""
    p = _prog(tmp_path, 'addr.py', ADDR_ID_SHAPE)
    calls = []
    clock = [0.0]

    def fake_monotonic():
        return clock[0]

    def fake_run_once(python, program, vector, **kwargs):
        calls.append(1)
        clock[0] += 1000.0  # each fake run burns far more than the budget
        return engine.Outcome('raised', 'IndexError', ['IndexError'], site={'file': 'addr.py'})

    monkeypatch.setattr(engine.time, 'monotonic', fake_monotonic)
    monkeypatch.setattr(engine, 'run_once', fake_run_once)
    v = engine.verify(p, ('IndexError',), budget_seconds=5, prove=False, concolic=False)
    assert v.verdict == 'unknown'
    assert 0 < len(calls) < engine._REPLAY_CONFIRM_K, (
        f"expected the gate to stop early once the budget ran out, got {len(calls)} calls")
    assert 'budget ran out' in v.reason


def test_hash_of_object_index_is_gated_too(tmp_path, monkeypatch):
    """Not just id(): a default object hash used as an index hits the same gate, denied
    directly by determinism_census's rule 2 ([internal ref]'s denied-builtins rule)."""
    p = _prog(tmp_path, 'addr_hash.py', """
        class P:
            pass
        x = hash(P()) % 3
        if __name__ == "__main__":
            print([0, 0][x])
    """)
    calls = []

    def fake_run_once(python, program, vector, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            return engine.Outcome('raised', 'IndexError', ['IndexError'], site={'file': 'addr_hash.py'})
        return engine.Outcome('clean')

    monkeypatch.setattr(engine, 'run_once', fake_run_once)
    v = engine.verify(p, ('IndexError',), budget_seconds=60, prove=False, concolic=False)
    assert v.verdict == 'unknown'
    assert 'not proven address-independent' in v.reason


# --- end to end, real subprocesses, no mocks: the exact [internal ref] shape ------------

def test_rc_id_in_list_shape_end_to_end_never_gives_an_unreplayable_false(tmp_path):
    """The exact shape from [internal ref] (``rc_id_in_list.py``), run for real: whatever
    verdict comes back must be SOUND. If it is FALSE, an independent fresh replay of its own
    witness -- right now, exactly what a validator would do -- must confirm it; UNKNOWN is
    always an acceptable answer for this program."""
    p = _prog(tmp_path, 'rc_id_in_list.py', ADDR_ID_SHAPE)
    v = engine.verify(p, ('IndexError',), budget_seconds=60)
    if v.verdict == 'false':
        ok, out = sv_replay(p, v.witness, ('IndexError',))
        assert ok, f"accepted FALSE witness does not independently replay: {out}"
    else:
        assert v.verdict == 'unknown', f"unexpected verdict {v.line()}"


# --- [internal ref]: a witness must consume exactly its own vector, not just end in the
# claimed exception -- matching eregion_sv.replay.validate's own length/kind/value check ----

def test_replayed_helper_matches_the_graders_rule():
    """``_replayed`` is a pure function: exercise its four cases directly, no subprocess."""
    from tharkun.sv.engine import Outcome, _replayed
    vector = [{"kind": "int", "value": 1}, {"kind": "int", "value": 2}]

    def outcome(draws):
        return Outcome('raised', 'IndexError', ['IndexError'], draws=draws)

    same = outcome([{"kind": "int", "value": 1, "site": "x:1"}, {"kind": "int", "value": 2, "site": "x:2"}])
    short = outcome([{"kind": "int", "value": 1, "site": "x:1"}])
    long_ = outcome([{"kind": "int", "value": 1, "site": "x:1"}, {"kind": "int", "value": 2, "site": "x:2"},
                      {"kind": "int", "value": 3, "site": "x:3"}])
    different_value = outcome([{"kind": "int", "value": 9, "site": "x:1"}, {"kind": "int", "value": 2, "site": "x:2"}])

    assert _replayed(vector, same)
    assert not _replayed(vector, short)
    assert not _replayed(vector, long_)
    assert not _replayed(vector, different_value)


def test_replay_rejects_a_witness_vector_shorter_than_what_the_run_consumes(tmp_path):
    """The run needs TWO draws; a one-entry vector still ends in the claimed exception (an
    empty list raises IndexError for any index at all), but it does not actually replay the
    recorded vector -- ``eregion_sv.replay.validate`` would refuse this witness too."""
    p = _prog(tmp_path, 'short.py', """
        import _sv_verifier
        if __name__ == "__main__":
            a = _sv_verifier.nondet_int()
            b = _sv_verifier.nondet_int()
            xs = []
            print(xs[b])
    """)
    witness = {"task": p, "exception": "IndexError", "vector": [{"kind": "int", "value": 0}]}
    ok, out = engine.replay(p, witness, ("IndexError",))
    assert not ok
    assert out.exc == "IndexError"       # the run DID raise the claimed exception ...
    assert len(out.draws) == 2           # ... but it consumed 2 draws, not the 1 given


def test_replay_rejects_a_witness_vector_longer_than_what_the_run_consumes(tmp_path):
    """The run needs only ONE draw; a two-entry vector's second entry is never consumed."""
    p = _prog(tmp_path, 'long.py', """
        import _sv_verifier
        if __name__ == "__main__":
            a = _sv_verifier.nondet_int()
            xs = []
            print(xs[a])
    """)
    witness = {"task": p, "exception": "IndexError",
               "vector": [{"kind": "int", "value": 0}, {"kind": "int", "value": 5}]}
    ok, out = engine.replay(p, witness, ("IndexError",))
    assert not ok
    assert out.exc == "IndexError"
    assert len(out.draws) == 1


def test_gate_rejects_a_confirmation_replay_with_a_different_vector_length(tmp_path, monkeypatch):
    """The k-of-k gate (``found()``'s fresh-replay requirement) must apply the same check:
    a confirmation run that raises the claimed exception but consumed a DIFFERENT number of
    draws than the witness's own vector is not a real replay of it."""
    p = _prog(tmp_path, 'addr.py', ADDR_ID_SHAPE)
    calls = []

    def fake_run_once(python, program, vector, **kwargs):
        calls.append(vector)
        if len(calls) == 1:
            return engine.Outcome('raised', 'IndexError', ['IndexError'], site={'file': 'addr.py'})
        # same exception, but this confirmation run drew something the empty witness vector
        # ([], since this program forces no draws) never recorded
        return engine.Outcome('raised', 'IndexError', ['IndexError'], site={'file': 'addr.py'},
                              draws=[{'kind': 'int', 'value': 0}])

    monkeypatch.setattr(engine, 'run_once', fake_run_once)
    v = engine.verify(p, ('IndexError',), budget_seconds=60, prove=False, concolic=False)
    assert v.verdict == 'unknown'
    assert len(calls) == 2


# --- MG (PR 97 review): the k-of-k loop must vary hashseed, or hash-order-dependent -------
# violations reproduce k/k under a fixed hashseed while failing under a validator's own ----

def test_k_of_k_loop_varies_hashseed_per_replay(tmp_path, monkeypatch):
    """Every k-of-k replay must use a DIFFERENT, non-zero hashseed -- reusing the default 0
    (like the discovery run) would let a hash-order-dependent witness "confirm" trivially."""
    p = _prog(tmp_path, 'addr.py', ADDR_ID_SHAPE)
    seen_hashseeds = []

    def fake_run_once(python, program, vector, *, hashseed=0, **kwargs):
        seen_hashseeds.append(hashseed)
        return engine.Outcome('raised', 'IndexError', ['IndexError'], site={'file': 'addr.py'})

    monkeypatch.setattr(engine, 'run_once', fake_run_once)
    v = engine.verify(p, ('IndexError',), budget_seconds=60, prove=False, concolic=False)
    assert v.line() == 'FALSE(IndexError)'
    # seen_hashseeds[0] is the discovery run (always the default 0, not part of this loop);
    # the final seed-0 run is the separate mandatory validator replay.
    assert seen_hashseeds[-1] == 0
    confirm_seeds = seen_hashseeds[1:-1]
    assert len(confirm_seeds) == engine._REPLAY_CONFIRM_K - 1
    assert all(s != 0 for s in confirm_seeds), f"a k-of-k replay used the default hashseed 0: {confirm_seeds}"
    assert len(set(confirm_seeds)) == len(confirm_seeds), f"k-of-k replays reused a hashseed: {confirm_seeds}"


def test_h01_strset_first_never_wrongly_false(tmp_path):
    """MG's atk97/h01: a str set's iteration order is PYTHONHASHSEED-dependent. Under the
    fixed hashseed=0 the discovery run always uses, next(iter({'alpha','beta','gamma'}))
    always gives 'alpha' -- a validator under a different (usually random) hashseed very
    likely will not. Must be UNKNOWN, not FALSE."""
    p = _prog(tmp_path, 'h01.py', """
        def main():
            s = {"alpha", "beta", "gamma"}
            x = next(iter(s))
            if x == "alpha":
                [][0]
        main()
    """)
    v = engine.verify(p, ("IndexError",), budget_seconds=60)
    assert v.verdict == 'unknown', f"unexpected verdict {v.line()}"


def test_h04_strset_with_draw_never_wrongly_false(tmp_path):
    """MG's atk97/h04: same mechanism as h01, reached behind a controlled draw. Must be
    UNKNOWN, not FALSE."""
    p = _prog(tmp_path, 'h04.py', """
        import _sv_verifier
        def main():
            n = _sv_verifier.nondet_int()
            s = {"u", "v", "w"}
            if n == 1 and list(s)[0] == "u":
                [][0]
        main()
    """)
    v = engine.verify(p, ("IndexError",), budget_seconds=60)
    assert v.verdict == 'unknown', f"unexpected verdict {v.line()}"


# --- MG (PR 97 review): every reason text this code can produce must survive --------
# eregion_sv.soundness's gate parser -----------------------------------------------
#
# __main__.py prints "tharkun-sv: {v.runs} runs in {v.seconds:.1f}s; {v.reason}", a line
# that does NOT match that parser's _DIAGNOSTIC ("tharkun-sv: <lowercase-phase>: " or
# "tharkun-sv: witness {") because it starts with a digit (the run count), not a phase
# name. A reason containing a bare TRUE/FALSE/UNKNOWN/ERROR word (case-insensitive,
# \b-bounded) then trips the parser's "malformed or embedded verdict token" check --
# exactly what happened on det_id_via_vars.py, which intermittently printed
# "tharkun-sv: 2 runs in 0.1s; FALSE witness not proven address-independent ...".
# eregion-sv is a separate repo, not importable from tharkun's own test env, so this
# mirrors its two regexes by hand (soundness.py's own comment: "Wrapper verdict
# grammar, with balanced reasons") -- keep these in sync by hand if that grammar ever
# changes.
_VERDICT_TOKEN = re.compile(r'\b(?:TRUE|FALSE|UNKNOWN|ERROR)\b', re.IGNORECASE)
_DIAGNOSTIC = re.compile(r'tharkun-sv: (?:[a-z][a-z0-9_-]*: |witness \{)')


def _assert_gate_safe(line: str) -> None:
    assert not (_VERDICT_TOKEN.search(line) and not _DIAGNOSTIC.match(line)), (
        f"line would break eregion_sv.soundness._parse_verdict: {line!r}")


def test_fresh_replay_confirm_reason_survives_the_soundness_gate_parser(tmp_path, monkeypatch):
    """MG's exact repro shape: a discovery run that raises once, then a failing
    confirmation -- the UNKNOWN reason this produces must not break the gate."""
    p = _prog(tmp_path, 'addr.py', ADDR_ID_SHAPE)
    calls = []

    def fake_run_once(python, program, vector, **kwargs):
        calls.append(vector)
        if len(calls) == 1:
            return engine.Outcome('raised', 'IndexError', ['IndexError'], site={'file': 'addr.py'})
        return engine.Outcome('clean')

    monkeypatch.setattr(engine, 'run_once', fake_run_once)
    v = engine.verify(p, ('IndexError',), budget_seconds=60, prove=False, concolic=False)
    assert v.verdict == 'unknown'
    _assert_gate_safe(f"tharkun-sv: {v.runs} runs in {v.seconds:.1f}s; {v.reason}")


def test_k_of_k_budget_exhausted_reason_survives_the_soundness_gate_parser(tmp_path, monkeypatch):
    p = _prog(tmp_path, 'addr.py', ADDR_ID_SHAPE)
    clock = [0.0]

    def fake_run_once(python, program, vector, **kwargs):
        clock[0] += 1000.0
        return engine.Outcome('raised', 'IndexError', ['IndexError'], site={'file': 'addr.py'})

    monkeypatch.setattr(engine.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(engine, 'run_once', fake_run_once)
    v = engine.verify(p, ('IndexError',), budget_seconds=5, prove=False, concolic=False)
    assert v.verdict == 'unknown'
    _assert_gate_safe(f"tharkun-sv: {v.runs} runs in {v.seconds:.1f}s; {v.reason}")


def test_all_reworded_reason_literals_survive_the_soundness_gate_parser():
    """The literal strings [internal ref]/PR-97-review touched, checked directly -- some of
    these paths (a fork-derived confirm failure, a certifier-unsound detection) are
    impractical to trigger live through the public verify() API in a unit test."""
    samples = [
        "violation candidate failed fresh replay",
        "violation witness not proven address-independent (does not resolve: line 4: "
        "uses __builtins__) and the budget ran out before 20/20 replay confirmations "
        "(3/20 so far)",
        "violation witness not proven address-independent (line 2: uses id) and replay "
        "did not reproduce it on every attempt (1/20 confirmed)",
    ]
    for reason in samples:
        _assert_gate_safe(f"tharkun-sv: 3 runs in 0.2s; {reason}")
    _assert_gate_safe("tharkun-sv: certifier-unsound: proof:no-site said the property "
                      "holds; a run violates IndexError")


# --- MG's second finding: env-read exemption must be a SEPARATE, independent census re-run,
# never inferred from the original census's single (first-only) reported reason -----------

def test_env_read_then_address_dependent_construct_still_gated(tmp_path):
    """MG's atk97/h05: sys.stdin.read() THEN a str-set order read. The ORIGINAL census
    reports only its FIRST refusal ("uses sys.stdin") -- a naive fix that skipped k-of-k on
    seeing that reason alone would wrongly trust the SEPARATE, genuinely address/order-
    dependent construct later in the same function. Must stay UNKNOWN, never FALSE."""
    p = _prog(tmp_path, 'h05.py', """
        import sys
        def main():
            data = sys.stdin.read()
            s = {"alpha", "beta", "gamma"}
            x = next(iter(s))
            if x == "alpha":
                [][0]
        main()
    """)
    v = engine.verify(p, ("IndexError",), budget_seconds=60)
    assert v.verdict == 'unknown', f"unexpected verdict {v.line()}"


def test_address_dependent_construct_then_env_read_still_gated(tmp_path):
    """MG's atk97/h06: the SAME shape with the order reversed (str-set order read first,
    then an argv read) -- the original census's first-hit reason differs, but the result
    must be identical: still UNKNOWN, never FALSE."""
    p = _prog(tmp_path, 'h06.py', """
        import sys
        def main():
            s = {"alpha", "beta", "gamma"}
            x = next(iter(s))
            n = len(sys.argv)
            if x == "alpha" and n >= 1:
                [][0]
        main()
    """)
    v = engine.verify(p, ("IndexError",), budget_seconds=60)
    assert v.verdict == 'unknown', f"unexpected verdict {v.line()}"


def test_pure_env_read_still_skips_k_of_k(tmp_path, monkeypatch):
    """The actual fix this hotfix is for: a program whose ONLY census refusal is an
    environment read (nothing else wrong) must still skip the 20-replay confirmation
    loop, paying only for discovery and validator replay. argv is read (tripping the census)
    but its VALUE is never used -- the violation
    is unconditional, isolating the fast-path claim from argv's actual content."""
    p = _prog(tmp_path, 'argv_only.py', """
        import sys
        _ = sys.argv
        xs = [1, 2]
        print(xs[5])
    """)
    calls = []
    real_run_once = engine.run_once

    def counting_run_once(*a, **k):
        calls.append(1)
        return real_run_once(*a, **k)

    monkeypatch.setattr(engine, 'run_once', counting_run_once)
    v = engine.verify(p, ('IndexError',), budget_seconds=60)
    assert v.line() == 'FALSE(IndexError)'
    assert len(calls) == 2, f"expected discovery and validator replay (env-read fast path), got {len(calls)}"
