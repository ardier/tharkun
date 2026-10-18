"""A hybrid finding is a draw vector whose fresh SV replay raises the property."""
import importlib.util
import json
import math
import os
import sys
import textwrap
from pathlib import Path

import pytest

from tharkun.hybrid.fdp_codec import atheris_available
from tharkun.sv.engine import replay, run_once, verify


def _prog(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_obligation_search_replays_fresh(tmp_path):
    pytest.importorskip("tharkun.solve.obligations")  # lane OB is merged separately
    program = tmp_path / "divide.py"
    program.write_text("import _sv_verifier\nif __name__ == \"__main__\":\n"
                       "    n = _sv_verifier.nondet_int()\n    if n > 50:\n"
                       "        b = n - 75\n        a = 1 / b\n")
    verdict = verify(str(program), ("ZeroDivisionError",), budget_seconds=60,
                     jobs=1, prove=False, concolic=False, stages=["hybrid"])
    assert verdict.verdict == "false"
    # the finding is replayed fresh on the real program (the claim); WHICH stage got there first depends on how much of the
    # budget the fuzzer was given, so the component is not asserted (under load the program's own constant 75 is enumerated
    # first)
    assert verdict.component.startswith(("hybrid:", "enum:"))
    assert replay(str(program), verdict.witness, ("ZeroDivisionError",))[0]


# --- [internal ref]: nondet_float() proposals are the four values the helper produces ---------------

@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_the_worker_proposes_only_values_nondet_float_can_produce(tmp_path, monkeypatch):
    from tharkun.sv import _child

    worker = _load_hybrid_worker(tmp_path, monkeypatch)
    assert len(worker._FLOAT_VALUES) == 4 and all(_child.float_producible(v) for v in worker._FLOAT_VALUES)
    assert {repr(v) for v in worker._FLOAT_VALUES} == {"inf", "-inf", "nan", "-0.0"}


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_a_nan_only_violation_is_found_by_the_hybrid_stage(tmp_path):
    program = tmp_path / "nan_only.py"
    program.write_text("import _sv_verifier\nif __name__ == \"__main__\":\n"
                       "    x = _sv_verifier.nondet_float()\n    if x != x:\n        raise IndexError()\n")
    verdict = verify(str(program), ("IndexError",), budget_seconds=60, jobs=1, prove=False, concolic=False, stages=["hybrid"])
    assert verdict.verdict == "false", verdict.line()
    assert replay(str(program), verdict.witness, ("IndexError",))[0]


# --- [internal ref]-followup: a huge-magnitude proposal is clamped BEFORE it is ever logged -----

def _load_hybrid_worker(tmp_path, monkeypatch, program_text="if __name__ == '__main__':\n    pass\n",
                        classes=()):
    """``_hybrid_worker.py`` reads its whole request (target program, checks, ...) from
    ``THARKUN_SV_HYBRID_REQ`` at IMPORT time and hard-imports ``atheris`` -- loaded fresh
    from its own file (never via the package import cache, so this never pollutes a real
    worker subprocess's state) with a minimal request (by default an otherwise-inert program)."""
    program = tmp_path / "inert.py"
    program.write_text(program_text)
    (tmp_path / "found").mkdir()
    req = tmp_path / "req.json"
    req.write_text(json.dumps({"program": str(program), "checks": {}, "classes": list(classes),
                               "metrics": {}, "found": str(tmp_path / "found")}))
    monkeypatch.setenv("THARKUN_SV_HYBRID_REQ", str(req))
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "src", "tharkun", "sv", "_hybrid_worker.py")
    spec = importlib.util.spec_from_file_location("_hybrid_worker_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_huge_float_proposal_is_clamped_before_it_would_be_logged(tmp_path, monkeypatch):
    """The exact live repro (MAIN 2026-09-28, bm_chaos_nondet/data-lookup): Atheris
    proposed 4.15e300 for ``nondet_float_range(0, 1)``. The raw proposal used to be
    logged verbatim (out of domain -- [internal ref]'s _child.py fix would then correctly refuse
    to confirm it, silently losing a real finding); it must now be snapped into the call's
    domain before logging.  [internal ref]: the domain is what ``random.uniform(a, b)`` can return, so
    the clamp of a huge proposal is the LARGEST producible value, ``1 - 2**-53`` (the bound
    ``1.0`` itself is refused)."""
    worker = _load_hybrid_worker(tmp_path, monkeypatch)
    huge = 4.15e300
    top = 1.0 - 2.0 ** -53
    clamped = worker._clamp_float_range(huge, 0.0, 1.0)
    assert clamped == top
    assert worker._clamp_float_range(-huge, 0.0, 1.0) == 0.0
    assert worker._clamp_float_range(0.5, 0.0, 1.0) == 0.5
    assert worker._clamp_float_range(math.nan, 0.0, 1.0) == 0.0
    # order-independent: a call site with a > b snaps the same way (its domain runs from 1.0 down)
    assert worker._clamp_float_range(huge, 1.0, 0.0) == 1.0
    from tharkun.sv import _child
    for v in (clamped, 0.5, worker._clamp_float_range(0.1, 0.0, 1.0), worker._clamp_float_range(0.3, 5.0, 10.0)):
        assert _child.float_range_producible(0.0, 1.0, v) or v >= 5.0


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_the_clamped_witness_is_in_range_and_replays_confirmed(tmp_path, monkeypatch):
    """The recorded witness holds exactly what the program saw: a vector with the
    CLAMPED value at a nondet_float_range(0, 1) position replays as the value the
    program actually reaches, and confirms via Tharkun's own fresh replay -- the
    counterpart on the eregion-sv side (the grading shim) is exercised by the
    soundness-gate campaign, not reachable from this repo's own tests."""
    worker = _load_hybrid_worker(tmp_path, monkeypatch)
    program = tmp_path / "range_bug.py"
    program.write_text(
        "import _sv_verifier\n"
        "if __name__ == '__main__':\n"
        "    t = _sv_verifier.nondet_float_range(0.0, 1.0)\n"
        "    if t > 0.9999999999999:\n"
        "        raise IndexError()\n"
    )
    clamped = worker._clamp_float_range(4.15e300, 0.0, 1.0)
    vector = [{"kind": "float", "value": clamped.hex()}]
    out = run_once(sys.executable, str(program), vector, timeout=30)
    assert out.status == "raised" and out.exc == "IndexError"
    assert out.invalid_draw is False
    confirmed, _ = replay(str(program), {"vector": vector}, ("IndexError",))
    assert confirmed is True


# --- the actual live repro (MAIN 2026-09-28): bm_chaos_nondet/data-lookup ---------------

CHAOS_NONDET = Path(__file__).parent / "fixtures" / "bm_chaos_nondet.py"


def test_bm_chaos_nondet_raw_huge_proposal_is_invalid_not_a_wrong_false():
    """Before clamping anywhere: the SV-Benchmarks task itself
    (``tests/fixtures/bm_chaos_nondet.py``, copied verbatim from the real corpus, MIT
    licensed), fed the exact raw Atheris proposal (4.15e300) for its own
    ``nondet_float_range(0, 1)`` call (line 245, ``args_thickness``) with nothing else
    drawn. [internal ref]'s _child.py fix refuses it outright -- this is the wrong-FALSE [internal ref]
    exists to prevent, confirmed against the real task, not a toy."""
    huge = 4.15e300
    out = run_once(sys.executable, str(CHAOS_NONDET), [{"kind": "float", "value": huge.hex()}],
                   timeout=30)
    assert out.status == "raised" and out.exc == "ValueError"
    assert out.invalid_draw is True


def test_bm_chaos_nondet_clamped_proposal_is_the_real_confirmable_false(tmp_path, monkeypatch):
    """MAIN 2026-09-28: the DETERMINISTIC evidence -- feed the exact huge proposal
    Atheris drew for ``nondet_float_range(0, 1)`` on the real
    ``bm_chaos_nondet``/``data-lookup`` task, clamped exactly as [internal ref]-followup's
    ``_clamp_float_range`` would before ever logging it, and confirm the resulting
    witness is in range and replays CONFIRMED -- through Tharkun's own fresh replay
    always, and through ``eregion_sv.replay`` too when that sibling repo is importable
    (best-effort: this repo's own test suite has no standing dependency on it; the
    soundness-gate campaign is the standing authority for that side, this is instead a
    convenience check that runs when the checkout happens to have it on the path,
    e.g. on a lane host with eregion-sv checked out next to tharkun)."""
    worker = _load_hybrid_worker(tmp_path, monkeypatch)
    huge = 4.15e300
    clamped = worker._clamp_float_range(huge, 0.0, 1.0)
    assert clamped == 1.0 - 2.0 ** -53       # [internal ref]: the largest value uniform(0, 1) returns; 1.0 itself is refused

    # only the ONE position (thickness) is given -- every later draw (width, height,
    # iterations, filename, rng_seed) falls back to its kind's default, exactly like an
    # unguided Atheris/enumeration proposal that only cared about this one call site.
    out = run_once(sys.executable, str(CHAOS_NONDET), [{"kind": "float", "value": clamped.hex()}],
                   timeout=30)
    assert out.status == "raised" and out.exc == "IndexError", (
        "the real task's own bug (an im[x][h-y-1] index out of range once thickness "
        "is pinned to the domain's own upper bound, everything else default) must "
        "still be reachable through the CLAMPED value"
    )
    assert out.invalid_draw is False, "a clamped, in-domain draw must never be invalid"
    assert out.draws[0]["kind"] == "float"
    recorded_thickness = float.fromhex(out.draws[0]["value"])
    assert recorded_thickness == 1.0 - 2.0 ** -53, "the recorded witness must hold the CLAMPED value, not 4.15e300"
    # the bound itself is not a draw uniform(0, 1) can return: the same program with thickness 1.0 is refused, never a FALSE
    refused = run_once(sys.executable, str(CHAOS_NONDET), [{"kind": "float", "value": (1.0).hex()}], timeout=30)
    assert refused.invalid_draw is True and refused.violated(("IndexError",)) is None

    # the WITNESS is built from every draw the run actually consumed (engine.py::_witness's
    # own construction), not just the one position this test cared about -- a witness
    # missing a later draw the run also made is not a real recording of what happened.
    witness_vector = [{k: v for k, v in d.items() if k != "site"} for d in out.draws]

    confirmed, replayed = replay(str(CHAOS_NONDET), {"vector": witness_vector}, ("IndexError",))
    assert confirmed is True
    assert 0.0 <= float.fromhex(replayed.draws[0]["value"]) <= 1.0, \
        "the recorded witness must be in range"

    try:
        import eregion_sv.replay as _eregion_replay  # noqa: PLC0415 -- optional sibling repo
    except ImportError:
        pytest.skip("eregion_sv not on the path in this checkout -- soundness-gate covers it")
    eregion_confirmed, _ = _eregion_replay.validate(
        sys.executable, {"task": str(CHAOS_NONDET), "vector": witness_vector}, ("IndexError",))
    assert eregion_confirmed is True
def test_assert_fallback_when_census_refuses_the_whole_program(tmp_path):
    """docs/SV.md "3a": solve/census.py::obligation_census is sound only as a whole-
    program ALLOWLIST -- reachable computed reflection refuses the census for
    a class, even for an assert nowhere near it. sv/hybrid.py::_assert_fallback must
    still find the program's own assert directly from the AST."""
    from tharkun.solve.census import obligation_census
    from tharkun.sv.hybrid import _assert_fallback

    source = textwrap.dedent("""
        import _sv_verifier

        def f(s):
            return len(s) == 3

        s = _sv_verifier.nondet_str()
        name = "__name__"
        getattr(type(s), name)  # reachable computed reflection refuses the census
        assert f(s) == (len(s) == 3)
    """)
    census = obligation_census(source, ("AssertionError",), "prog.py")
    assert all(o.ast_path == "$coverage" for o in census.obligations), (
        "fixture must actually reproduce the whole-program census refusal"
    )
    fallback = _assert_fallback(source, "prog.py", ("AssertionError",))
    assert len(fallback) == 1
    assert fallback[0].exception == "AssertionError"
    assert fallback[0].invariant.value == "False"
    assert fallback[0].ast_path != "$coverage"


def test_assert_fallback_is_empty_when_assertionerror_is_not_the_property(tmp_path):
    from tharkun.sv.hybrid import _assert_fallback
    source = "assert 1 == 1\n"
    assert _assert_fallback(source, "prog.py", ("ZeroDivisionError",)) == []


def test_hybrid_starts_and_spends_budget_when_the_census_refuses(tmp_path):
    """End-to-end [internal ref] regression: before this fix, a program whose obligation census
    refuses (reachable computed reflection unrelated to the assert) never even started
    Atheris -- "0 obligations", no hybrid: log line, budget unspent. Now it starts."""
    pytest.importorskip("tharkun.solve.obligations")
    if not atheris_available():
        pytest.skip("Atheris is optional")
    p = _prog(tmp_path, 'refused.py', """
        import _sv_verifier

        def f(s):
            return len(s) == 3

        s = _sv_verifier.nondet_str()
        name = "__name__"
        getattr(type(s), name)  # reachable computed reflection refuses the census
        assert f(s) == (len(s) == 3)
    """)
    logs = []
    verdict = verify(p, ("AssertionError",), budget_seconds=8, jobs=1, prove=False,
                     concolic=False, stages=["hybrid"], log=logs.append)
    assert any("hybrid:" in m and "obligations" in m for m in logs)
    assert any("assert fallback" in m for m in logs)


# --- [internal ref]: regex-inverted / literal-seeded str candidates ------------------------------

def test_regex_patterns_found_from_re_compile_and_re_sub(tmp_path):
    from tharkun.sv.hybrid import _regex_patterns
    import ast
    source = textwrap.dedent("""
        import re
        TAG_RE = re.compile(r'<[a-z]+>')
        def strip(s):
            return re.sub(r'<[a-z]+>', '', s)
    """)
    tree = ast.parse(source, "prog.py")
    patterns = _regex_patterns(tree)
    assert r'<[a-z]+>' in patterns


def test_regex_vectors_substitute_only_str_positions_and_try_both_bools(tmp_path):
    import ast
    from tharkun.sv.engine import Outcome
    from tharkun.sv.hybrid import _regex_vectors
    from tharkun.sv.stages import Context

    source = textwrap.dedent("""
        import re
        import _sv_verifier
        s = _sv_verifier.nondet_str()
        b = _sv_verifier.nondet_bool()
        assert not re.fullmatch(r'a+', s)
    """)
    tree = ast.parse(source, "prog.py")
    first = Outcome("clean", draws=[
        {"kind": "str", "value": "", "site": "prog.py:4"},
        {"kind": "bool", "value": False, "site": "prog.py:5"},
    ])
    ctx = Context("prog.py", ("AssertionError",), first=first)
    cands = list(_regex_vectors(ctx, tree))
    assert cands, "expected at least one regex-derived candidate"
    for c in cands:
        assert c.component == "hybrid:regex"
        kinds = [d["kind"] for d in c.vector]
        assert kinds == ["str", "bool"]
    # both a False and a True bool setting are tried, not just the observed default
    assert {c.vector[1]["value"] for c in cands} == {False, True}
    # the substituted string is a real match of the pattern (fullmatch semantics: "a+")
    import re as _re
    assert all(_re.fullmatch(r"a+", c.vector[0]["value"]) for c in cands)


def test_regex_vectors_empty_without_a_str_draw(tmp_path):
    import ast
    from tharkun.sv.engine import Outcome
    from tharkun.sv.hybrid import _regex_vectors
    from tharkun.sv.stages import Context

    tree = ast.parse("import re\nre.compile(r'a+')\n", "prog.py")
    first = Outcome("clean", draws=[{"kind": "int", "value": 1, "site": "prog.py:1"}])
    ctx = Context("prog.py", ("AssertionError",), first=first)
    assert list(_regex_vectors(ctx, tree)) == []


def test_a_regex_fullmatch_kill_is_found_end_to_end(tmp_path):
    """The shape [internal ref] targets: an assert that is FALSE exactly for strings matching a
    literal regex the program itself compiles -- python-string-utils' strip_html/ISBN
    kills need a specific shape (an HTML tag, a digit run); this is the same shape at
    toy scale, small enough that a fullmatch-True Hypothesis draw finds it reliably."""
    pytest.importorskip("tharkun.solve.obligations")
    if not atheris_available():
        pytest.skip("Atheris is optional")
    p = _prog(tmp_path, 'regex_kill.py', """
        import re
        import _sv_verifier

        PAT = re.compile(r'[0-9]{4}')

        def f(s):
            return bool(PAT.fullmatch(s))

        def f_mut(s):
            return False

        s = _sv_verifier.nondet_str()
        assert f(s) == f_mut(s)
    """)
    verdict = verify(p, ("AssertionError",), budget_seconds=60, jobs=1, prove=False,
                     concolic=False, stages=["hybrid"])
    assert verdict.verdict == "false"
    assert replay(p, verdict.witness, ("AssertionError",))[0]


def test_hybrid_reads_its_obligations_from_the_runs_table_not_from_a_second_census(tmp_path, monkeypatch):
    """[internal ref] step 3a: the census is computed once, before stage 1, and lives in the table; the hybrid
    stage reads its rows there instead of walking the program again."""
    from tharkun.sv import concolic as sv_concolic
    from tharkun.sv import hybrid
    from tharkun.sv.concolic import Solved
    from tharkun.sv.engine import Outcome
    from tharkun.sv.obligation_table import ObligationTable
    from tharkun.sv.stages import Context

    p = _prog(tmp_path, 'zd.py', """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        y = 10 // x
    """)
    from tharkun.solve.census import obligation_census
    table = ObligationTable(obligation_census(open(p).read(), ("ZeroDivisionError",), p))

    class StubFuzzer:
        def __init__(self, *a, **k): pass
        def start(self): raise OSError("no fuzzer in this test")
        def stop(self): pass
        def is_running(self): return False
        def cpu_seconds(self): return 0.0
        def coverage(self): return 0

    seen = {}

    def fake_solve(program, classes, **kwargs):
        seen["extra_sites"] = kwargs["extra_sites"]
        return Solved()

    def no_second_census(*a, **k):
        raise AssertionError("the hybrid stage must read the table, not recompute the census")

    monkeypatch.setattr(hybrid, "atheris_available", lambda: True)
    monkeypatch.setattr(hybrid, "AtherisBackend", StubFuzzer)
    monkeypatch.setattr(sv_concolic, "solve", fake_solve)
    monkeypatch.setattr("tharkun.solve.census.obligation_census", no_second_census)
    logs = []
    ctx = Context(p, ("ZeroDivisionError",), first=Outcome("clean"), log=logs.append, table=table)
    assert list(hybrid.search(ctx, 10.0)) == []
    assert any("hybrid: 1 obligations" in m for m in logs), logs
    assert [r["line"] for r in seen["extra_sites"]] == [4] and seen["extra_sites"][0]["exc"] == "ZeroDivisionError"


def test_the_hybrid_handoff_to_the_symbolic_stage_carries_the_call_target_facts(tmp_path, monkeypatch):
    """[internal ref]: every ``concolic.solve`` call hands over the run's ``call_target`` records and proven class names
    (``Context.call_facts``), the hybrid stage's handoff included -- without them the symbolic executor leaves a
    call of a class unmodelled (the ``call Richards`` line of a campaign log)."""
    from tharkun.sv import concolic as sv_concolic
    from tharkun.sv import hybrid
    from tharkun.sv import scheduler
    from tharkun.sv.concolic import Solved
    from tharkun.sv.engine import Outcome
    from tharkun.sv.obligation_table import ObligationTable
    from tharkun.sv.stages import Context

    p = _prog(tmp_path, 'zd2.py', """
        import _sv_verifier

        class Box:
            def __init__(self, v):
                self.v = v

        x = _sv_verifier.nondet_int()
        y = 10 // Box(x).v
    """)
    from tharkun.solve.census import obligation_census
    table = ObligationTable(obligation_census(open(p).read(), ("ZeroDivisionError",), p))
    scheduler.publish_module_namespace(table, p)

    class StubFuzzer:
        def __init__(self, *a, **k): pass
        def start(self): raise OSError("no fuzzer in this test")
        def stop(self): pass
        def is_running(self): return False
        def cpu_seconds(self): return 0.0
        def coverage(self): return 0

    seen = {}

    def fake_solve(program, classes, **kwargs):
        seen.update(kwargs)
        return Solved()

    monkeypatch.setattr(hybrid, "atheris_available", lambda: True)
    monkeypatch.setattr(hybrid, "AtherisBackend", StubFuzzer)
    monkeypatch.setattr(sv_concolic, "solve", fake_solve)
    ctx = Context(p, ("ZeroDivisionError",), first=Outcome("clean"), log=lambda m: None, table=table)
    assert list(hybrid.search(ctx, 10.0)) == []
    assert [r["callee"] for r in seen["call_targets"] if r["target"] == "user_class"] == ["Box"]
    assert seen["proven_classes"] == ["Box"]
    assert Context(p, ("ZeroDivisionError",)).call_facts() == {"call_targets": None, "proven_classes": ()}


# --- [internal ref] (nondet_str): a str proposal is built directly from string.printable characters, one per byte ---

@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_every_byte_maps_to_one_printable_character_and_printable_ascii_stays_itself(tmp_path, monkeypatch):
    """The grading harness refuses every character outside ``string.printable`` (the run is invalid there), so
    the worker builds the proposal from printable characters directly: one per input byte, whatever the byte.
    A byte that is already a printable character is that character (a program literal written into the corpus
    as ASCII bytes, or spliced in by libFuzzer's comparison hooks, reads back as the literal); the rest fold
    onto the visible ASCII range.  Same discipline as ``_clamp_float_range`` for a float range."""
    import string

    worker = _load_hybrid_worker(tmp_path, monkeypatch)
    everything = bytes(range(256))
    mapped = worker._printable_str(everything)
    assert len(mapped) == 256                                  # one character per byte, none dropped
    assert all(c in string.printable for c in mapped), mapped
    for ok in (b"", b"abc", b"qqqqq", b"\t\n\r\x0b\x0c ", string.printable.encode()):
        assert worker._printable_str(ok) == ok.decode()        # printable ASCII is untouched, whitespace included
    for raw in (b"\x00", b"\x7f", b"\x80", b"\xe9", b"\xff"):
        assert worker._printable_str(raw) in string.printable and worker._printable_str(raw) != ""
    assert worker._printable_str(b"\xe9\x00") == worker._printable_str(b"\xe9\x00")   # deterministic


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_the_worker_draws_a_str_straight_from_the_input_bytes_and_logs_what_the_program_saw(tmp_path, monkeypatch):
    """End to end through ``TestOneInput``: the input bytes ``qqqqq`` ARE the string ``"qqqqq"`` the program
    sees (so a target like ``sm_count_loop``'s is one libFuzzer mutation away from a seed literal), the find
    is recorded with exactly that value, and an input byte outside the domain is folded before the program
    sees it AND before it is recorded -- the vector the child replays holds only printable characters."""
    import string

    program = ("import _sv_verifier\n"
               "if __name__ == '__main__':\n"
               "    s = _sv_verifier.nondet_str()\n"
               "    if s == 'qqqqq':\n"
               "        raise IndexError()\n")
    worker = _load_hybrid_worker(tmp_path, monkeypatch, program, classes=("IndexError",))
    worker.TestOneInput(b"\xe9\x00\x01zz")                     # folded: the program never sees a non-printable
    assert list((tmp_path / "found").iterdir()) == []
    worker.TestOneInput(b"qqqqq")
    found = [json.loads(f.read_text()) for f in (tmp_path / "found").iterdir()]
    assert found == [[{"kind": "str", "value": "qqqqq"}]]
    out = run_once(sys.executable, str(tmp_path / "inert.py"), found[0], timeout=30)
    assert out.invalid_draw is False and out.violated(("IndexError",)) == "IndexError"
    assert all(c in string.printable for c in worker._printable_str(b"\xe9\x00\x01zz"))


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_the_worker_matches_the_builtin_class_not_a_user_class_with_its_name(tmp_path, monkeypatch):
    """[internal ref]: ``class IndexError(Exception)`` raised by the program is no property hit; ``class E(IndexError)`` is."""
    shadow = ("import _sv_verifier\n"
              "class IndexError(Exception):\n    pass\n"
              "if __name__ == '__main__':\n"
              "    s = _sv_verifier.nondet_int()\n"
              "    raise IndexError()\n")
    worker = _load_hybrid_worker(tmp_path, monkeypatch, shadow, classes=("IndexError",))
    assert worker._BUILTIN_CLASSES["IndexError"] is IndexError
    worker.TestOneInput(b"\x00\x00\x00\x00\x00\x00\x00\x00")
    assert list((tmp_path / "found").iterdir()) == []
    sub = ("import _sv_verifier\n"
           "class E(IndexError):\n    pass\n"
           "if __name__ == '__main__':\n"
           "    s = _sv_verifier.nondet_int()\n"
           "    raise E()\n")
    second = tmp_path / "second"
    second.mkdir()
    worker2 = _load_hybrid_worker(second, monkeypatch, sub, classes=("IndexError",))
    worker2.TestOneInput(b"\x00\x00\x00\x00\x00\x00\x00\x00")
    assert len(list((second / "found").iterdir())) == 1


@pytest.mark.skipif(not atheris_available(), reason="Atheris is optional")
def test_the_mapped_str_witness_replays_confirmed_where_the_raw_one_is_refused(tmp_path, monkeypatch):
    worker = _load_hybrid_worker(tmp_path, monkeypatch)
    program = tmp_path / "str_bug.py"
    program.write_text(
        "import _sv_verifier\n"
        "if __name__ == '__main__':\n"
        "    s = _sv_verifier.nondet_str()\n"
        "    if len(s) == 1:\n"
        "        raise IndexError()\n"
    )
    raw = run_once(sys.executable, str(program), [{"kind": "str", "value": "é"}], timeout=30)
    assert raw.invalid_draw is True and raw.violated(("IndexError",)) is None
    mapped = worker._printable_str(b"\xe9")
    vector = [{"kind": "str", "value": mapped}]
    out = run_once(sys.executable, str(program), vector, timeout=30)
    assert out.invalid_draw is False and out.violated(("IndexError",)) == "IndexError"
    assert replay(str(program), {"vector": vector}, ("IndexError",))[0] is True
