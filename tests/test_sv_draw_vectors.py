"""Draw vectors from a producer ([internal ref], docs/SV.md "Draw vectors from a producer").

A fact producer may propose whole draw vectors (``draw_vector`` facts).  The rules this file pins, worst
failure first:

* the program is NEVER rewritten for replay or verdict: a vector reaches it only through the child's
  ``_sv_verifier`` shadow, and only the scheduler's one FALSE path (fresh native replay) decides anything;
* a draw outside its domain is rejected OUT OF BAND: ``invalid_draw`` is set before anything can raise or
  swallow, so a program that catches the refusal (``try``/``except``, ``contextlib.suppress``, a context manager
  whose ``__exit__`` swallows) and then fails an assertion is NOT a violation -- through every way the program may
  reach the draw (alias, ``getattr``, ``sys.modules``, ``globals()``, ``__import__``);
* a vector is pre-checked against the bounds the first run logged (it costs no run), but the child's own
  refusal is what holds when the first run never reached the call;
* nothing a producer says can make a TRUE: a vector is never certified and never discharges an obligation;
* what a hostile producer can cost is bounded: malformed or oversized vectors are dropped at the door and at
  most ``MAX_PRODUCER_VECTORS`` are replayed.

The checker the facts stage uses needs CPython 3.14; the end-to-end tests skip elsewhere.
"""
from __future__ import annotations

import json
import sys
import textwrap

import pytest

from tharkun.knowledge import KnowledgeBase
from tharkun.sv import engine, verify
from tharkun.sv import eregion_bridge as bridge
from tharkun.sv import fact_producers as fp
from tharkun.sv import stages
from tharkun.sv.engine import Outcome, run_once, vector_admissible

from tests.fake_producers import install

needs314 = pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="the facts stage needs CPython 3.14")
ZD = ("ZeroDivisionError",)
AS = ("AssertionError",)


def i(v):
    return {"kind": "int", "value": v}


def b(v):
    return {"kind": "bool", "value": v}


def f(v):
    return {"kind": "float", "value": float(v).hex()}


def s(v):
    return {"kind": "str", "value": v}


s_ = s


# ---------------------------------------------------------------------------- the door (eregion_bridge)
def _publish(vector, producer="fake", **extra):
    kb = KnowledgeBase()
    bridge.publish_fact(kb, dict({"kind": "draw_vector", "vector": vector}, **extra), producer=producer)
    return kb


def test_a_well_formed_vector_is_published_as_evidence_keyed_by_its_entries():
    kb = _publish([i(5), b(True), f(1.5), s("ab c\n")], site={"file": "t.py", "line": 3, "column": 1,
                                                              "function": "<module>"}, evidence={"engine": "x"})
    (item,) = kb.evidence(kind=bridge.DRAW_VECTOR)
    assert item.value["vector"] == [i(5), b(True), f(1.5), s("ab c\n")]
    assert item.value["site"]["line"] == 3 and item.value["evidence"] == {"engine": "x"}


def test_the_same_vector_from_two_producers_is_one_item_with_two_derivations():
    kb = _publish([i(1)], producer="a")
    bridge.publish_fact(kb, {"kind": "draw_vector", "vector": [i(1)]}, producer="b")
    (item,) = kb.evidence(kind=bridge.DRAW_VECTOR)
    assert sorted(d.producer for d in item.derivations) == ["a", "b"]


@pytest.mark.parametrize("vector", [
    "not a list", [], [i(1)] * (bridge.MAX_VECTOR_ENTRIES + 1),
    [{"kind": "int", "value": 1, "site": "t.py:3"}],           # an entry is {kind, value} and nothing else
    [{"kind": "int"}], [{"value": 1}], [3],
    [{"kind": "rindex", "value": 0, "n": 3}], [{"kind": "rfloat", "value": 0.5}],
    [{"kind": "argv", "value": ["x"]}], [{"kind": "argv_full", "value": []}], [{"kind": "bytes", "value": "x"}],
    [{"kind": "int", "value": True}], [{"kind": "bool", "value": 1}], [{"kind": "int", "value": 1.0}],
    [{"kind": "int", "value": "5"}],
    [i(1 << (bridge.MAX_INT_BITS + 1))],
    [{"kind": "float", "value": 1.5}], [{"kind": "float", "value": "not-hex"}],
    [s("\x00")], [s("é")], [s("\x7f")], [s("a" * (bridge.MAX_STR_CHARS + 1))],
    [{"kind": "str", "value": b"ab"}],
], ids=lambda v: str(v)[:40])
def test_a_malformed_or_out_of_domain_vector_is_refused_at_the_door(vector):
    kb = KnowledgeBase()
    with pytest.raises(ValueError):
        bridge.publish_fact(kb, {"kind": "draw_vector", "vector": vector}, producer="fake")
    assert not list(kb.evidence(kind=bridge.DRAW_VECTOR))


def test_the_widest_values_the_door_admits_are_admitted():
    kb = _publish([i(-(1 << bridge.MAX_INT_BITS) + 1), i((1 << bridge.MAX_INT_BITS) - 1), f(float("inf")),
                   f(float("nan")), s("a" * bridge.MAX_STR_CHARS)])
    assert len(list(kb.evidence(kind=bridge.DRAW_VECTOR))) == 1


# ------------------------------------------------------------------ the pre-check (engine.vector_admissible)
def _first(draws, ranges):
    return Outcome("clean", draws=draws, ranges=ranges)


def test_a_vector_that_repeats_the_first_run_up_to_a_float_range_call_is_checked_against_its_domain():
    """[internal ref]: the domain is what ``random.uniform(a, b)`` can return, not ``[a, b]``: ``1.0`` is out of it for ``(0.0, 1.0)``."""
    first = _first([{"kind": "bool", "value": True}, {"kind": "float", "value": f(0.5)["value"]}],
                   [[1, f(0.0)["value"], f(1.0)["value"]]])
    assert vector_admissible([b(True), f(0.25)], first)
    assert vector_admissible([b(True), f(0.0)], first) and vector_admissible([b(True), f(1 - 2.0 ** -53)], first)
    assert not vector_admissible([b(True), f(1.0)], first)
    assert not vector_admissible([b(True), f(0.1)], first)
    assert not vector_admissible([b(True), f(1.5)], first)
    assert not vector_admissible([b(True), f(-0.1)], first)
    assert not vector_admissible([b(True), f(float("nan"))], first)
    assert not vector_admissible([b(True), f(float("inf"))], first)


def test_the_value_at_a_position_does_not_move_the_call_at_it():
    # position 0 is the same call whatever value the vector puts there
    first = _first([{"kind": "float", "value": f(0.5)["value"]}], [[0, f(0.0)["value"], f(1.0)["value"]]])
    assert not vector_admissible([f(7.0)], first)


def test_the_bounds_stop_applying_where_the_vector_leaves_the_first_run():
    # the vector draws False where the first run drew True: what sits at position 1 is now unknown
    first = _first([{"kind": "bool", "value": True}, {"kind": "float", "value": f(0.5)["value"]}],
                   [[1, f(0.0)["value"], f(1.0)["value"]]])
    assert vector_admissible([b(False), f(7.0)], first)


def test_with_no_first_run_ranges_only_the_type_and_domain_rules_apply():
    assert vector_admissible([i(3), f(1e300)], _first([], []))
    assert not vector_admissible([s("\x00")], _first([], []))
    assert not vector_admissible([{"kind": "int", "value": True}], _first([], []))
    assert not vector_admissible([{"kind": "argv", "value": []}], _first([], []))
    assert not vector_admissible([{"kind": "argv_full", "value": []}], _first([], []))
    assert not vector_admissible("nope", _first([], []))


# ----------------------------------------------------------- the child logs the bounds (and only the child decides)
def _task(tmp_path, body, name="t.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body).lstrip("\n"))
    return str(path)


FLOAT_RANGE = """
    import _sv_verifier
    if __name__ == "__main__":
        x = _sv_verifier.nondet_float_range(0, 1)
        y = _sv_verifier.nondet_float_range(5.0, 2.0)
"""


def test_the_child_logs_the_bounds_of_every_float_range_call_beside_the_draws_not_in_them(tmp_path):
    out = run_once(sys.executable, _task(tmp_path, FLOAT_RANGE), [f(0.5), f(3.0)], timeout=30)
    # [position, a, b] in CALL order ([internal ref]: uniform(5.0, 2.0) and uniform(2.0, 5.0) are different value sets)
    assert out.status == "clean" and out.ranges == [[0, f(0)["value"], f(1)["value"]],
                                                    [1, f(5)["value"], f(2)["value"]]]
    assert all(set(d) <= {"kind", "value", "site"} for d in out.draws)     # a witness never carries a range


def test_logging_the_bounds_never_touches_a_bound_that_is_a_user_object(tmp_path):
    """The bounds are the task's own objects: logging them must not call ``__float__`` or a comparison on them
    (the grading harness's model does not), or a task that counts such calls would behave differently here."""
    path = _task(tmp_path, """
        import _sv_verifier
        CALLS = []

        class B:
            def __init__(self, v):
                self.v = v
            def __float__(self):
                CALLS.append("float")
                return float(self.v)
            def _o(self, o):
                return o.v if isinstance(o, B) else o
            def __lt__(self, o): return self.v < self._o(o)
            def __gt__(self, o): return self.v > self._o(o)
            def __le__(self, o): return self.v <= self._o(o)
            def __ge__(self, o): return self.v >= self._o(o)

        if __name__ == "__main__":
            try:
                x = _sv_verifier.nondet_float_range(B(0.0), B(1.0))
            except ValueError:
                pass                            # a bound that is not exactly int/float/bool has no value in the domain (#1159 follow-up): refused
            if CALLS:
                raise IndexError("the verifier touched a user bound")
    """)
    out = run_once(sys.executable, path, [f(0.5)], timeout=30)
    assert out.status == "clean" and out.ranges == [] and out.invalid_draw, (out.status, out.exc, out.msg)


def test_the_range_log_is_bounded(tmp_path):
    path = _task(tmp_path, """
        import _sv_verifier
        if __name__ == "__main__":
            for _ in range(5000):
                _sv_verifier.nondet_float_range(0.0, 1.0)
    """)
    out = run_once(sys.executable, path, [], timeout=60)
    assert out.status == "clean" and len(out.draws) == 5000 and len(out.ranges) == 4096


def test_an_out_of_range_draw_sets_invalid_draw_before_anything_can_raise_or_swallow(tmp_path):
    out = run_once(sys.executable, _task(tmp_path, FLOAT_RANGE), [f(1.5)], timeout=30)
    assert out.invalid_draw and out.violated(("ValueError",)) is None


# ------------------------------------------------- the stage: only replays, only what the door let in
def _stage_ctx(vectors, first=None):
    kb = KnowledgeBase()
    for v in vectors:
        bridge.publish_fact(kb, {"kind": "draw_vector", "vector": v}, producer="fake")

    class Table:
        pass
    table = Table()
    table.kb = kb
    log = []
    return stages.Context("t.py", AS, first=first or _first([], []), log=log.append, table=table), log


def test_the_stage_yields_each_admissible_vector_as_a_candidate_named_for_its_producer():
    ctx, _log = _stage_ctx([[i(5)], [i(6), b(True)]])
    cands = list(stages._producer(ctx, 5.0))
    assert [c.vector for c in cands] == [[i(5)], [i(6), b(True)]]
    assert {c.component for c in cands} == {"producer:fake"}


def test_the_stage_skips_a_vector_that_is_the_first_run_again():
    ctx, _log = _stage_ctx([[i(0), b(False), f(0.0), s("")], [i(0), i(1)]])
    assert [c.vector for c in stages._producer(ctx, 5.0)] == [[i(0), i(1)]]


def test_the_stage_does_not_replay_a_vector_the_child_would_refuse():
    first = _first([{"kind": "float", "value": f(0.5)["value"]}], [[0, f(0.0)["value"], f(1.0)["value"]]])
    ctx, log = _stage_ctx([[f(9.0)], [f(0.75)]], first=first)
    assert [c.vector for c in stages._producer(ctx, 5.0)] == [[f(0.75)]]
    assert any("would refuse" in m for m in log)


def test_the_stage_replays_at_most_max_producer_vectors():
    ctx, _log = _stage_ctx([[i(n)] for n in range(1, stages.MAX_PRODUCER_VECTORS + 40)])
    assert len(list(stages._producer(ctx, 5.0))) == stages.MAX_PRODUCER_VECTORS


def test_with_no_table_or_no_vectors_the_stage_yields_nothing():
    assert list(stages._producer(stages.Context("t.py", AS), 1.0)) == []
    ctx, _ = _stage_ctx([])
    assert list(stages._producer(ctx, 1.0)) == []


def test_the_stage_is_in_the_default_portfolio_before_the_concolic_stages():
    names = [st.name for st in stages.select()]
    assert "producer" in names and names.index("producer") < names.index("concolic")


# ------------------------------------------------------------------------------------- end to end
def producer_body(vectors):
    return textwrap.dedent("""
        import json
        VECTORS = json.loads(%r)

        def produce(request):
            return [{"kind": "draw_vector", "vector": v, "evidence": {"engine": "fake"}} for v in VECTORS]
    """) % (json.dumps(vectors),)


def with_producer(tmp_path, monkeypatch, vectors):
    where = tmp_path / "producer"
    where.mkdir()
    return install(where, monkeypatch, producer_body(vectors), only=True)


def inject(monkeypatch, vectors):
    """Put ``vectors`` into the run's store exactly as the facts stage does (``bridge.publish_fact``), whatever the
    census says: the facts stage asks a producer only about a task with a complete census and no gap, which
    leaves out the programs these attacks need (reflection, imports the census does not resolve).  The stage
    under test and everything after it are the real ones."""
    from tharkun.sv import fact_proof

    def publish_only(program, classes, *, table, **_kw):
        for v in vectors:
            bridge.publish_fact(table.kb, {"kind": "draw_vector", "vector": v}, producer="injected")
        return None

    monkeypatch.setattr(fact_proof, "prove_from_facts", publish_only)


def run(path, classes=ZD, **kw):
    from unittest.mock import patch
    lines = []
    kw.setdefault("budget_seconds", 60)
    kw.setdefault("jobs", 1)
    # the stage under test alone: no symbolic or guard stages, no program-literal or unbounded candidates
    kw.setdefault("stages", ("producer",))
    kw.setdefault("concolic", False)
    kw.setdefault("constants", False)
    kw.setdefault("deepen", False)
    with patch("tharkun.sv.engine._static_proof", return_value=None):
        verdict = verify(path, classes, log=lines.append, **kw)
    return verdict, lines


#: the violation sits at x == 123457, a value no program literal or small-first candidate names
ROOT = """
    import _sv_verifier
    if __name__ == "__main__":
        x = _sv_verifier.nondet_int()
        d = x - 100000 - 23457
        r = 10 // d
"""


@needs314
def test_a_proposed_vector_that_violates_is_replayed_natively_and_decides_false(tmp_path, monkeypatch):
    path = _task(tmp_path, ROOT)
    with monkeypatch.context() as m:
        # NO producer, not even one the environment installs (Eregion's own finds this straight-line divisor)
        m.setattr(fp, "discover", lambda: ())
        verdict, lines = run(path)
    assert verdict.verdict != "false"                       # nothing finds it without the producer
    with_producer(tmp_path, monkeypatch, [[i(123457)]])
    verdict, lines = run(path)
    assert verdict.verdict == "false" and verdict.exception == "ZeroDivisionError", lines
    assert verdict.component == "producer:fake"
    assert verdict.witness["vector"] == [i(123457)]         # exactly what the replay consumed
    assert any("(solved, producer:fake)" in m for m in lines)


@needs314
def test_a_vector_that_does_not_violate_decides_nothing(tmp_path, monkeypatch):
    path = _task(tmp_path, ROOT)
    with_producer(tmp_path, monkeypatch, [[i(7)], [i(-5)], [i(123456)]])
    verdict, lines = run(path)
    assert verdict.verdict != "false" and verdict.verdict != "true", lines
    assert sum("solved vector did not violate" in m for m in lines) == 3


@needs314
def test_a_vector_longer_than_the_run_draws_is_replayed_as_what_the_run_consumed(tmp_path, monkeypatch):
    with_producer(tmp_path, monkeypatch, [[i(123457), i(9), b(True), s("zz")]])
    verdict, lines = run(_task(tmp_path, ROOT))
    assert verdict.verdict == "false" and verdict.witness["vector"] == [i(123457)]


@needs314
def test_a_vector_with_the_wrong_kinds_is_harmless(tmp_path, monkeypatch):
    # the program draws an int; the vector offers a bool and a float there: the draw takes its default
    with_producer(tmp_path, monkeypatch, [[b(True)], [f(123457.0)], [s("123457")]])
    verdict, lines = run(_task(tmp_path, ROOT))
    assert verdict.verdict not in ("false", "true"), lines


@needs314
def test_a_lying_producer_cannot_make_a_safe_program_true_or_false(tmp_path, monkeypatch):
    safe = """
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            r = 10 // (x * x + 1)
    """
    with_producer(tmp_path, monkeypatch, [[i(0)], [i(1)], [i(-1)], [i(10 ** 6)]])
    verdict, lines = run(_task(tmp_path, safe), stages=("producer",), prove=False)
    assert verdict.verdict == "unknown" and verdict.component == "none", lines


@needs314
def test_a_vector_never_discharges_an_obligation_or_is_certified(tmp_path, monkeypatch):
    from tharkun.solve.census import obligation_census
    from tharkun.sv.fact_proof import prove_from_facts
    from tharkun.sv.obligation_table import OPEN, ObligationTable

    path = _task(tmp_path, ROOT)
    source = open(path).read()
    producer = with_producer(tmp_path, monkeypatch, [[i(123457)], [i(1)]])
    table = ObligationTable(obligation_census(source, ZD, path))
    spec = "CHECK( init(main()), ! uncaught(ZeroDivisionError))\n"
    assert prove_from_facts(path, ZD, source=source, specification=spec, table=table, python=sys.executable,
                            remaining=lambda: 60.0, say=lambda m: None) is None
    assert len(list(table.kb.evidence(kind=bridge.DRAW_VECTOR))) == 2
    assert not list(table.kb.evidence(kind=bridge.CERTIFIED_INVARIANT))
    assert all(table.status(o) == OPEN for o in table.rows(ZD))
    assert producer.name == "fake"


@needs314
def test_a_producer_that_floods_or_garbles_costs_nothing_but_its_valid_vectors(tmp_path, monkeypatch):
    vectors = [[i(n)] for n in range(1, 400)] + [[{"kind": "rindex", "value": 0, "n": 2}], "junk", [], [s("\x00")]]
    with_producer(tmp_path, monkeypatch, vectors)
    verdict, lines = run(_task(tmp_path, ROOT), budget_seconds=60)
    replayed = sum("solved vector did not violate" in m for m in lines)
    assert replayed == stages.MAX_PRODUCER_VECTORS and verdict.verdict != "false", lines
    assert any("dropped a malformed fact" in m for m in lines)


# ------------------------------------------------------------ attacks: a draw outside its domain, however reached
#: (module-level lines, the statement block that sets ``x`` -- written at indent 0, ``DRAW`` stands for the draw call)
SWALLOWERS = {
    "try_except": ("", "try:\n    x = DRAW\nexcept ValueError:\n    x = 5.0"),
    "try_except_exception": ("", "try:\n    x = DRAW\nexcept Exception:\n    x = 5.0"),
    "bare_except": ("", "try:\n    x = DRAW\nexcept:\n    x = 5.0"),
    "suppress": ("import contextlib", "x = 5.0\nwith contextlib.suppress(ValueError):\n    x = DRAW"),
    "suppress_exception": ("from contextlib import suppress", "x = 5.0\nwith suppress(Exception):\n    x = DRAW"),
    "swallowing_exit": ("class Swallow:\n    def __enter__(self):\n        return self\n"
                        "    def __exit__(self, *a):\n        return True",
                        "x = 5.0\nwith Swallow():\n    x = DRAW"),
    "finally_return": ("def draw():\n    ok = False\n    try:\n        v = DRAW\n        ok = True\n"
                       "        return v\n    finally:\n        if not ok:\n            return 5.0",
                       "x = draw()"),
}
DRAWS = {
    "plain": "_sv_verifier.nondet_float_range(0.0, 1.0)",
    "alias": "ALIAS(0.0, 1.0)",
    "getattr": "getattr(_sv_verifier, 'nondet_float_range')(0.0, 1.0)",
    "sys_modules": "sys.modules['_sv_verifier'].nondet_float_range(0.0, 1.0)",
    "globals": "globals()['_sv_verifier'].nondet_float_range(0.0, 1.0)",
    "dunder_import": "__import__('_sv_verifier').nondet_float_range(0.0, 1.0)",
    "from_import": "NFR(0.0, 1.0)",
    "int_bounds": "_sv_verifier.nondet_float_range(0, 1)",
    "reversed_bounds": "_sv_verifier.nondet_float_range(1.0, 0.0)",
}
HEAD = ("import sys\nimport _sv_verifier\nfrom _sv_verifier import nondet_float_range as NFR\n"
        "ALIAS = _sv_verifier.nondet_float_range\n")


def attack_program(swallow: str, draw: str, *, branch: bool) -> str:
    """The task sets ``x`` from a ``nondet_float_range(0, 1)`` call it lets the refusal of an out-of-range draw
    fall through to ``x = 5.0``, then asserts ``x <= 1``: violated only through a draw the harness marks invalid."""
    pre, block = (part.replace("DRAW", DRAWS[draw]) for part in SWALLOWERS[swallow])
    body = block + "\nassert x <= 1.0"
    if branch:
        body = "if _sv_verifier.nondet_bool():\n" + textwrap.indent(body, "    ")
    return HEAD + pre + "\nif __name__ == '__main__':\n" + textwrap.indent(body, "    ") + "\n"


@pytest.mark.filterwarnings("ignore::SyntaxWarning")        # a ``return`` in ``finally``: that is the attack
@pytest.mark.parametrize("swallow", sorted(SWALLOWERS))
@pytest.mark.parametrize("draw", sorted(DRAWS))
def test_an_out_of_range_draw_is_never_a_violation_however_the_program_swallows_the_refusal(tmp_path, swallow, draw):
    """Replayed directly (no pre-check in the way): the program catches the child's refusal and then fails its
    assertion, but ``invalid_draw`` was set first, so the run is not a violation -- the grading harness marks the
    same run invalid.  The in-range draw is not a violation either, so the program is simply safe."""
    path = _task(tmp_path, attack_program(swallow, draw, branch=False))
    compile(open(path).read(), path, "exec")
    bad = run_once(sys.executable, path, [f(5.0)], timeout=30)
    assert bad.invalid_draw, (bad.status, bad.exc, bad.msg)
    assert bad.violated(AS) is None and bad.violated(("ValueError",)) is None
    good = run_once(sys.executable, path, [f(0.5)], timeout=30)
    assert good.status == "clean" and not good.invalid_draw


@pytest.mark.filterwarnings("ignore::SyntaxWarning")
@pytest.mark.parametrize("swallow", sorted(SWALLOWERS))
@pytest.mark.parametrize("branch", [False, True], ids=["pre-checked", "child-refuses"])
def test_a_producer_cannot_turn_a_swallowed_refusal_into_a_false(tmp_path, monkeypatch, swallow, branch):
    """The vector goes through the whole scheduler: unconditionally drawn, the pre-check drops it; behind a branch
    the first run did not take, the pre-check cannot know the bounds and the CHILD refuses it.  Either way the
    verdict is not FALSE."""
    path = _task(tmp_path, attack_program(swallow, "plain", branch=branch))
    inject(monkeypatch, [[b(True), f(5.0)] if branch else [f(5.0)]])
    verdict, lines = run(path, AS)
    assert verdict.verdict != "false", lines
    if branch:
        assert any("solved vector did not violate" in m for m in lines), lines   # it WAS replayed, and refused
    else:
        assert any("would refuse" in m for m in lines), lines                   # it was never replayed


@pytest.mark.parametrize("draw", sorted(DRAWS))
def test_the_draw_may_be_reached_any_way_the_program_likes(tmp_path, monkeypatch, draw):
    path = _task(tmp_path, attack_program("try_except", draw, branch=True))
    inject(monkeypatch, [[b(True), f(5.0)]])
    verdict, lines = run(path, AS)
    assert verdict.verdict != "false", lines
    assert any("solved vector did not violate" in m for m in lines), lines


@needs314
def test_the_real_facts_stage_hands_a_swallowing_task_its_vector_and_the_verdict_stays_off_false(tmp_path,
                                                                                               monkeypatch):
    path = _task(tmp_path, """
        import _sv_verifier
        if __name__ == "__main__":
            ok = _sv_verifier.nondet_bool()
            if ok:
                try:
                    x = _sv_verifier.nondet_float_range(0.0, 1.0)
                except ValueError:
                    x = 5.0
                d = x - 5.0
                r = 10 / d
    """)
    with_producer(tmp_path, monkeypatch, [[b(True), f(5.0)]])
    verdict, lines = run(path)
    assert any("fake published 1 of 1" in m for m in lines), lines             # the producer WAS asked
    assert verdict.verdict != "false", lines


def test_a_non_printable_string_vector_never_reaches_a_run(tmp_path, monkeypatch):
    path = _task(tmp_path, """
        import _sv_verifier
        if __name__ == "__main__":
            try:
                s = _sv_verifier.nondet_str()
            except ValueError:
                s = "\\u00e9"
            assert s.isascii()
    """)
    # the door drops such a vector; and were one replayed anyway, the child refuses it before it is logged
    with pytest.raises(ValueError):
        bridge.publish_fact(KnowledgeBase(), {"kind": "draw_vector", "vector": [s_("\u00e9")]}, producer="x")
    refused = run_once(sys.executable, path, [s_("\u00e9")], timeout=30)
    assert refused.invalid_draw and refused.violated(AS) is None
    refused = run_once(sys.executable, path, [s_("\x00")], timeout=30)
    assert refused.invalid_draw and refused.violated(AS) is None


def test_a_vector_naming_a_draw_the_program_reaches_only_through_reflection_is_still_replayed_natively(tmp_path,
                                                                                                     monkeypatch):
    # the task's draw is reached by name lookup in its own globals; there is no rewrite for this to defeat
    path = _task(tmp_path, """
        import _sv_verifier
        if __name__ == "__main__":
            draw = globals()["_sv_verifier"].nondet_int
            x = draw()
            r = 10 // (x - 100000 - 23457)
    """)
    inject(monkeypatch, [[i(123457)]])
    verdict, lines = run(path)
    assert verdict.verdict == "false" and verdict.witness["vector"] == [i(123457)], lines
    assert verdict.component == "producer:injected"
