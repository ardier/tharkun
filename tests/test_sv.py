"""tharkun.sv: deciding an SV-Benchmarks ``! uncaught(E)`` property on real child interpreters."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

from tharkun.sv import program_of, read_property, replay, run_once, verify
from tharkun.sv import stages as _stages
from tharkun.sv.engine import harvest_constants

DL = ("KeyError", "IndexError")


def _prog(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return str(p)


def test_property_file_lists_the_classes(tmp_path):
    prp = tmp_path / "data-lookup.prp"
    prp.write_text("CHECK( init(main()), ! uncaught(KeyError))\nCHECK( init(main()), ! uncaught(IndexError))\n")
    assert read_property(str(prp)) == ("data-lookup", DL)
    bad = tmp_path / "bad.prp"
    bad.write_text("CHECK( init(main()), LTL(G ! call(reach_error())) )\n")
    with pytest.raises(ValueError):
        read_property(str(bad))


def test_task_definition_names_the_program(tmp_path):
    (tmp_path / "t.py").write_text("pass\n")
    yml = tmp_path / "t.yml"
    yml.write_text("format_version: '2.0'\n\ninput_files: 't.py'\n\nproperties: []\n")
    assert program_of(str(yml)) == str(tmp_path / "t.py")


def test_deterministic_clean_program_is_true(tmp_path):
    p = _prog(tmp_path, "det.py", """
        xs = [1, 2, 3]
        if __name__ == "__main__":
            print(xs[2])
    """)
    v = verify(p, DL, budget_seconds=60)
    assert v.verdict == "true" and v.runs == 2


def test_deterministic_violation_is_false_with_a_replayable_witness(tmp_path):
    p = _prog(tmp_path, "det_bad.py", """
        name = "out.txt",
        if __name__ == "__main__":
            open(name)
    """)
    v = verify(p, ("TypeError", "AttributeError"), budget_seconds=60)
    assert v.line() == "FALSE(TypeError)"
    assert v.witness["vector"] == [] and v.witness["site"]["file"] == "det_bad.py"
    ok, _ = replay(p, v.witness, ("TypeError",))
    assert ok


def test_search_finds_the_draw_that_violates(tmp_path):
    p = _prog(tmp_path, "nd.py", """
        import _sv_verifier
        xs = [10, 20, 30]
        if __name__ == "__main__":
            i = abs(_sv_verifier.nondet_int())
            print(xs[i])
    """)
    # enumeration alone: the program's own constants (10, 20, 30 and neighbours) rank
    # before 3, 4, 5, ... ([internal ref]); the solve stage answers it directly (test_sv_concolic).
    # hybrid ([internal ref]) can also satisfy this guard by Atheris fuzzing, so it is scoped out
    # here too: this test is about the enumeration order, not which stage answers first.
    v = verify(p, DL, budget_seconds=60, concolic=False, stages=[s for s in _stages.DEFAULT if s != "hybrid"])
    assert v.line() == "FALSE(IndexError)"
    assert v.witness["vector"] == [{"kind": "int", "value": 10}]
    ok, out = replay(p, v.witness, DL)
    assert ok and out.exc == "IndexError"


def test_a_float_range_endpoint_is_reached(tmp_path):
    p = _prog(tmp_path, "fr.py", """
        import _sv_verifier
        if __name__ == "__main__":
            t = _sv_verifier.nondet_float_range(0, 1)
            print(1 / (t * 2))
    """)
    v = verify(p, ("ZeroDivisionError", "FloatingPointError"), budget_seconds=60)
    assert v.line() == "FALSE(ZeroDivisionError)"


def test_nondet_float_range_out_of_domain_draw_is_invalid_never_a_witness(tmp_path):
    """[internal ref]: the grading harness's own model (eregion_sv/shim/_sv_verifier.py) REJECTS a
    ``nondet_float_range(a, b)`` draw outside ``[a, b]`` (raises, sets ``invalid_draw``);
    _child.py used to silently CLAMP it instead, while logging the RAW out-of-range value
    in the witness -- a witness Tharkun's own replay would confirm but the grading harness's
    replay would refuse (NOT CONFIRMED): a wrong FALSE. A hand-crafted vector with a raw
    draw outside the call's domain must now raise ValueError, set ``invalid_draw``, and
    ``violated()`` must refuse it even when the raised exception happens to match a
    requested property class -- the exact case a naive ``status == "raised" and exc in
    classes`` check would get wrong."""
    p = _prog(tmp_path, "fr_oob.py", """
        import _sv_verifier
        if __name__ == "__main__":
            _sv_verifier.nondet_float_range(0.0, 1.0)
    """)
    out_of_range = [{"kind": "float", "value": (5.0).hex()}]
    out = run_once(sys.executable, p, out_of_range, timeout=30)
    assert out.status == "raised" and out.exc == "ValueError"
    assert out.invalid_draw is True
    # even a property that names ValueError itself must not be "confirmed" by this run --
    # the harness's own replay never reaches the raise (it rejects the draw first).
    assert out.violated(("ValueError",)) is None
    assert out.violated(("KeyError", "IndexError")) is None
    confirmed, replayed_out = replay(p, {"vector": out_of_range}, ("ValueError",))
    assert confirmed is False
    assert replayed_out.invalid_draw is True


def test_nondet_float_range_in_domain_draw_still_works(tmp_path):
    """No regression: a draw ``random.uniform(a, b)`` can return ([internal ref]: ``0.0`` and ``0.5`` of ``(0.0, 1.0)``, and its largest
    value ``1 - 2**-53``; the bound ``1.0`` is NOT one) is returned as-is, the run is clean, and nothing is flagged invalid."""
    p = _prog(tmp_path, "fr_ok.py", """
        import _sv_verifier
        if __name__ == "__main__":
            t = _sv_verifier.nondet_float_range(0.0, 1.0)
            assert 0.0 <= t <= 1.0
    """)
    for value in (0.0, 1 - 2.0 ** -53, 0.5):
        out = run_once(sys.executable, p, [{"kind": "float", "value": value.hex()}], timeout=30)
        assert out.status == "clean" and out.invalid_draw is False
    out = run_once(sys.executable, p, [{"kind": "float", "value": (1.0).hex()}], timeout=30)
    assert out.invalid_draw is True                                      # uniform(0.0, 1.0) never returns 1.0


def test_nondet_float_range_endpoints_are_explicit_float_candidates(tmp_path):
    """[internal ref]: an out-of-domain draw is now refused rather than clamped, so the search must
    be steered TOWARD the domain instead of stumbling into it -- ``harvest_constants``
    proposes a call site's own ``a``/``b`` (and their midpoint) as ``float`` candidates
    explicitly, even when they are INT literals (``nondet_float_range(0, 1)``, common in
    the SV-Benchmarks set) that would otherwise only ever reach the ``int`` pool, never
    ``float`` -- the kind the draw itself actually is."""
    p = _prog(tmp_path, "fr_const.py", """
        import _sv_verifier
        if __name__ == "__main__":
            t = _sv_verifier.nondet_float_range(0, 1024)
            u = _sv_verifier.nondet_float_range(-5, 5)
    """)
    floats = harvest_constants(p)["float"]
    for expected in (0.0, 1024.0, 512.0, -5.0, 5.0, 0.0):
        assert expected in floats, (expected, floats)
    # [internal ref]: the largest value random.uniform(a, b) can return, next to the (mostly unproducible) bound itself
    assert 1024.0 - 1024.0 * 2.0 ** -53 in floats or any(1023.99 < f < 1024.0 for f in floats), floats
    assert any(4.99 < f < 5.0 for f in floats) or 5.0 in floats, floats


def test_finite_draw_tree_enumerated_is_true(tmp_path):
    p = _prog(tmp_path, "bools.py", """
        import _sv_verifier
        if __name__ == "__main__":
            a = _sv_verifier.nondet_bool()
            b = _sv_verifier.nondet_bool()
            xs = [0, 1, 2]
            print(xs[int(a) + int(b)])
    """)
    # prove=False: this test is about the ENUMERATION's TRUE; with the static provers on, whichever of absint and the enumeration
    # answers first decides the component (absint's own answer is checked in the absint tests)
    v = verify(p, DL, budget_seconds=60, prove=False)
    assert v.verdict == "true" and v.reason.startswith("finite draw tree enumerated")
    assert v.component == "finite_tree" and "certificate" in v.detail


def test_unbounded_draws_without_a_violation_are_unknown(tmp_path):
    p = _prog(tmp_path, "ok.py", """
        import _sv_verifier
        xs = [1]
        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            print(xs[0] * n)
    """)
    v = verify(p, DL, budget_seconds=60, run_timeout=30, prove=False)
    assert v.verdict == "unknown"


def test_a_static_proof_turns_an_unbounded_search_into_true(tmp_path):
    """[internal ref] group C: absint proves ``xs[0]`` safe for every draw; the canary finds nothing."""
    p = _prog(tmp_path, "ok.py", """
        import _sv_verifier
        xs = [1]
        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            print(xs[0] * n)
    """)
    v = verify(p, DL, budget_seconds=60, run_timeout=30)
    assert v.verdict == "true" and v.component == "proof:interval"
    assert "canary" in v.reason and v.detail["sites"]


def test_absint_writes_candidate_invariants_even_on_its_own_unknown(tmp_path):
    """[internal ref] SEV1 (owner, 2026-09-29; [internal ref]): absint's partial per-point
    invariants used to be thrown away entirely whenever its own OVERALL verdict was
    UNKNOWN (the common case) -- only a TRUE absint run ever reached the KnowledgeBase,
    through Verdict.detail["candidate_invariants"] -> sv/proof.py::finish ([internal ref] step 2,
    landed earlier). _static_proof now writes the SAME uncertified candidate_invariant
    Facts from absint's partial progress directly, whether or not its own verdict
    succeeds -- the stationary proof (drawflow) is the intended reader, deciding whether
    to consult it is real, separate, later work ([internal ref] SEV1's slice 2)."""
    from tharkun.facts import Fact
    from tharkun.knowledge import KnowledgeBase
    from tharkun.sv import engine as _engine

    p = _prog(tmp_path, "partial.py", """
        import _sv_verifier

        def f(xs, i):
            return xs[i]

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            total = 0
            while n > 0:
                total = total + n
                n = n - 1
            xs = [1, 2, 3]
            idx = _sv_verifier.nondet_int()
            print(f(xs, idx))
    """)
    from tharkun.solve.obligations import source_points
    import ast as _ast
    source = _engine.read_source(p)
    tree = _ast.parse(source, p)
    invariant_locations = {loc for node, (_, loc) in source_points(tree, source, p).items()
                           if isinstance(node, _ast.While)}
    from tharkun.solve.census import obligation_census
    from tharkun.sv.obligation_table import ObligationTable
    table = ObligationTable(obligation_census(source, DL, p))
    v = _engine._static_proof(p, DL, lambda: 30.0, lambda msg: None,
                              invariant_locations=invariant_locations, table=table)
    assert v is None, "the fixture must actually reproduce absint's own UNKNOWN"
    facts = [f for f in table.kb if f.kind == "candidate_invariant"]
    assert facts, "expected at least one candidate_invariant fact from absint's partial progress"
    assert all(f.certified is False and f.source == "absint" for f in facts)
    assert any("total" in f.value or "n" in f.value for f in facts)


def test_absint_partial_candidates_do_not_change_the_verdict(tmp_path):
    """Pure bookkeeping, same discipline as every other step 2 writer."""
    from tharkun.knowledge import KnowledgeBase
    from tharkun.sv import engine as _engine

    p = _prog(tmp_path, "partial_inert.py", """
        import _sv_verifier

        def f(xs, i):
            return xs[i]

        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            total = 0
            while n > 0:
                total = total + n
                n = n - 1
            xs = [1, 2, 3]
            idx = _sv_verifier.nondet_int()
            print(f(xs, idx))
    """)
    from tharkun.solve.obligations import source_points
    import ast as _ast
    source = _engine.read_source(p)
    tree = _ast.parse(source, p)
    invariant_locations = {loc for node, (_, loc) in source_points(tree, source, p).items()
                           if isinstance(node, _ast.While)}
    from tharkun.solve.census import obligation_census
    from tharkun.sv.obligation_table import ObligationTable
    v1 = _engine._static_proof(p, DL, lambda: 30.0, lambda msg: None,
                               invariant_locations=invariant_locations)
    table = ObligationTable(obligation_census(source, DL, p))
    v2 = _engine._static_proof(p, DL, lambda: 30.0, lambda msg: None,
                               invariant_locations=invariant_locations, table=table)
    assert v1 is None and v2 is None


def test_static_proof_publishes_absints_records_to_the_store_and_reads_class_hints_from_it(tmp_path, monkeypatch):
    """[internal ref] step 3a: absint's per-site records no longer come back through the return value; they
    are ``site_record`` Evidence in the table (one generation per run), and the class hints absint's
    second try needs are read from the table, not passed as an argument."""
    from tharkun.solve import absint as _absint
    from tharkun.solve.census import obligation_census
    from tharkun.sv import engine as _engine
    from tharkun.sv.obligation_table import ObligationTable

    p = _prog(tmp_path, "records.py", """
        import _sv_verifier
        xs = [1, 2, 3]
        if __name__ == "__main__":
            i = _sv_verifier.nondet_int()
            print(xs[i])
    """)
    source = open(p).read()
    table = ObligationTable(obligation_census(source, DL, p))
    seen_hints = []
    real_prove = _absint.prove

    def spy(src, classes, **kwargs):
        seen_hints.append(kwargs.get("class_hints"))
        return real_prove(src, classes, **kwargs)

    monkeypatch.setattr(_absint, "prove", spy)
    assert _engine._static_proof(p, DL, lambda: 30.0, lambda msg: None, table=table) is None
    records = table.site_records()
    assert records and any(r["status"] == "may_raise" for r in records)

    table.publish_class_hints({"C": ["f"]}, producer="execution")
    _engine._static_proof(p, DL, lambda: 30.0, lambda msg: None, table=table)
    assert seen_hints == [None, {"C": ["f"]}]                     # the second try reads the store
    assert len({e.key.split(":")[0] for e in table.kb.evidence(kind="site_record")}) == 2   # a new generation


def test_the_canary_search_overrules_nothing_it_cannot_find_but_a_false_wins(tmp_path):
    """A violation reachable by the search is FALSE even though proof stages ran first."""
    p = _prog(tmp_path, "bad.py", """
        import _sv_verifier
        if __name__ == "__main__":
            n = _sv_verifier.nondet_int()
            xs = [1, 2]
            print(xs[n])
    """)
    v = verify(p, DL, budget_seconds=60, run_timeout=30)
    assert v.line() == "FALSE(IndexError)"


def test_unseeded_random_choice_is_searched_and_seeded_random_is_deterministic(tmp_path):
    p = _prog(tmp_path, "rnd.py", """
        import random
        TABLE = {"a": 1, "b": 2}
        if __name__ == "__main__":
            print(TABLE[random.choice(["a", "b", "c"])])
    """)
    v = verify(p, DL, budget_seconds=60)
    assert v.line() == "FALSE(KeyError)"
    assert v.witness["vector"] == [{"kind": "rindex", "value": 2, "n": 3}]
    seeded = _prog(tmp_path, "seeded.py", """
        import random
        random.seed(4)
        if __name__ == "__main__":
            print([1, 2][random.randrange(2)])
    """)
    assert verify(seeded, DL, budget_seconds=60).verdict == "true"


def test_uncontrolled_nondeterminism_forbids_true(tmp_path):
    p = _prog(tmp_path, "argv.py", """
        import sys
        if __name__ == "__main__":
            print(len(sys.argv))
    """)
    v = verify(p, DL, budget_seconds=60)
    assert v.verdict == "unknown"


def test_an_exception_outside_the_property_does_not_count(tmp_path):
    p = _prog(tmp_path, "val.py", """
        if __name__ == "__main__":
            int("x")
    """)
    assert verify(p, DL, budget_seconds=60).verdict == "true"


def test_cli_prints_the_verdict_last_and_writes_the_witness(tmp_path):
    p = _prog(tmp_path, "nd.py", """
        import _sv_verifier
        if __name__ == "__main__":
            [0][_sv_verifier.nondet_int()]
    """)
    prp = tmp_path / "dl.prp"
    prp.write_text("CHECK( init(main()), ! uncaught(IndexError))\n")
    w = tmp_path / "w.json"
    out = subprocess.run([sys.executable, "-m", "tharkun.sv", p, "--property", str(prp),
                          "--budget-seconds", "60", "--witness-out", str(w)],
                         capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "FALSE(IndexError)"
    assert json.loads(w.read_text())["format"] == "sv-py-witness/1"
    rep = subprocess.run([sys.executable, "-m", "tharkun.sv", p, "--property", str(prp),
                          "--replay", str(w)], capture_output=True, text=True)
    assert rep.stdout.strip().splitlines()[-1] == "CONFIRMED"


def test_a_draw_that_sizes_a_huge_allocation_is_capped_not_fatal(tmp_path):
    """[internal ref]: the child's address space is capped, so ``[None] * 2**31`` ends that run with
    MemoryError instead of taking the verifier down, and the search goes on."""
    p = _prog(tmp_path, "alloc.py", """
        import _sv_verifier
        if __name__ == "__main__":
            n = 1 + abs(_sv_verifier.nondet_int())
            points = [None] * n
            print(points[n])
    """)
    v = verify(p, DL, budget_seconds=60, run_timeout=30, run_memory_bytes=512 * 2**20)
    assert v.line() == "FALSE(IndexError)"
    q = _prog(tmp_path, "alloc_ok.py", """
        import _sv_verifier
        if __name__ == "__main__":
            n = 1 + abs(_sv_verifier.nondet_int())
            points = [None] * n
            print(points[0])
    """)
    v = verify(q, DL, budget_seconds=60, run_timeout=30, run_memory_bytes=512 * 2**20, jobs=1,
               prove=False)  # the search alone; absint proves points[0] safe
    assert v.verdict == "unknown"


def test_a_program_constant_is_a_candidate(tmp_path):
    """[internal ref]: the program's own literals (and neighbours) are tried early."""
    p = _prog(tmp_path, "magic.py", """
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            if x == 48611:
                [][0]
    """)
    v = verify(p, DL, budget_seconds=60, jobs=1)
    assert v.line() == "FALSE(IndexError)" and v.runs < 20
    assert v.witness["vector"] == [{"kind": "int", "value": 48611}]


def test_unbounded_domains_keep_growing(tmp_path):
    """[internal ref]: past the fixed candidates the int sequence keeps going (10**e, 2**k, ...)."""
    p = _prog(tmp_path, "big.py", """
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            if x > 2 ** 70:
                [][0]
    """)
    v = verify(p, DL, budget_seconds=120, jobs=2)
    assert v.line() == "FALSE(IndexError)"


def test_parallel_children_give_a_timing_independent_witness(tmp_path):
    """[internal ref]: a batch's outcomes are read in cost order, so for a given ``jobs`` the verdict
    and witness do not depend on which child finishes first."""
    p = _prog(tmp_path, "two.py", """
        import _sv_verifier
        xs = [0, 1, 2, 3, 4, 5]
        if __name__ == "__main__":
            a = abs(_sv_verifier.nondet_int())
            b = abs(_sv_verifier.nondet_int())
            print(xs[a + b])
    """)
    runs = [verify(p, DL, budget_seconds=60, jobs=4) for _ in range(3)]
    assert {v.line() for v in runs} == {"FALSE(IndexError)"}
    assert all(v.witness["vector"] == runs[0].witness["vector"] for v in runs)
    assert all(replay(p, v.witness, DL)[0] for v in runs[:1])


def test_parallel_children_fill_a_batch_from_one_chain(tmp_path, monkeypatch):
    """[internal ref]: a program with one unbounded draw has a single candidate chain.  Each popped
    value queues the chain's next value before the next pop, so a batch still takes ``jobs``
    values and the children run together; the witness is the one ``jobs=1`` finds."""
    from tharkun.sv import engine
    p = _prog(tmp_path, "one.py", """
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            if x == 48611:
                [][0]
    """)
    sizes = []

    class Pool(engine.ThreadPoolExecutor):
        def map(self, fn, items):
            items = list(items)
            sizes.append(len(items))
            return super().map(fn, items)

    monkeypatch.setattr(engine, "ThreadPoolExecutor", Pool)
    v4 = verify(p, DL, budget_seconds=60, jobs=4, concolic=False, stages=[s for s in _stages.DEFAULT if s != "hybrid"])
    assert v4.line() == "FALSE(IndexError)" and sizes and max(sizes) == 4
    v1 = verify(p, DL, budget_seconds=60, jobs=1, concolic=False, stages=[s for s in _stages.DEFAULT if s != "hybrid"])
    assert v4.witness["vector"] == v1.witness["vector"] == [{"kind": "int", "value": 48611}]


@pytest.mark.wave2
def test_parallel_children_share_the_cpu_budget(tmp_path):
    """[internal ref]: the budget is CPU across all children, so each of a batch's children may use
    only its share of what remains.  Otherwise one batch of long runs overshoots the budget
    ``jobs``-fold, and under BenchExec the CPU limit ends the verifier as TIMEOUT."""
    import resource
    p = _prog(tmp_path, "spin.py", """
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            while x != 0:
                pass
    """)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    v = verify(p, DL, budget_seconds=4, jobs=4, prove=False)  # the search alone
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime + after.ru_stime) - (before.ru_utime + before.ru_stime)
    assert v.verdict == "unknown" and cpu < 8


def test_candidate_sequences():
    from tharkun.sv.engine import Candidates
    c = Candidates({"int": [42, 43, 41, -42]})
    ints = [c.value("int", r) for r in range(12)]
    assert ints[:5] == [0, 1, -1, 2, -2] and ints[5:9] == [42, 43, 41, -42]
    assert c.value("bool", 1) is True and c.value("bool", 2) is None
    assert c.value("rindex", 2, n=3) == 2 and c.value("rindex", 3, n=3) is None
    assert c.value("int", 500) is not None          # unbounded kinds never end


def test_ablation_switches(tmp_path):
    """--no-constants / --no-deepening turn [internal ref] / [internal ref] off (for measuring what they buy)."""
    p = _prog(tmp_path, "magic2.py", """
        import _sv_verifier
        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            if x == 48611:
                [][0]
    """)
    assert verify(p, DL, budget_seconds=60, jobs=1).line() == "FALSE(IndexError)"
    off = verify(p, DL, budget_seconds=60, jobs=1, constants=False, deepen=False, stages=())
    assert off.verdict == "unknown"


def test_current_runtime_and_symlink_need_no_probe(tmp_path, monkeypatch):
    from tharkun.sv import engine

    link = tmp_path / "current-python"
    link.symlink_to(sys.executable)
    monkeypatch.setattr(engine.subprocess, "run", lambda *a, **k: pytest.fail("current runtime must not be probed"))
    assert engine._python314(sys.executable)
    assert engine._python314(str(link))


@pytest.mark.parametrize("outcome", ["cpython 3 14", "cpython 3 13", "pypy 3 14", "garbage", "nonzero", "timeout"])
def test_foreign_runtime_has_one_fresh_bounded_probe(tmp_path, monkeypatch, outcome):
    from types import SimpleNamespace
    from tharkun.sv import engine

    foreign = tmp_path / "foreign-python"
    foreign.write_text("")
    foreign.chmod(0o700)
    calls = []

    def probe(args, **kwargs):
        calls.append((args, kwargs))
        assert args == [str(foreign), "-I", "-c", "import sys; print(sys.implementation.name, *sys.version_info[:2])"]
        assert kwargs == dict(capture_output=True, text=True, timeout=2, check=True)
        if outcome == "nonzero":
            raise subprocess.CalledProcessError(1, args)
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(args, 2)
        return SimpleNamespace(stdout=outcome)

    monkeypatch.setattr(engine.subprocess, "run", probe)
    assert engine._python314(str(foreign)) is (outcome == "cpython 3 14")
    assert len(calls) == 1
    assert engine._python314(str(foreign)) is (outcome == "cpython 3 14")
    assert len(calls) == 2, "successful paths must not cache admission"


@pytest.mark.parametrize("options,flag", [
    ('options:\n  language_version: "3.15"\n', None),
    ('options:\n  language_version: "3.15"\n', "3.14"),
    ('options:\n  language_version: 3.14\n', "3.14"),
    ('options:\n  language_version: null\n', None),
    ('options:\n  language_version: "3.14"\n  language_version: "3.15"\n', None),
])
@pytest.mark.parametrize("body", ["pass\n", "assert True\n", "from _sv_verifier import nondet_int\nassert nondet_int() == 0\n"])
def test_version_refusal_precedes_every_analysis_and_replay(tmp_path, monkeypatch, options, flag, body):
    from tharkun.solve import census
    from tharkun.sv import engine, scheduler

    marker = tmp_path / "executed"
    program = tmp_path / "program.py"
    program.write_text(f"open({str(marker)!r}, 'w').write('ran')\n" + body)
    task = tmp_path / "task.yml"
    task.write_text("input_files: program.py\n" + options)
    blocked = lambda *a, **k: pytest.fail("admission must precede analysis or execution")
    for module, name in [(engine, "read_source"), (engine, "run_once"), (engine, "_static_proof"),
                         (census, "obligation_census"), (scheduler, "publish_module_namespace")]:
        monkeypatch.setattr(module, name, blocked)
    v = engine.verify(str(program), ("AssertionError",), task=str(task), language_version=flag,
                      witness_dir=tmp_path / "proofs")
    assert v.verdict == "unknown" and v.runs == 0 and v.witness is None
    assert "unsupported" in v.reason and not (tmp_path / "proofs").exists()
    ok, out = engine.replay(str(program), {"vector": []}, ("AssertionError",),
                            task=str(task), language_version=flag)
    assert not ok and out.status == "unsupported-runtime" and "unsupported" in out.msg
    assert not marker.exists()


def test_runtime_refusal_precedes_no_site_analysis_and_keeps_legacy_calls(tmp_path, monkeypatch):
    from tharkun.sv import engine

    program = tmp_path / "program.py"
    program.write_text("pass\n")
    monkeypatch.setattr(engine, "_python314", lambda python: False)
    monkeypatch.setattr(engine, "read_source", lambda *a: pytest.fail("source analysis must not start"))
    v = engine.verify(str(program), ("IndexError",), language_version="3.14")
    assert v.verdict == "unknown" and v.runs == 0 and "unsupported runtime" in v.reason
    environment, reason = engine.admit_task(str(program), python="missing-python")
    assert reason is None and environment.files == (str(program),)
    monkeypatch.setattr(engine, "run_once", lambda *a, **k: engine.Outcome("clean"))
    assert engine.replay(str(program), {"vector": []}, ("IndexError",), python="missing-python")[1].status == "clean"


def test_verifier_runtime_itself_must_be_cpython314(tmp_path, monkeypatch):
    from tharkun.sv import engine

    program = tmp_path / "program.py"
    program.write_text("pass\n")
    monkeypatch.setattr(engine.sys, "version_info", (3, 13, 0))
    monkeypatch.setattr(engine, "_python314", lambda *a: pytest.fail("wrong verifier must refuse first"))
    assert "verifier requires" in engine.admit_task(str(program), language_version="3.14")[1]


@pytest.mark.parametrize("descriptor", [1, 2])
def test_foreign_runtime_refuses_undecodable_probe_bytes(tmp_path, descriptor):
    from tharkun.sv import engine

    foreign = tmp_path / "foreign-python"
    foreign.write_text(f"#!{sys.executable}\nimport os\nos.write(1, b'cpython 3 14\\n')\n"
                       f"os.write({descriptor}, b'\\xff')\n")
    foreign.chmod(0o700)
    assert not engine._python314(str(foreign))

def test_task_primary_input_must_match_requested_program_before_analysis(tmp_path, monkeypatch):
    from tharkun.sv import engine

    main = tmp_path / "z_main.py"
    main.write_text("raise IndexError\n")
    helper = tmp_path / "a_helper.py"
    helper.write_text("pass\n")
    task = tmp_path / "task.yml"
    task.write_text("input_files: z_main.py\nrequired_files: a_helper.py\n")
    monkeypatch.setattr(engine, "read_source", lambda *a: pytest.fail("must not analyze a helper as main"))
    verdict = engine.verify(str(helper), ("IndexError",), task=str(task), canary_seconds=0)
    assert verdict.verdict == "unknown" and verdict.runs == 0 and verdict.witness is None
    assert "primary input_files" in verdict.reason
    ok, out = engine.replay(str(helper), {"vector": []}, ("IndexError",), task=str(task))
    assert not ok and out.status == "unsupported-runtime" and "primary input_files" in out.msg


def test_metadata_free_api_still_bypasses_invocation_admission(tmp_path, monkeypatch):
    from tharkun.sv import engine

    program = tmp_path / "bare.py"
    program.write_text("pass\n")
    monkeypatch.setattr(engine, "_python314", lambda *a: pytest.fail("legacy calls do not probe"))
    assert engine.admit_task(str(program), python="missing-python")[1] is None
    monkeypatch.setattr(engine, "run_once", lambda *a, **k: engine.Outcome("clean"))
    assert engine.replay(str(program), {"vector": []}, ("IndexError",), python="missing-python")[1].status == "clean"
