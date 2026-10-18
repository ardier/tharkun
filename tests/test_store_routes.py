"""Deciding provers retain local premises until their exact TRUE reaches done()."""
import ast
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import pytest

from tharkun.solve.census import obligation_census
from tharkun.solve.sitekey import site_keys
from tharkun.sv import engine, fact_proof, product_proof, prove, scheduler, store_witness
from tharkun.sv import eregion_bridge as bridge
from tharkun.sv.obligation_table import ObligationTable, call_target_evidence_in, call_targets_in
from tests.fake_producers import NOMINATE_ALL, NOMINATE_FIRST, install


def task(tmp_path, source):
    path = tmp_path / "task.py"
    path.write_text(textwrap.dedent(source).lstrip("\n"))
    return str(path), path.read_text()


def source_table(source, filename, classes):
    tree = ast.parse(source)
    sites = site_keys(tree, source, filename)
    table = ObligationTable(obligation_census(source, classes, filename))
    table.bind_source(sites, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      classes=classes, location=sites[tree.body[0]].location())
    return table


@pytest.fixture
def finalized(monkeypatch):
    tables, decisions = [], []
    record_route = store_witness.record_route

    def create(*args, **kwargs):
        table = ObligationTable(*args, **kwargs)
        tables.append(table)
        return table

    def record(table, verdict):
        assert table.proof_routes() == ()  # no successful stage publishes early
        decisions.append(verdict)
        record_route(table, verdict)

    monkeypatch.setattr(scheduler, "ObligationTable", create)
    monkeypatch.setattr(store_witness, "record_route", record)
    return tables, decisions


@pytest.mark.parametrize("route,source,classes,options", [
    ("no-site", "assert True\nx = 1\n", ("AssertionError",), {}),
    ("product", "import _sv_verifier\nx = _sv_verifier.nondet_int()\nassert x // 2 == x >> 1\n",
     ("AssertionError",), {}),
    ("determinism", "xs = [1, 2, 3]\nprint(xs[2])\n", ("IndexError",), {"prove": False}),
    ("stationary", """
        import _sv_verifier
        import re
        def work(loops):
            for _ in range(loops):
                m = re.match("a(b)", "ab")
                g = m.group(1)
        work(abs(_sv_verifier.nondet_int()))
        """, ("KeyError", "IndexError"), {}),
])
def test_real_deciding_routes_publish_only_the_final_verdict(
        tmp_path, monkeypatch, finalized, route, source, classes, options):
    program, source = task(tmp_path, source)
    if route in {"product", "stationary"}:
        monkeypatch.setattr(engine, "_static_proof", lambda *a, **kw: None)
    result = engine.verify(program, classes, budget_seconds=60, jobs=1, canary_seconds=0,
                           concolic=False, stages=[], **options)
    tables, decisions = finalized
    assert result.verdict == "true" and decisions == [result]
    assert decisions[0] is result
    (evidence,) = tables[0].proof_routes()
    assert evidence.value["route"] == route and evidence.value["whole_program"] is True
    assert evidence.value["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert evidence.value["classes"] == tuple(sorted(classes))
    assert evidence.value["location"] == (program, 1, 1, "<module>")
    assert evidence.derivations[0].producer == result.component
    parents = evidence.derivations[0].parents
    if route == "stationary":
        assert parents and parents == tuple(e.identity for e in call_target_evidence_in(tables[0].kb))
        assert all(kind == "call_target" for kind, _ in parents)
        assert any(f.kind == "loop_trip_count" for f in tables[0].kb)
    else:
        assert parents == ()


@pytest.mark.parametrize("source,no_goals", [
    ("import _sv_verifier\nx = _sv_verifier.nondet_int()\nif False:\n    assert False\n", True),
    ("import _sv_verifier\nx = _sv_verifier.nondet_int()\nassert x // 2 == x >> 1\n", False),
])
def test_both_actual_product_success_branches_bind_a_route(tmp_path, source, no_goals):
    program, source = task(tmp_path, source)
    classes = ("AssertionError",)
    table = source_table(source, program, classes)
    verdict = product_proof.prove(source, classes, program=program, python=sys.executable)
    assert verdict is not None and verdict.verdict == "true"
    assert ("no site at all" in verdict.reason) is no_goals
    assert table.proof_routes() == ()
    store_witness.record_route(table, verdict)
    assert table.proof_routes()[0].value["route"] == "product"


@pytest.mark.parametrize("source", [
    "if False:\n    assert False\n",
    "import _sv_verifier\nif False:\n    assert False\n",
])
def test_product_without_a_modeled_draw_cannot_supply_a_route(tmp_path, source):
    program, source = task(tmp_path, source)
    table = source_table(source, program, ("AssertionError",))
    verdict = product_proof.prove(source, ("AssertionError",), program=program, python=sys.executable)
    assert verdict is None
    store_witness.record_route(table, verdict)
    assert table.proof_routes() == ()


def test_product_unknown_and_no_site_refusal_leave_no_route(tmp_path, monkeypatch):
    program, source = task(tmp_path, "import _sv_verifier\nx = _sv_verifier.nondet_int()\nassert x > 0\n")
    table = source_table(source, program, ("AssertionError",))
    monkeypatch.setattr(product_proof.z3.Solver, "check", lambda self: product_proof.z3.unknown)
    assert product_proof.prove(source, ("AssertionError",), program=program) is None
    assert prove.prove_static(program, ("AssertionError",)) is None
    assert table.proof_routes() == ()


def test_no_site_refuses_an_exception_without_a_census_table(tmp_path):
    program, _ = task(tmp_path, "pass\n")
    messages = []
    assert prove.prove_static(program, ("TypeError",), log=messages.append) is None
    assert messages == ["no-site: no site table for TypeError"]


def test_two_runs_without_a_determinism_certificate_publish_nothing(tmp_path, monkeypatch, finalized):
    program, _ = task(tmp_path, "xs = [1, 2, 3]\nprint(xs[2])\n")
    monkeypatch.setattr(engine, "_certify_determinism", lambda *a, **kw: None)
    verdict = engine.verify(program, ("IndexError",), prove=False, budget_seconds=60,
                            jobs=1, concolic=False, stages=[])
    tables, decisions = finalized
    assert verdict.verdict == "unknown" and decisions == [] and tables[0].proof_routes() == ()


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="producer source checker requires CPython 3.14")
def test_real_producer_proof_reaches_the_single_final_route_writer(tmp_path, monkeypatch, finalized):
    producer = tmp_path / "producer"
    producer.mkdir()
    install(producer, monkeypatch, NOMINATE_ALL, only=True)
    program, _ = task(tmp_path, """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        d = 2 * x + 1
        answer = 10 // d
    """)
    monkeypatch.setattr(engine, "_static_proof", lambda *a, **kw: None)
    verdict = engine.verify(program, ("ZeroDivisionError",), budget_seconds=60, jobs=1,
                            canary_seconds=0, concolic=False, stages=[])
    tables, decisions = finalized
    assert verdict.verdict == "true" and verdict.component == "proof:facts:fake"
    assert decisions == [verdict] and decisions[0] is verdict
    (route,) = tables[0].proof_routes()
    assert route.value["route"] == "producer"
    assert route.derivations[0].parents == tuple(
        e.identity for e in tables[0].kb.evidence(kind=bridge.CERTIFIED_INVARIANT))


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="producer source checker requires CPython 3.14")
@pytest.mark.parametrize("all_rows", [True, False])
def test_producer_route_has_only_this_invocations_checked_parents(tmp_path, monkeypatch, all_rows):
    producer = tmp_path / "producer"
    producer.mkdir()
    install(producer, monkeypatch, NOMINATE_ALL if all_rows else NOMINATE_FIRST, only=True)
    program, source = task(tmp_path, """
        import _sv_verifier
        x = _sv_verifier.nondet_int()
        d1 = 2 * x + 1
        a = 10 // d1
        d2 = 2 * x + 3
        b = 20 // d2
    """)
    classes = ("ZeroDivisionError",)
    table = source_table(source, program, classes)
    # Historical certification must not supply the un-nominated row in this invocation.
    for row in table.rows(classes):
        inv = row.invariant
        bridge.publish_fact(table.kb, {"kind": "invariant_candidate", "site": {
            "file": program, "line": inv.line, "column": inv.column, "function": inv.function},
            "value": inv.value, "evidence": {"engine": "fake"} if all_rows else {}}, producer="historical")
        bridge.publish_certified(table.kb, inv, checker=fact_proof.CHECKER, producer="fact_proof")
    # Stored nominations are deliberately eligible for rechecking; refuse the last current check.
    if not all_rows:
        from tharkun.sv import validate
        check = validate.check_candidates_individually
        last = table.rows(classes)[-1].invariant

        def current_check(*args, **kwargs):
            return {last: False} if args[4] == [last] else check(*args, **kwargs)

        monkeypatch.setattr(validate, "check_candidates_individually", current_check)
    verdict = fact_proof.prove_from_facts(
        program, classes, source=source, specification="CHECK( init(main()), ! uncaught(ZeroDivisionError))\n",
        table=table, python=sys.executable, remaining=lambda: 60, say=lambda msg: None)
    assert table.proof_routes() == ()
    if not all_rows:
        assert verdict is None
        return
    assert verdict is not None and verdict.verdict == "true"
    store_witness.record_route(table, verdict)
    (route,) = table.proof_routes()
    assert route.value["route"] == "producer"
    assert route.derivations[0].parents == tuple(dict.fromkeys(
        table.kb.evidence_item(bridge.CERTIFIED_INVARIANT, bridge.invariant_key(o.invariant)).identity
        for o in table.rows(classes)))


@pytest.mark.parametrize("capture_fails", [False, True])
def test_stationary_parents_are_captured_at_plan_read_not_after_probing(tmp_path, monkeypatch, capture_fails):
    program, source = task(tmp_path, """
        import _sv_verifier
        DATA = [3, 1, 2]
        def work(loops):
            for _ in range(loops):
                s = sorted(DATA)
                total = s[0] + s[2]
        work(abs(_sv_verifier.nondet_int()))
    """)
    classes = ("KeyError", "IndexError")
    table = source_table(source, program, classes)
    scheduler.publish_module_namespace(table, program)
    captured = call_target_evidence_in(table.kb)
    assert captured and call_targets_in(table.kb) == [e.value for e in captured]
    assert call_targets_in(None) == [] and call_target_evidence_in(None) == ()
    expected_parents = tuple(e.identity for e in captured)
    capture_errors = []
    if capture_fails:
        from tharkun.knowledge import Evidence

        def fail_identity(self):
            capture_errors.append(self.key)
            raise RuntimeError("route parent capture unavailable")

        monkeypatch.setattr(Evidence, "identity", property(fail_identity))
    stationary = prove._stationary

    def probe(*args, **kwargs):
        result = stationary(*args, **kwargs)
        # This genuine but later publication was not consumed by the draw plan.
        table.kb.publish("call_target", "later", {"proven": False}, producer="test")
        return result

    monkeypatch.setattr(prove, "_stationary", probe)
    verdict = prove.prove_dynamic(program, classes, python=sys.executable, remaining=lambda: 60,
                                  knowledge=table.kb)
    assert verdict is not None and verdict.verdict == "true" and table.proof_routes() == ()
    store_witness.record_route(table, verdict)
    if capture_fails:
        assert capture_errors and table.proof_routes() == ()
    else:
        assert table.proof_routes()[0].derivations[0].parents == expected_parents


def test_nonstationary_observations_are_not_route_premises(tmp_path):
    program, source = task(tmp_path, """
        import _sv_verifier
        DATA = []
        def work(loops):
            for _ in range(loops):
                DATA.append(1)
                first = DATA[0]
        work(abs(_sv_verifier.nondet_int()))
    """)
    classes = ("IndexError",)
    table = source_table(source, program, classes)
    verdict = prove.prove_dynamic(program, classes, python=sys.executable, remaining=lambda: 60,
                                  knowledge=table.kb)
    assert verdict is None
    store_witness.record_route(table, verdict)
    assert table.proof_routes() == ()


@pytest.mark.parametrize("change", ["object", "source", "filename", "classes", "component", "false", "unknown"])
def test_bound_route_cannot_be_reused_for_another_decision(tmp_path, change):
    program, source = task(tmp_path, "assert True\nx = 1\n")
    classes = ("AssertionError",)
    verdict = prove.prove_static(program, classes)
    assert verdict is not None
    original = asdict(verdict)
    assert "_store_route" not in original and set(verdict.detail) == {"classes"}
    if change == "object":
        verdict = copy.copy(verdict)
    elif change == "source":
        source = "assert True\nx = 2\n"
    elif change == "filename":
        program += ".other"
    elif change == "classes":
        classes = ("IndexError",)
    elif change == "component":
        verdict.component = "deterministic"
    else:
        verdict.verdict = change
    table = source_table(source, program, classes)
    store_witness.record_route(table, verdict)
    assert table.proof_routes() == ()


@pytest.mark.parametrize("verdict,route,parents", [
    (engine.Verdict("false", component="proof:no-site"), "no-site", ()),
    (engine.Verdict("true", component="symbolic:product"), "no-site", ()),
    (engine.Verdict("true", component="proof:facts:fake"), "producer", ()),
    (engine.Verdict("true", component="proof:stationary"), "stationary", (("loop_trip_count", "fake"),)),
])
def test_metadata_binding_never_converts_an_unrelated_result(verdict, route, parents):
    before = asdict(verdict)
    assert store_witness.bind_route(verdict, route, "pass\n", "task.py", ("TypeError",), parents=parents) is verdict
    assert not hasattr(verdict, "_store_route") and asdict(verdict) == before


def test_route_capture_and_publication_failures_preserve_the_proof(tmp_path, monkeypatch):
    program, source = task(tmp_path, "assert True\nx = 1\n")
    verdict = prove.prove_static(program, ("AssertionError",))
    before = asdict(verdict)
    table = source_table(source, program, ("AssertionError",))

    def fail(*args, **kwargs):
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr(table, "publish_route", fail)
    store_witness.record_route(table, verdict)
    monkeypatch.setattr(store_witness, "_RouteDecision", fail)
    assert prove.prove_static(program, ("AssertionError",)).verdict == "true"
    assert table.proof_routes() == () and asdict(verdict) == before


@pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="legacy source checker requires CPython 3.14")
@pytest.mark.parametrize("mode", ["published", "capture_error", "publication_error"])
def test_real_prover_route_preserves_legacy_witness_bytes(tmp_path, monkeypatch, finalized, mode):
    from tharkun.solve import obligations

    monkeypatch.setattr(obligations, "uuid4", lambda: "00000000-0000-4000-8000-000000000000")
    instant = datetime(2026, 10, 13, tzinfo=timezone.utc)
    monkeypatch.setattr(obligations, "datetime", SimpleNamespace(now=lambda tz: instant))
    program, _ = task(tmp_path, "assert True\nx = 1\n")
    options = dict(budget_seconds=60, jobs=1, canary_seconds=0, concolic=False, stages=[])
    with monkeypatch.context() as old_path:
        old_path.setattr(prove, "bind_route", lambda verdict, *a, **kw: verdict)
        baseline = engine.verify(program, ("AssertionError",), witness_dir=tmp_path / "baseline", **options)
    expected = Path(baseline.detail["witness_path"]).read_bytes()
    failures = []

    def fail(*args, **kwargs):
        failures.append(mode)
        raise RuntimeError("route metadata unavailable")

    if mode == "capture_error":
        monkeypatch.setattr(store_witness, "_RouteDecision", fail)
    elif mode == "publication_error":
        monkeypatch.setattr(ObligationTable, "publish_route", fail)
    actual = engine.verify(program, ("AssertionError",), witness_dir=tmp_path / "candidate", **options)
    assert actual.verdict == baseline.verdict == "true"
    assert actual.component == baseline.component == "proof:no-site"
    assert actual.detail["proof_checked"] is baseline.detail["proof_checked"] is True
    assert actual.detail["proof_reason"] == baseline.detail["proof_reason"]
    assert actual.detail["correctness_witness"] == baseline.detail["correctness_witness"]
    assert Path(actual.detail["witness_path"]).read_bytes() == expected
    tables, decisions = finalized
    assert decisions[0] is baseline and decisions[1] is actual
    assert tables[0].proof_routes() == ()
    assert len(tables[1].proof_routes()) == (1 if mode == "published" else 0)
    assert failures == ([] if mode == "published" else [mode])


def test_confirmed_canary_false_discards_bound_no_site_proposal(tmp_path, monkeypatch, finalized):
    program, source = task(tmp_path, "raise TypeError('violating child')\n")
    provisional = engine.Verdict("true", component="proof:no-site")
    store_witness.bind_route(provisional, "no-site", source, program, ("TypeError",))
    monkeypatch.setattr(engine, "_static_proof", lambda *a, **kw: provisional)
    result = engine.verify(program, ("TypeError",), budget_seconds=60, run_timeout=30,
                           jobs=1, canary_seconds=5, concolic=False, stages=[])
    tables, decisions = finalized
    assert result.verdict == "false" and result.exception == "TypeError" and result is not provisional
    assert result.detail["unsound_proof"] == "proof:no-site"
    assert tables[0].proof_routes() == () and decisions == []
