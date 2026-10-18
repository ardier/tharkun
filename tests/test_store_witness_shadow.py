"""Store shadow work runs after, and cannot alter, the frozen witness path."""
import ast
import copy
import hashlib
import json
import signal
import threading
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tharkun.solve import obligations as obligation_types
from tharkun.solve import terms, terms_z3
from tharkun.solve.census import obligation_census
from tharkun.solve.obligations import Invariant
from tharkun.solve.sitekey import site_keys
from tharkun.sv import proof, store_witness
from tharkun.sv.engine import Verdict
from tharkun.sv.obligation_table import ObligationTable
from tharkun.sv.validate import Validation


def source_table(source, filename, classes=("TypeError",)):
    tree = ast.parse(source)
    sites = site_keys(tree, source, filename)
    table = ObligationTable(obligation_census(source, classes, filename))
    location = sites[tree.body[0]].location() if tree.body else (filename, 1, 1, "<module>")
    table.bind_source(sites, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      classes=classes, location=location)
    return table


@pytest.fixture
def frozen_metadata(monkeypatch):
    monkeypatch.setattr(obligation_types, "uuid4", lambda: "00000000-0000-4000-8000-000000000000")
    instant = datetime(2026, 10, 13, tzinfo=timezone.utc)
    monkeypatch.setattr(obligation_types, "datetime", SimpleNamespace(now=lambda tz: instant))


def finish_task(tmp_path, monkeypatch, *, project, label):
    source = "left = 6\nright = 2\nanswer = left / right\n"
    filename = str(tmp_path / "task.py")
    (tmp_path / "task.py").write_text(source)
    classes = ("ZeroDivisionError",)
    table = source_table(source, filename, classes)
    monkeypatch.setattr(proof, "check_candidates", lambda *a, **kw: Validation(True, (), "frozen check"))
    monkeypatch.setattr(store_witness, "project", project)
    result = proof.finish(Verdict("true", component="proof:interval"), source, filename,
                          "CHECK( init(main()), ! uncaught(ZeroDivisionError))\n", table.census,
                          directory=tmp_path / label, knowledge=table.kb, table=table)
    return result


def test_shadow_starts_after_frozen_check_and_export(tmp_path, monkeypatch, frozen_metadata):
    baseline = finish_task(tmp_path, monkeypatch, project=lambda *a, **kw: ((), {}), label="baseline")
    from pathlib import Path

    expected = Path(baseline.detail["witness_path"]).read_bytes()

    def project(table, **kwargs):
        legacy = list((tmp_path / "candidate").glob("*/witness.yml"))
        assert len(legacy) == 1 and legacy[0].read_bytes() == expected
        return (Invariant(str(tmp_path / "task.py"), 3, 1, "<module>", "right != 0"),), {"checked_sites": 1}

    result = finish_task(tmp_path, monkeypatch, project=project, label="candidate")
    assert result.verdict == baseline.verdict == "true"
    assert result.detail["proof_checked"] is baseline.detail["proof_checked"] is True
    assert result.detail["proof_reason"] == baseline.detail["proof_reason"]
    assert result.detail["correctness_witness"] == baseline.detail["correctness_witness"]
    assert Path(result.detail["witness_path"]).read_bytes() == expected
    assert Path(result.detail["store_witness_path"]).name == "store-witness.yml"
    assert result.detail["store_checked_sites"] == 1


@pytest.mark.parametrize("failure", [RuntimeError("projection failed"), ValueError("invalid record")])
def test_shadow_exceptions_preserve_frozen_verdict_and_bytes(tmp_path, monkeypatch, frozen_metadata, failure):
    from pathlib import Path

    baseline = finish_task(tmp_path, monkeypatch, project=lambda *a, **kw: ((), {}), label="baseline")

    def fail(*args, **kwargs):
        raise failure

    result = finish_task(tmp_path, monkeypatch, project=fail, label="failed")
    assert result.verdict == "true" and result.detail["proof_checked"]
    assert Path(result.detail["witness_path"]).read_bytes() == Path(baseline.detail["witness_path"]).read_bytes()
    assert result.detail["correctness_witness"] == baseline.detail["correctness_witness"]
    assert result.detail["store_error"]


def test_shadow_write_failure_preserves_legacy_export(tmp_path, monkeypatch, frozen_metadata):
    from pathlib import Path

    baseline = finish_task(tmp_path, monkeypatch, project=lambda *a, **kw: ((), {}), label="baseline")
    old_write = proof._write

    def write(path, text):
        if path.name == "store-witness.yml":
            raise OSError("shadow output is read-only")
        old_write(path, text)

    monkeypatch.setattr(proof, "_write", write)
    inv = Invariant(str(tmp_path / "task.py"), 3, 1, "<module>", "right != 0")
    result = finish_task(tmp_path, monkeypatch, project=lambda *a, **kw: ((inv,), {}), label="failed")
    assert result.verdict == "true" and result.detail["proof_checked"]
    assert Path(result.detail["witness_path"]).read_bytes() == Path(baseline.detail["witness_path"]).read_bytes()
    assert "read-only" in result.detail["store_error"]


@pytest.mark.parametrize("before_serialization", [False, True])
def test_export_deadline_keeps_existing_reasons_and_reports_missing_file(
        tmp_path, monkeypatch, before_serialization):
    table, _ = scalar_table()
    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return 2 if calls >= (2 if before_serialization else 3) else 0

    monkeypatch.setattr(store_witness, "time", SimpleNamespace(monotonic=clock))
    invariant = Invariant("task.py", 3, 1, "<module>", "left < right")
    monkeypatch.setattr(store_witness, "project", lambda *a, **kw: ((invariant,), {
        "reasons": {"store_only:implication_unknown": 1}, "sites_established_and_rendered": 1}))
    detail = store_witness.shadow(table, "left = 1\nright = 2\nanswer = left < right\n", "task.py",
                                  "CHECK( init(main()), ! uncaught(TypeError))\n", directory=tmp_path, seconds=1)
    assert detail["store_reasons"] == {"store_only:implication_unknown": 1, "store_only:shadow_budget": 1}
    assert detail["store_sites_established_and_rendered"] == 1
    assert not (tmp_path / "store-witness.yml").exists()


@pytest.mark.parametrize("route,source,prove", [
    ("absint", "left = 1\nright = 2\nanswer = left < right\n", True),
    ("finite_tree", "from _sv_verifier import nondet_bool\nflag = nondet_bool()\nif flag:\n    pass\n", False),
], ids=["absint", "finite-tree"])
def test_real_scheduler_shadow_ledger_budget_preserves_legacy_bytes(
        tmp_path, monkeypatch, frozen_metadata, record_property, route, source, prove):
    from pathlib import Path
    from tharkun.sv import engine, scheduler
    from tharkun.sv.witness import loads

    program = tmp_path / "task.py"
    program.write_text(source)
    tables = []

    def table(*args, **kwargs):
        result = ObligationTable(*args, **kwargs)
        tables.append(result)
        return result

    monkeypatch.setattr(scheduler, "ObligationTable", table)
    options = dict(budget_seconds=60, run_timeout=30, jobs=1, prove=prove,
                   concolic=False, stages=[], canary_seconds=0)
    with monkeypatch.context() as baseline_patch:
        baseline_patch.setattr(store_witness, "shadow", lambda *a, **kw: {})
        baseline = engine.verify(str(program), ("TypeError",),
                                 witness_dir=tmp_path / "baseline", **options)
    candidate = engine.verify(str(program), ("TypeError",),
                              witness_dir=tmp_path / "candidate", **options)
    assert candidate.verdict == baseline.verdict == "true"
    assert candidate.component == baseline.component
    assert candidate.detail["proof_checked"] is baseline.detail["proof_checked"] is True
    assert candidate.detail["correctness_witness"] == baseline.detail["correctness_witness"]
    for key in ("witness_path", "obligations_path"):
        assert Path(candidate.detail[key]).read_bytes() == Path(baseline.detail[key]).read_bytes()
    assert [e.value["route"] for e in tables[-1].proof_routes()] == [route]
    details = {key: value for key, value in candidate.detail.items() if key.startswith("store_")}
    record_property("ledger_budget_shadow", json.dumps(details, sort_keys=True))
    assert "store_error" not in details
    if "store_witness_path" in details:
        document = loads(Path(details["store_witness_path"]).read_text())
        values = [Invariant.from_entry(inv).value for entry in document for inv in entry["content"]]
        assert values
        if route == "absint":
            assert tables[-1].site_records() and any(value != "True" for value in values)
    else:
        assert details["store_reasons"].get("store_only:shadow_budget", 0) > 0


def test_additive_source_binding_failure_preserves_requested_loop_locations(tmp_path, monkeypatch):
    from tharkun.sv import engine

    program = tmp_path / "loop.py"
    program.write_text("left = 1\nright = 2\nwhile left < right:\n    left += 1\n")
    requested = []

    def fail(*args, **kwargs):
        raise ValueError("additive context failure")

    def static(program, classes, remaining, say, files=None, invariant_locations=(), table=None):
        requested.extend(invariant_locations)
        return Verdict("true", component="test")

    monkeypatch.setattr(ObligationTable, "bind_source", fail)
    monkeypatch.setattr(engine, "_static_proof", static)
    monkeypatch.setattr(proof, "check_candidates", lambda *a, **kw: Validation(True, ()))
    result = engine.verify(str(program), ("TypeError",), budget_seconds=60, canary_seconds=0,
                           concolic=False, stages=[])
    assert result.verdict == "true" and (str(program), 3, 1, "<module>") in requested
    assert result.detail["store_reasons"] == {"store_only:source_context": 1}


@pytest.mark.parametrize("left,expected", [(0, 0), (4, 0.2), (100, 1.0)])
def test_scheduler_passes_remaining_ledger_share_to_shadow(tmp_path, monkeypatch, left, expected):
    from tharkun.sv import engine, scheduler

    program = tmp_path / "task.py"
    program.write_text("pass\n")
    seen = []
    monkeypatch.setattr(scheduler.Ledger, "remaining", lambda self: left)
    monkeypatch.setattr(engine, "_static_proof", lambda *a, **kw: Verdict("true", component="test"))
    monkeypatch.setattr(proof, "check_candidates", lambda *a, **kw: Validation(True, ()))

    def shadow(*args, **kwargs):
        seen.append(kwargs["seconds"])
        return {}

    monkeypatch.setattr(store_witness, "shadow", shadow)
    result = engine.verify(str(program), ("TypeError",), budget_seconds=30, canary_seconds=0,
                           concolic=False, stages=[])
    assert result.verdict == "true" and seen == [expected]


def test_shadow_runs_in_worker_threads_without_taking_process_signals(monkeypatch):
    table = source_table("x = 1\n", "task.py")
    called = []

    def project(*args, **kwargs):
        called.append(threading.current_thread())
        return (), {"checked_sites": 0}

    monkeypatch.setattr(store_witness, "project", project)
    monkeypatch.setattr(signal, "signal", lambda *a, **kw: pytest.fail("must not install signal handlers"))
    if hasattr(signal, "setitimer"):
        monkeypatch.setattr(signal, "setitimer", lambda *a, **kw: pytest.fail("must not install timers"))
    results = []
    worker = threading.Thread(target=lambda: results.append(
        store_witness.shadow(table, "x = 1\n", "task.py", "property", seconds=0.03)))
    worker.start()
    worker.join(timeout=2)
    assert not worker.is_alive() and called == [worker]
    assert results == [{"store_checked_sites": 0}]


def scalar_record(table, *, operation=ast.Compare, occurrence=0):
    node, site = [(node, site) for node, site in table.source_context.sites.items()
                  if isinstance(node, operation)][occurrence]
    if isinstance(node, ast.Compare):
        receiver, args, op, qualname = node.left, node.comparators, node.ops[0], "protocol.comparison"
    elif isinstance(node, ast.UnaryOp):
        receiver, args, op, qualname = node.operand, [], node.op, "protocol.unary_operator"
    elif isinstance(node, ast.AugAssign):
        receiver, args, op, qualname = node.target, [node.value], node.op, "protocol.binary_operator"
    else:
        receiver, args, op, qualname = node.left, [node.right], node.op, "protocol.binary_operator"

    def encode(value):
        return terms.canon(terms.var(value.id) if isinstance(value, ast.Name) else terms.literal(value.value))

    names = {value.id for value in (receiver, *args) if isinstance(value, ast.Name)}
    return {"site": [site.source_sha256, site.ast_path, "TypeError"], "status": "safe",
            "builtins": ["bool", "float", "int", "str", "type"],
            "contract": {"qualname": qualname, "exception": "TypeError"},
            "binding": {"receiver": encode(receiver), "args": [encode(value) for value in args],
                        "op": type(op).__name__},
            "facts": sorted(terms.canon(terms.typeis(terms.var(name), "int")) for name in names)}


def scalar_table(source="left = 1\nright = 2\nanswer = left < right\n"):
    table = source_table(source, "task.py")
    record = scalar_record(table)
    table.publish_site_records([record], producer="absint")
    return table, record


def test_typed_store_record_projects_real_safety_with_real_parents():
    table, _ = scalar_table()
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert metrics["established_sites"] == metrics["checked_sites"] == 1
    assert metrics["safe_sites"] == metrics["sites_established_and_rendered"] == 1
    assert invariants and all(inv.value != "True" for inv in invariants)
    assert (invariants[0].line, invariants[0].type) == (3, "location_invariant")
    proposed, established = table.safety()
    assert proposed.value["status"] == "proposed" and established.value["status"] == "established"
    assert proposed.value["canon"] == established.value["canon"]
    assert not proposed.certified and not established.certified
    assert established.value["parents"] == tuple(item.identity for item in table.site_record_evidence())
    builtins = {"type": type, "int": int, "bool": bool, "float": float, "str": str}
    assert eval(invariants[0].value, {"__builtins__": builtins}, {"left": 1, "right": 2}) is True
    # A selected int/int case is stronger than the whole scalar admission.
    assert eval(invariants[0].value, {"__builtins__": builtins}, {"left": True, "right": 2}) is False


def test_multiple_consistent_renderable_cases_project_their_disjunction():
    table, record = scalar_table()
    record["facts"] = sorted([
        terms.canon(terms.or_(terms.typeis(terms.var("left"), "int"), terms.typeis(terms.var("left"), "bool"))),
        terms.canon(terms.typeis(terms.var("right"), "int")),
    ])
    table.publish_site_records([record], producer="absint")
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert metrics["established_sites"] == 1 and len(invariants) == 1
    builtins = {"type": type, "int": int, "bool": bool, "float": float, "str": str}
    for left, expected in ((1, True), (True, True), (1.0, False)):
        assert eval(invariants[0].value, {"__builtins__": builtins}, {"left": left, "right": 2}) is expected


def test_consistent_none_case_stays_store_only_without_renderer_folding():
    table, record = scalar_table("left = 1\nright = 2\nanswer = left == right\n")
    record["facts"] = sorted([
        terms.canon(terms.or_(terms.typeis(terms.var("left"), "int"), terms.typeis(terms.var("left"), "NoneType"))),
        terms.canon(terms.typeis(terms.var("right"), "int")),
    ])
    table.publish_site_records([record], producer="absint")
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert metrics["established_sites"] == 1 and invariants == ()
    assert metrics["reasons"] == {"store_only:case_not_singled_out": 1}


@pytest.mark.parametrize("answer", ["yes", "no", "unknown", terms.Refused("unsupported")])
def test_exact_tag_domain_selects_later_case_with_one_confirmation(answer):
    from tharkun.solve.contracts.safety import Binding, case_safety, cases

    left, right = terms.var("left"), terms.var("right")
    facts = (terms.typeis(left, "str"), terms.typeis(right, "str"))
    choices = cases("protocol.comparison", "TypeError", Binding(left, (right,), "Lt"))
    matching = next(case for case in choices if terms.canon(case.domain) == terms.canon(terms.and_(*facts)))
    assert matching != choices[0]
    queried = []

    def query(goal):
        queried.append(goal)
        return answer

    result = store_witness._case_projection(choices, facts, query, ("str", "type"), float("inf"))
    assert queried == [case_safety(matching)]
    if answer == "yes":
        assert isinstance(result, str) and "str" in result
    else:
        assert isinstance(result, terms.Refused) and result.reason == "case_not_singled_out"


def test_value_refined_case_domain_keeps_its_bound_in_fallback_query():
    from tharkun.solve.contracts.safety import Case

    value = terms.var("value")
    tag = terms.typeis(value, "int")
    domain = terms.and_(tag, terms.gt(terms.as_int(value), terms.literal(0)))
    queries = []

    def query(goal):
        queries.append(goal)
        assert goal == domain
        return "yes"

    result = store_witness._case_projection((Case(domain, None),), (tag,), query,
                                            ("int", "type"), float("inf"))
    assert queries == [domain] and "> 0" in result


@pytest.mark.parametrize("source,operation,line,function", [
    ("left = 1\nright = 2\nanswer = left + right\n", ast.BinOp, 3, "<module>"),
    ("left = 1\nright = 2\nleft += right\n", ast.AugAssign, 3, "<module>"),
    ("left = 1\nanswer = -left\n", ast.UnaryOp, 2, "<module>"),
    ("left = 1\nright = 2\nif left < right:\n    pass\n", ast.Compare, 3, "<module>"),
    ("def outer(left, right):\n    def inner():\n        return left + right\n", ast.BinOp, 3, "outer.inner"),
])
def test_original_protocol_operation_and_statement_location(source, operation, line, function):
    table = source_table(source, "task.py")
    table.publish_site_records([scalar_record(table, operation=operation)], producer="absint")
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert metrics["established_sites"] == 1 and len(invariants) == 1
    assert (invariants[0].line, invariants[0].function, invariants[0].type) == (line, function, "location_invariant")


def test_nested_operation_does_not_claim_statement_prestate(monkeypatch):
    table = source_table("left = 1\nright = 2\nanswer = (left + right) + 1\n", "task.py")
    table.publish_site_records([scalar_record(table, operation=ast.BinOp, occurrence=1)], producer="absint")
    monkeypatch.setattr(terms_z3, "implies", lambda *a, **kw: pytest.fail("nested operation queried"))
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["reasons"] == {"store_only:fact_scope": 1}


@pytest.mark.parametrize("answer", ["no", "unknown", None, terms.Refused("unsupported")])
def test_only_exact_yes_establishes_a_safety_claim(monkeypatch, answer):
    table, _ = scalar_table()
    monkeypatch.setattr(terms_z3, "implies", lambda facts, goal, **kw:
                        "no" if goal == terms.literal(False) else answer)
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["established_sites"] == 0
    assert [fact.value["status"] for fact in table.safety()] == ["proposed"]
    assert metrics["reasons"]["store_only:domain_not_established"] == 1
    if answer == "unknown":
        assert metrics["reasons"]["store_only:implication_unknown"] >= 1


def test_contradictory_facts_cannot_vacuously_establish_safety():
    table, record = scalar_table()
    record["facts"].append(terms.canon(terms.typeis(terms.var("left"), "bool")))
    table.publish_site_records([record], producer="absint")
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["established_sites"] == 0
    assert metrics["checked_sites"] == metrics["proposed_sites"] == metrics["queries"] == 1
    assert metrics["reasons"] == {"store_only:facts_unsatisfiable": 1}
    assert [fact.value["status"] for fact in table.safety()] == ["proposed"]


@pytest.mark.parametrize("answer,reason", [
    ("unknown", "store_only:implication_unknown"),
    (terms.Refused("unsupported"), "store_only:unsupported"),
    (None, "store_only:implication_unknown"),
])
def test_unresolved_fact_consistency_never_reaches_safety_query(monkeypatch, answer, reason):
    table, _ = scalar_table()

    def implies(facts, goal, **kwargs):
        assert goal == terms.literal(False)
        return answer

    monkeypatch.setattr(terms_z3, "implies", implies)
    for expected_queries in (1, 0):
        invariants, metrics = store_witness.project(table, deadline=float("inf"))
        assert invariants == () and metrics["established_sites"] == 0
        assert metrics["checked_sites"] == metrics["proposed_sites"] == 1
        assert metrics["queries"] == expected_queries
        assert metrics["reasons"] == {reason: 1}
    assert all(fact.value["status"] == "proposed" for fact in table.safety())


def test_admitted_but_unproved_safety_stays_proposed(monkeypatch):
    table, _ = scalar_table()
    answers = iter(("no", "no", "yes"))
    monkeypatch.setattr(terms_z3, "implies", lambda *a, **kw: next(answers))
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["proposed_sites"] == 1
    assert metrics["reasons"] == {"store_only:proposed": 1}


@pytest.mark.parametrize("field,value", [
    ("site", None), ("contract", None), ("binding", None),
    ("site", ["wrong-source", "$.body[2].value", "TypeError"]),
    ("site", ["wrong-source", "missing-path", "TypeError"]),
    ("contract", {"qualname": "protocol.comparison", "exception": "ValueError"}),
    ("status", "may_raise"),
])
def test_incomplete_or_mismatched_records_never_use_true_or_query(monkeypatch, field, value):
    table, record = scalar_table()
    record[field] = value
    table.publish_site_records([record], producer="absint")
    monkeypatch.setattr(terms_z3, "implies", lambda *a, **kw: pytest.fail("invalid record queried"))
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["checked_sites"] == 0 and metrics["reasons"]


@pytest.mark.parametrize("change", ["fake_operand", "wrong_operator", "wrong_arity", "stale_path"])
def test_source_binding_is_verified_before_any_query(monkeypatch, change):
    table, original = scalar_table()
    record = copy.deepcopy(original)
    if change == "fake_operand":
        record["binding"]["receiver"] = terms.canon(terms.literal(1))
    elif change == "wrong_operator":
        record["binding"]["op"] = "Eq"
    elif change == "wrong_arity":
        record["binding"]["args"] = []
    else:
        record["site"][1] = "$.body[0].value"
    table.publish_site_records([record], producer="absint")
    monkeypatch.setattr(terms_z3, "implies", lambda *a, **kw: pytest.fail("forged binding queried"))
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["checked_sites"] == 0 and metrics["reasons"]


def test_duplicate_visits_intersect_must_facts_and_keep_every_parent(monkeypatch):
    source = "left = 1\nright = 2\nwhile left < right:\n    left = left + 1\n"
    table, first = scalar_table(source)
    second = copy.deepcopy(first)
    extra = terms.canon(terms.eq(terms.as_int(terms.var("left")), terms.literal(1)))
    first["facts"].append(extra)
    table.publish_site_records([first, second], producer="absint")
    queries = []

    def implies(facts, goal, **kwargs):
        queries.append(tuple(terms.canon(fact) for fact in facts))
        return "no" if goal == terms.literal(False) else "yes"

    monkeypatch.setattr(terms_z3, "implies", implies)
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert queries and all(query == tuple(second["facts"]) for query in queries)
    assert extra not in queries[0]
    assert metrics["established_sites"] == 1
    assert metrics["safe_sites"] == metrics["sites_established_and_rendered"] == 1
    assert invariants[0].type == "loop_invariant" and invariants[0].line == 3
    assert table.safety()[-1].value["parents"] == tuple(item.identity for item in table.site_record_evidence())


@pytest.mark.parametrize("statement", [
    "left += 1", "result = left + right", "left + right", "if left < right:\n    pass",
], ids=["augassign", "assign", "expr", "if_test"])
@pytest.mark.parametrize("loop,indent,heads,statement_line", [
    ("while left < right:\n{statement}\n    left = 2\n", 4, [(4, 1)], 5),
    ("while left < right:\n    left = 2\nelse:\n{statement}\n", 4, [(4, 1)], 7),
    ("while left < right:\n    while left < right:\n{statement}\n        left = 2\n",
     8, [(4, 1), (5, 5)], 6),
    ("while left < right:\n    while left < right:\n        left = 2\n    else:\n{statement}\n",
     8, [(4, 1), (5, 5)], 8),
], ids=["body", "else", "nested_body", "nested_else"])
def test_only_while_tests_are_loop_invariants_and_each_export_is_checked(
        tmp_path, statement, loop, indent, heads, statement_line):
    from pathlib import Path
    from textwrap import indent as indent_source

    from tharkun.sv.validate import validate
    from tharkun.sv.witness import loads

    source = "left = 1\nright = 2\nresult = 0\n" + loop.format(
        statement=indent_source(statement, " " * indent))
    filename = "task.py"
    specification = "CHECK( init(main()), ! uncaught(TypeError))\n"
    table = source_table(source, filename)
    records = [scalar_record(table, operation=operation, occurrence=index)
               for operation in (ast.Compare, ast.BinOp, ast.AugAssign)
               for index in range(sum(isinstance(node, operation) for node in table.source_context.sites))]
    table.publish_site_records(records, producer="absint")
    detail = store_witness.shadow(table, source, filename, specification,
                                  directory=tmp_path, seconds=float("inf"))
    assert detail["store_sites_established_and_rendered"] == len(heads) + 1
    assert detail["store_reasons"] == {}
    exported = Path(detail["store_witness_path"]).read_text()
    invariants = [Invariant.from_entry(item) for entry in loads(exported) for item in entry["content"]]
    expected = [(filename, line, column, "<module>", "loop_invariant") for line, column in heads]
    expected.append((filename, statement_line, indent + 1, "<module>", "location_invariant"))
    assert sorted((inv.file_name, inv.line, inv.column, inv.function, inv.type)
                  for inv in invariants) == sorted(expected)
    assert all(inv.value != "True" for inv in invariants)
    result = validate(source, filename, specification, exported)
    assert not result.confirmed
    assert result.reason == "Unchecked: unmodeled scalar typing construct While"


def test_frozen_checker_supported_loop_property_keeps_statement_locations():
    from dataclasses import replace

    from tharkun.sv.validate import validate
    from tharkun.sv.witness import dumps

    # Independent checker controls; the real STORE exports above retain TypeError.
    source = "i = 1\nwhile i < 10:\n    7 // i\n    i += 1\nelse:\n    7 // i\n"
    specification = "CHECK( init(main()), ! uncaught(ZeroDivisionError))\n"
    invariants = [
        Invariant("task.py", 2, 1, "<module>", "type(i) is int and 1 <= i <= 10", "loop_invariant"),
        Invariant("task.py", 3, 5, "<module>", "type(i) is int and i >= 1"),
        Invariant("task.py", 4, 5, "<module>", "type(i) is int and i < 10"),
        Invariant("task.py", 6, 5, "<module>", "type(i) is int and i == 10"),
    ]

    def check(facts):
        exported = dumps(obligation_types.witness(source, "task.py", specification, facts))
        return validate(source, "task.py", specification, exported)

    result = check(invariants)
    assert result.confirmed, result
    for index in (1, 2, 3):
        wrong_kind = list(invariants)
        wrong_kind[index] = replace(wrong_kind[index], type="loop_invariant")
        result = check(wrong_kind)
        assert not result.confirmed
        assert result.reason == "Unchecked: loop invariant must locate an original while head"


def test_unknown_tag_on_a_later_visit_cannot_borrow_first_visit_type():
    table, first = scalar_table("left = 1\nright = 2\nwhile left < right:\n    left = left + 1\n")
    second = copy.deepcopy(first)
    second["facts"] = [terms.canon(terms.typeis(terms.var("right"), "int"))]
    table.publish_site_records([first, second], producer="absint")
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["established_sites"] == 0
    assert metrics["reasons"]["store_only:domain_not_established"] == 1


@pytest.mark.parametrize("prefix,builtins", [("type = 7\n", ["int"]), ("int = 7\n", ["type"]),
                                          ("exec('pass')\n", []), ("import builtins\n", [])])
def test_producer_builtin_refusal_prevents_rendering(prefix, builtins):
    table, record = scalar_table(prefix + "left = 1\nright = 2\nanswer = left < right\n")
    record["builtins"] = builtins
    table.publish_site_records([record], producer="absint")
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["reasons"] == {"store_only:builtin_identity_unproved": 1}


@pytest.mark.parametrize("builtins", [None, [], ["type"], ["type", "int"], ["int", "int", "type"],
                                     ["int", "object", "type"], [1], "int,type"])
def test_missing_invalid_or_insufficient_builtin_records_never_render(builtins):
    table, record = scalar_table()
    if builtins is None:
        record.pop("builtins")
    else:
        record["builtins"] = builtins
    table.publish_site_records([record], producer="absint")
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["sites_established_and_rendered"] == 0
    assert metrics["safe_sites"] == metrics["established_sites"] == 1
    assert metrics["reasons"] == {"store_only:builtin_identity_unproved": 1}


def test_builtin_identity_intersects_all_visits():
    table, first = scalar_table()
    second = copy.deepcopy(first)
    second["builtins"].remove("int")
    table.publish_site_records([first, second], producer="absint")
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["safe_sites"] == metrics["established_sites"] == 1
    assert metrics["sites_established_and_rendered"] == 0
    assert metrics["reasons"] == {"store_only:builtin_identity_unproved": 1}


@pytest.mark.parametrize("prefix", ["import math\n", "print('safe')\n"])
def test_record_builtin_proof_survives_imports_and_calls(prefix):
    table, _ = scalar_table(prefix + "left = 1\nright = 2\nanswer = left < right\n")
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert len(invariants) == 1 and metrics["sites_established_and_rendered"] == 1
    assert metrics["reasons"] == {}


def test_only_record_evidence_is_read_for_builtin_identity(monkeypatch):
    table, record = scalar_table()
    original = table.kb.evidence

    def evidence(*args, **kwargs):
        assert kwargs.get("kind") != "module_constant"
        return original(*args, **kwargs)

    monkeypatch.setattr(table.kb, "evidence", evidence)
    assert store_witness.project(table, deadline=float("inf"))[0]
    record.pop("builtins")
    table.publish_site_records([record], producer="absint")
    assert store_witness.project(table, deadline=float("inf"))[0] == ()


def test_exact_implications_are_memoized_with_property_and_resource_policy(monkeypatch):
    table, _ = scalar_table()
    calls = []

    def implies(facts, goal, **kwargs):
        calls.append((tuple(terms.canon(f) for f in facts), terms.canon(goal), kwargs))
        return "no" if goal == terms.literal(False) else "yes"

    monkeypatch.setattr(terms_z3, "implies", implies)
    for _ in range(2):
        store_witness.project(table, specification="property-a", deadline=float("inf"))
    first_count = len(calls)
    assert first_count == 3  # fact consistency, full safety, first implied contract case
    store_witness.project(table, specification="property-b", deadline=float("inf"))
    store_witness.project(table, specification="property-b", resource_limit=1, deadline=float("inf"))
    assert len(calls) == 3 * first_count and calls[-1][2] == {"resource_limit": 1}


@pytest.mark.parametrize("limits", [{"max_sites": 0}, {"deadline": 0}])
def test_shadow_budget_refuses_remaining_sites_without_query(monkeypatch, limits):
    table, _ = scalar_table()
    monkeypatch.setattr(terms_z3, "implies", lambda *a, **kw: pytest.fail("budget exhausted"))
    invariants, metrics = store_witness.project(table, **limits)
    assert invariants == () and metrics["reasons"] == {"store_only:shadow_budget": 1}


@pytest.mark.parametrize("phase", ["map", "groups"])
@pytest.mark.parametrize("expires", [False, True])
def test_deadline_during_source_scans_is_contained_before_any_query(monkeypatch, phase, expires):
    table, _ = scalar_table()
    calls = 0
    expiry_call = 2 if phase == "map" else len(table.source_context.sites) + 1

    def clock():
        nonlocal calls
        calls += 1
        return 2 if expires and calls >= expiry_call else 0

    monkeypatch.setattr(store_witness, "time", SimpleNamespace(monotonic=clock))
    if expires:
        monkeypatch.setattr(terms_z3, "implies", lambda *a, **kw: pytest.fail("incomplete source map queried"))
    invariants, metrics = store_witness.project(table, deadline=1)
    if expires:
        assert invariants == () and metrics["checked_sites"] == 0
        assert metrics["reasons"] == {"store_only:shadow_budget": 1}
    else:
        assert len(invariants) == 1 and metrics["established_sites"] == 1


def test_deadline_also_contains_whole_program_route_projection(monkeypatch):
    table = source_table("pass\n", "task.py")
    store_witness.record_route(table, Verdict("true", component="finite_tree"))
    calls = 0

    def clock():
        nonlocal calls
        calls += 1
        return 2 if calls > len(table.source_context.sites) else 0

    monkeypatch.setattr(store_witness, "time", SimpleNamespace(monotonic=clock))
    invariants, metrics = store_witness.project(table, deadline=1)
    assert invariants == () and metrics["reasons"] == {"store_only:shadow_budget": 1}


def test_true_projection_requires_actual_full_route_for_this_source_and_classes():
    table = source_table("pass\n", "task.py", ("AttributeError", "TypeError"))
    context = table.source_context
    table.publish_route("absint", source_sha256=context.source_sha256, classes=("TypeError",),
                        producer="absint", whole_program=True, location=context.location)
    assert store_witness.project(table)[0] == ()
    store_witness.record_route(table, Verdict("true", component="finite_tree"))
    invariants, _ = store_witness.project(table)
    assert invariants == (Invariant("task.py", 1, 1, "<module>", "True"),)


@pytest.mark.parametrize("verdict", ["TRUE", "UNKNOWN"])
def test_absint_stage_never_publishes_a_final_route(tmp_path, monkeypatch, verdict):
    from tharkun.solve import absint
    from tharkun.sv import engine, prove

    source = "left = 1\nright = 2\nanswer = left < right\n"
    filename = str(tmp_path / "task.py")
    (tmp_path / "task.py").write_text(source)
    table = source_table(source, filename)
    record = scalar_record(table)
    result = SimpleNamespace(verdict=verdict, component="proof:shape", records=[record], sites=[],
                             lemmas=[], invariants=[], reason="test result")
    monkeypatch.setattr(engine, "proof_blocker", lambda *a, **kw: None)
    monkeypatch.setattr(prove, "prove_static", lambda *a, **kw: None)
    monkeypatch.setattr(absint, "prove", lambda *a, **kw: result)
    actual = engine._static_proof(filename, ("TypeError",), lambda: 30, lambda msg: None, table=table)
    assert table.proof_routes() == () and table.site_records() == [record]
    if verdict == "TRUE":
        assert actual.verdict == "true" and actual.detail["candidate_invariants"] == []
    else:
        assert actual is None


def test_final_absint_route_binds_the_returned_verdict_and_latest_parents(tmp_path, monkeypatch):
    from tharkun.solve import absint
    from tharkun.sv import engine, prove, scheduler

    program = tmp_path / "task.py"
    program.write_text("left = 1\nright = 2\nanswer = left < right\n")
    tables, returned, published = [], [], []
    static, record_route = engine._static_proof, store_witness.record_route

    def table(*args, **kwargs):
        value = ObligationTable(*args, **kwargs)
        tables.append(value)
        return value

    def absint_result(*args, **kwargs):
        tables[-1].publish_site_records([{"status": "may_raise"}], producer="absint")
        return SimpleNamespace(verdict="TRUE", component="proof:shape",
                               records=[scalar_record(tables[-1])], sites=[], lemmas=[], invariants=[])

    def static_result(*args, **kwargs):
        value = static(*args, **kwargs)
        assert tables[-1].proof_routes() == ()
        returned.append(value)
        return value

    def publish(table, verdict):
        assert verdict is returned[-1]
        published.append(verdict)
        record_route(table, verdict)

    monkeypatch.setattr(scheduler, "ObligationTable", table)
    monkeypatch.setattr(prove, "prove_static", lambda *a, **kw: None)
    monkeypatch.setattr(absint, "prove", absint_result)
    monkeypatch.setattr(engine, "_static_proof", static_result)
    monkeypatch.setattr(store_witness, "record_route", publish)
    monkeypatch.setattr(store_witness, "shadow", lambda *a, **kw: {})
    monkeypatch.setattr(proof, "check_candidates", lambda *a, **kw: Validation(True, ()))
    actual = engine.verify(str(program), ("TypeError",), budget_seconds=60, canary_seconds=0,
                           concolic=False, stages=[])
    assert actual is returned[0] and published == [actual] and actual.verdict == "true"
    (route,) = tables[-1].proof_routes()
    assert route.value["route"] == "absint" and route.value["whole_program"] is True
    assert route.value["sites"] == () and route.value["site_record_parent_count"] == 1
    assert route.derivations[0].producer == actual.component
    assert route.derivations[0].parents == tuple(e.identity for e in tables[-1].site_record_evidence())


def test_confirmed_canary_false_leaves_no_provisional_absint_route(tmp_path, monkeypatch):
    from tharkun.solve import absint
    from tharkun.sv import engine, prove, scheduler

    program = tmp_path / "canary.py"
    program.write_text("left = 1\nright = 'bad'\nanswer = left < right\n")
    tables, provisional, outcomes = [], [], []
    static, run_once = engine._static_proof, engine.run_once

    def table(*args, **kwargs):
        value = ObligationTable(*args, **kwargs)
        tables.append(value)
        return value

    def absint_result(*args, **kwargs):
        return SimpleNamespace(verdict="TRUE", component="proof:shape",
                               records=[scalar_record(tables[-1])], sites=[], lemmas=[], invariants=[])

    def static_result(*args, **kwargs):
        value = static(*args, **kwargs)
        assert value.verdict == "true" and tables[-1].proof_routes() == ()
        provisional.append(value)
        return value

    def child(*args, **kwargs):
        assert provisional and tables[-1].proof_routes() == ()
        outcome = run_once(*args, **kwargs)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(scheduler, "ObligationTable", table)
    monkeypatch.setattr(prove, "prove_static", lambda *a, **kw: None)
    monkeypatch.setattr(absint, "prove", absint_result)
    monkeypatch.setattr(engine, "_static_proof", static_result)
    monkeypatch.setattr(engine, "run_once", child)
    monkeypatch.setattr(store_witness, "record_route", lambda *a, **kw: pytest.fail("FALSE published a route"))
    actual = engine.verify(str(program), ("TypeError",), budget_seconds=60, run_timeout=30,
                           jobs=1, canary_seconds=5, concolic=False, stages=[])
    assert actual.verdict == "false" and actual.exception == "TypeError"
    assert actual is not provisional[0] and actual.detail["unsound_proof"] == "proof:shape"
    assert actual.detail["canary"]["completed"] and actual.detail["canary"]["reason"] == "violation"
    assert outcomes and outcomes[0].violated(("TypeError",)) == "TypeError"
    assert tables[-1].site_record_evidence() and tables[-1].proof_routes() == ()


@pytest.mark.parametrize("verdict", [
    "finite_tree", Verdict("false", component="proof:shape"), Verdict("unknown", component="finite_tree"),
    Verdict("true", component="absint"), Verdict("true", component="proof:no-site"),
    Verdict("true", component="symbolic:product"), Verdict("true", component="deterministic"),
    Verdict("true", component="proof:facts:fake"), Verdict("true", component="proof:stationary"),
])
def test_route_adapter_refuses_strings_non_true_and_unimplemented_routes(verdict):
    table = source_table("pass\n", "task.py")
    store_witness.record_route(table, verdict)
    assert table.proof_routes() == ()


def test_route_publication_failure_is_contained(monkeypatch):
    table = source_table("pass\n", "task.py")

    def fail(*args, **kwargs):
        raise ValueError("bad route publication")

    monkeypatch.setattr(table, "publish_route", fail)
    assert store_witness.record_route(table, Verdict("true", component="finite_tree")) is None
    assert table.proof_routes() == ()


def test_source_context_is_immutable_and_record_reader_keeps_legacy_shape():
    table = source_table("pass\n", "task.py")
    assert table.site_records() is None and table.site_record_evidence() == ()
    context = table.source_context
    with pytest.raises(TypeError):
        context.sites[ast.Pass()] = next(iter(context.sites.values()))
    with pytest.raises(ValueError, match="already bound"):
        table.bind_source(context.sites, source_sha256=context.source_sha256,
                          classes=context.classes, location=context.location)
    table.publish_site_records([{"status": "safe"}], producer="absint")
    table.publish_site_records([{"status": "may_raise"}], producer="absint")
    assert table.site_records() == [{"status": "may_raise"}]
    assert [item.value for item in table.site_record_evidence()] == table.site_records()


def test_shadow_rejects_source_context_from_another_program(monkeypatch):
    table, _ = scalar_table()
    monkeypatch.setattr(store_witness, "project", lambda *a, **kw: pytest.fail("stale source projected"))
    assert store_witness.shadow(table, "pass\n", "task.py", "property") == {
        "store_reasons": {"store_only:source_identity": 1}}


def test_shadow_keeps_logical_source_identity_separate_from_raw_input_hash(tmp_path, monkeypatch):
    from tharkun.sv.witness import loads

    source = "# coding: latin-1\nlabel = 'caf\xe9'\nleft = 1\nright = 2\nanswer = left < right\n"
    filename = str(tmp_path / "task.py")
    raw = source.encode("latin-1")
    table = source_table(source, filename)
    inv = Invariant(filename, 5, 1, "<module>", "left < right")
    monkeypatch.setattr(store_witness, "project", lambda *a, **kw: ((inv,), {}))
    result = store_witness.shadow(table, source, filename, "CHECK( init(main()), ! uncaught(TypeError))\n",
                                  directory=tmp_path, source_bytes=raw, seconds=1)
    document = loads((tmp_path / "store-witness.yml").read_text())
    assert result["store_witness_path"] == str(tmp_path / "store-witness.yml")
    assert table.source_context.source_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert document[0]["metadata"]["task"]["input_file_hashes"][filename] == hashlib.sha256(raw).hexdigest()
    assert hashlib.sha256(raw).hexdigest() != table.source_context.source_sha256


def test_explicit_direct_fact_scope_preserves_legacy_projection():
    legacy, _ = scalar_table()
    expected = store_witness.project(legacy, deadline=float("inf"))
    explicit, record = scalar_table()
    record["fact_scope"] = "site"
    explicit.publish_site_records([record], producer="absint")
    assert store_witness.project(explicit, deadline=float("inf")) == expected


@pytest.mark.parametrize("legacy_first", [False, True])
def test_direct_legacy_and_explicit_visits_have_the_same_effective_scope(legacy_first):
    table, legacy = scalar_table()
    explicit = copy.deepcopy(legacy)
    explicit["fact_scope"] = "site"
    records = [legacy, explicit] if legacy_first else [explicit, legacy]
    table.publish_site_records(records, producer="absint")
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert len(invariants) == 1 and metrics["reasons"] == {}
    assert table.safety()[-1].value["parents"] == tuple(
        item.identity for item in table.site_record_evidence())


@pytest.mark.parametrize("scope", [None, "", "unknown", False, 1, [], {}])
def test_unknown_fact_scope_refuses_before_facts_or_queries(monkeypatch, scope):
    table, record = scalar_table()
    record["fact_scope"] = scope
    # Intersecting these malformed facts would raise before reaching a query.
    record["facts"] = [{}]
    table.publish_site_records([record], producer="absint")
    monkeypatch.setattr(terms_z3, "implies", lambda *a, **kw: pytest.fail("invalid scope queried"))
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["reasons"] == {"store_only:fact_scope": 1}
    assert metrics["queries"] == metrics["proposed_sites"] == metrics["checked_sites"] == 0
    assert table.safety() == ()


@pytest.mark.parametrize("source,scope", [
    ("answer = left < right\n", "statement_entry"),
    ("answer = True and (left < right)\n", "site"),
    ("answer = True and (left < right)\n", None),
])
def test_fact_scope_must_match_the_shared_source_position(monkeypatch, source, scope):
    table = source_table("left = 1\nright = 2\n" + source, "task.py")
    record = scalar_record(table)
    if scope is not None:  # absent legacy scope is refused at a nested atom
        record["fact_scope"] = scope
    table.publish_site_records([record], producer="absint")
    monkeypatch.setattr(terms_z3, "implies", lambda *a, **kw: pytest.fail("mismatched scope queried"))
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["reasons"] == {"store_only:fact_scope": 1}
    assert table.safety() == ()


@pytest.mark.parametrize("first_scope,second_scope", [
    ("site", "statement_entry"), ("statement_entry", "site"),
])
def test_mixed_fact_scopes_never_intersect_visits(monkeypatch, first_scope, second_scope):
    table = source_table("left = 1\nright = 2\nanswer = True and (left < right)\n", "task.py")
    first = scalar_record(table)
    first["fact_scope"] = first_scope
    second = copy.deepcopy(first)
    second["fact_scope"] = second_scope
    second["facts"] = [{}]
    table.publish_site_records([first, second], producer="absint")
    monkeypatch.setattr(terms_z3, "implies", lambda *a, **kw: pytest.fail("mixed scopes queried"))
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["reasons"] == {"store_only:fact_scope": 1}
    assert table.safety() == ()


@pytest.mark.parametrize("suffix,occurrence,line,column,kind", [
    ("answer = (left < right) and (left < right)\n", 1, 3, 1, "location_invariant"),
    ("if not ((left < right) or (left < right)):\n    pass\n", 1, 3, 1, "location_invariant"),
    ("while (left < right) and (left < right):\n    left = right\n", 1, 3, 1, "loop_invariant"),
    ("while left < right:\n    if (left < right) and (left < right):\n        left = right\n",
     2, 4, 5, "location_invariant"),
])
def test_nested_statement_entry_uses_original_statement_and_loop_head(
        suffix, occurrence, line, column, kind):
    table = source_table("left = 1\nright = 2\n" + suffix, "task.py")
    record = scalar_record(table, occurrence=occurrence)
    record["fact_scope"] = "statement_entry"
    table.publish_site_records([record], producer="absint")
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert metrics["sites_established_and_rendered"] == 1 and metrics["reasons"] == {}
    (invariant,) = invariants
    assert (invariant.file_name, invariant.line, invariant.column, invariant.function, invariant.type) == (
        "task.py", line, column, "<module>", kind)
    assert invariant.value != "True"


@pytest.mark.parametrize("prefix,occurrence", [
    ("mutate()", 0), ("flag", 0), ("-bound < left", 1),
])
def test_claimed_entry_scope_cannot_bypass_shared_prefix_exclusions(monkeypatch, prefix, occurrence):
    source = "left = 1\nright = 2\nanswer = (" + prefix + ") and (left < right)\n"
    table = source_table(source, "task.py")
    record = scalar_record(table, occurrence=occurrence)
    record["fact_scope"] = "statement_entry"
    table.publish_site_records([record], producer="absint")
    monkeypatch.setattr(terms_z3, "implies", lambda *a, **kw: pytest.fail("unsafe source prefix queried"))
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["reasons"] == {"store_only:fact_scope": 1}
    assert table.safety() == ()


def test_nested_scope_intersects_visits_without_borrowing_first_entry_bounds(monkeypatch):
    table = source_table("left = 1\nright = 2\nanswer = True and (left < right)\n", "task.py")
    first = scalar_record(table)
    first["fact_scope"] = "statement_entry"
    second = copy.deepcopy(first)
    first["facts"].append(terms.canon(terms.eq(terms.as_int(terms.var("left")), terms.literal(1))))
    table.publish_site_records([first, second], producer="absint")
    original, seen = terms_z3.implies, []

    def query(facts, goal, **kwargs):
        seen.append(tuple(terms.canon(fact) for fact in facts))
        return original(facts, goal, **kwargs)

    monkeypatch.setattr(terms_z3, "implies", query)
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants and metrics["sites_established_and_rendered"] == 1
    assert seen and all(facts == tuple(second["facts"]) for facts in seen)
    assert table.safety()[-1].value["parents"] == tuple(
        item.identity for item in table.site_record_evidence())


def test_real_statement_entry_producer_records_project_without_scope_reinterpretation():
    from tharkun.solve.absint import prove

    source = "left = 1\nright = 2\nanswer = (left < right) and (left + right)\n"
    result = prove(source, ("TypeError",), filename="task.py")
    assert result.complete, result.explain()
    records = [record for record in result.records if record["exc"] == "TypeError"]
    assert sum(record.get("fact_scope") == "statement_entry" for record in records) == 2
    table = source_table(source, "task.py")
    table.publish_site_records(records, producer="absint")
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants and metrics["sites_established_and_rendered"] == 2
    assert metrics["reasons"] == {}
    assert all(inv.line == 3 and inv.type == "location_invariant" for inv in invariants)


def test_real_final_skipped_entry_cannot_reuse_the_divide_site_int_facts():
    from tharkun.solve.absint import prove

    source = "flag = True\ndenominator = 1\nwhile (flag == True) and 10 // denominator:\n    flag = False\n    denominator = None\n"
    result = prove(source, ("ZeroDivisionError",), filename="task.py")
    assert result.complete, result.explain()
    records = [record for record in result.records if record["exc"] == "ZeroDivisionError"]
    assert len(records) == 1 and records[0]["binding"] is not None
    assert records[0]["fact_scope"] == "statement_entry"
    assert records[0]["facts"] == []
    table = source_table(source, "task.py", ("ZeroDivisionError",))
    table.publish_site_records(records, producer="absint")
    invariants, metrics = store_witness.project(table, deadline=float("inf"))
    assert invariants == () and metrics["checked_sites"] == 1
    assert metrics["reasons"] == {"store_only:domain_not_established": 1}


def test_no_store_witness_skips_projection_and_keeps_legacy_bytes(tmp_path, monkeypatch, frozen_metadata):
    """``shadow_seconds=None`` (the CLI's ``--no-store-witness``): no projection, no store-witness.yml, no store_*
    detail, and the frozen witness is byte-identical to a run with the shadow on."""
    from pathlib import Path

    def project(*args, **kwargs):
        raise AssertionError("the store projection must not run")

    source = "left = 6\nright = 2\nanswer = left / right\n"
    filename = str(tmp_path / "task.py")
    (tmp_path / "task.py").write_text(source)
    table = source_table(source, filename, ("ZeroDivisionError",))
    monkeypatch.setattr(proof, "check_candidates", lambda *a, **kw: Validation(True, (), "frozen check"))
    baseline = finish_task(tmp_path, monkeypatch, project=lambda *a, **kw: ((), {}), label="baseline")
    monkeypatch.setattr(store_witness, "project", project)
    result = proof.finish(Verdict("true", component="proof:interval"), source, filename,
                          "CHECK( init(main()), ! uncaught(ZeroDivisionError))\n", table.census,
                          directory=tmp_path / "off", knowledge=table.kb, table=table, shadow_seconds=None)
    assert result.verdict == baseline.verdict == "true" and result.detail["proof_checked"]
    assert Path(result.detail["witness_path"]).read_bytes() == Path(baseline.detail["witness_path"]).read_bytes()
    assert not [key for key in result.detail if key.startswith("store_")]
    assert not list((tmp_path / "off").rglob("store-witness.yml"))


def test_cli_no_store_witness_writes_no_store_file_and_keeps_the_verdict(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    program = tmp_path / "task.py"
    program.write_text("left = 1\nright = 2\nanswer = left < right\n")
    prp = tmp_path / "dynamic-typing.prp"
    prp.write_text("CHECK( init(main()), ! uncaught(TypeError))\n")
    src = str(Path(__file__).resolve().parents[1] / "src")

    def run(label, *extra):
        out = subprocess.run([sys.executable, "-P", "-m", "tharkun.sv", str(program), "--property", str(prp),
                              "--budget-seconds", "120", "--jobs", "1", "--proof-dir", str(tmp_path / label), *extra],
                             capture_output=True, text=True, timeout=300, cwd=tmp_path,
                             env={"PATH": "/usr/bin:/bin", "PYTHONPATH": src, "PYTHONDONTWRITEBYTECODE": "1"})
        verdict = [line for line in out.stdout.splitlines() if line.split()[:1] in (["TRUE"], ["FALSE"], ["UNKNOWN"])]
        return out, verdict[-1:]

    on, verdict_on = run("on")
    off, verdict_off = run("off", "--no-store-witness")
    assert off.returncode == on.returncode == 0, (on.stderr[-2000:], off.stderr[-2000:])
    assert verdict_off == verdict_on == ["TRUE"]
    assert list((tmp_path / "off").rglob("witness.yml"))
    assert not list((tmp_path / "off").rglob("store-witness.yml"))
