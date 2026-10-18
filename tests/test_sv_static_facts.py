"""The component "static facts (module namespace)" (#1132b; ``sv/scheduler.py::publish_module_namespace``,
``sv/obligation_table.py::ObligationTable.publish_module_namespace``, docs/SV.md "Component contracts").

What is pinned:

* the two kinds, ``module_constant`` and ``regex_pattern``, reach the run's store with the agreed record
  (``solve/module_namespace.py::module_constants`` / ``regex_globals``), keyed by where the name or the regex is;
* ``proven`` is the whole story for a consumer: a name something in the import graph rebinds is published
  with ``proven`` false and the evidence;
* the kinds are hints: uncertified Evidence, and an analysis that crashes publishes nothing and never stops
  the run;
* ``scheduler.run`` calls the component once, before the first stage, with the task program.
"""
from __future__ import annotations

import textwrap

from tharkun.sv import scheduler, verify
from tharkun.sv.contracts import COMPONENTS, KINDS
from tharkun.sv.obligation_table import (CALL_TARGET, MODULE_CONSTANT, REGEX_PATTERN, ObligationTable,
                                         call_targets_in)
from tharkun.solve.obligations import ObligationCensus

TASK = """
    import re
    import _sv_verifier

    LIMIT = 50
    MODE = "fast"
    PATTERN = re.compile(r"^a+$", re.I)

    def bump():
        global MODE
        MODE = "slow"

    if __name__ == "__main__":
        x = _sv_verifier.nondet_int()
        if x > LIMIT and PATTERN.match("aA"):
            bump()
"""


def write(tmp_path, body=TASK, name="task.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body).lstrip("\n"))
    return str(path)


def empty_table():
    return ObligationTable(ObligationCensus(()))


def by_key(table, kind):
    return {e.key: e for e in table.kb.evidence(kind=kind)}


def test_the_kinds_and_the_component_are_registered_together():
    assert KINDS[MODULE_CONSTANT].store == "evidence" and not KINDS[MODULE_CONSTANT].certifiable
    assert KINDS[REGEX_PATTERN].store == "evidence" and not KINDS[REGEX_PATTERN].certifiable
    component = next(c for c in COMPONENTS if c.id == "static facts (module namespace)")
    assert "`module_constant`" in component.writes and "`regex_pattern`" in component.writes
    assert component.decides.startswith("nothing")


def test_module_constants_are_published_with_their_proof(tmp_path):
    table = empty_table()
    scheduler.publish_module_namespace(table, write(tmp_path))
    constants = by_key(table, MODULE_CONSTANT)
    assert constants["task.py:LIMIT"].value["proven"] is True
    assert constants["task.py:LIMIT"].value["value"] == 50
    assert constants["task.py:LIMIT"].value["rebinds"] == []
    # ``bump`` rebinds MODE through a ``global`` store: published, not proven, with the evidence
    mode = constants["task.py:MODE"].value
    assert mode["proven"] is False
    assert [r["kind"] for r in mode["rebinds"]] and all(r["module"] == "task.py" for r in mode["rebinds"])
    assert {d.producer for d in constants["task.py:LIMIT"].derivations} == {"module_namespace"}


def test_a_compiled_regex_is_published_with_pattern_and_flags(tmp_path):
    table = empty_table()
    scheduler.publish_module_namespace(table, write(tmp_path))
    (record,) = [e.value for e in table.kb.evidence(kind=REGEX_PATTERN)]
    assert record["kind"] == "regex_global"
    assert record["pattern"] == "^a+$" and record["flags"] == 2          # re.I is 2
    assert record["compiled_flags"] & 2
    assert record["bindings"] == [{"module": "task.py", "name": "PATTERN"}]
    assert record["provenance"] == {"producer": "tharkun", "method": "static-fold", "complete": True}
    key = f"task.py:{record['line']}:{record['col']}"
    assert key in by_key(table, REGEX_PATTERN)


def test_a_local_import_is_part_of_the_graph(tmp_path):
    (tmp_path / "helper.py").write_text("LIMIT = 5\nimport re\nWORD = re.compile(r'\\w+')\n")
    program = write(tmp_path, "from helper import *\nTHRESHOLD = LIMIT + 1\n")
    table = empty_table()
    scheduler.publish_module_namespace(table, program)
    constants = by_key(table, MODULE_CONSTANT)
    assert constants["task.py:LIMIT"].value["proven"] and constants["task.py:LIMIT"].value["value"] == 5
    assert constants["task.py:THRESHOLD"].value["value"] == 6
    assert [e.value["pattern"] for e in table.kb.evidence(kind=REGEX_PATTERN)] == ["\\w+"]


def test_a_graph_over_the_cap_is_published_unproven(tmp_path):
    """The component is not allowed to cost a package's worth of analysis (a 152-module graph measured 5-19 s):
    past ``STATIC_FACTS_MAX_MODULES`` modules the graph is incomplete, so nothing in it is proven."""
    count = scheduler.STATIC_FACTS_MAX_MODULES + 4
    for i in range(count):
        tail = f"import m{i + 1}\n" if i < count - 1 else ""
        (tmp_path / f"m{i}.py").write_text(f"{tail}X{i} = {i}\n")
    program = write(tmp_path, "import m0\nLIMIT = 3\n")
    table = empty_table()
    scheduler.publish_module_namespace(table, program)
    limit = by_key(table, MODULE_CONSTANT)["task.py:LIMIT"].value
    assert limit["proven"] is False and limit["provenance"]["complete"] is False
    assert "module cap reached" in limit["provenance"]["incomplete"]


def test_the_records_are_uncertified_evidence_and_certify_nothing(tmp_path):
    table = empty_table()
    scheduler.publish_module_namespace(table, write(tmp_path))
    assert list(table.kb) == []                      # no Fact at all, so no ``certified`` row
    assert table.kb.evidence(kind=MODULE_CONSTANT) and table.kb.evidence(kind=REGEX_PATTERN)


def test_an_unreadable_task_publishes_nothing(tmp_path):
    table = empty_table()
    scheduler.publish_module_namespace(table, str(tmp_path / "missing.py"))
    assert table.kb.evidence(kind=MODULE_CONSTANT) == () and table.kb.evidence(kind=REGEX_PATTERN) == ()


def test_an_analysis_that_crashes_publishes_nothing_and_the_run_goes_on(tmp_path, monkeypatch):
    import tharkun.solve.module_namespace as namespace

    def boom(*args, **kwargs):
        raise RecursionError("depth")

    monkeypatch.setattr(namespace, "module_constants", boom)
    said = []
    table = empty_table()
    scheduler.publish_module_namespace(table, write(tmp_path), said.append)
    assert table.kb.evidence() == ()
    assert said == ["static facts: not published (RecursionError)"]


def test_the_scheduler_runs_the_component_once_with_the_task_program(tmp_path, monkeypatch):
    seen = []
    real = scheduler.publish_module_namespace

    def spy(table, program, say=lambda msg: None):
        seen.append((table, program))
        real(table, program, say)

    monkeypatch.setattr(scheduler, "publish_module_namespace", spy)
    program = write(tmp_path)
    verdict = verify(program, ("ZeroDivisionError",), budget_seconds=60, concolic=False, prove=False, jobs=1,
                     max_runs=3)
    assert verdict.verdict in ("unknown", "true", "false")
    assert len(seen) == 1 and seen[0][1] == program
    table = seen[0][0]
    assert by_key(table, MODULE_CONSTANT)["task.py:LIMIT"].value["proven"] is True
    assert len(table.kb.evidence(kind=REGEX_PATTERN)) == 1


# --- the ``call_target`` kind ([internal ref]; solve/call_targets.py::call_targets) -------------------------------

CLASS_TASK = """
    import _sv_verifier

    class Canvas:
        def __init__(self, w):
            self.w = w

    def render(w):
        return Canvas(w).w

    if __name__ == "__main__":
        render(_sv_verifier.nondet_int())
"""


def test_the_call_target_kind_is_registered_evidence_and_written_by_the_static_facts_component():
    assert KINDS[CALL_TARGET].store == "evidence" and not KINDS[CALL_TARGET].certifiable
    assert KINDS[CALL_TARGET].writers == ("static facts",)
    component = next(c for c in COMPONENTS if c.id == "static facts (module namespace)")
    assert "`call_target`" in component.writes and "solve/call_targets.py::call_targets" in component.where


def test_call_targets_are_published_per_call_site_with_their_proof(tmp_path):
    table = empty_table()
    scheduler.publish_module_namespace(table, write(tmp_path, CLASS_TASK))
    records = table.call_targets()
    by_callee = {(r["scope"], r["callee"]): r for r in records}
    ctor = by_callee[("render", "Canvas")]
    assert (ctor["target"], ctor["name"], ctor["proven"]) == ("user_class", "Canvas", True)
    assert by_callee[("<main>", "render")]["target"] == "user_function"
    assert by_callee[("<main>", "_sv_verifier.nondet_int")]["target"] == "imported"
    keys = by_key(table, CALL_TARGET)
    assert len(keys) == len(records)
    assert all(k.startswith("task.py:") for k in keys)
    assert {d.producer for d in next(iter(keys.values())).derivations} == {"call_targets"}


def test_call_targets_are_uncertified_evidence_and_the_reader_accepts_the_store_itself(tmp_path):
    table = empty_table()
    scheduler.publish_module_namespace(table, write(tmp_path, CLASS_TASK))
    assert list(table.kb) == []
    assert call_targets_in(table.kb) == table.call_targets() and table.call_targets()
    assert call_targets_in(None) == []


def test_an_unreadable_task_publishes_no_call_targets(tmp_path):
    table = empty_table()
    scheduler.publish_module_namespace(table, str(tmp_path / "missing.py"))
    assert table.call_targets() == []


def test_a_crashing_call_target_producer_keeps_the_namespace_records(tmp_path, monkeypatch):
    import tharkun.solve.call_targets as producer

    def boom(*args, **kwargs):
        raise RecursionError("depth")

    monkeypatch.setattr(producer, "call_targets", boom)
    said = []
    table = empty_table()
    scheduler.publish_module_namespace(table, write(tmp_path), said.append)
    assert table.call_targets() == []
    assert table.kb.evidence(kind=MODULE_CONSTANT) and table.kb.evidence(kind=REGEX_PATTERN)
    assert any("call targets not published" in m for m in said)


def test_the_proven_class_names_are_read_back_for_the_symbolic_stage(tmp_path):
    table = empty_table()
    scheduler.publish_module_namespace(table, write(tmp_path, textwrap.dedent(CLASS_TASK) + "\nclass Spare:\n    pass\n\nSpare = 3\n"))
    assert table.proven_classes() == ["Canvas"]          # Spare is rebound: its constant is not proven
    assert KINDS["module_constant"].readers == ("symbolic",)
