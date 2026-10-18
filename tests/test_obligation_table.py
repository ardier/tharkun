"""The live obligation table ([internal ref] step 3a, ``sv/obligation_table.py``): rows and their decisions,
and what components write into it and read back.

The row/decision tests are the former ``sv/scheduler.py::_write_discharged`` / ``_write_violated``
tests, unchanged in what they pin: ``certified=True`` only for a whole-program discharge and an
unambiguous confirmed violation.
"""
from __future__ import annotations

import ast
from dataclasses import replace

import pytest

from tharkun.knowledge import KnowledgeBase
from tharkun.solve.obligations import Invariant, Obligation, ObligationCensus
from tharkun.solve.sitekey import site_keys
from tharkun.sv.concolic import Solved
from tharkun.sv.obligation_table import (
    CLASS_HINTS,
    DISCHARGED,
    OPEN,
    SITE_RECORD,
    VIOLATED,
    ObligationTable,
)


def census_of(*rows):
    return ObligationCensus(tuple(rows))


def ob(path, exc, line=3, column=1, gap=""):
    return Obligation("sha", path, exc, Invariant("p.py", line, column, "<module>", "False"), gap)


# ------------------------------------------------------------------ the rows

def test_one_open_uncertified_row_per_site_and_no_coverage_rows():
    table = ObligationTable(census_of(ob("Module.body[0]", "KeyError"),
                                      ob("$coverage", "TypeError", gap="no site table for TypeError")))
    assert isinstance(table.kb, KnowledgeBase)
    facts = list(table.kb)
    assert len(facts) == 1 and facts[0].subject == "Module.body[0]:KeyError"
    assert facts[0].kind == "obligation" and facts[0].value == OPEN and not facts[0].certified
    assert [o.ast_path for o in table.rows()] == ["Module.body[0]"]


def test_rows_and_open_rows_filter_by_class():
    table = ObligationTable(census_of(ob("a", "KeyError"), ob("b", "IndexError")))
    assert [o.ast_path for o in table.rows(("KeyError",))] == ["a"]
    assert [o.ast_path for o in table.open_rows(("KeyError", "IndexError"))] == ["a", "b"]


def test_discharge_marks_every_row_of_the_proven_classes_certified_and_only_those():
    census = census_of(ob("a", "KeyError"), ob("b", "IndexError"), ob("$coverage", "KeyError", gap="gap"))
    table = ObligationTable(census)
    table.discharge(("KeyError",), "proof:no-site")

    discharged = [f for f in table.kb if f.value == DISCHARGED]
    assert len(discharged) == 1
    assert discharged[0].subject == "a:KeyError" and discharged[0].certified is True
    assert discharged[0].source == "proof:no-site"
    assert [table.status(o) for o in table.rows()] == [DISCHARGED, OPEN]       # IndexError untouched
    assert [o.ast_path for o in table.open_rows()] == ["b"]


def test_violate_matches_by_line_and_exception_only():
    table = ObligationTable(census_of(ob("a", "KeyError", line=5),
                                      ob("b", "IndexError", line=5),            # same line, other exception
                                      ob("c", "KeyError", line=9)))              # other line
    table.violate("KeyError", 5, "enum:small_first")
    violated = [f for f in table.kb if f.value == VIOLATED]
    assert len(violated) == 1 and violated[0].subject == "a:KeyError" and violated[0].certified is True


def test_two_sites_one_line_never_certifies_the_uninvolved_sibling():
    """MG review, 2026-09-29 (P0): ``a / b + c / d`` -- two ZeroDivisionError obligations on one line,
    only one of which divided by zero.  A line-only match must stay uncertified for every candidate."""
    table = ObligationTable(census_of(ob("a", "ZeroDivisionError", line=5, column=1),
                                      ob("b", "ZeroDivisionError", line=5, column=9)))
    table.violate("ZeroDivisionError", 5, "enum:small_first")
    violated = [f for f in table.kb if f.value == VIOLATED]
    assert {f.subject for f in violated} == {"a:ZeroDivisionError", "b:ZeroDivisionError"}
    assert all(f.certified is False for f in violated)
    # an uncertified violation is a hint, not a decision: both rows stay OPEN
    assert [table.status(o) for o in table.rows()] == [OPEN, OPEN]


def test_a_witness_column_singles_out_exactly_one_row_and_certifies_it():
    table = ObligationTable(census_of(ob("a", "ZeroDivisionError", line=5, column=1),
                                      ob("b", "ZeroDivisionError", line=5, column=9)))
    table.violate("ZeroDivisionError", 5, "enum:small_first", column=9)
    violated = [f for f in table.kb if f.value == VIOLATED]
    assert len(violated) == 1 and violated[0].subject == "b:ZeroDivisionError" and violated[0].certified is True
    assert [table.status(o) for o in table.rows()] == [OPEN, VIOLATED]


def test_violate_does_nothing_without_a_line():
    table = ObligationTable(census_of(ob("a", "KeyError", line=5)))
    table.violate("KeyError", None, "x")
    assert not any(f.value == VIOLATED for f in table.kb)


def test_a_violation_beats_a_discharge():
    """A replay is real whatever a proof said (``found()`` names the unsound proof)."""
    table = ObligationTable(census_of(ob("a", "KeyError", line=5)))
    table.discharge(("KeyError",), "proof:interval")
    table.violate("KeyError", 5, "enum:small_first")
    assert table.status(table.rows()[0]) == VIOLATED


# ------------------------------------------------------------------ absint's site records

def test_site_records_come_back_as_the_latest_generation():
    table = ObligationTable(census_of())
    assert table.site_records() is None                               # nothing known: what concolic expects
    table.publish_site_records([], producer="absint")                  # an empty run is not a generation
    assert table.site_records() is None

    first = [{"line": 3, "exc": "KeyError", "status": "may_raise", "residual": "x"},
             {"line": 4, "exc": "KeyError", "status": "safe"}]
    table.publish_site_records(first, producer="absint")
    assert table.site_records() == first

    second = [{"line": 3, "exc": "KeyError", "status": "safe"}]       # the class_hints re-run saw more
    table.publish_site_records(second, producer="absint")
    assert table.site_records() == second                             # supersedes, as `hinted_records or records` did
    assert len(table.kb.evidence(kind=SITE_RECORD)) == 3              # ... and the first stays on record


def test_site_records_are_copies_so_a_reader_cannot_corrupt_the_store():
    table = ObligationTable(census_of())
    record = {"line": 3, "exc": "KeyError"}
    table.publish_site_records([record], producer="absint")
    record["line"] = 99
    assert table.site_records() == [{"line": 3, "exc": "KeyError"}]


# ------------------------------------------------------------------ a concolic stage's outcome

def test_a_solved_stage_is_read_back_with_its_open_lines_and_its_closed_rule():
    table = ObligationTable(census_of())
    assert table.solved("symbolic") is None                           # the stage never ran

    table.publish_solved("symbolic", Solved(ran=True, sites={10: "reached", 12: "open", 14: "unsat"}),
                         producer="concolic:symbolic")
    view = table.solved("symbolic")
    assert view.ran and view.sites == {10: "reached", 12: "open", 14: "unsat"}
    assert view.open_lines() == [12] and not view.closed

    table.publish_solved("drive", Solved(ran=True, sites={5: "unsat", 6: "unreached"}), producer="concolic:drive")
    assert table.solved("drive").closed                               # every site solved to "no solution"
    assert table.solved("symbolic").sites == {10: "reached", 12: "open", 14: "unsat"}   # stages stay apart


def test_a_stage_that_ran_with_no_site_is_ran_not_absent_and_never_closed():
    table = ObligationTable(census_of())
    table.publish_solved("symbolic", Solved(), producer="concolic:symbolic")   # no harness: ran=False, no sites
    view = table.solved("symbolic")
    assert view is not None and not view.ran and view.sites == {} and not view.closed and view.open_lines() == []


def test_a_later_run_of_the_same_stage_supersedes_the_earlier_one():
    table = ObligationTable(census_of())
    table.publish_solved("drive", Solved(ran=True, sites={1: "open"}), producer="a")
    table.publish_solved("drive", Solved(ran=True, sites={1: "reached"}), producer="a")
    assert table.solved("drive").sites == {1: "reached"}


def test_a_string_keyed_site_map_survives_the_round_trip():
    """``tests/test_scheduler.py`` mocks ``concolic.solve`` with ``'nd.py:1'``-style site keys."""
    table = ObligationTable(census_of())
    table.publish_solved("symbolic", Solved(ran=True, sites={"nd.py:1": "open"}), producer="x")
    assert table.solved("symbolic").sites == {"nd.py:1": "open"}


# ------------------------------------------------------------------ what the first run observed

def test_class_hints_are_published_once_and_a_consumer_sees_each_new_one_once():
    table = ObligationTable(census_of())
    assert table.class_hints() is None
    table.publish_class_hints({}, producer="execution")               # nothing observed publishes nothing
    assert table.since("static", CLASS_HINTS) == ()

    table.publish_class_hints({"Point": ["x", "y"]}, producer="execution")
    assert table.class_hints() == {"Point": ["x", "y"]}
    assert len(table.since("static", CLASS_HINTS)) == 1               # new for this consumer ...
    assert table.since("static", CLASS_HINTS) == ()                   # ... and not again
    assert len(table.since("other", CLASS_HINTS)) == 1                # another consumer has its own cursor


def safety_site():
    source = "answer = a / b\n"
    tree = ast.parse(source)
    return site_keys(tree, source, "task.py", "ZeroDivisionError")[tree.body[0].value]


def test_safety_status_and_parents_do_not_certify_an_obligation():
    site = safety_site()
    row = Obligation(site.source_sha256, site.ast_path, site.exception,
                     Invariant(*site.location(), "False"))
    table = ObligationTable(census_of(row))
    assert table.safety() == ()
    canon = '(term 1 (ne Bool (var Int "b") (lit Int 0)))'
    proposed = table.publish_safety(site, canon, producer="template")
    assert proposed.value["status"] == "proposed" and proposed.value["parents"] == ()
    route = table.publish_route("proof:interval", source_sha256=site.source_sha256,
                                classes=(site.exception,), sites=(site,), producer="absint")
    established = table.publish_safety(site, canon, status="established",
                                      parents=(route.identity,), producer="local implication")
    assert table.safety(site) == (proposed, established)
    assert established.value["parents"] == (route.identity,)
    assert not proposed.certified and not established.certified
    assert table.status(row) == OPEN
    assert table.safety(replace(site, source_sha256="different source")) == ()


@pytest.mark.parametrize("kwargs", [
    {"status": "certified"},
    {"status": "established"},
    {"parents": (("proof_route", "missing"),)},
])
def test_safety_rejects_unknown_status_or_missing_evidence(kwargs):
    table = ObligationTable(census_of())
    with pytest.raises(ValueError):
        table.publish_safety(safety_site(), "(term 1 (lit Bool true))", producer="test", **kwargs)
    assert table.safety() == ()


def test_route_publications_are_versioned_and_retain_derivations():
    table = ObligationTable(census_of())
    site = safety_site()
    table.publish_site_records([{"line": 1, "status": "safe"}], producer="absint")
    parent = table.kb.evidence(kind=SITE_RECORD)[0].identity
    first = table.publish_route("proof:interval", source_sha256=site.source_sha256,
                                classes=(site.exception,), sites=(site,), parents=(parent,), producer="absint")
    second = table.publish_route("proof:interval", source_sha256=site.source_sha256,
                                 classes=(site.exception,), sites=(site,), whole_program=True,
                                 location=site.location(), parents=(parent,), producer="absint")
    assert first.key.startswith("v2:") and first.key != second.key
    assert table.proof_routes() == (first, second)
    assert second.value["sites"] == (site.record_key(),)
    assert second.value["site_record_parent_count"] == 1
    assert second.derivations[0].parents == (parent,)
    repeated = table.publish_route("proof:interval", source_sha256=site.source_sha256,
                                   classes=(site.exception,), sites=(site,), parents=(parent,), producer="other")
    assert repeated.identity == first.identity and len(repeated.derivations) == 2
    assert len(table.proof_routes()) == 2


def test_whole_program_route_has_a_location_without_census_rows():
    table = ObligationTable(census_of())
    site = safety_site()
    route = table.publish_route("finite_tree", source_sha256=site.source_sha256,
                                classes=("TypeError",), whole_program=True,
                                location=site.location(), producer="enumeration")
    assert route.value["location"] == site.location()
    assert route.value["sites"] == () and route.value["site_record_parent_count"] == 0 and table.rows() == ()
    with pytest.raises(ValueError, match="projection location"):
        table.publish_route("finite_tree", source_sha256=site.source_sha256,
                            classes=("TypeError",), whole_program=True, producer="enumeration")


def test_route_counts_actual_site_record_parents_without_census_rows():
    table = ObligationTable(census_of())
    site = safety_site()
    table.publish_site_records([{"status": "safe"}, {"status": "safe"}], producer="absint")
    parents = tuple(e.identity for e in table.kb.evidence(kind=SITE_RECORD))
    first = table.publish_route("absint", source_sha256=site.source_sha256, classes=("TypeError",),
                                whole_program=True, location=site.location(), producer="absint",
                                parents=parents[:1])
    second = table.publish_route("absint", source_sha256=site.source_sha256, classes=("TypeError",),
                                 whole_program=True, location=site.location(), producer="absint", parents=parents)
    assert first.value["sites"] == second.value["sites"] == ()
    assert first.value["site_record_parent_count"] == 1 and second.value["site_record_parent_count"] == 2
    assert first.key != second.key and first.key.startswith("v2:")


def test_route_rejects_sites_outside_source_or_exception_coverage():
    table = ObligationTable(census_of())
    site = safety_site()
    for source, classes in [("different", (site.exception,)), (site.source_sha256, ("TypeError",))]:
        with pytest.raises(ValueError, match="outside"):
            table.publish_route("proof:interval", source_sha256=source, classes=classes,
                                sites=(site,), producer="absint")
    assert table.proof_routes() == ()


@pytest.mark.parametrize("site_change,canon,parents", [
    ({"exception": ""}, "(term 1 (lit Bool true))", ()),
    ({"source_sha256": ""}, "(term 1 (lit Bool true))", ()),
    ({"ast_path": ""}, "(term 1 (lit Bool true))", ()),
    ({"statement_location": ("task.py", 0, 1, "<module>")}, "(term 1 (lit Bool true))", ()),
    ({}, "", ()),
    ({}, "  ", ()),
    ({}, None, ()),
    ({}, "(term 1 (lit Bool true))", ("ab",)),
    ({}, "(term 1 (lit Bool true))", (("site_record",),)),
])
def test_invalid_safety_input_publishes_nothing(site_change, canon, parents):
    table = ObligationTable(census_of())
    with pytest.raises(ValueError):
        table.publish_safety(replace(safety_site(), **site_change), canon,
                             parents=parents, producer="test")
    assert table.safety() == () and table.kb.evidence() == ()


@pytest.mark.parametrize("changes", [
    {"route": ""}, {"source_sha256": ""}, {"source_sha256": None},
    {"classes": "TypeError"}, {"classes": ()}, {"classes": (None,)},
    {"whole_program": 1}, {"location": "task.py:1"},
    {"location": ("task.py", 1, 1)}, {"location": ("task.py", 1, 0, "<module>")},
    {"parents": ("ab",)}, {"parents": (("site_record", "missing"),)},
])
def test_invalid_route_input_publishes_nothing(changes):
    table = ObligationTable(census_of())
    kwargs = {"route": "finite_tree", "source_sha256": safety_site().source_sha256,
              "classes": ("TypeError",), "whole_program": True,
              "location": ("task.py", 1, 1, "<module>"), "producer": "test"}
    kwargs.update(changes)
    with pytest.raises(ValueError):
        table.publish_route(**kwargs)
    assert table.proof_routes() == () and table.safety() == ()


@pytest.mark.parametrize("flags, closed", [
    ([], True), (["file_effect"], True), (["file_unmodelled"], False),
    (["stack_unmodelled"], False), (["resource_exhausted"], False),
    (["file_effect", "import_state_unmodelled"], False),
])
def test_concolic_execution_flags_survive_store_publication(flags, closed):
    solved = Solved(ran=True, sites={5: "unsat", 6: "unreached"}, flags=flags)
    table = ObligationTable(census_of())
    table.publish_solved("symbolic", solved, producer="concolic:symbolic")
    view = table.solved("symbolic")
    assert view.flags == tuple(flags)
    assert view.sites == solved.sites
    assert view.closed is closed
    assert solved.closed is closed
    table.publish_solved("symbolic", Solved(ran=True, sites={5: "unsat"}), producer="concolic:symbolic")
    assert table.solved("symbolic").closed
    assert table.solved("symbolic").flags == ()
