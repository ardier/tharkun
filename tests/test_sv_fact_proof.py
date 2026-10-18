"""Facts to proof ([internal ref] step 3): producers -> store -> independent certification -> TRUE.

``verify`` runs the stage after the first run and before ``prove_dynamic``.  The
producers here are the throwaway ones of ``tests/fake_producers.py`` (the real one,
Eregion's, lives in another repo and is exercised by the umbrella's tests); what is
pinned is the CONSUMER's discipline:

* a producer's nomination counts only after the independent checker proved it;
* TRUE needs EVERY open obligation covered (a partial nomination decides nothing);
* a lying producer cannot make a violating program TRUE;
* with no producer installed, or with proofs ablated, the stage costs nothing.

The checker needs CPython 3.14 (``sv/validate.py``); the tests skip elsewhere.
"""
from __future__ import annotations

import sys
import textwrap

import pytest

from tharkun.solve.census import obligation_census
from tharkun.sv import fact_producers
from tharkun.sv import verify
from tharkun.sv.eregion_bridge import CERTIFIED_INVARIANT, INVARIANT_CANDIDATE
from tharkun.sv.fact_proof import prove_from_facts
from tharkun.sv.obligation_table import ObligationTable

from tests.fake_producers import NOMINATE_ALL, NOMINATE_FIRST, install

pytestmark = pytest.mark.skipif(sys.version_info[:2] != (3, 14), reason="the source checker needs CPython 3.14")

ZD = ("ZeroDivisionError",)

#: absint's intervals cannot see that 2*x + 1 is never 0 (parity); the checker's Z3 can
ODD = """
    import _sv_verifier

    if __name__ == "__main__":
        x = _sv_verifier.nondet_int()
        d = 2 * x + 1
        r = 10 // d
"""
#: the same shape with the violation back in: d == 0 at x == -1
EVEN = ODD.replace("2 * x + 1", "2 * x + 2")
TWO_SAFE_DIVISIONS = """
    import _sv_verifier

    if __name__ == "__main__":
        x = _sv_verifier.nondet_int()
        d1 = 2 * x + 1
        a = 10 // d1
        d2 = 2 * x + 3
        b = 20 // d2
"""


def task(tmp_path, body, name="t.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body).lstrip("\n"))
    return str(path)


def producer(tmp_path, monkeypatch, body):
    """Register a throwaway producer (its own directory, so it never sits next to the task)."""
    where = tmp_path / "producer"
    where.mkdir()
    return install(where, monkeypatch, body, only=True)


def run(path, **kw):
    lines = []
    kw.setdefault("budget_seconds", 60)
    kw.setdefault("jobs", 1)
    verdict = verify(path, ZD, log=lines.append, **kw)
    return verdict, lines


def test_a_certified_nomination_decides_true_and_names_the_producer(tmp_path, monkeypatch):
    producer(tmp_path, monkeypatch, NOMINATE_ALL)
    path = task(tmp_path, ODD)

    verdict, lines = run(path)

    assert verdict.verdict == "true"
    assert verdict.component == "proof:facts:fake"
    assert verdict.detail["proof_checked"] is True           # the finalizer's whole-witness check agrees
    assert any("published 1 of 1 fact(s)" in line for line in lines)
    assert any("1 of 1 candidate invariant(s) certified" in line for line in lines)
    assert "canary: no violation in" in verdict.reason         # a TRUE from facts is tested like any proof
    assert verdict.runs > 1
    assert verdict.detail["canary"] == dict(attempted=True, completed=True, cut=False,
                                           reason="no violation")


def test_without_a_producer_nothing_is_spent_and_nothing_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(fact_producers, "discover", lambda: ())
    verdict, lines = run(task(tmp_path, ODD))

    assert verdict.verdict == "unknown" and verdict.component == "none"
    assert not any(line.startswith("facts:") for line in lines)


def test_proofs_ablated_means_the_stage_does_not_run(tmp_path, monkeypatch):
    producer(tmp_path, monkeypatch, NOMINATE_ALL)
    verdict, lines = run(task(tmp_path, ODD), prove=False)

    assert verdict.verdict != "true"
    assert not any(line.startswith("facts:") for line in lines)


def test_a_lying_producer_cannot_make_a_violating_program_true(tmp_path, monkeypatch):
    """The producer nominates the obligation although x == -1 violates it.  The checker cannot
    prove it, so nothing is certified, and the search finds the violation."""
    producer(tmp_path, monkeypatch, NOMINATE_ALL)
    verdict, lines = run(task(tmp_path, EVEN), concolic=False)      # the search, not the solver, finds x == -1

    assert verdict.verdict == "false"
    assert any("0 of 1 candidate invariant(s) certified" in line for line in lines)
    assert not any("is covered by a certified invariant" in line for line in lines)


def test_a_partial_nomination_decides_nothing(tmp_path, monkeypatch):
    producer(tmp_path, monkeypatch, NOMINATE_FIRST)
    verdict, lines = run(task(tmp_path, TWO_SAFE_DIVISIONS))

    assert verdict.verdict == "unknown"
    assert any("1 of 2 obligation(s) covered" in line for line in lines)


def test_a_producer_that_fails_proves_nothing(tmp_path, monkeypatch):
    producer(tmp_path, monkeypatch, "def produce(request):\n    raise RuntimeError('boom')\n")
    verdict, lines = run(task(tmp_path, ODD))

    assert verdict.verdict == "unknown"
    assert any("producer fake failed" in line for line in lines)


def test_a_slow_producer_is_cut_at_the_stage_budget_and_proves_nothing(tmp_path, monkeypatch):
    """The stage's share is 15% of the CPU left (here ~3 s of 20): a producer that sleeps for a
    minute is killed there, not waited for."""
    import time

    producer(tmp_path, monkeypatch, "import time\ndef produce(request):\n    time.sleep(60)\n")
    started = time.monotonic()
    verdict, lines = run(task(tmp_path, ODD), budget_seconds=20)

    assert time.monotonic() - started < 45
    assert verdict.verdict == "unknown"
    assert any("cut at its budget" in line for line in lines)


def test_a_malformed_fact_is_dropped_the_rest_still_count(tmp_path, monkeypatch):
    producer(tmp_path, monkeypatch, """
        def produce(request):
            good = [{"kind": "invariant_candidate",
                     "site": {"file": o["file"], "line": o["line"], "column": o["column"], "function": o["function"]},
                     "value": o["value"]} for o in request["obligations"]]
            return [{"kind": "invariant_candidate", "value": "d != 0"},     # no site
                    {"kind": "from_the_future"}] + good
    """)
    verdict, lines = run(task(tmp_path, ODD))

    assert verdict.verdict == "true"
    assert any("dropped a malformed fact" in line for line in lines)
    assert any("published 1 of 3 fact(s)" in line for line in lines)


def test_candidates_are_read_back_from_the_store_and_certified_only_after_the_check(tmp_path, monkeypatch):
    """Drive the consumer directly: the store holds the candidate as a proposal first and the
    certified kind only for what the checker proved."""
    producer(tmp_path, monkeypatch, NOMINATE_ALL)
    path = task(tmp_path, ODD)
    source = open(path).read()
    census = obligation_census(source, ZD, path)
    table = ObligationTable(census)
    kb = table.kb
    lines = []

    verdict = prove_from_facts(path, ZD, source=source, specification="CHECK( init(main()), ! uncaught(ZeroDivisionError))\n",
                               table=table, python=sys.executable, remaining=lambda: 60.0, say=lines.append)

    assert verdict is not None and verdict.verdict == "true"
    (candidate,) = kb.evidence(kind=INVARIANT_CANDIDATE)
    (certified,) = kb.evidence(kind=CERTIFIED_INVARIANT)
    assert certified.value["invariant"] == candidate.value["invariant"] == census.obligations[0].invariant
    assert certified.derivations[0].parents == ((INVARIANT_CANDIDATE, candidate.key),)
    assert candidate.derivations[0].producer == "fake"


def test_a_row_already_decided_is_not_open_so_no_producer_is_asked_about_it(tmp_path, monkeypatch):
    """The stage reads the LIVE table: obligations a certified decision closed are not offered again."""
    producer(tmp_path, monkeypatch, "def produce(request):\n    raise SystemExit('must not run')\n")
    path = task(tmp_path, ODD)
    source = open(path).read()
    table = ObligationTable(obligation_census(source, ZD, path))
    table.discharge(ZD, "proof:interval")
    lines = []

    assert prove_from_facts(path, ZD, source=source, specification="CHECK( init(main()), ! uncaught(ZeroDivisionError))\n",
                            table=table, python=sys.executable, remaining=lambda: 60.0, say=lines.append) is None
    assert lines == []


def test_a_violated_row_blocks_a_facts_true_even_when_every_other_row_is_certified(tmp_path, monkeypatch):
    """Fail closed (MG review of 3a): a row a confirmed replay violated is a FALSE.  Even if the stage were
    reached with such a row (the scheduler ends the run in found() first), no producer is asked and no
    TRUE is decided over the rows that remain open."""
    producer(tmp_path, monkeypatch, "def produce(request):\n    raise SystemExit('must not run')\n")
    path = task(tmp_path, TWO_SAFE_DIVISIONS)
    source = open(path).read()
    table = ObligationTable(obligation_census(source, ZD, path))
    rows = table.rows(ZD)
    assert len(rows) == 2
    first = rows[0]
    table.violate("ZeroDivisionError", first.invariant.line, "test:replay", column=first.invariant.column)
    assert table.status(first) == "violated" and len(table.open_rows(ZD)) == 1
    lines = []

    assert prove_from_facts(path, ZD, source=source, specification="CHECK( init(main()), ! uncaught(ZeroDivisionError))\n",
                            table=table, python=sys.executable, remaining=lambda: 60.0, say=lines.append) is None
    assert lines == []                 # refused before any producer was started


def test_an_incomplete_census_or_an_adapter_gap_never_reaches_a_producer(tmp_path, monkeypatch):
    producer(tmp_path, monkeypatch, "def produce(request):\n    raise SystemExit('must not run')\n")
    # print(100 // d): the census cannot build an invariant for a division inside a call (a gap row)
    path = task(tmp_path, """
        import _sv_verifier

        if __name__ == "__main__":
            x = _sv_verifier.nondet_int()
            print(100 // (2 * x + 1))
    """)
    source = open(path).read()
    census = obligation_census(source, ZD, path)
    assert any(o.gap for o in census.obligations)
    lines = []

    assert prove_from_facts(path, ZD, source=source, specification="CHECK( init(main()), ! uncaught(ZeroDivisionError))\n",
                            table=ObligationTable(census), python=sys.executable,
                            remaining=lambda: 60.0, say=lines.append) is None
    assert lines == []                 # no producer was even started


SPEC = "CHECK( init(main()), ! uncaught(ZeroDivisionError))\n"


def prove_current(path, source, table, lines, *, classes=ZD, specification=SPEC,
                  remaining=lambda: 60.0):
    return prove_from_facts(path, classes, source=source, specification=specification,
                            table=table, python=sys.executable, remaining=remaining,
                            say=lines.append)


def test_stored_certification_is_checked_again_for_unchanged_source(tmp_path, monkeypatch):
    producer(tmp_path, monkeypatch, NOMINATE_ALL)
    path = task(tmp_path, ODD)
    source = open(path).read()
    table = ObligationTable(obligation_census(source, ZD, path))
    lines = []

    assert prove_current(path, source, table, lines).verdict == "true"
    again = ObligationTable(table.census, table.kb)
    assert prove_current(path, source, again, lines).verdict == "true"
    assert lines.count("facts: 1 of 1 candidate invariant(s) certified") == 2
    assert len(table.kb.evidence(kind=CERTIFIED_INVARIANT)) == 1


def test_stored_certification_cannot_survive_changed_source(tmp_path, monkeypatch):
    producer(tmp_path, monkeypatch, NOMINATE_ALL)
    path = task(tmp_path, ODD)
    source = open(path).read()
    table = ObligationTable(obligation_census(source, ZD, path))
    assert prove_current(path, source, table, []).verdict == "true"

    changed = source.replace("2 * x + 1", "2 * x + 2")
    from pathlib import Path
    Path(path).write_text(changed)
    current = ObligationTable(obligation_census(changed, ZD, path), table.kb)
    lines = []
    assert prove_current(path, changed, current, lines) is None
    assert "facts: 0 of 1 candidate invariant(s) certified" in lines
    assert len(current.kb.evidence(kind=CERTIFIED_INVARIANT)) == 1  # history, not current proof


@pytest.mark.parametrize("failure", ["crash", "budget"])
def test_stored_certification_cannot_survive_a_missing_current_check(tmp_path, monkeypatch, failure):
    producer(tmp_path, monkeypatch, NOMINATE_ALL)
    path = task(tmp_path, ODD)
    source = open(path).read()
    table = ObligationTable(obligation_census(source, ZD, path))
    assert prove_current(path, source, table, []).verdict == "true"
    current = ObligationTable(table.census, table.kb)
    lines = []
    remaining = lambda: 60.0
    if failure == "crash":
        from tharkun.sv import validate

        def crash(*args, **kwargs):
            raise RuntimeError("checker unavailable")

        monkeypatch.setattr(validate, "check_candidates_individually", crash)
    else:
        budget = iter([60.0, 0.0])  # admitted stage, exhausted before certification
        remaining = lambda: next(budget)

    assert prove_current(path, source, current, lines, remaining=remaining) is None
    assert "facts: 0 of 1 candidate invariant(s) certified" in lines


def test_partial_discharge_without_a_nomination_is_not_current_coverage(tmp_path, monkeypatch):
    producer(tmp_path, monkeypatch, NOMINATE_ALL)
    path = task(tmp_path, TWO_SAFE_DIVISIONS)
    source = open(path).read()
    table = ObligationTable(obligation_census(source, ZD, path))
    from tharkun.facts import Fact
    from tharkun.sv.obligation_table import DISCHARGED, OBLIGATION, subject_of
    table.kb.add(Fact(OBLIGATION, subject_of(table.rows(ZD)[0]), DISCHARGED,
                      source="previous:proof", certified=True))
    lines = []

    assert prove_current(path, source, table, lines) is None
    assert "facts: fake published 1 of 1 fact(s)" in "\n".join(lines)
    assert any("1 of 2 obligation(s) covered" in line for line in lines)


def test_partial_discharge_keeps_good_stored_nominations_checkable(tmp_path, monkeypatch):
    producer(tmp_path, monkeypatch, NOMINATE_ALL)
    path = task(tmp_path, TWO_SAFE_DIVISIONS)
    source = open(path).read()
    table = ObligationTable(obligation_census(source, ZD, path))
    assert prove_current(path, source, table, []).verdict == "true"
    from tharkun.facts import Fact
    from tharkun.sv.obligation_table import DISCHARGED, OBLIGATION, subject_of
    table.kb.add(Fact(OBLIGATION, subject_of(table.rows(ZD)[0]), DISCHARGED,
                      source="previous:proof", certified=True))
    current = ObligationTable(table.census, table.kb)
    lines = []

    verdict = prove_current(path, source, current, lines)
    assert verdict.verdict == "true"
    assert "facts: 2 of 2 candidate invariant(s) certified" in lines
    assert len(verdict.detail["certified"]) == 2


def test_stored_certification_is_checked_for_the_current_property(tmp_path, monkeypatch):
    producer(tmp_path, monkeypatch, NOMINATE_ALL)
    path = task(tmp_path, ODD + "    assert False\n")
    source = open(path).read()
    table = ObligationTable(obligation_census(source, ZD, path))
    assert prove_current(path, source, table, []).verdict == "true"

    classes = ("ZeroDivisionError", "AssertionError")
    specification = SPEC + "CHECK( init(main()), ! uncaught(AssertionError))\n"
    current = ObligationTable(obligation_census(source, classes, path), table.kb)
    lines = []
    assert prove_current(path, source, current, lines, classes=classes,
                         specification=specification) is None
    assert any(o.gap for o in current.rows(classes))  # a property adapter gap also fails closed


def test_stored_certification_cannot_cover_a_different_entry_point(tmp_path, monkeypatch):
    producer(tmp_path, monkeypatch, NOMINATE_ALL)
    path = task(tmp_path, ODD)
    source = open(path).read()
    table = ObligationTable(obligation_census(source, ZD, path))
    assert prove_current(path, source, table, []).verdict == "true"

    current = ObligationTable(table.census, table.kb)
    lines = []
    # Only init(main()) is supported by the checker; an old main certificate
    # must not bypass the current specification's refusal.
    specification = SPEC.replace("init(main())", "init(other())")
    assert prove_current(path, source, current, lines, specification=specification) is None
    assert "facts: 0 of 1 candidate invariant(s) certified" in lines
