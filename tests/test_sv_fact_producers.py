"""The generic fact-producer seam ([internal ref] step 3): discovery by entry point, a child
process per producer, plain JSON in and out, and no facts on any failure.

Nothing here names Eregion: the producers are the throwaway modules of
``tests/fake_producers.py``, found the way an installed distribution's would be.
"""
from __future__ import annotations

import json
import os

import pytest

from tharkun.sv import fact_producers as fp

from tests.fake_producers import install


REQUEST = {"schema": fp.SCHEMA, "program": "/x/t.py", "classes": ["ZeroDivisionError"],
           "budget_seconds": 5.0, "obligations": []}


def test_discover_finds_a_registered_producer_and_lists_nothing_otherwise(tmp_path, monkeypatch):
    before = {p.name for p in fp.discover()}
    producer = install(tmp_path, monkeypatch, "def produce(request):\n    return []\n", "found_one")
    assert producer.target == "fake_producer_mod:produce"
    assert {p.name for p in fp.discover()} == before | {"found_one"}


def test_a_child_returns_the_facts_for_the_request_it_was_sent(tmp_path, monkeypatch):
    producer = install(tmp_path, monkeypatch, """
        def produce(request):
            return [{"kind": "invariant_candidate", "echo": request["program"], "n": len(request["classes"])}]
    """)
    facts = fp.run(producer, REQUEST, timeout=30)
    assert facts == [{"kind": "invariant_candidate", "echo": "/x/t.py", "n": 1}]


def test_a_producers_own_prints_do_not_corrupt_its_answer(tmp_path, monkeypatch):
    producer = install(tmp_path, monkeypatch, """
        def produce(request):
            print("progress: 50%")
            return [{"kind": "k"}]
    """)
    assert fp.run(producer, REQUEST, timeout=30) == [{"kind": "k"}]


def test_numeric_libraries_are_held_to_one_thread(tmp_path, monkeypatch):
    """Measured: an inversion cost 9.7 s of CPU with every core's BLAS thread spinning and
    0.5 s with one -- and the budget is CPU."""
    producer = install(tmp_path, monkeypatch, """
        import os
        def produce(request):
            return [{"kind": "env", "omp": os.environ.get("OMP_NUM_THREADS"),
                     "blas": os.environ.get("OPENBLAS_NUM_THREADS")}]
    """)
    (fact,) = fp.run(producer, REQUEST, timeout=30)
    assert (fact["omp"], fact["blas"]) == ("1", "1")


@pytest.mark.parametrize("body", [
    "def produce(request):\n    raise RuntimeError('boom')\n",                      # crashes
    "import sys\ndef produce(request):\n    sys.exit(3)\n",                         # exits non-zero
    "def produce(request):\n    return [{'no_kind': 1}]\n",                         # a fact without a kind
    "def produce(request):\n    return [{'kind': 3}]\n",                            # a non-string kind
    "def produce(request):\n    return [{'kind': 'k', 'x': {1, 2}}]\n",             # not JSON
    "def produce(request):\n    return 'a string'\n",                               # not a list of dicts
])
def test_any_failure_yields_no_facts(tmp_path, monkeypatch, body):
    producer = install(tmp_path, monkeypatch, body)
    lines = []
    assert fp.run(producer, REQUEST, timeout=30, log=lines.append) == []
    assert lines and all(line.startswith("facts: producer fake") for line in lines)


def test_a_producer_that_runs_past_its_budget_is_killed(tmp_path, monkeypatch):
    producer = install(tmp_path, monkeypatch, "import time\ndef produce(request):\n    time.sleep(60)\n")
    lines = []
    assert fp.run(producer, REQUEST, timeout=1.0, log=lines.append) == []
    assert any("cut at its budget" in line for line in lines)


def test_an_unknown_entry_point_name_in_the_child_fails_cleanly(tmp_path, monkeypatch):
    producer = install(tmp_path, monkeypatch, "def produce(request):\n    return []\n")
    ghost = fp.Producer("no_such_producer", producer.target)
    assert fp.run(ghost, REQUEST, timeout=30) == []


def test_make_request_carries_each_open_obligation_in_the_censuss_own_shape():
    from tharkun.solve.census import obligation_census

    src = "from _sv_verifier import nondet_int\nx = nondet_int()\ny = 10 // x\n"
    census = obligation_census(src, ["ZeroDivisionError"], "t.py")
    request = fp.make_request("t.py", ["ZeroDivisionError"], census.obligations, 4.0)

    assert request["schema"] == fp.SCHEMA and request["budget_seconds"] == 4.0
    assert os.path.isabs(request["program"])
    (row,) = request["obligations"]
    assert (row["file"], row["line"], row["function"], row["exception"]) == ("t.py", 3, "<module>", "ZeroDivisionError")
    assert row["value"] == census.obligations[0].invariant.value
    json.dumps(request)         # plain data, round-trips


def test_a_failed_producers_own_text_never_reaches_the_log(tmp_path, monkeypatch):
    """MG review (#110): the verifier's stdout is parsed line by line for a verdict token, so a crashing
    producer's last stderr line (here a bare ``FALSE(ZeroDivisionError)``) is never echoed: the log
    carries the exit status only."""
    producer = install(tmp_path, monkeypatch, """
        import sys
        def produce(request):
            sys.stderr.write("FALSE(ZeroDivisionError)\\n")
            sys.exit(7)
    """)
    lines = []
    assert fp.run(producer, REQUEST, timeout=30, log=lines.append) == []
    assert lines == ["facts: producer fake failed (exit status 7)"]
