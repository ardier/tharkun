"""The contract x engine audit ([internal ref] batch 5, [internal ref]): the table is complete, measured and frozen
(one file per contract under ``tests/contracts/frozen/``).

``_engine_audit.py`` measures; ``frozen/<qualname>.json`` (one file per contract, ``_frozen.py``) freezes.  A cell that changes because
an engine learned (or lost) a contract fails here: close the cell and regenerate that contract's file
(``python -m tests.contracts._frozen write --audit --only <qualname>``).  ``test_engine_audit_consumed.py`` runs every consumed cell
end to end against CPython.
"""

from __future__ import annotations

import pathlib

import pytest

from tests.contracts import _engine_audit as A
from tests.contracts._engine_audit_expected import EXPECTED
from tests.contracts._engine_audit import SAMPLES
from tests.contracts._native import peer_runtime_check_type, run_native

REPORT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "contracts" / "REPORT.md"


def test_every_contract_has_a_sample_a_row_and_nothing_else_does():
    names = set(A.contract_names())
    assert set(SAMPLES) == names
    assert set(EXPECTED) == names


def test_every_row_has_a_cell_for_every_engine_with_a_known_code():
    for q, row in EXPECTED.items():
        assert set(row) == set(A.ENGINES), q
        assert set(row.values()) <= set(A.CODES), q


@pytest.mark.parametrize("qualname", sorted(SAMPLES))
def test_the_sample_runs_clean_on_cpython(qualname):
    """A sample is a valid program: nothing it does raises, so a verdict about it is a verdict about CPython."""
    src = SAMPLES[qualname]
    if qualname.startswith(("random.", "open")):
        pytest.skip("reads the clock/filesystem: valid by construction, not deterministic to pin")
    assert run_native(src, check_type_oracle=peer_runtime_check_type() if qualname == "_sv_verifier.check_type" else None) == "", qualname


@pytest.mark.parametrize("qualname", sorted(SAMPLES))
def test_the_measured_cells_are_the_frozen_ones(qualname):
    measured = {engine: cell[0] for engine, cell in A.measure(qualname).items()}
    assert measured == EXPECTED[qualname], qualname


def test_the_report_names_the_frozen_files_instead_of_embedding_the_table():
    """The table is ``tests/contracts/frozen/<qualname>.json``, one file per contract (``_frozen.py``): a table embedded in the report
    changes on every row any PR moves, so two PRs on different contracts conflicted in the document too."""
    if not REPORT.exists():
        pytest.skip("docs/contracts/REPORT.md is an internal report and is not part of the public release")
    text = REPORT.read_text()
    assert "engine-audit:begin" not in text, "REPORT.md embeds the audit table again: it must name the files instead"
    assert "tests/contracts/frozen/" in text and "tests.contracts._frozen render" in text


def test_the_gaps_are_the_cells_that_do_not_consume_and_do_not_refuse_with_a_reason():
    gaps = {q: [e for e in A.ENGINES if EXPECTED[q][e] in ("M", "G", "X")] for q in EXPECTED}
    assert sum(len(v) for v in gaps.values()) > 0   # the step-2 backlog is real, and listed in the report
