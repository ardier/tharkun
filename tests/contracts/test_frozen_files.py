"""The frozen files cover exactly the registry, once each ([internal ref]; ``tests/contracts/_frozen.py``).

``tests/contracts/frozen/<qualname>.json`` holds the audit row and the digest of one contract.  Two PRs on different contracts must touch
different files, so every contract has exactly one file and nothing else lives there; the format is the one ``_frozen.py`` writes, so the
generator is the only way a file changes.
"""

from __future__ import annotations

import json
import re

from tests.contracts import _frozen
from tests.contracts._engine_audit import CODES, ENGINES
from tharkun.solve.contracts import all_contracts


def _files():
    return sorted(_frozen.FROZEN_DIR.iterdir())


def test_the_files_are_exactly_the_registry_once_each():
    names = sorted(c.qualname for c in all_contracts())
    files = _files()
    assert all(f.suffix == ".json" for f in files), [f.name for f in files if f.suffix != ".json"]
    stems = sorted(f.stem for f in files)
    assert stems == names, (sorted(set(stems) - set(names)), sorted(set(names) - set(stems)))
    assert len(stems) == len(set(stems))


def test_every_file_has_a_row_for_every_engine_and_a_digest_in_the_canonical_format():
    for f in _files():
        text = f.read_text()
        record = json.loads(text)
        assert list(record) == ["audit", "digest"], f.name
        assert list(record["audit"]) == list(ENGINES), f.name
        assert set(record["audit"].values()) <= set(CODES), f.name
        assert re.fullmatch(r"[0-9a-f]{16}", record["digest"]), f.name
        assert text == _frozen.dumps(record), f"{f.name} is not in the canonical format: regenerate it with tests/contracts/_frozen.py"


def test_the_loaders_give_what_the_files_hold():
    from tests.contracts._engine_audit_expected import EXPECTED
    from tests.contracts._registry_digests import DIGESTS

    loaded = _frozen.load()
    assert EXPECTED == {q: r["audit"] for q, r in loaded.items()}
    assert DIGESTS == {q: r["digest"] for q, r in loaded.items()}
    assert list(EXPECTED) == sorted(EXPECTED)
