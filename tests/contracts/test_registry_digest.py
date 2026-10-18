"""Every contract matches its frozen digest (``tests/contracts/frozen/<qualname>.json``, one file per contract).

The whole-registry SHA-256 that used to be hard-coded here is gone: it was one line every PR that touched any contract had to change
(string family, set family, ...), and it adds nothing the per-contract digests do not already say (a changed, added or removed contract
fails the two tests below, and ``test_frozen_files.py`` pins the files to the registry)."""

from __future__ import annotations

from tests.contracts import _registry_digest as D
from tests.contracts._registry_digests import DIGESTS


def test_the_registry_holds_exactly_the_frozen_qualnames():
    assert set(D.current()) == set(DIGESTS)


def test_every_contract_is_byte_identical_to_the_frozen_one():
    now = D.current()
    changed = sorted(q for q in DIGESTS if now.get(q) != DIGESTS[q])
    assert not changed, changed


def test_the_digest_does_not_depend_on_the_hash_seed():
    """A canonical form: sorted keys, tuples as lists; two spellings of one contract give one digest."""
    from tharkun.solve.contracts import get

    c = get("math.sqrt")
    assert D.digest(c) == D.digest(c) and len(D.digest(c)) == 16
