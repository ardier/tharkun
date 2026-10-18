"""The frozen contract digests, loaded from ``tests/contracts/frozen/<qualname>.json`` (one file per contract, ``_frozen.py``).

A compatibility facade ([internal ref]): it holds no data, so a PR that changes a contract edits only that contract's file."""

from tests.contracts._frozen import digests

DIGESTS = digests()
