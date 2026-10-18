"""The frozen audit table, loaded from ``tests/contracts/frozen/<qualname>.json`` (one file per contract, ``_frozen.py``).

A compatibility facade ([internal ref]): it holds no data, so a PR that moves a cell edits only that contract's file, never this module."""

from tests.contracts._frozen import expected

EXPECTED = expected()
