"""The registry digest ([internal ref], the rules migration): a canonical, hash-seed independent fingerprint of every
``Contract`` the registry holds, so a behaviour-preserving move of a contract into a ``Rule`` class can be PROVEN
byte-identical.

``digest(contract)`` is a short SHA-256 of the contract's canonical JSON (every field, including each claim's
``tested`` flag).  ``tests/contracts/frozen/<qualname>.json`` freezes one per qualname (``_frozen.py``);
``test_registry_digest.py`` fails when a contract is added, removed or changed without the digests following.
``python -m tests.contracts._frozen write --digests`` freezes them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

from tharkun.solve.contracts import all_contracts


def canonical(contract) -> str:
    return json.dumps(dataclasses.asdict(contract), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(contract) -> str:
    return hashlib.sha256(canonical(contract).encode()).hexdigest()[:16]


def registry_digest() -> str:
    data = json.dumps([dataclasses.asdict(c) for c in all_contracts()],
                      sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode()).hexdigest()


def current() -> dict[str, str]:
    return {c.qualname: digest(c) for c in all_contracts()}


if __name__ == "__main__":
    # the digests of the registry as it is now; to FREEZE them use ``python -m tests.contracts._frozen write --digests``
    for q, d in sorted(current().items()):
        print(f"{q}  {d}")
