"""The frozen registry facts, ONE SMALL FILE PER CONTRACT ([internal ref]).

Two facts about every contract are frozen so that a change is a visible, reviewed edit:

* ``audit``: the engine-by-contract audit row (``_engine_audit.py``: the cell code C/M/R/G/X/- of the contract for each of the
  eight engines, measured, not asserted);
* ``digest``: the canonical, hash-seed independent SHA-256 prefix of the ``Contract`` (``_registry_digest.py``: a
  behaviour-preserving move of a contract proves itself byte-identical, a change of one is a digest change in the same commit).

They used to be two generated files (``_engine_audit_expected.py``, ``_registry_digests.py``) that every PR touching any contract
regenerated, so two PRs on different contracts always conflicted.  They are now ``tests/contracts/frozen/<qualname>.json``, one per
contract, holding both facts::

    {
      "audit": {"absint": "C", "symexec": "M", "product": "R", "stationary": "R", "census": "C", "avengers": "-", "draws": "-", "eregion": "G"},
      "digest": "822bdaa41cc8c8fb"
    }

(three lines; ``dumps`` is the one writer)

Two PRs that touch different contracts touch different files; a new contract is a new file.  ``tests/contracts/test_frozen_files.py``
fails unless the files cover exactly the registry, once each, in the canonical format this module writes.  The old module names stay as
thin loaders (``_engine_audit_expected.EXPECTED``, ``_registry_digests.DIGESTS``) so no test import moved.

Generate, from the repository root (the audit measures every contract: run it through ``testq`` with ``--jobs``)::

    python -m tests.contracts._frozen write [--audit] [--digests] [--jobs N] [--only QUALNAME ...]
    python -m tests.contracts._frozen render      # the audit as a Markdown table (one row per contract)
    python -m tests.contracts._frozen summary     # the per-engine counts

``write`` with neither flag does both.  It creates a file for a new contract, rewrites only the fact it was asked for in an existing
one, and deletes the file of a contract that is gone.
"""

from __future__ import annotations

import json
import pathlib

FROZEN_DIR = pathlib.Path(__file__).resolve().parent / "frozen"


def path_of(qualname: str) -> pathlib.Path:
    return FROZEN_DIR / f"{qualname}.json"


def dumps(record: dict) -> str:
    """The canonical text of one file: three lines, ``audit`` (engines in the audit's own order) then ``digest``."""
    if list(record) != ["audit", "digest"]:
        raise ValueError(f"a frozen record has exactly an audit row and a digest, not {list(record)}")
    return "{\n  \"audit\": " + json.dumps(record["audit"]) + ",\n  \"digest\": " + json.dumps(record["digest"]) + "\n}\n"


def load() -> dict[str, dict]:
    """``{qualname: {"audit": {engine: code}, "digest": str}}`` of every file, sorted by qualname."""
    out = {}
    for path in sorted(FROZEN_DIR.glob("*.json"), key=lambda p: p.stem):
        out[path.stem] = json.loads(path.read_text())
    return out


def expected() -> dict[str, dict[str, str]]:
    """The frozen audit table: ``{qualname: {engine: code}}`` (what ``_engine_audit_expected.EXPECTED`` was)."""
    return {q: rec["audit"] for q, rec in load().items()}


def digests() -> dict[str, str]:
    """The frozen contract digests: ``{qualname: digest}`` (what ``_registry_digests.DIGESTS`` was)."""
    return {q: rec["digest"] for q, rec in load().items()}


def write(qualname: str, *, audit: dict | None = None, digest: str | None = None) -> None:
    """Create or update ``frozen/<qualname>.json``: only the facts given are replaced, the other one is kept."""
    FROZEN_DIR.mkdir(exist_ok=True)
    path = path_of(qualname)
    record = json.loads(path.read_text()) if path.exists() else {}
    if not path.exists() and (audit is None or digest is None):
        raise ValueError(f"{qualname}: a new contract file needs both its audit row and its digest")
    if audit is not None:
        record["audit"] = audit
    if digest is not None:
        record["digest"] = digest
    path.write_text(dumps({k: record[k] for k in ("audit", "digest")}))


def prune(valid: set[str]) -> list[str]:
    """Delete the files of contracts that are not in ``valid`` (the registry's qualnames): the names removed."""
    gone = []
    for path in sorted(FROZEN_DIR.glob("*.json")):
        if path.stem not in valid:
            path.unlink()
            gone.append(path.stem)
    return gone


def _measure(qualname: str) -> tuple[str, dict[str, str]]:
    from tests.contracts import _engine_audit as A

    return qualname, {engine: cell[0] for engine, cell in A.measure(qualname).items()}


def _main(argv: list[str]) -> int:
    import argparse
    import multiprocessing as mp

    ap = argparse.ArgumentParser(prog="python -m tests.contracts._frozen")
    ap.add_argument("command", choices=["write", "render", "summary"])
    ap.add_argument("--audit", action="store_true", help="write: measure and freeze the audit rows")
    ap.add_argument("--digests", action="store_true", help="write: freeze the contract digests")
    ap.add_argument("--jobs", type=int, default=1, help="write --audit: measure this many contracts at once")
    ap.add_argument("--only", nargs="*", default=None, help="write: only these qualnames (default: every contract)")
    args = ap.parse_args(argv)
    if args.command in ("render", "summary"):
        from tests.contracts import _engine_audit as A

        table = expected()
        print(A.render(table) if args.command == "render" else A.summary(table))
        return 0
    both = not args.audit and not args.digests
    from tests.contracts import _registry_digest as D
    from tharkun.solve.contracts import all_contracts

    names = sorted(c.qualname for c in all_contracts())
    todo = names if args.only is None else sorted(set(args.only))
    unknown = [q for q in todo if q not in names]
    if unknown:
        raise SystemExit(f"not contracts of the registry: {unknown}")
    exists = {q: path_of(q).exists() for q in todo}
    want_audit = {q for q in todo if args.audit or both or not exists[q]}
    want_digest = {q for q in todo if args.digests or both or not exists[q]}
    rows: dict = {}
    if want_audit:
        todo_audit = sorted(want_audit)
        if args.jobs > 1:
            with mp.Pool(args.jobs) as pool:
                rows = dict(pool.map(_measure, todo_audit, chunksize=1))
        else:
            rows = dict(_measure(q) for q in todo_audit)
    now = D.current()
    for q in todo:
        # a new contract gets both facts (``want_*`` holds it for both); an existing file keeps the fact it was not asked for
        write(q, audit=rows.get(q), digest=now[q] if q in want_digest else None)
    if args.only is None:
        for q in prune(set(names)):
            print("removed the file of", q)
    print(f"wrote {len(todo)} contract file(s) in {FROZEN_DIR}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv[1:]))
