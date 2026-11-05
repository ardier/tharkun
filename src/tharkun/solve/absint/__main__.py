"""``python -m tharkun.solve.absint PROGRAM.py --property P.prp [--explain]``

Prints ``TRUE`` or ``UNKNOWN(reason)`` as the last line; ``--explain`` first
prints every check relevant to the property with the invariant fact that
discharges it (or ``UNPROVEN``).
"""

from __future__ import annotations

import argparse
import re
import sys
import tokenize

from . import prove


def _classes_of(prp_text: str) -> tuple:
    return tuple(re.findall(r"uncaught\((\w+)\)", prp_text))


def _read_source(path: str) -> str:
    """Read the way CPython reads it: PEP 263 encoding-cookie detection,
    defaulting to UTF-8, no newline normalization ([internal ref]). Duplicated from
    ``tharkun.sv.engine.read_source`` rather than imported: ``tharkun.solve``
    is a dependency-free layer (docs/LAYERS.md; tests/test_layering.py) that
    must not import ``tharkun.sv``."""
    with open(path, "rb") as raw:
        encoding, _ = tokenize.detect_encoding(raw.readline)
        raw.seek(0)
        return raw.read().decode(encoding)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tharkun.solve.absint")
    ap.add_argument("program")
    ap.add_argument("--property", required=True, help="a .prp file, or a comma list of exception classes")
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the per-site records as JSON lines")
    ap.add_argument("--time-budget", type=float, default=60.0)
    a = ap.parse_args(argv)
    if a.property.endswith(".prp"):
        with open(a.property, encoding="utf-8") as f:
            classes = _classes_of(f.read())
    else:
        classes = tuple(x.strip() for x in a.property.split(",") if x.strip())
    src = _read_source(a.program)
    r = prove(src, classes, filename=a.program, time_budget=a.time_budget)
    if a.explain:
        print(r.explain())
    if a.json:
        import json

        for rec in r.records:
            print(json.dumps(rec))
    print(f"tharkun-sv: component={r.component}")
    print("TRUE" if r.verdict == "TRUE" else f"UNKNOWN({r.reason})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
