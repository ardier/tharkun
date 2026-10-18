"""Measured metric, not a test: how often the REAL Pynguin search seeds a value inside a class-guard window.

``python scripts/pynguin_seed_hit_rate.py [-n RUNS] [--seconds PER_CLASS]`` runs ``tharkun.sv.avengers_seed.seed_values`` on the
``Bomb`` program of ``tests/test_sv_avengers_stage.py`` ``-n`` times (default 10) and prints ONE JSON line::

    {"metric": "pynguin_seed_hit_rate", "runs": 10, "hits": 7, "rate": 0.7, "seconds_each": [...], "pynguin": true}

It always exits 0: a randomized, time-boxed search has a success RATE, not a pass/fail, which is why no gate test asserts that
it succeeds (docs/TESTING.md, "Randomized and wall-clock-boxed checks").  Run by the nightly (WAT); a falling rate is the signal.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import textwrap
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

PROGRAM = textwrap.dedent("""
    import _sv_verifier

    class Bomb:
        def __init__(self, value: int):
            self.value = value

        def band(self):
            if 1_000_000 < self.value < 1_000_100:
                raise ValueError("boom")
            return "low"

    if __name__ == "__main__":
        x = _sv_verifier.nondet_int()
        Bomb(x).band()
""").lstrip("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", "--runs", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=15.0, help="Pynguin's search budget per class (the stage's own default)")
    a = ap.parse_args()
    from tharkun.avengers.backends import pynguin_available
    from tharkun.sv.avengers_seed import seed_values

    available = pynguin_available()
    hits, seconds = 0, []
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "reachable.py")
        with open(path, "w") as fh:
            fh.write(PROGRAM)
        for _ in range(a.runs if available else 0):
            t0 = time.monotonic()
            ints = seed_values(path, max_classes=1, per_class_seconds=a.seconds, total_seconds=2 * a.seconds).get("int", [])
            seconds.append(round(time.monotonic() - t0, 1))
            hits += any(1_000_000 < v < 1_000_100 for v in ints)
    runs = len(seconds)
    print(json.dumps({"metric": "pynguin_seed_hit_rate", "runs": runs, "hits": hits,
                      "rate": (hits / runs) if runs else None, "seconds_each": seconds, "pynguin": available}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
