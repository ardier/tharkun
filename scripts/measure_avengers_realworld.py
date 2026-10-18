#!/usr/bin/env python3
"""On/off measurement of the ``avengers`` SV stage on the SV-Benchmarks Python
Real-world set ([internal ref]).

NOT the official BenchExec/testq/cgroup harness (`eregion_sv.bench`,
`scripts/run_chunked.sh` in the `py-inv/eregion-sv` companion repo) -- that
needs infrastructure (a delegated cgroup scope, the sv-benchmarks corpus
checked out at the competition's pins, testq) this script does not assume.
It runs `python -m tharkun.sv` directly, twice per (task, property) pair
(DEFAULT stages vs DEFAULT+avengers), and compares verdict/runs/seconds.
Honest, reduced-fidelity (no cgroup memory accounting, no BenchExec CPU
measurement), but apples-to-apples: both configs run under identical
conditions on the same machine back to back.

Usage::

    python scripts/measure_avengers_realworld.py \\
        --svb /path/to/sv-benchmarks/python/pyperformance \\
        --python .venv/bin/python --out results.jsonl

The `sv-benchmarks` `python/` directory's `pyperformance/` task set is on the
`python-pyperformance-benchmarks` branch of
https://gitlab.com/sosy-lab/benchmarking/sv-benchmarks (a sparse checkout of
`python/` is enough); `--svb` should point at its `pyperformance/` folder.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

WITHOUT_STAGES = "guard_values,comparison,boundaries,composition,concolic"
WITH_STAGES = WITHOUT_STAGES + ",avengers"

_PROP_RE = re.compile(r"property_file:\s*(\S+)")


def task_properties(yml: Path) -> list[Path]:
    text = yml.read_text()
    return [(yml.parent / m.group(1)).resolve() for m in _PROP_RE.finditer(text)]


def run_one(python: str, src: Path, task: Path, prop: Path, stages: str,
            budget: float, run_timeout: float) -> dict:
    cmd = [
        python, "-m", "tharkun.sv", str(task), "--property", str(prop),
        "--budget-seconds", str(budget), "--run-timeout", str(run_timeout),
        "--jobs", "1", "--stages", stages,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(src)
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=budget + 60, env=env)
        out, err, rc = proc.stdout, proc.stderr[-2000:], proc.returncode
    except subprocess.TimeoutExpired:
        out, err, rc = "", "hard timeout", -1
    wall = time.monotonic() - t0
    verdict_line = runs = seconds = component = None
    for line in out.splitlines():
        if line.startswith(("TRUE", "FALSE", "UNKNOWN")):
            verdict_line = line.strip()
        if "tharkun-sv:" in line and "runs in" in line:
            m = re.search(r"(\d+) runs in ([\d.]+)s", line)
            if m:
                runs, seconds = int(m.group(1)), float(m.group(2))
        if line.startswith("tharkun-sv: component="):
            component = line.split("component=", 1)[1].split()[0]
    return {
        "task": task.name, "prop": prop.name, "stages": stages,
        "verdict": verdict_line or "", "runs": runs, "seconds": seconds,
        "component": component, "wall": wall, "rc": rc, "err_tail": err if rc != 0 else "",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--svb", required=True, help="sv-benchmarks python/pyperformance directory")
    ap.add_argument("--python", default=sys.executable, help="interpreter running tharkun.sv")
    ap.add_argument("--src", default=str(Path(__file__).resolve().parents[1] / "src"),
                    help="tharkun's src/ directory (for PYTHONPATH)")
    ap.add_argument("--budget-seconds", type=float, default=60.0)
    ap.add_argument("--run-timeout", type=float, default=20.0)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--out", default="avengers_realworld_results.jsonl")
    a = ap.parse_args(argv)

    task_dir, src = Path(a.svb), Path(a.src)
    ymls = sorted(task_dir.glob("*.yml"))
    pairs = [(yml, prop) for yml in ymls for prop in task_properties(yml)]
    print(f"{len(ymls)} tasks, {len(pairs)} (task,property) pairs", file=sys.stderr)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.parallel) as ex:
        futs = []
        for yml, prop in pairs:
            for stages in (WITHOUT_STAGES, WITH_STAGES):
                futs.append(ex.submit(run_one, a.python, src, yml, prop, stages,
                                      a.budget_seconds, a.run_timeout))
        for done, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            results.append(fut.result())
            if done % 10 == 0:
                print(f"{done}/{len(futs)} done", file=sys.stderr)

    with open(a.out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {a.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
