"""Generic fact producers for the SV scheduler ([internal ref] step 3, [internal ref] / .59).

A *fact producer* is a separately installed component that reads the task and
the open obligations and hands back plain, JSON-safe candidate facts.  Tharkun
publishes them into its :class:`~tharkun.knowledge.KnowledgeBase` through
``tharkun.sv.eregion_bridge``; a fact counts only after
``tharkun.sv.fact_proof`` has certified it independently (owner ruling,
2026-09-29: "a fact counts only after the per-candidate Z3 check certifies it").

Tharkun never imports or names a producer.  The charter (docs/LAYERS.md:
"tharkun never depends on eregion") holds because a producer announces
itself through the ``tharkun.sv.fact_producers`` entry-point group of its own
distribution; Eregion's ``eregion-codegen`` registers one (``eregion``).  With
no producer installed :func:`discover` returns ``()`` and the scheduler spends
nothing here -- there is no switch, and none is needed.

Why a child process, not an in-process call:

* a producer may be slow or may hang; the parent can kill it at the budget;
* the producer's imports (an inversion engine, numpy) must not load into the
  verifier, and its global state (``sys.modules``, environment variables, a
  working directory) must not leak into the verifier's later stages;
* CPU: a reaped child is charged by ``os.times()`` to the very ledger every
  stage shares (``sv/scheduler.py::run``'s ``cpu_seconds``), so the stage is
  accountable to the one CPU-second budget like every other component.

The child runs ``python -m tharkun.sv.fact_producers NAME``.  It reads one JSON
request from stdin and writes one JSON response to stdout; anything the
producer itself prints goes to stderr.  The schema is the one a producer
sees::

    request  = {"schema": "tharkun-fact-producer/1", "program": "/abs/task.py",
                "classes": ["ZeroDivisionError"], "budget_seconds": 12.0,
                "obligations": [{"file": ..., "line": ..., "column": ...,
                                 "function": ..., "exception": ..., "ast_path": ...,
                                 "value": "<the obligation's own invariant>"}]}
    response = {"schema": "tharkun-fact-producer/1", "facts": [{"kind": ..., ...}]}

A producer failure, timeout, malformed answer or non-zero exit yields NO facts:
a producer that crashes proves nothing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable, Iterable

#: the entry-point group a producer registers under
GROUP = "tharkun.sv.fact_producers"
SCHEMA = "tharkun-fact-producer/1"
#: a response larger than this is dropped (a runaway producer must not fill memory)
MAX_RESPONSE_BYTES = 8 * 2**20
MAX_FACTS = 10_000

#: numeric libraries start one busy-waiting thread per core on import; on a
#: many-core host that costs ~20x the CPU of the work itself (measured: one
#: inversion 9.7 s CPU unrestricted, 0.5 s with these set) and the budget is CPU.
_SINGLE_THREAD = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS")}


@dataclass(frozen=True)
class Producer:
    name: str     # the entry-point name; also the ``producer=`` of every derivation
    target: str   # "package.module:callable"


def discover() -> tuple[Producer, ...]:
    """Every installed producer, in name order.  A broken distribution proves
    nothing and is skipped, never raised."""
    try:
        points = metadata.entry_points(group=GROUP)
    except Exception:  # noqa: BLE001 -- unreadable metadata: no producers
        return ()
    found = {}
    for point in points:
        found.setdefault(point.name, Producer(point.name, point.value))
    return tuple(found[name] for name in sorted(found))


def make_request(program: str, classes: Iterable[str], obligations, budget_seconds: float) -> dict:
    """The request a producer sees (see the module docstring).

    ``obligations`` are census ``Obligation`` objects: only open, non-gap rows of
    the property's classes are passed, in the shape of the census's own
    ``Invariant`` (``solve/obligations.py``)."""
    rows = []
    for o in obligations:
        inv = o.invariant
        rows.append({"file": inv.file_name, "line": inv.line, "column": inv.column,
                     "function": inv.function, "exception": o.exception, "ast_path": o.ast_path,
                     "value": inv.value})
    return {"schema": SCHEMA, "program": os.path.abspath(program), "classes": list(classes),
            "budget_seconds": float(budget_seconds), "obligations": rows}


def run(producer: Producer, request: dict, *, timeout: float, memory_bytes: int | None = None,
        log: Callable[[str], None] = lambda msg: None) -> list[dict]:
    """Run one producer in a child and return its facts (``[]`` on any failure)."""
    from tharkun.sv.engine import _cap_memory

    env = dict(os.environ)
    env.update(_SINGLE_THREAD)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # the child must import this very tharkun, wherever it was loaded from: this file is
    # <root>/tharkun/sv/fact_producers.py
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [root, env.get("PYTHONPATH", "")]))
    try:
        proc = subprocess.run([sys.executable, "-m", "tharkun.sv.fact_producers", producer.name],
                              input=json.dumps(request), capture_output=True, text=True, env=env,
                              timeout=max(1.0, timeout), preexec_fn=_cap_memory(memory_bytes))
    except subprocess.TimeoutExpired:
        log(f"facts: producer {producer.name} cut at its budget")
        return []
    except (OSError, ValueError) as e:
        log(f"facts: producer {producer.name} did not start ({type(e).__name__})")
        return []
    if proc.returncode != 0:
        # the exit status only: a producer's own text (its last stderr line) must never reach the
        # verifier's stdout, which the SV harness parses line by line for a verdict token
        log(f"facts: producer {producer.name} failed (exit status {proc.returncode})")
        return []
    if len(proc.stdout) > MAX_RESPONSE_BYTES:
        log(f"facts: producer {producer.name} answer too large; dropped")
        return []
    try:
        response = json.loads(proc.stdout)
        facts = response["facts"]
        if response["schema"] != SCHEMA or not isinstance(facts, list) or len(facts) > MAX_FACTS:
            raise ValueError("bad response shape")
        if not all(isinstance(f, dict) and isinstance(f.get("kind"), str) for f in facts):
            raise ValueError("a fact without a kind")
    except (ValueError, KeyError, TypeError):
        log(f"facts: producer {producer.name} gave a malformed answer; dropped")
        return []
    return facts


def _child(name: str) -> int:
    """The child side: load the named producer, run it on the stdin request."""
    request = json.load(sys.stdin)
    target = next((p.target for p in discover() if p.name == name), None)
    if target is None:
        print(f"no producer named {name!r}", file=sys.stderr)
        return 2
    module, _, attr = target.partition(":")
    produce = getattr(__import__(module, fromlist=[attr]), attr)
    with redirect_stdout(sys.stderr):  # a producer's own prints must not corrupt the answer
        facts = produce(request)
    json.dump({"schema": SCHEMA, "facts": list(facts)}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(_child(sys.argv[1]))
