"""Solve for the draws that reach a violation site, instead of enumerating them ([internal ref] slice 2).

The small-first search RUNS draw vectors and waits for one to raise.  That fails
whenever the violation is guarded by a condition on a computed value
(``if x * x - 3 * x == 1519054: raise ...``): no enumeration window contains the
answer.  This stage SOLVES for it with Tharkun's own machinery:

1. :func:`tharkun.sv.sites.build_harness` lifts the task's entry code (the main block, the body of the
   one function the program ends by calling, or its module-level statements) into
   ``__sv_harness(d0, d1, ...)``, one parameter per ``_sv_verifier`` draw, and lists
   the violation sites (explicit ``raise`` / ``assert`` of a property class).
2. A helper process (``_concolic_worker.py``, address space capped like every child
   run) points :func:`tharkun.driver.drive_to_line` at each site.  The driver is
   Tharkun's own portfolio: the static guard solver (``solve/condition``), the
   live-snapshot mutation, minleaf and hypothesis/reach search -- each proposal
   verified by traced execution. Reach-search harvests live comparison operands.
3. For every site it reaches, the worker re-runs the harness with the solved inputs
   and records the draws in execution order -- exactly the child's draw vector.

The engine then REPLAYS each vector on the unmodified program in a fresh child;
only a run that really raises a property class is a FALSE.  So nothing here is
trusted: a wrong solve costs one run, never a wrong verdict.

Each candidate carries its ``component`` for the results (the results' component column):
``concolic:guard_solve`` (the static guard solver flipped a guard),
``concolic:z3`` (the SMT flipper), ``concolic:drive`` (snapshot / minleaf /
hypothesis, or no flip needed).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field

from tharkun.solve.environment import subprocess_environment
from tharkun.sv.sites import build_harness, harness_namespace_refusal, harness_refusal

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_concolic_worker.py")

#: technique names drive_to_line records, strongest attribution first
_COMPONENT = (("symexec", "concolic:symexec"), ("static", "concolic:guard_solve"))


@dataclass
class Candidate:
    vector: list
    component: str
    detail: dict = field(default_factory=dict)


@dataclass
class Solved:
    """What the concolic stage found: solved vectors to replay, and per site what the
    solvers concluded -- ``reached`` (a vector), ``unsat`` / ``unreached`` (no modelled
    run fires it: the engine may stop searching early), ``open`` (undecided)."""
    candidates: list = field(default_factory=list)
    sites: dict = field(default_factory=dict)       # harness line -> outcome
    ran: bool = False                              # a worker ran on at least one site
    flags: list = field(default_factory=list)

    @property
    def closed(self) -> bool:
        """Every site was solved to "no solution" (and there was at least one)."""
        return self.ran and not (set(self.flags) - {"file_effect"}) and bool(self.sites) and all(o in ("unsat", "unreached")
                                                     for o in self.sites.values())


def component_of(techniques) -> str:
    ts = [str(t) for t in techniques if t]
    for needle, comp in _COMPONENT:
        if any(needle in t for t in ts):
            return comp
    return "concolic:drive"


def solve(program: str, classes, *, budget_seconds: float, memory_bytes: int | None = None,
          python: str | None = None, extra_sites=None, symbolic: bool = True, drive: bool = True,
          lines=None, site_kind=None, log=None, obligations=None, call_targets=None,
          proven_classes=(), files=None) -> Solved:
    """Solve for draw vectors toward the violation sites of ``program``: explicit
    ``raise`` / ``assert``, implicit sites, and ``extra_sites`` (may-raise records of
    another analysis, on original lines -- see :func:`tharkun.sv.sites.build_harness`).

    ``symbolic`` runs the one-shot symbolic execution of the whole harness toward every
    site; ``drive`` runs the per-site stages (the backward-slice solve, then
    :func:`tharkun.driver.drive_to_line`).  ``lines`` restricts the sites to those
    harness lines (the ones an earlier symbolic-only call left open).  ``call_targets`` are the program's
    ``call_target`` records (``solve/call_targets.py``, published by ``scheduler.publish_module_namespace``); the
    whole-harness symbolic run resolves user-class calls through them (``sv/_symexec.py``), and without
    them (``None``) a call to a class stays unmodelled; ``proven_classes`` are the names of the proven
    class-origin ``module_constant`` records (``ObligationTable.proven_classes``)."""
    say = log or (lambda msg: None)
    from tharkun.sv.engine import read_source
    try:
        source = read_source(program)
    except (OSError, SyntaxError, UnicodeDecodeError):
        return Solved()
    refusal = harness_namespace_refusal(source)
    if refusal:
        say(f"concolic: {refusal}")
        return Solved(flags=["import_state_unmodelled"])
    from tharkun.solve.slicing import analysis_slice
    sliced = analysis_slice(source, classes, filename=program, obligations=obligations)
    if sliced.removed:
        say(f"slice: {len(sliced.removed)} statements removed; original replay required")
    h = build_harness(sliced.source, classes, extra_sites, call_targets)
    if h is None or not h.sites:
        why = ((harness_refusal(sliced.source, classes, extra_sites, call_targets) or "no liftable draw")
               if h is None else "no site")
        say(f"concolic: no harness or no site ({why})")
        return Solved()
    if site_kind is not None:
        h.sites = [s for s in h.sites if s["kind"] == site_kind]
        if not h.sites:
            return Solved()
    if lines is not None:
        h.sites = [s for s in h.sites if s["line"] in set(lines)]
        if not h.sites:
            return Solved()
    say(f"concolic: {len(h.params)} lifted draws ({h.entry}), {h.unlifted} left to the stub, "
        f"{len(h.sites)} sites at harness lines {[s['line'] for s in h.sites]}")
    with tempfile.TemporaryDirectory(prefix="tharkun-sv-concolic-") as tmp:
        path = os.path.join(tmp, "sv_harness.py")
        with open(path, "w") as fh:
            fh.write(h.source)
        req = os.path.join(tmp, "req.json")
        out = os.path.join(tmp, "out.json")
        with open(req, "w") as fh:
            json.dump({"harness": path, "program_dir": os.path.dirname(os.path.abspath(program)),
                       "source_binding": [h.source, os.path.abspath(program), source],
                       "sites": h.sites, "seed": h.seed(), "out": out, "feeds": h.feeds,
                       "params": h.params, "kinds": h.kinds, "symbolic": symbolic, "drive": drive,
                       "call_targets": h.scoped(call_targets), "proven_classes": list(proven_classes),
                       "files": list(files) if files is not None else [os.path.abspath(program)],
                       "scratch": tmp, "deadline": time.time() + budget_seconds}, fh)
        pkg_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        env = subprocess_environment()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [pkg_root, env.get("PYTHONPATH", "")]))
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        from tharkun.sv.engine import _cap_memory
        try:
            proc = subprocess.run([python or sys.executable, WORKER, req], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=max(1.0, budget_seconds + 5),
                                  preexec_fn=_cap_memory(memory_bytes))
        except subprocess.TimeoutExpired:
            say("concolic: worker timed out")
            return Solved()
        if not os.path.exists(out):
            say(f"concolic: worker failed: {(proc.stderr or '')[-300:]}")
            return Solved()
        try:
            with open(out) as fh:
                res = json.load(fh)
        except (OSError, ValueError):
            say("concolic: worker left a partial record")
            return Solved()
    solved = Solved(ran=True, sites={s["line"]: "open" for s in h.sites}, flags=res.get("flags", []))
    cands = solved.candidates
    calls = res.get("solver_calls", 0)
    for r in res.get("reached", []):
        comp = component_of(r.get("techniques", []))
        say(f"concolic: reached line {r['line']} ({r['exc']}) from {r['entry']} in {r['rounds']} rounds via "
            f"{r.get('techniques') or ['seed']} -> {comp}")
        cands.append(Candidate(r["vector"], comp, {"site_line": r["line"], "rounds": r["rounds"],
                                                   "entry": r["entry"], "solver_calls": calls,
                                                   "techniques": r.get("techniques", [])}))
        solved.sites[r["line"]] = "reached"
    for r in res.get("missed", []):
        solved.sites[r["line"]] = r.get("outcome", "open")
        say(f"concolic: line {r['line']} not reached ({solved.sites[r['line']]}): {r['status']}")
    say(f"concolic: {calls} solver rounds")
    return solved
