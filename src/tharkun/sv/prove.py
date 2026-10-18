"""Static proof stages of ``tharkun.sv`` ([internal ref] group C; lanes B and E).

``prove(program, classes, ...)`` runs after the deterministic check and before
the search.  It returns a TRUE :class:`~tharkun.sv.engine.Verdict` only with a
proof, else ``None``.  ``engine.verify`` then runs a short *canary* search: a
FALSE it finds wins over the proof (a wrong TRUE costs far more than an
UNKNOWN), and only a canary that finds nothing returns the proof's TRUE.

Stages, in order (the first TRUE wins):

0. ``proof:no-site`` -- :func:`prove_static` (lane B): ``solve/census.py`` finds no
   construct the program can run -- operator, library call, raise, assert --
   that can raise a property class with any values.  An allowlist over
   ``solve/pyresolve.py``'s resolution: anything reflective refuses.  Static.
1. ``proof:interval`` / ``proof:shape`` -- :mod:`tharkun.solve.absint`, a sound
   whole-program abstract interpreter (docs/ABSINT.md).  Static, never runs the
   program.

2. ``proof:stationary`` -- :func:`prove_dynamic` (lane B), run by ``engine.verify``
   after the first run when no static stage proved the property.  It combines a
   static claim with an observed one:

   * STATIC (``tharkun.solve.drawflow.plan_draws``): every ``nondet_int`` draw
     reaches the program only as the trip count ``c + abs(d)`` of ``for _ in
     range(...)`` loops whose target is never read (a *trip* draw), or only
     through ``abs(d) % M`` (a *quotient* draw); every other draw is
     ``nondet_bool``.  So an execution is fixed by the trip counts, the residues
     and the finite draws, and a loop body cannot observe its trip count.  And
     ``tharkun.solve.determinism.determinism_census`` certifies that a run is a
     function of the program and its draws (an ALLOWLIST over imports, calls and
     attributes; no set order and no address reaches the program).
   * OBSERVED (the probe child ``_probe.py``): with ``m = max(c, 1)``, the draw
     tree is enumerated over trip counts ``{c, ..., m + 1}``, every residue and
     every finite value, under two hash seeds.  In every run, every instance of
     a trip loop whose trip count is ``m + 1`` either left the loop early or
     reached the heads ``m + 1`` and ``m + 2`` with EQUAL canonical snapshots of
     the whole reachable state (all frames' locals out to the module, what they
     reach, the ``random`` state, the model's draw count).

   Why that covers every execution: between draws a run is deterministic, and a
   loop body sees no trip count, so the iteration after a head depends only on
   the state at that head.  With ``S(m) == S(m + 1)`` the iterations ``m + 1, m
   + 2, ...`` all start from the same state and do what iteration ``m + 1`` did
   in the ``m + 1`` run (no violation, no draw -- the draw count is part of the
   state), and leave the loop in ``S(m)``.  Trip counts ``c .. m`` ran directly
   (trip 0 included when ``c == 0``).  A probe run that raises a property class
   is replayed on the plain child and returned as FALSE with its witness.

Any exception inside a stage is an UNKNOWN for that stage, never a verdict.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor

from tharkun.facts import Fact
from tharkun.solve.determinism import determinism_census
from tharkun.solve.drawflow import plan_draws
from tharkun.sv.obligation_table import call_target_evidence_in
from tharkun.sv.store_witness import bind_route



def prove_static(program: str, classes, *, remaining=None, log=None, files=None):
    """``proof:no-site`` (``solve/census.py``): no construct of the program can raise
    ``classes``.  Static and instant; a TRUE ``Verdict`` or None."""
    from tharkun.solve.census import site_census
    from tharkun.sv.engine import Verdict, read_source

    say = log or (lambda msg: None)
    try:
        source = read_source(program)
        census = site_census(source, tuple(classes), program, declared_files=files or ())
    except Exception as e:  # noqa: BLE001 -- a proof stage never decides by crashing
        say(f"no-site: internal error {type(e).__name__}: {e}")
        return None
    if not census.no_site:
        say(f"no-site: {census.reason}")
        return None
    verdict = Verdict("true", reason=f"proof:no-site: {census.reason} (a census of every operator, "
                                  f"call and raise the program can run)",
                   component="proof:no-site", detail={"classes": list(classes)})
    return bind_route(verdict, "no-site", source, program, classes)


def prove(program: str, classes, *, python: str | None = None, budget: float = 60.0, log=None):
    """A proof that no execution of ``program`` raises an uncaught ``classes``, or None."""
    from tharkun.sv.engine import Verdict, read_source

    say = log or (lambda msg: None)
    static = prove_static(program, classes, log=log)
    if static is not None:
        return static
    try:
        from tharkun.solve.absint import prove as absint_prove

        source = read_source(program)
        r = absint_prove(source, tuple(classes), filename=program, time_budget=max(1.0, budget))
    except Exception as e:  # a proof stage never decides by crashing
        say(f"absint: internal error {type(e).__name__}: {e}")
        return None
    if r.verdict != "TRUE":
        say(f"absint: UNKNOWN ({r.reason}) in {r.seconds:.2f}s")
        return None
    lem = f"; lemmas {', '.join(r.lemmas)}" if r.lemmas else ""
    say(f"absint: TRUE, {len(r.sites)} property checks proven in {r.seconds:.2f}s{lem}")
    return Verdict("true", reason=f"absint proof: {len(r.sites)} checks{lem}", component=r.component,
                   detail={"sites": r.sites, "lemmas": r.lemmas})


PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_probe.py")
HASH_SEEDS = (0, 1)
MAX_RUNS = 400
MAX_FINITE = 64



def run_probe(python: str, program: str, vector: list, loops: list, quotients: list, *,
              timeout: float, hashseed: int = 0, memory_bytes: int | None = None, output_files=False, files=None) -> dict:
    """One probe run (``_probe.py``); a dict with at least ``status``."""
    from tharkun.sv.engine import _cap_memory
    with tempfile.TemporaryDirectory(prefix="tharkun-probe-") as tmp:
        req = os.path.join(tmp, "req.json")
        out = os.path.join(tmp, "out.json")
        with open(req, "w") as fh:
            json.dump({"program": os.path.abspath(program), "vector": vector, "out": out,
                       "loops": loops, "quotients": quotients, "output_files": output_files, "scratch": tmp,
                       "files": list(files) if files is not None else [os.path.abspath(program)]}, fh)
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONSTARTUP")}
        env["PYTHONHASHSEED"] = str(hashseed)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            proc = subprocess.run([python, PROBE, req], cwd=tmp, env=env, capture_output=True,
                                  text=True, timeout=timeout, preexec_fn=_cap_memory(memory_bytes))
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        if not os.path.exists(out):
            return {"status": "crashed", "msg": (proc.stderr or "")[-500:]}
        with open(out) as fh:
            return json.load(fh)


def _site_key(rec: dict, program: str) -> tuple[int, int] | None:
    try:
        file, line, col = rec["site"].rsplit(":", 2)
        if file != os.path.basename(program):
            return None
        return int(line), int(col)
    except (KeyError, ValueError, AttributeError):
        return None


def _strip(rec: dict) -> dict:
    return {k: v for k, v in rec.items() if k != "site"}


def _read(program: str, say):
    from tharkun.sv.engine import read_source
    try:
        return read_source(program)
    except (OSError, SyntaxError, UnicodeDecodeError) as e:
        say(f"prove: cannot read the program ({e})")
        return None


def prove_dynamic(program: str, classes, *, python: str, remaining, jobs: int = 1, log=None,
                  run_timeout: float = 120.0, memory_bytes: int | None = None, knowledge=None,
                  files=None):
    """The proofs that run the program: a TRUE/FALSE ``Verdict`` or ``None``.  Today
    ``proof:stationary``.  ``remaining()`` is the CPU budget left, which the caller keeps.

    ``knowledge`` (a :class:`~tharkun.knowledge.KnowledgeBase`, [internal ref] step 2 / bd
    [internal ref]'s ``loop_trip_count`` fact kind), when given, gets one
    UNCERTIFIED ``loop_trip_count`` Fact per probe run's own loop-head instance -- the
    stationary probe (``_probe.py``) already OBSERVES each trip loop's actual trip count
    for the vector it just ran; this only records that observation, never a new decision
    by itself, whether or not the stationarity PROOF itself ultimately succeeds."""
    say = log or (lambda msg: None)
    original = _read(program, say)
    if original is None:
        return None
    # Determinism is a whole-program property; slicing keeps only what one
    # exception class error-site anchor reaches; it must not blind the
    # determinism census to order-sensitivity elsewhere in the program ([internal ref]).
    det = determinism_census(original, program, declared_files=files or ())
    if not det.ok:
        # "deterministic between draws" is what makes one run per stand-in vector enough
        say(f"prove: stationary: determinism not certified ({det.reason})")
        return None
    from tharkun.solve.slicing import analysis_slice
    source = analysis_slice(original, classes, filename=program).source
    plan_inputs = call_target_evidence_in(knowledge)
    try:
        route_parents = tuple(item.identity for item in plan_inputs)
    except Exception:  # metadata capture cannot prevent the original proof
        route_parents = None
    plan = plan_draws(source, program, [item.value for item in plan_inputs])
    if not plan.ok:
        say(f"prove: stationary: no plan ({plan.reason})")
        return None
    if plan.output_files:
        from tharkun.solve.absint.interp import PROPERTY_EXCS
        if not set(classes) <= PROPERTY_EXCS:
            say("prove: stationary: output environmental exceptions are in the property")
            return None
    verdict = _stationary(program, classes, plan, python=python, remaining=remaining,
                          run_timeout=run_timeout, memory_bytes=memory_bytes, jobs=jobs, say=say,
                          knowledge=knowledge, files=files)
    if route_parents is None:
        return verdict
    return bind_route(verdict, "stationary", original, program, classes,
                      parents=route_parents)


def _write_loop_trip_count(knowledge, res: dict) -> None:
    """[internal ref] step 2 ([internal ref], W57's [internal ref] ask): one uncertified Fact per
    loop-head instance THIS probe run actually observed, mapping the loop back to the
    trip count it took AND the draw vector that produced it -- "the forward runs already
    see it" (the stationary probe tracks loop heads for its own proof already; this only
    records what it saw). Never certified: an observed trip count on one vector says
    nothing about every OTHER vector, and this never gates a verdict either way."""
    if knowledge is None:
        return
    vector = [{k: v for k, v in d.items() if k != "site"} for d in res.get("draws", [])]
    for inst in res.get("instances", []):
        trip = inst.get("trip")
        if trip is None:
            continue
        line, col = inst["loop"]
        knowledge.add(Fact(kind="loop_trip_count", subject=f"loop@{line}:{col}",
                           value=f"trip={trip} draws={vector}",
                           source="prove:stationary", certified=False))


def _stationary(program, classes, plan, *, python, remaining, run_timeout, memory_bytes, jobs, say,
                knowledge=None, files=None):
    from tharkun.sv.engine import Outcome, Verdict, _replayed, _resource_exhausted, _witness, run_once
    files = files if files is not None else [os.path.abspath(program)]
    loops = [{"line": lp.line, "col": lp.col, "dead": list(lp.dead), "m": max(lp.offset, 1)}
             for lp in plan.loops]
    m_of = {(lp.line, lp.col): max(lp.offset, 1) for lp in plan.loops}
    sites = {(d.line, d.col): d for d in plan.draws}
    quotients = [[d.line, d.col] for d in plan.draws if d.kind == "quotient"]
    child_memory = memory_bytes // jobs if memory_bytes else None
    runs = 0
    checked = 0

    def domain(rec: dict, index: int, res: dict):
        """The values of the draw at ``index`` that stand for all of its values."""
        kind = rec.get("kind")
        if kind == "bool":
            return [False, True]
        if kind == "rindex":
            n = int(rec.get("n") or 0)
            return list(range(n)) if 0 < n <= MAX_FINITE else None
        if kind == "str":
            # [internal ref] follow-up: a str draw plan_draws classified "str_truthy" (used
            # ONLY for truthiness and as open()'s own path -- see drawflow.py's own
            # comment) collapses to exactly 2 equivalence classes. Mirrors "bool" ->
            # [False, True] above -- BOTH the empty and non-empty case are covered
            # explicitly (owner's [internal ref] lesson: truthiness is value-dependent, never
            # a single stand-in value).
            site = sites.get(_site_key(rec, program))
            if site is not None and site.kind == "str_truthy":
                return ["", "x"]
            return None
        if kind != "int":
            return None
        site = sites.get(_site_key(rec, program))
        if site is None:
            return None
        if site.kind == "trip":
            m = max(site.offset, 1)
            return list(range(0, m + 2 - site.offset))
        if site.kind == "quotient":
            mod = res.get("moduli", {}).get(str(index))
            if mod is None:
                # the reduction never ran: the run ended first, the same for every value
                return [0] if index == len(res.get("draws", [])) - 1 else None
            tname, k = mod
            if tname != "int" or not isinstance(k, int) or abs(k) > MAX_FINITE:
                return None
            return list(range(abs(k))) or [0]
        return None

    def check(res: dict) -> str | None:
        """Why this run does not support the proof, or None."""
        st = res.get("status")
        if st not in ("clean", "exit", "raised"):
            return f"run {st}: {res.get('msg', '')[:120]}"
        if res.get("flags"):
            return f"uncontrolled nondeterminism {res['flags']}"
        for inst in res.get("instances", []):
            if "refused" in inst:
                return f"snapshot refused at loop {inst['loop']}: {inst['refused']}"
            m = m_of.get(tuple(inst["loop"]))
            if m is not None and inst.get("trip") == m + 1 and inst.get("heads", 0) >= m + 2:
                if inst.get("equal") is not True:
                    return f"loop {inst['loop']} is not stationary after {m} iteration(s)"
        return None

    work = [[]]
    done_vectors = set()
    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        while work:
            batch, work = work[:max(1, jobs // len(HASH_SEEDS))], work[max(1, jobs // len(HASH_SEEDS)):]
            if runs + len(batch) * len(HASH_SEEDS) > MAX_RUNS:
                say(f"prove: stationary: more than {MAX_RUNS} runs needed")
                return None
            left = remaining()
            if left <= 1:
                say("prove: stationary: out of budget")
                return None
            # the budget is CPU seconds of all children together; a proof run may be longer
            # than a search run (``run_timeout``), so only the budget share bounds it
            timeout = max(1.0, left / (len(batch) * len(HASH_SEEDS)))
            jobs_list = [(vec, seed) for vec in batch for seed in HASH_SEEDS]
            outs = list(pool.map(lambda vs: run_probe(python, program, vs[0], loops, quotients,
                                                      timeout=timeout, hashseed=vs[1],
                                                      memory_bytes=child_memory, output_files=plan.output_files, files=files), jobs_list))
            runs += len(jobs_list)
            by_vec = {}
            for (vec, seed), res in zip(jobs_list, outs):
                by_vec.setdefault(json.dumps(vec), []).append(res)
            for vec in batch:
                results = by_vec[json.dumps(vec)]
                base = results[0]
                for res in results:
                    if _resource_exhausted(Outcome(res.get("status"), mro=res.get("mro", []),
                                                   flags=res.get("flags", []))):
                        say("prove: stationary: resource-limited or unobserved run")
                        return None
                    _write_loop_trip_count(knowledge, res)
                    exc = res.get("exc") if res.get("status") == "raised" else None
                    if exc and any(c in res.get("mro", []) for c in classes):
                        # a real violation: confirm it on the plain child, then report it. The
                        # confirmation run must actually CONSUME wvec ([internal ref] [internal ref]):
                        # the probe run that found ``wvec`` is instrumented and can draw
                        # differently than the plain child replaying the same vector, and
                        # ``eregion_sv.replay.validate`` (the actual grader) refuses a witness
                        # whose run does not, even if it happens to also raise.
                        wvec = [_strip(d) for d in res.get("draws", [])]
                        out = run_once(python, program, wvec, timeout=max(1.0, min(run_timeout,
                                       remaining())), memory_bytes=child_memory, files=files)
                        hit = out.violated(classes) if _replayed(wvec, out) else None
                        if hit:
                            v = Verdict("false", hit, witness=_witness(program, "", hit, out))
                            v.component, v.detail = "proof:stationary-run", {"vector": wvec}
                            return v
                        say("prove: stationary: a probe violation did not replay")
                        return None
                    why = check(res)
                    if why:
                        say(f"prove: stationary: {why}")
                        return None
                    checked += sum(1 for inst in res.get("instances", [])
                                   if inst.get("equal") is True)
                draws0 = [_strip(d) for d in base.get("draws", [])]
                for res in results[1:]:
                    if [_strip(d) for d in res.get("draws", [])] != draws0 or \
                            (res.get("status"), res.get("exc")) != (base.get("status"), base.get("exc")):
                        say("prove: stationary: the run depends on the hash seed")
                        return None
                for j in range(len(vec), len(draws0)):
                    rec = base["draws"][j]
                    dom = domain(rec, j, base)
                    if dom is None:
                        say(f"prove: stationary: the draw at {rec.get('site')} "
                            f"({rec.get('kind')}) has no finite stand-in domain")
                        return None
                    executed = draws0[j]["value"]
                    for v in dom:
                        if v == executed or (rec["kind"] == "bool" and bool(v) == bool(executed)):
                            continue
                        child = draws0[:j] + [dict(draws0[j], value=v)]
                        key = json.dumps(child)
                        if key not in done_vectors:
                            done_vectors.add(key)
                            work.append(child)
    facts = [f"loop at line {lp.line}: trip count {lp.offset} + abs(draw), "
             f"state at head {max(lp.offset, 1) + 1} == head {max(lp.offset, 1) + 2}"
             for lp in plan.loops]
    for d in plan.draws:
        if d.kind == "quotient":
            facts.append(f"draw at line {d.line}: reaches the program only as abs(d) % M")
    v = Verdict("true", reason=f"proof:stationary: {runs} probe runs, {checked} stationary "
                               f"loop instance(s); " + "; ".join(facts))
    v.component, v.runs = "proof:stationary", runs
    v.detail = {"runs": runs, "stationary_instances": checked, "facts": facts,
                "loops": [[lp.line, lp.col, lp.offset] for lp in plan.loops],
                "draws": [[d.line, d.col, d.kind, d.offset] for d in plan.draws]}
    return v
