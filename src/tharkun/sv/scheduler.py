"""The order and budget logic ``sv/engine.py::verify`` calls (Step 1, [internal ref]) over the live store (step 3a).

``run`` is what ``verify`` ran inline before the split: the stages named in its own docstring below
(static, symbolic, execution, facts, portfolio, enumeration), and the closures that exist only to
serve them (the fresh-vs-forked child runner, the FALSE/TRUE finalizers).  Every reference to an
``engine.py`` name goes through the ``_engine`` module object (an attribute lookup at call time, not a
bound name), so a caller that monkeypatches ``tharkun.sv.engine.<anything>`` -- ``run_once``,
``_static_proof``, ``ThreadPoolExecutor``, or any other symbol ``verify`` used to close over -- reaches
this code exactly the way it reached ``verify`` before the move.  ``engine.py::verify`` keeps the
one-time setup this function's inputs already assume (reading the property, building
``classes``/``specification``) and is the only caller (docs/SV.md "3a").

Step 3a (built; docs/SV.md "3a: one scheduler") made two objects out of what used to be
``nonlocal`` closures and side channels:

* ``sv/ledger.py::Ledger`` -- the ONE CPU-second ledger every stage spends from;
* ``sv/obligation_table.py::ObligationTable`` -- the obligation table, kept in a
  ``tharkun.knowledge.KnowledgeBase``.  Components WRITE what they learn into it and READ their inputs
  from it: absint's site records reach the concolic stages and the seed stages through it
  (``Context.table``), a concolic stage's per-line outcomes reach the narrowed retry through it, and the
  first run's observed ``class_hints`` re-open absint through it -- no out-parameter, no
  ``Context.solved``/``Context.records`` field, no re-run keyed on a local variable.
  ``certified=True`` is written ONLY by ``ObligationTable.discharge`` (after a whole-program proof) and
  ``ObligationTable.violate`` (after a confirmed replay); nothing else may write it (MG's review bar).
  The per-run closures stay plain objects, since a CPU-second count or a fork-server pool has no
  natural home in a store.
"""

from __future__ import annotations

from tharkun.facts import Fact
from tharkun.knowledge import KnowledgeBase
from tharkun.sv import engine as _engine
from tharkun.sv.ledger import Ledger
from tharkun.sv.obligation_table import CLASS_HINTS, ObligationTable


#: the share of what remains each ``Stage.deepen`` round may spend ([internal ref]): geometric rounds, so the
#: whole ledger is reachable and a fuzzer that restarts each round keeps its corpus
DEEPEN_SHARE = 0.5


#: the import graph the SV component reads is at most this many modules (a task is one file plus the odd
#: sibling; a graph over the cap is incomplete, so nothing in it is proven).  Measured: a 152-module package
#: costs 5-19 s per call, which is not "no budget".
STATIC_FACTS_MAX_MODULES = 16


def publish_module_namespace(table: ObligationTable, program: str, say=lambda msg: None) -> None:
    """The component "static facts (module namespace)" (docs/SV.md "Component contracts"): publish what a
    static read of the task's import graph says about its module-level names -- which are never rebound
    (``module_constant``), which compiled regexes exist (``regex_pattern``) and what every call site of the
    entry module calls (``call_target``, [internal ref]).

    ``tharkun.solve.module_constants`` and ``regex_globals`` read source with ``ast`` and execute nothing,
    so this has no budget (the graph is capped at ``STATIC_FACTS_MAX_MODULES`` modules) and decides nothing: the
    kinds are uncertified hints.  The records are the SAME
    ones Eregion's synthesis asks for through ``tharkun.solve`` (a name is baked into a synthesized inverter
    only when its record is ``proven``).  An analysis that fails publishes nothing (a producer that
    crashes proves nothing); the run goes on."""
    from tharkun.solve.call_targets import call_targets
    from tharkun.solve.module_namespace import module_constants, regex_globals
    try:
        constants = module_constants(program, max_modules=STATIC_FACTS_MAX_MODULES)
        regexes = regex_globals(program, max_modules=STATIC_FACTS_MAX_MODULES)
    except Exception as e:  # noqa: BLE001 -- a hint producer that crashes publishes nothing
        say(f"static facts: not published ({type(e).__name__})")
        return
    table.publish_module_namespace(constants, regexes, producer="module_namespace")
    try:
        targets = call_targets(program, constants, max_modules=STATIC_FACTS_MAX_MODULES)
    except Exception as e:  # noqa: BLE001 -- the namespace records above stand; a crashed producer proves nothing
        say(f"static facts: call targets not published ({type(e).__name__})")
        return
    table.publish_call_targets(targets, producer="call_targets")


def _flip_draw(entry: dict):
    """A DIFFERENT value of ``entry``'s own kind, same site/position -- ``None`` when the
    kind has no well-defined "other" value here (``str`` past [internal ref]'s cap is skipped, not
    worth the CPU). Purely a differencing probe ([internal ref]-followup, MAIN 2026-09-28
    "input_sensitivity"): the exact alternate does not matter, only that it differs, so a
    run replayed with it isolates whether THIS ONE draw position moves the outcome."""
    kind, value = entry.get("kind"), entry.get("value")
    if kind == "bool":
        flipped = not value
    elif kind == "int":
        flipped = value + 1
    elif kind in ("float", "rfloat"):
        import math
        v = float.fromhex(value) if isinstance(value, str) else float(value)
        flipped = (v + 1.0) if math.isfinite(v) else 0.0
        if kind == "rfloat" and not (0.0 <= flipped < 1.0):
            flipped = 1.0 - v if 0.0 <= v < 1.0 else 0.5
        flipped = float(flipped).hex()
    elif kind == "rindex":
        n = entry.get("n") or 1
        flipped = (int(value) + 1) % max(n, 1)
    elif kind == "str":
        flipped = (value + "x") if len(value) < 256 else None
    else:
        flipped = None
    if flipped is None or flipped == value:
        return None
    return dict(entry, value=flipped)


def _write_input_sensitivity(knowledge: KnowledgeBase, program: str, python: str, out, run,
                             *, max_positions: int = 2) -> None:
    """[internal ref]-followup (MAIN 2026-09-28, [internal ref]): "which draw position(s)
    move which guard" -- a differencing pass over the deterministic FIRST run every task
    already performs (``run()``'s own ``first = run([], ...)``), the cheapest source
    available: for each of the first ``max_positions`` draws, replay with just that one
    position flipped (:func:`_flip_draw`) and see whether the observed outcome (status,
    exception, raise site) changes. A change is recorded as an UNCERTIFIED hint --
    ``input_sensitivity`` never gates a verdict, only tells a later component (Eregion
    [internal ref], the slicer, concolic) which draw a guard actually reacts to, empirically.
    Bounded to a small, fixed number of extra fresh replays (never proportional to the
    program's own draw count), and callers only invoke this when real search budget
    remains, so it never trades away a verdict for a hint."""
    if not out.draws:
        return
    baseline = (out.status, out.exc, (out.site or {}).get("line"))
    for i, entry in enumerate(out.draws[:max_positions]):
        flipped = _flip_draw(entry)
        if flipped is None:
            continue
        vector = [dict(e) for e in out.draws[:i]] + [flipped]
        alt = run(vector)
        if (alt.status, alt.exc, (alt.site or {}).get("line")) != baseline:
            site = entry.get("site", "?")
            knowledge.add(Fact(
                kind="input_sensitivity", subject=f"draw{i}@{site}",
                value=f"flip: {baseline[0]}/{baseline[1]} -> {alt.status}/{alt.exc}",
                source="scheduler:differencing", certified=False))


def run(program: str, classes, *, prop_name: str = "", python: str | None = None,
           budget_seconds: float = 600.0, run_timeout: float = 120.0, max_runs: int = 100000,
           run_memory_bytes: int | None = None, jobs: int | None = None,
           constants: bool = True, deepen: bool = True, concolic: bool = True,
           concolic_share: float = 0.4, prove: bool = True, canary_seconds: float = 60.0,
           early_stop_seconds: float = 60.0, stages=None, log=None, lazy_vectors: int = 4, lazy_seconds: float = 2.0,
           task: str | None = None, specification: str | None = None, witness_dir=None,
           language_version: str | None = None,
           checker_seconds: float = 3.0, cpu_time_only: bool = False,
           store_witness: bool = True) -> _engine.Verdict:
    """Decide ``! uncaught(E)`` for every ``E`` in ``classes`` on ``program``.

    ``budget_seconds`` is CPU time of the verifier and its children (what BenchExec
    limits), also used as a wall-clock cap unless ``cpu_time_only`` is set ([internal ref] test
    seam -- see ``engine.py::verify``'s docstring for the full rationale; production
    code never sets it).  ``task`` names the task's own ``.yml`` (declared/required
    files, [internal ref]'s shared I/O model) -- default ``None`` derives it from ``program``
    (``tharkun.solve.environment.TaskEnvironment.from_task``).  ``jobs`` children
    run side by side ([internal ref];
    default :func:`default_jobs`).  ``run_memory_bytes`` is the address space the child
    runs may use together ([internal ref]): each of the ``jobs`` concurrent children gets an equal
    share; a run that exhausts it ends with ``MemoryError`` and counts as inconclusive,
    never as TRUE.  ``constants`` / ``deepen`` switch off the program-constant candidates
    ([internal ref]) and the unbounded candidate tail ([internal ref]) -- for measuring what each buys.
    ``concolic`` ([internal ref] slice 2, :mod:`tharkun.sv.concolic`) first SOLVES for draws that
    reach each explicit violation site, with at most ``concolic_share`` of the budget;
    every solved vector is replayed like any other run, and enumeration follows.
    ``stages`` names the portfolio stages to run (:mod:`tharkun.sv.stages`; default
    ``stages.DEFAULT``); ``concolic=False`` drops the concolic stages.

    The stages run cheapest-DECIDING first ([internal ref] slice 3, owner ruling 2026-09-24):

    1. **static** (``prove``): :func:`_static_proof` -- census and abstract-interpretation
       proofs; nothing runs;
    1b. **exact product proof** (``prove``, only when stage 1 above did not already decide
        TRUE): :func:`tharkun.sv.product_proof.prove` -- a GENERAL, non-mutant-specific Z3
        proof of every path of a loop-free program from its own entry ([internal ref], [internal ref],
        owner ruling 2026-09-29); can decide TRUE (stored as ``proof``, same canary as stage 1)
        or FALSE (a Z3 model replay-confirmed on the unmodified program, through the same
        :func:`found` every other FALSE goes through); outside its scope it decides nothing;
    2. **symbolic** (``concolic``): one symbolic execution of the whole harness toward
       every open site; each solved vector is replayed (a FALSE needs a real violating run);
    3. **execution**: the default run (and, with no draws, a second under another hash
       seed); then, only if the program drew something and the run raised no
       nondeterminism flag, 3b: :func:`tharkun.sv.fact_proof.prove_from_facts` ([internal ref]
       step 3 -- installed fact producers such as Eregion's propose candidate facts, the
       store holds them, and TRUE needs every current obligation independently checked
       in this invocation; same canary as any proof); then
       ``tharkun.sv.prove.prove_dynamic`` ([internal ref]) when it is merged;
    4. **the portfolio** (:mod:`tharkun.sv.stages`): seed stages, then the per-site concolic
       drive on the sites stage 2 left open (at most ``concolic_share`` of what remains);
    5. **enumeration**, last.  It is CAPPED when it can only confirm: after a proof it
       is a canary of ``min(canary_seconds, budget/10)`` CPU seconds (a FALSE it finds
       wins, and names the proof that was unsound; ``canary_seconds=0`` is no canary: the proof's
       TRUE is returned the moment it is decided, :func:`start_canary`); when stage 2 solved every site to
       "no modelled run fires it" it runs ``min(early_stop_seconds, budget/10)`` and
       ends UNKNOWN early instead of spending the budget;
    6. **the remainder** ([internal ref]): enumeration draining its candidates never ends the run by itself.
       The stages that can keep searching an unbounded draw domain (``Stage.deepen``) get what is
       left, in rounds, until the goal (``ledger.goal``) is met or ``ledger.stopped()``."""
    say = log or (lambda msg: None)
    ledger = Ledger(budget_seconds, cpu_time_only=cpu_time_only, say=say)   # the ONE CPU-second ledger
    python = python or _engine.sys.executable
    environment, reason = _engine.admit_task(program, task=task, python=python, language_version=language_version)
    if reason:
        return _engine.Verdict("unknown", reason=reason)
    files = environment.files
    from tharkun.solve.census import obligation_census
    classes = tuple(classes)
    proof_source = _engine.read_source(program)
    api_reason = _engine.verifier_api_reason(proof_source)
    if api_reason:
        return _engine.Verdict("unknown", reason=api_reason)
    specification = specification if specification is not None else "".join(
        f"CHECK( init(main()), ! uncaught({exc}))\n" for exc in classes)
    from tharkun.sv.spec import parse_property
    if set(parse_property(specification)[1]) != set(classes):
        raise ValueError("specification and selected exception classes differ")
    native_reason = _engine.native_import_reason(proof_source, program)
    if native_reason:
        say(f"native-semantics: {native_reason}")
        return _engine.Verdict("unknown", reason=f"native-semantics: {native_reason}")
    # A program that binds a property exception name itself gets no TRUE (``engine.shadowed_property_names``).
    shadowed = set(_engine.shadowed_property_names(proof_source, classes))
    local_modules = _engine.local_import_files(program)
    if local_modules is None:
        shadowed.update(classes)       # unreadable or oversized local import graph: assume the worst
    for other in dict.fromkeys([*files[1:], *(local_modules or ())]):
        if str(other).endswith(".py"):
            try:
                shadowed.update(_engine.shadowed_property_names(_engine.read_source(other), classes))
            except (OSError, UnicodeDecodeError, ValueError):
                shadowed.update(classes)   # unreadable declared or imported source: assume the worst
    shadowed = tuple(sorted(shadowed))
    obligations = obligation_census(proof_source, classes, program, declared_files=files)
    # docs/SV.md "3a": the obligation table, in a KnowledgeBase; components write what they learn
    # into it and read their inputs from it (sv/obligation_table.py).
    table = ObligationTable(obligations)
    knowledge = table.kb
    publish_module_namespace(table, program, say)
    invariant_locations = {(o.invariant.file_name, o.invariant.line, o.invariant.column,
                            o.invariant.function) for o in obligations.obligations}
    from tharkun.solve.sitekey import site_keys
    try:
        proof_tree = _engine.ast.parse(proof_source, program)
        source_sites = site_keys(proof_tree, proof_source, program)
        invariant_locations.update(site.location() for node, site in source_sites.items()
                                   if isinstance(node, _engine.ast.While))
    except SyntaxError:
        pass
    try:
        import hashlib
        location = source_sites[proof_tree.body[0]].location() if proof_tree.body else (program, 1, 1, "<module>")
        table.bind_source(source_sites, source_sha256=hashlib.sha256(proof_source.encode("utf-8")).hexdigest(),
                          classes=classes, location=location)
    except Exception:  # source context is additive; failure leaves the old proof path available
        pass
    jobs = jobs or _engine.default_jobs()
    child_memory = run_memory_bytes // jobs if run_memory_bytes else None
    servers = ledger.servers         # live fork servers: their CPU is not reaped yet (the ledger counts it)
    remaining = ledger.remaining
    runs = 0
    #: the module names ANY run so far found absent ([internal ref]): absence is a fact of the environment, not of a run, so the union over
    #: runs is what the determinism census may take (``_engine._certify_determinism``)
    absent_seen: set = set()

    def run(vec, hashseed=0, share=1, observe_fields=None):
        # ``share`` children run at once and the budget is their CPU together, so each may use
        # only its share of what remains ([internal ref]) -- and of what a cap leaves.
        left = ledger.allowance()
        is_canary = canary is not None and canary_seconds > 0
        seconds = max(1.0, left / share) if is_canary else left / share
        with ledger.reserve(min(run_timeout, seconds), canary=is_canary) as granted:
            if granted <= 0:
                out = _engine.Outcome("timeout", msg="aggregate CPU allowance exhausted")
            else:
                if canary is not None and canary_seconds > 0:
                    canary["attempted"] = True
                out = _engine.run_once(python, program, vec, timeout=granted,
                           hashseed=hashseed, memory_bytes=child_memory, files=files,
                           observe_fields=observe_fields)
        if canary is not None and canary_seconds > 0:
            if out.status in ("timeout", "crashed") or _engine._resource_exhausted(out):
                canary.update(cut=True, reason=out.exc or out.status)
        absent_seen.update(out.absent_imports)
        return out

    def done(v: _engine.Verdict) -> _engine.Verdict:
        if v.verdict == "true" and shadowed:
            # withheld before any witness is written; FALSE (a built-in instance escaped) stays definitive
            _engine.withhold_true(v, shadowed)
            say(f"withheld: {v.reason}")
        if task is not None or language_version is not None:
            v.detail.update(language_version=environment.language_version if task and task.endswith((".yml", ".yaml"))
                            else language_version or environment.language_version,
                            verifier_runtime=f"{_engine.sys.implementation.name} {_engine.sys.version.split()[0]}",
                            target_python=python)
        if v.verdict == "true":
            try:
                from tharkun.sv.proof import finish
                if _engine.read_source(program) != proof_source:
                    raise ValueError("source changed during verification")
                try:
                    from tharkun.sv.store_witness import record_route
                    record_route(table, v)
                except Exception:  # additive evidence cannot affect the deciding verdict
                    pass
                v = finish(v, proof_source, program, specification, obligations,
                           directory=witness_dir, seconds=min(checker_seconds, max(0, remaining())), python=python,
                           knowledge=knowledge, table=table,
                           shadow_seconds=min(1.0, 0.05 * max(0, remaining())) if store_witness else None)
                # Tagged "proof: " so the SV harness's diagnostic-line grammar
                # recognizes it and does not mistake "checked=False" for a FALSE verdict.
                say(f"proof: checked={v.detail['proof_checked']} "
                    f"open_obligations={len(v.detail['open_obligations'])}")
                if v.detail.get("witness_write_error"):
                    # A write failure must never change the verdict ([internal ref] review) --
                    # only note it, tagged the same way.
                    say(f"proof: witness not written: {v.detail['witness_write_error']}")
            except Exception as e:
                v.verdict = "unknown"
                v.reason = f"proof checking/export failed: {type(e).__name__}: {e}"
                v.detail.update(proof_checked=False, proof_reason=v.reason,
                                open_obligations=[o.invariant.entry() for o in obligations.obligations])
            if v.verdict == "true":  # the try block above may have downgraded it
                table.discharge(classes, v.component)
        if canary is not None:
            status = dict(canary)
            status["completed"] = status["attempted"] and not status["cut"]
            if status["completed"]:
                status["reason"] = "violation" if v.verdict == "false" else "no violation"
            v.detail["canary"] = status
            if v is proof and canary_seconds > 0:
                if status["cut"]:
                    v.reason += f"; canary cut: {status['reason']}"
                elif not status["attempted"]:
                    v.reason += "; canary not attempted"
                else:
                    v.reason += f"; canary: no violation in {runs} runs"
        v.runs, v.seconds = runs, _engine.time.monotonic() - ledger.started
        return v

    proof = None                     # a TRUE the canary still has to survive
    canary = None

    capped = ledger.capped

    def start_canary(proof_: _engine.Verdict, why: str) -> _engine.Verdict | None:
        """A proof's canary begins: the search that follows may only confirm what the proof says (docs/SV.md "Static
        proofs before the search").  ``canary_seconds <= 0`` is NO canary: the proof's own verdict, at once.  ``None``:
        the canary is capped and the run goes on."""
        nonlocal canary
        canary = dict(attempted=False, completed=False, cut=False,
                      reason="disabled" if canary_seconds <= 0 else "no runs")
        if canary_seconds <= 0:
            say(f"no canary for {proof_.component} (canary_seconds={canary_seconds:g})")
            return done(proof_)
        capped(canary_seconds, why)
        return None

    def found(v: _engine.Verdict, *, fresh: bool = False) -> _engine.Verdict:
        """A FALSE: real, whatever a proof said -- and then that proof was unsound.

        [internal ref]: a witness only pins what the engine controls (the draw vector). If the
        violation also depends on something a validator's OWN replay cannot reproduce --
        an object address, a default hash, set/dict order over objects -- the same
        vector can raise here and not there. ``_engine._certify_determinism`` (the
        [internal ref] census) proves address-independence for the ordinary case for free; when
        it cannot, the witness is trusted only after ``_engine._REPLAY_CONFIRM_K``
        independent fresh replays ALL reproduce the exact same exception (see that
        constant's docstring for why, and for why that is the strongest available
        claim for an address-dependent failure, not a certainty).

        MG (PR 97 review): the k-of-k loop must vary ``hashseed`` across its own
        replays, never repeat the default 0 -- a violation whose value depends on
        STRING/BYTES hash order (``next(iter(a_str_set))``, ``list(a_str_set)``) is
        perfectly reproducible k/k times under a FIXED hashseed (CPython's seeded
        string hash is a pure function of (content, PYTHONHASHSEED)), even though a
        validator inherits whatever hashseed its own environment gives it -- usually
        NOT 0. Each loop iteration below replays at ``hashseed=confirmations`` (1, 2,
        3, ... -- always non-zero, always different from the last), so a hash-order-
        dependent witness almost certainly meets a seed where its order differs and
        fails a confirmation, the same way an address-dependent one almost certainly
        meets an ASLR offset where it does. The discovery run and the (pre-existing,
        unrelated) fresh-replay confirm just above both stay at the default hashseed
        0 -- unchanged, since they are not this loop and [internal ref]'s fix already covers
        the address-dependent case through them.

        Every replay here also has to CONSUME the vector it was given, not merely end in
        the right exception ([internal ref], [internal ref]): ``_engine._replayed`` is the same
        length/kind/value check ``eregion_sv.replay.validate`` (the actual grader) makes,
        so this self-check is never weaker than what a validator would accept."""
        nonlocal runs
        confirmations = 1                 # the discovery run itself is the first trial
        if not fresh:
            # Fork-derived discoveries must reproduce in an independent fresh interpreter.
            confirmed = run(v.witness['vector'])
            runs += 1
            if not (_engine._replayed(v.witness['vector'], confirmed) and
                    confirmed.violated((v.exception,)) is not None):
                # MG (PR 97 review): never a bare TRUE/FALSE/UNKNOWN/ERROR token in a
                # reason -- __main__.py's "N runs in Ts; <reason>" diagnostic line does
                # not match eregion_sv.soundness._DIAGNOSTIC (it starts with a digit,
                # not "tharkun-sv: <phase>: "), so a reason containing one of those
                # words as a whole word makes _parse_verdict raise on a perfectly
                # valid run. Every reason below is worded to avoid all four, and
                # tests/test_sv_address_replay.py checks this directly.
                return done(_engine.Verdict('unknown', reason='violation candidate failed fresh replay'))
            v.witness = _engine._witness(program, prop_name, v.exception, confirmed)
            confirmations += 1
        det = _engine._certify_determinism(program, absent_imports=frozenset(absent_seen), files=files)
        det_env = None
        if det is not None and not det.ok:
            # [internal ref] hotfix v2 (MG): a SEPARATE, independent re-run with env reads excused --
            # never inferred from `det`'s own single reported reason (see
            # `_certify_determinism`'s and `determinism_census`'s docstrings for why: this
            # module reports only its FIRST refusal, so a program with an env read AND a
            # genuinely unconfirmable construct elsewhere must not be trusted just because
            # the env read happened to be reported first).
            det_env = _engine._certify_determinism(program, env_reads_ok=True,
                                                   absent_imports=frozenset(absent_seen), files=files)
        certified = det is not None and (det.ok or (det_env is not None and det_env.ok))
        if not certified:
            why = (det_env.reason if det_env is not None
                   else det.reason if det is not None else "the program cannot be read")
            k = _engine._REPLAY_CONFIRM_K
            while confirmations < k:
                if remaining() <= 0:
                    return done(_engine.Verdict('unknown', reason=(
                        f"violation witness not proven address-independent ({why}) and the budget "
                        f"ran out before {k}/{k} replay confirmations ({confirmations}/{k} so far)")))
                rep = run(v.witness['vector'], hashseed=confirmations)
                runs += 1
                if not (_engine._replayed(v.witness['vector'], rep) and
                        rep.violated((v.exception,)) is not None):
                    return done(_engine.Verdict('unknown', reason=(
                        f"violation witness not proven address-independent ({why}) and replay did "
                        f"not reproduce it on every attempt ({confirmations}/{k} confirmed)")))
                confirmations += 1
        if proof is not None:
            v.detail = dict(v.detail, unsound_proof=proof.component, proof_reason=proof.reason)
            # MG (PR 97 review): lowercase phase prefix (matches
            # eregion_sv.soundness._DIAGNOSTIC's "tharkun-sv: <phase>: " shape) and no
            # bare TRUE/FALSE/UNKNOWN/ERROR word -- see the reason-wording note above.
            say(f"certifier-unsound: {proof.component} said the property holds; a run violates {v.exception}")
        # A witness confirmed above (fresh already, the k-of-k address-independence
        # loop, or both): record it. This never decides FALSE a second time --
        # found() already has, above; this only writes the same confirmed fact into
        # the table ([internal ref] step 2).
        site = (v.witness or {}).get("site") or {}
        ledger.goal.record()
        table.violate(v.exception, site.get("line"), v.component, column=site.get("column"))
        return done(v)

    def replay_solved(solved) -> _engine.Verdict | None:
        nonlocal runs
        for cand in solved.candidates:
            if ledger.stopped():
                break
            out = run(cand.vector)
            runs += 1
            hit = out.violated(classes)
            if hit:
                say(f"run {runs}: {hit} at {out.site} (solved, {cand.component})")
                return found(_engine.Verdict("false", hit, witness=_engine._witness(program, prop_name, hit, out),
                                     component=cand.component, detail=cand.detail), fresh=True)
            say(f"run {runs}: solved vector did not violate ({out.status} {out.exc or ''})")
        return None

    # --- stage 1: static proofs, nothing runs ---
    if prove:
        proof = _engine._static_proof(program, classes, remaining, say, files,
                                      invariant_locations=invariant_locations, table=table)
        if proof is not None:
            v = start_canary(proof, f"canary for {proof.component}")
            if v is not None:
                return v

    # --- stage 1b: exact whole-program symbolic proof for loop-free "product"
    # programs ([internal ref]'s equivalent-mutant proving, general first step; owner ruling
    # 2026-09-29: a GENERAL Z3 proof over the whole inter-procedural call stack,
    # never a mutant-specific rewrite). Only tried when stage 1 above did not
    # already decide TRUE; see tharkun.sv.product_proof's module docstring for the
    # exact scope and soundness gate (unmodelled -> UNKNOWN, never a guess).
    if prove and proof is None and remaining() > 1:
        try:
            from tharkun.sv import product_proof as _product_proof
            pv = _product_proof.prove(proof_source, classes, program=program, python=python,
                                      prop_name=prop_name,
                                      run_timeout=max(1.0, min(30.0, remaining())),
                                      timeout_ms=_product_proof.z3_timeout_ms(remaining()))
        except Exception as e:  # noqa: BLE001 -- a proof stage never decides by crashing
            say(f"symbolic:product: internal error {type(e).__name__}: {e}")
            pv = None
        if pv is not None:
            runs += 1
            if pv.verdict == "false":
                say(f"run {runs}: {pv.exception} (symbolic:product model, replay-confirmed)")
                return found(pv, fresh=True)
            say(f"symbolic:product: TRUE, {len(pv.detail.get('sites', []))} check(s) proven")
            proof = pv
            v = start_canary(proof, f"canary for {proof.component}")
            if v is not None:
                return v

    # --- stage 2: one symbolic execution toward every open site ---
    from tharkun.sv import stages as _stages
    portfolio = _stages.select(stages, off=() if concolic else ("concolic",))
    symbolic = None                  # the store's view of stage 2, once it has run
    if any(st.name == "concolic" for st in portfolio) and remaining() > 1:
        from tharkun.sv import concolic as _concolic
        # the worker gets one child's share of the memory cap, like every run, so the
        # verifier and whatever it has started stay inside the task's limit together;
        # it reads absint's site records from the store and writes its outcome back to it
        solved = _concolic.solve(program, classes, budget_seconds=min(remaining() * concolic_share, 60.0),
                                 memory_bytes=child_memory, python=None, extra_sites=table.site_records(),
                                 drive=False, log=say, call_targets=table.call_targets(),
                                 proven_classes=table.proven_classes(), files=files)
        table.publish_solved("symbolic", solved, producer="concolic:symbolic")
        symbolic = table.solved("symbolic")
        v = replay_solved(solved)
        if v is not None:
            return v
        if symbolic.closed:
            capped(early_stop_seconds, "every site solved to no modelled violation")

    # --- stage 3: execution ---
    field_demands = None
    if prove and proof is None:
        try:
            from tharkun.solve.absint.contracts import demands
            field_demands = demands(_engine.read_source(program))
        except (OSError, UnicodeDecodeError, SyntaxError):
            pass
    first = run([], observe_fields=field_demands)
    runs += 1
    say(f"run {runs}: {first.status} {first.exc or ''} draws={len(first.draws)} flags={first.flags}")
    hit = first.violated(classes)
    if hit:
        return found(_engine.Verdict("false", hit, witness=_engine._witness(program, prop_name, hit, first),
                             component="deterministic" if not first.draws else "enum:small_first"),
                     fresh=True)
    if first.status in ("timeout", "crashed") or _engine._resource_exhausted(first):
        return done(proof or _engine.Verdict("unknown", reason=f"first run {first.status}"))
    if remaining() > run_timeout:
        # [internal ref]-followup: the cheapest source is a run already happening -- flip a couple
        # of `first`'s own draws and see whether the outcome moves (bounded, uncertified
        # hint only; skipped entirely under tight budget so it never competes with the
        # search itself for CPU).
        _write_input_sensitivity(knowledge, program, python, first, run)
    # what the run observed about the program's classes goes into the store; absint's second try
    # is keyed on NEW class_hints evidence there, not on a local variable (docs/SV.md "3a")
    table.publish_class_hints(first.class_hints, producer="execution")
    if prove and proof is None and remaining() > 1 and table.since("static", CLASS_HINTS):
        proof = _engine._static_proof(program, classes, remaining, say, files=files, table=table)
        if proof is not None:
            v = start_canary(proof, f"canary for observed candidate {proof.component}")
            if v is not None:
                return v
    if not first.draws and not first.flags:
        second = run([], hashseed=1)
        runs += 1
        hit2 = second.violated(classes)
        if hit2:
            return found(_engine.Verdict("false", hit2, witness=_engine._witness(program, prop_name, hit2, second),
                                 component="deterministic"), fresh=True)
        if second.status in ("timeout", "crashed") or _engine._resource_exhausted(second):
            # a cut second run says nothing about hash order (under load it used to be
            # reported as "hash-order dependent")
            return done(proof or _engine.Verdict("unknown", reason=f"second run {second.status}"))
        if (second.status, second.exc) != (first.status, first.exc):
            return done(proof or _engine.Verdict("unknown", reason="hash-order dependent"))
        if proof is not None:
            return done(proof)
        # Two hash seeds are samples; only the certificate proves determinism.
        det = _engine._certify_determinism(program, absent_imports=frozenset(absent_seen), files=files)
        class_only = False
        if det is not None and not det.ok:
            # A SEPARATE re-run (like ``env_reads_ok`` above), never read off ``det``'s single reported reason:
            # this TRUE is the observed exception class alone (the run's ``status``/``exc`` just compared
            # equal under two hash seeds), so a program in which nothing can observe an exception in flight
            # may have a KeyError whose text names an address, and heaps ordered by (number, identity-only
            # instance).  ``determinism_census`` states the exact scope and refuses on any observer.
            det_class = _engine._certify_determinism(program, absent_imports=frozenset(absent_seen), files=files,
                                                     class_only_exceptions=True)
            if det_class is not None and det_class.ok:
                det, class_only = det_class, True
        if det is not None and det.ok:
            verdict = _engine.Verdict("true", reason="deterministic: one execution (certified" +
                                (", exceptions judged by class" if class_only else "") + f": {det.reason}), "
                                "observed under two hash seeds",
                                component="deterministic", detail={"certificate": det.reason})
            try:
                from tharkun.sv.store_witness import bind_route
                bind_route(verdict, "determinism", proof_source, program, classes)
            except Exception:
                pass
            return done(verdict)
        why = det.reason if det is not None else "the program cannot be read"
        return done(_engine.Verdict("unknown", reason=f"one execution observed, but determinism is not "
                                              f"certified: {why}"))
    # --- stage 3b: certified facts from installed fact producers ([internal ref] step 3, the
    # Eregion bridge). Only after the first run (a violation found there is cheaper) and
    # only with no uncontrolled-nondeterminism flag -- the same gate prove_dynamic keeps.
    # Producers propose; sv/fact_proof.py certifies each proposal independently and
    # decides TRUE only when every current obligation is independently checked in this invocation.
    # With no producer installed this is one entry-point lookup and nothing else.
    if prove and proof is None and not first.flags and remaining() > 1:
        from tharkun.sv.fact_proof import prove_from_facts
        try:
            fp = prove_from_facts(program, classes, source=proof_source, specification=specification,
                                  table=table, python=python, remaining=remaining, say=say,
                                  checker_seconds=checker_seconds, memory_bytes=child_memory)
        except Exception as e:  # noqa: BLE001 -- a proof stage never decides by crashing
            say(f"facts: internal failure ({type(e).__name__})")
            fp = None
        if fp is not None:
            proof = fp
            v = start_canary(proof, f"canary for {proof.component}")
            if v is not None:
                return v
    if prove and proof is None and not first.flags and remaining() > 1:
        try:
            from tharkun.sv.prove import prove_dynamic
        except (ImportError, AttributeError):
            prove_dynamic = None
        if prove_dynamic is not None:
            try:
                dyn = prove_dynamic(program, classes, python=python, remaining=remaining, jobs=jobs,
                                    log=say, run_timeout=run_timeout, memory_bytes=run_memory_bytes,
                                    knowledge=knowledge, files=files)
            except Exception as e:  # noqa: BLE001
                say(f"prove_dynamic: crashed ({type(e).__name__}: {e})")
                dyn = None
            if dyn is not None:
                runs += dyn.runs
            if dyn is not None and dyn.verdict == "false":
                if dyn.witness is not None:
                    dyn.witness["property"] = prop_name
                return found(dyn, fresh=True)
            if dyn is not None and dyn.verdict == "true":
                proof = dyn
                v = start_canary(proof, f"canary for {proof.component}")
                if v is not None:
                    return v

    ctx = _stages.Context(program, tuple(classes), python=python, memory_bytes=child_memory,
                          first=first, log=say, table=table, files=files, ledger=ledger, task=task)
    exploratory_runs = 0
    available = _engine.queue.Queue()
    server_lock = _engine.threading.Lock()

    def get_server():
        if not servers:
            with server_lock:
                if not servers:
                    from tharkun.sv.forkserver import ForkServer
                    for _ in range(jobs):
                        server = ForkServer(python)
                        ledger.add_server(server)
                        available.put(server)
        return available.get()

    def explore(vec, share=1):
        nonlocal exploratory_runs
        with server_lock:
            exploratory_runs += 1
            is_lazy = exploratory_runs <= lazy_vectors and (_engine.time.monotonic() - ledger.started) < lazy_seconds
        if _engine.sys.platform != 'linux' or is_lazy:
            return run(vec, share=share)
        server = get_server()
        try:
            with ledger.reserve(min(run_timeout, ledger.allowance() / share)) as granted:
                if granted <= 0:
                    out = _engine.Outcome("timeout", msg="aggregate CPU allowance exhausted")
                else:
                    out = server.run(python, program, vec, timeout=granted,
                                     memory_bytes=child_memory, files=files)
            absent_seen.update(out.absent_imports)
            return out
        finally:
            available.put(server)

    try:
        # --- stage 4: the solving portfolio (tharkun.sv.stages, [internal ref]): seed stages, and the
        # per-site concolic drive on the sites stage 2 left open ---
        # a helper process gets one child's share of the memory cap, like every run, so the
        # verifier and whatever it has started stay inside the task's limit together
        seeded: dict = {}
        proposed = 0                 # vectors the portfolio proposed, over every round ([internal ref])

        def replay_stream(stream) -> _engine.Verdict | None:
            """Replay each vector a stage proposes, in order, until one is a confirmed violation (the
            verdict) or the stop condition holds / the stream ends (``None``).  A stage that is a
            generator holds a helper process (the fuzzer): it is closed the moment this returns."""
            nonlocal runs, proposed
            try:
                for cand in stream:
                    if ledger.stopped():
                        break
                    out = run(cand.vector) if proof is not None else explore(cand.vector)
                    runs += 1
                    proposed += 1
                    hit = out.violated(classes)
                    if hit:
                        say(f"run {runs}: {hit} at {out.site} (solved, {cand.component})")
                        return found(_engine.Verdict("false", hit, witness=_engine._witness(program, prop_name, hit, out),
                                             component=cand.component, detail=cand.detail),
                                     fresh='forked' not in out.flags)
                    say(f"run {runs}: solved vector did not violate ({out.status} {out.exc or ''})")
            finally:
                close = getattr(stream, "close", None)
                if close is not None:
                    close()
            return None

        for stage in portfolio:
            if ledger.stopped(reserve=1):
                break
            if stage.seed is not None:
                vals = stage.seed(ctx)
                if vals:
                    seeded[stage.component] = vals
                continue
            if stage.propose is None:        # a stage that only deepens ([internal ref]) has no bounded look
                continue
            share = concolic_share if stage.name == "concolic" else stage.share
            v = replay_stream(stage.propose(ctx, ledger.share(share)))
            if v is not None:
                return v

        def spend_remainder() -> _engine.Verdict | None:
            """Hand what enumeration left to the stages that can search an unbounded domain further
            (``Stage.deepen``), in rounds of half the remaining budget each, until the goal is met
            or the ledger says stop.  A round in which no stage proposed anything and no CPU was
            spent means there is nothing left to try (no fuzzer installed): that, and only that, ends
            the run short of the budget."""
            deepening = [st for st in portfolio if st.deepen is not None]
            rounds = 0
            while deepening and not ledger.stopped(reserve=1) and runs < max_runs:
                rounds += 1
                spent_before, proposed_before = ledger.spent(), proposed
                for st in deepening:
                    if ledger.stopped(reserve=1):
                        break
                    budget = ledger.share(DEEPEN_SHARE)
                    say(f"deepen: round {rounds}, {st.name}, up to {budget:.0f}s CPU")
                    v = replay_stream(st.deepen(ctx, budget, rounds))
                    if v is not None:
                        return v
                if proposed == proposed_before and ledger.spent() - spent_before < 0.5:
                    say("deepen: nothing left to try")
                    break
            return None

        # --- stage 5: generational small-first search over draw vectors ---
        cands = _engine.Candidates(_engine.harvest_constants(program) if constants else None, deepen=deepen,
                           seeded=seeded)
        seen: set = set()
        heap: list = []
        tie = _engine.itertools.count()
        finite = not first.flags
        complete = True

        def strip(draws):
            return [{k: v for k, v in d.items() if k != "site"} for d in draws]

        def push(prefix: list, prefix_cost: int, entry: dict, r: int, cur: tuple) -> None:
            """Queue the first candidate of rank >= ``r`` for ``entry``'s position that differs
            from the executed value ``cur`` and has not been queued; its chain continues lazily."""
            while True:
                v = cands.value(entry["kind"], r, entry.get("n"), _engine.entry_base(entry))
                if v is None:
                    return
                k = _engine._key(entry["kind"], v)
                if k != cur:
                    alt = dict(entry, value=_engine._encode(entry["kind"], v))
                    child = prefix + [alt]
                    sig = tuple(_engine._key(c["kind"], _engine._decode(c)) for c in child)
                    if sig not in seen:
                        seen.add(sig)
                        _engine.heapq.heappush(heap, (prefix_cost + r, len(child), next(tie), child,
                                              (prefix, prefix_cost, entry, r, cur)))
                        return
                r += 1

        def expand(prefix_len: int, draws: list) -> None:
            nonlocal finite
            base = strip(draws)
            cost = sum(cands.rank(e) for e in base[:prefix_len])
            for j in range(prefix_len, len(base)):
                e = base[j]
                if e["kind"] not in _engine.FINITE_KINDS or (e["kind"] == "rindex" and e.get("n", 0) > _engine._RINDEX_CAP):
                    finite = False
                push(base[:j], cost, e, 0, _engine._key(e["kind"], _engine._decode(e)))
                cost += cands.rank(e)

        expand(0, first.draws)
        with _engine.ThreadPoolExecutor(max_workers=jobs) as pool:
            while heap:
                if ledger.stopped() or runs >= max_runs:
                    complete = False
                    break
                # Each popped candidate hands its chain to the next value BEFORE the next pop, so
                # that value can join the same batch: a program with one unbounded draw is a single
                # chain, and popping first would give batches of one and idle children ([internal ref]).
                batch = []
                while heap and len(batch) < min(jobs, max_runs - runs):
                    item = _engine.heapq.heappop(heap)
                    prefix, prefix_cost, entry, r, cur = item[4]
                    push(prefix, prefix_cost, entry, r + 1, cur)
                    batch.append(item)
                # Until the finite-tree proof is ruled out, every supporting run is fresh.
                # A static/dynamic proof's canary stays fresh too: no TRUE depends on a fork.
                runner = run if finite or proof is not None else explore
                outs = list(pool.map(lambda it: runner(it[3], share=len(batch)), batch))
                runs += len(batch)
                for item, out in zip(batch, outs):   # cost order: the verdict does not depend on timing
                    cost, vec = item[0], item[3]
                    if out.status in ("timeout", "crashed") or _engine._resource_exhausted(out):
                        complete = False
                        say(f"run {runs}: {out.status} cost={cost}")
                        continue
                    if out.flags:
                        finite = False
                    hit = out.violated(classes)
                    if hit:
                        say(f"run {runs}: {hit} at {out.site} cost={cost}")
                        # a seed stage's value in the vector is what the hit is owed to
                        order = ("small_first", "constants", "tail") + tuple(seeded)
                        src = max((cands.source(e) for e in strip(vec)), key=order.index,
                                  default="small_first")
                        return found(_engine.Verdict("false", hit, witness=_engine._witness(program, prop_name, hit, out),
                                             component=src if src in seeded else f"enum:{src}"),
                                     fresh='forked' not in out.flags)
                    expand(len(vec), out.draws)
        if complete and finite:
            # one run per draw vector covers the tree only if runs are deterministic given the draws
            det = _engine._certify_determinism(program, absent_imports=frozenset(absent_seen), files=files)
            if det is not None and det.ok:
                return done(_engine.Verdict("true", reason=f"finite draw tree enumerated; runs certified "
                                                   f"deterministic given the draws ({det.reason})",
                                    component="finite_tree", detail={"certificate": det.reason}))
            why = det.reason if det is not None else "the program cannot be read"
            return done(proof or _engine.Verdict("unknown", reason=f"finite draw tree enumerated, but determinism "
                                                       f"is not certified: {why}"))
        if proof is not None:
            return done(proof)
        # --- stage 6 ([internal ref]): enumeration ran dry, or stopped short of the budget -- that never ends
        # the run by itself.  What ends it is the goal being met (a confirmed violation: ``found``
        # returns) or the stop condition (budget, cap).  The stages that can still spend CPU
        # productively on an unbounded draw domain (``Stage.deepen``: the fuzzer) get the rest, in
        # rounds, each under the same ledger.
        if runs < max_runs and not ledger.stopped(reserve=1):
            v = spend_remainder()
            if v is not None:
                return v
        if symbolic is not None and symbolic.closed and not complete and remaining() > 0:
            return done(_engine.Verdict("unknown", reason=f"no violation found (every site solved to no modelled "
                                                  f"violation; search stopped early; {runs} runs)"))
        why = ledger.stop_reason(reserve=1) or "no stage left to try"
        return done(_engine.Verdict("unknown", reason=f"no violation found ({why}; {runs} runs)"))
    finally:
        for server in servers:
            server.close()
        ctx.cleanup.close()

