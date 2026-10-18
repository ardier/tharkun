"""Facts to proof: producers -> the store -> independent certification -> discharge
([internal ref] step 3; [internal ref] for the producer, [internal ref] for the store).

The scheduler stage (``sv/scheduler.py::run``, after the first run and before
``prove_dynamic``) calls :func:`prove_from_facts`.  It is the CONSUMER half of
the Eregion bridge and decides nothing on a producer's word:

1. every installed producer (``sv/fact_producers.py``, the generic plugin seam)
   gets the open obligations and returns plain candidate facts;
2. ``sv/eregion_bridge.py::publish_fact`` publishes them into the
   :class:`~tharkun.knowledge.KnowledgeBase` as ``Evidence`` -- the only door in;
3. THIS module reads the candidates back OUT of the store (not from the
   producer's return value) and checks each ``Invariant`` on its own with
   ``sv.validate.check_candidates_individually`` -- the independent Z3 source
   checker, which never sees the producer's reasoning;
4. a candidate the checker proved is published once more as
   ``CERTIFIED_INVARIANT`` (written nowhere else);
5. the stage decides TRUE only when EVERY current obligation of the property
   (census complete, no adapter gaps) has its own invariant successfully checked
   in THIS invocation. Stored certification and discharge remain historical data.
   Anything less decides nothing: an obligation nobody checked stays open and
   the later stages run as if this stage did not exist.

What a producer does and does not add.  A fact discharges an obligation only
by being certified, and the checker proves an invariant from the SOURCE alone,
so a producer's contribution here is the NOMINATION (which obligations to try,
with its own evidence kept as data for the record); the proof power is the
checker's.  Facts that carry information the checker cannot derive itself (a
loop invariant, a class state invariant, a periodic-family intersection) need a
verifier for their own shape first; ``sv.validate.FamilyIntersectionCandidate`` has
none (its claims are over the reals, and the float-soundness lemma a verifier
would need lives in absint, ``solve/absint/lemmas.py::TRIG_PAIR``), so the bridge
has no shape for it (docs/SV.md "3a": ruled, not built).

Soundness rules this stage keeps (the same as every proof stage, [internal ref] 3a):
a TRUE gets the same canary as any proof (``sv/scheduler.py::run`` caps the
enumeration after it); it is attempted only when the first run raised no
uncontrolled-nondeterminism flag; the interpreter must be CPython 3.14 (the
checker's own requirement); a producer that crashes, times out or answers
badly yields nothing.
"""
from __future__ import annotations

import time

from tharkun.sv import engine as _engine
from tharkun.sv import eregion_bridge as bridge
from tharkun.sv import fact_producers
from tharkun.sv.obligation_table import VIOLATED, ObligationTable
from tharkun.sv.store_witness import bind_route

#: the producers' share of the CPU left, and a hard cap on one call; certifying what they
#: propose runs inside the same budget
PRODUCER_SHARE = 0.15
PRODUCER_CAP_SECONDS = 30.0   # the whole stage is at most this plus one checker run
#: at most this many candidates are certified per call (each is a Z3 run)
MAX_CANDIDATES = 32
CHECKER = "sv.validate.check_candidates_individually"


def prove_from_facts(program: str, classes, *, source: str, specification: str, table: ObligationTable,
                     python: str, remaining, say, checker_seconds: float = 3.0,
                     memory_bytes: int | None = None):
    """A TRUE ``Verdict`` (component ``proof:facts:<producers>``) when fresh
    independent checks cover every current obligation, else ``None``. Never a FALSE: this
    stage proposes no draw vector, so nothing here needs a replay."""
    from tharkun.sv.proof import _python314
    from tharkun.sv.validate import check_candidates_individually

    from tharkun.solve.extern import native_import_reason

    if native_import_reason(source, program):
        return None
    obligations, knowledge = table.census, table.kb
    every = table.rows(classes)
    # Fail closed (MG review of [internal ref] step 3a): a row a confirmed replay VIOLATED is a FALSE; no proof
    # stage may decide TRUE around it.  The scheduler ends the run in found() before this stage, so this
    # never fires today; it keeps "a violated row is not open" from ever reading as "nothing left to
    # prove".  Gaps are checked over every row of the classes, as before step 3a, not only the open ones.
    if any(table.status(o) == VIOLATED for o in every):
        return None
    rows = table.open_rows(classes)       # the live table: a row a whole-program proof discharged is not open
    if not rows or not obligations.complete or any(o.gap for o in every):
        return None                       # nothing for facts to close, or the census is not a boundary
    producers = fact_producers.discover()
    if not producers or not _python314(python):
        return None
    budget = min(remaining() * PRODUCER_SHARE, PRODUCER_CAP_SECONDS)
    if budget < 1.0:
        return None
    stage_deadline = time.monotonic() + budget

    # 1-2: ask every producer; everything that comes back goes through the store's door
    request = fact_producers.make_request(program, classes, rows, budget)
    for producer in producers:
        started = time.monotonic()
        facts = fact_producers.run(producer, request, timeout=stage_deadline - started, memory_bytes=memory_bytes,
                                   log=say)
        published = 0
        for fact in facts:
            try:
                bridge.publish_fact(knowledge, fact, producer=producer.name)
                published += 1
            except (KeyError, ValueError, TypeError) as e:
                say(f"facts: {producer.name}: dropped a malformed fact ({type(e).__name__})")
        say(f"facts: {producer.name} published {published} of {len(facts)} fact(s) "
            f"in {time.monotonic() - started:.1f}s")

    # 3-4: read the candidates back out of the store and certify each one alone
    locations = {(o.invariant.file_name, o.invariant.line, o.invariant.column, o.invariant.function)
                 for o in every}
    candidates = [item.value["invariant"] for item in knowledge.evidence(kind=bridge.INVARIANT_CANDIDATE)
                  if (item.value["invariant"].file_name, item.value["invariant"].line,
                      item.value["invariant"].column, item.value["invariant"].function) in locations]
    certified = 0
    proven = set()  # only this invocation's successful source/property checks can cover a row
    proof_parents = {}
    # certification runs inside the same budget share; a producer that used all of it still
    # leaves room for ONE checker run, so the whole stage is at most budget + checker_seconds
    deadline = max(stage_deadline, time.monotonic() + checker_seconds)
    for inv in candidates[:MAX_CANDIDATES]:
        left = min(deadline - time.monotonic(), remaining())
        if left <= 0:
            say("facts: the certification budget ran out; the rest stay candidates")
            break
        try:
            result = check_candidates_individually(source, program, specification, obligations, [inv],
                                                   seconds=min(checker_seconds, left))
        except Exception as e:  # noqa: BLE001 -- a checker that crashes certifies nothing
            say(f"facts: checker crashed on a candidate ({type(e).__name__}); not certified")
            continue
        if result.get(inv) is True:
            bridge.publish_certified(knowledge, inv, checker=CHECKER, producer="fact_proof")
            certified += 1
            proven.add(inv)
            try:
                proof_parents[inv] = knowledge.evidence_item(
                    bridge.CERTIFIED_INVARIANT, bridge.invariant_key(inv)).identity
            except Exception:  # route metadata is additive to the existing proof
                pass
    say(f"facts: {certified} of {len(candidates)} candidate invariant(s) certified")

    # 5: old certified Evidence or DISCHARGED rows cannot substitute for a current check.
    # Producers still receive open rows; stored nominations for other current rows may be rechecked.
    uncovered = [o for o in every if o.invariant not in proven]
    if uncovered:
        say(f"facts: {len(every) - len(uncovered)} of {len(every)} obligation(s) covered; "
            f"first open: {uncovered[0].ast_path} {uncovered[0].exception}")
        return None
    names = sorted({d.producer for o in every
                    for d in knowledge.evidence_item(bridge.INVARIANT_CANDIDATE,
                                                     bridge.invariant_key(o.invariant)).derivations})
    sites = [f"{o.invariant.file_name}:{o.invariant.line}:{o.invariant.column}:{o.exception}" for o in every]
    say(f"facts: every obligation ({len(every)}) is covered by a certified invariant")
    verdict = _engine.Verdict("true", reason=f"certified facts: {len(every)} obligation(s) discharged ({', '.join(names)})",
                           component="proof:facts:" + "+".join(names),
                           detail={"sites": sites, "certified": [o.invariant.entry() for o in every]})
    if all(o.invariant in proof_parents for o in every):
        bind_route(verdict, "producer", source, program, classes,
                   parents=tuple(dict.fromkeys(proof_parents[o.invariant] for o in every)))
    return verdict
