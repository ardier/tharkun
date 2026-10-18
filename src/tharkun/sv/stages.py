"""The portfolio of solving stages ``tharkun.sv`` runs before its enumeration ([internal ref]).

``engine.verify`` decides a property by RUNNING draw vectors; each stage here is a
way of choosing better vectors, with a known cost and its own attribution name
(the ``component`` printed before the verdict, lane C's results column).  There
are two shapes:

* a **vector stage** proposes whole draw vectors (``propose(ctx, budget)`` yields
  :class:`Candidate`); the engine replays each one on the unmodified program in a
  fresh child, in order, while the stage's share of the budget lasts.  Only a
  replayed run that raises a property class is a FALSE, so a stage can be wrong
  at no cost but a run;
* a **seed stage** proposes candidate VALUES per draw kind (``seed(ctx)`` returns
  ``{kind: [values]}``); :class:`engine.Candidates` ranks them right after the
  kind's head values, and a FALSE whose vector used one is attributed to the stage.

Stages run in :data:`REGISTRY` order (cheapest first).  ``verify(stages=...)`` and
``--stages`` / ``--no-stage`` select them; a stage whose numbers show it adds
nothing stays out of :data:`DEFAULT`, with the numbers in its entry.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional


@dataclass
class Candidate:
    """One proposed draw vector and the component that produced it."""

    vector: list
    component: str
    detail: dict = field(default_factory=dict)


@dataclass
class Context:
    """What a stage may know about the task: the program, the property classes, the
    interpreter and memory cap its helper processes must use, and the first run."""

    program: str
    classes: tuple
    python: Optional[str] = None
    memory_bytes: Optional[int] = None
    first: object = None            # engine.Outcome of the all-defaults run
    log: Callable = lambda msg: None
    #: the run's obligation table ([internal ref] step 3a, ``sv/obligation_table.py``): what the earlier
    #: components learned -- absint's site records, a concolic stage's per-line outcomes -- is read
    #: from it, not handed over in fields
    table: object = None
    #: the task's declared files (``solve/environment.py::TaskEnvironment.files``; the program first): what a helper
    #: process that runs the program for real stages in its scratch directory, like the engine's child does
    files: Optional[tuple] = None
    #: the run's CPU-second ledger (``sv/ledger.py``, [internal ref]): a stage that starts a helper process
    #: (the fuzzer) registers it there, so its CPU counts against the one budget while it runs, and
    #: reads ``ledger.stopped()`` -- the one stop condition -- instead of keeping a clock of its own
    ledger: object = None
    #: the task's own ``.yml`` when there is one (declared/required files, ``sv/reach.py``), else ``None``
    task: Optional[str] = None
    #: what a stage keeps between rounds of ``Stage.deepen`` (the fuzzer's corpus directory, ...)
    state: dict = field(default_factory=dict)
    #: closed by the scheduler when the run ends; a stage puts what it created for ``state`` here
    cleanup: contextlib.ExitStack = field(default_factory=contextlib.ExitStack)

    def site_records(self):
        """The abstract interpreter's site records (the latest absint run's), or ``None``."""
        return self.table.site_records() if self.table is not None else None

    def call_facts(self) -> dict:
        """The ``call_target`` facts a ``concolic.solve`` call takes (``call_targets=``, ``proven_classes=``): what
        ``scheduler.publish_module_namespace`` published, or nothing known (``None``: the symbolic executor then
        leaves a call of a class unmodelled) when the run has no table."""
        if self.table is None:
            return {"call_targets": None, "proven_classes": ()}
        return {"call_targets": self.table.call_targets(), "proven_classes": self.table.proven_classes()}


@dataclass(frozen=True)
class Stage:
    name: str                        # selection name (``--stages``)
    component: str                   # attribution name (lane C's column)
    propose: Optional[Callable] = None   # (ctx, budget_seconds) -> Iterable[Candidate]
    seed: Optional[Callable] = None      # (ctx) -> {kind: [values]}
    share: float = 0.0               # vector stages: at most this share of the remaining budget
    #: (ctx, budget_seconds, round) -> Iterable[Candidate]: how the stage keeps searching an UNBOUNDED draw
    #: domain after enumeration ran dry ([internal ref]).  A generator, so each vector is replayed the moment it
    #: appears and the stage's helper process stops with the run; ``round`` counts from 1.
    deepen: Optional[Callable] = None


def _concolic(ctx: Context, budget: float) -> Iterable[Candidate]:
    """The per-site drive ([internal ref]).  After the engine's symbolic stage it runs only on the
    sites that stage left open; without it, the whole concolic stage runs here."""
    from tharkun.sv import concolic
    symbolic = ctx.table.solved("symbolic") if ctx.table is not None else None
    if symbolic is None:
        solved = concolic.solve(ctx.program, ctx.classes, budget_seconds=budget,
                                memory_bytes=ctx.memory_bytes, python=None, extra_sites=ctx.site_records(),
                                log=ctx.log, files=ctx.files, **ctx.call_facts())
    else:
        still = symbolic.open_lines()
        if not still:
            return []
        solved = concolic.solve(ctx.program, ctx.classes, budget_seconds=budget, memory_bytes=ctx.memory_bytes,
                                python=None, extra_sites=ctx.site_records(), symbolic=False, lines=still,
                                log=ctx.log, files=ctx.files, **ctx.call_facts())
    if ctx.table is not None:
        ctx.table.publish_solved("drive", solved, producer="concolic:drive")
    return solved.candidates


#: at most this many of a producer's vectors are replayed per run
MAX_PRODUCER_VECTORS = 64


def _producer(ctx: Context, budget: float) -> Iterable[Candidate]:
    """The draw vectors an installed fact producer proposed (docs/SV.md "Draw vectors from a producer").

    They were published into the run's store by the facts stage (``sv/fact_proof.py::prove_from_facts``, the one
    door a producer's answer comes through) and are READ from it here.  Each is checked out of band first
    (``engine.vector_admissible``: no draw the child would refuse, no float outside the bounds the first run
    saw) and then replayed like every other candidate: on the unmodified program, in a fresh child, through
    the scheduler's one FALSE path.  A vector the first run already is (every draw its kind's default) is not
    replayed again.  With no producer installed the store holds none and this yields nothing."""
    if ctx.table is None:
        return []
    from tharkun.sv import engine as _engine
    from tharkun.sv.eregion_bridge import DRAW_VECTOR

    defaults = {"bool": False, "int": 0, "float": 0.0, "str": ""}
    out = []
    for item in ctx.table.kb.evidence(kind=DRAW_VECTOR):
        vector = item.value["vector"]
        if not _engine.vector_admissible(vector, ctx.first):
            ctx.log("producer: a proposed draw vector holds a draw the child would refuse; not replayed")
            continue
        if all(_engine._key(e["kind"], _engine._decode(e)) == _engine._key(e["kind"], defaults[e["kind"]])
               for e in vector):
            continue
        names = "+".join(sorted({d.producer for d in item.derivations}))
        out.append(Candidate([dict(e) for e in vector], f"producer:{names}", {"site": item.value["site"]}))
        if len(out) >= MAX_PRODUCER_VECTORS:
            break
    return out


def _hybrid(ctx: Context, budget: float) -> Iterable[Candidate]:
    from tharkun.sv.hybrid import search
    return search(ctx, budget)


def _hybrid_deepen(ctx: Context, budget: float, round_: int) -> Iterable[Candidate]:
    from tharkun.sv.hybrid import deepen
    return deepen(ctx, budget, round_)


def _solver_seeds(ctx: Context, trick: str) -> dict:
    from tharkun.sv.solver_seeds import seed_values
    return seed_values(ctx.program, trick, ctx.site_records())


def _guard_values(ctx: Context) -> dict:
    from tharkun.sv.guard_values import seed_values
    return seed_values(ctx.program, log=ctx.log)


def _avengers(ctx: Context) -> dict:
    from tharkun.sv.avengers_seed import seed_values
    return seed_values(ctx.program, log=ctx.log, files=ctx.files)


#: every stage, cheapest first
REGISTRY: tuple = (
    # static: solve the program's guards over the draws (solve/arith, solve_condition);
    # a parse plus milliseconds per guard
    Stage("guard_values", "static:guard_values", seed=_guard_values),
    Stage("comparison", "static:comparison", seed=lambda c: _solver_seeds(c, "comparison")),
    Stage("boundaries", "static:boundaries", seed=lambda c: _solver_seeds(c, "boundaries")),
    Stage("composition", "static:composition", seed=lambda c: _solver_seeds(c, "composition")),
    # the draw vectors a fact producer proposed (Eregion's lift of the main block to a site, inverted);
    # replays only, bounded by MAX_PRODUCER_VECTORS, nothing at all with no producer installed
    Stage("producer", "producer", propose=_producer, share=0.05),
    # drive_to_line toward each violation site on the lifted harness ([internal ref])
    Stage("concolic", "concolic", propose=_concolic, share=0.4),
    # Atheris coverage search on the negation of each open SV obligation ([internal ref]): fuzzes
    # the task's own _sv_verifier draws with Atheris, keeps the closest numeric seed by
    # |divisor|, then hands a coverage plateau to the per-site solver. Its own
    # plateau/deadline loop (sv/hybrid.py::search) originally budgeted by wall-clock
    # (time.monotonic()); three back-to-back run-tasks --full-budget campaigns against
    # successive sv_2027 tips showed the SO gain varying +3/+2/+0 as host load rose, with
    # 0 lost/0 new wrong in all three (the size of the gain was load-sensitive, not
    # soundness). [internal ref] part 2 fixed this: AtherisBackend.cpu_seconds() (backend.py) reads
    # the fuzzer subprocess's own process CPU time from /proc, and search() now budgets
    # its loop and its handoff to the concolic solver on that instead -- what BenchExec
    # measures and what SV-COMP ties break on. Two more run-tasks --full-budget campaigns
    # after the fix, at genuinely different host concurrency (shared_task_cap 36 and 52),
    # agree exactly:
    #   [local path] (base 2529ec8, cap=36)
    #   .../20260928-170612-369602-[host]-1d67d1e6 (base 2529ec8, cap=52)
    # both: solve-only 20->22 (+2: sm_count_loop, so_count_loop_bug, both assertion-safety),
    # 0 lost, 0 new wrong official+adjusted, hybrid:atheris 0->4 both times. Real-world
    # 112->112 adjusted (0 new wrong) both times; Float 12->12 (0 new wrong) both times.
    Stage("hybrid", "hybrid:obligation", propose=_hybrid, share=0.15, deepen=_hybrid_deepen),
    # Avengers receiver synthesis ([internal ref]): construct classes the module defines and
    # harvest literal constructor/setup arguments, including whatever the restored
    # Pynguin backend proposes (tharkun.avengers.backends). Measured (direct
    # invocation, 60s/run, no BenchExec) on all 80 (task, property) pairs of the
    # 24-task SV-Benchmarks Real-world set, WITH the backend actually installed
    # and invoked: 0 pairs gained, 0 lost -- identical verdict on every pair,
    # including the 30 pairs over object-heavy tasks (richards/raytrace/go/chaos)
    # this stage targets. It also has a real cost: on 19/80 pairs it measurably
    # crowded out the concolic stage's own run budget (e.g. bm_raytrace_nondet
    # fell from 1104/1108/1092 runs to 1 on three of its four properties), because
    # its own search (including the Pynguin subprocess) can spend up to ~30s of a
    # 60s total budget before enumeration gets a turn. Out of DEFAULT per the
    # keep-only-if-gain rule -- this is the SAME 0/24-gain result the PR's earlier
    # (invalid) measurement reported, now independently confirmed with the
    # correct backend policy actually in effect. Available via
    # ``--stages ...,avengers`` for re-measurement as the portfolio or the corpus
    # changes, and the backend itself stays available to Avengers callers outside
    # SV regardless of this stage's default-on state.
    Stage("avengers", "avengers", seed=_avengers),
)

#: the stages ``verify`` runs unless told otherwise
DEFAULT: tuple = ("guard_values", "comparison", "boundaries", "composition", "producer", "concolic", "hybrid")


def select(names: Iterable[str] | None = None, off: Iterable[str] = ()) -> list:
    """The registry's stages named in ``names`` (default :data:`DEFAULT`) minus ``off``,
    in registry order.  An unknown name is an error."""
    known = {s.name for s in REGISTRY}
    want = set(DEFAULT if names is None else names)
    bad = (want | set(off)) - known
    if bad:
        raise ValueError(f"unknown stage(s) {sorted(bad)}; known: {sorted(known)}")
    want -= set(off)
    return [s for s in REGISTRY if s.name in want]
