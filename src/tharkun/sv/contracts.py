"""The component contracts of the SV scheduler, as data ([internal ref] step 3a, docs/SV.md "Component contracts").

NOT the stdlib contracts.  ``solve/contracts/`` and ``docs/contracts/`` hold the CONTRACTS OF PYTHON
LIBRARY CALLS (what ``re.match`` or ``dict.get`` may raise and return, for absint and the solver).  This
module holds the contracts of the SCHEDULER'S COMPONENTS: what census, absint, the concolic stages, the
fact producers and the rest read from and write to the run's store, what they cost and what they may
decide.  The two share a word and nothing else.

``sv/scheduler.py::run`` runs components in a fixed order, cheapest-DECIDING first (owner ruling
2026-09-24).  Each one has a contract -- what it reads, what it writes, what it costs and what it may
decide -- and the contracts are kept HERE, once, so that the code and the design doc cannot drift:

* :data:`KINDS` is every kind of row/evidence the run's store holds, with who writes and reads it and
  whether a row of that kind may ever be certified;
* :data:`COMPONENTS` is the stage order, one :class:`Component` per component;
* :func:`render_table` renders the docs/SV.md table from :data:`COMPONENTS`;
  ``tests/test_sv_contracts.py`` fails when the doc's table differs from it, when a ``where`` names a
  symbol that does not exist, when a component reads or writes a kind that is not registered, and when
  ``certified=True`` is written anywhere but ``sv/obligation_table.py`` (MG's review bar: only a
  whole-program proof and a confirmed replay certify).

The rule every contract states, from the owner's 2026-09-29 design principle ("approximation by
purpose"): PROOF components (census, absint, determinism, the stationary proof, the facts stage, the
finalizer) are always conservative and rely only on certified facts or on sources they check
themselves; SEARCH components (concolic, hybrid, seeds, enumeration, producers) are lax and may assume
anything -- soundness comes from the fresh-child replay, the only FALSE source.  ``ObligationTable``
is where that is enforced: only a decision is certified, everything else is a hint.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Kind:
    name: str
    store: str                 # "fact" (a subject-keyed Fact) or "evidence" (a keyed, immutable Evidence)
    certifiable: bool          # may a row of this kind ever be certified?
    writers: tuple
    readers: tuple
    meaning: str


KINDS = {k.name: k for k in (
    Kind("obligation", "fact", True, ("census", "ObligationTable.discharge", "ObligationTable.violate"),
         ("facts",), "a census row's status: open, discharged (whole-program proof), violated (replay)"),
    Kind("site_record", "evidence", False, ("absint",),
         ("symbolic", "concolic", "seeds", "store witness shadow"), "absint's per-site record; one generation per absint run"),
    Kind("site_safety", "fact", False, ("ObligationTable.publish_safety",),
         ("ObligationTable.safety",), "a site's canonical safety term, proposed/established status and evidence parents"),
    Kind("proof_route", "evidence", False, ("ObligationTable.publish_route",),
         ("ObligationTable.proof_routes", "store witness shadow"),
         "the source, exception classes, sites and site-record parent count of a deciding prover"),
    Kind("site_outcome", "evidence", False, ("symbolic", "concolic"),
         ("concolic", "scheduler"), "a concolic stage's outcome for one harness line"),
    Kind("stage_ran", "evidence", False, ("symbolic", "concolic"),
         ("concolic", "scheduler"), "a concolic stage finished, with or without a site"),
    Kind("class_hints", "evidence", False, ("execution",),
         ("absint",), "what the first run observed about the program's classes"),
    Kind("candidate_invariant", "fact", False, ("absint", "finalizer"), (),
         "an invariant a component proposed before it was checked (docs/SV.md 3a: no reader yet)"),
    Kind("input_sensitivity", "fact", False, ("execution",), (),
         "which draw position moves the outcome (a differencing hint; Eregion #1056 is the reader)"),
    Kind("loop_trip_count", "fact", False, ("stationary",), (),
         "observed trip counts of a loop under probe runs (no reader yet)"),
    Kind("invariant_candidate", "evidence", False, ("producers",), ("facts",),
         "a fact producer's nominated invariant at a source location (sv/eregion_bridge.py)"),
    Kind("certified_invariant", "evidence", False, ("facts",), ("facts", "finalizer"),
         "an invariant the independent per-candidate checker proved (written only by publish_certified)"),
    Kind("draw_vector", "evidence", False, ("producers",), ("producer vectors",),
         "a fact producer's proposed draw vector, replayed natively and never trusted (sv/eregion_bridge.py)"),
    Kind("regex_pattern", "evidence", False, ("static facts",), (),
         "a compiled regex of the task's import graph, statically folded: pattern, explicit and compiled flags, "
         "where it is defined, every name that binds it, and whether each is a proven module constant "
         "(solve/module_namespace.py::regex_globals; no scheduler reader yet, the string contracts of #1126 "
         "are the intended one)"),
    Kind("call_target", "evidence", False, ("static facts",), ("symbolic", "stationary", "finalizer"),
         "what one call site of the task's entry module calls: a user class / function / method, an imported "
         "or builtin callee or unknown, with its definition site and whether the name's binding is proven "
         "(solve/call_targets.py::call_targets; the readers are sv/_symexec.py through sv/concolic.py, and "
         "solve/drawflow.py through sv/prove.py)"),
    Kind("module_constant", "evidence", False, ("static facts",), ("symbolic",),
         "one module-level name of the task's entry module: whether anything in its import graph rebinds it, "
         "with the evidence when something does (solve/module_namespace.py::module_constants; Eregion's "
         "synthesis bakes a value only when it is proven; the symbolic stage reads the proven class names, "
         "ObligationTable.proven_classes)"),
)}


@dataclass(frozen=True)
class Component:
    id: str
    where: str                 # "path/under/tharkun::symbol" -- ``;``-separated when several
    reads: str                 # prose; every ``kind`` in backticks must be a registered kind
    writes: str
    cost: str
    decides: str


#: Stage order (cheapest-deciding first); the ``stage`` of a component is its place in
#: ``sv/scheduler.py::run``'s docstring.
COMPONENTS = (
    Component("static facts (module namespace)",
              "solve/module_namespace.py::module_constants;solve/module_namespace.py::regex_globals;"
              "solve/call_targets.py::call_targets;sv/scheduler.py::publish_module_namespace",
              "source (and the local modules it imports)", "`module_constant`, `regex_pattern`, `call_target`",
              "parse and fold, executes nothing, no budget", "nothing (hints and proofs of constancy only)"),
    Component("census (no-site)", "sv/prove.py::prove_static",
              "source, classes", "nothing shared", "instant", "TRUE only"),
    Component("absint", "solve/absint/__init__.py::prove;sv/engine.py::_static_proof",
              "source, classes, `class_hints` (the second try), invariant locations",
              "`site_record`, `candidate_invariant`",
              "up to 10% of remaining, capped 60s", "TRUE only"),
    Component("exact product proof", "sv/product_proof.py::prove",
              "source, classes", "nothing shared",
              "replay at most min(30s, remaining); Z3 timeout min(60s, max(10s, 15% of remaining))",
              "TRUE (Z3 UNSAT) or FALSE (a model replay-confirmed; never for draws inside defs)"),
    Component("symbolic (whole-harness)", "sv/concolic.py::solve",
              "program, classes, `site_record`", "`site_outcome`, `stage_ran`",
              "up to `concolic_share` of remaining, capped 60s CPU", "proposes only (replay decides FALSE)"),
    Component("execution (first/second run)", "sv/engine.py::run_once",
              "program, forced draws (none)", "`class_hints`, `input_sensitivity`",
              "one child, `run_timeout` share", "FALSE (replayed run)"),
    Component("determinism certificate", "solve/determinism.py::determinism_census",
              "source", "nothing shared", "parse + census, no execution", "TRUE only (given two agreeing runs)"),
    Component("facts (fact producers)", "sv/fact_proof.py::prove_from_facts;sv/fact_producers.py::run",
              "`obligation` (open rows), `invariant_candidate`",
              "`invariant_candidate` and `draw_vector` (the producers'), `certified_invariant`",
              "0.15 of remaining, capped 30s CPU, producers and certification together",
              "TRUE only, and only when every current obligation has its own invariant checked in this invocation"),
    Component("stationary", "sv/prove.py::prove_dynamic",
              "program, classes, `python`, `jobs`, `call_target`", "`loop_trip_count`; replays its probes via `run_once`",
              "its own loop, bounded by remaining and `MAX_RUNS`", "TRUE or FALSE"),
    Component("seed stages", "sv/guard_values.py::seed_values;sv/solver_seeds.py::seed_values;sv/avengers_seed.py::seed_values",
              "source (static), `site_record` (comparison/boundaries/composition)", "seeded values (scheduler-local)",
              "milliseconds to seconds, static", "hint only"),
    Component("producer vectors", "sv/stages.py::_producer",
              "`draw_vector`, the first run (for the pre-check)", "nothing shared",
              "one replay per vector, at most `MAX_PRODUCER_VECTORS`", "proposes only (replay decides FALSE)"),
    Component("symbolic (narrowed) / hybrid", "sv/stages.py::_concolic;sv/hybrid.py::search",
              "`obligation` (hybrid: the open rows), `site_record`, `site_outcome` (the lines the symbolic stage left open), the first run, and for hybrid the task code the program calls (`sv/reach.py`)",
              "`site_outcome`, `stage_ran`", "`stage.share` of remaining", "proposes only"),
    Component("enumeration", "sv/engine.py::Candidates",
              "seeded values, program literals, the first run's draws", "nothing shared",
              "the remainder of the ledger", "FALSE (replayed); TRUE via `finite_tree`"),
    Component("remainder (deepen)", "sv/scheduler.py::run;sv/hybrid.py::deepen",
              "`obligation` (the precise rows), what the earlier rounds kept (`Context.state`)", "nothing shared",
              "what enumeration left, in rounds of half the remaining; a cap still bounds it",
              "proposes only (replay decides FALSE)"),
    Component("finalizer", "sv/scheduler.py::run;sv/proof.py::finish",
              "`obligation` census, the verdict's `candidate_invariants` and bound route premises (`certified_invariant`, `call_target`)",
              "`proof_route` (actual final TRUE), `candidate_invariant`, `obligation` (discharged, via `ObligationTable.discharge`), `witness.yml`",
              "up to `checker_seconds` (3.0), shared with the YAML writes", "never downgrades; records what stays open"),
    Component("store witness shadow", "sv/store_witness.py::project;sv/store_witness.py::shadow",
              "source identity, `site_record`, `proof_route`",
              "`site_safety`, `store-witness.yml`, `store_*` metrics",
              "256 sites, 10000 Z3 resource units/query, up to 1s or 5% of remaining ledger", "nothing; runs after legacy exports"),
)

KIND_NAMES = frozenset(KINDS)


def render_table() -> str:
    """The docs/SV.md component-contract table (header and rows), generated from :data:`COMPONENTS`."""
    lines = ["| component | file::symbol | reads | writes | cost | may decide |", "|---|---|---|---|---|---|"]
    for c in COMPONENTS:
        where = "; ".join(f"`{w}`" for w in c.where.split(";"))
        lines.append(f"| {c.id} | {where} | {c.reads} | {c.writes} | {c.cost} | {c.decides} |")
    return "\n".join(lines)
