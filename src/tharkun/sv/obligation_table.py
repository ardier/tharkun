"""The live obligation table of a ``verify`` call ([internal ref] step 3a, docs/SV.md "The obligation table").

One table per task, kept IN the run's :class:`~tharkun.knowledge.KnowledgeBase` (owner, 2026-09-28:
"the obligation table must live in the KnowledgeBase"): a row per census obligation with its status
-- ``open``, ``discharged`` (by a whole-program proof) or ``violated`` (a witness that replayed in a
fresh child) -- and, next to the rows, everything components LEARNED, so that no component reads
another's result through a side channel:

========================  ==========  =====================================================
kind                      store       what it is
========================  ==========  =====================================================
``obligation``            Fact        the row status; ``certified`` only for a decision
``site_record``           Evidence    absint's per-site record (one generation per absint run)
``site_safety``           Fact        a canonical safety term, its status and evidence parents
``proof_route``           Evidence    the source, classes and sites a deciding prover covered
``site_outcome``          Evidence    a concolic stage's outcome for one harness line
``stage_ran``             Evidence    a concolic stage finished (even with no site)
``class_hints``           Evidence    what the first run observed about the program's classes
``module_constant``       Evidence    whether anything in the import graph rebinds a module-level name
``regex_pattern``         Evidence    a compiled regex of the import graph, statically folded
``call_target``           Evidence    what one call site of the entry module calls, and whether that is proven
========================  ==========  =====================================================

(``candidate_invariant``, ``input_sensitivity`` and ``loop_trip_count`` Facts and the producers'
``invariant_candidate`` / ``certified_invariant`` Evidence live in the same store; the full list with
writers and readers is ``tharkun.sv.contracts.KINDS``.)

Rules the code below keeps, and ``tests/test_sv_contracts.py`` pins:

* **Only a decision is certified.**  ``certified=True`` is written by exactly two methods here --
  :meth:`discharge` (after a whole-program proof) and :meth:`violate` (after a confirmed replay) --
  and nowhere else in ``tharkun.sv`` (MG's review bar: no third writer).  A hint, however
  strong-looking, is ``certified=False`` or plain Evidence.
* **Reads are explicit.**  A component asks the table for what it needs (:meth:`site_records`,
  :meth:`open_site_lines`, :meth:`class_hints`, :meth:`open_rows`); a proof component that must rely on a
  fact reads the certified kinds only (:meth:`status`, ``eregion_bridge.CERTIFIED_INVARIANT``).
* **A repeated question costs nothing.**  :meth:`since` hands a consumer only the evidence published
  after its own last look, so "run the static prover again" happens once per NEW piece of global
  information (docs/SV.md "What step 1 must not change"), never once per row.
"""
from __future__ import annotations

import ast
from collections.abc import Mapping
import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType

from tharkun.facts import Fact
from tharkun.knowledge import KnowledgeBase
from tharkun.solve.obligations import Invariant
from tharkun.solve.sitekey import SiteKey
from tharkun.sv.contracts import KINDS

OBLIGATION = "obligation"
SITE_RECORD = "site_record"
SITE_SAFETY = "site_safety"
PROOF_ROUTE = "proof_route"
SITE_OUTCOME = "site_outcome"
STAGE_RAN = "stage_ran"
CLASS_HINTS = "class_hints"
MODULE_CONSTANT = "module_constant"
REGEX_PATTERN = "regex_pattern"
CALL_TARGET = "call_target"

OPEN, DISCHARGED, VIOLATED = "open", "discharged", "violated"


def subject_of(o) -> str:
    """The row key of an ``Obligation``: ``"<ast_path>:<exception>"``."""
    return f"{o.ast_path}:{o.exception}"


@dataclass(frozen=True)
class SolvedView:
    """What one concolic stage left in the store: the harness lines it was asked about and what
    became of each (``reached`` / ``unsat`` / ``unreached`` / ``open``)."""

    stage: str
    ran: bool
    sites: dict = field(default_factory=dict)
    flags: tuple[str, ...] = ()

    @property
    def closed(self) -> bool:
        """Every site was solved to "no solution" (and there was at least one) -- the same rule as
        ``sv/concolic.py::Solved.closed``."""
        return (self.ran and not (set(self.flags) - {"file_effect"}) and bool(self.sites)
                and all(o in ("unsat", "unreached") for o in self.sites.values()))

    def open_lines(self) -> list:
        return [line for line, outcome in self.sites.items() if outcome == "open"]


@dataclass(frozen=True)
class SourceContext:
    """The original source map and task identity; context, never proof evidence."""

    sites: Mapping[ast.AST, SiteKey]
    source_sha256: str
    classes: tuple[str, ...]
    location: tuple[str, int, int, str]


class ObligationTable:
    def __init__(self, census, knowledge: KnowledgeBase | None = None) -> None:
        self.census = census
        self.kb = knowledge if knowledge is not None else KnowledgeBase()
        self._rows = tuple(o for o in census.obligations if o.ast_path != "$coverage")
        self._cursors: dict = {}
        self.source_context: SourceContext | None = None
        for o in self._rows:
            self.kb.add(Fact(kind=OBLIGATION, subject=subject_of(o), value=OPEN, source="census",
                             certified=False))

    def bind_source(self, sites, *, source_sha256: str, classes, location) -> None:
        """Bind the scheduler's once-built original AST map, including an empty census."""
        location = self._projection_location(location)
        if not isinstance(classes, (tuple, list, set, frozenset)):
            raise ValueError("source classes must be a collection of exception names")
        classes = tuple(sorted(set(classes)))
        sites = dict(sites)
        if (not isinstance(source_sha256, str) or not source_sha256 or not classes
                or any(not isinstance(exc, str) or not exc for exc in classes)
                or any(not isinstance(node, ast.AST) or not isinstance(site, SiteKey)
                       or site.source_sha256 != source_sha256 or site.location()[0] != location[0]
                       for node, site in sites.items())):
            raise ValueError("source context needs original nodes from this source and task")
        if self.source_context is not None:
            raise ValueError("source context is already bound")
        self.source_context = SourceContext(MappingProxyType(sites), source_sha256, classes, location)

    # --- the rows ------------------------------------------------------------------------
    def rows(self, classes=None) -> tuple:
        """Every per-site obligation (``$coverage`` rows are census-wide gaps, not sites), of
        ``classes`` when given."""
        return tuple(o for o in self._rows if classes is None or o.exception in classes)

    def status(self, o) -> str:
        """``open`` unless a CERTIFIED decision exists for the row; a violation beats a discharge
        (a replay is real whatever a proof said)."""
        decided = {f.value for f in self.kb.get(OBLIGATION, subject_of(o)) if f.certified}
        if VIOLATED in decided:
            return VIOLATED
        return DISCHARGED if DISCHARGED in decided else OPEN

    def open_rows(self, classes=None) -> tuple:
        return tuple(o for o in self.rows(classes) if self.status(o) == OPEN)

    # --- safety terms and the evidence that established them --------------------------------------
    def publish_safety(self, site: SiteKey, canon: str, *, producer: str,
                       status: str = "proposed", parents=()) -> Fact:
        """Store a canonical safety term; ``established`` requires named Evidence.

        This records a prover's claim, never an obligation decision. The reader
        must check the parents' coverage before relying on an established claim.
        """
        if (not isinstance(site, SiteKey)
                or any(not isinstance(value, str) or not value
                       for value in (site.exception, site.source_sha256, site.ast_path))
                or not isinstance(canon, str) or not canon.strip()):
            raise ValueError("safety needs a SiteKey with an exception and canonical term text")
        self._projection_location(site.location())
        if status not in ("proposed", "established"):
            raise ValueError("safety status must be proposed or established")
        parents = self._evidence_parents(parents)
        if status == "established" and not parents:
            raise ValueError("established safety needs evidence parents")
        fact = Fact(kind="site_safety", subject=site.row_subject(),
                    value={"site": site, "canon": canon, "status": status, "parents": parents},
                    source=producer, certified=False)
        self.kb.add(fact)
        return fact

    def safety(self, site: SiteKey | None = None) -> tuple[Fact, ...]:
        """All safety claims, including proposed ones; never conflated with decisions."""
        return tuple(fact for fact in self.kb if fact.kind == SITE_SAFETY
                     and (site is None or fact.value["site"].record_key() == site.record_key()))

    def publish_route(self, route: str, *, source_sha256: str, classes, producer: str,
                      sites=(), whole_program: bool = False, location=None, parents=()):
        """Publish the deciding prover's coverage, retaining its evidence derivation.

        The location is the whole-program route's witness projection, including
        when a property has no census sites. The v2 value counts actual site-record
        parents separately from census sites. A changed publication gets a new key.
        """
        if not isinstance(classes, (tuple, list, set, frozenset)):
            raise ValueError("route classes must be a collection of exception names")
        classes, sites = tuple(classes), tuple(sites)
        if (not isinstance(route, str) or not route or not isinstance(source_sha256, str) or not source_sha256
                or not classes or any(not isinstance(c, str) or not c for c in classes)):
            raise ValueError("route needs its name, source digest and exception classes")
        if type(whole_program) is not bool:
            raise ValueError("whole_program must be Boolean")
        if any(not isinstance(site, SiteKey) or site.source_sha256 != source_sha256
               or site.exception not in classes for site in sites):
            raise ValueError("route site is outside its source or exception coverage")
        if whole_program and location is None:
            raise ValueError("whole-program route needs a projection location")
        if location is not None:
            location = self._projection_location(location)
        parents = self._evidence_parents(parents)
        value = {"route": route, "source_sha256": source_sha256,
                 "classes": tuple(sorted(set(classes))),
                 "sites": tuple(sorted({site.record_key() for site in sites})),
                 "site_record_parent_count": sum(kind == SITE_RECORD for kind, _ in set(parents)),
                 "whole_program": whole_program, "location": location}
        key = "v2:" + hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()
        self.kb.publish("proof_route", key, value, producer=producer, parents=parents)
        return self.kb.evidence_item(PROOF_ROUTE, key)

    def proof_routes(self) -> tuple:
        return self.kb.evidence(kind=PROOF_ROUTE)

    @staticmethod
    def _projection_location(location):
        if not isinstance(location, (tuple, list)) or len(location) != 4:
            raise ValueError("projection location must contain file, line, column and function")
        Invariant(*location, "True")
        return tuple(location)

    def _evidence_parents(self, parents):
        if (not isinstance(parents, (tuple, list))
                or any(not isinstance(parent, (tuple, list)) or len(parent) != 2
                       or any(not isinstance(part, str) or not part for part in parent) for parent in parents)):
            raise ValueError("evidence parents must be kind/key pairs")
        parents = tuple(tuple(parent) for parent in parents)
        if any(self.kb.evidence_item(*parent) is None for parent in parents):
            raise ValueError("evidence parent is missing")
        return parents

    # --- decisions: the ONLY certified=True writers in tharkun.sv ---------------------------------
    def discharge(self, classes, component: str) -> None:
        """Called once, from ``sv/scheduler.py::run``'s ``done()``, only after a whole-program proof
        decided TRUE for every class in ``classes`` and the finalizer's independent check did not
        refuse it: every row of those classes is marked discharged, naming the proof.  Bookkeeping,
        never a second way to reach TRUE -- the proof already covers the whole program."""
        for o in self.rows(classes):
            self.kb.add(Fact(kind=OBLIGATION, subject=subject_of(o), value=DISCHARGED, source=component,
                             certified=True))

    def violate(self, exception: str, line, component: str, column=None) -> None:
        """Called once, from ``found()``, only AFTER a witness replayed FALSE in a fresh child.
        Matched by line and exception class against the census's own ``Invariant.line`` -- a heuristic
        (a row's AST-path identity is not reconstructed from a witness site), so more than one row can
        match (``a / b + c / d`` on one line: two ZeroDivisionError rows, one of which divided by zero).

        ``certified=True`` only when the match is UNAMBIGUOUS: exactly one row matches
        ``(line, exception)`` or, when the site carries a ``column``, exactly one matches
        ``(line, column, exception)`` (MG review, 2026-09-29: certifying every same-line row was
        unsound).  An ambiguous match still records that SOME row was violated, uncertified."""
        if line is None:
            return
        candidates = [o for o in self._rows if o.exception == exception and o.invariant.line == line]
        if column is not None:
            by_column = [o for o in candidates if o.invariant.column == column]
            if by_column:  # only narrow further if it finds a match; never drop to zero silently
                candidates = by_column
        certified = len(candidates) == 1
        for o in candidates:
            self.kb.add(Fact(kind=OBLIGATION, subject=subject_of(o), value=VIOLATED, source=component,
                             certified=certified))

    def _put(self, kind: str, key: str, value, producer: str) -> None:
        """Publish Evidence of a REGISTERED kind (``tharkun.sv.contracts.KINDS``); an unregistered one is
        a programming error, not a new kind by accident."""
        if KINDS.get(kind) is None or KINDS[kind].store != "evidence":
            raise ValueError(f"{kind!r} is not a registered Evidence kind")
        self.kb.publish(kind, key, value, producer=producer)

    # --- evidence: generations and cursors --------------------------------------------------------
    def _generation(self, kind: str, prefix: str = "") -> int:
        """How many publications of ``kind`` (under ``prefix``) have been made: the next one's number."""
        return len({e.key.split(":", 1)[0] for e in self.kb.evidence(kind=kind)
                    if e.key.startswith(prefix)})

    def since(self, consumer: str, kind: str) -> tuple:
        """The Evidence of ``kind`` published since ``consumer``'s last call for that kind, oldest
        first.  A consumer that acts only when this is non-empty acts once per new piece of
        information."""
        cursor = self._cursors.get((consumer, kind), 0)
        new_cursor, items = self.kb.evidence_since(cursor, kinds={kind})
        self._cursors[(consumer, kind)] = new_cursor
        return items

    # --- absint's site records ---------------------------------------------------------------------
    def publish_site_records(self, records, *, producer: str) -> None:
        """One absint run's per-site records (``solve/absint/__init__.py::AbsintResult.records``;
        docs/ABSINT.md "Site records").  Each non-empty run is one generation; a later run, seeing more
        (the observed ``class_hints``), supersedes the earlier one for every reader."""
        if not records:
            return
        generation = self._generation(SITE_RECORD) + 1
        for n, rec in enumerate(records):
            self._put(SITE_RECORD, f"g{generation}:{n}", dict(rec), producer=producer)

    def site_records(self):
        """The latest generation's records, or ``None`` when no absint run produced any (the value
        ``concolic.solve(extra_sites=...)`` and the seed stages already treat as "nothing known")."""
        items = self.site_record_evidence()
        return [e.value for e in items] if items else None

    def site_record_evidence(self) -> tuple:
        """Latest-generation records with their actual Evidence identities and parents."""
        items = self.kb.evidence(kind=SITE_RECORD)
        if not items:
            return ()
        latest = max(int(e.key.split(":", 1)[0][1:]) for e in items)
        return tuple(e for e in items if e.key.startswith(f"g{latest}:"))

    # --- a concolic stage's outcome ----------------------------------------------------------------
    def publish_solved(self, stage: str, solved, *, producer: str) -> None:
        """What a ``concolic.solve`` call left: that it finished and its per-line outcomes.  (The vectors
        it solved are replayed on the spot by the scheduler, the only FALSE source; a vector nobody reads
        later is not stored.)"""
        n = self._generation(STAGE_RAN, f"{stage}#") + 1
        tag = f"{stage}#{n}"
        self._put(STAGE_RAN, tag, {"ran": bool(solved.ran), "sites": len(solved.sites),
                                   "flags": tuple(getattr(solved, "flags", ()))}, producer=producer)
        for index, (line, outcome) in enumerate(solved.sites.items()):
            self._put(SITE_OUTCOME, f"{tag}:{index}", {"line": line, "outcome": outcome}, producer=producer)

    def solved(self, stage: str):
        """The latest :class:`SolvedView` of ``stage``, or ``None`` when that stage never ran."""
        ran = [e for e in self.kb.evidence(kind=STAGE_RAN) if e.key.startswith(f"{stage}#")]
        if not ran:
            return None
        tag = max((e.key for e in ran), key=lambda k: int(k.split("#", 1)[1]))
        sites = {e.value["line"]: e.value["outcome"] for e in self.kb.evidence(kind=SITE_OUTCOME)
                 if e.key.startswith(f"{tag}:")}
        value = self.kb.evidence_item(STAGE_RAN, tag).value
        return SolvedView(stage, bool(value["ran"]), sites, tuple(value.get("flags", ())))

    # --- what the first run observed ----------------------------------------------------------------
    def publish_class_hints(self, hints, *, producer: str) -> None:
        """The first run's observed class facts (``engine.Outcome.class_hints``): a hint for absint's
        second try, never a proof.  Empty hints publish nothing."""
        if not hints:
            return
        self._put(CLASS_HINTS, f"run{self._generation(CLASS_HINTS) + 1}:", dict(hints), producer=producer)

    def class_hints(self):
        items = self.kb.evidence(kind=CLASS_HINTS)
        return dict(items[-1].value) if items else None

    # --- what a static read of the import graph says (component "static facts (module namespace)") ----
    def publish_module_namespace(self, constants, regexes, *, producer: str) -> None:
        """The records ``solve/module_namespace.py::module_constants`` and ``regex_globals`` return for the
        task: one ``module_constant`` per module-level name (key ``<module>:<name>``) and one
        ``regex_pattern`` per compiled regex (key ``<module>:<line>:<col>``).  Both are hints about the
        SOURCE, never a verdict; a consumer that bakes a value reads ``proven`` and trusts nothing else."""
        for rec in constants:
            self._put(MODULE_CONSTANT, f"{rec['module']}:{rec['name']}", dict(rec), producer=producer)
        for rec in regexes:
            self._put(REGEX_PATTERN, f"{rec['module']}:{rec['line']}:{rec['col']}", dict(rec), producer=producer)

    def publish_call_targets(self, targets, *, producer: str) -> None:
        """The records ``solve/call_targets.py::call_targets`` returns for the task: one ``call_target`` per call
        site of the entry module (key ``<module>:<line>:<col>-<end_line>:<end_col>``: the calls of a chain
        ``a.f().g()`` start at one position).  A hint about the SOURCE, like the namespace
        records above: a consumer that acts on a callee reads ``proven`` and nothing else."""
        for rec in targets:
            key = f"{rec['module']}:{rec['line']}:{rec['col']}-{rec['end_line']}:{rec['end_col']}"
            self._put(CALL_TARGET, key, dict(rec), producer=producer)

    def call_targets(self) -> list:
        """Every published ``call_target`` record, in publication (source) order; ``[]`` when none was published."""
        return call_targets_in(self.kb)

    def proven_classes(self) -> list:
        """The names of the entry module's classes whose ``module_constant`` is proven (nothing in the import graph
        rebinds them): what the symbolic executor may let a program name denote when no call names it."""
        return sorted(e.value["name"] for e in self.kb.evidence(kind=MODULE_CONSTANT)
                      if e.value.get("proven") and e.value.get("origin") == "class")


def call_targets_in(knowledge: KnowledgeBase | None) -> list:
    """The legacy value-only view of the shared call-target Evidence reader.

    The symbolic executor uses this through ``ObligationTable.call_targets``;
    stationary retains the same inputs' identities for its final proof route.
    """
    return [e.value for e in call_target_evidence_in(knowledge)]


def call_target_evidence_in(knowledge: KnowledgeBase | None) -> tuple:
    """The same plan inputs with their identities, captured before stationary probing."""
    return tuple(knowledge.evidence(kind=CALL_TARGET)) if knowledge is not None else ()
