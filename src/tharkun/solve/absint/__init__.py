"""``tharkun.solve.absint`` -- a sound, whole-program static abstract interpreter.

``prove(source, exc_classes)`` decides, without running the program, whether
any execution of ``source`` (run as ``__main__``) can end with an uncaught
exception of one of ``exc_classes`` (the SV-COMP ``! uncaught(E)`` property).
It answers ``TRUE`` only with a proof: every operation of the program that can
raise one of those classes has a recorded *check* whose fact holds in the
proven invariant at that point.  Otherwise it answers ``UNKNOWN`` with the
first unproven check or the construct it does not model.  It never answers
FALSE -- a counterexample is the search's job.

Design and soundness argument: ``docs/ABSINT.md``.  The files:

* ``zone.py``    -- the difference-bound matrix (relational integer facts);
* ``values.py``  -- abstract values (a reduced product of type families,
  int intervals / symbolic bounds, float facts) and heap objects;
* ``state.py``   -- the abstract state and its join / widen / inclusion;
* ``interp.py``  -- the interpreter: statements, expressions, calls (inlined),
  loops (widening fixpoint + verified narrowing), builtin models;
* ``lemmas.py``  -- named numeric lemmas (e.g. the trig-pair bound) that the
  float domain applies, each cited by the checks it discharges.

Stdlib only, never imports or executes the target (``docs/LAYERS.md``).
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import time
from dataclasses import dataclass, field

PROPERTY_CLASSES = {
    "arithmetic": ("ZeroDivisionError", "FloatingPointError"),
    "data-lookup": ("KeyError", "IndexError"),
    "dynamic-typing": ("TypeError", "AttributeError"),
    "assertion-safety": ("AssertionError",),
}

#: the case-split retry's CPU allowance: at least this many seconds, or this multiple of the joined run
RETRY_FLOOR_SECONDS = 3.0
RETRY_MULTIPLE = 8.0

COMPONENT = {
    "arithmetic": "proof:interval",
    "data-lookup": "proof:interval",
    "dynamic-typing": "proof:shape",
    "assertion-safety": "proof:interval",
}


@dataclass
class AbsintResult:
    verdict: str  # "TRUE" | "UNKNOWN"
    component: str  # "proof:interval" | "proof:shape"
    sites: list = field(default_factory=list)  # (line, col, op, exc, fact)
    reason: str = ""
    lemmas: list = field(default_factory=list)
    seconds: float = 0.0
    checks_total: int = 0
    #: one record per (site, exception class) relevant to the property, in the shape
    #: shared with lanes A and B (docs/ABSINT.md "Site records")
    records: list = field(default_factory=list)
    #: True when the analysis ran to the end: then ``records`` covers EVERY operation
    #: of the program that can raise one of the classes (absint aborts on anything it
    #: does not model).  False after an abort: ``records`` is empty.
    complete: bool = False
    source_sha: str = ""
    invariants: list = field(default_factory=list)
    #: boolean draws this proof case-split (docs/ABSINT.md "Case splits on a boolean draw"); 0 for a joined run
    draw_splits: int = 0

    def explain(self) -> str:
        lines = [f"absint: {self.verdict} ({self.component})" + (f" -- {self.reason}" if self.reason else "")]
        if self.lemmas:
            lines.append("lemmas: " + ", ".join(self.lemmas))
        for line, col, op, exc, fact in self.sites:
            lines.append(f"  L{line}:{col} {op:<12} {exc:<30} {fact}")
        if self.draw_splits:
            lines.append(f"case splits on a boolean draw: {self.draw_splits}")
        lines.append(f"({len(self.sites)} relevant checks of {self.checks_total}, {self.seconds:.2f}s)")
        return "\n".join(lines)


def component_for(exc_classes) -> str:
    ex = set(exc_classes)
    if ex & {"TypeError", "AttributeError"}:
        return "proof:shape"
    return "proof:interval"


def prove(source: str, exc_classes, *, filename: str = "<program>", time_budget: float = 60.0,
          environment=None, class_hints=None, obligations=None,
          invariant_locations=(), split_draws=None) -> AbsintResult:
    """Prove absence for the audited ``PROPERTY_CLASSES``; unsupported class selections return UNKNOWN.

    ``split_draws`` (docs/ABSINT.md "Case splits on a boolean draw"): ``None`` runs the analysis joined and, only if
    that does not prove TRUE and the source draws a ``nondet_bool``, runs it once more with each such draw case-split
    on the time that is left (TRUE from either run, UNKNOWN as the joined run reported it otherwise); ``False`` never
    splits; ``True`` splits in the one run."""
    t0 = time.process_time()
    r = _prove_once(source, exc_classes, filename, time_budget, environment, class_hints, obligations,
                    invariant_locations, bool(split_draws))
    if (r.verdict == "TRUE" or split_draws is not None or "nondet_bool" not in source
            or r.reason.startswith(("environment:", "unsupported property classes", "syntax error"))
            or "time budget" in r.reason):
        return r
    spent = time.process_time() - t0
    # the retry never exceeds what the caller granted, and costs a bounded multiple of the joined run: it
    # re-runs the rest of a block per outcome, and an UNKNOWN that stays UNKNOWN must not take the time a
    # counterexample search needs
    left = min(time_budget - spent, max(RETRY_FLOOR_SECONDS, RETRY_MULTIPLE * spent))
    if left < 1.0:
        return r
    r2 = _prove_once(source, exc_classes, filename, left, environment, class_hints, obligations,
                     invariant_locations, True)
    if r2.verdict != "TRUE":
        return r
    r2.seconds = time.process_time() - t0
    return r2


def _prove_once(source, exc_classes, filename, time_budget, environment, class_hints, obligations,
                invariant_locations, split_draws) -> AbsintResult:
    from .interp import Analyzer, Unsupported
    from .state import Incompatible

    t0 = time.process_time()
    exc = tuple(exc_classes)
    comp = component_for(exc)
    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    unsupported = set(exc) - {cls for classes in PROPERTY_CLASSES.values() for cls in classes}
    if unsupported:
        return AbsintResult("UNKNOWN", comp, [], f"unsupported property classes: {', '.join(sorted(unsupported))}",
                            seconds=time.process_time() - t0, source_sha=sha)
    from tharkun.solve.environment import TaskEnvironment, proof_blocker
    blocker = proof_blocker(source, environment, filename=filename)
    if blocker:
        return AbsintResult("UNKNOWN", comp, [], f"environment: {blocker}",
                            seconds=time.process_time() - t0, source_sha=sha)
    try:
        from tharkun.solve.slicing import analysis_slice
        sliced = analysis_slice(source, exc, filename=filename, obligations=obligations)
        # [internal ref]: comprehensions are not modelled here; the exact lowering (solve/lowering) turns
        # them into the loops they are.  The analyzer interprets the lowered tree and reports on
        # the original program's sites and names (docs/LOWERING.md).  A comprehension the lowering
        # refuses stays in the tree, and the analyzer aborts on it as before.
        from tharkun.solve.lowering import lower_comprehensions
        lowered = lower_comprehensions(ast.parse(sliced.source, filename))
        declared_files = ()
        if isinstance(environment, TaskEnvironment) and environment.files and not filename.startswith("<"):
            primary = os.path.realpath(environment.files[0])
            if primary == os.path.realpath(filename):
                declared_files = environment.files
        an = Analyzer(sliced.source, filename, time_budget=time_budget,
                       class_hints=class_hints, invariant_locations=invariant_locations,
                       lowered=lowered if lowered.changed else None, site_source=source,
                       declared_files=declared_files, split_draws=split_draws)
        an.run()
    except (Unsupported, Incompatible) as e:
        return AbsintResult("UNKNOWN", comp, [], f"unsupported: {e}", seconds=time.process_time() - t0,
                            source_sha=sha)
    except RecursionError:
        return AbsintResult("UNKNOWN", comp, [], "analysis recursion depth", seconds=time.process_time() - t0,
                            source_sha=sha)
    except SyntaxError as e:
        return AbsintResult("UNKNOWN", comp, [], f"syntax error: {e}", seconds=time.process_time() - t0,
                            source_sha=sha)
    from tharkun.solve.obligations import Invariant
    from tharkun.solve.predicates import InvalidPredicate
    invariants = []
    for (loc, kind), parts in sorted(an.invariants.items()):
        if parts:
            try:
                invariants.append(Invariant(*loc, " and ".join(f"({p})" for p in sorted(parts)), type=kind))
            except InvalidPredicate:
                # Optional guidance exceeding the checked subset is not a native proof failure.
                pass
    records = []
    for (_l, _c, _op, e), rec in sorted(an.records.items(), key=lambda kv: kv[0]):
        if e in exc:
            records.append(dict(rec, source_sha=sha))
    sites = []
    failed = []
    lemmas = set()
    for c in sorted(an.checks.values(), key=lambda c: (c.line, c.col, c.op)):
        rel = c.excs & set(exc)
        if not rel:
            continue
        fact = "; ".join(c.facts)
        sites.append((c.line, c.col, c.op, ",".join(sorted(rel)), ("" if c.ok else "UNPROVEN: ") + fact))
        if not c.ok:
            failed.append(c)
        for f in c.facts:
            lemmas.update(re.findall(r"lemma ([A-Z_]+)", f))
    dt = time.process_time() - t0
    if failed:
        c = failed[0]
        return AbsintResult("UNKNOWN", comp, sites,
                            f"{len(failed)} unproven check(s); first: L{c.line}:{c.col} {c.op} "
                            f"{','.join(sorted(c.excs & set(exc)))}: {'; '.join(c.facts)}",
                            sorted(lemmas), dt, len(an.checks), records, True, sha, invariants)
    if not sites:
        # [internal ref] (soundness, defence in depth): a complete run with ZERO checks relevant to
        # `exc` is sound evidence of TRUE only when NOTHING in the source could have raised
        # one of those classes in the first place -- otherwise, "0 checks" can also mean
        # a construct absint does not fully understand quietly took the state to bottom
        # (the actual bug here: attr_load/call_method treated an unresolvable reflective
        # dunder, e.g. object.__repr__, as "this receiver contributes nothing" instead of
        # "unknown", making a live continuation look unreachable). site_census() is an
        # independent, coarser static scan (docs of tharkun.solve.census) that answers
        # "can ANY construct in this source raise one of these classes" without relying on
        # absint's own reachability reasoning; it must confirm there is truly no site before
        # 0 checks is allowed to mean TRUE.
        from tharkun.solve.census import site_census
        census = site_census(source, exc, filename=filename, declared_files=declared_files)
        # Narrower than "census disagrees": a census that found a genuine SYNTACTIC site
        # (e.g. a `raise KeyError` under a guard the census itself does not evaluate) is not
        # evidence of THIS bug class -- absint's own, more precise reachability reasoning is
        # allowed to prove that guard infeasible and legitimately record 0 checks (see
        # tests/test_sv_concolic.py::test_every_site_unsat_stops_the_search_early). The
        # signal this defence targets is specifically a census REFUSAL (site_census hit an
        # unmodelled construct too, e.g. "does not resolve: ... reflective attribute ...") --
        # when the ONE static scan that could have vetoed 0-checks-TRUE gives up for the same
        # kind of reason absint might have silently swallowed, TRUE is not safe to trust.
        if not census.no_site and "does not resolve" in census.reason:
            return AbsintResult("UNKNOWN", comp, sites,
                                f"0 checks, and the site census also could not resolve this "
                                f"program: {census.reason}",
                                sorted(lemmas), dt, len(an.checks), records, True, sha, invariants)
    return AbsintResult("TRUE", comp, sites, "", sorted(lemmas), dt, len(an.checks), records, True, sha, invariants,
                        draw_splits=an.splits_done)


__all__ = ["prove", "AbsintResult", "PROPERTY_CLASSES"]
