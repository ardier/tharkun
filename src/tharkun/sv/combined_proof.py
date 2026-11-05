"""Fresh per-check composition of complete absint and an exact goal slice.

Table records and search UNSAT results are eligibility hints only. Every cover
is rebuilt from the original source; floats stay with absint's binary64 model.
"""
from __future__ import annotations

import ast
import hashlib
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Coverage:
    source_sha256: str
    absint_sites: frozenset[str]
    product_sites: frozenset[str]


def cover(source, filename, classes, *, census=None, seconds=3.0):
    """Recompute the whole original cover without accepting producer claims."""
    if tuple(classes) != ("AssertionError",) or seconds <= 0:
        return None
    from tharkun.solve.absint import prove as absint_prove
    from tharkun.solve.census import obligation_census
    from tharkun.solve.product_slice import slice_product_goals
    from tharkun.sv import product_proof

    deadline = time.monotonic() + seconds
    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    census = obligation_census(source, classes, filename) if census is None else census
    if (not census.complete or not census.obligations
            or any(o.source_sha256 != sha or o.exception != "AssertionError" for o in census.obligations)):
        return None
    required = frozenset(o.ast_path for o in census.obligations)
    # Refuse outside the projection's effect boundary before repeating absint.
    # These are only erasure nominations until their fresh safe checks below.
    projected = slice_product_goals(source, filename, required, deadline=deadline)
    if projected is None:
        return None
    result = absint_prove(source, classes, filename=filename, time_budget=max(0.001, deadline - time.monotonic()))
    if not result.complete or result.source_sha != sha:
        return None
    groups = {}
    for record in result.records:
        site = record.get("site")
        if (record.get("kind") == "assert" and record.get("source_sha") == sha
                and isinstance(site, (list, tuple)) and len(site) == 3
                and site[0] == sha and site[2] == "AssertionError"):
            groups.setdefault(site[1], []).append(record["status"])
    safe = frozenset(path for path, states in groups.items() if all(s == "safe" for s in states))
    if not safe or required <= safe or time.monotonic() >= deadline:
        return None
    if (not projected.erased <= safe
            or not required <= safe | projected.retained or time.monotonic() >= deadline):
        return None
    proof = product_proof.prove(projected.source, classes, program=filename,
                                timeout_ms=max(1, int(1000 * (deadline - time.monotonic()))), _proof_only=True)
    if proof is None or proof.verdict != "true" or time.monotonic() >= deadline:
        return None
    return Coverage(sha, safe & required, projected.retained)


def prove(source, filename, classes, table, *, seconds=3.0):
    """An eligible mixed cover reaches the ordinary canary/finalizer."""
    records = table.site_records() or ()
    states = {r.get("status") for r in records if r.get("kind") == "assert"}
    if "safe" not in states or not states - {"safe"}:
        return None
    coverage = cover(source, filename, classes, seconds=seconds)
    if coverage is None:
        return None
    from tharkun.solve.obligations import Invariant, source_points
    from tharkun.sv.engine import Verdict
    from tharkun.sv.store_witness import bind_route

    tree = ast.parse(source, filename)
    location = source_points(tree, source, filename)[tree.body[0]][1]
    table.kb.publish("check_coverage", coverage.source_sha256,
                                {"source_sha256": coverage.source_sha256,
                                 "absint_sites": tuple(sorted(coverage.absint_sites)),
                                 "product_sites": tuple(sorted(coverage.product_sites))},
                                producer="proof:combined")
    verdict = Verdict("true", reason="complete source-bound absint and exact product-slice cover",
                      component="proof:combined", detail={
                          "absint_sites": sorted(coverage.absint_sites),
                          "product_sites": sorted(coverage.product_sites),
                          "candidate_invariants": [Invariant(*location, "True")]})
    return bind_route(verdict, "combined", source, filename, classes,
                      parents=(("check_coverage", coverage.source_sha256),))
