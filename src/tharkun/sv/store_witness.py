"""Bounded store-derived witness shadow; never changes the deciding proof path."""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import time

from tharkun.facts import DecisionKey
from tharkun.solve import terms
from tharkun.solve.obligations import Invariant, witness
from tharkun.solve.sitekey import prestate_info


@dataclass(frozen=True)
class _RouteDecision:
    verdict: object
    component: str
    route: str
    source_sha256: str
    filename: str
    classes: tuple
    parents: tuple


def bind_route(verdict, route, source, filename, classes, *, parents=()):
    """Keep a successful prover's premises on its actual Verdict until finalization.

    This private metadata is outside ``detail`` and the Verdict's dataclass fields;
    stage proposals write no Evidence and do not change legacy serialization.
    """
    try:
        from tharkun.sv.engine import Verdict

        expected = {"no-site": "proof:no-site", "product": "symbolic:product",
                    "determinism": "deterministic", "stationary": "proof:stationary"}.get(route)
        if (not isinstance(verdict, Verdict) or verdict.verdict != "true"
                or (route == "producer" and not verdict.component.startswith("proof:facts:"))
                or (route != "producer" and (expected is None or verdict.component != expected))):
            return verdict
        parents = tuple(parents)
        kind = {"producer": "certified_invariant", "stationary": "call_target"}.get(route)
        if (any(len(parent) != 2 or parent[0] != kind for parent in parents)
                or (route == "producer" and not parents)):
            return verdict
        verdict._store_route = _RouteDecision(
            verdict, verdict.component, route, hashlib.sha256(source.encode("utf-8")).hexdigest(),
            filename, tuple(sorted(set(classes))), parents)
    except Exception:  # metadata loss cannot change a prover's decision
        pass
    return verdict


def record_route(table, verdict):
    """Record the actual final TRUE passed by scheduler.done, never a stage proposal."""
    try:
        from tharkun.sv.engine import Verdict

        if not isinstance(verdict, Verdict) or verdict.verdict != "true":
            return
        context = table.source_context
        if context is None:
            return
        decision = getattr(verdict, "_store_route", None)
        if decision is not None:
            if (not isinstance(decision, _RouteDecision) or decision.verdict is not verdict
                    or decision.component != verdict.component
                    or decision.source_sha256 != context.source_sha256
                    or decision.filename != context.location[0] or decision.classes != context.classes):
                return
            route, parents = decision.route, decision.parents
        elif verdict.component in ("proof:interval", "proof:shape"):
            route = "absint"
            parents = tuple(e.identity for e in table.site_record_evidence())
        elif verdict.component == "finite_tree":
            route, parents = "finite_tree", ()
        else:
            return
        by_path = {site.ast_path: site for site in context.sites.values()}
        sites = [replace(by_path[row.ast_path], exception=row.exception)
                 for row in table.rows(context.classes) if row.ast_path in by_path]
        table.publish_route(route, source_sha256=context.source_sha256, classes=context.classes,
                            producer=verdict.component, sites=sites, whole_program=True,
                            location=context.location, parents=parents)
    except Exception:  # additive evidence must not interfere with the old verdict
        return


def _source_binding(node, parents, *, fact_scope="site"):
    from tharkun.solve.contracts.safety import Binding

    if prestate_info(node, parents)[0] != fact_scope:
        return terms.Refused("fact_scope")
    if isinstance(node, ast.BinOp):
        receiver, args, operator = node.left, (node.right,), node.op
    elif isinstance(node, ast.AugAssign):
        receiver, args, operator = node.target, (node.value,), node.op
    elif isinstance(node, ast.UnaryOp):
        receiver, args, operator = node.operand, (), node.op
    elif isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        receiver, args, operator = node.left, tuple(node.comparators), node.ops[0]
    else:
        return terms.Refused("source_operation")

    def leaf(value):
        return terms.var(value.id) if isinstance(value, ast.Name) else terms.literal(value.value)

    operands = tuple(leaf(value) for value in (receiver, *args))
    refused = next((value for value in operands if isinstance(value, terms.Refused)), None)
    if refused is not None:
        return refused
    return Binding(operands[0], operands[1:], type(operator).__name__)


def _names(term):
    if term.op == "var":
        return {term.args[0]}
    return set().union(*(_names(arg) for arg in term.args if isinstance(arg, terms.Term)))


def _builtin_names(records):
    """Only the producer's original-source/visit proof supplies builtin identity."""
    allowed = {"type", "int", "bool", "float", "str"}
    names = set(allowed)
    for record in records:
        supplied = record.get("builtins")
        if (type(supplied) is not list or any(type(name) is not str for name in supplied)
                or supplied != sorted(set(supplied)) or not set(supplied) <= allowed):
            return terms.Refused("builtin_identity_unproved")
        names.intersection_update(supplied)
    return tuple(sorted(names))


def _render(term, builtin_names):
    rendered = terms.to_python(term, builtin_names=builtin_names)
    if isinstance(rendered, terms.Refused) and rendered.reason == "unverified or shadowed builtin type test":
        return terms.Refused("builtin_identity_unproved")
    return rendered


def _case_projection(cases, facts, query, builtin_names, deadline):
    from tharkun.solve.contracts.safety import case_safety

    tag_domain = terms.canon(terms.and_(*(fact for fact in facts if fact.op == "typeis")))
    for case in cases:
        if time.monotonic() >= deadline:
            return terms.Refused("shadow_budget")
        if terms.canon(case.domain) == tag_domain:
            term = case_safety(case)
            if isinstance(term, terms.Refused):
                return term
            return _render(term, builtin_names) if query(term) == "yes" else terms.Refused("case_not_singled_out")
    for case in cases:
        if time.monotonic() >= deadline:
            return terms.Refused("shadow_budget")
        term = case_safety(case)
        if isinstance(term, terms.Refused):
            return term
        if query(term) == "yes":
            return _render(term, builtin_names)
    consistent = []
    for case in cases:
        if time.monotonic() >= deadline:
            return terms.Refused("shadow_budget")
        # Only a proved exclusion may remove a case. Unknown stays possible.
        if query(terms.not_(case.domain)) == "yes":
            continue
        term = case_safety(case)
        rendered = term if isinstance(term, terms.Refused) else _render(term, builtin_names)
        if isinstance(rendered, terms.Refused):
            if rendered.reason == "builtin_identity_unproved":
                return rendered
            return terms.Refused("case_not_singled_out")
        consistent.append(term)
    if not consistent:
        return terms.Refused("case_not_singled_out")
    return _render(terms.or_(*consistent), builtin_names)


def project(table, *, specification="", deadline=None, max_sites=256, resource_limit=10000):
    """Project proved safety and actual whole-program coverage from one task's store.

    Query resources, a site cap and a cooperative monotonic deadline bound this
    always-on path in every thread. No process-global timer is installed.
    """
    from tharkun.solve.contracts.safety import admission, binding_from_record, cases, safety
    from tharkun.solve import terms_z3

    deadline = time.monotonic() + 0.1 if deadline is None else deadline
    reasons = Counter()
    metrics = {"checked_sites": 0, "safe_sites": 0, "established_sites": 0,
               "sites_established_and_rendered": 0, "proposed_sites": 0, "queries": 0}
    invariants = []
    context = table.source_context
    if context is None:
        return (), {**metrics, "reasons": {"store_only:source_context": 1}}
    nodes = context.sites
    records = table.site_record_evidence()
    paths, parents = {}, {}
    for node, site in nodes.items():
        if time.monotonic() >= deadline:
            return (), {**metrics, "reasons": {"store_only:shadow_budget": max(1, len(records))}}
        paths[site.ast_path] = node
        parents.update((child, node) for child in ast.iter_child_nodes(node))
    groups = {}
    safe_groups = set()
    for evidence in records:
        if time.monotonic() >= deadline:
            # An incomplete group could omit a later visit's weaker must-facts.
            reasons["store_only:shadow_budget"] += len(records)
            return (), {**metrics, "reasons": dict(reasons)}
        record = evidence.value
        identity = record.get("site")
        if (not isinstance(identity, (list, tuple)) or len(identity) != 3
                or any(not isinstance(part, str) for part in identity)):
            reasons["store_only:site_identity"] += 1
            continue
        identity = tuple(identity)
        if identity not in groups and record.get("status") == "safe":
            safe_groups.add(identity)
        elif record.get("status") != "safe":
            safe_groups.discard(identity)
        groups.setdefault(identity, []).append(evidence)
        metrics["safe_sites"] = len(safe_groups)

    def implies(facts, goal, identity):
        if time.monotonic() >= deadline:
            return terms.Refused("shadow_budget")
        text = terms.canon(goal)
        key = DecisionKey(context.source_sha256, 0, desired=text,
                          requires=json.dumps({"facts": sorted(terms.canon(f) for f in facts),
                                               "site": identity,
                                               "property": specification, "classes": context.classes,
                                               "resource_limit": resource_limit, "policy": "v1"}, sort_keys=True))

        def query():
            metrics["queries"] += 1
            return terms_z3.implies(facts, goal, resource_limit=resource_limit)

        answer = table.kb.lookup_or_run(key, "static:store-witness-implies", query)
        if answer == "unknown":
            reasons["store_only:implication_unknown"] += 1
        return answer

    for index, (identity, evidence) in enumerate(groups.items()):
        if index >= max_sites or time.monotonic() >= deadline:
            reasons["store_only:shadow_budget"] += len(groups) - index
            break
        node = paths.get(identity[1])
        if (identity[0] != context.source_sha256 or node is None or identity[2] not in context.classes):
            reasons["store_only:source_identity"] += 1
            continue
        site = replace(nodes[node], exception=identity[2])
        records = [item.value for item in evidence]
        first = records[0]
        if any(record.get("status") != "safe" for record in records):
            reasons["store_only:not_safe"] += 1
            continue
        # Missing scope belongs only to legacy direct-site records; explicit
        # unknown scopes and mixed visits cannot supply statement-entry facts.
        fact_scopes = [record.get("fact_scope", "site") for record in records]
        if any(type(scope) is not str or scope not in ("site", "statement_entry")
               or scope != fact_scopes[0] for scope in fact_scopes):
            reasons["store_only:fact_scope"] += 1
            continue
        contract = first.get("contract")
        if (not isinstance(contract, dict) or set(contract) != {"qualname", "exception"}
                or not isinstance(contract["qualname"], str) or contract["exception"] != identity[2]
                or any(record.get("contract") != contract for record in records)):
            reasons["store_only:contract"] += 1
            continue
        binding = binding_from_record(first.get("binding"))
        if isinstance(binding, terms.Refused):
            reasons[str(binding)] += 1
            continue
        canonical_binding = {"receiver": None if binding.receiver is None else terms.canon(binding.receiver),
                             "args": [terms.canon(arg) for arg in binding.args], "op": binding.op}
        if canonical_binding != first["binding"]:
            reasons["store_only:binding_canon"] += 1
            continue
        expected = _source_binding(node, parents, fact_scope=fact_scopes[0])
        if isinstance(expected, terms.Refused):
            reasons[str(expected)] += 1
            continue
        if binding != expected or any(record.get("binding") != first["binding"] for record in records):
            reasons["store_only:source_binding"] += 1
            continue
        qualname = ("protocol.comparison" if isinstance(node, ast.Compare) else
                    "protocol.unary_operator" if isinstance(node, ast.UnaryOp) else "protocol.binary_operator")
        if contract["qualname"] != qualname:
            reasons["store_only:source_contract"] += 1
            continue
        goal = safety(contract["qualname"], identity[2], binding)
        if isinstance(goal, terms.Refused):
            reasons[str(goal)] += 1
            continue
        canon = terms.canon(goal)
        if not isinstance(canon, str):
            reasons[str(canon)] += 1
            continue
        evidence_parents = tuple(item.identity for item in evidence)
        table.publish_safety(site, canon, producer="store_witness", parents=evidence_parents)
        metrics["proposed_sites"] += 1
        fact_sets = [record.get("facts") for record in records]
        if any(not isinstance(facts, list) or any(not isinstance(f, str) for f in facts) for facts in fact_sets):
            reasons["store_only:facts"] += 1
            continue
        fact_texts = sorted(set.intersection(*(set(facts) for facts in fact_sets)))
        facts = tuple(terms.from_canon(text) for text in fact_texts)
        operand_names = set().union(*(_names(t) for t in (binding.receiver, *binding.args) if t is not None))
        if (any(not isinstance(fact, terms.Term) or fact.sort != terms.BOOL
                or terms.canon(fact) != text or not _names(fact) <= operand_names
                for text, fact in zip(fact_texts, facts)) or len(operand_names) > 4):
            reasons["store_only:facts"] += 1
            continue
        metrics["checked_sites"] += 1
        inconsistent = implies(facts, terms.literal(False), identity)
        if inconsistent != "no":
            if inconsistent == "yes":
                reasons["store_only:facts_unsatisfiable"] += 1
            elif isinstance(inconsistent, terms.Refused):
                reasons[str(inconsistent)] += 1
            elif inconsistent != "unknown":
                reasons["store_only:implication_unknown"] += 1
            continue
        if implies(facts, goal, identity) != "yes":
            domain = admission(contract["qualname"], identity[2], binding)
            if time.monotonic() >= deadline:
                reasons["store_only:shadow_budget"] += 1
            elif isinstance(domain, terms.Refused) or implies(facts, domain, identity) != "yes":
                reasons["store_only:domain_not_established"] += 1
            else:
                reasons["store_only:proposed"] += 1
            continue
        table.publish_safety(site, canon, producer="store_witness", status="established", parents=evidence_parents)
        metrics["established_sites"] += 1
        metrics["proposed_sites"] -= 1
        builtin_names = _builtin_names(records)
        if isinstance(builtin_names, terms.Refused):
            reasons[str(builtin_names)] += 1
            continue
        choices = cases(contract["qualname"], identity[2], binding)
        rendered = choices if isinstance(choices, terms.Refused) else _case_projection(
            choices, facts, lambda term: implies(facts, term, identity), builtin_names, deadline)
        if isinstance(rendered, terms.Refused):
            reasons[str(rendered)] += 1
            continue
        if rendered == "True":
            reasons["store_only:trivial_safety"] += 1
            continue
        from tharkun.solve.predicates import InvalidPredicate, check_predicate
        try:
            check_predicate(rendered, {name: "object" for name in operand_names}, builtin_names=builtin_names)
        except InvalidPredicate as exc:
            reasons[str(terms.Refused(str(exc)))] += 1
            continue
        statement = prestate_info(node, parents)[1]
        kind = ("loop_invariant" if isinstance(statement, ast.While)
                and (statement.test is node or fact_scopes[0] == "statement_entry")
                else "location_invariant")
        invariants.append(Invariant(*site.location(), rendered, type=kind))
        metrics["sites_established_and_rendered"] += 1

    routes = table.proof_routes()
    for index, evidence in enumerate(routes):
        if time.monotonic() >= deadline:
            reasons["store_only:shadow_budget"] += len(routes) - index
            break
        route = evidence.value
        if (route["whole_program"] is True and route["source_sha256"] == context.source_sha256
                and set(context.classes) <= set(route["classes"]) and route["location"] == context.location):
            invariants.append(Invariant(*context.location, "True"))
    metrics["reasons"] = dict(reasons)
    return tuple(dict.fromkeys(invariants)), metrics


def shadow(table, source, filename, specification, *, directory=None, source_bytes=None, seconds=0.1):
    """Contain all projection/export failures after the legacy checker and exports."""
    deadline = time.monotonic() + max(0, seconds)
    detail = {}
    try:
        context = table.source_context
        if context is not None and (context.source_sha256 != hashlib.sha256(source.encode("utf-8")).hexdigest()
                                    or context.location[0] != filename):
            return {"store_reasons": {"store_only:source_identity": 1}}
        invariants, metrics = project(table, specification=specification, deadline=deadline)
        detail = {"store_" + key: value for key, value in metrics.items()}
        if invariants and directory is not None:
            if time.monotonic() < deadline:
                from tharkun.sv.proof import _write
                from tharkun.sv.witness import dumps
                doc = witness(source, filename, specification, invariants, source_bytes=source_bytes)
                text = dumps(doc)
                if time.monotonic() < deadline:
                    path = Path(directory) / "store-witness.yml"
                    _write(path, text)
                    detail["store_witness_path"] = str(path)
            if "store_witness_path" not in detail:
                reasons = detail.setdefault("store_reasons", {})
                reasons["store_only:shadow_budget"] = reasons.get("store_only:shadow_budget", 0) + 1
        return detail
    except Exception as exc:
        return {**detail, "store_error": f"{type(exc).__name__}: {exc}"}
