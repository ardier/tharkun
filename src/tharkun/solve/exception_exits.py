"""Unordered scope-exit facts projected from the ORIGINAL physical inventory.

Source path facts are not a reachability certificate. Unknown substitutions
stay open; this module neither executes source nor performs an inverse walk.
"""
from __future__ import annotations

import ast
import builtins
from copy import deepcopy as _deepcopy


def deepcopy(value):
    """``copy.deepcopy`` for the JSON-shaped exit rows (dict/list of str/int/bool/None/float and frozen terms), without its per-call
    bookkeeping: 51 M calls in one census of ``bm_regex_v8_det`` (PERF, lane 1518).  Anything else is copied by ``copy.deepcopy``."""
    t = type(value)
    if t is dict:
        return {k: deepcopy(v) for k, v in value.items()}
    if t is list:
        return [deepcopy(v) for v in value]
    if value is None or t in (str, int, bool, float):
        return value
    return _deepcopy(value)


from tharkun.solve.contracts.registry import get
from tharkun.solve.contracts.types import ConstantArgsRaiseClaim
from tharkun.solve.exception_handlers import builtin_exception_chain, handler_match
from tharkun.solve.exception_dispatch import source_dispatch
from tharkun.solve.source_arguments import bind_source_call, immutable_default
from tharkun.solve.terms import canon, eval_term, from_canon, literal, not_, or_

_TRUE = canon(literal(True))
_SCALARS = frozenset({"int", "bool", "float", "str", "bytes", "NoneType"})
_IMPLICIT_NAMES = frozenset({"__name__", "__file__", "__package__", "__spec__", "__loader__",
                             "__doc__", "__annotations__", "__builtins__", "__class__"})


class _ScopeSummary:
    def __init__(self, inventory, scope, summarize):
        self.inventory, self.scope = inventory, scope
        self.summarize = summarize
        self.calls = {}
        self.index = inventory.indices[scope["file_name"]]
        self.parameters = set(scope["parameters"])
        self.inputs = {name: {"name": name, "kind": "parameter"} for name in scope["parameters"]}
        self.gaps = []
        self.effect_boundaries = []
        self.callback_free_sites = set()
        self.sites = [site for site in inventory.report["sites"] if site["scope"] == scope["id"]]
        self.nodes = [node for node, identity in self.index.node_scopes.items() if identity == scope["id"]]
        self.owner = next((node for node, identity in self.index.definitions.items()
                           if identity == scope["id"]), None)
        if self.owner is None:
            self.owner = next((node for node in self.nodes if isinstance(node, ast.Module)), None)
        self.by_span = {}
        for node in self.nodes:
            if hasattr(node, "lineno"):
                self.by_span.setdefault((node.lineno, node.col_offset, node.end_lineno, node.end_col_offset), []).append(node)

    def gap(self, reason):
        if reason not in self.gaps:
            self.gaps.append(reason)

    def node(self, site):
        nodes = self.by_span.get(tuple(site[k] for k in ("line", "col", "end_line", "end_col")), ())
        if site["kind"] == "raise":
            return next((n for n in nodes if isinstance(n, (ast.Raise, ast.Assert))), None)
        return next((n for n in nodes if isinstance(n, ast.expr)), nodes[0] if nodes else None)

    def expression(self, node, at, seen=()):
        if isinstance(node, ast.Constant):
            return ast.unparse(node)
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
            if isinstance(node.op, (ast.UAdd, ast.USub, ast.Invert)) and type(node.operand.value) in (int, float, bool):
                return ast.unparse(node)
        if isinstance(node, ast.Name):
            name = node.id
            bindings = self.index.bindings[self.scope["id"]].get(name, ())
            writes = [value for value in bindings if not isinstance(value, ast.arg)]
            if not writes and name in self.parameters:
                return name
            if name not in seen and len(writes) == 1 and isinstance(writes[0], ast.Name):
                assignment = self.index.parents.get(writes[0])
                if (isinstance(assignment, (ast.Assign, ast.AnnAssign)) and assignment.value is not None
                        and isinstance(getattr(self.owner, "body", None), list)
                        and assignment in self.owner.body and assignment.lineno < at.lineno):
                    return self.expression(assignment.value, assignment, (*seen, name))
                if name in self.parameters and getattr(assignment, "lineno", 0) >= at.lineno:
                    return name
            if not bindings:
                parent = self.index.scope_parents[self.scope["id"]]
                kind = "free" if name == "__class__" and self.scope["kind"] == "method" else "global"
                while parent is not None:
                    if self.index.scopes[parent]["kind"] == "class":
                        parent = self.index.scope_parents[parent]
                        continue
                    if name in self.index.bindings[parent]:
                        kind = "global" if self.index.scopes[parent]["kind"] == "module" else "free"
                        break
                    parent = self.index.scope_parents[parent]
                if name == "__class__" and self.scope["kind"] == "method":
                    kind = "free"
                self.inputs.setdefault(name, {"name": name, "kind": kind})
            self.gap("input binding/lifetime not translated: " + name)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in self.parameters:
            name = ast.unparse(node)
            self.inputs.setdefault(name, {"name": name, "kind": "receiver_field"})
            self.gap("receiver-field state requires a fixed object model: " + name)
        return None

    def text(self, text, at):
        if at is None:
            return None
        try:
            return self.expression(ast.parse(text, mode="eval").body, at)
        except (SyntaxError, ValueError, RecursionError):
            return None

    def paths(self, node):
        paths, child = [], node
        parent = self.index.parents.get(child)
        while parent is not None and self.index.node_scopes.get(parent) == self.scope["id"]:
            if isinstance(parent, ast.If) and (child in parent.body or child in parent.orelse):
                guard = parent.test
                if (isinstance(guard, ast.Compare) and len(guard.ops) == len(guard.comparators) == 1
                        and not isinstance(guard.ops[0], (ast.In, ast.NotIn))):
                    left = self.expression(guard.left, guard)
                    right = self.expression(guard.comparators[0], guard)
                    if left is not None and right is not None:
                        replacement = ast.Compare(left=ast.parse(left, mode="eval").body,
                                                  ops=guard.ops, comparators=[ast.parse(right, mode="eval").body])
                        paths.append({"expression": ast.unparse(replacement), "polarity": child in parent.body})
            child, parent = parent, self.index.parents.get(parent)
        return paths

    def base(self, site, exception):
        return {"class": exception, "class_id": site["class_ids"].get(exception),
                "condition": None, "condition_kind": "unknown", "domain": _TRUE,
                "bindings": {}, "operand_types": {}, "path_conditions": self.paths(self.node(site)),
                "sites": [site["id"]], "provenance": deepcopy(site["provenance"]),
                "maybe_caught": [], "unknown": True, "excluded_classes": [],
                "excluded_class_ids": [], "args": None, "args_kind": "unknown"}

    def explicit(self, site, node):
        provenance = next((p for p in site["provenance"] if p["kind"] == "explicit_raise"), None)
        if provenance is None:
            return None
        exception = "AssertionError" if isinstance(node, ast.Assert) else next(
            (name for name in site["classes"] if name != "*"), None)
        if exception is None:
            return None
        row = self.base(site, exception)
        row["provenance"] = [deepcopy(provenance)]
        row.update(condition=_TRUE, condition_kind="over_approximation", unknown=False)
        arguments = None
        if isinstance(node, ast.Raise) and provenance.get("legacy_constructor") and isinstance(node.exc, ast.Call):
            arguments = [self.expression(value, value) for value in node.exc.args]
        elif isinstance(node, ast.Assert):
            arguments = [] if node.msg is None else [self.expression(node.msg, node.msg)]
        if arguments is not None and all(value is not None for value in arguments):
            row.update(args=arguments, args_kind="exact")
        return row

    def cases(self, site, node):
        rows = []
        explicit = self.explicit(site, node)
        if explicit is not None:
            rows.append(explicit)
        for exception in site["classes"]:
            cases = site["conditions"].get(exception, ())
            if not cases and (explicit is None or explicit["class"] != exception):
                row = self.base(site, exception)
                row["reason"] = "no machine condition for the physical site"
                rows.append(row)
            for case in cases:
                if not case.get("unknown") and from_canon(case["raises"]) == literal(False):
                    continue
                row = self.base(site, exception)
                row["provenance"] = [deepcopy(p) for p in site["provenance"]
                                     if p["kind"] == "operation_type_contract" and p["contract"] == case["contract"]]
                if case.get("unknown"):
                    row["provenance"].append({"kind": "unknown", "contract": case["contract"],
                                              "reason": case.get("projection_refusal")})
                bindings = {name: self.text(expression, node) for name, expression in site["operands"].items()}
                translated = all(value is not None for value in bindings.values())
                row.update(operand_types=dict(case["operand_types"]),
                           excluded_classes=list(case.get("excluded_classes", ())))
                row["excluded_class_ids"] = [chain[0] for name in row["excluded_classes"]
                                             if (chain := builtin_exception_chain(name))]
                if translated:
                    row.update(bindings=bindings, domain=case["domain"], condition=case["condition"],
                               condition_kind="unknown" if case.get("unknown") else "over_approximation",
                               unknown=bool(case.get("unknown")))
                    if self.constant_false(row["domain"], bindings) or self.constant_false(row["condition"], bindings):
                        continue
                    if not row["unknown"]:
                        contract = get(case["contract"]) if case["contract"] is not None else None
                        claim = next((claim for claim in contract.raises if claim.exc == exception), None) if contract else None
                        if isinstance(claim, ConstantArgsRaiseClaim) and claim.tested:
                            row.update(args=[repr(value) for value in claim.args], args_kind="exact")
                else:
                    row["reason"] = "site operands cannot be expressed over scope inputs"
                    row["provenance"].append({"kind": "unknown", "reason": row["reason"],
                                              "original_domain": case["domain"],
                                              "original_condition": case["condition"]})
                if row["unknown"]:
                    row.setdefault("reason", case.get("projection_refusal") or "unknown condition")
                rows.append(row)
        if site["callee"] is not None:
            rows.extend(deepcopy(self.call(site, node)))
        else:
            dispatch = source_dispatch(self.index, self.scope, self.owner, node)
            for target, arguments, defaulted, class_id in dispatch:
                callee = self.index.definitions[target]
                summary = self.summarize(callee)
                if summary is None:
                    self.gap("recursive fixed-type dispatch is unsupported: " + callee)
                    continue
                mapping = {name: self.bound_expression(argument, node, name in defaulted)
                           for name, argument in arguments.items()}
                for row in self.translate(site, node, callee, summary, mapping):
                    row["provenance"].append({"kind": "fixed_type_dispatch", "class_id": class_id,
                                              "method": target.name, "callee": callee})
                    rows.append(row)
        return rows

    @staticmethod
    def substitute(expression, mapping):
        if expression is None:
            return None
        tree = ast.parse(expression, mode="eval")
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        if any(mapping.get(name) is None for name in names):
            return None

        class Replace(ast.NodeTransformer):
            def visit_Name(self, node):
                return ast.parse(mapping[node.id], mode="eval").body

        return ast.unparse(Replace().visit(tree).body)

    def bound_expression(self, value, node, defaulted=False):
        if defaulted:
            return immutable_default(value)
        if value is None:
            return None
        if isinstance(value, tuple):
            values = [self.bound_expression(element, node) for element in value]
            return "(" + ", ".join(values) + ("," if len(values) == 1 else "") + ")" if all(
                element is not None for element in values) else None
        if isinstance(value, dict):
            values = {name: self.bound_expression(element, node) for name, element in value.items()}
            return "{" + ", ".join(repr(name) + ": " + element for name, element in values.items()) + "}" if all(
                element is not None for element in values.values()) else None
        return self.expression(value, node)

    def call(self, site, node):
        if site["id"] in self.calls:
            return self.calls[site["id"]]
        callee = site["callee"]
        index = self.inventory.indices[self.inventory.report["scopes"][callee]["file_name"]]
        target = next((n for n, identity in index.definitions.items() if identity == callee), None)
        reason = None
        arguments, defaulted = None, frozenset()
        constructor = (isinstance(node.func, ast.Name)
                       and isinstance(self.index.resolve(node.func.id, self.scope["id"]), ast.ClassDef))
        if not isinstance(target, ast.FunctionDef) or constructor:
            reason = "source-call signature/descriptor binding is unsupported: " + callee
        else:
            arguments, defaulted, refusal = bind_source_call(target, node)
            if refusal is not None:
                reason = "source-call signature/descriptor binding is unsupported: " + callee + ": " + refusal
        summary = self.summarize(callee) if reason is None else None
        if reason is None and summary is None:
            reason = "recursion or scope dependency depth is unsupported: " + callee
        if reason is not None:
            self.gap(reason)
            row = self.base(site, "*")
            row.update(class_id=None, reason=reason)
            self.calls[site["id"]] = [row]
            return [row]

        mapping = {name: self.bound_expression(argument, node, name in defaulted)
                   for name, argument in arguments.items()}
        if any(value is None for value in mapping.values()):
            self.gap("source-call arguments cannot be expressed over own inputs: " + callee)
        # A global/free callable is an input, not a certificate that its source
        # definition is the object found at runtime. Only a prior local def is fixed.
        body = getattr(self.owner, "body", None)
        local = (index is self.index and isinstance(body, list) and target in body
                 and target.lineno < node.lineno and not self.index.opaque_bindings)
        if not local:
            self.expression(node.func.value if isinstance(node.func, ast.Attribute) else node.func, node)
            self.gap("source-call binding/lifetime is not certified: " + callee)
        elif summary["complete"] and all(value is not None for value in mapping.values()):
            self.callback_free_sites.add(site["id"])
        rows = self.translate(site, node, callee, summary, mapping)
        self.calls[site["id"]] = rows
        return rows

    def translate(self, site, node, callee, summary, mapping):
        for gap in summary["gaps"]:
            # Effects survive even a caller's bare handler. Broadening here is
            # deliberate until state transport can discharge the callee gap.
            self.gap("source-callee coverage/effects: " + callee + ": " + gap["reason"])
        rows = []
        for original in summary["exits"]:
            row = deepcopy(original)
            row["sites"] = list(dict.fromkeys([site["id"], *row["sites"]]))
            row["provenance"].append({"kind": "source_call", "callee": callee, "site": site["id"]})
            bindings = {name: self.substitute(value, mapping) for name, value in row["bindings"].items()}
            if any(value is None for value in bindings.values()):
                row.update(bindings={}, domain=_TRUE, condition=None, condition_kind="unknown", unknown=True,
                           reason="callee bindings cannot be translated to caller inputs")
            else:
                row["bindings"] = bindings
                if self.constant_false(row["domain"], bindings) or self.constant_false(row["condition"], bindings):
                    continue
            arguments = ([self.substitute(value, mapping) for value in row["args"]]
                         if row["args"] is not None else None)
            row["args"] = arguments if arguments is not None and all(value is not None for value in arguments) else None
            row["args_kind"] = "exact" if row["args"] is not None else "unknown"
            row["path_conditions"] = self.paths(node) + [
                {"expression": expression, "polarity": path["polarity"]}
                for path in row["path_conditions"]
                if (expression := self.substitute(path["expression"], mapping)) is not None]
            rows.append(row)
        return rows

    @staticmethod
    def constant_false(text, bindings):
        if text is None:
            return False
        try:
            values = {name: ast.literal_eval(expression) for name, expression in bindings.items()}
        except (ValueError, SyntaxError, TypeError, RecursionError):
            return from_canon(text) == literal(False)
        result = eval_term(from_canon(text), values)
        return result is False

    def stable_handler_binding(self, site, row):
        if self.inventory.refusal is not None or self.index.opaque_bindings or self.index.opaque_class_identities:
            return False
        for other in self.sites:
            if other["id"] in self.callback_free_sites:
                continue
            contracts = [p["contract"] for p in other["provenance"] if p["kind"] == "operation_type_contract"]
            if contracts and all((contract := get(name)) is not None and not contract.effects.may_execute_callback
                                 for name in contracts):
                continue
            if other is site:
                if row["unknown"]:
                    return False
                node = self.node(site)
                if isinstance(node, ast.Raise):
                    if not next((p.get("legacy_constructor") for p in site["provenance"]
                                 if p["kind"] == "explicit_raise"), False):
                        return False
                elif isinstance(node, ast.Assert):
                    if not isinstance(node.test, ast.Constant):
                        return False
                elif not row["operand_types"] or any(kind not in _SCALARS for kind in row["operand_types"].values()):
                    return False
                continue
            if any(p["kind"] == "explicit_raise" and p.get("legacy_constructor") for p in other["provenance"]):
                continue
            return False
        return True

    def route(self, site, row):
        chain = builtin_exception_chain(row["class"]) if row["class_id"] and row["class_id"].startswith("builtins.") else ()
        if not chain:
            chain = next((tuple(p["class_chain_ids"]) for p in row["provenance"]
                          if p.get("class_chain_ids") and p["class_chain_ids"][0] == row["class_id"]), ())
        for handler in site["handlers"]:
            if handler["unresolved"]:
                self.gap("handler lookup/validation is unresolved: " + str(handler["handler_span"]))
            relation = handler_match(row["class_id"], chain, handler, class_chain_complete=bool(chain))
            if relation == "caught" and not handler["bare"] and not self.stable_handler_binding(site, row):
                relation = "maybe_caught"
            if relation == "caught" and not row["maybe_caught"]:
                return None
            if relation == "maybe_caught":
                row["maybe_caught"].append(deepcopy(handler))
        return row

    def effects(self, site):
        """Do not erase state uncertainty when a local handler catches a callback."""
        names = [p["contract"] for p in site["provenance"] if p["kind"] == "operation_type_contract"]
        contracts = [get(name) for name in names]
        if not contracts:
            if site["callee"] is not None:
                self.call(site, self.node(site))
                return
            if isinstance(self.node(site), ast.Call) or any(p["kind"] == "unknown" for p in site["provenance"]):
                self.gap("unmodelled effects at " + site["id"])
            else:
                self.callback_free_sites.add(site["id"])
            return
        if any(contract is None for contract in contracts):
            self.gap("missing effect contract at " + site["id"])
            return
        effects = [contract.effects for contract in contracts]
        if any(effect.mutates_receiver or effect.mutates_args or effect.io or effect.randomness
               or effect.clock_or_env or not effect.deterministic for effect in effects):
            self.gap("state/world effects require a model at " + site["id"])
            return
        if not any(effect.may_execute_callback for effect in effects):
            self.callback_free_sites.add(site["id"])
            return
        node = self.node(site)
        if not isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.If, ast.IfExp, ast.While, ast.Assert, ast.Attribute, ast.Subscript)):
            self.gap("callback effects require a model at " + site["id"])
            return
        bindings = {name: self.text(expression, node) for name, expression in site["operands"].items()}
        if not all(value is not None for value in bindings.values()):
            self.gap("callback operand state is untranslated at " + site["id"])
            return
        # Primitive operator dispatch cannot invoke an unknown user dunder.
        # The guards themselves still come only from the existing case table.
        safe = {case["domain"] for cases in site["conditions"].values() for case in cases
                if not case.get("unknown") and case["operand_types"]
                and set(case["operand_types"]) == set(bindings)
                and all(kind in _SCALARS for kind in case["operand_types"].values())}
        domain = canon(not_(or_(*(from_canon(value) for value in sorted(safe))))) if safe else _TRUE
        if self.constant_false(domain, bindings):
            self.callback_free_sites.add(site["id"])
            return
        reason = "callback effects may invalidate later input bindings"
        self.effect_boundaries.append({
            "class": "*", "class_id": None, "condition": None, "condition_kind": "unknown",
            "domain": domain, "bindings": bindings, "operand_types": {}, "path_conditions": [],
            "sites": [other["id"] for other in self.sites],
            "provenance": [{"kind": "unknown", "reason": reason, "source_site": site["id"]}],
            "maybe_caught": [], "unknown": True, "reason": reason,
            "excluded_classes": [], "excluded_class_ids": [], "args": None, "args_kind": "unknown"})

    def build(self):
        statements = (ast.Return, ast.Raise, ast.Assert, ast.If, ast.Try, ast.Assign,
                      ast.AnnAssign, ast.Expr, ast.Pass, ast.FunctionDef)
        for name, bindings in self.index.bindings[self.scope["id"]].items():
            if len(bindings) > 1:
                self.gap("reference finalization on replacement is deferred: " + name)
        if self.scope["kind"] == "class":
            self.gap("class namespace execution is not modeled")
        if isinstance(self.owner, ast.AsyncFunctionDef):
            self.gap("async scope execution is deferred")
        for node in self.nodes:
            if isinstance(node, ast.stmt) and not isinstance(node, statements):
                self.gap("unsupported scope execution shape: " + type(node).__name__)
            if isinstance(node, ast.Try) and node.finalbody:
                self.gap("finally outcome is deferred")
            if isinstance(node, ast.Raise) and (node.exc is None or node.cause is not None):
                self.gap("re-raise or exception chaining is deferred")
            if isinstance(node, (ast.Yield, ast.YieldFrom, ast.GeneratorExp, ast.Await)):
                self.gap(type(node).__name__ + " outcome is deferred")
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                parent = self.index.parents.get(node)
                if isinstance(parent, ast.Call) and parent.func is node:
                    continue  # the call record covers target lookup and dispatch
                if isinstance(parent, ast.ExceptHandler):
                    continue  # unresolved handler identities remain maybe caught
                value = self.index.resolve(node.id, self.scope["id"])
                if node.id not in _IMPLICIT_NAMES and value is not None and value is getattr(builtins, node.id, None):
                    continue
                self.expression(node, node)
        rows = []
        for site in self.sites:
            self.effects(site)
        for site in self.sites:
            for row in self.cases(site, self.node(site)):
                if (routed := self.route(site, row)) is not None:
                    rows.append(routed)
        rows.extend(self.effect_boundaries)
        if self.inventory.refusal is not None:
            self.gap("source coverage refused: " + self.inventory.refusal)
        if self.gaps:
            rows.append({"class": "*", "class_id": None, "condition": None, "condition_kind": "unknown",
                         "domain": _TRUE, "bindings": {}, "operand_types": {}, "path_conditions": [],
                         "sites": [], "provenance": [{"kind": "unknown", "reason": reason} for reason in self.gaps],
                         "maybe_caught": [], "unknown": True, "reason": "; ".join(self.gaps),
                         "excluded_classes": [], "excluded_class_ids": [], "args": None, "args_kind": "unknown"})
        return {"scope": self.scope["id"], "inputs": list(self.inputs.values()), "exits": rows,
                "complete": not self.gaps and not self.effect_boundaries,
                "gaps": [{"reason": reason} for reason in self.gaps] + [
                    {key: boundary[key] for key in ("reason", "domain", "bindings", "sites")}
                    for boundary in self.effect_boundaries]}


def exit_summaries(inventory):
    """Project every indexed scope; an empty exceptional set says nothing about termination."""
    summaries, active = {}, set()

    def summarize(identity):
        if identity in active or len(active) >= 64:
            return None
        if identity not in summaries:
            active.add(identity)
            try:
                summaries[identity] = _ScopeSummary(inventory, inventory.report["scopes"][identity], summarize).build()
            finally:
                active.remove(identity)
        return summaries[identity]

    for identity in inventory.report["scopes"]:
        summarize(identity)
    return summaries


def module_exit_absent(inventory, filename, classes):
    """A complete module summary must itself cover every potentially wanted exit.

    No partial route is combined with the audited census. This does not infer
    normal feasibility, evaluate source predicates, or equate unknown with false.
    """
    summary = exit_summaries(inventory).get(str(filename) + "::<module>")
    if summary is None or not summary["complete"]:
        return False
    for row in summary["exits"]:
        remaining = list(classes)
        if row["class"] == "*":
            remaining = [name for name in remaining
                         if not set(row["excluded_class_ids"]).intersection(builtin_exception_chain(name))]
        if not remaining:
            continue
        chain = builtin_exception_chain(row["class"]) if row["class_id"] and row["class_id"].startswith("builtins.") else ()
        if not chain:
            chain = next((tuple(p["class_chain_ids"]) for p in row["provenance"]
                          if p.get("class_chain_ids") and p["class_chain_ids"][0] == row["class_id"]), ())
        wanted = {"bare": False, "unresolved": False, "class_ids": ["builtins." + name for name in remaining]}
        if handler_match(row["class_id"], chain, wanted, class_chain_complete=bool(chain)) != "not_caught":
            return False
    return True
