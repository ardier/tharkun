"""Census and witness-obligation views of the one ORIGINAL-source inventory.

``exception_sites`` owns physical discovery and the audited closed-program
coverage policy. This module keeps the historical proof and obligation APIs;
it has no separate exception-site walker or class table.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from tharkun.solve.contracts.adapters import obligations as _contracts
from tharkun.solve.pyresolve import Resolver, Top
from tharkun.solve.exception_sites import (
    MODULES, SITE_KINDS, _Refuse, _bound, _has_contract_rule,
    _InventoryScan as _Scan, _source_inventory,
)

@dataclass(frozen=True)
class Census:
    no_site: bool                                  # True: no wanted exception can escape the module
    reason: str = ""                               # why not (first site, or the refusal)
    sites: tuple[tuple[int, int, str], ...] = ()   # (line, col, what) -- the first few
    #: (line, col, Obligation) for sites a TESTED contract explains precisely ([internal ref] step 2).
    obligations: tuple[tuple[int, int, "_contracts.Obligation"], ...] = ()


def site_census(source: str, classes, filename: str = "<program>", limit: int = 8,
                *, declared_files=()) -> Census:
    """Can a complete inventory route certify no wanted module exit?

    The audited empty-site projection retains its whole-program certificate.
    A nonempty projection can instead use its complete scope-exit summary;
    partial typed facts and unknown dispatch never certify absence.
    """
    classes = tuple(classes)
    unknown = [c for c in classes if c not in SITE_KINDS]
    if unknown:
        return Census(False, f"no site table for {', '.join(unknown)}")
    inventory = _source_inventory(source, filename, declared_files=declared_files,
                                  census_classes=classes, include_conditions=True)
    if inventory.refusal is not None:
        return Census(False, inventory.refusal)
    if inventory.sites:
        from tharkun.solve.exception_exits import module_exit_absent

        if module_exit_absent(inventory, filename, classes):
            return Census(True, f"no uncaught {', '.join(classes)} at module exit (complete scope coverage)")
        line, col, what = inventory.sites[0]
        return Census(False, f"line {line}: {what}", tuple(inventory.sites[:limit]),
                      tuple(inventory.obligations[:limit]))
    return Census(True, f"no construct can raise {', '.join(classes)}")


def _is_literal(node: ast.AST) -> bool:
    """Is node a literal constant with no runtime execution or calls?"""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)) and isinstance(node.operand, ast.Constant):
        return isinstance(node.operand.value, (int, float, complex))
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(k is not None and _is_literal(k) and _is_literal(v) for k, v in zip(node.keys, node.values))
    return False


def _scan_unmodelled(tree: ast.AST, res: Resolver):
    """Yield (node, method_name, description) for unmodelled implicit calls / definition-time evals."""
    for node in ast.walk(tree):
        if not res.live(node):
            continue
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg is None and not (isinstance(kw.value, ast.Dict) and all(k is not None and _is_literal(k) for k in kw.value.keys)):
                    yield (kw.value, "keys", f"unmodelled implicit call: keys() from mapping unpacking (**{ast.unparse(kw.value)})")
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if k is None and not (isinstance(v, ast.Dict) and all(k2 is not None and _is_literal(k2) for k2 in v.keys)):
                    yield (v, "keys", f"unmodelled implicit call: keys() from dictionary display unpacking (**{ast.unparse(v)})")
        elif isinstance(node, ast.Starred):
            if not (isinstance(node.value, (ast.Tuple, ast.List)) and all(_is_literal(elt) for elt in node.value.elts)):
                yield (node.value, "__iter__", f"unmodelled implicit call: __iter__() from sequence unpacking (*{ast.unparse(node.value)})")
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                yield (item.context_expr, "__enter__", f"unmodelled implicit call: context manager ({ast.unparse(item.context_expr)})")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for dec in node.decorator_list:
                yield (dec, "decorator", f"unmodelled definition-time decorator: @{ast.unparse(dec)}")
            if getattr(node, "type_params", ()):
                yield (node, "type_params", "unmodelled definition-time type parameters")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if default is not None and not _is_literal(default):
                    yield (default, "default", f"unmodelled definition-time default: {ast.unparse(default)}")

def obligation_census(source: str, classes, filename: str = "<program>", *, declared_files=()):
    """Untruncated, class-specific witness predicates plus explicit coverage gaps.

    This reuses the census's audited coverage boundary; it is not a second proof
    of that boundary. The caller must check the predicates against source state.
    Unsupported safety rules remain open. Site IDs use AST paths, not line text.
    """
    import hashlib
    from tharkun.solve.obligations import Invariant, Obligation, ObligationCensus, source_points

    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    obligations, gaps = [], []

    def gap(exc, reason):
        gaps.append(f"{exc}: {reason}")
        obligations.append(Obligation(sha, "$coverage", exc,
            Invariant(filename, 1, 1, "<module>", "False"), reason))

    try:
        tree = ast.parse(source, filename)
    except (SyntaxError, ValueError) as e:
        for exc in sorted(set(classes)):
            gap(exc, f"parse: {e}")
        return ObligationCensus(tuple(obligations), tuple(gaps))
    points = source_points(tree, source, filename)
    parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    bound_builtins = {name for n in ast.walk(tree) for name in _bound(n)} & {"type", "len", "int", "float", "bool", "list", "tuple", "str", "bytes", "dict"}
    try:
        res = Resolver(tree, MODULES, filename=filename, _declared_files=declared_files)
    except (Top, _Refuse):
        res = None

    # shortcut: imported-file obligation identities omit filename; keep coverage open until identities are extended.
    has_imported_resolvers = bool(res is not None and len(res.all_resolvers()) > 1)

    for exc in sorted(set(classes)):
        if exc not in SITE_KINDS:
            gap(exc, f"no site table for {exc}")
            continue
        if has_imported_resolvers:
            gap(exc, "imported-module obligation locations are not represented")
        try:
            scan = _Scan(tree, (exc,), res or Resolver(
                tree, MODULES, filename=filename, _declared_files=declared_files))
            scan.run()
            if scan.refusal is not None:
                raise _Refuse(scan.refusal.removeprefix("does not resolve: "))
        except (Top, _Refuse) as e:
            gap(exc, f"coverage refused: {e}")
            continue
        seen = set()
        for node in scan.nodes:
            path, location = points[node]
            if path in seen:
                continue
            seen.add(path)
            value, reason = _safety_predicate(node, exc, parent)
            if bound_builtins:
                value, reason = "False", f"predicate builtins rebound: {', '.join(sorted(bound_builtins))}"
            from tharkun.solve.predicates import InvalidPredicate, parse_predicate
            try:
                parse_predicate(value)
            except InvalidPredicate as e:
                value, reason = "False", f"unsupported safety expression: {e}"
            obligations.append(Obligation(sha, path, exc, Invariant(*location, value), reason))

        if res is not None:
            for unmod_node, method, desc in _scan_unmodelled(tree, res):
                path, location = points.get(unmod_node, points.get(tree, ("$", (filename, 1, 1, "<module>"))))
                obligations.append(Obligation(sha, path, exc, Invariant(*location, "False"), desc))
                gaps.append(f"{exc}: {desc}")

    return ObligationCensus(tuple(obligations), tuple(gaps))


def _safety_predicate(node, exc, parent):
    """Sufficient safety conditions at a statement pre-state, never an eval string."""
    def atom(n):
        return isinstance(n, ast.Name) or (isinstance(n, ast.Constant) and type(n.value) in (int, float, bool))

    def direct(n):
        p = parent.get(n)
        return isinstance(p, (ast.Assign, ast.AnnAssign, ast.Return, ast.Expr)) and p.value is n

    if exc == "ZeroDivisionError" and isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
        a, b = (node.left, node.right) if isinstance(node, ast.BinOp) else (node.target, node.value)
        if atom(a) and atom(b) and (isinstance(node, ast.AugAssign) or direct(node)):
            left, right = ast.unparse(a), ast.unparse(b)
            numeric = f"(type({left}) is int or type({left}) is float or type({left}) is bool)"
            nonzero = (f"((type({right}) is int and {right} != 0) or "
                       f"(type({right}) is float and {right} != 0.0) or "
                       f"(type({right}) is bool and {right} != False))")
            return f"{numeric} and {nonzero}", ""
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and atom(node.slice) and direct(node):
        container, index = node.value.id, ast.unparse(node.slice)
        if exc == "IndexError":
            return " or ".join(f"(type({container}) is {kind} and type({index}) is int and -len({container}) <= {index} < len({container}))" for kind in ("list", "tuple", "str", "bytes")), ""
        if exc == "KeyError":
            return f"type({container}) is dict and {index} in {container}", "membership needs checked primitive key/element facts"
    if exc == "AssertionError" and isinstance(node, ast.Assert):
        from tharkun.solve.predicates import InvalidPredicate, parse_predicate
        value = ast.unparse(node.test)
        try:
            parse_predicate(value)
        except InvalidPredicate:
            pass
        else:
            return value, "assert test needs checked Boolean type, purity and totality"
    return "False", "no supported safety predicate for this operation/evaluation order"
