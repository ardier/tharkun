"""Run-derived nominations and Houdini pruning; the source checker alone certifies them."""
from __future__ import annotations

import ast
import itertools
import math

from tharkun.solve.obligations import source_points
from tharkun.solve.predicates import InvalidPredicate, TYPES, parse_predicate

NAME = "observations"
ASSERT_GAP = "assert test needs checked Boolean type, purity and totality"


def points(source, filename, census):
    if (not census.complete or not census.obligations
            or any(o.exception != "AssertionError" or o.gap not in ("", ASSERT_GAP) for o in census.obligations)):
        return []
    tree = ast.parse(source, filename)
    names = sorted({node.id for node in ast.walk(tree) if isinstance(node, ast.Name)})
    return [dict(zip(("file", "line", "column", "function"), location),
                 names=names,
                 type="loop_invariant" if isinstance(node, ast.While) else "location_invariant")
            for node, (_path, location) in source_points(tree, source, filename).items()
            if isinstance(node, (ast.Assert, ast.While))]


def _number(item):
    if item["kind"] == "int":
        value = item.get("value")
        return value if type(value) is int else int(value, 16) if isinstance(value, str) else None
    if item["kind"] == "float":
        value = float.fromhex(item["value"])
        return value if math.isfinite(value) else None
    return None


def _literal(value):
    return hex(value) if type(value) is int and value.bit_length() > 63 else repr(value)


def _holds(text, sample):
    """Evaluate checked predicate syntax on copied builtin scalars and reported lengths only."""
    try:
        tree = ast.Expression(parse_predicate(text))

        class Bind(ast.NodeTransformer):
            def visit_Call(self, node):
                if len(node.args) == 1 and isinstance(node.args[0], ast.Name):
                    item = sample[node.args[0].id]
                    if node.func.id == "len":
                        return ast.copy_location(ast.Constant(item["length"]), node)
                    if node.func.id == "type" and item["kind"] in TYPES:
                        return ast.copy_location(ast.Name(item["kind"], ast.Load()), node)
                raise ValueError("no observed pure call binding")

            def visit_Name(self, node):
                if node.id in TYPES:
                    return node
                item = sample[node.id]
                value = (float.fromhex(item["value"]) if item["kind"] == "float" else
                         _number(item) if item["kind"] == "int" else item["value"])
                if type(value) not in (int, bool, float, str):
                    raise ValueError("not a copied scalar")
                return ast.copy_location(ast.Constant(value), node)

        bound = ast.fix_missing_locations(Bind().visit(tree))
        builtin_types = {name: getattr(__import__("builtins"), name) for name in TYPES}
        return eval(compile(bound, "<observed-invariant>", "eval"), {"__builtins__": {}, **builtin_types}) is True
    except (InvalidPredicate, KeyError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return False


def _guesses(samples):
    names = set.intersection(*(set(sample) for sample in samples))
    numeric = {}
    for name in sorted(names):
        if not name.isidentifier():
            continue
        kinds = {sample[name]["kind"] for sample in samples}
        if kinds <= TYPES:
            yield " or ".join(f"type({name}) is {kind}" for kind in sorted(kinds))
        values = [_number(sample[name]) for sample in samples]
        if all(value is not None for value in values):
            yield f"{name} >= {_literal(min(values))}"
            yield f"{name} <= {_literal(max(values))}"
            if kinds == {"int"}:
                numeric[name] = values
        if all("length" in sample[name] for sample in samples):
            values = [sample[name]["length"] for sample in samples]
            yield f"len({name}) >= {min(values)}"
            yield f"len({name}) <= {max(values)}"
            numeric[f"len({name})"] = values
    for left, right in itertools.combinations(sorted(numeric), 2):
        xs, ys = numeric[left], numeric[right]
        other = next((i for i, x in enumerate(xs) if x != xs[0]), None)
        if other is None:
            yield f"({left}) - ({right}) == {_literal(xs[0] - ys[0])}"
        else:
            dx, dy = xs[other] - xs[0], ys[other] - ys[0]
            divisor = math.gcd(dx, dy)
            dx, dy = dx // divisor, dy // divisor
            yield f"{_literal(dy)} * ({left}) - {_literal(dx)} * ({right}) == {_literal(dy * xs[0] - dx * ys[0])}"


def produce(request):
    grouped = {}
    for observation in request.get("observations", ()):
        site = observation["site"]
        key = tuple(site[name] for name in ("file", "line", "column", "function", "type"))
        grouped.setdefault(key, []).extend(observation["samples"])
    obligations = {(row["file"], row["line"], row["column"], row["function"]): row["value"]
                   for row in request["obligations"]}
    for key, samples in grouped.items():
        if not samples:
            continue
        file, line, column, function, kind = key
        guesses = _guesses(samples)
        nomination = obligations.get(key[:4])
        if nomination:
            guesses = itertools.chain((nomination,), guesses)
        seen = set()
        for text in guesses:
            if text in seen or not all(_holds(text, sample) for sample in samples):
                continue
            seen.add(text)
            yield {"kind": "invariant_candidate", "type": kind, "value": text,
                   "site": {"file": file, "line": line, "column": column, "function": function},
                   "evidence": {"samples": len(samples), "origin": "native observations"}}
