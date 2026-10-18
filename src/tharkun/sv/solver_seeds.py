"""Typed boundary and composed-constant candidates for SV draw enumeration.

Ideas adapted from ardier/fuzzy-mutation (MIT): comparison dictionaries,
boundary draws, and bounded composition. Values are proposals, replayed by SV.
"""

import ast
import math

from tharkun.sv.guard_values import _Scope


def _literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        return None


def seed_values(program: str, trick: str, records=None) -> dict[str, list]:
    try:
        from tharkun.sv.engine import read_source

        tree = ast.parse(read_source(program))
    except (OSError, SyntaxError, ValueError):
        return {}
    out: dict[str, list] = {}
    seen = set()

    def add(kind, value):
        if kind not in ("int", "float", "str") or len(out.get(kind, [])) >= 64:
            return
        if kind == "int" and (type(value) is not int or abs(value) >= 2**64):
            return
        if kind == "float" and type(value) not in (int, float):
            return
        if kind == "str" and (type(value) is not str or len(value) > 128):
            return
        value = float(value) if kind == "float" else value
        key = (kind, repr(value))
        if key not in seen:
            seen.add(key)
            out.setdefault(kind, []).append(value)

    ignored = set()
    for node in ast.walk(tree):
        annotation = (node.annotation if isinstance(node, (ast.AnnAssign, ast.arg)) else
                      node.returns if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else
                      node.value if isinstance(node, getattr(ast, "TypeAlias", ())) else
                      node if isinstance(node, getattr(ast, "type_param", ())) else None)
        if annotation is not None:
            ignored.update(id(n) for n in ast.walk(annotation))
    bodies = [tree.body] + [n.body for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for body in bodies:
        scope = _Scope(body)
        if not scope.draws:
            continue
        comparisons = [n for st in body for n in ast.walk(st)
                       if isinstance(n, ast.Compare) and id(n) not in ignored]
        if trick == "comparison":
            for cmp in comparisons:
                if len(cmp.ops) != 1:
                    continue
                left, right = cmp.left, cmp.comparators[0]
                for expr, other in ((left, right), (right, left)):
                    names = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}
                    kinds = {scope.draws[n] for n in names & scope.draws.keys()}
                    if not kinds and isinstance(expr, ast.Name) and expr.id in scope.defs:
                        names = {n.id for n in ast.walk(scope.defs[expr.id][1]) if isinstance(n, ast.Name)}
                        kinds = {scope.draws[n] for n in names & scope.draws.keys()}
                    if len(kinds) == 1:
                        add(next(iter(kinds)), _literal(other))
        elif trick == "boundaries":
            for cmp in comparisons:
                for side in [cmp.left, *cmp.comparators]:
                    if not isinstance(side, ast.Name) or side.id not in scope.draws:
                        continue
                    kind = scope.draws[side.id]
                    for other in [cmp.left, *cmp.comparators]:
                        if other is side:
                            continue
                        bound = _literal(other)
                        if kind == "int" and type(bound) is int:
                            for v in (bound - 1, bound, bound + 1, 0):
                                add(kind, v)
                        elif kind == "float" and type(bound) in (int, float):
                            x = float(bound)
                            for v in (math.nextafter(x, -math.inf), x, math.nextafter(x, math.inf), 0.0):
                                add(kind, v)
            for kind in set(scope.draws.values()):
                if kind == "str":
                    add(kind, "")
                elif kind in ("int", "float"):
                    add(kind, 0)
        elif trick == "composition":
            lengths = {_literal(n.comparators[0]) for cmp in comparisons
                       for n in [cmp] if len(n.comparators) == 1 and isinstance(n.left, ast.Call)
                       and isinstance(n.left.func, ast.Name) and n.left.func.id == "len"}
            lengths = {n for n in lengths if type(n) is int and 0 <= n <= 64}
            for cmp in comparisons:
                if len(cmp.ops) != 1 or not isinstance(cmp.ops[0], ast.Eq):
                    continue
                for expr, other in ((cmp.left, cmp.comparators[0]), (cmp.comparators[0], cmp.left)):
                    target = _literal(other)
                    if type(target) is not str:
                        continue
                    if isinstance(expr, ast.Name) and expr.id in scope.defs:
                        expr = scope.defs[expr.id][1]
                    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add) and \
                            all(isinstance(x, ast.Name) and scope.draws.get(x.id) == "str" for x in (expr.left, expr.right)):
                        for n in lengths | {len(target) // 2}:
                            if 0 <= n <= len(target):
                                add("str", target[:n]); add("str", target[n:])
                    if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name) and \
                            scope.draws.get(expr.value.id) == "str" and isinstance(expr.slice, ast.Slice) and \
                            _literal(expr.slice.step) == -1:
                        add("str", target[::-1])
                    if isinstance(expr, ast.JoinedStr):
                        pieces = [x.value for x in expr.values if isinstance(x, ast.Constant) and isinstance(x.value, str)]
                        if len(pieces) == 1 and pieces[0] in target:
                            for part in target.split(pieces[0]):
                                add("str", part)
                                if part.lstrip("-").isdigit():
                                    add("int", int(part))
                    add("str", target)
            # Source literals can be useful as dictionary fragments and simple concatenations.
            # Only executable comparisons: no docstrings, annotations or comments.
            strings = [n.value for cmp in comparisons for n in ast.walk(cmp)
                       if isinstance(n, ast.Constant) and type(n.value) is str and 0 < len(n.value) <= 32]
            for a in strings[:12]:
                for b in strings[:12]:
                    if a != b and len(a) + len(b) <= 64:
                        add("str", a + b)
    if trick == "boundaries":
        for rec in records or ():
            types = rec.get("types", {})
            for fact in rec.get("scope", ()):
                try:
                    cmp = ast.parse(fact, mode="eval").body
                except SyntaxError:
                    continue
                if not isinstance(cmp, ast.Compare) or len(cmp.ops) != 1 or not isinstance(cmp.left, ast.Name):
                    continue
                kind = types.get(cmp.left.id)
                bound = _literal(cmp.comparators[0])
                if kind == "int" and type(bound) is int:
                    for v in (bound - 1, bound, bound + 1, 0):
                        add(kind, v)
                elif kind == "float" and type(bound) in (int, float):
                    x = float(bound)
                    for v in (math.nextafter(x, -math.inf), x, math.nextafter(x, math.inf), 0.0):
                        add(kind, v)
    return out
