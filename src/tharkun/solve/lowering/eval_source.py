"""Proof-only expansion of finite, default-namespace module code.

The source and callee must have no opaque bindings. Function/class bodies,
compiler warnings, future flags and namespace-writing expressions stay opaque.
Every inserted operation keeps the original code call's source provenance.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import time
import warnings

from .comprehension import Lowered
from .provenance import EVAL_SITE_ATTR, annotate, format_path, path_of, stand_for
from .scopes import COMPREHENSIONS, bound_names


def source_bindings(tree):
    nodes = list(ast.walk(tree))
    parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
    bindings = {}
    for node in nodes:
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            parent = parents.get(node)
            value = parent.value if isinstance(parent, (ast.Assign, ast.AnnAssign, ast.NamedExpr)) else None
            if isinstance(parent, ast.AnnAssign) and value is None:
                continue
            bindings.setdefault(node.id, []).append(value)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.ExceptHandler)):
            if node.name:
                bindings.setdefault(node.name, []).append(None)
            for param in getattr(node, 'type_params', ()):
                bindings.setdefault(param.name, []).append(None)
        elif isinstance(node, ast.arg):
            bindings.setdefault(node.arg, []).append(None)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bindings.setdefault((alias.asname or alias.name).split('.')[0], []).append(None)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                bindings.setdefault(name, []).append(None)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bindings.setdefault(node.name, []).append(None)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bindings.setdefault(node.rest, []).append(None)

    return bindings


def literal_sources(node, bindings, deadline=None):
    pending, seen, result = [node], set(), set()
    while pending:
        if deadline is not None and time.process_time() >= deadline:
            return None
        n = pending.pop()
        if isinstance(n, ast.Constant) and type(n.value) in (str, bytes):
            result.add(n.value)
        elif isinstance(n, ast.IfExp):
            pending.extend((n.body, n.orelse))
        elif isinstance(n, ast.Name):
            if n.id in seen:
                continue
            seen.add(n.id)
            values = bindings.get(n.id)
            if not values or any(v is None for v in values):
                return None
            pending.extend(values)
        else:
            return None
    return result or None


def lower_module_code(tree: ast.Module, *, enabled=('eval',), deadline: float | None = None, preserve_origins=False) -> Lowered:
    work = copy.deepcopy(tree)
    if not preserve_origins:
        annotate(work)
    nodes = list(ast.walk(work))
    enabled = set(enabled) - bound_names(work)
    if (not enabled or any(
            isinstance(n, ast.ImportFrom) and (n.module == '__future__' or any(a.name == '*' for a in n.names))
            for n in nodes)):
        return Lowered(work)
    bindings = source_bindings(work)

    def sources(node):
        return literal_sources(node, bindings, deadline)

    # An executed assignment cannot invalidate this original-source binding census.
    protected = {'eval', 'exec', 'compile'}
    protected_seen = set()
    pending = [n.args[0] for n in nodes if isinstance(n, ast.Call) and n.args
               and isinstance(n.func, ast.Name) and n.func.id in protected]
    while pending:
        if deadline is not None and time.process_time() >= deadline:
            return Lowered(work)
        expr = pending.pop()
        for name in (n.id for n in ast.walk(expr) if isinstance(n, ast.Name)):
            if name not in protected_seen:
                protected_seen.add(name)
                protected.add(name)
                pending.extend(v for v in bindings.get(name, ()) if v is not None)

    def parse(source, mode, anchor):
        text = source.lstrip(b' \t' if type(source) is bytes else ' \t') if mode == 'eval' else source
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                parsed = ast.parse(text, mode=mode)
                compile(parsed, '<' + mode + '>', mode, dont_inherit=True)
            if caught or any(isinstance(n, (ast.NamedExpr, *COMPREHENSIONS)) for n in ast.walk(parsed)):
                return None
            if deadline is not None and time.process_time() >= deadline:
                return None
        except (SyntaxError, ValueError, TypeError, RecursionError, OverflowError):
            return None
        annotate(parsed)
        digest = hashlib.sha256((b'str:' + source.encode('utf-8')) if type(source) is str
                                else b'bytes:' + source).hexdigest()
        for child in ast.walk(parsed):
            origin = format_path(path_of(child))
            stand_for(child, anchor, mode)
            setattr(child, EVAL_SITE_ATTR, mode + ':' + digest + ':' + origin)
        return parsed

    def test_source(node, items):
        tests = [ast.Compare(left=copy.deepcopy(node.args[0]), ops=[ast.Eq()],
                             comparators=[ast.Constant(value=s)]) for s, _ in items]
        test = tests[0] if len(tests) == 1 else ast.BoolOp(op=ast.Or(), values=tests)
        for child in ast.walk(test):
            stand_for(child, node, node.func.id)
        return test

    class Expand(ast.NodeTransformer):
        count = 0

        # These bodies have their own default locals, including lazy annotations.
        def visit_Lambda(self, node):
            return node

        def visit_FunctionDef(self, node):
            node.decorator_list = [self.visit(n) for n in node.decorator_list]
            node.args.defaults = [self.visit(n) for n in node.args.defaults]
            node.args.kw_defaults = [self.visit(n) if n is not None else None for n in node.args.kw_defaults]
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            node.decorator_list = [self.visit(n) for n in node.decorator_list]
            node.bases = [self.visit(n) for n in node.bases]
            for keyword in node.keywords:
                keyword.value = self.visit(keyword.value)
            return node

        def visit_ListComp(self, node):
            return node

        visit_SetComp = visit_ListComp
        visit_DictComp = visit_ListComp
        visit_GeneratorExp = visit_ListComp

        def visit_Expr(self, statement):
            node = statement.value
            if not ('exec' in enabled and isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'exec' and len(node.args) == 1 and not node.keywords
                    and isinstance(node.args[0], (ast.Constant, ast.Name))):
                return self.generic_visit(statement)
            alternatives = sources(node.args[0])
            if alternatives is None:
                return statement
            bodies = []
            for source in sorted(alternatives, key=lambda x: (type(x).__name__, repr(x))):
                parsed = parse(source, 'exec', node)
                if parsed is None or bound_names(parsed) & protected or any(
                        not (isinstance(s, ast.Pass) or isinstance(s, ast.Assign)
                             and all(isinstance(t, ast.Name) for t in s.targets)) for s in parsed.body):
                    return statement
                bodies.append((source, parsed.body or [stand_for(ast.Pass(), node, 'exec')]))

            def dispatch(items):
                if len(items) == 1:
                    return items[0][1]
                middle = len(items) // 2
                return [stand_for(ast.If(test=test_source(node, items[:middle]),
                    body=dispatch(items[:middle]), orelse=dispatch(items[middle:])), node, 'exec')]

            body = dispatch(bodies)
            if isinstance(node.args[0], ast.Name):
                body.insert(0, stand_for(ast.Expr(value=copy.deepcopy(node.args[0])), node, 'exec'))
            self.count += 1
            return body

        def visit_Call(self, node):
            if not ('eval' in enabled and isinstance(node.func, ast.Name) and node.func.id == 'eval' and len(node.args) == 1
                    and not node.keywords and isinstance(node.args[0], (ast.Constant, ast.Name))):
                return self.generic_visit(node)
            alternatives = sources(node.args[0])
            if alternatives is None:
                return node
            expressions = []
            for source in sorted(alternatives, key=lambda x: (type(x).__name__, repr(x))):
                if deadline is not None and time.process_time() >= deadline:
                    return node
                parsed = parse(source, 'eval', node)
                if parsed is None:
                    return node
                expressions.append((source, parsed.body))
            result = expressions[-1][1]
            # Preserve the source-name read even for a single alternative (notably NameError handlers).
            if isinstance(node.args[0], ast.Name):
                def dispatch(items):
                    if len(items) == 1:
                        return items[0][1]
                    middle = len(items) // 2
                    return stand_for(ast.IfExp(test=test_source(node, items[:middle]),
                        body=dispatch(items[:middle]), orelse=dispatch(items[middle:])), node, 'eval')
                result = dispatch(expressions)
                sequence = ast.Tuple(elts=[copy.deepcopy(node.args[0]), result], ctx=ast.Load())
                result = ast.Subscript(value=sequence, slice=ast.Constant(value=1), ctx=ast.Load())
                for child in (sequence, result, result.slice):
                    stand_for(child, node, 'eval')
                    setattr(child, EVAL_SITE_ATTR, 'source-read')
            self.count += 1
            return result

    expand = Expand()
    work = expand.visit(work)
    ast.fix_missing_locations(work)
    return Lowered(work, lowered=expand.count)
