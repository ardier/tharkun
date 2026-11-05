"""Private source snapshots for closed module compile-to-exec flows."""
from __future__ import annotations

import ast
import copy
import time
import warnings

from .comprehension import Lowered
from .eval_source import literal_sources, source_bindings
from .provenance import ROLE_ATTR, stand_for
from .scopes import COMPREHENSIONS, Names, bound_names, identifiers


def lower_compile_snapshots(tree, *, deadline=None):
    work = copy.deepcopy(tree)
    nodes = list(ast.walk(work))
    if 'compile' in bound_names(work) or any(isinstance(n, ast.ImportFrom)
            and (n.module == '__future__' or any(a.name == '*' for a in n.names)) for n in nodes):
        return Lowered(work)
    parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
    bindings = source_bindings(work)

    def module_scope(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, *COMPREHENSIONS)):
                return False
        return True

    candidates, payloads = {}, []
    for statement in nodes:
        if deadline is not None and time.process_time() >= deadline:
            return Lowered(work)
        if not (isinstance(statement, ast.Assign) and all(isinstance(t, ast.Name) for t in statement.targets)
                and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == 'compile' and module_scope(statement)):
            continue
        call = statement.value
        if not (len(call.args) == 3 and not call.keywords and isinstance(call.args[0], (ast.Constant, ast.Name))
                and isinstance(call.args[1], ast.Constant) and type(call.args[1].value) is str
                and isinstance(call.args[2], ast.Constant) and type(call.args[2].value) is str
                and call.args[2].value == 'exec'):
            continue
        sources = literal_sources(call.args[0], bindings, deadline)
        if sources is None:
            continue
        try:
            for source in sources:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    parsed = ast.parse(source, mode='exec')
                    compile(parsed, call.args[1].value, 'exec', dont_inherit=True)
                if caught:
                    return Lowered(work)
                payloads.append(parsed)
        except (SyntaxError, ValueError, TypeError, RecursionError, OverflowError):
            return Lowered(work)
        candidates[statement] = call
    if not candidates:
        return Lowered(work)

    calls = set(candidates.values())
    # An unexpanded code call can read or overwrite code bindings through its source text.
    for node in nodes:
        if not isinstance(node, ast.Name) or node.id not in {'eval', 'exec', 'compile'}:
            continue
        parent = parents.get(node)
        if not isinstance(parent, ast.Call) or parent.func is not node:
            return Lowered(work)
        if parent in calls:
            continue
        if not (node.id == 'exec' and len(parent.args) == 1 and not parent.keywords
                and isinstance(parent.args[0], ast.Name) and isinstance(parents.get(parent), ast.Expr)
                and module_scope(parent)):
            return Lowered(work)
    namespace_check = copy.deepcopy(work)
    for node in ast.walk(namespace_check):
        if isinstance(node, ast.Name) and node.id in {'eval', 'exec', 'compile'}:
            node.id = '_closed_code_intrinsic'
    if Names(namespace_check).reflective or any(
            isinstance(n, ast.Import) and any(a.name == '__main__' for a in n.names)
            or isinstance(n, ast.ImportFrom) and n.module == '__main__' for n in nodes):
        return Lowered(work)

    code_names = {t.id for s in candidates for t in s.targets}
    while True:
        if deadline is not None and time.process_time() >= deadline:
            return Lowered(work)
        aliases = {t.id for n in nodes if isinstance(n, ast.Assign) and isinstance(n.value, ast.Name)
                   and n.value.id in code_names and all(isinstance(t, ast.Name) for t in n.targets)
                   for t in n.targets}
        if aliases <= code_names:
            break
        code_names.update(aliases)
    if any(not (value in calls or isinstance(value, ast.Name) and value.id in code_names)
           for name in code_names for value in bindings.get(name, ())):
        return Lowered(work)
    if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'exec'
           and n.args[0].id not in code_names for n in nodes):
        return Lowered(work)
    # Every selector read must follow an unconditional module binding: private names must never
    # replace an observable NameError for an original code-object name.
    defined = set()
    for statement in work.body:
        reads = {n.id for n in ast.walk(statement) if isinstance(n, ast.Name)
                 and isinstance(n.ctx, ast.Load) and n.id in code_names}
        if not reads <= defined:
            return Lowered(work)
        writes = {n for n in ast.walk(statement) if isinstance(n, ast.Name)
                  and isinstance(n.ctx, ast.Store) and n.id in code_names}
        if writes:
            if not isinstance(statement, ast.Assign) or not writes <= set(statement.targets):
                return Lowered(work)
            defined.update(n.id for n in writes)
    # Source bodies must not observe the original code objects or overwrite their private selectors.
    if any(Names(body).reflective or identifiers(body) & (code_names | {'eval', 'exec', 'compile'}) for body in payloads):
        return Lowered(work)
    if any(any(value is None for value in bindings.get(name, ())) for name in code_names):
        return Lowered(work)
    if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))
           and identifiers(n) & code_names for n in nodes):
        return Lowered(work)
    for node in nodes:
        if not isinstance(node, ast.Name) or node.id not in code_names:
            continue
        parent = parents.get(node)
        if isinstance(node.ctx, ast.Store):
            if isinstance(parent, ast.Assign) and all(isinstance(t, ast.Name) for t in parent.targets):
                continue
        elif isinstance(node.ctx, ast.Load):
            if (isinstance(parent, ast.Assign) and parent.value is node
                    and all(isinstance(t, ast.Name) for t in parent.targets)):
                continue
            if (isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == 'exec'
                    and len(parent.args) == 1 and not parent.keywords and parent.args[0] is node
                    and isinstance(parents.get(parent), ast.Expr) and module_scope(parent)):
                continue
        return Lowered(work)

    names = Names(work)
    for payload in payloads:
        names.taken.update(identifiers(payload))
    renamed = {name: names.fresh('compiled_source') for name in sorted(code_names)}
    for statement, call in candidates.items():
        statement.value = copy.deepcopy(call.args[0])
        stand_for(statement, statement, 'compiled-source')
    for node in nodes:
        if isinstance(node, ast.Name) and node.id in renamed:
            node.id = renamed[node.id]
            setattr(node, ROLE_ATTR, 'compiled-source')
    return Lowered(work, synthetic=frozenset(names.synthetic), lowered=len(candidates))
