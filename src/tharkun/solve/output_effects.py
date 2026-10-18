"""Recognize output handles whose contents cannot return to the program.

This licenses a successful-output abstraction, with uncaught environmental
failures as terminating prefixes. Exception handlers, escaping handles and
reads refuse; the caller must exclude environmental exceptions from its goal.
"""
import ast

# The probe's output handle models exactly these.  The shared model (environment.py::OUTPUT_MODES)
# classifies more write-only modes (``x``, ``tw``, ``bw``, ...) as output; this proof accepts the
# subset it can replay.
MODES = frozenset({'w', 'wt', 'wb', 'a', 'at', 'ab'})


def output_only(tree):
    """Return whether output open() occurs; raise ValueError on unsupported uses."""
    nodes = list(ast.walk(tree))
    opens = [n for n in nodes if isinstance(n, ast.Name) and n.id == 'open']
    if not opens:
        return False
    # Imports read files too. Loading allowed libraries before any program
    # effects prevents discarded output from changing their module bodies.
    # ponytail: require an import preamble; relax with import/write effect ordering.
    #
    # [internal ref] follow-up (raytrace_nondet): the preamble is every LEADING top-level
    # statement that cannot itself call anything (so it cannot reach a write --
    # nothing has opened yet) -- not just "imports, then stop at the first
    # non-import". A docstring, an import, an undecorated def/class (defining a
    # function/class runs no code inside it), and a single-target literal
    # constant assignment all qualify; the module's own bm_raytrace_nondet.py
    # shape (constant assigns interleaved with imports, `import _sv_verifier`
    # last) needs exactly this. The first statement that COULD call something
    # (anything else) ends the preamble; every import after that point still
    # refuses, unchanged.
    def _cannot_call(node: ast.stmt) -> bool:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            return True                                    # a docstring
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return not node.decorator_list                 # defining calls nothing
        if isinstance(node, ast.ClassDef):
            return (not node.decorator_list and not node.keywords
                    and all(isinstance(b, ast.Name) for b in node.bases))
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)):
            return True                                     # e.g. SIZE = 7
        return False

    preamble = set()
    for node in tree.body:
        if _cannot_call(node):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                preamble.add(node)
        else:
            break
    if any(isinstance(node, (ast.Import, ast.ImportFrom)) and node not in preamble
           for node in nodes):
        raise ValueError('an import may read a file after output')
    parents = {child: node for node in nodes for child in ast.iter_child_nodes(node)}
    bindings = {}
    for node in nodes:
        name = (node.id if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del))
                else node.arg if isinstance(node, ast.arg)
                else node.name if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                else (node.asname or node.name.split('.')[0]) if isinstance(node, ast.alias) else None)
        if name:
            bindings.setdefault(name, []).append(node)
    # ponytail: reject every handler; allow unrelated ones after call-effect summaries.
    #
    # [internal ref] follow-up: a Try/TryStar with NO except handlers -- only a `finally`
    # (and/or an `else`, which never runs when an exception propagates) -- cannot
    # itself OBSERVE an I/O failure: nothing catches it, so it still propagates
    # out exactly as if the try/finally were not there. `n.handlers` is the AST's
    # own list of `except`/`except*` clauses; empty means no handler at all.
    if 'open' in bindings or any(
            (isinstance(n, (ast.Try, ast.TryStar)) and n.handlers) or isinstance(n, ast.AsyncWith)
            for n in nodes):
        raise ValueError('output open is rebound or an exception handler can observe I/O failure')
    handles = {}
    contexts = set()
    for node in opens:
        call = parents.get(node)
        if not isinstance(call, ast.Call) or call.func is not node:
            raise ValueError('open escapes as a value')
        if (not 1 <= len(call.args) <= 2 or any(isinstance(a, ast.Starred) for a in call.args)
                or any(k.arg not in {'mode', 'encoding', 'newline'} for k in call.keywords)):
            raise ValueError('unsupported output open arguments')
        keywords = {k.arg: k.value for k in call.keywords}
        mode = call.args[1] if len(call.args) == 2 else keywords.get('mode')
        if not isinstance(mode, ast.Constant) or mode.value not in MODES:
            raise ValueError('open mode may read')
        if len(call.args) == 2 and 'mode' in keywords:
            raise ValueError('duplicate output mode')
        if any(not isinstance(v, ast.Constant) for k, v in keywords.items() if k != 'mode'):
            raise ValueError('output encoding/newline must be literal')
        encoding = keywords.get('encoding')
        newline = keywords.get('newline')
        if encoding and encoding.value not in {'utf-8', 'ascii', 'latin1'}:
            raise ValueError('unsupported output encoding')
        if newline and newline.value not in {None, '', '\n', '\r', '\r\n'}:
            raise ValueError('unsupported output newline')
        if 'b' in mode.value and (encoding or newline):
            raise ValueError('binary output with text options')
        binding = parents.get(call)
        if isinstance(binding, ast.withitem) and binding.context_expr is call:
            target = binding.optional_vars
            contexts.add(binding)
        elif isinstance(binding, ast.Assign) and len(binding.targets) == 1:
            target = binding.targets[0]
        else:
            raise ValueError('output handle must bind a local name')
        if not isinstance(target, ast.Name) or bindings.get(target.id) != [target]:
            raise ValueError('output handle escapes or is rebound')
        handles[target.id] = target
    for node in nodes:
        if isinstance(node, ast.With) and any(item not in contexts for item in node.items):
            raise ValueError('another context manager can observe I/O failure')
        if not isinstance(node, ast.Name) or node.id not in handles or node is handles[node.id]:
            continue
        parent = parents.get(node)
        if isinstance(parent, ast.keyword) and parent.arg == 'file':
            call = parents.get(parent)
            if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                    and call.func.id == 'print' and 'print' not in bindings):
                continue
        if isinstance(parent, ast.Attribute) and parent.value is node and parent.attr in {'write', 'close', 'flush'}:
            call = parents.get(parent)
            if (isinstance(call, ast.Call) and call.func is parent and not call.keywords
                    and len(call.args) == (1 if parent.attr == 'write' else 0)
                    and not any(isinstance(a, ast.Starred) for a in call.args)
                    and isinstance(parents.get(call), ast.Expr)):
                continue
        raise ValueError('output handle is read, escapes, or exposes a return value')
    return True
