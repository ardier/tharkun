"""Which fuzzer field reaches which function's parameter.

The assist could only ever attack one function, because ``HarnessSpec`` named a single
``target_func`` and the codec mapped fields to it positionally.  On a real harness that
is most of the coverage left on the table — ntlm_auth's harness calls into three
modules, and pointing at one of them leaves the frontiers in the other two invisible.

Lifting that needs one fact the codec does not have: **which field feeds which callee
parameter**.  It is recoverable from the harness itself, because a harness is written
exactly this way::

    fdp = atheris.FuzzedDataProvider(data)
    mod_a.check_a(fdp.ConsumeUInt(4))                      # field 0 -> check_a.code
    mod_c.check_c(fdp.ConsumeUInt(4), fdp.ConsumeUInt(4))  # fields 1,2 -> check_c.n, m

So: walk the harness body in source order, counting ``Consume*`` calls so the numbering
matches :func:`~tharkun.hybrid.fdp_codec.record_schedule`, and whenever a consume sits
directly in a call's argument list, bind that field index to the callee's parameter of
the same position (resolved from the callee's own signature).

This is deliberately syntactic.  A harness that stores a field in a local first, or
builds a structure before passing it, is not matched — those calls are reported with no
bindings rather than guessed at, and the caller falls back to whatever explicit target it
was given.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_UNKNOWN = object()


@dataclass
class CallTarget:
    """A function the harness calls, and which fields reach its parameters."""

    func: str                                   # callee name, e.g. "check_c"
    module_path: str                            # file the callee is defined in
    params: list[str] = field(default_factory=list)          # callee signature order
    field_to_param: dict[int, str] = field(default_factory=dict)   # field idx -> param
    # Parameters the harness pins to a fixed value rather than a fuzzer field:
    #   get_seal_key(fdp.ConsumeInt(4), fdp.ConsumeBytes(124),
    #                SignSealConstants.CLIENT_SEALING)
    # Without these the call is missing a required argument and cannot be driven at all,
    # so every branch inside it stays dark for a reason that has nothing to do with
    # solving — measured, exactly the last 2 of 56 lines on ntlm_auth.
    fixed_args: dict[str, Any] = field(default_factory=dict)

    @property
    def param_to_field(self) -> dict[str, int]:
        return {p: i for i, p in self.field_to_param.items()}


def _is_consume(node: ast.expr) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr.startswith("Consume"))


def _resolve_const(node: ast.expr, harness_path: str, depth: int = 0) -> tuple[Any, bool]:
    """Statically evaluate *node* to a constant, following a dotted name to its source.

    Returns ``(value, True)`` when the value is known.  A harness pins arguments with
    module or class constants (``SignSealConstants.CLIENT_SEALING``), so a literal check
    alone is not enough — the name has to be chased through the import that bound it.
    """
    try:
        return ast.literal_eval(node), True
    except (ValueError, SyntaxError, TypeError):
        pass
    if depth > 2:
        return None, False

    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        owner, attr = node.value.id, node.attr
    elif isinstance(node, ast.Name):
        owner, attr = None, node.id
    else:
        return None, False

    search = [os.path.realpath(harness_path)]
    try:
        from tharkun.solve.frontend.import_resolver import parse_imports

        found = parse_imports(harness_path).get(owner or attr)
        if found:
            search.append(found)
    except Exception:
        pass

    for path in search:
        try:
            tree = ast.parse(Path(path).read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError):
            continue
        bodies = ([tree.body] if owner is None else
                  [n.body for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == owner])
        for body in bodies:
            for stmt in body:
                targets = (stmt.targets if isinstance(stmt, ast.Assign)
                           else [stmt.target] if isinstance(stmt, ast.AnnAssign) and stmt.value
                           else [])
                for tgt in targets:
                    if isinstance(tgt, ast.Name) and tgt.id == attr:
                        return _resolve_const(stmt.value, path, depth + 1)
    return None, False


def _callee_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _callee_qualifier(node: ast.Call) -> str | None:
    """The name the harness writes before the dot: ``compute_keys.get_seal_key`` -> module.

    A harness names its callee as ``module.func`` or ``Class.func``, and both resolve
    through the harness's own imports.  Honouring it is what keeps the mapping to
    functions the harness can *statically* reach: resolving by bare name alone would
    happily bind a same-named function in a module the harness never calls, and this very
    package defines seven names in more than one module.
    """
    if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        return node.func.value.id
    return None


def _index_package(package_dir: str) -> dict[str, list[tuple[str, list[str]]]]:
    """``{function name: [(defining file, parameter names), ...]}`` for a package tree.

    Every definition is kept, not just the first: a name defined in several modules is
    ambiguous, and which one the harness meant is decided by the qualifier it wrote, not
    by directory order.
    """
    out: dict[str, list[tuple[str, list[str]]]] = {}
    for path in sorted(Path(package_dir).rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = node.args
                names = [p.arg for p in (a.posonlyargs + a.args)]
                names = [n for n in names if n not in ("self", "cls")]
                out.setdefault(node.name, []).append((str(path), names))
    return out


def call_site_constants(package_dir: str, func_name: str) -> dict[str, Any]:
    """Parameters of *func_name* that every call site in the package fixes to one value.

    A guard can be satisfiable *in isolation* and unreachable *in context*. ntlm_auth's
    ``elif block_length != 8`` needs ``pad`` falsy — and ``pad=True`` is the default that
    every single ``.encrypt(...)`` call in the package accepts. A sound decider certifies the
    sliced guard satisfiable because it sees ``pad`` as a free variable; the program never
    offers it a falsy one.

    Deciding that requires a fact the guard does not contain: what the callers actually
    pass. A parameter is pinned when no call site overrides it (so its default stands) or
    when every call site passes the same literal. Feeding these back as preconditions is
    what turns "satisfiable" into "infeasible **here**", so a dead frontier is ruled out
    rather than searched forever.

    Deliberately conservative: any call site that passes something non-constant, or that
    could not be matched, drops the parameter from the result entirely.
    """
    root = Path(package_dir)
    defaults: dict[str, Any] = {}
    positional: list[str] = []
    found_def = False

    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                a = node.args
                names = [q.arg for q in (a.posonlyargs + a.args) if q.arg not in ("self", "cls")]
                positional = names
                found_def = True
                # Defaults bind to the LAST len(defaults) positional parameters.
                if a.defaults:
                    for name, dflt in zip(names[len(names) - len(a.defaults):], a.defaults):
                        val, ok = _resolve_const(dflt, str(path))
                        if ok:
                            defaults[name] = val
                break
    if not found_def or not defaults:
        return {}

    # Any call that supplies one of these parameters removes it from consideration
    # unless every such call supplies the identical constant.
    supplied: dict[str, list[Any]] = {}
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or _callee_name(node) != func_name:
                continue
            for pos, arg in enumerate(node.args):
                if pos < len(positional):
                    name = positional[pos]
                    if name in defaults:
                        val, ok = _resolve_const(arg, str(path))
                        supplied.setdefault(name, []).append(val if ok else _UNKNOWN)
            for kw in node.keywords:
                if kw.arg in defaults:
                    val, ok = _resolve_const(kw.value, str(path))
                    supplied.setdefault(kw.arg, []).append(val if ok else _UNKNOWN)

    pinned: dict[str, Any] = {}
    for name, dflt in defaults.items():
        seen = supplied.get(name, [])
        if not seen:
            pinned[name] = dflt                    # nobody overrides it
        elif all(v is not _UNKNOWN and v == seen[0] for v in seen):
            pinned[name] = seen[0]                 # everybody passes the same literal
    return pinned


def map_harness_calls(harness_path: str, package_dir: str) -> list[CallTarget]:
    """Recover every package function the harness calls, with its field bindings.

    Field numbering follows the harness's own execution order, which is what
    ``record_schedule`` records, so an index here indexes that schedule directly.
    """
    try:
        source = Path(os.path.realpath(harness_path)).read_text(encoding="utf-8",
                                                                errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return []

    known = _index_package(package_dir)
    if not known:
        return []

    try:
        from tharkun.solve.frontend.import_resolver import parse_imports

        harness_imports = parse_imports(harness_path)
    except Exception:
        harness_imports = {}

    def _resolve(name: str, qualifier: str | None):
        """The definition the harness actually names, or None if it is ambiguous.

        With a qualifier, the callee must live in the module that qualifier resolves to
        through the harness's own imports — so a same-named function in a module the
        harness never imports can never be selected. Without one, the name must be
        unique in the package; an ambiguous bare call is dropped rather than guessed.
        """
        defs = known.get(name) or []
        if not defs:
            return None
        if qualifier:
            want = harness_imports.get(qualifier)
            if want:
                want = os.path.realpath(want)
                for path, params in defs:
                    if os.path.realpath(path) == want:
                        return (path, params)
            return None
        return defs[0] if len({os.path.realpath(p) for p, _ in defs}) == 1 else None

    counter = {"n": 0}
    targets: dict[str, CallTarget] = {}

    def visit(node: ast.AST) -> None:
        """Walk in source order, numbering consumes as the harness would execute them."""
        if isinstance(node, ast.Call):
            name = _callee_name(node)
            entry = _resolve(name, _callee_qualifier(node)) if name else None
            # Number this call's own arguments first: a consume in an argument list is
            # consumed before the call itself happens.
            bindings: dict[int, str] = {}
            fixed: dict[str, Any] = {}
            for pos, arg in enumerate(node.args):
                if _is_consume(arg):
                    idx = counter["n"]
                    counter["n"] += 1
                    if entry and pos < len(entry[1]):
                        bindings[idx] = entry[1][pos]
                else:
                    # Not a fuzzer field. If the harness pinned it to a constant, keep
                    # the value — otherwise the call cannot even be made.
                    if entry and pos < len(entry[1]):
                        val, ok = _resolve_const(arg, harness_path)
                        if ok:
                            fixed[entry[1][pos]] = val
                    visit(arg)
            if _is_consume(node) and not isinstance(node.func, ast.Name):
                # the consume call itself, when it is not an argument of anything
                pass
            if entry is not None:
                mod_path, params = entry
                tgt = targets.get(name)
                if tgt is None:
                    tgt = CallTarget(func=name, module_path=mod_path, params=list(params))
                    targets[name] = tgt
                tgt.field_to_param.update(bindings)
                tgt.fixed_args.update(fixed)
            for kw in node.keywords:
                # A pinned keyword argument counts the same as a pinned positional one.
                if entry is not None and kw.arg and not _is_consume(kw.value):
                    val, ok = _resolve_const(kw.value, harness_path)
                    if ok:
                        targets[name].fixed_args[kw.arg] = val
                visit(kw.value)
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    # Only walk the entrypoint(s); module-level code does not consume fields.
    entries = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)
               and n.name in ("TestOneInput", "TestInput")]
    if not entries:
        entries = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

    # A harness entry often delegates to helpers in the same file; inline them in order.
    helpers = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    def walk_body(fn: ast.FunctionDef, depth: int = 0) -> None:
        if depth > 3:
            return
        for stmt in fn.body:
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call):
                    nm = _callee_name(node)
                    if nm in helpers and nm != fn.name and not _is_consume(node):
                        walk_body(helpers[nm], depth + 1)
                        break
            else:
                visit(stmt)
                continue
            visit(stmt)

    for fn in entries[:1]:
        walk_body(fn)

    return [t for t in targets.values() if t.field_to_param]
