"""Effect-rooted obligation slicing ([internal ref]), shared by proof and search.

The fixed-point data/control closure follows ardier/amonhen's
amonhen/anchor/dataflow.py and slicer/slicer.py (MIT, Ardi Madadi, 2026).
Unlike observation-conditioned slicing, effects are roots: reaching the
observation is part of the criterion. No dependency on the source project.

Only unused literal stores and inert, unreferenced function definitions are
removed. Calls (including argument/receiver mutations), exceptions, loops and
all control transfers survive. Reflection/unknown imports refuse pruning.
"""
from __future__ import annotations

import ast
import copy
import hashlib
from dataclasses import dataclass
from collections import Counter

from tharkun.solve.census import _bound

from tharkun.solve.pyresolve import Resolver, Top

# These modules expose no frame/module namespace inspection. Unknown imports
# retain the original, including imports through aliases.
_MODULES = frozenset({"_sv_verifier", "math", "cmath", "random", "array", "io"})
_REFLECTIVE = {"dir", "help", "eval", "exec", "compile", "__import__", "globals",
               "locals", "vars", "breakpoint", "__builtins__", "__file__",
               "__cached__", "__loader__", "__spec__"}


def _loads(node):
    return {n.id for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _literal(node):
    if isinstance(node, ast.Constant):
        return type(node.value) in (int, float, complex, bool, str, bytes, type(None))
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub))
            and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) in (int, float, complex))


def _inert_def(node):
    return (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("__") and not node.decorator_list
            and not node.returns and not getattr(node, "type_params", ())
            and all(a.annotation is None for a in ast.walk(node.args) if isinstance(a, ast.arg))
            and all(_literal(v) for v in node.args.defaults + [
                v for v in node.args.kw_defaults if v is not None]))


def slice_for(body: list, need: set, *, anchors=(), closed=False) -> list:
    """One shared closure, preserving original node identities in flat mode.

    Legacy harness prefixes use the two positional arguments and keep every
    uncertain effect. Whole-program clients set closed only after namespace
    resolution, and supply anchor AST nodes. They receive copied compound nodes.
    Names are deliberately scope-insensitive: collisions retain extra code.
    """
    tree = ast.Module(body=body, type_ignores=[])
    nodes = list(ast.walk(tree))
    # An unknown call can read globals or mutate arguments. Its whole statement
    # is an effect root; the closure includes the bodies of every named callee.
    # ponytail: scope-insensitive closure; add scoped summaries only if measured
    # slice sizes justify them, never assume an unresolved call is pure.
    if not closed:
        # A prefix does not describe the entry environment: even x = 0 may
        # release an object whose finalizer changes a needed value.
        return list(body)
    candidates = {
        id(n): {t.id for t in n.targets}
        for n in nodes if isinstance(n, ast.Assign)
        and all(isinstance(t, ast.Name) and not t.id.startswith("__") for t in n.targets)
        and _literal(n.value)
    }
    candidate_targets = {id(t) for n in nodes if id(n) in candidates for t in n.targets}
    # A reassignment could decref an object and execute __del__. Only names
    # whose EVERY binding is literal can lose a store.
    nonliteral = {n.id for n in nodes if isinstance(n, ast.Name)
                  and isinstance(n.ctx, (ast.Store, ast.Del)) and id(n) not in candidate_targets}
    special = {name for n in nodes if not isinstance(n, ast.Name) for name in _bound(n)}
    special.update(name for n in nodes if isinstance(n, (ast.Global, ast.Nonlocal)) for name in n.names)
    special.update(n.rest for n in nodes if isinstance(n, ast.MatchMapping) and n.rest)
    candidates = {k: names for k, names in candidates.items() if not names & (nonliteral | special)}
    # Class namespaces are observable by metaclasses/descriptors; leave whole
    # classes untouched, including their methods. No method dispatch guesses.
    for n in nodes:
        if isinstance(n, ast.ClassDef):
            for child in ast.walk(n):
                candidates.pop(id(child), None)
    # Only unique module-level function bindings are eligible. Definition-time
    # defaults/decorators/annotations with effects are never discarded.
    bindings = Counter(name for n in nodes for name in _bound(n))
    bindings.update(n.rest for n in nodes if isinstance(n, ast.MatchMapping) and n.rest)
    defs = {id(n): {n.name} for n in body
            if closed and _inert_def(n) and bindings[n.name] == 1}
    optional = candidates | defs
    anchor_ids = {id(a) for anchor in anchors for a in ast.walk(anchor)}
    live = set(need)
    keep = set(anchor_ids)
    for n in nodes:
        if isinstance(n, ast.stmt) and any(id(a) in anchor_ids for a in ast.walk(n)):
            keep.add(id(n))

    def collect(node):
        if id(node) in optional and id(node) not in keep and not optional[id(node)] & live:
            return
        keep.add(id(node))
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            live.add(node.id)
        elif isinstance(node, ast.Attribute):
            live.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            live.add(node.value)  # constant-name getattr and callback registries
        for child in ast.iter_child_nodes(node):
            collect(child)

    while True:
        before = (len(live), len(keep))
        collect(tree)
        if before == (len(live), len(keep)):
            break

    class Prune(ast.NodeTransformer):
        def visit(self, node):
            if id(node) in optional and id(node) not in keep:
                return ast.copy_location(ast.Pass(), node)
            node = copy.copy(node)
            for field, value in ast.iter_fields(node):
                if isinstance(value, list):
                    setattr(node, field, list(value))
            return super().visit(node)

    return [Prune().visit(st) for st in body]


@dataclass(frozen=True)
class Slice:
    source: str
    source_sha256: str
    removed: tuple[tuple[int, int], ...] = ()
    reason: str = ""


def slice_obligation(source: str, obligation: dict, *, filename: str = "<program>") -> Slice:
    """Slice for a witness-format location_invariant, not a safety certificate.

    Accepts {"invariant": {type, location, value, format}} plus an optional
    exception key. Locations are 1-based Unicode characters. The original
    location and statement (not only the predicate's names) are retained.
    Output preserves source coordinates, so consumers need no location remap.
    """
    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if not isinstance(obligation, dict) or not isinstance(obligation.get("invariant"), dict):
        raise ValueError("expected an invariant entry")
    inv = obligation["invariant"]
    loc = inv.get("location")
    if not isinstance(loc, dict) or not isinstance(inv.get("value"), str):
        raise ValueError("invalid obligation location/value")
    if (inv.get("type") != "location_invariant" or inv.get("format") != "python_expression"
            or loc.get("file_name") != filename or not isinstance(loc.get("function"), str)
            or type(loc.get("line")) is not int or loc["line"] < 1
            or type(loc.get("column")) is not int or loc["column"] < 1):
        raise ValueError("invalid obligation location/format")
    predicate = ast.parse(inv["value"], mode="eval")
    tree = ast.parse(source, filename)
    # CPython counts LF/CRLF, not Unicode separators embedded in literals.
    if "\r" in source.replace("\r\n", ""):
        return Slice(source, sha, reason="bare carriage return")
    parts = source.split("\n")
    lines = [part + "\n" for part in parts[:-1]] + [parts[-1]]
    anchors = [node for node, function in _statements(tree)
               if node.lineno == loc["line"] and function == loc["function"]
               and len(lines[node.lineno - 1].encode("utf-8")[:node.col_offset].decode("utf-8")) + 1 == loc["column"]]
    if len(anchors) != 1:
        raise ValueError("obligation does not identify one original statement")
    if not source.isascii():
        return Slice(source, sha, reason="non-ASCII source: retain byte and character coordinates")
    try:
        resolver = Resolver(tree, _MODULES)
        resolver.check_all()
        if any(isinstance(n, ast.Name) and n.id in _REFLECTIVE for n in ast.walk(tree)):
            raise Top("namespace inspection")
        if any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "__del__"
               for n in ast.walk(tree)):
            raise Top("finalizer scheduling can observe removed allocations")
    except Top as e:
        return Slice(source, sha, reason=str(e))
    body = slice_for(tree.body, _loads(predicate), anchors=anchors, closed=True)
    original = {(n.lineno, n.col_offset): n for n in ast.walk(tree) if isinstance(n, ast.stmt)}
    removed = tuple((n.lineno, n.col_offset) for n in ast.walk(ast.Module(body=body, type_ignores=[]))
                    if isinstance(n, ast.Pass) and not isinstance(original[n.lineno, n.col_offset], ast.Pass))
    # Blank deleted spans rather than unparse: original witness/trace coordinates,
    # Unicode byte offsets, comments and semicolon siblings stay valid.
    data = bytearray(source.encode("utf-8"))
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line.encode("utf-8")))
    for point in sorted(removed, reverse=True):
        n = original[point]
        start = offsets[n.lineno - 1] + n.col_offset
        end = offsets[n.end_lineno - 1] + n.end_col_offset
        for i in range(start, end):
            if data[i] not in (10, 13):
                data[i] = 32
        data[start:start + 1] = b"0"
    result = data.decode("utf-8")
    try:
        ast.parse(result, filename)
    except SyntaxError:
        # A multiline assignment can end before a same-line semicolon sibling.
        # Blanking its closing delimiter would strand that sibling's separator.
        return Slice(source, sha, reason="source layout cannot preserve statement boundaries")
    return Slice(result, sha, removed)


def _statements(node, function="<module>"):
    """Statement ownership, including nested/class-qualified function names."""
    if isinstance(node, ast.stmt):
        yield node, function
    for field, value in ast.iter_fields(node):
        scope = function
        if field == "body" and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = node.name if function == "<module>" else function + "." + node.name
        for child in value if isinstance(value, list) else [value]:
            if isinstance(child, ast.AST):
                yield from _statements(child, scope)


def _first_anchorable(entries):
    """The first entry that is a real per-site witness, never the census's own
    "$coverage" module placeholder (line 1, col 1, ``<module>``, value ``False``,
    [internal ref]) -- that placeholder anchors nothing, it only records that coverage was
    refused for a whole exception class."""
    for entry in entries:
        loc = entry.get("invariant", {}).get("location", {})
        if (loc.get("line"), loc.get("column"), loc.get("function")) == (1, 1, "<module>"):
            continue
        return entry
    return None


def _slice_or_fall_back(source: str, sha: str, entry: dict, filename: str) -> Slice:
    """slice_obligation, but analysis_slice never raises into its callers ([internal ref]):
    a real-site entry whose location still cannot be pinned to exactly one
    original statement (an unusual layout, not a coverage gap), or any other
    unexpected failure anchor selection provokes, degrades to the unsliced
    source with the exception recorded in the reason -- like every other
    anchor failure in this module, never a raise into the caller. A
    whole-program analyzer must still get a verdict, not a crash."""
    try:
        return slice_obligation(source, entry, filename=filename)
    except Exception as e:
        return Slice(source, sha, reason=f"{type(e).__name__}: {e}")


def analysis_slice(source: str, classes, *, filename="<program>", obligations=None) -> Slice:
    """Adapter for whole-program analyzers; every reachable effect is retained.

    Prefer open witness entries supplied by the caller. With none given, this
    anchors on the obligation census's ([internal ref], `solve.census.obligation_census`)
    own first open entry for these classes: an untruncated, class-specific
    witness -- a real per-site safety predicate where the census covers the
    site, or an explicit predicate-`False` coverage-gap entry (an unresolved
    class, an unmodelled implicit call, a definition-time eval) where it
    doesn't. Either way the entry is a real `Invariant`, not a display-only
    census result reduced to a location by this adapter. No coverage/truth
    claim is inferred beyond what `ObligationCensus.complete` itself already
    asserts.

    One anchor suffices for this conservative pass: it preserves ALL reachable
    effects, including the other error sites. Per-obligation consumers may
    instead call slice_obligation directly with each criterion.
    """
    from tharkun.solve.census import obligation_census

    sha = hashlib.sha256(source.encode("utf-8")).hexdigest()
    if obligations is not None:
        first = _first_anchorable(obligations)
        return (_slice_or_fall_back(source, sha, first, filename)
                if first is not None else Slice(source, sha, reason="no open obligations"))
    class_names = tuple(c.__name__ if isinstance(c, type) else str(c) for c in classes)
    census = obligation_census(source, class_names, filename)
    first = _first_anchorable(o.invariant.entry() for o in census.obligations)
    if first is None:
        return Slice(source, sha, reason="no open obligations")
    return _slice_or_fall_back(source, sha, first, filename)
