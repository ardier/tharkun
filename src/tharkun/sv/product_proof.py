"""Exact, whole-program Z3 proof for loop-free "product" programs ([internal ref]'s
equivalent-mutant proving; owner ruling 2026-09-29: try a GENERAL proof before any
mutant-specific rewrite -- see [internal ref]).

Scope, checked as a SHAPE, never assumed (soundness rule: unmodelled -> UNKNOWN,
never TRUE, never a kill without replay):

* every ``_sv_verifier.nondet_*()`` draw has kind ``int``, ``bool`` or ``str`` and no arguments.
  Numeric draws use :data:`_EXACT_KINDS`; string draws use the separate audited
  :mod:`tharkun.sv.exact_strings` code-point model. Bare module-level assignments are pre-bound; other
  calls run at their own sites, at module level or inside a def ([internal ref],
  at any ``if`` depth: ``def main(): x = nondet_int(); if not (0 <= x <= 9): return;
  assert ...`` then ``main()``) -- ``float`` is EXCLUDED, not merely unverified:
  :func:`tharkun.sv._symexec.make_var` models it as a z3 ``Real`` (exact rational
  arithmetic), the wrong domain for IEEE-754 double semantics, so an UNSAT over it
  would prove a claim about real numbers, not about the ``float`` values the
  program actually computes with. String draws never use that executor's search
  StringSort: proof strings are sequences of 32-bit code points, whose equality,
  concatenation, truthiness and branch merges preserve Python builtin semantics.
  Their alphabet deliberately overapproximates Unicode and string.printable;
  UNSAT proves the stronger claim, while SAT returns UNKNOWN here and native
  search owns any FALSE. String methods, identity, ordering, chained comparisons
  and the int-only check_type rewrite remain refused. A draw of an excluded kind,
  or a call outside the audited subset, makes the whole attempt
  bail out (``None``), never guessing at what it might mean (the [internal ref] bug [internal ref]
  flags: a silently-ignored observed variable). A top-level draw is one z3 term
  bound before the run; every other draw is a FRESH term at EVERY execution of its
  statement (``_symexec.py::run_exact(fresh_draws=)``), which is what a draw is: the
  replayed vector hands each execution its own value, so a def called twice draws twice
  and the proof covers every vector, not one value per statement;

* the WHOLE module sits inside the audited ALLOWLIST of
  :mod:`tharkun.sv.exact_subset` (every AST node at every depth, every name read,
  bound or called; anything not listed is refused -- exactness by construction, not
  by a list of attacks), and the property is exactly ``AssertionError`` (an implicit
  ``ZeroDivisionError`` is modelled only as "the path ends", never as a goal);

* the REST of the module runs through :mod:`tharkun.sv._symexec`'s existing
  whole-program interpreter UNCHANGED and IN FULL: the owner's condition
  (2026-09-29) is that Z3 gets the entire inter-procedural call stack, not a slice
  around the assertion. This module does not slice: it hands the interpreter every
  top-level statement (minus the draws, already pre-bound into Z3 vars, and the
  declarations -- imports/defs carry no runtime effect for it), and both ``f`` and
  ``f_mut`` in full, plus every function either one calls, transitively --
  ``tharkun.sv._symexec._Runner`` inlines calls up to its own ``CALL_DEPTH``; a call
  it cannot inline exactly (an unmodelled callee, keyword args, a closure over a
  mutated global, recursion past that depth) is ``Unsupported``, never approximated.
  It already models ``abs()`` as ``ite``, Python floor division and ``>>`` EXACTLY
  (including negative operands -- see ``_symexec.py``'s own module docstring), and
  unbounded Z3 ``Int``s for ``nondet_int()``.

* the run must be EXACT (:func:`tharkun.sv._symexec.run_exact`): no loop was cut at
  its unroll bound (there must be no loop at all, or one this module's interpreter
  can enumerate exactly within that bound), and NOTHING anywhere in the run -- not
  only on the path to a checked site -- was left unmodelled. One unsupported
  statement, call or operator anywhere aborts the whole attempt; there is no partial
  credit and no per-statement query.

The separate ``combined_proof`` route may supply an overapproximating scalar
projection after checking its original-source effect boundary and proving every
erased assertion with complete abstract interpretation. This module still audits
and proves the ENTIRE supplied tree, including retained callees in full. Its
private ``_proof_only`` mode suppresses SAT replay for such projections, whose
draw positions differ from the original. Ordinary whole-source calls are unchanged.

Before the audit, :func:`_normalize` rewrites what is inert for a run as ``__main__`` (the engine's child runs
the program as ``__main__``): ``if __name__ == "__main__":`` at the top level becomes its body (its ``else`` never
runs), and ``x: int = v`` becomes ``x = v`` (the annotation is a builtin type name, run nowhere); then
:func:`tharkun.sv.exact_unroll.unroll_for_loops` replaces every ``for i in range(<constant>)`` by the statements it
stands for ([internal ref] PR 2: an exact unrolling, not a bound -- see that module's docstring). Successful primitive `check_type(value, int)` statements become evaluation of
`value` after the whole-module audit; argument calls, draws and exceptions remain.
Guarded literal returns first retain all check-success obligations. If that
attempt is SAT, a separate exact model may end string/int-hint mismatch paths
after argument evaluation; unsupported members still refuse the whole proof.
The attempts share the solver budget and start from independent lifted environments.
Plain int/bool literal defaults and total keyword argument expressions are bound
before the audit; default evaluation is inert and total arguments have no effects.
Assigned function locals that shadow module variables are alpha-renamed into fresh
names; unsupported nested scopes remain refused. The lines are the original's.

Given that, and ONLY given that, :func:`prove` asks Z3 the two-sided question a
search can never answer, over every path at once:

* every explicit ``assert`` / matching ``raise`` is UNREACHABLE by ANY value of the
  free variables, proven, not sampled -- ``Verdict("true", component=
  "symbolic:product")``.
* Z3 finds an input that reaches one: that model is a CANDIDATE ONLY (the same "a
  wrong solve costs one run, never a wrong verdict" rule :mod:`tharkun.sv.concolic`
  documents) -- it is replayed on the UNMODIFIED program in a fresh child
  (:func:`tharkun.sv.engine.replay`), and only a replay that genuinely raises the
  property becomes ``Verdict("false", component="symbolic:product")``. A model that
  does not replay is not a kill: this returns ``None``, never a guess. When a draw sits inside a def, a
  model has no single vector (which draws a path consumes, and in what order, depends on the path),
  so such a program is only ever PROVED here: a model returns ``None`` and the search stages
  (``concolic.py``, whose harness lifts the entry function's draws) own the FALSE.

Called standalone, the FALSE side above is exactly ONE fresh-child replay -- sound
for a genuinely deterministic program (every product this module targets: pure
arithmetic over its own draws, no set/dict order, no address, no clock), but not
address/hash-order-independence CERTIFIED the way :mod:`tharkun.sv.scheduler`'s own
``found()`` insists on for the general case. Reached through
:func:`tharkun.sv.verify` (``scheduler.py``'s stage 1b), this module's FALSE verdict
is handed to that SAME ``found()`` (``fresh=True``, since the replay above already
counts as the first trial) before it is trusted, so it gets the k-of-k
address-independence confirmation for free, unchanged and undiminished, for any
program this scope admits that is not already known-deterministic.

Anything outside this scope -- a loop, a draw this module cannot see whole, a call
Z3's interpreter cannot model exactly, or a solver timeout -- returns ``None``
(UNKNOWN), unchanged, for whichever stage runs next.
"""
from __future__ import annotations

import ast

import z3

from tharkun.sv import exact_subset
from tharkun.sv._symexec import _WorkBudget, Unsupported, make_var, model_value, run_exact
from tharkun.sv.exact_subset import audit
from tharkun.sv.exact_unroll import unroll_for_loops
from tharkun.sv.exact_strings import variable as string_variable
from tharkun.sv.exact_decorators import lower_forwarding_decorators
from tharkun.sv.exact_closures import lower_returned_additive_closures
from tharkun.sv.sites import LIFTED

#: Numeric draw kinds, never all of LIFTED: float's Real model is not IEEE binary64.
#: Strings enter separately through exact_strings, not the search executor's StringSort.
_EXACT_KINDS = {"int", "bool"}

#: top-level statement kinds :func:`_body` strips: ONLY after :func:`tharkun.sv.
#: exact_subset.audit` has accepted the whole module, which guarantees every import is
#: ``_sv_verifier``, defs are plain and bound once, and classes have only audited
#: scalar constants and methods -- so stripping drops no eager target evaluation (the PR
#: #114 review holes t01-t04, and a04/a07/a08/a22 after them).
_DECLARATIONS = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _nondet_aliases(tree: ast.Module) -> set:
    return {a.asname or a.name for st in tree.body if isinstance(st, ast.Import)
            for a in st.names if a.name == "_sv_verifier"}


def _direct_imports(tree: ast.Module) -> dict:
    out = {}
    for st in tree.body:
        if isinstance(st, ast.ImportFrom) and st.module == "_sv_verifier":
            for a in st.names:
                out[a.asname or a.name] = a.name
    return out


def _nondet_name(call: ast.Call, aliases: set, direct: dict):
    f = call.func
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id in aliases:
        return f.attr
    if isinstance(f, ast.Name) and f.id in direct:
        return direct[f.id]
    return None


def _draw_kind(call: ast.Call, aliases: set, direct: dict):
    """The audited numeric/string kind of a resolved argument-free draw, else None."""
    name = _nondet_name(call, aliases, direct)
    if name == "nondet_str" and not call.args and not call.keywords:
        return "str"  # proof-only code-point sequences; never _symexec's search StringSort.
    if name in LIFTED and LIFTED[name] in _EXACT_KINDS and not call.args and not call.keywords:
        return LIFTED[name]
    return None


def _lift(tree: ast.Module):
    """``(draw kinds, lifted assignments, fresh calls, guarded checks, closed-callable mode)``
    when EVERY draw is a bare top-level assignment or a call evaluated at its own
    site by the exact interpreter (at module level or inside a def) --
    ``None`` otherwise (a draw
    anywhere else, a non-:data:`LIFTED` kind, args/keywords on the call, a top-level name drawn
    twice, a class outside the source-only plain-class model, or anything
    outside :mod:`tharkun.sv.exact_subset`'s allowlist -- a foreign import, an eager
    ``def`` default/decorator, a rebound builtin, a nested def/import, an attribute
    store outside that model, an ``__sv_*`` name, ...)."""
    plain_classes = any(isinstance(st, ast.ClassDef) for st in tree.body)
    aliases, direct = _nondet_aliases(tree), _direct_imports(tree)
    if not aliases and not direct:
        return None
    lifted_ids, order = set(), {}
    for st in tree.body:
        if not (isinstance(st, ast.Assign) and len(st.targets) == 1
                and isinstance(st.targets[0], ast.Name) and isinstance(st.value, ast.Call)):
            continue
        kind = _draw_kind(st.value, aliases, direct)
        if kind is not None:
            target = st.targets[0].id
            if target in order:
                return None              # redrawn name: not a shape this module models
            order[target] = kind
            lifted_ids.add(id(st))
    lifted_values = {st.value for st in tree.body if id(st) in lifted_ids}
    fresh = {}
    for st in tree.body:
        for node in ast.walk(st):
            if not isinstance(node, ast.Call) or _nondet_name(node, aliases, direct) is None \
                    or node in lifted_values:
                continue
            # Every unlifted call gets a fresh term per execution, including unrolled module-level loops.
            if _nondet_name(node, aliases, direct) == "check_type":
                continue                 # the exact-subset auditor validates this separately
            kind = _draw_kind(node, aliases, direct)
            if kind is None:
                return None
            fresh[id(node)] = kind
    if not order and not fresh:
        return None
    integer_checks = set()
    string_draws = "str" in order.values() or "str" in fresh.values()
    from tharkun.sv.exact_callables import ClosedCallables
    closed_callables = ClosedCallables(tree)
    closed_callables = closed_callables if closed_callables.active else None
    # Guarded literal returns use the existing code-point/type model, without widening numeric or string-draw admission.
    guarded_returns = (not string_draws and any(isinstance(n, ast.Return) and isinstance(n.value, ast.Constant)
        and type(n.value.value) is str for n in ast.walk(tree)) and (closed_callables is not None or any(isinstance(n, ast.Call)
        and _nondet_name(n, aliases, direct) == "check_type" for n in ast.walk(tree))))
    bool_checks = any(isinstance(n, ast.Call) and _nondet_name(n, aliases, direct) == "check_type"
        and len(n.args) == 2 and isinstance(n.args[1], ast.Name) and n.args[1].id == "bool" for n in ast.walk(tree))
    string_checks = any(isinstance(n, ast.Call) and _nondet_name(n, aliases, direct) == "check_type"
        and len(n.args) == 2 and isinstance(n.args[1], ast.Name) and n.args[1].id == "str" for n in ast.walk(tree))
    if bool_checks and string_draws:
        return None  # this checker extension is for int/bool draws, not another string-draw admission
    if plain_classes and closed_callables is not None:
        return None
    guarded_checks = set() if plain_classes or guarded_returns or bool_checks or string_checks else None
    from tharkun.sv.exact_strings import has_edge_calls
    if audit(tree, lifted_ids, frozenset(fresh), integer_checks=integer_checks,
             string_draws=plain_classes or string_draws or guarded_returns or has_edge_calls(tree), guarded_type_checks=guarded_checks,
             closed_callables=closed_callables) is not None:
        return None                      # outside the audited exact subset: UNKNOWN
    if integer_checks:
        class PreserveArgument(ast.NodeTransformer):
            def visit_Expr(self, node):
                if id(node.value) in integer_checks:
                    return ast.copy_location(ast.Expr(value=node.value.args[0]), node)
                return self.generic_visit(node)
        PreserveArgument().visit(tree)
    return order, lifted_ids, fresh, guarded_checks, closed_callables is not None


def _is_main_guard(st) -> bool:
    return (isinstance(st, ast.If) and isinstance(st.test, ast.Compare) and isinstance(st.test.left, ast.Name)
            and st.test.left.id == "__name__" and len(st.test.ops) == 1 and isinstance(st.test.ops[0], ast.Eq)
            and isinstance(st.test.comparators[0], ast.Constant) and st.test.comparators[0].value == "__main__")


class _Normalize(ast.NodeTransformer):
    """``x: int = v`` / ``xs: list[int] = v`` is ``x = v`` for a run (a function's local annotation is never evaluated, a
    module's is a builtin type name)."""

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if isinstance(node.target, ast.Name) and node.value is not None and (
                exact_subset._annotation_ok(node.annotation, False) or exact_subset._list_annotation(node.annotation)):
            return ast.copy_location(ast.Assign(targets=[node.target], value=node.value), node)
        return node


def _normalize_literal_defaults(tree: ast.Module) -> None:
    """Bind plain literal defaults/keywords when argument reordering has no effects."""
    signatures = {}
    for fd in tree.body:
        if not isinstance(fd, ast.FunctionDef):
            continue
        a = fd.args
        params = [p.arg for p in a.args]
        if not a.defaults or fd.decorator_list or a.posonlyargs or a.kwonlyargs or a.vararg or a.kwarg \
                or len(params) != len(set(params)) or any(not isinstance(v, ast.Constant)
                or type(v.value) not in (int, bool) for v in a.defaults):
            continue
        signatures[fd.name] = (params, dict(zip(params[-len(a.defaults):], a.defaults)))

    class Bind(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            sig = signatures.get(node.func.id) if isinstance(node.func, ast.Name) else None
            if sig is None:
                return node
            params, defaults = sig
            if len(node.args) > len(params) or not all(exact_subset._total(v)
                    for v in [*node.args, *(k.value for k in node.keywords)]):
                return node
            values = dict(zip(params, node.args))
            for kw in node.keywords:
                if kw.arg not in params or kw.arg in values:
                    return node
                values[kw.arg] = kw.value
            if any(name not in values and name not in defaults for name in params):
                return node
            node.args = [values[name] if name in values else ast.copy_location(
                ast.Constant(defaults[name].value), node) for name in params]
            node.keywords = []
            return node

    Bind().visit(tree)
    for fd in tree.body:
        if isinstance(fd, ast.FunctionDef) and fd.name in signatures:
            fd.args.defaults = []


def _rename_shadowed_locals(tree: ast.Module) -> None:
    """Keep assigned function locals separate from identically named module variables."""
    module_names = []
    exact_subset._store_names(tree.body, module_names)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    used.update(n.arg for n in ast.walk(tree) if isinstance(n, ast.arg))
    used.update(n.name for n in tree.body if isinstance(n, ast.FunctionDef))
    used.update(a.asname or a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in n.names)
    functions = [fd for fd in tree.body if isinstance(fd, ast.FunctionDef)]
    functions.extend(fd for cls in tree.body if isinstance(cls, ast.ClassDef)
                     for fd in cls.body if isinstance(fd, ast.FunctionDef))
    for fd in functions:
        if not isinstance(fd, ast.FunctionDef) or any(isinstance(n, (
                ast.FunctionDef, ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal,
                ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp))
                for st in fd.body for n in ast.walk(st)):
            continue
        locals_ = []
        exact_subset._store_names(fd.body, locals_)
        collisions = set(locals_) & set(module_names) - {a.arg for a in fd.args.args}
        replacements = {}
        for name in sorted(collisions):
            fresh = "tharkun_local"
            while fresh in used:
                fresh += "_"
            used.add(fresh)
            replacements[name] = fresh
        for st in fd.body:
            for n in ast.walk(st):
                if isinstance(n, ast.Name) and n.id in replacements:
                    n.id = replacements[n.id]


def _normalize(tree: ast.Module) -> None:
    """Normalize inert syntax, forwarding decorators and closed module callbacks in place."""
    body = []
    for st in tree.body:
        body.extend(st.body if _is_main_guard(st) else [st])
    tree.body = body
    _Normalize().visit(tree)
    # Authenticate the original factory before defaults/decorators/loops can erase its source shape.
    lower_returned_additive_closures(tree)
    _normalize_literal_defaults(tree)
    unroll_for_loops(tree)
    lower_forwarding_decorators(tree)
    from tharkun.sv.exact_callables import lift_named_lambdas, lift_scalar_lambdas
    lift_named_lambdas(tree)
    lift_scalar_lambdas(tree)
    _rename_shadowed_locals(tree)


def _body(tree: ast.Module, lifted_ids: set) -> list:
    """``tree.body`` minus the draw assigns (already pre-bound into ``env``) and the
    declarations (imports/defs -- see :data:`_DECLARATIONS`); everything else, in
    its original order, unsliced."""
    return [st for st in tree.body if id(st) not in lifted_ids and not isinstance(st, _DECLARATIONS)]


def z3_timeout_ms(remaining: float) -> int:
    """The Z3 timeout of the proof: ``min(60 s, max(10 s, 15% of the remaining budget))`` (a nonlinear postcondition such as
    ``-(a // -b) * b >= a`` takes about 12 s, so the competition's budget decides it), and never more than what REMAINS: a
    10 s triage run must not spend its whole budget in one solver call (``scheduler.py::run`` passes ``remaining()``)."""
    return int(1000 * min(60.0, max(10.0, 0.15 * remaining), remaining))


def _closed_sibling_tree(tree, source, program, declared_files):
    """Bind a zero-goal, effect-free sibling def as source, never as a live module.

    Imported identities/certificates remain unsupported. Only a closed expression
    function is relocated; no imported statement or goal is silently discarded.
    """
    import copy
    import hashlib
    import io
    import os
    import tokenize
    from pathlib import Path
    from tharkun.solve.extern import _task_import_closure
    from tharkun.sv._symexec import _ImportedDefs

    foreign = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
               and (isinstance(n, ast.Import) and any(a.name != "_sv_verifier" for a in n.names)
                    or isinstance(n, ast.ImportFrom) and (n.module != "_sv_verifier" or n.level))]
    if not foreign:
        return tree, ()
    files = _task_import_closure(program)
    if (files is None or len(files) != 2 or len(foreign) != 1
            or not isinstance(foreign[0], ast.ImportFrom) or foreign[0].level
            or not foreign[0].module or "." in foreign[0].module
            or len(foreign[0].names) != 1 or foreign[0].names[0].name == "*"):
        return None
    if declared_files is not None and tuple(map(os.path.abspath, declared_files)) != files:
        return None
    try:
        snapshots = tuple((p, Path(p).read_bytes()) for p in files)
        encoding, _ = tokenize.detect_encoding(io.BytesIO(snapshots[0][1]).readline)
        if snapshots[0][1].decode(encoding) != source:
            return None
        other = ast.parse(snapshots[1][1], files[1])
        module_body = other.body
        if module_body and isinstance(module_body[0], ast.Expr) and isinstance(module_body[0].value, ast.Constant) \
                and type(module_body[0].value.value) is str:
            module_body = module_body[1:]
        # The category is "the sibling's only import-time effect is defining functions": any number of other top-level `def`s are fine when
        # defining them evaluates nothing (no decorator, default, annotation or type parameter) and they are never imported; the imported
        # name must be defined exactly once.  Anything else at module level (statements, classes, imports, async defs) is refused.
        if len(foreign[0].names) != 1 or foreign[0].names[0].name == '*':
            return None
        wanted = foreign[0].names[0].name

        def inert_definition(item):
            if not isinstance(item, ast.FunctionDef):
                return False
            a = item.args
            return (not item.decorator_list and item.returns is None
                    and not getattr(item, "type_params", None) and not a.defaults and not any(d is not None for d in a.kw_defaults)
                    and all(x.annotation is None for x in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg) if x is not None))
        if not module_body or any(not inert_definition(item) for item in module_body):
            return None
        chosen = [item for item in module_body if item.name == wanted]
        if len(chosen) != 1:
            return None
        fd = chosen[0]
        args = fd.args
        if (fd.decorator_list or fd.returns is not None or args.defaults or args.kw_defaults
                or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg
                or getattr(fd, "type_params", None) or any(a.annotation is not None for a in args.args)):
            return None
        _ImportedDefs._closed_binds(fd)
        body = fd.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                and type(body[0].value.value) is str:
            body = body[1:]
        if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
            return None
        # No calls (even check_type), attributes, scopes, stores or goal-bearing sites.
        allowed = (ast.Name, ast.Load, ast.Constant, ast.BinOp, ast.Add, ast.Sub, ast.Mult,
                   ast.FloorDiv, ast.Mod, ast.UnaryOp, ast.UAdd, ast.USub, ast.Not)
        if any(not isinstance(n, allowed) for n in ast.walk(body[0].value)):
            return None
        imported = foreign[0]
        replacement = copy.deepcopy(fd)
        replacement.name = imported.names[0].asname or imported.names[0].name
        replacement.body = copy.deepcopy(body)
        # Imported code has no goals; keep all entry-site lines unmodified.
        for node in ast.walk(replacement):
            if hasattr(node, "lineno"):
                node.lineno = imported.lineno
                node.end_lineno = imported.end_lineno
        ast.copy_location(replacement, imported)
        tree.body[tree.body.index(imported)] = replacement
        if any(Path(path).read_bytes() != raw for path, raw in snapshots):
            return None
        return tree, tuple((path, hashlib.sha256(raw).hexdigest()) for path, raw in snapshots)
    except (OSError, SyntaxError, ValueError, Unsupported):
        return None


def prove(source: str, classes, *, program: str, python: str | None = None, prop_name: str = "",
          timeout_ms: int = 10000, run_timeout: float = 30.0, declared_files=None, _proof_only=False):
    """A ``Verdict`` (``"true"`` or ``"false"``) for ``program`` (whose text is
    ``source``), or ``None`` -- see the module docstring for the exact scope and the
    soundness gate. ``classes`` is the property's exception classes (e.g.
    ``("AssertionError",)``)."""
    from tharkun.sv import engine as _engine
    from tharkun.sv.sites import violation_sites
    from tharkun.sv.store_witness import bind_route

    if _engine.native_import_reason(source, program):
        return None
    if tuple(classes) != ("AssertionError",):
        # the interpreter models a failing assert / explicit raise as a goal, but an
        # IMPLICIT ZeroDivisionError / IndexError / ... only as "the path ends"
        return None
    budget = _WorkBudget(timeout_ms)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    closed = _closed_sibling_tree(tree, source, program, declared_files)
    if closed is None:
        return None
    tree, closure = closed

    def bind(verdict):
        if closure:
            import hashlib
            from pathlib import Path
            from tharkun.solve.extern import _task_import_closure
            try:
                if (_task_import_closure(program) != tuple(path for path, _ in closure)
                        or any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != sha
                               for path, sha in closure)):
                    return None
            except OSError:
                return None
            verdict.detail["source_closure"] = [dict(path=path, sha256=sha) for path, sha in closure]
        return bind_route(verdict, "product", source, program, classes)

    classes_present = any(isinstance(n, ast.ClassDef) for n in tree.body)
    try:
        budget.check()
        if classes_present:
            from tharkun.sv.exact_classes import plain_classes
            try:
                plain_classes(tree)
            except exact_subset._Refuse:
                return None
        _normalize(tree)
        budget.check()
        lifted = _lift(tree)
        budget.check()
    except (Unsupported, MemoryError, RecursionError):
        return None
    if lifted is None:
        return None
    order, lifted_ids, fresh, guarded_checks, closed_callables = lifted
    try:
        sites = violation_sites(source, classes)
    except Exception:  # noqa: BLE001 -- a proof stage never decides by crashing
        return None
    if not sites:
        return None
    lines = sorted({s["line"] for s in sites})
    env = {name: string_variable(name) if kind == "str" else make_var(name, kind) for name, kind in order.items()}
    body = _body(tree, lifted_ids)
    try:
        result = run_exact(tree, {}, dict(env), body, lines, fresh_draws=fresh, guarded_type_checks=guarded_checks,
                           closed_callables=closed_callables, plain_classes=classes_present,
                           _work_budget=budget)
    except Unsupported:
        return None
    except Exception:  # noqa: BLE001 -- a proof stage never decides by crashing
        return None
    if not result["exact"]:
        return None
    proof_scope = "sound whole-program symbolic proof" if result.get("identity_choices") else "exact whole-program symbolic proof"
    all_goals = [g for goals in result["goals"].values() for g in goals]
    type_obligations = result.get("type_obligations", [])
    # A failed success-contract precondition is a refusal, never an additional property or a FALSE witness.
    all_goals += type_obligations
    if not all_goals:
        try:
            budget.check()
        except Unsupported:
            return None
        verdict = _engine.Verdict(
            "true", reason=proof_scope + ": every checked site is "
                          "unreachable by any input (no site at all on any modelled path)",
            component="symbolic:product", detail={"sites": lines})
        return bind(verdict)
    s = z3.Solver()
    try:
        s.set("timeout", budget.remaining_ms())
        s.add(z3.Or(*[g if isinstance(g, z3.ExprRef) else z3.BoolVal(bool(g)) for g in all_goals]))
        res = s.check()
        budget.check()
    except (z3.Z3Exception, Unsupported):
        return None
    if res == z3.sat and type_obligations and guarded_checks:
        # Keep the success-only proof strict; retry with a separately audited non-property exit.
        try:
            result = run_exact(tree, {}, dict(env), body, lines, fresh_draws=fresh,
                               guarded_type_checks=guarded_checks, primitive_type_exits=True,
                               closed_callables=closed_callables, plain_classes=classes_present,
                               _work_budget=budget)
            if not result["exact"]:
                return None
            type_obligations = result.get("type_obligations", [])
            goals = [g for group in result["goals"].values() for g in group] + type_obligations
            remaining_ms = budget.remaining_ms()
            s.reset()
            s.set("timeout", remaining_ms)
            s.add(z3.Or(*[g if isinstance(g, z3.ExprRef) else z3.BoolVal(bool(g)) for g in goals]))
            res = s.check()
            budget.check()
        except Exception:  # noqa: BLE001 -- an unavailable exact model is a refusal
            return None
    if res == z3.unsat:
        verdict = _engine.Verdict(
            "true", reason=(f"{proof_scope}: {len(lines)} check(s) unreachable by any input -- "
                            + ("unknown identity choices retained, no loop bound hit" if result.get("identity_choices") else
                               "every operator, call and path modelled exactly, no loop bound hit")), component="symbolic:product",
            detail={"sites": lines})
        return bind(verdict)
    if _proof_only or res != z3.sat or fresh or "str" in order.values() or type_obligations:
        # shortcut: string draws overapproximate the native domain; native search owns their SAT/replay side.
        return None                      # solver timed out / gave up, or no single draw vector (module docstring)
    m = s.model()
    vector = [{"kind": kind, "value": model_value(m, env[name], kind)} for name, kind in order.items()]
    try:
        confirmed, out = _engine.replay(program, {"vector": vector}, classes, python=python,
                                        timeout=run_timeout)
    except Exception:  # noqa: BLE001 -- a proof stage never decides by crashing
        return None
    if not confirmed:
        return None                      # the Z3 model did not reproduce: no guess
    exc = out.violated(classes)
    return _engine.Verdict("false", exc, witness=_engine._witness(program, prop_name, exc, out),
                           component="symbolic:product", detail={"model": vector})
