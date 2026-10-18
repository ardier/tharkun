"""Exact, whole-program Z3 proof for loop-free "product" programs ([internal ref]'s
equivalent-mutant proving; owner ruling 2026-09-29: try a GENERAL proof before any
mutant-specific rewrite -- see [internal ref]).

Scope, checked as a SHAPE, never assumed (soundness rule: unmodelled -> UNKNOWN,
never TRUE, never a kill without replay):

* every ``_sv_verifier.nondet_*()`` draw is a bare statement
  ``name = _sv_verifier.nondet_TYPE()`` of kind ``int`` or ``bool`` (see
  :data:`_EXACT_KINDS`), at the top level of the module OR inside a def of it ([internal ref],
  at any ``if`` depth: ``def main(): x = nondet_int(); if not (0 <= x <= 9): return;
  assert ...`` then ``main()``) -- ``float`` is EXCLUDED, not merely unverified:
  :func:`tharkun.sv._symexec.make_var` models it as a z3 ``Real`` (exact rational
  arithmetic), the wrong domain for IEEE-754 double semantics, so an UNSAT over it
  would prove a claim about real numbers, not about the ``float`` values the
  program actually computes with; ``str`` is excluded because this module has not
  audited ``_symexec``'s string theory against the exact bar the rest of this file
  holds arithmetic to (MG review 2026-09-29, PR #114). A draw of an excluded kind,
  or one anywhere else (inside a loop, a comprehension, a lambda, a nested def, a
  loose expression, reused under a name already taken), makes the whole attempt
  bail out (``None``), never guessing at what it might mean (the [internal ref] bug [internal ref]
  flags: a silently-ignored observed variable). A top-level draw is one z3 term
  bound before the run; a draw inside a def is a FRESH term at EVERY execution of its
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

Before the audit, :func:`_normalize` rewrites what is inert for a run as ``__main__`` (the engine's child runs
the program as ``__main__``): ``if __name__ == "__main__":`` at the top level becomes its body (its ``else`` never
runs), and ``x: int = v`` becomes ``x = v`` (the annotation is a builtin type name, run nowhere); then
:func:`tharkun.sv.exact_unroll.unroll_for_loops` replaces every ``for i in range(<constant>)`` by the statements it
stands for ([internal ref] PR 2: an exact unrolling, not a bound -- see that module's docstring). Nothing else is rewritten, and
the lines are the original's.

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
from tharkun.sv._symexec import Unsupported, make_var, model_value, run_exact
from tharkun.sv.exact_subset import audit
from tharkun.sv.exact_unroll import unroll_for_loops
from tharkun.sv.sites import LIFTED

#: draw kinds this module accepts -- NOT all of :data:`LIFTED`. ``float`` is a real
#: gap, not merely unverified: ``_symexec.make_var("float")`` builds a z3 ``Real``
#: (unbounded, exact RATIONAL arithmetic), which is the wrong domain for IEEE-754
#: double semantics (rounding, NaN, +-0.0, overflow to inf) -- an UNSAT over ``Real``
#: proves a claim about REAL numbers, not about the ``float`` values the product
#: program actually computes with, so it could claim TRUE for a pair that a real
#: `float` run would tell apart. ``str`` is excluded for the same reason: this
#: module has not audited ``_symexec``'s string theory against the EXACT semantics
#: bar the rest of this file holds arithmetic to. Restricting to ``int``/``bool``
#: (the only kinds this module's own tests exercise, and the only kinds the owner's
#: toy needs) keeps the soundness argument to what is actually verified, rather
#: than inheriting it from :data:`LIFTED`'s broader, unaudited-for-this-purpose
#: list -- MG review 2026-09-29 (PR #114).
_EXACT_KINDS = {"int", "bool"}

#: top-level statement kinds :func:`_body` strips: ONLY after :func:`tharkun.sv.
#: exact_subset.audit` has accepted the whole module, which guarantees every import is
#: ``_sv_verifier`` and every def is a plain, decorator-free, default-free function
#: bound once -- so stripping them drops no eager evaluation and no rebinding (the PR
#: #114 review holes t01-t04, and a04/a07/a08/a22 after them).
_DECLARATIONS = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef)


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
    """The kind (``int`` / ``bool``) when ``call`` is ``_sv_verifier.nondet_TYPE()`` with no argument, else ``None``."""
    name = _nondet_name(call, aliases, direct)
    if name in LIFTED and LIFTED[name] in _EXACT_KINDS and not call.args and not call.keywords:
        return LIFTED[name]
    return None


def _lift(tree: ast.Module):
    """``({name: kind}, {id(stmt) for the top-level draw assigns}, {id(call): kind for the draws inside defs})``
    when EVERY ``_sv_verifier`` draw in the module is either a bare top-level statement
    ``name = _sv_verifier.nondet_TYPE()`` or a ``_sv_verifier.nondet_TYPE()`` call inside a def --
    ``None`` otherwise (a draw
    anywhere else, a non-:data:`LIFTED` kind, args/keywords on the call, a top-level name drawn
    twice, a top-level ``class`` this module does not model at all, or anything
    outside :mod:`tharkun.sv.exact_subset`'s allowlist -- a foreign import, an eager
    ``def`` default/decorator, a rebound builtin, a nested def/import, an attribute
    store, an ``__sv_*`` name, ...)."""
    if any(isinstance(st, ast.ClassDef) for st in tree.body):
        return None
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
        inside_def = isinstance(st, ast.FunctionDef)
        for node in ast.walk(st):
            if not isinstance(node, ast.Call) or _nondet_name(node, aliases, direct) is None \
                    or node in lifted_values:
                continue
            # every draw is a lifted top-level statement or a call inside a def; anywhere else (a loose
            # module-level expression) this bails out rather than silently drop it (the [internal ref] bug class [internal ref] flags)
            kind = _draw_kind(node, aliases, direct) if inside_def else None
            if kind is None:
                return None
            fresh[id(node)] = kind
    if not order and not fresh:
        return None
    if audit(tree, lifted_ids, frozenset(fresh)) is not None:
        return None                      # outside the audited exact subset: UNKNOWN
    return order, lifted_ids, fresh


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


def _normalize(tree: ast.Module) -> None:
    """Rewrite what is inert for a run of the program as ``__main__`` (see the module docstring), in place."""
    body = []
    for st in tree.body:
        body.extend(st.body if _is_main_guard(st) else [st])
    tree.body = body
    _Normalize().visit(tree)
    unroll_for_loops(tree)


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


def prove(source: str, classes, *, program: str, python: str | None = None, prop_name: str = "",
          timeout_ms: int = 10000, run_timeout: float = 30.0):
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
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    _normalize(tree)
    lifted = _lift(tree)
    if lifted is None:
        return None
    order, lifted_ids, fresh = lifted
    try:
        sites = violation_sites(source, classes)
    except Exception:  # noqa: BLE001 -- a proof stage never decides by crashing
        return None
    if not sites:
        return None
    lines = sorted({s["line"] for s in sites})
    env = {name: make_var(name, kind) for name, kind in order.items()}
    body = _body(tree, lifted_ids)
    try:
        result = run_exact(tree, {}, env, body, lines, fresh_draws=fresh)
    except Unsupported:
        return None
    except Exception:  # noqa: BLE001 -- a proof stage never decides by crashing
        return None
    if not result["exact"]:
        return None
    all_goals = [g for goals in result["goals"].values() for g in goals]
    if not all_goals:
        verdict = _engine.Verdict(
            "true", reason="exact whole-program symbolic proof: every checked site is "
                          "unreachable by any input (no site at all on any modelled path)",
            component="symbolic:product", detail={"sites": lines})
        return bind_route(verdict, "product", source, program, classes)
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.add(z3.Or(*[g if isinstance(g, z3.ExprRef) else z3.BoolVal(bool(g)) for g in all_goals]))
    try:
        res = s.check()
    except z3.Z3Exception:
        return None
    if res == z3.unsat:
        verdict = _engine.Verdict(
            "true", reason=f"exact whole-program symbolic proof: {len(lines)} check(s) "
                          f"unreachable by any input -- every operator, call and path modelled "
                          f"exactly, no loop bound hit", component="symbolic:product",
            detail={"sites": lines})
        return bind_route(verdict, "product", source, program, classes)
    if res != z3.sat or fresh:
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
