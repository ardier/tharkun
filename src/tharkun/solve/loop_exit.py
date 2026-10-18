"""Loop-exit reasoning: which loop-carried variables are pinned at a loop's exits.

SOLVING by the delineation rule (``docs/LAYERS.md``): a source anchor in, a
``FactEnv`` value-set claim out, no observed output, no target execution.  This
is the one tharkun call eregion makes today (``loop_exit_factenv`` behind
``EREGION_PIN_LOOP_EXIT_CONSTS``).

Split out of ``condition_solver`` (#0606, plan #0600).  The loop-shape helpers
(``_single_loop_exits``, ``_loop_touched_names``, ``_collect_break_exits``) are
also used by ``condition._resolve_scope_variables``; the import runs one way
(``condition`` -> ``loop_exit``), and ``tharkun.solve.condition`` re-exports every
name here so existing callers of the old module keep working.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from tharkun.solve.fact_env import FactEnv, VarFacts
from tharkun.solve.guard import try_extract_constant, unparse

_LOOP_EXIT_UNKNOWN = object()


def _single_loop_exits(loop_node: ast.AST) -> list[dict[str, Any]]:
    """Exit facts for ONE loop: a fact per ``break`` plus the natural fall-through."""
    out: list[dict[str, Any]] = []
    _collect_break_exits(loop_node.body, [], [], {}, out)
    if isinstance(loop_node, ast.While):
        test = loop_node.test
        # ``while True`` never falls through naturally
        if not (isinstance(test, ast.Constant) and test.value is True):
            out.append({
                "kind": "natural",
                "conditions": [f"not ({unparse(test)})"],
                "asserts": [],
                "assignments": {},
            })
    return out


def _loop_touched_names(loop_node: ast.AST) -> set[str]:
    """Names assigned anywhere in a loop body (so we know which vars it can change)."""
    names: set[str] = set()
    for n in ast.walk(loop_node):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign)):
            tgt = getattr(n, "target", None)
            if isinstance(tgt, ast.Name):
                names.add(tgt.id)
    return names


def _collect_loop_exit_facts(source_path: str, lineno: int) -> list[dict[str, Any]]:
    """Facts that hold at each exit of the loop(s) preceding or enclosing *lineno*.

    The other static resolver only scans top-level assignments, so a variable
    mutated inside a loop is invisible to it.  This walks loop bodies and records,
    for every exit, the conditions that select it -- letting a post-loop or
    in-loop guard on a loop-carried variable be reasoned about.

    One dict per exit:
        ``kind``        -- ``'break'`` or ``'natural'``
        ``conditions``  -- enclosing if-guard chain (break), or the *negated* loop
                           condition (natural exit)
        ``asserts``     -- assert tests on the path (assert-sensitive)
        ``assignments`` -- latest *constant* assignment to each name before the exit

    Conservative: only syntactically-present guards/asserts and constant
    assignments are recorded; anything non-constant is dropped, never guessed.
    """
    try:
        tree = ast.parse(Path(source_path).read_text(encoding="utf-8", errors="ignore"))
    except (OSError, SyntaxError):
        return []
    func: ast.AST | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if getattr(node, "lineno", 0) <= lineno <= getattr(node, "end_lineno", 1 << 30):
                func = node
    if func is None:
        return []
    out: list[dict[str, Any]] = []
    for node in ast.walk(func):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        loop_start = getattr(node, "lineno", 0)
        loop_end = getattr(node, "end_lineno", 0)
        # the loop precedes the point (post-loop guard) or encloses it (in-loop guard)
        if not (loop_end <= lineno or loop_start <= lineno <= loop_end):
            continue
        out.extend(_single_loop_exits(node))
    return out


# Public name for callers (eregion / glamdring) that want the raw per-exit facts.
collect_loop_exit_facts = _collect_loop_exit_facts


def loop_exit_factenv(source_path: str, lineno: int) -> Any:
    """Loop-carried variable knowledge at *lineno*, as a ``FactEnv``.

    For every loop-carried variable whose *every* break-exit assigns it a constant
    and the loop has *no* natural fall-through, ``eq`` is the set of those constants
    -- a sound whitelist of its possible values at the exit: a single value for a
    ``while True`` state machine (``q -> {2}``), or a value-set for a multi-terminal
    one (``q -> {2, 3}``).  Variables with a natural fall-through or a non-constant
    mutation are omitted (genuinely unknown -- never guessed).

    Returns a ``tharkun.solve.fact_env.FactEnv`` ready to feed back into
    ``solve_condition(..., facts=...)`` (constraint propagation), or to consume
    directly.  This is the bridge from loop-exit reasoning into the rest of the
    solver's narrowing.
    """
    exits = _collect_loop_exit_facts(source_path, lineno)
    breaks = [e for e in exits if e["kind"] == "break"]
    facts: dict[str, VarFacts] = {}
    if breaks and not any(e["kind"] == "natural" for e in exits):
        names: set[str] = set()
        for b in breaks:
            names |= set(b["assignments"])
        for name in names:
            vals = [b["assignments"].get(name, _LOOP_EXIT_UNKNOWN) for b in breaks]
            if _LOOP_EXIT_UNKNOWN not in vals:  # every break pins it to a constant
                facts[name] = VarFacts(eq=set(vals))
    return FactEnv(facts=facts)


def _collect_break_exits(
    stmts: list[ast.stmt],
    conds: list[str],
    asserts: list[str],
    assigns: dict[str, Any],
    out: list[dict[str, Any]],
) -> None:
    """Walk *stmts* tracking the if-guard path, asserts, and constant assignments;
    snapshot the accumulated facts at each ``break``.  Nested loops own their own
    breaks, so we do not descend into them."""
    for st in stmts:
        if isinstance(st, ast.Assign):
            val, ok = try_extract_constant(st.value)
            for tgt in st.targets:
                if isinstance(tgt, ast.Name):
                    assigns[tgt.id] = val if ok else _LOOP_EXIT_UNKNOWN
        elif isinstance(st, ast.AnnAssign) and st.value is not None:
            if isinstance(st.target, ast.Name):
                val, ok = try_extract_constant(st.value)
                assigns[st.target.id] = val if ok else _LOOP_EXIT_UNKNOWN
        elif isinstance(st, ast.AugAssign):
            if isinstance(st.target, ast.Name):
                assigns[st.target.id] = _LOOP_EXIT_UNKNOWN
        elif isinstance(st, ast.Assert):
            asserts = asserts + [unparse(st.test) or "<assert>"]
        elif isinstance(st, ast.If):
            tc = unparse(st.test) or "<cond>"
            _collect_break_exits(st.body, conds + [tc], asserts, dict(assigns), out)
            _collect_break_exits(st.orelse, conds + [f"not ({tc})"], asserts, dict(assigns), out)
        elif isinstance(st, ast.Break):
            out.append({
                "kind": "break",
                "conditions": list(conds),
                "asserts": list(asserts),
                "assignments": {
                    k: v for k, v in assigns.items() if v is not _LOOP_EXIT_UNKNOWN
                },
            })
        # ast.Continue / ast.Return / nested loops: not a break of THIS loop -> skip
