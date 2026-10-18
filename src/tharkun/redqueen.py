"""Input-to-state correspondence harvesting (RedQueen, NDSS'19), for the driver.

The SMT lane abstains on most real guards for a structural reason: it must *model*
the code — containers, string methods, f-strings, library calls — and 84.6% of its
abstentions are exactly those unmodeled constructs.  Symbolic execution cannot cross
that boundary; every Python concolic tool shares the limit.

RedQueen sidesteps modeling entirely.  It never asks "what does this transform do";
it *observes*, on a concrete run, the two operands of each comparison, and asks a much
weaker question: does an input value already flow into one operand?  If it does, then
the comparison itself reveals the value the input must take — the *other* operand — and
we place that value on the input.  A magic-number check ``x == 0xCAFE`` where ``x`` is
the input is solved by reading ``0xCAFE`` straight out of the comparison, no reasoning
about how ``x`` reached the compare site.  It "even works for cases that are usually
hard to solve", API calls and unknown instructions included, because the demanded value
is a runtime observation, not a static literal or a modeled expression.

**Our edge over the paper.** RedQueen must *guess* how the input was encoded (its
colorization step), because it works on raw bytes and cannot see field structure.  In
tharkun the driver controls typed parameters directly, and for the fuzzer the encoder
(`hybrid.fdp_codec`) is known exactly and re-verified against the real FDP.  So where
RedQueen guesses a placement, we know it: the demanded value is written onto the named
parameter (or field) with no colorization search.

Scope and honesty.  This module only *proposes*.  The driver verifies every proposal by
execution (`_flip_verifies`) before committing it — the tracer is the judge everywhere
in this codebase, and a wrong correspondence is discarded, not trusted.  That lets the
harvester be speculative on the hard case (input reaching an operand *through* an
unmodeled transform) without ever asserting an unverified result.

Correspondence kinds covered, weakest-to-strongest confidence:

  * ``identity``  — an operand *is* the input value.  Two sub-cases: the operand is the
    bare parameter (syntactic identity, high confidence), or the operand is a transform
    of the parameter that happens to be value-identical on the observed seed
    (``token.swapcase().swapcase()`` equals ``token``) — proposed speculatively, the
    driver's execution check decides whether the transform really was identity-like.
  * ``offset``    — the operand is ``p + k`` / ``p - k`` for a small concrete ``k``;
    demanded value ``d`` ⟹ propose ``p = d - k`` (resp. ``d + k``).
  * ``substring`` — the operand is a slice ``p[a:b]`` of a str/bytes input; splice the
    demanded value into that window (a ``prefix`` when the window starts at 0).

What this does **not** do: recover the input placement when the transform is *not*
value-identity-preserving on the seed and not one of the shapes above (e.g. a checksum,
a hash, base64) — there RedQueen's own colorization would also be guessing, and we
return nothing rather than a bogus proposal.  Mock the checksum instead (see CLAUDE.md).
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from typing import Any

from tharkun.solve.guard import extract_frontier
from tharkun.solve.normalize import normalize_guard

# Sentinel: an operand we could not evaluate against the live env.  Distinct from a
# legitimately-observed ``None`` operand (``x is None``), which we must not confuse.
_EVAL_FAILED = object()

# Cap on the numeric offset we will treat as a correspondence.  RedQueen's insight is that
# the *demanded value* is copied out of the compare; an arbitrarily large "offset" is not a
# correspondence, it is a computation.  Small k catches ``len(x) + 1`` / ``idx - 4`` shapes.
_MAX_OFFSET = 1 << 20


@dataclass
class Correspondence:
    """One observed input-to-state correspondence at a comparison.

    ``proposal`` is the concrete new value for ``input_name`` that the comparison
    *demands*; it is a proposal only — the driver verifies it by execution.
    """

    input_name: str
    kind: str            # "identity" | "substring" | "offset"
    demanded: Any        # the value the comparison wants (the other operand, observed live)
    proposal: Any        # concrete new value to place on input_name
    confidence: float
    comparison: str = ""  # unparsed comparison, for diagnostics
    operand: str = ""     # unparsed operand that carried the input, for diagnostics


def harvest(
    source_path: str,
    guard_line: int,
    env: dict[str, Any],
    params: set[str],
) -> list[Correspondence]:
    """Harvest input-to-state correspondences at the guard on *guard_line*.

    *env* is a live environment in which the guard's operands can be evaluated — the
    driver builds it from the module globals overlaid with the runtime snapshot, so a
    transform on the input (``compute_expected()``, ``x.decode()``) actually runs and we
    observe the value it produced.  *params* names the inputs the caller can change.

    Returns correspondences sorted by descending confidence.  Each names an input, the
    kind of correspondence, the demanded value, and a concrete proposal.  The list is
    advisory: the caller must verify every proposal by execution before trusting it.
    """
    try:
        frontier = extract_frontier(source_path, guard_line)
    except (ValueError, OSError, SyntaxError):
        return []

    try:
        normalized = normalize_guard(frontier.guard_expr)
    except Exception:
        normalized = frontier.guard_expr

    aliases = slice_aliases(source_path, guard_line, params)

    out: list[Correspondence] = []
    for cmp_node in _collect_comparisons(normalized):
        out.extend(_correspondences_for_comparison(cmp_node, env, params, aliases))

    # Strongest first; the driver tries them in order and stops at the first that verifies.
    out.sort(key=lambda c: c.confidence, reverse=True)
    return _dedup(out)


# ---------------------------------------------------------------------------
# Comparison collection
# ---------------------------------------------------------------------------


def _collect_comparisons(node: ast.expr) -> list[ast.Compare]:
    """Every single-operator ``Compare`` reachable through the boolean skeleton.

    We descend only through ``BoolOp`` / ``Not`` — the guard's boolean structure — not
    into the operands of a comparison, whose sub-expressions are the transforms whose
    *values* (not structure) we care about.  Normalization has already expanded chained
    comparisons, so each ``Compare`` we keep has exactly one operator.
    """
    found: list[ast.Compare] = []

    def walk(n: ast.expr) -> None:
        if isinstance(n, ast.Compare):
            if len(n.ops) == 1:
                found.append(n)
            return
        if isinstance(n, ast.BoolOp):
            for v in n.values:
                walk(v)
            return
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            walk(n.operand)
            return
        # Any other atom (a bare Call/Name used as a truth value) carries no
        # operand pair to harvest; nothing to do.

    walk(node)
    return found


# ---------------------------------------------------------------------------
# Per-comparison correspondence detection
# ---------------------------------------------------------------------------


def _correspondences_for_comparison(
    cmp_node: ast.Compare, env: dict[str, Any], params: set[str],
    aliases: dict[str, tuple[str, int | None, int | None]] | None = None,
) -> list[Correspondence]:
    from tharkun.solve.condition import _free_names

    if len(cmp_node.ops) != 1:
        return []
    op = cmp_node.ops[0]
    # RedQueen's mechanism is equality copying: it makes ``operand == demanded`` hold by
    # copying the demanded value onto the operand.  Inequalities do not name a single
    # target value to copy, so we do not claim a correspondence for them.
    if not isinstance(op, ast.Eq):
        return []

    left, right = cmp_node.left, cmp_node.comparators[0]
    cmp_src = ast.unparse(cmp_node)

    results: list[Correspondence] = []
    # Consider each side in turn as the *input-carrying* operand, the other as the
    # *demanded* value.  A valid correspondence needs the input side to depend on exactly
    # one parameter and the demanded side to be parameter-free (a runtime constant we can
    # copy) — otherwise "the value it wants" is itself a moving target.
    for inp_side, dem_side in ((left, right), (right, left)):
        inp_params = _free_names(inp_side) & params
        dem_params = _free_names(dem_side) & params
        if len(inp_params) != 1 or dem_params:
            continue
        p = next(iter(inp_params))
        if p not in env:
            continue

        dem_val = _safe_eval(dem_side, env)
        if dem_val is _EVAL_FAILED:
            continue
        cur = env.get(p, _EVAL_FAILED)
        inp_val = _safe_eval(inp_side, env)

        corr = _classify(op, inp_side, p, cur, inp_val, dem_val, cmp_src)
        if corr is not None:
            results.append(corr)

    if not results and aliases:
        results.extend(_alias_level(cmp_node, env, params, aliases, cmp_src))
    if not results:
        results.extend(_value_level(cmp_node, env, params, cmp_src))
    return results


def _alias_level(
    cmp_node: ast.Compare, env: dict[str, Any], params: set[str],
    aliases: dict[str, tuple[str, int | None, int | None]], cmp_src: str,
) -> list[Correspondence]:
    """Correspondences through a local that aliases a slice of a parameter.

    ``body == expected()`` where ``body = data[4:]`` ran earlier: the demand on ``body``
    is a demand on ``data``'s window, and splicing there can *lengthen* ``data`` — which
    a value copy into an existing window can never do.
    """
    from tharkun.solve.condition import _free_names

    out: list[Correspondence] = []
    left, right = cmp_node.left, cmp_node.comparators[0]

    for inp_side, dem_side in ((left, right), (right, left)):
        if not isinstance(inp_side, ast.Name) or inp_side.id not in aliases:
            continue
        if _free_names(dem_side) & params:
            continue                      # demanded side moves with the input
        param, lo, hi = aliases[inp_side.id]
        cur = env.get(param, _EVAL_FAILED)
        dem_val = _safe_eval(dem_side, env)
        if cur is _EVAL_FAILED or dem_val is _EVAL_FAILED:
            continue
        proposal = _splice(cur, lo, hi, dem_val)
        if proposal is None or _same_value(proposal, cur):
            continue
        out.append(Correspondence(
            input_name=param, kind="substring", demanded=dem_val, proposal=proposal,
            confidence=0.8, comparison=cmp_src,
            operand=f"{inp_side.id} = {param}[{'' if lo is None else lo}:"
                    f"{'' if hi is None else hi}]",
        ))
    return out


def _value_level(
    cmp_node: ast.Compare, env: dict[str, Any], params: set[str], cmp_src: str
) -> list[Correspondence]:
    """Correspondences found by *observed value*, with no syntactic link to a parameter.

    This is RedQueen's actual mechanism, and the reason it survives unmodeled code: a
    guard like ``head == expected()`` where ``head = token[:5]`` was computed earlier
    mentions no parameter at all, so a syntactic search finds nothing.  But at runtime
    ``head`` *holds the parameter's value* — and that is the correspondence.  We never
    have to know how the value got there, which is precisely the case symbolic execution
    cannot handle.

    Confidence stays below the syntactic kinds: the link is inferred from one
    observation, so it can be coincidence.  Execution remains the judge.
    """
    from tharkun.solve.condition import _free_names

    out: list[Correspondence] = []
    left, right = cmp_node.left, cmp_node.comparators[0]

    for inp_side, dem_side in ((left, right), (right, left)):
        # The demanded side must still be parameter-free, else its value moves when we
        # change the input and "what it wants" is not a fixed target.
        if _free_names(dem_side) & params:
            continue
        dem_val = _safe_eval(dem_side, env)
        inp_val = _safe_eval(inp_side, env)
        if dem_val is _EVAL_FAILED or inp_val is _EVAL_FAILED:
            continue
        if _same_value(inp_val, dem_val):
            continue                      # already equal; nothing to propose

        for p in sorted(params):
            cur = env.get(p, _EVAL_FAILED)
            if cur is _EVAL_FAILED:
                continue
            # (a) The operand carries the parameter's value verbatim.
            if _same_value(cur, inp_val):
                if _same_value(dem_val, cur):
                    continue                      # no-op: already holds that value
                out.append(Correspondence(
                    input_name=p, kind="identity", demanded=dem_val, proposal=dem_val,
                    confidence=0.55, comparison=cmp_src, operand=ast.unparse(inp_side),
                ))
                continue
            # (b) The parameter's value appears *inside* the operand — the operand is a
            # slice/segment of it.  Splice the demanded value into the matching window.
            if (isinstance(cur, (str, bytes)) and isinstance(inp_val, type(cur))
                    and isinstance(dem_val, type(cur)) and inp_val and inp_val in cur):
                start = cur.find(inp_val)
                proposal = _splice(cur, start, start + len(inp_val), dem_val)
                if proposal is not None:
                    out.append(Correspondence(
                        input_name=p, kind="substring", demanded=dem_val,
                        proposal=proposal, confidence=0.5, comparison=cmp_src,
                        operand=ast.unparse(inp_side),
                    ))
    return out


def _classify(
    op: ast.cmpop,
    inp_side: ast.expr,
    param: str,
    cur: Any,
    inp_val: Any,
    dem_val: Any,
    cmp_src: str,
) -> Correspondence | None:
    """Turn one (input-side, demanded-value) observation into a proposal, or ``None``."""
    operand_src = ast.unparse(inp_side)

    def make(kind: str, proposal: Any, conf: float) -> Correspondence | None:
        # A proposal equal to the value already held cannot flip anything — it is the
        # state we just observed failing. Discard it rather than spend a verification
        # round on a no-op. (It arises when the "demanded" side is a local derived from
        # the input: syntactically constant, but in truth a moving target.)
        if cur is not _EVAL_FAILED and _same_value(proposal, cur):
            return None
        return Correspondence(
            input_name=param, kind=kind, demanded=dem_val, proposal=proposal,
            confidence=conf, comparison=cmp_src, operand=operand_src,
        )

    # (1) Syntactic identity: the operand is the bare parameter.  ``p == d`` ⟹ p = d.
    # Highest confidence — no transform sits between input and compare.
    if isinstance(inp_side, ast.Name) and inp_side.id == param:
        return make("identity", dem_val, 0.95)   # None when it would be a no-op

    # (2) Numeric offset: operand is ``p + k`` or ``p - k`` with a small concrete k.
    off = _match_offset(inp_side, param)
    if off is not None and isinstance(dem_val, (int, float)) and not isinstance(dem_val, bool):
        k, _p_is_left, is_add = off
        # p + k == d ⟹ p = d - k ;  k + p == d ⟹ p = d - k ;  p - k == d ⟹ p = d + k.
        # (``k - p`` is not matched, so ``not is_add`` always means the ``p - k`` shape.)
        proposal = (dem_val - k) if is_add else (dem_val + k)
        return make("offset", proposal, 0.85)

    # (3) Substring / prefix: operand is a slice ``p[a:b]`` of a str/bytes input.  Splice
    # the demanded value into that window so the slice equals it.
    sub = _match_slice(inp_side, param)
    if sub is not None and isinstance(cur, (str, bytes)) and isinstance(dem_val, (str, bytes)):
        lo, hi = sub
        proposal = _splice(cur, lo, hi, dem_val)
        if proposal is not None:
            kind = "prefix" if (lo in (0, None)) else "substring"
            # ``prefix`` is reported under the "substring" family; keep the label precise
            # for diagnostics but the kind field stays within the documented vocabulary.
            return make("substring", proposal, 0.7 if kind == "substring" else 0.75)

    # (3b) Integer decoding: the operand is ``int.from_bytes(p[a:b], order)`` or an
    # equivalent ``struct.unpack``.  The comparison is against an int, so the input-side
    # is bytes and the demanded value has to be ENCODED back before it can be spliced in.
    # This is the shape binary parsers are made of — a magic/length/version field read
    # out of a header — and it is precisely where RedQueen is supposed to win, because
    # the demanded value exists only at runtime and no static solver can read it out of
    # the source.
    dec = _match_int_decode(inp_side, param)
    if dec is not None and isinstance(cur, (bytes, bytearray)) and isinstance(dem_val, int):
        lo, hi, order, signed = dec
        # `p[:2]` reports lo=None, meaning "from the start" — treat it as 0 rather than
        # falling back to a width guess, or a 2-byte field gets encoded at whatever
        # width the demanded value happens to need and the splice lands wrong.
        start = 0 if lo is None else lo
        width = (hi - start) if hi is not None else max(1, (dem_val.bit_length() + 7) // 8)
        if width <= 0:
            width = max(1, (dem_val.bit_length() + 7) // 8)
        try:
            raw = int(dem_val).to_bytes(width, order, signed=signed)
        except (OverflowError, ValueError):
            raw = None
        if raw is not None:
            proposal = _splice(bytes(cur), lo, hi, raw)
            if proposal is not None:
                return make("int_decode", proposal, 0.9)

    # (4) Identity through an unmodeled transform.  The operand depends on the parameter
    # but is not one of the shapes above.  If the transform is *value-identity* on the
    # observed seed (``token.swapcase().swapcase()`` == ``token``), copying the demanded
    # value onto the parameter is a strong bet; otherwise it is a weak, type-gated guess.
    # Either way the driver's execution check is the final judge, so a wrong guess is free.
    if inp_val is not _EVAL_FAILED and cur is not _EVAL_FAILED:
        if _same_value(inp_val, cur):
            return make("identity", dem_val, 0.6)
        if type(cur) is type(dem_val):
            # Types line up (str→str, int→int): the transform may still be identity-like
            # for a *different* input than the seed.  Low confidence, tried last.
            return make("identity", dem_val, 0.35)
    return None


# ---------------------------------------------------------------------------
# Shape matchers
# ---------------------------------------------------------------------------


def _match_int_decode(
    node: ast.expr, param: str
) -> tuple[int | None, int | None, str, bool] | None:
    """Match ``int.from_bytes(p[a:b], order)`` / ``struct.unpack(fmt, p[a:b])[0]``.

    Returns ``(lo, hi, byteorder, signed)`` for the slice of *param* that is decoded, so
    the caller can encode a demanded integer back into exactly those bytes.
    """
    # struct.unpack(...)[0] -> unwrap the subscript
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Call):
        node = node.value

    if not isinstance(node, ast.Call):
        return None

    fn = node.func
    # int.from_bytes(buf, "big") / int.from_bytes(buf, byteorder="little", signed=True)
    if (isinstance(fn, ast.Attribute) and fn.attr == "from_bytes"
            and isinstance(fn.value, ast.Name) and fn.value.id == "int"):
        if not node.args:
            return None
        window = _match_slice(node.args[0], param)
        if window is None:
            window = ((0, None) if isinstance(node.args[0], ast.Name)
                      and node.args[0].id == param else None)
        if window is None:
            return None
        order = "big"
        if len(node.args) > 1 and isinstance(node.args[1], ast.Constant):
            order = str(node.args[1].value)
        signed = False
        for kw in node.keywords:
            if kw.arg == "byteorder" and isinstance(kw.value, ast.Constant):
                order = str(kw.value.value)
            elif kw.arg == "signed" and isinstance(kw.value, ast.Constant):
                signed = bool(kw.value.value)
        if order not in ("big", "little"):
            return None
        return (window[0], window[1], order, signed)

    # struct.unpack(fmt, buf) — the format's first char carries the byte order.
    if (isinstance(fn, ast.Attribute) and fn.attr in ("unpack", "unpack_from")
            and isinstance(fn.value, ast.Name) and fn.value.id == "struct"
            and len(node.args) >= 2 and isinstance(node.args[0], ast.Constant)):
        fmt = str(node.args[0].value)
        window = _match_slice(node.args[1], param)
        if window is None:
            return None
        order = "little" if fmt[:1] in ("<", "=", "@") else "big"
        signed = any(c.islower() and c in "bhilq" for c in fmt[1:2])
        return (window[0], window[1], order, signed)

    return None


def _match_offset(
    node: ast.expr, param: str
) -> tuple[int | float, bool, bool] | None:
    """If *node* is ``p ± k`` (k a small numeric constant), return (k, p_is_left, is_add).

    ``p_is_left`` says whether the parameter is the left BinOp operand; ``is_add`` whether
    the operator is ``+``.  Only ``Add`` and ``Sub`` qualify — RedQueen copies a value out
    of a compare, and only an additive offset preserves that copy relationship.
    """
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, (ast.Add, ast.Sub)):
        return None
    left, right = node.left, node.right
    is_add = isinstance(node.op, ast.Add)

    def _is_param(n: ast.expr) -> bool:
        return isinstance(n, ast.Name) and n.id == param

    if _is_param(left):
        k = _const_num(right)
        if k is not None and abs(k) <= _MAX_OFFSET:
            return (k, True, is_add)
    if _is_param(right) and is_add:  # k + p is only a clean copy under addition
        k = _const_num(left)
        if k is not None and abs(k) <= _MAX_OFFSET:
            return (k, False, is_add)
    return None


def _const_num(node: ast.expr) -> int | float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _const_num(node.operand)
        return -inner if inner is not None else None
    return None


def _match_slice(node: ast.expr, param: str) -> tuple[int | None, int | None] | None:
    """If *node* is ``p[a:b]`` with constant (or omitted) bounds, return (a, b)."""
    if not isinstance(node, ast.Subscript):
        return None
    if not (isinstance(node.value, ast.Name) and node.value.id == param):
        return None
    sl = node.slice
    if not isinstance(sl, ast.Slice) or sl.step is not None:
        return None
    lo = _const_num(sl.lower) if sl.lower is not None else None
    hi = _const_num(sl.upper) if sl.upper is not None else None
    if (sl.lower is not None and lo is None) or (sl.upper is not None and hi is None):
        return None  # a non-constant bound; we cannot place the splice reliably
    lo = int(lo) if lo is not None else None
    hi = int(hi) if hi is not None else None
    return (lo, hi)


def slice_aliases(source_path: str, guard_line: int, params: set[str]) -> dict[str, tuple[str, int | None, int | None]]:
    """Locals that are a constant-bounded slice of a parameter, in the enclosing function.

    Parsers routinely split an input up front — ``head = data[:4]``, ``body = data[4:]``
    — and then guard on the *pieces*.  A guard on ``body`` mentions no parameter, and
    value-level matching cannot locate the window either: when ``body`` is empty (the
    common case for a short seed) its value gives no positional information at all, so
    there is nothing to match on and the input can never be grown to satisfy it.

    Reading the assignment recovers what the value cannot: ``body`` *is* ``data[4:]``, so
    a demand on ``body`` is a demand on that window of ``data``, and splicing there may
    legitimately lengthen the input.

    Returns ``{local: (param, lo, hi)}``.  A local reassigned anywhere else in the
    function is dropped — the alias must hold unconditionally to be safe to invert.
    """
    from pathlib import Path

    try:
        source = Path(os.path.realpath(source_path)).read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(source)
    except (OSError, SyntaxError):
        return {}

    fn = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno) or node.lineno
            if node.lineno <= guard_line <= end and (fn is None or node.lineno > fn.lineno):
                fn = node
    if fn is None:
        return {}

    aliases: dict[str, tuple[str, int | None, int | None]] = {}
    assign_counts: dict[str, int] = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        assign_counts[target.id] = assign_counts.get(target.id, 0) + 1
        for p in params:
            window = _match_slice(node.value, p)
            if window is not None:
                aliases[target.id] = (p, window[0], window[1])
                break

    # An alias only holds if the local is bound exactly once; a later rebinding would
    # make the window a lie.
    return {k: v for k, v in aliases.items() if assign_counts.get(k, 0) == 1}


def _splice(cur: Any, lo: int | None, hi: int | None, dem: Any) -> Any | None:
    """Return a copy of *cur* whose ``[lo:hi]`` window equals *dem* (extending if needed).

    str and bytes must not be mixed; a mismatch means we cannot form a valid proposal.
    We prioritise the slice *equalling* ``dem`` — the window may end up a different width
    than the original, which is correct: a prefix check ``p[:5] == "MAGIC"`` wants those
    exact bytes there regardless of how long the seed happened to be.
    """
    if isinstance(cur, str):
        if not isinstance(dem, str):
            return None
        filler = "\x00"
    elif isinstance(cur, bytes):
        if not isinstance(dem, bytes):
            return None
        filler = b"\x00"
    else:
        return None

    start = 0 if lo is None else (lo if lo >= 0 else max(len(cur) + lo, 0))
    # Grow the buffer with a benign filler so the window's start index exists.
    if start > len(cur):
        cur = cur + filler * (start - len(cur))
    head = cur[:start]
    if hi is None:
        return head + dem
    tail = cur[hi:] if hi <= len(cur) else cur[len(cur):]
    return head + dem + tail


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_eval(node: ast.expr, env: dict[str, Any]) -> Any:
    """Evaluate an operand expression against the live env, or return ``_EVAL_FAILED``.

    Guards are assumed pure (the same assumption minleaf makes when it evaluates the real
    guard).  Evaluating an operand runs the subject's own transform code, which is exactly
    how we *observe* the demanded value rather than trying to model it.
    """
    try:
        code = compile(ast.Expression(body=node), "<redqueen>", "eval")
        return eval(code, dict(env))  # copy: never let a transform mutate the caller's env
    except Exception:
        return _EVAL_FAILED


def _same_value(a: Any, b: Any) -> bool:
    """Value equality that also requires matching type, so ``1 == True`` does not fool us."""
    try:
        return type(a) is type(b) and a == b
    except Exception:
        return False


def _dedup(corrs: list[Correspondence]) -> list[Correspondence]:
    """Drop later correspondences that repeat an earlier (input, proposal) at lower conf."""
    seen: set[tuple] = set()
    out: list[Correspondence] = []
    for c in corrs:
        try:
            key = (c.input_name, c.kind, type(c.proposal), repr(c.proposal))
        except Exception:
            key = (c.input_name, c.kind, id(c.proposal))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
