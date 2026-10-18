"""One text for one meaning — so a constraint written two ways is one clause.

Path-condition clauses are stored as the source string ``ast.unparse`` produced, and a
clause's identity *is* that string. So ``(x > 5)`` and ``(5 < x)`` are two different
clauses, and because clause sets are merged by INTERSECTION at a control-flow join, a
constraint that two predecessors both established — written differently — is silently lost.
Measured: the join of ``(x > 5)`` and ``(5 < x)`` renders ``True``, and every solver
downstream then works without a fact that genuinely holds.

``canonicalize(text) -> text`` maps logically equivalent guards to the same string, so the
merge sees them as one. It does three things, in order:

1. **NNF** — reuse ``minleaf.guard_normalize``: chained comparisons expanded, negation
   pushed to the leaves (De Morgan + a complete comparison-inversion table), booleans
   flattened.
2. **Orientation** — move the literal to a fixed side and pick one direction per
   comparison, so ``5 < x`` and ``x > 5`` agree. This is what closes the three measured
   failures.
3. **Commutative ordering** — sort the operands of ``And`` / ``Or`` / ``==`` / ``!=`` by a
   total order, so ``a and b`` and ``b and a`` agree.

**Determinism is the whole point.** If the same clause canonicalizes differently across
runs the merge fails anyway, so every ordering here is a *total* order with an explicit
tie-break, and the result must be identical across processes — the discipline
``path_class`` and ``_write_seed`` already follow. Memoized by input text, since the result
feeds ``DecisionKey`` and instability there costs memo hits.

This is deliberately syntactic and shallow. It does not rearrange arithmetic
(``len(x) - 1 > 3`` stays distinct from ``len(x) > 4``) or fold constants — those are the
cases an e-graph reaches and this cannot, and they are handled, if the residue ever
justifies the dependency, by the optional egglog path.
"""

from __future__ import annotations

import ast

# Which direction each comparison is written in canonical form, and how to flip it. We
# keep `<` and `<=` (the literal to the RIGHT is the target after orientation), turning
# `>`/`>=` around. Equality and membership are symmetric enough to leave, but their
# operands still get ordered below.
_FLIP: dict[type, type] = {
    ast.Gt: ast.Lt,
    ast.GtE: ast.LtE,
    ast.Lt: ast.Gt,
    ast.LtE: ast.GtE,
}

# BinOps whose operands may be reordered without changing the value. Sub / Div / Mod / the
# shifts are excluded on purpose — they are not commutative.
_COMMUTATIVE_BINOPS = (ast.Add, ast.Mult, ast.BitAnd, ast.BitOr, ast.BitXor)

# A stable rank per AST node type, so operands of a commutative op have a total order
# regardless of how they were written.
_TYPE_RANK = {
    ast.Constant: 0, ast.Name: 1, ast.Attribute: 2, ast.Subscript: 3,
    ast.Call: 4, ast.BinOp: 5, ast.UnaryOp: 6, ast.BoolOp: 7, ast.Compare: 8,
}


def _key(node: ast.AST) -> tuple:
    """A total order on expressions: type rank, then the unparsed text as tie-break.

    The text tie-break is what makes it *total* — two Names order by their identifier,
    two Calls by their full source — and `ast.unparse` is deterministic, so the order is
    stable across processes.
    """
    try:
        text = ast.unparse(node)
    except Exception:
        text = repr(node)
    return (_TYPE_RANK.get(type(node), 99), text)


def _is_literal(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return True
    # `-1` parses as UnaryOp(USub, Constant), still a literal for orientation purposes.
    return (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant))


def _fold(node: ast.expr) -> ast.expr:
    """Fold constant arithmetic and drop identity operands, bottom-up.

    `5 + 3` -> `8`, `x + 0` -> `x`, `x * 1` -> `x`, `x - 0` -> `x`. Deliberately shallow:
    it does not distribute or reassociate, only evaluates a wholly-constant BinOp and
    removes the identities that appear verbatim in guards. Anything it cannot fold is
    returned untouched.
    """
    if isinstance(node, ast.BinOp):
        left, right = _fold(node.left), _fold(node.right)
        node.left, node.right = left, right
        lv = left.value if isinstance(left, ast.Constant) else None
        rv = right.value if isinstance(right, ast.Constant) else None
        # Identities that hold for any operand.
        if isinstance(node.op, (ast.Add, ast.Sub)) and rv == 0:
            return left
        if isinstance(node.op, ast.Add) and lv == 0:
            return right
        if isinstance(node.op, ast.Mult) and rv == 1:
            return left
        if isinstance(node.op, ast.Mult) and lv == 1:
            return right
        # Wholly-constant: evaluate. Bool is an int subclass; treat it as a value, not a
        # number, so `True + 1` is left alone rather than folded to 2.
        if (isinstance(left, ast.Constant) and isinstance(right, ast.Constant)
                and not isinstance(lv, bool) and not isinstance(rv, bool)
                and isinstance(lv, (int, float)) and isinstance(rv, (int, float))):
            try:
                return ast.copy_location(ast.Constant(value=_apply_binop(node.op, lv, rv)), node)
            except Exception:
                return node
        return node
    # Fold children of anything else, reassigning the result (a BinOp nested under a
    # Compare or BoolOp only folds if its parent stores the folded node back).
    for field_name, value in list(ast.iter_fields(node)):
        if isinstance(value, ast.AST):
            setattr(node, field_name, _fold(value))
        elif isinstance(value, list):
            setattr(node, field_name,
                    [_fold(v) if isinstance(v, ast.AST) else v for v in value])
    return node


def _apply_binop(op: ast.operator, a, b):
    import operator as _op

    table = {ast.Add: _op.add, ast.Sub: _op.sub, ast.Mult: _op.mul,
             ast.Mod: _op.mod, ast.BitAnd: _op.and_, ast.BitOr: _op.or_,
             ast.BitXor: _op.xor, ast.LShift: _op.lshift, ast.RShift: _op.rshift}
    fn = table.get(type(op))
    if fn is None:
        raise ValueError("unfoldable op")
    return fn(a, b)


def _linear_orient(node: ast.expr) -> ast.expr:
    """Move an additive constant across a comparison so the literal side is canonical.

    `len(x) - 1 > 3` -> `len(x) > 4`, `a + 2 == 5` -> `a == 3`. Only the shape
    `(term +/- k) <op> literal` with integer k and literal, which is the common guard —
    a symbolic term on one side, a literal plus an offset on the other. Deferring the
    general case to the e-graph.
    """
    if not (isinstance(node, ast.Compare) and len(node.ops) == 1):
        return node
    left, right = node.left, node.comparators[0]
    for term, lit, swap in ((left, right, False), (right, left, True)):
        if not (isinstance(lit, ast.Constant) and isinstance(lit.value, int)
                and not isinstance(lit.value, bool)):
            continue
        if not (isinstance(term, ast.BinOp) and isinstance(term.op, (ast.Add, ast.Sub))
                and isinstance(term.right, ast.Constant)
                and isinstance(term.right.value, int) and not isinstance(term.right.value, bool)):
            continue
        k = term.right.value
        # (t + k) op c  ->  t op (c - k) ;  (t - k) op c  ->  t op (c + k).
        new_c = lit.value - k if isinstance(term.op, ast.Add) else lit.value + k
        new_lit = ast.copy_location(ast.Constant(value=new_c), lit)
        if swap:
            return ast.copy_location(ast.Compare(left=new_lit, ops=node.ops,
                                                 comparators=[term.left]), node)
        return ast.copy_location(ast.Compare(left=term.left, ops=node.ops,
                                             comparators=[new_lit]), node)
    return node


def _orient(node: ast.AST) -> ast.AST:
    """Recursively orient comparisons and order commutative operands."""
    for child in ast.iter_child_nodes(node):
        _orient(child)

    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        op = node.ops[0]
        left, right = node.left, node.comparators[0]
        # Put a lone literal on the RIGHT: `5 < x` -> `x > 5`. When neither or both sides
        # are literals, order by the total key so the choice is still deterministic.
        flip = False
        if _is_literal(left) and not _is_literal(right):
            flip = True
        elif _is_literal(left) == _is_literal(right):
            flip = _key(left) > _key(right)
        if flip and type(op) in _FLIP:
            node.left, node.comparators[0] = right, left
            node.ops[0] = _FLIP[type(op)]()
        elif flip and isinstance(op, (ast.Eq, ast.NotEq)):
            # Symmetric: reorder operands, keep the operator.
            node.left, node.comparators[0] = right, left

    elif isinstance(node, ast.BoolOp):
        node.values.sort(key=_key)

    elif isinstance(node, ast.BinOp) and isinstance(node.op, _COMMUTATIVE_BINOPS):
        # Commutative arithmetic/bitwise: order operands so `a + b` and `b + a` agree, and
        # `x * 2` and `2 * x`. Sub / Div / Mod / shifts are NOT commutative and are left in
        # place — reordering them would invent a different value. Measured as egglog's only
        # arithmetic edge over orientation, absorbed here to keep egglog a shadow.
        if _key(node.left) > _key(node.right):
            node.left, node.right = node.right, node.left

    return node


def _dedup_boolops(node: ast.expr) -> ast.expr:
    """Drop duplicate operands of ``and`` / ``or``, collapsing ``a and a`` to ``a``.

    Bottom-up and REASSIGNING (like ``_fold``), because a BoolOp that dedups to a single
    operand must be replaced by that operand, which an in-place pass cannot do. Operands are
    compared by unparsed text, so two spellings orientation already unified (``x > 5`` and
    ``5 < x``) count as one — which is the point: idempotence has to see through spelling.
    """
    for field_name, value in list(ast.iter_fields(node)):
        if isinstance(value, ast.AST):
            setattr(node, field_name, _dedup_boolops(value))
        elif isinstance(value, list):
            setattr(node, field_name,
                    [_dedup_boolops(v) if isinstance(v, ast.AST) else v for v in value])
    if isinstance(node, ast.BoolOp):
        seen: dict[str, ast.expr] = {}
        for v in node.values:
            seen.setdefault(ast.unparse(v), v)
        vals = list(seen.values())
        if len(vals) == 1:
            return vals[0]                       # `a and a` -> `a`
        node.values = vals
    return node


def _linear_over_tree(node: ast.expr) -> ast.expr:
    """Apply `_linear_orient` at every comparison in the tree, bottom-up."""
    for field_name, value in list(ast.iter_fields(node)):
        if isinstance(value, ast.AST):
            setattr(node, field_name, _linear_over_tree(value))
        elif isinstance(value, list):
            setattr(node, field_name,
                    [_linear_over_tree(v) if isinstance(v, ast.AST) else v for v in value])
    return _linear_orient(node) if isinstance(node, ast.Compare) else node


def _is_zero_or_none(node: ast.expr | None) -> bool:
    """True for a slice bound that means "from the start": absent or the int 0."""
    return node is None or (isinstance(node, ast.Constant)
                            and isinstance(node.value, int)
                            and not isinstance(node.value, bool)
                            and node.value == 0)


def _is_unit_step_or_none(node: ast.expr | None) -> bool:
    """True for a slice step that means "every element": absent or the int 1."""
    return node is None or (isinstance(node, ast.Constant)
                            and isinstance(node.value, int)
                            and not isinstance(node.value, bool)
                            and node.value == 1)


def _prefix_slice_stop(node: ast.expr) -> int | None:
    """The k of a PURE prefix slice `X[:k]`, or None if this is not one.

    A pure prefix slice starts at 0 and steps by 1 — `X[:k]`, `X[0:k]`, `X[0:k:1]`.
    `X[1:3]` (offset start), `X[:k:2]` (non-unit step) and `X[a:b]` (symbolic stop) all
    return None, since none of them mean "the first k elements" and rewriting them to
    ``startswith`` would invent a fact.
    """
    if not isinstance(node, ast.Slice):
        return None
    if not (_is_zero_or_none(node.lower) and _is_unit_step_or_none(node.step)):
        return None
    stop = node.upper
    if (isinstance(stop, ast.Constant) and isinstance(stop.value, int)
            and not isinstance(stop.value, bool) and stop.value >= 0):
        return stop.value
    return None


def _as_startswith(node: ast.Compare) -> ast.expr | None:
    """`X[:k] == LIT` -> `X.startswith(LIT)` when the rewrite is exact, else None.

    Exact only when the slice is a pure prefix `X[:k]`, the operator is ``==``, and the
    other operand is a ``str``/``bytes`` literal whose length is exactly k. Under those
    conditions `X[:k] == LIT` and `X.startswith(LIT)` agree on every X: they match when X
    is long enough, and both are false when X is shorter than k (a short slice can never
    equal a length-k literal, and ``startswith`` of a longer prefix is false). A length
    mismatch (`X[:2] == b'a'`, `X[:2] == b'abc'`) is a genuinely different constraint and
    is declined — those must stay distinct. The literal may sit on either side; orientation
    has already moved it right for ``==``, but the swapped form is handled for safety.
    """
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq):
        return None
    left, right = node.left, node.comparators[0]
    for sub, lit in ((left, right), (right, left)):
        if not isinstance(sub, ast.Subscript):
            continue
        if not (isinstance(lit, ast.Constant) and isinstance(lit.value, (str, bytes))):
            continue
        k = _prefix_slice_stop(sub.slice)
        if k is None or k != len(lit.value):
            continue
        call = ast.Call(
            func=ast.Attribute(value=sub.value, attr="startswith", ctx=ast.Load()),
            args=[lit], keywords=[])
        return ast.copy_location(call, node)
    return None


def _slice_startswith(node: ast.expr) -> ast.expr:
    """Rewrite every qualifying prefix-slice equality into the ``startswith`` it means.

    Bottom-up so a slice-equality nested under an ``and`` is rewritten too; a non-matching
    comparison is returned untouched. Runs after orientation, so ``token[:2] == b'ab'`` and
    ``b'ab' == token[:2]`` have already collapsed to one shape before this sees them.
    """
    for field_name, value in list(ast.iter_fields(node)):
        if isinstance(value, ast.AST):
            setattr(node, field_name, _slice_startswith(value))
        elif isinstance(value, list):
            setattr(node, field_name,
                    [_slice_startswith(v) if isinstance(v, ast.AST) else v for v in value])
    if isinstance(node, ast.Compare):
        rewritten = _as_startswith(node)
        if rewritten is not None:
            return rewritten
    return node


def _wants_parens(node: ast.AST) -> bool:
    """True for a compound expression the producers would have parenthesized."""
    return isinstance(node, (ast.Compare, ast.BoolOp, ast.BinOp, ast.UnaryOp))


def canonicalize(text: str) -> str:
    """A single canonical source string for whatever *text* means.

    Returns *text* unchanged when it cannot be parsed — a non-expression is left for the
    caller to handle rather than guessed at.
    """
    cached = _CACHE.get(text)
    if cached is not None:
        return cached

    from tharkun.solve.normalize import normalize_guard

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        _CACHE[text] = text
        return text

    try:
        normalized = normalize_guard(tree.body, nan_safe=True)
        folded = _fold(normalized)
        linear = _linear_over_tree(folded)
        oriented = _orient(linear)
        # Drop duplicate and/or operands (`a and a` -> `a`) once orientation has unified how
        # each operand is spelled, so idempotence sees through `x > 5` vs `5 < x`.
        deduped = _dedup_boolops(oriented)
        # A prefix-slice equality IS a `startswith`; fold it after orientation has moved the
        # literal to a fixed side, so both spellings reach this as one shape.
        sliced = _slice_startswith(deduped)
        inner = ast.unparse(sliced)
        # Keep the outer parens the clause producers wrap with — `(x > 5)`, not `x > 5`.
        # ast.unparse drops them, but they make a rendered path condition far easier to
        # read (`(a) and (b)` rather than `a and b`), and wrapping is deterministic so it
        # costs nothing for dedup. A bare atom (a lone name) is left unwrapped.
        result = f"({inner})" if _wants_parens(sliced) else inner
    except Exception:
        result = text                       # never worse than the original

    _CACHE[text] = result
    return result


# Memoized by input text: the result feeds DecisionKey, and re-canonicalizing the same
# clause thousands of times on a hot path is pure waste. Cleared only by process exit.
_CACHE: dict[str, str] = {}
