"""Module-level egglog term language and rules — imported only when egglog is present.

Lifted from the standalone egglog study, with the four NaN-unsound ordering-negation
rewrites removed. egglog needs its sort and functions defined at module scope with
resolvable annotations, so this cannot live inside a lazily-built class; ``tharkun.egraph``
imports it behind an availability gate instead.

Not for direct use — go through ``tharkun.egraph.canonicalize``, which gates, catches, and
logs. This module raises if egglog is missing.
"""


from __future__ import annotations

import ast
import sys
from typing import Optional

from egglog import (
    EGraph,
    Expr,
    StringLike,
    Vec,
    function,
    get_callable_args,
    get_callable_fn,
    get_literal_value,
    i64Like,
    rewrite,
    ruleset,
    var,
    vars_,
)
from egglog import i64 as _i64

# --------------------------------------------------------------------------
# 1. THE TERM LANGUAGE
# --------------------------------------------------------------------------
# One sort, `Term`, for everything.  A single sort keeps the rewrite rules
# simple and lets a comparison appear as an operand of a boolean connective
# without any wrapper juggling.


class Term(Expr):
    """Every guard sub-expression is a Term."""

    @classmethod
    def var(cls, name: StringLike) -> Term: ...          # a program variable

    @classmethod
    def int(cls, v: i64Like) -> Term: ...                # integer literal

    @classmethod
    def str(cls, v: StringLike) -> Term: ...             # string literal

    @classmethod
    def bytes(cls, v: StringLike) -> Term: ...           # bytes literal, repr-encoded

    @classmethod
    def const(cls, name: StringLike) -> Term: ...        # None / True / False / big ints

    @classmethod
    def call(cls, name: StringLike, args: Vec[Term]) -> Term: ...
    # ^ UNINTERPRETED.  `int.from_bytes(token[:2], 'big')` becomes
    #   Term.call("int.from_bytes", Vec(<token[:2]>, Term.str("big")))
    #   It participates structurally -- rules can move it around, compare it,
    #   put it under a Not -- but no rule ever looks inside it, so none of its
    #   semantics are (unsoundly) assumed.


# Comparisons.  Two directions per relation so a rewrite can flip between them.
@function
def Lt(a: Term, b: Term) -> Term: ...
@function
def Le(a: Term, b: Term) -> Term: ...
@function
def Gt(a: Term, b: Term) -> Term: ...
@function
def Ge(a: Term, b: Term) -> Term: ...
@function
def Eq(a: Term, b: Term) -> Term: ...
@function
def Ne(a: Term, b: Term) -> Term: ...


# Booleans.
@function
def And(a: Term, b: Term) -> Term: ...
@function
def Or(a: Term, b: Term) -> Term: ...
@function
def Not(a: Term) -> Term: ...


# Arithmetic / bitwise.
@function
def Add(a: Term, b: Term) -> Term: ...
@function
def Sub(a: Term, b: Term) -> Term: ...
@function
def Mul(a: Term, b: Term) -> Term: ...
@function
def Mod(a: Term, b: Term) -> Term: ...
@function
def BitAnd(a: Term, b: Term) -> Term: ...
@function
def BitOr(a: Term, b: Term) -> Term: ...
@function
def Shl(a: Term, b: Term) -> Term: ...
@function
def Shr(a: Term, b: Term) -> Term: ...


# --------------------------------------------------------------------------
# 2. REWRITE RULES
# --------------------------------------------------------------------------

_a, _b, _c, _v, _t = vars_("a b c v t", Term)
_i = var("i", _i64)
_j = var("j", _i64)
_k = var("k", _i64)

# --- Unconditional: comparison flipping, De Morgan, double negation, comm ---
BASE_RULES = ruleset(
    # comparison direction.  `5 < x`  ==  `x > 5`
    rewrite(Lt(_a, _b)).to(Gt(_b, _a)),
    rewrite(Gt(_a, _b)).to(Lt(_b, _a)),
    rewrite(Le(_a, _b)).to(Ge(_b, _a)),
    rewrite(Ge(_a, _b)).to(Le(_b, _a)),
    # equality is symmetric
    rewrite(Eq(_a, _b)).to(Eq(_b, _a)),
    rewrite(Ne(_a, _b)).to(Ne(_b, _a)),
    # double negation
    rewrite(Not(Not(_a))).to(_a),
    # De Morgan, both directions
    rewrite(Not(And(_a, _b))).to(Or(Not(_a), Not(_b))),
    rewrite(Or(Not(_a), Not(_b))).to(Not(And(_a, _b))),
    rewrite(Not(Or(_a, _b))).to(And(Not(_a), Not(_b))),
    rewrite(And(Not(_a), Not(_b))).to(Not(Or(_a, _b))),
    # negated comparison -> complementary comparison.
    # NOTE: only sound for a TOTAL order.  True for int/str; NOT true for
    # floats with NaN (`not (x < y)` != `x >= y`).  DROPPED here (tharkun keeps the
    # NaN-safe forms in the orientation engine); the four ordering-negation rewrites
    # your guards can see NaN.
    # Eq/Ne complementarity (sound in general)
    rewrite(Not(Eq(_a, _b))).to(Ne(_a, _b)),
    rewrite(Ne(_a, _b)).to(Not(Eq(_a, _b))),
    # commutative arithmetic
    rewrite(Add(_a, _b)).to(Add(_b, _a)),
    rewrite(Mul(_a, _b)).to(Mul(_b, _a)),
    rewrite(BitAnd(_a, _b)).to(BitAnd(_b, _a)),
    rewrite(BitOr(_a, _b)).to(BitOr(_b, _a)),
)

# --- CONDITIONAL: subsumption with a side condition on constants -----------
#
#   And(Gt(v, i), Gt(v, j))  ==  Gt(v, i)   when i > j
#
# Syntax: extra positional args to `.to()` are the side conditions (FactLike).
# `_i > _j` where _i,_j are egglog i64 vars evaluates to `Unit`, which is
# exactly what a side condition wants.  The `Term.int(_i)` pattern is what
# binds the raw i64 out of the wrapped literal.
#
# Both an adjacent-pair form and a "head of a right-associated chain" form are
# given, because after frontend sorting the conjuncts of one variable land
# next to each other in a right-nested And.
SUBSUMPTION_RULES = ruleset(
    # x > i  and  x > j   ==>   x > max(i, j)
    rewrite(And(Gt(_v, Term.int(_i)), Gt(_v, Term.int(_j)))).to(Gt(_v, Term.int(_i)), _i > _j),
    rewrite(And(Gt(_v, Term.int(_i)), And(Gt(_v, Term.int(_j)), _c))).to(
        And(Gt(_v, Term.int(_i)), _c), _i > _j
    ),
    # x < i  and  x < j   ==>   x < min(i, j)
    rewrite(And(Lt(_v, Term.int(_i)), Lt(_v, Term.int(_j)))).to(Lt(_v, Term.int(_i)), _i < _j),
    rewrite(And(Lt(_v, Term.int(_i)), And(Lt(_v, Term.int(_j)), _c))).to(
        And(Lt(_v, Term.int(_i)), _c), _i < _j
    ),
    # x >= i and x >= j  ==>  x >= max(i, j)
    rewrite(And(Ge(_v, Term.int(_i)), Ge(_v, Term.int(_j)))).to(Ge(_v, Term.int(_i)), _i > _j),
    # x <= i and x <= j  ==>  x <= min(i, j)
    rewrite(And(Le(_v, Term.int(_i)), Le(_v, Term.int(_j)))).to(Le(_v, Term.int(_i)), _i < _j),
)

# AC rules.  Correct, but they are what makes saturation explode -- 9 conjuncts
# costs ~0.8 s.  Kept here for reference; the fast path does AC in Python.

# Constant folding, identities, and linear orientation -- mirrors the orientation engine
# so both satisfy the "constant-folding-and-linear" catalog group. egglog computes the new
# i64 in the result (`_i + _j`, `_i - _k`), the same mechanism subsumption uses for max/min.
def _linear(rel):
    # (t + k) rel c  ->  t rel (c - k) ;  (t - k) rel c  ->  t rel (c + k).
    return [
        rewrite(rel(Add(_t, Term.int(_k)), Term.int(_i))).to(rel(_t, Term.int(_i - _k))),
        rewrite(rel(Sub(_t, Term.int(_k)), Term.int(_i))).to(rel(_t, Term.int(_i + _k))),
    ]


FOLD_RULES = ruleset(
    rewrite(Add(Term.int(_i), Term.int(_j))).to(Term.int(_i + _j)),
    rewrite(Sub(Term.int(_i), Term.int(_j))).to(Term.int(_i - _j)),
    rewrite(Mul(Term.int(_i), Term.int(_j))).to(Term.int(_i * _j)),
    rewrite(Add(_a, Term.int(0))).to(_a),
    rewrite(Sub(_a, Term.int(0))).to(_a),
    rewrite(Mul(_a, Term.int(1))).to(_a),
    *_linear(Gt), *_linear(Lt), *_linear(Ge), *_linear(Le), *_linear(Eq), *_linear(Ne),
)

AC_RULES = ruleset(
    rewrite(And(_a, _b)).to(And(_b, _a)),
    rewrite(Or(_a, _b)).to(Or(_b, _a)),
    rewrite(And(And(_a, _b), _c)).to(And(_a, And(_b, _c))),
    rewrite(And(_a, And(_b, _c))).to(And(And(_a, _b), _c)),
    rewrite(Or(Or(_a, _b), _c)).to(Or(_a, Or(_b, _c))),
    rewrite(Or(_a, Or(_b, _c))).to(Or(Or(_a, _b), _c)),
)


# --------------------------------------------------------------------------
# 3. THE COST MODEL -- this is what makes extraction canonical
# --------------------------------------------------------------------------

def _lex_cost(egraph, expr, children_costs):
    """Cost = (size, canonical_string): a TOTAL order over terms.

    The extractor calls this once per e-node, bottom up.  `expr` is the e-node
    with its children replaced by opaque e-class handles -- so we must NOT use
    str(expr).  `children_costs` holds the already-chosen minimum cost of each
    child e-class, and because our cost carries the child's canonical string,
    we can rebuild the whole term's string bottom-up.

    Because the string component makes the order total, the minimum of an
    e-class is unique.  That removes the tie-break-by-insertion-order that
    makes the default cost model non-canonical.
    """
    lit = get_literal_value(expr)
    if lit is not None:
        return (1, repr(lit))
    fn = get_callable_fn(expr)
    head = repr(fn) if fn is not None else str(expr)
    size = 1 + sum(k[0] for k in children_costs)
    if not children_costs:
        return (size, head)
    return (size, head + "(" + ", ".join(k[1] for k in children_costs) + ")")


# --------------------------------------------------------------------------
# 4. PYTHON AST  ->  Term
# --------------------------------------------------------------------------

_I64_MIN, _I64_MAX = -(2**63), 2**63 - 1

_CMP = {
    ast.Lt: Lt, ast.LtE: Le, ast.Gt: Gt, ast.GtE: Ge,
    ast.Eq: Eq, ast.NotEq: Ne,
}
_BIN = {
    ast.Add: Add, ast.Sub: Sub, ast.Mult: Mul, ast.Mod: Mod,
    ast.BitAnd: BitAnd, ast.BitOr: BitOr, ast.LShift: Shl, ast.RShift: Shr,
}


class Unrepresentable(Exception):
    """The expression has no faithful Term encoding."""


def _dotted(node: ast.AST) -> Optional[str]:
    """`int.from_bytes` / `os.path.exists` -> a dotted name, else None."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return f"{base}.{node.attr}" if base else None
    return None


def _to_term(node: ast.AST) -> Term:
    # literals
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool) or v is None:
            return Term.const(repr(v))
        if isinstance(v, int):
            if _I64_MIN <= v <= _I64_MAX:
                return Term.int(v)
            return Term.const(repr(v))          # big int: keep as opaque constant
        if isinstance(v, str):
            return Term.str(v)
        if isinstance(v, bytes):
            return Term.bytes(repr(v))
        return Term.const(repr(v))              # float, complex, Ellipsis...

    if isinstance(node, ast.Name):
        return Term.var(node.id)

    # unary
    if isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            return Not(_to_term(node.operand))
        if isinstance(node.op, ast.USub):
            inner = node.operand
            if isinstance(inner, ast.Constant) and isinstance(inner.value, int) \
                    and not isinstance(inner.value, bool) and _I64_MIN <= -inner.value <= _I64_MAX:
                return Term.int(-inner.value)
            return Term.call("-", Vec(_to_term(inner)))
        if isinstance(node.op, ast.Invert):
            return Term.call("~", Vec(_to_term(node.operand)))
        if isinstance(node.op, ast.UAdd):
            return _to_term(node.operand)

    # boolean
    if isinstance(node, ast.BoolOp):
        op = And if isinstance(node.op, ast.And) else Or
        vals = [_to_term(v) for v in node.values]
        out = vals[-1]
        for t in reversed(vals[:-1]):
            out = op(t, out)
        return out

    # binary
    if isinstance(node, ast.BinOp):
        fn = _BIN.get(type(node.op))
        if fn is None:
            raise Unrepresentable(f"binop {type(node.op).__name__}")
        return fn(_to_term(node.left), _to_term(node.right))

    # comparison (chained comparisons are expanded into an And)
    if isinstance(node, ast.Compare):
        operands = [node.left, *node.comparators]
        parts = []
        for k, op in enumerate(node.ops):
            fn = _CMP.get(type(op))
            if fn is None:
                # `in`, `not in`, `is`, `is not` -> uninterpreted, but structural
                name = type(op).__name__.lower()
                parts.append(Term.call(name, Vec(_to_term(operands[k]), _to_term(operands[k + 1]))))
            else:
                parts.append(fn(_to_term(operands[k]), _to_term(operands[k + 1])))
        out = parts[-1]
        for t in reversed(parts[:-1]):
            out = And(t, out)
        return out

    # uninterpreted: calls, subscripts, attributes
    if isinstance(node, ast.Call):
        name = _dotted(node.func)
        if name is None:
            raise Unrepresentable("call with a non-name callee")
        args = [_to_term(a) for a in node.args]
        for kw in node.keywords:
            if kw.arg is None:
                raise Unrepresentable("**kwargs in call")
            args.append(Term.call(f"={kw.arg}", Vec(_to_term(kw.value))))
        return Term.call(name, Vec(*args) if args else Vec[Term]())

    if isinstance(node, ast.Attribute):
        name = _dotted(node)
        if name is not None:
            return Term.var(name)
        return Term.call(".{}".format(node.attr), Vec(_to_term(node.value)))

    if isinstance(node, ast.Subscript):
        return Term.call("[]", Vec(_to_term(node.value), _to_term(node.slice)))

    if isinstance(node, ast.Slice):
        lo = _to_term(node.lower) if node.lower else Term.const("None")
        hi = _to_term(node.upper) if node.upper else Term.const("None")
        st = _to_term(node.step) if node.step else Term.const("None")
        return Term.call("slice", Vec(lo, hi, st))

    if isinstance(node, ast.Tuple):
        return Term.call("tuple", Vec(*[_to_term(e) for e in node.elts]) if node.elts else Vec[Term]())

    raise Unrepresentable(type(node).__name__)


# --------------------------------------------------------------------------
# 5. Term  ->  Python source
# --------------------------------------------------------------------------
# There is no built-in "print this egglog term as Python".  `str(term)` gives
# egglog-flavoured Python (`Lt(Term.int(5), Term.var("x"))`), which is already a
# perfectly good canonical KEY.  To get real Python source back, the caller
# walks the term with get_callable_fn / get_callable_args / get_literal_value.

_INFIX = {
    "Lt": ("<", 6), "Le": ("<=", 6), "Gt": (">", 6), "Ge": (">=", 6),
    "Eq": ("==", 6), "Ne": ("!=", 6),
    "Or": ("or", 3), "And": ("and", 4),
    "BitOr": ("|", 7), "BitAnd": ("|&"[1], 9),
    "Shl": ("<<", 10), "Shr": (">>", 10),
    "Add": ("+", 11), "Sub": ("-", 11), "Mul": ("*", 12), "Mod": ("%", 12),
}
_ATOM = 100


def _unparse(t, prec: int = 0) -> str:
    """Walk an extracted Term and render Python source."""
    fn = get_callable_fn(t)
    name = repr(fn) if fn is not None else None
    args = get_callable_args(t) or ()

    if name == "Term.var":
        return str(get_literal_value(args[0]))
    if name == "Term.int":
        return repr(get_literal_value(args[0]))
    if name == "Term.str":
        return repr(get_literal_value(args[0]))
    if name in ("Term.bytes", "Term.const"):
        return str(get_literal_value(args[0]))

    if name == "Term.call":
        cname = str(get_literal_value(args[0]))
        vec = args[1]
        items = [_unparse(x, 0) for x in vec]
        if cname == "[]":
            return f"{items[0]}[{items[1]}]"
        if cname == "slice":
            lo, hi, st = items
            body = ("" if lo == "None" else lo) + ":" + ("" if hi == "None" else hi)
            if st != "None":
                body += ":" + st
            return body
        if cname == "tuple":
            return "(" + ", ".join(items) + ("," if len(items) == 1 else "") + ")"
        if cname in ("-", "~"):
            return f"{cname}{items[0]}"
        if cname in ("in", "notin", "is", "isnot"):
            opmap = {"in": "in", "notin": "not in", "is": "is", "isnot": "is not"}
            s = f"{items[0]} {opmap[cname]} {items[1]}"
            return f"({s})" if prec > 6 else s
        return f"{cname}({', '.join(items)})"

    if name == "Not":
        s = "not " + _unparse(args[0], 5)
        return f"({s})" if prec > 5 else s

    if name in _INFIX:
        op, p = _INFIX[name]
        s = f"{_unparse(args[0], p)} {op} {_unparse(args[1], p + 1)}"
        return f"({s})" if prec > p else s

    return str(t)   # should not happen


# --------------------------------------------------------------------------
# 6. AC IN THE FRONTEND  (linear, unlike the egglog AC rules)
# --------------------------------------------------------------------------

def _flatten(node: ast.AST, op_type):
    if isinstance(node, ast.BoolOp) and isinstance(node.op, op_type):
        out = []
        for v in node.values:
            out.extend(_flatten(v, op_type))
        return out
    return [node]


def _ac_sort(node: ast.AST) -> ast.AST:
    """Flatten / dedupe / sort every and-chain and or-chain, bottom up.

    This is associativity+commutativity done in linear time.  The equivalent
    egglog rewrites are correct but explode: 9 conjuncts cost ~0.8 s.
    """
    for field, value in list(ast.iter_fields(node)):
        if isinstance(value, ast.AST):
            setattr(node, field, _ac_sort(value))
        elif isinstance(value, list):
            setattr(node, field, [_ac_sort(v) if isinstance(v, ast.AST) else v for v in value])
    if isinstance(node, ast.BoolOp):
        parts = _flatten(node, type(node.op))
        keyed = {}
        for p in parts:
            keyed.setdefault(ast.unparse(p), p)      # dedupe by source
        node.values = [keyed[k] for k in sorted(keyed)]  # sort by source
        if len(node.values) == 1:
            return node.values[0]
    return node


# --------------------------------------------------------------------------
# 6b. PREFIX-SLICE == LITERAL  ->  startswith  (also a frontend fold)
# --------------------------------------------------------------------------
# A prefix-slice equality `X[:k] == LIT` (len(LIT) == k) is exactly `X.startswith(LIT)`,
# but the slice and the method call are unrelated Terms and the length side condition is
# not an egglog i64 fact -- so this is folded in the frontend before lifting, the same way
# AC is (`_ac_sort`).  Only the exact case fires: a pure prefix slice, `==`, and a
# str/bytes literal whose length matches the stop.  A length mismatch or an offset/strided
# slice is a different constraint and is left alone, so distinct clauses stay distinct.


def _prefix_slice_stop(node: ast.AST):
    """The k of a pure prefix slice `X[:k]` (start 0, unit step), else None."""
    if not isinstance(node, ast.Slice):
        return None
    lo, st = node.lower, node.step
    lo_ok = lo is None or (isinstance(lo, ast.Constant) and isinstance(lo.value, int)
                           and not isinstance(lo.value, bool) and lo.value == 0)
    st_ok = st is None or (isinstance(st, ast.Constant) and isinstance(st.value, int)
                           and not isinstance(st.value, bool) and st.value == 1)
    up = node.upper
    if (lo_ok and st_ok and isinstance(up, ast.Constant) and isinstance(up.value, int)
            and not isinstance(up.value, bool) and up.value >= 0):
        return up.value
    return None


def _startswith_fold(node: ast.AST) -> ast.AST:
    """Rewrite every qualifying `X[:k] == LIT` to `X.startswith(LIT)`, bottom up."""
    for field, value in list(ast.iter_fields(node)):
        if isinstance(value, ast.AST):
            setattr(node, field, _startswith_fold(value))
        elif isinstance(value, list):
            setattr(node, field, [_startswith_fold(v) if isinstance(v, ast.AST) else v
                                  for v in value])
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        left, right = node.left, node.comparators[0]
        for sub, lit in ((left, right), (right, left)):
            if not isinstance(sub, ast.Subscript):
                continue
            if not (isinstance(lit, ast.Constant) and isinstance(lit.value, (str, bytes))):
                continue
            k = _prefix_slice_stop(sub.slice)
            if k is None or k != len(lit.value):
                continue
            return ast.copy_location(ast.Call(
                func=ast.Attribute(value=sub.value, attr="startswith", ctx=ast.Load()),
                args=[lit], keywords=[]), node)
    return node


# --------------------------------------------------------------------------
# 7. PUBLIC API
# --------------------------------------------------------------------------

_CACHE: dict = {}

# A warmed EGraph per mode.  Installing the 22-rule ruleset into a *fresh*
# EGraph costs ~2.1 ms of the ~3.0 ms per call; installing it once and using
# push/pop for isolation cuts a call to ~0.6 ms.  push/pop fully reverts the
# e-graph, so clauses do not leak between calls (verified: a canonical form is
# unchanged after 300 unrelated clauses have passed through).  The state does
# creep, so the engine is recycled periodically.
_ENGINES: dict = {}
_ENGINE_USES: dict = {}
_ENGINE_TTL = 500


def _schedule(eg: EGraph, deep: bool) -> None:
    eg.run(BASE_RULES.saturate())
    eg.run(FOLD_RULES.saturate())
    if deep:
        eg.run(AC_RULES.saturate())
        eg.run(SUBSUMPTION_RULES.saturate())


def _engine(deep: bool) -> EGraph:
    eg = _ENGINES.get(deep)
    if eg is None or _ENGINE_USES.get(deep, 0) >= _ENGINE_TTL:
        eg = EGraph()
        eg.register(Term.var("_warm"))
        _schedule(eg, deep)                     # install rulesets once
        _ENGINES[deep] = eg
        _ENGINE_USES[deep] = 0
    return eg


def _egglog_canon(term: Term, deep: bool) -> Term:
    """Register, saturate, extract with the total-order cost model."""
    eg = _engine(deep)
    _ENGINE_USES[deep] = _ENGINE_USES.get(deep, 0) + 1
    eg.push()
    try:
        eg.register(term)
        _schedule(eg, deep)
        return eg.extract(term, True, _lex_cost)[0]
    finally:
        eg.pop()


def _one_round(text: str, deep: bool) -> str:
    """egglog pass (local rewrites) followed by a Python AC pass."""
    node = ast.parse(text, mode="eval").body
    node = _startswith_fold(node)              # frontend: prefix-slice == -> startswith
    term = _to_term(node)                      # whole expression, structure intact
    extracted = _egglog_canon(term, deep)      # egglog: flip / De Morgan / comm ...
    src = _unparse(extracted)
    node2 = ast.parse(src, mode="eval").body
    return ast.unparse(_ac_sort(node2))        # Python: AC, then stable printing


def canonicalize(text: str, *, deep: bool = False, cache: bool = True,
                 max_rounds: int = 6) -> str:
    """Map a Python guard expression to a canonical string.

    Logically-equivalent guards (up to the modelled rewrites) return the SAME
    string, so a clause set can be intersected by text identity.

    The two passes feed each other -- reordering an and-chain can expose a
    cheaper egglog representative, and a De Morgan flip can expose a new
    and-chain -- so they are iterated to a fixpoint.  If the pair oscillates
    instead of converging, the lexicographic minimum of the cycle is returned,
    which is still a deterministic function of the equivalence class.

    deep=True also runs the AC + constant-subsumption rules inside egglog.
    Correct, but superlinear in conjunct count -- small clause sets only.

    Never raises.  Anything unrepresentable is returned as-is (normalized by
    ast.unparse when it at least parses), so a hot-path caller cannot be
    broken by an exotic guard.
    """
    key = (text, deep)
    if cache and key in _CACHE:
        return _CACHE[key]
    try:
        cur = ast.unparse(ast.parse(text.strip(), mode="eval").body)
        seen = [cur]
        for _ in range(max_rounds):
            nxt = _one_round(cur, deep)
            if nxt == cur:
                break
            if nxt in seen:                    # cycle -> deterministic tie-break
                cur = min(seen[seen.index(nxt):])
                break
            seen.append(nxt)
            cur = nxt
        out = cur
    except Exception:
        try:
            out = ast.unparse(ast.parse(text.strip(), mode="eval").body)
        except Exception:
            out = text.strip()
    if cache:
        _CACHE[key] = out
    return out


# --------------------------------------------------------------------------
# 8. SELF-TEST
# --------------------------------------------------------------------------

if __name__ == "__main__":
    import statistics
    import time

    print("=" * 74)
    print("EQUIVALENCE GROUPS -- every member of a group must give ONE string")
    print("=" * 74)
    groups = [
        ["5 < x", "x > 5"],
        ["x >= 5", "5 <= x", "not (x < 5)"],
        ["x == 5", "5 == x"],
        ["x != 5", "5 != x", "not (x == 5)"],
        ["not (not (5 < x))", "x > 5", "5 < x"],
        ["not (a and b)", "(not a) or (not b)"],
        ["not (a or b)", "(not a) and (not b)"],
        ["5 < x and y < 9", "y < 9 and 5 < x", "x > 5 and 9 > y", "9 > y and x > 5"],
        ["a and b and c", "c and b and a", "b and (a and c)", "(c and a) and b"],
        ["x + 1 > 0", "0 < 1 + x", "0 < x + 1"],
        ["flags & 4", "4 & flags"],
        ["int.from_bytes(token[:2], 'big') > 100",
         "100 < int.from_bytes(token[:2], 'big')"],
        ["n % 7 == 3", "3 == n % 7"],
        ["0 < x <= 10", "x > 0 and x <= 10", "10 >= x and 0 < x"],
    ]
    ok = True
    for g in groups:
        outs = {canonicalize(s) for s in g}
        good = len(outs) == 1
        ok &= good
        print(f"  [{'OK ' if good else 'FAIL'}] {g[0]!r}")
        print(f"          -> {sorted(outs)}")
        if not good:
            for s in g:
                print(f"             {s!r:45s} -> {canonicalize(s)!r}")

    print()
    print("=" * 74)
    print("NON-EQUIVALENCE -- these must NOT collapse")
    print("=" * 74)
    for p, q in [("x > 5", "x > 6"), ("x > y", "y > x"), ("a and b", "a or b"),
                 ("x < 5", "x <= 5"), ("f(x)", "g(x)")]:
        cp, cq = canonicalize(p), canonicalize(q)
        good = cp != cq
        ok &= good
        print(f"  [{'OK ' if good else 'FAIL'}] {p!r} -> {cp!r}  vs  {q!r} -> {cq!r}")

    print()
    print("=" * 74)
    print("CONDITIONAL SUBSUMPTION (deep=True)")
    print("=" * 74)
    for g in [["x > 5 and x > 3", "x > 3 and x > 5", "3 < x and 5 < x"],
              ["x < 5 and x < 9", "x < 9 and x < 5"]]:
        outs = {canonicalize(s, deep=True) for s in g}
        print(f"  {g} -> {sorted(outs)}  ({'collapsed' if len(outs)==1 else 'NOT collapsed'})")

    print()
    print("=" * 74)
    print("DETERMINISM -- 200 repeats, both orderings")
    print("=" * 74)
    _CACHE.clear()
    seen = set()
    for _ in range(200):
        _CACHE.clear()
        seen.add(canonicalize("5 < x"))
        seen.add(canonicalize("x > 5"))
        seen.add(canonicalize("y < 9 and 5 < x"))
        seen.add(canonicalize("5 < x and y < 9"))
    print(f"  distinct outputs across 200 runs (2 equivalence classes): {len(seen)}")
    for s in sorted(seen):
        print(f"    {s!r}")

    print()
    print("=" * 74)
    print("COST")
    print("=" * 74)
    for label, expr, kw in [("simple  '5 < x'", "5 < x", {}),
                            ("2 conj          ", "5 < x and y < 9", {}),
                            ("6 conj          ", " and ".join(f"v{i} > {i}" for i in range(6)), {}),
                            ("simple, deep    ", "5 < x", {"deep": True})]:
        canonicalize(expr, cache=False, **kw)
        ts = []
        for _ in range(100):
            t0 = time.perf_counter()
            canonicalize(expr, cache=False, **kw)
            ts.append(time.perf_counter() - t0)
        print(f"  {label}: median={statistics.median(ts)*1e3:8.3f} ms   "
              f"p90={sorted(ts)[90]*1e3:8.3f} ms")
    t0 = time.perf_counter()
    for _ in range(10000):
        canonicalize("5 < x")
    t1 = time.perf_counter()
    print(f"  cached hit      : {(t1-t0)/10000*1e6:8.3f} us")

    print()
    print("=" * 74)
    print("FAILURE MODES -- must degrade, never raise")
    print("=" * 74)
    for bad in ["x >", "lambda q: q", "{i for i in y}", "f(*args)",
                "x > 99999999999999999999999999", "await foo()", "x @ y",
                "b'\\x00' in data", "obj.a.b.c == 1"]:
        try:
            print(f"  {bad!r:40s} -> {canonicalize(bad)!r}")
        except Exception as e:
            ok = False
            print(f"  {bad!r:40s} -> RAISED {type(e).__name__}: {e}")

    print()
    print("ALL EQUIVALENCE/NON-EQUIVALENCE CHECKS:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
