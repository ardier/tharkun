"""``refute_stdlib`` -- prove a conjunction of Python conditions unsatisfiable, stdlib only ([internal ref]).

SOLVING, by ``docs/LAYERS.md``'s rule: conditions over typed names in, a proof that
no assignment satisfies them all (or no claim) out.  Nothing is executed.

This is the Z3-free procedure behind ``proof:dead-path`` (lane B's ``sv/refute.py``
calls it first): a may-raise site is dead when ``residual AND scope`` has no model.
It answers ``(True, certificate)`` ONLY when that is proven; ``False`` means "not
proven", never "satisfiable".  Two independent procedures, the first that closes
the formula wins:

1. **Exact integer elimination** (:mod:`tharkun.solve.arith`) when every name is an
   ``int`` or a ``bool`` (a bool is an int in ``{0, 1}``; ``True`` / ``False`` are
   ``1`` / ``0``): linear equalities are eliminated exactly over the unbounded
   integers, ``//`` and ``%`` with Python's floor semantics; a constant that the
   coefficients' gcd does not divide, or a ground contradiction, refutes.
2. **Interval propagation** over :mod:`tharkun.solve.values` facts, for ``int``,
   ``float`` (IEEE binary64: NaN, ``-0.0``, infinities) and ``bool`` names: every
   comparison narrows its operand names in both polarities (``not (x < y)`` is NOT
   ``x >= y`` when a NaN is possible -- :func:`values.refine_compare` keeps that
   right), ``or`` joins its branches, and the narrowing is repeated to a fixed
   point.  A name narrowed to the empty set, or a condition that can no longer be
   true, refutes.  Where Python would raise while evaluating a condition (a zero
   divisor, an overflow), the value is left free -- the same convention as the Z3
   encoding -- so the procedure may keep states Python never reaches but never
   drops one.

Anything outside the fragment (strings, attributes, subscripts, unknown calls,
names without an ``int`` / ``float`` / ``bool`` type) makes the procedure refuse
that condition: it is then simply not used, which can only lose a proof.
"""

from __future__ import annotations

import ast
import copy
from typing import Optional

from tharkun.solve import values as V
from tharkun.solve.arith import solve_int_guard
from tharkun.solve.values import FloatFact, IntFact

__all__ = ["refute_stdlib"]

_ROUNDS = 40
_BOTTOM = None                     # an environment that has no model

_CMP = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}
_BIN = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "truediv",
        ast.FloorDiv: "floordiv", ast.Mod: "mod", ast.Pow: "pow"}


class _Refuse(Exception):
    pass


def refute_stdlib(residual: str, scope: list, types: dict) -> tuple:
    """``(True, certificate)`` when ``residual AND all(scope)`` has no model over the
    typed names (``types``: name -> ``"int"`` | ``"float"`` | ``"bool"``); else
    ``(False, reason)``.  Never raises."""
    texts = [residual] + [s for s in (scope or []) if isinstance(s, str) and s.strip()]
    try:
        conds = [ast.parse(t.strip(), mode="eval").body for t in texts]
    except SyntaxError as e:
        return False, f"unparsable condition: {e}"
    formula = " AND ".join(f"({t.strip()})" for t in texts)
    names = {n.id for c in conds for n in ast.walk(c)
             if isinstance(n, ast.Name) and n.id not in ("True", "False", "abs", "min", "max")}
    typed = {n: types.get(n) for n in names}
    try:
        ok, why = _by_integers(conds, typed)
        if ok:
            return True, f"tharkun.solve.arith: unsat({formula}) over the integers ({why}); " + _sig(typed)
        ok, why = _by_intervals(conds, typed)
        if ok:
            return True, f"tharkun.solve.values interval propagation: unsat({formula}) ({why}); " + _sig(typed)
        return False, why
    except RecursionError:
        return False, "formula too deep"


def _sig(typed: dict) -> str:
    return "types " + ", ".join(f"{n}: {t}" for n, t in sorted(typed.items()))


# --------------------------------------------------------------------------- #
# 1. exact integer elimination
# --------------------------------------------------------------------------- #

class _BoolsAsInts(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(value=int(node.value)), node)
        return node

    def visit_Name(self, node):
        if node.id in ("True", "False"):
            return ast.copy_location(ast.Constant(value=int(node.id == "True")), node)
        return node


def _by_integers(conds: list, typed: dict) -> tuple:
    if not typed or any(t not in ("int", "bool") for t in typed.values()):
        return False, "not all names are int / bool"
    parts = [_BoolsAsInts().visit(copy.deepcopy(c)) for c in conds]
    for n, t in typed.items():
        if t == "bool":
            parts.append(ast.parse(f"0 <= {n} <= 1", mode="eval").body)
    expr = parts[0] if len(parts) == 1 else ast.BoolOp(op=ast.And(), values=parts)
    res = solve_int_guard(expr, set(typed))
    if res.status == "unsat":
        return True, res.explanation
    return False, f"integers: {res.status} ({res.explanation})"


# --------------------------------------------------------------------------- #
# 2. interval propagation
# --------------------------------------------------------------------------- #

def _init(t: Optional[str]):
    if t == "int":
        return IntFact.top()
    if t == "float":
        return FloatFact.top()
    if t == "bool":
        return IntFact(0, 1)                    # a bool is the int 0 or 1 in arithmetic
    raise _Refuse(f"no int/float/bool type ({t!r})")


def _is_bottom(f) -> bool:
    return f.is_bottom()


def _eval(node: ast.expr, env: dict):
    """The fact of an arithmetic term (``IntFact`` / ``FloatFact``), or ``None`` when
    the term is outside the fragment (the caller then learns nothing from it)."""
    if isinstance(node, ast.Name):
        if node.id in ("True", "False"):
            return IntFact.const(int(node.id == "True"))
        return env.get(node.id)
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return IntFact.const(int(v))
        if isinstance(v, int):
            return IntFact.const(v)
        if isinstance(v, float):
            return FloatFact.const(v)
        return None
    if isinstance(node, ast.UnaryOp):
        a = _eval(node.operand, env)
        if a is None:
            return None
        if isinstance(node.op, ast.USub):
            return V.neg(a)[0]
        if isinstance(node.op, ast.UAdd):
            return a
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and \
            ast.dump(node.left) == ast.dump(node.right) and _pure_term(node.left):
        a = _eval(node.left, env)          # e * e of ONE value: never negative
        if a is None or a.is_bottom():
            return None
        return V.sq(a)[0]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        a, b = _eval(node.left, env), _eval(node.right, env)
        if a is None or b is None or a.is_bottom() or b.is_bottom():
            return None
        res, raises = V.binop(_BIN[type(node.op)], a, b)
        if res is None:
            return None
        if raises:
            # where Python raises, the condition leaves the value free
            return IntFact.top() if isinstance(res, IntFact) else FloatFact.top()
        return res
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        args = [_eval(a, env) for a in node.args]
        if any(a is None or a.is_bottom() for a in args):
            return None
        fn = {"abs": (V.abs_, 1), "min": (V.min_, 2), "max": (V.max_, 2)}.get(node.func.id)
        if fn is None or len(args) != fn[1]:
            return None
        res, raises = fn[0](*args)
        if res is None or raises:
            return None
        return res
    if isinstance(node, ast.IfExp):
        a, b = _eval(node.body, env), _eval(node.orelse, env)
        if a is None or b is None or type(a) is not type(b):
            return None
        return a.join(b)
    return None


def _pure_term(node: ast.expr) -> bool:
    """Names, constants, ``+ - *`` and signs only: evaluating it twice gives the same value
    and never raises (so no free value is involved)."""
    return all(isinstance(n, (ast.Name, ast.Constant, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub,
                              ast.Mult, ast.USub, ast.UAdd, ast.Load)) for n in ast.walk(node))


def _same_operand(op: str, f, positive: bool):
    """The fact of ``e`` given ``e <op> e`` is ``positive`` for ONE value ``e``: an int
    always equals itself; a float equals itself unless it is NaN."""
    reflexive = op in ("==", "<=", ">=")
    holds_if_number = reflexive
    if isinstance(f, IntFact):
        return f if holds_if_number == positive else IntFact.bottom()
    # float: a number satisfies e op e iff op is reflexive; NaN satisfies only !=
    number_ok = holds_if_number == positive
    nan_ok = (op == "!=") == positive
    return FloatFact(f.lo if number_ok else V._INF, f.hi if number_ok else -V._INF,
                     f.nan and nan_ok)


def _assign(env: dict, node: ast.expr, fact) -> Optional[dict]:
    """``env`` with the name ``node`` narrowed to ``fact`` (a no-op for a non-name)."""
    if not isinstance(node, ast.Name) or node.id not in env:
        return env
    cur = env[node.id]
    if type(fact) is not type(cur):
        return env
    new = cur.meet(fact)
    if new.is_bottom():
        return _BOTTOM
    if new == cur:
        return env
    out = dict(env)
    out[node.id] = new
    return out


def _refine(node: ast.expr, env: dict, positive: bool) -> Optional[dict]:
    """``env`` narrowed to the states where ``node`` is ``positive`` (``None``: none)."""
    if env is _BOTTOM:
        return _BOTTOM
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _refine(node.operand, env, not positive)
    if isinstance(node, ast.BoolOp):
        conj = isinstance(node.op, ast.And) == positive
        if conj:
            for v in node.values:
                env = _refine(v, env, positive)
                if env is _BOTTOM:
                    return _BOTTOM
            return env
        out = _BOTTOM
        for v in node.values:
            branch = _refine(v, env, positive)
            if branch is not _BOTTOM:
                out = branch if out is _BOTTOM else _join(out, branch)
        return out
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return env if node.value == positive else _BOTTOM
    if isinstance(node, ast.Name) and node.id in ("True", "False"):
        return env if (node.id == "True") == positive else _BOTTOM
    if isinstance(node, ast.Compare):
        if len(node.ops) > 1:
            pieces = []
            left = node.left
            for op, right in zip(node.ops, node.comparators):
                pieces.append(ast.Compare(left=left, ops=[op], comparators=[right]))
                left = right
            return _refine(ast.BoolOp(op=ast.And(), values=pieces), env, positive)
        op = _CMP.get(type(node.ops[0]))
        if op is None:
            return env
        a, b = _eval(node.left, env), _eval(node.comparators[0], env)
        if a is None or b is None:
            return env
        if a.is_bottom() or b.is_bottom():
            return _BOTTOM
        if ast.dump(node.left) == ast.dump(node.comparators[0]) and _pure_term(node.left):
            nf = _same_operand(op, a, positive)
            if nf.is_bottom():
                return _BOTTOM
            return _assign(env, node.left, nf)
        at, bt, af, bf = V.refine_compare(op, a, b)
        na, nb = (at, bt) if positive else (af, bf)
        if na.is_bottom() or nb.is_bottom():
            return _BOTTOM
        env = _assign(env, node.left, na)
        if env is _BOTTOM:
            return _BOTTOM
        return _assign(env, node.comparators[0], nb)
    # truthiness of a term: an int is true iff != 0; a float iff != 0.0 (NaN is true)
    f = _eval(node, env)
    if f is None:
        return env
    zero = IntFact.const(0) if isinstance(f, IntFact) else FloatFact.const(0.0)
    ft, _zt, ff, _zf = V.refine_compare("!=", f, zero)
    nf = ft if positive else ff
    if nf.is_bottom():
        return _BOTTOM
    return _assign(env, node, nf)


def _join(a: dict, b: dict) -> dict:
    return {k: a[k].join(b[k]) for k in a}


def _by_intervals(conds: list, typed: dict) -> tuple:
    env = {}
    for n, t in typed.items():
        try:
            env[n] = _init(t)
        except _Refuse:
            continue            # an untyped name: the conditions over it narrow nothing
    if not env:
        return False, "intervals: no int / float / bool names"
    for rnd in range(_ROUNDS):
        before = env
        for c in conds:
            env = _refine(c, env, True)
            if env is _BOTTOM:
                return True, f"empty after {rnd + 1} round(s)"
        if env == before:
            return False, "intervals: fixed point with a non-empty box"
    return False, f"intervals: no fixed point in {_ROUNDS} rounds"
