"""``affine_envelope`` -- a proven affine bracket for an integer expression (#0911).

SOLVING, by ``docs/LAYERS.md``'s rule.  The input is an expression over ONE
named integer variable.  The output is a claim about the expression's value
for EVERY integer value of that variable::

    a * x + lo  <=  expr(x)  <=  a * x + hi

It is computed from the expression's structure alone, with no observed output
and nothing executed.  Nothing here turns the claim into candidates.  A caller
that holds an observation ``y`` does that on its own side (corollary (a)): with
``a != 0``, ``expr(x) == y`` confines ``x`` to the finite interval between
``(y - hi) / a`` and ``(y - lo) / a``, which it enumerates and checks
(corollary (c): a sound superset, consumed by intersection).  The claim is
what makes that enumeration COMPLETE, not a window: no integer outside the
interval can satisfy the equality.

The analysis is an abstract interpretation over "affine plus bounded
residual".  Each subexpression is ``a * x + r`` with a rational ``a`` and a
residual ``r`` known to lie in a closed rational interval ``[lo, hi]``.  The
rules, each sound for every Python ``int`` (unbounded two's complement):

* ``k`` (an ``int`` literal) is ``0*x + [k, k]``, and ``x`` is ``1*x + [0, 0]``.
* ``e1 + e2``, ``e1 - e2``, ``-e`` and ``+e`` combine slopes and intervals.
* ``e * k`` and ``k * e`` scale both.  A product of two bounded (slope-0)
  values is their interval product.  ``x * x`` and any other product of two
  non-constant affine values DECLINES.
* ``e ^ c``: ``e ^ c - e`` lies in ``[-c, c]`` for ``c >= 0``, because
  ``e ^ c == e + c - 2 * (e & c)`` and ``0 <= e & c <= c``.  For ``c < 0``,
  ``e ^ c == -(e ^ ~c) - 1`` with ``~c >= 0``, so the slope flips.
* ``e & c`` is in ``[0, c]`` for ``c >= 0`` (slope 0).  For ``c < 0``, it is
  ``e - (e & ~c)``, which is ``e + [-~c, 0]``.
* ``e | c`` is ``e + c - (e & c)``, i.e. ``e + [0, c]``, for ``c >= 0``.  For
  ``c < 0``, it is ``-(~e & ~c) - 1``, which lies in ``[-~c - 1, -1]``.
* ``e % c`` is in ``[0, c - 1]`` for ``c > 0``, and in ``[c + 1, 0]`` for
  ``c < 0``.
* ``e // c`` is ``(e - e % c) / c``.  ``e << k`` is ``e * 2**k``, and
  ``e >> k`` is ``e // 2**k``, for ``k >= 0``.
* ``(e0, e1, ...)[i]`` with a constant integer ``i`` is ``ei``.
* ``min`` / ``max`` of values that all have the SAME slope take the min / max
  of their interval ends.

Everything else declines (``None``): another free name, a float or any
non-``int`` literal, true division, ``**``, a call, a comparison, and a
bitwise operator between two non-constant operands.  A decline is not a
claim.  Every operand is an integer, which is why the bitwise identities
apply: true division, the only operator that leaves the integers, declines.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Union

__all__ = ["AffineEnvelope", "affine_envelope"]

#: Nesting ceiling -- a bound on work, never a claim.  An expression deeper
#: than this declines.
_MAX_DEPTH = 200


@dataclass(frozen=True)
class AffineEnvelope:
    """``a * var + lo <= expr <= a * var + hi`` for every integer ``var``.

    ``a``, ``lo`` and ``hi`` are exact rationals (``lo <= hi``).
    """

    var: str
    a: Fraction
    lo: Fraction
    hi: Fraction

    def as_record(self) -> dict:
        """Plain data (rationals as ``"p/q"`` strings) for a caller that must
        not import tharkun types."""
        return {"var": self.var, "a": str(self.a), "lo": str(self.lo), "hi": str(self.hi)}


class _Decline(Exception):
    pass


@dataclass(frozen=True)
class _Aff:
    a: Fraction
    lo: Fraction
    hi: Fraction

    @property
    def const(self) -> Optional[Fraction]:
        return self.lo if self.a == 0 and self.lo == self.hi else None


def _k(value: int) -> _Aff:
    return _Aff(Fraction(0), Fraction(value), Fraction(value))


def _add(p: _Aff, q: _Aff) -> _Aff:
    return _Aff(p.a + q.a, p.lo + q.lo, p.hi + q.hi)


def _neg(p: _Aff) -> _Aff:
    return _Aff(-p.a, -p.hi, -p.lo)


def _scale(p: _Aff, k: Fraction) -> _Aff:
    lo, hi = p.lo * k, p.hi * k
    return _Aff(p.a * k, min(lo, hi), max(lo, hi))


def _shift(p: _Aff, lo: Fraction, hi: Fraction) -> _Aff:
    """``p + r`` with ``r`` in ``[lo, hi]``."""
    return _Aff(p.a, p.lo + lo, p.hi + hi)


def _int_const(p: _Aff) -> int:
    """The value of a constant operand, which must be an integer."""
    c = p.const
    if c is None or c.denominator != 1:
        raise _Decline("operand is not an integer constant")
    return int(c)


def _xor(e: _Aff, c: int) -> _Aff:
    if c >= 0:
        return _shift(e, Fraction(-c), Fraction(c))
    nc = ~c  # >= 0: e ^ c == -(e ^ ~c) - 1
    return _shift(_neg(_shift(e, Fraction(-nc), Fraction(nc))), Fraction(-1), Fraction(-1))


def _and(e: _Aff, c: int) -> _Aff:
    if c >= 0:
        return _Aff(Fraction(0), Fraction(0), Fraction(c))
    return _shift(e, Fraction(-(~c)), Fraction(0))


def _or(e: _Aff, c: int) -> _Aff:
    if c >= 0:
        return _shift(e, Fraction(0), Fraction(c))
    return _Aff(Fraction(0), Fraction(-(~c) - 1), Fraction(-1))


def _mod(c: int) -> _Aff:
    if c == 0:
        raise _Decline("modulo by zero")
    if c > 0:
        return _Aff(Fraction(0), Fraction(0), Fraction(c - 1))
    return _Aff(Fraction(0), Fraction(c + 1), Fraction(0))


def _floordiv(e: _Aff, c: int) -> _Aff:
    # e // c == (e - e % c) / c, and e % c is bounded by _mod(c).
    if c == 0:
        raise _Decline("floor division by zero")
    return _scale(_add(e, _neg(_mod(c))), Fraction(1, c))


class _Walker:
    def __init__(self, var: str):
        self.var = var

    def eval(self, node: ast.expr, depth: int = 0) -> _Aff:
        if depth > _MAX_DEPTH:
            raise _Decline("expression too deep")
        d = depth + 1
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise _Decline(f"non-int literal {node.value!r}")
            return _k(node.value)
        if isinstance(node, ast.Name):
            if node.id != self.var:
                raise _Decline(f"free name {node.id!r}")
            return _Aff(Fraction(1), Fraction(0), Fraction(0))
        if isinstance(node, ast.UnaryOp):
            v = self.eval(node.operand, d)
            if isinstance(node.op, ast.USub):
                return _neg(v)
            if isinstance(node.op, ast.UAdd):
                return v
            if isinstance(node.op, ast.Invert):  # ~e == -e - 1
                return _shift(_neg(v), Fraction(-1), Fraction(-1))
            raise _Decline(f"unary {type(node.op).__name__}")
        if isinstance(node, ast.BinOp):
            return self._binop(node, d)
        if isinstance(node, ast.Subscript):
            return self._subscript(node, d)
        if isinstance(node, ast.Call):
            return self._call(node, d)
        raise _Decline(f"expression {type(node).__name__}")

    def _binop(self, node: ast.BinOp, d: int) -> _Aff:
        left = self.eval(node.left, d)
        right = self.eval(node.right, d)
        op = node.op
        if isinstance(op, ast.Add):
            return _add(left, right)
        if isinstance(op, ast.Sub):
            return _add(left, _neg(right))
        if isinstance(op, ast.Mult):
            if left.const is not None:
                return _scale(right, left.const)
            if right.const is not None:
                return _scale(left, right.const)
            if left.a == 0 and right.a == 0:
                ends = [x * y for x in (left.lo, left.hi) for y in (right.lo, right.hi)]
                return _Aff(Fraction(0), min(ends), max(ends))
            raise _Decline("product of two non-constant values")
        if isinstance(op, (ast.BitXor, ast.BitAnd, ast.BitOr)):
            if right.const is not None:
                e, c = left, _int_const(right)
            elif left.const is not None:
                e, c = right, _int_const(left)
            else:
                raise _Decline(f"{type(op).__name__} of two non-constant values")
            if isinstance(op, ast.BitXor):
                return _xor(e, c)
            if isinstance(op, ast.BitAnd):
                return _and(e, c)
            return _or(e, c)
        if isinstance(op, ast.Mod):
            return _mod(_int_const(right))
        if isinstance(op, ast.FloorDiv):
            return _floordiv(left, _int_const(right))
        if isinstance(op, (ast.LShift, ast.RShift)):
            k = _int_const(right)
            if k < 0:
                raise _Decline("negative shift count")
            if isinstance(op, ast.LShift):
                return _scale(left, Fraction(2 ** k))
            return _floordiv(left, 2 ** k)
        raise _Decline(f"operator {type(op).__name__}")

    def _subscript(self, node: ast.Subscript, d: int) -> _Aff:
        if not isinstance(node.value, (ast.Tuple, ast.List)):
            raise _Decline("subscript of a non-literal sequence")
        # The index must be a constant integer (``-1`` parses as a unary op).
        i = _int_const(self.eval(node.slice, d))
        elts = node.value.elts
        if not -len(elts) <= i < len(elts):
            raise _Decline("index out of range")
        return self.eval(elts[i], d)

    def _call(self, node: ast.Call, d: int) -> _Aff:
        if not (isinstance(node.func, ast.Name) and node.func.id in ("min", "max")):
            raise _Decline("call")
        if node.keywords or len(node.args) < 2:
            raise _Decline(f"{getattr(node.func, 'id', '?')} shape")
        vals = [self.eval(a, d) for a in node.args]
        slope = vals[0].a
        if any(v.a != slope for v in vals):
            raise _Decline(f"{node.func.id} over different slopes")
        pick = min if node.func.id == "min" else max
        return _Aff(slope, pick(v.lo for v in vals), pick(v.hi for v in vals))


def affine_envelope(expr: Union[str, ast.expr], var: str) -> Optional[AffineEnvelope]:
    """The proven affine bracket of *expr* in the integer variable *var*.

    ``expr`` is Python source for ONE expression (or its AST), whose only free
    name is *var*.  Returns ``None`` when the expression is outside the
    analysis (see the module docstring), which says nothing about it.
    """
    if isinstance(expr, str):
        try:
            node = ast.parse(expr, mode="eval").body
        except SyntaxError:
            return None
    elif isinstance(expr, ast.Expression):
        node = expr.body
    else:
        node = expr
    if not isinstance(var, str) or not var.isidentifier():
        return None
    try:
        v = _Walker(var).eval(node)
    except (_Decline, RecursionError):
        return None
    return AffineEnvelope(var=var, a=v.a, lo=v.lo, hi=v.hi)
