"""Exact affine quotients of pure integer polynomials; no evaluation or degree cap."""
from __future__ import annotations

import ast
from collections import Counter
import time

from .linear import Lin


def affine_quotient(node, names, deadline):
    """Return a proved affine quotient, or None; callers retain division's zero check."""
    class Decline(Exception):
        pass

    def budget():
        if time.process_time() >= deadline:
            raise Decline

    def add(left, right, sign=1):
        out = dict(left)
        for mono, coefficient in right.items():
            budget()
            value = out.get(mono, 0) + sign * coefficient
            if value:
                out[mono] = value
            else:
                out.pop(mono, None)
        return out

    def multiply(left, right):
        out = {}
        for lm, lc in left.items():
            for rm, rc in right.items():
                budget()
                mono = tuple(sorted(lm + rm))
                value = out.get(mono, 0) + lc * rc
                if value:
                    out[mono] = value
                else:
                    out.pop(mono, None)
        return out

    def parse(root):
        values, stack = {}, [(root, False)]
        while stack:
            budget()
            item, ready = stack.pop()
            if isinstance(item, ast.Constant) and type(item.value) in {int, bool}:
                values[item] = {(): int(item.value)} if item.value else {}
            elif isinstance(item, ast.Name) and item.id in names:
                lin = names[item.id]
                values[item] = {(name,): coefficient for name, coefficient in lin.terms}
                if lin.c:
                    values[item][()] = lin.c
            elif isinstance(item, ast.UnaryOp) and isinstance(item.op, (ast.UAdd, ast.USub)):
                if not ready:
                    stack.extend([(item, True), (item.operand, False)])
                else:
                    sign = -1 if isinstance(item.op, ast.USub) else 1
                    values[item] = {m: sign * c for m, c in values[item.operand].items()}
            elif isinstance(item, ast.BinOp) and isinstance(item.op, (ast.Add, ast.Sub, ast.Mult)):
                if not ready:
                    stack.extend([(item, True), (item.right, False), (item.left, False)])
                else:
                    left, right = values[item.left], values[item.right]
                    values[item] = (multiply(left, right) if isinstance(item.op, ast.Mult)
                                    else add(left, right, -1 if isinstance(item.op, ast.Sub) else 1))
            else:
                raise Decline
        return values[root]

    try:
        numerator, denominator = parse(node.left), parse(node.right)
        if not denominator:
            return None
        order = lambda mono: (len(mono), mono)
        leading = max(denominator, key=order)
        coefficient = denominator[leading]
        quotient = {}
        while numerator:
            budget()
            current = max(numerator, key=order)
            remaining = Counter(current)
            remaining.subtract(leading)
            if any(n < 0 for n in remaining.values()) or numerator[current] % coefficient:
                return None
            mono = tuple(sorted(remaining.elements()))
            if len(mono) > 1:
                return None
            term = {mono: numerator[current] // coefficient}
            quotient = add(quotient, term)
            numerator = add(numerator, multiply(denominator, term), -1)
        return Lin({m[0]: c for m, c in quotient.items() if m}, quotient.get((), 0))
    except Decline:
        return None
