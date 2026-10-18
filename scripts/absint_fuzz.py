"""Differential soundness fuzzer for tharkun.solve.absint (docs/ABSINT.md).

Generates random small programs (ints, floats, lists, None, instances, loops, branches,
calls, division, indexing, attribute access), runs each concretely under many random
draw vectors, and checks every absint TRUE against the runs: a run that raises a
property exception while absint said TRUE for that property is a soundness bug.
The program is printed and the fuzzer exits 1.

    python scripts/absint_fuzz.py [--n 2000] [--runs 200] [--seed 0] [--protocol-bias 0.07] [--slice-bias 0.15] [--slice-shape-bias 0.15]
                                  [--set-bias 0.18] [--set-shape-bias 0.2]

[internal ref] batch 5 adds the protocol shapes: a class ``Q`` whose ``__iter__`` is fine and whose ``__len__`` --
chosen per instance by a draw -- returns, raises TypeError (after writing the global ``g``), ValueError,
AttributeError, or fails natively (``len(5)``), consumed by every callable that does or does not ask for a
length hint (``list``, ``sorted``, ``list.extend``, ``tuple``, a ``for``, ``len``), with a division by ``g``
after it; and calls into ``math`` (``floor``, ``fabs``, ``isqrt``, ``gcd``, ``factorial``, ``copysign``, ...).
The positive control for the first is the base tree before PR "batch 5" (a wrong TRUE within seconds).

[internal ref] adds the set family: three sets (``s`` and ``u``, a frozenset ``fs``) built from the list, the tuple, a
display, a comprehension and a range, then every method (``add``, ``remove``, ``discard``, ``pop``, ``clear``,
``copy``, ``update``, ``union``, ``intersection``, ``difference``, ``symmetric_difference`` and the ``*_update``
forms, the predicates), the operators ``| & - ^`` and their in-place forms, the comparisons, ``in``, ``len``,
iteration, unpacking and ``sorted``/``list``/``tuple``/``min``/``max``/``sum`` over them, with draws as elements
(so two sets alias, empty, repeat an element or never meet).  ``--set-bias`` raises how often a statement is one of
them.  The positive controls are broken trees (``discard`` that keeps the length lower bound, ``pop`` that proves
``KeyError`` away, ``add`` that forgets the element): each is caught within one shard.

[internal ref]/[internal ref] add the slice stores and the array methods: a slice assignment (``xs[a:b] = ...``) from a range, a dict, a dict's
keys or values, an array, a tuple or a list, then every read of the list; the methods of an ``'i'``, ``'B'`` and ``'f'``
array called directly and through a bound method (``m = ar.append; m(x)``), then reads of the array.  ``--slice-bias`` raises
how often a statement is one of them.  The positive controls are the sv_2027 tree 4e129af (both holes are wrong TRUEs within a
shard) and a tree with either fix undone.

The concrete runs have a 2 s alarm each (a generated loop can grow a list exponentially, and the script used to hang on such a
program), so a run that does not finish is no evidence either way, and ``prove`` has a 20 s budget.
"""

from __future__ import annotations

import argparse
import random
import sys
import types

from tharkun.solve.absint import PROPERTY_CLASSES, prove

INTS = ["a", "b", "c"]
#: float literals: both zeros, a big finite value, an infinity (1e999), an int (compared exactly)
FCONSTS = ["0.0", "1.0", "-1.0", "0.5", "2.0", "-0.0", "3", "1e308", "1e999", "-1e999"]


class Gen:
    #: probability that a statement is one of the protocol shapes (``PROTOCOL``); raise it to hunt for them
    protocol_bias = 0.07
    #: probability that a statement is one of the slice-store / array shapes (``SLICES``)
    slice_bias = 0.15
    #: probability that a statement is a slice LIFECYCLE (``slice_shape``): build a list, assign a slice from a source with a zero in
    #: it, read the list by division -- the shape that exposes a source whose elements were dropped
    slice_shape_bias = 0.15
    #: probability that a statement is one of the set shapes (``SETS``)
    set_bias = 0.18
    #: probability that a statement is a set LIFECYCLE (``set_shape``: build, change, read), the shape that exposes a
    #: wrong length bound or element summary the way ``length_shape`` does for a list
    set_shape_bias = 0.2

    def __init__(self, rng: random.Random):
        self.r = rng
        self.depth = 0

    def iexpr(self, d=0) -> str:
        r = self.r
        k = r.random()
        if d > 2 or k < 0.3:
            return r.choice(INTS + [str(r.randint(-3, 6)), "len(xs)", "nd()"])
        op = r.choice(["+", "-", "*", "//", "%", "abs", "min", "len", "idx", "ord", "tup", "red", "math", "math", "sete"])
        if op == "sete":
            return r.choice(["len(s)", "len(u)", "len(fs)", "sum(s)", "sum(u)", "sum(fs)", "max(u)", "min(s, default=5)",
                             "max(fs, default=-1)", "s.pop()", "int(a in s)", "int(b in fs)", "int(s <= u)", "int(s == u)",
                             "int(u.issuperset(s))", "int(not s)", "len(s | u)", "len(s & fs)", "len(u - s)", "len(s ^ u)",
                             "len(sorted(u))", "len(list(fs))", f"len(s.union(xs, [{self.iexpr(d + 1)}]))"])
        if op == "math":
            return r.choice([f"math.isqrt(abs({self.iexpr(d + 1)}))", f"math.gcd({self.iexpr(d + 1)}, {self.iexpr(d + 1)})",
                             f"math.factorial(min(abs({self.iexpr(d + 1)}), 6))", "math.floor(fl)", "math.ceil(fl)",
                             f"math.trunc({self.iexpr(d + 1)})", f"math.lcm({self.iexpr(d + 1)}, 3)",
                             f"math.comb(min(abs({self.iexpr(d + 1)}), 8), 2)", "int(math.fabs(fl))"])
        if op == "ord":
            return f"ord({self.sexpr(d + 1)})"
        if op == "tup":
            # [internal ref] batch 4: tuple() and the tuples it builds
            return r.choice([f"tuple(xs)[{self.iexpr(d + 1)}]", "len(tuple(xs))", "len(t)",
                             f"t[{self.iexpr(d + 1)}]",
                             f"tuple(range(min({self.iexpr(d + 1)}, 4)))[{self.iexpr(d + 1)}]"])
        if op == "red":
            # [internal ref]: the iterable-reducing builtins over the list, the tuple and a range
            return r.choice(["min(xs)", "max(xs)", "sum(xs)", "min(xs, default=2)", "max(xs, default=-1)",
                             "sum(xs, 1)", "sum(t)", "max(t)", "min(t, default=0)", "int(any(xs))", "int(all(xs))",
                             "int(any(t))", "sum(range(min(abs(a), 5)))", "max(range(min(abs(b), 5)), default=7)",
                             "len(sorted(xs, reverse=True))", "sorted(xs, reverse=True)[0]", "sorted(t)[-1]",
                             f"min(xs, default={self.iexpr(d + 1)})", f"sum(xs, {self.iexpr(d + 1)})"])
        if op == "abs":
            return f"abs({self.iexpr(d + 1)})"
        if op == "min":
            return f"min({self.iexpr(d + 1)}, {self.iexpr(d + 1)})"
        if op == "len":
            return "len(xs)"
        if op == "idx":
            return f"xs[{self.iexpr(d + 1)}]"
        if op == "*":
            return f"({self.iexpr(d + 1)} * {r.randint(-2, 3)})"
        return f"({self.iexpr(d + 1)} {op} {self.iexpr(d + 1)})"

    def sexpr(self, d=0) -> str:
        """A str expression: literals of length 0/1/2, chr(...) of an arbitrary int (ValueError
        when out of range, not a property class), concatenation ([internal ref] batch 4: ord/chr)."""
        r = self.r
        k = r.random()
        if d > 2 or k < 0.4:
            return r.choice(['"a"', '"bc"', '""', '"Z"'])
        if k < 0.7:
            return f"chr({self.iexpr(d + 1)})"
        if k < 0.85:
            return f"chr(65 + abs({self.iexpr(d + 1)}) % 26)"
        return f"({self.sexpr(d + 1)} + {self.sexpr(d + 1)})"

    def cond(self) -> str:
        r = self.r
        k = r.random()
        if k < 0.15:
            return "nb()"
        if k < 0.25:
            return "p is None"
        if k < 0.4:
            return f"fl {r.choice(['<', '<=', '==', '!=', '>', '>='])} {r.choice(FCONSTS)}"
        return f"{self.iexpr(1)} {r.choice(['<', '<=', '==', '!=', '>', '>='])} {self.iexpr(1)}"

    def length_shape(self, ind: int) -> list[str]:
        """A store indexed by a draw, a length-changing list method, then a read indexed by a draw: the shape
        of a stale length bound (a stored ``len(xs)`` named the length by its old value, so the elements read
        after the length changed were off by one and a divisor of 0 was proven absent)."""
        r, sp = self.r, " " * ind
        store = r.choice(["xs[nd()] = len(xs)", "xs[a] = len(xs)", "xs[a] = len(xs) - 1", "xs[nd()] = len(xs) - 2",
                          f"xs[a] = {self.iexpr(1)}", "n = len(xs)\n" + sp + "xs[a] = n"])
        change = r.choice(["xs.append(len(xs))", "xs.append(5)", "xs.append(xs[0])", "xs.insert(0, len(xs))",
                           "xs.insert(1, 7)", "xs.extend([len(xs)])", "xs.extend([4, 4])", "xs.pop()", "xs.pop(0)",
                           "xs += [len(xs)]", "xs[0:1] = [8, 8]", "xs *= 1"])
        read = r.choice(["c = 1 // xs[nd()]", "c = 1 // (xs[nd()] - 1)", "c = 1 // xs[a]", "assert xs[nd()] != 0",
                         "assert xs[a] >= 1"])
        return [sp + store, sp + change, sp + read]

    def slice_shape(self, ind: int) -> list[str]:
        """A list, a slice assignment from a range, a dict (keys, values), an array, a tuple or a list whose elements include a zero, then a
        read of the list that divides by an element: an element the model dropped makes the division look safe."""
        r, sp = self.r, " " * ind
        build = r.choice(["xs = [1]", "xs = [1, 2]", "xs = [1, 1, 1]", "xs = []", "xs = [a, 1]"])
        sl = r.choice(["xs[0:1]", "xs[:]", "xs[1:]", "xs[:1]", "xs[0:0]", "xs[len(xs):]", "xs[::1]"])
        src = r.choice(["range(1)", "range(2)", "range(a % 2)", "{0: 5}", "{0: 1, 1: 2}", "{5: 0}.keys()", "{1: 0}.values()", "{0: 1}.keys()",
                        "array('i', [0])", "array('i', [0, 1])", "array('f', [0.0])", "(0,)", "[0]", "[0, 1]", "'ab'", "()", "{}", "range(0)"])
        read = r.choice(["c = 10 // xs[0]", "c = 10 // xs[-1]", "for e in xs:\n" + sp + "    c = 10 // e", "c = 10 // min(xs)", "c = 10 // sum(xs)",
                         "c = 10 // max(xs)", "assert 0 not in xs", "assert all(xs)", "c = 10 // xs[len(xs) - 1]", "c = 10 // len(xs)",
                         "assert len(xs) >= 1", "x, y = xs\n" + sp + "c = 10 // y"])
        out = []
        for part in (build, f"{sl} = {src}", read):
            out.extend(sp + line.lstrip(" ") if i else sp + line for i, line in enumerate(part.split("\n")))
        return out

    def set_shape(self, ind: int) -> list[str]:
        """A set is built from constants and draws, changed (an element removed, added, the set emptied, combined
        with another), then read in a way that depends on what it holds: a length bound or an element summary that
        forgot the change makes a division by an element, a ``pop``, a ``remove`` or an assertion wrongly safe."""
        r, sp = self.r, " " * ind
        build = r.choice(["s = {1}", "s = {2}", "s = {1, 2}", "s = set()", "s = {a}", "s = {nd(), 1}", "s = set(xs)", "s = {0, 1}",
                          "s = {1.0, 2}", "s = frozenset([1, 2])\n" + sp + "s = set(s)"])
        change = r.choice(["s.discard(1)", "s.discard(a)", "s.remove(1)", "s.remove(a)", "s.clear()", "s.pop()", "s.add(0)", "s.add(a)",
                           "s.add(-1)", "s -= u", "s &= u", "s.update([0])", "s |= u", "t = s\n" + sp + "t.discard(1)",
                           "u = s\n" + sp + "u.clear()", "s = s - {1}", "s.symmetric_difference_update(u)", "s.difference_update(u)",
                           "s.intersection_update(xs)", "u = s.copy()\n" + sp + "u.add(0)", "s.update(s)", "s ^= u", "s = s.union([0])"])
        read = r.choice(["c = 1 // len(s)", "c = s.pop()", "s.pop()", "s.remove(1)", "s.remove(a)", "for v in s:\n" + sp + "    c = 10 // v",
                         "for v in s:\n" + sp + "    c = 10 // (v - 1)", "c = 10 // min(s)", "c = 10 // sum(s)", "assert len(s) >= 1",
                         "assert len(s) <= 1", "assert 1 in s", "assert 0 not in s", "c = 1 // (len(s) - 1)", "x, y = s", "assert s",
                         "assert not s", "assert s <= u", "assert s == u", "c = xs[len(s)]", "c = 10 // len(sorted(s))",
                         "assert len(s | u) >= len(s)"])
        out = []
        for part in (build, change, read):
            out.extend(sp + line.lstrip(" ") if i else sp + line for i, line in enumerate(part.split("\n")))
        return out

    def block(self, ind: int, n: int) -> list[str]:
        out = []
        for _ in range(n):
            out.extend(self.stmt(ind))
        return out or [" " * ind + "pass"]

    PROTOCOL = [
        ["q = Q(nd() % 6)"], ["xs = list(q)"], ["xs.extend(q)"], ["xs = sorted(q)"], ["t = tuple(q)"],
        ["for v in q:", "    c = v"], ["c = len(q)"], ["c = 10 // g"], ["g = 2"], ["xs = list(Q(nd() % 6))"],
        ["c = 10 // (g - 1)"], ["if q:", "    c = 1"],
    ]
    #: ``{I}`` is an int expression; ``xs`` the list, ``ar`` an 'i' array, ``fa`` an 'f' array, ``d`` a dict
    SLICES = [
        "xs[0:1] = range(min(abs({I}), 3))", "xs[:] = range(abs(a) % 3)", "xs[1:] = range(abs(b) % 3)", "xs[0:0] = range(2)",
        "xs[:] = d", "xs[0:1] = d", "xs[0:0] = d.keys()", "xs[:] = d.values()", "xs[1:] = d.keys()", "xs[:2] = {a: b}",
        "xs[:] = ar", "xs[0:1] = ar", "xs[0:1] = array('i', [{I}])", "xs[len(xs):] = array('i', [a, 0])", "xs[:] = fa",
        "xs[0:1] = (0,)", "xs[:2] = ({I}, 0)", "xs[:] = [{I}]", "xs[::1] = range(len(xs))", "xs[0:1] = 'ab'", "xs[0:1] = {a}",
        "xs[0:1] = d.items()", "xs[0:1] = a",
        "ar.append({I})", "ar.append(0)", "ar.append(2.5)", "ar.extend(xs)", "ar.extend([a, 1])", "ar.extend(ar)",
        "m = ar.append\nm({I})", "m = ar.append\nm(0)", "m = ar.append\nm('x')", "m = ar.extend\nm(xs)", "m = ar.extend\nm([a])",
        "fa.append(0.1)", "fa.append({I})", "m = fa.append\nm(0.1)", "m = fa.append\nm(0.5)", "m = fa.append\nm({I})",
        "m = fa.extend\nm([0.1, 0.5])", "fa.extend([0.1])", "m = ba.append\nm({I})", "m = ba.append\nm(300)", "ba.append(a)",
        "ar = array('i', [a, 1])", "ar = array('i')", "fa = array('f', [0.5])", "fa = array('f')", "ba = array('B', [1])",
        "c = 10 // ar[0]", "c = 10 // ar[{I}]", "c = 10 // (ar[0] - 1)", "c = 10 // len(ar)", "assert len(ar) >= 1",
        "assert fa[0] == 0.1", "assert fa[0] == 0.5", "assert fa[0] != 0.1", "assert fa[{I}] <= 1.0", "c = 10 // int(fa[0])",
        "c = 10 // ba[0]", "assert ba[0] >= 1", "xs.append(ar[0])", "xs = list(ar)", "xs = list(fa)",
    ]
    #: ``{I}`` is an int expression; every line is one statement (or a small block) over s, u (sets) and fs (a frozenset)
    SETS = [
        "s = set()", "s = set(xs)", "s = set(t)", "s = {a, b, 3}", "s = {nd(), 1}", "s = {x for x in xs}",
        "s = set(range(min(abs(a), 5)))", "u = set(s)", "u = s.copy()", "u = {1, 2}", "fs = frozenset(xs)",
        "fs = frozenset(s)", "fs = frozenset()", "u = s", "s = u", "s = {1.0, True}", "s = {fl}", "s = {None, 2}",
        "s.add({I})", "s.add(fl)", "s.add(t)", "s.add((a, 1))", "s.add(xs)", "s.add(p)", "s.add(None)", "s.add(True)",
        "s.remove({I})", "s.remove(fl)", "s.remove(1)", "s.remove(xs)", "s.remove({1})", "s.discard({I})", "s.discard({1})",
        "s.discard(None)", "s.pop()", "c = s.pop()", "s.clear()", "s.update(xs)", "s.update(u, [{I}])", "s.update(s)",
        "s.update()", "s.update(a)", "s.update(range(3))", "s.update('ab')", "s |= u", "s &= u", "s -= u", "s ^= u",
        "s |= fs", "s -= s", "s ^= s", "u = s | fs", "u = fs | s", "u = s & u", "u = s - fs", "u = s ^ u", "fs = fs | u",
        "fs |= u", "s.difference_update(u)", "s.intersection_update(xs)", "s.symmetric_difference_update(u)",
        "s.difference_update(s)", "u = s.union(xs, t)", "u = s.intersection(xs)", "u = s.difference(xs)",
        "u = s.symmetric_difference(xs)", "u = fs.union(u)", "u = fs.copy()", "u = s.union()", "fs.add(1)", "s.foo()",
        "s = s + u", "s |= xs", "c = int(s.issubset(xs))", "c = int(u.issuperset(s))", "c = int(s.isdisjoint(u))",
        "c = int(s <= u)", "c = int(s < fs)", "c = int(fs >= u)", "c = int(s == u)", "c = int(s != fs)",
        "c = int(s > 5)", "c = int({I} in s)", "c = int({I} not in u)", "c = int(fl in s)", "c = int(xs in s)",
        "c = 1 // len(s)", "c = 1 // (len(s) - len(u))", "c = 1 // (len(u) - 2)", "c = 1 // (len(s | u) - 2)",
        "c = 10 // ({I})", "c = xs[len(s)]", "c = xs[len(s) - 1]", "c = 10 // min(u)", "c = 10 // max(fs)",
        "c = 10 // sum(s)", "assert len(s) <= 2", "assert len(s) >= 1", "assert len(s) == len(u)", "assert {I} in s",
        "assert s <= u", "assert not u", "assert s == fs", "assert u != s", "assert s",
        "if s:\n    c = 1", "if not u:\n    c = 1 // len(fs)", "for v in s:\n    c = 10 // (v - a)",
        "for v in u:\n    s.add(v + 1)", "for v in fs:\n    c = v", "xs = sorted(s)", "xs = list(u)", "t = tuple(fs)",
        "xs.extend(s)", "xs += list(u)", "x, y = u", "x, = s", "c = min(s)", "c = max(u)", "c = sum(fs)", "t = tuple(s)",
        "xs = [s, u]\nc = len(xs[a % 2])", "ss = [set(), set()]\ns = ss[a % 2]\nu = ss[1]",
    ]
    MATH_FLOAT = ["fl = math.fabs(fl)", "fl = math.copysign(fl, a)", "fl = math.hypot(fl, 1.0)",
                  "fl = math.cos(fl)", "fl = math.tanh(fl)", "fl = math.atan2(fl, fl)", "fl = math.fmod(fl, 3.0)",
                  "fl = math.pow(2.0, 3)", "fl = float(math.isqrt(abs(a)))"]

    def stmt(self, ind: int) -> list[str]:
        r = self.r
        sp = " " * ind
        if r.random() < self.protocol_bias:
            return [sp + line for line in r.choice(self.PROTOCOL)]
        if r.random() < self.set_shape_bias:
            return self.set_shape(ind)
        if r.random() < self.set_bias:
            line = r.choice(self.SETS)
            while "{I}" in line:
                line = line.replace("{I}", self.iexpr(1), 1)
            return [sp + part for part in line.split("\n")]
        if r.random() < self.slice_shape_bias:
            return self.slice_shape(ind)
        if r.random() < self.slice_bias:
            line = r.choice(self.SLICES)
            while "{I}" in line:
                line = line.replace("{I}", self.iexpr(1), 1)
            return [sp + part for part in line.split("\n")]
        if r.random() < 0.04:
            return [sp + r.choice(self.MATH_FLOAT)]
        k = r.random()
        if r.random() < 0.06:
            return self.length_shape(ind)
        if self.depth < 2 and k < 0.12:
            self.depth += 1
            s = [sp + f"if {self.cond()}:"] + self.block(ind + 4, r.randint(1, 3))
            if r.random() < 0.5:
                s += [sp + "else:"] + self.block(ind + 4, r.randint(1, 2))
            self.depth -= 1
            return s
        if self.depth < 2 and k < 0.22:
            self.depth += 1
            v = r.choice(["i", "j"])
            s = [sp + f"for {v} in range(min({self.iexpr(1)}, 6)):"] + self.block(ind + 4, r.randint(1, 3))
            self.depth -= 1
            return s
        if self.depth < 2 and k < 0.27:
            self.depth += 1
            s = [sp + "g = 0", sp + f"while g < 5 and {self.cond()}:", sp + "    g += 1"] + self.block(ind + 4, r.randint(1, 2))
            self.depth -= 1
            return s
        if k < 0.45:
            return [sp + f"{r.choice(INTS)} = {self.iexpr()}"]
        if k < 0.52:
            return [sp + f"xs = {r.choice(['[0] * ', 'list(range(', '[1, 2, 3] * '])}"
                    + (f"min({self.iexpr(1)}, 5)" + ("))" if False else ""))]
        if k < 0.58:
            return [sp + f"xs.append({self.iexpr(1)})"]
        if k < 0.62:
            return [sp + "xs.pop()"]
        if k < 0.70:
            return [sp + f"xs[{self.iexpr(1)}] = {self.iexpr(1)}"]
        if k < 0.74:
            return [sp + f"a = f({self.iexpr(1)})"]
        if k < 0.78:
            return [sp + f"fl = {r.choice(['1.5', 'fl * 2.0', 'fl - 1.0', 'float(a)', 'fl / 2.0'])}"]
        if k < 0.82:
            return [sp + f"fl = {self.iexpr(1)} / (fl + {r.choice(['0.0', '1.0', '-0.5'])})"]
        if k < 0.87:
            return [sp + f"p = {r.choice(['None', 'P(a)', 'P(nd())'])}"]
        if k < 0.93:
            return [sp + f"b = p.{r.choice(['v', 'get()', 'w'])}" if r.random() < 0.9 else sp + "b = p.v"]
        if k < 0.96:
            return [sp + f"t = tuple({r.choice(['xs', 'range(min(' + self.iexpr(1) + ', 4))'])})"]
        if k < 0.975:
            # [internal ref]: a float draw, float literals and comparisons that decide a branch
            return [sp + f"fl = _sv_verifier.nondet_float_range({r.choice(FCONSTS)}, {r.choice(FCONSTS + ['fl'])})"]
        if k < 0.99:
            return [sp + f"assert fl {r.choice(['<', '<=', '>', '>=', '==', '!='])} {r.choice(FCONSTS)}"]
        return [sp + f"c = {self.iexpr()}"]

    def program(self) -> str:
        body = self.block(0, self.r.randint(3, 9))
        # list(range(... needs its closing parens
        fixed = []
        for line in body:
            if "list(range(" in line and line.count("(") > line.count(")"):
                line = line + ")" * (line.count("(") - line.count(")"))
            fixed.append(line)
        head = [
            "import _sv_verifier",
            "import math",
            "from array import array",
            "def nd():",
            "    return _sv_verifier.nondet_int()",
            "def nb():",
            "    return _sv_verifier.nondet_bool()",
            "class P:",
            "    __slots__ = ('v', 'w')",
            "    def __init__(self, v):",
            "        self.v = v",
            "        if v > 2:",
            "            self.w = v",
            "    def get(self):",
            "        return self.v + 1",
            "def f(w):",
            "    return w - 1 if w > 0 else w + 1",
            "g = 2",
            "class Q:",
            "    def __init__(self, m):",
            "        self.m = m",
            "        self.items = [1, 2]",
            "    def __iter__(self):",
            "        return iter(self.items)",
            "    def __len__(self):",
            "        global g",
            "        if self.m == 1:",
            "            g = g - 1",
            "            raise TypeError('no length')",
            "        if self.m == 2:",
            "            raise ValueError('boom')",
            "        if self.m == 3:",
            "            raise AttributeError('boom')",
            "        if self.m == 4:",
            "            return len(5)",
            "        return 2",
            "q = Q(0)",
            "a = nd()",
            "b = nd()",
            "c = 1",
            "fl = 1.0",
            "p = P(b)",
            "xs = [0, 1]",
            "t = (0, 1)",
            "d = {0: 1, 2: 3}",
            "ar = array('i', [1, 2])",
            "fa = array('f', [1.0])",
            "ba = array('B', [1])",
            "s = set(xs)",
            "u = {1, 2}",
            "fs = frozenset([1, 2])",
        ]
        return "\n".join(head + fixed) + "\n"


class _Invalid(Exception):
    """A rejected draw: the run is an invalid witness, never a violation (the shim's rule)."""


_FLOAT_POINTS = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 3.0, 1e308, -1e308, float("inf"), float("-inf"), float("nan"),
                 1e-308, 5e-324]


def _nondet_float_range(rng, a, b):
    """``eregion-sv/.../_sv_verifier.py::nondet_float_range``: any float is drawn, and it is
    rejected when ``isnan(v) or not min(a, b) <= v <= max(a, b)``."""
    lo, hi = min(a, b), max(a, b)
    v = rng.choice(_FLOAT_POINTS + [a, b, rng.uniform(-4.0, 4.0)])
    if v != v or not lo <= v <= hi:
        raise _Invalid
    return v


class _Timeout(BaseException):
    """A concrete run that did not finish in time: no exception was raised, so it is no evidence either way."""


def _alarm(_sig, _frame):
    raise _Timeout


def run_once(src: str, rng: random.Random):
    import signal

    signal.signal(signal.SIGALRM, _alarm)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        return _run_once(src, rng)
    except _Timeout:
        return None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _run_once(src: str, rng: random.Random):
    mod = types.ModuleType("_sv_verifier")
    mod.nondet_int = lambda: rng.choice([0, 0, 1, -1, 2, 3, rng.randint(-8, 12)])
    mod.nondet_bool = lambda: rng.random() < 0.5
    mod.nondet_float_range = lambda a, b: _nondet_float_range(rng, a, b)
    saved = sys.modules.get("_sv_verifier")
    sys.modules["_sv_verifier"] = mod
    try:
        exec(compile(src, "<fuzz>", "exec"), {"__name__": "__main__"})
        return None
    except _Invalid:
        return None
    except MemoryError:
        return None            # the address-space cap of ``main``: the run did not finish, so it is no evidence either way
    except Exception as e:  # noqa: BLE001 - we classify every exception
        return type(e)
    finally:
        if saved is not None:
            sys.modules["_sv_verifier"] = saved
        else:
            sys.modules.pop("_sv_verifier", None)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--protocol-bias", type=float, default=Gen.protocol_bias)
    ap.add_argument("--set-bias", type=float, default=Gen.set_bias)
    ap.add_argument("--set-shape-bias", type=float, default=Gen.set_shape_bias)
    ap.add_argument("--slice-bias", type=float, default=Gen.slice_bias)
    ap.add_argument("--slice-shape-bias", type=float, default=Gen.slice_shape_bias)
    a = ap.parse_args(argv)
    # the owner's rule: no process above 15 GB.  A shard is one process: cap its address space (a generated program that
    # builds a huge value then raises MemoryError, which is no evidence, instead of taking the host down)
    import resource

    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    Gen.protocol_bias = a.protocol_bias
    Gen.slice_bias = a.slice_bias
    Gen.slice_shape_bias = a.slice_shape_bias
    Gen.set_bias = a.set_bias
    Gen.set_shape_bias = a.set_shape_bias
    rng = random.Random(a.seed)
    stats = {"programs": 0, "true": 0, "unknown": 0}
    for k in range(a.n):
        src = Gen(random.Random(rng.random())).program()
        raised = set()
        rr = random.Random(k)
        for _ in range(a.runs):
            e = run_once(src, rr)
            if e is not None:
                raised |= {c.__name__ for c in e.__mro__}
        stats["programs"] += 1
        for prop, classes in PROPERTY_CLASSES.items():
            res = prove(src, classes, time_budget=20.0)
            stats["true" if res.verdict == "TRUE" else "unknown"] += 1
            if res.verdict == "TRUE" and raised & set(classes):
                print(f"SOUNDNESS BUG: {prop} TRUE but a run raised {sorted(raised & set(classes))}\n{src}")
                print(res.explain())
                return 1
    print(f"ok: {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
