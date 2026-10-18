"""Every kind of value into every consumer: an unhandled kind must be a named refusal (UNKNOWN), never an infeasible path ([internal ref]).

The class of bug ([internal ref], [internal ref], and the array unpack and ``random.choice`` of [internal ref]): an operation that does not know a value's
kind treats it as "bad" (a recorded ``TypeError`` that is not proven) and, when no alternative produced a value, ``set_bottom``s
the path -- so every statement after it is unchecked, and the proof of a program that raises there is TRUE.  The rule: an unmodelled
kind is ``Unsupported`` (a named refusal); the path is bottom only where the concrete semantics proves it impossible, and the
exception that makes it so is itself a recorded check (``KeyError`` of ``remove``, ``IndexError`` of ``choice`` on an empty list).

This is the matrix that finds it: for every value of every kind absint has (and the kinds it must refuse: set, frozenset, bytes,
bytearray, a generator, a complex) and every consumer (unpacking, iteration, ``len``, truth, subscripts and stores, every operator,
the builtins, ``random``, methods and attributes, calls, ``with``, formatting), the program

    <value>; <consumer of v>; reached = True; 1 // 0

is run natively.  If the consumer succeeds natively the sentinel ``1 // 0`` is reached, so absint must NOT say TRUE for
``ZeroDivisionError``; if it raises a property class natively, absint must not say TRUE for that class.  A TRUE it gives for a path
the consumer made infeasible is exactly the bug.
"""

from __future__ import annotations

import fcntl
import io
import os
import pickle
import re
import resource
import select
import signal
import sys
import types
import warnings

import pytest

from tharkun.solve.absint import PROPERTY_CLASSES, prove

PRELUDE = """\
import math
import re
import io
import random
import _sv_verifier
from array import array

class A:
    def m(self):
        return 1

class B:
    def __init__(self):
        self.x = 1

class E(Exception):
    pass

def f(x=1):
    return x

def mark():
    global reached
    reached = True
    return 1

"""

#: (name, setup) -- every kind absint models, and the kinds it must refuse
VALUES = [
    ("int", "v = 5"), ("zero", "v = 0"), ("bool", "v = True"), ("float", "v = 2.5"), ("str", "v = 'ab'"), ("empty_str", "v = ''"),
    ("none", "v = None"), ("list", "v = [1, 2]"), ("empty_list", "v = []"), ("list_of_lists", "v = [[1], [2]]"), ("tuple", "v = (1, 2)"),
    ("empty_tuple", "v = ()"), ("tuple_of_str", "v = ('a', 'b')"), ("array_i", "v = array('i', [1, 2])"), ("array_empty", "v = array('i')"),
    ("array_f", "v = array('f', [0.5, 1.5])"), ("range", "v = range(3)"), ("empty_range", "v = range(0)"), ("range_step", "v = range(0, 6, 2)"),
    ("dict", "v = {1: 2, 3: 4}"), ("empty_dict", "v = {}"), ("dict_str", "v = {'a': 1}"), ("keys", "v = {1: 2}.keys()"), ("values", "v = {1: 2}.values()"),
    ("items", "v = {1: 2}.items()"), ("iter_list", "v = iter([1, 2])"), ("iter_range", "v = iter(range(2))"), ("iter_empty", "v = iter(())"),
    ("instance", "v = A()"), ("instance_attr", "v = B()"), ("class", "v = A"), ("function", "v = f"), ("bound_method", "v = A().m"),
    ("builtin_len", "v = len"), ("builtin_type", "v = int"), ("exc_class", "v = ValueError"), ("exc_instance", "v = ValueError('x')"),
    ("user_exc", "v = E('x')"), ("module", "v = math"), ("pattern", "v = re.compile('a')"), ("stringio", "v = io.StringIO()"),
    ("set", "v = {1, 2}"), ("empty_set", "v = set()"), ("frozenset", "v = frozenset([1, 2])"), ("bytes", "v = b'ab'"), ("bytearray", "v = bytearray(b'ab')"),
    ("generator", "v = (x for x in [1, 2])"), ("complex", "v = 1j"), ("lambda", "v = lambda x: x"), ("nd_int", "v = _sv_verifier.nondet_int()"),
    ("seg_list", "v = [None] * 3"), ("seg_list_filled", "v = [None] * 3\nfor i in range(3):\n    v[i] = i"), ("list_loop", "for i in range(3):\n    v = [i]"),
    ("list_grown", "v = []\nfor i in range(3):\n    v.append(i)"), ("big_int", "v = 10 ** 30"), ("neg_int", "v = -3"), ("nan", "v = float('nan')"),
    ("inf", "v = float('inf')"), ("long_str", "v = 'x' * 70"), ("nested", "v = [[1, 2], [3, 4]]"), ("dict_of_lists", "v = {1: [1], 2: [2]}"),
    ("dict_none", "v = {1: None}"), ("array_B", "v = array('B', [1, 255])"), ("array_d", "v = array('d', [0.5])"), ("range_nd", "v = range(_sv_verifier.nondet_int() % 4)"),
    ("list_nd_len", "v = [0] * (_sv_verifier.nondet_int() % 3)"), ("tuple_nd", "v = (_sv_verifier.nondet_int(), 2)"),
    ("slots_inst", "class S:\n    __slots__ = ('a',)\n    def __init__(self):\n        self.a = 1\nv = S()"),
    ("seq_inst", "class Seq:\n    def __len__(self):\n        return 2\n    def __getitem__(self, i):\n        return i\nv = Seq()"),
    ("iter_inst", "class It:\n    def __iter__(self):\n        return iter([1, 2])\nv = It()"),
    ("len_inst", "class L:\n    def __len__(self):\n        return 0\nv = L()"),
    ("call_inst", "class Cl:\n    def __call__(self, x=1):\n        return x\nv = Cl()"),
    ("eq_inst", "class Q:\n    def __eq__(self, o):\n        return True\nv = Q()"),
    ("add_inst", "class N:\n    def __add__(self, o):\n        return 1\n    def __radd__(self, o):\n        return 2\nv = N()"),
    ("lt_inst", "class Lt:\n    def __lt__(self, o):\n        return True\nv = Lt()"),
    ("hash_inst", "class H:\n    def __hash__(self):\n        return 1\nv = H()"),
    ("ctx_inst", "class Ctx:\n    def __enter__(self):\n        return self\n    def __exit__(self, *a):\n        return False\nv = Ctx()"),
    ("bool_inst", "class Bo:\n    def __bool__(self):\n        return False\nv = Bo()"),
    ("index_inst", "class Ix:\n    def __index__(self):\n        return 1\nv = Ix()"),
    ("str_inst", "class St:\n    def __str__(self):\n        return 'x'\n    def __repr__(self):\n        return 'y'\nv = St()"),
    ("sub_inst", "class P:\n    pass\nclass Q2(P):\n    pass\nv = Q2()"),
    ("exc_sub", "class E2(ValueError):\n    pass\nv = E2('x')"), ("alias", "v = w0 = [1, 2]"),
    ("list0", "v = [0]"), ("tuple0", "v = (0,)"), ("array0", "v = array('i', [0])"), ("range1", "v = range(1)"), ("dict0", "v = {0: 1}"),
    ("keys0", "v = {0: 1}.keys()"), ("values0", "v = {1: 0}.values()"), ("iter0", "v = iter([0])"), ("set0", "v = {0}"), ("frozenset0", "v = frozenset([0])"),
    ("list_zero_loop", "v = []\nfor i in range(1):\n    v.append(i)"), ("seg_zero", "v = [None] * 1\nfor i in range(1):\n    v[i] = i"),
    ("stringio_content", "v = io.StringIO('a\\nb\\n')"), ("stringio_written", "v = io.StringIO()\nv.write('a\\nb\\n')"),
    ("outfile", "v = open('/dev/null', 'w')"), ("outfile_bin", "v = open('/dev/null', 'wb')"), ("tuple_nested", "v = ((1, 2), (3, 4))"),
    ("dict_keys_str", "v = {'a': 1, 'b': 2}.keys()"), ("iter_dict", "v = iter({1: 2})"), ("str_iter", "v = iter('ab')"),
]

#: single-operand consumers
CONSUMERS = [
    "x, y = v", "x, = v", "x, y, z = v", "(x, y), z = v, 1", "for x in v:\n    pass", "for x in v:\n    c = 1", "n = len(v)", "if v:\n    c = 1",
    "b = not v", "while v:\n    break", "x = v and 1", "x = v or 1", "x = 1 if v else 2", "b = 1 in v", "b = 'a' in v", "b = None in v", "b = [1] in v",
    "x = v[0]", "x = v[-1]", "x = v[1]", "x = v[5]", "x = v[0:1]", "x = v[::2]", "x = v[:]", "x = v['a']", "x = v[1.5]", "x = v[None]",
    "v[0] = 1", "v[1] = 1", "v[0:1] = [5]", "v['a'] = 1", "v[0] += 1", "x = v + v", "x = v + [1]", "x = [1] + v", "x = v + (1,)", "x = v + 1", "x = 1 + v",
    "x = v + 'a'", "x = v * 2", "x = 2 * v", "x = v * 0", "x = v * -1", "x = v * 2.5", "x = v - v", "x = v - 1", "x = 1 - v", "x = v % 2", "x = 2 % v",
    "x = v / 2", "x = 2 / v", "x = v // 2", "x = 2 // v", "x = v ** 2", "x = 2 ** v", "x = v | v", "x = v | 1", "x = v & v", "x = v ^ v", "x = v << 1",
    "x = v >> 1", "x = -v", "x = +v", "x = ~v", "x = abs(v)", "x = v < v", "x = v < 1", "x = 1 < v", "x = v <= 'a'", "x = v == 1", "x = v == v", "x = v != None",
    "x = v is None", "x = v is v", "x = v is not 1", "x = v == [1, 2]", "x = v == (1, 2)", "x = min(v)", "x = max(v)", "x = min(v, default=0)", "x = min(v, 1)",
    "x = sum(v)", "x = sum(v, 1)", "x = any(v)", "x = all(v)", "x = sorted(v)", "x = sorted(v, reverse=True)", "x = list(v)", "x = tuple(v)", "x = iter(v)",
    "x = range(v)", "x = range(1, v)", "x = isinstance(v, int)", "x = isinstance(v, (list, tuple))", "x = isinstance(v, A)", "print(v)", "x = int(v)",
    "x = float(v)", "x = str(v)", "x = repr(v)", "x = bool(v)", "x = divmod(v, 2)", "x = divmod(5, v)", "x = pow(v, 2)", "x = pow(2, v)", "x = round(v)",
    "x = math.sqrt(v)", "x = math.floor(v)", "x = math.gcd(v, 2)", "x = len(str(v))", "x = chr(v)", "x = ord(v)", "x = tuple(range(v))",
    "x = v.append(1)", "x = v.pop()", "x = v.foo", "x = v.__class__", "x = v.keys()", "x = v.copy()", "x = v.items()", "x = v.get(1)", "x = v.m()",
    "x = v.x", "v.foo = 1", "v.x = 1", "x = v()", "x = v(1)", "x = v(1, 2)", "xs = [0]\nxs.extend(v)", "xs = [0]\nxs += v", "xs = [0]\nxs[0:1] = v",
    "xs = [0]\nxs.append(v)", "xs = [0]\nxs.insert(0, v)", "xs = [v]\nx = xs[0]", "d = {}\nd[v] = 1", "x = {1: 2}[v]", "x = [1, 2][v]", "x = 'ab'[v]",
    "x = (1, 2)[v]", "x = {1: 2}.get(v)", "x = random.choice(v)", "x = random.randint(v, 5)", "x = random.randint(0, v)", "x = random.uniform(v, 2)",
    "x = random.random() * v", "x = 'a'.join(v)", "x = enumerate(v)", "x = zip(v, v)", "x = reversed(v)", "x = dict(v)", "x = set(v)", "x = frozenset(v)",
    "x = list(map(abs, v))", "x = [e for e in v]", "x = {e: 1 for e in v}", "x = [e * 2 for e in v if e]", "x = f(v)", "x = f(x=v)", "x = max(v, key=abs)",
    "with v:\n    pass", "x = f'{v}'", "x = '%s' % (v,)", "x = '%d' % v", "x = '%s' % v", "x = 'a' % v", "x = '{}'.format(v)", "x = 'ab'.index(v)",
    "x = 'ab'.startswith(v)", "x = [1, 2].index(v)", "x = [1, 2].count(v)", "x = [1, 2].remove(v)", "x = array('i', v)", "x = array('i', [1]) + v",
    "x = array('i', [1]) * v", "x = bytes(v)", "x = hash(v)", "x = id(v)", "x = type(v)", "x = callable(v)", "x = getattr(v, 'x')", "x = hasattr(v, 'x')",
    "x = next(v)", "x = iter(v).__next__()", "x = open(v)", "del v", "assert v", "assert not v", "assert v is None", "global_v = v\nx = global_v",
    "def g(a):\n    return a[0]\nx = g(v)", "def g(a):\n    return a + 1\nx = g(v)", "class C:\n    k = v\nx = C.k", "x = [v, v]\ny, z = x", "x = (v, v)\ny, z = x",
    "x = {1: v}\ny = x[1][0]", "x = {1: v}\ny, z = x[1]", "x = [v] * 2\ny = x[0] + 1", "x = A()\nx.k = v\ny = x.k[0]",
]

#: second operands for the two-operand consumers
OTHERS = ["1", "0", "'a'", "None", "[1]", "(1,)", "{1: 2}", "array('i', [1])", "range(2)", "2.5", "{1}", "A()"]
BINARY = ["x = v + w", "x = w + v", "x = v - w", "x = w - v", "x = v * w", "x = w * v", "x = v / w", "x = v // w", "x = v % w", "x = v ** w", "x = v | w",
          "x = v & w", "x = v ^ w", "x = v < w", "x = w <= v", "x = v == w", "x = v != w", "b = w in v", "b = v in w", "x = v[w]", "v[w] = 1",
          "x = divmod(v, w)", "x = min(v, w)", "x = max(w, v)", "x = [w][v]", "x = {w: 1}[v]", "xs = [0]\nxs[v:w] = [1]", "x = (v, w)\na, b = x",
          "x = random.choice(v) + w", "x = v is w", "x = isinstance(v, type(w))", "x = pow(v, w)", "x = range(v, w)", "x = round(v, w)", "x = list(v) + list(w)"]

#: consumers whose BODY or continuation holds the sentinel: a path wrongly made infeasible hides it (``{S}`` is the sentinel)
BLOCKS = [
    "for x in v:\n    {S}", "for x, y in v:\n    {S}", "for x in v:\n    {S}\nelse:\n    pass", "for x in v:\n    pass\nelse:\n    {S}",
    "for i in range(2):\n    for x in v:\n        {S}", "n = 0\nfor x in v:\n    n += 1\n    {S}", "while v:\n    {S}\n    break",
    "while not v:\n    {S}\n    break", "if v:\n    {S}", "if not v:\n    {S}", "if v is None:\n    {S}", "if v is not None:\n    {S}",
    "if v == 1:\n    {S}", "if v != 1:\n    {S}", "if v < 1:\n    {S}", "if v >= 1:\n    {S}", "if 1 in v:\n    {S}", "if 1 not in v:\n    {S}",
    "if len(v) == 0:\n    {S}", "if len(v) > 0:\n    {S}", "if isinstance(v, int):\n    {S}", "if isinstance(v, (list, tuple)):\n    {S}",
    "if isinstance(v, str):\n    {S}", "if isinstance(v, A):\n    {S}", "if callable(v):\n    {S}", "if v:\n    pass\nelse:\n    {S}",
    "x = v and 1\nif x:\n    {S}", "x = v or 1\nif x:\n    {S}", "x = 1 if v else 2\nif x == 1:\n    {S}", "x = 1 if v else 2\nif x == 2:\n    {S}",
    "with v:\n    {S}", "with v as x:\n    {S}", "a1, b1 = v, 1\n{S}", "x, y = v\n{S}", "(x, y), z = v, 1\n{S}", "x = [e for e in v]\n{S}",
    "x = [mark() // 0 for e in v]", "x = {e: mark() // 0 for e in v}", "x = [mark() for e in v if mark() // 0]", "x = sum(mark() // 0 for e in v)",
    "x = list(v)\nfor e in x:\n    {S}", "x = tuple(v)\nfor e in x:\n    {S}", "x = sorted(v)\nfor e in x:\n    {S}",
    "x = iter(v)\nfor e in x:\n    {S}", "x = min(v)\n{S}", "x = max(v)\n{S}", "x = sum(v)\n{S}", "x = len(v)\n{S}", "x = any(v)\n{S}",
    "x = v[0]\n{S}", "x = v[0:1]\n{S}", "v[0] = 1\n{S}", "v.append(1)\n{S}", "x = v.pop()\n{S}", "x = v + v\n{S}", "x = v * 2\n{S}",
    "x = random.choice(v)\n{S}", "x = random.choice(v)\nif x:\n    {S}", "def g(a):\n    for e in a:\n        {S}\ng(v)", "def g(a):\n    x, y = a\n    {S}\ng(v)",
    "def g(a):\n    if a:\n        {S}\ng(v)", "def g(a):\n    return a[0]\nx = g(v)\n{S}", "class K:\n    def m(self, a):\n        for e in a:\n            {S}\nK().m(v)",
    "x = f(v)\nif x:\n    {S}", "assert v\n{S}", "assert not v\n{S}", "assert v is None\n{S}", "assert isinstance(v, int)\n{S}",
    "t = (v, v)\nfor e in t:\n    {S}", "t = [v]\na2, = t\nfor e in a2:\n    {S}", "d = {1: v}\nfor e in d[1]:\n    {S}", "d = {1: v}\nif d[1]:\n    {S}",
    "d = {}\nd[1] = v\nfor e in d[1]:\n    {S}", "xs = [0]\nxs.extend(v)\nfor e in xs:\n    {S}", "xs = [0]\nxs += v\n{S}", "xs = [0]\nxs[0:1] = v\n{S}",
]

#: consumers that READ the elements of ``v``: an element dropped on the way makes a division by it look safe
ELEMENTS = [
    "xs = [1]\nxs[0:1] = v\nc = 10 // xs[0]", "xs = [1]\nxs.extend(v)\nc = 10 // xs[1]", "xs = [1]\nxs += v\nc = 10 // xs[-1]", "x, = v\nc = 10 // x",
    "x, y = v\nc = 10 // x", "for e in v:\n    c = 10 // e", "c = 10 // min(v)", "c = 10 // max(v)", "c = 10 // sum(v)", "c = 10 // v[0]", "c = 10 // v[-1]",
    "c = 10 // len(v)", "c = 10 // next(iter(v))", "c = 10 // list(v)[0]", "c = 10 // tuple(v)[-1]", "c = 10 // sorted(v)[0]", "c = 10 // random.choice(v)",
    "c = 10 // [e for e in v][0]", "c = 10 // sum(e for e in v)", "ys = list(v)\nfor e in ys:\n    c = 10 // e", "t = tuple(v)\nc = 10 // t[0]",
    "ys = []\nfor e in v:\n    ys.append(e)\nc = 10 // ys[0]", "it = iter(v)\nc = 10 // next(it)", "c = 10 // list(reversed(v))[0]", "c = 10 // list(enumerate(v))[0][1]",
    "c = 10 // list(zip(v, v))[0][0]", "c = 10 // max(v, default=1)", "c = 10 // min(v, default=1)", "c = 10 // (1 if any(v) else 0)", "d = {}\nfor e in v:\n    d[e] = 1\nc = 10 // list(d)[0]",
    "c = 10 // list(set(v))[0]", "c = 10 // list(frozenset(v))[0]", "c = 10 // list(map(abs, v))[0]", "c = 10 // list(filter(None, [0, 1]))[0]", "c = 10 // len(set(v))",
]

FUNCS = [("abs", 1), ("all", 1), ("any", 1), ("ascii", 1), ("bin", 1), ("bool", 1), ("bytearray", 1), ("bytes", 1), ("callable", 1), ("chr", 1),
         ("complex", 1), ("dict", 1), ("dir", 1), ("divmod", 2), ("enumerate", 1), ("filter", 2), ("float", 1), ("format", 2), ("frozenset", 1),
         ("getattr", 2), ("hasattr", 2), ("hash", 1), ("hex", 1), ("id", 1), ("int", 1), ("isinstance", 2), ("issubclass", 2), ("iter", 1),
         ("len", 1), ("list", 1), ("map", 2), ("max", 1), ("max", 2), ("min", 1), ("min", 2), ("next", 1), ("oct", 1), ("ord", 1), ("pow", 2),
         ("print", 1), ("range", 1), ("range", 2), ("repr", 1), ("reversed", 1), ("round", 1), ("round", 2), ("set", 1), ("slice", 1), ("sorted", 1),
         ("str", 1), ("sum", 1), ("sum", 2), ("tuple", 1), ("type", 1), ("vars", 1), ("zip", 1), ("zip", 2), ("memoryview", 1), ("open", 1),
         ("math.sqrt", 1), ("math.floor", 1), ("math.ceil", 1), ("math.fabs", 1), ("math.log", 1), ("math.log", 2), ("math.pow", 2), ("math.gcd", 2),
         ("math.isqrt", 1), ("math.factorial", 1), ("math.comb", 2), ("math.hypot", 2), ("math.atan2", 2), ("math.isnan", 1), ("math.fsum", 1),
         ("math.prod", 1), ("math.trunc", 1), ("math.copysign", 2), ("math.fmod", 2), ("math.ldexp", 2), ("math.dist", 2), ("math.sin", 1),
         ("random.choice", 1), ("random.randint", 2), ("random.randrange", 1), ("random.randrange", 2), ("random.uniform", 2), ("random.sample", 2),
         ("random.shuffle", 1), ("random.getrandbits", 1), ("random.seed", 1), ("random.random", 1), ("re.compile", 1), ("re.match", 2), ("re.sub", 3),
         ("io.StringIO", 1), ("array", 1), ("array", 2), ("copy.copy", 1), ("operator.add", 2), ("sorted", 2)]
FILLERS = ["1", "'a'", "[1, 2]", "None", "2.5"]


def generated_calls():
    out = []
    for name, arity in FUNCS:
        for pos in range(arity):
            for filler in FILLERS if arity > 1 else FILLERS[:1]:
                args = [filler] * arity
                args[pos] = "v"
                out.append(f"{'import copy, operator' + chr(10) if name.startswith(('copy.', 'operator.')) else ''}x = {name}({', '.join(args)})")
    return out


RECEIVERS = ["[1, 2]", "(1, 2)", "{1: 2}", "'ab'", "array('i', [1])", "range(3)", "{1: 2}.keys()", "iter([1])", "re.compile('a')", "io.StringIO()", "A()",
             "{1, 2}", "[None] * 2"]
#: the methods of every builtin type the analysis models, plus names no type has
METHOD_NAMES = ["append", "extend", "insert", "pop", "remove", "index", "count", "sort", "reverse", "copy", "clear", "get", "keys", "values", "items",
                "update", "setdefault", "popitem", "add", "discard", "union", "intersection", "difference", "issubset", "join", "split", "strip",
                "startswith", "endswith", "find", "replace", "format", "encode", "write", "read", "readline", "match", "search", "findall", "sub",
                "tobytes", "fromlist", "tolist", "foo", "x", "m", "real", "bit_length"]


def generated_methods():
    out = []
    for recv in RECEIVERS:
        for m in METHOD_NAMES:
            out.append(f"r = {recv}\nx = r.{m}(v)")
            out.append(f"r = {recv}\nx = r.{m}(1, v)")
            out.append(f"r = {recv}\nx = r.{m}(v, 1)")
    return out


def attribute_reads():
    return [f"x = v.{n}" for n in METHOD_NAMES] + [f"v.{n} = 1" for n in ("foo", "x", "real", "append")] + [f"x = v.{n}()" for n in METHOD_NAMES]


SENTINEL = "reached = True\nc = 1 // 0\n"


#: the values the argument-position matrices run over (every kind once)
CORE = {"int", "zero", "float", "str", "none", "list", "empty_list", "tuple", "array_i", "array_f", "array_empty", "range", "dict", "empty_dict", "keys",
        "values", "items", "iter_list", "instance", "class", "function", "bound_method", "builtin_len", "exc_instance", "module", "pattern", "stringio",
        "set", "frozenset", "bytes", "generator", "seg_list", "nested", "seq_inst", "call_inst"}


def programs():
    out = []
    calls, methods, attrs = generated_calls(), generated_methods(), attribute_reads()
    for vn, vs in VALUES:
        for cs in CONSUMERS:
            out.append((f"{vn} | {cs!r}", vs, cs))
        for ws in OTHERS:
            for cs in BINARY:
                out.append((f"{vn} | {cs!r} w={ws}", vs + f"\nw = {ws}", cs))
        for cs in calls + attrs:
            out.append((f"{vn} | {cs!r}", vs, cs))
        for cs in BLOCKS + ELEMENTS:
            out.append((f"{vn} | {cs!r}", vs, cs))
        if vn in CORE:
            for cs in methods:
                out.append((f"{vn} | {cs!r}", vs, cs))
    return out


PROGRAMS = programs()


#: what a native run of a generated program may use (the owner's rule: no process above 15 GB; a generated program is a child of a
#: worker with these caps, and a program that hits one is SKIPPED, never counted as passed)
CHILD_EXTRA_BYTES = 1536 << 20        # address space the child may add to what the worker already maps
CHILD_CPU_SECONDS = 3
CHILD_WALL_SECONDS = 4
WORKER_BYTES = 4 << 30                # address space of a worker process (absint included)
MAX_WORKERS = 11                      # workers x (WORKER_BYTES + CHILD_EXTRA_BYTES) stays under 64 GB


class Skipped(Exception):
    """A program whose native run hit a cap (memory, CPU or wall time) or died: no evidence either way."""


def _vm_bytes() -> int:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[0]) * os.sysconf("SC_PAGE_SIZE")


def native(src: str) -> tuple[bool, set[str] | None]:
    """``(sentinel reached, the exception class names raised before it)`` of ``src``, run in a forked child under
    ``RLIMIT_AS`` / ``RLIMIT_CPU`` and a wall alarm.  Raises :class:`Skipped` when the child hit a cap, was killed or
    raised ``MemoryError`` (the cap's own symptom)."""
    r, w = os.pipe()
    w_high = fcntl.fcntl(w, fcntl.F_DUPFD, 500)         # a generated ``open(5)`` must not close our pipe
    pid = os.fork()
    if pid == 0:
        status = 3
        try:
            os.close(r)
            os.close(w)
            cap = _vm_bytes() + CHILD_EXTRA_BYTES
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
            resource.setrlimit(resource.RLIMIT_CPU, (CHILD_CPU_SECONDS, CHILD_CPU_SECONDS))
            signal.alarm(CHILD_WALL_SECONDS)
            devnull = open(os.devnull, "w")
            sys.stdout = sys.stderr = devnull
            warnings.simplefilter("ignore")
            ns: dict = {"__name__": "__main__"}
            mod = types.ModuleType("_sv_verifier")
            mod.nondet_int = lambda: 1
            sys.modules["_sv_verifier"] = mod
            try:
                exec(compile(src, "<program>", "exec"), ns)
                result = (False, None, False)
            except MemoryError:
                result = (False, None, True)
            except Exception as e:  # noqa: BLE001
                reached = bool(ns.get("reached"))
                result = (reached, None if reached else sorted(c.__name__ for c in type(e).__mro__), False)
            os.write(w_high, pickle.dumps(result))
            status = 0
        except BaseException:  # noqa: BLE001  (a cap, a signal, an unpicklable result: the parent records SKIPPED)
            status = 3
        finally:
            os._exit(status)
    os.close(w_high)
    os.close(w)
    try:
        ready, _, _ = select.select([r], [], [], CHILD_WALL_SECONDS + 1)
        data = b""
        if ready:
            while True:
                chunk = os.read(r, 65536)
                if not chunk:
                    break
                data += chunk
        else:
            os.kill(pid, signal.SIGKILL)
    finally:
        os.close(r)
        _, wait_status = os.waitpid(pid, 0)
    if not data or not os.WIFEXITED(wait_status) or os.WEXITSTATUS(wait_status) != 0:
        raise Skipped("the child hit a cap or died")
    reached, raised, memory = pickle.loads(data)
    if memory:
        raise Skipped("MemoryError under the child's address-space cap")
    return reached, None if raised is None else set(raised)


def _inline(consumer: str) -> str:
    """``consumer`` with each ``{S}`` replaced by the sentinel at that line's indentation."""
    out = []
    for line in consumer.split("\n"):
        if "{S}" in line:
            indent = line[: len(line) - len(line.lstrip())]
            line = (indent + "reached = True\n" + indent + "c = 1 // 0") if line.strip() == "{S}" else line.replace("{S}", "pass")
        out.append(line)
    return "\n".join(out)


ALL_CLASSES = tuple(sorted({c for cs in PROPERTY_CLASSES.values() for c in cs}))


SKIPPED: list[str] = []          # identity marker: a skipped program is not a pass


def check(label: str, setup: str, consumer: str) -> list[str]:
    """The wrong TRUEs of one program (empty: none)."""
    # a consumer with the sentinel INSIDE (a body, a branch, a comprehension) has no sentinel after it: a body wrongly made infeasible
    # must not be masked by an unproven check on the continuation
    tail = "" if "{S}" in consumer or "mark()" in consumer else SENTINEL
    src = PRELUDE + setup + "\n" + _inline(consumer) + "\n" + tail
    try:
        reached, raised = native(src)
    except Skipped:
        return SKIPPED
    wrong = []
    if reached:
        classes = ["ZeroDivisionError"]
    else:
        classes = [c for c in (raised or ()) if c in ALL_CLASSES]
    for cls in classes:
        r = prove(src, [cls], filename="<program>", time_budget=20.0)
        if r.verdict == "TRUE":
            wrong.append(f"{label}: TRUE for {cls} but native {'reached the sentinel' if reached else 'raised ' + cls}\n{setup}\n{consumer}")
    return wrong


SHARDS = 64


def sample(shard: int):
    """A deterministic 1/SHARDS slice of every program (the gate runs a few; ``scan`` runs them all)."""
    return PROGRAMS[shard::SHARDS]


#: what the gate runs: eight of the 64 slices, i.e. every eighth program -- as before the split by kind (the 8 shards of 8 slices
#: ran 140-215 s each and were the critical path of the whole suite)
GATE_PROGRAMS = [program for shard in range(0, SHARDS, 8) for program in sample(shard)]


def kind_of(label: str) -> str:
    """The value kind a program's label starts with (``"int | 'x = len(v)'"`` -> ``"int"``)."""
    return label.split(" | ", 1)[0]


#: one test node per chunk of one value kind's gate programs: the same programs, in several hundred nodes of at most 80 programs
#: (a few seconds each, ~13 s for the slowest kind) -- split by kind first, then by position inside the kind, so no node is a large share of the matrix
CHUNK = 80
KINDS = list(dict.fromkeys(kind_of(label) for label, _, _ in GATE_PROGRAMS))
BY_KIND = {kind: [p for p in GATE_PROGRAMS if kind_of(p[0]) == kind] for kind in KINDS}
NODES = [(kind, i, -(-len(ps) // CHUNK)) for kind, ps in BY_KIND.items() for i in range(-(-len(ps) // CHUNK))]


def run_shard(items) -> tuple[list[str], int, int]:
    """``(wrong TRUEs, programs checked, programs skipped)``."""
    wrong, skipped = [], 0
    for label, setup, consumer in items:
        got = check(label, setup, consumer)
        if got is SKIPPED:
            skipped += 1
        else:
            wrong += got
    return wrong, len(items) - skipped, skipped


@pytest.mark.parametrize("kind,index,of", NODES, ids=[f"{kind}-{i}" for kind, i, _ in NODES])
def test_no_kind_of_value_makes_a_consumers_path_infeasible(kind, index, of):
    wrong, checked, skipped = run_shard(BY_KIND[kind][index::of])
    assert not wrong, f"{len(wrong)} wrong TRUE(s); first:\n{wrong[0]}\nall: {[w.splitlines()[0] for w in wrong]}"
    assert checked > 5 * skipped


def test_the_nodes_run_exactly_the_gate_programs():
    """Chunking by kind and position loses and repeats nothing: the nodes together are GATE_PROGRAMS, once each."""
    seen = [program[0] for kind, index, of in NODES for program in BY_KIND[kind][index::of]]
    assert sorted(seen) == sorted(label for label, _, _ in GATE_PROGRAMS)
    assert all(len(BY_KIND[kind][index::of]) <= CHUNK for kind, index, of in NODES)
