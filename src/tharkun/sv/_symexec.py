"""Symbolic execution of the SV harness up to one violation site, solved with z3 ([internal ref] slice 2).

``drive_to_line`` flips one guard at a time on concrete executions.  A violation
guarded by a quantity the program ASSEMBLES from many draws -- ``pick``'s
``key = sum(slots[i] << i)`` over sixteen ``nondet_bool()`` draws, compared with a
``needle`` -- has no single guard to flip: every draw matters at once.  This module
solves such a site directly:

1. **Slice.** Starting from the site's condition (the ``if <cond>: __sv_site(n)``
   marker :mod:`tharkun.sv.sites` inserted), walk the harness body backwards and keep
   only the statements that define a name the condition (or a kept statement)
   needs -- data and control dependence. The shared effect-rooted slicer also
   keeps calls, exceptions, mutations and loops that may prevent reaching it.
2. **Execute symbolically.**  Draws are z3 variables (``int`` -> ``Int``, ``bool`` ->
   ``Bool``, ``float`` -> ``Real``; a stream ``sK`` is a list of them, as long as the
   concrete default run consumed).  Loops over concrete ranges are unrolled; an
   ``if`` on a symbolic test runs both arms and MERGES the results with ``If`` (no
   path explosion: ``pick`` with N = 16 is one formula, not 2**16 paths).  Module
   constants come from the harness module itself.
3. **Solve** ``path condition AND site condition``; the model is the draw inputs.

Anything outside this subset raises :class:`Unsupported` and the caller falls back
to ``drive_to_line`` and enumeration.  Nothing here is trusted: the inputs are
re-run concretely and replayed on the unmodified program before any FALSE.
"""

from __future__ import annotations

import ast
import bisect
import builtins
import heapq
import itertools
from types import ModuleType
import copy
import operator
import re

from tharkun.solve.contracts.rules import Refused, get_rule
from tharkun.sv.symexec_visitor import NATIVE_METHODS, NATIVE_MODULE_FUNCTIONS, CallCtx, SymexecVisitor

import z3

_PYOP = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
         ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
         ast.Pow: operator.pow, ast.LShift: operator.lshift, ast.RShift: operator.rshift,
         ast.BitAnd: operator.and_, ast.BitOr: operator.or_, ast.BitXor: operator.xor}
_PYCMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
          ast.Gt: operator.gt, ast.GtE: operator.ge, ast.Is: operator.is_, ast.IsNot: operator.is_not}
#: builtin types a site condition may name (``isinstance(c, (list, tuple, str))``)
#: largest shift amount the interpreter evaluates (see ev_BinOp)
MAX_SHIFT = 4096
_TYPES = {t.__name__: t for t in (list, tuple, str, dict, int, float, bool, set, range)}


class Unsupported(Exception):
    """The site's slice uses a construct this executor does not model."""


#: sentinel: "no contract applies" ([internal ref] step 2), distinct from any legitimate
#: return value (including ``None``, which some modelled calls return for real).
_NO_CONTRACT = object()

_NATIVE_MODULES = {"bisect": bisect, "heapq": heapq, "copy": copy, "itertools": itertools}
_NATIVE_CALLABLES = {
    name: _NATIVE_MODULES[name.split(".")[0]].__dict__[name.split(".")[1]]
    for name in NATIVE_MODULE_FUNCTIONS
}


def _exact_type(value, classes):
    # Membership by equality can invoke a user metaclass while checking admission.
    return any(type(value) is cls for cls in classes)


class Ref:
    """A handle to a user-class instance on the per-path heap of ``sv/_symexec_objects.py``.

    A handle is interned (one instance per object), so ``a is b`` is object identity.  It carries only what
    never changes -- the object's number, its live class and the dunders its class chain defines -- and EVERY
    operation but identity is refused: the model of an object is the heap, and an operator, a conversion or a
    format would run user code this executor does not run (``__eq__`` of a class that defines one,
    ``__bool__``, ``__str__``, ``a + b``, ``len(a)``, ``a[i]``, ``for x in a``).  What Python does by default
    for an object that defines none of those (identity equality, truthiness ``True``, hashing by identity) is
    exactly what is answered."""

    __slots__ = ("oid", "cls", "user_dunders")

    def __init__(self, oid, cls, user_dunders):
        self.oid, self.cls, self.user_dunders = oid, cls, user_dunders

    def _user(self, *names):
        for name in names:
            if name in self.user_dunders:
                raise Unsupported(f"the user-defined {name} of {self.cls.__name__} would run")

    def _refuse(self, *_args):
        raise Unsupported(f"an operator, conversion or format applied to a {self.cls.__name__} object")

    def __eq__(self, other):
        self._user("__eq__")
        if isinstance(other, Ref):
            other._user("__eq__")
        return self is other

    def __ne__(self, other):
        self._user("__eq__", "__ne__")
        if isinstance(other, Ref):
            other._user("__eq__", "__ne__")
        return self is not other

    def __hash__(self):
        self._user("__eq__", "__hash__")
        return hash(self.oid)

    def __bool__(self):
        self._user("__bool__", "__len__")
        return True

    def __repr__(self):                 # debugging only: no model path may call it (``__str__`` refuses)
        return f"Ref({self.oid}, {self.cls.__name__})"

    __str__ = __format__ = _refuse
    __lt__ = __le__ = __gt__ = __ge__ = __index__ = __int__ = __float__ = _refuse
    __iter__ = __len__ = __getitem__ = __contains__ = __call__ = _refuse
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _refuse
    __floordiv__ = __rfloordiv__ = __mod__ = __rmod__ = __pow__ = __rpow__ = __neg__ = __pos__ = _refuse
    __and__ = __rand__ = __or__ = __ror__ = __xor__ = __rxor__ = __lshift__ = __rlshift__ = _refuse
    __rshift__ = __rrshift__ = __abs__ = __invert__ = _refuse


class SymList:
    """A list whose length is symbolic and whose elements are unknown (``[f(i) for i in
    range(n)]`` with symbolic ``n``): only ``len`` is modelled."""

    def __init__(self, length):
        self.length = length


class SymRange:
    """``range(start, stop, step)`` with at least one symbolic INT term ([internal ref] S1).

    ``step != 0`` holds on the path that built it (``range`` raises ``ValueError`` for a zero
    step, and a path that raises ends).  Only iteration is modelled, by ``_Runner._range_loop``;
    every other use of a range object (truth, comparison, ``len``, indexing, ``in``, iteration
    by hand) raises :class:`Unsupported` instead of falling back to the Python object's own
    behaviour, which would be a different program (an empty range is falsy, two ranges compare
    by value, ...)."""

    def __init__(self, start, stop, step):
        self.start, self.stop, self.step = start, stop, step

    def _refuse(self, *_args):
        raise Unsupported("symbolic range used as a value")

    __bool__ = __len__ = __iter__ = __getitem__ = __contains__ = _refuse
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _refuse
    __hash__ = None


class _OutHandle:
    """The file a write-only ``open()`` returned ([internal ref], ``contracts/rules_fileio.py``): ``binary`` is
    True for a binary mode.  It carries no content and no closed flag: what is written cannot
    reach the program again except through a read (a handle opened for output has none), and what
    a closed file changes (``ValueError``) and what the environment can do (``OSError``) are not
    property classes, so a path that raises them simply ends and the rest are modelled.  Every
    use but a call of the family (a truth test, a comparison, an attribute, a subscript) raises
    :class:`Unsupported`."""

    def __init__(self, binary: bool):
        self.binary = binary

    def _refuse(self, *_args):
        raise Unsupported("an output file used as a value")

    __bool__ = __len__ = __iter__ = __getitem__ = __contains__ = _refuse
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _refuse
    __hash__ = None


def _is_sym(v) -> bool:
    return isinstance(v, z3.ExprRef)


def _num(v):
    """``v`` as an arithmetic term (a Bool counts as 0/1, as in Python)."""
    if type(v) is re.RegexFlag:
        return int(v)  # [internal ref]: numeric operations return/use the integer view, not enum str.
    if isinstance(v, bool):
        return int(v)
    if _is_sym(v) and z3.is_bool(v):
        return z3.If(v, 1, 0)
    if isinstance(v, (int, float)) or _is_sym(v):
        return v
    raise Unsupported(f"not a number: {type(v).__name__}")


def _truth(v):
    if _is_sym(v):
        return v if z3.is_bool(v) else (z3.Length(v) > 0 if v.sort() == z3.StringSort() else v != 0)
    if isinstance(v, SymList):
        return v.length > 0
    return bool(v)


def _same_object(a, b):
    """``a`` when ``a`` and ``b`` are the one object; otherwise the paths disagree on which object a value
    is, which this executor cannot represent (an ``ite`` of two handles is no handle)."""
    if a is b:
        return a
    raise Unsupported("a value that is a different object (or not an object) on different paths")


def _ite(c, a, b):
    if isinstance(a, Ref) or isinstance(b, Ref):
        return _same_object(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise Unsupported("merging dicts with different keys")
        return {k: _ite(c, a[k], b[k]) for k in a}
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise Unsupported("merging lists of different lengths")
        return [_ite(c, x, y) for x, y in zip(a, b)]
    if isinstance(a, tuple) and isinstance(b, tuple):
        # Same recursive, elementwise merge as the list/list case above --
        # a symbolic dict index whose VALUES are tuples (e.g. hexiom's
        # ``LEVELS[level] = (board_text, solution_text)``) needs this to
        # merge the two branches' tuples element by element instead of
        # falling all the way to the numeric-only default below.
        if len(a) != len(b):
            raise Unsupported("merging tuples of different lengths")
        return tuple(_ite(c, x, y) for x, y in zip(a, b))
    if isinstance(a, SymList) or isinstance(b, SymList):
        la = a.length if isinstance(a, SymList) else (len(a) if isinstance(a, list) else None)
        lb = b.length if isinstance(b, SymList) else (len(b) if isinstance(b, list) else None)
        if la is None or lb is None:
            raise Unsupported("merging a list with a non-list")
        return SymList(z3.If(c, la, lb))
    if not (_is_sym(a) or _is_sym(b)) and type(a) is type(b) and a == b:
        return a
    if (_is_sym(a) and z3.is_bool(a)) or isinstance(a, bool):
        if (_is_sym(b) and z3.is_bool(b)) or isinstance(b, bool):
            return z3.If(c, a, b)
    def _is_strlike(v):
        return isinstance(v, str) or (_is_sym(v) and v.sort() == z3.StringSort())

    if _is_strlike(a) or _is_strlike(b):
        # A concrete Python ``str`` on either side is wrapped as a z3
        # string literal first -- the OLD condition only recognised a side
        # that was ALREADY a symbolic z3 string, so two DIFFERENT concrete
        # strings (e.g. two levels' board texts) fell through to the
        # numeric-only default below and raised "not a number: str".
        # Mirrors :func:`_choice_ite`'s own manual wrapping for the same
        # reason, generalised here so every _ite call site benefits (dict-
        # value merging included), not just a symbolic-index-into-a-list.
        # Both sides must be string-like: a real (not just Python-truthy)
        # type mismatch here would build a z3.If with mismatched sorts,
        # which raises inside z3 itself rather than declining cleanly.
        if not (_is_strlike(a) and _is_strlike(b)):
            raise Unsupported("merging a string with a non-string")
        av = z3.StringVal(a) if isinstance(a, str) else a
        bv = z3.StringVal(b) if isinstance(b, str) else b
        return z3.If(c, av, bv)
    return z3.If(c, _num(a), _num(b))


def _choice_ite(idx, seq):
    """``seq[idx]`` for a symbolic ``idx`` into a CONCRETE ``seq`` ([internal ref]): the same
    cascading ite :func:`_ite` already builds for a symbolic subscript, except every
    concrete ``str`` element is wrapped as a z3 literal first -- ``_ite`` only takes its
    string-sort branch when at least one side is already symbolic, so two *different*
    concrete strings (``'sun'`` vs ``'jupiter'``) would otherwise fall through to the
    numeric-only default and raise. Ints/floats/bools need no such wrapping: ``_ite``
    (via ``_num``) already merges those directly."""
    elems = [z3.StringVal(x) if isinstance(x, str) else x for x in seq]
    out = elems[-1]
    for i in range(len(elems) - 2, -1, -1):
        out = _ite(idx == i, elems[i], out)
    return out


_ARITH = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
}


class _LazyIter:
    """The value of ``zip(...)``/``enumerate(...)``/``reversed(...)``: a one-shot iterator.

    The contract-driven fold (``solve/contracts/adapters/concolic.py::concrete_fold``) builds
    its items eagerly. Only a consumer that reads the iterator right away (a ``for``, a
    comprehension, ``list``/``tuple``/``sorted``/``sum``) unwraps it, through
    ``_Exec._seq``; every other use sees an object that is neither a list nor a tuple and
    is ``Unsupported``. A second consumption is ``Unsupported`` too: a real iterator is
    exhausted after the first, and this executor does not model that.
    """

    __slots__ = ("_items",)

    def __init__(self, items: list):
        self._items = items

    def take(self) -> list:
        if self._items is None:
            raise Unsupported("an iterator consumed twice")
        items, self._items = self._items, None
        return items


class _Exec:
    def __init__(self, globals_: dict, env: dict, streams: dict):
        self.g = globals_
        self.env = env
        self.streams = streams          # param -> [symbolic draws]
        self.pos = {}                   # param -> next index
        self.pc = []                    # path conditions (control dependence)

    # --- expressions --------------------------------------------------------
    def ev(self, n):
        m = getattr(self, "ev_" + type(n).__name__, None)
        if m is None:
            raise Unsupported(type(n).__name__)
        return m(n)

    def _seq(self, v):
        """``v`` with a one-shot iterator unwrapped to the list it stands for."""
        return v.take() if isinstance(v, _LazyIter) else v

    def ev_Constant(self, n):
        return n.value

    def ev_Name(self, n):
        if n.id in self.env:
            return self.env[n.id]
        if n.id in self.g:
            v = self.g[n.id]
            if _exact_type(v, (int, float, bool, str, tuple, re.RegexFlag)) or v is None:
                return v
            if type(v) is list and all(_exact_type(x, (int, float, bool)) for x in v):
                return list(v)
            # a module-level dict table with plain (hashable, comparable) scalar keys
            # ([internal ref], e.g. ``BODIES = {"sun": (...), ...}``) -- only KEYS need to be
            # simple here; a value this executor cannot later handle (ev_Subscript,
            # ev_BinOp, ...) still raises Unsupported there, same as any other value.
            if type(v) is dict and all(_exact_type(k, (int, float, bool, str)) for k in v):
                return dict(v)
            if callable(v) or isinstance(v, type):
                return v
        if n.id in _TYPES:
            return _TYPES[n.id]
        raise Unsupported(f"name {n.id}")

    def ev_Tuple(self, n):
        return tuple(self.ev(e) for e in n.elts)

    def ev_List(self, n):
        return [self.ev(e) for e in n.elts]

    def ev_Dict(self, n):
        out = {}
        for kn, vn in zip(n.keys, n.values):
            if kn is None:
                raise Unsupported("dict unpacking")
            k = self.ev(kn)
            if _is_sym(k) or isinstance(k, (list, dict, SymList, Union)):
                raise Unsupported("symbolic dict key")
            out[k] = self.ev(vn)
        return out

    def ev_UnaryOp(self, n):
        v = self.ev(n.operand)
        if isinstance(n.op, ast.Not):
            t = _truth(v)
            return z3.Not(t) if _is_sym(t) else (not t)
        if isinstance(n.op, ast.USub):
            return -_num(v)
        if isinstance(n.op, ast.UAdd):
            return _num(v)
        raise Unsupported("unary op")

    def ev_BinOp(self, n):
        a, b = self.ev(n.left), self.ev(n.right)
        op = type(n.op)
        if op is ast.Mod and (type(a) is str or (_is_sym(a) and a.sort() == z3.StringSort())):
            return self._string_rule("str.__mod__", n, [a, b])

        if op in {ast.BitAnd,ast.BitOr,ast.BitXor} and (_is_sym(a) or _is_sym(b)) and any(type(value) is re.RegexFlag for value in (a,b)):
            raise Unsupported("re.RegexFlag: symbolic bitwise result requires an exact enum encoding")

        if op in (ast.LShift, ast.RShift) and isinstance(b, int) and b > MAX_SHIFT:
            # 2**b (symbolic) or a << b (concrete) is computed in-process: bound it, so a
            # huge constant makes the run inexact instead of a MemoryError / memory spike
            raise Unsupported(f"shift by {b} > {MAX_SHIFT}")
        if isinstance(a, list) and op is ast.Mult and isinstance(b, int):
            return a * b
        if isinstance(a, list) and op is ast.Mult and _is_sym(b):
            return SymList(z3.If(b > 0, b, 0))
        if isinstance(a, list) and isinstance(b, list) and op is ast.Add:
            return a + b
        if not (_is_sym(a) or _is_sym(b)):
            try:
                return _PYOP[op](a, b)
            except Exception as e:  # noqa: BLE001
                raise Unsupported(f"concrete op failed: {e}") from None
        if op is ast.Add and (isinstance(a, str) or (_is_sym(a) and a.sort() == z3.StringSort())):
            return z3.Concat(a, b)
        if op is ast.Mult and isinstance(b, int) and 0 <= b <= 64 and (_is_sym(a) and a.sort() == z3.StringSort()):
            return z3.Concat(*([a] * b)) if b > 1 else a if b else z3.StringVal("")
        a, b = _num(a), _num(b)
        if op in _ARITH:
            return _ARITH[op](a, b)
        if op is ast.LShift and isinstance(b, int) and b >= 0:
            return a * (2 ** b)
        if op is ast.RShift and isinstance(b, int) and b >= 0:
            return _floordiv(a, 2 ** b)
        if op is ast.FloorDiv and isinstance(b, int) and b > 0:
            return _floordiv(a, b)
        if op is ast.Mod and isinstance(b, int) and b > 0:
            _int_term(a)
            return a % b                 # z3's mod is Euclidean: equal to Python's for b > 0
        if op is ast.Pow and isinstance(b, int) and 0 <= b <= 8:
            r = 1
            for _ in range(b):
                r = r * a
            return r
        raise Unsupported(f"symbolic {op.__name__}")

    def ev_BoolOp(self, n):
        vals = [_truth(self.ev(v)) for v in n.values]
        if not any(_is_sym(v) for v in vals):
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        return z3.And(*vals) if isinstance(n.op, ast.And) else z3.Or(*vals)

    def ev_Compare(self, n):
        left = self.ev(n.left)
        parts = []
        for op, rn in zip(n.ops, n.comparators):
            right = self.ev(rn)
            parts.append(self._cmp(op, left, right))
            left = right
        if not any(_is_sym(p) for p in parts):
            return all(parts)
        return z3.And(*parts) if len(parts) > 1 else parts[0]

    def _cmp(self, op, a, b):
        if isinstance(op, (ast.In, ast.NotIn)):
            if type(b) is str or (_is_sym(b) and b.sort() == z3.StringSort()):
                result = self._string_rule("str.__contains__", None, [b, a])
                return (z3.Not(result) if _is_sym(result) else not result) if isinstance(op, ast.NotIn) else result
            if isinstance(b, (list, tuple)) and not any(_is_sym(x) for x in b):
                if _is_sym(a):
                    r = z3.Or(*[_num(a) == x for x in b]) if b else False
                else:
                    r = a in b
                return (z3.Not(r) if _is_sym(r) else not r) if isinstance(op, ast.NotIn) else r
            raise Unsupported("in")
        if not (_is_sym(a) or _is_sym(b)):
            try:
                return _PYCMP[type(op)](a, b)
            except Exception as e:  # noqa: BLE001
                raise Unsupported(f"concrete compare failed: {e}") from None
        if (isinstance(a, str) or isinstance(b, str) or
                (_is_sym(a) and a.sort() == z3.StringSort()) or
                (_is_sym(b) and b.sort() == z3.StringSort())):
            if not isinstance(op, (ast.Eq, ast.NotEq)):
                raise Unsupported("symbolic string comparison")
            return a == b if isinstance(op, ast.Eq) else a != b
        a, b = _num(a), _num(b)
        return {ast.Eq: lambda: a == b, ast.NotEq: lambda: a != b, ast.Lt: lambda: a < b,
                ast.LtE: lambda: a <= b, ast.Gt: lambda: a > b, ast.GtE: lambda: a >= b}[type(op)]()

    def _string_rule(self, name, node, args, kwargs=None):
        rule = get_rule(name)
        if rule is None:
            raise Unsupported(f"{name}: no rule")
        result = rule.accept(SymexecVisitor(self), CallCtx(node, args=tuple(args), kwargs=kwargs or {}))
        if isinstance(result, Refused):
            raise Unsupported(str(result))
        return result

    def ev_JoinedStr(self, n):
        values = [self.ev(x) for x in n.values]
        if any(_is_sym(x) for x in values):
            # Exact concatenation of already-formatted strings, including the empty f-string.
            self._string_rule("syntax.JoinedStr", n, [])
            return z3.Concat(*values) if len(values) > 1 else values[0] if values else ""
        return self._string_rule("syntax.JoinedStr", n, values)

    def ev_FormattedValue(self, n):
        value = self.ev(n.value)
        if n.conversion != -1:
            value = self._string_rule({115: "str", 114: "repr", 97: "ascii"}[n.conversion], n, [value])
        spec = self.ev(n.format_spec) if n.format_spec is not None else ""
        return self._string_rule("syntax.FormattedValue", n, [value, spec, -1])

    def ev_IfExp(self, n):
        t = _truth(self.ev(n.test))
        if not _is_sym(t):
            return self.ev(n.body) if t else self.ev(n.orelse)
        return _ite(t, self.ev(n.body), self.ev(n.orelse))

    def ev_Subscript(self, n):
        c = self.ev(n.value)
        if type(c) is str:
            if isinstance(n.slice, ast.Slice):
                key = slice(*(self.ev(x) if x is not None else None for x in (n.slice.lower, n.slice.upper, n.slice.step)))
            else:
                key = self.ev(n.slice)
            return self._string_rule("str.__getitem__", n, [c, key])
        import re
        if type(c) is re.Match:
            return self._string_rule("re.Match.__getitem__", n, [c, self.ev(n.slice)])

        if isinstance(n.slice, ast.Slice):
            raise Unsupported("slice")
        k = self.ev(n.slice)
        if _exact_type(c, (list, tuple)):
            if not _is_sym(k):
                return c[k]
            if not c:
                raise Unsupported("symbolic index into an empty list")
            out = c[-1]
            for i in range(len(c) - 2, -1, -1):
                out = _ite(k == i, c[i], out)
            return out
        raise Unsupported("subscript")

    #: the builtins whose keyword arguments ``ev_Call`` reads itself (``default=``, ``key=None``, ``start=``,
    #: ``reverse=``); a call with a keyword to anything else is refused
    _KEYWORD_BUILTINS = frozenset({"min", "max", "sum", "sorted", "print"})

    def _math_function(self, f):
        """The ``math`` function the callee expression ``f`` stands for (``math.sqrt``, or a name bound
        by ``from math import sqrt``), or ``None``.  Resolved by IDENTITY in the harness module's
        globals, so a local, or a program function that happens to be called ``sqrt``, is never taken
        for it (and the exact product proof, which runs with no globals, never sees one)."""
        import math

        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id not in self.env \
                and self.g.get(f.value.id) is math:
            return f.attr
        if isinstance(f, ast.Name) and f.id not in self.env:
            obj = self.g.get(f.id)
            name = getattr(obj, "__name__", None)
            if name is not None and callable(obj) and getattr(math, name, None) is obj:
                return name
        return None

    def ev_Attribute(self, n):
        """``math.pi`` and the other constants (``contracts/adapters/math_model.py``)."""
        import math

        import re
        if isinstance(n.value, ast.Name) and n.value.id not in self.env and self.g.get(n.value.id) is re:
            flag=getattr(re,n.attr,None)
            if type(flag) is re.RegexFlag:
                return flag  # [internal ref]: conversion/bitwise semantics need the actual enum.
        if isinstance(n.value, ast.Name) and n.value.id not in self.env and self.g.get(n.value.id) is math:
            rule = get_rule(f"math.{n.attr}")
            if rule is not None and rule.kind == "math_constant":
                value = self._visit_rule(rule.qualname, CallCtx(n))
                if value is not None and value is not _NO_CONTRACT:
                    return value
        if n.attr in {"pattern","flags","groups","groupindex","string","pos","endpos","lastindex","lastgroup"}:
            value=self.ev(n.value)
            if _exact_type(value, {re.Pattern,re.Match}) and hasattr(value,n.attr):
                return getattr(value,n.attr)
        raise Unsupported(type(n).__name__)

    def ev_Call(self, n):
        math_name = self._math_function(n.func)
        if math_name is not None:
            rule = get_rule(f"math.{math_name}")
            if rule is not None and rule.kind == "math_function":
                return self._visit_rule(rule.qualname, CallCtx(n))
            raise Unsupported(f"math.{math_name}: TypeError (constant is not callable)" if rule is not None and rule.kind == "math_constant" else f"math.{math_name} has no tested contract")
        import re
        f = n.func
        if any(k.arg is None for k in n.keywords):
            raise Unsupported("keyword expansion is not modelled")
        if isinstance(f, ast.Name) and f.id not in self.env and f.id not in getattr(self, "defs", {}) and not (getattr(self, "frames", []) and f.id in self.frames[0]):
            obj = self.g.get(f.id)
            for name in ("format", "repr", "ascii", "str"):
                if obj is getattr(builtins, name):
                    return self._string_rule(name, n, [self.ev(x) for x in n.args], {k.arg: self.ev(k.value) for k in n.keywords})
            for name in ("compile", "match", "search", "fullmatch", "findall", "finditer", "split", "sub", "subn", "escape", "purge"):
                if obj is getattr(re, name):
                    return self._string_rule("re." + name, n, [self.ev(x) for x in n.args], {k.arg: self.ev(k.value) for k in n.keywords})
        if isinstance(f, ast.Name) and not (getattr(self, "frames", []) and f.id in self.frames[0]) and f.id in {"format", "repr", "ascii", "str"} and f.id not in self.env and f.id not in getattr(self, "defs", {}) and (f.id not in self.g or self.g[f.id] is getattr(builtins,f.id)):
            return self._string_rule(f.id, n, [self.ev(x) for x in n.args], {k.arg: self.ev(k.value) for k in n.keywords if k.arg is not None})
        if isinstance(f, ast.Attribute):
            if isinstance(f.value, ast.Name) and f.value.id not in self.env and self.g.get(f.value.id) is re:
                return self._string_rule("re." + f.attr, n, [self.ev(x) for x in n.args], {k.arg: self.ev(k.value) for k in n.keywords if k.arg is not None})
            value = self.ev(f.value) if f.attr != "keys" and (get_rule("str." + f.attr) is not None or get_rule("re.Pattern." + f.attr) is not None or get_rule("re.Match." + f.attr) is not None) else None
            if _exact_type(value, {str, re.Pattern, re.Match}) or (_is_sym(value) and value.sort() == z3.StringSort()):
                owner = "re.Pattern" if type(value) is re.Pattern else "re.Match" if type(value) is re.Match else "str"
                return self._string_rule(owner + "." + f.attr, n, [value] + [self.ev(x) for x in n.args], {k.arg: self.ev(k.value) for k in n.keywords if k.arg is not None})
            if value is str and get_rule("str." + f.attr) is not None:
                return self._string_rule("str." + f.attr, n, [self.ev(x) for x in n.args], {k.arg: self.ev(k.value) for k in n.keywords if k.arg is not None})
        if n.keywords and not (isinstance(n.func, ast.Name) and n.func.id in self._KEYWORD_BUILTINS):
            raise Unsupported("keywords")
        f = n.func
        if isinstance(f, ast.Name):
            name = f.id
            if name == "__sv_draw":
                v = self.ev(n.args[1])
                if len(n.args) == 4:
                    lo, hi = self.ev(n.args[2]), self.ev(n.args[3])
                    v = z3.If(v < lo, lo, z3.If(v > hi, hi, v)) if _is_sym(v) else min(max(v, lo), hi)
                return v
            if name == "__sv_next":
                p = n.args[1].id
                i = self.pos.get(p, 0)
                self.pos[p] = i + 1
                s = self.streams.get(p, [])
                if i >= len(s):
                    raise Unsupported(f"stream {p} consumed past its modelled length")
                return s[i]
            if name == "__sv_index":
                # random.choice(seq), lifted ([internal ref]), read through its rule ([internal ref] step 2): ``_model_random_choice``
                return self._through_rule("random.choice", n)
            # the builtins below are read through their rules ([internal ref] step 2): ``_through_rule`` asks the rule to accept this
            # executor's visitor (``sv/symexec_visitor.py``), whose visit gates on a TESTED contract and runs ``_model_<name>``
            if name == "len":
                return self._through_rule("len", n)
            if name == "abs":
                return self._through_rule("abs", n)
            if name in ("min", "max") and (len(n.args) == 2 and not n.keywords or len(n.args) == 1):
                return self._through_rule(name, n)
            if name in ("any", "all") and len(n.args) == 1 and not n.keywords:
                return self._through_rule(name, n)
            if name in ("int", "bool") and len(n.args) == 1:
                if name == "bool":
                    return _truth(self.ev(n.args[0]))
                return self._through_rule("int", n)
            if name == "float" and len(n.args) == 1 and not n.keywords:
                return self._through_rule("float", n)
            if name in ("divmod", "pow") and len(n.args) == 2 and not n.keywords:
                return self._through_rule(name, n)
            if name == "print":
                return self._through_rule("print", n)
            if name == "isinstance":
                return self._through_rule("isinstance", n)
            if name == "sum" and 1 <= len(n.args) <= 2 and all(k.arg == "start" for k in n.keywords) \
                    and len(n.args) + len(n.keywords) <= 2:
                return self._through_rule("sum", n)
            if name == "range":
                return self._through_rule("range", n)
            if name == "sorted" and len(n.args) == 1 and all(k.arg in ("reverse", "key") for k in n.keywords):
                return self._through_rule("sorted", n)
            folded = self._contract_fold(name, n)
            if folded is not _NO_CONTRACT:
                return folded
        if isinstance(f, ast.Attribute):
            if f.attr == "keys" and not n.args and not n.keywords:
                return self._through_rule("dict.keys", n)
        raise Unsupported(f"call {ast.unparse(f)}")

    def _model_dict_keys(self, n, snapshot=None):
        if type(snapshot) is dict and not n.args and not n.keywords:
            return list(snapshot.keys())
        raise Unsupported("live dict views require alias tracking")

    def _references(self, obj) -> int:
        """How many places of the executor's state hold ``obj`` itself: the current environment, every caller's, and any list, tuple or
        dict reachable from them (a value this executor binds to a name or stores in a container)."""
        seen, count = set(), 0

        def walk(v):
            nonlocal count
            if v is obj:
                count += 1
                return
            if id(v) in seen:
                return
            seen.add(id(v))
            if isinstance(v, (list, tuple)):
                for x in v:
                    walk(x)
            elif isinstance(v, dict):
                for x in v.values():
                    walk(x)

        for env in [self.env, *getattr(self, "frames", [])]:
            for v in env.values():
                walk(v)
        return count

    def _mutable(self, name: str, kinds):
        """The list or dict ``name`` is bound to in THIS frame, for a mutation the executor models by rebinding the name.  A list is
        bound by identity (CPython's aliasing), and this executor's mutations are functional, so a mutation is modelled only where no
        other name, caller frame or container holds the object and the frame owns the name: otherwise it is refused by name, never
        applied to one of two names that CPython would change together."""
        obj = self.env.get(name)
        if not _exact_type(obj, kinds):
            raise Unsupported(f"mutation of {name}: not a {'/'.join(k.__name__ for k in kinds)} of this frame")
        if self._references(obj) != 1:
            raise Unsupported(f"mutation of {name}: the object is reachable by another name, frame or container (aliasing is not modelled)")
        return obj

    def _model_list___setitem__(self, t, v):
        c = self._mutable(t.value.id, (list,))
        k = self.ev(t.slice)
        c = list(c)
        if _is_sym(k):
            k = _num(k)
            if not z3.is_int(k):
                raise Unsupported("a list index that is not an int")
            m = len(c)
            self.pc.append(z3.And(k >= -m, k < m))
            c = [_ite(z3.Or(k == i, k == i - m), v, x) for i, x in enumerate(c)]
        elif _exact_type(k, (int, bool)):
            if not -len(c) <= k < len(c):
                raise _Dead()                          # IndexError: the run ends here, as CPython's does
            c[k] = v
        else:
            raise Unsupported("store under an index that is not an int")
        self.env[t.value.id] = c

    def _through_rule(self, qualname, n, args=None):
        """A call or an operation, read through its rule ([internal ref] step 2): the rule accepts this executor's visitor, whose visit
        refuses without a TESTED contract and otherwise runs the model ``_model_<qualname>`` below (``.`` read as ``_``) on the
        node and on ``args``, the operands already evaluated.  A refusal reaches the call as the ``Unsupported`` the statement is
        skipped with, as before the contract was consulted."""
        out = self._visit_rule(qualname, CallCtx(n, args=args))
        if out is _NO_CONTRACT:
            raise Unsupported(f"{qualname}: no tested contract")
        return out

    def _model_random_choice(self, n):
        """``random.choice(seq)`` as the harness lifts it ([internal ref]: ``__sv_index(kind, idx, seq)``): ``seq`` must be concrete (the same
        requirement ev_Subscript already has for a symbolic index); the index itself may be a fresh symbolic var with no other constraint
        here -- ``_Runner`` (the site-reachability walker) adds the in-range one."""
        idx, seq = self.ev(n.args[1]), self.ev(n.args[2])
        if not isinstance(seq, (list, tuple)):
            raise Unsupported("choice over an unmodelled sequence")
        if not seq:
            raise Unsupported("choice over an empty sequence")
        if not _is_sym(idx):
            return seq[max(0, min(int(idx), len(seq) - 1))]
        return _choice_ite(idx, seq)

    def _model_len(self, n):
        v = self.ev(n.args[0])
        if isinstance(v, SymList):
            return v.length
        if _is_sym(v) and v.sort() == z3.StringSort():
            return z3.Length(v)
        if isinstance(v, (list, tuple, str, dict)):
            return len(v)
        raise Unsupported("len")

    def _model_abs(self, n):
        v = _num(self.ev(n.args[0]))
        return z3.If(v >= 0, v, -v) if _is_sym(v) else abs(v)

    def _model_min(self, n):
        return self._model_extreme("min", n)

    def _model_max(self, n):
        return self._model_extreme("max", n)

    def _model_extreme(self, name, n):
        if len(n.args) == 2 and not n.keywords:
            a, b = _num(self.ev(n.args[0])), _num(self.ev(n.args[1]))
            if not (_is_sym(a) or _is_sym(b)):
                return min(a, b) if name == "min" else max(a, b)
            return z3.If(a <= b, a, b) if name == "min" else z3.If(a >= b, a, b)
        return self._extreme_of(name, n)

    def _model_any(self, n):
        return self._model_anyall("any", n)

    def _model_all(self, n):
        return self._model_anyall("all", n)

    def _model_anyall(self, name, n):
        self._short_circuit_ok(n.args[0])
        terms = [_truth(v) for v in self._iterate_expr(n.args[0])]
        if any(_is_sym(t) for t in terms):
            return z3.Or(*terms) if name == "any" else z3.And(*terms)
        return any(terms) if name == "any" else all(terms)

    #: the exact scalar classes ``int()`` / ``float()`` may be applied to NATIVELY: none runs a user hook (no ``__int__``,
    #: ``__float__`` or ``__index__`` of a user class is reachable), and a ``bool`` is an ``int``
    _SCALARS = (int, bool, float, str, bytes)

    def _native_scalar(self, qualname, v):
        """``qualname`` (``int`` or ``float``) applied by CPython itself (``Rule.native``) to the concrete exact scalar ``v``.  What
        the call would raise (``ValueError`` for an unparseable string, ``OverflowError`` for an infinity) is not a property
        class: the executor does not model the run past it, so the statement is refused."""
        if not _exact_type(v, self._SCALARS):
            raise Unsupported(f"{qualname}() of a symbolic or unmodelled argument")
        try:
            return get_rule(qualname).native(v)
        except (ValueError, OverflowError):
            raise Unsupported(f"{qualname}() raises on this argument") from None

    def _model_int(self, n):
        v = self.ev(n.args[0])
        if _is_sym(v) and v.sort() == z3.IntSort():
            return v
        if _is_sym(v) and z3.is_bool(v):
            return z3.If(v, 1, 0)
        if _is_sym(v):
            raise Unsupported("int() of a non-int")
        return self._native_scalar("int", v)

    def _model_float(self, n):
        v = self.ev(n.args[0])
        if _is_sym(v) and v.sort() == z3.RealSort():
            return v
        if _is_sym(v):
            raise Unsupported("float() of a symbolic non-float: IEEE rounding is not modelled")
        return self._native_scalar("float", v)

    def _model_divmod(self, n):
        a, b = self.ev(n.args[0]), self.ev(n.args[1])
        return (self.ev_BinOp(ast.BinOp(_Lit(a), ast.FloorDiv(), _Lit(b))),
                self.ev_BinOp(ast.BinOp(_Lit(a), ast.Mod(), _Lit(b))))

    def _model_pow(self, n):
        a, b = self.ev(n.args[0]), self.ev(n.args[1])
        return self.ev_BinOp(ast.BinOp(_Lit(a), ast.Pow(), _Lit(b)))

    def _model_print(self, n):
        """``print(*values, sep=, end=)``: nothing the executor tracks reads what it writes.  The values are evaluated (an
        argument the executor cannot model refuses the statement, as before); ``file=`` and ``flush=`` are refused, and so is a
        ``sep`` or ``end`` that is not ``None`` or a concrete ``str`` (CPython raises ``TypeError`` for another)."""
        for a in n.args:
            self.ev(a)
        for k in n.keywords:
            if k.arg not in ("sep", "end"):
                raise Unsupported(f"print({k.arg}=...)")
            v = self.ev(k.value)
            if v is not None and type(v) is not str:
                raise Unsupported(f"print({k.arg}=...) of a symbolic or non-str value")
        inexact = getattr(self, "inexact", None)
        if inexact is not None:
            inexact.append(f"line {n.lineno}: print: what it writes is not tracked")
        return None

    def _model_isinstance(self, n):
        return self._isinstance(self.ev(n.args[0]), self.ev(n.args[1]))

    def _model_sum(self, n):
        vals = self._iterate_expr(n.args[0])
        start = 0
        if len(n.args) == 2:
            start = self.ev(n.args[1])
        elif n.keywords:
            start = self.ev(n.keywords[0].value)
        total = _num(start)
        nums = [_num(v) for v in vals]
        if not (_is_sym(total) or any(_is_sym(v) for v in nums)):
            # every term is concrete: CPython's own ``sum``.  Since 3.12 it adds floats with Neumaier compensation
            # (``sum([0.1] * 10)`` is ``1.0``), which a left fold does not
            if not _exact_type(total, (int, float)) or any(not _exact_type(v, (int, float)) for v in nums):
                raise Unsupported("sum() of an unmodelled value")
            return get_rule("sum").native(nums, total)
        for v in nums:
            total = total + v
        return total

    def _model_range(self, n):
        args = [self.ev(a) for a in n.args]
        if any(_is_sym(a) for a in args):
            raise Unsupported("symbolic range")
        if not 1 <= len(args) <= 3 or any(not _exact_type(a, (int, bool)) for a in args):
            raise Unsupported("range() of a non-int or with a wrong arity: TypeError")
        if args[-1] == 0 and len(args) == 3:
            raise Unsupported("range() with a zero step: ValueError")
        out = range(*args)
        if len(out) > 4096:
            raise Unsupported("loop shape")
        return list(out)

    def _model_sorted(self, n):
        # [internal ref]: only a CONCRETE, comparable sequence -- never a symbolic sort
        # (this executor tracks no order relation over z3 terms), and dict
        # iterates its keys just like list()/for-in already do above.  ``reverse=`` is a
        # concrete truth value, ``key=None`` is no key (a key function is a call per element).
        reverse = False
        for k in n.keywords:
            kv = self.ev(k.value)
            if k.arg == "key":
                if kv is not None:
                    raise Unsupported("sorted() with a key function")
            elif _is_sym(kv) or isinstance(kv, (list, tuple, dict, SymList)):
                raise Unsupported("sorted() with a symbolic reverse")
            else:
                reverse = bool(kv)
        v = self._seq(self.ev(n.args[0]))
        if isinstance(v, dict):
            v = list(v)
        if not isinstance(v, (list, tuple)):
            raise Unsupported("sorted() of a non-sequence")
        if any(_is_sym(x) for x in v):
            raise Unsupported("sorted() of a symbolic sequence")
        try:
            return sorted(v, reverse=reverse)
        except TypeError:
            raise Unsupported("sorted() of an unorderable mix") from None

    def _short_circuit_ok(self, node):
        """``any`` / ``all`` stop at the first deciding element; this executor evaluates every element.  That is
        the same program unless an element runs code with effects: refused where the executor models effects
        (:class:`_Runner`); no effect is modelled here."""

    def _extreme_of(self, name, n):
        """``min(iterable)`` / ``max(iterable)`` over the unrolled elements (``default=`` and ``key=None`` are
        the keywords it takes): an If chain over the terms, a concrete fold when none is symbolic.  An empty
        iterable is the default, else a ValueError the executor does not model."""
        default, has_default = None, False
        for k in n.keywords:
            if k.arg == "default":
                default, has_default = self.ev(k.value), True
            elif k.arg == "key" and self.ev(k.value) is None:
                continue
            else:
                raise Unsupported(f"{name}() with the keyword {k.arg}")
        vals = [_num(v) for v in self._iterate_expr(n.args[0])]
        if not vals:
            if has_default:
                return default
            raise Unsupported(f"{name}() of an empty iterable")
        best = vals[0]
        for v in vals[1:]:
            if _is_sym(best) or _is_sym(v):
                best = z3.If(v < best, v, best) if name == "min" else z3.If(v > best, v, best)
            elif (v < best) if name == "min" else (v > best):
                best = v
        return best

    def _kind_of(self, v):
        """The Python class this executor's value ``v`` stands for (a z3 sort is not one)."""
        if isinstance(v, (list, SymList)):
            return list
        if isinstance(v, SymRange):
            return range
        if _is_sym(v):
            return bool if z3.is_bool(v) else int if v.sort() == z3.IntSort() else str if v.sort() == z3.StringSort() else float
        return type(v)

    def _isinstance(self, v, types):
        types = types if isinstance(types, tuple) else (types,)
        kind = self._kind_of(v)
        return any(isinstance(t, type) and issubclass(kind, t) for t in types)

    #: per builtin, the EXACT classes each positional argument must have for the native fold
    #: (``adapters/concolic.py::concrete_fold``). Exact, never a subclass: a z3 term, a
    #: ``SymList`` and any user object are therefore refused, and no argument carries a user
    #: hook. A container's ELEMENTS may be anything (z3 terms included): none of these
    #: builtins looks at one.
    _SEQUENCES = (list, tuple, str, dict)
    _FOLD_ARGS = {
        "ord": ((str,),), "chr": ((int, bool),), "tuple": (_SEQUENCES,), "list": (_SEQUENCES,),
        "enumerate": (_SEQUENCES, (int, bool)), "reversed": (_SEQUENCES,),
    }
    _LAZY_FOLDS = frozenset({"zip", "enumerate", "reversed"})

    def _visit_rule(self, qualname, ctx):
        rule = get_rule(qualname)
        if rule is None:
            return _NO_CONTRACT
        result = rule.accept(SymexecVisitor(self), ctx)
        # Keep the call's established named Unsupported diagnostic on refusal.
        return _NO_CONTRACT if isinstance(result, Refused) else result

    def _contract_fold(self, name, n):
        return self._visit_rule(name, CallCtx(n))

    def _contract_fold_model(self, name, n):
        """[internal ref] batch 4: a builtin call the contracts say is pure, evaluated natively on
        concrete builtin values (``adapters/concolic.py``) instead of ``Unsupported``; and the
        one-argument ``type(x)`` mapped through :meth:`_kind_of`. ``_NO_CONTRACT`` when nothing
        applies, so the caller raises the same ``Unsupported`` it always did."""
        from tharkun.solve.contracts.adapters import concolic as _contracts

        count, kw = len(n.args), bool(n.keywords)
        if name == "type":
            if not _contracts.type_of_one_argument(arg_count=count, has_keywords=kw):
                return _NO_CONTRACT
            v = self.ev(n.args[0])
            if isinstance(v, (Union, _Poison, _LazyIter)) or not (
                    _is_sym(v) or isinstance(v, (int, float, bool, str, list, tuple, dict, SymList, type(None)))):
                raise Unsupported("type() of an unmodelled value")
            return self._kind_of(v)
        fold = _contracts.concrete_fold(name, arg_count=count, has_keywords=kw)
        if fold is None:
            return _NO_CONTRACT
        arg = n.args[0] if len(n.args) == 1 else None
        if name in ("list", "tuple") and isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and arg.func.attr == "keys" \
                and not arg.args and not arg.keywords:
            # An immediately consumed exact-dict view cannot escape the snapshot.
            receiver = self.ev(arg.func.value)
            if type(receiver) is not dict:
                raise Unsupported("keys snapshot of a non-exact dict")
            args = [self._through_rule("dict.keys", arg, args=(receiver,))]
        else:
            args = [self._seq(self.ev(a)) for a in n.args]
        wanted = self._FOLD_ARGS.get(name)
        for i, a in enumerate(args):
            classes = wanted[i] if wanted is not None else self._SEQUENCES
            if not _exact_type(a, classes):
                raise Unsupported(f"{name}() of a symbolic or unmodelled argument")
        try:
            out = fold(*args)
        except (TypeError, ValueError):
            raise Unsupported(f"{name}() raises on these arguments") from None
        return _LazyIter(list(out)) if name in self._LAZY_FOLDS else out

    def _iterate_expr(self, n):
        """The values of a generator / list comprehension or a list, unrolled."""
        if isinstance(n, (ast.GeneratorExp, ast.ListComp)):
            if len(n.generators) != 1 or n.generators[0].ifs or n.generators[0].is_async:
                raise Unsupported("comprehension shape")
            gen = n.generators[0]
            if not isinstance(gen.target, ast.Name):
                raise Unsupported("comprehension target")
            it = self._seq(self.ev(gen.iter))
            if not isinstance(it, (list, tuple)):
                raise Unsupported("comprehension over a symbolic iterable")
            saved = self.env.get(gen.target.id, _MISSING)
            out = []
            for x in it:
                self.env[gen.target.id] = x
                out.append(self.ev(n.elt))
            if saved is _MISSING:
                self.env.pop(gen.target.id, None)
            else:
                self.env[gen.target.id] = saved
            return out
        v = self._seq(self.ev(n))
        if isinstance(v, (list, tuple)):
            return list(v)
        raise Unsupported("iterate")

    def ev_ListComp(self, n):
        gen = n.generators[0] if len(n.generators) == 1 else None
        if gen is not None and not gen.ifs and isinstance(gen.iter, ast.Call) \
                and isinstance(gen.iter.func, ast.Name) and gen.iter.func.id == "range" \
                and len(gen.iter.args) == 1:
            bound = self.ev(gen.iter.args[0])
            if _is_sym(bound):
                return SymList(z3.If(_num(bound) > 0, _num(bound), 0))
        return self._iterate_expr(n)

    ev_GeneratorExp = ev_ListComp

    # --- statements ---------------------------------------------------------
    def run(self, stmts):
        for st in stmts:
            self.st(st)

    def st(self, st):
        if isinstance(st, ast.Assign):
            v = self.ev(st.value)
            for t in st.targets:
                self.assign(t, v)
        elif isinstance(st, ast.AugAssign):
            target = st.target
            if isinstance(target, ast.Subscript):
                # ``xs[i] += v`` evaluates ``i`` ONCE (the load and the store use the same index); an index that draws
                # (``xs[draw()] += 1``) would otherwise load one fresh draw's cell and store into another's ([internal ref])
                target = ast.copy_location(ast.Subscript(target.value, _Lit(self.ev(target.slice)), ast.Store()), target)
            load = copy.copy(target)
            load.ctx = ast.Load()
            cur = self.ev(load)
            new = self.ev_BinOp(ast.BinOp(_Lit(cur), st.op, st.value))
            self.assign(target, new)
        elif isinstance(st, ast.If):
            t = _truth(self.ev(st.test))
            if not _is_sym(t):
                self.run(st.body if t else st.orelse)
                return
            before = dict(self.env)
            self.run(st.body)
            then_env = self.env
            self.env = dict(before)
            self.run(st.orelse)
            else_env = self.env
            merged = {}
            for k in set(then_env) | set(else_env):
                if k in then_env and k in else_env:
                    a, b = then_env[k], else_env[k]
                    merged[k] = a if a is b else _ite(t, a, b)
                else:
                    raise Unsupported(f"{k} defined on one arm only")
            self.env = merged
        elif isinstance(st, ast.For):
            it = self._seq(self.ev(st.iter))
            if not isinstance(it, (list, tuple)) or st.orelse:
                raise Unsupported("loop over a symbolic iterable")
            if len(it) > 4096:
                raise Unsupported("loop too long to unroll")
            for x in it:
                self.assign(st.target, x)
                for inner in st.body:
                    if isinstance(inner, (ast.Break, ast.Continue, ast.Return)):
                        raise Unsupported("loop control")
                    self.st(inner)
        elif isinstance(st, (ast.Pass, ast.Global)):
            pass
        elif isinstance(st, ast.Expr):
            pass                        # kept only for its (sliced-out) effects
        else:
            raise Unsupported(type(st).__name__)

    def assign(self, t, v):
        if isinstance(t, ast.Name):
            # the object itself, as CPython binds it: a list reachable by two names is DETECTED (``_mutable``) and its mutation refused,
            # where a copy here used to hide the second name (``ys = xs; xs.append(3)`` left ``ys`` unchanged)
            self.env[t.id] = v
        elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name):
            self._through_rule("list.__setitem__", t, args=(v,))
        elif isinstance(t, ast.Tuple) and isinstance(v, tuple) and len(t.elts) == len(v):
            for e, x in zip(t.elts, v):
                self.assign(e, x)
        else:
            raise Unsupported("assignment target")


_MISSING = object()
_NATIVE_ATTRS = frozenset(q.split(".", 1)[1] for q in NATIVE_METHODS)
_NATIVE_LIMIT = 4096                    # the most items a native iterator of ``NATIVE_MODULE_FUNCTIONS`` is materialized to


class _Lit(ast.expr):
    """An already-evaluated value inside an expression tree."""
    _fields = ()

    def __init__(self, value):
        super().__init__()
        self.value = value


def _ev_lit(self, n):
    return n.value


_Exec.ev__Lit = _ev_lit


def _int_term(a):
    if _is_sym(a) and a.sort() != z3.IntSort():
        raise Unsupported("integer operation on a non-integer term")


def _py_floordiv_sym(a, b):
    """``a // b`` for a symbolic, nonzero divisor, with Python's floor semantics over z3's Euclidean ``/``.

    Euclidean division by a POSITIVE divisor is floor division, and ``floor(a / b) == floor(-a / -b)``, so a negative
    divisor is flipped to a positive one: ``If(b > 0, a / b, (-a) / (-b))``.  This equals the earlier ``(a - r) / b`` with
    ``r`` the Python remainder (an ``If`` over ``a % b`` with a divisor of either sign), but z3 never divides by a term of
    unknown sign nor reasons about ``a % b`` and ``/ b`` together: ``-(a // -b) * b >= a`` over ``0 <= a, b <= 1024`` took
    z3 about 8 s that way and 2 ms this way.  The same form as ``solve/terms_z3.py``."""
    return z3.If(b > 0, a / b, (-a) / (-b))


def _py_mod_sym(a, b):
    """``a % b`` for a symbolic, nonzero divisor, with Python's sign-of-divisor semantics: ``a - b * (a // b)``, the
    definition, over the SAME quotient term :func:`_py_floordiv_sym` builds (z3 shares it), so ``(a // b) * b + a % b == a``
    is structural.  Measured: giving ``%`` its own positive-divisor form, ``If(b > 0, a % b, -((-a) % (-b)))``, made z3 give
    up on that identity (10 s timeout; 0.05 s with the earlier encoding, 0.0 s with this one)."""
    return a - b * _py_floordiv_sym(a, b)


def _floordiv(a, k):
    """``a // k`` for a positive constant ``k``: z3's integer division is Euclidean,
    which equals Python's floor division when the divisor is positive."""
    if not _is_sym(a):
        return a // k
    _int_term(a)
    return a / k


# ---------------------------------------------------------------------------
# slicing
# ---------------------------------------------------------------------------

def _stores(st) -> set:
    out = set()
    for n in ast.walk(st):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, ast.Subscript) and isinstance(n.ctx, ast.Store) and isinstance(n.value, ast.Name):
            out.add(n.value.id)
        elif (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and isinstance(n.func.value, ast.Name)):
            out.add(n.func.value.id)     # a method call may mutate its receiver
    return out


def _loads(node) -> set:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def _has_draw(st) -> bool:
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id in ("__sv_draw", "__sv_next") for n in ast.walk(st))


# One implementation for obligation proofs and search (legacy import preserved).
from tharkun.solve.slicing import slice_for


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def _path_to(body, line, path=()):
    """``(the statement list holding the marker, its index, enclosing (If, arm) pairs)``."""
    for i, st in enumerate(body):
        if isinstance(st, ast.If) and st.body and isinstance(st.body[0], ast.Expr) \
                and isinstance(st.body[0].value, ast.Call) and getattr(st.body[0].value.func, "id", "") == "__sv_site" \
                and st.body[0].lineno == line:
            return body, i, path
        if isinstance(st, ast.If):
            for arm, stmts in (("then", st.body), ("else", st.orelse)):
                r = _path_to(stmts, line, path + ((st, arm, body, i),))
                if r:
                    return r
    return None


def solve_site(harness_text: str, module_globals: dict, fn_name: str, line: int, params: list,
               kinds: list, stream_lens: dict, timeout_ms: int = 10000) -> dict | None:
    """Inputs of ``fn_name`` (the harness) that make the marker at ``line`` fire, or
    ``None`` (unsat, unknown, or :class:`Unsupported` -- the message says which)."""
    tree = ast.parse(harness_text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    found = _path_to(fn.body, line)
    if found is None:
        raise Unsupported("site not at statement level of the harness")
    stmts, idx, path = found
    marker = stmts[idx]
    # the statements that run before the marker: the prefix of each enclosing block
    prefix = []
    for (anc, arm, parent, pidx) in path:
        prefix.extend(parent[:pidx])
    prefix.extend(stmts[:idx])
    need = _loads(marker.test)
    for anc, _arm, _p, _i in path:
        need |= _loads(anc.test)
    kept = slice_for(prefix, need)

    env, streams = {}, {}
    for p, k in zip(params, kinds):
        if k.startswith("list:"):
            base = k[5:]
            n = stream_lens.get(p, 0)
            streams[p] = [_var(f"{p}_{i}", base) for i in range(n)]
            env[p] = list(streams[p])
        else:
            env[p] = _var(p, k)
    ex = _Exec(module_globals, env, streams)
    # run the kept prefix, re-entering each enclosing if on the site's arm
    kept_ids = {id(s) for s in kept}
    conds = []

    def run_block(body, upto=None):
        for st in body[: upto if upto is not None else len(body)]:
            if id(st) in kept_ids:
                ex.st(st)

    for (anc, arm, parent, pidx) in path:
        run_block(parent, pidx)
        t = _truth(ex.ev(anc.test))
        conds.append(t if arm == "then" else (z3.Not(t) if _is_sym(t) else (not t)))
    run_block(stmts, idx)
    goal = _truth(ex.ev(marker.test))
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    for c in conds + [goal]:
        if _is_sym(c):
            s.add(c)
        elif not c:
            return None
    for p, k in zip(params, kinds):
        if k == "float" and p in env:
            pass
    if s.check() != z3.sat:
        return None
    m = s.model()
    out = {}
    for p, k in zip(params, kinds):
        if k.startswith("list:"):
            out[p] = [_value(m, v, k[5:]) for v in streams[p]]
        else:
            out[p] = _value(m, env[p], k)
    return out


def _var(name, kind):
    if kind in ("int", "rindex"):        # rindex ([internal ref]): an unseeded random.choice
        return z3.Int(name)              # index -- an int like any other lifted draw
    if kind == "bool":
        return z3.Bool(name)
    if kind == "float":
        return z3.Real(name)
    if kind == "str":
        return z3.String(name)
    raise Unsupported(f"draw kind {kind}")


def _value(m, v, kind):
    r = m.eval(v, model_completion=True)
    if kind == "bool":
        return z3.is_true(r)
    if kind in ("int", "rindex"):
        return r.as_long()
    if kind == "str":
        return r.as_string()
    if kind == "float":
        f = r.as_fraction()
        return float(f.numerator) / float(f.denominator)
    raise Unsupported(kind)


# ===========================================================================
# slice 3: run to a target anywhere -- callees, symbolic loop bounds, typing
# ===========================================================================

#: iterations unrolled for a loop whose trip count depends on a draw; the model is
#: constrained to runs that leave the loop within this bound (replay checks the rest)
LOOP_BOUND = 64
#: draws a stream serves in the symbolic model (a run that reads more is not modelled)
STREAM_BOUND = 512
_POS = "\0pos:"
#: calls inlined deep
CALL_DEPTH = 12


class _Dead(Exception):
    """The current path raised (or returned) before reaching what follows."""


class _Poison:
    """A value the executor could not model (its defining statement was unsupported)."""


POISON = _Poison()


class Union:
    """A value whose Python TYPE depends on the path: ``[(guard, value), ...]`` with
    disjoint guards (``v = 0`` merged with ``v = "tag"``).  Only what the typing
    sites need is modelled: which members an operation accepts."""

    def __init__(self, members):
        self.members = members


def _pytype(v):
    if isinstance(v, (list, SymList)):
        return list
    if _is_sym(v):
        return bool if z3.is_bool(v) else int if v.sort() == z3.IntSort() else str if v.sort() == z3.StringSort() else float
    return type(v)


_REPR = {int: 1, float: 1.0, bool: True, str: "a", bytes: b"a", list: [1], tuple: (1,),
         type(None): None, dict: {1: 1}}


def _op_ok(op, ta, tb) -> bool:
    """Does ``a <op> b`` accept operands of types ``ta`` and ``tb`` (no TypeError)?"""
    if ta not in _REPR or tb not in _REPR:
        return True                      # unknown: do not claim a type error
    try:
        _PYOP[op](_REPR[ta], _REPR[tb])
    except TypeError:
        return False
    except Exception:  # noqa: BLE001 -- other errors are not typing errors
        return True
    return True


def _members(v, guard=True):
    if isinstance(v, Union):
        return [(z3.And(guard, g) if guard is not True else g, m) for g, m in v.members]
    return [(guard, v)]


def _typed_ite(c, a, b):
    """``_ite`` that keeps values of different types apart as a :class:`Union`."""
    if isinstance(a, Ref) or isinstance(b, Ref):
        return _same_object(a, b)
    if isinstance(a, Union) or isinstance(b, Union) or \
            _pytype(a) not in (_pytype(b),) and not ({_pytype(a), _pytype(b)} <= {int, bool, float}):
        return Union(_members(a, c) + _members(b, z3.Not(c)))
    return _ite(c, a, b)


def _conj(parts):
    parts = [p for p in parts if p is not True]
    if any(p is False for p in parts):
        return False
    if not parts:
        return True
    return parts[0] if len(parts) == 1 else z3.And(*parts)


class _Runner(_Exec):
    """Executes the WHOLE harness symbolically once and records, for every target site,
    the condition under which it fires on each path that reaches it.

    The path condition ``pc`` carries everything a later site depends on: the guards of
    the enclosing branches, the arms that did not die (an arm that raises or returns
    ends its path), asserts that passed, operand types that did not raise, and the bound
    of every loop unrolled to :data:`LOOP_BOUND`.  A statement the executor cannot model
    is skipped and the names it writes are POISONED (reading one later is
    :class:`Unsupported`); the sites that statement could reach are reported
    ``unsupported`` instead of solved."""

    def __init__(self, tree, globals_, env, streams, targets, call_targets=None, proven_classes=(),
                 fresh_draws=None):
        super().__init__(globals_, env, streams)
        #: ``{id(call node): kind}`` of the ``_sv_verifier.nondet_*()`` calls inside the program's own functions
        #: that a proof lets draw (``product_proof.py``): EACH EXECUTION of such a call is a fresh unconstrained
        #: ``kind`` term, which is what a draw is (the child's replayed vector hands every execution its own value).
        #: Empty for the search runs, where a draw is a harness parameter (``__sv_draw``) and a call left in a
        #: callee stays unmodelled.
        self.fresh_draws = fresh_draws or {}
        self.draw_count = 0
        self.defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        self.targets = set(targets)      # harness lines of the sites to solve for
        self.goals = {line: [] for line in self.targets}
        self.unsupported: dict = {}      # target line -> why its statement was not modelled
        self.poisoned = []               # every statement skipped as unmodelled (reasons)
        self.skipped = []                # EVERY statement whose run() hit Unsupported ([internal ref])
        #: statements this executor ANSWERED but whose effect it does not track (``print``: what it writes): a search does not mind, a
        #: proof (``run_exact``) does -- the run is not exact
        self.inexact = []
        self.bounded = False             # a loop was cut at LOOP_BOUND
        self.stream_use: dict = {}       # stream -> reads evaluated (bounds the positions read)
        self.frames = []                 # caller environments while a call is inlined
        self.depth = 0
        self.scope_stack = ["<main>"]    # the qualified name of the code running: the key of the call_target fact
        self.ctx_stack = [None]          # per inlined call: ``(defining class, kind, first parameter)`` of a method
        self._pre = {}                   # receivers evaluated once for the object-call check, see ev_Call
        self.reach = _targets_by_function(self.defs, self.targets)
        #: user-class support ([internal ref], ``_symexec_objects.py``): on only when the call_target records were handed
        #: in -- without them (``run_exact``, the proof) a call of a class stays unmodelled, as before
        self.objects = None
        if call_targets is not None:
            from tharkun.sv import _symexec_objects

            self.objects = _symexec_objects.ObjectModel(self, tree, globals_, call_targets, proven_classes)
            self.reach = self.objects.build_reach(self.defs, self.targets)

    # --- helpers ------------------------------------------------------------
    def _fresh_draw(self, kind):
        """One EXECUTION of a draw call inside a function: a new unconstrained term of ``kind``.  Its name holds a
        ``#``, which no Python identifier does: a program's own draw (``draw1 = nondet_int()``, bound to a term
        of that name) must never be the same term."""
        self.draw_count += 1
        return _var(f"fresh draw #{self.draw_count}", kind)

    def _path(self, *more):
        return _conj(list(self.pc) + list(more))

    def _holds(self, cond):
        """The rest of the current path runs only when ``cond`` holds."""
        c = _truth(cond)
        if _is_sym(c):
            self.pc.append(c)
        elif not c:
            raise _Dead()

    def _targets_in(self, st) -> set:
        lo, hi = st.lineno, (st.end_lineno or st.lineno)
        out = {line for line in self.targets if lo <= line <= hi}
        for n in ast.walk(st):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.reach:
                out |= self.reach[n.func.id]
            if self.objects is not None and isinstance(n, ast.Call):
                out |= self.objects.reach_of_call(n)
        return out

    def ev(self, n):
        if self._pre:
            hit = self._pre.get((id(n), self.depth), _MISSING)
            if hit is not _MISSING:
                return hit
        return super().ev(n)

    def ev_Name(self, n):
        if n.id not in self.env and self.frames and n.id in self.frames[0]:
            v = self.frames[0][n.id]     # a callee reading a global the main block assigned
        else:
            v = super().ev_Name(n)
        if v is POISON:
            raise Unsupported(f"{n.id} was not modelled")
        if self.objects is not None and isinstance(v, type) and n.id not in self.env \
                and not (self.frames and n.id in self.frames[0]):
            self.objects.check_class_name(n.id, v)      # a class read from the module's namespace, not a local
        return v

    # --- typing ---------------------------------------------------------------
    def ev_BinOp(self, n):
        a, b = self.ev(n.left), self.ev(n.right)
        if isinstance(a, Union) or isinstance(b, Union):
            out, bad = [], []
            for ga, ma in _members(a):
                for gb, mb in _members(b):
                    g = _conj([ga, gb])
                    if _op_ok(type(n.op), _pytype(ma), _pytype(mb)):
                        out.append((g, self._binop(n.op, ma, mb)))
                    else:
                        bad.append(g)
            if bad:                     # those members raise TypeError here: the path ends
                self._holds(z3.Not(z3.Or(*[_bool(g) for g in bad])))
            if not out:
                raise _Dead()
            if len(out) == 1:
                return out[0][1]
            return Union(out)
        return self._binop(n.op, a, b)

    def _binop(self, op, a, b):
        if isinstance(op, (ast.Mod, ast.FloorDiv)) and _is_sym(b) and not isinstance(a, (str, list)):
            # Python's floor semantics over z3's Euclidean ones; a zero divisor ends the path
            a, b = _num(a), _num(b)
            _int_term(a)
            _int_term(b)
            self._holds(b != 0)
            return _py_floordiv_sym(a, b) if isinstance(op, ast.FloorDiv) else _py_mod_sym(a, b)
        return _Exec.ev_BinOp(self, ast.BinOp(_Lit(a), op, _Lit(b)))

    def _ok_binop(self, opname, a, b):
        op = {"add": ast.Add, "sub": ast.Sub, "mul": ast.Mult, "truediv": ast.Div,
              "floordiv": ast.FloorDiv, "mod": ast.Mod, "pow": ast.Pow}[opname]
        ok = []
        for ga, ma in _members(a):
            for gb, mb in _members(b):
                if _op_ok(op, _pytype(ma), _pytype(mb)):
                    ok.append(_conj([ga, gb]))
        if not ok:
            return False
        if any(g is True for g in ok):
            return True
        return z3.Or(*[_bool(g) for g in ok])

    def _isinstance(self, v, types):
        if isinstance(v, Ref):
            return self.objects.isinstance_ref(v, types)
        if isinstance(v, Union):
            gs = [g for g, m in v.members if super()._isinstance(m, types)]
            return z3.Or(*[_bool(g) for g in gs]) if gs else False
        return super()._isinstance(v, types)

    def ev_Compare(self, n):
        if len(n.ops) == 1 and isinstance(n.ops[0], (ast.Is, ast.IsNot)) \
                and isinstance(n.comparators[0], ast.Constant) and n.comparators[0].value is None:
            v = self.ev(n.left)
            if isinstance(v, Union):
                gs = [g for g, m in v.members if m is None]
                g = z3.Or(*[_bool(x) for x in gs]) if gs else False
            else:
                g = v is None
            if isinstance(n.ops[0], ast.IsNot):
                return z3.Not(g) if _is_sym(g) else not g
            return g
        return super().ev_Compare(n)

    def ev_BoolOp(self, n):
        """Python's ``and`` / ``or`` with OPERAND semantics ([internal ref]): the value is the
        operand that decided the result, not its truth -- ``x or 7`` is ``x`` when ``x``
        is truthy, ``a and b`` is ``a`` when ``a`` is falsy -- and a later operand is
        evaluated only on the paths where the earlier ones let it run (so
        ``isinstance(c, dict) and k in c`` never evaluates ``k in c`` for a list).
        Folded right to left over the operands: ``v0 or REST`` is ``ite(truth(v0), v0,
        REST)``, ``v0 and REST`` is ``ite(truth(v0), REST, v0)``."""
        is_and = isinstance(n.op, ast.And)

        def go(i):
            v = self.ev(n.values[i])
            if i == len(n.values) - 1:
                return v
            t = _truth(v)
            if not _is_sym(t):
                return go(i + 1) if bool(t) == is_and else v
            nxt = self._guarded(t if is_and else z3.Not(t), lambda: go(i + 1))
            return _typed_ite(t, nxt, v) if is_and else _typed_ite(t, v, nxt)

        return go(0)

    def _guarded(self, guard, thunk):
        """``thunk()``, evaluated only on the paths where ``guard`` holds: what it adds to
        the path condition holds under ``guard`` alone."""
        mark = len(self.pc)
        base_heap = None
        if self.objects is not None:
            base_heap = self.objects.heap
            self.objects.heap = base_heap.fork()
        try:
            v = thunk()
        except _Dead:
            raise Unsupported("an operand that raises on some paths only") from None
        new = self.pc[mark:]
        del self.pc[mark:]
        if new:
            self.pc.append(z3.Implies(_bool(guard), _bool(_conj(new))))
        if base_heap is not None:
            # the effects of the operand hold where its guard does, and nowhere else
            self.objects.heap = self.objects.merge(_bool(guard), self.objects.heap, base_heap)
        return v

    def ev_IfExp(self, n):
        t = _truth(self.ev(n.test))
        if not _is_sym(t):
            return self.ev(n.body) if t else self.ev(n.orelse)
        a = self._guarded(t, lambda: self.ev(n.body))
        b = self._guarded(z3.Not(t), lambda: self.ev(n.orelse))
        return _typed_ite(t, a, b)

    # --- containers ------------------------------------------------------------
    def ev_Subscript(self, n):
        c = self.ev(n.value)
        if isinstance(n.slice, ast.Slice):
            if type(c) is str:
                return super().ev_Subscript(ast.Subscript(_Lit(c),n.slice,ast.Load()))
            raise Unsupported("slice")
        k = self.ev(n.slice)
        qualname = next((f"{cls.__name__}.__getitem__" for cls in (dict, list, tuple) if type(c) is cls), None)
        if qualname is not None:
            return self._through_rule(qualname, n, args=(c, k))
        return self._subscript_value(n, c, k)

    def _model_dict___getitem__(self, n, c, k):
        return self._subscript_value(n, c, k)

    _model_list___getitem__ = _model_tuple___getitem__ = _model_dict___getitem__

    def _subscript_value(self, n, c, k):
        """``c[k]`` for the operands already evaluated: the model of ``dict.__getitem__``, ``list.__getitem__`` and
        ``tuple.__getitem__`` (a ``KeyError`` or ``IndexError`` ends the path), and the rest of the subscripts."""
        if type(c) is dict:
            if not _is_sym(k):
                if k not in c:
                    raise _Dead()        # KeyError: the run ends here (its site came before)
                return c[k]
            keys = [x for x in c if isinstance(x, (int, bool))]
            if not keys:
                raise _Dead()
            self._holds(z3.Or(*[_num(k) == x for x in keys]))
            out = c[keys[-1]]
            for x in reversed(keys[:-1]):
                out = _ite(_num(k) == x, c[x], out)
            return out
        if _exact_type(c, (list, tuple)) and _is_sym(k):
            k = _num(k)
            if not z3.is_int(k):
                raise Unsupported("a sequence index that is not an int")
            m = len(c)
            if not m:
                raise _Dead()
            self._holds(z3.And(k >= -m, k < m))    # else IndexError ends the run
            out = c[-1]
            for i in range(m - 2, -1, -1):
                out = _ite(z3.Or(k == i, k == i - m), c[i], out)
            return out
        if _exact_type(c, (list, tuple, str)) and _exact_type(k, (int, bool)) and not -len(c) <= k < len(c):
            raise _Dead()
        return super().ev_Subscript(ast.Subscript(_Lit(c), _Lit(k), ast.Load()))

    def _cmp(self, op, a, b):
        if isinstance(op, (ast.In, ast.NotIn)) and isinstance(b, (dict, list, tuple)):
            items = list(b.keys()) if isinstance(b, dict) else list(b)
            if _is_sym(a) or any(_is_sym(x) for x in items):
                r = z3.Or(*[_num(a) == _num(x) for x in items]) if items else False
            else:
                r = a in items
            return (z3.Not(r) if _is_sym(r) else not r) if isinstance(op, ast.NotIn) else r
        return super()._cmp(op, a, b)

    def assign(self, t, v):
        if isinstance(t, ast.Attribute) and self.objects is not None:
            if any(isinstance(x, ast.Call) for x in ast.walk(t.value)):
                raise Unsupported("an attribute store through a call expression")
            self.objects.store(self.ev(t.value), t.attr, v)
            return
        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) \
                and isinstance(self.env.get(t.value.id), dict):
            self._through_rule("dict.__setitem__", t, args=(v,))
            return
        super().assign(t, v)

    def _model_dict___setitem__(self, t, v):
        k = self.ev(t.slice)
        if _is_sym(k):
            raise Unsupported("store under a symbolic dict key")
        d = dict(self._mutable(t.value.id, (dict,)))
        d[k] = v
        self.env[t.value.id] = d

    def ev_Attribute(self, n):
        """``obj.attr`` and ``Cls.attr`` through the object model (``math.pi`` and the like stay with the base)."""
        if self.objects is None:
            return super().ev_Attribute(n)
        base = n.value
        if isinstance(base, ast.Name) and self.objects._is_program_name(base.id) \
                and not self.objects.table.is_user_class(self.g.get(base.id)):
            return super().ev_Attribute(n)
        return self.objects.load(self.ev(base), n.attr)

    @staticmethod
    def _concrete(v, depth=0) -> bool:
        """A value CPython can compare, hash and print itself: an exact int, bool, float, str, bytes or None, or a tuple of them."""
        if _exact_type(v, (int, bool, float, str, bytes, type(None))):
            return True
        return type(v) is tuple and depth < 8 and all(_Runner._concrete(x, depth + 1) for x in v)

    def _native_method(self, qualname, n, recv):
        """A method of a builtin container run by CPython itself (``Rule.native``) on concrete operands ([internal ref] step 2).  A method that
        mutates runs on a COPY of the receiver, which then replaces the name's binding (``_mutable``: the name is this frame's and
        the object is held nowhere else); one that compares elements (``count``, ``index``, ``remove``, ``sort``) needs every element and
        argument concrete.  What CPython raises as ``IndexError``, ``KeyError`` or ``ValueError`` ends the path, as an uncaught raise
        does; any other exception, a symbolic index or key, a key function and a keyword but ``reverse`` refuse."""
        if qualname in ("dict.items", "dict.values"):
            raise Unsupported("live dict views require alias tracking")
        if type(recv) is not _TYPES[qualname.split(".")[0]]:
            raise Unsupported(f"{qualname}: a non-exact receiver")
        mutates, compares = NATIVE_METHODS[qualname]
        args = [self.ev(a) for a in n.args]
        kwargs = {}
        for k in n.keywords:
            if k.arg is None:
                raise Unsupported(f"{qualname}: keyword expansion")
            kwargs[k.arg] = self.ev(k.value)
        if qualname == "list.sort":
            if kwargs.pop("key", None) is not None:
                raise Unsupported("list.sort with a key function")
            if set(kwargs) - {"reverse"} or ("reverse" in kwargs and not _exact_type(kwargs["reverse"], (bool, int))):
                raise Unsupported("list.sort: a keyword other than a concrete reverse")
        elif kwargs:
            raise Unsupported(f"{qualname} with keywords")
        elements = list(recv) if _exact_type(recv, (list, tuple)) else []
        if compares and not all(self._concrete(x) for x in (*elements, *args)):
            raise Unsupported(f"{qualname} compares elements: a symbolic or unmodelled operand")
        owner = qualname.split(".")[0]
        if qualname in ("list.insert", "list.pop") and args and not _exact_type(args[0], (int, bool)):
            raise Unsupported(f"{qualname}: a symbolic or non-int index")
        if qualname in ("dict.pop", "dict.setdefault") and (not args or not self._concrete(args[0])):
            raise Unsupported(f"{qualname}: a symbolic key")
        if qualname == "dict.update" and not (len(args) == 1 and type(args[0]) is dict):
            raise Unsupported("dict.update: only an exact dict is modelled")
        if qualname == "list.extend" and not (len(args) == 1 and (_exact_type(args[0], (list, tuple, dict)) or
                                                                  type(args[0]) is str)):
            raise Unsupported("list.extend: only an exact list, tuple, dict or concrete str is modelled")
        if qualname == "list.index" and any(not _exact_type(a, (int, bool)) for a in args[1:]):
            raise Unsupported("list.index: a symbolic bound")
        if qualname == "tuple.index" and any(not _exact_type(a, (int, bool)) for a in args[1:]):
            raise Unsupported("tuple.index: a symbolic bound")
        target = list(recv) if owner == "list" else dict(recv) if owner == "dict" else recv
        rule = get_rule(qualname)
        try:
            out = rule.native(target, *args, **kwargs)
        except (IndexError, KeyError, ValueError):
            raise _Dead() from None
        except Exception as e:  # noqa: BLE001 -- anything else is a raise this executor does not model
            raise Unsupported(f"{qualname} raises {type(e).__name__} on these operands") from None
        if mutates:
            name = n.func.value.id
            self._mutable(name, (type(recv),))
            self.env[name] = target
        return out

    def _native_module_function(self, qualname, n):
        """A function of a standard-library module run by CPython itself (``Rule.native``) on concrete operands ([internal ref] step 2): the
        arguments of ``NATIVE_MODULE_FUNCTIONS``' functions are exact lists, tuples, strs, dicts, ints, or concrete scalars; a function
        that mutates its first argument runs on a COPY and rebinds the name (``_mutable``); one that compares elements needs every
        element concrete; an iterator comes back as a one-shot ``_LazyIter`` and a result of more than ``_NATIVE_LIMIT`` items is
        refused.  ``IndexError`` / ``KeyError`` / ``ValueError`` end the path, any other exception refuses."""
        import itertools

        mutates, compares = NATIVE_MODULE_FUNCTIONS[qualname]
        args = [self.ev(a) for a in n.args]
        kwargs = {}
        for k in n.keywords:
            if k.arg is None:
                raise Unsupported(f"{qualname}: keyword expansion")
            kwargs[k.arg] = self.ev(k.value)
        containers = (list, tuple, str, dict)
        if any(not (_exact_type(a, containers) or self._concrete(a) or
                    (qualname == "copy.copy" and _is_sym(a))) for a in args):
            raise Unsupported(f"{qualname}: a symbolic or unmodelled operand")
        elems = [x for a in args if _exact_type(a, (list, tuple)) for x in a]
        if compares and not all(self._concrete(x) for x in (*elems, *[a for a in args if not _exact_type(a, (list, tuple, dict))])):
            raise Unsupported(f"{qualname} compares elements: a symbolic or unmodelled operand")
        if qualname == "copy.copy":
            if len(args) != 1 or kwargs:
                raise Unsupported("copy.copy: an unmodelled call shape")
            v = args[0]
            if _is_sym(v):
                return v                     # an immutable value (a number, a string, a bool): copy.copy returns it
            if not _exact_type(v, (list, dict, tuple)) and not self._concrete(v):
                raise Unsupported("copy.copy of an unmodelled value")
        elif qualname == "copy.deepcopy":
            if len(args) != 1 or kwargs or not self._deep_concrete(args[0]):
                raise Unsupported("copy.deepcopy of a symbolic or unmodelled value")
        elif qualname in ("itertools.chain", "itertools.combinations", "itertools.permutations", "itertools.product", "itertools.islice"):
            iterables = args if qualname in ("itertools.chain", "itertools.product") else args[:1]
            if not iterables or any(not _exact_type(a, containers) for a in iterables):
                raise Unsupported(f"{qualname}: an iterable that is not an exact list, tuple, str or dict")
            rest = args[len(iterables):] if qualname not in ("itertools.chain", "itertools.product") else []
            if any(not _exact_type(a, (int, bool, type(None))) for a in rest) or any(not _exact_type(v, (int, bool)) for v in kwargs.values()):
                raise Unsupported(f"{qualname}: a symbolic or non-int count")
        elif qualname in ("heapq.nlargest", "heapq.nsmallest"):
            if len(args) != 2 or kwargs or type(args[0]) is not int or not _exact_type(args[1], containers):
                raise Unsupported(f"{qualname}: only (n, iterable) with a concrete int n is modelled")
        elif qualname == "bisect.bisect_left":
            if not 2 <= len(args) <= 4 or kwargs or not _exact_type(args[0], (list, tuple)) or any(not _exact_type(a, (int, bool)) for a in args[2:]):
                raise Unsupported("bisect.bisect_left: an unmodelled call shape")
        elif mutates:
            if len(args) != (2 if qualname != "heapq.heapify" else 1) or kwargs or type(args[0]) is not list \
                    or not isinstance(n.args[0], ast.Name):
                raise Unsupported(f"{qualname}: only a named list (and one item) is modelled")
        rule = get_rule(qualname)
        target = list(args[0]) if mutates else None
        call_args = ([target, *args[1:]] if mutates else args)
        try:
            out = rule.native(*call_args, **kwargs)
            if qualname.startswith("itertools."):
                out = list(itertools.islice(out, _NATIVE_LIMIT + 1))
                if len(out) > _NATIVE_LIMIT:
                    raise Unsupported(f"{qualname}: more than {_NATIVE_LIMIT} items")
        except (IndexError, KeyError, ValueError):
            raise _Dead() from None
        except Unsupported:
            raise
        except Exception as e:  # noqa: BLE001 -- anything else is a raise this executor does not model
            raise Unsupported(f"{qualname} raises {type(e).__name__} on these operands") from None
        if mutates:
            name = n.args[0].id
            self._mutable(name, (list,))
            self.env[name] = target
            return out
        return _LazyIter(out) if qualname.startswith("itertools.") else out

    def _deep_concrete(self, v, depth=0) -> bool:
        if self._concrete(v):
            return True
        if depth > 6:
            return False
        if _exact_type(v, (list, tuple)):
            return all(self._deep_concrete(x, depth + 1) for x in v)
        if type(v) is dict:
            return all(self._concrete(k) and self._deep_concrete(x, depth + 1) for k, x in v.items())
        return False

    def _model_random_choice(self, n):
        """``random.choice(seq)`` lifted ([internal ref]): unlike the base class's version, a symbolic index here gets a real bound (``0 <= idx < m``)
        on the path, so a solved index cannot rely on this ite chain's default arm doing what the child's clamp does for a negative value
        (it does not -- see ``_choice_ite``); an empty seq really does raise IndexError, ending this path like any other."""
        idx, seq = self.ev(n.args[1]), self.ev(n.args[2])
        if not isinstance(seq, (list, tuple)):
            raise Unsupported("choice over an unmodelled sequence")
        m = len(seq)
        if not m:
            raise _Dead()
        if not _is_sym(idx):
            return seq[max(0, min(int(idx), m - 1))]
        idx = _num(idx)
        self._holds(z3.And(idx >= 0, idx < m))
        return _choice_ite(idx, seq)

    def _model_list_append(self, n):
        cur = self._mutable(n.func.value.id, (list,))
        v = self.ev(n.args[0])
        self.assign(ast.Name(n.func.value.id, ast.Store()), cur + [v])
        return None

    def _model_dict_get(self, n, d):
        k = self.ev(n.args[0])
        dflt = self.ev(n.args[1]) if len(n.args) == 2 else None
        if not _is_sym(k):
            return d.get(k, dflt)
        out = dflt
        for x in reversed([x for x in d if isinstance(x, (int, bool))]):
            out = _typed_ite(_num(k) == x, d[x], out)
        return out

    def _short_circuit_ok(self, node):
        if self.objects is not None and any(isinstance(x, ast.Call) for x in ast.walk(node)):
            raise Unsupported("any()/all() over an expression that calls code: the executor evaluates every element")

    # --- calls ----------------------------------------------------------------
    def ev_Call(self, n):
        """A call of a user class or of a method (``_symexec_objects.py``), else the generic dispatch.  A receiver the
        object check had to evaluate (an expression that runs code) is evaluated ONCE: it is parked in ``_pre``
        for the generic dispatch, which evaluates it again by node."""
        kind = self.fresh_draws.get(id(n))
        if kind is not None:
            return self._fresh_draw(kind)
        if self.objects is None:
            return self._ev_call(n)
        mark = len(self._pre)
        try:
            handled, v = self.objects.call(n)
            if handled:
                return v
            return self._ev_call(n)
        finally:
            while len(self._pre) > mark:
                self._pre.popitem()

    def _ev_call(self, n):
        f = n.func
        if isinstance(f, ast.Attribute) and f.attr == "get" and 1 <= len(n.args) <= 2 and not n.keywords:
            d = self.ev(f.value)
            if isinstance(d, dict):
                return self._through_rule("dict.get", n, args=(d,))
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "__SV_POS":
            return None                  # the harness's own bookkeeping
        if isinstance(f, ast.Name) and f.id == "__sv_next":
            # the stream is an array and its position a term of the path: paths that took
            # different branches read different elements next, and merge like any value
            p = n.args[1].id
            if p not in self.streams:
                raise Unsupported(f"stream {p} not modelled")
            pos = self.env.get(_POS + p, 0)
            if pos is POISON:
                raise Unsupported(f"stream {p} read after an unmodelled draw")
            self._holds(pos < STREAM_BOUND if _is_sym(pos) else pos < STREAM_BOUND)
            self.env[_POS + p] = pos + 1
            self.stream_use[p] = min(STREAM_BOUND, self.stream_use.get(p, 0) + 1)
            return z3.Select(self.streams[p], pos)
        if isinstance(f, ast.Name) and f.id == "__sv_ok_binop":
            return self._ok_binop(n.args[0].value, self.ev(n.args[1]), self.ev(n.args[2]))
        if isinstance(f, ast.Name) and f.id in self.defs and not f.id.startswith("__sv") \
                and f.id not in self.env:
            return self._inline(self.defs[f.id], [self.ev(a) for a in n.args], n)
        if isinstance(f, ast.Attribute) and f.attr == "append" and isinstance(f.value, ast.Name) \
                and len(n.args) == 1 and not n.keywords:
            return self._through_rule("list.append", n)
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.attr in _NATIVE_ATTRS and f.value.id not in self.defs:
            try:
                recv = self.ev(f.value)
            except Unsupported:
                recv = None                     # a module or an unmodelled name: the generic dispatch below refuses it as before
            owner = next((cls.__name__ for cls in (list, dict, tuple) if type(recv) is cls), None)
            if owner is not None and f"{owner}.{f.attr}" in NATIVE_METHODS:
                return self._through_rule(f"{owner}.{f.attr}", n, args=(recv,))
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
            v = self._contract_call(f, n)
            if v is not _NO_CONTRACT:
                return v
        if isinstance(f, ast.Name) and f.id == "__sv_index":
            # random.choice(seq), lifted ([internal ref]), read through its rule ([internal ref] step 2): ``_Runner._model_random_choice``
            return self._through_rule("random.choice", n)
        if isinstance(f, ast.Name) and f.id == "range" and not n.keywords and self._is_builtin("range"):
            return self._through_rule("range", n)
        if isinstance(f, ast.Name) and f.id == "open" and self._is_builtin("open"):
            return self._open_output(n)
        if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                and isinstance(self.env.get(f.value.id), _OutHandle)):
            return self._handle_call(self.env[f.value.id], f.attr, n)
        return super().ev_Call(n)

    def _is_builtin(self, name):
        """``name`` still means the builtin: nothing of the harness binds it."""
        return not (
            name in self.env or name in self.defs or name in self.g
            or (self.frames and name in self.frames[0])
        )

    def _model_range(self, n):
        if not n.keywords and self._is_builtin("range"):
            return self._range(n)
        return super()._model_range(n)

    def _range(self, n):
        """``range(...)`` of INT arguments, exactly ([internal ref] S1).

        Every argument must be an int (a Python int or bool, a z3 ``Int``, or a z3 ``Bool``,
        which ``range`` takes as 0 or 1): anything else -- a float, a string, a typed union, a
        symbolic float -- raises :class:`Unsupported` (``range`` would raise ``TypeError``, or
        the term is not known to be an int); nothing is approximated.  A zero step is a
        ``ValueError`` at the call, before any iteration: the path ends there.  With every
        argument concrete the result is the list of its values, as before; with a symbolic
        one it is a :class:`SymRange`, iterated by :meth:`_range_loop`."""
        if not 1 <= len(n.args) <= 3:
            raise Unsupported("symbolic range")
        terms, symbolic = [], False
        for a in (self.ev(arg) for arg in n.args):
            if isinstance(a, int):                       # bool is an int
                terms.append(int(a))
            elif _is_sym(a) and a.sort() == z3.IntSort():
                terms.append(a)
                symbolic = True
            elif _is_sym(a) and z3.is_bool(a):
                terms.append(z3.If(a, 1, 0))
                symbolic = True
            else:
                raise Unsupported("symbolic range")
        start, stop, step = (0, terms[0], 1) if len(terms) == 1 else (
            terms[0], terms[1], terms[2] if len(terms) == 3 else 1)
        self._holds(step != 0)                           # range(): ValueError for a zero step
        if symbolic:
            return SymRange(start, stop, step)
        if len(range(start, stop, step)) > 4096:
            raise Unsupported("loop shape")
        return list(range(start, stop, step))

    def _text_ok(self, v):
        """The guard under which ``v`` is a ``str`` (True / a z3 condition), ``False`` when it never
        is: a concrete non-str, a z3 term of another sort, a list.  A typed union keeps its str
        members.  Anything this executor has no model of is :class:`Unsupported`."""
        if isinstance(v, Union):
            gs = [g for g, m in v.members if self._text_ok(m) is not False]
            return z3.Or(*[_bool(g) for g in gs]) if gs else False
        if isinstance(v, str) or (_is_sym(v) and v.sort() == z3.StringSort()):
            return True
        if isinstance(v, (int, float, bool, bytes, bytearray, list, tuple, dict, set, SymList, type(None))) or (
                _is_sym(v) and v.sort() in (z3.IntSort(), z3.RealSort(), z3.BoolSort())):
            return False
        raise Unsupported("output file: an argument this executor has no model of")

    def _bytes_ok(self, v):
        """The same for a binary handle: only concrete ``bytes`` / ``bytearray`` are buffers here."""
        if isinstance(v, Union):
            gs = [g for g, m in v.members if self._bytes_ok(m) is not False]
            return z3.Or(*[_bool(g) for g in gs]) if gs else False
        if isinstance(v, (bytes, bytearray)):
            return True
        if isinstance(v, (str, int, float, bool, list, tuple, dict, set, SymList, type(None))) or (
                _is_sym(v) and v.sort() in (z3.IntSort(), z3.RealSort(), z3.BoolSort(), z3.StringSort())):
            return False
        raise Unsupported("output file: an argument this executor has no model of")

    def _open_output(self, n):
        out = self._visit_rule("open.output", CallCtx(n))
        if out is _NO_CONTRACT:
            raise Unsupported("open: not a write-only constant mode")
        return out

    def _open_output_model(self, n):
        """``open(path, <constant write-only mode>, ...)`` ([internal ref], the shape contract ``open.output``,
        read through ``solve/contracts/adapters/concolic.py``): a fresh :class:`_OutHandle`.  Any
        other shape of ``open`` (a read, an update mode, a descriptor, ``buffering``, a non-constant
        mode) is :class:`Unsupported`, as before.

        The path must be a ``str`` or a z3 string; a concrete non-path (``None``, a float, a list) is
        the TypeError the contract names, which ends the path; an int may be a file descriptor, so a
        z3 or concrete int is Unsupported.  ``encoding``/``errors``/``newline`` must be ``str`` or
        ``None``.  What the environment does (OSError) and the text options' ValueErrors are not
        property classes: the model assumes ``open`` succeeds, and the replay of any model on the
        real program decides."""
        from tharkun.solve.contracts.adapters import concolic as _contracts
        from tharkun.solve.environment import open_effect

        if _contracts.output_open_contract() is None or open_effect(n) != "output":
            raise Unsupported("open: not a write-only constant mode")
        names = ("file", "mode", "buffering", "encoding", "errors", "newline")
        given = dict(zip(names, n.args))
        given.update({k.arg: k.value for k in n.keywords})
        if "buffering" in given:
            raise Unsupported("open: buffering is not modelled")
        mode = given["mode"].value                      # open_effect: a constant string
        values = {k: self.ev(node) for k, node in given.items() if k != "mode"}
        path = values["file"]
        if isinstance(path, Union) or isinstance(path, bool) or isinstance(path, int) or (
                _is_sym(path) and path.sort() in (z3.IntSort(), z3.BoolSort())):
            raise Unsupported("open: a path that may be a file descriptor")
        if not (isinstance(path, str) or (_is_sym(path) and path.sort() == z3.StringSort())):
            if isinstance(path, (float, list, tuple, dict, set, type(None), SymList)) or _is_sym(path):
                raise _Dead()                           # TypeError: expected str, bytes or os.PathLike
            raise Unsupported("open: a path this executor has no model of")
        for what in ("encoding", "errors", "newline"):
            if what in values:
                v = values[what]
                if not (v is None or isinstance(v, str) or (_is_sym(v) and v.sort() == z3.StringSort())):
                    if isinstance(v, (int, float, bool, list, tuple, dict, set, SymList)) or _is_sym(v):
                        raise _Dead()                   # TypeError: open() argument must be str or None
                    raise Unsupported(f"open: {what} this executor has no model of")
        binary = "b" in mode
        if binary and any(w in values for w in ("encoding", "errors", "newline")):
            raise Unsupported("open: a binary mode with a text option")
        return _OutHandle(binary)

    def _handle_call(self, handle, method, n):
        owner = "io.BufferedWriter" if handle.binary else "io.TextIOWrapper"
        out = self._visit_rule(f"{owner}.{method}", CallCtx(n, handle, method))
        if out is _NO_CONTRACT:
            raise Unsupported(f"output file: {method} is not modelled with this shape")
        return out

    def _handle_call_model(self, handle, method, n):
        """``write``, ``writelines``, ``flush``, ``close``, ``__enter__``, ``__exit__`` of an output
        file ([internal ref]): the contract's raise claims, as paths.  A wrong-typed argument is a TypeError
        (even on a closed file: the argument is parsed first), which ends the path; ``write`` returns
        the length written.  A shape the contract does not cover is Unsupported."""
        from tharkun.solve.contracts.adapters import concolic as _contracts

        if (_contracts.output_handle_contract(handle.binary, method) is None
                or not _contracts.output_handle_call_shape(method, len(n.args), bool(n.keywords))):
            raise Unsupported(f"output file: {method} is not modelled with this shape")
        args = [self.ev(a) for a in n.args]
        ok = self._bytes_ok if handle.binary else self._text_ok
        if method == "write":
            self._holds(ok(args[0]))
            v = args[0]
            if isinstance(v, Union):
                raise Unsupported("output file: the length of a typed union")
            return z3.Length(v) if _is_sym(v) else len(v)
        if method == "writelines":
            seq = args[0]
            if not isinstance(seq, (list, tuple)):
                raise Unsupported("output file: writelines of a value this executor cannot iterate")
            for item in seq:
                self._holds(ok(item))
            return None
        if method == "__enter__":
            return handle
        return None

    def _with_output(self, st):
        """``with <output file> [as name]: body`` ([internal ref]): ``__enter__`` returns the handle,
        ``__exit__`` closes it and returns ``None`` (the body's exception propagates unchanged).
        A body that leaves by ``return``/``break``/``continue`` is Unsupported (this executor runs
        a ``with`` body as a block).  Any other context manager is Unsupported, as before."""
        handles = []
        for item in st.items:
            handle = self.ev(item.context_expr)
            if not isinstance(handle, _OutHandle):
                raise Unsupported("With")
            handles.append((item, handle))
        for node in ast.walk(ast.Module(body=st.body, type_ignores=[])):
            if isinstance(node, (ast.Return, ast.Break, ast.Continue)):
                raise Unsupported("loop control in a with body")
        for item, handle in handles:
            enter = ast.Call(func=ast.Name(id="__enter__", ctx=ast.Load()), args=[], keywords=[])
            ast.copy_location(enter, item.context_expr)
            self._handle_call(handle, "__enter__", enter)
            if item.optional_vars is not None:
                if not isinstance(item.optional_vars, ast.Name):
                    raise Unsupported("With target")
                self.assign(item.optional_vars, handle)
        self.run(st.body)
        for item, handle in reversed(handles):
            exit_call = ast.Call(func=ast.Name(id="__exit__", ctx=ast.Load()),
                                 args=[_Lit(None), _Lit(None), _Lit(None)], keywords=[])
            ast.copy_location(exit_call, item.context_expr)
            self._handle_call(handle, "__exit__", exit_call)

    def _range_loop(self, st, r):
        """``for target in <SymRange>``: iteration ``k`` (``k = 0, 1, ...``) runs exactly when
        ``start + k*step`` is still short of ``stop`` in the direction of ``step`` -- the
        definition of ``range``'s length -- with ``target = start + k*step``.  Unrolled under
        those guards up to :data:`LOOP_BOUND`; a longer range is cut (the run is then marked
        bounded, which a proof refuses and a search only uses as a hint).  The target is
        assigned INSIDE the iteration, so after the loop it holds the last value it was given,
        or what it held before when the range was empty, as in Python; a target that did not
        exist before is not readable afterwards (poisoned)."""
        start, stop, step = r.start, r.stop, r.step

        def runs(k):
            cur = start + k * step
            if isinstance(step, int):
                return cur < stop if step > 0 else cur > stop
            return z3.Or(z3.And(step > 0, cur < stop), z3.And(step < 0, cur > stop))

        for k in range(LOOP_BOUND):
            assign = ast.Assign(targets=[st.target], value=_Lit(start + k * step), type_comment=None)
            ast.copy_location(assign, st)
            self._merge_if(runs(k), [assign, *st.body], [])
        self.bounded = True
        self._holds(z3.Not(runs(LOOP_BOUND)))

    def _contract_call(self, f: ast.Attribute, n: ast.Call):
        # Native string/regex calls are resolved by value or module identity in ev_Call.
        if get_rule(f"{f.value.id}.{f.attr}") is not None and f.value.id in {"str", "re"}:
            return _NO_CONTRACT
        # Math belongs to the base executor's dedicated call hook.
        if self._math_function(f) is not None or f.value.id == "math":
            return _NO_CONTRACT
        # the standard-library functions run natively (``NATIVE_MODULE_FUNCTIONS``): the module is the one the NAME is bound to in the
        # harness (identity), whatever the spelling (``import heapq as hq``), and never a local or a def of that name
        module = self.g.get(f.value.id)
        shadowed = (f.value.id in self.env or f.value.id in self.defs or
                    (self.frames and f.value.id in self.frames[0]))
        if not shadowed:
            for name, canonical in _NATIVE_MODULES.items():
                if module is canonical and type(module) is ModuleType:
                    qualname = f"{name}.{f.attr}"
                    rule = get_rule(qualname)
                    if qualname in NATIVE_MODULE_FUNCTIONS and rule is not None and module.__dict__.get(f.attr) is _NATIVE_CALLABLES[qualname]:
                        return self._through_rule(qualname, n)
        if f"{f.value.id}.{f.attr}" in NATIVE_MODULE_FUNCTIONS:
            return _NO_CONTRACT                  # the spelling of a module is not the module: refused, never run by name
        return self._visit_rule(f"{f.value.id}.{f.attr}", CallCtx(n))

    def _contract_call_model(self, f: ast.Attribute, n: ast.Call):
        """[internal ref] step 2: consult ``solve/contracts`` for a module-level call
        neither this override nor the base ``ev_Call`` special-cases, instead of
        going straight to ``Unsupported`` (see ``solve/contracts/adapters/
        concolic.py``). Returns :data:`_NO_CONTRACT` when nothing applies, so the
        caller falls through to the exact same ``Unsupported`` it would have
        raised without this hook. Wrapped so a bug here can only ever cost a
        solved case, never break one that worked before -- ``Unsupported`` and
        ``_Dead`` (a real, intended outcome: this path's precondition failed)
        propagate normally; anything else is swallowed.
        """
        from tharkun.solve.contracts.adapters import concolic as _contracts

        qualname = f"{f.value.id}.{f.attr}"
        arg_count, has_keywords = len(n.args), bool(n.keywords)
        try:
            if not n.keywords:
                rng = _contracts.range_return(qualname)
                if rng is not None and not n.args:
                    lo, hi, hi_inclusive = rng
                    r = z3.Real(self._contract_var_name(qualname))
                    self._holds(z3.And(r >= lo, r <= hi if hi_inclusive else r < hi))
                    return r
                # 2026-09-28 owner review: applicability (exact arity, no keywords)
                # is now checked INSIDE the shared common.py functions themselves
                # -- both return None for a call shape they haven't hand-verified,
                # the same conservative "nothing to say about this call" reading
                # as no contract at all.
                exc = _contracts.empty_first_arg_precondition(
                    qualname, arg_count=arg_count, has_keywords=has_keywords)
                if exc is not None and n.args:
                    container = self.ev(n.args[0])
                    length = (
                        container.length if isinstance(container, SymList)
                        else len(container) if isinstance(container, (list, tuple))
                        else None
                    )
                    if length is not None:
                        self._holds(length > 0)
                        # [internal ref] step 2 (2026-09-28 redesign): APPLY the contract's
                        # own mutates_receiver effect -- heappop removes exactly
                        # one element -- instead of leaving the argument's tracked
                        # length stale. Downgrades to a length-only SymList (an
                        # honest precision loss: elements are no longer modelled
                        # after this call), never a claim about which element
                        # remains (concolic has no element-type facts to update,
                        # unlike absint's elem-join -- SymList tracks length only).
                        delta_spec = _contracts.receiver_length_delta(
                            qualname, arg_count=arg_count, has_keywords=has_keywords)
                        if delta_spec is not None:
                            arg_index, delta, _new_elem_idx = delta_spec
                            if delta != 0 and isinstance(n.args[arg_index], ast.Name):
                                self.assign(ast.Name(n.args[arg_index].id, ast.Store()),
                                            SymList(length + delta))
                        return z3.Int(self._contract_var_name(qualname))
        except (Unsupported, _Dead):
            raise
        except Exception:  # noqa: BLE001 -- a contract-hook bug must not crash the solve
            pass
        return _NO_CONTRACT

    def _contract_var_name(self, qualname: str) -> str:
        seq = getattr(self, "_contract_seq", 0) + 1
        self._contract_seq = seq
        return f"_contract_{qualname.replace('.', '_')}_{seq}"

    def _inline(self, fd, args, call, qualname=None, ctx=None):
        """Run ``fd`` with ``args`` bound to its parameters.  ``qualname`` is the scope the call_target fact is keyed
        by (a method: ``Class.method``); ``ctx`` is ``(defining class, "function" | "static" | "class")`` for a
        method, so ``super()`` knows where it stands (``_symexec_objects.py``).  A method's decorators are the
        class table's business (it checked the live object), a function's are refused as before."""
        if call.keywords or len(args) != len(fd.args.args) or fd.args.vararg or fd.args.kwarg \
                or (ctx is not None and (fd.args.posonlyargs or fd.args.kwonlyargs)) \
                or (fd.decorator_list and ctx is None):
            raise Unsupported(f"call shape of {fd.name}")
        if any(isinstance(x, (ast.Global, ast.Nonlocal)) for x in ast.walk(fd)):
            raise Unsupported(f"{fd.name} writes a global")
        if self.depth >= CALL_DEPTH:
            raise Unsupported("call depth")
        saved_env = self.env
        self.frames.append(saved_env)
        self.env = {a.arg: v for a, v in zip(fd.args.args, args)}
        self.depth += 1
        self.scope_stack.append(qualname or fd.name)
        self.ctx_stack.append(None if ctx is None else (ctx[0], ctx[1], fd.args.args[0].arg if fd.args.args else None))
        try:
            return self._exec_fn(list(fd.body))
        finally:
            self.ctx_stack.pop()
            self.scope_stack.pop()
            self.depth -= 1
            self.frames.pop()
            self.env = saved_env

    def run_body(self, stmts):
        """The harness body, run as the function body it is.  A harness lifted from an ENTRY function (``def main():
        ... return ...`` then ``main()``, ``sites.py::_plan``) returns early: ``if not (lo <= x <= hi): return``
        ends the path, and what follows runs under the guard.  Every other statement runs through :meth:`run`, so a
        statement the executor cannot model is still skipped and its names poisoned, as when the body had no
        ``return``; the value of a ``return`` is dropped (the caller of the entry function drops it too) but is
        evaluated, since evaluating it can reach a site."""
        self._exec_fn(list(stmts), top=True)

    def _exec_fn(self, stmts, top=False):
        """Run a function body; its value is merged over the paths that return.  ``top``: the harness body
        itself (:meth:`run_body`), whose plain statements are skipped and poisoned, not propagated."""
        for i, st in enumerate(stmts):
            if isinstance(st, ast.Return):
                if st.value is None:
                    return None
                if top:
                    self.run([ast.copy_location(ast.Expr(st.value), st)])
                    return None
                return self.ev(st.value)
            if isinstance(st, ast.If) and _returns(st):
                t = _truth(self.ev(st.test))
                rest = stmts[i + 1:]
                if not _is_sym(t):
                    return self._exec_fn((st.body if t else st.orelse) + rest, top)
                base_env, base_pc = self.env, list(self.pc)
                base_heap = self.objects.heap if self.objects is not None else None
                vals = []
                for guard, arm in ((t, st.body), (z3.Not(t), st.orelse)):
                    self.env, self.pc = dict(base_env), base_pc + [guard]
                    if base_heap is not None:
                        self.objects.heap = base_heap.fork()
                    try:
                        v = self._exec_fn(arm + rest, top)
                        vals.append((_conj(self.pc[len(base_pc):]), v,
                                     self.objects.heap if base_heap is not None else None))
                    except _Dead:
                        pass
                self.env, self.pc = base_env, base_pc
                if not vals:
                    raise _Dead()
                if len(vals) == 1:
                    if base_heap is not None:
                        self.objects.heap = vals[0][2]
                    self._holds(vals[0][0])
                    return vals[0][1]
                if base_heap is not None:
                    self.objects.heap = self.objects.merge(_bool(vals[0][0]), vals[0][2], vals[1][2])
                self._either(vals[0][0], vals[1][0], t)
                return _typed_ite(_bool(vals[0][0]), vals[0][1], vals[1][1])
            if top:
                self.run([st])
            else:
                self.st(st)
        return None

    def _either(self, ga, gb, t):
        """After a merge: the path continues on arm a or arm b (skip the tautology)."""
        if _is_sym(t) and ga is t and _is_sym(gb) and z3.eq(gb, z3.Not(t)):
            return
        self.pc.append(z3.Or(_bool(ga), _bool(gb)))

    # --- statements -----------------------------------------------------------
    def run(self, stmts):
        for st in stmts:
            pc = list(self.pc)
            version = self.objects.version if self.objects is not None else 0
            try:
                self.st(st)
            except Unsupported as e:
                self.pc = pc
                self.skipped.append(f"line {st.lineno}: {e}")
                if self.objects is not None:
                    self.objects.after_skip(st, version, str(e))
                if _stores(st) or _has_draw(st) or any(
                        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in self.defs
                        for n in ast.walk(st)) or (self.objects is not None and self.objects.may_run_user_code(st)):
                    # a skipped ``print(...)`` changes nothing a later site reads
                    self.poisoned.append(f"line {st.lineno}: {e}")
                for line in self._targets_in(st):
                    self.unsupported.setdefault(line, str(e))
                for name in _stores(st):
                    self.env[name] = POISON
                if _has_draw(st):        # stream positions after it are unknown
                    for p in self.streams:
                        self.env[_POS + p] = POISON

    def st(self, st):
        # a site marker: record a target's condition, skip the others
        if isinstance(st, ast.If) and len(st.body) == 1 and isinstance(st.body[0], ast.Expr) \
                and isinstance(st.body[0].value, ast.Call) \
                and getattr(st.body[0].value.func, "id", "") == "__sv_site":
            if st.body[0].lineno in self.targets:
                self.goals[st.body[0].lineno].append(self._path(_truth(self.ev(st.test))))
            return
        if isinstance(st, ast.Raise):
            if st.lineno in self.targets:
                self.goals[st.lineno].append(self._path())
            raise _Dead()
        if isinstance(st, ast.Assert):
            t = _truth(self.ev(st.test))
            if st.lineno in self.targets:
                self.goals[st.lineno].append(self._path(z3.Not(t) if _is_sym(t) else (not t)))
            self._holds(t)
            return
        if isinstance(st, ast.Expr):
            self.ev(st.value)            # unmodelled calls: run() poisons their receivers
            return
        if isinstance(st, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            raise Unsupported(type(st).__name__)
        if isinstance(st, ast.If):
            t = _truth(self.ev(st.test))
            if not _is_sym(t):
                self.run(st.body if t else st.orelse)
                return
            self._merge_if(t, st.body, st.orelse)
            return
        if isinstance(st, ast.With):
            self._with_output(st)
            return
        if isinstance(st, ast.For):
            _check_loop(st)
            it = self._seq(self.ev(st.iter))
            if isinstance(it, SymRange):
                self._range_loop(st, it)
                return
            if isinstance(it, SymList):
                self._bounded_loop(st, lambda i: i < it.length, None, it.length)
                return
            if not isinstance(it, (list, tuple)) or len(it) > 4096:
                raise Unsupported("loop shape")
            for x in it:
                self.assign(st.target, x)
                self.run(st.body)
            return
        if isinstance(st, ast.While):
            _check_loop(st)
            unrolled, total = 0, 0
            while True:
                t = _truth(self.ev(st.test))
                if not _is_sym(t):
                    if not t:
                        return
                    total += 1
                    if total > 4096:
                        raise Unsupported("loop too long to unroll")
                    self.run(st.body)
                    continue
                if unrolled >= LOOP_BOUND:
                    self.bounded = True
                    self._holds(z3.Not(t))
                    return
                unrolled += 1
                self._merge_if(t, st.body, [])
        super().st(st)

    def _bounded_loop(self, st, guard, value, bound_term):
        if not isinstance(st.target, ast.Name):
            raise Unsupported("loop target")
        for i in range(LOOP_BOUND):
            self.env[st.target.id] = value(i) if value is not None else POISON
            self._merge_if(guard(i), st.body, [])
        self.bounded = True
        self._holds(bound_term <= LOOP_BOUND)

    def _merge_if(self, t, body, orelse):
        base_env, base_pc = self.env, list(self.pc)
        base_heap = self.objects.heap if self.objects is not None else None
        arms = []
        for guard, stmts in ((t, body), (z3.Not(t), orelse)):
            self.env, self.pc = dict(base_env), base_pc + [guard]
            if base_heap is not None:
                self.objects.heap = base_heap.fork()
            try:
                self.run(stmts)
                arms.append((_conj(self.pc[len(base_pc):]), self.env,
                             self.objects.heap if base_heap is not None else None))
            except _Dead:
                pass
        self.pc = base_pc
        if not arms:
            raise _Dead()
        if len(arms) == 1:
            self.env = arms[0][1]
            if base_heap is not None:
                self.objects.heap = arms[0][2]
            self._holds(arms[0][0])
            return
        (ga, ea, ha), (gb, eb, hb) = arms
        if base_heap is not None:
            self.objects.heap = self.objects.merge(_bool(ga), ha, hb)
        merged = {}
        for k in set(ea) | set(eb):
            miss = 0 if k.startswith(_POS) else POISON      # an unread stream is at position 0
            a, b = ea.get(k, miss), eb.get(k, miss)
            if a is b:
                merged[k] = a
            elif a is POISON or b is POISON:
                merged[k] = POISON
            else:
                try:
                    merged[k] = _typed_ite(_bool(ga), a, b)
                except Unsupported:
                    merged[k] = POISON
        self.env = merged
        self._either(ga, gb, t)


def _bool(g):
    return g if _is_sym(g) else z3.BoolVal(bool(g))


def _check_loop(st):
    if st.orelse:
        raise Unsupported("loop else")
    for n in ast.walk(ast.Module(body=st.body, type_ignores=[])):
        if isinstance(n, (ast.Break, ast.Continue, ast.Return)):
            raise Unsupported("loop control")


def _returns(st) -> bool:
    return any(isinstance(n, ast.Return) for n in ast.walk(st))


def _targets_by_function(defs: dict, targets) -> dict:
    """``{function: target lines inside it or inside a function it calls}``."""
    reach = {name: {t for t in targets if fd.lineno <= t <= (fd.end_lineno or fd.lineno)}
             for name, fd in defs.items()}
    calls = {name: {n.func.id for n in ast.walk(fd) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name) and n.func.id in defs}
             for name, fd in defs.items()}
    changed = True
    while changed:
        changed = False
        for name in defs:
            for callee in calls[name]:
                if not reach[callee] <= reach[name]:
                    reach[name] |= reach[callee]
                    changed = True
    return {name: r for name, r in reach.items() if r}


def solve_targets(harness_text: str, module_globals: dict, fn_name: str, lines, params: list,
                  kinds: list, stream_lens: dict, timeout_ms: int = 10000, call_targets=None,
                  proven_classes=()) -> dict:
    """``{line: (outcome, inputs_or_reason)}`` for every target line, from ONE symbolic run
    of the whole harness: calls inlined, loops whose trip count depends on a draw
    unrolled up to :data:`LOOP_BOUND` under their guards, values whose type depends on
    the path kept apart for typing sites.  ``outcome`` is

    * ``sat`` -- the draw inputs (to be run and replayed; nothing here is trusted);
    * ``unsat`` -- no modelled run fires the site (runs that leave every loop within
      ``LOOP_BOUND`` iterations and read streams up to their modelled length);
    * ``unreached`` -- no modelled run gets to the site, and nothing was left unmodelled;
    * ``unsupported`` -- a statement on the way was not modelled (the reason);
    * ``unknown`` -- the solver gave up.

    ``call_targets`` are the program's ``call_target`` records (``solve/call_targets.py``); with them (a list,
    possibly empty) the run models the program's own classes (``sv/_symexec_objects.py``), without them a call
    of a class is unmodelled.  ``proven_classes`` are the names of the class-origin ``module_constant`` records
    that are proven (a class used by name but never constructed)."""
    tree = ast.parse(harness_text)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    env, streams = {}, {}
    for p, k in zip(params, kinds):
        base = k[5:] if k.startswith("list:") else k
        if base not in ("int", "bool", "float", "str", "rindex"):
            env[p] = POISON              # unsupported draw kind
            if k.startswith("list:"):
                streams[p] = []
            continue
        if k.startswith("list:"):
            sort = {"int": z3.IntSort(), "bool": z3.BoolSort(), "float": z3.RealSort(), "str": z3.StringSort()}[base]
            streams[p] = z3.Array(p, z3.IntSort(), sort)
            env[p] = POISON              # read only through __sv_next
        else:
            env[p] = _var(p, k)
    r = _Runner(tree, module_globals, env, streams, lines, call_targets, proven_classes)
    try:
        r.run_body(fn.body)
    except _Dead:
        pass
    bound = f" (loops cut at {LOOP_BOUND} iterations)" if r.bounded else ""
    out = {}
    for line in lines:
        goals = [g for g in r.goals.get(line) or () if g is not False]
        if not goals:
            if line in r.unsupported:
                out[line] = ("unsupported", r.unsupported[line])
            elif r.poisoned:
                out[line] = ("unsupported", f"not reached; unmodelled: {r.poisoned[0]}")
            else:
                out[line] = ("unreached", "no modelled run gets there" + bound)
            continue
        s = z3.Solver()
        s.set("timeout", timeout_ms)
        s.add(z3.Or(*[_bool(g) for g in goals]))
        res = s.check()
        if res == z3.sat:
            m = s.model()
            inputs = {}
            for p, k in zip(params, kinds):
                base = k[5:] if k.startswith("list:") else k
                if base not in ("int", "bool", "float", "str", "rindex"):
                    inputs[p] = [] if k.startswith("list:") else ""
                elif k.startswith("list:"):
                    used = r.stream_use.get(p, 0)
                    inputs[p] = [_value(m, z3.Select(streams[p], i), base) for i in range(used)]
                else:
                    inputs[p] = _value(m, env[p], k)
            out[line] = ("sat", inputs)
        elif line in r.unsupported:
            out[line] = ("unsupported", r.unsupported[line])
        elif res == z3.unsat:
            out[line] = ("unsat", "no modelled run fires it" + bound)
        else:
            out[line] = ("unknown", "solver gave up")
    return out


# ===========================================================================
# for provers that need the WHOLE run to be exact, not just one site ([internal ref])
# ===========================================================================

def make_var(name: str, kind: str):
    """A fresh Z3 term for one ``_sv_verifier`` draw of ``kind`` -- the public half of
    :func:`_var`, for a caller building its OWN small symbolic run (not a full harness
    via :func:`solve_site` / :func:`solve_targets`)."""
    return _var(name, kind)


def model_value(model, term, kind):
    """The concrete Python value ``kind`` calls for, read back from a solved ``model``
    for ``term`` -- the public half of :func:`_value`, decoding what :func:`make_var`
    built."""
    return _value(model, term, kind)


def run_exact(tree: "ast.Module", module_globals: dict, env: dict, body: list, lines,
              fresh_draws: dict | None = None) -> dict:
    """Run ``body`` (a statement list drawn from ``tree``, e.g. a module's own top
    level with its draws already pre-bound into ``env`` and its declarations
    stripped) through the SAME whole-program interpreter :func:`solve_targets` uses
    -- the exact machinery :mod:`tharkun.sv.concolic` already trusts to find
    violations, with unbounded Z3 ``Int``s, ``abs`` as ``ite``, and Python floor
    division / ``>>`` modelled exactly (see this module's own docstring).

    Unlike :func:`solve_targets`, which reports ``unsat`` as "no modelled run fires
    it" per site (a hint a search may use to stop early, since a bound anywhere in
    the run could make that unsound), this returns whether the run was EXACT --
    nothing anywhere in it, not only on the path to a checked line, was left
    unmodelled or cut at a bound -- because a caller PROVING a universal property
    needs the whole run to be trustworthy, not merely one query answered.

    Returns ``{"exact": bool, "reason": str, "goals": {line: [z3 formulas]}}``;
    ``goals`` (the condition, on each path that reaches ``line``, under which it
    fires -- the same shape :func:`solve_targets` builds) is only meaningful when
    ``exact`` is true. Never raises for an unmodelled construct inside ``body`` --
    that is exactly what turns ``exact`` false; a raised :class:`Unsupported` means
    the TOP-LEVEL shape (``tree``/``body`` themselves) could not even be parsed
    into a run, which a caller should treat the same as "not exact".

    ``fresh_draws`` is ``{id(call node): kind}`` for the ``_sv_verifier.nondet_*()`` calls the caller let sit
    inside the program's functions (``product_proof.py``): each EXECUTION of such a call is a fresh unconstrained
    term of its kind (a function called twice draws twice), so the run covers every vector of draws, whatever
    path consumes them."""
    r = _Runner(tree, module_globals, env, {}, lines, fresh_draws=fresh_draws)
    try:
        r.run(list(body))
    except _Dead:
        pass
    if r.bounded:
        return {"exact": False, "reason": "a loop was cut at the unroll bound", "goals": {}}
    if r.poisoned or r.skipped or r.inexact:
        # ``run`` drops a statement it cannot model unless it stores a name, draws or
        # calls a def (a skipped ``print`` "changes nothing a later site reads") -- fine
        # for a search, fatal for a proof: an attribute store, a nested import/def, an
        # ``__sv_*`` call vanish that way. ANY skipped statement, anywhere, is inexact; so is a ``print``
        # the executor answers without tracking its output (``inexact``).
        first = (r.skipped or r.inexact or r.poisoned)[0]
        return {"exact": False, "reason": f"not fully modelled: {first}", "goals": {}}
    if r.unsupported:
        first = next(iter(r.unsupported.values()))
        return {"exact": False, "reason": f"not fully modelled: {first}", "goals": {}}
    return {"exact": True, "reason": "",
            "goals": {line: [g for g in r.goals.get(line) or () if g is not False] for line in lines}}


def _native_model(qualname):
    def model(self, n, recv):
        return self._native_method(qualname, n, recv)
    model.__name__ = "_model_" + qualname.replace(".", "_")
    return model


for _q in NATIVE_METHODS:
    setattr(_Runner, "_model_" + _q.replace(".", "_"), _native_model(_q))


def _native_module_model(qualname):
    def model(self, n):
        return self._native_module_function(qualname, n)
    model.__name__ = "_model_" + qualname.replace(".", "_")
    return model


for _q in NATIVE_MODULE_FUNCTIONS:
    setattr(_Runner, "_model_" + _q.replace(".", "_"), _native_module_model(_q))
