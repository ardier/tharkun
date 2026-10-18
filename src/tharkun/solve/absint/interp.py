"""The absint interpreter: an abstract interpretation of one whole Python program.

It walks the program's AST from the module body, inlining every call to the
program's own functions (context-sensitive by call string; recursion is not
modelled and aborts).  Loops are solved by a widening fixpoint followed by one
*verified* narrowing step: the narrowed head is kept only when the loop body
maps it into itself, so soundness never depends on the transfer functions
being monotone.

Every operation that can raise one of the property exceptions records a
**check**: ``(line, col, op, exceptions, ok, fact)``.  Checks are recorded only
in the final pass over a loop body (the pass that starts from the proven loop
invariant), so every recorded ``ok=True`` holds for every reachable state.
A check that is not proven records ``ok=False``; the analysis then continues
on the states where the operation did not raise (the raising executions end
the program, so they cannot reach a later site).

Anything the interpreter does not model raises :class:`Unsupported`, and the
verdict is UNKNOWN: absint never guesses.  Aborting constructs include
``try``/``with``, generators, closures and unsupported lambda forms, recursion, ``eval``/``exec``,
``getattr``/``setattr``, unmodelled builtins and library calls, and user
classes with dunder methods it does not model.
"""

from __future__ import annotations

import ast
import builtins as _builtins
import math
import os
import struct
import sys

from .. import values as fv
from ..contracts.adapters import math_model as _mm
from ..loop_family import ACCUMULATOR, AFFINE, ITERATOR, KERNEL, SEARCH, classify_loop_family
from . import lemmas
from ..lowering.provenance import ROLE_ATTR, path_of

from .errors import Unsupported
from .rand import RandomMixin
from .reducers import NO_HOOK_CLASSES, ReducersMixin
from .sets import FROZENSET_METHODS, SET_METHODS, SetsMixin
from .state import Frame, Incompatible, State, iv_of_len, vjoin
from .visitor import (AbsintVisitor, CallCtx, LengthHintCtx, MethodCtx, OutfileCtx, ProtocolCtx, SubscriptCtx, exact_int_value,
                      hook_free_value, int_or_none_value, non_int_value, serve_module_functions, serve_set_operations)
from .aval import (ANYSTR, ANY_STR, BOOL, BOT, FLOAT, FloatInfo, NONE, V, const_bool, const_int, const_str, int_iv)
from .heap import (ArrayObj, DefaultDictObj, DictObj, DictView, BoundMethod, BuiltinMethod, ClassObj, FuncObj, InstObj, IteratorObj, ListObj, OutFileObj, RangeObj, SetObj, StringIOObj, TupleObj, seg_var)
from .linear import (IntInfo, Lin, lin_bounds, lin_entails_le)
from .zone import ZERO

#: plain joins of the loop head before the first widening (0: widen at the first update, as Mopsa does)
WIDEN_DELAY = 0
MAX_ITERS = 40
#: results kept per (loop, call string, analyzer scratch) for fixpoint reuse; the oldest goes first
FIX_REUSE_WAYS = 8
MAX_DEPTH = 24
MAX_STAR_ARGS = 64
#: case splits on a boolean draw one analysis may make (docs/ABSINT.md "Case splits on a boolean draw"); a full tree
#: of depth 6 is 63 splits, and once the budget is spent the draws are analysed joined, as they always were
MAX_DRAW_SPLITS = 64

# A decimal int below this limit has at most CPython's minimum enabled conversion limit (640) digits.
_PRINT_INT_ABS_LIMIT = 10 ** 640

#: the seven property classes, in the contracts' order
PROPERTY_EXC_ORDER = ("TypeError", "AttributeError", "IndexError", "KeyError", "ZeroDivisionError", "FloatingPointError",
                      "AssertionError")

PROPERTY_EXCS = frozenset({
    "TypeError", "AttributeError", "IndexError", "KeyError",
    "ZeroDivisionError", "FloatingPointError", "AssertionError",
})

# builtin exception classes -> the property exceptions an instance of it is
# (itself and every property exception it is a subclass of)
_EXC_ANCESTORS: dict[str, frozenset] = {}
for _n in dir(_builtins):
    _o = getattr(_builtins, _n)
    if isinstance(_o, type) and issubclass(_o, BaseException):
        _EXC_ANCESTORS[_n] = frozenset(c.__name__ for c in _o.__mro__ if c.__name__ in PROPERTY_EXCS)

# exception classes that have a *builtin* subclass that is a property class, so
# `raise Base(...)` cannot raise a property exception but catching could.
_BUILTIN_NAMES = frozenset(dir(_builtins))

UNSUPPORTED_DUNDERS = frozenset({
    "__new__", "__getattr__", "__getattribute__", "__setattr__", "__delattr__", "__del__",
    "__init_subclass__", "__class_getitem__", "__set_name__", "__get__", "__set__", "__delete__",
    "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__", "__hash__",
    "__iter__", "__next__", "__getitem__", "__setitem__", "__delitem__", "__contains__",
    "__add__", "__radd__", "__iadd__", "__sub__", "__rsub__", "__isub__", "__mul__", "__rmul__",
    "__imul__", "__truediv__", "__rtruediv__", "__itruediv__", "__floordiv__", "__rfloordiv__",
    "__mod__", "__rmod__", "__pow__", "__rpow__", "__neg__", "__pos__", "__abs__", "__invert__",
    "__and__", "__or__", "__xor__", "__lshift__", "__rshift__", "__matmul__",
    "__index__", "__int__", "__float__", "__complex__", "__round__", "__trunc__", "__floor__",
    "__ceil__", "__bytes__", "__call__", "__enter__",
    "__exit__", "__missing__", "__reversed__", "__instancecheck__", "__subclasscheck__",
    "__mro_entries__", "__prepare__",
})

#: why a builtin of the family "calls the argument's dunder" runs no user hook here (see ``Analyzer._modelled``)
_NO_USER_DUNDER = ("no user hook runs: a class defining {} is refused where it is defined (UNSUPPORTED_DUNDERS), so no "
                   "instance the analysis can hold has one")


#: the methods of the builtin heap kinds the analyzer models by hand and reads through their rules ([internal ref] step 2): rule qualname
#: -> why no user hook runs (``Analyzer._modelled``).  A method that is not here is served by its model alone.
_METHOD_HOOKS = {
    "list.append": "the element is stored as it is: no protocol call on it",
    "list.insert": _NO_USER_DUNDER.format("__index__"),
    "list.pop": _NO_USER_DUNDER.format("__index__"),
    "list.reverse": "the elements are not compared or touched",
    "list.copy": "the elements are copied as they are: no protocol call on them",
    "list.clear": "the elements are dropped: no protocol call on them",
    "list.count": "an operand or element with a user comparison is refused (custom_comparison): == of numbers, strs and containers of them "
                  "runs no user code",
    "list.index": "an operand or element with a user comparison is refused (custom_comparison); the bounds are ints (__index__ is "
                  "refused where a class defines it)",
    "list.remove": "an operand or element with a user comparison is refused (custom_comparison): == of numbers, strs and containers of "
                   "them runs no user code",
    "list.sort": "the elements are numbers or strs (_elem_sortable): no __lt__ runs, and key is None",
    "tuple.count": "an operand or element with a user comparison is refused (custom_comparison)",
    "tuple.index": "an operand or element with a user comparison is refused (custom_comparison); the bounds are ints",
    "list.extend": "an instance's __iter__ body is analysed (iterable()) and its __len__ probe is ruled by protocol.length_hint "
                   "(consumer list.extend); a builtin iterable runs no hook",
    "dict.get": "a key that is not a scalar is refused (check_hash_key): the hash and equality of a str, int, float, bool or None "
                "run no user code",
    "dict.keys": "a view is built, no element is touched",
    "dict.values": "a view is built, no element is touched",
    "dict.items": "a view is built, no element is touched",
    "io.StringIO.write": "a class deriving from str is refused where it is defined (a builtin base other than object and the "
                         "exceptions is), so no __str__ of a str subclass runs",
    "io.StringIO.read": _NO_USER_DUNDER.format("__index__"),
    "io.StringIO.seek": _NO_USER_DUNDER.format("__index__"),
    "io.StringIO.getvalue": "the buffer is returned as it is",
    "list.__getitem__": _NO_USER_DUNDER.format("__index__"),
    "list.__setitem__": _NO_USER_DUNDER.format("__index__"),
    "tuple.__getitem__": _NO_USER_DUNDER.format("__index__"),
    "dict.__getitem__": "a key that is not a scalar is refused (check_hash_key): the hash and equality of a str, int, float, bool or "
                        "None run no user code",
    "collections.defaultdict.__getitem__": "the exact constructor model admits only the builtin list factory or None, and check_hash_key "
                                            "refuses non-scalar keys, so neither key nor factory runs user code",
    "dict.__setitem__": "a key that is not a scalar is refused (check_hash_key): the hash and equality of a str, int, float, bool or "
                        "None run no user code",
}


#: the call's own classes the model of a method argues without recording a check: rule qualname -> {class: why}
_METHOD_NATIVE = {
    "list.sort": {"TypeError": "the elements are numbers or strs that compare (_elem_sortable) and the keywords are key=None and reverse"},
    "io.StringIO.read": {"TypeError": "an accepted size is int- or bool-shaped; any other is refused (unmodelled size argument)"},
    "io.StringIO.seek": {"TypeError": "an accepted position is int- or bool-shaped; any other is refused (unmodelled position)"},
}


#: the functions of the ``random`` module the analyzer models by hand (``rand.py``) and reads through their rules ([internal ref] step 2):
#: rule qualname -> why no user hook runs (``Analyzer._modelled``).  ``random.random`` and ``random.seed`` have none: the contract
#: says ``never`` for every class their call shape leaves open.
_RANDOM_NO_HOOK = ("``RandomMixin._random_args`` refuses a user object by name (instance_value): no ``{}`` of a user class runs, and "
                   "an int, float, str, bool, None, list, tuple or array runs none")
_MODULE_HOOKS = {
    "_sv_verifier.nondet_list": "the callback is an exact alias of a supported zero-argument verifier scalar draw; its native body invokes no user code",
    "_sv_verifier.nondet_dict": "both callbacks are exact aliases of supported zero-argument verifier scalar draws; generated keys are primitive and invoke no user code",
    "_sv_verifier.check_type": "the modeled paths use one exact builtin class hint and a primitive scalar, or an exact list/dict whose hint matches; builtin isinstance checks invoke no user hook",
    "collections.defaultdict": "this narrow transfer accepts only the builtin list type or None as the factory and no initializer; construction does not call the factory",
    "copy.copy": "the argument is an exact builtin list: CPython uses list.copy, so no __copy__ or pickle reducer runs",
    "random.randint": _RANDOM_NO_HOOK.format("__index__"),
    "random.randrange": _RANDOM_NO_HOOK.format("__index__"),
    "random.choice": _RANDOM_NO_HOOK.format("__len__`` or ``__getitem__"),
    "random.uniform": _RANDOM_NO_HOOK.format("__sub__``, ``__mul__`` or ``__add__"),
    "random.random": "the call has no operand",
    "random.seed": "seed type-checks its argument directly (None, int, float, str, bytes, bytearray): no dunder is involved",
}


#: what a contract says about each property class, computed once per contract object (the contracts are immutable; a test that replaces
#: one gets a new object): ``(contract, plan, checks)`` keyed by ``id(contract)`` -- ``plan`` is ``((class, kind), ...)`` with kind
#: ``never`` / ``native`` (the call's own logic) / ``propagate`` (only through a dunder) / ``other``, ``checks`` is
#: ``adapters/absint.py::raise_checks``.  Hot paths (the read of an attribute) ask for it on every visit.
_CLAIMS: dict = {}


def _claims(contract):
    hit = _CLAIMS.get(id(contract))
    if hit is not None and hit[0] is contract:
        return hit
    from tharkun.solve.contracts.adapters import absint as _contracts
    from tharkun.solve.contracts.adapters.common import native_raise_condition, never_raises, raise_condition

    plan = []
    for exc in PROPERTY_EXC_ORDER:
        if never_raises(contract, exc):
            plan.append((exc, "never"))
        elif native_raise_condition(contract, exc) is not None:
            plan.append((exc, "native"))
        else:
            condition = raise_condition(contract, exc)
            plan.append((exc, "propagate" if condition is not None and condition.startswith("propagates uncaught") else "other"))
    _CLAIMS[id(contract)] = hit = (contract, tuple(plan), _contracts.raise_checks(contract))
    return hit


#: the object protocols the analyzer implements by hand (``special``, ``binary_special``, ``iterable``, ``truth_split``, ``_len_model``) and
#: reads through their rules ([internal ref] step 2): why a hook the contract admits ("propagates uncaught if the dunder raises C") is no new
#: source of the class here, and the claim about the dispatch itself that the model argued without recording a check
_PROTOCOL_BODY = ("the dunder's body is ANALYSED (call_method): an exception it raises is recorded at the node that raises it, and "
                  "a dunder the analysis cannot follow is a refusal")
_PROTOCOL_HOOKS = {
    "protocol.attribute_access": "a class defining __getattribute__ or __getattr__ is refused where it is defined (UNSUPPORTED_DUNDERS), so "
                                 "the lookup of an attribute of an instance the analysis can hold runs no user code",
}
#: the protocols whose dispatch is on the hot path of every program with an object: only the classes left unproven are recorded
_PROTOCOL_LEAN = frozenset({"protocol.attribute_access"})
_PROTOCOL_NATIVE = {
    "protocol.binary_operator": {"TypeError": "an applicable method was found (special refuses an unresolved dispatch and a "
                                              "NotImplemented result); a miss is the failing check binary_special records, which vetoes this"},
    "protocol.inplace_operator": {"TypeError": "an applicable __i<op>__ or binary method was found (special refuses an unresolved "
                                               "dispatch and a NotImplemented result); a miss is the failing check binary_special records"},
    "protocol.comparison": {"TypeError": "an applicable method was found, or == / != fell back to identity (no user code); an ordering "
                                         "with no method is the failing check binary_special records, which vetoes this"},
    "protocol.container_getitem": {"TypeError": "the class's __getitem__ was resolved (special refuses an unresolved dispatch)"},
    "protocol.container_len": {"TypeError": "the class's __len__ was resolved (special) and a result that is not an int is refused; a "
                                            "TypeError raised in its body is recorded at its own node"},
    "protocol.iteration": {"TypeError": "the class's __iter__ was resolved (special) and must return a modelled iterator (else refused)"},
    "protocol.attribute_access": {},
    "protocol.truthiness": {"TypeError": "a custom __bool__ is refused; the __len__ of a class that has one is the model of len (a result "
                                         "that is not an int is refused)"},
}


def _reflective_unresolved_name(name: str) -> bool:
    """True for a name that class_lookup() coming back None does NOT mean "this attribute
    does not exist" for -- a dunder (``__dict__``, ``__repr__``, ...) or ``__dict__`` itself,
    which every real class/instance provides implicitly (through `object`, or as CPython's own
    namespace view) whether or not the class defines anything of its own for it, and whose
    content (an address-showing repr/str, an identity hash, a class's own function objects, ...)
    can never be bounded either way. [internal ref] (soundness) / [internal ref] widened (MG, MM): a class that
    never defines one of these still resolves it at runtme; class_lookup() walking the MODELED
    MRO alone returns None for it, which is NOT the same fact as "this name does not exist": it
    means "unknown", not "absent". attr_load() and call_method() must raise Unsupported for one
    of these when class_lookup() comes back empty, rather than record a same-shaped result to an
    ordinary undefined name (a real AttributeError, sound to conclude bottom on). Mirrors the
    same over-approximation pyresolve.py's own REFLECTIVE/DUNDER_OK census already makes for a
    different purpose (no-site classification) -- every dunder-shaped name is untrusted here too,
    with no DUNDER_OK-style allowlist: the risk this guards is a modeling gap (an attribute that
    silently isn't resolvable), not a semantic one (whether invoking it does something hidden),
    so there is no dunder for which "unresolved in the modeled hierarchy" reliably means "does
    not exist on the real object" either.
    """
    return name == "__dict__" or (len(name) > 4 and name.startswith("__") and name.endswith("__"))


# Each admitted protocol is dispatched through the ordinary function/heap
# transfer below. Admission without a transfer is never a proof.
BINARY_METHODS = {
    ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "truediv",
    ast.FloorDiv: "floordiv", ast.Mod: "mod", ast.Pow: "pow",
}
COMPARISON_METHODS = {
    ast.Eq: ("__eq__", "__eq__"), ast.NotEq: ("__ne__", "__ne__"),
    ast.Lt: ("__lt__", "__gt__"), ast.LtE: ("__le__", "__ge__"),
    ast.Gt: ("__gt__", "__lt__"), ast.GtE: ("__ge__", "__le__"),
}
SUPPORTED_DUNDERS = {"__getitem__", "__setitem__", "__len__", "__iter__", "__call__"} | {
    "__" + prefix + name + "__" for name in BINARY_METHODS.values() for prefix in ("", "r", "i")
} | {name for pair in COMPARISON_METHODS.values() for name in pair}
UNSUPPORTED_DUNDERS = (UNSUPPORTED_DUNDERS - SUPPORTED_DUNDERS) | {"__length_hint__"}


class Check:
    __slots__ = ("line", "col", "op", "excs", "ok", "facts")

    def __init__(self, line, col, op, excs):
        self.line = line
        self.col = col
        self.op = op
        self.excs = set(excs)
        self.ok = True
        self.facts: list[str] = []


class _ProbeScope:
    """One active length probe (:meth:`Analyzer._probe_length`).  CPython's ``PyObject_LengthHint`` swallows
    a ``TypeError`` from ``__len__`` and iterates with no hint, so the probe must CONTINUE from every state
    in which one was raised.  An explicit ``raise`` of a ``TypeError`` is followed exactly: ``s_Raise`` keeps
    a copy of the state at the raise in ``raises``.  A ``TypeError`` that comes from an unproven check (an
    arity mismatch, ``len(5)``, a non-int result) leaves no such state, so it sets ``type_error`` and the
    probe refuses.  Every other class propagates, which is what ending the path models."""
    __slots__ = ("type_error", "raises", "fctx")

    def __init__(self, fctx=None):
        self.type_error = False
        self.raises: list = []
        self.fctx = fctx   # [internal ref]: the function the probe runs in; a TypeError leaving __len__ into it is swallowed


class FuncCtx:
    __slots__ = ("node", "local_names", "global_names", "returns", "is_module", "is_class",
                 "module_key", "raises")

    def __init__(self, node, local_names, global_names, is_module=False, is_class=False,
                 module_key=""):
        self.node = node
        self.local_names = local_names
        self.global_names = global_names
        self.returns: list = []
        self.raises: list = []
        self.is_module = is_module
        self.is_class = is_class
        # [internal ref]: "" is the program's own module (unchanged behaviour); see FuncObj.module_key.
        self.module_key = module_key


_EXC_GROUPS = frozenset({"ExceptionGroup", "BaseExceptionGroup"})
_ANY_LOOP = object()   # _finally_frame: any loop (a return leaves every loop)


class _Exc:
    """One exception in flight ([internal ref]): the state at the raise (a copy, owned by the exception), the class
    (a builtin exception name, or the heap address of a user exception class: identity, never a name), the
    instance value, the site, and the check the site would have recorded outside a try (``op``, ``fact``,
    ``residual``, ``operand``): it is recorded only where the exception escapes (``Analyzer._route_exc``)."""
    __slots__ = ("st", "cls", "val", "node", "op", "fact", "residual", "operand")

    def __init__(self, st, cls, val, node, op, fact, residual=None, operand=None):
        self.st = st
        self.cls = cls
        self.val = val
        self.node = node
        self.op = op
        self.fact = fact
        self.residual = residual
        self.operand = operand

    def with_state(self, st):
        return _Exc(st, self.cls, self.val, self.node, self.op, self.fact, self.residual, self.operand)


class _TryFrame:
    """One active try statement ([internal ref]), or the implicit ``try/finally`` of an except clause (the
    handler's exit deletes its ``as`` name and releases the exception).  ``finalizer`` is the finally
    block as a function of the state (``None`` without one); ``has_handlers`` whether the frame has
    except clauses.  Exits that leave the frame wait here until its finally has run on them
    (``Analyzer._finish_frame``): ``pending_returns``/``pending_breaks``/``pending_continues``, the
    exceptions ``raised`` in the body (for the handlers) and the ``unhandled`` ones (no clause matched,
    or raised in a handler or in else)."""
    __slots__ = ("node", "fctx", "lctx", "finalizer", "has_handlers", "in_handler", "in_finally",
                 "raised", "unhandled", "pending_returns", "pending_breaks", "pending_continues")

    def __init__(self, node, fctx, lctx, finalizer, has_handlers):
        self.node = node
        self.fctx = fctx
        self.lctx = lctx
        self.finalizer = finalizer
        self.has_handlers = has_handlers
        self.in_handler = False      # executing a handler or the else block: a raise there leaves the statement
        self.in_finally = False      # executing the finally block: a raise or jump there leaves the statement
        self.raised: list = []
        self.unhandled: list = []
        self.pending_returns: list = []
        self.pending_breaks: list = []
        self.pending_continues: list = []

class _FixpointRun:
    """One finished loop fixpoint, kept for reuse (``Analyzer._fixpoint``): the entry state it was solved
    from, the head it produced, and what running it changed in the analyzer besides the head."""
    __slots__ = ("entry", "head", "tmp", "epoch", "borrowed", "landmarks")

    def __init__(self, entry, head, tmp, epoch, borrowed, landmarks):
        self.entry, self.head, self.tmp = entry, head, tmp
        self.epoch, self.borrowed, self.landmarks = epoch, borrowed, landmarks


class LoopCtx:
    __slots__ = ("breaks", "continues")

    def __init__(self):
        self.breaks: list[State] = []
        self.continues: list[State] = []


def _class_has_private_name_mangling(class_node: ast.ClassDef) -> bool:
    """Whether a class suite contains an identifier that CPython would mangle."""
    class Visitor(ast.NodeVisitor):
        def __init__(self):
            self.classes: list[str] = []
            self.found = False

        def check(self, name):
            if (self.classes and self.classes[-1] and isinstance(name, str)
                    and name.startswith("__") and not name.endswith("__")):
                self.found = True

        def visit_Name(self, node):
            self.check(node.id)

        def visit_arg(self, node):
            self.check(node.arg)
            self.generic_visit(node)

        def visit_Global(self, node):
            for name in node.names:
                self.check(name)

        def visit_Nonlocal(self, node):
            for name in node.names:
                self.check(name)

        def visit_alias(self, node):
            imported = node.name.split(".", 1)[0]
            self.check(imported)
            if node.asname is not None:
                self.check(node.asname)

        def visit_FunctionDef(self, node):
            self.check(node.name)
            self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, node):
            if self.classes:
                self.check(node.name)
            # Class header expressions execute in the enclosing lexical class scope.
            for child in (*node.decorator_list, *node.bases, *node.keywords):
                self.visit(child)
            self.classes.append(node.name.lstrip("_"))
            for statement in node.body:
                self.visit(statement)
            self.classes.pop()

        def visit_keyword(self, node):
            if node.arg is not None:
                self.check(node.arg)
            self.generic_visit(node)

        def visit_ExceptHandler(self, node):
            self.check(node.name)
            self.generic_visit(node)

        def visit_MatchAs(self, node):
            self.check(node.name)
            self.generic_visit(node)

        def visit_MatchStar(self, node):
            self.check(node.name)
            self.generic_visit(node)

        def visit_MatchMapping(self, node):
            self.check(node.rest)
            self.generic_visit(node)

    visitor = Visitor()
    visitor.visit(class_node)
    return visitor.found


def _assigned_names(body) -> tuple[set[str], set[str]]:
    """Local names of a function body (Python scoping) and its ``global`` names."""
    loc: set[str] = set()
    glo: set[str] = set()

    class V_(ast.NodeVisitor):
        def visit_FunctionDef(self, n):
            loc.add(n.name)  # nested scopes are not entered

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, n):
            loc.add(n.name)

        def visit_Lambda(self, n):
            pass

        def visit_ListComp(self, n):
            pass

        visit_SetComp = visit_DictComp = visit_GeneratorExp = visit_ListComp

        def visit_Global(self, n):
            glo.update(n.names)

        def visit_Nonlocal(self, n):
            raise Unsupported("nonlocal", n)

        def visit_Name(self, n):
            if isinstance(n.ctx, (ast.Store, ast.Del)):
                loc.add(n.id)

        def visit_NamedExpr(self, n):
            raise Unsupported("assignment expression", n)

        def visit_Import(self, n):
            for a in n.names:
                loc.add((a.asname or a.name).split(".")[0])

        def visit_ImportFrom(self, n):
            for a in n.names:
                loc.add(a.asname or a.name)

        def visit_ExceptHandler(self, n):
            if n.name:
                loc.add(n.name)
            self.generic_visit(n)

    vis = V_()
    for s in body:
        vis.visit(s)
    return loc - glo, glo


def _is_generator(fnode) -> bool:
    for n in ast.walk(fnode):
        if isinstance(n, (ast.Yield, ast.YieldFrom, ast.Await)):
            # only count yields of this function, not nested ones
            return True
    return False


def _iv_add(a, b):
    return (None if a[0] is None or b[0] is None else a[0] + b[0],
            None if a[1] is None or b[1] is None else a[1] + b[1])


def _iv_neg(a):
    return (None if a[1] is None else -a[1], None if a[0] is None else -a[0])


def _iv_mul(a, b):
    if a == (0, 0) or b == (0, 0):
        return (0, 0)
    if None in a or None in b:
        # sign reasoning for half-bounded intervals
        def sign(x):
            lo, hi = x
            if lo is not None and lo >= 0:
                return 1
            if hi is not None and hi <= 0:
                return -1
            return 0

        sa, sb = sign(a), sign(b)
        if sa and sb:
            if sa * sb > 0:
                lo = (a[0] * b[0]) if sa > 0 else (a[1] * b[1])
                return (lo, None)
            hi = (a[0] * b[1]) if sa > 0 else (a[1] * b[0])
            return (None, hi)
        return (None, None)
    ps = [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]
    return (min(ps), max(ps))


def _iv_contains0(a):
    return (a[0] is None or a[0] <= 0) and (a[1] is None or a[1] >= 0)


def _pow2_k(n: int) -> "int | None":
    """``k`` such that ``n == 2**k``, or ``None`` when ``n`` is not a positive
    power of two ([internal ref]: shared by the a >> k == a // 2**k comparison-site rewrite
    in compare_split, see _shr_floordiv_identity_holds)."""
    if n <= 0 or (n & (n - 1)) != 0:
        return None
    return n.bit_length() - 1


class Analyzer(RandomMixin, ReducersMixin, SetsMixin):
    def __init__(self, source: str, filename: str = "<program>", time_budget: float = 60.0,
                 class_hints=None, invariant_locations=(), lowered=None, site_source: str | None = None,
                 declared_files=(), split_draws=False):
        import time as _t
        # docs/ABSINT.md "Case splits on a boolean draw": run a statement that holds one
        # ``_sv_verifier.nondet_bool()`` once per outcome and keep the rest of its block apart
        self.split_draws = split_draws
        self._forced_draws: dict = {}     # draw Call node -> (outcome, the activation it was forced in)
        self._partition_depth = 0         # >0 while a case split's outcomes run apart (see capture_invariant)
        self._split_budget = MAX_DRAW_SPLITS
        self.splits_done = 0
        self._clock = _t.process_time
        self.deadline = _t.process_time() + time_budget
        self.source = source
        self.filename = filename
        self.declared_files = tuple(declared_files)
        # [internal ref]: ``lowered`` is a ``solve.lowering`` result for this source.  The program that is
        # interpreted is its tree (comprehensions are loops there; every node keeps the line and
        # column of the original node it stands for), and everything this analyzer REPORTS about it
        # is put back in terms of the original program (``_orig``).
        self.lowered = lowered
        self.tree = lowered.tree if lowered is not None else ast.parse(source, filename)
        self.site_source = source if site_source is None else site_source
        self._record_source_tree = None
        self._record_source_nodes: dict[str, ast.AST] = {}
        self._record_source_parents: dict[ast.AST, ast.AST] = {}
        self._source_sites: dict[ast.AST, object] = {}
        self._site_parents: dict[ast.AST, ast.AST] = {}
        self._statement_entry_nodes: dict[ast.AST, list[ast.AST]] = {}
        if lowered is None:
            from tharkun.solve.sitekey import prestate_info, site_keys

            self._site_parents = {child: parent for parent in ast.walk(self.tree)
                                  for child in ast.iter_child_nodes(parent)}
            self._record_source_tree = self.tree if source == self.site_source else ast.parse(self.site_source, filename)
            original_keys = site_keys(self._record_source_tree, self.site_source, filename)
            self._record_source_nodes = {key.ast_path: node for node, key in original_keys.items()}
            if self._record_source_tree is self.tree:
                self._record_source_parents = self._site_parents
            else:
                self._record_source_parents = {child: parent for parent in ast.walk(self._record_source_tree)
                                               for child in ast.iter_child_nodes(parent)}
            self._record_builtin_vetoes = self._builtin_source_vetoes(self._record_source_tree)
            if source == self.site_source:
                self._source_sites = original_keys
            else:
                self._source_sites = self._map_source_sites(
                    self.tree, source, self.site_source, filename, self._record_source_tree,
                    original_keys=original_keys)
            for node in self._source_sites:
                scope, statement, _prefix = prestate_info(node, self._site_parents)
                if scope == "statement_entry":
                    if statement is not None:
                        self._statement_entry_nodes.setdefault(statement, []).append(node)
        self.invariant_locations = frozenset(invariant_locations)
        self.invariants = {}
        if self.invariant_locations:
            from tharkun.solve.obligations import source_points
            self.invariant_points = source_points(self.tree, source, filename)
        self.checks: dict[tuple, Check] = {}
        self.records: dict[tuple, dict] = {}
        self._record_meta: dict[tuple, dict] = {}
        self._record_keys_by_node: dict[ast.AST, set[tuple]] = {}
        self._record_rules: dict[str, bool] = {}
        self._record_binding_cache: dict[ast.AST, tuple[dict | None, tuple[str, ...]]] = {}
        self._record_source_function_scopes: dict[ast.AST, tuple[set[str], set[str]] | None] = {}
        self._record_source_class_bindings: dict[ast.AST, set[str] | None] = {}
        self._statement_entry_observations: dict[ast.AST, dict] = {}
        self.record = True
        # one entry per active call: (line, col, end_line, end_col) of the call node.  The START
        # position alone does not name a call: ``a + b + c`` is two ``__add__`` calls whose BinOp
        # nodes both start at ``a``, so two different calls shared one allocation site and the
        # second refused ("allocation would invalidate an in-flight heap reference", raytrace's
        # ``eye.vector + xcomp + ycomp``).  The site records' ``context`` keeps the (line, col) pair.
        self.callstring: list[tuple] = []
        self.active: list = []
        self.tmp = 0
        self.site_ids: dict = {}
        self.site_desc: dict[str, str] = {}
        self.epoch: set = set()
        self.borrowed: list[set] = [set()]
        self._probe_scopes: list = []  # active length probes (_probe_length); see check()
        self._taps: list = []  # (call node, [(excs, ok)]) of the hand-modelled builtins under a rule; see _modelled()
        self._try_stack: list[_TryFrame] = []    # [internal ref]: the active try statements, innermost last
        self._active_exc_stack: list[_Exc] = []  # [internal ref]: the exceptions being handled, innermost last
        self._fctx_stack: list[FuncCtx] = []     # [internal ref]: the function contexts executing, innermost last
        self._exit_count = 0                     # [internal ref]: bumped by every routed exit (see exec_block)
        self.visitor = AbsintVisitor(self)  # the contract rules' entry points (visitor.py)
        self._fix_cache: dict = {}  # see _fixpoint
        self.class_contracts: dict = {}
        self.class_hints = class_hints or {}
        self.features: set[str] = set()
        self.global_written: set[str] = set()
        self.module_ctx: FuncCtx | None = None
        for n in ast.walk(self.tree):
            if isinstance(n, ast.Global):
                self.global_written.update(n.names)
        # [internal ref] (import boundary, read-and-recurse): lazily built the first time an import
        # this Analyzer cannot model directly is seen; reuses solve/pyresolve.py::Resolver's
        # already-audited eligibility/cycle-guard/inert-package-init logic (extern.py) instead
        # of duplicating it. A failure to build it for ANY reason (including one unrelated to
        # imports) just means read-and-recurse never kicks in here -- module_attr's existing
        # ``V(top=True)`` fallback is unconditionally safe, never a soundness risk.
        self._cur_module_key = ""           # see site(); "" == the program's own file, always
        self._ext_resolver = "$unset$"
        self._ext_by_key: dict = {}
        self._ext_ok: set[str] = set()      # filenames whose top level executed cleanly
        self._ext_failed: set[str] = set()  # filenames that failed once (memoized refusal)
        self._ext_loading: set[str] = set()  # cycle guard

    # ================================================================ plumbing
    def _orig(self, node, text):
        """``text`` about the lowered program, at ``node``, in terms of the original program
        (``None`` when the original program has no way to say it there)."""
        if self.lowered is None or text is None or not text:
            return text
        return self.lowered.at_site(text, path_of(node))

    @staticmethod
    def _ast_path_prefixes(path: str) -> list[str]:
        if path == "$":
            return ["$"]
        if not path.startswith("$."):
            return []
        prefixes = ["$"]
        for part in path[2:].split("."):
            prefixes.append(prefixes[-1] + "." + part)
        return prefixes

    @classmethod
    def _map_source_sites(cls, tree, analyzed_source: str, source: str, filename: str,
                          original_tree=None, original_keys=None):
        """Join sliced nodes to original identities only along an exact, unchanged AST path."""
        from tharkun.solve.sitekey import site_keys

        current = site_keys(tree, analyzed_source, filename)
        if analyzed_source == source:
            return current
        original_tree = original_tree or ast.parse(source, filename)
        original = original_keys if original_keys is not None else site_keys(original_tree, source, filename)
        current_by_path = {key.ast_path: node for node, key in current.items()}
        original_by_path = {key.ast_path: (node, key) for node, key in original.items()}
        statement_matches = {}
        mapped = {}
        for node, key in current.items():
            prefixes = cls._ast_path_prefixes(key.ast_path)
            statement_indexes = [i for i, path in enumerate(prefixes)
                                 if isinstance(current_by_path.get(path), ast.stmt)]
            if not statement_indexes:
                continue
            statement_path = prefixes[statement_indexes[-1]]
            if statement_path not in statement_matches:
                current_statement = current_by_path[statement_path]
                original_pair = original_by_path.get(statement_path)
                original_statement = original_pair[0] if original_pair else None
                statement_matches[statement_path] = (
                    original_statement is not None
                    and type(current_statement) is type(original_statement)
                    and ast.dump(current_statement, include_attributes=True)
                    == ast.dump(original_statement, include_attributes=True)
                )
            original_pair = original_by_path.get(key.ast_path)
            original_key = original_pair[1] if original_pair else None
            if (statement_matches[statement_path] and original_key is not None
                    and key.function == original_key.function):
                mapped[node] = original_key
        return mapped

    def check(self, node, op: str, excs, ok: bool, fact: str = "", *, residual: str | None = None,
              operand=None, st: State | None = None) -> None:
        for tap_node, seen in self._taps:
            if tap_node is node:
                seen.append((frozenset(excs), ok))
        if self._probe_scopes and not ok and "TypeError" in excs and op != "raise":
            for scope in self._probe_scopes:     # whatever the pass, recording or not
                scope.type_error = True
        if not ok and self._try_stack:
            # [internal ref]: inside a try statement (of this function or of a caller: a handler up the call chain
            # may catch it) an unproven check is an exception in flight: one per class, routed to the
            # handlers from a copy of the raising state (_route_exc) and recorded only where it escapes.
            # A site with no state to hand over cannot be routed: refused, never recorded or dropped.
            if st is None:
                raise Unsupported(f"{op} may raise {','.join(sorted(excs))} inside a try: the raising state "
                                  "is not available at this site", node)
            for exc in sorted(excs):
                self._route_exc(_Exc(st.copy(), exc, V(st=frozenset({("excinst", exc)})), node, op, fact,
                                     residual, operand), self._fctx_stack[-1])
            return
        self._record_check(node, op, excs, ok, fact, residual=residual, operand=operand, st=st)

    def _record_check(self, node, op: str, excs, ok: bool, fact: str = "", *, residual: str | None = None,
                      operand=None, st: State | None = None) -> None:
        """Record a check at its site (``checks``: the proof; ``records``: the per-site records): what
        ``check`` does outside a try, and what an exception that escaped every handler does ([internal ref])."""
        if not self.record:
            return
        self._record(node, op, excs, ok, fact, residual, operand, st)
        fact = self._orig(node, fact) or ""
        key = (getattr(node, "lineno", 0), getattr(node, "col_offset", 0), op)
        c = self.checks.get(key)
        if c is None:
            c = self.checks[key] = Check(key[0], key[1], op, excs)
        c.excs |= set(excs)
        if not ok:
            if c.ok:
                c.facts = []  # show the failing context first
            c.ok = False
        if fact and fact not in c.facts and len(c.facts) < 4 and (ok == c.ok):
            c.facts.append(fact)

    # ---------------------------------------------------------- site records
    def _record(self, node, op, excs, ok, fact, residual, operand, st) -> None:
        """The per-site record shared with lanes A and B (tharkun sv/sites.py shape):
        ``line col exc kind function guards status fact operand residual scope types
        context loop_iteration``.  ``scope`` holds MUST facts only (they hold on every
        execution reaching the site); repeated visits intersect them."""
        line, col = getattr(node, "lineno", 0), getattr(node, "col_offset", 0)
        if self.lowered is not None:
            # a fact naming a fresh variable of the lowering is a fact about the original program
            # only inside the comprehension that variable belongs to; a synthetic name is nothing
            fact = self._orig(node, fact) or ""
            residual = self._orig(node, residual)
        for exc in sorted(excs):
            key = (line, col, op, exc)
            scope, types = self._scope_facts(st, residual) if (residual and st is not None) else (set(), {})
            if self.lowered is not None:
                scope = {r for r in (self._orig(node, f) for f in scope) if r}
                types = {r: v for r, v in ((self._orig(node, k), v) for k, v in types.items()) if r}
            site_key = self._source_sites.get(node)
            if site_key is not None:
                from dataclasses import replace
                site_key = replace(site_key, exception=exc)
            site = list(site_key.record_key()) if site_key is not None else None
            from tharkun.solve.sitekey import prestate_scope
            binding, binding_names = self._record_binding(node, site_key)
            fact_scope = None
            exact_binding_types = True
            contract = self._record_contract(op, exc, node)
            if prestate_scope(node, self._site_parents) == "statement_entry":
                observation = self._statement_entry_observations.get(node)
                if observation is not None and observation["valid"] and binding is not None:
                    binding = observation["binding"]
                    term_facts = sorted(observation["facts"])
                    builtins = sorted(observation["builtins"])
                    fact_scope = "statement_entry"
                else:
                    binding, binding_names, term_facts, builtins = None, (), [], []
            else:
                builtins = self._record_builtins(st, node, site_key)
                term_facts, exact_binding_types = (
                    self._record_terms(st, binding_names) if binding is not None and st is not None else ([], False)
                )
                if binding is not None and exact_binding_types:
                    fact_scope = "site"
            if binding is not None and not exact_binding_types:
                binding, binding_names, term_facts, fact_scope = None, (), [], None
            rec = self.records.get(key)
            ctx = [list(c[:2]) for c in self.callstring]
            if rec is None:
                self.records[key] = {
                    "line": line, "col": col, "exc": exc, "kind": op,
                    "function": getattr(self.active[-1], "name", "<lambda>") if self.active else "<main>", "guards": [],
                    "status": "safe" if ok else "may_raise", "fact": fact if ok else "",
                    "operand": (self._orig(node, ast.unparse(operand)) or "") if operand is not None else "",
                    "residual": residual, "scope": sorted(scope), "types": types,
                    "context": ctx, "loop_iteration": "any",
                    "site": site, "binding": binding, "facts": term_facts, "contract": contract,
                    "builtins": builtins,
                }
                if binding is not None:
                    self.records[key]["fact_scope"] = fact_scope
                self._record_meta[key] = {"site": tuple(site) if site is not None else None,
                                          "binding": binding, "contract": contract,
                                          "fact_scope": fact_scope, "node": node}
                self._record_keys_by_node.setdefault(node, set()).add(key)
                continue
            unsafe_before = rec["status"] == "may_raise"
            if not ok:
                rec["status"] = "may_raise"
                rec["fact"] = ""
            if ok and unsafe_before:
                pass  # a safe visit says nothing about the residual of an unsafe one (the first iteration, say)
            elif not ok and not unsafe_before:
                rec["residual"] = residual  # nor did the safe visits before the first unsafe one
            elif rec["residual"] != residual:
                rec["residual"] = None
            rec["scope"] = sorted(set(rec["scope"]) & scope)
            rec["types"] = {k: v for k, v in rec["types"].items() if types.get(k) == v}
            if rec["context"] != ctx:
                rec["context"] = []  # several contexts: the scope is their intersection
            meta = self._record_meta[key]
            site_identity = tuple(site) if site is not None else None
            if meta["site"] != site_identity:
                meta.update(site=None, binding=None, contract=None, fact_scope=None)
                rec.update(site=None, binding=None, contract=None, facts=[], builtins=[])
                rec.pop("fact_scope", None)
            elif meta["binding"] != binding or meta.get("fact_scope") != fact_scope:
                meta.update(binding=None, fact_scope=None)
                rec.update(binding=None, facts=[])
                rec.pop("fact_scope", None)
            elif meta["binding"] is not None:
                rec["facts"] = sorted(set(rec["facts"]) & set(term_facts))
            rec["builtins"] = sorted(set(rec["builtins"]) & set(builtins))
            if meta["contract"] != contract:
                meta["contract"] = None
                rec["contract"] = None

    def _record_contract(self, op, exc, node):
        direct = {"binop": "protocol.binary_operator", "divide": "protocol.binary_operator",
                  "unary": "protocol.unary_operator"}
        qualname = direct.get(op)
        if qualname is None and isinstance(op, str) and ":" in op:
            qualname = op.split(":", 1)[0]
        if qualname is None and op in {"operator", "compare"}:
            if isinstance(node, ast.Compare) and all(
                    type(cmp) in (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
                    for cmp in node.ops):
                qualname = "protocol.comparison"
            elif op == "operator" and isinstance(node, ast.BinOp):
                qualname = "protocol.binary_operator"
            elif op == "operator" and isinstance(node, ast.UnaryOp):
                qualname = "protocol.unary_operator"
        if not qualname:
            return None
        exists = self._record_rules.get(qualname)
        if exists is None:
            from tharkun.solve.contracts.rules import get_rule
            exists = get_rule(qualname) is not None
            self._record_rules[qualname] = exists
        return {"qualname": qualname, "exception": exc} if exists else None

    @staticmethod
    def _builtin_source_vetoes(tree) -> tuple[frozenset[str], bool]:
        """Return names with any source binding and whether reflection defeats all claims."""
        if tree is None:
            return frozenset(), True
        bound = set()
        reflective_names = {"globals", "locals", "vars", "exec", "eval", "setattr", "delattr",
                            "__builtins__", "__dict__"}
        type_parameter_nodes = tuple(
            cls for cls in (getattr(ast, "TypeVar", None), getattr(ast, "ParamSpec", None),
                            getattr(ast, "TypeVarTuple", None)) if cls is not None)
        reflective = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                bound.add(node.id)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.alias):
                if node.name == "*":
                    reflective = True
                else:
                    bound.add(node.asname or node.name.split(".", 1)[0])
                if node.name.split(".", 1)[0] == "builtins":
                    reflective = True
            elif isinstance(node, ast.ImportFrom) and node.module == "builtins":
                reflective = True
            elif isinstance(node, (ast.Global, ast.Nonlocal)):
                bound.update(node.names)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
            elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
                bound.add(node.name)
            elif isinstance(node, ast.MatchMapping) and node.rest:
                bound.add(node.rest)
            elif type_parameter_nodes and isinstance(node, type_parameter_nodes):
                bound.add(node.name)
            if isinstance(node, ast.Name) and node.id in reflective_names:
                reflective = True
            if isinstance(node, ast.Attribute) and node.attr in reflective_names:
                reflective = True
        return frozenset(bound), reflective

    def _record_builtins(self, st, node, site_key) -> list[str]:
        """Builtin identities from load_name's current, side-effect-free lookup order."""
        names = ("type", "int", "bool", "float", "str")
        if (st is None or st.bottom or site_key is None or self.lowered is not None
                or not self._fctx_stack):
            return []
        source_node = self._record_source_nodes.get(site_key.ast_path)
        if source_node is None:
            return []
        fctx = self._fctx_stack[-1]
        local_names = set(fctx.local_names) if not (fctx.is_module or fctx.is_class) else set()
        global_names = set(fctx.global_names)
        global_frame = st.frames[0]
        class_frame = st.frames[-1] if fctx.is_class else None
        bound_names, reflective = getattr(self, "_record_builtin_vetoes", (frozenset(), True))
        if reflective:
            return []
        result = []
        for name in names:
            if name in bound_names:
                continue
            if class_frame is not None and (name in class_frame.env or name in class_frame.maybe_unbound):
                continue
            if name in local_names and name not in global_names:
                continue
            key = self._global_key(name, fctx.module_key)
            if key in global_frame.env or key in global_frame.maybe_unbound:
                continue
            try:
                resolved = self.builtin_name(name, node)
            except (Unsupported, Incompatible):
                return []
            # ``type`` is a builtin identity even though the value interpreter
            # deliberately leaves its call result TOP (there is no b_type model).
            if (resolved.st == frozenset({("type", name)})
                    or (name == "type" and name in _BUILTIN_NAMES and resolved.top)):
                result.append(name)
        return sorted(result)

    @staticmethod
    def _statement_for(node, parents):
        cur = node
        while cur in parents:
            cur = parents[cur]
            if isinstance(cur, ast.stmt):
                return cur
        return None

    def _source_scope_matches(self, node, fctx, names):
        """Fail closed when sliced execution changed a name's original owner."""
        site = self._source_sites.get(node)
        original = self._record_source_nodes.get(site.ast_path) if site is not None else None
        if original is None:
            return False
        cur = original
        source_function = source_class = None
        while cur in self._record_source_parents:
            cur = self._record_source_parents[cur]
            if source_function is None and isinstance(cur, (ast.FunctionDef, ast.AsyncFunctionDef)):
                source_function = cur
            if source_class is None and isinstance(cur, ast.ClassDef):
                source_class = cur
        if source_function is not None:
            if (fctx.is_module or fctx.is_class or
                    (getattr(fctx.node, "name", None), getattr(fctx.node, "lineno", None),
                     getattr(fctx.node, "col_offset", None)) !=
                    (source_function.name, source_function.lineno, source_function.col_offset)):
                return False
            if source_function not in self._record_source_function_scopes:
                try:
                    local, global_names = _assigned_names(source_function.body)
                    args = source_function.args
                    local.update(n.arg for n in [*args.posonlyargs, *args.args, *args.kwonlyargs])
                    if args.vararg:
                        local.add(args.vararg.arg)
                    if args.kwarg:
                        local.add(args.kwarg.arg)
                    self._record_source_function_scopes[source_function] = (local - global_names, global_names)
                except Unsupported:
                    self._record_source_function_scopes[source_function] = None
            scope = self._record_source_function_scopes[source_function]
            if scope is None:
                return False
            local, global_names = scope
            for name in names:
                current = self._scope(name, fctx)
                if name in local and current != "local":
                    return False
                if name in global_names and current != "module":
                    return False
                if name not in local and current == "local":
                    return False
            return True
        if source_class is not None:
            if (not fctx.is_class or
                    (getattr(fctx.node, "name", None), getattr(fctx.node, "lineno", None),
                     getattr(fctx.node, "col_offset", None)) !=
                    (source_class.name, source_class.lineno, source_class.col_offset)):
                return False
            if source_class not in self._record_source_class_bindings:
                try:
                    local, global_names = _assigned_names(source_class.body)
                    self._record_source_class_bindings[source_class] = local - global_names
                except Unsupported:
                    self._record_source_class_bindings[source_class] = None
            local = self._record_source_class_bindings[source_class]
            if local is None:
                return False
            return True
        return fctx.is_module

    def _source_class_for(self, node):
        site = self._source_sites.get(node)
        original = self._record_source_nodes.get(site.ast_path) if site is not None else None
        cur = original
        while cur in self._record_source_parents:
            cur = self._record_source_parents[cur]
            if isinstance(cur, ast.ClassDef):
                return cur
        return None

    def _entry_value(self, st, fctx, name, node):
        if st.bottom or not st.frames:
            return None
        if fctx.is_class:
            frame = st.frames[-1]
            if name in frame.maybe_unbound:
                return None
            if name in frame.env:
                value = frame.env[name]
            else:
                source = self._source_class_for(node)
                local = self._record_source_class_bindings.get(source) if source is not None else None
                if local is None or name in local:
                    return None
                frame = st.frames[0]
                key = self._global_key(name, fctx.module_key)
                if key in frame.maybe_unbound or key not in frame.env:
                    return None
                value = frame.env[key]
        elif not (fctx.is_module or fctx.is_class) and self._scope(name, fctx) == "local":
            frame = st.frames[-1]
            if name in frame.maybe_unbound or name not in frame.env:
                return None
            value = frame.env[name]
        else:
            frame = st.frames[0]
            key = self._global_key(name, fctx.module_key)
            if key in frame.maybe_unbound or key not in frame.env:
                return None
            value = frame.env[key]
        if value.top or value.refs or value.st or value.is_bot():
            return None
        kinds = value.kinds()
        kind = "NoneType" if kinds == ["None"] else kinds[0] if len(kinds) == 1 else None
        return (value, kind) if kind in {"int", "bool", "float", "str", "NoneType"} else None

    def _prefix_comparison_safe(self, node, kinds):
        if isinstance(node, ast.BoolOp):
            return all(self._prefix_comparison_safe(item, kinds) for item in node.values)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return self._prefix_comparison_safe(node.operand, kinds)
        if isinstance(node, ast.Constant):
            return type(node.value) is bool
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            return False
        from tharkun.solve.contracts.safety import Binding, cases
        from tharkun.solve.terms import Refused, eval_term, literal, var

        operands = (node.left, *node.comparators)
        if not all(type(item) in (ast.Name, ast.Constant) for item in operands):
            return False
        values = []
        for item in operands:
            if isinstance(item, ast.Name):
                kind = kinds.get(item.id)
                if kind is None:
                    return False
                values.append(var(item.id))
            else:
                term = literal(item.value)
                if isinstance(term, Refused):
                    return False
                values.append(term)
        tested = cases("protocol.comparison", "TypeError",
                       Binding(values[0], tuple(values[1:]), type(node.ops[0]).__name__))
        if isinstance(tested, Refused):
            return False
        representatives = {"int": 0, "bool": False, "float": 0.0, "str": "", "NoneType": None}
        names = sorted({item.id for item in operands if isinstance(item, ast.Name)})
        domains = [representatives[kinds[name]] for name in names]
        env = dict(zip(names, domains))
        for case in tested:
            domain = eval_term(case.domain, env)
            if isinstance(domain, Refused):
                return False
            if domain is True:
                if case.raises is None:
                    return True
                raises = eval_term(case.raises, env)
                return raises is False
        return False

    def _observe_statement_entry(self, statement, st, fctx):
        if not self.record or st.bottom:
            return
        from tharkun.solve.sitekey import prestate_info
        for node in self._statement_entry_nodes.get(statement, ()):
            site = self._source_sites.get(node)
            binding, binding_names = self._record_binding(node, site)
            scope, classified_statement, prefix = prestate_info(node, self._site_parents)
            prefix_names = tuple(sorted({
                item.id for expr in prefix for item in ast.walk(expr) if isinstance(item, ast.Name)
            }))
            owner_names = tuple(sorted(set(binding_names) | set(prefix_names)))
            valid = (scope == "statement_entry" and classified_statement is statement
                     and binding is not None
                     and self._source_statement_matches(node, statement)
                     and self._source_scope_matches(node, fctx, owner_names))
            kinds = {}
            if valid:
                for name in owner_names:
                    value = self._entry_value(st, fctx, name, node)
                    if value is None:
                        valid = False
                        break
                    kinds[name] = value[1]
            # Prefix names are proof-only. The emitted facts remain bounded to
            # the operation's own binding, whose size is governed by the legacy cap.
            facts, exact = self._record_terms(st, binding_names) if valid else ([], False)
            valid = valid and exact and all(self._prefix_comparison_safe(expr, kinds) for expr in prefix)
            observed_builtins = self._record_builtins(st, node, site) if valid else []
            previous = self._statement_entry_observations.get(node)
            if previous is None:
                self._statement_entry_observations[node] = {
                    "valid": valid, "binding": binding if valid else None,
                    "facts": set(facts) if valid else set(), "builtins": set(observed_builtins) if valid else set(),
                }
            elif not previous["valid"] or not valid or previous["binding"] != binding:
                previous.update(valid=False, binding=None, facts=set(), builtins=set())
            else:
                previous["facts"].intersection_update(facts)
                previous["builtins"].intersection_update(observed_builtins)
            self._apply_statement_entry_observation(node)

    def _source_statement_matches(self, node, statement):
        site = self._source_sites.get(node)
        original = self._record_source_nodes.get(site.ast_path) if site is not None else None
        original_statement = self._statement_for(original, self._record_source_parents) if original is not None else None
        return (original_statement is not None
                and ast.dump(original_statement, include_attributes=True)
                == ast.dump(statement, include_attributes=True))

    def _apply_statement_entry_observation(self, node):
        observation = self._statement_entry_observations[node]
        for key in self._record_keys_by_node.get(node, ()):
            meta = self._record_meta[key]
            if meta.get("fact_scope") != "statement_entry":
                continue
            record = self.records[key]
            if not observation["valid"]:
                meta.update(binding=None, fact_scope=None)
                record.update(binding=None, facts=[], builtins=[])
                record.pop("fact_scope", None)
            else:
                record["facts"] = sorted(set(record["facts"]) & observation["facts"])
                record["builtins"] = sorted(set(record["builtins"]) & observation["builtins"])

    def _record_binding(self, node, site_key):
        from tharkun.solve.terms import PY, Refused, Term, canon, literal, var

        if self.active and isinstance(self.active[-1], ast.Lambda):
            return None, ()
        if node in self._record_binding_cache:
            return self._record_binding_cache[node]
        if site_key is not None:
            source_node = self._record_source_nodes.get(site_key.ast_path)
            while source_node in self._record_source_parents:
                source_node = self._record_source_parents[source_node]
                if isinstance(source_node, ast.Lambda):
                    return None, ()
        if site_key is None or self.lowered is not None or not self._site_parents:
            result = (None, ())
            self._record_binding_cache[node] = result
            return result
        from tharkun.solve.sitekey import prestate_info
        if prestate_info(node, self._site_parents)[0] is None:
            result = (None, ())
            self._record_binding_cache[node] = result
            return result
        op_node = None
        if isinstance(node, ast.BinOp):
            receiver, args, op_node = node.left, [node.right], node.op
        elif isinstance(node, ast.AugAssign):
            receiver, args, op_node = node.target, [node.value], node.op
        elif isinstance(node, ast.UnaryOp):
            receiver, args, op_node = node.operand, [], node.op
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and len(node.comparators) == 1:
            receiver, args, op_node = node.left, [node.comparators[0]], node.ops[0]
        else:
            result = (None, ())
            self._record_binding_cache[node] = result
            return result
        if type(receiver) not in (ast.Name, ast.Constant) or any(type(a) not in (ast.Name, ast.Constant) for a in args):
            result = (None, ())
            self._record_binding_cache[node] = result
            return result
        names = tuple(sorted({x.id for x in [receiver, *args] if isinstance(x, ast.Name)}))
        if len(names) > 4:
            result = (None, ())
            self._record_binding_cache[node] = result
            return result

        def encoded(value):
            term = var(value.id, PY) if isinstance(value, ast.Name) else literal(value.value)
            if isinstance(term, Refused) or not isinstance(term, Term):
                return None
            result = canon(term)
            return result if isinstance(result, str) else None

        recv = encoded(receiver)
        encoded_args = [encoded(arg) for arg in args]
        if recv is None or any(value is None for value in encoded_args):
            result = (None, ())
        else:
            result = ({"receiver": recv, "args": encoded_args,
                       "op": type(op_node).__name__ if op_node is not None else None}, names)
        self._record_binding_cache[node] = result
        return result

    def _record_terms(self, st: State, names) -> tuple[list[str], bool]:
        from tharkun.solve.terms import (Refused, as_bool, as_f64, as_int, canon, eq, ge, le, literal,
                                         sub, typeis, var)

        if st.bottom or len(names) > 4:
            return [], False
        facts = set()
        ints = {}
        tags = {"int", "bool", "float", "str", "NoneType"}
        exact_names = set()

        def add(term):
            if isinstance(term, Refused):
                return
            text = canon(term)
            if isinstance(text, str):
                facts.add(text)

        for name in names:
            frame = st.frames[-1]
            if name in frame.maybe_unbound:
                continue
            if name in frame.env:
                owner, value = frame, frame.env[name]
            elif name in st.frames[0].env:
                owner, value = st.frames[0], st.frames[0].env[name]
            else:
                continue
            if name in owner.maybe_unbound or value.top:
                continue
            py = var(name)
            kind = None
            if value.only_int():
                kind = "int"
                ints[name] = (owner, value.i)
            elif value.kinds() == ["bool"]:
                kind = "bool"
            elif value.kinds() == ["float"]:
                kind = "float"
            elif value.kinds() == ["str"]:
                kind = "str"
            elif value.kinds() == ["None"]:
                kind = "NoneType"
            if kind not in tags:
                continue
            exact_names.add(name)
            add(typeis(py, kind))
            if kind == "int":
                lo, hi = self.ibounds(st, value.i)
                cast = as_int(py)
                if lo is not None:
                    add(ge(cast, literal(lo)))
                if hi is not None:
                    add(le(cast, literal(hi)))
            elif kind == "bool" and len(value.b) == 1:
                add(eq(as_bool(py), literal(next(iter(value.b)))))
            elif kind == "float" and not value.f.fact.nan:
                fact = value.f.fact
                cast = as_f64(py)
                if math.isfinite(fact.lo):
                    add(ge(cast, literal(fact.lo)))
                if math.isfinite(fact.hi):
                    add(le(cast, literal(fact.hi)))
        for left in sorted(ints):
            for right in sorted(ints):
                if left == right:
                    continue
                lowner, lvalue = ints[left]
                rowner, rvalue = ints[right]
                c = st.zone.get(f"{lowner.fid}:{left}", f"{rowner.fid}:{right}")
                if c is not None:
                    add(le(sub(as_int(var(left)), as_int(var(right))), literal(c)))
        return sorted(facts), exact_names == set(names)

    def _scope_facts(self, st: State, residual: str) -> tuple[set, dict]:
        """Must-facts and types, over the names of ``residual``, in state ``st``."""
        try:
            names = sorted({n.id for n in ast.walk(ast.parse(residual, mode="eval")) if isinstance(n, ast.Name)})
        except SyntaxError:
            return set(), {}
        if st.bottom:
            return set(), {}
        fr = st.frames[-1]
        g = st.frames[0]
        facts: set = set()
        types: dict = {}
        zvars = {}
        for n in names:
            if n in fr.env:
                v, owner = fr.env[n], fr
            elif n in g.env:
                v, owner = g.env[n], g
            else:
                v, owner = None, None
            if v is None or v.top or n in owner.maybe_unbound:
                continue
            if v.only_int():
                types[n] = "int"
                zv = f"{owner.fid}:{n}"
                zvars[n] = zv
                lo, hi = st.zone.bounds(zv)
                if lo is not None:
                    facts.add(f"{n} >= {lo}")
                if hi is not None:
                    facts.add(f"{n} <= {hi}")
            elif v.kinds() == ["float"]:
                types[n] = "float"
                ff = v.f.fact
                if not ff.nan and ff.has_interval():
                    if math.isfinite(ff.lo):
                        facts.add(f"{n} >= {ff.lo!r}")
                    if math.isfinite(ff.hi):
                        facts.add(f"{n} <= {ff.hi!r}")
            elif v.kinds() == ["bool"]:
                types[n] = "bool"
            elif v.kinds() == ["str"]:
                types[n] = "str"
            elif v.kinds() == ["None"]:
                types[n] = "none"
            elif v.refs and not (v.i is not None or v.b or v.f or v.s is not None or v.st):
                types[n] = "object"
                if not v.n:
                    facts.add(f"{n} is not None")
        ks = sorted(zvars)
        for a in ks:
            for b in ks:
                if a != b:
                    c = st.zone.get(zvars[a], zvars[b])
                    if c is not None:
                        facts.add(f"{a} - {b} <= {c}")
        return facts, types

    def fresh(self, st: State) -> str:
        self.tmp += 1
        return f"${len(st.frames)}_{self.tmp}"

    def site(self, node, tag: str = "") -> tuple:
        # [internal ref]: AST nodes carry no filename of their own -- ``self._cur_module_key`` (pushed
        # / popped around a read-and-recurse call the SAME way ``callstring`` already is)
        # discriminates a site in an extern file from a same-line-and-column site in the
        # program's own file or another extern file. Always "" (unchanged) for any program
        # that never triggers read-and-recurse -- a constant prefix does not change relative
        # uniqueness, so this is a no-op for every program's own allocation-site identity.
        return (self._cur_module_key, tuple(self.callstring), getattr(node, "lineno", 0),
                getattr(node, "col_offset", 0), tag)

    def sid(self, site) -> str:
        s = self.site_ids.get(site)
        if s is None:
            s = self.site_ids[site] = f"s{len(self.site_ids)}"
            self.site_desc[s] = f"{site[3] or 'obj'}@{site[1]}:{site[2]}"
        return s

    def pretty(self, zv: str) -> str:
        if zv == ZERO:
            return "0"
        if zv.startswith("L."):
            return f"len({self.site_desc.get(zv[2:], zv[2:])})"
        if zv.startswith("R."):
            sid, part = zv[2:].rsplit(".", 1)
            return f"range@{self.site_desc.get(sid, sid)}.{part}"
        if zv.startswith("$"):
            return "tmp" + zv[1:]
        if ":" in zv:
            return zv.split(":", 1)[1]
        return zv

    def plin(self, lin: Lin | None) -> str:
        if lin is None:
            return "?"
        parts = []
        for v, c in lin.terms:
            n = self.pretty(v)
            parts.append(n if c == 1 else f"-{n}" if c == -1 else f"{c}*{n}")
        s = " + ".join(parts).replace("+ -", "- ")
        if lin.c or not s:
            s = f"{s} + {lin.c}".replace("+ -", "- ") if s else str(lin.c)
        return s

    # ------------------------------------------------------------ int helpers
    def ibounds(self, st: State, ii: IntInfo) -> tuple:
        z = st.zone
        lo = hi = None
        if ii.lin is not None:
            lo, hi = lin_bounds(ii.lin, z)
        if ii.lo is not None:
            v, c = ii.lo
            if v == ZERO:
                l2 = c
            else:
                b = z.get(ZERO, v)
                l2 = None if b is None else c - b
            if l2 is not None and (lo is None or l2 > lo):
                lo = l2
        if ii.hi is not None:
            v, c = ii.hi
            if v == ZERO:
                h2 = c
            else:
                b = z.get(v, ZERO)
                h2 = None if b is None else b + c
            if h2 is not None and (hi is None or h2 < hi):
                hi = h2
        return (lo, hi)

    def zone_assign(self, st: State, zn: str, ii: IntInfo) -> None:
        """``zn := ii`` in the zone (``zn`` must not occur in ``ii``)."""
        z = st.zone
        if z.bottom:
            return
        lo, hi = self.ibounds(st, ii)
        lin = ii.lin
        z.forget(zn)
        z.add_var(zn)
        if lin is not None:
            s = lin.single()
            if s is not None:
                if s[0] == ZERO:
                    z.assign_interval(zn, s[1], s[1])
                else:
                    if s[0] not in z.vars:
                        z.add_var(s[0])
                    z.assign_var(zn, s[0], s[1])
                return
            for t, coef in lin.terms:
                if coef != 1:
                    continue
                rest = lin.sub(Lin.var(t))
                rlo, rhi = lin_bounds(rest, z)
                if t in z.vars:
                    if rhi is not None:
                        z.add(zn, t, rhi)
                    if rlo is not None:
                        z.add(t, zn, -rlo)
        if hi is not None:
            z.add(zn, ZERO, hi)
        if lo is not None:
            z.add(ZERO, zn, -lo)
        if ii.lo is not None and ii.lo[0] != ZERO and ii.lo[0] in z.vars:
            z.add(ii.lo[0], zn, -ii.lo[1])
        if ii.hi is not None and ii.hi[0] != ZERO and ii.hi[0] in z.vars:
            z.add(zn, ii.hi[0], ii.hi[1])

    def to_temp(self, st: State, ii: IntInfo) -> IntInfo:
        t = self.fresh(st)
        self.zone_assign(st, t, ii)
        return IntInfo(Lin.var(t))

    def mat(self, st: State, v: V) -> V:
        """Materialize a value's int part into a temporary (heap reads, R2)."""
        if v.i is None:
            return v
        return v.replace(i=self.to_temp(st, v.i))

    def save_operand(self, st: State, v: V, upcoming: ast.expr) -> V:
        """Snapshot ``v`` (an already-evaluated left operand) before evaluating
        ``upcoming`` (the next operand to its right), unless ``upcoming`` is
        provably side-effect-free (a bare name or a constant).

        Python evaluates left-to-right and SAVES each operand before the next
        one runs. A right-hand operand that is a call (or contains one) can
        mutate a module global the left operand named -- e.g. in ``y >
        lower()``, ``lower()`` may reassign the module global ``y``.

        [internal ref] introduced this function to guard exactly that case (absint
        accepted the false relational invariant ``y <= x + 1`` this way, a
        wrong TRUE), because `load_name` only materialized a global read
        inside a function frame at the time. [internal ref] fixed that at the root:
        `load_name` now materializes EVERY read of a written global,
        module scope included, so ``v`` arriving here is already a fresh,
        decoupled temp whenever it came from a global read -- this function's
        own re-materialization of it is redundant for that case, not
        load-bearing. It is kept anyway for a narrower, purely precision-
        related reason unrelated to globals specifically: skipping the
        snapshot when ``upcoming`` provably cannot call anything (a bare
        name or constant) preserves an exact zone identity between ``v``
        and whatever it was derived from that a needless re-materialize
        would otherwise widen away (e.g. ``x * x``).
        """
        if isinstance(upcoming, (ast.Name, ast.Constant)):
            return v
        return self.mat(st, v)

    def summarize(self, st: State, v: V) -> V:
        """The form of a value stored in a summary: symbolic bounds, no lin."""
        ii = v.i
        if ii is None or ii.lin is None:
            return v
        s = ii.lin.single()
        if s is not None:
            if s[0] != ZERO and s[0].split(":", 1)[0] == "g" and self.pretty(s[0]) in self.global_written:
                lo, hi = self.ibounds(st, ii)
                return v.replace(i=IntInfo(None, None if lo is None else (ZERO, lo), None if hi is None else (ZERO, hi)))
            return v.replace(i=IntInfo(None, s, s))
        lo, hi = self.ibounds(st, ii)
        return v.replace(i=IntInfo(None, None if lo is None else (ZERO, lo), None if hi is None else (ZERO, hi)))

    def _const_bounds(self, st: State, v: V, sid: str) -> V:
        ii = v.i
        if ii is None:
            return v
        lo, hi = self.ibounds(st, ii)
        return v.replace(i=IntInfo(None, None if lo is None else (ZERO, lo), None if hi is None else (ZERO, hi)))

    def join_into(self, st: State, old: V, new: V) -> V:
        return vjoin(old, st.zone, self.summarize(st, new), st.zone)

    def fresh_pass(self) -> None:
        """A pass over a loop (the first iteration, a fixpoint step, the final pass) starts where a statement
        starts: what an unfinished statement left behind (the allocation epoch, the borrowed references of this
        frame) belongs to the pass before it, and must not make this one refuse an allocation it is entitled to
        (``gc`` does the same after every statement)."""
        self.epoch = set()
        self.borrowed[-1].clear()

    # --------------------------------------------------------------- the GC
    def gc(self, st: State) -> None:
        self.epoch = set()
        self.borrowed[-1].clear()
        if st.bottom:
            return
        depth = len(st.frames)
        for v in [v for v in st.zone.vars if v.startswith("$")]:
            d = int(v[1:].split("_", 1)[0])
            if d >= depth:
                st.kill_var(v)
        if depth > 1:
            # callers may hold in-flight references: collect the heap only at module level
            return
        # unreachable heap objects
        seen = set()
        work = []

        def push_v(v: V):
            for a in v.refs:
                if a not in seen:
                    seen.add(a)
                    work.append(a)

        for f in st.frames:
            for v in f.env.values():
                push_v(v)
        for v in st.pins:
            push_v(v)
        while work:
            a = work.pop()
            ob = st.heap.get(a)
            if ob is None:
                continue
            for v in ob.values():
                push_v(v)
            if ob.kind == "inst":
                if ob.cls not in seen:
                    seen.add(ob.cls)
                    work.append(ob.cls)
            elif ob.kind == "class":
                for b in ob.bases:
                    if b not in seen:
                        seen.add(b)
                        work.append(b)
            elif ob.kind == "bmeth":
                if ob.func not in seen:
                    seen.add(ob.func)
                    work.append(ob.func)
            elif ob.kind == "dictview":
                # a view is live (it sees the dict's later changes): the dicts it names are reachable through it
                for d in ob.dicts:
                    if d not in seen:
                        seen.add(d)
                        work.append(d)
        dead = [a for a in st.heap if a not in seen]
        if dead:
            zvs = []
            for a in dead:
                zvs.extend(st.heap[a].zvars())
                del st.heap[a]
            for zv in zvs:
                st.kill_var(zv)

    # -------------------------------------------------------------- the heap
    def alloc(self, st: State, node, make, tag: str):
        site = self.site(node, tag)
        if site in self.epoch:
            raise Unsupported("one allocation site reached twice within a statement", node)
        self.epoch.add(site)
        sid = self.sid(site)
        r = (site, "r")
        if r in st.heap:
            self.demote(st, site)
        st.heap[r] = make(sid)
        return r

    def borrow(self, st: State, value: V) -> V:
        # Python locals in a transfer are not rewritten by State.redirect.
        # Reachable fields remain in the heap and are rewritten there.
        self.borrowed[-1].update(addr[0] for addr in value.refs if addr[1] == "r")
        return value

    def demote(self, st: State, site) -> None:
        # An allocation at a site whose recent object a Python local of an enclosing transfer function still
        # holds is refused: the local is not rewritten when the object becomes a summary.  Every holder is
        # registered in the scope of the evaluation (or frame) whose locals they are (``eval``, ``call_func``),
        # so a value is held exactly while the code that produced it for its consumer is still running.
        if any(site in borrowed for borrowed in self.borrowed):
            raise Unsupported("allocation would invalidate an in-flight heap reference")
        r, s = (site, "r"), (site, "s")
        st.strip_terms(lambda t: t == r)
        ob = st.heap.pop(r)
        z = st.zone
        zvs = ob.zvars()
        if ob.kind == "inst":
            ob = InstObj(ob.cls, {k: (v, d, None) for k, (v, d, _r) in ob.attrs.items()}, ob.qf)
        if ob.kind == "list" and ob.pre is not None:
            ob = ListObj(ob.length, vjoin(ob.elem, z, ob.pre, z), None)
        if ob.kind in ("list", "tuple", "array") and isinstance(ob.length, str):
            ob = ob.replace(length=z.bounds(ob.length))
        elif ob.kind == "range":
            ob = RangeObj(z.bounds(ob.lo) if isinstance(ob.lo, str) else ob.lo,
                          z.bounds(ob.hi) if isinstance(ob.hi, str) else ob.hi, ob.step)
        try:
            st.heap[s] = ob if s not in st.heap else _ojoin(st.heap[s], z, ob, z)
        except Incompatible as e:
            raise Unsupported(f"summary join: {e}")
        st.redirect(r, s)
        for zv in zvs:
            st.kill_var(zv)

    def detach(self, st: State, v: V, sid: str) -> V:
        """Rewrite an in-flight value away from the zone variables of site ``sid``
        (they are about to name a *new* object at that site)."""
        ii = v.i
        if ii is None:
            return v
        own = lambda x: x == f"L.{sid}" or x.startswith(f"R.{sid}.")
        if not any(own(x) for x in ii.vars()):
            return v
        return v.replace(i=self.to_temp(st, ii))

    def new_list(self, st: State, node, length_ii: IntInfo | None, elem: V, tag="list", kind="list", items=None,
                 seg: bool = False, typecode=None):
        sid0 = self.sid(self.site(node, tag))
        if length_ii is not None:
            length_ii = self.detach(st, V(i=length_ii), sid0).i
        if elem.i is not None and any(x == f"L.{sid0}" or x.startswith(f"R.{sid0}.") for x in elem.i.vars()):
            # constant bounds only: materializing a summary would assert an element exists
            elem = self._const_bounds(st, elem, sid0)
        if items is not None:
            items = [self.detach(st, x, sid0) for x in items]

        def make(sid):
            if kind == "tuple" and items is not None:
                return TupleObj(tuple(self.summarize(st, x) for x in items))
            lv = f"L.{sid}"
            if length_ii is None:
                st.zone.forget(lv)
                st.zone.add_var(lv)
                st.zone.add(ZERO, lv, 0)
            else:
                self.zone_assign(st, lv, length_ii)
                st.zone.add(ZERO, lv, 0)
            if kind == "tuple":
                return TupleObj(None, lv, elem)
            if kind == "array":
                return ArrayObj(lv, elem, typecode)
            if seg:
                # a segmented list starts with an empty prefix: P = 0
                st.zone.assign_interval(seg_var(lv), 0, 0)
                return ListObj(lv, elem, BOT)
            return ListObj(lv, elem)

        return self.alloc(st, node, make, tag)

    def len_lin(self, st: State, ob) -> tuple:
        """(lin or None, (lo, hi)) of a sequence object's length."""
        if ob.kind == "tuple" and ob.items is not None:
            n = len(ob.items)
            return Lin.const(n), (n, n)
        if isinstance(ob.length, str):
            return Lin.var(ob.length).subst_consts(st.zone), st.zone.bounds(ob.length)
        lo, hi = ob.length
        return (Lin.const(lo) if lo is not None and lo == hi else None), (lo, hi)

    def update_obj(self, st: State, addrs, fn) -> None:
        """Apply ``fn(obj, strong) -> obj`` to every object in ``addrs``."""
        strong = len(addrs) == 1 and next(iter(addrs))[1] == "r"
        for a in addrs:
            ob = st.heap.get(a)
            if ob is None:
                continue
            nob = fn(ob, strong)
            if not strong:
                try:
                    nob = _ojoin(ob, st.zone, nob, st.zone)
                except Incompatible as e:
                    raise Unsupported(f"weak update: {e}")
            st.heap[a] = nob

    def set_len(self, st: State, ob, strong: bool, lin: Lin | None, lo=None, hi=None):
        """A new length for a list object; returns the updated object."""
        if isinstance(ob.length, str) and strong:
            lv = ob.length
            ii = IntInfo(lin) if lin is not None else IntInfo(None, None if lo is None else (ZERO, lo), None if hi is None else (ZERO, hi))
            s = lin.single() if lin is not None else None
            # ``ob`` is the replacement the caller built from the stored object and is about to store: its
            # summaries name ``lv`` by its old value, so they move with the stored ones (the in-flight object
            # is passed to the State method; a bound left naming the old length is a stale bound, and an
            # append then read an element one length too high or too low)
            if s is not None and s[0] == lv:
                st.zone.assign_var(lv, lv, s[1])
                ob = st.shift_bounds(lv, s[1], ob)
            else:
                if lin is not None and lv in lin.vars():
                    t = self.fresh(st)
                    st.zone.assign_var(t, lv, 0)
                    ii = IntInfo(_rename_lin(lin, lv, t))
                ob = st.rewrite_bounds_away(lv, ob)
                self.zone_assign(st, lv, ii)
            st.zone.add(ZERO, lv, 0)
            return ob
        # a summary (or weak) length: an interval
        if lin is not None:
            lo, hi = lin_bounds(lin, st.zone)
        lo = 0 if lo is None or lo < 0 else lo
        if isinstance(ob.length, str):
            # weak update of a recent object reached through several references:
            # the old length stays possible
            olo, ohi = st.zone.bounds(ob.length)
            ob = st.rewrite_bounds_away(ob.length, ob)
            nlo = None if olo is None else min(olo, lo)
            nhi = None if ohi is None or hi is None else max(ohi, hi)
            st.zone.assign_interval(ob.length, nlo, nhi)
            st.zone.add(ZERO, ob.length, 0)
            return ob
        return ob.replace(length=(lo, hi))

    # ============================================================== entry
    def run(self) -> None:
        st = State()
        st.frames.append(Frame("g"))
        loc, _ = _assigned_names(self.tree.body)
        ctx = FuncCtx(self.tree, loc, set(), is_module=True)
        self.module_ctx = ctx
        lemmas.reset_landmarks()
        st.frame.env["__name__"] = const_str("__main__")
        st.frame.env["__file__"] = ANYSTR
        st.frame.env["__doc__"] = V(s=ANY_STR, n=True)
        # runpy supplies these globals; treating them as NameError would erase real paths.
        for name in ("__builtins__", "__package__", "__spec__", "__loader__", "__cached__"):
            st.frame.env[name] = V(top=True)
        self._fctx_stack.append(ctx)
        try:
            self.exec_block(self.tree.body, st, ctx, None)
        finally:
            self._fctx_stack.pop()

    # ============================================================ statements
    def exec_block(self, stmts, st: State, fctx: FuncCtx, lctx: LoopCtx | None) -> State:
        for idx, s in enumerate(stmts):
            if st.bottom:
                return st
            if self.split_draws and idx + 1 < len(stmts):
                draw = self._draw_to_split(s, fctx, lctx)
                if draw is not None:
                    return self._exec_split(stmts, idx, st, fctx, lctx, draw)
            if self._try_stack:
                # [internal ref]: while a try statement is active, every path that ends must end by a modelled exit
                # (an exception routed to the handlers, a return/break/continue): a statement that ends
                # every path silently (a NameError, a non-property raise the site does not route, ...) would
                # hide the continuation a handler gives it.  Refused by name, never guessed.
                before = self._exit_count
                st = self.exec_stmt(s, st, fctx, lctx)
                if st.bottom and self._exit_count == before:
                    raise Unsupported("a statement inside a try ends every path without a modelled exception "
                                      "or jump", s)
            else:
                st = self.exec_stmt(s, st, fctx, lctx)
            if not st.bottom:
                self.gc(st)
        return st

    # ---------------------------------------------------- case split on a boolean draw
    def _draw_to_split(self, s, fctx: FuncCtx, lctx):
        """The one ``_sv_verifier.nondet_bool()`` call of statement ``s`` that a case split may force, or ``None``.

        Eligible: an ``Assign``/``AnnAssign`` value, an ``Expr`` value or an ``If`` test; recording passes only (a fixpoint
        step is not the pass that proves), never inside a loop body or a try (a draw's refusal is routed there), and only
        while the split budget lasts.  The call must sit where the statement evaluates it EXACTLY ONCE whenever it
        evaluates it at all (:meth:`_once_draws`).  Two or more such calls, or none, give ``None``."""
        if (not self.record or lctx is not None or self._try_stack or self._split_budget <= 0
                or fctx.module_key != "" or not isinstance(s, (ast.Assign, ast.AnnAssign, ast.Expr, ast.If))):
            return None
        root = s.test if isinstance(s, ast.If) else s.value
        if root is None:
            return None
        gov = self._governing_resolver(fctx)
        if gov is None:
            return None
        found: list = []
        self._once_draws(root, gov, found)
        return found[0] if len(found) == 1 else None

    def _once_draws(self, e, gov, found: list) -> None:
        """Collect into ``found`` the ``nondet_bool()`` calls of expression ``e`` that are evaluated at most once per
        evaluation of ``e``.  It descends only through operands Python always evaluates, left to right: a call's
        callee, arguments and keyword values, an operator's operands, the FIRST comparator of a comparison and the FIRST
        operand of ``and``/``or`` (later ones are conditional), the test of a conditional expression (not its arms), the
        base and index of an attribute or subscript, the elements of a display.  A lambda, a comprehension, a
        conditional arm and everything else is not entered."""
        if isinstance(e, ast.Call):
            if gov.call_name(e) == "_sv_verifier.nondet_bool":
                if not e.args and not e.keywords:
                    found.append(e)
                return
            self._once_draws(e.func, gov, found)
            for a in e.args:
                self._once_draws(a.value if isinstance(a, ast.Starred) else a, gov, found)
            for k in e.keywords:
                self._once_draws(k.value, gov, found)
        elif isinstance(e, ast.BinOp):
            self._once_draws(e.left, gov, found)
            self._once_draws(e.right, gov, found)
        elif isinstance(e, ast.UnaryOp):
            self._once_draws(e.operand, gov, found)
        elif isinstance(e, ast.Compare):
            self._once_draws(e.left, gov, found)
            self._once_draws(e.comparators[0], gov, found)
        elif isinstance(e, ast.BoolOp):
            self._once_draws(e.values[0], gov, found)
        elif isinstance(e, ast.IfExp):
            self._once_draws(e.test, gov, found)
        elif isinstance(e, ast.Attribute):
            self._once_draws(e.value, gov, found)
        elif isinstance(e, ast.Subscript):
            self._once_draws(e.value, gov, found)
            self._once_draws(e.slice, gov, found)
        elif isinstance(e, (ast.Tuple, ast.List, ast.Set)):
            for x in e.elts:
                self._once_draws(x.value if isinstance(x, ast.Starred) else x, gov, found)
        elif isinstance(e, ast.Dict):
            for k, v in zip(e.keys, e.values):
                if k is not None:
                    self._once_draws(k, gov, found)
                self._once_draws(v, gov, found)

    def _activation(self, st: State) -> tuple:
        """Which activation of which code is executing: the frame depth and the call string.  A callee, a deeper
        activation of the same function and a sibling call all differ in it."""
        return len(st.frames), tuple(self.callstring)

    def _exec_split(self, stmts, idx: int, st: State, fctx: FuncCtx, lctx, draw) -> State:
        """Run ``stmts[idx]`` once with ``draw`` forced to False and once forced to True, then the rest of the block
        (``stmts[idx + 1:]``) on each outcome; the two end states are joined only here, at the end of the block.  The
        draw takes exactly the values {False, True}, so the two runs cover every execution, and the join of their end
        states is what the block's joined analysis would have produced, with the correlation between the outcome and
        everything computed after it kept apart until then.  Returns, raises and the other exits the rest of the
        block routes are appended to the same lists from both outcomes, so the caller joins them as it always did.

        The force belongs to this activation (:meth:`_activation`) and to this one call node: ``call_func`` hides it
        from every callee, ``_fixpoint`` and the class-contract memo write nothing while one is pending, and
        ``capture_invariant`` records nothing while the outcomes run apart."""
        self._split_budget -= 1
        self.splits_done += 1
        epoch, held = set(self.epoch), set(self.borrowed[-1])
        act = self._activation(st)
        outs = []
        self.capture_invariant(stmts[idx], st)   # the entry of the split statement is a single, unsplit state
        self._partition_depth += 1
        try:
            for forced in (False, True):
                # a run starts where the statement started: what the other run left behind belongs to that run
                self.epoch, self.borrowed[-1] = set(epoch), set(held)
                sub = st.copy()
                self._forced_draws[draw] = (forced, act)
                try:
                    sub = self.exec_stmt(stmts[idx], sub, fctx, lctx)
                finally:
                    self._forced_draws.pop(draw, None)
                if not sub.bottom:
                    self.gc(sub)
                outs.append(self.exec_block(stmts[idx + 1:], sub, fctx, lctx))
        finally:
            self._partition_depth -= 1
        return outs[0].join(outs[1])

    def exec_stmt(self, s, st: State, fctx: FuncCtx, lctx) -> State:
        if self._clock() > self.deadline:
            raise Unsupported("analysis time budget exhausted", s)
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and getattr(s, "type_params", ()):
            raise Unsupported("PEP 695 type parameters are not modelled", s)
        type_alias = getattr(ast, "TypeAlias", None)
        if type_alias is not None and isinstance(s, type_alias):
            raise Unsupported("type alias statements are not modelled", s)
        if isinstance(s, ast.ClassDef) and _class_has_private_name_mangling(s):
            raise Unsupported("private name mangling in class scope is not modelled", s)
        m = getattr(self, "s_" + type(s).__name__, None)
        if m is None:
            raise Unsupported(f"statement {type(s).__name__}", s)
        if self.record and not isinstance(s, ast.While) and s in self._statement_entry_nodes:
            self._observe_statement_entry(s, st, fctx)
        self.capture_invariant(s, st)
        return m(s, st, fctx, lctx)

    def capture_invariant(self, node, st, *, loop=False):
        """Candidate scalar predicates, never trusted by the witness checker."""
        if not self.record or not self.invariant_locations or st.bottom or self._partition_depth:
            return  # (the outcomes of a case split run apart: an invariant is only taken from a joined state)
        if getattr(node, ROLE_ATTR, None) is not None:
            return  # a statement the lowering invented: the original program has no such point
        if isinstance(node, ast.While) and not loop:
            return  # Its witness point is the fixpoint head, not the first entry.
        entry = self.invariant_points.get(node)
        if entry is None:
            # [internal ref]/MG's PR 86 review: this analyzer's invariant_points is built ONLY
            # from the top-level program's own tree (Analyzer.__init__), but exec_stmt
            # calls capture_invariant for every statement it executes, including a
            # recursively-read extern file's top level (_exec_extern_toplevel) once
            # read-and-recurse crosses a function/package boundary -- an unconditional
            # ``self.invariant_points[node]`` KeyErrors on that node instead. It is safe
            # to skip: invariant_locations (scheduler.py's obligation_census /
            # source_points) is built from the SAME top-level source, so a location in
            # an extern file could never be requested here regardless.
            return
        loc = entry[1]
        if loc not in self.invariant_locations:
            return
        parts = set()
        hidden = (set(self.lowered.renamed) | set(self.lowered.synthetic)) if self.lowered is not None else ()
        for name, value in st.frames[-1].env.items():
            if name in hidden or name in st.frames[-1].maybe_unbound:
                continue  # a variable of the lowering, not of the original program
            if value.only_int():
                parts.add(f"type({name}) is int")
                lo, hi = self.ibounds(st, value.i)
                if lo is not None:
                    parts.add(f"{name} >= {lo}")
                if hi is not None:
                    parts.add(f"{name} <= {hi}")
            elif value.kinds() in (["float"], ["bool"]):
                kind = value.kinds()[0]
                parts.add(f"type({name}) is {kind}")
                if kind == "float" and not value.f.fact.nan:
                    fact = value.f.fact
                    if math.isfinite(fact.lo):
                        parts.add(f"{name} >= {fact.lo!r}")
                    if math.isfinite(fact.hi):
                        parts.add(f"{name} <= {fact.hi!r}")
        key = loc, "loop_invariant" if loop else "location_invariant"
        if key in self.invariants:
            self.invariants[key].intersection_update(parts)
        else:
            self.invariants[key] = parts

    def s_Pass(self, s, st, fctx, lctx):
        return st

    def s_Expr(self, s, st, fctx, lctx):
        if isinstance(s.value, ast.Constant) and isinstance(s.value.value, str):
            return st  # docstring
        if self._is_root_establishing_insert(s):
            # [internal ref] (a)/(b): the EXACT sys.path.insert(0, D) shape extern.py itself already
            # trusts to establish a read-and-recurse root -- its own import resolution is
            # handled entirely statically (extern.py, driven off the program's own AST, not
            # by tracking a runtime "sys.path" object here). The call itself has no property-
            # relevant effect once recognised this precisely, so it is a no-op here, not
            # "call of an unknown value" -- `sys` is not modelled at all otherwise, and this
            # is the ONLY sys.path shape given that trust; anything else still refuses exactly
            # as before.
            return st
        self.eval(s.value, st, fctx)
        return st

    def _is_root_establishing_insert(self, s: ast.Expr) -> bool:
        from tharkun.solve import extern as _extern
        file_dir = os.path.dirname(self.filename) if self.filename and os.path.isabs(self.filename) else None
        return _extern.is_root_establishing_call(s.value, file_dir)

    def s_Global(self, s, st, fctx, lctx):
        return st

    def s_Assign(self, s, st, fctx, lctx):
        v = self.eval(s.value, st, fctx)
        for t in s.targets:
            if st.bottom:
                break
            self.assign(t, v, st, fctx)
        return st

    def s_AnnAssign(self, s, st, fctx, lctx):
        if s.value is None:
            return st
        v = self.eval(s.value, st, fctx)
        self.assign(s.target, v, st, fctx)
        return st

    def s_AugAssign(self, s, st, fctx, lctx):
        t = s.target
        if isinstance(t, ast.Name):
            cur = self.borrow(st, self.load_name(t.id, t, st, fctx))
            v = self.binop(st, s.op, cur, self.eval(s.value, st, fctx), s, inplace=True)
            if not st.bottom:
                self.store_name(t.id, v, st, fctx)
            return st
        if isinstance(t, ast.Subscript):
            obj = self.eval(t.value, st, fctx)
            idx = self.eval_index(t.slice, st, fctx)
            cur = self.borrow(st, self.subscript_load(st, obj, idx, t))
            rhs = self.eval(s.value, st, fctx)
            v = self.binop(st, s.op, cur, rhs, s, inplace=True)
            if not st.bottom:
                self.subscript_store(st, obj, idx, v, t)
            return st
        if isinstance(t, ast.Attribute):
            obj = self.eval(t.value, st, fctx)
            cur = self.borrow(st, self.attr_load(st, obj, t.attr, t))
            rhs = self.eval(s.value, st, fctx)
            v = self.binop(st, s.op, cur, rhs, s, inplace=True)
            if not st.bottom:
                self.attr_store(st, obj, t.attr, v, t)
            return st
        raise Unsupported("augmented assignment target", s)

    def s_If(self, s, st, fctx, lctx):
        t, f = self.cond(s.test, st, fctx)
        t = self.exec_block(s.body, t, fctx, lctx) if not t.bottom else t
        f = self.exec_block(s.orelse, f, fctx, lctx) if not f.bottom else f
        return t.join(f)

    # ============================================================ exits ([internal ref])
    def _finally_frame(self, fctx: FuncCtx, lctx=_ANY_LOOP):
        """The innermost try frame of ``fctx`` whose finally a leaving exit still has to run: a return runs
        every enclosing finally of the function; a break or continue only those of the frames inside its
        own loop (a try around the loop is not left by them)."""
        for fr in reversed(self._try_stack):
            if (fr.fctx is fctx and fr.finalizer is not None and not fr.in_finally
                    and (lctx is _ANY_LOOP or fr.lctx is lctx)):
                return fr
        return None

    def _route_return(self, st: State, v: V, fctx: FuncCtx) -> None:
        if st.bottom:
            return
        self._exit_count += 1
        fr = self._finally_frame(fctx)
        if fr is not None:
            fr.pending_returns.append((st, v))
        else:
            fctx.returns.append((st, v))

    def _route_break(self, st: State, fctx: FuncCtx, lctx) -> None:
        if st.bottom:
            return
        self._exit_count += 1
        fr = self._finally_frame(fctx, lctx)
        if fr is not None:
            fr.pending_breaks.append(st)
        else:
            lctx.breaks.append(st)

    def _route_continue(self, st: State, fctx: FuncCtx, lctx) -> None:
        if st.bottom:
            return
        self._exit_count += 1
        fr = self._finally_frame(fctx, lctx)
        if fr is not None:
            fr.pending_continues.append(st)
        else:
            lctx.continues.append(st)

    def s_Return(self, s, st, fctx, lctx):
        if fctx.is_module or fctx.is_class:
            raise Unsupported("return outside a function", s)
        v = self.eval(s.value, st, fctx) if s.value is not None else NONE
        if not st.bottom:
            self._route_return(st.copy(), v, fctx)
        st.set_bottom()
        return st

    def s_Break(self, s, st, fctx, lctx):
        self._route_break(st.copy(), fctx, lctx)
        st.set_bottom()
        return st

    def s_Continue(self, s, st, fctx, lctx):
        self._route_continue(st.copy(), fctx, lctx)
        st.set_bottom()
        return st

    # ============================================================ exceptions ([internal ref])
    def _in_try_refuse(self, what: str, node) -> None:
        """A raise the model does not route: a non-property class whose path simply ends, a draw the
        harness rejects, a rule whose raises are not routed.  Outside a try it is what it always was;
        inside one the handlers would have to see it, so the analysis is refused by name."""
        if self._try_stack:
            raise Unsupported(f"{what} inside a try is not modelled", node)

    def _route_exc(self, exc: _Exc, fctx: FuncCtx) -> None:
        """Deliver an exception in flight: to the innermost try frame of ``fctx`` that receives it (its
        body: the handlers are tried when the body is done; its handler or else block, with a finally:
        the finally runs, then the exception continues outward), else out of the function
        (``fctx.raises``: ``call_func`` delivers it to the caller), else, at module level, it is
        uncaught: the site's check is recorded now, with the property classes the exception is."""
        if exc.st.bottom:
            return
        self._exit_count += 1
        if exc.val.refs and exc.val not in exc.st.pins:
            exc.st.pins = exc.st.pins + (exc.val,)    # the instance survives gc while in flight
        for fr in reversed(self._try_stack):
            if fr.fctx is not fctx or fr.in_finally:
                continue
            if fr.has_handlers and not fr.in_handler:
                fr.raised.append(exc)
                return
            if fr.finalizer is not None:
                fr.unhandled.append(exc)
                return
        if not fctx.is_module:
            fctx.raises.append(exc)
            return
        excs = self._prop_ancestors(exc.st, exc.cls)
        if isinstance(exc.cls, str):
            excs.add(exc.cls)
        fact = exc.fact
        if exc.op == "raise":
            prop = sorted(self._prop_ancestors(exc.st, exc.cls))
            fact = f"may raise {prop}" if prop else "raises a non-property exception"
        self._record_check(exc.node, exc.op, excs, False, fact, residual=exc.residual, operand=exc.operand,
                           st=exc.st)

    def _exc_base_of(self, st: State, addr):
        """The builtin exception class a user class derives from (``s_ClassDef`` records it as
        ``__exc_base__``), ``None`` for a class that is not an exception class.  A chain naming two
        different builtin bases cannot be matched exactly by one name: refused."""
        base = None
        c = addr
        while c is not None:
            co = st.heap[c]
            if "__exc_base__" in co.attrs:
                b = next(iter(co.attrs["__exc_base__"].st))[1]
                if base is not None and b != base:
                    raise Unsupported("a user exception class with two builtin exception bases", None)
                base = b
            c = co.bases[0] if co.bases else None
        return base

    @staticmethod
    def _plain_exception_constructor(name: str) -> bool:
        # CPython 3.14.4 Objects/exceptions.c: these use BaseException_new/init.
        # Python exposes distinct wrappers for the same C slot, so descriptor identity
        # is not its identity. Specialized constructors remain outside this subset.
        return name in {
            "BaseException", "Exception", "TypeError", "ValueError", "RuntimeError",
            "LookupError", "IndexError", "KeyError", "AssertionError", "ArithmeticError",
            "ZeroDivisionError", "FloatingPointError", "OverflowError",
        }

    def _is_subclass(self, st: State, exc, target) -> bool:
        """``exc`` is ``target`` or derives from it (each a builtin name or a user class address)."""
        if isinstance(target, str):
            tobj = getattr(_builtins, target, None)
            if not isinstance(tobj, type):
                return False
            if isinstance(exc, str):
                eobj = getattr(_builtins, exc, None)
                return isinstance(eobj, type) and issubclass(eobj, tobj)
            base = self._exc_base_of(st, exc)
            bobj = getattr(_builtins, base, None) if base is not None else None
            return isinstance(bobj, type) and issubclass(bobj, tobj)
        if isinstance(exc, str):
            return False      # a builtin class never derives from a user class
        return self.subclass(st, exc, target)

    def _prop_ancestors(self, st: State, cls) -> set:
        """The property classes an exception of class ``cls`` is (itself and its property ancestors)."""
        if isinstance(cls, str):
            return set(_EXC_ANCESTORS.get(cls, ()))
        base = self._exc_base_of(st, cls)
        return set(_EXC_ANCESTORS.get(base, ())) if base is not None else set()

    def _exc_name(self, st: State, cls) -> str:
        return cls if isinstance(cls, str) else st.heap[cls].name

    def s_Raise(self, s, st, fctx, lctx):
        if s.cause is not None:
            raise Unsupported("exception chaining effects on state are not modelled", s)
        if s.exc is None:
            # a bare raise re-raises the exception being handled.  Inside a finally block that is the
            # exception in flight (if any), which is not tracked: refused.
            if any(fr.in_finally for fr in self._try_stack):
                raise Unsupported("bare raise inside a finally block", s)
            if not self._active_exc_stack:
                raise Unsupported("bare raise outside an except handler", s)
            if st.bottom:
                return st
            active = self._active_exc_stack[-1]
            self._route_exc(_Exc(st.copy(), active.cls, active.val, s, "raise",
                                 f"re-raise of {self._exc_name(st, active.cls)}"), fctx)
            st.set_bottom()
            return st
        v = self.eval(s.exc, st, fctx)
        if st.bottom:
            return st
        for alt in self._raise_alternatives(v, s):
            rs = st.copy()
            cls, val = self._materialize_raise(rs, alt, s)
            if rs.bottom:
                continue      # constructing the exception ended the path (its own raise is routed)
            self._route_exc(_Exc(rs, cls, val, s, "raise", f"raises {self._exc_name(rs, cls)}"), fctx)
        st.set_bottom()
        return st

    def _raise_alternatives(self, v: V, node) -> list:
        """What ``raise v`` raises, one entry per alternative of ``v``: a builtin class or instance tag,
        a heap object (an instance, or a class that is instantiated), or a TypeError (not an exception)."""
        if v.top:
            raise Unsupported("raise of an unknown value", node)
        alts = []
        for t in v.st:
            if t[0] in ("exc", "excinst"):
                if t[1] in _EXC_GROUPS:
                    raise Unsupported("except* / ExceptionGroup is not modelled", node)
                alts.append(t)
            else:
                alts.append(("typeerror",))
        for a in v.refs:
            alts.append(("ref", a))
        if v.i is not None or v.b or v.f or v.s is not None or v.n:
            alts.append(("typeerror",))
        if not alts:
            alts.append(("typeerror",))
        return alts

    def _materialize_raise(self, st: State, alt, node):
        """The class and instance value of one alternative, built in ``st`` (a raised class is instantiated
        with no arguments, running its ``__init__`` like CPython does)."""
        kind = alt[0]
        if kind == "exc":
            if not self._plain_exception_constructor(alt[1]):
                self._in_try_refuse(f"{alt[1]} constructor", node)
            return alt[1], V(st=frozenset({("excinst", alt[1])}))
        if kind == "excinst":
            return alt[1], V(st=frozenset({alt}))
        if kind == "ref":
            ob = st.heap[alt[1]]
            if ob.kind == "inst" and self._exc_base_of(st, ob.cls) is not None:
                return ob.cls, V(refs=frozenset({alt[1]}))
            if ob.kind == "class" and self._exc_base_of(st, alt[1]) is not None:
                return alt[1], self.instantiate(st, alt[1], [], {}, node)
        return "TypeError", V(st=frozenset({("excinst", "TypeError")}))   # exceptions derive from BaseException

    # ============================================================ try ([internal ref])
    def s_TryStar(self, s, st, fctx, lctx):
        raise Unsupported("except* / ExceptionGroup is not modelled", s)

    def s_Try(self, s: ast.Try, st: State, fctx: FuncCtx, lctx: LoopCtx | None) -> State:
        """``try``/``except``/``else``/``finally``, exactly (docs/ABSINT.md, "try/except/else/finally")."""
        for h in s.handlers:
            if isinstance(h.type, ast.Name) and h.type.id in _EXC_GROUPS:
                raise Unsupported("except* / ExceptionGroup is not modelled", h)
        finalizer = (lambda x: self.exec_block(s.finalbody, x, fctx, lctx)) if s.finalbody else None
        frame = _TryFrame(s, fctx, lctx, finalizer, bool(s.handlers))
        self._try_stack.append(frame)
        end = self.exec_block(s.body, st, fctx, lctx)
        normal = None
        if not end.bottom:
            if s.orelse:
                frame.in_handler = True      # an exception in else is not for this statement's handlers
                try:
                    end = self.exec_block(s.orelse, end, fctx, lctx)
                finally:
                    frame.in_handler = False
            if not end.bottom:
                normal = end
        raised, frame.raised = frame.raised, []
        for exc in raised:
            h_end = self._dispatch_handlers(s, frame, exc, fctx, lctx)
            if h_end is not None:
                normal = h_end if normal is None else normal.join(h_end)
        out = self._finish_frame(frame, normal)
        if out is not None:
            return out
        st.set_bottom()
        return st

    def _handler_targets(self, h: ast.ExceptHandler, st: State, fctx):
        """What an except clause catches, evaluated in the state of the exception: ``None`` for a bare
        ``except``, the list of classes (builtin names or user class addresses) of a class or a tuple of
        classes, or ``"typeerror"`` when the clause names something that is not an exception class
        (CPython then raises TypeError in place of the exception).  A tuple expression is read element
        by element (no tuple is built: a site may not allocate twice in one statement); a clause whose
        value is not a single class is refused."""
        if h.type is None:
            return None
        if isinstance(h.type, ast.Tuple):
            items = [self.eval(e, st, fctx) for e in h.type.elts]
        else:
            tv = self.eval(h.type, st, fctx)
            if st.bottom:
                return None
            if len(tv.refs) == 1 and not (tv.top or tv.st or tv.i is not None or tv.b or tv.f or tv.s is not None
                                          or tv.n) and st.heap[next(iter(tv.refs))].kind == "tuple":
                ob = st.heap[next(iter(tv.refs))]
                if ob.items is None:
                    raise Unsupported("except clause: a tuple of unknown items", h.type)
                items = list(ob.items)
            else:
                items = [tv]
        if st.bottom:
            return None
        out = []
        for v in items:
            t = self._target_of(v, st, h.type)
            if t == "typeerror":
                return "typeerror"
            out.append(t)
        return out

    def _target_of(self, v: V, st: State, node):
        if v.top:
            raise Unsupported("except clause type is an unknown value", node)
        alts = len(v.st) + len(v.refs) + (1 if (v.i is not None or v.b or v.f or v.s is not None or v.n) else 0)
        if alts != 1:
            raise Unsupported("except clause type is not a single value", node)
        if v.st:
            t = next(iter(v.st))
            if t[0] != "exc":
                return "typeerror"
            if t[1] in _EXC_GROUPS:
                raise Unsupported("except* / ExceptionGroup is not modelled", node)
            return t[1]
        if v.refs:
            a = next(iter(v.refs))
            ob = st.heap[a]
            if ob.kind == "class" and self._exc_base_of(st, a) is not None:
                return a
        return "typeerror"

    def _dispatch_handlers(self, s: ast.Try, frame: _TryFrame, exc: _Exc, fctx, lctx):
        """Match one exception against the clauses in order; the end state of the handler that took it,
        or ``None`` (no clause matched, or the handler ended every path: those exits are pending)."""
        for h in s.handlers:
            targets = self._handler_targets(h, exc.st, fctx)
            if exc.st.bottom:
                raise Unsupported("evaluating an except clause type ends the path", h)
            if targets == "typeerror":
                frame.unhandled.append(_Exc(exc.st, "TypeError", V(st=frozenset({("excinst", "TypeError")})),
                                            h.type, "except-type",
                                            "the except clause names a non-exception: TypeError replaces the exception"))
                return None
            if targets is None or any(self._is_subclass(exc.st, exc.cls, t) for t in targets):
                return self._run_handler(h, frame, exc, fctx, lctx)
        frame.unhandled.append(exc)
        return None

    def _run_handler(self, h: ast.ExceptHandler, frame: _TryFrame, exc: _Exc, fctx, lctx):
        """``except E as e: BODY`` is ``except E: try: BODY finally: del e`` (CPython): the body runs
        inside an implicit frame whose finalizer deletes the name and releases the instance on EVERY
        exit (normal, return, break, continue, a raise), and whose pending exits then leave the handler."""
        h_st = exc.st
        frame.in_handler = True
        self._active_exc_stack.append(exc)
        try:
            if h.name:
                self.store_name(h.name, exc.val, h_st, fctx)
            hframe = _TryFrame(h, fctx, lctx, lambda x: self._leave_handler(h.name, exc.val, x, fctx), False)
            self._try_stack.append(hframe)
            end = self.exec_block(h.body, h_st, fctx, lctx)
            end = self._finish_frame(hframe, None if end.bottom else end)
        finally:
            self._active_exc_stack.pop()
            frame.in_handler = False
        return end

    def _leave_handler(self, name, val: V, st: State, fctx) -> State:
        if name:
            self.delete_name(name, st, fctx)
        if val.refs:
            st.pins = tuple(p for p in st.pins if p != val)
        return st

    def _finish_frame(self, frame: _TryFrame, normal):
        """The finally phase of a frame: the finalizer runs on the normal continuation and on every pending
        exit, each separately; an exit the finalizer itself ends (a return, break, continue or raise in
        the finally block, routed to the enclosing frames) is replaced by that one.  The frame is popped
        and the surviving exits continue outward.  Returns the normal continuation, or ``None``."""
        fin = frame.finalizer
        fctx = frame.fctx
        if fin is not None:
            frame.in_finally = True
            try:
                if normal is not None:
                    normal = fin(normal)
                    if normal.bottom:
                        normal = None
                returns = [(fin(x), v) for x, v in frame.pending_returns]
                breaks = [fin(x) for x in frame.pending_breaks]
                continues = [fin(x) for x in frame.pending_continues]
                unhandled = []
                for exc in frame.unhandled:
                    x = fin(exc.st)
                    if not x.bottom:
                        unhandled.append(exc.with_state(x))
            finally:
                frame.in_finally = False
            returns = [(x, v) for x, v in returns if not x.bottom]
            breaks = [x for x in breaks if not x.bottom]
            continues = [x for x in continues if not x.bottom]
        else:
            returns, breaks, continues, unhandled = (frame.pending_returns, frame.pending_breaks,
                                                     frame.pending_continues, frame.unhandled)
        popped = self._try_stack.pop()
        assert popped is frame
        for x, v in returns:
            self._route_return(x, v, fctx)
        for x in breaks:
            self._route_break(x, fctx, frame.lctx)
        for x in continues:
            self._route_continue(x, fctx, frame.lctx)
        for exc in unhandled:
            self._route_exc(exc, fctx)
        return normal

    def _exit_mark(self, fctx: FuncCtx):
        """Where the exits of ``fctx`` stand (``_solve_fixpoint``: exits routed inside the iterations are
        discarded like returns are; iteration 1 and the final pass keep theirs)."""
        return (len(fctx.raises), tuple((id(fr), len(fr.raised), len(fr.unhandled), len(fr.pending_returns),
                                         len(fr.pending_breaks), len(fr.pending_continues))
                                        for fr in self._try_stack if fr.fctx is fctx))

    def _exit_truncate(self, fctx: FuncCtx, mark) -> None:
        del fctx.raises[mark[0]:]
        frames = {id(fr): fr for fr in self._try_stack}
        for fid, a, b, c, d, e in mark[1]:
            fr = frames.get(fid)
            if fr is None:
                continue
            del fr.raised[a:]
            del fr.unhandled[b:]
            del fr.pending_returns[c:]
            del fr.pending_breaks[d:]
            del fr.pending_continues[e:]

    def s_Assert(self, s, st, fctx, lctx):
        t, f = self.cond(s.test, st, fctx)
        if not f.bottom and s.msg is not None:
            self.eval(s.msg, f, fctx)
        self.check(s, "assert", {"AssertionError"}, f.bottom,
                   "assertion condition proven" if f.bottom else "assertion may fail", st=f)
        return t

    def s_FunctionDef(self, s, st, fctx, lctx):
        if not (fctx.is_module or fctx.is_class):
            raise Unsupported("nested function (closures are not modelled)", s)
        flavor = "plain"
        for d in s.decorator_list:
            if isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod") and fctx.is_class:
                flavor = d.id
            else:
                raise Unsupported("decorator", s)
        if _is_generator(s):
            # a generator function: defining it is fine, calling it aborts
            pass
        defaults = tuple(self.summarize(st, self.eval(d, st, fctx)) for d in s.args.defaults)
        kwd = {p.arg: self.summarize(st, self.eval(d, st, fctx))
               for p, d in zip(s.args.kwonlyargs, s.args.kw_defaults) if d is not None}
        qual = s.name
        mkey = fctx.module_key
        addr = self.alloc(st, s, lambda sid: FuncObj(s, defaults, kwd, qual, flavor, mkey), "func")
        self.store_name(s.name, V(refs=frozenset({addr})), st, fctx)
        return st

    def e_Lambda(self, e, st, fctx):
        if not fctx.is_module:
            raise Unsupported("nested lambda (closures are not modelled)", e)
        defaults = tuple(self.summarize(st, self.eval(d, st, fctx)) for d in e.args.defaults)
        kwd = {p.arg: self.summarize(st, self.eval(d, st, fctx))
               for p, d in zip(e.args.kwonlyargs, e.args.kw_defaults) if d is not None}
        addr = self.alloc(st, e, lambda sid: FuncObj(e, defaults, kwd, "<lambda>", "plain", fctx.module_key), "func")
        return V(refs=frozenset({addr}))

    def s_ClassDef(self, s, st, fctx, lctx):
        if not fctx.is_module:
            raise Unsupported("class defined inside a function or class", s)
        if s.decorator_list or s.keywords:
            raise Unsupported("class decorator / metaclass", s)
        if len(s.bases) > 1:
            raise Unsupported("multiple inheritance", s)
        bases = []
        exc_base = None
        for b in s.bases:
            bv = self.eval(b, st, fctx)
            if bv.top or bv.i or bv.b or bv.f or bv.s is not None or bv.n:
                raise Unsupported("class base", s)
            if bv.st:
                if len(bv.st) != 1 or bv.refs:
                    raise Unsupported("class base", s)
                tag = next(iter(bv.st))
                if tag == ("type", "object"):
                    continue
                if tag[0] == "exc":
                    exc_base = tag[1]
                    continue
                raise Unsupported(f"class base {tag}", s)
            if len(bv.refs) != 1:
                raise Unsupported("class base", s)
            a = next(iter(bv.refs))
            if st.heap[a].kind != "class":
                raise Unsupported("class base", s)
            bases.append(a)
        # the class body runs in its own namespace
        loc, glo = _assigned_names(s.body)
        cctx = FuncCtx(s, loc, glo, is_class=True, module_key=fctx.module_key)
        st.frames.append(Frame(f"c{len(st.frames)}"))
        self.borrowed.append(set())
        self._fctx_stack.append(cctx)
        try:
            body_st = self.exec_block(s.body, st, cctx, None)
        finally:
            self._fctx_stack.pop()
            self.borrowed.pop()
        routed = 0
        for exc in cctx.raises:        # [internal ref]: an exception leaving the class body is the module's
            if not exc.st.bottom:
                self.pop_frame(exc.st)
                self._route_exc(exc, fctx)
                routed += 1
        if body_st.bottom:
            if routed:
                return body_st         # the class statement raised: the name is never bound, the path ends
            raise Unsupported("class body does not complete", s)
        fr = body_st.frames[-1]
        if fr.maybe_unbound:
            raise Unsupported("class body has conditionally bound names", s)
        attrs = {}
        for k, v in fr.env.items():
            attrs[k] = self.summarize(body_st, self.read_env_value(body_st, fr, k, v))
        for k in attrs:
            if k in UNSUPPORTED_DUNDERS:
                raise Unsupported(f"class defines {k}", s)
        slots = None
        if "__slots__" in fr.env:
            sv = fr.env["__slots__"]
            names = self._const_str_seq(body_st, sv)
            if names is None:
                raise Unsupported("non-constant __slots__", s)
            slots = tuple(names)
        if exc_base is not None:
            attrs["__exc_base__"] = V(st=frozenset({("exc", exc_base)}))
        # The class made by an EARLIER run of this statement (a loop body: the fixpoint runs it again from a state that
        # holds it) would become the summary of its site, and the instances and subclasses that name it by address
        # (``InstObj.cls``, ``ClassObj.bases``) are not redirected with the value references, so they would name the
        # NEW class.  Two classes of one site are never told apart here: refused, never conflated.
        class_site = self.site(s, "class")
        if (class_site, "r") in body_st.heap or (class_site, "s") in body_st.heap:
            raise Unsupported("a class statement that runs more than once (a loop body) is not modelled", s)
        addr = self.alloc(body_st, s, lambda sid: ClassObj(s.name, s, tuple(bases), attrs, slots), "class")
        self.pop_frame(body_st)
        self.store_name(s.name, V(refs=frozenset({addr})), body_st, fctx)
        return body_st

    def _const_str_seq(self, st, v: V):
        if v.s is not None and v.s != ANY_STR and len(v.s) == 1 and not v.refs:
            return [next(iter(v.s))]
        if len(v.refs) == 1 and v.s is None:
            ob = st.heap.get(next(iter(v.refs)))
            if ob is not None and ob.kind == "tuple" and ob.items is not None:
                out = []
                for it in ob.items:
                    if it.s is None or it.s == ANY_STR or len(it.s) != 1 or it.refs or it.i or it.n:
                        return None
                    out.append(next(iter(it.s)))
                return out
        return None

    def _import_known_to_succeed(self, module, fctx, node, *, name=None, alias=None):
        """Whether existing model/resolver metadata proves this import binds on the target."""
        # The SV-COMP harness injects this virtual module into sys.modules; it is
        # not resolved from the task's import roots.
        if module == "_sv_verifier":
            if name is None:
                return True
            from tharkun.solve.pyresolve import module_attrs
            attrs = module_attrs(module)
            return attrs is not None and name in attrs
        modeled = module in MODULES
        if not modeled:
            from tharkun.solve.contracts.registry import for_module
            modeled = any(contract.tested for contract in for_module(module))
        gov = self._governing_resolver(fctx)
        if gov is None:
            if modeled:
                raise Unsupported(f"{module} import origin is not resolved", node)
            return False

        # A task root can shadow a modeled stdlib module. Check the Resolver's
        # existing search-root snapshot before accepting any success metadata.
        statement = node
        while statement in gov.parent:
            parent = gov.parent[statement]
            if isinstance(parent, ast.Module):
                break
            statement = parent
        if gov._pkg_dir is None:
            roots = gov._snapshots.get(id(statement))
            if roots is None:
                if modeled:
                    raise Unsupported(f"{module} import origin is not resolved", node)
                return False
        else:
            roots = gov._toplevel_roots
        from tharkun.solve import extern
        if module in gov.allowed and extern.shadowed_stdlib_name(roots, {module}) is not None:
            raise Unsupported(f"task root shadows stdlib module {module!r}", node)

        if name is None and module in gov.imports.modules.values():
            return True
        if name is not None and alias is not None:
            resolved = gov.imports.names.get(alias)
            if (resolved is not None and resolved[1] == name
                    and resolved[0].startswith("$extern:")):
                return True
        if not modeled:
            return False
        if name is None:
            return True
        from tharkun.solve.pyresolve import module_attrs
        attrs = module_attrs(module)
        return attrs is not None and name in attrs

    def s_Import(self, s, st, fctx, lctx):
        for a in s.names:
            if a.asname is None and "." in a.name:
                raise Unsupported("dotted import", s)
            if not self._import_known_to_succeed(a.name, fctx, s):
                self._in_try_refuse("ImportError", s)
            gov = self._governing_resolver(fctx)
            alias = a.asname or a.name
            sub = gov.externs.get(alias) if gov is not None else None
            if (sub is not None and gov.imports.modules.get(alias) == a.name
                    and not self._ensure_extern_loaded(st, sub)):
                raise Unsupported(f"imported source module {a.name!r} does not complete", s)
            self.store_name(a.asname or a.name, V(st=frozenset({("module", a.name)})), st, fctx)
        return st

    def s_ImportFrom(self, s, st, fctx, lctx):
        if s.level:
            # [internal ref]: a relative import is only ever eligible when this file itself was read
            # in via read-and-recurse (module_key set) and the governing Resolver already
            # resolved it (e); a relative import in the TOP-level program keeps refusing,
            # unchanged.
            gov = self._governing_resolver(fctx) if fctx.module_key else None
            if gov is None:
                raise Unsupported("relative import", s)
            self._exec_import_from(s, st, fctx, gov)
            return st
        for a in s.names:
            if a.name == "*":
                raise Unsupported("star import", s)
            module = s.module or ""
            alias = a.asname or a.name
            if not self._import_known_to_succeed(module, fctx, s, name=a.name, alias=alias):
                self._in_try_refuse("ImportError", s)
            v = self.module_attr(st, module, a.name, s, fctx)
            self.store_name(alias, v, st, fctx)
        return st

    # ------------------------------------------------------------ loops
    def _reuse_history(self) -> tuple:
        """What an analysis can learn once and keep: the globals some function writes, and the extern
        files read in.  Both only grow, so equal sizes mean equal contents."""
        return len(self.global_written), len(self._ext_ok), len(self._ext_failed)

    def _fixpoint(self, entry: State, step, fctx: FuncCtx, node) -> State:
        """The loop head ``_solve_fixpoint`` computes, reused when the same loop is entered again from an
        equivalent state (docs/ABSINT.md, "Fixpoint reuse").

        The head is a function of the loop, the call string, the entry state and the analyzer's scratch
        (the float landmarks, the allocation epoch, the borrowed references, what has been learned about
        globals and extern files, the widening delay), so a run with all of those equal is the same run; the key
        holds them and ``State.equiv`` compares the entry state.  A hit also applies what the skipped run changed
        in the analyzer (the temporary counter, the epoch, the borrowed set, the landmarks), so every
        later step sees what it would have seen.  Only while checks are not recorded: inside an outer
        loop's fixpoint iterations, where the same inner loop is solved again and again.  The recording
        pass never reuses, and a loop inside a length probe (``_probe_length``) never does either: its raises
        are noted in that scope, and a skipped run would not note them."""
        if self.record or self._probe_scopes or entry.bottom or self._forced_draws:
            # (a pending case-split force: nothing is read from or written to the cache meanwhile)
            return self._solve_fixpoint(entry, step, fctx, node)
        history = self._reuse_history()
        key = (node, tuple(self.callstring), lemmas.landmark_key(), history, frozenset(self.epoch),
               tuple(frozenset(b) for b in self.borrowed), WIDEN_DELAY)
        runs = self._fix_cache.setdefault(key, [])
        for run in runs:
            if entry.equiv(run.entry):
                self.tmp += run.tmp
                self.epoch = set(run.epoch)
                self.borrowed[-1].clear()
                self.borrowed[-1].update(run.borrowed)
                lemmas.merge_landmarks(run.landmarks)
                return run.head.copy()
        tmp = self.tmp
        head = self._solve_fixpoint(entry, step, fctx, node)
        if self._reuse_history() == history:
            runs.append(_FixpointRun(entry.copy(), head.copy(), self.tmp - tmp, frozenset(self.epoch),
                                     frozenset(self.borrowed[-1]), lemmas.landmark_key()))
            if len(runs) > FIX_REUSE_WAYS:
                del runs[0]
        return head

    def _solve_fixpoint(self, entry: State, step, fctx: FuncCtx, node) -> State:
        """Least-ish post-fixpoint of ``head = entry ⊔ step(head)``."""
        saved_rec, saved_ret = self.record, fctx.returns
        mark = self._exit_mark(fctx)   # [internal ref]: exits routed inside the iterations are discarded like returns
        self.record = False
        try:
            head = entry.copy()
            it = 0
            while True:
                it += 1
                fctx.returns = []
                self._exit_truncate(fctx, mark)
                back = step(head.copy())
                if back.leq(head):
                    break
                new = head.join(back)
                head = head.widen(new) if it > WIDEN_DELAY else new
                if it > MAX_ITERS:
                    raise Unsupported("loop fixpoint did not converge", node)
            # one decreasing step, kept only if it is itself inductive
            fctx.returns = []
            self._exit_truncate(fctx, mark)
            cand = entry.join(step(head.copy()))
            fctx.returns = []
            self._exit_truncate(fctx, mark)
            if step(cand.copy()).leq(cand):
                head = cand
            # Family tags only choose relational candidates; every accepted
            # bound covers entry and is checked against the actual loop step.
            if classify_loop_family(node) in (ACCUMULATOR, AFFINE, ITERATOR, KERNEL, SEARCH):
                names = {n.target.id for stmt in node.body for n in ast.walk(stmt)
                         if isinstance(n, (ast.AugAssign, ast.AnnAssign))
                         and isinstance(n.target, ast.Name)}
                names.update(t.id for stmt in node.body for n in ast.walk(stmt)
                             if isinstance(n, ast.Assign) for t in n.targets
                             if isinstance(t, ast.Name))
                vars_ = sorted(v for v in entry.zone.vars if v.rsplit(":", 1)[-1] in names)
                if len(vars_) <= 4:
                    for a in vars_:
                        for b in vars_:
                            if a == b:
                                continue
                            for offset in (0, 1):
                                if head.zone.entails(a, b, offset) or not entry.zone.entails(a, b, offset):
                                    continue
                                proposed = head.copy()
                                proposed.zone.add(a, b, offset)
                                if not proposed.bottom and entry.leq(proposed):
                                    fctx.returns = []
                                    self._exit_truncate(fctx, mark)
                                    if step(proposed.copy()).leq(proposed):
                                        head = proposed
            return head
        finally:
            self._exit_truncate(fctx, mark)
            self.record, fctx.returns = saved_rec, saved_ret

    def _visit_invariant(self, entry: State, head: State, step, fctx: FuncCtx, node) -> State:
        """What a witness may claim at a loop head: the head is visited with the entry state first and with
        ``head`` after that, so the claim is their join.  The checker verifies preservation on its own and a
        join of two states need not be inductive, so it is kept only if one more step maps it into itself;
        otherwise it is solved up to a fixpoint.  Nobody reads it unless invariants are being recorded."""
        if not (self.record and self.invariant_locations):
            return head
        joined = entry.join(head)
        saved_rec, saved_ret = self.record, fctx.returns
        mark = self._exit_mark(fctx)
        self.record, fctx.returns = False, []
        try:
            if step(joined.copy()).leq(joined):
                return joined
        finally:
            self._exit_truncate(fctx, mark)
            self.record, fctx.returns = saved_rec, saved_ret
        return self._solve_fixpoint(joined, step, fctx, node)

    def s_While(self, s, st, fctx, lctx):
        def test_at_entry(state):
            if self.record and s in self._statement_entry_nodes:
                self._observe_statement_entry(s, state, fctx)
            return self.cond(s.test, state, fctx)

        def step(head):
            self.fresh_pass()
            t, _f = test_at_entry(head)
            lc = LoopCtx()
            end = self.exec_block(s.body, t, fctx, lc) if not t.bottom else t
            for c in lc.continues:
                end = end.join(c)
            return end

        # Iteration 1 on its own, from the entry state, recording as the caller does.  The fixpoint then
        # starts from where iteration 1 ended and describes iterations 2 and on; a site is proven only if
        # it is proven in this pass AND in the final pass (``check`` and ``_record`` never overwrite).
        self.fresh_pass()
        t1, f1 = test_at_entry(st.copy())
        lc1 = LoopCtx()
        end1 = t1
        if not t1.bottom:
            end1 = self.exec_block(s.body, t1, fctx, lc1)
            for c in lc1.continues:
                end1 = end1.join(c)
        if end1.leq(st):
            # iteration 1 adds nothing: the entry state is the fixpoint and this pass was the final pass
            self.capture_invariant(s, st, loop=True)
            out, breaks = f1, lc1.breaks
        else:
            head = self._fixpoint(end1, step, fctx, s)
            self.capture_invariant(s, self._visit_invariant(st, head, step, fctx, s), loop=True)
            # final pass: records checks, collects breaks and returns
            self.fresh_pass()
            t, f = test_at_entry(head.copy())
            lc = LoopCtx()
            if not t.bottom:
                self.exec_block(s.body, t, fctx, lc)
            out, breaks = f1.join(f), lc1.breaks + lc.breaks
        if s.orelse and not out.bottom:
            out = self.exec_block(s.orelse, out, fctx, lctx)
        for b in breaks:
            out = out.join(b)
        return out

    def s_For(self, s, st, fctx, lctx):
        if s.type_comment:
            pass
        itv = self.eval(s.iter, st, fctx)
        if st.bottom:
            return st
        itv = self.iterable(st, itv, s.iter)
        self.check_iterable(st, itv, s.iter)
        st.pins = st.pins + (itv,)
        # A ghost iteration counter G for `for i in range(c, hi)` with a constant c:
        # at the head G = iterations done, the body sees i = c + G exactly, and the
        # normal exit has G = max(0, hi - c).  This is what lets a store at
        # xs[i] extend a list's written prefix (segmentation) and prove it complete.
        ghost = None
        if len(itv.refs) == 1 and itv.s is None and not itv.st:
            rob = st.heap.get(next(iter(itv.refs)))
            if (rob is not None and rob.kind == "range" and rob.step == 1
                    and isinstance(rob.lo, str) and isinstance(rob.hi, str)):
                clo, chi = st.zone.bounds(rob.lo)
                if clo is not None and clo == chi:
                    ghost = (f"G.{s.lineno}.{s.col_offset}.{len(st.frames)}", rob.hi, clo)
                    st.zone.assign_interval(ghost[0], 0, 0)

        def body_entry(head):
            b = head
            if ghost is not None:
                g, hiv, c = ghost
                b.zone.add(g, hiv, -c - 1)  # an iteration remains: c + G <= hi - 1
                if b.bottom:
                    return b
                ev = V(i=IntInfo(Lin.var(g, c)))
            else:
                ev = self.iter_elem(b, b.pins[-1], s)
            if ev is None:
                b.set_bottom()
                return b
            self.assign(s.target, ev, b, fctx)
            return b

        def bump(x):
            if ghost is not None and not x.bottom:
                x.zone.assign_var(ghost[0], ghost[0], 1)
                x.shift_bounds(ghost[0], 1)
            return x

        def step(head):
            self.fresh_pass()
            b = body_entry(head)
            lc = LoopCtx()
            end = bump(self.exec_block(s.body, b, fctx, lc) if not b.bottom else b)
            for c in lc.continues:
                end = end.join(bump(c))
            return end

        def exhausted(x):
            """``x`` with the iterable run out."""
            x = x.copy()
            if ghost is not None and not x.bottom:
                g, hiv, c = ghost
                x.zone.add(hiv, g, c)  # exhausted: G >= hi - c
            return x

        # Iteration 1 on its own, from the entry state, recording as the caller does (see s_While).
        self.fresh_pass()
        b1 = body_entry(st.copy())
        lc1 = LoopCtx()
        end1 = b1
        if not b1.bottom:
            end1 = bump(self.exec_block(s.body, b1, fctx, lc1))
            for c in lc1.continues:
                end1 = end1.join(bump(c))
        if end1.leq(st):
            # iteration 1 adds nothing: the entry state is the fixpoint and this pass was the final pass
            out, breaks = exhausted(st), lc1.breaks
        else:
            head = self._fixpoint(end1, step, fctx, s)
            self.fresh_pass()
            b = body_entry(head.copy())
            lc = LoopCtx()
            if not b.bottom:
                self.exec_block(s.body, b, fctx, lc)
            # the iterable ran out before the first iteration, or after one or more
            out, breaks = exhausted(st).join(exhausted(head)), lc1.breaks + lc.breaks
        if s.orelse and not out.bottom:
            out = self.exec_block(s.orelse, out, fctx, lctx)
        for br in breaks:
            out = out.join(br)
        if not out.bottom:
            out.pins = out.pins[:-1]
            if ghost is not None:
                out.kill_var(ghost[0])
        return out

    def check_iterable(self, st: State, v: V, node) -> bool:
        """Record the "iter" TypeError check; True iff ``v`` is certainly iterable."""
        bad = v.i is not None or bool(v.b) or v.f or v.n
        for a in v.refs:
            k = st.heap[a].kind
            if k not in ("list", "tuple", "range", "dict", "dictview", "iterator", "array", "set"):
                if k == "inst":
                    raise Unsupported("iteration over an instance", node)
                if k == "stringio":
                    raise Unsupported("iteration over a StringIO (its lines are not modelled)", node)
                bad = True
            if k == "dictview" and st.heap[a].which == "items":
                raise Unsupported("iteration over dict.items()", node)
        for t in v.st:
            bad = True
        if v.top:
            raise Unsupported("iteration over an unknown value", node)
        self.check(node, "iter", {"TypeError"}, not bad, "iterable: " + ",".join(v.kinds()), st=st)
        return not bad

    def iter_elem(self, st: State, v: V, node) -> V | None:
        """One element of iterable ``v``, materialized (None: it has no element).

        SOUNDNESS: materializing a summary asserts that an element exists (its
        bounds become constraints of a temporary, and the zone closure then
        derives e.g. ``lo <= hi - 1``).  Call this only on paths where an
        element provably exists: a loop body, a successful unpack.  Elsewhere
        use :meth:`elem_summary`."""
        out = self.elem_summary(st, v)
        if out is None:
            return None
        return self.mat(st, out)

    def elem_summary(self, st: State, v: V) -> V | None:
        """Every element of iterable ``v`` (symbolic bounds, not materialized)."""
        out = BOT
        if v.s is not None:
            out = vjoin(out, st.zone, ANYSTR, st.zone)
        for a in v.refs:
            ob = st.heap.get(a)
            if ob is None:
                continue
            if ob.kind == "range":
                if ob.step > 0:
                    lo = (ob.lo, 0) if isinstance(ob.lo, str) else (None if ob.lo[0] is None else (ZERO, ob.lo[0]))
                    hi = (ob.hi, -1) if isinstance(ob.hi, str) else (None if ob.hi[1] is None else (ZERO, ob.hi[1] - 1))
                else:
                    hi = (ob.lo, 0) if isinstance(ob.lo, str) else (None if ob.lo[1] is None else (ZERO, ob.lo[1]))
                    lo = (ob.hi, 1) if isinstance(ob.hi, str) else (None if ob.hi[0] is None else (ZERO, ob.hi[0] + 1))
                ev = V(i=IntInfo(None, lo, hi))
                out = vjoin(out, st.zone, ev, st.zone)
            elif ob.kind in ("list", "array"):
                out = vjoin(out, st.zone, self.all_elems(st, ob), st.zone)
            elif ob.kind == "tuple":
                if ob.items is not None:
                    for x in ob.items:
                        out = vjoin(out, st.zone, x, st.zone)
                else:
                    out = vjoin(out, st.zone, ob.elem, st.zone)
            elif ob.kind == "dict":
                out = vjoin(out, st.zone, self.dict_keys_v(st, ob), st.zone)
            elif ob.kind == "set":
                out = vjoin(out, st.zone, ob.elem, st.zone)
            elif ob.kind == "iterator":
                elem = self.elem_summary(st, ob.source)
                if elem is not None:
                    out = vjoin(out, st.zone, elem, st.zone)
            elif ob.kind == "dictview":
                for d in ob.dicts:
                    dob = st.heap[d]
                    if ob.which == "keys":
                        out = vjoin(out, st.zone, self.dict_keys_v(st, dob), st.zone)
                    elif ob.which == "values":
                        out = vjoin(out, st.zone, self.dict_vals_v(st, dob), st.zone)
                    else:
                        return None
        if out.is_bot():
            return None
        return out

    # ------------------------------------------------------------ dicts
    def dict_keys_v(self, st: State, ob) -> V:
        from .state import _key_value
        out = BOT
        for k in ob.items:
            out = vjoin(out, st.zone, _key_value(k), st.zone)
        return vjoin(out, st.zone, ob.okeys, st.zone)

    def dict_vals_v(self, st: State, ob) -> V:
        out = BOT
        for v in ob.items.values():
            out = vjoin(out, st.zone, v, st.zone)
        return vjoin(out, st.zone, ob.ovals, st.zone)

    def dict_len(self, st: State, addrs) -> tuple:
        obs = [st.heap[d] for d in addrs]
        lo = min(len(o.items) for o in obs)
        hi = max(len(o.items) for o in obs) if all(o.exact for o in obs) else None
        return lo, hi

    def dict_method(self, st: State, a, name, args, kwargs, node) -> V:
        qualname = f"dict.{name}"
        if qualname in _METHOD_HOOKS:
            return self.visit_method_rule(qualname, MethodCtx(st, frozenset({a}), name, args, kwargs, node))
        return self._dict_method_model(st, a, name, args, kwargs, node)

    def _dict_method_model(self, st: State, a, name, args, kwargs, node) -> V:
        ob = st.heap[a]
        if name in ("keys", "values", "items"):
            self._arity(node, args, kwargs, 0, 0, name)
            va = self.alloc(st, node, lambda sid: DictView(frozenset({a}), name), "dictview")
            return V(refs=frozenset({va}))
        if name == "get":
            if not self._arity(node, args, kwargs, 1, 2, "get"):
                st.set_bottom()
                return BOT
            val, _ok, _f = self.dict_load(st, ob, args[0], node)
            dflt = args[1] if len(args) > 1 else NONE
            return vjoin(self.mat(st, val) if not val.is_bot() else BOT, st.zone, dflt, st.zone)
        raise Unsupported(f"dict.{name}", node)

    def key_consts(self, st: State, v: V):
        """The finite set of hashable constants ``v`` may be, or None.  An int is a set of constants when the
        zone bounds it: a name bound to a constant (``EMPTY, WHITE, BLACK = 0, 1, 2``) is one constant, a loop
        variable over ``range(3)`` is three; the bounds are sound, so every value the key may have is in the set."""
        if v.top or v.refs or v.st or v.f:
            return None
        out = set()
        if v.s is not None:
            if v.s == ANY_STR:
                return None
            out |= set(v.s)
        if v.b:
            out |= set(v.b)
        if v.n:
            out.add(None)
        if v.i is not None:
            lo, hi = self.ibounds(st, v.i)
            if lo is None or hi is None or hi - lo > 64:
                return None
            out |= set(range(lo, hi + 1))
        return out

    def _dict_key_may_equal(self, key: V, constants, definite) -> bool:
        """Whether a builtin scalar key may equal a key already definitely present."""
        if constants is not None:
            return any(candidate == definite for candidate in constants)
        if definite is None:
            return key.n
        if isinstance(definite, str):
            return key.s is not None
        if isinstance(definite, (bool, int)):
            # Python numeric keys coalesce across bool, int, and float.  When a
            # numeric key is not finite, retain every definite numeric entry.
            return key.i is not None or bool(key.b) or bool(key.f)
        return False

    def e_Dict(self, e, st, fctx):
        items = {}
        okeys = BOT
        ovals = BOT
        exact = True
        for k, x in zip(e.keys, e.values):
            if k is None:
                raise Unsupported("dict unpacking", e)
            kv = self.eval(k, st, fctx)
            xv = self.eval(x, st, fctx)
            if st.bottom:
                return BOT
            ks = self.key_consts(st, kv)
            value = self.summarize(st, xv)
            if ks is not None and len(ks) == 1:
                key = next(iter(ks))
                items[key] = value
                continue

            # shortcut: open keys lose identity/cardinality; refine them when lookup proofs need those facts.
            # References and static values can carry custom hash/equality hooks.
            if kv.top or kv.refs or kv.st or not (kv.i is not None or kv.b or kv.f or kv.s is not None or kv.n):
                raise Unsupported("dict literal with a non-constant key", e)

            for key, old in items.items():
                if self._dict_key_may_equal(kv, ks, key):
                    # An uncertain insertion cannot remove the key, but can
                    # replace its value if it collides.
                    items[key] = vjoin(old, st.zone, value, st.zone)
            okeys = vjoin(okeys, st.zone, self.summarize(st, kv), st.zone)
            ovals = vjoin(ovals, st.zone, value, st.zone)
            exact = False
        a = self.alloc(st, e, lambda sid: DictObj(items, exact, okeys, ovals), "dict")
        return V(refs=frozenset({a}))

    def check_hash_key(self, key: V, node):
        if key.top or key.refs or key.st:
            raise Unsupported("hash/equality protocol of a non-scalar key", node)

    def dict_load(self, st: State, ob, kv: V, node) -> tuple:
        """(value, ok) of ``d[k]``: KeyError is impossible when every key ``kv`` may be is present."""
        self.check_hash_key(kv, node)
        ks = self.key_consts(st, kv)
        if ks is None:
            ok = False
            out = self.dict_vals_v(st, ob)
            fact = "key not a known constant"
        else:
            missing = [k for k in ks if k not in ob.items]
            ok = not missing
            out = BOT
            for k in ks:
                if k in ob.items:
                    out = vjoin(out, st.zone, ob.items[k], st.zone)
            if missing and not ob.exact:
                out = vjoin(out, st.zone, ob.ovals, st.zone)
            fact = (f"keys {sorted(map(repr, ks))} all present" if ok else
                    f"key(s) {sorted(map(repr, missing))} may be absent")
        return out, ok, fact

    def _defaultdict_store_factory_list(self, st: State, address, key_summary: V, keys, value: V,
                                       strong: bool) -> None:
        """Record a factory-created list under a miss without inventing an unknown key identity."""
        ob = st.heap[address]
        items = dict(ob.items)
        if value.refs:
            self.borrow(st, value)
        if strong and ob.exact and keys is not None and len(keys) == 1:
            # A single concrete miss in a closed, uniquely targeted dict remains a definite entry.
            items[next(iter(keys))] = self.summarize(st, value)
            st.heap[address] = ob.replace(items=items)
            return
        st.heap[address] = ob.replace(
            exact=False,
            okeys=vjoin(ob.okeys, st.zone, key_summary, st.zone),
            ovals=vjoin(ob.ovals, st.zone, self.summarize(st, value), st.zone),
        )

    # ------------------------------------------------------------ assignment
    def assign(self, t, v: V, st: State, fctx) -> None:
        self.borrow(st, v)
        if st.bottom:
            return
        if isinstance(t, ast.Name):
            self.store_name(t.id, v, st, fctx)
        elif isinstance(t, (ast.Tuple, ast.List)):
            self.unpack(t, v, st, fctx)
        elif isinstance(t, ast.Subscript):
            obj = self.eval(t.value, st, fctx)
            idx = self.eval_index(t.slice, st, fctx)
            self.subscript_store(st, obj, idx, v, t)
        elif isinstance(t, ast.Attribute):
            obj = self.eval(t.value, st, fctx)
            self.attr_store(st, obj, t.attr, v, t)
        else:
            raise Unsupported(f"assignment target {type(t).__name__}", t)

    def unpack(self, t, v: V, st: State, fctx) -> None:
        self._in_try_refuse("unpacking (its ValueError is not routed)", t)
        v = self.iterable(st, v, t)
        n = len(t.elts)
        if any(isinstance(e, ast.Starred) for e in t.elts):
            raise Unsupported("starred assignment", t)
        if v.top:
            raise Unsupported("unpacking an unknown value", t)
        bad = v.i is not None or bool(v.b) or v.f or v.n or bool(v.st)
        parts = [BOT] * n
        feasible = False
        if v.s is not None:
            feasible = True
            parts = [vjoin(p, st.zone, ANYSTR, st.zone) for p in parts]
        for a in v.refs:
            ob = st.heap[a]
            if ob.kind == "tuple" and ob.items is not None:
                if len(ob.items) != n:
                    continue  # ValueError: not a property exception
                feasible = True
                parts = [vjoin(p, st.zone, x, st.zone) for p, x in zip(parts, ob.items)]
            elif ob.kind in ("list", "tuple", "array"):
                feasible = True
                parts = [vjoin(p, st.zone, self.all_elems(st, ob), st.zone) for p in parts]
            elif ob.kind in ("range", "iterator", "set"):
                ev = self.iter_elem(st, V(refs=frozenset({a})), t)
                if ev is not None:
                    feasible = True
                    parts = [vjoin(p, st.zone, ev, st.zone) for p in parts]
            elif ob.kind in ("dict", "dictview"):
                raise Unsupported("unpacking a dict or view", t)
            elif ob.kind == "stringio":
                raise Unsupported("unpacking a StringIO (its lines are not modelled)", t)
            else:
                bad = True
        self.check(t, "unpack", {"TypeError"}, not bad, "unpacking an iterable: " + ",".join(v.kinds()), st=st)
        if not feasible or any(p.is_bot() for p in parts):
            st.set_bottom()
            return
        parts = [self.borrow(st, self.mat(st, p)) for p in parts]
        for e, p in zip(t.elts, parts):
            self.assign(e, p, st, fctx)

    # ------------------------------------------------------------- names
    def _scope(self, name, fctx: FuncCtx):
        if fctx.is_module:
            return "module"
        if name in fctx.global_names:
            return "module"
        if name in fctx.local_names:
            return "local"
        return "enclosing"

    @staticmethod
    def _global_key(name: str, module_key: str) -> str:
        """The ``frames[0]`` env/zone key a MODULE-scope name uses. ``module_key == ""`` (the
        program's own module -- always true before [internal ref]) keeps the bare name, byte-identical
        to every program's behaviour before read-and-recurse existed; a nonzero module_key
        (a file read in via read-and-recurse) gets its own namespace, so a name the program
        happens to also use as a global of its own can never collide with -- or be silently
        joined to -- the SAME-spelled name in a DIFFERENT file's own globals."""
        return name if not module_key else f"{module_key}\x00{name}"

    def store_name(self, name, v: V, st: State, fctx) -> None:
        if st.bottom:
            return
        sc = self._scope(name, fctx)
        if sc == "enclosing":
            raise Unsupported("assignment to a non-local name", None)
        if sc == "module":
            self.write_var(st, 0, self._global_key(name, fctx.module_key), v)
        else:
            self.write_var(st, len(st.frames) - 1, name, v)

    def delete_name(self, name: str, st: State, fctx: FuncCtx) -> None:
        if st.bottom:
            return
        if name == "__builtins__":
            raise Unsupported("deleting the builtin namespace", None)
        sc = self._scope(name, fctx)
        idx = 0 if sc == "module" else (len(st.frames) - 1)
        var_name = self._global_key(name, fctx.module_key) if sc == "module" else name
        fr = st.frames[idx]
        zn = f"{fr.fid}:{var_name}"
        st.kill_var(zn)
        st.rewrite_bounds_away(zn)
        fr.env.pop(var_name, None)
        fr.maybe_unbound.discard(var_name)

    def write_var(self, st: State, idx: int, name: str, v: V) -> None:
        # Also covers imported module globals and formal parameters; run()'s
        # supplied default namespace is initialized directly, never through here.
        if name.rsplit("\x00", 1)[-1] == "__builtins__":
            raise Unsupported("binding the builtin namespace", None)
        fr = st.frames[idx]
        zn = f"{fr.fid}:{name}"
        z = st.zone
        if v.i is not None:
            ii = v.i
            s = ii.lin.single() if ii.lin is not None else None
            if s is not None and s[0] == zn:
                z.assign_var(zn, zn, s[1])
                st.shift_bounds(zn, s[1])
            else:
                if zn in ii.vars():
                    t = self.fresh(st)
                    z.assign_var(t, zn, 0)
                    ii = _rename_ii(ii, zn, t)
                st.rewrite_bounds_away(zn)
                self.zone_assign(st, zn, ii)
            v = v.replace(i=IntInfo(Lin.var(zn)))
        else:
            st.kill_var(zn)
        fr.env[name] = v
        fr.maybe_unbound.discard(name)

    def read_env_value(self, st: State, fr: Frame, name: str, v: V) -> V:
        if v.i is not None:
            return v.replace(i=IntInfo(Lin.var(f"{fr.fid}:{name}")))
        return v

    def load_name(self, name, node, st: State, fctx) -> V:
        sc = self._scope(name, fctx)
        # shortcut: refuse possibly unbound reads; split binding/absence paths when those proofs are needed.
        if fctx.is_class:
            # class bodies: class namespace, then globals, then builtins
            fr = st.frames[-1]
            if name in fr.maybe_unbound:
                raise Unsupported(f"class name {name!r} may be unbound", node)
            if name in fr.env:
                return fr.env[name]
        elif sc == "local":
            fr = st.frames[-1]
            if name in fr.maybe_unbound:
                raise Unsupported(f"local name {name!r} may be unbound", node)
            v = fr.env.get(name)
            if v is None:
                self._in_try_refuse("UnboundLocalError", node)
                st.set_bottom()  # UnboundLocalError: not a property exception
                return BOT
            return v
        # A free name of a module-level function or method is a module global.
        # [internal ref] (soundness, ROOT fix): materialize -- copy at read time, into a
        # fresh zone temp via mat()/to_temp() -- EVERY read of a global that
        # some function writes (self.global_written), not only when the read
        # itself happens inside a function frame (the old "len(st.frames) > 1"
        # gate). g.env[name] otherwise stays a LIVE reference into the mutable
        # zone slot "{fid}:{name}": if a later operand in the SAME statement
        # (module scope or not) runs a call that rebinds this same global via
        # store_name/write_var, the earlier read observes the call's NEW value
        # instead of the one that was actually live at read time -- unsound,
        # since CPython evaluates operands strictly left-to-right and a saved
        # operand must keep the value it had when it was evaluated. Always
        # over-approximating here (materializing even where nothing could
        # mutate the global before its value is used) costs only precision,
        # never soundness.
        g = st.frames[0]
        key = self._global_key(name, fctx.module_key)
        if key in g.maybe_unbound:
            raise Unsupported(f"global name {name!r} may be unbound", node)
        if key in g.env:
            v = g.env[key]
            if v.i is not None and key in self.global_written:
                return self.mat(st, v)
            return v
        b = self.builtin_name(name, node)
        if b.is_bot():
            self._in_try_refuse("NameError", node)
            st.set_bottom()  # NameError: not a property exception
        return b

    # ============================================================ expressions
    def eval(self, e, st: State, fctx) -> V:
        if st.bottom:
            return BOT
        m = getattr(self, "e_" + type(e).__name__, None)
        if m is None:
            raise Unsupported(f"expression {type(e).__name__}", e)
        # The values of this expression's operands are held by this evaluation, in a scope of its own: they are
        # dead when it returns (what it returns is registered in the enclosing scope below), so an allocation
        # that follows -- the next operand of a call, the next statement -- is not refused for them.
        self.borrowed.append(set())
        try:
            value = m(e, st, fctx)
        finally:
            self.borrowed.pop()
        return self.borrow(st, value)

    def e_Constant(self, e, st, fctx):
        c = e.value
        if c is None:
            return NONE
        if isinstance(c, bool):
            return const_bool(c)
        if isinstance(c, int):
            return const_int(c)
        if isinstance(c, float):
            return V(f=FloatInfo(fv.FloatFact.const(c)))
        if isinstance(c, str):
            return const_str(c) if len(c) <= 64 else ANYSTR
        raise Unsupported(f"constant of type {type(c).__name__}", e)

    def e_Name(self, e, st, fctx):
        return self.load_name(e.id, e, st, fctx)

    def e_Tuple(self, e, st, fctx):
        items = []
        for x in e.elts:
            if isinstance(x, ast.Starred):
                raise Unsupported("starred tuple item", e)
            items.append(self.eval(x, st, fctx))
        if st.bottom:
            return BOT
        a = self.new_list(st, e, None, BOT, tag="tuple", kind="tuple", items=items)
        return V(refs=frozenset({a}))

    def e_List(self, e, st, fctx):
        elem = BOT
        for x in e.elts:
            if isinstance(x, ast.Starred):
                raise Unsupported("starred list item", e)
            v = self.eval(x, st, fctx)
            elem = self.join_into(st, elem, v)
        if st.bottom:
            return BOT
        a = self.new_list(st, e, IntInfo(Lin.const(len(e.elts))), elem)
        return V(refs=frozenset({a}))

    def e_IfExp(self, e, st, fctx):
        t, f = self.cond(e.test, st, fctx)
        vt = self.eval(e.body, t, fctx) if not t.bottom else BOT
        vf = self.eval(e.orelse, f, fctx) if not f.bottom else BOT
        return self._merge_branches(st, [(t, vt), (f, vf)])

    def _merge_branches(self, st: State, outs) -> V:
        outs = [(s, v) for s, v in outs if not s.bottom]
        if not outs:
            st.set_bottom()
            return BOT
        tname = None
        if any(v.i is not None and v.i.lin is not None for _, v in outs) and len(outs) > 1:
            tname = self.fresh(st)
        res_s, res_v = None, BOT
        for s, v in outs:
            if tname is not None and v.i is not None:
                self.zone_assign(s, tname, v.i)
                v = v.replace(i=IntInfo(Lin.var(tname)))
            if res_s is None:
                res_s, res_v = s, v
            else:
                res_v = vjoin(res_v, res_s.zone, v, s.zone)
                res_s = res_s.join(s)
        _assign_state(st, res_s)
        return res_v

    def e_BoolOp(self, e, st, fctx):
        outs = []
        cur = st
        for i, x in enumerate(e.values):
            v = self.eval(x, cur, fctx)
            if cur.bottom:
                break
            if i == len(e.values) - 1:
                outs.append((cur, v))
                break
            t, f = self.truth_split(cur, v, x)
            if isinstance(e.op, ast.And):
                outs.append((f, v))
                cur = t
            else:
                outs.append((t, v))
                cur = f
            if cur.bottom:
                break
        return self._merge_branches(st, outs)

    def e_UnaryOp(self, e, st, fctx):
        v = self.eval(e.operand, st, fctx)
        if st.bottom:
            return BOT
        if isinstance(e.op, ast.Not):
            true, false = self.truth_split(st, v, e)
            return self._merge_branches(st, [(true, const_bool(False)), (false, const_bool(True))])
        return self.unop(st, e.op, v, e)

    def e_BinOp(self, e, st, fctx):
        if isinstance(e.op, ast.BitOr):
            hint = self._check_type_intstr_hint(e, st, fctx)
            if hint is not None:
                return hint
        l = self.eval(e.left, st, fctx)
        # save the left operand before the right runs: a call on the right may
        # mutate a global the left operand named ([internal ref]; see save_operand).
        l = self.save_operand(st, l, e.right)
        r = self.eval(e.right, st, fctx)
        if st.bottom:
            return BOT
        # `x * x` over one name: both reads see the same value (a Name read has no effect)
        square = (isinstance(e.op, ast.Mult) and isinstance(e.left, ast.Name)
                  and isinstance(e.right, ast.Name) and e.left.id == e.right.id)
        return self.binop(st, e.op, l, r, e, square=square)

    def _check_type_intstr_hint(self, node, st, fctx):
        """Tag one inline int|str hint only at a direct check_type argument."""
        gov = self._governing_resolver(fctx)
        if gov is None or fctx.is_class:
            return None
        parent = gov.parent.get(node)
        if isinstance(parent, ast.keyword):
            parent = gov.parent.get(parent)
        if not isinstance(parent, ast.Call) or gov.call_name(parent) != "_sv_verifier.check_type":
            return None
        hint_arg = False
        if len(parent.args) == 2 and not parent.keywords and parent.args[1] is node:
            hint_arg = True
        elif len(parent.args) == 1 and parent.args[0] is not node and not any(
                kw.arg == "value" for kw in parent.keywords):
            hint_arg = (len(parent.keywords) == 1 and parent.keywords[0].arg == "hint"
                        and parent.keywords[0].value is node)
        elif not parent.args and len(parent.keywords) == 2:
            keywords = {kw.arg: kw.value for kw in parent.keywords}
            hint_arg = (set(keywords) == {"value", "hint"} and keywords["hint"] is node)
        if not hint_arg:
            return None
        if not (isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name)
                and {node.left.id, node.right.id} == {"int", "str"}
                and gov.is_builtin(node.left) and gov.is_builtin(node.right)):
            return None
        left, right = self.eval(node.left, st, fctx), self.eval(node.right, st, fctx)
        if st.bottom:
            return BOT
        if (left != V(st=frozenset({("type", node.left.id)}))
                or right != V(st=frozenset({("type", node.right.id)}))):
            raise Unsupported("check_type union operands are not exact builtin classes", node)
        return V(st=frozenset({("check-type-union", "int", "str")}))

    def e_Compare(self, e, st, fctx):
        t, f = self.cond(e, st, fctx)
        outs = []
        if not t.bottom:
            outs.append((t, const_bool(True)))
        if not f.bottom:
            outs.append((f, const_bool(False)))
        return self._merge_branches(st, outs)

    def e_Attribute(self, e, st, fctx):
        v = self.eval(e.value, st, fctx)
        if st.bottom:
            return BOT
        return self.attr_load(st, v, e.attr, e)

    def e_Subscript(self, e, st, fctx):
        obj = self.eval(e.value, st, fctx)
        idx = self.eval_index(e.slice, st, fctx)
        if st.bottom:
            return BOT
        return self.subscript_load(st, obj, idx, e)

    def e_JoinedStr(self, e, st, fctx):
        parts = [self.eval(p, st, fctx) for p in e.values]
        return BOT if st.bottom else self.visit_rule("syntax.JoinedStr", st, parts, {}, e)

    def e_FormattedValue(self, e, st, fctx):
        value = self.eval(e.value, st, fctx)
        if not st.bottom and e.conversion != -1:
            value = self.visit_rule({115: "str", 114: "repr", 97: "ascii"}[e.conversion], st, [value], {}, e)
        if st.bottom:
            return BOT
        spec = self.eval(e.format_spec, st, fctx) if e.format_spec is not None else const_str("")
        return BOT if st.bottom else self.visit_rule("syntax.FormattedValue", st, [value, spec, const_int(-1)], {}, e)

    def check_formattable(self, st, v: V, node, has_spec=False):
        if v.top:
            raise Unsupported("formatting an unknown value", node)
        todo, seen = [v], set()
        while todo:
            item = todo.pop()
            if item.top:
                raise Unsupported("formatting an unknown element", node)
            if item.st:
                raise Unsupported("formatting a static object with unverified callbacks", node)
            for a in item.refs - seen:
                seen.add(a)
                ob = st.heap[a]
                if ob.kind == "inst":
                    for k in ("__str__", "__repr__", "__format__"):
                        if self.class_lookup(st, ob.cls, k) is not None:
                            raise Unsupported("formatting an instance with a custom __str__/__repr__", node)
                elif ob.kind in ("list", "tuple", "dict", "set"):
                    todo.extend(ob.values())
        if has_spec:
            self.check(node, "format", {"TypeError"}, not (v.refs or v.n or v.st), "format spec on a number or str", st=st)

    # ------------------------------------------------------------ indices
    def eval_index(self, sl, st: State, fctx):
        if isinstance(sl, ast.Slice):
            lo = self.eval(sl.lower, st, fctx) if sl.lower is not None else None
            hi = self.eval(sl.upper, st, fctx) if sl.upper is not None else None
            step = self.eval(sl.step, st, fctx) if sl.step is not None else None
            return ("slice", lo, hi, step, sl)
        return ("index", self.eval(sl, st, fctx))

    # ============================================================ conditions
    def cond(self, e, st: State, fctx) -> tuple[State, State]:
        """(state where ``e`` is truthy, state where it is falsy)."""
        if st.bottom:
            return st, st.copy()
        if isinstance(e, ast.BoolOp):
            if isinstance(e.op, ast.And):
                fs = []
                cur = st
                for x in e.values:
                    t, f = self.cond(x, cur, fctx)
                    fs.append(f)
                    cur = t
                    if cur.bottom:
                        break
                F = fs[0]
                for f in fs[1:]:
                    F = F.join(f)
                return cur, F
            ts = []
            cur = st
            for x in e.values:
                t, f = self.cond(x, cur, fctx)
                ts.append(t)
                cur = f
                if cur.bottom:
                    break
            T = ts[0]
            for t in ts[1:]:
                T = T.join(t)
            return T, cur
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.Not):
            t, f = self.cond(e.operand, st, fctx)
            return f, t
        if isinstance(e, ast.Compare):
            return self.cond_compare(e, st, fctx)
        v = self.eval(e, st, fctx)
        if st.bottom:
            return st, st.copy()
        return self.truth_split(st, v, e)

    def check_truthy(self, st, v: V, node):
        if v.top:
            raise Unsupported("truth value of an unknown value", node)
        for a in v.refs:
            ob = st.heap[a]
            if ob.kind == "inst":
                for k in ("__bool__", "__len__"):
                    if self.class_lookup(st, ob.cls, k) is not None:
                        raise Unsupported("instance truth value", node)

    def truth_split(self, st: State, v: V, node) -> tuple[State, State]:
        if self.instance_value(st, v):
            truths, falses = [], []
            for part in self.alternatives(v):
                branch = st.copy()
                cls = self.instance_class(branch, part)
                if cls is not None and self.class_lookup(branch, cls, "__bool__") is not None:
                    raise Unsupported("custom __bool__", node)

                def decide(branch=branch, part=part, cls=cls):
                    if cls is not None and self.class_lookup(branch, cls, "__len__") is not None:
                        part = self._len_model(branch, [part], {}, node)
                    t2, f2 = self.truth_split(branch, part, node) if cls is None or not part.refs else (
                        branch, branch.copy())
                    if cls is not None and part.refs:
                        f2.set_bottom()  # default object truth
                    return t2, f2
                # the truth of a user instance is ``protocol.truthiness``'s (its ``__len__`` body is analysed, no ``__bool__``)
                t2, f2 = decide() if cls is None else self.visit_protocol(
                    "protocol.truthiness", node, decide, _PROTOCOL_NATIVE["protocol.truthiness"])
                truths.append(t2)
                falses.append(f2)
            t, f = truths[0], falses[0]
            for other in truths[1:]:
                t = t.join(other)
            for other in falses[1:]:
                f = f.join(other)
            return t, f
        self.check_truthy(st, v, node)
        t, f = st.copy(), st.copy()
        can_t = can_f = False
        if v.i is not None:
            lo, hi = self.ibounds(st, v.i)
            if not (lo == 0 and hi == 0):
                can_t = True
            if (lo is None or lo <= 0) and (hi is None or hi >= 0):
                can_f = True
        if v.b:
            can_t = can_t or True in v.b
            can_f = can_f or False in v.b
        if v.f or v.s is not None:
            if v.s is not None and v.s != ANY_STR:
                can_t = can_t or any(x for x in v.s)
                can_f = can_f or any(not x for x in v.s)
            else:
                can_t = can_f = True
        if v.n:
            can_f = True
        if v.st:
            can_t = True
        for a in v.refs:
            ob = st.heap[a]
            if ob.kind in ("list", "tuple", "range", "dict", "dictview", "array"):
                can_t = can_f = True
            elif ob.kind == "set":
                # a set is truthy exactly when it has an element
                can_t = can_t or ob.length[1] != 0
                can_f = can_f or ob.length[0] == 0
            else:
                can_t = True
        # refine a variable that is only an int
        name = node.id if isinstance(node, ast.Name) else None
        if v.only_int() and v.i.lin is not None:
            d = v.i.lin
            s = d.single()
            if s is not None and s[0] != ZERO:
                x, c = s
                # true: x + c != 0 ; false: x + c == 0
                _refine_ne(t.zone, x, ZERO, -c)
                f.zone.add(x, ZERO, -c)
                f.zone.add(ZERO, x, c)
        if not can_t:
            t.set_bottom()
        if not can_f:
            f.set_bottom()
        if not t.bottom and name is not None:
            self._refine_name_type(t, name, drop_none=True, st_ref=node)
        return t, f

    def _refine_name_type(self, st: State, name: str, drop_none: bool, st_ref) -> None:
        fr = st.frames[-1]
        if name in fr.env:
            v = fr.env[name]
            if drop_none and v.n:
                nv = v.replace(n=False)
                if nv.is_bot():
                    st.set_bottom()
                else:
                    fr.env[name] = nv

    def cond_compare(self, e: ast.Compare, st: State, fctx):
        left_node = e.left
        lv = self.eval(e.left, st, fctx)
        falses = []
        cur = st
        for op, rnode in zip(e.ops, e.comparators):
            # save this operand before the next (rnode) runs: a call on the
            # right may mutate a global this operand named ([internal ref]; never drop
            # the branch it would otherwise falsify -- see save_operand).
            lv = self.save_operand(cur, lv, rnode)
            rv = self.eval(rnode, cur, fctx)
            if cur.bottom:
                break
            t, f = self.compare_split(cur, op, lv, rv, left_node, rnode, e)
            falses.append(f)
            cur = t
            lv, left_node = rv, rnode
            if cur.bottom:
                break
        if not falses:
            return cur, cur.copy()
        F = falses[0]
        for f in falses[1:]:
            F = F.join(f)
        return cur, F

    def compare_split(self, st: State, op, lv: V, rv: V, lnode, rnode, node):
        t, f = st.copy(), st.copy()
        if lv.top or rv.top:
            raise Unsupported("comparison with an unknown value", node)
        if isinstance(op, (ast.Is, ast.IsNot)):
            self._refine_is(t, f, op, lv, rv, lnode, rnode)
            return t, f
        if isinstance(op, (ast.In, ast.NotIn)):
            if rv.s is not None and rv.replace(s=None).is_bot():
                result = self.visit_rule("str.__contains__", st, [rv, lv], {}, node)
                yes, no = self.truth_split(st, result, node)
                return (no, yes) if isinstance(op, ast.NotIn) else (yes, no)
            self.check_in(st, lv, rv, node)
            member = self.set_membership(st, lv, rv, node)
            if member is True or member is False:
                (f if member == isinstance(op, ast.In) else t).set_bottom()
            return t, f
        if type(op) in COMPARISON_METHODS and (
                self.instance_value(st, lv) or self.instance_value(st, rv)):
            outs = []
            for left in self.alternatives(lv):
                for right in self.alternatives(rv):
                    branch = st.copy()
                    if self.instance_value(branch, left) or self.instance_value(branch, right):
                        value = self.binary_special(branch, op, left, right, node, comparison=True)
                        if not branch.bottom and value.kinds() != ["bool"]:
                            raise Unsupported("non-bool rich comparison result", node)
                    else:
                        bt, bf = self.compare_split(branch, op, left, right, lnode, rnode, node)
                        value = self._merge_branches(branch, [(bt, const_bool(True)), (bf, const_bool(False))])
                    outs.append((branch, value))
            value = self._merge_branches(st, outs)
            return self.truth_split(st, value, node)
        if self.custom_comparison(st, [lv, rv]):
            raise Unsupported("builtin comparison with user element methods", node)
        if isinstance(op, (ast.Eq, ast.NotEq)) and self.set_equality(st, op, lv, rv, node, t, f):
            return t, f
        if isinstance(op, (ast.Eq, ast.NotEq)):
            # no remaining user comparison callbacks
            dec = self._decide_eq(st, lv, rv)
            if dec is not None:
                eq_true = dec
                if isinstance(op, ast.NotEq):
                    eq_true = not eq_true
                (f if eq_true else t).set_bottom()
                return t, f
            # [internal ref] (MAIN's guarded rewrite, 2026-09-28): X // 2**k and X >> k for the
            # same bare-name X are the same VALUE, not just the same bounds, whenever
            # X is proven non-negative right now -- canonicalize the comparison to
            # "x == x" directly, rather than teaching the general Lin/zone machinery
            # a synthetic identity (tried first; reverted -- see commit history --
            # because a synthetic Lin propagated into write_var/zone_assign for any
            # STORED result and silently discarded its real interval bound).
            if self._shr_floordiv_identity_holds(st, lnode, rnode):
                eq_true = True
                if isinstance(op, ast.NotEq):
                    eq_true = not eq_true
                (f if eq_true else t).set_bottom()
                return t, f
            if lv.only_int() and rv.only_int() and lv.i.lin is not None and rv.i.lin is not None:
                d = lv.i.lin.sub(rv.i.lin).norm(st.zone)
                if isinstance(op, ast.Eq):
                    _refine_le(t.zone, d, 0)
                    _refine_le(t.zone, d.neg(), 0)
                    _refine_ne_lin(f.zone, d)
                else:
                    _refine_le(f.zone, d, 0)
                    _refine_le(f.zone, d.neg(), 0)
                    _refine_ne_lin(t.zone, d)
            self._prune_float_compare(st, op, lv, rv, t, f)
            return t, f
        # ordering comparisons
        if self.set_ordering(st, op, lv, rv, node, t, f):
            return t, f
        ok = self._orderable(st, lv, rv)
        self.check(node, "compare", {"TypeError"}, ok,
                   "ordering between " + ",".join(lv.kinds()) + " and " + ",".join(rv.kinds()), st=st)
        if lv.only_int() and rv.only_int() and lv.i.lin is not None and rv.i.lin is not None:
            d = lv.i.lin.sub(rv.i.lin).norm(st.zone)
            if isinstance(op, ast.Lt):
                _refine_le(t.zone, d, -1)
                _refine_le(f.zone, d.neg(), 0)
            elif isinstance(op, ast.LtE):
                _refine_le(t.zone, d, 0)
                _refine_le(f.zone, d.neg(), -1)
            elif isinstance(op, ast.Gt):
                _refine_le(t.zone, d.neg(), -1)
                _refine_le(f.zone, d, 0)
            elif isinstance(op, ast.GtE):
                _refine_le(t.zone, d.neg(), 0)
                _refine_le(f.zone, d, -1)
        self._prune_float_compare(st, op, lv, rv, t, f)
        return t, f

    _CMP_SYMBOL = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!="}

    def _prune_float_compare(self, st: State, op, lv: V, rv: V, t: State, f: State) -> None:
        """A comparison with a float operand decides a branch when the operands' facts leave it
        no pair: ``values.refine_compare`` (NaN makes every ordered comparison and ``==`` False
        and ``!=`` True; an int is compared with a float exactly) narrows the operands to the pairs
        for which the comparison is true and to those for which it is false, and an empty side is
        an impossible branch.  Only a purely numeric comparison with at least one float operand
        (int against int has the zone); the operands themselves are not narrowed in the branches."""
        symbol = self._CMP_SYMBOL.get(type(op))
        if symbol is None or not (lv.f or rv.f):
            return
        fa, fb = self._numeric_fact(st, lv), self._numeric_fact(st, rv)
        if fa is None or fb is None:
            return
        at, bt, af, bf = fv.refine_compare(symbol, fa, fb)
        if at.is_bottom() or bt.is_bottom():
            t.set_bottom()
        if af.is_bottom() or bf.is_bottom():
            f.set_bottom()

    def _decide_eq(self, st, lv: V, rv: V):
        """True/False when ``lv == rv`` is decided by constants, else None."""
        def const_of(v: V):
            if v.refs or v.st or v.f or v.top:
                return False, None
            vals = []
            if v.i is not None:
                lo, hi = self.ibounds(st, v.i)
                if lo is None or lo != hi:
                    return False, None
                vals.append(lo)
            if v.b:
                if len(v.b) != 1:
                    return False, None
                vals.append(next(iter(v.b)))
            if v.s is not None:
                if v.s == ANY_STR or len(v.s) != 1:
                    return False, None
                vals.append(next(iter(v.s)))
            if v.n:
                vals.append(None)
            if len(vals) != 1:
                return False, None
            return True, vals[0]

        ok1, a = const_of(lv)
        ok2, b = const_of(rv)
        if ok1 and ok2:
            return a == b
        return None

    def _shr_floordiv_identity_holds(self, st: State, lnode, rnode) -> bool:
        """[internal ref] (MAIN's guarded rewrite): True when ``{lnode, rnode}`` are
        ``X // 2**K`` and ``X >> K`` (either order) for the exact same bare NAME
        ``X``, and ``X``'s CURRENT value (read straight from the env, never
        re-evaluated -- a bare name has no side effect to repeat) is provably
        non-negative.  ``a >> k == a // 2**k`` for ``a >= 0`` is exact (CPython
        defines arbitrary-precision ``>>`` as floor division by a power of two,
        true for every sign too, but this only claims the non-negative case the
        ticket asked for).  Deliberately narrow: only a bare ``ast.Name`` for X
        (never a general expression -- evaluating one a second time here could
        repeat a call's side effect), and only Eq/NotEq consult it.
        """
        def floordiv_pow2(n):
            if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.FloorDiv)
                    and isinstance(n.left, ast.Name)
                    and isinstance(n.right, ast.Constant) and isinstance(n.right.value, int)):
                k = _pow2_k(n.right.value)
                if k is not None:
                    return n.left.id, k
            return None

        def rshift_k(n):
            if (isinstance(n, ast.BinOp) and isinstance(n.op, ast.RShift)
                    and isinstance(n.left, ast.Name)
                    and isinstance(n.right, ast.Constant) and isinstance(n.right.value, int)
                    and n.right.value >= 0):
                return n.left.id, n.right.value
            return None

        for a_node, b_node in ((lnode, rnode), (rnode, lnode)):
            fd = floordiv_pow2(a_node)
            sh = rshift_k(b_node)
            if fd is None or sh is None or fd != sh:
                continue
            name, _k = fd
            if st.bottom:
                return False
            fr = st.frames[-1]
            g = st.frames[0]
            v = fr.env.get(name)
            if v is None:
                v = g.env.get(name)
            if v is None or v.top:
                continue
            ii = self._as_int(st, v)
            if ii is None:
                continue
            lo, _hi = self.ibounds(st, ii)
            if lo is not None and lo >= 0:
                return True
        return False

    def _refine_is(self, t: State, f: State, op, lv: V, rv: V, lnode, rnode):
        # `x is None` / `x is not None` on a local name
        name = None
        other = None
        if isinstance(lnode, ast.Name) and rv.kinds() == ["None"]:
            name, other = lnode.id, lv
        elif isinstance(rnode, ast.Name) and lv.kinds() == ["None"]:
            name, other = rnode.id, rv
        if other is None:
            return
        is_true, is_false = (t, f) if isinstance(op, ast.Is) else (f, t)
        if not other.n:
            is_true.set_bottom()
            return
        if other.kinds() == ["None"]:
            is_false.set_bottom()
            return
        for s, keep_none in ((is_true, True), (is_false, False)):
            fr = s.frames[-1]
            target = fr.env if name in fr.env else None
            if target is None:
                continue
            v = target[name]
            nv = NONE if keep_none else v.replace(n=False)
            target[name] = nv
            if keep_none and v.i is not None:
                pass

    def _orderable(self, st, lv: V, rv: V) -> bool:
        num_l = lv.i is not None or bool(lv.b) or lv.f
        num_r = rv.i is not None or bool(rv.b) or rv.f
        if lv.n or rv.n or lv.st or rv.st:
            return False
        if lv.refs or rv.refs:
            return False
        if lv.s is not None and (num_r or rv.s is None):
            return False
        if rv.s is not None and (num_l or lv.s is None):
            return False
        return True

    def _orderable_contract_candidate(self, op, lv: V, rv: V) -> bool | None:
        """Differential-only scalar TypeError prediction from the tested comparison contract.

        ``None`` means the scalar template cannot decide this abstract value pair;
        callers must not treat that refusal as an unsafe comparison.
        """
        if (not isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE))
                or lv.is_bot() or rv.is_bot() or lv.top or rv.top
                or lv.refs or rv.refs or lv.st or rv.st):
            return None
        tags = {
            "int": 0,
            "bool": False,
            "float": 0.0,
            "str": "",
            "NoneType": None,
        }
        left_kinds = {"NoneType" if kind == "None" else kind for kind in lv.kinds()}
        right_kinds = {"NoneType" if kind == "None" else kind for kind in rv.kinds()}
        if (not left_kinds or not right_kinds
                or not left_kinds <= tags.keys() or not right_kinds <= tags.keys()):
            return None

        from tharkun.solve.contracts.safety import Binding, cases
        from tharkun.solve.terms import PY, Refused, eval_term, var

        binding = Binding(var("_left", PY), (var("_right", PY),), type(op).__name__)
        rows = cases("protocol.comparison", "TypeError", binding)
        if isinstance(rows, Refused) or not isinstance(rows, tuple) or not rows:
            return None
        may_raise = False
        for left_kind in left_kinds:
            for right_kind in right_kinds:
                env = {"_left": tags[left_kind], "_right": tags[right_kind]}
                matching = []
                for row in rows:
                    domain = eval_term(row.domain, env)
                    if isinstance(domain, Refused) or type(domain) is not bool:
                        return None
                    if domain:
                        matching.append(row)
                if len(matching) != 1:
                    return None
                raises = matching[0].raises
                if raises is not None:
                    result = eval_term(raises, env)
                    if isinstance(result, Refused) or type(result) is not bool:
                        return None
                    if result:
                        may_raise = True
        return not may_raise

    def check_in(self, st, lv: V, rv: V, node):
        if self.instance_value(st, rv) or self.custom_comparison(st, [lv, rv]):
            raise Unsupported("membership invoking user protocols", node)
        ok = not (rv.i is not None or rv.b or rv.f or rv.n or rv.st)
        for a in rv.refs:
            k = st.heap[a].kind
            if k == "set":
                # the element is hashed (a set is converted to a frozenset first)
                ok = ok and self._hashable(st, lv, node, allow_set=True)[1]
            elif k not in ("list", "tuple", "range", "dict", "dictview", "iterator", "array"):
                ok = False
            if k in ("dict", "dictview"):
                self.check_hash_key(lv, node)
            if k in ("dict", "dictview") and any(st.heap[r].kind in ("list", "dict") for r in lv.refs):
                ok = False  # unhashable key
        if rv.s is not None and (lv.refs or lv.i is not None or lv.f or lv.n or lv.b or lv.st):
            ok = False
        self.check(node, "in", {"TypeError"}, ok, "membership in " + ",".join(rv.kinds()), st=st)

    # ============================================================ operators

    def instance_value(self, st, v):
        return any(st.heap[a].kind == "inst" for a in v.refs)

    @staticmethod
    def alternatives(v):
        rest = v.replace(refs=frozenset())
        return ([rest] if not rest.is_bot() else []) + [
            V(refs=frozenset({a})) for a in sorted(v.refs, key=repr)]

    def map_alternatives(self, st, v, transfer):
        outs = []
        for part in self.alternatives(v):
            branch = st.copy()
            outs.append((branch, transfer(branch, part)))
        return self._merge_branches(st, outs)

    def special(self, st, recv, name, args, node, kwargs=None):
        # Python special-method lookup bypasses the instance dictionary.
        for a in recv.refs:
            ob = st.heap[a]
            cv = self.class_lookup(st, ob.cls, name) if ob.kind == "inst" else None
            if cv is None or self._plain_method(st, cv) is None:
                raise Unsupported(f"unresolved special method {name}", node)
        result = self.call_method(st, recv, name, args, kwargs or {}, node, special=True)
        if result.top:
            # Includes NotImplemented: fallback dispatch is not approximated
            # by accepting the first method's successful executions.
            raise Unsupported(f"unknown result / NotImplemented from {name}", node)
        return result

    def instance_class(self, st, v):
        if len(v.refs) == 1:
            ob = st.heap[next(iter(v.refs))]
            if ob.kind == "inst":
                return ob.cls
        return None

    def subclass(self, st, child, parent):
        while child is not None:
            if child == parent:
                return True
            bases = st.heap[child].bases
            child = bases[0] if bases else None
        return False

    def binary_special(self, st, op, left, right, node, inplace=False, comparison=False):
        """An operator with a user instance operand, dispatched as CPython does (``protocol.binary_operator``, ``protocol.inplace_operator``
        or ``protocol.comparison``: the rule of the operator is what admits the dispatch, [internal ref] step 2)."""
        if self.instance_class(st, left) is None and self.instance_class(st, right) is None:
            return self._binary_special_model(st, op, left, right, node, inplace, comparison)
        qualname = ("protocol.comparison" if comparison else "protocol.inplace_operator" if inplace else "protocol.binary_operator")
        return self.visit_protocol(qualname, node, lambda: self._binary_special_model(st, op, left, right, node, inplace, comparison),
                                   _PROTOCOL_NATIVE[qualname])

    def _binary_special_model(self, st, op, left, right, node, inplace=False, comparison=False):
        lc, rc = self.instance_class(st, left), self.instance_class(st, right)
        if isinstance(op, ast.Mod) and left.s is not None:
            raise Unsupported("string formatting with a user operand", node)
        if inplace and lc is None and (left.refs or left.s is not None):
            raise Unsupported("in-place builtin sequence with user protocol", node)
        if lc is None and rc is None:
            return self.binop(st, op, left, right, node, inplace=inplace)
        if comparison:
            lname, rname = COMPARISON_METHODS[type(op)]
        else:
            name = BINARY_METHODS.get(type(op))
            if name is None:
                raise Unsupported("unmodelled instance operator", node)
            lname, rname = "__" + name + "__", "__r" + name + "__"
            iname = "__i" + name + "__"
            if inplace and lc is not None and self.class_lookup(st, lc, iname) is not None:
                return self.special(st, left, iname, [right], node)
        lm = self.class_lookup(st, lc, lname) if lc is not None else None
        rm = self.class_lookup(st, rc, rname) if rc is not None else None
        linvert = rinvert = False
        if comparison and isinstance(op, ast.NotEq):
            # object.__ne__ delegates to __eq__, unless __ne__ is overridden.
            if lm is None and lc is not None:
                lm = self.class_lookup(st, lc, "__eq__")
                lname, linvert = "__eq__", True
            if rm is None and rc is not None:
                rm = self.class_lookup(st, rc, "__eq__")
                rname, rinvert = "__eq__", True
        if not comparison and lc == rc:
            rm = None  # same-type arithmetic never calls __radd__ etc.

        def invoke(recv, name, other, invert):
            result = self.special(st, recv, name, [other], node)
            if invert and not st.bottom:
                if result.kinds() != ["bool"]:
                    raise Unsupported("non-bool equality fallback", node)
                return V(b=frozenset(not b for b in result.b))
            return result

        reflected_first = (lc is not None and rc is not None and lc != rc and
                           self.subclass(st, rc, lc) and rm is not None and
                           (comparison or rm != self.class_lookup(st, lc, rname)))
        if reflected_first:
            return invoke(right, rname, left, rinvert)
        if lm is not None:
            return invoke(left, lname, right, linvert)
        if rm is not None:
            return invoke(right, rname, left, rinvert)
        if comparison and isinstance(op, (ast.Eq, ast.NotEq)):
            return BOOL  # object identity: either result, no user code
        self.check(node, "operator", {"TypeError"}, False, "no applicable special method", st=st)
        st.set_bottom()
        return BOT

    def custom_comparison(self, st, values):
        # Builtin container comparisons/membership may call element methods.
        # ponytail: refuse those callbacks; add element-wise transfers when needed.
        todo = list(values)
        seen = set()
        while todo:
            v = todo.pop()
            for a in v.refs - seen:
                seen.add(a)
                ob = st.heap[a]
                if ob.kind == "inst":
                    if any(self.class_lookup(st, ob.cls, name) is not None
                           for names in COMPARISON_METHODS.values() for name in names):
                        return True
                elif ob.kind in ("list", "tuple", "dict", "iterator", "array", "set"):
                    todo.extend(ob.values())
                elif ob.kind == "dictview":
                    todo.append(V(refs=ob.dicts))
        return False

    def iterable(self, st, value, node, *, consumer=None):
        """``iter(value)`` for a user instance (its ``__iter__`` body is analysed); anything else is returned as is.

        ``consumer`` names the callable consuming it (``"list"``, ``"sorted"``, ``"list.extend"``,
        ``"tuple"``, ...) when that callable asks the ORIGINAL object for a length hint before iterating
        (CPython's ``PyObject_LengthHint``: it calls ``__len__``, and swallows a TypeError from it).
        Whether a consumer does is a fact about the callable, not about the object:
        ``contracts/adapters/common.py::length_hint_rule`` is the hand-verified table (``list``, ``sorted``,
        ``list.extend`` probe; ``tuple``, ``set``, ``array.array`` do not; a ``for`` loop, an unpacking, ``min``,
        ``sum`` never do, which is ``consumer=None``).  A callable the table does not know is refused when the
        class defines ``__len__``: running the probe where CPython does not, or skipping it where CPython does,
        would each make a raise in ``__len__`` wrongly end, or wrongly miss, the path."""
        if self._try_stack and any(st.heap[a].kind not in ("list", "tuple", "range", "array")
                                   for a in value.refs):
            self._in_try_refuse("iteration with unmodelled non-property raises", node)
        if not self.instance_value(st, value):
            return value

        def convert(branch, part):
            cls = self.instance_class(branch, part)
            if cls is None:
                return part
            def iterate():
                out = self.borrow(branch, self.special(branch, part, "__iter__", [], node))
                if out.kinds() != ["obj"] or any(branch.heap[a].kind != "iterator" for a in out.refs):
                    raise Unsupported("__iter__ must return a modelled iterator", node)
                return out
            result = self.visit_protocol("protocol.iteration", node, iterate, _PROTOCOL_NATIVE["protocol.iteration"])
            if consumer is not None and self.class_lookup(branch, cls, "__len__") is not None:
                from tharkun.solve.contracts.rules import Refused, get_rule

                outcome = get_rule("protocol.length_hint").accept(self.visitor, LengthHintCtx(branch, part, node, consumer))
                if isinstance(outcome, Refused):
                    raise Unsupported(outcome.reason, node)
            return result
        return self.map_alternatives(st, value, convert)

    def _probe_length(self, st, part, node):
        """The ``__len__`` call of ``PyObject_LengthHint``, run on ``st``.  An exception it raises ends the
        path (CPython propagates it), EXCEPT a ``TypeError``: CPython swallows that one and iterates with no
        hint, so execution continues, with whatever state ``__len__`` left behind, in every state in which
        it was raised.  An explicit ``raise`` of one is followed exactly (the state at the raise, unwound to
        this frame depth, is joined into the continuation); a ``TypeError`` from an unproven check leaves no
        such state, so a probe in which one is possible is refused."""
        scope = _ProbeScope(self._fctx_stack[-1])
        depth = len(st.frames)
        self._probe_scopes.append(scope)
        try:
            self._len_model(st, [part], {}, node)
        finally:
            self._probe_scopes.pop()
        if scope.type_error:
            raise Unsupported("the length probe may raise a TypeError (not an explicit raise) that CPython "
                              "swallows: the continuation after it is not modelled", node)
        for raised in scope.raises:
            while len(raised.frames) > depth:
                self.pop_frame(raised)
            _assign_state(st, st.join(raised))

    def _flag_operator(self, st, op, values, node):
        """[internal ref]: readonly builtin enums retain CPython identity; larger domains refuse."""
        import operator
        from .native import abstract, concrete, product
        from ..contracts.rules_strformat import native_limit
        names = BINARY_METHODS | {ast.BitAnd: "and_", ast.BitOr: "or_", ast.BitXor: "xor",
                ast.LShift: "lshift", ast.RShift: "rshift", ast.Invert: "invert",
                ast.UAdd: "pos", ast.USub: "neg"}
        self._in_try_refuse("re.RegexFlag operator (its non-property raises are not routed)", node)
        name = names.get(type(op))
        alternatives = product([concrete(self,st,value) for value in values])
        if name is None or alternatives is None:
            raise Unsupported("re.RegexFlag: non-finite or unsupported operator",node)
        outputs,errors=[],set()
        for args in alternatives:
            refusal=native_limit("format",args,{})
            if refusal: raise Unsupported("re.RegexFlag operator: "+refusal,node)
            try: outputs.append(getattr(operator,name)(*args))
            except (TypeError,ValueError,ZeroDivisionError,OverflowError) as exc:
                errors.update(cls.__name__ for cls in type(exc).__mro__)
        for exc in PROPERTY_EXCS:
            self.check(node,"RegexFlag operator:"+exc,{exc},exc not in errors,"finite callback-free builtin operator")
        if not outputs:
            st.set_bottom();return BOT
        result=abstract(self,st,outputs,node)
        if result is None: raise Unsupported("re.RegexFlag operator: result not represented",node)
        return result

    def unop(self, st: State, op, v: V, node) -> V:
        if any(tag[0] == "native-flag" for tag in v.st):
            return self._flag_operator(st, op, [v], node)
        if v.top:
            raise Unsupported("operator on an unknown value", node)
        out = BOT
        bad = bool(v.s is not None or v.n or v.refs or v.st)
        if isinstance(op, ast.Invert) and v.f:
            bad = True
        ii = self._as_int(st, v)
        if ii is not None:
            if isinstance(op, ast.USub):
                lin = ii.lin.neg() if ii.lin is not None else None
                lo, hi = self.ibounds(st, ii)
                nlo, nhi = _iv_neg((lo, hi))
                out = V(i=IntInfo(lin, None if nlo is None else (ZERO, nlo), None if nhi is None else (ZERO, nhi)))
            elif isinstance(op, ast.UAdd):
                out = V(i=ii)
            elif isinstance(op, ast.Invert):
                lin = ii.lin.neg().shift(-1) if ii.lin is not None else None
                lo, hi = self.ibounds(st, ii)
                nlo, nhi = _iv_neg((lo, hi))
                out = V(i=IntInfo(lin, None if nhi is None else None, None))
                out = V(i=IntInfo(lin, None if nlo is None else (ZERO, nlo - 1), None if nhi is None else (ZERO, nhi - 1)))
        if v.f and not isinstance(op, ast.Invert):
            # [internal ref]: -x (USub) used to collapse a float's whole FloatFact to an
            # unconstrained FLOAT, the same TOP-join bug b_abs had -- which matters
            # here specifically because it fed abs(-e) a fully unbounded operand,
            # defeating any chance of the abs(-e) == abs(e) fact ever being visible
            # for floats.  +x (UAdd) is a no-op and keeps the operand's fact/term
            # unchanged; -x negates the interval exactly via the shared value-level
            # transfer (solve/values.py::neg) and drops the term (no lemma names a
            # negated value).
            if isinstance(op, ast.USub):
                res, _raises = fv.neg(v.f.fact)
                if _raises:
                    self._in_try_refuse("a float negation that may raise", node)
                out = vjoin(out, st.zone, V(f=FloatInfo(res)), st.zone)
            else:
                out = vjoin(out, st.zone, V(f=v.f), st.zone)
        self.check(node, "unary", {"TypeError"}, not bad, "unary operand: " + ",".join(v.kinds()), st=st)
        if out.is_bot():
            st.set_bottom()
        return out

    def _as_int(self, st, v: V) -> IntInfo | None:
        """The int view of v's int and bool parts (bool is an int subclass)."""
        ii = v.i
        if v.b:
            bs = sorted(int(x) for x in v.b)
            bi = IntInfo.interval(bs[0], bs[-1])
            if ii is None:
                return bi
            lo, hi = self.ibounds(st, ii)
            lo = None if lo is None else min(lo, bs[0])
            hi = None if hi is None else max(hi, bs[-1])
            return IntInfo.interval(lo, hi)
        return ii

    def binop(self, st: State, op, l: V, r: V, node, inplace=False, square=False) -> V:
        if isinstance(op, ast.Mod) and l.s is not None and l.replace(s=None).is_bot():
            return self.visit_rule("str.__mod__", st, [l, r], {}, node)
        if any(tag[0] == "native-flag" for value in (l,r) for tag in value.st):
            return self._flag_operator(st, op, [l,r], node)
        if l.top or r.top:
            raise Unsupported("operator on an unknown value", node)
        if self.instance_value(st, l) or self.instance_value(st, r):
            outs = []
            for left in self.alternatives(l):
                for right in self.alternatives(r):
                    branch = st.copy()
                    result = self.binary_special(branch, op, left, right, node, inplace=inplace)
                    outs.append((branch, result))
            return self._merge_branches(st, outs)
        # UNMODELLED MEANS ABORT: a "TypeError" verdict cuts the path, so it must be
        # exact.  Types, classes and functions do support some operators
        # (`int | None` is a UnionType), so an operator on them is not modelled.
        for side in (l, r):
            if side.st or any(st.heap[a].kind in ("class", "func", "bmeth", "bimeth", "range", "dict", "dictview")
                              for a in side.refs):
                # dicts merge with `|`, views are sets: not modelled
                raise Unsupported("operator on a type, class, function, range, dict or view", node)
        out = BOT
        bad = False  # a TypeError is possible
        zde = False  # a ZeroDivisionError is possible
        facts = []
        li, ri = self._as_int(st, l), self._as_int(st, r)
        lnum = li is not None or l.f
        rnum = ri is not None or r.f
        if self._try_stack:
            if l.f or r.f:
                self._in_try_refuse("float arithmetic (its non-property raises are not routed)", node)
            if l.s is not None or r.s is not None or l.refs or r.refs:
                self._in_try_refuse("builtin sequence/set arithmetic (its non-property raises are not routed)", node)
            if isinstance(op, (ast.Div, ast.Pow)):
                self._in_try_refuse("numeric division/power (its OverflowError is not routed)", node)
            if isinstance(op, (ast.LShift, ast.RShift)) and ri is not None:
                lo, _ = self.ibounds(st, ri)
                if lo is None or lo < 0:
                    self._in_try_refuse("shift with a possibly negative count (ValueError)", node)
        # ---- int x int
        if li is not None and ri is not None:
            v, z = self._int_binop(st, op, li, ri, node, facts)
            out = vjoin(out, st.zone, v, st.zone)
            zde = zde or z
        # ---- float involved (lane D's FloatFact transfer functions, plus value terms)
        if lnum and rnum and (l.f or r.f):
            if isinstance(op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
                res, raises, ffact = self._float_binop(st, op, self.float_view(st, l), self.float_view(st, r), node,
                                                       square=square)
                zde = zde or "ZeroDivisionError" in raises
                if ffact:
                    facts.append(ffact)
                out = vjoin(out, st.zone, V(f=res), st.zone)
            else:
                bad = True
        # ---- sets and frozensets (sets.py)
        set_out, set_bad = self.set_binop(st, op, l, r, node, inplace)
        out = vjoin(out, st.zone, set_out, st.zone)
        bad = bad or set_bad
        # ---- sequences and strings
        seq_out, seq_bad = self._seq_binop(st, op, l, r, node, inplace)
        out = vjoin(out, st.zone, seq_out, st.zone)
        bad = bad or seq_bad
        # ---- anything else is a TypeError
        if l.n or r.n or l.st or r.st:
            bad = True
        if (l.s is not None and (rnum and not isinstance(op, (ast.Mult, ast.Mod)))):
            bad = True
        if (r.s is not None and lnum and not isinstance(op, ast.Mult)):
            bad = True
        if isinstance(op, ast.Mult) and l.f and (r.s is not None or r.refs):
            bad = True                    # ``2.5 * 'ab'``, ``2.5 * [1]``: a sequence repeats by an int only
        if (l.refs and rnum and not isinstance(op, ast.Mult)) or (r.refs and lnum and not isinstance(op, ast.Mult)):
            bad = True
        if l.refs and r.s is not None or r.refs and l.s is not None:
            bad = True
        self.check(node, "binop", {"TypeError"}, not bad,
                   f"{type(op).__name__} on {','.join(l.kinds())} and {','.join(r.kinds())}", st=st)
        if isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
            left, right = (node.left, node.right) if isinstance(node, ast.BinOp) else (
                getattr(node, "target", None), getattr(node, "value", None))
            residual = None
            if right is not None:
                residual = (f"({ast.unparse(right)}) == 0" if not isinstance(op, ast.Pow) else
                            f"({ast.unparse(left)}) == 0 and ({ast.unparse(right)}) < 0")
            self.check(node, "divide", {"ZeroDivisionError"}, not zde, "; ".join(facts) or
                       ("divisor excludes 0" if not zde else "divisor may be 0"),
                       residual=residual, operand=right, st=st)
        if out.is_bot():
            st.set_bottom()
        return out

    # ------------------------------------------------------ floats
    def float_view(self, st: State, v: V) -> FloatInfo:
        """The numeric parts of ``v`` (int, bool, float) as one float fact."""
        fact = None
        term = None
        if v.f:
            fact = v.f.fact
            term = v.f.term
        ii = self._as_int(st, v)
        if ii is not None:
            lo, hi = self.ibounds(st, ii)
            ff, _r = fv.to_float(fv.IntFact(lo, hi))
            fact = ff if fact is None else fact.join(ff)
            if v.f:
                term = None
        if fact is None:
            fact = fv.FloatFact.bottom()
        return FloatInfo(fact, term, v.f.why if v.f else "")

    @staticmethod
    def _fconst(a: FloatInfo):
        f = a.fact
        if not f.nan and f.lo == f.hi and math.isfinite(f.lo):
            return f.lo
        return None

    _FOPS = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "truediv",
             ast.FloorDiv: "floordiv", ast.Mod: "mod", ast.Pow: "pow_"}

    def _float_binop(self, st: State, op, a: FloatInfo, b: FloatInfo, node, square=False):
        """(result, may_raise, fact text) of a float operation (at least one side float)."""
        fn = getattr(fv, self._FOPS[type(op)])
        term, why = None, ""
        ta, tb = a.term, b.term
        if isinstance(op, ast.Pow) and ta is not None and ta[0] == "sumsq" and a.fact.has_interval() and a.fact.lo < 0:
            # a sum of squares is >= 0 exactly (IEEE: x*x >= 0, and a sum of non-negatives >= +0)
            a = FloatInfo(fv.FloatFact(0.0, a.fact.hi, a.fact.nan), ta, a.why)
        res, raises = fn(a.fact, b.fact)
        if res is None:
            raise Unsupported("float power that may be complex", node)
        if isinstance(op, (ast.Add, ast.Mult)) and res.has_interval() and res.lo < 0.0:
            fa, fb = a.fact, b.fact
            nonneg = lambda f: f.has_interval() and f.lo >= 0.0
            nonpos = lambda f: f.has_interval() and f.hi <= 0.0
            if (nonneg(fa) and nonneg(fb)) or (isinstance(op, ast.Mult) and nonpos(fa) and nonpos(fb)):
                # IEEE: the sum of two non-negatives, or the product of two same-sign
                # values, is >= 0 exactly; outward rounding must not cross 0
                res = fv.FloatFact(0.0, res.hi, res.nan)
        if isinstance(op, ast.Mult):
            ca, cb = self._fconst(a), self._fconst(b)
            if (ta is not None and ta == tb) or square:
                term = ("sumsq", (ta,)) if ta is not None and ta == tb else None
                lo, _hi = lemmas.sq_interval(a.fact.lo, a.fact.hi) if a.fact.has_interval() else (0.0, 0.0)
                res = fv.FloatFact(max(res.lo, lo), res.hi, res.nan)
            elif ta is not None and ta[0] == "trig" and cb is not None:
                term = ("trig", ta[1], ta[2], ta[3] * cb)
            elif tb is not None and tb[0] == "trig" and ca is not None:
                term = ("trig", tb[1], tb[2], tb[3] * ca)
        elif isinstance(op, ast.Add):
            if ta is not None and tb is not None and ta[0] == "sumsq" and tb[0] == "sumsq":
                bases = tuple(sorted(ta[1] + tb[1], key=repr))
                term = ("sumsq", bases)
                lo, hi, why = self.sumsq_lower(st, bases)
                res = fv.FloatFact(max(lo, res.lo), min(hi, res.hi), res.nan)
        elif isinstance(op, ast.Div):
            if ta is not None and tb is not None:
                term = ("div", ta, tb)
        fact = ""
        if isinstance(op, ast.Pow):
            # ``0.0 ** negative`` is the ZeroDivisionError: the BASE may be 0 and the EXPONENT may be negative.  (The fact used to print
            # the exponent as "divisor in [-1.5, -1.5] may be 0", which reads as a division by a constant: bm_nbody's `r2 ** -1.5`.)
            base, exp = a.fact, b.fact
            fact = (f"base in [{base.lo:.6g}, {base.hi:.6g}]{' or nan' if base.nan else ''}, "
                    f"exponent in [{exp.lo:.6g}, {exp.hi:.6g}]{' or nan' if exp.nan else ''}: "
                    + ("0 ** negative is excluded" if "ZeroDivisionError" not in raises else "the base may be 0 and the exponent negative")
                    + (f" by {a.why}" if "ZeroDivisionError" not in raises and a.why else ""))
        elif isinstance(op, (ast.Div, ast.FloorDiv, ast.Mod)):
            d = b.fact
            fact = (f"divisor in [{d.lo:.6g}, {d.hi:.6g}]{' or nan' if d.nan else ''}"
                    + (" excludes 0" if "ZeroDivisionError" not in raises else " may be 0")
                    + (f" by {b.why}" if b.why else ""))
        return FloatInfo(res, term, why or (a.why if isinstance(op, ast.Add) else "")), raises, fact

    # ------------------------------------------------------ math (every function of the module)
    #: the math functions with a sound interval transfer in solve/values.py (lane D); the others get the
    #: closed range the model table states for them, or any float
    _MATH_FV = {"exp": fv.exp, "log": fv.log, "log10": fv.log10, "atan": fv.atan, "tanh": fv.tanh,
                "fabs": fv.fabs, "atan2": fv.atan2, "hypot": fv.hypot}
    _MATH_ROUND = {"floor": fv.floor, "ceil": fv.ceil, "trunc": fv.trunc}

    def _math_call(self, st, args, kwargs, node, name):
        """``math.<name>(...)``: the contract (``contracts/rules_math.py``) says WHICH property classes the
        call can raise, the shared table (``contracts/adapters/math_model.py``) says what shape it was
        verified for and what it returns, and this model proves the classes it can argue away for THESE
        abstract arguments: every call of the verified shape on exact int/bool/float arguments runs no
        user code, so only TypeError (argument kinds, discharged here) and, for ``log(x, base)``,
        ZeroDivisionError (``base == 1``) can occur; ValueError and OverflowError are not property
        classes (the path ends).  An argument that is not an exact number (a str, None, a container)
        leaves TypeError unproven; an iterable argument and a keyword call are not modelled."""
        from tharkun.solve.contracts.adapters import absint as _contracts
        from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES

        model = _contracts.math_function(name)
        contract = _contracts.contract_for_module_func("math", name)
        if model is None or contract is None or not contract.tested:
            raise Unsupported(f"math.{name} has no tested contract and model", node)
        shape = model.shape(len(args), kwargs)
        if shape == "unmodelled":
            raise Unsupported(f"math.{name} with keyword arguments", node)
        if shape == "typeerror":
            self.check(node, f"math.{name}:TypeError", {"TypeError"}, False,
                       "call shape: CPython raises TypeError (argument count or keyword)")
            st.set_bottom()
            return BOT
        if any(model.kind_of(i) == "iterable" for i in range(len(args))):
            raise Unsupported(f"math.{name} of an iterable is not modelled", node)
        kinds_ok = True
        for i, a in enumerate(args):
            if a.top:
                raise Unsupported("unknown argument", node)
            kind = model.kind_of(i)
            ok = not (a.s is not None or a.refs or a.st) and (kind == "intopt" or not a.n)
            if kind in ("int", "intopt"):
                ok = ok and not a.f
            kinds_ok = kinds_ok and ok
        result = self._math_value(st, args, node, name, model)
        checks = dict((exc, (ok, fact)) for exc, ok, fact in _contracts.raise_checks(contract))
        raised_by_numbers = _contracts.scalar_shape_raises(name)
        why = "a call of the verified shape on exact int/bool/float arguments runs no user code"
        for exc in ALL_PROPERTY_CLASSES:
            if exc == "TypeError":
                if kinds_ok:
                    checks[exc] = (True, "contract+model: argument count and kinds verified")
            elif exc in raised_by_numbers:
                # only log(x, base) is here: it divides by log(base), zero exactly when base == 1
                if len(args) == 1 or self._math_log_base_is_not_one(st, args[1]):
                    checks[exc] = (True, "contract+model: " + ("one-argument log" if len(args) == 1 else "base excludes 1"))
            else:
                checks[exc] = (True, "contract+model: " + why)
        for exc, (ok, fact) in checks.items():
            self.check(node, f"math.{name}:{exc}", {exc}, ok, fact)
        return result

    def _math_log_base_is_not_one(self, st, base: V) -> bool:
        """True iff ``base`` (an int, bool or float value) provably is never exactly 1."""
        if base.s is not None or base.n or base.refs or base.st:
            return False
        f = self.float_view(st, base).fact
        return not f.has_interval() or not (f.lo <= 1.0 <= f.hi)

    def _math_fact(self, st, v: V, exact_int: bool):
        """The solve/values.py fact of a numeric argument: the exact ``IntFact`` for an int/bool-only
        value when ``exact_int`` (``log`` takes an int of any size exactly), else its float view;
        ``None`` for a value mixing int and float parts where the exact view would differ."""
        ii = self._as_int(st, v)
        if exact_int and ii is not None:
            if v.f:
                return None
            lo, hi = self.ibounds(st, ii)
            return fv.IntFact(lo, hi)
        return self.float_view(st, v).fact

    def _math_value(self, st, args, node, name, model):
        """The abstract result of a modelled ``math`` call; ``st`` is set bottom when the call
        certainly raises (a TypeError for an argument with no numeric part, or ValueError /
        OverflowError for every value of the arguments)."""
        for i, a in enumerate(args):
            kind = model.kind_of(i)
            if kind == "iterable":
                continue
            has_number = bool(a.f) or self._as_int(st, a) is not None
            if not has_number and not (kind == "intopt" and a.n):
                st.set_bottom()   # no number among its possible values: the call certainly raises TypeError
                return BOT
        views = [self.float_view(st, a) for i, a in enumerate(args) if model.kind_of(i) == "real"]
        nan_in = any(fi.fact.nan for fi in views)
        result = model.result
        if result == "bool":
            return BOOL
        if result == "int":
            if name in self._MATH_ROUND:
                return self._math_round(st, args[0], name)
            return int_iv(model.int_min, None)
        if result in ("float,int", "float,float"):
            first = V(f=FloatInfo(fv.FloatFact(-math.inf, math.inf, nan_in)))
            second = int_iv(None, None) if result == "float,int" else first
            return V(refs=frozenset({self.new_list(
                st, node, IntInfo.interval(2, 2), vjoin(first, st.zone, second, st.zone), tag="tuple",
                kind="tuple", items=[first, second])}))
        if name in ("sin", "cos", "sqrt"):
            return self._math_trig_sqrt(st, args[0], name, node)
        transfer = self._MATH_FV.get(name)
        if transfer is not None and not (name == "log" and len(args) == 2):
            facts = [self._math_fact(st, a, exact_int=name in ("log", "log10")) for a in args]
            if all(f is not None for f in facts):
                res, _raises = transfer(*facts)   # ValueError / OverflowError: not property classes
                if _raises:
                    self._in_try_refuse(f"math.{name} (ValueError / OverflowError)", node)
                if res.is_bottom():
                    st.set_bottom()
                    return BOT
                return V(f=FloatInfo(res))
        lo, hi = model.float_range or (-math.inf, math.inf)
        return V(f=FloatInfo(fv.FloatFact(lo, hi, nan_in or model.nan_from_numbers)))

    def _math_round(self, st, v: V, name):
        """``floor``/``ceil``/``trunc``: an int or bool is its own result, a float rounds outward
        by its interval (ValueError for nan, OverflowError for an infinity end the path)."""
        ii = self._as_int(st, v)
        out = BOT
        if ii is not None:
            out = V(i=ii)
        if v.f:
            res, _raises = self._MATH_ROUND[name](v.f.fact)
            if _raises:
                self._in_try_refuse(f"math.{name} of a float (ValueError / OverflowError)", None)
            if not res.empty:
                part = int_iv(res.lo, res.hi)
                out = part if out.is_bot() else vjoin(out, st.zone, part, st.zone)
        if out.is_bot():
            st.set_bottom()
        return out

    def _math_trig_sqrt(self, st, v: V, name, node):
        """``sin``/``cos``/``sqrt``: the interval transfer plus the value terms the lemmas use."""
        a = self.float_view(st, v)
        res, _raises = getattr(fv, name)(a.fact)  # ValueError / OverflowError: not property classes
        if _raises:
            self._in_try_refuse(f"math.{name} (ValueError / OverflowError)", node)
        term, why = None, ""
        if name in ("sin", "cos"):
            key = None
            if v.only_int() and v.i.lin is not None:
                s = v.i.lin.single()
                if s is not None and s[0] != ZERO and s[1] == 0:
                    key = ("iv", s[0])
            if key is not None:
                term = ("trig", name, key, 1.0)
        elif name == "sqrt" and a.term is not None and a.fact.has_interval():
            term = ("sqrt", a.term, a.fact.lo, a.fact.hi)
            why = a.why
        if res.is_bottom():
            st.set_bottom()
            return BOT
        return V(f=FloatInfo(res, term, why))

    def _int_binop(self, st: State, op, a: IntInfo, b: IntInfo, node, facts):
        z = st.zone
        ia, ib = self.ibounds(st, a), self.ibounds(st, b)
        la, lb = a.lin, b.lin
        zde = False

        def mk(lin, iv):
            return V(i=IntInfo(lin, None if iv[0] is None else (ZERO, iv[0]), None if iv[1] is None else (ZERO, iv[1])))

        if isinstance(op, ast.Add):
            return mk(la.add(lb) if la is not None and lb is not None else None, _iv_add(ia, ib)), False
        if isinstance(op, ast.Sub):
            lin = la.sub(lb) if la is not None and lb is not None else None
            iv = _iv_add(ia, _iv_neg(ib))
            if lin is not None:
                llo, lhi = lin_bounds(lin, z)
                iv = (_max_lo(iv[0], llo), _min_hi(iv[1], lhi))
            return mk(lin, iv), False
        if isinstance(op, ast.Mult):
            lin = None
            if ia[0] is not None and ia[0] == ia[1] and lb is not None:
                lin = lb.scale(ia[0])
            elif ib[0] is not None and ib[0] == ib[1] and la is not None:
                lin = la.scale(ib[0])
            return mk(lin, _iv_mul(ia, ib)), False
        if isinstance(op, (ast.FloorDiv, ast.Mod, ast.Div)):
            if _iv_contains0(ib):
                zde = True
                facts.append(f"divisor in [{ib[0]}, {ib[1]}] may be 0")
            else:
                facts.append(f"divisor {self.plin(lb)} in [{ib[0]}, {ib[1]}] excludes 0")
            if isinstance(op, ast.Div):
                return FLOAT, zde
            if isinstance(op, ast.Mod):
                # result has the divisor's sign, |r| < |divisor|
                if ib[0] is not None and ib[0] > 0:
                    return mk(None, (0, None if ib[1] is None else ib[1] - 1)), zde
                if ib[1] is not None and ib[1] < 0:
                    return mk(None, (None if ib[0] is None else ib[0] + 1, 0)), zde
                return mk(None, (None, None)), zde
            # floor division
            if ib[0] is not None and ib[0] > 0 and ia[0] is not None and ia[0] >= 0:
                hi = None if ia[1] is None else ia[1] // ib[0]
                lo = 0 if ib[1] is None else ia[0] // ib[1]
                return mk(None, (lo, hi)), zde
            return mk(None, (None, None)), zde
        if isinstance(op, ast.Pow):
            if ib[0] is not None and ib[0] >= 0:
                if ia[0] is not None and ia[0] >= 0:
                    lo = ia[0] ** ib[0] if ib[0] < 64 and ia[0] < 2 ** 16 else None
                    return mk(None, (lo if lo is not None else 0, None)), False
                return mk(None, (None, None)), False
            # negative exponent: a float, ZeroDivisionError on base 0
            if _iv_contains0(ia):
                zde = True
            v = FLOAT
            if ib[1] is None or ib[1] >= 0:
                v = vjoin(mk(None, (None, None)), z, FLOAT, z)
            return v, zde
        if isinstance(op, (ast.LShift, ast.RShift)):
            if isinstance(op, ast.RShift) and ia[0] is not None and ia[0] >= 0:
                # [internal ref]: a >> k == a // 2**k for a >= 0 (true for every k >= 0; CPython
                # defines arbitrary-precision right shift as floor division by a power
                # of two).  With a known constant shift amount this is EXACT, matching
                # the floordiv-by-constant precision just above -- they are the same
                # operation, so they must produce the same bound, not the looser
                # "somewhere between 0 and a's own upper bound" this used to return.
                if ib[0] is not None and ib[0] == ib[1] and ib[0] >= 0:
                    k = ib[0]
                    lo = ia[0] >> k
                    hi = None if ia[1] is None else ia[1] >> k
                    return mk(None, (lo, hi)), False
                return mk(None, (0, ia[1])), False
            if isinstance(op, ast.LShift) and ia[0] is not None and ia[0] >= 0 and ib[0] is not None and ib[0] >= 0:
                lo = ia[0] << ib[0] if ib[0] < 64 else 0
                return mk(None, (lo, None)), False
            return mk(None, (None, None)), False
        if isinstance(op, ast.BitAnd):
            if ia[0] is not None and ia[0] >= 0 and ib[0] is not None and ib[0] >= 0:
                hi = None if ia[1] is None and ib[1] is None else min(x for x in (ia[1], ib[1]) if x is not None)
                return mk(None, (0, hi)), False
            return mk(None, (None, None)), False
        if isinstance(op, (ast.BitOr, ast.BitXor)):
            if ia[0] is not None and ia[0] >= 0 and ib[0] is not None and ib[0] >= 0:
                return mk(None, (0, None)), False
            return mk(None, (None, None)), False
        raise Unsupported(f"operator {type(op).__name__}", node)

    def _seq_binop(self, st: State, op, l: V, r: V, node, inplace):
        """String/list/tuple operands. Returns (value, TypeError possible)."""
        out = BOT
        bad = False
        z = st.zone
        # str
        if l.s is not None:
            if isinstance(op, ast.Add):
                if r.s is not None:
                    out = vjoin(out, z, ANYSTR, z)
                if r.i is not None or r.b or r.f or r.n or r.refs or r.st:
                    bad = True
            elif isinstance(op, ast.Mult):
                if r.i is not None or r.b:
                    out = vjoin(out, z, ANYSTR, z)
                if r.f or r.s is not None or r.n or r.refs:
                    bad = True
            elif isinstance(op, ast.Mod):
                raise Unsupported("%-formatting", node)
            else:
                bad = True
        if r.s is not None and isinstance(op, ast.Mult) and (l.i is not None or l.b):
            out = vjoin(out, z, ANYSTR, z)
        # arrays: only repetition (`arr * n` / `n * arr`, not in place -- [internal ref], the ONE
        # binop the ticket's scope calls for, e.g. `array('B', [0]) * n`) is modelled;
        # every other operator on an array (including `+`, which would need a
        # typecode-aware concatenation this scope does not build) is an unproven
        # TypeError, same as any other genuinely unsupported operand kind. `_seq_repeat`
        # itself is kind-agnostic (reads `ob.typecode` via getattr) so no separate array
        # path is needed there.
        for a in l.refs:
            ob = st.heap[a]
            if ob.kind == "array":
                if isinstance(op, ast.Mult) and inplace:
                    # `arr *= n` repeats the array IN PLACE (unlike a genuine TypeError,
                    # this always SUCCEEDS in real Python -- falling through to `bad = True`
                    # with no `out` contribution would make `out.is_bot()` true and
                    # wrongly `set_bottom()` a path that actually continues; list's own
                    # in-place repetition raises Unsupported for the identical reason).
                    raise Unsupported("in-place array repetition", node)
                if not isinstance(op, ast.Mult):
                    raise Unsupported(f"operator {type(op).__name__} on an array", node)
                ri = self._as_int(st, r)
                if ri is not None:
                    out = vjoin(out, z, self._seq_repeat(st, ob, ri, node), z)
                if r.f or r.n or r.s is not None or r.refs or r.st:
                    bad = True
        for b in r.refs:
            ob = st.heap[b]
            if ob.kind == "array":
                if isinstance(op, ast.Mult):
                    li = self._as_int(st, l)
                    if li is not None:
                        out = vjoin(out, z, self._seq_repeat(st, ob, li, node), z)
                    if l.f or l.n or l.s is not None or l.refs or l.st:
                        bad = True
                elif not l.refs:
                    bad = True
        # lists / tuples
        for a in l.refs:
            ob = st.heap[a]
            if ob.kind in ("array", "set"):
                continue                          # an array's operators are above, a set's are set_binop's
            if ob.kind not in ("list", "tuple"):
                if ob.kind == "inst":
                    raise Unsupported("operator on an instance", node)
                bad = True
                continue
            if isinstance(op, ast.Add):
                if ob.kind == "list" and inplace:
                    out = vjoin(out, z, self._list_extend(st, frozenset({a}), r, node, "list.__iadd__"), z)
                    continue
                for b in r.refs:
                    ob2 = st.heap[b]
                    if ob2.kind != ob.kind:
                        bad = True
                        continue
                    la, _ = self.len_lin(st, ob)
                    lb, _ = self.len_lin(st, ob2)
                    ln = la.add(lb) if la is not None and lb is not None else None
                    ea = self.all_elems(st, ob)
                    eb = self.all_elems(st, ob2)
                    elem = vjoin(ea, z, eb, z)
                    lo = _len_iv(st, ob)
                    lo2 = _len_iv(st, ob2)
                    iv = _iv_add(lo, lo2)
                    na = self.new_list(st, node, IntInfo(ln) if ln is not None else IntInfo.interval(*iv),
                                       elem, tag=ob.kind + "+", kind=ob.kind)
                    out = vjoin(out, z, V(refs=frozenset({na})), z)
                if r.i is not None or r.b or r.f or r.n or r.s is not None or r.st:
                    bad = True
            elif isinstance(op, ast.Mult):
                if inplace and ob.kind == "list":
                    # `xs *= n` repeats the list IN PLACE: every alias sees it (lane F, b01)
                    raise Unsupported("in-place list repetition", node)
                ri = self._as_int(st, r)
                if ri is not None:
                    out = vjoin(out, z, self._seq_repeat(st, ob, ri, node), z)
                if r.f or r.n or r.s is not None or r.refs or r.st:
                    bad = True
            else:
                bad = True
        for b in r.refs:
            ob = st.heap[b]
            if ob.kind in ("array", "set"):
                continue
            if ob.kind not in ("list", "tuple"):
                if ob.kind == "inst":
                    raise Unsupported("operator on an instance", node)
                bad = True
                continue
            if isinstance(op, ast.Mult):
                li = self._as_int(st, l)
                if li is not None:
                    out = vjoin(out, z, self._seq_repeat(st, ob, li, node), z)
            elif not l.refs:
                bad = True
        return out, bad

    def _seq_repeat(self, st: State, ob, n: IntInfo, node) -> V:
        la, liv = self.len_lin(st, ob)
        niv = self.ibounds(st, n)
        ln = None
        if la is not None and la.is_const() and n.lin is not None and niv[0] is not None and niv[0] >= 0:
            ln = n.lin.scale(la.c)
        elem = self.all_elems(st, ob)
        if ln is not None and ln.single() is not None:
            lii = IntInfo(ln)
        else:
            iv = _iv_mul(liv, (0 if niv[0] is None or niv[0] < 0 else niv[0], niv[1]))
            lii = IntInfo.interval(0 if iv[0] is None or iv[0] < 0 else iv[0], iv[1])
        a = self.new_list(st, node, lii, elem, tag=ob.kind + "*", kind=ob.kind, seg=ob.kind == "list",
                          typecode=getattr(ob, "typecode", None))
        return V(refs=frozenset({a}))

    # ============================================================ subscripts
    def index_ok(self, st: State, length_lin: Lin | None, length_iv, idx: IntInfo):
        """Is ``-len <= idx < len``?  Returns (ok, fact)."""
        z = st.zone
        ilo, ihi = self.ibounds(st, idx)
        up = lowok = False
        if length_lin is not None and idx.lin is not None:
            d = idx.lin.sub(length_lin).norm(z)
            dlo, dhi = lin_bounds(d, z)
            up = dhi is not None and dhi <= -1
            s = idx.lin.add(length_lin).norm(z)
            slo, _ = lin_bounds(s, z)
            lowok = slo is not None and slo >= 0
        if not up and ihi is not None and length_iv[0] is not None and ihi <= length_iv[0] - 1:
            up = True
        if not up and length_lin is not None and idx.hi is not None and idx.hi[0] != ZERO:
            # idx <= v + c  and  v + c <= len - 1
            v, c = idx.hi
            d = Lin.var(v, c).sub(length_lin).norm(z)
            _, dhi = lin_bounds(d, z)
            up = dhi is not None and dhi <= -1
        if not lowok and ilo is not None and ilo >= 0:
            lowok = True
        if not lowok and ilo is not None and length_iv[0] is not None and ilo >= -length_iv[0]:
            lowok = True
        fact = (f"index {self.plin(idx.lin)} in [{ilo}, {ihi}], len {self.plin(length_lin)} in "
                f"[{length_iv[0]}, {length_iv[1]}]: " + ("0 <= idx < len" if up and lowok else
                                                         "idx < len" if up else "not proven"))
        return up and lowok, fact

    # ------------------------------------------------------ segmentation
    def all_elems(self, st: State, ob) -> V:
        """Every element of a list or tuple (the prefix when it covers the list)."""
        if ob.kind == "tuple" and ob.items is not None:
            return _tuple_elem(st, ob)
        pre = getattr(ob, "pre", None)
        if pre is None:
            return ob.elem
        z = st.zone
        pv = seg_var(ob.length)
        if z.entails(ob.length, pv, 0):  # len <= P: every element is in the prefix
            return pre
        if z.entails(pv, ZERO, 0):  # P <= 0: no prefix
            return ob.elem
        return vjoin(pre, z, ob.elem, z)

    def elem_at(self, st: State, ob, idx: IntInfo) -> V:
        """The elements that index ``idx`` (already checked in range) can denote."""
        pre = getattr(ob, "pre", None)
        if pre is None:
            return ob.elem
        z = st.zone
        pv = seg_var(ob.length)
        lo, _ = self.ibounds(st, idx)
        if lo is not None and lo >= 0 and idx.lin is not None:
            d = idx.lin.sub(Lin.var(pv)).norm(z)
            dlo, dhi = lin_bounds(d, z)
            if dhi is not None and dhi <= -1:
                return pre
            if dlo is not None and dlo >= 0:
                return ob.elem
        return self.all_elems(st, ob)

    def store_elem(self, st: State, ob, strong: bool, idx: IntInfo, v: V):
        """``ob[idx] = v`` on a list object (idx checked in range on this path)."""
        pre = getattr(ob, "pre", None)
        if pre is None:
            return ob.replace(elem=self.join_into(st, ob.elem, v))
        if not strong:
            return ob.replace(elem=self.join_into(st, ob.elem, v), pre=self.join_into(st, pre, v))
        z = st.zone
        pv = seg_var(ob.length)
        lo, _ = self.ibounds(st, idx)
        if idx.lin is not None and lo is not None and lo >= 0:
            d = idx.lin.sub(Lin.var(pv)).norm(z)
            dlo, dhi = lin_bounds(d, z)
            if dlo == 0 and dhi == 0:
                # the store writes exactly the first slot past the prefix: extend it
                nob = ob.replace(pre=self.join_into(st, pre, v))
                z.assign_var(pv, pv, 1)
                return st.shift_bounds(pv, 1, nob)
            if dhi is not None and dhi <= -1:
                return ob.replace(pre=self.join_into(st, pre, v))
            if dlo is not None and dlo >= 1:
                return ob.replace(elem=self.join_into(st, ob.elem, v))
        return ob.replace(elem=self.join_into(st, ob.elem, v), pre=self.join_into(st, pre, v))

    def collapse(self, st: State, addrs) -> None:
        """Forget the segmentation of these lists (an operation shifts indices)."""
        for a in addrs:
            ob = st.heap.get(a)
            if ob is None or ob.kind != "list" or ob.pre is None:
                continue
            allv = vjoin(ob.elem, st.zone, ob.pre, st.zone)
            st.heap[a] = ListObj(ob.length, allv, None)
            st.kill_var(seg_var(ob.length))

    @staticmethod
    def _attr_residual(obj_node, name, v: V):
        if obj_node is None:
            return None
        o = ast.unparse(obj_node)
        if v.n and not v.refs:
            return f"({o}) is None"
        if v.n:
            return f"({o}) is None or not hasattr({o}, {name!r})"
        return f"not hasattr({o}, {name!r})"

    def _index_residual(self, node, st) -> dict:
        if isinstance(node, ast.Subscript) and not isinstance(node.slice, ast.Slice):
            seq, idx = ast.unparse(node.value), ast.unparse(node.slice)
            return {"residual": f"not (-len({seq}) <= ({idx}) < len({seq}))", "operand": node.slice, "st": st}
        return {}

    def subscript_load(self, st: State, obj: V, idx, node) -> V:
        quals = self._subscript_quals(st, obj, "__getitem__") if not obj.top else []
        if not quals:
            return self._subscript_load_model(st, obj, idx, node)
        return self._through_subscript_rules(quals, st, node, idx, lambda: self._subscript_load_model(st, obj, idx, node))

    def _subscript_load_model(self, st: State, obj: V, idx, node) -> V:
        if obj.s is not None and obj.replace(s=None).is_bot():
            if idx[0] == "index":
                key = idx[1]
            else:
                from .native import concrete, product
                options = product([concrete(self, st, x) if x is not None else [None] for x in idx[1:4]])
                key = None
                if options is not None:
                    try:
                        key = V(st=frozenset(("native-slice", slice(*xs)) for xs in options))
                    except TypeError:
                        pass            # a bound with no hash (a list, a dict): not finitely many VALUES a tag can hold
                if key is None:
                    # bounds that are not finitely many values: an int-bound slice when every present bound is an int or None
                    # (``s[:len(t)]``, ``s[:(1 if c else None)]``), which never raises; a slice with bounds that run no user code
                    # but may not be ints (a float, a list, a range: a TypeError); anything else (an object, an unknown value) is
                    # not modelled
                    bounds = [x for x in idx[1:4] if x is not None]
                    if all(int_or_none_value(x) for x in bounds):
                        key = V(st=frozenset({("int-slice",)}))
                    elif all(hook_free_value(st, x) for x in bounds):
                        key = V(st=frozenset({("bad-slice",) if any(non_int_value(x) for x in bounds) else ("any-slice",)}))
                    else:
                        raise Unsupported("str.__getitem__: non-finite slice bounds", node)
            return self.visit_rule("str.__getitem__", st, [obj, key], {}, node)
        if obj.st and all(t[0] in {"native-regex","abstract-match"} for t in obj.st):
            if idx[0] != "index":
                raise Unsupported("re.Match.__getitem__: slice", node)
            return self.visit_rule("re.Match.__getitem__", st, [obj, idx[1]], {}, node)
        if obj.top:
            raise Unsupported("subscript of an unknown value", node)
        if self.instance_value(st, obj):
            if idx[0] != "index":
                raise Unsupported("instance slice protocol", node)
            return self.map_alternatives(st, obj, lambda branch, part:
                self.visit_protocol("protocol.container_getitem", node,
                                    lambda: self.special(branch, part, "__getitem__", [idx[1]], node),
                                    _PROTOCOL_NATIVE["protocol.container_getitem"])
                if self.instance_value(branch, part) else self.subscript_load(branch, part, idx, node))
        if obj.st or any(st.heap[a].kind in ("class", "func") for a in obj.refs):
            raise Unsupported("subscript of a type or class (a generic alias)", node)
        out = BOT
        tbad = bool(obj.i is not None or obj.b or obj.f or obj.n or obj.st)
        ibad = False
        for a in obj.refs:
            ob = st.heap[a]
            if ob.kind in ("list", "tuple", "array"):
                if idx[0] == "slice":
                    out = vjoin(out, st.zone, self.slice_load(st, a, ob, idx, node), st.zone)
                    continue
                iv = idx[1]
                ii = self._as_int(st, iv)
                if iv.top:
                    raise Unsupported("unknown index", node)
                if iv.f or iv.s is not None or iv.n or iv.refs or iv.st:
                    tbad = True
                if ii is None:
                    continue
                if ob.kind == "tuple" and ob.items is not None:
                    lo, hi = self.ibounds(st, ii)
                    n = len(ob.items)
                    ok = lo is not None and hi is not None and -n <= lo and hi < n
                    self.check(node, "index", {"IndexError"}, ok, f"tuple index in [{lo}, {hi}], len {n}", st=st)
                    if lo is None or hi is None:
                        cands = range(-n, n)
                    else:
                        cands = range(max(lo, -n), min(hi, n - 1) + 1)
                    for k in cands:
                        out = vjoin(out, st.zone, ob.items[k], st.zone)
                    continue
                llin, liv = self.len_lin(st, ob)
                ok, fact = self.index_ok(st, llin, liv, ii)
                self.check(node, "index", {"IndexError"}, ok, fact, **self._index_residual(node, st))
                ibad = ibad or not ok
                e = self.elem_at(st, ob, ii)
                if not e.is_bot():
                    out = vjoin(out, st.zone, e, st.zone)
            elif ob.kind == "range":
                raise Unsupported("range subscript", node)
            elif ob.kind == "inst":
                raise Unsupported("instance subscript", node)
            elif ob.kind == "dict":
                if idx[0] == "slice":
                    if isinstance(ob, DefaultDictObj):
                        raise Unsupported("defaultdict slice keys", node)
                    # a slice is hashable since 3.12 (unless a bound is not): ``{}[0:1]`` is a KeyError, never a
                    # TypeError, and no key of a modelled dict is a slice
                    self.check(node, "key", {"KeyError"}, False, "a slice is never a key of a modelled dict", st=st)
                    tbad = True
                    continue
                kv = idx[1]
                if kv.top:
                    raise Unsupported("unknown key", node)
                if any(st.heap[r].kind in ("list", "dict") for r in kv.refs):
                    tbad = True  # unhashable key
                val, ok, fact = self.dict_load(st, ob, kv, node)
                if isinstance(ob, DefaultDictObj) and ob.factory == "list" and not ok:
                    keys = self.key_consts(st, kv)
                    if keys is None:
                        # Snapshot a scalar key before a repeated factory allocation can recycle its zone variables.
                        key_summary = self.summarize(st, self.mat(st, kv))
                    else:
                        from .state import _key_value
                        key_summary = BOT
                        for key in keys:
                            key_summary = vjoin(key_summary, st.zone, _key_value(key), st.zone)
                    # The new list may reuse/demote its previous allocation site; in-flight old values must be guarded.
                    self.borrow(st, val)
                    address = self.new_list(st, node, IntInfo.interval(0, 0), BOT,
                                             tag="defaultdict missing list")
                    fresh = V(refs=frozenset({address}))
                    self._defaultdict_store_factory_list(
                        st, a, key_summary, keys, fresh, strong=len(obj.refs) == 1 and a[1] == "r")
                    val = vjoin(val, st.zone, fresh, st.zone)
                    ok = True
                    fact = "defaultdict(list) inserts a fresh list for a missing scalar key"
                res = None
                if isinstance(node, ast.Subscript):
                    res = f"({ast.unparse(node.slice)}) not in {ast.unparse(node.value)}"
                self.check(node, "key", {"KeyError"}, ok, fact, residual=res,
                           operand=getattr(node, "slice", None), st=st)
                if not val.is_bot():
                    out = vjoin(out, st.zone, val, st.zone)
            else:
                tbad = True
        if obj.s is not None:
            if idx[0] == "slice":
                self._slice_parts(st, idx, node)
                out = vjoin(out, st.zone, ANYSTR, st.zone)
            else:
                iv = idx[1]
                if iv.top:
                    raise Unsupported("unknown index", node)
                # an index that is not an int or a bool is a TypeError (``'ab'[1.5]``), and no value comes of it
                if iv.f or iv.s is not None or iv.n or iv.refs or iv.st:
                    tbad = True
                self.check(node, "index", {"IndexError"}, False, "string index: string length not tracked", st=st)
                if self._as_int(st, iv) is not None:
                    out = vjoin(out, st.zone, ANYSTR, st.zone)
        self.check(node, "subscript", {"TypeError"}, not tbad, "subscript of " + ",".join(obj.kinds()), st=st)
        if out.is_bot():
            st.set_bottom()
            return BOT
        return self.mat(st, out)

    def _slice_parts(self, st, idx, node):
        self._in_try_refuse("slice operation (its ValueError is not routed)", node)
        _, lo, hi, step, _sl = idx
        for part in (lo, hi, step):
            if part is None:
                continue
            if part.top:
                raise Unsupported("unknown slice bound", node)
            bad = bool(part.f or part.s is not None or part.refs or part.st)
            self.check(node, "slice", {"TypeError"}, not bad, "slice bounds are int or None", st=st)
        stepc = None
        if step is None or (step.n and step.i is None and not step.b):
            stepc = 1
        elif step.only_int():
            a, b = self.ibounds(st, step.i)
            if a is not None and a == b:
                stepc = a
        return lo, hi, stepc

    def slice_load(self, st: State, a, ob, idx, node) -> V:
        lo, hi, stepc = self._slice_parts(st, idx, node)
        llin, liv = self.len_lin(st, ob)
        z = st.zone
        newlen = None  # an exact Lin
        if llin is not None:
            L = llin

            def val(v):
                if v is None or (v.n and v.i is None):
                    return None
                if v.only_int() and v.i.lin is not None:
                    return v.i.lin
                return False

            s, e = val(lo), val(hi)
            if stepc == 1 and s is not False and e is not False:
                s_ = Lin.const(0) if s is None else s
                e_ = L if e is None else e
                if (lin_entails_le(z, Lin.const(0), s_) and lin_entails_le(z, s_, e_)
                        and lin_entails_le(z, e_, L)):
                    newlen = e_.sub(s_).norm(z)
            elif stepc == -1 and e is None and s is not False:
                if s is None:
                    newlen = L
                elif lin_entails_le(z, Lin.const(0), s) and lin_entails_le(z, s, L, -1):
                    newlen = s.shift(1).norm(z)
        elem = self.all_elems(st, ob)
        if newlen is not None:
            lii = IntInfo(newlen)
        else:
            lii = IntInfo(None, (ZERO, 0), (llin.single() if llin is not None and llin.single() else
                                             (None if liv[1] is None else (ZERO, liv[1]))))
            if lii.hi is None:
                lii = IntInfo(None, (ZERO, 0), None)
        na = self.new_list(st, node, lii, elem, tag=ob.kind + "[:]", kind=ob.kind,
                           typecode=getattr(ob, "typecode", None))
        return V(refs=frozenset({na}))

    def subscript_store(self, st: State, obj: V, idx, v: V, node) -> None:
        quals = self._subscript_quals(st, obj, "__setitem__") if not obj.top else []
        if not quals:
            return self._subscript_store_model(st, obj, idx, v, node)
        return self._through_subscript_rules(quals, st, node, idx, lambda: self._subscript_store_model(st, obj, idx, v, node))

    def _subscript_store_model(self, st: State, obj: V, idx, v: V, node) -> None:
        if obj.top:
            raise Unsupported("subscript store on an unknown value", node)
        if self.instance_value(st, obj):
            if idx[0] != "index":
                raise Unsupported("instance slice assignment", node)
            def store(branch, part):
                if self.instance_value(branch, part):
                    self.special(branch, part, "__setitem__", [idx[1], v], node)
                else:
                    self.subscript_store(branch, part, idx, v, node)
                return NONE
            self.map_alternatives(st, obj, store)
            return
        tbad = bool(obj.i is not None or obj.b or obj.f or obj.n or obj.st or obj.s is not None)
        lists = set()
        arrays = set()
        dicts = set()
        for a in obj.refs:
            ob = st.heap[a]
            if ob.kind == "list":
                lists.add(a)
            elif ob.kind == "array":
                arrays.add(a)
            elif ob.kind == "dict":
                dicts.add(a)
            elif ob.kind == "inst":
                raise Unsupported("instance item assignment", node)
            else:
                tbad = True
        if dicts:
            if idx[0] == "slice":
                raise Unsupported("dict slice assignment", node)
            kv = idx[1]
            self.check_hash_key(kv, node)
            if kv.top:
                raise Unsupported("unknown key", node)
            if any(st.heap[r].kind in ("list", "dict") for r in kv.refs):
                tbad = True
            ks = self.key_consts(st, kv)
            strong = len(dicts) == 1 and next(iter(dicts))[1] == "r" and ks is not None and len(ks) == 1
            sv = self.summarize(st, v)
            for a in dicts:
                ob = st.heap[a]
                items = dict(ob.items)
                if strong:
                    items[next(iter(ks))] = sv
                    st.heap[a] = ob.replace(items=items)
                else:
                    for k, old in items.items():
                        if self._dict_key_may_equal(kv, ks, k):
                            items[k] = vjoin(old, st.zone, sv, st.zone)
                    st.heap[a] = ob.replace(items=items, exact=False,
                                            okeys=vjoin(ob.okeys, st.zone, kv, st.zone),
                                            ovals=vjoin(ob.ovals, st.zone, sv, st.zone))
            if not lists and not arrays:
                self.check(node, "subscript", {"TypeError"}, not tbad, "item assignment on " + ",".join(obj.kinds()), st=st)
                return
        if idx[0] == "slice":
            if arrays:
                # real array slice assignment additionally requires the source's LENGTH to
                # match for an extended (non-unit-step) slice, and re-validates every
                # element against the typecode -- narrow, unmodelled shape; refuse rather
                # than approximate it ([internal ref] scope is read-slicing, append, extend).
                raise Unsupported("array slice assignment", node)
            for a in lists:
                self.slice_store(st, a, idx, v, node, strong=len(lists) == 1)
            self.check(node, "subscript", {"TypeError"}, not tbad, "slice assignment on " + ",".join(obj.kinds()), st=st)
            return
        iv = idx[1]
        if iv.top:
            raise Unsupported("unknown index", node)
        if iv.f or iv.s is not None or iv.n or iv.refs or iv.st:
            tbad = True
        ii = self._as_int(st, iv)
        self.check(node, "subscript", {"TypeError"}, not tbad, "item assignment on " + ",".join(obj.kinds()), st=st)
        if ii is None:
            st.set_bottom()
            return
        for a in lists:
            ob = st.heap[a]
            llin, liv = self.len_lin(st, ob)
            ok, fact = self.index_ok(st, llin, liv, ii)
            self.check(node, "index", {"IndexError"}, ok, fact, **self._index_residual(node, st))
        typecode = None
        if arrays:
            typecodes = {st.heap[a].typecode for a in arrays}
            if len(typecodes) != 1:
                raise Unsupported("item assignment on arrays of different typecodes", node)
            typecode = next(iter(typecodes))
            for a in arrays:
                ob = st.heap[a]
                llin, liv = self.len_lin(st, ob)
                ok, fact = self.index_ok(st, llin, liv, ii)
                self.check(node, "index", {"IndexError"}, ok, fact, **self._index_residual(node, st))
        if not lists and not arrays:
            st.set_bottom()
            return
        if lists:
            self.update_obj(st, lists, lambda ob, strong: self.store_elem(st, ob, strong, ii, v))
        if arrays:
            checked_v = self._array_store_value(st, node, typecode, v)
            self.update_obj(st, arrays, lambda ob, strong: self.store_elem(st, ob, strong, ii, checked_v))

    def slice_store(self, st: State, a, idx, v: V, node, strong: bool) -> None:
        if any(st.heap[r].kind in ("inst", "iterator") for r in v.refs):
            raise Unsupported("slice assignment from a user protocol or iterator", node)
        lo, hi, stepc = self._slice_parts(st, idx, node)
        self.collapse(st, {a})
        ob = st.heap[a]
        z = st.zone
        # the source must be iterable
        src_elem = BOT
        src_len = None
        bad = bool(v.i is not None or v.b or v.f or v.n or v.st)
        if v.top:
            raise Unsupported("slice assignment from an unknown value", node)
        if v.s is not None:
            src_elem = ANYSTR
        for b in v.refs:
            ob2 = st.heap[b]
            if ob2.kind in ("list", "tuple", "array"):
                e2 = self.all_elems(st, ob2)
                src_elem = vjoin(src_elem, z, e2, z)
                if len(v.refs) == 1 and v.s is None:
                    src_len = self.len_lin(st, ob2)[0]
            elif ob2.kind in ("range", "dict", "dictview", "set"):
                # a valid source: its elements are in the list afterwards, and its length is not used.  ([internal ref]: these
                # fell into the branch below, the one for a source that is NOT iterable -- a recorded TypeError, and the
                # path went on with the list's OLD elements, so ``xs = [1]; xs[0:1] = range(0, 1); 10 // xs[0]`` was a
                # wrong TRUE.)  dict.items() has no element summary: refused.
                if ob2.kind == "dictview" and ob2.which == "items":
                    raise Unsupported("slice assignment from dict.items()", node)
                ev = self.elem_summary(st, V(refs=frozenset({b})))
                if ev is not None:
                    src_elem = vjoin(src_elem, z, ev, z)
            elif ob2.kind == "stringio":
                raise Unsupported("slice assignment from a StringIO (its lines are not modelled)", node)
            else:
                bad = True
        self.check(node, "slice-assign", {"TypeError"}, not bad, "slice assignment source is a sequence", st=st)
        llin, liv = self.len_lin(st, ob)
        newlen = None
        if stepc == 1 and llin is not None and src_len is not None:
            def val(x):
                if x is None or (x.n and x.i is None):
                    return None
                if x.only_int() and x.i.lin is not None:
                    return x.i.lin
                return False

            s, e = val(lo), val(hi)
            if s is not False and e is not False:
                s_ = Lin.const(0) if s is None else s
                e_ = llin if e is None else e
                if (lin_entails_le(z, Lin.const(0), s_) and lin_entails_le(z, s_, e_)
                        and lin_entails_le(z, e_, llin)):
                    newlen = llin.sub(e_.sub(s_)).add(src_len).norm(z)
        elif stepc is not None and stepc != 1:
            newlen = llin  # extended slices keep the length (or raise ValueError)

        def upd(ob, strong_):
            nob = ob.replace(elem=self.join_into(st, ob.elem, src_elem) if not src_elem.is_bot() else ob.elem)
            if newlen is not None and newlen == llin:
                return nob
            if newlen is not None:
                return self.set_len(st, nob, strong_, newlen)
            return self.set_len(st, nob, strong_, None, 0, None)

        self.update_obj(st, {a}, upd)

    # ============================================================ attributes
    def class_lookup(self, st: State, cls_addr, name):
        if name.startswith("__") and name not in SUPPORTED_DUNDERS | {
                "__init__", "__str__", "__repr__", "__format__", "__bool__"}:
            raise Unsupported("reflective or unmodelled class attribute " + name)
        seen = 0
        a = cls_addr
        while a is not None and seen < 50:
            co = st.heap.get(a)
            if co is None:
                return None
            if name in co.attrs:
                return co.attrs[name]
            a = co.bases[0] if co.bases else None
            seen += 1
        if a is not None:
            raise Unsupported("class lookup depth")
        return None

    def _mro_has_unmodelled_base(self, st: State, cls_addr) -> bool:
        """True if walking ``cls_addr``'s MRO (the same walk ``class_lookup`` does) passes
        through a class whose real base is a builtin this analyzer does not represent as a
        heap class -- currently only the exception family (`class E(Exception): pass`,
        recorded as the special `__exc_base__` marker attribute in `s_ClassDef`,
        never as a real entry in `bases`). [internal ref] part 2 (soundness, MG): when this is
        True, ``class_lookup()`` returning ``None`` for some OTHER, non-dunder name (``args``,
        ``with_traceback``, ...) does not mean the attribute is absent -- it means this
        analyzer's own partial view of the hierarchy stops short of the real one, which a
        real ``Exception``/``ValueError``/``BaseException``/``KeyError`` instance still
        provides. General rule (MG): every builtin base -- the exception family today, and
        any of ``int``/``str``/``list``/``dict`` and so on a future extension might allow as
        a base -- carries attributes outside the user-class model; a name miss anywhere in
        such an MRO is unknown, never a confident absence.
        """
        seen = 0
        a = cls_addr
        while a is not None and seen < 50:
            co = st.heap.get(a)
            if co is None:
                return False
            if "__exc_base__" in co.attrs:
                return True
            a = co.bases[0] if co.bases else None
            seen += 1
        return False

    def attr_load(self, st: State, v: V, name: str, node) -> V:
        """The read of an attribute: of a user instance it is ``protocol.attribute_access``'s (the rule of the lookup, [internal ref] step 2), the
        other receivers (a class, a builtin container, a module) are the model alone."""
        if not v.top and any(st.heap[a].kind == "inst" for a in v.refs):
            return self.visit_protocol("protocol.attribute_access", node, lambda: self._attr_load_model(st, v, name, node), None)
        return self._attr_load_model(st, v, name, node)

    def _attr_load_model(self, st: State, v: V, name: str, node) -> V:
        if v.top:
            raise Unsupported("attribute of an unknown value", node)
        out = BOT
        bad = False
        methods: dict = {}  # function address -> receivers
        for a in v.refs:
            ob = st.heap[a]
            if ob.kind == "inst":
                got = ob.attrs.get(name)
                # `name`'s FIRST hit walking ob.cls's MRO: ("slot", class_addr) for
                # an intact slot descriptor declared THERE (in that class's own
                # __slots__), ("attr", value) for a plain class-level value (an
                # ordinary class attribute/method, or a class-level override of
                # what used to be a slot descriptor -- `Cls.x = <plain value>`,
                # attr_store's class-kind branch, replaces that MRO entry outright),
                # or None if `name` is not a class-level attribute or slot anywhere
                # in the MRO.  [internal ref] review (MG): the earlier version of this check
                # used `_all_slots(ob.cls) is not None`, which bails to "no slots"
                # (has a __dict__) the moment ANY class in the MRO -- own or not --
                # lacks __slots__, so it never even looked for an override that
                # landed on a BASE class while the instantiated subclass itself
                # never redeclares __slots__ (the common case for empty subclasses)
                # -- a confirmed wrong TRUE (mg1072 b01-b03, b07).
                hit = self._mro_attr_or_slot(st, ob.cls, name)
                if got is not None:
                    gv = got[0]
                    if gv.f and len(v.refs) == 1 and a[1] == "r" and got[2] is not None:
                        # a read of one concrete object: the value's identity is its
                        # stored term, or "attribute `name` of `a` at version ver"
                        term = gv.f.term if gv.f.term is not None else ("rd", a, name, got[2])
                        gv = gv.replace(f=FloatInfo(gv.f.fact, term, gv.f.why))
                    out = vjoin(out, st.zone, self.mat(st, gv), st.zone)
                    if got[1] and (hit is None or hit[0] == "slot"):
                        continue
                if hit is None or hit[0] == "slot":
                    # [internal ref] (soundness) x [internal ref] (MRO lookup), merged by MM: no plain
                    # class-level value governs `name` here -- nothing in the modeled MRO
                    # ([internal ref]'s `class_lookup() is None`) or only an intact slot descriptor.
                    # A reflective/dunder name or an MRO through an unmodelled builtin base
                    # makes that miss "unknown", never a confident absence: refuse it, for
                    # the slot case too (refusing is only ever a loss of precision).
                    if _reflective_unresolved_name(name) or self._mro_has_unmodelled_base(st, ob.cls):
                        raise Unsupported(f"reflective read of the implicit/unresolved {name}", node)
                    # Otherwise: either an ordinary (non-slot) instance attribute with
                    # nothing recorded (or not definitely recorded) on every alias, or an
                    # intact slot descriptor with the same -- either way, unset on some
                    # reachable execution is a real AttributeError.
                    bad = True
                    continue
                # hit[0] == "attr": a plain class-level value is the FIRST thing
                # CPython's MRO lookup finds for `name` -- it is not a data
                # descriptor (this model does not represent properties/custom
                # descriptors; any that appear are refused elsewhere), so it wins
                # over a slot descriptor found later in the MRO. It does NOT
                # necessarily win over the instance's OWN recorded value the way
                # class attributes and instance __dict__ entries generally
                # interact, though: whether that recorded value is really invisible
                # (written back when `name` was still an intact slot -- CPython
                # then stored it through the descriptor, not into any __dict__) or
                # a genuine, still-visible __dict__ entry (written after the
                # override, on a hierarchy that has a __dict__ somewhere) depends
                # on when it was written, which this model does not track per
                # value. Both stay joined into `out` above -- sound either way --
                # and the class-level value joins in too, rather than replacing it.
                cv = hit[1]
                plain = self._plain_method(st, cv)
                if plain is not None:
                    methods.setdefault(plain, set()).add(a)
                    continue
                out = vjoin(out, st.zone, self.bind_class_attr(st, cv, V(refs=frozenset({a})), node), st.zone)
            elif ob.kind == "class":
                cv = self.class_lookup(st, a, name)
                if cv is None:
                    if _reflective_unresolved_name(name) or self._mro_has_unmodelled_base(st, a):
                        raise Unsupported(f"reflective read of the implicit/unresolved {name}", node)
                    bad = True
                    continue
                out = vjoin(out, st.zone, self.bind_class_attr(st, cv, None, node, cls_addr=a), st.zone)
            elif ob.kind in ("list",):
                if name in LIST_METHODS:
                    ma = self.alloc(st, node, lambda sid: BuiltinMethod(name, V(refs=frozenset({a}))), "bimeth")
                    out = vjoin(out, st.zone, V(refs=frozenset({ma})), st.zone)
                else:
                    raise Unsupported(f"list attribute {name}", node)
            elif ob.kind == "array":
                if name in ARRAY_METHODS:
                    ma = self.alloc(st, node, lambda sid: BuiltinMethod(name, V(refs=frozenset({a}))), "bimeth")
                    out = vjoin(out, st.zone, V(refs=frozenset({ma})), st.zone)
                else:
                    raise Unsupported(f"array attribute {name}", node)
            elif ob.kind == "set":
                if name in (FROZENSET_METHODS if ob.frozen else SET_METHODS):
                    ma = self.alloc(st, node, lambda sid: BuiltinMethod(name, V(refs=frozenset({a}))), "bimeth")
                    out = vjoin(out, st.zone, V(refs=frozenset({ma})), st.zone)
                elif _reflective_unresolved_name(name):
                    raise Unsupported(f"attribute {name} of a set", node)
                else:
                    bad = True            # a set has no attribute that is not a method: an AttributeError
            else:
                raise Unsupported(f"attribute {name} of a {ob.kind}", node)
        for t in v.st:
            out = vjoin(out, st.zone, self.static_attr(st, t, name, node), st.zone)
        for fa, recvs in sorted(methods.items(), key=lambda kv: repr(kv[0])):
            self_v = V(refs=frozenset(recvs))
            ma = self.alloc(st, node, lambda sid: BoundMethod(fa, self_v), "bmeth:" + self.sid(fa[0]))
            out = vjoin(out, st.zone, V(refs=frozenset({ma})), st.zone)
        if v.n:
            bad = True
        if v.i is not None or v.b or v.f or v.s is not None:
            raise Unsupported(f"attribute {name} of a builtin scalar", node)
        obj_node = getattr(node, "value", None)
        self.check(node, "attr", {"AttributeError"}, not bad, f".{name} on " + ",".join(v.kinds()),
                   residual=self._attr_residual(obj_node, name, v), operand=obj_node, st=st)
        if out.is_bot():
            st.set_bottom()
        return out

    def _plain_method(self, st: State, cv: V):
        """The function address when a class attribute is exactly one plain method."""
        if cv.top or len(cv.refs) != 1 or cv.st or cv.i is not None or cv.b or cv.f or cv.s is not None or cv.n:
            return None
        fa = next(iter(cv.refs))
        fo = st.heap.get(fa)
        if fo is not None and fo.kind == "func" and fo.flavor == "plain":
            return fa
        return None

    def bind_class_attr(self, st: State, cv: V, self_v: V | None, node, cls_addr=None) -> V:
        out = BOT
        for fa in cv.refs:
            fo = st.heap.get(fa)
            if fo is not None and fo.kind == "func":
                if fo.flavor == "staticmethod" or (self_v is None and fo.flavor == "plain"):
                    out = vjoin(out, st.zone, V(refs=frozenset({fa})), st.zone)
                elif fo.flavor == "classmethod":
                    raise Unsupported("classmethod", node)
                else:
                    ma = self.alloc(st, node, lambda sid: BoundMethod(fa, self_v), "bmeth")
                    out = vjoin(out, st.zone, V(refs=frozenset({ma})), st.zone)
            else:
                out = vjoin(out, st.zone, V(refs=frozenset({fa})), st.zone)
        rest = cv.replace(refs=frozenset())
        if not rest.is_bot():
            out = vjoin(out, st.zone, self.mat(st, rest), st.zone)
        return out

    def attr_store(self, st: State, obj: V, name: str, v: V, node) -> None:
        # ``default_factory`` is a writable C-level slot on defaultdict.  This
        # transfer does not update the factory metadata, so even a valid
        # assignment must stay unknown (including a receiver union).
        if name == "default_factory" and any(
                isinstance(st.heap.get(a), DefaultDictObj) for a in obj.refs):
            raise Unsupported("defaultdict.default_factory assignment", node)
        if name in {"__class__", "__dict__", "__weakref__"}:
            raise Unsupported("reflective attribute assignment " + name, node)
        if obj.top:
            raise Unsupported("attribute store on an unknown value", node)
        if obj.st or any(st.heap[a].kind in ("func",) for a in obj.refs):
            raise Unsupported("attribute store on a module, type or function", node)
        if any(st.heap[a].kind == "class" for a in obj.refs):
            if len(self.alternatives(obj)) > 1:
                self.map_alternatives(st, obj, lambda branch, part:
                    self.attr_store(branch, part, name, v, node) or NONE)
                return
            a = next(iter(obj.refs))
            ob = st.heap[a]
            if name.startswith("__"):
                raise Unsupported("class protocol attribute assignment", node)
            attrs = dict(ob.attrs)
            value = self.summarize(st, v)
            if a[1] != "r":
                if name not in attrs:
                    raise Unsupported("optional class attribute on summary objects", node)
                value = vjoin(attrs[name], st.zone, value, st.zone)
            attrs[name] = value
            st.heap[a] = ob.replace(attrs=attrs)
            return
        bad = bool(obj.n or obj.i is not None or obj.b or obj.f or obj.s is not None or obj.st)
        insts = set()
        for a in obj.refs:
            ob = st.heap[a]
            if ob.kind == "inst":
                # `slots`: every name declared in __slots__ anywhere in ob.cls's
                # MRO, or None if any class in it grants instances a __dict__
                # ([internal ref] review, MG: fixed to actually walk the whole MRO instead
                # of bailing to "has a __dict__" at the instantiated class alone --
                # see the matching comment in attr_load).
                slots = self._all_slots(st, ob.cls)
                # `hit`: `name`'s FIRST class-level occurrence in the MRO -- see
                # attr_load's `_mro_attr_or_slot` comment for the full case
                # analysis this mirrors.
                hit = self._mro_attr_or_slot(st, ob.cls, name)
                if slots is not None and (hit is None or hit[0] == "attr"):
                    # No __dict__ anywhere in the MRO, and `name` is either not a
                    # slot/class-attribute anywhere in it (an undeclared name), or
                    # its FIRST hit is a plain class-level value -- an ordinary
                    # class attribute/method, or a class-level override of what
                    # used to be a slot descriptor there (attr_store's class-kind
                    # branch above, e.g. `Cls.x = 5`). Neither is a data
                    # descriptor (this model does not represent settable custom
                    # descriptors), and there is no __dict__ to fall back to, so
                    # CPython's object.__setattr__ raises AttributeError ("...
                    # object attribute 'name' is read-only", or "... has no
                    # attribute 'name'" if never declared at all).
                    bad = True
                    continue
                insts.add(a)
            elif ob.kind == "class":
                raise Unsupported("class attribute assignment", node)
            elif ob.kind in ("list", "tuple", "array", "dict", "dictview", "range", "iterator", "pattern", "bmeth", "bimeth"):
                bad = True                # these have no ``__dict__`` and no such attribute: an AttributeError
            else:
                # a StringIO or a file handle takes any attribute (it has a ``__dict__``): not modelled, never a raise
                raise Unsupported(f"attribute assignment on a {ob.kind}", node)
        self.check(node, "attr-store", {"AttributeError", "TypeError"}, not bad, f"setting .{name}", st=st)
        if not insts:
            st.set_bottom()
            return

        def upd(a, ob, strong):
            attrs = dict(ob.attrs)
            sv = self.summarize(st, v)
            qf = {g: b for g, b in ob.qf.items() if name not in g}
            if strong:
                old = attrs.get(name)
                ver = 0 if old is None else (None if old[2] is None else old[2] + 1)
                attrs[name] = (sv, True, ver)
                nob = InstObj(ob.cls, attrs, qf)
                return self.recompute_qf(st, a, nob)
            if name not in attrs:
                attrs[name] = (sv, False, None)
            else:
                old, d, _ = attrs[name]
                attrs[name] = (vjoin(old, st.zone, sv, st.zone), d, None)
            return InstObj(ob.cls, attrs, qf)

        strong = len(insts) == 1 and next(iter(insts))[1] == "r"
        for a in insts:
            ob = st.heap[a]
            st.heap[a] = upd(a, ob, strong)

    # ------------------------------------------------ float object invariants
    def recompute_qf(self, st: State, addr, ob):
        """Re-derive the object's sum-of-squares invariant over its float attributes."""
        group = sorted(k for k, (v, d, _r) in ob.attrs.items()
                       if d and v.f and v.i is None and not v.b and v.s is None and not v.n
                       and not v.refs and not v.st and not v.top)
        if not 2 <= len(group) <= 6:
            return ob
        terms = []
        lo = 0.0
        hi = 0.0
        for k in group:
            v, _d, ver = ob.attrs[k]
            f = v.f.fact
            if not f.has_interval():
                return ob
            terms.append(v.f.term if v.f.term is not None else ("rd", addr, k, ver))
            a, b = lemmas.sq_interval(f.lo, f.hi)
            lo += a
            hi = math.inf if math.isinf(b) else math.nextafter(hi + b, math.inf)
        why = "interval squares"
        nb = lemmas.normalize_bounds(terms)
        if nb is not None:
            lo, hi, why = max(lo, nb[0]), min(hi, nb[1]), nb[2]
        else:
            tp = lemmas.trig_pair_lower(terms)
            if tp is not None and tp[0] > lo:
                lo, why = tp
        if lo <= 0.0 and math.isinf(hi):
            return ob
        lemmas.add_landmarks(lo, hi)
        qf = dict(ob.qf)
        qf[frozenset(group)] = (lo, hi, why)
        return InstObj(ob.cls, ob.attrs, qf)

    def sumsq_lower(self, st: State, bases) -> tuple[float, float, str]:
        """Bounds ``(lo, hi, why)`` of the real sum of b*b over value terms ``bases``."""
        best, why = 0.0, ""
        top = math.inf
        # (1) the bases are the current float attributes of ONE object: its invariant
        if bases and all(isinstance(t, tuple) and len(t) == 4 and t[0] == "rd" for t in bases):
            addrs = {t[1] for t in bases}
            names = [t[2] for t in bases]
            if len(addrs) == 1 and len(set(names)) == len(names):
                a = next(iter(addrs))
                ob = st.heap.get(a)
                if ob is not None and ob.kind == "inst" and all(
                        n in ob.attrs and ob.attrs[n][2] == t[3] for n, t in zip(names, bases)):
                    q = ob.qf.get(frozenset(names))
                    if q is not None:
                        lo = q[0] * (1 - 8 * (len(bases) + 1) * lemmas.EPS)
                        top = q[1] * (1 + 8 * (len(bases) + 1) * lemmas.EPS)
                        if lo > best:
                            best, why = lo, (f"object invariant sum-of-squares{sorted(names)} in "
                                             f"[{q[0]:.4g}, {q[1]:.4g}], established by {q[2]}")
        # (2) sin/cos pairs over one argument
        tp = lemmas.trig_pair_lower(list(bases))
        if tp is not None and tp[0] > best:
            best, why = tp
        return best, top, why

    # ------------------------------------------------ receiver focus
    def focus(self, st: State, s_addr, node, args, kwargs):
        """Materialize the one object of summary ``s_addr`` a method runs on."""
        site = s_addr[0] + ("focus",) + self.site(node)
        f_addr = (site, "r")
        ob = st.heap[s_addr]
        st.heap[f_addr] = InstObj(ob.cls, {k: (v, d, 0) for k, (v, d, _r) in ob.attrs.items()}, dict(ob.qf))
        # the object may be reached through any reference to the summary: may-alias
        def split(v: V) -> V:
            return v.replace(refs=v.refs | {f_addr}) if s_addr in v.refs else v

        st._map_values(split)
        args = [split(x) for x in args]
        st.strip_terms(lambda t: t == f_addr)
        return f_addr, args, {k: split(v) for k, v in kwargs.items()}

    def blur(self, st: State, f_addr, s_addr, v: V) -> V:
        st.strip_terms(lambda t: t == f_addr)
        fob = st.heap.pop(f_addr)
        fob = InstObj(fob.cls, {k: (x, d, None) for k, (x, d, _r) in fob.attrs.items()}, fob.qf)
        try:
            st.heap[s_addr] = _ojoin(st.heap[s_addr], st.zone, fob, st.zone)
        except Incompatible as e:
            raise Unsupported(f"blur: {e}")
        st.redirect(f_addr, s_addr)
        if f_addr in v.refs:
            v = v.replace(refs=(v.refs - {f_addr}) | {s_addr})
        if v.f and v.f.term is not None:
            v = v.replace(f=FloatInfo(v.f.fact, None, v.f.why))
        return v

    def refresh(self, st: State, addr) -> None:
        """A method starts on the recent object ``addr``: fresh attribute versions."""
        st.strip_terms(lambda t: t == addr)
        ob = st.heap[addr]
        st.heap[addr] = InstObj(ob.cls, {k: (v, d, 0) for k, (v, d, _r) in ob.attrs.items()}, ob.qf)

    def _all_slots(self, st, cls_addr):
        """The union of every name declared in __slots__ anywhere along the MRO,
        or None if any class in it grants instances a __dict__ -- because IT
        does not declare __slots__ at all (the common case for a subclass that
        never redeclares an empty `__slots__ = ()`), or because "__dict__" is
        itself one of its declared slot names (the explicit opt-in idiom,
        `__slots__ = (..., "__dict__")`).  [internal ref] review (MG): this used to
        return None (bail: "no slots to reason about") the moment the FIRST
        class visited -- own or not -- lacked __slots__, so a subclass that
        inherits real slots from a base but never redeclares __slots__ itself
        (again, the common case) made every check that calls this treat the
        whole hierarchy as slot-free, missing an inherited slot a class
        attribute can still shadow (mg1072 b01/b02/b03/b07 -- confirmed wrong
        TRUE). Now it walks the FULL MRO regardless, collecting slot names from
        whichever classes declare them and only concluding "has a __dict__"
        from the classes that actually say so."""
        names = set()
        has_dict = False
        a = cls_addr
        seen = 0
        while a is not None and seen < 50:
            co = st.heap[a]
            if co.slots is None or "__dict__" in co.slots:
                has_dict = True
            else:
                names.update(co.slots)
            a = co.bases[0] if co.bases else None
            seen += 1
        return None if has_dict else names

    def _mro_attr_or_slot(self, st, cls_addr, name):
        """Walking cls_addr's MRO in order (the class itself, then each base in
        turn), the FIRST class-level thing named `name`: ``("attr", value)`` if
        some class's OWN class-level attrs dict has it (an ordinary class
        attribute or method, or a class-level override of what used to be a
        slot descriptor there -- attr_store's class-kind branch replaces that
        MRO entry outright, e.g. `Cls.x = 5`), ``("slot", class_addr)`` if
        instead the first hit is an intact slot declared in THAT class's own
        `__slots__`, or None if `name` is not a class-level attribute or slot
        anywhere in the MRO (an ordinary, purely per-instance name). Mirrors
        `class_lookup`'s MRO walk but also considers each class's own
        (undisturbed) slot declarations, so it can tell a live slot descriptor
        apart from "nothing here yet" -- `class_lookup` alone cannot, since an
        undisturbed slot is never entered into a class's `attrs` dict at all.
        Raises Unsupported for the same reflective/unmodelled dunder names
        `class_lookup` refuses (`__dict__`, `__weakref__`, an unmodelled
        protocol, ...) -- this replaces `class_lookup` for instance attribute
        resolution, so it must not silently let one through as if it just
        weren't a class attribute anywhere ([internal ref] review follow-up: the first
        draft of this method did, turning `class_lookup`'s abort-the-whole-
        analysis-as-UNKNOWN guard against reflective `cell.__dict__[...] =`
        mutation into a wrong TRUE -- test_reflective_field_mutation_never_
        proves)."""
        if name.startswith("__") and name not in SUPPORTED_DUNDERS | {
                "__init__", "__str__", "__repr__", "__format__", "__bool__"}:
            raise Unsupported("reflective or unmodelled class attribute " + name)
        a = cls_addr
        seen = 0
        while a is not None and seen < 50:
            co = st.heap[a]
            if name in co.attrs:
                return ("attr", co.attrs[name])
            if co.slots is not None and name in co.slots:
                return ("slot", a)
            a = co.bases[0] if co.bases else None
            seen += 1
        return None

    # ============================================================ calls
    def call_keywords(self, e, st, fctx):
        kwargs = {}
        for keyword in e.keywords:
            value = self.eval(keyword.value, st, fctx)
            if st.bottom:
                return kwargs
            if keyword.arg is None:
                if value.top or len(value.refs) != 1 or not value.replace(refs=frozenset()).is_bot():
                    raise Unsupported("keyword expansion of an unknown mapping", e)
                ob = st.heap[next(iter(value.refs))]
                if ob.kind != "dict" or not ob.exact or any(type(k) is not str for k in ob.items):
                    raise Unsupported("keyword expansion requires exact string keys", e)
                entries = {k: self.borrow(st, self.mat(st, v)) for k, v in ob.items.items()}
            else:
                entries = {keyword.arg: value}
            duplicate = bool(kwargs.keys() & entries.keys())
            self.check(e, "call-keywords", {"TypeError"}, not duplicate, "keyword names are unique")
            if duplicate:
                st.set_bottom()
                return kwargs
            kwargs.update(entries)
        return kwargs

    def call_positional_args(self, e, st, fctx):
        """Evaluate positional arguments in order, expanding only fixed builtin sequences.

        A list's ``elem`` is an over-approximation of every slot, not a positional layout. Repeating it
        for a statically fixed length is sound but may lose slot correlations. Tuples with recorded
        ``items`` retain their per-slot values. Iterators, custom iterables, and unknown lengths are
        deliberately refused rather than invoking target ``__iter__`` methods or guessing an arity.
        """
        args = []
        expanded = 0
        for arg in e.args:
            if not isinstance(arg, ast.Starred):
                value = self.eval(arg, st, fctx)
                if st.bottom:
                    return args
                args.append(value)
                continue

            value = self.eval(arg.value, st, fctx)
            if st.bottom:
                return args
            if value.top or len(value.refs) != 1 or not value.replace(refs=frozenset()).is_bot():
                raise Unsupported("*args requires one exact builtin list or tuple", arg)
            self.borrow(st, value)
            ob = st.heap[next(iter(value.refs))]
            if type(ob) not in (ListObj, TupleObj):
                raise Unsupported("*args requires a builtin list or tuple", arg)
            if ob.kind == "tuple" and ob.items is not None:
                if len(ob.items) > MAX_STAR_ARGS:
                    raise Unsupported("*args expansion exceeds the fixed argument limit", arg)
                items = ob.items
            else:
                _lin, (lo, hi) = self.len_lin(st, ob)
                if lo is None or hi is None or lo != hi or lo < 0:
                    raise Unsupported("*args requires a fixed sequence length", arg)
                if lo > MAX_STAR_ARGS:
                    raise Unsupported("*args expansion exceeds the fixed argument limit", arg)
                if lo == 0:
                    items = []
                else:
                    elem = self.all_elems(st, ob)
                    if elem.is_bot():
                        raise Unsupported("*args sequence has no element summary", arg)
                    items = [elem] * lo
            if expanded + len(items) > MAX_STAR_ARGS:
                raise Unsupported("*args expansion exceeds the fixed argument limit", arg)
            expanded += len(items)
            args.extend(self.borrow(st, self.mat(st, item)) for item in items)
        return args

    def e_Call(self, e, st, fctx):
        has_starred = any(isinstance(a, ast.Starred) for a in e.args)
        if has_starred and isinstance(e.func, ast.Attribute):
            # This fast path resolves the method after evaluating arguments. A starred expression can
            # mutate the receiver, so keep the native pre-argument attribute lookup boundary explicit.
            raise Unsupported("method call with *args (attribute lookup order)", e)
        # method call fast path: obj.name(...)
        if isinstance(e.func, ast.Attribute):
            recv = self.eval(e.func.value, st, fctx)
            if st.bottom:
                return BOT
            args = (self.call_positional_args(e, st, fctx) if has_starred else
                    [self.eval(a, st, fctx) for a in e.args])
            kwargs = self.call_keywords(e, st, fctx)
            if st.bottom:
                return BOT
            return self.call_method(st, recv, e.func.attr, args, kwargs, e, operands=True)
        fv = self.eval(e.func, st, fctx)
        args = (self.call_positional_args(e, st, fctx) if has_starred else
                [self.eval(a, st, fctx) for a in e.args])
        kwargs = self.call_keywords(e, st, fctx)
        if st.bottom:
            return BOT
        return self.call(st, fv, args, kwargs, e, operands=True)

    def call_method(self, st: State, recv: V, name, args, kwargs, node, *, special=False, operands=False) -> V:
        if recv.top:
            raise Unsupported(f"method call on an unknown value: .{name}", node)
        outs = []
        bad = False
        groups: dict = {}
        others = []
        for a in recv.refs:
            ob = st.heap[a]
            if ob.kind == "inst" and (special or name not in ob.attrs):
                cv = self.class_lookup(st, ob.cls, name)
                fa = self._plain_method(st, cv) if cv is not None else None
                if fa is not None:
                    groups.setdefault(fa, set()).add(a)
                    continue
                if cv is None:
                    if _reflective_unresolved_name(name) or self._mro_has_unmodelled_base(st, ob.cls):
                        raise Unsupported(f"reflective call of the implicit/unresolved {name}", node)
                    bad = True
                    continue
            others.append(a)
        calls = [(fa, a) for fa, recvs in sorted(groups.items(), key=lambda kv: repr(kv[0]))
                 for a in sorted(recvs, key=repr)]
        # the sets of one kind are ONE alternative: a call on a recent object and the summary of its site (a set
        # made in a loop) is one call over both, so a result it allocates is allocated once, and a mutator
        # updates both weakly (``update_obj``)
        set_groups: dict = {}
        for a in [a for a in others if st.heap[a].kind == "set"]:
            ob = st.heap[a]
            if name in (FROZENSET_METHODS if ob.frozen else SET_METHODS):
                set_groups.setdefault(ob.frozen, []).append(a)
            elif _reflective_unresolved_name(name):
                raise Unsupported(f"reflective call of {name} on a set", node)
            else:
                bad = True            # e.g. add on a frozenset, or a name no set has: an AttributeError
        others = [a for a in others if st.heap[a].kind != "set"]
        nalts = len(calls) + len(others) + len(set_groups) + len(recv.st)
        for fa, a in calls:
            # one call per receiver object: strong updates on a recent object, and on a
            # focused copy of one object of a summary (blurred back afterwards)
            s2 = st.copy() if nalts > 1 else st
            from .contracts import apply as apply_contract
            # A stationary summary does not retain non-property exceptional exits.
            summary = None if self._try_stack else apply_contract(self, s2, a, fa, args, kwargs, node)
            if summary is not None:
                outs.append((s2, summary))
                continue
            if a[1] == "s":
                f_addr, cargs, ckwargs = self.focus(s2, a, node, list(args), kwargs)
                r = self.call_func(s2, fa, [V(refs=frozenset({f_addr}))] + cargs, ckwargs, node,
                                   operands=operands)
                if not s2.bottom:
                    r = self.blur(s2, f_addr, a, r)
            else:
                self.refresh(s2, a)
                r = self.call_func(s2, fa, [V(refs=frozenset({a}))] + list(args), kwargs, node,
                                   operands=operands)
            outs.append((s2, r))
        for a in others:
            ob = st.heap[a]
            if ob.kind not in ("inst", "class") and not (
                    self._try_stack and self._try_routes_builtin_raise(st, ob, name, args, kwargs)):
                self._in_try_refuse(f"{ob.kind}.{name} (builtin method raises are not routed)", node)
            s2 = st.copy() if nalts > 1 else st
            if ob.kind == "list":
                outs.append((s2, self.list_method(s2, frozenset({a}), name, args, kwargs, node)))
            elif ob.kind == "array":
                outs.append((s2, self.array_method(s2, frozenset({a}), name, args, kwargs, node)))
            elif ob.kind == "dict":
                outs.append((s2, self.dict_method(s2, a, name, args, kwargs, node)))
            elif ob.kind == "stringio":
                # [internal ref]: io.StringIO methods (write/getvalue/read/seek/tell).
                outs.append((s2, self.stringio_method(s2, a, name, args, kwargs, node)))
            elif ob.kind == "tuple":
                outs.append((s2, self.tuple_method(s2, a, name, args, kwargs, node)))
            elif ob.kind == "outfile":
                # [internal ref]: the write side of a file a write-only open() returned
                outs.append((s2, self.outfile_method(s2, a, name, args, kwargs, node)))
            elif ob.kind in ("inst", "class"):
                f = self.attr_load(s2, V(refs=frozenset({a})), name, node)
                if s2.bottom:
                    continue
                outs.append((s2, self.call(s2, f, args, kwargs, node)))
            else:
                raise Unsupported(f"method {name} of a {ob.kind}", node)
        for frozen, group in sorted(set_groups.items()):
            self._in_try_refuse(f"set.{name} (builtin method raises are not routed)", node)
            s2 = st.copy() if nalts > 1 else st
            outs.append((s2, self.set_method(s2, frozenset(group), name, args, kwargs, node)))
        for t in recv.st:
            s2 = st.copy() if nalts > 1 else st
            f = self.static_attr(s2, t, name, node)
            outs.append((s2, self.call(s2, f, args, kwargs, node)))
        if recv.n:
            bad = True
        if recv.s is not None and recv.i is None and not recv.b and not recv.f:
            outs.append((st, self.visit_rule("str." + name, st, [recv] + list(args), kwargs, node)))
        elif recv.i is not None or recv.b or recv.f or recv.s is not None:
            raise Unsupported(f"method {name} of a builtin scalar", node)
        obj_node = node.func.value if isinstance(getattr(node, "func", None), ast.Attribute) else None
        self.check(node, "attr", {"AttributeError"}, not bad, f".{name}() on " + ",".join(recv.kinds()),
                   residual=self._attr_residual(obj_node, name, recv), operand=obj_node, st=st)
        if len(outs) == 1 and outs[0][0] is st:
            return outs[0][1]
        if not outs:
            st.set_bottom()
            return BOT
        return self._merge_branches(st, outs)

    def call(self, st: State, fv: V, args, kwargs, node, operands=False) -> V:
        if fv.top:
            # name the callee: the refusal is what the engine reports for every site behind it
            callee = ast.unparse(node.func) if isinstance(node, ast.Call) else ""
            raise Unsupported("call of an unknown value" + (f": {callee}" if callee else ""), node)
        bad = bool(fv.i is not None or fv.b or fv.f or fv.s is not None or fv.n)
        alts = [("st", t) for t in fv.st] + [("ref", a) for a in fv.refs]
        self.check(node, "call", {"TypeError"}, not bad, "callee: " + ",".join(fv.kinds()), st=st)
        if not alts:
            st.set_bottom()
            return BOT
        if len(alts) == 1:
            return self._call1(st, alts[0], args, kwargs, node, operands)
        outs = []
        for alt in alts:
            s2 = st.copy()
            v = self._call1(s2, alt, args, kwargs, node, operands)
            outs.append((s2, v))
        return self._merge_branches(st, outs)

    def _call1(self, st: State, alt, args, kwargs, node, operands=False) -> V:
        kind, x = alt
        if kind == "st":
            return self.call_static(st, x, args, kwargs, node)
        ob = st.heap[x]
        if ob.kind == "func":
            return self.call_func(st, x, args, kwargs, node, operands=operands)
        if ob.kind == "bmeth":
            return self.call_func(st, ob.func, [ob.self_v] + list(args), kwargs, node, operands=operands)
        if ob.kind == "bimeth":
            recv = ob.recv
            if not (self._try_stack and recv.refs and all(
                    self._try_routes_builtin_raise(st, st.heap[a], ob.name, args, kwargs) for a in recv.refs)):
                self._in_try_refuse(f"bound builtin .{ob.name} (its raises are not routed)", node)
            kinds = {st.heap[r].kind for r in recv.refs}
            if kinds == {"set"}:
                return self.set_method(st, recv.refs, ob.name, args, kwargs, node)
            if kinds == {"array"}:
                # [internal ref]: the typecode check of an element write is array_method's (``_array_store_value``); a bound
                # ``a.append`` through list_method skipped it (``m = a.append; m(0.1)`` on an 'f' array kept the exact
                # 0.1, which CPython does not store)
                return self.array_method(st, recv.refs, ob.name, args, kwargs, node)
            if kinds == {"list"}:
                return self.list_method(st, recv.refs, ob.name, args, kwargs, node)
            raise Unsupported(f"a bound method of {sorted(kinds)}", node)
        if ob.kind == "class":
            return self.instantiate(st, x, args, kwargs, node, operands)
        if ob.kind == "inst" and self.class_lookup(st, ob.cls, "__call__") is not None:
            return self.special(st, V(refs=frozenset({x})), "__call__", args, node, kwargs)
        self.check(node, "call", {"TypeError"}, False, f"calling a {ob.kind}", st=st)
        st.set_bottom()
        return BOT

    def bind_args(self, st: State, fnode, fo: FuncObj, args, kwargs, node):
        for value in [*args, *kwargs.values(), *fo.defaults, *fo.kwdefaults.values()]:
            self.borrow(st, value)
        a = fnode.args
        params = [p.arg for p in a.posonlyargs + a.args]
        nposonly = len(a.posonlyargs)
        bound: dict[str, V] = {}
        extra = []
        extra_keywords = {}
        keyword_only = {p.arg for p in a.kwonlyargs}
        for i, v in enumerate(args):
            if i < len(params):
                bound[params[i]] = v
            else:
                extra.append(v)
        if extra and a.vararg is None:
            return None, f"{len(args)} positional arguments for {len(params)} parameters"
        for k, v in kwargs.items():
            if k in keyword_only or k in params[nposonly:]:
                if k in bound:
                    return None, f"multiple values for {k}"
                bound[k] = v
            elif a.kwarg is not None:
                extra_keywords[k] = self.summarize(st, v)
            else:
                return None, f"unexpected keyword argument {k}"
        nd = len(fo.defaults)
        for i, p in enumerate(params):
            if p not in bound:
                j = i - (len(params) - nd)
                if j >= 0:
                    bound[p] = fo.defaults[j]
                else:
                    return None, f"missing argument {p}"
        for name in keyword_only:
            if name not in bound:
                if name not in fo.kwdefaults:
                    return None, f"missing keyword-only argument {name}"
                bound[name] = fo.kwdefaults[name]
        if a.kwarg is not None:
            ka = self.alloc(st, node, lambda sid: DictObj(extra_keywords), "**kwargs")
            bound[a.kwarg.arg] = V(refs=frozenset({ka}))
        if a.vararg is not None:
            ta = self.new_list(st, node, None, BOT, tag="*args", kind="tuple", items=extra)
            bound[a.vararg.arg] = V(refs=frozenset({ta}))
        return bound, "arguments match the signature"

    def call_func(self, st: State, faddr, args, kwargs, node, *, operands=False, keep=None) -> V:
        """Run the user function ``faddr``.  ``operands`` is true for a call ``e_Call`` evaluated: the holders in
        the current scope are then that call's operands (and the ones ``bind_args`` adds for them), which are in
        the callee's frame once it is bound.  The frame is rewritten when an allocation demotes an object
        and no Python local reads an operand after the call, so for the callee's duration they are not held and
        it may allocate at their sites (``d[i] = d[i].combine(d[i + 1])`` builds a new object of the receiver's
        class at one site).  ``keep`` is the one value the caller reads afterwards (the instance
        ``instantiate`` returns); the holders come back when the callee returns."""
        fo = st.heap[faddr]
        fnode = fo.node
        if fnode in self.active:
            raise Unsupported("recursion", node)
        if len(self.active) >= MAX_DEPTH:
            raise Unsupported("call depth", node)
        if _is_generator(fnode):
            raise Unsupported("generator function", node)
        bound, why = self.bind_args(st, fnode, fo, args, kwargs, node)
        self.check(node, "call-args", {"TypeError"}, bound is not None, why, st=st)
        if bound is None:
            st.set_bottom()
            return BOT
        if isinstance(fnode, ast.Lambda):
            if any(isinstance(n, ast.NamedExpr) for n in ast.walk(fnode.body)):
                raise Unsupported("assignment expression in lambda", node)
            loc, glo = set(), set()
        else:
            loc, glo = _assigned_names(fnode.body)
        loc |= set(bound)
        # free variables of a nested function would be closures: not modelled
        # [internal ref]: a function's globals are fixed at DEFINITION time (its OWN file's module
        # namespace), never the caller's -- FuncObj.module_key carries that forward.
        fctx = FuncCtx(fnode, loc, glo, module_key=fo.module_key)
        st.frames.append(Frame(f"c{len(st.frames)}"))
        for k, v in bound.items():
            self.write_var(st, len(st.frames) - 1, k, v)
        held = self.borrowed[-1]
        saved_holders = set(held) if operands else None
        if operands:
            held.clear()
            if keep is not None:
                held.update(addr[0] for addr in keep.refs if addr[1] == "r")
        self.callstring.append((node.lineno, node.col_offset,
                                getattr(node, "end_lineno", None), getattr(node, "end_col_offset", None)))
        self.active.append(fnode)
        saved_forced, self._forced_draws = self._forced_draws, {}   # a pending case-split force is not the callee's
        saved_epoch = self.epoch
        saved_module_key = self._cur_module_key
        self._cur_module_key = fo.module_key   # [internal ref]: allocation sites stay file-discriminated
        self.borrowed.append(set())
        self._fctx_stack.append(fctx)
        try:
            self.epoch = set()
            if isinstance(fnode, ast.Lambda):
                value = self.eval(fnode.body, st, fctx)
                if not st.bottom:
                    self._route_return(st.copy(), value, fctx)
                st.set_bottom()
                end = st
            else:
                end = self.exec_block(fnode.body, st, fctx, None)
        finally:
            self._fctx_stack.pop()
            self.callstring.pop()
            self.active.pop()
            self._forced_draws = saved_forced
            self.epoch = saved_epoch
            self._cur_module_key = saved_module_key
            self.borrowed.pop()
            if saved_holders is not None:
                held.update(saved_holders)
        # [internal ref]: an exception that left the callee is raised in the caller, at the call, unwound by one frame.
        # The ONE exception: a TypeError leaving ``__len__`` into the function a length probe runs in is
        # swallowed by CPython's PyObject_LengthHint; the probe continues from that state (_probe_length).
        caller_fctx = self._fctx_stack[-1] if self._fctx_stack else None
        if fctx.raises and caller_fctx is None:
            raise Unsupported("exception from context-free function analysis", node)
        for exc in fctx.raises:
            if exc.st.bottom:
                continue
            self.pop_frame(exc.st)
            # PyObject_LengthHint catches TypeError by identity, including user subclasses.
            if (self._probe_scopes and self._is_subclass(exc.st, exc.cls, "TypeError")
                    and self._probe_scopes[-1].fctx is caller_fctx):
                self._probe_scopes[-1].raises.append(exc.st)
                continue
            self._route_exc(exc, caller_fctx)
        outs = list(fctx.returns)
        if not end.bottom:
            outs.append((end, NONE))
        outs = [(s, v) for s, v in outs if not s.bottom]
        if not outs:
            st.set_bottom()
            return BOT
        tname = f"${len(st.frames) - 1}_{self._bump()}"  # a caller-depth temporary
        res_s, res_v = None, BOT
        for s, v in outs:
            if v.i is not None:
                self.zone_assign(s, tname, v.i)
                v = v.replace(i=IntInfo(Lin.var(tname)))
            v = self._escape_frame_bounds(s, v)
            if res_s is None:
                res_s, res_v = s, v
            else:
                res_v = vjoin(res_v, res_s.zone, v, s.zone)
                res_s = res_s.join(s)
        self.pop_frame(res_s)
        _assign_state(st, res_s)
        return res_v

    def _bump(self):
        self.tmp += 1
        return self.tmp

    def _escape_frame_bounds(self, st: State, v: V) -> V:
        return v

    def pop_frame(self, st: State) -> None:
        fid = st.frames[-1].fid
        pre = fid + ":"
        for zv in [x for x in st.zone.vars if x.startswith(pre)]:
            st.kill_var(zv)
        # temporaries of the popped depth survive as caller temporaries (lower depth
        # names are never reused): rename is unnecessary because names are unique
        st.frames.pop()

    def instantiate(self, st: State, caddr, args, kwargs, node, operands=False) -> V:
        co = st.heap[caddr]
        # reject unmodelled class machinery along the MRO
        a = caddr
        exc_base = None
        while a is not None:
            c = st.heap[a]
            for k in c.attrs:
                if k in UNSUPPORTED_DUNDERS:
                    raise Unsupported(f"class defines {k}", node)
            if "__exc_base__" in c.attrs:
                exc_base = next(iter(c.attrs["__exc_base__"].st))[1]
            a = c.bases[0] if c.bases else None
        if exc_base is not None and not self._plain_exception_constructor(exc_base):
            self._in_try_refuse(f"{exc_base} constructor inherited by {co.name}", node)
        inst = self.alloc(st, node, lambda sid: InstObj(caddr, {}), "inst:" + co.name)
        iv = V(refs=frozenset({inst}))
        init = self.class_lookup(st, caddr, "__init__")
        if init is None:
            if exc_base is None:
                self.check(node, "call-args", {"TypeError"}, not args and not kwargs, "object() takes no arguments", st=st)
                if args or kwargs:
                    st.set_bottom()
                    return BOT
            elif kwargs:
                self.check(node, "call-args", {"TypeError"}, False, "builtin exception constructor takes no keywords", st=st)
                st.set_bottom()
                return BOT
            return iv
        fa = self._plain_method(st, init)
        if fa is None:
            raise Unsupported("__init__ must be a plain instance method", node)
        r = self.call_func(st, fa, [iv] + list(args), kwargs, node, operands=operands, keep=iv)
        if st.bottom:
            return BOT
        self.check(node, "init-return", {"TypeError"}, r.kinds() == ["None"], "__init__ returns None", st=st)
        return iv

    # ============================================================ exceptions
    def exc_classes_of(self, st: State, v: V, node) -> set:
        """The property exceptions an exception value (class or instance) may be."""
        out = set()
        if v.top:
            raise Unsupported("raise of an unknown value", node)
        for t in v.st:
            if t[0] in ("exc", "excinst"):
                out |= _EXC_ANCESTORS.get(t[1], set())
            else:
                out.add("TypeError")  # raising a non-exception
        for a in v.refs:
            ob = st.heap[a]
            cls = ob.cls if ob.kind == "inst" else a if ob.kind == "class" else None
            if cls is None:
                out.add("TypeError")
                continue
            base = None
            c = cls
            while c is not None:
                co = st.heap[c]
                if "__exc_base__" in co.attrs:
                    base = next(iter(co.attrs["__exc_base__"].st))[1]
                c = co.bases[0] if co.bases else None
            if base is None:
                out.add("TypeError")  # exceptions must derive from BaseException
            else:
                out |= _EXC_ANCESTORS.get(base, set())
        if v.i is not None or v.b or v.f or v.s is not None or v.n:
            out.add("TypeError")
        return out

    # ============================================================ builtins
    def builtin_name(self, name, node) -> V:
        if name in {"format", "repr", "ascii"}:
            return V(st=frozenset({("cbuiltin", name)}))
        if name in BUILTIN_FUNCS:
            return V(st=frozenset({("builtin", name)}))
        if name in BUILTIN_TYPES:
            return V(st=frozenset({("type", name)}))
        if name in _EXC_ANCESTORS:
            return V(st=frozenset({("exc", name)}))
        if name in ("True", "False", "None"):
            raise Unsupported("constant name", node)
        if name in _BUILTIN_NAMES:
            # [internal ref] batch 4: a builtin with no b_/t_ transfer function but a contract-driven
            # model (c_<name>, run through _contract_builtin_call) and a TESTED contract
            # (solve/contracts/adapters/absint.py::contract_for_builtin).
            if hasattr(self, "c_" + name):
                from tharkun.solve.contracts.adapters import absint as _contracts

                if _contracts.contract_for_builtin(name) is not None:
                    return V(st=frozenset({("cbuiltin", name)}))
            # any other unmodelled builtin is TOP: any use of it aborts (never a vacuous TypeError)
            return V(top=True)
        # NameError: not a property exception; the path ends
        return BOT

    def static_attr(self, st: State, tag, name, node) -> V:
        if tag[0] == "native-flag":
            raise Unsupported("re.RegexFlag attribute is not modelled",node)
        if tag[0] in {"abstract-pattern", "abstract-match", "native-regex"}:
            import re
            from tharkun.solve.contracts.rules import get_rule
            owner = "re.Pattern" if tag[0] == "abstract-pattern" or (tag[0] == "native-regex" and type(tag[1]) is re.Pattern) else "re.Match"
            qualname = owner + "." + name
            # [internal ref]: bare reads need admission too; Match.groups is a method, Pattern.groups data.
            if get_rule(qualname) is not None:
                return V(st=frozenset({("native-method", qualname, tag)}))
            if tag[0] == "native-regex" and name in {"pattern", "flags", "groups", "groupindex", "string", "pos", "endpos", "lastindex", "lastgroup"} and hasattr(tag[1], name):
                from .native import abstract
                result = abstract(self, st, [getattr(tag[1], name)], node)
                if result is not None:
                    return result
                raise Unsupported(f"{owner}.{name}: native attribute not represented", node)
            raise Unsupported(f"{qualname}: attribute has no rule or model", node)
        if tag == ("type", "str"):
            from tharkun.solve.contracts.rules import get_rule
            if get_rule("str." + name) is not None:
                return V(st=frozenset({("cbuiltin", "str." + name)}))
        if tag[0] == "module":
            v = self.module_attr(st, tag[1], name, node)
            if v.top:
                # a name no table models: its value is unknown, and so is whether it exists (``math.foo`` is an AttributeError)
                self.check(node, "module-attr", {"AttributeError"}, False, f"{tag[1]}.{name} is not modelled and may not exist", st=st)
            return v
        raise Unsupported(f"attribute {name} of {tag}", node)

    def module_attr(self, st: State, mod: str, name: str, node, fctx=None) -> V:
        if mod == "builtins" and name in {"format", "repr", "ascii", "str"}:
            return self.builtin_name(name, node)
        if mod == "re":
            import re
            flag=getattr(re,name,None)
            if type(flag) is re.RegexFlag:
                return const_int(int(flag)).replace(st=frozenset({("native-flag",flag)}))
        m = MODULES.get(mod)
        if m is not None:
            if name in m.get("consts", {}):
                if mod == "math":
                    # the constants of ``math`` are read through their rules ([internal ref] step 2): the value is CPython's own
                    return self.visit_rule(f"math.{name}", st, [], {}, node)
                return m["consts"][name]
            if name in m.get("funcs", ()):
                return V(st=frozenset({("mfunc", mod, name)}))
            # [internal ref] (the "os bug"): a side-effecting/unmodellable name (``os.system``,
            # ``os.remove``, ``os.environ`` used bare, ...) must abort, not fall through to
            # the generic "known module, unmodelled name -> top" case below -- that case is
            # for a merely UNAUDITED name (safe to let its VALUE be unknown), not one this
            # pass has already looked at and decided it cannot model soundly.
            if name in m.get("refused", ()):
                raise Unsupported(f"{mod}.{name} not modelled (side-effecting)", node)
        # [internal ref] step 2: a TESTED contract lets the analysis continue past an
        # otherwise-unmodelled module/function instead of aborting the whole proof
        # (see solve/contracts/adapters/absint.py) -- only ever an ADDITION to what
        # MODULES already models, and checked regardless of whether `mod` is a
        # known module at all (`m is None`, e.g. "heapq" -- not in MODULES,
        # entirely contract-modelled) or just this one name is unmodelled.
        from tharkun.solve.contracts.adapters import absint as _contracts

        if _contracts.contract_for_module_func(mod, name) is not None:
            return V(st=frozenset({("cfunc", mod, name)}))
        if m is not None:
            return V(top=True)  # a KNOWN module; this name just isn't modelled
        # [internal ref]: read-and-recurse -- `mod` names nothing MODULES or a contract
        # covers, so it may instead be an alias bound to a file this program's own
        # sys.path.insert(0, D) makes reachable (extern.py's conditions (a)-(e)).
        return self._extern_attr(st, name, node, fctx)

    # ---------------------------------------------------------- [internal ref]: read-and-recurse
    def _extern_resolver(self):
        """The shared ``pyresolve.py::Resolver`` built (once) over THIS program's own tree,
        reused purely for its already-audited file-finding (conditions (a)-(e), extern.py) and
        its OWN recursively-computed ``.externs``/``.imports`` for every file it read in
        transitively -- ``self._ext_by_key`` indexes ALL of them (this program plus every
        recursively-read file) by absolute path, so any execution context (module_key) can
        find the resolution that governs ITS OWN file. None on ANY failure to build it --
        read-and-recurse then simply never applies, the unconditionally-safe fallback
        (``V(top=True)``, exactly today's behaviour)."""
        if self._ext_resolver == "$unset$":
            try:
                from tharkun.solve.pyresolve import Resolver
                from tharkun.solve.determinism import DET_MODULES
                from tharkun.solve.contracts.rules import native_module_names
                # Absint may consume calls covered by these real native modules.
                # Keep the general determinism allowlist unchanged: this Resolver
                # only establishes import/name metadata for the absint transfer.
                r = Resolver(self.tree, DET_MODULES | native_module_names(), filename=self.filename,
                             _declared_files=self.declared_files)
                self._ext_resolver = r
                self._ext_by_key = {x.filename: x for x in r.all_resolvers()}
                self._ext_by_key[""] = r
            except Exception:
                self._ext_resolver = None
                self._ext_by_key = {}
        return self._ext_resolver

    def _governing_resolver(self, fctx):
        """The already-computed Resolver for the file ``fctx`` is currently executing in --
        the top-level program's own for ``module_key == ""``, or a recursively-read file's
        own (read-and-recurse already fully resolved its OWN imports when it was built)."""
        self._extern_resolver()
        key = fctx.module_key if fctx is not None else ""
        return self._ext_by_key.get(key)

    def _extern_attr(self, st: State, name: str, node, fctx) -> V:
        gov = self._governing_resolver(fctx)
        if gov is None:
            return V(top=True)
        return self._bind_value(st, gov, name, node)

    def _bind_value(self, st: State, gov, name: str, node=None) -> V:
        """The value ``name`` (as ``gov``'s own file binds it, via import) resolves to, or
        ``top`` -- looks it up in ``gov.imports.names``/``gov.externs`` (already fully
        resolved when ``gov`` was built) and, for an extern one, ensures its file's own top
        level has actually run (memoized) before reading its binding back."""
        entry = gov.imports.names.get(name)
        sub = gov.externs.get(name)
        if entry is None or sub is None or not entry[0].startswith("$extern:"):
            return V(top=True)
        if not self._ensure_extern_loaded(st, sub):
            return V(top=True)
        key = self._global_key(entry[1], sub.filename)
        if key in st.frames[0].maybe_unbound:
            raise Unsupported(f"imported name {name!r} may be unbound", node)
        return st.frames[0].env.get(key, V(top=True))

    def _exec_import_from(self, s: ast.ImportFrom, st: State, fctx, gov) -> None:
        """A relative ``from . import ...`` / ``from .mod import ...`` reached while executing
        a recursively-read file's own top level -- (e): resolves against the PACKAGE's own
        location, which is exactly what ``gov`` (built over THAT file) already did."""
        if not s.module:
            raise Unsupported("relative import", s)
        for a in s.names:
            if a.name == "*":
                from tharkun.solve import extern as _extern
                pkg_dir = _extern.package_dir_for(s.level, gov.filename)
                try:
                    found = _extern.find_in_roots(s.module, [pkg_dir]) if pkg_dir else None
                except _extern.ExternTop:
                    found = None
                if found is None:
                    raise Unsupported("star import does not resolve", s)
                target = found[0]
                for alias, (tag, _attr) in gov.imports.names.items():
                    if tag == f"$extern:{target}":
                        self.store_name(alias, self._bind_value(st, gov, alias, s), st, fctx)
                continue
            alias = a.asname or a.name
            self.store_name(alias, self._bind_value(st, gov, alias, s), st, fctx)

    def _ensure_extern_loaded(self, st: State, sub) -> bool:
        fname = sub.filename
        if fname in self._ext_ok:
            return True
        if fname in self._ext_failed or fname in self._ext_loading:
            return False
        self._ext_loading.add(fname)
        try:
            # [internal ref]/MG's PR 86 review: _bind_value only calls _ensure_extern_loaded for
            # the file a NAME is actually read from -- a package's own __init__.py, which
            # nothing may ever bind a name FROM, would otherwise never execute here even
            # though real Python always runs it as part of the import. pkg_inits (set by
            # pyresolve.py::Resolver._resolve_dotted / _check_inits) is this file's own
            # non-inert package-init chain, outermost to innermost -- run each BEFORE this
            # file's own top level, exactly real import order, each independently
            # memoized/trial-committed by this SAME recursive call.
            inits_ok = all(self._ensure_extern_loaded(st, init_sub)
                           for init_sub in sub.pkg_inits)
            ok = inits_ok and self._exec_extern_toplevel(st, sub)
        finally:
            self._ext_loading.discard(fname)
        (self._ext_ok if ok else self._ext_failed).add(fname)
        return ok

    def _exec_extern_toplevel(self, st: State, sub) -> bool:
        """Run ``sub``'s (a recursively-read file's) own module top level for real, into the
        SAME shared state, under its own module_key namespace -- (d): imported code is
        program code. Any construct absint cannot model there degrades to "not loaded"
        (``_extern_attr`` then falls back to ``V(top=True)``, sound) rather than aborting the
        whole analysis. Runs on a COPY of ``st`` first and only commits on a clean, non-bottom
        completion, so a partial failure never leaves a half-executed module's bindings
        behind."""
        for n in ast.walk(sub.tree):
            if isinstance(n, ast.Global):
                self.global_written.update(self._global_key(nm, sub.filename) for nm in n.names)
        loc, glo = _assigned_names(sub.tree.body)
        ext_ctx = FuncCtx(sub.tree, loc, glo, is_module=True, module_key=sub.filename)
        trial = st.copy()
        saved_module_key = self._cur_module_key
        self._cur_module_key = sub.filename
        # [internal ref]: a refused module must not leave its try frames or handled exceptions on the stacks
        depths = (len(self._try_stack), len(self._active_exc_stack), len(self._fctx_stack))
        self._fctx_stack.append(ext_ctx)
        try:
            end = self.exec_block(sub.tree.body, trial, ext_ctx, None)
        except Unsupported:
            return False
        finally:
            self._cur_module_key = saved_module_key
            del self._try_stack[depths[0]:]
            del self._active_exc_stack[depths[1]:]
            del self._fctx_stack[depths[2]:]
        if end.bottom:
            return False
        _assign_state(st, end)
        return True

    def _primitive_valueerror_call(self, st: State, name, args, kwargs, node) -> bool:
        """Admit narrow primitive call shapes whose exception behavior has no user coercion hook.

        ``round`` is admitted only for a direct one-positional-argument call over an exact
        built-in scalar; its NaN/infinity ValueError/OverflowError paths are handled by the
        model and refused inside a try unless the abstract float is definitely finite.
        """
        if name == "round":
            return (not kwargs and len(args) == 1 and isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name) and node.func.id == "round"
                    and len(node.args) == 1 and not node.keywords and not isinstance(node.args[0], ast.Starred)
                    and (args[0].only_int() or args[0].kinds() in (["bool"], ["float"])))
        if name == "divmod":
            return (not kwargs and isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "divmod" and len(node.args) == 2 and not node.keywords
                    and len(args) == 2 and all(v.only_int() for v in args))
        if name in ("str.index", "str.rindex"):
            if kwargs or len(args) != 2:
                return False
            receiver, needle = args
            if any(v.s is None or v.s == ANY_STR or not v.s or not v.replace(s=None).is_bot()
                   for v in (receiver, needle)):
                return False
            return all(sub not in text for text in receiver.s for sub in needle.s)
        if name in ("int", "float"):
            if (kwargs or len(args) != 1 or not isinstance(node, ast.Call) or node.keywords
                    or len(node.args) != 1 or not isinstance(node.args[0], ast.Constant)
                    or type(node.args[0].value) is not str):
                return False
            text = node.args[0].value.strip()
            # shortcut: digit-containing literals stay refused; add lexical coverage only for a needed invalid family.
            return (not any(c.isdecimal() for c in text)
                    and text.lower().lstrip("+-") not in {"nan", "inf", "infinity"})
        if kwargs or name not in ("chr", "range") or len(args) != (1 if name == "chr" else 3):
            return False
        if any(v.i is None or not v.replace(i=None).is_bot() for v in args):
            return False
        lo, hi = self.ibounds(st, args[-1].i)
        return ((lo == hi == 0) if name == "range" else
                (hi is not None and hi < 0) or (lo is not None and lo > 0x10FFFF))

    def _nonraising_print_call(self, st: State, name, args, kwargs, node) -> bool:
        """Whether this resolved builtin print call uses callback-free primitives safe for inherited stdout encoding."""
        if name != "print" or not isinstance(node, ast.Call) or len(args) != len(node.args):
            return False
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            return False
        for value in args:
            if value.top or value.refs or value.st:
                return False
            if not (value.i is not None or value.b or value.f or value.s is not None or value.n):
                return False
            if value.i is not None:
                lo, hi = self.ibounds(st, value.i)
                if lo is None or hi is None or max(abs(lo), abs(hi)) >= _PRINT_INT_ABS_LIMIT:
                    return False
            if value.s is not None:
                if value.s == ANY_STR or not value.s:
                    return False
                # Runner stdout inherits PYTHONIOENCODING; non-ASCII can raise under a strict codec.
                if any(not item.isascii() for item in value.s):
                    return False
        keyword_names = set()
        for keyword in node.keywords:
            if (keyword.arg not in ("sep", "end") or not isinstance(keyword.value, ast.Constant)
                    or type(keyword.value.value) is not str):
                return False
            value = kwargs.get(keyword.arg)
            if (value is None or value.top or value.refs or value.st or value.i is not None or value.b or value.f
                    or value.n or value.s is None or value.s == ANY_STR or not value.s):
                return False
            if (not keyword.value.value.isascii()
                    or any(not item.isascii() for item in value.s)):
                return False
            keyword_names.add(keyword.arg)
        return len(kwargs) == len(node.keywords) and set(kwargs) == keyword_names

    def call_static(self, st: State, tag, args, kwargs, node) -> V:
        kind = tag[0]
        if kind not in ("exc", "excinst") and not (
                (tag[:2] in (("cbuiltin", "chr"), ("builtin", "range"), ("type", "int"), ("type", "float"),
                             ("cbuiltin", "divmod"), ("cbuiltin", "round"))
                 and self._primitive_valueerror_call(st, tag[1], args, kwargs, node))
                or (tag[:2] == ("builtin", "print") and self._nonraising_print_call(st, tag[1], args, kwargs, node))):
            self._in_try_refuse(f"{'.'.join(map(str, tag[:3]))} (static callable raises are not routed)", node)
        if kind == "native-method":
            return self.visit_rule(tag[1], st, [V(st=frozenset({tag[2]}))] + list(args), kwargs, node)
        if kind == "type" and tag[1] == "str":
            return self.visit_rule("str", st, args, kwargs, node)
        if kind == "mfunc" and tag[1] == "re":
            return self.visit_rule("re." + tag[2], st, args, kwargs, node)
        if kind == "unmodelled":
            raise Unsupported(f"call of unmodelled {tag[1]}", node)
        if kind == "excinst":
            # an exception INSTANCE (``ValueError('x')``) is not callable
            self.check(node, "call", {"TypeError"}, False, "calling an exception instance", st=st)
            st.set_bottom()
            return BOT
        if kind == "exc":
            if tag[1] in ("ExceptionGroup", "BaseExceptionGroup"):
                raise Unsupported("except* / ExceptionGroup is not modelled", node)
            if not self._plain_exception_constructor(tag[1]):
                self._in_try_refuse(f"{tag[1]} constructor", node)
            self.check(node, "call-args", {"TypeError"}, not kwargs,
                       "builtin exception constructor takes no keywords", st=st)
            if kwargs:
                st.set_bottom()
                return BOT
            for a in args:
                self.check_formattable(st, a, node)
            return V(st=frozenset({("excinst", tag[1])}))
        if kind == "builtin":
            fn = getattr(self, "b_" + tag[1])
            return fn(st, args, kwargs, node)
        if kind == "type":
            fn = getattr(self, "t_" + tag[1], None)
            if fn is None:
                raise Unsupported(f"call of type {tag[1]}", node)
            return fn(st, args, kwargs, node)
        if kind == "mfunc":
            if tag[1] == "math" or f"{tag[1]}.{tag[2]}" in _MODULE_HOOKS:
                return self.visit_rule(f"{tag[1]}.{tag[2]}", st, args, kwargs, node)
            fn = getattr(self, "m_" + tag[1].replace(".", "_") + "_" + tag[2], None)
            if fn is None:
                raise Unsupported(f"call of {tag[1]}.{tag[2]}", node)
            return fn(st, args, kwargs, node)
        if kind == "cfunc":
            return self.visit_rule(f"{tag[1]}.{tag[2]}", st, args, kwargs, node)
        if kind == "cbuiltin":
            from tharkun.solve.contracts.adapters.common import BUILTIN_SHAPE_CONTRACTS

            return self.visit_rule(BUILTIN_SHAPE_CONTRACTS.get(tag[1], tag[1]), st, args, kwargs, node)
        raise Unsupported(f"call of {tag}", node)

    def visit_rule(self, qualname: str, st: State, args, kwargs, node) -> V:
        """Run the call of the contracted callable ``qualname`` through its rule (``contracts/rules.py``): the rule
        accepts this analyzer's visitor (``absint/visitor.py``), whose ``visit_<kind>`` is the transfer function, and
        a rule kind the visitor has no method for is a NAMED refusal (:class:`Refused`) reported as ``Unsupported``."""
        from tharkun.solve.contracts.rules import Refused, get_rule

        safe_print = self._nonraising_print_call(st, qualname, args, kwargs, node)
        if self._try_stack and not (self._primitive_valueerror_call(st, qualname, args, kwargs, node) or safe_print):
            # [internal ref]: a rule's non-property raises (the ValueError of str.index, ...) are not routed to handlers
            self._in_try_refuse(f"{qualname} (a contract rule: its non-property raises are not routed)", node)
        rule = get_rule(qualname)
        if rule is None:
            raise Unsupported(f"{qualname} has no rule", node)
        if self._try_stack and safe_print:
            from tharkun.solve.contracts.adapters import absint as _contracts
            if _contracts.contract_for_builtin(qualname) is None:
                raise Unsupported(f"{qualname} has no tested contract", node)
        if qualname in ("str.index", "str.rindex") and self._primitive_valueerror_call(st, qualname, args, kwargs, node):
            return self._native_string_search_valueerror(qualname, st, node)
        result = rule.accept(self.visitor, CallCtx(st, tuple(args), dict(kwargs), node))
        if isinstance(result, Refused):
            raise Unsupported(str(result), node)
        return result

    def _native_string_search_valueerror(self, qualname: str, st: State, node) -> V:
        """Route definite str.index/rindex absence only after the callable rule is admitted."""
        from tharkun.solve.contracts.adapters import absint as _contracts
        from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES

        contract = _contracts.contract_for_method("str", qualname.split(".", 1)[1])
        if contract is None:
            raise Unsupported(f"{qualname} has no tested contract", node)
        self._contract_checks(node, qualname, contract,
                              {exc: "primitive str operands run no user hook" for exc in ALL_PROPERTY_CLASSES})
        self.check(node, qualname, {"ValueError"}, False, "substring absent from finite strings", st=st)
        st.set_bottom()
        return BOT

    def visit_method_rule(self, qualname: str, ctx) -> V:
        """Run a method call of a builtin heap kind through its rule: the rule accepts this analyzer's visitor with a
        :class:`~tharkun.solve.absint.visitor.MethodCtx`; a rule kind the visitor has no method for is a named refusal."""
        from tharkun.solve.contracts.rules import Refused, get_rule

        rule = get_rule(qualname)
        if rule is None:
            raise Unsupported(f"{qualname} has no rule", ctx.node)
        result = rule.accept(self.visitor, ctx)
        if isinstance(result, Refused):
            raise Unsupported(str(result), ctx.node)
        return result

    def _contract_method(self, qualname: str, ctx, model) -> V:
        """The call of a method of a builtin heap kind, read through its rule: the contract must be TESTED, the model (the
        unchanged hand transfer function, ``model(st, args, kwargs, node)``) runs under ``_modelled``, and the seven classes are
        recorded as ``<qualname>:<class>`` -- as the contract states them, upgraded where the model argued (``_METHOD_HOOKS``)."""
        from tharkun.solve.contracts.adapters import absint as _contracts

        contract = _contracts.contract_for_builtin(qualname)
        if contract is None:
            raise Unsupported(f"{qualname} has no tested contract", ctx.node)
        result, proven = self._modelled(ctx.state, ctx.node, qualname, model, ctx.args, ctx.kwargs, _METHOD_HOOKS[qualname],
                                        native=_METHOD_NATIVE.get(qualname))
        self._contract_checks(ctx.node, qualname, contract, proven)
        return result

    def visit_protocol(self, qualname: str, node, run, native: dict | None = None):
        """Run the dispatch of an object protocol (``protocol.binary_operator`` ...) through its rule: ``run()`` is the unchanged
        transfer function (a zero-argument callable closed over the state and the operands), the rule accepts this analyzer's visitor
        with a :class:`~tharkun.solve.absint.visitor.ProtocolCtx`, and a rule kind the visitor has no method for is a named refusal."""
        from tharkun.solve.contracts.rules import Refused, get_rule

        rule = get_rule(qualname)
        if rule is None:
            raise Unsupported(f"{qualname} has no rule", node)
        result = rule.accept(self.visitor, ProtocolCtx(qualname, node, run, native))
        if isinstance(result, Refused):
            raise Unsupported(str(result), node)
        return result

    def _through_protocol(self, qualname: str, node, run, native: dict | None = None, lean: bool = False):
        """The dispatch of an object protocol, read through its rule: the contract must be TESTED (else a named refusal, not the
        dispatch of old), ``run()`` is the unchanged dispatch under a tap, and the seven classes are recorded as ``<qualname>:<class>``
        -- as the contract states them (a class the dunder's body can raise: proven because that body is analysed, ``_PROTOCOL_BODY``),
        upgraded where the dispatch argued ``native`` and vetoed where it recorded a failing check at this node."""
        from tharkun.solve.contracts.adapters import absint as _contracts

        contract = _contracts.contract_for_builtin(qualname)
        if contract is None:
            raise Unsupported(f"{qualname} has no tested contract", node)
        seen: list = []
        self._taps.append((node, seen))
        try:
            result = run()
        finally:
            self._taps.pop()
        self._contract_checks(node, qualname, contract, self._proven_classes(contract, seen, _PROTOCOL_HOOKS.get(qualname, _PROTOCOL_BODY), native),
                              lean=qualname in _PROTOCOL_LEAN)
        return result

    def _contract_module_model(self, qualname: str, st: State, model, args, kwargs, node) -> V:
        """The call of a function of a module the analyzer models by hand (``random.choice``: ``rand.py``), read through its rule: the
        contract must be TESTED (else a named refusal, not the model of old), the unchanged model (``model(st, args, kwargs, node)``)
        runs under ``_modelled``, and the seven classes are recorded as ``<qualname>:<class>`` -- as the contract states them,
        upgraded where the model argued (``_MODULE_HOOKS``)."""
        from tharkun.solve.contracts.adapters import absint as _contracts

        contract = _contracts.contract_for_builtin(qualname)
        if contract is None:
            raise Unsupported(f"{qualname} has no tested contract", node)
        result, proven = self._modelled(st, node, qualname, model, args, kwargs, _MODULE_HOOKS[qualname])
        self._contract_checks(node, qualname, contract, proven)
        return result

    def _contract_builtin_call(self, st: State, name: str, args, kwargs, node) -> V:
        """[internal ref] batch 4: a call ``builtin_name`` tagged ``"cbuiltin"`` -- a builtin whose
        contract (``solve/contracts/rules_builtins_more.py``) is TESTED and whose result shape
        this analysis models (``c_<name>``).

        The contract is the authority on WHICH property classes the call can raise: every
        one of the seven starts as the contract states it (``adapters/absint.py::raise_checks``:
        proven only for a TESTED ``never``, otherwise unproven).  The model then returns the
        abstract result AND the classes it has positively argued cannot occur for THESE
        abstract arguments (``proven``: class -> why); only those are upgraded to proven.
        A class the model does not mention stays exactly as unproven as the contract left it,
        so a model that forgets a class costs precision, never soundness.  A model refuses
        (``Unsupported``) rather than guess about an argument it does not understand."""
        from tharkun.solve.contracts.adapters import absint as _contracts

        if not (self._primitive_valueerror_call(st, name, args, kwargs, node)
                or self._nonraising_print_call(st, name, args, kwargs, node)):
            self._in_try_refuse(f"{name}() (a contract-driven call: its non-property raises are not routed)", node)
        contract = _contracts.contract_for_builtin(name)
        if contract is None:
            raise Unsupported(f"{name}() has no tested contract", node)
        checks = dict((exc, (ok, fact)) for exc, ok, fact in _contracts.raise_checks(contract))
        result, proven = getattr(self, "c_" + name)(st, args, kwargs, node)
        self._contract_checks(node, name, contract, proven)
        return result

    def _contract_checks(self, node, op: str, contract, proven, lean: bool = False) -> None:
        """Record the seven property-class checks of a contract-driven call: every class is as the
        contract states it (``raise_checks``: proven only for a TESTED ``never``), and only the
        classes ``proven`` (class -> why) are upgraded by what the model knows of THESE abstract
        arguments.  ``lean``: record only the classes left UNPROVEN (the read of an attribute is on the hot path of every program with an
        object, and a proven class is no obligation: nothing is lost, the verdict is the failing records)."""
        checks = dict((exc, (ok, fact)) for exc, ok, fact in _claims(contract)[2])
        for exc, fact in proven.items():
            checks[exc] = ((True, f"contract+model: {fact}") if fact is not None else
                           (False, "the model found a failing check of this call: the contract's never covers its tested shape"))
        for exc, (ok, fact) in checks.items():
            if lean and ok:
                continue
            if self._try_stack and exc in proven and proven[exc] is None:
                # _proven_classes vetoes a claim only after an actual failing model check.
                # check() already routed that check's original state; a duplicate contract
                # check has no raising state and must not invent another exception.
                continue
            # a distinct `op` per class, as in _contract_call: one shared op would merge the
            # seven into a single Check whose ok is dragged down by the worst class
            self.check(node, f"{op}:{exc}", {exc}, ok, fact)

    def _modelled(self, st: State, node, qualname: str, model, args, kwargs, hooks: str, *, native: dict | None = None,
                  type_free: str | None = None):
        """A builtin this analyzer modelled by hand before its contract was consulted, read through its rule ([internal ref] step 2: the
        ``c_ord`` recipe applied to a model that already existed).  Returns ``(result, proven)`` for ``_contract_builtin_call``.

        ``model`` is the transfer function exactly as it was; it records its own checks, and ``_contract_checks`` then records the
        seven classes as the CONTRACT states them (proven only for a tested ``never``).  A class is upgraded only on an argument
        the model has made, never on the absence of a record, and which kind of argument is the contract's say:

        * a class the contract admits only through a user hook (the claim ``dunder_propagation_claims`` writes, tested): ``hooks``
          says why none runs, a statement about what the analyzer accepts, written next to the model it is true of;
        * a class that is the call's OWN logic (``TypeError`` for arity and argument kind, ``IndexError`` of ``list.pop``;
          ``adapters/common.py::native_raise_condition``): upgraded when the model recorded at least one check of it at this call
          (``_arity`` records a ``TypeError`` one for every call) and every one of them held, or on the argument ``native`` names
          for it.  ``type_free`` names the reason a call that records no ``TypeError`` check has no source of it (``print``
          without ``sep``, ``end`` or ``file``);
        * an untested or unknown claim is never upgraded;
        * a class the model recorded a FAILING check of at this call (a wrong arity: the contract's claims, a ``never`` or a propagation
          through a dunder, hold for the shape its tests ran) is VETOED: ``proven[exc] = None``, which ``_contract_checks`` records
          unproven."""
        from tharkun.solve.contracts.adapters import absint as _contracts

        contract = _contracts.contract_for_builtin(qualname)
        seen: list = []
        self._taps.append((node, seen))
        try:
            result = model(st, args, kwargs, node)
        finally:
            self._taps.pop()
        return result, self._proven_classes(contract, seen, hooks, native, type_free)

    @staticmethod
    def _proven_classes(contract, seen: list, hooks: str, native: dict | None = None, type_free: str | None = None) -> dict:
        """The classes of ``contract`` a model may upgrade, from the checks it recorded at the call (``seen``: ``[(classes, ok)]``):
        see :meth:`_modelled` for the rules.  A vetoed class maps to ``None``."""
        proven: dict = {}
        for exc, kind in (() if contract is None else _claims(contract)[1]):
            if kind == "never":
                continue                                    # the contract already says it
            if kind == "native":
                held = [ok for excs, ok in seen if exc in excs]
                if native and exc in native:
                    proven[exc] = native[exc]
                elif held and all(held):
                    proven[exc] = "the model's own checks of this call held"
                elif not held and exc == "TypeError" and type_free is not None:
                    proven[exc] = type_free
            elif kind == "propagate":
                proven[exc] = hooks
        for excs, ok in seen:
            if not ok:
                # the contract's claims (a ``never``, a propagation through a dunder) hold for the call shape its tests ran: a call
                # the model found failing (a wrong arity) is not covered by them, so the class is NOT proven by them
                for exc in excs:
                    if contract is not None:
                        proven[exc] = None
        return proven

    def _subscript_quals(self, st: State, obj: V, op: str) -> list:
        """The rules of the builtin container kinds a subscript ``op`` (``__getitem__`` / ``__setitem__``) may reach: the qualnames
        with a hook table (``_METHOD_HOOKS``) among ``list``, ``tuple`` and ``dict`` receivers."""
        out = []
        for a in obj.refs:
            ob = st.heap[a]
            if isinstance(ob, DefaultDictObj) and op == "__getitem__":
                qualname = f"collections.defaultdict.{op}"
            else:
                qualname = f"{ob.kind}.{op}" if ob.kind in ("list", "tuple", "dict") else None
            if qualname in _METHOD_HOOKS and qualname not in out:
                out.append(qualname)
        return out

    def _through_subscript_rules(self, quals: list, st: State, node, idx, model):
        """A subscript of a builtin container, read through the rules of the kinds it may reach: every contract must be TESTED
        (else the subscript is a named refusal, not the model of old), the unchanged model runs ONCE, and each rule then records
        its seven classes as ``<qualname>:<class>`` from the checks the model made at this node (``AbsintVisitor`` visits)."""
        from tharkun.solve.contracts.adapters import absint as _contracts
        from tharkun.solve.contracts.rules import Refused, get_rule

        for q in quals:
            if _contracts.contract_for_builtin(q) is None:
                raise Unsupported(f"{q} has no tested contract", node)
        native: dict = {}
        if idx[0] == "slice":
            native["IndexError"] = "a slice key never raises IndexError: out-of-range bounds are clamped (the contract's own note)"
        elif not idx[1].top and self._as_int(st, idx[1]) is None:
            native["IndexError"] = "the key is not an int: a TypeError, never an IndexError"
        seen: list = []
        self._taps.append((node, seen))
        try:
            result = model()
        finally:
            self._taps.pop()
        for q in quals:
            out = get_rule(q).accept(self.visitor, SubscriptCtx(q, node, tuple(seen), native))
            if isinstance(out, Refused):
                raise Unsupported(str(out), node)
        return result

    def _record_subscript(self, qualname: str, ctx) -> None:
        from tharkun.solve.contracts.adapters import absint as _contracts

        contract = _contracts.contract_for_builtin(qualname)
        proven = self._proven_classes(contract, list(ctx.seen), _METHOD_HOOKS[qualname], dict(ctx.native))
        self._contract_checks(ctx.node, qualname, contract, proven)

    def c_open(self, st, args, kwargs, node):
        """``open(path, <constant write-only mode>, ...)`` ([internal ref], the shape contract ``open.output``):
        a fresh output handle.  Only the shape the environment model classifies as OUTPUT
        (``solve/environment.py::open_effect``) is modelled; a read, an update mode, a
        descriptor, ``buffering``, ``closefd``, ``opener`` or a mode that is not a constant stays
        unmodelled (Unsupported).

        The path must be a definite ``str``: an int may be a file descriptor, a bool too, and an
        object may run ``__fspath__``; a ``None``, float or other non-path part is a TypeError
        (unproven, recorded).  ``encoding``, ``errors`` and ``newline`` must be definite ``str`` or
        ``None``.  What ``open`` can raise beyond that (OSError, ValueError, LookupError) is not a
        property class and ends the path.  With a definite-``str`` path no user code runs, so the
        six classes the contract admits only through ``__fspath__`` cannot occur."""
        from tharkun.solve.environment import open_effect

        if open_effect(node) != "output":
            raise Unsupported("open: not a write-only constant mode", node)
        names = ("file", "mode", "buffering", "encoding", "errors", "newline")
        given = dict(zip(names, args))
        given.update(kwargs)
        if "buffering" in given:
            raise Unsupported("open: buffering is not modelled", node)
        mode_node = node.args[1] if len(node.args) > 1 else next(k.value for k in node.keywords if k.arg == "mode")
        binary = "b" in mode_node.value
        path = given["file"]
        if path.top or path.refs or path.st or path.i is not None or path.b:
            raise Unsupported("open: a path that may be a descriptor or an object", node)
        proven = {exc: "a str path: no __fspath__ runs"
                  for exc in ("AttributeError", "IndexError", "KeyError", "ZeroDivisionError",
                              "FloatingPointError", "AssertionError")}
        ok = path.s is not None and not (path.f or path.n)
        for what in ("encoding", "errors", "newline"):
            if what in given:
                v = given[what]
                if v.top or v.refs or v.st or v.i is not None or v.b or v.f:
                    raise Unsupported(f"open: {what} of an unknown type", node)
        if binary and any(w in given for w in ("encoding", "errors", "newline")):
            raise Unsupported("open: a binary mode with a text option", node)
        if ok:
            proven["TypeError"] = "the path is a str and every option is a str or None"
        addr = self.alloc(st, node, lambda sid: OutFileObj(binary), "outfile")
        return V(refs=frozenset({addr})), proven

    def c_ord(self, st, args, kwargs, node):
        """``ord(c)``: TypeError iff ``c`` is not a str (or bytes/bytearray) of length 1; the
        contract says it raises nothing else and runs no callback (a str subclass's
        ``__len__`` is never called), so the result is exactly the code point."""
        if not self._arity(node, args, kwargs, 1, 1, "ord"):
            st.set_bottom()
            return BOT, {}
        v = args[0]
        if v.top or v.refs or v.st:
            raise Unsupported("ord of an unknown or object value", node)
        out = BOT
        proven = {}
        if v.s is not None:
            if v.s == ANY_STR:
                out = int_iv(0, 0x10FFFF)
            else:
                codes = sorted(ord(c) for c in v.s if len(c) == 1)
                if codes:
                    out = int_iv(codes[0], codes[-1])
                if len(codes) == len(v.s) and not (v.i is not None or v.b or v.f or v.n):
                    proven["TypeError"] = "every possible argument is a str of length 1"
        # int/bool/float/None parts raise TypeError and contribute no continuing path
        if out.is_bot():
            self.check(node, "ord:arg", {"TypeError"}, False, "ord() always raises TypeError here")
            st.set_bottom()
            return BOT, {}
        return out, proven

    def c_chr(self, st, args, kwargs, node):
        raises_valueerror = self._primitive_valueerror_call(st, "chr", args, kwargs, node)
        if not raises_valueerror:
            self._in_try_refuse("chr() (ValueError for a code point out of range)", node)
        """``chr(i)``: TypeError iff ``i`` has no ``__index__``; an int or bool never runs a
        callback; an out-of-range value is ValueError (not a property class), so the result
        is a one-character str on every continuing path."""
        if not self._arity(node, args, kwargs, 1, 1, "chr"):
            st.set_bottom()
            return BOT, {}
        v = args[0]
        if v.top or v.refs or v.st:
            raise Unsupported("chr of an unknown or object value", node)
        # no object can reach here (refs/st refused above), so no __index__ runs and the
        # six classes the contract admits only through it cannot occur
        proven = {exc: "no object argument: no __index__ runs"
                  for exc in ("AttributeError", "IndexError", "KeyError", "ZeroDivisionError",
                              "FloatingPointError", "AssertionError")}
        if self._as_int(st, v) is None:
            self.check(node, "chr:arg", {"TypeError"}, False, "chr() always raises TypeError here")
            st.set_bottom()
            return BOT, {}
        if v.s is None and not (v.f or v.n):
            proven["TypeError"] = "the argument is an int or bool"
        if self._try_stack and raises_valueerror:
            self.check(node, "chr:range", {"ValueError"}, False,
                       "chr() argument is outside the Unicode ordinal range", st=st)
            st.set_bottom()
            return BOT, proven
        return ANYSTR, proven

    def c_tuple(self, st, args, kwargs, node):
        """``tuple(x)``: the same iteration as ``list(x)`` (``_materialize``), into an
        immutable tuple (an exact tuple comes back as itself). TypeError iff ``x`` is not
        iterable. The hooks the contract names (``__iter__``/``__next__``/``__getitem__``)
        either do not exist for a modelled builtin iterable or, for an instance, are
        ANALYSED by ``iterable()`` (its ``__iter__`` body is run abstractly and must return
        a modelled iterator; any other protocol is Unsupported), so the six classes the
        contract admits only through them are accounted for here."""
        result, iterable_ok = self._materialize(st, args, kwargs, node, "tuple")
        if result is BOT:
            return BOT, {}
        proven = {exc: "no hook runs for a builtin iterable; an instance's __iter__ body is analysed"
                  for exc in ("AttributeError", "IndexError", "KeyError", "ZeroDivisionError",
                              "FloatingPointError", "AssertionError")}
        if iterable_ok:
            proven["TypeError"] = "the argument is iterable"
        return result, proven

    def _contract_call(self, st: State, mod: str, name: str, args, kwargs, node) -> V:
        """[internal ref] step 2: the transfer function for a call ``module_attr`` tagged
        ``"cfunc"`` -- a contract exists but no hand-written ``m_<mod>_<name>``
        model does. Records an honest, TOTAL-over-property-classes check (see
        ``solve/contracts/adapters/absint.py::raise_checks``) instead of the
        ``Unsupported`` abort this call would otherwise cause, then returns the
        contract's return type when it is precise enough to be sound (float/str/
        bool only, and only when the contract cannot run an unmodelled callback
        -- see ``simple_return_kind``), else the same ``V(top=True)`` an abort
        would have left the rest of the analysis to work with anyway.

        Receives the ACTUAL call arguments, ``kwargs``, and ``st`` (the engine's
        abstract state) -- 2026-09-28 owner review: a contract-driven call must
        (a) check applicability -- this EXACT call's shape (positional count, no
        keywords) against ``solve/contracts/adapters/common.py::CALL_SHAPE`` --
        before trusting anything precise, including a bare "never raises
        TypeError" claim (a shape mismatch raises TypeError regardless of what
        the contract's own claim assumed); (b) check its PRECISE raise condition
        against what the state already knows, not just record a blanket unproven
        check; and (c) APPLY its declared effect on that state -- or, when no
        hand-verified precise effect applies to THIS call, conservatively HAVOC
        the receiver rather than leave it stale ("a stale abstract list is
        unsound"). All three are wired here via the hand-verified shapes
        ``common.py`` exposes, built on this codebase's own trusted list
        machinery (``len_lin``/``set_len``/``_len_plus``/``update_obj``/
        ``join_into``). Anything without a matching hand-verified shape keeps
        the prior, fully conservative treatment.
        """
        from tharkun.solve.contracts.adapters import absint as _contracts

        qualname = f"{mod}.{name}"
        self._in_try_refuse(f"{qualname} (a contract-driven call: its non-property raises are not routed)", node)
        contract = _contracts.contract_for_module_func(mod, name)
        checks = dict((exc, (ok, fact)) for exc, ok, fact in _contracts.raise_checks(contract))

        arg_count, has_kwargs = len(args), bool(kwargs)
        if checks["TypeError"][0] and not _contracts.applicable_call_shape(qualname, arg_count, has_kwargs):
            # A TESTED "never raises TypeError" claim implicitly assumes the
            # callable's own documented call shape; this call's shape was never
            # hand-verified against it, so don't let it stand as a proof.
            checks["TypeError"] = (False, "contract: call shape not hand-verified for this arity/keywords")

        precise_exc = _contracts.empty_first_arg_precondition(
            qualname, arg_count=arg_count, has_keywords=has_kwargs)
        length_spec = _contracts.receiver_length_delta(
            qualname, arg_count=arg_count, has_keywords=has_kwargs)
        addrs = None
        if args and (precise_exc is not None or length_spec is not None or contract.effects.mutates_receiver):
            v0 = args[0]
            if len(v0.refs) >= 1 and not (v0.top or v0.st or v0.i is not None or v0.f or v0.s is not None):
                cand = frozenset(a for a in v0.refs if st.heap.get(a) is not None and st.heap[a].kind == "list")
                if cand and len(cand) == len(v0.refs):
                    addrs = cand

        if precise_exc is not None and addrs is not None:
            ok_all, facts = True, []
            for a in addrs:
                ob = st.heap[a]
                llin, liv = self.len_lin(st, ob)
                ok = False
                if llin is not None:
                    lo, _hi = lin_bounds(llin, st.zone)
                    ok = lo is not None and lo >= 1
                elif liv[0] is not None:
                    ok = liv[0] >= 1
                ok_all = ok_all and ok
                facts.append(f"len >= 1 of {self.plin(llin)}" if ok else "may be empty")
            checks[precise_exc] = (ok_all, "contract+state: " + "; ".join(facts))

        for exc, (ok, fact) in checks.items():
            # A distinct `op` PER CLASS: `check()` keys its Check object by
            # (line, col, op), and a single shared op across all 7 classes would
            # merge them into one Check whose `ok` is dragged down to False by
            # whichever class is worst -- silently defeating a precise proof for
            # a DIFFERENT, individually-requested class at the same site.
            self.check(node, f"{qualname}:{exc}", {exc}, ok, fact)

        if addrs is not None and contract.effects.mutates_receiver:
            self.collapse(st, addrs)
            if length_spec is not None:
                arg_index, delta, new_elem_idx = length_spec
                if new_elem_idx is not None and new_elem_idx < len(args):
                    # APPLY the precise effect: a new element enters the receiver
                    # (heapq.heappush/bisect.insort) -- join its own value into the
                    # tracked element facts instead of leaving them stale.
                    new_val = args[new_elem_idx]
                    self.update_obj(st, addrs, lambda ob, s: self.set_len(
                        st, ob.replace(elem=self.join_into(st, ob.elem, new_val)), s,
                        *self._len_plus(st, ob, delta)))
                else:
                    # APPLY the precise effect: only length changes (heapq.heappop
                    # removes one; heapq.heapify's delta is 0, a precise no-op).
                    self.update_obj(st, addrs, lambda ob, s: self.set_len(
                        st, ob, s, *self._len_plus(st, ob, delta)))
            else:
                # No hand-verified precise effect for THIS call shape, but the
                # contract still says the receiver is mutated -- 2026-09-28 owner
                # review: "a stale abstract list is unsound". Conservatively
                # HAVOC it (unknown length, unknown elements) rather than leave
                # it exactly as it was before a call that changed it.
                self.update_obj(st, addrs, lambda ob, s: self.set_len(
                    st, ob.replace(elem=V(top=True)), s, None, 0, None))

        kind = _contracts.simple_return_kind(contract)
        if kind == "float":
            return FLOAT
        if kind == "str":
            return ANYSTR
        if kind == "bool":
            return BOOL
        return V(top=True)

    def _copy_copy_list_receiver(self, st, args, kwargs) -> bool:
        if len(args) != 1 or kwargs:
            return False
        value = args[0]
        if (value.top or value.st or value.i is not None or value.b or value.f or value.s is not None
                or value.n or len(value.refs) != 1):
            return False
        return st.heap[next(iter(value.refs))].kind == "list"

    def _copy_copy_list(self, st, args, kwargs, node):
        if not self._arity(node, args, kwargs, 1, 1, "copy.copy"):
            st.set_bottom()
            return BOT
        address = next(iter(args[0].refs))
        return self._list_method_model(st, {address}, "copy", [], {}, node)

    def _arity(self, node, args, kwargs, lo, hi, name) -> bool:
        ok = lo <= len(args) <= hi and not kwargs
        self.check(node, "call-args", {"TypeError"}, ok, f"{name}() arity")
        return ok

    def _bind_call(self, st, node, args, kwargs, what, params, required=(), positional_only=False):
        """A call bound to the parameters of a plain Python function, as CPython binds it ([internal ref]): each parameter by position or by
        NAME, ``required`` the ones with no default.  Returns ``{parameter: value}`` for the ones given, or ``None`` after recording
        the ``TypeError`` of a call that does not bind (too many positional arguments, an unknown keyword, a parameter given twice,
        a required one missing): that exception is certain whatever the values are, so the path ends.  ``positional_only``: a C
        method with no keyword parameters (``getrandbits(k, /)``): any keyword is the ``TypeError``.

        ``_arity`` is for the functions that take positional arguments only (``len``, ``abs``, ``list.append`` ...): there a keyword
        really is a ``TypeError``.  Using it for a function that takes keywords ended the path of a valid call, and everything
        after it went unchecked."""
        why = None
        bound = dict(zip(params, args))
        if len(args) > len(params):
            why = f"{what}() takes at most {len(params)} arguments ({len(args)} given)"
        elif kwargs and positional_only:
            why = f"{what}() takes no keyword arguments"
        else:
            for name, value in kwargs.items():
                if name not in params:
                    why = f"{what}() got an unexpected keyword argument {name!r}"
                    break
                if name in bound:
                    why = f"{what}() got multiple values for argument {name!r}"
                    break
                bound[name] = value
            else:
                missing = [p for p in required if p not in bound]
                if missing:
                    why = f"{what}() missing the argument {missing[0]!r}"
        self.check(node, "call-args", {"TypeError"}, why is None, why or f"{what}() call shape")
        if why is not None:
            st.set_bottom()
            return None
        return bound

    def _num_arg(self, st, v: V, node, what, allow_float=True) -> bool:
        if v.top:
            raise Unsupported("unknown argument", node)
        ok = not (v.s is not None or v.n or v.refs or v.st or (v.f and not allow_float))
        self.check(node, "arg-type", {"TypeError"}, ok, f"{what} argument: " + ",".join(v.kinds()))
        return ok

    def b_abs(self, st, args, kwargs, node):
        return self.visit_rule("abs", st, args, kwargs, node)

    def c_abs(self, st, args, kwargs, node):
        return self._modelled(st, node, "abs", self._abs_model, args, kwargs, _NO_USER_DUNDER.format("__abs__"))

    def _abs_model(self, st, args, kwargs, node):
        if not self._arity(node, args, kwargs, 1, 1, "abs"):
            st.set_bottom()
            return BOT
        v = args[0]
        self._num_arg(st, v, node, "abs")
        out = BOT
        ii = self._as_int(st, v)
        if ii is not None:
            lo, hi = self.ibounds(st, ii)
            if lo is not None and lo >= 0:
                out = V(i=ii)
            elif hi is not None and hi <= 0:
                out = self.unop(st, ast.USub(), V(i=ii), node)
            else:
                m = None if lo is None or hi is None else max(-lo, hi)
                out = int_iv(0, m)
        if v.f:
            # [internal ref]: abs(x) >= 0 for floats too, and NaN only propagates -- it is
            # never introduced.  The value-level transfer (solve/values.py::abs_)
            # already gets this exactly right (outward-rounded interval, -0.0
            # folded to +0.0, nan carried through unchanged); reuse it instead of
            # collapsing the float part to an unconstrained FLOAT (which used to
            # throw the sign and every bound away on every abs() of a float).
            fa = v.f
            res, _raises = fv.abs_(fa.fact)
            if _raises:
                self._in_try_refuse("abs() of a float that may raise", node)
            # abs(x) == x exactly when x is already known non-negative: keep the
            # value term in that case (mirrors the int branch's "V(i=ii)" reuse a
            # few lines up), so abs(-e) == abs(e) is provable for floats too once
            # BOTH sides are known non-negative -- e.g. abs(sqrt(y)) == sqrt(y).
            # A sign flip or a mixed/unknown sign has no "negated" lemma term, so
            # the term is dropped there, same as for a fresh, untermed value.
            term = fa.term if fa.fact.has_interval() and fa.fact.lo >= 0.0 else None
            why = fa.why if term is not None else ""
            out = vjoin(out, st.zone, V(f=FloatInfo(res, term, why)), st.zone)
        if out.is_bot():
            st.set_bottom()
        return out

    def b_len(self, st, args, kwargs, node):
        return self.visit_rule("len", st, args, kwargs, node)

    def c_len(self, st, args, kwargs, node):
        return self._modelled(st, node, "len", lambda st, args, kwargs, node: self._len_model(st, args, kwargs, node, "protocol.container_len"),
                              args, kwargs, "an instance's __len__ body is analysed (special); a builtin container runs no hook")

    def _len_model(self, st, args, kwargs, node, protocol=None):
        """``len(x)`` as the analyzer models it; the implicit ``len`` of a truth value and of a length probe (``truth_split``,
        ``_probe_length``) is this model, not the call a program writes.  ``protocol`` is the rule the length of an instance is read
        through (``protocol.container_len`` for the call ``len(x)``; the other two callers ask their own protocol)."""
        if not self._arity(node, args, kwargs, 1, 1, "len"):
            st.set_bottom()
            return BOT
        v = args[0]
        if v.top:
            raise Unsupported("len of an unknown value", node)
        bad = bool(v.i is not None or v.b or v.f or v.n or v.st)
        out = BOT
        set_lens = []
        for a in v.refs:
            ob = st.heap[a]
            if ob.kind in ("list", "tuple", "array"):
                lin, iv = self.len_lin(st, ob)
                ii = IntInfo(lin) if lin is not None else IntInfo.interval(*iv)
                out = vjoin(out, st.zone, V(i=self.to_temp(st, ii)), st.zone)
            elif ob.kind == "range":
                raise Unsupported("len(range)", node)
            elif ob.kind in ("dict", "dictview"):
                lo, hi = self.dict_len(st, {a} if ob.kind == "dict" else ob.dicts)
                out = vjoin(out, st.zone, int_iv(lo, hi), st.zone)
            elif ob.kind == "set":
                set_lens.append(ob.length)
            elif ob.kind == "inst":
                if len(v.refs) != 1 or v.replace(refs=frozenset()).kinds():
                    return self.map_alternatives(st, v, lambda branch, part:
                        self._len_model(branch, [part], {}, node, protocol))

                def length():
                    result = self.special(st, v, "__len__", [], node)
                    if st.bottom:
                        return BOT
                    if result.refs or result.st or result.n or result.f or result.s is not None:
                        raise Unsupported("non-integer __len__ result", node)
                    ii = self._as_int(st, result)
                    if ii is None or self.ibounds(st, ii)[0] is None or self.ibounds(st, ii)[0] < 0:
                        raise Unsupported("__len__ result may be negative", node)
                    hi = self.ibounds(st, ii)[1]
                    if self._try_stack and (hi is None or hi > sys.maxsize):
                        self._in_try_refuse("__len__ result may overflow Py_ssize_t", node)
                    return V(i=ii)
                if protocol is None:
                    return length()
                return self.visit_protocol(protocol, node, length, _PROTOCOL_NATIVE[protocol])
            else:
                bad = True
        if set_lens:
            # the bounds over every set it may be, copied into ONE temporary as a list's length is (a comparison
            # against it is then decided in the zone; two temporaries would join to no bound at all)
            his = [hi for _lo, hi in set_lens]
            ii = IntInfo.interval(min(lo for lo, _hi in set_lens), None if None in his else max(his))
            out = vjoin(out, st.zone, V(i=self.to_temp(st, ii)), st.zone)
        if v.s is not None:
            lengths = [len(x) for x in v.s] if v.s != ANY_STR else None
            out = vjoin(out, st.zone, int_iv(min(lengths), max(lengths)) if lengths else int_iv(0, None), st.zone)
        self.check(node, "arg-type", {"TypeError"}, not bad, "len of " + ",".join(v.kinds()))
        if out.is_bot():
            st.set_bottom()
        return out


    def b_iter(self, st, args, kwargs, node):
        if len(args) == 2 and not kwargs:
            raise Unsupported("callable/sentinel iterator", node)
        if not self._arity(node, args, kwargs, 1, 1, "iter"):
            st.set_bottom()
            return BOT
        value = self.iterable(st, args[0], node)
        self.check_iterable(st, value, node)
        if value.refs and all(st.heap[a].kind == "iterator" for a in value.refs):
            if value.replace(refs=frozenset()).is_bot():
                return value
        addr = self.alloc(st, node, lambda sid: IteratorObj(value), "iterator")
        return V(refs=frozenset({addr}))

    def b_range(self, st, args, kwargs, node):
        return self.visit_rule("range", st, args, kwargs, node)

    def c_range(self, st, args, kwargs, node):
        return self._modelled(st, node, "range", self._range_model, args, kwargs, _NO_USER_DUNDER.format("__index__"))

    def _range_model(self, st, args, kwargs, node):
        if not self._arity(node, args, kwargs, 1, 3, "range"):
            st.set_bottom()
            return BOT
        for v in args:
            self._num_arg(st, v, node, "range", allow_float=False)
        ints = [self._as_int(st, v) for v in args]
        if any(i is None for i in ints):
            st.set_bottom()
            return BOT
        if len(ints) == 1:
            lo, hi, step = IntInfo.interval(0, 0), ints[0], 1
        else:
            lo, hi = ints[0], ints[1]
            step = 1
            if len(ints) == 3:
                a, b = self.ibounds(st, ints[2])
                if a is None or a != b:
                    raise Unsupported("range with a non-constant step", node)
                step = a
                # step 0 raises ValueError: not a property exception
                if step == 0:
                    if self._try_stack:
                        self.check(node, "range:step", {"ValueError"}, False,
                                   "range() step is zero", st=st)
                    st.set_bottom()
                    return BOT

        def make(sid):
            lv, hv = f"R.{sid}.lo", f"R.{sid}.hi"
            self.zone_assign(st, lv, lo)
            self.zone_assign(st, hv, hi)
            return RangeObj(lv, hv, step)

        a = self.alloc(st, node, make, "range")
        return V(refs=frozenset({a}))

    def _range_len(self, st, ob):
        """(lin, (lo, hi)) of len(range) for a step-1 range."""
        if ob.step != 1:
            return None, (0, None)
        if isinstance(ob.lo, str) and isinstance(ob.hi, str):
            lin = Lin.var(ob.hi).sub(Lin.var(ob.lo)).norm(st.zone)
            if lin_entails_le(st.zone, Lin.const(0), lin):
                return lin, lin_bounds(lin, st.zone)
            lo, hi = lin_bounds(lin, st.zone)
            return None, (0, None if hi is None else max(hi, 0))
        lo_iv = st.zone.bounds(ob.lo) if isinstance(ob.lo, str) else ob.lo
        hi_iv = st.zone.bounds(ob.hi) if isinstance(ob.hi, str) else ob.hi
        hi = None if hi_iv[1] is None or lo_iv[0] is None else max(0, hi_iv[1] - lo_iv[0])
        return None, (0, hi)

    def _elem_sortable(self, elem: V) -> bool:
        """Whether comparing any two of ``elem``'s possible values with ``<`` is
        guaranteed to never raise TypeError, without modelling a custom ``__lt__``
        ([internal ref]): real Python compares int/bool/float freely (the numeric tower) and
        ``str`` only with ``str``; ``None``, an instance/class, or a mix of the two
        families refuses. This is imprecision, never a soundness gate the other way:
        an actually-homogeneous sequence whose abstract summary merely LOOKS mixed
        here is refused too (Unsupported), never wrongly accepted."""
        if elem.is_bot():
            return True
        if elem.refs or elem.st or elem.n or elem.top:
            return False
        if elem.s is not None:
            return elem.i is None and not elem.b and not elem.f
        return True

    def b_sorted(self, st, args, kwargs, node):
        return self.visit_rule("sorted", st, args, kwargs, node)

    def c_sorted(self, st, args, kwargs, node):
        """``sorted(x)`` ([internal ref]): a NEW list, built the same way ``t_list`` already
        builds one for each source kind it supports (range/dict/dictview/iterator/
        list/tuple) -- same length fact (``dict_len``/``len_lin``/``_range_len``,
        nothing more precise attempted), same element fact (``elem_summary``/
        ``all_elems``, a JOIN over every value the source may hold -- sorting can
        never introduce a value the source didn't already have, or remove one it
        did). The one thing ``t_list`` does not need and this DOES: ``_elem_sortable``
        refuses when the elements are not all mutually comparable without a custom
        ``__lt__`` this domain does not model -- real Python's own ``<`` protocol,
        not this analysis inventing an ordering."""
        self._sorted_keywords(st, kwargs, node)
        if len(args) != 1:
            # unlike key=/reverse=, extra POSITIONAL arguments really do raise
            # TypeError in real Python (sorted(iterable, /, *, key=..., reverse=...)
            # takes exactly one positional parameter) -- this IS a genuine,
            # unconditional TypeError, so the path past it really is unreachable.
            self.check(node, "call-args", {"TypeError"}, False, "sorted() arity")
            st.set_bottom()
            return BOT, {}
        v = args[0]
        if v.top:
            raise Unsupported("sorted() of an unknown value", node)
        v = self.iterable(st, v, node, consumer="sorted")
        # shortcut: retain length and per-slot facts; add per-key/order correlations only when assertions need them.
        items_dict = None
        if len(v.refs) == 1 and v.replace(refs=frozenset()).is_bot():
            src = next(iter(v.refs))
            view = st.heap[src]
            if view.kind == "dictview" and view.which == "items" and len(view.dicts) == 1:
                dict_addr = next(iter(view.dicts))
                candidate = st.heap.get(dict_addr)
                if candidate is not None and candidate.kind == "dict":
                    items_dict = candidate
        if items_dict is None:
            iter_ok = self.check_iterable(st, v, node)
        else:
            self.borrow(st, v)
            self.check(node, "iter", {"TypeError"}, True, "a single builtin dict.items() view is iterable", st=st)
            iter_ok = True
            lo, hi = self.dict_len(st, {dict_addr})
            keys, values = self.dict_keys_v(st, items_dict), self.dict_vals_v(st, items_dict)
            if hi == 0:
                elem = BOT
            else:
                if (keys.is_bot() or values.is_bot()
                        or not self._elem_sortable(keys) or not self._elem_sortable(values)):
                    raise Unsupported("sorted(dict.items()) with mixed or unorderable key/value facts", node)
                pair = self.new_list(st, node, None, BOT, tag="sorted-items-pair", kind="tuple",
                                     items=[keys, values])
                elem = self.borrow(st, V(refs=frozenset({pair})))
            out = self.new_list(st, node, IntInfo.interval(lo, hi), elem)
            proven = {exc: "dict.items() has primitive comparable key/value facts; no user hook runs"
                      for exc in NO_HOOK_CLASSES}
            proven["TypeError"] = "the argument is a single builtin dict.items() view"
            return V(refs=frozenset({out})), proven
        if len(v.refs) != 1 or v.s is not None:
            raise Unsupported("sorted() of several possible iterables", node)
        src = next(iter(v.refs))
        ob = st.heap[src]
        if ob.kind == "range":
            lin, iv = self._range_len(st, ob)
            ii = IntInfo(lin) if lin is not None and lin.single() else IntInfo.interval(*iv)
            elem = self.elem_summary(st, v) or BOT
        elif ob.kind in ("dict", "dictview"):
            if ob.kind == "dictview" and ob.which == "items":
                raise Unsupported("sorted(dict.items())", node)
            lo, hi = self.dict_len(st, {src} if ob.kind == "dict" else ob.dicts)
            ii = IntInfo.interval(lo, hi)
            elem = self.elem_summary(st, v) or BOT
        elif ob.kind == "iterator":
            ii = None
            elem = self.elem_summary(st, v) or BOT
        elif ob.kind in ("list", "tuple"):
            lin, iv = self.len_lin(st, ob)
            ii = IntInfo(lin) if lin is not None else IntInfo.interval(*iv)
            elem = self.all_elems(st, ob)
        elif ob.kind == "set":
            ii = IntInfo.interval(*ob.length)
            elem = ob.elem
        else:
            raise Unsupported(f"sorted() of a {ob.kind}", node)
        if not self._elem_sortable(elem):
            raise Unsupported("sorted() of a mixed or unorderable element type", node)
        a = self.new_list(st, node, ii, elem)
        proven = {exc: "numbers and strs only: no user hook runs" for exc in NO_HOOK_CLASSES}
        if iter_ok:
            proven["TypeError"] = "the argument is iterable and its elements are mutually orderable"
        return V(refs=frozenset({a})), proven

    def t_list(self, st, args, kwargs, node):
        return self.visit_rule("list", st, args, kwargs, node)

    def c_list(self, st, args, kwargs, node):
        return self._modelled(
            st, node, "list", self._list_model, args, kwargs,
            "an instance's __iter__ body is analysed (iterable()) and its __len__ probe is ruled by protocol.length_hint; "
            "a builtin iterable runs no hook")

    def c_collections_defaultdict(self, st, args, kwargs, node):
        return self._contract_module_model("collections.defaultdict", st, self._defaultdict_model,
                                           args, kwargs, node)

    def _defaultdict_model(self, st, args, kwargs, node):
        if kwargs or len(args) > 1:
            raise Unsupported("defaultdict initializer or keyword arguments", node)
        factory_value = args[0] if args else NONE
        if (factory_value.top or factory_value.i is not None or factory_value.b or factory_value.f
                or factory_value.s is not None or factory_value.refs):
            raise Unsupported("defaultdict with a user or unsupported factory", node)
        if factory_value.n and not factory_value.st:
            factory = None
        elif not factory_value.n and factory_value.st == frozenset({("type", "list")}):
            factory = "list"
        else:
            raise Unsupported("defaultdict with a mixed or unsupported factory", node)
        self.check(node, "call-args", {"TypeError"}, True,
                   "accepted no-factory or builtin-list factory call shape", st=st)
        address = self.alloc(st, node, lambda _sid: DefaultDictObj({}, True, BOT, BOT, factory),
                             "collections.defaultdict")
        return V(refs=frozenset({address}))

    def _list_model(self, st, args, kwargs, node):
        return self._materialize(st, args, kwargs, node, "list")[0]

    def _materialize(self, st, args, kwargs, node, kind):
        """``list(x)`` (``kind == "list"``) or ``tuple(x)`` (``"tuple"``): a NEW sequence of the
        elements of ``x``. Returns ``(result, iterable_ok)``; ``iterable_ok`` is True iff ``x``
        is certainly iterable (the TypeError check held). A tuple source is returned as ITSELF
        for ``tuple`` (CPython: an exact tuple comes back unchanged); every other source is
        copied, whose length and element summary are what ``x`` held when this ran."""
        if not self._arity(node, args, kwargs, 0, 1, kind):
            st.set_bottom()
            return BOT, False

        def new(length, elem, items=None):
            return V(refs=frozenset({self.new_list(st, node, length, elem, tag=kind, kind=kind, items=items)}))

        if not args:
            return new(IntInfo.interval(0, 0), BOT, items=[] if kind == "tuple" else None), True
        v = args[0]
        if v.top:
            raise Unsupported(f"{kind}() of an unknown value", node)
        v = self.iterable(st, v, node, consumer=kind)
        iterable_ok = self.check_iterable(st, v, node)
        if len(v.refs) != 1 or v.s is not None:
            if v.s is not None and not v.refs:
                return new(None, ANYSTR), iterable_ok
            raise Unsupported(f"{kind}() of several possible iterables", node)
        src = next(iter(v.refs))
        ob = st.heap[src]
        if ob.kind == "range":
            lin, iv = self._range_len(st, ob)
            ev = self.elem_summary(st, v) or BOT
            ii = IntInfo(lin) if lin is not None and lin.single() else IntInfo.interval(*iv)
            if lin is not None and lin.single() is None:
                ii = IntInfo.interval(*lin_bounds(lin, st.zone))
            # element bounds keep their symbolic range anchors
            if ob.step > 0:
                lo = (ob.lo, 0) if isinstance(ob.lo, str) else None
                hi = (ob.hi, -1) if isinstance(ob.hi, str) else None
                ev = V(i=IntInfo(None, lo, hi))
                # anchor the element bounds to stable variables (range vars die with the range)
                ev = self._reanchor(st, ev)
            return new(ii, ev), iterable_ok
        if ob.kind in ("dict", "dictview"):
            if ob.kind == "dictview" and ob.which == "items":
                raise Unsupported(f"{kind}(dict.items())", node)
            lo, hi = self.dict_len(st, {src} if ob.kind == "dict" else ob.dicts)
            elem = self.elem_summary(st, v) or BOT
            return new(IntInfo.interval(lo, hi), elem), iterable_ok
        if ob.kind == "iterator":
            elem = self.elem_summary(st, v) or BOT
            return new(None, elem), iterable_ok
        if kind == "tuple" and ob.kind == "tuple":
            return V(refs=frozenset({src})), iterable_ok
        if ob.kind == "set":
            return new(IntInfo.interval(*ob.length), ob.elem), iterable_ok
        if ob.kind in ("list", "tuple", "array"):
            lin, iv = self.len_lin(st, ob)
            elem = self.all_elems(st, ob)
            return new(IntInfo(lin) if lin is not None else IntInfo.interval(*iv), elem), iterable_ok
        raise Unsupported(f"{kind}() of a {ob.kind}", node)

    def _reanchor(self, st: State, v: V) -> V:
        """Rewrite bounds on range variables to equal program variables when possible."""
        ii = v.i
        if ii is None:
            return v
        z = st.zone

        def re(b):
            if b is None or b[0] == ZERO:
                return b
            x, c = b
            for y in sorted(z.vars):
                if y in (x, ZERO) or y.startswith(("$", "R.")):
                    continue
                if z.get(x, y) == 0 and z.get(y, x) == 0:
                    return (y, c)
            lo, hi = z.bounds(x)
            return b

        return v.replace(i=IntInfo(None, re(ii.lo), re(ii.hi)))

    def t_int(self, st, args, kwargs, node):
        return self.visit_rule("int", st, args, kwargs, node)

    def c_int(self, st, args, kwargs, node):
        return self._modelled(st, node, "int", self._int_model, args, kwargs, _NO_USER_DUNDER.format("__int__/__index__"))

    def _int_model(self, st, args, kwargs, node):
        if len(args) == 2 or "base" in kwargs:
            raise Unsupported("int conversion with an explicit base", node)
        if not self._arity(node, args, kwargs, 0, 1, "int"):
            st.set_bottom()
            return BOT
        if not args:
            return const_int(0)
        v = args[0]
        if self._primitive_valueerror_call(st, "int", args, kwargs, node):
            self.check(node, "int:literal", {"ValueError"}, False,
                       "int() literal has no decimal digit", st=st)
            st.set_bottom()
            return BOT
        if v.s is not None or v.f:
            # int("x") raises ValueError, int(inf)/int(nan) OverflowError/ValueError: not property classes, the
            # path continues on the accepted values; inside a try the handlers would see them ([internal ref]): refused
            self._in_try_refuse("int() of a str or float (ValueError / OverflowError)", node)
        ok = not (v.n or v.refs or v.st or v.top)
        self.check(node, "arg-type", {"TypeError"}, ok, "int() argument")
        ii = self._as_int(st, v)
        out = V(i=ii) if ii is not None else BOT
        if v.f or v.s is not None:
            out = vjoin(out, st.zone, int_iv(None, None), st.zone)
        if out.is_bot():
            st.set_bottom()
        return out

    def t_float(self, st, args, kwargs, node):
        return self.visit_rule("float", st, args, kwargs, node)

    def c_float(self, st, args, kwargs, node):
        return self._modelled(st, node, "float", self._float_model, args, kwargs, _NO_USER_DUNDER.format("__float__/__index__"))

    def _float_model(self, st, args, kwargs, node):
        if not self._arity(node, args, kwargs, 0, 1, "float"):
            st.set_bottom()
            return BOT
        if args:
            v = args[0]
            if self._primitive_valueerror_call(st, "float", args, kwargs, node):
                self.check(node, "float:literal", {"ValueError"}, False,
                           "float() literal has no decimal digit and is not NaN/Inf", st=st)
                st.set_bottom()
                return BOT
            if v.s is not None:
                self._in_try_refuse("float() of a str (ValueError)", node)
            ok = not (v.n or v.refs or v.st or v.top)
            self.check(node, "arg-type", {"TypeError"}, ok, "float() argument", st=st)
        return FLOAT

    def t_object(self, st, args, kwargs, node):
        raise Unsupported("object()", node)

    def _minmax(self, st, args, kwargs, node, is_min):
        """``min``/``max`` ([internal ref] batch): one positional iterable (``reducers.py``), or two or more
        positional values.  The tested contract (``contract_for_builtin``) gates both through
        ``_contract_builtin_call``; the result of ``c_min``/``c_max`` is ``(value, proven classes)``."""
        name = "min" if is_min else "max"
        if not args:
            self.check(node, "call-args", {"TypeError"}, False, f"{name}() arity")
            st.set_bottom()
            return BOT, {}
        if len(args) == 1:
            return self._minmax_iterable(st, args[0], kwargs, node, is_min)
        for k, kv in kwargs.items():
            if k != "key" or not (kv.n and kv.i is None and not kv.b and not kv.f and kv.s is None
                                  and not kv.refs and not kv.st and not kv.top):
                raise Unsupported(f"{name}() with keywords over several values", node)
        if self.custom_comparison(st, args):
            raise Unsupported("min/max invoking user comparison", node)
        ints = [self._as_int(st, a) for a in args]
        ok = True
        for a in args:
            ok = self._num_arg(st, a, node, "min/max") and ok
        proven = {exc: "no user comparison runs" for exc in NO_HOOK_CLASSES}
        if ok:
            proven["TypeError"] = "every argument is a number"
        if all(a.only_int() or (a.i is None and a.b) for a in args) and all(i is not None for i in ints):
            bs = [self.ibounds(st, i) for i in ints]
            if is_min:
                lo = None if any(b[0] is None for b in bs) else min(b[0] for b in bs)
                his = [b[1] for b in bs if b[1] is not None]
                hi = min(his) if his else None
            else:
                los = [b[0] for b in bs if b[0] is not None]
                lo = max(los) if los else None
                hi = None if any(b[1] is None for b in bs) else max(b[1] for b in bs)
            ii = IntInfo.interval(lo, hi)
            # relational: min(a, b) <= a, b ; max(a, b) >= a, b
            t = self.fresh(st)
            self.zone_assign(st, t, ii)
            for i in ints:
                if i.lin is not None and i.lin.single() is not None and i.lin.single()[0] != ZERO:
                    x, c = i.lin.single()
                    if is_min:
                        st.zone.add(t, x, c)
                    else:
                        st.zone.add(x, t, -c)
            return V(i=IntInfo(Lin.var(t))), proven
        out = BOT
        for a in args:
            out = vjoin(out, st.zone, a, st.zone)
        return out, proven

    def b_min(self, st, args, kwargs, node):
        return self.visit_rule("min", st, args, kwargs, node)

    def b_max(self, st, args, kwargs, node):
        return self.visit_rule("max", st, args, kwargs, node)

    def c_divmod(self, st, args, kwargs, node):
        """Model direct ``divmod(int, int)`` calls with one call-site ZeroDivisionError check."""
        if (kwargs or len(args) != 2 or not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
                or node.func.id != "divmod" or len(node.args) != 2 or node.keywords
                or any(not v.only_int() for v in args)):
            raise Unsupported("divmod() requires two exact-int operands", node)
        a, b = (v.i for v in args)
        if a is None or b is None:
            raise Unsupported("divmod() requires two exact-int operands", node)
        facts = []
        q, may_zero = self._int_binop(st, ast.FloorDiv(), a, b, node, facts)
        r, _ = self._int_binop(st, ast.Mod(), a, b, node, facts)
        lo, hi = self.ibounds(st, b)
        definitely_zero = lo == hi == 0
        a_lo, a_hi = self.ibounds(st, a)
        if a_lo is not None and a_lo == a_hi and lo is not None and lo == hi and not definitely_zero:
            q = const_int(a_lo // lo)
            r = const_int(a_lo % lo)
        proven = {exc: "exact builtin int operands run no user hook" for exc in PROPERTY_EXCS
                  if exc != "ZeroDivisionError"}
        if may_zero:
            proven["ZeroDivisionError"] = None
            if self._try_stack:
                self.check(node, "divmod:ZeroDivisionError", {"ZeroDivisionError"}, False,
                           "integer divisor may be zero", st=st)
        else:
            proven["ZeroDivisionError"] = "the integer divisor is nonzero"
        if definitely_zero:
            st.set_bottom()
            return BOT, proven
        addr = self.new_list(st, node, None, BOT, tag="tuple", kind="tuple", items=[q, r])
        return V(refs=frozenset({addr})), proven

    def c_round(self, st, args, kwargs, node):
        """Model one direct ``round`` call on a built-in int, bool, or float."""
        return self._modelled(
            st, node, "round", self._round_model, args, kwargs,
            "the argument is an exact built-in int, bool, or float, so no custom __round__ runs",
            native={exc: "one direct positional exact built-in numeric argument" for exc in PROPERTY_EXCS},
        )

    def _round_model(self, st, args, kwargs, node):
        if not self._primitive_valueerror_call(st, "round", args, kwargs, node):
            raise Unsupported("round() requires one direct positional exact built-in int, bool, or float", node)
        value = args[0]
        if value.only_int():
            return V(i=self._as_int(st, value))
        if value.kinds() == ["bool"]:
            ints = tuple(int(v) for v in value.b)
            return int_iv(min(ints), max(ints))

        rounded, raises = fv.round_(value.f.fact)
        if raises:
            self._in_try_refuse("round() of a float that may raise ValueError or OverflowError", node)
        if rounded.empty:
            st.set_bottom()
            return BOT
        return int_iv(rounded.lo, rounded.hi)

    def c_min(self, st, args, kwargs, node):
        return self._minmax(st, args, kwargs, node, True)

    def c_max(self, st, args, kwargs, node):
        return self._minmax(st, args, kwargs, node, False)

    def b_print(self, st, args, kwargs, node):
        return self.visit_rule("print", st, args, kwargs, node)

    def c_print(self, st, args, kwargs, node):
        """``print(*values, sep=, end=, file=, flush=)``: the contract's native sources are ``sep``/``end``/``file`` (TypeError) and
        a ``file`` without ``write`` (AttributeError).  Without them there is none; with ``file`` the model accepts only a
        modelled ``io.StringIO``, and the default is the process's own stdout, which no program the analysis accepts can rebind
        (a store to a module attribute, ``setattr`` and ``del`` of one are each refused)."""
        stdout = ("stdout is the process's own (rebinding it is refused) and file= is a modelled StringIO, which has write, "
                  "or refused")
        return self._modelled(
            st, node, "print", self._print_model, args, kwargs,
            "a printed instance with a custom __str__/__repr__/__format__ is refused (check_formattable)",
            native={"AttributeError": stdout},
            type_free="every source of a TypeError is checked by the model (sep and end are recorded) or accepted only as a "
                      "modelled StringIO (file), and there is none without the keywords")

    def _print_model(self, st, args, kwargs, node):
        for a in args:
            self.check_formattable(st, a, node)
        for k, v in kwargs.items():
            if k == "file":
                self._print_to_file(st, v, node)
                continue
            if k not in ("sep", "end", "flush"):
                raise Unsupported(f"print({k}=...)", node)
            if k == "flush":
                t, f = self.truth_split(st, v, node)
                self._merge_branches(st, [(t, NONE), (f, NONE)])
            else:
                if v.top:
                    raise Unsupported("unknown print separator", node)
                self.check(node, "print", {"TypeError"}, not (v.i or v.b or v.f or v.refs or v.st),
                           "separator/end must be str or None")
        return NONE

    def _print_to_file(self, st: State, v: V, node) -> None:
        """``print(..., file=X)`` ([internal ref]): sound only when every possible ``X``
        is a modelled ``io.StringIO`` buffer -- writes the printed text into it,
        coarsened to ``ANYSTR`` exactly like an explicit ``.write()`` call (see
        ``StringIOObj``). Anything else (``sys.stdout``, a user file-like object, an
        unknown or mixed value) is Unsupported: this pass models exactly one file-like
        kind, and guessing "print(file=X) never raises" for an unmodelled X would risk
        missing whatever ``X.write()`` itself can do."""
        if v.top or v.i is not None or v.b or v.f or v.s is not None or v.n or v.st:
            raise Unsupported("print(file=...) with an unmodelled destination", node)
        addrs = frozenset(a for a in v.refs if a in st.heap and st.heap[a].kind == "stringio")
        if addrs != v.refs:
            raise Unsupported("print(file=...) with an unmodelled destination", node)
        self.update_obj(st, addrs, lambda ob, strong: StringIOObj(ANYSTR))

    #: the builtin class names an ``isinstance`` classinfo may carry, and the scalar kinds each is a class of
    _ISINSTANCE_SCALARS = {"int": ("int", "bool"), "bool": ("bool",), "float": ("float",), "str": ("str",)}

    def _classinfo(self, st: State, t: V, node, names: set, classes: set, depth: int = 0) -> None:
        """Collect what the second argument of ``isinstance`` names: builtin class names (``str``, ``int``,
        ``float``, ``bool``, ``list``, ``tuple``, ``object``) into ``names``, user class addresses into ``classes``;
        a tuple of those is every one of them.  Anything else (an unknown value, a scalar, an exception class, a
        tuple that is a summary) refuses: it may be a TypeError, or a class whose instance check this model
        does not know."""
        if t.top or t.i is not None or t.b or t.f or t.s is not None or t.n or depth > 3:
            raise Unsupported("isinstance with an unsupported type argument", node)
        if not t.st and not t.refs:
            raise Unsupported("isinstance with an unsupported type argument", node)
        for tag in t.st:
            if tag[0] == "type" and tag[1] in ("str", "int", "float", "bool", "list", "object"):
                names.add(tag[1])
            elif tag == ("cbuiltin", "tuple"):
                names.add("tuple")
            else:
                raise Unsupported("isinstance with an unsupported type argument", node)
        for a in t.refs:
            ob = st.heap[a]
            if ob.kind == "class":
                classes.add(a)
            elif ob.kind == "tuple" and ob.items is not None:
                for item in ob.items:
                    self._classinfo(st, item, node, names, classes, depth + 1)
            else:
                raise Unsupported("isinstance with an unsupported type argument", node)

    def b_isinstance(self, st, args, kwargs, node):
        return self.visit_rule("isinstance", st, args, kwargs, node)

    def c_isinstance(self, st, args, kwargs, node):
        return self._modelled(
            st, node, "isinstance", self._isinstance_model, args, kwargs,
            "classinfo is a modelled class (a metaclass, hence an __instancecheck__, is refused where a class is defined) and an "
            "object with a __class__ of its own is refused")

    def _isinstance_model(self, st, args, kwargs, node):
        """``isinstance(x, classinfo)``: a bool decided per alternative of ``x``.  The classinfo is a builtin class
        (``str``, ``int``, ``float``, ``bool``, ``list``, ``tuple``, ``object``), a user class, or a tuple of them
        (``_classinfo``); every other classinfo refuses.  Scalars are exact instances of their kind (``bool`` is also
        an ``int``); an object is an instance of its class and of the classes along its single-inheritance chain
        (``s_ClassDef`` refuses multiple inheritance, metaclasses, and every builtin base except ``object`` and the
        exceptions), of ``list``/``tuple`` by its heap kind, and of ``object``.  Nothing else is an instance of a
        user class or a supported builtin name (a function, a module, a class, a str, a number).  An instance
        that carries a ``__class__`` of its own refuses (``isinstance`` consults it).  [internal ref] added the scalar
        names, [internal ref] the user classes, the containers and the tuple."""
        if not self._arity(node, args, kwargs, 2, 2, "isinstance"):
            st.set_bottom()
            return BOT
        v, t = args
        if v.top or t.top:
            raise Unsupported("isinstance of an unmodelled value", node)
        results: set = set()
        # a classinfo that may be one of several values is decided for each; a TUPLE is one value naming all its classes
        for alt in ([V(st=frozenset({tag})) for tag in t.st] + [V(refs=frozenset({a})) for a in t.refs]
                    or [t]):
            names: set = set()
            classes: set = set()
            self._classinfo(st, alt, node, names, classes)
            results |= self._isinstance_results(st, v, names, classes, node)
        if not results:
            st.set_bottom()
            return BOT
        if results == {True}:
            return const_bool(True)
        if results == {False}:
            return const_bool(False)
        return BOOL

    def _isinstance_results(self, st: State, v: V, names: set, classes: set, node) -> set:
        """The truth values ``isinstance(x, <names and classes>)`` takes over the alternatives of ``v``."""
        results: set = set()
        rest = v.replace(refs=frozenset(), st=frozenset())
        for kind in rest.kinds():
            scalar = kind if kind != "None" else None
            results.add("object" in names or any(scalar in self._ISINSTANCE_SCALARS[n]
                                                 for n in names & self._ISINSTANCE_SCALARS.keys()))
        if v.st:
            # a function, a builtin, a module or a class object: an instance of ``object`` and of nothing else here
            results.add("object" in names)
        for a in v.refs:
            ob = st.heap[a]
            if ob.kind == "inst":
                if "__class__" in ob.attrs:
                    raise Unsupported("isinstance of an object with a __class__", node)
                member = "object" in names
                c = ob.cls
                while c is not None:
                    if "__class__" in st.heap[c].attrs:
                        raise Unsupported("isinstance of an object whose class defines __class__", node)
                    member = member or c in classes
                    bases = st.heap[c].bases
                    c = bases[0] if bases else None
                results.add(member)
            else:
                results.add("object" in names or {"list": "list", "tuple": "tuple"}.get(ob.kind) in names)
        return results

    # ------------------------------------------------------ list methods
    def list_method(self, st: State, addrs, name, args, kwargs, node) -> V:
        """``addrs.name(*args)`` on a list: a method with a rule is read through it (``_METHOD_HOOKS``); the model below is the visit."""
        qualname = f"list.{name}"
        if qualname in _METHOD_HOOKS:
            return self.visit_method_rule(qualname, MethodCtx(st, frozenset(addrs), name, args, kwargs, node))
        return self._list_method_model(st, addrs, name, args, kwargs, node)

    def _list_method_model(self, st: State, addrs, name, args, kwargs, node) -> V:
        if name == "sort":
            return self._list_sort(st, frozenset(addrs), args, kwargs, node)
        if kwargs:
            raise Unsupported(f"list.{name} with keywords", node)
        addrs = frozenset(addrs)
        z = st.zone
        if name == "append":
            if not self._arity(node, args, kwargs, 1, 1, "append"):
                st.set_bottom()
                return BOT
            v = args[0]

            def app(ob, strong):
                # the rest segment [P, len) is empty: the new last element is all of it
                if (strong and getattr(ob, "pre", None) is not None and isinstance(ob.length, str)
                        and st.zone.entails(ob.length, seg_var(ob.length), 0)):
                    nob = ob.replace(elem=self.summarize(st, v))
                else:
                    nob = ob.replace(elem=self.join_into(st, ob.elem, v))
                return self.set_len(st, nob, strong, *self._len_plus(st, ob, 1))

            self.update_obj(st, addrs, app)
            return NONE
        if name == "insert":
            if not self._arity(node, args, kwargs, 2, 2, "insert"):
                st.set_bottom()
                return BOT
            self._num_arg(st, args[0], node, "insert index", allow_float=False)
            v = args[1]
            self.collapse(st, addrs)
            self.update_obj(st, addrs, lambda ob, s: self.set_len(
                st, ob.replace(elem=self.join_into(st, ob.elem, v)), s,
                *self._len_plus(st, ob, 1)))
            return NONE
        if name == "pop":
            if not self._arity(node, args, kwargs, 0, 1, "pop"):
                st.set_bottom()
                return BOT
            out = BOT
            for a in addrs:
                ob = st.heap[a]
                llin, liv = self.len_lin(st, ob)
                if args:
                    self._num_arg(st, args[0], node, "pop index", allow_float=False)
                    ii = self._as_int(st, args[0])
                    if ii is None:
                        st.set_bottom()
                        return BOT
                    ok, fact = self.index_ok(st, llin, liv, ii)
                else:
                    ok = False
                    if llin is not None:
                        lo, _ = lin_bounds(llin, z)
                        ok = lo is not None and lo >= 1
                    elif liv[0] is not None:
                        ok = liv[0] >= 1
                    fact = f"pop from a list of length {self.plin(llin)} >= 1" if ok else "list may be empty"
                obj_node = node.func.value if isinstance(getattr(node, "func", None), ast.Attribute) else None
                res = None
                if obj_node is not None:
                    o = ast.unparse(obj_node)
                    if args and node.args:
                        i = ast.unparse(node.args[0])
                        res = f"not (-len({o}) <= ({i}) < len({o}))"
                    else:
                        res = f"len({o}) == 0"
                self.check(node, "pop", {"IndexError"}, ok, fact, residual=res, operand=obj_node, st=st)
                out = vjoin(out, z, self.all_elems(st, ob), z)
            if out.is_bot():
                st.set_bottom()
                return BOT
            out = self.mat(st, out)
            # the non-raising path: the list had an element.  Collapse: pop(i) shifts
            # indices, and pop() could leave len < P, after which an append would land
            # below P while being recorded as a rest element.
            self.collapse(st, addrs)
            self.update_obj(st, addrs, lambda ob, s: self.set_len(st, ob, s, *self._len_plus(st, ob, -1)))
            return out
        if name == "reverse":
            self._arity(node, args, kwargs, 0, 0, "reverse")
            self.collapse(st, addrs)
            return NONE
        if name == "extend":
            if not self._arity(node, args, kwargs, 1, 1, "extend"):
                st.set_bottom()
                return BOT
            self._list_extend(st, addrs, args[0], node)
            return NONE
        if name == "copy":
            if not self._arity(node, args, kwargs, 0, 0, "copy"):
                st.set_bottom()
                return BOT
            if len(addrs) != 1:
                raise Unsupported("list.copy of several possible lists", node)
            ob = st.heap[next(iter(addrs))]
            lin, iv = self.len_lin(st, ob)
            na = self.new_list(st, node, IntInfo(lin) if lin is not None else IntInfo.interval(*iv), self.all_elems(st, ob))
            return V(refs=frozenset({na}))
        if name == "clear":
            if not self._arity(node, args, kwargs, 0, 0, "clear"):
                st.set_bottom()
                return BOT
            self.collapse(st, addrs)
            self.update_obj(st, addrs, lambda ob, s: self.set_len(st, ob.replace(elem=BOT), s, Lin.const(0)))
            return NONE
        if name in ("count", "index", "remove"):
            return self._list_search(st, addrs, name, args, kwargs, node)
        raise Unsupported(f"list.{name}", node)

    def _absent_search_method(self, st: State, ob, name, args) -> bool:
        return ((ob.kind == "list" and name in ("index", "remove") or
                 ob.kind == "tuple" and name == "index") and
                (self.len_lin(st, ob)[1][1] == 0 or bool(args) and self._search_absent(
                    st, ob, args[0], args[1:] if name == "index" else ())))

    def _try_routes_builtin_raise(self, st: State, ob, name, args, kwargs) -> bool:
        return (self._absent_search_method(st, ob, name, args) or
                ob.kind == "list" and name == "pop" and not args and not kwargs and
                self.len_lin(st, ob)[1][1] == 0)

    def _search_absent(self, st: State, ob, x: V, extra=()) -> bool:
        length_lo, length = self.len_lin(st, ob)[1]
        if length == 0:
            return True
        elem = self.all_elems(st, ob)
        # shortcut: empty index slices need exact length and bounds; add interval clipping when a task needs it.
        if (1 <= len(extra) <= 2 and length_lo is not None and length_lo == length
                and all(not (v.top or v.refs or v.st or v.f) for v in (elem, x))
                and all(exact_int_value(v) for v in extra)):
            bounds = [self.ibounds(st, self._as_int(st, v)) for v in extra]
            if all(lo is not None and lo == hi for lo, hi in bounds):
                clipped = [min(length, max(0, lo + length if lo < 0 else lo)) for lo, _ in bounds]
                if clipped[0] >= (clipped[1] if len(clipped) == 2 else length):
                    return True
        if self._decide_eq(st, elem, x) is False:
            return True
        if all(v.replace(i=None, b=None).is_bot() for v in (elem, x)):
            left, right = self._as_int(st, elem), self._as_int(st, x)
            if left is not None and right is not None:
                llo, lhi = self.ibounds(st, left)
                rlo, rhi = self.ibounds(st, right)
                return ((lhi is not None and rlo is not None and lhi < rlo) or
                        (rhi is not None and llo is not None and rhi < llo))
        # shortcut: only primitive constants, int/bool intervals and finite str sets; extend when a needed absence needs more facts.
        return (all(v.s is not None and v.s != ANY_STR and v.replace(s=None).is_bot() for v in (elem, x))
                and elem.s.isdisjoint(x.s))

    def _end_absent_search(self, st: State, node) -> V:
        if self._try_stack:
            self.check(node, "search", {"ValueError"}, False, "search interval has no matching element", st=st)
        st.set_bottom()
        return BOT

    def _search_hi(self, st: State, addrs, x: V, node, extra=()):
        """For ``count`` / ``index`` / ``remove`` of ``x`` over the sequences ``addrs`` (list or tuple heap objects): the largest length
        any of them may have (``None``: unbounded) and whether ``x`` is definitely absent from all of them.  An operand or element whose ``==`` runs user code,
        and an unknown value, refuse; ``extra`` are the ``start`` / ``stop`` bounds of ``index`` (ints only)."""
        if x.top:
            raise Unsupported("unknown argument", node)
        for v in extra:
            self._num_arg(st, v, node, "index bound", allow_float=False)
        hi, all_absent = 0, True
        for a in addrs:
            ob = st.heap[a]
            llin, liv = self.len_lin(st, ob)
            if self.custom_comparison(st, [x, self.all_elems(st, ob)]):
                raise Unsupported("list/tuple count, index or remove invoking a user comparison", node)
            h = liv[1]
            hi = None if (hi is None or h is None) else max(hi, h)
            all_absent = all_absent and self._search_absent(st, ob, x, extra)
        return hi, all_absent

    def _list_search(self, st: State, addrs, name, args, kwargs, node) -> V:
        """``count`` (an int in ``[0, len]``), ``index`` (an int in ``[0, len - 1]``; absent is a ``ValueError``, no property class, so the
        path ends) and ``remove`` (the length drops by one on the path that did not raise)."""
        lo_n, hi_n = {"count": (1, 1), "index": (1, 3), "remove": (1, 1)}[name]
        if not self._arity(node, args, kwargs, lo_n, hi_n, name):
            st.set_bottom()
            return BOT
        hi, all_absent = self._search_hi(st, addrs, args[0], node, args[1:])
        if name == "count":
            return int_iv(0, hi)
        if all_absent:
            return self._end_absent_search(st, node)
        if name == "index":
            return int_iv(0, None if hi is None else hi - 1)
        self.collapse(st, addrs)
        self.update_obj(st, addrs, lambda ob, s: self.set_len(st, ob, s, *self._len_plus(st, ob, -1)))
        return NONE

    def _list_sort(self, st: State, addrs, args, kwargs, node) -> V:
        """``list.sort()``: the elements stay (a permutation of them), the order is unknown (segmentation collapses); only numbers and
        strs that compare (``_elem_sortable``) and the keywords ``key=None`` and ``reverse=<bool>`` are modelled."""
        if args or any(k not in ("key", "reverse") for k in kwargs):
            self.check(node, "call-args", {"TypeError"}, False, "list.sort() takes no positional arguments and key/reverse keywords")
            st.set_bottom()
            return BOT
        self.check(node, "call-args", {"TypeError"}, True, "list.sort() arity")
        key = kwargs.get("key")
        if key is not None and not (key.n and key.i is None and not key.b and not key.f and key.s is None and not key.refs
                                    and not key.st and not key.top):
            raise Unsupported("list.sort with a key function", node)
        rev = kwargs.get("reverse")
        if rev is not None and (rev.top or rev.refs or rev.st or rev.s is not None or rev.f or rev.n):
            raise Unsupported("list.sort with a reverse that is not a bool or an int", node)
        for a in addrs:
            if not self._elem_sortable(self.all_elems(st, st.heap[a])):
                raise Unsupported("list.sort of a mixed or unorderable element type", node)
        self.collapse(st, addrs)
        return NONE

    def tuple_method(self, st: State, a, name, args, kwargs, node) -> V:
        """``count`` and ``index`` of a tuple (read-only: the length of a tuple is fixed)."""
        qualname = f"tuple.{name}"
        if qualname in _METHOD_HOOKS:
            return self.visit_method_rule(qualname, MethodCtx(st, frozenset({a}), name, args, kwargs, node))
        raise Unsupported(f"method {name} of a tuple", node)

    def _tuple_method_model(self, st: State, a, name, args, kwargs, node) -> V:
        lo_n, hi_n = {"count": (1, 1), "index": (1, 3)}[name]
        if not self._arity(node, args, kwargs, lo_n, hi_n, name):
            st.set_bottom()
            return BOT
        hi, all_absent = self._search_hi(st, frozenset({a}), args[0], node, args[1:])
        if name == "count":
            return int_iv(0, hi)
        if all_absent:
            return self._end_absent_search(st, node)
        return int_iv(0, None if hi is None else hi - 1)

    def _len_plus(self, st, ob, k):
        if isinstance(ob.length, str):
            return (Lin.var(ob.length, k),)
        lo, hi = ob.length
        return (None, None if lo is None else max(0, lo + k), None if hi is None else max(0, hi + k))

    def _list_extend(self, st, addrs, v: V, node, consumer="list.extend") -> V:
        v = self.iterable(st, v, node, consumer=consumer)
        if v.top:
            raise Unsupported("extend with an unknown value", node)
        self.check_iterable(st, v, node)
        src_elem = BOT
        add = None
        if v.s is not None:
            src_elem = ANYSTR
        for b in v.refs:
            ob2 = st.heap[b]
            if ob2.kind in ("list", "tuple", "array"):
                e2 = self.all_elems(st, ob2)
                src_elem = vjoin(src_elem, st.zone, e2, st.zone)
                if len(v.refs) == 1 and v.s is None:
                    add = self.len_lin(st, ob2)[0]
            elif ob2.kind in ("range", "iterator", "dict", "dictview", "set"):
                ev = self.elem_summary(st, V(refs=frozenset({b})))
                if ev is not None:
                    src_elem = vjoin(src_elem, st.zone, ev, st.zone)

        def upd(ob, strong):
            nob = ob.replace(elem=self.join_into(st, ob.elem, src_elem)) if not src_elem.is_bot() else ob
            if add is not None and add.is_const() and isinstance(ob.length, str):
                return self.set_len(st, nob, strong, Lin.var(ob.length, add.c))
            llin, liv = self.len_lin(st, ob)
            return self.set_len(st, nob, strong, None, liv[0], None)

        self.update_obj(st, frozenset(addrs), upd)
        return V(refs=frozenset(addrs))

    # ------------------------------------------------------ [internal ref]: array (docs/ABSINT.md
    # Slice 3). Reuses list's own heap machinery (ListObj/ArrayObj share length/elem/pre;
    # len_lin/index_ok/elem_at/store_elem/set_len/update_obj are all kind-agnostic) for
    # everything a list already gets for free (indexing, slicing, len, iteration); adds
    # only the typecode-gated element check list has no equivalent of.
    def _array_typecode(self, v: V, node) -> str:
        """The one-character typecode string ``v`` must be, or ``Unsupported`` -- a
        program-variable or ambiguous typecode is refused outright (never guessed)."""
        if v.top:
            raise Unsupported("array() of an unknown typecode", node)
        if v.i is not None or v.b or v.f or v.n or v.refs or v.st:
            raise Unsupported("array() typecode must be a string", node)
        if v.s is None or v.s == ANY_STR:
            raise Unsupported("array() typecode must be a known string constant", node)
        if len(v.s) != 1:
            raise Unsupported("array() of several possible typecodes", node)
        tc = next(iter(v.s))
        if len(tc) != 1 or tc not in _ARRAY_TYPECODES:
            raise Unsupported(f"array() typecode {tc!r} not modelled", node)
        return tc

    def _array_store_value(self, st: State, node, typecode: str, v: V) -> V:
        """Validate a value being written into a ``typecode`` array element
        (construction, append, extend, index/slice-repeat store) and return the value
        the element actually holds on the NON-raising path -- narrowed to what real
        ``array`` guarantees a successful store leaves behind (never ``set_bottom`` on
        an unrecognized shape: MG's PR 89 lesson, docs/ABSINT.md)."""
        if v.top:
            raise Unsupported("array element from an unknown value", node)
        is_float_code = typecode in _ARRAY_FLOAT_CODES
        bad_type = bool(v.s is not None or v.n or v.refs or v.st or (not is_float_code and v.f))
        self.check(node, "array:TypeError", {"TypeError"}, not bad_type,
                   f"array element must be {'a number' if is_float_code else 'an int'}, not "
                   + ",".join(v.kinds()))
        if is_float_code:
            return self._array_float_value(st, node, typecode, v)
        return self._array_int_value(st, node, typecode, v)

    def _array_exact_float(self, st: State, v: V):
        """``v``'s exact float value (a Python float), or ``None`` if it isn't one exactly
        known constant (an interval, a NaN-only fact, or an unresolved int)."""
        if v.f and v.f.fact.has_interval() and v.f.fact.lo == v.f.fact.hi and not v.f.fact.nan:
            return float(v.f.fact.lo)
        ii = self._as_int(st, v)
        if ii is not None:
            lo, hi = self.ibounds(st, ii)
            if lo is not None and lo == hi:
                return float(lo)
        return None

    def _array_float_value(self, st: State, node, typecode: str, v: V) -> V:
        exact = self._array_exact_float(st, v)
        if typecode == "d":
            # a C double IS a Python float: no rounding, no overflow, ever.
            if exact is not None:
                return V(f=FloatInfo(fv.FloatFact.const(exact)))
            return v if v.f else FLOAT
        # 'f': a C float is NARROWER than a Python double -- every finite value is
        # rounded to the nearest representable float32 on store (measured:
        # `array.array('f', [0.1])[0] == 0.10000000149011612`, never `0.1` exactly).
        # CPython's array module SATURATES an out-of-range magnitude to +-inf rather
        # than raising OverflowError (measured -- see _ARRAY_FLOAT_CODES above), so
        # struct's own OverflowError is not real array behaviour and must not be used
        # as a check here; struct IS used for the rounding itself, which matches
        # exactly (same IEEE-754 binary32 conversion CPython's arraymodule.c performs).
        if exact is not None:
            if math.isnan(exact):
                return V(f=FloatInfo(fv.FloatFact.const(float("nan"))))
            if abs(exact) > _FLOAT32_MAX:
                rounded = math.copysign(math.inf, exact)
            else:
                rounded = struct.unpack("<f", struct.pack("<f", exact))[0]
            return V(f=FloatInfo(fv.FloatFact.const(rounded)))
        return v if v.f else FLOAT

    def _array_int_value(self, st: State, node, typecode: str, v: V) -> V:
        lo_tc, hi_tc = _ARRAY_INT_RANGES[typecode]
        ii = self._as_int(st, v)
        if ii is None:
            self.check(node, "array:OverflowError", {"OverflowError"}, False,
                       f"value not known to be an int (typecode {typecode!r} range [{lo_tc},{hi_tc}])")
            return int_iv(lo_tc, hi_tc)
        lo, hi = self.ibounds(st, ii)
        fits = lo is not None and hi is not None and lo >= lo_tc and hi <= hi_tc
        self.check(node, "array:OverflowError", {"OverflowError"}, fits,
                   f"[{lo},{hi}] within typecode {typecode!r} range [{lo_tc},{hi_tc}]" if fits else
                   f"[{lo},{hi}] may be outside typecode {typecode!r} range [{lo_tc},{hi_tc}]")
        # a non-raising store's element is REAL: the meet of what it was known to be
        # and what the typecode guarantees -- sound (never widens), and gives every
        # LITERAL-range-checked store (`array('B', [0])`, W49's raytrace `Canvas`
        # clamp `max(0, min(255, ...))`) an exact/tight post-store fact instead of
        # falling back to the typecode's own full range on every store.
        new_lo = lo_tc if lo is None else max(lo, lo_tc)
        new_hi = hi_tc if hi is None else min(hi, hi_tc)
        if new_lo > new_hi:
            new_lo, new_hi = lo_tc, hi_tc
        return int_iv(new_lo, new_hi)

    def m_array_array(self, st, args, kwargs, node):
        if not self._arity(node, args, kwargs, 1, 2, "array"):
            st.set_bottom()
            return BOT
        typecode = self._array_typecode(args[0], node)
        if len(args) == 1:
            a = self.new_list(st, node, IntInfo.interval(0, 0), BOT, tag="array", kind="array",
                              typecode=typecode)
            return V(refs=frozenset({a}))
        init = args[1]
        if init.top:
            raise Unsupported("array() of an unknown initializer", node)
        init = self.iterable(st, init, node, consumer="array.array")
        self.check_iterable(st, init, node)
        if len(init.refs) != 1 or init.s is not None:
            raise Unsupported("array() of several possible initializers", node)
        src = next(iter(init.refs))
        ob = st.heap[src]
        if ob.kind not in ("list", "tuple", "array"):
            raise Unsupported(f"array() of a {ob.kind}", node)
        if ob.kind == "array" and ob.typecode != typecode:
            raise Unsupported("array() from an array of a different typecode", node)
        lin, iv = self.len_lin(st, ob)
        raw_elem = self.all_elems(st, ob)
        elem = self._array_store_value(st, node, typecode, raw_elem) if not raw_elem.is_bot() else BOT
        a = self.new_list(st, node, IntInfo(lin) if lin is not None else IntInfo.interval(*iv),
                          elem, tag="array", kind="array", typecode=typecode)
        return V(refs=frozenset({a}))

    def array_method(self, st: State, addrs, name, args, kwargs, node) -> V:
        if kwargs:
            raise Unsupported(f"array.{name} with keywords", node)
        addrs = frozenset(addrs)
        typecodes = {st.heap[a].typecode for a in addrs}
        if len(typecodes) != 1:
            raise Unsupported("array method on a summary spanning several typecodes", node)
        typecode = next(iter(typecodes))
        if name == "append":
            if not self._arity(node, args, kwargs, 1, 1, "append"):
                st.set_bottom()
                return BOT
            ev = self._array_store_value(st, node, typecode, args[0])

            def app(ob, strong):
                return self.set_len(st, ob.replace(elem=self.join_into(st, ob.elem, ev)), strong,
                                    *self._len_plus(st, ob, 1))

            self.update_obj(st, addrs, app)
            return NONE
        if name == "extend":
            if not self._arity(node, args, kwargs, 1, 1, "extend"):
                st.set_bottom()
                return BOT
            self._array_extend(st, addrs, typecode, args[0], node)
            return NONE
        raise Unsupported(f"array.{name}", node)

    def _array_extend(self, st: State, addrs, typecode: str, v: V, node) -> V:
        v = self.iterable(st, v, node, consumer="array.array.extend")
        if v.top:
            raise Unsupported("extend with an unknown value", node)
        self.check_iterable(st, v, node)
        if v.s is not None:
            raise Unsupported("array.extend from a string", node)
        src_elem = BOT
        add = None
        for b in v.refs:
            ob2 = st.heap[b]
            if ob2.kind in ("list", "tuple", "array"):
                if ob2.kind == "array" and ob2.typecode != typecode:
                    raise Unsupported("array.extend from an array of a different typecode", node)
                e2 = self.all_elems(st, ob2)
                src_elem = vjoin(src_elem, st.zone, e2, st.zone)
                if len(v.refs) == 1:
                    add = self.len_lin(st, ob2)[0]
            elif ob2.kind in ("range", "iterator"):
                ev = self.elem_summary(st, V(refs=frozenset({b})))
                if ev is not None:
                    src_elem = vjoin(src_elem, st.zone, ev, st.zone)
            else:
                raise Unsupported(f"array.extend from a {ob2.kind}", node)
        checked_elem = self._array_store_value(st, node, typecode, src_elem) if not src_elem.is_bot() else BOT

        def upd(ob, strong):
            nob = ob.replace(elem=self.join_into(st, ob.elem, checked_elem)) if not checked_elem.is_bot() else ob
            if add is not None and add.is_const() and isinstance(ob.length, str):
                return self.set_len(st, nob, strong, Lin.var(ob.length, add.c))
            llin, liv = self.len_lin(st, ob)
            return self.set_len(st, nob, strong, None, liv[0], None)

        self.update_obj(st, frozenset(addrs), upd)
        return V(refs=frozenset(addrs))

    # ------------------------------------------------------ module functions

    # -------------------------------------------------------------- os / os.path ([internal ref])
    def _os_arity_ok(self, args, kwargs, lo: int, hi: int, allowed_kw: frozenset = frozenset()) -> bool:
        """MG's PR 89 review: a call shape this coarse pass
        does not recognize is UNSUPPORTED, never a certain raise (``set_bottom``) -- a
        keyword this pass has simply not implemented yet (e.g. ``os.getenv(key,
        default=...)``, valid in real Python) is not the same fact as "every concrete call
        of this shape raises TypeError"."""
        unknown_kw = set(kwargs) - allowed_kw
        return lo <= len(args) <= hi and not unknown_kw

    def _os_path_arg(self, v: V, node, what: str) -> str | None:
        """Mirrors ``_re_str_arg``: ``Unsupported`` for a definitely-unknown-typed (``top``)
        argument (an `os.path.*` call this pass cannot even attempt), a checked TypeError
        for a definitely-non-str one (real `os.path.join`/etc. raise TypeError for anything
        else), and the argument's own single known str value when there is exactly one --
        letting the caller compute an EXACT result -- else ``None`` (the caller then falls
        back to a generic, still-sound, havoc string)."""
        ok = self._os_arg_type_ok(v, node, what, allow_str=True)
        if ok and v.s is not None and v.s != ANY_STR and len(v.s) == 1:
            return next(iter(v.s))
        return None

    def _os_is_str(self, v: V) -> bool:
        return v.s is not None and not (v.i is not None or v.b or v.f or v.n or v.refs or v.st)

    def _os_is_int(self, v: V) -> bool:
        # bool counts as int (real Python accepts one wherever the other is expected, e.g.
        # `os.urandom(True)`); this pass tracks them as separate kinds, so both are checked.
        return (v.i is not None or v.b) and not (v.s is not None or v.f or v.n or v.refs or v.st)

    def _os_is_none(self, v: V) -> bool:
        return v.n and not (v.i is not None or v.b or v.f or v.s is not None or v.refs or v.st)

    def _os_arg_type_ok(self, v: V, node, what: str, *, allow_str=False, allow_int=False,
                        allow_none=False) -> bool:
        """MG's PR 106 review: the HAVOC handlers below checked arity and ``top`` but never
        the argument's TYPE, so a definitely-wrong-typed argument (``os.getenv(5)``,
        ``os.urandom('x')``, ``os.path.isfile(None)`` -- all real, native TypeErrors) sailed
        through to a havoc RESULT instead of a checked violation, a wrong TRUE. Mirrors
        ``_re_str_arg``: ``Unsupported`` for a definitely-unknown-typed (``top``) argument
        (this pass cannot even attempt the call), a checked TypeError otherwise -- ``bool``
        counts as ``int`` (see ``_os_is_int``)."""
        if v.top:
            raise Unsupported(f"{what}: argument of unknown type", node)
        ok = ((allow_str and self._os_is_str(v)) or (allow_int and self._os_is_int(v))
              or (allow_none and self._os_is_none(v)))
        self.check(node, "arg-type", {"TypeError"}, ok, f"{what}: " + ",".join(v.kinds()))
        return ok

    def _os_str_result(self, s: str) -> V:
        return const_str(s) if len(s) <= 64 else ANYSTR

    def m_os_path_join(self, st, a, k, n):
        if not self._os_arity_ok(a, k, 1, 32):
            raise Unsupported("os.path.join: unmodelled arity/keywords", n)
        # Every argument is checked (not short-circuited on the first imprecise one), same
        # as m_re_sub checks pattern/repl/string independently -- a second bad argument
        # still gets its own recorded check.
        parts = [self._os_path_arg(v, n, "join") for v in a]
        if any(s is None for s in parts):
            return ANYSTR
        return self._os_str_result(os.path.join(*parts))

    def _os_path_unary(self, st, a, k, n, name, fn, only_if_absolute=False):
        if not self._os_arity_ok(a, k, 1, 1):
            raise Unsupported(f"os.path.{name}: unmodelled arity/keywords", n)
        s = self._os_path_arg(a[0], n, name)
        if s is not None and (not only_if_absolute or s.startswith("/")):
            # abspath of an ALREADY-ABSOLUTE literal never consults the real cwd -- exact
            # (condition (a)'s own precedent, extern.py::_is_dirname_abspath_file: `__file__`
            # is always such a literal; posixpath.abspath itself is `normpath(path)` for an
            # absolute path, no syscall at all). A RELATIVE literal's abspath genuinely
            # depends on the replay host's cwd, which this pass does not, and must not,
            # guess -- havoc, same as an argument this pass cannot see the value of.
            return self._os_str_result(fn(s))
        return ANYSTR

    def m_os_path_dirname(self, st, a, k, n):
        return self._os_path_unary(st, a, k, n, "dirname", os.path.dirname)

    def m_os_path_basename(self, st, a, k, n):
        return self._os_path_unary(st, a, k, n, "basename", os.path.basename)

    def m_os_path_normpath(self, st, a, k, n):
        return self._os_path_unary(st, a, k, n, "normpath", os.path.normpath)

    def m_os_path_abspath(self, st, a, k, n):
        return self._os_path_unary(st, a, k, n, "abspath", os.path.abspath, only_if_absolute=True)

    def m_os_path_realpath(self, st, a, k, n):
        # [internal ref] (MG's PR 106 review): UNLIKE abspath, realpath dereferences symlinks against
        # the REAL, LIVE filesystem (posixpath.realpath calls os.readlink in a loop) -- its
        # result for the exact same string argument can differ from host to host, or hour to
        # hour (MG's o06 attack: os.path.realpath("/var/run") == "/run" on [host], a real
        # symlink, but the literal string on another host/moment). Never computed exactly,
        # even for an absolute literal -- always a havoc string, the same bucket as
        # exists/isfile/isdir below, not join/dirname/.../abspath above.
        if not self._os_arity_ok(a, k, 1, 1):
            raise Unsupported("os.path.realpath: unmodelled arity/keywords", n)
        self._os_path_arg(a[0], n, "realpath")  # still records the str-type check
        return ANYSTR

    def m_os_path_splitext(self, st, a, k, n):
        if not self._os_arity_ok(a, k, 1, 1):
            raise Unsupported("os.path.splitext: unmodelled arity/keywords", n)
        s = self._os_path_arg(a[0], n, "splitext")
        if s is not None:
            root, ext = os.path.splitext(s)
            items = [self._os_str_result(root), self._os_str_result(ext)]
        else:
            items = [ANYSTR, ANYSTR]
        addr = self.new_list(st, n, None, BOT, tag="tuple", kind="tuple", items=items)
        return V(refs=frozenset({addr}))

    def _os_path_bool_query(self, st, a, k, n, name):
        # [internal ref] category (b): exists/isfile/isdir genuinely query the filesystem -- the
        # answer can differ between the analysis host and the replay host (or between two
        # runs on the SAME host), so it is never computed, even for a known literal path --
        # unlike the PURE os.path.* functions above, always a havoc bool, never exact. The
        # path argument is str OR int (a file descriptor, real Python accepts one) -- None
        # or a float genuinely raise TypeError (MG's PR 106 review, o05_exists_none.py).
        if not self._os_arity_ok(a, k, 1, 1):
            raise Unsupported(f"os.path.{name}: unmodelled arity/keywords", n)
        self._os_arg_type_ok(a[0], n, f"os.path.{name}", allow_str=True, allow_int=True)
        return BOOL

    def m_os_path_exists(self, st, a, k, n):
        return self._os_path_bool_query(st, a, k, n, "exists")

    def m_os_path_isfile(self, st, a, k, n):
        return self._os_path_bool_query(st, a, k, n, "isfile")

    def m_os_path_isdir(self, st, a, k, n):
        return self._os_path_bool_query(st, a, k, n, "isdir")

    def m_os_getenv(self, st, a, k, n):
        # [internal ref] category (b): an environment variable's value cannot be known statically --
        # str-or-None with no default, str-or-``default``'s own value with one (real Python
        # returns `default` UNCHANGED, not coerced to str, when the key is unset). The key
        # itself must be str (MG's PR 106 review: os.getenv(5) natively raises TypeError).
        if not self._os_arity_ok(a, k, 1, 2):
            raise Unsupported("os.getenv: unmodelled arity/keywords", n)
        self._os_arg_type_ok(a[0], n, "os.getenv", allow_str=True)
        if len(a) == 2:
            return vjoin(ANYSTR, st.zone, a[1], st.zone)
        return V(s=ANY_STR, n=True)

    def m_os_getcwd(self, st, a, k, n):
        if not self._os_arity_ok(a, k, 0, 0):
            raise Unsupported("os.getcwd: unmodelled arity/keywords", n)
        return ANYSTR

    def m_os_listdir(self, st, a, k, n):
        # real os.listdir accepts str, bytes, None (the cwd) or an int file descriptor;
        # this pass models str/None/int (MG's PR 106 review) -- bytes has no abstract value
        # here (a bytes literal is itself Unsupported at e_Constant, so it never reaches a
        # KNOWN-bytes value; an unknown-typed one still refuses via `top`, below).
        if not self._os_arity_ok(a, k, 0, 1):
            raise Unsupported("os.listdir: unmodelled arity/keywords", n)
        if a:
            self._os_arg_type_ok(a[0], n, "os.listdir", allow_str=True, allow_int=True, allow_none=True)
        addr = self.new_list(st, n, IntInfo.interval(0, None), ANYSTR)
        return V(refs=frozenset({addr}))

    def m_os_urandom(self, st, a, k, n):
        # MG's PR 106 review: os.urandom('x') natively raises TypeError (n must be int).
        if not self._os_arity_ok(a, k, 1, 1):
            raise Unsupported("os.urandom: unmodelled arity/keywords", n)
        self._os_arg_type_ok(a[0], n, "os.urandom", allow_int=True)
        return V(top=True)  # bytes have no abstract-value model in this pass (yet)

    def m_os_getpid(self, st, a, k, n):
        if not self._os_arity_ok(a, k, 0, 0):
            raise Unsupported("os.getpid: unmodelled arity/keywords", n)
        return int_iv(1, None)

    def m_os_cpu_count(self, st, a, k, n):
        if not self._os_arity_ok(a, k, 0, 0):
            raise Unsupported("os.cpu_count: unmodelled arity/keywords", n)
        return vjoin(int_iv(1, None), st.zone, NONE, st.zone)

    def m__sv_verifier_nondet_int(self, st, a, k, n):
        self._in_try_refuse("a harness draw (its refusal of an invalid draw is a ValueError)", n)
        self._arity(n, a, k, 0, 0, "nondet_int")
        return int_iv(None, None)

    def m__sv_verifier_nondet_bool(self, st, a, k, n):
        self._in_try_refuse("a harness draw (its refusal of an invalid draw is a ValueError)", n)
        self._arity(n, a, k, 0, 0, "nondet_bool")
        # a case split's outcome: used by the FIRST evaluation of that node, in the activation it was forced in
        pending = self._forced_draws.get(n)
        if pending is not None and pending[1] == self._activation(st):
            del self._forced_draws[n]
            return const_bool(pending[0])
        return BOOL

    def m__sv_verifier_nondet_float(self, st, a, k, n):
        self._in_try_refuse("a harness draw (its refusal of an invalid draw is a ValueError)", n)
        self._arity(n, a, k, 0, 0, "nondet_float")
        return FLOAT

    def m__sv_verifier_nondet_str(self, st, a, k, n):
        self._in_try_refuse("a harness draw (its refusal of an invalid draw is a ValueError)", n)
        self._arity(n, a, k, 0, 0, "nondet_str")
        return ANYSTR

    def _nondet_scalar_summary(self, st, callback: V, node, label: str) -> V:
        """Join only exact zero-argument scalar generators supplied by the verifier stub."""
        if callback.top or callback.refs or callback.i is not None or callback.b or callback.f or callback.s is not None or callback.n:
            raise Unsupported(f"{label} is not an exact verifier scalar callback", node)
        if not callback.st:
            raise Unsupported(f"{label} is not an exact verifier scalar callback", node)
        summaries = []
        for tag in callback.st:
            if len(tag) != 3 or tag[:2] != ("mfunc", "_sv_verifier"):
                raise Unsupported(f"{label} is not an exact verifier scalar callback", node)
            name = tag[2]
            if name == "nondet_int":
                summaries.append(int_iv(None, None))
            elif name == "nondet_bool":
                summaries.append(BOOL)
            elif name == "nondet_float":
                summaries.append(FLOAT)
            elif name == "nondet_str":
                summaries.append(ANYSTR)
            else:
                raise Unsupported(f"{label} uses an unsupported verifier generator: {name}", node)
        out = summaries[0]
        for value in summaries[1:]:
            out = vjoin(out, st.zone, value, st.zone)
        return out

    def m__sv_verifier_nondet_list(self, st, args, kwargs, node):
        """A fresh list of any nonnegative length, with independent supported scalar draws."""
        self._in_try_refuse("a harness container draw (callback errors are not routed)", node)
        bound = self._bind_call(st, node, args, kwargs, "_sv_verifier.nondet_list", ("nondet_elem",), ("nondet_elem",))
        if bound is None:
            return BOT
        elem = self._nondet_scalar_summary(st, bound["nondet_elem"], node, "nondet_list callback")
        length = IntInfo(None, (ZERO, 0), None)
        address = self.new_list(st, node, length, elem, tag="nondet_list")
        return V(refs=frozenset({address}))

    def m__sv_verifier_nondet_dict(self, st, args, kwargs, node):
        """A fresh inexact dict; callbacks may draw any keys/values and equal keys overwrite."""
        self._in_try_refuse("a harness container draw (callback/hash errors are not routed)", node)
        bound = self._bind_call(
            st, node, args, kwargs, "_sv_verifier.nondet_dict",
            ("nondet_key", "nondet_value"), ("nondet_key", "nondet_value"),
        )
        if bound is None:
            return BOT
        keys = self._nondet_scalar_summary(st, bound["nondet_key"], node, "nondet_dict key callback")
        values = self._nondet_scalar_summary(st, bound["nondet_value"], node, "nondet_dict value callback")
        address = self.alloc(st, node, lambda _sid: DictObj({}, False, keys, values), "nondet_dict")
        return V(refs=frozenset({address}))

    def _verifier_check_type_value_kind(self, st, value, node):
        """Return a single hook-free builtin kind for the deliberately narrow check_type model."""
        if value.top or value.st or value.is_bot():
            raise Unsupported("check_type value is not a known primitive or exact builtin container", node)
        kinds = []
        if value.i is not None:
            kinds.append("int")
        if value.b:
            kinds.append("bool")
        if value.f:
            kinds.append("float")
        if value.s is not None:
            kinds.append("str")
        if value.n:
            kinds.append("NoneType")
        if value.refs:
            ref_kinds = set()
            for address in value.refs:
                obj = st.heap.get(address)
                if type(obj) not in {ListObj, DictObj}:
                    raise Unsupported("check_type value may be a custom or unsupported object", node)
                ref_kinds.add(obj.kind)
            kinds.extend(sorted(ref_kinds))
        if len(kinds) != 1:
            raise Unsupported("check_type value has mixed or unsupported possible types", node)
        return kinds[0]

    def _check_type_user_class(self, st, value, caddr, node):
        """``check_type(value, C)`` for a class ``C`` of the program: the benchmark's runtime check is
        ``isinstance(value, C)`` and ``C`` has no metaclass here (``s_ClassDef`` refuses one), so it is exactly the
        decision ``isinstance`` already makes (``_isinstance_results``: the single-inheritance chain, a refusal for an
        object carrying its own ``__class__``).  Proven only when EVERY alternative of ``value`` is an instance;
        a possible mismatch formats ``value!r`` into the TypeError, which runs a user ``__repr__`` for an object,
        so only a scalar value may mismatch (an object that may mismatch refuses)."""
        if value.top or value.is_bot():
            raise Unsupported("check_type value is not a known primitive or object", node)
        results = self._isinstance_results(st, value, set(), {caddr}, node)
        if value.refs and results != {True}:
            raise Unsupported("check_type mismatch may invoke a user __repr__", node)
        matches = results == {True}
        self.check(node, "check_type", {"TypeError"}, matches,
                   "isinstance(value, <class of the program>) along its single-inheritance chain", st=st)
        if not matches and results == {False}:
            st.set_bottom()
            return BOT
        return V(n=True)

    def m__sv_verifier_check_type(self, st, args, kwargs, node):
        """Model exact builtin class hints over one known scalar/list/dict value kind, and a class of the program as isinstance."""
        bound = self._bind_call(
            st, node, args, kwargs, "_sv_verifier.check_type", ("value", "hint"), ("value", "hint")
        )
        if bound is None:
            return BOT
        hint = bound["hint"]
        if hint == V(st=frozenset({("check-type-union", "int", "str")})):
            value = bound["value"]
            if value.top or value.st or value.refs or value.is_bot():
                raise Unsupported("check_type union value is not a known builtin scalar", node)
            kinds = set()
            if value.i is not None:
                kinds.add("int")
            if value.b:
                kinds.add("bool")
            if value.s is not None:
                kinds.add("str")
            if value.f or value.n or not kinds:
                raise Unsupported("check_type union value has mixed or unsupported possible types", node)
            self.check(node, "check_type", {"TypeError"}, True,
                       "isinstance(value, int | str) under exact builtin classes", st=st)
            return V(n=True)
        if (not hint.top and hint.i is None and not hint.b and not hint.f and hint.s is None and not hint.n
                and not hint.st and len(hint.refs) == 1):
            caddr = next(iter(hint.refs))
            if st.heap[caddr].kind == "class":
                return self._check_type_user_class(st, bound["value"], caddr, node)
        if (hint.top or hint.i is not None or hint.b or hint.f or hint.s is not None or hint.n
                or hint.refs or len(hint.st) != 1):
            raise Unsupported("check_type hint is not one exact builtin class", node)
        tag = next(iter(hint.st))
        if len(tag) != 2 or tag[0] != "type" or tag[1] not in {"int", "bool", "float", "str", "list", "dict"}:
            raise Unsupported("check_type hint is outside the exact builtin class subset", node)
        value_kind = self._verifier_check_type_value_kind(st, bound["value"], node)
        hint_kind = tag[1]
        matches = value_kind == hint_kind or (hint_kind == "int" and value_kind == "bool")
        if not matches and value_kind in {"list", "dict"}:
            # On mismatch the upstream checker formats value!r in its TypeError;
            # builtin containers can still call a nested user's __repr__ there.
            raise Unsupported("check_type mismatch may invoke a nested container __repr__", node)
        self.check(
            node, "check_type", {"TypeError"}, matches,
            f"isinstance({value_kind}, {hint_kind}) under exact builtin classes", st=st,
        )
        if not matches:
            st.set_bottom()
            return BOT
        return V(n=True)

    def _numeric_fact(self, st: State, v: V):
        """``v`` as one numeric fact (a ``FloatFact``; an int/bool part is placed on the float line
        exactly as ``values.refine_compare`` compares it), or ``None`` when ``v`` is not purely
        numeric (a str, ``None``, an object, ``top``) or its int part is not a bounded interval."""
        if v.top or v.refs or v.st or v.s is not None or v.n:
            return None
        fact = v.f.fact if v.f else None
        ii = self._as_int(st, v)
        if ii is not None:
            lo, hi = self.ibounds(st, ii)
            part = fv._cmp_view(fv.IntFact(lo, hi))
            fact = part if fact is None else fact.join(part)
        return fact

    def m__sv_verifier_nondet_float_range(self, st, a, k, n):
        """``_sv_verifier.nondet_float_range(a, b)``, modelled BOTH ways ([internal ref], [internal ref]; [internal ref]).

        The helper's documentation says "a float in ``[a, b]``", its code is ``random.uniform(a, b)`` = ``a + (b - a) * random()``
        (never ``1.0`` for ``(0.0, 1.0)``; ``inf`` or ``nan`` when ``b - a`` overflows).  Owner ruling (2026-10-06): a TRUE is proven over
        the LARGER reading, a FALSE is confirmed only with a value the CODE can produce (``sv/_child.py``, the harness shim).  So the
        value here is the JOIN of two sets, a superset of each:

        * the documented one, ``v = <any float>; if isnan(v) or not min(a, b) <= v <= max(a, b): invalid`` ([internal ref]: the draws the
          grading shim accepted): the non-NaN floats in ``[min(a, b), max(a, b)]``, bounds inclusive, an infinite bound included,
          ``a > b`` legal.  ``min``/``max`` compare with ``<``/``>``, which are False for NaN, so a NaN bound is not symmetric: ``a`` NaN
          accepts nothing, ``b`` NaN with a number ``a`` accepts exactly ``a``;
        * the code's, ``Analyzer.uniform_formula`` (``rand.py``), the formula over the draw ``[0.0, 1.0]`` with the outward-rounded float
          transfer functions: it adds what the documentation excludes (``inf`` and ``nan`` for a span that overflows, ``nan`` for a
          ``nan`` bound).

        Bounds with no numeric fact give any float, ``nan`` included.  Non-numeric bounds are the TypeError ``min`` raises
        (recorded).  The call binds ``a`` and ``b`` by position or name ([internal ref])."""
        self._in_try_refuse("a harness draw (its refusal of an invalid draw is a ValueError)", n)
        bound = self._bind_call(st, n, a, k, "_sv_verifier.nondet_float_range", ("a", "b"), ("a", "b"))
        if bound is None:
            return BOT
        lo_v, hi_v = bound["a"], bound["b"]
        self._num_arg(st, lo_v, n, "nondet_float_range lo")
        self._num_arg(st, hi_v, n, "nondet_float_range hi")
        fa, fb = self._numeric_fact(st, lo_v), self._numeric_fact(st, hi_v)
        if fa is None or fb is None:
            # unresolved bounds: any float, nan included (the code's domain holds nan for a nan or overflowing span)
            return FLOAT
        pieces = []
        if fa.has_interval() and fb.has_interval():
            pieces.append((min(fa.lo, fb.lo), max(fa.hi, fb.hi)))
        if fa.has_interval() and fb.nan:
            pieces.append((fa.lo, fa.hi))
        documented = BOT
        if pieces:
            documented = V(f=FloatInfo(fv.FloatFact.range(min(p[0] for p in pieces), max(p[1] for p in pieces), nan=False)))
        code = self.uniform_formula(st, lo_v, hi_v, n)
        if st.bottom:
            return BOT
        return vjoin(documented, st.zone, code, st.zone)

    # ------------------------------------------------------------ io.StringIO
    def _io_str_arg(self, st, v: V, node, what: str) -> bool:
        """True only when ``v`` is definitely a plain str -- mirrors ``_re_str_arg``
        exactly (same reasoning: an unknown-type value aborts the proof rather than
        being guessed at; a definitely-non-str value is a recorded, unproven TypeError
        check, never a silent pass)."""
        if v.top:
            raise Unsupported("io: argument of unknown type", node)
        ok = v.s is not None and not (v.i is not None or v.b or v.f or v.n or v.refs or v.st)
        self.check(node, "arg-type", {"TypeError"}, ok, f"{what}: " + ",".join(v.kinds()))
        return ok

    def m_io_StringIO(self, st, a, k, n):
        """``io.StringIO(initial_value='', newline='\\n')`` ([internal ref]). ``newline``
        only controls universal-newline TRANSLATION -- irrelevant to a model that
        already coarsens content to "some string" the moment anything is written, so it
        is accepted without a type check (the same "unmodelled VALUE, not unmodelled
        SHAPE" reasoning ``re.compile``'s ``flags`` already uses). ``initial_value``
        given both positionally and by keyword, more than one positional argument, or
        any other keyword, is an unmodelled call SHAPE (Unsupported), never a guessed
        TypeError. No ``initial_value`` at all is the one case precise enough to keep
        exact: a fresh ``io.StringIO()`` starts as the literal empty string, not yet
        coarsened -- real code that checks emptiness right after construction (before
        any write) still gets a genuine fact."""
        unknown_kw = set(k) - {"initial_value", "newline"}
        if unknown_kw or len(a) > 1 or (a and "initial_value" in k):
            raise Unsupported("io.StringIO: unmodelled arity/keywords", n)
        initial = a[0] if a else k.get("initial_value")
        if initial is None:
            content = const_str("")
        else:
            self._io_str_arg(st, initial, n, "StringIO initial_value")
            content = ANYSTR
        addr = self.alloc(st, n, lambda sid: StringIOObj(content), "stringio")
        return V(refs=frozenset({addr}))

    def stringio_method(self, st: State, addr, name, args, kwargs, node) -> V:
        qualname = f"io.StringIO.{name}"
        if qualname in _METHOD_HOOKS:
            return self.visit_method_rule(qualname, MethodCtx(st, frozenset({addr}), name, args, kwargs, node))
        return self._stringio_method_model(st, addr, name, args, kwargs, node)

    def _stringio_method_model(self, st: State, addr, name, args, kwargs, node) -> V:
        """Methods on an ``io.StringIO`` buffer ([internal ref]). ``write`` takes exactly
        one str argument and coarsens ``content`` to ``ANYSTR`` (real ``write`` can start
        mid-buffer after a ``seek``, so even an "append" reading of it would be a
        precision claim this pass cannot back -- coarsening is the sound choice).
        ``getvalue``/``read`` both just return ``content`` (position is not tracked, so
        neither can be more precise than "the buffer, whatever it is now"); ``read``'s
        optional size argument is accepted unchecked past being int-or-bool-shaped (the
        VALUE does not change the result's TYPE). ``tell``/``seek`` return an
        unconstrained non-negative int, never connected back to ``content``. Every other
        StringIO method (readline, readlines, truncate, close, ...) still refuses --
        not modelled, exactly like any other unmodelled method of any other kind."""
        if name == "write":
            if len(args) != 1 or kwargs:
                raise Unsupported("StringIO.write: unmodelled arity/keywords", node)
            self._io_str_arg(st, args[0], node, "StringIO.write")
            self.update_obj(st, frozenset({addr}), lambda ob, strong: StringIOObj(ANYSTR))
            return int_iv(0, None)
        if name == "getvalue":
            if args or kwargs:
                raise Unsupported("StringIO.getvalue: unmodelled arity/keywords", node)
            ob = st.heap.get(addr)
            return ob.content if ob is not None else ANYSTR
        if name == "read":
            if len(args) > 1 or kwargs:
                raise Unsupported("StringIO.read: unmodelled arity/keywords", node)
            if args and (args[0].top or (args[0].i is None and not args[0].b)):
                raise Unsupported("StringIO.read: unmodelled size argument", node)
            ob = st.heap.get(addr)
            return ob.content if ob is not None else ANYSTR
        if name == "tell":
            if args or kwargs:
                raise Unsupported("StringIO.tell: unmodelled arity/keywords", node)
            return int_iv(0, None)
        if name == "seek":
            if not (1 <= len(args) <= 2) or kwargs:
                raise Unsupported("StringIO.seek: unmodelled arity/keywords", node)
            if args[0].top or (args[0].i is None and not args[0].b):
                raise Unsupported("StringIO.seek: unmodelled position argument", node)
            return int_iv(0, None)
        raise Unsupported(f"method {name} of an io.StringIO", node)

    # ------------------------------------------------ the write side of an output file ([internal ref])
    def _buffer_arg(self, st, v: V, node) -> bool:
        """True only when ``v`` is definitely an exact builtin buffer provider (an ``array``): a
        ``bytes``/``bytearray``/``memoryview`` has no abstract value here yet.  Unknown values and
        objects (a user class may define ``__buffer__``) abort; anything else definitely is NOT a
        buffer, and is a TypeError the caller records."""
        if v.top or v.st:
            raise Unsupported("output file: an argument of unknown type", node)
        for a in v.refs:
            ob = st.heap.get(a)
            if ob is None or ob.kind in ("inst", "class", "func", "bmeth", "bimeth"):
                raise Unsupported("output file: an object argument may provide a buffer", node)
        return bool(v.refs) and all(st.heap[a].kind == "array" for a in v.refs) and not (
            v.s is not None or v.i is not None or v.b or v.f or v.n)

    def outfile_method(self, st: State, addr, name, args, kwargs, node) -> V:
        """``write``, ``writelines``, ``flush``, ``close``, ``__enter__`` and ``__exit__`` of a file a
        write-only ``open()`` returned ([internal ref]).  Every raise claim is the contract's
        (``contracts/rules_fileio.py``, read through ``adapters/absint.py``); the model upgrades a class
        to proven only for what it knows of THESE arguments, and anything it does not understand
        aborts.  ``tell``/``seek`` and every attribute are unmodelled.

        Whether the file is closed is not tracked: the only exceptions that depends on
        (``ValueError``, ``OSError``) are not property classes, so a path that raises them ends and
        the paths that continue are the ones modelled.  A wrong-typed argument is a TypeError even
        on a closed file (the argument is parsed before the file is checked), which is what the
        claims say."""
        return self._outfile_call(st, st.heap[addr].binary, addr, name, args, kwargs, node)

    def _outfile_call(self, st: State, binary: bool, addr, name, args, kwargs, node) -> V:
        """One call of the family on a handle of kind ``binary`` (``addr`` is the receiver, which
        ``__enter__`` returns; ``__exit__`` of a ``with`` passes ``None``: the handle may have
        been rebound or collected by then, and the kind is all the call needs), through the
        method's rule (``contracts/rules_fileio.py``): its visit is ``_outfile_method``."""
        from tharkun.solve.contracts.adapters import absint as _contracts
        from tharkun.solve.contracts.rules import Refused, get_rule

        self._in_try_refuse(f"output file .{name} (its non-property raises are not routed)", node)
        rule = get_rule(f"{_contracts.OUTPUT_HANDLE_TYPES[binary]}.{name}")
        if rule is None:
            raise Unsupported(f"method {name} of an output file", node)
        result = rule.accept(self.visitor, OutfileCtx(st, binary, addr, name, tuple(args), dict(kwargs), node))
        if isinstance(result, Refused):
            raise Unsupported(str(result), node)
        return result

    def _outfile_method(self, st: State, binary: bool, addr, name, args, kwargs, node) -> V:
        """The transfer function of the output-file methods (the visit of every ``output_*`` rule)."""
        from tharkun.solve.contracts.adapters import absint as _contracts

        contract = _contracts.output_handle_contract(binary, name)
        if contract is None:
            raise Unsupported(f"method {name} of an output file", node)
        if not _contracts.output_handle_call_shape(name, len(args), bool(kwargs)):
            raise Unsupported(f"{name} of an output file: unmodelled arity/keywords", node)
        proven = {"TypeError": "the call has exactly the documented arguments"}
        result = NONE
        if name == "write":
            v = args[0]
            if binary:
                ok = self._buffer_arg(st, v, node)
                proven.pop("TypeError")
                if ok:
                    proven.update({exc: "an array argument: no __buffer__ runs"
                                   for exc in ("AttributeError", "IndexError", "KeyError", "ZeroDivisionError",
                                               "FloatingPointError", "AssertionError")})
                    proven["TypeError"] = "the argument is an array (a buffer provider)"
            else:
                if v.top:
                    raise Unsupported("output file: an argument of unknown type", node)
                ok = v.s is not None and not (v.i is not None or v.b or v.f or v.n or v.refs or v.st)
                proven.pop("TypeError")
                if ok:
                    proven["TypeError"] = "the argument is a str"
            result = int_iv(0, None)
        elif name == "writelines":
            proven.pop("TypeError")
            v = args[0]
            if v.top or v.st:
                raise Unsupported("output file: an argument of unknown type", node)
            for a in v.refs:
                kind = st.heap[a].kind if a in st.heap else None
                if kind in ("inst", "class", "func", "bmeth", "bimeth", "iterator", "dictview", "stringio", "pattern", "outfile"):
                    raise Unsupported("output file: an iterable that may run user code", node)
            seqs = bool(v.refs) and all(st.heap[a].kind in ("list", "tuple") for a in v.refs) and not (
                v.s is not None or v.i is not None or v.b or v.f or v.n)
            if seqs:
                elem = self.elem_summary(st, v)
                if elem is None:
                    ok = True            # every possible sequence is empty: nothing is written
                elif binary:
                    ok = self._buffer_arg(st, elem, node)
                else:
                    if elem.top:
                        raise Unsupported("output file: elements of unknown type", node)
                    ok = elem.s is not None and not (elem.i is not None or elem.b or elem.f or elem.n or elem.refs or elem.st)
                if ok:
                    proven.update({exc: "a list or tuple of exact elements: no iterator or element hook runs"
                                   for exc in ("AttributeError", "IndexError", "KeyError", "ZeroDivisionError",
                                               "FloatingPointError", "AssertionError")})
                    proven["TypeError"] = "a list or tuple whose every element is " + ("a buffer" if binary else "a str")
        elif name == "__enter__":
            result = V(refs=frozenset({addr}))
        self._contract_checks(node, f"{_contracts.OUTPUT_HANDLE_TYPES[binary]}.{name}", contract, proven)
        return result

    def s_With(self, s, st, fctx, lctx):
        """``with <expr> [as name]: body`` for the write side of an output file only ([internal ref]).  Every
        item must evaluate to output handles; ``__enter__`` returns the handle itself, and
        ``__exit__`` closes it and returns ``None``, so an exception of the body propagates
        unchanged and a path that leaves the body by ``return``/``break``/``continue`` loses
        nothing (what ``close`` can raise is not a property class).  Any other context manager
        stays unmodelled (Unsupported), as before."""
        handles = []
        for item in s.items:
            ctx = self.eval(item.context_expr, st, fctx)
            if st.bottom:
                return st
            if ctx.top or ctx.st or ctx.s is not None or ctx.i is not None or ctx.b or ctx.f or ctx.n or not ctx.refs \
                    or any(st.heap[a].kind != "outfile" for a in ctx.refs):
                raise Unsupported("with: a context manager other than a write-only output file", item.context_expr)
            entered = BOT
            kinds = set()
            for a in sorted(ctx.refs, key=str):
                kinds.add(st.heap[a].binary)
                entered = vjoin(entered, st.zone, self.outfile_method(st, a, "__enter__", [], {}, item.context_expr), st.zone)
            handles.append(kinds)
            if item.optional_vars is not None:
                if not isinstance(item.optional_vars, ast.Name):
                    raise Unsupported("with: the target is not a plain name", item.optional_vars)
                self.assign(item.optional_vars, entered, st, fctx)
        end = self.exec_block(s.body, st, fctx, lctx)
        if end.bottom:
            return end
        for kinds in reversed(handles):
            for binary in sorted(kinds):
                self._outfile_call(end, binary, None, "__exit__", [NONE, NONE, NONE], {}, s)
        return end


def _refine_le(z, d: Lin, k: int) -> None:
    """Meet the zone with ``d <= k`` when ``d`` is a zone-expressible form."""
    t = d.terms
    if not t:
        if d.c > k:
            z.bottom = True
            z.m.clear()
        return
    if len(t) == 1:
        (x, c), = t
        if c == 1:
            z.add(x, ZERO, k - d.c)
        elif c == -1:
            z.add(ZERO, x, k - d.c)
        return
    if len(t) == 2 and {t[0][1], t[1][1]} == {1, -1}:
        x = t[0][0] if t[0][1] == 1 else t[1][0]
        y = t[1][0] if t[0][1] == 1 else t[0][0]
        z.add(x, y, k - d.c)


def _refine_ne_lin(z, d: Lin) -> None:
    """Meet with ``d != 0`` where it tightens a bound (integers)."""
    t = d.terms
    if len(t) == 1 and t[0][1] in (1, -1):
        x = t[0][0]
        if t[0][1] == 1:
            _refine_ne(z, x, ZERO, -d.c)
        else:
            # -x + c != 0  <=>  x != c
            _refine_ne(z, x, ZERO, d.c)
    elif len(t) == 2 and {t[0][1], t[1][1]} == {1, -1}:
        x = t[0][0] if t[0][1] == 1 else t[1][0]
        y = t[1][0] if t[0][1] == 1 else t[0][0]
        _refine_ne(z, x, y, -d.c)
    elif not t and d.c == 0:
        z.bottom = True
        z.m.clear()


def _refine_ne(z, x: str, y: str, c: int) -> None:
    """Meet with ``x - y != c``: tighten a bound that sits exactly at ``c``."""
    if z.bottom:
        return
    up = z.get(x, y)
    down = z.get(y, x)
    if up is not None and up == c:
        z.add(x, y, c - 1)
    elif down is not None and -down == c:
        z.add(y, x, -c - 1)


def _max_lo(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _min_hi(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _rename_lin(lin: Lin, old: str, new: str) -> Lin:
    return Lin({(new if k == old else k): c for k, c in lin.terms}, lin.c)


def _rename_ii(ii: IntInfo, old: str, new: str) -> IntInfo:
    lin = _rename_lin(ii.lin, old, new) if ii.lin is not None else None
    lo = (new, ii.lo[1]) if ii.lo is not None and ii.lo[0] == old else ii.lo
    hi = (new, ii.hi[1]) if ii.hi is not None and ii.hi[0] == old else ii.hi
    return IntInfo(lin, lo, hi)


def _assign_state(dst: State, src: State) -> None:
    dst.frames = src.frames
    dst.heap = src.heap
    dst.zone = src.zone
    dst.pins = src.pins


def _tuple_elem(st: State, ob) -> V:
    e = BOT
    for x in ob.items:
        e = vjoin(e, st.zone, x, st.zone)
    return e


def _len_iv(st: State, ob):
    if ob.kind == "tuple" and ob.items is not None:
        return (len(ob.items), len(ob.items))
    return iv_of_len(ob.length, st.zone)


def _ojoin(a, za, b, zb):
    from .state import ojoin
    return ojoin(a, za, b, zb)


def _obj_zvars(ob):
    return ob.zvars() if ob is not None else set()


def _zvars_named(z, site, an) -> set:
    sid = an.site_ids.get(site)
    if sid is None:
        return set()
    return {v for v in z.vars if v == f"L.{sid}" or v.startswith(f"R.{sid}.")}


LIST_METHODS = frozenset({"append", "insert", "pop", "reverse", "extend", "copy", "index", "count",
                          "remove", "sort", "clear"})

# [internal ref]: array (docs/ABSINT.md Slice 3). Scope is deliberately narrow: append/extend, the
# same ones list gets among the mutators (insert/pop/remove/... refuse -- Unsupported --
# rather than guess at a shape nobody has needed yet, MG's PR 89 lesson: never set_bottom on
# an unrecognized shape). Only the FIXED-width C typecodes are modelled -- 'l'/'L' ("long") are
# refused outright: their itemsize is genuinely platform-dependent in CPython's own array
# module (4 bytes on Windows' LLP64 long, 8 on 64-bit Linux/macOS), unlike every other
# typecode here, whose width is the same across every mainstream platform/compiler. 'u'
# (unicode char) is refused too: no str/bytes element modelling exists here to back it, and
# it does not appear in any known SV-Benchmarks array use (measured: [internal ref]).
_ARRAY_INT_RANGES = {
    "b": (-128, 127), "B": (0, 255),
    "h": (-32768, 32767), "H": (0, 65535),
    "i": (-(2 ** 31), 2 ** 31 - 1), "I": (0, 2 ** 32 - 1),
    "q": (-(2 ** 63), 2 ** 63 - 1), "Q": (0, 2 ** 64 - 1),
}
_ARRAY_FLOAT_CODES = frozenset({"f", "d"})
_ARRAY_TYPECODES = frozenset(_ARRAY_INT_RANGES) | _ARRAY_FLOAT_CODES
# CPython's array module SATURATES an out-of-range 'f' element to +-inf rather than raising
# OverflowError (measured: `array.array('f', [1e300])[0] == inf`, no exception -- unlike
# `struct.pack('f', 1e300)`, which DOES raise; the two are not the same contract). So there is
# no float OverflowError check to make here at all -- only the ROUNDING to single precision,
# which is what the required 'f'-rounding never-TRUE control exercises. `struct` IS still used
# (for the rounding itself, an exact IEEE-754 binary32 round-trip) but only after this
# saturation check keeps its call in-domain, since `struct.pack('<f', ...)` would otherwise
# raise on exactly the magnitudes real `array` does not.
_FLOAT32_MAX = 3.4028234663852886e+38
ARRAY_METHODS = frozenset({"append", "extend"})

BUILTIN_FUNCS = frozenset({"abs", "len", "iter", "range", "min", "max", "print", "isinstance", "sorted"})
# [internal ref]: "str"/"bool" added so a bare reference to them (e.g. isinstance's 2nd argument) gets
# the same ("type", name) static tag "list"/"int"/"float"/"object" already do, instead of an
# unmodelled top -- CALLING str(x)/bool(x) is unaffected either way (no t_str/t_bool handler
# exists; call_static's "type" dispatch already raises Unsupported for a missing handler
# exactly like calling a top value already did, same outcome, different message only).
BUILTIN_TYPES = frozenset({"list", "int", "float", "object", "str", "bool", "dict"})

MODULES = {
    "math": {
        # EVERY function of the module (contracts/rules_math.py): _math_call is driven by the contract
        # and the shared model table (contracts/adapters/math_model.py); a name whose contract is
        # withdrawn or untested is Unsupported there. The constants (pi, e, tau, inf, nan) are read through their
        # rules (``module_attr`` -> ``visit_rule`` -> ``AbsintVisitor.visit_math_constant``): the names are this
        # table's, the VALUES are the contract's (``adapters/math_model.py::constant``).
        "funcs": _mm.modelled_names(),
        "consts": {name: None for name in _mm.CONSTANTS},
    },
    "random": {
        "funcs": {"choice", "random", "randint", "randrange", "getrandbits", "uniform", "seed", "getstate"},
    },
    "_sv_verifier": {
        "funcs": {"nondet_int", "nondet_bool", "nondet_float", "nondet_str", "nondet_float_range",
                   "nondet_list", "nondet_dict", "check_type"},
    },
    "io": {
        # [internal ref]: StringIO only (BytesIO is not used by any current SV
        # target; add it the same way if one shows up -- see m_io_StringIO's
        # docstring). The constructor is the only module-level entry point;
        # every other operation is a METHOD on the object it returns (see
        # stringio_method).
        "funcs": {"StringIO"},
    },
    "re": {
        # Every callable sibling dispatches through the shared Rule visitor.
        # Flags retain their native enum identity in module_attr.
        "funcs": {"compile", "match", "search", "fullmatch", "sub", "subn", "split", "findall", "finditer", "escape", "purge"},
    },
    # [internal ref] (the "os bug", [internal ref]): `os` used to be refused wholesale (not in
    # MODULES, not in determinism.py's DET_MODULES either) -- any program that so much as
    # wrote `import os`, even only to compute a path, could never get a single check proven.
    # solve/determinism.py's own OS_CONST_OK/OS_PATH_PURE/OS_HAVOC/OS_PATH_HAVOC document the
    # exact same three-way split this dict encodes for absint: pure os.path string functions
    # (computed EXACTLY when every argument is a single known str, else a sound havoc string
    # -- never a refusal); reads of real process/OS/filesystem state (env vars, cwd, a
    # directory listing, OS entropy, the PID, a path's existence, ...) become a HAVOC value of
    # the right type/shape (never Unsupported -- a program may use one of these for something
    # that does not affect a property this pass tracks, and this lets it keep proving the
    # REST of that program); anything else (`os.system`, `os.remove`, `os.environ` used bare,
    # ...) is listed under "refused" so its OWN attribute access aborts with a clear reason,
    # matching module_attr's normal "known module, unmodelled name -> top" default ONLY for
    # names this pass has genuinely not looked at -- these have, and were rejected on purpose.
    "os": {
        "consts": {
            "path": V(st=frozenset({("module", "os.path")})),
            "sep": const_str(os.sep), "linesep": const_str(os.linesep),
            "pathsep": const_str(os.pathsep), "extsep": const_str(os.extsep),
            "curdir": const_str(os.curdir), "pardir": const_str(os.pardir),
            "name": const_str(os.name), "devnull": const_str(os.devnull),
            **({"altsep": const_str(os.altsep)} if os.altsep else {"altsep": NONE}),
        },
        "funcs": {"getenv", "getcwd", "listdir", "urandom", "getpid", "cpu_count"},
        "refused": frozenset({
            # `environ` is refused, not modelled as a mapping (unlike `getenv`, its function
            # equivalent): this dict's "module"-tag trick (see `path` above) works for `.get`
            # dispatch (module_attr recurses the same way for a nested MODULE), but real
            # `os.environ` is a genuine Mapping instance -- SUBSCRIPTING it (`os.environ[k]`,
            # a real, common KeyError site) falls through the interpreter's subscript
            # handling with no heap object to find (a "module" tag has none), which read as
            # an IMPOSSIBLE operation (state -> bottom) rather than an unproven check --
            # caught by tests/fixtures/determinism/never_certified/d07_environ.py turning a
            # real, environment-dependent KeyError into a WRONG "TRUE, 0 checks". `getenv`
            # has no such trap (a plain function call, never subscripted); `environ` gets a
            # real mapping model as future work instead of shipping this shortcut.
            "environ",
            "remove", "unlink", "rename", "renames", "replace", "chdir", "fchdir", "mkdir",
            "makedirs", "rmdir", "removedirs", "system", "popen", "fork", "forkpty", "kill",
            "killpg", "abort", "_exit", "execl", "execle", "execlp", "execlpe", "execv",
            "execve", "execvp", "execvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe",
            "spawnv", "spawnve", "spawnvp", "spawnvpe", "putenv", "unsetenv", "open", "close",
            "dup", "dup2", "pipe", "truncate", "chmod", "chown", "lchown", "symlink", "link",
            "mkfifo", "mknod", "waitpid", "wait", "wait3", "wait4", "setuid", "setgid",
            "seteuid", "setegid", "setpgid", "setsid", "umask", "write", "read", "fdopen",
            "startfile",
        }),
    },
    "os.path": {
        "funcs": {"join", "dirname", "basename", "splitext", "normpath", "abspath", "realpath",
                  "exists", "isfile", "isdir"},
    },
    # [internal ref]: `array.array` is a class, but it is reached the same way `re.compile` is (a
    # module-level callable dispatched through the "mfunc" tag -- module_attr already builds
    # that tag for anything in a module's "funcs" set, so no call_static change is needed).
    # See m_array_array.
    "array": {
        "funcs": {"array"},
    },
}


def _served_by_contract(module: str, name: str) -> bool:
    """Is ``module.name`` reached through its contract's generic call (``Analyzer.module_attr`` tags it ``cfunc``)?
    Not when the analyzer models it by hand, refuses it, or holds it as a constant."""
    m = MODULES.get(module, {})
    return ((module, name) not in {("copy", "copy"), ("collections", "defaultdict")}
            and name not in m.get("funcs", ())
            and name not in m.get("refused", ()) and name not in m.get("consts", {}))


serve_module_functions(_served_by_contract)
serve_set_operations()
