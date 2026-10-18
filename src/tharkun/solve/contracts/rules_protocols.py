"""Contracts for object PROTOCOLS ([internal ref] batch 2) -- see ``docs/contracts/FORMAT.md``'s
"``protocol.NAME`` qualnames" section for what these are and why they exist:
Tharkun's abstract interpreter (`solve/absint/interp.py`) currently treats ANY
expression whose type it cannot pin down as `Unsupported` the moment it needs a
dunder the interpreter has no model for -- a user class using `__add__` (bm_raytrace,
chaos) or `__getitem__` (hexiom, scimark) stops the analysis cold. These contracts are
the tested, sourced specification of the DISPATCH MECHANISMS themselves, so that
absint (a later, separately reviewed step) can model "what CPython actually does with
an unknown object's dunders" instead of giving up.

Every claim below was read from CPython 3.14.4's own source (`Objects/abstract.c`,
`Objects/object.c`, `Objects/typeobject.c`, `Objects/iterobject.c`) and then verified
against the running interpreter with instrumented stub classes -- both directions,
every time: reading the C code alone would have missed the exact conditions (e.g.
`binary_op1`'s slot-pointer-identity check has no visible Python-level analogue to
guess at), and running examples alone would have missed why a case behaves the way it
does (e.g. why a subclass overriding `__radd__` skips `__add__` entirely, but a
subclass overriding `__gt__` for rich comparison does NOT get the same slot-identity
exemption). ``tests/contracts/test_protocols.py`` re-derives every ordering claim with
its own instrumented classes and a call-order log, independent of the source reading.

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims
from tharkun.solve.contracts.rules import Rule, Shape, register_rule
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec

_LANG = "https://docs.python.org/3.14/reference/datamodel.html"

# ------------------------------------------------------------------
# Operator dispatch
# ------------------------------------------------------------------


@register_rule
class ProtocolBinaryOperator(Rule):
    """The rule of ``protocol.binary_operator``."""

    SAMPLES = {'protocol.binary_operator': 'class A:\n    def __add__(self, other):\n        return 1\nx = A() + A()\n'}
    TWINS = {}

    names = ("protocol.binary_operator",)
    CONTRACT = Contract(
        qualname="protocol.binary_operator",
        raises=dunder_propagation_claims(
            "type(a).__add__(a, b), or type(b).__radd__(b, a) FIRST when type(b) is "
            "a PROPER SUBTYPE of type(a) AND its slot differs from type(a)'s (an "
            "unoverridden inherited slot does not trigger the priority swap)",
            native={
                "TypeError": (
                    "both the tried method(s) return NotImplemented or don't exist "
                    "at all -- neither operand's type supports the operator with the "
                    "other's type. If a and b are the SAME type, only __add__ is "
                    "tried once (no __radd__ retry) -- trying the identical slot "
                    "twice would be pointless"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # __add__/__radd__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=None,
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "The __add__/__radd__ example generalizes to every binary operator via "
            "its own dunder pair: __sub__/__rsub__, __mul__/__rmul__, __truediv__/"
            "__rtruediv__, __floordiv__/__rfloordiv__, __mod__/__rmod__, __pow__/"
            "__rpow__, __matmul__/__rmatmul__, __and__/__rand__, __or__/__ror__, "
            "__xor__/__rxor__, __lshift__/__rlshift__, __rshift__/__rrshift__ (and "
            "divmod()'s __divmod__/__rdivmod__, already in builtins.py). The "
            "subclass-priority rule: if the reflected method IS tried first and "
            "returns NotImplemented, the left method is then tried as normal (the "
            "priority is about ORDER, not exclusivity). BATCH 5 (tests/contracts/test_swallowing_native.py, group "
            "'notimplemented'): ONLY a RETURNED NotImplemented triggers the reflected retry -- a TypeError RAISED by "
            "__add__ propagates and __radd__ is not tried; the same for __iadd__ (a returned NotImplemented falls back "
            "to __add__, a raised TypeError propagates). `3 + inst` and `[1] + inst` call the instance's __radd__ (the "
            "latter BEFORE list's own concatenation); a subclass overriding the reflected method is tried first."
        ),
        source=(
            f"{_LANG}#object.__radd__",
            "CPython Objects/abstract.c::binary_op1 (the exact slot-pointer-identity "
            "and subtype checks), ::binary_op (the TypeError on double-NotImplemented)",
        ),
        tested=True,
    )


@register_rule
class ProtocolUnaryOperator(Rule):
    """The rule of ``protocol.unary_operator``."""

    SAMPLES = {'protocol.unary_operator': 'class A:\n    def __neg__(self):\n        return 1\nx = -A()\n'}
    TWINS = {}

    names = ("protocol.unary_operator",)
    CONTRACT = Contract(
        qualname="protocol.unary_operator",
        raises=dunder_propagation_claims(
            "the operand's __neg__ (for -x), __pos__ (for +x), or __invert__ (for ~x)",
            native={"TypeError": "the operand's type has no corresponding dunder"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # __neg__/__pos__/__invert__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=None,
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes="abs()/__abs__ is already covered as a builtin in builtins.py -- it is a unary operator in this same family, just spelled as a function call rather than a token.",
        source=(f"{_LANG}#object.__neg__", "CPython Objects/abstract.c::PyNumber_Negative"),
        tested=True,
    )


@register_rule
class ProtocolInplaceOperator(Rule):
    """The rule of ``protocol.inplace_operator``."""

    SAMPLES = {'protocol.inplace_operator': 'class A:\n    def __iadd__(self, other):\n        return self\na = A()\na += 1\n'}
    TWINS = {}

    names = ("protocol.inplace_operator",)
    CONTRACT = Contract(
        qualname="protocol.inplace_operator",
        raises=dunder_propagation_claims(
            "the target's __iadd__ (for +=), falling back to the full binary "
            "dispatch (protocol.binary_operator, i.e. __add__/__radd__) when "
            "__iadd__ is absent OR returns NotImplemented",
            native={
                "TypeError": "NEITHER side has __iadd__ NOR does the full binary dispatch (__add__/__radd__) find any support -- 'unsupported operand type(s) for +='",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # __iadd__ (or the __add__/__radd__ fallback) is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=None,
        effects=Effects(deterministic=True, may_execute_callback=True, notes="when __iadd__ succeeds, `x += y` REBINDS x to its return value -- it does not assume the object mutated itself, though in-place dunders conventionally do and then return self."),
        notes="generalizes the same way as binary_operator: __isub__, __imul__, __itruediv__, __ifloordiv__, __imod__, __ipow__, __imatmul__, __iand__, __ior__, __ixor__, __ilshift__, __irshift__, each falling back to its non-i counterpart.",
        source=(f"{_LANG}#object.__iadd__", "CPython Objects/abstract.c::binary_iop1"),
        tested=True,
    )


# ------------------------------------------------------------------
# Comparison
# ------------------------------------------------------------------


@register_rule
class ProtocolComparison(Rule):
    """The rule of ``protocol.comparison``."""

    SAMPLES = {'protocol.comparison': 'class A:\n    def __lt__(self, other):\n        return True\nx = A() < A()\n'}
    TWINS = {}

    names = ("protocol.comparison",)
    CONTRACT = Contract(
        qualname="protocol.comparison",
        raises=dunder_propagation_claims(
            "a.__lt__(b) for a < b (etc. per operator), or b's SWAPPED reflected "
            "method (__gt__ for <, __ge__ for <=, __eq__/__ne__ reflect to "
            "themselves) FIRST when type(b) is a proper subtype of type(a) -- "
            "UNCONDITIONALLY on subtype, unlike protocol.binary_operator's extra "
            "slot-identity check",
            native={
                "TypeError": (
                    "an ORDERING comparison (<, <=, >, >=) with neither side's method "
                    "returning non-NotImplemented. == and != are DIFFERENT: with no "
                    "method implementing them, they fall back to identity (a is b) "
                    "and its negation respectively, and NEVER raise TypeError for "
                    "this reason"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # __lt__/etc. is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("bool (or whatever a __lt__/etc. returns, if not bool -- unlike __bool__, rich comparison does NOT require an exact bool)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "BATCH 5 (tests/contracts/test_swallowing_native.py, group 'notimplemented'): == and != with both sides "
            "returning NotImplemented are IDENTITY (never TypeError), < with both NotImplemented is TypeError, a < b "
            "tries b.__gt__(a) when a.__lt__ returns NotImplemented; an exception RAISED by __eq__ propagates with no "
            "reflected retry. Containers compare by IDENTITY FIRST (list ==, `in`, index, count, dict lookup): a "
            "raising __eq__ is never called on the identical object, but is for a distinct one. sorted() compares "
            "with < (reflecting to >): __gt__ alone suffices, __le__ alone does not; max() compares with >."
        ),
        source=(
            f"{_LANG}#object.__lt__",
            "CPython Objects/object.c::do_richcompare (the subtype-priority check "
            "and the identity fallback for ==/!=)",
        ),
        tested=True,
    )


@register_rule
class ProtocolCall(Rule):
    """A call slot, including the construction path for class objects."""

    SAMPLES = {"protocol.call": "class A:\n    def __call__(self):\n        return 1\nx = A()()\n"}
    TWINS = {}

    names = ("protocol.call",)
    SHAPE = Shape(min_args=0, max_args=None, keywords_ok=True)
    CONTRACT = Contract(
        qualname="protocol.call",
        raises=dunder_propagation_claims(
            "the callable type's tp_call slot (for an instance, its type's __call__; "
            "for a class, the metaclass call path through __call__, __new__, and __init__)",
            native={
                "TypeError": (
                    "the evaluated target has no call slot, or the call slot/class "
                    "construction rejects its arguments or result; TypeError raised "
                    "by the invoked callback also propagates"
                ),
            },
            tested_native=frozenset({"TypeError"}),
            tested_propagation=True,
        ),
        may_raise_other=True,
        returns=None,
        effects=Effects(deterministic=False, may_execute_callback=True),
        source=(
            f"{_LANG}#object.__call__",
            "CPython Objects/call.c::PyObject_Call and Objects/typeobject.c::type_call",
        ),
        tested=True,
        notes=(
            "Calling an instance dispatches through its type slot; an instance-level "
            "attribute named __call__ does not replace that slot. Calling a class "
            "runs its metaclass call slot and may then run __new__ and __init__. "
            "User callbacks make the whole call's determinism unknown."
        ),
    )


# ------------------------------------------------------------------
# Attribute access
# ------------------------------------------------------------------


@register_rule
class ProtocolAttributeAccess(Rule):
    """The rule of ``protocol.attribute_access``."""

    SAMPLES = {'protocol.attribute_access': 'class A:\n    pass\na = A()\na.x = 1\ny = a.x\n'}
    TWINS = {}

    names = ("protocol.attribute_access",)
    CONTRACT = Contract(
        qualname="protocol.attribute_access",
        raises=dunder_propagation_claims(
            "__getattribute__ (ALWAYS called first, default PyObject_GenericGetAttr: "
            "type dict incl. base classes, then instance __dict__, then falls back "
            "to a DATA descriptor's __get__ ahead of the instance dict, a non-data "
            "one behind it) -- and only when __getattribute__ raises AttributeError, "
            "__getattr__ (if defined) is tried as a fallback",
            native={
                "AttributeError": (
                    "the name is not found by __getattribute__'s normal lookup, AND "
                    "either there is no __getattr__ or it also raises AttributeError. "
                    "A __slots__ class with no __dict__ raises this both for reading "
                    "an unset slot and for setting/reading a name outside __slots__"
                ),
            },
            tested_native=frozenset({"AttributeError"}),
        ),
        may_raise_other=True,  # __getattribute__/__getattr__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=None,
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "descriptor priority (data vs non-data): a DATA descriptor (defines "
            "__set__ or __delete__) on the class always wins over an instance "
            "__dict__ entry of the same name; a NON-DATA descriptor (only __get__, "
            "e.g. a plain function becoming a bound method) is SHADOWED by an "
            "instance __dict__ entry of the same name. __getattr__ is never even "
            "invoked for a name that ordinary lookup already resolves -- it is a "
            "fallback, not a hook on every access (unlike __getattribute__, which "
            "runs on EVERY attribute access, always). BATCH 5 (tests/contracts/test_swallowing_native.py, group "
            "'attribute'): the AttributeError that triggers __getattr__ includes one raised INSIDE a property getter "
            "or a descriptor's __get__ (the property's value is then replaced by __getattr__'s), and any other class "
            "from __getattribute__ propagates without consulting __getattr__. hasattr(o, n) and getattr(o, n, default) "
            "swallow ONLY AttributeError (a KeyError or TypeError from __getattr__ propagates); isinstance(o, C) "
            "swallows an AttributeError from o.__class__ (the answer is False) and propagates the rest; `del o.x` of a "
            "missing x converts the instance dict's KeyError to AttributeError; reading an unset slot is "
            "AttributeError. #1072: `Cls.x = 5` replaces the slot descriptor, instances then read the class value and "
            "their writes are AttributeError (a class variable named like a slot IN the class body is ValueError)."
        ),
        source=(
            f"{_LANG}#object.__getattr__",
            "CPython Objects/object.c::PyObject_GetAttr, ::_PyObject_GenericGetAttrWithDict "
            "(descriptor priority order)",
        ),
        tested=True,
    )


# ------------------------------------------------------------------
# Container protocols
# ------------------------------------------------------------------


@register_rule
class ProtocolContainerLen(Rule):
    """The rule of ``protocol.container_len``."""

    SAMPLES = {'protocol.container_len': 'class A:\n    def __len__(self):\n        return 2\nx = len(A())\n'}
    TWINS = {}

    names = ("protocol.container_len",)
    CONTRACT = Contract(
        qualname="protocol.container_len",
        raises=dunder_propagation_claims(
            "the object's own __len__ (a USER method -- CPython dispatches to it via "
            "the type slot, not a getattr/hasattr probe, so ANY exception it raises, "
            "not just TypeError/ValueError, propagates uncaught)",
            native={
                "TypeError": "__len__ returns something with no __index__ (e.g. a str or float), or raises TypeError itself",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: __len__ returns a negative int
        returns=ReturnSpec("int", ">= 0 -- a negative return is ValueError, not silently accepted or clamped", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "len(x) itself (the builtin) is covered in builtins.py, which already "
            "gets this propagation right; this entry is the __len__ RETURN-VALUE "
            "contract every container-like class must satisfy, which builtins.py's "
            "len() and this module's protocol.truthiness both consume. SOUNDNESS "
            "FIX (found by the gatekeeper): an earlier draft used never_claims here, "
            "wrongly marking IndexError/KeyError/AttributeError/ZeroDivisionError/"
            "FloatingPointError/AssertionError as 'never' -- a user __len__ that "
            "raises any of them propagates just as unconditionally as one that "
            "raises TypeError does. Repro: class X: __len__ = lambda self: 1/0 "
            "raises ZeroDivisionError from len(X()); the pattern generalizes to all "
            "7 classes and is what tests/contracts/test_protocols.py now sweeps. "
            "BATCH 5: that propagation is for a DIRECT call (len(x), a truth test). When "
            "list(x)/sorted(x)/list.extend(x)/... ask __len__ for a size hint "
            "(protocol.length_hint), a TypeError from it is SWALLOWED and every other class "
            "still propagates."
        ),
        source=(f"{_LANG}#object.__len__", "CPython Objects/abstract.c::PyObject_Size"),
        tested=True,
    )


@register_rule
class ProtocolLengthHint(Rule):
    """The rule of ``protocol.length_hint``."""

    SAMPLES = {'protocol.length_hint': 'class A:\n    def __len__(self):\n        return 2\n    def __iter__(self):\n        return iter([1, 2])\nx = list(A())\n'}
    TWINS = {}

    names = ("protocol.length_hint",)
    CONTRACT = Contract(
        qualname="protocol.length_hint",
        raises=dunder_propagation_claims(
            "the object's __len__ and, when it has no __len__, its __length_hint__ -- asked by PyObject_LengthHint, "
            "which list(x), sorted(x), list.extend(x), l += x, [*x], bytes(x), bytearray.extend(x) and "
            "operator.length_hint(x) call on the ORIGINAL object, after iter(x), to presize their result "
            "(tuple, set, frozenset, array.array and a for loop never call it)",
            native={
                "TypeError": (
                    "NOT raised when the object's __len__ or __length_hint__ raises it, returns a non-int, or cannot be "
                    "called: that TypeError is SWALLOWED (there is no hint, iteration proceeds, and the side effects "
                    "__len__ made before it raised stay). Raised by the machinery itself only when __length_hint__ "
                    "returns a non-int, a case the swallowing does not cover"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: a negative __len__/__length_hint__; OverflowError: beyond sys.maxsize
        returns=ReturnSpec("int", ">= 0; the consumer's default when __len__/__length_hint__ raise TypeError, are absent or "
                                  "__length_hint__ returns NotImplemented", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True,
                        notes="runs the object's own __len__/__length_hint__, whose side effects persist on the swallowing "
                              "path as on any other"),
        notes=(
            "PEP 424: 'it is purely an optimization and never required for correctness'. ONLY TypeError is swallowed -- "
            "AttributeError, KeyError, IndexError, ZeroDivisionError, AssertionError, FloatingPointError, ValueError and "
            "every other class propagate (an earlier model of this protocol in absint swallowed AttributeError too and "
            "gave wrong TRUEs). A negative result is ValueError, a result beyond sys.maxsize OverflowError, a non-int "
            "__len__ result is the swallowed TypeError of PyObject_Size. iter(x) runs BEFORE the probe, so an "
            "__iter__ that raises wins. truthiness (protocol.truthiness) and len() call __len__ WITHOUT the swallowing: "
            "their TypeError propagates. `adapters/common.py::LENGTH_HINT_RULES` is the table of consumers the engines "
            "act on, pinned by tests/contracts/test_swallowing_native.py (twelve consumers x every exception class)."
        ),
        source=(
            "https://peps.python.org/pep-0424/",
            "CPython Objects/abstract.c::PyObject_LengthHint",
            "CPython Modules/_operator.c::_operator_length_hint_impl",
        ),
        tested=True,
    )


@register_rule
class ProtocolContainerGetitem(Rule):
    """The rule of ``protocol.container_getitem``."""

    SAMPLES = {'protocol.container_getitem': 'class A:\n    def __getitem__(self, i):\n        return i\nx = A()[0]\n'}
    TWINS = {}

    names = ("protocol.container_getitem",)
    CONTRACT = Contract(
        qualname="protocol.container_getitem",
        raises=dunder_propagation_claims(
            "the object's __getitem__ (called with the key EXACTLY as written: int, "
            "slice(start, stop, step), tuple for x[a, b], or any other object for "
            "x[obj] -- Ellipsis for x[...])",
            native={
                "TypeError": "the object has no __getitem__ at all ('object is not subscriptable')",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # __getitem__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=None,
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "CRITICAL DIFFERENCE from list/tuple/str/bytes.__getitem__ (already "
            "contracted per-type in batch 1): for an ARBITRARY user class (this is "
            "the shape hexiom/scimark use), there is NO automatic negative-index "
            "wraparound and NO automatic slice-bounds clamping -- x[-1] passes -1 "
            "to __getitem__ VERBATIM, and x[1:100] passes slice(1, 100, None) "
            "verbatim. Wraparound/clamping is a batch-1-contracted BEHAVIOR of the "
            "specific builtin C types, not a language-level guarantee __getitem__ "
            "implementers get for free. __setitem__/__delitem__ raise their own "
            "'does not support item assignment'/'doesn't support item deletion' "
            "TypeErrors when absent, independently of __getitem__'s presence."
        ),
        source=(f"{_LANG}#object.__getitem__", "CPython Objects/abstract.c::PyObject_GetItem"),
        tested=True,
    )


@register_rule
class ProtocolContainsFallback(Rule):
    """The rule of ``protocol.contains_fallback``."""

    SAMPLES = {'protocol.contains_fallback': 'class A:\n    def __iter__(self):\n        return iter([1])\nx = 1 in A()\n'}
    TWINS = {}

    names = ("protocol.contains_fallback",)
    CONTRACT = Contract(
        qualname="protocol.contains_fallback",
        raises=dunder_propagation_claims(
            "__contains__ if defined; else __iter__ (scanning with == against each "
            "yielded element); else old-style __getitem__ iteration (index 0, 1, 2, "
            "... until IndexError, == against each result)",
            native={
                "TypeError": "the object has NONE of __contains__, __iter__, or __getitem__ ('argument of type X is not a container or iterable')",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # __contains__ (or its __iter__/__getitem__ fallback) is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("bool", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "the __getitem__ fallback here is the SAME old-style iteration protocol.iteration documents for `for` loops -- "
            "`x in obj` and `for _ in obj` degrade to identical index-based scanning when obj has __getitem__ but neither "
            "__contains__ nor __iter__. BATCH 5 (tests/contracts/test_swallowing_native.py, group 'contains'): the scan ENDS "
            "(the answer is False) on IndexError or StopIteration from __getitem__ only; an IndexError from __contains__ "
            "itself, or from an iterator's __next__, PROPAGATES (only the sequence protocol swallows it); "
            "__contains__ = None blocks both fallbacks and is a TypeError; a __contains__ that exists wins over an "
            "__iter__ that would raise. set.__contains__/remove/discard convert a SET key to a frozenset when hashing "
            "it raised TypeError (even a user __hash__ on a set subclass; any other class from the hash propagates), "
            "dict.__contains__ does not (an unhashable set key is TypeError); `1 in 'abc'` is TypeError and "
            "`'a' in range(3)` is False."
        ),
        source=(f"{_LANG}#object.__contains__", "CPython Objects/abstract.c::PySequence_Contains"),
        tested=True,
    )


# ------------------------------------------------------------------
# Iteration
# ------------------------------------------------------------------


@register_rule
class ProtocolIteration(Rule):
    """The rule of ``protocol.iteration``."""

    SAMPLES = {'protocol.iteration': 'class A:\n    def __iter__(self):\n        return iter([1])\nfor v in A():\n    pass\n'}
    TWINS = {}

    names = ("protocol.iteration",)
    CONTRACT = Contract(
        qualname="protocol.iteration",
        raises=dunder_propagation_claims(
            "iter(x) calls x.__iter__(); next(it) calls it.__next__(). If x has no "
            "__iter__ but DOES have __getitem__, iter(x) instead builds an OLD-STYLE "
            "iterator that calls x.__getitem__(0), x.__getitem__(1), ... until "
            "IndexError, which it translates into a clean StopIteration",
            native={
                "TypeError": "x has neither __iter__ nor __getitem__ ('object is not iterable')",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # __iter__/__next__ (or the old-style __getitem__ fallback) is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=None,
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "StopIteration raised DIRECTLY by __next__ ends a for-loop/comprehension "
            "cleanly (it is caught by the loop machinery, never propagates out of "
            "it). next(it, default) additionally catches StopIteration and returns "
            "default instead of raising; next(it) with no default lets StopIteration "
            "propagate to the caller. Old-style __getitem__ iteration ends cleanly on "
            "EITHER of two signals: IndexError (translated by PySeqIter, the documented "
            "mechanism) OR a StopIteration raised directly by __getitem__ itself (the "
            "general for-loop/comprehension rule above applies here too -- confirmed: "
            "a __getitem__ that raises StopIteration on its first call gives "
            "list(X()) == [] and 5 in X() is False, the same as an immediate "
            "IndexError would). Any OTHER exception from __getitem__ (e.g. a "
            "KeyError) propagates uncaught, it is not treated as end-of-loop. "
            "BATCH 5 (tests/contracts/test_swallowing_native.py, group 'iteration'): a subclass of IndexError matches too; "
            "an IndexError from __iter__ itself or from an iterator's __next__ PROPAGATES (only the sequence protocol "
            "translates it); __iter__ = None blocks the sequence fallback (TypeError); a __iter__ that returns a "
            "non-iterator is TypeError; reversed() ends on IndexError/StopIteration from __getitem__ too. "
            "next(it, default) swallows ONLY StopIteration (a KeyError from __next__ propagates, and a non-iterator is "
            "TypeError even with a default). iter(callable, sentinel) ends when the callable raises StopIteration and "
            "propagates everything else. A StopIteration raised by the function of map() (or by a for-loop's iterator "
            "chain) ends the iteration silently. PEP 479: a StopIteration raised inside a GENERATOR body becomes "
            "RuntimeError. Laziness moves exceptions in time: the first iterable of a generator expression is "
            "evaluated and iter()'d when the generator is CREATED (a TypeError there is not deferred), its body runs "
            "only when consumed; map() applies its function only on consumption. The length probe of list()/sorted()/"
            "list.extend() is protocol.length_hint. dict(obj) asks obj.keys and swallows an AttributeError from that "
            "lookup (it then iterates pairs); {**obj} CONVERTS an AttributeError of the mapping protocol to TypeError."
        ),
        source=(
            f"{_LANG}#object.__iter__",
            "CPython Objects/iterobject.c::PySeqIter_New/iter_iternext (the "
            "IndexError-to-StopIteration translation for old-style iteration)",
        ),
        tested=True,
    )


# ------------------------------------------------------------------
# Hashing
# ------------------------------------------------------------------


@register_rule
class ProtocolHashing(Rule):
    """The rule of ``protocol.hashing``."""

    SAMPLES = {'protocol.hashing': 'class A:\n    pass\nx = hash(A())\n'}
    TWINS = {}

    names = ("protocol.hashing",)
    CONTRACT = Contract(
        qualname="protocol.hashing",
        raises=dunder_propagation_claims(
            "the object's own __hash__, when it is not None (a USER method -- "
            "any exception it raises propagates uncaught, the same as any other "
            "dunder dispatched through a type slot)",
            native={
                "TypeError": (
                    "the object's __hash__ is None. This happens AUTOMATICALLY for "
                    "any class that defines __eq__ without also defining __hash__ "
                    "(Python sets __hash__ = None on that class so it does not "
                    "silently keep the identity-based hash while __eq__ says two "
                    "unequal-looking objects are equal) -- and this reapplies to "
                    "EVERY SUBCLASS that overrides __eq__ again without its own "
                    "__hash__, even if some ancestor further up was hashable"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # __hash__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("int", "hash(x) == hash(y) is REQUIRED whenever x == y (the language does not enforce this; a class breaking it just makes dict/set behave incorrectly, not raise)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft used "
            "never_claims, wrongly marking IndexError/KeyError/AttributeError/"
            "ZeroDivisionError/FloatingPointError/AssertionError as 'never' for a "
            "class with a REAL (non-None) __hash__ that itself raises one of them. "
            "Repro: class X: __hash__ = lambda self: (_ for _ in ()).throw(KeyError) "
            "-- more simply, a __hash__ body that raises KeyError propagates "
            "unchanged from hash(X()). BATCH 5 (tests/contracts/test_swallowing_native.py, group 'hashing'): a dict/set "
            "lookup propagates an exception from __hash__; it calls __eq__ only for two DISTINCT keys with the same hash "
            "(never for the identical object) and propagates what it raises; a dict subclass with __missing__ turns "
            "d[k]'s KeyError into the hook's value (get, pop, setdefault and `in` never call it; its own exception "
            "propagates), defaultdict[k] builds the value with its factory (KeyError only with no factory), Counter[k] "
            "of an absent key is 0, and {}.pop(unhashable) is KeyError (an empty dict never hashes). "
            "collections.abc.Mapping.__contains__/get swallow a KeyError from the user's __getitem__, and "
            "Sequence.__iter__/__contains__/__reversed__ stop on IndexError: Python-level mixins, not C."
        ),
        source=(
            f"{_LANG}#object.__hash__",
            "CPython Objects/typeobject.c (the __eq__-without-__hash__ -> "
            "__hash__=None slot rule, applied at class-creation time)",
        ),
        tested=True,
    )


# ------------------------------------------------------------------
# Truthiness and numeric conversion
# ------------------------------------------------------------------


@register_rule
class ProtocolTruthiness(Rule):
    """The rule of ``protocol.truthiness``."""

    SAMPLES = {'protocol.truthiness': 'class A:\n    def __len__(self):\n        return 1\nif A():\n    pass\n'}
    TWINS = {}

    names = ("protocol.truthiness",)
    CONTRACT = Contract(
        qualname="protocol.truthiness",
        raises=dunder_propagation_claims(
            "__bool__ if defined (tried FIRST, ignoring __len__ even if both are "
            "present); else __len__ (result != 0 is truthy); else True unconditionally",
            native={
                "TypeError": "__bool__ is defined and returns something that is not an EXACT bool (not just falsy/truthy -- e.g. returning an int raises, even 0 or 1)",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # __bool__ (or the __len__ fallback) is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("bool", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "a plain object() with neither dunder is truthy by default -- there is no 'raise if truthiness is undefined' "
            "case; every object has some truth value. BATCH 5 (tests/contracts/test_swallowing_native.py, group "
            "'truthiness'): __bool__ = None is a TypeError and does NOT fall back to __len__; a negative __len__ is "
            "ValueError; a TypeError raised by __len__ in a truth test PROPAGATES (unlike the length hint of "
            "protocol.length_hint, which swallows it); NotImplemented in a boolean context is a TypeError since 3.14, so "
            "`if a.__eq__(b):` raises when __eq__ returned NotImplemented."
        ),
        source=(f"{_LANG}#object.__bool__", "CPython Objects/object.c::PyObject_IsTrue"),
        tested=True,
    )


@register_rule
class ProtocolNumericConversion(Rule):
    """The rule of ``protocol.numeric_conversion``."""

    SAMPLES = {'protocol.numeric_conversion': 'class A:\n    def __int__(self):\n        return 1\nx = int(A())\n'}
    TWINS = {}

    names = ("protocol.numeric_conversion",)
    CONTRACT = Contract(
        qualname="protocol.numeric_conversion",
        raises=dunder_propagation_claims(
            "the object's own __index__ (a USER method -- any exception it raises "
            "propagates uncaught, same as every other dunder here)",
            native={
                "TypeError": (
                    "__index__ is REQUIRED (not __int__, not __float__) for: "
                    "slicing/indexing keys, hex()/oct()/bin(), operator.index(), and "
                    "anywhere the language needs an EXACT integer with no precision "
                    "loss. __int__/__float__ are the LOSSY numeric-conversion "
                    "protocols int()/float() (batch 1) accept in addition to "
                    "__index__. A class with only __int__ raises TypeError from "
                    "hex()/operator.index() even though int(x) would succeed on it; "
                    "or __index__ returns a non-int object, even if that object "
                    "has its own __index__ (no recursive coercion). An int SUBCLASS return is accepted "
                    "(normalized to exact int) with a DeprecationWarning, not a "
                    "TypeError, in 3.14.4 -- confirmed: class I(int): pass; class "
                    "X: __index__ = lambda self: I(1) -- operator.index(X()) == 1, "
                    "type is exact int, plus a DeprecationWarning naming the "
                    "removal-pending behavior. This differs from __bool__'s "
                    "exactness requirement (a bool subclass return -- there are "
                    "none possible since bool cannot be subclassed -- is moot; "
                    "__bool__ returning a non-bool at all, even an int, is a hard "
                    "TypeError with no deprecation path)"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # __index__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("int for __index__; matches int()/float()'s own ReturnSpec for those (batch 1, builtins.py)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "REVIEW FIX: an earlier draft claimed __index__ shares __bool__'s exact-"
            "type requirement -- it does not. __index__ accepts a strict int SUBCLASS "
            "return (normalized to exact int, with a DeprecationWarning -- see the "
            "native TypeError condition above for the confirmed repro); __bool__ "
            "accepts nothing but an exact bool, no deprecation path, a hard TypeError "
            "otherwise. Different from __len__'s requirement too (which accepts "
            "anything with ITS OWN __index__, one level of indirection looser than "
            "either). SOUNDNESS FIX (found by the gatekeeper): an "
            "earlier draft used never_claims, wrongly marking IndexError/KeyError/"
            "AttributeError/ZeroDivisionError/FloatingPointError/AssertionError as "
            "'never' for a class whose __index__ itself raises one of them. Repro: "
            "class X: __index__ = lambda self: 1/0 -- operator.index(X()) raises "
            "ZeroDivisionError, not TypeError."
        ),
        source=(f"{_LANG}#object.__index__", "CPython Objects/abstract.c::PyNumber_Index"),
        tested=True,
    )


# ------------------------------------------------------------------
# Context managers
# ------------------------------------------------------------------


@register_rule
class ProtocolContextManager(Rule):
    """The rule of ``protocol.context_manager``."""

    SAMPLES = {'protocol.context_manager': 'class A:\n    def __enter__(self):\n        return self\n    def __exit__(self, *e):\n        return False\nwith A():\n    pass\n'}
    TWINS = {}

    names = ("protocol.context_manager",)
    CONTRACT = Contract(
        qualname="protocol.context_manager",
        raises=dunder_propagation_claims(
            "__enter__() (called on `with` block entry; its return value binds to "
            "`as` name); then, on block exit, __exit__(exc_type, exc, tb) -- with "
            "(None, None, None) if the block completed cleanly, or the propagating "
            "exception's info otherwise",
            native={
                "TypeError": "the object lacks __enter__ or lacks __exit__ ('does not support the context manager protocol')",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # __enter__/__exit__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=None,
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "__exit__ returning a truthy value SUPPRESSES the exception the block "
            "raised (the `with` statement swallows it entirely -- execution "
            "continues after the block as if nothing happened); a falsy return (the "
            "common case, including None) lets it propagate. If __exit__ ITSELF "
            "raises a new exception, that new exception replaces the original as "
            "what propagates, with the original chained onto its __context__ "
            "(standard implicit exception chaining, not special-cased by `with`). BATCH 5 "
            "(tests/contracts/test_swallowing_native.py, group 'with'): ANY class the block raises is suppressed by a "
            "true __exit__ result, including AssertionError and ZeroDivisionError; the truth test of that result runs "
            "and may itself raise; without an exception the result is ignored; __enter__ raising means __exit__ is "
            "never called; a missing __exit__ or __enter__ is TypeError at the with statement. The same swallowing by "
            "other means: contextlib.suppress(E) (that class only), a @contextmanager generator that catches the "
            "exception thrown in at its yield, a `return` inside `finally`, and an except clause naming a non-exception "
            "(a TypeError only when an exception is matched, replacing it). Exceptions that never end the run: one "
            "raised by __del__ or a weakref callback (unraisable), one that ends a thread."
        ),
        source=(
            f"{_LANG}#object.__exit__",
            "https://docs.python.org/3.14/reference/compound_stmts.html#the-with-statement",
        ),
        tested=True,
    )


__all__ = ["ProtocolBinaryOperator", "ProtocolUnaryOperator", "ProtocolInplaceOperator", "ProtocolComparison", "ProtocolAttributeAccess", "ProtocolContainerLen", "ProtocolLengthHint", "ProtocolContainerGetitem", "ProtocolContainsFallback", "ProtocolIteration", "ProtocolHashing", "ProtocolTruthiness", "ProtocolNumericConversion", "ProtocolContextManager"]
