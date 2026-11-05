"""Contracts for ``list`` methods ([internal ref]).

Cross-checked against Nagini's ``list.sil`` (MPL-2.0, see
``docs/contracts/THIRD_PARTY_REFERENCES.md``): its Viper preconditions on
``list___getitem__``/``list___setitem__`` require ``-len(self) <= key < len(self)``
(the same boundary this module tests), and ``list_insert``/``list_append`` have NO
such precondition -- matching CPython's own clamping behavior for ``insert`` verified
below. Nagini has no ``list.pop``/``.remove``/``.sort`` model to cross-check against
(silence, not a claim -- see the reference doc).

The comparison-based methods (``remove``, ``index``, ``count``, ``sort``) call the
element's ``__eq__``/``__lt__`` (or the ``key`` function, for ``sort``), and CPython
dispatches to those via the type slot the same way ``len``/``int`` do for their
dunders, so any exception raised there propagates unchanged -- confirmed per-method in
``tests/contracts/test_list_.py``.

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec, RaiseArgsCase, ScopedArgsRaiseClaim

_DOCS = "https://docs.python.org/3.14/library/stdtypes.html#list"


@register_rule
class ListAppend(Rule):
    """The rule of ``list.append``."""

    SAMPLES = {'list.append': 'xs = [3, 1, 2]\nxs.append(4)\n'}
    TWINS = {'list.append': 'xs = [3, 1, 2]\nxs.append()\n'}

    names = ("list.append",)
    CONTRACT = Contract(
        qualname="list.append",
        raises=never_claims(tested=frozenset(ALL_PROPERTY_CLASSES)),
        may_raise_other=False,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True, notes="appends the argument as-is; no protocol call on it"),
        source=(_DOCS, "CPython Objects/listobject.c::listappend"),
        tested=True,
    )


@register_rule
class ListExtend(Rule):
    """The rule of ``list.extend``."""

    SAMPLES = {'list.extend': 'xs = [3, 1, 2]\nxs.extend([4])\n'}
    TWINS = {'list.extend': 'xs = [3, 1, 2]\nxs.extend()\n'}

    names = ("list.extend",)
    CONTRACT = Contract(
        qualname="list.extend",
        raises=dunder_propagation_claims(
            "__iter__ (or __getitem__ for the old-style iteration protocol) of the argument",
            native={"TypeError": "the argument is not iterable"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/listobject.c::list_extend"),
        tested=True,
    )


@register_rule
class ListInsert(Rule):
    """The rule of ``list.insert``."""

    SAMPLES = {'list.insert': 'xs = [3, 1, 2]\nxs.insert(0, 9)\n'}
    TWINS = {'list.insert': 'xs = [3, 1, 2]\nxs.insert()\n'}

    names = ("list.insert",)
    CONTRACT = Contract(
        qualname="list.insert",
        raises=dunder_propagation_claims(
            "the index argument's __index__",
            native={"TypeError": "index has no __index__ (e.g. a str or float)"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True),
        notes=(
            "insert() NEVER raises IndexError: an out-of-range index is clamped to "
            "the nearest valid position (index < -len inserts at 0; index > len "
            "appends at the end) -- confirmed on 3.14.4, matches Nagini's list_insert "
            "having no bounds precondition."
        ),
        source=(_DOCS, "CPython Objects/listobject.c::listinsert"),
        tested=True,
    )


@register_rule
class ListPop(Rule):
    """The rule of ``list.pop``."""

    SAMPLES = {'list.pop': 'xs = [3, 1, 2]\nv = xs.pop()\n'}
    TWINS = {}

    names = ("list.pop",)
    CONTRACT = Contract(
        qualname="list.pop",
        raises=dunder_propagation_claims(
            "the index argument's __index__",
            native={
                "TypeError": "index has no __index__",
                "IndexError": "the receiver is empty, or index (default -1) is out of range [-len, len)",
            },
            tested_native=frozenset({"TypeError", "IndexError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("the element type", "the removed element", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/listobject.c::listpop"),
        tested=True,
    )


@register_rule
class ListRemove(Rule):
    """The rule of ``list.remove``."""

    SAMPLES = {'list.remove': 'xs = [3, 1, 2]\nxs.remove(1)\n'}
    TWINS = {'list.remove': 'xs = [3, 1, 2]\nxs.remove()\n'}

    names = ("list.remove",)
    CONTRACT = Contract(
        qualname="list.remove",
        raises=dunder_propagation_claims(
            "__eq__ (of each element, compared against the argument)",
            native={},
            tested_native=frozenset(),
        ),
        may_raise_other=True,  # ValueError: x not in list
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True),
        notes="removes the first element == x (by value, via __eq__); no element compares as a native TypeError path here -- an incomparable type just never matches (ValueError), it does not itself raise (unless __eq__ does).",
        source=(_DOCS, "CPython Objects/listobject.c::list_remove"),
        tested=True,
    )


@register_rule
class ListIndex(Rule):
    """The rule of ``list.index``."""

    SAMPLES = {'list.index': 'xs = [3, 1, 2]\ni = xs.index(1)\n'}
    TWINS = {'list.index': 'xs = [3, 1, 2]\ni = xs.index()\n'}

    names = ("list.index",)
    CONTRACT = Contract(
        qualname="list.index",
        raises=dunder_propagation_claims(
            "__eq__ (of each element, compared against the argument)",
            native={
                "TypeError": "start/stop are given and have no __index__",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: x not in list[start:stop]
        returns=ReturnSpec("int", "0 <= result < len(self)", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/listobject.c::list_index_impl"),
        tested=True,
    )


@register_rule
class ListCount(Rule):
    """The rule of ``list.count``."""

    SAMPLES = {'list.count': 'xs = [3, 1, 2]\nc = xs.count(1)\n'}
    TWINS = {'list.count': 'xs = [3, 1, 2]\nc = xs.count()\n'}

    names = ("list.count",)
    CONTRACT = Contract(
        qualname="list.count",
        raises=dunder_propagation_claims(
            "__eq__ (of each element, compared against the argument)",
            native={},
            tested_native=frozenset(),
        ),
        may_raise_other=True,
        returns=ReturnSpec("int", "count(x) >= 0", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/listobject.c::list_count"),
        tested=True,
    )


@register_rule
class ListClear(Rule):
    """The rule of ``list.clear``."""

    SAMPLES = {'list.clear': 'xs = [3, 1, 2]\nxs.clear()\n'}
    TWINS = {'list.clear': 'xs = [3, 1, 2]\nxs.clear(0)\n'}

    names = ("list.clear",)
    CONTRACT = Contract(
        qualname="list.clear",
        raises=never_claims(tested=frozenset(ALL_PROPERTY_CLASSES)),
        may_raise_other=False,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/listobject.c::list_clear_impl"),
        tested=True,
    )


@register_rule
class ListCopy(Rule):
    """The rule of ``list.copy``."""

    SAMPLES = {'list.copy': 'xs = [3, 1, 2]\nys = xs.copy()\n'}
    TWINS = {'list.copy': 'xs = [3, 1, 2]\nys = xs.copy(0)\n'}

    names = ("list.copy",)
    CONTRACT = Contract(
        qualname="list.copy",
        raises=never_claims(tested=frozenset(ALL_PROPERTY_CLASSES)),
        may_raise_other=False,
        returns=ReturnSpec("list", "a new, shallow-copied list; elements are shared, not copied", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/listobject.c::list_copy"),
        tested=True,
    )


@register_rule
class ListReverse(Rule):
    """The rule of ``list.reverse``."""

    SAMPLES = {'list.reverse': 'xs = [3, 1, 2]\nxs.reverse()\n'}
    TWINS = {'list.reverse': 'xs = [3, 1, 2]\nxs.reverse(0)\n'}

    names = ("list.reverse",)
    CONTRACT = Contract(
        qualname="list.reverse",
        raises=never_claims(tested=frozenset(ALL_PROPERTY_CLASSES)),
        may_raise_other=False,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True, notes="reverses in place without comparing elements"),
        source=(_DOCS, "CPython Objects/listobject.c::listreverse_impl"),
        tested=True,
    )


@register_rule
class ListSort(Rule):
    """The rule of ``list.sort``."""

    SAMPLES = {'list.sort': 'xs = [3, 1, 2]\nxs.sort()\n'}
    TWINS = {'list.sort': 'xs = [3, 1, 2]\nxs.sort(0)\n'}

    names = ("list.sort",)
    CONTRACT = Contract(
        qualname="list.sort",
        raises=dunder_propagation_claims(
            "the key function (if given) or __lt__ (comparisons between elements)",
            native={"TypeError": "key= is given and is not callable"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("None", "sorts the receiver in place (stable); does not return the list", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/listobject.c::list_sort_impl"),
        tested=True,
    )


@register_rule
class ListGetitem(Rule):
    """The rule of ``list.__getitem__``."""

    SAMPLES = {'list.__getitem__': 'xs = [3, 1, 2]\nx = xs[0]\n'}
    TWINS = {}

    names = ("list.__getitem__",)
    CONTRACT = Contract(
        qualname="list.__getitem__",
        raises=tuple(ScopedArgsRaiseClaim(c.exc,c.condition,c.tested,args_cases=(
            RaiseArgsCase('native-bounds','exact builtin receiver and exact int/bool key representable as Py_ssize_t, outside -len(self) <= key < len(self)',
                          ('list index out of range',),True),
            RaiseArgsCase('oversized-int','exact builtin receiver and exact int key outside Py_ssize_t range; conversion fails before length adjustment',
                          ("cannot fit 'int' into an index-sized integer",),True),
        )) if c.exc=='IndexError' else c for c in dunder_propagation_claims(
            "the key argument's __index__ (for an int key; a slice key never raises IndexError/TypeError-for-bounds)",
            native={
                "TypeError": "key is neither an int-like object (has __index__) nor a slice",
                "IndexError": "integer conversion does not fit Py_ssize_t, or converted index is outside [-len(self), len(self)), or key.__index__ raises IndexError",
            },
            tested_native=frozenset({"TypeError", "IndexError"}),
        )),
        may_raise_other=True,
        returns=ReturnSpec("the element type, or list for a slice key", tested=True),
        effects=Effects(deterministic=True),
        notes="a slice key (list[a:b:c]) never raises IndexError -- out-of-range bounds are clamped, matching Nagini's list___getitem_slice__ having no bounds precondition.",
        source=(_DOCS, "https://github.com/python/cpython/blob/3.14/Objects/listobject.c", "CPython 3.14::list_subscript"),
        tested=True,
    )


@register_rule
class ListSetitem(Rule):
    """The rule of ``list.__setitem__``."""

    SAMPLES = {'list.__setitem__': 'xs = [3, 1, 2]\nxs[0] = 1\n'}
    TWINS = {}

    names = ("list.__setitem__",)
    CONTRACT = Contract(
        qualname="list.__setitem__",
        raises=dunder_propagation_claims(
            "the key argument's __index__ (for an int key)",
            native={
                "TypeError": "key is neither an int-like object nor a slice; or key is a (non-extended) slice and value is not iterable",
                "IndexError": "key is an int-like index outside [-len(self), len(self))",
            },
            tested_native=frozenset({"TypeError", "IndexError"}),
        ),
        may_raise_other=True,  # extended-slice assignment with a length mismatch: ValueError
        returns=None,
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/listobject.c::list_ass_subscript"),
        tested=True,
    )


@register_rule
class ListDelitem(Rule):
    """The rule of ``del xs[key]`` (``list.__delitem__``)."""

    SAMPLES = {'list.__delitem__': 'xs = [3, 1, 2]\ndel xs[0]\n'}
    TWINS = {}

    names = ("list.__delitem__",)
    CONTRACT = Contract(
        qualname="list.__delitem__",
        raises=dunder_propagation_claims(
            "the key argument's __index__ (for an int key, or a component of a slice key)",
            native={
                "TypeError": "key is neither an int-like object (has __index__) nor a slice",
                "IndexError": "integer conversion does not fit Py_ssize_t, or converted index is outside [-len(self), len(self)), or key.__index__ raises IndexError",
            },
            tested_native=frozenset({"TypeError", "IndexError"}),
        ),
        may_raise_other=True,  # a slice with step 0: ValueError
        returns=None,
        effects=Effects(mutates_receiver=True),
        notes="a slice key never raises IndexError (out-of-range bounds clamp) and never raises for a length mismatch, unlike __setitem__.",
        source=(_DOCS, "CPython Objects/listobject.c::list_ass_subscript"),
        tested=True,
    )


@register_rule
class ListContains(Rule):
    """The rule of ``x in xs`` (``list.__contains__``)."""

    SAMPLES = {'list.__contains__': 'xs = [3, 1, 2]\nb = 1 in xs\n'}
    TWINS = {}

    names = ("list.__contains__",)
    CONTRACT = Contract(
        qualname="list.__contains__",
        raises=dunder_propagation_claims(
            "__eq__ (of an element, compared with the argument; identity short-circuits) or the __bool__ of its result",
            native={},
            tested_native=frozenset(),
        ),
        may_raise_other=True,
        returns=ReturnSpec("bool", "True iff some element is the argument or compares equal to it; scans left to right and stops at the first match", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/listobject.c::list_contains"),
        tested=True,
    )


def _list_operator(name, dunder, native, returns, inplace):
    return Contract(
        qualname=f"list.__{name}__",
        raises=dunder_propagation_claims(dunder, native={"TypeError": native}, tested_native=frozenset({"TypeError"})),
        may_raise_other=True,  # MemoryError / OverflowError for a huge repeat count, or whatever a hook raises
        returns=returns,
        effects=Effects(mutates_receiver=True) if inplace else Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/listobject.c::list_concat, list_inplace_concat, list_repeat, list_inplace_repeat"),
        tested=True,
    )


_RIGHT = "the right operand's reflected method (__radd__ / __rmul__ run BEFORE the list's own sequence slot) or __index__ / __iter__"


@register_rule
class ListAdd(Rule):
    """The rule of ``xs + ys`` (``list.__add__``)."""

    SAMPLES = {'list.__add__': 'xs = [1]\nys = xs + [2]\n'}
    TWINS = {'list.__add__': 'xs = [1]\nys = xs + 5\n'}

    names = ("list.__add__",)
    CONTRACT = _list_operator("add", _RIGHT, "the right operand is not a list (a tuple included) and has no __radd__ that accepts the list",
                              ReturnSpec("list", "a NEW list of the elements of both; neither operand changes", tested=True), False)


@register_rule
class ListIadd(Rule):
    """The rule of ``xs += it`` (``list.__iadd__``)."""

    SAMPLES = {'list.__iadd__': 'xs = [1]\nxs += [2]\n'}
    TWINS = {'list.__iadd__': 'xs = [1]\nxs += 5\n'}

    names = ("list.__iadd__",)
    CONTRACT = _list_operator("iadd", "the right operand's __iter__ / __next__ (any iterable is accepted, unlike +)", "the right operand is not iterable",
                              ReturnSpec("list", "the receiver itself (the same object), extended in place", tested=True), True)


@register_rule
class ListMul(Rule):
    """The rule of ``xs * n`` (``list.__mul__``)."""

    SAMPLES = {'list.__mul__': 'xs = [1]\nys = xs * 2\n'}
    TWINS = {'list.__mul__': 'xs = [1]\nys = xs * "a"\n'}

    names = ("list.__mul__",)
    CONTRACT = _list_operator("mul", _RIGHT, "the count has no __index__ (a float, str or list) and no __rmul__ accepts the list",
                              ReturnSpec("list", "a NEW list repeating the elements n times (shared, not copied); [] for n <= 0", tested=True), False)


@register_rule
class ListImul(Rule):
    """The rule of ``xs *= n`` (``list.__imul__``)."""

    SAMPLES = {'list.__imul__': 'xs = [1]\nxs *= 2\n'}
    TWINS = {'list.__imul__': 'xs = [1]\nxs *= "a"\n'}

    names = ("list.__imul__",)
    CONTRACT = _list_operator("imul", _RIGHT, "the count has no __index__ (a float, str or list) and no __rmul__ accepts the list",
                              ReturnSpec("list", "the receiver itself (the same object), repeated in place; cleared for n <= 0", tested=True), True)


__all__ = ["ListAppend", "ListExtend", "ListInsert", "ListPop", "ListRemove", "ListIndex", "ListCount", "ListClear", "ListCopy", "ListReverse", "ListSort", "ListGetitem", "ListSetitem", "ListDelitem", "ListContains", "ListAdd", "ListIadd", "ListMul", "ListImul"]
