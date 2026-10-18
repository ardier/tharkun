"""Contracts for ``set`` methods ([internal ref]).

Every element-taking method here hashes its argument(s) the same way a ``dict`` key
does (``rules_dict.py``'s ``_UNHASHABLE`` pattern): unhashable raises TypeError, and a
hostile ``__hash__``/``__eq__`` propagates unchanged. ``union``/``intersection``/
``update`` additionally accept ANY iterable (not just another set), so they also
propagate the argument's own ``__iter__``/``__next__``.

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).

[internal ref] added the 24 rules the set family was missing (the rest of the methods, the operators and their in-place forms, the six
comparisons, ``in``, and ``frozenset``), each claim run on CPython 3.14.4 (``tests/contracts/test_set_.py``); absint consumes
them through ``solve/absint/visitor.py`` (``visit_set_*``) and ``solve/absint/sets.py``.
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/stdtypes.html#set"
_UNHASHABLE_ELEM = "an element has no __hash__ (e.g. a list or dict) or __hash__ raises TypeError"


@register_rule
class SetAdd(Rule):
    """The rule of ``set.add``."""

    SAMPLES = {'set.add': 's = set([1, 2])\ns.add(3)\n'}
    TWINS = {'set.add': 's = set([1, 2])\ns.add()\n'}

    names = ("set.add",)
    CONTRACT = Contract(
        qualname="set.add",
        raises=dunder_propagation_claims(
            "the argument's __hash__ (always) or __eq__ (only on a hash collision)",
            native={"TypeError": _UNHASHABLE_ELEM},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/setobject.c::set_add"),
        tested=True,
    )


@register_rule
class SetRemove(Rule):
    """The rule of ``set.remove``."""

    SAMPLES = {'set.remove': 's = set([1, 2])\ns.remove(1)\n'}
    TWINS = {'set.remove': 's = set([1, 2])\ns.remove()\n'}

    names = ("set.remove",)
    CONTRACT = Contract(
        qualname="set.remove",
        raises=dunder_propagation_claims(
            "the argument's __hash__ (always) or __eq__ (only on a hash collision)",
            native={
                "TypeError": _UNHASHABLE_ELEM,
                "KeyError": "the element is hashable but not present in the receiver",
            },
            tested_native=frozenset({"TypeError", "KeyError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/setobject.c::set_discard_key (raising path)"),
        tested=True,
    )


@register_rule
class SetDiscard(Rule):
    """The rule of ``set.discard``."""

    SAMPLES = {'set.discard': 's = set([1, 2])\ns.discard(1)\n'}
    TWINS = {'set.discard': 's = set([1, 2])\ns.discard()\n'}

    names = ("set.discard",)
    CONTRACT = Contract(
        qualname="set.discard",
        raises=dunder_propagation_claims(
            "the argument's __hash__ (always) or __eq__ (only on a hash collision)",
            native={"TypeError": _UNHASHABLE_ELEM},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("None", "removes the element if present; UNLIKE remove(), a missing element is not an error", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/setobject.c::set_discard"),
        tested=True,
    )


@register_rule
class SetPop(Rule):
    """The rule of ``set.pop``."""

    SAMPLES = {'set.pop': 's = set([1, 2])\nv = s.pop()\n'}
    TWINS = {'set.pop': 's = set([1, 2])\nv = s.pop(0)\n'}

    names = ("set.pop",)
    CONTRACT = Contract(
        qualname="set.pop",
        raises=never_claims(
            native={"KeyError": "the receiver is empty"},
            tested=frozenset(ALL_PROPERTY_CLASSES),
        ),
        may_raise_other=False,
        returns=ReturnSpec("the element type", "an ARBITRARY element, removed from the receiver (no ordering guarantee, unlike dict.popitem)", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/setobject.c::set_pop"),
        tested=True,
    )


@register_rule
class SetUnion(Rule):
    """The rule of ``set.union``."""

    SAMPLES = {'set.union': 's = set([1, 2])\nt = s.union([3])\n'}
    TWINS = {'set.union': 's = set([1, 2])\nt = s.union([3], 0)\n'}

    names = ("set.union",)
    CONTRACT = Contract(
        qualname="set.union",
        raises=dunder_propagation_claims(
            "each argument's __iter__/__next__, and each yielded element's __hash__/__eq__",
            native={"TypeError": "an argument is not iterable, or yields an unhashable element"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("set", "a NEW set; does not mutate the receiver or any argument", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/setobject.c::set_union"),
        tested=True,
    )


@register_rule
class SetIntersection(Rule):
    """The rule of ``set.intersection``."""

    SAMPLES = {'set.intersection': 's = set([1, 2])\nt = s.intersection([1])\n'}
    TWINS = {'set.intersection': 's = set([1, 2])\nt = s.intersection([1], 0)\n'}

    names = ("set.intersection",)
    CONTRACT = Contract(
        qualname="set.intersection",
        raises=dunder_propagation_claims(
            "each argument's __iter__/__next__, and each yielded element's __hash__/__eq__",
            native={"TypeError": "an argument is not iterable, or yields an unhashable element"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("set", "a NEW set; does not mutate the receiver or any argument", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/setobject.c::set_intersection_multi"),
        tested=True,
    )


@register_rule
class SetUpdate(Rule):
    """The rule of ``set.update``."""

    SAMPLES = {'set.update': 's = set([1, 2])\ns.update([3])\n'}
    TWINS = {'set.update': 's = set([1, 2])\ns.update([3], 0)\n'}

    names = ("set.update",)
    CONTRACT = Contract(
        qualname="set.update",
        raises=dunder_propagation_claims(
            "each argument's __iter__/__next__, and each yielded element's __hash__/__eq__",
            native={"TypeError": "an argument is not iterable, or yields an unhashable element"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/setobject.c::set_update"),
        tested=True,
    )


@register_rule
class SetClear(Rule):
    """The rule of ``set.clear``."""

    SAMPLES = {'set.clear': 's = set([1, 2])\ns.clear()\n'}
    TWINS = {'set.clear': 's = set([1, 2])\ns.clear(0)\n'}

    names = ("set.clear",)
    CONTRACT = Contract(
        qualname="set.clear",
        raises=never_claims(tested=frozenset(ALL_PROPERTY_CLASSES)),
        may_raise_other=False,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/setobject.c::set_clear"),
        tested=True,
    )


@register_rule
class SetCopy(Rule):
    """The rule of ``set.copy``."""

    SAMPLES = {'set.copy': 's = set([1, 2])\nt = s.copy()\n'}
    TWINS = {'set.copy': 's = set([1, 2])\nt = s.copy(0)\n'}

    names = ("set.copy",)
    CONTRACT = Contract(
        qualname="set.copy",
        raises=never_claims(tested=frozenset(ALL_PROPERTY_CLASSES)),
        may_raise_other=False,
        returns=ReturnSpec("set", "a new, shallow-copied set", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/setobject.c::set_copy"),
        tested=True,
    )


_ITER_ELEM = "each argument's __iter__/__next__, and each yielded element's __hash__/__eq__"
_ITER_TYPEERROR = "an argument is not iterable, or yields an unhashable element (a set is not converted)"
_ELEM_EQ = "an element's __eq__ (only on a hash collision between distinct objects)"
_ARITY = "wrong number of arguments or any keyword argument"
_OPERAND_TYPEERROR = (
    "the other operand is neither a set nor a frozenset and does not implement the reflected operator "
    "(a dict keys/items view does)"
)
_OPERAND_COMPARE_TYPEERROR = (
    "the other operand is neither a set nor a frozenset and does not implement the reflected comparison "
    "(a dict keys/items view does)"
)


def _iterable_method(name, c_func, returns, *, mutates=False, arity_note=""):
    return Contract(
        qualname=f"set.{name}",
        raises=dunder_propagation_claims(
            _ITER_ELEM,
            native={"TypeError": f"{_ITER_TYPEERROR}; also {_ARITY}{arity_note}"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=returns,
        effects=Effects(mutates_receiver=True) if mutates else Effects(deterministic=True),
        source=(_DOCS, f"CPython Objects/setobject.c::{c_func}"),
        tested=True,
    )


def _operator(name, c_func, *, inplace=False):
    returns = (ReturnSpec("set", "the receiver itself (the same object), mutated in place", tested=True) if inplace
               else ReturnSpec("set", "a NEW set (a frozenset when the LEFT operand is a frozenset); neither operand "
                                     "changes", tested=True))
    return Contract(
        qualname=f"set.__{name}__",
        raises=dunder_propagation_claims(
            _ELEM_EQ, native={"TypeError": _OPERAND_TYPEERROR}, tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=returns,
        effects=Effects(mutates_receiver=True) if inplace else Effects(deterministic=True),
        source=(_DOCS, f"CPython Objects/setobject.c::{c_func}"),
        tested=True,
    )


def _comparison(name, c_func):
    return Contract(
        qualname=f"set.__{name}__",
        raises=dunder_propagation_claims(
            _ELEM_EQ, native={"TypeError": _OPERAND_COMPARE_TYPEERROR}, tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("bool", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, f"CPython Objects/setobject.c::{c_func}"),
        tested=True,
    )


_S = "s = set([1, 2])\n"


@register_rule
class SetDifference(Rule):
    """The rule of ``set.difference``."""

    SAMPLES = {"set.difference": _S + "t = s.difference([1])\n"}
    TWINS = {"set.difference": _S + "t = s.difference([1], 0)\n"}
    names = ("set.difference",)
    CONTRACT = _iterable_method("difference", "set_difference_multi",
                                ReturnSpec("set", "a NEW set of the receiver's elements that no argument has; the "
                                                  "receiver is not changed", tested=True))


@register_rule
class SetSymmetricDifference(Rule):
    """The rule of ``set.symmetric_difference``."""

    SAMPLES = {"set.symmetric_difference": _S + "t = s.symmetric_difference([2, 3])\n"}
    TWINS = {"set.symmetric_difference": _S + "t = s.symmetric_difference()\n"}
    names = ("set.symmetric_difference",)
    CONTRACT = _iterable_method("symmetric_difference", "set_symmetric_difference",
                                ReturnSpec("set", "a NEW set of the elements in exactly one of the receiver and the "
                                                  "argument", tested=True), arity_note=" (exactly one argument)")


@register_rule
class SetIssubset(Rule):
    """The rule of ``set.issubset``."""

    SAMPLES = {"set.issubset": _S + "b = s.issubset([1, 2, 3])\n"}
    TWINS = {"set.issubset": _S + "b = s.issubset()\n"}
    names = ("set.issubset",)
    CONTRACT = _iterable_method("issubset", "set_issubset", ReturnSpec("bool", tested=True),
                                arity_note=" (exactly one argument)")


@register_rule
class SetIssuperset(Rule):
    """The rule of ``set.issuperset``."""

    SAMPLES = {"set.issuperset": _S + "b = s.issuperset([1])\n"}
    TWINS = {"set.issuperset": _S + "b = s.issuperset()\n"}
    names = ("set.issuperset",)
    CONTRACT = _iterable_method("issuperset", "set_issuperset", ReturnSpec("bool", tested=True),
                                arity_note=" (exactly one argument)")


@register_rule
class SetIsdisjoint(Rule):
    """The rule of ``set.isdisjoint``."""

    SAMPLES = {"set.isdisjoint": _S + "b = s.isdisjoint([5])\n"}
    TWINS = {"set.isdisjoint": _S + "b = s.isdisjoint()\n"}
    names = ("set.isdisjoint",)
    CONTRACT = _iterable_method("isdisjoint", "set_isdisjoint", ReturnSpec("bool", tested=True),
                                arity_note=" (exactly one argument)")


@register_rule
class SetDifferenceUpdate(Rule):
    """The rule of ``set.difference_update``."""

    SAMPLES = {"set.difference_update": _S + "s.difference_update([1])\n"}
    TWINS = {"set.difference_update": _S + "s.difference_update([1], 0)\n"}
    names = ("set.difference_update",)
    CONTRACT = _iterable_method("difference_update", "set_difference_update", ReturnSpec("None", tested=True),
                                mutates=True)


@register_rule
class SetIntersectionUpdate(Rule):
    """The rule of ``set.intersection_update``."""

    SAMPLES = {"set.intersection_update": _S + "s.intersection_update([1])\n"}
    TWINS = {"set.intersection_update": _S + "s.intersection_update([1], 0)\n"}
    names = ("set.intersection_update",)
    CONTRACT = _iterable_method("intersection_update", "set_intersection_update_multi",
                                ReturnSpec("None", tested=True), mutates=True)


@register_rule
class SetSymmetricDifferenceUpdate(Rule):
    """The rule of ``set.symmetric_difference_update``."""

    SAMPLES = {"set.symmetric_difference_update": _S + "s.symmetric_difference_update([2, 3])\n"}
    TWINS = {"set.symmetric_difference_update": _S + "s.symmetric_difference_update()\n"}
    names = ("set.symmetric_difference_update",)
    CONTRACT = _iterable_method("symmetric_difference_update", "set_symmetric_difference_update",
                                ReturnSpec("None", tested=True), mutates=True, arity_note=" (exactly one argument)")


@register_rule
class SetOr(Rule):
    """The rule of ``set | set``."""

    SAMPLES = {"set.__or__": _S + "t = s | {3}\n"}
    TWINS = {"set.__or__": _S + "t = s | 5\n"}
    names = ("set.__or__",)
    CONTRACT = _operator("or", "set_or")


@register_rule
class SetAnd(Rule):
    """The rule of ``set & set``."""

    SAMPLES = {"set.__and__": _S + "t = s & {2}\n"}
    TWINS = {"set.__and__": _S + "t = s & 5\n"}
    names = ("set.__and__",)
    CONTRACT = _operator("and", "set_and")


@register_rule
class SetSub(Rule):
    """The rule of ``set - set``."""

    SAMPLES = {"set.__sub__": _S + "t = s - {2}\n"}
    TWINS = {"set.__sub__": _S + "t = s - 5\n"}
    names = ("set.__sub__",)
    CONTRACT = _operator("sub", "set_sub")


@register_rule
class SetXor(Rule):
    """The rule of ``set ^ set``."""

    SAMPLES = {"set.__xor__": _S + "t = s ^ {2, 3}\n"}
    TWINS = {"set.__xor__": _S + "t = s ^ 5\n"}
    names = ("set.__xor__",)
    CONTRACT = _operator("xor", "set_xor")


@register_rule
class SetIor(Rule):
    """The rule of ``set |= set``."""

    SAMPLES = {"set.__ior__": _S + "s |= {3}\n"}
    TWINS = {"set.__ior__": _S + "s |= 5\n"}
    names = ("set.__ior__",)
    CONTRACT = _operator("ior", "set_ior", inplace=True)


@register_rule
class SetIand(Rule):
    """The rule of ``set &= set``."""

    SAMPLES = {"set.__iand__": _S + "s &= {2}\n"}
    TWINS = {"set.__iand__": _S + "s &= 5\n"}
    names = ("set.__iand__",)
    CONTRACT = _operator("iand", "set_iand", inplace=True)


@register_rule
class SetIsub(Rule):
    """The rule of ``set -= set``."""

    SAMPLES = {"set.__isub__": _S + "s -= {2}\n"}
    TWINS = {"set.__isub__": _S + "s -= 5\n"}
    names = ("set.__isub__",)
    CONTRACT = _operator("isub", "set_isub", inplace=True)


@register_rule
class SetIxor(Rule):
    """The rule of ``set ^= set``."""

    SAMPLES = {"set.__ixor__": _S + "s ^= {2, 3}\n"}
    TWINS = {"set.__ixor__": _S + "s ^= 5\n"}
    names = ("set.__ixor__",)
    CONTRACT = _operator("ixor", "set_ixor", inplace=True)


@register_rule
class SetLt(Rule):
    """The rule of ``set < set``."""

    SAMPLES = {"set.__lt__": _S + "b = s < {1, 2, 3}\n"}
    TWINS = {"set.__lt__": _S + "b = s < 5\n"}
    names = ("set.__lt__",)
    CONTRACT = _comparison("lt", "set_richcompare (Py_LT)")


@register_rule
class SetLe(Rule):
    """The rule of ``set <= set``."""

    SAMPLES = {"set.__le__": _S + "b = s <= {1, 2, 3}\n"}
    TWINS = {"set.__le__": _S + "b = s <= 5\n"}
    names = ("set.__le__",)
    CONTRACT = _comparison("le", "set_richcompare (Py_LE)")


@register_rule
class SetGt(Rule):
    """The rule of ``set > set``."""

    SAMPLES = {"set.__gt__": _S + "b = s > {1}\n"}
    TWINS = {"set.__gt__": _S + "b = s > 5\n"}
    names = ("set.__gt__",)
    CONTRACT = _comparison("gt", "set_richcompare (Py_GT)")


@register_rule
class SetGe(Rule):
    """The rule of ``set >= set``."""

    SAMPLES = {"set.__ge__": _S + "b = s >= {1}\n"}
    TWINS = {"set.__ge__": _S + "b = s >= 5\n"}
    names = ("set.__ge__",)
    CONTRACT = _comparison("ge", "set_richcompare (Py_GE)")


@register_rule
class SetEq(Rule):
    """The rule of ``set == x``: never an error."""

    SAMPLES = {"set.__eq__": _S + "b = s == {1, 2}\n"}
    names = ("set.__eq__",)
    CONTRACT = Contract(
        qualname="set.__eq__",
        raises=dunder_propagation_claims(_ELEM_EQ),
        may_raise_other=True,
        returns=ReturnSpec("bool", "False for any operand that is not a set or frozenset (never an error); two sets "
                                   "are equal iff they have the same length and every element of one is in the "
                                   "other", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/setobject.c::set_richcompare (Py_EQ)"),
        tested=True,
    )


@register_rule
class SetNe(Rule):
    """The rule of ``set != x``: never an error."""

    SAMPLES = {"set.__ne__": _S + "b = s != {1, 2}\n"}
    names = ("set.__ne__",)
    CONTRACT = Contract(
        qualname="set.__ne__",
        raises=dunder_propagation_claims(_ELEM_EQ),
        may_raise_other=True,
        returns=ReturnSpec("bool", "True for any operand that is not a set or frozenset (never an error)", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/setobject.c::set_richcompare (Py_NE)"),
        tested=True,
    )


@register_rule
class SetContains(Rule):
    """The rule of ``x in set``."""

    SAMPLES = {"set.__contains__": _S + "b = 1 in s\n"}
    TWINS = {"set.__contains__": _S + "b = [1] in s\n"}
    names = ("set.__contains__",)
    CONTRACT = Contract(
        qualname="set.__contains__",
        raises=dunder_propagation_claims(
            "the element's __hash__ (always) or __eq__ (only on a hash collision)",
            native={"TypeError": "the element has no __hash__ (e.g. a list or dict) or __hash__ raises TypeError; "
                                 "a SET element is converted to a frozenset first and is never the error"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("bool", "an element equal to the argument is a member; ``nan in {nan}`` holds only for "
                                   "the same nan object", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/setobject.c::set_contains, set_contains_key"),
        tested=True,
    )


@register_rule
class Frozenset(Rule):
    """The rule of ``frozenset(iterable)``."""

    SAMPLES = {"frozenset": "f = frozenset([1, 2])\n"}
    TWINS = {"frozenset": "f = frozenset([1, 2], 0)\n"}
    names = ("frozenset",)
    CONTRACT = Contract(
        qualname="frozenset",
        raises=dunder_propagation_claims(
            "the argument's __iter__ or __next__ (or __getitem__ under the sequence "
            "protocol), or an element's __hash__ or (on a hash collision) __eq__",
            native={
                "TypeError": (
                    "the argument is not iterable; an element is unhashable (a list, dict or set, or an object "
                    "whose __hash__ is None or returns a non-int); or frozenset is called with more than one "
                    "argument or any keyword"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec(
            "frozenset",
            "the distinct elements of the argument; frozenset() is the empty frozenset; an EXACT frozenset "
            "argument comes back as the same object",
            tested=True,
        ),
        effects=Effects(
            mutates_args=("iterable",),
            clock_or_env=True,
            deterministic=False,
            may_execute_callback=True,
            notes="the iteration ORDER of the result depends on the hash seed for str/bytes elements; an iterator "
                  "argument is advanced to exhaustion",
        ),
        source=(_DOCS, "CPython Objects/setobject.c::frozenset_vectorcall, make_new_frozenset"),
        tested=True,
    )


__all__ = ["SetAdd", "SetRemove", "SetDiscard", "SetPop", "SetUnion", "SetIntersection", "SetUpdate", "SetClear", "SetCopy", "SetDifference", "SetSymmetricDifference", "SetIssubset", "SetIssuperset", "SetIsdisjoint", "SetDifferenceUpdate", "SetIntersectionUpdate", "SetSymmetricDifferenceUpdate", "SetOr", "SetAnd", "SetSub", "SetXor", "SetIor", "SetIand", "SetIsub", "SetIxor", "SetLt", "SetLe", "SetGt", "SetGe", "SetEq", "SetNe", "SetContains", "Frozenset"]
