"""Contracts for ``dict`` methods ([internal ref]).

Cross-checked against Nagini's ``set_dict.sil`` (MPL-2.0, see
``docs/contracts/THIRD_PARTY_REFERENCES.md``): ``dict___getitem__`` requires
``dict___contains__(self, key)`` -- exactly the "key present" boundary tested here.
Nagini has no ``.pop``/``.setdefault``/``.popitem`` model to cross-check (silence, not
a claim).

Every keyed lookup here calls the key's ``__hash__`` first (and ``__eq__`` only on a
hash collision); either can raise anything, and CPython propagates it unchanged --
confirmed per-method below. This is the SAME propagation shape as ``list``'s
``__eq__``-based methods, just with a hash step in front.

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/stdtypes.html#dict"

#: shared native TypeError condition for anything that keys a dict with an
#: arbitrary object: hashing happens before any KeyError-style lookup can occur.
_UNHASHABLE = "the key has no __hash__ (e.g. a list or dict) or __hash__ raises TypeError"


@register_rule
class DictGetitem(Rule):
    """The rule of ``dict.__getitem__``."""

    SAMPLES = {'dict.__getitem__': 'd = {1: 2}\nv = d[1]\n'}
    TWINS = {}

    names = ("dict.__getitem__",)
    CONTRACT = Contract(
        qualname="dict.__getitem__",
        raises=dunder_propagation_claims(
            "the key's __hash__ (always) or __eq__ (only on a hash collision)",
            native={
                "TypeError": _UNHASHABLE,
                "KeyError": "the key is hashable but not present in the receiver",
            },
            tested_native=frozenset({"TypeError", "KeyError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("the value type", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/dictobject.c::dict_subscript"),
        tested=True,
    )


@register_rule
class DictSetitem(Rule):
    """The rule of ``dict.__setitem__``."""

    SAMPLES = {'dict.__setitem__': 'd = {1: 2}\nd[3] = 4\n'}
    TWINS = {}

    names = ("dict.__setitem__",)
    CONTRACT = Contract(
        qualname="dict.__setitem__",
        raises=dunder_propagation_claims(
            "the key's __hash__ (always) or __eq__ (only on a hash collision)",
            native={"TypeError": _UNHASHABLE},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=None,
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/dictobject.c::dict_ass_sub"),
        tested=True,
    )


@register_rule
class DictGet(Rule):
    """The rule of ``dict.get``."""

    SAMPLES = {'dict.get': 'd = {1: 2}\nv = d.get(1)\n'}
    TWINS = {'dict.get': 'd = {1: 2}\nv = d.get()\n'}

    names = ("dict.get",)
    CONTRACT = Contract(
        qualname="dict.get",
        raises=dunder_propagation_claims(
            "the key's __hash__ (always) or __eq__ (only on a hash collision)",
            native={"TypeError": _UNHASHABLE},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("the value type, or the default (None if unspecified)", "never raises KeyError -- that's the point of get()", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/dictobject.c::dict_get_impl"),
        tested=True,
    )


@register_rule
class DictPop(Rule):
    """The rule of ``dict.pop``."""

    SAMPLES = {'dict.pop': 'd = {1: 2}\nv = d.pop(1)\n'}
    TWINS = {'dict.pop': 'd = {1: 2}\nv = d.pop()\n'}

    names = ("dict.pop",)
    CONTRACT = Contract(
        qualname="dict.pop",
        raises=dunder_propagation_claims(
            "the key's __hash__ (only when the receiver is non-empty -- see notes) "
            "or __eq__ (only on a hash collision)",
            native={
                "TypeError": (
                    "the receiver is NON-EMPTY and the key has no __hash__ (or "
                    "__hash__ raises TypeError). An EMPTY receiver skips hashing "
                    "entirely (see notes), so this never fires there"
                ),
                "KeyError": (
                    "no default was given, and either the receiver is empty "
                    "(regardless of the key's hashability) or the key is hashable "
                    "but not present"
                ),
            },
            tested_native=frozenset({"TypeError", "KeyError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("the value type, or the default", "removes and returns the value for key", tested=True),
        effects=Effects(mutates_receiver=True),
        notes=(
            "SURPRISING, confirmed on 3.14.4: dict.pop() on an EMPTY receiver never "
            "hashes the key at all -- {}.pop([]) raises KeyError (not TypeError, "
            "despite the list being unhashable), and {}.pop([], 5) returns 5 with no "
            "exception. The same unhashable key on a NON-EMPTY dict DOES raise "
            "TypeError, with or without a default. dict.__getitem__/.get/.setdefault/"
            "__contains__ all hash unconditionally, even on an empty receiver -- this "
            "short-circuit is unique to .pop()."
        ),
        source=(_DOCS, "CPython Objects/dictobject.c::dict_pop_impl"),
        tested=True,
    )


@register_rule
class DictPopitem(Rule):
    """The rule of ``dict.popitem``."""

    SAMPLES = {'dict.popitem': 'd = {1: 2}\np = d.popitem()\n'}
    TWINS = {'dict.popitem': 'd = {1: 2}\np = d.popitem(0)\n'}

    names = ("dict.popitem",)
    CONTRACT = Contract(
        qualname="dict.popitem",
        raises=never_claims(
            native={"KeyError": "the receiver is empty"},
            tested=frozenset(ALL_PROPERTY_CLASSES),
        ),
        may_raise_other=False,
        returns=ReturnSpec("tuple[key type, value type]", "the LAST inserted (key, value) pair, LIFO order (guaranteed since 3.7 insertion-ordering)", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/dictobject.c::dict_popitem_impl"),
        tested=True,
    )


@register_rule
class DictSetdefault(Rule):
    """The rule of ``dict.setdefault``."""

    SAMPLES = {'dict.setdefault': 'd = {1: 2}\nv = d.setdefault(5, 6)\n'}
    TWINS = {'dict.setdefault': 'd = {1: 2}\nv = d.setdefault()\n'}

    names = ("dict.setdefault",)
    CONTRACT = Contract(
        qualname="dict.setdefault",
        raises=dunder_propagation_claims(
            "the key's __hash__ (always) or __eq__ (only on a hash collision)",
            native={"TypeError": _UNHASHABLE},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("the value type", "the existing value, or default (None if unspecified) after inserting it", tested=True),
        effects=Effects(mutates_receiver=True, notes="mutates only when the key is absent"),
        source=(_DOCS, "CPython Objects/dictobject.c::dict_setdefault_impl"),
        tested=True,
    )


@register_rule
class DictUpdate(Rule):
    """The rule of ``dict.update``."""

    SAMPLES = {'dict.update': 'd = {1: 2}\nd.update({5: 6})\n'}
    TWINS = {'dict.update': 'd = {1: 2}\nd.update({5: 6}, 0)\n'}

    names = ("dict.update",)
    CONTRACT = Contract(
        qualname="dict.update",
        raises=dunder_propagation_claims(
            "the argument's .keys() (if it has one) or its element iteration/hashing/__eq__ otherwise",
            native={
                "TypeError": (
                    "the positional argument has neither .keys() nor supports "
                    "iteration as (key, value) pairs; or a pair-iterable element is "
                    "not itself iterable; or a produced key is unhashable"
                ),
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: a pair-iterable element has the wrong length
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/dictobject.c::dict_update_common"),
        tested=True,
    )


@register_rule
class DictKeys(Rule):
    """The rule of ``dict.keys``."""

    SAMPLES = {'dict.keys': 'd = {1: 2}\nks = list(d.keys())\n'}
    TWINS = {'dict.keys': 'd = {1: 2}\nks = list(d.keys(0))\n'}

    names = ("dict.keys",)
    CONTRACT = Contract(
        qualname="dict.keys",
        raises=never_claims(tested=frozenset(ALL_PROPERTY_CLASSES)),
        may_raise_other=False,
        returns=ReturnSpec("dict_keys", "a live view over the receiver's keys (reflects later mutation)", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/dictobject.c::dict_keys"),
        tested=True,
    )


@register_rule
class DictValues(Rule):
    """The rule of ``dict.values``."""

    SAMPLES = {'dict.values': 'd = {1: 2}\nvs = list(d.values())\n'}
    TWINS = {'dict.values': 'd = {1: 2}\nvs = list(d.values(0))\n'}

    names = ("dict.values",)
    CONTRACT = Contract(
        qualname="dict.values",
        raises=never_claims(tested=frozenset(ALL_PROPERTY_CLASSES)),
        may_raise_other=False,
        returns=ReturnSpec("dict_values", "a live view over the receiver's values", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/dictobject.c::dict_values"),
        tested=True,
    )


@register_rule
class DictItems(Rule):
    """The rule of ``dict.items``."""

    SAMPLES = {'dict.items': 'd = {1: 2}\nfor k, v in d.items():\n    pass\n'}
    TWINS = {'dict.items': 'd = {1: 2}\nfor k, v in d.items(0):\n    pass\n'}

    names = ("dict.items",)
    CONTRACT = Contract(
        qualname="dict.items",
        raises=never_claims(tested=frozenset(ALL_PROPERTY_CLASSES)),
        may_raise_other=False,
        returns=ReturnSpec("dict_items", "a live view over the receiver's (key, value) pairs", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/dictobject.c::dict_items"),
        tested=True,
    )


@register_rule
class DictClear(Rule):
    """The rule of ``dict.clear``."""

    SAMPLES = {'dict.clear': 'd = {1: 2}\nd.clear()\n'}
    TWINS = {'dict.clear': 'd = {1: 2}\nd.clear(0)\n'}

    names = ("dict.clear",)
    CONTRACT = Contract(
        qualname="dict.clear",
        raises=never_claims(tested=frozenset(ALL_PROPERTY_CLASSES)),
        may_raise_other=False,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(_DOCS, "CPython Objects/dictobject.c::dict_clear"),
        tested=True,
    )


@register_rule
class DictCopy(Rule):
    """The rule of ``dict.copy``."""

    SAMPLES = {'dict.copy': 'd = {1: 2}\ne = d.copy()\n'}
    TWINS = {'dict.copy': 'd = {1: 2}\ne = d.copy(0)\n'}

    names = ("dict.copy",)
    CONTRACT = Contract(
        qualname="dict.copy",
        raises=never_claims(tested=frozenset(ALL_PROPERTY_CLASSES)),
        may_raise_other=False,
        returns=ReturnSpec("dict", "a new, shallow-copied dict", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/dictobject.c::dict_copy"),
        tested=True,
    )


__all__ = ["DictGetitem", "DictSetitem", "DictGet", "DictPop", "DictPopitem", "DictSetdefault", "DictUpdate", "DictKeys", "DictValues", "DictItems", "DictClear", "DictCopy"]
