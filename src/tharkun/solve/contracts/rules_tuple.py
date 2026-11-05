"""Contracts for ``tuple`` methods ([internal ref]).

``tuple`` is immutable, so it has only the two comparison-based methods
(``count``/``index``, same ``__eq__`` propagation shape as ``list``'s -- see
``rules_list.py``) plus ``__getitem__`` (same ``__index__``/bounds shape as
``list.__getitem__``, cross-checked against Nagini's ``tuple.sil``, MPL-2.0).

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec, RaiseArgsCase, ScopedArgsRaiseClaim

_DOCS = "https://docs.python.org/3.14/library/stdtypes.html#tuple"


@register_rule
class TupleIndex(Rule):
    """The rule of ``tuple.index``."""

    SAMPLES = {'tuple.index': 't = (1, 2, 1)\nx = t.index(2)\n'}
    TWINS = {'tuple.index': 't = (1, 2, 1)\nx = t.index()\n'}

    names = ("tuple.index",)
    CONTRACT = Contract(
        qualname="tuple.index",
        raises=dunder_propagation_claims(
            "__eq__ (of each element, compared against the argument)",
            native={"TypeError": "start/stop are given and have no __index__"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: x not in tuple
        returns=ReturnSpec("int", "0 <= result < len(self)", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/tupleobject.c::tuplecontains (index shares the scan)"),
        tested=True,
    )


@register_rule
class TupleCount(Rule):
    """The rule of ``tuple.count``."""

    SAMPLES = {'tuple.count': 't = (1, 2, 1)\nx = t.count(1)\n'}
    TWINS = {'tuple.count': 't = (1, 2, 1)\nx = t.count()\n'}

    names = ("tuple.count",)
    CONTRACT = Contract(
        qualname="tuple.count",
        raises=dunder_propagation_claims(
            "__eq__ (of each element, compared against the argument)",
            native={},
            tested_native=frozenset(),
        ),
        may_raise_other=True,
        returns=ReturnSpec("int", "count(x) >= 0", tested=True),
        effects=Effects(deterministic=True),
        source=(_DOCS, "CPython Objects/tupleobject.c::tuplecount"),
        tested=True,
    )


@register_rule
class TupleGetitem(Rule):
    """The rule of ``tuple.__getitem__``."""

    SAMPLES = {'tuple.__getitem__': 't = (1, 2, 1)\nx = t[0]\n'}
    TWINS = {}

    names = ("tuple.__getitem__",)
    CONTRACT = Contract(
        qualname="tuple.__getitem__",
        raises=tuple(ScopedArgsRaiseClaim(c.exc,c.condition,c.tested,args_cases=(
            RaiseArgsCase('native-bounds','exact builtin receiver and exact int/bool key representable as Py_ssize_t, outside -len(self) <= key < len(self)',
                          ('tuple index out of range',),True),
            RaiseArgsCase('oversized-int','exact builtin receiver and exact int key outside Py_ssize_t range; conversion fails before length adjustment',
                          ("cannot fit 'int' into an index-sized integer",),True),
        )) if c.exc=='IndexError' else c for c in dunder_propagation_claims(
            "the key argument's __index__ (for an int key)",
            native={
                "TypeError": "key is neither an int-like object (has __index__) nor a slice",
                "IndexError": "integer conversion does not fit Py_ssize_t, or converted index is outside [-len(self), len(self)), or key.__index__ raises IndexError",
            },
            tested_native=frozenset({"TypeError", "IndexError"}),
        )),
        may_raise_other=True,
        returns=ReturnSpec("the element type, or tuple for a slice key", tested=True),
        effects=Effects(deterministic=True),
        notes="a slice key never raises IndexError -- out-of-range bounds are clamped, same as list.__getitem__.",
        source=(_DOCS, "https://github.com/python/cpython/blob/3.14/Objects/tupleobject.c", "CPython 3.14::tuplesubscript"),
        tested=True,
    )


__all__ = ["TupleIndex", "TupleCount", "TupleGetitem"]
