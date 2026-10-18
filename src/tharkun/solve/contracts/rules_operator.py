"""Contracts for ``operator`` ([internal ref] batch 3): a representative slice, since most of
``operator``'s ~50 functions are one-line wrappers around a protocol batch 1 or batch
2 already contracts (``operator.add`` is ``protocol.binary_operator``'s ``__add__``
under another name; ``operator.index`` is ``protocol.numeric_conversion``'s
``__index__``). ``itemgetter``/``attrgetter`` are contracted here because they are NOT
thin one-liners -- they are callables-that-build-callables with their own
argument-count/dotted-path rules.

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/operator.html"


@register_rule
class OperatorItemgetter(Rule):
    """The rule of ``operator.itemgetter``."""

    SAMPLES = {'operator.itemgetter': 'import operator\nf = operator.itemgetter(0)\n'}
    TWINS = {'operator.itemgetter': 'import operator\nf = operator.itemgetter()\n'}

    names = ("operator.itemgetter",)
    CONTRACT = Contract(
        qualname="operator.itemgetter",
        raises=dunder_propagation_claims(
            "the CALL to the built getter invokes the target's __getitem__ with "
            "each stored index/key -- itemgetter(i)(x) is exactly x[i]. So this "
            "contract's condition is protocol.container_getitem's (batch 2), "
            "reached through an extra layer of indirection",
            native={
                "TypeError": "itemgetter() is called with ZERO index/key arguments at BUILD time (not at call time)",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: the target's __getitem__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec(
            "a callable; CALLING it returns the item type, or a tuple of them for 2+ indices",
            "itemgetter(i)(x) == x[i]; itemgetter(i, j)(x) == (x[i], x[j]) -- the MULTI-index form always returns a tuple, even for exactly one index if written itemgetter(i,) -- confirmed distinct from the single-index form's bare value",
            tested=True,
        ),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=(f"{_DOCS}#operator.itemgetter", "CPython Lib/operator.py::itemgetter / Modules/_operator.c"),
        tested=True,
    )


@register_rule
class OperatorAttrgetter(Rule):
    """The rule of ``operator.attrgetter``."""

    SAMPLES = {'operator.attrgetter': 'import operator\nf = operator.attrgetter("x")\n'}
    TWINS = {'operator.attrgetter': 'import operator\nf = operator.attrgetter()\n'}

    names = ("operator.attrgetter",)
    CONTRACT = Contract(
        qualname="operator.attrgetter",
        raises=dunder_propagation_claims(
            "the CALL to the built getter invokes getattr(x, name) for each "
            "(dotted) name -- attrgetter('a.b')(x) is exactly x.a.b, so this is "
            "protocol.attribute_access (batch 2) through the same indirection "
            "itemgetter has for container_getitem",
            native={
                "TypeError": "attrgetter() is called with ZERO name arguments at BUILD time, or a name is not a str",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: the target's __getattribute__/__getattr__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec(
            "a callable; CALLING it returns the attribute type, or a tuple for 2+ names",
            "attrgetter('a.b')(x) walks EACH dot-separated segment via getattr, left to right -- a missing attribute anywhere along the path raises AttributeError, same as writing x.a.b by hand",
            tested=True,
        ),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=(f"{_DOCS}#operator.attrgetter", "CPython Lib/operator.py::attrgetter / Modules/_operator.c"),
        tested=True,
    )


@register_rule
class OperatorAdd(Rule):
    """The rule of ``operator.add``."""

    SAMPLES = {'operator.add': 'import operator\nx = operator.add(1, 2)\n'}
    TWINS = {'operator.add': 'import operator\nx = operator.add()\n'}

    names = ("operator.add",)
    CONTRACT = Contract(
        qualname="operator.add",
        raises=dunder_propagation_claims(
            "exactly protocol.binary_operator's __add__/__radd__ dispatch (batch 2) -- operator.add(a, b) IS a + b, not a reimplementation",
            native={"TypeError": "neither operand's type supports + with the other's -- identical condition to protocol.binary_operator's"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: __add__/__radd__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("whatever a + b returns", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes="every operator.OP function (sub, mul, lt, contains, getitem, ...) has this exact shape: a thin call to the corresponding protocol dispatch, contracted once each in builtins.py/protocols.py rather than re-contracted per operator-module alias.",
        source=(f"{_DOCS}#operator.add", "CPython Lib/operator.py::add / Modules/_operator.c"),
        tested=True,
    )


__all__ = ["OperatorItemgetter", "OperatorAttrgetter", "OperatorAdd"]
