"""Contracts for ``copy.copy``/``copy.deepcopy`` ([internal ref] batch 3).

Read straight from ``Lib/copy.py`` (pure Python, no C fallback to guess at): both
functions try, in order, (1) an ATOMIC-type fast path that returns the object AS-IS
(``int``, ``str``, ``tuple``, ``range``, ... -- and, for ``copy.copy`` only,
``list``/``dict``/``set``/``bytearray`` via their own ``.copy()``), (2) the object's
own ``__copy__``/``__deepcopy__``, (3) a registered ``copyreg`` reductor, (4)
``__reduce_ex__(4)``, (5) ``__reduce__``. Step 4 is where "cannot pickle X" comes from
for objects the pickle protocol itself refuses (locks, modules, ...) -- a real,
tested ``TypeError``, not ``copy.Error`` (which only fires when an object has NEITHER
``__reduce_ex__`` NOR ``__reduce__`` at all, which no ordinary object lacks, since
``object`` provides ``__reduce_ex__`` by default -- practically unreachable and out of
scope here).

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/copy.html"


@register_rule
class CopyCopy(Rule):
    """The rule of ``copy.copy``."""

    SAMPLES = {'copy.copy': 'import copy\ny = copy.copy([1])\n'}
    TWINS = {'copy.copy': 'import copy\ny = copy.copy()\n'}

    names = ("copy.copy",)
    CONTRACT = Contract(
        qualname="copy.copy",
        raises=dunder_propagation_claims(
            "the object's __copy__ if defined; else its __reduce_ex__(4)/__reduce__ "
            "(the pickle protocol's own introspection, which some builtin types "
            "-- locks, modules, generators, ... -- refuse outright)",
            native={
                "TypeError": "the object's type refuses pickling ('cannot pickle X object') and has no __copy__ to bypass that",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: __copy__/__reduce_ex__/__reduce__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec(
            "same type as the argument",
            "an ATOMIC type (int, str, tuple, range, frozenset, ...) is returned "
            "UNCHANGED (identity, not a copy); list/dict/set/bytearray use their own "
            ".copy(); everything else is reconstructed via __copy__ or __reduce_ex__",
            tested=True,
        ),
        effects=Effects(deterministic=True, may_execute_callback=True, notes="a genuinely SHALLOW copy: nested mutable objects are shared with the original, not copied"),
        source=(f"{_DOCS}#copy.copy", "CPython Lib/copy.py::copy"),
        tested=True,
    )


@register_rule
class CopyDeepcopy(Rule):
    """The rule of ``copy.deepcopy``."""

    SAMPLES = {'copy.deepcopy': 'import copy\ny = copy.deepcopy([1])\n'}
    TWINS = {'copy.deepcopy': 'import copy\ny = copy.deepcopy()\n'}

    names = ("copy.deepcopy",)
    CONTRACT = Contract(
        qualname="copy.deepcopy",
        raises=dunder_propagation_claims(
            "the object's __deepcopy__(memo) if defined; else its __reduce_ex__(4)/__reduce__",
            native={
                "TypeError": "the object's type refuses pickling and has no __deepcopy__ to bypass that",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: __deepcopy__/__reduce_ex__/__reduce__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec(
            "same type as the argument",
            "atomic types are returned UNCHANGED (same as copy.copy); everything "
            "else is recursively reconstructed. A CYCLE in the object graph "
            "(x.attr is x, directly or through others) is handled correctly via the "
            "memo dict (keyed by id()) -- it does NOT recurse infinitely or duplicate "
            "the shared object",
            tested=True,
        ),
        effects=Effects(deterministic=True, may_execute_callback=True, notes="mutates nothing on the original; the memo dict is an internal implementation detail, not an observable side effect on the caller's objects"),
        source=(f"{_DOCS}#copy.deepcopy", "CPython Lib/copy.py::deepcopy"),
        tested=True,
    )


__all__ = ["CopyCopy", "CopyDeepcopy"]
