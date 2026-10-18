"""Contracts for the builtins the Real-world programs call that batch 1 left out ([internal ref] batch 4).

Same rules as ``builtins.py``: every claim was observed on CPython 3.14.4 (``tests/contracts/
test_builtins_more.py`` runs it), ``raises`` is TOTAL over the seven property classes, and
the two open-world flags (``may_raise_other``, ``effects.may_execute_callback``) are set
independently (``docs/contracts/FORMAT.md``).

Which builtins and why: an AST census of the 24 Real-world programs found these called with
no contract (``ord``/``chr`` in richards, ``tuple`` in unpack_sequence, ``zip``/``enumerate``
in spectral_norm and scimark, ``reversed`` in hexiom, ``set`` in go, ``list`` in 12 programs,
``type`` in chaos).  See ``docs/contracts/REPORT.md`` (Batch 4) for who consumes each.

Two things these contracts say that earlier ones rarely had to:

* The constructors ``list``/``tuple``/``set`` CONSUME their argument now, while ``zip``/
  ``enumerate``/``reversed`` only call ``iter()`` (or look up ``__reversed__``) NOW and run
  the rest lazily when the result is consumed.  A contract here describes the CALL, not the
  consumption of what it returns: ``zip(a, b)``'s per-step ``__next__`` calls (and its
  ``strict=True`` ``ValueError``) belong to whoever iterates the zip object (see
  ``protocol.iteration``).
* ``type`` has a 1-argument form that is pure fixed native code (``type_call``'s special
  case: a metaclass cannot intercept it) and a 3-argument form that creates a class and runs
  metaclass/``__init_subclass__``/``__set_name__`` code.  The contract is the honest union;
  ``adapters/common.py::NO_CALLBACK_SHAPES`` names the exact shape where the stronger fact holds.

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/functions.html"
_ALL_TESTED = frozenset(ALL_PROPERTY_CLASSES)
_TYPEERROR = frozenset({"TypeError"})

# the one TypeError clause every iterable-consuming call shares
_ITER_FAIL = (
    "has neither __iter__ nor __getitem__, __iter__ is None, or __iter__ returns a non-iterator"
)
_NOT_ITERABLE = f"the argument is not iterable (it {_ITER_FAIL})"


@register_rule
class Ord(Rule):
    """The rule of ``ord``."""

    SAMPLES = {'ord': 'x = ord("a")\n'}
    TWINS = {'ord': 'x = ord()\n'}

    names = ("ord",)
    CONTRACT = Contract(
        qualname="ord",
        raises=never_claims(
            {
                "TypeError": (
                    "the argument is not a str, bytes or bytearray (subclasses included) "
                    "of length exactly 1; or ord is called with other than exactly one "
                    "positional argument (ord(c=...) included)"
                ),
            },
            tested=_ALL_TESTED,
        ),
        may_raise_other=False,
        returns=ReturnSpec(
            "int",
            "0 <= result <= 0x10FFFF for a str argument (a lone surrogate included), "
            "0 <= result <= 255 for bytes/bytearray",
            tested=True,
        ),
        effects=Effects(
            may_execute_callback=False,
            notes=(
                "reads the C-level length and first code point; a str/bytes subclass's "
                "__len__/__index__/__getitem__ is never called (tested)"
            ),
        ),
        notes=(
            "may_raise_other=False: ord's only exception is TypeError "
            "(Python/bltinmodule.c::builtin_ord has no other raise), a property class, "
            "so there is nothing outside the seven to enumerate. A wrong length is "
            "TypeError, never ValueError."
        ),
        source=(f"{_DOCS}#ord", "CPython Python/bltinmodule.c::builtin_ord (v3.14.4)"),
        tested=True,
    )


@register_rule
class Chr(Rule):
    """The rule of ``chr``."""

    SAMPLES = {'chr': 'x = chr(65)\n'}
    TWINS = {'chr': 'x = chr()\n'}

    names = ("chr",)
    CONTRACT = Contract(
        qualname="chr",
        raises=dunder_propagation_claims(
            "__index__ (only for an argument that is not an exact int, bool or int subclass)",
            native={
                "TypeError": (
                    "the argument has no __index__ (a float, str or None is rejected), "
                    "its __index__ returns a non-int, or chr is called with other than "
                    "exactly one positional argument (chr(i=...) included)"
                ),
            },
            tested_native=_TYPEERROR,
        ),
        may_raise_other=True,  # ValueError: not in range(0x110000); anything __index__ raises
        returns=ReturnSpec("str", "a str of length exactly 1 (a lone surrogate included)", tested=True),
        effects=Effects(
            may_execute_callback=True,
            notes=(
                "an exact int, a bool or an int subclass runs NO callback: an int "
                "subclass's overridden __index__ is ignored (tested). Only an object that "
                "is none of those reaches __index__"
            ),
        ),
        notes=(
            "an out-of-range value (negative, or >= 0x110000, however large) is "
            "ValueError in 3.14, not OverflowError and not a property class."
        ),
        source=(f"{_DOCS}#chr", "CPython Python/bltinmodule.c::builtin_chr, Objects/unicodeobject.c::PyUnicode_FromOrdinal (v3.14.4)"),
        tested=True,
    )


@register_rule
class Type(Rule):
    """The rule of ``type``."""

    SAMPLES = {'type': 'x = type(3)\n'}
    TWINS = {'type': 'x = type()\n'}

    names = ("type",)
    CONTRACT = Contract(
        qualname="type",
        raises=dunder_propagation_claims(
            "the 3-argument form's metaclass __new__/__init__, a base's __init_subclass__, "
            "or a namespace value's __set_name__ (the 1-argument form runs no code)",
            native={
                "TypeError": (
                    "called with 0 or 2 arguments, or with any keyword; or, 3-argument "
                    "form, a name that is not a str, bases that is not a tuple of types, a "
                    "namespace that is not a dict, a metaclass conflict, an instance "
                    "layout conflict, or a bad __slots__. type(x) with exactly one "
                    "positional argument NEVER raises, for any x"
                ),
            },
            tested_native=_TYPEERROR,
        ),
        may_raise_other=True,  # 3-argument form: ValueError (null in name), any hook's exception
        returns=ReturnSpec(
            "object",
            "1 argument: the exact class of x (a __class__ property or a metaclass "
            "__call__ cannot change it); 3 arguments: normally a new class, but a base's "
            "metaclass __new__ may return any object",
            tested=True,
        ),
        effects=Effects(
            may_execute_callback=True,
            notes=(
                "3-argument form creates a class (and registers it in each base's "
                "__subclasses__); the 1-argument form is pure fixed native code, see "
                "adapters/common.py::NO_CALLBACK_SHAPES"
            ),
        ),
        notes=(
            "type(object=x) is a TypeError (type() takes 1 or 3 arguments): the "
            "argument is positional-only. A 1-argument type() ignores a hostile __class__ "
            "property and a metaclass __call__ because the callee is `type` itself."
        ),
        source=(f"{_DOCS}#type", "CPython Objects/typeobject.c::type_call, type_new (v3.14.4)"),
        tested=True,
    )


@register_rule
class List(Rule):
    """The rule of ``list``."""

    SAMPLES = {'list': 'x = list(range(3))\n'}
    TWINS = {'list': 'x = list(range(3), 0)\n'}

    names = ("list",)
    CONTRACT = Contract(
        qualname="list",
        raises=dunder_propagation_claims(
            "the argument's __iter__ or __next__ (or __getitem__ under the sequence "
            "protocol), or its __len__/__length_hint__ (PyObject_LengthHint)",
            native={
                "TypeError": (
                    f"{_NOT_ITERABLE}; or list is called with more than one argument or "
                    "any keyword (list(iterable=x) included)"
                ),
            },
            tested_native=_TYPEERROR,
        ),
        may_raise_other=True,  # ValueError: negative __length_hint__; anything a hook raises
        returns=ReturnSpec(
            "list",
            "a NEW list of the argument's elements in iteration order; list() is []; "
            "list(l) is never l itself",
            tested=True,
        ),
        effects=Effects(
            mutates_args=("iterable",),
            may_execute_callback=True,
            notes=(
                "an iterator/generator argument is advanced to exhaustion; a list, tuple, "
                "str, dict or range argument is not modified"
            ),
        ),
        notes=(
            "iteration ends silently on StopIteration from __next__, and on IndexError "
            "from a sequence-protocol __getitem__ (an IndexError from __iter__ or "
            "__next__ propagates). A __len__ or __length_hint__ raising TypeError is "
            "swallowed; any other class propagates (list only: tuple and set do not "
            "probe the length in 3.14.4)."
        ),
        source=(f"{_DOCS}#func-list", "CPython Objects/listobject.c::list___init___impl, list_extend (v3.14.4)"),
        tested=True,
    )


@register_rule
class Tuple(Rule):
    """The rule of ``tuple``."""

    SAMPLES = {'tuple': 'x = tuple([1, 2])\n'}
    TWINS = {'tuple': 'x = tuple([1, 2], 0)\n'}

    names = ("tuple",)
    CONTRACT = Contract(
        qualname="tuple",
        raises=dunder_propagation_claims(
            "the argument's __iter__ or __next__ (or __getitem__ under the sequence protocol)",
            native={
                "TypeError": (
                    f"{_NOT_ITERABLE}; or tuple is called with more than one argument or "
                    "any keyword (tuple(iterable=x) included)"
                ),
            },
            tested_native=_TYPEERROR,
        ),
        may_raise_other=True,  # anything a hook raises
        returns=ReturnSpec(
            "tuple",
            "a tuple of the argument's elements in iteration order; tuple() is (); an "
            "EXACT tuple argument comes back as the same object (a tuple subclass is copied)",
            tested=True,
        ),
        effects=Effects(
            mutates_args=("iterable",),
            may_execute_callback=True,
            notes=(
                "an iterator/generator argument is advanced to exhaustion; any other "
                "argument is not modified"
            ),
        ),
        notes="no length probe: a hostile __len__/__length_hint__ is never called (tested), unlike list.",
        source=(f"{_DOCS}#func-tuple", "CPython Objects/tupleobject.c::tuple_new_impl, Objects/abstract.c::PySequence_Tuple (v3.14.4)"),
        tested=True,
    )


@register_rule
class Set(Rule):
    """The rule of ``set``."""

    SAMPLES = {'set': 's = set([1, 2])\n'}
    TWINS = {'set': 's = set([1, 2], 0)\n'}

    names = ("set",)
    CONTRACT = Contract(
        qualname="set",
        raises=dunder_propagation_claims(
            "the argument's __iter__ or __next__ (or __getitem__ under the sequence "
            "protocol), or an element's __hash__ or (on a hash collision) __eq__",
            native={
                "TypeError": (
                    f"{_NOT_ITERABLE}; an element is unhashable (a list, dict or set, or "
                    "an object whose __hash__ is None or returns a non-int); or set is "
                    "called with more than one argument or any keyword (set(iterable=x) "
                    "included)"
                ),
            },
            tested_native=_TYPEERROR,
        ),
        may_raise_other=True,  # anything a hook raises
        returns=ReturnSpec(
            "set",
            "a NEW set of the distinct elements (by hash, then __eq__); set() is the "
            "empty set; membership, len and equality of the result do not depend on the "
            "hash seed",
            tested=True,
        ),
        effects=Effects(
            mutates_args=("iterable",),
            clock_or_env=True,
            deterministic=False,
            may_execute_callback=True,
            notes=(
                "ONLY the iteration ORDER of the result can differ between two processes: "
                "str/bytes (and tuples or frozensets of them) hash through the per-process "
                "hash seed (PYTHONHASHSEED). An iterator argument is advanced to exhaustion"
            ),
        ),
        notes=(
            "a consumer that never iterates the result (membership, len, equality, a "
            "set operator) may ignore the order dependence; one that does must not "
            "treat the result as a function of the arguments alone."
        ),
        source=(f"{_DOCS}#func-set", "CPython Objects/setobject.c::set_vectorcall, set_update_internal (v3.14.4)"),
        tested=True,
    )


@register_rule
class Zip(Rule):
    """The rule of ``zip``."""

    SAMPLES = {'zip': 'for a, b in zip([1], [2]):\n    pass\n'}
    TWINS = {'zip': 'for a, b in zip([1], [2], 0):\n    pass\n'}

    names = ("zip",)
    CONTRACT = Contract(
        qualname="zip",
        raises=dunder_propagation_claims(
            "each argument's __iter__ (called when zip is called; __getitem__ supplies "
            "the iterator under the sequence protocol), or the strict value's __bool__",
            native={
                "TypeError": (
                    f"an argument is not iterable (it {_ITER_FAIL}); "
                    "or a keyword other than strict is given"
                ),
            },
            tested_native=_TYPEERROR,
        ),
        may_raise_other=True,  # anything a hook raises
        returns=ReturnSpec(
            "zip",
            "a LAZY iterator of tuples, one element from each argument iterator per step, "
            "stopping at the shortest; zip() is empty. Nothing is consumed when zip is "
            "called",
            tested=True,
        ),
        effects=Effects(
            may_execute_callback=True,
            notes=(
                "the call itself only runs iter() on each argument; the arguments' "
                "__next__ run later, when the zip object is consumed, and advance any "
                "iterator argument then (not now). strict=True raises ValueError, also "
                "at consumption"
            ),
        ),
        notes=(
            "zip(x, strict=1) is accepted (any truthy/falsy value); an unequal-length "
            "strict zip is only a ValueError when iterated, never when constructed. This "
            "contract does not cover the consumption: see protocol.iteration."
        ),
        source=(f"{_DOCS}#zip", "CPython Python/bltinmodule.c::zip_new (v3.14.4)"),
        tested=True,
    )


@register_rule
class Enumerate(Rule):
    """The rule of ``enumerate``."""

    SAMPLES = {'enumerate': 'for i, v in enumerate([1, 2]):\n    pass\n'}
    TWINS = {'enumerate': 'for i, v in enumerate():\n    pass\n'}

    names = ("enumerate",)
    CONTRACT = Contract(
        qualname="enumerate",
        raises=dunder_propagation_claims(
            "the iterable's __iter__ (called when enumerate is called; __getitem__ "
            "supplies the iterator under the sequence protocol), or start's __index__",
            native={
                "TypeError": (
                    f"the iterable is not iterable (it {_ITER_FAIL}); "
                    "start has no __index__ (a float, str or None) or its __index__ "
                    "returns a non-int; or more than two arguments, or a keyword other "
                    "than iterable/start (both ARE accepted as keywords)"
                ),
            },
            tested_native=_TYPEERROR,
        ),
        may_raise_other=True,  # anything a hook raises
        returns=ReturnSpec(
            "enumerate",
            "a LAZY iterator of (index, element) pairs, index = start + position with "
            "start defaulting to 0 and unbounded (an arbitrarily large int is fine); "
            "nothing is consumed when enumerate is called",
            tested=True,
        ),
        effects=Effects(
            may_execute_callback=True,
            notes=(
                "an exact int, bool or int subclass as start runs no callback (an int "
                "subclass's overridden __index__ is ignored, tested); the iterable's "
                "__next__ runs later, at consumption"
            ),
        ),
        notes="enumerate(iterable=x) and enumerate(x, start=n) are valid; enumerate() with no iterable is a TypeError.",
        source=(f"{_DOCS}#enumerate", "CPython Objects/enumobject.c::enum_new_impl (v3.14.4)"),
        tested=True,
    )


@register_rule
class Reversed(Rule):
    """The rule of ``reversed``."""

    SAMPLES = {'reversed': 'for v in reversed([1, 2]):\n    pass\n'}
    TWINS = {'reversed': 'for v in reversed():\n    pass\n'}

    names = ("reversed",)
    CONTRACT = Contract(
        qualname="reversed",
        raises=dunder_propagation_claims(
            "the argument's __reversed__, or (when it defines none) its __len__ (called "
            "when reversed is called); a sequence-protocol __getitem__ runs later, at "
            "consumption",
            native={
                "TypeError": (
                    "the argument defines no __reversed__ (or sets it to None) and is not "
                    "a sequence (no __len__ or no __getitem__) -- a set, an iterator or a "
                    "generator is not reversible; or reversed is called with other than "
                    "exactly one positional argument (reversed(seq=...) included)"
                ),
            },
            tested_native=_TYPEERROR,
        ),
        may_raise_other=True,  # anything a hook raises
        returns=ReturnSpec(
            "object",
            "a lazy reversed iterator over a sequence, a dict, a dict keys/values/items "
            "view or a range; but an argument that defines __reversed__ decides the "
            "result, which may be any object",
            tested=True,
        ),
        effects=Effects(
            may_execute_callback=True,
            notes=(
                "a list/str/range/dict/tuple argument runs no callback; the reversed object "
                "keeps a reference to the sequence and reads each element when it is "
                "consumed (a later element assignment is visible), but its starting index "
                "is fixed by len() at the call (a later append is not)"
            ),
        ),
        notes=(
            "an argument's own __reversed__ is preferred over the sequence protocol even "
            "for a list/str subclass, and can raise or return anything. Consumption "
            "(__getitem__ per step) is not covered here."
        ),
        source=(f"{_DOCS}#reversed", "CPython Objects/enumobject.c::reversed_new_impl (v3.14.4)"),
        tested=True,
    )


__all__ = ["Ord", "Chr", "Type", "List", "Tuple", "Set", "Zip", "Enumerate", "Reversed"]
