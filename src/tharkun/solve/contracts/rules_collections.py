"""Contracts for ``collections`` ([internal ref] batch 3): ``deque``, ``defaultdict``,
``Counter``, ``OrderedDict``, ``namedtuple``.

``collections.namedtuple`` is a FACTORY -- it returns a freshly generated class, not
an instance -- so ``collections.namedtuple._replace`` documents the behavior every
namedtuple-generated class's ``_replace`` method has (verified against a
representative generated class in the tests, not a single fixed type the way
``deque``/``Counter`` are).

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, Shape, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/collections.html"
_ALL_TESTED = frozenset(ALL_PROPERTY_CLASSES)


@register_rule
class CollectionsDefaultdict(Rule):
    """The constructor shape of ``collections.defaultdict``."""

    SAMPLES = {"collections.defaultdict": "import collections\nd = collections.defaultdict(list)\n"}
    TWINS = {"collections.defaultdict": "import collections\nd = collections.defaultdict(list, {}, {})\n"}

    names = ("collections.defaultdict",)
    SHAPE = Shape(min_args=0, max_args=2, keywords_ok=True)
    CONTRACT = Contract(
        qualname="collections.defaultdict",
        raises=dunder_propagation_claims(
            "the initializer's __iter__/keys and key __hash__/__eq__",
            native={"TypeError": "wrong call shape, invalid default_factory, or invalid initializer/key"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("collections.defaultdict", "a new dict subclass whose missing-key behavior uses default_factory", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True,
                        notes="an optional mapping/iterable initializer can execute iteration, keys, hash, and equality callbacks"),
        source=(f"{_DOCS}#collections.defaultdict", "CPython Modules/_collectionsmodule.c::defdict_new/dict_init"),
        tested=True,
    )


@register_rule
class CollectionsDequeAppend(Rule):
    """The rule of ``collections.deque.append``."""

    SAMPLES = {'collections.deque.append': 'import collections\nq = collections.deque([1])\nq.append(2)\n'}
    TWINS = {'collections.deque.append': 'import collections\nq = collections.deque([1])\nq.append()\n'}

    names = ("collections.deque.append",)
    CONTRACT = Contract(
        qualname="collections.deque.append",
        raises=never_claims(tested=_ALL_TESTED),
        may_raise_other=False,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True, may_execute_callback=True, notes="if maxlen is set and the deque is full, evicts the leftmost element; releasing its last reference may run __del__ or weakref callbacks. Finalizer exceptions are unraisable, not propagated"),
        source=(f"{_DOCS}#collections.deque.append", "CPython Modules/_collectionsmodule.c::deque_append"),
        tested=True,
    )


@register_rule
class CollectionsDequeAppendleft(Rule):
    """The rule of ``collections.deque.appendleft``."""

    SAMPLES = {'collections.deque.appendleft': 'import collections\nq = collections.deque([1])\nq.appendleft(2)\n'}
    TWINS = {'collections.deque.appendleft': 'import collections\nq = collections.deque([1])\nq.appendleft()\n'}

    names = ("collections.deque.appendleft",)
    CONTRACT = Contract(
        qualname="collections.deque.appendleft",
        raises=never_claims(tested=_ALL_TESTED),
        may_raise_other=False,
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True, may_execute_callback=True, notes="if maxlen is set and full, evicts the rightmost element; releasing its last reference may run __del__ or weakref callbacks. Finalizer exceptions are unraisable, not propagated"),
        source=(f"{_DOCS}#collections.deque.append", "CPython Modules/_collectionsmodule.c::deque_appendleft"),
        tested=True,
    )


@register_rule
class CollectionsDequePop(Rule):
    """The rule of ``collections.deque.pop``."""

    SAMPLES = {'collections.deque.pop': 'import collections\nq = collections.deque([1])\nv = q.pop()\n'}
    TWINS = {'collections.deque.pop': 'import collections\nq = collections.deque([1])\nv = q.pop(0)\n'}

    names = ("collections.deque.pop",)
    CONTRACT = Contract(
        qualname="collections.deque.pop",
        raises=never_claims(
            native={"IndexError": "the deque is empty"},
            tested=_ALL_TESTED,
        ),
        may_raise_other=False,
        returns=ReturnSpec("the element type", "the rightmost element, removed", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(f"{_DOCS}#collections.deque.pop", "CPython Modules/_collectionsmodule.c::deque_pop"),
        tested=True,
    )


@register_rule
class CollectionsDequePopleft(Rule):
    """The rule of ``collections.deque.popleft``."""

    SAMPLES = {'collections.deque.popleft': 'import collections\nq = collections.deque([1])\nv = q.popleft()\n'}
    TWINS = {'collections.deque.popleft': 'import collections\nq = collections.deque([1])\nv = q.popleft(0)\n'}

    names = ("collections.deque.popleft",)
    CONTRACT = Contract(
        qualname="collections.deque.popleft",
        raises=never_claims(
            native={"IndexError": "the deque is empty"},
            tested=_ALL_TESTED,
        ),
        may_raise_other=False,
        returns=ReturnSpec("the element type", "the leftmost element, removed", tested=True),
        effects=Effects(mutates_receiver=True),
        source=(f"{_DOCS}#collections.deque.popleft", "CPython Modules/_collectionsmodule.c::deque_popleft"),
        tested=True,
    )


@register_rule
class CollectionsDequeExtend(Rule):
    """The rule of ``collections.deque.extend``."""

    SAMPLES = {'collections.deque.extend': 'import collections\nq = collections.deque([1])\nq.extend([2])\n'}
    TWINS = {'collections.deque.extend': 'import collections\nq = collections.deque([1])\nq.extend()\n'}

    names = ("collections.deque.extend",)
    CONTRACT = Contract(
        qualname="collections.deque.extend",
        raises=dunder_propagation_claims(
            "the argument's __iter__/__next__",
            native={"TypeError": "the argument is not iterable"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: the argument's __iter__/__next__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True, may_execute_callback=True, notes="if maxlen is set, elements pushed past the limit evict from the LEFT as extend proceeds -- can evict elements this very call just added"),
        source=(f"{_DOCS}#collections.deque.extend", "CPython Modules/_collectionsmodule.c::deque_extend"),
        tested=True,
    )


@register_rule
class CollectionsDequeRotate(Rule):
    """The rule of ``collections.deque.rotate``."""

    SAMPLES = {'collections.deque.rotate': 'import collections\nq = collections.deque([1, 2])\nq.rotate(1)\n'}
    TWINS = {'collections.deque.rotate': 'import collections\nq = collections.deque([1, 2])\nq.rotate(1, 0)\n'}

    names = ("collections.deque.rotate",)
    CONTRACT = Contract(
        qualname="collections.deque.rotate",
        raises=dunder_propagation_claims(
            "n's __index__ (a USER method) -- any exception it raises propagates uncaught",
            native={"TypeError": "n is given and has no __index__"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: n's __index__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("None", "n > 0 rotates right (last element(s) move to the front); n < 0 rotates left; n defaults to 1", tested=True),
        effects=Effects(mutates_receiver=True, may_execute_callback=True),
        notes=(
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft used "
            "never_claims, wrongly marking every class but TypeError as 'never' -- "
            "n's __index__ can raise any of them. Repro: deque([1,2]).rotate(X()) "
            "where X.__index__ raises IndexError propagates IndexError unchanged."
        ),
        source=(f"{_DOCS}#collections.deque.rotate", "CPython Modules/_collectionsmodule.c::deque_rotate"),
        tested=True,
    )


@register_rule
class CollectionsDefaultdictGetitem(Rule):
    """The rule of ``collections.defaultdict.__getitem__``."""

    SAMPLES = {'collections.defaultdict.__getitem__': 'import collections\ndd = collections.defaultdict(int)\nv = dd[1]\n'}
    TWINS = {}

    names = ("collections.defaultdict.__getitem__",)
    CONTRACT = Contract(
        qualname="collections.defaultdict.__getitem__",
        raises=dunder_propagation_claims(
            "default_factory() when the key is absent (via __missing__) -- default_factory's OWN exceptions propagate",
            native={
                "KeyError": "the key is absent AND default_factory is None (the constructor's default) -- the ONLY case this raises; a real factory NEVER lets a missing key raise KeyError",
            },
            tested_native=frozenset({"KeyError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: default_factory is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("the value type, or default_factory()'s return for a missing key", tested=True),
        effects=Effects(mutates_receiver=True, may_execute_callback=True, notes="a missing-key read INSERTS default_factory()'s result into the dict -- unlike dict.get(), reading a defaultdict has a side effect"),
        notes="the factory's CALLABILITY is checked eagerly at defaultdict(factory) construction time (TypeError 'first argument must be callable or None') -- unlike itertools.groupby's key=, which defers its callable check to first use.",
        source=(f"{_DOCS}#collections.defaultdict", "CPython Lib/collections/__init__.py::defaultdict (documented; __missing__ is implemented in C, Modules/_collectionsmodule.c)"),
        tested=True,
    )


@register_rule
class CollectionsCounterGetitem(Rule):
    """The rule of ``collections.Counter.__getitem__``."""

    SAMPLES = {'collections.Counter.__getitem__': 'import collections\nc = collections.Counter("aab")\nv = c["a"]\n'}
    TWINS = {}

    names = ("collections.Counter.__getitem__",)
    CONTRACT = Contract(
        qualname="collections.Counter.__getitem__",
        raises=dunder_propagation_claims(
            "the key's __hash__ (always) or __eq__ (only on a hash collision) -- "
            "looking up whether the key is present still requires hashing it, "
            "even though a miss returns 0 rather than raising",
            native={"TypeError": "the key has no __hash__ (or __hash__ raises TypeError)"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: the key's __hash__/__eq__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("int", "0 for ANY missing key -- NEVER raises KeyError, and (unlike defaultdict) does NOT insert the key either", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft used "
            "never_claims, missing that the lookup still hashes the key -- a "
            "hostile __hash__ propagates just as it does for dict.__getitem__ "
            "(batch 1), even though a well-behaved miss returns 0 instead of "
            "raising KeyError."
        ),
        source=(f"{_DOCS}#collections.Counter", "CPython Lib/collections/__init__.py::Counter.__missing__"),
        tested=True,
    )


@register_rule
class CollectionsCounterMostCommon(Rule):
    """The rule of ``collections.Counter.most_common``."""

    SAMPLES = {'collections.Counter.most_common': 'import collections\nc = collections.Counter("aab")\nv = c.most_common(1)\n'}
    TWINS = {'collections.Counter.most_common': "import collections\nc = collections.Counter('aab')\nv = c.most_common(1, 0)\n"}

    names = ("collections.Counter.most_common",)
    CONTRACT = Contract(
        qualname="collections.Counter.most_common",
        raises=dunder_propagation_claims(
            "n's comparison against the Counter's length (via heapq.nlargest when n is given)",
            native={"TypeError": "n is given and is not orderable against an int"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: n's comparison (via heapq.nlargest) is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("list[tuple[key, int]]", "sorted by count DESCENDING; n omitted or None returns ALL entries sorted, not just the top few", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True, notes="ties are broken by INSERTION ORDER (stable), not arbitrarily"),
        source=(f"{_DOCS}#collections.Counter.most_common", "CPython Lib/collections/__init__.py::Counter.most_common"),
        tested=True,
    )


@register_rule
class CollectionsOrderedDictMoveToEnd(Rule):
    """The rule of ``collections.OrderedDict.move_to_end``."""

    SAMPLES = {'collections.OrderedDict.move_to_end': 'import collections\no = collections.OrderedDict([(1, 2), (3, 4)])\no.move_to_end(1)\n'}
    TWINS = {'collections.OrderedDict.move_to_end': 'import collections\no = collections.OrderedDict([(1, 2), (3, 4)])\no.move_to_end()\n'}

    names = ("collections.OrderedDict.move_to_end",)
    CONTRACT = Contract(
        qualname="collections.OrderedDict.move_to_end",
        raises=dunder_propagation_claims(
            "the key's __hash__ (always) or __eq__ (only on a hash collision) -- same shape as dict.__getitem__ (batch 1)",
            native={
                "TypeError": "the key has no __hash__ (or __hash__ raises TypeError)",
                "KeyError": "the key is hashable but not present",
            },
            tested_native=frozenset({"TypeError", "KeyError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: the key's __hash__/__eq__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("None", "last=True (default) moves the key to the end; last=False moves it to the beginning", tested=True),
        effects=Effects(mutates_receiver=True, may_execute_callback=True),
        notes=(
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft used "
            "never_claims with only KeyError native, missing that looking up the "
            "key hashes it first -- a __hash__ that raises AttributeError (or any "
            "of the other classes) propagates the same way dict.__getitem__'s does."
        ),
        source=(f"{_DOCS}#collections.OrderedDict.move_to_end", "CPython Lib/collections/__init__.py::OrderedDict.move_to_end"),
        tested=True,
    )


@register_rule
class CollectionsNamedtuple(Rule):
    """The rule of ``collections.namedtuple``."""

    SAMPLES = {'collections.namedtuple': 'import collections\nP = collections.namedtuple("P", "x y")\n'}
    TWINS = {'collections.namedtuple': 'import collections\nP = collections.namedtuple()\n'}

    names = ("collections.namedtuple",)
    CONTRACT = Contract(
        qualname="collections.namedtuple",
        raises=dunder_propagation_claims(
            "field_names' __iter__/__next__ when it is an iterable other than a "
            "single space/comma-separated str (a USER method if field_names is a "
            "custom iterable)",
            native={
                "TypeError": "typename or a field name is not a str; or field_names is not iterable/a single space/comma-separated str",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # ValueError: invalid identifier, duplicate field name, or a leading-underscore/keyword field name without rename=True
        returns=ReturnSpec("a new class (a tuple subclass)", "NOT an instance -- calling the RESULT with the field values constructs one", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes=(
            "SOUNDNESS FIX (found by the gatekeeper): an earlier draft used "
            "never_claims, missing that field_names, when passed as a custom "
            "iterable (not the space/comma-separated str shorthand), is iterated "
            "with the object's own __iter__ -- a hostile one propagates whatever "
            "it raises. Repro: namedtuple('Pt', X()) where X.__iter__ raises "
            "KeyError propagates KeyError unchanged."
        ),
        source=(f"{_DOCS}#collections.namedtuple", "CPython Lib/collections/__init__.py::namedtuple"),
        tested=True,
    )


@register_rule
class CollectionsNamedtupleReplace(Rule):
    """The rule of ``collections.namedtuple._replace``."""

    SAMPLES = {'collections.namedtuple._replace': 'import collections\nP = collections.namedtuple("P", "x y")\np = P(1, 2)\nq = p._replace(x=3)\n'}
    TWINS = {'collections.namedtuple._replace': "import collections\nP = collections.namedtuple('P', 'x y')\np = P(1, 2)\nq = p._replace(0)\n"}

    names = ("collections.namedtuple._replace",)
    CONTRACT = Contract(
        qualname="collections.namedtuple._replace",
        raises=dunder_propagation_claims(
            "self._make and iteration over self, which subclasses may override",
            native={"TypeError": "an unknown field name is supplied to the default implementation, or a callback raises TypeError"},
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,
        returns=ReturnSpec("Any", "default _make returns a new instance of the namedtuple class; an overridden _make may return any object", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        notes="The generated method delegates through self._make and iterates self; subclass overrides may change its result, effects and exceptions.",
        source=(f"{_DOCS}#collections.namedtuple", "CPython Lib/collections/__init__.py::namedtuple's generated _replace"),
        tested=True,
    )


__all__ = ["CollectionsDequeAppend", "CollectionsDequeAppendleft", "CollectionsDequePop", "CollectionsDequePopleft", "CollectionsDequeExtend", "CollectionsDequeRotate", "CollectionsDefaultdictGetitem", "CollectionsCounterGetitem", "CollectionsCounterMostCommon", "CollectionsOrderedDictMoveToEnd", "CollectionsNamedtuple", "CollectionsNamedtupleReplace"]
