"""Contracts for ``heapq`` ([internal ref] batch 3).

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, Shape, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec, RaiseArgsCase, ScopedArgsRaiseClaim

_DOCS = "https://docs.python.org/3.14/library/heapq.html"
_ALL_TESTED = frozenset(ALL_PROPERTY_CLASSES)


@register_rule
class HeapqHeappush(Rule):
    """The rule of ``heapq.heappush``."""

    SAMPLES = {'heapq.heappush': 'import heapq\nh = [1, 2]\nheapq.heappush(h, 0)\n'}
    TWINS = {'heapq.heappush': 'import heapq\nh = [1, 2]\nheapq.heappush()\n'}

    names = ("heapq.heappush",)
    SHAPE = Shape(min_args=2, max_args=2)      # adapters/common.py::CALL_SHAPE: exactly 2 positional, no keyword
    CONTRACT = Contract(
        qualname="heapq.heappush",
        raises=dunder_propagation_claims(
            "__lt__, comparing the new item against existing heap elements while "
            "sifting it into place -- ONLY IF a comparison is actually needed "
            "(pushing onto an empty or single-element heap compares nothing)",
            native={},
            tested_native=frozenset(),
        ),
        may_raise_other=True,  # REVIEW FIX: __lt__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("None", tested=True),
        effects=Effects(mutates_receiver=True, may_execute_callback=True, notes="the list argument must ALREADY be heap-ordered (e.g. via heapify or prior heappush calls) -- pushing onto an arbitrary unordered list corrupts the invariant silently, it is not detected or rejected"),
        source=(f"{_DOCS}#heapq.heappush", "CPython Lib/heapq.py::heappush/_siftdown"),
        tested=True,
    )


@register_rule
class HeapqHeappop(Rule):
    """The rule of ``heapq.heappop``."""

    SAMPLES = {'heapq.heappop': 'import heapq\nh = [1, 2]\nv = heapq.heappop(h)\n'}
    TWINS = {'heapq.heappop': 'import heapq\nh = [1, 2]\nv = heapq.heappop()\n'}

    names = ("heapq.heappop",)
    SHAPE = Shape(min_args=1, max_args=1)      # adapters/common.py::CALL_SHAPE: exactly 1 positional, no keyword
    CONTRACT = Contract(
        qualname="heapq.heappop",
        raises=tuple(ScopedArgsRaiseClaim(c.exc,c.condition,c.tested,
                     args_cases=(RaiseArgsCase('native-empty','native list heap is empty before any comparison',('index out of range',),True),))
                     if c.exc=='IndexError' else c for c in dunder_propagation_claims(
            "__lt__, comparing the last element against children while sifting it "
            "into the vacated root position -- only needed when the heap has 2+ "
            "elements remaining after the pop",
            native={"IndexError": "the heap (list) is empty"},
            tested_native=frozenset({"IndexError"}),
        )),
        may_raise_other=True,  # REVIEW FIX: __lt__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("the element type", "the SMALLEST element, removed", tested=True),
        effects=Effects(mutates_receiver=True, may_execute_callback=True),
        source=(f"{_DOCS}#heapq.heappop", "CPython Lib/heapq.py::heappop"),
        tested=True,
    )


@register_rule
class HeapqHeapify(Rule):
    """The rule of ``heapq.heapify``."""

    SAMPLES = {'heapq.heapify': 'import heapq\nh = [3, 1, 2]\nheapq.heapify(h)\n'}
    TWINS = {'heapq.heapify': 'import heapq\nh = [3, 1, 2]\nheapq.heapify()\n'}

    names = ("heapq.heapify",)
    SHAPE = Shape(min_args=1, max_args=1)      # adapters/common.py::CALL_SHAPE: exactly 1 positional, no keyword
    CONTRACT = Contract(
        qualname="heapq.heapify",
        raises=dunder_propagation_claims(
            "__lt__, comparing elements while reordering the WHOLE list in place",
            native={},
            tested_native=frozenset(),
        ),
        may_raise_other=True,  # REVIEW FIX: __lt__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("None", "reorders x IN PLACE into heap order; does not return the list", tested=True),
        effects=Effects(mutates_receiver=True, may_execute_callback=True),
        source=(f"{_DOCS}#heapq.heapify", "CPython Lib/heapq.py::heapify"),
        tested=True,
    )


@register_rule
class HeapqNlargest(Rule):
    """The rule of ``heapq.nlargest``."""

    SAMPLES = {'heapq.nlargest': 'import heapq\nx = heapq.nlargest(1, [1, 2, 3])\n'}
    TWINS = {'heapq.nlargest': 'import heapq\nx = heapq.nlargest()\n'}

    names = ("heapq.nlargest",)
    CONTRACT = Contract(
        qualname="heapq.nlargest",
        raises=dunder_propagation_claims(
            "n's comparison against the iterable's length, the key function (if "
            "given), and __lt__ between elements while ranking them",
            native={
                "TypeError": "the iterable is not iterable, or n has no comparison support against an int",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: the key function and __lt__ are arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("list", "the n largest elements, DESCENDING; ties broken by original order (stable)", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True, notes="does NOT require the input to already be heap-ordered -- unlike heappush/heappop, it works on any iterable"),
        source=(f"{_DOCS}#heapq.nlargest", "CPython Lib/heapq.py::nlargest"),
        tested=True,
    )


@register_rule
class HeapqNsmallest(Rule):
    """The rule of ``heapq.nsmallest``."""

    SAMPLES = {'heapq.nsmallest': 'import heapq\nx = heapq.nsmallest(1, [1, 2, 3])\n'}
    TWINS = {'heapq.nsmallest': 'import heapq\nx = heapq.nsmallest()\n'}

    names = ("heapq.nsmallest",)
    CONTRACT = Contract(
        qualname="heapq.nsmallest",
        raises=dunder_propagation_claims(
            "n's comparison against the iterable's length, the key function (if "
            "given), and __lt__ between elements",
            native={
                "TypeError": "the iterable is not iterable, or n has no comparison support against an int",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: the key function and __lt__ are arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("list", "the n smallest elements, ASCENDING; ties broken by original order", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True),
        source=(f"{_DOCS}#heapq.nsmallest", "CPython Lib/heapq.py::nsmallest"),
        tested=True,
    )


__all__ = ["HeapqHeappush", "HeapqHeappop", "HeapqHeapify", "HeapqNlargest", "HeapqNsmallest"]


@register_rule
class HeapqMerge(Rule):
    """Construct the lazy merge; consuming it is a separate iterator operation."""

    names = ("heapq.merge",)
    SHAPE = Shape(keywords_ok=True)
    SAMPLES = {"heapq.merge": "import heapq\nresult = heapq.merge([1, 3], [2, 4])\n"}
    TWINS = {"heapq.merge": "import heapq\nresult = heapq.merge(unexpected=True)\n"}
    CONTRACT = Contract(
        qualname="heapq.merge",
        raises=never_claims(native={"TypeError": "argument binding receives a keyword other than key or reverse"},
                            tested=_ALL_TESTED),
        may_raise_other=True,  # Resource failures while allocating the generator are not ruled out.
        returns=ReturnSpec("Iterator[object]", "a fresh generator; input iteration, key calls and comparisons are deferred",
                           tested=True),
        effects=Effects(may_execute_callback=True, notes=(
            "The call only binds arguments and creates a generator. Consumption owns iterator advancement, "
            "reverse truth tests, key/comparison callbacks and their exceptions; retained-operand cleanup may "
            "also run callbacks. These are not constructor safety or purity claims for consumption.")),
        source=(f"{_DOCS}#heapq.merge", "https://github.com/python/cpython/blob/v3.14.4/Lib/heapq.py#L330",
                "https://github.com/python/typeshed/blob/58c33f3da508976e26d681039bb9a255f9509f5c/stdlib/heapq.pyi"),
        tested=True,
    )


NATIVE_HEAP_NAMES=('heapify','heappush','heappop','heapify_max','heappush_max','heappop_max',
                   'heappushpop','heappushpop_max','heapreplace','heapreplace_max')
_NEW_NAMES=NATIVE_HEAP_NAMES[3:]


def _native_heap_contract(name):
    pops=name.startswith(('heappop','heapreplace'))
    q='heapq.'+name
    claims=dunder_propagation_claims('__lt__ comparisons and truth of their results',
        native={'TypeError':'wrong positional-only binding, receiver is not a list, or a needed element comparison is undefined',
                **({'IndexError':'heap is empty before any comparison'} if pops else {})},
        tested_native=frozenset({'TypeError','IndexError'} if pops else {'TypeError'}))
    if pops:
        claims=tuple(ScopedArgsRaiseClaim(c.exc,c.condition,c.tested,
                      args_cases=(RaiseArgsCase('native-empty','native list heap is empty before any comparison',('index out of range',),True),))
                     if c.exc=='IndexError' else c for c in claims)
    return Contract(qualname=q,raises=claims,may_raise_other=True,
        returns=ReturnSpec('None' if name.startswith(('heapify','heappush')) and not name.startswith('heappushpop') else 'object',
                           'None for heapify/push; pop/replace return an original heap element; pushpop can return the new item; extreme-element guarantee requires a correctly ordered heap',tested=True),
        effects=Effects(mutates_receiver=True,may_execute_callback=True,
                        notes='comparison callbacks may raise after partial mutation; no heap-order validation'),
        source=('https://docs.python.org/3.14/library/heapq.html#'+q,
                'https://github.com/python/cpython/blob/3.14/Modules/_heapqmodule.c'),tested=True)


_NATIVE_CONTRACTS={'heapq.'+name:_native_heap_contract(name) for name in _NEW_NAMES}


@register_rule
class HeapqNative(Rule):
    names=tuple('heapq.'+name for name in _NEW_NAMES)
    SAMPLES={q:"import heapq\nh=[7,3]\nx="+q+"(h"+(',5' if q.rsplit('.',1)[1].startswith(('heappush','heapreplace')) else '')+")\n" for q in names}
    TWINS={q:"import heapq\nx="+q+"()\n" for q in names}

    @classmethod
    def shape_for(cls,qualname):
        n=2 if qualname.rsplit('.',1)[1].startswith(('heappush','heapreplace')) else 1
        return Shape(n,n)

    @classmethod
    def contract_for(cls,qualname):
        return _NATIVE_CONTRACTS[qualname]


__all__ += ['HeapqNative', 'NATIVE_HEAP_NAMES']
