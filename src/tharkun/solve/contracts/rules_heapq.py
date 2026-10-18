"""Contracts for ``heapq`` ([internal ref] batch 3).

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims, never_claims
from tharkun.solve.contracts.rules import Rule, Shape, register_rule
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, ReturnSpec

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
        raises=dunder_propagation_claims(
            "__lt__, comparing the last element against children while sifting it "
            "into the vacated root position -- only needed when the heap has 2+ "
            "elements remaining after the pop",
            native={"IndexError": "the heap (list) is empty"},
            tested_native=frozenset({"IndexError"}),
        ),
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
