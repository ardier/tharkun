"""Contracts for ``bisect`` ([internal ref] batch 3).

``bisect_left``/``bisect_right``/``insort_left``/``insort_right`` all share the same
comparison shape; ``bisect_left`` and ``insort`` (an alias of ``insort_right``) are
contracted here as the representatives -- ``bisect_right`` differs from
``bisect_left`` only in which side of a run of equal elements it inserts at (both
tested in ``tests/contracts/test_bisect_.py`` to confirm the distinction, even though
only one gets its own ``Contract`` entry).

Migrated to rules ([internal ref], [internal ref]): one :class:`~tharkun.solve.contracts.rules.Rule` class per qualname (the visitor kind is
the snake_case of the class name); every ``Contract`` is exactly the one this module held as plain data
(``tests/contracts/test_registry_digest.py`` proves it byte-identical).
"""

from __future__ import annotations

from tharkun.solve.contracts._helpers import dunder_propagation_claims
from tharkun.solve.contracts.rules import Rule, Shape, register_rule
from tharkun.solve.contracts.types import Contract, Effects, ReturnSpec

_DOCS = "https://docs.python.org/3.14/library/bisect.html"


@register_rule
class BisectBisectLeft(Rule):
    """The rule of ``bisect.bisect_left``."""

    SAMPLES = {'bisect.bisect_left': 'import bisect\nx = bisect.bisect_left([1, 2], 2)\n'}
    TWINS = {'bisect.bisect_left': 'import bisect\nx = bisect.bisect_left()\n'}

    names = ("bisect.bisect_left",)
    CONTRACT = Contract(
        qualname="bisect.bisect_left",
        raises=dunder_propagation_claims(
            "__lt__, comparing x against elements of a (binary search)",
            native={
                "TypeError": "a has no __len__ (is not a sequence), or lo/hi have no __index__",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: __lt__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("int", "the LEFTMOST position where x could be inserted keeping a sorted -- if x already occurs, the position BEFORE all equal elements", tested=True),
        effects=Effects(deterministic=True, may_execute_callback=True, notes="a must ALREADY be sorted -- an unsorted a produces a meaningless (but not erroring) result, the same non-detection heapq.heappush has for heap order"),
        source=(f"{_DOCS}#bisect.bisect_left", "CPython Lib/bisect.py::bisect_left / Modules/_bisectmodule.c"),
        tested=True,
    )


@register_rule
class BisectInsort(Rule):
    """The rule of ``bisect.insort``."""

    SAMPLES = {'bisect.insort': 'import bisect\nxs = [1, 3]\nbisect.insort(xs, 2)\n'}
    TWINS = {'bisect.insort': 'import bisect\nxs = [1, 3]\nbisect.insort()\n'}

    names = ("bisect.insort",)
    SHAPE = Shape(min_args=2, max_args=2)      # adapters/common.py::CALL_SHAPE: exactly 2 positional, no keyword
    CONTRACT = Contract(
        qualname="bisect.insort",
        raises=dunder_propagation_claims(
            "__lt__, comparing x against elements of a while locating the insertion point",
            native={
                "TypeError": "a has no __len__, or lo/hi have no __index__",
            },
            tested_native=frozenset({"TypeError"}),
        ),
        may_raise_other=True,  # REVIEW FIX: __lt__ is arbitrary user code; may raise ValueError, a custom exception, etc.
        returns=ReturnSpec("None", "inserts x into a IN PLACE, keeping it sorted; an alias of insort_right -- for equal elements, inserts AFTER them", tested=True),
        effects=Effects(mutates_receiver=True, may_execute_callback=True),
        source=(f"{_DOCS}#bisect.insort", "CPython Lib/bisect.py::insort_right"),
        tested=True,
    )


__all__ = ["BisectBisectLeft", "BisectInsort"]
