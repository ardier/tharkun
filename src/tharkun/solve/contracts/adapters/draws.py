"""Engine 4 adapter: Hypothesis strategies + ``hybrid/`` draws ([internal ref] step 2).

Two contract-driven capabilities, both named in the ticket's own words for
this engine:

* "boundary values from the raise conditions (empty list, zero, etc.)" --
  :func:`boundary_value_for_type` offers a type's own empty/zero value as an
  EXTRA candidate a strategy should sample, gated on real evidence: at least
  one TESTED, non-``"never"`` contract on that type's own receiver methods
  actually raising ``IndexError``/``KeyError`` (:func:`common.receiver_may_raise`)
  -- never a blind assumption that "empty" always matters, only ever an
  ADDITION to whatever a caller's strategy already draws.
* "lists that already satisfy the heap property for heap functions" --
  :func:`heap_ordered` turns a list of drawn elements into one
  ``heapq.heapify`` has already put in heap order, gated
  (:func:`heap_ordered_supported`) on a TESTED ``heapq.heapify`` contract
  actually existing in the registry, never assumed.

:func:`range_kwargs` reuses Engine 3's shared ``closed_range_return`` table
(the ticket's own ``random.random`` example) so a mock/patch return value or
a drawn strategy for a contracted no-argument call is realistic instead of a
type-only default.
"""

from __future__ import annotations

from tharkun.solve.contracts.visitors import DrawsVisitor
from tharkun.solve.contracts.adapters.common import (
    ZERO_ARGUMENT_CONSTRUCTORS, closed_range_return, constructor_contract, receiver_may_raise,
)
from tharkun.solve.contracts.registry import get

#: type name -> its own canonical empty/zero value.
_EMPTY_VALUE: dict[str, object] = {
    "list": [], "dict": {}, "set": set(), "frozenset": frozenset(),
    "tuple": (), "str": "", "bytes": b"", "bytearray": bytearray(),
}
#: the two SITE_KINDS-style classes an "empty container" boundary is evidence for.
_EMPTY_EVIDENCE_EXCS = ("IndexError", "KeyError")


def boundary_value_for_type(type_name: str) -> object | None:
    """The canonical empty/zero value for ``type_name``, when TESTED contract
    evidence backs it (some receiver method of this type really can raise
    ``IndexError``/``KeyError``) -- else ``None``. A value here is meant to be
    ADDED to a strategy's existing domain (e.g. ``st.one_of(st.just(v),
    original)``), never to replace or narrow it.
    """
    if type_name not in _EMPTY_VALUE:
        return None
    if not any(receiver_may_raise(type_name, exc, visitor=DrawsVisitor()) for exc in _EMPTY_EVIDENCE_EXCS):
        return None
    if type_name in ZERO_ARGUMENT_CONSTRUCTORS:
        # [internal ref] batch 4: built by calling the contracted constructor, a NEW empty
        # container each time; offered only while that contract is TESTED
        return ZERO_ARGUMENT_CONSTRUCTORS[type_name]() if constructor_contract(type_name, visitor=DrawsVisitor()) else None
    return _EMPTY_VALUE[type_name]


def heap_ordered_supported() -> bool:
    """True iff ``heapq.heapify`` has a TESTED contract in the registry -- the
    fact :func:`heap_ordered` relies on to claim its result is heap-ordered."""
    from tharkun.solve.contracts.rules import Refused, get_rule
    rule = get_rule("heapq.heapify")
    if rule is None:
        return False
    out = rule.accept(DrawsVisitor())
    return False if isinstance(out, Refused) else out


def heap_ordered(values: list) -> list:
    """``values`` reordered into heap order, for a Hypothesis strategy that
    should draw values a heap function (``heapq.heappop``/``heapreplace``/...)
    can consume directly, e.g. ``st.lists(st.integers()).map(heap_ordered)``.
    Callers should check :func:`heap_ordered_supported` first; this function
    itself always uses the real ``heapq.heapify`` (never a re-implementation).
    """
    import heapq

    out = list(values)
    heapq.heapify(out)
    return out


def range_kwargs(qualname: str) -> tuple[float, float, bool] | None:
    """``(lo, hi, hi_inclusive)`` for a TESTED no-argument call's closed return
    range -- see Engine 3's ``adapters.concolic.range_return`` for the same
    fact, read from the same shared table."""
    return closed_range_return(qualname)


__all__ = ["boundary_value_for_type", "heap_ordered_supported", "heap_ordered", "range_kwargs"]
