"""Engine 3 adapter: ``sv/_symexec.py``'s z3 symbolic executor ([internal ref] step 2).

Pure, stdlib-only contract reads for the two shapes ``_symexec.py``'s
``_Runner.ev_Call`` hook acts on -- the hand-verified, per-qualname tables
themselves live in ``common.py`` (shared with Engine 4's draws adapter, so
concolic's z3 facts and draws' Hypothesis strategies read the exact same
facts, not two tables that could quietly drift apart):

* a RETURN fact for a no-argument call with a TESTED, closed numeric return
  range (the ticket's own example: ``random.random()`` returns a value in
  ``[0, 1)``) -- becomes a z3 range constraint on a fresh symbolic result;
* a LENGTH PRECONDITION for a call whose contract names emptiness of its
  first argument as the exact, tested raise condition for a property class
  (the worked example: ``heapq.heappop`` raises ``IndexError`` iff the heap
  is empty) -- becomes a branch condition (``len(arg) > 0``) the path must
  satisfy to continue past the call, via ``_symexec.py``'s existing
  ``_holds``/``_Dead`` mechanism (unchanged, not reimplemented here);
* (batch 5, [internal ref]) the WHOLE ``math`` module through the shared table in ``math_model.py``
  (:func:`math_function`, :func:`math_constant`): ``sv/_symexec_math.py`` folds a call on concrete
  numbers, encodes the exact ones in z3 and refuses the rest with a reason;
* (batch 4, [internal ref]) a CONCRETE FOLD for a pure builtin called with concrete builtin values
  (:func:`concrete_fold`: ``ord``, ``chr``, ``tuple``, ``list``, ``zip``, ``enumerate``,
  ``reversed``) and the one-argument ``type(x)`` (:func:`type_of_one_argument`) -- the
  executor evaluates them natively instead of raising ``Unsupported`` for the call;
* ([internal ref]) the OUTPUT FILE family: a write-only ``open`` (:func:`output_open_contract`) and the
  methods of the handle it returns (``common.py::output_handle_contract``): the executor runs the
  program past them, ending a path where the contract says the call raises (a wrong-typed
  ``write`` argument) -- the same contracts ``solve/absint`` reads.
"""

from __future__ import annotations

from collections.abc import Callable

from tharkun.solve.contracts.adapters.common import CLOSED_RANGE_RETURN as _RANGE_RETURN
from tharkun.solve.contracts.adapters.common import EMPTY_FIRST_ARG_RAISES as _EMPTY_FIRST_ARG_RAISES
from tharkun.solve.contracts.adapters.common import (
    BUILTIN_SHAPE_CONTRACTS, closed_range_return, empty_first_arg_precondition,
    output_handle_call_shape, output_handle_contract, receiver_length_delta,
)
from tharkun.solve.contracts.adapters.common import resolve as contract_for
from tharkun.solve.contracts.adapters.common import runs_no_callback
from tharkun.solve.contracts.adapters.math_model import constant as math_constant
from tharkun.solve.contracts.adapters.math_model import math_function


def range_return(qualname: str) -> tuple[float, float, bool] | None:
    """``(lo, hi, hi_inclusive)`` for a TESTED no-argument call with a closed
    numeric return range, else ``None``. Concolic's own name for
    :func:`common.closed_range_return`."""
    return closed_range_return(qualname)


#: builtin -> (fewest, most) positional arguments, no keywords, at which the executor may
#: evaluate the call NATIVELY (``None``: no upper bound). Hand-verified against each contract
#: (``tests/contracts/test_builtins_more.py``): ``ord``/``chr``/``type`` take exactly one,
#: ``list``/``tuple`` at most one, ``enumerate`` an iterable and an optional start, ``zip`` any
#: number, ``reversed`` one. ``set`` is absent on purpose: its contract says the iteration
#: ORDER of the result depends on the per-process hash seed, so a value computed here would
#: not be the value the real run sees.
CONCRETE_FOLD_ARITY: dict[str, tuple[int, int | None]] = {
    "ord": (1, 1), "chr": (1, 1), "tuple": (0, 1), "list": (0, 1),
    "zip": (0, None), "enumerate": (1, 2), "reversed": (1, 1),
}
_NATIVE: dict[str, Callable] = {
    "ord": ord, "chr": chr, "tuple": tuple, "list": list, "zip": zip, "enumerate": enumerate,
    "reversed": reversed,
}


def concrete_fold(qualname: str, *, arg_count: int, has_keywords: bool) -> Callable | None:
    """The native builtin the executor may apply to this call's CONCRETE arguments, or ``None``.

    Applies only when the call's shape is :data:`CONCRETE_FOLD_ARITY`'s, the registry holds
    a TESTED contract for ``qualname`` and its effects are those of a pure function of the
    arguments (deterministic, no I/O, no randomness, no clock/environment, no receiver
    mutation). The contract admits callbacks (an argument's ``__iter__``/``__index__``) and,
    for ``list``/``tuple``, advancing an iterator argument; the caller discharges both by
    passing ONLY exact builtin ``list``/``tuple``/``str``/``dict``/``int``/``bool`` values --
    none of which carries a user hook, and none of which a native iterator-advance can reach
    -- and by never inspecting elements (``zip``/``enumerate``/``reversed``/``tuple``/``list``
    do not: tests/contracts/test_builtins_more.py, ``_Sentinel``).
    """
    bounds = CONCRETE_FOLD_ARITY.get(qualname)
    if bounds is None or has_keywords:
        return None
    lo, hi = bounds
    if arg_count < lo or (hi is not None and arg_count > hi):
        return None
    contract = contract_for(qualname)
    if contract is None or not contract.tested:
        return None
    e = contract.effects
    if e.io or e.randomness is not None or e.clock_or_env or not e.deterministic or e.mutates_receiver:
        return None
    return _NATIVE[qualname]


def type_of_one_argument(*, arg_count: int, has_keywords: bool) -> bool:
    """True iff ``type(x)`` may be answered without running anything: exactly one positional
    argument is hand-verified pure native code (``common.py::NO_CALLBACK_SHAPES``). The
    caller maps its OWN value representation to the class (a z3 sort is not a Python type)."""
    return runs_no_callback("type", arg_count=arg_count, has_keywords=has_keywords)


def output_open_contract():
    """The TESTED shape contract of the builtin ``open`` called with a constant write-only mode,
    or ``None`` (missing or untested: the executor then treats ``open`` as unmodelled)."""
    contract = contract_for(BUILTIN_SHAPE_CONTRACTS["open"])
    return contract if contract is not None and contract.tested else None


__all__ = [
    "contract_for", "empty_first_arg_precondition", "range_return", "receiver_length_delta",
    "CONCRETE_FOLD_ARITY", "concrete_fold", "type_of_one_argument", "math_function", "math_constant",
    "output_open_contract", "output_handle_contract", "output_handle_call_shape",
]
