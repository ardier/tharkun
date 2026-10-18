"""Engine 2 adapter: ``solve/absint/``'s call transfer function ([internal ref] step 2).

absint's own ``MODULES`` table (``solve/absint/interp.py::MODULES``) names only
``math``/``random``/``_sv_verifier``; a call to any other module -- or an
untabled function of a known one -- makes ``module_attr`` return
``V(top=True)``, and the generic "unmodelled call" path eventually raises
``Unsupported``, which ABORTS THE WHOLE PROOF (``prove()``'s only catch site,
``solve/absint/__init__.py``): one untabled call anywhere in the program turns
the entire analysis ``UNKNOWN``, even along paths irrelevant to the property
being decided.

This adapter lets ``interp.py`` avoid that abort when a TESTED contract exists
for the call, in exchange for recording an HONEST per-exception-class check at
that site instead of skipping the site's exceptions altogether: a "never"
claim becomes a PROVEN check (``ok=True``); anything else (a real condition,
``"unknown"``, or simply untested) becomes an UNPROVEN check (``ok=False``) --
never silently omitted, because ``prove()``'s soundness rests on every real
raise site having SOME recorded check (a site raise silently left unchecked
could let the whole-program sweep wrongly conclude TRUE).
"""

from __future__ import annotations

from tharkun.solve.contracts.adapters.common import (
    BUILTIN_SHAPE_CONTRACTS, OUTPUT_HANDLE_TYPES, applicable_call_shape, empty_first_arg_precondition,
    length_hint_rule, never_raises, output_handle_call_shape, output_handle_contract, raise_condition,
    receiver_length_delta,
)
from tharkun.solve.contracts.adapters.math_model import constant as math_constant
from tharkun.solve.contracts.adapters.math_model import math_function, modelled_names, scalar_shape_raises
from tharkun.solve.contracts.registry import get
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract


def contract_for_module_func(mod: str, name: str) -> Contract | None:
    """The contract for a module-level function call (``heapq.heappop`` ->
    ``get("heapq.heappop")``) -- the shape ``module_attr``/``call_static`` already
    resolve a module call to."""
    return get(f"{mod}.{name}")


def contract_for_builtin(name: str) -> Contract | None:
    """The TESTED contract for a call to the builtin ``name`` -- the gate a contract-driven
    absint model (``interp.py::Analyzer.c_<name>``, run by ``_contract_builtin_call``) is
    entered through. ``None`` (untested, missing) makes the call the same
    ``V(top=True)``/abort it was before any contract existed."""
    contract = get(BUILTIN_SHAPE_CONTRACTS.get(name, name))
    return contract if contract is not None and contract.tested else None


def contract_for_method(owner: str, name: str) -> Contract | None:
    """The TESTED contract of the method (or operator) ``owner.name`` of a builtin type (``set.add``,
    ``set.__or__``) -- the gate a contract-driven absint model of a method of a heap kind
    (``solve/absint/sets.py``) is entered through.  ``None`` (missing or untested) makes the call
    ``Unsupported``: no model reads a contract that is not tested.  A frozenset has no contracts of its own for
    the methods and operators it shares with a set (one C function each; ``tests/contracts/test_set_.py`` runs
    every one on both types), so its calls read ``set.<name>``."""
    contract = get(f"{owner}.{name}")
    return contract if contract is not None and contract.tested else None


def raise_checks(contract: Contract) -> tuple[tuple[str, bool, str], ...]:
    """``(exc, ok, fact)`` for every one of the 7 property classes absint tracks --
    TOTAL over :data:`ALL_PROPERTY_CLASSES`, so a caller recording every entry can
    never leave a real raise site unchecked. ``ok`` is True only for a TESTED
    "never" claim; every other case (a real, tested condition; ``"unknown"``; or
    simply untested) is ``ok=False`` -- sound by construction, whatever the
    contract turns out to say.
    """
    out = []
    for exc in ALL_PROPERTY_CLASSES:
        if never_raises(contract, exc):
            out.append((exc, True, "contract: tested never to raise"))
            continue
        condition = raise_condition(contract, exc)
        if condition is not None:
            out.append((exc, False, f"contract: may raise if {condition}"))
        else:
            out.append((exc, False, "contract: untested/unknown for this class"))
    return tuple(out)


def simple_return_kind(contract: Contract) -> str | None:
    """``'float'``/``'str'``/``'bool'`` when the TESTED return type is exactly
    that AND the contract cannot run an unmodelled callback, else ``None``.

    Deliberately narrow: absint has ready-made, already-sound "any value of this
    type" constants for exactly these three (``FLOAT``, ``ANYSTR``, ``BOOL`` in
    ``aval.py``) -- there is no analogous unconstrained-int constant in this
    codebase (every int value here carries zone/linear-form bounds), and
    ``ReturnSpec.shape`` is free-text English, not a machine-readable range, so
    deriving anything more precise than the bare type risks an unsound reading.
    ``int`` (and anything else) falls back to the caller's existing
    ``V(top=True)`` -- no less sound than before this adapter existed, just not
    more precise yet.

    2026-09-28 owner review: ``effects.may_execute_callback=True`` means "any
    object can come back" -- the callable's OWN dispatch may be exactly the
    documented type, but the arbitrary code it invokes (a ``key=``/``default=``
    callback, a dunder) is not bound by that claim at all (``types.py``'s own
    ``Effects.may_execute_callback`` docstring: "a consumer MUST NOT treat...
    as a... certificate for the WHOLE operation when this is True"). A declared
    return type is exactly such a certificate, so it is never trusted here when
    this flag is set, whatever the type string says.
    """
    if contract.effects.may_execute_callback:
        return None
    if contract.returns is None or not contract.returns.tested:
        return None
    t = contract.returns.type.strip()
    return t if t in ("float", "str", "bool") else None


__all__ = [
    "contract_for_module_func", "contract_for_builtin", "contract_for_method", "raise_checks", "simple_return_kind",
    "applicable_call_shape", "empty_first_arg_precondition", "receiver_length_delta",
    "output_handle_contract", "output_handle_call_shape", "OUTPUT_HANDLE_TYPES", "length_hint_rule",
    "math_function", "math_constant", "modelled_names", "scalar_shape_raises",
]
