"""Shared, stdlib-only contract-reading helpers every per-engine adapter builds on.

Pure translation from a :class:`~tharkun.solve.contracts.types.Contract` to the
plain facts an engine can safely act on -- nothing here executes a target,
imports a third-party library, or imports a tharkun module outside ``solve/``
(the SOLVE layer's charter, ``docs/LAYERS.md``).

The one hard safety rule every function here enforces: a claim is usable only
when it is ``tested=True``. An untested, missing, or ``"unknown"`` claim yields
``None`` / ``False``, and every caller MUST treat that as "may raise" / "nothing
known" -- never as a negative claim. This module only reads that gate; it never
weakens it.

2026-09-28 owner review, second hard rule: a precise, per-qualname fact below is
usable only for a call whose ACTUAL shape (positional argument count, no
keywords) matches :data:`CALL_SHAPE`'s hand-verified expectation for that
qualname. A contract's claim implicitly assumes it is being called the way its
own signature documents it; Python itself raises ``TypeError`` for a call that
doesn't match a callable's signature, before the callable's own body -- and
therefore before whatever the contract's *other* claims describe -- ever runs.
Trusting a claim (a raise condition, a return type, an effect) against a call
whose shape was never verified would let that mismatch through unnoticed. A
qualname absent from :data:`CALL_SHAPE`, or a call whose shape doesn't match,
gets NONE of the precise facts below -- always the same conservative reading
as if no contract existed for this call at all.
"""

from __future__ import annotations

from collections.abc import Callable

from tharkun.solve.contracts.registry import all_contracts, get
from tharkun.solve.contracts.types import Contract


def resolve(name: str) -> Contract | None:
    """The contract for a census/absint-style already-resolved call name.

    ``name`` is exactly what ``census.py``/``absint`` name a call after their own
    resolution: a bare builtin (``"len"``), a ``"module.func"`` qualname
    (``"math.sqrt"``), or a bare method name matched on ANY receiver (``"pop"``,
    ``"heappop"``) -- ``census.py``'s own rule, "a method name counts whatever
    object it is called on". An exact/module qualname is looked up directly; a
    bare method name resolves only when EXACTLY ONE contract ends in
    ``.<name>`` -- two different types' methods sharing a name is ambiguous and
    returns ``None`` rather than guessing which one applies.
    """
    direct = get(name)
    if direct is not None:
        return direct
    if "." in name:
        return None
    candidates = [c for c in all_contracts() if c.qualname.endswith(f".{name}")]
    return candidates[0] if len(candidates) == 1 else None


def raise_condition(contract: Contract, exc: str) -> str | None:
    """The TESTED, precise precondition for ``contract`` raising ``exc``.

    ``None`` when the claim is untested, ``"unknown"``, or ``"never"``. Use
    this only where the exact condition text will be surfaced to a human (an
    obligation, a diagnostic) or matched against a hand-written, per-qualname
    table an engine maintains for the few contracts it acts on precisely --
    never parsed as if it were a machine-readable predicate: ``condition`` is
    documented English (``types.py``), not a DSL.
    """
    try:
        claim = contract.raise_claim(exc)
    except KeyError:
        return None
    if not claim.tested or claim.condition in ("never", "unknown"):
        return None
    return claim.condition


def native_raise_condition(contract: Contract, exc: str) -> str | None:
    """Like :func:`raise_condition`, but only a claim about the callable's OWN logic: ``None`` for a class
    the contract admits only by PROPAGATION from a receiver or argument dunder ("propagates uncaught if the
    receiver's __lt__ raises IndexError").  Evidence that an empty container is worth trying needs the call
    itself to raise on it, which a propagation claim says nothing about."""
    condition = raise_condition(contract, exc)
    if condition is None or condition.startswith("propagates uncaught"):
        return None
    return condition


def never_raises(contract: Contract, exc: str) -> bool:
    """True iff ``contract`` is TESTED to never raise ``exc``, for any inputs."""
    try:
        claim = contract.raise_claim(exc)
    except KeyError:
        return False
    return claim.tested and claim.condition == "never"


def may_raise(contract: Contract, exc: str) -> bool:
    """True unless TESTED never to raise ``exc`` -- the sound default: possible."""
    return not never_raises(contract, exc)


def receiver_may_raise(type_name: str, exc: str, *, visitor=None) -> bool:
    """True iff some TESTED, non-``"never"`` claim for ``exc`` about the callable's OWN logic (not a dunder
    propagation, :func:`native_raise_condition`) exists among ``type_name``'s contracted methods
    (``"TYPE.method"`` qualnames)."""
    from tharkun.solve.contracts.registry import for_receiver

    if visitor is None:
        return any(native_raise_condition(c, exc) is not None for c in for_receiver(type_name))
    from tharkun.solve.contracts.rules import Refused, get_rule
    for contract in for_receiver(type_name):
        rule = get_rule(contract.qualname)
        if rule is not None:
            out = rule.accept(visitor, exc)
            if not isinstance(out, Refused) and out:
                return True
    return False


# --- hand-verified per-qualname shapes shared by engines 2 (absint) and 3 (concolic) ----
#
# ``RaiseClaim.condition``/``ReturnSpec.shape`` are documented English for humans, not a
# parseable DSL (``docs/contracts/FORMAT.md``): "does this condition mean len(x)==0" is
# answered by a small, explicit table verified once against the contract's own text --
# never by parsing it at runtime. Kept ONE place so concolic's z3 facts, absint's proof
# checks, and draws' Hypothesis strategies/boundary seeds read the exact same,
# single-sourced facts instead of separate hand-tables that could quietly drift apart.

#: qualname -> the EXACT positional argument count a precise fact below trusts, with
#: NO keywords -- e.g. ``heapq.heappop(heap)`` is exactly 1 positional argument, never
#: keyword-passable in real usage. A call shape not matching this exactly (extra/missing
#: positional arguments, or any keyword argument at all) makes every precise fact below
#: inapplicable for that call, whatever the qualname's contract says elsewhere.
CALL_SHAPE: dict[str, int] = {
    "heapq.heappop": 1,
    "heapq.heappush": 2,
    "heapq.heapify": 1,
    "bisect.insort": 2,
}


def applicable_call_shape(qualname: str, arg_count: int, has_keywords: bool) -> bool:
    """True iff this ACTUAL call's shape matches :data:`CALL_SHAPE`'s hand-verified
    expectation for ``qualname`` -- the applicability gate every precise fact below
    (and the absint adapter's own ``TypeError: never`` downgrade) shares."""
    expected = CALL_SHAPE.get(qualname)
    return expected is not None and not has_keywords and arg_count == expected


#: the output-file family ([internal ref], ``contracts/rules_fileio.py``): method name -> the positional argument
#: count (no keywords) the model of a write-only file handle trusts.  ``__exit__`` is the three
#: arguments a ``with`` statement passes.  Same discipline as :data:`CALL_SHAPE`: a call whose shape
#: is not exactly this is not covered by the contract's precise facts.
OUTPUT_HANDLE_METHOD_ARITY: dict[str, int] = {
    "write": 1, "writelines": 1, "flush": 0, "close": 0, "__enter__": 0, "__exit__": 3,
}

#: the handle type a write-only open returns, by binary-ness
OUTPUT_HANDLE_TYPES: dict[bool, str] = {False: "io.TextIOWrapper", True: "io.BufferedWriter"}

#: builtin name -> the qualname of the SHAPE contract its engine model reads instead of the union
#: contract of the same name (``open`` the union is untested: the type it returns depends on the mode)
BUILTIN_SHAPE_CONTRACTS: dict[str, str] = {"open": "open.output"}


def output_handle_contract(binary: bool, method: str) -> Contract | None:
    """The TESTED contract of ``method`` on a write-only text (``binary=False``) or binary handle,
    or ``None`` (not a method of the family, or the contract is missing or untested): the caller
    then treats the call as unmodelled."""
    if method not in OUTPUT_HANDLE_METHOD_ARITY:
        return None
    contract = get(f"{OUTPUT_HANDLE_TYPES[binary]}.{method}")
    return contract if contract is not None and contract.tested else None


def output_handle_call_shape(method: str, arg_count: int, has_keywords: bool) -> bool:
    """True iff this call of ``method`` has exactly the hand-verified shape of
    :data:`OUTPUT_HANDLE_METHOD_ARITY`."""
    expected = OUTPUT_HANDLE_METHOD_ARITY.get(method)
    return expected is not None and not has_keywords and arg_count == expected


#: qualname -> the positional arities (no keywords) at which the call is PURE FIXED NATIVE CODE:
#: it runs no user code and raises nothing, whatever its arguments, although the contract for
#: the qualname as a whole admits callbacks (another shape of the same callable does). Only
#: ``type(x)``: ``type_call`` special-cases exactly one positional argument, so a metaclass
#: ``__call__``, a ``__class__`` property or ``__getattribute__`` cannot intercept it, while
#: the 3-argument form creates a class and runs metaclass/``__init_subclass__``/``__set_name__``
#: code (``tests/contracts/test_builtins_more.py::TestType``). Same discipline as
#: :data:`CALL_SHAPE`: a claim holds for the hand-verified shape only.
NO_CALLBACK_SHAPES: dict[str, frozenset[int]] = {
    "type": frozenset({1}),
}


def runs_no_callback(qualname: str, *, arg_count: int, has_keywords: bool) -> bool:
    """True iff this EXACT call shape is hand-verified to run no user code and raise
    nothing (:data:`NO_CALLBACK_SHAPES`) AND the registry still holds a TESTED contract
    for ``qualname`` (a removed or untested contract withdraws the fact)."""
    arities = NO_CALLBACK_SHAPES.get(qualname)
    if arities is None or has_keywords or arg_count not in arities:
        return False
    contract = resolve(qualname)
    return contract is not None and contract.tested


#: qualname -> the positional arities (no keywords) at which the call's RESULT is a function of
#: the CLASS of its first argument only (never its value), and the call raises nothing and has
#: no effect, PROVIDED the first argument is an exact builtin scalar and any further argument
#: is a builtin class: ``type(x)`` and ``isinstance(x, <builtin class(es)>)``. The contracts
#: for both admit callbacks (a hostile ``__class__`` property, a metaclass
#: ``__instancecheck__``); the caller discharges that by passing a value with no user hook.
#: Used by ``solve/drawflow.py``: a trip value (always an exact int) may flow into such a call.
CLASS_ONLY_SHAPES: dict[str, frozenset[int]] = {
    "type": frozenset({1}),
    "isinstance": frozenset({2}),
}


def result_depends_only_on_class(qualname: str, *, arg_count: int, has_keywords: bool) -> bool:
    from tharkun.solve.contracts.rules import Refused, get_rule
    from tharkun.solve.contracts.visitors import StationaryVisitor
    rule = get_rule(qualname)
    if rule is None:
        return False
    out = rule.accept(StationaryVisitor(), (arg_count, has_keywords))
    return False if isinstance(out, Refused) else out


def _result_depends_only_on_class_model(qualname: str, *, arg_count: int, has_keywords: bool) -> bool:
    """True iff this EXACT call shape is hand-verified as class-only (:data:`CLASS_ONLY_SHAPES`)
    AND the registry still holds a TESTED contract for ``qualname``."""
    arities = CLASS_ONLY_SHAPES.get(qualname)
    if arities is None or has_keywords or arg_count not in arities:
        return False
    contract = resolve(qualname)
    return contract is not None and contract.tested


#: callable -> what it does with the ORIGINAL object of an iterable argument before iterating it. CPython's
#: ``PyObject_LengthHint`` (``Objects/abstract.c``) calls the object's ``__len__`` -- and swallows a
#: ``TypeError`` from it, propagating every other class -- but only the callables that presize a list from the
#: argument do: ``list(x)``, ``list.extend``/``+=``, ``sorted``, ``bytes(x)``, ``bytearray.extend`` and
#: ``operator.length_hint`` itself.  ``tuple``, ``set``, ``frozenset``, ``array.array(tc, x)`` and
#: ``array.extend`` never call it, nor does a ``for`` loop, an unpacking, ``min``/``max``/``sum``, ``str.join``
#: (it asks the iterator, not ``x``) or ``xs[:] = x``.  ``tests/contracts/test_swallowing_native.py`` sweeps every
#: entry, and a candidate list well beyond it, on CPython 3.14.4 with an instrumented object.  "probe": the
#: engine must model the call and its TypeError swallowing, or refuse; "skip": it must not run ``__len__`` at all.
LENGTH_HINT_RULES: dict[str, str] = {
    "list": "probe", "list.extend": "probe", "list.__iadd__": "probe", "sorted": "probe", "bytes": "probe",
    "bytearray.extend": "probe", "operator.length_hint": "probe",
    "tuple": "skip", "set": "skip", "frozenset": "skip", "array.array": "skip", "array.array.extend": "skip",
}


def length_hint_rule(consumer: str) -> str | None:
    """``"probe"`` or ``"skip"`` (:data:`LENGTH_HINT_RULES`) for ``consumer``, or ``None`` when the table does
    not know it or the registry no longer holds a TESTED ``protocol.length_hint`` contract (or a tested contract
    for the consumer itself): an engine must then refuse."""
    rule = LENGTH_HINT_RULES.get(consumer)
    if rule is None:
        return None
    protocol = get("protocol.length_hint")
    if protocol is None or not protocol.tested:
        return None
    contract = resolve(consumer)
    if contract is not None and not contract.tested:
        return None
    return rule


#: type name -> the builtin constructor that, called with no argument, returns a NEW EMPTY
#: container and raises nothing (each contract's ``returns`` says ``T() is the empty T`` and
#: its TypeError claim names only non-iterable/over-arity/keyword calls;
#: ``tests/contracts/test_builtins_more.py::TestIterableConstructors::test_no_argument_is_the_empty_container``).
ZERO_ARGUMENT_CONSTRUCTORS: dict[str, Callable] = {"list": list, "tuple": tuple, "set": set}
#: the same, called with ONE list argument of a single element: a NEW one-element container
#: (a non-iterable argument is the TypeError the contract names; a list is always iterable).
SINGLETON_CONSTRUCTORS = ZERO_ARGUMENT_CONSTRUCTORS


def constructor_contract(type_name: str, *, visitor=None) -> bool:
    if visitor is None:
        return _constructor_contract_model(type_name)
    from tharkun.solve.contracts.rules import Refused, get_rule
    rule = get_rule(type_name)
    if rule is None:
        return False
    out = rule.accept(visitor)
    return False if isinstance(out, Refused) else out


def _constructor_contract_model(type_name: str) -> bool:
    """True iff ``type_name`` is a constructor in :data:`ZERO_ARGUMENT_CONSTRUCTORS` whose
    contract is still TESTED, ``returns`` included -- the evidence draws and Avengers need
    before they build a value by calling it."""
    if type_name not in ZERO_ARGUMENT_CONSTRUCTORS:
        return False
    contract = resolve(type_name)
    return contract is not None and contract.tested and contract.returns is not None and contract.returns.tested


#: qualname -> the property class TESTED to fire iff the call's FIRST argument is empty
#: (``len(arg) == 0``). Only ``heapq.heappop`` -- ``heapq.heapreplace`` has NO contract
#: registered anywhere in ``solve/contracts`` (confirmed against the live registry,
#: 2026-09-28): an entry here for it would silently never fire, which is sound (every
#: reader already treats "no contract" as "nothing known") but misleading to keep.
EMPTY_FIRST_ARG_RAISES: dict[str, str] = {
    "heapq.heappop": "IndexError",
}

#: qualname -> (lo, hi, hi_inclusive) for a no-argument call with a TESTED, closed
#: numeric ``ReturnSpec.shape`` (``"in [0, 1)"`` for ``random.random``).
CLOSED_RANGE_RETURN: dict[str, tuple[float, float, bool]] = {
    "random.random": (0.0, 1.0, False),
}


def empty_first_arg_precondition(qualname: str, *, arg_count: int, has_keywords: bool) -> str | None:
    """The property exception class TESTED to fire iff ``qualname``'s first
    argument is empty, or ``None`` when the hand-verified shape doesn't apply,
    this call's shape doesn't match :data:`CALL_SHAPE`, the contract is
    missing, or its claim for that class isn't tested."""
    exc = EMPTY_FIRST_ARG_RAISES.get(qualname)
    if exc is None or not applicable_call_shape(qualname, arg_count, has_keywords):
        return None
    contract = resolve(qualname)
    if contract is None or raise_condition(contract, exc) is None:
        return None
    return exc


def closed_range_return(qualname: str) -> tuple[float, float, bool] | None:
    """``(lo, hi, hi_inclusive)`` for a TESTED no-argument call with a closed
    numeric return range, else ``None``."""
    spec = CLOSED_RANGE_RETURN.get(qualname)
    if spec is None:
        return None
    contract = resolve(qualname)
    if contract is None or contract.returns is None or not contract.returns.tested:
        return None
    return spec


#: qualname -> (receiver_arg_index, length_delta, new_element_arg_index | None): a
#: hand-verified, EXACT effect on ``len(<receiver>)`` after a successful call, and,
#: when the call also INSERTS a new element (not just removes one), which positional
#: argument holds it -- so the engine can join that argument's own value into the
#: receiver's element facts instead of leaving them stale (2026-09-28 review: "a stale
#: abstract list is unsound" -- ``heapq.heappush``/``bisect.insort`` change list length
#: AND contents, not just length). ``heapq.heapify`` reorders in place: same length,
#: same elements, so delta=0 and no new element -- listed explicitly (rather than left
#: to the general havoc fallback) because its effect really is a precise no-op on both
#: facts, verified against its own contract text and CPython source citation, the same
#: discipline as :data:`EMPTY_FIRST_ARG_RAISES`. Applies only at the exact
#: :data:`CALL_SHAPE` arity for this qualname; gated on ``Effects.mutates_receiver``
#: actually being True in the registry (never assumed).
RECEIVER_LENGTH_EFFECT: dict[str, tuple[int, int, int | None]] = {
    "heapq.heappop": (0, -1, None),
    "heapq.heappush": (0, 1, 1),
    "heapq.heapify": (0, 0, None),
    "bisect.insort": (0, 1, 1),
}


def receiver_length_delta(
    qualname: str, *, arg_count: int, has_keywords: bool
) -> tuple[int, int, int | None] | None:
    """``(receiver_arg_index, length_delta, new_element_arg_index)`` when TESTED
    contract evidence backs a hand-verified receiver-mutation effect for this
    EXACT call shape, else ``None`` -- see :data:`RECEIVER_LENGTH_EFFECT`."""
    spec = RECEIVER_LENGTH_EFFECT.get(qualname)
    if spec is None or not applicable_call_shape(qualname, arg_count, has_keywords):
        return None
    contract = resolve(qualname)
    if contract is None or not contract.effects.mutates_receiver:
        return None
    return spec


__all__ = [
    "resolve", "raise_condition", "native_raise_condition", "never_raises", "may_raise", "receiver_may_raise",
    "CALL_SHAPE", "applicable_call_shape", "NO_CALLBACK_SHAPES", "runs_no_callback",
    "CLASS_ONLY_SHAPES", "result_depends_only_on_class",
    "ZERO_ARGUMENT_CONSTRUCTORS", "SINGLETON_CONSTRUCTORS", "constructor_contract",
    "EMPTY_FIRST_ARG_RAISES", "CLOSED_RANGE_RETURN",
    "empty_first_arg_precondition", "closed_range_return",
    "RECEIVER_LENGTH_EFFECT", "receiver_length_delta",
    "LENGTH_HINT_RULES", "length_hint_rule",
]
