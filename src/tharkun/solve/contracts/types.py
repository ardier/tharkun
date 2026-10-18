"""Contract data model ([internal ref]) -- the shape every stdlib/builtin contract fills in.

SOLVING, by ``docs/LAYERS.md``'s rule: a :class:`Contract` is a claim about a callable
computed once, offline, from the docs/typeshed/CPython source and TESTED on CPython
3.14.4 -- never a value the target program can influence.  Nothing here imports or
runs a target; ``tests/contracts/`` runs the callables the contracts describe, which is
different and lives outside ``solve/``.

Why a contract, not a wider absint/census model directly: ``solve/census.py`` and
``solve/absint/interp.py`` each hand-roll their own small tables (``_CALL_SITES``,
``MODULES``) for exactly the calls their one proof needs. A contract is the superset
those tables want to grow into -- one row per callable, every property exception class
accounted for (not just the ones a given proof cares about), with the evidence and the
test that backs each claim recorded next to it. Wiring a proof to read from here instead
of its own table is a separate, reviewed step (batch 1 only writes and tests contracts).

The seven classes below are exactly the SV-COMP property classes tharkun decides
(``tharkun.solve.absint.PROPERTY_CLASSES``, ``docs/SV.md``). A :class:`Contract`'s
``raises`` is TOTAL over this set: every property class gets an explicit
:class:`RaiseClaim`, so a class the receiver/arguments cannot raise says ``"never"``
rather than being left out. Leaving a class out would read as "not modelled", which
silently degrades a caller from sound-unknown to unsound-true; saying ``"never"`` is
the actual claim, and it is the one that must be tested.

A raise condition that IS known says so in ``condition`` (a short, precise English
description of the condition on the receiver/arguments, e.g. ``"index out of range:
not -len(seq) <= index < len(seq)"``). A raise condition the author could not pin down
exactly still gets recorded, with ``condition="unknown"`` -- sound (it never claims the
class can't happen) but not useful for pruning. ``Contract.may_raise_other`` is the
open-world flag: True unless every non-property exception the callable can raise has
been enumerated and ruled irrelevant (batch 1 leaves this True almost everywhere; a
contract that sets it False must say why in ``source``).

``tested`` graduates from False to True only via ``tests/contracts/`` actually running
the callable on CPython 3.14.4 and observing the claim hold (docstring gate, not
vibes) -- see ``docs/contracts/FORMAT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tharkun.solve.absint import PROPERTY_CLASSES

#: the seven SV-COMP property exception classes every contract's ``raises`` must cover,
#: exactly (``tharkun.solve.absint.PROPERTY_CLASSES`` is the canonical source; this is
#: just the flat union, sorted for stable iteration/printing).
ALL_PROPERTY_CLASSES: tuple[str, ...] = tuple(
    sorted({exc for group in PROPERTY_CLASSES.values() for exc in group})
)

_ALLOWED_CONDITION_PREFIXES = ("unknown",)  # "unknown" is the one free-form sentinel


@dataclass(frozen=True)
class RaiseClaim:
    """One property exception class's status for a callable.

    ``condition``:
      * ``"never"``   -- the callable cannot raise ``exc`` (any receiver/arguments).
      * ``"unknown"`` -- it might; the exact condition is not pinned down (still sound).
      * anything else -- the exact condition on receiver/arguments under which ``exc``
        is raised (English, precise enough that a test can check both sides of it).
    ``tested`` is True only when a test in ``tests/contracts/`` exercises this specific
    claim on CPython 3.14.4: for ``"never"``, a battery of edge cases (Hypothesis plus
    explicit empty/zero/negative/None/wrong-type/huge/NaN/inf cases) observed NOT to
    raise ``exc``; for a specific condition, both the raising case and the boundary
    just outside it. ``"unknown"`` can never be ``tested`` -- there is no pinned-down
    condition to check, so a claim left ``"unknown"`` stays untested by construction.
    """

    exc: str
    condition: str
    tested: bool = False

    def __post_init__(self) -> None:
        if self.exc not in ALL_PROPERTY_CLASSES:
            raise ValueError(f"{self.exc!r} is not a property class: {ALL_PROPERTY_CLASSES}")
        if not self.condition:
            raise ValueError(f"{self.exc}: condition must not be empty")
        if self.condition == "unknown" and self.tested:
            raise ValueError(f"{self.exc}: condition 'unknown' cannot be tested")


@dataclass(frozen=True)
class ReturnSpec:
    """The return value's type and, where known, its range/shape.

    ``type`` is a typeshed-style annotation string (``"int"``, ``"list[int]"``,
    ``"float | None"``). ``shape`` is a short English description of a range or
    invariant beyond the bare type (``"in [0, 1)"``, ``"same length as seq"``,
    ``""`` when the type alone is everything known).
    """

    type: str
    shape: str = ""
    tested: bool = False


@dataclass(frozen=True)
class Effects:
    """Side effects beyond raising and returning.

    ``mutates_receiver``: the bound object (``self``/first positional arg for a
    function) is mutated in place.
    ``mutates_args``: names/positions of OTHER arguments mutated in place (e.g.
    ``sorted(x, key=...)`` calls ``key`` but does not mutate it -- most entries here
    are empty; ``list.extend`` mutates only the receiver, not the argument, so this is
    for callables like ``heapq.heapify`` families where an argument other than the
    receiver is the one that changes).
    ``io``: touches the filesystem, network, or standard streams.
    ``randomness``: ``None`` if none, else which generator/state is consumed (e.g.
    ``"random module global Mersenne Twister state"``, ``"os.urandom (CSPRNG, no
    process state)"``).
    ``clock_or_env``: reads the wall clock, a monotonic clock, or process/OS
    environment (env vars, PID, hostname) -- so two calls in the same process can
    observably differ even with everything else held fixed.
    ``deterministic``: False whenever any of ``io``, ``randomness`` or
    ``clock_or_env`` make the return value or raised exception depend on something
    outside the arguments; kept as an explicit field (not derived) because a callable
    can do I/O that is deterministic in effect but still worth flagging, and the
    loader should not have to re-derive the flag from the others.
    ``may_execute_callback``: defaults True until callback freedom is established.
    True when this callable's mechanism invokes arbitrary,
    unmodeled code supplied by the receiver or an argument (a dunder like
    ``__add__``/``__hash__``/``__iter__``, a ``key=``/``default=`` callback, ...).
    When True, every OTHER field on this ``Effects`` describes only the callable's
    own NATIVE dispatch mechanism -- e.g. ``deterministic=True`` means "the dispatch
    logic itself adds no nondeterminism", NOT "the whole operation is guaranteed
    deterministic". The invoked callback can do I/O, hold state, read the clock or
    environment, or mutate anything it can reach, regardless of what the other
    fields say (confirmed empirically: an ``__add__`` that reads ``os.environ``
    makes the result depend on process environment despite the contract's own
    dispatch having none of that -- see ``tests/contracts/test_protocols.py``).
    A consumer MUST NOT treat ``mutates_receiver=False``/``io=False``/
    ``deterministic=True`` etc. as a purity or determinism certificate for the
    WHOLE operation when this is True -- only when it is False do the other
    fields describe the whole operation.
    """

    mutates_receiver: bool = False
    mutates_args: tuple[str, ...] = ()
    io: bool = False
    randomness: str | None = None
    clock_or_env: bool = False
    deterministic: bool = True
    may_execute_callback: bool = True
    notes: str = ""


@dataclass(frozen=True)
class Contract:
    """The full claim for one callable.

    ``qualname`` is how the loader is queried: a bare name for a builtin (``"len"``),
    ``"TYPE.method"`` for a method looked up on any receiver of that type (``"list.pop"``,
    ``"dict.__getitem__"``), or ``"module.name"`` for a module-level function/constant
    (``"math.sqrt"``, ``"random.random"``).
    ``raises`` must have exactly one :class:`RaiseClaim` per class in
    :data:`ALL_PROPERTY_CLASSES` (validated in ``__post_init__``).
    ``tested`` is the contract-wide gate: True only when every ``RaiseClaim`` whose
    condition is not ``"unknown"``, and the ``ReturnSpec`` if present, are individually
    tested (also validated, not just asserted by the author).
    """

    qualname: str
    raises: tuple[RaiseClaim, ...]
    may_raise_other: bool
    returns: ReturnSpec | None
    effects: Effects
    source: tuple[str, ...]
    tested: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        seen = {c.exc for c in self.raises}
        missing = set(ALL_PROPERTY_CLASSES) - seen
        if missing:
            raise ValueError(f"{self.qualname}: missing raise claims for {sorted(missing)}")
        extra = seen - set(ALL_PROPERTY_CLASSES)
        if extra:
            raise ValueError(f"{self.qualname}: unknown exception classes {sorted(extra)}")
        if len(seen) != len(self.raises):
            raise ValueError(f"{self.qualname}: duplicate raise claims")
        if not self.source:
            raise ValueError(f"{self.qualname}: source must cite at least one piece of evidence")
        if self.tested:
            untested_raises = [c.exc for c in self.raises if c.condition != "unknown" and not c.tested]
            if untested_raises:
                raise ValueError(
                    f"{self.qualname}: tested=True but raise claim(s) untested: {untested_raises}"
                )
            if self.returns is not None and not self.returns.tested:
                raise ValueError(f"{self.qualname}: tested=True but return claim is untested")

    def raise_claim(self, exc: str) -> RaiseClaim:
        """The claim for ``exc`` (a property class); raises ``KeyError`` otherwise."""
        for c in self.raises:
            if c.exc == exc:
                return c
        raise KeyError(exc)


__all__ = [
    "ALL_PROPERTY_CLASSES",
    "RaiseClaim",
    "ReturnSpec",
    "Effects",
    "Contract",
]
