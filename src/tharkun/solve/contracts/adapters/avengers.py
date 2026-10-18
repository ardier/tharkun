"""Engine 5 adapter: ``avengers/`` object construction ([internal ref] step 2).

``avengers``'s own ``BaselineCandidatePass`` always builds the SMALLEST/
emptiest constructor argument for a container-typed parameter
(``_baseline_value``: ``"list"`` -> ``[]``, ``"dict"`` -> ``{}``, ...) --
which is exactly the WRONG default for a goal whose boundary call needs that
argument non-empty (a constructor that stores a list a later call pops from,
for instance). :func:`nonempty_seed_for_type` offers a SECOND, non-empty seed
for exactly the types where TESTED contract evidence shows their own methods
can raise ``IndexError``/``KeyError`` on an empty/falsy receiver -- reusing
the same evidence gate as Engine 4's ``draws.boundary_value_for_type``
(``common.receiver_may_raise``), applied in the opposite direction.

Nothing here replaces the baseline candidate: it is always an ADDITIONAL one,
and "nothing is believed -- promotion is always a replay" (``docs/AVENGERS.md``)
still decides for real whether it helps.
"""

from __future__ import annotations

from tharkun.solve.contracts.visitors import AvengersVisitor
from tharkun.solve.contracts.adapters.common import (
    SINGLETON_CONSTRUCTORS, constructor_contract, receiver_may_raise,
)

#: type name -> a small, non-empty/non-zero seed value.
_NONEMPTY_SEED: dict[str, object] = {
    "list": [0], "tuple": (0,), "dict": {0: 0}, "set": {0}, "frozenset": frozenset({0}),
    "str": "x", "bytes": b"x", "bytearray": bytearray(b"x"),
}
_EMPTY_EVIDENCE_EXCS = ("IndexError", "KeyError")


def nonempty_seed_for_type(type_name: str) -> object | None:
    """A non-empty seed for ``type_name``, when TESTED contract evidence backs
    it (some receiver method of this type really can raise ``IndexError``/
    ``KeyError`` on an empty/falsy receiver) -- else ``None``.
    """
    if type_name not in _NONEMPTY_SEED:
        return None
    if not any(receiver_may_raise(type_name, exc, visitor=AvengersVisitor()) for exc in _EMPTY_EVIDENCE_EXCS):
        return None
    if type_name in SINGLETON_CONSTRUCTORS:
        # [internal ref] batch 4: a one-element container built by the contracted constructor
        # (a NEW one each time), offered only while that contract is TESTED
        return SINGLETON_CONSTRUCTORS[type_name]([0]) if constructor_contract(type_name, visitor=AvengersVisitor()) else None
    return _NONEMPTY_SEED[type_name]


__all__ = ["nonempty_seed_for_type"]
