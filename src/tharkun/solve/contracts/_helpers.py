"""Shared construction helpers for contract modules ([internal ref]). Not part of the public API.

Most callables in this batch invoke exactly one receiver/argument dunder (or an
attribute like ``.write``) and otherwise raise only from their own, fixed C-level
logic. :func:`dunder_propagation_claims` builds the 7 :class:`RaiseClaim`\\ s for that
common shape once, instead of every contract module repeating the same boilerplate:
* the classes named in ``native`` get the given condition (the callable's own logic);
* every other property class gets "propagates uncaught if the receiver's DUNDER raises
  it" -- CPython dispatches to the dunder via the type slot, not a ``getattr``/``hasattr``
  probe, so a dunder that raises *any* exception (including a class this callable's own
  logic never raises directly) comes out of the call unchanged (confirmed empirically for
  ``len``/``int``/``float``/``abs``/``divmod``/``pow``/``sorted``/``min`` -- see
  ``tests/contracts/test_builtins.py``).
"""

from __future__ import annotations

from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, RaiseClaim


def dunder_propagation_claims(
    dunder: str,
    native: dict[str, str] | None = None,
    tested_native: frozenset[str] = frozenset(),
    tested_propagation: bool = True,
) -> tuple[RaiseClaim, ...]:
    """The 7 :class:`RaiseClaim`\\ s for a callable whose only path to a property
    exception class NOT in ``native`` is ``dunder`` (named for the message, e.g.
    ``"__len__"`` or ``"__index__``/``__int__``/``__trunc__``") propagating uncaught.
    """
    native = native or {}
    claims = []
    for exc in ALL_PROPERTY_CLASSES:
        if exc in native:
            claims.append(RaiseClaim(exc, native[exc], tested=exc in tested_native))
        else:
            claims.append(
                RaiseClaim(
                    exc,
                    f"propagates uncaught if the receiver's {dunder} raises {exc}",
                    tested=tested_propagation,
                )
            )
    return tuple(claims)


def never_claims(native: dict[str, str] | None = None, tested: frozenset[str] = frozenset()) -> tuple[RaiseClaim, ...]:
    """Property-exception claims with no escaping paths beyond ``native``.

    This does not certify absence of callbacks: cleanup code can execute while its
    exceptions remain unraisable. Audit callback effects separately.
    """
    native = native or {}
    claims = []
    for exc in ALL_PROPERTY_CLASSES:
        if exc in native:
            claims.append(RaiseClaim(exc, native[exc], tested=exc in tested))
        else:
            claims.append(RaiseClaim(exc, "never", tested=exc in tested))
    return tuple(claims)


__all__ = ["dunder_propagation_claims", "never_claims"]
