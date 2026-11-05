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


def stdlib_admission(qualname: str) -> str | None:
    """Why a standard-library callable may NOT be named by a recipe, or ``None`` when it may ([internal ref]).

    A callable is admitted when it has a TESTED
    contract that says no file, network, process, clock, environment or randomness effect (``Effects``), and
    refused -- with the reason as data -- when the registry has no contract for it, the contract is untested,
    it declares such an effect, or it is an unbounded attribute lookup. This is the native-dispatch policy, not a whole-operation
    purity certificate: declared receiver mutation and arbitrary user callbacks still apply
    (``types.Effects``). Consumers must not erase those effects after admitting a name.
    """
    from tharkun.solve.contracts.registry import get

    # builtins' contracts are registered by their bare name (`type`, `len`); a recipe names them `builtins.type` ([internal ref])
    contract = get(qualname) or (get(qualname[len("builtins."):]) if qualname.startswith("builtins.") else None)
    if contract is None:
        return f"stdlib-no-contract:{qualname}"
    if not contract.tested:
        return f"stdlib-untested-contract:{qualname}"
    if qualname in ('dataclasses._is_dataclass_instance', 'json.JSONEncoder.default',
                    'inspect.getattr_static', 'inspect.getmembers_static'):
        return f'stdlib-unbounded-lookup:{qualname}'

    if qualname in ('getattr', 'builtins.getattr'):
        return f'stdlib-unbounded-lookup:{qualname}'
    if qualname == 'functools.wraps':
        return f'stdlib-deferred-callback:{qualname}'
    effects = contract.effects
    for label, present in (("io", effects.io), ("randomness", effects.randomness), ("clock-or-env", effects.clock_or_env),
                           ("nondeterministic", not effects.deterministic)):
        if present:
            return f"stdlib-effect-{label}:{qualname}"
    return None


def factories() -> tuple:
    """Every admitted value factory (:class:`~tharkun.solve.contracts.rules_factories.FactorySpec`), qualname-sorted."""
    from tharkun.solve.contracts.rules import Refused, all_rules

    visitor = AvengersVisitor()
    found = [out for rule in all_rules() if rule.kind == "stdlib_factory"
             if not isinstance(out := rule.accept(visitor), Refused)]
    return tuple(found)


__all__ = ["nonempty_seed_for_type", "stdlib_admission", "factories"]
