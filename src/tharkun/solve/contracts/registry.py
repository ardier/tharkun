"""The contract loader ([internal ref]) -- the query API Tharkun code will read contracts through.

Registration is explicit and eager: each rule class owns its
:class:`~tharkun.solve.contracts.types.Contract`, and :func:`load_all`
imports every rule module by a plain, static ``import`` (never ``importlib`` --
``solve/``'s own layering rule, ``tests/test_layering.py``, bans execution/import
primitives everywhere under ``solve/`` except ``solve/loader.py``, the one sanctioned
place that may load a *target*; this loader only imports tharkun's own shipped
modules, but the rule is enforced textually, not by intent, so it stays static too).
The module-level ``REGISTRY`` dict is the single source the query functions read --
mirrors ``solve/census.py``'s and ``solve/absint/interp.py``'s own tables: a plain
dict any layer above ``solve/`` can read, no plugin discovery, no filesystem scan.

This module only reads/holds contracts; nothing here changes Tharkun's verdict logic
(the ticket's own rule -- wiring in is a separate, reviewed step).
"""

from __future__ import annotations

from tharkun.solve.contracts.types import Contract

#: qualname -> Contract, populated by load_all()
REGISTRY: dict[str, Contract] = {}

_loaded = False


def load_all() -> None:
    """Register every shipped rule once; each rule owns the Contract served by this API."""
    global _loaded
    if _loaded:
        return
    # Importing a rule module registers one Rule and its Contract per qualname.
    from tharkun.solve.contracts import (  # noqa: F401
        rules_bisect, rules_builtins, rules_builtins_more, rules_bytes, rules_collections, rules_copy, rules_dict,
        rules_fileio, rules_functools, rules_heapq, rules_io, rules_itertools, rules_list, rules_math, rules_operator,
        rules_protocols, rules_random, rules_re, rules_set, rules_str, rules_strformat, rules_tuple, rules_verifier,
    )
    _loaded = True


def get(qualname: str) -> Contract | None:
    """The contract for ``qualname``, or ``None`` if batch 1 has not written one."""
    load_all()
    return REGISTRY.get(qualname)


def all_contracts() -> tuple[Contract, ...]:
    """Every loaded contract, ``qualname``-sorted for stable output."""
    load_all()
    return tuple(REGISTRY[k] for k in sorted(REGISTRY))


def for_receiver(type_name: str) -> tuple[Contract, ...]:
    """Contracts of the form ``"TYPE.method"`` for ``type_name`` (e.g. ``"list"``)."""
    load_all()
    prefix = f"{type_name}."
    return tuple(REGISTRY[k] for k in sorted(REGISTRY) if k.startswith(prefix))


def for_module(module_name: str) -> tuple[Contract, ...]:
    """Contracts of the form ``"module.name"`` for ``module_name`` (e.g. ``"math"``)."""
    return for_receiver(module_name)


__all__ = ["REGISTRY", "load_all", "get", "all_contracts", "for_receiver", "for_module"]
