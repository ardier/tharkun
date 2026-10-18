"""``tharkun.solve.contracts`` -- tested stdlib/builtin call contracts ([internal ref]).

A :class:`Contract` records, for one callable, exactly which SV-COMP property
exception classes it can raise and under what condition, what it returns, and its
side effects -- each claim backed by a docs/typeshed/CPython citation and, once
``tested=True``, a passing test on CPython 3.14.4. See ``docs/contracts/FORMAT.md``
for the schema and ``docs/contracts/REPORT.md`` for batch-1 coverage.

Public surface:

    from tharkun.solve.contracts import get, all_contracts, for_receiver, for_module
    from tharkun.solve.contracts import Contract, RaiseClaim, ReturnSpec, Effects

Nothing here changes Tharkun's verdict logic -- this is data plus a loader. Wiring a
proof (``solve/census.py``, ``solve/absint``) to consult it is a separate, reviewed
step, per the ticket.
"""

from __future__ import annotations

from tharkun.solve.contracts.reachability import closed_call_hooks
from tharkun.solve.contracts.registry import all_contracts, for_module, for_receiver, get, load_all
from tharkun.solve.contracts.types import ALL_PROPERTY_CLASSES, Contract, Effects, RaiseClaim, ReturnSpec

__all__ = [
    "ALL_PROPERTY_CLASSES",
    "Contract",
    "RaiseClaim",
    "ReturnSpec",
    "Effects",
    "get",
    "all_contracts",
    "for_receiver",
    "for_module",
    "load_all",
    "closed_call_hooks",
]
