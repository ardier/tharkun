"""Engine 1 adapter: ``solve/census.py``'s site tables + call-site obligations ([internal ref]).

Redesigned per 2026-09-28 review (superseding the first cut, which surfaced a
contract's free-text English ``RaiseClaim.condition`` as if it were a checkable
safety condition -- e.g. ``"not (the heap (list) is empty)"``, which is not
Python and must never reach a solver).

``census.py`` already hand-rolls ``_CALL_SITES`` (a dict of exception class name ->
frozenset of callable/method names it knows can raise that class) over
``SITE_KINDS`` (5 of the 7 property classes). An EARLIER revision of this adapter
also EXTENDED that table from the contracts registry (a ``contract_call_sites``
that widened ``census.py``'s own site set by call name, purely additive and sound
by construction) -- removed ([internal ref], 2026-09-28 owner review): every hostile
construction that widening could ever catch is ALREADY caught by census's
existing, unconditional scan of a user dunder's own body (``pyresolve.py``'s
``_dead_defs()`` never marks a dunder dead, and "unknown external calls keep
every class method live" covers a non-dunder protocol method too, e.g.
``print(x, file=hostile)``'s ``.write()``) -- proven by construction (reflection
is refused, so any override MUST be real, visible Python source, and its def is
always live) and confirmed empirically against every widened builtin (`range`,
`abs`, `divmod`, `str`/`repr`/`ascii`/`format`, `print`, `set.update`/`set.add`,
`len`, `isinstance`, `random.shuffle`, including one level of indirection
through a helper function). It therefore added NO soundness, only imprecision
(RW ``proof:no-site`` regressions with no wrong-TRUE it prevented) -- see
``docs/SV.md``'s ``proof:no-site`` section for the argument in full.

The OBLIGATION half below is unaffected by this and is redesigned to match OB's
witness-format shape
(``solve/obligations.py`` on the still-unmerged [internal ref] slice-0 branch, read for its
schema only -- not imported, since that branch is someone else's unmerged work):
a ``location_invariant`` entry whose ``value`` is a checked ``python_expression``
string (``format="python_expression"``), never English. Typed safety templates
live in ``solve/contracts/safety/`` and are gated by the matching TESTED raise
claim. The existing ``heapq.heappop`` expression remains as a compatibility
path there until M5 adds sequence and length terms; this adapter delegates its
single applicability check and keeps the existing witness text. A contract's
English condition is never parsed as a formula. Anything without a matching
template becomes a coverage GAP (:data:`Gap`), carried explicitly -- never
dropped, never guessed.

Worked example: ``heapq.heappop(h)`` -- ``IndexError`` iff the heap is empty --
produces the Python expression ``len(h) > 0`` as the SAFETY condition (the
negation of the contract's raise condition), with ``h`` substituted from the
actual call site, not a fixed name.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

from tharkun.solve.contracts.adapters.common import raise_condition
from tharkun.solve.contracts.registry import get
from tharkun.solve.contracts.safety.heapq import legacy_call

#: the 5 exception classes census.py has a site table for (its own SITE_KINDS)
_CENSUS_CLASSES = (
    "ZeroDivisionError", "FloatingPointError", "AssertionError", "KeyError", "IndexError",
)

@dataclass(frozen=True)
class Invariant:
    """A ``location_invariant`` witness entry -- the shape OB's (unmerged)
    ``solve/obligations.py::Invariant`` uses, reproduced locally (not imported:
    that branch is unmerged and not mine to depend on) so this PR emits the
    SAME format from day one and needs no translation once [internal ref] lands.
    """

    file_name: str
    line: int
    column: int
    function: str
    value: str
    type: str = "location_invariant"

    def entry(self) -> dict:
        return {"invariant": {"type": self.type, "location": {
            "file_name": self.file_name, "line": self.line, "column": self.column,
            "function": self.function}, "value": self.value, "format": "python_expression"}}


@dataclass(frozen=True)
class Gap:
    """A call site this adapter recognises as a potential obligation but cannot
    express as a checked ``python_expression`` -- carried explicitly, exactly
    like census's own ``Census.reason``/``sites``, never silently dropped."""

    file_name: str
    line: int
    column: int
    qualname: str
    exc: str
    reason: str


@dataclass(frozen=True)
class Obligation:
    """One call site's safety obligation, OR the reason it is a gap instead.

    Exactly one of ``invariant``/``gap`` is set.
    """

    qualname: str
    exc: str
    invariant: Invariant | None = None
    gap: Gap | None = None

    def __post_init__(self) -> None:
        if (self.invariant is None) == (self.gap is None):
            raise ValueError("exactly one of invariant/gap must be set")



def _first_contracted_class(qualname: str, classes: tuple[str, ...]) -> tuple[object, str] | None:
    """The contract for ``qualname`` and the first of ``classes`` it has a real,
    TESTED claim for, if any -- used only to decide whether a template-less
    site is still contract-INFORMED (and so owes an explicit gap) or entirely
    outside contract awareness (and so owes nothing)."""
    contract = get(qualname)
    if contract is None or not contract.tested:
        return None
    for exc in classes:
        if raise_condition(contract, exc) is not None:
            return contract, exc
    return None


def obligation_for_call(
    qualname: str, call_node: ast.Call, *, file_name: str, function: str,
    classes: tuple[str, ...] = _CENSUS_CLASSES,
) -> Obligation | None:
    from tharkun.solve.contracts.rules import Refused, get_rule
    from tharkun.solve.contracts.visitors import CensusVisitor

    kwargs = dict(file_name=file_name, function=function, classes=classes)
    rule = get_rule(qualname)
    if rule is not None:
        out = rule.accept(CensusVisitor(), (call_node, kwargs))
        if not isinstance(out, Refused):
            return out
    # Keep the existing explicit Gap for calls without a checked template.
    return _obligation_for_call_model(qualname, call_node, **kwargs)


def _obligation_for_call_model(
    qualname: str, call_node: ast.Call, *, file_name: str, function: str,
    classes: tuple[str, ...] = _CENSUS_CLASSES,
) -> Obligation | None:
    """The :class:`Obligation` for a call to ``qualname`` (an EXACTLY resolved
    identity -- e.g. ``"heapq.heappop"``, never a bare short name matched by
    census's widened site table: applicability requires knowing precisely which
    callable this is, not just that SOME contracted callable shares its name).

    ``None`` only when ``qualname`` is entirely outside contract awareness for
    ``classes`` (no contract, untested, or every claim is ``"never"``/
    ``"unknown"``) -- there is nothing this system has to say about the site,
    so nothing is produced; that is not "dropping" a claim, since none was ever
    formed. Otherwise ALWAYS returns something -- an :class:`Invariant` when a
    hand-verified template applies cleanly (right
    arity, no keywords/``*args``, argument text inside the ``python_expression``
    grammar), a :class:`Gap` recording exactly why not otherwise. Per the
    ticket's rule: "anything inexpressible becomes a coverage gap, never
    dropped."
    """
    line, col = call_node.lineno, call_node.col_offset + 1  # witness columns are 1-based
    legacy = legacy_call(qualname, call_node)
    if legacy is None:
        found = _first_contracted_class(qualname, classes)
        if found is None:
            return None
        _contract, exc = found
        return Obligation(qualname, exc, gap=Gap(
            file_name, line, col, qualname, exc,
            "no hand-verified safety-expression template for this qualname yet"))
    exc = legacy.exc
    if legacy.reason is not None:
        return Obligation(qualname, exc, gap=Gap(
            file_name, line, col, qualname, exc, legacy.reason))
    return Obligation(qualname, exc, invariant=Invariant(file_name, line, col, function, legacy.value))


def witness(source: str, filename: str, specification: str, invariants: list[Invariant]) -> list[dict]:
    """A format-2.0 ``invariant_set`` for still-open obligations -- same shape as
    OB's (unmerged) ``solve/obligations.py::witness``, reproduced locally for
    the same reason as :class:`Invariant`."""
    import uuid

    return [{"entry_type": "invariant_set", "metadata": {
        "format_version": "2.0", "uuid": str(uuid.uuid4()),
        "creation_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer": {"name": "Tharkun", "version": "0.2.0"},
        "task": {"input_files": [filename], "input_file_hashes": {
            filename: hashlib.sha256(source.encode("utf-8")).hexdigest()},
            "specification": specification, "data_model": "CPython-3.14-binary64",
            "language": "Python"}},
        "content": [i.entry() for i in invariants]}]


__all__ = [
    "Invariant", "Gap", "Obligation",
    "obligation_for_call", "witness",
]
