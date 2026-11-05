"""``tharkun.solve.preimage`` -- candidate values of one operand from a demand on a call's result ([internal ref]).

The one backward-propagation capability for the str method family, the formatting rules, ``len``/``ord``/``chr``/``set``,
str ``+``/``*`` and the ``re`` module/``Pattern``/``Match`` rules, built ONCE in ``solve/`` and keyed by callable qualname:
a visitor over the existing contract rules (``solve/contracts/rules.py``; owner rule: one Rule class per contract,
every engine a Visitor, a missing visit a named refusal).  Its consumers are the driver's flipper for guards over
derived locals, ``_symexec``'s concretization and the hybrid fuzzer's seed composition (docs/DESIGN.md §8,
docs/contracts/FORMAT.md "Preimages").

* :func:`preimages` -- ONE call: ``preimages("str.split", Items(Length=Eq(2)), (HOLE, "@"))`` returns a
  :class:`~tharkun.solve.preimage.visitor.Preimage` (an exact ``transfer`` when the vocabulary can state the demand on
  the hole, and an iterator of forward-checked candidates) or a named
  :class:`~tharkun.solve.contracts.rules.Refused`;
* :func:`chain_preimages` -- a straight-line chain of such calls from a guarded value back to an entry value, composed
  with the transfers while they exist and with the candidates after that; every candidate it yields is re-checked by
  running the chain's native semantics forward.

Files and how they relate: ``want.py`` (the demand vocabulary; ``Member.holds`` uses ``regex_members.native_member``),
``regex_members.py`` (members of a regex and the bounded native check; stdlib ``re._parser`` only), ``visitor.py``
(``PreimageVisitor``: imports ``want`` and ``regex_members``, reads ``solve/contracts/rules.py`` rules,
``solve/contracts/adapters/str_shapes.py`` for argument kinds and ``solve/contracts/rules_strformat.py::native_limit``
for the native bounds), this module (the facade).  Everything here is SOLVING (docs/LAYERS.md): a candidate is a
witness the caller verifies by execution; nothing here runs a target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from tharkun.solve.preimage.want import (ANY, All, AnyOf, Anything, CaseMap, Compare, Contains, Eq, IsInstance, IsNone, IsSubclass,
                                         Items, Length, Member, Not, OneOf, Prefix, Suffix, Truth, Want)

__all__ = ["preimages", "chain_preimages", "Step", "HOLE", "Want", "Anything", "ANY", "Eq", "OneOf", "CaseMap", "Truth", "IsNone",
           "Compare", "Length", "Prefix", "Suffix", "Contains", "Member", "Items", "All", "AnyOf", "Not", "IsInstance", "IsSubclass"]


def __getattr__(name: str):
    if name == "HOLE":
        from tharkun.solve.preimage.visitor import HOLE
        return HOLE
    raise AttributeError(name)


def preimages(qualname: str, want: Want, args: tuple, kwargs: dict | None = None, *, prior: Want = ANY,
              pool: tuple = (), budget: int = 64, op: str | None = None):
    """The preimage of one call of ``qualname`` (``args`` holds exactly one ``HOLE``), or a named ``Refused``."""
    from tharkun.solve.contracts.rules import Refused, get_rule
    from tharkun.solve.preimage.visitor import ENGINE, PreimageCtx, PreimageVisitor
    rule = get_rule(qualname)
    if rule is None:
        return Refused(ENGINE, qualname, "no contract rule for this qualname")
    ctx = PreimageCtx(want=want, args=tuple(args), kwargs=tuple(sorted((kwargs or {}).items())), prior=prior,
                      pool=tuple(pool), budget=budget, op=op)
    return rule.accept(PreimageVisitor(), ctx)


@dataclass(frozen=True)
class Step:
    """One call of a chain: ``qualname`` applied to ``args`` (one ``HOLE``: the previous step's result, or the entry
    value for the last step), with ``kwargs`` and, for ``protocol.binary_operator``, ``op``."""

    qualname: str
    args: tuple
    kwargs: tuple = ()
    op: str | None = None


def chain_preimages(steps: list[Step], want: Want, *, prior: Want = ANY, pool: tuple = (), budget: int = 32,
                    per_step: int = 6) -> Iterator[Any] | Any:
    """Entry values ``e`` such that ``want(steps[0](steps[1](...steps[-1](e))))``.

    ``steps[0]`` is the call whose result the want is about (nearest the guard), ``steps[-1]`` the one applied to the
    entry value.  While a step has an exact ``transfer`` the demand moves back without materializing values; after that
    each candidate of a step becomes an ``Eq`` demand on the next one (at most ``per_step`` per step).  Returns a named
    ``Refused`` when a step refuses, else an iterator; every yielded value is re-checked forward through the whole
    chain before it is yielded."""
    from tharkun.solve.contracts.rules import Refused
    if not steps:
        return iter(())
    # one ANY-want preimage per step up front: a refusal is reported before anything is iterated, and its bounded
    # native is what the forward re-check runs
    probes = []
    for st in steps:
        probe = preimages(st.qualname, ANY, st.args, dict(st.kwargs), op=st.op, budget=1)
        if isinstance(probe, Refused):
            return probe
        probes.append(probe)

    def forward(e):
        value = e
        for st, probe in zip(reversed(steps), reversed(probes)):
            value = probe._native(*probe.ctx.with_value(value), **dict(st.kwargs))
        return value

    def rec(i: int, w: Want) -> Iterator[Any]:
        st = steps[i]
        pre = preimages(st.qualname, w, st.args, dict(st.kwargs), pool=pool, budget=per_step, op=st.op,
                        prior=prior if i == len(steps) - 1 else ANY)
        if isinstance(pre, Refused):
            return
        if i == len(steps) - 1:
            yield from pre
            return
        if pre.transfer is not None:
            yield from rec(i + 1, pre.transfer)
        for value in pre:
            yield from rec(i + 1, Eq(value))

    def checked() -> Iterator[Any]:
        seen = set()
        n = 0
        for e in rec(0, want):
            key = (type(e), repr(e))
            if key in seen:
                continue
            seen.add(key)
            try:
                ok = prior.holds(e) and want.holds(forward(e))
            except Exception:  # noqa: BLE001 -- a native failure (or an unchecked bound) means "not a preimage"
                ok = False
            if ok:
                yield e
                n += 1
                if n >= budget:
                    return
    return checked()
