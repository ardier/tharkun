"""The fact carrier of the solving core: per-variable value-set facts, and the
pure refine / join arithmetic over them (#0606 / #0611, plan #0600).

:func:`to_value` / :func:`from_value` convert to and from the typed carrier
:class:`tharkun.solve.values.Value` ([internal ref]), both soundly (never narrower).
``container_len`` / ``container_len_ne`` and the ``container_len_eq`` /
``container_len_ne`` ops close the last gap with eregion's ``apply_constraint``.

``VarFacts`` / ``FactEnv`` moved here from ``tharkun.minleaf.types`` (which
re-exports both names).  ``eregion_commons.fact_repr`` is eregion's own carrier:
the eregion solver bridge converts between the two, and a property test on the
eregion side (``test_fact_env_parity.py``) keeps :func:`refine` / :func:`join`
equal to its ``apply_constraint`` / ``merge_env`` on the shared ops -- the
duplication is deliberate (tharkun never depends on eregion) and tested.

Both carriers are frozen here and mutable there: every operation in this module
returns a NEW ``FactEnv`` and never touches its inputs.  The ``set``/``dict``
fields make instances unhashable -- never use them as dict/set keys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Iterable, Optional

#: An exact set (``eq``) wider than this collapses to "unknown" on ``join``.
MAX_CONST_SET = 8

#: The constraint ops :func:`refine` understands (the ops shared with eregion).
REFINE_OPS = ("==", "!=", "<", "<=", ">", ">=", "in", "not in",
              "container_len_eq", "container_len_ne")


@dataclass(frozen=True)
class VarFacts:
    """Per-variable analysis facts from upstream static/symbolic analysis."""
    eq: set[Any] | None = None   # possible values (whitelist)
    neq: set[Any] = field(default_factory=set)   # excluded values
    lo: int | float | None = None   # lower bound
    hi: int | float | None = None   # upper bound
    #: ``len(var) == K`` / ``len(var) != K`` facts (eregion parity, [internal ref]): an exact
    #: container length, and the lengths excluded
    container_len: int | None = None
    container_len_ne: set[int] = field(default_factory=set)


@dataclass(frozen=True)
class FactEnv:
    """Environment of analysis facts, keyed by variable name."""
    facts: dict[str, VarFacts] = field(default_factory=dict)
    possible: bool = True   # False means statically infeasible


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_consistent(env: FactEnv) -> bool:
    """False iff the facts are contradictory (the path is statically infeasible)."""
    return env.possible


def _impossible(env: FactEnv) -> FactEnv:
    return FactEnv(facts=env.facts, possible=False)


def _recorded(env: FactEnv, var: str, vf: VarFacts) -> FactEnv:
    """``env`` with ``var`` recorded (possibly as empty facts) and unchanged --
    the shape eregion's carrier leaves behind for an atom it cannot use."""
    facts = dict(env.facts)
    facts[var] = VarFacts(eq=set(vf.eq) if vf.eq is not None else None, neq=set(vf.neq), lo=vf.lo, hi=vf.hi,
                          container_len=vf.container_len, container_len_ne=set(vf.container_len_ne))
    return FactEnv(facts=facts, possible=env.possible)


def refine(env: FactEnv, var: str, op: str, value: Any) -> FactEnv:
    """``env`` narrowed by the atom ``var <op> value`` -- a monotone refinement.

    ``op`` is one of :data:`REFINE_OPS`; ``in`` / ``not in`` take a ``set``.
    A contradiction yields ``possible=False`` (the facts are left as they were).
    Ordering ops apply only to non-bool ints; anything else records the var and
    changes nothing -- SOUND: an atom the arithmetic does not model never
    narrows.  Mirrors ``eregion_commons.fact_repr.apply_constraint`` exactly
    (including its ``in`` bound-overwrite), so the two carriers agree.
    """
    if not env.possible:
        return env
    vf = env.facts.get(var, VarFacts())

    # len(var) == K / len(var) != K (eregion apply_constraint parity): a non-negative int
    # K only; a second, different exact length, or an exact length already excluded,
    # makes the path impossible
    if op in ("container_len_eq", "container_len_ne"):
        if not _is_int(value) or value < 0:
            return _recorded(env, var, vf)
        cl, cl_ne = vf.container_len, set(vf.container_len_ne)
        if op == "container_len_eq":
            if (cl is not None and cl != value) or value in cl_ne:
                return _impossible(env)
            cl = value
        else:
            if cl is not None and cl == value:
                return _impossible(env)
            cl_ne.add(value)
        facts = dict(env.facts)
        facts[var] = VarFacts(eq=set(vf.eq) if vf.eq is not None else None, neq=set(vf.neq),
                              lo=vf.lo, hi=vf.hi, container_len=cl, container_len_ne=cl_ne)
        return FactEnv(facts=facts, possible=True)

    eq = set(vf.eq) if vf.eq is not None else None
    neq = set(vf.neq)
    lo, hi = vf.lo, vf.hi

    if op == "==":
        eq = {value} if eq is None else {v for v in eq if v == value}
        if eq is not None and not eq:
            return _impossible(env)
        if _is_int(value):
            lo = value if lo is None else max(lo, value)
            hi = value if hi is None else min(hi, value)
        neq.discard(value)
    elif op == "!=":
        if eq is not None and value in eq:
            eq.discard(value)
            if not eq:
                return _impossible(env)
        neq.add(value)
    elif op == "in":
        if not isinstance(value, set):
            return _recorded(env, var, vf)  # not a literal collection: nothing to narrow
        eq = set(value) if eq is None else eq & set(value)
        if eq is not None and not eq:
            return _impossible(env)
        neq.difference_update(set(value))
        if eq is not None and all(_is_int(v) for v in eq):
            lo = min(eq) if eq else lo
            hi = max(eq) if eq else hi
    elif op == "not in":
        if not isinstance(value, set):
            return _recorded(env, var, vf)
        if eq is not None:
            eq.difference_update(set(value))
            if not eq:
                return _impossible(env)
        neq.update(set(value))
    elif _is_int(value):
        if op == "<":
            hi = value - 1 if hi is None else min(hi, value - 1)
        elif op == "<=":
            hi = value if hi is None else min(hi, value)
        elif op == ">":
            lo = value + 1 if lo is None else max(lo, value + 1)
        elif op == ">=":
            lo = value if lo is None else max(lo, value)

    if lo is not None and hi is not None and lo > hi:
        return _impossible(env)
    if eq is not None and (lo is not None or hi is not None):
        eq = {
            v for v in eq
            if not _is_int(v) or ((lo is None or v >= lo) and (hi is None or v <= hi))
        }
        if not eq:
            return _impossible(env)

    facts = dict(env.facts)
    facts[var] = VarFacts(eq=eq, neq=neq, lo=lo, hi=hi, container_len=vf.container_len,
                          container_len_ne=set(vf.container_len_ne))
    return FactEnv(facts=facts, possible=True)


def refine_all(env: FactEnv, atoms: Iterable[tuple[str, str, Any]]) -> FactEnv:
    """:func:`refine` folded over ``(var, op, value)`` atoms (a conjunction)."""
    for var, op, value in atoms:
        env = refine(env, var, op, value)
        if not env.possible:
            break
    return env


def join(a: FactEnv, b: FactEnv) -> FactEnv:
    """The union of two envs' possibilities (a disjunction of paths).

    Only vars known on BOTH sides survive; an exact set wider than
    :data:`MAX_CONST_SET` becomes unknown; bounds widen to the outer ones;
    an exclusion survives only if both sides exclude it.  An infeasible side
    contributes nothing.  Mirrors ``eregion_commons.fact_repr.merge_env``.
    """
    if not a.possible:
        return FactEnv(facts=dict(b.facts), possible=b.possible)
    if not b.possible:
        return FactEnv(facts=dict(a.facts), possible=a.possible)
    facts: Dict[str, VarFacts] = {}
    for name in set(a.facts) | set(b.facts):
        va, vb = a.facts.get(name), b.facts.get(name)
        if va is None or vb is None:
            continue
        eq: Optional[set[Any]] = None
        if va.eq is not None and vb.eq is not None:
            eq = set(va.eq) | set(vb.eq)
            if len(eq) > MAX_CONST_SET:
                eq = None
        lo = min(va.lo, vb.lo) if va.lo is not None and vb.lo is not None else None
        hi = max(va.hi, vb.hi) if va.hi is not None and vb.hi is not None else None
        cl = va.container_len if va.container_len is not None and va.container_len == vb.container_len \
            else None
        # a length is excluded only if both paths exclude it (like ``neq``)
        facts[name] = VarFacts(eq=eq, neq=set(va.neq) & set(vb.neq), lo=lo, hi=hi, container_len=cl,
                               container_len_ne=set(va.container_len_ne) & set(vb.container_len_ne))
    return FactEnv(facts=facts, possible=True)


def to_value(vf: VarFacts):
    """The :class:`tharkun.solve.values.Value` these facts describe (sound: every value
    ``vf`` allows is in it).  ``lo`` / ``hi`` bound the INT component only, as
    :func:`refine` applies ordering to non-bool ints only; an exact set pins the value to
    its members; a length exclusion that is not at an end of the known range is dropped
    (a wider fact, never a narrower one)."""
    from tharkun.solve.values import IntFact, Value
    out = Value.top()
    if vf.eq is not None:
        members = Value.bottom()
        for v in vf.eq:
            members = members.join(Value.equal_to(v))     # 0 == 0.0 == False
        out = members
    lo = vf.lo if vf.lo is None or _is_int(vf.lo) else math.ceil(vf.lo)
    hi = vf.hi if vf.hi is None or _is_int(vf.hi) else math.floor(vf.hi)
    if lo is not None or hi is not None:
        out = out.meet(Value(i=IntFact.range(lo, hi)))
    if vf.neq:
        try:
            out = out.meet(Value(neq=frozenset(vf.neq)))
        except TypeError:           # an unhashable exclusion: dropping it only widens
            pass
    if vf.container_len is not None:
        out = out.meet(Value(length=IntFact.const(vf.container_len)))
    return out


def from_value(v, max_set: int = MAX_CONST_SET) -> VarFacts:
    """The :class:`VarFacts` that best describe the :class:`Value` ``v`` (sound: every
    value in ``v`` satisfies the facts).  An exact ``eq`` set is emitted only when ``v``
    is a finite set of at most ``max_set`` scalars of the int / float (points only) /
    str / bytes / bool / None kinds."""
    lo, hi = v.i.lo, v.i.hi
    if v.i.is_bottom():
        lo = hi = None
    eq = None
    names = v.kinds.names
    if names is not None and names <= {"int", "float", "str", "bytes", "bool", "NoneType"}:
        vals: set = set()
        finite = True
        for k in names:
            if k == "int":
                if v.i.is_bottom():
                    continue
                if v.i.lo is None or v.i.hi is None or v.i.hi - v.i.lo + 1 > max_set:
                    finite = False
                    break
                vals.update(range(v.i.lo, v.i.hi + 1))
            elif k in ("str", "bytes"):
                part = v.s if k == "str" else v.by
                if part.consts is None:
                    finite = False
                    break
                vals.update(x for x in part.consts if v.length.contains(len(x)))
            elif k == "float":
                if v.f.nan or (v.f.has_interval() and v.f.lo != v.f.hi):
                    finite = False
                    break
                if v.f.has_interval():
                    vals.add(v.f.lo)
            elif k == "bool":
                vals.update(v.bools if v.bools is not None else (False, True))
            else:
                vals.add(None)
        if finite and len(vals) <= max_set:
            eq = {x for x in vals if not _excluded(v.neq, x)}
    cl = v.length.lo if (not v.length.is_bottom() and v.length.lo is not None
                         and v.length.lo == v.length.hi) else None
    return VarFacts(eq=eq, neq=set(v.neq), lo=lo, hi=hi, container_len=cl)


def _excluded(neq, x) -> bool:
    try:
        return x in neq
    except TypeError:
        return False


__all__ = [
    "FactEnv",
    "MAX_CONST_SET",
    "REFINE_OPS",
    "VarFacts",
    "is_consistent",
    "join",
    "refine",
    "refine_all",
    "from_value",
    "to_value",
]
