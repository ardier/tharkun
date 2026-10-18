"""One class per rule, one visitor per engine ([internal ref]): the shape of the whole contract registry.

Today a contract is data (:class:`~tharkun.solve.contracts.types.Contract`: what a callable may raise and return,
tested on CPython) and every engine reads it through its own adapter (``adapters/{absint,concolic,draws,...}.py``)
plus a hand table of its own.  Which engine understands which callable is then a fact about scattered ``if``
chains, and an engine that does not understand one fails in whatever way its author happened to write.

This module makes that structure explicit, for ALL contracts (owner, 2026-10-05: "all contracts or none"):

* a :class:`Rule` is ONE callable (or one parametrized family: ``str.split``/``str.rsplit``, the 57 ``math``
  functions) -- the thing the claims are about.  It holds the :class:`Contract` (raises/returns/effects, claim-level
  ``tested`` flags), the call :class:`Shape` (arities, keywords), the NATIVE semantics (CPython itself: what the
  contract tests run against, and what a candidate generator is checked against) and ``accept(visitor, ctx)``;
* a :class:`Visitor` is ONE engine (absint, _symexec, the candidate generator, the Atheris seeder, the fact
  publisher, ...), with one ``visit_<kind>(rule, ctx)`` method per rule kind it understands.  A rule kind it does not
  understand is a NAMED refusal (:class:`Refused`: engine, qualname, reason -- data), never a silent pass; a kind
  that cannot apply by design is declared :class:`NotApplicable` with its reason.

Adding a rule is one class; supporting it in an engine is one visit method.  Which engine consumes which rule is
static (:meth:`Visitor.implements`, :meth:`Visitor.cell`), which gives the engine-by-contract audit table without
probing (``tests/contracts/_engine_audit.py``, [internal ref]).

``registry.get(qualname)`` / ``all_contracts()`` / ``for_receiver`` keep working unchanged: :func:`register_rule`
registers each rule's ``Contract`` in :data:`registry.REGISTRY`, so the adapters that read contracts need not know
rules exist.  Nothing here changes a verdict; the claim-level ``tested`` flags are the Contract's, untouched.

SOLVING by ``docs/LAYERS.md``'s rule: stdlib and ``tharkun.solve`` only; nothing here runs a target.
"""

from __future__ import annotations

import bisect
import builtins
import collections
import copy
import functools
import heapq
import io
import itertools
import math
import operator
import random
import re
from dataclasses import dataclass
from typing import Any, ClassVar, Iterator

from tharkun.solve.contracts.types import Contract

__all__ = ["Shape", "Refused", "NotApplicable", "Rule", "Visitor", "register_rule", "all_rules", "get_rule",
           "RULES", "native_module_names", "native_module"]


@dataclass(frozen=True)
class Shape:
    """The call shape a rule's claims are valid for (``adapters/common.py::CALL_SHAPE``, generalised): the number of
    positional arguments (receiver included for a method) and whether keywords are allowed.  A call of another
    shape raises ``TypeError`` before the callable's body runs, so no other claim of the rule applies to it."""

    min_args: int = 0
    max_args: int | None = None            # None: no upper bound
    keywords_ok: bool = False
    #: for an operator whose claims depend on the KIND of an argument (``str.__getitem__``: ``"int"`` | ``"slice"``)
    key_kinds: tuple[str, ...] = ()

    def accepts(self, arg_count: int, has_keywords: bool = False) -> bool:
        if has_keywords and not self.keywords_ok:
            return False
        return arg_count >= self.min_args and (self.max_args is None or arg_count <= self.max_args)


@dataclass(frozen=True)
class Refused:
    """What a visitor answers for a rule kind it has no visit method for, and what its engine reports at run time
    (absint's ``Unsupported``, _symexec's skipped entry, the candidate generator's "no candidate")."""

    engine: str
    qualname: str
    reason: str

    def __str__(self) -> str:
        return f"{self.engine}: {self.qualname}: {self.reason}"


@dataclass(frozen=True)
class NotApplicable:
    """A rule kind an engine cannot apply BY DESIGN (the exact product has no library calls; the stationary proof
    needs an induction argument): declared on the visitor, distinct from a refusal by limitation."""

    engine: str
    qualname: str
    reason: str


_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _snake(name: str) -> str:
    return _CAMEL.sub("_", name).lower()


#: qualname -> the rule instance, filled by :func:`register_rule`
RULES: dict[str, "Rule"] = {}


class Rule:
    """One callable's rule (or, with ``names``, one member of a parametrized family).

    Subclass, set ``names`` (the qualnames the class covers) and give ``contract_for(qualname)``; or set the
    ``CONTRACT`` class attribute for a single-qualname rule.  ``kind`` is the visitor dispatch key: the snake_case of
    the class name unless set (``StrSplit`` -> ``str_split`` -> ``visit_str_split``).  :func:`register_rule` makes one
    instance per qualname.
    """

    #: the qualnames this class covers (one instance each); a single-qualname rule has one entry
    names: ClassVar[tuple[str, ...]] = ()
    #: the visitor dispatch key; defaults to the snake_case of the class name
    kind: ClassVar[str] = ""
    #: the claims, for a single-qualname rule (a family overrides :meth:`contract_for`)
    CONTRACT: ClassVar[Contract | None] = None
    #: the call shape the claims are valid for (a family overrides :meth:`shape_for` when members differ).  ``None`` =
    #: NO hand-verified shape: the discipline of ``adapters/common.py::CALL_SHAPE`` ("a qualname absent from the table
    #: gets none of the precise facts") -- :meth:`applicable` is False for every call.  A rule opts in by naming a
    #: shape it has swept against CPython.
    SHAPE: ClassVar[Shape | None] = None
    SAMPLES: ClassVar[dict[str, str]] = {}
    TWINS: ClassVar[dict[str, str]] = {}

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if "kind" not in cls.__dict__ or not cls.__dict__["kind"]:
            cls.kind = _snake(cls.__name__)

    def __init__(self, qualname: str) -> None:
        if qualname not in self.names:
            raise ValueError(f"{type(self).__name__} does not cover {qualname!r} (names: {self.names})")
        self.qualname = qualname

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.qualname}>"

    # --- what the rule says ------------------------------------------------------------------------
    @classmethod
    def contract_for(cls, qualname: str) -> Contract:
        """The :class:`Contract` of ``qualname`` (overridden by a table-driven family)."""
        if cls.CONTRACT is None:
            raise NotImplementedError(f"{cls.__name__} defines neither `CONTRACT` nor `contract_for`")
        return cls.CONTRACT

    @classmethod
    def shape_for(cls, qualname: str) -> Shape | None:
        return cls.SHAPE

    @property
    def contract(self) -> Contract:
        """The :class:`Contract` of this rule's qualname."""
        return self.contract_for(self.qualname)

    @property
    def shape(self) -> Shape | None:
        """The verified call shape, or ``None`` (unverified: no precise fact applies to any call)."""
        return self.shape_for(self.qualname)

    def raise_condition(self, exc: str) -> str | None:
        """The TESTED, precise condition under which the rule raises property class ``exc``; ``None`` when the claim
        is untested, ``"unknown"`` or ``"never"`` (the adapters' own reading, one place)."""
        from tharkun.solve.contracts.adapters.common import raise_condition
        return raise_condition(self.contract, exc)

    def never_raises(self, exc: str) -> bool:
        from tharkun.solve.contracts.adapters.common import never_raises
        return never_raises(self.contract, exc)

    def may_raise(self, exc: str) -> bool:
        return not self.never_raises(exc)

    @property
    def tested(self) -> bool:
        return self.contract.tested

    def applicable(self, arg_count: int, has_keywords: bool = False) -> bool:
        """Do the rule's precise claims apply to a call of this shape?  A call of another shape gets none of them, and
        a rule with no verified shape (``SHAPE = None``) applies to NO call: unverified is refusal, never "any"."""
        shape = self.shape
        return shape is not None and shape.accepts(arg_count, has_keywords)

    # --- the native semantics (CPython itself) -----------------------------------------------------
    def native(self, *args: Any, **kwargs: Any) -> Any:
        """Apply the rule to concrete values with CPython's own implementation: what the contract tests run
        against (3.14.4), what a model is checked against, and what a candidate generator forward-checks with.
        The default resolves the qualname (``len``, ``str.split``, ``math.sqrt``); a rule whose callable cannot be
        reached that way (an operator, a protocol) overrides it."""
        return resolve_native(self.qualname)(*args, **kwargs)

    # --- test support (tests only) -----------------------------------------------------------------
    @classmethod
    def sample(cls, qualname: str) -> str | None:
        """A valid program that calls ``qualname`` (the engine-audit table's per-rule probe), or ``None``."""
        return cls.SAMPLES.get(qualname)

    @classmethod
    def twin(cls, qualname: str) -> str | None:
        """The same call with the wrong number of arguments (a ``TypeError`` on CPython), or ``None``."""
        return cls.TWINS.get(qualname)

    # --- the visitor protocol ----------------------------------------------------------------------
    def accept(self, visitor: "Visitor", ctx: Any = None) -> Any:
        """``visitor.visit_<kind>(self, ctx)``, or the visitor's named refusal.  ``ctx`` is the engine's own context
        (an abstract state, a symbolic runner, a target value); it is NOT assumed to be a call site, since a
        protocol rule has none."""
        visit = getattr(visitor, "visit_" + self.kind, None)
        if visit is None:
            return visitor.visit_default(self, ctx)
        return visit(self, ctx)


#: the standard-library modules whose functions the contracts cover, imported statically (``solve/`` loads no module by
#: name: ``tests/test_layering.py``).  A module-level qualname's native is looked up here.
_NATIVE_MODULES = {
    "bisect": bisect, "collections": collections, "copy": copy, "functools": functools, "heapq": heapq, "io": io,
    "itertools": itertools, "math": math, "operator": operator, "random": random, "re": re,
}


def native_module_names() -> frozenset[str]:
    """The real stdlib module names represented in the native contract table."""
    return frozenset(_NATIVE_MODULES)


def native_module(name: str):
    """Return a statically imported native module from the contract table, if present."""
    return _NATIVE_MODULES.get(name)


def resolve_native(qualname: str):
    """The CPython callable for a registry qualname: a builtin (``len``), ``TYPE.method`` (``str.split`` is the
    unbound method), or ``module.name`` for one of :data:`_NATIVE_MODULES` (``math.sqrt``, ``re.match``).
    ``KeyError`` if there is none (a protocol has no callable)."""
    if "." not in qualname:
        try:
            return getattr(builtins, qualname)
        except AttributeError:
            raise KeyError(qualname) from None
    head, _, tail = qualname.partition(".")
    owner = getattr(builtins, head, None)
    if owner is None:
        owner = _NATIVE_MODULES.get(head)
    if owner is None:
        raise KeyError(qualname)
    obj = owner
    for part in tail.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            raise KeyError(qualname) from None
    return obj


class Visitor:
    """One engine's view of the rules: a ``visit_<kind>(rule, ctx)`` method per rule kind it understands.

    The default for a kind with no method is :meth:`visit_default`, a NAMED :class:`Refused`: the engine says which
    rule it could not handle and why, in data.  What a visit returns is the engine's own (an abstract value, a z3
    term, candidate inputs, evidence to publish); a refusal is always a :class:`Refused`.
    """

    #: the engine's name in refusals and in the audit table
    engine: ClassVar[str] = ""
    #: rule kind -> reason: kinds that cannot apply BY DESIGN (the audit's ``-`` cell), distinct from a limitation
    not_applicable: ClassVar[dict[str, str]] = {}

    def visit_default(self, rule: Rule, ctx: Any = None) -> Refused:
        why = self.not_applicable.get(rule.kind)
        reason = why if why is not None else f"{self.engine or type(self).__name__} has no visit for rule kind {rule.kind!r}"
        return Refused(self.engine or type(self).__name__, rule.qualname, reason)

    @classmethod
    def implements(cls) -> frozenset[str]:
        """The rule kinds this visitor has a ``visit_<kind>`` method for -- STATIC (no rule is applied)."""
        out = set()
        for name in dir(cls):
            if name.startswith("visit_") and name != "visit_default" and callable(getattr(cls, name)):
                out.add(name[len("visit_"):])
        return frozenset(out)

    @classmethod
    def cell(cls, rule: Rule) -> str:
        """The audit table's static cell for (rule, this engine): ``"C"`` consumed (a visit method exists), ``"-"``
        not applicable by design, ``"R"`` named refusal.  The probing cells (a hand model that never reads the
        contract, a generic refusal, a visit that is never called) stay with the audit's run-time probes."""
        if rule.kind in cls.implements():
            return "C"
        return "-" if rule.kind in cls.not_applicable else "R"


def register_rule(cls: type[Rule]) -> type[Rule]:
    """Register ``cls``: one instance per qualname in ``cls.names`` goes into :data:`RULES`, and each rule's
    Contract into :data:`registry.REGISTRY` (so ``get(qualname)`` and every adapter work unchanged).  Usable as a
    decorator.  A qualname registered twice -- as a rule or as a plain contract -- is a bug."""
    from tharkun.solve.contracts import registry

    if not cls.names:
        raise ValueError(f"{cls.__name__} covers no qualname (set `names`)")
    for qualname in cls.names:
        if qualname in RULES or qualname in registry.REGISTRY:
            raise ValueError(f"duplicate rule or contract for {qualname!r}")
    for qualname in cls.names:
        rule = cls(qualname)
        contract = rule.contract
        if contract.qualname != qualname:
            raise ValueError(f"{cls.__name__}: contract for {qualname!r} names {contract.qualname!r}")
        RULES[qualname] = rule
        registry.REGISTRY[qualname] = contract
    return cls


def all_rules() -> tuple[Rule, ...]:
    """Every registered rule, qualname-sorted (the audit has one row per qualname)."""
    from tharkun.solve.contracts.registry import load_all
    load_all()
    return tuple(RULES[k] for k in sorted(RULES))


def get_rule(qualname: str) -> Rule | None:
    from tharkun.solve.contracts.registry import load_all
    load_all()
    return RULES.get(qualname)


def kinds() -> Iterator[str]:
    """The distinct rule kinds registered (the visitor interface's vocabulary)."""
    return iter(sorted({r.kind for r in all_rules()}))
