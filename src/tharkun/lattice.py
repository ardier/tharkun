"""What is known at a program point — an abstract state that forks and merges.

Facts live in a flat global bag today, with no notion of *where* they hold, and that has
a concrete consequence: ``drive_to_line`` keeps its path condition as a list of guard
strings that **survives backtracking**. When the driver abandons a path it rewinds the
input values but keeps the facts, so the condition handed to every solver is a conjunction
drawn from several different paths — ``(x > 5)`` and ``(not (x > 5))`` can sit in it at
once, because a list of strings has no way to notice.

An abstract state fixes that by being a value rather than an accumulator. It **refines**
along a path (``meet``), it **forks** at a branch — the two edges of ``if G`` get
``state ⊓ G`` and ``state ⊓ ¬G`` — and where paths converge it **merges** (``join``).
Abandoning a path is then just going back to the state before the fork.

**May and must join in opposite directions**, and that is the whole reason the components
are separate classes rather than one bag with a flag:

* ``TypeSet`` and ``ValueSet`` are *may* facts — what a variable could be. At a merge they
  **union**: bytes on one branch and list on the other means bytes-or-list after. This is
  what makes a dynamically typed parameter representable at all.
* ``Clauses`` are *must* facts — what definitely holds. At a merge they **intersect**: a
  constraint that held on only one predecessor does not hold after. Union here would be a
  silent unsoundness, claiming a guard is satisfied when only one path satisfied it.

A single type carrying a ``may``/``must`` flag would let one mislabelled fact merge the
wrong way, so each component's direction is baked into its class and there is nothing to
pass.

Two special elements earn their keep immediately:

``BOTTOM``
    Unreachable. It is the identity of ``join``, so an infeasible edge contributes nothing
    to a merge — which is exactly what a sound decider's dead-branch proof means. Infeasibility
    stops being a special verdict and becomes a lattice value.

``TOP``
    Unconstrained: the entry state, and the identity of ``meet``.

The algebra here is not new to the project, it was just scattered. ``merge_leaf_facts``
(``minleaf/leaf_facts.py``) is already a correct meet, ``FactEnv.possible`` is already
bottom, and ``loop_exit_factenv`` (``condition_solver.py``) is already a join over the
values a loop's ``break`` paths pin. What did not exist anywhere is a *join* with a
partial order behind it, which is what a merge point needs.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from typing import Any


def _canon(text: str) -> str:
    """Canonical form of a clause's text; identity when canonicalization is unavailable."""
    try:
        from tharkun.solve.canonicalize import canonicalize

        return canonicalize(text)
    except Exception:
        return text

# Past this many distinct values a set stops being a useful description and becomes a
# list. Widening to "anything" keeps the domain finite, which is what makes a fixpoint
# terminate at all.
VALUE_SET_CAP = 32


@dataclass(frozen=True)
class Origin:
    """The program point a fact came from — the anchor a flat store cannot express.

    ``polarity`` is what makes this a *control-flow* anchor rather than a source location:
    a fact learned on the ``then`` edge of line 12 is genuinely a different fact from the
    same text on the ``else`` edge, and only the edge distinguishes them.

    Deliberately small, hashable and serializable. A raw ``ast`` node is none of those
    things in the way that matters — two structurally identical guards compare *unequal*
    (nodes have no ``__eq__``), so putting one in a fact's identity silently breaks
    deduplication. The node rides along separately as a cache.
    """

    path: str = ""
    lineno: int = 0
    end_lineno: int = 0
    polarity: str = ""          # "then" | "else" | "" (not from a branch)

    def __str__(self) -> str:
        where = f"{self.path}:{self.lineno}" if self.path else f"L{self.lineno}"
        return f"{where}[{self.polarity}]" if self.polarity else where


@dataclass(frozen=True)
class Clause:
    """One thing that must hold, with the program point that established it.

    ``node`` is a cache for consumers that need the shape rather than the text — RedQueen
    inverting ``int.from_bytes(token[:2], 'big') == 13987`` needs to match on the call, not
    read a string. It is excluded from equality and hashing because two parses of the same
    source are different objects, and including it would defeat the deduplication that
    makes ``Clauses`` a set. Re-parsing loses the original ``lineno`` (measured: 105 → 1),
    which is exactly why ``Origin`` is stored rather than recovered.
    """

    text: str
    origin: Origin = Origin()
    node: Any = field(default=None, compare=False, repr=False, hash=False)

    def __post_init__(self) -> None:
        # Identity is the text, and two spellings of one constraint must be one clause or
        # the intersection at a merge silently drops it. Canonicalize once, here, so every
        # comparison downstream is between canonical forms. Kept off the hot path by
        # canonicalize's own memo. A parse failure returns the text unchanged.
        canon = _canon(self.text)
        if canon != self.text:
            object.__setattr__(self, "text", canon)

    @property
    def names(self) -> frozenset[str]:
        """Free names a probe would have to bind. Builtins are not bindable."""
        from tharkun.solve.constants import BUILTIN_NAMES

        parsed = self.node
        if parsed is None:
            try:
                parsed = ast.parse(self.text, mode="eval").body
            except SyntaxError:
                return frozenset()
        return frozenset(n.id for n in ast.walk(parsed)
                         if isinstance(n, ast.Name) and n.id not in BUILTIN_NAMES)

    def __str__(self) -> str:
        return self.text


class _Lattice:
    """Shared shape: every component knows its own join and meet, and nothing else does."""

    def join(self, other):                       # pragma: no cover - interface
        raise NotImplementedError

    def meet(self, other):                       # pragma: no cover - interface
        raise NotImplementedError

    def is_top(self) -> bool:                    # pragma: no cover - interface
        raise NotImplementedError

    def is_bottom(self) -> bool:
        return False

    def leq(self, other) -> bool:
        """``self ⊑ other`` — true when joining adds nothing."""
        return self.join(other) == other


@dataclass(frozen=True)
class TypeSet(_Lattice):
    """Which types a subject may have. A *may* fact, so it joins by UNION.

    Python is dynamically typed and the signals routinely disagree for good reason: static
    inference reading ``token[:2]`` says ``list``, the run that happened says ``bytes``.
    ``list`` is not false, it is incomplete — the expression admits bytes, list, str and
    tuple alike, and the next path may take any of them. Collapsing to a winner points a
    generator at one shape and can rule out the answer before the search starts.

    ``types is None`` is TOP: nothing known, so anything is possible.
    """

    types: frozenset[str] | None = None

    def join(self, other: "TypeSet") -> "TypeSet":
        if self.types is None or other.types is None:
            return TypeSet(None)                 # one side admits anything
        return TypeSet(self.types | other.types)

    def meet(self, other: "TypeSet") -> "TypeSet":
        if self.types is None:
            return other
        if other.types is None:
            return self
        return TypeSet(self.types & other.types)

    def is_top(self) -> bool:
        return self.types is None

    def is_bottom(self) -> bool:
        return self.types is not None and not self.types


@dataclass(frozen=True)
class ValueSet(_Lattice):
    """Which concrete values a subject may take. A *may* fact, so it joins by UNION.

    Widens to TOP past :data:`VALUE_SET_CAP` distinct values, because past that point the
    set has stopped describing anything and an unbounded one would keep a fixpoint from
    terminating.
    """

    values: frozenset | None = None

    def join(self, other: "ValueSet") -> "ValueSet":
        if self.values is None or other.values is None:
            return ValueSet(None)
        merged = self.values | other.values
        return ValueSet(None if len(merged) > VALUE_SET_CAP else merged)

    def meet(self, other: "ValueSet") -> "ValueSet":
        if self.values is None:
            return other
        if other.values is None:
            return self
        return ValueSet(self.values & other.values)

    def is_top(self) -> bool:
        return self.values is None

    def is_bottom(self) -> bool:
        return self.values is not None and not self.values


@dataclass(frozen=True)
class Interval(_Lattice):
    """A numeric range, with the excluded values a ``!=`` guard establishes.

    Meet is the algebra already used by ``condition_solver._try_intersect_int_constraints``
    — ``>`` raises the floor, ``<`` lowers the ceiling, ``!=`` excludes a point — including
    its emptiness check, which that function has and ``merge_leaf_facts`` lacks. Join is
    the hull, and it must DROP the exclusions: a value excluded on one branch is perfectly
    possible via the other, so carrying exclusions through a merge would be unsound.
    """

    lo: Any = None                               # None = unbounded below
    hi: Any = None                               # None = unbounded above
    excluded: frozenset = frozenset()
    empty: bool = False                          # an explicit bottom

    def join(self, other: "Interval") -> "Interval":
        if self.empty:
            return other
        if other.empty:
            return self
        lo = None if self.lo is None or other.lo is None else min(self.lo, other.lo)
        hi = None if self.hi is None or other.hi is None else max(self.hi, other.hi)
        # Only an exclusion that holds on BOTH sides survives; anything else is reachable.
        return Interval(lo, hi, self.excluded & other.excluded)

    def meet(self, other: "Interval") -> "Interval":
        if self.empty or other.empty:
            return Interval(empty=True)
        lo = other.lo if self.lo is None else (
            self.lo if other.lo is None else max(self.lo, other.lo))
        hi = other.hi if self.hi is None else (
            self.hi if other.hi is None else min(self.hi, other.hi))
        if lo is not None and hi is not None and lo > hi:
            return Interval(empty=True)          # detected, not returned silently
        return Interval(lo, hi, self.excluded | other.excluded)

    def is_top(self) -> bool:
        return not self.empty and self.lo is None and self.hi is None and not self.excluded

    def is_bottom(self) -> bool:
        return self.empty


@dataclass(frozen=True)
class Clauses(_Lattice):
    """Constraints that definitely hold. A *must* fact, so it joins by INTERSECTION.

    This is the component whose direction is easy to get backwards and expensive to get
    wrong. A clause established on one predecessor and not the other does **not** hold
    after the merge; unioning would assert a guard is satisfied when only one path
    satisfied it, and every downstream solver would take that as given.

    Meet is union, because refining along a path only ever adds what must hold.
    """

    clauses: frozenset[Clause] = frozenset()

    def join(self, other: "Clauses") -> "Clauses":
        """Keep only what holds on BOTH sides — matched by MEANING, not by origin.

        Set intersection over ``Clause`` would compare origins too, so the same constraint
        established at two different program points — different lines, different call
        sites — would vanish at every merge. That is sound (facts are lost, never
        invented) but so lossy the join returns nothing in the common case. Two
        predecessors that each established ``x > 5`` really do leave ``x > 5`` holding
        afterwards, wherever they happened to learn it.

        So identity here is the clause text. The surviving clause keeps the origin of the
        side it was found on, which is one real program point rather than an invented one;
        the other is not lost from the store, only from this particular conjunction.
        """
        mine = {c.text: c for c in self.clauses}
        return Clauses(frozenset(mine[t] for t in
                                 mine.keys() & {c.text for c in other.clauses}))

    def meet(self, other: "Clauses") -> "Clauses":
        """Everything from both sides — refining along a path only ever adds.

        Deduplicated by text as well: re-learning the same constraint at a second point
        adds no information, and letting it in twice would make the rendered precondition
        grow without bound along a long path.
        """
        merged: dict[str, Clause] = {c.text: c for c in other.clauses}
        merged.update({c.text: c for c in self.clauses})
        return Clauses(frozenset(merged.values()))

    def is_top(self) -> bool:
        return not self.clauses

    def render(self, names: frozenset[str] | None = None) -> str:
        """The conjunction as a precondition string, for the SMT lane.

        Restricted to clauses whose free names the caller can bind — a clause naming an
        unbound variable is not merely useless to a probe, it is ill-formed. A dropped
        clause stays in the state and applies again as soon as a consumer mentions that
        variable.
        """
        keep = [c.text for c in sorted(self.clauses, key=lambda c: (c.origin.lineno, c.text))
                if names is None or (c.names and c.names <= names)]
        return " and ".join(dict.fromkeys(keep)) or "True"

    def contradictions(self) -> list[tuple[Clause, Clause]]:
        """Pairs where one clause is the negation of the other, up to canonical form.

        A list of strings could hold ``(x > 5)`` and ``(not (x > 5))`` at once and never
        notice; that is what made the old path condition unsound after a backtrack. Because
        clause text is now canonical, the negation test is semantic without a solver:
        ``x > 5`` negated and canonicalized is ``x <= 5``, so the ``(x > 5)`` / ``(x <= 5)``
        pair — which no textual ``(not ...)`` check would catch — is found. A deeper check
        is still a sound decider's job; this is the cheap, sound-for-comparisons layer.
        """
        by_text = {c.text: c for c in self.clauses}
        out: list[tuple[Clause, Clause]] = []
        seen: set[tuple[str, str]] = set()
        for text, clause in by_text.items():
            twin = by_text.get(_canon(f"not ({text})"))
            if twin is not None and twin.text != text:
                key = tuple(sorted((text, twin.text)))
                if key not in seen:
                    seen.add(key)
                    out.append((clause, twin))
        return out


@dataclass(frozen=True)
class VarState(_Lattice):
    """Everything known about one subject: the product of the components above.

    A product lattice joins componentwise, which is precisely what lets may- and must-
    facts about the same variable move in opposite directions at the same merge without
    anyone having to choose.
    """

    types: TypeSet = TypeSet()
    values: ValueSet = ValueSet()
    interval: Interval = Interval()

    def join(self, other: "VarState") -> "VarState":
        return VarState(self.types.join(other.types),
                        self.values.join(other.values),
                        self.interval.join(other.interval))

    def meet(self, other: "VarState") -> "VarState":
        return VarState(self.types.meet(other.types),
                        self.values.meet(other.values),
                        self.interval.meet(other.interval))

    def is_top(self) -> bool:
        return self.types.is_top() and self.values.is_top() and self.interval.is_top()

    def is_bottom(self) -> bool:
        # One empty component makes the whole variable unsatisfiable.
        return (self.types.is_bottom() or self.values.is_bottom()
                or self.interval.is_bottom())


@dataclass(frozen=True)
class FactState(_Lattice):
    """What is known at a program point.

    Refines along a path by ``meet``, forks at a branch, merges by ``join``. Unlike the
    accumulator it replaces, it is a *value*: abandoning a path is going back to the state
    before the fork, so a fact cannot outlive the path that produced it.

    ``unreachable`` is BOTTOM. It is the identity of ``join``, so an edge proved dead
    contributes nothing to a merge — which is what a dead-branch proof means.
    """

    vars: dict[str, VarState] = field(default_factory=dict)
    clauses: Clauses = Clauses()
    unreachable: bool = False

    def __post_init__(self) -> None:
        # Bottom in any component makes the whole state unreachable; deriving it here
        # means no caller can forget to check.
        if not self.unreachable and any(v.is_bottom() for v in self.vars.values()):
            object.__setattr__(self, "unreachable", True)

    # --- lattice operations ----------------------------------------------------
    def join(self, other: "FactState") -> "FactState":
        """Merge two incoming edges. An unreachable edge contributes nothing."""
        if self.unreachable:
            return other
        if other.unreachable:
            return self
        # A variable known on only ONE side is unconstrained after the merge: the other
        # path reached here without establishing anything about it.
        merged = {name: self.vars.get(name, VarState()).join(other.vars.get(name, VarState()))
                  for name in set(self.vars) | set(other.vars)}
        return FactState({k: v for k, v in merged.items() if not v.is_top()},
                         self.clauses.join(other.clauses))

    def meet(self, other: "FactState") -> "FactState":
        """Refine — take another step along the path, or add an edge's condition."""
        if self.unreachable or other.unreachable:
            return BOTTOM
        merged = {name: self.vars.get(name, VarState()).meet(other.vars.get(name, VarState()))
                  for name in set(self.vars) | set(other.vars)}
        return FactState(merged, self.clauses.meet(other.clauses))

    def is_top(self) -> bool:
        return not self.unreachable and not self.vars and self.clauses.is_top()

    def is_bottom(self) -> bool:
        return self.unreachable

    # --- convenience for producers --------------------------------------------
    def with_clause(self, text: str, origin: Origin = Origin(), node: Any = None) -> "FactState":
        """This state plus one more thing that must hold."""
        return replace(self, clauses=self.clauses.meet(
            Clauses(frozenset({Clause(text, origin, node)}))))

    def with_types(self, subject: str, types) -> "FactState":
        """Narrow what *subject* may be. Pass several — a parameter can be more than one."""
        cur = self.vars.get(subject, VarState())
        nxt = replace(cur, types=cur.types.meet(TypeSet(frozenset(types))))
        return replace(self, vars={**self.vars, subject: nxt})

    def with_interval(self, subject: str, lo=None, hi=None, excluded=frozenset()) -> "FactState":
        cur = self.vars.get(subject, VarState())
        nxt = replace(cur, interval=cur.interval.meet(Interval(lo, hi, frozenset(excluded))))
        return replace(self, vars={**self.vars, subject: nxt})

    def with_values(self, subject: str, values) -> "FactState":
        cur = self.vars.get(subject, VarState())
        nxt = replace(cur, values=cur.values.meet(ValueSet(frozenset(values))))
        return replace(self, vars={**self.vars, subject: nxt})

    def with_fresh_clause(self, text: str, origin: Origin = Origin(), node: Any = None) -> "FactState":
        """Like :meth:`with_clause`, but a re-established clause takes the FRESH origin.

        ``Clauses.meet`` dedups by text and keeps the clause already present, so a
        constraint learned a second time further along the path would silently keep its
        first origin — and the flow-sensitivity filter keys on ``origin.lineno``. Dropping
        the text twin first lets the new program point win.
        """
        fresh = Clause(text, origin, node)
        kept = frozenset(c for c in self.clauses.clauses if c.text != fresh.text)
        return replace(self, clauses=Clauses(kept | {fresh}))

    def filtering_clauses(self, keep) -> "FactState":
        """This state with only the clauses for which ``keep(clause)`` is true.

        The per-clause instrument the flow-sensitivity filter needs: unlike :meth:`drop`,
        which forgets everything about a name, this decides clause by clause — so a clause
        whose variable was reassigned *before* it was learned is kept while a stale twin of
        the same variable is dropped. Never touches the ``VarState`` half.
        """
        kept = frozenset(c for c in self.clauses.clauses if keep(c))
        return replace(self, clauses=Clauses(kept))

    def drop(self, names) -> "FactState":
        """This state with everything about *names* forgotten — the flow-sensitivity primitive.

        When a variable is reassigned, every clause and every ``VarState`` that describes
        it stopped being true of the *new* value: ``(a > 100)`` learned before ``a = b``
        says nothing about ``a`` afterwards. Origin cannot distinguish stale from fresh —
        ``Clause`` identity is its canonical text, so the two ``(a > 100)``s are one clause
        to the lattice (trap 2) — so staleness must be enforced actively, here, by matching
        on the names a clause actually mentions.

        A clause naming *any* dropped name goes, and so does each dropped name's own
        ``VarState``. This can OVER-approximate — a clause mentioning both a killed and a
        live variable is dropped whole — which loses precision but never invents a fact:
        the surviving state is always weaker (fewer musts) than the truth, which is the safe
        direction for a precondition. It never adds a clause, so it cannot make a live path
        read dead.
        """
        kill = frozenset(names)
        if not kill:
            return self
        kept = frozenset(c for c in self.clauses.clauses if not (c.names & kill))
        vars_ = {k: v for k, v in self.vars.items() if k not in kill}
        return replace(self, clauses=Clauses(kept), vars=vars_)

    def types_for(self, subject: str) -> frozenset[str] | None:
        """Every plausible type, or None for "anything". Never a single winner."""
        return self.vars.get(subject, VarState()).types.types

    def requires(self, names: frozenset[str] | None = None) -> str:
        """The path condition as a precondition string, for a sound decider and the static solver."""
        return self.clauses.render(names)


#: The entry state: nothing known, so nothing is ruled out.
TOP = FactState()

#: An unreachable program point. Identity of ``join``; absorbing for ``meet``.
BOTTOM = FactState(unreachable=True)


# How many distinct path classes one program point may keep before they are joined
# together. Bounded the same way VALUE_SET_CAP is: exceeding it loses precision, never
# soundness, which is what keeps the structure finite.
PATH_CLASS_BUDGET = 16


def path_class(executed: dict[str, set[int]] | set[int]) -> str:
    """A stable id for "the path this execution took".

    There is no branch-decision *sequence* anywhere in the system — the tracer records
    ``executed`` as a **set** of line numbers, so order and repetition are already gone.
    That set is still a coverage vector, and hashing it gives a usable path class: two
    executions that covered the same lines are treated as the same path and merged, which
    is precisely the aggregation wanted.

    It conflates loop-iteration counts and ordering, so it is an over-approximation —
    merging two executions that differed only in how many times a loop ran costs
    precision, not correctness. And the number of classes is bounded by the number of
    distinct coverage classes, which is what libFuzzer already bounds by admitting only
    feature-distinct inputs to its corpus.
    """
    import hashlib

    if isinstance(executed, (set, frozenset)):
        payload = ",".join(str(n) for n in sorted(executed))
    else:
        payload = ";".join(f"{p}:{','.join(str(n) for n in sorted(lines))}"
                           for p, lines in sorted(executed.items()))
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


#: Where a fact holds: a file and a line. Coarser than an Origin, which also names an edge.
Point = tuple


@dataclass
class FactIndex:
    """Facts as a function of the program point, aggregated as fuzzing runs.

    Two tiers, because they age differently:

    ``prior``
        What static analysis and a sound decider worked out before anything ran. Keyed by
        program point. Does not grow with fuzzing.

    ``observed``
        What executions actually showed, keyed by ``(point, path_class)``. Grows with the
        number of distinct paths, and is bounded by joining classes together at budget.

    Reading a point combines them with **join**, not meet. Static inference is a heuristic
    rather than a sound over-approximation: if the prior says ``x ∈ {int, str}`` and a run
    observes ``bytes``, meeting gives the empty set — BOTTOM, "unreachable" — which is
    flatly wrong about a path that demonstrably just executed. Joining says all three are
    possible, which is true.

    Explosion is handled by joining *more*. When a point exceeds its budget its classes
    are merged down to the bare point: precision degrades, soundness does not, and the
    structure stays finite. This is the widen-past-a-cap discipline ``VALUE_SET_CAP``
    already uses, rather than the truncate-a-ranked-list style used for frontiers — only
    the former is sound by construction.
    """

    prior: dict[Point, FactState] = field(default_factory=dict)
    observed: dict[tuple, FactState] = field(default_factory=dict)
    budget: int = PATH_CLASS_BUDGET
    widened: set = field(default_factory=set)      # points whose classes were collapsed

    # --- writing ---------------------------------------------------------------
    def learn_prior(self, point: Point, state: FactState) -> None:
        """Record what was worked out statically. Joined, since several analyses may speak."""
        self.prior[point] = self.prior.get(point, BOTTOM).join(state)

    def observe(self, point: Point, cls: str, state: FactState) -> None:
        """Record what an execution showed. Same class ⇒ joined, which is the aggregation."""
        if point in self.widened:
            cls = ""                                # this point no longer distinguishes paths
        key = (point, cls)
        self.observed[key] = self.observed.get(key, BOTTOM).join(state)
        if self._classes(point) > self.budget:
            self.widen(point)

    def widen(self, point: Point) -> None:
        """Join every class at *point* into one. The bound, and the only thing that gives it."""
        keys = [k for k in self.observed if k[0] == point]
        if len(keys) <= 1:
            return
        merged = BOTTOM
        for k in keys:
            merged = merged.join(self.observed.pop(k))
        self.observed[(point, "")] = merged
        self.widened.add(point)

    # --- reading ---------------------------------------------------------------
    def at(self, point: Point, cls: str | None = None) -> FactState:
        """Everything believed about *point*, optionally along one path class."""
        state = self.prior.get(point, BOTTOM)
        for (p, c), s in self.observed.items():
            if p == point and (cls is None or c == cls or c == ""):
                state = state.join(s)
        return state

    def _classes(self, point: Point) -> int:
        return sum(1 for k in self.observed if k[0] == point)

    # --- what a phase-2 decision would need ------------------------------------
    def clause_pressure(self) -> dict[str, int]:
        """Whether clause *representation* is exploding, as opposed to paths.

        ``distinct`` counts clauses by text; ``semantic`` counts them after stripping the
        spacing and operand order that ``ast.unparse`` fixes arbitrarily. A ratio well
        above 1 means near-duplicates are piling up and normalization — or, past that, an
        e-graph — would pay. Reported rather than acted on, so the decision has data
        behind it instead of a guess.
        """
        texts: set[str] = set()
        for s in list(self.prior.values()) + list(self.observed.values()):
            texts.update(c.text for c in s.clauses.clauses)
        rough = {"".join(sorted(t.replace(" ", ""))) for t in texts}
        return {"distinct": len(texts), "semantic": len(rough),
                "points": len(self.prior), "classes": len(self.observed),
                "widened": len(self.widened)}
