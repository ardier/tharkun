"""Shared facts and memoized static decisions for the solving strategies.

Static results are deterministic for an unchanged query. Snapshot and minleaf
results depend on live state; Hypothesis results are stochastic. Those results
are recorded as evidence, never reused as deterministic answers."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator

# How a strategy behaves under repetition — see the module docstring.
DETERMINISTIC = "deterministic"
STATEFUL = "stateful"
STOCHASTIC = "stochastic"

# Which class each strategy falls into. Getting this wrong in the "deterministic"
# direction would serve a stale answer for a question whose live state has moved on, so
# anything that reads program state is listed explicitly as STATEFUL.
STRATEGY_KIND: dict[str, str] = {
    "static": DETERMINISTIC,
    "snapshot": STATEFUL,
    "minleaf": STATEFUL,
    "redqueen": STATEFUL,
    "hypothesis": STOCHASTIC,
    "hypothesis-facts": STOCHASTIC,
    "reach-search": STOCHASTIC,
}


def is_memoizable(strategy: str) -> bool:
    """True when repeating *strategy* on an unchanged question cannot tell us anything.

    Unknown strategies are treated as NOT memoizable: a wrong "yes" here silently serves
    a stale answer, while a wrong "no" only costs time.
    """
    return STRATEGY_KIND.get(strategy.split(":")[0]) == DETERMINISTIC


@dataclass(frozen=True)
class DecisionKey:
    """Identity of a question put to a deterministic decider.

    The precondition is part of the identity, not context: the same guard under a
    stronger precondition is a *different* question, and is exactly the case where
    re-asking is worthwhile (a guard that was satisfiable in isolation can be infeasible
    once the callers' pinned values are folded in).
    """

    path: str
    line: int
    desired: str = "then"
    requires: str = "True"
    # The assumed typing is part of the question too: a sound decider is asked once per
    # candidate joint typing, and a different typing is a different decision problem.
    types: tuple[tuple[str, str], ...] = ()

    @staticmethod
    def of(path: str, line: int, desired: str, requires: str,
           types: dict | None = None) -> "DecisionKey":
        import os

        return DecisionKey(os.path.realpath(path), line, desired, requires,
                           tuple(sorted((types or {}).items())))


_PARSE_CACHE: dict[str, Any] = {}


def _parsed(text: str):
    """The expression form of a constraint, parsed once per distinct text."""
    node = _PARSE_CACHE.get(text)
    if node is None:
        try:
            node = ast.parse(text, mode="eval").body
        except SyntaxError:
            node = False                      # remembered failure; do not re-parse
        _PARSE_CACHE[text] = node
    return node or None


@dataclass(frozen=True)
class Constraint:
    """Something that must hold of *subject*, kept as an expression rather than a value.

    Storing the value loses the knowledge. A search that observes
    ``int.from_bytes(token[:2], "big") == 13987`` knows a great deal more than the number
    13987: it knows which bytes, in which order, produce it. Handing consumers the bare
    number asks each of them to rediscover the encoding — and measured, a search aimed
    with the number alone still failed, while RedQueen given the *expression* inverts it
    directly, because it can match the ``int.from_bytes`` shape.

    Both representations are available because consumers differ: a sound decider and the
    static solver want a precondition STRING, while RedQueen, minleaf and the predicate
    compiler want to walk the AST. Text is the stored form — it is what serializes if the
    store ever crosses a process boundary — and the node is derived and memoized, so the
    two cannot drift.
    """

    subject: str                  # the parameter this constrains
    text: str                     # canonical source form, e.g. "len(key) != 7"
    source: str = ""              # which strategy produced it
    certified: bool = False       # backed by a sound procedure, not inferred
    confidence: float = 1.0

    @property
    def node(self):
        """Parsed form for structural consumers; None when the text will not parse."""
        return _parsed(self.text)

    @property
    def names(self) -> set[str]:
        """Free names the expression mentions — what a probe would have to BIND.

        Builtins are not bindable and must not count: ``int.from_bytes(token[:2], 'big')
        == 13987`` mentions ``int``, and treating that as a required binding made the
        clause look unsatisfiable-by-this-consumer and silently dropped it from every
        projection.
        """
        n = self.node
        if n is None:
            return set()
        from tharkun.solve.constants import BUILTIN_NAMES

        return {x.id for x in ast.walk(n)
                if isinstance(x, ast.Name) and x.id not in BUILTIN_NAMES}

    @staticmethod
    def of_node(subject: str, node, **kw) -> "Constraint":
        """Build from an AST node, for producers that already have one."""
        return Constraint(subject=subject, text=ast.unparse(node), **kw)

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Fact:
    """One piece of evidence about a subject, and where it came from.

    ``kind`` is an open vocabulary — ``"type"``, ``"range"``, ``"width"``, ``"pinned"``,
    ``"operand"``, ``"path_condition"``.  ``subject`` is what it is about, usually a
    parameter name.  Provenance is kept so the scheduler can score which strategy is
    actually paying, and so a fact from a heuristic source is never mistaken for a
    certified one.
    """

    kind: str
    subject: str
    value: Any
    source: str = ""
    confidence: float = 1.0
    certified: bool = False       # backed by a sound procedure, not inferred


class FactStore:
    """Shared, additive knowledge base.

    Deliberately not a cache with an eviction policy: a decided verdict and a harvested
    operand stay true for the run, and the value of the store is that a later strategy
    sees what an earlier one learned.
    """

    def __init__(self) -> None:
        self._constraints: list[Constraint] = []
        # Namespaced by strategy. Two deciders asked about the same guard return
        # different SHAPES of answer (a solver result vs a ConditionSolution), so a
        # shared key space serves one strategy the other's verdict — which fails loudly
        # if you are lucky and silently if you are not.
        self._decisions: dict[tuple[str, DecisionKey], Any] = {}
        self._facts: dict[tuple[str, str], list[Fact]] = {}
        self.hits: Counter = Counter()      # per strategy: questions served from the store
        self.misses: Counter = Counter()    # per strategy: questions that had to be run
        self.yields: Counter = Counter()    # per strategy: answers that were committed
        # Bumped whenever anything is genuinely learned. A frontier that already resisted
        # everything at epoch E cannot be cracked again at epoch E — nothing has changed
        # for a solver to use — so it is skipped until the epoch moves. Memoizing the
        # individual queries is not enough: the ROUND still pays for the stateful and
        # search tiers, which on a toy with one unsolvable guard was 26.7s of five rounds
        # that produced no coverage at all.
        self.epoch: int = 0
        self._exhausted: dict[Any, int] = {}
        self._retries: dict[Any, int] = {}
        self.skipped: Counter = Counter()   # per scope kind: attacks not re-attempted

    # --- deterministic decisions ------------------------------------------------
    def decided(self, key: DecisionKey, source: str) -> bool:
        return (source, key) in self._decisions

    def decision(self, key: DecisionKey, source: str, default: Any = None) -> Any:
        return self._decisions.get((source, key), default)

    def record_decision(self, key: DecisionKey, verdict: Any, *, source: str) -> None:
        """Remember a verdict, but only for a decider that cannot change its mind."""
        if is_memoizable(source) and (source, key) not in self._decisions:
            self._decisions[(source, key)] = verdict
            self.epoch += 1

    def lookup_or_run(self, key: DecisionKey, source: str, run) -> Any:
        """The verdict for *key*, from the store when possible, else by running *run*.

        This is the whole point of the split: for a decision procedure a repeat is
        guaranteed waste, and on a measured ntlm_auth run that was 96.6% of the calls.
        """
        if is_memoizable(source) and (source, key) in self._decisions:
            self.hits[source] += 1
            return self._decisions[(source, key)]
        self.misses[source] += 1
        verdict = run()
        self.record_decision(key, verdict, source=source)
        return verdict

    # --- facts ------------------------------------------------------------------
    def add(self, fact: Fact) -> None:
        bucket = self._facts.setdefault((fact.kind, fact.subject), [])
        if fact not in bucket:
            bucket.append(fact)
            self.epoch += 1

    def add_all(self, facts) -> None:
        for f in facts:
            self.add(f)

    def get(self, kind: str, subject: str) -> list[Fact]:
        """Facts of *kind* about *subject*, certified first, then by confidence."""
        return sorted(self._facts.get((kind, subject), []),
                      key=lambda f: (not f.certified, -f.confidence))

    def best(self, kind: str, subject: str) -> Fact | None:
        got = self.get(kind, subject)
        return got[0] if got else None

    def types_for(self, subject: str, max_types: int = 4) -> list[str]:
        """Every plausible type for *subject*, most likely first — not just one.

        Python is dynamically typed, so a parameter genuinely can be more than one thing,
        and the signals that disagree are usually both right about something. Static
        inference reading ``token[:2]`` says ``list``; the run that happened says
        ``bytes``. ``list`` is not wrong, it is *incomplete* — the guard admits bytes,
        list, str and tuple alike — and the next run may well take a different path with
        a different type.

        Collapsing to a single winner throws that away and points a generator at one
        shape. The rest of the codebase already knows better: ``type_hints`` fuses
        signals into a TypeDistribution and ``candidate_typings`` enumerates joint
        typings so a sound decider can solve each. This is the same idea for the store —
        callers should generate over the union rather than bet on the mode.

        Ordered certified-then-confidence, deduplicated, so a consumer that can only take
        one still takes the best one.
        """
        return list(dict.fromkeys(f.value for f in self.get("type", subject)
                                  if isinstance(f.value, str)))[:max_types]

    def subjects(self, kind: str) -> list[str]:
        return sorted({s for (k, s) in self._facts if k == kind})

    def __iter__(self) -> Iterator[Fact]:
        for bucket in self._facts.values():
            yield from bucket

    def __len__(self) -> int:
        return sum(len(b) for b in self._facts.values())

    # --- constraints, and the projections consumers actually take ---------------
    def require(self, c: Constraint) -> None:
        """Record something that must hold. Additive; duplicates are ignored."""
        if c not in self._constraints:
            self._constraints.append(c)
            self.epoch += 1

    def constraints(self, names: set[str] | None = None) -> list[Constraint]:
        """Constraints whose free names the caller can bind, certified first.

        A clause naming a variable the consumer cannot bind is not merely useless — for a
        probe it is ill-formed — so filtering is the caller's contract, not an
        optimisation. A dropped clause stays in the store and applies again the moment a
        consumer does mention that variable.
        """
        out = [c for c in self._constraints
               if names is None or (c.names and c.names <= names)]
        return sorted(out, key=lambda c: (not c.certified, -c.confidence))

    def requires_str(self, names: set[str], base: str = "True") -> str:
        """Projection for a sound decider and the static solver: a precondition string."""
        clauses = [c.text for c in self.constraints(names)]
        if base and base != "True":
            clauses.insert(0, base)
        return " and ".join(dict.fromkeys(clauses)) or "True"

    def predicate_src(self, names: set[str]) -> str:
        """Projection for a search: the conjunction, as a compilable expression."""
        return self.requires_str(names)

    def goals_for(self, subject: str) -> list[Constraint]:
        """Projection for RedQueen: what must hold of one input, as expressions.

        RedQueen's whole job is inverting the transform between an input and the value a
        comparison demands, so it needs the shape — this is the consumer that a bare
        value cannot serve.
        """
        return [c for c in self.constraints() if c.subject == subject]

    # --- what is worth attempting again -----------------------------------------
    def bump(self, reason: str = "") -> None:
        """Record that the world moved — new coverage, a new seed, a new frontier.

        Coverage counts as new information even though it is not a *fact*: a frontier
        that resisted from one base can fall from another, so widening what we can reach
        must re-open everything that was written off.
        """
        self.epoch += 1

    def mark_exhausted(self, scope: Any, *, retriable: bool = False,
                       budget: int = 2) -> None:
        """*scope* was attacked with everything available and did not yield.

        ``retriable`` says a *randomized* search was part of what failed. A deterministic
        portfolio that comes up empty on unchanged state will come up empty again, so
        writing it off is free; a search will not necessarily, and the epoch cannot see
        the difference — a failed search records nothing, so it never moves the epoch and
        the scope would be shut out permanently. Such a scope gets a retry budget first,
        and is only written off once the budget is spent.
        """
        if retriable:
            left = self._retries.get(scope, budget)
            if left > 0:
                self._retries[scope] = left - 1
                return
        self._exhausted[scope] = self.epoch

    def is_exhausted(self, scope: Any) -> bool:
        """True when *scope* already resisted everything and nothing has been learned since.

        Sound because it is epoch-gated, not permanent: the moment any strategy records a
        decision, a fact, or new coverage, every exhausted scope is live again.
        """
        seen = self._exhausted.get(scope)
        return seen is not None and seen == self.epoch

    def retries_left(self, scope: Any, budget: int = 2) -> int:
        """Search attempts still owed to *scope* before it may be written off."""
        return self._retries.get(scope, budget)

    # --- accounting, for a scheduler to allocate on -----------------------------
    def credit(self, source: str) -> None:
        """Record that *source* produced an answer that was actually committed."""
        self.yields[source.split(":")[0].split("@")[0]] += 1

    def report(self) -> dict[str, dict[str, int]]:
        """Per-strategy hits / misses / committed answers.

        The numbers a power schedule needs: a strategy invoked 1158 times for 0 committed
        answers has earned less energy than one invoked 30 times for 30.
        """
        names = set(self.hits) | set(self.misses) | set(self.yields)
        return {n: {"served": self.hits[n], "ran": self.misses[n],
                    "committed": self.yields[n]} for n in sorted(names)}
