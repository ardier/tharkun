"""Semi-naive fixed-point scheduling over Tharkun's shared KnowledgeBase."""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Protocol

from tharkun.knowledge import Evidence, EvidenceKey, KnowledgeBase


@dataclass(frozen=True)
class Emission:
    """One pass proposal to publish in the shared knowledge base."""

    kind: str
    key: str
    value: Any
    parents: tuple[EvidenceKey, ...] = ()


@dataclass(frozen=True)
class RepeatPolicy:
    """How often a pass may sample one unchanged evidence epoch."""

    trials_per_activation: int = 1
    no_novelty_limit: int = 1

    def __post_init__(self) -> None:
        if self.trials_per_activation < 1:
            raise ValueError("trials_per_activation must be at least one")
        if self.no_novelty_limit < 1:
            raise ValueError("no_novelty_limit must be at least one")

    @classmethod
    def exploratory(cls, trials: int = 3, no_novelty: int = 2) -> "RepeatPolicy":
        """A bounded policy for nondeterministic generators."""

        return cls(trials, no_novelty)


class AvengersPass(Protocol):
    """A strategy that consumes evidence and proposes more evidence."""

    name: str
    consumes: frozenset[str]
    repeat: RepeatPolicy

    def run(
        self, knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]: ...


@dataclass(frozen=True)
class FunctionPass:
    """Convenient pass implementation backed by a Python callable."""

    name: str
    consumes: frozenset[str]
    function: Callable[[KnowledgeBase, tuple[Evidence, ...]], Iterable[Emission]]
    repeat: RepeatPolicy = field(default_factory=RepeatPolicy)

    def run(
        self, knowledge: KnowledgeBase, delta: tuple[Evidence, ...]
    ) -> Iterable[Emission]:
        return self.function(knowledge, delta)


@dataclass(frozen=True)
class PassMetrics:
    """What one strategy cost and what it was worth.

    ``novel`` counts emissions the knowledge base accepted as new canonical
    items, so it is the only field a schedule should rank on: a pass that
    re-proposes an existing recipe adds provenance, not progress.
    """

    name: str
    activations: int = 0
    runs: int = 0
    elapsed_s: float = 0.0
    emissions: int = 0
    novel: int = 0
    failures: int = 0

    @property
    def yield_per_second(self) -> float:
        """Novel evidence per second of wall time, ``inf`` before any cost."""

        if self.elapsed_s <= 0.0:
            return float("inf") if self.novel else 0.0
        return self.novel / self.elapsed_s


class Scheduler(Protocol):
    """Chooses the order strategies are offered their evidence delta in."""

    def order(
        self,
        passes: tuple[AvengersPass, ...],
        metrics: Mapping[str, PassMetrics],
    ) -> tuple[AvengersPass, ...]: ...


@dataclass(frozen=True)
class PortfolioOrder:
    """Keep the portfolio's declared order — cheap strategies were put first."""

    name: str = "portfolio"

    def order(
        self,
        passes: tuple[AvengersPass, ...],
        metrics: Mapping[str, PassMetrics],
    ) -> tuple[AvengersPass, ...]:
        return passes


@dataclass(frozen=True)
class YieldOrder:
    """Offer unmeasured strategies first, then the best novelty per second.

    A strategy nobody has run yet has no measurement to rank on, and skipping
    it would make the ranking self-fulfilling, so it is always tried before a
    measured one. Ties keep the portfolio's declared order.
    """

    name: str = "yield"

    def order(
        self,
        passes: tuple[AvengersPass, ...],
        metrics: Mapping[str, PassMetrics],
    ) -> tuple[AvengersPass, ...]:
        declared = {item.name: index for index, item in enumerate(passes)}

        def rank(item: AvengersPass) -> tuple[int, float, int]:
            measured = metrics.get(item.name)
            if measured is None or measured.runs == 0:
                return (0, 0.0, declared[item.name])
            return (1, -measured.yield_per_second, declared[item.name])

        return tuple(sorted(passes, key=rank))


@dataclass(frozen=True)
class SaturationBudget:
    """Independent ceilings for a fixed-point run."""

    max_seconds: float = 30.0
    max_evidence: int = 10_000
    max_pass_runs: int = 10_000
    # TOTAL wall time one strategy may spend across the whole run, not per
    # activation: the point is that a slow backend cannot starve the rest of
    # the portfolio, and a per-activation ceiling would let it do exactly that
    # by being activated again.
    max_pass_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_seconds <= 0 or self.max_evidence < 1 or self.max_pass_runs < 1:
            raise ValueError("all saturation budgets must be positive")
        if self.max_pass_seconds is not None and self.max_pass_seconds <= 0:
            raise ValueError("max_pass_seconds must be positive when set")


@dataclass(frozen=True)
class SaturationReport:
    stop_reason: str
    pass_runs: dict[str, int]
    evidence_count: int
    duration_s: float
    failures: tuple[Evidence, ...]
    metrics: Mapping[str, PassMetrics] = field(default_factory=dict)
    first_result_s: float | None = None

    @property
    def saturated(self) -> bool:
        return self.stop_reason == "saturated"


@dataclass
class _PassState:
    cursor: int = 0
    activated: bool = False


class Avengers:
    """Run independent strategies until their shared evidence saturates."""

    def __init__(
        self,
        passes: Iterable[AvengersPass],
        *,
        scheduler: Scheduler | None = None,
    ) -> None:
        self.passes = tuple(passes)
        self.scheduler: Scheduler = scheduler if scheduler is not None else PortfolioOrder()
        names = [item.name for item in self.passes]
        if len(names) != len(set(names)):
            raise ValueError("Avengers pass names must be unique")

    def run(
        self,
        initial: Iterable[Emission] = (),
        *,
        knowledge: KnowledgeBase | None = None,
        budget: SaturationBudget | None = None,
        milestone_kind: str | None = None,
    ) -> SaturationReport:
        kb = knowledge if knowledge is not None else KnowledgeBase()
        limits = budget if budget is not None else SaturationBudget()
        started = time.monotonic()
        states = {item.name: _PassState() for item in self.passes}
        metrics = {item.name: PassMetrics(item.name) for item in self.passes}
        runs: Counter[str] = Counter()
        first_result_s: float | None = None

        def note_milestone() -> None:
            nonlocal first_result_s
            if (
                first_result_s is None
                and milestone_kind is not None
                and kb.evidence(milestone_kind)
            ):
                first_result_s = time.monotonic() - started

        for item in initial:
            kb.publish(
                item.kind,
                item.key,
                item.value,
                producer="initial",
                parents=item.parents,
            )
        note_milestone()

        stop_reason = "saturated"
        while True:
            if time.monotonic() - started >= limits.max_seconds:
                stop_reason = "time-budget"
                break
            if len(kb.evidence()) >= limits.max_evidence:
                stop_reason = "evidence-budget"
                break
            if sum(runs.values()) >= limits.max_pass_runs:
                stop_reason = "pass-run-budget"
                break

            ran_any = False
            learned_any = False
            for strategy in self.scheduler.order(self.passes, metrics):
                state = states[strategy.name]
                cursor, delta = kb.evidence_since(
                    state.cursor,
                    kinds=strategy.consumes,
                )
                state.cursor = cursor
                if strategy.consumes:
                    if not delta:
                        continue
                elif state.activated:
                    continue
                state.activated = True
                ran_any = True
                measured = metrics[strategy.name]
                metrics[strategy.name] = replace(
                    measured, activations=measured.activations + 1
                )

                no_novelty = 0
                for _ in range(strategy.repeat.trials_per_activation):
                    if sum(runs.values()) >= limits.max_pass_runs:
                        stop_reason = "pass-run-budget"
                        break
                    measured = metrics[strategy.name]
                    if (
                        limits.max_pass_seconds is not None
                        and measured.elapsed_s >= limits.max_pass_seconds
                    ):
                        # One slow backend must not spend the whole portfolio's
                        # budget. The pass keeps its evidence; it just stops
                        # being offered further trials.
                        break
                    runs[strategy.name] += 1
                    before = len(kb.evidence())
                    novel = 0
                    published = 0
                    failed = 0
                    trial_started = time.monotonic()
                    try:
                        emissions = tuple(strategy.run(kb, delta))
                    except Exception as error:
                        failed = 1
                        digest = hashlib.sha256(
                            f"{type(error).__module__}.{type(error).__qualname__}:{error}".encode()
                        ).hexdigest()[:16]
                        emissions = (
                            Emission(
                                "pass-failure",
                                f"{strategy.name}:{digest}",
                                {
                                    "pass": strategy.name,
                                    "type": f"{type(error).__module__}.{type(error).__qualname__}",
                                    "detail": str(error)[:500],
                                },
                                tuple(item.identity for item in delta),
                            ),
                        )
                    for emission in emissions:
                        if len(kb.evidence()) >= limits.max_evidence:
                            stop_reason = "evidence-budget"
                            break
                        published += 1
                        added = kb.publish(
                            emission.kind,
                            emission.key,
                            emission.value,
                            producer=strategy.name,
                            parents=emission.parents,
                        )
                        novel += int(added)
                        learned_any |= added
                    trial_elapsed = time.monotonic() - trial_started
                    measured = metrics[strategy.name]
                    metrics[strategy.name] = replace(
                        measured,
                        runs=measured.runs + 1,
                        elapsed_s=measured.elapsed_s + trial_elapsed,
                        emissions=measured.emissions + published,
                        novel=measured.novel + novel,
                        failures=measured.failures + failed,
                    )
                    note_milestone()
                    if len(kb.evidence()) == before:
                        no_novelty += 1
                        if no_novelty >= strategy.repeat.no_novelty_limit:
                            break
                    else:
                        no_novelty = 0
                    # New output is deliberately not delivered as new input
                    # inside this activation. The outer semi-naive round makes
                    # self-consuming passes explicit and order-independent.
                if stop_reason != "saturated":
                    break

            if stop_reason != "saturated":
                break
            if not ran_any:
                break
            if not learned_any:
                # Every relevant delta was consumed and no canonical item was
                # added. Advancing the cursors makes the next scan quiescent.
                continue

        return SaturationReport(
            stop_reason=stop_reason,
            pass_runs=dict(runs),
            evidence_count=len(kb.evidence()),
            duration_s=time.monotonic() - started,
            failures=kb.evidence("pass-failure"),
            metrics=metrics,
            first_result_s=first_result_s,
        )
