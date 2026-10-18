"""High-level object synthesis API built from the Avengers pass portfolio."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from tharkun.avengers.engine import (
    Avengers,
    Emission,
    SaturationBudget,
    Scheduler,
)
from tharkun.avengers.executor import IsolatedExecutor
from tharkun.avengers.model import (
    AvengersReport,
    ConstructionRecipe,
    MethodCall,
    MethodCoverage,
    ObjectGoal,
    ReplayOutcome,
    SourceArtifact,
    ValidatedRecipe,
)
from tharkun.avengers.passes import (
    GOAL,
    REPLAY,
    SOURCE_ARTIFACT,
    VALIDATED,
    BaselineCandidatePass,
    ConstructorTemplatePass,
    FlattingPass,
    HypothesisCandidatePass,
    NestedCandidatePass,
    GeneratedSourcePass,
    ReplayPass,
    SetupSequencePass,
    public_method_calls,
)
from tharkun.knowledge import KnowledgeBase

if TYPE_CHECKING:  # pragma: no cover - import cycle guard
    from tharkun.avengers.store import TemplateStore


def default_passes(
    *,
    executor: IsolatedExecutor | None = None,
    max_setup_steps: int = 2,
    hypothesis_examples: int = 100,
    hypothesis_backends: tuple[str, ...] = ("hypothesis",),
    store: "TemplateStore | None" = None,
    compose_depth: int = 1,
    depth: int = 0,
    objectives: bool = False,
    backends: Iterable[str] = (),
    interpreter: str | None = None,
    backend_seconds: int = 20,
) -> tuple[object, ...]:
    """The initial object-construction portfolio, cheap passes first.

    A store, when given, seeds the portfolio first: replaying a recipe that
    already worked is the cheapest thing in it.  ``compose_depth`` bounds how
    far the portfolio will recurse to build an object another constructor
    needs; ``depth`` is how far down that recursion this portfolio already is.
    """

    isolated = executor if executor is not None else IsolatedExecutor()
    portfolio: list[object] = []
    if store is not None:
        from tharkun.avengers.store import StoreSeedPass

        portfolio.append(StoreSeedPass(store))
    portfolio += [
        ConstructorTemplatePass(),
        BaselineCandidatePass(),
        *(
            HypothesisCandidatePass(
                isolated,
                max_examples=hypothesis_examples,
                backend=backend,
            )
            for backend in hypothesis_backends
        ),
        GeneratedSourcePass(),
        NestedCandidatePass(
            synthesize=_nested_synthesizer(
                executor=isolated,
                store=store,
                max_setup_steps=max_setup_steps,
                hypothesis_examples=hypothesis_examples,
                hypothesis_backends=hypothesis_backends,
                compose_depth=compose_depth,
            ),
            store=store,
            max_depth=compose_depth,
            depth=depth,
        ),
        SetupSequencePass(max_steps=max_setup_steps),
        ReplayPass(isolated),
    ]
    if backends:
        from tharkun.avengers.backends import backend_passes

        # AFTER the cheap passes, deliberately. An external generator costs
        # seconds to tens of seconds before it says anything, and putting one
        # first made time-to-first-object an order of magnitude worse than the
        # built-in portfolio alone -- measured, on the benchmark below. The
        # replay pass is already in the list, so whatever a backend proposes is
        # picked up on the next semi-naive round.
        #
        # A backend that cannot run here is skipped rather than failing the
        # portfolio; backend_status() is where a caller finds out which.
        portfolio.extend(
            backend_passes(
                backends,
                executor=isolated,
                interpreter=interpreter,
                search_seconds=backend_seconds,
                max_examples=hypothesis_examples,
            )
        )
    if objectives:
        from tharkun.avengers.objectives import (
            CoverageSetupPass,
            NegativeSequencePass,
            StateDiversityPass,
        )

        portfolio += [
            StateDiversityPass(isolated, max_steps=max_setup_steps + 1),
            CoverageSetupPass(isolated),
            NegativeSequencePass(isolated),
        ]
    from tharkun.avengers.flatting import flatting_available

    if flatting_available():
        portfolio.append(FlattingPass(isolated))
    return tuple(portfolio)


def _nested_synthesizer(
    *,
    executor: IsolatedExecutor,
    store: "TemplateStore | None",
    max_setup_steps: int,
    hypothesis_examples: int,
    hypothesis_backends: tuple[str, ...],
    compose_depth: int,
    budget_s: float = 10.0,
):
    """Search for a nested class the way the outer goal is searched.

    The sub-search gets its own small budget and no boundary calls: all the
    caller needs is an object that constructs, and asking for more would spend
    the outer goal's time on a question nobody asked.
    """

    def search(source_path: str, class_name: str, depth: int):
        nested_goal = ObjectGoal(source_path, class_name)
        report = synthesize_object(
            nested_goal,
            executor=executor,
            store=store,
            max_setup_steps=max_setup_steps,
            hypothesis_examples=hypothesis_examples,
            hypothesis_backends=hypothesis_backends,
            budget=SaturationBudget(max_seconds=budget_s),
            max_method_probes=1,
            compose_depth=compose_depth,
            depth=depth,
        )
        return tuple(item.recipe for item in report.validated)

    return search


def synthesize_object(
    goal: ObjectGoal,
    *,
    passes: Iterable[object] | None = None,
    knowledge: KnowledgeBase | None = None,
    budget: SaturationBudget | None = None,
    executor: IsolatedExecutor | None = None,
    files: tuple[str, ...] | None = None,
    max_setup_steps: int = 2,
    hypothesis_examples: int = 100,
    hypothesis_backends: tuple[str, ...] = ("hypothesis",),
    source_artifacts: Iterable[SourceArtifact] = (),
    max_method_probes: int = 256,
    scheduler: Scheduler | None = None,
    store: "TemplateStore | None" = None,
    compose_depth: int = 1,
    depth: int = 0,
    objectives: bool = False,
    backends: Iterable[str] = (),
    interpreter: str | None = None,
    backend_seconds: int = 20,
) -> AvengersReport:
    """Find replay-validated construction programs for *goal*.

    The report retains every valid recipe; ``report.best`` chooses the
    shortest one. A missing witness is reported as unresolved, never as proof
    that the class or requested boundary state is impossible.

    When a ``store`` is given, recipes it already holds for this goal are
    offered to the portfolio as candidates and everything validated here is
    written back, so a second search starts where the first one stopped.

    ``objectives=True`` additionally chases state coverage, branch coverage and
    deliberately invalid call sequences, filling ``states``, ``negatives`` and
    ``coverage``.  It is off by default because each objective spends real
    child processes -- an order of magnitude more than construction alone --
    and the first question is usually just whether the class can be built.
    Time to first object is unaffected either way: the objectives consume
    validated recipes, so they start after the milestone is already recorded.

    ``backends`` names external generators to run -- any of ``"pynguin"``,
    ``"ghostwriter"``, ``"hypothesis-from-type"``.  Each is
    skipped when it is not installed for ``interpreter``; nothing any of them
    proposes is believed without the usual fresh-process replay.

    ``files`` confines the default executor to the declared target files. An
    explicit executor retains its own environment and tracing configuration.
    """

    if max_method_probes < 1:
        raise ValueError("max_method_probes must be positive")
    kb = knowledge if knowledge is not None else KnowledgeBase()
    # Coverage is an objective, so when the objectives run the default executor
    # traces. A caller that supplied its own executor keeps whatever it chose.
    isolated = executor if executor is not None else IsolatedExecutor(trace=objectives, files=files)
    portfolio = tuple(passes) if passes is not None else default_passes(
        executor=isolated,
        max_setup_steps=max_setup_steps,
        hypothesis_examples=hypothesis_examples,
        hypothesis_backends=hypothesis_backends,
        store=store,
        compose_depth=compose_depth,
        depth=depth,
        objectives=objectives,
        backends=backends,
        interpreter=interpreter,
        backend_seconds=backend_seconds,
    )
    initial = [Emission(GOAL, goal.key, goal)]
    initial.extend(
        Emission(SOURCE_ARTIFACT, artifact.key, artifact)
        for artifact in source_artifacts
    )
    saturation = Avengers(portfolio, scheduler=scheduler).run(
        initial,
        knowledge=kb,
        budget=budget,
        milestone_kind=VALIDATED,
    )
    validated = tuple(
        item.value for item in kb.evidence(VALIDATED) if isinstance(item.value, ValidatedRecipe)
    )
    failures = tuple(
        item.value
        for item in kb.evidence(REPLAY)
        # A directed goal is SATISFIED by an outcome that did not return, so
        # "not succeeded" is the wrong test: it would file this search's own
        # answers under its failures.
        if isinstance(item.value, ReplayOutcome) and not goal.satisfied(item.value)
    )
    method_coverage: list[MethodCoverage] = []
    method_universe: tuple[str, ...] = ()
    coverage_complete = True
    if not goal.calls:
        discovered = public_method_calls(goal)
        method_universe = tuple(call.name for call in discovered)
        methods = discovered[:max_method_probes]
        candidate_limit = max(1, max_method_probes // max(1, len(methods)))
        candidates = sorted(
            validated,
            key=lambda item: (
                len(item.recipe.setup),
                len(item.recipe.constructor.args) + len(item.recipe.constructor.kwargs),
                item.recipe.key,
            ),
        )[:candidate_limit]
        coverage_complete = (
            len(methods) == len(discovered) and len(candidates) == len(validated)
        )
        for item in candidates:
            outcomes: list[tuple[MethodCall, ReplayOutcome]] = []
            for call in methods:
                boundary_goal = ObjectGoal(goal.source_path, goal.class_name, (call,))
                recipe = ConstructionRecipe(
                    boundary_goal.key,
                    item.recipe.constructor,
                    item.recipe.setup,
                )
                outcomes.append((call, isolated.replay(boundary_goal, recipe)))
            method_coverage.append(MethodCoverage(item, tuple(outcomes)))
    from tharkun.avengers.objectives import (
        COVERAGE,
        NEGATIVE,
        STATE,
        CoverageObservation,
        observe_coverage,
    )

    states = tuple(item.value for item in kb.evidence(STATE))
    negatives = tuple(item.value for item in kb.evidence(NEGATIVE))
    observations = [
        item.value
        for item in kb.evidence(COVERAGE)
        if isinstance(item.value, CoverageObservation) and item.value.goal.key == goal.key
    ]
    coverage = max(observations, key=lambda item: item.reach) if observations else None
    if method_coverage:
        # The per-method probe matrix executed real code too. Leaving it out
        # would report a class as less covered than the search actually made it.
        probed = [
            ValidatedRecipe(item.validated.goal, item.validated.recipe, outcome)
            for item in method_coverage
            for _call, outcome in item.outcomes
        ]
        merged = observe_coverage(goal, [*validated, *probed])
        if merged is not None:
            coverage = merged

    if store is not None:
        store.save(goal, kb)
    return AvengersReport(
        validated=validated,
        stop_reason=saturation.stop_reason,
        pass_runs=saturation.pass_runs,
        evidence_count=saturation.evidence_count,
        unresolved=() if validated else (goal,),
        failures=failures,
        method_coverage=tuple(method_coverage),
        method_universe=method_universe,
        coverage_complete=coverage_complete,
        time_to_first_object_s=saturation.first_result_s,
        metrics=saturation.metrics,
        states=states,
        negatives=negatives,
        coverage=coverage,
    )
