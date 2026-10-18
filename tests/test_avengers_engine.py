from __future__ import annotations

import time

import pytest

from tharkun.avengers.engine import (
    Avengers,
    Emission,
    FunctionPass,
    PassMetrics,
    PortfolioOrder,
    RepeatPolicy,
    SaturationBudget,
    YieldOrder,
)
from tharkun.knowledge import KnowledgeBase


def _pipeline(order):
    def template(_kb, delta):
        return [Emission("template", item.key, "T", (item.identity,)) for item in delta]

    def candidates(_kb, delta):
        return [Emission("candidate", item.key, "C", (item.identity,)) for item in delta]

    def replay(_kb, delta):
        return [Emission("validated", item.key, "V", (item.identity,)) for item in delta]

    passes = {
        "template": FunctionPass("template", frozenset({"goal"}), template),
        "candidate": FunctionPass("candidate", frozenset({"template"}), candidates),
        "replay": FunctionPass("replay", frozenset({"candidate"}), replay),
    }
    kb = KnowledgeBase()
    report = Avengers(passes[name] for name in order).run(
        [Emission("goal", "g", "G")], knowledge=kb
    )
    return report, kb


def test_fixed_point_is_independent_of_pass_registration_order():
    forward, forward_kb = _pipeline(("template", "candidate", "replay"))
    reverse, reverse_kb = _pipeline(("replay", "candidate", "template"))
    assert forward.saturated and reverse.saturated
    assert {(x.kind, x.key) for x in forward_kb.evidence()} == {
        (x.kind, x.key) for x in reverse_kb.evidence()
    }
    assert forward_kb.evidence_item("validated", "g") is not None


def test_duplicate_candidate_keeps_derivations_and_wakes_replay_once():
    replayed = []

    def one(_kb, delta):
        return [Emission("candidate", "same", {"x": 1}, (delta[0].identity,))]

    def two(_kb, delta):
        return [Emission("candidate", "same", {"x": 1}, (delta[0].identity,))]

    def replay(_kb, delta):
        replayed.extend(item.key for item in delta)
        return [Emission("validated", item.key, True, (item.identity,)) for item in delta]

    kb = KnowledgeBase()
    Avengers(
        (
            FunctionPass("hypothesis", frozenset({"goal"}), one),
            FunctionPass("pynguin", frozenset({"goal"}), two),
            FunctionPass("replay", frozenset({"candidate"}), replay),
        )
    ).run([Emission("goal", "g", True)], knowledge=kb)
    candidate = kb.evidence_item("candidate", "same")
    assert [d.producer for d in candidate.derivations] == ["hypothesis", "pynguin"]
    assert replayed == ["same"]


def test_pass_failure_is_evidence_and_does_not_stop_peers():
    def broken(_kb, _delta):
        raise RuntimeError("boom")

    def healthy(_kb, delta):
        return [Emission("answer", item.key, 42) for item in delta]

    kb = KnowledgeBase()
    report = Avengers(
        (
            FunctionPass("broken", frozenset({"goal"}), broken),
            FunctionPass("healthy", frozenset({"goal"}), healthy),
        )
    ).run([Emission("goal", "g", True)], knowledge=kb)
    assert report.saturated
    assert kb.evidence_item("answer", "g").value == 42
    assert len(report.failures) == 1


def test_nondeterministic_pass_may_receive_several_trials_per_activation():
    attempts = []

    def stochastic(_kb, _delta):
        attempts.append(len(attempts))
        if len(attempts) == 2:
            return [Emission("answer", "lucky", True)]
        return []

    report = Avengers(
        (
            FunctionPass(
                "pynguin",
                frozenset({"goal"}),
                stochastic,
                RepeatPolicy.exploratory(trials=4, no_novelty=2),
            ),
        )
    ).run([Emission("goal", "g", True)])
    assert report.saturated
    assert len(attempts) >= 2


def test_pass_run_budget_is_reported_separately_from_saturation():
    noop = FunctionPass(
        "search",
        frozenset({"goal"}),
        lambda _kb, _delta: (),
        RepeatPolicy.exploratory(trials=10, no_novelty=10),
    )
    report = Avengers((noop,)).run(
        [Emission("goal", "g", True)],
        budget=SaturationBudget(max_pass_runs=2),
    )
    assert report.stop_reason == "pass-run-budget"



def test_metrics_count_runs_novelty_and_failures():
    def emit(_kb, delta):
        return [Emission("template", item.key, "T", (item.identity,)) for item in delta]

    def broken(_kb, _delta):
        raise RuntimeError("strategy is unavailable")

    kb = KnowledgeBase()
    report = Avengers(
        [
            FunctionPass("emit", frozenset({"goal"}), emit),
            FunctionPass("broken", frozenset({"goal"}), broken),
        ]
    ).run([Emission("goal", "g", "G")], knowledge=kb)

    assert report.metrics["emit"].runs == 1
    assert report.metrics["emit"].emissions == 1
    assert report.metrics["emit"].novel == 1
    assert report.metrics["emit"].failures == 0
    assert report.metrics["emit"].activations == 1
    # A raising pass still emits one pass-failure record, which is novel
    # evidence: the portfolio learns that the strategy is unusable here.
    assert report.metrics["broken"].failures == 1
    assert report.metrics["broken"].novel == 1


def test_a_repeat_that_learns_nothing_is_not_counted_as_novel():
    def constant(_kb, delta):
        return [Emission("template", "same", "T", (delta[0].identity,))]

    kb = KnowledgeBase()
    report = Avengers(
        [
            FunctionPass(
                "constant",
                frozenset({"goal"}),
                constant,
                RepeatPolicy.exploratory(trials=4, no_novelty=4),
            )
        ]
    ).run([Emission("goal", "g", "G")], knowledge=kb)

    assert report.metrics["constant"].runs == 4
    assert report.metrics["constant"].emissions == 4
    assert report.metrics["constant"].novel == 1


def test_first_result_time_is_recorded_for_the_requested_milestone():
    def validate(_kb, delta):
        return [Emission("validated", item.key, "V", (item.identity,)) for item in delta]

    report = Avengers([FunctionPass("validate", frozenset({"goal"}), validate)]).run(
        [Emission("goal", "g", "G")], milestone_kind="validated"
    )
    assert report.first_result_s is not None
    assert 0.0 <= report.first_result_s <= report.duration_s

    silent = Avengers([FunctionPass("validate", frozenset({"goal"}), validate)]).run(
        [Emission("goal", "g", "G")], milestone_kind="never-emitted"
    )
    assert silent.first_result_s is None


def test_a_milestone_present_in_the_initial_evidence_is_timed_at_zero():
    report = Avengers([]).run(
        [Emission("validated", "seed", "V")], milestone_kind="validated"
    )
    assert report.first_result_s is not None
    assert report.first_result_s < 1.0


def test_max_pass_seconds_stops_offering_trials_to_one_slow_strategy():
    calls = {"n": 0}

    def slow(_kb, delta):
        calls["n"] += 1
        time.sleep(0.02)
        return [Emission("template", f"t{calls['n']}", "T", (delta[0].identity,))]

    report = Avengers(
        [
            FunctionPass(
                "slow",
                frozenset({"goal"}),
                slow,
                RepeatPolicy.exploratory(trials=50, no_novelty=50),
            )
        ]
    ).run(
        [Emission("goal", "g", "G")],
        budget=SaturationBudget(max_seconds=10.0, max_pass_seconds=0.05),
    )

    # The ceiling is checked before a trial, so the pass may overshoot by at
    # most one trial; what matters is that it does not run all fifty.
    assert 1 <= report.metrics["slow"].runs < 50
    assert report.metrics["slow"].elapsed_s >= 0.02


def test_max_pass_seconds_must_be_positive():
    with pytest.raises(ValueError):
        SaturationBudget(max_pass_seconds=0.0)


def test_portfolio_order_is_the_default_and_keeps_the_declared_order():
    passes = (
        FunctionPass("a", frozenset({"goal"}), lambda _kb, _d: ()),
        FunctionPass("b", frozenset({"goal"}), lambda _kb, _d: ()),
    )
    engine = Avengers(passes)
    assert isinstance(engine.scheduler, PortfolioOrder)
    assert engine.scheduler.order(passes, {}) == passes


def test_yield_order_prefers_unmeasured_passes_then_novelty_per_second():
    passes = (
        FunctionPass("cheap", frozenset({"goal"}), lambda _kb, _d: ()),
        FunctionPass("costly", frozenset({"goal"}), lambda _kb, _d: ()),
        FunctionPass("fresh", frozenset({"goal"}), lambda _kb, _d: ()),
    )
    metrics = {
        "cheap": PassMetrics("cheap", runs=1, elapsed_s=0.01, novel=5),
        "costly": PassMetrics("costly", runs=1, elapsed_s=1.0, novel=1),
    }
    ordered = YieldOrder().order(passes, metrics)
    assert [item.name for item in ordered] == ["fresh", "cheap", "costly"]


def test_yield_order_reaches_the_same_fixed_point_as_the_portfolio_order():
    def template(_kb, delta):
        return [Emission("template", item.key, "T", (item.identity,)) for item in delta]

    def candidates(_kb, delta):
        return [Emission("candidate", item.key, "C", (item.identity,)) for item in delta]

    def build(scheduler):
        kb = KnowledgeBase()
        Avengers(
            [
                FunctionPass("template", frozenset({"goal"}), template),
                FunctionPass("candidate", frozenset({"template"}), candidates),
            ],
            scheduler=scheduler,
        ).run([Emission("goal", "g", "G")], knowledge=kb)
        return {(item.kind, item.key) for item in kb.evidence()}

    assert build(YieldOrder()) == build(PortfolioOrder())
