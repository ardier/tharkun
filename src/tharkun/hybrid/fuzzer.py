"""HybridFuzzer: run Atheris in the background, unblock its frontier with tharkun.

Ties the pieces together into the loop the user asked for:

  1. run Atheris in the background over the corpus (:mod:`backend`);
  2. watch its coverage; when it plateaus (random mutation is stuck on a hard guard),
  3. call the assist engine to solve the next high-value frontiers, and
  4. write the unblocking seeds into the corpus dir, where Atheris reloads them and
     keeps going past the guard.

The plateau decision and the loop are separated from wall-clock time (``clock``/``sleep``
are injectable) so the orchestration is deterministically testable with a
:class:`~tharkun.hybrid.backend.FakeBackend`, while the default is real time against a
real :class:`~tharkun.hybrid.backend.AtherisBackend`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

from tharkun.hybrid.assist import AssistEngine, HarnessSpec


@dataclass
class HybridStats:
    assists: int = 0                 # assist rounds fired
    seeds_written: int = 0           # unblocking seeds injected
    solved: list = field(default_factory=list)
    final_coverage: int = 0
    plateaus: int = 0
    elapsed_s: float = 0.0     # actual wall clock, so a benchmark can check its own parity
    reconciles: int = 0        # times the background corpus grew and facts were updated
    blockers: object = None    # the blocker-worklist summary (LedgerSummary) at run end


def should_assist(cov_now: int, best_cov: int, secs_since_gain: float, plateau_s: float) -> bool:
    """Assist when coverage has not advanced for at least *plateau_s* seconds."""
    return cov_now <= best_cov and secs_since_gain >= plateau_s


class HybridFuzzer:
    """Orchestrate an Atheris backend + the tharkun assist loop."""

    def __init__(
        self,
        spec: HarnessSpec,
        *,
        backend=None,
        plateau_s: float = 5.0,
        assist_cooldown_s: float = 3.0,
        assist_reserve_s: float = 4.0,   # time an assist round is assumed to need
        max_frontiers: int = 8,
    ) -> None:
        self.spec = spec
        self.assist = AssistEngine(spec, max_frontiers=max_frontiers)
        if backend is None:
            from tharkun.hybrid.backend import AtherisBackend
            backend = AtherisBackend(spec.harness_path, spec.corpus_dir)
        self.backend = backend
        self.plateau_s = plateau_s
        self.assist_cooldown_s = assist_cooldown_s
        self.assist_reserve_s = assist_reserve_s

    def run(
        self,
        duration_s: float,
        *,
        poll_s: float = 0.5,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> HybridStats:
        """Fuzz for *duration_s*, assisting whenever coverage plateaus.

        Returns :class:`HybridStats`.  ``clock``/``sleep`` are injectable so tests drive
        the loop without real waiting.
        """
        stats = HybridStats()
        self.backend.start()
        start = clock()
        best_cov = self.backend.coverage()
        last_gain = start
        last_assist = -1e9

        def _corpus_count() -> int:
            try:
                return len(os.listdir(self.spec.corpus_dir))
            except OSError:
                return 0

        last_corpus_n = _corpus_count()

        deadline = start + duration_s
        while clock() < deadline and self.backend.is_running():
            sleep(poll_s)
            now = clock()
            cov = self.backend.coverage()
            # Atheris runs in the background; when it grows the seed queue, fold the new
            # seeds into the facts right away (cheap dir-count gate first, so the common
            # no-growth poll costs a listdir, not a full corpus re-read). A dir-count rise
            # always changes the corpus fingerprint, so observe_new_corpus always bumps the
            # epoch here — count the reconcile on growth, not on its (package-mode-zero)
            # observed-seed return.
            n = _corpus_count()
            if n > last_corpus_n:
                last_corpus_n = n
                self.assist.observe_new_corpus()
                stats.reconciles += 1
            if cov > best_cov:
                best_cov = cov
                last_gain = now
                continue
            cooled = (now - last_assist) >= self.assist_cooldown_s
            if cooled and should_assist(cov, best_cov, now - last_gain, self.plateau_s):
                # Do not START a round that cannot finish inside the budget. An assist
                # can run for seconds (it solves and re-traces), and the old loop only
                # checked the clock at the top — so a run could overshoot duration_s
                # badly, which silently makes any timed comparison unfair.
                #
                # But when it is too late to assist, keep FUZZING to the deadline rather
                # than breaking out — the background Atheris should use every second of the
                # shared budget. Breaking here made the hybrid arm quit ~assist_reserve_s
                # early, so it ran LESS wall-clock than plain (a 3.8s skew, measured), which
                # is its own fairness bug even though it disfavours the hybrid.
                if clock() + self.assist_reserve_s > deadline:
                    continue
                stats.plateaus += 1
                res = self.assist.assist_once()
                stats.assists += 1
                stats.seeds_written += len(res.seeds_written)
                stats.solved.extend(res.solved)
                last_assist = now
                last_gain = clock()   # let Atheris reload and explore the new seeds
                if clock() >= deadline:
                    break

        self.backend.stop()
        stats.final_coverage = self.backend.coverage()
        stats.elapsed_s = clock() - start      # report it; overshoot must be visible
        stats.blockers = self.assist.ledger.summary()   # the worklist, at run end
        return stats

    def assist_step(self) -> list[str]:
        """Run one assist round out of band (no backend), returning new seed paths.

        Useful for a manual/offline hybrid loop, or to pre-seed a corpus before fuzzing.
        """
        return self.assist.assist_once().seeds_written
