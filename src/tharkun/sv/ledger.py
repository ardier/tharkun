"""The ONE CPU-second ledger of a ``verify`` call ([internal ref] step 3a, docs/SV.md "Order and the
CPU-second ledger").

Every stage of ``sv/scheduler.py::run`` spends from, and is bounded by, a single object
instead of a ``nonlocal`` pile each call site had to know by name:

* **spent** -- the CPU seconds of this process and every child it has reaped
  (``engine._cpu_seconds``: ``os.times()``), plus the live fork servers (:meth:`add_server`),
  whose children have not been reaped yet and are invisible to ``os.times()``;
* **remaining** -- ``budget_seconds`` minus what has been spent, and (unless ``cpu_time_only``,
  the [internal ref] test seam) also minus the wall-clock time since the start, whichever is tighter;
* **cap** -- the CPU second at which searching stops (:meth:`capped`): a proof's canary or an
  early stop once every site solved to "no modelled violation".  A cap only ever moves earlier.
  :meth:`allowance` is what ONE run may still use (remaining and what the cap leaves);
* **goal** -- what the run is FOR (:class:`Goal`, [internal ref]): a confirmed violation (or, when a caller asks
  for several, that many).  :meth:`stopped` is the ONE stop condition every stage reads: the goal is
  met, the budget is spent, or a cap has been reached.  A stage that runs dry (a finite enumeration, a
  fuzzer that plateaued) never ends the run by itself; the goal being met or the budget being
  exhausted does, so no stage needs a special case for "what if there is budget left".

This is the ledger ``sv/scheduler.py::run`` already had as closures (``remaining``,
``cpu_seconds``, ``cap``, ``capped``); moving it into an object changes no number it computes.
Every clock read goes through ``tharkun.sv.engine`` (``engine.time.monotonic``,
``engine._cpu_seconds``) looked up at call time, so a test that patches those names reaches the
ledger exactly as it reached the closures.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from threading import Lock
from typing import Callable

from tharkun.sv import engine as _engine


@dataclass
class Goal:
    """What a ``verify`` call is for: ``wanted`` confirmed violations (a replayed run that raises a
    property class, never a solver's say-so).  The default is the competition's: ONE, then stop.

    ``record`` is called by the scheduler's ``found`` (the single place a violation is confirmed), so
    the goal can only be met by the same replay-confirmation every FALSE already goes through.
    ``wanted > 1`` is ruled, not built: ``scheduler.run`` returns on the first confirmed violation,
    and a caller that wants several (the mutation-kill wrapper) drives ``verify`` once per input.
    """

    wanted: int = 1
    found: int = 0

    def record(self, n: int = 1) -> None:
        self.found += n

    @property
    def met(self) -> bool:
        return self.found >= self.wanted


class Ledger:
    def __init__(self, budget_seconds: float, *, cpu_time_only: bool = False,
                 say: Callable[[str], None] = lambda msg: None, goal: Goal | None = None) -> None:
        self.budget_seconds = budget_seconds
        self.cpu_time_only = cpu_time_only
        self.say = say
        self.goal = goal if goal is not None else Goal()
        self.started = _engine.time.monotonic()
        self.cpu_started = _engine._cpu_seconds()
        self.cap: float | None = None            # CPU second at which searching stops
        self._servers: list = []                 # live fork servers (their CPU is not reaped yet)
        self._reserved = 0.0
        self._reservation_lock = Lock()

    # --- what has been spent ------------------------------------------------------------
    def add_server(self, server) -> None:
        """Count a live fork server's CPU from now on."""
        self._servers.append(server)

    @property
    def servers(self) -> list:
        return self._servers

    def cpu_seconds(self) -> float:
        """The absolute CPU clock: reaped children via ``os.times`` plus live fork servers."""
        return _engine._cpu_seconds() + sum(server.cpu for server in self._servers)

    def spent(self) -> float:
        """CPU seconds this ``verify`` call has used."""
        return self.cpu_seconds() - self.cpu_started

    def remaining(self) -> float:
        cpu_left = self.budget_seconds - self.spent()
        if self.cpu_time_only:
            return cpu_left
        return min(cpu_left, self.budget_seconds - (_engine.time.monotonic() - self.started))

    # --- what a cap takes away ----------------------------------------------------------
    def capped(self, seconds: float, why: str) -> None:
        """Stop searching ``seconds`` of CPU from now (at most a tenth of the budget), unless a
        sooner cap is already in force."""
        end = self.cpu_seconds() + min(seconds, self.budget_seconds / 10)
        if self.cap is None or end < self.cap:
            self.cap = end
            self.say(f"search capped at {min(seconds, self.budget_seconds / 10):.0f}s CPU: {why}")

    def at_cap(self) -> bool:
        return self.cap is not None and self.cpu_seconds() >= self.cap

    # --- when to stop ([internal ref]) -----------------------------------------------------------
    def stopped(self, reserve: float = 0.0) -> bool:
        """The one stop condition: the goal is met, no more than ``reserve`` CPU seconds remain, or a
        cap has been reached.  A stage that has nothing left to try is NOT a reason to stop."""
        return self.goal.met or self.remaining() <= reserve or self.at_cap()

    def stop_reason(self, reserve: float = 0.0) -> str:
        """Why :meth:`stopped` holds (for the verdict's reason); ``""`` while it does not."""
        if self.goal.met:
            return "goal met"
        if self.remaining() <= reserve:
            return "budget exhausted"
        return "search capped" if self.at_cap() else ""

    def allowance(self) -> float:
        """What one run may still use: the remaining budget, and what the cap leaves of it."""
        if self.cap is None:
            return self.remaining()
        return min(self.remaining(), self.cap - self.cpu_seconds())

    def share(self, fraction: float) -> float:
        """``fraction`` of what remains, never more than the cap leaves."""
        budget = self.remaining() * fraction
        if self.cap is not None:
            budget = min(budget, max(0.0, self.cap - self.cpu_seconds()))
        return budget

    @contextmanager
    def reserve(self, seconds: float, *, canary: bool = False):
        """Hold a run's whole allowance, including startup, until its children finish."""
        with self._reservation_lock:
            # The scheduler admits a canary batch before its cap; every slot may finish.
            left = self.remaining() if canary else self.allowance()
            if canary:
                seconds = min(seconds, max(1.0, self.allowance() - self._reserved))
            granted = max(0.0, min(seconds, left - self._reserved))
            self._reserved += granted
        try:
            yield granted
        finally:
            with self._reservation_lock:
                self._reserved = max(0.0, self._reserved - granted)
