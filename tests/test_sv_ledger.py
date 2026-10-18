"""The ONE CPU-second ledger ([internal ref] step 3a, ``sv/ledger.py``).

It is the ledger ``sv/scheduler.py::run`` already had as closures; these tests pin the numbers it
computes (so moving it into an object changed none of them) with a fake clock patched in through
``tharkun.sv.engine`` -- the names the ledger looks up at call time, like the closures did.
"""
from __future__ import annotations

import pytest

from tharkun.sv import engine
from tharkun.sv.ledger import Ledger


class Clock:
    def __init__(self):
        self.cpu = 100.0          # absolute CPU clock (os.times) -- the ledger starts from here
        self.wall = 5000.0


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(engine, "_cpu_seconds", lambda: c.cpu)
    monkeypatch.setattr(engine.time, "monotonic", lambda: c.wall)
    return c


class Server:
    cpu = 0.0


def test_spent_and_remaining_count_from_the_ledgers_own_start(clock):
    ledger = Ledger(60.0)
    assert ledger.spent() == 0.0 and ledger.remaining() == 60.0
    clock.cpu += 10
    assert ledger.spent() == 10.0 and ledger.remaining() == 50.0


def test_remaining_is_the_tighter_of_cpu_and_wall_clock(clock):
    ledger = Ledger(60.0)
    clock.cpu += 10
    clock.wall += 45                           # a loaded host: little CPU, much waiting
    assert ledger.remaining() == 15.0
    assert Ledger(60.0, cpu_time_only=True).remaining() == 60.0     # the [internal ref] test seam ignores wall time


def test_a_live_fork_servers_cpu_is_counted_before_it_is_reaped(clock):
    ledger = Ledger(60.0)
    server = Server()
    ledger.add_server(server)
    server.cpu = 7.0
    assert ledger.spent() == 7.0 and ledger.remaining() == 53.0
    assert ledger.servers == [server]


def test_a_cap_stops_the_search_that_many_cpu_seconds_from_now(clock):
    lines = []
    ledger = Ledger(100.0, say=lines.append)
    clock.cpu += 20
    ledger.capped(5.0, "canary")
    assert ledger.cap == 125.0 and not ledger.at_cap()
    assert lines == ["search capped at 5s CPU: canary"]
    clock.cpu += 5
    assert ledger.at_cap() and ledger.allowance() == 0.0


def test_a_cap_is_at_most_a_tenth_of_the_budget_and_only_ever_moves_earlier(clock):
    ledger = Ledger(100.0)
    ledger.capped(60.0, "canary")                       # asked for 60, a tenth of 100 is 10
    assert ledger.cap == clock.cpu + 10.0
    ledger.capped(30.0, "later, longer")                # a later, longer cap changes nothing
    assert ledger.cap == clock.cpu + 10.0
    ledger.capped(2.0, "earlier")
    assert ledger.cap == clock.cpu + 2.0


def test_allowance_is_what_one_run_may_use_the_remaining_budget_and_what_a_cap_leaves(clock):
    ledger = Ledger(100.0)
    assert ledger.allowance() == 100.0                  # no cap: the whole remaining budget
    ledger.capped(10.0, "x")
    clock.cpu += 4
    assert ledger.allowance() == 6.0                    # the cap leaves 6 of the 96 remaining


def test_a_portfolio_stages_share_is_a_fraction_of_remaining_never_more_than_the_cap_leaves(clock):
    ledger = Ledger(100.0)
    assert ledger.share(0.15) == 15.0
    ledger.capped(10.0, "x")
    assert ledger.share(0.15) == 10.0                   # 15 would overrun the cap's 10
    clock.cpu += 9.5
    assert ledger.share(0.15) == 0.5                    # no minimum may exceed the remaining cap


def test_run_reservations_hold_the_aggregate_allowance_until_release(clock):
    ledger = Ledger(1.0)
    with ledger.reserve(0.8) as first:
        assert first == 0.8
        with ledger.reserve(0.8) as second:
            assert second == pytest.approx(0.2)
            with ledger.reserve(1.0) as exhausted:
                assert exhausted == 0
    with ledger.reserve(1.0) as recovered:
        assert recovered == 1.0


def test_run_reservations_respect_cpu_wall_and_cap_and_release_on_error(clock):
    ledger = Ledger(10.0)
    ledger.capped(1.0, "canary")
    clock.cpu += 0.9
    with pytest.raises(ValueError):
        with ledger.reserve(5.0) as granted:
            assert granted == pytest.approx(0.1)
            raise ValueError("failed startup")
    clock.wall += 9.95
    with ledger.reserve(5.0) as granted:
        assert granted == pytest.approx(0.05)


def test_an_admitted_canary_slot_can_reserve_after_its_batch_crosses_the_cap(clock):
    ledger = Ledger(10.0)
    ledger.capped(0.1, "canary")
    clock.cpu += 0.2
    assert ledger.stopped()
    with ledger.reserve(1.0) as ordinary:
        assert ordinary == 0
    with ledger.reserve(5.0, canary=True) as admitted:
        assert admitted == 1.0


@pytest.mark.parametrize("limiting_clock", ["cpu", "wall"])
def test_canary_reservations_cannot_exceed_the_remaining_task_budget(clock, limiting_clock):
    ledger = Ledger(10.0)
    setattr(clock, limiting_clock, getattr(clock, limiting_clock) + 9.75)
    ledger.capped(1.0, "canary")
    with ledger.reserve(1.0, canary=True) as first:
        assert first == 0.25
        with ledger.reserve(1.0, canary=True) as exhausted:
            assert exhausted == 0
    setattr(clock, limiting_clock, getattr(clock, limiting_clock) + 0.25)
    with ledger.reserve(1.0, canary=True) as exhausted:
        assert exhausted == 0


@pytest.mark.parametrize("canary", [False, True])
def test_parallel_run_reservations_cannot_overbook_a_subsecond_budget(clock, canary):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    ledger = Ledger(0.2)
    if canary:
        ledger.capped(0.01, "canary")
    held, release = Barrier(5), Barrier(5)
    grants = []
    def run():
        with ledger.reserve(1.0 if canary else 0.05, canary=canary) as granted:
            grants.append(granted)
            held.wait(timeout=5)
            release.wait(timeout=5)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run) for _ in range(4)]
        held.wait(timeout=5)
        assert len(grants) == 4 and sum(grants) <= 0.2
        assert sorted(grants) == pytest.approx([0, 0, 0, 0.2] if canary else [0.05] * 4)
        release.wait(timeout=5)
        for future in futures:
            future.result()
