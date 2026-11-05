"""Benchmark: plain Atheris vs the hybrid loop, on the same target and budget.

The claim the hybrid fuzzer makes is that tharkun unblocks guards random mutation
cannot reach.  This measures it: run the *same* harness for the *same* wall-clock
budget, once as plain Atheris and once with the assist loop, and report the coverage
each reaches — plus a target-specific "depth" probe (how far into a guard chain the
corpus got), which is the honest, human-readable signal.

    from tharkun.hybrid.bench import benchmark, format_report
    r = benchmark(spec, duration_s=20)
    print(format_report(r))

``benchmark`` is deterministic in structure (not in exact counts — a fuzzer is
stochastic); run it a few times if you want a distribution.  Atheris required.
"""

from __future__ import annotations

import glob
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Callable

from tharkun.hybrid.assist import HarnessSpec
from tharkun.hybrid.backend import AtherisBackend
from tharkun.hybrid.fuzzer import HybridFuzzer


@dataclass
class Arm:
    """One benchmark arm (plain or hybrid)."""

    name: str
    coverage: int = 0            # libFuzzer cov: counter reached
    corpus_size: int = 0
    depth: int = 0               # target-specific depth probe (0 if none supplied)
    assists: int = 0
    seeds_injected: int = 0
    duration_s: float = 0.0


@dataclass
class BenchResult:
    target: str
    plain: Arm
    hybrid: Arm
    notes: list[str] = field(default_factory=list)

    @property
    def coverage_gain(self) -> int:
        return self.hybrid.coverage - self.plain.coverage

    @property
    def depth_gain(self) -> int:
        return self.hybrid.depth - self.plain.depth

    @property
    def atheris_ran(self) -> bool:
        """True when Atheris genuinely executed in BOTH arms — a nonzero libFuzzer ``cov:``
        counter is impossible without a real instrumented run, so this is the proof the
        comparison is against a live fuzzer, not a stub."""
        return self.plain.coverage > 0 and self.hybrid.coverage > 0

    @property
    def duration_skew(self) -> float:
        """How far the two arms' wall-clock diverged. Both cap Atheris at the same budget, so
        a large skew would mean one arm got more fuzzing time — the fairness signal to watch."""
        return abs(self.hybrid.duration_s - self.plain.duration_s)


def _seed_corpus(corpus_dir: str, seeds: list[bytes]) -> None:
    os.makedirs(corpus_dir, exist_ok=True)
    for i, s in enumerate(seeds):
        with open(os.path.join(corpus_dir, f"seed{i}"), "wb") as f:
            f.write(s)


def _corpus_depth(corpus_dir: str, depth_probe: Callable[[bytes], int] | None) -> int:
    if depth_probe is None:
        return 0
    best = 0
    for p in glob.glob(os.path.join(corpus_dir, "*")):
        try:
            best = max(best, depth_probe(open(p, "rb").read()))
        except Exception:
            pass
    return best


# Both arms give Atheris the SAME self-limit, so neither subprocess can out-fuzz the other.
# The hybrid's advantage must come from the solver it runs CONCURRENTLY, never from more
# raw Atheris time — so the `-max_total_time` cap is identical and the parity is asserted
# below. +1 is a small grace so the subprocess does not self-terminate a beat before the
# parent's own deadline poll.
def _atheris_budget(duration_s: float) -> int:
    return int(duration_s) + 1


def _run_plain(spec: HarnessSpec, corpus_dir: str, duration_s: float,
               depth_probe, python: str | None) -> Arm:
    backend = AtherisBackend(
        spec.harness_path, corpus_dir, python=python,
        extra_args=[f"-max_total_time={_atheris_budget(duration_s)}"],
    )
    t0 = time.monotonic()
    backend.start()
    # Let Atheris run the budget; poll coverage so we capture the peak.
    while time.monotonic() - t0 < duration_s and backend.is_running():
        time.sleep(0.5)
    # Read coverage AFTER stop (which drains the stderr reader), identically to the hybrid
    # arm — reading before stop let late-buffered cov: lines land in one arm but not the
    # other, a small asymmetry that favoured whichever arm read last.
    backend.stop()
    return Arm(
        name="plain-atheris", coverage=backend.coverage(),
        corpus_size=len(glob.glob(os.path.join(corpus_dir, "*"))),
        depth=_corpus_depth(corpus_dir, depth_probe),
        duration_s=time.monotonic() - t0,
    )


def _run_hybrid(spec: HarnessSpec, corpus_dir: str, duration_s: float,
                depth_probe, python: str | None, plateau_s: float) -> Arm:
    backend = AtherisBackend(
        spec.harness_path, corpus_dir, python=python,
        # SAME self-limit as plain — the hybrid's edge is solver time, not Atheris time.
        # If a final assist blocks the parent past the deadline, the subprocess still
        # self-terminates here, so background Atheris cannot fuzz longer than plain's did.
        extra_args=[f"-max_total_time={_atheris_budget(duration_s)}"],
    )
    fuzzer = HybridFuzzer(spec, backend=backend, plateau_s=plateau_s, assist_cooldown_s=plateau_s)
    t0 = time.monotonic()
    stats = fuzzer.run(duration_s=duration_s, poll_s=0.5)
    return Arm(
        name="hybrid", coverage=backend.coverage(),
        corpus_size=len(glob.glob(os.path.join(corpus_dir, "*"))),
        depth=_corpus_depth(corpus_dir, depth_probe),
        assists=stats.assists, seeds_injected=stats.seeds_written,
        duration_s=time.monotonic() - t0,
    )


def benchmark(
    spec: HarnessSpec,
    *,
    duration_s: float = 20.0,
    seeds: list[bytes] | None = None,
    depth_probe: Callable[[bytes], int] | None = None,
    python: str | None = None,
    plateau_s: float = 2.5,
    parallel: bool = False,
) -> BenchResult:
    """Run both arms on fresh copies of the same seed corpus and compare.

    ``seeds`` is the shared starting corpus (default: one empty seed).  ``depth_probe``
    maps a corpus seed's bytes to an integer 'how deep did it get' for a human-readable
    signal; supply one that mirrors the target's guard chain.  Each arm gets its own
    temp corpus so they do not share finds.

    ``parallel`` starts the two arms at the same moment instead of one after the other
    (a test run halves its wall time): the arms share nothing -- own corpus, own Atheris
    subprocess -- and each still gets the full ``duration_s`` and the same ``-max_total_time``,
    so the comparison is the same one on a host with two free cores.
    """
    seeds = seeds if seeds is not None else [b"\x00"]
    plain_dir = tempfile.mkdtemp(prefix="bench-plain-")
    hybrid_dir = tempfile.mkdtemp(prefix="bench-hybrid-")
    try:
        _seed_corpus(plain_dir, seeds)
        _seed_corpus(hybrid_dir, seeds)

        hspec = HarnessSpec(
            harness_path=spec.harness_path, target_path=spec.target_path,
            target_func=spec.target_func, corpus_dir=hybrid_dir,
            harness_func=spec.harness_func, field_to_params=spec.field_to_params,
        )
        if parallel:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=2) as pool:
                plain_run = pool.submit(_run_plain, spec, plain_dir, duration_s, depth_probe, python)
                hybrid_run = pool.submit(_run_hybrid, hspec, hybrid_dir, duration_s, depth_probe, python, plateau_s)
                plain, hybrid = plain_run.result(), hybrid_run.result()
        else:
            plain = _run_plain(spec, plain_dir, duration_s, depth_probe, python)
            hybrid = _run_hybrid(hspec, hybrid_dir, duration_s, depth_probe, python, plateau_s)

        return BenchResult(target=os.path.basename(spec.target_path), plain=plain, hybrid=hybrid)
    finally:
        shutil.rmtree(plain_dir, ignore_errors=True)
        shutil.rmtree(hybrid_dir, ignore_errors=True)


def format_report(r: BenchResult) -> str:
    lines = [
        f"hybrid fuzzer benchmark — target: {r.target}",
        "-" * 60,
        f"{'arm':<16}{'cov':>7}{'corpus':>8}{'depth':>7}{'assists':>9}{'seeds':>7}{'secs':>7}",
    ]
    for a in (r.plain, r.hybrid):
        lines.append(f"{a.name:<16}{a.coverage:>7}{a.corpus_size:>8}{a.depth:>7}"
                     f"{a.assists:>9}{a.seeds_injected:>7}{a.duration_s:>7.1f}")
    lines.append("-" * 60)
    lines.append(f"coverage gain (hybrid - plain): {r.coverage_gain:+d}")
    lines.append(f"depth gain    (hybrid - plain): {r.depth_gain:+d}")
    # The honesty footer: whether Atheris genuinely ran in both arms, and whether they got
    # the same wall-clock (both cap Atheris identically, so a large skew is the red flag).
    lines.append(f"atheris genuinely ran (both):   {r.atheris_ran} "
                 f"(cov plain={r.plain.coverage}, hybrid={r.hybrid.coverage})")
    lines.append(f"wall-clock skew between arms:   {r.duration_skew:.1f}s "
                 f"(same Atheris budget both arms)")
    return "\n".join(lines)
